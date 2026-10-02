"""``BacktestVenue``: a quantlab run replayed on a nautilus ``BacktestEngine``.

One synthetic venue ``CRSP`` with a NETTING, MARGIN account at leverage 1 in
USD; one ``<PERMNO>.CRSP`` equity per tradable security; the feed of opening
and closing prints; the fee model; the corporate-action module; the open
submitter and the decision clock.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, RiskEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.objects import Money

from quantlab_trader.base.config import VenueConfig
from quantlab_trader.base.venue import Loop, ReplayRequest, Venue, VenueReport
from quantlab_trader.quantlab_run import QuantlabRun
from quantlab_trader.venue.backtest.clock import BacktestDecisionClock
from quantlab_trader.venue.backtest.corporate_actions import (
    CorporateActionModule,
    corporate_action_days,
    holder_split_factors,
)
from quantlab_trader.venue.backtest.feed import (
    CLOSE_TIME,
    build_feed,
    opening_prints,
    session_ns,
)
from quantlab_trader.venue.backtest.fees import FractionFeeModel, IbkrFixedFeeModel
from quantlab_trader.venue.backtest.fills import FractionSlippageFillModel
from quantlab_trader.venue.backtest.resolver import SYNTHETIC_VENUE, BacktestResolver
from quantlab_trader.venue.backtest.source import BacktestDecisionSource
from quantlab_trader.venue.backtest.submitter import BacktestOpenSubmitter

if TYPE_CHECKING:
    from quantlab_trader.strategy import PortfolioStrategy

#: The risk engine's submit-rate limit. The submitter sends a whole rebalance
#: at the open + 1 ns (ADR 0003), in one simulated instant; nautilus's
#: default, 100 per second, denied every order past the 100th of a bar. A
#: broker's message pacing belongs to a live venue's submitter, not here.
MAX_ORDER_SUBMIT_RATE = "1000000/00:00:01"

#: How long the engine runs past the last closing print, so the last
#: decision (at close + 1 ns) fires.
_AFTER_LAST_CLOSE = pd.Timedelta(hours=1)

#: Each ``ExecutionConfig.fee_model`` name and how it is built from the run.
_FEE_MODELS: dict[str, Callable[[QuantlabRun], FractionFeeModel | IbkrFixedFeeModel]] = {
    "fraction": lambda run: FractionFeeModel(run.fees),
    "ibkr_fixed": lambda run: IbkrFixedFeeModel(),
}

#: The fee model each loop takes when ``ExecutionConfig.fee_model`` is None
#: (#13, spec #18 stories 22-23): closed loop costs what the IBKR account
#: will, open loop what the quantlab run charged.
_DEFAULT_FEE_MODEL = {Loop.CLOSED: "ibkr_fixed", Loop.OPEN: "fraction"}


@dataclass(frozen=True)
class ExecutionConfig:
    """How the backtest venue executes: fees, slippage and starting cash.

    Parameters
    ----------
    fee_model : {"fraction", "ibkr_fixed"} or None
        ``"fraction"`` charges the run's ``fees`` fraction of the notional,
        ``"ibkr_fixed"`` IBKR Pro Fixed plus the SEC fee on sales; ``None``
        takes the loop's default (closed loop: ``"ibkr_fixed"``, open loop:
        ``"fraction"``).
    slippage : float or None
        Fractional slippage in ``[0, 1)``, against the order; ``None`` takes
        the run's.
    init_cash : float or None
        Starting cash, positive; ``None`` takes the run's.

    Raises
    ------
    ValueError
        For an unknown fee model, a slippage outside ``[0, 1)`` or a
        starting cash that is not positive.
    """

    fee_model: Literal["fraction", "ibkr_fixed"] | None = None
    slippage: float | None = None
    init_cash: float | None = None

    def __post_init__(self):
        if self.fee_model is not None and self.fee_model not in _FEE_MODELS:
            raise ValueError(
                f"ExecutionConfig.fee_model must be one of {sorted(_FEE_MODELS)} or None, "
                f"got {self.fee_model!r}"
            )
        if self.slippage is not None and not 0.0 <= self.slippage < 1.0:
            raise ValueError(f"ExecutionConfig.slippage must lie in [0, 1), got {self.slippage}")
        if self.init_cash is not None and not self.init_cash > 0:
            raise ValueError(f"ExecutionConfig.init_cash must be positive, got {self.init_cash}")

    def get_config(self) -> dict[str, Any]:
        """Return the fields as JSON values.

        Examples
        --------
        >>> ExecutionConfig(init_cash=1e6).get_config()
        {'fee_model': None, 'slippage': None, 'init_cash': 1000000.0}
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Rebuild the config ``get_config()`` returned.

        Examples
        --------
        >>> ExecutionConfig.from_config({"init_cash": 1e6}) == ExecutionConfig(init_cash=1e6)
        True
        """
        return cls(**config)


@dataclass(frozen=True)
class BacktestVenueConfig(VenueConfig):
    """The backtest venue and its execution settings.

    Examples
    --------
    >>> config = BacktestVenueConfig(ExecutionConfig(init_cash=1e6))
    >>> VenueConfig.from_config(config.get_config()) == config
    True
    """

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @classmethod
    def _from_fields(cls, fields: dict[str, Any]) -> Self:
        """Rebuild the nested ``ExecutionConfig``."""
        execution = fields.get("execution")
        if isinstance(execution, dict):
            fields = {**fields, "execution": ExecutionConfig.from_config(execution)}
        return cls(**fields)

    def build(self, run: QuantlabRun, request: ReplayRequest) -> BacktestVenue:
        """Return the backtest venue executing ``request``, its execution settings resolved.

        An unset fee model is the loop's default (closed loop: ``"ibkr_fixed"``,
        open loop: ``"fraction"``); unset slippage and starting cash are the
        run's.

        Raises
        ------
        ValueError
            For a run whose own slippage lies outside ``[0, 1)``.
        """
        execution = self.execution
        fee_model = _FEE_MODELS[execution.fee_model or _DEFAULT_FEE_MODEL[request.loop]](run)
        slippage = run.slippage if execution.slippage is None else execution.slippage
        return BacktestVenue(
            run,
            request,
            fee_model=fee_model,
            fill_model=FractionSlippageFillModel(slippage),
            init_cash=run.init_cash if execution.init_cash is None else execution.init_cash,
        )


class BacktestVenue(Venue):
    """The backtest venue: resolver, submitter, source and clock around a ``BacktestEngine``.

    Parameters
    ----------
    run : QuantlabRun
        The run replayed.
    request : ReplayRequest
        The window, the securities to create instruments for and, closed
        loop, the prediction panel and history start for the decision source.
    fee_model : FractionFeeModel or IbkrFixedFeeModel
        The fee model; it records the fills it charges its minimum.
    fill_model : FractionSlippageFillModel
        The slippage.
    init_cash : float
        The starting cash.

    Attributes
    ----------
    fee_model, fill_model, init_cash
        As given.
    """

    def __init__(
        self,
        run: QuantlabRun,
        request: ReplayRequest,
        *,
        fee_model: FractionFeeModel | IbkrFixedFeeModel,
        fill_model: FractionSlippageFillModel,
        init_cash: float,
    ):
        self.fee_model = fee_model
        self.fill_model = fill_model
        self.init_cash = init_cash
        self.source = BacktestDecisionSource(run, request)
        calendar = self.source.calendar()
        self.resolver = BacktestResolver(request.permnos, calendar[0])
        self.corporate_actions = corporate_action_days(self.source.prices)
        self.submitter = BacktestOpenSubmitter(
            calendar,
            opening_prints(self.source.prices),
            holder_split_factors(self.corporate_actions),
        )
        self.clock = BacktestDecisionClock(calendar)

    def run(self, strategy: PortfolioStrategy) -> VenueReport:
        """Build the engine, run the window through ``strategy`` and dispose of it.

        Returns
        -------
        VenueReport
            The fills the fee model charged its minimum.
        """
        engine = BacktestEngine(
            BacktestEngineConfig(
                logging=LoggingConfig(log_level="ERROR"),
                risk_engine=RiskEngineConfig(max_order_submit_rate=MAX_ORDER_SUBMIT_RATE),
            )
        )
        try:
            engine.add_venue(
                SYNTHETIC_VENUE,
                OmsType.NETTING,
                AccountType.MARGIN,
                [Money(self.init_cash, USD)],
                base_currency=USD,
                default_leverage=Decimal(1),
                fee_model=self.fee_model,
                fill_model=self.fill_model,
                modules=[
                    CorporateActionModule(
                        self.corporate_actions,
                        self.source.delisting_settlements(),
                        self.resolver,
                        on_event=strategy.on_corporate_action,
                    )
                ],
            )
            for instrument in self.resolver.instruments():
                engine.add_instrument(instrument)
            engine.add_data(build_feed(self.source.prices, self.resolver))
            engine.add_strategy(strategy)
            calendar = self.source.calendar()
            engine.run(
                start=pd.Timestamp(int(session_ns(calendar[0], pd.Timedelta(0))), tz="UTC"),
                end=pd.Timestamp(
                    int(session_ns(calendar[-1], CLOSE_TIME) + _AFTER_LAST_CLOSE.value),
                    tz="UTC",
                ),
            )
        finally:
            engine.dispose()
        return VenueReport(minimum_fee_orders=frozenset(self.fee_model.minimum_fee_orders))
