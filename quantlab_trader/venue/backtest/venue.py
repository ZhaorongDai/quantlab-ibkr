"""``BacktestVenue``: a quantlab run replayed on a nautilus ``BacktestEngine``.

One synthetic venue ``CRSP`` with a NETTING, MARGIN account at leverage 1 in
USD; one ``<PERMNO>.CRSP`` equity per tradable security; the feed of opening
and closing prints; the fee model; the corporate-action module; the open
submitter and the decision clock.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.objects import Money

from quantlab_trader.base.config import VenueConfig
from quantlab_trader.base.venue import Venue
from quantlab_trader.quantlab_run import QuantlabRun
from quantlab_trader.venue.backtest.clock import BacktestDecisionClock
from quantlab_trader.venue.backtest.corporate_actions import CorporateActionModule
from quantlab_trader.venue.backtest.feed import CLOSE_TIME, build_feed, session_ns
from quantlab_trader.venue.backtest.fees import FractionFeeModel
from quantlab_trader.venue.backtest.resolver import SYNTHETIC_VENUE, BacktestResolver
from quantlab_trader.venue.backtest.source import BacktestDecisionSource
from quantlab_trader.venue.backtest.submitter import BacktestOpenSubmitter

if TYPE_CHECKING:
    from quantlab_trader.strategy import PortfolioStrategy

#: How long the engine runs past the last closing print, so the last
#: decision (at close + 1 ns) fires.
_AFTER_LAST_CLOSE = pd.Timedelta(hours=1)


@dataclass(frozen=True)
class ExecutionConfig:
    """How the backtest venue executes: fees, slippage and starting cash.

    Parameters
    ----------
    fee_model : {"fraction", "ibkr_fixed"} or None
        ``"fraction"`` charges the run's ``fees`` fraction of the notional;
        ``None`` takes the loop's default (open loop: ``"fraction"``).
    slippage : float or None
        Fractional slippage; ``None`` takes the run's.
    init_cash : float or None
        Starting cash; ``None`` takes the run's.
    """

    fee_model: Literal["fraction", "ibkr_fixed"] | None = None
    slippage: float | None = None
    init_cash: float | None = None

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

    def build(
        self,
        run: QuantlabRun,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        permnos: tuple,
        loop: str,
    ) -> BacktestVenue:
        """Return the backtest venue replaying ``run`` from ``start`` to ``end``."""
        return BacktestVenue(run, self.execution, start=start, end=end, permnos=permnos, loop=loop)


class BacktestVenue(Venue):
    """The backtest venue: resolver, submitter, source and clock around a ``BacktestEngine``.

    Parameters
    ----------
    run : QuantlabRun
        The run replayed.
    execution : ExecutionConfig
        Fees, slippage and starting cash.
    start, end : pandas.Timestamp
        The window, inclusive.
    permnos : tuple
        The securities to create instruments for.
    loop : {"open", "closed"}
        The replay's loop, which picks the default fee model.

    Raises
    ------
    ValueError
        For an execution setting this version cannot simulate yet: the
        ``"ibkr_fixed"`` fee model, a closed-loop default fee, or nonzero
        slippage.
    """

    def __init__(
        self,
        run: QuantlabRun,
        execution: ExecutionConfig,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        permnos: tuple,
        loop: str,
    ):
        fee_model = execution.fee_model or ("fraction" if loop == "open" else "ibkr_fixed")
        if fee_model != "fraction":
            raise ValueError(
                f"BacktestVenue: fee model {fee_model!r} is not available yet; "
                f"set execution.fee_model='fraction'"
            )
        slippage = run.slippage if execution.slippage is None else execution.slippage
        if slippage != 0.0:
            raise ValueError(
                f"BacktestVenue: slippage {slippage} cannot be simulated yet; set "
                f"execution.slippage=0 to replay without it"
            )
        self.init_cash = run.init_cash if execution.init_cash is None else execution.init_cash
        self.fee_rate = run.fees
        self.source = BacktestDecisionSource(run, start, end, permnos)
        calendar = self.source.calendar()
        self.resolver = BacktestResolver(permnos, calendar[0])
        opening_prints = (
            self.source.prices["open"].transpose("timestamp", "symbol").notnull().to_pandas()
        )
        self.submitter = BacktestOpenSubmitter(calendar, opening_prints)
        self.clock = BacktestDecisionClock(calendar)

    def run(self, strategy: PortfolioStrategy) -> None:
        """Build the engine, run the window through ``strategy`` and dispose of it."""
        engine = BacktestEngine(
            BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR"))
        )
        try:
            engine.add_venue(
                SYNTHETIC_VENUE,
                OmsType.NETTING,
                AccountType.MARGIN,
                [Money(self.init_cash, USD)],
                base_currency=USD,
                default_leverage=Decimal(1),
                fee_model=FractionFeeModel(self.fee_rate),
                modules=[
                    CorporateActionModule(
                        self.source.delisting_settlements(), self.resolver
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
