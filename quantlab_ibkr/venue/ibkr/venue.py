"""``IbkrVenue``: one live day of a quantlab run on IBKR, through a nautilus ``TradingNode``.

Live trading is a daily batch (#47, ADR 0010): one ``run`` connects a
``TradingNode`` with the IB adapter's data and execution clients to the
Gateway, reconciles the account (positions and cash are IBKR's), lets the
strategy decide the last closed bar t once if t is a rebalance bar of the
run's cadence (``LiveDecisionClock``), submits the orders as market-on-open
(``IbkrOpenSubmitter``), waits until IBKR has accepted them, and stops. The
parts:

- ``LiveDecisionSource``: t's row of the run's live prediction store, its raw
  close and delisting from the run's price dataset;
- ``LiveDecisionClock``: fires the cycle of t once; it decides on a rebalance
  bar with a prediction row (``targets``), and only marks the account on any
  other day;
- ``IbkrResolver``: each symbol's IBKR contract as of t, looked up before the
  node starts (``IbapiContractClient``, or an injected ``ContractClient``),
  and the conId cache kept in the live directory between days;
- ``IbkrOpenSubmitter``: ``MKT`` / ``OPG`` before the order deadline.

Safety: the account (``account_id``, else the ``TWS_ACCOUNT`` environment
variable) must be an IBKR paper account (``DU...``) unless ``allow_live`` is
set; a day that decides refuses to start after the order deadline; a dry run
decides and reports the orders without submitting any. Credentials never
pass through here: the Gateway holds the login.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from nautilus_trader.adapters.interactive_brokers.common import IB
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersDataClientConfig,
    InteractiveBrokersExecClientConfig,
)
from nautilus_trader.adapters.interactive_brokers.factories import (
    InteractiveBrokersLiveDataClientFactory,
    InteractiveBrokersLiveExecClientFactory,
)
from nautilus_trader.config import (
    LiveExecEngineConfig,
    LiveRiskEngineConfig,
    LoggingConfig,
    TradingNodeConfig,
)
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import TraderId

from quantlab.dataset.base import TickerLookup
from quantlab.runs.live_predictions import LivePredictionStore
from quantlab_ibkr.account import derived_cash
from quantlab_ibkr.base.config import VenueConfig
from quantlab.portfolio.base import Decision
from quantlab_ibkr.base.venue import BarInputs, Loop, ReplayRequest, Venue, VenueReport
from quantlab_ibkr.decision import ConstructorTargets, TargetSource
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.venue.ibkr.clock import LiveDecision, LiveDecisionClock
from quantlab_ibkr.venue.ibkr.contracts import IbapiContractClient
from quantlab_ibkr.venue.ibkr.resolver import (
    UNPARSEABLE_CONTRACT,
    ContractClient,
    IbkrResolver,
    ResolutionGap,
)
from quantlab_ibkr.venue.ibkr.reports import IbkrOrderState
from quantlab_ibkr.venue.ibkr.source import LiveDecisionSource
from quantlab_ibkr.venue.ibkr.submitter import (
    MAX_ORDERS_PER_SECOND,
    ORDER_DEADLINE,
    IbkrOpenSubmitter,
    order_deadline,
    parse_deadline,
)

if TYPE_CHECKING:
    from quantlab_ibkr.strategy import PortfolioStrategy

#: The environment variable naming the IBKR account when the config does not.
ACCOUNT_ENV = "TWS_ACCOUNT"
#: The prefix of an IBKR paper account.
PAPER_PREFIX = "DU"
#: The conId cache's file in the live directory: conId -> symbol, kept across days.
CON_ID_CACHE = "con_ids.json"
#: The trader id of the live node.
TRADER_ID = "QUANTLAB-IBKR"
#: The risk engine's submit-rate limit: the submitter paces the orders itself
#: (``max_orders_per_second``), so the risk engine never denies one for rate.
MAX_ORDER_SUBMIT_RATE = "1000/00:00:01"
#: How often the run checks whether the day is done, in seconds.
_POLL_SECS = 0.25


class OrderDeadlinePassed(ValueError):
    """A day that decides, started after its order deadline.

    Examples
    --------
    >>> issubclass(OrderDeadlinePassed, ValueError)
    True
    """


@dataclass(frozen=True)
class IbkrVenueConfig(VenueConfig):
    """The IBKR venue: the Gateway, the account, the live prediction store and the guards.

    Parameters
    ----------
    prediction_store : str or None
        The run's live prediction store (quantlab #233); required to build.
    live_dir : str or None
        The live run directory; the conId cache is kept there
        (``CON_ID_CACHE``). ``None`` keeps no cache.
    host : str, default "127.0.0.1"
        The Gateway's host.
    port : int, default 4002
        Its API port (IB Gateway, paper).
    client_id : int, default 1
        The trading node's API client id (data and execution share it).
    account_id : str or None
        The IBKR account; ``None`` reads the ``TWS_ACCOUNT`` environment
        variable.
    allow_live : bool, default False
        Allow an account that is not a paper account (``DU...``).
    order_deadline : str, default "09:20"
        ``HH:MM`` America/New_York on the weekday after t; a deciding day
        refuses to start after it, and no order is submitted after it.
    dry_run : bool, default False
        Decide and report the orders, submit none.
    force_decide : bool, default False
        Decide t even when it is not a rebalance bar of the run's cadence, to
        exercise the decision, the contract lookup and the orders on any day;
        only with ``dry_run``.
    contract_client_id : int or None
        The API client id of the contract lookup made before the node
        starts; ``None`` is ``client_id + 1``.
    reports_client_id : int or None
        The API client id the live steps read IBKR's order reports on
        (``IbapiOrderReports``); ``None`` is ``client_id + 2``.
    max_orders_per_second : int, default 40
        The submitter's pace.
    timeout_secs : float, default 300
        How long a run waits for the node to connect, reconcile, decide and
        have the orders accepted before it stops anyway (``timed_out``).

    Raises
    ------
    ValueError
        For a deadline that is not ``HH:MM``, a pace or timeout that is not
        positive, or ``force_decide`` without ``dry_run``.

    Examples
    --------
    >>> config = IbkrVenueConfig(prediction_store="live/live_predictions.zarr", client_id=7)
    >>> VenueConfig.from_config(config.get_config()) == config
    True
    >>> config.get_config()["name"]
    'quantlab_ibkr.venue.ibkr.venue.IbkrVenueConfig'
    """

    prediction_store: str | None = None
    live_dir: str | None = None
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1
    account_id: str | None = None
    allow_live: bool = False
    order_deadline: str = ORDER_DEADLINE
    dry_run: bool = False
    force_decide: bool = False
    contract_client_id: int | None = None
    reports_client_id: int | None = None
    max_orders_per_second: int = MAX_ORDERS_PER_SECOND
    timeout_secs: float = 300.0

    def __post_init__(self):
        parse_deadline(self.order_deadline)
        if self.force_decide and not self.dry_run:
            raise ValueError(
                "IbkrVenueConfig.force_decide needs dry_run: a forced decision is never submitted"
            )
        if self.max_orders_per_second < 1:
            raise ValueError(
                f"IbkrVenueConfig.max_orders_per_second must be at least 1, "
                f"got {self.max_orders_per_second}"
            )
        if not self.timeout_secs > 0:
            raise ValueError(
                f"IbkrVenueConfig.timeout_secs must be positive, got {self.timeout_secs}"
            )

    def account(self) -> str:
        """Return the IBKR account, refusing a live one unless ``allow_live``.

        Raises
        ------
        ValueError
            If no account is configured or set in ``TWS_ACCOUNT``, or the
            account is not a paper account (``DU...``) and ``allow_live`` is
            False.

        Examples
        --------
        >>> IbkrVenueConfig(account_id="DU1234567").account()
        'DU1234567'
        >>> IbkrVenueConfig(account_id="U1234567").account()
        Traceback (most recent call last):
        ...
        ValueError: account U1234567 is not an IBKR paper account (DU...); set allow_live=True to trade it
        """
        account = self.account_id or os.environ.get(ACCOUNT_ENV)
        if not account:
            raise ValueError(
                f"no IBKR account: set IbkrVenueConfig.account_id or the {ACCOUNT_ENV} "
                f"environment variable"
            )
        if not account.startswith(PAPER_PREFIX) and not self.allow_live:
            raise ValueError(
                f"account {account} is not an IBKR paper account ({PAPER_PREFIX}...); "
                f"set allow_live=True to trade it"
            )
        return account

    def build(
        self,
        run: QuantlabRun,
        request: ReplayRequest,
        *,
        contract_client: ContractClient | None = None,
        ticker_lookup: TickerLookup | None = None,
        now: pd.Timestamp | None = None,
        working_orders: Iterable[IbkrOrderState] = (),
    ) -> IbkrVenue:
        """Return the venue of today's live day of ``run``; nothing is submitted yet.

        Parameters
        ----------
        run : QuantlabRun
            The quantlab run traded.
        request : ReplayRequest
            Closed loop; ``permnos`` are the symbols known to be held (priced
            and resolved besides the predicted ones). Its window and
            prediction panel are not read: t is the price dataset's last bar.
        contract_client : ContractClient, optional
            The contract lookup; ``None`` connects an ``IbapiContractClient``
            to the Gateway for the lookups.
        ticker_lookup : TickerLookup, optional
            Names the symbols as of t; ``None`` is the price dataset's.
        now : pandas.Timestamp, optional
            The current time, for the deadline; ``None`` is the wall clock.
        working_orders : Iterable of IbkrOrderState, optional
            The orders already working at IBKR for t (an earlier run of the
            day): the submitter adopts them instead of sending them again.

        Raises
        ------
        ValueError
            For an open-loop request, no prediction store, no ticker lookup,
            or an account ``account`` refuses.
        OrderDeadlinePassed
            If t is decided, not in a dry run, and ``now`` is past its order
            deadline.

        Examples
        --------
        The live CLI builds the venue and runs the strategy on it::

            venue = IbkrVenueConfig(prediction_store=store, live_dir=live_dir).build(
                quantlab_run, ReplayRequest(t, t, held, Loop.CLOSED)
            )
            report = venue.run(strategy)
        """
        return IbkrVenue(
            run,
            request,
            self,
            contract_client=contract_client,
            ticker_lookup=ticker_lookup,
            now=now,
            working_orders=working_orders,
        )


class LiveTargets(TargetSource):
    """The live day's targets: the rule's decision when the day decides, nothing otherwise.

    On a day the clock decides (``LiveDecision.decides``) the rule decides t
    through ``ConstructorTargets``; on a hold (a bar off the cadence, no
    prediction row) the cycle only marks the account.

    Parameters
    ----------
    constructor : ConstructorTargets
        The closed loop's targets on the run's decision inputs.
    decision : LiveDecision
        Whether the day decides.

    Examples
    --------
    ::

        cycle = DecisionCycle(venue.targets())
    """

    def __init__(self, constructor: ConstructorTargets, decision: LiveDecision):
        self.constructor = constructor
        self.decision = decision

    def targets(self, inputs: BarInputs, current_weights: pd.Series) -> Decision | None:
        """Return the rule's decision of t on a deciding day, ``None`` on a hold.

        Examples
        --------
        ::

            venue.targets().targets(venue.source.inputs(t), current_weights)
        """
        if not self.decision.decides:
            return None
        return self.constructor.targets(inputs, current_weights)

    def read_sources(self) -> list[tuple[object, str]]:
        """The rule's recorded reads (``ConstructorTargets.read_sources``)."""
        return self.constructor.read_sources()


class IbkrVenue(Venue):
    """One live day: the four parts around a ``TradingNode`` connected to the Gateway.

    Attributes
    ----------
    config : IbkrVenueConfig
        As given.
    account_id : str
        The account traded (the paper guard passed).
    decision : LiveDecision
        Whether t is decided, and the hold reason when not.
    decision_inputs : quantlab.portfolio.decision_inputs.DecisionInputs
        The run's decision inputs on an open-ended schedule
        (``QuantlabRun.decision_inputs()``), whose ``rebalances`` is the
        clock's cadence; the strategy's ``ConstructorTargets`` decides with
        them.
    position_gaps : list of ResolutionGap
        The account positions no symbol maps to, read when the strategy
        starts; they are left out of the holdings.
    init_cash : float
        The account's cash when the strategy starts (NaN before ``run``).
    timed_out : bool
        ``run`` stopped the node at ``timeout_secs`` before the day was done.

    Examples
    --------
    Built by ``IbkrVenueConfig.build``, then run around the strategy::

        venue = IbkrVenueConfig(prediction_store=store).build(quantlab_run, request)
        report = venue.run(strategy)
        venue.decision.hold_reason, venue.submitter.submitted, venue.resolver.gaps
    """

    def __init__(
        self,
        run: QuantlabRun,
        request: ReplayRequest,
        config: IbkrVenueConfig,
        *,
        contract_client: ContractClient | None = None,
        ticker_lookup: TickerLookup | None = None,
        now: pd.Timestamp | None = None,
        working_orders: Iterable[IbkrOrderState] = (),
    ):
        if request.loop is not Loop.CLOSED:
            raise ValueError("the IBKR venue trades closed loop only")
        if config.prediction_store is None:
            raise ValueError("IbkrVenueConfig.prediction_store is required to trade live")
        self.config = config
        self.account_id = config.account()
        self.init_cash = float("nan")
        self.position_gaps: list[ResolutionGap] = []
        self.timed_out = False
        seed = self._read_con_ids()
        held = tuple(dict.fromkeys([*request.permnos, *seed.values()]))
        self.source = LiveDecisionSource(
            run, LivePredictionStore(config.prediction_store), symbols=held
        )
        self.decision_inputs = run.decision_inputs()
        self.clock = LiveDecisionClock(
            self.source.t,
            self.decision_inputs.rebalances,
            self.source.hold_reason,
            force=config.force_decide,
            on_schedule=self._on_start,
        )
        self.decision = self.clock.decision
        if self.decision.decides and not config.dry_run:
            deadline = order_deadline(self.source.t, config.order_deadline)
            now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
            if now >= deadline:
                raise OrderDeadlinePassed(
                    f"refusing to decide {self.source.t.date()}: its order deadline "
                    f"{deadline} has passed"
                )
        lookup = run.price_dataset.ticker_lookup() if ticker_lookup is None else ticker_lookup
        if lookup is None:
            raise ValueError(
                f"the price dataset {type(run.price_dataset).__name__} names no tickers; "
                f"the IBKR venue resolves contracts by ticker (ADR 0004)"
            )
        symbols = self.source.symbols if self.decision.decides else held
        if contract_client is None:
            with IbapiContractClient(
                config.host,
                config.port,
                config.contract_client_id or config.client_id + 1,
            ) as client:
                self.resolver = self._resolve(symbols, lookup, client, seed)
        else:
            self.resolver = self._resolve(symbols, lookup, contract_client, seed)
        symbol_of = self.resolver.con_id_cache
        self.submitter = IbkrOpenSubmitter(
            self.resolver,
            order_deadline=config.order_deadline,
            dry_run=config.dry_run,
            max_orders_per_second=config.max_orders_per_second,
            working={
                (symbol_of[order.con_id], order.side): order.order_ref
                for order in working_orders
                if order.con_id in symbol_of
            },
        )

    def targets(self) -> LiveTargets:
        """Return the day's targets: the rule on the run's decision inputs, when t is decided.

        Examples
        --------
        ::

            strategy = PortfolioStrategy(
                venue=venue, cycle=DecisionCycle(venue.targets()), recorder=recorder
            )
        """
        return LiveTargets(ConstructorTargets(self.decision_inputs), self.decision)

    def _resolve(
        self,
        symbols: Iterable[Hashable],
        lookup: TickerLookup,
        client: ContractClient,
        seed: Mapping[int, Hashable],
    ) -> IbkrResolver:
        """Resolve ``symbols`` as of t and keep every unmappable one out of the targets."""
        t = self.source.t
        resolver = IbkrResolver(symbols, t, lookup, client, con_id_cache=seed)
        resolver.instruments()  # parses each contract; an unparseable one is a gap
        unparseable = {
            gap.symbol for gap in resolver.gaps
            if gap.reason == UNPARSEABLE_CONTRACT and gap.date == t
        }
        self.source.exclude(resolver.unresolved(t) | unparseable)
        return resolver

    def _read_con_ids(self) -> dict[int, Hashable]:
        path = self._con_id_path()
        if path is None or not path.is_file():
            return {}
        return {int(k): v for k, v in json.loads(path.read_text()).items()}

    def _write_con_ids(self) -> None:
        path = self._con_id_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = {str(k): v for k, v in sorted(self.resolver.con_id_cache.items())}
        path.write_text(json.dumps(cache, indent=1))

    def _con_id_path(self) -> Path | None:
        return None if self.config.live_dir is None else Path(self.config.live_dir) / CON_ID_CACHE

    def _on_start(self, strategy: PortfolioStrategy) -> None:
        """Report the reconciled positions no symbol maps to, and read the starting cash."""
        held = [
            position.instrument_id
            for position in strategy.cache.positions_open()
            if int(position.signed_decimal_qty())
        ]
        self.position_gaps = self.resolver.report_positions(held, self.source.t)
        self.init_cash = derived_cash(strategy.cache)

    def node_config(self) -> TradingNodeConfig:
        """Return the trading node's config: IBKR data and execution clients, reconciliation on.

        Both clients load the resolved contracts through the resolver's
        instrument provider config and share one API connection
        (``client_id``).

        Examples
        --------
        ::

            config = venue.node_config()
            config.exec_clients[IB].account_id  # 'DU1234567'
        """
        provider = self.resolver.provider_config()
        connection = {
            "ibg_host": self.config.host,
            "ibg_port": self.config.port,
            "ibg_client_id": self.config.client_id,
            "instrument_provider": provider,
        }
        return TradingNodeConfig(
            trader_id=TraderId(TRADER_ID),
            logging=LoggingConfig(log_level="INFO"),
            exec_engine=LiveExecEngineConfig(reconciliation=True),
            risk_engine=LiveRiskEngineConfig(max_order_submit_rate=MAX_ORDER_SUBMIT_RATE),
            data_clients={IB: InteractiveBrokersDataClientConfig(**connection)},
            exec_clients={
                IB: InteractiveBrokersExecClientConfig(account_id=self.account_id, **connection)
            },
        )

    def build_node(self, loop: asyncio.AbstractEventLoop | None = None) -> TradingNode:
        """Return the trading node with IBKR's client factories added; nothing connects yet.

        Examples
        --------
        ::

            node = venue.build_node(asyncio.new_event_loop())
            node.trader.add_strategy(strategy)
            node.build()
        """
        node = TradingNode(config=self.node_config(), loop=loop)
        node.add_data_client_factory(IB, InteractiveBrokersLiveDataClientFactory)
        node.add_exec_client_factory(IB, InteractiveBrokersLiveExecClientFactory)
        return node

    def run(self, strategy: PortfolioStrategy) -> VenueReport:
        """Connect, reconcile, decide t once (or hold), submit, wait for acceptance, stop.

        Returns
        -------
        VenueReport
            IBKR charges and reports its own commissions; no slippage model.

        Raises
        ------
        Exception
            What the decision raised, once the node has stopped.
        RuntimeError
            If the node stops before the day is done.

        Examples
        --------
        ::

            report = venue.run(strategy)
            venue.submitter.submitted, venue.submitter.unfilled, venue.timed_out
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        node = self.build_node(loop)
        try:
            node.trader.add_strategy(strategy)
            node.build()
            loop.run_until_complete(self._session(node, strategy))
        finally:
            node.dispose()
            if not loop.is_closed():
                loop.close()
        self._write_con_ids()
        if self.clock.error is not None:
            raise self.clock.error
        return VenueReport(fees="IBKR's commissions, as IBKR reports them", slippage=None)

    def done(self, strategy: PortfolioStrategy) -> bool:
        """Whether the day is done: decided (or held), every order released and none in flight.

        Examples
        --------
        ::

            while not venue.done(strategy):
                await asyncio.sleep(0.25)
        """
        return (
            self.clock.done
            and self.submitter.released
            and not strategy.cache.orders_inflight(strategy_id=strategy.id)
        )

    async def _session(self, node: TradingNode, strategy: PortfolioStrategy) -> None:
        """Run the node until the day is done or ``timeout_secs`` pass, then stop it."""
        loop = asyncio.get_running_loop()
        running = asyncio.ensure_future(node.run_async())
        give_up = loop.time() + self.config.timeout_secs
        try:
            while not self.done(strategy):
                if running.done():
                    running.result()
                    raise RuntimeError("the trading node stopped before the day was done")
                if loop.time() >= give_up:
                    self.timed_out = True
                    break
                await asyncio.sleep(_POLL_SECS)
        finally:
            await node.stop_async()
            running.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await running
