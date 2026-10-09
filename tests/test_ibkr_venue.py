"""The IBKR venue (#50): market-on-open submitter, paper guard, dry run and node assembly.

No Gateway: the submitter runs inside the real ``PortfolioStrategy``,
registered on nautilus's message bus, cache and a test clock, with the risk
engine's endpoint captured (the fake execution path); the venue is built on a
fixture quantlab run and live prediction store with an injected contract
client and ticker lookup, and its ``TradingNode`` is assembled, never
started. Locked:

- order shape: a nautilus ``MarketOrder``, ``TimeInForce.AT_THE_OPEN`` (the
  IB adapter's ``OPG``), side, whole quantity and the resolver's instrument,
  sells first and paced;
- no order leaves after the deadline (09:20 New York on the weekday after
  t), and a deciding day refuses to build after it;
- a dry run decides and reports every order without submitting one;
- an order IBKR rejects, cancels or expires while the node runs is reported
  through ``on_next_open_unfilled``;
- only a paper account (``DU...``) unless ``allow_live``;
- the node's IBKR clients: host, port, client id, account, reconciliation on,
  and the instrument provider built from the resolver's contracts;
- the account read: an IBKR position no symbol maps to is left out of the
  holdings, and IBKR's reported cash is the cash.
"""

import asyncio
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ibapi.tag_value import TagValue
from nautilus_trader.adapters.interactive_brokers.common import IB, IBContract, IBContractDetails
from nautilus_trader.adapters.interactive_brokers.parsing.execution import (
    MAP_ORDER_TYPE,
    MAP_TIME_IN_FORCE,
)
from nautilus_trader.accounting.factory import AccountFactory
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import MessageBus, TestClock
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OrderSide, OrderType, TimeInForce
from nautilus_trader.model.events import AccountState
from nautilus_trader.model.identifiers import AccountId, InstrumentId, TraderId
from nautilus_trader.model.objects import AccountBalance, Money
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.stubs.events import TestEventStubs

from quantlab.dataset.base import SymbolName, TickerLookup
from quantlab_ibkr.account import derived_cash, holdings
from quantlab_ibkr.base.config import VenueConfig
from quantlab_ibkr.base.venue import BarInputs, Loop, NextOpenOrder, ReplayRequest
from quantlab_ibkr.decision import ConstructorTargets, DecisionCycle
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.strategy import PortfolioStrategy
from quantlab_ibkr.venue.ibkr.resolver import IbkrResolver
from quantlab_ibkr.venue.ibkr.submitter import DRY_RUN, IbkrOpenSubmitter, order_deadline
from quantlab_ibkr.venue.ibkr.venue import IbkrVenueConfig, OrderDeadlinePassed
from tests.test_live_source_clock import _extend_prices, _run, _store

T = pd.Timestamp("2026-10-07")  # a Wednesday
BEFORE_DEADLINE = pd.Timestamp("2026-10-08 09:00", tz="America/New_York")
AFTER_DEADLINE = pd.Timestamp("2026-10-08 09:21", tz="America/New_York")

#: Symbol -> (ticker, conId, primary exchange); 10003 has no contract.
CONTRACTS = {
    10001: ("AAA", 1001, "NASDAQ"),
    10002: ("BRK.B", 1002, "NYSE"),
    10004: ("DDD", 1004, "NYSE"),
}


class Names(TickerLookup):
    def names(self, symbols, day):
        return [
            SymbolName(CONTRACTS[s][0] if s in CONTRACTS else ("CCC" if s == 10003 else str(s)))
            for s in symbols
        ]


class Contracts:
    """A contract client on ``CONTRACTS``; CCC (10003) has none."""

    def __init__(self):
        self.asked = []

    def details(self, symbol, primary_exchange):
        self.asked.append(symbol)
        for ticker, con_id, primary in CONTRACTS.values():
            if symbol == ticker.replace(".", " "):
                return [IBContractDetails(
                    contract=IBContract(
                        secType="STK", conId=con_id, exchange="SMART", primaryExchange=primary,
                        symbol=symbol, localSymbol=symbol, currency="USD",
                    ),
                    minTick=0.01, secIdList=[TagValue("ISIN", f"US{con_id:010d}")],
                )]
        return []


def _resolver(symbols=(10001, 10002, 10003, 10004)):
    return IbkrResolver(symbols, T, Names(), Contracts())


class Recorder:
    def __init__(self):
        self.submitted, self.unfilled = [], []

    def order_submitted(self, order, client_order_id):
        self.submitted.append((order, client_order_id))

    def order_unfilled(self, order, reason):
        self.unfilled.append((order, reason))

    def order_refused(self, client_order_id, status, reason):
        pass


def _strategy(submitter, now):
    """A ``PortfolioStrategy`` on nautilus's bus, cache and a test clock at ``now``.

    The risk engine's endpoint is captured: the fake execution path.
    """
    clock = TestClock()
    clock.set_time(pd.Timestamp(now).tz_convert("UTC").value)
    trader_id = TraderId("TESTER-001")
    msgbus = MessageBus(trader_id=trader_id, clock=clock)
    cache = Cache()
    portfolio = Portfolio(msgbus=msgbus, cache=cache, clock=clock)
    commands = []
    msgbus.register(endpoint="RiskEngine.execute", handler=commands.append)
    recorder = Recorder()
    venue = SimpleNamespace(resolver=submitter.resolver, submitter=submitter)
    strategy = PortfolioStrategy(venue=venue, cycle=None, recorder=recorder)
    strategy.register(trader_id, portfolio, msgbus, cache, clock)
    submitter.attach(strategy)
    return strategy, clock, commands, recorder


def _orders():
    return [
        NextOpenOrder(10001, "BUY", 25, T),
        NextOpenOrder(10002, "SELL", 7, T),
        NextOpenOrder(10004, "BUY", 3, T),
    ]


def test_orders_are_market_on_open_whole_shares_sells_first():
    submitter = IbkrOpenSubmitter(_resolver())
    strategy, _, commands, recorder = _strategy(submitter, BEFORE_DEADLINE)
    submitter.submit(_orders(), T)

    shape = [
        (str(c.order.instrument_id), c.order.side, c.order.quantity.as_decimal(),
         c.order.order_type, c.order.time_in_force)
        for c in commands
    ]
    assert shape == [
        ("BRK-B.NYSE", OrderSide.SELL, Decimal(7), OrderType.MARKET, TimeInForce.AT_THE_OPEN),
        ("AAA.NASDAQ", OrderSide.BUY, Decimal(25), OrderType.MARKET, TimeInForce.AT_THE_OPEN),
        ("DDD.NYSE", OrderSide.BUY, Decimal(3), OrderType.MARKET, TimeInForce.AT_THE_OPEN),
    ]
    # The IB adapter sends them as IBKR's MKT / OPG.
    assert MAP_ORDER_TYPE[OrderType.MARKET] == "MKT"
    assert MAP_TIME_IN_FORCE[TimeInForce.AT_THE_OPEN] == "OPG"
    assert [o.permno for o, _ in recorder.submitted] == [10002, 10001, 10004]
    assert submitter.released and recorder.unfilled == []
    assert strategy.cache.orders_inflight() == []  # nothing sent past the risk endpoint


def test_orders_are_paced():
    submitter = IbkrOpenSubmitter(_resolver(), max_orders_per_second=2)
    _, clock, commands, _ = _strategy(submitter, BEFORE_DEADLINE)
    submitter.submit(_orders(), T)
    assert len(commands) == 2 and not submitter.released
    for handler in clock.advance_time(clock.timestamp_ns() + 1_000_000_000):
        handler.handle()
    assert len(commands) == 3 and submitter.released


def test_no_order_leaves_after_the_deadline():
    assert order_deadline(T) == pd.Timestamp("2026-10-08 09:20", tz="America/New_York")
    assert order_deadline(pd.Timestamp("2026-10-09"), "09:00") == pd.Timestamp(
        "2026-10-12 09:00", tz="America/New_York"
    )
    submitter = IbkrOpenSubmitter(_resolver())
    _, _, commands, recorder = _strategy(submitter, AFTER_DEADLINE)
    submitter.submit(_orders(), T)
    assert commands == [] and submitter.submitted == []
    assert [o.permno for o, _ in recorder.unfilled] == [10002, 10001, 10004]
    assert all("past the order deadline 2026-10-08 09:20" in r for _, r in recorder.unfilled)


def test_a_dry_run_reports_every_order_and_submits_none():
    submitter = IbkrOpenSubmitter(_resolver(), dry_run=True)
    _, _, commands, recorder = _strategy(submitter, AFTER_DEADLINE)
    submitter.submit(_orders(), T)
    assert commands == [] and submitter.submitted == []
    assert [(o.permno, r) for o, r in recorder.unfilled] == [
        (10002, DRY_RUN), (10001, DRY_RUN), (10004, DRY_RUN)
    ]
    assert [o.permno for o in submitter.decided] == [10002, 10001, 10004]


@pytest.mark.parametrize(
    "event, status",
    [
        (TestEventStubs.order_canceled, "canceled"),
        (TestEventStubs.order_rejected, "rejected"),
        (TestEventStubs.order_expired, "expired"),
    ],
)
def test_an_order_ibkr_ends_unfilled_is_reported_once(event, status):
    submitter = IbkrOpenSubmitter(_resolver())
    strategy, _, commands, recorder = _strategy(submitter, BEFORE_DEADLINE)
    submitter.submit(_orders()[:1], T)
    (command,) = commands
    for _ in range(2):  # a repeated event is reported once
        strategy.msgbus.publish(f"events.order.{strategy.id}", event(command.order))
    assert [(o.permno, o.quantity) for o, _ in recorder.unfilled] == [(10001, 25)]
    assert recorder.unfilled[0][1].startswith(f"{status} by IBKR")


def test_only_a_paper_account_unless_allow_live(monkeypatch):
    monkeypatch.setenv("TWS_ACCOUNT", "DU7654321")
    assert IbkrVenueConfig().account() == "DU7654321"
    assert IbkrVenueConfig(account_id="DU1111111").account() == "DU1111111"
    monkeypatch.setenv("TWS_ACCOUNT", "U7654321")
    with pytest.raises(ValueError, match="not an IBKR paper account"):
        IbkrVenueConfig().account()
    assert IbkrVenueConfig(allow_live=True).account() == "U7654321"
    monkeypatch.delenv("TWS_ACCOUNT")
    with pytest.raises(ValueError, match="TWS_ACCOUNT"):
        IbkrVenueConfig().account()


def test_config_round_trips_and_refuses_a_bad_deadline():
    config = IbkrVenueConfig(prediction_store="s.zarr", live_dir="live", order_deadline="09:15")
    assert VenueConfig.from_config(config.get_config()) == config
    with pytest.raises(ValueError, match="HH:MM"):
        IbkrVenueConfig(order_deadline="9am")


@pytest.fixture
def live_run(tmp_path):
    """A run, its prices extended to a rebalance bar t, and t's live prediction row."""
    run_dir = _run(tmp_path)
    bars = _extend_prices(tmp_path)  # 19 bars: the last, bar 18, rebalances (every 3)
    store = _store(tmp_path, run_dir, [(bars[-1], [0.4, 0.3, 0.2, 0.1])])
    return QuantlabRun.load(run_dir), store, bars[-1], tmp_path


def _build(live_run, now, **fields):
    run, store, t, root = live_run
    fields = {"account_id": "DU1234567", **fields}
    config = IbkrVenueConfig(
        prediction_store=str(store.path), live_dir=str(root / "live_run"),
        host="10.0.0.5", port=4002, client_id=7, **fields,
    )
    return config.build(
        run, ReplayRequest(t, t, (), Loop.CLOSED),
        contract_client=Contracts(), ticker_lookup=Names(), now=now,
    )


def test_a_deciding_day_refuses_to_start_after_its_deadline(live_run):
    t = live_run[2]
    late = order_deadline(t) + pd.Timedelta(minutes=1)
    with pytest.raises(OrderDeadlinePassed, match="order deadline"):
        _build(live_run, late)
    assert _build(live_run, late, dry_run=True).decision.decides  # a dry run may run late


def test_force_decide_needs_a_dry_run():
    with pytest.raises(ValueError, match="force_decide needs dry_run"):
        IbkrVenueConfig(force_decide=True)
    config = IbkrVenueConfig(dry_run=True, force_decide=True)
    assert VenueConfig.from_config(config.get_config()) == config


def test_building_refuses_a_live_account(live_run):
    t = live_run[2]
    with pytest.raises(ValueError, match="not an IBKR paper account"):
        _build(live_run, order_deadline(t) - pd.Timedelta(hours=1), account_id="U1234567")


def test_venue_assembles_the_node_without_connecting(live_run):
    t = live_run[2]
    venue = _build(live_run, order_deadline(t) - pd.Timedelta(hours=1))
    assert venue.decision.decides and venue.source.t == t
    # 10003 has no contract: kept out of the targets.
    assert venue.resolver.unresolved(t) == {10003}
    predictions = venue.source.inputs(t).predictions["ret_5"]
    np.testing.assert_array_equal(predictions.values, [0.4, 0.3, np.nan, 0.1])

    config = venue.node_config()
    assert config.exec_engine.reconciliation
    data, exec_ = config.data_clients[IB], config.exec_clients[IB]
    for client in (data, exec_):
        assert (client.ibg_host, client.ibg_port, client.ibg_client_id) == ("10.0.0.5", 4002, 7)
        assert client.instrument_provider == venue.resolver.provider_config()
        assert sorted(c.conId for c in client.instrument_provider.load_contracts) == [
            1001, 1002, 1004
        ]
    assert exec_.account_id == "DU1234567"
    # Loading every contract can take minutes (us3000: about 3,000); the
    # engines may take the run's whole timeout to connect.
    assert config.timeout_connection == venue.config.timeout_secs

    loop = asyncio.new_event_loop()
    try:
        node = venue.build_node(loop)
        assert str(node.trader_id) == "QUANTLAB-IBKR"
        assert not node.kernel.is_running()
    finally:
        loop.close()


def test_a_held_day_resolves_only_the_held_symbols(live_run, tmp_path):
    run, _, t, root = live_run
    (root / "live_run").mkdir()
    (root / "live_run" / "con_ids.json").write_text('{"1002": 10002}')
    # A store without t's row: the day holds, so no deadline applies and only
    # the held symbol (from the conId cache) is looked up.
    store = _store(tmp_path / "other", run.run_dir, [(t - pd.Timedelta(days=1), [0.1] * 4)])
    client = Contracts()
    venue = IbkrVenueConfig(
        prediction_store=str(store.path), live_dir=str(root / "live_run"),
        account_id="DU1234567",
    ).build(
        run, ReplayRequest(t, t, (), Loop.CLOSED),
        contract_client=client, ticker_lookup=Names(),
        now=order_deadline(t) + pd.Timedelta(hours=3),
    )
    assert not venue.decision.decides
    assert venue.decision.hold_reason.startswith("no prediction row")
    assert client.asked == ["BRK B"]
    assert venue.resolver.con_id_cache == {1002: 10002}


def test_holdings_leave_out_positions_no_symbol_maps_to_and_cash_is_ibkrs():
    resolver = _resolver()

    def position(instrument, quantity, avg_px=10.0):
        return SimpleNamespace(
            instrument_id=InstrumentId.from_str(instrument),
            signed_decimal_qty=lambda: Decimal(quantity),
            avg_px_open=avg_px,
        )

    positions = [position("AAA.NASDAQ", 10), position("XYZ.NYSE", 5), position("DDD.NYSE", -2)]
    reported = SimpleNamespace(info={"TotalCashValue": 1234.5})
    account = SimpleNamespace(
        last_event=reported,
        base_currency=None,
        balance_total=lambda currency: SimpleNamespace(as_double=lambda: 99_999.0),
    )
    cache = SimpleNamespace(positions_open=lambda: positions, accounts=lambda: [account])
    assert holdings(cache, resolver) == {10001: 10, 10004: -2}
    gaps = resolver.report_positions([p.instrument_id for p in positions], T)
    assert [g.instrument_id for g in gaps] == ["XYZ.NYSE"]
    assert derived_cash(cache) == 1234.5

    # Without a reported cash (the backtest venue): balance minus the positions' cost.
    account.last_event = SimpleNamespace(info={})
    assert derived_cash(cache) == 99_999.0 - (10 - 2 + 5) * 10.0


class CycleRecorder(Recorder):
    def __init__(self):
        super().__init__()
        self.cycles = []

    def record_cycle(self, t, result):
        self.cycles.append((t, result))


def test_the_strategy_decides_t_on_the_venue_and_submits_market_on_open(live_run):
    """The unchanged ``PortfolioStrategy`` on the IBKR venue's parts, up to the risk engine."""
    t = live_run[2]
    venue = _build(live_run, order_deadline(t) - pd.Timedelta(hours=10))
    clock = TestClock()
    clock.set_time((order_deadline(t) - pd.Timedelta(hours=10)).tz_convert("UTC").value)
    trader_id = TraderId("TESTER-001")
    msgbus = MessageBus(trader_id=trader_id, clock=clock)
    cache = Cache()
    net_liquidation = Money(1_000_000, USD)  # IBKR's balance; the cash is TotalCashValue
    cache.add_account(AccountFactory.create(AccountState(
        account_id=AccountId("IB-DU1234567"), account_type=AccountType.MARGIN,
        base_currency=USD, reported=True,
        balances=[AccountBalance(net_liquidation, Money(0, USD), net_liquidation)],
        margins=[], info={"TotalCashValue": 10_000.0}, event_id=UUID4(), ts_event=0, ts_init=0,
    )))
    commands = []
    msgbus.register(endpoint="RiskEngine.execute", handler=commands.append)
    recorder = CycleRecorder()
    strategy = PortfolioStrategy(
        venue=venue, cycle=DecisionCycle(ConstructorTargets(venue.decision_inputs)),
        recorder=recorder,
    )
    strategy.register(trader_id, Portfolio(msgbus=msgbus, cache=cache, clock=clock), msgbus, cache, clock)

    strategy.on_start()  # what the node calls once it has reconciled the account
    for handler in clock.advance_time(clock.timestamp_ns() + 1):
        handler.handle()

    assert venue.clock.fired and venue.clock.error is None and venue.init_cash == 10_000.0
    assert venue.position_gaps == []
    ((decided, result),) = recorder.cycles
    assert decided == t and result.equity == 10_000.0
    # TopN(2) on [0.4, 0.3, excluded, 0.1]: half each in 10001 (AAA) and 10002 (BRK.B).
    assert {str(c.order.instrument_id) for c in commands} == {"AAA.NASDAQ", "BRK-B.NYSE"}
    assert all(
        c.order.order_type == OrderType.MARKET
        and c.order.time_in_force == TimeInForce.AT_THE_OPEN
        and c.order.side == OrderSide.BUY
        for c in commands
    )
    assert [(o.permno, o.quantity) for o in venue.submitter.submitted] == [
        (o.permno, o.quantity) for o in result.orders
    ]
    assert venue.done(strategy)


def test_a_forced_decision_runs_the_rule_on_a_bar_off_the_cadence():
    from quantlab_ibkr.venue.ibkr.clock import LiveDecision
    from quantlab_ibkr.venue.ibkr.venue import LiveTargets

    t = pd.Timestamp("2026-10-02")
    calls = []

    class Rule:
        decision_inputs = SimpleNamespace(rebalances=lambda bar: False)

        def decide(self, bar, predictions, current_weights):
            calls.append(bar)
            return "decided"

    close = pd.Series({10001: 30.0})
    inputs = BarInputs(timestamp=t, predictions=xr.Dataset(), close=close, delisted=close.isna())
    forced = LiveTargets(Rule(), LiveDecision(t, True, None))
    assert forced.targets(inputs, pd.Series(dtype=float)) == "decided" and calls == [t]
    held = LiveTargets(Rule(), LiveDecision(t, False, "not a rebalance bar"))
    assert held.targets(inputs, pd.Series(dtype=float)) is None and calls == [t]
