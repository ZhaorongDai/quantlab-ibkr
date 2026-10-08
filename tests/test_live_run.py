"""The live run directory and its steps (#51): ``live.decide`` / ``live.record`` on a fake IBKR.

No Gateway. The fixture quantlab run (TopN(2), rebalancing every 3 bars) is
traded live over its last bars, one simulated day per bar: the price store
is cut to end at the day's bar (as the daily prediction job leaves it), the
live prediction store holds the run's prediction panel row of that bar, and
``FakeVenue`` runs the unchanged ``PortfolioStrategy`` on nautilus's bus,
cache and a test clock over a fake account (``Broker``: cash as IBKR's
``TotalCashValue``, positions in the cache, orders taken at the risk
engine's endpoint, filled at the next bar's raw open with the run's fee
fraction, reported back as IBKR executions and completed orders). Locked:

- the appended live run directory holds the files a closed-loop backtest of
  the same bars writes (decisions with current weights, holdings, equity,
  orders, events, metrics), modulo venue-only fields;
- re-running ``decide`` on a recorded bar does nothing; an order already
  working at IBKR for the bar is adopted, never sent again; re-running
  ``record`` adds no fill twice;
- the Decision recheck passes on the live run directory, read from the live
  prediction store;
- an order IBKR rejects is recorded once, ``rejected`` with IBKR's reason;
- the command line builds the live config from a file and flags.
"""

import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from ibapi.tag_value import TagValue
from nautilus_trader.accounting.factory import AccountFactory
from nautilus_trader.adapters.interactive_brokers.common import IBContract, IBContractDetails
from nautilus_trader.cache.cache import Cache
from nautilus_trader.common.component import MessageBus, TestClock
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide
from nautilus_trader.model.events import AccountState
from nautilus_trader.model.identifiers import AccountId, PositionId, TraderId
from nautilus_trader.model.objects import AccountBalance, Money, Price, Quantity
from nautilus_trader.model.position import Position
from nautilus_trader.portfolio.portfolio import Portfolio
from nautilus_trader.test_kit.stubs.events import TestEventStubs
from nautilus_trader.test_kit.stubs.execution import TestExecStubs

from quantlab.dataset.base import SymbolName, TickerLookup
from quantlab.runs.live_predictions import LivePredictionStore
from quantlab_ibkr import cli, live
from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.base.venue import NextOpenOrder, VenueReport
from quantlab_ibkr.decision import CycleResult
from quantlab_ibkr.outputs import LiveRecorder, RunRecorder
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.runner import run as run_backtest
from quantlab_ibkr.venue.backtest.venue import BacktestVenueConfig, ExecutionConfig
from quantlab_ibkr.venue.ibkr.reports import IbkrExecution, IbkrOrderState, decision_date_of
from quantlab_ibkr.venue.ibkr.submitter import order_deadline
from quantlab_ibkr.venue.ibkr.venue import TRADER_ID, IbkrVenue, IbkrVenueConfig
from tests.quantlab_run_fixture import write_crsp_store
from tests.test_closed_loop_replay import LABELS, PERMNOS, _market, _predictions
from tests.test_live_source_clock import N_BARS, _run

#: The live days: bars FIRST..N_BARS-1 of the run (rebalancing on bars 6 and 9).
FIRST = 5
#: Symbol -> (ticker, conId).
CONTRACTS = {10001: ("AAA", 1001), 10002: ("BBB", 1002), 10003: ("CCC", 1003), 10004: ("DDD", 1004)}
SYMBOL_OF = {con_id: permno for permno, (_, con_id) in CONTRACTS.items()}


class Names(TickerLookup):
    def names(self, symbols, day):
        return [SymbolName(CONTRACTS[s][0]) for s in symbols]


class Contracts:
    def details(self, symbol, primary_exchange):
        for ticker, con_id in CONTRACTS.values():
            if symbol == ticker:
                return [IBContractDetails(
                    contract=IBContract(
                        secType="STK", conId=con_id, exchange="SMART", primaryExchange="NASDAQ",
                        symbol=symbol, localSymbol=symbol, currency="USD",
                    ),
                    minTick=0.01, secIdList=[TagValue("ISIN", f"US{con_id:010d}")],
                )]
        return []


class Broker:
    """A fake IBKR account and its order reports (``OrderReports``)."""

    def __init__(self, cash, opens, bars, fee_rate):
        self.cash, self.opens, self.bars, self.fee_rate = cash, opens, bars, fee_rate
        self.positions: dict = {}
        self.working: list[IbkrOrderState] = []
        self.execs: list[IbkrExecution] = []
        self.completed: list[IbkrOrderState] = []
        self.received = 0
        self.reject = set()

    # OrderReports
    def executions(self, account):
        return list(self.execs)

    def open_orders(self, account):
        return list(self.working)

    def completed_orders(self, account):
        return list(self.completed)

    def accept(self, command):
        order = command.order
        con_id = CONTRACTS[SYMBOL_OF_INSTRUMENT[str(order.instrument_id)]][1]
        self.received += 1
        self.working.append(IbkrOrderState(
            order.client_order_id.value, con_id, order.side.name,
            int(order.quantity.as_decimal()), "PreSubmitted",
        ))

    def open_auction(self, k):
        """Fill every working order at bar k's raw open."""
        when = pd.Timestamp(f"{self.bars[k].date()} 09:30", tz="America/New_York").tz_convert("UTC")
        for order in self.working:
            permno = SYMBOL_OF[order.con_id]
            if permno in self.reject:
                self.completed.append(order.__class__(
                    order.order_ref, order.con_id, order.side, order.quantity, "Cancelled",
                    "Order rejected - no opening auction",
                ))
                continue
            price = self.opens[permno][k]
            fee = round(order.quantity * price * self.fee_rate, 2)
            sign = 1 if order.side == "BUY" else -1
            self.cash -= sign * order.quantity * price + fee
            self.positions[permno] = self.positions.get(permno, 0) + sign * order.quantity
            self.execs.append(IbkrExecution(
                f"E{len(self.execs)}", order.order_ref, order.con_id, order.side,
                order.quantity, price, fee, when,
            ))
            self.completed.append(IbkrOrderState(
                order.order_ref, order.con_id, order.side, order.quantity, "Filled",
            ))
        self.working = []


SYMBOL_OF_INSTRUMENT = {f"{ticker}.NASDAQ": permno for permno, (ticker, _) in CONTRACTS.items()}


class FakeVenue(IbkrVenue):
    """The IBKR venue with its node replaced by nautilus's bus, cache and a test clock."""

    def __init__(self, *args, broker, now, **kwargs):
        super().__init__(*args, now=now, **kwargs)
        self.broker, self.now = broker, now

    def run(self, strategy):
        clock = TestClock()
        clock.set_time(pd.Timestamp(self.now).tz_convert("UTC").value)
        trader_id = TraderId(TRADER_ID)
        msgbus = MessageBus(trader_id=trader_id, clock=clock)
        cache = Cache()
        balance = Money(10_000_000, USD)
        cache.add_account(AccountFactory.create(AccountState(
            account_id=AccountId("IB-DU1234567"), account_type=AccountType.MARGIN,
            base_currency=USD, reported=True,
            balances=[AccountBalance(balance, Money(0, USD), balance)], margins=[],
            info={"TotalCashValue": self.broker.cash}, event_id=UUID4(), ts_event=0, ts_init=0,
        )))
        instruments = {str(i.id): i for i in self.resolver.instruments()}
        for permno, quantity in self.broker.positions.items():
            if not quantity:
                continue
            instrument = instruments[f"{CONTRACTS[permno][0]}.NASDAQ"]
            order = TestExecStubs.market_order(
                instrument=instrument,
                order_side=OrderSide.BUY if quantity > 0 else OrderSide.SELL,
                quantity=Quantity(abs(quantity), 0),
            )
            fill = TestEventStubs.order_filled(
                order, instrument, last_px=Price(10.0, 2), position_id=PositionId(f"P-{permno}")
            )
            cache.add_position(Position(instrument, fill), OmsType.NETTING)
        msgbus.register(endpoint="RiskEngine.execute", handler=self.broker.accept)
        strategy.register(trader_id, Portfolio(msgbus=msgbus, cache=cache, clock=clock), msgbus, cache, clock)
        strategy.start()  # RUNNING, so order events reach it; calls on_start
        for handler in clock.advance_time(clock.timestamp_ns() + 1):
            handler.handle()
        for event in self.broker_rejections(strategy):
            strategy.msgbus.publish(f"events.order.{strategy.id}", event)
        self._write_con_ids()
        if self.clock.error is not None:
            raise self.clock.error
        return VenueReport(fees="IBKR's commissions, as IBKR reports them")

    def broker_rejections(self, strategy):
        """IBKR rejects on submission the orders of ``broker.reject_on_submit``."""
        events = []
        for order in strategy.cache.orders():
            permno = SYMBOL_OF_INSTRUMENT[str(order.instrument_id)]
            if permno in getattr(self.broker, "reject_on_submit", ()):
                events.append(TestEventStubs.order_rejected(order))
                self.broker.working = [
                    w for w in self.broker.working if w.order_ref != order.client_order_id.value
                ]
        return events


class Day:
    """The live setup: the run, its stores, the broker and the config."""

    def __init__(self, root: Path):
        self.root = root
        self.run_dir = _run(root)
        self.run = QuantlabRun.load(self.run_dir)
        self.bars = pd.bdate_range("2024-01-02", periods=N_BARS)
        self.open, self.close = _market(N_BARS, seed=1)
        self.panel = _predictions(N_BARS, seed=2)["ret_5"]
        self.store = LivePredictionStore(root / "live" / "live_predictions.zarr")
        self.header = LivePredictionStore.header(
            LABELS, run_dir=self.run_dir, checkpoint=root / "m.joblib"
        )
        self.broker = Broker(self.run.init_cash, self.open, self.bars, self.run.fees)
        self.config = TraderConfig(
            str(self.run_dir),
            IbkrVenueConfig(
                prediction_store=str(self.store.path),
                live_dir=str(root / "live" / "run"),
                account_id="DU1234567",
            ),
        )

    def predict(self, k: int) -> None:
        """Append bar k's row (the run's prediction panel row) to the live store, as the daily job."""
        if self.store.exists and self.store.has(self.bars[k]):
            return
        row = xr.Dataset(
            {"ret_5": (("timestamp", "symbol"), np.array([[self.panel[p][k] for p in PERMNOS]]))},
            coords={"timestamp": [self.bars[k]], "symbol": np.array(PERMNOS, dtype=np.int64)},
        )
        self.store.append(row, self.header, {"data_fingerprint": {}})

    def cut_prices(self, k: int) -> None:
        """Leave the price store ending at bar k, as the daily job leaves it before bar k+1."""
        write_crsp_store(
            self.root / "quantlab" / "crsp.zarr",
            self.bars[: k + 1],
            {p: v[: k + 1] for p, v in self.open.items()},
            {p: v[: k + 1] for p, v in self.close.items()},
        )

    def factory(self, k: int):
        def build(run, request, *, now, working_orders):
            return FakeVenue(
                run, request, self.config.venue,
                contract_client=Contracts(), ticker_lookup=Names(),
                working_orders=working_orders, broker=self.broker, now=now,
            )
        return build

    def now(self, k: int) -> pd.Timestamp:
        return order_deadline(self.bars[k]) - pd.Timedelta(hours=10)

    def decide(self, k: int):
        self.cut_prices(k)
        self.predict(k)
        return live.decide(
            self.config, now=self.now(k), venue_factory=self.factory(k), order_reports=self.broker
        )

    def record(self):
        return live.record(self.config, order_reports=self.broker)

    def trade(self, days):
        for k in days:
            self.decide(k)
            if k + 1 < N_BARS:
                self.broker.open_auction(k + 1)
                self.record()

    @property
    def live_dir(self) -> Path:
        return Path(self.config.venue.live_dir)


def _backtest(day: Day, start: int, end: int) -> Path:
    return run_backtest(TraderConfig(
        str(day.run_dir),
        BacktestVenueConfig(ExecutionConfig(fee_model="fraction", slippage=0.0)),
        start=str(day.bars[start].date()), end=str(day.bars[end].date()),
        output_dir=str(day.root / "backtest"),
    ))


def _zarr(path: Path) -> xr.Dataset:
    with xr.open_zarr(path) as dataset:
        return dataset.load()


def _events(path: Path, live_only=("live_hold", "excluded_symbols", "position_gap",
                                    "adopted_orders", "timed_out")) -> list:
    events = json.loads((path / "events.json").read_text())["events"]
    return [e for e in events if e["type"] not in live_only]


@pytest.fixture
def day(tmp_path):
    return Day(tmp_path)


def test_appended_live_days_write_the_closed_loop_backtests_files(day):
    backtest = _backtest(day, FIRST, N_BARS - 1)
    day.trade(range(FIRST, N_BARS))
    live_dir = day.live_dir

    for name, variables in (
        ("decisions.zarr", ("weight", "current_weight")),
        ("holdings.zarr", ("holding",)),
        ("equity.zarr", ("value", "returns")),
    ):
        expected, got = _zarr(backtest / name), _zarr(live_dir / name)
        for variable in variables:
            xr.testing.assert_allclose(got[variable], expected[variable], rtol=1e-9)
    decisions = _zarr(live_dir / "decisions.zarr")
    assert list(decisions["timestamp"].values) == [day.bars[6], day.bars[9]]
    assert len(_zarr(live_dir / "equity.zarr")["timestamp"]) == N_BARS - FIRST  # a row per day

    expected, got = _zarr(backtest / "orders.zarr"), _zarr(live_dir / "orders.zarr")
    for variable in ("decision_date", "symbol", "side", "quantity", "status", "filled_quantity", "reason"):
        np.testing.assert_array_equal(got[variable].values, expected[variable].values)
    for variable in ("fill_price", "fee"):
        np.testing.assert_allclose(got[variable].values, expected[variable].values, rtol=1e-9)
    assert set(got["status"].values) == {"filled"}

    assert _events(live_dir) == _events(backtest)
    expected = json.loads((backtest / "metrics.json").read_text())["whole"]
    got = json.loads((live_dir / "metrics.json").read_text())["whole"]
    for key in ("Total Return [%]", "End Value", "Total Fees Paid", "Total Orders"):
        assert got[key] == pytest.approx(expected[key], rel=1e-9), key
    assert (live_dir / "report.html").is_file() and (live_dir / "config.json").is_file()
    config = json.loads((live_dir / "config.json").read_text())
    assert sorted(config["data_fingerprint"]) == [str(day.bars[6].date()), str(day.bars[9].date())]

    recheck = live.recheck(day.config)
    assert recheck["bars_checked"] == 2 and recheck["bars_differing"] == 0
    assert json.loads((live_dir / live.RECHECK_FILE).read_text())["bars_differing"] == 0


def test_rerunning_decide_and_record_is_idempotent(day):
    day.trade(range(FIRST, 6))
    rebalance = 6
    first = day.decide(rebalance)
    assert first.decided and first.status == "recorded" and len(first.submitted) == 2
    sent = day.broker.received
    files = {
        name: _zarr(day.live_dir / name)
        for name in ("decisions.zarr", "orders.zarr", "equity.zarr", "holdings.zarr")
    }

    again = day.decide(rebalance)
    assert again.status == "already_recorded" and day.broker.received == sent
    for name, dataset in files.items():
        xr.testing.assert_identical(_zarr(day.live_dir / name), dataset)

    # An earlier run sent the orders and failed before recording them: its
    # orders are working at IBKR, tagged with the decision date, and adopted.
    refs = [o.order_ref for o in day.broker.working]
    assert {decision_date_of(r) for r in refs} == {day.bars[rebalance]}
    backup = day.root / "before"
    shutil.copytree(day.live_dir, backup)
    journal = json.loads((day.live_dir / "journal.json").read_text())
    journal["cycles"] = journal["cycles"][:-1]
    journal["orders"] = []
    journal["events"] = []
    (day.live_dir / "journal.json").write_text(json.dumps(journal))
    adopted = day.decide(rebalance)
    assert len(adopted.adopted) == 2 and adopted.submitted == ()
    assert day.broker.received == sent
    orders = _zarr(day.live_dir / "orders.zarr")
    assert sorted(orders["status"].values) == ["submitted", "submitted"]
    xr.testing.assert_identical(orders, files["orders.zarr"])

    day.broker.open_auction(rebalance + 1)
    filled = day.record()
    assert filled.fills == 2 and filled.recheck["bars_differing"] == 0
    after = _zarr(day.live_dir / "orders.zarr")
    assert set(after["status"].values) == {"filled"}
    again = day.record()
    assert again.fills == 0
    xr.testing.assert_identical(_zarr(day.live_dir / "orders.zarr"), after)
    journal = json.loads((day.live_dir / "journal.json").read_text())
    assert len(journal["fills"]) == 2


def test_an_order_ibkr_cancels_at_the_open_is_recorded_unfilled_with_its_reason(day):
    day.trade(range(FIRST, 6))
    day.decide(6)
    day.broker.reject = {SYMBOL_OF[day.broker.working[0].con_id]}
    day.broker.open_auction(7)
    result = day.record()
    assert len(result.ended) == 1 and result.fills == 1
    orders = _zarr(day.live_dir / "orders.zarr")
    status = dict(zip(orders["symbol"].values.tolist(), orders["status"].values.tolist()))
    reason = dict(zip(orders["symbol"].values.tolist(), orders["reason"].values.tolist()))
    (canceled,) = day.broker.reject
    assert status[canceled] == "unfilled"
    assert reason[canceled] == "Cancelled by IBKR: Order rejected - no opening auction"
    events = json.loads((day.live_dir / "events.json").read_text())["events"]
    assert [e["symbol"] for e in events if e["type"] == "unfilled_order"] == [canceled]


def test_a_rejected_order_is_recorded_once_as_rejected_with_ibkrs_reason(day):
    day.trade(range(FIRST, 6))
    day.broker.reject_on_submit = {10001, 10002, 10003, 10004}
    result = day.decide(6)
    assert len(result.unfilled) == 2
    orders = _zarr(day.live_dir / "orders.zarr")
    assert set(orders["status"].values) == {"rejected"}
    assert set(orders["reason"].values) == {"ORDER_REJECTED"}
    events = json.loads((day.live_dir / "events.json").read_text())["events"]
    assert [e for e in events if e["type"] == "unfilled_order"] == []


@pytest.mark.parametrize("refused_first", [True, False])
def test_the_recorder_keeps_one_status_for_a_rejection(refused_first):
    config = TraderConfig("run", BacktestVenueConfig(), name="unit")
    recorder = RunRecorder(config, None, init_cash=1000.0)
    t = pd.Timestamp("2024-01-03")
    order = NextOpenOrder(10001, "BUY", 5, t)
    decision = type("D", (), {})()
    decision.weights = xr.DataArray([0.5], dims="symbol", coords={"symbol": [10001]})
    decision.failure, decision.events = None, {}
    recorder.record_cycle(t, CycleResult(decision, (order,), 1000.0, pd.Series(dtype=float)))
    recorder.order_submitted(order, "O-1")
    steps = [
        lambda: recorder.order_refused("O-1", "rejected", "no permissions"),
        lambda: recorder.order_unfilled(order, "rejected by IBKR: no permissions"),
    ]
    for step in steps if refused_first else steps[::-1]:
        step()
    (row,) = recorder._orders
    assert (row["status"], row["reason"]) == ("rejected", "no permissions")
    assert recorder._events == []


def test_the_journal_restores_everything_recorded(day):
    day.trade(range(FIRST, 8))
    recorder = LiveRecorder(day.config, day.run)
    again = RunRecorder(day.config, day.run, init_cash=0.0)
    again.restore(recorder.state())
    assert again.state() == RunRecorder.state(recorder)
    assert recorder.held_symbols() and recorder.working_orders() == []


def test_the_command_line_builds_the_live_config_from_a_file_and_flags(tmp_path):
    config = TraderConfig("runs/real", IbkrVenueConfig(prediction_store="p.zarr", live_dir="live"))
    path = tmp_path / "live.json"
    path.write_text(json.dumps(config.get_config()))
    parser = cli._parser()
    built = cli._live_config(parser, parser.parse_args(
        ["live", "decide", str(path), "--dry-run", "--port", "4004"]
    ))
    assert built.venue.dry_run and built.venue.port == 4004
    assert built.venue.prediction_store == "p.zarr" and built.quantlab_run == "runs/real"
    flags = cli._live_config(parser, parser.parse_args([
        "live", "record", "--quantlab-run", "runs/real", "--prediction-store", "p.zarr",
        "--live-dir", "live",
    ]))
    assert flags == config
    with pytest.raises(SystemExit):
        cli._live_config(parser, parser.parse_args(["live", "decide", "--quantlab-run", "r"]))


def test_the_command_refuses_a_live_account_before_connecting(day, monkeypatch, capsys):
    config = dataclasses.replace(
        day.config, venue=dataclasses.replace(day.config.venue, account_id=None)
    )
    path = day.root / "live.json"
    path.write_text(json.dumps(config.get_config()))
    monkeypatch.setenv("TWS_ACCOUNT", "U1234567")
    assert cli.main(["live", "decide", str(path)]) == 1
    assert "not an IBKR paper account" in capsys.readouterr().err
