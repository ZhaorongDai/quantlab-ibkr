"""The close briefing's trades, holdings, contributors and events (#68) on the fake live harness.

The live days are ``tests/test_live_run.py``'s ``Day`` (TopN(2) rebalancing
on bars 6 and 9, filling at the opens of bars 7 and 10). Locked:

- the top holdings, the holdings count and the cash weight are the live run
  directory's ``holdings.zarr`` on t;
- each security's contribution is its holding's change of value over t plus
  the cash it moved (fills on t, cash corporate actions), over the previous
  bar's equity: the holding weight times the security's return on a bar
  without fills; with the cash residual they sum to t's return;
- t's turnover is quantlab's one-sided turnover of t's decision
  (``sum |weight - current_weight|``), the order count the orders decided
  on t;
- the events listed are t's entries of ``events.json`` (an unfilled order on
  the bar after its decision, when it should have filled), the orders the
  venue refused at t's submission, and the Decision recheck's latest result;
- a security the ticker lookup cannot name is shown by its PERMNO.
"""

import json

import numpy as np
import pytest
import xarray as xr

from quantlab.dataset.base import SymbolName, TickerLookup
from quantlab_ibkr import briefing, live
from quantlab_ibkr.outputs import LiveRecorder, read_equity
from tests.test_close_briefing import BENCHMARK
from tests.test_live_run import CONTRACTS, FIRST, SYMBOL_OF, Day
from tests.test_live_source_clock import N_BARS


@pytest.fixture
def day(tmp_path):
    return Day(tmp_path, benchmark=BENCHMARK)


def _zarr(path):
    with xr.open_zarr(path) as dataset:
        return dataset.load()


def _contributions(book):
    return {f.symbol: f.pct for f in (*book.contributors, *book.detractors)}


def test_the_top_holdings_count_and_cash_weight_are_the_directorys_holdings_on_t(day):
    day.trade(range(FIRST, N_BARS))
    book = briefing.close_briefing(day.config).book
    held = _zarr(day.live_dir / "holdings.zarr")["holding"].isel(timestamp=-1).to_series()
    held = held[held != 0.0].sort_values(ascending=False)
    assert [f.symbol for f in book.top_holdings] == list(held.index[:5])
    assert [f.pct for f in book.top_holdings] == pytest.approx(list(held.values[:5] * 100.0), rel=1e-12)
    assert book.holdings == len(held) == 2
    assert book.cash_weight_pct == pytest.approx((1.0 - held.sum()) * 100.0, rel=1e-12)


def test_on_a_bar_without_fills_a_contribution_is_the_holding_weight_times_the_return(day):
    day.trade(range(FIRST, N_BARS))  # t = bar 11: holdings since bar 10's open, no fill on t
    book = briefing.close_briefing(day.config).book
    holdings = _zarr(day.live_dir / "holdings.zarr")["holding"].to_pandas()
    previous = holdings.iloc[-2]
    expected = {
        permno: previous[permno] * (day.close[permno][N_BARS - 1] / day.close[permno][N_BARS - 2] - 1.0) * 100
        for permno in previous.index
        if previous[permno] != 0.0
    }
    assert _contributions(book) == pytest.approx(expected, rel=1e-9)
    assert sorted(f.pct for f in book.contributors) == sorted(v for v in expected.values() if v > 0)
    assert [f.pct for f in book.contributors] == sorted((f.pct for f in book.contributors), reverse=True)
    assert [f.pct for f in book.detractors] == sorted(f.pct for f in book.detractors)
    r = read_equity(day.live_dir)["returns"].values[-1] * 100.0
    assert sum(expected.values()) + book.cash_contribution_pct == pytest.approx(r, rel=1e-12)
    assert book.cash_contribution_pct == pytest.approx(0.0, abs=1e-9)  # cash earns nothing


def test_on_a_fill_bar_the_contributions_and_cash_sum_to_ts_return(day):
    day.trade(range(FIRST, 8))  # t = bar 7: bar 6's orders filled at its open
    book = briefing.close_briefing(day.config).book
    equity = read_equity(day.live_dir)
    before = float(equity["value"].values[-2])
    fills = [f for f in LiveRecorder(day.config, day.run).state()["fills"]
             if f["timestamp"].startswith(str(day.bars[7].date()))]
    assert len(fills) == 2
    expected = {
        f["symbol"]: (f["size"] * (day.close[f["symbol"]][7] - f["price"]) - f["fee"]) / before * 100.0
        for f in fills
    }
    got = _contributions(book)
    assert got == pytest.approx(expected, rel=1e-6)
    r = float(equity["returns"].values[-1]) * 100.0
    assert sum(got.values()) + book.cash_contribution_pct == pytest.approx(r, abs=1e-12)
    assert book.cash_contribution_pct == pytest.approx(0.0, abs=1e-6)


def test_turnover_and_orders_are_ts_fills_and_ts_decision(day):
    day.trade(range(FIRST, 10))  # t = bar 9, a rebalance; bar 10's fills recorded after it
    book = briefing.close_briefing(day.config).book
    decisions = _zarr(day.live_dir / "decisions.zarr").sel(timestamp=day.bars[9])
    traded = np.abs(decisions["weight"].values - np.nan_to_num(decisions["current_weight"].values))
    assert book.decided_turnover_pct == pytest.approx(np.nansum(traded) * 100.0, rel=1e-12)
    assert book.turnover_pct == 0.0 and book.filled_orders == 0  # bar 6's orders filled on bar 7
    orders = _zarr(day.live_dir / "orders.zarr")
    assert book.orders == int((orders["decision_date"].values == np.datetime64(day.bars[9])).sum()) == 3
    rows = dict(briefing.close_briefing(day.config).rows)
    assert rows["决策换手 (Decided turnover)"] == f"{book.decided_turnover_pct:.2f}%"
    assert rows["新订单 (Orders)"] == "3"

    day.decide(10)  # t = bar 10: bar 9's orders filled at its open; no decision, no orders
    book = briefing.close_briefing(day.config).book
    fills = [f for f in LiveRecorder(day.config, day.run).state()["fills"]
             if f["timestamp"].startswith(str(day.bars[10].date()))]
    before = float(read_equity(day.live_dir)["value"].values[-2])
    expected = sum(abs(f["size"]) * f["price"] for f in fills) / before * 100.0
    assert book.turnover_pct == pytest.approx(expected, rel=1e-12)  # quantlab's turnover of the fill bar
    metrics = json.loads((day.live_dir / "metrics.json").read_text())["whole"]
    assert metrics["Total Turnover [%]"] == pytest.approx(
        expected + briefing_turnover_at(day, 7), rel=1e-9
    )
    assert book.filled_orders == len(fills) == 3
    assert book.decided_turnover_pct is None and book.orders == 0
    rows = dict(briefing.close_briefing(day.config).rows)
    assert rows["决策换手 (Decided turnover)"] == "未调仓 (no rebalance)"
    assert rows["成交换手 (Turnover)"] == f"{expected:.2f}%"
    assert rows["成交订单 (Filled orders)"] == "3"


def briefing_turnover_at(day, k):
    """quantlab's turnover of bar k's fills, recomputed from the journal."""
    equity = read_equity(day.live_dir)
    before = float(equity["value"].sel(timestamp=day.bars[k - 1]))
    fills = [f for f in LiveRecorder(day.config, day.run).state()["fills"]
             if f["timestamp"].startswith(str(day.bars[k].date()))]
    return sum(abs(f["size"]) * f["price"] for f in fills) / before * 100.0


def test_the_events_listed_are_ts_entries_and_the_recheck_result(day):
    day.trade(range(FIRST, 6))
    day.decide(6)
    day.broker.reject = {SYMBOL_OF[day.broker.working[0].con_id]}
    (canceled,) = day.broker.reject
    day.broker.open_auction(7)
    day.record()
    day.decide(7)
    # A position gap on t and one on an earlier bar, as decide would add them.
    recorder = LiveRecorder(day.config, day.run)
    for k in (6, 7):
        recorder.add_event({"type": "position_gap", "timestamp": str(day.bars[k].date()),
                            "instrument_id": "ZZZ.NASDAQ", "reason": "no symbol maps to it"})
    recorder.write()

    brief = briefing.close_briefing(day.config)
    events = json.loads((day.live_dir / "events.json").read_text())["events"]
    on_t = [e for e in events if e.get("timestamp") == str(day.bars[7].date())]
    unfilled = [e for e in events if e["type"] == "unfilled_order"]
    assert [e["symbol"] for e in unfilled] == [canceled]  # decided on 6, missed bar 7's open
    assert list(brief.book.events) == unfilled + on_t
    assert [e["type"] for e in on_t] == ["position_gap"]
    recheck = json.loads((day.live_dir / live.RECHECK_FILE).read_text())
    assert brief.book.recheck == recheck
    rows = brief.rows
    labels = [label for label, _ in rows]
    assert labels.count("未成交订单 (Unfilled)") == 1 and labels.count("持仓缺口 (Position gap)") == 1
    assert f"{CONTRACTS[canceled][0]}: Cancelled by IBKR" not in brief.text  # no ticker sidecar
    assert f"{canceled}: Cancelled by IBKR" in brief.text
    assert dict(rows)["决策复核 (Decision recheck)"].startswith("通过 (pass)")
    assert f"{day.bars[6].date()}" in dict(rows)["决策复核 (Decision recheck)"]

    day.broker.open_auction(8)
    day.record()
    day.decide(8)  # t = 8: the unfilled order and bar 7's gap are no longer t's
    assert briefing.close_briefing(day.config).book.events == ()


def test_a_dividend_counts_in_its_securitys_contribution(day):
    day.trade(range(FIRST, 9))  # t = bar 8, held since bar 7's open
    held = _zarr(day.live_dir / "holdings.zarr")["holding"].isel(timestamp=-1).to_series()
    permno = int(held.idxmax())
    plain = _contributions(briefing.close_briefing(day.config).book)
    recorder = LiveRecorder(day.config, day.run)
    recorder.add_event({"type": "corporate_action", "action": "DIVIDEND",
                        "timestamp": str(day.bars[8].date()), "symbol": permno,
                        "quantity": 10, "amount": 2.5, "per_share": 0.25})
    recorder.write()
    brief = briefing.close_briefing(day.config)
    before = float(read_equity(day.live_dir)["value"].values[-2])
    got = _contributions(brief.book)
    assert got[permno] == pytest.approx(plain[permno] + 2.5 / before * 100.0, rel=1e-9)
    # The account's cash never received it here, so the residual carries it back out.
    assert brief.book.cash_contribution_pct == pytest.approx(-2.5 / before * 100.0, rel=1e-6)
    assert dict(brief.rows)["公司行动 (Corporate action)"] == f"DIVIDEND {permno}: 10 股, 现金 +2.50 USD"


def test_orders_refused_at_ts_submission_are_listed(day):
    day.trade(range(FIRST, 6))
    day.broker.reject_on_submit = {10001, 10002, 10003, 10004}
    day.decide(6)
    book = briefing.close_briefing(day.config).book
    assert book.orders == 2 and len(book.refused_orders) == 2
    assert {o["status"] for o in book.refused_orders} == {"rejected"}
    rows = briefing.close_briefing(day.config).rows
    assert [label for label, _ in rows].count("被拒订单 (Rejected)") == 2
    # The records before bar 6 rechecked no decided bar; with no recheck file at all, it has not run.
    assert dict(rows)["决策复核 (Decision recheck)"].startswith("无新调仓日 (nothing new)")
    (day.live_dir / live.RECHECK_FILE).unlink()
    brief = briefing.close_briefing(day.config)
    assert brief.book.recheck is None
    assert dict(brief.rows)["决策复核 (Decision recheck)"] == "尚未运行 (not run yet)"


class _Partial(TickerLookup):
    """Names 10002 and 10003 only; any other symbol gets its own id, as TickerLookup promises."""

    def names(self, symbols, day):
        known = {s: CONTRACTS[s][0] for s in (10002, 10003)}
        return [SymbolName(known.get(s, str(s))) for s in symbols]


def test_a_security_without_a_ticker_is_shown_by_its_permno(day, monkeypatch):
    day.trade(range(FIRST, N_BARS))  # t = bar 11: holds 10003 and 10004
    dataset = type(day.run.price_dataset)
    monkeypatch.setattr(dataset, "ticker_lookup", lambda self: _Partial())
    brief = briefing.close_briefing(day.config)
    names = {f.symbol: f.ticker for f in brief.book.top_holdings}
    assert names == {10003: "CCC", 10004: "10004"}
    labels = [label for label, _ in brief.rows]
    assert "CCC" in labels and "10004" in labels
    assert "CCC" in brief.text and "10004" in brief.html

    monkeypatch.setattr(dataset, "ticker_lookup", lambda self: None)
    names = {f.symbol: f.ticker for f in briefing.close_briefing(day.config).book.top_holdings}
    assert names == {10003: "10003", 10004: "10004"}
