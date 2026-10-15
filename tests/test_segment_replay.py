"""The close briefing's closed-loop replay of the current strategy segment (#67).

On ``tests/test_live_run.py``'s ``Day`` harness, with a broker charging IBKR
Pro Fixed (the closed-loop backtest's default fee model). Locked:

- filled at the opening print at the backtest's fee, the replay's curve
  equals the live curve and the return gap and tracking error are zero;
- a fill worse than the opening print gives a negative gap of the cost;
- the replay decides from the live prediction store, not the quantlab run's
  own panel;
- rerunning the briefing replaces the replay's directory and leaves the live
  run directory's own files unchanged;
- a segment that did not start from an empty account is stated;
- the live side counts the unfilled orders and fees of the orders decided
  before t only, as the replay (which never decides t) does: t's orders fill
  at t+1's open, after the close the briefing reports (#73).
"""

import json
import logging
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from nautilus_trader.model.enums import OrderSide

from quantlab_ibkr import briefing, cli, segment_replay
from quantlab_ibkr.account_history import AccountHistory, StrategySegment
from quantlab_ibkr.outputs import read_equity
from quantlab_ibkr.venue.backtest.fees import IbkrFixedFeeModel
from tests.test_close_briefing import BENCHMARK, _snapshot
from tests.test_live_run import FIRST, Broker, Day, _backtest
from tests.test_closed_loop_replay import _market, _predictions
from tests.test_live_source_clock import EXTRA_BARS, N_BARS

#: The first bar the live days submit orders on: the segment's first decision date.
REBALANCE = 6


class IbkrFixedBroker(Broker):
    """The fake account charging IBKR Pro Fixed, as the closed-loop backtest does."""

    def fee(self, side, quantity, price):
        return float(IbkrFixedFeeModel.charge(OrderSide[side], quantity, Decimal(str(price))))


class SlippedBroker(IbkrFixedBroker):
    """Fills every order ``SLIP`` dollars worse than the opening print."""

    SLIP = 0.05

    def fill_price(self, order, k):
        open_ = super().fill_price(order, k)
        return round(open_ + (self.SLIP if order.side == "BUY" else -self.SLIP), 2)


def _day(tmp_path, broker=IbkrFixedBroker) -> Day:
    day = Day(tmp_path, benchmark=BENCHMARK)
    day.broker = broker(day.run.init_cash, day.open, day.bars, day.run.fees)
    return day


def _zarr(path):
    with xr.open_zarr(path) as dataset:
        return dataset.load()


def test_the_replay_equals_live_when_filled_at_the_open_at_the_backtests_fee(tmp_path, caplog):
    day = _day(tmp_path)
    day.trade(range(FIRST, N_BARS))
    with caplog.at_level(logging.INFO, logger=segment_replay.__name__):
        brief = briefing.close_briefing(day.config)
    replay = brief.replay
    assert brief.replay_failure is None
    assert replay.start == day.bars[REBALANCE] and replay.end == brief.t
    assert replay.empty_start and replay.held_at_start == 0

    live = read_equity(day.live_dir)["value"].sel(timestamp=slice(replay.start, None))
    assert replay.init_cash == float(live.values[0])
    xr.testing.assert_allclose(replay.curve["live_value"], live.rename("live_value"), rtol=1e-12)
    np.testing.assert_allclose(
        replay.curve["replay_value"].values, live.values, rtol=1e-9
    )
    assert replay.gap_pct == pytest.approx(0.0, abs=1e-7)
    assert replay.tracking_error_pct == pytest.approx(0.0, abs=1e-7)
    assert replay.unfilled_live == replay.unfilled_replay == 0
    assert replay.fees_live > 0 and replay.fees_replay == pytest.approx(replay.fees_live, abs=1e-9)

    # The chart's replay curve is on the live curve's axis, from the segment's start.
    cumulative = brief.curve["cumulative_return"].sel(timestamp=slice(replay.start, None))
    np.testing.assert_allclose(
        brief.curve["replay_cumulative_return"].sel(timestamp=slice(replay.start, None)).values,
        cumulative.values, rtol=1e-9, atol=1e-12,
    )
    assert np.isnan(brief.curve["replay_cumulative_return"].sel(timestamp=day.bars[FIRST]))

    run_dir = day.live_dir / segment_replay.BRIEFING_DIR / segment_replay.REPLAY_DIR
    assert replay.run_dir == run_dir and (run_dir / "equity.zarr").is_dir()
    record = json.loads((run_dir / segment_replay.RECORD_FILE).read_text())
    assert record["prediction_store"] == str(day.store.path)
    assert record["start"] == str(replay.start.date()) and record["seconds"] == replay.seconds
    assert replay.seconds > 0
    assert any("replay" in r.getMessage() and " s" in r.getMessage() for r in caplog.records)

    rows = dict(brief.rows)
    assert rows["累计收益差 (Return Gap)"] == "+0.00 pp"
    assert rows["未成交订单 (Unfilled Orders)"] == "实盘 0 / 回放 0"
    assert "空账户" in rows["起点 (Start)"]
    for label, text in brief.rows:
        assert f"{label}: {text}" in brief.text
    print(f"replay wall time: {replay.seconds:.2f} s")


def test_the_replay_runs_on_live_days_past_the_quantlab_runs_window(tmp_path):
    day = _day(tmp_path)
    n = N_BARS + EXTRA_BARS  # the daily jobs extend the prices and the live store past the run
    day.bars = pd.bdate_range("2024-01-02", periods=n)
    day.open, day.close = _market(n, seed=1)
    day.panel = _predictions(n, seed=2)["ret_5"]
    day.broker = IbkrFixedBroker(day.run.init_cash, day.open, day.bars, day.run.fees)
    for k in range(FIRST, n):
        day.decide(k)
        if k + 1 < n:
            day.broker.open_auction(k + 1)
            day.record()
    brief = briefing.close_briefing(day.config)
    assert brief.replay_failure is None and brief.t == day.bars[n - 1] > day.run.window[1]
    replay = brief.replay
    # Past the run's window the backtest venue's book drifts from live by a fraction of a
    # cent once (0.0025 USD on one fill bar, measured; orders, fills and fees identical).
    np.testing.assert_allclose(replay.curve["replay_value"], replay.curve["live_value"], atol=0.01)
    assert replay.gap_pct == pytest.approx(0.0, abs=1e-4)
    assert (n - 1) % 3 == 0 and replay.unfilled_replay == replay.unfilled_live == 0  # t rebalances
    decided = _zarr(day.live_dir / "decisions.zarr")["timestamp"].values
    assert decided[-1] > np.datetime64(day.run.window[1])  # a rebalance past the run's last bar
    # t decides live; the replay's window ends at t, whose decision moves nothing before t+1.
    live = _zarr(day.live_dir / "decisions.zarr")["weight"]
    assert live["timestamp"].values[-1] == np.datetime64(brief.t)
    xr.testing.assert_allclose(
        _zarr(replay.run_dir / "decisions.zarr")["weight"], live.isel(timestamp=slice(None, -1))
    )


def test_a_fill_worse_than_the_open_gives_a_negative_gap_of_its_cost(tmp_path):
    day = _day(tmp_path, SlippedBroker)
    day.trade(range(FIRST, REBALANCE + 3))  # bar 6's orders fill at bar 7; t = 8
    brief = briefing.close_briefing(day.config)
    replay = brief.replay

    orders = _zarr(day.live_dir / "orders.zarr")
    assert set(orders["side"].values) == {"BUY"} and set(orders["status"].values) == {"filled"}
    slip = SlippedBroker.SLIP * orders["filled_quantity"].values.sum()
    extra_fees = replay.fees_live - replay.fees_replay
    live_value, replay_value = (replay.curve[v].values[-1] for v in ("live_value", "replay_value"))
    assert live_value - replay_value == pytest.approx(-slip - extra_fees, abs=1e-6)
    assert replay.gap_pct == pytest.approx((live_value - replay_value) / replay.init_cash * 100.0)
    assert replay.gap_pct < 0 and replay.tracking_error_pct > 0
    assert dict(brief.rows)["累计收益差 (Return Gap)"] == f"{replay.gap_pct:+.2f} pp"


def test_the_replay_decides_from_the_live_prediction_store(tmp_path):
    day = _day(tmp_path)
    # Live predicts the opposite of the run's own panel.
    day.panel = {p: [-x for x in values] for p, values in day.panel.items()}
    day.trade(range(FIRST, N_BARS))
    replay = briefing.close_briefing(day.config).replay

    live = _zarr(day.live_dir / "decisions.zarr")["weight"]
    replayed = _zarr(replay.run_dir / "decisions.zarr")["weight"]
    xr.testing.assert_allclose(replayed, live)
    np.testing.assert_allclose(replay.curve["replay_value"], replay.curve["live_value"], rtol=1e-9)

    own_panel = _zarr(_backtest(day, REBALANCE, N_BARS - 1) / "decisions.zarr")["weight"]
    assert not own_panel.fillna(0).equals(live.reindex_like(own_panel).fillna(0))


def test_rerunning_the_briefing_replaces_the_replay_and_leaves_the_live_files(tmp_path):
    day = _day(tmp_path)
    day.trade(range(FIRST, 8))
    before = _snapshot(day.live_dir)
    first = briefing.close_briefing(day.config).replay
    (first.run_dir / "stale.txt").write_text("from the earlier briefing")
    second = briefing.close_briefing(day.config).replay
    assert second.run_dir == first.run_dir and not (second.run_dir / "stale.txt").exists()
    assert sorted(p.name for p in (day.live_dir / segment_replay.BRIEFING_DIR).iterdir()) == [
        segment_replay.REPLAY_DIR
    ]
    assert _snapshot(day.live_dir) == before


def test_live_brief_states_the_replays_wall_time(tmp_path, capsys):
    day = _day(tmp_path)
    day.trade(range(FIRST, 8))
    path = day.root / "live.json"
    path.write_text(json.dumps(day.config.get_config()))
    assert cli.main(["live", "brief", str(path), "--no-send", "--output", str(day.root / "b")]) == 0
    assert re.search(r"closed-loop replay: \d+\.\d s", capsys.readouterr().err)


def test_a_segment_not_starting_from_an_empty_account_is_stated(tmp_path):
    day = _day(tmp_path)
    day.trade(range(FIRST, N_BARS))
    # The segment entered as beginning at bar 8, holding what bar 6's decision bought.
    history = AccountHistory.of(day.config.venue)
    start = day.bars[8]
    history.write([StrategySegment(str(day.live_dir.resolve()), "live", start)])
    brief = briefing.close_briefing(day.config)
    replay = brief.replay
    assert replay.start == start and not replay.empty_start and replay.held_at_start == 2
    live = read_equity(day.live_dir)["value"].sel(timestamp=start)
    assert replay.init_cash == float(live)
    assert float(replay.curve["replay_value"].values[0]) == pytest.approx(replay.init_cash)
    text = dict(brief.rows)["起点 (Start)"]
    assert "账户非空" in text and "2 只" in text and f"{replay.init_cash:,.2f}" in text
    assert "账户非空" in brief.html and "账户非空" in brief.text


def test_a_failed_replay_is_stated_and_the_briefing_still_builds(tmp_path, monkeypatch):
    day = _day(tmp_path)
    day.trade(range(FIRST, 8))
    briefing.close_briefing(day.config)  # an earlier day's replay, now stale

    def refuse(*args, **kwargs):
        raise ValueError("no prediction row")

    monkeypatch.setattr(segment_replay, "run_backtest", refuse)
    brief = briefing.close_briefing(day.config)
    assert brief.replay is None and "no prediction row" in brief.replay_failure
    assert "no prediction row" in dict(brief.rows)["回放 (Replay)"]
    assert "replay_cumulative_return" not in brief.curve
    assert not (day.live_dir / segment_replay.BRIEFING_DIR / segment_replay.REPLAY_DIR).exists()


def test_the_live_side_leaves_out_the_orders_decided_on_t(tmp_path):
    # Bar 9 rebalances: its orders fill at bar 10's open and are recorded, but
    # the last marked bar is still 9, so the briefing of t = 9 must not count them.
    day = _day(tmp_path)
    day.trade(range(FIRST, 10))
    t = day.bars[9]
    orders = _zarr(day.live_dir / "orders.zarr")
    decided = pd.DatetimeIndex(orders["decision_date"].values)
    assert (decided == t).any() and np.nansum(orders["fee"].values[decided == t]) > 0
    before = float(np.nansum(orders["fee"].values[decided < t]))

    replay = briefing.close_briefing(day.config).replay
    assert replay.end == t
    assert replay.fees_live == pytest.approx(before, abs=1e-9)
    assert replay.fees_live == pytest.approx(replay.fees_replay, abs=1e-9)
    assert replay.unfilled_live == replay.unfilled_replay == 0
