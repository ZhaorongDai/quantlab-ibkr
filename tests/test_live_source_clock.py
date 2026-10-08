"""The IBKR venue's live decision source and clock (#48), on fixtures without a Gateway.

A closed-loop-ready quantlab run (``build_constructor_run``, TopN, rebalancing
every 3 bars) and a live prediction store of it (quantlab #233's
``LivePredictionStore``) written here. Locked:

- the clock's cadence is the backtest's rebalance bars on the run's window,
  and keeps counting from the run's anchor past its last bar once the price
  store grows;
- the source hands t's prediction row, raw close and delisting mark, reading
  nothing after t;
- a missing prediction row, or a store row for a bar after the prices' last,
  is a hold with a reason, never a decision: the day's inputs carry no
  predictions, and the clock still fires the cycle, which only marks the
  account (#51: a row of equity and holdings per day);
- excluded symbols lose their predictions; a store of another run is refused.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.runs.live_predictions import LivePredictionStore
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.venue.ibkr.clock import LiveDecisionClock
from quantlab_ibkr.venue.ibkr.source import LiveDecisionSource
from tests.quantlab_run_fixture import build_constructor_run, write_crsp_store
from tests.test_closed_loop_replay import LABELS, PERMNOS, _market, _predictions, _topn

N_BARS = 12
EXTRA_BARS = 7
PERIODS = 3


def _run(root: Path):
    bars = pd.bdate_range("2024-01-02", periods=N_BARS)
    open_, close = _market(N_BARS, seed=1)
    run_dir, _ = build_constructor_run(
        root / "quantlab", bars, open_, close, _predictions(N_BARS, seed=2), _topn(), LABELS,
        rebalance_periods=PERIODS,
    )
    return run_dir


def _extend_prices(root: Path) -> pd.DatetimeIndex:
    """Rewrite the run's price store with ``EXTRA_BARS`` more bars, as the daily job extends it."""
    bars = pd.bdate_range("2024-01-02", periods=N_BARS + EXTRA_BARS)
    open_, close = _market(N_BARS + EXTRA_BARS, seed=1)
    write_crsp_store(root / "quantlab" / "crsp.zarr", bars, open_, close)
    return bars


def _row(t, values, permnos=PERMNOS) -> xr.Dataset:
    return xr.Dataset(
        {"ret_5": (("timestamp", "symbol"), np.array([values], dtype=float))},
        coords={"timestamp": [pd.Timestamp(t)], "symbol": np.array(permnos, dtype=np.int64)},
    )


def _store(root: Path, run_dir: Path, rows) -> LivePredictionStore:
    store = LivePredictionStore(root / "live" / "live_predictions.zarr")
    header = LivePredictionStore.header(
        LABELS, run_dir=run_dir, checkpoint=root / "model.joblib"
    )
    for t, values in rows:
        store.append(_row(t, values), header, {"data_fingerprint": {}})
    return store


@pytest.fixture
def run_dir(tmp_path):
    return _run(tmp_path)


def test_cadence_is_the_backtests_on_the_window_and_continues_past_its_end(tmp_path, run_dir):
    run = QuantlabRun.load(run_dir)
    weights = run.rebalance_table()["weight"].transpose("timestamp", "symbol").to_pandas()
    backtest_bars = list(weights.index[weights.notna().any(axis=1)])
    window = list(weights.index)
    rebalances = run.decision_inputs().rebalances
    live = [t for t in window if LiveDecisionClock(t, rebalances).decision.rebalances]
    assert live == backtest_bars == [window[k] for k in range(0, N_BARS, PERIODS)]

    bars = _extend_prices(tmp_path)
    rebalances = QuantlabRun.load(run_dir).decision_inputs().rebalances
    past_end = [
        t for t in bars[N_BARS:] if LiveDecisionClock(t, rebalances).decision.rebalances
    ]
    assert past_end == [bars[k] for k in range(N_BARS, len(bars)) if k % PERIODS == 0]
    assert len(past_end) == 3


def test_source_hands_the_last_bars_row_close_and_delisting(tmp_path, run_dir):
    bars = _extend_prices(tmp_path)
    t = bars[-1]
    store = _store(tmp_path, run_dir, [(bars[-2], [0.1, 0.2, 0.3, 0.4]), (t, [0.4, np.nan, 0.2, 0.1])])
    run = QuantlabRun.load(run_dir)
    source = LiveDecisionSource(run, store)

    assert source.t == t and source.hold_reason is None
    assert list(source.calendar()) == [t]
    assert source.symbols == (10001, 10003, 10004)
    inputs = source.inputs(t)
    assert inputs.timestamp == t
    assert inputs.predictions["ret_5"].dims == ("symbol",)
    np.testing.assert_array_equal(inputs.predictions["ret_5"].values, [0.4, np.nan, 0.2, 0.1])
    _, close = _market(N_BARS + EXTRA_BARS, seed=1)
    assert inputs.close.to_dict() == {p: close[p][-1] for p in (10001, 10003, 10004)}
    assert not inputs.delisted.any()
    with pytest.raises(ValueError, match="decides"):
        source.inputs(bars[-2])

    clock = LiveDecisionClock(source.t, run.decision_inputs().rebalances, source.hold_reason)
    assert clock.decision.decides == ((len(bars) - 1) % PERIODS == 0)


def test_source_reads_nothing_after_t(tmp_path, run_dir):
    bars = _extend_prices(tmp_path)
    t = bars[N_BARS + 2]
    store = _store(tmp_path, run_dir, [(t, [0.1, 0.2, 0.3, 0.4])])
    source = LiveDecisionSource(QuantlabRun.load(run_dir), store, t=t)
    _, close = _market(N_BARS + EXTRA_BARS, seed=1)
    assert source.inputs(t).close.to_dict() == {p: close[p][N_BARS + 2] for p in PERMNOS}
    assert source._prices.prices["timestamp"].values[-1] == np.datetime64(t)


def test_a_missing_prediction_row_is_a_hold_never_a_decision(tmp_path, run_dir):
    bars = _extend_prices(tmp_path)
    rebalance_bar = bars[N_BARS]  # 12 % 3 == 0
    store = _store(tmp_path, run_dir, [(bars[N_BARS - 1], [0.1, 0.2, 0.3, 0.4])])
    run = QuantlabRun.load(run_dir)
    source = LiveDecisionSource(run, store, t=rebalance_bar, symbols=[10002])

    assert source.hold_reason.startswith(f"no prediction row for {rebalance_bar.date()}")
    assert source.symbols == (10002,)
    clock = LiveDecisionClock(source.t, run.decision_inputs().rebalances, source.hold_reason)
    assert clock.decision.rebalances and not clock.decision.decides
    assert clock.decision.hold_reason == source.hold_reason
    inputs = source.inputs(rebalance_bar)
    assert inputs.predictions is None and list(inputs.close.index) == [10002]


def test_a_prediction_row_after_the_prices_is_a_hold(tmp_path, run_dir):
    after = pd.Timestamp("2024-01-18")  # the prices end 2024-01-17
    store = _store(tmp_path, run_dir, [(pd.Timestamp("2024-01-17"), [0.1] * 4), (after, [0.2] * 4)])
    run = QuantlabRun.load(run_dir)
    source = LiveDecisionSource(run, store)
    assert source.t == pd.Timestamp("2024-01-17")
    assert source.hold_reason == (
        "the prediction store's last row 2024-01-18 is later than the prices' last bar 2024-01-17"
    )
    clock = LiveDecisionClock(source.t, lambda t: True, source.hold_reason)
    assert not clock.decision.decides


def test_a_bar_off_the_cadence_holds_with_its_reason(tmp_path, run_dir):
    t = pd.Timestamp("2024-01-17")  # bar 11
    store = _store(tmp_path, run_dir, [(t, [0.1] * 4)])
    run = QuantlabRun.load(run_dir)
    source = LiveDecisionSource(run, store)
    clock = LiveDecisionClock(source.t, run.decision_inputs().rebalances, source.hold_reason)
    assert source.hold_reason is None
    assert clock.decision.hold_reason == "2024-01-17 is not a rebalance bar of the run's cadence"


def test_clock_fires_the_cycle_once_on_a_decided_bar_and_on_a_hold():
    t = pd.Timestamp("2024-01-18")
    alerts, decided, started = [], [], []

    class Clock:
        def utc_now(self):
            return pd.Timestamp("2024-01-19 12:00", tz="UTC")

        def set_time_alert(self, name, alert_time, callback):
            alerts.append((name, alert_time))
            callback(None)

    class Strategy:
        clock = Clock()

        def on_decision_time(self, bar):
            decided.append(bar)

    clock = LiveDecisionClock(t, lambda bar: True, on_schedule=started.append)
    clock.schedule(Strategy())
    assert decided == [t] and alerts == [("decision-2024-01-18", Clock().utc_now())]
    assert clock.done and clock.error is None and len(started) == 1

    # A hold still runs the cycle once (marking the account); it decides nothing.
    held = LiveDecisionClock(t, lambda bar: True, "no prediction row")
    assert not held.decision.decides and not held.done
    held.schedule(Strategy())
    assert decided == [t, t] and held.done and held.fired


def test_a_decision_error_is_kept_for_the_venue():
    class Strategy:
        class clock:
            @staticmethod
            def utc_now():
                return pd.Timestamp("2024-01-19 12:00", tz="UTC")

            @staticmethod
            def set_time_alert(name, alert_time, callback):
                callback(None)

        @staticmethod
        def on_decision_time(t):
            raise ValueError("boom")

    clock = LiveDecisionClock(pd.Timestamp("2024-01-18"), lambda bar: True)
    clock.schedule(Strategy())
    assert isinstance(clock.error, ValueError) and clock.done


def test_excluded_symbols_lose_their_predictions(tmp_path, run_dir):
    store = _store(tmp_path, run_dir, [(pd.Timestamp("2024-01-17"), [0.1, 0.2, 0.3, 0.4])])
    source = LiveDecisionSource(QuantlabRun.load(run_dir), store)
    source.exclude([10002, 10004])
    values = source.inputs(source.t).predictions["ret_5"].values
    np.testing.assert_array_equal(values, [0.1, np.nan, 0.3, np.nan])


def test_a_store_of_another_run_is_refused(tmp_path, run_dir):
    other = _run(tmp_path / "other")
    store = _store(tmp_path, other, [(pd.Timestamp("2024-01-17"), [0.1] * 4)])
    with pytest.raises(ValueError, match="holds the live predictions of"):
        LiveDecisionSource(QuantlabRun.load(run_dir), store)


def test_a_forced_clock_decides_a_bar_off_the_cadence_but_still_holds_without_a_row():
    t = pd.Timestamp("2026-10-07")
    forced = LiveDecisionClock(t, lambda bar: False, force=True).decision
    assert forced.decides and forced.rebalances
    held = LiveDecisionClock(t, lambda bar: False, "no prediction row", force=True).decision
    assert not held.decides and held.hold_reason == "no prediction row"
