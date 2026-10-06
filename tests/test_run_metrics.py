"""metrics.json, report.html and tracking of a trader run, at the ``runner.run`` seam.

The open-loop fixture is ``test_open_loop_replay``'s, whose orders and equity
are worked by hand there:

- fills at bar 1's open: BUY 500 @ 10.5 (fee 5.25), BUY 250 @ 20.2 (5.05);
  cash -310.30;
- fills at bar 3's open: SELL 500 @ 12.4 (6.20), BUY 258 @ 22.6 (5.83);
- the SELL decided on bar 5 has no next open (not a rejected order);
- equity 10 000, 10 439.70, 11 189.70, 11 730.87, 12 238.87, 12 746.87.

Everything below follows from those numbers.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import LedoitWolfEstimatorConfig, MeanVarianceConfig, TopNConfig
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.tracking.base import Tracker, TrackingRun
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab_trader.base.config import TraderConfig
from quantlab_trader.runner import run
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
from tests.quantlab_run_fixture import build_constructor_run, build_quantlab_run
from tests.test_open_loop_replay import BARS, CLOSE, OPEN, WEIGHTS

#: Every run a RecordingTracker opened, in order.
RUNS: list["RecordingRun"] = []


class RecordingRun(TrackingRun):
    """A tracking run that keeps what it is sent."""

    def __init__(self, project, group, name, config):
        self.project, self.group, self.name, self.config = project, group, name, config
        self.summary: dict = {}
        self.files: list = []
        self.failed = None

    def _log(self, metrics, step):
        pass

    def _summarize(self, metrics):
        self.summary.update(metrics)

    def _update_config(self, params):
        self.config.update(params)

    def _log_table(self, name, columns, rows, top_bars):
        pass

    def _log_file(self, path):
        self.files.append(path)

    def _finish(self, *, failed):
        self.failed = failed


@dataclass(frozen=True, kw_only=True)
class RecordingTracker(Tracker):
    """A tracker whose runs land in ``RUNS``."""

    def _open(self, *, project, group, name, config):
        RUNS.append(RecordingRun(project, group, name, config))
        return RUNS[-1]


def _metrics(run_dir) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def _open_replay(root, *, tracker=None, quantlab_tracker=None, **fields):
    quantlab_run = build_quantlab_run(
        root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        **({} if quantlab_tracker is None else {"tracker": quantlab_tracker}),
    )
    config = TraderConfig(
        quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
        output_dir=str(root / "trader"), tracker=tracker, **fields,
    )
    return quantlab_run, run(config)


@pytest.fixture(scope="module")
def open_replay(tmp_path_factory):
    return _open_replay(tmp_path_factory.mktemp("metrics_open"))


def test_whole_block_has_quantlab_names_and_the_hand_worked_values(open_replay):
    _, run_dir = open_replay

    whole = _metrics(run_dir)["whole"]

    assert whole["Start"] == "2024-01-02T00:00:00"
    assert whole["End"] == "2024-01-09T00:00:00"
    assert whole["Period"] == "6 days 00:00:00"
    assert whole["Start Value"] == 10_000.0
    assert whole["End Value"] == pytest.approx(12_746.87, abs=1e-9)
    assert whole["Total Return [%]"] == pytest.approx(27.4687, abs=1e-9)
    assert whole["Total Orders"] == 4
    assert whole["Total Fees Paid"] == pytest.approx(22.33, abs=1e-9)
    # quantlab's whole block has no Traded Notional; its slices do.
    assert "Traded Notional" not in whole
    turnovers = [10_300 / 10_000, (6200 + 5830.8) / 11_189.70]
    assert whole["Total Turnover [%]"] == pytest.approx(100 * sum(turnovers), abs=1e-9)
    assert whole["Turnover per Rebalance [%]"] == pytest.approx(50 * sum(turnovers), abs=1e-9)
    # Annualized: 252 bars a year, one rebalance every bar.
    assert whole["Annualized Turnover [%]"] == pytest.approx(
        50 * sum(turnovers) * 252, abs=1e-6
    )
    for name in ("Sharpe Ratio", "Max Drawdown [%]", "Sortino Ratio", "Rebalance Win Rate [%]"):
        assert name in whole
    # A whole-share book bought from a full account on bar 1 goes up every bar.
    assert whole["Max Drawdown [%]"] is None


def test_execution_block_from_the_hand_ledger(open_replay):
    _, run_dir = open_replay

    execution = _metrics(run_dir)["execution"]

    # The sale decided on the last bar has no fill bar: not a rejected order.
    assert (execution["rejected_order_count"], execution["rejected_orders"]) == (0, [])
    assert execution["settlements"] == []
    # Bar 2's target 1.0 for 10002 ends at 508 shares: 508 * 22 / 11 189.70.
    assert execution["max_target_deviation"] == pytest.approx(13.7 / 11_189.70, rel=1e-12)
    trader = execution["trader"]
    assert trader["commissions"] == pytest.approx(22.33)
    assert trader["minimum_fee_hits"] == 0
    assert trader["peak_cash_debit"] == pytest.approx(310.30)


def test_a_run_without_a_model_has_whole_metrics_only(open_replay):
    _, run_dir = open_replay

    metrics = _metrics(run_dir)

    assert sorted(metrics) == ["execution", "notes", "whole"]


def test_the_report_is_written(open_replay):
    _, run_dir = open_replay

    page = (run_dir / "report.html").read_text()

    assert run_dir.name in page
    assert "Total Orders" in page
    assert "Execution (event-driven)" in page


def _write_split(quantlab_run, **split):
    path = quantlab_run / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics.update(split)
    path.write_text(json.dumps(metrics))


SPLIT = dict(
    training_window=["2023-01-03", "2024-01-03"],
    in_sample_range=["2024-01-02", "2024-01-04"],
    out_of_sample_ranges=[["2024-01-05", "2024-01-09"]],
)


def test_in_and_out_of_sample_use_the_quantlab_run_ranges(tmp_path):
    quantlab_run = build_quantlab_run(tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS)
    _write_split(quantlab_run, **SPLIT)

    metrics = _metrics(
        run(TraderConfig(quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(),
                         loop="open", output_dir=str(tmp_path / "trader")))
    )

    for key, value in SPLIT.items():
        assert metrics[key] == value
    in_sample, out_of_sample = metrics["in_sample"], metrics["out_of_sample"]
    assert in_sample["Total Return [%]"] == pytest.approx(11.897, abs=1e-9)
    assert out_of_sample["Total Return [%]"] == pytest.approx(
        (12_746.87 / 11_189.70 - 1) * 100, abs=1e-9
    )
    assert (in_sample["Total Orders"], out_of_sample["Total Orders"]) == (2, 2)
    assert in_sample["Total Fees Paid"] == pytest.approx(10.30)
    assert in_sample["Traded Notional"] == pytest.approx(5250 + 5050, abs=1e-9)
    assert out_of_sample["Total Fees Paid"] == pytest.approx(12.03)
    assert in_sample["Total Turnover [%]"] == pytest.approx(103.0)
    # 10001 is sold on bar 3, 10002 is still held.
    assert (out_of_sample["Total Closed Trades"], out_of_sample["Total Open Trades"]) == (1, 1)


def test_a_narrowed_window_cuts_the_ranges(tmp_path):
    quantlab_run = build_quantlab_run(tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS)
    _write_split(quantlab_run, **SPLIT)

    metrics = _metrics(
        run(TraderConfig(quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(),
                         loop="open", output_dir=str(tmp_path / "trader"),
                         start="2024-01-04", end="2024-01-08"))
    )

    assert metrics["in_sample_range"] == ["2024-01-04", "2024-01-04"]
    assert metrics["out_of_sample_ranges"] == [["2024-01-05", "2024-01-08"]]
    assert metrics["in_sample"]["Period"] == "1 days 00:00:00"


BENCHMARK = ({90000: [99.0, 100.0, 101.0, 103.0, 102.0, 104.0]},
             {90000: [100.0, 101.0, 102.0, 102.0, 103.0, 105.0]})


def test_benchmark_and_relative_follow_the_quantlab_run_benchmark(tmp_path):
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS, benchmark=BENCHMARK
    )
    quantlab_metrics = json.loads((quantlab_run / "metrics.json").read_text())

    metrics = _metrics(
        run(TraderConfig(quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(),
                         loop="open", output_dir=str(tmp_path / "trader")))
    )

    benchmark = metrics["benchmark"]
    assert benchmark["axis_symbol"] == quantlab_metrics["benchmark"]["axis_symbol"]
    assert benchmark["whole"]["Total Return [%]"] == pytest.approx(
        quantlab_metrics["benchmark"]["whole"]["Total Return [%]"], rel=1e-12
    )
    relative = metrics["relative"]["whole"]
    assert relative["Strategy Total Return [%]"] == pytest.approx(27.4687, abs=1e-9)
    assert relative["Total Return Difference [%]"] == pytest.approx(
        27.4687 - quantlab_metrics["benchmark"]["whole"]["Total Return [%]"], abs=1e-9
    )
    assert relative["Bars"] == 6
    assert "Rebalance Win Rate vs Benchmark [%]" in relative


def test_metrics_go_to_the_configured_tracker(tmp_path):
    _, run_dir = _open_replay(tmp_path, tracker=RecordingTracker(project="trader_tests"))

    tracked = RUNS[-1]
    assert (tracked.project, tracked.name) == ("trader_tests", run_dir.name)
    assert tracked.summary["whole/Total Orders"] == 4
    assert tracked.summary["whole/Total Fees Paid"] == pytest.approx(22.33)
    assert tracked.summary["execution/trader/peak_cash_debit"] == pytest.approx(310.30)
    assert tracked.files == [run_dir / "report.html"]
    # The run config gains the records config.json resolved.
    assert "quantlab_data_fingerprint" in tracked.config
    assert tracked.failed is False


def test_metrics_go_to_the_quantlab_run_tracker_by_default(tmp_path):
    _, run_dir = _open_replay(tmp_path, quantlab_tracker=RecordingTracker())

    quantlab_tracked, tracked = RUNS[-2:]
    assert tracked.name == run_dir.name
    assert tracked.project == quantlab_tracked.project == "WeightsVectorBt_backtest"
    assert tracked.summary["whole/Total Orders"] == 4


def test_closed_loop_records_rule_events_and_minimum_fees(tmp_path):
    # Three bars at 50; rebalance bars 0 and 2, and the last never decides.
    # Bar 0: 10002 and 10003 tie at the cut, 10002 wins on symbol order (one
    # tied symbol left out); 0.5 of 10 000 buys 100 shares at 50, whose
    # USD 0.50 per-share charge is lifted to the USD 1.00 minimum.
    bars = pd.bdate_range("2024-01-02", periods=3)
    prices = {p: [50.0] * 3 for p in (10001, 10002, 10003, 10004)}
    scores = {10001: 0.03, 10002: 0.02, 10003: 0.02, 10004: 0.01}
    quantlab_run, _ = build_constructor_run(
        tmp_path / "quantlab", bars, prices, prices,
        {"ret_5": {p: [s] * 3 for p, s in scores.items()}},
        TopNConstructor(TopNConfig(direction="long_only", top_n=2)),
        [LabelSpec("ret_5", "raw", 1, 5)], rebalance_periods=2,
    )

    metrics = _metrics(
        run(TraderConfig(quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(),
                         output_dir=str(tmp_path / "trader")))
    )

    assert metrics["portfolio_construction"] == {
        "failed_bar_count": 0,
        "failed_bars": [],
        "tie_at_cutoff": {"count": 1, "bars": [{"bar": "2024-01-02T00:00:00", "count": 1}]},
    }
    trader = metrics["execution"]["trader"]
    assert (trader["minimum_fee_hits"], trader["commissions"]) == (2, 2.0)
    assert metrics["execution"]["max_target_deviation"] == 0.0
    assert metrics["whole"]["End Value"] == pytest.approx(9_998.0)


def test_closed_loop_held_bars_are_the_failed_bars(tmp_path):
    # Four symbols under a 0.2 cap cannot hold a fully invested book.
    bars = pd.bdate_range("2024-01-02", periods=10)
    rng = np.random.default_rng(1)
    prices = {p: list(np.round(30 * np.exp(np.cumsum(rng.normal(0, 0.02, 10))), 2))
              for p in (10001, 10002, 10003, 10004)}
    rule = MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            covariance=LedoitWolfEstimator(LedoitWolfEstimatorConfig(lookback_bars=3)),
            risk_aversion=5.0, ic=0.05, weight_cap=0.2,
        )
    )
    quantlab_run, failed = build_constructor_run(
        tmp_path / "quantlab", bars, prices, prices,
        {"ret_5": {p: list(rng.normal(0, 0.01, 6)) for p in prices}},
        rule, [LabelSpec("ret_5", "raw", 1, 5)], first_bar=4, rebalance_periods=2,
    )

    metrics = _metrics(
        run(TraderConfig(quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(),
                         output_dir=str(tmp_path / "trader")))
    )

    block = metrics["portfolio_construction"]
    assert block["failed_bars"] == failed
    assert block["failed_bar_count"] == 3
    assert metrics["execution"]["max_target_deviation"] is None
    assert metrics["whole"]["Total Orders"] == 0


def test_a_narrowed_window_measures_the_benchmark_from_its_first_close(tmp_path):
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS, benchmark=BENCHMARK
    )
    with xr.open_zarr(quantlab_run / "equity.zarr") as equity:
        value = equity["benchmark_value"].values

    run_dir = run(TraderConfig(quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(),
                               loop="open", output_dir=str(tmp_path / "trader"),
                               start="2024-01-03"))

    # Both books start at the close of the window's first bar.
    expected = (value[-1] / value[1] - 1) * 100
    assert _metrics(run_dir)["benchmark"]["whole"]["Total Return [%]"] == pytest.approx(
        expected, rel=1e-12
    )
    assert _metrics(run_dir)["relative"]["whole"]["Benchmark Total Return [%]"] == pytest.approx(
        expected, rel=1e-12
    )
