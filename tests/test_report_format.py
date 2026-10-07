"""report.html inputs and metrics.json of a trader run in quantlab's format (#33).

At the ``runner.run`` seam: the open-loop replay of a fixture quantlab run
(``test_open_loop_replay``'s, hand-worked there) is compared with the
quantlab run itself. quantlab's ``write_backtest_report`` is wrapped on both
sides so the arguments each run hands the page can be compared where their
meanings coincide.
"""

import json

import numpy as np
import pandas as pd
import xarray as xr

import pytest

import quantlab.backtest.base as quantlab_backtest
import quantlab_trader.outputs as trader_outputs
from quantlab_trader.base.config import TraderConfig
from quantlab_trader.runner import run
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
from tests.quantlab_run_fixture import build_quantlab_run
from tests.test_open_loop_replay import BARS, CLOSE, OPEN, WEIGHTS

BENCHMARK = ({90000: [99.0, 100.0, 101.0, 103.0, 102.0, 104.0]},
             {90000: [100.0, 101.0, 102.0, 102.0, 103.0, 105.0]})


def _capture(monkeypatch, module) -> list[dict]:
    """Record every call of ``module.write_backtest_report``, then make it."""
    calls: list[dict] = []
    original = module.write_backtest_report

    def recording(value, path, **kwargs):
        calls.append({"value": value, **kwargs})
        return original(value, path, **kwargs)

    monkeypatch.setattr(module, "write_backtest_report", recording)
    return calls


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    root = tmp_path_factory.mktemp("report_format")
    with pytest.MonkeyPatch.context() as patch:
        quantlab_calls = _capture(patch, quantlab_backtest)
        quantlab_run = build_quantlab_run(
            root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS, benchmark=BENCHMARK
        )
        trader_calls = _capture(patch, trader_outputs)
        run_dir = run(TraderConfig(quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(),
                                   loop="open", output_dir=str(root / "trader")))
    return dict(
        quantlab_run=quantlab_run, run_dir=run_dir,
        quantlab=quantlab_calls[-1], trader=trader_calls[-1],
    )


def _metrics(run_dir) -> dict:
    return json.loads((run_dir / "metrics.json").read_text())


def test_whole_has_quantlabs_rows_in_quantlabs_order(replay):
    quantlab = _metrics(replay["quantlab_run"])["whole"]
    trader = _metrics(replay["run_dir"])["whole"]

    assert list(trader) == list(quantlab)
    for name in ("Start", "End", "Period", "Start Value"):
        assert trader[name] == quantlab[name]


def test_max_gross_exposure_is_the_largest_book_over_its_value(replay):
    # Bar 1's close: 500 * 11 + 250 * 21 = 10 750 held on cash -310.30.
    whole = _metrics(replay["run_dir"])["whole"]

    assert whole["Max Gross Exposure [%]"] == pytest.approx(100 * 10_750 / 10_439.70, rel=1e-12)


def test_setup_has_quantlabs_lines_then_the_trader_lines(replay):
    quantlab, trader = replay["quantlab"]["summary"], replay["trader"]["summary"]

    assert list(trader)[: len(quantlab)] == list(quantlab)
    assert list(trader)[len(quantlab):] == ["Quantlab run", "Loop", "Slippage", "Init cash"]
    for label in quantlab:
        if label not in ("Fees", "Deepest drawdown (valley to recovery)"):
            assert trader[label] == quantlab[label], label
    assert trader["Fees"] == "0.001 of the traded notional"
    assert trader["Quantlab run"] == str(replay["quantlab_run"])
    assert (trader["Loop"], trader["Slippage"], trader["Init cash"]) == ("open loop", "0.0", "10,000.00")


def test_timeline_equals_quantlabs(replay):
    assert replay["trader"]["windows"] == replay["quantlab"]["windows"]


def test_chart_inputs_name_and_draw_the_benchmark_as_quantlab_does(replay):
    quantlab, trader = replay["quantlab"], replay["trader"]

    # trader simulates no cost-free or risk-model counterfactual, so it hands
    # the page no attribution; quantlab hands its own as None without a model.
    assert set(trader) - {"extra_tables"} == set(quantlab) - {"attribution", "factor_attribution"}
    for name in ("in_sample_range", "init_cash", "benchmark_name", "bars_per_year"):
        assert trader[name] == quantlab[name], name
    np.testing.assert_allclose(
        trader["benchmark_value"].values, quantlab["benchmark_value"].values, rtol=1e-12
    )


def test_portfolio_tab_draws_the_decided_targets(replay):
    decided = xr.open_zarr(replay["run_dir"] / "decisions.zarr").load()["weight"]

    xr.testing.assert_identical(replay["trader"]["weights"], decided)


def test_holdings_tab_shows_the_actual_holdings_at_each_close(replay):
    holdings = replay["trader"]["holdings"].transpose("timestamp", "symbol").to_pandas()

    # Bar 1: 500 * 11 and 250 * 21 over 10 439.70; bar 3 on: 508 of 10002.
    equity = [10_000.0, 10_439.70, 11_189.70, 11_730.87, 12_238.87, 12_746.87]
    expected = pd.DataFrame(
        {
            10001: [0.0, 5_500 / equity[1], 6_000 / equity[2], 0.0, 0.0, 0.0],
            10002: [0.0, 5_250 / equity[1], 5_500 / equity[2]]
            + [508 * c / e for c, e in zip((23, 24, 25), equity[3:])],
        },
        index=BARS.astype("datetime64[ns]"),
    )
    pd.testing.assert_frame_equal(
        holdings[[10001, 10002]], expected, check_names=False, check_freq=False, atol=1e-9
    )
    stored = xr.open_zarr(replay["run_dir"] / "holdings.zarr").load()["holding"]
    np.testing.assert_allclose(stored.values, replay["trader"]["holdings"].values)
    # The fixture's price dataset names no ticker lookup: each symbol is its id.
    assert replay["trader"]["holding_names"]["10002"] == [("2024-01-03", "10002", "")]
    turnover = replay["trader"]["turnover"]
    np.testing.assert_allclose(
        turnover.values, [10_300 / 10_000, (6_200 + 5_830.8) / 11_189.70], rtol=1e-12
    )


def test_trader_only_execution_facts_are_their_own_table(replay):
    table = replay["trader"]["extra_tables"]["Execution (event-driven)"]

    assert table["Commissions"] == pytest.approx(22.33)
    assert table["Minimum-fee hits"] == 0
    assert table["Peak cash debit"] == pytest.approx(310.30)
    assert {"Dividends", "Value distributions", "Cash in lieu", "Splits",
            "Implied splits", "Factor mismatches"} <= set(table)


def _as_run_cv(quantlab_run):
    """Rewrite a run's metrics.json in a ``run_cv()`` run's layout: two folds, stitched."""
    path = quantlab_run / "metrics.json"
    metrics = json.loads(path.read_text())
    folds = [
        (0, ["2023-01-03", "2024-01-02"], "2024-01-02", "2024-01-04"),
        (1, ["2023-01-03", "2024-01-05"], "2024-01-05", "2024-01-09"),
    ]
    path.write_text(json.dumps({
        "stitched": {
            "whole": metrics["whole"],
            "training_windows": [training for _, training, _, _ in folds],
            "in_sample_ranges": [[start, start] for _, _, start, _ in folds],
            "out_of_sample_ranges": [["2024-01-03", "2024-01-04"], ["2024-01-08", "2024-01-09"]],
        },
        "folds": [
            {
                "fold": fold,
                "metrics": {
                    "training_window": training,
                    "in_sample_range": [start, start],
                    "whole": {"Start": f"{start}T00:00:00", "End": f"{end}T00:00:00"},
                },
            }
            for fold, training, start, end in folds
        ],
        "notes": [],
    }))


def test_timeline_has_the_run_cv_folds_cut_to_the_trader_window(tmp_path, monkeypatch):
    quantlab_run = build_quantlab_run(tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS)
    _as_run_cv(quantlab_run)
    calls = _capture(monkeypatch, trader_outputs)

    run(TraderConfig(quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
                     output_dir=str(tmp_path / "trader"), start="2024-01-03"))

    windows = calls[-1]["windows"]
    assert windows["backtest"] == ("2024-01-03", "2024-01-09")
    assert windows["in_sample"] == [["2024-01-05", "2024-01-05"]]
    assert windows["folds"] == [
        {"label": "fold 0", "training": ["2023-01-03", "2024-01-02"],
         "traded": ("2024-01-03", "2024-01-04"), "in_sample": None},
        {"label": "fold 1", "training": ["2023-01-03", "2024-01-05"],
         "traded": ("2024-01-05", "2024-01-09"), "in_sample": ["2024-01-05", "2024-01-05"]},
    ]
