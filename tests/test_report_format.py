"""report.html inputs and metrics.json of a trader run in quantlab's format (#33).

At the ``runner.run`` seam: the open-loop replay of a fixture quantlab run
(``test_open_loop_replay``'s, hand-worked there) is compared with the
quantlab run itself. quantlab's ``write_backtest_report`` is wrapped on both
sides so the record each run hands the page (account record, metrics,
weights, config, benchmark record, folds, extras) can be compared where
their meanings coincide, and the two pages are compared where they draw the
same thing (the Setup lines, the timeline, the benchmark curve).
"""

import html as html_module
import json
import re

import numpy as np
import pandas as pd
import xarray as xr

import pytest

import quantlab.backtest.base as quantlab_backtest
import quantlab_ibkr.outputs as trader_outputs
from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.runner import run
from quantlab_ibkr.venue.backtest.venue import BacktestVenueConfig
from tests.quantlab_run_fixture import build_quantlab_run
from tests.test_open_loop_replay import BARS, CLOSE, OPEN, WEIGHTS

BENCHMARK = ({90000: [99.0, 100.0, 101.0, 103.0, 102.0, 104.0]},
             {90000: [100.0, 101.0, 102.0, 102.0, 103.0, 105.0]})


def _capture(monkeypatch, module) -> list[dict]:
    """Record every call of ``module.write_backtest_report``, then make it."""
    calls: list[dict] = []
    original = module.write_backtest_report

    def recording(account, path, **kwargs):
        original(account, path, **kwargs)
        # The page is written into a staging directory renamed afterwards.
        calls.append({"account": account, "page": path.read_text(encoding="utf-8"), **kwargs})

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


def _page(call) -> str:
    return call["page"]


def _setup(page: str) -> dict[str, str]:
    """The page's Setup table, label -> text, in page order."""
    start = page.index("<h2>Setup</h2>")
    block = page[start : page.index("</table>", start)]
    rows = re.findall(r"<tr><th>(.*?)</th><td>(?:<div class=\"scroll\">)?(.*?)(?:</div>)?</td></tr>", block)
    return {html_module.unescape(label): html_module.unescape(text) for label, text in rows}


def _timeline(page: str) -> str:
    """The page's windows timeline, caption to drawing."""
    start = page.index("<h2>Windows</h2>")
    return page[start : page.index("</svg>", start)]


def _tooltips(page: str) -> list[str]:
    return re.findall(r"<title>([^<]+)</title>", _timeline(page))


def _traces(page: str) -> dict:
    traces, pos = {}, 0
    while (i := page.find("Plotly.newPlot(", pos)) >= 0:
        batch, pos = json.JSONDecoder().raw_decode(page, page.index("[", i))
        traces.update({trace["name"]: trace for trace in batch})
    return traces


def _y(trace) -> np.ndarray:
    import plotly.io as pio

    return np.asarray(pio.from_json(json.dumps({"data": [trace]})).data[0].y, dtype=float)


def _holdings_data(page: str) -> dict:
    found = re.search(r'<script type="application/json" id="holdings-data">(.*?)</script>', page, re.S)
    return json.loads(found.group(1))


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
    quantlab, trader = _setup(_page(replay["quantlab"])), _setup(_page(replay["trader"]))

    assert list(trader)[: len(quantlab)] == list(quantlab)
    assert list(trader)[len(quantlab):] == ["Quantlab run", "Loop", "Slippage", "Init cash"]
    for label in quantlab:
        if label not in ("Fees", "Deepest drawdown (valley to recovery)"):
            assert trader[label] == quantlab[label], label
    assert trader["Fees"] == "0.001 of the traded notional"
    assert trader["Quantlab run"] == str(replay["quantlab_run"])
    assert (trader["Loop"], trader["Slippage"], trader["Init cash"]) == ("open loop", "0.0", "10,000.00")


def test_timeline_equals_quantlabs(replay):
    assert _timeline(_page(replay["trader"])) == _timeline(_page(replay["quantlab"]))


def test_the_report_gets_quantlabs_record_and_draws_the_benchmark_as_quantlab_does(replay):
    quantlab, trader = replay["quantlab"], replay["trader"]

    # The trader hands its cut folds; quantlab reads its own from its metrics.
    assert set(trader) - {"folds"} == set(quantlab) == {
        "account", "page", "title", "metrics", "weights", "config", "benchmark", "extras",
    }
    # trader simulates no cost-free or risk-model counterfactual, so its
    # extras hold no attribution; quantlab's are None without a model.
    assert trader["extras"].attribution is None and trader["extras"].factor_attribution is None
    assert trader["config"] == quantlab["config"]
    assert trader["metrics"].get("in_sample_range") == quantlab["metrics"].get("in_sample_range")
    assert trader["metrics"]["benchmark"]["symbol"] == quantlab["metrics"]["benchmark"]["symbol"]
    for account in (trader["account"], quantlab["account"]):
        assert account.year_freq / pd.Timedelta(account.bar_interval) == 252.0
    assert trader["account"].init_cash == quantlab["account"].init_cash
    assert trader["benchmark"].source == quantlab["benchmark"].source
    # The benchmark is the benchmark dataset's price return held from the
    # first close (no entry fee), read from the dataset so a live day past
    # the run's window has it; quantlab's own curve also charged the fee on
    # buying it, a constant factor 1 / (1 + fees) from the second bar on.
    assert trader["benchmark"].value is None
    expected = quantlab["benchmark"].value.values.copy()
    expected[1:] *= 1.0 + 0.001
    drawn = _y(_traces(_page(trader))["benchmark_equity"])
    np.testing.assert_allclose(drawn, expected, rtol=1e-12)


def test_portfolio_tab_draws_the_decided_targets(replay):
    decided = xr.open_zarr(replay["run_dir"] / "decisions.zarr").load()["weight"]

    xr.testing.assert_identical(replay["trader"]["weights"], decided)


def test_holdings_tab_shows_the_actual_holdings_at_each_close(replay):
    holdings = replay["trader"]["account"].holdings.transpose("timestamp", "symbol").to_pandas()

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
    np.testing.assert_allclose(stored.values, replay["trader"]["account"].holdings.values)
    # The fixture's price dataset names no ticker lookup: each symbol is its id.
    page = _page(replay["trader"])
    assert ["10002", "", "10002"] in _holdings_data(page)["names"]
    np.testing.assert_allclose(
        _y(_traces(page)["turnover"]), [10_300 / 10_000, (6_200 + 5_830.8) / 11_189.70], rtol=1e-12
    )


def test_trader_only_execution_facts_are_their_own_table(replay):
    table = replay["trader"]["extras"].tables["Execution (event-driven)"]

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

    assert calls[-1]["folds"] == [
        {"fold": 0, "training_window": ["2023-01-03", "2024-01-02"],
         "traded": ("2024-01-03", "2024-01-04"), "in_sample_range": None},
        {"fold": 1, "training_window": ["2023-01-03", "2024-01-05"],
         "traded": ("2024-01-05", "2024-01-09"), "in_sample_range": ["2024-01-05", "2024-01-05"]},
    ]
    page = _page(calls[-1])
    assert "2024-01-03 .. 2024-01-09" in _timeline(page)
    tips = _tooltips(page)
    assert [tip for tip in tips if tip.startswith("fold")] == [
        "fold 0 training 2023-01-03 .. 2024-01-02", "fold 0 traded 2024-01-03 .. 2024-01-04",
        "fold 1 training 2023-01-03 .. 2024-01-05", "fold 1 traded 2024-01-05 .. 2024-01-09",
        "fold 1 in-sample 2024-01-05 .. 2024-01-05",
    ]
    assert [tip for tip in tips if tip.startswith("in-sample")] == ["in-sample 2024-01-05 .. 2024-01-05"]
