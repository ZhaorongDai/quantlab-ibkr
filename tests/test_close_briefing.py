"""The close briefing of a live run directory (#61): ``briefing.close_briefing`` on the fake live harness.

The live days are ``tests/test_live_run.py``'s ``Day`` (a fake IBKR account
traded over the fixture run's last bars), with a benchmark store beside the
run (one PERMNO, the run's ``benchmark_dataset``). Locked:

- the segment figures are quantlab's ``return_stats`` / ``relative_stats`` of
  the live run directory's equity and the run's benchmark on its bars, and
  agree with the directory's ``metrics.json``; the day figures are t's;
- a bar never decided is absent from the curve; the annualised figures are
  absent before ``ANNUALISED_FROM_BARS`` bars;
- a live run directory with no marked bar is refused;
- the HTML references the chart by its content id and the plain text
  carries the same figures;
- building writes nothing into the live run directory;
- ``live brief --no-send --output DIR`` writes the three files.
"""

import html
import json

import numpy as np
import pandas as pd
import pytest

from quantlab.runs import backtest_stats
from quantlab_ibkr import briefing, cli
from quantlab_ibkr.metrics import benchmark_record
from quantlab_ibkr.outputs import read_equity
from tests.test_live_run import FIRST, Day
from tests.test_live_source_clock import N_BARS

#: The benchmark's raw (open, close): one PERMNO over the run's bars.
_SPY_CLOSE = 400.0 * np.cumprod(1.0 + np.random.default_rng(7).normal(0.0, 0.01, N_BARS))
BENCHMARK = ({84398: list(_SPY_CLOSE * 0.999)}, {84398: list(_SPY_CLOSE)})


@pytest.fixture
def day(tmp_path):
    return Day(tmp_path, benchmark=BENCHMARK)


def _stats(day: Day):
    """quantlab's return and relative statistics of the live directory's equity and benchmark."""
    equity = read_equity(day.live_dir)
    timestamps = equity["timestamp"].values
    year_freq = backtest_stats.year_freq("1D", 252, 390)
    benchmark = benchmark_record(day.run.benchmark(timestamps[0], timestamps[-1]), timestamps)
    whole = [(str(pd.Timestamp(timestamps[0]).date()), str(pd.Timestamp(timestamps[-1]).date()))]
    returns = backtest_stats.return_stats(equity["returns"], bar_interval="1D", year_freq=year_freq)
    relative = backtest_stats.relative_stats(
        equity["returns"], benchmark.returns, bar_interval="1D", year_freq=year_freq, ranges=whole
    )
    return equity, benchmark, returns, relative


def test_the_segment_figures_are_quantlabs_statistics_and_agree_with_metrics_json(day):
    day.trade(range(FIRST, N_BARS))
    brief = briefing.close_briefing(day.config)
    equity, benchmark, returns, relative = _stats(day)

    assert brief.t == day.bars[N_BARS - 1]
    segment = brief.segment
    assert segment.bars == N_BARS - FIRST
    assert segment.total_return_pct == returns["Total Return [%]"]
    assert segment.volatility_pct == returns["Annualized Volatility [%]"]
    assert segment.sharpe == returns["Sharpe Ratio"]
    assert segment.max_drawdown_pct == returns["Max Drawdown [%]"]
    assert segment.excess_return_pct == relative["Excess Return [%]"]
    assert segment.information_ratio == relative["Information Ratio"]
    assert segment.benchmark_total_return_pct == relative["Benchmark Total Return [%]"]
    growth = np.cumprod(1.0 + equity["returns"].values)
    assert segment.current_drawdown_pct == pytest.approx(
        (1.0 - growth[-1] / growth.max()) * 100.0, abs=1e-12
    )

    metrics = json.loads((day.live_dir / "metrics.json").read_text())
    assert segment.total_return_pct == pytest.approx(metrics["whole"]["Total Return [%]"], rel=1e-12)
    assert segment.sharpe == pytest.approx(metrics["whole"]["Sharpe Ratio"], rel=1e-12)
    assert segment.max_drawdown_pct == pytest.approx(metrics["whole"]["Max Drawdown [%]"], rel=1e-12)
    for name, key in (
        ("excess_return_pct", "Excess Return [%]"),
        ("information_ratio", "Information Ratio"),
    ):
        assert getattr(segment, name) == pytest.approx(metrics["relative"]["whole"][key], rel=1e-12)

    r, b = equity["returns"].values[-1], benchmark.returns.values[-1]
    value = equity["value"].values
    assert brief.day.return_pct == pytest.approx(r * 100.0, rel=1e-12)
    assert brief.day.pnl == pytest.approx(value[-1] - value[-2], rel=1e-9)
    assert brief.day.benchmark_return_pct == pytest.approx(b * 100.0, rel=1e-12)
    assert brief.day.excess_pct == pytest.approx(((1 + r) / (1 + b) - 1) * 100.0, rel=1e-12)


def test_a_bar_never_decided_is_absent_and_annualised_figures_wait_for_enough_bars(day, monkeypatch):
    day.trade(range(FIRST, 8))
    day.trade([9])  # bar 8 is never decided: no equity row, no curve point
    brief = briefing.close_briefing(day.config)
    expected = [day.bars[k] for k in (5, 6, 7, 9)]
    assert list(brief.curve["timestamp"].values) == expected
    assert brief.segment.bars == 4
    assert brief.segment.annualized_return_pct is None
    assert brief.segment.annualized_excess_pct is None

    monkeypatch.setattr(briefing, "ANNUALISED_FROM_BARS", 4)
    brief = briefing.close_briefing(day.config)
    _, _, returns, relative = _stats(day)
    assert brief.segment.annualized_return_pct == returns["Annualized Return [%]"]
    assert brief.segment.annualized_excess_pct == relative["Annualized Excess Return [%]"]


def test_a_live_run_directory_with_no_marked_bar_is_refused(day):
    with pytest.raises(ValueError, match="no marked bar"):
        briefing.close_briefing(day.config)


def test_the_html_references_the_chart_by_content_id_and_the_text_carries_the_same_figures(day):
    day.trade(range(FIRST, N_BARS))
    brief = briefing.close_briefing(day.config)
    assert brief.png.startswith(b"\x89PNG\r\n\x1a\n")
    assert f'src="cid:{briefing.CHART_CID}"' in brief.html
    page = html.unescape(brief.html)
    labels = [label for label, _ in brief.rows]
    for label, text in brief.rows:
        assert label in page and text in page, label
        assert f"{label}: {text}" in brief.text, label
    rows = dict(brief.rows)
    assert rows["信息比率 (IR)"] == f"{brief.segment.information_ratio:.2f}"
    assert rows["累计收益 (Total Return)"] == f"{brief.segment.total_return_pct:+.2f}%"
    assert rows["当日盈亏 (PnL)"] == f"{brief.day.pnl:+,.2f} USD"
    assert "年化收益 (Annualized Return)" not in labels  # fewer than 60 bars
    assert brief.subject == f"【IBKR 模拟盘】live {brief.t.date()} 收盘简报"  # live/<strategy>/run
    assert brief.subject in brief.text


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_building_writes_nothing_into_the_live_run_directory(day):
    day.trade(range(FIRST, 8))
    before = _snapshot(day.live_dir)
    briefing.close_briefing(day.config)
    assert _snapshot(day.live_dir) == before


def test_live_brief_no_send_writes_the_three_files(day, capsys):
    day.trade(range(FIRST, 8))
    path = day.root / "live.json"
    path.write_text(json.dumps(day.config.get_config()))
    out = day.root / "brief"
    assert cli.main(["live", "brief", str(path), "--no-send", "--output", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(briefing.FILES)
    png, page, text = (out / name for name in briefing.FILES)
    brief = briefing.close_briefing(day.config)
    assert png.read_bytes().startswith(b"\x89PNG")
    assert page.read_text(encoding="utf-8") == brief.html
    assert text.read_text(encoding="utf-8") == brief.text
    assert str(out) in capsys.readouterr().out

    with pytest.raises(SystemExit):  # sending is #64
        cli.main(["live", "brief", str(path)])
    assert "--no-send" in capsys.readouterr().err
