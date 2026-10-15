"""The close briefing across the Account history (#65): ``briefing.close_briefing`` over two segments.

On the fake live harness of ``tests/test_live_run.py`` with a benchmark
(``tests/test_close_briefing.py``'s): the live run directory ``live/run``
trades the account over bars 5 to 8 (its segment begins on bar 6, its first
submission), then ``next/run`` takes over on bar 9 and trades to the last
bar. Locked:

- the account curve spans both directories' marked bars and is continuous
  at the switch (the switch bar's return is the account's, from the first
  directory's last mark to the second's first);
- one switch line, at the second segment's first decision date, named by
  its strategy;
- the whole-account figures are quantlab's ``return_stats`` /
  ``relative_stats`` of the concatenated returns and the benchmark on the
  account's bars; the segment figures are the current directory's alone;
- no history, or a directory not in it, falls back to the single-segment
  briefing and says so;
- a segment whose equity record is missing is refused, naming its directory.
"""

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

from quantlab.runs import backtest_stats
from quantlab_ibkr import briefing
from quantlab_ibkr.account_history import AccountHistory
from quantlab_ibkr.metrics import benchmark_record
from quantlab_ibkr.outputs import read_equity
from tests.test_close_briefing import BENCHMARK
from tests.test_live_run import FIRST, Day
from tests.test_live_source_clock import N_BARS

SWITCH = 9

YEAR = backtest_stats.year_freq("1D", 252, 390)


@pytest.fixture(scope="module")
def day(tmp_path_factory):
    """Two strategy segments on one account: ``live/run`` to bar 8, ``next/run`` from bar 9."""
    day = Day(tmp_path_factory.mktemp("account"), benchmark=BENCHMARK)
    day.first_dir = day.live_dir
    day.trade(range(FIRST, SWITCH))
    day.config = dataclasses.replace(
        day.config,
        venue=dataclasses.replace(day.config.venue, live_dir=str(day.root / "next" / "run")),
    )
    day.trade(range(SWITCH, N_BARS))
    return day


def _account_returns(day: Day) -> pd.Series:
    """The account's returns: the first record, then the second's with its first bar chained to the first's last mark."""
    first = read_equity(day.first_dir).to_dataframe()
    second = read_equity(day.live_dir).to_dataframe()
    returns = second["returns"].copy()
    returns.iloc[0] = second["value"].iloc[0] / first["value"].iloc[-1] - 1.0
    return pd.concat([first["returns"], returns])


def test_the_account_curve_spans_both_segments_and_is_continuous_at_the_switch(day):
    brief = briefing.close_briefing(day.config)
    first = read_equity(day.first_dir)
    second = read_equity(day.live_dir)
    expected = list(first["timestamp"].values) + list(second["timestamp"].values)
    curve = brief.account_curve
    assert list(curve["timestamp"].values) == expected
    np.testing.assert_array_equal(
        curve["value"].values, np.concatenate([first["value"].values, second["value"].values])
    )
    # Continuous: the cumulative return moves by the account's value change at every bar.
    value = curve["value"].values
    growth = 1.0 + curve["cumulative_return"].values
    np.testing.assert_allclose(growth / growth[0], value / value[0], rtol=1e-12)
    assert curve["cumulative_return"].values[0] == 0.0
    assert curve["benchmark_cumulative_return"].values[0] == 0.0
    assert brief.t == day.bars[N_BARS - 1]


def test_the_switch_line_sits_at_the_second_segments_first_decision_date(day):
    brief = briefing.close_briefing(day.config)
    assert brief.switches == ((day.bars[SWITCH], "next"),)
    assert [s.strategy for s in brief.segments] == ["live", "next"]
    assert f"{day.bars[SWITCH].date()} next" in brief.text


def test_the_account_figures_are_quantlabs_statistics_of_the_concatenated_returns(day):
    brief = briefing.close_briefing(day.config)
    returns = _account_returns(day)
    timestamps = returns.index.values
    benchmark = benchmark_record(day.run.benchmark(timestamps[0], timestamps[-1]), timestamps)
    whole = [(str(returns.index[0].date()), str(returns.index[-1].date()))]
    stats = backtest_stats.return_stats(returns.to_xarray(), bar_interval="1D", year_freq=YEAR)
    relative = backtest_stats.relative_stats(
        returns.to_xarray(), benchmark.returns, bar_interval="1D", year_freq=YEAR, ranges=whole
    )
    account = brief.account
    assert (account.start, account.end, account.bars) == (
        returns.index[0], returns.index[-1], len(returns)
    )
    assert account.total_return_pct == pytest.approx(stats["Total Return [%]"], rel=1e-12)
    assert account.volatility_pct == pytest.approx(stats["Annualized Volatility [%]"], rel=1e-12)
    assert account.sharpe == pytest.approx(stats["Sharpe Ratio"], rel=1e-12)
    assert account.max_drawdown_pct == pytest.approx(stats["Max Drawdown [%]"], rel=1e-12)
    assert account.excess_return_pct == pytest.approx(relative["Excess Return [%]"], rel=1e-12)
    assert account.information_ratio == pytest.approx(relative["Information Ratio"], rel=1e-12)
    assert account.benchmark_total_return_pct == pytest.approx(
        relative["Benchmark Total Return [%]"], rel=1e-12
    )

    # The segment figures are the current directory's alone, as #61 builds them.
    equity = read_equity(day.live_dir)
    own = backtest_stats.return_stats(equity["returns"], bar_interval="1D", year_freq=YEAR)
    assert brief.segment.start == day.bars[SWITCH]
    assert brief.segment.total_return_pct == own["Total Return [%]"]
    assert brief.segment.sharpe == own["Sharpe Ratio"]

    rows = dict(brief.rows)
    assert rows["账户累计收益 (Account Total Return)"] == f"{account.total_return_pct:+.2f}%"
    assert rows["累计收益 (Total Return)"] == f"{brief.segment.total_return_pct:+.2f}%"
    for label, text in brief.rows:
        assert f"{label}: {text}" in brief.text


def test_no_history_or_a_directory_not_in_it_falls_back_to_the_single_segment_briefing(
    day, tmp_path
):
    config = dataclasses.replace(
        day.config, venue=dataclasses.replace(day.config.venue, history_dir=str(tmp_path))
    )
    assert AccountHistory.of(config.venue).segments == ()  # an account with no history
    briefs = [briefing.close_briefing(config)]
    # A history that does not name the current directory: only the first segment.
    account = AccountHistory.of(day.config.venue)
    AccountHistory.of(config.venue).write(account.segments[:1])
    briefs.append(briefing.close_briefing(config))
    for brief in briefs:
        assert brief.account is None and brief.account_curve is None
        assert brief.segments == () and brief.switches == ()
        assert briefing.SINGLE_SEGMENT_NOTE in brief.text
        assert brief.segment.start == day.bars[SWITCH]
        assert not any(label.startswith("账户") for label, _ in brief.rows)


def test_a_segment_without_an_equity_record_is_refused_naming_its_directory(day):
    record = day.first_dir / "equity.zarr"
    aside = day.first_dir / "equity.zarr.aside"
    record.rename(aside)
    try:
        with pytest.raises(ValueError, match=re.escape(f"no equity record in {day.first_dir.resolve()}")):
            briefing.close_briefing(day.config)
    finally:
        aside.rename(record)
