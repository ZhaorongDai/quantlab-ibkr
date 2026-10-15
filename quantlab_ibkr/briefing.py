"""The close briefing of a live run directory (#60, #61; ADR 0012).

``close_briefing(config)`` describes the account a live config trades as of
its last marked bar t: the live run directory's equity record (cash plus
holdings at each raw close, ``equity.zarr``) against the quantlab run's
benchmark on the same bars. It is built after the ``decide`` step that marks
t, so its figures use the same marks as the directory's ``metrics.json`` and
``report.html``; the opening fills of t+1 appear in the next briefing.

The briefing carries:

- ``curve``: the chart's series on the directory's bars, rebased to the
  first marked bar: the live curve's cumulative return, the benchmark's and
  the live curve's drawdown;
- ``day``: t's return, PnL in dollars, the benchmark's return on t and the
  excess (``DayFigures``), the account's when there is an account history, so
  a switch day (named in ``DayFigures.switch``) shows the account's move across
  the switch (#70);
- ``segment``: the strategy segment's figures (``SegmentFigures``), quantlab's
  ``return_stats`` and ``relative_stats`` of the equity's returns and the
  benchmark's, so they agree with ``metrics.json``; the annualised return and
  excess only from ``ANNUALISED_FROM_BARS`` bars on;
- ``book``: what the strategy did and holds on t (``BookFigures``, #68): t's
  turnover, orders, holdings count and cash weight, the five largest
  holdings, the five largest contributors and detractors of t's return, t's
  events and the Decision recheck's latest result, securities named by the
  price dataset's ticker lookup (the PERMNO when it has no ticker);
- ``png``, ``html`` and ``text``: the chart (matplotlib's Agg canvas, two
  panels on one time axis), the e-mail's HTML body (inline styles only, the
  chart referenced by its content id ``CHART_CID``) and its plain-text
  alternative, carrying the same figures in Chinese, each metric's English
  abbreviation in brackets.

Across strategy switches (#65) the briefing covers the account's whole
**Account history** (``quantlab_ibkr.account_history``) up to the live run
directory's own segment:

- ``account_curve``: the account's returns are each segment's equity-record
  returns over its own span, concatenated in history order (the first
  segment from its record's first marked bar, each later one from its first
  decision date, every segment but the current one to its last), their
  cumulative product rebased to the account's first marked bar, the
  benchmark (the current quantlab run's, for every segment) rebased to the
  same bar. A later segment's first return is the
  account's, its first mark over the previous segment's last (a live run
  directory's own first return is 0, having no earlier mark), so the curve
  is continuous at the switch;
- ``switches``: each later segment's first decision date and strategy, a
  vertical line on the chart;
- ``account``: the whole account's figures (``SegmentFigures``, quantlab's
  statistics of the concatenated returns), shown beside the segment's;
  ``segment`` stays the current directory's whole record, as without a
  history.

Without a history, or with one that does not name the directory, the
briefing is the single segment's and its text says so
(``SINGLE_SEGMENT_NOTE``); a segment whose equity record is missing is
refused.

``replay`` is the closed-loop replay of the current strategy segment
(``segment_replay``, #67): its curve is drawn beside the live one and its
figures (the return gap, the tracking error, the unfilled orders and the
fees, live and replayed) form a section of their own; a replay that fails
is stated (``replay_failure``) and the briefing still builds.

Building reads the directories and the quantlab run; the only thing it
writes is the replay's run directory, ``<live_dir>/briefing/replay``,
replaced each time. Sending the briefing is a later step of #60.
"""

from __future__ import annotations

import html as html_lib
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs import backtest_stats
from quantlab.runs.run_metrics import BenchmarkRecord
from quantlab_ibkr import segment_replay
from quantlab_ibkr.account_history import AccountHistory, StrategySegment
from quantlab_ibkr._support.jsonable import python_scalar
from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.metrics import bar_clock, benchmark_record
from quantlab_ibkr.outputs import LiveRecorder, read_equity
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.segment_replay import SegmentReplay

logger = logging.getLogger(__name__)

#: The content id the HTML body references the chart by (``<img src="cid:...">``).
CHART_CID = "close-briefing-chart"

#: The annualised return and excess are shown from this many bars on.
ANNUALISED_FROM_BARS = 60

#: The section a briefing without an account history naming its directory carries.
SINGLE_SEGMENT_NOTE = "无账户历史 (account history) 记录本目录：仅显示当前策略段。"

#: The titles of the segment's and the whole account's cumulative sections, shown side by side.
SEGMENT_TITLE, ACCOUNT_TITLE = "当前策略段 (Strategy segment)", "整个账户 (Account)"
#: How many holdings, contributors and detractors the book lists.
TOP = 5

#: The files ``CloseBriefing.write`` writes: the chart, the HTML body, the plain text.
FILES = ("close_briefing.png", "close_briefing.html", "close_briefing.txt")


@dataclass(frozen=True)
class DayFigures:
    """The figures of decision date t, in percent except ``pnl``.

    Attributes
    ----------
    return_pct : float
        t's return of the account (of the live curve without an account
        history).
    pnl : float
        t's change of the account's value, in dollars.
    benchmark_return_pct : float or None
        The benchmark's return on t; ``None`` without a benchmark, NaN when
        it has not published t yet.
    excess_pct : float or None
        ``(1 + r) / (1 + b) - 1``, the geometric excess of t, as quantlab's
        ``Excess Return [%]`` compounds it.
    switch : str or None
        The strategy that took the account over on t, when t is a strategy
        switch; the figures are then the account's across the switch (#70).

    Examples
    --------
    >>> DayFigures(return_pct=0.5, pnl=500.0, benchmark_return_pct=None, excess_pct=None).pnl
    500.0
    """

    return_pct: float
    pnl: float
    benchmark_return_pct: float | None
    excess_pct: float | None
    switch: str | None = None


@dataclass(frozen=True)
class SegmentFigures:
    """The cumulative figures of a strategy segment, in quantlab's units.

    The ``*_pct`` fields are percentages; ``max_drawdown_pct`` and
    ``current_drawdown_pct`` are positive (quantlab's ``Max Drawdown [%]``).
    A benchmark figure is ``None`` without a benchmark, an annualised one
    ``None`` before ``ANNUALISED_FROM_BARS`` bars.

    Attributes
    ----------
    start, end : pandas.Timestamp
        The segment's first and last marked bar.
    bars : int
        Its marked bars.
    total_return_pct, annualized_return_pct, volatility_pct, sharpe, max_drawdown_pct
        ``return_stats``' ``Total Return [%]``, ``Annualized Return [%]``,
        ``Annualized Volatility [%]``, ``Sharpe Ratio``, ``Max Drawdown [%]``.
    current_drawdown_pct : float
        How far the compounded curve is below its running peak at t.
    benchmark_total_return_pct, excess_return_pct, annualized_excess_pct, information_ratio
        ``relative_stats``' ``Benchmark Total Return [%]``, ``Excess Return
        [%]``, ``Annualized Excess Return [%]``, ``Information Ratio``.

    Examples
    --------
    ``close_briefing`` fills them::

        segment = close_briefing(config).segment
        segment.total_return_pct, segment.information_ratio
    """

    start: pd.Timestamp
    end: pd.Timestamp
    bars: int
    total_return_pct: float
    annualized_return_pct: float | None
    volatility_pct: float
    sharpe: float
    max_drawdown_pct: float
    current_drawdown_pct: float
    benchmark_total_return_pct: float | None
    excess_return_pct: float | None
    annualized_excess_pct: float | None
    information_ratio: float | None


@dataclass(frozen=True)
class SecurityFigure:
    """One security's figure in the book: a holding's weight or a contribution, in percent.

    Attributes
    ----------
    symbol : Hashable
        The security's id on the live run directory's ``symbol`` axis (a PERMNO).
    ticker : str
        Its ticker on t through the price dataset's ticker lookup, the
        PERMNO itself (as ``str``) when the lookup has none.
    pct : float
        The weight or the contribution, in percent.

    Examples
    --------
    >>> SecurityFigure(symbol=10001, ticker="AAA", pct=12.5).ticker
    'AAA'
    """

    symbol: object
    ticker: str
    pct: float


@dataclass(frozen=True)
class BookFigures:
    """What the strategy did and holds on decision date t (#68).

    A security's contribution is its holding's change of value over t (the
    previous marked bar's close to t's, at the raw closes the equity marks)
    plus the cash it moved on t (next-open fills and their fees, a delisting
    settlement, dividends and other cash corporate actions), over the
    previous bar's equity: on a bar without fills or corporate actions, the
    holding's weight times the security's return. What no security moved
    (``cash_contribution_pct``) closes the sum to t's return.

    Attributes
    ----------
    turnover_pct : float
        quantlab's turnover of t's fills (``backtest_stats.turnover``, as
        ``metrics.json`` sums it): the one-sided notional the previous
        decision's next-open orders traded at t's open over the previous
        bar's equity; 0 without fills.
    filled_orders : int
        The next-open orders that filled at t's open.
    decided_turnover_pct : float or None
        The same one-sided turnover of t's own decision, ``sum |weight -
        current_weight|`` over its decided symbols (the trade sized at t's
        close over t's equity, filling at t+1's open); ``None`` when t is
        not a decided rebalance.
    orders : int
        The next-open orders decided on t (``orders.zarr``).
    holdings : int
        The securities held at t's close (``holdings.zarr``).
    cash_weight_pct : float
        Cash over equity at t's close, ``1 - sum(holding)``.
    top_holdings : tuple of SecurityFigure
        The ``TOP`` largest holdings by absolute weight, largest first.
    contributors, detractors : tuple of SecurityFigure
        The ``TOP`` largest positive and negative contributions, largest
        first; empty on the first marked bar.
    cash_contribution_pct : float or None
        t's return less the securities' contributions; ``None`` on the
        first marked bar.
    events : tuple of dict
        t's entries of ``events.json``, in its order: those dated t, and
        the unfilled orders whose next marked bar after their decision
        date is t (the open they missed).
    refused_orders : tuple of dict
        The orders decided on t the venue rejected or the risk engine
        denied at submission (``orders.zarr`` rows: ``symbol``, ``side``,
        ``quantity``, ``status``, ``reason``).
    recheck : dict or None
        The Decision recheck's latest result (``decision_recheck.json``),
        ``None`` before any recheck ran.
    tickers : dict
        The ticker on t of every security the book names (its figures, its
        events and refused orders), the PERMNO as ``str`` without one.

    Examples
    --------
    ``close_briefing`` fills them::

        book = close_briefing(config).book
        [(f.ticker, f.pct) for f in book.top_holdings]
    """

    turnover_pct: float
    filled_orders: int
    decided_turnover_pct: float | None
    orders: int
    holdings: int
    cash_weight_pct: float
    top_holdings: tuple[SecurityFigure, ...]
    contributors: tuple[SecurityFigure, ...]
    detractors: tuple[SecurityFigure, ...]
    cash_contribution_pct: float | None
    events: tuple[dict, ...]
    refused_orders: tuple[dict, ...]
    recheck: dict | None
    tickers: dict


@dataclass(frozen=True)
class CloseBriefing:
    """The close briefing of a live run directory as of decision date t.

    Attributes
    ----------
    t : pandas.Timestamp
        The last marked bar.
    name : str
        The strategy's name: the directory holding the live run directory
        (``live/<strategy>/ibkr``, as ``scripts/live_daily.sh`` lays it out),
        the name the step notifications carry.
    benchmark : str or None
        The benchmark's name (the quantlab run's), ``None`` without one.
    curve : xarray.Dataset
        On ``timestamp`` (the marked bars): ``value``, ``cumulative_return``
        (fractions, compounded from the first bar, whose return is 0),
        ``benchmark_cumulative_return`` (when there is a benchmark) and
        ``drawdown`` (negative or 0).
    day : DayFigures
    segment : SegmentFigures
    sections : tuple
        The figures as the e-mail shows them: ``(title, ((label, text), ...))``
        per section, the day's then the segment's; a figure that is absent
        (no benchmark, too few bars to annualise) has no row.
    subject : str
        The e-mail's subject, ``【IBKR 模拟盘】<strategy> <t> 收盘简报`` as the step
        notifications name theirs.
    png : bytes
        The chart.
    html : str
        The e-mail's HTML body; the chart is ``<img src="cid:CHART_CID">``.
    text : str
        The plain-text alternative, with the same figures.
    book : BookFigures
        t's trades, holdings, contributors and events.
    account : SegmentFigures or None
        The whole account's figures across its strategy segments; ``None``
        when the briefing falls back to the single segment.
    account_curve : xarray.Dataset or None
        ``curve``'s series on the account's bars, rebased to its first marked
        bar; the chart draws it. ``None`` on the fallback.
    segments : tuple of StrategySegment
        The account history's segments up to this directory's, oldest first;
        empty on the fallback.
    switches : tuple of (pandas.Timestamp, str)
        ``(first decision date, strategy)`` of each segment after the first.
    replay : SegmentReplay or None
        The closed-loop replay of the current strategy segment; ``None``
        when it failed. ``curve`` and ``account_curve`` carry it as
        ``replay_cumulative_return``, rebased to the live curve at the
        segment's start (NaN before it).
    replay_failure : str or None
        Why the replay failed; ``None`` when it ran.

    Examples
    --------
    ::

        brief = close_briefing(config)
        brief.write("out")  # close_briefing.png, .html, .txt
        if brief.account is not None:  # the account history names the directory
            brief.account.total_return_pct, brief.switches
    """

    t: pd.Timestamp
    name: str
    benchmark: str | None
    curve: xr.Dataset
    day: DayFigures
    segment: SegmentFigures
    sections: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    subject: str
    png: bytes
    html: str
    text: str
    book: BookFigures
    account: SegmentFigures | None = None
    account_curve: xr.Dataset | None = None
    segments: tuple[StrategySegment, ...] = ()
    switches: tuple[tuple[pd.Timestamp, str], ...] = ()
    replay: SegmentReplay | None = None
    replay_failure: str | None = None

    @property
    def rows(self) -> tuple[tuple[str, str], ...]:
        """Every section's ``(label, text)`` rows, in order.

        Examples
        --------
        ::

            dict(close_briefing(config).rows)["信息比率 (IR)"]
        """
        return tuple(row for _, rows in self.sections for row in rows)

    def write(self, directory: str | Path) -> list[Path]:
        """Write the chart, the HTML body and the plain text to ``directory`` (``FILES``).

        Returns
        -------
        list of pathlib.Path
            The three files written.

        Examples
        --------
        ::

            close_briefing(config).write("/tmp/brief")
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        png, page, text = (directory / name for name in FILES)
        png.write_bytes(self.png)
        page.write_text(self.html, encoding="utf-8")
        text.write_text(self.text, encoding="utf-8")
        return [png, page, text]


def close_briefing(config: TraderConfig) -> CloseBriefing:
    """Build the close briefing of ``config``'s live run directory as of its last marked bar.

    Parameters
    ----------
    config : TraderConfig
        A live config (``IbkrVenueConfig`` with a ``live_dir``).

    Returns
    -------
    CloseBriefing

    Raises
    ------
    ValueError
        If the config cannot trade live, the live run directory has no
        marked bar, or a segment of the account history has no equity record
        (or no marked bar in its span).

    Examples
    --------
    ::

        brief = close_briefing(TraderConfig.from_config(json.loads(Path("live.json").read_text())))
        print(brief.text)
    """
    from quantlab_ibkr.live import live_venue

    live_dir = Path(live_venue(config).live_dir)
    run = QuantlabRun.load(config.quantlab_run)
    if not LiveRecorder(config, run).started:
        raise ValueError(f"no marked bar in {live_dir}: run `live decide` first")
    equity = read_equity(live_dir)
    timestamps = pd.DatetimeIndex(equity["timestamp"].values)
    bar_interval, year_freq = bar_clock(
        timestamps, run.trading_days_per_year, run.session_minutes_per_day
    )
    benchmark = benchmark_record(run.benchmark(timestamps[0], timestamps[-1]), timestamps.values)
    returns = equity["returns"]

    curve = _curve(equity, benchmark)
    segment = _figures(curve, returns, benchmark, bar_interval, year_freq)
    t = timestamps[-1]
    name = live_dir.resolve().parent.name
    symbol = None if benchmark is None else (benchmark.symbol or "基准")
    history = _account_history(config, live_dir, t)
    account = account_curve = None
    segments, switches = (), ()
    if history is None:
        day = _day(equity, benchmark)
        after_segment = (SINGLE_SEGMENT_NOTE, ())
    else:
        segments, account_equity = history
        switches = tuple((s.first, s.strategy) for s in segments[1:])
        account_bars = pd.DatetimeIndex(account_equity["timestamp"].values)
        account_benchmark = benchmark_record(
            run.benchmark(account_bars[0], account_bars[-1]), account_bars.values
        )
        account_curve = _curve(account_equity, account_benchmark)
        account = _figures(
            account_curve, account_equity["returns"], account_benchmark, bar_interval, year_freq
        )
        # t's figures are the account's: on a switch day the directory's own
        # first return is 0, while the account moved from the last segment's mark.
        # The current segment's first account bar (never empty: _account_history
        # refuses a segment without a marked bar in its span).
        current_start = account_bars[account_bars >= segments[-1].first][0]
        switch = segments[-1].strategy if len(segments) > 1 and current_start == t else None
        day = _day(account_equity, account_benchmark, switch)
        after_segment = _account_section(symbol, account, len(segments), switches)
    sections = _after_segment(_sections(t, symbol, day, segment), after_segment)
    book = _book(config, run, live_dir, equity)
    sections = sections + _book_sections(book)
    replay, replay_failure = _replay(config, run, live_dir, timestamps)
    if replay is not None:
        curve = segment_replay.replay_curve(curve, replay)
        if account_curve is not None:
            account_curve = segment_replay.replay_curve(account_curve, replay)
    sections = (*sections, segment_replay.replay_section(replay, replay_failure))
    subject = f"【IBKR 模拟盘】{name} {t.date()} 收盘简报"
    return CloseBriefing(
        t=t,
        name=name,
        benchmark=symbol,
        curve=curve,
        day=day,
        segment=segment,
        sections=sections,
        subject=subject,
        png=_chart(curve if account_curve is None else account_curve, name, t, symbol, switches),
        html=_html(subject, t, sections),
        text=_text(subject, t, sections),
        account=account,
        account_curve=account_curve,
        segments=segments,
        switches=switches,
        book=book,
        replay=replay,
        replay_failure=replay_failure,
    )


def _day(equity: xr.Dataset, benchmark: BenchmarkRecord | None, switch: str | None = None) -> DayFigures:
    """The figures of ``equity``'s last bar against ``benchmark`` on the same bars."""
    r = float(equity["returns"].values[-1])
    b = None if benchmark is None else float(benchmark.returns.values[-1])
    value = np.asarray(equity["value"].values, dtype=float)
    return DayFigures(
        return_pct=r * 100.0,
        pnl=float(value[-1] - value[-2]) if len(value) > 1 else 0.0,
        benchmark_return_pct=None if b is None else b * 100.0,
        excess_pct=None if b is None else ((1.0 + r) / (1.0 + b) - 1.0) * 100.0,
        switch=switch,
    )


def _replay(
    config: TraderConfig, run: QuantlabRun, live_dir: Path, timestamps: pd.DatetimeIndex
) -> tuple[SegmentReplay | None, str | None]:
    """``(replay, None)``, or ``(None, reason)`` when the replay fails; logged either way."""
    try:
        start = segment_replay.segment_start(config, live_dir, timestamps)
        return segment_replay.replay_segment(config, run, live_dir, start, timestamps[-1]), None
    except Exception as error:  # the briefing states it and still goes out
        logger.exception("closed-loop replay of %s failed", live_dir)
        return None, f"{type(error).__name__}: {error}"


def _figures(
    curve: xr.Dataset,
    returns: xr.DataArray,
    benchmark: BenchmarkRecord | None,
    bar_interval: str,
    year_freq: str,
) -> SegmentFigures:
    """quantlab's statistics of ``returns`` (and the benchmark's) on ``curve``'s bars."""
    timestamps = pd.DatetimeIndex(curve["timestamp"].values)
    stats = backtest_stats.return_stats(returns, bar_interval=bar_interval, year_freq=year_freq)
    relative = (
        None
        if benchmark is None
        else backtest_stats.relative_stats(
            returns,
            benchmark.returns,
            bar_interval=bar_interval,
            year_freq=year_freq,
            ranges=[(str(timestamps[0].date()), str(timestamps[-1].date()))],
        )
    )
    annualised = len(timestamps) >= ANNUALISED_FROM_BARS
    return SegmentFigures(
        start=timestamps[0],
        end=timestamps[-1],
        bars=len(timestamps),
        total_return_pct=stats["Total Return [%]"],
        annualized_return_pct=stats["Annualized Return [%]"] if annualised else None,
        volatility_pct=stats["Annualized Volatility [%]"],
        sharpe=stats["Sharpe Ratio"],
        max_drawdown_pct=stats["Max Drawdown [%]"],
        current_drawdown_pct=float(-curve["drawdown"].values[-1] * 100.0),
        benchmark_total_return_pct=None if relative is None else relative["Benchmark Total Return [%]"],
        excess_return_pct=None if relative is None else relative["Excess Return [%]"],
        annualized_excess_pct=(
            relative["Annualized Excess Return [%]"] if relative is not None and annualised else None
        ),
        information_ratio=None if relative is None else relative["Information Ratio"],
    )


def _account_history(config: TraderConfig, live_dir: Path, t: pd.Timestamp):
    """The account history's segments up to ``live_dir``'s and the account's equity over them.

    Returns ``None`` when the account has no history or it does not name
    ``live_dir``. The equity is ``value`` and ``returns`` on ``timestamp``:
    each segment's record over its span (the first from its first marked
    bar, a later one from its first decision date; to its last, the current
    one to ``t``), a later segment's first return chained to the previous
    segment's last mark.
    """
    from quantlab_ibkr.live import live_venue

    segments = AccountHistory.of(live_venue(config)).segments
    here = str(live_dir.resolve())
    index = next((k for k, s in enumerate(segments) if s.live_dir == here), None)
    if index is None:
        return None
    segments = segments[: index + 1]
    pieces = []
    for k, segment in enumerate(segments):
        try:
            record = read_equity(Path(segment.live_dir)).to_dataframe()[["value", "returns"]]
        except FileNotFoundError:
            raise ValueError(
                f"no equity record in {segment.live_dir}, the account history's segment "
                f"{segment.strategy} ({segment.first.date()} on): the close briefing needs "
                "every segment's equity record"
            ) from None
        last = t if k == index else segment.last
        record = record.loc[(segment.first if k else None) : last].copy()
        if record.empty:
            raise ValueError(
                f"no marked bar in {segment.live_dir} over its segment {segment.strategy} "
                f"({segment.first.date()} to {last.date()})"
            )
        if pieces:  # the account's return over the switch, not the directory's first 0
            previous = pieces[-1]["value"].iloc[-1]
            record.loc[record.index[0], "returns"] = record["value"].iloc[0] / previous - 1.0
        pieces.append(record)
    frame = pd.concat(pieces)
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
    return segments, xr.Dataset.from_dataframe(frame)


def _curve(equity: xr.Dataset, benchmark) -> xr.Dataset:
    """The chart's series: cumulative returns from the first bar and the live curve's drawdown.

    A live record's first return is 0 (an account's is its first segment's),
    so the curve starts at 0 on the first marked bar.
    """
    returns = np.nan_to_num(np.asarray(equity["returns"].values, dtype=float))
    growth = np.cumprod(1.0 + returns)
    coords = {"timestamp": equity["timestamp"].values}
    data = {
        "value": equity["value"],
        "cumulative_return": ("timestamp", growth - 1.0),
        "drawdown": ("timestamp", growth / np.maximum.accumulate(growth) - 1.0),
    }
    if benchmark is not None:
        b = np.nan_to_num(np.asarray(benchmark.returns.values, dtype=float))
        data["benchmark_cumulative_return"] = ("timestamp", np.cumprod(1.0 + b) - 1.0)
    return xr.Dataset(data, coords=coords)


def _shown(form: str):
    """A formatter of ``form``; NaN (not published yet, too few bars) shows as 暂无."""
    return lambda value: "暂无" if np.isnan(value) else form.format(value)


#: Signed percent, unsigned percent, ratio, signed dollars.
_pct, _level, _ratio, _dollars = (
    _shown(form) for form in ("{:+.2f}%", "{:.2f}%", "{:.2f}", "{:+,.2f} USD")
)


def _took_over(strategy: str) -> str:
    """The switch day's row: the strategy that took the account over."""
    return f"{strategy} 接手账户"


def _depth(value: float) -> str:
    """A drawdown, positive, in percent; NaN (no drawdown yet) as 0."""
    return f"{0.0 if np.isnan(value) else value:.2f}%"


def _sections(t, symbol, day: DayFigures, segment: SegmentFigures) -> tuple:
    """The e-mail's figures, as ``CloseBriefing.sections``; a ``None`` figure has no row."""

    def rows(*figures):
        return tuple((label, show(value)) for label, value, show in figures if value is not None)

    s = segment
    return (
        (
            f"当日 ({t.date()})",
            rows(
                ("策略切换日 (Switch day)", day.switch, _took_over),
                ("当日收益 (Return)", day.return_pct, _pct),
                ("当日盈亏 (PnL)", day.pnl, _dollars),
                (f"基准当日收益 ({symbol})", day.benchmark_return_pct, _pct),
                ("当日超额 (Excess)", day.excess_pct, _pct),
            ),
        ),
        (
            f"{SEGMENT_TITLE}: {s.start.date()} 至 {s.end.date()}, {s.bars} 个交易日",
            _cumulative_rows(s, symbol),
        ),
    )


def _cumulative_rows(s: SegmentFigures, symbol, account: bool = False) -> tuple:
    """A span's cumulative figures as rows; the account's labels start with 账户 (Account ...)."""

    def label(chinese: str, english: str) -> str:
        return f"账户{chinese} (Account {english})" if account else f"{chinese} ({english})"

    figures = (
        (label("累计收益", "Total Return"), s.total_return_pct, _pct),
        (label("年化收益", "Annualized Return"), s.annualized_return_pct, _pct),
        (f"{'账户' if account else ''}基准累计收益 ({symbol})", s.benchmark_total_return_pct, _pct),
        (label("超额收益", "Excess Return"), s.excess_return_pct, _pct),
        (label("年化超额", "Annualized Excess"), s.annualized_excess_pct, _pct),
        (label("年化波动率", "Volatility"), s.volatility_pct, _level),
        (label("夏普比率", "Sharpe"), s.sharpe, _ratio),
        (label("信息比率", "IR"), s.information_ratio, _ratio),
        (label("最大回撤", "Max DD"), s.max_drawdown_pct, _depth),
        (label("当前回撤", "Current DD"), s.current_drawdown_pct, _depth),
    )
    return tuple((name, show(value)) for name, value, show in figures if value is not None)


def _account_section(symbol, account: SegmentFigures, n_segments: int, switches) -> tuple:
    """The whole account's cumulative section, its strategy switches the last row."""
    a = account
    title = (
        f"{ACCOUNT_TITLE}: {a.start.date()} 至 {a.end.date()}, {a.bars} 个交易日, "
        f"{n_segments} 个策略段"
    )
    rows = _cumulative_rows(a, symbol, account=True)
    if switches:
        rows += (("策略切换 (Switch)", "; ".join(f"{d.date()} {name}" for d, name in switches)),)
    return (title, rows)


def _after_segment(sections: tuple, section: tuple) -> tuple:
    """``sections`` with ``section`` right after the segment's (at the end without one)."""
    at = next(
        (k + 1 for k, (title, _) in enumerate(sections) if title.startswith(SEGMENT_TITLE)),
        len(sections),
    )
    return sections[:at] + (section,) + sections[at:]


def _chart(
    curve: xr.Dataset, name: str, t: pd.Timestamp, symbol: str | None, switches=()
) -> bytes:
    """The chart as PNG: cumulative returns above, the live curve's drawdown beneath.

    Each of ``switches`` (``(date, strategy)``) is a dashed vertical line on
    both panels, labelled with the strategy at the top panel's upper edge.

    Drawn on matplotlib's Agg canvas (no pyplot, no global backend). The
    labels are English, so no CJK font is needed where it is drawn.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8.0, 5.0), dpi=100)
    FigureCanvasAgg(figure)
    top, bottom = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    when = pd.DatetimeIndex(curve["timestamp"].values)
    top.plot(when, curve["cumulative_return"].values * 100.0, color="#1f5aa6", label="Live")
    if "benchmark_cumulative_return" in curve:
        top.plot(
            when, curve["benchmark_cumulative_return"].values * 100.0,
            color="#888888", label=symbol or "Benchmark",
        )
    if "replay_cumulative_return" in curve:
        top.plot(
            when, curve["replay_cumulative_return"].values * 100.0,
            color="#e67e22", linestyle="--", label="Closed-loop replay",
        )
    top.axhline(0.0, color="#cccccc", linewidth=0.8)
    top.set_ylabel("Cumulative return [%]")
    top.set_title(f"{name}  {t.date()}")
    top.legend(loc="upper left", frameon=False)
    drawdown = curve["drawdown"].values * 100.0
    bottom.fill_between(when, drawdown, 0.0, color="#c0392b", alpha=0.35, linewidth=0)
    bottom.plot(when, drawdown, color="#c0392b", linewidth=0.8)
    bottom.set_ylabel("Drawdown [%]")
    for axis in (top, bottom):
        axis.grid(True, color="#eeeeee")
    for when_switch, strategy in switches:
        for axis in (top, bottom):
            axis.axvline(when_switch, color="#555555", linestyle="--", linewidth=0.9)
        top.annotate(
            strategy, (when_switch, 1.0), xycoords=("data", "axes fraction"),
            xytext=(3, -3), textcoords="offset points", va="top", fontsize=8, color="#555555",
        )
    figure.autofmt_xdate()
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()


def _html(subject: str, t: pd.Timestamp, sections) -> str:
    """The HTML body: inline styles only, the chart by its content id."""
    escape = html_lib.escape
    cell = "padding:4px 12px;border-bottom:1px solid #eeeeee;"
    beside = '<td style="vertical-align:top;padding:0 16px 0 0;">'
    parts = [
        '<div style="font-family:Helvetica,Arial,sans-serif;color:#222222;max-width:760px;">',
        f'<h2 style="font-size:18px;margin:0 0 4px 0;">{escape(subject)}</h2>',
        f'<p style="font-size:13px;color:#666666;margin:0 0 12px 0;">'
        f"账户截至 {t.date()} 收盘 (close)，按实盘记录的权益 (equity) 计算。</p>",
        f'<img src="cid:{CHART_CID}" alt="cumulative return and drawdown" '
        'style="width:100%;max-width:760px;display:block;margin:0 0 12px 0;">',
    ]

    def table(title, rows):
        return [
            f'<h3 style="font-size:15px;margin:16px 0 4px 0;">{escape(title)}</h3>',
            '<table style="border-collapse:collapse;font-size:14px;">',
            *(
                f'<tr><td style="{cell}">{escape(label)}</td>'
                f'<td style="{cell}text-align:right;font-family:Menlo,Consolas,monospace;">'
                f"{escape(text)}</td></tr>"
                for label, text in rows
            ),
            "</table>",
        ]

    for k, (title, rows) in enumerate(sections):
        if title.startswith(ACCOUNT_TITLE) and k and sections[k - 1][0].startswith(SEGMENT_TITLE):
            continue  # drawn beside the segment's
        following = sections[k + 1] if k + 1 < len(sections) else ("", ())
        if title.startswith(SEGMENT_TITLE) and following[0].startswith(ACCOUNT_TITLE):
            # The segment's and the account's cumulative figures side by side.
            parts += ['<table style="border-collapse:collapse;"><tr>', beside, *table(title, rows)]
            parts += ["</td>", beside, *table(*following), "</td></tr></table>"]
        else:
            parts += table(title, rows)
    parts.append("</div>")
    return "\n".join(parts)


def _text(subject: str, t: pd.Timestamp, sections) -> str:
    """The plain-text alternative: the same figures, one per line."""
    lines = [subject, f"账户截至 {t.date()} 收盘 (close)，按实盘记录的权益 (equity) 计算。"]
    for title, rows in sections:
        lines += ["", title, *(f"  {label}: {text}" for label, text in rows)]
    return "\n".join(lines) + "\n"


def _book(config: TraderConfig, run: QuantlabRun, live_dir: Path, equity: xr.Dataset) -> BookFigures:
    """``BookFigures`` of the live run directory's last marked bar (the class docstring)."""
    from quantlab_ibkr.live import RECHECK_FILE

    timestamps = pd.DatetimeIndex(equity["timestamp"].values)
    t = timestamps[-1]
    previous = timestamps[-2] if len(timestamps) > 1 else None
    holdings = _open(live_dir / "holdings.zarr")["holding"].transpose("timestamp", "symbol").to_pandas()
    holdings.columns = [python_scalar(v) for v in holdings.columns]
    held = holdings.loc[t].fillna(0.0)
    held = held[held != 0.0]

    decided_turnover = _decided_turnover(_open(live_dir / "decisions.zarr"), t)
    orders = _open(live_dir / "orders.zarr")
    on_t = pd.DatetimeIndex(orders["decision_date"].values) == t if orders.sizes.get("order") else []
    refused = tuple(
        {
            "symbol": python_scalar(orders["symbol"].values[i]),
            **{key: str(orders[key].values[i]) for key in ("side", "status", "reason")},
            "quantity": int(orders["quantity"].values[i]),
        }
        for i in np.flatnonzero(on_t)
        if str(orders["status"].values[i]) in ("rejected", "denied")
    )

    state = LiveRecorder(config, run).state()
    filled = [f for f in state["fills"] if pd.Timestamp(f["timestamp"]) == t]
    turnover = backtest_stats.turnover(
        xr.Dataset({
            "timestamp": ("order", np.array([t] * len(filled), dtype="datetime64[ns]")),
            "size": ("order", np.array([f["size"] for f in filled], dtype=float)),
            "price": ("order", np.array([f["price"] for f in filled], dtype=float)),
        }),
        equity["value"],
        init_cash=float(state["init_cash"]),
    )
    events = tuple(e for e in state["events"] if _event_bar(e, timestamps) == t)

    contributions: dict = {}
    cash_contribution = None
    if previous is not None:
        contributions = _contributions(holdings, equity, state, previous, t)
        r = float(equity["returns"].sel(timestamp=t)) * 100.0
        cash_contribution = r - sum(contributions.values())

    ranked = sorted(contributions.items(), key=lambda item: -item[1])
    largest = held.reindex(held.abs().sort_values(ascending=False, kind="stable").index)
    lists = {
        "top_holdings": [(s, w * 100.0) for s, w in largest.items()][:TOP],
        "contributors": [(s, v) for s, v in ranked if v > 0][:TOP],
        "detractors": [(s, v) for s, v in reversed(ranked) if v < 0][:TOP],
    }
    named = [s for pairs in lists.values() for s, _ in pairs]
    for record in (*events, *refused):
        named += [record["symbol"]] if "symbol" in record else list(record.get("symbols", ()))
    named = list(dict.fromkeys(named))
    tickers = dict(zip(named, _tickers(run, named, t)))
    recheck_path = live_dir / RECHECK_FILE
    return BookFigures(
        turnover_pct=float(turnover.values.sum() * 100.0),
        filled_orders=len({f["order"] for f in filled}),
        decided_turnover_pct=decided_turnover,
        orders=int(np.count_nonzero(on_t)),
        holdings=len(held),
        cash_weight_pct=float((1.0 - held.sum()) * 100.0),
        **{
            name: tuple(SecurityFigure(s, tickers[s], float(v)) for s, v in pairs)
            for name, pairs in lists.items()
        },
        cash_contribution_pct=cash_contribution,
        events=events,
        refused_orders=refused,
        recheck=json.loads(recheck_path.read_text()) if recheck_path.is_file() else None,
        tickers=tickers,
    )


def _decided_turnover(decisions: xr.Dataset, t: pd.Timestamp) -> float | None:
    """The one-sided turnover of t's decision in percent; ``None`` when t decided nothing."""
    if t not in pd.DatetimeIndex(decisions["timestamp"].values):
        return None
    row = decisions.sel(timestamp=t)
    weight = np.asarray(row["weight"].values, dtype=float)
    current = np.nan_to_num(np.asarray(row["current_weight"].values, dtype=float))
    decided = np.isfinite(weight)  # NaN keeps the holding: no trade
    return float(np.abs(weight[decided] - current[decided]).sum() * 100.0)


def _contributions(
    holdings: pd.DataFrame, equity: xr.Dataset, state: dict, previous: pd.Timestamp, t: pd.Timestamp
) -> dict:
    """Each security's contribution to t's return in percent (``BookFigures``), nonzero ones only.

    Its holding's value at t's close less at the previous bar's, plus the
    cash it moved after the previous close up to t: next-open fills (price
    and fee), a delisting settlement, cash corporate actions; a split moves
    no cash. Over the previous bar's equity.
    """
    start = float(equity["value"].sel(timestamp=previous))
    end = float(equity["value"].sel(timestamp=t))
    moved = (holdings.loc[t].fillna(0.0) * end - holdings.loc[previous].fillna(0.0) * start).to_dict()

    def within(when) -> bool:
        return previous < pd.Timestamp(when) <= t

    for fill in state["fills"]:
        if within(fill["timestamp"]):
            cash = -fill["size"] * fill["price"] - fill["fee"]
            moved[fill["symbol"]] = moved.get(fill["symbol"], 0.0) + cash
    for event in state["events"]:
        if event.get("type") != "corporate_action" or not within(event["timestamp"]):
            continue
        if event["action"] == "DELIST":
            sign = 1 if event["side"] == "BUY" else -1
            cash = -sign * event["quantity"] * event["price"] - event["fee"]
        elif "side" in event:  # a split's venue fill: the share count changes, never the cash
            continue
        else:  # cash booked without a fill (CASH_ACTIONS); a logged factor day books 0
            cash = float(event.get("amount", 0.0))
        moved[event["symbol"]] = moved.get(event["symbol"], 0.0) + cash
    return {s: v / start * 100.0 for s, v in moved.items() if v != 0.0}


def _open(path: Path) -> xr.Dataset:
    with xr.open_zarr(path) as dataset:
        return dataset.load()


def _tickers(run: QuantlabRun, symbols: list, t: pd.Timestamp) -> list[str]:
    """The tickers of ``symbols`` on t through the price dataset's ticker lookup, else the PERMNOs."""
    lookup = run.price_dataset.ticker_lookup() if symbols else None
    if lookup is None:
        return [str(s) for s in symbols]
    return [name.ticker for name in lookup.names(symbols, t.date())]


def _event_bar(event: dict, timestamps: pd.DatetimeIndex) -> pd.Timestamp | None:
    """The marked bar an event belongs to: its date; an unfilled order's, the bar after its decision."""
    if event.get("type") == "unfilled_order":
        position = timestamps.searchsorted(pd.Timestamp(event["decision_date"]), side="right")
        return timestamps[position] if position < len(timestamps) else None
    when = event.get("timestamp")
    return None if when is None else pd.Timestamp(when)


#: The event types' labels in the e-mail.
_EVENT_LABELS = {
    "hold": "暂停调仓 (Hold)",
    "live_hold": "暂停调仓 (Hold)",
    "rule_event": "规则事件 (Rule event)",
    "unfilled_order": "未成交订单 (Unfilled)",
    "corporate_action": "公司行动 (Corporate action)",
    "excluded_symbols": "排除标的 (Excluded)",
    "position_gap": "持仓缺口 (Position gap)",
    "adopted_orders": "接管订单 (Adopted orders)",
    "timed_out": "超时 (Timed out)",
}


def _event_text(event: dict, name) -> str:
    """One event as the e-mail shows it; ``name(symbol)`` is the security's ticker."""
    kind = event.get("type")
    if kind == "hold":
        return str(event["failure"])
    if kind == "live_hold":
        return str(event["reason"])
    if kind == "rule_event":
        detail = event["count"] if "count" in event else ", ".join(name(s) for s in event["symbols"])
        return f"{event['name']}: {detail}"
    if kind == "unfilled_order":
        return f"{event['side']} {event['quantity']} {name(event['symbol'])}: {event['reason']}"
    if kind == "corporate_action":
        if "side" in event:
            detail = f"{event['side']} {event['quantity']} @ {event['price']:.2f}"
        else:
            detail = f"{event['quantity']} 股, 现金 {event['amount']:+,.2f} USD"
        return f"{event['action']} {name(event['symbol'])}: {detail}"
    if kind in ("excluded_symbols", "adopted_orders"):
        symbols = ", ".join(name(s) for s in event["symbols"])
        return f"{symbols}: {event['reason']}" if "reason" in event else symbols
    if kind == "position_gap":
        return f"{event['instrument_id']}: {event['reason']}"
    if kind == "timed_out":
        return "下单步骤超时 (the decide step timed out)"
    return json.dumps({k: v for k, v in event.items() if k != "type"}, ensure_ascii=False)


def _recheck_text(recheck: dict | None) -> str:
    """The Decision recheck's latest result in one line."""
    if recheck is None:
        return "尚未运行 (not run yet)"
    through = f"截至 {recheck['last_bar']}" if recheck.get("last_bar") else ""
    if recheck["bars_differing"]:
        bars = ", ".join(str(bar["timestamp"])[:10] for bar in recheck["differing_bars"])
        return f"不一致 (differs): {recheck['bars_differing']}/{recheck['bars_checked']} 个调仓日 ({bars}) {through}".rstrip()
    if not recheck["bars_checked"]:
        return f"无新调仓日 (nothing new) {through}".rstrip()
    return f"通过 (pass): 复核 {recheck['bars_checked']} 个调仓日, {through}".rstrip(", ")


def _book_sections(book: BookFigures) -> tuple:
    """The book's sections, as ``CloseBriefing.sections``: trades, holdings, contributors, events."""
    def securities(figures, show):
        return tuple((f.ticker, show(f.pct)) for f in figures)

    decided = book.decided_turnover_pct
    trades = (
        ("成交换手 (Turnover)", _level(book.turnover_pct)),
        ("成交订单 (Filled orders)", str(book.filled_orders)),
        ("决策换手 (Decided turnover)", "未调仓 (no rebalance)" if decided is None else _level(decided)),
        ("新订单 (Orders)", str(book.orders)),
        ("持仓数 (Holdings)", str(book.holdings)),
        ("现金权重 (Cash)", _level(book.cash_weight_pct)),
    )
    if book.cash_contribution_pct is not None:
        trades += (("现金及其他贡献 (Cash & other)", _pct(book.cash_contribution_pct)),)
    sections = [("持仓与交易 (Book)", trades)]
    for title, figures, show in (
        ("前五大持仓 (Top holdings)", book.top_holdings, _level),
        ("贡献前五 (Top contributors)", book.contributors, _pct),
        ("拖累前五 (Top detractors)", book.detractors, _pct),
    ):
        if figures:
            sections.append((title, securities(figures, show)))
    def name(symbol):
        return book.tickers.get(symbol, str(symbol))

    events = tuple(
        (_EVENT_LABELS.get(event.get("type"), str(event.get("type"))), _event_text(event, name))
        for event in book.events
    ) + tuple(
        ("被拒订单 (Rejected)", f"{o['side']} {o['quantity']} {name(o['symbol'])}: {o['reason']}")
        for o in book.refused_orders
    )
    sections.append(
        ("事件 (Events)", (events or (("无事件 (No events)", "-"),)) + (
            ("决策复核 (Decision recheck)", _recheck_text(book.recheck)),
        ))
    )
    return tuple(sections)
