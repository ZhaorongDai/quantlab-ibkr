"""The close briefing's Closed-loop replay of the current strategy segment (#60, #67).

``replay_segment`` runs the closed-loop backtest (``runner.run``, closed
loop, the backtest venue) on the live run directory's quantlab run over the
current strategy segment, from its first decision date to t, so the gap
between the live curve and the replay is execution alone (ADR 0001, 0012):

- the replay decides from the **same predictions** live decided from, the
  live prediction store's rows (``IbkrVenueConfig.prediction_store``), not
  the quantlab run's own panel;
- it starts in cash at the segment's starting equity (the live equity at the
  segment's first marked bar) and charges the trader's default closed-loop
  fee model (IBKR Pro Fixed); a segment that did not start from an empty
  account is stated (``SegmentReplay.empty_start``);
- the segment's first decision date is the account history's segment of
  this live run directory (``account_history``), else the directory's first
  marked bar;
- its run directory is ``<live_dir>/briefing/replay`` (``BRIEFING_DIR``,
  ``REPLAY_DIR``), replaced each time and holding ``replay.json``
  (``RECORD_FILE``: the inputs and the wall time); the live run directory's
  own files are never touched;
- the wall time is logged (``logging``, this module's logger), at warning
  level past ``SLOW_REPLAY_SECONDS``, the cue for an incremental replay.

``replay_section`` gives the briefing's figures of it: the cumulative return
gap (live minus replay, quantlab's ``Total Return Difference [%]``), the
tracking error (``Tracking Error [%]`` of the live returns against the
replay's), the unfilled orders and the cumulative fees, live and replayed.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs import backtest_stats
from quantlab.runs.live_predictions import LivePredictionStore
from quantlab.tracking.base import NullTracker
from quantlab.utils.date_range import bar_label
from quantlab_ibkr.account_history import AccountHistory
from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.base.venue import Loop
from quantlab_ibkr.metrics import UNFILLED_STATUSES, bar_clock
from quantlab_ibkr.outputs import read_equity, read_max_capital
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.runner import run as run_backtest
from quantlab_ibkr.venue.backtest.venue import BacktestVenueConfig, ExecutionConfig

logger = logging.getLogger(__name__)

#: The live run directory's subdirectory holding what the briefing computes.
BRIEFING_DIR = "briefing"
#: The replay's run directory, under ``BRIEFING_DIR``.
REPLAY_DIR = "replay"
#: The replay's record in its run directory: inputs and wall time.
RECORD_FILE = "replay.json"
#: A replay slower than this is logged as a warning (the cue for an incremental replay).
SLOW_REPLAY_SECONDS = 15 * 60


@dataclass(frozen=True)
class SegmentReplay:
    """A closed-loop replay of a live strategy segment and its gap to the live curve.

    Attributes
    ----------
    start, end : pandas.Timestamp
        The segment's first decision date (its first marked bar) and t.
    run_dir : pathlib.Path
        The replay's trader run directory (``<live_dir>/briefing/replay``).
    init_cash : float
        The replay's starting cash: the live equity at ``start``.
    held_at_start : int
        Securities the live account held at ``start``'s close.
    curve : xarray.Dataset
        On the live run directory's marked bars from ``start``:
        ``live_value`` and ``replay_value``.
    live_return_pct, replay_return_pct : float
        The segment's compounded returns, live and replayed, in percent.
    gap_pct : float
        Live minus replay, in percentage points.
    tracking_error_pct : float
        The annualised standard deviation of the per-bar live minus replay
        returns, in percent; NaN before two bars.
    unfilled_live, unfilled_replay : int
        The segment's orders decided before t that did not fill
        (``metrics.UNFILLED_STATUSES``). t's own orders fill at t+1's open,
        after the close the briefing reports (the record step may already
        have recorded them), and the replay, whose window ends at t, never
        decides t, so neither side counts them (#73).
    fees_live, fees_replay : float
        The cumulative fees in dollars of the segment's orders decided
        before t, as for the unfilled orders.
    seconds : float
        The replay's wall time.

    Examples
    --------
    ``close_briefing`` builds it::

        replay = close_briefing(config).replay
        replay.gap_pct, replay.tracking_error_pct
    """

    start: pd.Timestamp
    end: pd.Timestamp
    run_dir: Path
    init_cash: float
    held_at_start: int
    curve: xr.Dataset
    live_return_pct: float
    replay_return_pct: float
    gap_pct: float
    tracking_error_pct: float
    unfilled_live: int
    unfilled_replay: int
    fees_live: float
    fees_replay: float
    seconds: float

    @property
    def empty_start(self) -> bool:
        """Whether the segment started from an empty account (cash only).

        Examples
        --------
        ::

            if not replay.empty_start:
                print(f"the account held {replay.held_at_start} securities at the start")
        """
        return self.held_at_start == 0


def segment_start(config: TraderConfig, live_dir: Path, marked: pd.DatetimeIndex) -> pd.Timestamp:
    """Return the current strategy segment's first marked bar in ``live_dir``.

    The account history's segment of ``live_dir`` gives the first decision
    date (``AccountHistory.of(config.venue)``); without a history, or one
    that does not name the directory, the directory's first marked bar.

    Parameters
    ----------
    config : TraderConfig
        The live config.
    live_dir : pathlib.Path
        Its live run directory.
    marked : pandas.DatetimeIndex
        The directory's marked bars, in order.

    Raises
    ------
    ValueError
        If the segment begins after the directory's last marked bar.

    Examples
    --------
    ::

        start = segment_start(config, live_dir, pd.DatetimeIndex(read_equity(live_dir)["timestamp"]))
    """
    try:
        segments = AccountHistory.of(config.venue).segments
    except ValueError:
        segments = ()
    resolved = str(Path(live_dir).resolve())
    own = [segment for segment in segments if segment.live_dir == resolved]
    first = own[-1].first if own else marked[0]
    later = marked[marked >= first]
    if not len(later):
        raise ValueError(f"the strategy segment begins {first.date()}, after the last marked bar")
    return later[0]


def replay_segment(
    config: TraderConfig, run: QuantlabRun, live_dir: Path, start: pd.Timestamp, end: pd.Timestamp
) -> SegmentReplay:
    """Replay ``live_dir``'s segment ``start``..``end`` closed loop and compare it to live.

    The replay sizes each cycle on the ``max_capital`` the live journal
    recorded for it (``read_max_capital``), so a capped segment is replayed
    under each day's cap; a bar the journal does not hold takes the live
    config's. The previous replay's directory (and any trader run directory
    a failed replay left beside it) is removed first, so a failed replay
    leaves no stale one behind.

    Parameters
    ----------
    config : TraderConfig
        The live config: its quantlab run and live prediction store.
    run : QuantlabRun
        The loaded quantlab run (its calendar for the tracking error).
    live_dir : pathlib.Path
        The live run directory.
    start, end : pandas.Timestamp
        The segment's first marked bar and t.

    Returns
    -------
    SegmentReplay

    Raises
    ------
    ValueError
        If the prediction store holds another run's predictions, or the
        replay cannot run (``runner.run``).

    Examples
    --------
    ::

        replay = replay_segment(config, run, live_dir, start, t)
        replay.run_dir  # <live_dir>/briefing/replay
    """
    live_dir = Path(live_dir)
    briefing_dir = live_dir / BRIEFING_DIR
    target = briefing_dir / REPLAY_DIR
    for stale in briefing_dir.glob(f"{REPLAY_DIR}*") if briefing_dir.is_dir() else ():
        shutil.rmtree(stale, ignore_errors=True)
    equity = read_equity(live_dir)["value"].sel(timestamp=slice(start, end))
    init_cash = float(equity.values[0])
    store = LivePredictionStore(config.venue.prediction_store)
    predictions = _store_panel(store, run, start, end)
    replay_config = TraderConfig(
        config.quantlab_run,
        BacktestVenueConfig(ExecutionConfig(init_cash=init_cash)),
        loop=Loop.CLOSED,
        start=bar_label(start),
        end=bar_label(end),
        output_dir=str(briefing_dir),
        tracker=NullTracker(),
        name=REPLAY_DIR,
        max_capital=config.max_capital,
    )
    began = time.perf_counter()
    caps = {
        bar: cap for bar, cap in read_max_capital(live_dir).items() if start <= bar <= end
    }
    run_dir = run_backtest(replay_config, predictions=predictions, max_capital_at=caps)
    seconds = time.perf_counter() - began
    run_dir.rename(target)
    bars = len(equity["timestamp"])
    level = logging.WARNING if seconds > SLOW_REPLAY_SECONDS else logging.INFO
    logger.log(
        level, "closed-loop replay of %s, %s to %s (%d marked bars): %.1f s",
        live_dir, bar_label(start), bar_label(end), bars, seconds,
    )

    held_at_start = _held(live_dir, start)
    (target / RECORD_FILE).write_text(json.dumps({
        "live_dir": str(live_dir),
        "quantlab_run": config.quantlab_run,
        "prediction_store": str(store.path),
        "start": bar_label(start),
        "end": bar_label(end),
        "init_cash": init_cash,
        "held_at_start": held_at_start,
        "trader_run": run_dir.name,
        "seconds": seconds,
    }, indent=2))

    replayed = read_equity(target)["value"].reindex(timestamp=equity["timestamp"].values)
    curve = xr.Dataset({"live_value": equity, "replay_value": replayed})
    live_returns, replay_returns = (
        _returns(curve[name], init_cash) for name in ("live_value", "replay_value")
    )
    timestamps = pd.DatetimeIndex(equity["timestamp"].values)
    bar_interval, year_freq = bar_clock(
        timestamps, run.trading_days_per_year, run.session_minutes_per_day
    )
    relative = backtest_stats.relative_stats(
        live_returns, replay_returns, bar_interval=bar_interval, year_freq=year_freq,
        ranges=[(bar_label(start), bar_label(end))],
    )
    unfilled_live, fees_live = _orders(live_dir, start, end)
    unfilled_replay, fees_replay = _orders(target, start, end)
    return SegmentReplay(
        start=start,
        end=end,
        run_dir=target,
        init_cash=init_cash,
        held_at_start=held_at_start,
        curve=curve,
        live_return_pct=relative["Strategy Total Return [%]"],
        replay_return_pct=relative["Benchmark Total Return [%]"],
        gap_pct=relative["Total Return Difference [%]"],
        tracking_error_pct=relative["Tracking Error [%]"],
        unfilled_live=unfilled_live,
        unfilled_replay=unfilled_replay,
        fees_live=fees_live,
        fees_replay=fees_replay,
        seconds=seconds,
    )


def replay_curve(curve: xr.Dataset, replay: SegmentReplay) -> xr.Dataset:
    """Return the briefing's ``curve`` with ``replay_cumulative_return`` on its axis.

    The replay is rebased to the live curve's cumulative return at the
    segment's start and is NaN before it.

    Examples
    --------
    ::

        curve = replay_curve(curve, replay_segment(config, run, live_dir, start, t))
    """
    timestamps = curve["timestamp"].values
    base = 1.0 + float(curve["cumulative_return"].sel(timestamp=replay.start))
    growth = replay.curve["replay_value"] / replay.init_cash
    cumulative = (base * growth - 1.0).reindex(timestamp=timestamps)
    return curve.assign(replay_cumulative_return=("timestamp", cumulative.values))


def replay_section(replay: SegmentReplay | None, failure: str | None = None) -> tuple:
    """Return the briefing's section of the replay: ``(title, ((label, text), ...))``.

    Without a replay, one row says why it failed.

    Examples
    --------
    >>> replay_section(None, "no prediction row")
    ('闭环回放 (Closed-loop replay)', (('回放 (Replay)', '失败 (failed): no prediction row'),))
    """
    title = "闭环回放 (Closed-loop replay)"
    if replay is None:
        return title, (("回放 (Replay)", f"失败 (failed): {failure}"),)
    r = replay
    cash = f"{r.init_cash:,.2f} USD"
    start = (
        f"空账户 (empty account), 现金 {cash}"
        if r.empty_start
        else f"账户非空 (non-empty account): 实盘持有 {r.held_at_start} 只证券, 回放从 {cash} 现金开始"
    )
    return (
        f"{title}: {r.start.date()} 至 {r.end.date()}, 与实盘之差即执行差异",
        (
            ("起点 (Start)", start),
            ("实盘累计收益 (Live Return)", _signed(r.live_return_pct, "%")),
            ("回放累计收益 (Replay Return)", _signed(r.replay_return_pct, "%")),
            ("累计收益差 (Return Gap)", _signed(r.gap_pct, " pp")),
            ("跟踪误差 (Tracking Error)", "暂无" if np.isnan(r.tracking_error_pct) else f"{r.tracking_error_pct:.2f}%"),
            ("未成交订单 (Unfilled Orders)", f"实盘 {r.unfilled_live} / 回放 {r.unfilled_replay}"),
            ("累计费用 (Fees)", f"实盘 {r.fees_live:,.2f} USD / 回放 {r.fees_replay:,.2f} USD"),
        ),
    )


def _signed(value: float, unit: str) -> str:
    """``value`` signed to two decimals; a value rounding to zero shows as +0.00."""
    if np.isnan(value):
        return "暂无"
    return f"{round(value, 2) + 0.0:+.2f}{unit}"


def _store_panel(
    store: LivePredictionStore, run: QuantlabRun, start: pd.Timestamp, end: pd.Timestamp
) -> xr.Dataset:
    """The live prediction store's rows ``start``..``end``, one variable per label."""
    owner = store.attrs().get("run_dir")
    if owner is not None and Path(owner).resolve() != Path(run.run_dir).resolve():
        raise ValueError(f"{store.path} holds the live predictions of {owner}, not of {run.run_dir}")
    with xr.open_zarr(store.path) as data:
        panel = data.sel(timestamp=slice(start, end)).load()
    panel.attrs = {}
    return panel.drop_vars([c for c in panel.coords if c not in ("timestamp", "symbol")])


def _returns(value: xr.DataArray, init_cash: float) -> xr.DataArray:
    """Per-bar returns of ``value``, the first against ``init_cash``."""
    previous = value.shift(timestamp=1)
    previous[0] = init_cash
    return value / previous - 1.0


def _held(live_dir: Path, t: pd.Timestamp) -> int:
    """The securities held at ``t``'s close (``holdings.zarr``)."""
    with xr.open_zarr(live_dir / "holdings.zarr") as holdings:
        row = holdings["holding"].sel(timestamp=t).values
    return int(np.count_nonzero(np.isfinite(row) & (row != 0.0)))


def _orders(run_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, float]:
    """The unfilled orders and the fees of the orders decided from ``start`` to before ``end``.

    An order decided on ``end`` (t) fills at t+1's open, after the close the
    briefing reports, and the replay never decides t (#73).
    """
    with xr.open_zarr(run_dir / "orders.zarr") as orders:
        dates = pd.DatetimeIndex(orders["decision_date"].values)
        decided = (dates >= start) & (dates < end)
        status = orders["status"].values[decided]
        fees = np.asarray(orders["fee"].values[decided], dtype=float)
    return int(np.isin(status, UNFILLED_STATUSES).sum()), float(np.nansum(fees))
