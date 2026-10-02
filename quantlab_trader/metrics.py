"""A trader run's ``metrics.json``, in quantlab's layout and names (ADR 0007).

``run_metrics`` turns what a trader run recorded (equity, fills, orders,
decisions and events) into the blocks a quantlab run's ``metrics.json``
has, computed by quantlab's own public statistics
(``quantlab.utils.backtest_stats``) so the numbers are comparable:

- ``whole``: the return statistics over the trader window, ``Start Value``
  and ``End Value``, ``Total Orders`` (fills of next-open orders), ``Total Fees
  Paid``, ``Traded Notional``, the position-level trade counts, the three
  turnover rows and the win rates;
- ``in_sample`` / ``out_of_sample``: the same over the quantlab run's ranges,
  cut to the trader window, ``None`` where nothing is left; a run without a
  model (``run_weights()``) has neither, as in quantlab;
- ``execution``: ``rejected_order_count`` and ``rejected_orders`` (next-open
  orders with a fill bar in the window that did not fill; the holding was
  kept), ``max_target_deviation``, ``settlements`` (delisted holdings settled
  into cash), and ``trader``, the facts quantlab has no name for:
  commissions, minimum-fee hits, dividends, splits, implied splits (#27), value distributions and
  the peak cash debit;
- ``portfolio_construction`` (closed loop): the held bars and the rule's
  events, as quantlab's cross-section backtester records them;
- ``benchmark`` / ``relative``: against the quantlab run's benchmark curve
  on the trader's bars, when the run had one;
- the quantlab run's split keys, cut to the trader window, and ``notes``.

``max_target_deviation`` is quantlab's valuation-basis definition, because
trader sizes against t's close: the largest ``|w - held weight|`` over every
finite target of a decision with a fill bar, the held weight being the
position after the fill bar valued at t's raw close over t's close-marked
equity. Whole-share rounding, unfilled orders and partial fills show in it;
settled securities are left out.

This module imports no nautilus code.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.utils import backtest_stats
from quantlab_trader._support.jsonable import python_scalar

#: Order statuses of a next-open order that ended without a fill.
UNFILLED_STATUSES = ("unfilled", "rejected", "denied")

#: Corporate actions that move cash without a fill, and the trader block's
#: name for each.
CASH_ACTIONS = {
    "DIVIDEND": "dividends",
    "CASH_IN_LIEU": "cash_in_lieu",
    "DISTRIBUTION": "value_distributions",
}


@dataclass(frozen=True)
class CycleRecord:
    """One decision cycle as the metrics read it.

    Attributes
    ----------
    timestamp : pandas.Timestamp
        The decision date t.
    weights : pandas.Series or None
        The decided weights per PERMNO (NaN keeps the holding); ``None`` on
        a bar that did not rebalance.
    current_weights : pandas.Series
        Each holding's weight at t's raw close before the decision.
    equity : float
        Equity at t's raw close.

    Examples
    --------
    >>> record = CycleRecord(
    ...     timestamp=pd.Timestamp("2024-01-03"),
    ...     weights=pd.Series({10001: 0.5, 10002: float("nan")}),
    ...     current_weights=pd.Series({10002: 0.4}),
    ...     equity=1_000_000.0,
    ... )
    >>> record.weights.isna().sum()  # one holding kept
    np.int64(1)
    """

    timestamp: pd.Timestamp
    weights: pd.Series | None
    current_weights: pd.Series
    equity: float


def bar_label(value) -> str:
    """Return quantlab's label of a bar: an ISO date at midnight, else an ISO timestamp.

    Examples
    --------
    >>> bar_label(pd.Timestamp("2024-01-02")), bar_label(pd.Timestamp("2024-01-02 15:30"))
    ('2024-01-02', '2024-01-02T15:30:00')
    """
    ts = pd.Timestamp(value)
    return ts.strftime("%Y-%m-%d") if ts == ts.normalize() else ts.isoformat()


def window_benchmark(returns: xr.DataArray, timestamps) -> xr.DataArray:
    """Return the benchmark's per-bar returns over the trader's bars, held from the first close.

    The trader's book starts in cash at the close of its first bar, so the
    benchmark's first return in the window is 0, as quantlab's own is on the
    run's first bar; the rest are the quantlab run's.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-02", periods=3)
    >>> run_returns = xr.DataArray([0.0, 0.01, 0.02], dims="timestamp", coords={"timestamp": bars})
    >>> window_benchmark(run_returns, bars[1:]).values.tolist()
    [0.0, 0.02]
    """
    window = returns.sel(timestamp=timestamps).copy()
    window[0] = 0.0
    return window


def run_metrics(
    *,
    equity: xr.Dataset,
    fills: xr.Dataset,
    orders: xr.Dataset,
    cycles: Sequence[CycleRecord],
    events: Sequence[Mapping],
    closes: pd.DataFrame,
    init_cash: float,
    bar_interval,
    year_freq: pd.Timedelta,
    rebalance_periods: int,
    split: Mapping,
    benchmark: Mapping | None,
    closed_loop: bool,
    notes: Sequence[str] = (),
) -> dict:
    """Return a trader run's metrics, quantlab's ``metrics.json`` layout.

    Parameters
    ----------
    equity : xarray.Dataset
        ``value`` and ``returns`` on ``timestamp``, the run's ``equity.zarr``.
    fills : xarray.Dataset
        One fill of a next-open order per entry of ``fill``: ``timestamp``
        (its bar), ``symbol``, ``size`` (signed shares), ``price``, ``fee``
        and ``minimum_fee`` (the venue charged its minimum).
    orders : xarray.Dataset
        The run's ``orders.zarr`` plus ``decided_quantity``, the quantity the
        decision sized (before a split rescaled it).
    cycles : sequence of CycleRecord
        Every decision cycle of the run, one per bar.
    events : sequence of dict
        The run's ``events.json`` events.
    closes : pandas.DataFrame
        Raw closes, decision dates by PERMNO, carried forward.
    init_cash : float
        Starting cash.
    bar_interval
        The bar spacing.
    year_freq : pandas.Timedelta
        One year, ``backtest_stats.year_freq``.
    rebalance_periods : int
        Bars between rebalances.
    split : Mapping
        The quantlab run's split keys (``QuantlabRun.split``).
    benchmark : Mapping or None
        ``QuantlabRun.benchmark()``: ``returns``, ``symbol``, ``axis_symbol``.
    closed_loop : bool
        Whether the run decided with quantlab's rule (adds
        ``portfolio_construction``).
    notes : sequence of str
        The run's notes.

    Returns
    -------
    dict
        The metrics, with the blocks the module docstring lists; values are
        Python and pandas scalars (timestamps, timedeltas, NaN), to be made
        JSON-safe by the writer.

    Examples
    --------
    ``RunRecorder.write`` computes them from what it collected and the
    quantlab run::

        metrics = run_metrics(
            equity=equity, fills=fills, orders=orders, cycles=cycles, events=events,
            closes=closes, init_cash=1_000_000.0, bar_interval=pd.Timedelta("1D"),
            year_freq=backtest_stats.year_freq("1D", 252, 390), rebalance_periods=5,
            split=run.split(), benchmark=run.benchmark(), closed_loop=True, notes=NOTES,
        )
        metrics["whole"]["Total Return [%]"]
    """
    stats = _Stats(equity, fills, init_cash, bar_interval, year_freq, rebalance_periods)
    timestamps = equity["timestamp"].values
    calendar = pd.DatetimeIndex(timestamps)
    whole_range = [(bar_label(timestamps[0]), bar_label(timestamps[-1]))]
    trades = _trades(fills, events)

    whole = {
        **stats.returns(whole_range),
        "Start Value": float(init_cash),
        "End Value": float(equity["value"].values[-1]),
        **stats.records(whole_range),
        "Total Trades": len(trades),
        **_trade_counts(trades, whole_range),
        **stats.win_rates(whole_range),
    }
    settlements = _settlements(events, calendar)
    metrics: dict = {
        "whole": whole,
        "execution": {
            **_rejected_orders(orders, calendar),
            "max_target_deviation": _max_target_deviation(
                cycles, orders, closes, _settled(events), calendar
            ),
            "settlements": settlements,
            "trader": _trader_facts(fills, events, cycles),
        },
    }

    cut = _cut_split(split, calendar)
    slices: dict[str, list] = {}
    if "out_of_sample_ranges" in cut:
        in_sample = cut.get("in_sample_ranges")
        if in_sample is None:
            in_sample = [cut["in_sample_range"]] if cut.get("in_sample_range") else []
        slices = {"in_sample": in_sample, "out_of_sample": cut["out_of_sample_ranges"]}
    for name, ranges in slices.items():
        metrics[name] = (
            {
                **stats.returns(ranges),
                **stats.records(ranges),
                **_trade_counts(trades, ranges),
                **stats.win_rates(ranges),
            }
            if ranges
            else None
        )

    if benchmark is not None:
        bench = window_benchmark(benchmark["returns"], timestamps)
        metrics["benchmark"] = {
            "symbol": benchmark.get("symbol"),
            "axis_symbol": benchmark.get("axis_symbol"),
            **{
                name: stats.benchmark_returns(bench, ranges) if ranges else None
                for name, ranges in {"whole": whole_range, **slices}.items()
            },
        }
        metrics["relative"] = {
            name: stats.relative(bench, ranges) if ranges else None
            for name, ranges in {"whole": whole_range, **slices}.items()
        }
    if closed_loop:
        metrics["portfolio_construction"] = _portfolio_construction(events)
    metrics.update(cut)
    metrics["notes"] = list(notes)
    return metrics


class _Stats:
    """quantlab's statistics of one run, bound to its series and conventions."""

    def __init__(self, equity, fills, init_cash, bar_interval, year_freq, rebalance_periods):
        self.value = equity["value"]
        self.returns_series = equity["returns"]
        self.fills = fills
        self.bar_interval = pd.Timedelta(bar_interval)
        self.year_freq = year_freq
        self.rebalance_periods = rebalance_periods
        self.turnover = backtest_stats.turnover(
            xr.Dataset(
                {name: ("order", fills[name].values) for name in ("timestamp", "size", "price")}
            )
            if fills.sizes.get("fill", 0)
            else xr.Dataset(),
            self.value,
            init_cash,
        )

    def returns(self, ranges) -> dict:
        return backtest_stats.return_stats(
            self.returns_series,
            bar_interval=self.bar_interval,
            year_freq=self.year_freq,
            ranges=ranges,
        )

    def benchmark_returns(self, bench, ranges) -> dict:
        return backtest_stats.return_stats(
            bench, bar_interval=self.bar_interval, year_freq=self.year_freq, ranges=ranges
        )

    def records(self, ranges) -> dict:
        """``Total Orders``, ``Total Fees Paid``, ``Traded Notional`` and turnover in ``ranges``."""
        fills = self.fills
        if fills.sizes.get("fill", 0):
            inside = backtest_stats.in_ranges(fills["timestamp"].values, ranges)
            size = np.abs(fills["size"].values.astype(float))[inside]
            price = fills["price"].values.astype(float)[inside]
            # quantlab counts its fills (one order record each).
            orders = int(inside.sum())
            fees = float(fills["fee"].values[inside].sum())
            notional = float((size * price).sum())
        else:
            orders, fees, notional = 0, 0.0, 0.0
        turnover = self.turnover.isel(
            timestamp=backtest_stats.in_ranges(self.turnover["timestamp"].values, ranges)
        )
        return {
            "Total Orders": orders,
            "Total Fees Paid": fees,
            "Traded Notional": notional,
            **backtest_stats.turnover_stats(
                turnover,
                bar_interval=self.bar_interval,
                year_freq=self.year_freq,
                rebalance_periods=self.rebalance_periods,
            ),
        }

    def win_rates(self, ranges, bench=None) -> dict:
        return backtest_stats.win_rates(
            self.returns_series,
            self.fills["timestamp"].values
            if self.fills.sizes.get("fill", 0)
            else np.array([], dtype="datetime64[ns]"),
            ranges=ranges,
            benchmark_returns=bench,
        )

    def relative(self, bench, ranges) -> dict:
        return {
            **backtest_stats.relative_stats(
                self.returns_series,
                bench,
                bar_interval=self.bar_interval,
                year_freq=self.year_freq,
                ranges=ranges,
            ),
            **self.win_rates(ranges, bench),
        }


def _cut_split(split: Mapping, calendar: pd.DatetimeIndex) -> dict:
    """Return the split keys with every range cut to the trader's bars.

    A range is narrowed to its first and last trader bar and dropped when
    none is left; a singular range that empties becomes ``None``. Training
    windows are records of the model and are copied as they are.
    """

    def cut(pair):
        if pair is None:
            return None
        inside = calendar[
            (calendar >= pd.Timestamp(backtest_stats.label_ns(pair[0])))
            & (calendar <= pd.Timestamp(backtest_stats.label_ns(pair[1])))
        ]
        return [bar_label(inside[0]), bar_label(inside[-1])] if len(inside) else None

    out = {}
    for key, value in split.items():
        if key.startswith("training_window"):
            out[key] = value
        elif key == "in_sample_range":
            out[key] = cut(value)
        else:
            out[key] = [pair for pair in (cut(v) for v in value) if pair is not None]
    return out


def _fill_bar(decision_date, calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Return the bar after ``decision_date`` in ``calendar``, or ``None`` on the last bar."""
    position = calendar.searchsorted(pd.Timestamp(decision_date), side="right")
    return calendar[position] if position < len(calendar) else None


def _rejected_orders(orders: xr.Dataset, calendar: pd.DatetimeIndex) -> dict:
    """quantlab's rejected orders: an order with a fill bar that did not fill; the holding was kept."""
    rejected = []
    for row in range(orders.sizes.get("order", 0)):
        if str(orders["status"].values[row]) not in UNFILLED_STATUSES:
            continue
        fill_bar = _fill_bar(orders["decision_date"].values[row], calendar)
        if fill_bar is None:
            continue
        symbol = str(python_scalar(orders["symbol"].values[row]))
        rejected.append(
            {
                "symbol": symbol,
                "axis_symbol": symbol,
                "signal_timestamp": pd.Timestamp(orders["decision_date"].values[row]).isoformat(),
                "fill_timestamp": fill_bar.isoformat(),
                "reason": str(orders["reason"].values[row]),
            }
        )
    return {"rejected_order_count": len(rejected), "rejected_orders": rejected}


def _settlements(events: Sequence[Mapping], calendar: pd.DatetimeIndex) -> list[dict]:
    """quantlab's settlement records of the delisting venue fills."""
    out = []
    for event in events:
        if event.get("type") != "corporate_action" or event.get("action") != "DELIST":
            continue
        settled = pd.Timestamp(event["timestamp"])
        position = calendar.searchsorted(settled)
        symbol = str(event["symbol"])
        out.append(
            {
                "symbol": symbol,
                "axis_symbol": symbol,
                "delisting_timestamp": calendar[position - 1].isoformat() if position else None,
                "settlement_timestamp": settled.isoformat(),
                "price": float(event["price"]),
                "quantity": int(event["quantity"]),
            }
        )
    return out


def _settled(events: Sequence[Mapping]) -> set:
    """``(settlement bar, PERMNO)`` of every delisting settlement."""
    return {
        (pd.Timestamp(e["timestamp"]), e["symbol"])
        for e in events
        if e.get("type") == "corporate_action" and e.get("action") == "DELIST"
    }


def _max_target_deviation(
    cycles: Sequence[CycleRecord],
    orders: xr.Dataset,
    closes: pd.DataFrame,
    settled: set,
    calendar: pd.DatetimeIndex,
) -> float | None:
    """The largest ``|w - held weight|`` after a fill bar, valued at t's close (module docstring)."""
    filled: dict[tuple[pd.Timestamp, Hashable], float] = {}
    for row in range(orders.sizes.get("order", 0)):
        quantity = int(orders["quantity"].values[row])
        if quantity == 0:
            continue
        # Back to the decided shares: a split may have rescaled the order.
        shares = orders["filled_quantity"].values[row] * (
            orders["decided_quantity"].values[row] / quantity
        )
        sign = 1.0 if str(orders["side"].values[row]) == "BUY" else -1.0
        key = (pd.Timestamp(orders["decision_date"].values[row]), python_scalar(orders["symbol"].values[row]))
        filled[key] = filled.get(key, 0.0) + sign * float(shares)

    gaps = []
    for cycle in cycles:
        fill_bar = _fill_bar(cycle.timestamp, calendar)
        if cycle.weights is None or fill_bar is None:
            continue
        for permno, weight in cycle.weights.items():
            if not np.isfinite(weight) or (fill_bar, permno) in settled:
                continue
            held = float(cycle.current_weights.get(permno, 0.0))
            traded = filled.get((cycle.timestamp, permno), 0.0)
            if traded:  # an order was sized, so the close exists
                held += traded * float(closes.at[cycle.timestamp, permno]) / cycle.equity
            gaps.append(abs(weight - held))
    return max(gaps) if gaps else None


def _trader_facts(fills: xr.Dataset, events: Sequence[Mapping], cycles: Sequence[CycleRecord]) -> dict:
    """The ``execution.trader`` block: facts quantlab's engine has no name for."""
    facts: dict = {
        "commissions": float(fills["fee"].values.sum()) if fills.sizes.get("fill", 0) else 0.0,
        "minimum_fee_hits": int(fills["minimum_fee"].values.sum())
        if fills.sizes.get("fill", 0)
        else 0,
    }
    for name in CASH_ACTIONS.values():
        facts[name] = {"count": 0, "amount": 0.0}
    facts["splits"] = {"count": 0, "share_change": 0}
    facts["implied_splits"] = {"count": 0, "share_change": 0}
    for event in events:
        if event.get("type") != "corporate_action":
            continue
        action = event.get("action")
        if action in CASH_ACTIONS:
            block = facts[CASH_ACTIONS[action]]
            block["count"] += 1
            block["amount"] = round(block["amount"] + float(event["amount"]), 2)
        elif action in ("SPLIT", "IMPLIED_SPLIT"):
            block = facts["splits" if action == "SPLIT" else "implied_splits"]
            block["count"] += 1
            sign = 1 if event["side"] == "BUY" else -1
            block["share_change"] += sign * int(event["quantity"])
    cash = [c.equity * (1.0 - float(c.current_weights.sum())) for c in cycles]
    facts["peak_cash_debit"] = max(0.0, -min(cash)) if cash else 0.0
    return facts


def _portfolio_construction(events: Sequence[Mapping]) -> dict:
    """quantlab's ``portfolio_construction`` block from the run's hold and rule events."""
    failed = [
        pd.Timestamp(e["timestamp"]).isoformat() for e in events if e.get("type") == "hold"
    ]
    block: dict = {"failed_bar_count": len(failed), "failed_bars": failed}
    for event in events:
        if event.get("type") != "rule_event":
            continue
        record = {"bar": pd.Timestamp(event["timestamp"]).isoformat()}
        if "count" in event:
            record["count"] = event["count"]
        else:
            record["symbols"] = list(event["symbols"])
        entry = block.setdefault(event["name"], {"count": 0, "bars": []})
        entry["count"] += record.get("count", len(record.get("symbols", ())))
        entry["bars"].append(record)
    return block


@dataclass
class _Trade:
    """One position-level round trip: from flat to flat."""

    symbol: Hashable
    entry: pd.Timestamp
    exit: pd.Timestamp | None = None


def _trades(fills: xr.Dataset, events: Sequence[Mapping]) -> list[_Trade]:
    """Position-level round trips of the strategy's fills, splits and delisting settlements."""
    changes: list[tuple[pd.Timestamp, int, Hashable, int]] = []
    for row in range(fills.sizes.get("fill", 0)):
        changes.append(
            (
                pd.Timestamp(fills["timestamp"].values[row]),
                1,
                python_scalar(fills["symbol"].values[row]),
                int(fills["size"].values[row]),
            )
        )
    for event in events:
        if event.get("type") == "corporate_action" and event.get("action") in ("SPLIT", "IMPLIED_SPLIT", "DELIST"):
            sign = 1 if event["side"] == "BUY" else -1
            # Corporate actions act at the open, before the bar's own fills.
            changes.append(
                (pd.Timestamp(event["timestamp"]), 0, event["symbol"], sign * int(event["quantity"]))
            )
    changes.sort(key=lambda c: (c[0], c[1]))
    positions: dict[Hashable, int] = {}
    open_trades: dict[Hashable, _Trade] = {}
    trades: list[_Trade] = []
    for when, _, symbol, delta in changes:
        before = positions.get(symbol, 0)
        after = before + delta
        positions[symbol] = after
        if before != 0 and (after == 0 or np.sign(after) != np.sign(before)):
            open_trades.pop(symbol).exit = when
        if after != 0 and (before == 0 or np.sign(after) != np.sign(before)):
            open_trades[symbol] = _Trade(symbol, when)
            trades.append(open_trades[symbol])
    return trades


def _trade_counts(trades: Sequence[_Trade], ranges) -> dict:
    """quantlab's ``Total Closed Trades`` (exit inside ``ranges``) and ``Total Open Trades`` (open at each range's end)."""
    closed = sum(
        1
        for trade in trades
        if trade.exit is not None
        and backtest_stats.in_ranges(np.array([trade.exit.to_datetime64()]), ranges)[0]
    )
    open_count = 0
    for _, end in ranges:
        end_ts = pd.Timestamp(backtest_stats.label_ns(end))
        open_count += sum(
            1
            for trade in trades
            if trade.entry <= end_ts and (trade.exit is None or trade.exit > end_ts)
        )
    return {"Total Closed Trades": closed, "Total Open Trades": open_count}

