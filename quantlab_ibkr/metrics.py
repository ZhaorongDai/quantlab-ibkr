"""A trader run's ``metrics.json``, in quantlab's layout and names (ADR 0007).

``run_metrics`` turns what a trader run recorded (equity, fills, orders,
decisions and events) into the blocks a quantlab run's ``metrics.json``
has. It builds the run's account record (quantlab's ``AccountRecord``: the
equity, the next-open fills, the round trips, the rebalance record of the
decision cycles, the rejected orders and the largest target deviation) and
the whole rows quantlab's engine reads from its own portfolio, then hands
both to quantlab's ``quantlab.runs.run_metrics.run_metrics``, the function
that assembles a quantlab run's blocks, so the layout and the numbers are
quantlab's by construction (quantlab ADR 0034):

- ``whole``: quantlab's rows in quantlab's order (a vectorbt run's): the return statistics over the trader window, ``Start
  Value`` and ``End Value``, ``Max Gross Exposure [%]``, ``Total Fees Paid``,
  the round-trip trade statistics, the three turnover rows, ``Total Orders``
  (fills of next-open orders) and the win rates;

  the turnover rows and the win rates count the run's decided rebalances
  (``decided_rebalances``: the cycles with a finite target), as quantlab
  counts the decided bars of its rebalance record, never the bars with
  fills; a delisting settlement is no fill here, so it is no turnover;
- ``in_sample`` / ``out_of_sample``: quantlab's slice rows (return
  statistics, ``Total Orders``, ``Total Fees Paid``, ``Traded Notional``, the
  closed and open round trips, turnover, win rates) over the quantlab run's
  ranges, cut to the trader window, ``None`` where nothing is left; a run
  without a model (``run_weights()``) has neither, as in quantlab;
- ``execution``: ``rejected_order_count`` and ``rejected_orders`` (next-open
  orders with a fill bar in the window that did not fill; the holding was
  kept), ``max_target_deviation``, ``settlements`` (delisted holdings settled
  into cash), and ``trader``, the facts quantlab has no name for:
  commissions, minimum-fee hits, dividends, cash in lieu, value
  distributions, splits, implied splits (#27), factor mismatches (#28) and
  the peak cash debit;
- ``benchmark`` / ``relative``: against the quantlab run's benchmark curve
  on the trader's bars, when the run had one;
- the quantlab run's split keys, cut to the trader window;
- ``portfolio_construction`` (closed loop): the held bars and the rule's
  events, as quantlab's cross-section backtester records them, and
  ``notes``.

``max_target_deviation`` is quantlab's valuation-basis definition, because
trader sizes against t's close: the largest ``|w - held weight|`` over every
finite target of a decision with a fill bar, the held weight being the
position after the fill bar valued at t's raw close over t's close-marked
equity. Whole-share rounding, unfilled orders and partial fills show in it;
settled securities are left out.

The round trips are quantlab's (``backtest_stats.round_trips``, glossary
"Round-trip trade") of the run's actual fills: next-open fills and
delisting settlements, which close a trip, the settlement before the bar's
next-open fills. Dividends, value distributions and cash in lieu are the
trip's cash flows on their ex-date bar. Splits and implied splits do not end
a trip: share counts are restated in the units of the window's first bar (a
split taking ``q`` shares to ``q'`` multiplies the security's factor by
``q' / q``; every later fill's size is divided by it and its price and the
close multiplied by it), so values are unchanged; a reverse split leaving
no share closes the trip at price 0, its cash in lieu then the trip's last
flow. A delisted security is marked on its delisting bar at its settlement
price, as the equity is. ``Max Gross Exposure [%]`` is
``backtest_stats.exposure_stats`` of the same fills and closes against the
cash at each close.

This module imports no nautilus code.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.utils import date_range
from quantlab.runs import backtest_stats
from quantlab.runs.run_metrics import AccountRecord, BenchmarkRecord
from quantlab.runs.run_metrics import run_metrics as quantlab_run_metrics
from quantlab_ibkr._support.jsonable import python_scalar

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
    >>> int(record.weights.isna().sum())  # one holding kept
    1
    """

    timestamp: pd.Timestamp
    weights: pd.Series | None
    current_weights: pd.Series
    equity: float


def decided_rebalances(cycles: Sequence[CycleRecord]) -> np.ndarray:
    """Return the bars of the run's decided rebalances: the cycles with a finite target.

    A cycle that did not rebalance (``weights`` is ``None``) or held
    (all NaN: a construction failure, or a table row without a target) is
    not one; a rebalance that placed no order is. These are the decided bars
    of quantlab's rebalance record, which its turnover and win rates count.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-02", periods=3)
    >>> cycles = [
    ...     CycleRecord(bars[0], pd.Series({10001: 1.0}), pd.Series(dtype=float), 1e4),
    ...     CycleRecord(bars[1], None, pd.Series({10001: 1.0}), 1e4),
    ...     CycleRecord(bars[2], pd.Series({10001: float("nan")}), pd.Series({10001: 1.0}), 1e4),
    ... ]
    >>> decided_rebalances(cycles).astype(str).tolist()
    ['2024-01-02T00:00:00.000000000']
    """
    return np.array(
        [
            pd.Timestamp(c.timestamp).to_datetime64()
            for c in cycles
            if c.weights is not None and np.isfinite(c.weights.to_numpy(dtype=float)).any()
        ],
        dtype="datetime64[ns]",
    )


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
    >>> later = pd.bdate_range("2024-01-03", periods=3)
    >>> window_benchmark(run_returns, later).values.tolist()
    [0.0, 0.02, nan]
    """
    # A bar the benchmark has not published yet (a live day marked before
    # the vendor's evening update) is NaN, not an error.
    window = returns.reindex(timestamp=timestamps).copy()
    window[0] = 0.0
    return window

def rebalance_record(cycles: Sequence[CycleRecord], timestamps) -> xr.Dataset:
    """Return the run's rebalance record on ``timestamps``: quantlab's ``scheduled`` and ``decided``.

    A cycle is scheduled when it was a rebalance bar (``weights`` is not
    ``None``) and decided when it returned a finite target
    (``decided_rebalances``); a cycle off ``timestamps`` is left out.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-02", periods=3)
    >>> cycles = [
    ...     CycleRecord(bars[0], pd.Series({10001: 1.0}), pd.Series(dtype=float), 1e4),
    ...     CycleRecord(bars[1], None, pd.Series({10001: 1.0}), 1e4),
    ...     CycleRecord(bars[2], pd.Series({10001: float("nan")}), pd.Series({10001: 1.0}), 1e4),
    ... ]
    >>> record = rebalance_record(cycles, bars.values)
    >>> record["scheduled"].values.tolist(), record["decided"].values.tolist()
    ([True, False, True], [True, False, False])
    """
    calendar = pd.DatetimeIndex(timestamps)
    scheduled = calendar.isin(
        pd.DatetimeIndex([pd.Timestamp(c.timestamp) for c in cycles if c.weights is not None])
    )
    decided = calendar.isin(pd.DatetimeIndex(decided_rebalances(cycles)))
    return xr.Dataset(
        {"scheduled": ("timestamp", scheduled), "decided": ("timestamp", decided)},
        coords={"timestamp": calendar.values},
    )


def _orders(fills: xr.Dataset) -> xr.Dataset:
    """The next-open fills as quantlab's account record takes them: on ``order``, with ``fees``."""
    if not fills.sizes.get("fill", 0):
        return xr.Dataset()
    return xr.Dataset(
        {
            "timestamp": ("order", fills["timestamp"].values),
            "symbol": ("order", fills["symbol"].values),
            "size": ("order", fills["size"].values),
            "price": ("order", fills["price"].values),
            "fees": ("order", fills["fee"].values),
        }
    )


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
        Every decision cycle of the run, one per bar; its decided
        rebalances (``decided_rebalances``) mark the rebalances the
        turnover and the win rates count.
    events : sequence of dict
        The run's ``events.json`` events.
    closes : pandas.DataFrame
        Raw closes, the run's bars by PERMNO, NaN where there was none.
    init_cash : float
        Starting cash.
    bar_interval
        The bar spacing.
    year_freq : pandas.Timedelta
        One year, ``backtest_stats.year_freq``.
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
            year_freq=backtest_stats.year_freq("1D", 252, 390),
            split=run.split(), benchmark=run.benchmark(start, end), closed_loop=True, notes=NOTES,
        )
        metrics["whole"]["Total Return [%]"]
    """
    timestamps = equity["timestamp"].values
    calendar = pd.DatetimeIndex(timestamps)
    whole_range = [(date_range.bar_label(timestamps[0]), date_range.bar_label(timestamps[-1]))]
    executed = _executions(fills, events, closes.reindex(calendar))
    trips = backtest_stats.round_trips(
        executed.fills, executed.close, cash_flows=executed.cash_flows
    )
    cash = xr.DataArray(
        _cash(cycles),
        dims="timestamp",
        coords={"timestamp": [c.timestamp for c in cycles]},
    ).reindex(timestamp=calendar)
    account = AccountRecord(
        value=equity["value"],
        returns=equity["returns"],
        orders=_orders(fills),
        rebalances=rebalance_record(cycles, timestamps),
        init_cash=float(init_cash),
        bar_interval=pd.Timedelta(bar_interval),
        year_freq=year_freq,
        trades=trips,
        rejected_orders=_rejected_orders(orders, calendar),
        max_target_deviation=_max_target_deviation(
            cycles, orders, closes, _settled(events), calendar
        ),
    )

    returns = backtest_stats.return_stats(
        account.returns, bar_interval=account.bar_interval, year_freq=year_freq, ranges=whole_range
    )
    engine_rows = {
        **{name: returns[name] for name in ("Start", "End", "Period")},
        "Start Value": float(init_cash),
        "End Value": float(equity["value"].values[-1]),
        "Total Return [%]": returns["Total Return [%]"],
        **backtest_stats.exposure_stats(executed.fills, executed.close, cash),
        "Total Fees Paid": float(fills["fee"].values.sum()) if fills.sizes.get("fill", 0) else 0.0,
        "Max Drawdown [%]": returns["Max Drawdown [%]"],
        "Max Drawdown Duration": returns["Max Drawdown Duration"],
        **backtest_stats.round_trip_stats(trips, bar_interval=bar_interval),
        **{
            name: returns[name]
            for name in ("Sharpe Ratio", "Calmar Ratio", "Omega Ratio", "Sortino Ratio")
        },
    }
    metrics = quantlab_run_metrics(
        account,
        engine_rows=engine_rows,
        split=_cut_split(split, calendar),
        benchmark=None
        if benchmark is None
        else BenchmarkRecord(
            window_benchmark(benchmark["returns"], timestamps),
            symbol=benchmark.get("symbol"),
            axis_symbol=benchmark.get("axis_symbol"),
        ),
    )
    metrics["execution"]["settlements"] = _settlements(events, calendar)
    metrics["execution"]["trader"] = _trader_facts(fills, events, cycles)
    if closed_loop:
        metrics["portfolio_construction"] = _portfolio_construction(events)
    metrics["notes"] = list(notes)
    return metrics


def _cut(pair, calendar: pd.DatetimeIndex) -> list[str] | None:
    """Return ``pair`` narrowed to its first and last bar of ``calendar``; ``None`` when none is left."""
    if pair is None:
        return None
    inside = calendar[
        (calendar >= pd.Timestamp(date_range.label_ns(pair[0])))
        & (calendar <= pd.Timestamp(date_range.label_ns(pair[1])))
    ]
    if not len(inside):
        return None
    return [date_range.bar_label(inside[0]), date_range.bar_label(inside[-1])]


def _cut_split(split: Mapping, calendar: pd.DatetimeIndex) -> dict:
    """Return the split keys with every range cut to the trader's bars.

    A range is narrowed to its first and last trader bar and dropped when
    none is left; a singular range that empties becomes ``None``. Training
    windows are records of the model and are copied as they are.
    """
    out = {}
    for key, value in split.items():
        if key.startswith("training_window"):
            out[key] = value
        elif key == "in_sample_range":
            out[key] = _cut(value, calendar)
        else:
            out[key] = [pair for pair in (_cut(v, calendar) for v in value) if pair is not None]
    return out


def cut_folds(folds: Sequence[Mapping] | None, timestamps) -> list[dict] | None:
    """Return a ``run_cv()`` run's fold rows with their bars cut to the trader's.

    ``traded`` and ``in_sample_range`` are narrowed to the trader's bars
    (``None`` when none is left); the training window, a record of the
    model, is kept. ``None`` (a run without folds) stays ``None``.

    Parameters
    ----------
    folds : sequence of dict or None
        ``QuantlabRun.report_records()["folds"]``.
    timestamps : array_like of datetime64
        The trader run's bars.

    Returns
    -------
    list of dict or None
        The rows, as quantlab's ``report_windows`` takes them.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-02", periods=4)
    >>> folds = [{"fold": 0, "training_window": ["2023-01-03", "2023-12-29"],
    ...           "traded": ("2023-12-27", "2024-01-03"), "in_sample_range": None}]
    >>> cut_folds(folds, bars.values)[0]["traded"]
    ('2024-01-02', '2024-01-03')
    """
    if folds is None:
        return None
    calendar = pd.DatetimeIndex(timestamps)
    rows = []
    for fold in folds:
        traded = _cut(fold.get("traded"), calendar)
        rows.append(
            {
                **fold,
                "traded": None if traded is None else tuple(traded),
                "in_sample_range": _cut(fold.get("in_sample_range"), calendar),
            }
        )
    return rows


def _fill_bar(decision_date, calendar: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Return the bar after ``decision_date`` in ``calendar``, or ``None`` on the last bar."""
    position = calendar.searchsorted(pd.Timestamp(decision_date), side="right")
    return calendar[position] if position < len(calendar) else None


def _rejected_orders(orders: xr.Dataset, calendar: pd.DatetimeIndex) -> list[dict]:
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
    return rejected


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
    facts["mismatches"] = {"count": 0}
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
        elif action == "MISMATCH":
            facts["mismatches"]["count"] += 1
    cash = _cash(cycles)
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


@dataclass(frozen=True)
class _Executed:
    """The run's executions on one split-adjusted basis, as quantlab's round trips take them."""

    fills: xr.Dataset
    close: xr.DataArray
    cash_flows: xr.Dataset


class _Change(NamedTuple):
    """One change of a holding, in raw shares."""

    bar: pd.Timestamp
    #: 0 for a corporate action (09:30), 1 for a next-open fill after it.
    stage: int
    permno: Hashable
    #: ``"fill"`` (a next-open fill or a delisting settlement) or ``"split"``.
    kind: str
    size: int
    price: float
    fee: float


def _executions(fills: xr.Dataset, events: Sequence[Mapping], closes: pd.DataFrame) -> _Executed:
    """Restate the fills, cash flows and closes split-adjusted (module docstring).

    ``closes`` are the raw closes on the run's bars by PERMNO. Symbols become
    strings, as ``backtest_stats.round_trips`` labels them. Changes of one
    bar are taken corporate actions first, each stage in the order recorded.
    """
    changes = [
        _Change(
            pd.Timestamp(fills["timestamp"].values[row]),
            1,
            python_scalar(fills["symbol"].values[row]),
            "fill",
            int(fills["size"].values[row]),
            float(fills["price"].values[row]),
            float(fills["fee"].values[row]),
        )
        for row in range(fills.sizes.get("fill", 0))
    ]
    flows: list[tuple[pd.Timestamp, str, float]] = []
    settlements: list[tuple[pd.Timestamp, Hashable, float]] = []
    for event in events:
        if event.get("type") != "corporate_action":
            continue
        action, when, permno = event["action"], pd.Timestamp(event["timestamp"]), event["symbol"]
        if action in ("SPLIT", "IMPLIED_SPLIT", "DELIST"):
            sign = 1 if event["side"] == "BUY" else -1
            changes.append(
                _Change(
                    when, 0, permno, "fill" if action == "DELIST" else "split",
                    sign * int(event["quantity"]), float(event["price"]), float(event["fee"]),
                )
            )
            if action == "DELIST":
                settlements.append((when, permno, float(event["price"])))
        elif action in CASH_ACTIONS and event["amount"]:
            flows.append((when, str(permno), float(event["amount"])))
    changes.sort(key=lambda change: (change.bar, change.stage))

    permnos = list(closes.columns) + sorted(
        {c.permno for c in changes} - set(closes.columns), key=str
    )
    close = closes.reindex(columns=permnos).astype(float)
    for when, permno, price in settlements:
        # The bar before the settlement bar is the delisting bar, on which
        # the equity marks the holding at its settlement price.
        position = close.index.searchsorted(when)
        if position:
            close.loc[close.index[position - 1], permno] = price

    factors = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    held: dict[Hashable, int] = {}
    rows: list[tuple[pd.Timestamp, str, float, float, float]] = []
    for change in changes:
        before = held.get(change.permno, 0)
        after = held[change.permno] = before + change.size
        factor = float(factors.at[change.bar, change.permno])
        if change.kind == "fill":
            rows.append(
                (change.bar, str(change.permno), change.size / factor, change.price * factor, change.fee)
            )
        elif after == 0:
            # A reverse split that leaves no share: the trip ends here.
            rows.append((change.bar, str(change.permno), -before / factor, 0.0, 0.0))
        elif before:
            factors.loc[change.bar:, change.permno] *= after / before

    adjusted = close * factors
    timestamp, symbol, size, price, fee = (list(column) for column in zip(*rows)) if rows else ([],) * 5
    flow_timestamp, flow_symbol, amount = (list(column) for column in zip(*flows)) if flows else ([],) * 3
    return _Executed(
        fills=xr.Dataset(
            {
                "timestamp": ("fill", np.array(timestamp, dtype="datetime64[ns]")),
                "symbol": ("fill", np.array(symbol, dtype=str)),
                "size": ("fill", np.array(size, dtype=float)),
                "price": ("fill", np.array(price, dtype=float)),
                "fees": ("fill", np.array(fee, dtype=float)),
            }
        ),
        close=xr.DataArray(
            adjusted.to_numpy(dtype=float),
            dims=("timestamp", "symbol"),
            coords={
                "timestamp": adjusted.index.values,
                "symbol": np.array([str(c) for c in adjusted.columns], dtype=str),
            },
        ),
        cash_flows=xr.Dataset(
            {
                "timestamp": ("flow", np.array(flow_timestamp, dtype="datetime64[ns]")),
                "symbol": ("flow", np.array(flow_symbol, dtype=str)),
                "amount": ("flow", np.array(amount, dtype=float)),
            }
        ),
    )


def _cash(cycles: Sequence[CycleRecord]) -> list[float]:
    """The cash at each cycle's close: equity less the holdings, short-sale proceeds included."""
    return [c.equity * (1.0 - float(c.current_weights.sum())) for c in cycles]
