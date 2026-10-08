"""The closed-versus-open block: the closed loop's decided weights against the run's table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
import xarray as xr

from quantlab_ibkr.parity.checks import sorted_orders
from quantlab_ibkr.parity.market import Market
from quantlab_ibkr.parity.report import Statistics, delta
from quantlab_ibkr.parity.rung import RungResult
from quantlab_ibkr.quantlab_run import QuantlabRun


class _Independence(NamedTuple):
    """When a rule's decision does not depend on holdings (ADR 0007).

    ``rule`` tests the rebuilt rule; ``unpredicted_label`` names the label
    whose missing prediction on a held symbol makes a bar depend on holdings
    (``None`` when such a symbol is treated like an unheld one).
    """

    rule: Callable[[object], bool]
    unpredicted_label: Callable[[object], str | None]


#: Rules whose decision does not depend on holdings when nothing is locked
#: and nothing holds (ADR 0007), by class path. TopN: a held symbol without a
#: score gets 0.0 like an unheld one. Mean-variance: with neither a turnover
#: penalty nor a ``min_trade`` (a solved change below it is not traded, so
#: the decision keeps current weights) nor a ``candidate_top_k`` (the pool is
#: the top k plus every held symbol), and on a bar where no held symbol lacks
#: an expected-return prediction (it enters the problem with mu = 0, and
#: joins the bar's covariance estimate).
_HOLDING_INDEPENDENT_RULES = {
    "quantlab.portfolio.predefined.top_n.TopNConstructor": _Independence(
        rule=lambda rule: True, unpredicted_label=lambda rule: None
    ),
    "quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer": _Independence(
        rule=lambda rule: (
            float(rule.config.turnover_penalty) == 0.0
            and float(rule.config.min_trade) == 0.0
            and rule.config.candidate_top_k is None
        ),
        unpredicted_label=lambda rule: rule.config.expected_return_label,
    ),
}


def _locked(run: QuantlabRun, positions: pd.DataFrame, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Per bar: does the book hold a position its dataset marks not tradable (a locked one)?"""
    held = positions.columns[(positions.abs() > 1e-9).any(axis=0)]
    if not len(held):
        return np.zeros(len(timestamps), dtype=bool)
    dataset = run.price_dataset
    prices = dataset.panel(timestamps[0], timestamps[-1], symbols=list(held)).load()
    tradable = (
        dataset.tradable_bars(prices, run.market.fill_price_column)
        .transpose("timestamp", "symbol")
        .to_pandas()
        .reindex(index=timestamps, columns=held, fill_value=False)
    )
    holding = positions[held].abs() > 1e-9
    return (holding & ~tradable.astype(bool)).any(axis=1).to_numpy()


def _held_unpredicted(held: pd.DataFrame, predicted: pd.DataFrame) -> pd.Series:
    """Per bar of ``held``: is a symbol held (a nonzero entry) without a finite prediction?"""
    symbols = held.columns[(held.fillna(0.0) != 0.0).any(axis=0)]
    if not len(symbols):
        return pd.Series(False, index=held.index)
    holding = held[symbols].fillna(0.0) != 0.0
    finite = np.isfinite(predicted.reindex(index=held.index, columns=symbols).to_numpy(dtype=np.float64))
    return (holding & ~finite).any(axis=1)


def closed_vs_open(
    run: QuantlabRun,
    table: xr.DataArray,
    market: Market,
    rungs: dict[str, RungResult],
    closed: RungResult,
    closed_dir: Path,
    statistics: Statistics,
) -> dict:
    """The closed-versus-open block: decided weights against ``weights.zarr``, runs against T.

    A rebalance bar of the closed loop is **holding-independent** when the
    rule is one of ``_HOLDING_INDEPENDENT_RULES``, neither book holds (the
    closed loop's hold event, an all-NaN row of the table), neither book
    has a locked position (held after the bar, not tradable at it) and,
    for a mean-variance rule, neither book holds a symbol without a finite
    expected-return prediction at the bar: the quantlab run's book is L0's
    positions, the closed loop's is its trader run's (for the locked test)
    and the current weights its decisions record (for the prediction
    test). The
    weight distance of a bar is the L1 norm of the difference, a NaN
    (keep) counting as 0.

    Examples
    --------
    The block of a run with a prediction panel, given its rungs and closed-loop run::

        block = closed_vs_open(run, table, market, rungs, closed, closed_dir, statistics)
        block["holding_independent_bars_equal"] == block["holding_independent_bars"]
    """
    with xr.open_zarr(closed_dir / "decisions.zarr") as decisions:
        decisions = decisions[["weight", "current_weight"]].transpose("timestamp", "symbol").load()
    decided = decisions["weight"]
    events = json.loads((closed_dir / "events.json").read_text())["events"]
    held_bars = {pd.Timestamp(e["timestamp"]) for e in events if e.get("type") == "hold"}
    rule = run.constructor()
    independence = None if rule is None else _HOLDING_INDEPENDENT_RULES.get(rule.import_path)
    rule_independent = bool(independence and independence.rule(rule))

    timestamps = market.timestamps
    table = table.to_pandas()
    closed_rows = decided.to_pandas()
    symbols = table.columns.union(closed_rows.columns)
    locked_bars = _locked(run, rungs["L0"].positions, timestamps) | _locked(
        run, closed.positions, timestamps
    )
    unpredicted_bars = pd.Series(False, index=closed_rows.index)
    label = independence.unpredicted_label(rule) if rule_independent else None
    if label is not None:
        predicted = (
            run.prediction_panel().predictions[label].transpose("timestamp", "symbol").to_pandas()
        )
        quantlab_book = rungs["L0"].positions.reindex(closed_rows.index)
        unpredicted_bars = _held_unpredicted(
            decisions["current_weight"].to_pandas(), predicted
        ) | _held_unpredicted(quantlab_book, predicted)
    compared = independent = equal = independent_equal = 0
    distances = []
    for t in closed_rows.index.intersection(timestamps):
        i = timestamps.get_loc(t)
        a = closed_rows.loc[t].reindex(symbols).to_numpy(dtype=np.float64)
        b = table.loc[t].reindex(symbols).to_numpy(dtype=np.float64)
        same = bool(np.array_equal(a, b, equal_nan=True))
        holds = t in held_bars or bool(np.isnan(table.loc[t].to_numpy()).all())
        is_independent = (
            rule_independent and not holds and not locked_bars[i] and not unpredicted_bars.loc[t]
        )
        compared += 1
        equal += same
        independent += is_independent
        independent_equal += is_independent and same
        distances.append(float(np.abs(np.nan_to_num(a) - np.nan_to_num(b)).sum()))

    t_rung = rungs["T"]
    t_orders, c_orders = sorted_orders(t_rung.orders), sorted_orders(closed.orders)
    orders_equal = len(t_orders) == len(c_orders) and bool(
        (t_orders.to_numpy() == c_orders.to_numpy()).all()
    )
    closed_row, open_row = statistics.row(closed), statistics.row(t_rung)
    return {
        "rule": None if rule is None else rule.import_path,
        "holding_independent_rule": rule_independent,
        "rebalance_bars_compared": compared,
        "bars_equal": equal,
        "holding_independent_bars": independent,
        "holding_independent_bars_equal": independent_equal,
        "max_weight_l1_distance": max(distances) if distances else None,
        "mean_weight_l1_distance": float(np.mean(distances)) if distances else None,
        "orders_equal": orders_equal,
        "max_equity_difference": float(
            np.max(np.abs(closed.equity.to_numpy() - t_rung.equity.to_numpy()))
        ),
        "closed_loop": closed_row,
        "delta": {
            key: delta(closed_row[key], open_row[key])
            for key in closed_row
            if key != "rung"
        },
    }
