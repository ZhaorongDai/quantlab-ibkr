"""The **Decision recheck**: quantlab's rule run again on each closed-loop decision's context.

Every closed-loop decision bar is decided again from the run's
``DecisionInputs``, the prediction panel's row of t and the current weights
the trader run recorded as handed to the rule (``decisions.zarr``
``current_weight``), through the closed loop's own ``ConstructorTargets``.
The rechecked weights must equal the decided ones bit for bit (a NaN, a
kept holding, equal to a NaN), and a recorded hold must be rechecked as a
failure with the same message. A bar that is not holding-independent is
checked too, so a decision error is told apart from execution drift on
every rebalance bar. The decision bars' window is read once
(``ConstructorTargets.preloaded``), which decides exactly what reading it
bar by bar decides.

A live run directory (#51) is rechecked the same way, its prediction rows
read from the live prediction store its days decided from
(``predictions=LivePredictionStore(...)``), and the symbols a live day left
out of the rule (``excluded_symbols`` events: no IBKR contract) masked as
that day masked them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs.live_predictions import LivePredictionStore
from quantlab_ibkr._support.jsonable import python_scalar
from quantlab_ibkr.decision import ConstructorTargets
from quantlab_ibkr.quantlab_run import QuantlabRun


def decision_recheck(
    run: QuantlabRun,
    closed_dir: Path,
    *,
    predictions: LivePredictionStore | Callable[[pd.Timestamp], xr.Dataset] | None = None,
) -> dict:
    """Run the Decision recheck of the closed-loop run in ``closed_dir``.

    Parameters
    ----------
    run : QuantlabRun
        The quantlab run the closed loop replayed.
    closed_dir : pathlib.Path
        The closed-loop trader run directory (its ``decisions.zarr`` and
        ``events.json``), or a live run directory.
    predictions : LivePredictionStore or Callable, optional
        Where each bar's prediction row comes from: the live prediction
        store a live run decided from, or ``row(t) -> xarray.Dataset``;
        ``None`` is the run's prediction panel.

    Returns
    -------
    dict
        ``bars_checked``, ``bars_differing`` and ``differing_bars``: per
        differing bar its ``timestamp``, the ``symbols`` that differ (each
        with its ``decided`` and ``rechecked`` weight, ``None`` for NaN),
        and, when the hold differs, ``failure`` with the ``decided`` and
        ``rechecked`` messages (``None`` when the bar was not a hold);
        ``error`` instead when the rule refused the rebuilt context.

    Examples
    --------
    The recheck of a parity report's closed loop::

        recheck = decision_recheck(run, closed_dir)
        recheck["bars_differing"] == 0

    and of a live run directory, on its live prediction store::

        decision_recheck(run, live_dir, predictions=LivePredictionStore(store_path))
    """
    with xr.open_zarr(closed_dir / "decisions.zarr") as decisions:
        decisions = decisions[["weight", "current_weight"]].transpose("timestamp", "symbol").load()
    events = json.loads((closed_dir / "events.json").read_text())["events"]
    failures = {
        pd.Timestamp(e["timestamp"]): e["failure"] for e in events if e.get("type") == "hold"
    }
    excluded: dict[pd.Timestamp, list] = {}
    for e in events:
        if e.get("type") == "excluded_symbols":
            excluded.setdefault(pd.Timestamp(e["timestamp"]), []).extend(e["symbols"])
    timestamps = pd.DatetimeIndex(decisions["timestamp"].values)
    if not len(timestamps):
        return {"bars_checked": 0, "bars_differing": 0, "differing_bars": []}
    targets = ConstructorTargets(run.decision_inputs())
    row_of = _rows(run, predictions)
    decided = decisions["weight"].to_pandas()
    current = decisions["current_weight"].to_pandas()
    differing = []
    with targets.preloaded(timestamps[0], timestamps[-1]):
        for t in timestamps:
            row = _masked(row_of(t), excluded.get(t, ()))
            difference = _recheck_bar(
                t, targets, row, decided.loc[t], current.loc[t], failures.get(t)
            )
            if difference is not None:
                differing.append(difference)
    return {
        "bars_checked": len(timestamps),
        "bars_differing": len(differing),
        "differing_bars": differing,
    }


def _recheck_bar(
    t: pd.Timestamp,
    targets: ConstructorTargets,
    predictions: xr.Dataset,
    decided: pd.Series,
    current: pd.Series,
    failure: str | None,
) -> dict | None:
    """Recheck one bar; return what differs, or ``None`` when the decision is the same."""
    day = t.strftime("%Y-%m-%d")
    try:
        decision = targets.decide(t, predictions, current)
    except ValueError as error:
        return {"timestamp": day, "error": str(error)}
    rechecked = pd.Series(
        decision.weights.values.astype(np.float64),
        index=[python_scalar(v) for v in decision.weights["symbol"].values],
    )
    symbols = decided.index.union(rechecked.index, sort=False)
    a = decided.reindex(symbols).to_numpy(dtype=np.float64)
    b = rechecked.reindex(symbols).to_numpy(dtype=np.float64)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    difference: dict = {"timestamp": day}
    if not same.all():
        difference["symbols"] = [
            {"symbol": python_scalar(s), "decided": _weight(x), "rechecked": _weight(y)}
            for s, x, y, equal in zip(symbols, a, b, same)
            if not equal
        ]
    if decision.failure != failure:
        difference["failure"] = {"decided": failure, "rechecked": decision.failure}
    return difference if len(difference) > 1 else None


def _rows(run: QuantlabRun, predictions) -> Callable[[pd.Timestamp], xr.Dataset]:
    """Return ``row(t)``: the prediction row of bar t, one variable per label on ``symbol``."""
    if predictions is None:
        panel = run.prediction_panel().predictions
        return lambda t: panel.sel(timestamp=t, drop=True)
    if isinstance(predictions, LivePredictionStore):
        store = predictions

        def row(t: pd.Timestamp) -> xr.Dataset:
            found = store.row(t)
            return found.drop_vars([c for c in found.coords if c != "symbol"])

        return row
    return predictions


def _masked(row: xr.Dataset, excluded) -> xr.Dataset:
    """``row`` with the predictions of ``excluded`` symbols masked, as the live source masks them."""
    if not len(excluded):
        return row
    keep = ~np.isin(row["symbol"].values, list(excluded))
    return row.where(xr.DataArray(keep, dims="symbol", coords={"symbol": row["symbol"]}))


def _weight(value: float) -> float | None:
    """A weight for the report: ``None`` for NaN (a kept holding)."""
    return None if np.isnan(value) else float(value)
