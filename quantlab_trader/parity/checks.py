"""The ladder's end checks: L0 against the run, T against L5 (ADR 0007)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab_trader.parity.market import PRICE_DECIMALS
from quantlab_trader.parity.rung import RungResult


#: The relative tolerance of L0 against the run (quantlab's cross-platform anchor).
L0_RTOL = 1e-12


def l0_check(l0: RungResult, run_value: xr.DataArray) -> dict:
    """L0 against the run's ``equity.zarr``: the largest relative error per bar.

    Examples
    --------
    >>> from quantlab_trader.parity.rung import RungResult
    >>> bars = pd.bdate_range("2024-01-02", periods=2)
    >>> l0 = RungResult("L0", pd.Series([1e6, 1.01e6], index=bars), 1e6, pd.DataFrame())
    >>> recorded = xr.DataArray([1e6, 1.01e6], dims="timestamp", coords={"timestamp": bars})
    >>> l0_check(l0, recorded)
    {'passed': True, 'max_rel_error': 0.0, 'tolerance': 1e-12}
    """
    expected = run_value.to_pandas()
    got = l0.equity.reindex(expected.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.abs(got.to_numpy() - expected.to_numpy()) / np.abs(expected.to_numpy())
    worst = float(np.nanmax(error)) if np.isfinite(got).all() else math.inf
    return {"passed": bool(worst <= L0_RTOL), "max_rel_error": worst, "tolerance": L0_RTOL}


def t_check(t: RungResult, l5: RungResult, trader_dir: Path) -> dict:
    """Check T against L5 (ADR 0007).

    The same orders, fill prices at 4 decimals, fees to the cent,
    rejections and settlements; equity within USD 0.01 per fill so far.

    The fills counted are the next-open orders' and the venue fills
    (splits, delisting settlements) of the trader run.

    Examples
    --------
    T against L5, with T's run directory for its venue fills::

        check = t_check(rungs["T"], rungs["L5"], trader_dir)
        check["passed"], check["equity_max_error"]
    """
    t_orders, l5_orders = sorted_orders(t.orders), sorted_orders(l5.orders)
    keys = ["fill_bar", "symbol", "side", "quantity"]
    orders_equal = len(t_orders) == len(l5_orders) and all(
        (t_orders[k].to_numpy() == l5_orders[k].to_numpy()).all() for k in keys
    )
    price_error = fee_error = None
    if orders_equal and len(t_orders):
        price_error = float(
            np.max(np.abs(t_orders["price"].round(PRICE_DECIMALS) - l5_orders["price"].round(PRICE_DECIMALS)))
        )
        fee_error = float(np.max(np.abs(t_orders["fee"] - l5_orders["fee"])))
    rejections_equal = _keyed(t.rejected) == _keyed(l5.rejected)
    settlements_equal = _keyed_settlements(t.settlements) == _keyed_settlements(l5.settlements)

    events = json.loads((trader_dir / "events.json").read_text())["events"]
    fill_days = list(pd.DatetimeIndex(t.orders["fill_bar"])) + [
        pd.Timestamp(e["timestamp"])
        for e in events
        if e.get("type") == "corporate_action" and "side" in e
    ]
    timestamps = t.equity.index
    fills = np.zeros(len(timestamps))
    np.add.at(fills, timestamps.searchsorted(pd.DatetimeIndex(fill_days)), 1)
    tolerance = 0.01 * np.cumsum(fills)
    error = np.abs(t.equity.to_numpy() - l5.equity.reindex(timestamps).to_numpy())
    worst = int(np.nanargmax(error)) if np.isfinite(error).any() else 0
    equity_ok = bool(np.isfinite(error).all() and (error <= tolerance + _EQUITY_SLACK).all())
    passed = (
        orders_equal
        and (price_error is None or price_error == 0.0)
        and (fee_error is None or fee_error < 0.005)
        and rejections_equal
        and settlements_equal
        and equity_ok
    )
    return {
        "passed": bool(passed),
        "orders_equal": bool(orders_equal),
        "orders": [len(t_orders), len(l5_orders)],
        "fill_price_max_error": price_error,
        "fee_max_error": fee_error,
        "rejections_equal": bool(rejections_equal),
        "settlements_equal": bool(settlements_equal),
        "equity_within_tolerance": equity_ok,
        "equity_max_error": float(error[worst]),
        "equity_tolerance_at_max": float(tolerance[worst]),
    }


#: Float slack on the equity tolerance (USD), for a bar with no fill yet.
_EQUITY_SLACK = 1e-6


def sorted_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Orders keyed for comparison: symbols as strings, whole quantities, in a fixed order.

    Examples
    --------
    >>> from quantlab_trader.parity.rung import orders_frame
    >>> day = pd.Timestamp("2024-01-03")
    >>> rows = [(day, 10002, "SELL", 5.0000001, 40.0, 0.2), (day, 10001, "BUY", 10.0, 50.0, 0.5)]
    >>> sorted_orders(orders_frame(rows))[["symbol", "side", "quantity"]]
      symbol  side  quantity
    0  10001   BUY      10.0
    1  10002  SELL       5.0
    """
    if not len(orders):
        return pd.DataFrame(columns=["fill_bar", "symbol", "side", "quantity", "price", "fee"])
    keyed = orders.assign(
        symbol=orders["symbol"].map(str),
        quantity=orders["quantity"].round(6),
    )
    return keyed.sort_values(["fill_bar", "symbol", "side"], kind="stable").reset_index(drop=True)


def _keyed(pairs) -> list:
    """``(bar, symbol)`` pairs as sorted comparable keys."""
    return sorted((pd.Timestamp(bar), str(symbol)) for bar, symbol in pairs)


def _keyed_settlements(settlements) -> list:
    """Settlements as sorted keys: bar, symbol, price at 4 decimals, whole quantity."""
    return sorted(
        (pd.Timestamp(bar), str(symbol), round(float(price), PRICE_DECIMALS), round(float(q), 6))
        for bar, symbol, price, q in settlements
    )
