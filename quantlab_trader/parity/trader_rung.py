"""T and the closed loop: a trader run directory in the report's terms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab_trader.metrics import UNFILLED_STATUSES
from quantlab_trader.parity.market import Market
from quantlab_trader.parity.rung import RungResult
from quantlab_trader.venue.backtest.venue import ExecutionConfig


def trader_rung(
    trader_dir: Path, market: Market, execution: ExecutionConfig, name: str = "T"
) -> RungResult:
    """A trader run directory (T, or the closed loop) in the report's terms.

    Examples
    --------
    A trader run of the quantlab run, as T::

        run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
        table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
        market = Market.load(run, table)
        execution = ExecutionConfig(fee_model="fraction", slippage=run.slippage, init_cash=run.init_cash)
        t = trader_rung(Path("runs/WeightsVectorBt_20261001_trader_20261002"), market, execution)
        t.peak_cash_debit
    """
    timestamps = market.timestamps
    with xr.open_zarr(trader_dir / "equity.zarr") as equity:
        value = equity["value"].to_pandas().reindex(timestamps)
    with xr.open_zarr(trader_dir / "orders.zarr") as frame:
        table = frame.load().to_dataframe()
    metrics = json.loads((trader_dir / "metrics.json").read_text())
    events = json.loads((trader_dir / "events.json").read_text())["events"]
    fill_position = timestamps.searchsorted(pd.DatetimeIndex(table["decision_date"]), side="right")
    has_fill_bar = fill_position < len(timestamps)
    fill_bar = pd.DatetimeIndex(
        [timestamps[k] if k < len(timestamps) else pd.NaT for k in fill_position]
    )
    filled = (table["filled_quantity"] > 0).to_numpy()
    orders = pd.DataFrame(
        {
            "fill_bar": fill_bar[filled],
            "symbol": table["symbol"].to_numpy()[filled],
            "side": table["side"].to_numpy(dtype=str)[filled],
            "quantity": table["filled_quantity"].to_numpy(dtype=np.float64)[filled],
            "price": table["fill_price"].to_numpy(dtype=np.float64)[filled],
            "fee": table["fee"].to_numpy(dtype=np.float64)[filled],
        }
    )
    unfilled = table["status"].isin(UNFILLED_STATUSES).to_numpy() & has_fill_bar
    execution_block = metrics["execution"]
    # Holdings after each bar: the strategy's fills plus the venue fills.
    changes = [
        (bar, symbol, quantity if side == "BUY" else -quantity)
        for bar, symbol, side, quantity in zip(
            orders["fill_bar"], orders["symbol"], orders["side"], orders["quantity"]
        )
    ] + [
        (pd.Timestamp(e["timestamp"]), e["symbol"], (1.0 if e["side"] == "BUY" else -1.0) * e["quantity"])
        for e in events
        if e.get("type") == "corporate_action" and "side" in e
    ]
    positions = (
        pd.DataFrame(changes, columns=["timestamp", "symbol", "change"])
        .pivot_table(index="timestamp", columns="symbol", values="change", aggfunc="sum")
        .reindex(timestamps)
        .fillna(0.0)
        .cumsum()
    )
    return RungResult(
        name=name,
        run_dir=trader_dir,
        positions=positions,
        equity=value,
        init_cash=float(execution.init_cash),
        orders=orders,
        rejected=list(zip(fill_bar[unfilled], table["symbol"].to_numpy()[unfilled])),
        settlements=[
            (
                pd.Timestamp(e["timestamp"]),
                e["symbol"],
                float(e["price"]),
                float(e["quantity"] if e["side"] == "SELL" else -e["quantity"]),
            )
            for e in events
            if e.get("type") == "corporate_action" and e.get("action") == "DELIST"
        ],
        max_target_deviation=execution_block.get("max_target_deviation"),
        buys_capped=None,
        peak_cash_debit=float(execution_block["trader"]["peak_cash_debit"]),
    )
