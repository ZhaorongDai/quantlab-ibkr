"""L0 and L1: the run's ``weights.zarr`` re-run through quantlab's own engine.

trader never re-implements vectorbt's sizing, rejections or settlements:
the backtester is rebuilt from the run's ``config.json`` by class path
(which loads quantlab's backtest layer, allowed in this package only, ADR
0008) and asked for ``run_weights``.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.utils.module import load_backtester_from_config
from quantlab_trader.parity.market import Market, ffill, shift
from quantlab_trader.parity.rung import RungResult
from quantlab_trader.parity.vectorbt_ledger import VBT_ABS, VBT_REL
from quantlab_trader.quantlab_run import QuantlabRun


def quantlab_rung(
    run: QuantlabRun, table: xr.DataArray, market: Market, sizing_basis: str, name: str
) -> tuple[RungResult, dict | None]:
    """Re-run the run's ``weights.zarr`` through quantlab's engine; return the rung and fingerprint.

    The backtester is rebuilt from ``config.json`` without its model,
    tracker, benchmark and output directory (``run_weights`` reads none of
    them), on the rebalance table's window, with ``sizing_basis``.

    Examples
    --------
    L0 and L1 of a run::

        run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
        table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
        market = Market.load(run, table)
        l0, fingerprint = quantlab_rung(run, table, market, "fill", "L0")
        l1, _ = quantlab_rung(run, table, market, "valuation", "L1")
        l1.buys_capped <= l0.buys_capped + len(l1.orders)
    """
    config = copy.deepcopy(run.config)
    timestamps = market.timestamps
    config.update(
        model=None,
        model_mode=None,
        checkpoint=None,
        output_dir=None,
        benchmark_dataset=None,
        tracker={"project": None, "name": "quantlab.base.tracking.NullTracker"},
        sizing_basis=sizing_basis,
        start_date=timestamps[0].strftime("%Y-%m-%d"),
        end_date=timestamps[-1].strftime("%Y-%m-%d"),
    )
    backtester = load_backtester_from_config(config, run_dir=run.run_dir)
    simulation = backtester.run_weights(table).simulation
    equity = simulation.value.to_pandas().reindex(timestamps)

    column = {str(s): j for j, s in enumerate(market.symbols)}
    settled = {
        (pd.Timestamp(r["settlement_timestamp"]), column[str(r["axis_symbol"])])
        for r in simulation.settlements
    }
    records = simulation.orders
    bars = timestamps.get_indexer(pd.DatetimeIndex(records["timestamp"].values))
    cols = np.array([column[str(s)] for s in records["symbol"].values], dtype=np.int64)
    signed = np.where(records["side"].values.astype(str) == "Buy", 1.0, -1.0) * np.asarray(
        records["size"].values, dtype=np.float64
    )
    position = np.zeros((len(timestamps), len(market.symbols)))
    np.add.at(position, (bars, cols), signed)
    position = np.cumsum(position, axis=0)
    valuation = ffill(market.valuation)
    cash = equity.to_numpy() - np.nansum(position * valuation, axis=1)

    strategy = np.array(
        [(timestamps[b], c) not in settled for b, c in zip(bars, cols)], dtype=bool
    )
    orders = pd.DataFrame(
        {
            "fill_bar": timestamps[bars[strategy]],
            "symbol": market.symbols[cols[strategy]],
            "side": np.where(signed[strategy] > 0, "BUY", "SELL"),
            "quantity": np.abs(signed[strategy]),
            "price": np.asarray(records["price"].values, dtype=np.float64)[strategy],
            "fee": np.asarray(records["fees"].values, dtype=np.float64)[strategy],
        }
    )
    rung = RungResult(
        name=name,
        equity=equity,
        init_cash=run.init_cash,
        orders=orders,
        rejected=[
            (pd.Timestamp(r["fill_timestamp"]), market.symbols[column[str(r["axis_symbol"])]])
            for r in simulation.rejected_orders
        ],
        settlements=[
            (
                pd.Timestamp(r["settlement_timestamp"]),
                market.symbols[column[str(r["axis_symbol"])]],
                float(r["price"]),
                None,
            )
            for r in simulation.settlements
        ],
        max_target_deviation=simulation.max_target_deviation,
        buys_capped=_capped_buys(
            market, run, sizing_basis, position, cash, bars, cols, signed
        ),
        peak_cash_debit=max(0.0, -float(np.min(cash))),
        positions=pd.DataFrame(position, index=timestamps, columns=pd.Index(market.symbols)),
    )
    return rung, backtester.get_config().get("data_fingerprint")


def _capped_buys(market, run, sizing_basis, position, cash, bars, cols, signed) -> int:
    """Count the buys vectorbt filled short of their target: cut by the cash cap.

    The requested size is vectorbt's ``target * V / sizing price - position``
    with ``V`` the book before the fill bar valued at the sizing prices (the
    fill prices, or the signal bar's valuation prices).
    """
    target = shift(market.weights)
    fill, valuation = ffill(market.fill), ffill(market.valuation)
    sizing = fill if sizing_basis == "fill" else shift(valuation)
    capped = 0
    for b, c, size in zip(bars, cols, signed):
        if size <= 0 or b == 0 or not np.isfinite(target[b, c]):
            continue
        before = position[b - 1]
        held = before != 0
        book = cash[b - 1] + float(np.sum(before[held] * sizing[b, held]))
        requested = target[b, c] * book / sizing[b, c] - before[c]
        if size < requested - VBT_REL * abs(requested) - VBT_ABS:
            capped += 1
    return capped
