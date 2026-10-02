"""The parity report: per-rung statistics, the report directory, ``parity.json`` and ``parity.zarr``."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.utils import backtest_stats
from quantlab_trader._support.jsonable import jsonable
from quantlab_trader.parity.rung import RUNG_DESCRIPTIONS, RUNGS, RungResult
from quantlab_trader.quantlab_run import QuantlabRun


class Statistics:
    """The report's per-rung statistics, by quantlab's public functions and the run's annualization.

    Examples
    --------
    ::

        run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
        table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
        market = Market.load(run, table)
        statistics = Statistics(run, market.timestamps)
        statistics.year_freq
    """

    def __init__(self, run: QuantlabRun, timestamps: pd.DatetimeIndex):
        self.rebalance_periods = run.rebalance_periods
        self.bar_interval = (
            pd.Series(np.diff(timestamps.values)).mode().iloc[0]
            if len(timestamps) > 1
            else pd.Timedelta(days=1)
        )
        self.year_freq = backtest_stats.year_freq(
            self.bar_interval, run.trading_days_per_year, run.session_minutes_per_day
        )

    def row(self, rung: RungResult) -> dict:
        """Return the report row of ``rung``: its statistics and execution counts.

        Examples
        --------
        ::

            row = Statistics(run, market.timestamps).row(rungs["L5"])
            row["final_equity"], row["total_fees"], row["orders"]
        """
        equity = rung.equity
        previous = equity.shift(1)
        previous.iloc[0] = rung.init_cash
        returns = xr.DataArray(
            (equity / previous - 1.0).to_numpy(),
            dims="timestamp",
            coords={"timestamp": equity.index.values},
        )
        stats = backtest_stats.return_stats(
            returns, bar_interval=self.bar_interval, year_freq=self.year_freq
        )
        orders = rung.orders
        flows = (
            xr.Dataset(
                {
                    "timestamp": ("order", pd.DatetimeIndex(orders["fill_bar"]).values),
                    "size": ("order", orders["quantity"].to_numpy(dtype=np.float64)),
                    "price": ("order", orders["price"].to_numpy(dtype=np.float64)),
                }
            )
            if len(orders)
            else xr.Dataset()
        )
        turnover = backtest_stats.turnover_stats(
            backtest_stats.turnover(
                flows,
                xr.DataArray(equity.to_numpy(), dims="timestamp", coords={"timestamp": equity.index.values}),
                rung.init_cash,
            ),
            bar_interval=self.bar_interval,
            year_freq=self.year_freq,
            rebalance_periods=self.rebalance_periods,
        )
        return {
            "rung": rung.name,
            "final_equity": float(equity.iloc[-1]),
            "total_return_pct": stats["Total Return [%]"],
            "annualized_return_pct": stats["Annualized Return [%]"],
            "sharpe_ratio": stats["Sharpe Ratio"],
            "max_drawdown_pct": stats["Max Drawdown [%]"],
            "total_turnover_pct": turnover["Total Turnover [%]"],
            "annualized_turnover_pct": turnover["Annualized Turnover [%]"],
            "total_fees": float(orders["fee"].sum()) if len(orders) else 0.0,
            "orders": int(len(orders)),
            "rejected_orders": len(rung.rejected),
            "settlements": len(rung.settlements),
            "max_target_deviation": rung.max_target_deviation,
            "buys_capped": rung.buys_capped,
            "peak_cash_debit": rung.peak_cash_debit,
        }


def rung_rows(rungs: dict[str, RungResult], statistics: Statistics) -> list[dict]:
    """One row per rung, in ladder order, each with its delta to the rung above.

    Examples
    --------
    ::

        rows = rung_rows(rungs, Statistics(run, market.timestamps))
        [(row["rung"], row.get("delta", {}).get("final_equity")) for row in rows]
    """
    rows, above = [], None
    for name in RUNGS:
        if name not in rungs:
            continue
        row = statistics.row(rungs[name])
        row["description"] = RUNG_DESCRIPTIONS[name]
        if above is not None:
            row["delta"] = {
                key: delta(row[key], above[key])
                for key in row
                if key not in ("rung", "description", "delta")
            }
        rows.append(row)
        above = row
    return rows


def delta(value, above):
    """``value - above``, or ``None`` when either is missing.

    Examples
    --------
    >>> delta(3.0, 1.0), delta(None, 1.0)
    (2.0, None)
    """
    if value is None or above is None:
        return None
    return value - above


def report_dir(run: QuantlabRun, output_dir) -> Path:
    """Return the path of a fresh report directory (not created).

    Examples
    --------
    ::

        report_dir(QuantlabRun.load("runs/WeightsVectorBt_20261001"), "parity")
        # parity/WeightsVectorBt_20261001_parity_<YYYYmmdd_HHMMSS_ffffff>
    """
    output_dir = Path(output_dir) if output_dir is not None else run.run_dir.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"{run.run_dir.name}_parity_{stamp}"


def write_report(parity_dir: Path, report: dict, rungs: dict, closed: RungResult | None) -> Path:
    """Write ``parity.json`` and ``parity.zarr`` into the report directory.

    Examples
    --------
    ::

        write_report(parity_dir, report, rungs, closed=None)
        json.loads((parity_dir / "parity.json").read_text())["checks"]
    """
    names = [name for name in RUNGS if name in rungs]
    timestamps = rungs[names[0]].equity.index
    equity = np.array([rungs[name].equity.reindex(timestamps).to_numpy() for name in names])
    variables = {"equity": (("rung", "timestamp"), equity)}
    if closed is not None:
        variables["closed_equity"] = ("timestamp", closed.equity.reindex(timestamps).to_numpy())
    for name in names:
        orders = rungs[name].orders
        if not len(orders):
            continue
        dim = f"{name}_order"
        variables.update(
            {
                f"{name}_fill_bar": (dim, pd.DatetimeIndex(orders["fill_bar"]).values),
                f"{name}_symbol": (dim, np.asarray(orders["symbol"].tolist())),
                f"{name}_side": (dim, orders["side"].to_numpy(dtype=str)),
                f"{name}_quantity": (dim, orders["quantity"].to_numpy(dtype=np.float64)),
                f"{name}_price": (dim, orders["price"].to_numpy(dtype=np.float64)),
                f"{name}_fee": (dim, orders["fee"].to_numpy(dtype=np.float64)),
            }
        )
    xr.Dataset(variables, coords={"rung": names, "timestamp": timestamps.values}).to_zarr(
        parity_dir / "parity.zarr", mode="w"
    )
    (parity_dir / "parity.json").write_text(json.dumps(jsonable(report), indent=2))
    return parity_dir
