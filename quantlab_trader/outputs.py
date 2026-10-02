"""``RunRecorder``: collects one trader run and writes its run directory.

``<output_dir>/<name>_<loop>_<stamp>/`` holds:

- ``config.json``: the ``TraderConfig`` plus records of the quantlab run (its
  data fingerprints);
- ``decisions.zarr``: ``weight`` on ``(timestamp, symbol)``, one row per
  decision, in rebalance-table format (comparable with ``weights.zarr``);
- ``equity.zarr``: ``value`` (cash plus holdings at the raw close) and
  ``returns`` on ``timestamp``, quantlab's layout;
- ``orders.zarr``: one row per next-open order on ``order``: ``decision_date``,
  ``symbol``, ``side``, ``quantity``, ``status`` (``filled``,
  ``partially_filled``, ``unfilled``, ``rejected``, ``denied``, or
  ``submitted``/``pending`` for an order the run ended on),
  ``filled_quantity``, ``fill_price`` (volume-weighted), ``fee`` and
  ``reason``;
- ``events.json``: ``{"events": [...]}``, each with a ``type``: holds, rule
  events, unfilled orders and corporate actions (venue fills, which are
  never in ``orders.zarr``).

The directory is written under a temporary name and renamed when complete, so
a failed run leaves no half-written directory.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab_trader.base.config import TraderConfig
from quantlab_trader.base.venue import MARKET_TZ, NextOpenOrder
from quantlab_trader.decision import CycleResult
from quantlab_trader.quantlab_run import QuantlabRun


class RunRecorder:
    """Collect a trader run's decisions, equity, orders and events, then write them.

    Parameters
    ----------
    config : TraderConfig
        The run's config.
    run : QuantlabRun
        The quantlab run executed.
    init_cash : float
        Starting cash, the base of the first bar's return.
    """

    def __init__(self, config: TraderConfig, run: QuantlabRun, init_cash: float):
        self.config = config
        self.run = run
        self.init_cash = float(init_cash)
        self._decisions: dict[pd.Timestamp, pd.Series] = {}
        self._equity: dict[pd.Timestamp, float] = {}
        self._orders: list[dict] = []
        self._order_rows: dict[tuple[Hashable, pd.Timestamp], int] = {}
        self._client_rows: dict[str, int] = {}
        self._events: list[dict] = []

    def record_cycle(self, t: pd.Timestamp, result: CycleResult) -> None:
        """Record one decision cycle: equity, the decision and its orders."""
        self._equity[t] = result.equity
        if result.decision is None:
            return
        self._decisions[t] = result.decision.weights
        if result.decision.failure is not None:
            self._events.append(
                {"type": "hold", "timestamp": _day(t), "failure": result.decision.failure}
            )
        for event in result.decision.events:
            self._events.append({"type": "rule_event", "timestamp": _day(t), **event})
        for order in result.orders:
            self._order_rows[(order.permno, order.decision_date)] = len(self._orders)
            self._orders.append(
                dict(
                    decision_date=order.decision_date,
                    symbol=order.permno,
                    side=order.side,
                    quantity=order.quantity,
                    status="pending",
                    filled=0,
                    notional=0.0,
                    fee=0.0,
                    reason="",
                )
            )

    def order_submitted(self, order: NextOpenOrder, client_order_id: str) -> None:
        """Link the venue's order id to the next-open order it carries."""
        row = self._order_rows[(order.permno, order.decision_date)]
        self._client_rows[client_order_id] = row
        self._orders[row]["status"] = "submitted"

    def order_filled(self, client_order_id: str, *, price: float, quantity: int, fee: float) -> None:
        """Add one fill to its order; fills of other orders are ignored."""
        row = self._client_rows.get(client_order_id)
        if row is None:
            return
        record = self._orders[row]
        record["filled"] += quantity
        record["notional"] += price * quantity
        record["fee"] += fee
        record["status"] = "filled" if record["filled"] >= record["quantity"] else "partially_filled"

    def order_unfilled(self, order: NextOpenOrder, reason: str) -> None:
        """Mark a next-open order that ended without a fill."""
        row = self._order_rows[(order.permno, order.decision_date)]
        self._orders[row].update(status="unfilled", reason=reason)
        self._events.append(
            {
                "type": "unfilled_order",
                "decision_date": _day(order.decision_date),
                "symbol": _json_scalar(order.permno),
                "side": order.side,
                "quantity": order.quantity,
                "reason": reason,
            }
        )

    def corporate_action(
        self,
        action: str,
        *,
        ts_ns: int,
        permno: Hashable,
        side: str,
        quantity: int,
        price: float,
        fee: float,
    ) -> None:
        """Record a venue fill of a corporate action (``DELIST``, ...) as an event."""
        self._events.append(
            {
                "type": "corporate_action",
                "action": action,
                "timestamp": _day(pd.Timestamp(ts_ns, tz="UTC").tz_convert(MARKET_TZ)),
                "symbol": _json_scalar(permno),
                "side": side,
                "quantity": quantity,
                "price": price,
                "fee": fee,
            }
        )

    def order_refused(self, client_order_id: str, status: str, reason: str) -> None:
        """Mark an order the venue rejected or the risk engine denied."""
        row = self._client_rows.get(client_order_id)
        if row is not None:
            self._orders[row].update(status=status, reason=reason)

    def write(self) -> Path:
        """Write the run directory and return its path."""
        output_dir = Path(self.config.output_dir or self.run.run_dir.parent)
        name = self.config.name or self.run.run_dir.name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = output_dir / f"{name}_{self.config.loop}_{stamp}"
        partial = output_dir / f".{run_dir.name}.partial"
        partial.mkdir(parents=True)
        try:
            self._write_config(partial / "config.json")
            self._decisions_dataset().to_zarr(partial / "decisions.zarr", mode="w")
            self._equity_dataset().to_zarr(partial / "equity.zarr", mode="w")
            self._orders_dataset().to_zarr(partial / "orders.zarr", mode="w")
            (partial / "events.json").write_text(json.dumps({"events": self._events}, indent=2))
            partial.rename(run_dir)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return run_dir

    def _write_config(self, path: Path) -> None:
        config = self.config.get_config()
        config["quantlab_data_fingerprint"] = self.run.data_fingerprint
        path.write_text(json.dumps(config, indent=2))

    def _decisions_dataset(self) -> xr.Dataset:
        if not self._decisions:
            weight = xr.DataArray(
                np.empty((0, 0)), dims=("timestamp", "symbol"),
                coords={"timestamp": pd.DatetimeIndex([]), "symbol": []},
            )
        else:
            frame = pd.DataFrame(self._decisions).T.sort_index()
            frame.index.name, frame.columns.name = "timestamp", "symbol"
            weight = xr.DataArray(frame.astype(float), dims=("timestamp", "symbol"))
        return xr.Dataset({"weight": weight})

    def _equity_dataset(self) -> xr.Dataset:
        value = pd.Series(self._equity, dtype=float).sort_index()
        previous = value.shift(1)
        previous.iloc[:1] = self.init_cash
        timestamps = pd.DatetimeIndex(value.index, name="timestamp")
        return xr.Dataset(
            {
                "value": ("timestamp", value.to_numpy()),
                "returns": ("timestamp", (value / previous - 1.0).to_numpy()),
            },
            coords={"timestamp": timestamps},
        )

    def _orders_dataset(self) -> xr.Dataset:
        rows = self._orders
        filled = np.array([r["filled"] for r in rows], dtype=np.int64)
        notional = np.array([r["notional"] for r in rows], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            fill_price = np.where(filled > 0, notional / np.maximum(filled, 1), np.nan)
        columns = {
            "decision_date": np.array([r["decision_date"] for r in rows], dtype="datetime64[ns]"),
            "symbol": np.array([r["symbol"] for r in rows]),
            "side": np.array([r["side"] for r in rows], dtype=str),
            "quantity": np.array([r["quantity"] for r in rows], dtype=np.int64),
            "status": np.array([r["status"] for r in rows], dtype=str),
            "filled_quantity": filled,
            "fill_price": fill_price,
            "fee": np.array([r["fee"] for r in rows], dtype=float),
            "reason": np.array([r["reason"] for r in rows], dtype=str),
        }
        return xr.Dataset(
            {name: ("order", values) for name, values in columns.items()},
            coords={"order": np.arange(len(rows))},
        )


def _day(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def _json_scalar(value):
    """Return ``value`` as a JSON-serialisable scalar (numpy scalars unwrapped)."""
    return value.item() if hasattr(value, "item") else value
