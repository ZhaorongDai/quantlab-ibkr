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
  ``submitted``/``pending`` for an order the run ended on; ``quantity`` is
  the submitted one, rescaled across a split),
  ``filled_quantity``, ``fill_price`` (volume-weighted), ``fee`` and
  ``reason``;
- ``events.json``: ``{"events": [...]}``, each with a ``type``: holds
  (``hold``, the rule's ``failure``), the rule's events (``rule_event``,
  its ``name`` and the ``symbols`` it names or the ``count`` it gives, as
  quantlab's ``metrics.json`` records them), unfilled orders and corporate actions: venue fills (``side``,
  ``quantity``, ``price``, ``fee``; never in ``orders.zarr``), cash bookings
  and logged factor days (``quantity`` held, ``amount`` of cash);
- ``metrics.json``: quantlab's layout and names (``quantlab_trader.metrics``);
- ``report.html``: quantlab's backtest report of the run.

The directory is written under a temporary name and renamed when complete, so
a failed run leaves no half-written directory.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Hashable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.utils import backtest_stats
from quantlab.utils.backtest_report import write_backtest_report
from quantlab_trader.base.config import TraderConfig
from quantlab_trader.base.venue import MARKET_TZ, NextOpenOrder
from quantlab_trader.decision import CycleResult
from quantlab_trader.metrics import Cycle, bar_label, run_metrics
from quantlab_trader.quantlab_run import QuantlabRun

#: The notes of every trader run's ``metrics.json`` and report.
NOTES = (
    "Executed on raw prices in whole shares, sized at t's raw close and filled at "
    "t+1's opening print; dividends are cash and splits change share counts.",
    "No borrow or short-financing cost is modelled, so short-side returns are "
    "optimistic.",
    "Total Orders counts the next-open orders that filled; corporate-action venue "
    "fills (splits, delisting settlements) are not orders.",
)


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

    Attributes
    ----------
    run_name : str
        The run directory's name, fixed when the recorder is made, so a
        tracking run can be opened under it before the run.
    metrics : dict or None
        The metrics ``write()`` wrote.
    """

    def __init__(self, config: TraderConfig, run: QuantlabRun, init_cash: float):
        self.config = config
        self.run = run
        self.init_cash = float(init_cash)
        name = config.name or run.run_dir.name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_name = f"{name}_{config.loop}_{stamp}"
        self.metrics: dict | None = None
        self._cycles: list[Cycle] = []
        self._fills: list[dict] = []
        self._decisions: dict[pd.Timestamp, pd.Series] = {}
        self._equity: dict[pd.Timestamp, float] = {}
        self._orders: list[dict] = []
        self._order_rows: dict[tuple[Hashable, pd.Timestamp], int] = {}
        self._client_rows: dict[str, int] = {}
        self._events: list[dict] = []

    def record_cycle(self, t: pd.Timestamp, result: CycleResult) -> None:
        """Record one decision cycle: equity, the decision and its orders."""
        self._equity[t] = result.equity
        decided = None if result.decision is None else result.decision.weights
        self._cycles.append(
            Cycle(
                timestamp=pd.Timestamp(t),
                weights=None
                if decided is None
                else pd.Series(
                    decided.values.astype(float),
                    index=[_json_scalar(v) for v in decided["symbol"].values],
                ),
                current_weights=result.current_weights,
                equity=result.equity,
            )
        )
        if result.decision is None:
            return
        weights = result.decision.weights
        self._decisions[t] = pd.Series(
            weights.values, index=[_json_scalar(v) for v in weights["symbol"].values]
        )
        if result.decision.failure is not None:
            self._events.append(
                {"type": "hold", "timestamp": _day(t), "failure": result.decision.failure}
            )
        for name, value in result.decision.events.items():
            event = {"type": "rule_event", "timestamp": _day(t), "name": name}
            if isinstance(value, (int, np.integer)):
                event["count"] = int(value)
            else:
                event["symbols"] = [_json_scalar(v) for v in value]
            self._events.append(event)
        for order in result.orders:
            self._order_rows[(order.permno, order.decision_date)] = len(self._orders)
            self._orders.append(
                dict(
                    decision_date=order.decision_date,
                    symbol=order.permno,
                    side=order.side,
                    quantity=order.quantity,
                    decided=order.quantity,
                    status="pending",
                    filled=0,
                    notional=0.0,
                    fee=0.0,
                    reason="",
                )
            )

    def order_submitted(self, order: NextOpenOrder, client_order_id: str) -> None:
        """Link the venue's order id to the next-open order it carries.

        An order the venue rescaled across a split (ADR 0009) is recorded at
        the quantity submitted, its decided quantity kept in ``reason``.
        """
        row = self._order_rows[(order.permno, order.decision_date)]
        self._client_rows[client_order_id] = row
        record = self._orders[row]
        record["status"] = "submitted"
        if order.quantity != record["quantity"]:
            record["reason"] = (
                f"rescaled across a split from {record['quantity']} decided shares"
            )
            record["quantity"] = order.quantity

    def order_filled(
        self,
        client_order_id: str,
        *,
        ts_ns: int,
        price: float,
        quantity: int,
        fee: float,
        minimum_fee: bool = False,
    ) -> None:
        """Add one fill to its order; fills of other orders are ignored.

        Parameters
        ----------
        client_order_id : str
            The venue's id of the order filled.
        ts_ns : int
            The fill's UNIX nanoseconds; its bar is the market-time-zone date.
        price, quantity, fee
            The fill.
        minimum_fee : bool
            Whether the venue charged its minimum commission on it.
        """
        row = self._client_rows.get(client_order_id)
        if row is None:
            return
        record = self._orders[row]
        self._fills.append(
            dict(
                order=row,
                timestamp=pd.Timestamp(_market_day(ts_ns)),
                symbol=record["symbol"],
                size=quantity if record["side"] == "BUY" else -quantity,
                price=price,
                fee=fee,
                minimum_fee=bool(minimum_fee),
            )
        )
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
        """Record a venue fill of a corporate action as an event, never as an order.

        Parameters
        ----------
        action : str
            The kind, from the fill's ``CORPORATE_ACTION_<KIND>`` tag
            (``DELIST``, ``SPLIT``).
        ts_ns : int
            The fill's UNIX nanoseconds; the event is dated in the market's time zone.
        permno : Hashable
            The security.
        side : {"BUY", "SELL"}
        quantity : int
            Shares filled.
        price : float
            The fill price.
        fee : float
            The commission charged, zero under every trader fee model.
        """
        self._events.append(
            {
                "type": "corporate_action",
                "action": action,
                "timestamp": _market_day(ts_ns),
                "symbol": _json_scalar(permno),
                "side": side,
                "quantity": quantity,
                "price": price,
                "fee": fee,
            }
        )

    def corporate_action_cash(
        self,
        action: str,
        *,
        ts_ns: int,
        permno: Hashable,
        quantity: int,
        amount: float,
        **detail,
    ) -> None:
        """Record a corporate action booked without a fill, or a logged factor day.

        Parameters
        ----------
        action : str
            ``DIVIDEND``, ``CASH_IN_LIEU``, ``DISTRIBUTION``, or a logged
            ``FINAL``/``OTHER`` factor day (``amount`` 0).
        ts_ns : int
            When the venue applied it; dated in the market's time zone.
        permno : Hashable
            The security.
        quantity : int
            The signed holding it applied to.
        amount : float
            Cash moved into the account; negative when paid out.
        **detail
            Facts of the action, recorded as given.
        """
        self._events.append(
            {
                "type": "corporate_action",
                "action": action,
                "timestamp": _market_day(ts_ns),
                "symbol": _json_scalar(permno),
                "quantity": quantity,
                "amount": amount,
                **detail,
            }
        )

    def order_refused(self, client_order_id: str, status: str, reason: str) -> None:
        """Mark an order the venue rejected or the risk engine denied."""
        row = self._client_rows.get(client_order_id)
        if row is not None:
            earlier = self._orders[row]["reason"]
            reason = f"{earlier}; {reason}" if earlier else reason
            self._orders[row].update(status=status, reason=reason)

    def write(self) -> Path:
        """Write the run directory and return its path."""
        output_dir = Path(self.config.output_dir or self.run.run_dir.parent)
        run_dir = output_dir / self.run_name
        partial = output_dir / f".{run_dir.name}.partial"
        partial.mkdir(parents=True)
        try:
            self._write_config(partial / "config.json")
            decisions = self._decisions_dataset()
            equity = self._equity_dataset()
            orders = self._orders_dataset()
            decisions.to_zarr(partial / "decisions.zarr", mode="w")
            equity.to_zarr(partial / "equity.zarr", mode="w")
            orders.to_zarr(partial / "orders.zarr", mode="w")
            (partial / "events.json").write_text(json.dumps({"events": self._events}, indent=2))
            self._write_metrics_and_report(partial, decisions, equity, orders)
            partial.rename(run_dir)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return run_dir

    def _write_metrics_and_report(
        self, directory: Path, decisions: xr.Dataset, equity: xr.Dataset, orders: xr.Dataset
    ) -> None:
        """Compute the metrics, then write ``metrics.json`` and ``report.html``."""
        run = self.run
        timestamps = pd.DatetimeIndex(equity["timestamp"].values)
        # quantlab's engine: the most common spacing of the window's bars.
        bar_interval = (
            pd.Series(np.diff(timestamps.values)).mode().iloc[0]
            if len(timestamps) > 1
            else pd.Timedelta(days=1)
        )
        year_freq = backtest_stats.year_freq(
            bar_interval, run.trading_days_per_year, run.session_minutes_per_day
        )
        closes = (
            run.price_dataset.panel(timestamps[0], timestamps[-1])["close"]
            .transpose("timestamp", "symbol")
            .to_pandas()
            .ffill()
        )
        closes.columns = [_json_scalar(v) for v in closes.columns]
        benchmark = run.benchmark()
        fills = self._fills_dataset()
        self.metrics = run_metrics(
            equity=equity,
            fills=fills,
            orders=orders.assign(
                decided_quantity=("order", np.array([r["decided"] for r in self._orders], dtype=np.int64))
            ),
            cycles=self._cycles,
            events=self._events,
            closes=closes,
            init_cash=self.init_cash,
            bar_interval=bar_interval,
            year_freq=year_freq,
            rebalance_periods=run.rebalance_periods,
            split=run.split(),
            benchmark=benchmark,
            closed_loop=self.config.loop == "closed",
            notes=NOTES,
        )
        (directory / "metrics.json").write_text(
            json.dumps(_jsonable(self.metrics), indent=2)
        )
        report_benchmark = {}
        if benchmark is not None:
            returns = benchmark["returns"].sel(timestamp=equity["timestamp"].values)
            report_benchmark = dict(
                benchmark_returns=returns,
                benchmark_value=self.init_cash * (1.0 + returns.fillna(0.0)).cumprod("timestamp"),
                benchmark_name=str(benchmark.get("symbol") or "benchmark"),
            )
        execution = self.config.venue.get_config().get("execution", {})
        write_backtest_report(
            equity["value"],
            directory / "report.html",
            in_sample_range=self.metrics.get("in_sample_range"),
            notes=list(NOTES),
            title=self.run_name,
            summary={
                "Quantlab run": str(run.run_dir),
                "Loop": f"{self.config.loop} loop",
                "Bar interval": str(bar_interval),
                "Rebalance every": f"{run.rebalance_periods} bars",
                "Execution": ", ".join(f"{k}={v}" for k, v in execution.items()) or "-",
            },
            metrics=self.metrics,
            returns=equity["returns"],
            init_cash=self.init_cash,
            weights=decisions["weight"] if decisions.sizes["timestamp"] else None,
            turnover=backtest_stats.turnover(
                fills.rename(fill="order") if fills.sizes.get("fill", 0) else xr.Dataset(),
                equity["value"],
                self.init_cash,
            ),
            bars_per_year=year_freq / bar_interval,
            windows={
                "backtest": (bar_label(timestamps[0]), bar_label(timestamps[-1])),
                "bars": len(timestamps),
                "in_sample": list(self.metrics.get("in_sample_ranges") or [])
                + ([self.metrics["in_sample_range"]] if self.metrics.get("in_sample_range") else []),
                "out_of_sample": list(self.metrics.get("out_of_sample_ranges") or []),
                "folds": [],
            },
            **report_benchmark,
        )

    def _fills_dataset(self) -> xr.Dataset:
        """The fills of next-open orders on ``fill``, as ``run_metrics`` reads them."""
        rows = self._fills
        return xr.Dataset(
            {
                "order": ("fill", np.array([r["order"] for r in rows], dtype=np.int64)),
                "timestamp": ("fill", np.array([r["timestamp"] for r in rows], dtype="datetime64[ns]")),
                "symbol": ("fill", np.array([r["symbol"] for r in rows])),
                "size": ("fill", np.array([r["size"] for r in rows], dtype=np.int64)),
                "price": ("fill", np.array([r["price"] for r in rows], dtype=float)),
                "fee": ("fill", np.array([r["fee"] for r in rows], dtype=float)),
                "minimum_fee": ("fill", np.array([r["minimum_fee"] for r in rows], dtype=bool)),
            }
        )

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


def _market_day(ts_ns: int) -> str:
    """Return the market-time-zone date of UNIX nanoseconds ``ts_ns``."""
    return _day(pd.Timestamp(ts_ns, tz="UTC").tz_convert(MARKET_TZ))


def _jsonable(value):
    """Return ``value`` as strict JSON values, as quantlab's ``to_jsonable`` does.

    NaN, infinities and NaT become ``None``, timestamps ISO strings,
    timedeltas their ``str``, numpy scalars Python ones.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (np.datetime64, np.timedelta64)):
        if np.isnat(value):
            return None
        value = pd.Timestamp(value) if isinstance(value, np.datetime64) else pd.Timedelta(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _json_scalar(value):
    """Return ``value`` as a JSON-serialisable scalar (numpy scalars unwrapped)."""
    return value.item() if hasattr(value, "item") else value
