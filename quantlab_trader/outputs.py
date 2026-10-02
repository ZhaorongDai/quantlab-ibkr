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
import shutil
from collections.abc import Callable, Hashable
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.utils import backtest_stats
from quantlab.utils.backtest_report import write_backtest_report
from quantlab_trader._support.jsonable import jsonable, python_scalar
from quantlab_trader.base.config import TraderConfig
from quantlab_trader.base.venue import MARKET_TZ, Loop, NextOpenOrder, VenueReport
from quantlab_trader.decision import CycleResult
from quantlab_trader.metrics import CycleRecord, bar_label, run_metrics, window_benchmark
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

    Examples
    --------
    ``runner.run`` makes one per run, the strategy fills it and the run ends
    with ``write``::

        recorder = RunRecorder(config, quantlab_run, init_cash=venue.init_cash)
        strategy = PortfolioStrategy(venue=venue, cycle=cycle, recorder=recorder)
        run_dir = recorder.write(venue.run(strategy))
    """

    def __init__(self, config: TraderConfig, run: QuantlabRun, init_cash: float):
        self.config = config
        self.run = run
        self.init_cash = float(init_cash)
        name = config.name or run.run_dir.name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_name = f"{name}_{config.loop.value}_{stamp}"
        self.metrics: dict | None = None
        self._cycles: list[CycleRecord] = []
        self._fills: list[dict] = []
        self._decisions: dict[pd.Timestamp, pd.Series] = {}
        self._equity: dict[pd.Timestamp, float] = {}
        self._orders: list[dict] = []
        self._order_rows: dict[tuple[Hashable, pd.Timestamp], int] = {}
        self._client_rows: dict[str, int] = {}
        self._events: list[dict] = []

    def record_cycle(self, t: pd.Timestamp, result: CycleResult) -> None:
        """Record one decision cycle: equity, the decision and its orders.

        Parameters
        ----------
        t : pandas.Timestamp
            The decision date.
        result : CycleResult
            What the cycle produced.

        Examples
        --------
        ``PortfolioStrategy.on_decision_time`` records every cycle::

            result = cycle.run(inputs, positions, cash)
            recorder.record_cycle(t, result)
        """
        self._equity[t] = result.equity
        decided = None if result.decision is None else result.decision.weights
        self._cycles.append(
            CycleRecord(
                timestamp=pd.Timestamp(t),
                weights=None
                if decided is None
                else pd.Series(
                    decided.values.astype(float),
                    index=[python_scalar(v) for v in decided["symbol"].values],
                ),
                current_weights=result.current_weights,
                equity=result.equity,
            )
        )
        if result.decision is None:
            return
        weights = result.decision.weights
        self._decisions[t] = pd.Series(
            weights.values, index=[python_scalar(v) for v in weights["symbol"].values]
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
                event["symbols"] = [python_scalar(v) for v in value]
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

        Parameters
        ----------
        order : NextOpenOrder
            The order as submitted.
        client_order_id : str
            The venue's id of the market order carrying it.

        Examples
        --------
        ``PortfolioStrategy.submit_next_open`` links each market order it
        submits::

            recorder.order_submitted(order, market_order.client_order_id.value)
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

        Examples
        --------
        ``PortfolioStrategy.on_order_filled`` records each fill of a
        next-open order::

            recorder.order_filled(
                "O-20240104-143000-001-000-1", ts_ns=ts_ns, price=30.12, quantity=25, fee=1.0
            )
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
            )
        )
        record["filled"] += quantity
        record["notional"] += price * quantity
        record["fee"] += fee
        record["status"] = "filled" if record["filled"] >= record["quantity"] else "partially_filled"

    def order_unfilled(self, order: NextOpenOrder, reason: str) -> None:
        """Mark a next-open order that ended without a fill.

        Parameters
        ----------
        order : NextOpenOrder
            The order decided at the close.
        reason : str
            Why it was not filled.

        Examples
        --------
        ``PortfolioStrategy.on_next_open_unfilled`` records it::

            recorder.order_unfilled(order, "no next open in the backtest window")
        """
        row = self._order_rows[(order.permno, order.decision_date)]
        self._orders[row].update(status="unfilled", reason=reason)
        self._events.append(
            {
                "type": "unfilled_order",
                "decision_date": _day(order.decision_date),
                "symbol": python_scalar(order.permno),
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
            (``DELIST``, ``SPLIT``, ``IMPLIED_SPLIT``).
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

        Examples
        --------
        ``PortfolioStrategy.on_order_filled`` records a venue fill this way,
        here a delisting settlement selling 100 shares::

            recorder.corporate_action(
                "DELIST", ts_ns=ts_ns, permno=10001, side="SELL",
                quantity=100, price=12.5, fee=0.0,
            )
        """
        self._events.append(
            {
                "type": "corporate_action",
                "action": action,
                "timestamp": _market_day(ts_ns),
                "symbol": python_scalar(permno),
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
            ``FINAL``/``OTHER`` factor day, a ``MISMATCH`` day or ``DELISTING_PAYMENT`` (``amount`` 0).
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

        Examples
        --------
        ``PortfolioStrategy.on_corporate_action`` records what a venue
        reports, here a USD 0.24 dividend on 100 shares::

            recorder.corporate_action_cash(
                "DIVIDEND", ts_ns=ts_ns, permno=10001, quantity=100,
                amount=24.0, per_share=0.24,
            )
        """
        self._events.append(
            {
                "type": "corporate_action",
                "action": action,
                "timestamp": _market_day(ts_ns),
                "symbol": python_scalar(permno),
                "quantity": quantity,
                "amount": amount,
                **detail,
            }
        )

    def order_refused(self, client_order_id: str, status: str, reason: str) -> None:
        """Mark an order the venue rejected or the risk engine denied.

        Parameters
        ----------
        client_order_id : str
            The venue's id of the order; an id of no next-open order is ignored.
        status : {"rejected", "denied"}
            The order's final status.
        reason : str
            The venue's or risk engine's reason, appended to any earlier one.

        Examples
        --------
        ``PortfolioStrategy.on_order_rejected`` records a rejection::

            recorder.order_refused(event.client_order_id.value, "rejected", str(event.reason))
        """
        row = self._client_rows.get(client_order_id)
        if row is not None:
            earlier = self._orders[row]["reason"]
            reason = f"{earlier}; {reason}" if earlier else reason
            self._orders[row].update(status=status, reason=reason)

    def write(self, report: VenueReport) -> Path:
        """Write the run directory and return its path.

        Parameters
        ----------
        report : VenueReport
            What the venue reported about the run's execution: which fills
            it charged its minimum commission.

        Returns
        -------
        pathlib.Path
            The run directory.

        Examples
        --------
        ``runner.run`` writes the directory once the venue has run::

            run_dir = recorder.write(venue.run(strategy))
            metrics = json.loads((run_dir / "metrics.json").read_text())
        """
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
            self._write_metrics_and_report(partial, decisions, equity, orders, report)
            partial.rename(run_dir)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return run_dir

    def _write_metrics_and_report(
        self,
        directory: Path,
        decisions: xr.Dataset,
        equity: xr.Dataset,
        orders: xr.Dataset,
        report: VenueReport,
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
        closes.columns = [python_scalar(v) for v in closes.columns]
        benchmark = run.benchmark()
        fills = self._fills_dataset(report)
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
            closed_loop=self.config.loop is Loop.CLOSED,
            notes=NOTES,
        )
        (directory / "metrics.json").write_text(
            json.dumps(jsonable(self.metrics), indent=2)
        )
        report_benchmark = {}
        if benchmark is not None:
            returns = window_benchmark(benchmark["returns"], equity["timestamp"].values)
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
                "Loop": f"{self.config.loop.value} loop",
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

    def _fills_dataset(self, report: VenueReport) -> xr.Dataset:
        """The fills of next-open orders on ``fill``, as ``run_metrics`` reads them.

        ``minimum_fee`` marks the fills of the orders ``report`` lists as
        charged the venue's minimum commission.
        """
        rows = self._fills
        minimum = {
            self._client_rows[client_id]
            for client_id in report.minimum_fee_orders
            if client_id in self._client_rows
        }
        return xr.Dataset(
            {
                "order": ("fill", np.array([r["order"] for r in rows], dtype=np.int64)),
                "timestamp": ("fill", np.array([r["timestamp"] for r in rows], dtype="datetime64[ns]")),
                "symbol": ("fill", np.array([r["symbol"] for r in rows])),
                "size": ("fill", np.array([r["size"] for r in rows], dtype=np.int64)),
                "price": ("fill", np.array([r["price"] for r in rows], dtype=float)),
                "fee": ("fill", np.array([r["fee"] for r in rows], dtype=float)),
                "minimum_fee": ("fill", np.array([r["order"] in minimum for r in rows], dtype=bool)),
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

