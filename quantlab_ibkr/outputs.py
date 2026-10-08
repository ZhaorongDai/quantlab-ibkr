"""``RunRecorder``: collects one trader run and writes its run directory.

``<output_dir>/<name>_<loop>_<stamp>/`` holds:

- ``config.json``: the ``TraderConfig`` plus records of the quantlab run (its
  data fingerprints, ``quantlab_data_fingerprint``) and the run's own
  ``data_fingerprint``: the reads of a closed loop's decisions, recorded by
  quantlab's ``DataRecorder`` under the quantlab run's component paths
  (``None`` for an open loop);
- ``decisions.zarr``: ``weight`` on ``(timestamp, symbol)``, one row per
  decision, in rebalance-table format (comparable with ``weights.zarr``),
  and ``current_weight`` on the same axes, the current weights the cycle
  handed the rule at that decision: each holding's value at t's raw close
  over equity, 0 on a decided symbol not held, NaN on a symbol outside the
  bar's decision (a closed-loop decision's symbols include every holding).
  The parity report's **Decision recheck** runs the rule again on them;
- ``holdings.zarr``: ``holding`` on ``(timestamp, symbol)``, the actual book
  at each close (each holding's value over equity, 0 when not held, every
  decided symbol included), quantlab's layout, which the report's Holdings
  tab shows beside the decided targets;
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
- ``metrics.json``: quantlab's layout and names (``quantlab_ibkr.metrics``);
- ``report.html``: quantlab's backtest report of the run, built with
  quantlab's report-input builders.

The directory is written under a temporary name and renamed when complete, so
a failed run leaves no half-written directory.

The **live run directory** (``LiveRecorder``, #51) holds the same files,
appended one live day at a time: each step restores everything recorded so
far from the directory's ``journal.json``, records its day through the same
per-cycle and per-order records and writes every file again from the whole
history. Live events add ``live_hold`` (a rebalance bar not decided, with its
reason), ``excluded_symbols`` (symbols left out of a decision: no IBKR
contract), ``position_gap`` (an IBKR position no symbol maps to),
``adopted_orders`` and ``timed_out``. Its files whose names a quantlab
run directory also uses (``config.json``, ``equity.zarr``, ``metrics.json``,
``report.html``) are named only here, and read back with ``read_config``,
``read_equity``, ``read_metrics`` and ``report_path``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Hashable, Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs import backtest_stats
from quantlab.runs.backtest_report import (
    report_chart_inputs,
    report_holdings_inputs,
    report_portfolio_inputs,
    report_summary,
    report_windows,
    write_backtest_report,
)
from quantlab_ibkr._support.jsonable import jsonable, python_scalar
from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.base.venue import MARKET_TZ, Loop, NextOpenOrder, VenueReport
from quantlab_ibkr.decision import CycleResult
from quantlab_ibkr.metrics import CycleRecord, cut_folds, run_metrics, window_benchmark
from quantlab_ibkr.quantlab_run import QuantlabRun

#: The notes of every trader run's ``metrics.json`` and report.
NOTES = (
    "Executed on raw prices in whole shares, sized at t's raw close and filled at "
    "t+1's opening print; dividends are cash and splits change share counts.",
    "No borrow or short-financing cost is modelled, so short-side returns are "
    "optimistic.",
    "Total Orders counts the next-open orders that filled; corporate-action venue "
    "fills (splits, delisting settlements) are not orders.",
    "The trade metrics are position round trips of the actual fills, open to flat: "
    "dividends, value distributions and cash in lieu received while open count in a "
    "trip's PnL, a split does not end it and a delisting settlement does.",
    "The Portfolio tab draws the decided targets; the Holdings tab shows them beside "
    "the actual holdings at each close.",
    "The triangles on the equity curve mark the deepest drawdown from its valley to "
    "its recovery.",
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
    data_fingerprint : dict or None
        The run's recorded reads (a quantlab ``DataRecorder``'s ``records``),
        set before ``write()``; ``None`` when nothing was recorded.

    Examples
    --------
    ``runner.run`` makes one per run, the strategy fills it and the run ends
    with ``write``::

        recorder = RunRecorder(config, quantlab_run, init_cash=venue.init_cash)
        strategy = PortfolioStrategy(venue=venue, cycle=cycle, recorder=recorder)
        run_dir = recorder.write(venue.run(strategy))
    """

    def __init__(
        self,
        config: TraderConfig,
        run: QuantlabRun,
        init_cash: float,
        *,
        run_name: str | None = None,
    ):
        self.config = config
        self.run = run
        self.init_cash = float(init_cash)
        name = config.name or run.run_dir.name
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_name = run_name or f"{name}_{config.loop.value}_{stamp}"
        self.metrics: dict | None = None
        self.data_fingerprint: dict | None = None
        self._cycles: list[CycleRecord] = []
        self._fills: list[dict] = []
        self._decisions: dict[pd.Timestamp, pd.Series] = {}
        self._current: dict[pd.Timestamp, pd.Series] = {}
        self._equity: dict[pd.Timestamp, float] = {}
        self._orders: list[dict] = []
        self._order_rows: dict[tuple[Hashable, pd.Timestamp], int] = {}
        self._client_rows: dict[str, int] = {}
        self._trade_ids: set[str] = set()
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
        decided = None if result.decision is None else result.decision.weights
        self._add_cycle(
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

    def _add_cycle(self, cycle: CycleRecord) -> None:
        """Add one cycle's equity, decision and current weights (``record_cycle``, a restore)."""
        t = cycle.timestamp
        self._equity[t] = cycle.equity
        self._cycles.append(cycle)
        if cycle.weights is None:
            return
        symbols = list(cycle.weights.index)
        self._decisions[t] = cycle.weights
        self._current[t] = cycle.current_weights.reindex(symbols, fill_value=0.0).astype(float)

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
        record["client_order_id"] = client_order_id
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
        trade_id: str | None = None,
    ) -> None:
        """Add one fill to its order; fills of other orders, and a fill seen before, are ignored.

        Parameters
        ----------
        client_order_id : str
            The venue's id of the order filled.
        ts_ns : int
            The fill's UNIX nanoseconds; its bar is the market-time-zone date.
        price, quantity, fee
            The fill.
        trade_id : str, optional
            The venue's id of the fill (IBKR's execution id); a fill whose
            id was already recorded is ignored, so a live record step run
            twice adds each fill once.

        Examples
        --------
        ``PortfolioStrategy.on_order_filled`` records each fill of a
        next-open order::

            recorder.order_filled(
                "O-20240104-143000-001-000-1", ts_ns=ts_ns, price=30.12, quantity=25, fee=1.0
            )
        """
        row = self._client_rows.get(client_order_id)
        if row is None or (trade_id is not None and trade_id in self._trade_ids):
            return
        record = self._orders[row]
        fill = dict(
            order=row,
            timestamp=pd.Timestamp(_market_day(ts_ns)),
            symbol=record["symbol"],
            size=quantity if record["side"] == "BUY" else -quantity,
            price=price,
            fee=fee,
        )
        if trade_id is not None:
            self._trade_ids.add(trade_id)
            fill["trade_id"] = trade_id
        self._fills.append(fill)
        record["filled"] += quantity
        record["notional"] += price * quantity
        record["fee"] += fee
        record["status"] = "filled" if record["filled"] >= record["quantity"] else "partially_filled"

    def order_unfilled(self, order: NextOpenOrder, reason: str) -> None:
        """Mark a next-open order that ended without a fill.

        An order already rejected or denied keeps that status and its
        reason (``order_refused``): a live venue reports a rejection both as
        nautilus's ``OrderRejected`` and as an unfilled order, and the order
        is recorded once, as rejected with the venue's reason.

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
        if self._orders[row]["status"] in _REFUSED:
            return
        self._orders[row].update(status="unfilled", reason=reason)
        self._events.append(_unfilled_event(order, reason))

    def order_ended(self, client_order_id: str, reason: str) -> None:
        """Mark an order the venue ended (canceled, expired) without filling it whole.

        Without a fill it is ``unfilled``, as ``order_unfilled`` records it;
        partly filled it stays ``partially_filled`` with ``reason``, and the
        event names the shares left unfilled. An id of no next-open order,
        or of an order already ended, is ignored.

        Parameters
        ----------
        client_order_id : str
            The venue's id of the order.
        reason : str
            Why it ended.

        Examples
        --------
        The live record step ends an order IBKR canceled at the open::

            recorder.order_ended("O-20261008-090501-IBKR-20261007-1", "Cancelled by IBKR: ...")
        """
        row = self._client_rows.get(client_order_id)
        if row is None:
            return
        record = self._orders[row]
        if record["status"] not in _WORKING:
            return
        order = NextOpenOrder(
            record["symbol"], record["side"], record["quantity"] - record["filled"],
            record["decision_date"],
        )
        if not record["filled"]:
            self.order_unfilled(order, reason)
            return
        record["reason"] = reason
        self._events.append(_unfilled_event(order, reason))

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

        An order already reported unfilled for the same refusal (a live
        venue's account of a rejection) is recorded once: rejected, with the
        venue's reason, its ``unfilled_order`` event removed.

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
        if row is None:
            return
        record = self._orders[row]
        if record["status"] == "unfilled":
            # Reported unfilled first (the venue's account of the same
            # rejection): the refusal replaces it, status, reason and event.
            self._events = [
                e for e in self._events
                if not (
                    e.get("type") == "unfilled_order"
                    and e["symbol"] == python_scalar(record["symbol"])
                    and e["decision_date"] == _day(record["decision_date"])
                )
            ]
            record.update(status=status, reason=reason)
            return
        earlier = record["reason"]
        reason = f"{earlier}; {reason}" if earlier else reason
        record.update(status=status, reason=reason)

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
            metrics = read_metrics(run_dir)
        """
        output_dir = Path(self.config.output_dir or self.run.run_dir.parent)
        run_dir = output_dir / self.run_name
        partial = output_dir / f".{run_dir.name}.partial"
        partial.mkdir(parents=True)
        try:
            self._write_files(partial, report)
            partial.rename(run_dir)
        except BaseException:
            shutil.rmtree(partial, ignore_errors=True)
            raise
        return run_dir

    def _write_files(self, directory: Path, report: VenueReport) -> None:
        """Write every file of the run directory into ``directory``."""
        self._write_config(directory / _CONFIG_FILE)
        decisions = self._decisions_dataset()
        holdings = self._holdings_dataset(decisions)
        equity = self._equity_dataset()
        orders = self._orders_dataset()
        decisions.to_zarr(directory / "decisions.zarr", mode="w")
        holdings.to_zarr(directory / "holdings.zarr", mode="w")
        equity.to_zarr(directory / _EQUITY_FILE, mode="w")
        orders.to_zarr(directory / "orders.zarr", mode="w")
        (directory / "events.json").write_text(json.dumps({"events": self._events}, indent=2))
        self._write_metrics_and_report(directory, decisions, holdings, equity, orders, report)

    def state(self) -> dict:
        """Return everything recorded so far as JSON values; ``restore`` takes it back.

        The live run directory keeps it as its journal (``LiveRecorder``):
        each day's step restores it, adds the day and writes every file
        again from the whole history.

        Examples
        --------
        ::

            again = RunRecorder(config, quantlab_run, init_cash=0.0)
            again.restore(recorder.state())
        """
        return {
            "init_cash": self.init_cash,
            "cycles": [
                {
                    "timestamp": cycle.timestamp.isoformat(),
                    "weights": None if cycle.weights is None else _pairs(cycle.weights),
                    "current_weights": _pairs(cycle.current_weights),
                    "equity": cycle.equity,
                }
                for cycle in self._cycles
            ],
            "orders": [
                {**r, "decision_date": pd.Timestamp(r["decision_date"]).isoformat()}
                for r in self._orders
            ],
            "fills": [{**f, "timestamp": f["timestamp"].isoformat()} for f in self._fills],
            "events": self._events,
        }

    def restore(self, state: Mapping) -> None:
        """Take back what ``state`` returned, replacing everything recorded.

        Examples
        --------
        ::

            recorder.restore(json.loads(journal.read_text()))
        """
        self.init_cash = float(state["init_cash"])
        self._cycles, self._decisions, self._current, self._equity = [], {}, {}, {}
        for cycle in state["cycles"]:
            self._add_cycle(
                CycleRecord(
                    timestamp=pd.Timestamp(cycle["timestamp"]),
                    weights=None if cycle["weights"] is None else _series(cycle["weights"]),
                    current_weights=_series(cycle["current_weights"]),
                    equity=float(cycle["equity"]),
                )
            )
        self._orders = [
            {**r, "decision_date": pd.Timestamp(r["decision_date"])} for r in state["orders"]
        ]
        self._order_rows = {
            (r["symbol"], r["decision_date"]): row for row, r in enumerate(self._orders)
        }
        self._client_rows = {
            r["client_order_id"]: row
            for row, r in enumerate(self._orders)
            if r.get("client_order_id") is not None
        }
        self._fills = [{**f, "timestamp": pd.Timestamp(f["timestamp"])} for f in state["fills"]]
        self._trade_ids = {f["trade_id"] for f in self._fills if "trade_id" in f}
        self._events = [dict(e) for e in state["events"]]

    def _write_metrics_and_report(
        self,
        directory: Path,
        decisions: xr.Dataset,
        holdings: xr.Dataset,
        equity: xr.Dataset,
        orders: xr.Dataset,
        report: VenueReport,
    ) -> None:
        """Compute the metrics, then write ``metrics.json`` and ``report.html``.

        The page's inputs come from quantlab's builders, as a quantlab run's
        do: ``report_summary`` (Setup: the quantlab run's config, ``Fees``
        the venue's actual fee model, then the trader lines),
        ``report_windows`` (the quantlab run's folds, cut to the trader's
        bars), ``report_chart_inputs`` (the deepest drawdown, the benchmark)
        ``report_portfolio_inputs`` of the decided targets and the actual
        fills, and ``report_holdings_inputs`` of the actual holdings at each
        close, named through the price dataset's ticker lookup;
        ``execution.trader`` is the "Execution (event-driven)" table. The
        attribution tabs are left off: trader simulates no cost-free or
        risk-model counterfactual.
        """
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
        )
        closes.columns = [python_scalar(v) for v in closes.columns]
        benchmark = run.benchmark()
        fills = self._fills_dataset(report)
        # A live run's fills at the open after its last marked close (the
        # record step runs before that bar closes) join the metrics with the
        # next day's equity row; a backtest's fills are all on its bars.
        fills = fills.isel(fill=fills["timestamp"].values <= timestamps[-1].to_datetime64())
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
        (directory / _METRICS_FILE).write_text(
            json.dumps(jsonable(self.metrics), indent=2)
        )

        records = run.report_records()
        block = dict(self.metrics)
        if records["trained_checkpoint"] is not None:
            block["trained_checkpoint"] = records["trained_checkpoint"]
        span = backtest_stats.drawdown_span(equity["value"])
        summary = report_summary(
            records["recipe"],
            block,
            bar_interval=bar_interval,
            drawdown_span=span,
            benchmark_source=records["benchmark_source"],
        )
        summary["Fees"] = report.fees or "-"
        summary.update(
            {
                "Quantlab run": str(run.run_dir),
                "Loop": f"{self.config.loop.value} loop",
                "Slippage": "-" if report.slippage is None else str(report.slippage),
                "Init cash": f"{self.init_cash:,.2f}",
            }
        )
        benchmark_curves = {}
        if benchmark is not None:
            returns = window_benchmark(benchmark["returns"], equity["timestamp"].values)
            benchmark_curves = dict(
                benchmark_returns=returns,
                benchmark_value=self.init_cash * (1.0 + returns.fillna(0.0)).cumprod("timestamp"),
            )
        write_backtest_report(
            equity["value"],
            directory / _REPORT_FILE,
            title=self.run_name,
            summary=summary,
            windows=report_windows(
                timestamps.values, block, cut_folds(records["folds"], timestamps.values)
            ),
            metrics=self.metrics,
            extra_tables={"Execution (event-driven)": _execution_table(self.metrics)},
            **report_chart_inputs(
                block,
                list(NOTES),
                returns=equity["returns"],
                init_cash=self.init_cash,
                drawdown_span=span,
                **benchmark_curves,
            ),
            **report_portfolio_inputs(
                decisions["weight"],
                fills.rename(fill="order")
                if fills.sizes.get("fill", 0)
                else xr.Dataset({name: ("order", []) for name in ("timestamp", "size", "price")}),
                equity["value"],
                init_cash=self.init_cash,
                bar_interval=bar_interval,
                trading_days_per_year=run.trading_days_per_year,
                session_minutes_per_day=run.session_minutes_per_day,
            ),
            **report_holdings_inputs(
                holdings["holding"], decisions["weight"],
                label=_symbol_names(run.price_dataset.ticker_lookup()),
            ),
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
        config["data_fingerprint"] = self.data_fingerprint
        path.write_text(json.dumps(config, indent=2))

    def _decisions_dataset(self) -> xr.Dataset:
        """``weight`` and ``current_weight`` on ``(timestamp, symbol)``, one row per decision."""
        if not self._decisions:
            empty = xr.DataArray(
                np.empty((0, 0)), dims=("timestamp", "symbol"),
                coords={"timestamp": pd.DatetimeIndex([]), "symbol": []},
            )
            return xr.Dataset({"weight": empty, "current_weight": empty})
        frame = pd.DataFrame(self._decisions).T.sort_index()
        frame.index.name, frame.columns.name = "timestamp", "symbol"
        current = pd.DataFrame(self._current).T.reindex(index=frame.index, columns=frame.columns)
        current.index.name, current.columns.name = "timestamp", "symbol"
        return xr.Dataset(
            {
                "weight": xr.DataArray(frame.astype(float), dims=("timestamp", "symbol")),
                "current_weight": xr.DataArray(current.astype(float), dims=("timestamp", "symbol")),
            }
        )

    def _holdings_dataset(self, decisions: xr.Dataset) -> xr.Dataset:
        """``holding`` on ``(timestamp, symbol)``: each holding's value at the close over equity.

        One row per bar, the actual book after the bar's fills and corporate
        actions; a security not held is 0. The symbols are every one held or
        in ``decisions``, so a target that never filled has a column.
        """
        frame = pd.DataFrame(
            {cycle.timestamp: cycle.current_weights for cycle in self._cycles}
        ).T
        frame = frame.reindex(
            index=pd.DatetimeIndex([cycle.timestamp for cycle in self._cycles]),
            columns=sorted(set(frame.columns) | {python_scalar(v) for v in decisions["symbol"].values}),
        ).astype(float).fillna(0.0)
        frame.index.name, frame.columns.name = "timestamp", "symbol"
        return xr.Dataset({"holding": xr.DataArray(frame, dims=("timestamp", "symbol"))})

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


#: Order statuses a venue's refusal sets; final.
_REFUSED = ("rejected", "denied")
#: Order statuses of an order that may still fill.
_WORKING = ("pending", "submitted", "partially_filled")


def _unfilled_event(order: NextOpenOrder, reason: str) -> dict:
    """The ``unfilled_order`` event of ``order``."""
    return {
        "type": "unfilled_order",
        "decision_date": _day(order.decision_date),
        "symbol": python_scalar(order.permno),
        "side": order.side,
        "quantity": order.quantity,
        "reason": reason,
    }


def _pairs(series: pd.Series) -> list:
    """``[[symbol, value], ...]`` of a weight series, NaN as ``None``: JSON keeps the symbols' types."""
    return [
        [python_scalar(k), None if not np.isfinite(v) else float(v)]
        for k, v in zip(series.index, series.to_numpy(dtype=float))
    ]


def _series(pairs: list) -> pd.Series:
    """The weight series ``_pairs`` wrote."""
    return pd.Series(
        [np.nan if v is None else float(v) for _, v in pairs],
        index=[k for k, _ in pairs],
        dtype=float,
    )


class LiveRecorder(RunRecorder):
    """The live run directory: one directory appended per day, in a trader run's formats.

    ``<live_dir>/`` holds the files ``RunRecorder`` writes (``config.json``,
    ``decisions.zarr``, ``holdings.zarr``, ``equity.zarr``, ``orders.zarr``,
    ``events.json``, ``metrics.json``, ``report.html``) plus ``journal.json``,
    everything recorded so far (``RunRecorder.state``). Each live step
    restores the journal, records its day through ``RunRecorder``'s own
    per-cycle and per-order records, saves the journal and writes every file
    again from the whole history, so the files are the ones one run over all
    the days would write. ``data_fingerprint`` is per day: each day's
    decision reads, keyed by the bar decided.

    Parameters
    ----------
    config : TraderConfig
        The live config; its venue's ``live_dir`` is the directory.
    run : QuantlabRun
        The quantlab run traded (the strategy's recipe).

    Attributes
    ----------
    live_dir : pathlib.Path
        The live run directory.
    fees : str or None
        The venue's fee statement (``VenueReport.fees``), kept for the report.

    Examples
    --------
    ``quantlab_ibkr.live.decide`` restores the directory, runs the day and
    writes it back::

        recorder = LiveRecorder(config, quantlab_run)
        if not recorder.has_cycle(t):
            report = venue.run(PortfolioStrategy(venue=venue, cycle=cycle, recorder=recorder))
            recorder.write(report)
    """

    def __init__(self, config: TraderConfig, run: QuantlabRun):
        live_dir = getattr(config.venue, "live_dir", None)
        if live_dir is None:
            raise ValueError("a live run needs its venue's live_dir")
        self.live_dir = Path(live_dir)
        super().__init__(config, run, init_cash=float("nan"), run_name=self.live_dir.name)
        self.data_fingerprint: dict = {}
        self.fees: str | None = None
        journal = self.live_dir / _JOURNAL_FILE
        if journal.is_file():
            state = json.loads(journal.read_text())
            self.restore(state)
            self.data_fingerprint = dict(state.get("data_fingerprint") or {})
            self.fees = state.get("fees")

    @property
    def started(self) -> bool:
        """Whether any day has been recorded."""
        return bool(self._cycles)

    def has_cycle(self, t: pd.Timestamp) -> bool:
        """Whether the cycle of bar ``t`` is recorded (its decide step ran).

        Examples
        --------
        ::

            if recorder.has_cycle(t):
                print(f"{t.date()} is already recorded")
        """
        return any(cycle.timestamp == pd.Timestamp(t) for cycle in self._cycles)

    def equity_at(self, t: pd.Timestamp) -> float | None:
        """Return the equity the cycle of ``t`` marked, ``None`` when not recorded.

        Examples
        --------
        ::

            recorder.equity_at(t)
        """
        return self._equity.get(pd.Timestamp(t))

    def fill_count(self) -> int:
        """Return how many fills are recorded.

        Examples
        --------
        ::

            recorder.fill_count()
        """
        return len(self._fills)

    def held_symbols(self) -> tuple:
        """Return the symbols the account may hold: the last cycle's holdings and every order since.

        Examples
        --------
        ::

            request = ReplayRequest(t, t, recorder.held_symbols(), Loop.CLOSED)
        """
        if not self._cycles:
            return ()
        last = self._cycles[-1]
        held = [s for s, w in last.current_weights.items() if np.isfinite(w) and w != 0.0]
        ordered = [r["symbol"] for r in self._orders if r["decision_date"] >= last.timestamp]
        return tuple(dict.fromkeys([*held, *ordered]))

    def working_orders(self) -> list[dict]:
        """Return the orders that may still fill: submitted, or partly filled, with a venue id.

        Each is ``{"client_order_id", "symbol", "side", "quantity", "filled",
        "decision_date"}``.

        Examples
        --------
        ::

            for order in recorder.working_orders():
                print(order["client_order_id"], order["quantity"] - order["filled"])
        """
        return [
            {k: r[k] for k in ("client_order_id", "symbol", "side", "quantity", "filled", "decision_date")}
            for r in self._orders
            if r["status"] in ("submitted", "partially_filled") and r.get("client_order_id")
        ]

    def add_event(self, event: Mapping) -> None:
        """Append a live event (a hold, symbols left out, an account gap) to ``events.json``.

        Examples
        --------
        ::

            recorder.add_event({"type": "live_hold", "timestamp": "2026-10-07", "reason": reason})
        """
        self._events.append(dict(event))

    def state(self) -> dict:
        """``RunRecorder.state`` plus the per-day fingerprints and the fee statement."""
        return {**super().state(), "data_fingerprint": self.data_fingerprint, "fees": self.fees}

    def write(self, report: VenueReport | None = None) -> Path:
        """Save the journal, then write every file of the live run directory again.

        The journal is saved first (a file replaced whole), so a day whose
        orders went out is never lost to a failure in the metrics or the
        report; the other files are written beside it under a temporary
        name and swapped in.

        Parameters
        ----------
        report : VenueReport, optional
            The day's venue report; its ``fees`` statement is kept. ``None``
            keeps the journal's.

        Returns
        -------
        pathlib.Path
            The live run directory.

        Examples
        --------
        ::

            live_dir = recorder.write(venue.run(strategy))
            read_metrics(live_dir)["whole"]
        """
        if report is not None and report.fees is not None:
            self.fees = report.fees
        if np.isnan(self.init_cash) and self._cycles:
            # The first day's equity: its return is 0, as a backtest's first bar.
            self.init_cash = float(self._cycles[0].equity)
        live = self.live_dir
        live.mkdir(parents=True, exist_ok=True)
        journal = live / _JOURNAL_FILE
        partial = live / f".{_JOURNAL_FILE}.partial"
        partial.write_text(json.dumps(self.state(), indent=1))
        partial.replace(journal)
        staging = live / ".partial"
        trash = live / ".replaced"
        for directory in (staging, trash):
            shutil.rmtree(directory, ignore_errors=True)
        staging.mkdir()
        try:
            self._write_files(staging, VenueReport(fees=self.fees))
            trash.mkdir()
            for path in sorted(staging.iterdir()):
                target = live / path.name
                if target.exists():
                    target.rename(trash / path.name)
                path.rename(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(trash, ignore_errors=True)
        return live


def _symbol_names(lookup) -> Callable[[list, object], list[tuple[str, str | None]]]:
    """Return the Holdings tab's ``label``: ``(ticker, company)`` of symbols as of a day.

    Names through ``lookup`` (the price dataset's ``ticker_lookup()``), as
    quantlab's engine does, and each symbol itself (as ``str``) with no
    company when the dataset names no lookup.

    Examples
    --------
    >>> _symbol_names(None)([10001, 10002], None)
    [('10001', None), ('10002', None)]
    """
    if lookup is None:
        return lambda symbols, day: [(str(symbol), None) for symbol in symbols]
    return lambda symbols, day: [(name.ticker, name.company) for name in lookup.names(symbols, day)]


def _day(t: pd.Timestamp) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d")


def _market_day(ts_ns: int) -> str:
    """Return the market-time-zone date of UNIX nanoseconds ``ts_ns``."""
    return _day(pd.Timestamp(ts_ns, tz="UTC").tz_convert(MARKET_TZ))


def _execution_table(metrics: dict) -> dict:
    """The "Execution (event-driven)" rows: ``execution.trader``, facts quantlab has no row for."""
    trader = metrics["execution"]["trader"]
    return {
        "Commissions": trader["commissions"],
        "Minimum-fee hits": trader["minimum_fee_hits"],
        "Dividends": trader["dividends"]["count"],
        "Dividend cash": trader["dividends"]["amount"],
        "Value distributions": trader["value_distributions"]["count"],
        "Value distribution cash": trader["value_distributions"]["amount"],
        "Cash in lieu": trader["cash_in_lieu"]["count"],
        "Cash in lieu amount": trader["cash_in_lieu"]["amount"],
        "Splits": trader["splits"]["count"],
        "Split share change": trader["splits"]["share_change"],
        "Implied splits": trader["implied_splits"]["count"],
        "Implied split share change": trader["implied_splits"]["share_change"],
        "Factor mismatches": trader["mismatches"]["count"],
        "Peak cash debit": trader["peak_cash_debit"],
    }


_CONFIG_FILE = "config.json"
_EQUITY_FILE = "equity.zarr"
_METRICS_FILE = "metrics.json"
_REPORT_FILE = "report.html"
_JOURNAL_FILE = "journal.json"


def read_config(run_dir: Path) -> dict:
    """Return a trader run's config: its ``TraderConfig`` and the quantlab run's records.

    Examples
    --------
    ::

        tracking.update_config(read_config(run_dir))
    """
    return json.loads((Path(run_dir) / _CONFIG_FILE).read_text())


def read_metrics(run_dir: Path) -> dict:
    """Return a trader run's metrics, quantlab's layout and names.

    Examples
    --------
    ::

        read_metrics(run_dir)["whole"]["Total Return [%]"]
    """
    return json.loads((Path(run_dir) / _METRICS_FILE).read_text())


def read_equity(run_dir: Path) -> xr.Dataset:
    """Return a trader run's equity curve, ``value`` and ``returns`` on ``timestamp``, loaded.

    Examples
    --------
    ::

        value = read_equity(run_dir)["value"]
    """
    with xr.open_zarr(Path(run_dir) / _EQUITY_FILE) as equity:
        return equity.load()


def report_path(run_dir: Path) -> Path:
    """Return where a trader run's HTML report is.

    Examples
    --------
    ::

        tracking.log_file(report_path(run_dir))
    """
    return Path(run_dir) / _REPORT_FILE
