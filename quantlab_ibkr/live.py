"""``quantlab-ibkr live``: a live day in two steps, appended to one live run directory (#51).

Live trading is a daily batch (#47). Each weekday, after quantlab's daily
prediction job has extended the run's stores to the last closed bar t and
appended t's row to the live prediction store:

- ``decide`` (before the open): builds the IBKR venue on t, runs the
  strategy's decision cycle once on the reconciled account (marking it at
  t's raw close every day, deciding t on a rebalance bar of the run's
  cadence) and submits the orders market-on-open before the order deadline;
  then appends the day's cycle (equity, holdings, decision with the current
  weights handed to the rule, orders) to the live run directory;
- ``record`` (after the open): reads IBKR's executions, open and completed
  orders, appends the fills of the orders still working (each IBKR execution
  once) and ends the ones IBKR canceled or expired, rewrites the directory's
  files and runs the Decision recheck on it (``recheck``).

Both are idempotent. ``decide`` on a bar already recorded does nothing, and
an order IBKR already has working for t (an earlier run that failed before
recording) is adopted, never sent again: the strategy's order id tag is the
decision date (``reports.decision_tag``), so each IBKR order names the bar
that decided it. ``record`` adds a fill whose execution id it has seen once
and leaves ended orders alone.

The live run directory is ``RunRecorder``'s run directory, rewritten whole
from its journal by ``outputs.LiveRecorder`` each step. A dry run decides and
reports the orders and writes nothing.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from nautilus_trader.config import StrategyConfig

from quantlab.runs.live_predictions import LivePredictionStore
from quantlab.runs.record import DataRecorder
from quantlab_ibkr._support.jsonable import jsonable, python_scalar
from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.base.venue import Loop, NextOpenOrder, ReplayRequest
from quantlab_ibkr.decision import DecisionCycle
from quantlab_ibkr.outputs import LiveRecorder
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.strategy import PortfolioStrategy
from quantlab_ibkr.venue.ibkr.reports import (
    IbapiOrderReports,
    IbkrExecution,
    IbkrOrderState,
    OrderReports,
    decision_date_of,
    decision_tag,
)
from quantlab_ibkr.venue.ibkr.source import last_price_bar
from quantlab_ibkr.venue.ibkr.venue import IbkrVenue, IbkrVenueConfig

#: The Decision recheck's result in the live run directory.
RECHECK_FILE = "decision_recheck.json"

#: The reason an order IBKR ended is recorded with.
ENDED_BY_IBKR = "{status} by IBKR: {reason}"


@dataclass(frozen=True)
class DecideResult:
    """What a ``decide`` step did.

    Attributes
    ----------
    t : pandas.Timestamp
        The bar (the price dataset's last).
    live_dir : pathlib.Path
    status : {"recorded", "already_recorded", "dry_run"}
        ``already_recorded``: t's cycle was in the live run directory, so
        nothing ran.
    decided : bool
        t was decided (a rebalance bar with a prediction row).
    hold_reason : str or None
        Why t was not decided.
    orders : tuple of NextOpenOrder
        The orders decided.
    submitted, adopted : tuple of NextOpenOrder
        The orders sent to IBKR, and those found already working there.
    unfilled : tuple
        ``(order, reason)`` of each order reported unfilled (dry run, past
        the deadline, refused by IBKR).
    equity : float or None
        The account at t's raw close, as the cycle marked it.

    Examples
    --------
    ::

        result = decide(config)
        print(result.summary())
    """

    t: pd.Timestamp
    live_dir: Path
    status: str
    decided: bool = False
    hold_reason: str | None = None
    orders: tuple[NextOpenOrder, ...] = ()
    submitted: tuple[NextOpenOrder, ...] = ()
    adopted: tuple[NextOpenOrder, ...] = ()
    unfilled: tuple = ()
    equity: float | None = None

    def summary(self) -> str:
        """Return the lines the command prints.

        Examples
        --------
        >>> DecideResult(pd.Timestamp("2026-10-07"), Path("live"), "already_recorded").summary()
        '2026-10-07 is already recorded in live; nothing to do'
        """
        day = self.t.date()
        if self.status == "already_recorded":
            return f"{day} is already recorded in {self.live_dir}; nothing to do"
        lines = [
            f"{day}: equity {self.equity:,.2f}" if self.equity is not None else f"{day}",
            f"decided {len(self.orders)} orders" if self.decided else f"held: {self.hold_reason}",
        ]
        lines += [f"  {o.side} {o.quantity} {o.permno}" for o in self.orders]
        if self.adopted:
            lines.append(f"{len(self.adopted)} orders already working at IBKR, not sent again")
        lines.append(f"submitted {len(self.submitted)} market-on-open orders")
        lines += [f"  unfilled {o.side} {o.quantity} {o.permno}: {r}" for o, r in self.unfilled]
        if self.status == "dry_run":
            lines.append("dry run: nothing submitted, nothing recorded")
        else:
            lines.append(f"recorded in {self.live_dir}")
        return "\n".join(lines)


@dataclass(frozen=True)
class RecordResult:
    """What a ``record`` step did.

    Attributes
    ----------
    live_dir : pathlib.Path
    fills : int
        IBKR executions added.
    ended : tuple of str
        Client order ids of the orders IBKR ended without filling whole.
    working : tuple of str
        Orders IBKR still reports working.
    unknown : tuple of str
        Orders IBKR reports nothing about (no execution, open or completed
        order: the record step ran on a later day than the open); left as
        they were.
    recheck : dict
        The Decision recheck of the live run directory.

    Examples
    --------
    ::

        result = record(config)
        result.recheck["bars_differing"] == 0
    """

    live_dir: Path
    fills: int = 0
    ended: tuple[str, ...] = ()
    working: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()
    recheck: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return the lines the command prints.

        Examples
        --------
        >>> print(RecordResult(Path("live"), recheck={"bars_checked": 0, "bars_differing": 0}).summary())
        added 0 fills; 0 orders ended unfilled; 0 still working
        Decision recheck: 0 bars checked, 0 differing
        recorded in live
        """
        lines = [
            f"added {self.fills} fills; {len(self.ended)} orders ended unfilled; "
            f"{len(self.working)} still working"
        ]
        if self.unknown:
            lines.append(
                f"IBKR reports nothing on {len(self.unknown)} orders (run record on the day "
                f"of the open): {', '.join(self.unknown)}"
            )
        lines.append(
            f"Decision recheck: {self.recheck.get('bars_checked', 0)} bars checked, "
            f"{self.recheck.get('bars_differing', 0)} differing"
        )
        lines.append(f"recorded in {self.live_dir}")
        return "\n".join(lines)


def live_venue(config: TraderConfig) -> IbkrVenueConfig:
    """Return ``config``'s IBKR venue, refusing a config that cannot trade live.

    Raises
    ------
    ValueError
        If the venue is not an ``IbkrVenueConfig`` with a ``live_dir`` and a
        ``prediction_store``, or the loop is open.

    Examples
    --------
    >>> live_venue(TraderConfig("run", IbkrVenueConfig(prediction_store="p.zarr")))
    Traceback (most recent call last):
    ...
    ValueError: a live run needs IbkrVenueConfig.live_dir (the live run directory)
    """
    venue = config.venue
    if not isinstance(venue, IbkrVenueConfig):
        raise ValueError(f"live trading needs an IbkrVenueConfig, got {type(venue).__name__}")
    if venue.live_dir is None:
        raise ValueError("a live run needs IbkrVenueConfig.live_dir (the live run directory)")
    if venue.prediction_store is None:
        raise ValueError("a live run needs IbkrVenueConfig.prediction_store")
    if config.loop is not Loop.CLOSED:
        raise ValueError("live trading is closed loop only")
    return venue


def _reports(venue: IbkrVenueConfig, given: OrderReports | None):
    """IBKR's order reports: ``given``, else a TWS API connection of the reports' client id."""
    if given is not None:
        return contextlib.nullcontext(given)
    client_id = venue.reports_client_id or venue.client_id + 2
    return IbapiOrderReports(venue.host, venue.port, client_id, timeout=venue.timeout_secs)


#: ``venue_factory(run, request, *, now, working_orders) -> IbkrVenue``.
VenueFactory = Callable[..., IbkrVenue]


def decide(
    config: TraderConfig,
    *,
    now: pd.Timestamp | None = None,
    venue_factory: VenueFactory | None = None,
    order_reports: OrderReports | None = None,
) -> DecideResult:
    """Run the decide step of the last closed bar t and append it to the live run directory.

    Parameters
    ----------
    config : TraderConfig
        The quantlab run (the strategy's recipe) and its ``IbkrVenueConfig``.
    now : pandas.Timestamp, optional
        The current time, for the order deadline; the wall clock by default.
    venue_factory : Callable, optional
        Builds the venue; ``config.venue.build`` by default (a test passes a
        venue on fakes).
    order_reports : OrderReports, optional
        IBKR's order reports; a TWS API connection by default.

    Returns
    -------
    DecideResult

    Raises
    ------
    ValueError
        If the config cannot trade live, the account is refused (paper
        guard), or the day decides after its order deadline
        (``OrderDeadlinePassed``).

    Examples
    --------
    ::

        result = decide(TraderConfig.from_config(json.loads(Path("live.json").read_text())))
        print(result.summary())
    """
    venue_config = live_venue(config)
    run = QuantlabRun.load(config.quantlab_run)
    t = last_price_bar(run)
    recorder = LiveRecorder(config, run)
    if recorder.has_cycle(t) and not venue_config.dry_run:
        return DecideResult(t, recorder.live_dir, "already_recorded")
    account = venue_config.account()  # the paper guard, before any connection
    working: tuple[IbkrOrderState, ...] = ()
    if not venue_config.dry_run:
        with _reports(venue_config, order_reports) as reports:
            working = tuple(
                order for order in reports.open_orders(account)
                if decision_date_of(order.order_ref) == t
            )
    build = venue_factory or venue_config.build
    venue = build(
        run,
        ReplayRequest(t, t, recorder.held_symbols(), Loop.CLOSED),
        now=now,
        working_orders=working,
    )
    targets = venue.targets()
    strategy = PortfolioStrategy(
        venue=venue,
        cycle=DecisionCycle(targets),
        recorder=recorder,
        config=StrategyConfig(order_id_tag=decision_tag(t)),
    )
    with DataRecorder(keys=targets.read_sources(), owner=recorder.run_name) as reads:
        report = venue.run(strategy)
    decision = venue.decision
    orders = tuple(venue.submitter.decided)
    result = DecideResult(
        t,
        recorder.live_dir,
        "dry_run" if venue_config.dry_run else "recorded",
        decided=decision.decides,
        hold_reason=decision.hold_reason,
        orders=orders,
        submitted=tuple(venue.submitter.submitted),
        adopted=tuple(venue.submitter.adopted),
        unfilled=tuple(venue.submitter.unfilled),
        equity=recorder.equity_at(t),
    )
    if venue_config.dry_run:
        return result
    day = str(t.date())
    if decision.decides:
        recorder.data_fingerprint[day] = reads.records
        if venue.source.excluded:
            recorder.add_event(
                {
                    "type": "excluded_symbols",
                    "timestamp": day,
                    "symbols": [python_scalar(s) for s in venue.source.excluded],
                    "reason": "no single IBKR contract as of the decision date",
                }
            )
    elif decision.rebalances:
        recorder.add_event({"type": "live_hold", "timestamp": day, "reason": decision.hold_reason})
    for gap in venue.position_gaps:
        recorder.add_event(
            {
                "type": "position_gap",
                "timestamp": day,
                "instrument_id": gap.instrument_id,
                "reason": gap.reason,
            }
        )
    if venue.submitter.adopted:
        recorder.add_event(
            {
                "type": "adopted_orders",
                "timestamp": day,
                "symbols": [python_scalar(o.permno) for o in venue.submitter.adopted],
            }
        )
    if venue.timed_out:
        recorder.add_event({"type": "timed_out", "timestamp": day})
    recorder.write(report)
    return result


def record(config: TraderConfig, *, order_reports: OrderReports | None = None) -> RecordResult:
    """Run the record step: append IBKR's fills of the working orders, then recheck.

    Every order the live run directory holds as submitted (or partly
    filled) is looked up in IBKR's reports: its executions are added as
    fills (each execution id once), an order IBKR completed without filling
    it whole is ended with IBKR's status and reason, one still open stays
    working. The directory's files are written again and the Decision
    recheck runs on it (``RECHECK_FILE``).

    Parameters
    ----------
    config : TraderConfig
        The live config.
    order_reports : OrderReports, optional
        IBKR's order reports; a TWS API connection by default.

    Raises
    ------
    ValueError
        If the config cannot trade live, or nothing has been decided yet.

    Examples
    --------
    ::

        result = record(config)
        print(result.summary())
    """
    venue_config = live_venue(config)
    run = QuantlabRun.load(config.quantlab_run)
    recorder = LiveRecorder(config, run)
    if not recorder.started:
        raise ValueError(f"nothing to record: no day is decided in {recorder.live_dir}")
    working = recorder.working_orders()
    fills = 0
    ended, still, unknown = [], [], []
    if working:
        account = venue_config.account()
        with _reports(venue_config, order_reports) as reports:
            executions = list(reports.executions(account))
            open_refs = {o.order_ref for o in reports.open_orders(account)}
            completed = {o.order_ref: o for o in reports.completed_orders(account)}
        fills = _add_fills(recorder, working, executions)
        for order in recorder.working_orders():
            ref = order["client_order_id"]
            done = completed.get(ref)
            if ref in open_refs:
                still.append(ref)
            elif done is not None and done.status != "Filled":
                recorder.order_ended(
                    ref,
                    ENDED_BY_IBKR.format(status=done.status, reason=done.reason or "no reason given"),
                )
                ended.append(ref)
            else:
                unknown.append(ref)
    recorder.write()
    result = recheck(config, run=run)
    return RecordResult(
        recorder.live_dir, fills, tuple(ended), tuple(still), tuple(unknown), result
    )


def _add_fills(
    recorder: LiveRecorder, working: Sequence[dict], executions: Sequence[IbkrExecution]
) -> int:
    """Add the executions of the working orders, in time order; return how many were new."""
    refs = {order["client_order_id"] for order in working}
    before = recorder.fill_count()
    for execution in sorted(executions, key=lambda e: (e.time, e.exec_id)):
        if execution.order_ref in refs:
            recorder.order_filled(
                execution.order_ref,
                ts_ns=pd.Timestamp(execution.time).value,
                price=execution.price,
                quantity=execution.shares,
                fee=execution.commission,
                trade_id=execution.exec_id,
            )
    return recorder.fill_count() - before


def recheck(config: TraderConfig, *, run: QuantlabRun | None = None) -> dict:
    """Run the Decision recheck on the live run directory and write it there.

    Each decided bar is decided again from the live prediction store's row,
    the symbols the day left out masked, and the current weights recorded as
    handed to the rule; the result is written to ``RECHECK_FILE``.

    Returns
    -------
    dict
        ``decision_recheck``'s result.

    Examples
    --------
    ::

        recheck(config)["bars_differing"] == 0
    """
    from quantlab_ibkr.parity.decision_recheck import decision_recheck

    venue_config = live_venue(config)
    run = run or QuantlabRun.load(config.quantlab_run)
    live_dir = Path(venue_config.live_dir)
    result = decision_recheck(
        run, live_dir, predictions=LivePredictionStore(venue_config.prediction_store)
    )
    (live_dir / RECHECK_FILE).write_text(json.dumps(jsonable(result), indent=2))
    return result

