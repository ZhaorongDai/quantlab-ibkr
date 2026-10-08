"""The IBKR venue's open submitter: market-on-open orders before a deadline (ADR 0003).

Each ``NextOpenOrder`` decided at the close of t is submitted at once as a
nautilus ``MarketOrder`` with ``TimeInForce.AT_THE_OPEN``, which the IB
adapter sends as IBKR's ``MKT`` / ``OPG``: it fills in the opening auction of
the session after t, the open the backtest fills at. Sells go first, and at
most ``max_orders_per_second`` orders leave per second (IBKR paces API
messages; the rest follow on time alerts one second apart).

An order is refused once the deadline has passed: ``order_deadline`` (default
09:20, America/New_York) on the weekday after t, ahead of the exchanges'
opening-auction cut-off (~09:28). A refused order is reported unfilled with
the reason, as is every order of a dry run, which decides and reports
without submitting. While the node runs, an order IBKR rejects, cancels or
expires is reported through ``strategy.on_next_open_unfilled``; what happens
at the open, after the node has stopped, is the record step's (#51).

An order already **working** at IBKR for the same decision (an earlier run of
the day that submitted it, then failed before recording it) is never sent
again: the submitter adopts the working order (``working``, keyed by symbol
and side) and records its id as the submitted one.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Hashable, Mapping, Sequence
from typing import TYPE_CHECKING

import pandas as pd
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderCanceled, OrderExpired, OrderRejected

from quantlab_ibkr.base.venue import MARKET_TZ, InstrumentResolver, NextOpenOrder, OpenSubmitter

if TYPE_CHECKING:
    from quantlab_ibkr.strategy import PortfolioStrategy

#: The default order deadline, market time.
ORDER_DEADLINE = "09:20"
#: Orders submitted per second at most.
MAX_ORDERS_PER_SECOND = 40

#: Reasons recorded for an order reported unfilled.
DRY_RUN = "dry run: not submitted"
PAST_DEADLINE = "not submitted: past the order deadline {deadline}"
ENDED_UNFILLED = "{status} by IBKR: {reason}"


def parse_deadline(text: str) -> datetime.time:
    """Return the ``HH:MM`` deadline as a time, refusing anything else.

    Examples
    --------
    >>> parse_deadline("09:20")
    datetime.time(9, 20)
    >>> parse_deadline("9.20")
    Traceback (most recent call last):
    ...
    ValueError: order_deadline must be HH:MM in market time, got '9.20'
    """
    try:
        return datetime.datetime.strptime(text, "%H:%M").time()
    except (TypeError, ValueError):
        raise ValueError(f"order_deadline must be HH:MM in market time, got {text!r}") from None


def order_deadline(decision_date: pd.Timestamp, deadline: str = ORDER_DEADLINE) -> pd.Timestamp:
    """Return when orders decided at the close of ``decision_date`` must be submitted by.

    ``deadline`` market time on the weekday after ``decision_date``. An
    exchange holiday is not skipped, so a run on a holiday morning after the
    deadline is refused although the auction is a day later: conservative.

    Examples
    --------
    >>> order_deadline(pd.Timestamp("2026-10-09"))  # a Friday: Monday 09:20 New York
    Timestamp('2026-10-12 09:20:00-0400', tz='America/New_York')
    """
    day = (pd.Timestamp(decision_date).normalize() + pd.offsets.BDay(1)).date()
    return pd.Timestamp(datetime.datetime.combine(day, parse_deadline(deadline))).tz_localize(
        MARKET_TZ
    )


class IbkrOpenSubmitter(OpenSubmitter):
    """Submit next-open orders as IBKR market-on-open orders, sells first, before the deadline.

    Parameters
    ----------
    resolver : InstrumentResolver
        Maps an order's PERMNO to its instrument, so an IBKR order event can
        be traced back to the order.
    order_deadline : str, default ORDER_DEADLINE
        ``HH:MM`` market time on the weekday after the decision date.
    dry_run : bool, default False
        Report every order unfilled (``DRY_RUN``) instead of submitting it.
    max_orders_per_second : int, default MAX_ORDERS_PER_SECOND
        Orders submitted per second at most.
    working : Mapping, optional
        ``(symbol, side) -> client order id`` of the orders already working
        at IBKR for the decision date; a decided order matching one is
        adopted instead of submitted.

    Attributes
    ----------
    decided : list of NextOpenOrder
        Every order handed over, in submission order (sells first).
    submitted : list of NextOpenOrder
        The orders submitted to IBKR.
    unfilled : list of tuple
        ``(order, reason)`` of each order reported unfilled.
    adopted : list of NextOpenOrder
        The orders found working at IBKR and not submitted again.
    released : bool
        Every order handed over has been submitted or reported.

    Examples
    --------
    >>> from quantlab_ibkr.venue.backtest.resolver import BacktestResolver
    >>> resolver = BacktestResolver([10001], pd.Timestamp("2026-10-07"))
    >>> isinstance(IbkrOpenSubmitter(resolver), OpenSubmitter)
    True
    """

    def __init__(
        self,
        resolver: InstrumentResolver,
        *,
        order_deadline: str = ORDER_DEADLINE,
        dry_run: bool = False,
        max_orders_per_second: int = MAX_ORDERS_PER_SECOND,
        working: Mapping[tuple[Hashable, str], str] | None = None,
    ):
        parse_deadline(order_deadline)
        if max_orders_per_second < 1:
            raise ValueError(
                f"max_orders_per_second must be at least 1, got {max_orders_per_second}"
            )
        self.resolver = resolver
        self.order_deadline = order_deadline
        self.dry_run = dry_run
        self.max_orders_per_second = max_orders_per_second
        self.decided: list[NextOpenOrder] = []
        self.submitted: list[NextOpenOrder] = []
        self.unfilled: list[tuple[NextOpenOrder, str]] = []
        self.adopted: list[NextOpenOrder] = []
        self.working = dict(working or {})
        self.released = True
        self._strategy: PortfolioStrategy | None = None
        self._by_instrument: dict[Hashable, NextOpenOrder] = {}
        self._ended: set = set()

    def attach(self, strategy: PortfolioStrategy) -> None:
        """Bind the strategy and listen to its order events.

        Examples
        --------
        ``PortfolioStrategy.on_start`` attaches itself::

            venue.submitter.attach(strategy)
        """
        self._strategy = strategy
        strategy.msgbus.subscribe(topic=f"events.order.{strategy.id}", handler=self.on_order_event)

    def submit(self, orders: Sequence[NextOpenOrder], decision_date: pd.Timestamp) -> None:
        """Submit ``orders`` as market-on-open orders now, sells first, paced.

        Refused (reported unfilled) after the deadline; reported, not
        submitted, in a dry run.

        Examples
        --------
        ``PortfolioStrategy.on_decision_time`` hands over the cycle's orders::

            venue.submitter.submit(result.orders, t)
        """
        queued = sorted(orders, key=lambda order: order.side != "SELL")
        self.decided.extend(queued)
        if self.working and not self.dry_run:
            queued = [order for order in queued if not self._adopt(order)]
        if not queued:
            return
        if self.dry_run:
            for order in queued:
                self._report(order, DRY_RUN)
            return
        self.released = False
        batch = self.max_orders_per_second
        self._release(queued[:batch], decision_date)
        rest = [queued[k:k + batch] for k in range(batch, len(queued), batch)]
        if not rest:
            self.released = True
            return
        now = pd.Timestamp(self._strategy.clock.utc_now())
        for k, orders_k in enumerate(rest, start=1):
            self._strategy.clock.set_time_alert(
                f"next-open-{pd.Timestamp(decision_date).date()}-{k}",
                now + pd.Timedelta(seconds=k),
                lambda _event, orders_k=orders_k, last=k == len(rest): self._release(
                    orders_k, decision_date, last=last
                ),
            )

    def _release(
        self, orders: Sequence[NextOpenOrder], decision_date: pd.Timestamp, *, last: bool = False
    ) -> None:
        """Submit ``orders`` unless the deadline has passed; then mark ``released`` if ``last``."""
        strategy = self._strategy
        deadline = order_deadline(decision_date, self.order_deadline)
        now = pd.Timestamp(strategy.clock.utc_now()).tz_convert(MARKET_TZ)
        for order in orders:
            if now >= deadline:
                self._report(order, PAST_DEADLINE.format(deadline=deadline))
                continue
            instrument_id = self.resolver.instrument_id(order.permno, order.decision_date)
            self._by_instrument[instrument_id] = order
            self.submitted.append(order)
            strategy.submit_next_open(order, TimeInForce.AT_THE_OPEN)
        if last:
            self.released = True

    def _adopt(self, order: NextOpenOrder) -> bool:
        """Record ``order`` as the working IBKR order of its symbol and side, if there is one."""
        client_order_id = self.working.pop((order.permno, order.side), None)
        if client_order_id is None:
            return False
        self.adopted.append(order)
        self._strategy.recorder.order_submitted(order, client_order_id)
        return True

    def _report(self, order: NextOpenOrder, reason: str) -> None:
        self.unfilled.append((order, reason))
        self._strategy.on_next_open_unfilled(order, reason)

    def on_order_event(self, event) -> None:
        """Report a submitted order IBKR rejected, canceled or expired, once.

        The order is reported with its unfilled quantity; a fill before the
        cancellation is the strategy's (``on_order_filled``).

        Examples
        --------
        The strategy's order events reach it through the message bus once
        attached::

            msgbus.publish(f"events.order.{strategy.id}", OrderCanceled(...))
        """
        if not isinstance(event, (OrderRejected, OrderCanceled, OrderExpired)):
            return
        order = self._by_instrument.get(event.instrument_id)
        if order is None or event.client_order_id in self._ended:
            return
        self._ended.add(event.client_order_id)
        status = {OrderRejected: "rejected", OrderCanceled: "canceled", OrderExpired: "expired"}
        reason = str(getattr(event, "reason", "") or "no reason given")
        filled = 0
        cached = self._strategy.cache.order(event.client_order_id)
        if cached is not None:
            filled = int(cached.filled_qty.as_decimal())
        if filled:
            reason = f"{reason}; {filled} of {order.quantity} shares filled first"
            order = dataclasses.replace(order, quantity=order.quantity - filled)
        self._report(order, ENDED_UNFILLED.format(status=status[type(event)], reason=reason))
