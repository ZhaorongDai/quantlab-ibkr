"""The backtest venue's open submitter: DAY market orders at the next open + 1 ns (ADR 0003).

nautilus 1.231 rejects ``AT_THE_OPEN`` orders, so the submitter holds the
orders decided at the close of t and, one nanosecond after the opening prints
of the next bar, submits each as a plain ``MarketOrder`` with
``TimeInForce.DAY``, sells first; it fills at the opening print. An order
decided on the window's last bar has no next open and is reported unfilled.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
from nautilus_trader.model.enums import TimeInForce

from quantlab_trader.base.venue import NextOpenOrder, OpenSubmitter
from quantlab_trader.venue.backtest.feed import OPEN_TIME, session_ns

if TYPE_CHECKING:
    from quantlab_trader.strategy import PortfolioStrategy

#: Reason recorded for an order decided on the last bar of the window.
NO_NEXT_OPEN = "no next open in the backtest window"


class BacktestOpenSubmitter(OpenSubmitter):
    """Queue next-open orders and release them at the next bar's open + 1 ns.

    Parameters
    ----------
    calendar : pandas.DatetimeIndex
        The window's bars; the next open of t is the open of the bar after t.
    """

    def __init__(self, calendar: pd.DatetimeIndex):
        self._calendar = pd.DatetimeIndex(calendar)
        self._strategy: PortfolioStrategy | None = None

    def attach(self, strategy: PortfolioStrategy) -> None:
        """Bind the strategy that submits the orders."""
        self._strategy = strategy

    def submit(self, orders: Sequence[NextOpenOrder], decision_date: pd.Timestamp) -> None:
        """Queue ``orders`` for the open after ``decision_date``."""
        if not orders:
            return
        strategy = self._strategy
        position = self._calendar.searchsorted(decision_date, side="right")
        if position >= len(self._calendar):
            for order in orders:
                strategy.on_next_open_unfilled(order, NO_NEXT_OPEN)
            return
        next_bar = self._calendar[position]
        queued = sorted(orders, key=lambda order: order.side != "SELL")
        strategy.clock.set_time_alert(
            f"next-open-{next_bar.date()}",
            pd.Timestamp(session_ns(next_bar, OPEN_TIME) + 1, tz="UTC"),
            lambda _event: self._release(queued),
        )

    def _release(self, orders: Sequence[NextOpenOrder]) -> None:
        """Submit the queued orders as DAY market orders, in queue order."""
        for order in orders:
            self._strategy.submit_next_open(order, TimeInForce.DAY)
