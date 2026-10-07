"""The backtest venue's decision clock: a time alert at close(t) + 1 ns (ADR 0003)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from quantlab_ibkr.base.venue import DecisionClock
from quantlab_ibkr.venue.backtest.feed import CLOSE_TIME, session_ns

if TYPE_CHECKING:
    from quantlab_ibkr.strategy import PortfolioStrategy


class BacktestDecisionClock(DecisionClock):
    """Fire the strategy's decision one nanosecond after each bar's closing print.

    The alert comes after every instrument's closing print and daily bar of
    that timestamp, so the decision sees the whole close.

    Parameters
    ----------
    calendar : pandas.DatetimeIndex
        The bars to decide on.

    Examples
    --------
    >>> clock = BacktestDecisionClock(pd.bdate_range("2024-01-02", periods=2))
    >>> isinstance(clock, DecisionClock)
    True
    """

    def __init__(self, calendar: pd.DatetimeIndex):
        self._calendar = pd.DatetimeIndex(calendar)

    def schedule(self, strategy: PortfolioStrategy) -> None:
        """Set one time alert per bar calling ``strategy.on_decision_time(t)``.

        Parameters
        ----------
        strategy : PortfolioStrategy
            The strategy whose nautilus clock takes the alerts.

        Examples
        --------
        A stand-in for the strategy's clock that prints each alert it is set:

        >>> from types import SimpleNamespace
        >>> class PrintingClock:
        ...     def set_time_alert(self, name, alert_time, callback):
        ...         print(name, alert_time)
        >>> clock = BacktestDecisionClock(pd.bdate_range("2024-01-02", periods=2))
        >>> clock.schedule(SimpleNamespace(clock=PrintingClock()))
        decision-2024-01-02 2024-01-02 21:00:00.000000001+00:00
        decision-2024-01-03 2024-01-03 21:00:00.000000001+00:00
        """
        fire_ns = session_ns(self._calendar, CLOSE_TIME) + 1
        for t, ns in zip(self._calendar, fire_ns):
            strategy.clock.set_time_alert(
                f"decision-{t.date()}",
                pd.Timestamp(int(ns), tz="UTC"),
                lambda _event, t=t: strategy.on_decision_time(t),
            )
