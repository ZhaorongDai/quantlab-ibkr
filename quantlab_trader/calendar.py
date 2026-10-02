"""``RebalanceCalendar``: which bars a closed-loop replay rebalances on (ADR 0008).

Rebalance bars are counted every ``rebalance_periods`` bars on the price
dataset's calendar from the **anchor**, the first timestamp of the quantlab
run's prediction panel. Narrowing a replay's window does not move them, and
the replay's last bar never rebalances: an order decided there has no next
open to fill at (quantlab's ``rebalance_mask`` excludes it the same way).
"""

from __future__ import annotations

import pandas as pd


class RebalanceCalendar:
    """The rebalance bars of a replay, anchored on the run's prediction panel.

    Parameters
    ----------
    bars : pandas.DatetimeIndex
        The price dataset's bars from the anchor (its first entry) to the
        replay's last bar.
    rebalance_periods : int
        Bars between rebalances, at least 1.

    Raises
    ------
    ValueError
        If ``bars`` is empty or ``rebalance_periods`` is smaller than 1.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-01", periods=6)
    >>> calendar = RebalanceCalendar(bars, 2)
    >>> [calendar.rebalances(b) for b in bars]
    [True, False, True, False, True, False]
    >>> RebalanceCalendar(bars[:5], 2).rebalances(bars[4])  # the replay's last bar
    False
    """

    def __init__(self, bars: pd.DatetimeIndex, rebalance_periods: int):
        if rebalance_periods < 1:
            raise ValueError(f"rebalance_periods must be >= 1, got {rebalance_periods}")
        bars = pd.DatetimeIndex(bars)
        if not len(bars):
            raise ValueError("RebalanceCalendar needs at least the anchor bar")
        self.anchor = bars[0]
        self.rebalance_periods = int(rebalance_periods)
        self._bars = bars

    def rebalances(self, t: pd.Timestamp) -> bool:
        """Return whether bar ``t`` is a rebalance bar.

        A bar before the anchor or off the price calendar never is.

        Examples
        --------
        >>> calendar = RebalanceCalendar(pd.bdate_range("2024-01-02", periods=4), 2)
        >>> calendar.rebalances(pd.Timestamp("2024-01-04")), calendar.rebalances(pd.Timestamp("2024-01-01"))
        (True, False)
        """
        position = self._bars.get_indexer([pd.Timestamp(t)])[0]
        if position < 0 or position == len(self._bars) - 1:
            return False
        return bool(position % self.rebalance_periods == 0)
