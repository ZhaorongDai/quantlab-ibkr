"""``RebalanceCalendar`` re-implements quantlab's ``rebalance_mask``; this locks them together.

trader cannot import quantlab's backtest layer outside the ``parity``
package (ADR 0008), so the calendar counts rebalance bars itself. Anchored
on a window's first bar, it must mark exactly the bars ``rebalance_mask``
marks.
"""

import pandas as pd
import pytest

from quantlab.backtest.selection import rebalance_mask
from quantlab_trader.calendar import RebalanceCalendar


@pytest.mark.parametrize("n_bars", [1, 2, 3, 7, 10, 31])
@pytest.mark.parametrize("rebalance_periods", [1, 2, 3, 5, 21])
def test_the_calendar_marks_the_bars_rebalance_mask_marks(n_bars, rebalance_periods):
    bars = pd.bdate_range("2024-01-02", periods=n_bars)
    calendar = RebalanceCalendar(bars, rebalance_periods)

    marked = [calendar.rebalances(bar) for bar in bars]

    assert marked == rebalance_mask(n_bars, rebalance_periods).tolist()
