"""The backtest venue's decision source: one bar of the run's price dataset at a time."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quantlab_trader.base.venue import DecisionInputs, DecisionSource
from quantlab_trader.quantlab_run import QuantlabRun


class BacktestDecisionSource(DecisionSource):
    """Read decision inputs from the run's price dataset, nothing later than t.

    ``tradable`` and ``delisted`` are the dataset's own
    ``tradable_bars``/``delisting_bars`` on the run's market columns, the raw
    close is carried forward over a security's missing bars (so a halted
    holding is marked at its last close), and the decision prices are the
    run's valuation column from the window's first bar to t.

    Parameters
    ----------
    run : QuantlabRun
        The run whose price dataset is read.
    start, end : pandas.Timestamp
        The window, inclusive.
    permnos : Sequence
        The securities the strategy can trade.
    """

    def __init__(
        self,
        run: QuantlabRun,
        start: pd.Timestamp,
        end: pd.Timestamp,
        permnos: Sequence,
    ):
        dataset = run.price_dataset
        prices = dataset.panel(start, end, symbols=list(permnos)).load()
        self.prices = prices
        self._calendar = pd.DatetimeIndex(prices["timestamp"].values)
        self._close = prices["close"].transpose("timestamp", "symbol").to_pandas().ffill()
        self._tradable = (
            dataset.tradable_bars(prices, run.market["fill_price_column"]).to_pandas()
        )
        self._delisted = (
            dataset.delisting_bars(prices, run.market["valuation_price_column"]).to_pandas()
        )
        self._valuation = prices[run.market["valuation_price_column"]].transpose(
            "timestamp", "symbol"
        )

    def calendar(self) -> pd.DatetimeIndex:
        """Return the window's bars."""
        return self._calendar

    def inputs(self, t: pd.Timestamp) -> DecisionInputs:
        """Return the inputs of the decision at the close of ``t``."""
        return DecisionInputs(
            timestamp=t,
            predictions=None,
            tradable=self._tradable.loc[t],
            close=self._close.loc[t],
            valuation_history=self._valuation.sel(timestamp=slice(None, t)),
            delisted=self._delisted.loc[t],
        )
