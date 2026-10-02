"""The backtest venue's decision source: one bar of the run's price dataset at a time."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab_trader.base.venue import DecisionInputs, DecisionSource
from quantlab_trader.quantlab_run import QuantlabRun
from quantlab_trader.venue.backtest.resolver import PRICE_PRECISION


@dataclass(frozen=True)
class DelistingSettlement:
    """A delisted security's holding, settled into cash on the bar after its delisting bar.

    Attributes
    ----------
    permno : Hashable
        The security.
    delisting_date : pandas.Timestamp
        Its delisting bar b (``MarketDataset.delisting_bars``).
    settlement_date : pandas.Timestamp
        Bar b + 1, whose open the holding is settled at.
    price : float
        The last valuation in raw prices, at the instruments' precision.
    """

    permno: Hashable
    delisting_date: pd.Timestamp
    settlement_date: pd.Timestamp
    price: float


class BacktestDecisionSource(DecisionSource):
    """Read decision inputs from the run's price dataset, nothing later than t.

    ``tradable`` and ``delisted`` are the dataset's own
    ``tradable_bars``/``delisting_bars`` on the run's market columns, the raw
    close is carried forward over a security's missing bars (so a halted
    holding is marked at its last close), and the decision prices are the
    run's valuation column from the window's first bar to t.

    On a delisting bar b the raw close is the security's **last valuation**
    in raw prices: its last raw close grown by the valuation column's return
    since that close. On a CRSP delisting row, which has no raw price and an
    adjusted close carrying the delisting return, that is the delisting
    return applied to the last raw close; on a bar with a raw close it is
    that close. A holding is marked at it on b and settled at it on b + 1
    (quantlab ADR 0014), as quantlab values and settles it at the valuation
    column.

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
        valuation_column = run.market["valuation_price_column"]
        self._tradable = (
            dataset.tradable_bars(prices, run.market["fill_price_column"]).to_pandas()
        )
        self._delisted = dataset.delisting_bars(prices, valuation_column).to_pandas()
        self._valuation = prices[valuation_column].transpose("timestamp", "symbol")
        raw_close = prices["close"].transpose("timestamp", "symbol").to_pandas()
        last_value = self._last_valuation(raw_close, self._valuation.to_pandas())
        self._settlements = self._delisting_settlements(last_value)
        self._close = raw_close.mask(self._delisted, last_value).ffill()

    @staticmethod
    def _last_valuation(raw_close: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
        """Return each bar's last valuation in raw prices, at the instruments' precision.

        The last raw close times the valuation's return since that close; the
        last raw close itself where the valuation cannot say.
        """
        last_close = raw_close.ffill()
        valuation_at_last_close = valuation.where(raw_close.notna()).ffill()
        grown = last_close * valuation / valuation_at_last_close
        value = grown.where(np.isfinite(grown) & (grown > 0), last_close)
        return value.round(PRICE_PRECISION)

    def _delisting_settlements(self, last_value: pd.DataFrame) -> tuple[DelistingSettlement, ...]:
        """Return the settlement of every delisting bar that has a next bar in the window."""
        settlements = []
        bars, symbols = np.nonzero(self._delisted.to_numpy())
        for b, s in sorted(zip(bars, symbols)):
            price = last_value.iat[b, s]
            if b + 1 >= len(self._calendar) or not np.isfinite(price):
                continue
            settlements.append(
                DelistingSettlement(
                    permno=self._delisted.columns[s],
                    delisting_date=self._calendar[b],
                    settlement_date=self._calendar[b + 1],
                    price=float(price),
                )
            )
        return tuple(settlements)

    def calendar(self) -> pd.DatetimeIndex:
        """Return the window's bars."""
        return self._calendar

    def delisting_settlements(self) -> tuple[DelistingSettlement, ...]:
        """Return the window's delisting settlements, in time order."""
        return self._settlements

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
