"""The backtest venue's decision source: one bar of the run's price dataset at a time."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

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
    run's valuation column from ``history_start`` (the window's first bar by
    default) to t.

    With ``predictions`` (closed loop) each bar's inputs carry its row of
    the prediction panel, and ``tradable`` and the decision prices are on
    the panel's symbols, which a rule's context is built on (quantlab's
    panel loop builds it on the same ones); otherwise they are on
    ``permnos``.

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
    predictions : xarray.Dataset, optional
        The prediction panel over the window, one variable per label on
        ``(timestamp, symbol)``.
    history_start : pandas.Timestamp, optional
        The first bar of the decision-price history; ``start`` by default.
    """

    def __init__(
        self,
        run: QuantlabRun,
        start: pd.Timestamp,
        end: pd.Timestamp,
        permnos: Sequence,
        *,
        predictions: xr.Dataset | None = None,
        history_start: pd.Timestamp | None = None,
    ):
        dataset = run.price_dataset
        fill_column = run.market["fill_price_column"]
        valuation_column = run.market["valuation_price_column"]
        prices = dataset.panel(start, end, symbols=list(permnos)).load()
        self.prices = prices
        self._calendar = pd.DatetimeIndex(prices["timestamp"].values)
        self._predictions = predictions
        decision_symbols = (
            list(permnos) if predictions is None else list(predictions["symbol"].values)
        )
        history = (
            dataset.panel(
                start if history_start is None else history_start,
                end,
                symbols=decision_symbols,
            )[[fill_column, valuation_column]]
            .transpose("timestamp", "symbol")
            .load()
        )
        self._tradable = (
            dataset.tradable_bars(history.sel(timestamp=slice(start, end)), fill_column)
            .to_pandas()
        )
        self._valuation = history[valuation_column]
        self._delisted = dataset.delisting_bars(prices, valuation_column).to_pandas()
        raw_close = prices["close"].transpose("timestamp", "symbol").to_pandas()
        last_value = self._last_valuation(
            raw_close, prices[valuation_column].transpose("timestamp", "symbol").to_pandas()
        )
        self._settlements = self._delisting_settlements(last_value)
        self._close = raw_close.mask(self._delisted, last_value).ffill()

    @staticmethod
    def _last_valuation(raw_close: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
        """Return each bar's last valuation in raw prices, at the instruments' precision.

        The last raw close times the valuation's return since that close; the
        last raw close itself where the valuation cannot say.
        """
        # Both from the same bar: the last one with a raw close and a valuation.
        paired = raw_close.notna() & valuation.notna()
        last_close = raw_close.ffill()
        grown = raw_close.where(paired).ffill() * valuation / valuation.where(paired).ffill()
        # A total-loss delisting (valuation 0) settles at 0, not at the last close.
        value = grown.where(np.isfinite(grown) & (grown >= 0), last_close)
        return value.round(PRICE_PRECISION)

    def _delisting_settlements(self, last_value: pd.DataFrame) -> tuple[DelistingSettlement, ...]:
        """Return the settlement of every delisting bar that has a next bar in the window."""
        settlements = []
        bars, columns = np.nonzero(self._delisted.to_numpy())
        for bar, column in sorted(zip(bars, columns)):
            price = last_value.iat[bar, column]
            if bar + 1 >= len(self._calendar) or not np.isfinite(price):
                continue
            settlements.append(
                DelistingSettlement(
                    permno=self._delisted.columns[column],
                    delisting_date=self._calendar[bar],
                    settlement_date=self._calendar[bar + 1],
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
        predictions = None
        if self._predictions is not None and t in self._predictions.indexes["timestamp"]:
            predictions = self._predictions.sel(timestamp=t, drop=True)
        return DecisionInputs(
            timestamp=t,
            predictions=predictions,
            tradable=self._tradable.loc[t],
            close=self._close.loc[t],
            valuation_history=self._valuation.sel(timestamp=slice(None, t)),
            delisted=self._delisted.loc[t],
        )
