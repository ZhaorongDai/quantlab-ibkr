"""The run's prices over the ladder's window, and the ``[T, S]`` array helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from quantlab_ibkr.quantlab_run import QuantlabRun

#: Decimals of a price and of money in trader's backtest venue.
PRICE_DECIMALS, MONEY_DECIMALS = 4, 2


@dataclass(frozen=True)
class Market:
    """The run's prices over its window, as ``[T, S]`` arrays on the table's axes.

    ``fill``/``valuation`` are the run's (adjusted) market columns as read,
    NaN where the market had no price; ``open``/``close`` the raw prices;
    ``delisted`` the dataset's delisting bars on the valuation column;
    ``split_factor``, ``cumfacshr`` and ``dividend`` the CRSP fields,
    ``adj_close`` CRSP's ``adjClose`` (chained from its total return), which
    the price-implied share changes are read from (#27).

    Examples
    --------
    A run's window, on the symbols its table ever trades::

        run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
        table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
        market = Market.load(run, table)
        market.close.shape == (len(market.timestamps), len(market.symbols))
    """

    timestamps: pd.DatetimeIndex
    symbols: np.ndarray
    weights: np.ndarray
    fill: np.ndarray
    valuation: np.ndarray
    open: np.ndarray
    close: np.ndarray
    delisted: np.ndarray
    split_factor: np.ndarray
    cumfacshr: np.ndarray
    dividend: np.ndarray
    adj_close: np.ndarray

    @classmethod
    def load(cls, run: QuantlabRun, table: xr.DataArray) -> Market:
        """Load the window of ``table`` for the symbols it ever gives a nonzero target.

        A symbol never given one is never held by any rung, so it is left
        out (a market-wide table has thousands of them).

        Examples
        --------
        ::

            run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
            table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
            market = Market.load(run, table)
            market.weights.shape == market.close.shape
        """
        traded = (np.nan_to_num(np.asarray(table.values, dtype=np.float64)) != 0.0).any(axis=0)
        table = table.isel(symbol=np.flatnonzero(traded))
        timestamps = pd.DatetimeIndex(table["timestamp"].values)
        symbols = table["symbol"].values
        dataset = run.price_dataset
        prices = (
            dataset.panel(timestamps[0], timestamps[-1], symbols=list(symbols))
            .load()
            .reindex(timestamp=timestamps, symbol=symbols)
        )
        valuation_column = run.market.valuation_price_column

        def array(name):
            return np.asarray(
                prices[name].transpose("timestamp", "symbol").values, dtype=np.float64
            )

        return cls(
            timestamps=timestamps,
            symbols=symbols,
            weights=np.asarray(table.values, dtype=np.float64),
            fill=array(run.market.fill_price_column),
            valuation=array(valuation_column),
            open=array("open"),
            close=array("close"),
            delisted=np.asarray(
                dataset.delisting_bars(prices, valuation_column)
                .transpose("timestamp", "symbol")
                .values,
                dtype=bool,
            ),
            split_factor=array("splitFactor"),
            cumfacshr=array("cumfacshr"),
            dividend=array("divCash"),
            adj_close=array("adjClose"),
        )


def ffill(values: np.ndarray) -> np.ndarray:
    """Forward-fill a ``[T, S]`` array along time.

    Examples
    --------
    >>> ffill(np.array([[1.0, np.nan], [np.nan, 2.0]]))
    array([[ 1., nan],
           [ 1.,  2.]])
    """
    return pd.DataFrame(values).ffill().to_numpy()


def shift(values: np.ndarray) -> np.ndarray:
    """Shift a ``[T, S]`` array one bar later; the first row is NaN.

    Examples
    --------
    >>> shift(np.array([[1.0], [2.0]]))
    array([[nan],
           [ 1.]])
    """
    out = np.full_like(values, np.nan, dtype=np.float64)
    out[1:] = values[:-1]
    return out
