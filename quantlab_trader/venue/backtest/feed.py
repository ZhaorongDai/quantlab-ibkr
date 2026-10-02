"""The backtest venue's market data: opening and closing prints and daily bars (ADR 0003).

Per instrument and day the feed carries a ``TradeTick`` at 09:30 ET at the raw
open (the opening print orders fill at), a ``TradeTick`` at 16:00 ET at the raw
close (the closing print) and the daily ``Bar`` at 16:00 ET, all at the
instrument's price precision. A missing price is no print.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from nautilus_trader.model.data import Bar, BarType, TradeTick
from nautilus_trader.model.enums import AggressorSide
from nautilus_trader.model.identifiers import TradeId
from nautilus_trader.model.objects import Price, Quantity

from quantlab_trader.venue.backtest.resolver import PRICE_PRECISION, BacktestResolver

#: The exchange time zone of the session.
SESSION_TZ = "America/New_York"
#: Time of the opening print.
OPEN_TIME = pd.Timedelta(hours=9, minutes=30)
#: Time of the closing print and the daily bar.
CLOSE_TIME = pd.Timedelta(hours=16)
#: Size of every print: large enough that no order is ever short of liquidity.
PRINT_SIZE = Quantity(10_000_000_000, 0)


def session_ns(dates: pd.DatetimeIndex | pd.Timestamp, time: pd.Timedelta):
    """Return UNIX nanoseconds of ``time`` ET on each of ``dates`` (naive dates)."""
    if isinstance(dates, pd.Timestamp):
        return int(session_ns(pd.DatetimeIndex([dates]), time)[0])
    stamps = (pd.DatetimeIndex(dates).normalize() + time).tz_localize(SESSION_TZ)
    return stamps.tz_convert("UTC").as_unit("ns").asi8


def build_feed(prices: xr.Dataset, resolver: BacktestResolver) -> list:
    """Return the feed's ticks and bars for every resolved instrument.

    Parameters
    ----------
    prices : xarray.Dataset
        Raw ``open`` and ``close`` on ``(timestamp, symbol)``, and ``high``
        and ``low`` when the dataset has them (the bar's range otherwise
        spans the open and close).
    resolver : BacktestResolver
        Maps each symbol to its instrument.

    Returns
    -------
    list
        ``TradeTick`` and ``Bar`` objects, unsorted.
    """
    dates = pd.DatetimeIndex(prices["timestamp"].values)
    open_ns, close_ns = session_ns(dates, OPEN_TIME), session_ns(dates, CLOSE_TIME)
    data: list = []
    for instrument in resolver.instruments():
        permno = resolver.permno(instrument.id)
        row = prices.sel(symbol=permno)
        opens, closes = row["open"].values, row["close"].values
        highs, lows = np.fmax(opens, closes), np.fmin(opens, closes)
        if "high" in row:
            highs = np.fmax(row["high"].values, highs)
        if "low" in row:
            lows = np.fmin(row["low"].values, lows)
        bar_type = BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL")
        for i in range(len(dates)):
            if np.isfinite(opens[i]):
                data.append(_print(instrument.id, opens[i], f"{permno}-{i}-O", open_ns[i]))
            if np.isfinite(closes[i]):
                data.append(_print(instrument.id, closes[i], f"{permno}-{i}-C", close_ns[i]))
            if np.isfinite(opens[i]) and np.isfinite(closes[i]):
                data.append(
                    Bar(
                        bar_type,
                        _price(opens[i]),
                        _price(highs[i]),
                        _price(lows[i]),
                        _price(closes[i]),
                        PRINT_SIZE,
                        int(close_ns[i]),
                        int(close_ns[i]),
                    )
                )
    return data


def _price(value: float) -> Price:
    """Return ``value`` as a price at the instruments' precision."""
    return Price(round(float(value), PRICE_PRECISION), PRICE_PRECISION)


def _print(instrument_id, price: float, trade_id: str, ts: int) -> TradeTick:
    """Return one print of ``instrument_id`` at ``price`` stamped ``ts``."""
    return TradeTick(
        instrument_id,
        _price(price),
        PRINT_SIZE,
        AggressorSide.NO_AGGRESSOR,
        TradeId(trade_id),
        int(ts),
        int(ts),
    )
