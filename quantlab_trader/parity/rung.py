"""One rung's simulation in the report's terms, and the ladder's fixed order."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


#: The rungs, in their fixed order (the gaps do not commute).
RUNGS = ("L0", "L1", "L2", "L3", "L4", "L5", "T")


#: What each rung changes from the rung above.
RUNG_DESCRIPTIONS = {
    "L0": "quantlab run_weights, as the run did",
    "L1": "sized at t's valuation close, not t+1's fill price (sizing_basis='valuation')",
    "L2": "reference ledger: cash no longer capped",
    "L3": "raw prices; splits change share counts, dividends and distributions are cash; "
    "trader's order and settlement rules",
    "L4": "whole shares (trunc); splits floor with cash in lieu",
    "L5": "trader's fee and slippage models, prices at 4 decimals, money at the cent",
    "T": "trader open-loop replay (nautilus)",
}


@dataclass
class RungResult:
    """One rung's simulation, in the terms the report compares.

    Attributes
    ----------
    name : str
        The rung.
    equity : pandas.Series
        Equity after each bar's close, on the window's bars.
    init_cash : float
        The starting cash.
    orders : pandas.DataFrame
        One row per fill of a strategy order (settlements are not orders):
        ``fill_bar``, ``symbol``, ``side`` (``BUY``/``SELL``), ``quantity``
        (positive), ``price``, ``fee``.
    rejected : list of tuple
        ``(fill bar, symbol)`` of every rejected order.
    settlements : list of tuple
        ``(settlement bar, symbol, price, quantity)`` of every delisting
        settlement of a holding; ``quantity`` is the signed holding, ``None``
        where quantlab's engine does not record it (L0, L1).
    max_target_deviation : float or None
        The largest gap between a target weight and the weight held after
        its fill bar, at the sizing prices.
    buys_capped : int or None
        Buys a cash cap cut (L0, L1) or would have cut (the ledger: a buy
        that leaves cash below zero, which vectorbt cuts or rejects);
        ``None`` for T, whose run does not record cash per fill.
    peak_cash_debit : float
        The most negative end-of-bar cash, as a positive number; 0 if never.
    positions : pandas.DataFrame or None
        The holdings after each bar, by symbol, where the comparison of
        loops needs them.
    run_dir : pathlib.Path or None
        A trader rung's run directory.

    Examples
    --------
    >>> bars = pd.bdate_range("2024-01-02", periods=2)
    >>> result = RungResult("L0", pd.Series([1e6, 1.01e6], index=bars), 1e6, pd.DataFrame())
    >>> round(float(result.equity.iloc[-1]) / result.init_cash - 1, 6), result.rejected
    (0.01, [])
    """

    name: str
    equity: pd.Series
    init_cash: float
    orders: pd.DataFrame
    rejected: list = field(default_factory=list)
    settlements: list = field(default_factory=list)
    max_target_deviation: float | None = None
    buys_capped: int | None = None
    peak_cash_debit: float = 0.0
    positions: pd.DataFrame | None = None
    run_dir: Path | None = None


def orders_frame(rows) -> pd.DataFrame:
    """The ``RungResult.orders`` frame of ``(fill_bar, symbol, side, quantity, price, fee)`` rows.

    Examples
    --------
    >>> orders_frame([(pd.Timestamp("2024-01-03"), 10001, "BUY", 100.0, 52.5, 0.53)])
        fill_bar  symbol side  quantity  price   fee
    0 2024-01-03   10001  BUY     100.0   52.5  0.53
    """
    return pd.DataFrame(rows, columns=["fill_bar", "symbol", "side", "quantity", "price", "fee"])
