"""The account as the decision cycle sees it: holdings by PERMNO and derived cash.

Read from nautilus's ``Cache``, which presents one account interface in the
backtest and live. Cash is *derived*: a MARGIN account's balance does not
deduct the cost of open positions, so cash is the balance minus
``sum(signed_qty * avg_px_open)`` over open positions. nautilus's unrealised
PnL is never used; equity is marked by the decision cycle at t's raw close.
"""

from __future__ import annotations

from collections.abc import Hashable

from quantlab_ibkr.base.venue import InstrumentResolver


def holdings(cache, resolver: InstrumentResolver) -> dict[Hashable, int]:
    """Return the signed whole-share holding of every open position, by PERMNO.

    Parameters
    ----------
    cache : nautilus_trader.cache.Cache
        The trader's cache.
    resolver : InstrumentResolver
        Maps each position's instrument back to its PERMNO.

    Returns
    -------
    dict
        Signed share counts by PERMNO; flat positions are left out.

    Examples
    --------
    A stand-in for nautilus's cache, holding one short position:

    >>> from decimal import Decimal
    >>> from types import SimpleNamespace
    >>> import pandas as pd
    >>> from quantlab_ibkr.venue.backtest.resolver import BacktestResolver
    >>> resolver = BacktestResolver([10107], pd.Timestamp("2024-01-02"))
    >>> position = SimpleNamespace(
    ...     instrument_id=resolver.instrument_id(10107, pd.Timestamp("2024-01-02")),
    ...     signed_decimal_qty=lambda: Decimal("-25"),
    ... )
    >>> holdings(SimpleNamespace(positions_open=lambda: [position]), resolver)
    {10107: -25}
    """
    held: dict[Hashable, int] = {}
    for position in cache.positions_open():
        # Read from the decimal, as on_order_filled reads fills: the float
        # signed_qty is inexact for many share counts.
        quantity = int(position.signed_decimal_qty())
        if quantity:
            permno = resolver.permno(position.instrument_id)
            held[permno] = held.get(permno, 0) + quantity
    return held


def derived_cash(cache) -> float:
    """Return the account's cash: margin balance minus the open positions' cost.

    Parameters
    ----------
    cache : nautilus_trader.cache.Cache
        The trader's cache, holding one account.

    Returns
    -------
    float
        The account's cash.

    Raises
    ------
    ValueError
        If the cache does not hold exactly one account.

    Examples
    --------
    ``PortfolioStrategy.on_decision_time`` reads the account this way::

        result = cycle.run(inputs, holdings(strategy.cache, resolver), derived_cash(strategy.cache))
    """
    accounts = cache.accounts()
    if len(accounts) != 1:
        raise ValueError(f"expected one account, found {len(accounts)}")
    account = accounts[0]
    balance = account.balance_total(account.base_currency).as_double()
    cost = sum(
        float(position.signed_decimal_qty()) * position.avg_px_open
        for position in cache.positions_open()
    )
    return balance - cost
