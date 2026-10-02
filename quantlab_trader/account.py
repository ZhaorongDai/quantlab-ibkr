"""The account as the decision cycle sees it: holdings by PERMNO and derived cash.

Read from nautilus's ``Cache``, which presents one account interface in the
backtest and live. Cash is *derived*: a MARGIN account's balance does not
deduct the cost of open positions, so cash is the balance minus
``sum(signed_qty * avg_px_open)`` over open positions. nautilus's unrealised
PnL is never used; equity is marked by the decision cycle at t's raw close.
"""

from __future__ import annotations

from collections.abc import Hashable

from quantlab_trader.base.venue import InstrumentResolver


def holdings(cache, resolver: InstrumentResolver) -> dict[Hashable, int]:
    """Return the signed whole-share holding of every open position, by PERMNO.

    Parameters
    ----------
    cache : nautilus_trader.cache.Cache
        The trader's cache.
    resolver : InstrumentResolver
        Maps each position's instrument back to its PERMNO.
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

    Raises
    ------
    ValueError
        If the cache does not hold exactly one account.
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
