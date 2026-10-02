"""L5's money as nautilus keeps it: cents, average open prices and realized PnL."""

from __future__ import annotations

import math

import numpy as np

from quantlab_trader.parity.market import MONEY_DECIMALS


class NautilusMoney:
    """L5's cash, kept as trader's account derives it from nautilus.

    trader's cash is a MARGIN account's balance less the open positions'
    cost, ``signed_qty * avg_px_open`` (``account.derived_cash``). nautilus
    books into that balance, per fill, the commission and, on a fill that
    reduces a position, the realized PnL ``closed_qty * (fill px -
    avg_px_open)`` (the reverse for a short) as ``Money``, rounded to the
    cent; a fill that adds to a position moves its average open price
    instead, ``(avg * qty + px * fill_qty) / (qty + fill_qty)``, and one that
    opens a position (or the rest of one that flips it) sets it to the fill
    price. Cash credits
    (dividends, cash in lieu, distributions) are cents. Kept this way, L5's
    equity is trader's to float precision, so a whole-share target on a
    truncation boundary is cut the same way in both; an exact-notional cash
    differs by the PnL's rounding, a few cents over thousands of fills.

    Examples
    --------
    >>> money = NautilusMoney(1000.0, 1)
    >>> money.fill(0, 0.0, 10.0, 50.0, 1.0)  # buy 10 at 50, fee 1
    >>> money.cash(np.array([10.0]))
    499.0
    >>> money.fill(0, 10.0, -10.0, 55.0, 1.0)  # sell them at 55: 50 of PnL
    >>> money.cash(np.array([0.0]))
    1048.0
    """

    def __init__(self, init_cash: float, n_symbols: int):
        self.cents = round(float(init_cash) * 10**MONEY_DECIMALS)
        self.avg = np.zeros(n_symbols)

    def credit(self, amount: float) -> None:
        """Book ``amount`` (at the cent) into the balance.

        Examples
        --------
        >>> money = NautilusMoney(1000.0, 1)
        >>> money.credit(0.4)  # a dividend
        >>> money.cents
        100040
        """
        self.cents += round(amount * 10**MONEY_DECIMALS)

    def fill(self, j: int, held: float, signed_qty: float, price: float, fee: float) -> None:
        """Book a fill of ``signed_qty`` at ``price`` on column ``j``, holding ``held`` before it.

        Examples
        --------
        >>> money = NautilusMoney(1000.0, 1)
        >>> money.fill(0, 0.0, 10.0, 50.0, 0.0)
        >>> money.fill(0, 10.0, 10.0, 60.0, 0.0)  # adds: the average open moves
        >>> float(money.avg[0])
        55.0
        """
        qty = abs(signed_qty)
        if held and (held > 0) != (signed_qty > 0):
            closed = min(qty, abs(held))
            points = price - self.avg[j] if held > 0 else self.avg[j] - price
            self.credit(nautilus_money(closed * 1.0 * points))
            if qty > abs(held):
                self.avg[j] = price
            elif qty == abs(held):
                self.avg[j] = 0.0
        elif held:
            start = abs(held)
            self.avg[j] = (self.avg[j] * start + price * qty) / (start + qty)
        else:
            # A new position opens at the fill price itself, not px * q / q.
            self.avg[j] = price
        self.credit(-fee)

    def cash(self, position: np.ndarray) -> float:
        """trader's derived cash: the balance less the open positions' cost.

        Examples
        --------
        >>> money = NautilusMoney(1000.0, 1)
        >>> money.fill(0, 0.0, -10.0, 50.0, 1.0)  # a short opens at 50
        >>> money.cash(np.array([-10.0]))
        1499.0
        """
        held = position != 0
        return self.cents / 10**MONEY_DECIMALS - float(np.sum(position[held] * self.avg[held]))


def nautilus_money(value: float) -> float:
    """Return ``value`` at the cent, as nautilus's ``Money`` stores a float.

    ``value * 100`` rounded half away from zero, in binary: 17.685 is 17.68,
    its double times 100 being 1768.4999...

    Examples
    --------
    >>> nautilus_money(17.685), nautilus_money(-0.125)
    (17.68, -0.13)
    """
    scaled = abs(float(value) * 10**MONEY_DECIMALS)
    whole = math.floor(scaled)
    rounded = whole + 1 if scaled - whole >= 0.5 else whole
    return math.copysign(rounded, value) / 10**MONEY_DECIMALS
