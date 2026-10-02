"""The backtest venue's fill model: fractional slippage (a nautilus ``FillModel`` subclass)."""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

from nautilus_trader.backtest.models import FillModel
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import BookOrder
from nautilus_trader.model.enums import BookType, OrderSide
from nautilus_trader.model.objects import Price, Quantity

#: Liquidity of each simulated level: enough for any order (no volume cap, ADR 0003).
_UNLIMITED = 10**12


def slipped_price(price: Decimal, side: OrderSide, slippage: float, precision: int) -> Decimal:
    """Return ``price`` moved against an order of ``side`` by the fraction ``slippage``.

    A buy pays ``price * (1 + slippage)``, a sale gets ``price * (1 -
    slippage)``; the result is rounded to ``precision`` decimals against the
    order (up for a buy, down for a sale).

    Examples
    --------
    >>> slipped_price(Decimal("0.4123"), OrderSide.BUY, 0.001, 4)
    Decimal('0.4128')
    >>> slipped_price(Decimal("0.4321"), OrderSide.SELL, 0.001, 4)
    Decimal('0.4316')
    """
    fraction = Decimal(repr(float(slippage)))
    factor = 1 + fraction if side == OrderSide.BUY else 1 - fraction
    rounding = ROUND_CEILING if side == OrderSide.BUY else ROUND_FLOOR
    return (price * factor).quantize(Decimal(1).scaleb(-precision), rounding=rounding)


class FractionSlippageFillModel(FillModel):
    """Fill every order at the market price moved against it by a fraction.

    nautilus's built-in slippage is a tick, not a fraction; this model hands
    the matching engine a one-level book at the slipped prices instead
    (verified on 1.231 in the nautilus research note). quantlab's
    ``slippage`` is the same fraction of the fill price. A zero slippage
    leaves the engine's own fill logic in place. Venue fills of corporate
    actions and delistings bypass every fill model (``apply_fills``).

    Parameters
    ----------
    slippage : float
        The fraction, in ``[0, 1)``.

    Raises
    ------
    ValueError
        If ``slippage`` is outside ``[0, 1)``.
    """

    def __init__(self, slippage: float):
        super().__init__()
        slippage = float(slippage)
        if not 0.0 <= slippage < 1.0:
            raise ValueError(f"slippage must lie in [0, 1), got {slippage}")
        self.slippage = slippage

    def get_orderbook_for_fill_simulation(self, instrument, order, best_bid, best_ask):
        """Return a one-level book at the slipped bid and ask, or ``None`` without slippage."""
        if self.slippage == 0.0:
            return None
        precision = instrument.price_precision
        book = OrderBook(instrument_id=instrument.id, book_type=BookType.L2_MBP)
        for order_id, (side, price, taker) in enumerate(
            ((OrderSide.BUY, best_bid, OrderSide.SELL), (OrderSide.SELL, best_ask, OrderSide.BUY)),
            start=1,
        ):
            slipped = slipped_price(price.as_decimal(), taker, self.slippage, precision)
            book.add(
                BookOrder(
                    side=side,
                    price=Price(slipped, precision),
                    size=Quantity(_UNLIMITED, instrument.size_precision),
                    order_id=order_id,
                ),
                0,
                0,
            )
        return book
