"""The backtest venue's fee models (nautilus ``FeeModel`` subclasses)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Money

from quantlab_trader.base.venue import corporate_action_kind


def is_fee_free(order) -> bool:
    """Return whether ``order`` is a venue fill that no fee model charges.

    Corporate-action venue fills (``CORPORATE_ACTION_*``, ADR 0009) and
    nautilus's own expiration settlements (``EXPIRATION_*_CLOSE``) move no
    commission.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> is_fee_free(SimpleNamespace(tags=["CORPORATE_ACTION_DELIST"]))
    True
    >>> is_fee_free(SimpleNamespace(tags=None))
    False
    """
    tags = order.tags or ()
    return corporate_action_kind(tags) is not None or any(
        tag.startswith("EXPIRATION_") and tag.endswith("_CLOSE") for tag in tags
    )


class FractionFeeModel(FeeModel):
    """Commission as a fraction of the filled notional, quantlab's ``fees``.

    nautilus rounds the commission to the currency's precision (cents).
    Corporate-action and expiration venue fills are free (``is_fee_free``).

    Parameters
    ----------
    rate : float
        The fraction of ``fill_qty * fill_px`` charged.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)

    def get_commission(self, order, fill_qty, fill_px, instrument) -> Money:
        """Return ``rate * fill_qty * fill_px`` in the instrument's quote currency."""
        if is_fee_free(order):
            return Money(0, instrument.quote_currency)
        notional = fill_qty.as_double() * fill_px.as_double()
        return Money(notional * self.rate, instrument.quote_currency)


class IbkrFixedFeeModel(FeeModel):
    """IBKR Pro Fixed commission plus the SEC fee on sales (ADR 0003).

    The commission is USD 0.005 per share, at least USD 1.00 and at most 1%
    of the trade value; a sale adds the SEC fee of 0.0000206 x its value. The
    total is computed in decimals and rounded half up to the cent once.
    Tiered pricing, FINRA TAF and CAT fees are not modelled. Each fill is
    charged as one order (the backtest fills an order whole).
    Corporate-action and expiration venue fills are free (``is_fee_free``).

    Examples
    --------
    >>> IbkrFixedFeeModel.charge(OrderSide.BUY, 100, Decimal("10.5105"))
    Decimal('1.00')
    >>> IbkrFixedFeeModel.charge(OrderSide.SELL, 2500, Decimal("0.4316"))
    Decimal('10.81')
    """

    PER_SHARE = Decimal("0.005")
    MINIMUM = Decimal("1.00")
    MAXIMUM_RATE = Decimal("0.01")
    SEC_FEE_RATE = Decimal("0.0000206")

    @classmethod
    def charge(cls, side: OrderSide, quantity, price) -> Decimal:
        """Return the fee in USD, rounded to the cent, for ``quantity`` shares at ``price``."""
        value = Decimal(quantity) * Decimal(price)
        per_share = max(cls.PER_SHARE * Decimal(quantity), cls.MINIMUM)
        commission = min(per_share, cls.MAXIMUM_RATE * value)
        if side == OrderSide.SELL:
            commission += cls.SEC_FEE_RATE * value
        return commission.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def get_commission(self, order, fill_qty, fill_px, instrument) -> Money:
        """Return the IBKR Pro Fixed fee of the fill in the instrument's quote currency."""
        if is_fee_free(order):
            return Money(0, instrument.quote_currency)
        fee = self.charge(order.side, fill_qty.as_decimal(), fill_px.as_decimal())
        return Money(fee, instrument.quote_currency)
