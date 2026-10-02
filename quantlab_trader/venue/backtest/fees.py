"""The backtest venue's fee models (nautilus ``FeeModel`` subclasses)."""

from __future__ import annotations

from nautilus_trader.backtest.models import FeeModel
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
