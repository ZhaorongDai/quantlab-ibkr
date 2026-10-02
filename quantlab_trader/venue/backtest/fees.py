"""The backtest venue's fee models (nautilus ``FeeModel`` subclasses)."""

from __future__ import annotations

from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.model.objects import Money


class FractionFeeModel(FeeModel):
    """Commission as a fraction of the filled notional, quantlab's ``fees``.

    nautilus rounds the commission to the currency's precision (cents).

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
        notional = fill_qty.as_double() * fill_px.as_double()
        return Money(notional * self.rate, instrument.quote_currency)
