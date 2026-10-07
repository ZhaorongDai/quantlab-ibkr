"""The backtest venue's fee models (nautilus ``FeeModel`` subclasses)."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.objects import Money

from quantlab_ibkr.base.venue import corporate_action_kind


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

    Attributes
    ----------
    minimum_fee_orders : set of str
        Always empty: a fraction of the notional has no minimum.

    Examples
    --------
    >>> FractionFeeModel(0.001).rate, FractionFeeModel(0.001).minimum_fee_orders
    (0.001, set())
    """

    def __init__(self, rate: float):
        super().__init__()
        self.rate = float(rate)
        self.minimum_fee_orders: set[str] = set()

    @property
    def description(self) -> str:
        """The model as a run report's Setup states it.

        Examples
        --------
        >>> FractionFeeModel(0.001).description
        '0.001 of the traded notional'
        """
        return f"{self.rate:g} of the traded notional"

    def get_commission(self, order, fill_qty, fill_px, instrument) -> Money:
        """Return ``rate * fill_qty * fill_px`` in the instrument's quote currency.

        Parameters
        ----------
        order : nautilus_trader.model.orders.Order
            The order filled; its tags say whether the fill is free.
        fill_qty : nautilus_trader.model.objects.Quantity
            Shares filled.
        fill_px : nautilus_trader.model.objects.Price
            The fill price.
        instrument : nautilus_trader.model.instruments.Instrument
            The instrument traded.

        Examples
        --------
        nautilus calls it on each fill; a stand-in order without tags:

        >>> from types import SimpleNamespace
        >>> import pandas as pd
        >>> from nautilus_trader.model.objects import Price, Quantity
        >>> from quantlab_ibkr.venue.backtest.resolver import BacktestResolver
        >>> instrument = BacktestResolver([10001], pd.Timestamp("2024-01-02")).instruments()[0]
        >>> order = SimpleNamespace(tags=None)
        >>> FractionFeeModel(0.001).get_commission(order, Quantity(100, 0), Price(10.5, 4), instrument)
        Money(1.05, USD)
        """
        if is_fee_free(order):
            return Money(0, instrument.quote_currency)
        # The share count from its decimal: Quantity.as_double() is inexact for many.
        notional = float(fill_qty.as_decimal()) * fill_px.as_double()
        return Money(notional * self.rate, instrument.quote_currency)


class IbkrFixedFeeModel(FeeModel):
    """IBKR Pro Fixed commission plus the SEC fee on sales (#13, spec #18 story 22).

    The commission is USD 0.005 per share, at least USD 1.00 and at most 1%
    of the trade value; a sale adds the SEC fee of 0.0000206 x its value. The
    total is computed in decimals and rounded half up to the cent once.
    Tiered pricing, FINRA TAF and CAT fees are not modelled. Each fill is
    charged as one order (the backtest fills an order whole).
    Corporate-action and expiration venue fills are free (``is_fee_free``).

    Attributes
    ----------
    minimum_fee_orders : set of str
        Client order ids of the fills charged the USD 1.00 minimum
        (``minimum_applies``), recorded as they are charged.

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

    def __init__(self):
        super().__init__()
        self.minimum_fee_orders: set[str] = set()

    @property
    def description(self) -> str:
        """The model as a run report's Setup states it.

        Examples
        --------
        >>> IbkrFixedFeeModel().description
        'IBKR Pro Fixed: USD 0.005 per share, min USD 1.00, max 1% of value; SEC fee 0.0000206 of a sale'
        """
        return (
            f"IBKR Pro Fixed: USD {self.PER_SHARE} per share, min USD {self.MINIMUM}, "
            f"max {float(self.MAXIMUM_RATE * 100):g}% of value; SEC fee {self.SEC_FEE_RATE} of a sale"
        )

    @classmethod
    def charge(cls, side: OrderSide, quantity: Decimal | int, price: Decimal) -> Decimal:
        """Return the fee in USD, rounded to the cent, for ``quantity`` shares at ``price``.

        Examples
        --------
        3,000 shares at USD 20: 0.005 per share, plus the SEC fee on a sale:

        >>> IbkrFixedFeeModel.charge(OrderSide.BUY, 3000, Decimal("20"))
        Decimal('15.00')
        >>> IbkrFixedFeeModel.charge(OrderSide.SELL, 3000, Decimal("20"))
        Decimal('16.24')
        """
        value = Decimal(quantity) * Decimal(price)
        per_share = max(cls.PER_SHARE * Decimal(quantity), cls.MINIMUM)
        commission = min(per_share, cls.MAXIMUM_RATE * value)
        if side == OrderSide.SELL:
            commission += cls.SEC_FEE_RATE * value
        return commission.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    @classmethod
    def minimum_applies(cls, quantity: Decimal | int, price: Decimal | float) -> bool:
        """Return whether the USD 1.00 minimum sets the commission of ``quantity`` at ``price``.

        The per-share charge is below the minimum and the 1% cap does not
        cut the minimum.

        Examples
        --------
        >>> IbkrFixedFeeModel.minimum_applies(100, 10.0), IbkrFixedFeeModel.minimum_applies(300, 10.0)
        (True, False)
        >>> IbkrFixedFeeModel.minimum_applies(10, 5.0)  # the 1% cap, USD 0.50, applies
        False
        """
        value = Decimal(quantity) * Decimal(str(price))
        return cls.PER_SHARE * Decimal(quantity) < cls.MINIMUM <= cls.MAXIMUM_RATE * value

    def get_commission(self, order, fill_qty, fill_px, instrument) -> Money:
        """Return the IBKR Pro Fixed fee of the fill in the instrument's quote currency.

        A fill charged the minimum adds its order to ``minimum_fee_orders``.

        Parameters
        ----------
        order : nautilus_trader.model.orders.Order
            The order filled; its tags say whether the fill is free.
        fill_qty : nautilus_trader.model.objects.Quantity
            Shares filled.
        fill_px : nautilus_trader.model.objects.Price
            The fill price.
        instrument : nautilus_trader.model.instruments.Instrument
            The instrument traded.

        Examples
        --------
        nautilus calls it on each fill; a stand-in order of 100 shares, which
        the USD 1.00 minimum prices:

        >>> from types import SimpleNamespace
        >>> import pandas as pd
        >>> from nautilus_trader.model.objects import Price, Quantity
        >>> from quantlab_ibkr.venue.backtest.resolver import BacktestResolver
        >>> instrument = BacktestResolver([10001], pd.Timestamp("2024-01-02")).instruments()[0]
        >>> order = SimpleNamespace(tags=None, side=OrderSide.BUY, client_order_id=SimpleNamespace(value="O-1"))
        >>> model = IbkrFixedFeeModel()
        >>> model.get_commission(order, Quantity(100, 0), Price(10.5, 4), instrument)
        Money(1.00, USD)
        >>> model.minimum_fee_orders
        {'O-1'}
        """
        if is_fee_free(order):
            return Money(0, instrument.quote_currency)
        quantity, price = fill_qty.as_decimal(), fill_px.as_decimal()
        if self.minimum_applies(quantity, price):
            self.minimum_fee_orders.add(order.client_order_id.value)
        fee = self.charge(order.side, quantity, price)
        return Money(fee, instrument.quote_currency)
