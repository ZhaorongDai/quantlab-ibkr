"""The backtest venue's instrument resolver: ``<PERMNO>.CRSP`` on one synthetic venue (ADR 0004)."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from decimal import Decimal

import pandas as pd
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import Equity, Instrument
from nautilus_trader.model.objects import Price, Quantity

from quantlab_ibkr.base.venue import InstrumentResolver

#: The single simulated venue every backtest instrument lives on; not an exchange.
SYNTHETIC_VENUE = Venue("CRSP")

#: Price decimals of every backtest instrument: CRSP's sub-cent raw prices
#: (bid/ask midpoints) are kept to within half a hundredth of a cent.
PRICE_PRECISION = 4


class BacktestResolver(InstrumentResolver):
    """Identity mapping of PERMNOs onto ``<PERMNO>.CRSP`` equities.

    Each instrument is a USD ``Equity`` with ``price_precision=4``,
    ``price_increment=0.0001``, ``lot_size=1``, zero maker/taker fees (the
    venue's fee model charges) and margin factors of 1 for the venue's MARGIN
    account at leverage 1.

    Parameters
    ----------
    permnos : Iterable
        The securities to create instruments for, as quantlab's symbol labels.
    ts : pandas.Timestamp
        When the instruments come into existence (the window's first bar).

    Examples
    --------
    >>> resolver = BacktestResolver([10107], pd.Timestamp("2024-01-02"))
    >>> str(resolver.instrument_id(10107, pd.Timestamp("2024-01-03")))
    '10107.CRSP'
    >>> resolver.permno(resolver.instrument_id(10107, pd.Timestamp("2024-01-03")))
    10107
    """

    def __init__(self, permnos: Iterable[Hashable], ts: pd.Timestamp):
        ts_ns = pd.Timestamp(ts).value  # a naive date counts as UTC midnight
        self._ids: dict[Hashable, InstrumentId] = {}
        self._permnos: dict[InstrumentId, Hashable] = {}
        self._instruments: list[Instrument] = []
        for permno in permnos:
            symbol = Symbol(str(permno))
            instrument_id = InstrumentId(symbol, SYNTHETIC_VENUE)
            self._ids[permno] = instrument_id
            self._permnos[instrument_id] = permno
            self._instruments.append(
                Equity(
                    instrument_id=instrument_id,
                    raw_symbol=symbol,
                    currency=USD,
                    price_precision=PRICE_PRECISION,
                    price_increment=Price(10**-PRICE_PRECISION, PRICE_PRECISION),
                    lot_size=Quantity(1, 0),
                    ts_event=ts_ns,
                    ts_init=ts_ns,
                    margin_init=Decimal(1),
                    margin_maint=Decimal(1),
                    maker_fee=Decimal(0),
                    taker_fee=Decimal(0),
                )
            )

    def instrument_id(self, permno: Hashable, as_of: pd.Timestamp) -> InstrumentId:
        """Return ``<permno>.CRSP``; the PERMNO never changes, so ``as_of`` plays no part.

        Examples
        --------
        >>> resolver = BacktestResolver([10107], pd.Timestamp("2024-01-02"))
        >>> resolver.instrument_id(10107, pd.Timestamp("2030-01-02"))
        InstrumentId('10107.CRSP')
        """
        return self._ids[permno]

    def permno(self, instrument_id: InstrumentId) -> Hashable:
        """Return the PERMNO of ``instrument_id``, with quantlab's label type.

        Examples
        --------
        >>> resolver = BacktestResolver([10107], pd.Timestamp("2024-01-02"))
        >>> resolver.permno(InstrumentId.from_str("10107.CRSP"))
        10107
        """
        return self._permnos[instrument_id]

    def instruments(self) -> Sequence[Instrument]:
        """Return every instrument, in the order the PERMNOs were given.

        Examples
        --------
        >>> resolver = BacktestResolver([14593, 10107], pd.Timestamp("2024-01-02"))
        >>> [(str(i.id), i.price_precision) for i in resolver.instruments()]
        [('14593.CRSP', 4), ('10107.CRSP', 4)]
        """
        return tuple(self._instruments)
