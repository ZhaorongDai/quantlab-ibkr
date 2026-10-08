"""The IBKR venue's instrument resolver: permaticker -> ticker as of t -> IBKR contract (ADR 0004).

A quantlab symbol (a Sharadar permaticker) is named as of the decision date
by the price dataset's ``TickerLookup`` (the store's ticker sidecar, so FB
before 2022-06-09 and META from then on), then looked up as a US stock
contract through a ``ContractClient`` (SMART routing, USD, optionally a
primary exchange). The contract's nautilus ``InstrumentId`` is the IB
adapter's ``IB_SIMPLIFIED`` one, ``<localSymbol with spaces as '-'>.<primary
exchange>`` (``META.NASDAQ``, ``BRK-B.NYSE``), the id the adapter's
instrument provider gives the same contract when the node loads it with
``convert_exchange_to_mic_venue=False``. Each contract's conId is cached, and
the reverse mapping goes through it, so a held position stays the same
security across a rename.

A symbol without a ticker or a single contract, and an account position no
symbol maps to, is reported as a ``ResolutionGap`` with the symbol and date,
never guessed.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from nautilus_trader.adapters.interactive_brokers.common import IBContract, IBContractDetails
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersInstrumentProviderConfig,
    SymbologyMethod,
)
from nautilus_trader.adapters.interactive_brokers.parsing.instruments import (
    ib_contract_to_instrument_id,
    parse_instrument,
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument

from quantlab.dataset.base import TickerLookup
from quantlab_ibkr.base.venue import InstrumentResolver

#: The currency of every contract the resolver accepts (US equities).
CURRENCY = "USD"

#: Reasons a ``ResolutionGap`` carries.
NO_TICKER = "no ticker"
NO_CONTRACT = "no contract"
AMBIGUOUS_CONTRACT = "ambiguous contract"
UNPARSEABLE_CONTRACT = "unparseable contract"
UNMAPPED_POSITION = "unmapped position"


class ContractClient(Protocol):
    """IBKR's contract lookup, injected so that tests need no Gateway.

    The live venue supplies one backed by the IB client
    (``reqContractDetails`` on a ``STK``/``SMART``/``USD`` contract, each
    result converted with ``IBContractDetails.from_contract_details``).

    Examples
    --------
    A client on a fixed table:

    >>> class Table:
    ...     def __init__(self, rows):
    ...         self.rows = rows
    ...     def details(self, symbol, primary_exchange):
    ...         return list(self.rows.get(symbol, []))
    >>> Table({}).details("AAPL", None)
    []
    """

    def details(self, symbol: str, primary_exchange: str | None) -> list[IBContractDetails]:
        """Return the contract details of US stock ``symbol`` in IBKR's spelling (``BRK B``).

        Parameters
        ----------
        symbol : str
            IBKR's symbol: the ticker with a share class after a space.
        primary_exchange : str or None
            IBKR's primary exchange (``NASDAQ``, ``NYSE``, ...) to narrow the
            lookup to; ``None`` for any.

        Examples
        --------
        ::

            client.details("BRK B", None)  # [IBContractDetails(...)]
        """
        ...


@dataclass(frozen=True)
class ResolutionGap:
    """A symbol or position the resolver could not map, with its date.

    Attributes
    ----------
    symbol : Hashable or None
        The quantlab symbol (permaticker); ``None`` for an account position
        no symbol maps to.
    date : pandas.Timestamp
        The decision date the mapping was asked for.
    reason : str
        ``"no ticker"``, ``"no contract"``, ``"ambiguous contract"``,
        ``"unparseable contract"`` or ``"unmapped position"``.
    ticker : str or None
        The ticker the symbol had that day, when known.
    instrument_id : str or None
        The position's instrument, for an unmapped position.

    Examples
    --------
    >>> ResolutionGap(222222, pd.Timestamp("2022-06-09"), "no contract", ticker="ZZZZ")
    ResolutionGap(symbol=222222, date=Timestamp('2022-06-09 00:00:00'), reason='no contract', ticker='ZZZZ', instrument_id=None)
    """

    symbol: Hashable | None
    date: pd.Timestamp
    reason: str
    ticker: str | None = None
    instrument_id: str | None = None


class UnresolvedSymbol(KeyError):
    """Raised for a symbol or instrument the resolver has no mapping for.

    Examples
    --------
    >>> issubclass(UnresolvedSymbol, KeyError)
    True
    """


def ib_symbol(ticker: str) -> str:
    """Return IBKR's symbol for a vendor ticker: a share class follows a space.

    Parameters
    ----------
    ticker : str
        The ticker as Sharadar or CRSP writes it (``BRK.B``, ``BF-B``).

    Examples
    --------
    >>> ib_symbol("BRK.B"), ib_symbol("BF-B"), ib_symbol("AAPL")
    ('BRK B', 'BF B', 'AAPL')
    """
    return ticker.strip().upper().replace(".", " ").replace("-", " ")


def _venue(contract: IBContract) -> str:
    """Return the venue the IB instrument provider gives a SMART-routed stock contract."""
    if contract.primaryExchange and contract.primaryExchange != "SMART":
        return contract.primaryExchange
    return contract.exchange


@dataclass(frozen=True)
class _Resolved:
    """One symbol's contract as of one day."""

    ticker: str
    details: IBContractDetails
    instrument_id: InstrumentId


class IbkrResolver(InstrumentResolver):
    """Maps quantlab symbols to IBKR stock contracts as of the decision date, and back.

    The symbols are resolved as of ``as_of`` when the resolver is built, so
    ``instruments()`` and ``provider_config()`` are ready before the node
    starts; ``instrument_id`` for another date resolves on demand. Each
    (symbol, date) is looked up once, a gap included.

    Parameters
    ----------
    symbols : Iterable of Hashable
        quantlab symbols (permatickers) to resolve: the predicted and the
        held ones.
    as_of : pandas.Timestamp
        The decision date.
    ticker_lookup : quantlab.dataset.base.TickerLookup
        Names the symbols as of a day; the price dataset's
        ``ticker_lookup()``. A symbol it names by its own id has no ticker.
    client : ContractClient
        IBKR's contract lookup.
    primary_exchanges : Mapping, optional
        IBKR primary exchange per symbol, to narrow a lookup.
    con_id_cache : Mapping of int to Hashable, optional
        conId -> symbol from earlier runs (``con_id_cache`` of a previous
        resolver), so a holding resolved on another day maps back.

    Attributes
    ----------
    as_of : pandas.Timestamp
        The decision date.

    Examples
    --------
    >>> resolver = _doctest_resolver()
    >>> str(resolver.instrument_id(199059, pd.Timestamp("2022-06-09")))
    'BRK-B.NYSE'
    >>> resolver.permno(InstrumentId.from_str("BRK-B.NYSE"))
    199059
    >>> resolver.gaps
    (ResolutionGap(symbol=111111, date=Timestamp('2022-06-09 00:00:00'), reason='no ticker', ticker=None, instrument_id=None),)
    """

    def __init__(
        self,
        symbols: Iterable[Hashable],
        as_of: pd.Timestamp,
        ticker_lookup: TickerLookup,
        client: ContractClient,
        *,
        primary_exchanges: Mapping[Hashable, str] | None = None,
        con_id_cache: Mapping[int, Hashable] | None = None,
    ):
        self.as_of = pd.Timestamp(as_of)
        self._symbols = tuple(dict.fromkeys(symbols))
        self._lookup = ticker_lookup
        self._client = client
        self._primary = dict(primary_exchanges or {})
        self._by_day: dict[tuple[Hashable, pd.Timestamp], _Resolved | ResolutionGap] = {}
        self._permnos: dict[InstrumentId, Hashable] = {}
        self._con_ids: dict[int, Hashable] = dict(con_id_cache or {})
        self._gaps: list[ResolutionGap] = []
        for symbol in self._symbols:
            self._resolve(symbol, self.as_of)

    def _gap(self, gap: ResolutionGap) -> ResolutionGap:
        self._gaps.append(gap)
        return gap

    def _ticker(self, symbol: Hashable, day: pd.Timestamp) -> str | None:
        """Return the symbol's ticker on ``day``, or ``None`` when the lookup has none."""
        name = self._lookup.names([symbol], day.date())[0]
        try:
            own = str(int(symbol))
        except (TypeError, ValueError):
            own = str(symbol)
        return None if name.ticker in (own, str(symbol)) else name.ticker

    def _resolve(self, symbol: Hashable, day: pd.Timestamp) -> _Resolved | ResolutionGap:
        key = (symbol, day)
        if key in self._by_day:
            return self._by_day[key]
        ticker = self._ticker(symbol, day)
        if ticker is None:
            result = self._gap(ResolutionGap(symbol, day, NO_TICKER))
        else:
            found = [
                d for d in self._client.details(ib_symbol(ticker), self._primary.get(symbol))
                if d.contract is not None
                and d.contract.secType == "STK"
                and d.contract.currency == CURRENCY
            ]
            by_con_id = {d.contract.conId: d for d in found}
            if not by_con_id:
                result = self._gap(ResolutionGap(symbol, day, NO_CONTRACT, ticker=ticker))
            elif len(by_con_id) > 1:
                result = self._gap(ResolutionGap(symbol, day, AMBIGUOUS_CONTRACT, ticker=ticker))
            else:
                (details,) = by_con_id.values()
                instrument_id = ib_contract_to_instrument_id(
                    details.contract, _venue(details.contract), SymbologyMethod.IB_SIMPLIFIED
                )
                result = _Resolved(ticker, details, instrument_id)
                self._permnos[instrument_id] = symbol
                self._con_ids[details.contract.conId] = symbol
        self._by_day[key] = result
        return result

    def instrument_id(self, permno: Hashable, as_of: pd.Timestamp) -> InstrumentId:
        """Return the ``IB_SIMPLIFIED`` id of ``permno``'s contract on ``as_of``.

        Raises
        ------
        UnresolvedSymbol
            The symbol has no ticker or no single contract that day (the gap
            is in ``gaps``).

        Examples
        --------
        >>> resolver = _doctest_resolver()
        >>> resolver.instrument_id(194817, pd.Timestamp("2022-06-08"))
        InstrumentId('FB.NASDAQ')
        >>> resolver.instrument_id(194817, pd.Timestamp("2022-06-09"))
        InstrumentId('META.NASDAQ')
        """
        day = pd.Timestamp(as_of)
        result = self._resolve(permno, day)
        if isinstance(result, ResolutionGap):
            raise UnresolvedSymbol(f"{permno} on {day.date()}: {result.reason}")
        return result.instrument_id

    def permno(self, instrument_id: InstrumentId) -> Hashable:
        """Return the symbol of an instrument the resolver resolved, through its conId.

        Raises
        ------
        UnresolvedSymbol
            No resolved contract has this id.

        Examples
        --------
        >>> resolver = _doctest_resolver()
        >>> resolver.permno(InstrumentId.from_str("META.NASDAQ"))
        194817
        """
        try:
            return self._permnos[instrument_id]
        except KeyError:
            raise UnresolvedSymbol(f"no symbol maps to {instrument_id}") from None

    def permno_for_con_id(self, con_id: int) -> Hashable:
        """Return the symbol of IBKR contract ``con_id``, from this or an earlier run.

        Raises
        ------
        UnresolvedSymbol
            The conId is not cached.

        Examples
        --------
        >>> _doctest_resolver().permno_for_con_id(107113386)
        194817
        """
        try:
            return self._con_ids[con_id]
        except KeyError:
            raise UnresolvedSymbol(f"no symbol maps to conId {con_id}") from None

    @property
    def con_id_cache(self) -> dict[int, Hashable]:
        """conId -> symbol of every contract seeded or resolved, to seed the next run.

        Examples
        --------
        >>> _doctest_resolver().con_id_cache
        {107113386: 194817, 72063691: 199059}
        """
        return dict(self._con_ids)

    @property
    def gaps(self) -> tuple[ResolutionGap, ...]:
        """Every gap reported so far, in order: symbols, then positions.

        Examples
        --------
        >>> [gap.reason for gap in _doctest_resolver().gaps]
        ['no ticker']
        """
        return tuple(self._gaps)

    def unresolved(self, as_of: pd.Timestamp | None = None) -> frozenset:
        """Return the symbols without a contract on ``as_of`` (default the decision date).

        Examples
        --------
        >>> _doctest_resolver().unresolved()
        frozenset({111111})
        """
        day = self.as_of if as_of is None else pd.Timestamp(as_of)
        return frozenset(
            symbol
            for (symbol, at), result in self._by_day.items()
            if at == day and isinstance(result, ResolutionGap)
        )

    def report_positions(
        self, instrument_ids: Iterable[InstrumentId], as_of: pd.Timestamp
    ) -> list[ResolutionGap]:
        """Report each account position no symbol maps to, and return those gaps.

        The positions are left alone: the caller keeps them out of the
        strategy's holdings.

        Examples
        --------
        >>> resolver = _doctest_resolver()
        >>> held = [InstrumentId.from_str("META.NASDAQ"), InstrumentId.from_str("XYZ.NYSE")]
        >>> [g.instrument_id for g in resolver.report_positions(held, pd.Timestamp("2022-06-09"))]
        ['XYZ.NYSE']
        """
        day = pd.Timestamp(as_of)
        return [
            self._gap(ResolutionGap(None, day, UNMAPPED_POSITION, instrument_id=str(i)))
            for i in instrument_ids
            if i not in self._permnos
        ]

    def _resolved(self) -> list[_Resolved]:
        """The decision date's contracts in symbol order, one per conId."""
        seen, out = set(), []
        for symbol in self._symbols:
            result = self._by_day[(symbol, self.as_of)]
            if isinstance(result, _Resolved) and result.details.contract.conId not in seen:
                seen.add(result.details.contract.conId)
                out.append(result)
        return out

    def instruments(self) -> Sequence[Instrument]:
        """Return the decision date's instruments, parsed as the IB adapter parses them.

        A contract the adapter cannot parse (no ISIN, say) is reported as an
        ``"unparseable contract"`` gap and left out.

        Examples
        --------
        >>> [str(i.id) for i in _doctest_resolver().instruments()]
        ['META.NASDAQ', 'BRK-B.NYSE']
        """
        instruments = []
        for result in self._resolved():
            try:
                instruments.append(parse_instrument(
                    result.details, result.instrument_id.venue.value, SymbologyMethod.IB_SIMPLIFIED
                ))
            except ValueError:
                self._gap(ResolutionGap(
                    self._permnos[result.instrument_id], self.as_of, UNPARSEABLE_CONTRACT,
                    ticker=result.ticker, instrument_id=str(result.instrument_id),
                ))
        return tuple(instruments)

    @property
    def load_contracts(self) -> frozenset[IBContract]:
        """The decision date's contracts as the instrument provider loads them, by conId.

        Examples
        --------
        >>> sorted(c.conId for c in _doctest_resolver().load_contracts)
        [72063691, 107113386]
        """
        return frozenset(
            IBContract(
                secType="STK",
                conId=r.details.contract.conId,
                exchange="SMART",
                primaryExchange=r.details.contract.primaryExchange,
                symbol=r.details.contract.symbol,
                currency=CURRENCY,
            )
            for r in self._resolved()
        )

    def provider_config(self, **kwargs) -> InteractiveBrokersInstrumentProviderConfig:
        """Return the instrument provider config the node loads the resolved contracts with.

        ``IB_SIMPLIFIED`` symbology without MIC conversion, so the provider
        gives each contract the id ``instrument_id`` returns.

        Parameters
        ----------
        **kwargs
            Further ``InteractiveBrokersInstrumentProviderConfig`` fields
            (``cache_validity_days``, ...).

        Examples
        --------
        >>> config = _doctest_resolver().provider_config()
        >>> config.symbology_method.name, len(config.load_contracts)
        ('IB_SIMPLIFIED', 2)
        """
        return InteractiveBrokersInstrumentProviderConfig(
            load_contracts=self.load_contracts,
            symbology_method=SymbologyMethod.IB_SIMPLIFIED,
            convert_exchange_to_mic_venue=False,
            **kwargs,
        )


def _doctest_resolver() -> IbkrResolver:
    """Return a resolver on fixed tables for the docstrings' Examples.

    194817 is FB until 2022-06-08 and META from 2022-06-09, 199059 is BRK.B
    and 111111 has no ticker; resolved as of 2022-06-09.
    """
    from datetime import date

    from ibapi.tag_value import TagValue
    from quantlab.dataset.base import SymbolName

    class Names(TickerLookup):
        def names(self, symbols, day):
            def name(s):
                if s == 194817:
                    return SymbolName("FB" if day < date(2022, 6, 9) else "META")
                return SymbolName("BRK.B") if s == 199059 else SymbolName(str(s))
            return [name(s) for s in symbols]

    def details(con_id, symbol, primary, isin):
        return IBContractDetails(
            contract=IBContract(secType="STK", conId=con_id, exchange="SMART",
                                primaryExchange=primary, symbol=symbol,
                                localSymbol=symbol, currency=CURRENCY),
            minTick=0.01, secIdList=[TagValue("ISIN", isin)],
        )

    table = {
        "FB": [details(107113386, "FB", "NASDAQ", "US30303M1027")],
        "META": [details(107113386, "META", "NASDAQ", "US30303M1027")],
        "BRK B": [details(72063691, "BRK B", "NYSE", "US0846707026")],
    }

    class Client:
        def details(self, symbol, primary_exchange):
            return table.get(symbol, [])

    return IbkrResolver([194817, 199059, 111111], pd.Timestamp("2022-06-09"), Names(), Client())
