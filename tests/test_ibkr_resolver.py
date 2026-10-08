"""The IBKR instrument resolver (ADR 0004): permaticker -> ticker as of t -> IBKR contract.

The ticker comes from a Sharadar store's ticker sidecar, the contract from an
injected contract client (a fake here, so no Gateway is needed). Fixtures:

- 194817 traded as FB until 2022-06-08 and as META from 2022-06-09: the same
  IBKR contract (conId 107113386) under both tickers;
- 199059 is BRK.B, a dual-class ticker, IBKR's ``BRK B`` on NYSE;
- 111111 is not in the sidecar (no ticker);
- 222222 is ZZZZ, which IBKR does not know (no contract);
- 333333 is DUAL, for which IBKR returns two US stock contracts (ambiguous).
"""

import json

import pandas as pd
import pytest
from ibapi.tag_value import TagValue
from nautilus_trader.adapters.interactive_brokers.common import IBContract, IBContractDetails
from nautilus_trader.adapters.interactive_brokers.config import (
    InteractiveBrokersInstrumentProviderConfig,
    SymbologyMethod,
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Equity

from quantlab.dataset.sharadar.tickers import SharadarTickerLookup
from quantlab_ibkr.base.venue import InstrumentResolver
from quantlab_ibkr.venue.ibkr.resolver import (
    IbkrResolver,
    ResolutionGap,
    UnresolvedSymbol,
)

T = pd.Timestamp("2022-06-09")
BEFORE_RENAME = pd.Timestamp("2022-06-08")


@pytest.fixture
def lookup(tmp_path):
    sidecar = tmp_path / "sharadar_sep_1d.zarr.sharadar_tickers.json"
    sidecar.write_text(json.dumps({
        "table": "sep",
        "intervals": {
            "194817": [
                {"start": None, "ticker": "FB", "company": "FACEBOOK INC"},
                {"start": "2022-06-09", "ticker": "META", "company": "META PLATFORMS INC"},
            ],
            "199059": [{"start": None, "ticker": "BRK.B", "company": "BERKSHIRE HATHAWAY INC"}],
            "222222": [{"start": None, "ticker": "ZZZZ", "company": "NOBODY INC"}],
            "333333": [{"start": None, "ticker": "DUAL", "company": "TWO LISTINGS INC"}],
        },
    }))
    return SharadarTickerLookup(sidecar)


def _details(con_id, symbol, primary, isin, currency="USD"):
    return IBContractDetails(
        contract=IBContract(
            secType="STK", conId=con_id, exchange="SMART", primaryExchange=primary,
            symbol=symbol, localSymbol=symbol, currency=currency,
        ),
        minTick=0.01,
        secIdList=[TagValue("ISIN", isin)],
    )


class FakeContractClient:
    """IBKR's contract lookup on a fixed table; records every request."""

    def __init__(self):
        self.requests = []
        self.table = {
            "FB": [_details(107113386, "FB", "NASDAQ", "US30303M1027")],
            "META": [_details(107113386, "META", "NASDAQ", "US30303M1027")],
            "BRK B": [_details(72063691, "BRK B", "NYSE", "US0846707026")],
            "DUAL": [
                _details(1, "DUAL", "NYSE", "US0000000011"),
                _details(2, "DUAL", "NASDAQ", "US0000000029"),
            ],
        }

    def details(self, symbol, primary_exchange):
        self.requests.append((symbol, primary_exchange))
        return [
            d for d in self.table.get(symbol, [])
            if primary_exchange is None or d.contract.primaryExchange == primary_exchange
        ]


@pytest.fixture
def client():
    return FakeContractClient()


def test_is_an_instrument_resolver(lookup, client):
    assert isinstance(IbkrResolver([194817], T, lookup, client), InstrumentResolver)


def test_renamed_ticker_resolves_to_the_decision_dates_ticker(lookup, client):
    resolver = IbkrResolver([194817], T, lookup, client)

    assert resolver.instrument_id(194817, T) == InstrumentId.from_str("META.NASDAQ")
    assert resolver.instrument_id(194817, BEFORE_RENAME) == InstrumentId.from_str("FB.NASDAQ")
    assert ("FB", None) in client.requests and ("META", None) in client.requests


def test_reverse_mapping_follows_the_con_id_across_a_rename(lookup, client):
    resolver = IbkrResolver([194817], T, lookup, client)

    assert resolver.permno(InstrumentId.from_str("META.NASDAQ")) == 194817
    assert resolver.permno_for_con_id(107113386) == 194817
    # A position the account still carries under the old ticker's id, once
    # that id has been resolved, maps to the same security.
    resolver.instrument_id(194817, BEFORE_RENAME)
    assert resolver.permno(InstrumentId.from_str("FB.NASDAQ")) == 194817


def test_dual_class_ticker_becomes_ib_simplified(lookup, client):
    resolver = IbkrResolver([199059], T, lookup, client)

    assert str(resolver.instrument_id(199059, T)) == "BRK-B.NYSE"
    assert client.requests == [("BRK B", None)]
    assert resolver.permno(InstrumentId.from_str("BRK-B.NYSE")) == 199059


def test_symbol_label_type_is_kept(lookup, client):
    resolver = IbkrResolver([194817], T, lookup, client)

    assert type(resolver.permno(resolver.instrument_id(194817, T))) is int


def test_unmappable_symbols_are_reported_with_symbol_and_date(lookup, client):
    resolver = IbkrResolver([194817, 111111, 222222, 333333], T, lookup, client)

    assert resolver.gaps == (
        ResolutionGap(111111, T, "no ticker"),
        ResolutionGap(222222, T, "no contract", ticker="ZZZZ"),
        ResolutionGap(333333, T, "ambiguous contract", ticker="DUAL"),
    )
    assert resolver.unresolved(T) == frozenset({111111, 222222, 333333})
    with pytest.raises(UnresolvedSymbol, match="222222.*2022-06-09.*no contract"):
        resolver.instrument_id(222222, T)
    # One gap does not stop the others.
    assert str(resolver.instrument_id(194817, T)) == "META.NASDAQ"


def test_a_gap_is_reported_once(lookup, client):
    resolver = IbkrResolver([222222], T, lookup, client)

    for _ in range(2):
        with pytest.raises(UnresolvedSymbol):
            resolver.instrument_id(222222, T)

    assert len(resolver.gaps) == 1
    assert client.requests == [("ZZZZ", None)]


def test_primary_exchange_narrows_the_lookup(lookup, client):
    resolver = IbkrResolver(
        [333333], T, lookup, client, primary_exchanges={333333: "NASDAQ"}
    )

    assert str(resolver.instrument_id(333333, T)) == "DUAL.NASDAQ"
    assert client.requests == [("DUAL", "NASDAQ")]
    assert resolver.gaps == ()


def test_non_usd_contracts_are_not_candidates(lookup, client):
    client.table["META"] = [
        _details(9, "META", "MEXI", "US30303M1027", currency="MXN"),
        *client.table["META"],
    ]

    resolver = IbkrResolver([194817], T, lookup, client)

    assert str(resolver.instrument_id(194817, T)) == "META.NASDAQ"


def test_unmapped_positions_are_reported_and_left_alone(lookup, client):
    resolver = IbkrResolver([194817], T, lookup, client)
    held = [InstrumentId.from_str("META.NASDAQ"), InstrumentId.from_str("XYZ.NYSE")]

    gaps = resolver.report_positions(held, T)

    assert gaps == [ResolutionGap(None, T, "unmapped position", instrument_id="XYZ.NYSE")]
    assert resolver.gaps[-1] == gaps[0]
    with pytest.raises(UnresolvedSymbol):
        resolver.permno(InstrumentId.from_str("XYZ.NYSE"))


def test_instruments_and_provider_config_come_from_the_resolved_contracts(lookup, client):
    resolver = IbkrResolver([194817, 199059, 222222], T, lookup, client)

    instruments = resolver.instruments()
    assert [str(i.id) for i in instruments] == ["META.NASDAQ", "BRK-B.NYSE"]
    assert all(isinstance(i, Equity) for i in instruments)
    assert instruments[1].isin == "US0846707026"

    config = resolver.provider_config()
    assert isinstance(config, InteractiveBrokersInstrumentProviderConfig)
    assert config.symbology_method == SymbologyMethod.IB_SIMPLIFIED
    assert config.convert_exchange_to_mic_venue is False
    assert config.load_contracts == resolver.load_contracts
    assert {(c.conId, c.exchange, c.primaryExchange, c.symbol, c.currency) for c in config.load_contracts} == {
        (107113386, "SMART", "NASDAQ", "META", "USD"),
        (72063691, "SMART", "NYSE", "BRK B", "USD"),
    }


def test_con_id_cache_seeds_the_reverse_mapping(lookup, client):
    # Yesterday's run resolved 199059; today it is not among the symbols.
    resolver = IbkrResolver([194817], T, lookup, client, con_id_cache={72063691: 199059})

    assert resolver.permno_for_con_id(72063691) == 199059
    assert resolver.con_id_cache == {72063691: 199059, 107113386: 194817}
