---
status: accepted
date: 2026-10-01
---

# The strategy keys securities by PERMNO; each venue resolves them to its instruments

trader's strategy identifies a security by its CRSP **PERMNO**, the key of quantlab's prediction
panel, of every `PortfolioContext` and of the weights `construct()` returns. A nautilus
`InstrumentId` exists only where an order is submitted or a position is read, and the strategy
gets it from an **instrument resolver** owned by the venue, never by building one itself:

- `instrument_id(permno, as_of)`: the venue's instrument for that security on that decision date;
- `permno(instrument_id)`: the reverse, used to turn the account's positions into PERMNO holdings.

In the backtest the resolver is the identity mapping onto one synthetic venue: PERMNO 10107 is
`10107.CRSP`, an `Equity` with `raw_symbol="10107"`, `USD`, `price_precision=4`,
`price_increment=0.0001`, `lot_size=1` and zero maker/taker fees (fees come from the execution
fee model). One is created per PERMNO that has a finite prediction anywhere in the window, since
no constructor can enter a security it has no prediction for. A ticker change is invisible (the
PERMNO does not change); a delisting is the instrument's `InstrumentClose` plus its settlement
price. Tickers appear only in reports, through quantlab's `CrspTickerLookup`.

The live IBKR resolver (built later) maps a PERMNO to `TICKER.PRIMARY_EXCHANGE` as of the
decision date (the adapter's default symbology, share class `BRK.B` becoming `BRK-B`), and maps
back by the contract's conId cached when the instrument is loaded, so a held position stays the
same security across a rename. Its mapping rows come from quantlab, not from a vendor reader in
trader: CRSP's `stksecurityinfohist` (ticker and share class through `CrspSymbology`,
`primaryexch`, `cusip9`) seeds them, with the ISIN derived from the CUSIP checked against IB's
contract details.

## Why

The rule is that the backtest behaves as live trading will. With the strategy keyed by PERMNO
both venues run the same decision code, and the decision data (predictions, constructor state,
locked positions) never has to follow a ticker change. Ticker-at-date ids in the backtest would
look like live ids but would make a rename end one instrument and start another, so a held
position would have to be migrated by hand, and two securities that used one ticker years apart
would share an id.

## Considered options

- Ticker-at-date instrument ids in the backtest (`MSFT.XNAS`), to match live ids. Rejected:
  renames split a holding across two instruments, tickers are reused, and CRSP's `primaryexch`
  changes during a security's life, so the id would not be stable even without a rename.
- One instrument per PERMNO in the run's whole price panel. Rejected: the market universe holds
  thousands of PERMNOs that are never predicted, and cost and memory grow with the data fed.
- Real exchange venues in the backtest (`XNYS`, `XNAS`). Rejected: each simulated venue has its
  own account, while the portfolio is one account.

## Consequences

- Bar and tick prices must be built at the instrument's precision: nautilus rejects data whose
  price precision differs (`backtest/engine.pyx` checks it for trade ticks and bars), and `Price`
  silently rounds to its precision. Four decimals keep CRSP's sub-cent raw prices (bid/ask
  midpoints) to within half a hundredth of a cent; live instruments use IB's `minTick` instead.
- A live holding the resolver cannot map back to a PERMNO (a manual trade, a corporate action
  creating a new security) is an operations question for the live effort.
