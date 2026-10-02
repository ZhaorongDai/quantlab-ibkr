---
status: accepted
date: 2026-10-01
---

# The backtest venue books corporate actions as venue fills; cash is derived from the margin balance

NautilusTrader 1.231 has no API to change a position's quantity, and a MARGIN account's balance is
not cash: it moves by realized PnL, commissions and adjustments, never by a fill's notional
(verified). So the backtest venue applies corporate actions with one `SimulationModule`
(`venue/backtest/corporate_actions.py`) that holds a schedule built from the run's price dataset
(ADR 0006: `splitFactor != 1`, `divCash != 0`, `delisting_bars`) and acts at 09:30 ET of the
ex-date, before the open + 1 ns orders of ADR 0003:

- **split**: one venue-generated fill on the holder's position of the share-count change,
  `floor(q * k) - q`, at **price 0**, so no cash moves; the fraction is credited as cash in lieu
  at the pre-split close / k through `exchange.adjust_account`;
- **dividend**: `exchange.adjust_account(divCash * signed_qty)` on the position held at the prior
  close (a short pays);
- **delisting**: a venue-generated fill closing the position at its last valuation on bar b+1
  (quantlab ADR 0014), from the same module.

Each venue fill is a `MarketOrder` carrying the position's trader and strategy ids and a
`CORPORATE_ACTION_*` tag, added to the cache, marked submitted and accepted through the venue's
execution client and filled with `OrderMatchingEngine.apply_fills`, the path nautilus's own
expiration settlement uses. Every trader fee model returns zero for those tags (and for
`EXPIRATION_*_CLOSE`). `account.py` derives
`cash = balance - sum(signed_qty * avg_px_open)` and
`equity = cash + sum(signed_qty * last trade price)`; it never uses `Portfolio.unrealized_pnls`,
which was stale in the prototype. A next-open order queued before an ex-date is in pre-split
shares, so the backtest open submitter multiplies it by that day's splitFactor (floor) before
submitting: the economic quantity is unchanged, only its unit. This refines ADR 0003's "never
changes a quantity".

Evidence: branch `prototype/corporate-actions`, `prototypes/corporate_actions/` (#15). Cash, share
counts and equity match an independent ledger after every close for splitFactor 2, 3, 1.5 and
1/3, longs and shorts, dividends and a delisting.

## Considered options

- Close the position at the pre-split close and reopen `q * k` at close / k. Rejected: it books a
  fictitious realized PnL on every split, resets the cost basis, and rounds the reopen price to
  the tick (0.01 of cash drift at 3:1, verified).
- `InstrumentClose(CONTRACT_EXPIRED)` + `settlement_prices` for delisting, as #3 and #13 had it.
  Works (verified: last valuation, zero fee with the tag rule), and stays the fallback. Not chosen
  so that all three actions come from one schedule, one module and one tag family, with no static
  per-venue settlement table and no extra data objects in the feed.
- Reading equity as `balance + Portfolio.unrealized_pnls`. Rejected: stale between position
  events in the prototype (100327.00 against a true 100527.00).

## Consequences

- The Strategy receives `OrderFilled` events for orders it never submitted. `PortfolioStrategy`
  must treat `CORPORATE_ACTION_*` fills as venue events (recorded in `events.json`), not as its
  next-open orders.
- On a reverse split the price-0 sell books a fictitious realized loss and leaves `avg_px_open`
  at the old price; nautilus's position statistics are therefore not reported by trader. Cash and
  equity stay exact.
- Live, IBKR books splits and dividends itself; how they reach nautilus (fills, position
  reports, account updates) and what IBKR does to an on-open order across a split are for the
  live effort.
