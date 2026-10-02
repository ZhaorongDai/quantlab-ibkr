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
  (quantlab ADR 0014), from the same module. CIZ books a cash merger's payment in
  `dlynonorddivamt` on the delisting row (measured on the S&P 500 store, 2026-10-02: all 181
  delisting rows without a price carry `divCash` equal to the last close grown by the delisting
  return), so `divCash` on a row without a raw close, on the delisting or settlement bar of a
  settled delisting, is the proceeds the settlement pays and moves no cash (logged as
  `DELISTING_PAYMENT`).

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

## Amendment (2026-10-01, split-factor research)

CRSP's price factor alone does not give a holder's share count
([research](https://github.com/ZhaorongDai/quantlab-trader/issues/16)). The venue therefore uses
the **holder split factor** `k`, booked only on days where `splitFactor ≈ shareFactor`
(`shareFactor = cumfacshr[t-1] / cumfacshr[t]`, from the dataset's `cumfacshr` variable): `k`
drives the split venue fill, cash in lieu (pre-split close / k) and the open submitter's rescale of
a queued order. A day with `splitFactor > 1` and `shareFactor = 1` is a **value distribution**
(spin-off and the like): the share count is unchanged and the holder is credited
`q · (splitFactor - 1) · close[t]` in cash at the next open, in addition to that day's `divCash`
(measured: CIZ never carries a spin-off's value in `divCash`). A split is booked only for a finite
`k > 0`: a final distribution (`disfacpr = -1`, a merger or liquidation) gives `k = 0` and goes
through the delisting path, never a split to zero shares. Any other factor day leaves the position
alone and is logged. No position is opened in a spin-off's new PERMNO. Measured on 2000-2025 CRSP
([#17](https://github.com/ZhaorongDai/quantlab-trader/issues/17)): 6,721 holder-split days, 657
value-distribution days, 5,911 final events; the store's `splitFactor` equals the `cumfacpr` ratio
on every comparable day.

## Amendment (2026-10-02, price-implied share changes, #27)

The factors alone miss share changes CRSP's return knows about
([#27](https://github.com/ZhaorongDai/quantlab-trader/issues/27)): on the market store a reverse
split whose share factor disagrees with its price factor (PERMNO 18217, 2024-03-15) was left as
`OTHER`, and a reverse split after a six-week halt with no factor at all (PERMNO 14051, 1:67)
was not booked, so raw-price holdings jumped by multiples no holder had.

quantlab chains `adjClose` from CRSP's total return `ret` (a missing `ret` counts as 0), so on a
bar t with a raw close, with p the last bar before it with a raw close, the share change that
conserves a holder's value given `adjClose`'s return is (p and t both with a raw close and an
`adjClose`)

    x = (adjClose[t] / adjClose[p] * close[p] - C - Q * divCash[t]) / (Q * close[t])

where `Q` and `C` are the shares and cash per share held at p that actions booked on rows
between p and t (rows without a raw close or an `adjClose`, inside a halt; a distribution there
is paid at the raw close when the row has one, else at the pre-split close / k) made of it; `Q = 1, C = 0` on an
ordinary day. `divCash[t]` is paid on the pre-split shares, as the venue pays it, so a dividend
is not counted twice, and a value distribution keeps its cash booking (its day is never checked).
The rule, on a day the factors leave alone (no factor day, or `OTHER`):

- an `OTHER` day whose `splitFactor` (moving, finite, > 0) agrees with x within
  `PRICE_FACTOR_RTOL = 1%` is a holder **split by `splitFactor`**, the price factor CRSP's `ret`
  uses, whatever the share factor says (18217: k 0.035404, x 0.035395);
- otherwise, if `x > 1 + IMPLIED_SPLIT_TOL` or `x < 1 / (1 + IMPLIED_SPLIT_TOL)` with
  `IMPLIED_SPLIT_TOL = 0.2`, the day is an **implied split** by x: event kind `IMPLIED_SPLIT`
  (venue fill tag `CORPORATE_ACTION_IMPLIED_SPLIT`), booked exactly as a split (whole shares
  toward zero, cash in lieu at the pre-split close / x, a queued next-open order rescaled by x,
  floor); `metrics.json` counts them under `execution.trader.implied_splits`;
- otherwise a day without a factor whose x differs from 1 by more than `PRICE_FACTOR_RTOL` is
  logged as `MISMATCH` (nothing booked), and an `OTHER` day stays `OTHER`; both log x as
  `implied_factor`.

`SPLIT`, `DISTRIBUTION` and `FINAL` days, dividends and delistings are unchanged (a delisting
row has no raw close, so it is never checked). The parity ledger (L3-L5) implements the rule
independently, sharing only the three tolerances (ADR 0007).

**Tolerance.** Measured on the market store, 2020-2024 (5.2 million no-factor days with a raw
close and a previous one): x is within 1e-4 of 1 on all but 49 of them. 43 are more than 1% off:
30 are booked as implied splits (4 with a finite `ret` that implies 1:20 to 1:100, 26 with a
missing `ret` after a halt or a delisting-like collapse) and 13 are logged as `MISMATCH`
(ordinary moves across a halt with a missing `ret`, between 2% and 20%, left to the raw prices).
All 4 `OTHER` days have a `splitFactor` within 1% of x and become splits by it. As a check of the
formula, x agrees with `splitFactor` within 1% on all 1,101 booked holder-split days, and with
the price factor of all 112 value distributions. 1% is far above the ordinary noise; 20% is the
threshold of the #27 scan. Above it the raw ledger follows `adjClose`, which is what quantlab's
own run values the holding at.

Known cost: where `ret` is missing across a collapse rather than a reverse split (for example
PERMNO 90090 2023-03-13, 70.00 to 0.13, x 538), the implied split conserves the holder's value
as `adjClose` does, not the loss the raw prices show. That is a data question for quantlab's
`adjClose` (a missing return counts as 0), not for the venue; the run's `events.json` lists every
`IMPLIED_SPLIT` fill (date, PERMNO, share change) so such days can be audited.
