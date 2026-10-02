# PROTOTYPE: corporate actions in the backtest venue (#15)

Throwaway code. It answers one question and is kept on branch `prototype/corporate-actions` as a
primary source; it is not part of the package.

**Question.** How does the backtest venue apply a split, a cash dividend and a delisting to
NautilusTrader positions and cash, with the Strategy unaware of the venue, so that
`equity = cash + sum(qty * raw close)` holds after each? Is a MARGIN account's balance that cash
figure?

**Run** (nautilus_trader 1.231.0, from the repo root):

```
~/projects/quantlab/.venv/bin/python prototypes/corporate_actions/run.py            # chosen mechanism
~/projects/quantlab/.venv/bin/python prototypes/corporate_actions/run.py --help     # variants
```

Setup: venue `CRSP`, ids `<PERMNO>.CRSP`, USD, price precision 4, lot 1, NETTING, MARGIN
(leverage 1, `margin_init=1`), an IBKR-Fixed-like per-share fee (min 1.00). Per security-day an
opening `TradeTick` at 09:30 ET and a closing one at 16:00 ET (ADR 0003). Four securities: a long
through a split with a pre-split exit order queued for the ex-date open, a long with a partial
sell and a dividend, a long that delists, and a short that pays a dividend and splits 3:2. The
Strategy replays a fixed order schedule at open + 1 ns and records state after each close; an
independent ledger in the same file checks every close.

## Result

The chosen mechanism passes every check, for splitFactor 2, 3 and 1/3 (reverse, with cash in
lieu), on longs and shorts:

```
=== split A, 10001 splitFactor 2, delisting module, scale pending True ===
-- applied by the venue module:
   01-07 09:30 split x2 10001.CRSP: 100 -> 200, cash in lieu +0.00
   01-07 09:30 dividend 0.5 x 60 10002.CRSP = +30.00
   01-07 09:30 dividend 0.25 x -100 10004.CRSP = -25.00
   01-08 09:30 split x1.5 10004.CRSP: -100 -> -150, cash in lieu -0.00
-- fills the Strategy received:
   ...
   01-07 09:30 10001.CRSP BUY 100 @ 0.0000 fee 0.00 USD [CA-SPLIT-...]
   01-07 09:30 10001.CRSP SELL 200 @ 51.5000 fee 1.00 USD [O-20260107-143000-001-000-6]
   01-08 09:30 10004.CRSP SELL 50 @ 0.0000 fee 0.00 USD [CA-SPLIT-...]
   01-09 09:30 10003.CRSP SELL 100 @ 22.5000 fee 0.00 USD [CA-DELIST-...]
-- after each close (n = nautilus-derived, r = reference ledger):
   2026-01-05 ... | n cash=85996.00 r cash=85996.00 | n equity=100316.00 r equity=100316.00 | margin balance=99996.00 (== cash: False) | ...
   2026-01-06 ... | n cash=88003.00 r cash=88003.00 | n equity=100527.00 r equity=100527.00 | margin balance=100003.00 (== cash: False) | ...
   2026-01-07 ... | n cash=98307.00 r cash=98307.00 | n equity=100832.00 r equity=100832.00 | margin balance=100307.00 (== cash: False) | ...
   2026-01-08 ... | n cash=98307.00 r cash=98307.00 | n equity=100963.01 r equity=100963.01 | margin balance=100307.00 (== cash: False) | ...
   2026-01-09 ... | n cash=100557.00 r cash=100557.00 | n equity=101025.00 r equity=101025.00 | margin balance=100557.00 (== cash: True) | ...
PASS
```

(The `True` on the last day is a coincidence: the open long and short cost bases cancel.)

## Findings (all verified by running `run.py`)

1. **A MARGIN account's balance is not cash.** It moves only by realized PnL, commissions and
   `adjust_account` adjustments, never by a fill's notional. The cash figure the identity needs
   is `cash = balance - sum(signed_qty * avg_px_open)` over open positions; it matched the
   reference ledger at every close, through buys, a partial sell, a short, splits, dividends and a
   delisting. `account.py` must compute it this way.
2. **`Portfolio.unrealized_pnls(venue)` is stale here** even with trade ticks subscribed: on day 2
   `balance + unrealized` was 100327.00 against a true equity of 100527.00. Equity must be
   `cash + sum(signed_qty * cache.trade_tick(id).price)`, not `balance + unrealized`.
3. **A `SimulationModule` can generate fills on the strategy's position.** `process(ts_now)` runs
   once per timestamp after commands settle (`engine.pyx:1926-1930`); at the ex-date's 09:30 tick
   it runs before the open + 1 ns orders. It builds a `MarketOrder` with the position's
   `trader_id`/`strategy_id`, `cache.add_order(order, position_id=...)`,
   `exec_client.generate_order_submitted/accepted`, then `matching_engine.apply_fills(order,
   [(price, qty)], TAKER, None, position)`: the same path `check_instrument_expiration` uses for
   settlements (`engine.pyx:5934-5981`). The Strategy receives an ordinary `OrderFilled` for an
   order it never submitted, tagged `CORPORATE_ACTION_*`.
4. **Split, variant A (chosen): one fill of the share-count change at price 0.** Cash does not
   move, the quantity becomes `floor(q * k)` (toward zero for shorts), and the fraction is cash in
   lieu at `pre-split close / k` through `adjust_account`. Exact for any factor: no price is
   rounded. On a forward split nautilus's `avg_px_open` becomes `old / k` (the cost basis is kept);
   on a reverse split the price-0 *sell* books a fictitious realized loss (-6701 on 100 -> 33) and
   leaves `avg_px_open` at the old price. Equity and derived cash are still exact; only nautilus's
   own position statistics are off, and trader does not use them.
5. **Split, variant B (rejected): close at the pre-split close, reopen at close / k.** It realizes
   a fictitious PnL on every split, opens a new cost basis, and the reopen price is rounded to the
   tick: at 3:1 (101 / 3 = 33.6667) cash drifted by 0.01 from the ledger (`--split B --factor 3`
   prints FAIL).
6. **Dividend: `exchange.adjust_account(Money(divCash * signed_qty))`** at 09:30 of the ex-date,
   on the position held at the prior close. A short pays (-25.00 on -100 x 0.25). Fee-free by
   construction.
7. **Delisting: both mechanisms settle at the last valuation with no fee.**
   `InstrumentClose(CONTRACT_EXPIRED)` + `settlement_prices` fills at 22.5 with fee 0.00 once the
   fee model returns zero for `EXPIRATION_*_CLOSE` tags (this closes the "untested" item of #3).
   The module's own settlement fill (the default; `--delist native` for the other) gives
   identical numbers.
8. **A next-open order queued across a split is in pre-split shares.** Without rescaling
   (`--no-scale-pending`), an exit of 100 pre-split shares sold 100 of the 200 post-split shares
   and left half the position open. The prototype's submitter multiplies such orders by the
   ex-date's splitFactor (floor), which sold all 200.
9. CRSP's `splitFactor` is `dlycumfacpr[t-1] / dlycumfacpr[t]` (`quantlab/dataset/crsp/__init__.py:32`),
   which is 2.0 for a 2:1 split, the convention used here.

## Choice

The backtest venue's `corporate_actions.py` is one `SimulationModule` holding a schedule built
from the price dataset (`splitFactor != 1`, `divCash != 0`, `delisting_bars`) and applying, at
09:30 ET of the ex-date (delisting: of bar b+1):

- split: variant A, a price-0 fill of the share-count change plus cash in lieu;
- dividend: `adjust_account`;
- delisting: a settlement fill at the last valuation from the same module (one mechanism, one tag
  family, one event record), rather than `InstrumentClose` + `settlement_prices`, which is a
  verified, equivalent fallback;

all tagged `CORPORATE_ACTION_*` and zero-fee in every trader fee model. The open submitter
rescales a queued order by the ex-date's splitFactor. See ADR 0009.
