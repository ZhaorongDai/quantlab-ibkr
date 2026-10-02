---
status: accepted
date: 2026-10-01
---

# The strategy decides at the close; the venue submits its orders for the next open

trader's strategy runs one **decision** per rebalance bar, after the close of day t, and the code
is the same in the backtest and in IBKR paper trading. It builds the `PortfolioContext`, calls
`construct()`, and sizes each target weight into whole shares from what is known at that moment:
account equity marked at the **raw close of t**, the raw close of t, and the current position,
`trunc(w * equity / close) - position`. It never reads day t+1's open. The result is a list of
**next-open orders** (security, side, quantity), sells before buys, which the strategy hands to
the venue's **open submitter**. The open submitter is the only order code that differs between
venues:

- **Live (IBKR, built later):** it submits each order as a market-on-open order
  (`MarketOrder`, `TimeInForce.AT_THE_OPEN`, sent as `MKT`/`OPG`) before a configured deadline,
  ahead of IBKR's 09:28 ET Nasdaq cut-off; the opening auction fills it.
- **Backtest:** NautilusTrader 1.231 rejects `AT_THE_OPEN` market orders ("is not currently
  supported", `backtest/engine.pyx:5333-5339`), so the submitter holds the orders and, at the
  open of t+1 plus 1 ns (after every symbol's opening `TradeTick` is applied), submits each one as
  a plain `MarketOrder` with `TimeInForce.DAY`, which fills at the open. An order whose security
  has no opening print that day is not submitted: it is a **rejected order** in quantlab's ADR
  0014 sense, the holding is kept, and the strategy is told through the same callback the live
  submitter uses when IBKR leaves an on-open order unfilled.

The backtest's feed carries, per security and day, an opening `TradeTick` at 09:30 ET, a closing
`TradeTick` at 16:00 ET and the daily `Bar` at 16:00 ET. The closing tick is what marks equity at
the close: nautilus prices positions from the last trade before it falls back to bars, so without
it equity at the decision is marked at that morning's open (verified, see below).

## Why

The rule is that the backtest behaves as live trading will. A live on-open order is sized before
the open is known, so the backtest sizes from the close too; the open gap then lands in the fill,
in leftover or borrowed cash and in the next decision, as it will live. The submitter is allowed
to look at t+1 (the opening print) because it plays the market's answer, which quantlab's ADR 0014
puts on the execution side; it never changes a quantity.

## Considered options

- Submit `AT_THE_OPEN` in both modes and turn it into an open fill inside nautilus. Rejected: the
  matching engine is Cython and rejects the time in force in a `cdef` method that cannot be
  overridden from Python. An `ExecAlgorithm` that receives the on-open order and spawns a market
  child at the open fills at the open (verified), but the strategy's own order never reaches a
  venue: spawning its whole quantity with `reduce_primary=True` raises (an order cannot be
  updated to quantity 0), and with `reduce_primary=False` it stays `INITIALIZED` for good while
  a child order `...-E1` carries the fill. The order life cycle the strategy sees then differs
  from live, which defeats the purpose.
- Plain market orders in both modes, submitted at the open. Rejected: live it is a continuous-
  session market order, not the opening auction, and needs a process awake at 09:30 instead of
  orders queued the evening before.
- Size at the open in the backtest (the research pattern of #3, and vectorbt's sizing at the fill
  price). Rejected: it uses a price live cannot know when the order is due.
- `LatencyModel` to push orders from the decision to the open. Rejected in #3: commands settle
  before other symbols' same-timestamp ticks, so many fill at the previous close.

## Consequences

- The strategy holds the budget, not the venue. nautilus's risk engine does no balance check at
  all for a margin account (`risk/engine.pyx:690-691`, "TODO: Determine risk controls for
  margin"; verified: buys of twice the equity filled). Sizing at the close keeps gross exposure at
  the constructor's weights; an opening gap makes cash slightly negative or positive, as on IBKR's
  margin account. vectorbt caps buys at cash instead; the parity ticket accounts for that.
- One more tick per security and day in the feed (about 3.8M data objects at 500 x 10y).
- Fills differ from live by the price only: the backtest fills at CRSP's open, live at the
  official opening auction price (IBKR also says it may simulate market orders on some
  exchanges). Auction partial fills, and IBKR's handling of an on-open order for a halted symbol,
  are not modelled and are to be checked on the paper account.
