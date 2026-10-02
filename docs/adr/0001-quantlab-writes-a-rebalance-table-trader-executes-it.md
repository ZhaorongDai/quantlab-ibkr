---
status: accepted
date: 2026-10-01
---

# quantlab writes a rebalance table; trader executes it

quantlab and quantlab-trader meet at two files, not at code. quantlab decides what to hold and
writes a **rebalance table**: target weights per decision date and symbol, the D-03 weights
contract quantlab's own backtests already write as `weights.zarr`. trader decides how to get
there: it turns weights into share orders at execution time and fills them through
NautilusTrader, simulated (backtest) or against IBKR (paper). In live trading trader writes back
a **position snapshot** after each cycle, and quantlab reads it as the current holdings when it
writes the next day's table, so locked positions, turnover penalties and top-n retention see the
account as it is rather than quantlab's simulated book.

A backtest is an **open-loop replay**: quantlab writes the whole period's table once and trader
replays it; nautilus fills do not feed back into later decisions. Differences between the
simulated book the table was built on and nautilus's fills are what the parity check measures.

The table carries weights, not shares: share counts depend on the account's equity and cash at
execution, which only the execution side knows.

## Considered options

- Run quantlab's per-bar decision inside a nautilus Strategy (factors, prediction and
  `PortfolioConstructor.construct(context)` with a context built from nautilus state). Rejected:
  the live process would carry torch, xgboost, KunQuant and the CRSP readers; trader would depend
  on quantlab's decision internals; and the trading is daily, computed after the close or before
  the open, so nothing needs a decision inside the event loop.
- Closed-loop backtest (call quantlab back on every bar with nautilus's holdings). Rejected for
  v1: it is the same coupling as above; daily halts and rounding make the open-loop error small
  and measurable.
- Live trading on quantlab's simulated book, without a position snapshot. Rejected: rounding,
  rejections and partial fills make the real account drift from the simulated one, and the drift
  compounds through every rule that reads current weights.

## Consequences

- trader does not import quantlab's model, factor or portfolio layers at run time.
- quantlab gains a daily entry point: given a decision date and a position snapshot, write that
  date's rebalance table (and needs that date's market data, which CRSP does not provide live).
- The two file formats are the public contract between the projects and are versioned.
