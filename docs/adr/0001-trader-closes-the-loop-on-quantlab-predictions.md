---
status: accepted
date: 2026-10-01
---

# trader closes the loop on quantlab's predictions

quantlab and quantlab-trader split a strategy where it stops depending on holdings. Factors and
the return model's predictions do not depend on what the account holds, so quantlab computes them
in batch and hands trader a **prediction panel**: the whole period for a backtest, one decision
date per day in live trading. Portfolio construction does depend on holdings (locked positions,
turnover penalties, top-n retention), so trader runs it: on each rebalance bar it builds a
`PortfolioContext` from the account NautilusTrader holds, calls quantlab's
`PortfolioConstructor.construct(context)` in process (the per-bar interface of quantlab's ADR
0012), sizes the resulting weights into share orders, and fills them at the next open.

That loop is the same in a backtest (**closed-loop replay**, the default) and in IBKR paper
trading, which is what makes the event-driven backtest an accurate rehearsal of live trading:
rounding, rejected orders and partial fills feed into every later decision.

trader also keeps an **open-loop replay**: executing a finished **rebalance table** (quantlab's
`weights.zarr`) without calling back into construction. It is cheap and isolates execution: the
gap between it and quantlab's vectorbt run is execution alone, and the gap between closed and
open loop is the feedback of real holdings.

trader imports quantlab's portfolio layer at run time, never its factor or model layers. Weights,
not shares, cross the boundary; share counts depend on equity and cash, known only at execution.

## Considered options

- Run the whole decision (factors, prediction, construction) inside a nautilus Strategy.
  Rejected: the live process would carry torch, xgboost, KunQuant and the CRSP readers, and
  re-run holding-independent work on every bar.
- quantlab writes a rebalance table, trader only executes it, with live holdings sent back to
  quantlab as a position snapshot file (the first version of this decision, same day). Rejected:
  a backtest could then only be open loop, which is not accurate; and a per-bar file round trip
  through a quantlab process is too slow over thousands of bars.
- Closed loop only. Rejected: open-loop replay costs almost nothing and is the only way to
  separate execution differences from feedback differences.

## Consequences

- quantlab must make the one-bar decision public: building a `PortfolioContext` (tradable,
  returns, staleness), checking locked positions, and rebuilding a constructor from a run's
  config without loading the model.
- quantlab gains a daily prediction step for a single decision date, which needs that date's
  market data (CRSP does not provide it live).
- The prediction panel and rebalance table formats are the public file contract between the
  projects and are versioned.
