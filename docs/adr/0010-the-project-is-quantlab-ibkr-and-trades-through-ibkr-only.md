---
status: accepted
date: 2026-10-07
---

# The project is quantlab-ibkr and trades through IBKR only

quantlab-trader is renamed quantlab-ibkr (repository, distribution, package `quantlab_ibkr`,
command `quantlab-ibkr`). Interactive Brokers is the one broker it will ever trade through; a
second broker is a second project, not a second venue here.

**What this fixes.** The live venue is `venue/ibkr/` and nothing else (ADR 0008's layout already
named it). Code may assume IBKR's conventions wherever live trading needs a convention: its order
types and opening-auction cut-off for next-open orders (ADR 0003), its commission schedule, its
contract identifiers in the instrument resolver (ADR 0004), and its reporting of corporate actions
and cash. No broker interface, broker registry or broker-neutral order model is added above the
venue; the backtest venue simulates IBKR, so its closed-loop default fee model stays IBKR Pro
Fixed (`ibkr_fixed`).

**What does not change.** The venue seam stays: it separates the backtest from live trading, not
one broker from another, and the backtest venue is its second implementation. The `fraction`
fee model stays as well: it is quantlab's cost convention, used by the open loop and the parity
ladder (ADR 0007) to compare with quantlab's run, not another broker's schedule. "trader" remains
the word for this project's role beside quantlab (the executor of a quantlab run), so
`TraderConfig`, the "Trader run" of the glossary and `metrics.json`'s `execution.trader` block keep
their names.
