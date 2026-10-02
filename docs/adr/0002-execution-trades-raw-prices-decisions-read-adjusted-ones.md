---
status: accepted
date: 2026-10-01
---

# Execution trades raw prices; decisions read adjusted ones

The bars trader feeds NautilusTrader carry the **raw** prices a broker would quote (CRSP `open`
and `close` from the quantlab run's price dataset), orders are whole shares at those prices, a
split changes the share count of a held position, and a dividend is booked as cash on its
ex-date. Decisions keep reading quantlab's **adjusted** data: the prediction panel was computed
from it, and the `returns` window of a `PortfolioContext` is built from adjusted closes so a split
is not a crash.

The rule is that the backtest behaves as live trading will. A live account holds whole shares
bought at quoted prices and receives dividends as cash; on adjusted prices, a share count is a
fiction (early adjusted prices can be a fraction of the quote, so the count is several times too
large) and so are fees charged on it.

## Considered options

- Execute on adjusted prices, as quantlab's vectorbt engine does. Rejected: simpler and directly
  comparable to vectorbt, but share counts, per-share fees and cash are not what an account sees.

## Consequences

- trader applies corporate actions to held positions (splits, dividends, delisting settlement)
  from the price dataset's raw fields.
- The open-loop replay against quantlab's vectorbt run differs by the adjusted/raw execution
  basis as well as by integer shares; the parity ticket accounts for it.
