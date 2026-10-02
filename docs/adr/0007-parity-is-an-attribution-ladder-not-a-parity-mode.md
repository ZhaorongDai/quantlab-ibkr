---
status: accepted
date: 2026-10-01
---

# Parity is an attribution ladder, not a parity mode

trader's open-loop replay and quantlab's vectorbt run of the same rebalance table cannot be equal:
trader sizes whole shares at t's raw close on close-marked equity, lets cash go negative, books
dividends as cash and charges its own fee model (ADR 0002, 0003); vectorbt sizes fractional
shares on adjusted prices against the portfolio valued at t+1's open and caps buys at cash.
Parity therefore does not ask for equal results. It asks that **every difference is attributed**:
a **parity ladder** of simulations walks from quantlab's run to trader's run one convention at a
time, the two ends of the ladder are asserted to reproduce the real runs, and each rung's gap is
reported. The unexplained residual is zero by construction, so acceptance is "fully attributed",
never "small".

trader has **no parity mode**. No execution setting sizes at the open, trades fractional shares,
executes adjusted prices or caps cash to look like vectorbt. Parity is a diagnostic and never a
reason to make the backtest less like live trading.

**The ladder** (open loop, one quantlab run's `weights.zarr`, fixed order):

| Rung | Simulation | Changes from the rung above |
|---|---|---|
| L0 | quantlab `run_weights`, as the run did | none: must reproduce the run's `equity.zarr` |
| L1 | quantlab `run_weights` with close sizing | size from t's valuation close, not t+1's open (vectorbt `val_price`) |
| L2 | reference ledger | cash no longer capped |
| L3 | reference ledger | raw prices; splits change share counts, dividends are cash |
| L4 | reference ledger | whole shares, `trunc` |
| L5 | reference ledger | trader's fee and slippage models, cent and tick rounding |
| T  | trader open-loop replay (nautilus) | none: must equal L5 |

L0 and L1 run quantlab's own engine, so trader never re-implements vectorbt's sizing, rejections or
settlements; quantlab's vectorbt engine gains a sizing-basis option for L1 (default unchanged:
the fill price). L2-L5 run the **reference ledger**, a small numpy simulator in trader's parity
module that applies trader's execution conventions one switch at a time. It is an independent
oracle of the nautilus run, never called by the strategy. The order of the rungs is fixed and
reported, because the gaps do not commute.

**Asserted** (synthetic fixtures in CI, and on every real parity run):

- L0 equals the quantlab run's `equity.zarr` to a relative 1e-12 per bar (quantlab's own anchor
  tolerance across macOS and Linux).
- T equals L5: the same orders (PERMNO, fill bar, side, integer quantity), the same fill prices at
  the instrument's precision, the same fees to the cent, the same rejected orders and settlements,
  and equity per bar within USD 0.01 times the number of fills so far (money rounding).
- On fixtures where no buy is cash-capped, L2 equals L1 to a relative 1e-12, which validates the
  ledger against vectorbt; on fixtures without corporate actions L3 equals L2, with integral
  target sizes L4 equals L3, with the fraction fee model L5 differs from L4 by rounding only.
- **Closed versus open loop**: on every rebalance bar where the decision does not depend on
  holdings (no locked position in either book, no hold; TopN, or mean-variance without turnover
  penalty), the closed loop's decided weights equal the run's `weights.zarr` row bit for bit, and
  in a fixture where that holds on every bar, the closed-loop and open-loop trader runs have
  identical orders and equity.

**Reported only**: every rung's gap (and closed versus open loop) on real data, with no numeric
gate.

**Reference data.** CI uses synthetic, offline, model-free fixtures built through quantlab's
public API (a CRSP-shaped `MarketDataset` with raw and adjusted prices, a split, a dividend, a
halt, a late listing, a delisting and an opening gap that makes vectorbt cap a buy; predictions
written as a `PredictionPanel`; the quantlab run produced by `construct_panel` and `run_weights`).
trader imports nothing from quantlab's `tests/`. Real parity runs are quantlab runs on the training
server's CRSP stores, written after ADR 0006's changes: `sp500_xgb` (TopN), `sp500_xgb_mvo`
(mean-variance) and `market_xgb` (TopN on the market universe).

**The parity report** is a directory `parity.json` + `parity.zarr`: the inputs (run directories and
their data fingerprints, which must agree), the end checks and their maximum errors, one row per
rung (final equity, total and annualised return, Sharpe, max drawdown, turnover, total fees,
order count, rejected orders, settlements, `max_target_deviation`, buys capped, peak cash debit)
with its delta to the rung above, the closed-versus-open block (rebalance bars compared, bars
equal, max and mean L1 distance of the weights, equity and statistic deltas) and, in the zarr,
each rung's per-bar equity.

**trader's `metrics.json`** uses quantlab's layout and names where the meaning is the same:
`whole`, `in_sample` and `out_of_sample` (ranges copied from the quantlab run's `metrics.json`)
with the return statistics computed by the same function as quantlab's, plus `Total Orders`,
`Total Fees Paid` and the turnover rows; `execution` with `rejected_order_count`,
`rejected_orders`, `max_target_deviation` and the settlements; `portfolio_construction` (held
bars, closed loop); `benchmark` and `relative` against the quantlab run's `benchmark_value`.
Trader-only facts go under `execution.trader` (commissions, minimum-fee hits, dividends, splits,
peak cash debit). Trade-level statistics (position round trips, win rate) are written under
quantlab's names but are not parity quantities.

## Considered options

- A parity execution setting in trader (open sizing, fractional shares, adjusted prices, cash cap)
  so trader can be asserted equal to vectorbt. Rejected: it is order-path code live never runs,
  and open sizing cannot exist at all in a decision made at t's close (ADR 0003).
- Compare quantlab and trader directly with loose tolerances on the headline statistics.
  Rejected: a tolerance hides a bug as easily as it absorbs a convention, and real gaps depend
  on the data.
- Run every rung in vectorbt. Rejected: vectorbt cannot leave cash uncapped, book dividends as
  cash or charge a per-share fee with a minimum and a cap.
- Re-implement the whole ladder, L0 included, in trader. Rejected: it copies quantlab's execution
  plan; quantlab's engine stays the only vectorbt code.

## Consequences

- quantlab: the vectorbt engine gains a sizing-basis option (fill price, default; or t's valuation
  close through `val_price`), and the returns-based statistics and turnover rows of
  `metrics.json` move into a public function in a module that imports neither the model nor the
  dataset layer, which trader's metrics call.
- trader's parity package is the only trader package allowed to import quantlab's backtest layer;
  a layout test locks that the strategy, venues and run path do not.
- A parity run uses the quantlab run's rebalance bars, whatever the calendar anchor decided later
  for live.
- The ladder is open loop; closed-loop gaps are reported as feedback, against an open loop run
  with the same `execution` block.
