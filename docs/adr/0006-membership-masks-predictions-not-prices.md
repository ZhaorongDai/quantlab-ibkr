---
status: accepted
date: 2026-10-01
---

# Membership masks predictions, never prices; the run's price dataset is the unmasked market

An index's point-in-time membership is a **universe**: it says which securities a strategy may
enter at bar t. It is a decision input, so quantlab applies it to the prediction panel (NaN where
the PERMNO is not a member at t) and never to prices. The quantlab run's `price_dataset` is an
**unmasked market dataset**: a price is missing only where the market had none (before listing,
a halt, a no-trade day, after delisting). trader takes everything market-side from that one
dataset and has no execution-price setting of its own:

- **execution prices** (ADR 0002): raw `open` and `close`;
- **corporate actions**: `splitFactor`, `divCash` on the ex-date, and the delisting row
  (`MarketDataset.delisting_bars`, CRSP `is_delisting`);
- **decision prices**: `adjClose`, for the `returns` window and staleness of each
  `PortfolioContext`.

`tradable`, `staleness` and delisting are market facts computed from those bars, with the
dataset's own `tradable_bars` / `delisting_bars` on quantlab's `MarketSpec` columns, so they agree
with quantlab's vectorbt run by construction. Membership never enters them. A security that leaves
the index stays tradable: its prediction turns NaN, and what happens to a holding is the rule's
call (TopN sells it at the next open, mean-variance holds it at mu=0), as it would be live.

trader feeds bars for the whole window to every instrument (one per PERMNO with a finite
prediction, ADR 0004), member or not on a given day. It refuses a run whose price dataset is not a
`MarketDataset` or lacks `open`, `close`, `adjClose`, `splitFactor` or `divCash`.

## Why

The rule is that the backtest behaves as live trading will. Live, a stock that drops out of the
S&P 500 is still quoted and can be sold at the next open. Every `sp500_*` and `nasdaq100_*`
example today passes `members.zarr` (the price panel `.where(is_member)`) as `price_dataset`, so
in quantlab's own run a leaver loses its prices: it is untradable (a locked position) on its first
non-member bar, and `delisting_bars`, finding no later price in the store, settles it into cash at
its last member close, a sale that never happens. That store also holds only `adj*`, `close`,
`volume` and `ret`, so it cannot supply raw opens or corporate actions at all.

Membership is also not in the prediction panel today, contrary to what instrument identity (#8)
assumed: the examples' factors read the unmasked `prices.zarr`, and `predict_window` returns a
prediction for every symbol with features. Membership reaches the decision only through the masked
price dataset's `tradable`. Feeding trader unmasked bars without moving membership into the
predictions would let it buy non-members.

## Considered options

- trader config names a separate unmasked CRSP store for execution bars, and the run's
  `price_dataset` keeps supplying decision prices and tradability. Rejected: trader would trade a
  universe defined by one dataset at prices from another, quantlab's own run would keep selling
  leavers at their last member close, and parity would need a membership term to explain the gap.
- Derive tradability from membership (non-member = untradable), as the masked store does today.
  Rejected: it locks holdings that live trading would sell, and it is not a market fact.
- Keep membership in the price dataset and let trader read raw fields from a second store only for
  execution. Rejected: two stores on one PERMNO axis that disagree on which cells exist.

## Consequences

- quantlab: `sp500_*` and `nasdaq100_*` examples pass the unmasked index store
  (`CrspStockDataset` over `wrds_crsp_{index}_1d.zarr`) as `price_dataset`; `members.zarr` remains
  only the label's input. Their published metrics change.
- quantlab: the backtester masks predictions with point-in-time membership at t before
  `construct_panel` and before writing `predictions.zarr` (ADR 0005), and `predict_date` applies
  the same mask, so the panel trader reads already carries the universe. The mechanism is
  quantlab's to choose; the mask reads nothing later than t.
- Parity needs no membership term: quantlab and trader read the same price dataset and the same
  masked predictions. Runs written before this change are not valid parity inputs.
- The market universe's CRSP security filter drops rows per day, so a security that stops being,
  say, common stock loses its prices like a delisting. That is a mask on prices too, and is open.
