---
status: accepted, amended 2026-10-04 (quantlab ADR 0020)
date: 2026-10-01
---

# The venue is the live seam; trader imports an exact quantlab allowlist

trader is one Strategy, one pure decision core and a set of venues. Everything that differs
between the backtest and IBKR paper trading sits behind a **venue**, so the live effort adds a
venue package and changes nothing else:

```
quantlab_trader/
  base/config.py        TraderConfig, root VenueConfig (frozen dataclasses, get_config/from_config)
  base/venue.py         Venue and its four parts (InstrumentResolver, OpenSubmitter,
                        DecisionSource, DecisionClock); BarInputs, NextOpenOrder;
                        Loop, ReplayRequest (what VenueConfig.build is asked to run) and
                        VenueReport (what Venue.run returns, e.g. minimum-fee fills, #29)
  quantlab_run.py       QuantlabRun.load(run_dir): the only reader of a quantlab run directory
  decision.py           DecisionCycle: targets -> whole-share next-open orders; TargetSource
                        with ConstructorTargets (closed loop) and TableTargets (open loop)
  account.py            positions by PERMNO, close-marked equity and current weights, read
                        from nautilus's Cache/Portfolio
  strategy.py           PortfolioStrategy: the one nautilus Strategy, backtest and live
  outputs.py            RunRecorder: the trader run directory
  runner.py             run(config): wires run, venue, strategy and recorder
  parity/               parity ladder and report (ADR 0007): ladder.py (the driver),
                        one module per step (#32); the only package that may
                        import quantlab's backtest layer
  cli.py                `quantlab-trader backtest` (and `parity`, #9)
  venue/backtest/       venue.py (BacktestVenue), feed.py, resolver.py, submitter.py,
                        source.py, clock.py, fees.py, fills.py, corporate_actions.py
  venue/ibkr/           not created in v1; the live effort's package
```

Every top-level entry of `venue/` is a venue, as every top-level entry of quantlab's `dataset/`
is a dataset. `__init__.py` files are empty; nothing is re-exported.

**The venue's parts.** `InstrumentResolver` (ADR 0004), `OpenSubmitter` (ADR 0003), a
`DecisionSource` that returns one bar's `BarInputs` (the prediction row, raw close, delisting
marks) reading nothing later than t, and a `DecisionClock` that
calls the Strategy's one decision method (backtest: a time alert at close(t) + 1 ns; live: a timer
after the daily prediction job). The market data feed, fee and fill models and corporate actions
are not parts: they are inside the backtest venue, because live they are the broker's (IBKR
charges the commission, fills the auction and books splits and dividends). The account is not a
part either: nautilus's Cache and Portfolio already present one account interface in backtest and
live, and `account.py` computes equity as cash plus each position at the raw close of t, the same
number the order sizing uses, in both modes.

**The decision core is nautilus-free.** `DecisionCycle.run(inputs, positions, cash)` returns the
`Decision` and the sized orders (`trunc(w * equity / close) - position`, sells first, NaN weight =
no order). Closed and open loop differ only in the `TargetSource`: quantlab's `DecisionInputs`
(`rebalances`, `context`) and the rule's `decide`, or a `weights.zarr` row.

**The decision-price history is a slice, not a rolling buffer.** At each decision the source hands
`build_context` the run's `adjClose` from `bar_before(anchor, lookback_bars)` to t, the same span
quantlab's panel loop forward-fills over. `lookback_bars + 1` bars (the carried minimum) is not
enough for identical contexts: a security without a price on the first bar of a short window loses
its forward-fill seed, so its first return is NaN where quantlab's is finite, and Ledoit-Wolf drops
it. Memory is a few tens of MB for a daily panel of the predicted PERMNOs; trimming is a private
optimisation of the source, allowed only if the context test still passes.

**The rebalance anchor is the run's.** Rebalance bars are counted every `rebalance_periods` bars
on the price dataset's calendar from the first timestamp of the run's `predictions.zarr` (the
window's first bar for `run()`, the first fold's for `run_cv()`). Narrowing trader's window with
`start`/`end` does not move it, and the live run keeps counting past the run's end.

**The quantlab allowlist.** trader's source may import exactly `quantlab.base.portfolio`,
`quantlab.portfolio.decision_inputs`, `quantlab.base.data` (the `MarketDataset` check),
`quantlab.core.component` (`get_cls_from_path`),
`quantlab.tracking.base`, `quantlab.utils.backtest_report`, and the public returns-statistics
module ADR 0007 has quantlab add for `metrics.json`. At run time it also loads, by class path from
`config.json`, the rule's module (`quantlab.portfolio.*`), the price dataset's module
(`quantlab.dataset.*`) and the tracker's (`quantlab.tracking.*`). The one exception is the `parity`
package, which may also import quantlab's backtest layer to run the ladder's quantlab rungs (ADR 0007).
`tests/test_quantlab_boundary.py` locks both: an ast scan of every trader module outside
`parity/` against the allowlist, and a subprocess that loads a fixture run and runs one decision through
`runner.py`, then asserts that no `quantlab.model`, `quantlab.factor`, `quantlab.label`,
`quantlab.backtest`, `torch`, `xgboost`, `KunQuant` or `vectorbt` is in `sys.modules`.

## Why

The rule is that the backtest behaves as live trading will, and the live effort is later. Putting
every difference behind one venue makes "the Strategy is the same" checkable: the IBKR adapter is a
new `venue/ibkr/` package and a `VenueConfig`, and a diff that touches `strategy.py` or
`decision.py` is a design failure. Keeping the decision core free of nautilus makes it testable
on plain arrays, which is where parity bugs will be found.

## Considered options

- Account as a fifth venue part. Rejected: one adapter is a hypothetical seam, and nautilus's own
  Portfolio is already the seam between its backtest and live engines.
- Feed, fees and corporate actions as venue parts the Strategy sees. Rejected: live, the broker
  owns them; the Strategy would carry interfaces with no live implementation.
- A rolling buffer of `lookback_bars + 1` decision prices. Rejected: not identical to quantlab's
  panel loop (forward-fill seed), and parity would carry an unexplained gap on halted securities.
- Reading quantlab's market columns from the backtester class (`MARKET`). Rejected: importing
  `quantlab.backtest.predefined.us_equity` loads vectorbt. quantlab writes a `market` block
  (`fill_price_column`, `valuation_price_column`) into `config.json` instead.
- The `execution` block on `TraderConfig` (#5). Moved into `BacktestVenueConfig.execution`: fees,
  slippage and the simulated starting cash exist only in the backtest venue.

## Consequences

- quantlab adds the `market` block to a run's `config.json`, next to ADR 0005's
  `predictions.zarr` and public `build_context` / `decide`.
- Order of work: the first tracer bullet is the open-loop replay through the whole trader stack
  (config, `QuantlabRun`, backtest venue with open and close prints, Strategy, `DecisionCycle` with
  `TableTargets`, outputs, CLI, boundary test) on a small synthetic run, plus quantlab's `market`
  block. It needs nothing else from quantlab, and closed loop then adds only `ConstructorTargets`
  and the history slice, once ADR 0005's API lands (worked on in parallel). ADR 0006's quantlab
  changes come before any parity run on real data.
- Splits change a position's quantity, which nautilus does not model; the corporate-actions
  ticket starts with a spike.

## Amended by quantlab ADR 0019 (trader #34)

trader assembles no decision inputs of its own. `QuantlabRun.decision_inputs(end)` returns
quantlab's `DecisionInputs.from_run(run_dir, end=end)`: the bound rule, the price dataset, the
market columns, the rebalance period and the anchor. `ConstructorTargets` asks it whether a bar
`rebalances` and for the bar's `context` (with the account's holdings), then calls the rule's
`decide`. `calendar.py` (`RebalanceCalendar`), `runner._history_start`, `ReplayRequest.history_start`,
the source's decision-price history and trader's alignment of predicted and held symbols are
deleted; the venue's per-bar value type is renamed `BarInputs`, since quantlab's `DecisionInputs`
now names the decision's inputs.

The paragraph "The decision-price history is a slice, not a rolling buffer" no longer holds:
quantlab bounds every decision to the rule's last `history_bars` raw prices, forward-filled within
that window, so a decision no longer depends on where the history starts, and the replay keeps no
history of its own. The rebalance anchor is still the run's (the prediction panel's first bar),
and the replay's last bar still never rebalances (`end`).

## Amendment (2026-10-04, quantlab ADR 0020)

trader reads a quantlab run only through quantlab's `quantlab.runs.backtest_run.BacktestRun`,
which joins the allowlist: the window, market columns, annualization, execution settings,
rebalance period, initial cash and data fingerprint as typed values; the rebalance table,
prediction panel, metrics and equity curve as loaded objects; the price dataset, rule and tracker
rebuilt from the run's recipe (`rebuild(field)`). The `market` block and the data fingerprint live
in the run's `run.json`, not its `config.json`. The parity rung rebuilds the backtester with
`rebuild_backtester(**overrides)` instead of editing a config dict. trader names no file of a
quantlab run and indexes no key of its config; `tests/test_quantlab_boundary.py` adds a source scan
for both, outside `outputs.py`, which owns trader's own run directory and its files of the same
names. A run's window is its first and last simulated bar, so a replay window narrowed by
`start`/`end` must lie within the bars the run simulated. `quantlab.core.component` stays on the allowlist for `get_cls_from_path` (a tracker named in
a trader config).
