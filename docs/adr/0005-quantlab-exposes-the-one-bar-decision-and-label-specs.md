---
status: accepted
date: 2026-10-01
---

# quantlab exposes the one-bar decision, and constructors bind to label specs

trader's closed loop (ADR 0001) needs four things from quantlab that are private or missing today.
quantlab adds them as follows. These are decisions about quantlab's code, recorded here because
trader depends on their shape. quantlab records its own ADR when it implements them.

**Label specs replace the predictor in `bind`.** `PortfolioConstructor.bind(predictor)` reads label
*objects*: their `get_factor_names()`, and `span_bars()` for mean-variance (`mean_variance.py:433-447`),
plus `predictor.label_scales`. A process that has no model can only satisfy that by faking label
objects. So `quantlab/portfolio/base.py` gains a frozen `LabelSpec(name, scale, delay, span)`
(`span` is `None` for a label that is not a `Forward` label), and `bind` takes
`Sequence[LabelSpec]`. `quantlab/backtest/base.py` gains `label_specs(predictor)`, which derives the
specs from `labels`, `label_delays` and `label_scales`. The backtester binds through it
(`us_equity.py:102` today). This is the only metadata a rule may read about a prediction.

**The prediction panel is a portfolio-layer file.** `quantlab/portfolio/prediction_panel.py`
(framework level, not `predefined/`) holds `PredictionPanel(predictions, labels)` with
`write(path)` and `read(path)`. The format is `predictions.zarr`: one variable per label name on
`(timestamp, symbol)`, with `attrs` holding `format_version` and `labels`, a JSON list of the specs.
The backtester writes it into every run directory that has a model. For `run()` this is the window's
predictions after they are reindexed onto the price axes (`backtest.py:1741`), which is exactly what
`construct_panel` read. For `run_cv()` it is the concatenated fold predictions, written to the
stitched directory only. A `run_weights()` run has no model and writes no panel. The same module has
`load_constructor(run_dir)`: it rebuilds the rule from `config.json["constructor"]` through its
class's `from_config` and binds it to the panel's label specs. It never touches the model or factor
layers.

**The one-bar decision is two public constructor methods.** Both trader and `construct_panel` call
them:

- `build_context(timestamp, predictions, tradable, current_weights, *, valuation_price=None,
  factors=None) -> PortfolioContext`. It takes the raw valuation prices ending at `timestamp`
  (adjusted closes, ADR 0002) and computes the `returns` window and `staleness` with the same private
  formula `_check_prices` uses today (`portfolio.py:1036-1043`). It refuses the same inputs
  `construct_panel` refuses: prices missing when `lookback_bars > 0`, and factor values missing
  when `required_factors()` is not empty. Staleness is counted within the history given, and NaN when
  that history holds no price.
- `decide(context) -> Decision`. It calls `construct`, runs the row check that `_checked_row` does
  today (`portfolio.py:1077`): no mixed NaN, no change to a locked position, no weight on a symbol
  that is neither tradable nor held. It also turns a `PortfolioConstructionError` into a hold.
  `Decision` is a frozen dataclass of `weights` (on `context.symbols`, all NaN = hold), `failure`
  (the error message, or `None`) and `events` (the row's `attrs["events"]`). A broken contract still
  raises `ValueError`, because that is a bug and not a hold. `decide` is not meant to be overridden.

`construct_panel` keeps its signature, and its loop becomes "build the bar's context, `decide`,
queue". `_Book` stays private: in trader, `current_weights` come from nautilus's holdings. A test
locks that, at every rebalance bar, `build_context` on the same price history gives the context the
panel loop built.

**The daily prediction step is one function.** `quantlab/backtest/prediction.py` holds
`predict_date(run_dir, date) -> PredictionPanel`. It rebuilds the backtester from the run's
`config.json`. It loads the checkpoint the run used: `checkpoint` for a load-mode run, the recorded
`trained_checkpoint` for a train-mode run. A `run_cv()` run is refused, because live trading serves
one model. It then calls `predict_window(date, date)` and wraps the result with `label_specs`. It
never trains, and it never extends factor stores: refreshing the data up to `date` (the price
dataset, and `factor.extend` under `factor_data_strategy="read"`) is the data job's work, and it runs
first. If the data has no bar at `date`, `predict_date` raises and never falls back to an earlier
date. A symbol without a prediction is NaN, as in the backtest, so it is never selectable. quantlab
adds no CLI. The later live effort wraps `predict_date` in its own daily job and writes one panel per
decision date with `PredictionPanel.write`.

## Considered options

- Keep `bind(predictor)` and give trader a metadata object that pretends to be a predictor, with
  stub label objects that answer `get_factor_names()` and `span_bars()`. Rejected: it is a second
  implementation of the label interface, kept in step only by convention, and a rule reading
  another label method would break it silently.
- Make the row check a standalone public function, `check_row(weights, context)`, and let callers
  catch `PortfolioConstructionError` themselves. Rejected: every caller would have to repeat the
  sequence of construct, check, and convert failure to hold. If either step is missing, trader
  diverges from the backtest.
- A one-bar `construct_panel` call, with prices from `bar_before(D, lookback)`. Rejected: `_Book`
  starts flat, so the real holdings cannot be passed in (research #4).
- Put the reader and writer for `predictions.zarr` in `base/backtest.py`. Rejected: that module
  imports the model and dataset layers, which trader must not load at run time.
- `predict_date` extends factor stores itself. Rejected: it would give the prediction a write side
  effect on shared stores, and it would mix the live data refresh (out of scope here) into the model
  step.

## Consequences

- `TopNConstructor.bind` and `MeanVarianceOptimizer.bind` read specs. `MeanVarianceOptimizer` refuses
  a spec whose `span` is `None` with the same message it gives today. `_label_names` and
  `_label_span` disappear.
- trader's run-time imports from quantlab are `quantlab.portfolio.base`,
  `quantlab.portfolio.prediction_panel` and the configured rule's module, plus the dataset layer
  for its bars.
- A rule that declares `required_factors()` needs factor values at each bar, and trader cannot
  compute them without the factor layer. trader v1 refuses such a rule. No shipped rule declares
  any.
- Where the rebalance calendar is anchored is still open. `rebalance_mask` stays a backtest-layer
  helper, counted from the start of the window.

## Amended by quantlab ADR 0019 (trader #34)

The one-bar decision is no longer two constructor methods. quantlab moved the assembly of a
bar's inputs into one module, `quantlab/portfolio/decision_inputs.py`: `DecisionInputs.context(t,
predictions, current_weights)` builds the context (from the rule's last `history_bars` raw
prices, so staleness no longer depends on the history given), `rebalances(t)` answers the
schedule, and the rule keeps `decide`. `build_context` and `construct_panel` are deleted, and
`load_constructor` with `quantlab/portfolio/prediction_panel.py` is replaced by
`DecisionInputs.from_run(run_dir)`; `PredictionPanel` lives in `quantlab.portfolio.base`. trader's
run-time imports from quantlab are therefore `quantlab.portfolio.base` and
`quantlab.portfolio.decision_inputs` for the decision. The rebalance anchor is settled: the
prediction panel's first bar, the last bar never rebalancing.
