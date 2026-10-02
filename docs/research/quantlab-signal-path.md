# quantlab's batch signal path for one decision date

Research for [#4](https://github.com/ZhaorongDai/quantlab-trader/issues/4). Question: what does
quantlab need to produce one bar's target weights in batch, given data up to a decision date D and
an already-trained model (load mode)? Which pieces are callable for a single date today, which are
private, and how much warm-up do the example strategies need? Streaming is out of scope.

Sources: quantlab source at commit `f16dd10` (paths relative to `~/projects/quantlab`), plus
`~/projects/kun_nt_factor` (a local folder with no commits). The answers come from reading the
code. Two facts were also checked by running them (see "Smoke check"). This document reports
facts and proposes no API.

## TL;DR

- `BaseBacktester.run()` cannot produce the weights for one date. With `start_date == end_date
  == D` the window has one bar, and `rebalance_mask` never rebalances on the last bar of a
  window (`quantlab/backtest/selection.py:47-49`). The row comes back all-NaN ("hold").
  `run()` also simulates, needs fill prices for the window and writes a run directory.
- The single-date path today means calling public pieces yourself: rebuild the configured objects
  (`load_backtester_from_config`), `model.load(checkpoint)`, `model.predict_window(D, D)`, build a
  `PortfolioContext` by hand, then `constructor.construct(context)`. Each of those is public.
- Two parts of that path are private: computing the context's `returns` and `staleness` from
  prices (`PortfolioConstructor._check_prices`), and the locked-position row check
  (`_checked_row`). `_Book`, which supplies `current_weights` inside the backtest, is also
  private. It models a simulated book that starts flat at the window's first bar. In live trading
  that input is the position snapshot, so a live caller would not use `_Book` at all.
- Warm-up for the three example strategies:

  | example | model warm-up | factor warm-up (`warmup_bars`) | rule lookback |
  |---|---|---|---|
  | `sp500_xgb` / `market_xgb` | 0 | 400 | 0 |
  | `sp500_xgb_mvo` | 0 | 400 | 126 |

  Factors dominate in every case: 400 bars of OHLCV before D, plus D itself. The 400 is a value
  set in the example config, not a computed minimum for Alpha101/Alpha158.

## 1. The `run()` path in load mode, step by step

`run()` (`quantlab/base/backtest.py:802`) checks that a model is configured and that the label
delays match the engine (`_check_label_delays`, `:1868`). Then, inside `_run_window`
(`:1310`), it runs these steps:

1. **Prepare the model** (`_prepare_model`, `:1771`). In load mode this calls
   `_load_model_checkpoint` (`:1938`), which checks that the file exists, reads the `config.json`
   sidecar for the training dates (`_read_checkpoint_config`, `:1968`), then calls
   `model.check_checkpoint(path)` and `model.load(path)`. The training dates are used only for
   the in-sample/out-of-sample split of the metrics. They do not affect any weight.
2. **Predict the window** (`_predict_window`, `:2006`). This records fingerprints, then calls
   `model.predict_window(start, end)`.
   - `BaseModel.predict_window` (`quantlab/base/model.py:790`) calls `_collect_all_features(start,
     end)` and `predict_panel(features)`, then cuts the result to `start..end`.
   - `_collect_all_features` (`model.py:497`) asks each factor for its panel through
     `_request_panel` (`model.py:436`): `factor.read(first, end)` under
     `factor_data_strategy="read"`, `factor.compute(first, end)` under `"cal"`. Here `first` is
     `start` moved back by the **model's** `warmup_bars` (`_feature_start`, `model.py:523`).
   - The model's warm-up is 0 for row models (`model.py:484`), including XGBoost. For torch heads
     it is `window_bars - 1` (`quantlab/model/torch_model.py:443-451`).
   - `Factor.compute(start, end)` (`quantlab/base/factor.py:336`) adds its **own**
     `warmup_bars`, counted in bars on the dataset calendar (`_input_range`/`_warm_start`,
     `factor.py:389-430`, using `MarketDataset.bar_before`, `quantlab/base/data.py:750`). It then
     trims the result back to `start..end`.
   - So under `"cal"`, the request for one date D reads `bar_before(D, factor.warmup_bars)..D`.
     Under `"read"` it reads only `D` from the factor store, and the store must already contain
     D.
   - Ensembles (`ModelEnsemble`/`SeedEnsemble`) combine member predictions in their own
     `predict_window` (`quantlab/model/ensemble.py:451`).
   - Labels are never read in load mode. The label objects matter only for their metadata:
     `label_delays`, and `span_bars()` for the mean-variance `bind`.
3. **Load prices** (`_load_prices`, `backtest.py:2020`): `price_dataset.panel(start, end)` with
   the fill column (`adjOpen`) and the valuation column (`adjClose`)
   (`US_EQUITY_MARKET`, `quantlab/backtest/predefined/us_equity.py:24-29`). Predictions are then
   reindexed onto the price axes (`backtest.py:1741`).
4. **Generate signals** (`USEquityCrossectionSelectStockVectorBt._generate_signals`,
   `us_equity.py:104`):
   - `tradable = price_dataset.tradable_bars(prices, "adjOpen")` (`:122`). By default this is
     "has a real adjOpen at the bar" (`data.py:1953`); nothing in the repo overrides it.
   - `mask = rebalance_mask(n_bars, rebalance_periods)` (`:123`).
   - Raw fill and valuation prices from `constructor.lookback_bars` bars before the window
     (`_price_history`, `:154`).
   - `delisted = price_dataset.delisting_bars(...)` (`data.py:1985`).
   - The panels of `constructor.required_factors()` over the window (`_required_factor_panels`,
     `:138`). These are computed with `Factor.compute`, so each one warms itself up.
   - Then `constructor.construct_panel(...)` (`:125`).
5. **`construct_panel`** (`quantlab/base/portfolio.py:800`):
   - Builds a `_PriceHistory` (`_check_prices`, `:993`). Returns are computed from the
     forward-filled valuation price, `ffill / ffill.shift(1) - 1` (`:1042`), and staleness is
     the number of bars since the last finite valuation (`:1043`).
   - Starts a `_Book` (`:218`) at 1.0 cash and no shares.
   - Loops `construct(context)` over the rebalance bars. Before each one,
     `book.weights_at(position)` (`:922`) replays the earlier targets at the next bar's fill
     price, applying rejected orders, delisting settlements and cash-capped buys in the order
     sells first, then buys from smallest to largest (`:257-283`).
   - Each returned row goes through `_checked_row` (`:1077`), then `book.queue(position, row)`
     (`:972`).
   - A `PortfolioConstructionError` holds the bar, and the bar is listed in
     `attrs["failed_bars"]`.
6. Then the contract check (`_assert_weights_contract`, `backtest.py:2291`), the vectorbt
   simulation, the metrics, and the run directory. None of this is needed to decide weights.

**The rebalance phase belongs to the window.** `rebalance_mask` marks bar 0 of the window and every
`k`-th bar after it, and never the last bar (`selection.py:13-50`). A live caller rebalancing
every 5 bars has to choose its own anchor. To match a backtest, it would count bars from that
backtest's `start_date` on the price dataset's calendar.

## 2. `PortfolioContext`: each field and where it comes from

`PortfolioContext` is a public frozen dataclass (`portfolio.py:66-131`). It can be built by hand;
the docstring example does exactly that.

| field | type | in the backtest it comes from | for one live date D it would come from |
|---|---|---|---|
| `timestamp` | `pd.Timestamp` | the prediction bar (`portfolio.py:929`) | D |
| `predictions` | `Dataset` on `symbol`, one var per label | `predictions.isel(timestamp=t)` (`:930`), from `model.predict_window` reindexed onto price symbols (`backtest.py:1741`) | `model.predict_window(D, D)` squeezed to `symbol` (public) |
| `tradable` | bool `DataArray` on `symbol` | `price_dataset.tradable_bars(prices, "adjOpen")` (`us_equity.py:122`; `data.py:1953`) | same call on a one-bar panel at D (public) |
| `current_weights` | float `DataArray` on `symbol` | `_Book.weights_at` (`portfolio.py:240`), the simulated book valued at this bar's ffilled `adjClose`; 0.0 before the first rebalance | the position snapshot. `_Book` is private, and in this use it would start flat. |
| `returns` | `(timestamp, symbol)` window of `lookback_bars` one-bar returns ending at D, or `None` | `_PriceHistory.returns` sliced to `lookback` rows (`portfolio.py:922-927`, formula `:1042`) | ffilled `adjClose` over `bar_before(D, lookback)..D`. The formula is in a private method, so a caller has to reimplement it. |
| `staleness` | float `DataArray` on `symbol`, or `None` | `_PriceHistory.staleness[position]` (`:1043`, `:941-946`) | bars since the last finite `adjClose`. Same as `returns`: private, reimplemented by the caller. |
| `factors` | `Dataset` on `symbol`, or `None` | `required_factors()` panels (`us_equity.py:138-152`), sliced at the bar (`:940`) | `factor.compute(D, D)` for each required factor (public). No shipped rule declares any. |

`context.locked` (`portfolio.py:145`) is derived as "held and not tradable". `context.symbols` is
the `tradable` axis (`:134`).

## 3. What is public and what is private

The public pieces, which are enough to decide one date:

- `quantlab.utils.module.load_backtester_from_config(config)` (`quantlab/utils/module.py:215`).
  Given a run's `config.json`, it rebuilds the price dataset, the model (through its class's
  `from_config`), the constructor and the benchmark exactly as configured (`:336-371`).
  Construction calls `constructor.bind(model)` (`us_equity.py:102`), so a `MeanVarianceOptimizer`
  already has its span. Fields: `backtester.config.model`, `.constructor`, `.price_dataset`.
- Predictor members (ADR 0008, `backtest.py:87-178`): `check_checkpoint(path)`,
  `load(path)`, `predict_window(start, end)`, `labels`, `label_scales`.
  - `BaseModel.load` is at `model.py:987` and `check_checkpoint` at `model.py:1056`.
  - Ensembles load an `ensemble.json` (`ensemble.py:1022`).
  - `BaseModel.predict_panel(features)` (`model.py:1212`) is public on `BaseModel` but is not a
    protocol member.
- Factors: `read(start, end)` (`factor.py:268`), `compute(start, end)` (`:336`), `build(start,
  end)` (`:448`) and `extend(end)` (`:483`). `extend` appends bars after the recorded range,
  computed with warm-up, so it is the daily path for `factor_data_strategy="read"`.
- Datasets: `panel(start, end, symbols)` (`data.py:680`), `bar_before(date, n)` (`:750`),
  `tradable_bars` (`:1953`), `delisting_bars` (`:1985`).
- Portfolio: `PortfolioContext`, `PortfolioConstructor.bind`, `construct(context)`
  (`portfolio.py:774`), `lookback_bars` (`:715`), `required_factors()` (`:728`),
  `construct_panel(...)` (`:800`), and `PortfolioConstructionError` (`:50`, which means "hold
  this bar").
- `quantlab.backtest.selection.rebalance_mask` (`selection.py:13`).

The private pieces, which a single-date caller would have to reimplement or skip:

- `_check_prices` / `_PriceHistory` (`portfolio.py:199`, `:993`): the `returns` and `staleness`
  formulas.
- `_Book` (`:218`): the simulated holdings. A live caller does not need it, because it has the
  snapshot instead.
- `_checked_row` (`:1077`): refuses a row that mixes NaN and finite values, a row that moves a
  locked position, and weight on a symbol that is neither tradable nor held. `construct` alone
  does not run this check.
- In the backtester: `_prepare_model` / `_load_model_checkpoint` (a sidecar read plus
  check-then-load, `backtest.py:1771`, `:1938`), `_predict_window` (fingerprinting),
  `_generate_signals` / `_price_history` / `_required_factor_panels` (`us_equity.py:104-195`),
  `_assert_weights_contract` (`backtest.py:2291`; checks gross exposure ≤ 1).
- In the model: `_collect_all_features` / `_feature_start` (`model.py:497`, `:523`). ADR 0008
  keeps these private on purpose, and `predict_window` is the public way to reach them.

Three ways to get the weights for D today, and what each gives:

1. **`run()` over `[D, D]`**: an all-NaN row, because the last bar never rebalances.
   Over `[D, D+1]` it needs D+1 prices and simulates a fill. Neither is usable before D+1's open.
2. **`construct_panel` on a one-bar prediction panel** with `rebalance=[True]` and a price
   history from `bar_before(D, lookback)`. This is public and computes `returns`/`staleness` and
   the row check for you. But `_Book` starts flat at the first price row, so `current_weights`
   is 0 for every symbol. Two consequences:
   - The top-n rule's locked-position handling and the optimiser's turnover penalty see an empty
     book.
   - The only way to get non-zero holdings is to pass predictions back to the strategy's
     inception, and that replays quantlab's simulated book rather than the account.
3. **Hand-built `PortfolioContext` → `construct(context)`**: the only path that can take the real
   holdings. It reimplements the `returns`/`staleness` formulas (`portfolio.py:1036-1043`) and
   the `_checked_row` checks.

`quantlab.api` (ADR 0011) exposes `compute_factors`, `forward_returns`, `analyze_factors` and
`backtest(weights)` (`quantlab/api/__init__.py:50`, `:137`, `:240`, `:375`). None of these
predicts or decides.

## 4. Warm-up and data needed per decision, by example

The settings shared by all three examples: `HORIZON = 5`; Alpha101Stock and Alpha158Stock with
`warmup_bars=400`; `factor_data_strategy="read"`; `model_mode="load"`.

Factor `warmup_bars=400` is at `sp500_xgb.py:88,93`, `market_xgb.py:88,93` and
`sp500_xgb_mvo.py:100,105`. The `"read"` strategy is at `sp500_xgb.py:159`, `market_xgb.py:130`
and `sp500_xgb_mvo.py:186`.

The factor inputs are `adjOpen, adjHigh, adjLow, adjClose, adjVolume`. Both alpha sets wrap every
output in `CrossSectionalZScore`, which is computed per bar and adds no history
(`quantlab/factor/predefined/alpha101.py:99-104`, `alpha158.py:152-161`).

| | `sp500_xgb` | `market_xgb` | `sp500_xgb_mvo` |
|---|---|---|---|
| model | `XGBoostRegressor` (warm-up 0) | same | `ModelEnsemble` of two XGB (`ret_5`, `vol_5`), warm-up 0 |
| rule | `TopN(long_only, top_n=50)` (`:201`) | `TopN(long_only, top_n=100)` (`:172`) | `MeanVariance` + `LedoitWolf(lookback_bars=126)` (`:235`), `candidate_top_k=200`, `weight_cap=0.02` |
| `rebalance_periods` | 5 (`:200`) | 5 (`:171`) | `HORIZON` = 5 (`:253`) |
| rule `lookback_bars` | 0 | 0 | 126 (`mean_variance.py:349-357`) |
| factor history for D | 400 bars before D, plus D | same | same |
| price history for D | D only (`adjOpen` for tradability) | D only | `adjClose` from `bar_before(D, 126)` to D, plus `adjOpen` at D |
| factor dataset | `prices.zarr` (every bar of every past or present member) | `wrds_crsp_market_1d.zarr` directly | `prices.zarr` |
| price dataset (tradability) | `members.zarr`: NaN outside membership, so tradable means member with an open at D | market store | `members.zarr` |

Notes:

- **Ledoit-Wolf coverage** (`ledoit_wolf.py:126-139`). A symbol is covered only if all 126 returns
  are finite (126 returns need 127 priced bars, since the first return is NaN), the price is not
  flat, and staleness ≤ `max_stale_bars` (default 5, `quantlab/base/config.py:838-855`).
  - In this example the price dataset is `members.zarr`, so a stock becomes buyable only after
    127 bars of membership (comment at `sp500_xgb_mvo.py:231-235`).
  - A held symbol that is not covered is closed and reported as `closed_without_risk`.
- **Under `"read"`** a decision at D needs the factor stores extended to D first
  (`factor.extend(D)`). That reads `bar_before(first new bar, 400)..D` of OHLCV, so the 400-bar
  requirement moves into store maintenance. The prediction itself then reads only row D from the
  stores. Under `"cal"` the prediction computes the 400-bar warm-up itself.
- Either way, each `compute` call compiles the KunQuant graph again. `_lib` is set to `None`
  after every run (`quantlab/factor/kunquant.py:373-392`).
- **sp500 derived stores.** `prices.zarr` and `members.zarr` are rewritten in full by
  `prepare_stores()` from the CRSP index store and the membership store (`sp500_xgb.py:107-136`,
  `mode="w"`). A daily decision therefore also needs index membership at D.
- **Torch examples** (`*_gats`, `*_master`, `*_realmlp`; not in scope) add `window_bars - 1` model
  warm-up on top of the factor warm-up.
- **What data D needs**, summed up:
  - OHLCV (split- and dividend-adjusted) for the universe, 401 bars ending at D.
  - Membership at D (sp500).
  - A finite `adjOpen` at D, for tradability.
  - For mean-variance, `adjClose` for 127 bars.
  - The model checkpoint (`.joblib`, or `ensemble.json` plus member checkpoints).
  - The current holdings as weights on the same symbol axis. CRSP symbols are PERMNO integers.
  - CRSP is not a live feed, which trader's ADR 0001 already notes.
- **Timing.** `tradable` at D uses D's open, and the features use bars up to D's close. Labels have
  `delay=1` (`quantlab/label/predefined/fret.py:130`), which matches the engine's
  `fill_delay_bars = 1` (`quantlab/backtest/engine_vectorbt.py:103`). So a weight decided at D
  after the close fills at D+1's open. A "before the open of D+1" run has the same information.

## 5. `~/projects/kun_nt_factor`

It is a two-file prototype (`kun_factor_indicator.py`, `example_backtest.py`) with no git commits.
Its parts and whether a batch path could reuse them:

- **Compiled-library cache.** `KunFactorEngine._compile` passes `tempdir=cache_dir,
  keep_files=True` to `cfake.compileit` (`kun_factor_indicator.py:162-189`). The README says this
  persists the compiled library so a restart loads it instead of compiling.
  - This is relevant to the batch path, because quantlab recompiles Alpha101/Alpha158 on every
    `compute` (see section 4).
  - Not verified here: whether `compileit` actually skips recompiling when the files already
    exist.
- **NautilusTrader scaffolding** (`example_backtest.py:59-72`, `:177-215`): a `make_equity` that
  builds `Equity` instruments, plus `BacktestEngine` / `add_venue(CASH, NETTING)` /
  `add_instrument` / `add_data` setup. This is generic NT boilerplate that trader could copy. It
  does not touch quantlab.
- **Not reusable for this question:**
  - The streaming `StreamContext` engine and `KunFactorActor`. The actor aligns per-symbol bars by
    `ts_event` before each cross-sectional step (`:380-457`).
  - `KunFactorData` message publishing.
  - These are all streaming. The factors there are KunQuant's predefined `Alpha101` graphs, not
    quantlab's `Alpha101Stock`/`Alpha158Stock` classes, which have no `CrossSectionalZScore`.
  - `amount` is approximated as `close * volume`.

## Smoke check

Run on 2026-10-01 against quantlab `f16dd10` with `uv run python`:

- `rebalance_mask(1, 5)` returns `[False]`, and `rebalance_mask(6, 5)` returns
  `[True, False, False, False, False, False]`. This confirms that a one-bar window never
  rebalances.
- `TopNConstructor(top_n=2).construct(...)` on a hand-built `PortfolioContext` returned
  `[0.25, 0.25, 0.0, 0.5]`. The context had 4 symbols, one held at 0.5 and not tradable (locked),
  and one without a prediction. The locked weight is kept, and the remaining 0.5 is split over the
  top two. This confirms that the single-date `construct` path works with the holdings passed in
  from outside.
