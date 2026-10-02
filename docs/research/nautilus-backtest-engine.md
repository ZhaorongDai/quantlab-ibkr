# Nautilus backtest engine for daily cross-sectional portfolios

Research for issue #3 (map #1). Question: how does NautilusTrader 1.231's backtest engine
run a daily cross-sectional portfolio of hundreds of US equities, and how close can it get to
quantlab's vectorbt backtest semantics?

## Sources and method

- Installed package `nautilus_trader 1.231.0` in quantlab's venv
  (`~/projects/quantlab/.venv/lib/python3.13/site-packages/nautilus_trader/`). The wheel ships
  the Cython `.pyx` sources, so line numbers below are from that version:
  - `backtest/engine.pyx` (`BacktestEngine`, `SimulatedExchange`, `OrderMatchingEngine`)
  - `backtest/models/fill.pyx`, `backtest/models/fee.pyx`
  - `model/instruments/equity.pyx`, `risk/engine.pyx`, `accounting/accounts/cash.pyx`
- Local clone `~/projects/nautilus_trader` at `develop` 64dd3d4c1b (2026-06-09). Its
  `pyproject.toml` says **1.229.0**, two minors behind the installed wheel. Docs quoted from the
  clone are marked "(docs, 1.229 clone)". Each one used here was checked against the 1.231 source,
  or run against 1.231.
  Online equivalent: <https://nautilustrader.io/docs/latest/concepts/backtesting>.
- quantlab's semantics: `quantlab/backtest/engine_vectorbt.py`, `docs/adr/0012-*`,
  `docs/adr/0014-decisions-use-information-at-t-execution-simulates-the-market.md`.
- Experiments: small scripts run with `~/projects/quantlab/.venv/bin/python` (Python 3.13,
  macOS arm64) on synthetic data. Each claim marked **verified** was observed in one of these runs.
  The scripts are summarised inline and were not committed.

## The vectorbt semantics to match

From `quantlab/backtest/engine_vectorbt.py` (`_simulate`, `_execution_plan`):

- `Portfolio.from_orders(size_type="targetpercent", direction="both", group_by=True,
  cash_sharing=True, call_seq="auto")`. Weights are shifted one bar, so a weight formed at the
  close of t fills at t+1's `open` (D-05). The target percentage is measured against portfolio
  value **at the fill price** of t+1. The holdings are fractional shares.
- `fees` and `slippage` are fractions of the trade's price or notional.
- A NaN weight keeps the holding. A finite target whose **raw** open at t+1 is NaN is a rejected
  order: it expires, the holding is kept, and it is recorded.
- A symbol delisted on bar b is settled on bar b+1 into cash at its **last valuation** (close),
  with **no fee and no slippage** (ADR 0014).
- No borrow cost on shorts.

## Findings

### 1. Loading many instruments and daily bars

- **Low-level `BacktestEngine`** (`add_venue`, `add_instrument`, `add_data`, `run`) is enough at
  this scale. `add_data(sort=True)` re-sorts the whole stream on every call, so for many
  instruments you should either call `add_data(..., sort=False)` once per instrument followed by
  `sort_data()` once, or call `add_data` a single time with a combined list (docs, 1.229 clone:
  `docs/concepts/backtesting.md` "Loading large datasets efficiently").
  `add_data_iterator(generator)` and manual `run(streaming=True)` / `clear_data()` / `end()` cover
  streams that do not fit in memory.
- **High-level `BacktestNode`** + `BacktestRunConfig` + `BacktestDataConfig` reads from a
  `ParquetDataCatalog` (Nautilus's own Parquet layout), with optional `chunk_size`
  (`backtest/config.py:414`). It suits parameter sweeps and data larger than RAM. A daily panel
  of 500 x 10y is about 1.3M bars (plus 1.3M open ticks, see §2), which fits in memory easily
  (§7). A catalog would be one more storage format next to quantlab's zarr stores, so the
  low-level engine fed straight from quantlab's xarray panels is the simpler path.
- **Bar timestamps:** with `bar_execution=True` (the default) Nautilus requires `ts_init` to be
  the bar's **close** time (docs, 1.229 clone: "Bar-based execution"). quantlab's daily timestamps
  are dates, so the converter must stamp each daily bar at 16:00 America/New_York in UTC.
- **Instrument:** `Equity` hard-codes `size_precision=0` and `size_increment=1`
  (`model/instruments/equity.pyx:115-117`). Only whole shares are possible. Fees come from
  `maker_fee`/`taker_fee` on the instrument.
- **Gotcha (verified):** under pandas 3, `pd.bdate_range(...).asi8` is in **microseconds**
  unless you call `.as_unit("ns")`. Nautilus expects nanoseconds, and with µs values every event
  lands in January 1970.

### 2. Filling at t+1 open (D-05)

Nautilus has **no native next-bar-open mode**. The clone's docs say so explicitly ("A native
'next-bar-open' execution mode is not provided", `backtesting.md` "Order submission timing"),
and the 1.231 matching engine **rejects** `TimeInForce.AT_THE_OPEN`/`AT_THE_CLOSE` market and
limit orders with "is not currently supported" (`backtest/engine.pyx:5333-5339`, `5431-5437`).
`process_bar` turns each bar into four trades, O, H, L, C, all stamped at the bar's `ts_init`
(`engine.pyx:4850-4950`).

Measured with two symbols on daily bars, a market BUY formed in `on_bar(t)`, 1.231:

| Setup | Fill price | Verified |
|---|---|---|
| Submit in `on_bar(t)`, no latency | close of t | yes |
| Submit in `on_bar(t)`, `LatencyModel(1h)`, bars only | close of t+1 | yes |
| Add an open `TradeTick` at 09:30 of each day, submit in `on_bar(t)` with `LatencyModel(1h)` | open of t+1 for **some** instruments; others filled at close of t (see below) | yes |
| Add an open `TradeTick` at 09:30 of each day; a `clock.set_time_alert` at `open_ts + 1ns` submits all orders, with no latency model | **open of t+1 for every instrument** | yes (334 of 334 fills matched the synthetic open) |

**Recommended pattern (verified):** for every symbol and day, emit
`TradeTick(price=open, ts=09:30 ET)` and `Bar(O,H,L,C, ts=16:00 ET)`. The strategy schedules one
time alert per trading day at `open_ts + 1ns`, after every open tick at that timestamp has
updated its book. In that callback it reads the precomputed weight row for the fill day and
submits market orders, which fill at the open immediately. At that moment
`Portfolio.unrealized_pnls` marks positions at the last trade price, which is today's open. So
target-percent sizing is measured against NAV at the fill price, as in vectorbt.

The latency variant is a trap. Queued commands settle at the first data point at or after their
arrival time, *before* the remaining instruments' ticks with the same timestamp are applied. In
the test, the short leg filled at the previous close (51) rather than the open (52). With 500
symbols that would be most of the book. Do not rely on `LatencyModel` for open fills.

### 3. Fees and slippage

- **Fees:** `MakerTakerFeeModel` charges `notional * instrument.taker_fee` (or `maker_fee`)
  (`backtest/models/fee.pyx:67-112`). This is the same fractional fee as vectorbt's `fees`.
  Commission is rounded to the currency precision (cents): `10 * 100.25 * 0.001 = 1.0025` was
  booked as `1.00 USD` (verified). `FixedFeeModel` and `PerContractFeeModel` also exist, and
  `FeeModel.get_commission(order, fill_qty, fill_px, instrument)` can be subclassed in Python.
- **Slippage:** the built-in L1/bar slippage is **tick-based**. `FillModel(prob_slippage=p)`
  moves a fill by one tick with probability p, and `OneTickSlippageFillModel` always moves it by
  one tick (`fill.pyx`; docs "Slippage and spread handling"). None of the built-ins applies a
  fractional slippage. A **Python subclass** of `FillModel` overriding
  `get_orderbook_for_fill_simulation` to return a one-level L2 book at
  `best_ask*(1+s)` / `best_bid*(1-s)` does work. With `s=0.0025` a BUY at open 100.00 filled at
  **100.25** (verified). The fee is then charged on the slipped notional, as in vectorbt. The
  price is rounded to the instrument's tick, which is $0.01, so slippage on low-priced stocks is
  quantised.

### 4. Target weights to orders

Nautilus has no target-percent order type. The strategy computes, per symbol,
`delta = floor(w * NAV / open) - net_position` and submits sells first, then buys (the equivalent
of vectorbt's `call_seq="auto"`). On a MARGIN account, NAV is
`account.balance_total(USD) + portfolio.unrealized_pnls(venue)[USD]`, because a margin account's
balance does not deduct the notional of purchases (verified: after buying, `total` stayed near
the starting cash with the notional shown as `locked`). On a CASH account it is
`balance_total + net exposures`.

**Integer shares** are the systematic gap. At $10M and 100 names, each holding is about $100k, so
rounding error is under 0.1% per name. At small capital it is larger.

### 5. Cash vs margin account

- **CASH** account: short sales are **rejected**: "SHORT SELLING not permitted on a CASH account"
  (verified). Use it only for long-only books.
- **MARGIN** account with `default_leverage=Decimal(1)` and `margin_init=margin_maint=1` on each
  `Equity` accepted a short and a long together (verified). `allow_cash_borrowing` exists for
  cash accounts (`add_venue` docstring) but was not tested.
- The `RiskEngine` denies orders whose notional exceeds the free balance
  (`risk/engine.pyx:952`, `971`, `1004`, `1029`). Sizing at 99% gross plus fees and rounding can
  trip it. vectorbt instead caps buys at available cash. The perf run below used
  `RiskEngineConfig(bypass=True)`. For a parity run, either keep the risk engine on and size
  slightly under, or bypass it and check cash yourself. No borrow fee is modelled in either
  engine.

### 6. Halts, missing bars, delisting

- **Missing bar or halt: no automatic rejection (verified).** A BUY submitted for a symbol that
  has no tick and no bar on the fill day filled at its **stale** last price, the previous close.
  `InstrumentStatus` does not help: `OrderMatchingEngine.process_status` only moves
  `CLOSED -> OPEN` (`engine.pyx:4817-4830`), and nothing else in the matching engine reads
  `market_status`. ADR 0014's rejected order must therefore be implemented in the strategy: skip
  any symbol whose `cache.trade_tick(id).ts_event` is not today's open timestamp, and record it.
  This is exactly what vectorbt does from the raw-open NaN test.
- **Delisting settlement (verified):** emitting
  `InstrumentClose(id, price, InstrumentCloseType.CONTRACT_EXPIRED, ts=open of b+1)` for an
  `Equity` makes `check_instrument_expiration` cancel the symbol's open orders and close its
  position with a reduce-only market order tagged `EXPIRATION_<venue>_CLOSE`
  (`engine.pyx:4832-4848`, `5934-5981`). With
  `add_venue(settlement_prices={id: last_valuation})`, the position closes at that price through
  `apply_fills`, bypassing the fill model, so no slippage applies (verified: a settlement price of
  40.00 was used verbatim). That matches ADR 0014's "cash at last valuation". **But the fee model
  still charges commission** on the settlement (verified: 0.40 USD at 0.1%). For parity, use a
  `FeeModel` subclass that returns zero when `order.tags` contains `EXPIRATION_*_CLOSE` (not
  tested). `settlement_prices` is one static price per instrument, which is fine because a
  symbol delists once.

### 7. Run time at ~500 symbols x 10 years

Synthetic benchmark (`perf.py`): N symbols, D business days, one open `TradeTick` and one daily
`Bar` per symbol-day, and the daily-alert rebalancer from §2. It holds an equal-weight top-100
book with heavy turnover (about 100 orders a day), uses a MARGIN account, `MakerTakerFeeModel`,
risk engine bypassed, log level ERROR, on an Apple Silicon Mac, Nautilus 1.231:

| N x D | data objects | build objects | add+sort | `run()` | orders | max RSS |
|---|---|---|---|---|---|---|
| 50 x 250 | 25k | 0.1 s | <0.1 s | 0.9 s | 2.7k | 0.25 GiB |
| 500 x 250 | 250k | 0.7 s | 0.1 s | 19.7 s | 26k | 0.75 GiB |
| 500 x 2520 | 2.52M | 7.3 s | 1.3 s | **223.8 s** | 255k | 3.6 GiB (`time -l`: 4.2 GB RSS, 6.1 GB peak footprint) |

Whole process wall time for the 10-year run: 317 s, including interpreter start-up and teardown.

The run is dominated by per-order event processing and by the Python rebalance loop over 500
symbols a day. Data volume costs little. A lower-turnover strategy (weekly rebalance, or a
band around target weights) would be proportionally faster. vectorbt does the same panel in
seconds, so Nautilus is a parity and realism check, not a sweep engine.

### 8. Strategy vs Actor

`Strategy` subclasses `Actor` and adds order management (docs, 1.229 clone:
`concepts/strategies.md`, `concepts/actors.md`). Because signals are batch-computed by quantlab,
one **Strategy** that holds the precomputed `[T, S]` weight panel and submits orders is enough.
An **Actor** is only worth adding for non-trading duties, for example recording NAV or rejected
orders, or publishing signals on the message bus if signal generation later moves inside
Nautilus.

## Known gaps vs vectorbt

| Topic | vectorbt (quantlab) | Nautilus 1.231 | Closable? |
|---|---|---|---|
| Fill at t+1 open | price column shift | no native mode; open tick + `open_ts+1ns` alert | yes (verified) |
| Target percent | `size_type="targetpercent"` at fill-bar NAV | computed in strategy at open NAV | yes (verified) |
| Share size | fractional | `Equity` whole shares only | no (rounding error; small at $10M) |
| Fees | fraction of notional | `MakerTakerFeeModel`, rounded to cents | yes |
| Slippage | fraction of price | custom `FillModel` subclass, rounded to tick | yes (verified) |
| Buys beyond cash | capped by cash | risk engine denies the order | partly (size under, or bypass) |
| Rejected order (no open) | automatic from NaN open | stale fill unless the strategy skips it | yes, in strategy |
| Delisting settlement | last valuation, no fee/slippage | `InstrumentClose` + `settlement_prices`; fee charged | yes with a fee-model tweak (untested) |
| Shorts | `direction="both"` | MARGIN account required | yes (verified) |
| Speed | seconds | minutes at 500 x 10y with daily turnover (§7) | n/a |
