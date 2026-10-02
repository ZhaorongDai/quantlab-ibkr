# IBKR adapter capabilities for US-equity daily trading

Research for issue #2. Question: what does NautilusTrader 1.231's Interactive Brokers adapter
support for a daily US-equity portfolio strategy whose signals are computed in batch after the
close / before the open and fill at the next open, run against an IBKR **paper** account?

Date: 2026-10-01.

## Sources and how to read the citations

- **Installed package (authoritative for behaviour):** `nautilus_trader` 1.231.0 in
  `~/projects/quantlab/.venv/lib/python3.13/site-packages/nautilus_trader/` (version from the
  wheel's `METADATA`). Paths below are relative to that directory, e.g.
  `adapters/interactive_brokers/execution.py:1219`. Line numbers are from 1.231.0.
- **Nautilus docs:** `docs/integrations/ib.md` in the local clone `~/projects/nautilus_trader`
  (commit `64dd3d4c1b`, 2026-06-09, `pyproject.toml` says **1.229.0**, so the clone is two minor
  versions *behind* the installed 1.231). Doc claims were checked against the installed source;
  where they disagree, the source wins. Online copy:
  <https://nautilustrader.io/docs/latest/integrations/ib>.
- **IBKR docs:** TWS API reference pages under
  <https://www.interactivebrokers.com/docs/tws-api/doc/> and the order-type pages, fetched
  2026-10-01. Some IBKR Campus pages could not be fetched (Cloudflare/JS); claims that rest on a
  search snippet or an IBKR forum reply rather than a reference page are marked **(weak source)**.
- **Unverified** means no primary source was found or the behaviour needs a live paper session
  to confirm. Nothing here was run against IB Gateway.

## Summary

| Need | Answer | Confidence |
|---|---|---|
| Daily bars, historical | Yes: `1-DAY-LAST` (and BID/ASK/MID) through `RequestBars` or `HistoricInteractiveBrokersClient.request_bars` | source |
| Adjusted bars | **No dividend adjustment.** Adapter only sends `whatToShow` TRADES/BID/ASK/MIDPOINT; TRADES is split-adjusted, not dividend-adjusted. `ADJUSTED_LAST` is not reachable through the adapter | source + IBKR docs |
| Daily bars, live subscription | Works (`reqHistoricalData(keepUpToDate=True)`), but a completed day bar is only emitted when the next day's bar starts or ~24h after the last update, so it does **not** fire at the close | source (inferred from code) |
| Delisted symbols in history | Not available from IB ("data for securities which are no longer trading") | IBKR docs |
| Contract identification | `TICKER.PRIMARY_EXCHANGE` (e.g. `AAPL.NASDAQ`, `SPY.ARCA`) → `STK`, `exchange=SMART`, `primaryExchange=<venue>`; conId kept on the contract; full-universe load not supported | source |
| Paper connection | TWS 7497 / Gateway 4002 (or `DockerizedIBGateway`, `trading_mode="paper"`), `account_id="DU..."` or `TWS_ACCOUNT` | docs + source |
| Next-open fill | MOO = `MarketOrder` + `TimeInForce.AT_THE_OPEN` → IB `MKT`/`OPG`; LOO = `LimitOrder` + `AT_THE_OPEN` → `LMT`/`OPG` | source + IBKR docs |
| Close fill | MOC = `MarketOrder` + `AT_THE_CLOSE` → IB `MOC`/`DAY`; LOC = `LimitOrder` + `AT_THE_CLOSE` → `LOC`/`DAY` | source + IBKR docs |
| Fractional shares | **No.** `Equity` hard-codes `size_precision=0`; the risk engine rejects quantities with more precision. IBKR also said (2024) fractional orders are not supported via the TWS API | source + weak source |
| Account / positions | Account state from `reqAccountSummary` (NetLiquidation, TotalCashValue, FullAvailableFunds, margin); positions from `reqPositions`; open orders, fills, mass-status reconciliation implemented; OMS NETTING, account type MARGIN | source |
| Market data subscriptions | Default `market_data_type=REALTIME`; `DELAYED`/`DELAYED_FROZEN` selectable. IBKR says historical data is for market-data subscribers; paper can share the live user's subscriptions | config source + IBKR docs (partly weak) |

## 1. Daily bar data

### Historical

- Two paths reach `reqHistoricalData(keepUpToDate=False)`:
  - inside a node: `Strategy.request_bars(...)` → `InteractiveBrokersDataClient._request_bars`
    (`adapters/interactive_brokers/data.py:606-665`), which refuses non-time bars and chunks the
    range via `get_historical_bars_chunked` (`data.py:668`);
  - offline: `HistoricInteractiveBrokersClient.request_bars(bar_specifications=["1-DAY-LAST"], ...)`
    (`adapters/interactive_brokers/historical/client.py:185-303`), meant for downloading to a
    `ParquetDataCatalog` (`ib.md` "Historical data and backtesting").
- Supported bar sizes map in `parsing/data.py:67` (`bar_spec_to_bar_size`): day bars only with
  step 1 (`"1 day"`), plus `1 week`; anything else raises.
- Price type → `whatToShow` in `parsing/data.py:46-58` (`what_to_show`): `LAST→TRADES`,
  `BID→BID`, `ASK→ASK`, `MID→MIDPOINT` (and `AGGTRADES` for PAXOS crypto). There is **no path to
  `ADJUSTED_LAST`** (grep of the adapter finds no `ADJUSTED`).
- IBKR semantics of those values:
  - "TRADES data is adjusted for splits, but not dividends."
    (<https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bar-what-to-show/trades.md>)
  - "ADJUSTED_LAST data is adjusted for splits and dividends."
    (<https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bar-what-to-show/adjusted-last.md>)
  - So adapter daily bars are **split-adjusted, dividend-unadjusted**. A fully unadjusted series
    (raw prints, as CRSP `PRC`) is also not available from the adapter.
- Timestamps for day bars (`client/market_data.py:1630-1686`): IB returns `YYYYMMDD`; the adapter
  parses it as **midnight UTC of the trade date** for `ts_event` and sets
  `ts_init = ts_event + 1 day - 1ns` (so 23:59:59.999999999 UTC of the same date). Not the NYSE
  close time. `formatDate=2` (UTC) is used for requests (`client/market_data.py:624-651`); the
  Nautilus docs also ask that TWS/Gateway be configured to send UTC timestamps (`ib.md` "Getting
  started" warning).
- One-off historical requests do **not** drop an in-progress bar (`process_historical_data`,
  `client/market_data.py:1106-1119`, has no completeness check), so a `1-DAY` request made during
  the session would include today's partial bar. Inferred from code; request after the close to
  avoid it.
- IBKR limits that matter:
  - Max duration for `1 day` bars: 365 D / 52 W / 12 M / 68 Y per request
    (<https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bars/max-duration-per-bar-size.md>);
    the adapter chunks long ranges itself.
  - Unavailable: "Data for securities which are no longer trading", and history before a move to a
    new exchange, "also applied to contract which specifies `SMART`"
    (<https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-data-limitations/unavailable-historical-data.md>).
    IB history is therefore **survivorship-biased** and cannot replace CRSP for research or
    backtests; at most it can supply the latest day for live decisions.
  - Pacing: API requests are capped at market-data-lines / 2 per second, 50/s for the default 100
    lines (<https://www.interactivebrokers.com/docs/tws-api/doc/pacing-limitations/introduction.md>).
    Nautilus docs warn pacing violations can disable the API session for minutes (`ib.md`
    "Data limitations"). Separate historical-data pacing rules exist for bars ≤30s only.

### Live subscription

- `_subscribe_bars` (`data.py:265-288`): a 5-second bar uses `reqRealTimeBars`; every other size
  (including `1-DAY`) uses `subscribe_historical_bars`, i.e. `reqHistoricalData(keepUpToDate=True)`
  with ~300 bars of backfill (`client/market_data.py:470-570`). IBKR: keepUpToDate updates every
  ~4-6 s and is only available for Trades/Midpoint/Bid/Ask
  (<https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/historical-bars/keep-up-to-date.md>).
- Emission rule with the default `handle_revised_bars=False` (`client/market_data.py:1355-1420`,
  `1292-1353`): a bar is published only when a **newer** bar date arrives (the previous bar is then
  complete), or when a timeout of `bar duration + 1 s` after the latest update expires. For a day
  bar that is ~24 h. Consequence (inferred from code, unverified live): a subscribed `1-DAY-LAST`
  bar for date D reaches `on_bar` around the next session's first update, not at D's close.
  `handle_revised_bars=True` (a `LiveDataClientConfig` field, `live/config.py:237`) instead
  publishes every intraday revision.
- For an after-close batch, the simpler pattern is a clock timer after the close plus a one-off
  `request_bars` for `1-DAY`, or not using IB for data at all (see Open risks).

## 2. Instruments and contract identification

- Symbology (`InteractiveBrokersInstrumentProviderConfig.symbology_method`, default
  `IB_SIMPLIFIED`, `config.py:187`). For stocks the instrument id is
  `{localSymbol}.{primaryExchange}` with spaces → `-` (`parsing/instruments.py:1343-1345`), e.g.
  `AAPL.NASDAQ`, `SPY.ARCA`, `BF-B.NYSE` (`ib.md` "Symbology"). `IB_RAW` gives
  `AAPL=STK.SMART`. `convert_exchange_to_mic_venue=True` turns venues into MICs (`NASDAQ→XNAS`,
  `NYSE→XNYS`); `symbol_to_mic_venue` overrides per symbol (`config.py:193-194`).
- Id → contract (`parsing/instruments.py:1564-1574`, `_decode_stock_contract`): `AAPL.NASDAQ`
  becomes `IBContract(secType="STK", exchange="SMART", primaryExchange="NASDAQ",
  localSymbol="AAPL")`; the provider then calls `reqContractDetails` to qualify it. So orders are
  SMART-routed and the venue part of the id only disambiguates the listing. A per-order
  `params={"exchange": "..."}` overrides routing (`ib.md` "Order params";
  `execution.py:1287-1291`).
- conId: every loaded instrument stores the qualified contract (with `conId`) in
  `instrument.info["contract"]` and `provider.contract_id_to_instrument_id`
  (`providers.py:816-820`). Loading by conId works through `load_contracts` with an `IBContract`
  (`providers.py:357-361`); building an `IBContract(conId=...)` for stocks specifically was not
  tested (unverified).
- Symbol list → instruments: there is **no full-universe load** ("Interactive Brokers does not
  support loading the full IB instrument universe with `load_all=True`", `ib.md` "Loading
  instruments"); `load_all_async` only loads `load_ids` + `load_contracts`
  (`providers.py:285-295`). So the daily symbol list must be turned into
  `TICKER.PRIMARY_EXCHANGE` ids (or `IBContract`s) and loaded at start, or requested before use.
  Each id costs one `reqContractDetails` round-trip (`providers.py:329-370`); a few hundred names
  are therefore a startup cost bounded by pacing. `cache_validity_days` / `pickle_path` cache
  contract details across days (`config.py:136-141`).
- Equity parse (`parsing/instruments.py:377-401`): price increment from `minTick`, `lot_size=100`,
  and an ISIN pulled from `secIdList`; a stock with **no ISIN raises** `ValueError("No ISIN
  found")`, which the provider logs and **skips** (`providers.py:796-804`). Whether IB returns an
  ISIN for every US common stock on a paper account is unverified.
- quantlab keys stocks by CRSP PERMNO and its tickers come from CRSP; a PERMNO → (ticker, primary
  exchange) mapping must be produced on the quantlab side. Share classes need IB's localSymbol
  form (`BF B` → `BF-B`).

## 3. Paper connection

- Ports (`ib.md` "Default ports"): TWS paper **7497** (live 7496), IB Gateway paper **4002**
  (live 4001). Configure `ibg_host`/`ibg_port`/`ibg_client_id` on both
  `InteractiveBrokersDataClientConfig` and `InteractiveBrokersExecClientConfig`
  (`config.py:244-252`, `297-305`).
- Account: `account_id="DU..."` or env `TWS_ACCOUNT`; the factory refuses to build without one and
  builds `AccountId(f"{issuer}-{account}")`, issuer defaulting to `IB`
  (`factories.py:316-325`).
- Docker: `DockerizedIBGatewayConfig(username, password, trading_mode="paper",
  read_only_api=True, timeout=300, container_image="ghcr.io/gnzsnz/ib-gateway:stable")`
  (`config.py:63-69`); credentials may come from `TWS_USERNAME` / `TWS_PASSWORD` (`ib.md`
  "Environment variables"). **`read_only_api` defaults to `True` and must be set `False` to place
  orders** (`ib.md` "Establish connection to Dockerized IB Gateway").
- Client ids must be unique per connection (`ib.md` "Client ID conflicts"); data and exec clients
  can share or use separate ids (`ib.md` "Multi-client configuration").
- Reconnection: `IB_MAX_CONNECTION_ATTEMPTS` env var, `connection_timeout` default 300 s (`ib.md`
  "Connection management"). On reconnect, keepUpToDate bar subscriptions are re-issued and backfill
  from the disconnection point (`client/market_data.py:541-557`).
- IBKR: paper "Trades ... will not actually execute on any exchange", prices "determined by real
  market" conditions; trial (unfunded) accounts are not supported by any API, but a created,
  unfunded account's paper user may request delayed data (IBKR Campus lesson and staff reply,
  <https://www.interactivebrokers.com/campus/trading-lessons/request-paper-trading-account/>,
  **weak source** for the reply).

## 4. Order types for next-open and close fills

Mapping in `adapters/interactive_brokers/parsing/execution.py:37-70` and its use in
`execution.py:1219-1224` (`_transform_order_to_ib_order`):

| Intent | Nautilus order | IB `orderType` / `tif` |
|---|---|---|
| Market-on-open (MOO) | `MarketOrder`, `TimeInForce.AT_THE_OPEN` | `MKT` / `OPG` |
| Limit-on-open (LOO) | `LimitOrder`, `TimeInForce.AT_THE_OPEN` | `LMT` / `OPG` |
| Market-on-close (MOC) | `MarketOrder`, `TimeInForce.AT_THE_CLOSE` | `MOC` / `DAY` |
| Limit-on-close (LOC) | `LimitOrder`, `TimeInForce.AT_THE_CLOSE` | `LOC` / `DAY` |
| Plain market at/after open | `MarketOrder`, `TimeInForce.DAY` | `MKT` / `DAY` |

- This matches IBKR's own definitions: MOO is `orderType="MKT"`, `tif="OPG"`
  (<https://www.interactivebrokers.com/docs/general/order-types/market-orders/market-on-open>);
  MOC is `orderType="MOC"`
  (<https://www.interactivebrokers.com/docs/general/order-types/market-orders/market-on-close>).
- Nautilus core allows it: `MarketOrder` documents MOO/MOC as `AT_THE_OPEN`/`AT_THE_CLOSE` and only
  forbids `GTD` (`model/orders/market.pyx:56-57, 137`).
- IBKR cut-offs (<https://www.interactivebrokers.com/en/trading/orders/moo.php>): "Nasdaq MOO (and
  LOO) orders must be submitted prior to 09:28 ET"; late LOO accepted until 9:29:30 with
  repricing; "cancelation or modification of On-Open orders will not be permitted after the
  imbalance information has been published (9:25 a.m. ET)". "IB may simulate market orders on
  exchanges." NYSE-listed cut-offs and MOC cut-offs were not confirmed (unverified).
- What an OPG order submitted after the cut-off does through the adapter is unverified; IB status
  `Inactive` maps to `REJECTED` (`parsing/execution.py:100`).
- Whether the **paper** simulator fills MOO at the official opening auction price, and handles
  OPG/MOC like production, is unverified; IBKR only says paper prices follow real market
  conditions.
- Batch submit/modify/cancel, order lists, OCA, brackets and IB conditions are supported (`ib.md`
  "Batch operations", "Contingent orders"); `post_only` and quote-quantity orders on non-inverse
  instruments raise (`execution.py:1208-1214`).

## 5. Fractional shares

- Nautilus side, definitive: `Equity.__init__` passes `size_precision=0` and
  `size_increment=1` ("No fractional units", `model/instruments/equity.pyx:115-117`), and the
  IB equity parser does not override it (`parsing/instruments.py:377-401`). The risk engine
  rejects any order whose quantity precision exceeds the instrument's
  (`risk/engine.pyx:1055-1057`). So **orders must be whole shares**; weights → shares needs
  rounding (down) on the trader side, and fractional positions held in the account would not
  round-trip into Nautilus quantities cleanly (unverified how reconciliation treats them).
- IB side: an IBKR staff reply (Oct 2024) states "Fractional trading is supported via FIX/CTCI but
  not via API at this time"
  (<https://www.interactivebrokers.com/campus/trading-lessons/fractional-shares/>, **weak source**,
  may be outdated).
- `cashQty` is used only for inverse (crypto) BUY orders (`execution.py:1229-1243`); there is no
  notional/cash-amount stock order path.

## 6. Account, position and portfolio queries

- Exec client is registered with `OmsType.NETTING`, `AccountType.MARGIN`, no base currency
  (multi-currency) (`execution.py:206-217`).
- Account state: `reqAccountSummary(group "All", all tags)` (`client/account.py:66-72`); once
  NetLiquidation, TotalCashValue, FullAvailableFunds, FullInitMarginReq, FullMaintMarginReq arrive
  for a currency, the client emits an `AccountState` with `total=NetLiquidation`,
  `free=FullAvailableFunds`, margins, and `info["TotalCashValue"]`; the full raw summary is cached
  under key `accountSummary:<account>` (`execution.py:230-236`, `1665-1712`). `QueryAccount`
  triggers a fresh pull (`execution.py:992-1010`). Note: `total` is net liquidation, not cash.
- Positions: `generate_position_status_reports` via `reqPositions` (`execution.py:830`), so the
  node reconciles external positions at start; also `generate_order_status_reports`
  (`reqOpenOrders`, or `reqAllOpenOrders` with `fetch_all_open_orders=True`) and
  `generate_fill_reports` (`reqExecutions`) (`execution.py:532`, `660`; `client/order.py:115-141`).
  In a strategy these surface as the usual `self.cache.positions()`, `self.portfolio.*` and
  `self.cache.account(...)` (`ib.md` multi-account examples).
- `reqExecutions` only returns the current day's executions by IB's design (not checked in IBKR
  docs; unverified), so fill history for the position snapshot should be persisted by trader.

## 7. Market-data subscription requirements

- `InteractiveBrokersDataClientConfig.market_data_type` defaults to `REALTIME`
  (`config.py:248`) and is sent once with `reqMarketDataType` on connect (`data.py:144`,
  `client/market_data.py:106-118`). Options: `REALTIME`, `FROZEN`, `DELAYED`, `DELAYED_FROZEN`.
  IBKR: requesting delayed still returns real-time where subscribed
  (<https://www.interactivebrokers.com/docs/tws-api/doc/market-data-delayed/market-data-type-behavior.md>).
- IBKR: "Historical Market data is available for Interactive Brokers market data subscribers"
  (<https://www.interactivebrokers.com/docs/tws-api/doc/market-data-historical/introduction.md>).
  Whether daily `TRADES` bars for US stocks need a paid US equity bundle, or work on
  `DELAYED`, was not confirmed (unverified); the Nautilus docs recommend `DELAYED_FROZEN` for
  testing without subscriptions (`ib.md` "Market data permissions").
- Paper accounts can share the live user's subscriptions; one paper account per live user
  (IBKR market-data-subscriptions page, seen only as a search snippet, **weak source**:
  <https://www.interactivebrokers.com/campus/ibkr-api-page/market-data-subscriptions/>).
- Default 100 market-data lines; they bound streaming subscriptions and API request pacing
  (pacing page above). A daily strategy that never streams quotes does not consume lines for
  long.
- Error 354 "Requested market data is not subscribed" is the symptom of missing permissions
  (`ib.md` "Error codes").

## Open risks for downstream decisions

1. **Data source for the live decision date.** IB daily bars are dividend-unadjusted,
   split-adjusted, survivorship-biased and ticker-keyed. Mixing them with CRSP history inside one
   factor window changes returns on ex-dividend days and around splits. Decide whether the live
   day comes from IB (and how it is spliced onto CRSP) or from another vendor.
2. **Day-bar emission timing.** Subscribed `1-DAY` bars arrive roughly a day late; use a timer +
   one-off request, or keep data out of the node.
3. **Whole shares only.** Weight → share rounding and the resulting residual cash/drift must be
   handled by trader and reflected in the position snapshot.
4. **MOO cut-off and paper auction fidelity.** Orders must be in before ~09:25-09:28 ET; paper
   fills at the open are simulated and unverified against the official open.
5. **Symbology.** PERMNO → `TICKER.PRIMARY_EXCHANGE`; stocks without an ISIN in IB's
   `secIdList` are silently skipped; ticker changes need a per-day mapping.
6. **Startup cost.** One contract-details round-trip per symbol; use `pickle_path` /
   `cache_validity_days` and stay within pacing.
7. **Docs vs installed version.** The local clone documents 1.229; all behaviour above is from the
   installed 1.231 source. Re-check after any upgrade.
