# quantlab-ibkr

Event-driven backtesting and live trading through Interactive Brokers (IBKR paper first) of
[quantlab](../quantlab) strategies on NautilusTrader, driven directly by the backtest configuration
quantlab writes. IBKR is the only broker: the backtest venue simulates it and the live venue is
IBKR's (ADR 0010; formerly quantlab-trader).

## Status

`quantlab-ibkr backtest --quantlab-run DIR` replays a quantlab run on NautilusTrader and writes a
trader run directory: closed loop by default (quantlab's portfolio rule decides every rebalance bar
on the holdings the account has), or `--loop open` to execute the run's rebalance table
(`weights.zarr`) as it is.

`quantlab-ibkr parity --quantlab-run DIR [--output-dir DIR] [--fee-model fraction|ibkr_fixed]`
writes the run's parity report (`parity.json` + `parity.zarr`, ADR 0007): the ladder from
quantlab's vectorbt run to trader's open loop one convention at a time, its end checks (the
command exits with 2 when one fails) and, for a run with a prediction panel, closed versus open
loop with its Decision recheck (quantlab's rule run again on each closed-loop decision's context,
actual holdings included, which must give the decided weights bit for bit).

The design is in `docs/adr/`; the v1 spec is issue #18.

## Live trading (IBKR paper)

`quantlab-ibkr live` trades a quantlab run's strategy on an IBKR account as a daily batch
(spec #47). The quantlab run directory stays the strategy's recipe: the rule, its decision
inputs and the rebalance cadence are rebuilt from it, and each day's predictions come from
the run's **live prediction store**, which quantlab's `scripts/live/predict_day.py` appends
after the vendor update (quantlab `docs/live.md`). The trader process loads no model code.

Each weekday has two steps, each a short run:

- `quantlab-ibkr live decide CONFIG.json` (before the open). t is the last bar of the run's
  price store. It connects to the Gateway, reconciles the account (positions and cash are
  IBKR's), runs the strategy's decision cycle once on t, and submits the orders as
  market-on-open (`MKT`, `OPG`), sells first. The cycle marks the account at t's raw close
  every day. It decides t only on a rebalance bar of the run's cadence (counted on past the
  backtest's end) that has a prediction row. Any other day holds and submits nothing.
- `quantlab-ibkr live record CONFIG.json` (after the open, the same day). It reads IBKR's
  executions, open orders and completed orders, adds the fills of the orders still working,
  ends the orders IBKR canceled or expired with IBKR's reason, and rewrites the live run
  directory. Then it runs the Decision recheck on that directory. The command exits with 2
  when the recheck differs.
- `quantlab-ibkr live recheck CONFIG.json` runs the Decision recheck alone, and writes
  `decision_recheck.json`. Each decided bar is decided again from the live store's row, with
  the recorded current weights. The result must equal the decision bit for bit.

The config is a `TraderConfig` JSON file whose venue is an `IbkrVenueConfig`. Flags given
with the file override its venue fields: `--dry-run`, `--host`, `--port`, `--client-id`,
`--account-id`, `--order-deadline`, `--prediction-store`, `--live-dir`. Without a file,
`--quantlab-run`, `--prediction-store` and `--live-dir` are required.

```json
{
  "quantlab_run": "/data/quantlab/runs/ibkr_barra_closed_loop/real/backtest/<run>",
  "venue": {
    "name": "quantlab_ibkr.venue.ibkr.venue.IbkrVenueConfig",
    "prediction_store": "/data/quantlab/live/sp500_xgb_mvo/live_predictions.zarr",
    "live_dir": "/data/quantlab/live/sp500_xgb_mvo/ibkr",
    "host": "127.0.0.1",
    "port": 4002,
    "client_id": 1,
    "order_deadline": "09:20"
  }
}
```

```bash
cd ~/projects/quantlab-ibkr
export TWS_ACCOUNT=DU1234567            # the paper account
.venv/bin/quantlab-ibkr live decide /data/quantlab/live/sp500_xgb_mvo/live.json --dry-run
.venv/bin/quantlab-ibkr live decide /data/quantlab/live/sp500_xgb_mvo/live.json   # before 09:20 ET
.venv/bin/quantlab-ibkr live record /data/quantlab/live/sp500_xgb_mvo/live.json   # after 09:30 ET
```

Each run prints t, the equity, the decision or the hold reason, and the orders. Exit status
is 0 on success and 1 when the run is refused (reason on stderr). `record` and `recheck`
exit with 2 when the Decision recheck differs.

**Safety.**

- Credentials come only from the environment. `TWS_ACCOUNT` names the account unless the
  config sets `account_id`, and the Gateway holds the login.
- Only a paper account (`DU...`) is traded unless the config sets `allow_live`. The check
  runs before any connection.
- `--dry-run` decides and prints the orders. It submits nothing and records nothing.
  Add `--force-decide` to decide t even when it is not a rebalance bar, which exercises the
  rule, the contract lookup and the orders on any day; it is refused without `--dry-run`.

**Deadline.** Orders must reach IBKR before the opening auction's cut-off (about 09:28 ET).
A day that decides refuses to start after `order_deadline` (default 09:20 New York time),
and no order leaves after it. The deadline falls on the next weekday after t. An exchange
holiday is not skipped. If decide runs on a holiday morning after 09:20, it is refused,
although the auction is a day later. Run it again the next trading morning.

**Connections.** The trading node uses `client_id`. The contract lookup uses
`contract_client_id` (default `client_id + 1`). The order reports use `reports_client_id`
(default `client_id + 2`). No other API client may use these ids.

**The live run directory** (`live_dir`) holds the files a closed-loop backtest's run
directory holds: `decisions.zarr` (with `current_weight`), `holdings.zarr`, `equity.zarr`,
`orders.zarr`, `events.json`, `metrics.json` and `report.html`. The steps append to them one
day at a time, and the files are rewritten from `journal.json` each time. The directory also
holds `con_ids.json` (the conId cache) and `decision_recheck.json`.

- Equity and holdings of bar t are written by decide. It marks the account at t's close, so
  each row is a day's decide step.
- Fills at the open of t+1 are written by record, and join the metrics with t+1's row.
- Live events add `live_hold`, `excluded_symbols`, `position_gap`, `adopted_orders` and
  `timed_out`.

**Re-running is safe.**

- `decide` on a bar already recorded does nothing.
- An order already working at IBKR for t is adopted, never sent again. This covers a run
  that submitted and then failed before recording. Every order's IBKR reference is its
  nautilus client order id, tagged with the decision date (`O-...-IBKR-20261007-n`).
- `record` adds each IBKR execution once.
- IBKR reports executions of the current day only. Run `record` on the day of the open.
  An order IBKR reports nothing about is left as `submitted` and listed in the output.

A day skipped entirely leaves no equity row for its bar. The next decide marks the next bar.

`scripts/live_daily.sh` runs a live day on the server. Debian's cron has no `CRON_TZ`, so cron calls
it every hour and the script checks the New York time itself (daylight saving needs no edit):

```
0 * * * * $HOME/projects/quantlab-ibkr/scripts/live_daily.sh
```

- 06:00 ET: Sharadar's `update.py --rebuild-dropped`, then quantlab's daily prediction job; the
  pair is retried every 15 minutes until the job has t's row or 08:30 ET passes (a vendor table a
  day late, such as SP500 membership, is waited for); from 08:35 ET, `live decide`. A morning
  without t's row by 08:30 holds the day: no order is sent on stale data.
- 10:00 ET: `live record` (IBKR's fills, then the Decision recheck).
- Weekends are skipped; on a market holiday decide finds t already decided and does nothing.
- Credentials come from owner-only files: `~/.config/quantlab/sharadar.env` (`SHARADAR_API_KEY`)
  and `~/.config/quantlab/ibkr.env` (`TWS_ACCOUNT`). Logs go to `<live dir>/logs/<date>.log`.
- `scripts/live_daily.sh morning` or `record` runs one step now.

```
uv sync
uv run pytest
```
