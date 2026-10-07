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
loop.

The design is in `docs/adr/`; the v1 spec is issue #18.

```
uv sync
uv run pytest
```
