# quantlab-trader

Event-driven backtesting and live (IBKR paper) trading of [quantlab](../quantlab) strategies on
NautilusTrader, driven directly by the backtest configuration quantlab writes.

## Status

Open-loop replay works end to end: `quantlab-trader backtest --quantlab-run DIR --loop open`
executes a quantlab run's rebalance table (`weights.zarr`) on NautilusTrader and writes a trader
run directory. The design is in `docs/adr/`; the v1 spec is issue #18.

```
uv sync
uv run pytest
```
