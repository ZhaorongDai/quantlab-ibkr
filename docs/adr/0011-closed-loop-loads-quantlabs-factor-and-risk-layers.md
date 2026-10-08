---
status: accepted
date: 2026-10-08
---

# Closed loop loads quantlab's factor and risk layers

A rule that declares factors or a factor risk model (`declared_inputs().factors`,
`declared_inputs().risk_model`) used to be refused in closed loop (ADR 0005), because trader
could not reach them without `quantlab.factor`, which ADR 0008 forbade. A Barra-optimised
mean-variance rule (USE4 factor risk model, style-exposure bounds) declares exactly that: its risk
model's exposures are a `BarraStyle` factor, rebuilt by class path from the run's `config.json`
even when the exposures are only read from their store.

trader now loads `quantlab.factor` and `quantlab.risk` at run time, by class path from the run's
config, and takes the rule's factors and risk forecast from quantlab's `DecisionInputs` unchanged.
The refusal is removed.

## Considered

- **A store-only exposure source in quantlab** (a risk-layer component that reads an exposure
  store by path, so the risk model's config names no factor). Keeps the factor layer and KunQuant
  out of the trader process, but changes quantlab's risk-model config for the sake of one
  consumer's import list.
- **trader reads the zarr stores itself.** Duplicates quantlab's risk forecast; the parity
  ladder would then compare two implementations instead of one decision.

## Consequences

- ADR 0008's allowlist stays an ast scan of trader's own source: trader still imports no
  `quantlab.factor` or `quantlab.risk` module by name; they arrive only through the run's config.
  The run-time forbidden list drops `quantlab.factor` and `KunQuant`; `quantlab.model`,
  `quantlab.label`, `quantlab.backtest`, `torch`, `xgboost` and `vectorbt` stay forbidden.
- The live process will carry KunQuant and the factor stores' readers. The trader process is no
  longer the light one ADR 0008 described.
- ADR 0005's "trader v1 refuses such a rule" no longer holds.
