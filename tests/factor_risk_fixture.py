"""A real factor risk model over a fixture price store (ADR 0011).

``factor_risk_model`` builds quantlab's ``Use4RiskModel`` (two styles, two
industries, a country) through its public risk API: its exposures factor
``FixtureExposures`` passes through variables of the price store
(``risk_model_variables``), and its regression and estimate stores are built
under a temporary directory. No forecast is stubbed.

This module is kept apart from ``quantlab_run_fixture`` because the trader
process rebuilds ``FixtureExposures`` by class path from a run's recipe: it
imports quantlab's factor and risk layers only, never the backtest layer
the run fixture needs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.dataset.base import MarketDataset
from quantlab.factor.base import Factor
from quantlab.factor.config import BaseFactorConfig
from quantlab.risk.config import Use4RiskConfig
from quantlab.risk.predefined.use4 import Use4RiskModel


#: The fixture risk model's styles; ``FixtureExposures`` also outputs the
#: industry code and the estimation-universe flag.
RISK_STYLES = ("style_a", "style_b")
EXPOSURES = (*RISK_STYLES, "industry", "estu")


class FixtureExposures(Factor):
    """A factor risk model's exposures: the price store's exposure variables, unchanged.

    Stands in for ``BarraStyle``, as a factor the risk model's config names
    and the trader process rebuilds by class path from the run's recipe
    (ADR 0011).
    """

    config_cls = BaseFactorConfig

    def _get_factor_names(self):
        return EXPOSURES

    def _compute_panel(self, inputs):
        return inputs[list(EXPOSURES)]


def risk_model_variables(
    permnos: Sequence[int], n_bars: int, seed: int
) -> dict[str, dict[int, list[float]]]:
    """Return the store variables a factor risk model reads, per PERMNO over ``n_bars``.

    Random style exposures, the first half of ``permnos`` in industry 1 and
    the rest in industry 2, everyone in the estimation universe, a constant
    market cap and a zero risk-free rate.
    """
    rng = np.random.default_rng(seed)
    half = len(permnos) // 2
    return {
        **{s: {p: list(rng.normal(size=n_bars)) for p in permnos} for s in RISK_STYLES},
        "industry": {p: [1.0 if k < half else 2.0] * n_bars for k, p in enumerate(permnos)},
        "estu": {p: [1.0] * n_bars for p in permnos},
        "marketcap": {p: [float(rng.lognormal(20, 1))] * n_bars for p in permnos},
        "risk_free": {p: [0.0] * n_bars for p in permnos},
    }


def factor_risk_model(
    root: Path,
    dataset: MarketDataset,
    bars: pd.DatetimeIndex,
    first_bar: int,
    exposure_data_strategy: str = "read",
) -> Use4RiskModel:
    """Build a ``Use4RiskModel`` on ``dataset`` and its stores under ``root``.

    The exposures are ``FixtureExposures`` of ``dataset`` (whose store
    carries ``risk_model_variables``); under ``"read"`` their store is built
    over every bar. The regression store covers ``bars[1:]`` and the
    estimate store the run's window, from ``bars[first_bar]``. The model's
    volatility regime adjustment is warm only after some 18 regression bars
    (before that its factor covariance is huge and a fully invested book
    infeasible), so ``first_bar`` should be 20 or more.
    """
    root = Path(root)
    exposures = FixtureExposures(
        BaseFactorConfig(
            warmup_bars=0, dataset=dataset, file_path=str(root / "exposures.zarr")
        )
    )
    first, last = (b.strftime("%Y-%m-%d") for b in (bars[0], bars[-1]))
    if exposure_data_strategy == "read":
        exposures.build(first, last)
    model = Use4RiskModel(
        Use4RiskConfig(
            exposures=exposures,
            dataset=dataset,
            exposure_data_strategy=exposure_data_strategy,
            style_names=RISK_STYLES,
            industry_name="industry",
            industries=(1, 2),
            estu_name="estu",
            min_industry_members=2,
            regression_path=str(root / "regression.zarr"),
            estimate_path=str(root / "estimate.zarr"),
            volatility_half_life=5.0,
            volatility_window=10,
            correlation_half_life=10.0,
            correlation_window=15,
            specific_half_life=5.0,
            specific_window=10,
            min_observations=5,
            vra_half_life=5.0,
            vra_window=10,
        )
    )
    model.regression.build(bars[1].strftime("%Y-%m-%d"), last)
    model.estimate.build(bars[first_bar].strftime("%Y-%m-%d"), last)
    return model
