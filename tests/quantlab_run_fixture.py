"""Synthetic, model-free quantlab runs built through quantlab's public API.

``build_quantlab_run`` writes a small CRSP-shaped price store (the
``CrspStockDataset`` panel layout: integer PERMNOs on ``symbol``, raw
``open``/``high``/``low``/``close``, the adjusted ``adj*`` group, ``divCash``,
``splitFactor``, ``cumfacpr``, ``cumfacshr``) and backtests a given rebalance
table on it with quantlab's ``WeightsVectorBt.run_weights``, which writes a
real run directory (read through quantlab's ``BacktestRun``). Nothing here
imports quantlab's ``tests/``.

``build_constructor_run`` builds a closed-loop run the way ADR 0007 has
the parity fixtures built: a model-free ``PredictionPanel``, quantlab's
``DecisionInputs.weights`` over it (the inputs quantlab's cross-section
backtester assembles: tradability, the rebalance schedule, the rule's
``history_bars`` price window, the delisting marks), ``run_weights`` of a
``CrossSectionBacktestConfig``
carrying the rule (so the run's recipe records it), and the panel written
where a run with a model keeps its prediction panel (``PREDICTION_PANEL``).

``build_factor_risk_run`` is a ``build_constructor_run`` whose rule is
mean-variance on a real factor risk model (ADR 0011): quantlab's
``Use4RiskModel`` over two styles and two industries
(``tests/factor_risk_fixture.py``), its exposure variables written into the
price store, its regression and estimate stores built in the temporary
directory through quantlab's public risk API and read by a
``FactorRiskStoreEstimator``, with bounds on the book's style exposures.
Nothing about the forecast is stubbed.

``build_quantlab_run(member=...)`` instead runs on a membership-masked
derived store, the way quantlab's ``sp500_*`` examples build
``members.zarr``: the CRSP panel's ``adj*``, ``close`` and ``volume``
columns, NaN where the PERMNO is not a member, read by a ``StockDataset``.
trader refuses such a run (ADR 0006).

Later tickets extend the fixture by passing their own raw prices and
variables (``variables=``: a split, a dividend, a halt, a delisting) and
their own table (``weights=``); ``adjusted_scale`` keeps the adjusted group a
fixed multiple of the raw one so a raw/adjusted mix-up shows in the numbers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.predefined.weights import WeightsVectorBt
from quantlab.backtest.config import CrossSectionBacktestConfig, WeightsBacktestConfig
from quantlab.dataset.config import CrspDatasetConfig, DatasetConfig
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig, MeanVarianceConfig
from quantlab.portfolio.predefined.factor_risk import FactorRiskStoreEstimator
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.runs.prediction_panel import LabelSpec, PredictionPanel
from quantlab.tracking.base import Tracker
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.stock import StockDataset
from quantlab.portfolio.decision_inputs import DecisionInputs
from tests.factor_risk_fixture import factor_risk_model, risk_model_variables

#: Where a quantlab run with a model keeps its prediction panel; the fixture
#: writes one into a model-free run (the one place outside quantlab that names
#: a run file, for a test stand-in only).
PREDICTION_PANEL = "predictions.zarr"

#: The adjusted group is the raw group times this, so a raw/adjusted mix-up
#: changes every number a test checks.
ADJUSTED_SCALE = 0.5


def write_crsp_store(
    path: Path,
    bars: pd.DatetimeIndex,
    open_: Mapping[int, Sequence[float]],
    close: Mapping[int, Sequence[float]],
    *,
    variables: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
    drop_variables: Sequence[str] = (),
    adjusted_scale: float = ADJUSTED_SCALE,
) -> None:
    """Write a CRSP-shaped daily panel to the Zarr store at ``path``.

    ``open_`` and ``close`` are the raw prices per PERMNO (NaN where the
    market had none). ``high``/``low`` are the max/min of the two, the
    adjusted group is the raw one times ``adjusted_scale``, corporate-action
    fields are neutral (``splitFactor`` 1, ``divCash`` 0, factors 1) unless
    ``variables`` overrides them, and ``drop_variables`` leaves names out.
    """
    permnos = sorted(close)
    raw_open = np.array([open_[p] for p in permnos], dtype=float).T
    raw_close = np.array([close[p] for p in permnos], dtype=float).T
    priced = np.isfinite(raw_close).astype(float)
    priced[priced == 0] = np.nan
    panel = {
        "open": raw_open,
        "high": np.fmax(raw_open, raw_close),
        "low": np.fmin(raw_open, raw_close),
        "close": raw_close,
        "volume": 1e6 * priced,
        "divCash": 0.0 * priced,
        "splitFactor": 1.0 * priced,
        "cumfacpr": 1.0 * priced,
        "cumfacshr": 1.0 * priced,
    }
    for name in ("open", "high", "low", "close"):
        panel["adj" + name.capitalize()] = panel[name] * adjusted_scale
    panel["adjVolume"] = panel["volume"]
    for name, values in (variables or {}).items():
        panel[name] = np.array([values[p] for p in permnos], dtype=float).T
    dataset = xr.Dataset(
        {
            name: (("timestamp", "symbol"), values)
            for name, values in panel.items()
            if name not in drop_variables
        },
        coords={"timestamp": bars, "symbol": np.array(permnos, dtype=np.int64)},
    )
    dataset.to_zarr(path, mode="w")


def build_quantlab_run(
    root: Path,
    bars: pd.DatetimeIndex,
    open_: Mapping[int, Sequence[float]],
    close: Mapping[int, Sequence[float]],
    weights: Mapping[int, Sequence[float]],
    *,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
    slippage: float = 0.0,
    fill_price_column: str = "adjOpen",
    valuation_price_column: str = "adjClose",
    variables: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
    drop_variables: Sequence[str] = (),
    benchmark: tuple[Mapping[int, Sequence[float]], Mapping[int, Sequence[float]]] | None = None,
    tracker: Tracker | None = None,
    member: Mapping[int, Sequence[bool]] | None = None,
) -> Path:
    """Write a CRSP-shaped store and a quantlab weights run on it; return the run dir.

    ``weights`` is the rebalance table per PERMNO over ``bars`` (NaN keeps
    the holding); every PERMNO of ``close`` must be in it. ``benchmark`` is
    the raw ``(open, close)`` of one PERMNO, written to its own store and
    bought and held by the run; ``tracker`` is the run's tracker. With
    ``member`` (per PERMNO and bar), the run's price dataset is instead a
    membership-masked ``StockDataset`` derived from the store (quantlab's
    examples' ``members.zarr``), which trader refuses (ADR 0006).
    """
    root = Path(root)
    dataset = _crsp_dataset(
        root, bars, open_, close, variables=variables, drop_variables=drop_variables
    )
    if member is not None:
        dataset = _members_dataset(root, dataset, member)
    start, end = (b.strftime("%Y-%m-%d") for b in (bars[0], bars[-1]))
    permnos = sorted(close)
    table = xr.DataArray(
        np.array([weights[p] for p in permnos], dtype=float).T,
        dims=("timestamp", "symbol"),
        coords={"timestamp": bars, "symbol": np.array(permnos, dtype=np.int64)},
    )
    backtester = WeightsVectorBt(
        WeightsBacktestConfig(
            price_dataset=dataset,
            start_date=start,
            end_date=end,
            output_dir=str(root / "runs"),
            rebalance_periods=1,
            fees=fees,
            slippage=slippage,
            init_cash=init_cash,
            fill_price_column=fill_price_column,
            valuation_price_column=valuation_price_column,
            trading_days_per_year=252,
            session_minutes_per_day=390,
            benchmark_dataset=None
            if benchmark is None
            else _crsp_dataset(root / "benchmark", bars, *benchmark),
            **({} if tracker is None else {"tracker": tracker}),
        )
    )
    return Path(backtester.run_weights(table).run_dir)


#: The columns quantlab's ``sp500_*`` examples keep in their derived stores
#: (less ``ret``, which the fixture store does not have).
MEMBERS_COLUMNS = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume", "close", "volume")


def _members_dataset(
    root: Path, crsp: CrspStockDataset, member: Mapping[int, Sequence[bool]]
) -> StockDataset:
    """Derive a members-style store from ``crsp`` and return the ``StockDataset`` reading it.

    The store is the panel's ``MEMBERS_COLUMNS``, NaN where ``member`` is
    False, as quantlab's ``sp500_*`` examples write ``members.zarr``.
    """
    start, end = crsp.config.start_date, crsp.config.end_date
    prices = crsp.panel(start, end)[list(MEMBERS_COLUMNS)].load()
    permnos = prices["symbol"].values
    mask = xr.DataArray(
        np.array([member[p] for p in permnos], dtype=bool).T,
        dims=("timestamp", "symbol"),
        coords={"timestamp": prices["timestamp"], "symbol": permnos},
    )
    path = root / "members.zarr"
    prices.where(mask).to_zarr(path, mode="w")
    return StockDataset(
        DatasetConfig(
            zarr_file_path=str(path),
            raw_data_dir_path=str(root / "raw"),
            market="us_equity",
            frequency="1d",
            start_date=start,
            end_date=end,
        )
    )


def _crsp_dataset(root: Path, bars, open_, close, **store) -> CrspStockDataset:
    """Write the store under ``root`` and return the ``CrspStockDataset`` reading it."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "crsp.zarr"
    write_crsp_store(path, bars, open_, close, **store)
    return CrspStockDataset(
        CrspDatasetConfig(
            zarr_file_path=str(path),
            raw_data_dir_path=str(root / "raw"),
            reference_dir=str(root / "reference"),
            start_date=bars[0].strftime("%Y-%m-%d"),
            end_date=bars[-1].strftime("%Y-%m-%d"),
        )
    )


def build_constructor_run(
    root: Path,
    bars: pd.DatetimeIndex,
    open_: Mapping[int, Sequence[float]],
    close: Mapping[int, Sequence[float]],
    predictions: Mapping[str, Mapping[int, Sequence[float]]],
    rule: PortfolioConstructor | Callable[[CrspStockDataset], PortfolioConstructor],
    labels: Sequence[LabelSpec],
    *,
    first_bar: int = 0,
    rebalance_periods: int = 1,
    init_cash: float = 10_000.0,
    fees: float = 0.001,
    slippage: float = 0.0,
    variables: Mapping[str, Mapping[int, Sequence[float]]] | None = None,
) -> tuple[Path, list[str]]:
    """Write a CRSP-shaped store and a closed-loop-ready quantlab run; return it.

    The store holds every bar of ``bars``; the run's window starts at
    ``bars[first_bar]`` (the bars before it are the rule's warm-up).
    ``predictions`` gives each label's prediction per PERMNO over the
    window's bars. ``rule`` may be a function of the price dataset, for a
    rule built on it (a factor risk model's stores). Returns the run
    directory and the ISO timestamps of the bars ``DecisionInputs.weights``
    held after a failure.
    """
    root = Path(root)
    dataset = _crsp_dataset(root, bars, open_, close, variables=variables)
    if not isinstance(rule, PortfolioConstructor):
        rule = rule(dataset)
    window = bars[first_bar:]
    first, last = window[0], window[-1]
    prices = dataset.panel(first, last).load()
    symbols = prices["symbol"].values
    panel = xr.Dataset(
        {
            name: (
                ("timestamp", "symbol"),
                np.array([values[p] for p in symbols], dtype=float).T,
            )
            for name, values in predictions.items()
        },
        coords={"timestamp": window, "symbol": symbols},
    )
    rule.bind(labels)
    weights = DecisionInputs(
        dataset,
        rule,
        fill_column="adjOpen",
        valuation_column="adjClose",
        rebalance_periods=rebalance_periods,
        anchor=first,
    ).weights(panel, delisted=dataset.delisting_bars(prices, "adjClose"))
    failed = list(weights.attrs.pop("failed_bars"))
    weights.attrs.clear()
    backtester = USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=dataset,
            start_date=first.strftime("%Y-%m-%d"),
            end_date=last.strftime("%Y-%m-%d"),
            output_dir=str(root / "runs"),
            rebalance_periods=rebalance_periods,
            constructor=rule,
            fees=fees,
            slippage=slippage,
            init_cash=init_cash,
        )
    )
    run_dir = Path(backtester.run_weights(weights).run_dir)
    # A model-free stand-in for a run with a model: the panel goes where
    # quantlab's BacktestRun.predictions reads a run's prediction panel.
    PredictionPanel(panel, labels).write(run_dir / PREDICTION_PANEL)
    return run_dir, failed


def build_factor_risk_run(
    root: Path,
    bars: pd.DatetimeIndex,
    open_: Mapping[int, Sequence[float]],
    close: Mapping[int, Sequence[float]],
    predictions: Mapping[str, Mapping[int, Sequence[float]]],
    labels: Sequence[LabelSpec],
    *,
    first_bar: int,
    exposure_bounds: Mapping[str, tuple[float, float]],
    exposure_data_strategy: str = "read",
    seed: int = 7,
    **mean_variance,
) -> tuple[Path, list[str]]:
    """Write a closed-loop-ready quantlab run of mean-variance on a factor risk model.

    ``build_constructor_run`` with the store carrying
    ``risk_model_variables`` and the rule a ``MeanVarianceOptimizer`` whose
    covariance is a ``FactorRiskStoreEstimator`` of ``factor_risk_model``,
    with ``exposure_bounds`` on the risk model's exposures. The first label
    is the expected return; ``mean_variance`` holds the remaining
    ``MeanVarianceConfig`` fields (``risk_aversion`` and ``ic`` default to
    5 and 0.05) and ``build_constructor_run``'s keywords.
    """
    root = Path(root)
    run_fields = {
        k: mean_variance.pop(k)
        for k in ("rebalance_periods", "init_cash", "fees", "slippage")
        if k in mean_variance
    }
    mean_variance = {"risk_aversion": 5.0, "ic": 0.05, **mean_variance}

    def rule(dataset):
        model = factor_risk_model(
            root / "risk", dataset, bars, first_bar, exposure_data_strategy
        )
        return MeanVarianceOptimizer(
            MeanVarianceConfig(
                expected_return_label=labels[0].name,
                covariance=FactorRiskStoreEstimator(
                    FactorRiskStoreEstimatorConfig(risk_model=model)
                ),
                exposure_bounds=dict(exposure_bounds),
                **mean_variance,
            )
        )

    return build_constructor_run(
        root, bars, open_, close, predictions, rule, labels,
        first_bar=first_bar,
        variables=risk_model_variables(sorted(close), len(bars), seed),
        **run_fields,
    )
