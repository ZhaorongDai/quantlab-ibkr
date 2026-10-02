"""Synthetic, model-free quantlab runs built through quantlab's public API.

``build_quantlab_run`` writes a small CRSP-shaped price store (the
``CrspStockDataset`` panel layout: integer PERMNOs on ``symbol``, raw
``open``/``high``/``low``/``close``, the adjusted ``adj*`` group, ``divCash``,
``splitFactor``, ``cumfacpr``, ``cumfacshr``) and backtests a given rebalance
table on it with quantlab's ``WeightsVectorBt.run_weights``, which writes a
real run directory (``config.json`` with its ``market`` block,
``weights.zarr``, ``equity.zarr``, ...). Nothing here imports quantlab's
``tests/``.

``build_constructor_run`` builds a closed-loop run the way ADR 0007 has
the parity fixtures built: a model-free ``PredictionPanel``, the rule's own
``construct_panel`` over it with the inputs quantlab's cross-section
backtester hands it (``tradable_bars``, ``rebalance_mask``, the fill and
valuation prices from ``bar_before(first, lookback_bars)``,
``delisting_bars``), ``run_weights`` of a ``CrossSectionBacktestConfig``
carrying the rule (so ``config.json`` records it), and the panel written as
the run's ``predictions.zarr``.

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

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.backtest.predefined.weights import WeightsVectorBt
from quantlab.backtest.selection import rebalance_mask
from quantlab.base.config import (
    CrossSectionBacktestConfig,
    CrspDatasetConfig,
    DatasetConfig,
    WeightsBacktestConfig,
)
from quantlab.base.data import InsufficientHistoryError
from quantlab.base.portfolio import LabelSpec, PortfolioConstructor, PredictionPanel
from quantlab.base.tracking import Tracker
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.stock import StockDataset

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
    rule: PortfolioConstructor,
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
    window's bars. Returns the run directory and the ISO timestamps of the
    bars ``construct_panel`` held after a failure.
    """
    root = Path(root)
    dataset = _crsp_dataset(root, bars, open_, close, variables=variables)
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
    try:
        history_start = dataset.bar_before(first, rule.lookback_bars)
    except InsufficientHistoryError as exc:
        history_start = dataset.bar_before(first, exc.available)
    history = dataset.panel(history_start, last).load()
    weights = rule.construct_panel(
        panel,
        dataset.tradable_bars(prices, "adjOpen"),
        rebalance_mask(len(window), rebalance_periods),
        fill_price=history["adjOpen"],
        valuation_price=history["adjClose"],
        delisted=dataset.delisting_bars(prices, "adjClose"),
    )
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
    PredictionPanel(panel, labels).write(run_dir / PredictionPanel.FILE_NAME)
    return run_dir, failed
