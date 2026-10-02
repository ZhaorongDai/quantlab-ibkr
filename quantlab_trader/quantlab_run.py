"""``QuantlabRun``: the only reader of a quantlab run directory.

A quantlab run directory holds the backtest's ``config.json`` (the price
dataset's config, the window, costs and the ``market`` block naming the fill
and valuation price columns) and its outputs, among them the rebalance table
``weights.zarr``, its ``metrics.json`` (the in-sample and out-of-sample
ranges a trader run's metrics are split by), ``equity.zarr`` (the
benchmark's curve, when the run had one) and, for a run with a model, its prediction panel
``predictions.zarr``, from which the run's portfolio construction rule is
rebuilt (quantlab's ``load_constructor``). trader learns everything about
the run here, without importing quantlab's backtest layer (which loads
vectorbt).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd
import xarray as xr

from quantlab.base.data import MarketDataset
from quantlab.base.portfolio import PortfolioConstructor, PredictionPanel
from quantlab.base.tracking import NullTracker, Tracker
from quantlab.portfolio.prediction_panel import load_constructor
from quantlab.utils.module import get_cls_from_path, load_dataset_from_config

#: The split keys of a quantlab run's ``metrics.json`` (``run()`` writes the
#: singular ``in_sample_range`` and ``training_window``, a ``run_cv()`` run the
#: plural ones under ``stitched``); a ``run_weights()`` run has none.
SPLIT_KEYS: tuple[str, ...] = (
    "training_window",
    "training_windows",
    "in_sample_range",
    "in_sample_ranges",
    "out_of_sample_ranges",
)

#: The US-equity annualization quantlab's ``US_EQUITY_MARKET`` uses, for a run
#: whose config does not record its own (only a weights config does).
US_EQUITY_TRADING_DAYS = 252
US_EQUITY_SESSION_MINUTES = 390

#: Price variables trader cannot execute without: the raw open (fills) and
#: close (sizing and the equity mark), the adjusted close (decision prices,
#: ADR 0002), and the corporate-action fields the backtest venue books splits,
#: value distributions and dividends from (ADR 0009).
REQUIRED_PRICE_VARIABLES: tuple[str, ...] = (
    "open",
    "close",
    "adjClose",
    "splitFactor",
    "cumfacshr",
    "divCash",
)


@dataclass(frozen=True)
class QuantlabRun:
    """A quantlab backtest run, as trader executes it.

    Attributes
    ----------
    run_dir : pathlib.Path
        The run directory, absolute.
    config : dict
        Its ``config.json``.
    price_dataset : quantlab.base.data.MarketDataset
        The run's price dataset, rebuilt from its config.
    market : dict
        The ``market`` block: ``fill_price_column`` and
        ``valuation_price_column``, the run's (adjusted) decision columns.
    rebalance_periods : int
        Bars between rebalances.
    window : tuple of pandas.Timestamp
        The run's first and last bar dates (``start_date``, ``end_date``).
    init_cash : float
        The run's starting cash.
    fees : float
        The run's fee, a fraction of the traded notional.
    slippage : float
        The run's slippage, a fraction of the fill price.
    data_fingerprint : dict or None
        The run's record of the data it read, for a trader run's config.
    trading_days_per_year, session_minutes_per_day : int
        The run's annualization: its config's, or US equity's (252, 390).
    """

    run_dir: Path
    config: dict
    price_dataset: MarketDataset
    market: dict
    rebalance_periods: int
    window: tuple[pd.Timestamp, pd.Timestamp]
    init_cash: float
    fees: float
    slippage: float
    data_fingerprint: dict | None
    trading_days_per_year: int = US_EQUITY_TRADING_DAYS
    session_minutes_per_day: int = US_EQUITY_SESSION_MINUTES

    @classmethod
    def load(cls, run_dir: str | Path) -> Self:
        """Read the run at ``run_dir``, refusing one trader cannot execute.

        Parameters
        ----------
        run_dir : str or pathlib.Path
            A quantlab backtest run directory.

        Returns
        -------
        QuantlabRun

        Raises
        ------
        ValueError
            If ``run_dir`` has no ``config.json``, it has no ``market`` block, the price dataset is
            not a ``MarketDataset``, or its store lacks any of
            ``REQUIRED_PRICE_VARIABLES``.
        """
        run_dir = Path(run_dir).resolve()
        if not (run_dir / "config.json").is_file():
            raise ValueError(f"{run_dir} is not a quantlab run directory: no config.json")
        config = json.loads((run_dir / "config.json").read_text())
        market = config.get("market")
        if not market:
            raise ValueError(
                f"quantlab run {run_dir}: config.json has no market block "
                f"(fill_price_column, valuation_price_column); rerun it with a "
                f"quantlab that writes one (quantlab #106)"
            )
        dataset = load_dataset_from_config(config["price_dataset"], run_dir=run_dir)
        if not isinstance(dataset, MarketDataset):
            raise ValueError(
                f"quantlab run {run_dir}: the price dataset "
                f"{type(dataset).__name__} is not a MarketDataset"
            )
        window = (pd.Timestamp(config["start_date"]), pd.Timestamp(config["end_date"]))
        variables = set(dataset.panel(*window).data_vars)
        missing = [name for name in REQUIRED_PRICE_VARIABLES if name not in variables]
        if missing:
            raise ValueError(
                f"quantlab run {run_dir}: the price dataset "
                f"{type(dataset).__name__} lacks {missing}; trader executes on raw "
                f"open/close, decides on adjClose (ADR 0002) and books corporate "
                f"actions from splitFactor, cumfacshr and divCash (ADR 0009)"
            )
        return cls(
            run_dir=run_dir,
            config=config,
            price_dataset=dataset,
            market=dict(market),
            rebalance_periods=int(config["rebalance_periods"]),
            window=window,
            init_cash=float(config["init_cash"]),
            fees=float(config["fees"]),
            slippage=float(config["slippage"]),
            data_fingerprint=config.get("data_fingerprint"),
            trading_days_per_year=int(
                config.get("trading_days_per_year") or US_EQUITY_TRADING_DAYS
            ),
            session_minutes_per_day=int(
                config.get("session_minutes_per_day") or US_EQUITY_SESSION_MINUTES
            ),
        )

    def split(self) -> dict:
        """Return the run's in-sample/out-of-sample split, as its ``metrics.json`` records it.

        The ``SPLIT_KEYS`` present at the level that carries them (the
        top level of a ``run()`` run, ``stitched`` of a ``run_cv()`` run);
        empty for a run without a model (``run_weights()``) or without a
        ``metrics.json``.
        """
        path = self.run_dir / "metrics.json"
        if not path.is_file():
            return {}
        metrics = json.loads(path.read_text())
        level = metrics.get("stitched", metrics)
        return {key: level[key] for key in SPLIT_KEYS if key in level}

    def benchmark(self) -> dict | None:
        """Return the run's benchmark, or ``None`` when it had none.

        Returns
        -------
        dict or None
            ``returns``: the benchmark's per-bar returns on the run's bars
            (``equity.zarr``'s ``benchmark_returns``); ``symbol`` and
            ``axis_symbol``: its names from ``metrics.json``.
        """
        with xr.open_zarr(self.run_dir / "equity.zarr") as equity:
            if "benchmark_returns" not in equity:
                return None
            returns = equity["benchmark_returns"].load()
        path = self.run_dir / "metrics.json"
        metrics = json.loads(path.read_text()) if path.is_file() else {}
        info = metrics.get("stitched", metrics).get("benchmark") or {}
        return {
            "returns": returns,
            "symbol": info.get("symbol"),
            "axis_symbol": info.get("axis_symbol"),
        }

    def tracker(self) -> Tracker:
        """Return the run's tracker, rebuilt from ``config.json``; ``NullTracker`` without one."""
        recorded = self.config.get("tracker")
        if not recorded:
            return NullTracker()
        cls = get_cls_from_path(recorded["name"])
        return cls.from_config(recorded)

    @property
    def backtester_class(self) -> str:
        """The class name of the run's backtester (``WeightsVectorBt``, ...)."""
        return str(self.config.get("name", "quantlab")).rsplit(".", 1)[-1]

    def rebalance_table(self) -> xr.Dataset:
        """Return the run's ``weights.zarr``: ``weight`` on ``(timestamp, symbol)``, loaded."""
        with xr.open_zarr(self.run_dir / "weights.zarr") as table:
            return table.load()

    def prediction_panel(self) -> PredictionPanel:
        """Return the run's prediction panel, ``predictions.zarr``, loaded.

        Raises
        ------
        ValueError
            If the run has no ``predictions.zarr`` (a ``run_weights()`` run
            has no model and so no panel).
        """
        return PredictionPanel.read(self._require_prediction_panel())

    def constructor(self) -> PortfolioConstructor:
        """Return the run's portfolio construction rule, bound to its label specs.

        Refused (ADR 0005, spec #18 stories 33-34) are a run whose valuation
        column is not an adjusted price (decision prices are adjusted, ADR
        0002), a run without a prediction panel, and a rule that declares
        ``required_factors()``, which trader cannot compute without the
        factor layer.

        Raises
        ------
        ValueError
            For any of the refusals above.
        """
        valuation = self.market["valuation_price_column"]
        if not valuation.startswith("adj"):
            raise ValueError(
                f"quantlab run {self.run_dir}: its valuation column {valuation!r} is not "
                f"an adjusted price; a closed-loop replay decides on adjusted closes "
                f"(ADR 0002)"
            )
        self._require_prediction_panel()
        rule = load_constructor(self.run_dir)
        if rule.required_factors():
            raise ValueError(
                f"quantlab run {self.run_dir}: {type(rule).__name__} declares "
                f"required_factors(); trader cannot compute factors and does not "
                f"replay such a rule closed-loop"
            )
        return rule

    def _require_prediction_panel(self) -> Path:
        """Return the path of ``predictions.zarr``, refusing a run without one."""
        path = self.run_dir / PredictionPanel.FILE_NAME
        if not path.exists():
            raise ValueError(
                f"quantlab run {self.run_dir} has no {PredictionPanel.FILE_NAME}; a "
                f"closed-loop replay needs the prediction panel of a run with a "
                f"model (run() or run_cv()); replay a run_weights() run with "
                f"loop='open'"
            )
        return path
