"""``QuantlabRun``: the run trader executes, read through quantlab's ``BacktestRun``.

A quantlab backtest run directory is read only through quantlab's
``quantlab.runs.backtest_run.BacktestRun`` (quantlab ADR 0020): its window,
market columns, annualization, execution settings, rebalance period,
initial cash and data fingerprint as typed values; its rebalance table,
prediction panel, metrics and equity curve as loaded objects; its price
dataset, rule and tracker rebuilt from its recipe. trader names no file of
the run and indexes no key of its config, and learns everything without
importing quantlab's backtest layer (which loads vectorbt).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd
import xarray as xr

from quantlab.dataset.base import MarketDataset
from quantlab.portfolio.base import PortfolioConstructor
from quantlab.runs.prediction_panel import PredictionPanel
from quantlab.tracking.base import NullTracker, Tracker
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.runs.backtest_run import BacktestRun, Market
from quantlab.utils import date_range
from quantlab.runs import backtest_stats

#: The split keys of a quantlab run's metrics (``run()`` records the singular
#: ``in_sample_range`` and ``training_window``, a ``run_cv()`` run the plural
#: ones under ``stitched``); a ``run_weights()`` run has none.
SPLIT_KEYS: tuple[str, ...] = (
    "training_window",
    "training_windows",
    "in_sample_range",
    "in_sample_ranges",
    "out_of_sample_ranges",
)

#: Price variables trader cannot execute without: the raw open (fills) and
#: close (sizing and the equity mark), the adjusted close (decision prices,
#: ADR 0002), and the corporate-action fields the backtest venue books splits,
#: value distributions and dividends from (ADR 0009).
#: A membership-masked derived store (adjusted columns, NaN off-membership)
#: has no raw open or corporate-action fields, so this check is the refusal of
#: a price dataset that is not an unmasked market dataset (ADR 0006); trader
#: does not scan NaN patterns for masking.
REQUIRED_PRICE_VARIABLES: tuple[str, ...] = (
    "open",
    "close",
    "adjClose",
    "splitFactor",
    "cumfacshr",
    "divCash",
)


class ClosedLoopRefused(ValueError):
    """A quantlab run trader can replay open-loop but not closed-loop.

    Raised by ``QuantlabRun.decision_inputs`` for a run valued at raw prices
    or a rule declaring factors or a factor risk model (``declared_inputs()``);
    the parity ladder then
    leaves out its closed-versus-open block instead of failing.

    Examples
    --------
    >>> issubclass(ClosedLoopRefused, ValueError)
    True
    """


@dataclass(frozen=True)
class QuantlabRun:
    """A quantlab backtest run, as trader executes it.

    Attributes
    ----------
    run_dir : pathlib.Path
        The run directory, absolute.
    backtest_run : quantlab.runs.backtest_run.BacktestRun
        The run, as quantlab reads it.
    price_dataset : quantlab.dataset.base.MarketDataset
        The run's price dataset, rebuilt from its recipe.
    market : quantlab.runs.backtest_run.Market
        ``fill_price_column`` and ``valuation_price_column``, the run's
        (adjusted) decision columns.
    rebalance_periods : int
        Bars between rebalances.
    window : tuple of pandas.Timestamp
        The run's first and last bar.
    init_cash : float
        The run's starting cash.
    fees : float
        The run's fee, a fraction of the traded notional.
    slippage : float
        The run's slippage, a fraction of the fill price.
    data_fingerprint : dict or None
        The run's record of the data it read, for a trader run's config.
    trading_days_per_year, session_minutes_per_day : int
        The run's annualization.

    Examples
    --------
    Loaded from a quantlab backtest run directory, then read for its
    decision column and window::

        run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
        valuation_column = run.market.valuation_price_column
        start, end = run.window
    """

    run_dir: Path
    backtest_run: BacktestRun
    price_dataset: MarketDataset
    market: Market
    rebalance_periods: int
    window: tuple[pd.Timestamp, pd.Timestamp]
    init_cash: float
    fees: float
    slippage: float
    data_fingerprint: dict | None
    trading_days_per_year: int
    session_minutes_per_day: int

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
            If ``run_dir`` is not a quantlab backtest run quantlab can read,
            the price dataset is not a ``MarketDataset``, or its store lacks
            any of ``REQUIRED_PRICE_VARIABLES``, as a membership-masked
            derived store does (ADR 0006).

        Examples
        --------
        >>> QuantlabRun.load("no/such/run")
        Traceback (most recent call last):
        ...
        ValueError: ... is not a quantlab run directory: ...

        A run directory written by quantlab's backtester::

            run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
        """
        run_dir = Path(run_dir).resolve()
        try:
            backtest_run = BacktestRun.open(run_dir)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(f"{run_dir} is not a quantlab run directory: {error}") from error
        dataset = backtest_run.rebuild("price_dataset")
        if not isinstance(dataset, MarketDataset):
            raise ValueError(
                f"quantlab run {run_dir}: the price dataset "
                f"{type(dataset).__name__} is not a MarketDataset"
            )
        window = tuple(pd.Timestamp(bar) for bar in backtest_run.window)
        variables = set(dataset.panel(*window).data_vars)
        missing = [name for name in REQUIRED_PRICE_VARIABLES if name not in variables]
        if missing:
            raise ValueError(
                f"quantlab run {run_dir}: the price dataset "
                f"{type(dataset).__name__} lacks {missing}; it must be an unmasked "
                f"market dataset (membership masks predictions, never prices, so "
                f"not a membership-masked derived store; ADR 0006) "
                f"with raw open/close to execute on, adjClose to decide on (ADR 0002) "
                f"and splitFactor, cumfacshr and divCash to book corporate actions "
                f"from (ADR 0009)"
            )
        execution = backtest_run.execution
        return cls(
            run_dir=run_dir,
            backtest_run=backtest_run,
            price_dataset=dataset,
            market=backtest_run.market,
            rebalance_periods=backtest_run.rebalance_periods,
            window=window,
            init_cash=backtest_run.init_cash,
            fees=float(execution.fees),
            slippage=float(execution.slippage),
            data_fingerprint=backtest_run.data_fingerprint or None,
            trading_days_per_year=backtest_run.annualization.trading_days_per_year,
            session_minutes_per_day=backtest_run.annualization.session_minutes_per_day,
        )

    def split(self) -> dict:
        """Return the run's in-sample/out-of-sample split, as its metrics record it.

        The ``SPLIT_KEYS`` present at the level that carries them (the
        top level of a ``run()`` run, ``stitched`` of a ``run_cv()`` run);
        empty for a run without a model (``run_weights()``).

        Examples
        --------
        Empty for a ``run_weights()`` run; a model's run gives its
        ``training_window``, ``in_sample_range`` and ``out_of_sample_ranges``::

            split = QuantlabRun.load("runs/XGBoostVectorBt_20261001").split()
            training_window = split["training_window"]
        """
        metrics = self.backtest_run.metrics()
        level = metrics.get("stitched", metrics)
        return {key: level[key] for key in SPLIT_KEYS if key in level}

    def benchmark(self) -> dict | None:
        """Return the run's benchmark, or ``None`` when it had none.

        Returns
        -------
        dict or None
            ``returns``: the benchmark's per-bar returns on the run's bars
            (its equity curve's ``benchmark_returns``); ``symbol`` and
            ``axis_symbol``: its names from the run's metrics.

        Examples
        --------
        ::

            benchmark = run.benchmark()
            returns = None if benchmark is None else benchmark["returns"]
        """
        equity = self.backtest_run.equity()
        if "benchmark_returns" not in equity:
            return None
        metrics = self.backtest_run.metrics()
        info = metrics.get("stitched", metrics).get("benchmark") or {}
        return {
            "returns": equity["benchmark_returns"],
            "symbol": info.get("symbol"),
            "axis_symbol": info.get("axis_symbol"),
        }

    def equity(self) -> xr.Dataset:
        """Return the run's equity curve: ``value`` and ``returns`` on ``timestamp``.

        Examples
        --------
        ::

            run_value = run.equity()["value"]
        """
        return self.backtest_run.equity()

    def report_records(self) -> dict:
        """Return what a trader run's report restates from this run's records.

        Returns
        -------
        dict
            ``recipe``: the run's recipe, the config mapping quantlab's
            ``report_summary`` reads; ``folds``: one row per fold of a
            ``run_cv()`` run, in fold order, as quantlab's
            ``report_windows`` takes them (``fold``, ``training_window``,
            ``traded``: the ``bar_label`` of the fold's first and last bar,
            ``in_sample_range``), else ``None``; ``trained_checkpoint``: the
            checkpoint a train-mode run trained, else ``None``;
            ``benchmark_source``: where the benchmark was read from (its
            store, or the dataset held in memory), as quantlab's Setup names
            it, else ``None``.

        Examples
        --------
        ::

            records = run.report_records()
            windows = report_windows(timestamps, metrics, records["folds"])
        """
        metrics = self.backtest_run.metrics()
        folds = None
        if isinstance(metrics.get("folds"), list):
            folds = [
                {
                    "fold": fold["fold"],
                    "training_window": fold["metrics"].get("training_window"),
                    "traded": tuple(
                        date_range.bar_label(fold["metrics"]["whole"][key])
                        for key in ("Start", "End")
                    ),
                    "in_sample_range": fold["metrics"].get("in_sample_range"),
                }
                for fold in metrics["folds"]
            ]
        return {
            "recipe": self.backtest_run.recipe(),
            "folds": folds,
            "trained_checkpoint": metrics.get("trained_checkpoint"),
            "benchmark_source": self.backtest_run.benchmark_source,
        }

    def tracker(self) -> Tracker:
        """Return the run's tracker, rebuilt from its recipe; ``NullTracker`` without one.

        Examples
        --------
        ``runner.run`` tracks a trader run here unless its config names a
        tracker::

            tracker = config.tracker or run.tracker()
        """
        return self.backtest_run.rebuild("tracker") or NullTracker()

    def constructor(self) -> PortfolioConstructor | None:
        """Return the run's portfolio construction rule, rebuilt; ``None`` without one.

        Examples
        --------
        ::

            rule = run.constructor()
        """
        return self.backtest_run.rebuild("constructor")

    @property
    def backtester_class(self) -> str:
        """The class name of the run's backtester.

        Examples
        --------
        ::

            project = f"{run.backtester_class}_backtest"
        """
        return self.backtest_run.backtester_class.rsplit(".", 1)[-1]

    def rebalance_table(self) -> xr.Dataset:
        """Return the run's rebalance table: ``weight`` on ``(timestamp, symbol)``, loaded.

        Examples
        --------
        Open loop executes it as a frame of decision dates by PERMNO::

            table = run.rebalance_table()["weight"].transpose("timestamp", "symbol").to_pandas()
        """
        return self.backtest_run.weights()

    @property
    def has_prediction_panel(self) -> bool:
        """Whether the run has a prediction panel (a run with a model).

        Examples
        --------
        ::

            loop = Loop.CLOSED if run.has_prediction_panel else Loop.OPEN
        """
        return self.backtest_run.has_predictions

    def prediction_panel(self) -> PredictionPanel:
        """Return the run's prediction panel, loaded.

        Raises
        ------
        ValueError
            If the run has no prediction panel (a ``run_weights()`` run has
            no model and so no panel).

        Examples
        --------
        Closed loop reads the panel's predictions, one variable per label::

            predictions = run.prediction_panel().predictions
        """
        self._require_prediction_panel()
        return self.backtest_run.predictions()

    def _require_prediction_panel(self) -> None:
        """Refuse a run without a prediction panel, without reading the panel."""
        if not self.has_prediction_panel:
            raise ValueError(
                f"quantlab run {self.run_dir} has no prediction panel; a closed-loop "
                f"replay needs the prediction panel of a run with a model (run() or "
                f"run_cv()); replay a run_weights() run with loop='open'"
            )

    def decision_inputs(self, end: pd.Timestamp | None = None) -> DecisionInputs:
        """Return the run's decision inputs, its rule bound to its label specs.

        quantlab's ``DecisionInputs.from_run``: the bound rule, the price
        dataset, the market columns, the rebalance period and the anchor (the
        prediction panel's first bar). ``end`` is the replay's last bar,
        which never rebalances; ``None`` leaves the schedule open-ended, as a
        live run's is, counting on past the run's last bar (ADR 0008). Refused (ADR
        0005, spec #18 stories 33-34) are a run whose valuation column is not
        an adjusted price (decision prices are adjusted, ADR 0002), a run
        without a prediction panel, and a rule that declares factors or a
        factor risk model (``declared_inputs()``), which trader cannot
        compute or read without the factor layer.

        Raises
        ------
        ClosedLoopRefused
            For a run valued at raw prices or a rule declaring factors or a
            factor risk model.
        ValueError
            For a run without a prediction panel.

        Examples
        --------
        Closed loop decides with them::

            targets = ConstructorTargets(run.decision_inputs(end))
        """
        valuation = self.market.valuation_price_column
        if not valuation.startswith("adj"):
            raise ClosedLoopRefused(
                f"quantlab run {self.run_dir}: its valuation column {valuation!r} is not "
                f"an adjusted price; a closed-loop replay decides on adjusted closes "
                f"(ADR 0002)"
            )
        self._require_prediction_panel()
        inputs = DecisionInputs.from_run(self.run_dir, end=end)
        rule = inputs.constructor
        declared = rule.declared_inputs()
        if declared.factors or declared.risk_model is not None:
            raise ClosedLoopRefused(
                f"quantlab run {self.run_dir}: {type(rule).__name__} declares factors or "
                f"a factor risk model (declared_inputs()); trader cannot compute or read "
                f"them and does not replay such a rule closed-loop"
            )
        return inputs
