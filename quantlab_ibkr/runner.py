"""``run(config) -> Path``: wire a quantlab run, a venue, the strategy and the recorder."""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.runs.record import DataRecorder
from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.base.venue import Loop, ReplayRequest
from quantlab_ibkr.decision import (
    ConstructorTargets,
    DecisionCycle,
    TableTargets,
    TargetSource,
)
from quantlab_ibkr.outputs import RunRecorder, report_path, read_config
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.strategy import PortfolioStrategy

#: The metric blocks a tracking run's summary receives, when present:
#: quantlab's tracked blocks plus ``execution``.
TRACKED_BLOCKS = ("whole", "in_sample", "out_of_sample", "benchmark", "relative", "execution")


def run(config: TraderConfig) -> Path:
    """Execute the quantlab run ``config`` names and write a trader run directory.

    Closed loop (ADR 0001, 0008) rebuilds the run's decision inputs
    (quantlab's ``DecisionInputs``) and decides every rebalance bar on the
    account's holdings with the run's prediction panel; open loop executes
    the run's rebalance table. A closed loop's reads (the rule's prices,
    factors, a factor risk model's exposures and estimate) go through
    quantlab's recorded-read path: a quantlab ``DataRecorder`` keyed as the
    run's backtester keys them (``ConstructorTargets.read_sources``) is open
    while the venue runs, and its records are the trader run's
    ``data_fingerprint`` (``config.json``); an open loop's is ``None``.

    The run is tracked through ``config.tracker``, or the quantlab run's own
    tracker (``NullTracker`` when it had none), in the quantlab run's
    project (``{Backtester}_backtest`` unless the tracker sets its own): one
    tracking run named as the run directory, whose config gains the trader
    run's config (``outputs.read_config``) and whose summary holds the metric
    blocks of ``TRACKED_BLOCKS`` and which carries ``report.html``. A run
    that raises is finished as failed.

    Parameters
    ----------
    config : TraderConfig
        Which quantlab run, which venue, which loop and window.

    Returns
    -------
    pathlib.Path
        The trader run directory.

    Raises
    ------
    ValueError
        If the quantlab run cannot be executed (see ``QuantlabRun.load`` and,
        closed loop, ``QuantlabRun.decision_inputs``), the window is empty or
        outside the run's, or the venue refuses its execution settings.

    Examples
    --------
    Replaying a quantlab run on the backtest venue, closed loop::

        from quantlab_ibkr.base.config import TraderConfig
        from quantlab_ibkr.outputs import read_metrics
        from quantlab_ibkr.venue.backtest.venue import BacktestVenueConfig

        run_dir = run(TraderConfig("runs/WeightsVectorBt_20261001", BacktestVenueConfig()))
        read_metrics(run_dir)["whole"]["Total Return [%]"]
    """
    quantlab_run = QuantlabRun.load(config.quantlab_run)
    start, end = _window(config, quantlab_run)
    if config.loop is Loop.CLOSED:
        targets, request = _closed_loop(quantlab_run, start, end)
    else:
        targets, request = _open_loop(quantlab_run, start, end)
    venue = config.venue.build(quantlab_run, request)
    recorder = RunRecorder(config, quantlab_run, init_cash=venue.init_cash)
    strategy = PortfolioStrategy(
        venue=venue, cycle=DecisionCycle(targets), recorder=recorder
    )
    tracker = config.tracker or quantlab_run.tracker()
    with tracker.start_run(
        project=f"{quantlab_run.backtester_class}_backtest",
        group=None,
        name=recorder.run_name,
        config=config.get_config(),
    ) as tracking:
        reads = (
            DataRecorder(keys=targets.read_sources(), owner=recorder.run_name)
            if isinstance(targets, ConstructorTargets)
            else None
        )
        with reads or contextlib.nullcontext():
            report = venue.run(strategy)
        recorder.data_fingerprint = None if reads is None else reads.records
        run_dir = recorder.write(report)
        tracking.update_config(read_config(run_dir))
        tracking.summarize(
            {
                block: recorder.metrics[block]
                for block in TRACKED_BLOCKS
                if recorder.metrics.get(block) is not None
            }
        )
        tracking.log_file(report_path(run_dir))
    return run_dir


def _open_loop(
    run: QuantlabRun, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[TargetSource, ReplayRequest]:
    """Return the rebalance table's targets and the request for the PERMNOs it ever weights."""
    table = (
        run.rebalance_table()["weight"]
        .sel(timestamp=slice(start, end))
        .transpose("timestamp", "symbol")
        .to_pandas()
    )
    # A security never given a nonzero target can never be held.
    permnos = tuple(table.columns[(table.fillna(0.0) != 0.0).any(axis=0)])
    return TableTargets(table), ReplayRequest(start, end, permnos, Loop.OPEN)


def _closed_loop(
    run: QuantlabRun, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[TargetSource, ReplayRequest]:
    """Return the constructor's targets and the request for the PERMNOs it can trade.

    A rule only trades what it can select, a security with a finite
    prediction, so instruments are made for the PERMNOs with any finite
    prediction in the window; every holding is one of them.
    """
    inputs = run.decision_inputs(end)
    if inputs.anchor > end:
        raise ValueError(
            f"replay window ends {end.date()}, before the prediction panel starts "
            f"{inputs.anchor.date()}"
        )
    predictions = run.prediction_panel().predictions.sel(timestamp=slice(start, end))
    finite = np.zeros(predictions.sizes["symbol"], dtype=bool)
    for name in predictions.data_vars:
        finite |= np.isfinite(
            predictions[name].transpose("timestamp", "symbol").values
        ).any(axis=0)
    permnos = tuple(v.item() for v in predictions["symbol"].values[finite])
    return ConstructorTargets(inputs), ReplayRequest(
        start, end, permnos, Loop.CLOSED, predictions=predictions
    )


def _window(config: TraderConfig, run: QuantlabRun) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the replay window: the run's, narrowed by ``config.start``/``end``."""
    run_start, run_end = run.window
    start = run_start if config.start is None else pd.Timestamp(config.start)
    end = run_end if config.end is None else pd.Timestamp(config.end)
    if start < run_start or end > run_end or start > end:
        raise ValueError(
            f"replay window {start.date()}..{end.date()} must lie inside the run's "
            f"{run_start.date()}..{run_end.date()}"
        )
    return start, end
