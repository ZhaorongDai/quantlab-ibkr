"""``run(config) -> Path``: wire a quantlab run, a venue, the strategy and the recorder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.base.data import InsufficientHistoryError
from quantlab_trader.base.config import TraderConfig
from quantlab_trader.calendar import RebalanceCalendar
from quantlab_trader.decision import (
    ConstructorTargets,
    DecisionCycle,
    TableTargets,
    TargetSource,
)
from quantlab_trader.outputs import RunRecorder
from quantlab_trader.quantlab_run import QuantlabRun
from quantlab_trader.strategy import PortfolioStrategy

#: The metric blocks a tracking run's summary receives, when present:
#: quantlab's tracked blocks plus ``execution``.
TRACKED_BLOCKS = ("whole", "in_sample", "out_of_sample", "benchmark", "relative", "execution")


def run(config: TraderConfig) -> Path:
    """Execute the quantlab run ``config`` names and write a trader run directory.

    Closed loop (ADR 0001, 0008) rebuilds the run's rule and decides every
    rebalance bar on the account's holdings, with the run's prediction
    panel and the decision prices from ``bar_before(anchor, lookback_bars)``;
    open loop executes the run's ``weights.zarr``.

    The run is tracked through ``config.tracker``, or the quantlab run's own
    tracker (``NullTracker`` when it had none), in the quantlab run's
    project (``{Backtester}_backtest`` unless the tracker sets its own): one
    tracking run named as the run directory, whose config gains the
    resolved records of ``config.json`` and whose summary holds the metric
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
        closed loop, ``QuantlabRun.constructor``), the window is empty or
        outside the run's, or the venue refuses its execution settings.
    """
    quantlab_run = QuantlabRun.load(config.quantlab_run)
    start, end = _window(config, quantlab_run)
    if config.loop == "closed":
        targets, permnos, venue_inputs = _closed_loop(quantlab_run, start, end)
    else:
        targets, permnos = _open_loop(quantlab_run, start, end)
        venue_inputs = {}
    venue = config.venue.build(
        quantlab_run, start=start, end=end, permnos=permnos, loop=config.loop, **venue_inputs
    )
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
        venue.run(strategy)
        run_dir = recorder.write()
        tracking.update_config(json.loads((run_dir / "config.json").read_text()))
        tracking.summarize(
            {
                block: recorder.metrics[block]
                for block in TRACKED_BLOCKS
                if recorder.metrics.get(block) is not None
            }
        )
        tracking.log_file(run_dir / "report.html")
    return run_dir


def _open_loop(
    run: QuantlabRun, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[TargetSource, tuple]:
    """Return the rebalance table's targets and the PERMNOs it ever gives a nonzero weight."""
    table = (
        run.rebalance_table()["weight"]
        .sel(timestamp=slice(start, end))
        .transpose("timestamp", "symbol")
        .to_pandas()
    )
    # A security never given a nonzero target can never be held.
    permnos = tuple(table.columns[(table.fillna(0.0) != 0.0).any(axis=0)])
    return TableTargets(table), permnos


def _closed_loop(
    run: QuantlabRun, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[TargetSource, tuple, dict]:
    """Return the constructor's targets, the PERMNOs it can trade and the venue's inputs.

    A rule only trades what it can select, a security with a finite
    prediction, so instruments are made for the PERMNOs with any finite
    prediction in the window; every holding is one of them.
    """
    rule = run.constructor()
    panel = run.prediction_panel().predictions
    anchor = pd.Timestamp(panel["timestamp"].values[0])
    if anchor > end:
        raise ValueError(
            f"replay window ends {end.date()}, before the prediction panel starts "
            f"{anchor.date()}"
        )
    dataset = run.price_dataset
    calendar = RebalanceCalendar(
        pd.DatetimeIndex(dataset.panel(anchor, end)["timestamp"].values),
        run.rebalance_periods,
    )
    predictions = panel.sel(timestamp=slice(start, end))
    finite = np.zeros(predictions.sizes["symbol"], dtype=bool)
    for name in predictions.data_vars:
        finite |= np.isfinite(
            predictions[name].transpose("timestamp", "symbol").values
        ).any(axis=0)
    permnos = tuple(v.item() for v in predictions["symbol"].values[finite])
    return (
        ConstructorTargets(rule, calendar),
        permnos,
        dict(predictions=predictions, history_start=_history_start(dataset, anchor, rule.lookback_bars)),
    )


def _history_start(dataset, anchor: pd.Timestamp, lookback: int) -> pd.Timestamp:
    """Return ``bar_before(anchor, lookback)``, or the dataset's first bar when it holds fewer.

    quantlab's backtester starts the rule's prices there too, with a short
    first window when the dataset holds fewer bars.
    """
    try:
        return dataset.bar_before(anchor, lookback)
    except InsufficientHistoryError as exc:
        return dataset.bar_before(anchor, exc.available)


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
