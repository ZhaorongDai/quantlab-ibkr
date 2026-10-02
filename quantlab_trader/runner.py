"""``run(config) -> Path``: wire a quantlab run, a venue, the strategy and the recorder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab_trader.base.config import TraderConfig
from quantlab_trader.decision import DecisionCycle, TableTargets
from quantlab_trader.outputs import RunRecorder
from quantlab_trader.quantlab_run import QuantlabRun
from quantlab_trader.strategy import PortfolioStrategy


def run(config: TraderConfig) -> Path:
    """Execute the quantlab run ``config`` names and write a trader run directory.

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
        If the quantlab run cannot be executed (see ``QuantlabRun.load``), the
        window is empty or outside the run's, or the venue refuses its
        execution settings.
    NotImplementedError
        For ``loop="closed"``, which needs quantlab's prediction panel.
    """
    quantlab_run = QuantlabRun.load(config.quantlab_run)
    start, end = _window(config, quantlab_run)
    if config.loop != "open":
        raise NotImplementedError(
            "closed-loop replay is not available yet (quantlab-trader #23); "
            "use loop='open'"
        )
    table = (
        quantlab_run.rebalance_table()["weight"]
        .sel(timestamp=slice(start, end))
        .transpose("timestamp", "symbol")
        .to_pandas()
    )
    # A security never given a nonzero target can never be held.
    permnos = tuple(table.columns[(table.fillna(0.0) != 0.0).any(axis=0)])
    venue = config.venue.build(
        quantlab_run, start=start, end=end, permnos=permnos, loop=config.loop
    )
    recorder = RunRecorder(config, quantlab_run, init_cash=venue.init_cash)
    strategy = PortfolioStrategy(
        venue=venue, cycle=DecisionCycle(TableTargets(table)), recorder=recorder
    )
    venue.run(strategy)
    return recorder.write()


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
