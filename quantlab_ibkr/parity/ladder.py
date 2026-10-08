"""The parity ladder and its report (ADR 0007).

``parity(quantlab_run)`` walks from a quantlab run to trader's open-loop
replay of it one execution convention at a time and writes a **parity
report**, so that every difference between the two is attributed to a
convention (the residual is zero by construction, never "small"):

======  ==============================================================
Rung    Simulation (and what it changes from the rung above)
======  ==============================================================
L0      quantlab ``run_weights``, as the run did: reproduces its equity
L1      quantlab ``run_weights`` with ``sizing_basis="valuation"``: sized
        at t's valuation close, not t+1's fill price
L2      reference ledger: cash no longer capped
L3      reference ledger: raw prices; splits change share counts,
        dividends and value distributions are cash, delistings settle at
        the last valuation in raw prices; orders follow trader's
        decision cycle (an order is a change of holding, an order without
        an opening print is rejected)
L4      reference ledger: whole shares, ``trunc``; splits floor toward
        zero with cash in lieu
L5      reference ledger: trader's fee and slippage models, prices at the
        instruments' 4 decimals, money at the cent
T       trader's open-loop replay (``runner.run``): must equal L5
======  ==============================================================

L0 and L1 run quantlab's own engine (the ``parity`` package is the only
trader package that imports quantlab's backtest layer, ADR 0008); trader
never re-implements vectorbt's sizing, rejections or settlements. L2-L5 run
the **reference ledger**, a numpy simulator in this package, independent of
the nautilus run: it re-implements the corporate-action and sizing
arithmetic rather than calling the venue's, sharing only the tolerances of
the classification (``FACTOR_RTOL``, ``PRICE_FACTOR_RTOL``,
``IMPLIED_SPLIT_TOL``) and the pure cost functions
``IbkrFixedFeeModel.charge`` and ``slipped_price`` (unit-tested on their
own; the ladder then checks how nautilus applies them).

The report directory ``<output_dir>/<run name>_parity_<stamp>/`` holds:

- ``parity.json``: the inputs (run directory, trader runs, the execution
  block, the data fingerprints and whether they agree, and
  ``closed_loop_refused``, why a run with a prediction panel was not replayed
  closed-loop: ``ClosedLoopRefused``'s message, else ``None``), the end checks
  (``L0_equals_run``, ``T_equals_L5`` and, when the run can be replayed
  closed-loop, ``closed_weights_equal_on_holding_independent_bars`` and
  ``decision_recheck``) with their maximum errors, one row per rung with its
  delta to the rung above, and the closed-versus-open block, whose
  ``decision_recheck`` part is the **Decision recheck**: every closed-loop
  decision bar decided again by quantlab's rule from the run's
  ``DecisionInputs``, the bar's prediction row and the current weights the
  closed loop handed the rule, which must give the decided weights bit for
  bit (``bars_checked``, ``bars_differing`` and, per differing bar, the
  symbols with both weights);
- ``parity.zarr``: ``equity`` on ``(rung, timestamp)`` and, closed loop,
  ``closed_equity`` on ``timestamp``;
- ``trader/``: the trader run directories T (and the closed loop) wrote.

This module is the ladder's driver; each step is a module of the package:

=====================  ===================================================
``market``             the run's prices on the window (``Market``)
``rung``               ``RungResult``, the rungs' order and descriptions
``quantlab_rungs``     L0, L1: quantlab's engine
``vectorbt_ledger``    L2: vectorbt's execution, cash uncapped
``trader_ledger``      L3-L5: trader's conventions, one switch at a time
``nautilus_money``     L5's money as nautilus keeps it
``trader_rung``        T and the closed loop, read from their run directories
``checks``             the end checks (``L0_equals_run``, ``T_equals_L5``)
``closed_loop``        the closed-versus-open block
``decision_recheck``   its Decision recheck
``report``             the statistics rows, ``parity.json`` and ``parity.zarr``
=====================  ===================================================
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

from quantlab.tracking.base import NullTracker
from quantlab_ibkr import runner
from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.base.venue import Loop
from quantlab_ibkr.parity.checks import l0_check, t_check
from quantlab_ibkr.parity.closed_loop import closed_vs_open
from quantlab_ibkr.parity.decision_recheck import decision_recheck
from quantlab_ibkr.parity.market import Market
from quantlab_ibkr.parity.quantlab_rungs import quantlab_rung
from quantlab_ibkr.parity.report import Statistics, report_dir, rung_rows, write_report
from quantlab_ibkr.parity.rung import RUNGS, RungResult
from quantlab_ibkr.parity.trader_ledger import Conventions, trader_ledger
from quantlab_ibkr.parity.trader_rung import trader_rung
from quantlab_ibkr.parity.vectorbt_ledger import vectorbt_ledger
from quantlab_ibkr.quantlab_run import ClosedLoopRefused, QuantlabRun
from quantlab_ibkr.venue.backtest.venue import BacktestVenueConfig, ExecutionConfig


#: What the reference ledger shares with the code it checks, for the report.
LEDGER_NOTE = (
    "L2-L5 are a numpy re-implementation of the execution conventions, independent of "
    "the nautilus run; they share only FACTOR_RTOL, PRICE_FACTOR_RTOL and IMPLIED_SPLIT_TOL "
    "(the tolerances of the corporate-action classification) and the pure cost functions "
    "IbkrFixedFeeModel.charge and slipped_price, which are unit-tested on their own."
)


def parity(quantlab_run, *, output_dir=None, execution: ExecutionConfig | None = None) -> Path:
    """Run the parity ladder of ``quantlab_run`` and write its report.

    Parameters
    ----------
    quantlab_run : str or pathlib.Path
        A quantlab backtest run directory.
    output_dir : str or pathlib.Path, optional
        Where the report directory goes; the run directory's parent by default.
    execution : ExecutionConfig, optional
        The backtest venue's execution block of L5, T and the closed loop;
        unset fields resolve to the open loop's defaults (the ``fraction``
        fee model, the run's slippage and starting cash), so both loops
        execute alike.

    Returns
    -------
    pathlib.Path
        The report directory; its ``parity.json`` ``checks`` say whether
        each end of the ladder held (``passed``).

    Raises
    ------
    ValueError
        If trader cannot execute the run (``QuantlabRun.load``).

    Examples
    --------
    A run whose ladder holds at both ends reports every check passed
    (``L0_equals_run``, ``T_equals_L5`` and ``data_fingerprints_agree``)::

        report_dir = parity("runs/WeightsVectorBt_20261001", output_dir="parity")
        report = json.loads((report_dir / "parity.json").read_text())
        {name: check["passed"] for name, check in report["checks"].items()}
    """
    run = QuantlabRun.load(quantlab_run)
    execution = _resolved(execution or ExecutionConfig(), run)
    table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
    market = Market.load(run, table)
    rungs: dict[str, RungResult] = {}
    rungs["L0"], fingerprint = quantlab_rung(run, table, market, "fill", "L0")
    rungs["L1"], _ = quantlab_rung(run, table, market, "valuation", "L1")
    rungs["L2"] = vectorbt_ledger(market, run, "L2")
    ledger = Conventions(
        whole_shares=False,
        trader_costs=False,
        fee_model="fraction",
        fee_rate=run.fees,
        slippage=run.slippage,
        init_cash=run.init_cash,
    )
    rungs["L3"] = trader_ledger(market, ledger, "L3")
    ledger = dataclasses.replace(ledger, whole_shares=True)
    rungs["L4"] = trader_ledger(market, ledger, "L4")
    rungs["L5"] = trader_ledger(
        market,
        dataclasses.replace(
            ledger,
            trader_costs=True,
            fee_model=execution.fee_model,
            slippage=execution.slippage,
            init_cash=execution.init_cash,
        ),
        "L5",
    )
    parity_dir = report_dir(run, output_dir)
    partial = parity_dir.with_name(f".{parity_dir.name}.partial")
    partial.mkdir(parents=True)
    try:
        report, closed = _ladder_end(run, table, market, rungs, execution, partial, parity_dir)
        report["inputs"]["data_fingerprint"] = run.data_fingerprint
        report["inputs"]["rerun_data_fingerprint"] = fingerprint
        agree = _fingerprints_agree(run.data_fingerprint, fingerprint)
        report["inputs"]["fingerprints_agree"] = agree
        report["checks"]["data_fingerprints_agree"] = {"passed": agree is not False, "agree": agree}
        write_report(partial, report, rungs, closed)
        partial.rename(parity_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return parity_dir


def _ladder_end(run, table, market, rungs, execution, partial: Path, parity_dir: Path):
    """Run T (and the closed loop), check both ends and build the report.

    The trader runs are written under ``partial/trader`` and recorded at
    their final place under ``parity_dir``.
    """
    trader_dir = _trader_run(run, execution, Loop.OPEN, partial / "trader")
    rungs["T"] = trader_rung(trader_dir, market, execution)
    run_value = run.equity()["value"]
    checks = {
        "L0_equals_run": l0_check(rungs["L0"], run_value),
        "T_equals_L5": t_check(rungs["T"], rungs["L5"], trader_dir),
    }
    statistics = Statistics(run, market.timestamps)
    inputs = {
        "quantlab_run": str(run.run_dir),
        "trader_run": str(parity_dir / "trader" / trader_dir.name),
        "closed_loop_run": None,
        "closed_loop_refused": None,
        "execution": execution.get_config(),
        "rung_order": list(RUNGS),
        "reference_ledger": LEDGER_NOTE,
    }
    closed = loops = None
    closed_dir = None
    if run.has_prediction_panel:
        try:
            closed_dir = _trader_run(run, execution, Loop.CLOSED, partial / "trader")
        except ClosedLoopRefused as refusal:
            inputs["closed_loop_refused"] = str(refusal)
    if closed_dir is not None:
        inputs["closed_loop_run"] = str(parity_dir / "trader" / closed_dir.name)
        closed = trader_rung(closed_dir, market, execution, name="closed")
        loops = closed_vs_open(run, table, market, rungs, closed, closed_dir, statistics)
        checks["closed_weights_equal_on_holding_independent_bars"] = {
            "passed": loops["holding_independent_bars_equal"]
            == loops["holding_independent_bars"],
            "bars": loops["holding_independent_bars"],
            "bars_equal": loops["holding_independent_bars_equal"],
        }
        recheck = loops["decision_recheck"] = decision_recheck(run, closed_dir)
        checks["decision_recheck"] = {
            "passed": recheck["bars_differing"] == 0,
            "bars_checked": recheck["bars_checked"],
            "bars_differing": recheck["bars_differing"],
        }
    report = {
        "format_version": 1,
        "inputs": inputs,
        "checks": checks,
        "rungs": rung_rows(rungs, statistics),
        "closed_vs_open": loops,
    }
    return report, closed


def _resolved(execution: ExecutionConfig, run: QuantlabRun) -> ExecutionConfig:
    """``execution`` with every unset field resolved: open-loop fee model, the run's slippage and cash."""
    return ExecutionConfig(
        fee_model=execution.fee_model or "fraction",
        slippage=run.slippage if execution.slippage is None else execution.slippage,
        init_cash=run.init_cash if execution.init_cash is None else execution.init_cash,
    )


def _fingerprints_agree(recorded: dict | None, rerun: dict | None) -> bool | None:
    """Whether L0 read the data the run recorded; ``None`` when they share no request.

    A quantlab data fingerprint maps a component path to one entry per
    distinct request. The entries both sides hold (same key, same
    ``request``) are compared by ``digest`` alone, as quantlab compares.
    """
    recorded, rerun = recorded or {}, rerun or {}
    shared = [
        (entry["digest"], other["digest"])
        for key in sorted(set(recorded) & set(rerun))
        for entry in recorded[key]
        for other in rerun[key]
        if entry["request"] == other["request"]
    ]
    if not shared:
        return None
    return all(old == new for old, new in shared)


def _trader_run(run: QuantlabRun, execution: ExecutionConfig, loop: Loop, output_dir: Path) -> Path:
    """Replay the run with trader's ``runner.run``, untracked, under ``execution``."""
    return runner.run(
        TraderConfig(
            quantlab_run=str(run.run_dir),
            venue=BacktestVenueConfig(execution),
            loop=loop,
            output_dir=str(output_dir),
            tracker=NullTracker(),
        )
    )
