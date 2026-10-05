"""trader imports an exact allowlist of quantlab modules (ADR 0008) and reads quantlab
runs only through quantlab's ``BacktestRun``.

Three locks:

- an ``ast`` scan of every trader module: each ``import`` of quantlab names a
  module on ``ALLOWED`` (or a name inside one); the modules of the
  ``parity`` package alone may also import quantlab's backtest layer, to
  run the ladder's quantlab rungs (ADR 0007);
- a subprocess that replays a fixture run through ``runner.run`` and then
  finds none of quantlab's model, factor, label or backtest layers, nor torch,
  xgboost, KunQuant or vectorbt, in ``sys.modules``, open loop and closed
  loop (TopN; mean-variance with Ledoit-Wolf, which loads cvxpy), with the
  command line imported (it imports the ``parity`` package only inside the
  ``parity`` command), and no module of the ``parity`` package either. The
  rule's, dataset's and tracker's modules may still load by class path from
  the run's recipe;
- a source scan of every trader module: no string literal names a file of a
  quantlab run directory (``config.json``, ``run.json``, ``metrics.json``,
  ``weights.zarr``, ``equity.zarr``, ``settlements.json``, ``predictions.zarr``,
  ``report.html``, ``inputs/...``), and no run config is indexed by key
  (``.config[...]`` / ``.config.get(...)`` on a run, or ``config["market"]``-style
  reads), outside ``outputs.py``, which owns trader's own run directory and its
  files of the same names (quantlab ADR 0020).
"""

import ast
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import LedoitWolfConfig, MeanVarianceConfig, TopNConfig
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from tests.quantlab_run_fixture import build_constructor_run, build_quantlab_run

PACKAGE = Path(__file__).resolve().parents[1] / "quantlab_trader"

#: The quantlab modules trader's source may import (ADR 0008), the public
#: return-statistics module (ADR 0007) among them.
ALLOWED = {
    "quantlab.base.portfolio",
    "quantlab.portfolio.decision_inputs",
    "quantlab.runs.prediction_panel",
    "quantlab.runs.backtest_run",
    "quantlab.dataset.base",
    "quantlab.core.component",
    "quantlab.tracking.base",
    "quantlab.utils.backtest_report",
    "quantlab.utils.backtest_stats",
    "quantlab.utils.date_range",
}

#: Module prefixes that must never load in a trader process.
FORBIDDEN = (
    "quantlab.model",
    "quantlab.factor",
    "quantlab.label",
    "quantlab.backtest",
    "torch",
    "xgboost",
    "KunQuant",
    "vectorbt",
)


def _quantlab_imports(path: Path) -> list[str]:
    """Return the quantlab modules ``path`` imports, as dotted names."""
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] == "quantlab"]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "quantlab":
                if node.module in ALLOWED:
                    found.append(node.module)
                else:  # `from quantlab.x import y` may import the module x.y
                    found += [f"{node.module}.{a.name}" for a in node.names]
    return found


#: The one package whose modules may import quantlab's backtest layer, too.
PARITY = PACKAGE / "parity"


def _allowed(path: Path, name: str) -> bool:
    if name in ALLOWED:
        return True
    return PARITY in path.parents and (
        name == "quantlab.backtest" or name.startswith("quantlab.backtest.")
    )


def test_trader_source_imports_only_the_quantlab_allowlist():
    modules = list(PACKAGE.rglob("*.py"))
    assert len(modules) > 10 and (PARITY / "ladder.py") in modules

    offending = {
        str(p.relative_to(PACKAGE)): name
        for p in modules
        for name in _quantlab_imports(p)
        if not _allowed(p, name)
    }

    assert offending == {}


def _weights_run(root):
    bars = pd.bdate_range("2024-01-02", periods=4)
    return build_quantlab_run(
        root,
        bars,
        {10001: [10.0, 10.0, 11.0, 11.0]},
        {10001: [10.0, 11.0, 11.0, 12.0]},
        {10001: [1.0, np.nan, 0.0, np.nan]},
    )


def _constructor_run(root, rule, warmup):
    bars = pd.bdate_range("2024-01-02", periods=warmup + 4)
    permnos = (10001, 10002, 10003)
    steps = np.arange(len(bars))
    close = {p: list(10.0 * k + np.sin(steps * k)) for k, p in enumerate(permnos, 1)}
    run_dir, _ = build_constructor_run(
        root, bars, close, close,
        {"ret_5": {p: [0.01 * k, -0.02 * k, 0.03, 0.01] for k, p in enumerate(permnos, 1)}},
        rule, [LabelSpec("ret_5", "raw", 1, 5)], first_bar=warmup,
    )
    return run_dir


#: Each replay: its loop and how its quantlab run is built. The closed-loop
#: rule's module (and mean-variance's cvxpy) loads by class path.
REPLAYS = {
    "open": ("open", _weights_run),
    "closed_topn": (
        "closed",
        lambda root: _constructor_run(
            root, TopNConstructor(TopNConfig(direction="long_only", top_n=2)), 0
        ),
    ),
    "closed_mean_variance": (
        "closed",
        lambda root: _constructor_run(
            root,
            MeanVarianceOptimizer(
                MeanVarianceConfig(
                    expected_return_label="ret_5",
                    risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=3)),
                    risk_aversion=5.0,
                    ic=0.05,
                )
            ),
            4,
        ),
    ),
}


@pytest.mark.parametrize("replay", sorted(REPLAYS))
def test_a_replay_loads_no_model_factor_label_backtest_or_heavy_library(tmp_path, replay):
    loop, build = REPLAYS[replay]
    quantlab_run = build(tmp_path / "quantlab")
    script = textwrap.dedent(
        f"""
        import json, sys
        import quantlab_trader.cli
        from quantlab_trader.base.config import TraderConfig
        from quantlab_trader.runner import run
        from quantlab_trader.venue.backtest.venue import BacktestVenueConfig

        run_dir = run(TraderConfig(
            quantlab_run={str(quantlab_run)!r}, venue=BacktestVenueConfig(),
            loop={loop!r}, output_dir={str(tmp_path / "trader")!r},
        ))
        loaded = [m for m in sys.modules if m.startswith({FORBIDDEN!r})]
        loaded += [
            m for m in sys.modules
            if m == "quantlab_trader.parity" or m.startswith("quantlab_trader.parity.")
        ]
        print(json.dumps({{"run_dir": str(run_dir), "loaded": loaded}}))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=PACKAGE.parent,
    )

    report = json.loads(completed.stdout.strip().splitlines()[-1])
    run_dir = Path(report["run_dir"])
    assert (run_dir / "orders.zarr").exists()
    with xr.open_zarr(run_dir / "decisions.zarr") as decisions:
        assert decisions.sizes["timestamp"] > 0
    assert report["loaded"] == []


def test_trader_assembles_no_decision_inputs_of_its_own():
    """No rebalance calendar, history start or symbol alignment of trader's own (quantlab#125):
    the schedule and the context come from quantlab's ``DecisionInputs``."""
    defined, imported = set(), set()
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.Attribute):
                defined.add(node.attr)
            elif isinstance(node, ast.ImportFrom) and node.module == "quantlab.portfolio.decision_inputs":
                imported.update(alias.name for alias in node.names)
    retired = {"RebalanceCalendar", "_history_start", "history_start", "build_context", "bar_before", "rebalance_mask"}
    assert not (PACKAGE / "calendar.py").exists()
    assert defined & retired == set()
    assert "DecisionInputs" in imported


#: The files of a quantlab run directory; trader reads them through ``BacktestRun``.
_RUN_FILES = (
    "config.json", "run.json", "metrics.json", "weights.zarr", "equity.zarr",
    "settlements.json", "predictions.zarr", "report.html", "fingerprint.json",
)
_RUN_FILE = re.compile(
    r"""["'](?:(?:[^"'\n]*/)?(?:%s)|inputs/[^"'\n]*)["']""" % "|".join(map(re.escape, _RUN_FILES))
)
_KEY_READ = re.compile(
    r"\b\w*run\w*\.config\s*(?:\[|\.get\()|\bconfig\s*\[\s*[\"'](?:market|data_fingerprint|price_dataset|"
    r"benchmark_dataset|constructor|tracker|start_date|end_date|init_cash|fees|slippage|rebalance_periods)[\"']"
)
#: trader's own run directory writes files of the same names, named only here.
_OWNER = PACKAGE / "outputs.py"


def _scan_offences(sources) -> list[str]:
    return [
        f"{name}:{number}: {line.strip()}"
        for name, lines in sources
        for number, line in enumerate(lines, start=1)
        if _RUN_FILE.search(line) or _KEY_READ.search(line)
    ]


def test_trader_names_no_quantlab_run_file_and_indexes_no_run_config():
    sources = [
        (str(path.relative_to(PACKAGE)), path.read_text().splitlines())
        for path in sorted(PACKAGE.rglob("*.py"))
        if path != _OWNER
    ]
    assert len(sources) > 10

    assert _scan_offences(sources) == []


def test_the_scan_catches_what_it_locks():
    caught = _scan_offences([("x.py", [
        'config = json.loads((run_dir / "config.json").read_text())',
        'table = xr.open_zarr(run.run_dir / "weights.zarr")',
        'path = run_dir / "inputs/price_dataset.zarr"',
        'market = config["market"]',
        'rule = run.config.get("constructor")',
    ])])
    assert len(caught) == 5
    assert _scan_offences([("x.py", [
        "table = run.rebalance_table()",
        "market = run.market.valuation_price_column",
        'raise ValueError(f"{run_dir} has no config.json")',
    ])]) == []
