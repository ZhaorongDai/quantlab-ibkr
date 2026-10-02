"""trader imports an exact allowlist of quantlab modules (ADR 0008).

Two locks:

- an ``ast`` scan of every trader module but ``parity.py``: each ``import`` of
  quantlab names a module on ``ALLOWED`` (or a name inside one);
- a subprocess that replays a fixture run through ``runner.run`` and then
  finds none of quantlab's model, factor, label or backtest layers, nor torch,
  xgboost, KunQuant or vectorbt, in ``sys.modules``. The rule's, dataset's and
  tracker's modules may still load by class path from ``config.json``.
"""

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from tests.quantlab_run_fixture import build_quantlab_run

PACKAGE = Path(__file__).resolve().parents[1] / "quantlab_trader"

#: The quantlab modules trader's source may import (ADR 0008). quantlab's
#: public return-statistics module joins when the metrics ticket adds it.
ALLOWED = {
    "quantlab.base.portfolio",
    "quantlab.portfolio.prediction_panel",
    "quantlab.base.data",
    "quantlab.utils.module",
    "quantlab.base.tracking",
    "quantlab.utils.backtest_report",
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


def test_trader_source_imports_only_the_quantlab_allowlist():
    modules = [p for p in PACKAGE.rglob("*.py") if p.name != "parity.py"]
    assert len(modules) > 10

    offending = {
        str(p.relative_to(PACKAGE)): name
        for p in modules
        for name in _quantlab_imports(p)
        if name not in ALLOWED
    }

    assert offending == {}


def test_a_replay_loads_no_model_factor_label_backtest_or_heavy_library(tmp_path):
    bars = pd.bdate_range("2024-01-02", periods=4)
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab",
        bars,
        {10001: [10.0, 10.0, 11.0, 11.0]},
        {10001: [10.0, 11.0, 11.0, 12.0]},
        {10001: [1.0, np.nan, 0.0, np.nan]},
    )
    script = textwrap.dedent(
        f"""
        import json, sys
        from quantlab_trader.base.config import TraderConfig
        from quantlab_trader.runner import run
        from quantlab_trader.venue.backtest.venue import BacktestVenueConfig

        run_dir = run(TraderConfig(
            quantlab_run={str(quantlab_run)!r}, venue=BacktestVenueConfig(),
            loop="open", output_dir={str(tmp_path / "trader")!r},
        ))
        loaded = [m for m in sys.modules if m.startswith({FORBIDDEN!r})]
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
    assert (Path(report["run_dir"]) / "orders.zarr").exists()
    assert report["loaded"] == []
