"""``quantlab-trader``: the command line.

``quantlab-trader backtest CONFIG.json`` runs the ``TraderConfig`` a JSON file
holds (``TraderConfig.get_config()``'s layout, as a trader run's
``config.json``); ``quantlab-trader backtest --quantlab-run DIR [--loop
open|closed] [--start DATE] [--end DATE] [--output-dir DIR]`` runs a quantlab
run on the backtest venue with its defaults. Either prints the trader run
directory.

``quantlab-trader parity --quantlab-run DIR [--output-dir DIR] [--fee-model
fraction|ibkr_fixed] [--slippage X] [--init-cash X]`` writes the run's parity
report (ADR 0007) and prints its directory; it exits with 2 when an end check
of the report fails (the report is still written). ``parity`` is imported
only by this command, so ``backtest`` never loads quantlab's backtest layer.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from quantlab_trader.base.config import TraderConfig
from quantlab_trader.base.venue import Loop
from quantlab_trader.runner import run
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig, ExecutionConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab-trader")
    commands = parser.add_subparsers(dest="command", required=True)
    backtest = commands.add_parser(
        "backtest", help="replay a quantlab run on the backtest venue"
    )
    backtest.add_argument("config", nargs="?", help="a TraderConfig JSON file")
    backtest.add_argument("--quantlab-run", help="a quantlab run directory")
    backtest.add_argument("--loop", choices=[loop.value for loop in Loop], default=None)
    backtest.add_argument("--start", default=None)
    backtest.add_argument("--end", default=None)
    backtest.add_argument("--output-dir", default=None)
    parity = commands.add_parser(
        "parity", help="write the parity report of a quantlab run (ADR 0007)"
    )
    parity.add_argument("--quantlab-run", required=True, help="a quantlab run directory")
    parity.add_argument("--output-dir", default=None)
    parity.add_argument("--fee-model", choices=("fraction", "ibkr_fixed"), default=None)
    parity.add_argument("--slippage", type=float, default=None)
    parity.add_argument("--init-cash", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status.

    Parameters
    ----------
    argv : Sequence of str, optional
        Arguments without the program name; ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        0 on success, 1 when the run is refused (the reason is printed to
        stderr), 2 when ``parity`` writes a report whose end checks fail.

    Examples
    --------
    A directory that is not a quantlab run is refused with status 1:

    >>> import contextlib, io
    >>> with contextlib.redirect_stderr(io.StringIO()) as stderr:
    ...     status = main(["backtest", "--quantlab-run", "no/such/run"])
    >>> status, stderr.getvalue().strip().endswith("is not a quantlab run directory: no config.json")
    (1, True)

    From the shell, ``quantlab-trader`` runs ``main`` and prints the trader
    run directory::

        quantlab-trader backtest --quantlab-run runs/WeightsVectorBt_20261001 --loop open
        quantlab-trader parity --quantlab-run runs/WeightsVectorBt_20261001
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "parity":
        return _parity(args)
    if (args.config is None) == (args.quantlab_run is None):
        parser.error("backtest takes either CONFIG.json or --quantlab-run DIR")
    if args.config is not None:
        if any(v is not None for v in (args.loop, args.start, args.end, args.output_dir)):
            parser.error("--loop/--start/--end/--output-dir go with --quantlab-run")
        config = TraderConfig.from_config(json.loads(Path(args.config).read_text()))
    else:
        config = TraderConfig(
            quantlab_run=args.quantlab_run,
            venue=BacktestVenueConfig(),
            loop=Loop(args.loop) if args.loop else Loop.CLOSED,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
        )
    try:
        run_dir = run(config)
    except (ValueError, NotImplementedError) as error:
        print(f"quantlab-trader: {error}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


def _parity(args: argparse.Namespace) -> int:
    """Run ``quantlab-trader parity``; return the exit status."""
    from quantlab_trader.parity import parity

    try:
        execution = ExecutionConfig(
            fee_model=args.fee_model, slippage=args.slippage, init_cash=args.init_cash
        )
        parity_dir = parity(args.quantlab_run, output_dir=args.output_dir, execution=execution)
    except (ValueError, NotImplementedError) as error:
        print(f"quantlab-trader: {error}", file=sys.stderr)
        return 1
    print(parity_dir)
    checks = json.loads((parity_dir / "parity.json").read_text())["checks"]
    failed = sorted(name for name, check in checks.items() if not check["passed"])
    if failed:
        print(f"quantlab-trader: parity checks failed: {', '.join(failed)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
