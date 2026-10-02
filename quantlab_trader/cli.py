"""``quantlab-trader``: the command line.

``quantlab-trader backtest CONFIG.json`` runs the ``TraderConfig`` a JSON file
holds (``TraderConfig.get_config()``'s layout, as a trader run's
``config.json``); ``quantlab-trader backtest --quantlab-run DIR [--loop
open|closed] [--start DATE] [--end DATE] [--output-dir DIR]`` runs a quantlab
run on the backtest venue with its defaults. Either prints the trader run
directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from quantlab_trader.base.config import TraderConfig
from quantlab_trader.runner import run
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab-trader")
    commands = parser.add_subparsers(dest="command", required=True)
    backtest = commands.add_parser(
        "backtest", help="replay a quantlab run on the backtest venue"
    )
    backtest.add_argument("config", nargs="?", help="a TraderConfig JSON file")
    backtest.add_argument("--quantlab-run", help="a quantlab run directory")
    backtest.add_argument("--loop", choices=("closed", "open"), default=None)
    backtest.add_argument("--start", default=None)
    backtest.add_argument("--end", default=None)
    backtest.add_argument("--output-dir", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line; return the exit status.

    Parameters
    ----------
    argv : Sequence of str, optional
        Arguments without the program name; ``sys.argv[1:]`` by default.
    """
    parser = _parser()
    args = parser.parse_args(argv)
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
            loop=args.loop or "closed",
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
        )
    print(run(config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
