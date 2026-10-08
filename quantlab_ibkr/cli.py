"""``quantlab-ibkr``: the command line.

``quantlab-ibkr backtest CONFIG.json`` runs the ``TraderConfig`` a JSON file
holds (``TraderConfig.get_config()``'s layout, as a trader run's
``config.json``); ``quantlab-ibkr backtest --quantlab-run DIR [--loop
open|closed] [--start DATE] [--end DATE] [--output-dir DIR]`` runs a quantlab
run on the backtest venue with its defaults. Either prints the trader run
directory.

``quantlab-ibkr parity --quantlab-run DIR [--output-dir DIR] [--fee-model
fraction|ibkr_fixed] [--slippage X] [--init-cash X]`` writes the run's parity
report (ADR 0007) and prints its directory; it exits with 2 when an end check
of the report fails (the report is still written). ``parity`` is imported
only by this command, so ``backtest`` never loads quantlab's backtest layer.

``quantlab-ibkr live decide|record|recheck [CONFIG.json] [flags]`` runs one
step of a live day on IBKR (``quantlab_ibkr.live``, #51): ``decide`` before
the open, ``record`` after it (exit 2 when the Decision recheck of the live
run differs), ``recheck`` alone (exit 2 likewise). The config is a
``TraderConfig`` JSON file whose venue is an ``IbkrVenueConfig``, or the
flags ``--quantlab-run``, ``--prediction-store`` and ``--live-dir``; venue
flags given with a file override its fields. Credentials never pass here:
the account is ``TWS_ACCOUNT`` unless the config names it, and the Gateway
holds the login. The live modules, and nautilus's IB adapter, are imported
only by this command.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.base.venue import Loop
from quantlab_ibkr.runner import run
from quantlab_ibkr.venue.backtest.venue import BacktestVenueConfig, ExecutionConfig


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab-ibkr")
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
    live = commands.add_parser("live", help="one step of a live day on IBKR (#51)")
    steps = live.add_subparsers(dest="step", required=True)
    for step, text in (
        ("decide", "before the open: decide the last closed bar, submit market-on-open"),
        ("record", "after the open: append IBKR's fills, then the Decision recheck"),
        ("recheck", "run the Decision recheck on the live run directory"),
    ):
        command = steps.add_parser(step, help=text)
        command.add_argument("config", nargs="?", help="a TraderConfig JSON file (IBKR venue)")
        command.add_argument("--quantlab-run", help="the quantlab run directory (the recipe)")
        command.add_argument("--prediction-store", help="the run's live prediction store")
        command.add_argument("--live-dir", help="the live run directory")
        command.add_argument("--host", default=None)
        command.add_argument("--port", type=int, default=None)
        command.add_argument("--client-id", type=int, default=None)
        command.add_argument("--account-id", default=None, help="default: $TWS_ACCOUNT")
        command.add_argument("--order-deadline", default=None, help="HH:MM New York")
        command.add_argument("--dry-run", action="store_true", default=None)
        command.add_argument(
            "--force-decide", action="store_true", default=None,
            help="with --dry-run: decide t even off the rebalance cadence",
        )
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
    >>> status, "is not a quantlab run directory" in stderr.getvalue()
    (1, True)

    From the shell, ``quantlab-ibkr`` runs ``main`` and prints the trader
    run directory::

        quantlab-ibkr backtest --quantlab-run runs/WeightsVectorBt_20261001 --loop open
        quantlab-ibkr parity --quantlab-run runs/WeightsVectorBt_20261001
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "parity":
        return _parity(args)
    if args.command == "live":
        return _live(parser, args)
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
        print(f"quantlab-ibkr: {error}", file=sys.stderr)
        return 1
    print(run_dir)
    return 0


def _parity(args: argparse.Namespace) -> int:
    """Run ``quantlab-ibkr parity``; return the exit status."""
    from quantlab_ibkr.parity.ladder import parity

    try:
        execution = ExecutionConfig(
            fee_model=args.fee_model, slippage=args.slippage, init_cash=args.init_cash
        )
        parity_dir = parity(args.quantlab_run, output_dir=args.output_dir, execution=execution)
    except (ValueError, NotImplementedError) as error:
        print(f"quantlab-ibkr: {error}", file=sys.stderr)
        return 1
    print(parity_dir)
    checks = json.loads((parity_dir / "parity.json").read_text())["checks"]
    failed = sorted(name for name, check in checks.items() if not check["passed"])
    if failed:
        print(f"quantlab-ibkr: parity checks failed: {', '.join(failed)}", file=sys.stderr)
        return 2
    return 0


#: Flags that override the IBKR venue's fields.
_VENUE_FLAGS = {
    "prediction_store": "prediction_store",
    "live_dir": "live_dir",
    "host": "host",
    "port": "port",
    "client_id": "client_id",
    "account_id": "account_id",
    "order_deadline": "order_deadline",
    "dry_run": "dry_run",
    "force_decide": "force_decide",
}


def _live_config(parser: argparse.ArgumentParser, args: argparse.Namespace) -> TraderConfig:
    """The live ``TraderConfig``: the JSON file, or the flags, with the venue flags applied."""
    from quantlab_ibkr.venue.ibkr.venue import IbkrVenueConfig

    flags = {
        name: getattr(args, flag)
        for flag, name in _VENUE_FLAGS.items()
        if getattr(args, flag) is not None
    }
    if args.config is not None:
        config = TraderConfig.from_config(json.loads(Path(args.config).read_text()))
        if args.quantlab_run is not None:
            config = dataclasses.replace(config, quantlab_run=args.quantlab_run)
        if not isinstance(config.venue, IbkrVenueConfig):
            parser.error(f"{args.config}: the venue is not an IbkrVenueConfig")
        return dataclasses.replace(config, venue=dataclasses.replace(config.venue, **flags))
    if args.quantlab_run is None or "prediction_store" not in flags or "live_dir" not in flags:
        parser.error(
            "live takes CONFIG.json or --quantlab-run, --prediction-store and --live-dir"
        )
    return TraderConfig(quantlab_run=args.quantlab_run, venue=IbkrVenueConfig(**flags))


def _live(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run ``quantlab-ibkr live <step>``; return the exit status."""
    from quantlab_ibkr import live

    try:
        config = _live_config(parser, args)
        if args.step == "decide":
            print(live.decide(config).summary())
            return 0
        if args.step == "record":
            result = live.record(config)
            print(result.summary())
            recheck = result.recheck
        else:
            recheck = live.recheck(config)
            print(
                f"Decision recheck: {recheck['bars_checked']} bars checked, "
                f"{recheck['bars_differing']} differing"
            )
    except (ValueError, NotImplementedError, ConnectionError, TimeoutError) as error:
        print(f"quantlab-ibkr: {error}", file=sys.stderr)
        return 1
    if recheck["bars_differing"]:
        print("quantlab-ibkr: the Decision recheck of the live run differs", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
