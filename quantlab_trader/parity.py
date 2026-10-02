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

L0 and L1 run quantlab's own engine (this is the only trader module that
imports quantlab's backtest layer, ADR 0008); trader never re-implements
vectorbt's sizing, rejections or settlements. L2-L5 run the **reference
ledger**, a numpy simulator written here, independent of the nautilus run:
it re-implements the corporate-action and sizing arithmetic rather than
calling the venue's, sharing only the tolerances of the classification
(``FACTOR_RTOL``, ``PRICE_FACTOR_RTOL``, ``IMPLIED_SPLIT_TOL``) and the pure cost functions
``IbkrFixedFeeModel.charge`` and ``slipped_price`` (unit-tested on their
own; the ladder then checks how nautilus applies them).

The report directory ``<output_dir>/<run name>_parity_<stamp>/`` holds:

- ``parity.json``: the inputs (run directory, trader runs, the execution
  block, the data fingerprints and whether they agree), the end checks
  (``L0_equals_run``, ``T_equals_L5`` and, when the run can be replayed
  closed-loop, ``closed_weights_equal_on_holding_independent_bars``) with
  their maximum errors, one row per rung with its delta to the rung above,
  and the closed-versus-open block;
- ``parity.zarr``: ``equity`` on ``(rung, timestamp)`` and, closed loop,
  ``closed_equity`` on ``timestamp``;
- ``trader/``: the trader run directories T (and the closed loop) wrote.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from nautilus_trader.model.enums import OrderSide

from quantlab.base.portfolio import PredictionPanel
from quantlab.base.tracking import NullTracker
from quantlab.utils import backtest_stats
from quantlab.utils.module import load_backtester_from_config
from quantlab_trader import runner
from quantlab_trader.base.config import TraderConfig
from quantlab_trader.metrics import UNFILLED_STATUSES
from quantlab_trader.quantlab_run import QuantlabRun
from quantlab_trader.venue.backtest.corporate_actions import (
    FACTOR_RTOL,
    IMPLIED_SPLIT_TOL,
    PRICE_FACTOR_RTOL,
)
from quantlab_trader.venue.backtest.fees import IbkrFixedFeeModel
from quantlab_trader.venue.backtest.fills import slipped_price
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig, ExecutionConfig

#: The rungs, in their fixed order (the gaps do not commute).
RUNGS = ("L0", "L1", "L2", "L3", "L4", "L5", "T")

#: What each rung changes from the rung above.
RUNG_DESCRIPTIONS = {
    "L0": "quantlab run_weights, as the run did",
    "L1": "sized at t's valuation close, not t+1's fill price (sizing_basis='valuation')",
    "L2": "reference ledger: cash no longer capped",
    "L3": "raw prices; splits change share counts, dividends and distributions are cash; "
    "trader's order and settlement rules",
    "L4": "whole shares (trunc); splits floor with cash in lieu",
    "L5": "trader's fee and slippage models, prices at 4 decimals, money at the cent",
    "T": "trader open-loop replay (nautilus)",
}

#: What the reference ledger shares with the code it checks, for the report.
LEDGER_NOTE = (
    "L2-L5 are a numpy re-implementation of the execution conventions, independent of "
    "the nautilus run; they share only FACTOR_RTOL, PRICE_FACTOR_RTOL and IMPLIED_SPLIT_TOL "
    "(the tolerances of the corporate-action classification) and the pure cost functions "
    "IbkrFixedFeeModel.charge and slipped_price, which are unit-tested on their own."
)

#: The relative tolerance of L0 against the run (quantlab's cross-platform anchor).
L0_RTOL = 1e-12


@dataclass
class RungResult:
    """One rung's simulation, in the terms the report compares.

    Attributes
    ----------
    name : str
        The rung.
    equity : pandas.Series
        Equity after each bar's close, on the window's bars.
    init_cash : float
        The starting cash.
    orders : pandas.DataFrame
        One row per fill of a strategy order (settlements are not orders):
        ``fill_bar``, ``symbol``, ``side`` (``BUY``/``SELL``), ``quantity``
        (positive), ``price``, ``fee``.
    rejected : list of tuple
        ``(fill bar, symbol)`` of every rejected order.
    settlements : list of tuple
        ``(settlement bar, symbol, price, quantity)`` of every delisting
        settlement of a holding; ``quantity`` is the signed holding, ``None``
        where quantlab's engine does not record it (L0, L1).
    max_target_deviation : float or None
        The largest gap between a target weight and the weight held after
        its fill bar, at the sizing prices.
    buys_capped : int or None
        Buys a cash cap cut (L0, L1) or would have cut (the ledger: a buy
        that leaves cash below zero, which vectorbt cuts or rejects);
        ``None`` for T, whose run does not record cash per fill.
    peak_cash_debit : float
        The most negative end-of-bar cash, as a positive number; 0 if never.
    positions : pandas.DataFrame or None
        The holdings after each bar, by symbol, where the comparison of
        loops needs them.
    run_dir : pathlib.Path or None
        A trader rung's run directory.
    """

    name: str
    equity: pd.Series
    init_cash: float
    orders: pd.DataFrame
    rejected: list = field(default_factory=list)
    settlements: list = field(default_factory=list)
    max_target_deviation: float | None = None
    buys_capped: int | None = None
    peak_cash_debit: float = 0.0
    positions: pd.DataFrame | None = None
    run_dir: Path | None = None


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
    >>> report_dir = parity("runs/WeightsVectorBt_20261001", output_dir="parity")  # doctest: +SKIP
    >>> report = json.loads((report_dir / "parity.json").read_text())  # doctest: +SKIP
    >>> {name: check["passed"] for name, check in report["checks"].items()}  # doctest: +SKIP
    {'L0_equals_run': True, 'T_equals_L5': True, 'data_fingerprints_agree': True}
    """
    run = QuantlabRun.load(quantlab_run)
    execution = _resolved(execution or ExecutionConfig(), run)
    table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
    market = _Market.load(run, table)
    rungs: dict[str, RungResult] = {}
    rungs["L0"], fingerprint = _quantlab_rung(run, table, market, "fill", "L0")
    rungs["L1"], _ = _quantlab_rung(run, table, market, "valuation", "L1")
    rungs["L2"] = _vectorbt_ledger(market, run, "L2")
    ledger = _Conventions(
        whole_shares=False,
        trader_costs=False,
        fee_model="fraction",
        fee_rate=run.fees,
        slippage=run.slippage,
        init_cash=run.init_cash,
    )
    rungs["L3"] = _trader_ledger(market, ledger, "L3")
    ledger = dataclasses.replace(ledger, whole_shares=True)
    rungs["L4"] = _trader_ledger(market, ledger, "L4")
    rungs["L5"] = _trader_ledger(
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
    parity_dir = _report_dir(run, output_dir)
    partial = parity_dir.with_name(f".{parity_dir.name}.partial")
    partial.mkdir(parents=True)
    try:
        report, closed = _ladder_end(run, table, market, rungs, execution, partial, parity_dir)
        report["inputs"]["data_fingerprint"] = run.data_fingerprint
        report["inputs"]["rerun_data_fingerprint"] = fingerprint
        agree = _fingerprints_agree(run.data_fingerprint, fingerprint)
        report["inputs"]["fingerprints_agree"] = agree
        report["checks"]["data_fingerprints_agree"] = {"passed": agree is not False, "agree": agree}
        _write(partial, report, rungs, closed)
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
    trader_dir = _trader_run(run, execution, "open", partial / "trader")
    rungs["T"] = _trader_rung(trader_dir, market, execution)
    with xr.open_zarr(run.run_dir / "equity.zarr") as recorded:
        run_value = recorded["value"].load()
    checks = {
        "L0_equals_run": _l0_check(rungs["L0"], run_value),
        "T_equals_L5": _t_check(rungs["T"], rungs["L5"], trader_dir),
    }
    statistics = _Statistics(run, market.timestamps)
    inputs = {
        "quantlab_run": str(run.run_dir),
        "trader_run": str(parity_dir / "trader" / trader_dir.name),
        "closed_loop_run": None,
        "execution": execution.get_config(),
        "rung_order": list(RUNGS),
        "reference_ledger": LEDGER_NOTE,
    }
    closed = closed_vs_open = None
    if (run.run_dir / PredictionPanel.FILE_NAME).exists():
        closed_dir = _trader_run(run, execution, "closed", partial / "trader")
        inputs["closed_loop_run"] = str(parity_dir / "trader" / closed_dir.name)
        closed = _trader_rung(closed_dir, market, execution, name="closed")
        closed_vs_open = _closed_vs_open(run, table, market, rungs, closed, closed_dir, statistics)
        checks["closed_weights_equal_on_holding_independent_bars"] = {
            "passed": closed_vs_open["holding_independent_bars_equal"]
            == closed_vs_open["holding_independent_bars"],
            "bars": closed_vs_open["holding_independent_bars"],
            "bars_equal": closed_vs_open["holding_independent_bars_equal"],
        }
    report = {
        "format_version": 1,
        "inputs": inputs,
        "checks": checks,
        "rungs": _rung_rows(rungs, statistics),
        "closed_vs_open": closed_vs_open,
    }
    return report, closed


@dataclass(frozen=True)
class _Market:
    """The run's prices over its window, as ``[T, S]`` arrays on the table's axes.

    ``fill``/``valuation`` are the run's (adjusted) market columns as read,
    NaN where the market had no price; ``open``/``close`` the raw prices;
    ``delisted`` the dataset's delisting bars on the valuation column;
    ``split_factor``, ``cumfacshr`` and ``dividend`` the CRSP fields,
    ``adj_close`` CRSP's ``adjClose`` (chained from its total return), which
    the price-implied share changes are read from (#27).
    """

    timestamps: pd.DatetimeIndex
    symbols: np.ndarray
    weights: np.ndarray
    fill: np.ndarray
    valuation: np.ndarray
    open: np.ndarray
    close: np.ndarray
    delisted: np.ndarray
    split_factor: np.ndarray
    cumfacshr: np.ndarray
    dividend: np.ndarray
    adj_close: np.ndarray

    @classmethod
    def load(cls, run: QuantlabRun, table: xr.DataArray) -> _Market:
        """Load the window of ``table`` for the symbols it ever gives a nonzero target.

        A symbol never given one is never held by any rung, so it is left
        out (a market-wide table has thousands of them).
        """
        traded = (np.nan_to_num(np.asarray(table.values, dtype=np.float64)) != 0.0).any(axis=0)
        table = table.isel(symbol=np.flatnonzero(traded))
        timestamps = pd.DatetimeIndex(table["timestamp"].values)
        symbols = table["symbol"].values
        dataset = run.price_dataset
        prices = (
            dataset.panel(timestamps[0], timestamps[-1], symbols=list(symbols))
            .load()
            .reindex(timestamp=timestamps, symbol=symbols)
        )
        valuation_column = run.market["valuation_price_column"]

        def array(name):
            return np.asarray(
                prices[name].transpose("timestamp", "symbol").values, dtype=np.float64
            )

        return cls(
            timestamps=timestamps,
            symbols=symbols,
            weights=np.asarray(table.values, dtype=np.float64),
            fill=array(run.market["fill_price_column"]),
            valuation=array(valuation_column),
            open=array("open"),
            close=array("close"),
            delisted=np.asarray(
                dataset.delisting_bars(prices, valuation_column)
                .transpose("timestamp", "symbol")
                .values,
                dtype=bool,
            ),
            split_factor=array("splitFactor"),
            cumfacshr=array("cumfacshr"),
            dividend=array("divCash"),
            adj_close=array("adjClose"),
        )


def _resolved(execution: ExecutionConfig, run: QuantlabRun) -> ExecutionConfig:
    """``execution`` with every unset field resolved: open-loop fee model, the run's slippage and cash."""
    return ExecutionConfig(
        fee_model=execution.fee_model or "fraction",
        slippage=run.slippage if execution.slippage is None else execution.slippage,
        init_cash=run.init_cash if execution.init_cash is None else execution.init_cash,
    )


def _fingerprints_agree(recorded: dict | None, rerun: dict | None) -> bool | None:
    """Whether L0 read the price data the run recorded (its digest); ``None`` without a record."""
    common = sorted(set(recorded or {}) & set(rerun or {}))
    if not common:
        return None
    return all(rerun[name].get("digest") == recorded[name].get("digest") for name in common)


def _quantlab_rung(
    run: QuantlabRun, table: xr.DataArray, market: _Market, sizing_basis: str, name: str
) -> tuple[RungResult, dict | None]:
    """Re-run the run's ``weights.zarr`` through quantlab's engine; return the rung and fingerprint.

    The backtester is rebuilt from ``config.json`` without its model,
    tracker, benchmark and output directory (``run_weights`` reads none of
    them), on the rebalance table's window, with ``sizing_basis``.
    """
    config = copy.deepcopy(run.config)
    timestamps = market.timestamps
    config.update(
        model=None,
        model_mode=None,
        checkpoint=None,
        output_dir=None,
        benchmark_dataset=None,
        tracker={"project": None, "name": "quantlab.base.tracking.NullTracker"},
        sizing_basis=sizing_basis,
        start_date=timestamps[0].strftime("%Y-%m-%d"),
        end_date=timestamps[-1].strftime("%Y-%m-%d"),
    )
    backtester = load_backtester_from_config(config, run_dir=run.run_dir)
    simulation = backtester.run_weights(table).simulation
    equity = simulation.value.to_pandas().reindex(timestamps)

    column = {str(s): j for j, s in enumerate(market.symbols)}
    settled = {
        (pd.Timestamp(r["settlement_timestamp"]), column[str(r["axis_symbol"])])
        for r in simulation.settlements
    }
    records = simulation.orders
    bars = timestamps.get_indexer(pd.DatetimeIndex(records["timestamp"].values))
    cols = np.array([column[str(s)] for s in records["symbol"].values], dtype=np.int64)
    signed = np.where(records["side"].values.astype(str) == "Buy", 1.0, -1.0) * np.asarray(
        records["size"].values, dtype=np.float64
    )
    position = np.zeros((len(timestamps), len(market.symbols)))
    np.add.at(position, (bars, cols), signed)
    position = np.cumsum(position, axis=0)
    valuation = _ffill(market.valuation)
    cash = equity.to_numpy() - np.nansum(position * valuation, axis=1)

    strategy = np.array(
        [(timestamps[b], c) not in settled for b, c in zip(bars, cols)], dtype=bool
    )
    orders = pd.DataFrame(
        {
            "fill_bar": timestamps[bars[strategy]],
            "symbol": market.symbols[cols[strategy]],
            "side": np.where(signed[strategy] > 0, "BUY", "SELL"),
            "quantity": np.abs(signed[strategy]),
            "price": np.asarray(records["price"].values, dtype=np.float64)[strategy],
            "fee": np.asarray(records["fees"].values, dtype=np.float64)[strategy],
        }
    )
    rung = RungResult(
        name=name,
        equity=equity,
        init_cash=run.init_cash,
        orders=orders,
        rejected=[
            (pd.Timestamp(r["fill_timestamp"]), market.symbols[column[str(r["axis_symbol"])]])
            for r in simulation.rejected_orders
        ],
        settlements=[
            (
                pd.Timestamp(r["settlement_timestamp"]),
                market.symbols[column[str(r["axis_symbol"])]],
                float(r["price"]),
                None,
            )
            for r in simulation.settlements
        ],
        max_target_deviation=simulation.max_target_deviation,
        buys_capped=_capped_buys(
            market, run, sizing_basis, position, cash, bars, cols, signed
        ),
        peak_cash_debit=max(0.0, -float(np.min(cash))),
        positions=pd.DataFrame(position, index=timestamps, columns=pd.Index(market.symbols)),
    )
    return rung, backtester.get_config().get("data_fingerprint")


def _capped_buys(market, run, sizing_basis, position, cash, bars, cols, signed) -> int:
    """Count the buys vectorbt filled short of their target: cut by the cash cap.

    The requested size is vectorbt's ``target * V / sizing price - position``
    with ``V`` the book before the fill bar valued at the sizing prices (the
    fill prices, or the signal bar's valuation prices).
    """
    target = _shift(market.weights)
    fill, valuation = _ffill(market.fill), _ffill(market.valuation)
    sizing = fill if sizing_basis == "fill" else _shift(valuation)
    capped = 0
    for b, c, size in zip(bars, cols, signed):
        if size <= 0 or b == 0 or not np.isfinite(target[b, c]):
            continue
        before = position[b - 1]
        held = before != 0
        book = cash[b - 1] + float(np.sum(before[held] * sizing[b, held]))
        requested = target[b, c] * book / sizing[b, c] - before[c]
        if size < requested - _VBT_REL * abs(requested) - _VBT_ABS:
            capped += 1
    return capped


#: vectorbt's closeness tolerances (``vectorbt.utils.math_``).
_VBT_REL, _VBT_ABS = 1e-9, 1e-12


def _is_close(a: float, b: float) -> bool:
    """vectorbt's ``is_close_nb``."""
    if a == b:
        return True
    return abs(a - b) <= max(_VBT_REL * max(abs(a), abs(b)), _VBT_ABS)


def _add(a: float, b: float) -> float:
    """vectorbt's ``add_nb``: ``a + b``, snapped to 0 when they cancel."""
    if np.sign(a) != np.sign(b) and _is_close(abs(a), abs(b)):
        return 0.0
    if np.sign(a) == np.sign(b) and _is_close(a + b, 0.0):
        return 0.0
    return a + b


def _ffill(values: np.ndarray) -> np.ndarray:
    """Forward-fill a ``[T, S]`` array along time."""
    return pd.DataFrame(values).ffill().to_numpy()


def _shift(values: np.ndarray) -> np.ndarray:
    """Shift a ``[T, S]`` array one bar later; the first row is NaN."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    out[1:] = values[:-1]
    return out


def _vectorbt_ledger(market: _Market, run: QuantlabRun, name: str) -> RungResult:
    """L2: vectorbt's valuation-basis execution, with cash left uncapped.

    The rules of quantlab's engine at the fill bar, re-implemented: a
    weight at t is the target at t + 1 (NaN keeps), sized as ``w * V / p -
    position`` with ``V`` the book at t's valuation prices ``p``, filled at
    t + 1's fill price moved by the run's slippage and charged the run's fee
    fraction; an order whose raw fill price or sizing price is NaN is
    rejected (recorded when it would have traded); a holding delisted on bar
    b is settled on b + 1 at its last valuation, without fee or slippage,
    whatever the weights ask. Orders run in vectorbt's order (ascending
    order value, sells first). Unlike vectorbt, a buy is never cut to the
    cash: cash may go negative, and each buy vectorbt would have cut is
    counted in ``buys_capped``.
    """
    timestamps, symbols = market.timestamps, market.symbols
    n_bars, n_symbols = market.weights.shape
    target = _shift(market.weights)
    fill, valuation = _ffill(market.fill), _ffill(market.valuation)
    sizing = _shift(valuation)
    settle = np.zeros((n_bars, n_symbols), dtype=bool)
    settle[1:] = market.delisted[:-1]
    rejected = np.isfinite(target) & (np.isnan(market.fill) | np.isnan(sizing)) & ~settle
    size = np.where(rejected, np.nan, target)
    size[settle] = 0.0
    price = np.where(settle, valuation, fill)

    cash = float(run.init_cash)
    position = np.zeros(n_symbols)
    equity = np.empty(n_bars)
    cash_path = np.empty(n_bars)
    orders, settlements, rejections = [], [], []
    capped = 0
    deviation = None
    for i in range(n_bars):
        if i > 0:
            held = position != 0
            book = cash + float(np.sum(position[held] * sizing[i, held]))
            for j in np.flatnonzero(rejected[i] & ((target[i] != 0) | (np.abs(position) > 1e-9))):
                rejections.append((timestamps[i], symbols[j]))
            columns = np.flatnonzero(np.isfinite(size[i]))
            order_value = size[i, columns] * book - position[columns] * sizing[i, columns]
            for j in columns[np.argsort(order_value, kind="stable")]:
                # A settlement closes the holding (vectorbt: a target of 0, or
                # a target amount of 0 at a last valuation of 0).
                shares = -position[j] if settle[i, j] else size[i, j] * book / sizing[i, j] - position[j]
                if not np.isfinite(shares) or _is_close(shares, 0.0):
                    continue
                rate = 0.0 if settle[i, j] else run.fees
                slip = 0.0 if settle[i, j] else run.slippage
                if shares > 0:
                    paid = shares * price[i, j] * (1 + slip)
                    fee = paid * rate
                    if not (_is_close(paid + fee, cash) or paid + fee < cash):
                        capped += 1
                    cash = _add(cash, -(paid + fee))
                    fill_price = price[i, j] * (1 + slip)
                else:
                    received = -shares * price[i, j] * (1 - slip)
                    fee = received * rate
                    cash = cash + (received - fee)
                    fill_price = price[i, j] * (1 - slip)
                if settle[i, j]:
                    settlements.append((timestamps[i], symbols[j], float(price[i, j]), float(position[j])))
                else:
                    orders.append(
                        (timestamps[i], symbols[j], "BUY" if shares > 0 else "SELL", abs(shares), fill_price, fee)
                    )
                position[j] = _add(position[j], shares)
            compared = np.isfinite(target[i]) & ~settle[i]
            if compared.any() and book > 0:
                held_weight = position * np.nan_to_num(sizing[i]) / book
                gap = float(np.max(np.abs(target[i] - held_weight)[compared]))
                deviation = gap if deviation is None else max(deviation, gap)
        held = position != 0
        equity[i] = cash + float(np.sum(position[held] * valuation[i, held]))
        cash_path[i] = cash
    return RungResult(
        name=name,
        equity=pd.Series(equity, index=timestamps),
        init_cash=float(run.init_cash),
        orders=_orders_frame(orders),
        rejected=rejections,
        settlements=settlements,
        max_target_deviation=deviation,
        buys_capped=capped,
        peak_cash_debit=max(0.0, -float(cash_path.min())),
    )


@dataclass(frozen=True)
class _Conventions:
    """The switches of the trader-convention ledger (rungs L3-L5).

    Attributes
    ----------
    whole_shares : bool
        L4 on: ``trunc`` sizing, splits floored toward zero with cash in lieu.
    trader_costs : bool
        L5 on: trader's fee model and slippage, opening prints and fill
        prices at 4 decimals, money at the cent.
    fee_model : {"fraction", "ibkr_fixed"}
        The fee model of L5; L3 and L4 charge the run's fraction.
    fee_rate : float
        The fraction of the notional charged (the run's ``fees``).
    slippage : float
        The fraction a fill moves against the order.
    init_cash : float
    """

    whole_shares: bool
    trader_costs: bool
    fee_model: str
    fee_rate: float
    slippage: float
    init_cash: float


#: Slack of a whole-share floor against binary rounding (300 * (1/3) is 99.99...).
_SHARE_EPS = 1e-9
#: Decimals of a price and of money in trader's backtest venue.
_PRICE_DECIMALS, _MONEY_DECIMALS = 4, 2


def _trader_ledger(market: _Market, conventions: _Conventions, name: str) -> RungResult:
    """L3-L5: the run's rebalance table executed by trader's conventions, on raw prices.

    Per bar, as trader's backtest venue does it (ADR 0002, 0003, 0009,
    quantlab ADR 0014):

    - 09:30 of bar t, on the holding of the prior close: a dividend pays
      ``divCash * q`` (not a delisting payment: ``divCash`` on a row without
      a raw close, on the delisting or settlement bar of a settled
      delisting, which the settlement pays); a holder split (``splitFactor`` k finite and positive,
      equal to the share factor ``cumfacshr[t-1] / cumfacshr[t]``) makes the
      holding ``q * k`` (whole shares: floored toward zero, the fraction paid
      at the pre-split close / k); a value distribution (k > 1, share factor
      1) pays ``q * (k - 1) * close[t]`` (the pre-split close / k without a
      close); a share change the prices imply is booked as a split by its
      factor (``_holder_days``, #27); a holding delisted on bar t - 1 is settled at its last
      valuation in raw prices, the last raw close grown by the valuation
      column's return since;
    - the open of t: the orders decided at the close of t - 1, sells first,
      fill at the raw open moved by the slippage and pay the fee; an order
      without an opening print is rejected (the holding is kept), one queued
      across a split (implied or not) is rescaled by k;
    - the close of t: equity is cash plus each holding at its raw close
      (carried forward over a halt; the last valuation on a delisting
      bar); each finite target of the table's row becomes an order of
      ``w * equity / close - q`` shares (``trunc`` of the target with whole
      shares), none where the target is the holding's current weight.

    Cash is never capped; ``buys_capped`` counts the buys a cap would have
    cut: every buy that leaves cash below zero.
    """
    timestamps, symbols = market.timestamps, market.symbols
    n_bars, n_symbols = market.weights.shape
    whole, costs = conventions.whole_shares, conventions.trader_costs
    money = (lambda x: round(float(x), _MONEY_DECIMALS)) if costs else float

    close = market.close
    last_close = _ffill(close)
    paired = np.isfinite(close) & np.isfinite(market.valuation)
    with np.errstate(divide="ignore", invalid="ignore"):
        grown = (
            _ffill(np.where(paired, close, np.nan))
            * market.valuation
            / _ffill(np.where(paired, market.valuation, np.nan))
        )
    last_value = np.where(np.isfinite(grown) & (grown >= 0), grown, last_close)
    if costs:
        last_value = np.round(last_value, _PRICE_DECIMALS)
    mark = _ffill(np.where(market.delisted, last_value, close))

    split_factor = market.split_factor
    with np.errstate(divide="ignore", invalid="ignore"):
        share_factor = _shift(_ffill(market.cumfacshr)) / market.cumfacshr
    dividend = np.where(np.isfinite(market.dividend), market.dividend, 0.0)
    pre_close = _shift(last_close)
    kind = _factor_days(split_factor, share_factor)
    kind[0] = ""
    kind, holder = _holder_days(market, kind, dividend, pre_close)

    book = _NautilusMoney(conventions.init_cash, n_symbols) if costs else None
    cash = float(conventions.init_cash)
    position = np.zeros(n_symbols)
    equity = np.empty(n_bars)
    cash_path = np.empty(n_bars)
    orders, settlements, rejections = [], [], []
    capped = 0
    deviation = None
    pending: list[tuple[int, str, float]] = []
    decision = None
    for i in range(n_bars):
        if i > 0:
            # 09:30: corporate actions on the prior close's holdings, then settlements.
            for j in np.flatnonzero((kind[i] != "") | (dividend[i] != 0.0)):
                q = position[j]
                if q == 0:
                    continue
                # A delisting payment (divCash on a row without a raw close, on
                # the delisting or the settlement bar of a settled delisting)
                # is the proceeds the settlement pays, not a dividend.
                payment = np.isnan(close[i, j]) and (
                    market.delisted[i - 1, j] or (market.delisted[i, j] and i + 1 < n_bars)
                )
                if dividend[i, j] and not payment:
                    cash += money(q * dividend[i, j])
                    if book:
                        book.credit(money(q * dividend[i, j]))
                k = split_factor[i, j]
                if kind[i, j] == "SPLIT":
                    k = holder[i, j]
                    if whole:
                        shares = math.copysign(math.floor(abs(q) * k + _SHARE_EPS), q)
                        fraction = q * k - shares
                        if fraction and np.isfinite(pre_close[i, j]):
                            cash += money(fraction * pre_close[i, j] / k)
                            if book:
                                book.credit(money(fraction * pre_close[i, j] / k))
                        if book and shares != q:
                            # The venue's split fill: the share change at price 0.
                            book.fill(j, q, shares - q, 0.0, 0.0)
                        position[j] = shares
                    else:
                        position[j] = q * k
                elif kind[i, j] == "DISTRIBUTION":
                    price = close[i, j] if np.isfinite(close[i, j]) else pre_close[i, j] / k
                    cash += money(q * (k - 1.0) * price)
                    if book:
                        book.credit(money(q * (k - 1.0) * price))
            for j in np.flatnonzero(market.delisted[i - 1]):
                price = last_value[i - 1, j]
                if position[j] == 0 or not np.isfinite(price):
                    continue
                cash += position[j] * price
                if book:
                    book.fill(j, position[j], -position[j], float(price), 0.0)
                settlements.append((timestamps[i], symbols[j], float(price), float(position[j])))
                position[j] = 0.0
            # The open: the orders decided at the close of i - 1.
            traded = np.zeros(n_symbols)
            for j, side, quantity in pending:
                if not np.isfinite(market.open[i, j]):
                    rejections.append((timestamps[i], symbols[j]))
                    continue
                decided = quantity
                if kind[i, j] == "SPLIT":
                    k = holder[i, j]
                    quantity = math.floor(quantity * k + _SHARE_EPS) if whole else quantity * k
                    if quantity == 0:
                        rejections.append((timestamps[i], symbols[j]))
                        continue
                price, fee = _fill(market.open[i, j], side, quantity, conventions)
                sign = 1.0 if side == "BUY" else -1.0
                cash -= sign * quantity * price + fee
                if book:
                    book.fill(j, position[j], sign * quantity, price, fee)
                position[j] += sign * quantity
                if book:
                    cash = book.cash(position)
                if side == "BUY" and cash < 0:
                    capped += 1
                traded[j] += sign * decided  # in the decision's (pre-split) shares
                orders.append((timestamps[i], symbols[j], side, quantity, price, fee))
            if decision is not None:
                gap = _decision_gap(decision, traded, market.delisted[i - 1])
                if gap is not None:
                    deviation = gap if deviation is None else max(deviation, gap)
            pending, decision = [], None
        # The close: mark, then decide.
        held = position != 0
        if book:
            cash = book.cash(position)
        if not np.isfinite(mark[i, held]).all():
            raise ValueError(f"{name} at {timestamps[i].date()}: a holding has no raw close")
        equity[i] = cash + float(np.sum(position[held] * mark[i, held]))
        cash_path[i] = cash
        if i == n_bars - 1:
            break
        pending, decision = _decide(market.weights[i], position, mark[i], equity[i], whole)
    return RungResult(
        name=name,
        equity=pd.Series(equity, index=timestamps),
        init_cash=float(conventions.init_cash),
        orders=_orders_frame(orders),
        rejected=rejections,
        settlements=settlements,
        max_target_deviation=deviation,
        buys_capped=capped,
        peak_cash_debit=max(0.0, -float(cash_path.min())),
    )


class _NautilusMoney:
    """L5's cash, kept as trader's account derives it from nautilus.

    trader's cash is a MARGIN account's balance less the open positions'
    cost, ``signed_qty * avg_px_open`` (``account.derived_cash``). nautilus
    books into that balance, per fill, the commission and, on a fill that
    reduces a position, the realized PnL ``closed_qty * (fill px -
    avg_px_open)`` (the reverse for a short) as ``Money``, rounded to the
    cent; a fill that adds to a position moves its average open price
    instead, ``(avg * qty + px * fill_qty) / (qty + fill_qty)``, and one that
    opens a position (or the rest of one that flips it) sets it to the fill
    price. Cash credits
    (dividends, cash in lieu, distributions) are cents. Kept this way, L5's
    equity is trader's to float precision, so a whole-share target on a
    truncation boundary is cut the same way in both; an exact-notional cash
    differs by the PnL's rounding, a few cents over thousands of fills.
    """

    def __init__(self, init_cash: float, n_symbols: int):
        self.cents = round(float(init_cash) * 10**_MONEY_DECIMALS)
        self.avg = np.zeros(n_symbols)

    def credit(self, amount: float) -> None:
        """Book ``amount`` (at the cent) into the balance."""
        self.cents += round(amount * 10**_MONEY_DECIMALS)

    def fill(self, j: int, held: float, signed_qty: float, price: float, fee: float) -> None:
        """Book a fill of ``signed_qty`` at ``price`` on column ``j``, holding ``held`` before it."""
        qty = abs(signed_qty)
        if held and (held > 0) != (signed_qty > 0):
            closed = min(qty, abs(held))
            points = price - self.avg[j] if held > 0 else self.avg[j] - price
            self.credit(_nautilus_money(closed * 1.0 * points))
            if qty > abs(held):
                self.avg[j] = price
            elif qty == abs(held):
                self.avg[j] = 0.0
        elif held:
            start = abs(held)
            self.avg[j] = (self.avg[j] * start + price * qty) / (start + qty)
        else:
            # A new position opens at the fill price itself, not px * q / q.
            self.avg[j] = price
        self.credit(-fee)

    def cash(self, position: np.ndarray) -> float:
        """trader's derived cash: the balance less the open positions' cost."""
        held = position != 0
        return self.cents / 10**_MONEY_DECIMALS - float(np.sum(position[held] * self.avg[held]))


def _factor_days(split_factor: np.ndarray, share_factor: np.ndarray) -> np.ndarray:
    """Classify every bar: ``"SPLIT"``, ``"DISTRIBUTION"`` or ``""`` (nothing to book).

    A holder split has a finite positive price factor k != 1 equal to the
    share factor; a value distribution has k > 1 and a share factor of 1;
    a final event (k = 0) is left to the delisting path and anything else
    changes nothing. "Equal" is ``FACTOR_RTOL``, shared with the venue.
    """
    k, s = split_factor, share_factor
    with np.errstate(invalid="ignore"):
        k_moves = ~np.isnan(k) & ~np.isclose(k, 1.0, rtol=FACTOR_RTOL, atol=0.0)
        s_moves = ~np.isnan(s) & ~np.isclose(s, 1.0, rtol=FACTOR_RTOL, atol=0.0)
        finite = np.isfinite(k) & np.isfinite(s)
        split = finite & (k > 0) & k_moves & np.isclose(k, s, rtol=FACTOR_RTOL, atol=0.0)
        distribution = finite & (k > 1) & k_moves & ~s_moves & ~split
    return np.where(split, "SPLIT", np.where(distribution, "DISTRIBUTION", ""))


def _holder_days(
    market: _Market, kind: np.ndarray, dividend: np.ndarray, pre_close: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Check the factor days against the prices; return the kinds and each split's holder factor.

    ``adjClose`` is chained from CRSP's total return, so a holder of one
    share at the last raw close (``anchor``) is worth ``adjClose[t] /
    adjClose[anchor] * close[anchor]`` at t. Walking the bars, each
    symbol's holding per anchor share (``units`` shares and ``paid`` cash)
    takes the dividends, distributions and splits booked since the anchor;
    on a bar with a raw close, the share change that conserves that worth
    is ``x = (worth - paid - units * divCash) / (units * close)``. A bar
    the factors leave alone (no factor kind) becomes a split by
    ``splitFactor`` when that agrees with x (``PRICE_FACTOR_RTOL``) and a
    factor exists, else a split by x when x lies beyond
    ``IMPLIED_SPLIT_TOL`` either way; a final event (k = 0) stays the
    delisting path's. Returns ``kind`` with those bars
    marked ``"SPLIT"`` and the holder factor of every split (NaN elsewhere).
    """
    close, adj, k = market.close, market.adj_close, market.split_factor
    n_bars, n_symbols = close.shape
    kind = kind.copy()
    holder = np.where(kind == "SPLIT", k, np.nan)
    units, paid = np.ones(n_symbols), np.zeros(n_symbols)
    anchor_close, anchor_adj = np.full(n_symbols, np.nan), np.full(n_symbols, np.nan)
    limit = 1.0 + IMPLIED_SPLIT_TOL
    for i in range(n_bars):
        priced = np.isfinite(close[i]) & np.isfinite(adj[i])
        if i > 0:
            paid += units * dividend[i]
            spun = (kind[i] == "DISTRIBUTION") & ~priced
            # As the ledger pays it: at the close of t, or the pre-split close / k without one.
            price = np.where(np.isfinite(close[i]), close[i], pre_close[i] / np.where(spun, k[i], 1.0))
            paid += np.where(spun, units * (k[i] - 1.0) * price, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = (adj[i] / anchor_adj * anchor_close - paid) / (units * close[i])
                # A final event (k = 0) is the delisting path's, never a split.
                open_ = (kind[i] == "") & (k[i] != 0.0) & priced & np.isfinite(x) & (x > 0)
                has_factor = (
                    np.isfinite(k[i])
                    & (k[i] > 0)
                    & (np.abs(k[i] - 1.0) > FACTOR_RTOL * np.maximum(k[i], 1.0))
                )
                agrees = has_factor & (np.abs(x - k[i]) <= PRICE_FACTOR_RTOL * np.maximum(x, k[i]))
                implied = (x > limit) | (x < 1.0 / limit)
                by_factor = open_ & agrees
                by_prices = open_ & ~agrees & implied
            holder[i] = np.where(by_factor, k[i], np.where(by_prices, x, holder[i]))
            kind[i] = np.where(by_factor | by_prices, "SPLIT", kind[i])
            units = np.where((kind[i] == "SPLIT") & ~priced, units * k[i], units)
        anchor_close = np.where(priced, close[i], anchor_close)
        anchor_adj = np.where(priced, adj[i], anchor_adj)
        units, paid = np.where(priced, 1.0, units), np.where(priced, 0.0, paid)
    return kind, holder


def _fill(open_price: float, side: str, quantity: float, conventions: _Conventions) -> tuple[float, float]:
    """Return the fill price and fee of an order at the open ``open_price``."""
    if not conventions.trader_costs:
        slip = conventions.slippage
        price = open_price * (1 + slip if side == "BUY" else 1 - slip)
        return price, conventions.fee_rate * quantity * price
    printed = Decimal(repr(round(float(open_price), _PRICE_DECIMALS)))
    order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
    if conventions.slippage:
        printed = slipped_price(printed, order_side, conventions.slippage, _PRICE_DECIMALS)
    price = float(printed)
    if conventions.fee_model == "ibkr_fixed":
        fee = float(IbkrFixedFeeModel.charge(order_side, int(quantity), printed))
    else:
        fee = _nautilus_money(quantity * price * conventions.fee_rate)
    return price, fee


def _nautilus_money(value: float) -> float:
    """Return ``value`` at the cent, as nautilus's ``Money`` stores a float.

    ``value * 100`` rounded half away from zero, in binary: 17.685 is 17.68,
    its double times 100 being 1768.4999...
    """
    scaled = abs(float(value) * 10**_MONEY_DECIMALS)
    whole = math.floor(scaled)
    rounded = whole + 1 if scaled - whole >= 0.5 else whole
    return math.copysign(rounded, value) / 10**_MONEY_DECIMALS


def _decide(weights, position, mark, equity, whole):
    """trader's decision cycle on one table row: the orders (sells first) and its record."""
    sells, buys = [], []
    for j in np.flatnonzero(np.isfinite(weights)):
        w = weights[j]
        q = position[j]
        if q and w == q * mark[j] / equity:
            continue
        if w == 0.0:
            target = 0.0
        else:
            if not np.isfinite(mark[j]) or mark[j] <= 0:
                raise ValueError(f"target {w} for column {j} has no raw close to size it at")
            target = w * equity / mark[j]
            if whole:
                target = float(math.trunc(target))
        delta = target - q
        if delta > 0:
            buys.append((j, "BUY", delta))
        elif delta < 0:
            sells.append((j, "SELL", -delta))
    return sells + buys, (weights, position.copy(), mark, equity)


def _decision_gap(decision, traded, settled) -> float | None:
    """trader's ``max_target_deviation`` of one decision: after its fill bar, at t's close."""
    weights, position, mark, equity = decision
    compared = np.isfinite(weights) & ~settled
    if not compared.any() or not equity > 0:
        return None
    held = (position + traded) * np.nan_to_num(mark) / equity
    return float(np.max(np.abs(weights - held)[compared]))


def _orders_frame(rows) -> pd.DataFrame:
    """The ``RungResult.orders`` frame of ``(fill_bar, symbol, side, quantity, price, fee)`` rows."""
    return pd.DataFrame(rows, columns=["fill_bar", "symbol", "side", "quantity", "price", "fee"])


class _Statistics:
    """The report's per-rung statistics, by quantlab's public functions and the run's annualization."""

    def __init__(self, run: QuantlabRun, timestamps: pd.DatetimeIndex):
        self.rebalance_periods = run.rebalance_periods
        self.bar_interval = (
            pd.Series(np.diff(timestamps.values)).mode().iloc[0]
            if len(timestamps) > 1
            else pd.Timedelta(days=1)
        )
        self.year_freq = backtest_stats.year_freq(
            self.bar_interval, run.trading_days_per_year, run.session_minutes_per_day
        )

    def row(self, rung: RungResult) -> dict:
        """Return the report row of ``rung``: its statistics and execution counts."""
        equity = rung.equity
        previous = equity.shift(1)
        previous.iloc[0] = rung.init_cash
        returns = xr.DataArray(
            (equity / previous - 1.0).to_numpy(),
            dims="timestamp",
            coords={"timestamp": equity.index.values},
        )
        stats = backtest_stats.return_stats(
            returns, bar_interval=self.bar_interval, year_freq=self.year_freq
        )
        orders = rung.orders
        flows = (
            xr.Dataset(
                {
                    "timestamp": ("order", pd.DatetimeIndex(orders["fill_bar"]).values),
                    "size": ("order", orders["quantity"].to_numpy(dtype=np.float64)),
                    "price": ("order", orders["price"].to_numpy(dtype=np.float64)),
                }
            )
            if len(orders)
            else xr.Dataset()
        )
        turnover = backtest_stats.turnover_stats(
            backtest_stats.turnover(
                flows,
                xr.DataArray(equity.to_numpy(), dims="timestamp", coords={"timestamp": equity.index.values}),
                rung.init_cash,
            ),
            bar_interval=self.bar_interval,
            year_freq=self.year_freq,
            rebalance_periods=self.rebalance_periods,
        )
        return {
            "rung": rung.name,
            "final_equity": float(equity.iloc[-1]),
            "total_return_pct": stats["Total Return [%]"],
            "annualized_return_pct": stats["Annualized Return [%]"],
            "sharpe_ratio": stats["Sharpe Ratio"],
            "max_drawdown_pct": stats["Max Drawdown [%]"],
            "total_turnover_pct": turnover["Total Turnover [%]"],
            "annualized_turnover_pct": turnover["Annualized Turnover [%]"],
            "total_fees": float(orders["fee"].sum()) if len(orders) else 0.0,
            "orders": int(len(orders)),
            "rejected_orders": len(rung.rejected),
            "settlements": len(rung.settlements),
            "max_target_deviation": rung.max_target_deviation,
            "buys_capped": rung.buys_capped,
            "peak_cash_debit": rung.peak_cash_debit,
        }


def _rung_rows(rungs: dict[str, RungResult], statistics: _Statistics) -> list[dict]:
    """One row per rung, in ladder order, each with its delta to the rung above."""
    rows, above = [], None
    for name in RUNGS:
        if name not in rungs:
            continue
        row = statistics.row(rungs[name])
        row["description"] = RUNG_DESCRIPTIONS[name]
        if above is not None:
            row["delta"] = {
                key: _delta(row[key], above[key])
                for key in row
                if key not in ("rung", "description", "delta")
            }
        rows.append(row)
        above = row
    return rows


def _delta(value, above):
    """``value - above``, or ``None`` when either is missing."""
    if value is None or above is None:
        return None
    return value - above


def _trader_run(run: QuantlabRun, execution: ExecutionConfig, loop: str, output_dir: Path) -> Path:
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


def _trader_rung(
    trader_dir: Path, market: _Market, execution: ExecutionConfig, name: str = "T"
) -> RungResult:
    """A trader run directory (T, or the closed loop) in the report's terms."""
    timestamps = market.timestamps
    with xr.open_zarr(trader_dir / "equity.zarr") as equity:
        value = equity["value"].to_pandas().reindex(timestamps)
    with xr.open_zarr(trader_dir / "orders.zarr") as frame:
        table = frame.load().to_dataframe()
    metrics = json.loads((trader_dir / "metrics.json").read_text())
    events = json.loads((trader_dir / "events.json").read_text())["events"]
    fill_position = timestamps.searchsorted(pd.DatetimeIndex(table["decision_date"]), side="right")
    has_fill_bar = fill_position < len(timestamps)
    fill_bar = pd.DatetimeIndex(
        [timestamps[k] if k < len(timestamps) else pd.NaT for k in fill_position]
    )
    filled = (table["filled_quantity"] > 0).to_numpy()
    orders = pd.DataFrame(
        {
            "fill_bar": fill_bar[filled],
            "symbol": table["symbol"].to_numpy()[filled],
            "side": table["side"].to_numpy(dtype=str)[filled],
            "quantity": table["filled_quantity"].to_numpy(dtype=np.float64)[filled],
            "price": table["fill_price"].to_numpy(dtype=np.float64)[filled],
            "fee": table["fee"].to_numpy(dtype=np.float64)[filled],
        }
    )
    unfilled = table["status"].isin(UNFILLED_STATUSES).to_numpy() & has_fill_bar
    execution_block = metrics["execution"]
    # Holdings after each bar: the strategy's fills plus the venue fills.
    changes = [
        (bar, symbol, quantity if side == "BUY" else -quantity)
        for bar, symbol, side, quantity in zip(
            orders["fill_bar"], orders["symbol"], orders["side"], orders["quantity"]
        )
    ] + [
        (pd.Timestamp(e["timestamp"]), e["symbol"], (1.0 if e["side"] == "BUY" else -1.0) * e["quantity"])
        for e in events
        if e.get("type") == "corporate_action" and "side" in e
    ]
    positions = (
        pd.DataFrame(changes, columns=["timestamp", "symbol", "change"])
        .pivot_table(index="timestamp", columns="symbol", values="change", aggfunc="sum")
        .reindex(timestamps)
        .fillna(0.0)
        .cumsum()
    )
    return RungResult(
        name=name,
        run_dir=trader_dir,
        positions=positions,
        equity=value,
        init_cash=float(execution.init_cash),
        orders=orders,
        rejected=list(zip(fill_bar[unfilled], table["symbol"].to_numpy()[unfilled])),
        settlements=[
            (
                pd.Timestamp(e["timestamp"]),
                e["symbol"],
                float(e["price"]),
                float(e["quantity"] if e["side"] == "SELL" else -e["quantity"]),
            )
            for e in events
            if e.get("type") == "corporate_action" and e.get("action") == "DELIST"
        ],
        max_target_deviation=execution_block.get("max_target_deviation"),
        buys_capped=None,
        peak_cash_debit=float(execution_block["trader"]["peak_cash_debit"]),
    )


def _t_check(t: RungResult, l5: RungResult, trader_dir: Path) -> dict:
    """Check T against L5 (ADR 0007).

    The same orders, fill prices at 4 decimals, fees to the cent,
    rejections and settlements; equity within USD 0.01 per fill so far.

    The fills counted are the next-open orders' and the venue fills
    (splits, delisting settlements) of the trader run.
    """
    t_orders, l5_orders = _sorted_orders(t.orders), _sorted_orders(l5.orders)
    keys = ["fill_bar", "symbol", "side", "quantity"]
    orders_equal = len(t_orders) == len(l5_orders) and all(
        (t_orders[k].to_numpy() == l5_orders[k].to_numpy()).all() for k in keys
    )
    price_error = fee_error = None
    if orders_equal and len(t_orders):
        price_error = float(
            np.max(np.abs(t_orders["price"].round(_PRICE_DECIMALS) - l5_orders["price"].round(_PRICE_DECIMALS)))
        )
        fee_error = float(np.max(np.abs(t_orders["fee"] - l5_orders["fee"])))
    rejections_equal = _keyed(t.rejected) == _keyed(l5.rejected)
    settlements_equal = _keyed_settlements(t.settlements) == _keyed_settlements(l5.settlements)

    events = json.loads((trader_dir / "events.json").read_text())["events"]
    fill_days = list(pd.DatetimeIndex(t.orders["fill_bar"])) + [
        pd.Timestamp(e["timestamp"])
        for e in events
        if e.get("type") == "corporate_action" and "side" in e
    ]
    timestamps = t.equity.index
    fills = np.zeros(len(timestamps))
    np.add.at(fills, timestamps.searchsorted(pd.DatetimeIndex(fill_days)), 1)
    tolerance = 0.01 * np.cumsum(fills)
    error = np.abs(t.equity.to_numpy() - l5.equity.reindex(timestamps).to_numpy())
    worst = int(np.nanargmax(error)) if np.isfinite(error).any() else 0
    equity_ok = bool(np.isfinite(error).all() and (error <= tolerance + _EQUITY_SLACK).all())
    passed = (
        orders_equal
        and (price_error is None or price_error == 0.0)
        and (fee_error is None or fee_error < 0.005)
        and rejections_equal
        and settlements_equal
        and equity_ok
    )
    return {
        "passed": bool(passed),
        "orders_equal": bool(orders_equal),
        "orders": [len(t_orders), len(l5_orders)],
        "fill_price_max_error": price_error,
        "fee_max_error": fee_error,
        "rejections_equal": bool(rejections_equal),
        "settlements_equal": bool(settlements_equal),
        "equity_within_tolerance": equity_ok,
        "equity_max_error": float(error[worst]),
        "equity_tolerance_at_max": float(tolerance[worst]),
    }


#: Float slack on the equity tolerance (USD), for a bar with no fill yet.
_EQUITY_SLACK = 1e-6


def _sorted_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Orders keyed for comparison: symbols as strings, whole quantities, in a fixed order."""
    if not len(orders):
        return pd.DataFrame(columns=["fill_bar", "symbol", "side", "quantity", "price", "fee"])
    keyed = orders.assign(
        symbol=orders["symbol"].map(str),
        quantity=orders["quantity"].round(6),
    )
    return keyed.sort_values(["fill_bar", "symbol", "side"], kind="stable").reset_index(drop=True)


def _keyed(pairs) -> list:
    """``(bar, symbol)`` pairs as sorted comparable keys."""
    return sorted((pd.Timestamp(bar), str(symbol)) for bar, symbol in pairs)


def _keyed_settlements(settlements) -> list:
    """Settlements as sorted keys: bar, symbol, price at 4 decimals, whole quantity."""
    return sorted(
        (pd.Timestamp(bar), str(symbol), round(float(price), _PRICE_DECIMALS), round(float(q), 6))
        for bar, symbol, price, q in settlements
    )


#: Rules whose decision does not depend on holdings when nothing is locked
#: and nothing holds (ADR 0007): TopN, and mean-variance without a turnover
#: penalty; by class path, with the config test.
_HOLDING_INDEPENDENT_RULES = {
    "quantlab.portfolio.predefined.top_n.TopNConstructor": lambda config: True,
    "quantlab.portfolio.predefined.mean_variance.MeanVarianceOptimizer": lambda config: float(
        config.get("turnover_penalty", 0.0)
    )
    == 0.0,
}


def _locked(run: QuantlabRun, positions: pd.DataFrame, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Per bar: does the book hold a position its dataset marks not tradable (a locked one)?"""
    held = positions.columns[(positions.abs() > 1e-9).any(axis=0)]
    if not len(held):
        return np.zeros(len(timestamps), dtype=bool)
    dataset = run.price_dataset
    prices = dataset.panel(timestamps[0], timestamps[-1], symbols=list(held)).load()
    tradable = (
        dataset.tradable_bars(prices, run.market["fill_price_column"])
        .transpose("timestamp", "symbol")
        .to_pandas()
        .reindex(index=timestamps, columns=held, fill_value=False)
    )
    holding = positions[held].abs() > 1e-9
    return (holding & ~tradable.astype(bool)).any(axis=1).to_numpy()


def _closed_vs_open(
    run: QuantlabRun,
    table: xr.DataArray,
    market: _Market,
    rungs: dict[str, RungResult],
    closed: RungResult,
    closed_dir: Path,
    statistics: _Statistics,
) -> dict:
    """The closed-versus-open block: decided weights against ``weights.zarr``, runs against T.

    A rebalance bar of the closed loop is **holding-independent** when the
    rule is one of ``_HOLDING_INDEPENDENT_RULES``, neither book holds (the
    closed loop's hold event, an all-NaN row of the table) and neither book
    has a locked position (held after the bar, not tradable at it): the
    quantlab run's book is L0's, the closed loop's is its trader run's. The
    weight distance of a bar is the L1 norm of the difference, a NaN
    (keep) counting as 0.
    """
    with xr.open_zarr(closed_dir / "decisions.zarr") as decisions:
        decided = decisions["weight"].transpose("timestamp", "symbol").load()
    events = json.loads((closed_dir / "events.json").read_text())["events"]
    held_bars = {pd.Timestamp(e["timestamp"]) for e in events if e.get("type") == "hold"}
    constructor = run.config.get("constructor") or {}
    is_independent_config = _HOLDING_INDEPENDENT_RULES.get(constructor.get("name"))
    rule_independent = bool(is_independent_config and is_independent_config(constructor))

    timestamps = market.timestamps
    table = table.to_pandas()
    closed_rows = decided.to_pandas()
    symbols = table.columns.union(closed_rows.columns)
    locked_bars = _locked(run, rungs["L0"].positions, timestamps) | _locked(
        run, closed.positions, timestamps
    )
    compared = independent = equal = independent_equal = 0
    distances = []
    for t in closed_rows.index.intersection(timestamps):
        i = timestamps.get_loc(t)
        a = closed_rows.loc[t].reindex(symbols).to_numpy(dtype=np.float64)
        b = table.loc[t].reindex(symbols).to_numpy(dtype=np.float64)
        same = bool(np.array_equal(a, b, equal_nan=True))
        holds = t in held_bars or bool(np.isnan(table.loc[t].to_numpy()).all())
        is_independent = rule_independent and not holds and not locked_bars[i]
        compared += 1
        equal += same
        independent += is_independent
        independent_equal += is_independent and same
        distances.append(float(np.abs(np.nan_to_num(a) - np.nan_to_num(b)).sum()))

    t_rung = rungs["T"]
    t_orders, c_orders = _sorted_orders(t_rung.orders), _sorted_orders(closed.orders)
    orders_equal = len(t_orders) == len(c_orders) and bool(
        (t_orders.to_numpy() == c_orders.to_numpy()).all()
    )
    closed_row, open_row = statistics.row(closed), statistics.row(t_rung)
    return {
        "rule": constructor.get("name"),
        "holding_independent_rule": rule_independent,
        "rebalance_bars_compared": compared,
        "bars_equal": equal,
        "holding_independent_bars": independent,
        "holding_independent_bars_equal": independent_equal,
        "max_weight_l1_distance": max(distances) if distances else None,
        "mean_weight_l1_distance": float(np.mean(distances)) if distances else None,
        "orders_equal": orders_equal,
        "max_equity_difference": float(
            np.max(np.abs(closed.equity.to_numpy() - t_rung.equity.to_numpy()))
        ),
        "closed_loop": closed_row,
        "delta": {
            key: _delta(closed_row[key], open_row[key])
            for key in closed_row
            if key != "rung"
        },
    }


def _l0_check(l0: RungResult, run_value: xr.DataArray) -> dict:
    """L0 against the run's ``equity.zarr``: the largest relative error per bar."""
    expected = run_value.to_pandas()
    got = l0.equity.reindex(expected.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.abs(got.to_numpy() - expected.to_numpy()) / np.abs(expected.to_numpy())
    worst = float(np.nanmax(error)) if np.isfinite(got).all() else math.inf
    return {"passed": bool(worst <= L0_RTOL), "max_rel_error": worst, "tolerance": L0_RTOL}


def _report_dir(run: QuantlabRun, output_dir) -> Path:
    """Return the path of a fresh report directory (not created)."""
    output_dir = Path(output_dir) if output_dir is not None else run.run_dir.parent
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return output_dir / f"{run.run_dir.name}_parity_{stamp}"


def _write(parity_dir: Path, report: dict, rungs: dict, closed: RungResult | None) -> Path:
    """Write ``parity.json`` and ``parity.zarr`` into the report directory."""
    names = [name for name in RUNGS if name in rungs]
    timestamps = rungs[names[0]].equity.index
    equity = np.array([rungs[name].equity.reindex(timestamps).to_numpy() for name in names])
    variables = {"equity": (("rung", "timestamp"), equity)}
    if closed is not None:
        variables["closed_equity"] = ("timestamp", closed.equity.reindex(timestamps).to_numpy())
    for name in names:
        orders = rungs[name].orders
        if not len(orders):
            continue
        dim = f"{name}_order"
        variables.update(
            {
                f"{name}_fill_bar": (dim, pd.DatetimeIndex(orders["fill_bar"]).values),
                f"{name}_symbol": (dim, np.asarray(orders["symbol"].tolist())),
                f"{name}_side": (dim, orders["side"].to_numpy(dtype=str)),
                f"{name}_quantity": (dim, orders["quantity"].to_numpy(dtype=np.float64)),
                f"{name}_price": (dim, orders["price"].to_numpy(dtype=np.float64)),
                f"{name}_fee": (dim, orders["fee"].to_numpy(dtype=np.float64)),
            }
        )
    xr.Dataset(variables, coords={"rung": names, "timestamp": timestamps.values}).to_zarr(
        parity_dir / "parity.zarr", mode="w"
    )
    (parity_dir / "parity.json").write_text(json.dumps(_jsonable(report), indent=2))
    return parity_dir


def _jsonable(value):
    """Return ``value`` as strict JSON values (NaN and infinities as ``None``)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value
