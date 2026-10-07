"""L2: the reference ledger with vectorbt's execution, cash left uncapped."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab_ibkr.parity.market import Market, ffill, shift
from quantlab_ibkr.parity.rung import RungResult, orders_frame
from quantlab_ibkr.quantlab_run import QuantlabRun


#: vectorbt's closeness tolerances (``vectorbt.utils.math_``).
VBT_REL, VBT_ABS = 1e-9, 1e-12


def _is_close(a: float, b: float) -> bool:
    """vectorbt's ``is_close_nb``."""
    if a == b:
        return True
    return abs(a - b) <= max(VBT_REL * max(abs(a), abs(b)), VBT_ABS)


def _add(a: float, b: float) -> float:
    """vectorbt's ``add_nb``: ``a + b``, snapped to 0 when they cancel."""
    if np.sign(a) != np.sign(b) and _is_close(abs(a), abs(b)):
        return 0.0
    if np.sign(a) == np.sign(b) and _is_close(a + b, 0.0):
        return 0.0
    return a + b


def vectorbt_ledger(market: Market, run: QuantlabRun, name: str) -> RungResult:
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

    Examples
    --------
    L2 of a run, its buys a cash cap would have cut counted::

        run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
        table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
        market = Market.load(run, table)
        l2 = vectorbt_ledger(market, run, "L2")
        l2.buys_capped, l2.peak_cash_debit
    """
    timestamps, symbols = market.timestamps, market.symbols
    n_bars, n_symbols = market.weights.shape
    target = shift(market.weights)
    fill, valuation = ffill(market.fill), ffill(market.valuation)
    sizing = shift(valuation)
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
        orders=orders_frame(orders),
        rejected=rejections,
        settlements=settlements,
        max_target_deviation=deviation,
        buys_capped=capped,
        peak_cash_debit=max(0.0, -float(cash_path.min())),
    )
