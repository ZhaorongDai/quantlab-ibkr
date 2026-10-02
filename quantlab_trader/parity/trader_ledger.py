"""L3-L5: the reference ledger with trader's execution conventions, one switch at a time.

An independent numpy oracle of the nautilus run, never called by the
strategy: it re-implements the corporate-action and sizing arithmetic
rather than calling the venue's, sharing only the tolerances of the
classification (``FACTOR_RTOL``, ``PRICE_FACTOR_RTOL``,
``IMPLIED_SPLIT_TOL``) and the pure cost functions
``IbkrFixedFeeModel.charge`` and ``slipped_price``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
from nautilus_trader.model.enums import OrderSide

from quantlab_trader.parity.market import MONEY_DECIMALS, PRICE_DECIMALS, Market, ffill, shift
from quantlab_trader.parity.nautilus_money import NautilusMoney, nautilus_money
from quantlab_trader.parity.rung import RungResult, orders_frame
from quantlab_trader.venue.backtest.corporate_actions import (
    FACTOR_RTOL,
    IMPLIED_SPLIT_TOL,
    PRICE_FACTOR_RTOL,
)
from quantlab_trader.venue.backtest.fees import IbkrFixedFeeModel
from quantlab_trader.venue.backtest.fills import slipped_price


@dataclass(frozen=True)
class Conventions:
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

    Examples
    --------
    >>> import dataclasses
    >>> l3 = Conventions(
    ...     whole_shares=False, trader_costs=False, fee_model="fraction",
    ...     fee_rate=0.001, slippage=0.0, init_cash=1e6,
    ... )
    >>> l4 = dataclasses.replace(l3, whole_shares=True)
    >>> l4.whole_shares, l4.trader_costs
    (True, False)
    """

    whole_shares: bool
    trader_costs: bool
    fee_model: str
    fee_rate: float
    slippage: float
    init_cash: float


#: Slack of a whole-share floor against binary rounding (300 * (1/3) is 99.99...).
_SHARE_EPS = 1e-9


def trader_ledger(market: Market, conventions: Conventions, name: str) -> RungResult:
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

    Examples
    --------
    L3 and L4 of a run::

        import dataclasses

        run = QuantlabRun.load("runs/WeightsVectorBt_20261001")
        table = run.rebalance_table()["weight"].transpose("timestamp", "symbol")
        market = Market.load(run, table)
        l3 = Conventions(
            whole_shares=False, trader_costs=False, fee_model="fraction",
            fee_rate=run.fees, slippage=run.slippage, init_cash=run.init_cash,
        )
        l4 = trader_ledger(market, dataclasses.replace(l3, whole_shares=True), "L4")
        l4.orders["quantity"].mod(1).eq(0).all()
    """
    timestamps, symbols = market.timestamps, market.symbols
    n_bars, n_symbols = market.weights.shape
    whole, costs = conventions.whole_shares, conventions.trader_costs
    money = (lambda x: round(float(x), MONEY_DECIMALS)) if costs else float

    close = market.close
    last_close = ffill(close)
    paired = np.isfinite(close) & np.isfinite(market.valuation)
    with np.errstate(divide="ignore", invalid="ignore"):
        grown = (
            ffill(np.where(paired, close, np.nan))
            * market.valuation
            / ffill(np.where(paired, market.valuation, np.nan))
        )
    last_value = np.where(np.isfinite(grown) & (grown >= 0), grown, last_close)
    if costs:
        last_value = np.round(last_value, PRICE_DECIMALS)
    mark = ffill(np.where(market.delisted, last_value, close))

    split_factor = market.split_factor
    with np.errstate(divide="ignore", invalid="ignore"):
        share_factor = shift(ffill(market.cumfacshr)) / market.cumfacshr
    dividend = np.where(np.isfinite(market.dividend), market.dividend, 0.0)
    pre_close = shift(last_close)
    kind = _factor_days(split_factor, share_factor)
    kind[0] = ""
    kind, holder = _holder_days(market, kind, dividend, pre_close)

    book = NautilusMoney(conventions.init_cash, n_symbols) if costs else None
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
        orders=orders_frame(orders),
        rejected=rejections,
        settlements=settlements,
        max_target_deviation=deviation,
        buys_capped=capped,
        peak_cash_debit=max(0.0, -float(cash_path.min())),
    )


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
    market: Market, kind: np.ndarray, dividend: np.ndarray, pre_close: np.ndarray
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
    factor exists, else a split by x when x lies below
    ``1 / (1 + IMPLIED_SPLIT_TOL)``, the reverse-split shape only (#28: a
    share increase would preserve value lost in a collapse); a final event (k = 0) stays the
    delisting path's. Returns ``kind`` with those bars
    marked ``"SPLIT"`` and the holder factor of every split (NaN elsewhere).
    """
    close, adj, k = market.close, market.adj_close, market.split_factor
    n_bars, n_symbols = close.shape
    kind = kind.copy()
    holder = np.where(kind == "SPLIT", k, np.nan)
    units, paid = np.ones(n_symbols), np.zeros(n_symbols)
    anchor_close, anchor_adj = np.full(n_symbols, np.nan), np.full(n_symbols, np.nan)
    reverse_split_bound = 1.0 / (1.0 + IMPLIED_SPLIT_TOL)
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
                implied = x < reverse_split_bound
                by_factor = open_ & agrees
                by_prices = open_ & ~agrees & implied
            holder[i] = np.where(by_factor, k[i], np.where(by_prices, x, holder[i]))
            kind[i] = np.where(by_factor | by_prices, "SPLIT", kind[i])
            units = np.where((kind[i] == "SPLIT") & ~priced, units * k[i], units)
        anchor_close = np.where(priced, close[i], anchor_close)
        anchor_adj = np.where(priced, adj[i], anchor_adj)
        units, paid = np.where(priced, 1.0, units), np.where(priced, 0.0, paid)
    return kind, holder


def _fill(open_price: float, side: str, quantity: float, conventions: Conventions) -> tuple[float, float]:
    """Return the fill price and fee of an order at the open ``open_price``."""
    if not conventions.trader_costs:
        slip = conventions.slippage
        price = open_price * (1 + slip if side == "BUY" else 1 - slip)
        return price, conventions.fee_rate * quantity * price
    printed = Decimal(repr(round(float(open_price), PRICE_DECIMALS)))
    order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
    if conventions.slippage:
        printed = slipped_price(printed, order_side, conventions.slippage, PRICE_DECIMALS)
    price = float(printed)
    if conventions.fee_model == "ibkr_fixed":
        fee = float(IbkrFixedFeeModel.charge(order_side, int(quantity), printed))
    else:
        fee = nautilus_money(quantity * price * conventions.fee_rate)
    return price, fee


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
