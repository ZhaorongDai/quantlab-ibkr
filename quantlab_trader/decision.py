"""The decision core: target weights to whole-share next-open orders (ADR 0003, ADR 0008).

``DecisionCycle.run(inputs, positions, cash)`` is the one cycle that runs after
the close of every bar, in the backtest and live alike. It marks equity at t's
raw close, asks its ``TargetSource`` for the bar's target weights and sizes
each finite target into ``trunc(w * equity / close) - position`` shares, sells
first. A NaN target keeps the holding, and so does a target equal to the
holding's current weight (a locked position the rule kept). Closed and open
loop differ only in the target source: ``ConstructorTargets`` runs quantlab's
``build_context`` + ``decide`` on the rebalance calendar, ``TableTargets``
reads a ``weights.zarr`` row. Both return quantlab's ``Decision``.

This module imports no nautilus code: it is tested on plain arrays.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.portfolio import Decision, PortfolioConstructor
from quantlab_trader._support.jsonable import python_scalar
from quantlab_trader.base.venue import DecisionInputs, NextOpenOrder
from quantlab_trader.calendar import RebalanceCalendar


@dataclass(frozen=True)
class CycleResult:
    """What one decision cycle produced.

    Attributes
    ----------
    decision : quantlab.base.portfolio.Decision or None
        The bar's decision (weights on ``symbol``, all NaN to hold, the
        failure that made it a hold, the rule's events); ``None`` on a bar
        that does not rebalance.
    orders : tuple of NextOpenOrder
        The sized orders, sells first.
    equity : float
        Cash plus every holding at t's raw close.
    current_weights : pandas.Series
        Each holding's close value over equity, per PERMNO.
    """

    decision: Decision | None
    orders: tuple[NextOpenOrder, ...]
    equity: float
    current_weights: pd.Series


class TargetSource(ABC):
    """Where a decision cycle gets its target weights."""

    @abstractmethod
    def targets(
        self, inputs: DecisionInputs, current_weights: pd.Series
    ) -> Decision | None:
        """Return the bar's decision, or ``None`` when the bar does not rebalance."""


class TableTargets(TargetSource):
    """Open loop: the rows of a finished rebalance table, executed as they are.

    Parameters
    ----------
    table : pandas.DataFrame
        Target weights, decision dates on the index and PERMNOs on the
        columns (quantlab's ``weights.zarr`` as a frame). A NaN keeps the
        holding.

    Examples
    --------
    >>> table = pd.DataFrame({10001: [0.5]}, index=[pd.Timestamp("2024-01-03")])
    >>> TableTargets(table).row(pd.Timestamp("2024-01-03")).weights.values.tolist()
    [0.5]
    >>> TableTargets(table).row(pd.Timestamp("2024-01-04")) is None
    True
    """

    def __init__(self, table: pd.DataFrame):
        self._table = table

    def row(self, t: pd.Timestamp) -> Decision | None:
        """Return the table's row at ``t`` as a decision, or ``None`` without one."""
        if t not in self._table.index:
            return None
        row = self._table.loc[t].astype(float)
        return Decision(
            weights=xr.DataArray(row.to_numpy(), dims="symbol", coords={"symbol": row.index})
        )

    def targets(
        self, inputs: DecisionInputs, current_weights: pd.Series
    ) -> Decision | None:
        """Return the table's row at the decision date; holdings play no part."""
        return self.row(inputs.timestamp)


class ConstructorTargets(TargetSource):
    """Closed loop: quantlab's constructor decides each rebalance bar on the account's holdings.

    On a rebalance bar of ``calendar`` it hands the rule's ``build_context``
    the bar's prediction row, tradability and decision-price history from
    ``inputs`` and the holdings' current weights, then returns the rule's
    ``decide``. A held security the prediction row lacks (live, a daily
    panel without it) joins the context with NaN predictions, so the rule
    treats it like any held name without a prediction; it is tradable where
    the inputs say so and locked otherwise.

    Parameters
    ----------
    constructor : quantlab.base.portfolio.PortfolioConstructor
        The bound rule (``load_constructor``).
    calendar : RebalanceCalendar
        Which bars rebalance.
    """

    def __init__(self, constructor: PortfolioConstructor, calendar: RebalanceCalendar):
        self.constructor = constructor
        self.calendar = calendar

    def targets(
        self, inputs: DecisionInputs, current_weights: pd.Series
    ) -> Decision | None:
        """Return the rule's decision on a rebalance bar, ``None`` on any other."""
        t = inputs.timestamp
        if not self.calendar.rebalances(t):
            return None
        if inputs.predictions is None:
            raise ValueError(f"ConstructorTargets at {t.date()}: the bar has no prediction row")
        predictions = inputs.predictions.drop_vars("timestamp", errors="ignore")
        held = current_weights[current_weights != 0.0]
        symbols = pd.Index(predictions["symbol"].values)
        absent = held.index[~held.index.isin(symbols)]
        if len(absent):
            symbols = symbols.append(pd.Index(absent))
            predictions = predictions.reindex(symbol=symbols)
        tradable = inputs.tradable.reindex(symbols, fill_value=False).astype(bool)
        context = self.constructor.build_context(
            t,
            predictions,
            xr.DataArray(tradable.to_numpy(), dims="symbol", coords={"symbol": symbols}),
            xr.DataArray(held.to_numpy(dtype=float), dims="symbol", coords={"symbol": held.index}),
            valuation_price=inputs.valuation_history,
        )
        return self.constructor.decide(context)


class DecisionCycle:
    """Turn one bar's targets and the account into next-open orders.

    Parameters
    ----------
    targets : TargetSource
        Where the bar's target weights come from.

    Examples
    --------
    >>> t = pd.Timestamp("2024-01-03")
    >>> cycle = DecisionCycle(TableTargets(pd.DataFrame({10001: [1.0]}, index=[t])))
    >>> close = pd.Series({10001: 30.0})
    >>> inputs = DecisionInputs(t, None, close.notna(), close, None, close.isna())
    >>> cycle.run(inputs, positions={}, cash=1000.0).orders
    (NextOpenOrder(permno=10001, side='BUY', quantity=33, decision_date=Timestamp('2024-01-03 00:00:00')),)
    """

    def __init__(self, targets: TargetSource):
        self.targets = targets

    def run(
        self,
        inputs: DecisionInputs,
        positions: Mapping[Hashable, int],
        cash: float,
    ) -> CycleResult:
        """Run the cycle at the close of ``inputs.timestamp``.

        Parameters
        ----------
        inputs : DecisionInputs
            What is known at the close of t.
        positions : Mapping
            Signed whole-share holdings per PERMNO; zero entries are ignored.
        cash : float
            The account's derived cash.

        Returns
        -------
        CycleResult
            The decision, the orders (sells first), equity and current weights.

        Raises
        ------
        ValueError
            If a holding, or a nonzero target, has no raw close to value or
            size it at, or equity is not positive.
        """
        held = {p: int(q) for p, q in positions.items() if int(q) != 0}
        close = inputs.close
        unpriced = [p for p in held if not _finite(close.get(p, np.nan))]
        if unpriced:
            raise ValueError(
                f"DecisionCycle at {inputs.timestamp.date()}: holdings {unpriced} "
                f"have no raw close to value them at"
            )
        equity = float(cash) + sum(q * float(close[p]) for p, q in held.items())
        if not equity > 0.0:
            raise ValueError(
                f"DecisionCycle at {inputs.timestamp.date()}: equity {equity} is "
                f"not positive; there is nothing to size targets against"
            )
        current_weights = pd.Series(
            {p: q * float(close[p]) / equity for p, q in held.items()}, dtype=float
        )
        decision = self.targets.targets(inputs, current_weights)
        if decision is None:
            return CycleResult(None, (), equity, current_weights)
        return CycleResult(
            decision,
            self._size(decision.weights, inputs, held, equity, current_weights),
            equity,
            current_weights,
        )

    @staticmethod
    def _size(
        weights: xr.DataArray,
        inputs: DecisionInputs,
        held: Mapping[Hashable, int],
        equity: float,
        current_weights: pd.Series,
    ) -> tuple[NextOpenOrder, ...]:
        """Size each finite target into a whole-share order; sells first.

        A target equal to the holding's current weight keeps the position:
        a rule keeps a locked position at exactly that weight, and sizing it
        through ``trunc`` could round a share away.
        """
        sells, buys = [], []
        for label, weight in zip(weights["symbol"].values, weights.values):
            if not _finite(weight):
                continue
            permno = python_scalar(label)
            position = held.get(permno, 0)
            if position and weight == current_weights.get(permno, np.nan):
                continue
            if weight == 0.0:
                target = 0
            else:
                price = inputs.close.get(permno, np.nan)
                if not _finite(price) or price <= 0.0:
                    raise ValueError(
                        f"DecisionCycle at {inputs.timestamp.date()}: target "
                        f"{weight} for {permno} has no raw close to size it at"
                    )
                target = math.trunc(weight * equity / float(price))
            delta = target - position
            if delta > 0:
                buys.append(NextOpenOrder(permno, "BUY", delta, inputs.timestamp))
            elif delta < 0:
                sells.append(NextOpenOrder(permno, "SELL", -delta, inputs.timestamp))
        return tuple(sells + buys)


def _finite(value) -> bool:
    """Return whether ``value`` is a finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
