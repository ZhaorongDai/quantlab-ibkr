"""The decision core: target weights to whole-share next-open orders (ADR 0003, ADR 0008).

``DecisionCycle.run(inputs, positions, cash)`` is the one cycle that runs after
the close of every bar, in the backtest and live alike. It marks equity at t's
raw close, asks its ``TargetSource`` for the bar's target weights and sizes
each finite target into ``trunc(w * equity / close) - position`` shares, sells
first. A NaN target keeps the holding. Closed and open loop differ only in the
target source.

This module imports no nautilus code: it is tested on plain arrays.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab_trader.base.venue import DecisionInputs, NextOpenOrder


@dataclass(frozen=True)
class Decision:
    """One bar's targets: weights to hold after the bar, or a hold with its reason.

    trader's record of a decision, with the fields of quantlab's constructor
    decision (weights, failure, events), so the closed loop's constructor
    targets and the open loop's table rows reach the cycle and the run
    directory in one shape.

    Attributes
    ----------
    weights : pandas.Series
        Target weight per PERMNO; NaN keeps the holding.
    failure : str or None
        Why the bar was held, or ``None``.
    events : tuple of dict
        The rule's events of this bar.

    Examples
    --------
    >>> Decision(pd.Series({10001: 0.5})).failure is None
    True
    """

    weights: pd.Series
    failure: str | None = None
    events: tuple[dict, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CycleResult:
    """What one decision cycle produced.

    Attributes
    ----------
    decision : Decision or None
        The bar's decision; ``None`` on a bar that does not rebalance.
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
    >>> TableTargets(table).row(pd.Timestamp("2024-01-03")).weights.tolist()
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
        return Decision(self._table.loc[t].astype(float))

    def targets(
        self, inputs: DecisionInputs, current_weights: pd.Series
    ) -> Decision | None:
        """Return the table's row at the decision date; holdings play no part."""
        return self.row(inputs.timestamp)


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
            self._size(decision.weights, inputs, held, equity),
            equity,
            current_weights,
        )

    @staticmethod
    def _size(
        weights: pd.Series,
        inputs: DecisionInputs,
        held: Mapping[Hashable, int],
        equity: float,
    ) -> tuple[NextOpenOrder, ...]:
        """Size each finite target into a whole-share order; sells first."""
        sells, buys = [], []
        for permno, weight in weights.items():
            if not _finite(weight):
                continue
            position = held.get(permno, 0)
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
