"""The decision core: target weights to whole-share next-open orders (ADR 0003, ADR 0008).

``DecisionCycle.run(inputs, positions, cash)`` is the one cycle that runs after
the close of every bar, in the backtest and live alike. It marks equity at t's
raw close, asks its ``TargetSource`` for the bar's target weights and sizes
each finite target into ``trunc(w * equity / close) - position`` shares, sells
first. A NaN target keeps the holding, and so does a target equal to the
holding's current weight (a locked position the rule kept). Closed and open
loop differ only in the target source: ``ConstructorTargets`` asks quantlab's
``DecisionInputs`` whether the bar rebalances and for its context, then runs
the rule's ``decide``; ``TableTargets`` reads a ``weights.zarr`` row. Both
return quantlab's ``Decision``.

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

from quantlab.portfolio.base import Decision
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab_ibkr._support.jsonable import python_scalar
from quantlab_ibkr.base.venue import BarInputs, NextOpenOrder


@dataclass(frozen=True)
class CycleResult:
    """What one decision cycle produced.

    Attributes
    ----------
    decision : quantlab.portfolio.base.Decision or None
        The bar's decision (weights on ``symbol``, all NaN to hold, the
        failure that made it a hold, the rule's events); ``None`` on a bar
        that does not rebalance.
    orders : tuple of NextOpenOrder
        The sized orders, sells first.
    equity : float
        Cash plus every holding at t's raw close.
    current_weights : pandas.Series
        Each holding's close value over equity, per PERMNO.

    Examples
    --------
    >>> t = pd.Timestamp("2024-01-03")
    >>> cycle = DecisionCycle(TableTargets(pd.DataFrame({10001: [0.5]}, index=[t])))
    >>> close = pd.Series({10001: 30.0})
    >>> result = cycle.run(
    ...     BarInputs(timestamp=t, predictions=None, close=close, delisted=close.isna()),
    ...     positions={10001: 10}, cash=700.0,
    ... )
    >>> result.equity, result.current_weights.to_dict(), [o.quantity for o in result.orders]
    (1000.0, {10001: 0.3}, [6])
    """

    decision: Decision | None
    orders: tuple[NextOpenOrder, ...]
    equity: float
    current_weights: pd.Series


class TargetSource(ABC):
    """Where a decision cycle gets its target weights.

    Examples
    --------
    Open loop reads a rebalance table, closed loop runs quantlab's rule:

    >>> issubclass(TableTargets, TargetSource), issubclass(ConstructorTargets, TargetSource)
    (True, True)
    """

    @abstractmethod
    def targets(
        self, inputs: BarInputs, current_weights: pd.Series
    ) -> Decision | None:
        """Return the bar's decision, or ``None`` when the bar does not rebalance.

        Parameters
        ----------
        inputs : BarInputs
            What is known at the close of t.
        current_weights : pandas.Series
            Each holding's weight at t's raw close, per PERMNO.

        Examples
        --------
        >>> t = pd.Timestamp("2024-01-03")
        >>> source = TableTargets(pd.DataFrame({10001: [1.0]}, index=[t]))
        >>> close = pd.Series({10001: 30.0})
        >>> inputs = BarInputs(timestamp=t, predictions=None, close=close, delisted=close.isna())
        >>> source.targets(inputs, pd.Series(dtype=float)).weights.values.tolist()
        [1.0]
        """


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
        """Return the table's row at ``t`` as a decision, or ``None`` without one.

        Examples
        --------
        >>> t = pd.Timestamp("2024-01-03")
        >>> table = pd.DataFrame({10001: [0.5], 10002: [np.nan]}, index=[t])
        >>> TableTargets(table).row(t).weights.to_series().to_dict()
        {10001: 0.5, 10002: nan}
        """
        if t not in self._table.index:
            return None
        row = self._table.loc[t].astype(float)
        return Decision(
            weights=xr.DataArray(row.to_numpy(), dims="symbol", coords={"symbol": row.index})
        )

    def targets(
        self, inputs: BarInputs, current_weights: pd.Series
    ) -> Decision | None:
        """Return the table's row at the decision date; holdings play no part.

        Examples
        --------
        >>> t = pd.Timestamp("2024-01-03")
        >>> source = TableTargets(pd.DataFrame({10001: [0.5]}, index=[t]))
        >>> close = pd.Series({10001: 30.0})
        >>> inputs = BarInputs(timestamp=t, predictions=None, close=close, delisted=close.isna())
        >>> source.targets(inputs, pd.Series({10001: 0.2})).weights.values.tolist()
        [0.5]
        """
        return self.row(inputs.timestamp)


class ConstructorTargets(TargetSource):
    """Closed loop: quantlab's constructor decides each rebalance bar on the account's holdings.

    quantlab's ``DecisionInputs`` (``QuantlabRun.decision_inputs``) holds the
    run's bound rule, price dataset and rebalance schedule. On a bar it
    ``rebalances``, its ``context`` is built from the bar's prediction row
    and the holdings' current weights (tradability, the decision-price
    window and staleness read from the price dataset), and the rule's
    ``decide`` returns the decision. A held security the prediction row
    lacks joins the context with NaN predictions, so the rule treats it like
    any held name without a prediction: tradable where the dataset says so,
    locked otherwise.

    Parameters
    ----------
    decision_inputs : quantlab.portfolio.decision_inputs.DecisionInputs
        The run's decision inputs.

    Examples
    --------
    quantlab's top-1 rule, rebalancing every other bar:

    >>> from quantlab.portfolio.config import TopNConfig
    >>> from quantlab.dataset.memory import FrameDataset
    >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
    >>> bars = pd.bdate_range("2024-01-02", periods=3)
    >>> price = xr.DataArray(
    ...     [[30.0, 40.0]] * 3, dims=("timestamp", "symbol"),
    ...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
    ... )
    >>> source = ConstructorTargets(DecisionInputs(
    ...     FrameDataset(xr.Dataset({"adjOpen": price, "adjClose": price})),
    ...     TopNConstructor(TopNConfig(direction="long_only", top_n=1)),
    ...     fill_column="adjOpen", valuation_column="adjClose",
    ...     rebalance_periods=2, anchor=bars[0],
    ... ))
    >>> row = xr.Dataset({"ret": ("symbol", [0.3, 0.1])}, coords={"symbol": ["AAA", "BBB"]})
    >>> close = pd.Series({"AAA": 30.0, "BBB": 40.0})
    >>> inputs = BarInputs(timestamp=bars[0], predictions=row, close=close, delisted=close.isna())
    >>> source.targets(inputs, pd.Series(dtype=float)).weights.values.tolist()
    [1.0, 0.0]
    """

    def __init__(self, decision_inputs: DecisionInputs):
        self.decision_inputs = decision_inputs

    def targets(
        self, inputs: BarInputs, current_weights: pd.Series
    ) -> Decision | None:
        """Return the rule's decision on a rebalance bar, ``None`` on any other.

        Parameters
        ----------
        inputs : BarInputs
            What the venue knows at the close of t, with the bar's prediction row.
        current_weights : pandas.Series
            Each holding's weight at t's raw close, per PERMNO.

        Raises
        ------
        ValueError
            If a rebalance bar's inputs carry no prediction row.

        Examples
        --------
        A source rebalancing every other bar, on its second bar:

        >>> from quantlab.portfolio.config import TopNConfig
        >>> from quantlab.dataset.memory import FrameDataset
        >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
        >>> bars = pd.bdate_range("2024-01-02", periods=3)
        >>> price = xr.DataArray(
        ...     [[30.0]] * 3, dims=("timestamp", "symbol"), coords={"timestamp": bars, "symbol": ["AAA"]}
        ... )
        >>> source = ConstructorTargets(DecisionInputs(
        ...     FrameDataset(xr.Dataset({"adjOpen": price, "adjClose": price})),
        ...     TopNConstructor(TopNConfig(direction="long_only", top_n=1)),
        ...     fill_column="adjOpen", valuation_column="adjClose",
        ...     rebalance_periods=2, anchor=bars[0],
        ... ))
        >>> close = pd.Series({"AAA": 30.0})
        >>> later = BarInputs(timestamp=bars[1], predictions=None, close=close, delisted=close.isna())
        >>> source.targets(later, pd.Series({"AAA": 1.0})) is None
        True
        """
        t = inputs.timestamp
        if not self.decision_inputs.rebalances(t):
            return None
        if inputs.predictions is None:
            raise ValueError(f"ConstructorTargets at {t.date()}: the bar has no prediction row")
        return self.decide(t, inputs.predictions, current_weights)

    def decide(
        self, t: pd.Timestamp, predictions: xr.Dataset, current_weights: pd.Series
    ) -> Decision:
        """Run the rule on the context of ``t`` built from a prediction row and holdings.

        The one context the closed loop decides from, and the one the
        parity report's **Decision recheck** rebuilds from a run's recorded
        current weights. The holdings are handed to quantlab in PERMNO
        order, so a held security without a prediction takes the same place
        in the context whatever order the account listed it in.

        Parameters
        ----------
        t : pandas.Timestamp
            The decision date, a rebalance bar.
        predictions : xarray.Dataset
            The bar's prediction row, one variable per label on ``symbol``.
        current_weights : pandas.Series
            Each holding's weight at t's raw close, per PERMNO; a zero or
            NaN entry is not held.

        Returns
        -------
        quantlab.portfolio.base.Decision
            The rule's decision on the context's symbols.

        Examples
        --------
        quantlab's top-1 rule on a book holding BBB:

        >>> from quantlab.portfolio.config import TopNConfig
        >>> from quantlab.dataset.memory import FrameDataset
        >>> from quantlab.portfolio.predefined.top_n import TopNConstructor
        >>> bars = pd.bdate_range("2024-01-02", periods=2)
        >>> price = xr.DataArray(
        ...     [[30.0, 40.0]] * 2, dims=("timestamp", "symbol"),
        ...     coords={"timestamp": bars, "symbol": ["AAA", "BBB"]},
        ... )
        >>> source = ConstructorTargets(DecisionInputs(
        ...     FrameDataset(xr.Dataset({"adjOpen": price, "adjClose": price})),
        ...     TopNConstructor(TopNConfig(direction="long_only", top_n=1)),
        ...     fill_column="adjOpen", valuation_column="adjClose",
        ...     rebalance_periods=1, anchor=bars[0],
        ... ))
        >>> row = xr.Dataset({"ret": ("symbol", [0.3, 0.1])}, coords={"symbol": ["AAA", "BBB"]})
        >>> source.decide(bars[0], row, pd.Series({"BBB": 1.0})).weights.values.tolist()
        [1.0, 0.0]
        """
        held = current_weights[current_weights.notna() & (current_weights != 0.0)].sort_index()
        context = self.decision_inputs.context(
            t,
            predictions,
            xr.DataArray(held.to_numpy(dtype=float), dims="symbol", coords={"symbol": held.index}),
        )
        return self.decision_inputs.constructor.decide(context)


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
    >>> inputs = BarInputs(timestamp=t, predictions=None, close=close, delisted=close.isna())
    >>> cycle.run(inputs, positions={}, cash=1000.0).orders
    (NextOpenOrder(permno=10001, side='BUY', quantity=33, decision_date=Timestamp('2024-01-03 00:00:00')),)
    """

    def __init__(self, targets: TargetSource):
        self.targets = targets

    def run(
        self,
        inputs: BarInputs,
        positions: Mapping[Hashable, int],
        cash: float,
    ) -> CycleResult:
        """Run the cycle at the close of ``inputs.timestamp``.

        Parameters
        ----------
        inputs : BarInputs
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

        Examples
        --------
        Moving a 20-share holding to half of a 1,000 equity, sells first:

        >>> t = pd.Timestamp("2024-01-03")
        >>> table = pd.DataFrame({10001: [0.0], 10002: [0.5]}, index=[t])
        >>> close = pd.Series({10001: 25.0, 10002: 40.0})
        >>> inputs = BarInputs(timestamp=t, predictions=None, close=close, delisted=close.isna())
        >>> result = DecisionCycle(TableTargets(table)).run(inputs, {10001: 20}, cash=500.0)
        >>> [(o.permno, o.side, o.quantity) for o in result.orders]
        [(10001, 'SELL', 20), (10002, 'BUY', 12)]
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
        inputs: BarInputs,
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
