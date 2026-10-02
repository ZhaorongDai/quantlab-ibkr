"""`DecisionCycle.run`: target weights to whole-share next-open orders, on plain arrays.

What is locked here (ADR 0003, ADR 0008):

- sizing is ``trunc(w * equity / close) - position`` from t's raw close;
- equity is ``cash + sum(qty * close)``, marked at t's raw close;
- a NaN weight keeps the holding (no order), a symbol absent from the row too;
- sells come before buys;
- a bar without targets (not a rebalance bar) yields no decision and no orders.
"""

import math

import numpy as np
import pandas as pd
import pytest

from quantlab_trader.base.venue import DecisionInputs, NextOpenOrder
from quantlab_trader.decision import DecisionCycle, TableTargets

T = pd.Timestamp("2024-01-03")


def _inputs(close: dict, t=T) -> DecisionInputs:
    close = pd.Series(close, dtype=float)
    return DecisionInputs(
        timestamp=t,
        predictions=None,
        tradable=close.notna(),
        close=close,
        valuation_history=None,
        delisted=pd.Series(False, index=close.index),
    )


def _table(rows: dict) -> pd.DataFrame:
    """A rebalance table: timestamps on the index, PERMNOs on the columns."""
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("timestamp")


def _cycle(weights: dict, t=T) -> DecisionCycle:
    return DecisionCycle(TableTargets(_table({t: weights})))


def test_sizes_whole_shares_from_the_close_and_close_marked_equity():
    # equity = 1000 cash + 10 * 50 = 1500; 10001: trunc(0.5 * 1500 / 30) = 25,
    # 10002: trunc(0.5 * 1500 / 50) = 15 -> sell 10 - 15 = buy 5.
    result = _cycle({10001: 0.5, 10002: 0.5}).run(
        _inputs({10001: 30.0, 10002: 50.0}), positions={10002: 10}, cash=1000.0
    )

    assert result.equity == 1500.0
    assert result.orders == (
        NextOpenOrder(10001, "BUY", 25, T),
        NextOpenOrder(10002, "BUY", 5, T),
    )


def test_truncates_toward_zero_for_both_sides():
    # equity 1000; long 0.5 at 33 -> 15.15 -> 15; short -0.5 at 33 -> -15.15 -> -15.
    result = _cycle({10001: 0.5, 10002: -0.5}).run(
        _inputs({10001: 33.0, 10002: 33.0}), positions={}, cash=1000.0
    )

    assert result.orders == (
        NextOpenOrder(10002, "SELL", 15, T),
        NextOpenOrder(10001, "BUY", 15, T),
    )


def test_sells_come_before_buys_whatever_the_symbol_order():
    result = _cycle({10001: 1.0, 10002: 0.0}).run(
        _inputs({10001: 10.0, 10002: 10.0}), positions={10002: 100}, cash=0.0
    )

    assert [o.side for o in result.orders] == ["SELL", "BUY"]
    assert result.orders == (
        NextOpenOrder(10002, "SELL", 100, T),
        NextOpenOrder(10001, "BUY", 100, T),
    )


def test_a_nan_weight_or_an_absent_symbol_keeps_the_holding():
    result = _cycle({10001: np.nan, 10002: 0.0}).run(
        _inputs({10001: 10.0, 10002: 20.0, 10003: 5.0}),
        positions={10001: 7, 10002: 3, 10003: 4},
        cash=100.0,
    )

    assert result.orders == (NextOpenOrder(10002, "SELL", 3, T),)
    assert math.isnan(result.decision.weights[10001])


def test_no_order_when_the_target_equals_the_position():
    result = _cycle({10001: 0.5}).run(
        _inputs({10001: 10.0}), positions={10001: 50}, cash=500.0
    )

    assert result.orders == ()


def test_a_bar_without_targets_has_no_decision_and_no_orders():
    result = _cycle({10001: 1.0}, t=pd.Timestamp("2024-01-02")).run(
        _inputs({10001: 10.0}), positions={10001: 5}, cash=50.0
    )

    assert result.decision is None
    assert result.orders == ()
    assert result.equity == 100.0


def test_current_weights_are_the_close_marked_holdings_over_equity():
    result = _cycle({10001: np.nan}).run(
        _inputs({10001: 10.0, 10002: 20.0}),
        positions={10001: 30, 10002: -10},
        cash=200.0,
    )

    # equity = 200 + 300 - 200 = 300
    assert result.equity == 300.0
    assert result.current_weights.to_dict() == {10001: 1.0, 10002: -200.0 / 300.0}


def test_a_held_security_without_a_close_is_refused():
    with pytest.raises(ValueError, match="10002"):
        _cycle({10001: 0.5}).run(
            _inputs({10001: 10.0, 10002: np.nan}), positions={10002: 1}, cash=10.0
        )


def test_a_nonzero_target_without_a_close_is_refused():
    with pytest.raises(ValueError, match="10002"):
        _cycle({10002: 0.5}).run(
            _inputs({10001: 10.0, 10002: np.nan}), positions={}, cash=10.0
        )
