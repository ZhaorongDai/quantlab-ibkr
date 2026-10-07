"""`DecisionCycle.run`: target weights to whole-share next-open orders, on plain arrays.

What is locked here (ADR 0003, ADR 0008):

- sizing is ``trunc(w * equity / close) - position`` from t's raw close;
- equity is ``cash + sum(qty * close)``, marked at t's raw close;
- a NaN weight keeps the holding (no order), a symbol absent from the row too;
- sells come before buys;
- a bar without targets (not a rebalance bar) yields no decision and no orders;
- a target equal to the holding's current weight (a locked position) keeps it;
- closed loop: ``ConstructorTargets`` decides the bars quantlab's
  ``DecisionInputs.rebalances`` names through its ``context`` and the rule's
  ``decide``, and a held security missing from the prediction row joins the
  context unpredicted (tradable where the price dataset has a fill price).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.portfolio.config import TopNConfig
from quantlab.portfolio.base import Decision
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.portfolio.decision_inputs import DecisionInputs
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab_ibkr.base.venue import BarInputs, NextOpenOrder
from quantlab_ibkr.decision import ConstructorTargets, DecisionCycle, TableTargets
from tests.quantlab_run_fixture import _crsp_dataset

T = pd.Timestamp("2024-01-03")


def _inputs(close: dict, t=T) -> BarInputs:
    close = pd.Series(close, dtype=float)
    return BarInputs(
        timestamp=t,
        predictions=None,
        close=close,
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
    assert math.isnan(result.decision.weights.sel(symbol=10001))


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


def test_an_account_without_positive_equity_is_refused():
    with pytest.raises(ValueError, match="equity"):
        _cycle({10001: 0.5}).run(_inputs({10001: 10.0}), positions={10001: -1}, cash=10.0)


def test_a_target_equal_to_the_current_weight_keeps_the_position():
    # equity 123.45 + 2 * 13.7 = 150.85; 2 * 13.7 / 150.85 * 150.85 / 13.7
    # is 1.9999999999999998, which trunc would size to 1 share.
    cycle = DecisionCycle(_RepeatCurrent())

    result = cycle.run(_inputs({10001: 13.7}), positions={10001: 2}, cash=123.45)

    assert result.orders == ()


class _RepeatCurrent(TableTargets):
    """A target source that keeps every holding at its current weight (a locked rule)."""

    def __init__(self):
        pass

    def targets(self, inputs, current_weights):
        return Decision(
            xr.DataArray(
                current_weights.to_numpy(), dims="symbol",
                coords={"symbol": current_weights.index},
            )
        )


def _constructor_targets(root, tradable: dict, anchor=T, end=None):
    """Top-1 targets on a CRSP store of the bars around ``T``; tradable at ``T`` per ``tradable``."""
    rule = TopNConstructor(TopNConfig(direction="long_only", top_n=1))
    rule.bind([LabelSpec("ret", "raw", 1, 1)])
    bars = pd.DatetimeIndex([T - pd.offsets.BDay(), T, T + pd.offsets.BDay()])
    open_ = {p: [10.0, 10.0 if ok else np.nan, 10.0] for p, ok in tradable.items()}
    close = {p: [10.0, 10.0, 10.0] for p in tradable}
    return ConstructorTargets(
        DecisionInputs(
            _crsp_dataset(Path(root), bars, open_, close), rule,
            fill_column="adjOpen", valuation_column="adjClose",
            rebalance_periods=1, anchor=anchor, end=end,
        )
    )


def _closed_inputs(close: dict, predictions: dict) -> BarInputs:
    close = pd.Series(close, dtype=float)
    return BarInputs(
        timestamp=T,
        predictions=xr.Dataset(
            {"ret": ("symbol", list(predictions.values()))},
            coords={"symbol": list(predictions)},
        ),
        close=close,
        delisted=pd.Series(False, index=close.index),
    )


def test_constructor_targets_decide_through_the_rule_on_a_rebalance_bar(tmp_path):
    cycle = DecisionCycle(_constructor_targets(tmp_path, {10001: True, 10002: True}))

    result = cycle.run(
        _closed_inputs({10001: 10.0, 10002: 20.0}, {10001: 0.1, 10002: 0.2}),
        positions={},
        cash=1000.0,
    )

    assert result.decision.weights.sel(symbol=10002).item() == 1.0
    assert result.orders == (NextOpenOrder(10002, "BUY", 50, T),)


def test_constructor_targets_skip_a_bar_that_does_not_rebalance(tmp_path):
    # T is the replay's last bar: an order decided there has no next bar to fill on.
    cycle = DecisionCycle(_constructor_targets(tmp_path, {10001: True}, anchor=T - pd.offsets.BDay(), end=T))

    result = cycle.run(_closed_inputs({10001: 10.0}, {10001: 0.1}), positions={}, cash=10.0)

    assert result.decision is None


def test_a_held_security_missing_from_the_prediction_row_is_decided_as_unpredicted(tmp_path):
    # 10003 is held but the panel has no row for it: tradable, it is sold;
    # without a fill price it would be locked and kept.
    close = {10001: 10.0, 10002: 20.0, 10003: 5.0}
    predictions = {10001: 0.1, 10002: 0.2}

    sold = DecisionCycle(_constructor_targets(tmp_path / "sold", {10001: True, 10002: True, 10003: True})).run(
        _closed_inputs(close, predictions), positions={10003: 100}, cash=500.0,
    )
    kept = DecisionCycle(_constructor_targets(tmp_path / "kept", {10001: True, 10002: True, 10003: False})).run(
        _closed_inputs(close, predictions), positions={10003: 100}, cash=500.0,
    )

    assert sold.orders == (
        NextOpenOrder(10003, "SELL", 100, T),
        NextOpenOrder(10002, "BUY", 50, T),
    )
    assert [o.permno for o in kept.orders] == [10002]
    assert kept.decision.weights.sel(symbol=10003).item() == 0.5


def test_a_hold_keeps_every_holding():
    class _Hold(TableTargets):
        def __init__(self):
            pass

        def targets(self, inputs, current_weights):
            return Decision(
                xr.DataArray([np.nan, np.nan], dims="symbol", coords={"symbol": [10001, 10002]}),
                failure="infeasible",
            )

    result = DecisionCycle(_Hold()).run(
        _inputs({10001: 10.0, 10002: 20.0}), positions={10001: 5, 10002: 3}, cash=100.0
    )

    assert result.decision.failure == "infeasible"
    assert result.orders == ()
