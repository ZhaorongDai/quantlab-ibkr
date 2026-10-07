"""Holder share changes the prices imply (#27, #28, ADR 0009 amendments of 2026-10-02 and 2026-10-03).

At the ``runner.run(TraderConfig)`` seam. CRSP's ``adjClose`` is chained
from its total return ``ret``, so ``adjClose[t] / adjClose[p]``, with p the
last bar with a raw close, is what a holder's position earned. A day whose
raw prices disagree with that return by more than the factors explain is a
share change the venue did not book:

- a factor day whose ``splitFactor`` agrees with the return is booked as a
  holder split by ``splitFactor`` even when the share factor disagrees;
- a day without a usable factor (none at all, or one that disagrees with the
  return) whose price-implied factor
  ``x = (adjClose[t] / adjClose[p] * close[p] - divCash[t]) / close[t]``
  lies below ``1 / 1.2`` (a reverse split) is booked as an ``IMPLIED_SPLIT``:
  whole shares and cash in lieu as for a split, a queued order rescaled by x;
- any other disagreement (more than 1%), an x above 1.2 included (#28: a
  collapse with a missing return), is logged as ``MISMATCH`` and the raw
  price move is realised.

Raw prices (open = close), ``--`` no price, ``adjClose`` CRSP-chained:

======  ====  ====  ======================  ===========  ===========  =========  ====  ====
PERMNO  b0    b1    b2                      b3           b4           b5         b6    b7
======  ====  ====  ======================  ===========  ===========  =========  ====  ====
20001   0.50  0.50  30.0 [k .025, s .02]    30.0         30.3         30.0       30.6  30.9
20002   2.0   2.0   2.0                     21.0 [ret 5%] 21.2        21.0       21.4  21.6
20003   0.10  0.10  --                      --           --           5.0        5.1   5.2
                                                                      [ret NaN: adjClose flat]
20004   40    40.4  40.0                    40.8         41.0 [div .40] 40.6     41.2  41.5
20005   10    10    --                      11 [ret NaN] 11.1         11.0       11.2  11.3
======  ====  ====  ======================  ===========  ===========  =========  ====  ====

20001 is PERMNO 18217 of #27 (a reverse split whose share factor disagrees
with its price factor; CRSP's return is the price factor's), 20002 a reverse
split with no factor and a CRSP return that knows it (12350 2023-12-21),
20003 PERMNO 14051 (a halt, then about 1:50 with no factor and a NaN
return), 20004 the control (a dividend, no share change), 20005 a halt
with a NaN return and a 10% move: within the tolerance, logged only.

The table buys at the close of bar 0 (0.1001, 0.2001, 0.05011, 0.2, 0.1 of
100 000, fee 0.1%) and exits 20002 at the close of bar 2. The hand ledger:

- bar 1 open: +20020 @0.50, +10005 @2, +50110 @0.10, +500 @40.4, +1000 @10;
  notional 65 231, fees 10.01 + 20.01 + 5.01 + 20.20 + 10.00: cash 34 703.77.
- bar 2 09:30: 20001 splits by k = 0.025: 20020 -> 500, cash in lieu
  0.5 * 0.50 / 0.025 = 10: cash 34 713.77.
- bar 3 09:30: 20002's implied x = (2.1 / 2 * 2) / 21 = 0.1: 10005 -> 1000,
  cash in lieu 0.5 * 2 / 0.1 = 10; the queued sale of 10005 is rescaled to
  1000, sold @21, fee 21: cash 55 702.77. 20005's x = 10 / 11 is logged.
- bar 4 09:30: 20004's dividend 500 * 0.40 = 200: cash 55 902.77.
- bar 5 09:30: 20003's implied x = 0.10 / 5 = 0.02: 50110 -> 1002, cash in
  lieu 0.2 * 0.10 / 0.02 = 1: cash 55 903.77.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.runner import run
from quantlab_ibkr.venue.backtest.venue import BacktestVenueConfig
from tests.quantlab_run_fixture import build_quantlab_run

NAN = np.nan
BARS = pd.bdate_range("2024-01-02", periods=8)
PERMNOS = (20001, 20002, 20003, 20004, 20005)
CLOSE = {
    20001: [0.50, 0.50, 30.0, 30.0, 30.3, 30.0, 30.6, 30.9],
    20002: [2.0, 2.0, 2.0, 21.0, 21.2, 21.0, 21.4, 21.6],
    20003: [0.10, 0.10, NAN, NAN, NAN, 5.0, 5.1, 5.2],
    20004: [40.0, 40.4, 40.0, 40.8, 41.0, 40.6, 41.2, 41.5],
    20005: [10.0, 10.0, NAN, 11.0, 11.1, 11.0, 11.2, 11.3],
}
OPEN = CLOSE
_PRE_DIVIDEND = 41.0 / 41.4
ADJ_CLOSE = {
    20001: [0.5, 0.5] + [c * 0.025 for c in CLOSE[20001][2:]],
    20002: [2.0, 2.0, 2.0] + [c * 0.1 for c in CLOSE[20002][3:]],
    20003: [0.10, 0.10, NAN, NAN, NAN, 0.10, 0.102, 0.104],
    20004: [c * _PRE_DIVIDEND for c in CLOSE[20004][:4]] + CLOSE[20004][4:],
    20005: [10.0, 10.0, NAN] + [c * 10 / 11 for c in CLOSE[20005][3:]],
}


def _priced(value):
    return {p: [value if np.isfinite(c) else NAN for c in CLOSE[p]] for p in PERMNOS}


SPLIT_FACTOR = _priced(1.0)
SPLIT_FACTOR[20001][2] = 0.025
CUMFACSHR = _priced(1.0)
CUMFACSHR[20001] = [1.0, 1.0] + [50.0] * 6
DIV_CASH = _priced(0.0)
DIV_CASH[20004][4] = 0.40
WEIGHTS = {p: [NAN] * len(BARS) for p in PERMNOS}
for _p, _w in zip(PERMNOS, (0.1001, 0.2001, 0.05011, 0.2, 0.1)):
    WEIGHTS[_p][0] = _w
WEIGHTS[20002][2] = 0.0

#: The hand ledger: cash after each close.
CASH = [100_000.0, 34_703.77, 34_713.77, 55_702.77, 55_902.77, 55_903.77, 55_903.77, 55_903.77]
#: Shares of 20001..20005 after each close.
SHARES = [
    [0, 0, 0, 0, 0],
    [20020, 10005, 50110, 500, 1000],
    [500, 10005, 50110, 500, 1000],
    [500, 0, 50110, 500, 1000],
    [500, 0, 50110, 500, 1000],
    [500, 0, 1002, 500, 1000],
    [500, 0, 1002, 500, 1000],
    [500, 0, 1002, 500, 1000],
]
#: A halted holding is marked at its last raw close.
MARK = {p: list(pd.Series(CLOSE[p]).ffill()) for p in PERMNOS}


def _ledger_equity() -> list[float]:
    return [
        cash + sum(q * MARK[p][bar] for p, q in zip(PERMNOS, shares) if q)
        for bar, (cash, shares) in enumerate(zip(CASH, SHARES))
    ]


def build_run(root):
    return build_quantlab_run(
        root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        init_cash=100_000.0,
        variables={
            "adjClose": ADJ_CLOSE,
            "splitFactor": SPLIT_FACTOR,
            "cumfacshr": CUMFACSHR,
            "divCash": DIV_CASH,
        },
    )


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    root = tmp_path_factory.mktemp("implied_splits")
    return run(
        TraderConfig(
            quantlab_run=str(build_run(root)),
            venue=BacktestVenueConfig(),
            loop="open",
            output_dir=str(root / "trader"),
        )
    )


def test_equity_after_every_close_equals_the_hand_ledger(replay):
    equity = xr.open_zarr(replay / "equity.zarr").load()["value"].values

    np.testing.assert_allclose(equity, _ledger_equity(), rtol=0, atol=1e-6)


def _orders(run_dir) -> pd.DataFrame:
    return xr.open_zarr(run_dir / "orders.zarr").load().to_dataframe()


def _corporate_actions(run_dir) -> list[dict]:
    events = json.loads((run_dir / "events.json").read_text())["events"]
    return [e for e in events if e["type"] == "corporate_action"]


def test_share_counts_after_every_close_equal_the_hand_ledger(replay):
    changes = pd.DataFrame(0, index=BARS, columns=list(PERMNOS))
    for order in _orders(replay).itertuples():
        fill_bar = BARS[BARS.get_loc(order.decision_date) + 1]
        sign = 1 if order.side == "BUY" else -1
        changes.loc[fill_bar, order.symbol] += sign * order.filled_quantity
    for event in _corporate_actions(replay):
        if "side" in event:
            sign = 1 if event["side"] == "BUY" else -1
            changes.loc[pd.Timestamp(event["timestamp"]), event["symbol"]] += sign * event["quantity"]

    assert changes.cumsum().to_numpy().tolist() == SHARES


def test_a_next_open_order_queued_across_an_implied_split_is_rescaled(replay):
    sale = _orders(replay).query("symbol == 20002 and side == 'SELL'").iloc[0]

    assert (sale["quantity"], sale["filled_quantity"], sale["status"]) == (1000, 1000, "filled")
    assert sale["fill_price"] == 21.0


def test_the_share_changes_are_booked_and_the_mismatch_logged(replay):
    events = sorted(_corporate_actions(replay), key=lambda e: (e["timestamp"], e["symbol"], e["action"]))
    for event in events:
        for name in ("split_factor", "implied_factor"):
            if event.get(name) is not None:
                event[name] = round(event[name], 9)

    assert events == [
        {"type": "corporate_action", "action": "CASH_IN_LIEU", "timestamp": "2024-01-04",
         "symbol": 20001, "quantity": 20020, "amount": 10.0, "split_factor": 0.025},
        {"type": "corporate_action", "action": "SPLIT", "timestamp": "2024-01-04",
         "symbol": 20001, "side": "SELL", "quantity": 19520, "price": 0.0, "fee": 0.0},
        {"type": "corporate_action", "action": "CASH_IN_LIEU", "timestamp": "2024-01-05",
         "symbol": 20002, "quantity": 10005, "amount": 10.0, "split_factor": 0.1},
        {"type": "corporate_action", "action": "IMPLIED_SPLIT", "timestamp": "2024-01-05",
         "symbol": 20002, "side": "SELL", "quantity": 9005, "price": 0.0, "fee": 0.0},
        {"type": "corporate_action", "action": "MISMATCH", "timestamp": "2024-01-05",
         "symbol": 20005, "quantity": 1000, "amount": 0.0, "split_factor": 1.0,
         "share_factor": 1.0, "implied_factor": round(10 / 11, 9)},
        {"type": "corporate_action", "action": "DIVIDEND", "timestamp": "2024-01-08",
         "symbol": 20004, "quantity": 500, "amount": 200.0, "per_share": 0.4},
        {"type": "corporate_action", "action": "CASH_IN_LIEU", "timestamp": "2024-01-09",
         "symbol": 20003, "quantity": 50110, "amount": 1.0, "split_factor": 0.02},
        {"type": "corporate_action", "action": "IMPLIED_SPLIT", "timestamp": "2024-01-09",
         "symbol": 20003, "side": "SELL", "quantity": 49108, "price": 0.0, "fee": 0.0},
    ]


def test_metrics_count_implied_splits_apart_from_splits(replay):
    trader = json.loads((replay / "metrics.json").read_text())["execution"]["trader"]

    assert trader["splits"] == {"count": 1, "share_change": -19520}
    assert trader["implied_splits"] == {"count": 2, "share_change": -9005 - 49108}
    assert trader["cash_in_lieu"] == {"count": 3, "amount": pytest.approx(21.0)}


@pytest.fixture(scope="module")
def ladder(tmp_path_factory):
    from quantlab_ibkr.parity.ladder import parity

    root = tmp_path_factory.mktemp("implied_splits_parity")
    parity_dir = parity(build_run(root), output_dir=root / "parity")
    report = json.loads((parity_dir / "parity.json").read_text())
    with xr.open_zarr(parity_dir / "parity.zarr") as data:
        return report, data.load()


def test_the_reference_ledger_books_the_share_changes_and_t_equals_l5(ladder):
    report, data = ladder

    np.testing.assert_allclose(
        data["equity"].sel(rung="L5").values, _ledger_equity(), rtol=0, atol=1e-6
    )
    assert report["checks"]["T_equals_L5"]["passed"], report["checks"]["T_equals_L5"]


def test_a_split_on_a_row_without_a_price_is_not_booked_again_by_the_prices(tmp_path):
    """20006 is halted on bar 2, whose row carries a 2:1 split; it reopens at 25.5.

    adjClose (50, 50, --, 51) says the holder earned 2%, which the split
    booked on bar 2 already explains: 100 -> 200 shares, x = 51 / (2 * 25.5)
    = 1 on bar 3, nothing more booked. 20007 (100 @10, fee 1) trades on
    every bar, so bar 2 has data and its 09:30 actions are booked on it.
    Cash 10 000 - 5 000 - 5 - 1 000 - 1 = 3 994.
    """
    from quantlab_ibkr.parity.ladder import parity

    bars = BARS[:4]
    close = {20006: [50.0, 50.0, NAN, 25.5], 20007: [10.0] * 4}
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", bars, close, close,
        {20006: [0.5, NAN, NAN, NAN], 20007: [0.1, NAN, NAN, NAN]},
        variables={
            "adjClose": {20006: [50.0, 50.0, NAN, 51.0], 20007: [10.0] * 4},
            "splitFactor": {20006: [1.0, 1.0, 2.0, 1.0], 20007: [1.0] * 4},
            "cumfacshr": {20006: [2.0, 2.0, 1.0, 1.0], 20007: [1.0] * 4},
            "divCash": {20006: [0.0] * 4, 20007: [0.0] * 4},
        },
    )
    expected = [10_000.0] + [3_994.0 + 1_000 + q * c for q, c in ((100, 50), (200, 50), (200, 25.5))]
    run_dir = run(
        TraderConfig(
            quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
            output_dir=str(tmp_path / "trader"),
        )
    )
    parity_dir = parity(quantlab_run, output_dir=tmp_path / "parity")

    equity = xr.open_zarr(run_dir / "equity.zarr").load()["value"].values
    np.testing.assert_allclose(equity, expected, rtol=0, atol=1e-6)
    assert [e["action"] for e in _corporate_actions(run_dir)] == ["SPLIT"]
    with xr.open_zarr(parity_dir / "parity.zarr") as data:
        np.testing.assert_allclose(
            data["equity"].sel(rung="L5").values, expected, rtol=0, atol=1e-6
        )
    assert json.loads((parity_dir / "parity.json").read_text())["checks"]["T_equals_L5"]["passed"]


def test_a_distribution_on_a_row_without_an_adjusted_close_is_netted_alike(tmp_path):
    """20008 pays a value distribution (k 1.25) on bar 1, a row with a raw close
    of 6 but no adjClose; both the venue and the reference ledger pay it at
    that close and net it out of bar 2's implied factor, so T equals L5."""
    from quantlab_ibkr.parity.ladder import parity

    bars = BARS[:4]
    close = {20008: [10.0, 6.0, 8.2, 8.1], 20009: [10.0] * 4}
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", bars, close, close,
        {20008: [0.5, NAN, NAN, NAN], 20009: [0.1, NAN, NAN, NAN]},
        variables={
            "adjClose": {20008: [10.0, NAN, 11.422, 11.3], 20009: [10.0] * 4},
            "splitFactor": {20008: [1.0, 1.25, 1.0, 1.0], 20009: [1.0] * 4},
            "cumfacshr": {20008: [1.0] * 4, 20009: [1.0] * 4},
            "divCash": {20008: [0.0] * 4, 20009: [0.0] * 4},
        },
    )
    parity_dir = parity(quantlab_run, output_dir=tmp_path / "parity")

    assert json.loads((parity_dir / "parity.json").read_text())["checks"]["T_equals_L5"]["passed"]


def test_a_collapse_without_a_return_is_realised_and_logged_not_booked(tmp_path):
    """20010 halts on bar 2 and reopens at 0.50, down from 10, with a NaN
    return, so adjClose stays flat (10, 10, --, 10) and the implied factor is
    x = 10 / 0.5 = 20: a share increase that would preserve value the holder
    lost (#28). Only the reverse-split shape (x below 1 / 1.2) is booked; this
    day is logged as MISMATCH, the 100 shares stay 100 and the loss is
    realised at the raw price. 20011 (100 @10, fee 1) trades on every bar.
    Cash 10 000 - 1 000 - 1 - 1 000 - 1 = 7 998.
    """
    from quantlab_ibkr.parity.ladder import parity

    bars = BARS[:4]
    close = {20010: [10.0, 10.0, NAN, 0.5], 20011: [10.0] * 4}
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", bars, close, close,
        {20010: [0.1, NAN, NAN, NAN], 20011: [0.1, NAN, NAN, NAN]},
        variables={
            "adjClose": {20010: [10.0, 10.0, NAN, 10.0], 20011: [10.0] * 4},
            "splitFactor": {20010: [1.0, 1.0, NAN, 1.0], 20011: [1.0] * 4},
            "cumfacshr": {20010: [1.0, 1.0, NAN, 1.0], 20011: [1.0] * 4},
            "divCash": {20010: [0.0, 0.0, NAN, 0.0], 20011: [0.0] * 4},
        },
    )
    expected = [10_000.0, 9_998.0, 9_998.0, 7_998.0 + 100 * 0.5 + 1_000]
    run_dir = run(
        TraderConfig(
            quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
            output_dir=str(tmp_path / "trader"),
        )
    )
    parity_dir = parity(quantlab_run, output_dir=tmp_path / "parity")

    equity = xr.open_zarr(run_dir / "equity.zarr").load()["value"].values
    np.testing.assert_allclose(equity, expected, rtol=0, atol=1e-6)
    assert _corporate_actions(run_dir) == [
        {"type": "corporate_action", "action": "MISMATCH", "timestamp": "2024-01-05",
         "symbol": 20010, "quantity": 100, "amount": 0.0, "split_factor": 1.0,
         "share_factor": 1.0, "implied_factor": 20.0},
    ]
    trader = json.loads((run_dir / "metrics.json").read_text())["execution"]["trader"]
    assert trader["implied_splits"] == {"count": 0, "share_change": 0}
    with xr.open_zarr(parity_dir / "parity.zarr") as data:
        np.testing.assert_allclose(
            data["equity"].sel(rung="L5").values, expected, rtol=0, atol=1e-6
        )
    assert json.loads((parity_dir / "parity.json").read_text())["checks"]["T_equals_L5"]["passed"]
