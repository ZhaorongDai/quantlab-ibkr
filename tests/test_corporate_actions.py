"""Corporate actions at the ``runner.run(TraderConfig)`` seam (ADR 0009 and its amendment).

The backtest venue books, at 09:30 ET of the ex-date, a holder split as a
price-0 venue fill of ``floor(q * k) - q`` plus cash in lieu at the pre-split
close / k, a value distribution as cash ``q * (k - 1) * close[t]`` on top of
that day's ``divCash``, a dividend as ``divCash * signed_qty``; a final event
(``k = 0``) is left to the delisting path and any other factor day is logged.
A next-open order queued across a split is rescaled by k (floor).

Raw prices (open/close), corporate-action days in brackets; ``k`` is
``splitFactor``, the share factor ``cumfacshr[t-1] / cumfacshr[t]``:

======  ===========  ===========  ============  ==============  ===============  ===========  ==============  ============
PERMNO  bar 0        bar 1        bar 2         bar 3           bar 4            bar 5        bar 6           bar 7
======  ===========  ===========  ============  ==============  ===============  ===========  ==============  ============
10001   49 / 50      50 / 51      52 / 52       26.5 / 27 [2:1] 27 / 27.5        27.5 / 28    28 / 28         28 / 28.5
10002   39 / 40      40 / 41      41 / 42       42 / 42         126 / 127 [1:3]  127 / 128    128 / 129 [0.9]  129 / 130
10003   16 / 16      16 / 15.5    15 / 15       10.2 / 10 [3:2] 10 / 10.4        10.5 / 10.6  10.6 / 10.8     10.8 / 11
                                                                                 [div 0.10]
10004   99 / 100     100 / 101    101 / 100     100 / 102       102 / 104        85 / 84      84 / 85         85 / 86
                                  [div 0.50]                                     [k 1.25, share 1, div 0.25]
10005   19.5 / 20    20 / 20.5    20.5 / 21     21 / 21.2       21.2 / 21.4      21.4 / 21.5  -- [k 0, final]  --
10006   9.8 / 10     10 / 10.2    10.2 / 10.4   10.4 / 10.5     -- [delisting]   --           --              --
======  ===========  ===========  ============  ==============  ===============  ===========  ==============  ============

The rebalance table buys at the close of bar 0 (weights 0.2, 0.2, -0.1, 0.2,
0.1, -0.1 of 100 000, fee 0.1%) and exits 10001 at the close of bar 2; every
other row keeps. 10005's bar 6 and 10006's bar 4 are CRSP delisting rows with
an adjusted close carrying the delisting return (+10% and -20%): last
valuations 21.5 * 1.1 = 23.65 and 10.5 * 0.8 = 8.40.

The hand ledger (cash at each close; shares 10001..10006):

- bar 0: nothing held, cash 100 000.
- bar 1 open: +400 @50, +500 @40, -625 @16, +200 @100, +500 @20, -1000 @10;
  notional 90 000, fees 90: cash 49 910.
- bar 2 09:30: 10004's dividend 200 * 0.50 = +100: cash 50 010. Decide: sell
  400 of 10001 (pre-split shares).
- bar 3 09:30: 10001 splits 2:1, 400 -> 800 (no cash in lieu); the short
  10003 splits 3:2, -625 -> floor(937.5) = -937, cash in lieu
  -0.5 * 15 / 1.5 = -5.00. The queued exit is rescaled to 800 and sold @26.5,
  fee 21.20: cash 71 183.80.
- bar 4 09:30: 10002's 1:3 reverse split, 500 -> 166, cash in lieu
  (500 / 3 - 166) * 42 * 3 = 84.00: cash 71 267.80. 10006 is marked at 8.40.
- bar 5 09:30: the short 10006 is settled, buy 1000 @8.40 (-8 400); 10003's
  dividend -937 * 0.10 = -93.70; 10004's spin-off 200 * 0.25 * 84 = 4 200
  plus its dividend 200 * 0.25 = 50: cash 62 774.10 + 4 250 = 67 024.10.
- bar 6 09:30: 10005's final event (k = 0) and 10002's unmatched factor day
  (k = 0.9, share factor 1) change nothing and are logged. 10005 is marked at
  23.65 on its delisting bar.
- bar 7 09:30: 10005 is settled, sell 500 @23.65: cash 78 849.10.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab_ibkr.base.config import TraderConfig
from quantlab_ibkr.runner import run
from quantlab_ibkr.venue.backtest.venue import BacktestVenueConfig
from tests.quantlab_run_fixture import ADJUSTED_SCALE, build_quantlab_run

NAN = np.nan
BARS = pd.bdate_range("2024-01-02", periods=8)
PERMNOS = (10001, 10002, 10003, 10004, 10005, 10006)
OPEN = {
    10001: [49, 50, 52, 26.5, 27, 27.5, 28, 28],
    10002: [39, 40, 41, 42, 126, 127, 128, 129],
    10003: [16, 16, 15, 10.2, 10, 10.5, 10.6, 10.8],
    10004: [99, 100, 101, 100, 102, 85, 84, 85],
    10005: [19.5, 20, 20.5, 21, 21.2, 21.4, NAN, NAN],
    10006: [9.8, 10, 10.2, 10.4, NAN, NAN, NAN, NAN],
}
CLOSE = {
    10001: [50, 51, 52, 27, 27.5, 28, 28, 28.5],
    10002: [40, 41, 42, 42, 127, 128, 129, 130],
    10003: [16, 15.5, 15, 10, 10.4, 10.6, 10.8, 11],
    10004: [100, 101, 100, 102, 104, 84, 85, 86],
    10005: [20, 20.5, 21, 21.2, 21.4, 21.5, NAN, NAN],
    10006: [10, 10.2, 10.4, 10.5, NAN, NAN, NAN, NAN],
}
ADJ_CLOSE = {p: [c * ADJUSTED_SCALE for c in CLOSE[p]] for p in PERMNOS}
ADJ_CLOSE[10005][6] = 21.5 * 1.1 * ADJUSTED_SCALE
ADJ_CLOSE[10006][4] = 10.5 * 0.8 * ADJUSTED_SCALE


def _ones():
    return {p: [1.0] * len(BARS) for p in PERMNOS}


SPLIT_FACTOR = _ones()
SPLIT_FACTOR[10001][3] = 2.0
SPLIT_FACTOR[10002][4] = 1 / 3
SPLIT_FACTOR[10002][6] = 0.9
SPLIT_FACTOR[10003][3] = 1.5
SPLIT_FACTOR[10004][5] = 1.25
SPLIT_FACTOR[10005][6] = 0.0
CUMFACSHR = {
    10001: [2, 2, 2, 1, 1, 1, 1, 1],
    10002: [1, 1, 1, 1, 3, 3, 3, 3],
    10003: [1.5, 1.5, 1.5, 1, 1, 1, 1, 1],
    10004: [1.0] * 8,
    10005: [1, 1, 1, 1, 1, 1, 0, NAN],
    10006: [1, 1, 1, 1, 1, NAN, NAN, NAN],
}
DIV_CASH = {p: [0.0] * len(BARS) for p in PERMNOS}
DIV_CASH[10003][5] = 0.10
DIV_CASH[10004][2] = 0.50
DIV_CASH[10004][5] = 0.25
WEIGHTS = {p: [NAN] * len(BARS) for p in PERMNOS}
for _p, _w in zip(PERMNOS, (0.2, 0.2, -0.1, 0.2, 0.1, -0.1)):
    WEIGHTS[_p][0] = _w
WEIGHTS[10001][2] = 0.0

#: The hand ledger: cash after each close.
CASH = [100_000.0, 49_910.0, 50_010.0, 71_183.80, 71_267.80, 67_024.10, 67_024.10, 78_849.10]
#: Shares of 10001..10006 after each close.
SHARES = [
    [0, 0, 0, 0, 0, 0],
    [400, 500, -625, 200, 500, -1000],
    [400, 500, -625, 200, 500, -1000],
    [0, 500, -937, 200, 500, -1000],
    [0, 166, -937, 200, 500, -1000],
    [0, 166, -937, 200, 500, 0],
    [0, 166, -937, 200, 500, 0],
    [0, 166, -937, 200, 0, 0],
]
#: The raw close each holding is marked at (a delisting bar at its last valuation).
MARK = {**CLOSE, 10005: CLOSE[10005][:6] + [23.65, NAN], 10006: CLOSE[10006][:4] + [8.40] + [NAN] * 3}


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    root = tmp_path_factory.mktemp("corporate_actions")
    quantlab_run = build_quantlab_run(
        root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        init_cash=100_000.0,
        variables={
            "adjClose": ADJ_CLOSE,
            "splitFactor": SPLIT_FACTOR,
            "cumfacshr": CUMFACSHR,
            "divCash": DIV_CASH,
        },
    )
    config = TraderConfig(
        quantlab_run=str(quantlab_run),
        venue=BacktestVenueConfig(),
        loop="open",
        output_dir=str(root / "trader"),
    )
    return run(config)


def _ledger_equity() -> list[float]:
    equity = []
    for bar, (cash, shares) in enumerate(zip(CASH, SHARES)):
        held = sum(q * MARK[p][bar] for p, q in zip(PERMNOS, shares) if q)
        equity.append(cash + held)
    return equity


def test_equity_after_every_close_equals_the_hand_ledger(replay):
    equity = xr.open_zarr(replay / "equity.zarr").load()["value"].values

    np.testing.assert_allclose(equity, _ledger_equity(), rtol=0, atol=1e-6)


def _orders(run_dir) -> pd.DataFrame:
    return xr.open_zarr(run_dir / "orders.zarr").load().to_dataframe()


def _corporate_actions(run_dir) -> list[dict]:
    events = json.loads((run_dir / "events.json").read_text())["events"]
    return [e for e in events if e["type"] == "corporate_action"]


def test_share_counts_after_every_close_equal_the_hand_ledger(replay):
    # Every share change is a next-open fill (orders.zarr, filled at the open
    # after its decision date) or a venue fill (events.json).
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


def test_a_next_open_order_queued_across_a_split_is_rescaled(replay):
    exit_order = _orders(replay).query("symbol == 10001 and side == 'SELL'").iloc[0]

    assert exit_order["quantity"] == 800
    assert exit_order["filled_quantity"] == 800
    assert exit_order["status"] == "filled"
    assert exit_order["fill_price"] == 26.5
    assert "400" in exit_order["reason"]


def _by_day(events: list[dict]) -> list[dict]:
    # Actions of one 09:30 are booked in no particular order.
    return sorted(events, key=lambda e: (e["timestamp"], e["symbol"], e["action"]))


def test_corporate_actions_are_events_with_zero_fees(replay):
    assert _by_day(_corporate_actions(replay)) == _by_day([
        {"type": "corporate_action", "action": "DIVIDEND", "timestamp": "2024-01-04",
         "symbol": 10004, "quantity": 200, "amount": 100.0, "per_share": 0.5},
        {"type": "corporate_action", "action": "SPLIT", "timestamp": "2024-01-05",
         "symbol": 10001, "side": "BUY", "quantity": 400, "price": 0.0, "fee": 0.0},
        {"type": "corporate_action", "action": "SPLIT", "timestamp": "2024-01-05",
         "symbol": 10003, "side": "SELL", "quantity": 312, "price": 0.0, "fee": 0.0},
        {"type": "corporate_action", "action": "CASH_IN_LIEU", "timestamp": "2024-01-05",
         "symbol": 10003, "quantity": -625, "amount": -5.0, "split_factor": 1.5},
        {"type": "corporate_action", "action": "SPLIT", "timestamp": "2024-01-08",
         "symbol": 10002, "side": "SELL", "quantity": 334, "price": 0.0, "fee": 0.0},
        {"type": "corporate_action", "action": "CASH_IN_LIEU", "timestamp": "2024-01-08",
         "symbol": 10002, "quantity": 500, "amount": 84.0, "split_factor": 1 / 3},
        {"type": "corporate_action", "action": "DELIST", "timestamp": "2024-01-09",
         "symbol": 10006, "side": "BUY", "quantity": 1000, "price": 8.4, "fee": 0.0},
        {"type": "corporate_action", "action": "DIVIDEND", "timestamp": "2024-01-09",
         "symbol": 10003, "quantity": -937, "amount": -93.7, "per_share": 0.1},
        {"type": "corporate_action", "action": "DIVIDEND", "timestamp": "2024-01-09",
         "symbol": 10004, "quantity": 200, "amount": 50.0, "per_share": 0.25},
        {"type": "corporate_action", "action": "DISTRIBUTION", "timestamp": "2024-01-09",
         "symbol": 10004, "quantity": 200, "amount": 4_200.0, "split_factor": 1.25},
        {"type": "corporate_action", "action": "OTHER", "timestamp": "2024-01-10",
         "symbol": 10002, "quantity": 166, "amount": 0.0, "split_factor": 0.9,
         "share_factor": 1.0, "implied_factor": 1.0},
        {"type": "corporate_action", "action": "FINAL", "timestamp": "2024-01-10",
         "symbol": 10005, "quantity": 500, "amount": 0.0, "split_factor": 0.0,
         "share_factor": None, "implied_factor": None},
        {"type": "corporate_action", "action": "DELIST", "timestamp": "2024-01-11",
         "symbol": 10005, "side": "SELL", "quantity": 500, "price": 23.65, "fee": 0.0},
    ])


def test_corporate_action_fills_are_not_in_the_orders(replay):
    orders = _orders(replay)

    # Six entries at bar 1 and the rescaled exit: nothing the venue booked.
    assert len(orders) == 7
    assert orders["status"].tolist() == ["filled"] * 7


@pytest.mark.parametrize("missing", ["splitFactor", "cumfacshr", "divCash"])
def test_a_price_dataset_without_corporate_action_fields_is_refused(tmp_path, missing):
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", BARS[:3],
        {10001: OPEN[10001][:3]}, {10001: CLOSE[10001][:3]}, {10001: [0.5, NAN, NAN]},
        drop_variables=(missing,),
    )
    config = TraderConfig(
        quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
        output_dir=str(tmp_path / "trader"),
    )

    with pytest.raises(ValueError, match=missing):
        run(config)
    assert not (tmp_path / "trader").exists()


def test_a_split_after_a_halt_with_a_dividend_and_an_order_rescaled_to_nothing(tmp_path):
    """10001 is halted on bar 2 and goes 1:3 on bar 3 with a 0.30 dividend; 10002
    goes 1:3 on bar 3 too, across a queued sale of 2 pre-split shares.

    - bar 1 open: +100 of 10001 @50 and +100 of 10002 @25, fees 7.50: cash 2 492.50.
    - close of bar 2: 10001 has no price (marked 51); sell 2 of 10002 (target
      trunc(0.2584 * 10 292.50 / 27) = 98).
    - bar 3 09:30: 10001's dividend on the 100 pre-split shares, +30; its split
      100 -> 33 with cash in lieu at the last close before the halt, 1 * 51;
      10002's split 100 -> 33, cash in lieu 1 * 27: cash 2 600.50. The sale of
      2 pre-split shares is floor(2 / 3) = 0 shares: unfilled.
    """
    bars = BARS[:5]
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", bars,
        {10001: [49, 50, NAN, 155, 156], 10002: [24, 25, 26, 80, 81]},
        {10001: [50, 51, NAN, 156, 157], 10002: [25, 26, 27, 81, 82]},
        {10001: [0.5, NAN, NAN, NAN, NAN], 10002: [0.25, NAN, 0.2584, NAN, NAN]},
        variables={
            "splitFactor": {10001: [1, 1, NAN, 1 / 3, 1], 10002: [1, 1, 1, 1 / 3, 1]},
            "cumfacshr": {10001: [1, 1, NAN, 3, 3], 10002: [1, 1, 1, 3, 3]},
            "divCash": {10001: [0, 0, NAN, 0.3, 0], 10002: [0.0] * 5},
        },
    )
    run_dir = run(
        TraderConfig(
            quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
            output_dir=str(tmp_path / "trader"),
        )
    )

    equity = xr.open_zarr(run_dir / "equity.zarr").load()["value"].values
    np.testing.assert_allclose(
        equity,
        [10_000.0, 10_192.50, 10_292.50, 2_600.50 + 33 * 156 + 33 * 81, 2_600.50 + 33 * 157 + 33 * 82],
        rtol=0, atol=1e-6,
    )
    cash = {(e["symbol"], e["action"]): e["amount"] for e in _corporate_actions(run_dir) if "amount" in e}
    assert cash == {(10001, "DIVIDEND"): 30.0, (10001, "CASH_IN_LIEU"): 51.0, (10002, "CASH_IN_LIEU"): 27.0}
    sale = _orders(run_dir).query("symbol == 10002 and side == 'SELL'").iloc[0]
    assert (sale["quantity"], sale["status"], sale["filled_quantity"]) == (2, "unfilled", 0)
    assert "0 shares" in sale["reason"]


def test_metrics_report_dividends_splits_and_distributions_under_execution_trader(replay):
    metrics = json.loads((replay / "metrics.json").read_text())

    trader = metrics["execution"]["trader"]
    # Dividends 100 - 93.70 + 50; cash in lieu -5 + 84; one spin-off of 4 200.
    assert trader["dividends"] == {"count": 3, "amount": pytest.approx(56.30)}
    assert trader["cash_in_lieu"] == {"count": 2, "amount": pytest.approx(79.0)}
    assert trader["value_distributions"] == {"count": 1, "amount": pytest.approx(4200.0)}
    # 10001 +400, 10003 -312 (a short), 10002 -334.
    assert trader["splits"] == {"count": 3, "share_change": 400 - 312 - 334}
    assert trader["commissions"] == pytest.approx(90.0 + 21.20)
    assert trader["minimum_fee_hits"] == 0
    assert trader["peak_cash_debit"] == 0.0
    settled = [
        (s["symbol"], s["delisting_timestamp"][:10], s["settlement_timestamp"][:10], s["price"])
        for s in metrics["execution"]["settlements"]
    ]
    assert settled == [
        ("10006", "2024-01-08", "2024-01-09", 8.40),
        ("10005", "2024-01-10", "2024-01-11", 23.65),
    ]
    whole = metrics["whole"]
    # Six round trips; 10001 (sold after its split), 10006 and 10005 closed.
    assert (whole["Total Trades"], whole["Total Closed Trades"], whole["Total Open Trades"]) == (6, 3, 3)
    assert whole["Total Orders"] == 7


def test_round_trip_statistics_equal_the_hand_count(replay):
    """Position round trips of the hand ledger, fees 0.1% of each next-open notional.

    Closed: 10001 bought 400 @50 (fee 20), split 2:1 and sold 800 @26.5 (fee
    21.20): PnL 21 200 - 20 000 - 41.20 = 1 158.80 over 2 bars; 10005 500 @20
    (fee 10) settled @23.65: 1 815 over 6 bars; the short 10006 1 000 @10 (fee
    10) settled @8.40: 1 590 over 4 bars. Open at bar 7's close: 10002 (500
    @40, fee 20, 166 shares after its reverse split, cash in lieu +84) marked
    166 * 130: 1 644; the short 10003 (625 @16, fee 10, -937 after its split,
    cash in lieu -5, dividend -93.70) marked 937 * 11: -415.70; 10004 (200
    @100, fee 20, dividends 100 + 50, spin-off 4 200) marked 200 * 86: 1 530.
    """
    whole = json.loads((replay / "metrics.json").read_text())["whole"]

    closed_pnl = [1_158.80, 1_815.0, 1_590.0]
    closed_return = [1_158.80 / 20_000, 1_815.0 / 10_000, 1_590.0 / 10_000]
    assert (whole["Total Trades"], whole["Total Closed Trades"], whole["Total Open Trades"]) == (6, 3, 3)
    assert whole["Open Trade PnL"] == pytest.approx(1_644.0 - 415.70 + 1_530.0, abs=1e-6)
    assert whole["Win Rate [%]"] == 100.0
    assert whole["Best Trade [%]"] == pytest.approx(100 * max(closed_return), rel=1e-9)
    assert whole["Worst Trade [%]"] == pytest.approx(100 * min(closed_return), rel=1e-9)
    assert whole["Avg Winning Trade [%]"] == pytest.approx(100 * sum(closed_return) / 3, rel=1e-9)
    assert whole["Avg Losing Trade [%]"] is None
    assert whole["Avg Winning Trade Duration"] == "4 days 00:00:00"
    assert whole["Avg Losing Trade Duration"] is None
    assert whole["Profit Factor"] is None  # no losing trip: infinite
    assert whole["Expectancy"] == pytest.approx(sum(closed_pnl) / 3, rel=1e-9)
