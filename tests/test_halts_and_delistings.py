"""Halts and delistings at the ``runner.run(TraderConfig)`` seam (quantlab ADR 0014).

A halt is a bar without an opening print: a next-open order for that bar is
not submitted, it ends unfilled and the holding is kept. A delisted holding is
settled into cash at its last valuation on the bar after its delisting bar,
by a zero-fee venue fill tagged ``CORPORATE_ACTION_DELIST`` that is recorded
in ``events.json``, never in ``orders.zarr``.

10002's bar 4 is a CRSP delisting row: no raw price, an adjusted close that
carries a -30% delisting return. Its last valuation in raw prices is the last
raw close grown by the adjusted return since: 23 * 8.05 / 11.5 = 16.10.

==========  =====  =====  ====  ====  ====  ====
bar         0      1      2     3     4     5
==========  =====  =====  ====  ====  ====  ====
10001 open  9.5    10.5   11.6  --    13.5  14.5
10001 close 10     11     12    13    14    15
10002 open  19.5   20.2   21.8  22.6  --    --
10002 close 20     21     22    23    --    --
10002 adjC  10     10.5   11    11.5  8.05  --
10001 w     0.5    NaN    0     NaN   0     NaN
10002 w     0.5    NaN    NaN   NaN   NaN   NaN
==========  =====  =====  ====  ====  ====  ====

- close of bar 0: buy 500 of 10001 and 250 of 10002; they fill at bar 1's
  open (10.5, 20.2) with fees 5.25 and 5.05: cash -310.30.
- close of bar 2: sell 500 of 10001; bar 3 has no opening print for it (a
  halt), so the order is not submitted and ends unfilled; 500 shares kept.
- close of bar 4: 10002 is on its delisting bar and is marked at its last
  valuation, 16.10: equity -310.30 + 500*14 + 250*16.10 = 10 714.70; sell 500
  of 10001 again.
- open of bar 5: 10002 is settled, 250 at 16.10 with no fee (cash 3 714.70);
  then 10001's sale fills at 14.5 with fee 7.25: cash and equity 10 957.45.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab_trader.base.config import TraderConfig
from quantlab_trader.runner import run
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
from tests.quantlab_run_fixture import ADJUSTED_SCALE, build_quantlab_run, write_crsp_store

NAN = np.nan
BARS = pd.bdate_range("2024-01-02", periods=6)
OPEN = {
    10001: [9.5, 10.5, 11.6, NAN, 13.5, 14.5],
    10002: [19.5, 20.2, 21.8, 22.6, NAN, NAN],
}
CLOSE = {
    10001: [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    10002: [20.0, 21.0, 22.0, 23.0, NAN, NAN],
}
ADJ_CLOSE = {
    10001: [c * ADJUSTED_SCALE for c in CLOSE[10001]],
    10002: [10.0, 10.5, 11.0, 11.5, 8.05, NAN],
}
WEIGHTS = {
    10001: [0.5, NAN, 0.0, NAN, 0.0, NAN],
    10002: [0.5, NAN, NAN, NAN, NAN, NAN],
}


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    root = tmp_path_factory.mktemp("halts_and_delistings")
    quantlab_run = build_quantlab_run(
        root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        variables={"adjClose": ADJ_CLOSE},
    )
    config = TraderConfig(
        quantlab_run=str(quantlab_run),
        venue=BacktestVenueConfig(),
        loop="open",
        output_dir=str(root / "trader"),
    )
    return run(config)


def _orders(run_dir) -> pd.DataFrame:
    return xr.open_zarr(run_dir / "orders.zarr").load().to_dataframe()


def _events(run_dir, kind) -> list[dict]:
    events = json.loads((run_dir / "events.json").read_text())["events"]
    return [e for e in events if e["type"] == kind]


def test_an_order_into_a_halt_is_not_filled_and_the_holding_is_kept(replay):
    orders = _orders(replay)

    assert orders["decision_date"].tolist() == [BARS[0], BARS[0], BARS[2], BARS[4]]
    assert orders["symbol"].tolist() == [10001, 10002, 10001, 10001]
    assert orders["side"].tolist() == ["BUY", "BUY", "SELL", "SELL"]
    # The halted sale was never sent; the holding of 500 is sold again at bar 5.
    assert orders["quantity"].tolist() == [500, 250, 500, 500]
    assert orders["status"].tolist() == ["filled", "filled", "unfilled", "filled"]
    assert orders["filled_quantity"].tolist() == [500, 250, 0, 500]
    np.testing.assert_allclose(orders["fill_price"].iloc[[0, 1, 3]], [10.5, 20.2, 14.5])
    assert "opening print" in orders["reason"].iloc[2]


def test_the_halt_is_recorded_as_an_unfilled_order(replay):
    orders = _orders(replay)

    assert _events(replay, "unfilled_order") == [
        {
            "type": "unfilled_order",
            "decision_date": "2024-01-04",
            "symbol": 10001,
            "side": "SELL",
            "quantity": 500,
            "reason": orders["reason"].iloc[2],
        }
    ]


def test_a_delisted_holding_is_settled_at_its_last_valuation_without_a_fee(replay):
    assert _events(replay, "corporate_action") == [
        {
            "type": "corporate_action",
            "action": "DELIST",
            "timestamp": "2024-01-09",
            "symbol": 10002,
            "side": "SELL",
            "quantity": 250,
            "price": 16.1,
            "fee": 0.0,
        }
    ]


def test_corporate_action_fills_are_not_the_strategys_orders(replay):
    orders = _orders(replay)

    assert 10002 not in orders["symbol"].iloc[2:].tolist()
    assert orders["filled_quantity"].sum() == 500 + 250 + 500


def test_equity_is_cash_plus_holdings_through_the_halt_and_the_delisting(replay):
    equity = xr.open_zarr(replay / "equity.zarr").load()

    np.testing.assert_allclose(
        equity["value"].values,
        [10_000.0, 10_439.70, 11_189.70, 11_939.70, 10_714.70, 10_957.45],
        rtol=0,
        atol=1e-9,
    )


def test_a_total_loss_delisting_settles_at_zero(tmp_path):
    # A -100% delisting return: the delisting row's adjusted close is 0.
    # quantlab's vectorbt engine refuses to settle at a price of 0, so the
    # run is built on the -30% store and the store is then rewritten.
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        variables={"adjClose": ADJ_CLOSE},
    )
    write_crsp_store(
        tmp_path / "quantlab" / "crsp.zarr", BARS, OPEN, CLOSE,
        variables={"adjClose": {**ADJ_CLOSE, 10002: [10.0, 10.5, 11.0, 11.5, 0.0, NAN]}},
    )
    run_dir = run(
        TraderConfig(
            quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
            output_dir=str(tmp_path / "trader"),
        )
    )

    [settlement] = _events(run_dir, "corporate_action")
    assert (settlement["quantity"], settlement["price"], settlement["fee"]) == (250, 0.0, 0.0)
    equity = xr.open_zarr(run_dir / "equity.zarr").load()["value"].values
    # Bar 4: -310.30 + 500*14 + 250*0; bar 5: cash after selling 500 at 14.5 less 7.25.
    np.testing.assert_allclose(equity[4:], [6_689.70, 6_932.45], rtol=0, atol=1e-9)



@pytest.mark.parametrize("adj_close_on_delisting_row", [8.05, NAN])
def test_a_delisting_payment_in_div_cash_is_not_paid_on_top_of_the_settlement(
    tmp_path, adj_close_on_delisting_row
):
    # CRSP books a cash merger's payment in dlynonorddivamt on the delisting
    # row, so the store's divCash there is the delisting proceeds. The
    # settlement at the last valuation stands for them; they are not also a
    # dividend. With an adjusted close on the row (8.05) the delisting bar is
    # bar 4 and the settlement pays 16.10; without one, as the real CRSP
    # stores have it, the delisting bar is bar 3 and the settlement pays 23.
    adj_close = {**ADJ_CLOSE, 10002: [10.0, 10.5, 11.0, 11.5, adj_close_on_delisting_row, NAN]}
    div_cash = {10001: [0.0] * 6, 10002: [0.0, 0.0, 0.0, 0.0, 16.10, NAN]}
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        variables={"adjClose": adj_close, "divCash": div_cash},
    )
    run_dir = run(
        TraderConfig(
            quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
            output_dir=str(tmp_path / "trader"),
        )
    )

    actions = [e["action"] for e in _events(run_dir, "corporate_action")]
    assert "DIVIDEND" not in actions
    assert actions.count("DELIST") == 1
    [settlement] = [e for e in _events(run_dir, "corporate_action") if e["action"] == "DELIST"]
    price = 16.10 if np.isfinite(adj_close_on_delisting_row) else 23.0
    assert settlement["price"] == pytest.approx(price)
    # Bar 5: cash after the settlement and after selling 500 of 10001 at 14.5 less 7.25.
    equity = xr.open_zarr(run_dir / "equity.zarr").load()["value"].values
    np.testing.assert_allclose(
        equity[-1], -310.30 + 250 * price + 500 * 14.5 - 7.25, rtol=0, atol=1e-9
    )


def test_metrics_record_the_halt_as_a_rejected_order_and_the_settlement(replay):
    metrics = json.loads((replay / "metrics.json").read_text())

    execution = metrics["execution"]
    assert execution["rejected_order_count"] == 1
    assert execution["rejected_orders"] == [
        {
            "symbol": "10001",
            "axis_symbol": "10001",
            "signal_timestamp": "2024-01-04T00:00:00",
            "fill_timestamp": "2024-01-05T00:00:00",
            "reason": _orders(replay)["reason"].iloc[2],
        }
    ]
    assert execution["settlements"] == [
        {
            "symbol": "10002",
            "axis_symbol": "10002",
            "delisting_timestamp": "2024-01-08T00:00:00",
            "settlement_timestamp": "2024-01-09T00:00:00",
            "price": 16.1,
            "quantity": 250,
        }
    ]
    # The settlement is not an order: two buys and the final sale filled.
    assert metrics["whole"]["Total Orders"] == 3
    assert metrics["whole"]["Total Fees Paid"] == pytest.approx(5.25 + 5.05 + 7.25)
    # Both round trips closed: 10002 by its settlement, 10001 by the sale.
    assert (metrics["whole"]["Total Trades"], metrics["whole"]["Total Closed Trades"]) == (2, 2)
    assert metrics["whole"]["Total Open Trades"] == 0
