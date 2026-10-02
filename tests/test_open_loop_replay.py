"""Open-loop replay end to end, at the ``runner.run(TraderConfig)`` seam.

A model-free quantlab run (``tests/quantlab_run_fixture.py``) is replayed on
NautilusTrader through the whole trader stack. Its raw prices are twice its
adjusted ones and every open gaps away from the previous close, so trading on
adjusted prices, filling at the close or sizing at the open all change the
numbers below. The expected values are worked by hand from the raw prices:

==========  =====  =====  ====  ====  ====  ====
bar         0      1      2     3     4     5
==========  =====  =====  ====  ====  ====  ====
10001 open  9.5    10.5   11.6  12.4  13.5  14.5
10001 close 10     11     12    13    14    15
10002 open  19.5   20.2   21.8  22.6  23.7  24.6
10002 close 20     21     22    23    24    25
10001 w     0.5    NaN    0     NaN   NaN   0
10002 w     0.5    NaN    1     NaN   NaN   0.5
==========  =====  =====  ====  ====  ====  ====

- close of bar 0: equity 10 000; buy trunc(5000/10) = 500 of 10001 and
  trunc(5000/20) = 250 of 10002; both fill at bar 1's open (10.5, 20.2) with
  a 0.1% fee (5.25, 5.05): cash -310.30, a margin account's opening gap.
- close of bar 2: equity -310.30 + 500*12 + 250*22 = 11 189.70; sell 500 of
  10001 first, then buy trunc(11189.70/22) - 250 = 258 of 10002; they fill at
  bar 3's open (12.4, 22.6) with fees 6.20 and 5.83: cash 46.87.
- close of bar 5: sell 254 of 10002 (trunc(0.5*12746.87/25) = 254); there is
  no next open in the window, so it ends unfilled.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab_trader.base.config import TraderConfig
from quantlab_trader.runner import run
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
from tests.quantlab_run_fixture import build_quantlab_run

NAN = np.nan
BARS = pd.bdate_range("2024-01-02", periods=6)
OPEN = {
    10001: [9.5, 10.5, 11.6, 12.4, 13.5, 14.5],
    10002: [19.5, 20.2, 21.8, 22.6, 23.7, 24.6],
    10003: [5.0] * 6,
}
CLOSE = {
    10001: [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    10002: [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
    10003: [5.0] * 6,
}
WEIGHTS = {
    10001: [0.5, NAN, 0.0, NAN, NAN, 0.0],
    10002: [0.5, NAN, 1.0, NAN, NAN, 0.5],
    10003: [0.0, NAN, 0.0, NAN, NAN, 0.0],
}
RUN_FILES = [
    "config.json", "decisions.zarr", "equity.zarr", "events.json", "metrics.json",
    "orders.zarr", "report.html",
]


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    root = tmp_path_factory.mktemp("open_loop")
    quantlab_run = build_quantlab_run(root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS)
    config = TraderConfig(
        quantlab_run=str(quantlab_run),
        venue=BacktestVenueConfig(),
        loop="open",
        output_dir=str(root / "trader"),
    )
    return dict(config=config, quantlab_run=quantlab_run, run_dir=run(config))


def _orders(run_dir) -> pd.DataFrame:
    return xr.open_zarr(run_dir / "orders.zarr").load().to_dataframe()


def test_writes_the_trader_run_directory(replay):
    run_dir = replay["run_dir"]

    assert run_dir.parent == Path(replay["config"].output_dir)
    assert sorted(p.name for p in run_dir.iterdir()) == RUN_FILES


def test_config_json_rebuilds_the_trader_config(replay):
    saved = json.loads((replay["run_dir"] / "config.json").read_text())

    assert TraderConfig.from_config(saved) == replay["config"]
    assert saved["quantlab_run"] == str(replay["quantlab_run"])


def test_decisions_are_the_rebalance_table(replay):
    decided = xr.open_zarr(replay["run_dir"] / "decisions.zarr").load()
    table = xr.open_zarr(replay["quantlab_run"] / "weights.zarr").load()

    xr.testing.assert_equal(decided["weight"], table["weight"])


def test_orders_are_sized_at_the_close_and_filled_at_the_opening_print(replay):
    orders = _orders(replay["run_dir"])

    assert orders["decision_date"].tolist() == [BARS[0]] * 2 + [BARS[2]] * 2 + [BARS[5]]
    assert orders["symbol"].tolist() == [10001, 10002, 10001, 10002, 10002]
    assert orders["side"].tolist() == ["BUY", "BUY", "SELL", "BUY", "SELL"]
    assert orders["quantity"].tolist() == [500, 250, 500, 258, 254]
    assert orders["status"].tolist() == ["filled"] * 4 + ["unfilled"]
    np.testing.assert_allclose(orders["fill_price"][:4], [10.5, 20.2, 12.4, 22.6])
    np.testing.assert_allclose(orders["fee"][:4], [5.25, 5.05, 6.20, 5.83])
    assert math.isnan(orders["fill_price"].iloc[4])


def test_equity_is_derived_cash_plus_holdings_at_the_raw_close(replay):
    equity = xr.open_zarr(replay["run_dir"] / "equity.zarr").load()

    assert list(equity["timestamp"].values) == list(BARS.values)
    np.testing.assert_allclose(
        equity["value"].values,
        [10_000.0, 10_439.70, 11_189.70, 11_730.87, 12_238.87, 12_746.87],
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        equity["returns"].values,
        np.r_[0.0, equity["value"].values[1:] / equity["value"].values[:-1] - 1],
    )


def test_an_order_without_a_next_open_is_recorded_as_unfilled(replay):
    events = json.loads((replay["run_dir"] / "events.json").read_text())
    orders = _orders(replay["run_dir"])

    unfilled = [e for e in events["events"] if e["type"] == "unfilled_order"]
    assert unfilled == [
        {
            "type": "unfilled_order",
            "decision_date": "2024-01-09",
            "symbol": 10002,
            "side": "SELL",
            "quantity": 254,
            "reason": orders["reason"].iloc[4],
        }
    ]
    assert "next open" in orders["reason"].iloc[4]


@pytest.mark.parametrize(
    ("missing", "fill", "valuation"),
    [("open", "adjOpen", "adjClose"), ("close", "adjOpen", "adjClose"), ("adjClose", "open", "close")],
)
def test_a_price_dataset_without_raw_prices_or_adjusted_closes_is_refused(
    tmp_path, missing, fill, valuation
):
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        fill_price_column=fill, valuation_price_column=valuation,
        drop_variables=(missing,),
    )
    config = TraderConfig(
        quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
        output_dir=str(tmp_path / "trader"),
    )

    with pytest.raises(ValueError, match=missing):
        run(config)
    assert not (tmp_path / "trader").exists()


def test_a_membership_masked_price_dataset_is_refused(tmp_path):
    # quantlab's sp500 examples' members.zarr: adjusted columns, close and
    # volume only, NaN where the PERMNO is not a member (10003 leaves after bar 2).
    member = {10001: [True] * 6, 10002: [True] * 6, 10003: [True] * 3 + [False] * 3}
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS, member=member
    )
    config = TraderConfig(
        quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
        output_dir=str(tmp_path / "trader"),
    )

    with pytest.raises(ValueError) as refused:
        run(config)
    message = str(refused.value)
    assert "unmasked market dataset" in message
    assert "StockDataset" in message
    assert "['open', 'splitFactor', 'cumfacshr', 'divCash']" in message
    assert not (tmp_path / "trader").exists()


def test_the_cli_replays_a_quantlab_run_and_prints_the_run_directory(tmp_path, capsys):
    from quantlab_trader.cli import main

    quantlab_run = build_quantlab_run(tmp_path / "quantlab", BARS, OPEN, CLOSE, WEIGHTS)

    status = main(
        ["backtest", "--quantlab-run", str(quantlab_run), "--loop", "open",
         "--start", "2024-01-03", "--output-dir", str(tmp_path / "trader")]
    )

    run_dir = Path(capsys.readouterr().out.strip())
    assert status == 0
    assert run_dir.parent == tmp_path / "trader"
    saved = json.loads((run_dir / "config.json").read_text())
    assert (saved["loop"], saved["start"]) == ("open", "2024-01-03")
    equity = xr.open_zarr(run_dir / "equity.zarr").load()
    assert equity["timestamp"].values[0] == BARS[1]


def test_the_cli_runs_a_saved_trader_config(replay, tmp_path, capsys):
    from quantlab_trader.cli import main

    saved = json.loads((replay["run_dir"] / "config.json").read_text())
    saved["output_dir"] = str(tmp_path)
    (tmp_path / "trader.json").write_text(json.dumps(saved))

    assert main(["backtest", str(tmp_path / "trader.json")]) == 0

    run_dir = Path(capsys.readouterr().out.strip())
    orders = _orders(run_dir)
    assert orders["quantity"].tolist() == [500, 250, 500, 258, 254]


def test_the_cli_reports_a_refused_run_without_a_traceback(tmp_path, capsys):
    from quantlab_trader.cli import main

    status = main(["backtest", "--quantlab-run", str(tmp_path / "missing")])

    assert status == 1
    assert "quantlab-trader:" in capsys.readouterr().err


def test_every_next_open_order_of_a_bar_is_submitted_however_many(tmp_path):
    # nautilus's risk engine denies submissions beyond 100 per second by
    # default; the venue submits a whole rebalance at the open + 1 ns, so a
    # 150-security rebalance must go through in full.
    permnos = range(20001, 20151)
    bars = pd.bdate_range("2024-01-02", periods=3)
    open_ = {p: [10.0, 10.0, 10.0] for p in permnos}
    close = {p: [10.0, 10.0, 10.0] for p in permnos}
    weights = {p: [1 / 200, NAN, NAN] for p in permnos}
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", bars, open_, close, weights, init_cash=1_000_000.0
    )
    run_dir = run(
        TraderConfig(
            quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
            output_dir=str(tmp_path / "trader"),
        )
    )

    orders = xr.open_zarr(run_dir / "orders.zarr").load().to_dataframe()
    assert len(orders) == 150
    assert (orders["status"] == "filled").all()
    assert (orders["filled_quantity"] == 500).all()


def test_a_fill_of_a_quantity_whose_double_is_inexact_is_recorded_whole(tmp_path):
    # nautilus's Quantity(59353).as_double() is 59352.99999999999; a fill of
    # 59353 shares must be recorded as 59353 filled, not 59352 and partial.
    bars = pd.bdate_range("2023-03-29", periods=3)
    open_ = {10001: [0.1150, 0.1155, 0.1126]}
    close = {10001: [0.1202, 0.1131, 0.1080]}
    init_cash = (59353 + 0.5) * 0.1202 / 0.5
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", bars, open_, close, {10001: [0.5, NAN, NAN]},
        init_cash=init_cash, fees=0.0005,
    )
    run_dir = run(
        TraderConfig(
            quantlab_run=str(quantlab_run), venue=BacktestVenueConfig(), loop="open",
            output_dir=str(tmp_path / "trader"),
        )
    )

    orders = xr.open_zarr(run_dir / "orders.zarr").load().to_dataframe()
    assert orders["quantity"].tolist() == [59353]
    assert orders["filled_quantity"].tolist() == [59353]
    assert orders["status"].tolist() == ["filled"]
