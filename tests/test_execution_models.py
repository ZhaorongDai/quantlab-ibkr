"""The backtest venue's fee and slippage models, at the ``runner.run(TraderConfig)`` seam.

IBKR Pro Fixed (ADR 0003, #13): USD 0.005 per share, at least USD 1.00 and at
most 1% of the trade value per order, plus the SEC fee of 0.0000206 x the
sale value on sales; the total is rounded to the cent. Fractional slippage
moves every fill against the order (buys up, sells down) by the given
fraction of the opening print and rounds the price, again against the order,
to the instrument's 0.0001 tick.

The quantlab run carries ``slippage=0.001``, which an open-loop replay takes
by default. The expected values are worked by hand from the raw prices:

===========  =====  ======  ====  ======
bar          0      1       2     3
===========  =====  ======  ====  ======
10001 open   9.9    10.5    10.8  11.2
10001 close  10     10.6    11    11.3
10002 open   0.39   0.4123  0.43  0.4321
10002 close  0.40   0.42    0.44  0.45
10003 open   19.8   20.2    21.5  22.6
10003 close  20     21      22    23
weights      0.1 / 0.1 / 0.5    0 / 0 / 0
===========  =====  ======  ====  ======

- close of bar 0 (equity 10 000): buy 100 of 10001, 2 500 of 10002 and 250
  of 10003. They fill at bar 1's open slipped up: 10.5105, 0.4127123 ->
  0.4128 and 20.2202. Commissions: 100 shares cost 0.50 per share, raised to
  the 1.00 minimum; 2 500 shares cost 12.50, capped at 1% of 1 032.00 =
  10.32; 250 shares cost 1.25. Cash 10 000 - 7 138.10 - 12.57 = 2 849.33.
- close of bar 2 (equity 2 849.33 + 1 100 + 1 100 + 5 500 = 10 549.33): sell
  everything. The sales fill at bar 3's open slipped down: 11.1888, 0.4316679
  -> 0.4316 and 22.5774. Fees: 1.00 + 0.0230 = 1.02 (minimum), 10.79 +
  0.0222 = 10.81 (cap), 1.25 + 0.1163 = 1.37. Cash 2 849.33 + 7 842.23 -
  13.20 = 10 678.36.
"""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab_trader.base.config import TraderConfig
from quantlab_trader.runner import run
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig, ExecutionConfig
from tests.quantlab_run_fixture import build_quantlab_run

NAN = np.nan
BARS = pd.bdate_range("2024-01-02", periods=4)
OPEN = {
    10001: [9.9, 10.5, 10.8, 11.2],
    10002: [0.39, 0.4123, 0.43, 0.4321],
    10003: [19.8, 20.2, 21.5, 22.6],
}
CLOSE = {
    10001: [10.0, 10.6, 11.0, 11.3],
    10002: [0.40, 0.42, 0.44, 0.45],
    10003: [20.0, 21.0, 22.0, 23.0],
}
WEIGHTS = {
    10001: [0.1, NAN, 0.0, NAN],
    10002: [0.1, NAN, 0.0, NAN],
    10003: [0.5, NAN, 0.0, NAN],
}


@pytest.fixture(scope="module")
def quantlab_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("execution_models")
    return build_quantlab_run(root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS, slippage=0.001)


def _replay(quantlab_run, output_dir, execution=ExecutionConfig()):
    config = TraderConfig(
        quantlab_run=str(quantlab_run),
        venue=BacktestVenueConfig(execution),
        loop="open",
        output_dir=str(output_dir),
    )
    return config, run(config)


def _orders(run_dir) -> pd.DataFrame:
    return xr.open_zarr(run_dir / "orders.zarr").load().to_dataframe()


@pytest.fixture(scope="module")
def ibkr_replay(quantlab_run, tmp_path_factory):
    return _replay(
        quantlab_run, tmp_path_factory.mktemp("ibkr"), ExecutionConfig(fee_model="ibkr_fixed")
    )


def test_ibkr_fixed_charges_the_minimum_the_cap_and_the_sec_fee_on_slipped_fills(ibkr_replay):
    _, run_dir = ibkr_replay
    orders = _orders(run_dir)

    assert orders["symbol"].tolist() == [10001, 10002, 10003] * 2
    assert orders["side"].tolist() == ["BUY"] * 3 + ["SELL"] * 3
    assert orders["quantity"].tolist() == [100, 2500, 250] * 2
    assert orders["status"].tolist() == ["filled"] * 6
    np.testing.assert_allclose(
        orders["fill_price"], [10.5105, 0.4128, 20.2202, 11.1888, 0.4316, 22.5774],
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(
        orders["fee"], [1.00, 10.32, 1.25, 1.02, 10.81, 1.37], rtol=0, atol=1e-12
    )


def test_equity_carries_the_slipped_fills_and_the_ibkr_fees(ibkr_replay):
    _, run_dir = ibkr_replay
    equity = xr.open_zarr(run_dir / "equity.zarr").load()["value"].values

    np.testing.assert_allclose(
        equity,
        [10_000.0, 2_849.33 + 1_060 + 1_050 + 5_250, 10_549.33, 10_678.36],
        rtol=0,
        atol=1e-9,
    )


@pytest.fixture(scope="module")
def override_replay(quantlab_run, tmp_path_factory):
    return _replay(
        quantlab_run,
        tmp_path_factory.mktemp("override"),
        ExecutionConfig(fee_model="fraction", slippage=0.0, init_cash=20_000.0),
    )


def test_execution_overrides_replace_the_runs_fee_slippage_and_cash(override_replay):
    _, run_dir = override_replay
    orders = _orders(run_dir)

    # Sized from 20 000, filled at the raw opens, charged the run's 0.1%.
    assert orders["quantity"].tolist() == [200, 5000, 500] * 2
    np.testing.assert_allclose(
        orders["fill_price"], [10.5, 0.4123, 20.2, 11.2, 0.4321, 22.6], rtol=0, atol=1e-12
    )
    notional = orders["quantity"] * orders["fill_price"]
    assert (np.abs(orders["fee"] - 0.001 * notional) <= 0.005 + 1e-9).all()
    equity = xr.open_zarr(run_dir / "equity.zarr").load()["value"].values
    assert equity[0] == 20_000.0


@pytest.mark.parametrize("replay", ["ibkr_replay", "override_replay"])
def test_execution_overrides_round_trip_through_config_json(replay, request):
    config, run_dir = request.getfixturevalue(replay)
    saved = json.loads((run_dir / "config.json").read_text())

    assert TraderConfig.from_config(saved) == config
    assert saved["venue"]["execution"] == config.venue.execution.get_config()


@pytest.mark.parametrize(
    ("loop", "execution", "fee_model", "slippage", "init_cash"),
    [
        ("closed", ExecutionConfig(), "IbkrFixedFeeModel", 0.001, 10_000.0),
        ("open", ExecutionConfig(), "FractionFeeModel", 0.001, 10_000.0),
        ("closed", ExecutionConfig("fraction", 0.002, 5e4), "FractionFeeModel", 0.002, 5e4),
        ("open", ExecutionConfig(fee_model="ibkr_fixed"), "IbkrFixedFeeModel", 0.001, 10_000.0),
    ],
)
def test_each_loop_resolves_its_default_execution(
    quantlab_run, loop, execution, fee_model, slippage, init_cash
):
    from quantlab_trader.quantlab_run import QuantlabRun

    venue = BacktestVenueConfig(execution).build(
        QuantlabRun.load(quantlab_run), start=BARS[0], end=BARS[-1], permnos=(10001,), loop=loop
    )

    assert type(venue.fee_model).__name__ == fee_model
    assert (venue.fill_model.slippage, venue.init_cash) == (slippage, init_cash)


def test_the_open_loop_default_fee_is_the_runs_fraction(quantlab_run):
    from quantlab_trader.quantlab_run import QuantlabRun

    venue = BacktestVenueConfig().build(
        QuantlabRun.load(quantlab_run), start=BARS[0], end=BARS[-1], permnos=(10001,), loop="open"
    )

    assert venue.fee_model.rate == 0.001


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"fee_model": "tiered"}, "fee_model"),
        ({"slippage": -0.001}, "slippage"),
        ({"slippage": 1.0}, "slippage"),
        ({"init_cash": 0.0}, "init_cash"),
    ],
)
def test_an_execution_setting_the_venue_cannot_simulate_is_refused(fields, match):
    with pytest.raises(ValueError, match=match):
        ExecutionConfig(**fields)


def test_ibkr_fixed_charges_no_fee_on_a_delisting_settlement(tmp_path):
    from tests.test_halts_and_delistings import ADJ_CLOSE
    from tests.test_halts_and_delistings import BARS as HALT_BARS
    from tests.test_halts_and_delistings import CLOSE as HALT_CLOSE
    from tests.test_halts_and_delistings import OPEN as HALT_OPEN
    from tests.test_halts_and_delistings import WEIGHTS as HALT_WEIGHTS

    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", HALT_BARS, HALT_OPEN, HALT_CLOSE, HALT_WEIGHTS,
        variables={"adjClose": ADJ_CLOSE},
    )
    _, run_dir = _replay(quantlab_run, tmp_path / "trader", ExecutionConfig("ibkr_fixed"))

    events = json.loads((run_dir / "events.json").read_text())["events"]
    [settlement] = [e for e in events if e["type"] == "corporate_action"]
    assert (settlement["action"], settlement["quantity"], settlement["fee"]) == ("DELIST", 250, 0.0)
    # The strategy's own orders pay IBKR Pro Fixed: 500 shares cost 2.50.
    assert _orders(run_dir)["fee"].iloc[0] == 2.50
