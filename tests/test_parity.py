"""The parity ladder at its seam, ``parity(quantlab_run)`` -> parity report (ADR 0007).

The report directory holds ``parity.json`` (inputs, end checks, one row per
rung, the closed-versus-open block) and ``parity.zarr`` (each rung's equity
per bar). What is locked here, on synthetic model-free quantlab runs:

- L0, quantlab's ``run_weights`` re-run as the run did, reproduces the run's
  ``equity.zarr`` to a relative 1e-12;
- T, trader's open loop, equals L5, the reference ledger with trader's
  costs and rounding: the same orders, fill prices, fees, rejections and
  settlements, and equity within USD 0.01 per fill so far;
- the rung identities: L2 = L1 when no buy is cash-capped, L3 = L2 without
  corporate actions, L4 = L3 with integral sizes, L5 = L4 up to rounding
  under the fraction fee model;
- closed-loop weights equal ``weights.zarr`` on holding-independent bars.

The full market below has, in raw prices (``--`` no price):

- 10001 gaps 5% up at bar 1's open (a cash-capped buy for vectorbt) and
  splits 2:1 on bar 4, with an order queued across the split;
- 10002 reverse-splits 1:3 on bar 5 (cash in lieu);
- 10003 is shorted and pays a 0.40 dividend on bar 4;
- 10004 has a value distribution (k = 1.25, share factor 1) plus a 0.25
  dividend on bar 6;
- 10005 is halted on bar 4, when its sale is due (a rejected order);
- 10006's bar 6 is a CRSP delisting row (+10% delisting return), settled on
  bar 7;
- 10007's bar 7 is a delisting row with a -100% delisting return (a
  zero valuation), settled on bar 8 at 0;
- 10008 lists on bar 3.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import LedoitWolfConfig, MeanVarianceConfig, TopNConfig
from quantlab.base.portfolio import LabelSpec
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab_trader.parity import parity
from quantlab_trader.venue.backtest.venue import ExecutionConfig
from tests.quantlab_run_fixture import ADJUSTED_SCALE, build_constructor_run, build_quantlab_run

NAN = np.nan
BARS = pd.bdate_range("2024-01-02", periods=10)
PERMNOS = tuple(range(10001, 10009))
OPEN = {
    10001: [49.5, 52.5, 51.5, 52.8, 26.6, 27.3, 27.7, 28.1, 28.6, 29.2],
    10002: [39.8, 40.2, 40.7, 41.1, 41.3, 124.8, 125.5, 126.3, 127.1, 126.8],
    10003: [29.9, 30.2, 30.35, 30.0, 29.4, 29.6, 29.95, 30.25, 30.35, 30.2],
    10004: [99.5, 100.6, 101.2, 100.8, 102.4, 103.6, 84.5, 84.7, 85.4, 86.1],
    10005: [19.9, 20.1, 20.3, 20.5, NAN, 20.8, 21.05, 21.1, 21.25, 21.2],
    10006: [9.9, 10.05, 10.15, 10.25, 10.3, 10.35, NAN, NAN, NAN, NAN],
    10007: [14.9, 15.1, 15.15, 15.2, 15.35, 15.3, 15.1, NAN, NAN, NAN],
    10008: [NAN, NAN, NAN, 11.9, 12.1, 12.3, 12.45, 12.4, 12.55, 12.7],
}
CLOSE = {
    10001: [50, 51, 52, 53, 27, 27.5, 28, 28.4, 29, 29.5],
    10002: [40, 40.5, 41, 41.2, 41.5, 125, 126, 127, 126.5, 128],
    10003: [30, 30.3, 30.1, 29.8, 29.5, 29.9, 30.2, 30.4, 30.1, 30.6],
    10004: [100, 101, 100.5, 102, 103, 104, 84, 85, 86, 85.5],
    10005: [20, 20.2, 20.4, 20.6, NAN, 20.9, 21, 21.2, 21.1, 21.3],
    10006: [10, 10.1, 10.2, 10.3, 10.25, 10.4, NAN, NAN, NAN, NAN],
    10007: [15, 15.2, 15.1, 15.3, 15.4, 15.2, 15.0, NAN, NAN, NAN],
    10008: [NAN, NAN, NAN, 12, 12.2, 12.4, 12.3, 12.5, 12.6, 12.8],
}
#: Ex-date factors and dividends: (permno, bar) -> (splitFactor, share factor, divCash).
ACTIONS = {
    (10001, 4): (2.0, 2.0, 0.0),
    (10002, 5): (1 / 3, 1 / 3, 0.0),
    (10003, 4): (1.0, 1.0, 0.40),
    (10004, 6): (1.25, 1.0, 0.25),
}
#: Delisting rows: (permno, bar) -> the delisting return carried by adjClose.
DELISTINGS = {(10006, 6): 0.10, (10007, 7): -1.0}

WEIGHTS = {p: [NAN] * len(BARS) for p in PERMNOS}
_ROWS = {
    0: {10001: 0.25, 10002: 0.25, 10004: 0.2, 10005: 0.1, 10006: 0.1, 10007: 0.1},
    2: {10001: 0.2, 10002: 0.2, 10003: -0.1},
    3: {10001: 0.22, 10005: 0.0, 10008: 0.1},
    4: {10001: 0.18, 10002: 0.24},
    5: {10004: 0.22, 10003: -0.12},
    6: {10006: 0.0, 10005: 0.08},
    7: {10007: 0.0, 10001: 0.2},
    8: {10002: 0.2, 10008: 0.12},
}
for _bar, _row in _ROWS.items():
    for _permno, _weight in _row.items():
        WEIGHTS[_permno][_bar] = _weight


def corporate_action_variables(close, actions, bars=BARS, scale=ADJUSTED_SCALE):
    """Return the CRSP corporate-action and adjusted-price variables of a raw market.

    ``splitFactor`` and ``cumfacshr`` follow ``actions``; the adjusted group
    is the raw one times a backward cumulative factor (each ex-date divides
    earlier prices by its ``splitFactor`` and scales them by ``1 - div /
    previous close``) times ``scale``, so adjusted returns are total returns
    as CRSP's are.
    """
    n = len(bars)
    permnos = sorted(close)
    split_factor, cumfacshr, div_cash, factor = {}, {}, {}, {}
    for p in permnos:
        raw = np.array(close[p], dtype=float)
        priced = np.isfinite(raw)
        k = np.ones(n)
        share = np.ones(n)
        div = np.zeros(n)
        for (q, bar), (kk, ss, dd) in actions.items():
            if q == p:
                k[bar], share[bar], div[bar] = kk, ss, dd
        shares_cum = np.ones(n)
        for t in range(n - 1, 0, -1):
            shares_cum[t - 1] = shares_cum[t] * share[t]
        f = np.ones(n)
        last_close = pd.Series(raw).ffill().to_numpy()
        for t in range(n - 1, 0, -1):
            f[t - 1] = f[t] / k[t]
            if div[t]:
                f[t - 1] *= 1.0 - div[t] / last_close[t - 1]
        split_factor[p] = list(np.where(priced, k, NAN))
        cumfacshr[p] = list(np.where(priced, shares_cum, NAN))
        div_cash[p] = list(np.where(priced, div, NAN))
        factor[p] = f * scale
    return split_factor, cumfacshr, div_cash, factor


def market_variables(open_, close, actions=None, delistings=None, bars=BARS):
    """The ``variables=`` of ``build_quantlab_run`` for a market with corporate actions.

    A delisting row has no raw price and an ``adjClose`` of the last adjusted
    close grown by its delisting return.
    """
    actions, delistings = actions or {}, delistings or {}
    split_factor, cumfacshr, div_cash, factor = corporate_action_variables(close, actions, bars)
    adj = {name: {} for name in ("adjOpen", "adjHigh", "adjLow", "adjClose")}
    for p in close:
        o, c = np.array(open_[p], dtype=float), np.array(close[p], dtype=float)
        adj["adjOpen"][p] = list(o * factor[p])
        adj["adjClose"][p] = list(c * factor[p])
        adj["adjHigh"][p] = list(np.fmax(o, c) * factor[p])
        adj["adjLow"][p] = list(np.fmin(o, c) * factor[p])
    for (p, bar), ret in delistings.items():
        last = pd.Series(adj["adjClose"][p][:bar]).ffill().iloc[-1]
        adj["adjClose"][p][bar] = last * (1.0 + ret)
    return {
        "splitFactor": split_factor,
        "cumfacshr": cumfacshr,
        "divCash": div_cash,
        **adj,
    }


def _report(parity_dir):
    report = json.loads((parity_dir / "parity.json").read_text())
    with xr.open_zarr(parity_dir / "parity.zarr") as data:
        return report, data.load()


def _equity(data, rung):
    return data["equity"].sel(rung=rung).values


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    root = tmp_path_factory.mktemp("parity_full")
    run_dir = build_quantlab_run(
        root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        init_cash=100_000.0, fees=0.001, slippage=0.001,
        variables=market_variables(OPEN, CLOSE, ACTIONS, DELISTINGS),
    )
    return run_dir, _report(parity(run_dir, output_dir=root / "parity"))


def test_l0_reproduces_the_runs_equity(full):
    run_dir, (report, data) = full
    with xr.open_zarr(run_dir / "equity.zarr") as equity:
        expected = equity["value"].values

    np.testing.assert_allclose(_equity(data, "L0"), expected, rtol=1e-12, atol=0)
    check = report["checks"]["L0_equals_run"]
    assert check["passed"] and check["max_rel_error"] <= 1e-12


def _scaled_weights(scale):
    return {p: [w * scale for w in WEIGHTS[p]] for p in PERMNOS}


@pytest.fixture(scope="module")
def uncapped(tmp_path_factory):
    """The full market with every target scaled to 80%: no buy is cash-capped."""
    root = tmp_path_factory.mktemp("parity_uncapped")
    run_dir = build_quantlab_run(
        root / "quantlab", BARS, OPEN, CLOSE, _scaled_weights(0.8),
        init_cash=100_000.0, fees=0.001, slippage=0.001,
        variables=market_variables(OPEN, CLOSE, ACTIONS, DELISTINGS),
    )
    return run_dir, _report(parity(run_dir, output_dir=root / "parity"))


def _row(report, rung):
    return next(row for row in report["rungs"] if row["rung"] == rung)


def test_l2_equals_l1_when_no_buy_is_capped(uncapped):
    _, (report, data) = uncapped

    assert _row(report, "L1")["buys_capped"] == 0
    np.testing.assert_allclose(_equity(data, "L2"), _equity(data, "L1"), rtol=1e-12, atol=0)
    assert _row(report, "L2")["orders"] == _row(report, "L1")["orders"]
    assert _row(report, "L2")["settlements"] == _row(report, "L1")["settlements"] == 2


def test_the_ledger_leaves_a_buy_vectorbt_capped_uncapped(full):
    _, (report, data) = full

    assert _row(report, "L1")["buys_capped"] >= 1
    assert _row(report, "L2")["buys_capped"] >= 1
    assert _row(report, "L2")["peak_cash_debit"] > 0.0
    assert _row(report, "L1")["peak_cash_debit"] == 0.0
    assert not np.allclose(_equity(data, "L2"), _equity(data, "L1"), rtol=1e-12, atol=0)


@pytest.fixture(scope="module")
def no_actions(tmp_path_factory):
    """The full market and table without splits, dividends or distributions."""
    root = tmp_path_factory.mktemp("parity_no_actions")
    run_dir = build_quantlab_run(
        root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        init_cash=100_000.0, fees=0.001, slippage=0.001,
        variables=market_variables(OPEN, CLOSE, {}, DELISTINGS),
    )
    return run_dir, _report(parity(run_dir, output_dir=root / "parity"))


def test_l3_equals_l2_without_corporate_actions(no_actions):
    _, (report, data) = no_actions

    np.testing.assert_allclose(_equity(data, "L3"), _equity(data, "L2"), rtol=1e-12, atol=0)
    assert _row(report, "L3")["orders"] == _row(report, "L2")["orders"]
    assert _row(report, "L3")["settlements"] == _row(report, "L2")["settlements"] == 2
    # trader also records 10006's sale, decided on its delisting bar, which
    # finds no opening print; vectorbt's settlement replaces it silently.
    # 10007's zero valuation is already its target weight 0: no order.
    assert _row(report, "L3")["rejected_orders"] == _row(report, "L2")["rejected_orders"] + 1



@pytest.fixture(scope="module")
def delisting_payments(tmp_path_factory):
    """``no_actions`` with CRSP's delisting payment on 10006's delisting row.

    CRSP books a cash merger's payment in ``dlynonorddivamt`` on the
    delisting row, so the store's ``divCash`` there is the delisting proceeds
    (the last close grown by the delisting return), which the settlement at
    the last valuation already pays.
    """
    root = tmp_path_factory.mktemp("parity_delisting_payments")
    variables = market_variables(OPEN, CLOSE, {}, DELISTINGS)
    variables["divCash"][10006][6] = 10.4 * 1.10
    run_dir = build_quantlab_run(
        root / "quantlab", BARS, OPEN, CLOSE, WEIGHTS,
        init_cash=100_000.0, fees=0.001, slippage=0.001, variables=variables,
    )
    return run_dir, _report(parity(run_dir, output_dir=root / "parity"))


def test_a_delisting_payment_is_not_also_paid_as_a_dividend(delisting_payments):
    _, (report, data) = delisting_payments

    np.testing.assert_allclose(_equity(data, "L3"), _equity(data, "L2"), rtol=1e-12, atol=0)
    assert report["checks"]["T_equals_L5"]["passed"]


INTEGRAL_BARS = pd.bdate_range("2024-01-02", periods=6)
INTEGRAL_OPEN = {
    10001: [10, 10, 11.5, 6.2, 6.45, 6.55],
    10002: [25, 25, 25.5, 25.2, 25.8, 26.4],
}
INTEGRAL_CLOSE = {
    10001: [10, 11, 12.5, 6.4, 6.5, 6.6],
    10002: [25, 24, 25, 25.5, 26, 26.5],
}
#: 500 and 200 shares at bar 0; 360 and 270 at bar 2, the sale of 140 queued
#: across 10001's 2:1 split on bar 3 (280 post-split shares).
INTEGRAL_WEIGHTS = {
    10001: [0.5, NAN, 0.4, NAN, NAN, NAN],
    10002: [0.5, NAN, 0.6, NAN, NAN, NAN],
}


@pytest.fixture(scope="module")
def integral(tmp_path_factory):
    root = tmp_path_factory.mktemp("parity_integral")
    run_dir = build_quantlab_run(
        root / "quantlab", INTEGRAL_BARS, INTEGRAL_OPEN, INTEGRAL_CLOSE, INTEGRAL_WEIGHTS,
        init_cash=10_000.0, fees=0.0, slippage=0.0,
        variables=market_variables(
            INTEGRAL_OPEN, INTEGRAL_CLOSE, {(10001, 3): (2.0, 2.0, 0.0)}, bars=INTEGRAL_BARS
        ),
    )
    return run_dir, _report(parity(run_dir, output_dir=root / "parity"))


def _orders(data, rung):
    """A rung's orders from ``parity.zarr``, as a frame."""
    dim = f"{rung}_order"
    return pd.DataFrame(
        {name: data[f"{rung}_{name}"].values
         for name in ("fill_bar", "symbol", "side", "quantity", "price", "fee")}
    ) if dim in data.dims else pd.DataFrame()


def test_l4_equals_l3_with_integral_sizes(integral):
    _, (report, data) = integral

    np.testing.assert_allclose(_equity(data, "L4"), _equity(data, "L3"), rtol=1e-12, atol=0)
    l3, l4 = _orders(data, "L3"), _orders(data, "L4")
    assert l4["quantity"].tolist() == l3["quantity"].tolist() == [500, 200, 280, 70]


def test_l5_differs_from_l4_by_rounding_only_under_the_fraction_fee_model(full):
    _, (report, data) = full
    l4, l5 = _orders(data, "L4"), _orders(data, "L5")

    keys = ["fill_bar", "symbol", "side", "quantity"]
    pd.testing.assert_frame_equal(l5[keys], l4[keys])
    assert np.abs(l5["price"] - l4["price"]).max() <= 1e-4
    assert np.abs(l5["fee"] - l4["fee"]).max() <= 0.005 + 1e-9
    # Rounding moves equity by at most: a tick per share and half a cent per
    # fill, half a cent per cash booking (5 corporate-action credits), and
    # half a tick per share of a last valuation at 4 decimals (2 delistings).
    per_bar = pd.Series(l4["quantity"] * 1e-4 + 0.005).groupby(l4["fill_bar"].values).sum()
    bound = per_bar.reindex(BARS, fill_value=0.0).cumsum().to_numpy() + 5 * 0.005 + 2 * 1e3 * 5e-5
    gap = np.abs(_equity(data, "L5") - _equity(data, "L4"))
    assert (gap <= bound).all()
    assert gap.max() > 0  # the rounding is there


@pytest.fixture(scope="module")
def full_ibkr(tmp_path_factory, full):
    run_dir, _ = full
    root = tmp_path_factory.mktemp("parity_full_ibkr")
    execution = ExecutionConfig(fee_model="ibkr_fixed", slippage=0.0005, init_cash=120_000.0)
    return run_dir, _report(parity(run_dir, output_dir=root, execution=execution))


@pytest.mark.parametrize("case", ["full", "full_ibkr"])
def test_t_equals_l5(case, request):
    _, (report, data) = request.getfixturevalue(case)
    check = report["checks"]["T_equals_L5"]
    l5, t = _orders(data, "L5"), _orders(data, "T")

    keys = ["fill_bar", "symbol", "side", "quantity"]
    pd.testing.assert_frame_equal(t[keys], l5[keys])
    assert (t["price"].round(4) == l5["price"].round(4)).all()
    assert np.abs(t["fee"] - l5["fee"]).max() < 0.005
    assert check["passed"], check
    assert check["orders_equal"] and check["rejections_equal"] and check["settlements_equal"]
    assert _row(report, "T")["rejected_orders"] == _row(report, "L5")["rejected_orders"] == 2
    assert _row(report, "T")["settlements"] == _row(report, "L5")["settlements"] == 2
    assert check["equity_max_error"] <= check["equity_tolerance_at_max"]


CLOSED_PERMNOS = (10001, 10002, 10003, 10004)
LABELS = [LabelSpec("ret_5", "raw", 1, 5)]


def _random_market(n_bars, seed=1):
    rng = np.random.default_rng(seed)
    open_, close = {}, {}
    for k, permno in enumerate(CLOSED_PERMNOS):
        path = (20.0 + 10 * k) * np.exp(np.cumsum(rng.normal(0, 0.02, n_bars)))
        close[permno] = list(np.round(path, 2))
        open_[permno] = list(np.round(path * (1 + rng.normal(0, 0.005, n_bars)), 2))
    return open_, close


def _constructor_run(root, rule, n_bars, first_bar, *, predictions=None, halt=None):
    """A closed-loop-ready run; ``halt=(permno, bar)`` takes that bar's prices away."""
    bars = pd.bdate_range("2024-01-02", periods=n_bars)
    open_, close = _random_market(n_bars)
    if halt is not None:
        permno, bar = halt
        open_[permno][bar] = close[permno][bar] = NAN
    if predictions is None:
        rng = np.random.default_rng(2)
        predictions = {p: list(rng.normal(0, 0.01, n_bars - first_bar)) for p in CLOSED_PERMNOS}
    run_dir, _ = build_constructor_run(
        root / "quantlab", bars, open_, close, {"ret_5": predictions}, rule, LABELS,
        first_bar=first_bar, rebalance_periods=2, init_cash=100_000.0,
    )
    return run_dir, _report(parity(run_dir, output_dir=root / "parity"))


def _mean_variance():
    return MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=5)),
            risk_aversion=5.0,
            ic=0.05,
            weight_cap=0.6,
        )
    )


@pytest.mark.parametrize(
    "rule, n_bars, first_bar",
    [
        (lambda: TopNConstructor(TopNConfig(direction="long_only", top_n=2)), 12, 0),
        (_mean_variance, 18, 6),
    ],
    ids=["topn", "mean_variance"],
)
def test_closed_loop_weights_equal_the_rebalance_table_on_holding_independent_bars(
    tmp_path, rule, n_bars, first_bar
):
    _, (report, data) = _constructor_run(tmp_path, rule(), n_bars, first_bar)
    block = report["closed_vs_open"]

    assert report["checks"]["closed_weights_equal_on_holding_independent_bars"]["passed"]
    assert block["holding_independent_rule"]
    assert block["rebalance_bars_compared"] == block["holding_independent_bars"] > 0
    assert block["bars_equal"] == block["rebalance_bars_compared"]
    assert block["max_weight_l1_distance"] == 0.0
    # Every bar holding-independent: closed and open loop trade alike.
    assert block["orders_equal"]
    assert block["max_equity_difference"] == 0.0
    np.testing.assert_array_equal(data["closed_equity"].values, _equity(data, "T"))


def test_a_locked_position_takes_its_bar_out_of_the_closed_loop_comparison(tmp_path):
    # 10001 always ranks first, so it is held when a halt locks it on bar 4.
    n_bars = 12
    predictions = {p: [0.05 - 0.01 * k] * n_bars for k, p in enumerate(CLOSED_PERMNOS)}
    _, (report, _) = _constructor_run(
        tmp_path, TopNConstructor(TopNConfig(direction="long_only", top_n=2)), n_bars, 0,
        predictions=predictions, halt=(10001, 4),
    )
    block = report["closed_vs_open"]

    assert block["holding_independent_bars"] == block["rebalance_bars_compared"] - 1
    assert report["checks"]["closed_weights_equal_on_holding_independent_bars"]["passed"]


def test_a_run_without_a_prediction_panel_has_no_closed_loop_block(full):
    _, (report, data) = full

    assert report["closed_vs_open"] is None
    assert "closed_weights_equal_on_holding_independent_bars" not in report["checks"]
    assert "closed_equity" not in data


def test_the_cli_writes_a_parity_report(tmp_path, capsys, full):
    from quantlab_trader.cli import main

    run_dir, _ = full

    status = main(
        ["parity", "--quantlab-run", str(run_dir), "--output-dir", str(tmp_path),
         "--fee-model", "ibkr_fixed"]
    )

    parity_dir = Path(capsys.readouterr().out.strip())
    assert status == 0
    assert parity_dir.parent == tmp_path
    report = json.loads((parity_dir / "parity.json").read_text())
    assert report["inputs"]["execution"]["fee_model"] == "ibkr_fixed"
    assert all(check["passed"] for check in report["checks"].values())


def test_the_cli_refuses_a_directory_that_is_not_a_run(tmp_path, capsys):
    from quantlab_trader.cli import main

    assert main(["parity", "--quantlab-run", str(tmp_path / "missing")]) == 1
    assert "quantlab-trader:" in capsys.readouterr().err


def test_l5_rounds_a_fee_at_half_a_cent_as_trader_does(tmp_path):
    # 1179 shares at 15.00 with a 0.1% fee: 17.685 in decimals, a hair below
    # in binary; nautilus rounds the binary value times 100, half away from 0.
    bars = pd.bdate_range("2024-01-02", periods=3)
    run_dir = build_quantlab_run(
        tmp_path / "quantlab", bars, {10001: [15.0, 15.0, 15.0]}, {10001: [15.0, 15.0, 15.0]},
        {10001: [1.0, NAN, NAN]}, init_cash=17_685.5, fees=0.001,
    )

    report, data = _report(parity(run_dir, output_dir=tmp_path / "parity"))

    assert _orders(data, "T")["fee"].tolist() == _orders(data, "L5")["fee"].tolist() == [17.68]
    assert report["checks"]["T_equals_L5"]["passed"]


@pytest.mark.parametrize("case", ["full", "full_ibkr"])
def test_l5_cash_is_trader_cash_to_the_float(case, request):
    # trader's cash is nautilus's margin balance, which books each reducing
    # fill's realized PnL rounded to the cent, less the open positions' cost
    # at their average open price. L5 keeps its money the same way, so the
    # equity both size from agrees to float precision, and a whole-share
    # target on a truncation boundary is cut the same way in both.
    _, (report, data) = request.getfixturevalue(case)

    np.testing.assert_allclose(_equity(data, "T"), _equity(data, "L5"), rtol=0, atol=1e-6)
