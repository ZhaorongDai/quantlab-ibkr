"""Closed-loop replay end to end, at the ``runner.run(TraderConfig)`` seam.

A model-free quantlab run with a prediction panel
(``tests/quantlab_run_fixture.py:build_constructor_run``) is replayed with
quantlab's own decision inputs rebuilt from the run (``DecisionInputs.from_run``).
What is locked here (ADR 0007, ADR 0008):

- on holding-independent bars (no locked position, no hold) the decided
  weights equal the run's ``weights.zarr`` row bit for bit, for TopN and for
  mean-variance with Ledoit-Wolf (whose returns window quantlab reads from
  the rule's last ``history_bars`` decision prices);
- where every bar is holding-independent, closed and open loop under the
  same execution block give identical orders and equity;
- rebalance bars are counted from the panel's first timestamp, and
  narrowing the window does not move them;
- a rule's failure is a hold, recorded with its message;
- runs trader cannot replay closed-loop are refused.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import LedoitWolfConfig, MeanVarianceConfig, TopNConfig
from quantlab.runs.prediction_panel import LabelSpec
from quantlab.portfolio.predefined.ledoit_wolf import LedoitWolfRiskModel
from quantlab.portfolio.predefined.mean_variance import MeanVarianceOptimizer
from quantlab.portfolio.predefined.top_n import TopNConstructor
from quantlab_trader.base.config import TraderConfig
from quantlab_trader.runner import run
from quantlab_trader.venue.backtest.venue import BacktestVenueConfig, ExecutionConfig
from tests.quantlab_run_fixture import build_constructor_run, build_quantlab_run

PERMNOS = (10001, 10002, 10003, 10004)
LABELS = [LabelSpec("ret_5", "raw", 1, 5)]
#: Both loops of a closed-versus-open comparison execute alike.
SAME_EXECUTION = BacktestVenueConfig(ExecutionConfig(fee_model="fraction"))


def _market(n_bars: int, seed: int):
    """Raw opens and closes per PERMNO: random walks, opens gapping from the previous close."""
    rng = np.random.default_rng(seed)
    close, open_ = {}, {}
    for k, permno in enumerate(PERMNOS):
        path = (20.0 + 10 * k) * np.exp(np.cumsum(rng.normal(0, 0.02, n_bars)))
        close[permno] = list(np.round(path, 2))
        open_[permno] = list(np.round(path * (1 + rng.normal(0, 0.005, n_bars)), 2))
    return open_, close


def _predictions(n_bars: int, seed: int):
    rng = np.random.default_rng(seed)
    return {"ret_5": {p: list(rng.normal(0, 0.01, n_bars)) for p in PERMNOS}}


def _topn():
    return TopNConstructor(TopNConfig(direction="long_only", top_n=2))


def _mean_variance(weight_cap=0.6):
    return MeanVarianceOptimizer(
        MeanVarianceConfig(
            expected_return_label="ret_5",
            risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=5)),
            risk_aversion=5.0,
            ic=0.05,
            weight_cap=weight_cap,
        )
    )


#: (rule factory, store bars, first window bar): mean-variance warms up 6 bars.
RULES = {"topn": (_topn, 12, 0), "mean_variance": (_mean_variance, 18, 6)}


def _build(root, rule_name, *, rebalance_periods=2, **kwargs):
    make_rule, n_bars, first_bar = RULES[rule_name]
    bars = pd.bdate_range("2024-01-02", periods=n_bars)
    open_, close = _market(n_bars, seed=1)
    run_dir, failed = build_constructor_run(
        root / "quantlab",
        bars,
        open_,
        close,
        _predictions(n_bars - first_bar, seed=2),
        kwargs.pop("rule", None) or make_rule(),
        LABELS,
        first_bar=first_bar,
        rebalance_periods=rebalance_periods,
        **kwargs,
    )
    return run_dir, failed, bars[first_bar:]


def _replay(quantlab_run, root, loop, venue=SAME_EXECUTION, **fields):
    return run(
        TraderConfig(
            quantlab_run=str(quantlab_run), venue=venue, loop=loop,
            output_dir=str(root / "trader"), **fields,
        )
    )


def _zarr(path) -> xr.Dataset:
    with xr.open_zarr(path) as data:
        return data.load()


def _events(run_dir) -> list[dict]:
    return json.loads((run_dir / "events.json").read_text())["events"]


_REPLAYS: dict[str, dict] = {}


def _replays(rule_name, tmp_path_factory) -> dict:
    """Build the rule's run and replay it closed and open loop, once per session."""
    if rule_name not in _REPLAYS:
        root = tmp_path_factory.mktemp(f"closed_{rule_name}")
        _REPLAYS[rule_name] = _replay_both(root, rule_name)
    return _REPLAYS[rule_name]


@pytest.fixture(params=sorted(RULES))
def replays(request, tmp_path_factory):
    return _replays(request.param, tmp_path_factory)


def _replay_both(root, rule_name) -> dict:
    quantlab_run, failed, window = _build(root, rule_name)
    return dict(
        quantlab_run=quantlab_run,
        failed=failed,
        window=window,
        closed=_replay(quantlab_run, root, "closed"),
        open=_replay(quantlab_run, root, "open"),
    )


def _decided_and_table(replays):
    """Return the closed loop's decisions and the run's rows at the same bars."""
    decided = _zarr(replays["closed"] / "decisions.zarr")["weight"]
    table = _zarr(replays["quantlab_run"] / "weights.zarr")["weight"]
    expected = table.sel(timestamp=decided["timestamp"]).transpose(*decided.dims)
    return decided.sel(symbol=expected["symbol"]), expected, table


def test_the_constructor_decides_every_rebalance_bar_from_the_anchor(replays):
    decided, _, table = _decided_and_table(replays)

    # Every other bar from the panel's first, the window's last excepted.
    assert list(decided["timestamp"].values) == list(replays["window"][:-1:2].values)
    assert replays["failed"] == []
    assert np.isfinite(decided.values).all()
    assert table.drop_sel(timestamp=decided["timestamp"].values).isnull().all()


def test_decided_weights_equal_the_run_rebalance_table(replays):
    decided, expected, _ = _decided_and_table(replays)

    np.testing.assert_allclose(decided.values, expected.values, rtol=0, atol=1e-8)


# Bit for bit for mean-variance too since quantlab #112 leaves the turnover
# term out of the problem at a zero penalty.
@pytest.mark.parametrize("rule_name", sorted(RULES))
def test_decided_weights_equal_the_run_rebalance_table_bit_for_bit(rule_name, tmp_path_factory):
    replays = _replays(rule_name, tmp_path_factory)
    decided, expected, _ = _decided_and_table(replays)

    assert np.array_equal(decided.values, expected.values)


def test_closed_and_open_loop_give_identical_orders_and_equity(replays):
    closed = _zarr(replays["closed"] / "orders.zarr").to_dataframe()
    opened = _zarr(replays["open"] / "orders.zarr").to_dataframe()
    closed_equity = _zarr(replays["closed"] / "equity.zarr")
    open_equity = _zarr(replays["open"] / "equity.zarr")

    assert len(closed) > 4
    pd.testing.assert_frame_equal(closed, opened)
    xr.testing.assert_identical(closed_equity, open_equity)


def test_narrowing_the_window_does_not_move_the_rebalance_bars(tmp_path):
    quantlab_run, _, window = _build(tmp_path, "topn")

    run_dir = _replay(
        quantlab_run, tmp_path, "closed",
        start=str(window[1].date()), end=str(window[-3].date()),
    )

    decided = _zarr(run_dir / "decisions.zarr")["weight"]
    table = _zarr(quantlab_run / "weights.zarr")["weight"]
    # Anchored on window[0]: bars 2, 4, 6 and 8; bar 9 is the replay's last.
    assert list(decided["timestamp"].values) == list(window[[2, 4, 6, 8]].values)
    expected = table.sel(timestamp=decided["timestamp"]).transpose(*decided.dims)
    assert np.array_equal(decided.sel(symbol=expected["symbol"]).values, expected.values)


def test_a_rule_failure_is_a_hold_recorded_with_its_message(tmp_path):
    # Four symbols under a 0.2 cap cannot hold a fully invested book.
    quantlab_run, failed, window = _build(
        tmp_path, "mean_variance", rule=_mean_variance(weight_cap=0.2)
    )

    run_dir = _replay(quantlab_run, tmp_path, "closed")

    holds = [e for e in _events(run_dir) if e["type"] == "hold"]
    assert [pd.Timestamp(e["timestamp"]) for e in holds] == [
        pd.Timestamp(b) for b in failed
    ]
    assert len(holds) == len(window[:-1:2])
    assert all("infeasible" in e["failure"] for e in holds)
    assert _zarr(run_dir / "decisions.zarr")["weight"].isnull().all()
    assert _zarr(run_dir / "orders.zarr").sizes["order"] == 0


def test_closed_loop_is_the_default_and_charges_ibkr_fixed_fees(tmp_path, capsys):
    from quantlab_trader.cli import main

    quantlab_run, _, _ = _build(tmp_path, "topn")

    status = main(
        ["backtest", "--quantlab-run", str(quantlab_run), "--output-dir", str(tmp_path / "t")]
    )

    run_dir = Path(capsys.readouterr().out.strip())
    assert status == 0
    assert json.loads((run_dir / "config.json").read_text())["loop"] == "closed"
    orders = _zarr(run_dir / "orders.zarr")
    # IBKR Pro Fixed: a few hundred shares cost the USD 1.00 minimum (the
    # run's 0.1% would charge cents).
    assert (orders["fee"].values == 1.0).sum() >= 4


def test_a_run_without_a_prediction_panel_is_refused(tmp_path):
    bars = pd.bdate_range("2024-01-02", periods=4)
    quantlab_run = build_quantlab_run(
        tmp_path / "quantlab", bars, {10001: [10.0] * 4}, {10001: [10.0] * 4},
        {10001: [1.0, np.nan, np.nan, np.nan]},
    )

    with pytest.raises(ValueError, match="no prediction panel"):
        _replay(quantlab_run, tmp_path, "closed")
    assert not (tmp_path / "trader").exists()


class FactorTopN(TopNConstructor):
    """A rule declaring factors trader cannot compute."""

    def required_factors(self):
        return ["a factor"]


def _edit_config(run_dir: Path, edit) -> None:
    config = json.loads((run_dir / "config.json").read_text())
    edit(config)
    (run_dir / "config.json").write_text(json.dumps(config))


def _edit_record(run_dir: Path, edit) -> None:
    """Edit the quantlab run's ``run.json``, to check a refusal of a run quantlab wrote otherwise."""
    record = json.loads((run_dir / "run.json").read_text())
    edit(record)
    (run_dir / "run.json").write_text(json.dumps(record))


def test_a_rule_declaring_required_factors_is_refused(tmp_path):
    quantlab_run, _, _ = _build(tmp_path, "topn")
    _edit_config(
        quantlab_run,
        lambda c: c["constructor"].update(name=f"{__name__}.FactorTopN"),
    )

    with pytest.raises(ValueError, match="required_factors"):
        _replay(quantlab_run, tmp_path, "closed")


def test_a_run_valued_at_raw_prices_is_refused_closed_loop(tmp_path):
    quantlab_run, _, _ = _build(tmp_path, "topn")
    _edit_record(
        quantlab_run, lambda r: r["market"].update(valuation_price_column="close")
    )

    with pytest.raises(ValueError, match="adjusted"):
        _replay(quantlab_run, tmp_path, "closed")


def test_a_locked_position_is_kept_and_other_bars_still_equal_the_run(tmp_path):
    # 10001 always ranks first, so both books hold it when it halts (no
    # opening print) on rebalance bar 4: locked, its weight depends on the
    # holdings. Every other rebalance bar is holding-independent.
    n_bars = 12
    bars = pd.bdate_range("2024-01-02", periods=n_bars)
    open_, close = _market(n_bars, seed=1)
    open_[10001][4] = np.nan
    predictions = _predictions(n_bars, seed=2)
    predictions["ret_5"][10001] = [1.0] * n_bars
    quantlab_run, _ = build_constructor_run(
        tmp_path / "quantlab", bars, open_, close, predictions, _topn(), LABELS,
        rebalance_periods=2,
    )

    run_dir = _replay(quantlab_run, tmp_path, "closed")

    decided, expected, _ = _decided_and_table(
        {"closed": run_dir, "quantlab_run": quantlab_run}
    )
    free = decided["timestamp"] != np.datetime64(bars[4])
    assert np.array_equal(decided.values[free.values], expected.values[free.values])
    locked = decided.sel(timestamp=bars[4], symbol=10001).item()
    assert 0.0 < locked < 1.0
    assert decided.sel(timestamp=bars[4]).sum().item() == pytest.approx(1.0)
    orders = _zarr(run_dir / "orders.zarr").to_dataframe()
    on_halt = orders[orders["decision_date"] == bars[4]]
    assert 10001 not in on_halt["symbol"].tolist()
    assert len(on_halt) > 0
