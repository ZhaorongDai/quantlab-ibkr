"""The parity ladder's data check compares quantlab's per-request data fingerprints.

A quantlab run's ``data_fingerprint`` maps each component path to a list of
entries, one per distinct request; L0's rerun records its own, and so does a
closed loop (its reads a bar at a time), whose keys parity re-reads at the
run's own requests. The check compares the entries both sides share (same
key, same ``request``) by digest alone, and has no verdict when they share
none.
"""

import json

import numpy as np
import pytest
import xarray as xr
import zarr

from quantlab_ibkr.parity.ladder import _fingerprints_agree, parity
from tests.test_closed_loop_replay import DECLARING, _replay


def _entry(start: str, digest: str, variables=("adjClose", "adjOpen")) -> dict:
    """One recorded request with its digest."""
    return {
        "request": {"start": start, "end": "2024-03-01", "symbols": None, "variables": None if variables is None else list(variables)},
        "digest": digest,
    }


def test_shared_requests_with_equal_digests_agree():
    recorded = {"price_dataset": [_entry("2024-01-02", "a"), _entry("2023-12-01", "h")],
                "model.factors.0.dataset": [_entry("2023-12-20", "f", variables=None)]}
    rerun = {"price_dataset": [_entry("2024-01-02", "a")]}

    assert _fingerprints_agree(recorded, rerun) is True


def test_a_changed_digest_disagrees():
    recorded = {"price_dataset": [_entry("2024-01-02", "a")]}
    rerun = {"price_dataset": [_entry("2024-01-02", "b")]}

    assert _fingerprints_agree(recorded, rerun) is False


def test_nothing_shared_has_no_verdict():
    recorded = {"price_dataset": [_entry("2024-01-02", "a")]}

    assert _fingerprints_agree(recorded, {"price_dataset": [_entry("2024-01-03", "a")]}) is None
    assert _fingerprints_agree(recorded, {"benchmark_dataset": [_entry("2024-01-02", "a")]}) is None
    assert _fingerprints_agree(None, recorded) is None


# The closed loop's store reads (#43): a run whose rule declares factors or a
# factor risk model records, as quantlab's run() does, what its rule read
# (the exposures and estimate stores, or the price dataset a computed factor
# reads). The closed loop records its own reads through quantlab's recorded-read
# path, and parity re-reads the run's requests of every key the closed loop read
# and compares them by digest, so a store rebuilt after the run is caught.

#: The reads of each declaring rule, keyed as quantlab's backtester keys them;
#: ``<key>:all`` is a read of every variable (a factor computed from the prices).
EXPOSURES = "constructor.covariance.risk_model.exposures"
ESTIMATE = "constructor.covariance.risk_model.estimate"
RULE_READS = {
    "factors": {"price_dataset:all"},
    "risk_model": {EXPOSURES, ESTIMATE},
    "risk_model_cal": {"price_dataset:all", ESTIMATE},
}


def _reads(fingerprint: dict) -> set[str]:
    """The keys of a data fingerprint, plus ``<key>:all`` for a read of every variable."""
    return set(fingerprint) | {
        f"{key}:all"
        for key, entries in fingerprint.items()
        if any(entry["request"]["variables"] is None for entry in entries)
    }


@pytest.fixture(scope="module", params=sorted(DECLARING))
def declaring(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(f"fingerprints_{request.param}")
    (run_dir, _), window = DECLARING[request.param](root / "quantlab")
    return request.param, root, run_dir, window


def _parity(run_dir, output_dir) -> dict:
    return json.loads((parity(run_dir, output_dir=output_dir) / "parity.json").read_text())


def test_the_run_records_what_its_rule_read(declaring):
    rule, _, run_dir, _ = declaring
    recorded = json.loads((run_dir / "run.json").read_text())["data_fingerprint"]

    assert RULE_READS[rule] <= _reads(recorded)


def test_the_closed_loop_records_its_reads_through_quantlab(declaring, tmp_path):
    rule, _, run_dir, _ = declaring

    trader = _replay(run_dir, tmp_path, "closed")

    recorded = json.loads((trader / "config.json").read_text())["data_fingerprint"]
    assert RULE_READS[rule] <= _reads(recorded)
    assert all(entry["digest"] for entries in recorded.values() for entry in entries)


def test_an_open_loop_records_no_reads(declaring, tmp_path):
    _, _, run_dir, _ = declaring

    trader = _replay(run_dir, tmp_path, "open")

    assert json.loads((trader / "config.json").read_text())["data_fingerprint"] is None


def test_parity_compares_the_closed_loops_reads_with_the_runs(declaring, tmp_path):
    rule, _, run_dir, _ = declaring

    report = _parity(run_dir, tmp_path / "parity")

    check = report["checks"]["data_fingerprints_agree"]
    assert check["passed"] and check["agree"] is True
    assert check["differing"] == []
    assert {key.partition(":")[0] for key in RULE_READS[rule]} <= set(check["compared"]["closed_loop"])
    assert RULE_READS[rule] <= _reads(report["inputs"]["closed_loop_reread_data_fingerprint"])


def _rewrite_one_value(store, bar) -> None:
    """Change one value of the store's first variable at ``bar``, in place, as a rebuild would."""
    with xr.open_zarr(store) as data:
        name = next(n for n, v in data.data_vars.items() if "timestamp" in v.dims and v.dtype.kind == "f")
        variable = data[name]
        row = int(np.flatnonzero(data["timestamp"].values == np.datetime64(bar))[0])
        axis = variable.dims.index("timestamp")
    array = zarr.open_group(str(store), mode="r+")[name]
    index = tuple(row if k == axis else 0 for k in range(array.ndim))
    array[index] = array[index] + 1.0


@pytest.mark.parametrize("store, key", [("estimate.zarr", ESTIMATE), ("exposures.zarr", EXPOSURES)])
def test_a_store_rewritten_after_the_run_fails_the_check_and_is_named(tmp_path, store, key):
    (run_dir, _), window = DECLARING["risk_model"](tmp_path / "quantlab")
    _rewrite_one_value(tmp_path / "quantlab" / "risk" / store, window[0])

    check = _parity(run_dir, tmp_path / "parity")["checks"]["data_fingerprints_agree"]

    assert not check["passed"] and check["agree"] is False
    assert check["differing"] == [key]
