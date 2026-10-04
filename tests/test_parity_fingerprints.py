"""The parity ladder's data check compares quantlab's per-request data fingerprints.

A quantlab run's ``data_fingerprint`` maps each component path to a list of
entries, one per distinct request; L0's rerun records its own. The check
compares the entries both sides share (same key, same ``request``) by digest
alone, and has no verdict when they share none.
"""

from quantlab_trader.parity.ladder import _fingerprints_agree


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
