"""The ladder's data check: the data L0 and the closed loop read against the run's record.

A quantlab data fingerprint maps a component path to one entry per distinct
request (``request``: the first and last bar read, symbols, variables) with a
``digest`` of the values read. Two records are compared on the requests both
hold, by digest alone, as quantlab compares.

L0 re-runs the run's rebalance table through quantlab's engine, which reads
the window's prices as the run did, so its record shares the run's price
request. The closed loop records its own reads (``runner.run``), but a bar at
a time: its requests are one bar each where the run's whole-panel path read
the window in one request, so the two records share no request to compare.
``closed_loop_reread`` therefore re-reads, through the components the closed
loop decided with (``ConstructorTargets.read_sources``), the run's own
requests of every key the closed loop read, and that record is compared
with the run's: a store rebuilt since the run (an exposures or estimate store,
the prices a declared factor is computed from) gives another digest.
"""

from __future__ import annotations

from pathlib import Path

from quantlab.runs.record import DataRecorder
from quantlab_ibkr.decision import ConstructorTargets
from quantlab_ibkr.outputs import read_config
from quantlab_ibkr.quantlab_run import QuantlabRun


def fingerprints_agree(recorded: dict | None, rerun: dict | None) -> bool | None:
    """Whether ``rerun`` read the data ``recorded`` holds; ``None`` when they share no request.

    The entries both sides hold (same key, same ``request``) are compared by
    ``digest`` alone.

    Examples
    --------
    >>> request = {"start": "2024-01-02", "end": "2024-01-31", "symbols": None, "variables": None}
    >>> fingerprints_agree({"price_dataset": [{"request": request, "digest": "a"}]},
    ...                    {"price_dataset": [{"request": request, "digest": "b"}]})
    False
    """
    shared = _shared(recorded, rerun)
    if not shared:
        return None
    return all(old == new for _, old, new in shared)


def differing_keys(recorded: dict | None, rerun: dict | None) -> list[str]:
    """Return the keys with a shared request whose digests differ, sorted.

    Examples
    --------
    >>> request = {"start": "2024-01-02", "end": "2024-01-31", "symbols": None, "variables": None}
    >>> differing_keys({"a": [{"request": request, "digest": "1"}]}, {"a": [{"request": request, "digest": "2"}]})
    ['a']
    """
    return sorted({key for key, old, new in _shared(recorded, rerun) if old != new})


def compared_keys(recorded: dict | None, rerun: dict | None) -> list[str]:
    """Return the keys with at least one request both records hold, sorted.

    Examples
    --------
    >>> request = {"start": "2024-01-02", "end": "2024-01-31", "symbols": None, "variables": None}
    >>> compared_keys({"a": [{"request": request, "digest": "1"}]}, {"b": [{"request": request, "digest": "1"}]})
    []
    """
    return sorted({key for key, _, _ in _shared(recorded, rerun)})


def _shared(recorded: dict | None, rerun: dict | None) -> list[tuple[str, str, str]]:
    """Return ``(key, recorded digest, rerun digest)`` for every request both records hold."""
    recorded, rerun = recorded or {}, rerun or {}
    return [
        (key, entry["digest"], other["digest"])
        for key in sorted(set(recorded) & set(rerun))
        for entry in recorded[key]
        for other in rerun[key]
        if entry["request"] == other["request"]
    ]


def combined_agreement(*verdicts: bool | None) -> bool | None:
    """Combine verdicts: False if any is False, None if all are None, else True.

    Examples
    --------
    >>> combined_agreement(True, None), combined_agreement(None, None), combined_agreement(True, False)
    (True, None, False)
    """
    if any(verdict is False for verdict in verdicts):
        return False
    if all(verdict is None for verdict in verdicts):
        return None
    return True


def closed_loop_reread(run: QuantlabRun, closed_dir: Path) -> tuple[dict, dict]:
    """Re-read the run's requests of every key the closed loop read; return both records.

    The closed loop's record is its run's ``data_fingerprint``. The run's
    decision inputs are rebuilt as the closed loop rebuilt them, and every
    request the run recorded under a key the closed loop also read is read
    again through the same component inside a quantlab ``DataRecorder``: a
    dataset by ``panel`` (the request's symbols and variables), a factor by
    ``read``, a factor risk model's store (``<model key>.<store>``) by its
    store's ``read``. A key no component of the decision inputs answers, or
    a request with no bar, is not re-read.

    Parameters
    ----------
    run : QuantlabRun
        The quantlab run the closed loop replayed.
    closed_dir : pathlib.Path
        The closed-loop trader run directory.

    Returns
    -------
    tuple of dict
        The closed loop's record and the re-read record (``{}`` when nothing
        was re-read), both ``{key: [entry, ...]}``.

    Examples
    --------
    A closed loop that read a factor risk model's estimate store::

        closed, reread = closed_loop_reread(run, closed_dir)
        fingerprints_agree(run.data_fingerprint, reread)
    """
    closed = read_config(closed_dir).get("data_fingerprint") or {}
    sources = ConstructorTargets(run.decision_inputs()).read_sources()
    components: dict[str, object] = {}
    for item, key in sources:
        components.setdefault(key, item)
    with DataRecorder(keys=sources, owner=f"parity of {run.run_dir.name}") as reads:
        for key, entries in (run.data_fingerprint or {}).items():
            if key not in closed:
                continue
            for entry in entries:
                _read_again(components, key, entry["request"])
    return closed, reads.records


def _read_again(components: dict[str, object], key: str, request: dict) -> None:
    """Issue the recorded ``request`` of ``key`` again on the component it names."""
    start, end = request["start"], request["end"]
    if start is None or end is None:
        return
    item = components.get(key)
    if item is None:  # a factor risk model's store: "<model key>.<store>"
        model_key, _, store = key.rpartition(".")
        model = components.get(model_key)
        if model is None or not hasattr(getattr(model, store, None), "read"):
            return
        getattr(model, store).read(start, end)
    elif hasattr(item, "panel"):  # a dataset
        item.panel(start, end, symbols=request["symbols"], variables=request["variables"])
    elif hasattr(item, "read"):  # a factor read from its store
        item.read(start, end)
