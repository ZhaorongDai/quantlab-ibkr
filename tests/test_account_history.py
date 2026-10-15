"""The Account history (#63): ``live.decide`` keeps one file of Strategy segments per account.

On the fake live harness of ``tests/test_live_run.py`` (TopN(2) rebalancing
on bars 6 and 9, a fake IBKR broker). Locked:

- the first real submission of a live run directory begins a segment on its
  t; later decides of the same directory leave the history unchanged;
- a dry run and an ``already_recorded`` decide never write it;
- a second directory's first real submission begins a new segment and ends
  the first at the previous bar;
- the first directory deciding after that is refused before any order is
  sent, with nothing recorded in its live run directory;
- ``live history show`` prints the segments, or says the account has none.
"""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from quantlab_ibkr import cli
from quantlab_ibkr.account_history import AccountHistory, StrategySegment, TakenOver
from tests.test_live_run import FIRST, Day

ACCOUNT = "DU1234567"


@pytest.fixture
def day(tmp_path):
    return Day(tmp_path)


def _history(day: Day) -> AccountHistory:
    return AccountHistory.of(day.config.venue)


def _switch(day: Day, live_dir: Path) -> None:
    """Point the harness's live config at another live run directory."""
    day.config = dataclasses.replace(
        day.config, venue=dataclasses.replace(day.config.venue, live_dir=str(live_dir))
    )


def _dry(day: Day, dry: bool = True) -> None:
    """Make the harness's decides dry runs (forced to decide), or real ones again."""
    day.config = dataclasses.replace(
        day.config,
        venue=dataclasses.replace(day.config.venue, dry_run=dry, force_decide=dry),
    )


def test_the_history_lives_beside_the_live_run_directories(day):
    history = _history(day)
    assert history.path == day.root / f"account_history_{ACCOUNT}.json"
    assert history.account == ACCOUNT and history.segments == ()


def test_the_first_real_submission_begins_a_segment_and_later_decides_leave_it(day):
    day.trade(range(FIRST, 6))  # bar 5 marks the account and holds: nothing submitted
    assert not _history(day).path.exists()

    result = day.decide(6)
    assert result.submitted
    (segment,) = _history(day).segments
    assert segment == StrategySegment(
        live_dir=str(day.live_dir.resolve()), strategy="live", first=day.bars[6], last=None
    )
    assert segment.open
    written = _history(day).path.read_bytes()

    day.broker.open_auction(7)
    day.record()
    day.trade(range(7, 11))  # bar 9 submits again
    assert _history(day).path.read_bytes() == written


def test_a_dry_run_and_an_already_recorded_decide_never_write_the_history(day):
    day.trade(range(FIRST, 6))
    _dry(day)
    result = day.decide(6)
    assert result.status == "dry_run" and result.orders
    assert not _history(day).path.exists()

    _dry(day, False)
    day.decide(6)
    written = _history(day).path.read_bytes()
    _dry(day)
    day.decide(7)
    _dry(day, False)
    assert day.decide(6).status == "already_recorded"
    assert _history(day).path.read_bytes() == written


def test_an_adopted_order_on_a_directorys_first_day_begins_its_segment(day):
    day.trade(range(FIRST, 6))
    day.decide(6)
    # An earlier run sent bar 6's orders and failed before recording anything:
    # the directory's journal and the history are gone, the orders work at IBKR.
    (day.live_dir / "journal.json").unlink()
    _history(day).path.unlink()
    sent = day.broker.received
    result = day.decide(6)
    assert result.adopted and result.submitted == () and day.broker.received == sent
    (segment,) = _history(day).segments
    assert segment.first == day.bars[6] and segment.open


def test_a_second_directory_takes_over_and_the_first_is_refused(day):
    first_dir = day.live_dir
    day.trade(range(FIRST, 9))
    _switch(day, day.root / "next" / "run")
    result = day.decide(9)
    assert result.submitted

    first, second = _history(day).segments
    assert (first.live_dir, first.first, first.last) == (
        str(first_dir.resolve()), day.bars[6], day.bars[8]
    )
    assert (second.live_dir, second.strategy, second.first, second.last) == (
        str((day.root / "next" / "run").resolve()), "next", day.bars[9], None
    )
    assert not first.open and second.open

    day.broker.open_auction(10)
    day.record()
    _switch(day, first_dir)
    journal = (first_dir / "journal.json").read_bytes()
    history = _history(day).path.read_bytes()
    sent = day.broker.received
    with pytest.raises(ValueError, match="taken over"):
        day.decide(10)
    assert day.broker.received == sent
    assert (first_dir / "journal.json").read_bytes() == journal
    assert _history(day).path.read_bytes() == history

    _dry(day)  # a dry run of the old directory is still allowed: it sends nothing
    assert day.decide(10).status == "dry_run"


def test_live_history_show_prints_the_segments(day, capsys):
    assert cli.main(["live", "history", "show", ACCOUNT, "--history-dir", str(day.root)]) == 0
    assert f"no account history for {ACCOUNT}" in capsys.readouterr().out

    day.trade(range(FIRST, 9))
    _switch(day, day.root / "next" / "run")
    day.decide(9)
    assert cli.main(["live", "history", "show", ACCOUNT, "--history-dir", str(day.root)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"account history of {ACCOUNT}: 2 strategy segments"
    assert lines[1].split()[:4] == ["live", str(day.bars[6].date()), "to", str(day.bars[8].date())]
    assert lines[2].split()[:4] == ["next", str(day.bars[9].date()), "to", "open"]


def test_the_history_directory_can_be_set_in_the_config(day, tmp_path):
    venue = dataclasses.replace(day.config.venue, history_dir=str(tmp_path / "histories"))
    assert AccountHistory.of(venue).path == tmp_path / "histories" / f"account_history_{ACCOUNT}.json"


def test_a_closed_segment_refuses_its_directory_and_a_foreign_file_is_refused(tmp_path):
    history = AccountHistory(tmp_path, ACCOUNT)
    history.write([
        StrategySegment(str(tmp_path / "a" / "ibkr"), "a", pd.Timestamp("2026-10-01"),
                        pd.Timestamp("2026-10-02")),
    ])
    with pytest.raises(TakenOver, match="was closed"):
        history.check(tmp_path / "a" / "ibkr", pd.Timestamp("2026-10-05"))
    history.path.rename(AccountHistory(tmp_path, "DU7654321").path)
    with pytest.raises(ValueError, match="not a format-1 account history of DU7654321"):
        AccountHistory(tmp_path, "DU7654321").segments
