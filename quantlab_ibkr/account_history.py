"""The Account history: the ordered Strategy segments one IBKR account has run (ADR 0012).

One JSON file per account, ``account_history_<ACCOUNT>.json``, is the only
place a strategy switch is written down. It sits beside the live run
directories: in ``IbkrVenueConfig.history_dir`` when the config sets it,
else in the parent of the live run directory's parent, the directory the
strategies' directories share (``/data/quantlab/live`` for the live run
directory ``/data/quantlab/live/<strategy>/ibkr``). A segment's strategy is
the name of the live run directory's parent (``<strategy>``).

Each **Strategy segment** names its live run directory (absolute), the
strategy, its first decision date and its last one (``None`` while it is
the current, open segment). ``live.decide`` keeps the history current:

- before a real decide connects, ``check`` refuses a live run directory
  whose segment another directory has taken over (one account runs one
  strategy at a time);
- after a real decide submitted orders, or adopted orders already working
  at IBKR (a first day that failed before recording), ``begin`` starts the
  directory's segment on t and ends the previous one at the bar before t;
  for the current segment's own directory it does nothing.

A dry run never reads or writes it. Every write replaces the file
atomically. The segments run before the history existed are entered once
by ``live history init`` (#66, ``AccountHistory.init``): each directory's
segment begins on its first order's decision date (``orders.zarr`` holds only
real orders) and, but for the last, ends on its last marked bar
(``equity.zarr``) before the next segment begins, written through the same
validation (``AccountHistory.write``).

The file::

    {"format": 1, "account": "DU1234567",
     "segments": [{"live_dir": "/data/quantlab/live/a/ibkr", "strategy": "a",
                   "first": "2026-10-01", "last": "2026-10-14"},
                  {"live_dir": "/data/quantlab/live/b/ibkr", "strategy": "b",
                   "first": "2026-10-15", "last": null}]}
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab_ibkr.outputs import read_equity

#: The account history file's layout version.
FORMAT = 1

#: The file name of an account's history.
FILE_NAME = "account_history_{account}.json"


class TakenOver(ValueError):
    """A real decide from a live run directory whose segment another directory has taken over.

    Examples
    --------
    >>> issubclass(TakenOver, ValueError)
    True
    """


@dataclass(frozen=True)
class StrategySegment:
    """The decision dates over which one live run directory traded the account.

    Attributes
    ----------
    live_dir : str
        The live run directory, absolute.
    strategy : str
        The strategy's name (the live run directory's parent's name).
    first : pandas.Timestamp
        The first decision date: the directory's first real submission.
    last : pandas.Timestamp or None
        The last decision date; ``None`` for the current segment.

    Examples
    --------
    >>> segment = StrategySegment("/live/a/ibkr", "a", pd.Timestamp("2026-10-01"), None)
    >>> segment.open
    True
    >>> StrategySegment.from_dict(segment.to_dict()) == segment
    True
    """

    live_dir: str
    strategy: str
    first: pd.Timestamp
    last: pd.Timestamp | None = None

    @property
    def open(self) -> bool:
        """Whether this is the current segment (no last decision date yet)."""
        return self.last is None

    def to_dict(self) -> dict:
        """Return the segment as the file holds it.

        Examples
        --------
        >>> StrategySegment("/live/a/ibkr", "a", pd.Timestamp("2026-10-01")).to_dict()
        {'live_dir': '/live/a/ibkr', 'strategy': 'a', 'first': '2026-10-01', 'last': None}
        """
        return {
            "live_dir": self.live_dir,
            "strategy": self.strategy,
            "first": str(self.first.date()),
            "last": None if self.last is None else str(self.last.date()),
        }

    @classmethod
    def from_dict(cls, data: dict) -> StrategySegment:
        """Return the segment ``to_dict`` wrote.

        Examples
        --------
        >>> StrategySegment.from_dict(
        ...     {"live_dir": "/l", "strategy": "a", "first": "2026-10-01", "last": "2026-10-02"}
        ... ).last
        Timestamp('2026-10-02 00:00:00')
        """
        last = data.get("last")
        return cls(
            data["live_dir"],
            data["strategy"],
            pd.Timestamp(data["first"]),
            None if last is None else pd.Timestamp(last),
        )

    def summary(self) -> str:
        """Return the segment's line in ``live history show``.

        Examples
        --------
        >>> StrategySegment("/live/a/ibkr", "a", pd.Timestamp("2026-10-01")).summary()
        'a  2026-10-01 to open  /live/a/ibkr'
        """
        last = "open" if self.last is None else str(self.last.date())
        return f"{self.strategy}  {self.first.date()} to {last}  {self.live_dir}"


def strategy_of(live_dir: str | Path) -> str:
    """Return the strategy name of a live run directory: its parent's name.

    Examples
    --------
    >>> strategy_of("/data/quantlab/live/us3000_tdM_mvo/ibkr")
    'us3000_tdM_mvo'
    """
    return Path(live_dir).resolve().parent.name


class AccountHistory:
    """One IBKR account's history file.

    Parameters
    ----------
    directory : str or pathlib.Path
        The directory holding the account histories.
    account : str
        The IBKR account id.

    Examples
    --------
    >>> import tempfile
    >>> history = AccountHistory(tempfile.mkdtemp(), "DU1234567")
    >>> history.path.name, history.segments
    ('account_history_DU1234567.json', ())
    >>> history.begin("/live/a/ibkr", pd.Timestamp("2026-10-01"), previous_bar=None)
    True
    >>> history.begin("/live/b/ibkr", pd.Timestamp("2026-10-05"),
    ...               previous_bar=pd.Timestamp("2026-10-02"))
    True
    >>> print(history.summary())
    account history of DU1234567: 2 strategy segments
    a  2026-10-01 to 2026-10-02  /live/a/ibkr
    b  2026-10-05 to open  /live/b/ibkr
    >>> history.check("/live/a/ibkr", pd.Timestamp("2026-10-06"))
    Traceback (most recent call last):
    ...
    quantlab_ibkr.account_history.TakenOver: account DU1234567: /live/a/ibkr's segment (a, 2026-10-01 to 2026-10-02) was taken over by /live/b/ibkr (b) on 2026-10-05; refusing to trade the account from it
    """

    def __init__(self, directory: str | Path, account: str):
        self.directory = Path(directory)
        self.account = account
        self.path = self.directory / FILE_NAME.format(account=account)

    @classmethod
    def of(cls, venue) -> AccountHistory:
        """Return the history of an ``IbkrVenueConfig``'s account, where the config keeps it.

        ``venue.history_dir`` when set, else the parent of the live run
        directory's parent. The account is ``venue.account()``, so the paper
        guard applies.

        Raises
        ------
        ValueError
            If the venue has neither ``history_dir`` nor ``live_dir``, or its
            account is refused.

        Examples
        --------
        >>> from quantlab_ibkr.venue.ibkr.venue import IbkrVenueConfig
        >>> AccountHistory.of(
        ...     IbkrVenueConfig(live_dir="/data/quantlab/live/a/ibkr", account_id="DU1234567")
        ... ).path
        PosixPath('/data/quantlab/live/account_history_DU1234567.json')
        """
        if venue.history_dir is not None:
            directory = Path(venue.history_dir)
        elif venue.live_dir is not None:
            directory = Path(venue.live_dir).resolve().parent.parent
        else:
            raise ValueError("an account history needs IbkrVenueConfig.history_dir or live_dir")
        return cls(directory, venue.account())

    @property
    def segments(self) -> tuple[StrategySegment, ...]:
        """The segments in order, oldest first; empty without a file."""
        if not self.path.is_file():
            return ()
        data = json.loads(self.path.read_text())
        if data.get("format") != FORMAT or data.get("account") != self.account:
            raise ValueError(
                f"{self.path} is not a format-{FORMAT} account history of {self.account} "
                f"(format {data.get('format')!r}, account {data.get('account')!r})"
            )
        return tuple(StrategySegment.from_dict(s) for s in data["segments"])

    @property
    def current(self) -> StrategySegment | None:
        """The open segment, if any."""
        segments = self.segments
        return segments[-1] if segments and segments[-1].open else None

    def is_current(self, live_dir: str | Path) -> bool:
        """Whether ``live_dir`` is the current (open) segment's directory.

        Examples
        --------
        >>> import tempfile
        >>> history = AccountHistory(tempfile.mkdtemp(), "DU1234567")
        >>> history.is_current("/live/a/ibkr")
        False
        >>> history.begin("/live/a/ibkr", pd.Timestamp("2026-10-01"), previous_bar=None)
        True
        >>> history.is_current("/live/a/ibkr")
        True
        """
        current = self.current
        return current is not None and current.live_dir == str(Path(live_dir).resolve())

    def check(self, live_dir: str | Path, t: pd.Timestamp) -> None:
        """Refuse a real decide of ``t`` from ``live_dir`` that cannot begin or continue a segment.

        Raises
        ------
        TakenOver
            If ``live_dir`` has a segment and is not the current one.
        ValueError
            If ``live_dir`` would begin a segment on or before the current
            segment's first decision date.

        Examples
        --------
        >>> import tempfile
        >>> history = AccountHistory(tempfile.mkdtemp(), "DU1234567")
        >>> history.check("/live/a/ibkr", pd.Timestamp("2026-10-01"))  # no history: allowed
        >>> history.begin("/live/a/ibkr", pd.Timestamp("2026-10-05"), previous_bar=None)
        True
        >>> history.check("/live/b/ibkr", pd.Timestamp("2026-10-05"))
        Traceback (most recent call last):
        ...
        ValueError: account DU1234567: /live/b/ibkr cannot begin a segment on 2026-10-05, on or before the current segment's first decision date 2026-10-05 (/live/a/ibkr)
        """
        live_dir = str(Path(live_dir).resolve())
        segments = self.segments
        mine = [k for k, s in enumerate(segments) if s.live_dir == live_dir]
        if mine and not (mine[-1] == len(segments) - 1 and segments[-1].open):
            own = segments[mine[-1]]
            if mine[-1] + 1 < len(segments):
                after = segments[mine[-1] + 1]
                by = f"taken over by {after.live_dir} ({after.strategy}) on {after.first.date()}"
            else:
                by = "closed"
            raise TakenOver(
                f"account {self.account}: {live_dir}'s segment ({own.strategy}, "
                f"{own.first.date()} to {own.last.date()}) was {by}; "
                f"refusing to trade the account from it"
            )
        current = self.current
        if not mine and current is not None and pd.Timestamp(t) <= current.first:
            raise ValueError(
                f"account {self.account}: {live_dir} cannot begin a segment on "
                f"{pd.Timestamp(t).date()}, on or before the current segment's first "
                f"decision date {current.first.date()} ({current.live_dir})"
            )

    def begin(
        self, live_dir: str | Path, t: pd.Timestamp, *, previous_bar: pd.Timestamp | None
    ) -> bool:
        """Begin ``live_dir``'s segment on ``t``, ending the current one at ``previous_bar``.

        Parameters
        ----------
        live_dir : str or pathlib.Path
            The live run directory that submitted real orders for ``t``.
        t : pandas.Timestamp
            The decision date of that submission.
        previous_bar : pandas.Timestamp or None
            The bar before ``t``, the current segment's last decision date;
            needed only when another directory's segment is open.

        Returns
        -------
        bool
            Whether a segment began (``False``: ``live_dir`` is the current
            segment's directory, nothing is written).

        Raises
        ------
        TakenOver, ValueError
            As ``check``.

        Examples
        --------
        >>> import tempfile
        >>> history = AccountHistory(tempfile.mkdtemp(), "DU1234567")
        >>> history.begin("/live/a/ibkr", pd.Timestamp("2026-10-01"), previous_bar=None)
        True
        >>> history.begin("/live/a/ibkr", pd.Timestamp("2026-10-02"), previous_bar=None)
        False
        >>> history.begin("/live/b/ibkr", pd.Timestamp("2026-10-05"),
        ...               previous_bar=pd.Timestamp("2026-10-02"))
        True
        >>> [(s.strategy, s.last) for s in history.segments]
        [('a', Timestamp('2026-10-02 00:00:00')), ('b', None)]
        """
        self.check(live_dir, t)
        live_dir = str(Path(live_dir).resolve())
        segments = list(self.segments)
        if segments and segments[-1].open:
            if segments[-1].live_dir == live_dir:
                return False
            if previous_bar is None:
                raise ValueError("ending the current segment needs the bar before t")
            segments[-1] = dataclasses.replace(segments[-1], last=pd.Timestamp(previous_bar))
        segments.append(StrategySegment(live_dir, strategy_of(live_dir), pd.Timestamp(t)))
        self.write(segments)
        return True

    def write(self, segments: Sequence[StrategySegment]) -> None:
        """Replace the file with ``segments``, atomically, after checking they are consistent.

        Raises
        ------
        ValueError
            If a segment ends before it begins, a closed segment follows an
            open one, segments overlap or are out of order, or a live run
            directory appears twice.

        Examples
        --------
        >>> import tempfile
        >>> history = AccountHistory(tempfile.mkdtemp(), "DU1234567")
        >>> history.write([
        ...     StrategySegment("/live/a/ibkr", "a", pd.Timestamp("2026-10-01")),
        ...     StrategySegment("/live/b/ibkr", "b", pd.Timestamp("2026-10-05")),
        ... ])
        Traceback (most recent call last):
        ...
        ValueError: segment /live/a/ibkr is open but not the last
        """
        for k, segment in enumerate(segments):
            if segment.last is not None and segment.last < segment.first:
                raise ValueError(f"segment {segment.live_dir} ends before it begins")
            if segment.open and k != len(segments) - 1:
                raise ValueError(f"segment {segment.live_dir} is open but not the last")
            if k and segment.first <= segments[k - 1].last:
                raise ValueError(
                    f"segment {segment.live_dir} begins on {segment.first.date()}, not after "
                    f"the previous segment's last decision date {segments[k - 1].last.date()}"
                )
        if len({s.live_dir for s in segments}) != len(segments):
            raise ValueError("a live run directory appears in two segments")
        payload = {
            "format": FORMAT, "account": self.account, "segments": [s.to_dict() for s in segments]
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        # Staged beside the file, so the replace is a same-filesystem rename.
        handle, staged = tempfile.mkstemp(dir=self.directory, prefix=f".{self.path.name}.")
        try:
            with os.fdopen(handle, "w") as file:
                json.dump(payload, file, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.chmod(staged, 0o644)  # mkstemp's 0600 otherwise
            os.replace(staged, self.path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise

    def init(self, live_dirs: Sequence[str | Path]) -> tuple[StrategySegment, ...]:
        """Write the segments ``live_dirs`` ran, in that order, into an account without a history.

        Each directory's segment begins on the decision date of its first
        submitted order, the bar ``decide`` would have begun it on: an order
        recorded ``not submitted:`` (past the order deadline, outside the
        after-open window) never reached IBKR, and a dry run records none.
        Each but the last ends on the last bar it marked (``equity.zarr``)
        before the next one began, the last stays open.

        Parameters
        ----------
        live_dirs : sequence of str or pathlib.Path
            The live run directories that traded the account, oldest first.

        Returns
        -------
        tuple of StrategySegment
            The segments written.

        Raises
        ------
        ValueError
            If the account already has a history; a directory submitted no
            order, has no equity record, or marked no bar of its segment; the
            next directory began on or before this one; this one has orders
            decided after the next one began; or ``write`` refuses the segments.

        Examples
        --------
        ::

            history = AccountHistory("/data/quantlab/live", "DU1234567")
            history.init(["/data/quantlab/live/a/ibkr", "/data/quantlab/live/b/ibkr"])
            print(history.summary())

        >>> import tempfile
        >>> AccountHistory(tempfile.mkdtemp(), "DU1234567").init([])
        Traceback (most recent call last):
        ...
        ValueError: live history init needs at least one live run directory
        """
        if not live_dirs:
            raise ValueError("live history init needs at least one live run directory")
        if self.segments:
            raise ValueError(
                f"account {self.account} already has a history ({self.path}); "
                "live history init only enters the segments run before it existed"
            )
        traded = [_Traded.read(Path(d)) for d in live_dirs]
        segments = []
        for k, this in enumerate(traded):
            first, last = this.orders.min(), None
            if k + 1 < len(traded):
                following = traded[k + 1]
                begins = following.orders.min()
                if begins <= first:
                    raise ValueError(
                        f"{following.live_dir} begins on {begins.date()}, not after "
                        f"{this.live_dir} began on {first.date()}"
                    )
                if (this.orders > begins).any():
                    raise ValueError(
                        f"{this.live_dir} has orders decided on {this.orders.max().date()}, "
                        f"after {following.live_dir} began on {begins.date()}"
                    )
                last = this.marked[this.marked < begins].max()
            span = this.marked[this.marked >= first]
            if last is not None:
                span = span[span <= last]
            if span.empty:
                raise ValueError(
                    f"{this.live_dir} marked no bar of its segment from {first.date()} on"
                )
            segments.append(StrategySegment(this.live_dir, strategy_of(this.live_dir), first, last))
        self.write(segments)
        return tuple(segments)

    def summary(self) -> str:
        """Return what ``live history show`` prints.

        Examples
        --------
        >>> import tempfile
        >>> AccountHistory(tempfile.mkdtemp(), "DU1234567").summary()
        'no account history for DU1234567'
        """
        segments = self.segments
        if not segments:
            return f"no account history for {self.account}"
        lines = [f"account history of {self.account}: {len(segments)} strategy segments"]
        lines += [s.summary() for s in segments]
        return "\n".join(lines)


#: The reason prefix of an order recorded but never sent to IBKR
#: (``venue.ibkr.submitter.PAST_DEADLINE``, ``OUTSIDE_SESSION``).
NOT_SUBMITTED = "not submitted:"


@dataclass(frozen=True)
class _Traded:
    """What ``init`` reads of a live run directory: its submitted orders' decision dates and marked bars."""

    live_dir: str
    orders: pd.DatetimeIndex
    marked: pd.DatetimeIndex

    @classmethod
    def read(cls, live_dir: Path) -> _Traded:
        try:
            with xr.open_zarr(live_dir / "orders.zarr") as orders:
                sent = np.array(
                    [not str(reason).startswith(NOT_SUBMITTED) for reason in orders["reason"].values],
                    dtype=bool,
                )
                decided = pd.DatetimeIndex(orders["decision_date"].values[sent])
        except FileNotFoundError:
            decided = pd.DatetimeIndex([])
        if decided.empty:
            raise ValueError(f"{live_dir} submitted no order: it never traded the account")
        try:
            marked = pd.DatetimeIndex(read_equity(live_dir)["timestamp"].values)
        except FileNotFoundError:
            raise ValueError(f"{live_dir} has no equity record") from None
        return cls(str(live_dir.resolve()), decided, marked)
