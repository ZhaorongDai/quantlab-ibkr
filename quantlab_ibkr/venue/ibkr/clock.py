"""The IBKR venue's decision clock: fire once, for the last closed bar.

A live invocation decides at most one bar t (the live source's). The clock
fires ``strategy.on_decision_time(t)`` once, as soon as the strategy starts
(after the node has connected and reconciled the account), on every day, as
the backtest's clock fires on every bar: the cycle marks the account at t's
raw close, so the live run directory has a row of equity and holdings per
bar (#51). It **decides** t only when t is a rebalance bar of the run's
cadence and the source can decide it (``LiveDecision.decides``); on any
other day the venue's targets (``IbkrVenue.targets``) decide nothing. The cadence
is quantlab's: ``DecisionInputs.rebalances`` of the run's decision inputs with
an open-ended schedule (``QuantlabRun.decision_inputs()``), counting every
``rebalance_periods`` bars of the price dataset's calendar from the run's
anchor and on past the run's last bar (ADR 0008). A day that does not decide
holds, with its reason.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from quantlab_ibkr.base.venue import DecisionClock

if TYPE_CHECKING:
    from quantlab_ibkr.strategy import PortfolioStrategy

#: The hold reason of a bar off the run's rebalance cadence.
NOT_A_REBALANCE_BAR = "{t} is not a rebalance bar of the run's cadence"


@dataclass(frozen=True)
class LiveDecision:
    """Whether the live clock decides its bar, and why not when it holds.

    Attributes
    ----------
    t : pandas.Timestamp
        The bar.
    rebalances : bool
        t is a rebalance bar of the run's cadence.
    hold_reason : str or None
        Why t is not decided; ``None`` when it is.

    Examples
    --------
    >>> LiveDecision(pd.Timestamp("2026-10-07"), True, None).decides
    True
    """

    t: pd.Timestamp
    rebalances: bool
    hold_reason: str | None

    @property
    def decides(self) -> bool:
        """Whether the clock fires the decision of t."""
        return self.hold_reason is None


class LiveDecisionClock(DecisionClock):
    """Fire the strategy's decision cycle once for t; ``decision`` says whether it decides.

    Parameters
    ----------
    t : pandas.Timestamp
        The bar (the live source's ``t``).
    rebalances : Callable
        ``rebalances(t) -> bool``, the run's cadence
        (``QuantlabRun.decision_inputs().rebalances``).
    hold_reason : str or None, optional
        Why the source cannot decide t (``LiveDecisionSource.hold_reason``).
    force : bool, default False
        Treat t as a rebalance bar whatever the cadence says (a dry run's
        ``force_decide``); a ``hold_reason`` still holds.
    on_schedule : Callable, optional
        Called with the strategy when it starts, before the decision fires
        (the venue reads the reconciled account there).

    Attributes
    ----------
    decision : LiveDecision
        Whether t is decided, and why not.
    fired : bool
        The cycle of t has run.
    error : BaseException or None
        What the decision raised, kept for the venue to raise after stopping
        the node.

    Examples
    --------
    >>> t = pd.Timestamp("2026-10-07")
    >>> LiveDecisionClock(t, lambda bar: True).decision.decides
    True
    >>> LiveDecisionClock(t, lambda bar: False).decision.hold_reason
    "2026-10-07 is not a rebalance bar of the run's cadence"
    >>> LiveDecisionClock(t, lambda bar: True, "no prediction row").decision.hold_reason
    'no prediction row'
    >>> LiveDecisionClock(t, lambda bar: False, force=True).decision.decides
    True
    """

    def __init__(
        self,
        t: pd.Timestamp,
        rebalances: Callable[[pd.Timestamp], bool],
        hold_reason: str | None = None,
        *,
        force: bool = False,
        on_schedule: Callable[[PortfolioStrategy], None] | None = None,
    ):
        t = pd.Timestamp(t)
        rebalance = force or bool(rebalances(t))
        if not rebalance:
            reason = NOT_A_REBALANCE_BAR.format(t=t.date())
        else:
            reason = hold_reason
        self.decision = LiveDecision(t, rebalance, reason)
        self._on_schedule = on_schedule
        self.scheduled = False
        self.fired = False
        self.error: BaseException | None = None

    @property
    def done(self) -> bool:
        """Whether the clock has nothing left to do: the cycle of t ran (or could not).

        Examples
        --------
        >>> LiveDecisionClock(pd.Timestamp("2026-10-07"), lambda bar: False).done
        False
        """
        return self.scheduled and self.fired

    def schedule(self, strategy: PortfolioStrategy) -> None:
        """Run ``on_schedule``, then fire ``strategy.on_decision_time(t)`` now.

        The decision is a time alert at the strategy clock's current time,
        which a live clock fires at once. An error ``on_schedule`` or the
        decision raises is kept in ``error`` (no decision follows an
        ``on_schedule`` error).

        Examples
        --------
        A stand-in for the strategy whose clock fires each alert at once:

        >>> from types import SimpleNamespace
        >>> class FiringClock:
        ...     def utc_now(self):
        ...         return pd.Timestamp("2026-10-08 12:00", tz="UTC")
        ...     def set_time_alert(self, name, alert_time, callback):
        ...         print(name)
        ...         callback(None)
        >>> strategy = SimpleNamespace(
        ...     clock=FiringClock(), on_decision_time=lambda t: print("decide", t.date())
        ... )
        >>> clock = LiveDecisionClock(pd.Timestamp("2026-10-07"), lambda bar: True)
        >>> clock.schedule(strategy)
        decision-2026-10-07
        decide 2026-10-07
        >>> clock.done
        True
        """
        self.scheduled = True
        if self._on_schedule is not None:
            try:
                self._on_schedule(strategy)
            except Exception as error:  # no decision on an account it could not read
                self.error = error
                self.fired = True
                return
        t = self.decision.t

        def fire(_event) -> None:
            try:
                strategy.on_decision_time(t)
            except Exception as error:  # raised again by the venue once the node stops
                self.error = error
            finally:
                self.fired = True

        strategy.clock.set_time_alert(f"decision-{t.date()}", strategy.clock.utc_now(), fire)
