"""The venue: everything that differs between the backtest and live trading (ADR 0008).

Live trading is IBKR's, the only broker (ADR 0010): the seam separates the
backtest from IBKR, not one broker from another.

A venue is four parts. The ``InstrumentResolver`` maps a PERMNO to the venue's
instrument (ADR 0004), the ``OpenSubmitter`` sends next-open orders to the
market (ADR 0003), the ``DecisionSource`` hands over one bar's
``BarInputs`` reading nothing later than that bar, and the
``DecisionClock`` fires the strategy's one decision method after the close.
The market data feed, fee and fill models and corporate actions are not
parts: they are internals of the backtest venue, because live they are the
broker's. The account is not a part either; nautilus's ``Cache`` and
``Portfolio`` already present one account interface in both modes.

This module imports no nautilus code at run time, so the decision core can
use its value types (``BarInputs``, ``NextOpenOrder``) on plain arrays.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.instruments import Instrument

    from quantlab_ibkr.strategy import PortfolioStrategy

#: The time zone of the market every venue trades (US equities): fills and
#: events are dated in it.
MARKET_TZ = "America/New_York"

#: Tag prefix of a venue fill: an order and fill a venue books on the
#: strategy's position for a corporate action, tagged
#: ``CORPORATE_ACTION_<KIND>`` (``CORPORATE_ACTION_DELIST``, ...). It is never
#: one of the strategy's next-open orders (ADR 0009).
CORPORATE_ACTION_TAG = "CORPORATE_ACTION"


def corporate_action_kind(tags) -> str | None:
    """Return the corporate action an order's tags mark it with, or ``None``.

    Parameters
    ----------
    tags : Iterable of str or None
        An order's tags.

    Examples
    --------
    >>> corporate_action_kind(["CORPORATE_ACTION_DELIST"])
    'DELIST'
    >>> corporate_action_kind(["CORPORATE_ACTIONS"]) is None
    True
    """
    prefix = CORPORATE_ACTION_TAG + "_"
    for tag in tags or ():
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return None


class Loop(StrEnum):
    """A replay's loop: where a decision cycle's target weights come from.

    ``CLOSED`` runs quantlab's constructor on the account's holdings (the
    same loop as live trading); ``OPEN`` executes the quantlab run's
    rebalance table as it is. Its value is its JSON form.

    Examples
    --------
    >>> Loop("open") is Loop.OPEN, f"{Loop.CLOSED} loop"
    (True, 'closed loop')
    """

    CLOSED = "closed"
    OPEN = "open"


@dataclass(frozen=True, eq=False)
class ReplayRequest:
    """What a venue is asked to execute: the window, the instruments and the loop.

    ``VenueConfig.build`` takes one; a venue reads what it needs and ignores
    the rest (a live venue has no prediction panel to replay). Compared by
    identity, as it carries a prediction panel.

    Attributes
    ----------
    start, end : pandas.Timestamp
        The window, both inclusive.
    permnos : tuple
        The securities the strategy can trade (quantlab's symbol labels).
    loop : Loop
        The replay's loop (a string is converted to its ``Loop``); the backtest venue picks its default fee model
        from it (ADR 0003).
    predictions : xarray.Dataset or None
        Closed loop: the prediction panel over the window, whose rows the
        decision source hands out; ``None`` in open loop.

    Examples
    --------
    >>> request = ReplayRequest(
    ...     pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05"), (10001,), Loop.OPEN
    ... )
    >>> request.loop, request.predictions
    (<Loop.OPEN: 'open'>, None)
    """

    start: pd.Timestamp
    end: pd.Timestamp
    permnos: tuple
    loop: Loop
    predictions: xr.Dataset | None = None

    def __post_init__(self):
        object.__setattr__(self, "loop", Loop(self.loop))


@dataclass(frozen=True)
class VenueReport:
    """What only the venue knows about a finished run's execution.

    Attributes
    ----------
    minimum_fee_orders : frozenset of str
        Client order ids of the orders whose fill was charged the venue's
        minimum commission, counted in a trader run's
        ``execution.trader.minimum_fee_hits``; empty for a venue without a
        minimum. Keyed by order because the backtest venue fills an order
        whole; a venue that fills in parts reports only orders whose every
        fill hit the minimum.
    fees : str or None
        The fee model the venue charged, as the report's Setup states it;
        ``None`` when the venue does not say.
    slippage : float or None
        The fractional slippage the venue filled with; ``None`` when it does
        not say.

    Examples
    --------
    >>> VenueReport().minimum_fee_orders
    frozenset()
    >>> VenueReport(fees="0.001 of the traded notional", slippage=0.0).fees
    '0.001 of the traded notional'
    """

    minimum_fee_orders: frozenset[str] = frozenset()
    fees: str | None = None
    slippage: float | None = None


@dataclass(frozen=True)
class BarInputs:
    """What one decision cycle reads from the venue at the close of bar t.

    Every series is indexed by PERMNO, quantlab's symbol label kept as is
    (an integer for a CRSP panel). The rule's decision inputs (tradability,
    the decision-price window, factors) are not here: quantlab's
    ``DecisionInputs`` reads them from the run's price dataset (ADR 0008).

    Attributes
    ----------
    timestamp : pandas.Timestamp
        The decision date t.
    predictions : xarray.Dataset or None
        One row of the prediction panel, a variable per label on ``symbol``;
        ``None`` when the target source does not read predictions (open loop).
    close : pandas.Series
        Raw close of t, the last raw close for a security without a price at
        t; NaN before a security's first price. Sizes orders and marks equity.
    delisted : pandas.Series
        Booleans: t is the security's delisting bar
        (``MarketDataset.delisting_bars``).

    Examples
    --------
    >>> close = pd.Series({10001: 30.0})
    >>> inputs = BarInputs(
    ...     timestamp=pd.Timestamp("2024-01-03"), predictions=None,
    ...     close=close, delisted=close.isna(),
    ... )
    >>> inputs.close[10001]
    np.float64(30.0)
    """

    timestamp: pd.Timestamp
    predictions: xr.Dataset | None
    close: pd.Series
    delisted: pd.Series


@dataclass(frozen=True)
class NextOpenOrder:
    """A whole-share order sized at the close of t, meant to fill at the next open.

    Attributes
    ----------
    permno : Hashable
        The security, as quantlab's symbol label.
    side : {"BUY", "SELL"}
        The order side.
    quantity : int
        Shares, positive.
    decision_date : pandas.Timestamp
        The bar whose close sized the order.

    Examples
    --------
    >>> NextOpenOrder(10001, "BUY", 25, pd.Timestamp("2024-01-03")).quantity
    25
    """

    permno: Hashable
    side: Literal["BUY", "SELL"]
    quantity: int
    decision_date: pd.Timestamp


class InstrumentResolver(ABC):
    """A venue's two-way mapping between PERMNOs and its instruments (ADR 0004).

    Examples
    --------
    The backtest venue's resolver maps each PERMNO onto ``<PERMNO>.CRSP``:

    >>> from quantlab_ibkr.venue.backtest.resolver import BacktestResolver
    >>> resolver = BacktestResolver([10107], pd.Timestamp("2024-01-02"))
    >>> isinstance(resolver, InstrumentResolver)
    True
    """

    @abstractmethod
    def instrument_id(self, permno: Hashable, as_of: pd.Timestamp) -> InstrumentId:
        """Return the venue's instrument for ``permno`` on decision date ``as_of``.

        Examples
        --------
        >>> from quantlab_ibkr.venue.backtest.resolver import BacktestResolver
        >>> resolver = BacktestResolver([10107], pd.Timestamp("2024-01-02"))
        >>> str(resolver.instrument_id(10107, pd.Timestamp("2024-01-03")))
        '10107.CRSP'
        """

    @abstractmethod
    def permno(self, instrument_id: InstrumentId) -> Hashable:
        """Return the PERMNO of ``instrument_id``, as quantlab's symbol label.

        Examples
        --------
        >>> from nautilus_trader.model.identifiers import InstrumentId
        >>> from quantlab_ibkr.venue.backtest.resolver import BacktestResolver
        >>> resolver = BacktestResolver([10107], pd.Timestamp("2024-01-02"))
        >>> resolver.permno(InstrumentId.from_str("10107.CRSP"))
        10107
        """

    @abstractmethod
    def instruments(self) -> Sequence[Instrument]:
        """Return every instrument the venue trades in this run.

        Examples
        --------
        >>> from quantlab_ibkr.venue.backtest.resolver import BacktestResolver
        >>> resolver = BacktestResolver([10107, 14593], pd.Timestamp("2024-01-02"))
        >>> [str(instrument.id) for instrument in resolver.instruments()]
        ['10107.CRSP', '14593.CRSP']
        """


class OpenSubmitter(ABC):
    """Sends next-open orders to the market (ADR 0003).

    The only order code that differs between venues. An order that ends
    without a fill is reported through ``strategy.on_next_open_unfilled``.

    Examples
    --------
    The strategy attaches when it starts and hands over each cycle's orders::

        venue.submitter.attach(strategy)  # in strategy.on_start()
        venue.submitter.submit(result.orders, t)  # after each decision
    """

    @abstractmethod
    def attach(self, strategy: PortfolioStrategy) -> None:
        """Bind the strategy that submits the orders and hears about unfilled ones.

        Examples
        --------
        ``PortfolioStrategy.on_start`` attaches it::

            venue.submitter.attach(strategy)
        """

    @abstractmethod
    def submit(self, orders: Sequence[NextOpenOrder], decision_date: pd.Timestamp) -> None:
        """Take the orders decided at the close of ``decision_date`` for the next open.

        Examples
        --------
        ``PortfolioStrategy.on_decision_time`` submits each cycle's orders,
        sells first::

            venue.submitter.submit(result.orders, t)
        """


class DecisionSource(ABC):
    """Supplies one bar's inputs, reading nothing later than that bar.

    Examples
    --------
    The strategy reads one bar's inputs at each decision and runs the cycle
    on them::

        inputs = venue.source.inputs(t)
        result = cycle.run(inputs, positions, cash)
    """

    @abstractmethod
    def inputs(self, t: pd.Timestamp) -> BarInputs:
        """Return the inputs of the decision at the close of ``t``.

        Examples
        --------
        ::

            inputs = venue.source.inputs(pd.Timestamp("2024-01-03"))
            result = cycle.run(inputs, positions, cash)
        """

    @abstractmethod
    def calendar(self) -> pd.DatetimeIndex:
        """Return the bars a decision cycle runs on.

        Examples
        --------
        The backtest venue builds its clock and submitter on it::

            calendar = venue.source.calendar()
            clock = BacktestDecisionClock(calendar)
        """


class DecisionClock(ABC):
    """Fires the decision cycle after the close of each bar of the source's calendar.

    Examples
    --------
    ``PortfolioStrategy.on_start`` hands the strategy to the clock::

        venue.clock.schedule(strategy)
    """

    @abstractmethod
    def schedule(self, strategy: PortfolioStrategy) -> None:
        """Arrange for ``strategy.on_decision_time(t)`` to be called after each close.

        Examples
        --------
        ``PortfolioStrategy.on_start`` calls it once::

            venue.clock.schedule(strategy)
        """


class Venue(ABC):
    """The four parts that differ between backtest and live, and the run loop.

    Fees, fills, the feed and corporate actions are not on this interface
    (ADR 0008): what the run's outputs need from them comes back from
    ``run`` as a ``VenueReport``.

    Besides nautilus's order events, a venue tells the strategy two things
    through its venue events, which any venue may emit:
    ``strategy.on_next_open_unfilled(order, reason)`` for a next-open order
    that ended without a fill, and ``strategy.on_corporate_action(...)`` for
    a corporate action applied to a holding without a fill (the backtest
    venue books them from the run's price dataset; a live venue reports the
    broker's).

    Attributes
    ----------
    resolver : InstrumentResolver
    submitter : OpenSubmitter
    source : DecisionSource
    clock : DecisionClock
    init_cash : float
        The account's starting cash, the base of the first bar's return and
        the run's Start Value: the simulated deposit in a backtest, the
        account's cash when the node starts live. Not a part: the run
        directory needs it, the strategy never reads it.

    Examples
    --------
    ``runner.run`` builds the venue from the config and runs the strategy on
    it::

        venue = config.venue.build(quantlab_run, request)
        report = venue.run(PortfolioStrategy(venue=venue, cycle=cycle, recorder=recorder))
    """

    init_cash: float
    resolver: InstrumentResolver
    submitter: OpenSubmitter
    source: DecisionSource
    clock: DecisionClock

    @abstractmethod
    def run(self, strategy: PortfolioStrategy) -> VenueReport:
        """Build and run the engine (backtest) or node (live) around ``strategy``.

        Returns
        -------
        VenueReport
            What the venue alone knows about the run's execution.

        Examples
        --------
        ::

            report = venue.run(strategy)
            run_dir = recorder.write(report)
        """
