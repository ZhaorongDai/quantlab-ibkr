"""The venue: everything that differs between the backtest and live trading (ADR 0008).

A venue is four parts. The ``InstrumentResolver`` maps a PERMNO to the venue's
instrument (ADR 0004), the ``OpenSubmitter`` sends next-open orders to the
market (ADR 0003), the ``DecisionSource`` hands over one bar's
``DecisionInputs`` reading nothing later than that bar, and the
``DecisionClock`` fires the strategy's one decision method after the close.
The market data feed, fee and fill models and corporate actions are not
parts: they are internals of the backtest venue, because live they are the
broker's. The account is not a part either; nautilus's ``Cache`` and
``Portfolio`` already present one account interface in both modes.

This module imports no nautilus code at run time, so the decision core can
use its value types (``DecisionInputs``, ``NextOpenOrder``) on plain arrays.
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

    from quantlab_trader.strategy import PortfolioStrategy

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
        decision source hands out and whose symbols the decision inputs are
        on; ``None`` in open loop.
    history_start : pandas.Timestamp or None
        Closed loop: where the decision-price history starts,
        ``bar_before(anchor, lookback_bars)`` (ADR 0008); ``None`` means
        ``start``.

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
    history_start: pd.Timestamp | None = None

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

    Examples
    --------
    >>> VenueReport().minimum_fee_orders
    frozenset()
    """

    minimum_fee_orders: frozenset[str] = frozenset()


@dataclass(frozen=True)
class DecisionInputs:
    """What one decision cycle may know at the close of bar t.

    Every series is indexed by PERMNO, quantlab's symbol label kept as is
    (an integer for a CRSP panel).

    Attributes
    ----------
    timestamp : pandas.Timestamp
        The decision date t.
    predictions : xarray.Dataset or None
        One row of the prediction panel, a variable per label on ``symbol``;
        ``None`` when the target source does not read predictions (open loop).
    tradable : pandas.Series
        Booleans: the security has a real fill price at t
        (``MarketDataset.tradable_bars`` on the run's fill column).
    close : pandas.Series
        Raw close of t, the last raw close for a security without a price at
        t; NaN before a security's first price. Sizes orders and marks equity.
    valuation_history : xarray.DataArray or None
        Decision prices (the run's valuation column) up to and including t,
        on ``(timestamp, symbol)``; ``None`` where nothing reads them.
    delisted : pandas.Series
        Booleans: t is the security's delisting bar
        (``MarketDataset.delisting_bars``).

    Examples
    --------
    >>> close = pd.Series({10001: 30.0})
    >>> inputs = DecisionInputs(
    ...     timestamp=pd.Timestamp("2024-01-03"), predictions=None,
    ...     tradable=close.notna(), close=close, valuation_history=None,
    ...     delisted=close.isna(),
    ... )
    >>> inputs.close[10001]
    np.float64(30.0)
    """

    timestamp: pd.Timestamp
    predictions: xr.Dataset | None
    tradable: pd.Series
    close: pd.Series
    valuation_history: xr.DataArray | None
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
    """A venue's two-way mapping between PERMNOs and its instruments (ADR 0004)."""

    @abstractmethod
    def instrument_id(self, permno: Hashable, as_of: pd.Timestamp) -> InstrumentId:
        """Return the venue's instrument for ``permno`` on decision date ``as_of``."""

    @abstractmethod
    def permno(self, instrument_id: InstrumentId) -> Hashable:
        """Return the PERMNO of ``instrument_id``, as quantlab's symbol label."""

    @abstractmethod
    def instruments(self) -> Sequence[Instrument]:
        """Return every instrument the venue trades in this run."""


class OpenSubmitter(ABC):
    """Sends next-open orders to the market (ADR 0003).

    The only order code that differs between venues. An order that ends
    without a fill is reported through ``strategy.on_next_open_unfilled``.
    """

    @abstractmethod
    def attach(self, strategy: PortfolioStrategy) -> None:
        """Bind the strategy that submits the orders and hears about unfilled ones."""

    @abstractmethod
    def submit(self, orders: Sequence[NextOpenOrder], decision_date: pd.Timestamp) -> None:
        """Take the orders decided at the close of ``decision_date`` for the next open."""


class DecisionSource(ABC):
    """Supplies one bar's decision inputs, reading nothing later than that bar."""

    @abstractmethod
    def inputs(self, t: pd.Timestamp) -> DecisionInputs:
        """Return the inputs of the decision at the close of ``t``."""

    @abstractmethod
    def calendar(self) -> pd.DatetimeIndex:
        """Return the bars a decision cycle runs on."""


class DecisionClock(ABC):
    """Fires the decision cycle after the close of each bar of the source's calendar."""

    @abstractmethod
    def schedule(self, strategy: PortfolioStrategy) -> None:
        """Arrange for ``strategy.on_decision_time(t)`` to be called after each close."""


class Venue(ABC):
    """The four parts that differ between backtest and live, and the run loop.

    Fees, fills, the feed and corporate actions are not on this interface
    (ADR 0008): what the run's outputs need from them comes back from
    ``run`` as a ``VenueReport``.

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
        """
