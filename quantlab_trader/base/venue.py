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

    Attributes
    ----------
    resolver : InstrumentResolver
    submitter : OpenSubmitter
    source : DecisionSource
    clock : DecisionClock
    init_cash : float
        The account's starting cash, the base of the first bar's return (the
        simulated deposit in a backtest; the account's cash at start live).
    """

    init_cash: float
    resolver: InstrumentResolver
    submitter: OpenSubmitter
    source: DecisionSource
    clock: DecisionClock

    @abstractmethod
    def run(self, strategy: PortfolioStrategy) -> None:
        """Build and run the engine (backtest) or node (live) around ``strategy``."""
