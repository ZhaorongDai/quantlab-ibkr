"""``PortfolioStrategy``: the one nautilus ``Strategy``, in the backtest and live (ADR 0008).

It knows nothing about the venue it runs on beyond the four venue parts: the
decision clock calls ``on_decision_time(t)`` after the close of each bar, the
strategy reads the account, runs the ``DecisionCycle`` on the decision
source's inputs and hands the orders to the open submitter, which submits each
through ``submit_next_open`` at the time and with the time in force its venue
needs. A diff to this module when a venue is added is a design failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from quantlab_trader.account import derived_cash, holdings
from quantlab_trader.base.venue import NextOpenOrder, Venue, corporate_action_kind
from quantlab_trader.decision import DecisionCycle

if TYPE_CHECKING:
    from quantlab_trader.outputs import RunRecorder


class PortfolioStrategy(Strategy):
    """Decide after the close, submit for the next open, record everything.

    Parameters
    ----------
    venue : Venue
        The venue whose parts the strategy uses.
    cycle : DecisionCycle
        The decision core.
    recorder : RunRecorder
        Collects decisions, equity, orders and events for the run directory.
    config : nautilus_trader.config.StrategyConfig, optional
        nautilus's strategy config.
    """

    def __init__(
        self,
        *,
        venue: Venue,
        cycle: DecisionCycle,
        recorder: RunRecorder,
        config: StrategyConfig | None = None,
    ):
        super().__init__(config or StrategyConfig())
        self.venue = venue
        self.cycle = cycle
        self.recorder = recorder

    def on_start(self) -> None:
        """Attach to the submitter and let the venue's clock schedule the decisions."""
        self.venue.submitter.attach(self)
        self.venue.clock.schedule(self)

    def on_decision_time(self, t: pd.Timestamp) -> None:
        """Run one decision cycle at the close of ``t``: the one decision method."""
        inputs = self.venue.source.inputs(t)
        result = self.cycle.run(
            inputs, holdings(self.cache, self.venue.resolver), derived_cash(self.cache)
        )
        self.recorder.record_cycle(t, result)
        self.venue.submitter.submit(result.orders, t)

    def submit_next_open(self, order: NextOpenOrder, time_in_force: TimeInForce) -> None:
        """Submit ``order`` as a market order now; the submitter picks when and how.

        Parameters
        ----------
        order : NextOpenOrder
            The order decided at the close.
        time_in_force : nautilus_trader.model.enums.TimeInForce
            ``DAY`` at the backtest's opening print; ``AT_THE_OPEN`` live.
        """
        instrument_id = self.venue.resolver.instrument_id(order.permno, order.decision_date)
        market_order = self.order_factory.market(
            instrument_id,
            OrderSide[order.side],
            Quantity(order.quantity, 0),
            time_in_force=time_in_force,
        )
        self.recorder.order_submitted(order, market_order.client_order_id.value)
        self.submit_order(market_order)

    def on_next_open_unfilled(self, order: NextOpenOrder, reason: str) -> None:
        """Record a next-open order that ended without a fill; the holding is kept."""
        self.recorder.order_unfilled(order, reason)

    def on_corporate_action(
        self,
        action: str,
        *,
        ts_ns: int,
        permno,
        quantity: int,
        amount: float,
        **detail,
    ) -> None:
        """Record a corporate action the venue applied to a holding without a fill.

        Cash it moved (``DIVIDEND``, ``CASH_IN_LIEU``, ``DISTRIBUTION``) or a
        factor day it left the holding alone on (``FINAL``, ``OTHER``).

        Parameters
        ----------
        action : str
            The kind.
        ts_ns : int
            When it was applied, UNIX nanoseconds.
        permno : Hashable
            The security.
        quantity : int
            The signed holding it applied to.
        amount : float
            Cash moved into the account (negative: paid out).
        **detail
            Facts of the action (``per_share``, ``split_factor``, ...).
        """
        self.recorder.corporate_action_cash(
            action, ts_ns=ts_ns, permno=permno, quantity=quantity, amount=amount, **detail
        )

    def on_order_filled(self, event) -> None:
        """Record a fill of one of the strategy's orders, or a venue fill as an event.

        A fill tagged ``CORPORATE_ACTION_<KIND>`` is a venue fill (ADR 0009):
        the venue booked it on the strategy's position for a corporate
        action, so it is recorded in the run's events, never as an order.
        """
        kind = corporate_action_kind(self.cache.order(event.client_order_id).tags)
        if kind is not None:
            self.recorder.corporate_action(
                kind,
                ts_ns=event.ts_event,
                permno=self.venue.resolver.permno(event.instrument_id),
                side=event.order_side.name,
                quantity=int(event.last_qty.as_double()),
                price=event.last_px.as_double(),
                fee=event.commission.as_double(),
            )
            return
        self.recorder.order_filled(
            event.client_order_id.value,
            price=event.last_px.as_double(),
            quantity=int(event.last_qty.as_double()),
            fee=event.commission.as_double(),
        )

    def on_order_rejected(self, event) -> None:
        """Record an order the venue rejected."""
        self.recorder.order_refused(event.client_order_id.value, "rejected", str(event.reason))

    def on_order_denied(self, event) -> None:
        """Record an order nautilus's risk engine denied."""
        self.recorder.order_refused(event.client_order_id.value, "denied", str(event.reason))
