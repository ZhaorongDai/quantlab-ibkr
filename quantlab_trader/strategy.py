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

    Examples
    --------
    ``runner.run`` builds the strategy and hands it to the venue, which adds it
    to its engine (backtest) or node (live)::

        strategy = PortfolioStrategy(
            venue=venue, cycle=DecisionCycle(targets), recorder=recorder
        )
        report = venue.run(strategy)
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
        """Attach to the submitter and let the venue's clock schedule the decisions.

        Examples
        --------
        nautilus calls it when the engine or node starts the strategy::

            engine.add_strategy(strategy)
            engine.run()  # calls strategy.on_start()
        """
        self.venue.submitter.attach(self)
        self.venue.clock.schedule(self)

    def on_decision_time(self, t: pd.Timestamp) -> None:
        """Run one decision cycle at the close of ``t``: the one decision method.

        Parameters
        ----------
        t : pandas.Timestamp
            The decision date, a bar of the decision source's calendar.

        Examples
        --------
        The venue's decision clock calls it after each close; the backtest's
        sets a time alert per bar::

            strategy.clock.set_time_alert(
                "decision-2024-01-03",
                alert_time,
                lambda _event: strategy.on_decision_time(pd.Timestamp("2024-01-03")),
            )
        """
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

        Examples
        --------
        The venue's open submitter calls it when the order should reach the
        market; the backtest's does so one nanosecond after the next open::

            strategy.submit_next_open(
                NextOpenOrder(10001, "BUY", 25, pd.Timestamp("2024-01-03")),
                TimeInForce.DAY,
            )
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
        """Record a next-open order that ended without a fill; the holding is kept.

        Parameters
        ----------
        order : NextOpenOrder
            The order decided at the close.
        reason : str
            Why it was not filled, recorded in ``orders.zarr`` and ``events.json``.

        Examples
        --------
        The venue's open submitter calls it, for example for an order whose
        security has no opening print on its fill bar::

            strategy.on_next_open_unfilled(order, "no opening print on 2024-01-04")
        """
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

        A venue event, like ``on_next_open_unfilled``: any venue may call it,
        and the strategy only records what it is told. The backtest venue's
        corporate-action module calls it for the actions it books from the
        run's price dataset; a live venue reports the broker's corporate
        actions on the account (dividends credited, cash in lieu of
        fractional shares) through it. A corporate action that changes a
        share count arrives as a venue fill instead (``on_order_filled``,
        ADR 0009).

        The kinds are cash moved (``DIVIDEND``, ``CASH_IN_LIEU``,
        ``DISTRIBUTION``), a day the holding was left alone on (``FINAL``,
        ``OTHER``, ``MISMATCH``), or a delisting payment the settlement pays
        instead (``DELISTING_PAYMENT``).

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

        Examples
        --------
        A venue reporting a USD 0.24 dividend on 100 shares held::

            strategy.on_corporate_action(
                "DIVIDEND", ts_ns=ts_ns, permno=10001, quantity=100,
                amount=24.0, per_share=0.24,
            )
        """
        self.recorder.corporate_action_cash(
            action, ts_ns=ts_ns, permno=permno, quantity=quantity, amount=amount, **detail
        )

    def on_order_filled(self, event) -> None:
        """Record a fill of one of the strategy's orders, or a venue fill as an event.

        A fill tagged ``CORPORATE_ACTION_<KIND>`` is a venue fill (ADR 0009):
        the venue booked it on the strategy's position for a corporate
        action, so it is recorded in the run's events, never as an order.

        Parameters
        ----------
        event : nautilus_trader.model.events.OrderFilled
            The fill.

        Examples
        --------
        nautilus calls it for every fill of an order on the strategy's
        positions, so a test or a venue never calls it directly::

            engine.run()  # each fill reaches strategy.on_order_filled(event)
        """
        # Quantity.as_double() is inexact for many share counts (59353 is
        # 59352.99999999999), so the count is read from its decimal.
        kind = corporate_action_kind(self.cache.order(event.client_order_id).tags)
        if kind is not None:
            self.recorder.corporate_action(
                kind,
                ts_ns=event.ts_event,
                permno=self.venue.resolver.permno(event.instrument_id),
                side=event.order_side.name,
                quantity=int(event.last_qty.as_decimal()),
                price=event.last_px.as_double(),
                fee=event.commission.as_double(),
            )
            return
        self.recorder.order_filled(
            event.client_order_id.value,
            ts_ns=event.ts_event,
            price=event.last_px.as_double(),
            quantity=int(event.last_qty.as_decimal()),
            fee=event.commission.as_double(),
        )

    def on_order_rejected(self, event) -> None:
        """Record an order the venue rejected.

        Parameters
        ----------
        event : nautilus_trader.model.events.OrderRejected
            The rejection, whose ``reason`` is recorded.

        Examples
        --------
        nautilus calls it when the venue rejects one of the strategy's
        orders::

            engine.run()  # a rejection reaches strategy.on_order_rejected(event)
        """
        self.recorder.order_refused(event.client_order_id.value, "rejected", str(event.reason))

    def on_order_denied(self, event) -> None:
        """Record an order nautilus's risk engine denied.

        Parameters
        ----------
        event : nautilus_trader.model.events.OrderDenied
            The denial, whose ``reason`` is recorded.

        Examples
        --------
        nautilus calls it when its risk engine denies one of the strategy's
        orders before it reaches the venue::

            engine.run()  # a denial reaches strategy.on_order_denied(event)
        """
        self.recorder.order_refused(event.client_order_id.value, "denied", str(event.reason))
