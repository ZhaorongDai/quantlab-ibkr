"""The backtest venue's corporate actions, booked as venue fills (ADR 0009).

``CorporateActionModule`` is a nautilus ``SimulationModule`` holding a
schedule built from the run's price dataset. It acts at 09:30 ET of the day an
action takes effect, after that timestamp's opening prints and before the
open + 1 ns next-open orders (ADR 0003). Each action is a **venue fill**: a
``MarketOrder`` carrying the position's trader and strategy ids and a
``CORPORATE_ACTION_<KIND>`` tag, added to the cache, marked submitted and
accepted through the venue's execution client and filled with
``OrderMatchingEngine.apply_fills``, the path nautilus's own expiration
settlement uses. The strategy hears an ordinary ``OrderFilled`` for an order
it never submitted and records it as a venue event; every trader fee model
charges nothing for it.

Booked so far: the delisting settlement (quantlab ADR 0014), closing a
delisted holding at its last valuation on the bar after its delisting bar.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from nautilus_trader.backtest.config import SimulationModuleConfig
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.enums import LiquiditySide, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, VenueOrderId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.model.orders import MarketOrder

from quantlab_trader.base.venue import CORPORATE_ACTION_TAG
from quantlab_trader.venue.backtest.feed import OPEN_TIME, session_ns
from quantlab_trader.venue.backtest.resolver import PRICE_PRECISION, BacktestResolver
from quantlab_trader.venue.backtest.source import DelistingSettlement


class CorporateActionModule(SimulationModule):
    """Book the window's corporate actions on the strategy's positions as venue fills.

    Parameters
    ----------
    delistings : Sequence of DelistingSettlement
        The delisted securities, settled at the open of their settlement date.
    resolver : BacktestResolver
        Maps each security to its instrument.
    """

    def __init__(self, delistings: Sequence[DelistingSettlement], resolver: BacktestResolver):
        super().__init__(SimulationModuleConfig())
        self._actions = sorted(
            (
                (
                    session_ns(settlement.settlement_date, OPEN_TIME),
                    resolver.instrument_id(settlement.permno, settlement.delisting_date),
                    settlement.price,
                )
                for settlement in delistings
            ),
            key=lambda action: action[0],
        )

    def process(self, ts_now: int) -> None:
        """Book every action due at or before ``ts_now``."""
        while self._actions and self._actions[0][0] <= ts_now:
            _, instrument_id, price = self._actions.pop(0)
            for position in self.cache.positions_open(None, instrument_id):
                self._settle(position, price)

    def _settle(self, position, price: float) -> None:
        """Close ``position`` at ``price``: the delisting settlement."""
        quantity = abs(int(position.signed_qty))
        if quantity:
            side = OrderSide.SELL if position.signed_qty > 0 else OrderSide.BUY
            self._venue_fill(position, side, quantity, price, "DELIST")

    def _venue_fill(self, position, side: OrderSide, quantity: int, price: float, kind: str) -> None:
        """Book an order and its fill on ``position``, as nautilus's expiration settlement does."""
        now = self.clock.timestamp_ns()
        order = MarketOrder(
            trader_id=position.trader_id,
            strategy_id=position.strategy_id,
            instrument_id=position.instrument_id,
            client_order_id=ClientOrderId(f"CA-{kind}-{uuid.uuid4().hex[:12]}"),
            order_side=side,
            quantity=Quantity(quantity, 0),
            init_id=UUID4(),
            ts_init=now,
            time_in_force=TimeInForce.DAY,
            tags=[f"{CORPORATE_ACTION_TAG}_{kind}"],
        )
        self.cache.add_order(order, position_id=position.id)
        client = self.exchange.exec_client
        client.generate_order_submitted(
            order.strategy_id, order.instrument_id, order.client_order_id, now
        )
        client.generate_order_accepted(
            order.strategy_id,
            order.instrument_id,
            order.client_order_id,
            VenueOrderId(order.client_order_id.value),
            now,
        )
        self.exchange.get_matching_engine(position.instrument_id).apply_fills(
            order,
            [(Price(price, PRICE_PRECISION), Quantity(quantity, 0))],
            LiquiditySide.TAKER,
            None,
            self.cache.position(position.id),
        )

    def log_diagnostics(self, logger) -> None:
        """Nothing to log."""

    def reset(self) -> None:
        """Nothing to reset: the schedule is built once per engine."""
