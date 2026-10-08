"""IBKR's own account of the live orders: executions, open orders and completed orders.

The live record step (#51) reconciles the orders a day's decide step sent
with what IBKR reports after the open, and the decide step checks that no
order of the bar it decides is already working at IBKR. Both read three
reports, through a short TWS API connection on a client id of its own
(``IbapiOrderReports``), or through any ``OrderReports`` (the tests' fake):

- ``executions``: the account's fills of the day (``reqExecutions``), each
  with its commission;
- ``open_orders``: the orders still working (``reqAllOpenOrders``);
- ``completed_orders``: the orders IBKR ended today, filled, canceled or
  expired, with its reason (``reqCompletedOrders``).

An order is told apart by its reference, the nautilus client order id the
IB adapter sends as IBKR's ``orderRef``. The live strategy's order id tag is
the decision date (``decision_tag``), so the reference names the bar that
decided the order (``decision_date_of``). IBKR reports executions of the
current day only (unless the Gateway is set to keep more), so the record step
runs on the day of the open.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from ibapi.client import EClient
from ibapi.execution import ExecutionFilter
from ibapi.wrapper import EWrapper


def decision_tag(t: pd.Timestamp) -> str:
    """Return the strategy's order id tag of decision date ``t``: ``YYYYMMDD``.

    nautilus puts it in every client order id the strategy makes
    (``O-<date>-<time>-<trader tag>-<tag>-<count>``), and the IB adapter
    sends that id to IBKR as the order's reference.

    Examples
    --------
    >>> decision_tag(pd.Timestamp("2026-10-07"))
    '20261007'
    """
    return pd.Timestamp(t).strftime("%Y%m%d")


def decision_date_of(order_ref: str) -> pd.Timestamp | None:
    """Return the decision date an order reference names, or ``None`` for another order.

    Examples
    --------
    >>> decision_date_of("O-20261008-090501-IBKR-20261007-3")
    Timestamp('2026-10-07 00:00:00')
    >>> decision_date_of("manual order") is None
    True
    """
    parts = (order_ref or "").split(":", 1)[0].split("-")
    if len(parts) != 6 or parts[0] != "O":
        return None
    try:
        return pd.Timestamp(pd.to_datetime(parts[4], format="%Y%m%d"))
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class IbkrExecution:
    """One fill IBKR reports.

    Attributes
    ----------
    exec_id : str
        IBKR's execution id, unique per fill.
    order_ref : str
        The order's reference (the nautilus client order id).
    con_id : int
        The contract.
    side : {"BUY", "SELL"}
    shares : int
    price : float
    commission : float
        The commission IBKR charged, 0 until it has reported it.
    time : pandas.Timestamp
        When it filled, UTC.

    Examples
    --------
    >>> IbkrExecution("0001", "O-1", 1001, "BUY", 25, 30.12, 1.0,
    ...               pd.Timestamp("2026-10-08 13:30", tz="UTC")).shares
    25
    """

    exec_id: str
    order_ref: str
    con_id: int
    side: str
    shares: int
    price: float
    commission: float
    time: pd.Timestamp


@dataclass(frozen=True)
class IbkrOrderState:
    """An order IBKR reports open or completed.

    Attributes
    ----------
    order_ref : str
        The order's reference (the nautilus client order id).
    con_id : int
    side : {"BUY", "SELL"}
    quantity : int
    status : str
        IBKR's status (``PreSubmitted``, ``Submitted``, ``Filled``,
        ``Cancelled``, ``Inactive``, ...).
    reason : str
        IBKR's completed status text ("" for an open order).

    Examples
    --------
    >>> IbkrOrderState("O-1", 1001, "BUY", 25, "Cancelled", "Order canceled").status
    'Cancelled'
    """

    order_ref: str
    con_id: int
    side: str
    quantity: int
    status: str
    reason: str = ""


class OrderReports(Protocol):
    """Where the live steps read IBKR's reports of the account's orders.

    Examples
    --------
    ``IbapiOrderReports`` reads them from the Gateway; a test passes a fake::

        with IbapiOrderReports("127.0.0.1", 4002, client_id=3) as reports:
            fills = reports.executions("DU1234567")
    """

    def executions(self, account: str) -> Sequence[IbkrExecution]:
        """The account's executions of the day."""

    def open_orders(self, account: str) -> Sequence[IbkrOrderState]:
        """The account's orders still working."""

    def completed_orders(self, account: str) -> Sequence[IbkrOrderState]:
        """The account's orders IBKR ended today."""


_SIDES = {"BOT": "BUY", "SLD": "SELL", "BUY": "BUY", "SELL": "SELL"}


class _App(EWrapper, EClient):
    """The ibapi wrapper and client in one, collecting the three reports."""

    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.ready = threading.Event()
        self.executions: dict[int, list] = {}
        self.commissions: dict[str, float] = {}
        self.open_orders: list = []
        self.completed: list = []
        self.done: dict[str, threading.Event] = {}

    def event(self, name: str) -> threading.Event:
        return self.done.setdefault(name, threading.Event())

    def nextValidId(self, orderId: int):  # noqa: N802, N803 (ibapi's names)
        self.ready.set()

    def execDetails(self, reqId: int, contract, execution):  # noqa: N802, N803
        self.executions.setdefault(reqId, []).append((contract, execution))

    def execDetailsEnd(self, reqId: int):  # noqa: N802, N803
        self.event(f"executions-{reqId}").set()

    def commissionAndFeesReport(self, commissionAndFeesReport):  # noqa: N802, N803
        report = commissionAndFeesReport
        self.commissions[report.execId] = float(report.commissionAndFees)

    def openOrder(self, orderId, contract, order, orderState):  # noqa: N802, N803
        self.open_orders.append((contract, order, orderState))

    def openOrderEnd(self):  # noqa: N802
        self.event("open").set()

    def completedOrder(self, contract, order, orderState):  # noqa: N802, N803
        self.completed.append((contract, order, orderState))

    def completedOrdersEnd(self):  # noqa: N802
        self.event("completed").set()


class IbapiOrderReports:
    """Read IBKR's executions, open and completed orders over a TWS API connection of its own.

    Parameters
    ----------
    host : str
        The Gateway's host.
    port : int
        Its API port (4002: IB Gateway, paper).
    client_id : int
        A client id no other connection uses.
    timeout : float, default 30
        Seconds to wait for the connection and for each report.

    Examples
    --------
    ::

        with IbapiOrderReports("127.0.0.1", 4002, client_id=3) as reports:
            working = reports.open_orders("DU1234567")
    """

    def __init__(self, host: str, port: int, client_id: int, *, timeout: float = 30.0):
        self.host, self.port, self.client_id, self.timeout = host, port, client_id, timeout
        self._app: _App | None = None
        self._thread: threading.Thread | None = None
        self._ids = itertools.count(1)

    def __enter__(self) -> IbapiOrderReports:
        app = _App()
        app.connect(self.host, self.port, self.client_id)
        self._thread = threading.Thread(target=app.run, name="ibapi-reports", daemon=True)
        self._thread.start()
        if not app.ready.wait(self.timeout):
            app.disconnect()
            raise ConnectionError(
                f"no TWS API connection to {self.host}:{self.port} (client id "
                f"{self.client_id}) within {self.timeout}s"
            )
        self._app = app
        return self

    def __exit__(self, *exc) -> None:
        if self._app is not None:
            self._app.disconnect()
            self._app = None
        if self._thread is not None:
            self._thread.join(timeout=self.timeout)
            self._thread = None

    def _require(self) -> _App:
        if self._app is None:
            raise RuntimeError("IbapiOrderReports is used inside its `with` block only")
        return self._app

    def _wait(self, event: threading.Event, what: str) -> None:
        if not event.wait(self.timeout):
            raise TimeoutError(f"no {what} from IBKR within {self.timeout}s")

    def executions(self, account: str) -> list[IbkrExecution]:
        """Return the account's executions of the day, each with its commission.

        Examples
        --------
        ::

            reports.executions("DU1234567")  # [IbkrExecution(...), ...]
        """
        from nautilus_trader.adapters.interactive_brokers.parsing.execution import (
            timestring_to_timestamp,
        )

        app = self._require()
        request = next(self._ids)
        done = app.event(f"executions-{request}")
        selection = ExecutionFilter()
        selection.acctCode = account
        app.reqExecutions(request, selection)
        self._wait(done, "executions")
        found = app.executions.pop(request, [])
        # Commission reports follow the executions; give them a moment.
        deadline = threading.Event()
        for _ in range(int(self.timeout * 10)):
            if all(e.execId in app.commissions for _, e in found):
                break
            deadline.wait(0.1)
        return [
            IbkrExecution(
                exec_id=execution.execId,
                order_ref=(execution.orderRef or "").rsplit(":", 1)[0],
                con_id=int(contract.conId),
                side=_SIDES.get(execution.side, execution.side),
                shares=int(execution.shares),
                price=float(execution.price),
                commission=app.commissions.get(execution.execId, 0.0),
                time=pd.Timestamp(timestring_to_timestamp(execution.time)),
            )
            for contract, execution in found
            if execution.acctNumber == account
        ]

    def open_orders(self, account: str) -> list[IbkrOrderState]:
        """Return the account's orders still working, placed by any client.

        Examples
        --------
        ::

            reports.open_orders("DU1234567")
        """
        app = self._require()
        app.open_orders.clear()
        done = app.event("open")
        done.clear()
        app.reqAllOpenOrders()
        self._wait(done, "open orders")
        return [_state(c, o, s, "") for c, o, s in app.open_orders if o.account == account]

    def completed_orders(self, account: str) -> list[IbkrOrderState]:
        """Return the account's orders IBKR ended today, with IBKR's reason.

        Examples
        --------
        ::

            reports.completed_orders("DU1234567")
        """
        app = self._require()
        app.completed.clear()
        done = app.event("completed")
        done.clear()
        app.reqCompletedOrders(False)
        self._wait(done, "completed orders")
        return [
            _state(c, o, s, s.completedStatus or "") for c, o, s in app.completed
            if o.account == account
        ]


def _state(contract, order, order_state, reason: str) -> IbkrOrderState:
    return IbkrOrderState(
        order_ref=(order.orderRef or "").rsplit(":", 1)[0],
        con_id=int(contract.conId),
        side=_SIDES.get(order.action, order.action),
        quantity=int(order.totalQuantity),
        status=order_state.status,
        reason=reason,
    )
