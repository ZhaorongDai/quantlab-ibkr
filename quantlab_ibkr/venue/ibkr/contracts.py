"""IBKR's contract lookup over the TWS API, the live ``ContractClient`` of the resolver.

A short-lived ibapi connection to the Gateway, on a client id of its own, made
before the trading node starts: the resolver needs the contracts to build the
node's instrument provider config. Each lookup is a ``reqContractDetails`` on
a ``STK`` / ``SMART`` / ``USD`` contract, each answer converted with
``IBContractDetails.from_contract_details``. Not tested against a Gateway in
the suite: the resolver's tests inject a table instead (``ContractClient``).
"""

from __future__ import annotations

import itertools
import threading

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
from nautilus_trader.adapters.interactive_brokers.common import IBContractDetails

#: TWS error codes that end a contract lookup with no answer: no security
#: definition found, and an ambiguous or invalid request.
_NO_CONTRACT_CODES = frozenset({200, 321})


class _App(EWrapper, EClient):
    """The ibapi wrapper and client in one, collecting contract details by request id."""

    def __init__(self):
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.ready = threading.Event()
        self.results: dict[int, list] = {}
        self.finished: dict[int, threading.Event] = {}
        self.errors: dict[int, str] = {}

    def nextValidId(self, orderId: int):  # noqa: N802, N803 (ibapi's names)
        self.ready.set()

    def contractDetails(self, reqId: int, contractDetails):  # noqa: N802, N803
        self.results.setdefault(reqId, []).append(contractDetails)

    def contractDetailsEnd(self, reqId: int):  # noqa: N802, N803
        self.finished.setdefault(reqId, threading.Event()).set()

    def error(self, reqId, errorTime, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802, N803
        if reqId in self.finished and errorCode in _NO_CONTRACT_CODES:
            self.errors[reqId] = f"{errorCode}: {errorString}"
            self.finished[reqId].set()


class IbapiContractClient:
    """Look up US stock contracts through a TWS API connection of its own.

    Parameters
    ----------
    host : str
        The Gateway's host.
    port : int
        Its API port (4002: IB Gateway, paper).
    client_id : int
        A client id no other connection uses (not the trading node's).
    timeout : float, default 30
        Seconds to wait for the connection and for each lookup.

    Examples
    --------
    ::

        with IbapiContractClient("127.0.0.1", 4002, client_id=11) as client:
            resolver = IbkrResolver(symbols, t, lookup, client)
    """

    def __init__(self, host: str, port: int, client_id: int, *, timeout: float = 30.0):
        self.host, self.port, self.client_id, self.timeout = host, port, client_id, timeout
        self._app: _App | None = None
        self._thread: threading.Thread | None = None
        self._ids = itertools.count(1)

    def __enter__(self) -> IbapiContractClient:
        app = _App()
        app.connect(self.host, self.port, self.client_id)
        self._thread = threading.Thread(target=app.run, name="ibapi-contracts", daemon=True)
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

    def details(self, symbol: str, primary_exchange: str | None) -> list[IBContractDetails]:
        """Return the contract details of US stock ``symbol`` (IBKR's spelling, ``BRK B``).

        Raises
        ------
        RuntimeError
            Outside the ``with`` block.
        TimeoutError
            If the Gateway does not answer within ``timeout``.

        Examples
        --------
        ::

            client.details("BRK B", "NYSE")  # [IBContractDetails(...)]
        """
        app = self._app
        if app is None:
            raise RuntimeError("IbapiContractClient is used inside its `with` block only")
        contract = Contract()
        contract.secType, contract.symbol = "STK", symbol
        contract.exchange, contract.currency = "SMART", "USD"
        if primary_exchange:
            contract.primaryExchange = primary_exchange
        request = next(self._ids)
        done = app.finished.setdefault(request, threading.Event())
        app.reqContractDetails(request, contract)
        if not done.wait(self.timeout):
            raise TimeoutError(f"no contract details for {symbol!r} within {self.timeout}s")
        return [
            IBContractDetails.from_contract_details(found)
            for found in app.results.pop(request, [])
        ]
