"""The backtest venue's corporate actions, booked at 09:30 ET of the ex-date (ADR 0009).

``CorporateActionModule`` is a nautilus ``SimulationModule`` holding a
schedule built from the run's price dataset (``corporate_action_days``). It
acts at 09:30 ET of the day an action takes effect, after that timestamp's
opening prints and before the open + 1 ns next-open orders (ADR 0003), on the
position held at the prior close:

- **dividend** (``divCash != 0``): ``divCash * signed_qty`` in cash through
  ``exchange.adjust_account`` (a short pays);
- **holder split** (finite ``k > 0``, ``k != 1``, ``k`` equal to the share
  factor ``cumfacshr[t-1] / cumfacshr[t]``, ``k = splitFactor``): a venue fill
  of ``floor(|q| * k) - |q|`` shares (toward zero for a short) at price 0, so
  no cash moves, and the fraction as cash in lieu at the pre-split close / k;
- **value distribution** (``k > 1``, share factor 1: a spin-off and the like):
  ``q * (k - 1) * close[t]`` in cash, on top of that day's ``divCash``; no
  position is opened in the distributed security;
- **final event** (``k = 0``, a merger or liquidation): nothing; the holding
  is settled by the delisting path. It is logged, as is any **other** factor
  day, which leaves the position alone;
- **price-implied share change** (#27): ``adjClose`` is chained from CRSP's
  total return, so on a day with a raw close the factor the prices imply,
  ``x = (adjClose[t] / adjClose[p] * close[p] - divCash[t]) / close[t]``
  (p the last bar with a raw close and an ``adjClose``, net of what was
  booked on rows between them), is the holder's share change that conserves value. A factor day
  whose ``splitFactor`` agrees with x (``PRICE_FACTOR_RTOL``) is a holder
  split by ``splitFactor`` even when the share factor disagrees; a day
  without a usable factor whose x lies outside
  ``[1 / (1 + IMPLIED_SPLIT_TOL), 1 + IMPLIED_SPLIT_TOL]`` is an
  **implied split** booked as a split by x (``IMPLIED_SPLIT``); a smaller
  disagreement above ``PRICE_FACTOR_RTOL`` is logged as ``MISMATCH``;
- **delisting** (quantlab ADR 0014): a venue fill closing the delisted
  holding at its last valuation on the bar after its delisting bar. CRSP
  books a cash merger's payment as a distribution (``dlynonorddivamt``) on
  the delisting row, a row without a raw price, so ``divCash`` there is the
  delisting proceeds, which the settlement already pays: such a
  **delisting payment** (``divCash`` on a row without a raw close, on the
  delisting bar or the settlement bar of a settled delisting) moves no cash
  and is logged as ``DELISTING_PAYMENT``.

A dividend on a split's ex-date is paid on the shares held at the prior
close, before the split. Each fill is a **venue fill**: a ``MarketOrder``
carrying the position's trader and strategy ids and a
``CORPORATE_ACTION_<KIND>`` tag, added to the cache, marked submitted and
accepted through the venue's execution client and filled with
``OrderMatchingEngine.apply_fills``, the path nautilus's own expiration
settlement uses. The strategy hears an ordinary ``OrderFilled`` for an order
it never submitted and records it as a venue event; every trader fee model
charges nothing for it. Cash movements and logged days reach the strategy
through ``on_corporate_action``.

nautilus runs a module only on timestamps that carry data, so an action
falls due at the first data timestamp at or after 09:30: on a bar where no
instrument has an opening print it is booked at that bar's close instead.
Nothing is lost: a security without an opening print has no next-open order
filled before its action.
"""

from __future__ import annotations

import dataclasses
import math
import uuid
from collections import deque
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from nautilus_trader.backtest.config import SimulationModuleConfig
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import LiquiditySide, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import ClientOrderId, VenueOrderId
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.orders import MarketOrder

from quantlab_trader.base.venue import CORPORATE_ACTION_TAG
from quantlab_trader.venue.backtest.feed import OPEN_TIME, session_ns
from quantlab_trader.venue.backtest.resolver import PRICE_PRECISION, BacktestResolver
from quantlab_trader.venue.backtest.source import DelistingSettlement

#: Relative tolerance of "equal" factors (the price and share factors of a
#: holder split, a factor and 1), as the measurements on #17 used.
FACTOR_RTOL = 1e-4
#: Relative tolerance of a factor agreeing with the factor the prices imply
#: (#27): ``splitFactor`` against CRSP's return, and the threshold above which
#: a day without a usable factor is logged as a ``MISMATCH``. Measured on the
#: market store 2020-2024: on every ordinary day the two agree within 1e-4.
PRICE_FACTOR_RTOL = 0.01
#: A price-implied share change is booked when it exceeds this, either way:
#: ``x > 1 + IMPLIED_SPLIT_TOL`` or ``x < 1 / (1 + IMPLIED_SPLIT_TOL)`` (#27).
IMPLIED_SPLIT_TOL = 0.2
#: Slack of the share floors against binary rounding (300 * (1/3) is 99.99...).
_SHARE_EPS = 1e-9

#: What a corporate-action day is (ADR 0009 and its amendments); ``None`` for
#: a dividend alone.
FactorKind = Literal["SPLIT", "DISTRIBUTION", "FINAL", "OTHER", "IMPLIED_SPLIT", "MISMATCH"]
#: The kinds booked as a holder split (a price-0 venue fill, cash in lieu, a
#: queued order rescaled).
HOLDER_SPLIT_KINDS = ("SPLIT", "IMPLIED_SPLIT")


def classify_factor_day(split_factor: float, share_factor: float) -> FactorKind | None:
    """Return what a day with these price and share factors is.

    Parameters
    ----------
    split_factor : float
        ``splitFactor`` of the day, ``cumfacpr[t-1] / cumfacpr[t]``.
    share_factor : float
        ``cumfacshr[t-1] / cumfacshr[t]``.

    Examples
    --------
    >>> classify_factor_day(2.0, 2.0), classify_factor_day(1 / 3, 1 / 3)
    ('SPLIT', 'SPLIT')
    >>> classify_factor_day(1.25, 1.0), classify_factor_day(0.0, float("inf"))
    ('DISTRIBUTION', 'FINAL')
    >>> classify_factor_day(0.9, 1.0), classify_factor_day(1.0, 1.0) is None
    ('OTHER', True)
    >>> classify_factor_day(float("nan"), float("nan")) is None  # a halt
    True
    """
    k, s = float(split_factor), float(share_factor)
    if k == 0.0:
        return "FINAL"
    k_moves = not np.isnan(k) and not math.isclose(k, 1.0, rel_tol=FACTOR_RTOL)
    s_moves = not np.isnan(s) and not math.isclose(s, 1.0, rel_tol=FACTOR_RTOL)
    if not (k_moves or s_moves):
        return None
    if math.isfinite(k) and k > 0 and k_moves and math.isfinite(s) and math.isclose(
        k, s, rel_tol=FACTOR_RTOL
    ):
        return "SPLIT"
    if math.isfinite(k) and k > 1 and k_moves and math.isfinite(s) and not s_moves:
        return "DISTRIBUTION"
    return "OTHER"


def reconcile_with_prices(
    kind: FactorKind | None, split_factor: float, implied_factor: float
) -> FactorKind | None:
    """Return what a day is once its factors are checked against its prices (#27).

    Only a day the factors leave alone (``None`` or ``"OTHER"``) changes: a
    factor day whose ``splitFactor`` agrees with the price-implied factor
    becomes a ``"SPLIT"`` by ``splitFactor``; otherwise an implied factor
    beyond ``IMPLIED_SPLIT_TOL`` makes it an ``"IMPLIED_SPLIT"``, and one
    beyond ``PRICE_FACTOR_RTOL`` makes a day without a factor a
    ``"MISMATCH"`` (logged only).

    Parameters
    ----------
    kind : str or None
        What the factors make the day (``classify_factor_day``).
    split_factor : float
        ``splitFactor`` of the day.
    implied_factor : float
        ``x``, the share change that conserves the holder's value given
        ``adjClose``'s return; NaN where the prices cannot say.

    Examples
    --------
    >>> reconcile_with_prices("OTHER", 0.035404, 0.035395)  # PERMNO 18217
    'SPLIT'
    >>> reconcile_with_prices(None, 1.0, 0.0763 / 5.1357)  # PERMNO 14051
    'IMPLIED_SPLIT'
    >>> reconcile_with_prices(None, 1.0, 10 / 11), reconcile_with_prices(None, 1.0, 1.00005)
    ('MISMATCH', None)
    >>> reconcile_with_prices("OTHER", 0.9, 1.0), reconcile_with_prices("SPLIT", 2.0, 1.0)
    ('OTHER', 'SPLIT')
    """
    x, k = float(implied_factor), float(split_factor)
    if kind not in (None, "OTHER") or not (math.isfinite(x) and x > 0):
        return kind
    if (
        kind == "OTHER"
        and math.isfinite(k)
        and k > 0
        and not math.isclose(k, 1.0, rel_tol=FACTOR_RTOL)
        and math.isclose(x, k, rel_tol=PRICE_FACTOR_RTOL)
    ):
        return "SPLIT"
    if max(x, 1.0 / x) > 1.0 + IMPLIED_SPLIT_TOL:
        return "IMPLIED_SPLIT"
    if kind is None and not math.isclose(x, 1.0, rel_tol=PRICE_FACTOR_RTOL):
        return "MISMATCH"
    return kind


def split_shares(quantity: int, k: float) -> int:
    """Return the whole shares ``quantity`` becomes in a holder split by ``k``, toward zero.

    Examples
    --------
    >>> split_shares(500, 1 / 3), split_shares(-625, 1.5), split_shares(300, 1 / 3)
    (166, -937, 100)
    """
    shares = math.floor(abs(quantity) * k + _SHARE_EPS)
    return shares if quantity >= 0 else -shares


@dataclass(frozen=True)
class CorporateActionDay:
    """One security's ex-date: a factor day, a dividend, or both.

    Attributes
    ----------
    permno : Hashable
        The security.
    date : pandas.Timestamp
        The ex-date t.
    kind : {"SPLIT", "DISTRIBUTION", "FINAL", "OTHER", "IMPLIED_SPLIT", "MISMATCH"} or None
        What the factors make the day (``classify_factor_day``), checked
        against its prices (``reconcile_with_prices``); ``None`` for a
        dividend alone.
    split_factor : float
        ``splitFactor[t]``, the holder split factor ``k`` of a split.
    share_factor : float
        ``cumfacshr[t-1] / cumfacshr[t]``, with the last ``cumfacshr`` before
        t when t-1 has none.
    dividend : float
        ``divCash[t]`` per share; 0 for none.
    pre_close : float
        The last raw close before t.
    close : float
        The raw close of t; NaN without one.
    implied_factor : float
        The share change the prices imply (``reconcile_with_prices``); NaN
        where they cannot say (no raw close of t or before it).
    """

    permno: Hashable
    date: pd.Timestamp
    kind: FactorKind | None
    split_factor: float
    share_factor: float
    dividend: float
    pre_close: float
    close: float
    implied_factor: float = math.nan

    @property
    def holder_factor(self) -> float:
        """``k`` of a holder split: ``splitFactor``, or the implied factor of an implied split."""
        return self.implied_factor if self.kind == "IMPLIED_SPLIT" else self.split_factor

    @property
    def distribution_price(self) -> float:
        """The price a value distribution's per-share cash is ``k - 1`` times.

        The raw close of t; without one, the pre-split close / k, which is
        what the close of t is when the price falls by exactly the
        distribution.
        """
        if np.isfinite(self.close):
            return float(self.close)
        return float(self.pre_close) / float(self.split_factor)


def corporate_action_days(prices: xr.Dataset) -> tuple[CorporateActionDay, ...]:
    """Return the panel's corporate-action days after its first bar, in time order.

    The first bar has none: no position is open before the first next-open
    fill, at the open of the second bar.

    A day is a dividend, a factor day, or a day whose prices disagree with
    ``adjClose``'s return (``reconcile_with_prices``). The implied factor of
    a bar t with a raw close is
    ``(adjClose[t] / adjClose[p] * close[p] - C - Q * divCash[t]) / (Q * close[t])``,
    with p the last bar with a raw close and an ``adjClose``, and ``Q``, ``C``
    the shares and cash per share held at p that the actions booked on the
    rows between p and t (a halt's rows without a raw close or an
    ``adjClose``) made of it.

    Parameters
    ----------
    prices : xarray.Dataset
        The run's price panel on ``(timestamp, symbol)`` with raw ``close``,
        ``adjClose``, ``splitFactor``, ``cumfacshr`` and ``divCash``.

    Returns
    -------
    tuple of CorporateActionDay

    Examples
    --------
    >>> ones = [[1.0], [1.0], [1.0]]
    >>> panel = xr.Dataset(
    ...     {
    ...         "close": (("timestamp", "symbol"), [[40.0], [42.0], [127.0]]),
    ...         "splitFactor": (("timestamp", "symbol"), [[1.0], [1.0], [1 / 3]]),
    ...         "cumfacshr": (("timestamp", "symbol"), [[1.0], [1.0], [3.0]]),
    ...         "divCash": (("timestamp", "symbol"), [[0.0], [0.5], [0.0]]),
    ...         "adjClose": (("timestamp", "symbol"), [[13.0], [13.8], [127 / 3 * 0.99]]),
    ...     },
    ...     coords={"timestamp": pd.bdate_range("2024-01-02", periods=3), "symbol": [10001]},
    ... )
    >>> [(d.date.day, d.kind, d.dividend, d.pre_close) for d in corporate_action_days(panel)]
    [(3, None, 0.5, 40.0), (4, 'SPLIT', 0.0, 42.0)]
    """

    def frame(name: str) -> pd.DataFrame:
        return prices[name].transpose("timestamp", "symbol").to_pandas().astype(float)

    split_factor, dividend, close = frame("splitFactor"), frame("divCash"), frame("close")
    adj_close, cumfacshr = frame("adjClose"), frame("cumfacshr")
    with np.errstate(divide="ignore", invalid="ignore"):
        share_factor = cumfacshr.ffill().shift(1) / cumfacshr
    pre_close = close.ffill().shift(1)
    dividend = dividend.where(np.isfinite(dividend), 0.0)
    # The anchor of bar t: the last bar before it with a raw close and an adjClose.
    priced = (close.notna() & adj_close.notna()).to_numpy()
    anchor_close = close.where(priced).ffill().shift(1).to_numpy()
    anchor_return = (adj_close / adj_close.where(priced).ffill().shift(1)).to_numpy()
    closes, cash = close.to_numpy(), dividend.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        naive = (anchor_return * anchor_close - cash) / closes
    # Candidates only: classify_factor_day is per element, the panel is not.
    k_values, s_values = split_factor.to_numpy(), share_factor.to_numpy()
    with np.errstate(invalid="ignore"):
        actions = (
            (cash != 0.0)
            | (~np.isnan(k_values) & ~np.isclose(k_values, 1.0, rtol=FACTOR_RTOL, atol=0.0))
            | (~np.isnan(s_values) & ~np.isclose(s_values, 1.0, rtol=FACTOR_RTOL, atol=0.0))
        )
        disagree = priced & np.isfinite(naive) & ~np.isclose(
            naive, 1.0, rtol=PRICE_FACTOR_RTOL, atol=0.0
        )
    actions[0, :] = False
    disagree[0, :] = False
    # The first priced bar after an action on a row without a price: its
    # implied factor must net out what that action booked.
    after_unpriced = np.zeros_like(actions)
    for bar, column in zip(*np.nonzero(actions & ~priced)):
        later = np.flatnonzero(priced[bar + 1 :, column])
        if later.size:
            after_unpriced[bar + 1 + later[0], column] = True
    visit = actions | disagree | after_unpriced
    days = []
    for column in range(visit.shape[1]):
        held_shares, held_cash = 1.0, 0.0  # per share held at the anchor
        for bar in np.flatnonzero(visit[:, column]):
            permno = split_factor.columns[column]
            k, s = split_factor.iat[bar, column], share_factor.iat[bar, column]
            day = CorporateActionDay(
                permno=permno,
                date=pd.Timestamp(split_factor.index[bar]),
                kind=classify_factor_day(k, s),
                split_factor=float(k),
                share_factor=float(s),
                dividend=float(cash[bar, column]),
                pre_close=float(pre_close.iat[bar, column]),
                close=float(closes[bar, column]),
            )
            if not priced[bar, column]:
                held_cash += held_shares * day.dividend
                if day.kind == "DISTRIBUTION":
                    held_cash += held_shares * (day.split_factor - 1.0) * day.distribution_price
                elif day.kind == "SPLIT":
                    held_shares *= day.split_factor
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    implied = (
                        anchor_return[bar, column] * anchor_close[bar, column]
                        - held_cash
                        - held_shares * day.dividend
                    ) / (held_shares * day.close)
                held_shares, held_cash = 1.0, 0.0
                day = dataclasses.replace(
                    day,
                    implied_factor=float(implied),
                    kind=reconcile_with_prices(day.kind, day.split_factor, implied),
                )
            if day.kind is not None or day.dividend != 0.0:
                days.append(day)
    return tuple(sorted(days, key=lambda day: day.date))


def holder_split_factors(
    days: Sequence[CorporateActionDay],
) -> dict[tuple[pd.Timestamp, Hashable], float]:
    """Return ``k`` of every holder split, implied ones too, by ``(ex-date, permno)``.

    It is what rescales a queued order.

    Parameters
    ----------
    days : Sequence of CorporateActionDay
        The window's ex-dates (``corporate_action_days``).

    Returns
    -------
    dict

    Examples
    --------
    >>> day = CorporateActionDay(
    ...     10001, pd.Timestamp("2024-01-04"), "SPLIT", 2.0, 2.0, 0.0, 52.0, 27.0
    ... )
    >>> holder_split_factors([day])
    {(Timestamp('2024-01-04 00:00:00'), 10001): 2.0}
    """
    return {
        (day.date, day.permno): day.holder_factor
        for day in days
        if day.kind in HOLDER_SPLIT_KINDS
    }


#: Called with ``(action, ts_ns=, permno=, quantity=, amount=, **detail)`` for
#: an action booked without a fill (cash) or logged.
CorporateActionSink = Callable[..., None]


class CorporateActionModule(SimulationModule):
    """Book the window's corporate actions on the strategy's positions.

    Parameters
    ----------
    days : Sequence of CorporateActionDay
        The ex-dates, booked at their 09:30 ET.
    delistings : Sequence of DelistingSettlement
        The delisted securities, settled at the open of their settlement date.
    resolver : BacktestResolver
        Maps each security to its instrument.
    on_event : callable, optional
        Hears every action booked without a fill and every logged day (the
        strategy's ``on_corporate_action``).

    Examples
    --------
    Built by ``BacktestVenue.run`` and handed to ``add_venue(modules=...)``;
    an empty schedule books nothing:

    >>> from quantlab_trader.venue.backtest.resolver import BacktestResolver
    >>> module = CorporateActionModule((), (), BacktestResolver((), pd.Timestamp("2024-01-02")))
    >>> module.process(0)
    """

    def __init__(
        self,
        days: Sequence[CorporateActionDay],
        delistings: Sequence[DelistingSettlement],
        resolver: BacktestResolver,
        on_event: CorporateActionSink | None = None,
    ):
        super().__init__(SimulationModuleConfig())
        self._resolver = resolver
        self._on_event = on_event or (lambda *args, **kwargs: None)
        due = [(session_ns(day.date, OPEN_TIME), day.date, day) for day in days]
        due += [
            (session_ns(s.settlement_date, OPEN_TIME), s.delisting_date, s)
            for s in delistings
        ]
        self._due = deque(sorted(due, key=lambda item: item[0]))
        self._delisting_rows = {
            (date, s.permno)
            for s in delistings
            for date in (s.delisting_date, s.settlement_date)
        }

    def process(self, ts_now: int) -> None:
        """Book every action due at or before ``ts_now`` (09:30 ET of its day)."""
        while self._due and self._due[0][0] <= ts_now:
            _, as_of, action = self._due.popleft()
            instrument_id = self._resolver.instrument_id(action.permno, as_of)
            for position in self.cache.positions_open(None, instrument_id):
                if isinstance(action, DelistingSettlement):
                    self._settle(position, action.price)
                else:
                    self._book(position, action, ts_now)

    def _book(self, position, day: CorporateActionDay, ts_now: int) -> None:
        """Book ``day`` on ``position``: its dividend, then its factor day."""
        quantity = int(round(position.signed_qty))
        if not quantity:
            return
        report = dict(ts_ns=ts_now, permno=day.permno, quantity=quantity)
        if day.dividend and self.is_delisting_payment(day):
            self._on_event(
                "DELISTING_PAYMENT", amount=0.0, per_share=day.dividend, **report
            )
        elif day.dividend:
            amount = self._credit(quantity * day.dividend)
            self._on_event("DIVIDEND", amount=amount, per_share=day.dividend, **report)
        if day.kind in HOLDER_SPLIT_KINDS:
            self._split(position, quantity, day, report)
        elif day.kind == "DISTRIBUTION":
            amount = self._credit(
                quantity * (day.split_factor - 1.0) * day.distribution_price
            )
            self._on_event(
                "DISTRIBUTION", amount=amount, split_factor=day.split_factor, **report
            )
        elif day.kind is not None:
            self._on_event(
                day.kind,
                amount=0.0,
                split_factor=_json_float(day.split_factor),
                share_factor=_json_float(day.share_factor),
                implied_factor=_json_float(day.implied_factor),
                **report,
            )

    def is_delisting_payment(self, day: CorporateActionDay) -> bool:
        """Whether ``day``'s ``divCash`` is a settled delisting's proceeds, not a dividend.

        It is when the row has no raw close and is the delisting bar or the
        settlement bar of one of the module's delistings.
        """
        return (
            not np.isfinite(day.close)
            and (day.date, day.permno) in self._delisting_rows
        )

    def _split(self, position, quantity: int, day: CorporateActionDay, report: dict) -> None:
        """A price-0 venue fill of the share-count change plus cash in lieu of the fraction."""
        k = day.holder_factor
        shares = split_shares(quantity, k)
        delta = shares - quantity
        if delta:
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            self._venue_fill(position, side, abs(delta), 0.0, day.kind)
        fraction = quantity * k - shares
        if fraction and np.isfinite(day.pre_close):
            amount = self._credit(fraction * day.pre_close / k)
            if amount:
                self._on_event(
                    "CASH_IN_LIEU", amount=amount, split_factor=k, **report
                )

    def _credit(self, amount: float) -> float:
        """Move ``amount`` (cents) into the account; return what was moved."""
        amount = round(float(amount), 2)
        if amount:
            self.exchange.adjust_account(Money(amount, USD))
        return amount

    def _settle(self, position, price: float) -> None:
        """Close ``position`` at ``price``: the delisting settlement."""
        quantity = abs(int(round(position.signed_qty)))
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
            position,
        )

    def log_diagnostics(self, logger) -> None:
        """Nothing to log."""

    def reset(self) -> None:
        """Nothing to reset: the schedule is built once per engine."""


def _json_float(value: float) -> float | None:
    """Return ``value``, or ``None`` where JSON has no number for it."""
    return float(value) if math.isfinite(value) else None
