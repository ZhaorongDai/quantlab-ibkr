"""PROTOTYPE (throwaway, ticket #15): corporate actions applied by the backtest venue.

Run (from the repo root):

    ~/projects/quantlab/.venv/bin/python prototypes/corporate_actions/run.py
    ... --split B             # close-and-reopen split instead of the zero-price delta fill
    ... --factor 0.25         # splitFactor of 10001 (2 = 2:1, 3 = 3:1, 0.25 = 1:4 reverse)
    ... --delist native       # settle the delisting with InstrumentClose + settlement_prices
    ... --no-scale-pending    # do not rescale the next-open order queued across the split

Four synthetic CRSP equities on venue CRSP (USD, price precision 4, lot 1, NETTING, MARGIN):

    10001.CRSP  long 100; split (splitFactor k) ex day 3; an exit order decided at the close of
                day 2 (sell 100, pre-split shares) is queued for the open of day 3
    10002.CRSP  long 100, sell 40 at day 2's open; cash dividend 0.50/share ex day 3
    10003.CRSP  long 100; delisting row on day 4, settled at the open of day 5 at day 4's close
    10004.CRSP  short 100; cash dividend 0.25/share ex day 3 (the short pays); 3:2 split ex day 4

The Strategy only replays a fixed order schedule at each open + 1 ns and records state after each
close; it never looks at corporate actions. The venue's ``CorporateActionModule`` (a nautilus
``SimulationModule``) applies them at the ex-date's 09:30 tick, before the open + 1 ns orders.

After every close it checks, against an independent reference ledger:

    nautilus net quantity                         == reference quantity
    cash := balance - sum(signed_qty * avg_px_open) == reference cash
    equity := cash + sum(signed_qty * raw close)  == reference equity
    (and reports whether the MARGIN balance itself equals the cash figure: it does not)
"""

from __future__ import annotations

import argparse
import math
import uuid
from decimal import Decimal

import pandas as pd

from nautilus_trader.backtest.config import SimulationModuleConfig
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FeeModel
from nautilus_trader.backtest.modules import SimulationModule
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import InstrumentClose, TradeTick
from nautilus_trader.model.enums import (
    AccountType,
    AggressorSide,
    InstrumentCloseType,
    LiquiditySide,
    OmsType,
    OrderSide,
    TimeInForce,
)
from nautilus_trader.model.identifiers import ClientOrderId, InstrumentId, Symbol, TradeId, Venue, VenueOrderId
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

VENUE = Venue("CRSP")
PREC = 4
DAYS = pd.bdate_range("2026-01-05", periods=5)  # Mon..Fri; index 0..4 = "day 1..5"
INIT_CASH = 100_000.0
FEE_PER_SHARE, FEE_MIN = 0.005, 1.0  # IBKR-Fixed-like; would charge a share-count change if not zeroed
CA_TAG = "CORPORATE_ACTION"


def ts_ns(day: pd.Timestamp, hhmm: str) -> int:
    return pd.Timestamp(f"{day.date()} {hhmm}", tz="America/New_York").tz_convert("UTC").value


def iid(permno: int) -> InstrumentId:
    return InstrumentId(Symbol(str(permno)), VENUE)


def fee(qty: int) -> float:
    return max(FEE_MIN, FEE_PER_SHARE * qty)


# --------------------------------------------------------------- synthetic CRSP data ----------
# Adjusted-basis (open, close) paths; raw = adjusted / cumulative split factor from the ex-date on.
ADJ = {
    10001: [(100.0, 101.0), (101.0, 101.0), (103.0, 102.0), (102.0, 104.0), (104.0, 106.0)],
    10002: [(50.0, 50.2), (50.2, 50.4), (49.9, 50.0), (50.0, 50.1), (50.1, 50.3)],
    10003: [(20.0, 21.0), (21.0, 22.0), (22.0, 22.25), (22.25, 22.5), None],  # no bars after the delisting
    10004: [(30.0, 29.0), (29.0, 28.0), (27.5, 27.0), (27.0, 26.0), (26.0, 25.5)],
}
SPLIT: dict[tuple[int, int], float] = {(10004, 3): 1.5}  # (permno, day) -> splitFactor; 10001's set by --factor
DIV = {(10002, 2): 0.50, (10004, 2): 0.25}                # (permno, day) -> divCash per share on the ex-date
DELIST = {10003: 3}                                       # permno -> delisting bar b; settled at open of b+1
RAW: dict[int, list] = {}

# Order schedule the Strategy replays: day -> [(permno, side, qty)], decided at the previous close.
ORDERS = {
    0: [(10001, "BUY", 100), (10002, "BUY", 100), (10003, "BUY", 100), (10004, "SELL", 100)],
    1: [(10002, "SELL", 40)],
    2: [(10001, "SELL", 100)],  # decided at the close of day 2 on 100 pre-split shares: a full exit
}


def build_raw(k: float) -> None:
    SPLIT[(10001, 2)] = k
    for p, rows in ADJ.items():
        cum, out = 1.0, []
        for d, row in enumerate(rows):
            cum *= SPLIT.get((p, d), 1.0)
            out.append(None if row is None else (round(row[0] / cum, PREC), round(row[1] / cum, PREC)))
        RAW[p] = out


def split_shares(q: int, k: float, pre_close: float) -> tuple[int, float]:
    """New whole-share count (toward zero) and cash in lieu of the fraction at pre_close / k."""
    exact = abs(q) * k
    new = math.floor(exact + 1e-9)
    sign = 1 if q > 0 else -1
    return sign * new, sign * round((exact - new) * pre_close / k, 2)


def scaled_orders(day: int, scale_pending: bool) -> list[tuple[int, str, int]]:
    """What the venue's open submitter sends at open(day) + 1 ns."""
    out = []
    for p, side, q in ORDERS.get(day, []):
        k = SPLIT.get((p, day), 1.0) if scale_pending else 1.0
        out.append((p, side, math.floor(q * k + 1e-9)))
    return out


# ------------------------------------------------------------------- venue side --------------

class CorporateActionFeeModel(FeeModel):
    """Per-share fee; zero for corporate-action fills and expiration settlements."""

    def get_commission(self, order, fill_qty, fill_px, instrument):
        tags = order.tags or []
        if any(t.startswith(CA_TAG) or (t.startswith("EXPIRATION_") and t.endswith("_CLOSE")) for t in tags):
            return Money(0, USD)
        return Money(fee(int(fill_qty.as_double())), USD)


class CorporateActionModule(SimulationModule):
    """Applies splits, dividends and (variant 'module') delisting settlements at 09:30 of the ex-date."""

    def __init__(self, actions, split_variant: str, pre_close: dict):
        super().__init__(SimulationModuleConfig())
        self._actions = sorted(actions, key=lambda a: a[0])  # (ts, kind, instrument_id, value)
        self._split_variant = split_variant
        self._pre_close = pre_close  # (instrument_id) -> raw close before the ex-date
        self.applied: list[str] = []

    def process(self, ts_now: int) -> None:
        while self._actions and self._actions[0][0] <= ts_now:
            _, kind, instrument_id, value = self._actions.pop(0)
            for position in self.cache.positions_open(None, instrument_id):
                getattr(self, f"_{kind}")(position, value, ts_now)

    def _split(self, position, k, ts_now):
        q = int(position.signed_qty)
        pre = self._pre_close[position.instrument_id]
        new_q, cash_in_lieu = split_shares(q, k, pre)
        if self._split_variant == "A":
            # A: one synthetic fill of the share-count change at price 0: no cash moves.
            delta = new_q - q
            if delta:
                side = OrderSide.BUY if delta > 0 else OrderSide.SELL
                self._fill(position, side, abs(delta), Price(0, PREC), "SPLIT")
            if cash_in_lieu:
                self.exchange.adjust_account(Money(cash_in_lieu, USD))
        else:
            # B: close all at the pre-split close, reopen new_q at pre-split close / k (rounded to
            # the tick). The fraction is implicitly sold at the close, so no separate cash in lieu.
            self._fill(position, OrderSide.SELL if q > 0 else OrderSide.BUY, abs(q), Price(pre, PREC), "SPLIT_CLOSE")
            self._fill(position, OrderSide.BUY if q > 0 else OrderSide.SELL, abs(new_q),
                       Price(pre / k, PREC), "SPLIT_REOPEN")
        self.applied.append(f"{_et(ts_now)} split x{k:g} {position.instrument_id}: {q} -> {new_q},"
                            f" cash in lieu {cash_in_lieu:+.2f}")

    def _dividend(self, position, per_share, ts_now):
        amount = round(float(position.signed_qty) * per_share, 2)  # a short pays it
        self.exchange.adjust_account(Money(amount, USD))
        self.applied.append(f"{_et(ts_now)} dividend {per_share} x {position.signed_qty:g}"
                            f" {position.instrument_id} = {amount:+.2f}")

    def _delist(self, position, last_valuation, ts_now):
        q = int(position.signed_qty)
        self._fill(position, OrderSide.SELL if q > 0 else OrderSide.BUY, abs(q), Price(last_valuation, PREC), "DELIST")
        self.applied.append(f"{_et(ts_now)} delisting settled {position.instrument_id} {q} @ {last_valuation}")

    def _fill(self, position, side, qty, px, label):
        """A venue-generated order + fill on the holder's position, as check_instrument_expiration does."""
        now = self.clock.timestamp_ns()
        order = MarketOrder(
            trader_id=position.trader_id,
            strategy_id=position.strategy_id,
            instrument_id=position.instrument_id,
            client_order_id=ClientOrderId(f"CA-{label}-{uuid.uuid4().hex[:8]}"),
            order_side=side,
            quantity=Quantity(qty, 0),
            init_id=UUID4(),
            ts_init=now,
            time_in_force=TimeInForce.DAY,
            tags=[f"{CA_TAG}_{label}"],
        )
        self.cache.add_order(order, position_id=position.id)
        client = self.exchange.exec_client
        client.generate_order_submitted(order.strategy_id, order.instrument_id, order.client_order_id, now)
        client.generate_order_accepted(order.strategy_id, order.instrument_id, order.client_order_id,
                                       VenueOrderId(order.client_order_id.value), now)
        engine = self.exchange.get_matching_engine(position.instrument_id)
        engine.apply_fills(order, [(px, Quantity(qty, 0))], LiquiditySide.TAKER, None,
                           self.cache.position(position.id))

    def log_diagnostics(self, logger):
        pass

    def reset(self):
        pass


def _et(ts: int) -> str:
    return str(pd.Timestamp(ts, tz="UTC").tz_convert("America/New_York").strftime("%m-%d %H:%M"))


# ---------------------------------------------------------------- strategy side --------------

class ScheduleConfig(StrategyConfig, frozen=True):
    instrument_ids: tuple[str, ...] = ()
    scale_pending: bool = True


class ScheduleStrategy(Strategy):
    """Venue-agnostic stand-in for PortfolioStrategy: submits the schedule, records each close."""

    def __init__(self, config):
        super().__init__(config)
        self.records, self.fills = [], []

    def on_start(self):
        for s in self.config.instrument_ids:
            self.subscribe_trade_ticks(InstrumentId.from_str(s))
        for d, day in enumerate(DAYS):
            if d in ORDERS:
                self.clock.set_time_alert(f"open{d}", pd.Timestamp(ts_ns(day, "09:30") + 1, tz="UTC"),
                                          lambda e, d=d: self._submit(d))
            self.clock.set_time_alert(f"close{d}", pd.Timestamp(ts_ns(day, "16:00") + 1, tz="UTC"), self._record)

    def _submit(self, d):
        # stands in for the venue's open submitter (ADR 0003), which may rescale for an ex-date split
        for p, side, q in scaled_orders(d, self.config.scale_pending):
            self.submit_order(self.order_factory.market(iid(p), OrderSide[side], Quantity(q, 0),
                                                        time_in_force=TimeInForce.DAY))

    def on_order_filled(self, event):
        self.fills.append(f"{_et(event.ts_event)} {event.instrument_id} {event.order_side.name}"
                          f" {event.last_qty} @ {event.last_px} fee {event.commission} [{event.client_order_id}]")

    def _record(self, event):
        acct = self.portfolio.account(VENUE)
        balance = acct.balance_total(USD).as_double()
        open_pos = self.cache.positions_open()
        cost = sum(float(p.signed_qty) * p.avg_px_open for p in open_pos)
        mv = sum(float(p.signed_qty) * self.cache.trade_tick(p.instrument_id).price.as_double() for p in open_pos)
        unreal = self.portfolio.unrealized_pnls(VENUE).get(USD, Money(0, USD)).as_double()
        self.records.append(dict(
            balance=balance, cash=balance - cost, mv=mv, unreal=unreal,
            qty={int(s.split(".")[0]): float(self.portfolio.net_position(InstrumentId.from_str(s)))
                 for s in self.config.instrument_ids},
        ))


# ------------------------------------------------------------------------- run ---------------

def run_nautilus(split_variant, delist_variant, scale_pending):
    engine = BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(log_level="ERROR")))
    actions, settlement = [], {}
    for (p, d), k in SPLIT.items():
        actions.append((ts_ns(DAYS[d], "09:30"), "split", iid(p), k))
    for (p, d), c in DIV.items():
        actions.append((ts_ns(DAYS[d], "09:30"), "dividend", iid(p), c))
    for p, b in DELIST.items():
        if delist_variant == "module":
            actions.append((ts_ns(DAYS[b + 1], "09:30"), "delist", iid(p), RAW[p][b][1]))
        else:
            settlement[iid(p)] = RAW[p][b][1]
    pre_close = {iid(p): RAW[p][d - 1][1] for (p, d) in SPLIT}
    module = CorporateActionModule(actions, split_variant, pre_close)
    engine.add_venue(
        VENUE, OmsType.NETTING, AccountType.MARGIN, [Money(INIT_CASH, USD)],
        base_currency=USD, default_leverage=Decimal(1), modules=[module],
        fee_model=CorporateActionFeeModel(), settlement_prices=settlement or None,
    )
    t0 = ts_ns(DAYS[0], "00:00")
    for p in RAW:
        engine.add_instrument(Equity(iid(p), Symbol(str(p)), USD, PREC, Price(10 ** -PREC, PREC), Quantity(1, 0),
                                     t0, t0, margin_init=Decimal(1), margin_maint=Decimal(1)))
    data, n = [], 0
    for p, rows in RAW.items():
        for day, row in zip(DAYS, rows):
            if row is None:
                continue
            for hhmm, px in (("09:30", row[0]), ("16:00", row[1])):
                n += 1
                ts = ts_ns(day, hhmm)
                data.append(TradeTick(iid(p), Price(px, PREC), Quantity(1_000_000, 0),
                                      AggressorSide.NO_AGGRESSOR, TradeId(str(n)), ts, ts))
        if delist_variant == "native" and p in DELIST:
            ts = ts_ns(DAYS[DELIST[p] + 1], "09:30")
            data.append(InstrumentClose(iid(p), Price(RAW[p][DELIST[p]][1], PREC),
                                        InstrumentCloseType.CONTRACT_EXPIRED, ts, ts))
    engine.add_data(data)
    strat = ScheduleStrategy(ScheduleConfig(instrument_ids=tuple(str(iid(p)) for p in RAW),
                                            scale_pending=scale_pending))
    engine.add_strategy(strat)
    engine.run(end=pd.Timestamp(ts_ns(DAYS[-1], "17:00"), tz="UTC"))
    return engine, strat, module


def reference_ledger(scale_pending):
    """Independent ledger: corporate actions at 09:30, orders at the open, state after the close."""
    cash, qty, out = INIT_CASH, {p: 0 for p in RAW}, []
    for d in range(len(DAYS)):
        for p in RAW:
            if (p, d) in SPLIT and qty[p]:
                qty[p], cil = split_shares(qty[p], SPLIT[(p, d)], RAW[p][d - 1][1])
                cash += cil
            if (p, d) in DIV:
                cash += round(qty[p] * DIV[(p, d)], 2)
            if p in DELIST and d == DELIST[p] + 1:
                cash += qty[p] * RAW[p][DELIST[p]][1]
                qty[p] = 0
        for p, side, q in scaled_orders(d, scale_pending):
            sign = 1 if side == "BUY" else -1
            cash -= sign * q * RAW[p][d][0] + fee(q)
            qty[p] += sign * q
        close = {p: next(r[1] for r in reversed(RAW[p][: d + 1]) if r) for p in RAW}
        out.append(dict(cash=cash, qty=dict(qty), equity=cash + sum(qty[p] * close[p] for p in RAW)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["A", "B"], default="A")
    ap.add_argument("--factor", type=float, default=2.0)
    ap.add_argument("--delist", choices=["native", "module"], default="module")
    ap.add_argument("--no-scale-pending", dest="scale_pending", action="store_false")
    args = ap.parse_args()
    build_raw(args.factor)
    engine, strat, module = run_nautilus(args.split, args.delist, args.scale_pending)
    ref = reference_ledger(args.scale_pending)

    print(f"=== split {args.split}, 10001 splitFactor {args.factor:g}, delisting {args.delist},"
          f" scale pending {args.scale_pending} ===")
    print("-- applied by the venue module:")
    for a in module.applied:
        print("  ", a)
    print("-- fills the Strategy received:")
    for f in strat.fills:
        print("  ", f)
    print("-- after each close (n = nautilus-derived, r = reference ledger):")
    ok = True
    for day, rec, r in zip(DAYS, strat.records, ref):
        n_equity = rec["cash"] + rec["mv"]
        c = {
            "qty": all(abs(rec["qty"][p] - r["qty"][p]) < 1e-9 for p in RAW),
            "cash": abs(rec["cash"] - r["cash"]) < 1e-6,
            "equity": abs(n_equity - r["equity"]) < 1e-6,
        }
        ok &= all(c.values())
        print(f"   {day.date()} qty={ {p: int(v) for p, v in rec['qty'].items()} }"
              f" | n cash={rec['cash']:.2f} r cash={r['cash']:.2f}"
              f" | n equity={n_equity:.2f} r equity={r['equity']:.2f}"
              f" | margin balance={rec['balance']:.2f} (== cash: {abs(rec['balance'] - r['cash']) < 1e-6})"
              f" | balance+Portfolio.unrealized={rec['balance'] + rec['unreal']:.2f}"
              f" | {'ok' if all(c.values()) else c}")
    for p in (10001, 10004):
        for pos in engine.cache.positions(instrument_id=iid(p)):
            print(f"-- {pos.id}: signed_qty={pos.signed_qty:g} avg_px_open={pos.avg_px_open:.4f}"
                  f" realized_pnl={pos.realized_pnl} closed={pos.is_closed}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
