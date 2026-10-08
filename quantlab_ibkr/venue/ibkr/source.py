"""The IBKR venue's decision source: the live prediction row of the last closed bar.

Live trading is a daily batch (#47): one invocation decides at most one bar,
the last closed bar t, which is the last bar of the quantlab run's price
dataset (the daily prediction job extends it, quantlab #233). The bar's
predictions are its row of the run's **live prediction store**
(``quantlab.runs.live_predictions.LivePredictionStore``), written by quantlab
from the run's checkpoint; its raw close and delisting mark come from the
run's price dataset, read up to t and no later, exactly as the backtest venue
reads them (``BacktestDecisionSource`` over a window ending at t).

A day without a prediction row for t, or a store holding a row for a bar the
prices do not reach yet, is a **hold**: ``hold_reason`` says why, and the live
clock never fires on it.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.runs.live_predictions import LivePredictionStore
from quantlab_ibkr.base.venue import BarInputs, DecisionSource, Loop, ReplayRequest
from quantlab_ibkr.quantlab_run import QuantlabRun
from quantlab_ibkr.venue.backtest.source import BacktestDecisionSource

#: Bars of raw-price history read before t, so a security without a price at
#: t is still marked at its last raw close (a halted holding).
HISTORY_BARS = 260

#: Hold reasons; formatted with the bar labels.
NO_PREDICTION_ROW = "no prediction row for {t} in {store}"
PREDICTIONS_AHEAD = (
    "the prediction store's last row {last} is later than the prices' last bar {t}"
)


def _day(t: pd.Timestamp) -> str:
    return str(pd.Timestamp(t).date())


class LiveDecisionSource(DecisionSource):
    """Hand out the inputs of the last closed bar t, from the live prediction store.

    Parameters
    ----------
    run : QuantlabRun
        The quantlab run traded live: its price dataset and (through the
        store's ``run_dir``) the predictions' owner.
    store : LivePredictionStore or str or pathlib.Path
        The run's live prediction store.
    symbols : Iterable of Hashable, optional
        Symbols to price besides the predicted ones: the account's holdings
        as far as they are known before the node connects (the conId cache).
    t : pandas.Timestamp, optional
        The decision bar; the price dataset's last bar when ``None``. A bar
        after that is refused.
    history_bars : int, default HISTORY_BARS
        Bars of raw-price history read before t.

    Attributes
    ----------
    t : pandas.Timestamp
        The decision bar.
    last_price_bar : pandas.Timestamp
        The price dataset's last bar.
    hold_reason : str or None
        Why t cannot be decided (no prediction row, or predictions ahead of
        the prices); ``None`` when it can.
    symbols : tuple
        The symbols priced: the predicted ones (a finite prediction in t's
        row) then ``symbols``.
    excluded : frozenset
        Symbols whose predictions ``inputs`` masks (``exclude``).

    Raises
    ------
    ValueError
        If the store holds another run's predictions, or ``t`` is after the
        price dataset's last bar.

    Examples
    --------
    The IBKR venue builds it from the run and its live store::

        source = LiveDecisionSource(QuantlabRun.load(run_dir), "live/live_predictions.zarr")
        if source.hold_reason is None:
            inputs = source.inputs(source.t)
    """

    def __init__(
        self,
        run: QuantlabRun,
        store: LivePredictionStore | str | Path,
        *,
        symbols: Iterable[Hashable] = (),
        t: pd.Timestamp | None = None,
        history_bars: int = HISTORY_BARS,
    ):
        self.store = store if isinstance(store, LivePredictionStore) else LivePredictionStore(store)
        owner = self.store.attrs().get("run_dir")
        if owner is not None and Path(owner).resolve() != Path(run.run_dir).resolve():
            raise ValueError(
                f"{self.store.path} holds the live predictions of {owner}, not of {run.run_dir}"
            )
        dataset = run.price_dataset
        self.last_price_bar = pd.Timestamp(
            dataset.bar_after(run.window[0], np.iinfo(np.int64).max)
        )
        self.t = self.last_price_bar if t is None else pd.Timestamp(t)
        if self.t > self.last_price_bar:
            raise ValueError(
                f"decision bar {_day(self.t)} is after the prices' last bar "
                f"{_day(self.last_price_bar)}"
            )
        self.hold_reason = self._hold_reason()
        self._row = None if self.hold_reason else self._read_row()
        predicted = [] if self._row is None else _predicted(self._row)
        self.symbols = tuple(dict.fromkeys([*predicted, *symbols]))
        self.excluded: frozenset = frozenset()
        # The raw-close window, not a decision input: the rule's history and
        # the cadence are quantlab's DecisionInputs'.
        bars = dataset.calendar(pd.Timestamp(0), self.t)
        start = bars[max(len(bars) - 1 - history_bars, 0)]
        self._prices = (
            BacktestDecisionSource(
                run, ReplayRequest(start, self.t, self.symbols, Loop.CLOSED)
            )
            if self.symbols
            else None
        )

    def _hold_reason(self) -> str | None:
        bars = self.store.bars()
        if len(bars) and bars[-1] > self.last_price_bar:
            return PREDICTIONS_AHEAD.format(
                last=_day(bars[-1]), t=_day(self.last_price_bar)
            )
        if self.t not in bars:
            return NO_PREDICTION_ROW.format(t=_day(self.t), store=self.store.path)
        return None

    def _read_row(self) -> xr.Dataset:
        row = self.store.row(self.t)
        return row.drop_vars([c for c in row.coords if c != "symbol"])

    def exclude(self, symbols: Iterable[Hashable]) -> None:
        """Mask the predictions of ``symbols`` for the day: the rule cannot select them.

        The IBKR venue excludes the symbols its resolver could not map to a
        single contract, so no target is ever set on one.

        Examples
        --------
        ::

            source.exclude(resolver.unresolved(source.t))
        """
        self.excluded = self.excluded | frozenset(symbols)

    def calendar(self) -> pd.DatetimeIndex:
        """Return the one bar this source decides: t.

        Examples
        --------
        ::

            (t,) = source.calendar()
        """
        return pd.DatetimeIndex([self.t])

    def inputs(self, t: pd.Timestamp) -> BarInputs:
        """Return the inputs of the decision at the close of t.

        Parameters
        ----------
        t : pandas.Timestamp
            The source's bar.

        Raises
        ------
        ValueError
            For another bar, or a bar that holds (``hold_reason``).

        Examples
        --------
        ::

            inputs = source.inputs(source.t)
            result = cycle.run(inputs, positions, cash)
        """
        t = pd.Timestamp(t)
        if t != self.t:
            raise ValueError(f"the live source decides {_day(self.t)} only, not {_day(t)}")
        if self.hold_reason is not None:
            raise ValueError(f"{_day(t)} holds: {self.hold_reason}")
        predictions = self._row
        if self.excluded:
            mask = xr.DataArray(
                ~np.isin(predictions["symbol"].values, list(self.excluded)),
                dims="symbol",
                coords={"symbol": predictions["symbol"]},
            )
            predictions = predictions.where(mask)
        if self._prices is None:
            empty = pd.Series(dtype=float)
            return BarInputs(t, predictions, empty, empty.astype(bool))
        bar = self._prices.inputs(t)
        return BarInputs(t, predictions, bar.close, bar.delisted)


def _predicted(row: xr.Dataset) -> list:
    """Return the symbols of ``row`` with a finite prediction of any label, in row order."""
    finite = np.zeros(row.sizes["symbol"], dtype=bool)
    for name in row.data_vars:
        finite |= np.isfinite(np.asarray(row[name].values, dtype=float))
    return [v.item() for v in row["symbol"].values[finite]]
