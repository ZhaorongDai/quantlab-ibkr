"""``QuantlabRun``: the only reader of a quantlab run directory.

A quantlab run directory holds the backtest's ``config.json`` (the price
dataset's config, the window, costs and the ``market`` block naming the fill
and valuation price columns) and its outputs, among them the rebalance table
``weights.zarr``. trader learns everything about the run here, without
importing quantlab's backtest layer (which loads vectorbt).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pandas as pd
import xarray as xr

from quantlab.base.data import MarketDataset
from quantlab.utils.module import load_dataset_from_config

#: Price variables trader cannot execute without: the raw open (fills) and
#: close (sizing and the equity mark), the adjusted close (decision prices,
#: ADR 0002), and the corporate-action fields the backtest venue books splits,
#: value distributions and dividends from (ADR 0009).
REQUIRED_PRICE_VARIABLES: tuple[str, ...] = (
    "open",
    "close",
    "adjClose",
    "splitFactor",
    "cumfacshr",
    "divCash",
)


@dataclass(frozen=True)
class QuantlabRun:
    """A quantlab backtest run, as trader executes it.

    Attributes
    ----------
    run_dir : pathlib.Path
        The run directory, absolute.
    config : dict
        Its ``config.json``.
    price_dataset : quantlab.base.data.MarketDataset
        The run's price dataset, rebuilt from its config.
    market : dict
        The ``market`` block: ``fill_price_column`` and
        ``valuation_price_column``, the run's (adjusted) decision columns.
    rebalance_periods : int
        Bars between rebalances.
    window : tuple of pandas.Timestamp
        The run's first and last bar dates (``start_date``, ``end_date``).
    init_cash : float
        The run's starting cash.
    fees : float
        The run's fee, a fraction of the traded notional.
    slippage : float
        The run's slippage, a fraction of the fill price.
    data_fingerprint : dict or None
        The run's record of the data it read, for a trader run's config.
    """

    run_dir: Path
    config: dict
    price_dataset: MarketDataset
    market: dict
    rebalance_periods: int
    window: tuple[pd.Timestamp, pd.Timestamp]
    init_cash: float
    fees: float
    slippage: float
    data_fingerprint: dict | None

    @classmethod
    def load(cls, run_dir: str | Path) -> Self:
        """Read the run at ``run_dir``, refusing one trader cannot execute.

        Parameters
        ----------
        run_dir : str or pathlib.Path
            A quantlab backtest run directory.

        Returns
        -------
        QuantlabRun

        Raises
        ------
        ValueError
            If ``run_dir`` has no ``config.json``, it has no ``market`` block, the price dataset is
            not a ``MarketDataset``, or its store lacks any of
            ``REQUIRED_PRICE_VARIABLES``.
        """
        run_dir = Path(run_dir).resolve()
        if not (run_dir / "config.json").is_file():
            raise ValueError(f"{run_dir} is not a quantlab run directory: no config.json")
        config = json.loads((run_dir / "config.json").read_text())
        market = config.get("market")
        if not market:
            raise ValueError(
                f"quantlab run {run_dir}: config.json has no market block "
                f"(fill_price_column, valuation_price_column); rerun it with a "
                f"quantlab that writes one (quantlab #106)"
            )
        dataset = load_dataset_from_config(config["price_dataset"], run_dir=run_dir)
        if not isinstance(dataset, MarketDataset):
            raise ValueError(
                f"quantlab run {run_dir}: the price dataset "
                f"{type(dataset).__name__} is not a MarketDataset"
            )
        window = (pd.Timestamp(config["start_date"]), pd.Timestamp(config["end_date"]))
        variables = set(dataset.panel(*window).data_vars)
        missing = [name for name in REQUIRED_PRICE_VARIABLES if name not in variables]
        if missing:
            raise ValueError(
                f"quantlab run {run_dir}: the price dataset "
                f"{type(dataset).__name__} lacks {missing}; trader executes on raw "
                f"open/close, decides on adjClose (ADR 0002) and books corporate "
                f"actions from splitFactor, cumfacshr and divCash (ADR 0009)"
            )
        return cls(
            run_dir=run_dir,
            config=config,
            price_dataset=dataset,
            market=dict(market),
            rebalance_periods=int(config["rebalance_periods"]),
            window=window,
            init_cash=float(config["init_cash"]),
            fees=float(config["fees"]),
            slippage=float(config["slippage"]),
            data_fingerprint=config.get("data_fingerprint"),
        )

    def rebalance_table(self) -> xr.Dataset:
        """Return the run's ``weights.zarr``: ``weight`` on ``(timestamp, symbol)``, loaded."""
        with xr.open_zarr(self.run_dir / "weights.zarr") as table:
            return table.load()
