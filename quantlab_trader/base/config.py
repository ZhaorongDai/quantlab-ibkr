"""trader's run configuration: ``TraderConfig`` and the root ``VenueConfig``.

Frozen dataclasses with ``get_config()`` / ``from_config()``, as quantlab's
configs: ``get_config()`` returns plain JSON values plus, for a class chosen
at run time (a venue config, a tracker), its import path under ``"name"``, and
``from_config()`` rebuilds the same object from that dict, so a trader run's
``config.json`` rebuilds the run.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Self

import pandas as pd
import xarray as xr

from quantlab.base.tracking import Tracker
from quantlab.utils.module import get_cls_from_path

if TYPE_CHECKING:
    from quantlab_trader.base.venue import Venue
    from quantlab_trader.quantlab_run import QuantlabRun


def _import_path(obj: Any) -> str:
    """Return ``module.QualName`` of ``obj``'s class."""
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class VenueConfig(ABC):
    """Root of the venue configs; a subclass names one venue and its settings.

    ``get_config()`` returns the fields plus the subclass's import path under
    ``"name"``; ``VenueConfig.from_config`` picks the subclass from it.
    """

    @abstractmethod
    def build(
        self,
        run: QuantlabRun,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        permnos: tuple,
        loop: str,
        predictions: xr.Dataset | None = None,
        history_start: pd.Timestamp | None = None,
    ) -> Venue:
        """Return the venue that executes ``run`` from ``start`` to ``end``.

        Parameters
        ----------
        run : QuantlabRun
            The quantlab run being executed.
        start, end : pandas.Timestamp
            The replay window, both inclusive.
        permnos : tuple
            The securities the strategy can trade (quantlab's symbol labels).
        loop : {"closed", "open"}
            The replay's loop; a backtest venue picks its default fee model
            from it (ADR 0003), a live venue ignores it.
        predictions : xarray.Dataset, optional
            Closed loop: the prediction panel over the window, whose rows the
            decision source hands out and whose symbols the decision inputs
            are on.
        history_start : pandas.Timestamp, optional
            Closed loop: where the decision-price history starts,
            ``bar_before(anchor, lookback_bars)`` (ADR 0008); ``start`` when
            omitted.
        """

    def get_config(self) -> dict[str, Any]:
        """Return the fields as JSON values plus ``"name"``, the class's import path."""
        return {**dataclasses.asdict(self), "name": _import_path(self)}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Rebuild the venue config ``get_config()`` returned.

        Called on ``VenueConfig`` itself, the subclass named by ``"name"``
        rebuilds it; called on a subclass, that subclass does.
        """
        target = _class_named(config["name"], cls) if "name" in config else cls
        fields = {k: v for k, v in config.items() if k != "name"}
        return target._from_fields(fields)

    @classmethod
    def _from_fields(cls, fields: dict[str, Any]) -> Self:
        """Build the config from its JSON fields; a subclass rebuilds nested values."""
        return cls(**fields)


def _class_named(path: str, base: type) -> type:
    """Return the class at import ``path``, refusing one that is not a ``base``."""
    target = get_cls_from_path(path)
    if not (isinstance(target, type) and issubclass(target, base)):
        raise TypeError(f"{path} is not a {base.__name__}")
    return target


@dataclass(frozen=True)
class TraderConfig:
    """One trader run: which quantlab run to execute, on which venue, and how.

    Parameters
    ----------
    quantlab_run : str
        The quantlab run directory (its ``config.json``, ``weights.zarr``, ...).
    venue : VenueConfig
        The venue; ``BacktestVenueConfig`` in v1.
    loop : {"closed", "open"}, default "closed"
        Closed-loop replay runs quantlab's constructor on the account's
        holdings; open-loop replay executes the run's rebalance table.
    start, end : str or None
        Narrow the run's window (inclusive); ``None`` keeps the run's.
    output_dir : str or None
        Where the run directory is created; ``None`` puts it beside the
        quantlab run directory.
    tracker : quantlab.base.tracking.Tracker or None
        Where metrics are logged; ``None`` means the quantlab run's tracker.
    name : str or None
        Prefix of the run directory's name; ``None`` uses the quantlab run
        directory's name.

    Examples
    --------
    >>> from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
    >>> config = TraderConfig("/runs/WeightsVectorBt_1", BacktestVenueConfig(), loop="open")
    >>> TraderConfig.from_config(config.get_config()) == config
    True
    """

    quantlab_run: str
    venue: VenueConfig
    loop: Literal["closed", "open"] = "closed"
    start: str | None = None
    end: str | None = None
    output_dir: str | None = None
    tracker: Tracker | None = None
    name: str | None = None

    def __post_init__(self):
        if self.loop not in ("closed", "open"):
            raise ValueError(f"TraderConfig.loop must be 'closed' or 'open', got {self.loop!r}")
        if not isinstance(self.venue, VenueConfig):
            raise TypeError(
                f"TraderConfig.venue must be a VenueConfig, got {type(self.venue).__name__}"
            )

    def get_config(self) -> dict[str, Any]:
        """Return the config as JSON values; the venue and tracker carry their ``"name"``."""
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        fields["venue"] = self.venue.get_config()
        fields["tracker"] = None if self.tracker is None else self.tracker.get_config()
        return fields

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Rebuild the config ``get_config()`` returned; unknown keys are records and ignored.

        Parameters
        ----------
        config : Mapping
            ``get_config()``'s dict, for example a trader run's ``config.json``.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        fields = {k: v for k, v in config.items() if k in names}
        fields["venue"] = VenueConfig.from_config(fields["venue"])
        tracker = fields.get("tracker")
        if tracker is not None:
            fields["tracker"] = _class_named(tracker["name"], Tracker).from_config(tracker)
        return cls(**fields)
