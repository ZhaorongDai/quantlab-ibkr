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
from typing import TYPE_CHECKING, Any, Self

from quantlab.base.tracking import Tracker
from quantlab.utils.module import get_cls_from_path
from quantlab_trader.base.venue import Loop

if TYPE_CHECKING:
    from quantlab_trader.base.venue import ReplayRequest, Venue
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

    Examples
    --------
    >>> from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
    >>> isinstance(BacktestVenueConfig(), VenueConfig)
    True
    """

    @abstractmethod
    def build(self, run: QuantlabRun, request: ReplayRequest) -> Venue:
        """Return the venue that executes ``run`` as ``request`` asks.

        Parameters
        ----------
        run : QuantlabRun
            The quantlab run being executed.
        request : ReplayRequest
            The window, the instruments, the loop and, closed loop, the
            prediction panel and the start of the decision-price history.

        Examples
        --------
        ``runner.run`` builds the venue it runs the strategy on::

            venue = config.venue.build(quantlab_run, request)
        """

    def get_config(self) -> dict[str, Any]:
        """Return the fields as JSON values plus ``"name"``, the class's import path.

        Examples
        --------
        >>> from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
        >>> BacktestVenueConfig().get_config()["name"]
        'quantlab_trader.venue.backtest.venue.BacktestVenueConfig'
        """
        return {**dataclasses.asdict(self), "name": _import_path(self)}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Self:
        """Rebuild the venue config ``get_config()`` returned.

        Called on ``VenueConfig`` itself, the subclass named by ``"name"``
        rebuilds it; called on a subclass, that subclass does.

        Examples
        --------
        >>> from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
        >>> config = VenueConfig.from_config(BacktestVenueConfig().get_config())
        >>> type(config).__name__
        'BacktestVenueConfig'
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
    loop : Loop or {"closed", "open"}, default Loop.CLOSED
        Closed-loop replay runs quantlab's constructor on the account's
        holdings; open-loop replay executes the run's rebalance table. A
        string is converted to its ``Loop``.
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
    loop: Loop = Loop.CLOSED
    start: str | None = None
    end: str | None = None
    output_dir: str | None = None
    tracker: Tracker | None = None
    name: str | None = None

    def __post_init__(self):
        try:
            object.__setattr__(self, "loop", Loop(self.loop))
        except ValueError:
            raise ValueError(
                f"TraderConfig.loop must be one of {[loop.value for loop in Loop]}, "
                f"got {self.loop!r}"
            ) from None
        if not isinstance(self.venue, VenueConfig):
            raise TypeError(
                f"TraderConfig.venue must be a VenueConfig, got {type(self.venue).__name__}"
            )

    def get_config(self) -> dict[str, Any]:
        """Return the config as JSON values; the venue and tracker carry their ``"name"``.

        Examples
        --------
        >>> from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
        >>> config = TraderConfig("/runs/WeightsVectorBt_1", BacktestVenueConfig(), loop="open")
        >>> fields = config.get_config()
        >>> fields["loop"], fields["venue"]["name"].rsplit(".", 1)[-1], fields["tracker"]
        ('open', 'BacktestVenueConfig', None)
        """
        fields = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        fields["loop"] = self.loop.value
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

        Examples
        --------
        >>> from quantlab_trader.venue.backtest.venue import BacktestVenueConfig
        >>> recorded = {
        ...     **TraderConfig("/runs/WeightsVectorBt_1", BacktestVenueConfig()).get_config(),
        ...     "quantlab_records": {},  # a record config.json adds; ignored
        ... }
        >>> TraderConfig.from_config(recorded).loop
        <Loop.CLOSED: 'closed'>
        """
        names = {f.name for f in dataclasses.fields(cls)}
        fields = {k: v for k, v in config.items() if k in names}
        fields["venue"] = VenueConfig.from_config(fields["venue"])
        tracker = fields.get("tracker")
        if tracker is not None:
            fields["tracker"] = _class_named(tracker["name"], Tracker).from_config(tracker)
        return cls(**fields)
