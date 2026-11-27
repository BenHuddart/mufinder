"""Types used by the convergence configuration UI and adapter."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
import re
from typing import Generic, TypeAlias, TypeVar
from uuid import UUID

from mufinder.science.kpoint_grids import format_kpoint_grid


KPointGrid: TypeAlias = tuple[int, int, int]
SweepValue: TypeAlias = float | KPointGrid

SweepKeyT = TypeVar("SweepKeyT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class SweepKeyCodec(Generic[SweepKeyT]):
    """How one convergence sweep spells the key of a sweep point.

    A cutoff sweep is keyed by a float energy in eV and a k-point sweep by a
    Monkhorst-Pack grid triplet. The sweep widget and controller handle both
    through this codec, so neither repeats the conversions.
    """

    #: Turn a raw value (persisted state, a runtime-state key, a signal
    #: payload) into the key. Raises ``TypeError`` or ``ValueError``.
    normalize: Callable[[object], SweepKeyT]
    #: Parse the key column's text. Raises ``ValueError``.
    parse: Callable[[str], SweepKeyT]
    #: The key column's text for a key.
    format: Callable[[SweepKeyT], str]
    #: How dialogs and warnings name a key (with its unit).
    describe: Callable[[SweepKeyT], str]
    #: The key's JSON form in the persisted form state.
    to_state: Callable[[SweepKeyT], object]

    def coerce(self, raw: object) -> SweepKeyT | None:
        """Return the key for ``raw``, or ``None`` when it is not one."""

        try:
            return self.normalize(raw)
        except (TypeError, ValueError):
            return None


def normalize_kpoint_grid(raw: object) -> KPointGrid:
    """Return ``raw`` (a three-item list or tuple) as a positive integer grid."""

    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("K-point grid must contain exactly three integers.")
    grid = (int(raw[0]), int(raw[1]), int(raw[2]))
    if grid[0] <= 0 or grid[1] <= 0 or grid[2] <= 0:
        raise ValueError("K-point grid values must be positive integers.")
    return grid


def parse_kpoint_grid(text: str) -> KPointGrid:
    """Parse a grid typed as three integers with any separators (``2 x 2 x 1``)."""

    parts = [part for part in re.split(r"[^0-9]+", text.strip()) if part]
    if len(parts) != 3:
        raise ValueError("K-point grid must contain exactly three integers.")
    return normalize_kpoint_grid(parts)


def parse_grid_mapping_key(value: str) -> KPointGrid | None:
    """Return the grid a ``job_ids_by_grid`` key such as ``"2x2x1"`` spells.

    ``None`` when the key is not three ``x``-separated positive integers.
    """

    parts = value.strip().split("x")
    if len(parts) != 3:
        return None
    try:
        grid = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None
    if grid[0] <= 0 or grid[1] <= 0 or grid[2] <= 0:
        return None
    return grid


CUTOFF_KEY_CODEC: SweepKeyCodec[float] = SweepKeyCodec(
    normalize=lambda raw: float(raw),  # type: ignore[arg-type]
    parse=float,
    format=lambda cutoff: f"{float(cutoff):.8g}",
    describe=lambda cutoff: f"{float(cutoff):.8g} eV",
    to_state=float,
)

KPOINT_GRID_CODEC: SweepKeyCodec[KPointGrid] = SweepKeyCodec(
    normalize=normalize_kpoint_grid,
    parse=parse_kpoint_grid,
    format=format_kpoint_grid,
    describe=format_kpoint_grid,
    to_state=list,
)


@dataclass(frozen=True, slots=True)
class ConvergenceConfig:
    """Structured convergence UI payload."""

    sweep_parameter: str = "cut_off_energy"
    sweep_values: tuple[SweepValue, ...] = ()
    shared_params: dict[str, object] = field(default_factory=dict)
    overrides: dict[frozenset[SweepValue], dict[str, object]] = field(default_factory=dict)
    name_prefix: str = "convergence"

    def __post_init__(self) -> None:
        if self.sweep_parameter not in {"cut_off_energy", "kpoint_mp_grid"}:
            raise ValueError(
                "Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'."
            )


def convergence_config_xc_functional(config: ConvergenceConfig) -> str | None:
    """Return the single XC functional every sweep point in ``config`` uses.

    ``None`` means the sweep has no single answer: the functional is unset, or
    per-point overrides give different points different functionals. Callers
    use this to record which functional a convergence default was established
    with, so an ambiguous sweep must not be reported as any one functional.
    """

    functionals: set[str | None] = set()
    for sweep_value in config.sweep_values:
        value = config.shared_params.get("xc_functional")
        for members, override in config.overrides.items():
            if sweep_value in members and "xc_functional" in override:
                value = override["xc_functional"]
        text = str(value).strip().upper() if value is not None else ""
        functionals.add(text or None)
    if len(functionals) != 1:
        return None
    return next(iter(functionals))


@dataclass(frozen=True, slots=True)
class SweepPointRunState:
    """Runtime job/result information shown alongside each convergence row."""

    status: str = ""
    total_energy: float | None = None
    max_force: float | None = None
    max_stress: float | None = None
    job_id: UUID | None = None
    job_label: str | None = None
