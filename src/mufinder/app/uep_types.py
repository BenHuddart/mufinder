"""Types used by the UEP workflow UI, adapter, and analysis wizard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias
from uuid import UUID


SupercellMatrix: TypeAlias = tuple[
    tuple[int, int, int],
    tuple[int, int, int],
    tuple[int, int, int],
]


@dataclass(frozen=True, slots=True)
class UepSourceRow:
    """One user-configured CASTEP source job for UEP analysis."""

    label: str
    supercell_matrix: SupercellMatrix | None = None
    parameter_values: dict[str, object] = field(default_factory=dict)
    added_advanced: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class UepConfig:
    """Structured persisted UEP workflow configuration."""

    name_prefix: str = "uep"
    rows: tuple[UepSourceRow, ...] = ()


@dataclass(frozen=True, slots=True)
class UepRunState:
    """Runtime state shown for one configured UEP source row."""

    status: str = ""
    total_energy: float | None = None
    potential_grid_available: bool | None = None
    job_id: UUID | None = None
    job_label: str | None = None
    error_message: str | None = None
    density_available: bool | None = None
