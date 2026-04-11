"""Types used by the pristine-relaxation workflow UI and adapter."""

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
class PristineRelaxationRow:
    """One user-configured pristine-relaxation job row."""

    label: str
    fix_all_cell: bool = False
    supercell_matrix: SupercellMatrix | None = None
    parameter_values: dict[str, object] = field(default_factory=dict)
    added_advanced: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PristineRelaxationConfig:
    """Structured pristine-relaxation configuration payload."""

    name_prefix: str = "pristine_relax"
    rows: tuple[PristineRelaxationRow, ...] = ()


@dataclass(frozen=True, slots=True)
class RelaxationRunState:
    """Runtime status/result information for one relaxation row."""

    status: str = ""
    total_energy: float | None = None
    max_force: float | None = None
    lattice_a: float | None = None
    lattice_b: float | None = None
    lattice_c: float | None = None
    job_id: UUID | None = None
    job_label: str | None = None
    geom_converged: bool | None = None
    error_message: str | None = None
