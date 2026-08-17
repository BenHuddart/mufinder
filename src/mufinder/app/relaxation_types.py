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
class MuonRelaxationRow:
    """One staged muon structural-relaxation job row.

    ``imported`` marks a row that was ingested from a calculation performed
    outside MuFinder rather than staged and run here. Such a row is read-only:
    its parameters describe a run that has already happened, so editing them
    would misdescribe it, and re-running it would discard the very result the
    import exists to preserve.

    The flag must survive every round trip that touches a row — the widget's
    state collection and restoration, and the configuration serialiser pair.
    Configuring any row re-registers the whole definition from the widget's
    current state, so a flag dropped anywhere in that loop silently unlocks
    every imported row while its entities and files remain.
    """

    site_id: UUID
    label: str
    initial_fractional_position: tuple[float, float, float]
    charge_state: str = "positive_muon"
    supercell_matrix: SupercellMatrix | None = None
    parameter_values: dict[str, object] = field(default_factory=dict)
    added_advanced: tuple[str, ...] = ()
    imported: bool = False
    #: Groups rows ingested by one import, so they can be reported and removed
    #: together. ``None`` on a natively staged row.
    import_batch_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class MuonRelaxationConfig:
    """Structured muon-relaxation configuration payload."""

    rows: tuple[MuonRelaxationRow, ...] = ()


@dataclass(frozen=True, slots=True)
class RelaxationRunState:
    """Runtime status/result information for one relaxation row."""

    status: str = ""
    total_energy: float | None = None
    max_force: float | None = None
    lattice_a: float | None = None
    lattice_b: float | None = None
    lattice_c: float | None = None
    final_fractional_position: tuple[float, float, float] | None = None
    job_id: UUID | None = None
    job_label: str | None = None
    geom_converged: bool | None = None
    error_message: str | None = None
