"""Types used by the convergence configuration UI and adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias
from uuid import UUID


KPointGrid: TypeAlias = tuple[int, int, int]
SweepValue: TypeAlias = float | KPointGrid


@dataclass(frozen=True, slots=True)
class ConvergenceConfig:
    """Structured convergence UI payload."""

    sweep_parameter: str = "cut_off_energy"
    sweep_values: tuple[SweepValue, ...] = ()
    # Backward-compatibility field for existing cutoff code paths.
    cutoff_energies: tuple[float, ...] = ()
    shared_params: dict[str, object] = field(default_factory=dict)
    overrides: dict[frozenset[SweepValue], dict[str, object]] = field(default_factory=dict)
    name_prefix: str = "convergence"

    def __post_init__(self) -> None:
        if self.sweep_parameter not in {"cut_off_energy", "kpoint_mp_grid"}:
            raise ValueError(
                "Convergence sweep_parameter must be 'cut_off_energy' or 'kpoint_mp_grid'."
            )

        if self.sweep_values and self.cutoff_energies and self.sweep_parameter == "cut_off_energy":
            cutoff_values = tuple(float(value) for value in self.sweep_values)
            if cutoff_values != tuple(float(value) for value in self.cutoff_energies):
                raise ValueError(
                    "When both sweep_values and cutoff_energies are provided they must match."
                )

        if not self.sweep_values and self.sweep_parameter == "cut_off_energy" and self.cutoff_energies:
            object.__setattr__(
                self,
                "sweep_values",
                tuple(float(value) for value in self.cutoff_energies),
            )

        if not self.cutoff_energies and self.sweep_parameter == "cut_off_energy":
            object.__setattr__(
                self,
                "cutoff_energies",
                tuple(float(value) for value in self.sweep_values),
            )


@dataclass(frozen=True, slots=True)
class SweepPointRunState:
    """Runtime job/result information shown alongside each convergence row."""

    status: str = ""
    total_energy: float | None = None
    max_force: float | None = None
    job_id: UUID | None = None
    job_label: str | None = None


# Backward-compatible alias used by existing cutoff tests and call sites.
CutoffRunState = SweepPointRunState
