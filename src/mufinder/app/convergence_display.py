"""Shared formatting helpers for convergence UI surfaces."""

from __future__ import annotations

from mufinder.domain.models import MANUAL_DEFAULT_METHOD


def format_convergence_criteria(
    *,
    energy_threshold: float | None,
    force_threshold: float | None,
    stress_threshold: float | None = None,
) -> str:
    """Return formatted convergence criteria text.

    The ``stress_threshold`` argument is optional. When supplied it is appended
    as a ``dS <= X GPa`` clause, matching the convention used by CASTEP's
    ``GEOM_STRESS_TOL`` keyword.
    """

    if energy_threshold is None and force_threshold is None and stress_threshold is None:
        return "Not set"

    energy_text = (
        f"dE <= {float(energy_threshold):.6g} eV/atom"
        if energy_threshold is not None
        else "dE <= n/a"
    )
    force_text = (
        f"dF <= {float(force_threshold):.6g} eV/Ang" if force_threshold is not None else "dF <= n/a"
    )
    parts = [energy_text, force_text]
    if stress_threshold is not None:
        parts.append(f"dS <= {float(stress_threshold):.6g} GPa")
    return ", ".join(parts)


def format_default_criteria(
    *,
    method: str | None,
    energy_threshold: float | None,
    force_threshold: float | None,
    stress_threshold: float | None = None,
) -> str:
    """Return the criteria of a project default, or ``Set manually`` for a typed-in one."""

    if method == MANUAL_DEFAULT_METHOD:
        return "Set manually"
    return format_convergence_criteria(
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
        stress_threshold=stress_threshold,
    )


def format_cutoff_energy(cutoff_energy: float | None) -> str:
    """Return a formatted cutoff-energy label."""

    if cutoff_energy is None:
        return "Not set"
    return f"{float(cutoff_energy):.8g} eV"


def format_kpoint_grid(kpoint_grid: tuple[int, int, int] | None) -> str:
    """Return a formatted Monkhorst-Pack grid label."""

    if kpoint_grid is None:
        return "Not set"
    return f"{int(kpoint_grid[0])} x {int(kpoint_grid[1])} x {int(kpoint_grid[2])}"
