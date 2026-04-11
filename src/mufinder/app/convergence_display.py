"""Shared formatting helpers for convergence UI surfaces."""

from __future__ import annotations


def format_convergence_method(method: str | None) -> str:
    """Return a user-facing convergence method label."""

    if method is None:
        return "Not set"

    normalized = method.strip().lower()
    labels = {
        "delta-threshold": "Delta-Threshold",
        "derivative": "Derivative (dE/dEcut)",
        "asymptotic": "Asymptotic Extrapolation",
    }
    return labels.get(normalized, method)


def format_convergence_criteria(
    *,
    energy_threshold: float | None,
    force_threshold: float | None,
) -> str:
    """Return formatted convergence criteria text."""

    if energy_threshold is None and force_threshold is None:
        return "Not set"

    energy_text = (
        f"dE <= {float(energy_threshold):.6g} eV/atom"
        if energy_threshold is not None
        else "dE <= n/a"
    )
    force_text = (
        f"dF <= {float(force_threshold):.6g} eV/Ang"
        if force_threshold is not None
        else "dF <= n/a"
    )
    return f"{energy_text}, {force_text}"


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
