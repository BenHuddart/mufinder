"""Cutoff-convergence analysis helpers for completed result series."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    """Structured result for one convergence check."""

    converged: bool
    method: str
    details: str
    converged_value: float | None = None
    deltas: tuple[tuple[float, float], ...] | None = None
    fitted_curve: tuple[tuple[float, float], ...] | None = None
    extrapolated_energy: float | None = None
    fallback_suggestion: str | None = None


def _clean_series(cutoffs: Sequence[float], values: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    pairs: list[tuple[float, float]] = []
    for cutoff, value in zip(cutoffs, values, strict=False):
        try:
            cutoff_value = float(cutoff)
            observed = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(cutoff_value) or not math.isfinite(observed):
            continue
        pairs.append((cutoff_value, observed))

    pairs.sort(key=lambda pair: pair[0])

    deduped: dict[float, float] = {}
    for cutoff, observed in pairs:
        deduped[cutoff] = observed

    cleaned_pairs = sorted(deduped.items(), key=lambda pair: pair[0])
    return tuple(pair[0] for pair in cleaned_pairs), tuple(pair[1] for pair in cleaned_pairs)


def delta_threshold_energy(
    cutoffs: Sequence[float],
    energies: Sequence[float],
    *,
    num_atoms: int,
    threshold: float,
) -> ConvergenceResult:
    """Check convergence via energy delta between successive cutoffs."""
    x, y = _clean_series(cutoffs, energies)
    if len(x) < 2:
        return ConvergenceResult(
            converged=False,
            method="delta-threshold",
            details="Need at least two completed energy points for Delta-Threshold analysis.",
            fallback_suggestion="Run more cutoff calculations and retry Delta-Threshold.",
        )

    if num_atoms <= 0:
        return ConvergenceResult(
            converged=False,
            method="delta-threshold",
            details="Host atom count must be positive for per-atom energy normalization.",
        )

    deltas: list[tuple[float, float]] = []
    converged_value: float | None = None
    converged_delta: float | None = None
    for index in range(len(x) - 1):
        delta_per_atom = abs(y[index + 1] - y[index]) / float(num_atoms)
        deltas.append((x[index + 1], delta_per_atom))
        if converged_value is None and delta_per_atom <= threshold:
            converged_value = x[index + 1]
            converged_delta = delta_per_atom

    if converged_value is None:
        return ConvergenceResult(
            converged=False,
            method="delta-threshold",
            details=(
                "Energy did not reach the requested per-atom delta threshold "
                f"({threshold:.6g} eV/atom)."
            ),
            deltas=tuple(deltas),
        )

    return ConvergenceResult(
        converged=True,
        method="delta-threshold",
        converged_value=converged_value,
        details=(
            "Energy converged by Delta-Threshold at "
            f"{converged_value:.8g} eV (delta={float(converged_delta):.6g} eV/atom)."
        ),
        deltas=tuple(deltas),
    )


def derivative_energy(
    cutoffs: Sequence[float],
    energies: Sequence[float],
    *,
    num_atoms: int,
    threshold: float,
) -> ConvergenceResult:
    """Check convergence via absolute numerical derivative dE/d(cutoff)."""
    x, y = _clean_series(cutoffs, energies)
    if len(x) < 2:
        return ConvergenceResult(
            converged=False,
            method="derivative",
            details="Need at least two completed energy points for derivative analysis.",
            fallback_suggestion="Run more cutoff calculations and retry derivative analysis.",
        )

    if num_atoms <= 0:
        return ConvergenceResult(
            converged=False,
            method="derivative",
            details="Host atom count must be positive for per-atom normalization.",
        )

    steps = [x[index + 1] - x[index] for index in range(len(x) - 1) if (x[index + 1] - x[index]) > 0.0]
    if not steps:
        return ConvergenceResult(
            converged=False,
            method="derivative",
            details="Cutoff values must increase to compute dE/d(cutoff).",
            fallback_suggestion="Use Delta-Threshold when cutoff spacing is irregular or repeated.",
        )

    mean_step = float(sum(steps) / len(steps))
    derivative_threshold = threshold / mean_step

    rates: list[tuple[float, float]] = []
    converged_value: float | None = None
    converged_rate: float | None = None
    for index in range(len(x) - 1):
        step = x[index + 1] - x[index]
        if step <= 0.0:
            continue
        rate = abs((y[index + 1] - y[index]) / step) / float(num_atoms)
        rates.append((x[index + 1], rate))
        if converged_value is None and rate <= derivative_threshold:
            converged_value = x[index + 1]
            converged_rate = rate

    if converged_value is None:
        return ConvergenceResult(
            converged=False,
            method="derivative",
            details=(
                "Energy derivative did not reach the effective threshold "
                f"{derivative_threshold:.6g} (eV/atom)/eV."
            ),
            deltas=tuple(rates),
            fallback_suggestion="Try Delta-Threshold for a more conservative convergence decision.",
        )

    return ConvergenceResult(
        converged=True,
        method="derivative",
        converged_value=converged_value,
        details=(
            "Energy converged by derivative at "
            f"{converged_value:.8g} eV (|dE/dEcut|={float(converged_rate):.6g} (eV/atom)/eV, "
            f"effective threshold={derivative_threshold:.6g})."
        ),
        deltas=tuple(rates),
    )


def _exponential_model(cutoff: np.ndarray, e_inf: float, a_param: float, b_param: float) -> np.ndarray:
    return e_inf + a_param * np.exp(-b_param * cutoff)


def asymptotic_extrapolation_energy(
    cutoffs: Sequence[float],
    energies: Sequence[float],
    *,
    num_atoms: int,
    threshold: float,
) -> ConvergenceResult:
    """Fit an exponential decay model to estimate infinite-cutoff energy."""
    x, y = _clean_series(cutoffs, energies)
    if len(x) < 4:
        return ConvergenceResult(
            converged=False,
            method="asymptotic",
            details="Asymptotic extrapolation needs at least four completed energy points.",
            fallback_suggestion="Use Delta-Threshold until more cutoff points are available.",
        )

    if num_atoms <= 0:
        return ConvergenceResult(
            converged=False,
            method="asymptotic",
            details="Host atom count must be positive for per-atom normalization.",
        )

    try:
        from scipy.optimize import OptimizeWarning, curve_fit
    except ImportError:
        return ConvergenceResult(
            converged=False,
            method="asymptotic",
            details="SciPy is required for asymptotic extrapolation but is not available.",
            fallback_suggestion="Install SciPy or use Delta-Threshold.",
        )

    x_data = np.asarray(x, dtype=float)
    y_data = np.asarray(y, dtype=float)
    e_inf0 = float(np.min(y_data))
    a0 = float(y_data[0] - e_inf0)
    span = max(float(np.max(x_data) - np.min(x_data)), 1.0)
    b0 = 2.0 / span

    with np.errstate(all="ignore"):
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                params, _cov = curve_fit(
                    _exponential_model,
                    x_data,
                    y_data,
                    p0=(e_inf0, a0, b0),
                    bounds=([-np.inf, -np.inf, 1e-12], [np.inf, np.inf, np.inf]),
                    maxfev=20000,
                )
        except (RuntimeError, ValueError, OptimizeWarning):
            return ConvergenceResult(
                converged=False,
                method="asymptotic",
                details="Asymptotic fit failed to converge for this dataset.",
                fallback_suggestion="Use Delta-Threshold for a robust recommendation.",
            )

    e_inf = float(params[0])
    predicted = _exponential_model(x_data, *params)
    residuals = y_data - predicted
    rmse = float(np.sqrt(np.mean(np.square(residuals)))) / float(num_atoms)
    variance = float(np.var(y_data))
    r_squared = 1.0 if variance <= 1e-16 else 1.0 - float(np.var(residuals) / variance)

    if (not math.isfinite(r_squared)) or r_squared < 0.80 or rmse > max(threshold * 5.0, 1e-6):
        return ConvergenceResult(
            converged=False,
            method="asymptotic",
            details=(
                "Asymptotic fit quality is insufficient "
                f"(R^2={r_squared:.3f}, RMSE={rmse:.6g} eV/atom)."
            ),
            fallback_suggestion="Use Delta-Threshold for this dataset.",
        )

    gap_by_cutoff: list[tuple[float, float]] = []
    converged_value: float | None = None
    for cutoff in x_data:
        energy_at_cutoff = float(_exponential_model(np.asarray([cutoff]), *params)[0])
        gap = abs(energy_at_cutoff - e_inf) / float(num_atoms)
        gap_by_cutoff.append((float(cutoff), gap))
        if converged_value is None and gap <= threshold:
            converged_value = float(cutoff)

    fitted_x = np.linspace(float(np.min(x_data)), float(np.max(x_data)), 200)
    fitted_y = _exponential_model(fitted_x, *params)
    fitted_curve = tuple((float(xi), float(yi)) for xi, yi in zip(fitted_x, fitted_y, strict=False))

    if converged_value is None:
        return ConvergenceResult(
            converged=False,
            method="asymptotic",
            details=(
                "Asymptotic model did not reach the requested energy tolerance "
                f"({threshold:.6g} eV/atom) over sampled cutoffs."
            ),
            deltas=tuple(gap_by_cutoff),
            fitted_curve=fitted_curve,
            extrapolated_energy=e_inf,
            fallback_suggestion="Use Delta-Threshold or extend the cutoff range.",
        )

    return ConvergenceResult(
        converged=True,
        method="asymptotic",
        converged_value=converged_value,
        details=(
            "Energy converged by asymptotic extrapolation at "
            f"{converged_value:.8g} eV (E_inf={e_inf:.12g} eV, RMSE={rmse:.6g} eV/atom)."
        ),
        deltas=tuple(gap_by_cutoff),
        fitted_curve=fitted_curve,
        extrapolated_energy=e_inf,
    )


def delta_threshold_force(
    cutoffs: Sequence[float],
    max_forces: Sequence[float],
    *,
    threshold: float,
) -> ConvergenceResult:
    """Check force convergence from successive absolute force deltas."""
    x, y = _clean_series(cutoffs, max_forces)
    if len(x) < 2:
        return ConvergenceResult(
            converged=False,
            method="force-delta-threshold",
            details="Need at least two completed max-force points for force convergence analysis.",
            fallback_suggestion="Run more cutoff calculations and retry force analysis.",
        )

    deltas: list[tuple[float, float]] = []
    converged_value: float | None = None
    converged_delta: float | None = None
    for index in range(len(x) - 1):
        delta = abs(y[index + 1] - y[index])
        deltas.append((x[index + 1], delta))
        if converged_value is None and delta <= threshold:
            converged_value = x[index + 1]
            converged_delta = delta

    if converged_value is None:
        return ConvergenceResult(
            converged=False,
            method="force-delta-threshold",
            details=(
                "Force variation did not reach the threshold "
                f"({threshold:.6g} eV/Ang)."
            ),
            deltas=tuple(deltas),
        )

    return ConvergenceResult(
        converged=True,
        method="force-delta-threshold",
        converged_value=converged_value,
        details=(
            "Force converged by Delta-Threshold at "
            f"{converged_value:.8g} eV (delta={float(converged_delta):.6g} eV/Ang)."
        ),
        deltas=tuple(deltas),
    )


def delta_threshold_stress(
    cutoffs: Sequence[float],
    max_stresses: Sequence[float],
    *,
    threshold: float,
) -> ConvergenceResult:
    """Check stress convergence from successive absolute max-stress deltas.

    Mirrors :func:`delta_threshold_force`. The threshold and delta are expressed
    in GPa, matching CASTEP's ``GEOM_STRESS_TOL`` convention.
    """
    x, y = _clean_series(cutoffs, max_stresses)
    if len(x) < 2:
        return ConvergenceResult(
            converged=False,
            method="stress-delta-threshold",
            details="Need at least two completed max-stress points for stress convergence analysis.",
            fallback_suggestion="Run more sweep points with CALCULATE_STRESS=true and retry.",
        )

    deltas: list[tuple[float, float]] = []
    converged_value: float | None = None
    converged_delta: float | None = None
    for index in range(len(x) - 1):
        delta = abs(y[index + 1] - y[index])
        deltas.append((x[index + 1], delta))
        if converged_value is None and delta <= threshold:
            converged_value = x[index + 1]
            converged_delta = delta

    if converged_value is None:
        return ConvergenceResult(
            converged=False,
            method="stress-delta-threshold",
            details=(
                "Stress variation did not reach the threshold "
                f"({threshold:.6g} GPa)."
            ),
            deltas=tuple(deltas),
        )

    return ConvergenceResult(
        converged=True,
        method="stress-delta-threshold",
        converged_value=converged_value,
        details=(
            "Stress converged by Delta-Threshold at "
            f"{converged_value:.8g} eV (delta={float(converged_delta):.6g} GPa)."
        ),
        deltas=tuple(deltas),
    )


def _spacing_annotation(
    grid_label: str,
    converged_value: float,
    point_spacings: Mapping[float, float] | None,
) -> str:
    """Build a ``"spacing X 1/Ang"`` fragment, or ``""`` if not applicable.

    The (1,1,1) "Gamma-only" grid has no well-defined k-point spacing — every
    spacing above the cell's coarsest reciprocal transition produces it — so
    we suppress the annotation rather than emit a value that could not be
    used to reproduce the grid in CASTEP.
    """
    if point_spacings is None:
        return ""
    spacing = point_spacings.get(float(converged_value))
    if spacing is None:
        return ""
    # Strip the "Grid: " prefix and parse the n_i tuple. A grid where every
    # n_i is 1 represents Gamma-only sampling; skip the spacing annotation.
    grid_text = grid_label.split(":", maxsplit=1)[-1].strip().replace(" ", "")
    parts = grid_text.split("x")
    if len(parts) == 3:
        try:
            ints = tuple(int(p) for p in parts)
        except ValueError:
            ints = None
        if ints == (1, 1, 1):
            return ""
    return f"spacing {spacing:.4g} 1/Ang"


def _apply_kpoint_label(
    result: ConvergenceResult,
    point_labels: Mapping[float, str],
    *,
    quantity: str,
    delta_unit: str,
    point_spacings: Mapping[float, float] | None = None,
) -> ConvergenceResult:
    """Rewrite the details string of a converged result to use a grid label.

    Called only for k-point sweeps where the raw converged_value is a k-point
    count (e.g. 27) that should be displayed as a grid string (e.g. "3 x 3 x 3")
    rather than a bare number followed by "eV". When ``point_spacings`` is
    supplied, the corresponding k-point spacing is appended in parentheses.
    """
    if not result.converged or result.converged_value is None:
        return result
    grid_label = point_labels.get(float(result.converged_value))
    if grid_label is None:
        return result
    converged_delta = next(
        (d for v, d in (result.deltas or ()) if v == result.converged_value),
        None,
    )
    method_label = result.method.replace("-", " ").title() if result.method else "Delta-Threshold"
    details = f"{quantity} converged by {method_label} at {grid_label}"

    annotations: list[str] = []
    spacing_text = _spacing_annotation(grid_label, result.converged_value, point_spacings)
    if spacing_text:
        annotations.append(spacing_text)
    if converged_delta is not None:
        annotations.append(f"delta={converged_delta:.6g} {delta_unit}")
    if annotations:
        details += f" ({', '.join(annotations)})"
    details += "."
    return replace(result, details=details)


def apply_kpoint_label_to_stress_result(
    result: ConvergenceResult,
    point_labels: Mapping[float, str],
    *,
    point_spacings: Mapping[float, float] | None = None,
) -> ConvergenceResult:
    """Rewrite a stress :class:`ConvergenceResult` to use a k-point grid label."""

    return _apply_kpoint_label(
        result,
        point_labels,
        quantity="Stress",
        delta_unit="GPa",
        point_spacings=point_spacings,
    )


def analyze_kpoint_convergence(
    *,
    kpoint_counts: Sequence[float],
    energies: Sequence[float],
    max_forces: Sequence[float],
    num_atoms: int,
    energy_threshold: float,
    force_threshold: float,
    point_labels: Mapping[float, str] | None = None,
    point_spacings: Mapping[float, float] | None = None,
) -> tuple[ConvergenceResult, ConvergenceResult]:
    """Run k-point convergence using delta-threshold only over k-point counts.

    When *point_labels* is supplied (mapping each k-point count to its grid
    string, e.g. ``{27.0: "3 x 3 x 3"}``), the ``details`` fields of the
    returned results are rewritten to reference the grid label instead of the
    raw numeric count. When *point_spacings* is also supplied, the
    corresponding spacing (in 1/Ang) is appended to the details string.
    """

    energy = delta_threshold_energy(
        kpoint_counts,
        energies,
        num_atoms=num_atoms,
        threshold=energy_threshold,
    )
    force = delta_threshold_force(kpoint_counts, max_forces, threshold=force_threshold)

    if point_labels is not None:
        energy = _apply_kpoint_label(
            energy,
            point_labels,
            quantity="Energy",
            delta_unit="eV/atom",
            point_spacings=point_spacings,
        )
        force = _apply_kpoint_label(
            force,
            point_labels,
            quantity="Force",
            delta_unit="eV/Ang",
            point_spacings=point_spacings,
        )

    return energy, force


def analyze_convergence(
    *,
    method: str,
    cutoffs: Sequence[float],
    energies: Sequence[float],
    max_forces: Sequence[float],
    num_atoms: int,
    energy_threshold: float,
    force_threshold: float,
) -> tuple[ConvergenceResult, ConvergenceResult]:
    """Run selected energy convergence method and force delta-threshold method."""
    method_key = method.strip().lower()
    if method_key == "delta-threshold":
        energy = delta_threshold_energy(cutoffs, energies, num_atoms=num_atoms, threshold=energy_threshold)
    elif method_key == "derivative":
        energy = derivative_energy(cutoffs, energies, num_atoms=num_atoms, threshold=energy_threshold)
    elif method_key == "asymptotic":
        energy = asymptotic_extrapolation_energy(cutoffs, energies, num_atoms=num_atoms, threshold=energy_threshold)
    else:
        raise ValueError(f"Unsupported energy convergence method: {method}")

    force = delta_threshold_force(cutoffs, max_forces, threshold=force_threshold)
    return energy, force


def recommended_value(
    energy_result: ConvergenceResult,
    force_result: ConvergenceResult,
    stress_result: ConvergenceResult | None = None,
) -> float | None:
    """Return the combined recommendation from energy, force and (optionally) stress.

    Every supplied criterion must have converged: the recommendation is the
    largest of their converged values, so the adopted setting satisfies all of
    them. When any supplied criterion did not converge within the sweep there is
    no recommendation (see :func:`unconverged_criteria` for which). A stress
    result is a criterion only when supplied; ``None`` means stress is not used.
    """
    results: tuple[ConvergenceResult | None, ...] = (energy_result, force_result, stress_result)
    supplied = [result for result in results if result is not None]
    if unconverged_criteria(energy_result, force_result, stress_result):
        return None
    return max(float(result.converged_value) for result in supplied)  # type: ignore[arg-type]


def unconverged_criteria(
    energy_result: ConvergenceResult,
    force_result: ConvergenceResult,
    stress_result: ConvergenceResult | None = None,
) -> tuple[str, ...]:
    """Return the names of the supplied criteria that did not converge, in order."""

    named = (("energy", energy_result), ("force", force_result), ("stress", stress_result))
    return tuple(
        name
        for name, result in named
        if result is not None and not (result.converged and result.converged_value is not None)
    )
