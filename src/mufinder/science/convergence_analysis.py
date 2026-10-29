"""Convergence analysis of completed cutoff and k-point sweeps.

One criterion judges every quantity: a sweep point is converged when it, and
every finer point, lies within the tolerance of the finest point of the sweep
(per atom for the energy). The finest point is the reference, so it is never
certified itself and a finer point must exist.

Because the reference is only the best answer the sweep has, a cutoff sweep's
energies are also fitted with a power law ``E = E_inf + c * Ecut**m`` to check
that the finest cutoff lies within the tolerance of the extrapolated limit. The
fit never changes the recommendation; it only says whether the sweep is long
enough. K-point sweeps get no fit: convergence with sampling is not a smooth
decay.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import math
from typing import Sequence
import warnings

import numpy as np

#: :attr:`SweepLengthCheck.status` values.
SWEEP_LONG_ENOUGH = "long_enough"
SWEEP_TOO_SHORT = "too_short"
SWEEP_NOT_ESTIMATED = "not_estimated"

# Power-law fit of the sweep-length check. The exponent is free between these
# bounds; a fit that ends on either bound is not a power law. A tail is used
# only if its energy never rises between neighbours by more than this fraction
# of the tolerance, and the fit is used only if its RMS residual is within the
# same fraction.
_MIN_FIT_POINTS = 4
_EXPONENT_BOUNDS = (-20.0, -0.5)
_EXPONENT_BOUND_MARGIN = 0.02
_FIT_TOLERANCE_FRACTION = 0.25
_FIT_CURVE_SAMPLES = 100


@dataclass(frozen=True, slots=True)
class ConvergenceResult:
    """The outcome of the reference criterion for one quantity of a sweep.

    ``summary`` and ``explanation`` are the two halves of the result line, for
    example ``"converged at 600 eV"`` and ``"within 0.0008 eV/atom of 800 eV,
    as is every finer cutoff"``; :attr:`details` joins them into a sentence.
    ``deviations`` holds each point's distance from the reference (per atom
    for the energy). A cutoff sweep's energy result also carries the fitted
    curve and limit of its sweep-length check, for plotting.
    """

    converged: bool
    quantity: str
    summary: str
    explanation: str
    converged_value: float | None = None
    reference_value: float | None = None
    deviations: tuple[tuple[float, float], ...] | None = None
    fitted_curve: tuple[tuple[float, float], ...] | None = None
    extrapolated_energy: float | None = None

    @property
    def details(self) -> str:
        """The result as one sentence."""

        if not self.explanation:
            return f"{self.quantity} {self.summary}."
        return f"{self.quantity} {self.summary}: {self.explanation}."


@dataclass(frozen=True, slots=True)
class SweepLengthCheck:
    """Whether a cutoff sweep reaches the tolerance of its extrapolated limit.

    ``status`` is :data:`SWEEP_LONG_ENOUGH`, :data:`SWEEP_TOO_SHORT` or
    :data:`SWEEP_NOT_ESTIMATED`. ``finest_gap`` is the fitted distance of the
    finest cutoff from the limit in eV/atom, ``required_cutoff`` the cutoff at
    which that distance equals the tolerance, and ``suggested_cutoff`` (too
    short only) that cutoff rounded up to a multiple of the sweep's last step.
    ``extrapolated_energy`` and ``fitted_curve`` are total energies in eV.
    """

    status: str
    summary: str
    explanation: str
    extrapolated_energy: float | None = None
    exponent: float | None = None
    finest_gap: float | None = None
    required_cutoff: float | None = None
    suggested_cutoff: float | None = None
    fitted_curve: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class ConvergenceAnalysis:
    """Every criterion of one sweep, with the sweep-length check of a cutoff sweep."""

    energy: ConvergenceResult
    force: ConvergenceResult
    stress: ConvergenceResult | None
    sweep_length: SweepLengthCheck | None
    reference_text: str

    @property
    def recommended(self) -> float | None:
        """The largest converged value, or ``None`` unless every criterion converged."""

        return recommended_value(self.energy, self.force, self.stress)

    @property
    def unconverged(self) -> tuple[str, ...]:
        """The criteria that did not converge, in order."""

        return unconverged_criteria(self.energy, self.force, self.stress)


def _amount(value: float) -> str:
    """Format a deviation to two significant figures without an exponent."""

    if value == 0.0:
        return "0"
    return np.format_float_positional(
        float(value), precision=2, unique=True, fractional=False, trim="-"
    )


def _clean_series(
    points: Sequence[float], values: Sequence[float]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Drop non-finite pairs, sort by point and keep the last value of a repeated point."""

    deduped: dict[float, float] = {}
    for point, value in sorted(
        (
            (float(point), float(value))
            for point, value in zip(points, values, strict=False)
            if _is_finite_number(point) and _is_finite_number(value)
        ),
        key=lambda pair: pair[0],
    ):
        deduped[point] = value
    return tuple(deduped), tuple(deduped.values())


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class _SweepWording:
    point_text: Callable[[float], str]
    summary_text: Callable[[float], str]
    finer: str
    reference_suffix: str


def _cutoff_wording() -> _SweepWording:
    def point_text(value: float) -> str:
        return f"{value:.8g} eV"

    return _SweepWording(point_text, point_text, "finer cutoff", "the finest cutoff in the sweep")


def _kpoint_wording(
    point_labels: Mapping[float, str] | None,
    point_spacings: Mapping[float, float] | None,
) -> _SweepWording:
    labels = {float(key): str(value) for key, value in (point_labels or {}).items()}

    def point_text(value: float) -> str:
        return labels.get(float(value), f"{value:.8g} k-points")

    def summary_text(value: float) -> str:
        label = point_text(value)
        spacing = _spacing_annotation(label, value, point_spacings)
        return f"{label} ({spacing})" if spacing else label

    return _SweepWording(point_text, summary_text, "denser grid", "the densest grid in the sweep")


def _spacing_annotation(
    grid_label: str,
    converged_value: float,
    point_spacings: Mapping[float, float] | None,
) -> str:
    """Build a ``"spacing X 1/Ang"`` fragment, or ``""`` if not applicable.

    The ``1 x 1 x 1`` grid has no well-defined k-point spacing (every spacing
    above the cell's coarsest reciprocal transition produces it), so it gets
    no annotation rather than a value that could not reproduce the grid.
    """

    if point_spacings is None:
        return ""
    spacing = point_spacings.get(float(converged_value))
    if spacing is None:
        return ""
    parts = grid_label.split(":", maxsplit=1)[-1].replace(" ", "").split("x")
    if len(parts) == 3:
        try:
            if tuple(int(part) for part in parts) == (1, 1, 1):
                return ""
        except ValueError:
            pass
    return f"spacing {spacing:.4g} 1/Ang"


def _reference_result(
    points: Sequence[float],
    values: Sequence[float],
    *,
    quantity: str,
    unit: str,
    threshold: float,
    scale: float,
    wording: _SweepWording,
) -> ConvergenceResult:
    """Apply the reference criterion to one quantity."""

    x, y = _clean_series(points, values)
    if len(x) < 2:
        return ConvergenceResult(
            converged=False,
            quantity=quantity,
            summary="not converged within the sweep",
            explanation="needs at least two completed points",
        )

    reference = y[-1]
    deviations = tuple((point, abs(value - reference) / scale) for point, value in zip(x, y))
    converged_index: int | None = None
    for index in range(len(x) - 2, -1, -1):
        if deviations[index][1] > threshold:
            break
        converged_index = index

    reference_text = wording.point_text(x[-1])
    if converged_index is None:
        point, deviation = deviations[-2]
        return ConvergenceResult(
            converged=False,
            quantity=quantity,
            summary="not converged within the sweep",
            explanation=(
                f"{wording.point_text(point)} is {_amount(deviation)} {unit} from {reference_text}"
            ),
            reference_value=x[-1],
            deviations=deviations,
        )

    point, deviation = deviations[converged_index]
    return ConvergenceResult(
        converged=True,
        quantity=quantity,
        summary=f"converged at {wording.summary_text(point)}",
        explanation=(
            f"within {_amount(deviation)} {unit} of {reference_text}, as is every {wording.finer}"
        ),
        converged_value=point,
        reference_value=x[-1],
        deviations=deviations,
    )


def _power_law(x: np.ndarray, limit: float, scale: float, exponent: float) -> np.ndarray:
    return limit + scale * np.power(x, exponent)


def _fit_tail(
    x: np.ndarray, y: np.ndarray, threshold: float
) -> tuple[tuple[float, float, float] | None, str]:
    """Fit one tail (``x`` scaled to the finest cutoff, ``y`` in eV/atom from it).

    Returns the parameters ``(E_inf, c, m)`` in those units, or ``None`` and the
    reason the tail cannot be used.
    """

    slack = threshold * _FIT_TOLERANCE_FRACTION
    if float(np.max(np.diff(y))) > slack:
        return None, "the energy does not fall steadily towards the finest cutoff"

    from scipy.optimize import OptimizeWarning, curve_fit

    lowest = float(np.min(y))
    lower = [-np.inf, 0.0, _EXPONENT_BOUNDS[0]]
    upper = [lowest, np.inf, _EXPONENT_BOUNDS[1]]
    start = (lowest - 1e-5, max(float(y[0] - y[-1]), 1e-9), -2.0)
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("error", OptimizeWarning)
            params, _covariance = curve_fit(
                _power_law, x, y, p0=start, bounds=(lower, upper), maxfev=20000
            )
    except (RuntimeError, ValueError, OptimizeWarning):
        return None, "the power-law fit did not converge"

    limit, scale, exponent = (float(value) for value in params)
    low, high = _EXPONENT_BOUNDS
    if exponent <= low * (1.0 - _EXPONENT_BOUND_MARGIN) or exponent >= high * (
        1.0 + _EXPONENT_BOUND_MARGIN
    ):
        return None, "the decay is not a power law"
    residual = float(np.sqrt(np.mean(np.square(y - _power_law(x, limit, scale, exponent)))))
    if not math.isfinite(residual) or residual > slack:
        return None, "the power law does not follow the points closely enough"
    return (limit, scale, exponent), ""


def check_sweep_length(
    cutoffs: Sequence[float],
    energies: Sequence[float],
    *,
    num_atoms: int,
    threshold: float,
) -> SweepLengthCheck:
    """Check that the finest cutoff is within ``threshold`` (eV/atom) of the fitted limit.

    ``E = E_inf + c * Ecut**m`` (``c >= 0``, ``E_inf`` no higher than the lowest
    energy, ``-20 <= m <= -0.5``) is fitted to the longest tail of four or more
    cutoffs, dropping the coarsest first, whose energy never rises by more than
    a quarter of the tolerance per atom between neighbours. The fit is used only
    if its exponent is off both bounds and its RMS residual is within a quarter
    of the tolerance per atom. The sweep is too short when the fitted distance
    of the finest cutoff from ``E_inf`` exceeds the tolerance.
    """

    if num_atoms <= 0:
        raise ValueError("The simulation-cell atom count must be positive.")
    x_all, e_all = _clean_series(cutoffs, energies)
    if len(x_all) < _MIN_FIT_POINTS:
        return SweepLengthCheck(
            status=SWEEP_NOT_ESTIMATED,
            summary="limit not estimated",
            explanation="the fit needs at least four cutoffs",
        )

    x = np.asarray(x_all, dtype=float)
    finest_energy = float(e_all[-1])
    y = (np.asarray(e_all, dtype=float) - finest_energy) / float(num_atoms)
    finest = float(x[-1])
    reason = ""
    for size in range(len(x), _MIN_FIT_POINTS - 1, -1):
        params, reason = _fit_tail(x[-size:] / finest, y[-size:], threshold)
        if params is not None:
            return _sweep_check_from_fit(x[-size:], params, finest_energy, num_atoms, threshold)
    return SweepLengthCheck(
        status=SWEEP_NOT_ESTIMATED, summary="limit not estimated", explanation=reason
    )


def _sweep_check_from_fit(
    x: np.ndarray,
    params: tuple[float, float, float],
    finest_energy: float,
    num_atoms: int,
    threshold: float,
) -> SweepLengthCheck:
    limit, gap, exponent = params
    finest = float(x[-1])
    samples = np.linspace(float(x[0]), finest, _FIT_CURVE_SAMPLES)
    curve = finest_energy + float(num_atoms) * _power_law(samples / finest, *params)
    fitted_curve = tuple((float(a), float(b)) for a, b in zip(samples, curve))
    extrapolated = finest_energy + float(num_atoms) * limit
    required = finest * (threshold / gap) ** (1.0 / exponent) if gap > 0.0 else None
    common = {
        "extrapolated_energy": extrapolated,
        "exponent": exponent,
        "finest_gap": gap,
        "required_cutoff": required,
        "fitted_curve": fitted_curve,
    }
    finest_text = f"{finest:.8g} eV"
    if gap <= threshold or required is None:
        return SweepLengthCheck(
            status=SWEEP_LONG_ENOUGH,
            summary="long enough",
            explanation=f"{finest_text} is within {_amount(gap)} eV/atom of the fitted limit",
            **common,
        )
    step = finest - float(x[-2])
    # Round to a millionth of a step first: the fitted ``required`` carries
    # least-squares noise, and an exact multiple must not gain a whole step.
    suggested = math.ceil(round(required / step, 6)) * step
    return SweepLengthCheck(
        status=SWEEP_TOO_SHORT,
        summary="too short",
        explanation=(
            f"{finest_text} is {_amount(gap)} eV/atom from the fitted limit; "
            f"extend the sweep to {suggested:.8g} eV and analyse again"
        ),
        suggested_cutoff=suggested,
        **common,
    )


def _analyze(
    *,
    points: Sequence[float],
    energies: Sequence[float],
    max_forces: Sequence[float],
    max_stresses: Sequence[float] | None,
    num_atoms: int,
    energy_threshold: float,
    force_threshold: float,
    stress_threshold: float | None,
    wording: _SweepWording,
) -> tuple[ConvergenceResult, ConvergenceResult, ConvergenceResult | None, str]:
    if num_atoms <= 0:
        raise ValueError("The simulation-cell atom count must be positive.")
    energy = _reference_result(
        points,
        energies,
        quantity="Energy",
        unit="eV/atom",
        threshold=energy_threshold,
        scale=float(num_atoms),
        wording=wording,
    )
    force = _reference_result(
        points,
        max_forces,
        quantity="Force",
        unit="eV/Ang",
        threshold=force_threshold,
        scale=1.0,
        wording=wording,
    )
    stress = None
    if max_stresses is not None and stress_threshold is not None:
        stress = _reference_result(
            points,
            max_stresses,
            quantity="Stress",
            unit="GPa",
            threshold=stress_threshold,
            scale=1.0,
            wording=wording,
        )
    reference_text = (
        f"{wording.point_text(energy.reference_value)}, {wording.reference_suffix}"
        if energy.reference_value is not None
        else ""
    )
    return energy, force, stress, reference_text


def analyze_convergence(
    *,
    cutoffs: Sequence[float],
    energies: Sequence[float],
    max_forces: Sequence[float],
    num_atoms: int,
    energy_threshold: float,
    force_threshold: float,
    max_stresses: Sequence[float] | None = None,
    stress_threshold: float | None = None,
) -> ConvergenceAnalysis:
    """Judge a cutoff sweep against its finest cutoff and check the sweep's length.

    Energies are total energies in eV, compared per atom against
    ``energy_threshold`` (eV/atom); maximum forces (eV/Ang) and, when both are
    supplied, maximum stresses (GPa) are compared directly. Raises
    :class:`ValueError` for a non-positive atom count.
    """

    energy, force, stress, reference_text = _analyze(
        points=cutoffs,
        energies=energies,
        max_forces=max_forces,
        max_stresses=max_stresses,
        num_atoms=num_atoms,
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
        stress_threshold=stress_threshold,
        wording=_cutoff_wording(),
    )
    sweep_length = check_sweep_length(
        cutoffs, energies, num_atoms=num_atoms, threshold=energy_threshold
    )
    if sweep_length.fitted_curve is not None:
        energy = _with_fit(energy, sweep_length)
    return ConvergenceAnalysis(energy, force, stress, sweep_length, reference_text)


def _with_fit(result: ConvergenceResult, check: SweepLengthCheck) -> ConvergenceResult:
    return replace(
        result,
        fitted_curve=check.fitted_curve,
        extrapolated_energy=check.extrapolated_energy,
    )


def analyze_kpoint_convergence(
    *,
    kpoint_counts: Sequence[float],
    energies: Sequence[float],
    max_forces: Sequence[float],
    num_atoms: int,
    energy_threshold: float,
    force_threshold: float,
    max_stresses: Sequence[float] | None = None,
    stress_threshold: float | None = None,
    point_labels: Mapping[float, str] | None = None,
    point_spacings: Mapping[float, float] | None = None,
) -> ConvergenceAnalysis:
    """Judge a k-point sweep, ordered by k-point count, against its densest grid.

    ``point_labels`` maps each count to its grid (``{27.0: "3 x 3 x 3"}``) for
    the wording, and ``point_spacings`` to the spacing that reproduces it,
    which is named beside a converged grid other than ``1 x 1 x 1``. There is
    no sweep-length check.
    """

    energy, force, stress, reference_text = _analyze(
        points=kpoint_counts,
        energies=energies,
        max_forces=max_forces,
        max_stresses=max_stresses,
        num_atoms=num_atoms,
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
        stress_threshold=stress_threshold,
        wording=_kpoint_wording(point_labels, point_spacings),
    )
    return ConvergenceAnalysis(energy, force, stress, None, reference_text)


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
