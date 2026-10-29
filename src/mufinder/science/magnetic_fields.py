"""Magnetic field calculation helpers for MuFinder."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np

from mufinder.domain.constants import (
    BOHR_MAGNETON_J_PER_T,
    METRES_PER_ANGSTROM,
    MU0_OVER_4PI_T_M_PER_A,
    MU0_T_M_PER_A,
    MUON_GAMMA_OVER_2PI_MHZ_T,
)
from mufinder.domain.models import MuonSite, Structure
from mufinder.science.ase_bridge import reduced_formula
from mufinder.science.lattice import nearest_image_distance
from mufinder.science.spglib_compat import (
    SYMMETRY_TOLERANCE_ANG,
    spglib_cell,
    symmetry_operations,
    symmetry_orbit,
)
from mufinder.science.demagnetization import DemagnetizingSample, demagnetizing_tensor
from mufinder.science._muesr_compat import muesr_imports
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    MuesrModelData,
    get_muesr_model_data,
    resolve_atom_selector,
    split_muesr_model_data_into_helices,
)
from mufinder.science.muon_distortion import (
    MuonDistortion,
    equivalent_site_distortion,
    map_muon_distortion,
)

MUON_GYROMAGNETIC_RATIO_MHZ_PER_T = MUON_GAMMA_OVER_2PI_MHZ_T
CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB = (
    (2.0 * MU0_T_M_PER_A / 3.0) * BOHR_MAGNETON_J_PER_T / (METRES_PER_ANGSTROM**3)
)
# muesr returns dipolar tensors as the lattice sum of (3 r̂ r̂ - 1) / r^3 in Å^-3;
# this factor, mu0 mu_B / (4 pi Å^3), turns that sum into T per mu_B.
DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3 = (
    MU0_OVER_4PI_T_M_PER_A * BOHR_MAGNETON_J_PER_T / (METRES_PER_ANGSTROM**3)
)
# mu0 M, in T, of a magnetisation of 1 mu_B per Å^3.
MAGNETIZATION_FIELD_T_PER_MUB_PER_ANGSTROM_MINUS3 = (
    MU0_T_M_PER_A * BOHR_MAGNETON_J_PER_T / (METRES_PER_ANGSTROM**3)
)
CONTACT_COUPLING_UNIT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("angstrom^-3", "Å⁻³"),
    ("T_per_muB", "T/μB"),
    ("MHz_per_muB", "MHz/μB"),
)
_CONTACT_COUPLING_FACTORS_TO_ANGSTROM_MINUS3 = {
    "angstrom^-3": 1.0,
    "a^-3": 1.0,
    "ang^-3": 1.0,
    "T_per_muB": 1.0 / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
    "MHz_per_muB": (
        (1.0 / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T) / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB
    ),
}


class MagneticFieldError(ValueError):
    """Raised when magnetic field calculation inputs are invalid."""


class MagneticFieldDependencyError(ImportError):
    """Raised when the optional muesr backend is unavailable."""


@dataclass(frozen=True, slots=True)
class MagneticFieldOptions:
    """Calculation options for a magnetic field run."""

    supercell: tuple[int, int, int] = (100, 100, 100)
    radius: float | None = None
    include_lorentz: bool = True
    demag_shape: str = "none"
    plate_normal_axis: str = "c"
    demag_sample: DemagnetizingSample | None = None
    # Measured bulk magnetisation in mu_B per formula unit, Cartesian; None takes it from the model.
    measured_magnetization: tuple[float, float, float] | None = None
    nnn: int = 2
    rcont: float = 10.0
    contact_coupling: float | None = None
    applied_field: tuple[float, float, float] = (0.0, 0.0, 0.0)
    find_equivalent_sites: bool = False
    equivalent_tolerance: float = 1e-3
    commensurate_tolerance: float = 1e-8
    commensurate_max_period: int = 64
    nangles: int = 32_768
    phase_sample_count: int = 200_000
    phase_sampling: str = "auto"
    random_seed: int | None = None
    max_grid_samples: int = 1_000_000
    histogram_bins: int = 512
    approximant_tolerance: float = 1e-3
    approximant_max_period: int = 12
    max_approximant_fields: int = 512


#: The closest a muon site may sit to any host atom, in Angstrom. The dipolar and
#: contact sums diverge as the muon approaches an ion, so a site nearer than this is
#: refused rather than calculated.
MIN_MUON_ATOM_DISTANCE_ANG = 0.5


@dataclass(frozen=True, slots=True)
class NearestHostAtom:
    """The host atom closest to a muon site, over every periodic image."""

    index: int
    species: str
    distance_ang: float


def nearest_host_atom(
    structure: Structure, muon_position: Sequence[float]
) -> NearestHostAtom | None:
    """Return the host atom nearest ``muon_position`` (fractional), or ``None`` for an empty cell."""

    nearest: NearestHostAtom | None = None
    for index, (species, position) in enumerate(
        zip(structure.species, structure.fractional_positions, strict=True)
    ):
        distance = nearest_image_distance(muon_position, position, structure.lattice_vectors)
        if nearest is None or distance < nearest.distance_ang:
            nearest = NearestHostAtom(index=index, species=str(species), distance_ang=distance)
    return nearest


def muon_site_clearance_error(
    structure: Structure, muon_position: Sequence[float], *, label: str = "The muon site"
) -> str | None:
    """Return why ``muon_position`` is too close to a host atom, or ``None`` when it is clear.

    A site is refused when it lies within :data:`MIN_MUON_ATOM_DISTANCE_ANG` of any
    periodic image of a host atom; the message names that atom (numbered from 1)
    and its distance.
    """

    nearest = nearest_host_atom(structure, muon_position)
    if nearest is None or nearest.distance_ang >= MIN_MUON_ATOM_DISTANCE_ANG:
        return None
    return (
        f"{label} is {nearest.distance_ang:.2f} Å from {nearest.species} (atom {nearest.index + 1}); "
        f"a muon site must be at least {MIN_MUON_ATOM_DISTANCE_ANG:g} Å from every atom."
    )


# A distribution counts as converged in a setting when halving that setting moves its
# cumulative distribution of |B| by less than this fraction (Kolmogorov-Smirnov distance).
CONVERGENCE_TOLERANCE = 0.01
# Fewer samples than this per occupied histogram bin leave p(|B|) noise-limited.
MIN_SAMPLES_PER_OCCUPIED_BIN = 20.0


def contact_coupling_to_angstrom_minus3(value: float, unit: str) -> float:
    """Convert a contact hyperfine coupling value to muesr's ACont unit."""

    unit_key = str(unit).strip()
    factor = _CONTACT_COUPLING_FACTORS_TO_ANGSTROM_MINUS3.get(unit_key)
    if factor is None:
        raise MagneticFieldError(f"unknown contact coupling unit '{unit}'")
    return float(value) * factor


@dataclass(frozen=True, slots=True)
class MagneticFieldResult:
    """One vector magnetic-field result in Tesla.

    ``calculation_parameters`` records the contact coupling and contact shell
    (``nnn``, ``rcont``) the contact term used; a result saved without them has
    an empty mapping.
    """

    site_label: str
    muon_position: tuple[float, float, float]
    dipolar: tuple[float, float, float]
    lorentz: tuple[float, float, float]
    contact: tuple[float, float, float]
    demag: tuple[float, float, float]
    applied: tuple[float, float, float]
    distortion: tuple[float, float, float] | None = None
    calculation_parameters: dict[str, object] = field(default_factory=dict)

    @property
    def total(self) -> tuple[float, float, float]:
        total = (
            np.asarray(self.dipolar, dtype=float)
            + np.asarray(self.lorentz, dtype=float)
            + np.asarray(self.contact, dtype=float)
            + np.asarray(self.demag, dtype=float)
            + np.asarray(self.applied, dtype=float)
        )
        if self.distortion is not None:
            total = total + np.asarray(self.distortion, dtype=float)
        return _tuple3(total)

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(np.asarray(self.total, dtype=float)))

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "site_label": self.site_label,
            "muon_position": list(self.muon_position),
            "dipolar_T": list(self.dipolar),
            "lorentz_T": list(self.lorentz),
            "contact_T": list(self.contact),
            "demag_T": list(self.demag),
            "applied_T": list(self.applied),
            "total_T": list(self.total),
            "magnitude_T": self.magnitude,
        }
        if self.distortion is not None:
            payload["distortion_T"] = list(self.distortion)
        if self.calculation_parameters:
            payload["calculation_parameters"] = dict(self.calculation_parameters)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> MagneticFieldResult:
        """Rebuild a result from :meth:`to_json` output; ``total_T`` and ``magnitude_T`` are derived."""

        distortion = payload.get("distortion_T")
        raw_parameters = payload.get("calculation_parameters")
        return cls(
            site_label=str(payload.get("site_label", "")),
            muon_position=_tuple3(payload["muon_position"]),
            dipolar=_tuple3(payload["dipolar_T"]),
            lorentz=_tuple3(payload["lorentz_T"]),
            contact=_tuple3(payload["contact_T"]),
            demag=_tuple3(payload["demag_T"]),
            applied=_tuple3(payload["applied_T"]),
            distortion=None if distortion is None else _tuple3(distortion),
            calculation_parameters=(
                dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
            ),
        )


@dataclass(frozen=True, slots=True)
class DistributionConvergenceCheck:
    """How sensitive one sampled ``p(|B|)`` distribution is to one numerical setting.

    ``setting`` names the :class:`MagneticFieldOptions` field (``nangles``,
    ``phase_sample_count`` or ``histogram_bins``). For ``nangles`` and
    ``phase_sample_count``, ``metric`` is the largest difference between the
    cumulative distributions of ``|B|`` with the full setting and with half of it
    (the Kolmogorov-Smirnov distance); for ``histogram_bins`` it is the mean
    number of samples per occupied bin.
    """

    setting: str
    label: str
    converged: bool
    metric: float
    message: str

    def to_json(self) -> dict[str, object]:
        return {
            "setting": self.setting,
            "label": self.label,
            "converged": bool(self.converged),
            "metric": float(self.metric),
            "message": self.message,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> DistributionConvergenceCheck:
        return cls(
            setting=str(payload.get("setting", "")),
            label=str(payload.get("label", "")),
            converged=bool(payload.get("converged", False)),
            metric=float(payload.get("metric", 0.0)),
            message=str(payload.get("message", "")),
        )


# Field terms of a distribution, in the order they are reported.
DISTRIBUTION_FIELD_TERMS = ("dipolar", "lorentz", "contact", "demag", "applied")


@dataclass(frozen=True, slots=True)
class FieldTermStatistics:
    """Mean vector and RMS magnitude, in T, of one field term over the phase samples.

    The RMS is ``sqrt(<|B_term|^2>)``. For a helix the dipolar mean is near zero
    while every sample carries a large dipolar field; the RMS shows that size.
    """

    mean_T: tuple[float, float, float]
    rms_T: float

    def to_json(self) -> dict[str, object]:
        return {"mean_T": list(self.mean_T), "rms_T": float(self.rms_T)}

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> FieldTermStatistics:
        return cls(mean_T=_tuple3(payload["mean_T"]), rms_T=float(payload["rms_T"]))


@dataclass(frozen=True, slots=True)
class MagneticFieldDistributionResult:
    """Field distribution for an incommensurate magnetic structure.

    ``terms`` maps each name in :data:`DISTRIBUTION_FIELD_TERMS` to its
    statistics over the samples, and ``total_rms_T`` is the RMS of ``|B|``.
    ``applied_T`` and ``demag_T`` are the constant applied and demagnetizing
    vectors included in every row of ``fields_T``. A result saved before the
    breakdown existed has empty ``terms`` and ``None`` for the rest.
    """

    site_label: str
    muon_position: tuple[float, float, float]
    fields_T: np.ndarray
    histogram_centres_T: np.ndarray
    histogram_counts: np.ndarray
    calculation_parameters: dict[str, object] = field(default_factory=dict)
    convergence: tuple[DistributionConvergenceCheck, ...] = ()
    terms: Mapping[str, FieldTermStatistics] = field(default_factory=dict)
    total_rms_T: float | None = None
    applied_T: tuple[float, float, float] | None = None
    demag_T: tuple[float, float, float] | None = None

    @property
    def magnitudes_T(self) -> np.ndarray:
        return np.linalg.norm(self.fields_T, axis=1)

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "site_label": self.site_label,
            "muon_position": list(self.muon_position),
            "fields_T": np.asarray(self.fields_T, dtype=float).tolist(),
            "histogram_centres_T": np.asarray(self.histogram_centres_T, dtype=float).tolist(),
            "histogram_counts": np.asarray(self.histogram_counts, dtype=float).tolist(),
            "mean_T": np.mean(self.fields_T, axis=0).tolist(),
            "std_T": np.std(self.fields_T, axis=0).tolist(),
            "calculation_parameters": dict(self.calculation_parameters),
            "convergence": [check.to_json() for check in self.convergence],
        }
        if self.terms:
            payload["terms"] = {name: term.to_json() for name, term in self.terms.items()}
        if self.total_rms_T is not None:
            payload["total_rms_T"] = float(self.total_rms_T)
        if self.applied_T is not None:
            payload["applied_T"] = list(self.applied_T)
        if self.demag_T is not None:
            payload["demag_T"] = list(self.demag_T)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> MagneticFieldDistributionResult:
        """Rebuild a result from :meth:`to_json` output, including one saved without a breakdown."""

        fields = np.asarray(payload.get("fields_T", []), dtype=float)
        if fields.size == 0:
            fields = np.zeros((0, 3), dtype=float)
        elif fields.ndim != 2 or fields.shape[1] != 3:
            raise ValueError(f"fields_T must have shape (n, 3), got {fields.shape}")
        terms = _terms_from_json(payload.get("terms"))
        raw_parameters = payload.get("calculation_parameters")
        raw_checks = payload.get("convergence")
        return cls(
            site_label=str(payload.get("site_label", "")),
            muon_position=_tuple3(payload.get("muon_position", (0.0, 0.0, 0.0))),
            fields_T=fields,
            histogram_centres_T=np.asarray(payload.get("histogram_centres_T", []), dtype=float),
            histogram_counts=np.asarray(payload.get("histogram_counts", []), dtype=float),
            calculation_parameters=(
                dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
            ),
            convergence=tuple(
                DistributionConvergenceCheck.from_json(check)
                for check in (raw_checks if isinstance(raw_checks, Sequence) else ())
                if isinstance(check, Mapping)
            ),
            terms=terms,
            total_rms_T=(
                None if payload.get("total_rms_T") is None else float(payload["total_rms_T"])
            ),
            applied_T=None if payload.get("applied_T") is None else _tuple3(payload["applied_T"]),
            demag_T=None if payload.get("demag_T") is None else _tuple3(payload["demag_T"]),
        )


def _terms_from_json(raw_terms: object) -> dict[str, FieldTermStatistics]:
    """Return the saved term statistics, or none unless every term is present and well formed.

    A partial breakdown would not sum to the total, so a missing, unknown or
    malformed term drops it whole and the result reads as one saved without a
    breakdown. The terms come back in :data:`DISTRIBUTION_FIELD_TERMS` order.
    """

    if not isinstance(raw_terms, Mapping) or set(raw_terms) != set(DISTRIBUTION_FIELD_TERMS):
        return {}
    try:
        return {
            name: FieldTermStatistics.from_json(raw_terms[name])
            for name in DISTRIBUTION_FIELD_TERMS
        }
    except (KeyError, TypeError, ValueError):
        return {}


@dataclass(frozen=True, slots=True)
class RegeneratedPhaseSamples:
    """The phase sampling of a distribution, rebuilt from its calculation parameters.

    ``phases_turns`` has shape ``(n_samples, n_active_q)``: each row is one
    sampled phase per independent incommensurate wavevector, in turns.
    ``component_phase_coefficients`` has one row per backend component, giving
    the phase of that component's lookup table as a combination of those phases.
    """

    phases_turns: np.ndarray
    component_phase_coefficients: np.ndarray


def regenerate_phase_samples(
    result: MagneticFieldDistributionResult,
) -> RegeneratedPhaseSamples:
    """Rebuild the phase samples a distribution was evaluated at, bit for bit.

    A grid is a function of the sample count and the number of independent
    wavevectors; Monte Carlo samples come from the recorded
    ``phase_sampling_seed``. Raises :class:`MagneticFieldError` for a result
    saved before the seed and phase coefficients were recorded.
    """

    params = result.calculation_parameters
    if "phase_sampling_seed" not in params or "component_phase_coefficients" not in params:
        raise MagneticFieldError(
            "this distribution was saved without its phase sampling; recalculate it"
        )
    grid = params.get("resolved_phase_sampling") == "grid"
    try:
        n_dim = len(params.get("phase_basis_rlu") or [])
        raw_coefficients = list(params["component_phase_coefficients"])
        coefficients = np.asarray(raw_coefficients, dtype=float).reshape(
            len(raw_coefficients), n_dim
        )
        sample_count = int(params["phase_sample_count"])
        seed = None if grid or n_dim == 0 else int(params["phase_sampling_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MagneticFieldError(f"this distribution's phase sampling is malformed: {exc}") from exc
    if n_dim == 0:
        phases = np.zeros((1, 0), dtype=float)
    elif grid:
        axis_count = int(round(sample_count ** (1.0 / n_dim)))
        if axis_count**n_dim != sample_count:
            raise MagneticFieldError(
                f"{sample_count} grid samples do not form a {n_dim}-dimensional phase grid"
            )
        phases = _phase_grid(axis_count, n_dim)
    else:
        phases = _monte_carlo_phases(seed, sample_count, n_dim)
    return RegeneratedPhaseSamples(phases_turns=phases, component_phase_coefficients=coefficients)


@dataclass(frozen=True, slots=True)
class CommensurateApproximant:
    """Rational propagation vectors that approximate an incommensurate model.

    ``q_rlu`` holds the model's distinct component wavevectors and
    ``approximant_rlu`` the rational wavevector that replaces each one.
    ``periods`` is the magnetic supercell, in host-cell repeats along a, b and c,
    over which the approximant repeats, and ``max_deviation_rlu`` the largest
    change of any wavevector component.
    """

    q_rlu: tuple[tuple[float, float, float], ...]
    approximant_rlu: tuple[tuple[float, float, float], ...]
    denominators: tuple[tuple[int, int, int], ...]
    periods: tuple[int, int, int]
    max_deviation_rlu: float

    @property
    def is_exact(self) -> bool:
        return self.max_deviation_rlu == 0.0

    def describe(self) -> str:
        """Return a short human-readable description, e.g. ``q = (1/3, 0, 0)``."""

        vectors = "; ".join(
            "q = ("
            + ", ".join(_fraction_text(value, denominator) for value, denominator in zip(q, dens))
            + ")"
            for q, dens in zip(self.approximant_rlu, self.denominators, strict=True)
        )
        a, b, c = self.periods
        return f"{vectors} with a {a} x {b} x {c} magnetic supercell"

    def to_json(self) -> dict[str, object]:
        return {
            "q_rlu": [list(q) for q in self.q_rlu],
            "approximant_rlu": [list(q) for q in self.approximant_rlu],
            "denominators": [list(d) for d in self.denominators],
            "periods": list(self.periods),
            "max_deviation_rlu": float(self.max_deviation_rlu),
        }


@dataclass(frozen=True, slots=True)
class DipolarTensorResult:
    """One dipolar tensor result in T/mu_B."""

    site_label: str
    muon_position: tuple[float, float, float]
    tensor_T_per_muB: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]

    def to_json(self) -> dict[str, object]:
        return {
            "site_label": self.site_label,
            "muon_position": list(self.muon_position),
            "dipolar_tensor_T_per_muB": [list(row) for row in self.tensor_T_per_muB],
        }


def assigned_magnetic_atom_indices(
    structure: Structure,
    model: MagneticStructureModel,
) -> tuple[int, ...]:
    """Return unit-cell atom indices targeted by enabled, non-zero magnetic components."""

    atom_indices: set[int] = set()
    for component in model.components:
        if not component.enabled or not component.terms:
            continue
        if all(np.allclose(term.coefficient.to_array(), 0.0) for term in component.terms):
            continue
        atom_indices.update(resolve_atom_selector(structure, component.selector))
    return tuple(sorted(atom_indices))


def is_commensurate_propagation_vector(
    k_rlu: Sequence[float],
    *,
    tolerance: float = 1e-8,
    max_period: int = 64,
) -> bool:
    """Return whether a propagation vector is commensurate within a bounded period."""

    k = _as_vector(k_rlu, "k_rlu")
    for period in range(1, int(max_period) + 1):
        values = period * k
        if np.all(np.abs(values - np.rint(values)) <= float(tolerance)):
            return True
    return False


def classify_magnetic_model(
    structure: Structure,
    model: MagneticStructureModel,
    *,
    tolerance: float = 1e-8,
    max_period: int = 64,
) -> str:
    """Classify a model as commensurate or incommensurate."""

    models = get_muesr_model_data(structure, model, merge_equivalent_k=False)
    if not models:
        raise MagneticFieldError("magnetic structure has no non-zero Fourier components")
    if all(
        is_commensurate_propagation_vector(m.k_rlu, tolerance=tolerance, max_period=max_period)
        for m in models
    ):
        return "commensurate"
    return "incommensurate"


def find_commensurate_approximant(
    structure: Structure,
    model: MagneticStructureModel,
    *,
    tolerance: float = 1e-3,
    max_period: int = 12,
) -> CommensurateApproximant | None:
    """Return the shortest-period rational approximant of every component wavevector.

    Each wavevector component ``q_i`` is replaced by the fraction ``p/n`` with the
    smallest ``n <= max_period`` for which ``|q_i - p/n| <= tolerance`` (in
    reciprocal-lattice units). Returns ``None`` when any component has no such
    fraction, so the model is genuinely incommensurate at that tolerance.
    """

    if float(tolerance) < 0.0:
        raise MagneticFieldError("approximant tolerance must not be negative")
    if int(max_period) < 1:
        raise MagneticFieldError("approximant maximum period must be positive")
    models = get_muesr_model_data(structure, model, merge_equivalent_k=True)
    if not models:
        raise MagneticFieldError("magnetic structure has no non-zero Fourier components")
    q_vectors: list[tuple[float, float, float]] = []
    approximants: list[tuple[float, float, float]] = []
    denominators: list[tuple[int, int, int]] = []
    for model_data in models:
        rational = _rational_propagation_vector(
            model_data.k_rlu, tolerance=tolerance, max_period=max_period
        )
        if rational is None:
            return None
        q_vectors.append(_tuple3(model_data.k_rlu))
        approximants.append(_tuple3(rational[0] / rational[1]))
        denominators.append((int(rational[1][0]), int(rational[1][1]), int(rational[1][2])))
    periods = tuple(math.lcm(*(dens[axis] for dens in denominators)) for axis in range(3))
    deviation = max(
        float(np.max(np.abs(np.asarray(q) - np.asarray(approx))))
        for q, approx in zip(q_vectors, approximants, strict=True)
    )
    return CommensurateApproximant(
        q_rlu=tuple(q_vectors),
        approximant_rlu=tuple(approximants),
        denominators=tuple(denominators),
        periods=(int(periods[0]), int(periods[1]), int(periods[2])),
        max_deviation_rlu=deviation,
    )


def find_equivalent_muon_sites(
    structure: Structure,
    muon_position: Sequence[float],
    *,
    tolerance: float = 1e-3,
) -> tuple[tuple[float, float, float], ...]:
    """Return unique symmetry-equivalent fractional muon positions."""

    mu = _canonical_fractional(muon_position)
    cell = spglib_cell(structure)
    if cell is None:
        raise MagneticFieldError(
            "spglib could not determine symmetry operations: a species is not an element"
        )
    operations = symmetry_operations(cell, symprec=SYMMETRY_TOLERANCE_ANG)
    if operations is None:
        raise MagneticFieldError("spglib could not determine symmetry operations")
    # ``tolerance`` is a fractional-coordinate distance, so the orbit is deduplicated without a lattice.
    unique = symmetry_orbit(mu, operations, lattice_vectors=None, tolerance=tolerance)
    return tuple(_tuple3(_canonical_fractional(value)) for value in unique)


def calculate_finite_dipolar_field(
    positions_cart_ang: Sequence[Sequence[float]],
    moments_mu_b: Sequence[Sequence[float]],
    muon_cart_ang: Sequence[float],
) -> tuple[float, float, float]:
    """Calculate the finite-cell dipolar field in Tesla without muesr."""

    positions = np.asarray(positions_cart_ang, dtype=float)
    moments = np.asarray(moments_mu_b, dtype=float)
    muon = _as_vector(muon_cart_ang, "muon_cart_ang")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise MagneticFieldError("positions_cart_ang must have shape (n, 3)")
    if moments.shape != positions.shape:
        raise MagneticFieldError("moments_mu_b must match positions_cart_ang shape")

    r_cart = muon[np.newaxis, :] - positions
    r_norm_ang = np.linalg.norm(r_cart, axis=1)
    mask = r_norm_ang > 1e-10
    if not np.any(mask):
        return (0.0, 0.0, 0.0)
    r_cart = r_cart[mask]
    moments = moments[mask]
    r_norm_m = r_norm_ang[mask] * METRES_PER_ANGSTROM
    r_hat = r_cart / r_norm_ang[mask, np.newaxis]
    m_si = moments * BOHR_MAGNETON_J_PER_T
    inv_r3 = 1.0 / (r_norm_m**3)
    mdotr = np.sum(m_si * r_hat, axis=1)
    terms = (3.0 * mdotr[:, np.newaxis] * r_hat - m_si) * inv_r3[:, np.newaxis]
    return _tuple3(MU0_OVER_4PI_T_M_PER_A * np.sum(terms, axis=0))


def calculate_distortion_field_correction(
    structure: Structure,
    model: MagneticStructureModel,
    distortion: MuonDistortion,
) -> tuple[float, float, float]:
    """Return the dipolar-field change in Tesla from muon-induced host-ion displacements.

    The correction is the finite-cell difference between the field of the
    displaced ions and the field of the same ions at their pristine sites. Each
    ion keeps the moment of its pristine site, evaluated from ``model``.
    """

    total = np.zeros(3, dtype=float)
    for model_data in get_muesr_model_data(structure, model, merge_equivalent_k=True):
        dipolar, _contact = _component_distortion_changes(model_data, distortion, None)
        total += dipolar.real
    return _tuple3(total)


def _component_distortion_changes(
    model_data: MuesrModelData,
    distortion: MuonDistortion,
    contact_options: MagneticFieldOptions | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one Fourier component's complex dipolar and contact changes, in T.

    Each displaced ion carries its pristine site's coefficient advanced to its
    cell, ``F exp(-2 pi i q . L)``, so the change in field when the component's
    order is shifted by a phase ``theta`` (in turns) is
    ``Re[exp(-2 pi i theta) dB]``, matching moments ``Re[F exp(-2 pi i q . R)]``.
    The contact change is zero unless ``contact_options`` sets a contact coupling;
    it follows muLFC's contact rule for one backend call, which sees only the
    atoms with non-zero coefficients in ``model_data``.
    """

    fc = np.asarray(model_data.fc, dtype=complex)[distortion.atom_indices]
    q_rlu = np.asarray(model_data.k_rlu, dtype=float).reshape(3)
    coefficients = fc * np.exp(-2j * np.pi * (distortion.cell_offsets @ q_rlu))[:, np.newaxis]
    origin = np.zeros(3, dtype=float)
    dipolar = np.zeros(3, dtype=complex)
    for part, scale in ((coefficients.real, 1.0), (coefficients.imag, 1j)):
        if not np.any(part):
            continue
        displaced = calculate_finite_dipolar_field(distortion.distorted_vectors, part, origin)
        pristine = calculate_finite_dipolar_field(distortion.pristine_vectors, part, origin)
        dipolar += scale * (np.asarray(displaced) - np.asarray(pristine))
    contact = np.zeros(3, dtype=complex)
    if contact_options is not None and contact_options.contact_coupling is not None:
        # muesr drops an atom from the call when its coefficients are all close to zero.
        magnetic = ~np.all(np.abs(fc) <= 1e-8, axis=1)
        reach = contact_shell_reach_limit(distortion.supercell_lattice)
        shells = [
            contact_shell_mean(
                vectors,
                coefficients,
                magnetic,
                nnn=int(contact_options.nnn),
                rcont=float(contact_options.rcont),
                reach_limit=reach,
            )
            for vectors in (distortion.distorted_vectors, distortion.pristine_vectors)
        ]
        contact = (
            float(contact_options.contact_coupling)
            * CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB
            * (shells[0] - shells[1])
        )
    return dipolar, contact


def contact_shell_mean(
    vectors: np.ndarray,
    coefficients: np.ndarray,
    magnetic: np.ndarray,
    *,
    nnn: int,
    rcont: float,
    reach_limit: float,
) -> np.ndarray:
    """Return muLFC's contact average of ``coefficients`` over the muon's nearest magnetic ions.

    The ions are the ``nnn`` nearest magnetic ions closer than ``rcont``,
    weighted by ``1 / r^3`` and normalised by the sum of those weights. The
    relaxed supercell holds every ion only within ``reach_limit`` of the muon, so
    a shell that reaches further could miss a nearer periodic image.
    """

    if int(nnn) < 1:
        return np.zeros(3, dtype=complex)
    distances = np.linalg.norm(vectors, axis=1)
    candidates = np.flatnonzero(magnetic & (distances < rcont))
    ordered = candidates[np.argsort(distances[candidates], kind="stable")]
    shell = ordered[: int(nnn)]
    reach = float(distances[shell[-1]]) if shell.size == int(nnn) else rcont
    if reach >= reach_limit:
        remedy = (
            "fewer Contact neighbours or a smaller Contact radius"
            if shell.size == int(nnn)
            else f"a smaller Contact radius (only {shell.size} magnetic ions lie within it, "
            f"fewer than its {int(nnn)} Contact neighbours, so the shell reaches the radius)"
        )
        raise MagneticFieldError(
            f"the contact shell reaches {reach:.2f} Å from the muon but the relaxed supercell only holds "
            f"every ion within {reach_limit:.2f} Å; use a larger relaxed supercell, or {remedy} in "
            "Calculation Settings"
        )
    if ordered.size > shell.size:
        # muLFC keeps whichever tied ion its parallel lattice walk meets first, so a tie
        # across the last place of the shell leaves its contact field undefined.
        tied = ordered[np.abs(distances[ordered] - reach) <= _CONTACT_SHELL_TIE_ANG]
        if tied.size > np.count_nonzero(np.isin(tied, shell)) and not np.allclose(
            coefficients[tied], coefficients[tied[0]], atol=1e-8, rtol=0.0
        ):
            raise MagneticFieldError(
                f"the contact shell is ambiguous: {tied.size} magnetic ions with different moments tie at "
                f"{reach:.3f} Å from the muon for the last of its {int(nnn)} places, and muesr keeps an "
                "arbitrary one of them; set Contact neighbours (nnn) in Calculation Settings to include "
                "or exclude the whole tied set"
            )
    if not shell.size:
        return np.zeros(3, dtype=complex)
    weights = distances[shell] ** -3
    return np.sum(weights[:, np.newaxis] * coefficients[shell], axis=0) / np.sum(weights)


_CONTACT_SHELL_TIE_ANG = 1e-6
"""Distance difference within which two ions tie for a place in the contact shell."""


def contact_shell_reach_limit(supercell_lattice: np.ndarray) -> float:
    """Return the radius of the largest sphere about the muon that the wrapped relaxed supercell holds."""

    lattice = np.asarray(supercell_lattice, dtype=float)
    volume = abs(float(np.linalg.det(lattice)))
    heights = [
        volume / float(np.linalg.norm(np.cross(lattice[(axis + 1) % 3], lattice[(axis + 2) % 3])))
        for axis in range(3)
    ]
    return 0.5 * min(heights)


def _lookup_row_phases_turns(
    k_rlu: Sequence[float], muon_fractional: Sequence[float], supercell: Sequence[int], nangles: int
) -> np.ndarray:
    """Return the order phase, in turns, that each row of a muLFC incommensurate lookup samples.

    muLFC measures the incommensurate phase from a reference point displaced by
    the muon's position and twice the cell it places the muon in, so with
    :func:`_muesr_fourier_components` row ``j`` holds the field of the order
    shifted by ``j / nangles - q . (mu + 2 floor(supercell / 2))``.
    """

    reference = np.asarray(muon_fractional, dtype=float) + 2.0 * np.floor(
        np.asarray(supercell, dtype=float) / 2.0
    )
    offset = -float(np.dot(np.asarray(k_rlu, dtype=float), reference))
    return np.arange(int(nangles), dtype=float) / float(nangles) + offset


def _phase_shifted_fields(change: np.ndarray, phases_turns: np.ndarray) -> np.ndarray:
    """Return ``Re[exp(-2 pi i theta) change]`` for each phase ``theta`` in turns, shape ``(n, 3)``."""

    factors = np.exp(-2j * np.pi * np.asarray(phases_turns, dtype=float))
    return (factors[:, np.newaxis] * np.asarray(change, dtype=complex)[np.newaxis, :]).real


def _distortion_changes(
    models: Sequence[MuesrModelData],
    distortion: MuonDistortion,
    options: MagneticFieldOptions,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return each commensurate component's propagation vector and complex dipolar plus contact change."""

    changes = []
    for model_data in models:
        dipolar, contact = _component_distortion_changes(model_data, distortion, options)
        changes.append((np.asarray(model_data.k_rlu, dtype=float), dipolar + contact))
    return changes


def _distortion_correction(
    changes: Sequence[tuple[np.ndarray, np.ndarray]], translation: Sequence[int]
) -> tuple[float, float, float]:
    """Return the distortion correction, in T, at the site translated by ``translation``.

    Translating the muon by ``T`` shifts each commensurate component's order by
    the phase ``q . T``.
    """

    total = np.zeros(3, dtype=float)
    for q_rlu, change in changes:
        phase = float(np.dot(q_rlu, translation))
        total += _phase_shifted_fields(change, np.array([phase]))[0]
    return _tuple3(total)


def calculate_commensurate_fields(
    structure: Structure,
    model: MagneticStructureModel,
    muon_sites: Sequence[MuonSite | Sequence[float]],
    *,
    options: MagneticFieldOptions | None = None,
    distorted_structures: Sequence[Structure | None] | None = None,
) -> tuple[MagneticFieldResult, ...]:
    """Calculate commensurate vector fields with the optional muesr backend.

    ``distorted_structures`` optionally gives, per entry of ``muon_sites``, the
    relaxed supercell containing that muon. A site with one receives the
    muon-induced distortion correction
    (:func:`calculate_distortion_field_correction`), carried to every
    symmetry-equivalent site when ``options.find_equivalent_sites`` is set.
    """

    opts = options or MagneticFieldOptions()
    distortions = _site_distortions(structure, muon_sites, distorted_structures, opts)
    models = get_muesr_model_data(structure, model, merge_equivalent_k=True)
    if not models:
        raise MagneticFieldError("magnetic structure has no non-zero Fourier components")
    noncomm = [
        m.k_rlu
        for m in models
        if not is_commensurate_propagation_vector(
            m.k_rlu,
            tolerance=opts.commensurate_tolerance,
            max_period=opts.commensurate_max_period,
        )
    ]
    if noncomm:
        raise MagneticFieldError("commensurate field calculation received an incommensurate model")

    locfield, find_largest_sphere, _dipten = _load_muesr_engines()
    muons = _expanded_muon_positions(structure, muon_sites, opts)
    sample = _build_muesr_sample(structure, [position for _label, position in muons])
    radius = (
        float(opts.radius)
        if opts.radius is not None
        else float(find_largest_sphere(sample, list(opts.supercell)))
    )

    dipolar, lorentz, contact = _commensurate_field_terms(
        structure,
        models,
        [position for _label, position in muons],
        opts,
        radius,
        locfield,
    )

    applied = _as_vector(opts.applied_field, "applied_field")
    demag = _demag_field(structure, models, opts)
    results: list[MagneticFieldResult] = []
    for index, (label, position) in enumerate(muons):
        distortion = distortions[index]
        results.append(
            MagneticFieldResult(
                site_label=label,
                muon_position=_tuple3(position),
                dipolar=_tuple3(dipolar[index]),
                lorentz=_tuple3(lorentz[index]),
                contact=_tuple3(contact[index]),
                demag=_tuple3(demag),
                applied=_tuple3(applied),
                distortion=(
                    None
                    if distortion is None
                    else _distortion_correction(
                        _distortion_changes(models, distortion, opts), (0, 0, 0)
                    )
                ),
                calculation_parameters=_contact_calculation_parameters(opts),
            )
        )
    return tuple(results)


def calculate_commensurate_approximant_fields(
    structure: Structure,
    model: MagneticStructureModel,
    muon_sites: Sequence[MuonSite | Sequence[float]],
    *,
    options: MagneticFieldOptions | None = None,
    distorted_structures: Sequence[Structure | None] | None = None,
) -> tuple[MagneticFieldResult, ...]:
    """Calculate the finite field list of a model's commensurate approximant.

    Every component wavevector is replaced by its rational approximant
    (:func:`find_commensurate_approximant` with ``options.approximant_tolerance``
    and ``options.approximant_max_period``). The approximant repeats over a
    finite magnetic supercell, so translation-equivalent copies ``r + T`` of a
    muon site see a finite set of fields. One result is returned per muon site
    and per translation ``T`` that gives a distinct set of magnetic phases; the
    field at ``r + T`` is the commensurate sum with every Fourier phase advanced
    by ``q . T``. Each listed translation stands for the same number of
    translation-equivalent sites, so the list carries equal weights.

    ``distorted_structures`` works as for :func:`calculate_commensurate_fields`;
    the correction at ``r + T`` uses the approximant order advanced by ``q . T``.
    """

    opts = options or MagneticFieldOptions()
    distortions = _site_distortions(structure, muon_sites, distorted_structures, opts)
    models = get_muesr_model_data(structure, model, merge_equivalent_k=True)
    if not models:
        raise MagneticFieldError("magnetic structure has no non-zero Fourier components")
    rationals: list[tuple[np.ndarray, np.ndarray]] = []
    for model_data in models:
        rational = _rational_propagation_vector(
            model_data.k_rlu,
            tolerance=opts.approximant_tolerance,
            max_period=opts.approximant_max_period,
        )
        if rational is None:
            raise MagneticFieldError(
                "magnetic structure has no commensurate approximant within "
                f"{float(opts.approximant_tolerance):g} r.l.u. and period {int(opts.approximant_max_period)}"
            )
        rationals.append(rational)
    translations = _distinct_phase_translations(rationals, limit=int(opts.max_approximant_fields))
    if translations is None:
        raise MagneticFieldError(
            f"the commensurate approximant has more than {int(opts.max_approximant_fields)} inequivalent "
            "translations; lower the approximant period or sample a distribution instead"
        )
    approximant_models = [
        replace(model_data, k_rlu=numerators / denominators)
        for model_data, (numerators, denominators) in zip(models, rationals, strict=True)
    ]

    locfield, find_largest_sphere, _dipten = _load_muesr_engines()
    muons = _expanded_muon_positions(structure, muon_sites, opts)
    positions = [position for _label, position in muons]
    sample = _build_muesr_sample(structure, positions)
    radius = (
        float(opts.radius)
        if opts.radius is not None
        else float(find_largest_sphere(sample, list(opts.supercell)))
    )
    applied = _as_vector(opts.applied_field, "applied_field")
    demag = _demag_field(structure, approximant_models, opts)

    terms_by_translation = []
    for translation in translations:
        shifted = [
            replace(
                model_data,
                phi=np.asarray(model_data.phi, dtype=float)
                + float(np.dot(model_data.k_rlu, translation)),
            )
            for model_data in approximant_models
        ]
        terms_by_translation.append(
            _commensurate_field_terms(structure, shifted, positions, opts, radius, locfield)
        )

    results: list[MagneticFieldResult] = []
    for index, (label, position) in enumerate(muons):
        changes = (
            None
            if distortions[index] is None
            else _distortion_changes(approximant_models, distortions[index], opts)
        )
        for translation, (dipolar, lorentz, contact) in zip(
            translations, terms_by_translation, strict=True
        ):
            site_label = label
            if len(translations) > 1:
                site_label = f"{label} + ({translation[0]}, {translation[1]}, {translation[2]})"
            results.append(
                MagneticFieldResult(
                    site_label=site_label,
                    muon_position=_tuple3(
                        np.asarray(position, dtype=float) + np.asarray(translation, dtype=float)
                    ),
                    dipolar=_tuple3(dipolar[index]),
                    lorentz=_tuple3(lorentz[index]),
                    contact=_tuple3(contact[index]),
                    demag=_tuple3(demag),
                    applied=_tuple3(applied),
                    distortion=(
                        None if changes is None else _distortion_correction(changes, translation)
                    ),
                    calculation_parameters=_contact_calculation_parameters(opts),
                )
            )
    return tuple(results)


def _commensurate_field_terms(
    structure: Structure,
    models: Sequence[object],
    positions: Sequence[np.ndarray],
    options: MagneticFieldOptions,
    radius: float,
    locfield,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Sum the dipolar, Lorentz and contact fields of commensurate models at each position."""

    dipolar = [np.zeros(3, dtype=float) for _ in positions]
    lorentz = [np.zeros(3, dtype=float) for _ in positions]
    contact = [np.zeros(3, dtype=float) for _ in positions]
    for model_data in models:
        sample = _build_muesr_sample(structure, positions)
        sample.new_mm()
        sample.mm.k = model_data.k_rlu
        sample.mm.fc = _muesr_fourier_components(model_data, options.supercell)
        sample.mm.phi = model_data.phi
        fields = locfield(
            sample,
            "s",
            list(options.supercell),
            radius,
            nnn=int(options.nnn),
            rcont=float(options.rcont),
        )
        for index, local_field in enumerate(fields):
            if options.contact_coupling is not None:
                local_field.ACont = float(options.contact_coupling)
            dipolar[index] += _field_vector(local_field, "Dipolar", "D")
            if options.include_lorentz:
                lorentz[index] += _field_vector(local_field, "Lorentz", "L")
            contact[index] += _field_vector(local_field, "Contact", "C")
    return dipolar, lorentz, contact


def _rational_propagation_vector(
    k_rlu: Sequence[float],
    *,
    tolerance: float,
    max_period: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return integer numerators and denominators approximating ``k_rlu`` axis by axis."""

    k = _as_vector(k_rlu, "k_rlu")
    numerators = np.zeros(3, dtype=int)
    denominators = np.ones(3, dtype=int)
    for axis, value in enumerate(k):
        for period in range(1, int(max_period) + 1):
            numerator = int(np.rint(value * period))
            if abs(value - numerator / period) <= float(tolerance) + 1e-12:
                divisor = math.gcd(numerator, period)
                numerators[axis] = numerator // divisor
                denominators[axis] = period // divisor
                break
        else:
            return None
    return numerators, denominators


def _distinct_phase_translations(
    rationals: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    limit: int,
) -> tuple[tuple[int, int, int], ...] | None:
    """Return one lattice translation per distinct set of approximant phases ``q . T``.

    The phase sets form the finite group generated by the phases of the three
    unit translations, so a breadth-first walk over that group, stepping one unit
    translation at a time, reaches each set once without enumerating the whole
    magnetic supercell. The first translation to reach a set represents it, and
    every set is reached by the same number of supercell translations. Returns
    ``None`` as soon as more than ``limit`` sets are found.
    """

    def phases(translation: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
        return tuple(
            _phase_numerator(numerators, denominators, translation)
            for numerators, denominators in rationals
        )

    origin = (0, 0, 0)
    representatives = {phases(origin): origin}
    queue = deque([origin])
    while queue:
        translation = queue.popleft()
        for axis in range(3):
            step = list(translation)
            step[axis] += 1
            candidate = (step[0], step[1], step[2])
            key = phases(candidate)
            if key in representatives:
                continue
            representatives[key] = candidate
            if len(representatives) > int(limit):
                return None
            queue.append(candidate)
    return tuple(sorted(representatives.values()))


def _phase_numerator(
    numerators: np.ndarray, denominators: np.ndarray, translation: Sequence[int]
) -> tuple[int, int]:
    """Return ``q . T mod 1`` exactly, as ``(numerator, denominator)`` over the lcm of q's denominators."""

    common = math.lcm(*(int(value) for value in denominators))
    total = sum(
        int(numerators[axis]) * int(translation[axis]) * (common // int(denominators[axis]))
        for axis in range(3)
    )
    return (total % common, common)


def calculate_dipolar_tensors(
    structure: Structure,
    model: MagneticStructureModel,
    muon_sites: Sequence[MuonSite | Sequence[float]],
    *,
    options: MagneticFieldOptions | None = None,
) -> tuple[dict[str, object], ...]:
    """Calculate dipolar tensors for atoms targeted by a magnetic model.

    This compatibility wrapper keeps the historical model-based entry point.
    New code should prefer :func:`calculate_dipolar_tensors_for_atoms`, which
    accepts the magnetic-ion selection directly.
    """

    opts = options or MagneticFieldOptions()
    indices = assigned_magnetic_atom_indices(structure, model)
    results = calculate_dipolar_tensors_for_atoms(
        structure,
        muon_sites,
        magnetic_ion_indices=indices,
        options=opts,
    )
    return tuple(result.to_json() for result in results)


def calculate_dipolar_tensors_for_atoms(
    structure: Structure,
    muon_sites: Sequence[MuonSite | Sequence[float]],
    *,
    magnetic_ion_indices: Sequence[int] | None = None,
    magnetic_ion_selector: AtomSelector | None = None,
    options: MagneticFieldOptions | None = None,
) -> tuple[DipolarTensorResult, ...]:
    """Calculate dipolar tensors for explicitly selected magnetic ions."""

    opts = options or MagneticFieldOptions()
    indices = _resolve_magnetic_ion_indices(
        structure,
        magnetic_ion_indices=magnetic_ion_indices,
        magnetic_ion_selector=magnetic_ion_selector,
    )

    _locfield, find_largest_sphere, dipten = _load_muesr_engines()
    muons = _expanded_muon_positions(structure, muon_sites, opts)
    radius_sample = _build_muesr_sample(structure, [position for _label, position in muons])
    radius = (
        float(opts.radius)
        if opts.radius is not None
        else float(find_largest_sphere(radius_sample, list(opts.supercell)))
    )
    sample = _build_muesr_sample(structure, [position for _label, position in muons])
    sample.new_mm()
    sample.mm.k = np.zeros(3, dtype=float)
    sample.mm.fc = _unit_tensor_fourier_components(structure, indices)
    sample.mm.phi = np.zeros(len(structure.species), dtype=float)
    tensors = [
        np.asarray(tensor, dtype=float) * DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3
        for tensor in dipten(sample, list(opts.supercell), radius)
    ]
    if len(tensors) != len(muons):
        raise MagneticFieldError("muesr returned an unexpected number of dipolar tensors")
    return tuple(
        DipolarTensorResult(
            site_label=label,
            muon_position=_tuple3(position),
            tensor_T_per_muB=_tensor3x3(tensor),
        )
        for (label, position), tensor in zip(muons, tensors)
    )


def calculate_incommensurate_field_distributions(
    structure: Structure,
    model: MagneticStructureModel,
    muon_sites: Sequence[MuonSite | Sequence[float]],
    *,
    options: MagneticFieldOptions | None = None,
    distorted_structures: Sequence[Structure | None] | None = None,
) -> tuple[MagneticFieldDistributionResult, ...]:
    """Calculate field distributions for incommensurate structures.

    ``distorted_structures`` works as for :func:`calculate_commensurate_fields`.
    Each component's complex distortion change ``dB`` is added to its dipolar
    and contact lookup tables as ``Re[exp(-2 pi i theta) dB]`` at the phase
    ``theta`` each lookup row samples, so every phase sample, term and
    convergence check includes the correction.
    """

    opts = options or MagneticFieldOptions()
    if int(opts.nangles) < 1:
        raise MagneticFieldError("nangles must be positive")
    if int(opts.phase_sample_count) < 1:
        raise MagneticFieldError("phase_sample_count must be positive")
    if int(opts.histogram_bins) < 1:
        raise MagneticFieldError("histogram_bins must be positive")
    locfield, find_largest_sphere, _dipten = _load_muesr_engines()
    models = get_muesr_model_data(structure, model, merge_equivalent_k=False)
    if not models:
        raise MagneticFieldError("magnetic structure has no non-zero Fourier components")

    muons = _expanded_muon_positions(structure, muon_sites, opts)
    distortions = _site_distortions(structure, muon_sites, distorted_structures, opts)
    radius_sample = _build_muesr_sample(structure, [position for _label, position in muons])
    radius = (
        float(opts.radius)
        if opts.radius is not None
        else float(find_largest_sphere(radius_sample, list(opts.supercell)))
    )
    calculation_parameters = _distribution_calculation_parameters(opts, radius)
    phase_basis_rlu, phase_coeffs_by_model = _active_phase_basis_and_coefficients(
        models,
        tolerance=opts.commensurate_tolerance,
        max_period=opts.commensurate_max_period,
    )
    active_q_indices = _active_component_q_indices(models)
    resolved_sampling = _resolved_phase_sampling(
        n_dim=phase_basis_rlu.shape[0],
        nangles=int(opts.nangles),
        phase_sample_count=int(opts.phase_sample_count),
        sampling=opts.phase_sampling,
        max_grid_samples=int(opts.max_grid_samples),
    )
    # Monte Carlo always runs from a recorded seed, so the samples can be rebuilt later.
    phase_sampling_seed = (
        None
        if resolved_sampling == "grid"
        else (
            int(opts.random_seed)
            if opts.random_seed is not None
            else int(np.random.SeedSequence().entropy) % 2**63
        )
    )
    phase_samples_turns = _build_phase_samples_turns(
        n_dim=phase_basis_rlu.shape[0],
        nangles=int(opts.nangles),
        phase_sample_count=int(opts.phase_sample_count),
        sampling=opts.phase_sampling,
        random_seed=phase_sampling_seed,
        max_grid_samples=int(opts.max_grid_samples),
    )
    calculation_parameters["independent_q_indices"] = list(active_q_indices)
    calculation_parameters["phase_basis_rlu"] = phase_basis_rlu.tolist()
    calculation_parameters["phase_sample_count"] = int(phase_samples_turns.shape[0])
    calculation_parameters["resolved_phase_sampling"] = resolved_sampling
    calculation_parameters["phase_sampling_seed"] = phase_sampling_seed
    component_totals_by_site: list[list[np.ndarray]] = [[] for _ in muons]
    # Per site, the dipolar, Lorentz and contact lookup tables of every component.
    component_terms_by_site: list[dict[str, list[np.ndarray]]] = [
        {"dipolar": [], "lorentz": [], "contact": []} for _ in muons
    ]
    component_phase_coeffs: list[tuple[float, ...]] = []

    # One backend call per entry: incommensurate components split into circular helices.
    backend_entries: list[tuple[str, Sequence[float], MuesrModelData]] = []
    for model_data, phase_coeff in zip(models, phase_coeffs_by_model, strict=True):
        mode = (
            "s"
            if is_commensurate_propagation_vector(
                model_data.k_rlu,
                tolerance=opts.commensurate_tolerance,
                max_period=opts.commensurate_max_period,
            )
            else "i"
        )
        model_entries = (
            split_muesr_model_data_into_helices(model_data) if mode == "i" else (model_data,)
        )
        backend_entries.extend((mode, phase_coeff, entry) for entry in model_entries)
    # The distortion changes, and so the contact-shell checks, come before any lattice sum.
    changes_by_site = [
        None
        if distortion is None
        else [
            _component_distortion_changes(entry, distortion, opts) for *_, entry in backend_entries
        ]
        for distortion in distortions
    ]

    for entry_index, (mode, phase_coeff, model_entry) in enumerate(backend_entries):
        sample = _build_muesr_sample(structure, [position for _label, position in muons])
        sample.new_mm()
        sample.mm.k = model_entry.k_rlu
        sample.mm.fc = _muesr_fourier_components(model_entry, opts.supercell)
        sample.mm.phi = model_entry.phi
        kwargs: dict[str, Any] = {}
        if mode == "i":
            kwargs["nangles"] = int(opts.nangles)
        fields = locfield(
            sample,
            mode,
            list(opts.supercell),
            radius,
            nnn=int(opts.nnn),
            rcont=float(opts.rcont),
            **kwargs,
        )
        for index, local_field in enumerate(fields):
            if opts.contact_coupling is not None:
                local_field.ACont = float(opts.contact_coupling)
            dip = _field_array(local_field, "Dipolar", "D", nangles=int(opts.nangles))
            lor = (
                _field_array(local_field, "Lorentz", "L", nangles=int(opts.nangles))
                if opts.include_lorentz
                else np.zeros_like(dip)
            )
            con = _field_array(local_field, "Contact", "C", nangles=int(opts.nangles))
            site_changes = changes_by_site[index]
            if site_changes is not None:
                row_phases = (
                    _lookup_row_phases_turns(
                        model_entry.k_rlu, muons[index][1], opts.supercell, int(opts.nangles)
                    )
                    if mode == "i"
                    else np.zeros(int(opts.nangles), dtype=float)
                )
                dip_change, con_change = site_changes[entry_index]
                dip = dip + _phase_shifted_fields(dip_change, row_phases)
                con = con + _phase_shifted_fields(con_change, row_phases)
            component_totals_by_site[index].append(dip + lor + con)
            site_terms = component_terms_by_site[index]
            site_terms["dipolar"].append(dip)
            site_terms["lorentz"].append(lor)
            site_terms["contact"].append(con)
        component_phase_coeffs.append(tuple(float(value) for value in phase_coeff))
    calculation_parameters["component_phase_coefficients"] = [
        list(coeff) for coeff in component_phase_coeffs
    ]

    # The demagnetizing field depends only on the bulk magnetisation, so it is
    # the same at every sampled phase point.
    applied = _as_vector(opts.applied_field, "applied_field")
    demag = _demag_field(structure, models, opts)
    offset = applied + demag
    half_samples = _half_resolution_phase_samples(phase_samples_turns, resolved_sampling)
    # The lookup rows each phase sample reads depend only on the phases, so every
    # site and term shares them.
    full_indices = _component_lookup_indices(
        component_phase_coeffs, phase_samples_turns, nangles=int(opts.nangles)
    )
    half_lookup_indices = (
        _component_lookup_indices(
            component_phase_coeffs, phase_samples_turns, nangles=int(opts.nangles), stride=2
        )
        if phase_basis_rlu.shape[0] and int(opts.nangles) % 2 == 0
        else None
    )
    half_sample_indices = (
        _component_lookup_indices(component_phase_coeffs, half_samples, nangles=int(opts.nangles))
        if half_samples is not None
        else None
    )
    results: list[MagneticFieldDistributionResult] = []
    for index, (label, position) in enumerate(muons):
        term_fields = {
            name: _sum_component_lookups(tables, full_indices)
            for name, tables in component_terms_by_site[index].items()
        }
        fields = sum(term_fields.values()) + offset[np.newaxis, :]
        magnitudes = np.linalg.norm(fields, axis=1)
        terms = {name: _sampled_term_statistics(values) for name, values in term_fields.items()}
        terms["demag"] = _constant_term_statistics(demag)
        terms["applied"] = _constant_term_statistics(applied)
        centres, counts = _field_magnitude_histogram(magnitudes, bins=int(opts.histogram_bins))
        convergence = _distribution_convergence_checks(
            magnitudes,
            counts,
            half_lookup_magnitudes=(
                np.linalg.norm(
                    _sum_component_lookups(
                        component_totals_by_site[index], half_lookup_indices, stride=2
                    )
                    + offset,
                    axis=1,
                )
                if half_lookup_indices is not None
                else None
            ),
            half_sample_magnitudes=(
                np.linalg.norm(
                    _sum_component_lookups(component_totals_by_site[index], half_sample_indices)
                    + offset,
                    axis=1,
                )
                if half_sample_indices is not None
                else None
            ),
            nangles=int(opts.nangles),
            lookup_limited=(
                resolved_sampling == "grid"
                and phase_basis_rlu.shape[0] > 0
                and _grid_axis_count(
                    n_dim=phase_basis_rlu.shape[0],
                    nangles=int(opts.nangles),
                    phase_sample_count=int(opts.phase_sample_count),
                )
                == int(opts.nangles)
            ),
        )
        results.append(
            MagneticFieldDistributionResult(
                site_label=label,
                muon_position=_tuple3(position),
                fields_T=fields,
                histogram_centres_T=centres,
                histogram_counts=counts,
                calculation_parameters={
                    **calculation_parameters,
                    "distortion_corrected": changes_by_site[index] is not None,
                },
                convergence=convergence,
                terms={name: terms[name] for name in DISTRIBUTION_FIELD_TERMS},
                total_rms_T=float(np.sqrt(np.mean(magnitudes**2))),
                applied_T=_tuple3(applied),
                demag_T=_tuple3(demag),
            )
        )
    return tuple(results)


def _sampled_term_statistics(fields: np.ndarray) -> FieldTermStatistics:
    return FieldTermStatistics(
        mean_T=_tuple3(np.mean(fields, axis=0)),
        rms_T=float(np.sqrt(np.mean(np.sum(fields**2, axis=1)))),
    )


def _constant_term_statistics(vector: np.ndarray) -> FieldTermStatistics:
    return FieldTermStatistics(mean_T=_tuple3(vector), rms_T=float(np.linalg.norm(vector)))


def _half_resolution_phase_samples(
    phase_samples_turns: np.ndarray, sampling: str
) -> np.ndarray | None:
    """Return the phase samples at half the sample budget, or ``None`` for a single phase.

    Monte Carlo keeps every other sample; a grid keeps every other point along
    each phase axis, the grid a budget of half the points per axis would build.
    """

    phases = np.asarray(phase_samples_turns, dtype=float)
    n_dim = phases.shape[1]
    if n_dim == 0 or phases.shape[0] < 2:
        return None
    if sampling == "grid":
        axis_count = int(round(phases.shape[0] ** (1.0 / n_dim)))
        if axis_count < 2 or axis_count**n_dim != phases.shape[0]:
            return None
        grid = phases.reshape((axis_count,) * n_dim + (n_dim,))
        return grid[(slice(None, None, 2),) * n_dim].reshape(-1, n_dim)
    return phases[::2]


def _distribution_convergence_checks(
    magnitudes: np.ndarray,
    counts: np.ndarray,
    *,
    half_lookup_magnitudes: np.ndarray | None,
    half_sample_magnitudes: np.ndarray | None,
    nangles: int,
    lookup_limited: bool,
) -> tuple[DistributionConvergenceCheck, ...]:
    """Check a p(|B|) distribution against half the lookup angles, half the samples and its bin filling.

    ``half_lookup_magnitudes`` is ``None`` for a single phase or an odd lookup
    count, which has no uniform half-resolution lookup. ``lookup_limited`` marks a grid whose axes already use every lookup angle, so
    more samples need more lookup angles.
    """

    checks: list[DistributionConvergenceCheck] = []
    tolerance_percent = 100.0 * CONVERGENCE_TOLERANCE
    more_samples = "more lookup angles per q" if lookup_limited else "more distribution samples"
    if half_lookup_magnitudes is not None:
        distance = _cdf_distance(magnitudes, half_lookup_magnitudes)
        converged = distance <= CONVERGENCE_TOLERANCE
        checks.append(
            DistributionConvergenceCheck(
                setting="nangles",
                label="Lookup angles per q",
                converged=converged,
                metric=distance,
                message=(
                    f"Halving the lookup angles to {nangles // 2} shifts the |B| distribution by {100.0 * distance:.2g}%"
                    + (
                        "."
                        if converged
                        else f" (more than {tolerance_percent:g}%); double Lookup angles per q."
                    )
                ),
            )
        )
    if half_sample_magnitudes is not None:
        distance = _cdf_distance(magnitudes, half_sample_magnitudes)
        converged = distance <= CONVERGENCE_TOLERANCE
        checks.append(
            DistributionConvergenceCheck(
                setting="phase_sample_count",
                label="Distribution samples",
                converged=converged,
                metric=distance,
                message=(
                    f"Halving the samples to {int(half_sample_magnitudes.size)} shifts the |B| distribution "
                    f"by {100.0 * distance:.2g}%"
                    + (
                        "."
                        if converged
                        else f" (more than {tolerance_percent:g}%); use {more_samples}."
                    )
                ),
            )
        )
    occupied = int(np.count_nonzero(counts))
    per_bin = float(np.asarray(magnitudes).size) / float(max(occupied, 1))
    converged = per_bin >= MIN_SAMPLES_PER_OCCUPIED_BIN or occupied <= 1
    checks.append(
        DistributionConvergenceCheck(
            setting="histogram_bins",
            label="Histogram bins",
            converged=converged,
            metric=per_bin,
            message=(
                f"{per_bin:.0f} samples per occupied histogram bin"
                + (
                    "."
                    if converged
                    else f" (fewer than {MIN_SAMPLES_PER_OCCUPIED_BIN:g}); use fewer histogram bins or {more_samples}."
                )
            ),
        )
    )
    return tuple(checks)


def _cdf_distance(reference: np.ndarray, other: np.ndarray) -> float:
    """Return the Kolmogorov-Smirnov distance between two samples of field magnitudes."""

    first = np.sort(np.asarray(reference, dtype=float).ravel())
    second = np.sort(np.asarray(other, dtype=float).ravel())
    points = np.concatenate([first, second])
    first_cdf = np.searchsorted(first, points, side="right") / float(first.size)
    second_cdf = np.searchsorted(second, points, side="right") / float(second.size)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def _active_phase_basis_and_coefficients(
    models: Sequence[object],
    *,
    tolerance: float,
    max_period: int,
) -> tuple[np.ndarray, list[tuple[float, ...]]]:
    """Build the independent physical phase basis used to combine lookup tables."""

    q_vectors: list[np.ndarray] = []
    for model_data in models:
        q_rlu = np.asarray(model_data.k_rlu, dtype=float).reshape(3)
        if is_commensurate_propagation_vector(q_rlu, tolerance=tolerance, max_period=max_period):
            continue
        if _rank_increases(q_vectors, q_rlu, tolerance=tolerance):
            q_vectors.append(q_rlu)

    if not q_vectors:
        return np.zeros((0, 3), dtype=float), [() for _model_data in models]

    basis = np.asarray(q_vectors, dtype=float)
    coeffs: list[tuple[float, ...]] = []
    for model_data in models:
        q_rlu = np.asarray(model_data.k_rlu, dtype=float).reshape(3)
        if is_commensurate_propagation_vector(q_rlu, tolerance=tolerance, max_period=max_period):
            coeffs.append(tuple(0.0 for _ in range(basis.shape[0])))
            continue
        solved, *_ = np.linalg.lstsq(basis.T, q_rlu, rcond=None)
        if not np.allclose(solved @ basis, q_rlu, atol=max(float(tolerance), 1e-10), rtol=0.0):
            raise MagneticFieldError(
                "could not express component q vector in the sampled phase basis"
            )
        coeffs.append(tuple(float(value) for value in solved))
    return basis, coeffs


def _rank_increases(
    existing: Sequence[np.ndarray], candidate: np.ndarray, *, tolerance: float
) -> bool:
    candidate = np.asarray(candidate, dtype=float).reshape(3)
    if float(np.linalg.norm(candidate)) <= float(tolerance):
        return False
    if not existing:
        return True
    before = np.linalg.matrix_rank(np.asarray(existing, dtype=float), tol=float(tolerance))
    after = np.linalg.matrix_rank(
        np.vstack([np.asarray(existing, dtype=float), candidate]), tol=float(tolerance)
    )
    return int(after) > int(before)


def _active_component_q_indices(
    models: Sequence[object], *, tolerance: float = 1e-12
) -> tuple[int, ...]:
    active: set[int] = set()
    for model_data in models:
        for q_coeff in getattr(model_data, "q_coeffs", ()) or ():
            coeff = np.asarray(q_coeff, dtype=float)
            if coeff.shape != (3,):
                continue
            active.update(int(index) for index in np.where(np.abs(coeff) > float(tolerance))[0])
    return tuple(sorted(active))


def _build_phase_samples_turns(
    *,
    n_dim: int,
    nangles: int,
    phase_sample_count: int,
    sampling: str,
    random_seed: int | None,
    max_grid_samples: int,
) -> np.ndarray:
    n_dim = int(n_dim)
    if n_dim == 0:
        return np.zeros((1, 0), dtype=float)
    if int(nangles) < 1:
        raise MagneticFieldError("nangles must be positive")
    if int(phase_sample_count) < 1:
        raise MagneticFieldError("phase_sample_count must be positive")

    mode = _resolved_phase_sampling(
        n_dim=n_dim,
        nangles=int(nangles),
        phase_sample_count=int(phase_sample_count),
        sampling=sampling,
        max_grid_samples=int(max_grid_samples),
    )

    if mode == "grid":
        axis_count = _grid_axis_count(
            n_dim=n_dim,
            nangles=int(nangles),
            phase_sample_count=int(phase_sample_count),
        )
        sample_count = axis_count**n_dim
        if sample_count > int(max_grid_samples):
            raise MagneticFieldError(
                "requested grid phase sampling is too large; reduce incommensurate samples "
                "or use Monte Carlo sampling"
            )
        return _phase_grid(axis_count, n_dim)
    return _monte_carlo_phases(random_seed, int(phase_sample_count), n_dim)


def _phase_grid(axis_count: int, n_dim: int) -> np.ndarray:
    """Return a uniform grid of ``axis_count`` phases, in turns, along each of ``n_dim`` axes."""

    axis_values = np.arange(axis_count, dtype=float) / float(axis_count)
    meshes = np.meshgrid(*([axis_values] * n_dim), indexing="ij")
    return np.stack([mesh.ravel() for mesh in meshes], axis=1)


def _monte_carlo_phases(seed: int | None, sample_count: int, n_dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.random((int(sample_count), n_dim), dtype=float)


def _resolved_phase_sampling(
    *,
    n_dim: int,
    nangles: int,
    phase_sample_count: int,
    sampling: str,
    max_grid_samples: int,
) -> str:
    """Return ``grid`` or ``montecarlo``: the method a requested sampling mode runs as."""

    mode = str(sampling).strip().lower()
    if mode not in {"auto", "grid", "montecarlo", "monte_carlo", "random"}:
        raise MagneticFieldError("phase_sampling must be one of 'auto', 'grid', or 'montecarlo'")
    if mode in {"monte_carlo", "random"}:
        return "montecarlo"
    if mode != "auto":
        return mode
    if int(n_dim) == 0:
        return "grid"
    grid_axis_count = _grid_axis_count(
        n_dim=int(n_dim),
        nangles=int(nangles),
        phase_sample_count=int(phase_sample_count),
    )
    grid_count = grid_axis_count ** int(n_dim)
    return "grid" if int(n_dim) <= 2 and grid_count <= int(max_grid_samples) else "montecarlo"


def _grid_axis_count(*, n_dim: int, nangles: int, phase_sample_count: int) -> int:
    target = max(1, int(phase_sample_count))
    axis_count = int(np.floor(target ** (1.0 / float(n_dim))))
    return max(1, min(int(nangles), axis_count))


def _combine_component_samples(
    component_fields: Sequence[np.ndarray],
    component_q_coeffs: Sequence[Sequence[float]],
    phase_samples_turns: np.ndarray,
    *,
    nangles: int,
    stride: int = 1,
) -> np.ndarray:
    """Sum component lookup fields at each sampled phase point.

    ``stride`` keeps every ``stride``-th lookup row, which evaluates the
    distribution at ``nangles / stride`` lookup angles without new backend calls;
    it must divide ``nangles`` so the kept rows stay uniformly spaced.
    """

    indices = _component_lookup_indices(
        component_q_coeffs, phase_samples_turns, nangles=nangles, stride=stride
    )
    return _sum_component_lookups(component_fields, indices, stride=stride)


def _component_lookup_indices(
    component_q_coeffs: Sequence[Sequence[float]],
    phase_samples_turns: np.ndarray,
    *,
    nangles: int,
    stride: int = 1,
) -> list[np.ndarray]:
    """Return, per component, the lookup row each phase sample reads.

    The rows index a lookup table of ``nangles`` angles thinned to every
    ``stride``-th row; ``stride`` must divide ``nangles``.
    """

    if int(stride) < 1 or int(nangles) % int(stride) != 0:
        raise MagneticFieldError("lookup stride must divide the number of lookup angles")
    phases = np.asarray(phase_samples_turns, dtype=float)
    if phases.ndim != 2:
        raise MagneticFieldError("phase_samples_turns must have shape (n, n_phase_dimensions)")
    rows = int(nangles) // int(stride)
    indices: list[np.ndarray] = []
    for q_coeff in component_q_coeffs:
        coeff = np.asarray(q_coeff, dtype=float)
        if coeff.shape != (phases.shape[1],):
            raise MagneticFieldError(
                "component phase coefficient length does not match sampled phase dimensions"
            )
        phase_turns = np.mod(phases @ coeff, 1.0)
        indices.append(np.rint(phase_turns * float(rows)).astype(int) % rows)
    return indices


def _sum_component_lookups(
    component_fields: Sequence[np.ndarray], indices: Sequence[np.ndarray], *, stride: int = 1
) -> np.ndarray:
    """Sum each component's lookup field at the rows :func:`_component_lookup_indices` chose."""

    sample_count = indices[0].shape[0] if indices else 0
    combined = np.zeros((sample_count, 3), dtype=float)
    for fields, rows in zip(component_fields, indices, strict=True):
        lookup = np.asarray(fields, dtype=float)
        if lookup.ndim != 2 or lookup.shape[1] != 3:
            raise MagneticFieldError(
                f"component field lookup must have shape (nangles, 3), got {lookup.shape}"
            )
        combined += lookup[:: int(stride)][rows]
    return combined


def _field_magnitude_histogram(
    magnitudes: np.ndarray, *, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a robust histogram for field magnitudes, including delta-like ranges."""

    bins = int(bins)
    if bins < 1:
        raise MagneticFieldError("histogram_bins must be positive")
    values = np.asarray(magnitudes, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size != values.size:
        raise MagneticFieldError("field distribution contains non-finite magnitudes")
    if finite.size == 0:
        raise MagneticFieldError("field distribution contains no samples")

    data_min = float(np.min(finite))
    data_max = float(np.max(finite))
    centre = 0.5 * (data_min + data_max)
    span = data_max - data_min
    min_span = max(abs(centre) * 1e-6, 1e-9)
    hist_range = None
    if span < min_span:
        half_span = 0.5 * min_span
        low = centre - half_span
        high = centre + half_span
        if data_min >= 0.0:
            low = max(0.0, low)
            if high <= low:
                high = low + min_span
        hist_range = (low, high)

    try:
        counts, edges = np.histogram(finite, bins=bins, range=hist_range)
    except ValueError as exc:
        if "Too many bins" not in str(exc):
            raise
        half_span = 0.5 * max(abs(centre) * 1e-6, 1e-9)
        low = max(0.0, centre - half_span) if data_min >= 0.0 else centre - half_span
        high = max(low + 2.0 * half_span, centre + half_span)
        counts, edges = np.histogram(finite, bins=bins, range=(low, high))
    centres = edges[:-1] + 0.5 * np.diff(edges)
    return centres, counts


def _contact_calculation_parameters(options: MagneticFieldOptions) -> dict[str, object]:
    return {
        "contact_coupling": (
            None if options.contact_coupling is None else float(options.contact_coupling)
        ),
        "nnn": int(options.nnn),
        "rcont": float(options.rcont),
    }


def _distribution_calculation_parameters(
    options: MagneticFieldOptions, radius: float
) -> dict[str, object]:
    return {
        "supercell": list(options.supercell),
        "radius": float(radius),
        "include_lorentz": bool(options.include_lorentz),
        "demag_shape": options.demag_shape,
        "plate_normal_axis": options.plate_normal_axis,
        "demag_sample": (
            None if options.demag_sample is None else options.demag_sample.summary_json()
        ),
        "measured_magnetization": _optional_list(
            _measured_magnetization_vector(options.measured_magnetization)
        ),
        "nnn": int(options.nnn),
        "rcont": float(options.rcont),
        "contact_coupling": (
            None if options.contact_coupling is None else float(options.contact_coupling)
        ),
        "applied_field": list(options.applied_field),
        "find_equivalent_sites": bool(options.find_equivalent_sites),
        "equivalent_tolerance": float(options.equivalent_tolerance),
        "commensurate_tolerance": float(options.commensurate_tolerance),
        "commensurate_max_period": int(options.commensurate_max_period),
        "nangles": int(options.nangles),
        "phase_sample_count": int(options.phase_sample_count),
        "phase_sampling": str(options.phase_sampling),
        "random_seed": options.random_seed,
        "max_grid_samples": int(options.max_grid_samples),
        "histogram_bins": int(options.histogram_bins),
    }


def _load_muesr_engines():
    try:
        with muesr_imports():
            from muesr.engines import find_largest_sphere, locfield
            from muesr.engines.clfc import dipten
    except ImportError:
        try:
            with muesr_imports():
                from muesr.engines.clfc import dipten, find_largest_sphere, locfield
        except ImportError as exc:
            raise MagneticFieldDependencyError(
                "muesr is required for periodic magnetic field calculations. "
                "Install muesr and muLFC to enable this workflow."
            ) from exc
    return locfield, find_largest_sphere, dipten


def _muesr_fourier_components(model_data: MuesrModelData, supercell: Sequence[int]) -> np.ndarray:
    """Return muesr Fourier components that measure the order from the muon's own cell.

    muesr places each muon in cell ``floor(supercell / 2)`` of its supercell but
    takes the Fourier phases ``k . R`` from the supercell origin, so its moments
    are those of MuFinder's convention translated by that cell. Multiplying every
    component by ``exp(2 pi i k . floor(supercell / 2))`` undoes the translation:
    the field at a muon site is the field at that site, whatever the supercell
    size. The phase goes into the components rather than ``mm.phi``, which keeps
    muLFC on its fully tested zero-phase path and keeps each circular helix
    circular.
    """

    muon_cell = np.floor(np.asarray(supercell, dtype=float) / 2.0)
    turns = float(np.dot(np.asarray(model_data.k_rlu, dtype=float), muon_cell))
    return np.asarray(model_data.fc, dtype=complex) * np.exp(2j * np.pi * turns)


def _build_muesr_sample(structure: Structure, muon_positions_frac: Sequence[Sequence[float]]):
    try:
        with muesr_imports():
            from muesr.core.atoms import Atoms as MuesrAtoms
            from muesr.core.sample import Sample as MuesrSample
    except ImportError as exc:
        raise MagneticFieldDependencyError(
            "muesr is required for periodic magnetic field calculations."
        ) from exc
    atoms = MuesrAtoms(
        symbols=list(structure.species),
        scaled_positions=np.asarray(structure.fractional_positions, dtype=float),
        cell=np.asarray(structure.lattice_vectors, dtype=float),
        pbc=[True, True, True],
    )
    sample = MuesrSample()
    sample.cell = atoms
    for position in muon_positions_frac:
        sample.add_muon(_canonical_fractional(position))
    return sample


def _expanded_muon_positions(
    structure: Structure,
    muon_sites: Sequence[MuonSite | Sequence[float]],
    options: MagneticFieldOptions,
) -> list[tuple[str, np.ndarray]]:
    expanded: list[tuple[str, np.ndarray]] = []
    for index, site in enumerate(muon_sites, start=1):
        if isinstance(site, MuonSite):
            label = site.label or f"site {index}"
            position = _canonical_fractional(site.current_fractional_position())
        else:
            label = f"site {index}"
            position = _canonical_fractional(site)
        clearance_error = muon_site_clearance_error(structure, position, label=label)
        if clearance_error is not None:
            raise MagneticFieldError(clearance_error)
        if not options.find_equivalent_sites:
            expanded.append((label, position))
            continue
        equivalents = find_equivalent_muon_sites(
            structure,
            position,
            tolerance=float(options.equivalent_tolerance),
        )
        for equiv_index, equiv in enumerate(equivalents, start=1):
            expanded.append((f"{label} equiv {equiv_index}", np.asarray(equiv, dtype=float)))
    if not expanded:
        raise MagneticFieldError("no muon positions provided")
    return expanded


def _site_distortions(
    structure: Structure,
    muon_sites: Sequence[MuonSite | Sequence[float]],
    distorted_structures: Sequence[Structure | None] | None,
    options: MagneticFieldOptions,
) -> list[MuonDistortion | None]:
    """Return one distortion pairing (or ``None``) per expanded muon position."""

    if distorted_structures is None:
        return [None] * len(_expanded_muon_positions(structure, muon_sites, options))
    if len(distorted_structures) != len(muon_sites):
        raise MagneticFieldError("distorted_structures must have one entry per muon site")
    out: list[MuonDistortion | None] = []
    for site, distorted in zip(muon_sites, distorted_structures, strict=True):
        positions = [
            position for _label, position in _expanded_muon_positions(structure, [site], options)
        ]
        if distorted is None:
            out.extend([None] * len(positions))
            continue
        representative = site.current_fractional_position() if isinstance(site, MuonSite) else site
        distortion = map_muon_distortion(
            structure, distorted, representative, error=MagneticFieldError
        )
        out.extend(
            equivalent_site_distortion(
                structure,
                distortion,
                position,
                tolerance=float(options.equivalent_tolerance),
                error=MagneticFieldError,
            )
            for position in positions
        )
    return out


def _resolve_magnetic_ion_indices(
    structure: Structure,
    *,
    magnetic_ion_indices: Sequence[int] | None,
    magnetic_ion_selector: AtomSelector | None,
) -> tuple[int, ...]:
    if magnetic_ion_indices is not None and magnetic_ion_selector is not None:
        raise MagneticFieldError(
            "provide either magnetic_ion_indices or magnetic_ion_selector, not both"
        )
    atom_count = len(structure.species)
    if magnetic_ion_selector is not None:
        indices = resolve_atom_selector(structure, magnetic_ion_selector)
    elif magnetic_ion_indices is not None:
        try:
            indices = tuple(sorted({int(index) for index in magnetic_ion_indices}))
        except (TypeError, ValueError) as exc:
            raise MagneticFieldError("magnetic ion indices must be integers") from exc
    else:
        indices = ()
    if not indices:
        raise MagneticFieldError(
            "choose at least one magnetic ion for the dipolar tensor calculation"
        )
    if any(index < 0 or index >= atom_count for index in indices):
        raise MagneticFieldError("magnetic ion selection contains an out-of-range atom index")
    return indices


def _unit_tensor_fourier_components(structure: Structure, indices: Sequence[int]) -> np.ndarray:
    fc = np.zeros((len(structure.species), 3), dtype=complex)
    fc[np.asarray(indices, dtype=int), 2] = 1.0 + 0.0j
    return fc


def _demag_field(
    structure: Structure, models: Sequence[MuesrModelData], options: MagneticFieldOptions
) -> np.ndarray:
    """Return the demagnetizing field ``-mu0 N M_meas`` of the bulk magnetisation, in T.

    ``M_meas`` is ``options.measured_magnetization`` when it is set and the
    model's uniform moment otherwise.
    """

    measured = _measured_magnetization_field(structure, options.measured_magnetization)
    tensor = _demag_tensor(options)
    if tensor is None:
        return np.zeros(3, dtype=float)
    bulk = measured if measured is not None else _bulk_magnetization_field(structure, models)
    return -(tensor @ bulk)


def _measured_magnetization_field(
    structure: Structure, magnetization: Sequence[float] | None
) -> np.ndarray | None:
    """Return ``mu0 M`` in T for a measured magnetisation in mu_B per formula unit.

    The formula unit is the reduced formula of the structure's non-muon atoms,
    so a supercell gives the same field as its primitive cell.
    """

    moment = _measured_magnetization_vector(magnetization)
    if moment is None:
        return None
    _formula, formula_units = reduced_formula(structure.species)
    if formula_units <= 0:
        raise MagneticFieldError(
            "measured_magnetization needs a structure with at least one non-muon atom"
        )
    volume = abs(float(np.linalg.det(np.asarray(structure.lattice_vectors, dtype=float))))
    return MAGNETIZATION_FIELD_T_PER_MUB_PER_ANGSTROM_MINUS3 * moment * formula_units / volume


def _measured_magnetization_vector(magnetization: object) -> np.ndarray | None:
    """Return a validated measured magnetisation in mu_B per formula unit, or None."""

    if magnetization is None:
        return None
    try:
        moment = np.asarray(magnetization, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MagneticFieldError(
            "measured_magnetization must be three numbers in mu_B per formula unit"
        ) from exc
    if moment.shape != (3,):
        raise MagneticFieldError(
            "measured_magnetization must be a three-vector in mu_B per formula unit"
        )
    if not np.all(np.isfinite(moment)):
        raise MagneticFieldError("measured_magnetization must be finite")
    return moment


def _optional_list(values: np.ndarray | None) -> list[float] | None:
    return None if values is None else [float(value) for value in values]


def _bulk_magnetization_field(structure: Structure, models: Sequence[MuesrModelData]) -> np.ndarray:
    """Return ``mu0 M`` in T for the moment per cell of the uniform part of the order.

    Only Fourier components whose wavevector is a reciprocal lattice vector give
    every cell the same moment; every other component averages to zero over the
    sample, so an antiferromagnet or a helix has no bulk magnetisation.
    """

    moment = np.zeros(3, dtype=float)
    for model_data in models:
        k_rlu = np.asarray(model_data.k_rlu, dtype=float)
        if np.allclose(k_rlu, np.round(k_rlu), rtol=0.0, atol=1e-10):
            moment += np.sum(np.asarray(model_data.fc, dtype=complex), axis=0).real
    volume = abs(float(np.linalg.det(np.asarray(structure.lattice_vectors, dtype=float))))
    return MAGNETIZATION_FIELD_T_PER_MUB_PER_ANGSTROM_MINUS3 * moment / volume


def _demag_tensor(options: MagneticFieldOptions) -> np.ndarray | None:
    if options.demag_sample is not None:
        return demagnetizing_tensor(options.demag_sample)
    shape = options.demag_shape.strip().lower()
    if shape in {"", "none"}:
        return None
    if shape == "sphere":
        tensor = np.eye(3, dtype=float) / 3.0
    elif shape in {"infinite_plate", "plate", "infinite plate"}:
        axis = {"a": 0, "x": 0, "b": 1, "y": 1, "c": 2, "z": 2}.get(
            options.plate_normal_axis.lower(), 2
        )
        tensor = np.zeros((3, 3), dtype=float)
        tensor[axis, axis] = 1.0
    else:
        raise MagneticFieldError(f"unknown demagnetizing shape '{options.demag_shape}'")
    return tensor


def _field_vector(local_field: object, primary: str, fallback: str) -> np.ndarray:
    value = getattr(local_field, primary, None)
    if value is None:
        value = getattr(local_field, fallback, None)
    if value is None:
        return np.zeros(3, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 2:
        arr = arr[0]
    return arr.reshape(3)


def _field_array(local_field: object, primary: str, fallback: str, *, nangles: int) -> np.ndarray:
    value = getattr(local_field, primary, None)
    if value is None:
        value = getattr(local_field, fallback, None)
    if value is None:
        return np.zeros((nangles, 3), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3,):
        return np.repeat(arr[np.newaxis, :], nangles, axis=0)
    if arr.ndim == 2 and arr.shape[1] == 3:
        if arr.shape[0] == nangles:
            return arr
        if arr.shape[0] == 1:
            return np.repeat(arr, nangles, axis=0)
    raise MagneticFieldError(f"muesr returned unexpected field shape {arr.shape}")


def _fraction_text(value: float, denominator: int) -> str:
    numerator = int(round(float(value) * int(denominator)))
    if numerator == 0:
        return "0"
    if int(denominator) == 1:
        return str(numerator)
    return f"{numerator}/{int(denominator)}"


def _canonical_fractional(position: Sequence[float]) -> np.ndarray:
    arr = _as_vector(position, "muon position")
    out = np.mod(arr, 1.0)
    out[np.isclose(out, 1.0, atol=1e-10)] = 0.0
    return out


def _as_vector(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3,):
        raise MagneticFieldError(f"{name} must be a three-vector")
    return arr


def _tuple3(values: Sequence[float]) -> tuple[float, float, float]:
    arr = _as_vector(values, "vector")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _tensor3x3(
    values: Sequence[Sequence[float]],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (3, 3):
        raise MagneticFieldError(f"muesr returned unexpected dipolar tensor shape {arr.shape}")
    return (
        (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
        (float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
        (float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2])),
    )
