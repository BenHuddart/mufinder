"""Magnetic field calculation helpers for MuFinder."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mufinder.domain.models import MuonSite, Structure
from mufinder.science.spglib_compat import spglib_cell, symmetry_operations, symmetry_orbit
from mufinder.science.demagnetization import DemagnetizingSample, demagnetizing_tensor
from mufinder.science._muesr_compat import muesr_imports
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    get_muesr_model_data,
    moments_at_atoms,
    resolve_atom_selector,
    split_muesr_model_data_into_helices,
)
from mufinder.science.muon_distortion import MuonDistortion, equivalent_site_distortion, map_muon_distortion

MU0_OVER_4PI = 1e-7
MU0 = 4.0 * np.pi * 1e-7
MU_B = 9.274_010_078_3e-24
ANGSTROM_TO_M = 1e-10
MUON_GYROMAGNETIC_RATIO_MHZ_PER_T = 135.538_96
CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB = (2.0 * MU0 / 3.0) * MU_B / (ANGSTROM_TO_M**3)
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
    "MHz_per_muB": (1.0 / MUON_GYROMAGNETIC_RATIO_MHZ_PER_T) / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
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
    nnn: int = 2
    rcont: float = 10.0
    contact_coupling: float | None = None
    applied_field: tuple[float, float, float] = (0.0, 0.0, 0.0)
    find_equivalent_sites: bool = False
    equivalent_tolerance: float = 1e-3
    commensurate_tolerance: float = 1e-8
    commensurate_max_period: int = 64
    nangles: int = 4096
    phase_sample_count: int = 200_000
    phase_sampling: str = "auto"
    random_seed: int | None = None
    max_grid_samples: int = 1_000_000
    histogram_bins: int = 512


def contact_coupling_to_angstrom_minus3(value: float, unit: str) -> float:
    """Convert a contact hyperfine coupling value to muesr's ACont unit."""

    unit_key = str(unit).strip()
    factor = _CONTACT_COUPLING_FACTORS_TO_ANGSTROM_MINUS3.get(unit_key)
    if factor is None:
        raise MagneticFieldError(f"unknown contact coupling unit '{unit}'")
    return float(value) * factor


@dataclass(frozen=True, slots=True)
class MagneticFieldResult:
    """One vector magnetic-field result in Tesla."""

    site_label: str
    muon_position: tuple[float, float, float]
    dipolar: tuple[float, float, float]
    lorentz: tuple[float, float, float]
    contact: tuple[float, float, float]
    demag: tuple[float, float, float]
    applied: tuple[float, float, float]
    distortion: tuple[float, float, float] | None = None

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
        return payload


@dataclass(frozen=True, slots=True)
class MagneticFieldDistributionResult:
    """Field distribution for an incommensurate magnetic structure."""

    site_label: str
    muon_position: tuple[float, float, float]
    fields_T: np.ndarray
    histogram_centres_T: np.ndarray
    histogram_counts: np.ndarray
    calculation_parameters: dict[str, object] = field(default_factory=dict)

    @property
    def magnitudes_T(self) -> np.ndarray:
        return np.linalg.norm(self.fields_T, axis=1)

    def to_json(self) -> dict[str, object]:
        return {
            "site_label": self.site_label,
            "muon_position": list(self.muon_position),
            "fields_T": np.asarray(self.fields_T, dtype=float).tolist(),
            "histogram_centres_T": np.asarray(self.histogram_centres_T, dtype=float).tolist(),
            "histogram_counts": np.asarray(self.histogram_counts, dtype=float).tolist(),
            "mean_T": np.mean(self.fields_T, axis=0).tolist(),
            "std_T": np.std(self.fields_T, axis=0).tolist(),
            "calculation_parameters": dict(self.calculation_parameters),
        }


@dataclass(frozen=True, slots=True)
class DipolarTensorResult:
    """One dipolar tensor result in T/mu_B."""

    site_label: str
    muon_position: tuple[float, float, float]
    tensor_T_per_muB: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]

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
        raise MagneticFieldError("spglib could not determine symmetry operations: a species is not an element")
    operations = symmetry_operations(cell, symprec=1e-5)
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
    r_norm_m = r_norm_ang[mask] * ANGSTROM_TO_M
    r_hat = r_cart / r_norm_ang[mask, np.newaxis]
    m_si = moments * MU_B
    inv_r3 = 1.0 / (r_norm_m**3)
    mdotr = np.sum(m_si * r_hat, axis=1)
    terms = (3.0 * mdotr[:, np.newaxis] * r_hat - m_si) * inv_r3[:, np.newaxis]
    return _tuple3(MU0_OVER_4PI * np.sum(terms, axis=0))


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

    moments = moments_at_atoms(structure, model, distortion.atom_indices, distortion.cell_offsets)
    origin = np.zeros(3, dtype=float)
    distorted = calculate_finite_dipolar_field(distortion.distorted_vectors, moments, origin)
    pristine = calculate_finite_dipolar_field(distortion.pristine_vectors, moments, origin)
    return _tuple3(np.asarray(distorted, dtype=float) - np.asarray(pristine, dtype=float))


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
    radius = float(opts.radius) if opts.radius is not None else float(find_largest_sphere(sample, list(opts.supercell)))

    dipolar = [np.zeros(3, dtype=float) for _ in muons]
    lorentz = [np.zeros(3, dtype=float) for _ in muons]
    contact = [np.zeros(3, dtype=float) for _ in muons]
    for model_data in models:
        sample = _build_muesr_sample(structure, [position for _label, position in muons])
        sample.new_mm()
        sample.mm.k = model_data.k_rlu
        sample.mm.fc = model_data.fc
        sample.mm.phi = model_data.phi
        fields = locfield(sample, "s", list(opts.supercell), radius, nnn=int(opts.nnn), rcont=float(opts.rcont))
        for index, local_field in enumerate(fields):
            if opts.contact_coupling is not None:
                local_field.ACont = float(opts.contact_coupling)
            dipolar[index] += _field_vector(local_field, "Dipolar", "D")
            if opts.include_lorentz:
                lorentz[index] += _field_vector(local_field, "Lorentz", "L")
            contact[index] += _field_vector(local_field, "Contact", "C")

    applied = _as_vector(opts.applied_field, "applied_field")
    results: list[MagneticFieldResult] = []
    for index, (label, position) in enumerate(muons):
        demag = _demag_field(lorentz[index], opts)
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
                    None if distortion is None else calculate_distortion_field_correction(structure, model, distortion)
                ),
            )
        )
    return tuple(results)


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
    radius = float(opts.radius) if opts.radius is not None else float(find_largest_sphere(radius_sample, list(opts.supercell)))
    sample = _build_muesr_sample(structure, [position for _label, position in muons])
    sample.new_mm()
    sample.mm.k = np.zeros(3, dtype=float)
    sample.mm.fc = _unit_tensor_fourier_components(structure, indices)
    sample.mm.phi = np.zeros(len(structure.species), dtype=float)
    tensors = [np.asarray(tensor, dtype=float) for tensor in dipten(sample, list(opts.supercell), radius)]
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

    Muon-induced distortion corrections are not available for distributions;
    passing any distorted structure raises :class:`MagneticFieldError`.
    """

    opts = options or MagneticFieldOptions()
    if distorted_structures is not None and any(item is not None for item in distorted_structures):
        raise MagneticFieldError(
            "muon-induced distortion corrections are available for commensurate magnetic structures only"
        )
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
    radius_sample = _build_muesr_sample(structure, [position for _label, position in muons])
    radius = float(opts.radius) if opts.radius is not None else float(find_largest_sphere(radius_sample, list(opts.supercell)))
    calculation_parameters = _distribution_calculation_parameters(opts, radius)
    phase_basis_rlu, phase_coeffs_by_model = _active_phase_basis_and_coefficients(
        models,
        tolerance=opts.commensurate_tolerance,
        max_period=opts.commensurate_max_period,
    )
    active_q_indices = _active_component_q_indices(models)
    phase_samples_turns = _build_phase_samples_turns(
        n_dim=phase_basis_rlu.shape[0],
        nangles=int(opts.nangles),
        phase_sample_count=int(opts.phase_sample_count),
        sampling=opts.phase_sampling,
        random_seed=opts.random_seed,
        max_grid_samples=int(opts.max_grid_samples),
    )
    calculation_parameters["independent_q_indices"] = list(active_q_indices)
    calculation_parameters["phase_basis_rlu"] = phase_basis_rlu.tolist()
    calculation_parameters["phase_sample_count"] = int(phase_samples_turns.shape[0])
    component_totals_by_site: list[list[np.ndarray]] = [[] for _ in muons]
    component_lorentz_by_site: list[list[np.ndarray]] = [[] for _ in muons]
    component_phase_coeffs: list[tuple[float, ...]] = []

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
        model_entries = split_muesr_model_data_into_helices(model_data) if mode == "i" else (model_data,)
        for model_entry in model_entries:
            sample = _build_muesr_sample(structure, [position for _label, position in muons])
            sample.new_mm()
            sample.mm.k = model_entry.k_rlu
            sample.mm.fc = model_entry.fc
            sample.mm.phi = model_entry.phi
            kwargs: dict[str, Any] = {}
            if mode == "i":
                kwargs["nangles"] = int(opts.nangles)
            fields = locfield(sample, mode, list(opts.supercell), radius, nnn=int(opts.nnn), rcont=float(opts.rcont), **kwargs)
            for index, local_field in enumerate(fields):
                if opts.contact_coupling is not None:
                    local_field.ACont = float(opts.contact_coupling)
                dip = _field_array(local_field, "Dipolar", "D", nangles=int(opts.nangles))
                lor = _field_array(local_field, "Lorentz", "L", nangles=int(opts.nangles)) if opts.include_lorentz else 0.0
                con = _field_array(local_field, "Contact", "C", nangles=int(opts.nangles))
                component_totals_by_site[index].append(dip + lor + con)
                component_lorentz_by_site[index].append(lor if isinstance(lor, np.ndarray) else np.zeros((int(opts.nangles), 3), dtype=float))
            component_phase_coeffs.append(tuple(float(value) for value in phase_coeff))

    applied = _as_vector(opts.applied_field, "applied_field")
    demag_tensor = _demag_tensor(opts)
    results: list[MagneticFieldDistributionResult] = []
    for index, (label, position) in enumerate(muons):
        fields = _combine_component_samples(
            component_totals_by_site[index],
            component_phase_coeffs,
            phase_samples_turns,
            nangles=int(opts.nangles),
        )
        if demag_tensor is not None:
            lorentz_fields = _combine_component_samples(
                component_lorentz_by_site[index],
                component_phase_coeffs,
                phase_samples_turns,
                nangles=int(opts.nangles),
            )
            fields = fields - 1.5 * (lorentz_fields @ demag_tensor.T)
        fields = fields + applied[np.newaxis, :]
        magnitudes = np.linalg.norm(fields, axis=1)
        centres, counts = _field_magnitude_histogram(magnitudes, bins=int(opts.histogram_bins))
        results.append(
            MagneticFieldDistributionResult(
                site_label=label,
                muon_position=_tuple3(position),
                fields_T=fields,
                histogram_centres_T=centres,
                histogram_counts=counts,
                calculation_parameters=calculation_parameters,
            )
        )
    return tuple(results)


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
            raise MagneticFieldError("could not express component q vector in the sampled phase basis")
        coeffs.append(tuple(float(value) for value in solved))
    return basis, coeffs


def _rank_increases(existing: Sequence[np.ndarray], candidate: np.ndarray, *, tolerance: float) -> bool:
    candidate = np.asarray(candidate, dtype=float).reshape(3)
    if float(np.linalg.norm(candidate)) <= float(tolerance):
        return False
    if not existing:
        return True
    before = np.linalg.matrix_rank(np.asarray(existing, dtype=float), tol=float(tolerance))
    after = np.linalg.matrix_rank(np.vstack([np.asarray(existing, dtype=float), candidate]), tol=float(tolerance))
    return int(after) > int(before)


def _active_component_q_indices(models: Sequence[object], *, tolerance: float = 1e-12) -> tuple[int, ...]:
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

    mode = str(sampling).strip().lower()
    if mode not in {"auto", "grid", "montecarlo", "monte_carlo", "random"}:
        raise MagneticFieldError("phase_sampling must be one of 'auto', 'grid', or 'montecarlo'")
    if mode in {"monte_carlo", "random"}:
        mode = "montecarlo"
    if mode == "auto":
        grid_axis_count = _grid_axis_count(
            n_dim=n_dim,
            nangles=int(nangles),
            phase_sample_count=int(phase_sample_count),
        )
        grid_count = grid_axis_count**n_dim
        mode = "grid" if n_dim <= 2 and grid_count <= int(max_grid_samples) else "montecarlo"

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
        axis_values = np.arange(axis_count, dtype=float) / float(axis_count)
        meshes = np.meshgrid(*([axis_values] * n_dim), indexing="ij")
        coords = np.stack([mesh.ravel() for mesh in meshes], axis=1)
    else:
        rng = np.random.default_rng(seed=random_seed)
        coords = rng.random((int(phase_sample_count), n_dim), dtype=float)

    return coords


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
) -> np.ndarray:
    phases = np.asarray(phase_samples_turns, dtype=float)
    if phases.ndim != 2:
        raise MagneticFieldError("phase_samples_turns must have shape (n, n_phase_dimensions)")
    combined = np.zeros((phases.shape[0], 3), dtype=float)
    for fields, q_coeff in zip(component_fields, component_q_coeffs, strict=True):
        lookup = np.asarray(fields, dtype=float)
        if lookup.ndim != 2 or lookup.shape[1] != 3:
            raise MagneticFieldError(f"component field lookup must have shape (nangles, 3), got {lookup.shape}")
        coeff = np.asarray(q_coeff, dtype=float)
        if coeff.shape != (phases.shape[1],):
            raise MagneticFieldError("component phase coefficient length does not match sampled phase dimensions")
        phase_turns = np.mod(phases @ coeff, 1.0)
        indices = np.rint(phase_turns * float(nangles)).astype(int) % int(nangles)
        combined += lookup[indices]
    return combined


def _field_magnitude_histogram(magnitudes: np.ndarray, *, bins: int) -> tuple[np.ndarray, np.ndarray]:
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


def _distribution_calculation_parameters(options: MagneticFieldOptions, radius: float) -> dict[str, object]:
    return {
        "supercell": list(options.supercell),
        "radius": float(radius),
        "include_lorentz": bool(options.include_lorentz),
        "demag_shape": options.demag_shape,
        "plate_normal_axis": options.plate_normal_axis,
        "demag_sample": None if options.demag_sample is None else options.demag_sample.to_json(),
        "nnn": int(options.nnn),
        "rcont": float(options.rcont),
        "contact_coupling": None if options.contact_coupling is None else float(options.contact_coupling),
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
            position = _canonical_fractional(site.fractional_position)
        else:
            label = f"site {index}"
            position = _canonical_fractional(site)
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
        positions = [position for _label, position in _expanded_muon_positions(structure, [site], options)]
        if distorted is None:
            out.extend([None] * len(positions))
            continue
        representative = site.fractional_position if isinstance(site, MuonSite) else site
        distortion = map_muon_distortion(structure, distorted, representative, error=MagneticFieldError)
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
        raise MagneticFieldError("provide either magnetic_ion_indices or magnetic_ion_selector, not both")
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
        raise MagneticFieldError("choose at least one magnetic ion for the dipolar tensor calculation")
    if any(index < 0 or index >= atom_count for index in indices):
        raise MagneticFieldError("magnetic ion selection contains an out-of-range atom index")
    return indices


def _unit_tensor_fourier_components(structure: Structure, indices: Sequence[int]) -> np.ndarray:
    fc = np.zeros((len(structure.species), 3), dtype=complex)
    fc[np.asarray(indices, dtype=int), 2] = 1.0 + 0.0j
    return fc


def _demag_field(lorentz: np.ndarray, options: MagneticFieldOptions) -> np.ndarray:
    tensor = _demag_tensor(options)
    if tensor is None:
        return np.zeros(3, dtype=float)
    return -1.5 * (tensor @ np.asarray(lorentz, dtype=float))


def _demag_tensor(options: MagneticFieldOptions) -> np.ndarray | None:
    if options.demag_sample is not None:
        return demagnetizing_tensor(options.demag_sample)
    shape = options.demag_shape.strip().lower()
    if shape in {"", "none"}:
        return None
    if shape == "sphere":
        tensor = np.eye(3, dtype=float) / 3.0
    elif shape in {"infinite_plate", "plate", "infinite plate"}:
        axis = {"a": 0, "x": 0, "b": 1, "y": 1, "c": 2, "z": 2}.get(options.plate_normal_axis.lower(), 2)
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


def _tensor3x3(values: Sequence[Sequence[float]]) -> tuple[
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
