"""Nuclear dipolar relaxation helpers for MuFinder."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from configparser import ConfigParser
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from importlib import resources
from typing import Literal

from ase import Atoms
import numpy as np

from mufinder.domain.constants import (
    HBAR_J_S,
    METRES_PER_ANGSTROM,
    MU0_OVER_4PI_T_M_PER_A,
    MUON_GAMMA_RAD_S_T,
    NUCLEAR_MAGNETON_J_PER_T,
)
from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import element_symbol, structure_to_ase_atoms
from mufinder.science.lattice import (
    MUON_FRACTIONAL_SNAP_TOLERANCE,
    cartesian_to_fractional,
    cell_volume,
    lattice_plane_heights,
    nearest_image_distance_matrix,
    reduced_lattice,
    wrap_fractional,
)
from mufinder.science.muon_distortion import MuonDistortion, map_muon_distortion

SEC2_TO_US2 = 1e-12

#: Radius (Å) of the explicit lattice sum around the muon. The nuclei beyond it
#: enter through the analytic continuum tail (:func:`_tail_moments`).
NUCLEAR_SUM_RADIUS_A = 20.0
#: Each element's sum reaches at least this many mean spacings of its nuclei, which holds
#: its continuum tail to about 0.05 % in σ when they are too sparse for 20 Å (one Cu in 16 Å).
_SUM_RADIUS_MIN_SPACINGS = 5.0

NuclearRelaxationMode = Literal[
    "tf_single_crystal",
    "zf_single_crystal",
    "tf_polycrystal",
    "zf_polycrystal",
]

_ALLOWED_MODES = {
    "tf_single_crystal",
    "zf_single_crystal",
    "tf_polycrystal",
    "zf_polycrystal",
}


class NuclearRelaxationError(ValueError):
    """Raised when nuclear relaxation inputs are invalid."""


@dataclass(frozen=True, slots=True)
class Isotope:
    """Nuclear isotope data used for second-moment calculations."""

    element: str
    mass_number: int | None
    abundance: float
    spin: float
    moment_mu_n: float
    quadrupole_1e28_m2: float
    gamma_rad_s_t: float

    def to_json(self) -> dict[str, object]:
        return {
            "element": self.element,
            "mass_number": self.mass_number,
            "abundance": self.abundance,
            "spin": self.spin,
            "moment_mu_n": self.moment_mu_n,
            "quadrupole_1e28_m2": self.quadrupole_1e28_m2,
            "gamma_rad_s_t": self.gamma_rad_s_t,
        }


@dataclass(frozen=True, slots=True)
class NuclearRelaxationOptions:
    """Options for nuclear second-moment calculations."""

    mode: NuclearRelaxationMode = "zf_polycrystal"
    direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    include_quadrupolar: bool = False
    elements: tuple[str, ...] | None = None
    use_distortion_correction: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "direction": list(self.direction),
            "include_quadrupolar": self.include_quadrupolar,
            "elements": None if self.elements is None else list(self.elements),
            "use_distortion_correction": self.use_distortion_correction,
        }


@dataclass(frozen=True, slots=True)
class NuclearRelaxationResult:
    """Main nuclear relaxation result for one muon site."""

    site_label: str
    muon_position: tuple[float, float, float]
    sigma: float
    sigma_squared: float
    m2: float
    delta_field_t: float
    mode: str
    convergence_radius_a: float
    n_nuclei_summed: int
    per_element_sigma_squared: dict[str, float]
    sigma_perp_squared: float | None = None
    sigma_parallel_squared: float | None = None
    distortion_correction: "DistortionCorrectionResult | None" = None

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "site_label": self.site_label,
            "muon_position": list(self.muon_position),
            "sigma": self.sigma,
            "sigma_squared": self.sigma_squared,
            "m2": self.m2,
            "delta_field_t": self.delta_field_t,
            "mode": self.mode,
            "convergence_radius_a": self.convergence_radius_a,
            "n_nuclei_summed": self.n_nuclei_summed,
            "per_element_sigma_squared": dict(self.per_element_sigma_squared),
            "sigma_perp_squared": self.sigma_perp_squared,
            "sigma_parallel_squared": self.sigma_parallel_squared,
        }
        if self.distortion_correction is not None:
            payload["distortion_correction"] = self.distortion_correction.to_json()
        return payload


@dataclass(frozen=True, slots=True)
class DistortionCorrectionResult:
    """Diagnostic payload for muon-induced distortion corrections."""

    mode: str
    supercell_matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    sigma_squared_undistorted: float
    sigma_squared_pristine_supercell: float
    sigma_squared_distorted_supercell: float
    distortion_correction_sigma_squared: float
    sigma_squared_corrected: float
    sigma_corrected: float
    n_atoms_unwrapped: int
    undistorted_result: NuclearRelaxationResult
    pristine_supercell_result: NuclearRelaxationResult
    distorted_supercell_result: NuclearRelaxationResult

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "supercell_matrix": [list(row) for row in self.supercell_matrix],
            "sigma_squared_undistorted": self.sigma_squared_undistorted,
            "sigma_squared_pristine_supercell": self.sigma_squared_pristine_supercell,
            "sigma_squared_distorted_supercell": self.sigma_squared_distorted_supercell,
            "distortion_correction_sigma_squared": self.distortion_correction_sigma_squared,
            "sigma_squared_corrected": self.sigma_squared_corrected,
            "sigma_corrected": self.sigma_corrected,
            "n_atoms_unwrapped": self.n_atoms_unwrapped,
            "undistorted_result": self.undistorted_result.to_json(),
            "pristine_supercell_result": self.pristine_supercell_result.to_json(),
            "distorted_supercell_result": self.distorted_supercell_result.to_json(),
        }


@dataclass(frozen=True, slots=True)
class NuclearWidthTensors:
    """Moment tensors of a nuclear lattice sum, giving the widths along any direction.

    The single-crystal widths are polynomials in the direction: the ZF widths
    are quadratic in it and the TF width quartic. One sum of
    ``S = Σ w / r⁶``, ``M_ij = Σ w r̂_i r̂_j / r⁶`` and
    ``M_ijkl = Σ w r̂_i r̂_j r̂_k r̂_l / r⁶`` over the nuclei, with ``w`` the
    isotope-weighted ``γ² I(I+1)``, therefore gives every direction at once.
    Each tensor already carries the prefactor ``(μ₀/4π)² ħ² γ_μ² / 3``, so the
    widths come out in μs⁻² without further factors.

    ``isotropic``, ``quadratic`` and ``quartic`` use the plain weights, for the
    TF width; ``zf_isotropic`` and ``zf_quadratic`` use the quadrupolar
    weights when ``include_quadrupolar`` is set, and equal the plain ones
    otherwise.
    """

    site_label: str
    muon_position: tuple[float, float, float]
    isotropic: float
    quadratic: np.ndarray
    quartic: np.ndarray
    zf_isotropic: float
    zf_quadratic: np.ndarray
    include_quadrupolar: bool
    convergence_radius_a: float
    n_nuclei_summed: int

    def tf_sigma_squared(self, direction: Sequence[float] | np.ndarray) -> np.ndarray | float:
        """Return σ²_TF (μs⁻²) for a field along ``direction``, or along each row of an array.

        This is the secular second moment of the nuclear field components
        along the field, ``9 n n n n M₄ − 6 n n M₂ + S``.
        """

        n, scalar = _unit_directions(direction)
        quadratic = np.einsum("...i,ij,...j->...", n, self.quadratic, n)
        quartic = np.einsum("ijkl,...k,...l->...ij", self.quartic, n, n)
        quartic = np.einsum("...i,...ij,...j->...", n, quartic, n)
        values = 9.0 * quartic - 6.0 * quadratic + self.isotropic
        return float(values) if scalar else values

    def zf_sigma_squared(self, direction: Sequence[float] | np.ndarray) -> np.ndarray | float:
        """Return the zero-field Kubo-Toyabe σ⊥² (μs⁻²) for polarisation along ``direction``."""

        n, scalar = _unit_directions(direction)
        quadratic = np.einsum("...i,ij,...j->...", n, self.zf_quadratic, n)
        values = 0.5 * (5.0 * self.zf_isotropic - 3.0 * quadratic)
        return float(values) if scalar else values

    def zf_parallel_sigma_squared(
        self, direction: Sequence[float] | np.ndarray
    ) -> np.ndarray | float:
        """Return σ∥² (μs⁻²), the field variance along a polarisation ``direction``."""

        n, scalar = _unit_directions(direction)
        quadratic = np.einsum("...i,ij,...j->...", n, self.zf_quadratic, n)
        values = 3.0 * quadratic + self.zf_isotropic
        return float(values) if scalar else values

    @property
    def tf_polycrystal_sigma_squared(self) -> float:
        """The sphere average of σ²_TF, the TF polycrystal width (μs⁻²)."""

        return 0.8 * self.isotropic

    @property
    def zf_polycrystal_sigma_squared(self) -> float:
        """The sphere average of σ⊥², the ZF polycrystal width (μs⁻²)."""

        return 2.0 * self.zf_isotropic

    def to_json(self) -> dict[str, object]:
        return {
            "site_label": self.site_label,
            "muon_position": list(self.muon_position),
            "isotropic": float(self.isotropic),
            "quadratic": np.asarray(self.quadratic, dtype=float).tolist(),
            "quartic": np.asarray(self.quartic, dtype=float).tolist(),
            "zf_isotropic": float(self.zf_isotropic),
            "zf_quadratic": np.asarray(self.zf_quadratic, dtype=float).tolist(),
            "include_quadrupolar": bool(self.include_quadrupolar),
            "convergence_radius_a": float(self.convergence_radius_a),
            "n_nuclei_summed": int(self.n_nuclei_summed),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, object]) -> NuclearWidthTensors:
        try:
            quadratic = np.asarray(payload["quadratic"], dtype=float).reshape(3, 3)
            quartic = np.asarray(payload["quartic"], dtype=float).reshape(3, 3, 3, 3)
            zf_quadratic = np.asarray(payload["zf_quadratic"], dtype=float).reshape(3, 3)
            position = np.asarray(payload["muon_position"], dtype=float).reshape(3)
            return cls(
                site_label=str(payload.get("site_label", "")),
                muon_position=(float(position[0]), float(position[1]), float(position[2])),
                isotropic=float(payload["isotropic"]),
                quadratic=quadratic,
                quartic=quartic,
                zf_isotropic=float(payload["zf_isotropic"]),
                zf_quadratic=zf_quadratic,
                include_quadrupolar=bool(payload.get("include_quadrupolar", False)),
                convergence_radius_a=float(payload.get("convergence_radius_a", 0.0)),
                n_nuclei_summed=int(payload.get("n_nuclei_summed", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NuclearRelaxationError(f"malformed nuclear width tensors: {exc}") from exc


def gyromagnetic_ratio_from_moment(moment_mu_n: float, spin: float) -> float:
    """Return nuclear gyromagnetic ratio in rad/(s*T)."""

    if spin <= 0.0:
        return 0.0
    return (float(moment_mu_n) * NUCLEAR_MAGNETON_J_PER_T) / (HBAR_J_S * float(spin))


@lru_cache(maxsize=1)
def load_isotopes() -> dict[str, list[Isotope]]:
    """Load bundled natural-abundance isotope data."""

    text = resources.files("mufinder.data").joinpath("isotopes.ini").read_text(encoding="utf-8")
    parser = ConfigParser()
    parser.optionxform = str
    parser.read_string(text)
    data: dict[str, list[Isotope]] = {}
    for element in parser.sections():
        values = parser[element]
        abundances = _parse_csv_numbers(values.get("abundances", ""))
        spins = _parse_csv_numbers(values.get("I", ""))
        moments = _parse_csv_numbers(values.get("moments", ""))
        quadrupoles = _parse_csv_numbers(values.get("Q", ""))
        masses = _parse_csv_ints(values.get("masses", ""))

        n_iso = max(len(abundances), len(spins), len(moments), len(quadrupoles), len(masses))
        if n_iso == 0:
            data[element] = []
            continue
        abundances = _pad(abundances, n_iso, 0.0)
        spins = _pad(spins, n_iso, 0.0)
        moments = _pad(moments, n_iso, 0.0)
        quadrupoles = _pad(quadrupoles, n_iso, 0.0)
        masses = _pad(masses, n_iso, None)

        isotopes: list[Isotope] = []
        for index in range(n_iso):
            spin = float(spins[index])
            moment = float(moments[index])
            isotopes.append(
                Isotope(
                    element=element,
                    mass_number=masses[index],
                    abundance=float(abundances[index]) / 100.0,
                    spin=spin,
                    moment_mu_n=moment,
                    quadrupole_1e28_m2=float(quadrupoles[index]),
                    gamma_rad_s_t=gyromagnetic_ratio_from_moment(moment, spin),
                )
            )
        data[element] = isotopes
    return data


def isotopes_for_element(element: str) -> list[Isotope]:
    """Return isotope entries for one element symbol."""

    key = element_symbol(element, error=NuclearRelaxationError)
    try:
        return load_isotopes()[key]
    except KeyError as exc:
        raise NuclearRelaxationError(f"No isotope data available for element '{key}'") from exc


def weighted_spin_factor(
    isotopes: Iterable[Isotope], *, include_quadrupolar: bool = False
) -> float:
    """Return abundance-weighted gamma^2 I(I+1) for one element."""

    total = 0.0
    for iso in isotopes:
        if iso.spin <= 0.0 or iso.abundance <= 0.0:
            continue
        factor = 1.0
        if include_quadrupolar and iso.spin > 0.5:
            if abs(iso.spin - round(iso.spin)) < 1e-12:
                factor = 2.0 / 3.0
            else:
                i = float(iso.spin)
                factor = (16.0 * i * (i + 1.0) + 3.0 * (2.0 * i + 1.0)) / (24.0 * i * (i + 1.0))
        total += iso.abundance * factor * (iso.gamma_rad_s_t**2) * iso.spin * (iso.spin + 1.0)
    return float(total)


def calculate_nuclear_relaxation(
    structure: Structure,
    muon_position: Sequence[float],
    *,
    site_label: str = "site 1",
    options: NuclearRelaxationOptions | None = None,
) -> NuclearRelaxationResult:
    """Calculate an undistorted periodic nuclear relaxation width."""

    opts = options or NuclearRelaxationOptions()
    atoms = structure_to_atoms(structure, skip_muon=True)
    return _calculate_second_moment(
        atoms,
        muon_position,
        opts.mode,
        site_label=site_label,
        field_direction=opts.direction,
        polarisation_direction=opts.direction,
        muon_position_fractional=True,
        include_quadrupolar=opts.include_quadrupolar,
        elements=opts.elements,
    )


def calculate_nuclear_width_tensors(
    structure: Structure,
    muon_position: Sequence[float],
    *,
    site_label: str = "site 1",
    include_quadrupolar: bool = False,
    elements: Sequence[str] | None = None,
) -> NuclearWidthTensors:
    """Sum the nuclear moment tensors at a fractional ``muon_position`` once.

    The sum runs over the undistorted periodic host, like
    :func:`calculate_nuclear_relaxation`: explicitly to
    :data:`NUCLEAR_SUM_RADIUS_A`, or further for sparse nuclei, plus the isotropic
    continuum tail beyond it.
    """

    atoms = structure_to_atoms(structure, skip_muon=True)
    selected = _selected_elements(elements)
    muon_cart = _muon_position_cartesian(atoms, muon_position, True)
    radii = _sum_radii(atoms, muon_cart, selected)
    vectors, distances, symbols = _nuclear_vectors_within(atoms, muon_cart, radii)
    isotropic, quadratic, quartic, zf_isotropic, zf_quadratic = _moment_tensors(
        vectors, distances * METRES_PER_ANGSTROM, symbols, include_quadrupolar
    )
    prefactor = _prefactor_sigma_squared() * SEC2_TO_US2
    tail = prefactor * sum(_tail_moments(atoms, radii).values())
    zf_tail = (
        prefactor * sum(_tail_moments(atoms, radii, include_quadrupolar=True).values())
        if include_quadrupolar
        else tail
    )
    return NuclearWidthTensors(
        site_label=site_label,
        muon_position=tuple(
            wrap_fractional(
                np.asarray(muon_position, dtype=float).reshape(3),
                snap_tolerance=MUON_FRACTIONAL_SNAP_TOLERANCE,
            )
        ),
        isotropic=isotropic + tail,
        quadratic=quadratic + tail * _ISOTROPIC_QUADRATIC,
        quartic=quartic + tail * _ISOTROPIC_QUARTIC,
        zf_isotropic=zf_isotropic + zf_tail,
        zf_quadratic=zf_quadratic + zf_tail * _ISOTROPIC_QUADRATIC,
        include_quadrupolar=include_quadrupolar,
        convergence_radius_a=max(radii.values()),
        n_nuclei_summed=int(distances.size),
    )


#: Sphere averages of ``r̂_i r̂_j`` and ``r̂_i r̂_j r̂_k r̂_l``, a tail's share of each tensor.
_ISOTROPIC_QUADRATIC = np.eye(3) / 3.0
_ISOTROPIC_QUARTIC = (
    np.einsum("ij,kl->ijkl", np.eye(3), np.eye(3))
    + np.einsum("ik,jl->ijkl", np.eye(3), np.eye(3))
    + np.einsum("il,jk->ijkl", np.eye(3), np.eye(3))
) / 15.0


def _moment_tensors(
    vectors_a: np.ndarray,
    distances_m: np.ndarray,
    symbols: np.ndarray,
    include_quadrupolar: bool,
) -> tuple[float, np.ndarray, np.ndarray, float, np.ndarray]:
    unit = vectors_a / np.linalg.norm(vectors_a, axis=1)[:, None]
    base = _prefactor_sigma_squared() * SEC2_TO_US2 / distances_m**6
    weights = np.zeros_like(base)
    zf_weights = np.zeros_like(base)
    for element in np.unique(symbols):
        mask = symbols == element
        isotopes = isotopes_for_element(str(element))
        weights[mask] = weighted_spin_factor(isotopes)
        zf_weights[mask] = weighted_spin_factor(isotopes, include_quadrupolar=include_quadrupolar)
    terms = weights * base
    zf_terms = zf_weights * base
    return (
        float(np.sum(terms)),
        np.einsum("k,ki,kj->ij", terms, unit, unit),
        np.einsum("k,ki,kj,kl,km->ijlm", terms, unit, unit, unit, unit),
        float(np.sum(zf_terms)),
        np.einsum("k,ki,kj->ij", zf_terms, unit, unit),
    )


def calculate_nuclear_relaxation_with_distortion(
    pristine_structure: Structure,
    distorted_structure: Structure,
    muon_position: Sequence[float],
    *,
    site_label: str = "site 1",
    options: NuclearRelaxationOptions | None = None,
) -> NuclearRelaxationResult:
    """Calculate nuclear relaxation with a local distortion correction."""

    opts = options or NuclearRelaxationOptions(use_distortion_correction=True)
    correction = calculate_distortion_correction(
        pristine_structure,
        distorted_structure,
        muon_position,
        site_label=site_label,
        options=opts,
    )
    base = correction.undistorted_result
    corrected_parallel = None
    if (
        opts.mode == "zf_single_crystal"
        and base.sigma_parallel_squared is not None
        and correction.distorted_supercell_result.sigma_parallel_squared is not None
        and correction.pristine_supercell_result.sigma_parallel_squared is not None
    ):
        corrected_parallel = (
            base.sigma_parallel_squared
            + correction.distorted_supercell_result.sigma_parallel_squared
            - correction.pristine_supercell_result.sigma_parallel_squared
        )
    return NuclearRelaxationResult(
        site_label=base.site_label,
        muon_position=base.muon_position,
        sigma=correction.sigma_corrected,
        sigma_squared=correction.sigma_squared_corrected,
        m2=correction.sigma_squared_corrected
        if opts.mode.startswith("tf")
        else 2.0 * correction.sigma_squared_corrected,
        delta_field_t=float((correction.sigma_corrected * 1e6) / MUON_GAMMA_RAD_S_T),
        mode=opts.mode,
        convergence_radius_a=base.convergence_radius_a,
        n_nuclei_summed=base.n_nuclei_summed,
        per_element_sigma_squared=dict(base.per_element_sigma_squared),
        sigma_perp_squared=correction.sigma_squared_corrected
        if opts.mode == "zf_single_crystal"
        else None,
        sigma_parallel_squared=corrected_parallel,
        distortion_correction=correction,
    )


def calculate_distortion_correction(
    pristine_structure: Structure,
    distorted_structure: Structure,
    muon_position: Sequence[float],
    *,
    site_label: str = "site 1",
    options: NuclearRelaxationOptions | None = None,
) -> DistortionCorrectionResult:
    """Calculate a finite-supercell correction for muon-induced distortions."""

    opts = options or NuclearRelaxationOptions()
    distortion = map_muon_distortion(
        pristine_structure,
        distorted_structure,
        muon_position,
        error=NuclearRelaxationError,
    )
    muon_unit = distortion.muon_fractional
    distorted_cell = _finite_cell_atoms(distortion, distortion.distorted_vectors)
    pristine_cell = _finite_cell_atoms(distortion, distortion.pristine_vectors)

    finite_opts = NuclearRelaxationOptions(
        mode=opts.mode,
        direction=opts.direction,
        include_quadrupolar=opts.include_quadrupolar,
        elements=opts.elements,
        use_distortion_correction=False,
    )
    distorted_supercell_result = _calculate_second_moment_finite_cell(
        distorted_cell,
        (0.5, 0.5, 0.5),
        finite_opts.mode,
        site_label=f"{site_label} distorted supercell",
        field_direction=finite_opts.direction,
        polarisation_direction=finite_opts.direction,
        include_quadrupolar=finite_opts.include_quadrupolar,
        elements=finite_opts.elements,
    )
    pristine_supercell_result = _calculate_second_moment_finite_cell(
        pristine_cell,
        (0.5, 0.5, 0.5),
        finite_opts.mode,
        site_label=f"{site_label} pristine supercell",
        field_direction=finite_opts.direction,
        polarisation_direction=finite_opts.direction,
        include_quadrupolar=finite_opts.include_quadrupolar,
        elements=finite_opts.elements,
    )
    undistorted_result = calculate_nuclear_relaxation(
        pristine_structure,
        muon_unit,
        site_label=site_label,
        options=finite_opts,
    )

    correction_sigma2 = (
        distorted_supercell_result.sigma_squared - pristine_supercell_result.sigma_squared
    )
    corrected_sigma2 = undistorted_result.sigma_squared + correction_sigma2
    corrected_sigma = float(np.sqrt(max(corrected_sigma2, 0.0)))

    return DistortionCorrectionResult(
        mode=opts.mode,
        supercell_matrix=distortion.supercell_matrix,
        sigma_squared_undistorted=undistorted_result.sigma_squared,
        sigma_squared_pristine_supercell=pristine_supercell_result.sigma_squared,
        sigma_squared_distorted_supercell=distorted_supercell_result.sigma_squared,
        distortion_correction_sigma_squared=correction_sigma2,
        sigma_squared_corrected=corrected_sigma2,
        sigma_corrected=corrected_sigma,
        n_atoms_unwrapped=distortion.n_atoms_unwrapped,
        undistorted_result=undistorted_result,
        pristine_supercell_result=pristine_supercell_result,
        distorted_supercell_result=distorted_supercell_result,
    )


def structure_to_atoms(structure: Structure, *, skip_muon: bool = True) -> Atoms:
    """Convert a MuFinder structure to ASE Atoms with sanitized element labels.

    Compatibility wrapper over :func:`mufinder.science.ase_bridge.structure_to_ase_atoms`
    that keeps this module's muon-skipping default and error type.
    """

    atoms = structure_to_ase_atoms(
        structure,
        skip_muon=skip_muon,
        sanitize_symbols=True,
        record_species_labels=True,
        error=NuclearRelaxationError,
    )
    if len(atoms) == 0:
        raise NuclearRelaxationError("Structure does not contain any host atoms.")
    return atoms


def _calculate_second_moment(
    atoms: Atoms,
    muon_position: Sequence[float],
    mode: str,
    *,
    site_label: str,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    polarisation_direction: Sequence[float] = (0.0, 0.0, 1.0),
    muon_position_fractional: bool = True,
    include_quadrupolar: bool = False,
    elements: Sequence[str] | None = None,
) -> NuclearRelaxationResult:
    _validate_mode(mode, include_quadrupolar=include_quadrupolar)
    field_dir = _unit_vector(field_direction, "direction") if mode == "tf_single_crystal" else None
    pol_dir = (
        _unit_vector(polarisation_direction, "direction") if mode == "zf_single_crystal" else None
    )
    selected = _selected_elements(elements)
    muon_cart = _muon_position_cartesian(atoms, muon_position, muon_position_fractional)
    radii = _sum_radii(atoms, muon_cart, selected)
    vectors, distances, symbols = _nuclear_vectors_within(atoms, muon_cart, radii)
    sigma2, sigma2_parallel, per_element = _evaluate_mode(
        mode,
        vectors * METRES_PER_ANGSTROM,
        distances * METRES_PER_ANGSTROM,
        symbols,
        field_dir,
        pol_dir,
        include_quadrupolar,
    )
    # Every mode's angular factor averages over the sphere to its polycrystal
    # prefactor, so the isotropic tail adds the same amount to each width of a mode.
    prefactor = _polycrystal_prefactor(mode) * SEC2_TO_US2
    for element, moment in _tail_moments(
        atoms, radii, include_quadrupolar=include_quadrupolar
    ).items():
        tail = prefactor * moment
        sigma2 += tail
        if sigma2_parallel is not None:
            sigma2_parallel += tail
        per_element[element] = per_element.get(element, 0.0) + tail
    return _result_from_sigma(
        site_label=site_label,
        muon_position=tuple(
            wrap_fractional(
                np.asarray(muon_position, dtype=float).reshape(3),
                snap_tolerance=MUON_FRACTIONAL_SNAP_TOLERANCE,
            )
        )
        if muon_position_fractional
        else tuple(float(v) for v in muon_position),
        mode=mode,
        sigma_squared=sigma2,
        sigma_squared_parallel=sigma2_parallel,
        per_element=per_element,
        convergence_radius_a=max(radii.values()),
        n_nuclei_summed=int(distances.size),
    )


def _calculate_second_moment_finite_cell(
    atoms: Atoms,
    muon_position: Sequence[float],
    mode: str,
    *,
    site_label: str,
    field_direction: Sequence[float] = (0.0, 0.0, 1.0),
    polarisation_direction: Sequence[float] = (0.0, 0.0, 1.0),
    include_quadrupolar: bool = False,
    elements: Sequence[str] | None = None,
) -> NuclearRelaxationResult:
    _validate_mode(mode, include_quadrupolar=include_quadrupolar)
    field_dir = _unit_vector(field_direction, "direction") if mode == "tf_single_crystal" else None
    pol_dir = (
        _unit_vector(polarisation_direction, "direction") if mode == "zf_single_crystal" else None
    )
    muon_cart = _muon_position_cartesian(atoms, muon_position, True)
    selected = _selected_elements(elements)
    vecs_a, dists_a, symbols = _iter_nuclear_vectors_finite_cell(atoms, muon_cart, selected)
    if dists_a.size == 0:
        raise NuclearRelaxationError("No nuclei found in the selected finite cell.")
    sigma2, sigma2_parallel, per_element = _evaluate_mode(
        mode,
        vecs_a * METRES_PER_ANGSTROM,
        dists_a * METRES_PER_ANGSTROM,
        symbols,
        field_dir,
        pol_dir,
        include_quadrupolar,
    )
    return _result_from_sigma(
        site_label=site_label,
        muon_position=tuple(float(v) for v in muon_position),
        mode=mode,
        sigma_squared=sigma2,
        sigma_squared_parallel=sigma2_parallel,
        per_element=per_element,
        convergence_radius_a=float(np.max(dists_a)),
        n_nuclei_summed=int(dists_a.size),
    )


def _evaluate_mode(
    mode: str,
    vectors_m: np.ndarray,
    distances_m: np.ndarray,
    symbols: np.ndarray,
    field_direction: np.ndarray | None,
    polarisation_direction: np.ndarray | None,
    include_quadrupolar: bool,
) -> tuple[float, float | None, dict[str, float]]:
    if mode == "tf_single_crystal":
        assert field_direction is not None
        sigma2, per_element = _compute_tf_single(vectors_m, distances_m, symbols, field_direction)
        return sigma2, None, per_element
    if mode == "tf_polycrystal":
        sigma2, per_element = _compute_tf_poly(distances_m, symbols)
        return sigma2, None, per_element
    if mode == "zf_single_crystal":
        assert polarisation_direction is not None
        sigma2_perp, sigma2_parallel, per_element = _compute_zf_single(
            vectors_m,
            distances_m,
            symbols,
            polarisation_direction,
            include_quadrupolar,
        )
        return sigma2_perp, sigma2_parallel, per_element
    sigma2, per_element = _compute_zf_poly(distances_m, symbols, include_quadrupolar)
    return sigma2, None, per_element


def _compute_tf_single(
    vectors_m: np.ndarray,
    distances_m: np.ndarray,
    symbols: np.ndarray,
    field_direction: np.ndarray,
) -> tuple[float, dict[str, float]]:
    cos2 = (vectors_m @ field_direction) ** 2 / (distances_m**2)
    angular = (3.0 * cos2 - 1.0) ** 2
    pref = _prefactor_sigma_squared()
    return _sum_elements(symbols, pref * angular / (distances_m**6), include_quadrupolar=False)


def _compute_tf_poly(
    distances_m: np.ndarray, symbols: np.ndarray
) -> tuple[float, dict[str, float]]:
    pref = _polycrystal_prefactor("tf_polycrystal")
    return _sum_elements(symbols, pref / (distances_m**6), include_quadrupolar=False)


def _compute_zf_single(
    vectors_m: np.ndarray,
    distances_m: np.ndarray,
    symbols: np.ndarray,
    polarisation_direction: np.ndarray,
    include_quadrupolar: bool,
) -> tuple[float, float, dict[str, float]]:
    cos2 = (vectors_m @ polarisation_direction) ** 2 / (distances_m**2)
    pref = _prefactor_sigma_squared()
    # Kubo-Toyabe per-component convention: the transverse width entering the
    # ZF Gaussian envelope is sigma_perp^2 = gamma_mu^2 <B_x^2 + B_y^2> / 2,
    # i.e. half of the full transverse variance. With this normalization the
    # orientational (powder) average of sigma_perp^2 equals the zf_polycrystal
    # width of Hayano et al., PRB 20, 850 (1979).
    sigma_perp, per_element = _sum_elements(
        symbols,
        0.5 * pref * (5.0 - 3.0 * cos2) / (distances_m**6),
        include_quadrupolar=include_quadrupolar,
    )
    sigma_parallel, _parallel = _sum_elements(
        symbols,
        pref * (3.0 * cos2 + 1.0) / (distances_m**6),
        include_quadrupolar=include_quadrupolar,
    )
    return sigma_perp, sigma_parallel, per_element


def _compute_zf_poly(
    distances_m: np.ndarray,
    symbols: np.ndarray,
    include_quadrupolar: bool,
) -> tuple[float, dict[str, float]]:
    pref = _polycrystal_prefactor("zf_polycrystal")
    return _sum_elements(symbols, pref / (distances_m**6), include_quadrupolar=include_quadrupolar)


def _sum_elements(
    symbols: np.ndarray,
    base_terms_s: np.ndarray,
    *,
    include_quadrupolar: bool,
) -> tuple[float, dict[str, float]]:
    sigma2_s = 0.0
    per_element: dict[str, float] = {}
    for sym in np.unique(symbols):
        element = str(sym)
        mask = symbols == element
        weight = weighted_spin_factor(
            isotopes_for_element(element), include_quadrupolar=include_quadrupolar
        )
        contrib = weight * float(np.sum(base_terms_s[mask]))
        sigma2_s += contrib
        per_element[element] = contrib * SEC2_TO_US2
    return sigma2_s * SEC2_TO_US2, per_element


def _result_from_sigma(
    *,
    site_label: str,
    muon_position: Sequence[float],
    mode: str,
    sigma_squared: float,
    sigma_squared_parallel: float | None,
    per_element: dict[str, float],
    convergence_radius_a: float,
    n_nuclei_summed: int,
) -> NuclearRelaxationResult:
    sigma = float(np.sqrt(max(sigma_squared, 0.0)))
    return NuclearRelaxationResult(
        site_label=site_label,
        muon_position=tuple(float(v) for v in muon_position),
        sigma=sigma,
        sigma_squared=float(sigma_squared),
        m2=float(sigma_squared if mode.startswith("tf") else 2.0 * sigma_squared),
        delta_field_t=float((sigma * 1e6) / MUON_GAMMA_RAD_S_T),
        mode=mode,
        convergence_radius_a=float(convergence_radius_a),
        n_nuclei_summed=int(n_nuclei_summed),
        per_element_sigma_squared={str(k): float(v) for k, v in per_element.items()},
        sigma_perp_squared=float(sigma_squared) if mode == "zf_single_crystal" else None,
        sigma_parallel_squared=None
        if sigma_squared_parallel is None
        else float(sigma_squared_parallel),
    )


def _prefactor_sigma_squared() -> float:
    return ((MU0_OVER_4PI_T_M_PER_A**2) * (HBAR_J_S**2) * (MUON_GAMMA_RAD_S_T**2)) / 3.0


def _polycrystal_prefactor(mode: str) -> float:
    """Return the prefactor of ``w / r⁶`` in a polycrystal width, TF or ZF as ``mode`` says."""

    return _prefactor_sigma_squared() * (0.8 if mode.startswith("tf") else 2.0)


def _selected_elements(elements: Sequence[str] | None) -> set[str] | None:
    if elements is None:
        return None
    return {element_symbol(element, error=NuclearRelaxationError) for element in elements}


def _sum_radii(atoms: Atoms, muon_cart: np.ndarray, elements: set[str] | None) -> dict[str, float]:
    """Return the explicit sum's radius (Å) for each selected element of the host.

    It is the fixed radius unless that element's nuclei are too sparse for its tail,
    whatever else the host holds. A muon far from all of them, in the vacuum of a slab
    cell, reaches twice as far as the nearest, since a mean density says nothing about
    its surroundings. An element with no nuclear spin adds nothing and stays at the
    fixed radius.
    """

    symbols = np.asarray(atoms.get_chemical_symbols())
    cell = atoms.cell.array
    muon = cartesian_to_fractional(muon_cart, cell).reshape(1, 3)
    scaled = atoms.get_scaled_positions(wrap=False)
    radii: dict[str, float] = {}
    for symbol in np.unique(symbols):
        element = str(symbol)
        if elements is not None and element not in elements:
            continue
        radii[element] = NUCLEAR_SUM_RADIUS_A
        if weighted_spin_factor(isotopes_for_element(element)) <= 0.0:
            continue
        nuclei = symbols == element
        spacing = (cell_volume(cell.tolist()) / int(np.sum(nuclei))) ** (1.0 / 3.0)
        nearest = float(nearest_image_distance_matrix(muon, scaled[nuclei], cell).min())
        radii[element] = max(
            NUCLEAR_SUM_RADIUS_A, _SUM_RADIUS_MIN_SPACINGS * spacing, 2.0 * nearest
        )
    if not radii:
        raise NuclearRelaxationError(
            f"No nuclei found within {NUCLEAR_SUM_RADIUS_A:g} Å of the muon."
        )
    return radii


def _tail_moments(
    atoms: Atoms,
    radii: Mapping[str, float],
    *,
    include_quadrupolar: bool = False,
) -> dict[str, float]:
    """Return each element's ``Σ w / r⁶`` over its nuclei beyond its radius in ``radii``.

    They are treated as a uniform continuum of weight density ``n_w``, whose
    integral from ``R`` outwards is ``4π n_w / (3 R³)``.
    """

    volume_m3 = cell_volume(atoms.cell.array.tolist()) * METRES_PER_ANGSTROM**3
    symbols = atoms.get_chemical_symbols()
    moments: dict[str, float] = {}
    for element, radius_a in radii.items():
        weight = weighted_spin_factor(
            isotopes_for_element(element), include_quadrupolar=include_quadrupolar
        )
        radius_m = radius_a * METRES_PER_ANGSTROM
        moments[element] = (
            4.0 * np.pi * weight * symbols.count(element) / (3.0 * volume_m3 * radius_m**3)
        )
    return moments


def _nuclear_vectors_within(
    atoms: Atoms,
    muon_cart: np.ndarray,
    radii: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return vectors and distances (Å) to the nuclei within their radii, and their symbols.

    A point within ``R`` of the muon has a fractional coordinate of at most
    ``R / h`` along each axis, with ``h`` the spacing of the lattice planes
    normal to it, and folding an atom next to the muon adds at most ½, so the
    translation grid holds every such nucleus in a cell of any shape. The grid
    runs over a reduced basis of the lattice, whose plane heights do not shrink
    with a shear of the input cell, so its size depends on the lattice alone.
    """

    reach = max(radii.values())
    basis, _transform = reduced_lattice(atoms.cell.array, error=NuclearRelaxationError)
    limits = np.ceil(
        reach / lattice_plane_heights(basis, error=NuclearRelaxationError) + 0.5
    ).astype(int)
    grid = np.meshgrid(*(np.arange(-limit, limit + 1) for limit in limits), indexing="ij")
    translations = np.stack(grid, axis=-1).reshape(-1, 3) @ basis
    folded = np.linalg.solve(basis.T, (atoms.get_positions() - muon_cart).T).T
    offsets = (folded - np.round(folded)) @ basis
    vectors: list[np.ndarray] = []
    symbols: list[np.ndarray] = []
    for symbol, offset in zip(atoms.get_chemical_symbols(), offsets, strict=True):
        if symbol not in radii:
            continue
        candidates = translations + offset
        squared = np.einsum("ij,ij->i", candidates, candidates)
        within = candidates[(squared <= radii[symbol] ** 2) & (squared >= 1e-20)]
        vectors.append(within)
        symbols.append(np.full(len(within), symbol, dtype=object))
    if not any(len(block) for block in vectors):
        raise NuclearRelaxationError(f"No nuclei found within {reach:g} Å of the muon.")
    stacked = np.concatenate(vectors)
    return stacked, np.linalg.norm(stacked, axis=1), np.concatenate(symbols)


def _iter_nuclear_vectors_finite_cell(
    atoms: Atoms,
    muon_cart: np.ndarray,
    elements: set[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vecs: list[np.ndarray] = []
    dists: list[float] = []
    syms: list[str] = []
    for sym, pos in zip(atoms.get_chemical_symbols(), atoms.get_positions(), strict=False):
        if elements is not None and str(sym) not in elements:
            continue
        dr = np.asarray(pos, dtype=float) - muon_cart
        dist = float(np.linalg.norm(dr))
        if dist < 1e-10:
            continue
        vecs.append(dr)
        dists.append(dist)
        syms.append(str(sym))
    if not vecs:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), np.array([], dtype=object)
    return (
        np.asarray(vecs, dtype=float),
        np.asarray(dists, dtype=float),
        np.asarray(syms, dtype=object),
    )


def _finite_cell_atoms(distortion: MuonDistortion, vectors: np.ndarray) -> Atoms:
    """Place host ions at ``vectors`` from a muon at the centre of the correction cell."""

    cell = np.asarray(distortion.supercell_lattice, dtype=float)
    centre = np.full(3, 0.5) @ cell
    return Atoms(symbols=list(distortion.symbols), positions=centre + vectors, cell=cell, pbc=True)


def _muon_position_cartesian(
    atoms: Atoms, muon_position: Sequence[float], fractional: bool
) -> np.ndarray:
    pos = np.asarray(muon_position, dtype=float).reshape(3)
    if fractional:
        return np.asarray(atoms.cell.cartesian_positions(pos.reshape(1, 3))[0], dtype=float)
    return pos


def _unit_vector(vec: Sequence[float], name: str) -> np.ndarray:
    unit, _ = _unit_directions(np.asarray(vec, dtype=float).reshape(3), name)
    return unit


def _unit_directions(
    direction: Sequence[float] | np.ndarray, name: str = "direction"
) -> tuple[np.ndarray, bool]:
    """Return ``direction`` as unit rows and whether it was a single vector."""

    arr = np.asarray(direction, dtype=float)
    if arr.shape[-1:] != (3,):
        raise NuclearRelaxationError(f"{name} must have 3 components, got shape {arr.shape}")
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norms == 0.0):
        raise NuclearRelaxationError(f"{name} must be non-zero.")
    return arr / norms, arr.ndim == 1


def _validate_mode(mode: str, *, include_quadrupolar: bool) -> None:
    if mode not in _ALLOWED_MODES:
        raise NuclearRelaxationError(f"Unsupported mode '{mode}'.")
    if include_quadrupolar and mode.startswith("tf"):
        raise NuclearRelaxationError("Quadrupolar correction is only supported in ZF modes.")


def _parse_number_token(token: str) -> float:
    token = token.strip()
    if not token:
        raise ValueError("empty numeric token")
    if "/" in token:
        return float(Fraction(token))
    return float(token)


def _parse_csv_numbers(raw: str) -> list[float]:
    if raw.strip() == "":
        return []
    return [_parse_number_token(part) for part in raw.split(",")]


def _parse_csv_ints(raw: str) -> list[int | None]:
    if raw.strip() == "":
        return []
    return [int(part.strip()) for part in raw.split(",")]


def _pad(values: list, count: int, fill) -> list:
    if len(values) >= count:
        return values
    return [*values, *([fill] * (count - len(values)))]
