"""Nuclear dipolar relaxation helpers for MuFinder."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from configparser import ConfigParser
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from importlib import resources
from typing import Literal

from ase import Atoms
import numpy as np

from mufinder.domain.constants import MUON_GAMMA_OVER_2PI_MHZ_T, MUON_GAMMA_RAD_S_T
from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import element_symbol, structure_to_ase_atoms
from mufinder.science.muon_distortion import MuonDistortion, map_muon_distortion

MU0_OVER_4PI = 1e-7
HBAR = 1.054_571_817e-34
MU_N = 5.050_783_699e-27
ANGSTROM_TO_M = 1e-10
GAMMA_MU_OVER_2PI_MHZ_T = MUON_GAMMA_OVER_2PI_MHZ_T
GAMMA_MU_RAD_S_T = MUON_GAMMA_RAD_S_T
SEC2_TO_US2 = 1e-12

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
    radius: float | None = None
    convergence_rtol: float = 1e-6
    use_distortion_correction: bool = False

    def to_json(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "direction": list(self.direction),
            "include_quadrupolar": self.include_quadrupolar,
            "elements": None if self.elements is None else list(self.elements),
            "radius": self.radius,
            "convergence_rtol": self.convergence_rtol,
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


def gyromagnetic_ratio_from_moment(moment_mu_n: float, spin: float) -> float:
    """Return nuclear gyromagnetic ratio in rad/(s*T)."""

    if spin <= 0.0:
        return 0.0
    return (float(moment_mu_n) * MU_N) / (HBAR * float(spin))


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


def weighted_spin_factor(isotopes: Iterable[Isotope], *, include_quadrupolar: bool = False) -> float:
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
        radius=opts.radius,
        convergence_rtol=opts.convergence_rtol,
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
        m2=correction.sigma_squared_corrected if opts.mode.startswith("tf") else 2.0 * correction.sigma_squared_corrected,
        delta_field_t=float((correction.sigma_corrected * 1e6) / GAMMA_MU_RAD_S_T),
        mode=opts.mode,
        convergence_radius_a=base.convergence_radius_a,
        n_nuclei_summed=base.n_nuclei_summed,
        per_element_sigma_squared=dict(base.per_element_sigma_squared),
        sigma_perp_squared=correction.sigma_squared_corrected if opts.mode == "zf_single_crystal" else None,
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
        radius=opts.radius,
        convergence_rtol=opts.convergence_rtol,
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

    correction_sigma2 = distorted_supercell_result.sigma_squared - pristine_supercell_result.sigma_squared
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
    radius: float | None = None,
    convergence_rtol: float = 1e-6,
) -> NuclearRelaxationResult:
    _validate_mode(mode, include_quadrupolar=include_quadrupolar)
    if radius is not None and radius <= 0.0:
        raise NuclearRelaxationError("radius must be > 0")

    field_dir = _unit_vector(field_direction, "direction") if mode == "tf_single_crystal" else None
    pol_dir = _unit_vector(polarisation_direction, "direction") if mode == "zf_single_crystal" else None
    muon_cart = _muon_position_cartesian(atoms, muon_position, muon_position_fractional)
    selected = None if elements is None else {element_symbol(element, error=NuclearRelaxationError) for element in elements}
    radii = [float(radius)] if radius is not None else list(_auto_radii_sequence())

    prev_sigma2: float | None = None
    final_sigma2 = 0.0
    final_sigma2_parallel: float | None = None
    final_per_element: dict[str, float] = {}
    final_n = 0
    final_radius = radii[-1]
    for rad in radii:
        vecs_a, dists_a, symbols = _iter_nuclear_vectors(atoms, muon_cart, rad, selected)
        if dists_a.size == 0:
            continue
        sigma2, sigma2_parallel, per_element = _evaluate_mode(
            mode,
            vecs_a * ANGSTROM_TO_M,
            dists_a * ANGSTROM_TO_M,
            symbols,
            field_dir,
            pol_dir,
            include_quadrupolar,
        )
        final_sigma2 = sigma2
        final_sigma2_parallel = sigma2_parallel
        final_per_element = per_element
        final_n = int(dists_a.size)
        final_radius = rad
        if prev_sigma2 is not None and radius is None and np.isclose(sigma2, prev_sigma2, rtol=convergence_rtol, atol=0.0):
            break
        prev_sigma2 = sigma2

    if final_n == 0:
        raise NuclearRelaxationError("No nuclei found within the selected radius.")
    return _result_from_sigma(
        site_label=site_label,
        muon_position=_canonical_fractional(muon_position) if muon_position_fractional else tuple(float(v) for v in muon_position),
        mode=mode,
        sigma_squared=final_sigma2,
        sigma_squared_parallel=final_sigma2_parallel,
        per_element=final_per_element,
        convergence_radius_a=final_radius,
        n_nuclei_summed=final_n,
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
    pol_dir = _unit_vector(polarisation_direction, "direction") if mode == "zf_single_crystal" else None
    muon_cart = _muon_position_cartesian(atoms, muon_position, True)
    selected = None if elements is None else {element_symbol(element, error=NuclearRelaxationError) for element in elements}
    vecs_a, dists_a, symbols = _iter_nuclear_vectors_finite_cell(atoms, muon_cart, selected)
    if dists_a.size == 0:
        raise NuclearRelaxationError("No nuclei found in the selected finite cell.")
    sigma2, sigma2_parallel, per_element = _evaluate_mode(
        mode,
        vecs_a * ANGSTROM_TO_M,
        dists_a * ANGSTROM_TO_M,
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


def _compute_tf_poly(distances_m: np.ndarray, symbols: np.ndarray) -> tuple[float, dict[str, float]]:
    pref = ((MU0_OVER_4PI**2) * (HBAR**2) * (GAMMA_MU_RAD_S_T**2)) * (4.0 / 15.0)
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
    pref = ((MU0_OVER_4PI**2) * (HBAR**2) * (GAMMA_MU_RAD_S_T**2)) * (2.0 / 3.0)
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
        weight = weighted_spin_factor(isotopes_for_element(element), include_quadrupolar=include_quadrupolar)
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
        delta_field_t=float((sigma * 1e6) / GAMMA_MU_RAD_S_T),
        mode=mode,
        convergence_radius_a=float(convergence_radius_a),
        n_nuclei_summed=int(n_nuclei_summed),
        per_element_sigma_squared={str(k): float(v) for k, v in per_element.items()},
        sigma_perp_squared=float(sigma_squared) if mode == "zf_single_crystal" else None,
        sigma_parallel_squared=None if sigma_squared_parallel is None else float(sigma_squared_parallel),
    )


def _prefactor_sigma_squared() -> float:
    return ((MU0_OVER_4PI**2) * (HBAR**2) * (GAMMA_MU_RAD_S_T**2)) / 3.0


def _iter_nuclear_vectors(
    atoms: Atoms,
    muon_cart: np.ndarray,
    radius_a: float,
    elements: set[str] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell = np.asarray(atoms.cell.array, dtype=float)
    frac = np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    symbols = np.asarray(atoms.get_chemical_symbols(), dtype=object)
    lim = _translation_limits(cell, radius_a)
    vecs: list[np.ndarray] = []
    dists: list[float] = []
    syms: list[str] = []
    for index, fpos in enumerate(frac):
        sym = str(symbols[index])
        if elements is not None and sym not in elements:
            continue
        for na in range(-lim[0], lim[0] + 1):
            for nb in range(-lim[1], lim[1] + 1):
                for nc in range(-lim[2], lim[2] + 1):
                    pos = np.dot(fpos + np.asarray([na, nb, nc], dtype=float), cell)
                    dr = pos - muon_cart
                    dist = float(np.linalg.norm(dr))
                    if dist < 1e-10 or dist > radius_a:
                        continue
                    vecs.append(dr)
                    dists.append(dist)
                    syms.append(sym)
    if not vecs:
        return np.zeros((0, 3), dtype=float), np.zeros(0, dtype=float), np.array([], dtype=object)
    return np.asarray(vecs, dtype=float), np.asarray(dists, dtype=float), np.asarray(syms, dtype=object)


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
    return np.asarray(vecs, dtype=float), np.asarray(dists, dtype=float), np.asarray(syms, dtype=object)


def _translation_limits(cell: np.ndarray, radius_a: float) -> np.ndarray:
    norms = np.linalg.norm(cell, axis=1)
    min_norm = float(np.min(norms))
    if min_norm <= 0.0:
        raise NuclearRelaxationError("Structure cell must be non-singular.")
    nmax = int(np.ceil(radius_a / min_norm)) + 1
    return np.array([nmax, nmax, nmax], dtype=int)


def _finite_cell_atoms(distortion: MuonDistortion, vectors: np.ndarray) -> Atoms:
    """Place host ions at ``vectors`` from a muon at the centre of the correction cell."""

    cell = np.asarray(distortion.supercell_lattice, dtype=float)
    centre = np.full(3, 0.5) @ cell
    return Atoms(symbols=list(distortion.symbols), positions=centre + vectors, cell=cell, pbc=True)


def _muon_position_cartesian(atoms: Atoms, muon_position: Sequence[float], fractional: bool) -> np.ndarray:
    pos = np.asarray(muon_position, dtype=float).reshape(3)
    if fractional:
        return np.asarray(atoms.cell.cartesian_positions(pos.reshape(1, 3))[0], dtype=float)
    return pos


def _unit_vector(vec: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise NuclearRelaxationError(f"{name} must be non-zero.")
    return arr / norm


def _canonical_fractional(frac: Sequence[float], *, atol: float = 1e-6) -> tuple[float, float, float]:
    arr = np.mod(np.asarray(frac, dtype=float).reshape(3), 1.0)
    arr[np.isclose(arr, 1.0, atol=atol, rtol=0.0)] = 0.0
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _auto_radii_sequence() -> Iterable[float]:
    for radius in np.arange(6.0, 34.1, 2.0):
        yield float(radius)


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
