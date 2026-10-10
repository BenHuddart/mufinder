"""Tests for nuclear dipolar relaxation calculations."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import pytest
from ase import Atoms

from mufinder.domain.models import Structure
from mufinder.science import nuclear_relaxation as nr
from mufinder.science.nuclear_relaxation import (
    ANGSTROM_TO_M,
    GAMMA_MU_RAD_S_T,
    HBAR,
    MU0_OVER_4PI,
    DistortionCorrectionResult,
    Isotope,
    NuclearRelaxationError,
    NuclearRelaxationOptions,
    calculate_distortion_correction,
    calculate_nuclear_relaxation,
    calculate_nuclear_relaxation_with_distortion,
    gyromagnetic_ratio_from_moment,
    isotopes_for_element,
    load_isotopes,
    structure_to_atoms,
    weighted_spin_factor,
)
from tests.support.structures import host_structure, hydrogen_host


def _single_nucleus_structure(symbol: str = "H") -> Structure:
    return host_structure(label=symbol, lattice=100.0, species=(symbol,), positions=((0.0, 0.0, 0.01),))


def _distorted_structure(host_position=(0.0, 0.0, 0.0), cell_length: float = 2.0) -> Structure:
    return host_structure(
        label="distorted",
        lattice=cell_length,
        species=("H", "H:mu"),
        positions=(host_position, (0.5, 0.5, 0.5)),
        parent_structure_id=hydrogen_host().id,
    )


@pytest.fixture
def isotope_table(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    """Serve ``load_isotopes`` a synthetic isotope table instead of the bundled one."""

    def _install(text: str) -> None:
        class _Resource:
            def joinpath(self, _name: str) -> "_Resource":
                return self

            def read_text(self, encoding: str) -> str:
                return text

        monkeypatch.setattr(nr.resources, "files", lambda _package: _Resource())
        load_isotopes.cache_clear()

    yield _install
    load_isotopes.cache_clear()


# Isotope data and spin weighting


def test_isotope_table_loads_cu_and_gamma_reference() -> None:
    cu = isotopes_for_element("Cu")

    assert len(cu) == 2
    assert cu[0].mass_number == 63
    assert cu[1].mass_number == 65
    assert cu[0].abundance + cu[1].abundance == pytest.approx(1.0)
    gamma = gyromagnetic_ratio_from_moment(moment_mu_n=2.227, spin=1.5)
    assert gamma / (2.0 * np.pi * 1e6) == pytest.approx(11.317, rel=2e-4)


def test_load_isotopes_is_cached() -> None:
    assert load_isotopes() is load_isotopes()


def test_isotope_to_json_roundtrip_keys() -> None:
    iso = isotopes_for_element("H")[0]
    payload = iso.to_json()
    assert payload["element"] == "H"
    assert set(payload) == {
        "element",
        "mass_number",
        "abundance",
        "spin",
        "moment_mu_n",
        "quadrupole_1e28_m2",
        "gamma_rad_s_t",
    }


def test_isotopes_for_element_unknown_raises() -> None:
    with pytest.raises(NuclearRelaxationError, match="No isotope data"):
        isotopes_for_element("Zz")


def test_isotopes_for_element_strips_mass_numbers_from_symbols() -> None:
    assert isotopes_for_element("Cu63") == isotopes_for_element("Cu")
    assert isotopes_for_element("H1") == isotopes_for_element("H")


@pytest.mark.parametrize(
    ("symbol", "message"),
    [("   ", "must not be empty"), ("123", "Could not resolve element")],
    ids=["blank", "no-element"],
)
def test_isotopes_for_element_rejects_unreadable_symbols(symbol: str, message: str) -> None:
    with pytest.raises(NuclearRelaxationError, match=message):
        isotopes_for_element(symbol)


def test_load_isotopes_parses_fractions_and_pads_missing_columns(isotope_table) -> None:
    isotope_table("[Xa]\nmasses=1, 2\nabundances=60,40\nI=1/2\nmoments=1.0,2.0\nQ=0\n\n[Xb]\n")

    table = load_isotopes()

    first, second = table["Xa"]
    assert (first.mass_number, first.abundance, first.spin) == (1, 0.6, 0.5)
    # Columns shorter than the longest one are padded: spin and Q default to 0.
    assert (second.mass_number, second.abundance, second.spin, second.quadrupole_1e28_m2) == (2, 0.4, 0.0, 0.0)
    assert second.gamma_rad_s_t == 0.0
    assert table["Xb"] == []


def test_load_isotopes_rejects_an_empty_numeric_entry(isotope_table) -> None:
    isotope_table("[Xa]\nabundances=50, ,50\n")

    with pytest.raises(ValueError, match="empty numeric token"):
        load_isotopes()


def test_gyromagnetic_ratio_zero_for_nonpositive_spin() -> None:
    assert gyromagnetic_ratio_from_moment(2.5, 0.0) == 0.0
    assert gyromagnetic_ratio_from_moment(2.5, -1.0) == 0.0


def test_weighted_spin_factor_skips_zero_spin_and_zero_abundance() -> None:
    spinless = Isotope(
        element="X",
        mass_number=1,
        abundance=1.0,
        spin=0.0,
        moment_mu_n=0.0,
        quadrupole_1e28_m2=0.0,
        gamma_rad_s_t=0.0,
    )
    no_abundance = Isotope(
        element="X",
        mass_number=2,
        abundance=0.0,
        spin=1.5,
        moment_mu_n=2.0,
        quadrupole_1e28_m2=0.0,
        gamma_rad_s_t=1.0e8,
    )
    assert weighted_spin_factor([spinless, no_abundance]) == 0.0


def test_weighted_spin_factor_integer_spin_quadrupolar_branch() -> None:
    iso = Isotope(
        element="X",
        mass_number=14,
        abundance=1.0,
        spin=1.0,  # integer spin > 1/2 -> factor 2/3
        moment_mu_n=0.4,
        quadrupole_1e28_m2=0.02,
        gamma_rad_s_t=1.0e7,
    )
    plain = weighted_spin_factor([iso], include_quadrupolar=False)
    quad = weighted_spin_factor([iso], include_quadrupolar=True)
    assert quad == pytest.approx(plain * (2.0 / 3.0))


def test_weighted_spin_factor_half_integer_spin_quadrupolar_branch() -> None:
    i = 1.5  # half-integer spin > 1/2 -> bespoke formula
    iso = Isotope(
        element="X",
        mass_number=23,
        abundance=1.0,
        spin=i,
        moment_mu_n=2.2,
        quadrupole_1e28_m2=0.1,
        gamma_rad_s_t=7.0e7,
    )
    plain = weighted_spin_factor([iso], include_quadrupolar=False)
    quad = weighted_spin_factor([iso], include_quadrupolar=True)
    expected_factor = (16.0 * i * (i + 1.0) + 3.0 * (2.0 * i + 1.0)) / (24.0 * i * (i + 1.0))
    assert quad == pytest.approx(plain * expected_factor)
    # Half-integer correction factor differs from the integer 2/3 case.
    assert expected_factor != pytest.approx(2.0 / 3.0)


# Options, results and structure conversion


def test_options_to_json_serialises_elements_and_direction() -> None:
    opts = NuclearRelaxationOptions(
        mode="zf_single_crystal",
        direction=(1.0, 0.0, 0.0),
        include_quadrupolar=True,
        elements=("H", "Cu"),
        radius=8.0,
        convergence_rtol=1e-5,
        use_distortion_correction=True,
    )
    payload = opts.to_json()
    assert payload["mode"] == "zf_single_crystal"
    assert payload["direction"] == [1.0, 0.0, 0.0]
    assert payload["elements"] == ["H", "Cu"]
    assert payload["radius"] == 8.0
    assert payload["use_distortion_correction"] is True


def test_options_to_json_none_elements() -> None:
    payload = NuclearRelaxationOptions().to_json()
    assert payload["elements"] is None


def test_result_to_json_without_distortion_omits_key() -> None:
    result = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="zf_single_crystal", radius=2.0),
    )
    payload = result.to_json()
    assert "distortion_correction" not in payload
    assert payload["muon_position"] == [0.0, 0.0, 0.0]
    assert payload["per_element_sigma_squared"]["H"] == pytest.approx(
        result.per_element_sigma_squared["H"]
    )
    assert payload["sigma_perp_squared"] is not None


def test_structure_to_atoms_skips_muon_and_maps_label() -> None:
    structure = Structure(
        label="with muon",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["H", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    atoms = structure_to_atoms(structure, skip_muon=True)
    assert len(atoms) == 1
    assert atoms.get_chemical_symbols() == ["H"]
    assert atoms.info["mufinder_species_labels"] == ["H"]


def test_structure_to_atoms_keeps_muon_as_hydrogen() -> None:
    structure = Structure(
        label="with muon",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["O", "mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    atoms = structure_to_atoms(structure, skip_muon=False)
    assert atoms.get_chemical_symbols() == ["O", "H"]
    assert atoms.info["mufinder_species_labels"] == ["O", "mu"]


def test_structure_to_atoms_raises_when_only_muon_present() -> None:
    structure = Structure(
        label="muon only",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["mu"],
        fractional_positions=[[0.5, 0.5, 0.5]],
    )
    with pytest.raises(NuclearRelaxationError, match="does not contain any host atoms"):
        structure_to_atoms(structure, skip_muon=True)


# Second moments by measurement geometry


def test_tf_single_crystal_matches_manual_single_nucleus_formula() -> None:
    result = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(
            mode="tf_single_crystal",
            direction=(0.0, 0.0, 1.0),
            radius=2.0,
        ),
    )

    h_iso = isotopes_for_element("H")[0]
    r_m = 1.0 * ANGSTROM_TO_M
    weight = (h_iso.gamma_rad_s_t**2) * h_iso.spin * (h_iso.spin + 1.0)
    sigma2_s = ((MU0_OVER_4PI**2) * (HBAR**2) * (GAMMA_MU_RAD_S_T**2) / 3.0) * weight * (4.0 / (r_m**6))

    assert result.sigma_squared == pytest.approx(sigma2_s * 1e-12)
    assert result.m2 == pytest.approx(result.sigma_squared)


def test_zf_and_tf_polycrystal_ratio_is_five_over_two() -> None:
    tf = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="tf_polycrystal", radius=2.0),
    )
    zf = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=2.0),
    )

    assert zf.sigma_squared / tf.sigma_squared == pytest.approx(2.5)
    assert zf.m2 / tf.m2 == pytest.approx(5.0)


def test_zf_single_crystal_exposes_parallel_and_perp_widths() -> None:
    result = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(
            mode="zf_single_crystal",
            direction=(0.0, 0.0, 1.0),
            radius=2.0,
        ),
    )

    assert result.sigma_perp_squared is not None
    assert result.sigma_parallel_squared is not None
    # Nucleus on the polarisation axis (theta = 0, cos^2 theta = 1):
    # sigma_parallel^2 ∝ (3 cos^2 + 1) = 4, while the KT per-component
    # transverse width is sigma_perp^2 ∝ (1/2)(5 - 3 cos^2) = 1, so the
    # ratio is exactly 4.
    assert result.sigma_parallel_squared / result.sigma_perp_squared == pytest.approx(4.0)


def test_zf_single_crystal_matches_manual_single_nucleus_formula() -> None:
    # Nucleus at cartesian (1, 0, 1) Å from the muon: theta = 45 degrees to the
    # z polarisation axis, so cos^2 theta = 1/2 and r = sqrt(2) Å.
    structure = Structure(
        label="H",
        lattice_vectors=[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        species=["H"],
        fractional_positions=[[0.01, 0.0, 0.01]],
    )
    result = calculate_nuclear_relaxation(
        structure,
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(
            mode="zf_single_crystal",
            direction=(0.0, 0.0, 1.0),
            radius=3.0,
        ),
    )

    h_iso = isotopes_for_element("H")[0]
    r_m = np.sqrt(2.0) * ANGSTROM_TO_M
    cos2 = 0.5
    weight = (h_iso.gamma_rad_s_t**2) * h_iso.spin * (h_iso.spin + 1.0)
    base = (MU0_OVER_4PI**2) * (HBAR**2) * (GAMMA_MU_RAD_S_T**2) * weight / (r_m**6)
    # KT per-component transverse width: sigma_perp^2 = (1/6)(5 - 3cos^2) base.
    expected_perp = base * (5.0 - 3.0 * cos2) / 6.0
    # Longitudinal per-component width: sigma_par^2 = (1/3)(3cos^2 + 1) base.
    expected_parallel = base * (3.0 * cos2 + 1.0) / 3.0

    assert result.sigma_perp_squared == pytest.approx(expected_perp * 1e-12)
    assert result.sigma_squared == pytest.approx(expected_perp * 1e-12)
    assert result.sigma_parallel_squared == pytest.approx(expected_parallel * 1e-12)


def test_zf_single_crystal_powder_average_matches_polycrystal() -> None:
    # Regression for the ZF single-crystal normalization: in the KT
    # per-component convention the orientational average of the single-crystal
    # widths must reproduce the Hayano et al. PRB 20, 850 (1979) polycrystal
    # width. Before the fix sigma_perp^2 averaged to twice the polycrystal
    # value.
    structure = _single_nucleus_structure()
    zf_poly = calculate_nuclear_relaxation(
        structure,
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=2.0),
    )

    # Deterministic Fibonacci spherical design: near-uniform coverage with
    # quadrature error well below the assertion tolerance at modest cost.
    n_dirs = 64
    indices = np.arange(n_dirs, dtype=float)
    z = 1.0 - (2.0 * indices + 1.0) / n_dirs
    phi = np.pi * (3.0 - np.sqrt(5.0)) * indices
    rho = np.sqrt(1.0 - z**2)
    directions = np.column_stack([rho * np.cos(phi), rho * np.sin(phi), z])

    perp_values = []
    parallel_values = []
    for direction in directions:
        single = calculate_nuclear_relaxation(
            structure,
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(
                mode="zf_single_crystal",
                direction=tuple(direction),
                radius=2.0,
            ),
        )
        perp_values.append(single.sigma_perp_squared)
        parallel_values.append(single.sigma_parallel_squared)

    assert np.mean(perp_values) == pytest.approx(zf_poly.sigma_squared, rel=0.01)
    assert np.mean(parallel_values) == pytest.approx(zf_poly.sigma_squared, rel=0.01)


def test_auto_radius_convergence_path_runs() -> None:
    # No explicit radius -> exercises the auto radii sequence + convergence break.
    result = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="zf_polycrystal"),
    )
    assert result.convergence_radius_a >= 6.0
    assert result.n_nuclei_summed >= 1


def test_element_filter_excludes_other_species() -> None:
    structure = Structure(
        label="HO",
        lattice_vectors=[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        species=["H", "O"],
        fractional_positions=[[0.0, 0.0, 0.01], [0.0, 0.01, 0.0]],
    )
    only_h = calculate_nuclear_relaxation(
        structure,
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=2.0, elements=("H",)),
    )
    assert set(only_h.per_element_sigma_squared) == {"H"}


def test_fractional_muon_position_is_wrapped_into_the_cell() -> None:
    structure = hydrogen_host(3.0)
    options = NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0)

    wrapped = calculate_nuclear_relaxation(structure, (1.0, 1.5, -0.25), options=options)
    snapped = calculate_nuclear_relaxation(structure, (1.0 - 1e-9, 0.5, 0.75), options=options)

    assert wrapped.muon_position == pytest.approx((0.0, 0.5, 0.75))
    # Values within rounding of 1.0 snap back to 0.0.
    assert snapped.muon_position == pytest.approx((0.0, 0.5, 0.75))


# Second-moment validation


def test_quadrupolar_correction_is_rejected_for_tf_modes() -> None:
    with pytest.raises(NuclearRelaxationError, match="Quadrupolar"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure(),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(
                mode="tf_polycrystal",
                include_quadrupolar=True,
                radius=2.0,
            ),
        )


def test_missing_isotope_data_reports_element() -> None:
    with pytest.raises(NuclearRelaxationError, match="He"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure("He"),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(radius=2.0),
        )


def test_unsupported_mode_raises() -> None:
    with pytest.raises(NuclearRelaxationError, match="Unsupported mode"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure(),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(mode="bogus", radius=2.0),  # type: ignore[arg-type]
        )


def test_nonpositive_radius_raises() -> None:
    with pytest.raises(NuclearRelaxationError, match="radius must be > 0"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure(),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=-1.0),
        )


def test_no_nuclei_within_radius_raises() -> None:
    # Muon coincides with the single nucleus, so nothing is summed.
    structure = Structure(
        label="coincident",
        lattice_vectors=[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        species=["H"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    with pytest.raises(NuclearRelaxationError, match="No nuclei found within the selected radius"):
        calculate_nuclear_relaxation(
            structure,
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=2.0),
        )


def test_zero_direction_vector_raises_in_single_crystal_mode() -> None:
    with pytest.raises(NuclearRelaxationError, match="direction must be non-zero"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure(),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(
                mode="tf_single_crystal",
                direction=(0.0, 0.0, 0.0),
                radius=2.0,
            ),
        )


# Distortion correction


def test_distortion_correction_is_zero_for_identical_host_positions() -> None:
    result = calculate_distortion_correction(
        hydrogen_host(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
    )

    assert result.supercell_matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert result.distortion_correction_sigma_squared == pytest.approx(0.0, abs=1e-12)
    assert result.sigma_squared_corrected == pytest.approx(result.sigma_squared_undistorted)


def test_distortion_correction_changes_for_local_displacement() -> None:
    result = calculate_distortion_correction(
        hydrogen_host(),
        _distorted_structure(host_position=(0.1, 0.0, 0.0)),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
    )

    assert abs(result.distortion_correction_sigma_squared) > 1e-12


def test_distortion_correction_detects_a_doubled_supercell() -> None:
    # A 2-atom host relaxed in a 1x2x1 supercell is matched to the doubled pristine cell.
    pristine = Structure(
        label="2-atom host",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["H", "H"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    distorted = Structure(
        label="2x supercell distorted",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 3.0]],
        species=["H", "H", "H", "H", "H:mu"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.25, 0.5],
            [0.0, 0.5, 0.0],
            [0.5, 0.75, 0.5],
            [0.25, 0.25, 0.25],
        ],
        parent_structure_id=pristine.id,
    )
    result = calculate_distortion_correction(
        pristine,
        distorted,
        (0.25, 0.125, 0.25),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
    )
    assert isinstance(result, DistortionCorrectionResult)
    assert result.supercell_matrix[1][1] == 2
    assert result.n_atoms_unwrapped >= 0


def test_distortion_correction_is_independent_of_atom_order() -> None:
    # Listing the distorted atoms in a different order from the pristine cell
    # forces the nearest-neighbour atom mapping instead of the identity.
    pristine = host_structure(
        label="H2Li",
        lattice=3.0,
        species=("H", "H", "Li"),
        positions=((0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.5)),
    )

    def distorted(species, positions) -> Structure:
        return host_structure(
            label="distorted", lattice=3.0, species=species, positions=positions, parent_structure_id=pristine.id
        )

    muon = (0.25, 0.25, 0.25)
    hydrogens = ((0.02, 0.0, 0.0), (0.5, 0.0, 0.0))
    same_order = distorted(("H", "H", "Li", "H:mu"), (*hydrogens, (0.5, 0.5, 0.5), muon))
    reordered = distorted(("Li", "H", "H", "H:mu"), ((0.5, 0.5, 0.5), *hydrogens, muon))
    options = NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0)

    expected = calculate_distortion_correction(pristine, same_order, muon, options=options)
    result = calculate_distortion_correction(pristine, reordered, muon, options=options)

    assert abs(result.distortion_correction_sigma_squared) > 1e-12
    assert result.distortion_correction_sigma_squared == pytest.approx(expected.distortion_correction_sigma_squared)


def test_distortion_correction_reports_unmatched_supercell() -> None:
    with pytest.raises(NuclearRelaxationError, match="matching pristine supercell"):
        calculate_distortion_correction(
            hydrogen_host(),
            _distorted_structure(cell_length=2.5),
            (0.5, 0.5, 0.5),
            options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
        )


def test_distortion_correction_requires_muon_marker() -> None:
    no_muon = Structure(
        label="no muon",
        lattice_vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        species=["H"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        parent_structure_id=hydrogen_host().id,
    )
    with pytest.raises(NuclearRelaxationError, match="does not contain a muon marker"):
        calculate_distortion_correction(
            hydrogen_host(),
            no_muon,
            (0.5, 0.5, 0.5),
            options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
        )


def test_distortion_correction_rejects_multiple_muons() -> None:
    two_muons = Structure(
        label="two muons",
        lattice_vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        species=["H", "H:mu", "mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=hydrogen_host().id,
    )
    with pytest.raises(NuclearRelaxationError, match="more than one muon marker"):
        calculate_distortion_correction(
            hydrogen_host(),
            two_muons,
            (0.5, 0.5, 0.5),
            options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
        )


def test_with_distortion_default_options_enable_correction() -> None:
    # No options -> default NuclearRelaxationOptions(use_distortion_correction=True)
    result = calculate_nuclear_relaxation_with_distortion(
        hydrogen_host(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
    )
    assert result.distortion_correction is not None
    assert result.mode == "zf_polycrystal"


def test_with_distortion_zf_polycrystal_m2_is_double_sigma_squared() -> None:
    result = calculate_nuclear_relaxation_with_distortion(
        hydrogen_host(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
    )
    assert result.mode == "zf_polycrystal"
    assert result.m2 == pytest.approx(2.0 * result.sigma_squared)
    assert result.distortion_correction is not None
    assert result.sigma_perp_squared is None
    # delta_field_t derives from sigma_corrected.
    expected_delta = (result.sigma * 1e6) / GAMMA_MU_RAD_S_T
    assert result.delta_field_t == pytest.approx(expected_delta)


def test_with_distortion_tf_mode_m2_equals_sigma_squared() -> None:
    result = calculate_nuclear_relaxation_with_distortion(
        hydrogen_host(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="tf_polycrystal", radius=6.0),
    )
    assert result.m2 == pytest.approx(result.sigma_squared)
    assert result.sigma_parallel_squared is None


def test_with_distortion_zf_single_crystal_corrects_parallel_width() -> None:
    result = calculate_nuclear_relaxation_with_distortion(
        hydrogen_host(),
        _distorted_structure(host_position=(0.1, 0.0, 0.0)),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(
            mode="zf_single_crystal",
            direction=(0.0, 0.0, 1.0),
            radius=6.0,
        ),
    )
    assert result.mode == "zf_single_crystal"
    assert result.sigma_perp_squared == pytest.approx(result.sigma_squared)
    assert result.sigma_parallel_squared is not None
    corr = result.distortion_correction
    base = corr.undistorted_result
    expected_parallel = (
        base.sigma_parallel_squared
        + corr.distorted_supercell_result.sigma_parallel_squared
        - corr.pristine_supercell_result.sigma_parallel_squared
    )
    assert result.sigma_parallel_squared == pytest.approx(expected_parallel)


def test_result_and_distortion_to_json_include_nested_payloads() -> None:
    result = calculate_nuclear_relaxation_with_distortion(
        hydrogen_host(),
        _distorted_structure(host_position=(0.1, 0.0, 0.0)),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
    )
    payload = result.to_json()
    assert "distortion_correction" in payload
    dc = payload["distortion_correction"]
    assert isinstance(dc, dict)
    assert len(dc["supercell_matrix"]) == 3
    assert "undistorted_result" in dc
    assert "pristine_supercell_result" in dc
    assert "distorted_supercell_result" in dc
    # Nested results are themselves serialised dicts.
    assert isinstance(dc["undistorted_result"], dict)


# Internal guards the public entry points never reach


def test_lattice_sum_rejects_a_singular_cell() -> None:
    # Private helper: ASE rejects a singular cell before calculate_nuclear_relaxation reaches the lattice sum.
    cell = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(NuclearRelaxationError, match="non-singular"):
        nr._translation_limits(cell, 5.0)


def test_distortion_unwrapping_rejects_mismatched_cells_and_species() -> None:
    # Private helper: calculate_distortion_correction aligns the cells and checks the supercell match first.
    distorted = Atoms("H", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3.0, pbc=True)
    pristine = Atoms("H", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
    with pytest.raises(NuclearRelaxationError, match="cells must match"):
        nr._pair_distorted_to_pristine_indices(distorted, pristine)

    distorted = Atoms("HO", scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 3.0, pbc=True)
    pristine = Atoms("HH", scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 3.0, pbc=True)
    with pytest.raises(NuclearRelaxationError, match="no atoms of species"):
        nr._pair_distorted_to_pristine_indices(distorted, pristine)
