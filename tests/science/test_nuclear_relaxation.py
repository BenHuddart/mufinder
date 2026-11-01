"""Tests for nuclear dipolar relaxation calculations."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import json
import time

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science import nuclear_relaxation as nr
from mufinder.domain.constants import (
    HBAR_J_S,
    METRES_PER_ANGSTROM,
    MU0_OVER_4PI_T_M_PER_A,
    MUON_GAMMA_RAD_S_T,
)
from mufinder.science.nuclear_relaxation import (
    DistortionCorrectionResult,
    Isotope,
    NuclearRelaxationError,
    NuclearRelaxationOptions,
    NuclearWidthTensors,
    calculate_distortion_correction,
    calculate_nuclear_relaxation,
    calculate_nuclear_relaxation_with_distortion,
    calculate_nuclear_width_tensors,
    gyromagnetic_ratio_from_moment,
    isotopes_for_element,
    load_isotopes,
    structure_to_atoms,
    weighted_spin_factor,
)
from mufinder.science.polarisation import fibonacci_sphere
from tests.support.structures import host_structure, hydrogen_host


def _single_nucleus_structure(symbol: str = "H") -> Structure:
    return host_structure(
        label=symbol, lattice=100.0, species=(symbol,), positions=((0.0, 0.0, 0.01),)
    )


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
    assert (second.mass_number, second.abundance, second.spin, second.quadrupole_1e28_m2) == (
        2,
        0.4,
        0.0,
        0.0,
    )
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
        use_distortion_correction=True,
    )
    payload = opts.to_json()
    assert payload["mode"] == "zf_single_crystal"
    assert payload["direction"] == [1.0, 0.0, 0.0]
    assert payload["elements"] == ["H", "Cu"]
    assert not {"radius", "convergence_rtol"} & set(payload)
    assert payload["use_distortion_correction"] is True


def test_options_to_json_none_elements() -> None:
    payload = NuclearRelaxationOptions().to_json()
    assert payload["elements"] is None


def test_result_to_json_without_distortion_omits_key() -> None:
    result = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="zf_single_crystal"),
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
        ),
    )

    h_iso = isotopes_for_element("H")[0]
    r_m = 1.0 * METRES_PER_ANGSTROM
    weight = (h_iso.gamma_rad_s_t**2) * h_iso.spin * (h_iso.spin + 1.0)
    sigma2_s = (
        ((MU0_OVER_4PI_T_M_PER_A**2) * (HBAR_J_S**2) * (MUON_GAMMA_RAD_S_T**2) / 3.0)
        * weight
        * (4.0 / (r_m**6))
    )

    assert result.sigma_squared == pytest.approx(sigma2_s * 1e-12)
    assert result.m2 == pytest.approx(result.sigma_squared)


def test_zf_and_tf_polycrystal_ratio_is_five_over_two() -> None:
    tf = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="tf_polycrystal"),
    )
    zf = calculate_nuclear_relaxation(
        _single_nucleus_structure(),
        (0.0, 0.0, 0.0),
        options=NuclearRelaxationOptions(mode="zf_polycrystal"),
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
        ),
    )

    h_iso = isotopes_for_element("H")[0]
    r_m = np.sqrt(2.0) * METRES_PER_ANGSTROM
    cos2 = 0.5
    weight = (h_iso.gamma_rad_s_t**2) * h_iso.spin * (h_iso.spin + 1.0)
    base = (MU0_OVER_4PI_T_M_PER_A**2) * (HBAR_J_S**2) * (MUON_GAMMA_RAD_S_T**2) * weight / (r_m**6)
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
        options=NuclearRelaxationOptions(mode="zf_polycrystal"),
    )

    # Deterministic Fibonacci spherical design: near-uniform coverage with
    # quadrature error well below the assertion tolerance at modest cost.
    directions = fibonacci_sphere(64)

    perp_values = []
    parallel_values = []
    for direction in directions:
        single = calculate_nuclear_relaxation(
            structure,
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(
                mode="zf_single_crystal",
                direction=tuple(direction),
            ),
        )
        perp_values.append(single.sigma_perp_squared)
        parallel_values.append(single.sigma_parallel_squared)

    assert np.mean(perp_values) == pytest.approx(zf_poly.sigma_squared, rel=0.01)
    assert np.mean(parallel_values) == pytest.approx(zf_poly.sigma_squared, rel=0.01)


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
        options=NuclearRelaxationOptions(mode="zf_polycrystal", elements=("H",)),
    )
    assert set(only_h.per_element_sigma_squared) == {"H"}


def test_fractional_muon_position_is_wrapped_into_the_cell() -> None:
    structure = hydrogen_host(3.0)
    options = NuclearRelaxationOptions(mode="zf_polycrystal")

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
            ),
        )


def test_missing_isotope_data_reports_element() -> None:
    with pytest.raises(NuclearRelaxationError, match="He"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure("He"),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(),
        )


def test_unsupported_mode_raises() -> None:
    with pytest.raises(NuclearRelaxationError, match="Unsupported mode"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure(),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(mode="bogus"),  # type: ignore[arg-type]
        )


def test_selecting_no_nuclei_raises() -> None:
    with pytest.raises(NuclearRelaxationError, match="No nuclei found within 20 Å"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure(),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(mode="zf_polycrystal", elements=("O",)),
        )


def test_zero_direction_vector_raises_in_single_crystal_mode() -> None:
    with pytest.raises(NuclearRelaxationError, match="direction must be non-zero"):
        calculate_nuclear_relaxation(
            _single_nucleus_structure(),
            (0.0, 0.0, 0.0),
            options=NuclearRelaxationOptions(
                mode="tf_single_crystal",
                direction=(0.0, 0.0, 0.0),
            ),
        )


# Distortion correction


def test_distortion_correction_is_zero_for_identical_host_positions() -> None:
    result = calculate_distortion_correction(
        hydrogen_host(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="zf_polycrystal"),
    )

    assert result.supercell_matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert result.distortion_correction_sigma_squared == pytest.approx(0.0, abs=1e-12)
    assert result.sigma_squared_corrected == pytest.approx(result.sigma_squared_undistorted)


def test_distortion_correction_changes_for_local_displacement() -> None:
    result = calculate_distortion_correction(
        hydrogen_host(),
        _distorted_structure(host_position=(0.1, 0.0, 0.0)),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="zf_polycrystal"),
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
        (0.25, 0.5, 0.25),
        options=NuclearRelaxationOptions(mode="zf_polycrystal"),
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
            label="distorted",
            lattice=3.0,
            species=species,
            positions=positions,
            parent_structure_id=pristine.id,
        )

    muon = (0.25, 0.25, 0.25)
    hydrogens = ((0.02, 0.0, 0.0), (0.5, 0.0, 0.0))
    same_order = distorted(("H", "H", "Li", "H:mu"), (*hydrogens, (0.5, 0.5, 0.5), muon))
    reordered = distorted(("Li", "H", "H", "H:mu"), ((0.5, 0.5, 0.5), *hydrogens, muon))
    options = NuclearRelaxationOptions(mode="zf_polycrystal")

    expected = calculate_distortion_correction(pristine, same_order, muon, options=options)
    result = calculate_distortion_correction(pristine, reordered, muon, options=options)

    assert abs(result.distortion_correction_sigma_squared) > 1e-12
    assert result.distortion_correction_sigma_squared == pytest.approx(
        expected.distortion_correction_sigma_squared
    )


def test_distortion_correction_reports_unmatched_supercell() -> None:
    with pytest.raises(NuclearRelaxationError, match="matching pristine supercell"):
        calculate_distortion_correction(
            hydrogen_host(),
            _distorted_structure(cell_length=2.5),
            (0.5, 0.5, 0.5),
            options=NuclearRelaxationOptions(mode="zf_polycrystal"),
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
            options=NuclearRelaxationOptions(mode="zf_polycrystal"),
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
            options=NuclearRelaxationOptions(mode="zf_polycrystal"),
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
        options=NuclearRelaxationOptions(mode="zf_polycrystal"),
    )
    assert result.mode == "zf_polycrystal"
    assert result.m2 == pytest.approx(2.0 * result.sigma_squared)
    assert result.distortion_correction is not None
    assert result.sigma_perp_squared is None
    # delta_field_t derives from sigma_corrected.
    expected_delta = (result.sigma * 1e6) / MUON_GAMMA_RAD_S_T
    assert result.delta_field_t == pytest.approx(expected_delta)


def test_with_distortion_tf_mode_m2_equals_sigma_squared() -> None:
    result = calculate_nuclear_relaxation_with_distortion(
        hydrogen_host(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="tf_polycrystal"),
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
        options=NuclearRelaxationOptions(mode="zf_polycrystal"),
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


# Moment tensors: every direction from one lattice sum


def _low_symmetry_host() -> Structure:
    """An orthorhombic H/F/Cu host with no symmetry that would hide a wrong tensor index."""

    return Structure(
        label="HFCu",
        lattice_vectors=[[4.1, 0.0, 0.0], [0.3, 5.2, 0.0], [0.2, 0.4, 6.3]],
        species=["H", "F", "Cu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.31, 0.17, 0.62], [0.55, 0.71, 0.24]],
    )


_TENSOR_SITE = (0.23, 0.41, 0.37)


def _random_directions(count: int = 10) -> np.ndarray:
    directions = np.random.default_rng(254).normal(size=(count, 3))
    return directions / np.linalg.norm(directions, axis=1)[:, None]


def test_width_tensors_give_the_direct_single_crystal_widths_in_any_direction() -> None:
    structure = _low_symmetry_host()
    tensors = calculate_nuclear_width_tensors(structure, _TENSOR_SITE)

    for direction in _random_directions():
        tf = calculate_nuclear_relaxation(
            structure,
            _TENSOR_SITE,
            options=NuclearRelaxationOptions(mode="tf_single_crystal", direction=tuple(direction)),
        )
        zf = calculate_nuclear_relaxation(
            structure,
            _TENSOR_SITE,
            options=NuclearRelaxationOptions(mode="zf_single_crystal", direction=tuple(direction)),
        )
        assert tensors.tf_sigma_squared(direction) == pytest.approx(tf.sigma_squared, rel=1e-10)
        assert tensors.zf_sigma_squared(direction) == pytest.approx(zf.sigma_squared, rel=1e-10)
        assert tensors.zf_parallel_sigma_squared(direction) == pytest.approx(
            zf.sigma_parallel_squared, rel=1e-10
        )
    # Many directions at once give the same values as one at a time.
    directions = _random_directions()
    np.testing.assert_allclose(
        tensors.tf_sigma_squared(directions),
        [tensors.tf_sigma_squared(direction) for direction in directions],
        rtol=1e-12,
    )


def test_width_tensor_sphere_averages_are_the_polycrystal_widths() -> None:
    structure = _low_symmetry_host()
    tensors = calculate_nuclear_width_tensors(structure, _TENSOR_SITE)
    poly = {
        mode: calculate_nuclear_relaxation(
            structure, _TENSOR_SITE, options=NuclearRelaxationOptions(mode=mode)
        ).sigma_squared
        for mode in ("tf_polycrystal", "zf_polycrystal")
    }
    # A 2000-point Fibonacci sphere integrates the quartic exactly enough for 1e-4.
    sphere = fibonacci_sphere(2000)

    assert np.mean(tensors.tf_sigma_squared(sphere)) == pytest.approx(
        poly["tf_polycrystal"], rel=1e-4
    )
    assert np.mean(tensors.zf_sigma_squared(sphere)) == pytest.approx(
        poly["zf_polycrystal"], rel=1e-4
    )
    assert tensors.tf_polycrystal_sigma_squared == pytest.approx(poly["tf_polycrystal"], rel=1e-12)
    assert tensors.zf_polycrystal_sigma_squared == pytest.approx(poly["zf_polycrystal"], rel=1e-12)


def test_width_tensors_put_the_quadrupolar_factor_on_the_zero_field_widths_only() -> None:
    structure = _low_symmetry_host()
    plain = calculate_nuclear_width_tensors(structure, _TENSOR_SITE)
    quadrupolar = calculate_nuclear_width_tensors(structure, _TENSOR_SITE, include_quadrupolar=True)
    zf = calculate_nuclear_relaxation(
        structure,
        _TENSOR_SITE,
        options=NuclearRelaxationOptions(
            mode="zf_single_crystal",
            direction=(0.0, 0.6, 0.8),
            include_quadrupolar=True,
        ),
    )

    # Cu (I = 3/2) carries the quadrupolar factor; H and F (I = 1/2) do not.
    assert quadrupolar.zf_sigma_squared((0.0, 0.6, 0.8)) == pytest.approx(
        zf.sigma_squared, rel=1e-10
    )
    assert quadrupolar.zf_isotropic != pytest.approx(plain.zf_isotropic)
    assert quadrupolar.isotropic == pytest.approx(plain.isotropic, rel=1e-12)
    np.testing.assert_allclose(quadrupolar.quartic, plain.quartic, rtol=1e-12)


def test_width_tensors_wrap_the_site_and_round_trip_through_json() -> None:
    tensors = calculate_nuclear_width_tensors(
        _low_symmetry_host(), (1.23, -0.59, 0.37), site_label="site 2", elements=["H", "F"]
    )
    restored = NuclearWidthTensors.from_json(json.loads(json.dumps(tensors.to_json())))

    assert tensors.convergence_radius_a >= nr.NUCLEAR_SUM_RADIUS_A
    assert tensors.muon_position == pytest.approx(_TENSOR_SITE)
    assert restored.site_label == "site 2"
    assert restored.tf_sigma_squared((1.0, 2.0, 3.0)) == pytest.approx(
        tensors.tf_sigma_squared((1.0, 2.0, 3.0)), rel=1e-12
    )
    assert restored.n_nuclei_summed == tensors.n_nuclei_summed
    # Only H and F are summed: their quadrupolar weights equal the plain ones.
    assert tensors.zf_isotropic == pytest.approx(tensors.isotropic, rel=1e-12)


@pytest.mark.parametrize(
    ("direction", "message"),
    [((0.0, 0.0, 0.0), "non-zero"), ((1.0, 0.0), "3 components")],
)
def test_width_tensors_reject_a_bad_direction(direction, message: str) -> None:
    tensors = calculate_nuclear_width_tensors(_low_symmetry_host(), _TENSOR_SITE)

    with pytest.raises(NuclearRelaxationError, match=message):
        tensors.tf_sigma_squared(direction)


def test_width_tensors_reject_bad_inputs() -> None:
    with pytest.raises(NuclearRelaxationError, match="No nuclei"):
        calculate_nuclear_width_tensors(
            _single_nucleus_structure(), (0.5, 0.5, 0.5), elements=["O"]
        )
    with pytest.raises(NuclearRelaxationError, match="malformed"):
        NuclearWidthTensors.from_json({"site_label": "site 1", "quadratic": [1.0]})


# The lattice sum: one radius, translations from the plane heights, an analytic tail

_CU_A = 3.615
_CU_PRIMITIVE = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * _CU_A
#: The same fcc lattice in a sheared basis: its shortest vector is far longer than a plane height.
_CU_SHEARED = np.array([[1, 0, 0], [3, 1, 0], [-2, 2, 1]]) @ _CU_PRIMITIVE
_ALL_MODES = (
    ("zf_polycrystal", (0.0, 0.0, 1.0)),
    ("tf_polycrystal", (0.0, 0.0, 1.0)),
    ("zf_single_crystal", (0.0, 0.0, 1.0)),
    ("zf_single_crystal", (1.0, 1.0, 1.0)),
    ("tf_single_crystal", (0.0, 0.0, 1.0)),
    ("tf_single_crystal", (1.0, 1.0, 1.0)),
)


def _cu_conventional() -> Structure:
    return host_structure(
        label="Cu",
        lattice=_CU_A,
        species=("Cu", "Cu", "Cu", "Cu"),
        positions=((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
    )


def _cu_in_basis(basis: np.ndarray) -> Structure:
    return Structure(
        label="Cu",
        lattice_vectors=basis.tolist(),
        species=["Cu"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _cof2_host() -> Structure:
    x = 0.306
    return Structure(
        label="CoF2",
        lattice_vectors=[[4.695, 0.0, 0.0], [0.0, 4.695, 0.0], [0.0, 0.0, 3.18]],
        species=["Co", "Co", "F", "F", "F", "F"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [x, x, 0.0],
            [-x, -x, 0.0],
            [0.5 + x, 0.5 - x, 0.5],
            [0.5 - x, 0.5 + x, 0.5],
        ],
    )


def _direct_sums(
    structure: Structure, muon: tuple[float, float, float], radius: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``S``, ``M_ij`` and ``M_ijkl`` to ``radius`` by a plain sum over a generous grid."""

    lattice = np.asarray(structure.lattice_vectors, dtype=float)
    reach = radius * np.linalg.norm(np.linalg.inv(lattice), axis=0) + np.abs(muon) + 2.0
    axes = (np.arange(-limit, limit + 1) for limit in np.ceil(reach).astype(int))
    translations = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3) @ lattice
    prefactor = (MU0_OVER_4PI_T_M_PER_A * HBAR_J_S * MUON_GAMMA_RAD_S_T) ** 2 / 3.0 * 1e-12
    isotropic, quadratic, quartic = 0.0, np.zeros((3, 3)), np.zeros((3, 3, 3, 3))
    for symbol, position in zip(structure.species, structure.fractional_positions, strict=True):
        vectors = translations + (np.asarray(position) - np.asarray(muon)) @ lattice
        vectors = vectors[np.einsum("ij,ij->i", vectors, vectors) <= radius**2]
        distances = np.linalg.norm(vectors, axis=1)
        unit = vectors / distances[:, None]
        terms = (
            prefactor
            * weighted_spin_factor(isotopes_for_element(symbol))
            / (distances * METRES_PER_ANGSTROM) ** 6
        )
        isotropic += float(np.sum(terms))
        quadratic += np.einsum("k,ki,kj->ij", terms, unit, unit)
        for i in range(3):
            quartic[i] += np.einsum("k,k,kj,kl,km->jlm", terms, unit[:, i], unit, unit, unit)
    return isotropic, quadratic, quartic


def _tensors_from_sums(site, isotropic, quadratic, quartic) -> NuclearWidthTensors:
    return NuclearWidthTensors(
        site_label="direct sum",
        muon_position=site,
        isotropic=isotropic,
        quadratic=quadratic,
        quartic=quartic,
        zf_isotropic=isotropic,
        zf_quadratic=quadratic,
        include_quadrupolar=False,
        convergence_radius_a=0.0,
        n_nuclei_summed=0,
    )


def _width_along(tensors: NuclearWidthTensors, mode: str, direction) -> float:
    return {
        "zf_polycrystal": lambda: tensors.zf_polycrystal_sigma_squared,
        "tf_polycrystal": lambda: tensors.tf_polycrystal_sigma_squared,
        "zf_single_crystal": lambda: tensors.zf_sigma_squared(direction),
        "tf_single_crystal": lambda: tensors.tf_sigma_squared(direction),
    }[mode]()


def _relaxation(structure: Structure, site, mode: str, direction):
    return calculate_nuclear_relaxation(
        structure, site, options=NuclearRelaxationOptions(mode=mode, direction=direction)
    )


@pytest.mark.parametrize(("mode", "direction"), _ALL_MODES)
def test_the_same_lattice_in_a_sheared_basis_gives_the_same_width(mode: str, direction) -> None:
    octahedral = np.array([0.5, 0.5, 0.5]) * _CU_A
    conventional = _relaxation(_cu_conventional(), (0.5, 0.5, 0.5), mode, direction)

    for basis in (_CU_PRIMITIVE, _CU_SHEARED):
        site = tuple(octahedral @ np.linalg.inv(basis))
        result = _relaxation(_cu_in_basis(basis), site, mode, direction)
        assert result.sigma == pytest.approx(conventional.sigma, rel=1e-3)
        assert result.n_nuclei_summed == conventional.n_nuclei_summed


def test_an_extreme_shear_of_the_cell_costs_nothing_and_changes_nothing() -> None:
    # A unimodular shear of 1000 leaves the lattice alone but makes one plane height 1/1000 of
    # a vector: a translation grid in the input basis would hold about 10⁸ cells.
    basis = np.array([[1, 0, 0], [1000, 1, 0], [0, 0, 1]]) @ _CU_PRIMITIVE
    site = tuple((np.array([0.5, 0.5, 0.5]) * _CU_A) @ np.linalg.inv(basis))
    expected = calculate_nuclear_width_tensors(_cu_conventional(), (0.5, 0.5, 0.5))

    tensors = calculate_nuclear_width_tensors(_cu_in_basis(basis), site)

    assert tensors.isotropic == pytest.approx(expected.isotropic, rel=1e-9)
    assert tensors.n_nuclei_summed == expected.n_nuclei_summed


@pytest.mark.parametrize(("mode", "direction"), _ALL_MODES)
def test_the_sum_with_its_tail_matches_a_brute_force_sum_to_60_angstrom(
    mode: str, direction
) -> None:
    host = _cof2_host()
    site = (0.0, 0.5, 0.0)
    isotropic, quadratic, quartic = _direct_sums(host, site, 60.0)
    brute_force = _tensors_from_sums(site, isotropic, quadratic, quartic)

    result = _relaxation(host, site, mode, direction)

    assert result.sigma_squared == pytest.approx(
        _width_along(brute_force, mode, direction), rel=5e-3
    )
    assert result.convergence_radius_a == nr.NUCLEAR_SUM_RADIUS_A


def test_the_tail_is_the_continuum_integral_with_its_isotropic_share_of_each_tensor() -> None:
    host = _cof2_host()
    site = (0.0, 0.5, 0.0)
    isotropic, quadratic, quartic = _direct_sums(host, site, nr.NUCLEAR_SUM_RADIUS_A)
    prefactor = (MU0_OVER_4PI_T_M_PER_A * HBAR_J_S * MUON_GAMMA_RAD_S_T) ** 2 / 3.0 * 1e-12
    weight = sum(weighted_spin_factor(isotopes_for_element(symbol)) for symbol in host.species)
    volume = 4.695 * 4.695 * 3.18 * METRES_PER_ANGSTROM**3
    radius = nr.NUCLEAR_SUM_RADIUS_A * METRES_PER_ANGSTROM
    tail = prefactor * 4.0 * np.pi * (weight / volume) / (3.0 * radius**3)

    tensors = calculate_nuclear_width_tensors(host, site)

    assert tensors.isotropic == pytest.approx(isotropic + tail, rel=1e-9)
    np.testing.assert_allclose(
        tensors.quadratic - quadratic, tail * np.eye(3) / 3.0, atol=1e-9 * tail
    )
    # Contracting the tail's share gives S back, and it adds 4/5 S along any direction.
    excess = tensors.quartic - quartic
    assert np.einsum("iijj", excess) == pytest.approx(tail, rel=1e-6)
    plain = _tensors_from_sums(site, isotropic, quadratic, quartic)
    for direction in _random_directions(4):
        assert tensors.tf_sigma_squared(direction) - plain.tf_sigma_squared(direction) == (
            pytest.approx(0.8 * tail, rel=1e-6)
        )
        assert tensors.zf_sigma_squared(direction) - plain.zf_sigma_squared(direction) == (
            pytest.approx(2.0 * tail, rel=1e-6)
        )


def test_the_tail_counts_only_the_selected_elements_and_their_quadrupolar_weights() -> None:
    host = _cof2_host()
    site = (0.0, 0.5, 0.0)
    everything = _relaxation(host, site, "zf_polycrystal", (0.0, 0.0, 1.0))
    fluorine = calculate_nuclear_relaxation(
        host, site, options=NuclearRelaxationOptions(mode="zf_polycrystal", elements=("F",))
    )
    quadrupolar = calculate_nuclear_width_tensors(
        _cu_conventional(), (0.5, 0.5, 0.5), include_quadrupolar=True
    )
    copper = calculate_nuclear_width_tensors(_cu_conventional(), (0.5, 0.5, 0.5))

    assert sum(everything.per_element_sigma_squared.values()) == pytest.approx(
        everything.sigma_squared, rel=1e-12
    )
    assert fluorine.sigma_squared == pytest.approx(
        everything.per_element_sigma_squared["F"], rel=1e-12
    )
    # Copper's two isotopes are both I = 3/2, so one factor scales the whole sum and its tail.
    assert quadrupolar.zf_isotropic / copper.isotropic == pytest.approx(
        (16.0 * 3.75 + 3.0 * 4.0) / (24.0 * 3.75), rel=1e-12
    )


@pytest.mark.parametrize(("mode", "direction"), _ALL_MODES)
def test_sparse_nuclei_reach_further_and_keep_the_tail_accurate(mode: str, direction) -> None:
    # One nucleus in a 16 Å cell, the muon as far from it as it can be: 20 Å would hold only
    # the eight corners and the isotropic tail would be 4 % out along the worst direction.
    host = host_structure(label="Cu", lattice=16.0, species=("Cu",), positions=((0.0, 0.0, 0.0),))
    site = (0.5, 0.5, 0.5)
    isotropic, quadratic, quartic = _direct_sums(host, site, 200.0)
    brute_force = _tensors_from_sums(site, isotropic, quadratic, quartic)

    result = _relaxation(host, site, mode, direction)

    assert result.convergence_radius_a == pytest.approx(80.0)
    assert result.sigma_squared == pytest.approx(
        _width_along(brute_force, mode, direction), rel=5e-3
    )


def test_a_dilute_element_selected_alone_still_gives_a_width() -> None:
    # In a 44 Å cell only the one Fe is selected, and it lies 38 Å from the muon.
    host = Structure(
        label="dilute",
        lattice_vectors=[[44.0, 0.0, 0.0], [0.0, 44.0, 0.0], [0.0, 0.0, 44.0]],
        species=["Cu", "Fe"],
        fractional_positions=[[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    )

    result = calculate_nuclear_relaxation(
        host, (0.5, 0.5, 0.5), options=NuclearRelaxationOptions(elements=("Fe",))
    )

    assert result.sigma > 0.0
    assert result.convergence_radius_a == pytest.approx(220.0)
    assert set(result.per_element_sigma_squared) == {"Fe"}


def test_a_dilute_element_keeps_its_reach_beside_dense_ones() -> None:
    # 64 Fe make the cell dense, but its one H needs the same reach as if it were alone, so
    # the width of the whole host is the sum of the widths of its elements.
    coordinates = (2.0, 6.0, 10.0, 14.0)
    fe = [
        [x / 16.0, y / 16.0, z / 16.0]
        for x in coordinates
        for y in coordinates
        for z in coordinates
    ]
    host = Structure(
        label="dilute H",
        lattice_vectors=[[16.0, 0.0, 0.0], [0.0, 16.0, 0.0], [0.0, 0.0, 16.0]],
        species=["H", *["Fe"] * 64],
        fractional_positions=[[0.0, 0.0, 0.0], *fe],
    )
    site = (0.5, 0.5, 0.5)
    options = {"mode": "tf_single_crystal", "direction": (0.0, 0.0, 1.0)}

    whole = calculate_nuclear_relaxation(host, site, options=NuclearRelaxationOptions(**options))
    parts = [
        calculate_nuclear_relaxation(
            host, site, options=NuclearRelaxationOptions(elements=(element,), **options)
        )
        for element in ("H", "Fe")
    ]

    assert whole.sigma_squared == pytest.approx(sum(p.sigma_squared for p in parts), rel=1e-12)
    assert whole.per_element_sigma_squared["H"] == pytest.approx(parts[0].sigma_squared, rel=1e-12)
    assert whole.convergence_radius_a == pytest.approx(80.0)


def test_a_muon_in_the_vacuum_of_a_slab_reaches_the_slab() -> None:
    # The mean density of this cell gives about 20 Å, and the nearest nucleus is 25 Å away.
    host = Structure(
        label="slab",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 60.0]],
        species=["H"] * 8,
        fractional_positions=[[0.0, 0.0, z / 60.0] for z in range(8)],
    )
    site = (0.0, 0.0, 35.0 / 60.0)
    isotropic, _quadratic, _quartic = _direct_sums(host, site, 200.0)

    tensors = calculate_nuclear_width_tensors(host, site)

    assert tensors.convergence_radius_a == pytest.approx(50.0)
    assert tensors.isotropic == pytest.approx(isotropic, rel=5e-3)


def test_tensors_and_modes_agree_with_the_tail_included() -> None:
    host = _low_symmetry_host()
    tensors = calculate_nuclear_width_tensors(host, _TENSOR_SITE)

    for mode, direction in _ALL_MODES:
        assert _relaxation(host, _TENSOR_SITE, mode, direction).sigma_squared == pytest.approx(
            _width_along(tensors, mode, direction), rel=1e-10
        )


def test_a_muon_outside_the_home_cell_gives_the_same_width() -> None:
    host = _low_symmetry_host()
    home = calculate_nuclear_width_tensors(host, _TENSOR_SITE)
    away = calculate_nuclear_width_tensors(
        host, (_TENSOR_SITE[0] + 7.0, _TENSOR_SITE[1] - 11.0, _TENSOR_SITE[2] + 3.0)
    )

    assert away.isotropic == pytest.approx(home.isotropic, rel=1e-12)
    assert away.n_nuclei_summed == home.n_nuclei_summed


def test_a_cof2_width_takes_under_150_milliseconds() -> None:
    host = _cof2_host()
    timings = []
    for _ in range(5):
        start = time.perf_counter()
        calculate_nuclear_relaxation(host, (0.0, 0.5, 0.0))
        calculate_nuclear_width_tensors(host, (0.0, 0.5, 0.0))
        timings.append(time.perf_counter() - start)

    assert min(timings) < 0.150


def test_options_carry_neither_a_radius_nor_a_tolerance() -> None:
    assert not {"radius", "convergence_rtol"} & set(NuclearRelaxationOptions().to_json())
    with pytest.raises(TypeError):
        NuclearRelaxationOptions(radius=8.0)  # type: ignore[call-arg]
