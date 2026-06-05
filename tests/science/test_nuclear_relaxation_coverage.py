"""Additional coverage for nuclear dipolar relaxation helpers.

These tests target functions, branches, and error paths not exercised by
``tests/science/test_nuclear_relaxation.py``: the ``to_json`` serialisers,
quadrupolar weighting branches, validation/error paths, the distortion-aware
public entry point, and the low-level geometry/parsing helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from mufinder.domain.models import Structure
from mufinder.science.nuclear_relaxation import (
    GAMMA_MU_RAD_S_T,
    DistortionCorrectionResult,
    Isotope,
    NuclearRelaxationError,
    NuclearRelaxationOptions,
    NuclearRelaxationResult,
    calculate_distortion_correction,
    calculate_nuclear_relaxation,
    calculate_nuclear_relaxation_with_distortion,
    gyromagnetic_ratio_from_moment,
    isotopes_for_element,
    load_isotopes,
    structure_to_atoms,
    weighted_spin_factor,
)
from mufinder.science import nuclear_relaxation as nr


# ---------------------------------------------------------------------------
# Fixtures / helpers (mirroring the existing test module's structure builders)
# ---------------------------------------------------------------------------


def _single_nucleus_structure(symbol: str = "H") -> Structure:
    return Structure(
        label=symbol,
        lattice_vectors=[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]],
        species=[symbol],
        fractional_positions=[[0.0, 0.0, 0.01]],
    )


def _host_structure() -> Structure:
    return Structure(
        label="H host",
        lattice_vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        species=["H"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _distorted_structure(host_position=(0.0, 0.0, 0.0), cell_length: float = 2.0) -> Structure:
    return Structure(
        label="distorted",
        lattice_vectors=[
            [cell_length, 0.0, 0.0],
            [0.0, cell_length, 0.0],
            [0.0, 0.0, cell_length],
        ],
        species=["H", "H:mu"],
        fractional_positions=[list(host_position), [0.5, 0.5, 0.5]],
        parent_structure_id=_host_structure().id,
    )


# ---------------------------------------------------------------------------
# gyromagnetic_ratio_from_moment / weighted_spin_factor
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# isotope loading / lookup
# ---------------------------------------------------------------------------


def test_load_isotopes_is_cached() -> None:
    assert load_isotopes() is load_isotopes()


def test_isotopes_for_element_unknown_raises() -> None:
    with pytest.raises(NuclearRelaxationError, match="No isotope data"):
        isotopes_for_element("Zz")


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


# ---------------------------------------------------------------------------
# to_json serialisers
# ---------------------------------------------------------------------------


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


def test_result_and_distortion_to_json_include_nested_payloads() -> None:
    result = calculate_nuclear_relaxation_with_distortion(
        _host_structure(),
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


# ---------------------------------------------------------------------------
# calculate_nuclear_relaxation_with_distortion
# ---------------------------------------------------------------------------


def test_with_distortion_zf_polycrystal_m2_is_double_sigma_squared() -> None:
    result = calculate_nuclear_relaxation_with_distortion(
        _host_structure(),
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
        _host_structure(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="tf_polycrystal", radius=6.0),
    )
    assert result.m2 == pytest.approx(result.sigma_squared)
    assert result.sigma_parallel_squared is None


def test_with_distortion_zf_single_crystal_corrects_parallel_width() -> None:
    result = calculate_nuclear_relaxation_with_distortion(
        _host_structure(),
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


def test_with_distortion_default_options_enable_correction() -> None:
    # No options -> default NuclearRelaxationOptions(use_distortion_correction=True)
    result = calculate_nuclear_relaxation_with_distortion(
        _host_structure(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
    )
    assert result.distortion_correction is not None
    assert result.mode == "zf_polycrystal"


# ---------------------------------------------------------------------------
# structure_to_atoms
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# validation / error paths in the second-moment driver
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# distortion correction error / mapping branches
# ---------------------------------------------------------------------------


def test_distortion_correction_requires_muon_marker() -> None:
    no_muon = Structure(
        label="no muon",
        lattice_vectors=[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        species=["H"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        parent_structure_id=_host_structure().id,
    )
    with pytest.raises(NuclearRelaxationError, match="does not contain a muon marker"):
        calculate_distortion_correction(
            _host_structure(),
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
        parent_structure_id=_host_structure().id,
    )
    with pytest.raises(NuclearRelaxationError, match="more than one muon marker"):
        calculate_distortion_correction(
            _host_structure(),
            two_muons,
            (0.5, 0.5, 0.5),
            options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
        )


def test_distortion_correction_multi_atom_mapping_path() -> None:
    # A 2-atom host -> a 1x2x1 supercell distortion exercises the nearest-
    # neighbour mapping branch (atom ordering differs from pristine supercell).
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


# ---------------------------------------------------------------------------
# low-level helper branches reached directly
# ---------------------------------------------------------------------------


def test_element_symbol_rejects_empty() -> None:
    with pytest.raises(NuclearRelaxationError, match="must not be empty"):
        nr._element_symbol("   ")


def test_element_symbol_rejects_unresolvable() -> None:
    with pytest.raises(NuclearRelaxationError, match="Could not resolve element"):
        nr._element_symbol("123")


def test_element_symbol_strips_trailing_digits() -> None:
    assert nr._element_symbol("Cu63") == "Cu"
    assert nr._element_symbol("H1") == "H"


def test_translation_limits_rejects_singular_cell() -> None:
    cell = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(NuclearRelaxationError, match="non-singular"):
        nr._translation_limits(cell, 5.0)


def test_canonical_fractional_wraps_and_snaps() -> None:
    assert nr._canonical_fractional((1.0, 1.5, -0.25)) == pytest.approx((0.0, 0.5, 0.75))
    # Values numerically equal to 1.0 snap back to 0.0.
    assert nr._canonical_fractional((1.0 - 1e-12, 0.0, 0.0))[0] == pytest.approx(0.0)


def test_parse_csv_numbers_handles_fractions_and_blanks() -> None:
    assert nr._parse_csv_numbers("") == []
    assert nr._parse_csv_numbers("1/2, 3, 0.5") == [0.5, 3.0, 0.5]


def test_parse_number_token_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty numeric token"):
        nr._parse_number_token("   ")


def test_parse_csv_ints_handles_blanks() -> None:
    assert nr._parse_csv_ints("") == []
    assert nr._parse_csv_ints("63, 65") == [63, 65]


def test_pad_extends_to_count() -> None:
    assert nr._pad([1.0], 3, 0.0) == [1.0, 0.0, 0.0]
    # Already long enough -> unchanged.
    assert nr._pad([1.0, 2.0], 1, 0.0) == [1.0, 2.0]


def test_pair_distorted_to_pristine_rejects_mismatched_cells() -> None:
    distorted = Atoms("H", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 3.0, pbc=True)
    pristine = Atoms("H", scaled_positions=[[0.0, 0.0, 0.0]], cell=np.eye(3) * 4.0, pbc=True)
    with pytest.raises(NuclearRelaxationError, match="cells must match"):
        nr._pair_distorted_to_pristine_indices(distorted, pristine)


def test_pair_distorted_to_pristine_rejects_missing_species() -> None:
    distorted = Atoms("HO", scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 3.0, pbc=True)
    pristine = Atoms("HH", scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], cell=np.eye(3) * 3.0, pbc=True)
    with pytest.raises(NuclearRelaxationError, match="no atoms of species"):
        nr._pair_distorted_to_pristine_indices(distorted, pristine)


def test_pair_distorted_to_pristine_identity_fast_path() -> None:
    cell = np.eye(3) * 3.0
    atoms = Atoms("HH", scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], cell=cell, pbc=True)
    mapping = nr._pair_distorted_to_pristine_indices(atoms.copy(), atoms.copy())
    assert mapping == [0, 1]


def test_pair_distorted_to_pristine_nearest_neighbour_mapping() -> None:
    cell = np.eye(3) * 3.0
    # Symbol lists differ in *order* (HOH vs OHH), so the identity fast-path is
    # skipped and the nearest-neighbour species mapping loop is exercised. The
    # distorted O sits nearest the pristine O, and each H maps to its closest H.
    pristine = Atoms(
        "OHH",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        cell=cell,
        pbc=True,
    )
    distorted = Atoms(
        "HOH",
        scaled_positions=[[0.49, 0.01, 0.0], [0.01, 0.99, 0.99], [0.0, 0.51, 0.0]],
        cell=cell,
        pbc=True,
    )
    mapping = nr._pair_distorted_to_pristine_indices(distorted, pristine)
    # distorted[0]=H near pristine[1]; distorted[1]=O near pristine[0];
    # distorted[2]=H near pristine[2].
    assert mapping == [1, 0, 2]


def test_result_dataclass_to_json_with_explicit_distortion() -> None:
    base = NuclearRelaxationResult(
        site_label="s",
        muon_position=(0.0, 0.0, 0.0),
        sigma=1.0,
        sigma_squared=1.0,
        m2=2.0,
        delta_field_t=0.5,
        mode="zf_polycrystal",
        convergence_radius_a=6.0,
        n_nuclei_summed=3,
        per_element_sigma_squared={"H": 1.0},
    )
    payload = base.to_json()
    # distortion_correction is None on the dataclass, so the key is omitted.
    assert "distortion_correction" not in payload
    assert payload["sigma_parallel_squared"] is None
