"""Tests for nuclear dipolar relaxation calculations."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.nuclear_relaxation import (
    ANGSTROM_TO_M,
    GAMMA_MU_RAD_S_T,
    HBAR,
    MU0_OVER_4PI,
    NuclearRelaxationError,
    NuclearRelaxationOptions,
    calculate_distortion_correction,
    calculate_nuclear_relaxation,
    gyromagnetic_ratio_from_moment,
    isotopes_for_element,
)


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
        lattice_vectors=[[cell_length, 0.0, 0.0], [0.0, cell_length, 0.0], [0.0, 0.0, cell_length]],
        species=["H", "H:mu"],
        fractional_positions=[list(host_position), [0.5, 0.5, 0.5]],
        parent_structure_id=_host_structure().id,
    )


def test_isotope_table_loads_cu_and_gamma_reference() -> None:
    cu = isotopes_for_element("Cu")

    assert len(cu) == 2
    assert cu[0].mass_number == 63
    assert cu[1].mass_number == 65
    assert cu[0].abundance + cu[1].abundance == pytest.approx(1.0)
    gamma = gyromagnetic_ratio_from_moment(moment_mu_n=2.227, spin=1.5)
    assert gamma / (2.0 * np.pi * 1e6) == pytest.approx(11.317, rel=2e-4)


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


def test_distortion_correction_is_zero_for_identical_host_positions() -> None:
    result = calculate_distortion_correction(
        _host_structure(),
        _distorted_structure(),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
    )

    assert result.supercell_matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    assert result.distortion_correction_sigma_squared == pytest.approx(0.0, abs=1e-12)
    assert result.sigma_squared_corrected == pytest.approx(result.sigma_squared_undistorted)


def test_distortion_correction_changes_for_local_displacement() -> None:
    result = calculate_distortion_correction(
        _host_structure(),
        _distorted_structure(host_position=(0.1, 0.0, 0.0)),
        (0.5, 0.5, 0.5),
        options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
    )

    assert abs(result.distortion_correction_sigma_squared) > 1e-12


def test_distortion_correction_reports_unmatched_supercell() -> None:
    with pytest.raises(NuclearRelaxationError, match="matching pristine supercell"):
        calculate_distortion_correction(
            _host_structure(),
            _distorted_structure(cell_length=2.5),
            (0.5, 0.5, 0.5),
            options=NuclearRelaxationOptions(mode="zf_polycrystal", radius=6.0),
        )

