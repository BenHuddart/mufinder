"""Tests for the physical constants in :mod:`mufinder.domain.constants`."""

from __future__ import annotations

import math

import pytest
import scipy.constants

from mufinder.castep_io import ts_parser
from mufinder.domain import constants
from mufinder.domain.constants import (
    MUON_GAMMA_OVER_2PI_MHZ_T,
    MUON_GAMMA_RAD_S_T,
    MUON_GAMMA_RAD_US_T,
)
from mufinder.science import magnetic_fields
from tests.support import codata


def test_muon_gamma_matches_codata() -> None:
    assert MUON_GAMMA_OVER_2PI_MHZ_T == pytest.approx(codata.MUON_GAMMA_OVER_2PI_MHZ_T, rel=1e-8)
    assert MUON_GAMMA_RAD_S_T == pytest.approx(codata.MUON_GAMMA_RAD_S_T, rel=1e-8)


def test_muon_gamma_is_derived_from_scipy_magnetic_moment() -> None:
    moment_j_per_t = scipy.constants.physical_constants["muon mag. mom."][0]
    assert MUON_GAMMA_RAD_S_T == abs(2.0 * moment_j_per_t / scipy.constants.hbar)
    assert MUON_GAMMA_OVER_2PI_MHZ_T == pytest.approx(
        MUON_GAMMA_RAD_S_T / (2.0 * math.pi) / 1e6, rel=1e-15
    )
    assert MUON_GAMMA_RAD_US_T == pytest.approx(MUON_GAMMA_RAD_S_T * 1e-6, rel=1e-15)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("PLANCK_CONSTANT_J_S", scipy.constants.h),
        ("HBAR_J_S", scipy.constants.hbar),
        ("LIGHT_SPEED_M_S", scipy.constants.c),
        ("ELEMENTARY_CHARGE_C", scipy.constants.e),
        ("BOLTZMANN_J_K", scipy.constants.k),
        ("MU0_T_M_PER_A", scipy.constants.mu_0),
        ("VACUUM_PERMITTIVITY_F_PER_M", scipy.constants.epsilon_0),
        ("BOHR_MAGNETON_J_PER_T", scipy.constants.physical_constants["Bohr magneton"][0]),
        ("NUCLEAR_MAGNETON_J_PER_T", scipy.constants.physical_constants["nuclear magneton"][0]),
        ("ATOMIC_MASS_KG", scipy.constants.physical_constants["atomic mass constant"][0]),
        ("HARTREE_EV", scipy.constants.physical_constants["Hartree energy in eV"][0]),
        ("MUON_MASS_AMU", scipy.constants.physical_constants["muon mass in u"][0]),
        ("MUON_MASS_KG", scipy.constants.physical_constants["muon mass"][0]),
        ("METRES_PER_ANGSTROM", 1e-10),
    ],
)
def test_each_constant_is_taken_from_scipy(name: str, expected: float) -> None:
    assert getattr(constants, name) == expected


def test_derived_constants_keep_their_units() -> None:
    assert constants.BOHR_RADIUS_ANGSTROM == pytest.approx(0.529_177_210_544, rel=1e-11)
    assert constants.MU0_OVER_4PI_T_M_PER_A == pytest.approx(
        constants.MU0_T_M_PER_A / (4 * math.pi)
    )
    # mu_0 is measured since the 2019 SI, but stays within 1e-9 of 4 pi 1e-7.
    assert constants.MU0_OVER_4PI_T_M_PER_A == pytest.approx(1e-7, rel=1e-9)
    assert constants.MUON_MASS_AMU * constants.ATOMIC_MASS_KG == pytest.approx(
        constants.MUON_MASS_KG, rel=1e-9
    )


def test_constants_are_codata_2022() -> None:
    # Raises on a SciPy older than the 1.15 floor, which ships CODATA 2018 and would
    # render a different muon mass into CASTEP SPECIES_MASS.
    assert constants.MUON_MASS_AMU == codata.MUON_MASS_AMU
    assert constants.BOHR_MAGNETON_J_PER_T == codata.BOHR_MAGNETON_J_PER_T
    assert constants.MU0_T_M_PER_A == codata.MU0_T_M_PER_A


def test_modules_derive_their_factors_from_the_home() -> None:
    assert magnetic_fields.DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3 == (
        constants.MU0_OVER_4PI_T_M_PER_A
        * constants.BOHR_MAGNETON_J_PER_T
        / constants.METRES_PER_ANGSTROM**3
    )
    assert ts_parser.BOHR_TO_ANGSTROM == constants.BOHR_RADIUS_ANGSTROM
    assert ts_parser.HARTREE_TO_EV == constants.HARTREE_EV
