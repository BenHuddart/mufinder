"""Tests for the muon physical constants in :mod:`mufinder.domain.constants`."""

from __future__ import annotations

import math

import pytest
import scipy.constants

from mufinder.domain.constants import MUON_GAMMA_OVER_2PI_MHZ_T, MUON_GAMMA_RAD_S_T
from mufinder.science import magnetic_fields, nuclear_relaxation

# CODATA 2018: gamma_mu / 2 pi = 135.538 8094 MHz T^-1.
CODATA_MUON_GAMMA_OVER_2PI_MHZ_T = 135.538_8094


def test_muon_gamma_matches_codata() -> None:
    assert MUON_GAMMA_OVER_2PI_MHZ_T == pytest.approx(CODATA_MUON_GAMMA_OVER_2PI_MHZ_T, rel=1e-8)
    assert MUON_GAMMA_RAD_S_T == pytest.approx(
        2.0 * math.pi * CODATA_MUON_GAMMA_OVER_2PI_MHZ_T * 1e6, rel=1e-8
    )


def test_muon_gamma_is_derived_from_scipy_magnetic_moment() -> None:
    moment_j_per_t = scipy.constants.physical_constants["muon mag. mom."][0]
    assert MUON_GAMMA_RAD_S_T == abs(2.0 * moment_j_per_t / scipy.constants.hbar)
    assert MUON_GAMMA_OVER_2PI_MHZ_T == pytest.approx(
        MUON_GAMMA_RAD_S_T / (2.0 * math.pi) / 1e6, rel=1e-15
    )


def test_science_modules_use_the_domain_muon_gamma() -> None:
    assert nuclear_relaxation.GAMMA_MU_RAD_S_T == MUON_GAMMA_RAD_S_T
    assert nuclear_relaxation.GAMMA_MU_OVER_2PI_MHZ_T == MUON_GAMMA_OVER_2PI_MHZ_T
    assert magnetic_fields.MUON_GYROMAGNETIC_RATIO_MHZ_PER_T == MUON_GAMMA_OVER_2PI_MHZ_T
