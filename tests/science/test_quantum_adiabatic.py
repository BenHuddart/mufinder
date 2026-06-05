"""Tests for single-particle adiabatic quantum approximations."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.science.quantum_adiabatic import (
    DirectionalPotential,
    effective_mass_from_relaxed_scan,
    solve_single_particle_ground_state,
)
from mufinder.science.quantum_phonons import (
    ATOMIC_MASS_KG,
    CM_INV_ZPE_TO_EV,
    ELEMENTARY_CHARGE_C,
    LIGHT_SPEED_M_S,
    MUON_MASS_AMU,
)


def test_single_particle_solver_matches_harmonic_oscillator_zpe() -> None:
    frequency_cm = 1000.0
    omega = 2.0 * np.pi * LIGHT_SPEED_M_S * 100.0 * frequency_cm
    mass_kg = MUON_MASS_AMU * ATOMIC_MASS_KG
    x_A = np.linspace(-2.0, 2.0, 161)
    x_m = x_A * 1e-10
    potential_eV = 0.5 * mass_kg * omega**2 * x_m**2 / ELEMENTARY_CHARGE_C

    result = solve_single_particle_ground_state(
        DirectionalPotential(
            label="L",
            displacement_A=tuple(float(value) for value in x_A),
            energy_eV=tuple(float(value) for value in potential_eV),
            effective_mass_amu=MUON_MASS_AMU,
        ),
        grid_points=401,
    )

    assert result.ground_energy_eV == pytest.approx(frequency_cm * CM_INV_ZPE_TO_EV, rel=0.03)
    assert result.width_A > 0.0
    assert sum(result.probability_density) > 0.0


def test_effective_mass_from_relaxed_scan_adds_host_motion() -> None:
    mass = effective_mass_from_relaxed_scan(
        muon_mass_amu=0.113,
        atom_masses_amu=(14.0, 16.0),
        host_displacements_per_muon_A=((0.1, 0.0, 0.0), (0.0, 0.2, 0.0)),
    )

    assert mass == pytest.approx(0.113 + 14.0 * 0.01 + 16.0 * 0.04)
