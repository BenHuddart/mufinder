"""Tests for single-particle adiabatic quantum approximations."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.science.quantum_adiabatic import (
    AdiabaticApproximationError,
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


def _potential(
    *,
    displacement_A: tuple[float, ...] = (-1.0, 0.0, 1.0),
    energy_eV: tuple[float, ...] = (1.0, 0.0, 1.0),
    effective_mass_amu: float = 0.113,
) -> DirectionalPotential:
    return DirectionalPotential(
        label="L",
        displacement_A=displacement_A,
        energy_eV=energy_eV,
        effective_mass_amu=effective_mass_amu,
    )


# solve_single_particle_ground_state


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


def test_unsorted_input_is_reordered_and_solves() -> None:
    result = solve_single_particle_ground_state(
        _potential(displacement_A=(1.0, -1.0, 0.0), energy_eV=(1.0, 1.0, 0.0)),
        grid_points=51,
    )

    assert result.grid_A[0] == pytest.approx(-1.0)
    assert result.grid_A[-1] == pytest.approx(1.0)
    # Potential is shifted so its minimum is zero.
    assert min(result.potential_eV) == pytest.approx(0.0)
    # Probability density is normalised.
    assert np.trapezoid(result.probability_density, result.grid_A) == pytest.approx(1.0, rel=1e-6)
    assert result.width_A > 0.0


def test_too_few_grid_points_raises() -> None:
    with pytest.raises(AdiabaticApproximationError, match="five grid points"):
        solve_single_particle_ground_state(_potential(), grid_points=4)


@pytest.mark.parametrize(
    ("potential", "message"),
    [
        (_potential(displacement_A=(-1.0, 0.0, 1.0), energy_eV=(0.0, 1.0)), "one-dimensional and equal length"),
        (
            # A 2D array trips the ndim guard before the length check.
            DirectionalPotential(
                label="L",
                displacement_A=((-1.0, 0.0), (1.0, 2.0)),  # type: ignore[arg-type]
                energy_eV=((0.0, 1.0), (2.0, 3.0)),  # type: ignore[arg-type]
                effective_mass_amu=0.113,
            ),
            "one-dimensional and equal length",
        ),
        (_potential(displacement_A=(-1.0, 1.0), energy_eV=(1.0, 1.0)), "three line-scan points"),
        (_potential(effective_mass_amu=0.0), "Effective mass must be positive"),
        (_potential(displacement_A=(0.0, 0.0, 1.0), energy_eV=(0.0, 1.0, 2.0)), "must be unique"),
    ],
    ids=["mismatched-lengths", "two-dimensional", "too-few-points", "zero-mass", "duplicate-points"],
)
def test_single_particle_solver_rejects_invalid_scan(potential: DirectionalPotential, message: str) -> None:
    with pytest.raises(AdiabaticApproximationError, match=message):
        solve_single_particle_ground_state(potential)


# effective_mass_from_relaxed_scan


def test_effective_mass_from_relaxed_scan_adds_host_motion() -> None:
    mass = effective_mass_from_relaxed_scan(
        muon_mass_amu=0.113,
        atom_masses_amu=(14.0, 16.0),
        host_displacements_per_muon_A=((0.1, 0.0, 0.0), (0.0, 0.2, 0.0)),
    )

    assert mass == pytest.approx(0.113 + 14.0 * 0.01 + 16.0 * 0.04)


def test_empty_inputs_return_bare_muon_mass() -> None:
    mass = effective_mass_from_relaxed_scan(
        muon_mass_amu=0.113,
        atom_masses_amu=(),
        host_displacements_per_muon_A=(),
    )

    assert mass == pytest.approx(0.113)


@pytest.mark.parametrize(
    ("muon_mass", "atom_masses", "displacements", "message"),
    [
        (0.0, (14.0,), ((0.1, 0.0, 0.0),), "Muon mass must be positive"),
        (0.113, (14.0, 16.0), ((0.1, 0.0, 0.0),), "equal length"),
        (0.113, (0.0,), ((0.1, 0.0, 0.0),), "Host atom masses must be positive"),
        (0.113, (14.0,), ((0.1, 0.0),), "three components"),
    ],
    ids=["zero-muon-mass", "mismatched-lengths", "zero-host-mass", "two-components"],
)
def test_effective_mass_from_relaxed_scan_rejects_invalid_input(
    muon_mass: float,
    atom_masses: tuple[float, ...],
    displacements: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    with pytest.raises(AdiabaticApproximationError, match=message):
        effective_mass_from_relaxed_scan(
            muon_mass_amu=muon_mass,
            atom_masses_amu=atom_masses,
            host_displacements_per_muon_A=displacements,  # type: ignore[arg-type]
        )
