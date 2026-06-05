"""Coverage tests for adiabatic helper error branches.

These exercise the validation/error paths in
:mod:`mufinder.science.quantum_adiabatic` that the happy-path tests in
``test_quantum_adiabatic.py`` do not reach. All inputs are deterministic; no
network, GUI, files, sleeps, or randomness are used.
"""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.science.quantum_adiabatic import (
    AdiabaticApproximationError,
    DirectionalPotential,
    effective_mass_from_relaxed_scan,
    solve_single_particle_ground_state,
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


# --- solve_single_particle_ground_state error branches --------------------


def test_too_few_grid_points_raises() -> None:
    with pytest.raises(AdiabaticApproximationError, match="five grid points"):
        solve_single_particle_ground_state(_potential(), grid_points=4)


def test_mismatched_array_lengths_raises() -> None:
    with pytest.raises(AdiabaticApproximationError, match="one-dimensional and equal length"):
        solve_single_particle_ground_state(
            _potential(displacement_A=(-1.0, 0.0, 1.0), energy_eV=(0.0, 1.0)),
        )


def test_non_one_dimensional_arrays_raise() -> None:
    # A 2D displacement array trips the ``ndim != 1`` guard before the
    # length check.
    bad = DirectionalPotential(
        label="L",
        displacement_A=((-1.0, 0.0), (1.0, 2.0)),  # type: ignore[arg-type]
        energy_eV=((0.0, 1.0), (2.0, 3.0)),  # type: ignore[arg-type]
        effective_mass_amu=0.113,
    )
    with pytest.raises(AdiabaticApproximationError, match="one-dimensional and equal length"):
        solve_single_particle_ground_state(bad)


def test_too_few_scan_points_raises() -> None:
    with pytest.raises(AdiabaticApproximationError, match="three line-scan points"):
        solve_single_particle_ground_state(
            _potential(displacement_A=(-1.0, 1.0), energy_eV=(1.0, 1.0)),
        )


def test_non_positive_effective_mass_raises() -> None:
    with pytest.raises(AdiabaticApproximationError, match="Effective mass must be positive"):
        solve_single_particle_ground_state(_potential(effective_mass_amu=0.0))


def test_duplicate_displacement_points_raise() -> None:
    with pytest.raises(AdiabaticApproximationError, match="must be unique"):
        solve_single_particle_ground_state(
            _potential(displacement_A=(0.0, 0.0, 1.0), energy_eV=(0.0, 1.0, 2.0)),
        )


def test_unsorted_input_is_reordered_and_solves() -> None:
    # Sorting happens internally; an unsorted but valid scan must succeed and
    # produce a grid spanning the min/max displacement.
    result = solve_single_particle_ground_state(
        _potential(displacement_A=(1.0, -1.0, 0.0), energy_eV=(1.0, 1.0, 0.0)),
        grid_points=51,
    )
    assert result.grid_A[0] == pytest.approx(-1.0)
    assert result.grid_A[-1] == pytest.approx(1.0)
    # Potential is shifted so its minimum is zero.
    assert min(result.potential_eV) == pytest.approx(0.0)
    # Probability density is normalised to a positive integral.
    assert np.trapezoid(result.probability_density, result.grid_A) == pytest.approx(1.0, rel=1e-6)
    assert result.width_A > 0.0


# --- effective_mass_from_relaxed_scan error branches -----------------------


def test_non_positive_muon_mass_raises() -> None:
    with pytest.raises(AdiabaticApproximationError, match="Muon mass must be positive"):
        effective_mass_from_relaxed_scan(
            muon_mass_amu=0.0,
            atom_masses_amu=(14.0,),
            host_displacements_per_muon_A=((0.1, 0.0, 0.0),),
        )


def test_mismatched_mass_and_derivative_lengths_raise() -> None:
    with pytest.raises(AdiabaticApproximationError, match="equal length"):
        effective_mass_from_relaxed_scan(
            muon_mass_amu=0.113,
            atom_masses_amu=(14.0, 16.0),
            host_displacements_per_muon_A=((0.1, 0.0, 0.0),),
        )


def test_non_positive_host_mass_raises() -> None:
    with pytest.raises(AdiabaticApproximationError, match="Host atom masses must be positive"):
        effective_mass_from_relaxed_scan(
            muon_mass_amu=0.113,
            atom_masses_amu=(0.0,),
            host_displacements_per_muon_A=((0.1, 0.0, 0.0),),
        )


def test_wrong_derivative_component_count_raises() -> None:
    with pytest.raises(AdiabaticApproximationError, match="three components"):
        effective_mass_from_relaxed_scan(
            muon_mass_amu=0.113,
            atom_masses_amu=(14.0,),
            host_displacements_per_muon_A=((0.1, 0.0),),  # type: ignore[arg-type]
        )


def test_empty_inputs_return_bare_muon_mass() -> None:
    # No host atoms: the loop body never runs and the bare muon mass returns.
    mass = effective_mass_from_relaxed_scan(
        muon_mass_amu=0.113,
        atom_masses_amu=(),
        host_displacements_per_muon_A=(),
    )
    assert mass == pytest.approx(0.113)
