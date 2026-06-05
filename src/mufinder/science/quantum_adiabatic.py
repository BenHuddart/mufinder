"""Single-particle adiabatic quantum approximations for muon line scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mufinder.science.quantum_phonons import (
    ATOMIC_MASS_KG,
    ELEMENTARY_CHARGE_C,
    HBAR_J_S,
)


class AdiabaticApproximationError(ValueError):
    """Raised when a single-particle approximation input is invalid."""


@dataclass(frozen=True, slots=True)
class DirectionalPotential:
    """One 1D effective potential along a muon displacement direction."""

    label: str
    displacement_A: tuple[float, ...]
    energy_eV: tuple[float, ...]
    effective_mass_amu: float
    approximation: str = "weakly_bound"


@dataclass(frozen=True, slots=True)
class SingleParticleResult:
    """Ground-state result for one 1D effective potential."""

    label: str
    grid_A: tuple[float, ...]
    potential_eV: tuple[float, ...]
    ground_energy_eV: float
    wavefunction: tuple[float, ...]
    probability_density: tuple[float, ...]
    width_A: float


def solve_single_particle_ground_state(
    potential: DirectionalPotential,
    *,
    grid_points: int = 401,
) -> SingleParticleResult:
    """Solve the 1D finite-difference Schrodinger equation for the ground state."""

    if grid_points < 5:
        raise AdiabaticApproximationError("At least five grid points are required.")
    x_points = np.asarray(potential.displacement_A, dtype=float)
    energy_points = np.asarray(potential.energy_eV, dtype=float)
    if x_points.ndim != 1 or energy_points.ndim != 1 or len(x_points) != len(energy_points):
        raise AdiabaticApproximationError("Displacement and energy arrays must be one-dimensional and equal length.")
    if len(x_points) < 3:
        raise AdiabaticApproximationError("At least three line-scan points are required.")
    if potential.effective_mass_amu <= 0.0:
        raise AdiabaticApproximationError("Effective mass must be positive.")

    order = np.argsort(x_points)
    x_points = x_points[order]
    energy_points = energy_points[order]
    if np.any(np.diff(x_points) <= 0.0):
        raise AdiabaticApproximationError("Displacement points must be unique.")

    grid_A = np.linspace(float(x_points[0]), float(x_points[-1]), grid_points)
    potential_eV = np.interp(grid_A, x_points, energy_points)
    potential_eV -= float(np.min(potential_eV))

    dx_m = float((grid_A[1] - grid_A[0]) * 1e-10)
    mass_kg = potential.effective_mass_amu * ATOMIC_MASS_KG
    kinetic_eV = HBAR_J_S**2 / (2.0 * mass_kg * dx_m**2) / ELEMENTARY_CHARGE_C
    diagonal = 2.0 * kinetic_eV + potential_eV
    off_diagonal = np.full(grid_points - 1, -kinetic_eV)
    hamiltonian = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    psi = eigenvectors[:, 0]
    norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, grid_A))
    if norm <= 0.0:
        raise AdiabaticApproximationError("Ground-state wavefunction normalization failed.")
    psi = psi / norm
    density = np.abs(psi) ** 2
    mean = float(np.trapezoid(grid_A * density, grid_A))
    width = float(np.sqrt(np.trapezoid(((grid_A - mean) ** 2) * density, grid_A)))

    return SingleParticleResult(
        label=potential.label,
        grid_A=tuple(float(value) for value in grid_A),
        potential_eV=tuple(float(value) for value in potential_eV),
        ground_energy_eV=float(eigenvalues[0]),
        wavefunction=tuple(float(value) for value in psi),
        probability_density=tuple(float(value) for value in density),
        width_A=width,
    )


def effective_mass_from_relaxed_scan(
    *,
    muon_mass_amu: float,
    atom_masses_amu: tuple[float, ...],
    host_displacements_per_muon_A: tuple[tuple[float, float, float], ...],
) -> float:
    """Return ``m_mu + sum_i m_i |dR_i/dx_mu|^2`` for a strongly-bound scan."""

    if muon_mass_amu <= 0.0:
        raise AdiabaticApproximationError("Muon mass must be positive.")
    if len(atom_masses_amu) != len(host_displacements_per_muon_A):
        raise AdiabaticApproximationError("Mass and displacement-derivative arrays must have equal length.")
    total = float(muon_mass_amu)
    for mass, derivative in zip(atom_masses_amu, host_displacements_per_muon_A, strict=True):
        if mass <= 0.0:
            raise AdiabaticApproximationError("Host atom masses must be positive.")
        vector = np.asarray(derivative, dtype=float)
        if vector.shape != (3,):
            raise AdiabaticApproximationError("Each host displacement derivative must have three components.")
        total += float(mass) * float(np.dot(vector, vector))
    return total
