"""UEP optimisation results for tests, built without a potential."""

from __future__ import annotations

import numpy as np

from mufinder.science.uep import StationaryPointClassification, UepOptimizationResult

_CUBE_LATTICE = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))


def fake_optimization(
    fractional: tuple[float, float, float],
    *,
    total: float | None = None,
    classical: float | None = None,
    label: str = "minimum",
    success: bool = True,
    lattice_vectors: tuple[tuple[float, float, float], ...] = _CUBE_LATTICE,
) -> UepOptimizationResult:
    """An optimization endpoint; the classical energy defaults to the total energy."""

    lattice = np.asarray(lattice_vectors, dtype=float)
    cartesian = tuple(
        float(value) for value in (np.asarray(fractional, dtype=float) @ lattice).tolist()
    )
    return UepOptimizationResult(
        start_position_ang=cartesian,
        optimized_position_ang=cartesian,
        optimized_fractional_position=fractional,
        classical_energy_ev=float(total if classical is None else classical),
        zero_point_energy_ev=0.0 if total is not None else None,
        total_energy_ev=None if total is None else float(total),
        success=success,
        message="",
        iterations=1,
        classification=StationaryPointClassification(
            label=label,
            eigenvalues=(1.0, 1.0, 1.0),
            positive_count=3,
            negative_count=0,
            near_zero_count=0,
        ),
    )
