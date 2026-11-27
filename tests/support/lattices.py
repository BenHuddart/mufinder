"""Lattice factories in standard settings and a brute-force shortest-translation oracle."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def hexagonal_lattice(a: float, c: float) -> list[list[float]]:
    """Lattice vectors of a hexagonal cell, ``a`` along x and ``b`` at 120° to it (Å)."""

    return [[a, 0.0, 0.0], [-a / 2, a * math.sqrt(3) / 2, 0.0], [0.0, 0.0, c]]


def monoclinic_lattice(a: float, b: float, c: float, beta_degrees: float) -> list[list[float]]:
    """Lattice vectors of a monoclinic cell with unique axis ``b`` (Å, degrees)."""

    beta = math.radians(beta_degrees)
    return [[a, 0.0, 0.0], [0.0, b, 0.0], [c * math.cos(beta), 0.0, c * math.sin(beta)]]


def triclinic_lattice(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> list[list[float]]:
    """Lattice vectors in the standard setting: ``a`` along x, ``b`` in the xy plane."""

    alpha, beta, gamma = (math.radians(angle) for angle in (alpha, beta, gamma))
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    return [
        [a, 0.0, 0.0],
        [b * math.cos(gamma), b * math.sin(gamma), 0.0],
        [cx, cy, math.sqrt(c * c - cx * cx - cy * cy)],
    ]


def brute_force_shortest_translation(
    lattice: Sequence[Sequence[float]] | np.ndarray, upper_bound: float = math.inf
) -> float:
    """The shortest non-zero ``n1 a1 + n2 a2 + n3 a3`` by an exhaustive search (Å).

    A translation ``t`` no longer than ``L``, the shortest lattice vector or the
    length of any translation passed as ``upper_bound``, has
    ``|n_i| = |t . c_i| <= L |c_i|``, ``c_i`` the columns of the inverse lattice, so
    every coefficient up to that reach is tried and none is missed.
    """

    lattice = np.asarray(lattice, dtype=float)
    longest_needed = min(float(np.min(np.linalg.norm(lattice, axis=1))), upper_bound)
    column_norms = np.linalg.norm(np.linalg.inv(lattice), axis=0)
    reaches = [math.ceil(longest_needed * (1 + 1e-9) * float(norm)) for norm in column_norms]
    if math.prod(2 * reach + 1 for reach in reaches) > 2_000_000:
        raise ValueError(f"A search to coefficients {reaches} is too large for a test.")
    steps = [np.arange(-reach, reach + 1, dtype=float) for reach in reaches]
    coefficients = np.stack(np.meshgrid(*steps), axis=-1).reshape(-1, 3)
    coefficients = coefficients[np.any(coefficients != 0.0, axis=1)]
    return float(np.min(np.linalg.norm(coefficients @ np.asarray(lattice, dtype=float), axis=1)))
