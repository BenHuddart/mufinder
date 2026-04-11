"""Supercell detection and normalization helpers for structure comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.supercell import build_supercell_structure


@dataclass(frozen=True, slots=True)
class SupercellDetection:
    """Result of inferring a supercell transform between two lattices."""

    is_supercell: bool
    matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    determinant: int
    residual: float


@dataclass(frozen=True, slots=True)
class NormalizedStructurePair:
    """Comparison-ready structures with supercell metadata."""

    experimental: Structure
    dft: Structure
    supercell_matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    is_supercell: bool
    warnings: tuple[str, ...] = ()


_IDENTITY_MATRIX = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _matrix_from_array(values: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    return (
        (int(values[0, 0]), int(values[0, 1]), int(values[0, 2])),
        (int(values[1, 0]), int(values[1, 1]), int(values[1, 2])),
        (int(values[2, 0]), int(values[2, 1]), int(values[2, 2])),
    )


def detect_supercell_matrix(
    experimental: Structure,
    dft: Structure,
    *,
    tolerance: float = 0.2,
) -> SupercellDetection:
    """Infer an integer supercell matrix M such that dft_lattice ~= M * exp_lattice."""

    experimental_lattice = np.asarray(experimental.lattice_vectors, dtype=float)
    dft_lattice = np.asarray(dft.lattice_vectors, dtype=float)

    try:
        candidate = dft_lattice @ np.linalg.inv(experimental_lattice)
    except np.linalg.LinAlgError:
        return SupercellDetection(
            is_supercell=False,
            matrix=_IDENTITY_MATRIX,
            determinant=1,
            residual=float("inf"),
        )

    rounded = np.rint(candidate).astype(int)
    residual = float(np.linalg.norm(candidate - rounded))
    determinant = int(round(float(np.linalg.det(rounded))))
    is_supercell = residual <= tolerance and determinant > 0

    return SupercellDetection(
        is_supercell=is_supercell,
        matrix=_matrix_from_array(rounded),
        determinant=determinant,
        residual=residual,
    )


def tile_to_supercell(
    experimental: Structure,
    matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]],
) -> Structure:
    """Expand experimental structure into the provided supercell matrix."""

    expanded, _spins = build_supercell_structure(
        experimental,
        [[int(value) for value in row] for row in matrix],
        atomic_spins=None,
        propagation_vector=None,
    )
    expanded.label = f"{experimental.label} tiled"
    expanded.parent_structure_id = experimental.id
    return expanded


def normalize_for_comparison(
    experimental: Structure,
    dft: Structure,
    *,
    tolerance: float = 0.2,
) -> NormalizedStructurePair:
    """Normalize structures into a common periodic frame for fair comparison."""

    detection = detect_supercell_matrix(experimental, dft, tolerance=tolerance)

    if detection.is_supercell and detection.determinant > 1:
        tiled = tile_to_supercell(experimental, detection.matrix)
        return NormalizedStructurePair(
            experimental=tiled,
            dft=dft,
            supercell_matrix=detection.matrix,
            is_supercell=True,
            warnings=(),
        )

    if detection.is_supercell:
        return NormalizedStructurePair(
            experimental=experimental,
            dft=dft,
            supercell_matrix=detection.matrix,
            is_supercell=False,
            warnings=(),
        )

    return NormalizedStructurePair(
        experimental=experimental,
        dft=dft,
        supercell_matrix=_IDENTITY_MATRIX,
        is_supercell=False,
        warnings=(
            "Could not identify an exact supercell relation; comparison used the original lattices.",
        ),
    )
