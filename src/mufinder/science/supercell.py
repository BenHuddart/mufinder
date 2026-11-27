"""Supercell expansion helpers with magnetic spin propagation."""

from __future__ import annotations

from collections.abc import Sequence
from math import cos, pi

from ase.build import make_supercell
import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import structure_to_ase_atoms


def normalize_supercell_matrix(matrix: Sequence[Sequence[int]]) -> np.ndarray:
    """Validate and normalize a 3x3 integer supercell matrix."""

    if len(matrix) != 3:
        raise ValueError("Supercell matrix must have exactly three rows.")
    rows: list[list[int]] = []
    for row in matrix:
        if len(row) != 3:
            raise ValueError("Supercell matrix rows must each contain exactly three integers.")
        rows.append([int(row[0]), int(row[1]), int(row[2])])

    parsed = np.asarray(rows, dtype=int)
    determinant = int(round(float(np.linalg.det(parsed))))
    if determinant == 0:
        raise ValueError("Supercell matrix must be invertible (determinant != 0).")
    if determinant < 0:
        raise ValueError("Supercell matrix determinant must be positive.")
    return parsed


def build_supercell_structure(
    structure: Structure,
    supercell_matrix: Sequence[Sequence[int]],
    *,
    atomic_spins: Sequence[float] | None = None,
    propagation_vector: Sequence[float] | None = None,
) -> tuple[Structure, list[float]]:
    """Expand a structure and propagate per-atom spins across the supercell."""

    matrix = normalize_supercell_matrix(supercell_matrix)
    base_atoms = structure_to_ase_atoms(structure)
    expanded_atoms = make_supercell(base_atoms, matrix)

    expanded_frac = np.asarray(expanded_atoms.get_scaled_positions(wrap=True), dtype=float)
    expanded_species = list(expanded_atoms.get_chemical_symbols())

    base_frac = np.mod(np.asarray(structure.fractional_positions, dtype=float), 1.0)
    base_species = np.asarray(structure.species, dtype=object)
    base_spins = _normalized_spins(atomic_spins, len(structure.species))
    k_vector = _normalized_k_vector(propagation_vector)

    expanded_spins: list[float] = []
    for species, frac in zip(expanded_species, expanded_frac):
        base_index, translation = _map_expanded_atom_to_base(
            species=species,
            frac_super=frac,
            matrix=matrix,
            base_species=base_species,
            base_frac=base_frac,
        )
        base_spin = base_spins[base_index]
        phase = 2.0 * pi * float(np.dot(k_vector, translation))
        expanded_spins.append(float(base_spin * cos(phase)))

    properties = dict(structure.properties)
    properties["atomic_spins"] = [float(value) for value in expanded_spins]
    properties["propagation_vector"] = [float(value) for value in k_vector.tolist()]
    properties["supercell_matrix"] = [[int(value) for value in row] for row in matrix.tolist()]
    properties["pbc"] = [bool(value) for value in expanded_atoms.pbc.tolist()]

    expanded_structure = Structure(
        label=f"{structure.label or 'structure'}_supercell",
        origin=structure.origin,
        lattice_vectors=expanded_atoms.cell.array.tolist(),
        species=expanded_species,
        fractional_positions=expanded_frac.tolist(),
        source_file=structure.source_file,
        parent_structure_id=structure.id,
        properties=properties,
    )
    return expanded_structure, expanded_spins


#: How far a component of ``k`` may sit from a value that fits, in reciprocal-lattice
#: units: a third typed as ``0.333`` fits a tripled cell, and a sixfold one.
PROPAGATION_FIT_TOLERANCE = 1e-3


def supercell_fits_propagation_vector(
    supercell_matrix: Sequence[Sequence[int]],
    propagation_vector: Sequence[float] | None,
    *,
    tolerance: float = PROPAGATION_FIT_TOLERANCE,
) -> bool:
    """Return whether the spins of the supercell repeat across its boundary.

    :func:`build_supercell_structure` multiplies each copied spin by ``cos(2π k·T)``,
    with ``T`` the lattice translation of the copy in host cell units. The supercell's
    vectors are the rows ``A`` of ``supercell_matrix`` in the same units, so the
    modulated spins match their periodic images only when ``k·A`` is a whole number
    for each of the three rows. ``k`` fits when one vector ``k'`` within ``tolerance``
    of it, component by component, gives the nearest whole numbers for all three rows
    at once, so a rounded ``k`` stays a fit in a longer cell and the answer does not
    depend on which basis of the supercell the matrix writes. A zero or missing ``k``
    always fits, and so does one within ``tolerance`` of zero.
    """

    k_vector = _normalized_k_vector(propagation_vector)
    if not np.any(k_vector):
        return True
    rows = np.asarray(supercell_matrix, dtype=float).reshape(3, 3)
    whole = np.round(rows @ k_vector)
    try:
        fitting = np.linalg.solve(rows, whole)
    except np.linalg.LinAlgError:
        return False
    return bool(np.all(np.abs(fitting - k_vector) <= tolerance + 1e-12))


def _normalized_spins(spins: Sequence[float] | None, count: int) -> np.ndarray:
    parsed = np.zeros(count, dtype=float)
    if spins is None:
        return parsed

    for index, value in enumerate(spins):
        if index >= count:
            break
        parsed[index] = float(value)
    return parsed


def _normalized_k_vector(propagation_vector: Sequence[float] | None) -> np.ndarray:
    parsed = np.zeros(3, dtype=float)
    if propagation_vector is None:
        return parsed

    for index, value in enumerate(propagation_vector):
        if index >= 3:
            break
        parsed[index] = float(value)
    return parsed


def _map_expanded_atom_to_base(
    *,
    species: str,
    frac_super: np.ndarray,
    matrix: np.ndarray,
    base_species: np.ndarray,
    base_frac: np.ndarray,
) -> tuple[int, np.ndarray]:
    # Convert supercell fractional coordinate back to host-cell coordinates.
    frac_in_base = np.asarray(frac_super, dtype=float) @ matrix

    candidate_indices = np.where(base_species == species)[0]
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(base_species))

    best_index = int(candidate_indices[0])
    best_translation = np.zeros(3, dtype=float)
    best_error = float("inf")

    for index in candidate_indices:
        diff = frac_in_base - base_frac[int(index)]
        translation = np.rint(diff)
        residual = diff - translation
        error = float(np.linalg.norm(residual))
        if error < best_error:
            best_error = error
            best_index = int(index)
            best_translation = translation

    return best_index, best_translation
