"""Map a relaxed muon supercell onto its pristine host for finite-cell corrections.

A muon relaxation displaces the host ions around the muon. Local properties such
as nuclear second moments and dipolar fields are corrected for those
displacements with a finite-cell difference: the property of the displaced
ions in the relaxed supercell minus the property of the same ions at their
pristine sites, added to the converged pristine lattice sum. This module builds
the pairing those corrections need. It validates that the relaxed structure is
a supercell of the host, that the correction cell is finite, that the selected
muon site is the relaxed muon, and that every displaced ion maps onto exactly
one pristine site. It also carries a pairing to a symmetry-equivalent muon site.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import is_muon_label, structure_to_ase_atoms
from mufinder.science.spglib_compat import operation_images, spglib_cell, symmetry_operations
from mufinder.science.supercell_matching import detect_supercell_matrix_from_lattices

SupercellMatrix = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]

DEFAULT_MUON_TOLERANCE_ANG = 0.5
"""Largest distance between the selected site and the relaxed muon, after lattice translations."""

_SYMMETRY_ATOM_TOLERANCE_ANG = 0.05


class MuonDistortionError(ValueError):
    """Raised when a relaxed muon structure cannot be paired with its pristine host."""


@dataclass(frozen=True, slots=True)
class MuonDistortion:
    """Displaced host ions around one muon site, paired with their pristine sites.

    Every array has one row per host ion of the relaxed supercell. Vectors run
    from the muon to the ion in Angstrom, in the pristine host frame: the
    pristine vectors start at the selected site, the distorted vectors at the
    relaxed muon, so a rigid drift of the relaxed cell cancels. ``atom_indices``
    and ``cell_offsets`` name each ion's pristine site as an atom of the
    pristine structure, indexed in its full species list (any muon marker
    included), plus an integer lattice translation of the host cell.
    """

    supercell_matrix: SupercellMatrix
    supercell_lattice: np.ndarray
    muon_fractional: tuple[float, float, float]
    muon_offset_ang: float
    symbols: tuple[str, ...]
    atom_indices: np.ndarray
    cell_offsets: np.ndarray
    pristine_vectors: np.ndarray
    distorted_vectors: np.ndarray
    n_atoms_unwrapped: int

    @property
    def displacements(self) -> np.ndarray:
        """Return each ion's displacement relative to the muon, in Angstrom."""

        return self.distorted_vectors - self.pristine_vectors


def map_muon_distortion(
    pristine_structure: Structure,
    distorted_structure: Structure,
    muon_position: Sequence[float],
    *,
    muon_tolerance_ang: float = DEFAULT_MUON_TOLERANCE_ANG,
    error: type[Exception] = MuonDistortionError,
) -> MuonDistortion:
    """Pair the host ions of a relaxed muon supercell with their pristine sites.

    ``muon_position`` is the selected site in fractional coordinates of the
    pristine host cell. ``distorted_structure`` is the relaxed supercell with
    exactly one muon marker. Every validation failure raises ``error``.
    """

    unit_symbols, unit_frac, unit_rows = _host_sites(pristine_structure, error=error)
    if not unit_symbols:
        raise error("Pristine structure does not contain any host atoms.")
    all_symbols = list(
        structure_to_ase_atoms(
            distorted_structure, sanitize_symbols=True, error=error
        ).get_chemical_symbols()
    )
    labels = [str(label) for label in distorted_structure.species]
    muon_indices = [index for index, label in enumerate(labels) if is_muon_label(label)]
    if not muon_indices:
        raise error("Distorted structure does not contain a muon marker.")
    if len(muon_indices) > 1:
        raise error("Distorted structure contains more than one muon marker.")
    muon_index = muon_indices[0]
    host_rows = [index for index in range(len(labels)) if index != muon_index]
    if not host_rows:
        raise error("Distorted structure does not contain any host atoms.")

    detection = detect_supercell_matrix_from_lattices(
        pristine_structure.lattice_vectors,
        distorted_structure.lattice_vectors,
    )
    # A detected supercell has a positive determinant, so the correction cell is finite.
    if not detection.is_supercell:
        raise error("Could not identify a matching pristine supercell for the distorted structure.")
    matrix = np.asarray(detection.matrix, dtype=float)
    unit_lattice = np.asarray(pristine_structure.lattice_vectors, dtype=float)
    supercell_lattice = matrix @ unit_lattice

    host_symbols = [all_symbols[index] for index in host_rows]
    _check_composition(unit_symbols, host_symbols, detection.determinant, error=error)

    # Express the relaxed supercell in the pristine host frame and recentre it on its muon.
    frac_super = np.asarray(distorted_structure.fractional_positions, dtype=float)
    muon_super = frac_super[muon_index]
    relative_super = frac_super[host_rows] - muon_super
    relative_super -= np.round(relative_super)
    muon_unit = muon_super @ matrix
    site = _canonical_fractional(muon_position)
    lattice_shift = np.round(muon_unit - site)
    muon_unit = muon_unit - lattice_shift
    muon_offset = float(np.linalg.norm((muon_unit - site) @ unit_lattice))
    if muon_offset > float(muon_tolerance_ang):
        raise error(
            f"The selected muon site is {muon_offset:.3f} Å from the muon in the distorted structure; "
            "choose the site that structure was relaxed from."
        )
    displaced_unit = muon_unit + relative_super @ matrix

    host_indices, cell_offsets = _nearest_pristine_sites(
        displaced_unit, host_symbols, unit_frac, unit_symbols, unit_lattice
    )
    _check_one_to_one(host_indices, cell_offsets, matrix, error=error)

    pristine_unit = unit_frac[host_indices] + cell_offsets
    pristine_vectors = (pristine_unit - site) @ unit_lattice
    distorted_vectors = (displaced_unit - muon_unit) @ unit_lattice
    pristine_relative_super = (pristine_unit - site) @ np.linalg.inv(matrix)
    unwrapped = int(np.count_nonzero(np.any(np.round(pristine_relative_super) != 0.0, axis=1)))

    return MuonDistortion(
        supercell_matrix=detection.matrix,
        supercell_lattice=supercell_lattice,
        muon_fractional=(float(site[0]), float(site[1]), float(site[2])),
        muon_offset_ang=muon_offset,
        symbols=tuple(host_symbols),
        atom_indices=unit_rows[host_indices],
        cell_offsets=cell_offsets,
        pristine_vectors=pristine_vectors,
        distorted_vectors=distorted_vectors,
        n_atoms_unwrapped=unwrapped,
    )


def equivalent_site_distortion(
    pristine_structure: Structure,
    distortion: MuonDistortion,
    muon_position: Sequence[float],
    *,
    tolerance: float = 1e-3,
    error: type[Exception] = MuonDistortionError,
) -> MuonDistortion:
    """Carry ``distortion`` to a symmetry-equivalent muon site of the pristine host.

    The host symmetry operation that maps the original site onto
    ``muon_position`` (within ``tolerance`` in fractional units) rotates the
    displacement pattern and relabels the pristine sites. Magnetic moments are
    not part of the host symmetry, so callers evaluate them at the relabelled
    sites.
    """

    target = _canonical_fractional(muon_position)
    source = np.asarray(distortion.muon_fractional, dtype=float)
    if _same_fractional(source, target, tolerance):
        return distortion
    cell = spglib_cell(pristine_structure)
    if cell is None:
        raise error("spglib could not determine symmetry operations: a species is not an element")
    operations = symmetry_operations(cell, symprec=1e-5)
    if operations is None:
        raise error("spglib could not determine symmetry operations")
    rotations, translations = operations
    images = operation_images(source, rotations, translations)
    for index, image in enumerate(images):
        if _same_fractional(image, target, tolerance):
            return _apply_operation(
                pristine_structure,
                distortion,
                np.asarray(rotations[index], dtype=float),
                np.asarray(translations[index], dtype=float) + np.round(target - image),
                target,
                error=error,
            )
    raise error(
        "The requested muon site is not symmetry-equivalent to the distorted structure's site."
    )


def _apply_operation(
    pristine_structure: Structure,
    distortion: MuonDistortion,
    rotation: np.ndarray,
    translation: np.ndarray,
    target: np.ndarray,
    *,
    error: type[Exception],
) -> MuonDistortion:
    lattice = np.asarray(pristine_structure.lattice_vectors, dtype=float)
    unit_symbols, unit_frac, unit_rows = _host_sites(pristine_structure, error=error)
    cartesian_rotation = np.linalg.inv(lattice) @ rotation.T @ lattice

    structure_frac = np.asarray(pristine_structure.fractional_positions, dtype=float)
    pristine_unit = structure_frac[distortion.atom_indices] + distortion.cell_offsets
    images = pristine_unit @ rotation.T + translation
    host_indices, cell_offsets = _nearest_pristine_sites(
        images, distortion.symbols, unit_frac, unit_symbols, lattice
    )
    residual = (images - unit_frac[host_indices] - cell_offsets) @ lattice
    if np.any(np.linalg.norm(residual, axis=1) > _SYMMETRY_ATOM_TOLERANCE_ANG):
        raise error("A host symmetry operation did not map the pristine host onto itself.")
    return replace(
        distortion,
        muon_fractional=(float(target[0]), float(target[1]), float(target[2])),
        atom_indices=unit_rows[host_indices],
        cell_offsets=cell_offsets,
        pristine_vectors=distortion.pristine_vectors @ cartesian_rotation,
        distorted_vectors=distortion.distorted_vectors @ cartesian_rotation,
    )


def _host_sites(
    structure: Structure, *, error: type[Exception]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return the element symbols, fractional positions and structure indices of a structure's host atoms."""

    atoms = structure_to_ase_atoms(structure, skip_muon=True, sanitize_symbols=True, error=error)
    rows = np.asarray(
        [index for index, label in enumerate(structure.species) if not is_muon_label(label)],
        dtype=int,
    )
    frac = (
        np.asarray(structure.fractional_positions, dtype=float).reshape(-1, 3)[rows].reshape(-1, 3)
    )
    return list(atoms.get_chemical_symbols()), frac, rows


def _check_composition(
    unit_symbols: Sequence[str],
    host_symbols: Sequence[str],
    determinant: int,
    *,
    error: type[Exception],
) -> None:
    foreign = sorted(set(host_symbols) - set(unit_symbols))
    if foreign:
        raise error(
            f"Pristine structure has no atoms of species '{foreign[0]}' required by the distorted structure."
        )
    for symbol in sorted(set(unit_symbols)):
        expected = determinant * sum(1 for value in unit_symbols if value == symbol)
        found = sum(1 for value in host_symbols if value == symbol)
        if found != expected:
            raise error(
                f"The distorted structure has {found} {symbol} atom(s); its {determinant}-cell pristine "
                f"supercell has {expected}."
            )


def _nearest_pristine_sites(
    positions_unit: np.ndarray,
    symbols: Sequence[str],
    unit_frac: np.ndarray,
    unit_symbols: Sequence[str],
    lattice: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest same-species host site (atom index, lattice offset) for every position."""

    atom_indices = np.zeros(len(symbols), dtype=int)
    cell_offsets = np.zeros((len(symbols), 3), dtype=float)
    unit_symbol_array = np.asarray(unit_symbols, dtype=object)
    for row, (position, symbol) in enumerate(zip(positions_unit, symbols, strict=True)):
        candidates = np.flatnonzero(unit_symbol_array == symbol)
        deltas = position[np.newaxis, :] - unit_frac[candidates]
        offsets = np.round(deltas)
        distances = np.linalg.norm((deltas - offsets) @ lattice, axis=1)
        best = int(np.argmin(distances))
        atom_indices[row] = int(candidates[best])
        cell_offsets[row] = offsets[best]
    return atom_indices, cell_offsets


def _check_one_to_one(
    atom_indices: np.ndarray,
    cell_offsets: np.ndarray,
    matrix: np.ndarray,
    *,
    error: type[Exception],
) -> None:
    # Two ions pair with the same pristine site when their offsets differ by a supercell translation.
    folded = np.mod(cell_offsets @ np.linalg.inv(matrix), 1.0)
    folded[np.isclose(folded, 1.0, atol=1e-6)] = 0.0
    keys = {
        (int(index), *(round(float(value), 6) for value in offset))
        for index, offset in zip(atom_indices, folded, strict=True)
    }
    duplicates = len(atom_indices) - len(keys)
    if duplicates:
        raise error(
            f"The distorted-to-pristine atom mapping is not one-to-one: {duplicates} displaced atom(s) share a "
            "pristine site. Check that the distorted structure was relaxed from this host."
        )


def _same_fractional(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return bool(np.all(np.abs(delta - np.round(delta)) <= float(tolerance)))


def _canonical_fractional(position: Sequence[float]) -> np.ndarray:
    arr = np.asarray(position, dtype=float).reshape(3)
    out = np.mod(arr, 1.0)
    out[np.isclose(out, 1.0, atol=1e-10)] = 0.0
    return out
