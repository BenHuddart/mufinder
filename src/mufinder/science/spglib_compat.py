"""spglib access for the science modules.

This module is the one home for symmetry plumbing: it wraps spglib without its
deprecated warning paths, builds the ``(lattice, positions, numbers)`` cell from
a structure, caches each host's symmetry operations, and expands a position into
its symmetry orbit. Callers pass their symmetry and deduplication tolerances
explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from ase.data import atomic_numbers
import numpy as np
import spglib

from mufinder.science.lattice import deduplicate_periodic_positions, wrap_fractional


def configure_spglib_error_handling() -> None:
    """Disable spglib's deprecated warning-based error handling when available."""

    error_module = getattr(spglib, "error", None)
    if error_module is None or not hasattr(error_module, "OLD_ERROR_HANDLING"):
        return
    error_module.OLD_ERROR_HANDLING = False


def _dataset_helpers():
    globals_dict = getattr(spglib.get_symmetry_dataset, "__globals__", {})
    expand_cell = globals_dict.get("_expand_cell")
    low_level = globals_dict.get("_spglib")
    dataset_type = globals_dict.get("SpglibDataset", getattr(spglib, "SpglibDataset", None))
    if expand_cell is None or low_level is None or dataset_type is None:
        raise AttributeError("spglib low-level dataset helpers are unavailable")
    return expand_cell, low_level, dataset_type


def get_crystal_symmetry_dataset(
    cell,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
):
    """Return a non-magnetic spglib dataset without deprecated warning paths."""

    try:
        expand_cell, low_level, dataset_type = _dataset_helpers()
        lattice, positions, numbers, _ = expand_cell(cell)
        dataset = low_level.dataset(
            lattice,
            positions,
            numbers,
            0,
            float(symprec),
            float(angle_tolerance),
        )
    except Exception:  # noqa: BLE001 - spglib raises typed errors for malformed cells.
        return None
    return dataset_type(**dataset)


configure_spglib_error_handling()


def get_crystal_symmetry(
    cell,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
) -> dict[str, Any] | None:
    """Return non-magnetic symmetry operations without spglib's deprecated wrapper."""

    dataset = get_crystal_symmetry_dataset(
        cell,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    if dataset is None:
        return None

    if hasattr(dataset, "rotations"):
        rotations = dataset.rotations
        translations = dataset.translations
        equivalent_atoms = dataset.equivalent_atoms
    else:  # pragma: no cover - compatibility with mapping-style datasets.
        rotations = dataset["rotations"]
        translations = dataset["translations"]
        equivalent_atoms = dataset["equivalent_atoms"]

    return {
        "rotations": np.asarray(rotations, dtype=int),
        "translations": np.asarray(translations, dtype=float),
        "equivalent_atoms": np.asarray(equivalent_atoms, dtype=int),
    }


# Cell construction, cached symmetry operations and orbits.
#
# Every science module that asks spglib about a MuFinder structure builds its
# ``(lattice, positions, numbers)`` cell, its operation list and its orbits here,
# with the symmetry and deduplication tolerances passed explicitly by the caller.

SpglibCell = tuple[np.ndarray, np.ndarray, np.ndarray]
SymmetryOperations = tuple[np.ndarray, np.ndarray]


def species_to_atomic_numbers(species: Sequence[str]) -> list[int] | None:
    """Return the atomic number of each species label, or ``None`` if any is not an element.

    Labels are stripped and capitalised first, so ``"fe"`` and ``" Fe "`` both read as iron;
    labelled species such as ``"Fe1"`` or ``"H:mu"`` are not elements.
    """

    numbers: list[int] = []
    for symbol in species:
        number = atomic_numbers.get(str(symbol).strip().capitalize())
        if number is None:
            return None
        numbers.append(int(number))
    return numbers


def spglib_cell(
    structure: Any = None,
    *,
    lattice_vectors: Sequence[Sequence[float]] | None = None,
    species: Sequence[str] | None = None,
    fractional_positions: Sequence[Sequence[float]] | None = None,
) -> SpglibCell | None:
    """Return the spglib ``(lattice, positions, numbers)`` cell for a structure.

    Pass a structure-like object (``lattice_vectors``, ``species`` and
    ``fractional_positions`` attributes) or the three parts as keywords; a
    keyword overrides the matching attribute. Returns ``None`` when a species is
    not an element (:func:`species_to_atomic_numbers`) or the species and
    position counts differ.
    """

    lattice = (
        lattice_vectors
        if lattice_vectors is not None
        else getattr(structure, "lattice_vectors", None)
    )
    labels = species if species is not None else getattr(structure, "species", None)
    positions = (
        fractional_positions
        if fractional_positions is not None
        else getattr(structure, "fractional_positions", None)
    )
    if lattice is None or labels is None or positions is None:
        raise ValueError("spglib_cell needs lattice_vectors, species and fractional_positions.")
    numbers = species_to_atomic_numbers(labels)
    if numbers is None or len(numbers) != len(positions):
        return None
    return (
        np.asarray(lattice, dtype=float),
        np.asarray(positions, dtype=float).reshape(-1, 3),
        np.asarray(numbers, dtype=int),
    )


@lru_cache(maxsize=32)
def _cached_symmetry_operations(
    lattice: tuple[tuple[float, ...], ...],
    positions: tuple[tuple[float, ...], ...],
    numbers: tuple[int, ...],
    symprec: float,
) -> SymmetryOperations | None:
    """Run spglib once per cell; the arrays are read-only because callers share them."""

    symmetry = get_crystal_symmetry(
        (
            np.asarray(lattice, dtype=float),
            np.asarray(positions, dtype=float).reshape(-1, 3),
            np.asarray(numbers, dtype=int),
        ),
        symprec=symprec,
    )
    if symmetry is None:
        return None
    rotations = np.asarray(symmetry.get("rotations", []), dtype=float).reshape(-1, 3, 3)
    translations = np.asarray(symmetry.get("translations", []), dtype=float).reshape(-1, 3)
    rotations.setflags(write=False)
    translations.setflags(write=False)
    return rotations, translations


def symmetry_operations(cell: SpglibCell, *, symprec: float) -> SymmetryOperations | None:
    """Return the cell's stacked rotations ``(n, 3, 3)`` and translations ``(n, 3)``.

    Results are cached on the cell's values and ``symprec``, so repeated calls for
    one host run spglib once; the returned arrays are read-only. Returns ``None``
    when spglib finds no symmetry for the cell.
    """

    lattice, positions, numbers = cell
    return _cached_symmetry_operations(
        tuple(tuple(float(value) for value in row) for row in np.asarray(lattice, dtype=float)),
        tuple(
            tuple(float(value) for value in row)
            for row in np.asarray(positions, dtype=float).reshape(-1, 3)
        ),
        tuple(int(value) for value in np.asarray(numbers)),
        float(symprec),
    )


def clear_symmetry_operations_cache() -> None:
    """Forget every cached :func:`symmetry_operations` result."""

    _cached_symmetry_operations.cache_clear()


def operation_images(
    position: Sequence[float],
    rotations: Sequence[np.ndarray] | np.ndarray,
    translations: Sequence[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """Apply every symmetry operation to ``position`` at once (unwrapped images, one per row)."""

    count = min(len(rotations), len(translations))
    if count == 0:
        return np.empty((0, 3), dtype=float)
    rotation_stack = np.asarray(rotations[:count], dtype=float).reshape(count, 3, 3)
    translation_stack = np.asarray(translations[:count], dtype=float).reshape(count, 3)
    return rotation_stack @ np.asarray(position, dtype=float) + translation_stack


def symmetry_orbit(
    position: Sequence[float],
    operations: SymmetryOperations | tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None,
    *,
    lattice_vectors: Sequence[Sequence[float]] | None,
    tolerance: float,
) -> list[list[float]]:
    """Return the distinct wrapped images of ``position`` under ``operations``, in operation order.

    Images within ``tolerance`` of an earlier kept image are merged
    (:func:`mufinder.science.lattice.deduplicate_periodic_positions`): the
    tolerance is in Angstrom for a lattice and in fractional units for
    ``lattice_vectors=None``. Without operations the orbit is the wrapped
    position alone.
    """

    if operations is None:
        return [wrap_fractional(position)]
    rotations, translations = operations
    images = operation_images(position, rotations, translations)
    return deduplicate_periodic_positions(np.mod(images, 1.0), lattice_vectors, tolerance=tolerance)
