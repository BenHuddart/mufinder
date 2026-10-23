"""Conversion from MuFinder structures to ASE ``Atoms``.

This is the one home for turning a :class:`~mufinder.domain.models.Structure`
(or any structure-like object with ``lattice_vectors``, ``species`` and
``fractional_positions``) into an ASE ``Atoms`` object, for the muon-label
and element-symbol rules that conversion needs, and for the reduced formula
those rules give (:func:`reduced_formula`). Science modules that hand a
structure to ASE call :func:`structure_to_ase_atoms` with the options they need
instead of building ``Atoms`` themselves.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import math
import re

from ase import Atoms
import numpy as np

#: Species labels that mark the muon, compared case-insensitively after stripping.
MUON_LABELS = frozenset({"x", "h:mu", "mu", "muon"})

_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


class StructureConversionError(ValueError):
    """Raised when a structure cannot be converted to ASE ``Atoms``."""


def is_muon_label(symbol: object) -> bool:
    """Return whether ``symbol`` is one of the muon species labels in :data:`MUON_LABELS`."""

    return str(symbol).strip().casefold() in MUON_LABELS


def element_symbol(symbol: object, *, error: type[Exception] = StructureConversionError) -> str:
    """Return the chemical element a species label starts with (``"Fe1"`` -> ``"Fe"``).

    Raises ``error`` for a blank label or one that does not start with an element
    symbol, so each caller keeps its own exception type.
    """

    raw = str(symbol).strip()
    if not raw:
        raise error("Element symbol must not be empty.")
    match = _ELEMENT_RE.match(raw)
    if match is None:
        raise error(f"Could not resolve element symbol from '{symbol}'.")
    return match.group(1)


def reduced_formula(species: Iterable[object]) -> tuple[str, int]:
    """Return the reduced formula of the non-muon species and its count per cell.

    ``("Fe2O3", 6)`` for twelve Fe and eighteen O labels: elements keep their
    first-seen order and the count is the greatest common divisor of the element
    counts. Labels that do not start with an element symbol are skipped. A cell
    with no such atoms gives ``("", 0)``.
    """

    counts: Counter[str] = Counter()
    for label in species:
        if is_muon_label(label):
            continue
        try:
            counts[element_symbol(label)] += 1
        except StructureConversionError:
            continue
    if not counts:
        return "", 0
    divisor = math.gcd(*counts.values())
    formula = "".join(
        f"{symbol}{count // divisor if count // divisor > 1 else ''}"
        for symbol, count in counts.items()
    )
    return formula, divisor


def structure_to_ase_atoms(
    structure: object,
    *,
    skip_muon: bool = False,
    sanitize_symbols: bool = False,
    record_species_labels: bool = False,
    error: type[Exception] = StructureConversionError,
) -> Atoms:
    """Convert a MuFinder structure (or structure-like object) to periodic ASE ``Atoms``.

    ``structure`` needs ``lattice_vectors`` (a 3x3 matrix, rows are the lattice
    vectors in Angstrom), ``species`` and ``fractional_positions`` of matching
    length; anything else raises ``error``.

    Options:

    ``skip_muon``
        Drop every site whose label is a muon label (:func:`is_muon_label`).
    ``sanitize_symbols``
        Map muon labels to ``"H"`` and reduce every other label to its element
        symbol (:func:`element_symbol`), so labelled species such as ``"Fe1"``
        convert. Without it the labels go to ASE unchanged.
    ``record_species_labels``
        Store the original labels of the kept sites in
        ``atoms.info["mufinder_species_labels"]``.
    """

    lattice_vectors = getattr(structure, "lattice_vectors", None)
    species = getattr(structure, "species", None)
    fractional_positions = getattr(structure, "fractional_positions", None)
    if lattice_vectors is None or species is None or fractional_positions is None:
        raise error("Structure must define lattice_vectors, species, and fractional_positions.")
    if len(species) != len(fractional_positions):
        raise error("Structure species and fractional_positions must have matching lengths.")
    try:
        cell = np.asarray(lattice_vectors, dtype=float)
    except (TypeError, ValueError) as exc:
        raise error("Structure lattice vectors must form a 3x3 matrix.") from exc
    if cell.shape != (3, 3):
        raise error("Structure lattice vectors must form a 3x3 matrix.")

    labels: list[str] = []
    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    for raw_label, position in zip(species, fractional_positions):
        label = str(raw_label)
        muon = is_muon_label(label)
        if skip_muon and muon:
            continue
        if sanitize_symbols:
            symbols.append("H" if muon else element_symbol(label, error=error))
        else:
            symbols.append(label)
        labels.append(label)
        positions.append(tuple(float(value) for value in position))  # type: ignore[arg-type]

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=np.asarray(positions, dtype=float).reshape(-1, 3),
        cell=cell,
        pbc=[True, True, True],
    )
    if record_species_labels:
        atoms.info["mufinder_species_labels"] = labels
    return atoms


__all__ = [
    "MUON_LABELS",
    "StructureConversionError",
    "element_symbol",
    "is_muon_label",
    "structure_to_ase_atoms",
]
