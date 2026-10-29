"""Readers for the magnetic metadata persisted in ``Structure.properties``.

A host structure can carry its magnetic order in three places: scalar per-atom
spins (``atomic_spins`` / ``initial_magnetic_moments``), an MCIF-derived
``magnetic_structure`` mapping whose ``moments`` entries hold a ``vector``, and
a ``propagation_vector`` (top level or inside ``magnetic_structure``). The
helpers here are the one place that interprets those persisted keys, so the
workflow bridge, the project service and the science modules read them the
same way. They are pure functions of the stored data and import nothing outside
the domain layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mufinder.domain.models import Structure

#: Magnitude above which a spin counts as nonzero.
SPIN_TOLERANCE = 1e-12


def has_nonzero_spins(spins: Sequence[float] | None, *, tolerance: float = SPIN_TOLERANCE) -> bool:
    """Return whether ``spins`` holds any moment whose magnitude exceeds ``tolerance``.

    ``None`` (no spins supplied) and an all-zero sequence both count as "no spins".
    """

    if spins is None:
        return False
    return any(abs(float(value)) > tolerance for value in spins)


def atomic_spins_from_magnetic_moments(
    magnetic_structure: Mapping[str, Any],
    *,
    atom_count: int,
    honour_atom_index: bool = True,
) -> list[float] | None:
    """Return per-atom spins taken from the z component of each moment vector.

    The result has ``atom_count`` entries, zero where no moment applies. A moment
    without a numeric three-component ``vector`` is skipped.

    With ``honour_atom_index`` a moment that carries an ``atom_index`` is placed
    at that index (an unparsable or out-of-range index skips the moment) and a
    moment without one is placed at its list position. Without it every moment is
    placed at its list position and moments beyond ``atom_count`` are ignored.

    Returns ``None`` unless at least one moment supplied a numeric spin, so a
    caller can tell "no usable moments" apart from "moments that are all zero".
    """

    moments = magnetic_structure.get("moments")
    if not isinstance(moments, list) or not moments:
        return None

    count = max(0, int(atom_count))
    spins = [0.0] * count
    saw_numeric = False
    for index, moment in enumerate(moments):
        if not isinstance(moment, dict):
            continue
        atom_index = index
        if honour_atom_index and "atom_index" in moment:
            try:
                atom_index = int(moment["atom_index"])
            except (TypeError, ValueError):
                continue
        if atom_index < 0 or atom_index >= count:
            continue
        vector = moment.get("vector")
        if not isinstance(vector, list) or len(vector) < 3:
            continue
        try:
            spins[atom_index] = float(vector[2])
        except (TypeError, ValueError):
            continue
        saw_numeric = True
    return spins if saw_numeric else None


def atomic_spins_from_structure(
    structure: Structure, *, atom_count: int | None = None
) -> list[float] | None:
    """Return spins derived from ``structure.properties["magnetic_structure"]``.

    ``atom_count`` defaults to the number of species. Moments are placed in list
    order (``atom_index`` is not consulted), which is how a stored structure's
    fallback spins have always been read. Returns ``None`` when the structure has
    no magnetic-structure mapping or no moment supplied a numeric spin.
    """

    magnetic = structure.properties.get("magnetic_structure")
    if not isinstance(magnetic, dict):
        return None
    count = len(structure.species) if atom_count is None else int(atom_count)
    return atomic_spins_from_magnetic_moments(magnetic, atom_count=count, honour_atom_index=False)


def magnetic_moments_are_collinear(
    magnetic_structure: Mapping[str, Any],
    *,
    tolerance: float = 1e-6,
    angle_tolerance: float = 1e-3,
) -> bool:
    """Return whether every nonzero moment ``vector`` lies along one axis.

    Parallel and antiparallel moments both count as collinear: two unit vectors
    are along one axis when their cross product is at most ``angle_tolerance``
    (radians, about 0.06 degrees by default, so moments written to three or four
    decimals still count). Moments without a numeric three-component ``vector``,
    and moments shorter than ``tolerance``, are ignored, so a mapping with fewer
    than two usable moments is collinear.
    """

    moments = magnetic_structure.get("moments")
    if not isinstance(moments, list):
        return True
    axis: tuple[float, float, float] | None = None
    for moment in moments:
        if not isinstance(moment, Mapping):
            continue
        vector = moment.get("vector")
        if not isinstance(vector, (list, tuple)) or len(vector) < 3:
            continue
        try:
            x, y, z = (float(vector[0]), float(vector[1]), float(vector[2]))
        except (TypeError, ValueError):
            continue
        length = (x * x + y * y + z * z) ** 0.5
        if length <= tolerance:
            continue
        unit = (x / length, y / length, z / length)
        if axis is None:
            axis = unit
            continue
        cross = (
            axis[1] * unit[2] - axis[2] * unit[1],
            axis[2] * unit[0] - axis[0] * unit[2],
            axis[0] * unit[1] - axis[1] * unit[0],
        )
        if (cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) ** 0.5 > angle_tolerance:
            return False
    return True


def propagation_vector_from_structure(
    structure: Structure, *, strict: bool = False
) -> tuple[float, float, float]:
    """Return the structure's propagation vector in reciprocal-lattice units.

    The top-level ``propagation_vector`` property wins; otherwise the one inside
    ``magnetic_structure`` is used; a structure with neither has ``(0, 0, 0)``.

    By default each of the first three components is read on its own, so a
    short vector is zero-padded and a non-numeric component reads as zero.
    With ``strict`` the stored value must be exactly three numeric components,
    and anything else reads as ``(0, 0, 0)``.
    """

    source: object = structure.properties.get("propagation_vector")
    if source is None:
        magnetic = structure.properties.get("magnetic_structure")
        if isinstance(magnetic, Mapping):
            source = magnetic.get("propagation_vector")

    if isinstance(source, (str, bytes, Mapping)):
        return (0.0, 0.0, 0.0)
    try:
        components = list(source)  # type: ignore[call-overload]
    except TypeError:
        return (0.0, 0.0, 0.0)

    if strict:
        if len(components) != 3:
            return (0.0, 0.0, 0.0)
        try:
            return (float(components[0]), float(components[1]), float(components[2]))
        except (TypeError, ValueError):
            return (0.0, 0.0, 0.0)

    vector = [0.0, 0.0, 0.0]
    for index, value in enumerate(components[:3]):
        try:
            vector[index] = float(value)
        except (TypeError, ValueError):
            vector[index] = 0.0
    return (vector[0], vector[1], vector[2])


__all__ = [
    "SPIN_TOLERANCE",
    "atomic_spins_from_magnetic_moments",
    "atomic_spins_from_structure",
    "has_nonzero_spins",
    "magnetic_moments_are_collinear",
    "propagation_vector_from_structure",
]
