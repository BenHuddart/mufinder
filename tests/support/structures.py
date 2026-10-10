"""Host-structure factory shared by the app, science, services and workflow tests.

``host_structure`` builds a :class:`~mufinder.domain.models.Structure` from a
lattice, species and fractional positions. Every call returns fresh lists and a
deep copy of ``properties``, so a test that mutates its structure cannot leak
into another test, even when a module binds its own values once with
``functools.partial``. The presets cover the hosts many modules share.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from typing import Any

from mufinder.domain.models import Structure

Vector = Sequence[float]

#: Two-atom basis with the second atom at the body centre.
BODY_CENTRE: tuple[Vector, ...] = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))
#: Two-atom basis with the second atom a quarter of the way along the diagonal.
QUARTER_DIAGONAL: tuple[Vector, ...] = ((0.0, 0.0, 0.0), (0.25, 0.25, 0.25))


def cubic_lattice(a: float) -> list[list[float]]:
    """Lattice vectors of a cube with edge ``a`` (Å)."""

    return [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]


def host_structure(
    *,
    label: str = "Host",
    lattice: float | Sequence[Vector] = 5.0,
    species: Sequence[str] = ("Si", "Si"),
    positions: Sequence[Vector] = BODY_CENTRE,
    properties: Mapping[str, Any] | None = None,
    **fields: Any,
) -> Structure:
    """Build a host structure; the defaults are two Si atoms in a 5 Å cube.

    ``lattice`` is either a cubic edge length or explicit lattice vectors.
    ``fields`` passes any other ``Structure`` field (``id``, ``origin``,
    ``source_file``, ``parent_structure_id``) through unchanged.
    """

    if len(species) != len(positions):
        raise ValueError(f"{len(species)} species but {len(positions)} positions")
    if isinstance(lattice, (int, float)):
        lattice_vectors = cubic_lattice(lattice)
    else:
        lattice_vectors = [list(vector) for vector in lattice]
    return Structure(
        label=label,
        lattice_vectors=lattice_vectors,
        species=list(species),
        fractional_positions=[list(position) for position in positions],
        properties={} if properties is None else copy.deepcopy(dict(properties)),
        **fields,
    )


def si_pair_host(
    a: float = 4.0,
    *,
    label: str = "Si",
    positions: Sequence[Vector] = QUARTER_DIAGONAL,
    **overrides: Any,
) -> Structure:
    """Two Si atoms in an ``a`` Å cube, the second on the quarter diagonal."""

    return host_structure(label=label, lattice=a, positions=positions, **overrides)


def silicon_oxide_host(**overrides: Any) -> Structure:
    """Three-atom O-Si-O host in a 10 Å cube."""

    values: dict[str, Any] = {
        "lattice": 10.0,
        "species": ("O", "Si", "O"),
        "positions": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    }
    values.update(overrides)
    return host_structure(**values)


def hydrogen_host(a: float = 2.0, *, label: str = "H host", **overrides: Any) -> Structure:
    """A single H atom at the origin of an ``a`` Å cube."""

    return host_structure(
        label=label,
        lattice=a,
        species=("H",),
        positions=((0.0, 0.0, 0.0),),
        **overrides,
    )
