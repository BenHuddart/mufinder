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

import numpy as np

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


def magnetic_silicon_oxide_host(**overrides: Any) -> Structure:
    """Si and O in a 5 x 6 x 7 Å box with spins ``[1.0, -1.5]`` and ``k = (0, 0, 1/4)``."""

    values: dict[str, Any] = {
        "lattice": [[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]],
        "species": ("Si", "O"),
        "positions": ((0.0, 0.0, 0.0), (0.25, 0.5, 0.75)),
        "source_file": "imports/host.cif",
        "properties": {
            "pbc": [True, True, True],
            "atomic_spins": [1.0, -1.5],
            "propagation_vector": [0.0, 0.0, 0.25],
        },
    }
    values.update(overrides)
    return host_structure(**values)


def hexagonal_magnesium_oxide_host(**overrides: Any) -> Structure:
    """Mg and O in a hexagonal cell (a = 4.2 Å, c = 6.8 Å, gamma = 120°)."""

    a = 4.2
    c = 6.8
    values: dict[str, Any] = {
        "label": "Hex",
        "lattice": [[a, 0.0, 0.0], [-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0], [0.0, 0.0, c]],
        "species": ("Mg", "O"),
        "positions": ((0.0, 0.0, 0.0), (1.0 / 3.0, 2.0 / 3.0, 0.25)),
        "source_file": "imports/hex.cif",
        "properties": {"pbc": [True, True, True]},
    }
    values.update(overrides)
    return host_structure(**values)


def relaxed_muon_supercell(
    host: Structure,
    repeats: Sequence[int],
    muon_position: Vector,
    displacements: Mapping[tuple[float, float, float], Vector] | None = None,
    *,
    label: str = "Relaxed muon supercell",
) -> Structure:
    """A diagonal ``repeats`` supercell of ``host`` with an ``H:mu`` and displaced host atoms.

    ``muon_position`` is in host fractional coordinates. ``displacements`` maps a
    host-frame fractional site, taken modulo the supercell, to a Cartesian
    displacement in Å. Atoms are listed cell by cell, then in host order, and
    the muon comes last.
    """

    counts = [int(value) for value in repeats]
    lattice = [
        [float(value) * counts[row] for value in host.lattice_vectors[row]] for row in range(3)
    ]
    moves = {
        tuple(
            round(float(value) % counts[axis], 6) for axis, value in enumerate(site)
        ): displacement
        for site, displacement in (displacements or {}).items()
    }
    species: list[str] = []
    positions: list[list[float]] = []
    for i in range(counts[0]):
        for j in range(counts[1]):
            for k in range(counts[2]):
                for symbol, position in zip(host.species, host.fractional_positions):
                    unit = [float(position[0]) + i, float(position[1]) + j, float(position[2]) + k]
                    key = tuple(round(value % counts[axis], 6) for axis, value in enumerate(unit))
                    shift = moves.get(key, (0.0, 0.0, 0.0))
                    cartesian = [
                        sum(
                            unit[axis] * float(host.lattice_vectors[axis][column])
                            for axis in range(3)
                        )
                        + shift[column]
                        for column in range(3)
                    ]
                    species.append(str(symbol))
                    positions.append(_fractional_in(lattice, cartesian))
    species.append("H:mu")
    positions.append([float(muon_position[axis]) / counts[axis] for axis in range(3)])
    return Structure(
        label=label,
        lattice_vectors=lattice,
        species=species,
        fractional_positions=positions,
        parent_structure_id=host.id,
    )


def _fractional_in(lattice: Sequence[Vector], cartesian: Vector) -> list[float]:
    return [
        float(value) % 1.0
        for value in np.linalg.solve(np.asarray(lattice, dtype=float).T, cartesian)
    ]
