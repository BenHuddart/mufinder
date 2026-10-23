"""Typed data structures for CASTEP ``.cell`` and ``.param`` content."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias


ScalarPrimitive: TypeAlias = str | int | float | bool
ScalarSequence: TypeAlias = tuple[ScalarPrimitive, ...]
ScalarValue: TypeAlias = ScalarPrimitive | ScalarSequence


@dataclass(frozen=True, slots=True)
class ValueWithUnit:
    """Represent a scalar or tuple value with an explicit unit token."""

    value: ScalarValue
    unit: str


@dataclass(frozen=True, slots=True)
class KeywordAssignment:
    """A generic keyword/value assignment for either file type."""

    name: str
    value: ScalarValue | ValueWithUnit | None = None


@dataclass(frozen=True, slots=True)
class BlockSection:
    """A generic CASTEP block.

    Rows are stored as tuples to keep serialization deterministic.
    """

    name: str
    rows: tuple[tuple[ScalarPrimitive, ...], ...]
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class LatticeCart:
    """A ``LATTICE_CART`` block."""

    vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class LatticeABC:
    """A ``LATTICE_ABC`` block."""

    lengths: tuple[float, float, float]
    angles: tuple[float, float, float]
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class AtomicPosition:
    """One ionic position row."""

    species: str
    coordinates: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class PositionBlock:
    """A ``POSITIONS_FRAC`` or ``POSITIONS_ABS`` block."""

    positions: tuple[AtomicPosition, ...]
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class SpeciesPotential:
    """One ``SPECIES_POT`` row."""

    species: str
    potential: str


@dataclass(frozen=True, slots=True)
class KPoint:
    """One weighted k-point entry."""

    coordinates: tuple[float, float, float]
    weight: float


CoordinateMode: TypeAlias = Literal["frac", "abs"]


@dataclass(slots=True)
class CellFile:
    """Structured representation of a CASTEP ``.cell`` file."""

    lattice_cart: LatticeCart | None = None
    lattice_abc: LatticeABC | None = None
    positions_frac: PositionBlock | None = None
    positions_abs: PositionBlock | None = None
    atomic_spins: tuple[float, ...] | None = None
    species_pot: tuple[SpeciesPotential, ...] = ()
    kpoints_list: tuple[KPoint, ...] = ()
    kpoint_mp_grid: tuple[int, int, int] | None = None
    kpoint_mp_spacing: float | ValueWithUnit | None = None
    symmetry_generate: bool = False
    scalars: tuple[KeywordAssignment, ...] = ()
    blocks: tuple[BlockSection, ...] = ()


@dataclass(slots=True)
class ParamFile:
    """Structured representation of a CASTEP ``.param`` file."""

    keywords: tuple[KeywordAssignment, ...] = field(default_factory=tuple)
    blocks: tuple[BlockSection, ...] = field(default_factory=tuple)
