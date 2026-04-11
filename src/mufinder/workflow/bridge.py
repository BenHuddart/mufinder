"""Bridge between domain Structure objects and workflow StructureSpec."""

from __future__ import annotations

from collections.abc import Sequence

from mufinder.castep_io.types import AtomicPosition, LatticeCart, PositionBlock
from mufinder.domain.models import Structure

from .types import StructureSpec


class StructureBridgeError(ValueError):
    """Raised when a Structure cannot be converted to a StructureSpec."""


def _has_nonzero_spin(values: Sequence[float], *, tolerance: float = 1e-12) -> bool:
    return any(abs(float(value)) > tolerance for value in values)


def _as_vector3(values: list[float], *, label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise StructureBridgeError(f"{label} must contain exactly three components.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _spins_from_magnetic_structure(structure: Structure) -> list[float] | None:
    magnetic = structure.properties.get("magnetic_structure")
    if not isinstance(magnetic, dict):
        return None

    moments = magnetic.get("moments")
    if not isinstance(moments, list) or not moments:
        return None

    spins = [0.0] * len(structure.species)
    saw_numeric = False
    for index, moment in enumerate(moments[: len(spins)]):
        if not isinstance(moment, dict):
            continue
        vector = moment.get("vector")
        if not isinstance(vector, list) or len(vector) < 3:
            continue
        try:
            spins[index] = float(vector[2])
            saw_numeric = True
        except (TypeError, ValueError):
            continue

    return spins if saw_numeric else None


def structure_to_spec(structure: Structure) -> StructureSpec:
    """Convert a domain Structure into immutable workflow structure data."""

    if len(structure.lattice_vectors) != 3:
        raise StructureBridgeError("Structure must have exactly three lattice vectors.")

    if not structure.species or not structure.fractional_positions:
        raise StructureBridgeError("Structure must contain at least one species and one position.")

    if len(structure.species) != len(structure.fractional_positions):
        raise StructureBridgeError(
            f"Species count ({len(structure.species)}) does not match position count "
            f"({len(structure.fractional_positions)})."
        )

    lattice_cart = LatticeCart(
        vectors=(
            _as_vector3(structure.lattice_vectors[0], label="Lattice vector 1"),
            _as_vector3(structure.lattice_vectors[1], label="Lattice vector 2"),
            _as_vector3(structure.lattice_vectors[2], label="Lattice vector 3"),
        ),
        unit="ang",
    )

    positions = tuple(
        AtomicPosition(
            species=species,
            coordinates=_as_vector3(position, label=f"Fractional position {index + 1}"),
        )
        for index, (species, position) in enumerate(zip(structure.species, structure.fractional_positions))
    )

    raw_spins = structure.properties.get("atomic_spins")
    if raw_spins is None:
        raw_spins = structure.properties.get("initial_magnetic_moments")
    fallback_spins = _spins_from_magnetic_structure(structure)
    if raw_spins is None:
        raw_spins = fallback_spins

    atomic_spins: tuple[float, ...] | None = None
    if raw_spins is not None:
        if not isinstance(raw_spins, list):
            raise StructureBridgeError("Structure atomic_spins must be a list of numeric values.")
        if len(raw_spins) != len(structure.species):
            raise StructureBridgeError(
                f"Atomic spin count ({len(raw_spins)}) does not match species count ({len(structure.species)})."
            )
        try:
            parsed_spins = tuple(float(value) for value in raw_spins)
        except (TypeError, ValueError) as exc:
            raise StructureBridgeError("Structure atomic_spins values must be numeric.") from exc
        if (
            not isinstance(structure.properties.get("supercell_asset"), dict)
            and fallback_spins is not None
            and not _has_nonzero_spin(parsed_spins)
            and _has_nonzero_spin(fallback_spins)
        ):
            atomic_spins = tuple(float(value) for value in fallback_spins)
        else:
            atomic_spins = parsed_spins
    elif fallback_spins is not None and _has_nonzero_spin(fallback_spins):
        atomic_spins = tuple(float(value) for value in fallback_spins)

    unique_species = tuple(dict.fromkeys(structure.species))

    return StructureSpec(
        lattice_cart=lattice_cart,
        positions_frac=PositionBlock(positions=positions),
        species=unique_species,
        atomic_spins=atomic_spins,
        label=structure.label,
        source_structure_id=structure.id,
    )
