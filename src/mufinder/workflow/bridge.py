"""Bridge between domain Structure objects and workflow StructureSpec."""

from __future__ import annotations

from mufinder.castep_io.types import AtomicPosition, LatticeCart, PositionBlock
from mufinder.domain.magnetic_properties import atomic_spins_from_structure, has_nonzero_spins
from mufinder.domain.models import Structure

from .types import StructureSpec


class StructureBridgeError(ValueError):
    """Raised when a Structure cannot be converted to a StructureSpec."""


def _as_vector3(values: list[float], *, label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise StructureBridgeError(f"{label} must contain exactly three components.")
    return (float(values[0]), float(values[1]), float(values[2]))


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
        for index, (species, position) in enumerate(
            zip(structure.species, structure.fractional_positions)
        )
    )

    raw_spins = structure.properties.get("atomic_spins")
    if raw_spins is None:
        raw_spins = structure.properties.get("initial_magnetic_moments")
    fallback_spins = atomic_spins_from_structure(structure)
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
            and not has_nonzero_spins(parsed_spins)
            and has_nonzero_spins(fallback_spins)
        ):
            atomic_spins = tuple(float(value) for value in fallback_spins)
        else:
            atomic_spins = parsed_spins
    elif fallback_spins is not None and has_nonzero_spins(fallback_spins):
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
