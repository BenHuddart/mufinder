"""Symmetry-guided helpers for Energy Barriers workflows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from ase import Atoms
from ase.build import make_supercell
import numpy as np
from scipy.optimize import linear_sum_assignment

from mufinder.domain.models import Structure
from mufinder.science.spglib_compat import get_crystal_symmetry


_MUON_LABELS = {"x", "h:mu", "mu", "muon"}
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


class EnergyBarrierError(ValueError):
    """Raised when an Energy Barriers input cannot be prepared."""


@dataclass(frozen=True, slots=True)
class MuonPath:
    """One symmetry-distinct muon path between two relaxed structures."""

    start_fractional: tuple[float, float, float]
    end_fractional: tuple[float, float, float]
    length_ang: float


@dataclass(frozen=True, slots=True)
class PreparedTransitionState:
    """Reactant/product structures aligned to one selected transition path."""

    reactant: Structure
    product: Structure


def enumerate_distinct_muon_paths(
    *,
    pristine_structure: Structure,
    reactant_structure: Structure,
    product_structure: Structure,
    symprec: float = 1e-3,
    tol: float = 1e-4,
    max_distance_ang: float | None = None,
) -> tuple[MuonPath, ...]:
    """Return symmetry-distinct ordered paths from reactant to product muon sites."""

    pristine = _structure_to_atoms(pristine_structure, require_muon=False)
    reactant = _structure_to_atoms(reactant_structure, require_muon=True)
    product = _structure_to_atoms(product_structure, require_muon=True)

    _validate_transition_state_cells(reactant, product)
    pristine_supercell = _make_supercell_like(pristine, reactant)

    reactant_muon = _single_muon_position(reactant)
    product_muon = _single_muon_position(product)

    start_positions = _symmetry_equivalent_muon_positions(
        pristine_supercell,
        reactant_muon,
        symprec=symprec,
        tol=tol,
    )
    end_positions = _symmetry_equivalent_muon_positions(
        pristine_supercell,
        product_muon,
        symprec=symprec,
        tol=tol,
    )

    if max_distance_ang is None:
        max_distance_ang = _default_max_distance(
            pristine_supercell,
            reactant_muon,
            product_muon,
            start_positions,
            end_positions,
        )

    symmetry = get_crystal_symmetry(
        (pristine_supercell.cell.array, pristine_supercell.get_scaled_positions(), pristine_supercell.numbers),
        symprec=float(symprec),
    )
    if symmetry is None:
        raise EnergyBarrierError("Could not determine host symmetry for path enumeration.")
    rotations = symmetry["rotations"]

    unique: dict[tuple[int, int, int], MuonPath] = {}
    for start in start_positions:
        for end in end_positions:
            delta = _wrapped_fractional_delta(np.asarray(start, dtype=float), np.asarray(end, dtype=float))
            length = float(np.linalg.norm(delta @ reactant.cell.array))
            if length <= tol or length > float(max_distance_ang):
                continue
            key = _canonical_delta_key(delta, rotations=rotations, tol=tol)
            candidate = MuonPath(
                start_fractional=tuple(float(value % 1.0) for value in start),
                end_fractional=tuple(float(value % 1.0) for value in end),
                length_ang=length,
            )
            existing = unique.get(key)
            if existing is None or candidate.length_ang < existing.length_ang:
                unique[key] = candidate

    return tuple(sorted(unique.values(), key=lambda path: (path.length_ang, path.end_fractional, path.start_fractional)))


def prepare_transition_state_structures(
    *,
    pristine_structure: Structure,
    reactant_structure: Structure,
    product_structure: Structure,
    reactant_anchor_fractional: Sequence[float],
    product_anchor_fractional: Sequence[float],
    symprec: float = 1e-3,
    tol: float = 1e-4,
) -> PreparedTransitionState:
    """Align reactant/product structures to one chosen path anchor pair."""

    pristine = _structure_to_atoms(pristine_structure, require_muon=False)
    reactant = _structure_to_atoms(reactant_structure, require_muon=True)
    product = _structure_to_atoms(product_structure, require_muon=True)

    _validate_transition_state_cells(reactant, product)
    pristine_supercell = _make_supercell_like(pristine, reactant)

    reactant_target = np.asarray(tuple(float(value) for value in reactant_anchor_fractional), dtype=float)
    product_target = np.asarray(tuple(float(value) for value in product_anchor_fractional), dtype=float)
    if reactant_target.shape != (3,) or product_target.shape != (3,):
        raise EnergyBarrierError("Transition-state anchor positions must contain exactly three fractional coordinates.")

    aligned_reactant = _transform_structure_by_symmetry(
        pristine_supercell,
        reactant,
        _single_muon_position(reactant),
        reactant_target,
        symprec=symprec,
        tol=tol,
    )
    aligned_product = _transform_structure_by_symmetry(
        pristine_supercell,
        product,
        _single_muon_position(product),
        product_target,
        symprec=symprec,
        tol=tol,
    )

    _reorder_atoms_like(aligned_reactant, aligned_product)
    _minimise_displacements(aligned_reactant, aligned_product)

    return PreparedTransitionState(
        reactant=_atoms_to_structure(aligned_reactant, reactant_structure, origin="energy_barriers_transition_state"),
        product=_atoms_to_structure(aligned_product, product_structure, origin="energy_barriers_transition_state"),
    )


def _default_max_distance(
    pristine_supercell: Atoms,
    reactant_muon: np.ndarray,
    product_muon: np.ndarray,
    start_positions: tuple[tuple[float, float, float], ...],
    end_positions: tuple[tuple[float, float, float], ...],
) -> float:
    minima: list[float] = []
    for origin, destinations in ((reactant_muon, end_positions), (product_muon, start_positions)):
        best = np.inf
        for destination in destinations:
            delta = _wrapped_fractional_delta(origin, np.asarray(destination, dtype=float))
            length = float(np.linalg.norm(delta @ pristine_supercell.cell.array))
            if 1e-12 < length < best:
                best = length
        if np.isfinite(best):
            minima.append(float(best))
    if not minima:
        raise EnergyBarrierError("No distinct muon paths were found between the selected relaxed structures.")
    return 1.5 * max(minima)


def _canonical_delta_key(delta: np.ndarray, *, rotations: Sequence[np.ndarray], tol: float) -> tuple[int, int, int]:
    candidates = []
    for rotation in rotations:
        rotated = np.asarray(rotation, dtype=float) @ delta
        wrapped = ((rotated + 0.5) % 1.0) - 0.5
        candidates.append(tuple(int(value) for value in np.round(wrapped / tol).astype(int)))
        candidates.append(tuple(int(value) for value in np.round((-wrapped) / tol).astype(int)))
    return min(candidates)


def _validate_transition_state_cells(lhs: Atoms, rhs: Atoms) -> None:
    if len(lhs) != len(rhs):
        raise EnergyBarrierError("Transition-state endpoints must contain the same number of atoms.")
    if list(lhs.numbers) != list(rhs.numbers):
        raise EnergyBarrierError("Transition-state endpoints must contain atoms in matching species order.")
    if not np.allclose(lhs.cell.array, rhs.cell.array, atol=1e-8):
        raise EnergyBarrierError("Transition-state endpoints must share the same simulation cell.")


def _wrapped_fractional_delta(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    return ((delta + 0.5) % 1.0) - 0.5


def _is_muon_label(symbol: str) -> bool:
    return str(symbol).strip().casefold() in _MUON_LABELS


def _element_symbol(symbol: str) -> str:
    raw = str(symbol).strip()
    match = _ELEMENT_RE.match(raw)
    if match is None:
        raise EnergyBarrierError(f"Could not resolve element symbol from '{symbol}'.")
    return match.group(1)


def _structure_to_atoms(structure: Structure, *, require_muon: bool) -> Atoms:
    labels = [str(symbol) for symbol in structure.species]
    symbols = ["H" if _is_muon_label(symbol) else _element_symbol(symbol) for symbol in labels]
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=[list(position) for position in structure.fractional_positions],
        cell=np.asarray(structure.lattice_vectors, dtype=float),
        pbc=[True, True, True],
    )
    atoms.info["mufinder_species_labels"] = list(labels)
    spins = structure.properties.get("atomic_spins")
    if isinstance(spins, list) and len(spins) == len(labels):
        atoms.info["mufinder_atomic_spins"] = [float(value) for value in spins]
        atoms.set_initial_magnetic_moments([float(value) for value in spins])
    if require_muon and sum(1 for label in labels if _is_muon_label(label)) != 1:
        raise EnergyBarrierError("Transition-state endpoint structures must contain exactly one muon.")
    return atoms


def _atoms_to_structure(atoms: Atoms, template: Structure, *, origin: str) -> Structure:
    labels = atoms.info.get("mufinder_species_labels")
    if not isinstance(labels, list) or len(labels) != len(atoms):
        labels = [str(symbol) for symbol in atoms.get_chemical_symbols()]
    result = Structure(
        label=template.label,
        origin=origin,
        lattice_vectors=[[float(component) for component in row] for row in atoms.cell.array.tolist()],
        species=[str(label) for label in labels],
        fractional_positions=[[float(component) for component in row] for row in atoms.get_scaled_positions(wrap=True).tolist()],
        source_file=template.source_file,
        parent_structure_id=template.parent_structure_id,
        properties=dict(template.properties),
    )
    spins = atoms.info.get("mufinder_atomic_spins")
    if isinstance(spins, list) and len(spins) == len(labels):
        result.properties["atomic_spins"] = [float(value) for value in spins]
    return result


def _single_muon_position(atoms: Atoms) -> np.ndarray:
    labels = atoms.info.get("mufinder_species_labels", atoms.get_chemical_symbols())
    muon_indices = [index for index, label in enumerate(labels) if _is_muon_label(str(label))]
    if len(muon_indices) != 1:
        raise EnergyBarrierError("Transition-state endpoint structures must contain exactly one muon.")
    return np.asarray(atoms.get_scaled_positions(wrap=True)[muon_indices[0]], dtype=float)


def _apply_symmetry_operation(positions: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return (np.asarray(positions, dtype=float) @ np.asarray(rotation, dtype=float).T) + np.asarray(translation, dtype=float)


def _symmetry_equivalent_muon_positions(
    structure: Atoms,
    muon_position: np.ndarray,
    *,
    symprec: float,
    tol: float,
) -> tuple[tuple[float, float, float], ...]:
    symmetry = get_crystal_symmetry(
        (structure.cell.array, structure.get_scaled_positions(), structure.numbers),
        symprec=float(symprec),
    )
    if symmetry is None:
        return ()
    seen: set[tuple[int, int, int]] = set()
    positions: list[tuple[float, float, float]] = []
    for rotation, translation in zip(symmetry["rotations"], symmetry["translations"], strict=False):
        transformed = _apply_symmetry_operation(muon_position.reshape(1, 3), rotation, translation).reshape(3)
        wrapped = np.mod(transformed, 1.0)
        key = tuple(int(value) for value in np.round(wrapped / tol).astype(int))
        if key in seen:
            continue
        seen.add(key)
        positions.append(tuple(float(value) for value in wrapped))
    return tuple(positions)


def _make_supercell_like(pristine: Atoms, target: Atoms) -> Atoms:
    matrix = np.rint(target.cell.array @ np.linalg.inv(pristine.cell.array)).astype(int)
    if not np.allclose(matrix @ pristine.cell.array, target.cell.array, atol=1e-6):
        raise EnergyBarrierError("Could not construct a pristine supercell matching the relaxed endpoint cell.")
    return make_supercell(pristine, matrix)


def _is_symmetry_equivalent(
    pristine: Atoms,
    position: np.ndarray,
    target: np.ndarray,
    *,
    symprec: float,
    tol: float,
) -> bool:
    for equivalent in _symmetry_equivalent_muon_positions(pristine, position, symprec=symprec, tol=tol):
        delta = _wrapped_fractional_delta(np.asarray(equivalent, dtype=float), np.asarray(target, dtype=float))
        if np.allclose(delta, 0.0, atol=tol):
            return True
    return False


def _transform_structure_by_symmetry(
    pristine: Atoms,
    relaxed: Atoms,
    current_muon: np.ndarray,
    target_muon: np.ndarray,
    *,
    symprec: float,
    tol: float,
) -> Atoms:
    if not _is_symmetry_equivalent(pristine, current_muon, target_muon, symprec=symprec, tol=tol):
        raise EnergyBarrierError("Selected path endpoint is not symmetry-equivalent to the relaxed muon position.")

    symmetry = get_crystal_symmetry(
        (pristine.cell.array, pristine.get_scaled_positions(), pristine.numbers),
        symprec=float(symprec),
    )
    if symmetry is None:
        raise EnergyBarrierError("Could not determine a symmetry operation for the selected transition-state path.")
    rotation = None
    translation = None
    current_wrapped = np.mod(current_muon, 1.0)
    target_wrapped = np.mod(target_muon, 1.0)
    for candidate_rotation, candidate_translation in zip(
        symmetry["rotations"], symmetry["translations"], strict=False
    ):
        transformed = _apply_symmetry_operation(current_wrapped.reshape(1, 3), candidate_rotation, candidate_translation).reshape(3)
        if np.allclose(_wrapped_fractional_delta(transformed, target_wrapped), 0.0, atol=tol):
            rotation = np.asarray(candidate_rotation, dtype=float)
            translation = np.asarray(candidate_translation, dtype=float)
            break
    if rotation is None or translation is None:
        raise EnergyBarrierError("Could not determine a symmetry operation for the selected transition-state path.")

    muon_index = _muon_index(relaxed)
    positions = relaxed.get_scaled_positions(wrap=True)
    displacements = np.zeros_like(positions)
    for index, position in enumerate(positions):
        if index == muon_index:
            continue
        displacements[index] = _wrapped_fractional_delta(current_wrapped, np.asarray(position, dtype=float))

    transformed = relaxed.copy()
    new_positions = np.zeros_like(positions)
    transformed_displacements = _apply_symmetry_operation(displacements, rotation, np.zeros(3, dtype=float))
    for index in range(len(positions)):
        if index == muon_index:
            new_positions[index] = target_wrapped
            continue
        new_positions[index] = target_wrapped + transformed_displacements[index]
    transformed.set_scaled_positions(new_positions)
    return transformed


def _muon_index(atoms: Atoms) -> int:
    labels = atoms.info.get("mufinder_species_labels", atoms.get_chemical_symbols())
    matches = [index for index, label in enumerate(labels) if _is_muon_label(str(label))]
    if len(matches) != 1:
        raise EnergyBarrierError("Transition-state endpoint structures must contain exactly one muon.")
    return int(matches[0])


def _minimum_image_distance(lhs: np.ndarray, rhs: np.ndarray, cell: np.ndarray) -> float:
    delta = _wrapped_fractional_delta(lhs, rhs)
    return float(np.linalg.norm(delta @ cell))


def _reorder_atoms_like(reference: Atoms, target: Atoms) -> None:
    penalties = np.zeros((len(reference), len(target)), dtype=float)
    reference_positions = reference.get_scaled_positions(wrap=True)
    target_positions = target.get_scaled_positions(wrap=True)
    reference_numbers = np.asarray(reference.numbers, dtype=int)
    target_numbers = np.asarray(target.numbers, dtype=int)
    cell = reference.cell.array
    for row in range(len(reference)):
        for column in range(len(target)):
            if reference_numbers[row] != target_numbers[column]:
                penalties[row, column] = 1e9
                continue
            penalties[row, column] = _minimum_image_distance(reference_positions[row], target_positions[column], cell)
    row_ids, column_ids = linear_sum_assignment(penalties)
    order = [int(column) for column in column_ids[np.argsort(row_ids)]]
    target.set_scaled_positions(target_positions[order])
    target.set_atomic_numbers(target_numbers[order])
    labels = target.info.get("mufinder_species_labels")
    if isinstance(labels, list) and len(labels) == len(order):
        target.info["mufinder_species_labels"] = [str(labels[index]) for index in order]
    spins = target.info.get("mufinder_atomic_spins")
    if isinstance(spins, list) and len(spins) == len(order):
        target.info["mufinder_atomic_spins"] = [float(spins[index]) for index in order]


def _minimise_displacements(reference: Atoms, target: Atoms) -> None:
    reference_positions = reference.get_scaled_positions(wrap=True)
    target_positions = target.get_scaled_positions(wrap=True)
    adjusted = np.zeros_like(target_positions)
    for index, (lhs, rhs) in enumerate(zip(reference_positions, target_positions, strict=False)):
        adjusted[index] = lhs + _wrapped_fractional_delta(lhs, rhs)
    target.set_scaled_positions(adjusted)