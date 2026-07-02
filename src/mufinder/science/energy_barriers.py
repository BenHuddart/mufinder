"""Symmetry-guided helpers for Energy Barriers workflows."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Sequence

from ase import Atoms
from ase.build import make_supercell
from ase.mep import NEB, idpp_interpolate
import numpy as np
from scipy.optimize import linear_sum_assignment

from mufinder.domain.models import Structure
from mufinder.science.spglib_compat import get_crystal_symmetry


_LOGGER = logging.getLogger(__name__)

_MUON_LABELS = {"x", "h:mu", "mu", "muon"}
_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")

# Number of images used for the internal IDPP interpolation. Odd so there is an exact
# central image at reaction coordinate 0.5 -- the only image CASTEP consumes (it places the
# intermediate at the climbing image and re-interpolates each half-band linearly).
_IDPP_BAND_IMAGES = 7
# Accept the IDPP intermediate unless it reduces the muon's clearance from neighbouring atoms
# by more than this margin (Angstrom) versus the plain linear midpoint -- a cheap safety net
# against a misbehaving optimiser, while still using IDPP for ties and improvements.
_IDPP_CLEARANCE_MARGIN_ANG = 1e-3


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
    if not start_positions or not end_positions:
        return ()

    start_array = np.asarray(start_positions, dtype=float)
    end_array = np.asarray(end_positions, dtype=float)
    deltas = _wrapped_fractional_delta(start_array[:, None, :], end_array[None, :, :])
    lengths = np.linalg.norm(deltas @ reactant.cell.array, axis=2)
    mask = (lengths > tol) & (lengths <= float(max_distance_ang))
    start_indexes, end_indexes = np.nonzero(mask)
    if start_indexes.size == 0:
        return ()

    ordered_indexes = np.argsort(lengths[start_indexes, end_indexes], kind="stable")
    canonical_by_delta: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for ordered_index in ordered_indexes.tolist():
        start_index = int(start_indexes[ordered_index])
        end_index = int(end_indexes[ordered_index])
        delta = np.asarray(deltas[start_index, end_index], dtype=float)
        delta_key = tuple(int(value) for value in np.rint(delta / tol).astype(int))
        key = canonical_by_delta.get(delta_key)
        if key is None:
            key = _canonical_delta_key(delta, rotations=rotations, tol=tol)
            canonical_by_delta[delta_key] = key
        candidate = MuonPath(
            start_fractional=tuple(float(value % 1.0) for value in start_array[start_index]),
            end_fractional=tuple(float(value % 1.0) for value in end_array[end_index]),
            length_ang=float(lengths[start_index, end_index]),
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


def idpp_intermediate_structure(reactant: Structure, product: Structure) -> Structure:
    """Return the central NEB image for a reactant->product muon hop, seeded by IDPP.

    CASTEP's NEB reads a single INTERMEDIATE structure, places it at the climbing image and
    linearly interpolates each half-band. A plain linear midpoint can drop the muon onto a host
    atom (huge initial forces / non-convergence). IDPP (Smidstrup et al., J. Chem. Phys. 140,
    214106 (2014)) interpolates the pairwise distance matrix instead, bowing the muon around
    obstacles; we optimise a small band and return its central image. On any failure -- or if the
    optimiser somehow worsens the muon's clearance -- we fall back to the linear minimum-image
    midpoint so job generation never blocks. Both fall-backs log a warning, because a silent
    degradation to the linear midpoint reintroduces exactly the muon-on-host overlap IDPP prevents.
    """

    reactant_atoms = _structure_to_atoms(reactant, require_muon=True)
    product_atoms = _structure_to_atoms(product, require_muon=True)
    _validate_transition_state_cells(reactant_atoms, product_atoms)

    linear = _linear_midpoint_atoms(reactant_atoms, product_atoms)
    central = linear
    try:
        idpp = _idpp_central_image(reactant_atoms, product_atoms)
        idpp_clearance = _muon_host_clearance(idpp)
        linear_clearance = _muon_host_clearance(linear)
        if idpp_clearance + _IDPP_CLEARANCE_MARGIN_ANG >= linear_clearance:
            central = idpp
        else:
            _LOGGER.warning(
                "IDPP intermediate for %r reduced the muon-host clearance "
                "(%.3f < %.3f A); using the linear midpoint instead.",
                reactant.label,
                idpp_clearance,
                linear_clearance,
            )
    except Exception:  # noqa: BLE001 - seeding is a heuristic; never block job generation
        _LOGGER.warning(
            "IDPP seeding failed for %r; falling back to the linear midpoint.",
            reactant.label,
            exc_info=True,
        )
        central = linear

    return _atoms_to_structure(central, template=reactant, origin="energy_barriers_neb_intermediate")


def _linear_midpoint_atoms(reactant_atoms: Atoms, product_atoms: Atoms) -> Atoms:
    reactant_frac = reactant_atoms.get_scaled_positions(wrap=True)
    product_frac = product_atoms.get_scaled_positions(wrap=True)
    # Wrap each displacement into (-0.5, 0.5] so the muon takes the short path across the
    # periodic boundary rather than the long way round the cell.
    midpoint_frac = reactant_frac + 0.5 * _wrapped_fractional_delta(reactant_frac, product_frac)
    midpoint = reactant_atoms.copy()
    midpoint.set_scaled_positions(midpoint_frac)
    return midpoint


def _idpp_central_image(reactant_atoms: Atoms, product_atoms: Atoms) -> Atoms:
    # The band size is decoupled from CASTEP's image count (TSSEARCH_MAX_PATH_POINTS): CASTEP
    # consumes only the single central image and re-interpolates each half-band, so this controls
    # the IDPP optimisation quality, not the eventual NEB resolution. Odd so an exact rc=0.5 image
    # exists.
    images_count = max(5, int(_IDPP_BAND_IMAGES))
    if images_count % 2 == 0:
        images_count += 1
    images = [reactant_atoms.copy() for _ in range(images_count)]
    images[-1] = product_atoms.copy()
    # 'improvedtangent' is ASE's recommended (and default-pending) tangent estimator; set it
    # explicitly to silence the deprecation warning and get a better-behaved IDPP optimisation.
    neb = NEB(images, method="improvedtangent")
    neb.interpolate(mic=True)  # linear spread of the interior images (no file output)
    # Refine with IDPP; traj=None/log=None suppress the idpp.traj/idpp.log files ASE writes by default.
    idpp_interpolate(neb, traj=None, log=None, mic=True)
    return images[images_count // 2]


def _muon_host_clearance(atoms: Atoms) -> float:
    muon = _muon_index(atoms)
    frac = atoms.get_scaled_positions(wrap=True)
    cell = atoms.cell.array
    distances = [
        _minimum_image_distance(frac[muon], frac[index], cell)
        for index in range(len(atoms))
        if index != muon
    ]
    # A muon-only cell has no host atoms to collide with -> infinite clearance.
    return min(distances) if distances else float("inf")


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
    rotations = np.asarray(symmetry["rotations"], dtype=float)
    translations = np.asarray(symmetry["translations"], dtype=float)
    transformed = (rotations @ np.asarray(muon_position, dtype=float).reshape(3, 1)).reshape(-1, 3) + translations
    wrapped = np.mod(transformed, 1.0)
    keys = np.rint(wrapped / tol).astype(int)
    _unique_keys, unique_indexes = np.unique(keys, axis=0, return_index=True)
    ordered_indexes = sorted(int(index) for index in unique_indexes.tolist())
    return tuple(
        tuple(float(value) for value in wrapped[index])
        for index in ordered_indexes
    )


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
