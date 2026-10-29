"""Symmetry-guided helpers for Energy Barriers workflows."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Sequence

from ase import Atoms
from ase.build import make_supercell
from ase.mep import NEB, idpp_interpolate
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from mufinder.domain.magnetic_properties import (
    has_nonzero_spins,
    magnetic_moments_are_collinear,
    propagation_vector_from_structure,
)
from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import is_muon_label, structure_to_ase_atoms
from mufinder.science.lattice import (
    nearest_image_delta_matrix,
    nearest_image_deltas,
    nearest_image_distance_matrix,
)
from mufinder.science.spglib_compat import (
    SYMMETRY_TOLERANCE_ANG,
    SymmetryOperations,
    symmetry_operations,
    symmetry_orbit,
)
from mufinder.science.supercell import build_supercell_structure


_LOGGER = logging.getLogger(__name__)


# Number of images used for the internal IDPP interpolation. Odd so there is an exact
# central image at reaction coordinate 0.5 -- the only image CASTEP consumes (it places the
# intermediate at the climbing image and re-interpolates each half-band linearly).
_IDPP_BAND_IMAGES = 7
# Accept the IDPP intermediate unless it reduces the muon's clearance from neighbouring atoms
# by more than this margin (Angstrom) versus the plain linear midpoint -- a cheap safety net
# against a misbehaving optimiser, while still using IDPP for ties and improvements.
_IDPP_CLEARANCE_MARGIN_ANG = 1e-3
# ``Atoms.info`` key for the pristine supercell's collinear spins; present only when path
# symmetry respects the magnetic order.
_PATH_SYMMETRY_SPINS = "mufinder_path_symmetry_spins"
# Two spins match when they differ by at most this fraction of the largest spin magnitude:
# loose enough for spins read from calculated populations (2.97 against -2.96), tight
# enough to tell sublattices of a ferrimagnet apart.
_SPIN_MATCH_RELATIVE_TOLERANCE = 0.05


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


@dataclass(frozen=True, slots=True)
class NebFarmDistribution:
    """How CASTEP spreads a NEB band's images over task farms.

    ``images`` is ``TSSEARCH_MAX_PATH_POINTS`` -- the intermediate images only.
    CASTEP's first sweep also computes the two endpoints, so it distributes
    ``images + 2``; every later sweep distributes ``images``. Wall-clock per
    sweep tracks the *busiest* farm, hence ``efficiency`` below.
    """

    images: int
    farms: int
    #: Images per farm on the first sweep (endpoints included).
    first_sweep_counts: tuple[int, ...]
    #: Images per farm on every subsequent sweep -- the steady-state cost.
    steady_counts: tuple[int, ...]
    #: Steady-state parallel efficiency, ``images / (farms * max(steady_counts))``.
    #: 1.0 when every farm carries the same load.
    efficiency: float
    #: Farms with no steady-state work at all (``farms > images``).
    idle_farms: int


def neb_farm_image_counts(total_images: int, farms: int) -> tuple[int, ...]:
    """Return the per-farm image counts for a CASTEP NEB sweep.

    Mirrors ``tssearch_neb_farm_indices`` in CASTEP's ``tssearch_neb.f90``: a
    contiguous block split where the first ``total_images % farms`` farms each
    take one extra image.
    """

    total = max(0, int(total_images))
    count = max(1, int(farms))
    quotient, remainder = divmod(total, count)
    return tuple(quotient + 1 if index < remainder else quotient for index in range(count))


def neb_farm_distribution(*, images: int, farms: int) -> NebFarmDistribution:
    """Describe the image-to-farm split for a NEB band.

    Used both for the configuration hint shown to the user and for
    submission-time validation, so the arithmetic lives in exactly one place.
    """

    image_count = max(0, int(images))
    farm_count = max(1, int(farms))
    steady = neb_farm_image_counts(image_count, farm_count)
    busiest = max(steady) if steady else 0
    efficiency = (image_count / (farm_count * busiest)) if busiest else 0.0
    return NebFarmDistribution(
        images=image_count,
        farms=farm_count,
        first_sweep_counts=neb_farm_image_counts(image_count + 2, farm_count),
        steady_counts=steady,
        efficiency=efficiency,
        idle_farms=sum(1 for count in steady if count == 0),
    )


def path_symmetry_uses_magnetic_order(
    *,
    pristine_structure: Structure,
    reactant_structure: Structure,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> bool:
    """Return whether the host's magnetic order lowers the path symmetry.

    Path symmetry is the group of crystal operations, in the pristine supercell
    matching ``reactant_structure``'s cell, that keep the host's collinear spin
    pattern or flip every spin together: the symmetry of a collinear,
    spin-polarised calculation without spin-orbit coupling. This returns ``True``
    only when that group is smaller than the crystal group, so some hops that the
    crystal relates are distinct. Without per-atom ``atomic_spins``, or with
    non-collinear ``magnetic_structure`` moments, the path symmetry is the crystal
    symmetry and this returns ``False``.
    """

    reactant = _structure_to_atoms(reactant_structure, require_muon=True)
    pristine_supercell = _pristine_supercell(pristine_structure, reactant)
    crystal = _crystal_operations(pristine_supercell, symprec=symprec)
    path = _path_symmetry_operations(pristine_supercell, symprec=symprec)
    if crystal is None or path is None:
        return False
    return len(path[0]) < len(crystal[0])


def enumerate_distinct_muon_paths(
    *,
    pristine_structure: Structure,
    reactant_structure: Structure,
    product_structure: Structure,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    tol: float = 1e-4,
    max_distance_ang: float | None = None,
) -> tuple[MuonPath, ...]:
    """Return symmetry-distinct ordered paths from reactant to product muon sites.

    Each candidate hop is a (start, end) pair of positions from the start and
    end orbits. Two hops are equivalent when one symmetry operation maps the
    start onto the other start and the end onto the other end; when both
    endpoints lie on one orbit, a hop and its reverse are also equivalent. One
    member of each class is returned: the first in length order (all members
    have the same length). The operations are the path symmetry described in
    :func:`path_symmetry_uses_magnetic_order`: crystal operations that keep the
    host's collinear spin pattern or flip it as a whole.
    """

    reactant = _structure_to_atoms(reactant_structure, require_muon=True)
    product = _structure_to_atoms(product_structure, require_muon=True)

    _validate_transition_state_cells(reactant, product)
    pristine_supercell = _pristine_supercell(pristine_structure, reactant)

    reactant_muon = _single_muon_position(reactant)
    product_muon = _single_muon_position(product)

    operations = _path_symmetry_operations(pristine_supercell, symprec=symprec)
    if operations is None:
        raise EnergyBarrierError("Could not determine host symmetry for path enumeration.")

    start_positions = _symmetry_equivalent_muon_positions(operations, reactant_muon, tol=tol)
    end_positions = _symmetry_equivalent_muon_positions(operations, product_muon, tol=tol)

    if max_distance_ang is None:
        max_distance_ang = _default_max_distance(
            pristine_supercell,
            reactant_muon,
            product_muon,
            start_positions,
            end_positions,
        )

    if not start_positions or not end_positions:
        return ()

    start_array = np.asarray(start_positions, dtype=float)
    end_array = np.asarray(end_positions, dtype=float)
    deltas = nearest_image_delta_matrix(start_array, end_array, reactant.cell.array)
    lengths = np.linalg.norm(deltas @ reactant.cell.array, axis=2)
    mask = (lengths > tol) & (lengths <= float(max_distance_ang))
    start_indexes, end_indexes = np.nonzero(mask)
    if start_indexes.size == 0:
        return ()

    ordered_indexes = np.argsort(lengths[start_indexes, end_indexes], kind="stable")
    pair_starts = start_indexes[ordered_indexes]
    pair_ends = end_indexes[ordered_indexes]
    classes = _pair_orbit_classes(
        pair_starts,
        pair_ends,
        start_array=start_array,
        end_array=end_array,
        operations=operations,
        tol=tol,
    )

    unique: dict[int, MuonPath] = {}
    for pair, root in enumerate(classes):
        if root in unique:
            continue
        start_index = int(pair_starts[pair])
        end_index = int(pair_ends[pair])
        unique[root] = MuonPath(
            start_fractional=tuple(float(value % 1.0) for value in start_array[start_index]),
            end_fractional=tuple(float(value % 1.0) for value in end_array[end_index]),
            length_ang=float(lengths[start_index, end_index]),
        )

    return tuple(
        sorted(
            unique.values(),
            key=lambda path: (path.length_ang, path.end_fractional, path.start_fractional),
        )
    )


def equivalent_path_indexes(
    *,
    pristine_structure: Structure,
    reactant_structure: Structure,
    product_structure: Structure,
    paths: Sequence[MuonPath],
    hops: Sequence[tuple[Sequence[float], Sequence[float]]],
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    tol: float = 1e-4,
) -> list[int | None]:
    """Return, for each stored hop, the index in ``paths`` of the path it is equivalent to.

    ``paths`` are :func:`enumerate_distinct_muon_paths` results for the same
    endpoints, and each hop is a (start, end) pair of supercell fractional
    positions, as a generated path stores them. A hop matches a path when one
    path-symmetry operation maps its start onto the path's start and its end
    onto the path's end (or, when both endpoints are one site, onto the end and
    start), and the two hops have the same nearest-image length. ``None`` marks
    a hop that matches no path.
    """

    reactant = _structure_to_atoms(reactant_structure, require_muon=True)
    product = _structure_to_atoms(product_structure, require_muon=True)
    _validate_transition_state_cells(reactant, product)
    pristine_supercell = _pristine_supercell(pristine_structure, reactant)
    operations = _path_symmetry_operations(pristine_supercell, symprec=symprec)
    if operations is None or not paths:
        return [None] * len(hops)
    rotations, translations = (np.asarray(array, dtype=float) for array in operations)
    cell = reactant.cell.array
    same_site = _orbits_coincide(
        np.asarray(
            _symmetry_equivalent_muon_positions(
                operations, _single_muon_position(reactant), tol=tol
            ),
            dtype=float,
        ),
        np.asarray(
            _symmetry_equivalent_muon_positions(
                operations, _single_muon_position(product), tol=tol
            ),
            dtype=float,
        ),
        tol=tol,
    )
    path_starts = np.asarray([path.start_fractional for path in paths], dtype=float)
    path_ends = np.asarray([path.end_fractional for path in paths], dtype=float)
    path_lengths = np.asarray([path.length_ang for path in paths], dtype=float)

    def lands_on(images: np.ndarray, targets: np.ndarray) -> np.ndarray:
        offsets = images[:, None, :] - targets[None, :, :]
        return np.all(np.abs(offsets - np.rint(offsets)) <= tol, axis=-1)

    matches: list[int | None] = []
    for start, end in hops:
        start_array = np.asarray(start, dtype=float).reshape(3)
        end_array = np.asarray(end, dtype=float).reshape(3)
        length = float(
            np.linalg.norm(
                nearest_image_delta_matrix([start_array], [end_array], cell)[0, 0] @ cell
            )
        )
        image_starts = rotations @ start_array + translations
        image_ends = rotations @ end_array + translations
        found = lands_on(image_starts, path_starts) & lands_on(image_ends, path_ends)
        if same_site:
            found |= lands_on(image_ends, path_starts) & lands_on(image_starts, path_ends)
        candidates = np.flatnonzero(
            found.any(axis=0) & (np.abs(path_lengths - length) <= max(tol, 1e-6 * length))
        )
        matches.append(int(candidates[0]) if candidates.size else None)
    return matches


def _pair_orbit_classes(
    pair_starts: np.ndarray,
    pair_ends: np.ndarray,
    *,
    start_array: np.ndarray,
    end_array: np.ndarray,
    operations: SymmetryOperations,
    tol: float,
) -> list[int]:
    """Group candidate hops by pair orbit and return each hop's class.

    Hop ``k`` runs from ``start_array[pair_starts[k]]`` to the nearest image of
    ``end_array[pair_ends[k]]``. Taking the hops in order, each hop not yet classed
    opens a class holding every hop that an operation (R, t) maps it onto: its start
    and end are the images of the hop's start and end. An operation is an isometry
    of the lattice, so it maps a nearest-image hop onto a nearest-image hop and the
    two indexes identify the image hop. When the two orbits are one orbit, the
    reversed images join the class too. A class is numbered by its first hop.
    """

    rotations, translations = (np.asarray(array, dtype=float) for array in operations)
    start_lookup = _PeriodicLookup(start_array, tol=tol)
    end_lookup = _PeriodicLookup(end_array, tol=tol)

    pair_index = np.full((len(start_array), len(end_array)), -1, dtype=int)
    pair_index[pair_starts, pair_ends] = np.arange(len(pair_starts))

    # When both orbits are one orbit, a reversed hop (end -> start) is looked up as a
    # start -> end pair through these index maps.
    end_in_start = start_lookup.indexes(end_array)
    start_in_end = end_lookup.indexes(start_array)
    same_site = _orbits_coincide(start_array, end_array, tol=tol)

    def members(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        found = (starts >= 0) & (ends >= 0)
        targets = pair_index[starts[found], ends[found]]
        return targets[targets >= 0]

    classes = np.full(len(pair_starts), -1, dtype=int)
    for pair in range(len(pair_starts)):
        if classes[pair] >= 0:
            continue
        classes[pair] = pair
        image_starts = start_lookup.indexes(
            rotations @ start_array[pair_starts[pair]] + translations
        )
        image_ends = end_lookup.indexes(rotations @ end_array[pair_ends[pair]] + translations)
        targets = members(image_starts, image_ends)
        if same_site:
            reversed_starts = np.where(image_ends >= 0, end_in_start[image_ends], -1)
            reversed_ends = np.where(image_starts >= 0, start_in_end[image_starts], -1)
            targets = np.concatenate([targets, members(reversed_starts, reversed_ends)])
        targets = targets[classes[targets] < 0]
        classes[targets] = pair
    return [int(value) for value in classes]


def _orbits_coincide(start_array: np.ndarray, end_array: np.ndarray, *, tol: float) -> bool:
    """Return whether two orbits are one orbit, so a hop between them may be reversed."""

    if len(start_array) != len(end_array):
        return False
    return bool(np.all(_PeriodicLookup(start_array, tol=tol).indexes(end_array) >= 0)) and bool(
        np.all(_PeriodicLookup(end_array, tol=tol).indexes(start_array) >= 0)
    )


class _PeriodicLookup:
    """Find positions of one orbit by fractional distance, periodic in every axis."""

    def __init__(self, orbit: np.ndarray, *, tol: float) -> None:
        self._tree = cKDTree(_wrap_unit(orbit), boxsize=1.0)
        self._tol = float(tol)

    def indexes(self, positions: np.ndarray) -> np.ndarray:
        """Return the index of the orbit member within ``tol`` of each position, or ``-1``."""

        distances, indexes = self._tree.query(
            _wrap_unit(np.asarray(positions, dtype=float).reshape(-1, 3)),
            distance_upper_bound=self._tol,
        )
        return np.where(np.isfinite(distances), indexes, -1).astype(int)


def _wrap_unit(positions: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates into ``[0, 1)``; ``np.mod`` can round a tiny negative to 1."""

    wrapped = np.mod(positions, 1.0)
    wrapped[wrapped >= 1.0] = 0.0
    return wrapped


def prepare_transition_state_structures(
    *,
    pristine_structure: Structure,
    reactant_structure: Structure,
    product_structure: Structure,
    reactant_anchor_fractional: Sequence[float],
    product_anchor_fractional: Sequence[float],
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    tol: float = 1e-4,
) -> PreparedTransitionState:
    """Align reactant/product structures to one chosen path anchor pair.

    Each endpoint is moved onto its anchor by a path-symmetry operation
    (:func:`path_symmetry_uses_magnetic_order`). Such an operation keeps the
    host's collinear spin pattern or flips it as a whole; without spin-orbit
    coupling a global flip leaves the energy unchanged, so the moved endpoint is
    still a relaxed endpoint for the reactant's spins.
    """

    reactant = _structure_to_atoms(reactant_structure, require_muon=True)
    product = _structure_to_atoms(product_structure, require_muon=True)

    _validate_transition_state_cells(reactant, product)
    pristine_supercell = _pristine_supercell(pristine_structure, reactant)

    reactant_target = np.asarray(
        tuple(float(value) for value in reactant_anchor_fractional), dtype=float
    )
    product_target = np.asarray(
        tuple(float(value) for value in product_anchor_fractional), dtype=float
    )
    if reactant_target.shape != (3,) or product_target.shape != (3,):
        raise EnergyBarrierError(
            "Transition-state anchor positions must contain exactly three fractional coordinates."
        )

    operations = _path_symmetry_operations(pristine_supercell, symprec=symprec)
    if operations is None:
        raise EnergyBarrierError(
            "Could not determine a symmetry operation for the selected transition-state path."
        )
    aligned_reactant, aligned_product = (
        _transform_structure_by_symmetry(
            pristine_supercell,
            relaxed,
            _single_muon_position(relaxed),
            target,
            operations=operations,
            symprec=symprec,
            tol=tol,
        )
        for relaxed, target in ((reactant, reactant_target), (product, product_target))
    )

    _reorder_atoms_like(aligned_reactant, aligned_product)
    _minimise_displacements(aligned_reactant, aligned_product)

    return PreparedTransitionState(
        reactant=_atoms_to_structure(
            aligned_reactant, reactant_structure, origin="energy_barriers_transition_state"
        ),
        product=_atoms_to_structure(
            aligned_product, product_structure, origin="energy_barriers_transition_state"
        ),
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
        if not destinations:
            continue
        lengths = nearest_image_distance_matrix(
            np.asarray(origin, dtype=float).reshape(1, 3),
            np.asarray(destinations, dtype=float),
            pristine_supercell.cell.array,
        )[0]
        lengths = lengths[lengths > 1e-12]
        if lengths.size:
            minima.append(float(lengths.min()))
    if not minima:
        raise EnergyBarrierError(
            "No distinct muon paths were found between the selected relaxed structures."
        )
    return 1.5 * max(minima)


def _validate_transition_state_cells(lhs: Atoms, rhs: Atoms) -> None:
    if len(lhs) != len(rhs):
        raise EnergyBarrierError(
            "Transition-state endpoints must contain the same number of atoms."
        )
    if list(lhs.numbers) != list(rhs.numbers):
        raise EnergyBarrierError(
            "Transition-state endpoints must contain atoms in matching species order."
        )
    if not np.allclose(lhs.cell.array, rhs.cell.array, atol=1e-8):
        raise EnergyBarrierError("Transition-state endpoints must share the same simulation cell.")


def _wrapped_fractional_delta(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    return ((delta + 0.5) % 1.0) - 0.5


def _structure_to_atoms(structure: Structure, *, require_muon: bool) -> Atoms:
    atoms = structure_to_ase_atoms(
        structure,
        sanitize_symbols=True,
        record_species_labels=True,
        error=EnergyBarrierError,
    )
    labels = atoms.info["mufinder_species_labels"]
    spins = structure.properties.get("atomic_spins")
    if isinstance(spins, list) and len(spins) == len(labels):
        atoms.info["mufinder_atomic_spins"] = [float(value) for value in spins]
        atoms.set_initial_magnetic_moments([float(value) for value in spins])
    if require_muon and sum(1 for label in labels if is_muon_label(label)) != 1:
        raise EnergyBarrierError(
            "Transition-state endpoint structures must contain exactly one muon."
        )
    return atoms


def _atoms_to_structure(atoms: Atoms, template: Structure, *, origin: str) -> Structure:
    labels = atoms.info.get("mufinder_species_labels")
    if not isinstance(labels, list) or len(labels) != len(atoms):
        labels = [str(symbol) for symbol in atoms.get_chemical_symbols()]
    result = Structure(
        label=template.label,
        origin=origin,
        lattice_vectors=[
            [float(component) for component in row] for row in atoms.cell.array.tolist()
        ],
        species=[str(label) for label in labels],
        fractional_positions=[
            [float(component) for component in row]
            for row in atoms.get_scaled_positions(wrap=True).tolist()
        ],
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

    return _atoms_to_structure(
        central, template=reactant, origin="energy_barriers_neb_intermediate"
    )


def _linear_midpoint_atoms(reactant_atoms: Atoms, product_atoms: Atoms) -> Atoms:
    reactant_frac = reactant_atoms.get_scaled_positions(wrap=True)
    product_frac = product_atoms.get_scaled_positions(wrap=True)
    # Each atom moves to its nearest image so the muon takes the short path across the
    # periodic boundary rather than the long way round the cell.
    midpoint_frac = reactant_frac + 0.5 * nearest_image_deltas(
        reactant_frac, product_frac, reactant_atoms.cell.array
    )
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
    distances = nearest_image_distance_matrix(
        frac[muon : muon + 1], np.delete(frac, muon, axis=0), atoms.cell.array
    )[0]
    # A muon-only cell has no host atoms to collide with -> infinite clearance.
    return float(distances.min()) if distances.size else float("inf")


def _single_muon_position(atoms: Atoms) -> np.ndarray:
    labels = atoms.info.get("mufinder_species_labels", atoms.get_chemical_symbols())
    muon_indices = [index for index, label in enumerate(labels) if is_muon_label(str(label))]
    if len(muon_indices) != 1:
        raise EnergyBarrierError(
            "Transition-state endpoint structures must contain exactly one muon."
        )
    return np.asarray(atoms.get_scaled_positions(wrap=True)[muon_indices[0]], dtype=float)


def _apply_symmetry_operation(
    positions: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return (np.asarray(positions, dtype=float) @ np.asarray(rotation, dtype=float).T) + np.asarray(
        translation, dtype=float
    )


def _symmetry_equivalent_muon_positions(
    operations: SymmetryOperations,
    muon_position: np.ndarray,
    *,
    tol: float,
) -> tuple[tuple[float, float, float], ...]:
    orbit = symmetry_orbit(muon_position, operations, lattice_vectors=None, tolerance=tol)
    return tuple((float(x), float(y), float(z)) for x, y, z in orbit)


def _crystal_operations(structure: Atoms, *, symprec: float) -> SymmetryOperations | None:
    """Return the supercell's crystal operations, from atomic numbers alone."""

    return symmetry_operations(
        (structure.cell.array, structure.get_scaled_positions(), structure.numbers),
        symprec=float(symprec),
    )


def _path_symmetry_operations(structure: Atoms, *, symprec: float) -> SymmetryOperations | None:
    """Return the operations path symmetry uses for the pristine supercell.

    These are the crystal operations, filtered by the collinear spins that
    :func:`_pristine_supercell` stores under ``_PATH_SYMMETRY_SPINS``: an
    operation is kept when it maps every atom onto one with the same spin, or
    every atom onto one with the opposite spin (a global spin flip). Without
    stored spins every crystal operation is kept.
    """

    operations = _crystal_operations(structure, symprec=symprec)
    spins = structure.info.get(_PATH_SYMMETRY_SPINS)
    if operations is None or not isinstance(spins, list) or len(spins) != len(structure):
        return operations
    values = np.asarray(spins, dtype=float)
    atol = _SPIN_MATCH_RELATIVE_TOLERANCE * float(np.max(np.abs(values)))
    positions = _wrap_unit(structure.get_scaled_positions())
    tree = cKDTree(positions, boxsize=1.0)
    rotations, translations = operations
    keep = []
    for rotation, translation in zip(rotations, translations, strict=True):
        # Crystal operations map atoms onto atoms of the same element, so the
        # nearest atom to each image is its image.
        _distances, images = tree.query(_wrap_unit(positions @ rotation.T + translation))
        mapped = values[images]
        keep.append(
            bool(np.allclose(mapped, values, rtol=0.0, atol=atol))
            or bool(np.allclose(mapped, -values, rtol=0.0, atol=atol))
        )
    mask = np.asarray(keep, dtype=bool)
    return rotations[mask], translations[mask]


def _collinear_host_spins(structure: Structure) -> list[float] | None:
    """Return the host's ``atomic_spins`` when they define a collinear magnetic order."""

    raw = structure.properties.get("atomic_spins")
    if not isinstance(raw, (list, tuple)) or len(raw) != len(structure.species):
        return None
    try:
        spins = [float(value) for value in raw]
    except (TypeError, ValueError):
        return None
    if not has_nonzero_spins(spins):
        return None
    magnetic = structure.properties.get("magnetic_structure")
    if isinstance(magnetic, dict) and not magnetic_moments_are_collinear(magnetic):
        return None
    return spins


def _pristine_supercell(pristine_structure: Structure, target: Atoms) -> Atoms:
    """Return the pristine host expanded to ``target``'s cell.

    With collinear host spins the expansion is
    :func:`mufinder.science.supercell.build_supercell_structure`, the one the
    workflows build their supercells with, so the spins follow the host's
    propagation vector; they are stored under ``_PATH_SYMMETRY_SPINS`` for
    :func:`_path_symmetry_operations`.
    """

    pristine = _structure_to_atoms(pristine_structure, require_muon=False)
    spins = _collinear_host_spins(pristine_structure)
    if spins is None:
        return _make_supercell_like(pristine, target)
    element_host = Structure(
        label=pristine_structure.label,
        lattice_vectors=pristine.cell.array.tolist(),
        species=list(pristine.get_chemical_symbols()),
        fractional_positions=pristine.get_scaled_positions(wrap=False).tolist(),
    )
    try:
        expanded, expanded_spins = build_supercell_structure(
            element_host,
            _supercell_matrix_like(pristine, target).tolist(),
            atomic_spins=spins,
            propagation_vector=list(propagation_vector_from_structure(pristine_structure)),
        )
    except ValueError as exc:
        raise EnergyBarrierError(
            "Could not construct a pristine supercell matching the relaxed endpoint cell."
        ) from exc
    supercell = _structure_to_atoms(expanded, require_muon=False)
    supercell.info[_PATH_SYMMETRY_SPINS] = [float(value) for value in expanded_spins]
    return supercell


def _supercell_matrix_like(pristine: Atoms, target: Atoms) -> np.ndarray:
    matrix = np.rint(target.cell.array @ np.linalg.inv(pristine.cell.array)).astype(int)
    if not np.allclose(matrix @ pristine.cell.array, target.cell.array, atol=1e-6):
        raise EnergyBarrierError(
            "Could not construct a pristine supercell matching the relaxed endpoint cell."
        )
    return matrix


def _make_supercell_like(pristine: Atoms, target: Atoms) -> Atoms:
    return make_supercell(pristine, _supercell_matrix_like(pristine, target))


def _operation_onto(
    operations: SymmetryOperations, current: np.ndarray, target: np.ndarray, *, tol: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the first operation mapping ``current`` onto ``target`` (periodically), or ``None``."""

    for rotation, translation in zip(*operations, strict=True):
        transformed = _apply_symmetry_operation(current.reshape(1, 3), rotation, translation)
        if np.allclose(_wrapped_fractional_delta(transformed.reshape(3), target), 0.0, atol=tol):
            return np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    return None


def _transform_structure_by_symmetry(
    pristine: Atoms,
    relaxed: Atoms,
    current_muon: np.ndarray,
    target_muon: np.ndarray,
    *,
    operations: SymmetryOperations,
    symprec: float,
    tol: float,
) -> Atoms:
    current_wrapped = np.mod(current_muon, 1.0)
    target_wrapped = np.mod(target_muon, 1.0)
    operation = _operation_onto(operations, current_wrapped, target_wrapped, tol=tol)
    if operation is None:
        crystal = _crystal_operations(pristine, symprec=symprec)
        if crystal is not None and _operation_onto(
            crystal, current_wrapped, target_wrapped, tol=tol
        ):
            raise EnergyBarrierError(
                "Selected path endpoint is related to the relaxed muon position only by "
                "operations that change the host's magnetic order; relax that site "
                "separately and choose it as the endpoint."
            )
        raise EnergyBarrierError(
            "Selected path endpoint is not symmetry-equivalent to the relaxed muon position."
        )
    rotation, translation = operation

    muon_index = _muon_index(relaxed)
    positions = relaxed.get_scaled_positions(wrap=True)
    displacements = nearest_image_deltas(
        np.broadcast_to(current_wrapped, positions.shape), positions, relaxed.cell.array
    )
    displacements[muon_index] = 0.0

    transformed = relaxed.copy()
    new_positions = np.zeros_like(positions)
    transformed_displacements = _apply_symmetry_operation(
        displacements, rotation, np.zeros(3, dtype=float)
    )
    for index in range(len(positions)):
        if index == muon_index:
            new_positions[index] = target_wrapped
            continue
        new_positions[index] = target_wrapped + transformed_displacements[index]
    transformed.set_scaled_positions(new_positions)
    return transformed


def _muon_index(atoms: Atoms) -> int:
    labels = atoms.info.get("mufinder_species_labels", atoms.get_chemical_symbols())
    matches = [index for index, label in enumerate(labels) if is_muon_label(str(label))]
    if len(matches) != 1:
        raise EnergyBarrierError(
            "Transition-state endpoint structures must contain exactly one muon."
        )
    return int(matches[0])


def _reorder_atoms_like(reference: Atoms, target: Atoms) -> None:
    reference_positions = reference.get_scaled_positions(wrap=True)
    target_positions = target.get_scaled_positions(wrap=True)
    reference_numbers = np.asarray(reference.numbers, dtype=int)
    target_numbers = np.asarray(target.numbers, dtype=int)
    penalties = nearest_image_distance_matrix(
        reference_positions, target_positions, reference.cell.array
    )
    penalties[reference_numbers[:, None] != target_numbers[None, :]] = 1e9
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
    # Each target atom moves to the image nearest its reference atom, so the muon's
    # image is the one the generated path was measured to.
    target.set_scaled_positions(
        reference_positions
        + nearest_image_deltas(reference_positions, target_positions, reference.cell.array)
    )
