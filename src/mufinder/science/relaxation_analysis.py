"""Analysis helpers for completed structural muon relaxations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import match_structures
from mufinder.science.lattice import (
    minimum_image_distance,
    nearest_image_cartesian_delta,
    nearest_image_delta,
    nearest_image_distance_matrix,
)
from mufinder.science.site_generation import (
    closest_symmetry_equivalent_position,
    symmetry_equivalent_positions,
)
from mufinder.science.spglib_compat import SYMMETRY_TOLERANCE_ANG
from mufinder.science.supercell_matching import NormalizedStructurePair, normalize_for_comparison
from mufinder.science.uep import StationaryPointClassification, UepOptimizationResult
from mufinder.science.uep_clustering import (
    UepBatchSummary,
    UepClusteringConfig,
    UepClusteringResult,
    cluster_prepared_uep_dataset,
    prepare_uep_clustering_dataset,
)


class RelaxationAnalysisError(ValueError):
    """Raised when relaxation-analysis inputs or options are invalid."""


@dataclass(frozen=True, slots=True)
class RelaxedMuonSite:
    """One completed structural-relaxation endpoint used for analysis."""

    job_id: UUID
    job_label: str
    charge_state: str
    structure_id: UUID
    site_id: UUID | None
    folded_fractional_position: tuple[float, float, float]
    cartesian_position_ang: tuple[float, float, float]
    total_energy_ev: float | None
    relative_energy_ev: float | None = None


@dataclass(frozen=True, slots=True)
class RelaxationClusterSummary:
    """Summary of one cluster of relaxed muon sites."""

    cluster_index: int
    size: int
    representative_index: int
    representative_job_id: UUID
    representative_fractional_position: tuple[float, float, float]
    representative_cartesian_position_ang: tuple[float, float, float]
    minimum_relative_energy_ev: float | None
    average_relative_energy_ev: float | None
    standard_deviation_ev: float | None
    member_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ConnectedComponentsResult:
    """Result of MuFinder-style graph connected-components clustering."""

    tolerance_ang: float
    cluster_labels: tuple[int, ...]
    distance_matrix_ang: tuple[tuple[float, ...], ...]
    adjacency_matrix: tuple[tuple[int, ...], ...]
    clusters: tuple[RelaxationClusterSummary, ...]


@dataclass(frozen=True, slots=True)
class AnchoredRelaxedSite:
    """Site position after optional non-mutating symmetry anchoring."""

    site_index: int
    cluster_index: int
    original_fractional_position: tuple[float, float, float]
    shifted_fractional_position: tuple[float, float, float]
    shifted: bool


@dataclass(frozen=True, slots=True)
class DisplacementPoint:
    """One matched host-ion displacement relative to a chosen muon site."""

    species: str
    comparison_index: int
    relaxed_index: int
    distance_from_muon_ang: float
    displacement_ang: float
    radial_displacement_ang: float


@dataclass(frozen=True, slots=True)
class DisplacementAnalysisResult:
    """Matched host-ion displacement analysis for one clustered site."""

    points: tuple[DisplacementPoint, ...]
    warnings: tuple[str, ...]
    comparison_structure: Structure


def _wrap_fractional(position: Sequence[float]) -> tuple[float, float, float]:
    if len(position) != 3:
        raise RelaxationAnalysisError("Fractional positions must contain exactly three components.")
    return (float(position[0]) % 1.0, float(position[1]) % 1.0, float(position[2]) % 1.0)


def _muon_position_in_normalized_frame(
    muon_fractional_position: Sequence[float],
    supercell_matrix: Sequence[Sequence[int]],
    *,
    is_supercell: bool,
) -> tuple[float, float, float]:
    muon = _wrap_fractional(muon_fractional_position)
    if not is_supercell:
        return muon
    matrix = np.asarray(supercell_matrix, dtype=float)
    try:
        normalized = np.asarray(muon, dtype=float) @ np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return muon
    return _wrap_fractional(normalized.tolist())


def _muon_image_nearest_marker(
    muon_fractional_position: Sequence[float],
    relaxed_structure: Structure,
    normalized: NormalizedStructurePair,
) -> tuple[float, float, float] | None:
    """The image of the muon position nearest the relaxed structure's muon marker, in the supercell frame.

    ``None`` when the comparison was not tiled or the relaxed structure has no
    single muon marker.
    """

    markers = [
        position
        for species, position in zip(
            relaxed_structure.species, relaxed_structure.fractional_positions, strict=True
        )
        if str(species).strip().casefold() == "h:mu"
    ]
    if not normalized.is_supercell or len(markers) != 1:
        return None
    matrix = np.asarray(normalized.supercell_matrix, dtype=float)
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return None
    # The comparison cell's own lattice: the supercell rows are ``matrix`` times
    # its rows. The search runs in that cell's frame, since every copy of the
    # site is a comparison-cell translation of the given position.
    comparison_lattice = inverse @ np.asarray(normalized.dft.lattice_vectors, dtype=float)
    position = np.asarray(muon_fractional_position, dtype=float)
    marker = np.asarray(markers[0], dtype=float) @ matrix
    image = marker + np.asarray(nearest_image_delta(position, marker, comparison_lattice.tolist()))
    return _wrap_fractional((image @ inverse).tolist())


def _minimum_energy(sites: Sequence[RelaxedMuonSite]) -> float | None:
    finite = [
        float(site.total_energy_ev)
        for site in sites
        if site.total_energy_ev is not None and np.isfinite(float(site.total_energy_ev))
    ]
    return min(finite) if finite else None


def grouped_relative_energies(
    entries: Mapping[str, tuple[object, float | None]],
) -> dict[str, float | None]:
    """Return each entry's energy relative to the lowest energy in its group.

    ``entries`` maps a key to ``(group, total_energy_ev)``. Total energies are
    only comparable between calculations on the same cell with the same charge,
    so callers group by whatever makes two energies comparable; a key whose
    energy is missing or non-finite maps to ``None``, as does every key in a
    group with no finite energy.
    """

    minima: dict[object, float] = {}
    for group, energy in entries.values():
        if energy is None or not np.isfinite(float(energy)):
            continue
        current = minima.get(group)
        if current is None or float(energy) < current:
            minima[group] = float(energy)
    relative: dict[str, float | None] = {}
    for key, (group, energy) in entries.items():
        if energy is None or not np.isfinite(float(energy)) or group not in minima:
            relative[key] = None
        else:
            relative[key] = float(energy) - minima[group]
    return relative


def normalize_site_energies(sites: Sequence[RelaxedMuonSite]) -> tuple[RelaxedMuonSite, ...]:
    """Return sites sorted by increasing total energy with relative energies set."""

    minimum = _minimum_energy(sites)

    def key(site: RelaxedMuonSite) -> tuple[float, str]:
        energy = float(site.total_energy_ev) if site.total_energy_ev is not None else float("inf")
        return (energy, site.job_label.casefold())

    normalized: list[RelaxedMuonSite] = []
    for site in sorted(sites, key=key):
        relative = (
            None
            if minimum is None or site.total_energy_ev is None
            else float(site.total_energy_ev) - float(minimum)
        )
        normalized.append(
            RelaxedMuonSite(
                job_id=site.job_id,
                job_label=site.job_label,
                charge_state=site.charge_state,
                structure_id=site.structure_id,
                site_id=site.site_id,
                folded_fractional_position=_wrap_fractional(site.folded_fractional_position),
                cartesian_position_ang=tuple(float(value) for value in site.cartesian_position_ang),
                total_energy_ev=None
                if site.total_energy_ev is None
                else float(site.total_energy_ev),
                relative_energy_ev=relative,
            )
        )
    return tuple(normalized)


def symmetry_distance_matrix(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    *,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> tuple[tuple[float, ...], ...]:
    """Return MuFinder-paper distances between relaxed sites.

    Each distance is minimized over symmetry-equivalent positions of the second
    site and over periodic images, with the nearest image found by the lattice
    metric (:func:`~mufinder.science.lattice.nearest_image_distance_matrix`), so
    it is right in skewed cells too.
    """

    count = len(sites)
    lattice = host_structure.lattice_vectors
    matrix = np.zeros((count, count), dtype=float)
    equivalents_by_index = [
        symmetry_equivalent_positions(
            site.folded_fractional_position,
            lattice_vectors=lattice,
            host_species=host_structure.species,
            host_positions_frac=host_structure.fractional_positions,
            symprec=symprec,
        )
        for site in sites
    ]
    positions = [site.folded_fractional_position for site in sites]
    for right_index in range(1, count):
        # Every earlier site against every equivalent of this one, in one broadcast.
        equivalents = equivalents_by_index[right_index] or [positions[right_index]]
        distances = nearest_image_distance_matrix(
            positions[:right_index], equivalents, lattice
        ).min(axis=1)
        matrix[:right_index, right_index] = distances
        matrix[right_index, :right_index] = distances
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())


def _connected_components(adjacency: np.ndarray) -> list[list[int]]:
    count = int(adjacency.shape[0])
    seen = [False] * count
    components: list[list[int]] = []
    for start in range(count):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = np.flatnonzero(adjacency[node]).astype(int).tolist()
            for neighbour in neighbours:
                if seen[neighbour]:
                    continue
                seen[neighbour] = True
                stack.append(int(neighbour))
        components.append(sorted(component))
    return components


def _cluster_summaries(
    sites: Sequence[RelaxedMuonSite],
    components: Sequence[Sequence[int]],
    labels: Sequence[int],
) -> tuple[RelaxationClusterSummary, ...]:
    summaries: list[RelaxationClusterSummary] = []
    for cluster_index, members in enumerate(components, start=1):
        member_indexes = tuple(int(value) for value in members)
        if not member_indexes:
            continue
        representative_index = min(
            member_indexes,
            key=lambda idx: (
                float("inf")
                if sites[idx].total_energy_ev is None
                else float(sites[idx].total_energy_ev),
                idx,
            ),
        )
        relative = [
            float(sites[index].relative_energy_ev)
            for index in member_indexes
            if sites[index].relative_energy_ev is not None
            and np.isfinite(float(sites[index].relative_energy_ev))
        ]
        representative = sites[representative_index]
        summaries.append(
            RelaxationClusterSummary(
                cluster_index=int(labels[representative_index]),
                size=len(member_indexes),
                representative_index=representative_index,
                representative_job_id=representative.job_id,
                representative_fractional_position=representative.folded_fractional_position,
                representative_cartesian_position_ang=representative.cartesian_position_ang,
                minimum_relative_energy_ev=min(relative) if relative else None,
                average_relative_energy_ev=float(np.mean(relative)) if relative else None,
                standard_deviation_ev=float(np.std(relative)) if relative else None,
                member_indexes=member_indexes,
            )
        )
    summaries.sort(
        key=lambda summary: (
            float("inf")
            if summary.minimum_relative_energy_ev is None
            else float(summary.minimum_relative_energy_ev),
            summary.cluster_index,
        )
    )
    remap = {summary.cluster_index: index + 1 for index, summary in enumerate(summaries)}
    return tuple(
        RelaxationClusterSummary(
            cluster_index=remap[summary.cluster_index],
            size=summary.size,
            representative_index=summary.representative_index,
            representative_job_id=summary.representative_job_id,
            representative_fractional_position=summary.representative_fractional_position,
            representative_cartesian_position_ang=summary.representative_cartesian_position_ang,
            minimum_relative_energy_ev=summary.minimum_relative_energy_ev,
            average_relative_energy_ev=summary.average_relative_energy_ev,
            standard_deviation_ev=summary.standard_deviation_ev,
            member_indexes=summary.member_indexes,
        )
        for summary in summaries
    )


def _adjacency_within(distances: np.ndarray, tolerance_ang: float) -> np.ndarray:
    count = int(distances.shape[0])
    adjacency = np.zeros((count, count), dtype=int)
    if count:
        adjacency = ((distances < float(tolerance_ang)) & (~np.eye(count, dtype=bool))).astype(int)
    return adjacency


def cluster_connected_components(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    *,
    tolerance_ang: float,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> ConnectedComponentsResult:
    """Cluster relaxed sites with the graph method described in MuFinder."""

    ordered_sites = normalize_site_energies(sites)
    if tolerance_ang < 0.0:
        raise RelaxationAnalysisError("Distance tolerance must be non-negative.")
    count = len(ordered_sites)
    distances = np.asarray(
        symmetry_distance_matrix(ordered_sites, host_structure, symprec=symprec),
        dtype=float,
    )
    adjacency = _adjacency_within(distances, tolerance_ang)
    raw_components = _connected_components(adjacency)
    labels = np.zeros(count, dtype=int)
    for raw_index, members in enumerate(raw_components, start=1):
        for member in members:
            labels[int(member)] = raw_index
    summaries = _cluster_summaries(ordered_sites, raw_components, labels.tolist())
    label_remap = {
        raw_label: summary.cluster_index
        for summary in summaries
        for raw_label in {int(labels[index]) for index in summary.member_indexes}
    }
    normalized_labels = tuple(
        int(label_remap.get(int(label), int(label))) for label in labels.tolist()
    )
    return ConnectedComponentsResult(
        tolerance_ang=float(tolerance_ang),
        cluster_labels=normalized_labels,
        distance_matrix_ang=tuple(
            tuple(float(value) for value in row) for row in distances.tolist()
        ),
        adjacency_matrix=tuple(tuple(int(value) for value in row) for row in adjacency.tolist()),
        clusters=summaries,
    )


def scan_connected_component_tolerances(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    *,
    start_ang: float = 0.0,
    stop_ang: float = 1.0,
    points: int = 51,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> tuple[tuple[float, int], ...]:
    """Return cluster counts across a tolerance scan."""

    if points <= 1:
        raise RelaxationAnalysisError("Tolerance scans need at least two points.")
    start = float(start_ang)
    stop = float(stop_ang)
    if start < 0.0 or stop < 0.0 or stop < start:
        raise RelaxationAnalysisError("Tolerance scan bounds must be non-negative and increasing.")
    tolerances = np.linspace(start, stop, int(points))
    ordered_sites = normalize_site_energies(sites)
    # The symmetry distances do not depend on the tolerance, so measure them once for the whole scan.
    distances = np.asarray(
        symmetry_distance_matrix(ordered_sites, host_structure, symprec=symprec),
        dtype=float,
    )
    rows: list[tuple[float, int]] = []
    for tolerance in tolerances:
        # One cluster per connected component, exactly as ``cluster_connected_components`` counts them.
        components = _connected_components(_adjacency_within(distances, float(tolerance)))
        rows.append((float(tolerance), len(components)))
    return tuple(rows)


def anchored_sites(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    cluster_labels: Sequence[int],
    *,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
    tolerance: float = 1e-6,
) -> tuple[AnchoredRelaxedSite, ...]:
    """Return non-mutating symmetry-shifted positions clumped by cluster."""

    ordered_sites = normalize_site_energies(sites)
    if len(ordered_sites) != len(cluster_labels):
        raise RelaxationAnalysisError("Cluster labels must match the number of relaxed sites.")
    labels = [int(value) for value in cluster_labels]
    anchors: dict[int, tuple[float, float, float]] = {}
    for label in sorted(set(labels)):
        indexes = [index for index, current in enumerate(labels) if current == label]
        representative = min(
            indexes,
            key=lambda idx: (
                float("inf")
                if ordered_sites[idx].total_energy_ev is None
                else float(ordered_sites[idx].total_energy_ev),
                idx,
            ),
        )
        anchors[label] = ordered_sites[representative].folded_fractional_position

    anchored: list[AnchoredRelaxedSite] = []
    for index, site in enumerate(ordered_sites):
        label = labels[index]
        shifted = closest_symmetry_equivalent_position(
            site.folded_fractional_position,
            anchor_frac=anchors[label],
            lattice_vectors=host_structure.lattice_vectors,
            host_species=host_structure.species,
            host_positions_frac=host_structure.fractional_positions,
            symprec=symprec,
        )
        original = site.folded_fractional_position
        is_shifted = minimum_image_distance(
            original, shifted, host_structure.lattice_vectors
        ) > float(tolerance)
        anchored.append(
            AnchoredRelaxedSite(
                site_index=index,
                cluster_index=label,
                original_fractional_position=original,
                shifted_fractional_position=_wrap_fractional(shifted),
                shifted=is_shifted,
            )
        )
    return tuple(anchored)


def _fake_uep_optimizations(sites: Sequence[RelaxedMuonSite]) -> tuple[UepOptimizationResult, ...]:
    classification = StationaryPointClassification(
        label="relaxed",
        eigenvalues=(0.0, 0.0, 0.0),
        positive_count=0,
        negative_count=0,
        near_zero_count=3,
    )
    optimizations: list[UepOptimizationResult] = []
    for site in normalize_site_energies(sites):
        energy = float(site.total_energy_ev) if site.total_energy_ev is not None else 0.0
        optimizations.append(
            UepOptimizationResult(
                start_position_ang=site.cartesian_position_ang,
                optimized_position_ang=site.cartesian_position_ang,
                optimized_fractional_position=site.folded_fractional_position,
                classical_energy_ev=energy,
                zero_point_energy_ev=None,
                total_energy_ev=energy,
                success=True,
                message="completed structural relaxation",
                iterations=0,
                classification=classification,
            )
        )
    return tuple(optimizations)


#: Smallest span each relaxed-site feature column is scaled by before the
#: hierarchical and k-means methods compare sites: 0.01 eV of energy and 0.05 of
#: a fractional coordinate. Differences below these scales (numerical noise
#: between relaxations that reached the same site) never separate two sites,
#: however small the spread of the whole workspace is.
FEATURE_SPAN_FLOOR_ENERGY_EV = 0.01
FEATURE_SPAN_FLOOR_FRACTIONAL = 0.05


def _feature_span_floors(column_count: int) -> np.ndarray:
    # The features are the energy then the three asymmetric fractional coordinates.
    return np.asarray(
        [FEATURE_SPAN_FLOOR_ENERGY_EV] + [FEATURE_SPAN_FLOOR_FRACTIONAL] * (column_count - 1),
        dtype=float,
    )


def floored_feature_scaling(genome_vectors: Sequence[Sequence[float]]) -> np.ndarray:
    """Scale each relaxed-site feature column to its span, with the span floored.

    Each column is shifted to start at zero and divided by its span (max minus
    min) or by its physical floor (:data:`FEATURE_SPAN_FLOOR_ENERGY_EV`,
    :data:`FEATURE_SPAN_FLOOR_FRACTIONAL`), whichever is larger. Plain min-max
    scaling would stretch positional noise of 1e-7 over the whole [0, 1]
    range and split one site into several.
    """

    vectors = np.asarray(genome_vectors, dtype=float)
    if vectors.size == 0:
        return vectors.reshape(0, 0)
    floors = _feature_span_floors(vectors.shape[1])
    minima = vectors.min(axis=0)
    spans = np.maximum(vectors.max(axis=0) - minima, floors)
    return (vectors - minima) / spans


def distinct_feature_count(genome_vectors: Sequence[Sequence[float]]) -> int:
    """Return how many distinct sites the relaxed-site features describe.

    Two sites are the same when every feature differs by less than its
    physical floor, and sameness is chained (single linkage), so the count is
    the number of connected components of that graph. K-means never asks for
    more clusters than this, so it cannot split one site to fill ``k``.
    """

    vectors = np.asarray(genome_vectors, dtype=float)
    if vectors.size == 0:
        return 0
    scaled = vectors / _feature_span_floors(vectors.shape[1])
    differences = np.abs(scaled[:, None, :] - scaled[None, :, :]).max(axis=-1)
    adjacency = ((differences < 1.0) & ~np.eye(len(vectors), dtype=bool)).astype(int)
    return len(_connected_components(adjacency))


def cluster_soprano_style(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    *,
    method: str,
    hierarchical_t: float = 0.3,
    kmeans_k: int = 4,
    symprec: float = SYMMETRY_TOLERANCE_ANG,
) -> UepClusteringResult:
    """Cluster structural-relaxation endpoints with pymuon-suite-style features.

    The features (energy and asymmetric-unit fractional position) are those
    of the UEP clustering, scaled by :func:`floored_feature_scaling` rather
    than plain min-max, so noise between relaxations of one site never
    separates them. K-means asks for at most :func:`distinct_feature_count`
    clusters; ``summary.clustering_kmeans_k`` holds the number it used.
    """

    active = UepClusteringConfig(
        clustering_method=str(method).strip().lower(),
        clustering_hier_t=float(hierarchical_t),
        clustering_kmeans_k=int(kmeans_k),
        symprec=float(symprec),
    )
    prepared = prepare_uep_clustering_dataset(
        _fake_uep_optimizations(sites),
        source_structure=host_structure,
        config=active,
    )
    if prepared.optimizations:
        scaled = floored_feature_scaling(prepared.genome_vectors)
        # The cached linkage was built from the plain min-max features, so it is dropped.
        prepared = replace(
            prepared,
            normalized_genome_vectors=tuple(
                tuple(float(value) for value in row) for row in scaled.tolist()
            ),
            linkage_matrix=(),
        )
        if active.clustering_method == "kmeans":
            distinct = max(distinct_feature_count(prepared.genome_vectors), 1)
            active = replace(active, clustering_kmeans_k=min(int(kmeans_k), distinct))
    return cluster_prepared_uep_dataset(prepared, config=active)


def uep_summary_to_relaxation_clusters(
    sites: Sequence[RelaxedMuonSite],
    summary: UepBatchSummary,
) -> tuple[RelaxationClusterSummary, ...]:
    """Convert a UEP-style clustering summary into relaxation-analysis rows."""

    ordered_sites = normalize_site_energies(sites)
    clusters: list[RelaxationClusterSummary] = []
    for cluster in summary.clusters:
        representative = ordered_sites[int(cluster.representative_index)]
        relative = [
            float(ordered_sites[index].relative_energy_ev)
            for index in cluster.member_indexes
            if ordered_sites[index].relative_energy_ev is not None
        ]
        clusters.append(
            RelaxationClusterSummary(
                cluster_index=int(cluster.cluster_index),
                size=int(cluster.size),
                representative_index=int(cluster.representative_index),
                representative_job_id=representative.job_id,
                representative_fractional_position=representative.folded_fractional_position,
                representative_cartesian_position_ang=representative.cartesian_position_ang,
                minimum_relative_energy_ev=min(relative) if relative else None,
                average_relative_energy_ev=float(np.mean(relative)) if relative else None,
                standard_deviation_ev=float(np.std(relative)) if relative else None,
                member_indexes=tuple(int(value) for value in cluster.member_indexes),
            )
        )
    return tuple(clusters)


def muon_fractional_position_from_cell(cell: object) -> tuple[float, float, float] | None:
    """Return the last ``H:mu`` fractional coordinate in a parsed cell, unwrapped.

    ``cell`` is anything with parallel ``species`` and ``fractional_positions``
    sequences, such as a parsed relaxed ``-out.cell``. The coordinate is
    returned as written, without folding into the unit cell. ``None`` when the
    cell has no ``H:mu`` atom or the last one does not carry a three-component
    list or tuple position.
    """

    species = getattr(cell, "species", ())
    positions = getattr(cell, "fractional_positions", ())
    for atom_species, position in reversed(list(zip(species, positions, strict=False))):
        if str(atom_species).strip().casefold() != "h:mu":
            continue
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            return None
        return (float(position[0]), float(position[1]), float(position[2]))
    return None


def folded_muon_position_from_structure(structure: Structure) -> tuple[float, float, float] | None:
    """Return the last H:mu fractional coordinate in a relaxed structure."""

    for species, position in reversed(
        list(zip(structure.species, structure.fractional_positions, strict=False))
    ):
        if str(species).strip().casefold() != "h:mu":
            continue
        if isinstance(position, Sequence) and len(position) == 3:
            return _wrap_fractional(position)
    return None


def host_only_structure(structure: Structure) -> Structure:
    """Return a copy of a relaxed structure without the implanted muon atom."""

    keep = [
        index
        for index, species in enumerate(structure.species)
        if str(species).strip().casefold() != "h:mu"
    ]
    return Structure(
        id=structure.id,
        label=structure.label,
        origin=structure.origin,
        lattice_vectors=[list(row) for row in structure.lattice_vectors],
        species=[structure.species[index] for index in keep],
        fractional_positions=[list(structure.fractional_positions[index]) for index in keep],
        source_file=structure.source_file,
        parent_structure_id=structure.parent_structure_id,
        properties=dict(structure.properties),
    )


def analyze_displacements(
    *,
    relaxed_structure: Structure,
    comparison_structure: Structure,
    muon_fractional_position: Sequence[float],
) -> DisplacementAnalysisResult:
    """Match ions and compute radial displacements relative to the muon.

    The muon position is interpreted in the comparison-structure fractional
    frame. When the comparison structure is tiled into a supercell for matching,
    the muon position is converted into that normalized supercell frame before
    distances are evaluated. A position in the comparison cell names one site
    in each of the supercell's copies of that cell, so when the relaxed
    structure carries a muon marker the copy nearest the marker is used: the
    distances are measured from the site the muon relaxed to. Nearest means
    with the lattice metric (:func:`~mufinder.science.lattice.nearest_image_delta`),
    as do the ion distances and displacements below, so skewed cells are
    handled.
    """

    relaxed_host = host_only_structure(relaxed_structure)
    normalized = normalize_for_comparison(
        comparison_structure,
        relaxed_host,
    )
    matching = match_structures(normalized.experimental, normalized.dft)
    lattice = normalized.dft.lattice_vectors
    muon = _muon_image_nearest_marker(muon_fractional_position, relaxed_structure, normalized)
    if muon is None:
        muon = _muon_position_in_normalized_frame(
            muon_fractional_position,
            normalized.supercell_matrix,
            is_supercell=normalized.is_supercell,
        )
    points: list[DisplacementPoint] = []
    for pair in matching.pairs:
        comparison_frac = normalized.experimental.fractional_positions[pair.experimental_index]
        relaxed_frac = normalized.dft.fractional_positions[pair.dft_index]
        displacement_vector = np.asarray(
            nearest_image_cartesian_delta(relaxed_frac, comparison_frac, lattice), dtype=float
        )
        muon_to_comparison = np.asarray(
            nearest_image_cartesian_delta(comparison_frac, muon, lattice), dtype=float
        )
        distance = float(np.linalg.norm(muon_to_comparison))
        if distance <= 1e-12:
            radial = 0.0
        else:
            radial = float(np.dot(displacement_vector, muon_to_comparison / distance))
        points.append(
            DisplacementPoint(
                species=str(normalized.experimental.species[pair.experimental_index]),
                comparison_index=int(pair.experimental_index),
                relaxed_index=int(pair.dft_index),
                distance_from_muon_ang=distance,
                displacement_ang=float(np.linalg.norm(displacement_vector)),
                radial_displacement_ang=radial,
            )
        )
    points.sort(
        key=lambda point: (point.distance_from_muon_ang, point.species, point.comparison_index)
    )
    warnings = list(normalized.warnings)
    warnings.extend(matching.warnings)
    return DisplacementAnalysisResult(
        points=tuple(points),
        warnings=tuple(warnings),
        comparison_structure=normalized.experimental,
    )


def dataclass_to_plain(value: Any) -> Any:
    """Convert nested analysis dataclasses into JSON-safe plain values."""

    if hasattr(value, "__dataclass_fields__"):
        return {
            field: dataclass_to_plain(getattr(value, field))
            for field in value.__dataclass_fields__  # type: ignore[attr-defined]
        }
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): dataclass_to_plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [dataclass_to_plain(item) for item in value]
    return value


def parse_tolerance_scan(payload: object) -> tuple[tuple[float, int], ...]:
    """Parse persisted tolerance-scan rows defensively."""

    rows: list[tuple[float, int]] = []
    if not isinstance(payload, Iterable) or isinstance(payload, (str, bytes, Mapping)):
        return ()
    for entry in payload:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        try:
            rows.append((float(entry[0]), int(entry[1])))
        except (TypeError, ValueError):
            continue
    rows.sort(key=lambda item: item[0])
    deduped: dict[float, int] = {}
    for tolerance, count in rows:
        deduped[round(float(tolerance), 10)] = int(count)
    return tuple((float(tolerance), int(count)) for tolerance, count in sorted(deduped.items()))
