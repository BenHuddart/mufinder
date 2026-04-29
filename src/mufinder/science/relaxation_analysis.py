"""Analysis helpers for completed structural muon relaxations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.atom_matching import match_structures
from mufinder.science.site_generation import (
    closest_symmetry_equivalent_position,
    lattice_frac_to_cart,
    symmetry_equivalent_positions,
)
from mufinder.science.supercell_matching import normalize_for_comparison
from mufinder.science.uep import (
    StationaryPointClassification,
    UepBatchSummary,
    UepClusteringConfig,
    UepClusteringResult,
    UepOptimizationResult,
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


def _fractional_delta_minimum_image(
    left: Sequence[float],
    right: Sequence[float],
) -> np.ndarray:
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    return delta - np.round(delta)


def _pbc_distance(
    left: Sequence[float],
    right: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
) -> float:
    delta = _fractional_delta_minimum_image(left, right)
    cart = np.asarray(lattice_frac_to_cart(delta.tolist(), lattice_vectors), dtype=float)
    return float(np.linalg.norm(cart))


def _pbc_delta_cartesian(
    left: Sequence[float],
    right: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
) -> np.ndarray:
    delta = _fractional_delta_minimum_image(left, right)
    return np.asarray(lattice_frac_to_cart(delta.tolist(), lattice_vectors), dtype=float)


def _minimum_energy(sites: Sequence[RelaxedMuonSite]) -> float | None:
    finite = [
        float(site.total_energy_ev)
        for site in sites
        if site.total_energy_ev is not None and np.isfinite(float(site.total_energy_ev))
    ]
    return min(finite) if finite else None


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
                total_energy_ev=None if site.total_energy_ev is None else float(site.total_energy_ev),
                relative_energy_ev=relative,
            )
        )
    return tuple(normalized)


def symmetry_distance_matrix(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    *,
    symprec: float = 1e-5,
) -> tuple[tuple[float, ...], ...]:
    """Return MuFinder-paper distances between relaxed sites.

    Each distance is minimized over symmetry-equivalent positions of the second
    site, with periodic minimum-image translations included in the metric.
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
    for left_index, left in enumerate(sites):
        for right_index in range(left_index + 1, count):
            distances = [
                _pbc_distance(left.folded_fractional_position, equivalent, lattice)
                for equivalent in equivalents_by_index[right_index]
            ]
            distance = min(distances) if distances else _pbc_distance(
                left.folded_fractional_position,
                sites[right_index].folded_fractional_position,
                lattice,
            )
            matrix[left_index, right_index] = distance
            matrix[right_index, left_index] = distance
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
                float("inf") if sites[idx].total_energy_ev is None else float(sites[idx].total_energy_ev),
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


def cluster_connected_components(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    *,
    tolerance_ang: float,
    symprec: float = 1e-5,
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
    adjacency = np.zeros((count, count), dtype=int)
    if count:
        adjacency = ((distances < float(tolerance_ang)) & (~np.eye(count, dtype=bool))).astype(int)
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
    normalized_labels = tuple(int(label_remap.get(int(label), int(label))) for label in labels.tolist())
    return ConnectedComponentsResult(
        tolerance_ang=float(tolerance_ang),
        cluster_labels=normalized_labels,
        distance_matrix_ang=tuple(tuple(float(value) for value in row) for row in distances.tolist()),
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
    symprec: float = 1e-5,
) -> tuple[tuple[float, int], ...]:
    """Return cluster counts across a tolerance scan."""

    if points <= 1:
        raise RelaxationAnalysisError("Tolerance scans need at least two points.")
    start = float(start_ang)
    stop = float(stop_ang)
    if start < 0.0 or stop < 0.0 or stop < start:
        raise RelaxationAnalysisError("Tolerance scan bounds must be non-negative and increasing.")
    tolerances = np.linspace(start, stop, int(points))
    rows: list[tuple[float, int]] = []
    for tolerance in tolerances:
        result = cluster_connected_components(
            sites,
            host_structure,
            tolerance_ang=float(tolerance),
            symprec=symprec,
        )
        rows.append((float(tolerance), len(result.clusters)))
    return tuple(rows)


def anchored_sites(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    cluster_labels: Sequence[int],
    *,
    symprec: float = 1e-5,
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
                float("inf") if ordered_sites[idx].total_energy_ev is None else float(ordered_sites[idx].total_energy_ev),
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
        is_shifted = _pbc_distance(original, shifted, host_structure.lattice_vectors) > float(tolerance)
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


def cluster_soprano_style(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    *,
    method: str,
    hierarchical_t: float = 0.3,
    kmeans_k: int = 4,
    symprec: float = 1e-5,
) -> UepClusteringResult:
    """Cluster structural-relaxation endpoints with pymuon-suite-style features."""

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


def folded_muon_position_from_structure(structure: Structure) -> tuple[float, float, float] | None:
    """Return the last H:mu fractional coordinate in a relaxed structure."""

    for species, position in reversed(list(zip(structure.species, structure.fractional_positions, strict=False))):
        if str(species).strip().casefold() != "h:mu":
            continue
        if isinstance(position, Sequence) and len(position) == 3:
            return _wrap_fractional(position)
    return None


def host_only_structure(structure: Structure) -> Structure:
    """Return a copy of a relaxed structure without the implanted muon atom."""

    keep = [index for index, species in enumerate(structure.species) if str(species).strip().casefold() != "h:mu"]
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
    """Match ions and compute radial displacements relative to the muon."""

    relaxed_host = host_only_structure(relaxed_structure)
    normalized = normalize_for_comparison(
        comparison_structure,
        relaxed_host,
    )
    matching = match_structures(normalized.experimental, normalized.dft)
    lattice = normalized.dft.lattice_vectors
    muon = _wrap_fractional(muon_fractional_position)
    points: list[DisplacementPoint] = []
    for pair in matching.pairs:
        comparison_frac = normalized.experimental.fractional_positions[pair.experimental_index]
        relaxed_frac = normalized.dft.fractional_positions[pair.dft_index]
        displacement_vector = _pbc_delta_cartesian(comparison_frac, relaxed_frac, lattice)
        muon_to_comparison = _pbc_delta_cartesian(muon, comparison_frac, lattice)
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
    points.sort(key=lambda point: (point.distance_from_muon_ang, point.species, point.comparison_index))
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
