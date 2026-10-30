"""Clustering of optimized UEP muon positions into candidate stopping sites.

Two clusterers group the endpoints of ``mufinder.science.uep`` optimizations:
a legacy hierarchical clustering in Cartesian space
(``cluster_optimized_positions``) and the pymuon-suite-compatible clustering
built on the optional Soprano dependency, which first prepares reusable
features for a batch (``prepare_uep_clustering_dataset``) and then replays
hierarchical or k-means clustering over them
(``cluster_prepared_uep_dataset``). Clusters are numbered by ascending average
classical energy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isnan

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from scipy.cluster import hierarchy, vq
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import distance as spdistance

from mufinder.science.ase_bridge import structure_to_ase_atoms
from mufinder.science.spglib_compat import SYMMETRY_TOLERANCE_ANG, configure_spglib_error_handling
from mufinder.science.uep import UepError, UepOptimizationResult


_UEP_KMEANS_SEED = 0


@dataclass(frozen=True, slots=True)
class UepClusterSummary:
    """One cluster of optimized UEP stopping-site candidates."""

    cluster_index: int
    size: int
    representative_index: int
    representative_position_ang: tuple[float, float, float]
    representative_fractional_position: tuple[float, float, float]
    minimum_energy_ev: float | None
    average_energy_ev: float | None
    standard_deviation_ev: float | None
    member_indexes: tuple[int, ...]
    representative_asymmetric_fractional_position: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class UepClusterSimilaritySummary:
    """Distance between two clusters in the raw pymuon-style feature space."""

    left_cluster_index: int
    right_cluster_index: int
    distance: float


@dataclass(frozen=True, slots=True)
class UepBatchSummary:
    """Clustered summary of a batch UEP search."""

    threshold_ang: float
    clusters: tuple[UepClusterSummary, ...]
    clustering_method: str = "legacy_cartesian_hier"
    clustering_hier_t: float | None = None
    clustering_kmeans_k: int | None = None
    clustering_linkage_method: str | None = None
    feature_mode: str = "legacy_cartesian_positions"
    similarities: tuple[UepClusterSimilaritySummary, ...] = ()


@dataclass(frozen=True, slots=True)
class UepClusteringConfig:
    """Configuration for pymuon-suite-style clustering of optimized sites."""

    clustering_method: str = "hier"
    clustering_hier_t: float = 0.3
    clustering_kmeans_k: int = 4
    clustering_linkage_method: str = "single"
    symprec: float = SYMMETRY_TOLERANCE_ANG
    feature_mode: str = "soprano_defect_asymmetric_fpos"


@dataclass(frozen=True, slots=True)
class UepClusteringResult:
    """Full output of one Soprano-backed clustering pass."""

    config: UepClusteringConfig
    cluster_labels: tuple[int, ...]
    asymmetric_fractional_positions: tuple[tuple[float, float, float], ...]
    genome_vectors: tuple[tuple[float, ...], ...]
    normalized_genome_vectors: tuple[tuple[float, ...], ...]
    summary: UepBatchSummary
    linkage_matrix: tuple[tuple[float, float, float, float], ...] = ()
    kmeans_centroids: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class UepPreparedClusteringDataset:
    """Cached Soprano-derived clustering features for one relaxed-site batch."""

    config: UepClusteringConfig
    optimizations: tuple[UepOptimizationResult, ...]
    asymmetric_fractional_positions: tuple[tuple[float, float, float], ...]
    genome_vectors: tuple[tuple[float, ...], ...]
    normalized_genome_vectors: tuple[tuple[float, ...], ...]
    linkage_method: str
    linkage_matrix: tuple[tuple[float, float, float, float], ...] = ()


def cluster_optimized_positions(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
    *,
    threshold_ang: float = 0.2,
) -> UepBatchSummary:
    """Cluster optimized UEP endpoints in Cartesian space."""

    if threshold_ang <= 0.0:
        raise UepError("threshold_ang must be positive.")
    if not optimizations:
        return UepBatchSummary(
            threshold_ang=float(threshold_ang),
            clusters=(),
            clustering_method="legacy_cartesian_hier",
            clustering_hier_t=float(threshold_ang),
            clustering_linkage_method="average",
            feature_mode="legacy_cartesian_positions",
        )

    positions = np.array([entry.optimized_position_ang for entry in optimizations], dtype=float)
    if len(optimizations) == 1:
        energy = _clustering_energy(optimizations[0])
        summary = UepClusterSummary(
            cluster_index=1,
            size=1,
            representative_index=0,
            representative_position_ang=optimizations[0].optimized_position_ang,
            representative_fractional_position=optimizations[0].optimized_fractional_position,
            minimum_energy_ev=energy,
            average_energy_ev=energy,
            standard_deviation_ev=0.0,
            member_indexes=(0,),
        )
        return UepBatchSummary(
            threshold_ang=float(threshold_ang),
            clusters=(summary,),
            clustering_method="legacy_cartesian_hier",
            clustering_hier_t=float(threshold_ang),
            clustering_linkage_method="average",
            feature_mode="legacy_cartesian_positions",
        )

    linkage_matrix = linkage(positions, method="average")
    labels = fcluster(linkage_matrix, t=float(threshold_ang), criterion="distance")
    cluster_ids = sorted(set(int(value) for value in labels))

    clusters: list[UepClusterSummary] = []
    for cluster_index, cluster_id in enumerate(cluster_ids, start=1):
        member_indexes = tuple(
            index for index, label in enumerate(labels) if int(label) == cluster_id
        )
        members = [optimizations[index] for index in member_indexes]
        numeric_energies = [_clustering_energy(entry) for entry in members]
        representative_offset = min(
            range(len(members)),
            key=lambda member_index: (
                _clustering_energy(members[member_index]),
                member_index,
            ),
        )
        representative_index = member_indexes[representative_offset]
        representative = optimizations[representative_index]
        minimum = min(numeric_energies) if numeric_energies else None
        average = float(np.mean(numeric_energies)) if numeric_energies else None
        deviation = float(np.std(numeric_energies)) if numeric_energies else None
        clusters.append(
            UepClusterSummary(
                cluster_index=cluster_index,
                size=len(member_indexes),
                representative_index=representative_index,
                representative_position_ang=representative.optimized_position_ang,
                representative_fractional_position=representative.optimized_fractional_position,
                minimum_energy_ev=minimum,
                average_energy_ev=average,
                standard_deviation_ev=deviation,
                member_indexes=member_indexes,
            )
        )

    normalized, _cluster_index_map = _renumber_cluster_summaries(clusters, optimizations)
    return UepBatchSummary(
        threshold_ang=float(threshold_ang),
        clusters=normalized,
        clustering_method="legacy_cartesian_hier",
        clustering_hier_t=float(threshold_ang),
        clustering_linkage_method="average",
        feature_mode="legacy_cartesian_positions",
    )


def _import_soprano_clustering():
    try:
        from soprano.analyse.phylogen import Gene, PhylogenCluster
        from soprano.collection import AtomsCollection
        from soprano.utils import compute_asymmetric_distmat
    except ImportError as exc:  # pragma: no cover - exercised when dependency is unavailable.
        raise UepError(
            "pymuon-suite-compatible UEP clustering requires the optional 'Soprano' dependency."
        ) from exc
    return Gene, PhylogenCluster, AtomsCollection, compute_asymmetric_distmat


def _clustering_energy(entry: UepOptimizationResult) -> float:
    if entry.total_energy_ev is not None and not isnan(entry.total_energy_ev):
        return float(entry.total_energy_ev)
    if not isnan(entry.classical_energy_ev):
        return float(entry.classical_energy_ev)
    raise UepError(
        "UEP clustering requires a finite classical or total energy for every optimization."
    )


def relaxed_site_number_map(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> dict[int, int]:
    """Return 1-based relaxed-site numbers ordered globally by classical energy."""

    ordered_indexes = sorted(
        range(len(optimizations)),
        key=lambda idx: (
            float(optimizations[idx].classical_energy_ev)
            if not isnan(optimizations[idx].classical_energy_ev)
            else float("inf"),
            int(idx),
        ),
    )
    return {
        int(result_index): int(site_number)
        for site_number, result_index in enumerate(ordered_indexes, start=1)
    }


def _cluster_average_classical_energy(
    cluster: UepClusterSummary,
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> float:
    classical_energies = [
        float(optimizations[int(index)].classical_energy_ev)
        for index in cluster.member_indexes
        if not isnan(optimizations[int(index)].classical_energy_ev)
    ]
    if not classical_energies:
        return float("inf")
    return float(np.mean(classical_energies))


def _cluster_minimum_classical_energy(
    cluster: UepClusterSummary,
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> float:
    classical_energies = [
        float(optimizations[int(index)].classical_energy_ev)
        for index in cluster.member_indexes
        if not isnan(optimizations[int(index)].classical_energy_ev)
    ]
    if not classical_energies:
        return float("inf")
    return min(classical_energies)


def _renumber_cluster_summaries(
    clusters: list[UepClusterSummary],
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> tuple[tuple[UepClusterSummary, ...], dict[int, int]]:
    ordered = sorted(
        clusters,
        key=lambda cluster: (
            _cluster_average_classical_energy(cluster, optimizations),
            _cluster_minimum_classical_energy(cluster, optimizations),
            int(cluster.cluster_index),
        ),
    )
    cluster_index_map = {
        int(cluster.cluster_index): index + 1 for index, cluster in enumerate(ordered)
    }
    normalized = tuple(
        UepClusterSummary(
            cluster_index=index + 1,
            size=cluster.size,
            representative_index=cluster.representative_index,
            representative_position_ang=cluster.representative_position_ang,
            representative_fractional_position=cluster.representative_fractional_position,
            minimum_energy_ev=cluster.minimum_energy_ev,
            average_energy_ev=cluster.average_energy_ev,
            standard_deviation_ev=cluster.standard_deviation_ev,
            member_indexes=cluster.member_indexes,
            representative_asymmetric_fractional_position=cluster.representative_asymmetric_fractional_position,
        )
        for index, cluster in enumerate(ordered)
    )
    return normalized, cluster_index_map


def _renumber_cluster_similarity_summaries(
    similarities: tuple[UepClusterSimilaritySummary, ...],
    cluster_index_map: dict[int, int],
) -> tuple[UepClusterSimilaritySummary, ...]:
    remapped: list[UepClusterSimilaritySummary] = []
    for similarity in similarities:
        left_index = cluster_index_map.get(
            int(similarity.left_cluster_index), int(similarity.left_cluster_index)
        )
        right_index = cluster_index_map.get(
            int(similarity.right_cluster_index), int(similarity.right_cluster_index)
        )
        ordered_left, ordered_right = sorted((left_index, right_index))
        remapped.append(
            UepClusterSimilaritySummary(
                left_cluster_index=int(ordered_left),
                right_cluster_index=int(ordered_right),
                distance=float(similarity.distance),
            )
        )
    return tuple(
        sorted(
            remapped,
            key=lambda item: (item.distance, item.left_cluster_index, item.right_cluster_index),
        )
    )


def _asymmetric_images_centre(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> np.ndarray:
    reference_index = min(
        range(len(optimizations)), key=lambda idx: (_clustering_energy(optimizations[idx]), idx)
    )
    reference = np.asarray(
        optimizations[reference_index].optimized_fractional_position, dtype=float
    )
    # Avoid high-symmetry centres, which Soprano warns can produce unstable image selection.
    jitter = np.array((1.61803398875e-6, 2.41421356237e-6, 3.14159265359e-6), dtype=float)
    return (reference + jitter) % 1.0


def _muonated_atoms_for_clustering(
    host_atoms: Atoms,
    optimization: UepOptimizationResult,
    *,
    index: int,
) -> Atoms:
    scaled_positions = np.vstack(
        [
            host_atoms.get_scaled_positions(wrap=True),
            np.asarray(optimization.optimized_fractional_position, dtype=float),
        ]
    )
    structure = Atoms(
        host_atoms.get_chemical_symbols() + ["H"],
        scaled_positions=scaled_positions,
        cell=host_atoms.cell.copy(),
        pbc=True,
        info={"name": f"uep_{index + 1}"},
    )
    structure.calc = SinglePointCalculator(structure, energy=_clustering_energy(optimization))
    return structure


def _cluster_similarity_summaries(
    cluster_groups: list[np.ndarray],
    genome_vectors: np.ndarray,
) -> tuple[UepClusterSimilaritySummary, ...]:
    if len(cluster_groups) < 2:
        return ()
    centres = np.array(
        [np.average(genome_vectors[group], axis=0) for group in cluster_groups], dtype=float
    )
    distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)
    pairs: list[UepClusterSimilaritySummary] = []
    for left_index in range(len(cluster_groups) - 1):
        for right_index in range(left_index + 1, len(cluster_groups)):
            pairs.append(
                UepClusterSimilaritySummary(
                    left_cluster_index=left_index + 1,
                    right_cluster_index=right_index + 1,
                    distance=float(distances[left_index, right_index]),
                )
            )
    return tuple(
        sorted(
            pairs,
            key=lambda item: (item.distance, item.left_cluster_index, item.right_cluster_index),
        )
    )


def _validate_clustering_config(config: UepClusteringConfig) -> UepClusteringConfig:
    active = config or UepClusteringConfig()
    method = active.clustering_method.strip().lower()
    if method not in {"hier", "kmeans"}:
        raise UepError("clustering_method must be 'hier' or 'kmeans'.")
    if active.clustering_hier_t <= 0.0:
        raise UepError("clustering_hier_t must be positive.")
    if active.clustering_kmeans_k <= 0:
        raise UepError("clustering_kmeans_k must be positive.")
    return UepClusteringConfig(
        clustering_method=method,
        clustering_hier_t=float(active.clustering_hier_t),
        clustering_kmeans_k=int(active.clustering_kmeans_k),
        clustering_linkage_method=str(active.clustering_linkage_method).strip().lower() or "single",
        symprec=float(active.symprec),
        feature_mode=active.feature_mode,
    )


def prepare_uep_clustering_dataset(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
    *,
    source_structure: object,
    config: UepClusteringConfig | None = None,
) -> UepPreparedClusteringDataset:
    """Prepare reusable Soprano-derived clustering features for a relaxed-site batch."""

    configure_spglib_error_handling()
    active = _validate_clustering_config(config or UepClusteringConfig())
    if not optimizations:
        return UepPreparedClusteringDataset(
            config=active,
            optimizations=(),
            asymmetric_fractional_positions=(),
            genome_vectors=(),
            normalized_genome_vectors=(),
            linkage_method=active.clustering_linkage_method,
            linkage_matrix=(),
        )

    Gene, PhylogenCluster, AtomsCollection, compute_asymmetric_distmat = (
        _import_soprano_clustering()
    )
    host_atoms = structure_to_ase_atoms(source_structure, error=UepError)
    collection = AtomsCollection(
        [
            _muonated_atoms_for_clustering(host_atoms, optimization, index=index)
            for index, optimization in enumerate(optimizations)
        ]
    )
    images_centre = _asymmetric_images_centre(optimizations)

    def _parse_asymmetric_fpos(coll):
        points = coll.all.get_scaled_positions()[:, -1, :]
        _, images = compute_asymmetric_distmat(
            host_atoms,
            points,
            return_images=True,
            images_centre=images_centre,
            symprec=active.symprec,
        )
        return images

    genes = [
        Gene("energy", 1.0, {}),
        Gene("mufinder_defect_asymmetric_fpos", 1.0, parser=_parse_asymmetric_fpos),
    ]
    clusterer = PhylogenCluster(collection, genes=genes)
    genome_vectors, _ = clusterer.get_genome_vectors()
    normalized_vectors, _ = clusterer.get_genome_vectors_norm()
    linkage_matrix = np.empty((0, 4), dtype=float)
    if len(optimizations) > 1:
        linkage_matrix = np.asarray(
            clusterer.get_linkage(method=active.clustering_linkage_method),
            dtype=float,
        )
    return UepPreparedClusteringDataset(
        config=active,
        optimizations=tuple(optimizations),
        asymmetric_fractional_positions=tuple(
            tuple(float(value) for value in row.tolist())
            for row in np.asarray(genome_vectors[:, 1:4], dtype=float)
        ),
        genome_vectors=tuple(
            tuple(float(value) for value in row.tolist())
            for row in np.asarray(genome_vectors, dtype=float)
        ),
        normalized_genome_vectors=tuple(
            tuple(float(value) for value in row.tolist())
            for row in np.asarray(normalized_vectors, dtype=float)
        ),
        linkage_method=active.clustering_linkage_method,
        linkage_matrix=tuple(
            tuple(float(value) for value in row.tolist())
            for row in np.asarray(linkage_matrix, dtype=float)
        ),
    )


def _linkage_matrix_for_prepared(
    prepared: UepPreparedClusteringDataset,
    *,
    method: str,
) -> np.ndarray:
    if len(prepared.optimizations) <= 1:
        return np.empty((0, 4), dtype=float)
    if method == prepared.linkage_method and prepared.linkage_matrix:
        return np.asarray(prepared.linkage_matrix, dtype=float)
    normalized_vectors = np.asarray(prepared.normalized_genome_vectors, dtype=float)
    condensed = spdistance.pdist(normalized_vectors, metric="euclidean")
    return np.asarray(hierarchy.linkage(condensed, method=method), dtype=float)


def _cluster_groups_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for cluster_id in sorted(set(int(value) for value in labels)):
        groups.append(np.where(labels == cluster_id)[0].astype(int))
    return groups


def _build_clustering_result(
    prepared: UepPreparedClusteringDataset,
    *,
    config: UepClusteringConfig,
    cluster_labels: np.ndarray,
    cluster_groups: list[np.ndarray],
    linkage_matrix: np.ndarray | None = None,
    kmeans_centroids: np.ndarray | None = None,
) -> UepClusteringResult:
    method = config.clustering_method
    optimizations = prepared.optimizations
    genome_vectors = np.asarray(prepared.genome_vectors, dtype=float)
    asymmetric_positions = np.asarray(prepared.asymmetric_fractional_positions, dtype=float)
    clusters: list[UepClusterSummary] = []
    for cluster_index, members in enumerate(cluster_groups, start=1):
        group = np.asarray(members, dtype=int)
        if group.size == 0:
            continue
        member_indexes = tuple(int(value) for value in group.tolist())
        cluster_members = [optimizations[index] for index in member_indexes]
        finite_total_energies = [
            float(member.total_energy_ev)
            for member in cluster_members
            if member.total_energy_ev is not None and not isnan(member.total_energy_ev)
        ]
        representative_index = min(
            member_indexes, key=lambda idx: (_clustering_energy(optimizations[idx]), idx)
        )
        representative = optimizations[representative_index]
        clusters.append(
            UepClusterSummary(
                cluster_index=cluster_index,
                size=len(member_indexes),
                representative_index=representative_index,
                representative_position_ang=representative.optimized_position_ang,
                representative_fractional_position=representative.optimized_fractional_position,
                minimum_energy_ev=min(finite_total_energies) if finite_total_energies else None,
                average_energy_ev=float(np.mean(finite_total_energies))
                if finite_total_energies
                else None,
                standard_deviation_ev=float(np.std(finite_total_energies))
                if finite_total_energies
                else None,
                member_indexes=member_indexes,
                representative_asymmetric_fractional_position=tuple(
                    float(value) for value in asymmetric_positions[representative_index].tolist()
                ),
            )
        )

    normalized_clusters, cluster_index_map = _renumber_cluster_summaries(clusters, optimizations)
    summary = UepBatchSummary(
        threshold_ang=float(config.clustering_hier_t if method == "hier" else 0.0),
        clusters=normalized_clusters,
        clustering_method=method,
        clustering_hier_t=float(config.clustering_hier_t),
        clustering_kmeans_k=(int(len(cluster_groups)) if method == "kmeans" else None),
        clustering_linkage_method=config.clustering_linkage_method if method == "hier" else None,
        feature_mode=config.feature_mode,
        similarities=_renumber_cluster_similarity_summaries(
            _cluster_similarity_summaries(cluster_groups, genome_vectors),
            cluster_index_map,
        ),
    )
    return UepClusteringResult(
        config=config,
        cluster_labels=tuple(
            int(value) for value in np.asarray(cluster_labels, dtype=int).tolist()
        ),
        asymmetric_fractional_positions=prepared.asymmetric_fractional_positions,
        genome_vectors=prepared.genome_vectors,
        normalized_genome_vectors=prepared.normalized_genome_vectors,
        linkage_matrix=tuple(
            tuple(float(value) for value in row.tolist())
            for row in np.asarray(
                np.empty((0, 4), dtype=float) if linkage_matrix is None else linkage_matrix,
                dtype=float,
            )
        ),
        kmeans_centroids=tuple(
            tuple(float(value) for value in row.tolist())
            for row in np.asarray(
                np.empty((0, 0), dtype=float) if kmeans_centroids is None else kmeans_centroids,
                dtype=float,
            )
        ),
        summary=summary,
    )


def cluster_prepared_uep_dataset(
    prepared: UepPreparedClusteringDataset,
    *,
    config: UepClusteringConfig | None = None,
    kmeans_point_groups: Sequence[Sequence[int]] | None = None,
) -> UepClusteringResult:
    """Replay pymuon-suite-style clustering over a prepared relaxed-site dataset.

    ``kmeans_point_groups`` makes k-means cluster one point per group (the
    group's first member) and give every member its group's label, so repeated
    points neither weight the clustering nor let it pick one centroid twice.
    """

    active = _validate_clustering_config(config or prepared.config)
    method = active.clustering_method
    if not prepared.optimizations:
        return UepClusteringResult(
            config=active,
            cluster_labels=(),
            asymmetric_fractional_positions=(),
            genome_vectors=(),
            normalized_genome_vectors=(),
            linkage_matrix=(),
            kmeans_centroids=(),
            summary=UepBatchSummary(
                threshold_ang=float(active.clustering_hier_t if method == "hier" else 0.0),
                clusters=(),
                clustering_method=method,
                clustering_hier_t=float(active.clustering_hier_t),
                clustering_kmeans_k=(
                    int(active.clustering_kmeans_k) if method == "kmeans" else None
                ),
                clustering_linkage_method=active.clustering_linkage_method
                if method == "hier"
                else None,
                feature_mode=active.feature_mode,
                similarities=(),
            ),
        )

    if method == "hier":
        if len(prepared.optimizations) == 1:
            cluster_labels = np.array([1], dtype=int)
            cluster_groups = [np.array([0], dtype=int)]
            linkage_matrix = np.empty((0, 4), dtype=float)
        else:
            linkage_matrix = _linkage_matrix_for_prepared(
                prepared, method=active.clustering_linkage_method
            )
            cluster_labels = hierarchy.fcluster(
                linkage_matrix,
                t=max(float(active.clustering_hier_t), 1e-12),
                criterion="distance",
            ).astype(int)
            cluster_groups = _cluster_groups_from_labels(cluster_labels)
        return _build_clustering_result(
            prepared,
            config=active,
            cluster_labels=cluster_labels,
            cluster_groups=cluster_groups,
            linkage_matrix=linkage_matrix,
        )

    cluster_count = int(active.clustering_kmeans_k)
    if cluster_count > len(prepared.optimizations):
        raise UepError("clustering_kmeans_k must not exceed the number of optimized sites.")
    normalized_vectors = np.asarray(prepared.normalized_genome_vectors, dtype=float)
    points = (
        normalized_vectors
        if kmeans_point_groups is None
        else normalized_vectors[[int(group[0]) for group in kmeans_point_groups]]
    )
    centroids, _distortion = vq.kmeans(
        points,
        cluster_count,
        seed=_UEP_KMEANS_SEED,
    )
    zero_based_labels, _distances = vq.vq(normalized_vectors, centroids)
    cluster_labels = zero_based_labels.astype(int) + 1
    cluster_groups = [
        np.where(zero_based_labels == index)[0].astype(int)
        for index in range(int(np.max(zero_based_labels)) + 1)
    ]
    return _build_clustering_result(
        prepared,
        config=active,
        cluster_labels=cluster_labels,
        cluster_groups=cluster_groups,
        kmeans_centroids=np.asarray(centroids, dtype=float),
    )


def cluster_uep_optimizations(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
    *,
    source_structure: object,
    config: UepClusteringConfig | None = None,
) -> UepClusteringResult:
    """Cluster optimized UEP endpoints using the Soprano machinery used by pymuon-suite."""

    prepared = prepare_uep_clustering_dataset(
        optimizations,
        source_structure=source_structure,
        config=config,
    )
    return cluster_prepared_uep_dataset(prepared, config=config)
