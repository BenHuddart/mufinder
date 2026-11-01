"""Hierarchical and k-means clustering of relaxed muon sites over energy and position.

The features of a batch of sites are the energy and the asymmetric-unit
fractional position, found with the optional Soprano dependency
(:func:`prepare_feature_dataset`). :func:`cluster_prepared_features` replays
either method over them, and clusters are numbered by ascending average
classical energy. ``mufinder.science.relaxation_analysis`` scales the features
and drives these methods for the relaxed-site ``Clustering...`` dialog.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isnan

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
from scipy.cluster import hierarchy, vq
from scipy.spatial import distance as spdistance

from mufinder.science.ase_bridge import structure_to_ase_atoms
from mufinder.science.spglib_compat import SYMMETRY_TOLERANCE_ANG, configure_spglib_error_handling
from mufinder.science.uep import UepError, UepOptimizationResult


_KMEANS_SEED = 0


@dataclass(frozen=True, slots=True)
class FeatureClusterSummary:
    """One cluster of relaxed muon sites."""

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
class FeatureClusterSimilarity:
    """Distance between two clusters in the raw pymuon-style feature space."""

    left_cluster_index: int
    right_cluster_index: int
    distance: float


@dataclass(frozen=True, slots=True)
class FeatureClusteringSummary:
    """The clusters one hierarchical or k-means pass found."""

    threshold_ang: float
    clusters: tuple[FeatureClusterSummary, ...]
    clustering_method: str = "hier"
    clustering_hier_t: float | None = None
    clustering_kmeans_k: int | None = None
    clustering_linkage_method: str | None = None
    feature_mode: str = "soprano_defect_asymmetric_fpos"
    similarities: tuple[FeatureClusterSimilarity, ...] = ()


@dataclass(frozen=True, slots=True)
class FeatureClusteringConfig:
    """The method and its setting for one clustering pass."""

    clustering_method: str = "hier"
    clustering_hier_t: float = 0.3
    clustering_kmeans_k: int = 4
    clustering_linkage_method: str = "single"
    symprec: float = SYMMETRY_TOLERANCE_ANG
    feature_mode: str = "soprano_defect_asymmetric_fpos"


@dataclass(frozen=True, slots=True)
class FeatureClusteringResult:
    """Full output of one Soprano-backed clustering pass.

    ``cluster_labels`` gives each site its cluster's ``cluster_index`` from
    ``summary``, its rank by mean classical energy. ``kmeans_centroids`` follow that
    order: row ``i`` is the centroid of cluster ``i + 1``.
    """

    config: FeatureClusteringConfig
    cluster_labels: tuple[int, ...]
    asymmetric_fractional_positions: tuple[tuple[float, float, float], ...]
    genome_vectors: tuple[tuple[float, ...], ...]
    normalized_genome_vectors: tuple[tuple[float, ...], ...]
    summary: FeatureClusteringSummary
    linkage_matrix: tuple[tuple[float, float, float, float], ...] = ()
    kmeans_centroids: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True, slots=True)
class PreparedFeatureDataset:
    """The Soprano-derived clustering features of one batch of relaxed sites."""

    config: FeatureClusteringConfig
    optimizations: tuple[UepOptimizationResult, ...]
    asymmetric_fractional_positions: tuple[tuple[float, float, float], ...]
    genome_vectors: tuple[tuple[float, ...], ...]
    normalized_genome_vectors: tuple[tuple[float, ...], ...]
    linkage_method: str
    linkage_matrix: tuple[tuple[float, float, float, float], ...] = ()


def _import_soprano_clustering():
    try:
        from soprano.analyse.phylogen import Gene, PhylogenCluster
        from soprano.collection import AtomsCollection
        from soprano.utils import compute_asymmetric_distmat
    except ImportError as exc:  # pragma: no cover - exercised when dependency is unavailable.
        raise UepError(
            "Hierarchical and k-means clustering require the optional 'Soprano' dependency."
        ) from exc
    return Gene, PhylogenCluster, AtomsCollection, compute_asymmetric_distmat


def _clustering_energy(entry: UepOptimizationResult) -> float:
    if entry.total_energy_ev is not None and not isnan(entry.total_energy_ev):
        return float(entry.total_energy_ev)
    if not isnan(entry.classical_energy_ev):
        return float(entry.classical_energy_ev)
    raise UepError("Clustering requires a finite classical or total energy for every site.")


def _cluster_average_classical_energy(
    cluster: FeatureClusterSummary,
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
    cluster: FeatureClusterSummary,
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
    clusters: list[FeatureClusterSummary],
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
) -> tuple[tuple[FeatureClusterSummary, ...], dict[int, int]]:
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
        FeatureClusterSummary(
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
    similarities: tuple[FeatureClusterSimilarity, ...],
    cluster_index_map: dict[int, int],
) -> tuple[FeatureClusterSimilarity, ...]:
    remapped: list[FeatureClusterSimilarity] = []
    for similarity in similarities:
        left_index = cluster_index_map.get(
            int(similarity.left_cluster_index), int(similarity.left_cluster_index)
        )
        right_index = cluster_index_map.get(
            int(similarity.right_cluster_index), int(similarity.right_cluster_index)
        )
        ordered_left, ordered_right = sorted((left_index, right_index))
        remapped.append(
            FeatureClusterSimilarity(
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
        info={"name": f"site_{index + 1}"},
    )
    structure.calc = SinglePointCalculator(structure, energy=_clustering_energy(optimization))
    return structure


def _cluster_similarity_summaries(
    cluster_groups: list[np.ndarray],
    genome_vectors: np.ndarray,
) -> tuple[FeatureClusterSimilarity, ...]:
    if len(cluster_groups) < 2:
        return ()
    centres = np.array(
        [np.average(genome_vectors[group], axis=0) for group in cluster_groups], dtype=float
    )
    distances = np.linalg.norm(centres[:, None] - centres[None, :], axis=-1)
    pairs: list[FeatureClusterSimilarity] = []
    for left_index in range(len(cluster_groups) - 1):
        for right_index in range(left_index + 1, len(cluster_groups)):
            pairs.append(
                FeatureClusterSimilarity(
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


def _validate_clustering_config(config: FeatureClusteringConfig) -> FeatureClusteringConfig:
    active = config or FeatureClusteringConfig()
    method = active.clustering_method.strip().lower()
    if method not in {"hier", "kmeans"}:
        raise UepError("clustering_method must be 'hier' or 'kmeans'.")
    if active.clustering_hier_t <= 0.0:
        raise UepError("clustering_hier_t must be positive.")
    if active.clustering_kmeans_k <= 0:
        raise UepError("clustering_kmeans_k must be positive.")
    return FeatureClusteringConfig(
        clustering_method=method,
        clustering_hier_t=float(active.clustering_hier_t),
        clustering_kmeans_k=int(active.clustering_kmeans_k),
        clustering_linkage_method=str(active.clustering_linkage_method).strip().lower() or "single",
        symprec=float(active.symprec),
        feature_mode=active.feature_mode,
    )


def prepare_feature_dataset(
    optimizations: list[UepOptimizationResult] | tuple[UepOptimizationResult, ...],
    *,
    source_structure: object,
    config: FeatureClusteringConfig | None = None,
) -> PreparedFeatureDataset:
    """Return the reusable Soprano-derived clustering features of a batch of relaxed sites."""

    configure_spglib_error_handling()
    active = _validate_clustering_config(config or FeatureClusteringConfig())
    if not optimizations:
        return PreparedFeatureDataset(
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
    return PreparedFeatureDataset(
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
    prepared: PreparedFeatureDataset,
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


def dendrogram_node_clusters(
    cluster_labels: Sequence[int],
    linkage_matrix: Sequence[Sequence[float]],
) -> dict[int, int | None]:
    """Return the cluster of every dendrogram node, ``None`` for a node spanning several.

    Nodes are numbered as in SciPy: leaves ``0`` to ``n - 1`` by site index,
    then row ``i`` of the linkage matrix as node ``n + i``. A node belongs to a
    cluster when all its leaves do.
    """

    count = len(cluster_labels)
    nodes: dict[int, int | None] = {index: int(label) for index, label in enumerate(cluster_labels)}
    for row_index, row in enumerate(linkage_matrix):
        left = nodes[int(row[0])]
        right = nodes[int(row[1])]
        nodes[count + row_index] = left if left is not None and left == right else None
    return nodes


def _cluster_groups_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for cluster_id in sorted(set(int(value) for value in labels)):
        groups.append(np.where(labels == cluster_id)[0].astype(int))
    return groups


def _build_clustering_result(
    prepared: PreparedFeatureDataset,
    *,
    config: FeatureClusteringConfig,
    cluster_groups: list[np.ndarray],
    linkage_matrix: np.ndarray | None = None,
    kmeans_centroids: np.ndarray | None = None,
) -> FeatureClusteringResult:
    method = config.clustering_method
    optimizations = prepared.optimizations
    genome_vectors = np.asarray(prepared.genome_vectors, dtype=float)
    asymmetric_positions = np.asarray(prepared.asymmetric_fractional_positions, dtype=float)
    clusters: list[FeatureClusterSummary] = []
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
            FeatureClusterSummary(
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
    site_labels = [0] * len(optimizations)
    for cluster in normalized_clusters:
        for member in cluster.member_indexes:
            site_labels[member] = cluster.cluster_index
    centroids = (
        np.empty((0, 0), dtype=float)
        if kmeans_centroids is None
        else kmeans_centroids[
            [group - 1 for group in sorted(cluster_index_map, key=cluster_index_map.__getitem__)]
        ]
    )
    summary = FeatureClusteringSummary(
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
    return FeatureClusteringResult(
        config=config,
        cluster_labels=tuple(site_labels),
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
        kmeans_centroids=tuple(tuple(float(value) for value in row.tolist()) for row in centroids),
        summary=summary,
    )


def cluster_prepared_features(
    prepared: PreparedFeatureDataset,
    *,
    config: FeatureClusteringConfig | None = None,
    kmeans_point_groups: Sequence[Sequence[int]] | None = None,
) -> FeatureClusteringResult:
    """Cluster a prepared batch of relaxed sites by the configured method.

    ``kmeans_point_groups`` makes k-means cluster one point per group (the
    group's first member) and give every member its group's label, so repeated
    points neither weight the clustering nor let it pick one centroid twice.
    """

    active = _validate_clustering_config(config or prepared.config)
    method = active.clustering_method
    if not prepared.optimizations:
        return FeatureClusteringResult(
            config=active,
            cluster_labels=(),
            asymmetric_fractional_positions=(),
            genome_vectors=(),
            normalized_genome_vectors=(),
            linkage_matrix=(),
            kmeans_centroids=(),
            summary=FeatureClusteringSummary(
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
            cluster_groups = [np.array([0], dtype=int)]
            linkage_matrix = np.empty((0, 4), dtype=float)
        else:
            linkage_matrix = _linkage_matrix_for_prepared(
                prepared, method=active.clustering_linkage_method
            )
            cluster_groups = _cluster_groups_from_labels(
                hierarchy.fcluster(
                    linkage_matrix,
                    t=max(float(active.clustering_hier_t), 1e-12),
                    criterion="distance",
                ).astype(int)
            )
        return _build_clustering_result(
            prepared,
            config=active,
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
        seed=_KMEANS_SEED,
    )
    zero_based_labels, _distances = vq.vq(normalized_vectors, centroids)
    cluster_groups = [
        np.where(zero_based_labels == index)[0].astype(int)
        for index in range(int(np.max(zero_based_labels)) + 1)
    ]
    return _build_clustering_result(
        prepared,
        config=active,
        cluster_groups=cluster_groups,
        kmeans_centroids=np.asarray(centroids, dtype=float),
    )
