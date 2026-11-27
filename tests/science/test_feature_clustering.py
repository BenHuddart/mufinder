"""Tests for hierarchical and k-means clustering over the relaxed-site features."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mufinder.science import feature_clustering as feature_clustering_module
from mufinder.science.feature_clustering import (
    FeatureClusteringConfig,
    FeatureClusteringResult,
    PreparedFeatureDataset,
    cluster_prepared_features,
    dendrogram_node_clusters,
    prepare_feature_dataset,
)
from mufinder.science.uep import UepError, UepOptimizationResult, UepStructure

from tests.support.uep_results import fake_optimization as _fake_optimization

_CUBE_LATTICE = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))

_soprano_required = pytest.mark.skipif(
    importlib.util.find_spec("soprano") is None,
    reason="soprano optional dependency not installed",
)


def _simple_cubic_host() -> UepStructure:
    return UepStructure(
        lattice_vectors=_CUBE_LATTICE,
        species=("Si",),
        positions_ang=((0.0, 0.0, 0.0),),
        fractional_positions=((0.0, 0.0, 0.0),),
        atomic_spins=None,
    )


def _cluster_features(
    optimizations: list[UepOptimizationResult],
    *,
    source_structure: object,
    config: FeatureClusteringConfig | None = None,
) -> FeatureClusteringResult:
    prepared = prepare_feature_dataset(
        optimizations, source_structure=source_structure, config=config
    )
    return cluster_prepared_features(prepared, config=config)


def _prepared_dataset(
    optimizations: tuple[UepOptimizationResult, ...] = (),
    genome_vectors: tuple[tuple[float, ...], ...] = (),
) -> PreparedFeatureDataset:
    """A prepared clustering dataset built by hand, so replaying it needs no Soprano."""

    return PreparedFeatureDataset(
        config=FeatureClusteringConfig(),
        optimizations=optimizations,
        asymmetric_fractional_positions=tuple(
            entry.optimized_fractional_position for entry in optimizations
        ),
        genome_vectors=genome_vectors,
        normalized_genome_vectors=genome_vectors,
        linkage_method="single",
        linkage_matrix=(),
    )


@_soprano_required
def test_soprano_backed_clustering_collapses_symmetry_equivalent_sites() -> None:
    result = _cluster_features(
        [
            _fake_optimization((0.25, 0.25, 0.25), total=-1.0),
            _fake_optimization((0.75, 0.75, 0.75), total=-1.0),
        ],
        source_structure=_simple_cubic_host(),
        config=FeatureClusteringConfig(clustering_method="hier", clustering_hier_t=0.05),
    )

    assert len(result.summary.clusters) == 1
    assert result.summary.clusters[0].member_indexes == (0, 1)
    assert result.summary.clusters[0].representative_index == 0
    assert result.summary.clusters[0].representative_asymmetric_fractional_position is not None
    assert result.asymmetric_fractional_positions[0] == pytest.approx(
        result.asymmetric_fractional_positions[1]
    )


@_soprano_required
def test_soprano_backed_clustering_merges_periodic_boundary_equivalents() -> None:
    result = _cluster_features(
        [
            _fake_optimization((0.98, 0.10, 0.10), total=-1.0),
            _fake_optimization((0.02, 0.10, 0.10), total=-1.0),
        ],
        source_structure=_simple_cubic_host(),
        config=FeatureClusteringConfig(clustering_method="hier", clustering_hier_t=0.05),
    )

    assert len(result.summary.clusters) == 1
    assert result.asymmetric_fractional_positions[0] == pytest.approx(
        result.asymmetric_fractional_positions[1]
    )


@_soprano_required
def test_soprano_backed_hierarchical_clustering_and_stats() -> None:
    optimizations = [
        _fake_optimization((0.24, 0.24, 0.24), total=-1.00),
        _fake_optimization((0.26, 0.25, 0.25), total=-0.80),
        _fake_optimization((0.50, 0.50, 0.50), total=-0.40),
        _fake_optimization((0.52, 0.49, 0.51), total=-0.30),
    ]
    result = _cluster_features(
        optimizations,
        source_structure=_simple_cubic_host(),
        config=FeatureClusteringConfig(clustering_method="hier", clustering_hier_t=0.30),
    )

    assert result.summary.clustering_method == "hier"
    assert len(result.summary.clusters) == 2
    member_sets = sorted(
        tuple(sorted(cluster.member_indexes)) for cluster in result.summary.clusters
    )
    assert member_sets == [(0, 1), (2, 3)]
    first = next(cluster for cluster in result.summary.clusters if cluster.member_indexes == (0, 1))
    assert first.representative_index == 0
    assert first.minimum_energy_ev == pytest.approx(-1.0)
    assert first.average_energy_ev == pytest.approx(-0.9)
    assert first.standard_deviation_ev == pytest.approx(0.1)
    member_average_classical = [
        float(
            np.mean([optimizations[index].classical_energy_ev for index in cluster.member_indexes])
        )
        for cluster in result.summary.clusters
    ]
    assert member_average_classical == sorted(member_average_classical)
    assert [cluster.cluster_index for cluster in result.summary.clusters] == [1, 2]
    assert result.summary.similarities
    assert result.summary.similarities[0].left_cluster_index == 1
    assert result.summary.similarities[0].right_cluster_index == 2
    assert result.summary.similarities[0].distance > 0.0


@_soprano_required
def test_soprano_backed_kmeans_clustering_uses_same_features() -> None:
    result = _cluster_features(
        [
            _fake_optimization((0.24, 0.24, 0.24), total=-1.00),
            _fake_optimization((0.26, 0.25, 0.25), total=-0.80),
            _fake_optimization((0.50, 0.50, 0.50), total=-0.40),
            _fake_optimization((0.52, 0.49, 0.51), total=-0.30),
        ],
        source_structure=_simple_cubic_host(),
        config=FeatureClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=2),
    )

    assert result.summary.clustering_method == "kmeans"
    assert len(result.summary.clusters) == 2
    member_sets = sorted(
        tuple(sorted(cluster.member_indexes)) for cluster in result.summary.clusters
    )
    assert member_sets == [(0, 1), (2, 3)]
    assert len(result.genome_vectors) == 4
    assert len(result.normalized_genome_vectors) == 4


@_soprano_required
def test_prepared_uep_clustering_dataset_supports_replayable_hier_and_kmeans_modes() -> None:
    prepared = prepare_feature_dataset(
        [
            _fake_optimization((0.24, 0.24, 0.24), total=-1.00),
            _fake_optimization((0.26, 0.25, 0.25), total=-0.80),
            _fake_optimization((0.50, 0.50, 0.50), total=-0.40),
            _fake_optimization((0.52, 0.49, 0.51), total=-0.30),
        ],
        source_structure=_simple_cubic_host(),
    )

    hier = cluster_prepared_features(
        prepared,
        config=FeatureClusteringConfig(clustering_method="hier", clustering_hier_t=0.30),
    )
    kmeans = cluster_prepared_features(
        prepared,
        config=FeatureClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=2),
    )

    assert prepared.linkage_matrix
    assert hier.linkage_matrix
    assert len(hier.summary.clusters) == 2
    assert len(kmeans.kmeans_centroids) == 2
    assert len(kmeans.summary.clusters) == 2


@_soprano_required
@pytest.mark.parametrize(
    "config",
    [
        FeatureClusteringConfig(clustering_method="hier", clustering_hier_t=0.30),
        FeatureClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=3),
    ],
    ids=["hier", "kmeans"],
)
def test_clustering_labels_sites_with_their_clusters_energy_rank(config) -> None:
    pairs = (
        ((0.65, 0.65, 0.65), (0.66, 0.67, 0.64)),
        ((0.10, 0.10, 0.10), (0.12, 0.09, 0.11)),
        ((0.35, 0.90, 0.35), (0.36, 0.91, 0.34)),
    )
    # Two energy orders of the same geometry: whichever labels the clustering
    # first assigns, at least one order differs from them.
    for ranks in ((3, 1, 2), (1, 3, 2)):
        prepared = prepare_feature_dataset(
            [
                _fake_optimization(position, total=-1.0 + 0.3 * rank + 0.01 * offset)
                for pair, rank in zip(pairs, ranks, strict=True)
                for offset, position in enumerate(pair)
            ],
            source_structure=_simple_cubic_host(),
        )

        result = cluster_prepared_features(prepared, config=config)

        assert result.cluster_labels == tuple(rank for rank in ranks for _ in range(2))
        assert [cluster.cluster_index for cluster in result.summary.clusters] == [1, 2, 3]
        for cluster in result.summary.clusters:
            assert {result.cluster_labels[site] for site in cluster.member_indexes} == {
                cluster.cluster_index
            }
        vectors = np.asarray(result.normalized_genome_vectors)
        assert len(result.kmeans_centroids) == (3 if config.clustering_method == "kmeans" else 0)
        for row, centroid in enumerate(result.kmeans_centroids):
            members = np.asarray(result.summary.clusters[row].member_indexes)
            assert np.allclose(centroid, vectors[members].mean(axis=0))


def test_prepare_feature_dataset_empty_returns_empty_features() -> None:
    prepared = prepare_feature_dataset([], source_structure=_simple_cubic_host())
    assert prepared.optimizations == ()
    assert prepared.genome_vectors == ()
    assert prepared.linkage_matrix == ()


def test_cluster_prepared_features_empty_returns_empty_result() -> None:
    prepared = PreparedFeatureDataset(
        config=FeatureClusteringConfig(),
        optimizations=(),
        asymmetric_fractional_positions=(),
        genome_vectors=(),
        normalized_genome_vectors=(),
        linkage_method="single",
        linkage_matrix=(),
    )
    result = cluster_prepared_features(
        prepared, config=FeatureClusteringConfig(clustering_method="hier")
    )
    assert result.cluster_labels == ()
    assert result.summary.clusters == ()
    assert result.summary.clustering_method == "hier"


def test_cluster_prepared_features_empty_kmeans_reports_k() -> None:
    prepared = PreparedFeatureDataset(
        config=FeatureClusteringConfig(),
        optimizations=(),
        asymmetric_fractional_positions=(),
        genome_vectors=(),
        normalized_genome_vectors=(),
        linkage_method="single",
        linkage_matrix=(),
    )
    result = cluster_prepared_features(
        prepared,
        config=FeatureClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=3),
    )
    assert result.summary.clustering_method == "kmeans"
    assert result.summary.clustering_kmeans_k == 3


def test_cluster_prepared_features_single_hier_returns_one_cluster() -> None:
    prepared = PreparedFeatureDataset(
        config=FeatureClusteringConfig(),
        optimizations=(_fake_optimization((0.25, 0.25, 0.25), classical=-1.0, total=-1.0),),
        asymmetric_fractional_positions=((0.25, 0.25, 0.25),),
        genome_vectors=((-1.0, 0.25, 0.25, 0.25),),
        normalized_genome_vectors=((0.0, 0.0, 0.0, 0.0),),
        linkage_method="single",
        linkage_matrix=(),
    )
    result = cluster_prepared_features(
        prepared,
        config=FeatureClusteringConfig(clustering_method="hier", clustering_hier_t=0.3),
    )
    assert result.cluster_labels == (1,)
    assert len(result.summary.clusters) == 1
    assert result.summary.clusters[0].member_indexes == (0,)


def test_cluster_prepared_features_kmeans_rejects_too_many_clusters() -> None:
    prepared = PreparedFeatureDataset(
        config=FeatureClusteringConfig(),
        optimizations=(_fake_optimization((0.1, 0.1, 0.1), classical=-1.0, total=-1.0),),
        asymmetric_fractional_positions=((0.1, 0.1, 0.1),),
        genome_vectors=((-1.0, 0.1, 0.1, 0.1),),
        normalized_genome_vectors=((0.0, 0.0, 0.0, 0.0),),
        linkage_method="single",
        linkage_matrix=(),
    )
    with pytest.raises(UepError, match="must not exceed the number of optimized sites"):
        cluster_prepared_features(
            prepared,
            config=FeatureClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=5),
        )


def test_cluster_prepared_features_ranks_cluster_similarities_after_renumbering() -> None:
    prepared = _prepared_dataset(
        optimizations=(
            _fake_optimization((0.1, 0.1, 0.1), total=3.0),
            _fake_optimization((0.4, 0.4, 0.4), total=1.0),
            _fake_optimization((0.7, 0.7, 0.7), total=2.0),
        ),
        genome_vectors=((0.0,), (1.0,), (5.0,)),
    )

    result = cluster_prepared_features(
        prepared,
        config=FeatureClusteringConfig(clustering_method="hier", clustering_hier_t=0.1),
    )

    # Clusters are numbered by energy (sites 1, 2, 0) and their similarities
    # are reported between the renumbered clusters, closest pair first.
    assert [cluster.member_indexes for cluster in result.summary.clusters] == [(1,), (2,), (0,)]
    assert [
        (similarity.left_cluster_index, similarity.right_cluster_index, similarity.distance)
        for similarity in result.summary.similarities
    ] == [(1, 3, 1.0), (1, 2, 4.0), (2, 3, 5.0)]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (FeatureClusteringConfig(clustering_method="dbscan"), "must be 'hier' or 'kmeans'"),
        (FeatureClusteringConfig(clustering_hier_t=0.0), "clustering_hier_t must be positive"),
        (FeatureClusteringConfig(clustering_kmeans_k=0), "clustering_kmeans_k must be positive"),
    ],
    ids=["unknown-method", "non-positive-hier-t", "non-positive-kmeans-k"],
)
def test_cluster_prepared_features_validates_config(
    config: FeatureClusteringConfig, message: str
) -> None:
    with pytest.raises(UepError, match=message):
        cluster_prepared_features(_prepared_dataset(), config=config)


@pytest.mark.parametrize(
    ("method", "linkage", "expected_linkage"),
    [("HIER", "  Single  ", "single"), ("hier", "   ", "single"), ("Hier", "Average", "average")],
)
def test_cluster_prepared_features_normalizes_config(
    method: str, linkage: str, expected_linkage: str
) -> None:
    result = cluster_prepared_features(
        _prepared_dataset(),
        config=FeatureClusteringConfig(clustering_method=method, clustering_linkage_method=linkage),
    )

    assert result.config.clustering_method == "hier"
    assert result.config.clustering_linkage_method == expected_linkage


@_soprano_required
@pytest.mark.parametrize(
    ("host", "message"),
    [
        (
            type(
                "_Partial",
                (),
                {
                    "lattice_vectors": _CUBE_LATTICE,
                    "species": ("Si",),
                    "fractional_positions": None,
                },
            )(),
            "must define lattice_vectors, species",
        ),
        (
            type(
                "_Mismatched",
                (),
                {
                    "lattice_vectors": _CUBE_LATTICE,
                    "species": ("Si", "Si"),
                    "fractional_positions": ((0.0, 0.0, 0.0),),
                },
            )(),
            "matching lengths",
        ),
    ],
    ids=["missing-positions", "mismatched-lengths"],
)
def test_prepare_feature_dataset_rejects_incomplete_host(host: object, message: str) -> None:
    with pytest.raises(UepError, match=message):
        prepare_feature_dataset(
            [_fake_optimization((0.1, 0.1, 0.1), total=-1.0)], source_structure=host
        )


def test_asymmetric_images_centre_jitters_the_lowest_energy_site() -> None:
    # Private helper: the jittered centre is consumed inside Soprano and never returned.
    optimizations = [
        _fake_optimization((0.5, 0.5, 0.5), total=-1.0),
        _fake_optimization((0.1, 0.1, 0.1), total=0.0),
    ]
    centre = feature_clustering_module._asymmetric_images_centre(optimizations)
    assert centre.shape == (3,)
    assert np.all(centre != np.array([0.5, 0.5, 0.5]))
    assert np.all(np.abs(centre - 0.5) < 1e-3)


def test_dendrogram_node_clusters_keep_a_cluster_only_while_all_leaves_share_it() -> None:
    linkage = ((0.0, 1.0, 0.1, 2.0), (2.0, 3.0, 0.2, 2.0), (4.0, 5.0, 0.9, 4.0))

    assert dendrogram_node_clusters((1, 1, 2, 2), linkage) == {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 1,
        5: 2,
        6: None,
    }
    assert dendrogram_node_clusters((1,), ()) == {0: 1}
