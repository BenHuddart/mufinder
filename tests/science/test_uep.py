"""Tests for MuFinder-native UEP source loading and field analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_soprano_required = pytest.mark.skipif(
    importlib.util.find_spec("soprano") is None,
    reason="soprano optional dependency not installed",
)

from mufinder.science.uep import (
    StationaryPointClassification,
    UepClusteringConfig,
    UepError,
    UepField,
    UepGridSliceSample,
    UepIsosurfaceMesh,
    UepStructure,
    UepOptimizationResult,
    UepPotentialGrid,
    fractional_point_to_cartesian,
    miller_indices_to_cartesian_normal,
    plane_from_best_fit,
    plane_from_fractional_anchor_and_miller,
    plane_from_point_and_normal,
    plane_from_three_points,
    _reshape_pyvista_faces,
    cluster_uep_optimizations,
    cluster_prepared_uep_dataset,
    cluster_optimized_positions,
    calculate_optimization_zero_point_energy,
    deserialize_potential_grid,
    extract_isosurface,
    load_uep_source,
    optimize_muon_position,
    prepare_uep_clustering_dataset,
    potential_grid,
    preview_potential_grid,
    serialize_potential_grid,
    probe_point,
    sample_line,
    sample_plane,
    sample_grid_slice,
    write_potential_grid_xsf,
)


def _fixture_path(name: str) -> Path:
    return Path("tests/fixtures/uep") / name


def test_load_uep_source_reads_neutral_fixture() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    assert source.structure.species == ("Si", "Si")
    assert source.gw_factor == pytest.approx(5.0)
    assert source.neutrality_warning is None or "assuming a neutral host" in source.neutrality_warning


def test_load_uep_source_uses_shared_mulliken_spin_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mufinder.science.uep.parse_castep_mulliken_atomic_spins", lambda _path: (1.2, -1.2))

    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    assert source.structure.atomic_spins == pytest.approx((1.2, -1.2))


def test_load_uep_source_rejects_non_neutral_density_fixture() -> None:
    with pytest.raises(UepError, match="neutral host cell"):
        load_uep_source(
            den_fmt_path=_fixture_path("Si2-charged.den_fmt"),
            castep_path=_fixture_path("Si2-charged.castep"),
            cell_path=_fixture_path("Si2.cell"),
        )


def test_uep_field_respects_periodicity_for_equivalent_points() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    field = UepField(source)

    origin = (0.2, 0.3, 0.4)
    lattice_vector = source.structure.lattice_vectors[0]
    translated = (
        origin[0] + lattice_vector[0],
        origin[1] + lattice_vector[1],
        origin[2] + lattice_vector[2],
    )
    origin_probe = probe_point(field, origin)
    translated_probe = probe_point(field, translated)

    assert translated_probe.fractional_position == pytest.approx(origin_probe.fractional_position)
    assert translated_probe.potential_total_ev == pytest.approx(origin_probe.potential_total_ev, rel=1e-9)


def test_uep_field_gradient_matches_potential_finite_difference() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    field = UepField(source)

    point = np.array((1.0, 1.2, 1.4), dtype=float)
    delta_ang = 1e-5
    gradient = np.asarray(field.gradient(point)[0][0], dtype=float)
    finite_difference = np.zeros(3, dtype=float)
    for axis in range(3):
        step = np.zeros(3, dtype=float)
        step[axis] = delta_ang
        forward = float(field.potential(point + step)[0][0])
        backward = float(field.potential(point - step)[0][0])
        finite_difference[axis] = (forward - backward) / (2.0 * delta_ang)

    assert gradient == pytest.approx(finite_difference, rel=1e-5, abs=1e-4)


def test_uep_field_hessian_matches_potential_finite_difference() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    field = UepField(source)

    point = np.array((1.0, 1.2, 1.4), dtype=float)
    delta_ang = 1e-4
    hessian = np.asarray(field.hessian(point)[0][0], dtype=float)
    finite_difference = np.zeros((3, 3), dtype=float)
    center = float(field.potential(point)[0][0])
    for row in range(3):
        axis_step = np.zeros(3, dtype=float)
        axis_step[row] = delta_ang
        finite_difference[row, row] = (
            float(field.potential(point + axis_step)[0][0])
            - (2.0 * center)
            + float(field.potential(point - axis_step)[0][0])
        ) / (delta_ang**2)
    for row in range(3):
        for column in range(row + 1, 3):
            step_row = np.zeros(3, dtype=float)
            step_column = np.zeros(3, dtype=float)
            step_row[row] = delta_ang
            step_column[column] = delta_ang
            mixed = (
                float(field.potential(point + step_row + step_column)[0][0])
                - float(field.potential(point + step_row - step_column)[0][0])
                - float(field.potential(point - step_row + step_column)[0][0])
                + float(field.potential(point - step_row - step_column)[0][0])
            ) / (4.0 * delta_ang**2)
            finite_difference[row, column] = mixed
            finite_difference[column, row] = mixed

    assert hessian == pytest.approx(finite_difference, rel=1e-4, abs=1e-3)


def test_uep_sampling_returns_expected_shapes() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si8.den_fmt"),
        castep_path=_fixture_path("Si8.castep"),
        cell_path=_fixture_path("Si8.cell"),
    )
    field = UepField(source)

    line = sample_line(field, start_ang=(0.0, 0.0, 0.0), end_ang=(1.0, 1.0, 1.0), num_points=5)
    plane = sample_plane(
        field,
        origin_ang=(0.0, 0.0, 0.0),
        corner_x_ang=(1.0, 0.0, 0.0),
        corner_y_ang=(0.0, 1.0, 0.0),
        points_x=4,
        points_y=3,
    )

    assert len(line.distance_ang) == 5
    assert len(line.potential_total_ev) == 5
    assert len(plane.potential_total_ev) == 4
    assert len(plane.potential_total_ev[0]) == 3


def test_sample_plane_rejects_collinear_axes() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si8.den_fmt"),
        castep_path=_fixture_path("Si8.castep"),
        cell_path=_fixture_path("Si8.cell"),
    )
    field = UepField(source)

    with pytest.raises(UepError, match="must not be collinear"):
        sample_plane(
            field,
            origin_ang=(0.0, 0.0, 0.0),
            corner_x_ang=(1.0, 0.0, 0.0),
            corner_y_ang=(2.0, 0.0, 0.0),
            points_x=3,
            points_y=3,
        )


def test_sample_grid_slice_returns_periodic_plane_sample() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        grid=(4, 4, 4),
        potential_total_ev=np.arange(64, dtype=float).reshape((4, 4, 4)),
    )
    plane = plane_from_point_and_normal(
        anchor_ang=(0.4, 0.6, 0.8),
        normal_ang=(0.0, 0.0, 1.0),
        lattice_vectors=grid.lattice_vectors,
        points_u=5,
        points_v=6,
    )

    sample = sample_grid_slice(grid, plane)

    assert isinstance(sample, UepGridSliceSample)
    assert len(sample.potential_total_ev) == 5
    assert len(sample.potential_total_ev[0]) == 6
    assert sample.minimum_potential_ev <= sample.maximum_potential_ev


def test_sample_grid_slice_wraps_periodically_for_equivalent_planes() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        grid=(4, 4, 4),
        potential_total_ev=np.arange(64, dtype=float).reshape((4, 4, 4)),
    )
    first = plane_from_point_and_normal(
        anchor_ang=(0.4, 0.6, 0.8),
        normal_ang=(0.0, 1.0, 1.0),
        lattice_vectors=grid.lattice_vectors,
        points_u=6,
        points_v=5,
    )
    second = plane_from_point_and_normal(
        anchor_ang=(2.4, 0.6, 0.8),
        normal_ang=(0.0, 1.0, 1.0),
        lattice_vectors=grid.lattice_vectors,
        points_u=6,
        points_v=5,
    )

    first_sample = sample_grid_slice(grid, first)
    second_sample = sample_grid_slice(grid, second)

    assert np.asarray(first_sample.potential_total_ev, dtype=float) == pytest.approx(
        np.asarray(second_sample.potential_total_ev, dtype=float)
    )


def test_potential_grid_serialization_preserves_gw_factor() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        grid=(2, 2, 2),
        potential_total_ev=np.arange(8, dtype=float).reshape((2, 2, 2)),
        gw_factor=7.5,
    )

    restored = deserialize_potential_grid(serialize_potential_grid(grid))

    assert restored is not None
    assert restored.gw_factor == pytest.approx(7.5)
    assert restored.potential_total_ev == pytest.approx(grid.potential_total_ev)


def test_plane_from_three_points_rejects_collinear_points() -> None:
    with pytest.raises(UepError, match="must not be collinear"):
        plane_from_three_points(
            point_a_ang=(0.0, 0.0, 0.0),
            point_b_ang=(1.0, 0.0, 0.0),
            point_c_ang=(2.0, 0.0, 0.0),
            lattice_vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        )


def test_plane_from_best_fit_uses_selected_points() -> None:
    plane = plane_from_best_fit(
        points_ang=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        lattice_vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
    )

    assert plane.anchor_ang == pytest.approx((0.5, 0.5, 0.0))
    assert abs(float(np.dot(np.asarray(plane.normal_ang, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)))) == pytest.approx(1.0)


def test_plane_from_fractional_anchor_and_miller_uses_hexagonal_lattice_coordinates() -> None:
    lattice_vectors = (
        (4.2, 0.0, 0.0),
        (-2.1, 0.5 * np.sqrt(3.0) * 4.2, 0.0),
        (0.0, 0.0, 6.8),
    )

    plane = plane_from_fractional_anchor_and_miller(
        anchor_fractional=(0.25, 0.5, 0.75),
        miller_indices=(1.0, 0.0, 0.0),
        lattice_vectors=lattice_vectors,
        points_u=9,
        points_v=9,
    )

    normal = np.asarray(plane.normal_ang, dtype=float)
    assert plane.anchor_ang == pytest.approx(fractional_point_to_cartesian(lattice_vectors, (0.25, 0.5, 0.75)))
    assert plane.normal_ang == pytest.approx(miller_indices_to_cartesian_normal(lattice_vectors, (1.0, 0.0, 0.0)))
    assert float(np.dot(normal, np.asarray(lattice_vectors[1], dtype=float))) == pytest.approx(0.0, abs=1e-9)
    assert float(np.dot(normal, np.asarray(lattice_vectors[2], dtype=float))) == pytest.approx(0.0, abs=1e-9)


def test_zero_shifted_grid_slice_preserves_shifted_bounds() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
        grid=(4, 4, 4),
        potential_total_ev=np.linspace(-3.0, 5.0, 64, dtype=float).reshape((4, 4, 4)),
    )
    shifted = UepPotentialGrid(
        lattice_vectors=grid.lattice_vectors,
        grid=grid.grid,
        potential_total_ev=np.asarray(grid.potential_total_ev, dtype=float) - float(np.min(grid.potential_total_ev)),
    )
    plane = plane_from_point_and_normal(
        anchor_ang=(0.5, 0.5, 0.5),
        normal_ang=(1.0, 0.0, 1.0),
        lattice_vectors=grid.lattice_vectors,
        points_u=8,
        points_v=8,
    )

    sample = sample_grid_slice(shifted, plane)

    assert sample.minimum_potential_ev >= 0.0
    assert sample.maximum_potential_ev <= float(np.max(shifted.potential_total_ev)) + 1e-9


def test_optimize_muon_position_and_cluster_summary() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    field = UepField(source)

    first = optimize_muon_position(field, start_position_ang=(1.0, 1.0, 1.0), max_steps=30)
    second = optimize_muon_position(field, start_position_ang=(1.05, 1.0, 1.0), max_steps=30)

    assert isinstance(first, UepOptimizationResult)
    assert isinstance(second, UepOptimizationResult)
    assert first.success is True
    assert second.success is True
    assert first.classification.label == "minimum"
    assert second.classification.label == "minimum"
    assert np.linalg.norm(np.asarray(first.optimized_position_ang) - np.array((1.0, 1.0, 1.0), dtype=float)) > 0.1
    assert np.linalg.norm(np.asarray(second.optimized_position_ang) - np.array((1.05, 1.0, 1.0), dtype=float)) > 0.1

    summary = cluster_optimized_positions([first, second], threshold_ang=0.3)
    assert len(summary.clusters) >= 1
    assert summary.clusters[0].size >= 1


def test_legacy_cluster_summary_numbers_clusters_by_average_classical_energy() -> None:
    optimizations = [
        _fake_optimization((0.10, 0.10, 0.10), total=0.90),
        _fake_optimization((0.11, 0.10, 0.10), total=1.10),
        _fake_optimization((0.75, 0.75, 0.75), total=-0.50),
    ]
    summary = cluster_optimized_positions(optimizations, threshold_ang=0.2)

    assert len(summary.clusters) == 2
    member_average_classical = [
        float(np.mean([optimizations[index].classical_energy_ev for index in cluster.member_indexes]))
        for cluster in summary.clusters
    ]
    assert member_average_classical == sorted(member_average_classical)
    assert summary.clusters[0].member_indexes == (2,)
    assert summary.clusters[0].cluster_index == 1


def test_optimize_muon_position_can_skip_and_then_restore_quantum_corrections() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    field = UepField(source)

    classical = optimize_muon_position(
        field,
        start_position_ang=(1.0, 1.0, 1.0),
        max_steps=30,
        include_zero_point_energy=False,
    )
    quantum = calculate_optimization_zero_point_energy(field, classical)
    direct = optimize_muon_position(field, start_position_ang=(1.0, 1.0, 1.0), max_steps=30)

    assert classical.zero_point_energy_ev is None
    assert classical.total_energy_ev is None
    assert quantum.zero_point_energy_ev is not None
    assert quantum.total_energy_ev is not None
    assert quantum.zero_point_energy_ev == pytest.approx(direct.zero_point_energy_ev)
    assert quantum.total_energy_ev == pytest.approx(direct.total_energy_ev)
    assert quantum.classical_energy_ev == pytest.approx(direct.classical_energy_ev)


def _simple_cubic_host() -> UepStructure:
    return UepStructure(
        lattice_vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        species=("Si",),
        positions_ang=((0.0, 0.0, 0.0),),
        fractional_positions=((0.0, 0.0, 0.0),),
        atomic_spins=None,
    )


def _fake_optimization(
    fractional: tuple[float, float, float],
    *,
    total: float,
) -> UepOptimizationResult:
    lattice = np.asarray(_simple_cubic_host().lattice_vectors, dtype=float)
    cartesian = tuple(float(value) for value in (np.asarray(fractional, dtype=float) @ lattice).tolist())
    return UepOptimizationResult(
        start_position_ang=cartesian,
        optimized_position_ang=cartesian,
        optimized_fractional_position=fractional,
        classical_energy_ev=float(total),
        zero_point_energy_ev=0.0,
        total_energy_ev=float(total),
        success=True,
        message="",
        iterations=1,
        classification=StationaryPointClassification(
            label="minimum",
            eigenvalues=(1.0, 1.0, 1.0),
            positive_count=3,
            negative_count=0,
            near_zero_count=0,
        ),
    )


@_soprano_required
def test_soprano_backed_clustering_collapses_symmetry_equivalent_sites() -> None:
    result = cluster_uep_optimizations(
        [
            _fake_optimization((0.25, 0.25, 0.25), total=-1.0),
            _fake_optimization((0.75, 0.75, 0.75), total=-1.0),
        ],
        source_structure=_simple_cubic_host(),
        config=UepClusteringConfig(clustering_method="hier", clustering_hier_t=0.05),
    )

    assert len(result.summary.clusters) == 1
    assert result.summary.clusters[0].member_indexes == (0, 1)
    assert result.summary.clusters[0].representative_index == 0
    assert result.summary.clusters[0].representative_asymmetric_fractional_position is not None
    assert result.asymmetric_fractional_positions[0] == pytest.approx(result.asymmetric_fractional_positions[1])


@_soprano_required
def test_soprano_backed_clustering_merges_periodic_boundary_equivalents() -> None:
    result = cluster_uep_optimizations(
        [
            _fake_optimization((0.98, 0.10, 0.10), total=-1.0),
            _fake_optimization((0.02, 0.10, 0.10), total=-1.0),
        ],
        source_structure=_simple_cubic_host(),
        config=UepClusteringConfig(clustering_method="hier", clustering_hier_t=0.05),
    )

    assert len(result.summary.clusters) == 1
    assert result.asymmetric_fractional_positions[0] == pytest.approx(result.asymmetric_fractional_positions[1])


@_soprano_required
def test_soprano_backed_hierarchical_clustering_and_stats() -> None:
    optimizations = [
        _fake_optimization((0.24, 0.24, 0.24), total=-1.00),
        _fake_optimization((0.26, 0.25, 0.25), total=-0.80),
        _fake_optimization((0.50, 0.50, 0.50), total=-0.40),
        _fake_optimization((0.52, 0.49, 0.51), total=-0.30),
    ]
    result = cluster_uep_optimizations(
        optimizations,
        source_structure=_simple_cubic_host(),
        config=UepClusteringConfig(clustering_method="hier", clustering_hier_t=0.30),
    )

    assert result.summary.clustering_method == "hier"
    assert len(result.summary.clusters) == 2
    member_sets = sorted(tuple(sorted(cluster.member_indexes)) for cluster in result.summary.clusters)
    assert member_sets == [(0, 1), (2, 3)]
    first = next(cluster for cluster in result.summary.clusters if cluster.member_indexes == (0, 1))
    assert first.representative_index == 0
    assert first.minimum_energy_ev == pytest.approx(-1.0)
    assert first.average_energy_ev == pytest.approx(-0.9)
    assert first.standard_deviation_ev == pytest.approx(0.1)
    member_average_classical = [
        float(np.mean([optimizations[index].classical_energy_ev for index in cluster.member_indexes]))
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
    result = cluster_uep_optimizations(
        [
            _fake_optimization((0.24, 0.24, 0.24), total=-1.00),
            _fake_optimization((0.26, 0.25, 0.25), total=-0.80),
            _fake_optimization((0.50, 0.50, 0.50), total=-0.40),
            _fake_optimization((0.52, 0.49, 0.51), total=-0.30),
        ],
        source_structure=_simple_cubic_host(),
        config=UepClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=2),
    )

    assert result.summary.clustering_method == "kmeans"
    assert len(result.summary.clusters) == 2
    member_sets = sorted(tuple(sorted(cluster.member_indexes)) for cluster in result.summary.clusters)
    assert member_sets == [(0, 1), (2, 3)]
    assert len(result.genome_vectors) == 4
    assert len(result.normalized_genome_vectors) == 4


@_soprano_required
def test_prepared_uep_clustering_dataset_supports_replayable_hier_and_kmeans_modes() -> None:
    prepared = prepare_uep_clustering_dataset(
        [
            _fake_optimization((0.24, 0.24, 0.24), total=-1.00),
            _fake_optimization((0.26, 0.25, 0.25), total=-0.80),
            _fake_optimization((0.50, 0.50, 0.50), total=-0.40),
            _fake_optimization((0.52, 0.49, 0.51), total=-0.30),
        ],
        source_structure=_simple_cubic_host(),
    )

    hier = cluster_prepared_uep_dataset(
        prepared,
        config=UepClusteringConfig(clustering_method="hier", clustering_hier_t=0.30),
    )
    kmeans = cluster_prepared_uep_dataset(
        prepared,
        config=UepClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=2),
    )

    assert prepared.linkage_matrix
    assert hier.linkage_matrix
    assert len(hier.summary.clusters) == 2
    assert len(kmeans.kmeans_centroids) == 2
    assert len(kmeans.summary.clusters) == 2


def test_potential_grid_and_isosurface_extraction() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si8.den_fmt"),
        castep_path=_fixture_path("Si8.castep"),
        cell_path=_fixture_path("Si8.cell"),
    )
    field = UepField(source)

    grid = potential_grid(field)
    level = 0.5 * (float(grid.potential_total_ev.min()) + float(grid.potential_total_ev.max()))
    mesh = extract_isosurface(grid, level_ev=level)

    assert isinstance(mesh, UepIsosurfaceMesh)
    assert len(mesh.vertices_ang) >= 3
    assert len(mesh.faces) >= 1
    assert mesh.minimum_potential_ev <= mesh.level_ev <= mesh.maximum_potential_ev


def test_extract_isosurface_can_use_pyvista_contour_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePolyData:
        def __init__(self) -> None:
            self.n_cells = 1
            self.points = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=float,
            )
            self.faces = np.array([3, 0, 1, 2], dtype=int)

        def triangulate(self):
            return self

    class _FakeStructuredGrid:
        def __init__(self, *_coords) -> None:
            self.scalars = None

        def __setitem__(self, _name: str, values: np.ndarray) -> None:
            self.scalars = np.asarray(values, dtype=float)

        def contour(self, *, isosurfaces, scalars: str):
            assert isosurfaces == [0.25]
            assert scalars == "potential_total_ev"
            assert self.scalars is not None
            return _FakePolyData()

    class _FakePyVista:
        StructuredGrid = _FakeStructuredGrid

    monkeypatch.setattr("mufinder.science.uep._pyvista_modules", lambda: _FakePyVista())
    grid = UepPotentialGrid(
        lattice_vectors=((1.0, 0.0, 0.0), (0.2, 1.0, 0.0), (0.0, 0.3, 1.0)),
        grid=(2, 2, 2),
        potential_total_ev=np.array(
            [
                [[0.0, 0.2], [0.1, 0.3]],
                [[0.2, 0.4], [0.3, 0.5]],
            ],
            dtype=float,
        ),
    )

    mesh = extract_isosurface(grid, level_ev=0.25)

    assert isinstance(mesh, UepIsosurfaceMesh)
    assert mesh.faces == ((0, 1, 2),)


def test_reshape_pyvista_faces_triangulates_polygons() -> None:
    faces = _reshape_pyvista_faces(np.array([4, 0, 1, 2, 3, 3, 4, 5, 6], dtype=int))

    assert faces.tolist() == [[0, 1, 2], [0, 2, 3], [4, 5, 6]]


def test_preview_potential_grid_caps_resolution_and_preserves_range() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si8.den_fmt"),
        castep_path=_fixture_path("Si8.castep"),
        cell_path=_fixture_path("Si8.cell"),
    )
    field = UepField(source)
    grid = potential_grid(field)

    preview = preview_potential_grid(grid, max_grid_points=12)

    assert preview.grid == (12, 12, 12)
    assert preview.potential_total_ev.shape == (12, 12, 12)
    assert float(preview.potential_total_ev.min()) == pytest.approx(float(grid.potential_total_ev.min()))
    assert float(preview.potential_total_ev.max()) == pytest.approx(float(grid.potential_total_ev.max()))


def test_extract_isosurface_rejects_level_outside_range() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si8.den_fmt"),
        castep_path=_fixture_path("Si8.castep"),
        cell_path=_fixture_path("Si8.cell"),
    )
    field = UepField(source)
    grid = potential_grid(field)

    with pytest.raises(UepError, match="outside the available range"):
        extract_isosurface(grid, level_ev=float(grid.potential_total_ev.max()) + 1.0)


def test_write_potential_grid_xsf_includes_structure_and_grid(tmp_path) -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    field = UepField(source)
    grid = potential_grid(field)

    output_path = tmp_path / "uep_potential.xsf"
    written_path = write_potential_grid_xsf(output_path, structure=source.structure, potential=grid)
    content = written_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    dimension_line = next(line for line in lines if line.strip() == "BEGIN_DATAGRID_3D_UEP_POTENTIAL")
    dimensions = lines[lines.index(dimension_line) + 1].strip()

    assert written_path == output_path
    assert "CRYSTAL" in content
    assert "PRIMVEC" in content
    assert "PRIMCOORD" in content
    assert "BEGIN_DATAGRID_3D_UEP_POTENTIAL" in content
    assert dimensions == f"{grid.grid[0] + 1} {grid.grid[1] + 1} {grid.grid[2] + 1}"


def test_write_potential_grid_xsf_duplicates_periodic_boundaries(tmp_path) -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    field = UepField(source)
    grid = potential_grid(field)

    output_path = tmp_path / "uep_potential.xsf"
    write_potential_grid_xsf(output_path, structure=source.structure, potential=grid)
    content = output_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    header_index = lines.index("BEGIN_DATAGRID_3D_UEP_POTENTIAL")
    nx, ny, nz = (int(token) for token in lines[header_index + 1].split())
    data_tokens: list[float] = []
    for line in lines[header_index + 6 :]:
        if line.strip() == "END_DATAGRID_3D_UEP_POTENTIAL":
            break
        data_tokens.extend(float(token) for token in line.split())
    xsf_values = np.asarray(data_tokens, dtype=float).reshape((nx, ny, nz), order="F")

    assert np.allclose(xsf_values[-1, :, :], xsf_values[0, :, :])
    assert np.allclose(xsf_values[:, -1, :], xsf_values[:, 0, :])
    assert np.allclose(xsf_values[:, :, -1], xsf_values[:, :, 0])
