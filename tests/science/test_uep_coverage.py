"""Additional coverage for MuFinder-native UEP analysis helpers.

These tests deliberately target uncovered functions, branches, and error
paths in :mod:`mufinder.science.uep` that the primary suite in
``tests/science/test_uep.py`` does not exercise. They use small synthetic
inputs only -- no network, GUI, randomness, sleeps, or external files beyond
``tmp_path`` scratch files written inside each test.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from mufinder.science.uep import (
    IonicPseudopotential,
    StationaryPointClassification,
    UepClusterSimilaritySummary,
    UepClusteringConfig,
    UepError,
    UepField,
    UepIsosurfaceMesh,
    UepOptimizationResult,
    UepPotentialGrid,
    UepPreparedClusteringDataset,
    UepSlicePlane,
    UepSourceData,
    UepStructure,
    _asymmetric_images_centre,
    _cartesian_to_fractional,
    _cluster_average_classical_energy,
    _cluster_groups_from_labels,
    _cluster_minimum_classical_energy,
    _cluster_similarity_summaries,
    _clustering_energy,
    _fractional_to_cartesian,
    _interpolate_isosurface_vertex,
    _interpolate_periodic_grid_trilinear,
    _order_polygon_vertices,
    _periodic_resample_axis,
    _polygonize_tetrahedron,
    _renumber_cluster_similarity_summaries,
    _reshape_pyvista_faces,
    _structure_like_to_ase_atoms,
    _validate_clustering_config,
    cartesian_normal_to_lattice_plane_components,
    castep_total_charge,
    classify_stationary_point,
    cluster_optimized_positions,
    cluster_prepared_uep_dataset,
    deserialize_potential_grid,
    extract_isosurface,
    fractional_point_to_cartesian,
    load_uep_source,
    miller_indices_to_cartesian_normal,
    normalize_plane_normal,
    parse_castep_pseudopotentials,
    parse_formatted_density,
    particle_mass_kg_from_amu,
    plane_basis_from_normal,
    plane_from_best_fit,
    plane_from_point_and_normal,
    potential_grid,
    prepare_uep_clustering_dataset,
    preview_potential_grid,
    relaxed_site_number_map,
    sample_grid_slice,
    sample_line,
    sample_plane,
    serialize_potential_grid,
    write_potential_grid_xsf,
    zero_point_energy_ev,
)


def _fixture_path(name: str) -> Path:
    return Path("tests/fixtures/uep") / name


_CUBE_LATTICE = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))


def _simple_structure() -> UepStructure:
    return UepStructure(
        lattice_vectors=_CUBE_LATTICE,
        species=("Si",),
        positions_ang=((0.0, 0.0, 0.0),),
        fractional_positions=((0.0, 0.0, 0.0),),
        atomic_spins=None,
    )


def _fake_optimization(
    fractional: tuple[float, float, float],
    *,
    classical: float,
    total: float | None = None,
    label: str = "minimum",
) -> UepOptimizationResult:
    lattice = np.asarray(_CUBE_LATTICE, dtype=float)
    cartesian = tuple(float(value) for value in (np.asarray(fractional, dtype=float) @ lattice).tolist())
    return UepOptimizationResult(
        start_position_ang=cartesian,
        optimized_position_ang=cartesian,
        optimized_fractional_position=fractional,
        classical_energy_ev=float(classical),
        zero_point_energy_ev=0.0 if total is not None else None,
        total_energy_ev=None if total is None else float(total),
        success=True,
        message="",
        iterations=1,
        classification=StationaryPointClassification(
            label=label,
            eigenvalues=(1.0, 1.0, 1.0),
            positive_count=3,
            negative_count=0,
            near_zero_count=0,
        ),
    )


# ---------------------------------------------------------------------------
# IonicPseudopotential property guard
# ---------------------------------------------------------------------------


def test_ionic_pseudopotential_gaussian_width_property_raises() -> None:
    pp = IonicPseudopotential(species="Si", ionic_charge=4.0, core_radius_ang=1.0)
    with pytest.raises(AttributeError, match="gw_factor"):
        _ = pp.gaussian_width_ang


# ---------------------------------------------------------------------------
# serialize / deserialize potential grid
# ---------------------------------------------------------------------------


def test_serialize_potential_grid_omits_missing_gw_factor() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(2, 2, 2),
        potential_total_ev=np.arange(8, dtype=float).reshape((2, 2, 2)),
        gw_factor=None,
    )
    payload = serialize_potential_grid(grid)
    assert "gw_factor" not in payload
    assert payload["grid"] == [2, 2, 2]


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {"lattice_vectors": [[1, 0, 0], [0, 1, 0]], "grid": [2, 2, 2], "potential_total_ev": []},
        {"lattice_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "grid": [2, 2], "potential_total_ev": []},
    ],
)
def test_deserialize_potential_grid_rejects_malformed_payloads(payload: object) -> None:
    assert deserialize_potential_grid(payload) is None


def test_deserialize_potential_grid_rejects_non_numeric_lattice() -> None:
    payload = {
        "lattice_vectors": [["x", 0, 0], [0, 1, 0], [0, 0, 1]],
        "grid": [2, 2, 2],
        "potential_total_ev": np.zeros((2, 2, 2)).tolist(),
    }
    assert deserialize_potential_grid(payload) is None


def test_deserialize_potential_grid_rejects_shape_mismatch() -> None:
    payload = {
        "lattice_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "grid": [2, 2, 2],
        "potential_total_ev": np.zeros((2, 2, 3)).tolist(),
    }
    assert deserialize_potential_grid(payload) is None


def test_deserialize_potential_grid_rejects_bad_gw_factor() -> None:
    payload = {
        "lattice_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "grid": [2, 2, 2],
        "potential_total_ev": np.zeros((2, 2, 2)).tolist(),
        "gw_factor": "nope",
    }
    assert deserialize_potential_grid(payload) is None


def test_deserialize_potential_grid_accepts_null_gw_factor() -> None:
    payload = {
        "lattice_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "grid": [2, 2, 2],
        "potential_total_ev": np.zeros((2, 2, 2)).tolist(),
        "gw_factor": None,
    }
    restored = deserialize_potential_grid(payload)
    assert restored is not None
    assert restored.gw_factor is None


# ---------------------------------------------------------------------------
# write_potential_grid_xsf error paths
# ---------------------------------------------------------------------------


def test_write_potential_grid_xsf_rejects_non_3d_grid(tmp_path: Path) -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(4,),
        potential_total_ev=np.zeros(4, dtype=float),
    )
    with pytest.raises(UepError, match="3D electrostatic-potential grid"):
        write_potential_grid_xsf(tmp_path / "out.xsf", structure=_simple_structure(), potential=grid)


def test_write_potential_grid_xsf_rejects_bad_lattice(tmp_path: Path) -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # type: ignore[arg-type]
        grid=(2, 2, 2),
        potential_total_ev=np.zeros((2, 2, 2), dtype=float),
    )
    with pytest.raises(UepError, match="three lattice vectors"):
        write_potential_grid_xsf(tmp_path / "out.xsf", structure=_simple_structure(), potential=grid)


def test_write_potential_grid_xsf_rejects_species_position_mismatch(tmp_path: Path) -> None:
    structure = UepStructure(
        lattice_vectors=_CUBE_LATTICE,
        species=("Si", "Si"),
        positions_ang=((0.0, 0.0, 0.0),),
        fractional_positions=((0.0, 0.0, 0.0),),
    )
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(2, 2, 2),
        potential_total_ev=np.zeros((2, 2, 2), dtype=float),
    )
    with pytest.raises(UepError, match="one Cartesian position per structure species"):
        write_potential_grid_xsf(tmp_path / "out.xsf", structure=structure, potential=grid)


# ---------------------------------------------------------------------------
# parse_formatted_density error paths + happy path
# ---------------------------------------------------------------------------


_GOOD_DEN_FMT = """ BEGIN header

           Real Lattice(A)
     2.0000000     0.0000000     0.0000000
     0.0000000     2.0000000     0.0000000
     0.0000000     0.0000000     2.0000000

   1                            ! nspins
   2     2     2                ! fine FFT grid along <a,b,c>
 END header
     1     1     1           1.0
     2     1     1           1.0
     1     2     1           1.0
     2     2     1           1.0
     1     1     2           1.0
     2     1     2           1.0
     1     2     2           1.0
     2     2     2           1.0
"""


def test_parse_formatted_density_reads_synthetic_grid(tmp_path: Path) -> None:
    path = tmp_path / "good.den_fmt"
    path.write_text(_GOOD_DEN_FMT, encoding="utf-8")
    density = parse_formatted_density(path)
    assert density.grid == (2, 2, 2)
    assert density.nspins == 1
    assert density.values.shape == (2, 2, 2, 1)
    assert float(np.sum(density.values)) == pytest.approx(8.0)


def test_parse_formatted_density_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(UepError, match="Failed to read formatted density"):
        parse_formatted_density(tmp_path / "missing.den_fmt")


def test_parse_formatted_density_rejects_incomplete_header(tmp_path: Path) -> None:
    path = tmp_path / "no_header.den_fmt"
    path.write_text(" BEGIN header\n   1   ! nspins\n END header\n", encoding="utf-8")
    with pytest.raises(UepError, match="complete CASTEP header"):
        parse_formatted_density(path)


def test_parse_formatted_density_rejects_bad_lattice_row(tmp_path: Path) -> None:
    text = " BEGIN header\n           Real Lattice(A)\n     2.0 0.0\n     0.0 2.0 0.0\n     0.0 0.0 2.0\n END header\n"
    path = tmp_path / "bad_lat.den_fmt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UepError, match="lattice vectors"):
        parse_formatted_density(path)


def test_parse_formatted_density_rejects_bad_nspins(tmp_path: Path) -> None:
    text = " BEGIN header\nxx                            ! nspins\n END header\n"
    path = tmp_path / "bad_nspins.den_fmt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UepError, match="nspins"):
        parse_formatted_density(path)


def test_parse_formatted_density_rejects_index_out_of_range(tmp_path: Path) -> None:
    text = _GOOD_DEN_FMT.replace("     2     2     2           1.0", "     9     1     1           1.0")
    path = tmp_path / "oor.den_fmt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UepError, match="out of range"):
        parse_formatted_density(path)


def test_parse_formatted_density_rejects_non_numeric_row(tmp_path: Path) -> None:
    text = _GOOD_DEN_FMT.replace("     2     2     2           1.0", "     2     2     2           bad")
    path = tmp_path / "nonnum.den_fmt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UepError, match="Invalid density row"):
        parse_formatted_density(path)


def test_parse_formatted_density_rejects_incomplete_data(tmp_path: Path) -> None:
    text = _GOOD_DEN_FMT.replace("     2     2     2           1.0\n", "")
    path = tmp_path / "short.den_fmt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UepError, match="incomplete"):
        parse_formatted_density(path)


# ---------------------------------------------------------------------------
# parse_castep_pseudopotentials + castep_total_charge
# ---------------------------------------------------------------------------


def test_parse_castep_pseudopotentials_reads_fixture() -> None:
    parsed = parse_castep_pseudopotentials(_fixture_path("Si2.castep"))
    assert "Si" in parsed
    assert parsed["Si"].ionic_charge == pytest.approx(4.0)
    assert parsed["Si"].core_radius_ang > 0.0


def test_parse_castep_pseudopotentials_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(UepError, match="Failed to read CASTEP file"):
        parse_castep_pseudopotentials(tmp_path / "missing.castep")


def test_parse_castep_pseudopotentials_without_report_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.castep"
    path.write_text("no pseudopotential data here\n", encoding="utf-8")
    with pytest.raises(UepError, match="Could not find ionic pseudopotential metadata"):
        parse_castep_pseudopotentials(path)


def test_castep_total_charge_detects_neutral_marker(tmp_path: Path) -> None:
    path = tmp_path / "neutral.castep"
    path.write_text("The unit cell is neutral as expected\n", encoding="utf-8")
    assert castep_total_charge(path) == 0.0


def test_castep_total_charge_returns_last_reported_value(tmp_path: Path) -> None:
    path = tmp_path / "charged.castep"
    path.write_text("total charge of cell = 1.0\nfinal total charge = -0.5\n", encoding="utf-8")
    assert castep_total_charge(path) == pytest.approx(-0.5)


def test_castep_total_charge_returns_none_without_markers(tmp_path: Path) -> None:
    path = tmp_path / "blank.castep"
    path.write_text("nothing relevant\n", encoding="utf-8")
    assert castep_total_charge(path) is None


def test_castep_total_charge_missing_file_returns_none(tmp_path: Path) -> None:
    assert castep_total_charge(tmp_path / "missing.castep") is None


# ---------------------------------------------------------------------------
# load_uep_source guards
# ---------------------------------------------------------------------------


def test_load_uep_source_rejects_non_positive_gw_factor() -> None:
    with pytest.raises(UepError, match="gw_factor must be positive"):
        load_uep_source(
            den_fmt_path=_fixture_path("Si2.den_fmt"),
            castep_path=_fixture_path("Si2.castep"),
            cell_path=_fixture_path("Si2.cell"),
            gw_factor=0.0,
        )


def test_load_uep_source_reports_missing_castep_structure(tmp_path: Path) -> None:
    bad_castep = tmp_path / "broken.castep"
    bad_castep.write_text("not a real castep file\n", encoding="utf-8")
    with pytest.raises(UepError, match="Failed to read host structure"):
        load_uep_source(
            den_fmt_path=_fixture_path("Si2.den_fmt"),
            castep_path=bad_castep,
        )


# ---------------------------------------------------------------------------
# UepField construction guards and array probing
# ---------------------------------------------------------------------------


def _si2_field() -> UepField:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    return UepField(source)


def test_uep_field_rejects_zero_density() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    zeroed_density = type(source.density)(
        lattice_vectors=source.density.lattice_vectors,
        grid=source.density.grid,
        nspins=source.density.nspins,
        values=np.zeros_like(source.density.values),
    )
    broken = UepSourceData(
        density=zeroed_density,
        structure=source.structure,
        pseudopotentials=source.pseudopotentials,
        castep_path=source.castep_path,
        den_fmt_path=source.den_fmt_path,
        cell_path=source.cell_path,
        gw_factor=source.gw_factor,
    )
    with pytest.raises(UepError, match="integrates to zero"):
        UepField(broken)


def test_uep_field_rejects_non_positive_ionic_charge() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    zero_pps = tuple(
        IonicPseudopotential(species=pp.species, ionic_charge=0.0, core_radius_ang=pp.core_radius_ang)
        for pp in source.pseudopotentials
    )
    broken = UepSourceData(
        density=source.density,
        structure=source.structure,
        pseudopotentials=zero_pps,
        castep_path=source.castep_path,
        den_fmt_path=source.den_fmt_path,
        cell_path=source.cell_path,
        gw_factor=source.gw_factor,
    )
    with pytest.raises(UepError, match="ionic charge must be positive"):
        UepField(broken)


def test_uep_field_validate_points_rejects_wrong_1d_shape() -> None:
    field = _si2_field()
    with pytest.raises(UepError, match="exactly three components"):
        field.potential(np.array([1.0, 2.0], dtype=float))


def test_uep_field_validate_points_rejects_wrong_2d_shape() -> None:
    field = _si2_field()
    with pytest.raises(UepError, match=r"shape \(N, 3\)"):
        field.potential(np.zeros((4, 2), dtype=float))


def test_uep_field_batched_evaluations_share_single_point_values() -> None:
    field = _si2_field()
    point = np.array([1.0, 1.2, 1.4], dtype=float)
    batch = np.vstack([point, point])
    pot_total, pot_e, pot_i = field.potential(batch)
    rho_total, rho_e, rho_i = field.charge_density(batch)
    assert pot_total.shape == (2,)
    assert pot_total[0] == pytest.approx(pot_total[1])
    assert pot_e[0] + pot_i[0] == pytest.approx(pot_total[0])
    assert rho_e[0] + rho_i[0] == pytest.approx(rho_total[0])


def test_uep_field_fractional_position_wraps_into_unit_cell() -> None:
    field = _si2_field()
    fractional = field.fractional_position((10.0, -5.0, 2.0))
    assert all(0.0 <= value < 1.0 for value in fractional)


# ---------------------------------------------------------------------------
# _periodic_resample_axis + preview_potential_grid
# ---------------------------------------------------------------------------


def test_periodic_resample_axis_rejects_out_of_range_axis() -> None:
    with pytest.raises(UepError, match="axis is out of range"):
        _periodic_resample_axis(np.zeros((4, 4)), 3, axis=5)


def test_periodic_resample_axis_rejects_tiny_target() -> None:
    with pytest.raises(UepError, match="at least two points"):
        _periodic_resample_axis(np.zeros((4, 4)), 1, axis=0)


def test_periodic_resample_axis_returns_copy_when_sizes_match() -> None:
    source = np.arange(12, dtype=float).reshape((3, 4))
    resampled = _periodic_resample_axis(source, 3, axis=0)
    assert resampled.shape == source.shape
    assert resampled is not source
    assert np.allclose(resampled, source)


def test_preview_potential_grid_rejects_thin_grid() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(1, 4, 4),
        potential_total_ev=np.zeros((1, 4, 4), dtype=float),
    )
    with pytest.raises(UepError, match="at least two points per axis"):
        preview_potential_grid(grid)


def test_preview_potential_grid_returns_unchanged_when_below_cap() -> None:
    values = np.arange(8, dtype=float).reshape((2, 2, 2))
    grid = UepPotentialGrid(lattice_vectors=_CUBE_LATTICE, grid=(2, 2, 2), potential_total_ev=values)
    preview = preview_potential_grid(grid, max_grid_points=12)
    assert preview.grid == (2, 2, 2)
    assert preview.potential_total_ev is not values
    assert np.allclose(preview.potential_total_ev, values)


def test_preview_potential_grid_handles_flat_resample() -> None:
    values = np.full((8, 8, 8), 3.0, dtype=float)
    grid = UepPotentialGrid(lattice_vectors=_CUBE_LATTICE, grid=(8, 8, 8), potential_total_ev=values)
    preview = preview_potential_grid(grid, max_grid_points=4)
    assert preview.grid == (4, 4, 4)
    assert np.allclose(preview.potential_total_ev, 3.0)


# ---------------------------------------------------------------------------
# coordinate transforms and plane helpers
# ---------------------------------------------------------------------------


def test_cartesian_to_fractional_round_trip() -> None:
    cartesian = np.array([[1.0, 2.0, 3.0], [0.5, 0.0, 1.5]], dtype=float)
    fractional = _cartesian_to_fractional(_CUBE_LATTICE, cartesian)
    back = _fractional_to_cartesian(_CUBE_LATTICE, fractional)
    assert np.allclose(back, cartesian)


def test_cartesian_to_fractional_rejects_bad_lattice() -> None:
    with pytest.raises(UepError, match="three lattice vectors"):
        _cartesian_to_fractional(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), np.zeros((1, 3)))  # type: ignore[arg-type]


def test_cartesian_to_fractional_rejects_bad_points() -> None:
    with pytest.raises(UepError, match="Cartesian 3-vectors"):
        _cartesian_to_fractional(_CUBE_LATTICE, np.zeros((2, 2)))


def test_fractional_to_cartesian_rejects_bad_shape() -> None:
    with pytest.raises(UepError, match="3-vectors"):
        _fractional_to_cartesian(_CUBE_LATTICE, np.zeros((2, 4)))


def test_fractional_point_to_cartesian_rejects_bad_shape() -> None:
    with pytest.raises(UepError, match="3-vectors"):
        fractional_point_to_cartesian(_CUBE_LATTICE, np.zeros(4))


def test_miller_indices_to_cartesian_normal_rejects_bad_lattice() -> None:
    with pytest.raises(UepError, match="three lattice vectors"):
        miller_indices_to_cartesian_normal(((1.0, 0.0, 0.0),), (1.0, 0.0, 0.0))  # type: ignore[arg-type]


def test_miller_indices_to_cartesian_normal_rejects_bad_indices_shape() -> None:
    with pytest.raises(UepError, match="Miller indices must be 3-vectors"):
        miller_indices_to_cartesian_normal(_CUBE_LATTICE, (1.0, 0.0))


def test_miller_indices_to_cartesian_normal_rejects_zero_indices() -> None:
    with pytest.raises(UepError, match="must not all be zero"):
        miller_indices_to_cartesian_normal(_CUBE_LATTICE, (0.0, 0.0, 0.0))


def test_cartesian_normal_to_lattice_plane_components_round_trips_axis() -> None:
    components = cartesian_normal_to_lattice_plane_components(_CUBE_LATTICE, (0.0, 0.0, 1.0))
    assert components == pytest.approx((0.0, 0.0, 1.0))


def test_cartesian_normal_to_lattice_plane_components_rejects_bad_lattice() -> None:
    with pytest.raises(UepError, match="three lattice vectors"):
        cartesian_normal_to_lattice_plane_components(((1.0, 0.0, 0.0),), (0.0, 0.0, 1.0))  # type: ignore[arg-type]


def test_normalize_plane_normal_rejects_bad_shape() -> None:
    with pytest.raises(UepError, match="Plane normals must be 3-vectors"):
        normalize_plane_normal((1.0, 0.0))


def test_normalize_plane_normal_rejects_zero_length() -> None:
    with pytest.raises(UepError, match="non-zero length"):
        normalize_plane_normal((0.0, 0.0, 0.0))


def test_plane_basis_from_normal_is_orthonormal_for_x_aligned_normal() -> None:
    basis_u, basis_v = plane_basis_from_normal((1.0, 0.0, 0.0))
    u = np.asarray(basis_u, dtype=float)
    v = np.asarray(basis_v, dtype=float)
    assert float(np.linalg.norm(u)) == pytest.approx(1.0)
    assert float(np.linalg.norm(v)) == pytest.approx(1.0)
    assert float(np.dot(u, v)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.dot(u, [1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-9)


def test_plane_from_point_and_normal_rejects_bad_anchor() -> None:
    with pytest.raises(UepError, match="anchors must be 3-vectors"):
        plane_from_point_and_normal(
            anchor_ang=(0.0, 0.0),
            normal_ang=(0.0, 0.0, 1.0),
            lattice_vectors=_CUBE_LATTICE,
        )


def test_plane_from_best_fit_rejects_too_few_points() -> None:
    with pytest.raises(UepError, match="at least four"):
        plane_from_best_fit(
            points_ang=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            lattice_vectors=_CUBE_LATTICE,
        )


def test_plane_from_best_fit_rejects_collinear_points() -> None:
    with pytest.raises(UepError, match="collinear"):
        plane_from_best_fit(
            points_ang=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
            ],
            lattice_vectors=_CUBE_LATTICE,
        )


# ---------------------------------------------------------------------------
# trilinear interpolation + grid slice guards
# ---------------------------------------------------------------------------


def _ramp_grid() -> UepPotentialGrid:
    return UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
        grid=(4, 4, 4),
        potential_total_ev=np.arange(64, dtype=float).reshape((4, 4, 4)),
    )


def test_interpolate_periodic_grid_trilinear_rejects_thin_grid() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(1, 4, 4),
        potential_total_ev=np.zeros((1, 4, 4), dtype=float),
    )
    with pytest.raises(UepError, match="at least two points per axis"):
        _interpolate_periodic_grid_trilinear(grid, np.zeros((1, 3), dtype=float))


def test_interpolate_periodic_grid_trilinear_matches_grid_nodes() -> None:
    grid = _ramp_grid()
    sampled = _interpolate_periodic_grid_trilinear(grid, np.array([[0.0, 0.0, 0.0]], dtype=float))
    assert sampled[0] == pytest.approx(0.0)


def test_sample_grid_slice_rejects_thin_plane() -> None:
    plane = UepSlicePlane(
        anchor_ang=(0.0, 0.0, 0.0),
        normal_ang=(0.0, 0.0, 1.0),
        basis_u_ang=(1.0, 0.0, 0.0),
        basis_v_ang=(0.0, 1.0, 0.0),
        extent_u_ang=(0.0, 1.0),
        extent_v_ang=(0.0, 1.0),
        points_u=1,
        points_v=5,
    )
    with pytest.raises(UepError, match="at least two points"):
        sample_grid_slice(_ramp_grid(), plane)


# ---------------------------------------------------------------------------
# isosurface internals
# ---------------------------------------------------------------------------


def test_interpolate_isosurface_vertex_handles_equal_values() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])
    midpoint = _interpolate_isosurface_vertex(a, b, 1.0, 1.0, level_ev=1.0)
    assert np.allclose(midpoint, [1.0, 0.0, 0.0])


def test_interpolate_isosurface_vertex_interpolates_linearly() -> None:
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])
    vertex = _interpolate_isosurface_vertex(a, b, 0.0, 2.0, level_ev=1.0)
    assert np.allclose(vertex, [1.0, 0.0, 0.0])


def test_order_polygon_vertices_returns_small_inputs_unchanged() -> None:
    points = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    assert _order_polygon_vertices(points) is points


def test_order_polygon_vertices_sorts_square_into_cyclic_order() -> None:
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    ordered = _order_polygon_vertices(points)
    assert len(ordered) == 4
    # Consecutive edges of a convex polygon never cross the centre twice; check
    # that the winding around the centre is monotonic in angle.
    center = np.mean(ordered, axis=0)
    angles = [math.atan2(p[1] - center[1], p[0] - center[0]) for p in ordered]
    diffs = np.diff(np.unwrap(angles))
    assert np.all(diffs > 0) or np.all(diffs < 0)


def test_order_polygon_vertices_returns_collinear_points_unchanged() -> None:
    points = [
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([3.0, 0.0, 0.0]),
    ]
    assert _order_polygon_vertices(points) is points


def test_polygonize_tetrahedron_returns_empty_when_level_outside() -> None:
    points = (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    triangles = _polygonize_tetrahedron(points, (0.0, 0.1, 0.2, 0.3), level_ev=5.0)
    assert triangles == []


def test_polygonize_tetrahedron_emits_triangle_for_single_crossing() -> None:
    points = (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    triangles = _polygonize_tetrahedron(points, (-1.0, 1.0, 1.0, 1.0), level_ev=0.0)
    assert len(triangles) == 1
    assert all(vertex.shape == (3,) for vertex in triangles[0])


def test_polygonize_tetrahedron_triangulates_quad_cross_section() -> None:
    points = (
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    # Two vertices below, two above the level produces a quadrilateral slice.
    triangles = _polygonize_tetrahedron(points, (-1.0, -1.0, 1.0, 1.0), level_ev=0.0)
    assert len(triangles) == 2


def test_reshape_pyvista_faces_stops_on_truncated_buffer() -> None:
    faces = _reshape_pyvista_faces(np.array([3, 0, 1, 2, 4, 5, 6], dtype=int))
    assert faces.tolist() == [[0, 1, 2]]


# ---------------------------------------------------------------------------
# extract_isosurface fallback path (force pyvista off)
# ---------------------------------------------------------------------------


def _linear_ramp_grid() -> UepPotentialGrid:
    values = np.linspace(0.0, 1.0, 4 * 4 * 4, dtype=float).reshape((4, 4, 4))
    return UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(4, 4, 4),
        potential_total_ev=values,
    )


def test_extract_isosurface_fallback_builds_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mufinder.science.uep._pyvista_modules", lambda: None)
    grid = _linear_ramp_grid()
    level = 0.5 * (float(grid.potential_total_ev.min()) + float(grid.potential_total_ev.max()))
    mesh = extract_isosurface(grid, level_ev=level)
    assert isinstance(mesh, UepIsosurfaceMesh)
    assert len(mesh.vertices_ang) >= 3
    assert len(mesh.faces) >= 1


def test_extract_isosurface_rejects_thin_grid() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(1, 4, 4),
        potential_total_ev=np.zeros((1, 4, 4), dtype=float),
    )
    with pytest.raises(UepError, match="at least two points per axis"):
        extract_isosurface(grid, level_ev=0.0)


def test_extract_isosurface_pyvista_returns_none_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EmptyContour:
        n_cells = 0

    class _FakeStructuredGrid:
        def __init__(self, *_coords) -> None:
            pass

        def __setitem__(self, _name: str, _values: np.ndarray) -> None:
            pass

        def contour(self, *, isosurfaces, scalars):  # noqa: ARG002
            return _EmptyContour()

    class _FakePyVista:
        StructuredGrid = _FakeStructuredGrid

    monkeypatch.setattr("mufinder.science.uep._pyvista_modules", lambda: _FakePyVista())
    grid = _linear_ramp_grid()
    level = 0.5 * (float(grid.potential_total_ev.min()) + float(grid.potential_total_ev.max()))
    mesh = extract_isosurface(grid, level_ev=level)
    # When pyvista yields nothing the analytic fallback still produces a mesh.
    assert isinstance(mesh, UepIsosurfaceMesh)
    assert len(mesh.faces) >= 1


# ---------------------------------------------------------------------------
# classification + zero-point energy
# ---------------------------------------------------------------------------


def test_classify_stationary_point_labels_all_categories() -> None:
    assert classify_stationary_point(np.diag([1.0, 1.0, 1.0])).label == "minimum"
    assert classify_stationary_point(np.diag([-1.0, -1.0, -1.0])).label == "maximum"
    assert classify_stationary_point(np.diag([1.0, -1.0, 1.0])).label == "saddle"
    flat = classify_stationary_point(np.diag([0.0, 1.0, 1.0]))
    assert flat.label == "flat"
    assert flat.near_zero_count == 1


def test_zero_point_energy_returns_none_for_unstable_extremum() -> None:
    assert zero_point_energy_ev(np.diag([-1.0, 1.0, 1.0])) is None


def test_zero_point_energy_positive_for_minimum() -> None:
    value = zero_point_energy_ev(np.diag([1.0, 1.0, 1.0]))
    assert value is not None
    assert value > 0.0


def test_zero_point_energy_rejects_non_positive_mass() -> None:
    with pytest.raises(UepError, match="particle_mass_kg must be positive"):
        zero_point_energy_ev(np.diag([1.0, 1.0, 1.0]), particle_mass_kg=0.0)


def test_particle_mass_kg_from_amu_scales_linearly() -> None:
    one = particle_mass_kg_from_amu(1.0)
    two = particle_mass_kg_from_amu(2.0)
    assert two == pytest.approx(2.0 * one)
    assert one > 0.0


# ---------------------------------------------------------------------------
# sampling guards
# ---------------------------------------------------------------------------


def test_sample_line_rejects_too_few_points() -> None:
    field = _si2_field()
    with pytest.raises(UepError, match="at least two points"):
        sample_line(field, start_ang=(0.0, 0.0, 0.0), end_ang=(1.0, 1.0, 1.0), num_points=1)


def test_sample_plane_rejects_too_few_points() -> None:
    field = _si2_field()
    with pytest.raises(UepError, match="at least two points along each axis"):
        sample_plane(
            field,
            origin_ang=(0.0, 0.0, 0.0),
            corner_x_ang=(1.0, 0.0, 0.0),
            corner_y_ang=(0.0, 1.0, 0.0),
            points_x=1,
            points_y=3,
        )


def test_sample_plane_rejects_zero_span_axes() -> None:
    field = _si2_field()
    with pytest.raises(UepError, match="non-zero patch"):
        sample_plane(
            field,
            origin_ang=(0.0, 0.0, 0.0),
            corner_x_ang=(0.0, 0.0, 0.0),
            corner_y_ang=(0.0, 1.0, 0.0),
            points_x=3,
            points_y=3,
        )


# ---------------------------------------------------------------------------
# legacy clustering guards and helpers
# ---------------------------------------------------------------------------


def test_cluster_optimized_positions_rejects_non_positive_threshold() -> None:
    with pytest.raises(UepError, match="threshold_ang must be positive"):
        cluster_optimized_positions([], threshold_ang=0.0)


def test_cluster_optimized_positions_empty_returns_no_clusters() -> None:
    summary = cluster_optimized_positions([], threshold_ang=0.2)
    assert summary.clusters == ()
    assert summary.clustering_linkage_method == "average"


def test_cluster_optimized_positions_single_entry_returns_one_cluster() -> None:
    summary = cluster_optimized_positions(
        [_fake_optimization((0.25, 0.25, 0.25), classical=-1.0, total=-1.0)],
        threshold_ang=0.2,
    )
    assert len(summary.clusters) == 1
    assert summary.clusters[0].member_indexes == (0,)
    assert summary.clusters[0].standard_deviation_ev == pytest.approx(0.0)


def test_clustering_energy_prefers_total_then_classical() -> None:
    with_total = _fake_optimization((0.1, 0.1, 0.1), classical=2.0, total=-1.0)
    assert _clustering_energy(with_total) == pytest.approx(-1.0)
    without_total = _fake_optimization((0.1, 0.1, 0.1), classical=3.0, total=None)
    assert _clustering_energy(without_total) == pytest.approx(3.0)


def test_clustering_energy_rejects_non_finite_energies() -> None:
    entry = _fake_optimization((0.1, 0.1, 0.1), classical=float("nan"), total=None)
    with pytest.raises(UepError, match="finite classical or total energy"):
        _clustering_energy(entry)


def test_relaxed_site_number_map_orders_by_classical_energy() -> None:
    optimizations = [
        _fake_optimization((0.1, 0.1, 0.1), classical=2.0, total=2.0),
        _fake_optimization((0.2, 0.2, 0.2), classical=-1.0, total=-1.0),
        _fake_optimization((0.3, 0.3, 0.3), classical=float("nan"), total=None),
    ]
    mapping = relaxed_site_number_map(optimizations)
    # Lowest classical energy -> site 1, NaN energy pushed to the end.
    assert mapping[1] == 1
    assert mapping[0] == 2
    assert mapping[2] == 3


def test_cluster_average_and_minimum_classical_energy_handle_nan() -> None:
    optimizations = [
        _fake_optimization((0.1, 0.1, 0.1), classical=1.0, total=1.0),
        _fake_optimization((0.2, 0.2, 0.2), classical=3.0, total=3.0),
    ]
    cluster = cluster_optimized_positions(optimizations, threshold_ang=10.0).clusters[0]
    assert _cluster_average_classical_energy(cluster, optimizations) == pytest.approx(2.0)
    assert _cluster_minimum_classical_energy(cluster, optimizations) == pytest.approx(1.0)


def test_cluster_average_classical_energy_all_nan_returns_inf() -> None:
    from mufinder.science.uep import UepClusterSummary

    optimizations = [_fake_optimization((0.1, 0.1, 0.1), classical=float("nan"), total=None)]
    cluster = UepClusterSummary(
        cluster_index=1,
        size=1,
        representative_index=0,
        representative_position_ang=(0.0, 0.0, 0.0),
        representative_fractional_position=(0.0, 0.0, 0.0),
        minimum_energy_ev=None,
        average_energy_ev=None,
        standard_deviation_ev=None,
        member_indexes=(0,),
    )
    assert math.isinf(_cluster_average_classical_energy(cluster, optimizations))
    assert math.isinf(_cluster_minimum_classical_energy(cluster, optimizations))


# ---------------------------------------------------------------------------
# similarity-summary helpers
# ---------------------------------------------------------------------------


def test_cluster_similarity_summaries_single_group_is_empty() -> None:
    assert _cluster_similarity_summaries([np.array([0])], np.zeros((1, 4))) == ()


def test_cluster_similarity_summaries_ranks_by_distance() -> None:
    genome = np.array([[0.0], [1.0], [5.0]], dtype=float)
    groups = [np.array([0]), np.array([1]), np.array([2])]
    summaries = _cluster_similarity_summaries(groups, genome)
    assert len(summaries) == 3
    distances = [s.distance for s in summaries]
    assert distances == sorted(distances)


def test_renumber_cluster_similarity_summaries_remaps_and_sorts() -> None:
    similarities = (
        UepClusterSimilaritySummary(left_cluster_index=2, right_cluster_index=1, distance=3.0),
        UepClusterSimilaritySummary(left_cluster_index=1, right_cluster_index=2, distance=1.0),
    )
    remapped = _renumber_cluster_similarity_summaries(similarities, {1: 5, 2: 1})
    assert remapped[0].distance == pytest.approx(1.0)
    assert remapped[0].left_cluster_index <= remapped[0].right_cluster_index


def test_cluster_groups_from_labels_groups_by_id() -> None:
    groups = _cluster_groups_from_labels(np.array([2, 1, 2, 1]))
    assert [g.tolist() for g in groups] == [[1, 3], [0, 2]]


def test_asymmetric_images_centre_avoids_high_symmetry() -> None:
    optimizations = [
        _fake_optimization((0.5, 0.5, 0.5), classical=-1.0, total=-1.0),
        _fake_optimization((0.1, 0.1, 0.1), classical=0.0, total=0.0),
    ]
    centre = _asymmetric_images_centre(optimizations)
    assert centre.shape == (3,)
    # The reference (lowest-energy) site at (0.5, 0.5, 0.5) is jittered away
    # from the exact high-symmetry centre.
    assert np.all(centre != np.array([0.5, 0.5, 0.5]))
    assert np.all(np.abs(centre - 0.5) < 1e-3)


# ---------------------------------------------------------------------------
# clustering config validation + structure adapter
# ---------------------------------------------------------------------------


def test_validate_clustering_config_rejects_unknown_method() -> None:
    with pytest.raises(UepError, match="must be 'hier' or 'kmeans'"):
        _validate_clustering_config(UepClusteringConfig(clustering_method="dbscan"))


def test_validate_clustering_config_rejects_non_positive_hier_t() -> None:
    with pytest.raises(UepError, match="clustering_hier_t must be positive"):
        _validate_clustering_config(UepClusteringConfig(clustering_hier_t=0.0))


def test_validate_clustering_config_rejects_non_positive_kmeans_k() -> None:
    with pytest.raises(UepError, match="clustering_kmeans_k must be positive"):
        _validate_clustering_config(UepClusteringConfig(clustering_kmeans_k=0))


def test_validate_clustering_config_normalizes_fields() -> None:
    active = _validate_clustering_config(
        UepClusteringConfig(clustering_method="HIER", clustering_linkage_method="  Single  ")
    )
    assert active.clustering_method == "hier"
    assert active.clustering_linkage_method == "single"


def test_validate_clustering_config_defaults_blank_linkage_method() -> None:
    active = _validate_clustering_config(UepClusteringConfig(clustering_linkage_method="   "))
    assert active.clustering_linkage_method == "single"


def test_structure_like_to_ase_atoms_requires_full_metadata() -> None:
    class _Partial:
        lattice_vectors = _CUBE_LATTICE
        species = ("Si",)
        fractional_positions = None

    with pytest.raises(UepError, match="must define lattice_vectors, species"):
        _structure_like_to_ase_atoms(_Partial())


def test_structure_like_to_ase_atoms_requires_matching_lengths() -> None:
    class _Mismatched:
        lattice_vectors = _CUBE_LATTICE
        species = ("Si", "Si")
        fractional_positions = ((0.0, 0.0, 0.0),)

    with pytest.raises(UepError, match="matching lengths"):
        _structure_like_to_ase_atoms(_Mismatched())


def test_structure_like_to_ase_atoms_builds_periodic_atoms() -> None:
    atoms = _structure_like_to_ase_atoms(_simple_structure())
    assert atoms.get_chemical_symbols() == ["Si"]
    assert bool(np.all(atoms.pbc))


# ---------------------------------------------------------------------------
# prepared/empty clustering datasets (no Soprano needed)
# ---------------------------------------------------------------------------


def test_prepare_uep_clustering_dataset_empty_returns_empty_features() -> None:
    prepared = prepare_uep_clustering_dataset([], source_structure=_simple_structure())
    assert prepared.optimizations == ()
    assert prepared.genome_vectors == ()
    assert prepared.linkage_matrix == ()


def test_cluster_prepared_uep_dataset_empty_returns_empty_result() -> None:
    prepared = UepPreparedClusteringDataset(
        config=UepClusteringConfig(),
        optimizations=(),
        asymmetric_fractional_positions=(),
        genome_vectors=(),
        normalized_genome_vectors=(),
        linkage_method="single",
        linkage_matrix=(),
    )
    result = cluster_prepared_uep_dataset(prepared, config=UepClusteringConfig(clustering_method="hier"))
    assert result.cluster_labels == ()
    assert result.summary.clusters == ()
    assert result.summary.clustering_method == "hier"


def test_cluster_prepared_uep_dataset_empty_kmeans_reports_k() -> None:
    prepared = UepPreparedClusteringDataset(
        config=UepClusteringConfig(),
        optimizations=(),
        asymmetric_fractional_positions=(),
        genome_vectors=(),
        normalized_genome_vectors=(),
        linkage_method="single",
        linkage_matrix=(),
    )
    result = cluster_prepared_uep_dataset(
        prepared,
        config=UepClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=3),
    )
    assert result.summary.clustering_method == "kmeans"
    assert result.summary.clustering_kmeans_k == 3


def test_cluster_prepared_uep_dataset_kmeans_rejects_too_many_clusters() -> None:
    prepared = UepPreparedClusteringDataset(
        config=UepClusteringConfig(),
        optimizations=(_fake_optimization((0.1, 0.1, 0.1), classical=-1.0, total=-1.0),),
        asymmetric_fractional_positions=((0.1, 0.1, 0.1),),
        genome_vectors=((-1.0, 0.1, 0.1, 0.1),),
        normalized_genome_vectors=((0.0, 0.0, 0.0, 0.0),),
        linkage_method="single",
        linkage_matrix=(),
    )
    with pytest.raises(UepError, match="must not exceed the number of optimized sites"):
        cluster_prepared_uep_dataset(
            prepared,
            config=UepClusteringConfig(clustering_method="kmeans", clustering_kmeans_k=5),
        )


def test_cluster_prepared_uep_dataset_single_hier_returns_one_cluster() -> None:
    prepared = UepPreparedClusteringDataset(
        config=UepClusteringConfig(),
        optimizations=(_fake_optimization((0.25, 0.25, 0.25), classical=-1.0, total=-1.0),),
        asymmetric_fractional_positions=((0.25, 0.25, 0.25),),
        genome_vectors=((-1.0, 0.25, 0.25, 0.25),),
        normalized_genome_vectors=((0.0, 0.0, 0.0, 0.0),),
        linkage_method="single",
        linkage_matrix=(),
    )
    result = cluster_prepared_uep_dataset(
        prepared,
        config=UepClusteringConfig(clustering_method="hier", clustering_hier_t=0.3),
    )
    assert result.cluster_labels == (1,)
    assert len(result.summary.clusters) == 1
    assert result.summary.clusters[0].member_indexes == (0,)


# ---------------------------------------------------------------------------
# potential_grid sanity (real field) for the FFT ifft path
# ---------------------------------------------------------------------------


def test_potential_grid_matches_direct_potential_at_a_grid_node() -> None:
    field = _si2_field()
    grid = potential_grid(field)
    assert grid.grid == field.source.density.grid
    # The (0,0,0) FFT node corresponds to the Cartesian origin.
    direct = float(field.potential(np.zeros(3, dtype=float))[0][0])
    assert float(grid.potential_total_ev[0, 0, 0]) == pytest.approx(direct, rel=1e-6, abs=1e-6)
