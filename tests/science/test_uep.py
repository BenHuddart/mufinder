"""Tests for MuFinder-native UEP source loading and field analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mufinder.science.uep import (
    UepError,
    UepField,
    UepIsosurfaceMesh,
    UepOptimizationResult,
    cluster_optimized_positions,
    extract_isosurface,
    load_uep_source,
    optimize_muon_position,
    potential_grid,
    probe_point,
    sample_line,
    sample_plane,
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


def test_optimize_muon_position_and_cluster_summary() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    field = UepField(source)

    first = optimize_muon_position(field, start_position_ang=(1.0, 1.0, 1.0), max_steps=10)
    second = optimize_muon_position(field, start_position_ang=(1.05, 1.0, 1.0), max_steps=10)

    assert isinstance(first, UepOptimizationResult)
    assert isinstance(second, UepOptimizationResult)

    summary = cluster_optimized_positions([first, second], threshold_ang=0.3)
    assert len(summary.clusters) >= 1
    assert summary.clusters[0].size >= 1


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
