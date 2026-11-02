"""Tests for MuFinder-native UEP source loading and field analysis."""

from __future__ import annotations

from dataclasses import asdict
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mufinder.science import uep as uep_module
from mufinder.science.uep import (
    DEFAULT_OPTIMIZER_MAX_STEPS,
    IonicPseudopotential,
    StationaryPointClassification,
    UepError,
    UepField,
    UepGridSliceSample,
    UepIsosurfaceMesh,
    UepOptimizationResult,
    UepPotentialGrid,
    UepSlicePlane,
    UepSourceData,
    UepStructure,
    cartesian_normal_to_lattice_plane_components,
    castep_total_charge,
    classify_stationary_point,
    deserialize_potential_grid,
    extract_isosurface,
    fractional_point_to_cartesian,
    load_uep_source,
    miller_indices_to_cartesian_normal,
    normalize_plane_normal,
    optimize_muon_position,
    parse_castep_pseudopotentials,
    parse_formatted_density,
    particle_mass_kg_from_amu,
    plane_basis_from_normal,
    plane_from_best_fit,
    plane_from_fractional_anchor_and_miller,
    plane_from_point_and_normal,
    plane_from_three_points,
    potential_grid,
    potential_above_minimum,
    potential_level_ceiling,
    ionic_gaussian_resolution,
    preview_potential_grid,
    probe_point,
    sample_grid_slice,
    sample_line,
    optimization_result_from_dict,
    classification_from_dict,
    sample_plane,
    serialize_potential_grid,
    write_potential_grid_xsf,
    zero_point_energy_ev,
)

_CUBE_LATTICE = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))


def _fixture_path(name: str) -> Path:
    return Path("tests/fixtures/uep") / name


def _si2_source() -> UepSourceData:
    return load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )


def _si2_field() -> UepField:
    return UepField(_si2_source())


def _simple_cubic_host() -> UepStructure:
    return UepStructure(
        lattice_vectors=_CUBE_LATTICE,
        species=("Si",),
        positions_ang=((0.0, 0.0, 0.0),),
        fractional_positions=((0.0, 0.0, 0.0),),
        atomic_spins=None,
    )


def _ramp_grid() -> UepPotentialGrid:
    return UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
        grid=(4, 4, 4),
        potential_total_ev=np.arange(64, dtype=float).reshape((4, 4, 4)),
    )


def _linear_ramp_grid() -> UepPotentialGrid:
    values = np.linspace(0.0, 1.0, 4 * 4 * 4, dtype=float).reshape((4, 4, 4))
    return UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(4, 4, 4),
        potential_total_ev=values,
    )


def _thin_grid() -> UepPotentialGrid:
    return UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(1, 4, 4),
        potential_total_ev=np.zeros((1, 4, 4), dtype=float),
    )


def _axis_plane(*, extent: float = 1.0, points_u: int = 2, points_v: int = 2) -> UepSlicePlane:
    return UepSlicePlane(
        anchor_ang=(0.0, 0.0, 0.0),
        normal_ang=(0.0, 0.0, 1.0),
        basis_u_ang=(1.0, 0.0, 0.0),
        basis_v_ang=(0.0, 1.0, 0.0),
        extent_u_ang=(0.0, extent),
        extent_v_ang=(0.0, extent),
        points_u=points_u,
        points_v=points_v,
    )


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


# CASTEP inputs: formatted density, pseudopotentials and cell charge


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
    text = _GOOD_DEN_FMT.replace(
        "     2     2     2           1.0", "     9     1     1           1.0"
    )
    path = tmp_path / "oor.den_fmt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(UepError, match="out of range"):
        parse_formatted_density(path)


def test_parse_formatted_density_rejects_non_numeric_row(tmp_path: Path) -> None:
    text = _GOOD_DEN_FMT.replace(
        "     2     2     2           1.0", "     2     2     2           bad"
    )
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


def test_ionic_pseudopotential_gaussian_width_property_raises() -> None:
    pp = IonicPseudopotential(species="Si", ionic_charge=4.0, core_radius_ang=1.0)
    with pytest.raises(AttributeError, match="gw_factor"):
        _ = pp.gaussian_width_ang


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


# Loading a UEP source


def test_load_uep_source_reads_neutral_fixture() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    assert source.structure.species == ("Si", "Si")
    assert source.gw_factor == pytest.approx(5.0)
    assert (
        source.neutrality_warning is None or "assuming a neutral host" in source.neutrality_warning
    )


def test_load_uep_source_uses_shared_mulliken_spin_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mufinder.science.uep.parse_castep_mulliken_atomic_spins", lambda _path: (1.2, -1.2)
    )

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


# Field evaluation


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
    assert translated_probe.potential_total_ev == pytest.approx(
        origin_probe.potential_total_ev, rel=1e-9
    )


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
        IonicPseudopotential(
            species=pp.species, ionic_charge=0.0, core_radius_ang=pp.core_radius_ang
        )
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


# Line and plane sampling


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


# Potential grids: serialisation, previews and XSF export


def test_potential_grid_matches_direct_potential_at_a_grid_node() -> None:
    field = _si2_field()
    grid = potential_grid(field)
    assert grid.grid == field.source.density.grid
    # The (0,0,0) FFT node corresponds to the Cartesian origin.
    direct = float(field.potential(np.zeros(3, dtype=float))[0][0])
    assert float(grid.potential_total_ev[0, 0, 0]) == pytest.approx(direct, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("gw_factor", [7.5, None], ids=["with-gw-factor", "without-gw-factor"])
def test_potential_grid_serialization_round_trips_gw_factor(gw_factor: float | None) -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 4.0)),
        grid=(2, 2, 2),
        potential_total_ev=np.arange(8, dtype=float).reshape((2, 2, 2)),
        gw_factor=gw_factor,
    )

    payload = serialize_potential_grid(grid)
    restored = deserialize_potential_grid(payload)

    assert payload["grid"] == [2, 2, 2]
    # A grid without a Gaussian-width factor serialises without the key.
    assert ("gw_factor" in payload) is (gw_factor is not None)
    assert restored is not None
    assert restored.gw_factor == pytest.approx(gw_factor)
    assert restored.potential_total_ev == pytest.approx(grid.potential_total_ev)


def test_preview_potential_grid_keeps_axes_already_under_the_cap() -> None:
    values = np.arange(8 * 8 * 2, dtype=float).reshape((8, 8, 2))
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE, grid=(8, 8, 2), potential_total_ev=values
    )

    preview = preview_potential_grid(grid, max_grid_points=4)

    assert preview.grid == (4, 4, 2)
    assert np.array_equal(preview.potential_total_ev, values[::2, ::2, :])


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {"lattice_vectors": [[1, 0, 0], [0, 1, 0]], "grid": [2, 2, 2], "potential_total_ev": []},
        {
            "lattice_vectors": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "grid": [2, 2],
            "potential_total_ev": [],
        },
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


def test_a_preview_keeps_the_grids_own_values_on_a_coarser_lattice() -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si8.den_fmt"),
        castep_path=_fixture_path("Si8.castep"),
        cell_path=_fixture_path("Si8.cell"),
    )
    grid = potential_grid(UepField(source))

    preview = preview_potential_grid(grid, max_grid_points=12)

    assert preview.grid == (12, 12, 12)
    assert np.array_equal(preview.potential_total_ev, grid.potential_total_ev[::3, ::3, ::3])


def test_a_preview_surface_sits_at_the_level_it_was_drawn_for() -> None:
    # The window read one level while a stretched, interpolated preview drew another.
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si8.den_fmt"),
        castep_path=_fixture_path("Si8.castep"),
        cell_path=_fixture_path("Si8.cell"),
    )
    field = UepField(source)
    grid = potential_grid(field)
    level = float(grid.potential_total_ev.min()) + 2.0

    mesh = extract_isosurface(preview_potential_grid(grid, max_grid_points=18), level_ev=level)

    vertices = np.asarray(mesh.vertices_ang)
    picked = vertices[np.random.default_rng(3).integers(0, len(vertices), 40)]
    assert np.abs(field.potential(picked)[0] - level).max() < 0.5


def test_an_axis_with_no_divisor_keeps_all_its_points_in_a_preview() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE, grid=(7, 8, 2), potential_total_ev=np.zeros((7, 8, 2))
    )

    assert preview_potential_grid(grid, max_grid_points=4).grid == (7, 4, 2)


def test_a_preview_axis_takes_the_divisor_closest_to_the_limit() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE, grid=(25, 58, 64), potential_total_ev=np.zeros((25, 58, 64))
    )

    assert preview_potential_grid(grid, max_grid_points=18).grid == (25, 29, 16)


def test_the_potential_above_its_minimum_starts_at_zero_and_keeps_its_differences() -> None:
    values = np.arange(8, dtype=float).reshape((2, 2, 2)) - 3.5
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE, grid=(2, 2, 2), potential_total_ev=values, gw_factor=5.0
    )

    shifted = potential_above_minimum(grid)

    assert shifted.potential_total_ev.min() == 0.0
    assert np.allclose(shifted.potential_total_ev - values, 3.5)
    assert shifted.gw_factor == 5.0


def test_the_level_ceiling_is_the_95th_percentile_unless_that_is_the_minimum() -> None:
    spiked = np.zeros((10, 10, 10))
    spiked.flat[:100] = np.linspace(1.0, 3.0, 100)
    spiked.flat[0] = 500.0
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE, grid=(10, 10, 10), potential_total_ev=spiked
    )
    assert potential_level_ceiling(grid) == pytest.approx(np.percentile(spiked, 95))
    assert potential_level_ceiling(grid) < 500.0

    one_spike = np.zeros((10, 10, 10))
    one_spike.flat[0] = 2.0
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE, grid=(10, 10, 10), potential_total_ev=one_spike
    )
    assert potential_level_ceiling(grid) == 2.0


@pytest.mark.parametrize("gw_factor", [5.0, 10.0])
def test_the_gaussian_resolution_follows_the_grid_and_the_width(gw_factor: float) -> None:
    source = load_uep_source(
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        gw_factor=gw_factor,
    )
    lattice = np.asarray(source.density.lattice_vectors, dtype=float)
    g_max = np.min(
        np.asarray(source.density.grid) // 2 * 2.0 * np.pi / np.linalg.norm(lattice, axis=1)
    )
    width = min(pp.core_radius_ang for pp in source.pseudopotentials) / gw_factor

    assert ionic_gaussian_resolution(UepField(source)) == pytest.approx(width * g_max)


def test_preview_potential_grid_returns_unchanged_when_below_cap() -> None:
    values = np.arange(8, dtype=float).reshape((2, 2, 2))
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE, grid=(2, 2, 2), potential_total_ev=values
    )
    preview = preview_potential_grid(grid, max_grid_points=12)
    assert preview.grid == (2, 2, 2)
    assert preview.potential_total_ev is not values
    assert np.allclose(preview.potential_total_ev, values)


def test_preview_potential_grid_rejects_thin_grid() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(1, 4, 4),
        potential_total_ev=np.zeros((1, 4, 4), dtype=float),
    )
    with pytest.raises(UepError, match="at least two points per axis"):
        preview_potential_grid(grid)


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
    dimension_line = next(
        line for line in lines if line.strip() == "BEGIN_DATAGRID_3D_UEP_POTENTIAL"
    )
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


def test_write_potential_grid_xsf_rejects_non_3d_grid(tmp_path: Path) -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(4,),
        potential_total_ev=np.zeros(4, dtype=float),
    )
    with pytest.raises(UepError, match="3D electrostatic-potential grid"):
        write_potential_grid_xsf(
            tmp_path / "out.xsf", structure=_simple_cubic_host(), potential=grid
        )


def test_write_potential_grid_xsf_rejects_bad_lattice(tmp_path: Path) -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # type: ignore[arg-type]
        grid=(2, 2, 2),
        potential_total_ev=np.zeros((2, 2, 2), dtype=float),
    )
    with pytest.raises(UepError, match="three lattice vectors"):
        write_potential_grid_xsf(
            tmp_path / "out.xsf", structure=_simple_cubic_host(), potential=grid
        )


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


# Coordinates and slice planes


def test_fractional_point_to_cartesian_scales_by_the_lattice() -> None:
    assert fractional_point_to_cartesian(_CUBE_LATTICE, (0.25, 0.5, 0.375)) == pytest.approx(
        (1.0, 2.0, 1.5)
    )


def test_fractional_point_to_cartesian_rejects_bad_lattice() -> None:
    with pytest.raises(UepError, match="three lattice vectors"):
        fractional_point_to_cartesian(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (0.0, 0.0, 0.0))  # type: ignore[arg-type]


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
    assert abs(
        float(
            np.dot(
                np.asarray(plane.normal_ang, dtype=float), np.array([0.0, 0.0, 1.0], dtype=float)
            )
        )
    ) == pytest.approx(1.0)


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
    assert plane.anchor_ang == pytest.approx(
        fractional_point_to_cartesian(lattice_vectors, (0.25, 0.5, 0.75))
    )
    assert plane.normal_ang == pytest.approx(
        miller_indices_to_cartesian_normal(lattice_vectors, (1.0, 0.0, 0.0))
    )
    assert float(np.dot(normal, np.asarray(lattice_vectors[1], dtype=float))) == pytest.approx(
        0.0, abs=1e-9
    )
    assert float(np.dot(normal, np.asarray(lattice_vectors[2], dtype=float))) == pytest.approx(
        0.0, abs=1e-9
    )


# Grid slices


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


def test_zero_shifted_grid_slice_preserves_shifted_bounds() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0)),
        grid=(4, 4, 4),
        potential_total_ev=np.linspace(-3.0, 5.0, 64, dtype=float).reshape((4, 4, 4)),
    )
    shifted = UepPotentialGrid(
        lattice_vectors=grid.lattice_vectors,
        grid=grid.grid,
        potential_total_ev=np.asarray(grid.potential_total_ev, dtype=float)
        - float(np.min(grid.potential_total_ev)),
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


def test_sample_grid_slice_reproduces_grid_node_values() -> None:
    # A 2 Å cell on a 4-point grid has nodes every 0.5 Å.
    sample = sample_grid_slice(_ramp_grid(), _axis_plane(extent=0.5))

    np.testing.assert_allclose(sample.potential_total_ev, ((0.0, 4.0), (16.0, 20.0)))


def test_sample_grid_slice_rejects_thin_plane() -> None:
    with pytest.raises(UepError, match="at least two points"):
        sample_grid_slice(_ramp_grid(), _axis_plane(points_u=1, points_v=5))


def test_sample_grid_slice_rejects_thin_grid() -> None:
    with pytest.raises(UepError, match="at least two points per axis"):
        sample_grid_slice(_thin_grid(), _axis_plane())


def test_sample_grid_slice_rejects_bad_lattice() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=((2.0, 0.0, 0.0), (0.0, 2.0, 0.0)),  # type: ignore[arg-type]
        grid=(4, 4, 4),
        potential_total_ev=np.zeros((4, 4, 4), dtype=float),
    )

    with pytest.raises(UepError, match="three lattice vectors"):
        sample_grid_slice(grid, _axis_plane())


# Isosurfaces


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


def test_extract_isosurface_fallback_builds_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("mufinder.science.uep._pyvista_modules", lambda: None)
    grid = _linear_ramp_grid()
    level = 0.5 * (float(grid.potential_total_ev.min()) + float(grid.potential_total_ev.max()))
    mesh = extract_isosurface(grid, level_ev=level)
    assert isinstance(mesh, UepIsosurfaceMesh)
    assert len(mesh.vertices_ang) >= 3
    assert len(mesh.faces) >= 1


def test_extract_isosurface_can_use_pyvista_contour_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def test_extract_isosurface_fallback_wraps_a_single_peak_in_triangles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uep_module, "_pyvista_modules", lambda: None)
    values = np.zeros((4, 4, 4), dtype=float)
    values[1, 1, 1] = 1.0
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE, grid=(4, 4, 4), potential_total_ev=values
    )

    mesh = extract_isosurface(grid, level_ev=0.5)

    # Each tetrahedron touching the peak has one corner above the level, so it
    # is cut by a triangle through the midpoints of the edges leaving the peak.
    assert len(mesh.faces) == 24
    distances = np.linalg.norm(np.asarray(mesh.vertices_ang) - np.array([1.0, 1.0, 1.0]), axis=1)
    assert np.all(distances <= 0.5 * math.sqrt(3.0) + 1e-9)


def test_extract_isosurface_pyvista_returns_none_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


def _fake_pyvista(points: np.ndarray, faces: np.ndarray) -> object:
    """A PyVista stand-in whose contour returns the given points and face buffer."""

    class _FakePolyData:
        n_cells = 1

        def __init__(self) -> None:
            self.points = points
            self.faces = faces

        def triangulate(self):
            return self

    class _FakeStructuredGrid:
        def __init__(self, *_coords) -> None:
            pass

        def __setitem__(self, _name: str, _values: np.ndarray) -> None:
            pass

        def contour(self, *, isosurfaces, scalars: str):  # noqa: ARG002
            return _FakePolyData()

    class _FakePyVista:
        StructuredGrid = _FakeStructuredGrid

    return _FakePyVista()


@pytest.mark.parametrize(
    ("faces", "expected"),
    [
        ([4, 0, 1, 2, 3, 3, 4, 5, 6], ((0, 1, 2), (0, 2, 3), (4, 5, 6))),
        ([3, 0, 1, 2, 4, 5, 6], ((0, 1, 2),)),
    ],
    ids=["fans-polygons-into-triangles", "stops-at-truncated-buffer"],
)
def test_extract_isosurface_reads_pyvista_face_buffers(
    monkeypatch: pytest.MonkeyPatch, faces, expected
) -> None:
    points = np.arange(21, dtype=float).reshape((7, 3))
    monkeypatch.setattr(
        uep_module, "_pyvista_modules", lambda: _fake_pyvista(points, np.array(faces, dtype=int))
    )

    mesh = extract_isosurface(_linear_ramp_grid(), level_ev=0.5)

    assert mesh.faces == expected
    assert len(mesh.vertices_ang) == 7


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


def test_extract_isosurface_rejects_thin_grid() -> None:
    grid = UepPotentialGrid(
        lattice_vectors=_CUBE_LATTICE,
        grid=(1, 4, 4),
        potential_total_ev=np.zeros((1, 4, 4), dtype=float),
    )
    with pytest.raises(UepError, match="at least two points per axis"):
        extract_isosurface(grid, level_ev=0.0)


# Stationary points and muon optimisation


def test_classify_stationary_point_labels_all_categories() -> None:
    assert classify_stationary_point(np.diag([1.0, 1.0, 1.0])).label == "minimum"
    assert classify_stationary_point(np.diag([-1.0, -1.0, -1.0])).label == "maximum"
    assert classify_stationary_point(np.diag([1.0, -1.0, 1.0])).label == "saddle"
    flat = classify_stationary_point(np.diag([0.0, 1.0, 1.0]))
    assert flat.label == "flat"
    assert flat.near_zero_count == 1


def test_zero_point_energy_positive_for_minimum() -> None:
    value = zero_point_energy_ev(np.diag([1.0, 1.0, 1.0]))
    assert value is not None
    assert value > 0.0


def test_zero_point_energy_returns_none_for_unstable_extremum() -> None:
    assert zero_point_energy_ev(np.diag([-1.0, 1.0, 1.0])) is None


def test_zero_point_energy_rejects_non_positive_mass() -> None:
    with pytest.raises(UepError, match="particle_mass_kg must be positive"):
        zero_point_energy_ev(np.diag([1.0, 1.0, 1.0]), particle_mass_kg=0.0)


def test_particle_mass_kg_from_amu_scales_linearly() -> None:
    one = particle_mass_kg_from_amu(1.0)
    two = particle_mass_kg_from_amu(2.0)
    assert two == pytest.approx(2.0 * one)
    assert one > 0.0


def test_optimize_muon_position_moves_to_a_minimum() -> None:
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
    assert (
        np.linalg.norm(
            np.asarray(first.optimized_position_ang) - np.array((1.0, 1.0, 1.0), dtype=float)
        )
        > 0.1
    )
    assert (
        np.linalg.norm(
            np.asarray(second.optimized_position_ang) - np.array((1.05, 1.0, 1.0), dtype=float)
        )
        > 0.1
    )


def test_a_saved_optimisation_result_round_trips_and_a_malformed_one_is_refused() -> None:
    result = UepOptimizationResult(
        start_position_ang=(1.0, 1.0, 1.0),
        optimized_position_ang=(1.5, 1.5, 1.5),
        optimized_fractional_position=(0.25, 0.25, 0.25),
        classical_energy_ev=-1.5,
        zero_point_energy_ev=0.2,
        total_energy_ev=-1.3,
        success=True,
        message="converged",
        iterations=4,
        classification=StationaryPointClassification(
            label="minimum",
            eigenvalues=(1.0, 2.0, 3.0),
            positive_count=3,
            negative_count=0,
            near_zero_count=0,
        ),
    )

    assert optimization_result_from_dict(asdict(result)) == result
    assert optimization_result_from_dict({}) is None
    assert optimization_result_from_dict({**asdict(result), "classification": None}) is None
    assert optimization_result_from_dict({**asdict(result), "iterations": "many"}) is None
    assert optimization_result_from_dict("not a mapping") is None
    assert classification_from_dict("not a mapping") is None
    assert classification_from_dict({"eigenvalues": [1.0, 2.0, 3.0], "positive_count": "x"}) is None


class _QuadraticField:
    """A potential with the given curvatures about the origin, in eV/Å²."""

    def __init__(self, curvatures: tuple[float, float, float]) -> None:
        self._hessian = np.diag(curvatures)

    def fractional_position(self, position):
        return tuple(float(value) for value in position)

    def potential(self, position):
        point = np.asarray(position, dtype=float).reshape(3)
        value = np.array([0.5 * point @ self._hessian @ point])
        return value, value, np.zeros(1)

    def gradient(self, position):
        point = np.asarray(position, dtype=float).reshape(3)
        value = (self._hessian @ point).reshape(1, 3)
        return value, value, np.zeros((1, 3))

    def hessian(self, position):
        value = self._hessian.reshape(1, 3, 3)
        return value, value, np.zeros((1, 3, 3))


def test_optimize_muon_position_returns_the_zero_point_energy_of_a_minimum() -> None:
    field = _QuadraticField((2.0, 3.0, 4.0))

    result = optimize_muon_position(field, start_position_ang=(0.1, -0.2, 0.3))

    assert result.classification.label == "minimum"
    assert result.zero_point_energy_ev == pytest.approx(
        zero_point_energy_ev(np.diag((2.0, 3.0, 4.0)))
    )
    assert result.total_energy_ev == pytest.approx(
        result.classical_energy_ev + result.zero_point_energy_ev
    )


def test_optimize_muon_position_returns_no_zero_point_energy_for_a_saddle() -> None:
    field = _QuadraticField((2.0, 3.0, -4.0))

    result = optimize_muon_position(field, start_position_ang=(0.0, 0.0, 0.0))

    assert result.classification.label == "saddle"
    assert result.zero_point_energy_ev is None
    assert result.total_energy_ev is None


def test_optimize_muon_position_returns_no_zero_point_energy_when_it_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    field = _QuadraticField((2.0, 3.0, 4.0))
    monkeypatch.setattr(
        uep_module,
        "minimize",
        lambda *_a, **_k: SimpleNamespace(
            x=np.array([0.1, 0.1, 0.1]), success=False, message="limit", nit=1
        ),
    )

    result = optimize_muon_position(field, start_position_ang=(0.1, 0.1, 0.1))

    assert result.success is False
    assert result.classification.label == "minimum"
    assert result.zero_point_energy_ev is None
    assert result.total_energy_ev is None


def test_the_force_tolerance_is_a_gradient_norm_in_ev_per_angstrom() -> None:
    field = _QuadraticField((1.0, 1.0, 1.0))
    start = (0.1, 0.0, 0.0)

    stopped = optimize_muon_position(
        field, start_position_ang=start, force_tolerance_ev_per_ang=1.0
    )
    moved = optimize_muon_position(field, start_position_ang=start, force_tolerance_ev_per_ang=1e-5)

    # The gradient at the start is 0.1 eV/Å: under 1, over 1e-5.
    assert stopped.optimized_position_ang == pytest.approx(start)
    assert stopped.iterations == 0
    assert np.linalg.norm(moved.optimized_position_ang) < 1e-5


def test_the_iteration_limit_has_one_default() -> None:
    limit = inspect.signature(optimize_muon_position).parameters["max_steps"].default

    assert limit == DEFAULT_OPTIMIZER_MAX_STEPS == 300


def test_isosurface_polygon_ordering_leaves_collinear_points_alone() -> None:
    # Private helper: a plane cuts a tetrahedron in a triangle or a planar quadrilateral, never a line.
    points = [np.array([float(x), 0.0, 0.0]) for x in range(4)]
    assert uep_module._order_polygon_vertices(points) is points


def test_isosurface_vertex_on_a_flat_edge_is_the_midpoint() -> None:
    # Private helper: an edge only crosses the level when its end values differ.
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])
    assert np.allclose(
        uep_module._interpolate_isosurface_vertex(a, b, 1.0, 1.0, level_ev=1.0), [1.0, 0.0, 0.0]
    )
