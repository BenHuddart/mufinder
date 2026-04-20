"""Tests for the interactive UEP analysis dialog."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.app.uep_wizard import UepAnalysisDialog
from mufinder.domain.models import Structure
from mufinder.science.uep import extract_isosurface, potential_grid


def _fixture_path(name: str) -> Path:
    return Path("tests/fixtures/uep") / name


def _source_structure_with_spins() -> Structure:
    return Structure(
        id=uuid4(),
        label="Source host",
        lattice_vectors=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"atomic_spins": [1.5, -1.5]},
    )


def test_uep_wizard_probe_and_promote_emits_payload(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    promoted: list[dict] = []
    dialog.promote_requested.connect(lambda payload: promoted.append(payload))

    dialog._probe_point.set_values((0.25, 0.25, 0.25))
    dialog._run_probe()
    assert "Total potential" in dialog._probe_output.toPlainText()

    dialog._promote_current_probe()
    assert promoted
    assert promoted[0]["generation_method"] == "uep_probe"
    dialog.close()


def test_uep_wizard_theory_text_and_summary_include_reference_and_spin_note(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    theory_text = dialog._theory_text.toPlainText()
    summary_text = dialog._summary_text.toPlainText()

    assert "10.1063/5.0012381" in theory_text
    assert "Wyckoff" in theory_text
    assert "muonium" in theory_text
    assert "Mulliken spin moments recovered from CASTEP output" in summary_text
    dialog.close()


def test_uep_wizard_batch_search_populates_cluster_table_and_promotes(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    promoted: list[dict] = []
    dialog.promote_requested.connect(lambda payload: promoted.append(payload))

    dialog._batch_count.setValue(2)
    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()

    assert dialog._batch_table.rowCount() >= 1
    dialog._batch_table.selectRow(0)
    viewer_structure = dialog._batch_viewer._structure
    assert viewer_structure is not None
    assert viewer_structure.species.count("X") >= 1
    assert "Showing cluster" in dialog._batch_viewer_summary.text()
    dialog._promote_selected_cluster()

    assert promoted
    assert promoted[0]["generation_method"] == "uep_batch_cluster"
    dialog.close()


def test_uep_wizard_batch_search_spacing_mode_generates_spacing_summary(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._set_batch_mode("spacing")
    dialog._batch_min_muon_muon.setValue(1.2)
    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()

    assert dialog._batch_count.isEnabled() is False
    assert dialog._batch_min_muon_muon.isEnabled() is True
    summary = dialog._batch_summary.toPlainText()
    assert "Spacing-constrained starts generated" in summary
    assert "Min muon-muon distance (A): 1.200" in summary
    dialog.close()


def test_uep_wizard_state_snapshot_restores_generated_analysis(qapp) -> None:
    first = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    first._probe_point.set_values((0.25, 0.25, 0.25))
    first._run_probe()
    first._line_points.setValue(11)
    first._run_line_sample()
    first._set_batch_mode("spacing")
    first._batch_min_muon_muon.setValue(1.2)
    first._batch_max_steps.setValue(5)
    first._run_batch_search()
    first._batch_table.selectRow(0)
    snapshot = first.state_snapshot()
    first.close()

    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_state=snapshot,
    )

    assert "Total potential" in restored._probe_output.toPlainText()
    assert restored._line_data.toPlainText().startswith("distance_A,total_eV")
    assert restored._batch_table.rowCount() >= 1
    assert restored._batch_promote.isEnabled() is True
    assert restored._batch_mode() == "spacing"
    assert restored._batch_count.isEnabled() is False
    assert restored._batch_min_muon_muon.isEnabled() is True
    restored_viewer_structure = restored._batch_viewer._structure
    assert restored_viewer_structure is not None
    assert restored_viewer_structure.species.count("X") >= 1
    restored.close()


def test_uep_wizard_restores_isosurface_overlay(qapp) -> None:
    first = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(first._field)
    level = 0.5 * (float(grid.potential_total_ev.min()) + float(grid.potential_total_ev.max()))
    mesh = extract_isosurface(grid, level_ev=level)
    first._potential_grid_cache = grid
    first._render_isosurface_mesh(mesh)
    snapshot = first.state_snapshot()
    first.close()

    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_state=snapshot,
    )

    assert restored._latest_isosurface_mesh is not None
    assert restored._isosurface_viewer._isosurface_vertices is not None
    assert restored._isosurface_min_label.text().endswith("eV")
    assert restored._isosurface_max_label.text().endswith("eV")
    assert "Displaying" in restored._isosurface_status_label.text()
    restored.close()


def test_uep_wizard_isosurface_level_editor_tracks_full_range(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    dialog._apply_potential_grid_cache(grid)
    minimum = float(grid.potential_total_ev.min())
    maximum = float(grid.potential_total_ev.max())

    assert dialog._isosurface_level_input.isEnabled() is True
    assert dialog._isosurface_min_label.text() == f"{minimum:.2f} eV"
    assert dialog._isosurface_max_label.text() == f"{maximum:.2f} eV"

    target_level = minimum + 0.75 * (maximum - minimum)
    dialog._set_isosurface_level(target_level)
    level = dialog._current_isosurface_level()
    assert level == pytest.approx(target_level, abs=1e-3)

    mesh = extract_isosurface(grid, level_ev=level if level is not None else target_level)
    dialog._render_isosurface_mesh(mesh)

    assert dialog._isosurface_min_label.text() == f"{minimum:.2f} eV"
    assert dialog._isosurface_max_label.text() == f"{maximum:.2f} eV"
    assert "filled isosurface" in dialog._isosurface_status_label.text()
    assert dialog._isosurface_status_kind == "success"
    assert dialog._isosurface_viewer._isosurface_color == "#dc2626"
    dialog.close()


def test_uep_wizard_exports_cached_potential_grid_to_xsf(qapp, monkeypatch, tmp_path) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    assert dialog._isosurface_export_button.isEnabled() is False

    grid = potential_grid(dialog._field)
    dialog._apply_potential_grid_cache(grid)
    assert dialog._isosurface_export_button.isEnabled() is True

    output_path = tmp_path / "uep_export.xsf"
    monkeypatch.setattr(
        "mufinder.app.uep_wizard.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(output_path), "XSF Files (*.xsf)"),
    )

    dialog._export_potential_grid_xsf()

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "BEGIN_DATAGRID_3D_UEP_POTENTIAL" in content
    assert "exported to" in dialog._isosurface_status_label.text()
    assert dialog._isosurface_status_kind == "exported"
    dialog.close()


def test_uep_wizard_viewer_uses_source_structure_spins(qapp) -> None:
    source_structure = _source_structure_with_spins()
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id=source_structure.id,
        source_structure=source_structure,
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    assert dialog._viewer_host_structure.properties["atomic_spins"] == [1.5, -1.5]
    assert dialog._isosurface_viewer._spins is not None
    assert dialog._isosurface_viewer._spins.tolist() == pytest.approx([1.5, -1.5])
    dialog.close()
