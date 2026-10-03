"""Tests for the interactive UEP analysis dialog."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QPushButton, QSizePolicy

from mufinder.app.theme import current_palette
from mufinder.app.uep_clustering_dialog import cluster_color_map
from mufinder.app.uep_wizard import UepAnalysisDialog, _AddSlicePlaneDialog, _EditSlicePlaneDialog, _SlicePlaneState
from mufinder.app.atomic_structure_viewer import MUON_VIEWER_RADIUS_ANG, _CameraState
from mufinder.domain.models import Structure
from mufinder.science.uep import (
    cartesian_normal_to_lattice_plane_components,
    extract_isosurface,
    fractional_point_to_cartesian,
    miller_indices_to_cartesian_normal,
    plane_from_fractional_anchor_and_miller,
    potential_grid,
    serialize_potential_grid,
)


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


def _slice_plane_for_dialog(dialog: UepAnalysisDialog):
    anchor = dialog._source.structure.fractional_positions[0]
    return plane_from_fractional_anchor_and_miller(
        anchor_fractional=anchor,
        miller_indices=(0.0, 0.0, 1.0),
        lattice_vectors=dialog._source.structure.lattice_vectors,
        points_u=21,
        points_v=21,
    )


def _run_batch_search_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _start_inline(thread) -> None:
        thread.run()
        thread.finished.emit()

    monkeypatch.setattr("mufinder.app.uep_wizard._BatchSearchThread.start", _start_inline)


def _run_batch_zpe_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _start_inline(thread) -> None:
        thread.run()
        thread.finished.emit()

    monkeypatch.setattr("mufinder.app.uep_wizard._BatchZpeThread.start", _start_inline)


def _run_isosurface_threads_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _start_inline(thread) -> None:
        thread.run()
        thread.finished.emit()

    monkeypatch.setattr("mufinder.app.uep_wizard._IsosurfaceGridThread.start", _start_inline)
    monkeypatch.setattr("mufinder.app.uep_wizard._IsosurfaceMeshThread.start", _start_inline)


def test_uep_analysis_dialog_shrinks_for_small_screens(qapp) -> None:
    """Isosurface/slices controls scroll instead of forcing a tall minimum.

    Guards the SC-1 small-screen fix: the dialog's minimum height must fit a
    13" MacBook's ~850px usable height, the controls column above each viewer
    must live in a scrollable area (not push the viewer off-screen), and the
    slider/viewer minimums must be small enough to avoid re-inflating the
    dialog's minimum size.
    """
    from PySide6.QtWidgets import QScrollArea

    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    assert dialog.minimumWidth() == 980
    assert dialog.minimumHeight() == 640

    assert isinstance(dialog._isosurface_controls_scroll, QScrollArea)
    assert dialog._isosurface_controls_scroll.widgetResizable()
    assert isinstance(dialog._slice_controls_scroll, QScrollArea)
    assert dialog._slice_controls_scroll.widgetResizable()

    assert dialog._isosurface_slider.minimumWidth() == 400
    assert dialog._isosurface_viewer.minimumHeight() == 320
    assert dialog._slice_viewer.minimumHeight() == 320

    assert dialog._batch_sampling_card.minimumHeight() == 240
    optimizer_card = dialog._batch_run.parentWidget()
    log_card = dialog._batch_summary.parentWidget()
    assert optimizer_card.minimumHeight() == 240
    assert log_card.minimumHeight() == 240

    dialog.close()


def test_add_slice_plane_dialog_allows_point_normal_without_selection(qapp) -> None:
    dialog = _AddSlicePlaneDialog(
        selected_positions_ang=[],
        selected_positions_fractional=[],
        lattice_vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
    )

    assert dialog._point_normal.isEnabled() is True
    assert dialog._point_normal.isChecked() is True
    assert dialog._anchor_input.values() == pytest.approx((0.0, 0.0, 0.0))
    assert "manually specifying the anchor" in dialog._summary_label.text()

    dialog._accept_plane()

    plane = dialog.plane()
    assert plane is not None
    assert plane.anchor_ang == pytest.approx((0.0, 0.0, 0.0))
    assert plane.normal_ang == pytest.approx((0.0, 0.0, 1.0))
    dialog.close()


def test_add_slice_plane_dialog_defaults_point_to_selected_atom(qapp) -> None:
    lattice_vectors = (
        (4.2, 0.0, 0.0),
        (-2.1, 0.5 * np.sqrt(3.0) * 4.2, 0.0),
        (0.0, 0.0, 6.8),
    )
    anchor_fractional = (1.0 / 3.0, 2.0 / 3.0, 0.25)
    dialog = _AddSlicePlaneDialog(
        selected_positions_ang=[fractional_point_to_cartesian(lattice_vectors, anchor_fractional)],
        selected_positions_fractional=[anchor_fractional],
        lattice_vectors=lattice_vectors,
    )

    assert dialog._point_normal.isEnabled() is True
    assert dialog._anchor_input.values() == pytest.approx(anchor_fractional)
    dialog.close()


def test_add_slice_plane_dialog_uses_fractional_anchor_and_miller_indices_for_hexagonal_cell(qapp) -> None:
    lattice_vectors = (
        (4.2, 0.0, 0.0),
        (-2.1, 0.5 * np.sqrt(3.0) * 4.2, 0.0),
        (0.0, 0.0, 6.8),
    )
    dialog = _AddSlicePlaneDialog(
        selected_positions_ang=[],
        selected_positions_fractional=[],
        lattice_vectors=lattice_vectors,
    )
    dialog._anchor_input.set_values((0.25, 0.5, 0.75))
    dialog._normal_input.set_values((1.0, 0.0, 0.0))

    dialog._accept_plane()

    plane = dialog.plane()
    assert plane is not None
    assert plane.anchor_ang == pytest.approx(fractional_point_to_cartesian(lattice_vectors, (0.25, 0.5, 0.75)))
    assert plane.normal_ang == pytest.approx(miller_indices_to_cartesian_normal(lattice_vectors, (1.0, 0.0, 0.0)))
    dialog.close()


def test_add_slice_plane_dialog_defaults_normal_from_current_view_direction(qapp) -> None:
    lattice_vectors = (
        (4.2, 0.0, 0.0),
        (-2.1, 0.5 * np.sqrt(3.0) * 4.2, 0.0),
        (0.0, 0.0, 6.8),
    )
    default_components = cartesian_normal_to_lattice_plane_components(lattice_vectors, (0.0, 0.0, -1.0))
    dialog = _AddSlicePlaneDialog(
        selected_positions_ang=[],
        selected_positions_fractional=[],
        default_normal_components=default_components,
        lattice_vectors=lattice_vectors,
    )

    assert dialog._normal_input.values() == pytest.approx(default_components)
    assert "current viewer direction" in dialog._point_normal_hint.text()
    dialog.close()


def test_edit_slice_plane_dialog_uses_fractional_anchor_and_lattice_plane_components(qapp) -> None:
    lattice_vectors = (
        (4.2, 0.0, 0.0),
        (-2.1, 0.5 * np.sqrt(3.0) * 4.2, 0.0),
        (0.0, 0.0, 6.8),
    )
    plane = plane_from_fractional_anchor_and_miller(
        anchor_fractional=(0.25, 0.5, 0.75),
        miller_indices=(1.0, 0.0, 1.0),
        lattice_vectors=lattice_vectors,
        points_u=21,
        points_v=21,
    )
    dialog = _EditSlicePlaneDialog(
        plane_number=1,
        plane=plane,
        lattice_vectors=lattice_vectors,
    )

    assert dialog._anchor.values() == pytest.approx((0.25, 0.5, 0.75))
    assert dialog._normal.values() == pytest.approx(
        cartesian_normal_to_lattice_plane_components(lattice_vectors, plane.normal_ang)
    )

    edited = dialog.edited_plane()
    assert edited.anchor_ang == pytest.approx(plane.anchor_ang)
    assert edited.normal_ang == pytest.approx(plane.normal_ang)
    dialog.close()


def test_uep_wizard_probe_has_no_direct_promotion_workflow(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._probe_point.set_values((0.25, 0.25, 0.25))
    dialog._run_probe()
    assert "Total potential" in dialog._probe_output.toPlainText()
    assert not hasattr(dialog, "promote_requested")
    assert not hasattr(dialog, "_probe_promote")
    assert not any(button.text().startswith("Promote") for button in dialog.findChildren(QPushButton))
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
    assert "E_zp" in theory_text
    assert "Mulliken spin moments recovered from CASTEP output" in summary_text
    dialog.close()


def test_uep_wizard_exposes_only_branded_batch_isosurface_and_slice_tabs(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    labels = [dialog._tabs.tabText(index) for index in range(dialog._tabs.count())]
    palette = current_palette()
    dialog.show()
    dialog._tabs.setCurrentIndex(2)
    qapp.processEvents()

    assert labels == ["Batch Search", "Isosurfaces", "Slices"]
    assert dialog._tabs.objectName() == "uepAnalysisTabs"
    assert palette.accent in dialog._tabs.styleSheet()
    assert palette.bg_surface in dialog._tabs.styleSheet()
    assert dialog._slice_colorbar_canvas.minimumHeight() == 180
    assert dialog._slice_colorbar_canvas.maximumHeight() == 320
    assert dialog._slice_colorbar_canvas.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    assert dialog._slice_manage_muons_button.isVisibleTo(dialog) is True
    dialog.close()


def test_uep_wizard_initial_state_gw_factor_overrides_default_and_rejects_mismatched_grid(qapp) -> None:
    baseline = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    cached_grid = potential_grid(baseline._field)
    baseline.close()

    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_state={
            "gw_factor": 7.0,
            "potential_grid": serialize_potential_grid(cached_grid),
        },
    )

    assert restored._source.gw_factor == pytest.approx(7.0)
    assert restored._gw_factor_input.value() == pytest.approx(7.0)
    assert restored._slice_gw_factor_input.value() == pytest.approx(7.0)
    assert restored._batch_gw_factor_input.value() == pytest.approx(7.0)
    assert restored._potential_grid_cache is None
    restored.close()


def test_uep_wizard_restores_legacy_visible_tab_indexes_into_new_tab_order(qapp) -> None:
    for saved_index, expected_index in ((4, 1), (5, 2), (6, 0)):
        dialog = UepAnalysisDialog(
            source_job_label="uep_source_row1",
            source_structure_id="11111111-1111-1111-1111-111111111111",
            source_structure=_source_structure_with_spins(),
            den_fmt_path=_fixture_path("Si2.den_fmt"),
            castep_path=_fixture_path("Si2.castep"),
            cell_path=_fixture_path("Si2.cell"),
            initial_state={"current_tab": saved_index},
        )

        assert dialog._tabs.currentIndex() == expected_index
        dialog.close()


def test_uep_wizard_pressing_enter_on_gw_factor_rebuilds_field_and_recalculates_grid(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._probe_point.set_values((0.25, 0.25, 0.25))
    dialog._run_probe()

    started: list[bool] = []
    monkeypatch.setattr(dialog, "_start_isosurface_grid_calculation", lambda: started.append(True))

    dialog._gw_factor_input.setValue(7.0)
    dialog._gw_factor_input.returnPressed.emit()

    assert dialog._source.gw_factor == pytest.approx(7.0)
    assert dialog.state_snapshot()["gw_factor"] == pytest.approx(7.0)
    assert dialog._gw_factor_input.value() == pytest.approx(7.0)
    assert dialog._slice_gw_factor_input.value() == pytest.approx(7.0)
    assert dialog._batch_gw_factor_input.value() == pytest.approx(7.0)
    assert dialog._probe_output.toPlainText() == ""
    assert not hasattr(dialog, "_probe_promote")
    assert dialog._potential_grid_raw_cache is None
    assert dialog._potential_grid_cache is None
    assert started == [True]
    assert "gw_factor): 7.000" in dialog._summary_text.toPlainText()
    dialog.close()


def test_uep_wizard_leaving_gw_factor_field_without_enter_reverts_to_active_value(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._gw_factor_input.setValue(7.0)
    dialog._on_gw_factor_editing_finished()

    assert dialog._source.gw_factor == pytest.approx(5.0)
    assert dialog._gw_factor_input.value() == pytest.approx(5.0)
    assert dialog._slice_gw_factor_input.value() == pytest.approx(5.0)
    assert dialog._batch_gw_factor_input.value() == pytest.approx(5.0)
    dialog.close()


def test_uep_wizard_gw_factor_tooltip_and_controls_are_present(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    assert "Gaussian width factor" in dialog._gw_factor_input.toolTip()
    assert "Press Enter" in dialog._gw_factor_input.toolTip()
    assert dialog._slice_gw_factor_input.toolTip() == dialog._gw_factor_input.toolTip()
    assert dialog._batch_gw_factor_input.toolTip() == dialog._gw_factor_input.toolTip()
    assert not any(button.text() == "Apply" for button in dialog.findChildren(QPushButton))
    dialog.close()


def test_uep_wizard_pressing_enter_on_batch_gw_factor_rebuilds_and_syncs_all_tabs(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    started: list[bool] = []
    monkeypatch.setattr(dialog, "_start_isosurface_grid_calculation", lambda: started.append(True))

    dialog._batch_gw_factor_input.setValue(8.0)
    dialog._batch_gw_factor_input.returnPressed.emit()

    assert dialog._source.gw_factor == pytest.approx(8.0)
    assert dialog._gw_factor_input.value() == pytest.approx(8.0)
    assert dialog._slice_gw_factor_input.value() == pytest.approx(8.0)
    assert dialog._batch_gw_factor_input.value() == pytest.approx(8.0)
    assert started == [True]
    dialog.close()


def test_uep_wizard_clustering_window_populates_cluster_table_without_direct_promotion(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._batch_count.setValue(12)
    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()

    assert dialog._batch_table.topLevelItemCount() == 0
    assert dialog._batch_clustering_button.isEnabled() is True
    dialog._open_clustering_dialog()

    assert dialog._batch_table.topLevelItemCount() >= 1
    cluster_item = dialog._batch_table.topLevelItem(0)
    assert cluster_item is not None
    assert cluster_item.childCount() >= 1
    cluster_item.setSelected(True)
    dialog._batch_table.setCurrentItem(cluster_item)
    qapp.processEvents()
    viewer_structure = dialog._batch_viewer._structure
    assert viewer_structure is not None
    assert viewer_structure.species.count("X") == 0
    assert dialog._batch_viewer._point_overlay_specs
    assert "Showing all" in dialog._batch_viewer_summary.text()
    child_energies = [
        float(cluster_item.child(index).text(5))
        for index in range(cluster_item.childCount())
        if cluster_item.child(index).text(5)
    ]
    assert child_energies == sorted(child_energies)
    all_site_labels = [
        item.child(index).text(0)
        for item_index in range(dialog._batch_table.topLevelItemCount())
        if (item := dialog._batch_table.topLevelItem(item_index)) is not None
        for index in range(item.childCount())
    ]
    assert sorted(int(label.removeprefix("Site ")) for label in all_site_labels) == list(range(1, len(all_site_labels) + 1))
    assert not hasattr(dialog, "promote_requested")
    assert not hasattr(dialog, "_promote_selected_cluster")
    dialog.close()


def test_uep_wizard_clustering_window_close_clears_reference_and_allows_reopen(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._run_batch_search()
    dialog._open_clustering_dialog()
    first_dialog = dialog._clustering_dialog

    assert first_dialog is not None

    first_dialog.close()
    qapp.processEvents()

    assert dialog._clustering_dialog is None

    dialog._open_clustering_dialog()

    assert dialog._clustering_dialog is not None
    assert dialog._clustering_dialog is not first_dialog
    dialog.close()


def test_uep_wizard_state_change_keeps_child_dialogs_after_parent_handlers(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    events: list[str] = []

    class _FakeClusteringDialog:
        def state_snapshot(self) -> dict[str, object]:
            return {"method": "hier"}

        def close(self) -> bool:
            return True

    class _FakeChildDialog:
        def close(self) -> bool:
            return True

    clustering_dialog = _FakeClusteringDialog()
    isosurface_dialog = _FakeChildDialog()
    slice_dialog = _FakeChildDialog()
    dialog._clustering_dialog = clustering_dialog
    dialog._isosurface_muon_dialog = isosurface_dialog
    dialog._slice_muon_dialog = slice_dialog

    def _keep_child(child) -> None:
        if child is clustering_dialog:
            events.append("keep-clustering")
        elif child is isosurface_dialog:
            events.append("keep-isosurface")
        elif child is slice_dialog:
            events.append("keep-slice")

    monkeypatch.setattr(UepAnalysisDialog, "_keep_dialog_in_front", staticmethod(_keep_child))
    dialog.state_changed.connect(lambda _state: events.append("parent-handler"))

    dialog._emit_state_changed()

    assert events == ["parent-handler", "keep-clustering", "keep-isosurface", "keep-slice"]
    dialog.close()


def test_uep_wizard_site_selection_updates_viewer_without_direct_promotion(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._batch_count.setValue(2)
    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()
    dialog._open_clustering_dialog()

    cluster_item = dialog._batch_table.topLevelItem(0)
    assert cluster_item is not None
    site_item = cluster_item.child(0)
    assert site_item is not None
    site_item.setSelected(True)
    dialog._batch_table.setCurrentItem(site_item)
    qapp.processEvents()

    assert "selected optimized muon positions" in dialog._batch_viewer_summary.text()
    assert dialog._batch_viewer._point_overlay_specs
    assert not hasattr(dialog, "_promote_selected_cluster")
    dialog.close()


def test_uep_wizard_isosurface_muon_management_adds_and_removes_clustered_muons(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)

    def _mock_zpe(_field, optimization):
        return optimization.__class__(
            start_position_ang=optimization.start_position_ang,
            optimized_position_ang=optimization.optimized_position_ang,
            optimized_fractional_position=optimization.optimized_fractional_position,
            classical_energy_ev=optimization.classical_energy_ev,
            zero_point_energy_ev=0.125,
            total_energy_ev=optimization.classical_energy_ev + 0.125,
            success=optimization.success,
            message=optimization.message,
            iterations=optimization.iterations,
            classification=optimization.classification,
        )

    monkeypatch.setattr("mufinder.app.uep_wizard.calculate_optimization_zero_point_energy", _mock_zpe)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._batch_count.setValue(12)
    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()
    dialog._latest_batch_results = [_mock_zpe(dialog._field, result) for result in dialog._latest_batch_results]
    dialog._open_clustering_dialog()

    candidate = dialog._clustered_muon_candidates()[0]
    dialog._add_candidate_muons_to_viewer("isosurface", [candidate.key], True)

    assert len(dialog._isosurface_viewer_muons) == 1
    assert dialog._isosurface_viewer._structure is not None
    assert dialog._isosurface_viewer._structure.species.count("X") == 1
    expected_color = cluster_color_map([candidate.cluster_index])[candidate.cluster_index]
    assert dialog._isosurface_viewer._structure.properties["atom_colors"][-1] == expected_color
    viewer_entry = dialog._isosurface_viewer_muons[0]
    assert viewer_entry.linked_isosurface_layer_key is not None
    layer_index = dialog._find_isosurface_layer_index_by_key(viewer_entry.linked_isosurface_layer_key)
    assert layer_index is not None
    linked_layer = dialog._isosurface_layers[layer_index]
    assert linked_layer.color == expected_color
    assert linked_layer.managed_muon_key == viewer_entry.key
    assert linked_layer.level_ev == pytest.approx(float(candidate.total_energy_ev) + dialog._potential_grid_zero_shift_ev)
    assert linked_layer.status_text == (
        "Isosurface at total (classical + zero-point) energy "
        f"{float(candidate.total_energy_ev):.6f} eV for muon {candidate.label}."
    )

    dialog._set_current_viewer_muon_selection("isosurface", [viewer_entry.key])

    host_count = len(dialog._viewer_host_structure.species)
    assert dialog._isosurface_viewer._selected_indexes == [host_count]
    legend_texts = [
        label.text()
        for label in dialog._isosurface_legend_entries.findChildren(type(dialog._isosurface_min_label))
    ]
    assert any(candidate.label in text for text in legend_texts)
    assert any(f"Cluster {candidate.cluster_index}" in text for text in legend_texts)

    dialog._remove_viewer_muons("isosurface", [viewer_entry.key])

    assert dialog._isosurface_viewer_muons == []
    assert dialog._isosurface_viewer._structure is dialog._viewer_host_structure
    assert dialog._find_isosurface_layer_index_by_key(viewer_entry.linked_isosurface_layer_key) is None
    dialog.close()


def test_uep_wizard_editing_isosurface_muon_label_updates_linked_status_and_restores(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)

    def _mock_zpe(_field, optimization):
        return optimization.__class__(
            start_position_ang=optimization.start_position_ang,
            optimized_position_ang=optimization.optimized_position_ang,
            optimized_fractional_position=optimization.optimized_fractional_position,
            classical_energy_ev=optimization.classical_energy_ev,
            zero_point_energy_ev=0.125,
            total_energy_ev=optimization.classical_energy_ev + 0.125,
            success=optimization.success,
            message=optimization.message,
            iterations=optimization.iterations,
            classification=optimization.classification,
        )

    monkeypatch.setattr("mufinder.app.uep_wizard.calculate_optimization_zero_point_energy", _mock_zpe)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._batch_count.setValue(12)
    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()
    dialog._latest_batch_results = [_mock_zpe(dialog._field, result) for result in dialog._latest_batch_results]
    dialog._open_clustering_dialog()

    candidate = dialog._clustered_muon_candidates()[0]
    dialog._add_candidate_muons_to_viewer("isosurface", [candidate.key], True)
    viewer_entry = dialog._isosurface_viewer_muons[0]
    assert viewer_entry.linked_isosurface_layer_key is not None
    layer_index = dialog._find_isosurface_layer_index_by_key(viewer_entry.linked_isosurface_layer_key)
    assert layer_index is not None

    dialog._open_isosurface_muon_dialog()
    item = dialog._isosurface_muon_dialog._current_table.topLevelItem(0)
    assert item is not None
    assert bool(item.flags() & Qt.ItemFlag.ItemIsEditable)

    item.setText(0, "Renamed Site")
    qapp.processEvents()

    assert dialog._isosurface_viewer_muons[0].label == "Renamed Site"
    linked_layer = dialog._isosurface_layers[layer_index]
    assert linked_layer.status_text == (
        "Isosurface at total (classical + zero-point) energy "
        f"{float(candidate.total_energy_ev):.6f} eV for muon Renamed Site."
    )

    dialog._on_isosurface_tab_changed(layer_index)
    assert dialog._isosurface_status_label.text() == linked_layer.status_text

    snapshot = dialog.state_snapshot()
    dialog.close()

    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_state=snapshot,
    )

    assert restored._isosurface_viewer_muons[0].label == "Renamed Site"
    restored_layer_index = restored._find_isosurface_layer_index_by_key(viewer_entry.linked_isosurface_layer_key)
    assert restored_layer_index is not None
    assert restored._isosurface_layers[restored_layer_index].status_text == (
        "Isosurface at total (classical + zero-point) energy "
        f"{float(candidate.total_energy_ev):.6f} eV for muon Renamed Site."
    )
    restored.close()


def test_uep_wizard_isosurface_muon_changes_preserve_spin_toggle(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._isosurface_viewer._show_spins.setChecked(False)

    dialog._add_custom_muon_to_viewer("isosurface", (0.125, 0.25, 0.375))
    assert dialog._isosurface_viewer._show_spins.isChecked() is False

    muon_key = dialog._isosurface_viewer_muons[0].key
    dialog._remove_viewer_muons("isosurface", [muon_key])
    assert dialog._isosurface_viewer._show_spins.isChecked() is False
    dialog.close()


def test_uep_wizard_slice_muon_management_restores_custom_muons(qapp) -> None:
    first = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    first._add_custom_muon_to_viewer("slice", (0.125, 0.25, 0.375))
    assert len(first._slice_viewer_muons) == 1
    entry = first._slice_viewer_muons[0]
    first._set_current_viewer_muon_selection("slice", [entry.key])

    host_count = len(first._viewer_host_structure.species)
    assert first._slice_viewer._structure is not None
    assert first._slice_viewer._structure.species.count("X") == 1
    assert first._slice_viewer._selected_indexes == [host_count]

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

    assert len(restored._slice_viewer_muons) == 1
    assert restored._slice_viewer._structure is not None
    assert restored._slice_viewer._structure.species.count("X") == 1
    assert restored._slice_viewer_muons[0].fractional_position == pytest.approx((0.125, 0.25, 0.375))
    assert restored._slice_viewer._selected_indexes == [host_count]
    restored.close()


def test_uep_wizard_slice_muon_overlays_use_distinct_colors_and_legend(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._add_custom_muon_to_viewer("slice", (0.125, 0.25, 0.375))
    dialog._add_custom_muon_to_viewer("slice", (0.375, 0.25, 0.125))

    assert dialog._slice_viewer._structure is not None
    atom_colors = dialog._slice_viewer._structure.properties["atom_colors"]
    assert atom_colors[-2] != atom_colors[-1]
    legend_texts = [
        label.text()
        for label in dialog._slice_muon_legend_entries.findChildren(type(dialog._slice_status_label))
    ]
    assert any("Custom 1" in text for text in legend_texts)
    assert any("Custom 2" in text for text in legend_texts)
    dialog.close()


def test_uep_wizard_removing_last_slice_muon_resets_dialog_horizontal_scroll(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._add_custom_muon_to_viewer("slice", (0.125, 0.25, 0.375))
    dialog._open_slice_muon_dialog()
    qapp.processEvents()

    scroll = dialog._slice_muon_dialog._current_table.horizontalScrollBar()
    header = dialog._slice_muon_dialog._current_table.header()
    assert scroll is not None
    assert header is not None
    scroll.setValue(scroll.maximum())
    header.setOffset(max(header.offset(), 24))
    qapp.processEvents()

    muon_key = dialog._slice_viewer_muons[0].key
    dialog._remove_viewer_muons("slice", [muon_key])
    qapp.processEvents()

    assert scroll.value() == scroll.minimum()
    assert header.offset() == 0
    dialog.close()


def test_uep_wizard_slice_muon_changes_preserve_spin_toggle(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._slice_viewer._show_spins.setChecked(False)

    dialog._add_custom_muon_to_viewer("slice", (0.125, 0.25, 0.375))
    assert dialog._slice_viewer._show_spins.isChecked() is False

    muon_key = dialog._slice_viewer_muons[0].key
    dialog._remove_viewer_muons("slice", [muon_key])
    assert dialog._slice_viewer._show_spins.isChecked() is False
    dialog.close()


def test_uep_wizard_batch_search_uses_compact_control_card_layout(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    summary_parent = dialog._batch_summary.parentWidget()
    assert summary_parent is not None
    assert summary_parent.objectName() == "uepIsosurfaceInfoCard"
    assert dialog._batch_summary.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert dialog._batch_summary.minimumHeight() >= 148

    control_cards = dialog._batch_tab.findChildren(type(summary_parent), options=Qt.FindChildOption.FindChildrenRecursively)
    object_names = [widget.objectName() for widget in control_cards]
    assert object_names.count("analysisSliderPanel") >= 2
    assert object_names.count("uepIsosurfaceInfoCard") >= 1
    assert dialog._batch_progress.parentWidget() is not None
    assert dialog._batch_progress.parentWidget().objectName() == "analysisSliderPanel"
    assert dialog._batch_zpe_progress.parentWidget() is not None
    assert dialog._batch_zpe_progress.parentWidget().objectName() == "uepIsosurfaceInfoCard"
    assert dialog._batch_progress.format() == "Ready"
    assert dialog._batch_zpe_progress.format() == "Ready"
    assert dialog._batch_max_attempts.parentWidget() is not None
    assert dialog._batch_max_attempts.parentWidget().objectName() == "analysisSliderPanel"
    assert dialog._batch_optimization_tolerance.value() == pytest.approx(1e-5)
    assert dialog._batch_max_steps.value() == 100
    assert "pymuon-suite UEP workflow" in dialog._batch_optimization_tolerance.toolTip()
    assert "E_zp" in dialog._batch_calculate_zpe.toolTip()
    assert current_palette().accent in dialog._batch_progress.styleSheet()
    assert current_palette().success in dialog._batch_zpe_progress.styleSheet()
    assert dialog._batch_gw_factor_input.parentWidget() is not None
    assert dialog._batch_gw_factor_input.parentWidget().objectName() == "analysisSliderPanel"
    assert dialog._batch_calculate_zpe.parentWidget() is summary_parent
    assert not hasattr(dialog, "_batch_promote")
    assert not hasattr(dialog, "_batch_cluster_threshold")
    assert dialog._batch_clustering_button.parentWidget() is dialog._batch_sampling_card
    dialog.close()


def test_uep_wizard_isosurface_muon_dialog_places_include_checkbox_in_clustered_card(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    muon_dialog = dialog._create_viewer_muon_dialog(viewer_kind="isosurface")

    assert muon_dialog._include_isosurfaces is not None
    assert muon_dialog._include_isosurfaces.parentWidget() is not None
    assert muon_dialog._include_isosurfaces.parentWidget().title() == "Clustered Muons"
    assert [muon_dialog._candidate_table.headerItem().text(index) for index in range(muon_dialog._candidate_table.columnCount())] == [
        "Cluster / Site",
        "fX",
        "fY",
        "fZ",
        "Classical energy\n(eV)",
        "ZPE\n(eV)",
        "Total energy\n(eV)",
    ]
    assert [muon_dialog._current_table.headerItem().text(index) for index in range(muon_dialog._current_table.columnCount())] == [
        "Muon",
        "Source",
        "fX",
        "fY",
        "fZ",
        "Classical energy\n(eV)",
        "ZPE\n(eV)",
        "Total energy\n(eV)",
        "Isosurface",
    ]
    muon_dialog.close()
    dialog.close()


def test_uep_wizard_batch_search_default_split_and_headers_are_compact(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog.show()
    qapp.processEvents()

    sizes = dialog._batch_splitter.sizes()
    total = sum(sizes)
    ratio = 0.0 if total <= 0 else float(sizes[0]) / float(total)

    assert 0.45 <= ratio <= 0.55
    assert [dialog._batch_table.headerItem().text(index) for index in range(dialog._batch_table.columnCount())] == [
        "Cluster / Site",
        "Size",
        "fX",
        "fY",
        "fZ",
        "Classical energy\n(eV)",
        "ZPE\n(eV)",
        "Total energy\n(eV)",
    ]
    assert dialog._batch_table.indentation() == 10
    assert dialog._batch_table.property("squareTopCorners") == "true"
    assert dialog._batch_table.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOn
    assert dialog._tabs.styleSheet().count("margin-top: 0px;") == 1
    assert dialog._batch_tab.layout().contentsMargins().top() == 2
    assert dialog._isosurface_tab.layout().contentsMargins().top() == 2
    assert dialog._slices_tab.layout().contentsMargins().top() == 2
    assert dialog._batch_sampling_card.minimumHeight() == 240
    optimizer_card = dialog._batch_run.parentWidget()
    log_card = dialog._batch_summary.parentWidget()
    assert optimizer_card is not None
    assert log_card is not None
    assert optimizer_card.minimumHeight() == 240
    assert log_card.minimumHeight() == 240
    dialog.close()


def test_uep_wizard_batch_search_starts_worker_thread_and_shows_busy_progress(qapp, monkeypatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    started: list[object] = []
    monkeypatch.setattr("mufinder.app.uep_wizard._BatchSearchThread.start", lambda self: started.append(self))

    dialog._run_batch_search()

    assert len(started) == 1
    assert dialog._batch_search_thread is started[0]
    assert dialog._batch_run.isEnabled() is False
    assert dialog._batch_progress.minimum() == 0
    assert dialog._batch_progress.maximum() == 0
    assert dialog._batch_progress.format() == "Packing starts..."
    assert dialog._batch_calculate_zpe.isEnabled() is False
    assert dialog._batch_zpe_progress.format() == "Ready"
    assert "color: #FFFFFF;" not in dialog._batch_progress.styleSheet()
    dialog.close()


def test_uep_wizard_batch_zpe_before_search_logs_warning(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._run_batch_zpe()

    assert "no successful optimized sites are available yet" in dialog._batch_summary.toPlainText()
    assert dialog._batch_zpe_progress.format() == "Ready"
    dialog.close()


def test_uep_wizard_batch_search_logs_reduced_initial_starts(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    def _fake_generate_random_sites(**kwargs):
        assert kwargs["count"] == 6
        assert kwargs["min_muon_muon"] == pytest.approx(1.2)
        assert kwargs["allow_partial"] is True
        return ([SimpleNamespace(cartesian=[1.0, 1.0, 1.0], fractional=[0.1, 0.1, 0.1])], 25)

    monkeypatch.setattr("mufinder.app.uep_wizard.generate_random_sites", _fake_generate_random_sites)

    dialog._batch_count.setValue(6)
    dialog._batch_min_muon_muon.setValue(1.2)
    dialog._batch_optimization_tolerance.setValue(2e-5)
    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()

    summary = dialog._batch_summary.toPlainText()
    assert "Requested random starts: 6" in summary
    assert "Initial starts generated: 1" in summary
    assert "Min muon-muon distance (A): 1.200" in summary
    assert "Optimization tolerance (eV): 2e-05" in summary
    assert "muon-muon distance constraint limited packing" in summary
    assert "Relaxed sites available for clustering" in summary
    assert dialog._batch_progress.format() == "Batch search complete"
    assert "color: #FFFFFF;" in dialog._batch_progress.styleSheet()
    assert dialog._batch_table.topLevelItemCount() == 0
    assert dialog._batch_clustering_button.isEnabled() is True
    dialog.close()


def test_uep_wizard_state_snapshot_restores_generated_analysis(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    _run_batch_zpe_inline(monkeypatch)

    def _mock_zpe(_field, optimization):
        return replace(
            optimization,
            zero_point_energy_ev=0.125,
            total_energy_ev=optimization.classical_energy_ev + 0.125,
        )

    monkeypatch.setattr("mufinder.app.uep_wizard.calculate_optimization_zero_point_energy", _mock_zpe)
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
    first._batch_min_muon_muon.setValue(1.2)
    first._batch_optimization_tolerance.setValue(3e-5)
    first._batch_max_steps.setValue(5)
    first._run_batch_search()
    assert first._latest_batch_results
    first._latest_batch_results[0] = replace(first._latest_batch_results[0], success=True)
    first._run_batch_zpe()
    first._open_clustering_dialog()
    assert first._clustering_dialog is not None
    first._clustering_dialog._method_combo.setCurrentIndex(1)
    first._clustering_dialog._kmeans_k_spin.setValue(2)
    first._clustering_dialog._structure_viewer._show_bonds.setChecked(True)
    first._clustering_dialog._structure_viewer._camera_state = _CameraState(29.0, 61.0, -8.0)
    first._clustering_dialog._structure_viewer._axis.set_from_state(first._clustering_dialog._structure_viewer._camera_state)
    cluster_item = first._batch_table.topLevelItem(0)
    assert cluster_item is not None
    cluster_item.setSelected(True)
    first._batch_table.setCurrentItem(cluster_item)
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
    restored.show()
    qapp.processEvents()
    expected_zpe_flags = [
        entry.get("zero_point_energy_ev") is not None for entry in snapshot["batch_results"] if isinstance(entry, dict)
    ]

    assert "Total potential" in restored._probe_output.toPlainText()
    assert restored._line_data.toPlainText().startswith("distance_A,total_eV")
    assert restored._batch_table.topLevelItemCount() >= 1
    assert restored._batch_count.isEnabled() is True
    assert restored._batch_min_muon_muon.isEnabled() is True
    assert restored._batch_optimization_tolerance.value() == pytest.approx(3e-5)
    assert [result.zero_point_energy_ev is not None for result in restored._latest_batch_results] == expected_zpe_flags
    restored_viewer_structure = restored._batch_viewer._structure
    assert restored_viewer_structure is not None
    assert restored_viewer_structure.species.count("X") == 0
    assert restored._batch_viewer._point_overlay_specs
    assert np.allclose(restored._batch_viewer._point_overlay_specs[0].radii, MUON_VIEWER_RADIUS_ANG)
    assert restored._batch_clustering_button.isEnabled() is True
    assert restored._batch_progress.format() == "Batch search complete"
    assert "color: #FFFFFF;" in restored._batch_progress.styleSheet()
    assert restored._batch_zpe_progress.format() == "ZPE calculation complete"
    assert "color: #FFFFFF;" in restored._batch_zpe_progress.styleSheet()
    assert restored._clustering_dialog is None

    restored._open_clustering_dialog()

    assert restored._clustering_dialog is not None
    assert restored._clustering_dialog._method_key() == "kmeans"
    assert restored._clustering_dialog._kmeans_k_spin.value() == 2
    assert restored._clustering_dialog._structure_viewer._show_bonds.isChecked() is True
    assert restored._clustering_dialog._structure_viewer._camera_state == _CameraState(29.0, 61.0, -8.0)
    restored.close()


def test_uep_wizard_batch_zpe_starts_worker_thread_and_updates_progress(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()

    started: list[object] = []
    monkeypatch.setattr("mufinder.app.uep_wizard._BatchZpeThread.start", lambda self: started.append(self))
    monkeypatch.setattr(dialog, "_batch_has_successful_optimizations", lambda: True)

    dialog._run_batch_zpe()

    assert len(started) == 1
    assert dialog._batch_zpe_thread is started[0]
    assert dialog._batch_calculate_zpe.isEnabled() is False
    assert dialog._batch_zpe_progress.minimum() == 0
    assert dialog._batch_zpe_progress.maximum() == 0
    assert dialog._batch_zpe_progress.format() == "Preparing ZPE..."
    dialog.close()


def test_uep_wizard_batch_zpe_populates_quantum_energies(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    _run_batch_zpe_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._batch_max_steps.setValue(5)
    dialog._run_batch_search()
    assert all(result.zero_point_energy_ev is None for result in dialog._latest_batch_results)
    monkeypatch.setattr(dialog, "_batch_has_successful_optimizations", lambda: True)

    dialog._run_batch_zpe()

    assert dialog._batch_zpe_progress.format() == "ZPE calculation complete"
    assert "color: #FFFFFF;" in dialog._batch_zpe_progress.styleSheet()
    assert all(
        result.zero_point_energy_ev is None or isinstance(result.zero_point_energy_ev, float)
        for result in dialog._latest_batch_results
    )
    assert "Stable minima with finite ZPE" in dialog._batch_summary.toPlainText()
    dialog.close()


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
    first._apply_potential_grid_cache(grid)
    first._render_isosurface_mesh(mesh)
    snapshot = first.state_snapshot()
    first.close()

    # As MainWindow does, hand the grid back from its persisted result.
    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_potential_grid=grid,
        initial_state=snapshot,
    )

    assert restored._latest_isosurface_mesh is not None
    assert restored._isosurface_viewer._isosurface_vertices is not None
    assert restored._isosurface_min_label.text().endswith("eV")
    assert restored._isosurface_max_label.text().endswith("eV")
    assert "Displaying" in restored._isosurface_status_label.text()
    assert restored._isosurface_slider.isEnabled() is True
    assert restored._isosurface_level_input.isEnabled() is True
    assert restored._potential_grid_cache is not None
    assert restored._isosurface_color_button.toolTip() == "Choose the isosurface preview color."
    assert len(restored._isosurface_layers) == 1
    restored.close()


def test_uep_wizard_stays_open_when_isosurface_calculation_completes(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog.show()
    qapp.processEvents()

    dialog._start_isosurface_grid_calculation()
    qapp.processEvents()

    assert dialog.isVisible() is True
    assert dialog._potential_grid_cache is not None
    assert dialog._active_isosurface_layer().mesh is not None
    assert "filled isosurface preview" in dialog._isosurface_status_label.text()
    dialog.close()


def test_uep_wizard_recalculating_grid_refreshes_all_isosurface_layers(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    minimum = float(grid.potential_total_ev.min())
    maximum = float(grid.potential_total_ev.max())
    first_level = minimum + 0.35 * (maximum - minimum)
    second_level = minimum + 0.7 * (maximum - minimum)
    dialog._apply_potential_grid_cache(grid)
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=first_level))
    dialog._add_isosurface_layer()
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=second_level))
    dialog._on_isosurface_tab_changed(0)

    dialog._start_isosurface_grid_calculation()
    qapp.processEvents()

    assert len(dialog._isosurface_layers) == 2
    assert all(layer.mesh is not None for layer in dialog._isosurface_layers)
    layer_statuses = [(layer.status_kind, layer.status_text) for layer in dialog._isosurface_layers]
    assert all(layer.status_kind == "success" for layer in dialog._isosurface_layers), layer_statuses
    assert not any("Calculating" in layer.status_text for layer in dialog._isosurface_layers)
    assert len(dialog._isosurface_viewer._isosurface_specs) == 2
    dialog.close()


def test_uep_wizard_recalculating_grid_keeps_visible_isosurface_until_ready(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    minimum = float(grid.potential_total_ev.min())
    maximum = float(grid.potential_total_ev.max())
    level = minimum + 0.45 * (maximum - minimum)
    dialog._apply_potential_grid_cache(grid)
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=level))
    original_mesh = dialog._active_isosurface_layer().mesh
    original_specs = list(dialog._isosurface_viewer._isosurface_specs)
    assert original_mesh is not None
    assert original_specs

    sync_calls: list[str] = []
    original_sync = dialog._sync_isosurface_viewer_layers

    def _record_sync() -> None:
        sync_calls.append("sync")
        original_sync()

    monkeypatch.setattr(dialog, "_sync_isosurface_viewer_layers", _record_sync)
    monkeypatch.setattr("mufinder.app.uep_wizard._IsosurfaceGridThread.start", lambda _thread: None)

    dialog._start_isosurface_grid_calculation()

    assert sync_calls == []
    assert dialog._active_isosurface_layer().mesh is original_mesh
    assert dialog._isosurface_viewer._isosurface_specs == original_specs
    assert dialog._active_isosurface_layer().status_kind == "busy"
    dialog.close()


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
    assert dialog._potential_grid_preview_cache is not None
    assert dialog._potential_grid_preview_cache.grid[0] <= grid.grid[0]

    target_level = minimum + 0.75 * (maximum - minimum)
    dialog._set_isosurface_level(target_level)
    level = dialog._current_isosurface_level()
    assert level == pytest.approx(target_level, abs=1e-3)

    mesh = extract_isosurface(grid, level_ev=level if level is not None else target_level)
    dialog._render_isosurface_mesh(mesh)

    assert dialog._isosurface_min_label.text() == f"{minimum:.2f} eV"
    assert dialog._isosurface_max_label.text() == f"{maximum:.2f} eV"
    assert "filled isosurface preview" in dialog._isosurface_status_label.text()
    assert dialog._isosurface_status_kind == "success"
    assert dialog._isosurface_viewer._isosurface_color == "#dc2626"
    assert dialog._active_isosurface_layer().mesh is not None
    dialog.close()


def test_uep_wizard_isosurface_refresh_preserves_manual_spin_visibility(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    level = 0.5 * (float(grid.potential_total_ev.min()) + float(grid.potential_total_ev.max()))
    mesh = extract_isosurface(grid, level_ev=level)

    assert dialog._isosurface_viewer._show_spins.isChecked() is True
    dialog._isosurface_viewer._show_spins.setChecked(False)

    dialog._render_isosurface_mesh(mesh)

    assert dialog._isosurface_viewer._show_spins.isChecked() is False
    dialog.close()


def test_uep_wizard_choose_isosurface_color_updates_viewer_and_state(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    level = 0.5 * (float(grid.potential_total_ev.min()) + float(grid.potential_total_ev.max()))
    mesh = extract_isosurface(grid, level_ev=level)
    dialog._apply_potential_grid_cache(grid)
    dialog._render_isosurface_mesh(mesh)

    monkeypatch.setattr(
        "mufinder.app.uep_wizard.QColorDialog.getColor",
        lambda *_args, **_kwargs: QColor("#12ab34"),
    )

    dialog._choose_isosurface_color()

    assert dialog._active_isosurface_layer().color == "#12ab34"
    assert dialog._isosurface_viewer._isosurface_color == "#12ab34"
    assert dialog._isosurface_layer_tabs._layer_colors[0].name() == "#12ab34"
    assert dialog.state_snapshot()["isosurface_color"] == "#12ab34"
    assert dialog._isosurface_color_button.text() == ""
    dialog.close()


def test_uep_wizard_multi_isosurface_tabs_add_switch_delete_and_restore(qapp) -> None:
    first = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(first._field)
    minimum = float(grid.potential_total_ev.min())
    maximum = float(grid.potential_total_ev.max())
    first_level = minimum + 0.35 * (maximum - minimum)
    second_level = minimum + 0.7 * (maximum - minimum)
    first._apply_potential_grid_cache(grid)
    first._render_isosurface_mesh(extract_isosurface(grid, level_ev=first_level))
    first._set_isosurface_color("#dc2626", emit_state=False)

    first._add_isosurface_layer()

    assert len(first._isosurface_layers) == 2
    assert first._active_isosurface_index == 1
    assert first._isosurface_layer_tabs.isHidden() is False
    assert first._isosurface_delete_button.isEnabled() is True

    first._set_isosurface_color("#12ab34", emit_state=False)
    first._render_isosurface_mesh(extract_isosurface(grid, level_ev=second_level))

    assert len(first._isosurface_viewer._isosurface_specs) == 2
    assert first._isosurface_viewer._isosurface_specs[0].color == "#dc2626"
    assert first._isosurface_viewer._isosurface_specs[1].color == "#12ab34"
    assert first._isosurface_layer_tabs._layer_colors[1].name() == "#12ab34"
    assert first._isosurface_legend_entries_layout.count() >= 2

    first._on_isosurface_tab_changed(0)

    assert first._active_isosurface_index == 0
    assert first._active_isosurface_layer().color == "#dc2626"
    assert first._current_isosurface_level() == pytest.approx(first_level, abs=1e-3)
    first_legend_row = first._isosurface_legend_entries_layout.itemAt(0).widget()
    assert first_legend_row is not None
    legend_texts = [label.text() for label in first_legend_row.findChildren(type(first._isosurface_min_label))]
    assert any(text.endswith(" eV") for text in legend_texts)

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

    assert len(restored._isosurface_layers) == 2
    assert restored._active_isosurface_index == 0
    assert restored._isosurface_layer_tabs.count() == 2
    assert len(restored._isosurface_viewer._isosurface_specs) == 2
    assert restored._isosurface_layers[0].color == "#dc2626"
    assert restored._isosurface_layers[1].color == "#12ab34"
    assert restored._isosurface_legend_entries_layout.count() >= 2

    restored._on_isosurface_tab_changed(1)
    assert restored._active_isosurface_index == 1
    assert restored._current_isosurface_level() == pytest.approx(second_level, abs=1e-3)

    restored._delete_current_isosurface_layer()
    assert len(restored._isosurface_layers) == 1
    assert restored._isosurface_layer_tabs.isHidden() is False
    assert len(restored._isosurface_viewer._isosurface_specs) == 1
    restored.close()


def test_uep_wizard_switching_isosurface_tabs_does_not_rebuild_tab_bar(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    minimum = float(grid.potential_total_ev.min())
    maximum = float(grid.potential_total_ev.max())
    first_level = minimum + 0.35 * (maximum - minimum)
    second_level = minimum + 0.7 * (maximum - minimum)
    dialog._apply_potential_grid_cache(grid)
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=first_level))
    dialog._add_isosurface_layer()
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=second_level))

    remove_calls: list[int] = []
    original_remove_tab = dialog._isosurface_layer_tabs.removeTab

    def _record_remove(index: int) -> None:
        remove_calls.append(int(index))
        original_remove_tab(index)

    monkeypatch.setattr(dialog._isosurface_layer_tabs, "removeTab", _record_remove)

    dialog._isosurface_layer_tabs.setCurrentIndex(0)
    qapp.processEvents()

    assert remove_calls == []
    assert dialog._active_isosurface_index == 0
    assert dialog._active_isosurface_layer().mesh is not None
    dialog.close()


def test_uep_wizard_delete_current_isosurface_via_button_keeps_window_open(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    minimum = float(grid.potential_total_ev.min())
    maximum = float(grid.potential_total_ev.max())
    first_level = minimum + 0.35 * (maximum - minimum)
    second_level = minimum + 0.7 * (maximum - minimum)
    dialog._apply_potential_grid_cache(grid)
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=first_level))
    dialog._add_isosurface_layer()
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=second_level))
    dialog.show()
    qapp.processEvents()

    dialog._isosurface_delete_button.click()
    qapp.processEvents()

    assert dialog.isVisible() is True
    assert len(dialog._isosurface_layers) == 1
    assert dialog._isosurface_layer_tabs.count() == 1
    assert dialog._active_isosurface_index == 0
    dialog.close()


def test_uep_wizard_isosurface_mesh_busy_keeps_level_controls_enabled(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._isosurface_mesh_thread = object()

    dialog._set_isosurface_busy(True, status_text="Extracting isosurface preview...")

    assert dialog._isosurface_slider.isEnabled() is True
    assert dialog._isosurface_level_input.isEnabled() is True
    assert dialog._isosurface_compute_button.isEnabled() is False
    dialog._isosurface_mesh_thread = None
    dialog.close()


def test_uep_wizard_slider_schedule_triggers_live_preview_and_debounced_request(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._set_isosurface_level(dialog._current_isosurface_level() or 0.0)

    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(
        dialog,
        "_render_live_isosurface_preview",
        lambda level: calls.append(("live", float(level))),
    )
    monkeypatch.setattr(
        dialog,
        "_queue_isosurface_request",
        lambda level, *, immediate: calls.append(("queue", float(level))),
    )

    dialog._schedule_isosurface_from_slider()

    assert len(calls) == 2
    assert calls[0][0] == "live"
    assert calls[1][0] == "queue"
    assert len(calls) == 2
    assert calls[0][1] == pytest.approx(calls[1][1])
    dialog.close()


def test_uep_wizard_live_preview_status_explains_approximate_then_refined(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))

    dialog._render_live_isosurface_preview(dialog._current_isosurface_level() or 0.0)

    status = dialog._isosurface_status_label.text()
    assert "approximate live preview" in status
    assert "pause or release the slider" in status
    assert dialog._isosurface_status_kind == "info"
    dialog.close()


def test_uep_wizard_live_preview_does_not_update_legend_until_refined(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    minimum = float(grid.potential_total_ev.min())
    maximum = float(grid.potential_total_ev.max())
    settled_level = minimum + 0.35 * (maximum - minimum)
    preview_level = minimum + 0.7 * (maximum - minimum)
    dialog._apply_potential_grid_cache(grid)
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=settled_level))

    legend_row = dialog._isosurface_legend_entries_layout.itemAt(0).widget()
    assert legend_row is not None
    settled_texts = [label.text() for label in legend_row.findChildren(type(dialog._isosurface_min_label))]
    assert any(f"{settled_level:.6f} eV" == text for text in settled_texts)

    dialog._render_live_isosurface_preview(preview_level)

    legend_row_after = dialog._isosurface_legend_entries_layout.itemAt(0).widget()
    assert legend_row_after is not None
    after_texts = [label.text() for label in legend_row_after.findChildren(type(dialog._isosurface_min_label))]
    assert any(f"{settled_level:.6f} eV" == text for text in after_texts)
    assert all(f"{preview_level:.6f} eV" != text for text in after_texts)
    dialog.close()


def test_uep_wizard_slider_sync_does_not_rebuild_legend(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._set_isosurface_level(dialog._current_isosurface_level() or 0.0)

    calls = {"legend": 0}
    monkeypatch.setattr(
        dialog,
        "_refresh_isosurface_legend",
        lambda: calls.__setitem__("legend", calls["legend"] + 1),
    )

    dialog._isosurface_slider.setValue(min(dialog._isosurface_slider.maximum(), dialog._isosurface_slider.value() + 25))

    assert calls["legend"] == 0
    dialog.close()


def test_uep_wizard_dispatch_waits_for_slider_release_before_refining(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._pending_isosurface_level_ev = dialog._current_isosurface_level()
    dialog._pending_isosurface_layer_index = dialog._active_isosurface_index
    dialog._isosurface_slider.setSliderDown(True)

    calls = {"timer": 0}
    monkeypatch.setattr(
        dialog._isosurface_preview_timer,
        "start",
        lambda: calls.__setitem__("timer", calls["timer"] + 1),
    )
    monkeypatch.setattr(
        dialog,
        "_start_isosurface_mesh_calculation",
        lambda _level: pytest.fail("refined extraction should wait until the slider is released"),
    )

    dialog._dispatch_pending_isosurface_request()

    assert calls["timer"] == 1
    assert dialog._pending_isosurface_level_ev is not None
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


def test_uep_wizard_zero_minimum_toggle_shifts_displayed_grid_and_mesh(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    raw_minimum = float(grid.potential_total_ev.min())
    raw_maximum = float(grid.potential_total_ev.max())
    raw_level = raw_minimum + 0.35 * (raw_maximum - raw_minimum)
    dialog._apply_potential_grid_cache(grid)
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=raw_level))

    assert dialog._isosurface_zero_minimum.isEnabled() is True
    assert dialog._isosurface_zero_minimum.isChecked() is False
    assert dialog._potential_grid_raw_cache is not None
    assert float(dialog._potential_grid_raw_cache.potential_total_ev.min()) == pytest.approx(raw_minimum)

    dialog._isosurface_zero_minimum.setChecked(True)

    shifted_level = raw_level - raw_minimum
    assert dialog._isosurface_min_label.text() == "0.00 eV"
    assert dialog._isosurface_max_label.text() == f"{(raw_maximum - raw_minimum):.2f} eV"
    assert dialog._active_isosurface_layer().mesh is not None
    assert dialog._active_isosurface_layer().mesh.level_ev == pytest.approx(shifted_level)
    assert dialog._latest_isosurface_mesh is not None
    assert dialog._latest_isosurface_mesh.level_ev == pytest.approx(shifted_level)
    assert dialog._current_isosurface_level() == pytest.approx(shifted_level, abs=1e-3)
    assert dialog._potential_grid_cache is not None
    assert float(dialog._potential_grid_cache.potential_total_ev.min()) == pytest.approx(0.0)

    legend_row = dialog._isosurface_legend_entries_layout.itemAt(0).widget()
    assert legend_row is not None
    legend_texts = [label.text() for label in legend_row.findChildren(type(dialog._isosurface_min_label))]
    assert any(text == f"{shifted_level:.6f} eV" for text in legend_texts)
    dialog.close()


def test_uep_wizard_zero_minimum_export_uses_shifted_grid(qapp, monkeypatch, tmp_path) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._isosurface_zero_minimum.setChecked(True)

    captured: dict[str, object] = {}
    output_path = tmp_path / "uep_export_zeroed.xsf"
    monkeypatch.setattr(
        "mufinder.app.uep_wizard.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(output_path), "XSF Files (*.xsf)"),
    )

    def _capture_export(path, *, structure, potential):
        captured["path"] = Path(path)
        captured["potential"] = potential
        Path(path).write_text("mock xsf", encoding="utf-8")
        return Path(path)

    monkeypatch.setattr("mufinder.app.uep_wizard.write_potential_grid_xsf", _capture_export)

    dialog._export_potential_grid_xsf()

    exported_grid = captured["potential"]
    assert float(exported_grid.potential_total_ev.min()) == pytest.approx(0.0)
    assert output_path.exists()
    dialog.close()


def test_uep_wizard_zero_minimum_snapshot_restores_raw_and_shifted_grid(qapp) -> None:
    first = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(first._field)
    raw_minimum = float(grid.potential_total_ev.min())
    raw_maximum = float(grid.potential_total_ev.max())
    raw_level = raw_minimum + 0.4 * (raw_maximum - raw_minimum)
    first._apply_potential_grid_cache(grid)
    first._render_isosurface_mesh(extract_isosurface(grid, level_ev=raw_level))
    first._isosurface_zero_minimum.setChecked(True)

    snapshot = first.state_snapshot()
    assert snapshot["isosurface_zero_minimum"] is True
    # The raw grid is persisted once, as its own result, and handed back
    # through ``initial_potential_grid``; the per-change UI state omits it.
    assert "potential_grid" not in snapshot
    first.close()

    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_potential_grid=grid,
        initial_state=snapshot,
    )

    assert restored._isosurface_zero_minimum.isChecked() is True
    assert restored._potential_grid_raw_cache is not None
    assert float(restored._potential_grid_raw_cache.potential_total_ev.min()) == pytest.approx(raw_minimum)
    assert restored._potential_grid_cache is not None
    assert float(restored._potential_grid_cache.potential_total_ev.min()) == pytest.approx(0.0)
    assert restored._active_isosurface_layer().mesh is not None
    assert restored._active_isosurface_layer().mesh.level_ev == pytest.approx(raw_level - raw_minimum)
    assert restored._isosurface_min_label.text() == "0.00 eV"
    restored.close()


def test_uep_wizard_restores_legacy_state_grid_and_single_isosurface_mesh(qapp) -> None:
    # Projects saved before the state snapshot dropped its "potential_grid" and
    # top-level "isosurface_mesh" copies still restore from them.
    baseline = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(baseline._field)
    level = float(grid.potential_total_ev.min()) + 0.4 * float(np.ptp(grid.potential_total_ev))
    mesh = extract_isosurface(grid, level_ev=level)
    baseline.close()

    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_state={
            "potential_grid": serialize_potential_grid(grid),
            "isosurface_mesh": {
                "level_ev": mesh.level_ev,
                "minimum_potential_ev": mesh.minimum_potential_ev,
                "maximum_potential_ev": mesh.maximum_potential_ev,
                "vertices_ang": [list(vertex) for vertex in mesh.vertices_ang],
                "faces": [list(face) for face in mesh.faces],
            },
        },
    )

    assert restored._potential_grid_raw_cache is not None
    assert np.allclose(restored._potential_grid_raw_cache.potential_total_ev, grid.potential_total_ev)
    restored_mesh = restored._active_isosurface_layer().mesh
    assert restored_mesh is not None
    assert restored_mesh.level_ev == pytest.approx(level)
    assert len(restored_mesh.faces) == len(mesh.faces)
    snapshot = restored.state_snapshot()
    assert "potential_grid" not in snapshot and "isosurface_mesh" not in snapshot
    assert snapshot["isosurface_layers"][0]["mesh"]["level_ev"] == pytest.approx(level)
    restored.close()


def test_uep_wizard_state_snapshot_reuses_serialized_isosurface_mesh(qapp) -> None:
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
    low, high = float(grid.potential_total_ev.min()), float(grid.potential_total_ev.max())
    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=low + 0.4 * (high - low)))

    first = dialog.state_snapshot()["isosurface_layers"][0]["mesh"]
    assert first is not None
    assert dialog.state_snapshot()["isosurface_layers"][0]["mesh"] is first

    dialog._render_isosurface_mesh(extract_isosurface(grid, level_ev=low + 0.6 * (high - low)))
    second = dialog.state_snapshot()["isosurface_layers"][0]["mesh"]
    assert second is not first
    assert second["level_ev"] == pytest.approx(low + 0.6 * (high - low))
    assert len(dialog._serialized_mesh_cache) == 1
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


def test_uep_wizard_viewer_aligns_project_spins_when_castep_reorders_atoms() -> None:
    source_structure = _source_structure_with_spins()
    source_structure.species = ["Co", "Co", "F", "F"]
    source_structure.fractional_positions = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.0],
        [0.75, 0.75, 0.0],
    ]
    source_structure.properties["atomic_spins"] = [1.0, -1.0, 0.0, 0.0]

    dialog = UepAnalysisDialog.__new__(UepAnalysisDialog)
    dialog._source_job_label = "uep_source_row1"
    dialog._source_structure = source_structure
    dialog._source = SimpleNamespace(
        structure=SimpleNamespace(
            lattice_vectors=source_structure.lattice_vectors,
            species=("F", "F", "Co", "Co"),
            fractional_positions=(
                (0.25, 0.25, 0.0),
                (0.75, 0.75, 0.0),
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.5),
            ),
            atomic_spins=None,
        )
    )

    viewer_structure = dialog._build_viewer_host_structure()

    assert viewer_structure.species == ["F", "F", "Co", "Co"]
    assert viewer_structure.properties["atomic_spins"] == pytest.approx([0.0, 0.0, 1.0, -1.0])


def test_uep_wizard_viewer_hides_tiny_mulliken_ligand_spins() -> None:
    source_structure = _source_structure_with_spins()
    dialog = UepAnalysisDialog.__new__(UepAnalysisDialog)
    dialog._source_job_label = "uep_source_row1"
    dialog._source_structure = source_structure
    dialog._source = SimpleNamespace(
        structure=SimpleNamespace(
            lattice_vectors=source_structure.lattice_vectors,
            species=("F", "F", "Co", "Co"),
            fractional_positions=(
                (0.25, 0.25, 0.0),
                (0.75, 0.75, 0.0),
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.5),
            ),
            atomic_spins=(-0.037, 0.038, 2.658, -2.656),
        )
    )

    viewer_structure = dialog._build_viewer_host_structure()

    assert viewer_structure.properties["atomic_spins"] == pytest.approx([0.0, 0.0, 2.658, -2.656])


def test_uep_wizard_slice_controls_share_zero_minimum_with_isosurface(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))

    dialog._slice_zero_minimum.setChecked(True)

    assert dialog._slice_zero_minimum.isChecked() is True
    assert dialog._isosurface_zero_minimum.isChecked() is True
    assert dialog._isosurface_min_label.text() == "0.00 eV"
    assert dialog._slice_min_label.text() == "0.00 eV"
    assert dialog._slice_colorbar_figure.get_constrained_layout() is False
    dialog.close()


def test_uep_wizard_slice_saturation_window_shifts_with_zero_minimum_reference(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    grid = potential_grid(dialog._field)
    raw_minimum = float(grid.potential_total_ev.min())
    raw_maximum = float(grid.potential_total_ev.max())
    dialog._apply_potential_grid_cache(grid)
    dialog._set_slice_bounds(raw_minimum + 0.5, raw_minimum + 1.5, emit_state=False)

    dialog._slice_zero_minimum.setChecked(True)

    assert dialog._slice_min_input.value() == pytest.approx(0.5)
    assert dialog._slice_max_input.value() == pytest.approx(1.5)
    colorbar_axis = dialog._slice_colorbar_figure.axes[0]
    texts = [text.get_text() for text in colorbar_axis.texts]
    assert any(text.startswith("1.500") for text in texts)
    assert any(text.startswith("0.500") for text in texts)

    dialog._slice_zero_minimum.setChecked(False)

    assert dialog._slice_min_input.value() == pytest.approx(raw_minimum + 0.5)
    assert dialog._slice_max_input.value() == pytest.approx(raw_minimum + 1.5)
    assert dialog._slice_min_label.text() == f"{raw_minimum:.2f} eV"
    assert dialog._slice_max_label.text() == f"{raw_maximum:.2f} eV"
    dialog.close()


def test_uep_wizard_adds_slice_plane_and_updates_viewer(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    plane = _slice_plane_for_dialog(dialog)

    class _FakeDialog:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def plane(self):
            return plane

    monkeypatch.setattr("mufinder.app.uep_wizard._AddSlicePlaneDialog", _FakeDialog)

    dialog._open_add_slice_plane_dialog()

    assert len(dialog._slice_planes) == 1
    assert dialog._slice_planes[0].sample is not None
    assert len(dialog._slice_viewer._slice_specs) == 1
    assert dialog._slice_viewer._slice_specs[0].cmap == "coolwarm"
    assert dialog._slice_viewer._slice_specs[0].alpha == pytest.approx(0.72)
    assert dialog._slice_plane_buttons_layout.count() >= 1
    dialog.close()


def test_uep_wizard_adding_slice_plane_preserves_viewer_camera(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    plane = _slice_plane_for_dialog(dialog)
    dialog._slice_viewer._camera_state = _CameraState(17.0, 38.0, 9.0)
    dialog._slice_viewer._axis.set_from_state(dialog._slice_viewer._camera_state)
    dialog._slice_viewer._limits = ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))

    class _FakeDialog:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def plane(self):
            return plane

    monkeypatch.setattr("mufinder.app.uep_wizard._AddSlicePlaneDialog", _FakeDialog)

    dialog._open_add_slice_plane_dialog()

    assert dialog._slice_viewer._camera_state == _CameraState(17.0, 38.0, 9.0)
    assert dialog._slice_viewer._limits == ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))
    dialog.close()


def test_uep_wizard_sync_slice_viewer_restores_live_pyvista_snapshot(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    viewer = dialog._slice_viewer
    viewer._backend_name = "pyvista"
    viewer._camera_state = _CameraState(17.0, 38.0, 9.0)
    viewer._axis.set_from_state(viewer._camera_state)
    viewer._limits = ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))

    live_position = np.array([9.0, 8.0, 7.0], dtype=float)
    live_focal_point = np.array([1.0, 2.0, 3.0], dtype=float)
    live_up = np.array([0.0, 1.0, 0.0], dtype=float)
    camera = SimpleNamespace(
        position=tuple(live_position),
        focal_point=tuple(live_focal_point),
        up=tuple(live_up),
        parallel_scale=2.75,
    )
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(camera=camera),
        draw_idle=lambda: None,
    )
    viewer._pyvista_camera_snapshot = (
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float),
        0.25,
    )

    monkeypatch.setattr(viewer, "set_slice_planes", lambda _surfaces: setattr(camera, "parallel_scale", 0.5))
    render_calls = {"count": 0}
    monkeypatch.setattr(viewer, "_render", lambda: render_calls.__setitem__("count", render_calls["count"] + 1))

    dialog._sync_slice_viewer()

    assert render_calls["count"] == 1
    assert viewer._pyvista_camera_snapshot is not None
    position, focal_point, up, parallel_scale = viewer._pyvista_camera_snapshot
    assert position == pytest.approx(live_position)
    assert focal_point == pytest.approx(live_focal_point)
    assert up == pytest.approx(live_up)
    assert parallel_scale == pytest.approx(2.75)
    dialog.close()


def test_uep_wizard_passes_viewer_direction_into_add_slice_plane_dialog(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    monkeypatch.setattr(dialog._slice_viewer, "current_view_direction", lambda: (0.0, 0.0, -1.0))
    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, *args, **kwargs) -> None:
            _ = args
            captured.update(kwargs)

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr("mufinder.app.uep_wizard._AddSlicePlaneDialog", _FakeDialog)

    dialog._open_add_slice_plane_dialog()

    assert captured["default_normal_components"] == pytest.approx(
        cartesian_normal_to_lattice_plane_components(dialog._source.structure.lattice_vectors, (0.0, 0.0, -1.0))
    )
    dialog.close()


def test_uep_wizard_add_slice_plane_dialog_treats_selected_muons_like_atoms(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._add_custom_muon_to_viewer("slice", (0.125, 0.25, 0.375))
    host_count = len(dialog._viewer_host_structure.species)
    dialog._slice_viewer.select_atoms([host_count])
    captured: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, *args, **kwargs) -> None:
            _ = args
            captured.update(kwargs)

        def exec(self):
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr("mufinder.app.uep_wizard._AddSlicePlaneDialog", _FakeDialog)

    dialog._open_add_slice_plane_dialog()

    expected_fractional = (0.125, 0.25, 0.375)
    assert captured["selected_positions_fractional"] == [expected_fractional]
    assert captured["selected_positions_ang"] == pytest.approx(
        [fractional_point_to_cartesian(dialog._source.structure.lattice_vectors, expected_fractional)]
    )
    dialog.close()


def test_uep_wizard_deleting_slice_plane_preserves_viewer_camera(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._slice_planes = [
        _SlicePlaneState(
            label="Plane 1",
            plane=plane_from_fractional_anchor_and_miller(
                anchor_fractional=dialog._source.structure.fractional_positions[0],
                miller_indices=(0.0, 0.0, 1.0),
                lattice_vectors=dialog._source.structure.lattice_vectors,
                points_u=161,
                points_v=161,
            ),
        )
    ]
    dialog._resample_all_slice_planes()
    dialog._slice_viewer._camera_state = _CameraState(21.0, 57.0, -11.0)
    dialog._slice_viewer._axis.set_from_state(dialog._slice_viewer._camera_state)
    dialog._slice_viewer._limits = ((-4.0, 4.0), (-5.0, 5.0), (-6.0, 6.0))

    class _DummySignal:
        def connect(self, _callback) -> None:
            return None

    class _FakeDialog:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)
            self.plane_deleted = _DummySignal()

        def exec(self):
            return 2

    monkeypatch.setattr("mufinder.app.uep_wizard._EditSlicePlaneDialog", _FakeDialog)

    dialog._edit_slice_plane(0)

    assert dialog._slice_planes == []
    assert dialog._slice_viewer._camera_state == _CameraState(21.0, 57.0, -11.0)
    assert dialog._slice_viewer._limits == ((-4.0, 4.0), (-5.0, 5.0), (-6.0, 6.0))
    dialog.close()


def test_uep_wizard_slice_colorbar_marks_clipped_bounds(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._slice_planes = [
        _SlicePlaneState(
            label="Plane 1",
            plane=plane_from_fractional_anchor_and_miller(
                anchor_fractional=dialog._source.structure.fractional_positions[0],
                miller_indices=(0.0, 0.0, 1.0),
                lattice_vectors=dialog._source.structure.lattice_vectors,
                points_u=161,
                points_v=161,
            ),
        )
    ]
    dialog._resample_all_slice_planes()
    sample = dialog._slice_planes[0].sample
    assert sample is not None
    dialog._set_slice_bounds(sample.minimum_potential_ev + 0.05, sample.maximum_potential_ev - 0.05, emit_state=False)

    assert len(dialog._slice_colorbar_figure.axes) == 1
    colorbar_axis = dialog._slice_colorbar_figure.axes[0]
    texts = [text.get_text() for text in colorbar_axis.texts]
    assert any("(and below)" in text for text in texts)
    assert any("(and above)" in text for text in texts)
    assert list(colorbar_axis.get_yticks()) == []
    dialog.close()


def test_uep_wizard_slice_slider_defers_colorbar_refresh_until_release(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._slice_planes = [
        _SlicePlaneState(
            label="Plane 1",
            plane=plane_from_fractional_anchor_and_miller(
                anchor_fractional=dialog._source.structure.fractional_positions[0],
                miller_indices=(0.0, 0.0, 1.0),
                lattice_vectors=dialog._source.structure.lattice_vectors,
                points_u=161,
                points_v=161,
            ),
        )
    ]
    dialog._resample_all_slice_planes()
    full_shape = dialog._slice_viewer._slice_specs[0].x_coordinates.shape
    calls = {"count": 0}
    original_refresh = dialog._refresh_slice_colorbar

    def _counting_refresh() -> None:
        calls["count"] += 1
        original_refresh()

    monkeypatch.setattr(dialog, "_refresh_slice_colorbar", _counting_refresh)

    baseline = calls["count"]
    dialog._on_slice_slider_changed(2500, 7500)
    after_drag = calls["count"]
    drag_shape = dialog._slice_viewer._slice_specs[0].x_coordinates.shape
    dialog._dispatch_pending_slice_preview()
    preview_shape = dialog._slice_viewer._slice_specs[0].x_coordinates.shape
    dialog._on_slice_slider_released()
    released_shape = dialog._slice_viewer._slice_specs[0].x_coordinates.shape

    assert after_drag == baseline
    assert calls["count"] == baseline + 1
    assert drag_shape == full_shape
    assert preview_shape[0] < full_shape[0]
    assert preview_shape[1] < full_shape[1]
    assert released_shape == full_shape
    dialog.close()


def test_uep_wizard_restores_slice_state_and_hide_atoms(qapp) -> None:
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
    plane = _slice_plane_for_dialog(dialog)

    dialog._slice_planes = [_SlicePlaneState(label="Plane 1", plane=plane)]
    dialog._resample_all_slice_planes()
    dialog._slice_hide_atoms.setChecked(True)
    snapshot = dialog.state_snapshot()
    dialog.close()

    # As MainWindow does, hand the grid back from its persisted result.
    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_potential_grid=grid,
        initial_state=snapshot,
    )

    assert len(restored._slice_planes) == 1
    assert restored._slice_planes[0].sample is not None
    assert len(restored._slice_viewer._slice_specs) == 1
    assert restored._slice_hide_atoms.isChecked() is True
    assert restored._slice_viewer._hide_atoms_in_front_of_planes is True
    restored.close()


def test_uep_wizard_close_shuts_down_viewers_and_threads(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    assert dialog.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose) is True
    emitted_states: list[dict[str, object]] = []
    dialog.state_changed.connect(lambda state: emitted_states.append(state))

    viewer_calls = {"isosurface": 0, "slices": 0, "batch": 0}
    monkeypatch.setattr(dialog._isosurface_viewer, "shutdown", lambda: viewer_calls.__setitem__("isosurface", viewer_calls["isosurface"] + 1))
    monkeypatch.setattr(dialog._slice_viewer, "shutdown", lambda: viewer_calls.__setitem__("slices", viewer_calls["slices"] + 1))
    monkeypatch.setattr(dialog._batch_viewer, "shutdown", lambda: viewer_calls.__setitem__("batch", viewer_calls["batch"] + 1))

    class _FakeThread:
        def __init__(self) -> None:
            self.interrupt_calls = 0
            self.quit_calls = 0
            self.wait_calls = 0

        def requestInterruption(self) -> None:
            self.interrupt_calls += 1

        def quit(self) -> None:
            self.quit_calls += 1

        def wait(self) -> bool:
            self.wait_calls += 1
            return True

    grid_thread = _FakeThread()
    mesh_thread = _FakeThread()
    batch_thread = _FakeThread()
    dialog._isosurface_grid_thread = grid_thread
    dialog._isosurface_mesh_thread = mesh_thread
    dialog._batch_search_thread = batch_thread
    current_tab = dialog._tabs.currentIndex()

    dialog.close()

    assert emitted_states
    assert emitted_states[-1]["current_tab"] == current_tab
    assert viewer_calls == {"isosurface": 1, "slices": 1, "batch": 1}
    assert grid_thread.interrupt_calls == 1
    assert grid_thread.quit_calls == 1
    assert grid_thread.wait_calls == 1
    assert mesh_thread.interrupt_calls == 1
    assert mesh_thread.quit_calls == 1
    assert mesh_thread.wait_calls == 1
    assert batch_thread.interrupt_calls == 1
    assert batch_thread.quit_calls == 1
    assert batch_thread.wait_calls == 1
