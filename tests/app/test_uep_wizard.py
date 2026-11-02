"""Tests for the interactive UEP analysis dialog."""

from __future__ import annotations

from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

from mufinder.persistence.codec import to_jsonable
from mufinder.services.project_service import ProjectService
from mufinder.services.uep_grouping import UEP_ANALYSIS_STATES_KEY, regroup_saved_uep_searches
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QPushButton,
    QWidget,
)

from mufinder.app import theme as theme_module
from mufinder.app.theme import cluster_color_map, current_palette
from mufinder.app.uep_wizard import (
    UepAnalysisDialog,
    _AddSlicePlaneDialog,
    _EditSlicePlaneDialog,
    _SlicePlaneState,
)
from mufinder.app.atomic_structure_viewer import (
    MUON_VIEWER_RADIUS_ANG,
    AtomicStructureViewer,
    _CameraState,
)
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.science.site_generation import generate_random_sites, symmetry_equivalent_positions
from mufinder.science.uep import (
    DEFAULT_FORCE_TOLERANCE_EV_PER_ANG,
    DEFAULT_OPTIMIZER_MAX_STEPS,
    StationaryPointClassification,
    load_uep_source,
    UepOptimizationResult,
    cartesian_normal_to_lattice_plane_components,
    extract_isosurface,
    fractional_point_to_cartesian,
    miller_indices_to_cartesian_normal,
    plane_from_fractional_anchor_and_miller,
    potential_grid,
    serialize_potential_grid,
    write_potential_grid_xsf,
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


def _show_level(dialog: UepAnalysisDialog, level_above_minimum: float) -> None:
    """Draw the active isosurface at a level, as the worker's mesh does when it arrives."""

    dialog._render_isosurface_mesh_for_layer(
        extract_isosurface(dialog._potential_grid_cache, level_ev=level_above_minimum),
        layer_index=dialog._active_isosurface_index,
    )


def _run_batch_search_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the search in this thread from seeded starts, so it finds the same minima every time."""

    def _start_inline(thread) -> None:
        thread.run()
        thread.finished.emit()

    monkeypatch.setattr("mufinder.app.uep_wizard._BatchSearchThread.start", _start_inline)
    monkeypatch.setattr(
        "mufinder.app.uep_wizard.generate_random_sites", partial(generate_random_sites, seed=7)
    )


def _run_isosurface_threads_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _start_inline(thread) -> None:
        thread.run()
        thread.finished.emit()

    monkeypatch.setattr("mufinder.app.uep_wizard._IsosurfaceGridThread.start", _start_inline)
    monkeypatch.setattr("mufinder.app.uep_wizard._IsosurfaceMeshThread.start", _start_inline)


def test_uep_analysis_dialog_shows_no_unlaid_widgets_over_the_tab_bar(qapp) -> None:
    # The summary, probe, line and plane pages are no longer tabs. Left visible
    # they were drawn at the default 100x30 geometry over the first tab.
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

    layout = dialog.layout()
    laid_out = {layout.itemAt(index).widget() for index in range(layout.count())}
    stray = [
        child
        for child in dialog.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly)
        if child.isVisible() and not child.isWindow() and child not in laid_out
    ]

    assert stray == []
    assert dialog._probe_output.isVisible() is False
    dialog._probe_point.set_values((0.25, 0.25, 0.25))
    dialog._run_probe()
    assert "Total potential" in dialog._probe_output.toPlainText()
    dialog.close()


def test_uep_analysis_dialog_shrinks_for_small_screens(qapp) -> None:
    """The Potential controls scroll, in a column beside the viewer, rather than force a tall minimum."""
    from PySide6.QtWidgets import QScrollArea

    dialog = _si2_dialog()

    assert dialog.minimumWidth() == 980
    assert dialog.minimumHeight() == 640

    scroll = dialog._potential_controls_scroll
    assert isinstance(scroll, QScrollArea)
    assert scroll.widgetResizable()
    assert not scroll.focusPolicy() & Qt.FocusPolicy.TabFocus
    assert dialog._potential_viewer.minimumHeight() == 320

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


def test_add_slice_plane_dialog_uses_fractional_anchor_and_miller_indices_for_hexagonal_cell(
    qapp,
) -> None:
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
    assert plane.anchor_ang == pytest.approx(
        fractional_point_to_cartesian(lattice_vectors, (0.25, 0.5, 0.75))
    )
    assert plane.normal_ang == pytest.approx(
        miller_indices_to_cartesian_normal(lattice_vectors, (1.0, 0.0, 0.0))
    )
    dialog.close()


def test_add_slice_plane_dialog_defaults_normal_from_current_view_direction(qapp) -> None:
    lattice_vectors = (
        (4.2, 0.0, 0.0),
        (-2.1, 0.5 * np.sqrt(3.0) * 4.2, 0.0),
        (0.0, 0.0, 6.8),
    )
    default_components = cartesian_normal_to_lattice_plane_components(
        lattice_vectors, (0.0, 0.0, -1.0)
    )
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


@pytest.mark.parametrize("confirmed", [False, True])
def test_edit_slice_plane_dialog_confirms_before_removing_the_plane(
    qapp, monkeypatch, confirmed: bool
) -> None:
    lattice_vectors = ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0))
    plane = plane_from_fractional_anchor_and_miller(
        anchor_fractional=(0.5, 0.5, 0.5),
        miller_indices=(0.0, 0.0, 1.0),
        lattice_vectors=lattice_vectors,
        points_u=21,
        points_v=21,
    )
    dialog = _EditSlicePlaneDialog(plane_number=2, plane=plane, lattice_vectors=lattice_vectors)
    asked: list[tuple[object, ...]] = []

    def _confirm(parent, title, text, detail="", accept_label="Remove", *args, **kwargs):
        asked.append((parent, title, text, accept_label))
        return confirmed

    monkeypatch.setattr("mufinder.app.workflow_table_utils.confirm_destructive", _confirm)
    deleted: list[bool] = []
    dialog.plane_deleted.connect(lambda: deleted.append(True))
    delete_button = next(
        button for button in dialog.findChildren(QPushButton) if button.text() == "Remove plane"
    )

    delete_button.click()

    assert asked == [(dialog, "Remove slice plane", "Remove plane 2?", "Remove plane")]
    assert deleted == ([True] if confirmed else [])
    assert dialog.result() == (2 if confirmed else 0)
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
    assert not any(
        button.text().startswith("Promote") for button in dialog.findChildren(QPushButton)
    )
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


def test_the_window_has_two_tabs_one_potential_viewer_and_no_compute_button(qapp) -> None:
    dialog = _si2_dialog()

    labels = [dialog._tabs.tabText(index) for index in range(dialog._tabs.count())]
    palette = current_palette()

    assert labels == ["Batch Search", "Potential"]
    assert dialog._tabs.objectName() == "uepAnalysisTabs"
    assert palette.accent in dialog._tabs.styleSheet()
    assert palette.bg_surface in dialog._tabs.styleSheet()
    viewers = dialog.findChildren(AtomicStructureViewer)
    assert {viewer for viewer in viewers} == {dialog._batch_viewer, dialog._potential_viewer}
    button_texts = {button.text() for button in dialog.findChildren(QPushButton)}
    assert "Calculate Cell Potential Grid" not in button_texts
    assert "Export XSF…" in button_texts
    assert "Add plane…" in button_texts
    assert "Add / remove muon" in button_texts
    checkbox_texts = {box.text() for box in dialog.findChildren(QCheckBox)}
    assert "Set minimum potential to zero" not in checkbox_texts
    assert "Hide atoms in front of planes" in checkbox_texts
    assert dialog._manage_muons_button.isVisibleTo(dialog._potential_tab) is True
    dialog.close()


def test_a_saved_width_factor_is_the_one_in_use_and_a_stored_grid_is_not_read(qapp) -> None:
    baseline = _si2_dialog()
    other_factor_grid = potential_grid(baseline._field)
    baseline.close()

    restored = _si2_dialog(
        initial_state={
            "gw_factor": 7.0,
            "potential_grid": serialize_potential_grid(other_factor_grid),
        },
    )

    assert restored._source.gw_factor == pytest.approx(7.0)
    assert restored._gw_factor_input.value() == pytest.approx(7.0)
    assert restored._potential_grid_cache is None
    restored.close()


@pytest.mark.parametrize(
    ("saved_index", "expected_index"),
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 0),
        (4, 1),
        (5, 1),
        (6, 0),
        (7, 0),
        ("not a tab", 0),
    ],
)
def test_a_saved_tab_restores(qapp, saved_index, expected_index) -> None:
    dialog = _si2_dialog(initial_state={"current_tab": saved_index})

    assert dialog._tabs.currentIndex() == expected_index
    dialog.close()


def test_the_potential_tab_saves_and_restores_as_its_visible_index(qapp) -> None:
    first = _si2_dialog()
    first._tabs.setCurrentWidget(first._potential_tab)
    snapshot = first.state_snapshot()
    first.close()

    assert snapshot["current_tab"] == 1
    restored = _si2_dialog(initial_state=snapshot)
    assert restored._tabs.currentWidget() is restored._potential_tab
    restored.close()


def test_pressing_enter_on_the_width_factor_rebuilds_the_field_and_recalculates_the_grid(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._probe_point.set_values((0.25, 0.25, 0.25))
    dialog._run_probe()
    dialog.show()
    dialog._tabs.setCurrentWidget(dialog._potential_tab)

    started: list[bool] = []
    monkeypatch.setattr(dialog, "_start_isosurface_grid_calculation", lambda: started.append(True))

    dialog._gw_factor_input.setValue(7.0)
    dialog._gw_factor_input.returnPressed.emit()

    assert dialog._source.gw_factor == pytest.approx(7.0)
    assert dialog.state_snapshot()["gw_factor"] == pytest.approx(7.0)
    assert dialog._gw_factor_input.value() == pytest.approx(7.0)
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
    dialog.close()


def test_the_width_factor_is_one_control_above_the_tabs(qapp) -> None:
    dialog = _si2_dialog()

    assert "Press Enter" in dialog._gw_factor_input.toolTip()
    assert not any(button.text() == "Apply" for button in dialog.findChildren(QPushButton))
    assert dialog._tabs.isAncestorOf(dialog._gw_factor_input) is False
    assert (
        sum(1 for label in dialog.findChildren(QLabel) if label.text() == "Gaussian width factor")
        == 1
    )
    dialog.close()


def test_uep_wizard_batch_search_populates_the_groups_when_it_ends_without_direct_promotion(
    qapp, monkeypatch
) -> None:
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
    dialog._batch_max_steps.setValue(30)
    dialog._run_batch_search()

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
    assert sorted(int(label.removeprefix("Start ")) for label in all_site_labels) == list(
        range(1, len(all_site_labels) + 1)
    )
    assert not hasattr(dialog, "promote_requested")
    assert not hasattr(dialog, "_promote_selected_cluster")
    dialog.close()


def test_uep_wizard_state_change_keeps_child_dialogs_after_parent_handlers(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = _si2_dialog()
    events: list[str] = []

    class _FakeChildDialog:
        def close(self) -> bool:
            return True

    muon_dialog = _FakeChildDialog()
    dialog._viewer_muon_dialog = muon_dialog

    def _keep_child(child) -> None:
        if child is muon_dialog:
            events.append("keep-muons")

    monkeypatch.setattr(UepAnalysisDialog, "_keep_dialog_in_front", staticmethod(_keep_child))
    dialog.state_changed.connect(lambda _state: events.append("parent-handler"))

    dialog._emit_state_changed()

    assert events == ["parent-handler", "keep-muons"]
    dialog.close()


def test_uep_wizard_site_selection_updates_viewer_without_direct_promotion(
    qapp, monkeypatch
) -> None:
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
    dialog._batch_max_steps.setValue(30)
    dialog._run_batch_search()

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


def test_uep_wizard_isosurface_muon_management_adds_and_removes_clustered_muons(
    qapp, monkeypatch
) -> None:
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
    dialog._batch_max_steps.setValue(30)
    dialog._run_batch_search()
    dialog._latest_batch_results = [
        _mock_zpe(dialog._field, result) for result in dialog._latest_batch_results
    ]

    candidate = dialog._clustered_muon_candidates()[0]
    dialog._add_candidate_muons_to_viewer([candidate.key], True)

    assert len(dialog._viewer_muons) == 1
    assert dialog._potential_viewer._structure is not None
    assert dialog._potential_viewer._structure.species.count("X") == 1
    expected_color = cluster_color_map([candidate.cluster_index])[candidate.cluster_index]
    assert dialog._potential_viewer._structure.properties["atom_colors"][-1] == expected_color
    viewer_entry = dialog._viewer_muons[0]
    assert viewer_entry.linked_isosurface_layer_key is not None
    layer_index = dialog._find_isosurface_layer_index_by_key(
        viewer_entry.linked_isosurface_layer_key
    )
    assert layer_index is not None
    linked_layer = dialog._isosurface_layers[layer_index]
    assert linked_layer.color == expected_color
    assert linked_layer.managed_muon_key == viewer_entry.key
    assert linked_layer.level_ev == pytest.approx(
        float(candidate.total_energy_ev) - dialog._potential_minimum_ev
    )
    assert linked_layer.mesh is not None

    dialog._set_current_viewer_muon_selection([viewer_entry.key])

    host_count = len(dialog._viewer_host_structure.species)
    assert dialog._potential_viewer._selected_indexes == [host_count]
    legend_texts = [
        label.text()
        for label in dialog._viewer_muon_legend_entries.findChildren(
            type(dialog._isosurface_min_label)
        )
    ]
    assert any(candidate.label in text for text in legend_texts)
    assert any(f"Site {candidate.cluster_index}" in text for text in legend_texts)

    dialog._remove_viewer_muons([viewer_entry.key])

    assert dialog._viewer_muons == []
    assert dialog._potential_viewer._structure is dialog._viewer_host_structure
    assert (
        dialog._find_isosurface_layer_index_by_key(viewer_entry.linked_isosurface_layer_key) is None
    )
    dialog.close()


def test_uep_wizard_editing_a_muon_label_keeps_its_isosurface_and_restores(
    qapp, monkeypatch
) -> None:
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
    dialog._batch_max_steps.setValue(30)
    dialog._run_batch_search()
    dialog._latest_batch_results = [
        _mock_zpe(dialog._field, result) for result in dialog._latest_batch_results
    ]

    candidate = dialog._clustered_muon_candidates()[0]
    dialog._add_candidate_muons_to_viewer([candidate.key], True)
    viewer_entry = dialog._viewer_muons[0]
    assert viewer_entry.linked_isosurface_layer_key is not None
    layer_index = dialog._find_isosurface_layer_index_by_key(
        viewer_entry.linked_isosurface_layer_key
    )
    assert layer_index is not None

    dialog._open_viewer_muon_dialog()
    item = dialog._viewer_muon_dialog._current_table.topLevelItem(0)
    assert item is not None
    assert bool(item.flags() & Qt.ItemFlag.ItemIsEditable)

    item.setText(0, "Renamed Site")
    qapp.processEvents()

    assert dialog._viewer_muons[0].label == "Renamed Site"
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

    assert restored._viewer_muons[0].label == "Renamed Site"
    restored_layer_index = restored._find_isosurface_layer_index_by_key(
        viewer_entry.linked_isosurface_layer_key
    )
    assert restored_layer_index is not None
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
    dialog._potential_viewer._show_spins.setChecked(False)

    dialog._add_custom_muon_to_viewer((0.125, 0.25, 0.375))
    assert dialog._potential_viewer._show_spins.isChecked() is False

    muon_key = dialog._viewer_muons[0].key
    dialog._remove_viewer_muons([muon_key])
    assert dialog._potential_viewer._show_spins.isChecked() is False
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

    first._add_custom_muon_to_viewer((0.125, 0.25, 0.375))
    assert len(first._viewer_muons) == 1
    entry = first._viewer_muons[0]
    first._set_current_viewer_muon_selection([entry.key])

    host_count = len(first._viewer_host_structure.species)
    assert first._potential_viewer._structure is not None
    assert first._potential_viewer._structure.species.count("X") == 1
    assert first._potential_viewer._selected_indexes == [host_count]

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

    assert len(restored._viewer_muons) == 1
    assert restored._potential_viewer._structure is not None
    assert restored._potential_viewer._structure.species.count("X") == 1
    assert restored._viewer_muons[0].fractional_position == pytest.approx((0.125, 0.25, 0.375))
    assert restored._potential_viewer._selected_indexes == [host_count]
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

    dialog._add_custom_muon_to_viewer((0.125, 0.25, 0.375))
    dialog._add_custom_muon_to_viewer((0.375, 0.25, 0.125))

    assert dialog._potential_viewer._structure is not None
    atom_colors = dialog._potential_viewer._structure.properties["atom_colors"]
    assert atom_colors[-2] != atom_colors[-1]
    legend_texts = [
        label.text() for label in dialog._viewer_muon_legend_entries.findChildren(QLabel)
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

    dialog._add_custom_muon_to_viewer((0.125, 0.25, 0.375))
    dialog._open_viewer_muon_dialog()
    qapp.processEvents()

    scroll = dialog._viewer_muon_dialog._current_table.horizontalScrollBar()
    header = dialog._viewer_muon_dialog._current_table.header()
    assert scroll is not None
    assert header is not None
    scroll.setValue(scroll.maximum())
    header.setOffset(max(header.offset(), 24))
    qapp.processEvents()

    muon_key = dialog._viewer_muons[0].key
    dialog._remove_viewer_muons([muon_key])
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
    dialog._potential_viewer._show_spins.setChecked(False)

    dialog._add_custom_muon_to_viewer((0.125, 0.25, 0.375))
    assert dialog._potential_viewer._show_spins.isChecked() is False

    muon_key = dialog._viewer_muons[0].key
    dialog._remove_viewer_muons([muon_key])
    assert dialog._potential_viewer._show_spins.isChecked() is False
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

    card = dialog._batch_controls_card
    assert card.objectName() == "analysisSliderPanel"
    assert [
        frame.objectName()
        for frame in dialog._batch_tab.findChildren(QFrame)
        if frame.objectName() == "analysisSliderPanel"
    ] == ["analysisSliderPanel"]
    for control in (
        dialog._batch_count,
        dialog._batch_min_muon_muon,
        dialog._batch_min_muon_atom,
        dialog._batch_run,
        dialog._batch_progress,
        dialog._batch_advanced_toggle,
    ):
        assert control.parentWidget() is card
    assert dialog._batch_progress.format() == "Ready"
    assert dialog._batch_progress.objectName() == "uepBatchProgress"
    assert dialog._batch_progress.styleSheet() == ""
    assert not hasattr(dialog, "_batch_calculate_zpe")
    assert not hasattr(dialog, "_batch_zpe_progress")
    assert not hasattr(dialog, "_batch_gw_factor_input")
    assert not hasattr(dialog, "_batch_promote")
    assert not hasattr(dialog, "_batch_cluster_threshold")
    assert not hasattr(dialog, "_batch_clustering_button")
    assert dialog._batch_grouping_tolerance.value() == pytest.approx(0.25)
    assert dialog._batch_grouping_tolerance.parentWidget() is dialog._batch_table.parentWidget()
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

    dialog._open_viewer_muon_dialog()
    muon_dialog = dialog._viewer_muon_dialog

    assert muon_dialog._include_isosurfaces.parentWidget() is not None
    assert muon_dialog._include_isosurfaces.parentWidget().title() == "Batch-search sites"
    assert [
        muon_dialog._candidate_table.headerItem().text(index)
        for index in range(muon_dialog._candidate_table.columnCount())
    ] == [
        "Site / Start",
        "fX",
        "fY",
        "fZ",
        "Classical energy\n(eV)",
        "ZPE\n(eV)",
        "Total energy\n(eV)",
    ]
    assert [
        muon_dialog._current_table.headerItem().text(index)
        for index in range(muon_dialog._current_table.columnCount())
    ] == [
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
    assert [
        dialog._batch_table.headerItem().text(index)
        for index in range(dialog._batch_table.columnCount())
    ] == [
        "Site / Start",
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
    dialog.close()


def test_uep_wizard_batch_search_starts_worker_thread_and_shows_busy_progress(
    qapp, monkeypatch
) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    started: list[object] = []
    monkeypatch.setattr(
        "mufinder.app.uep_wizard._BatchSearchThread.start", lambda self: started.append(self)
    )

    dialog._run_batch_search()

    assert len(started) == 1
    assert dialog._batch_search_thread is started[0]
    assert dialog._batch_run.isEnabled() is False
    assert dialog._batch_progress.minimum() == 0
    assert dialog._batch_progress.maximum() == 0
    assert dialog._batch_progress.format() == "Packing starts…"
    assert dialog._batch_progress.property("generationState") == "active"
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

    monkeypatch.setattr(
        "mufinder.app.uep_wizard.generate_random_sites", _fake_generate_random_sites
    )

    dialog._batch_count.setValue(6)
    dialog._batch_min_muon_muon.setValue(1.2)
    dialog._batch_force_tolerance.setValue(2e-5)
    dialog._batch_max_steps.setValue(30)
    dialog._run_batch_search()

    summary = dialog._batch_summary.toPlainText()
    assert "Requested random starts: 6" in summary
    assert "Initial starts generated: 1" in summary
    assert "Min muon-muon distance (Å): 1.200" in summary
    assert "Force tolerance (eV/Å): 2e-05" in summary
    assert "muon-muon distance constraint limited packing" in summary
    assert "Grouping: " in summary
    assert "within 0.25 Å and 1 meV" in summary
    assert "available for clustering" not in summary
    assert dialog._batch_progress.format() == "Batch search complete"
    assert dialog._batch_progress.property("generationState") == "complete"
    dialog.close()


def test_uep_wizard_state_snapshot_restores_generated_analysis(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
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
    first._batch_force_tolerance.setValue(3e-5)
    first._batch_max_steps.setValue(30)
    first._run_batch_search()
    assert first._latest_batch_results
    first._batch_grouping_tolerance.setValue(0.4)
    cluster_item = first._batch_table.topLevelItem(0)
    assert cluster_item is not None
    cluster_item.setSelected(True)
    first._batch_table.setCurrentItem(cluster_item)
    snapshot = first.state_snapshot()
    first_summary = first._latest_batch_summary
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
        entry.get("zero_point_energy_ev") is not None
        for entry in snapshot["batch_results"]
        if isinstance(entry, dict)
    ]

    assert "Total potential" in restored._probe_output.toPlainText()
    assert restored._line_data.toPlainText().startswith("distance_A,total_eV")
    assert restored._batch_table.topLevelItemCount() >= 1
    assert restored._batch_count.isEnabled() is True
    assert restored._batch_min_muon_muon.isEnabled() is True
    assert restored._batch_force_tolerance.value() == pytest.approx(3e-5)
    assert [
        result.zero_point_energy_ev is not None for result in restored._latest_batch_results
    ] == expected_zpe_flags
    restored_viewer_structure = restored._batch_viewer._structure
    assert restored_viewer_structure is not None
    assert restored_viewer_structure.species.count("X") == 0
    assert restored._batch_viewer._point_overlay_specs
    assert np.allclose(restored._batch_viewer._point_overlay_specs[0].radii, MUON_VIEWER_RADIUS_ANG)
    assert restored._batch_grouping_tolerance.value() == pytest.approx(0.4)
    assert restored._latest_batch_summary == first_summary
    assert restored._batch_progress.format() == "Batch search complete"
    assert restored._batch_progress.property("generationState") == "complete"
    assert "clustering_window_state" not in snapshot
    assert "batch_zpe_completed" not in snapshot
    restored.close()


def test_the_grid_rebuilds_the_surface_of_every_isosurface(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = _si2_dialog()
    grid = potential_grid(dialog._field)
    span = float(np.ptp(grid.potential_total_ev))
    dialog._apply_potential_grid_cache(grid)
    _show_level(dialog, 0.35 * span)
    dialog._add_isosurface_layer()
    _show_level(dialog, 0.7 * span)
    dialog._on_isosurface_tab_changed(0)

    dialog._start_isosurface_grid_calculation()
    qapp.processEvents()

    assert len(dialog._isosurface_layers) == 2
    assert all(layer.mesh is not None for layer in dialog._isosurface_layers)
    assert len(dialog._potential_viewer._isosurface_specs) == 2
    assert not dialog._potential_status.text()
    dialog.close()


def test_levels_start_at_zero_and_the_slider_stops_at_the_clipped_top(qapp) -> None:
    dialog = _si2_dialog()
    grid = potential_grid(dialog._field)
    dialog._apply_potential_grid_cache(grid)
    values = grid.potential_total_ev
    above_minimum = values - values.min()
    ceiling = float(np.percentile(above_minimum, 95))

    assert dialog._isosurface_level_input.isEnabled() is True
    assert dialog._isosurface_level_input.minimum() == 0.0
    assert dialog._isosurface_min_label.text() == "0 eV"
    assert dialog._isosurface_max_label.text() == f"{ceiling:.2f} eV"
    assert ceiling < float(above_minimum.max())

    dialog._isosurface_slider.setValue(dialog._isosurface_slider.maximum())
    assert dialog._current_isosurface_level() == pytest.approx(ceiling)
    dialog._isosurface_slider.setValue(0)
    assert dialog._current_isosurface_level() == pytest.approx(0.0)
    dialog.close()


def test_a_level_above_the_clipped_top_can_be_typed_and_survives_a_reopen(qapp) -> None:
    dialog = _si2_dialog()
    grid = potential_grid(dialog._field)
    dialog._apply_potential_grid_cache(grid)
    top = float(np.ptp(grid.potential_total_ev))
    typed = 0.9 * top

    dialog._isosurface_level_input.setValue(typed)
    dialog._isosurface_level_input.editingFinished.emit()

    assert dialog._current_isosurface_level() == pytest.approx(typed, abs=1e-3)
    assert dialog._isosurface_slider.value() == dialog._isosurface_slider.maximum()
    snapshot = dialog.state_snapshot()
    dialog.close()

    restored = _si2_dialog(initial_state=snapshot)
    restored._apply_potential_grid_cache(grid)
    assert restored._isosurface_layers[0].level_ev == pytest.approx(typed, abs=1e-3)
    restored.close()


def test_uep_wizard_isosurface_refresh_preserves_manual_spin_visibility(qapp) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))

    assert dialog._potential_viewer._show_spins.isChecked() is True
    dialog._potential_viewer._show_spins.setChecked(False)

    _show_level(dialog, 5.0)

    assert dialog._potential_viewer._show_spins.isChecked() is False
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
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    _show_level(dialog, 5.0)

    monkeypatch.setattr(
        "mufinder.app.uep_wizard.QColorDialog.getColor",
        lambda *_args, **_kwargs: QColor("#12ab34"),
    )

    dialog._choose_isosurface_color()

    assert dialog._active_isosurface_layer().color == "#12ab34"
    assert dialog._potential_viewer._isosurface_color == "#12ab34"
    assert dialog._isosurface_layer_tabs._layer_colors[0].name() == "#12ab34"
    assert dialog.state_snapshot()["isosurface_layers"][0]["color"] == "#12ab34"
    assert dialog._isosurface_color_button.text() == ""
    dialog.close()


def test_uep_wizard_multi_isosurface_tabs_add_switch_delete_and_restore(qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "mufinder.app.workflow_table_utils.confirm_destructive", lambda *_args, **_kwargs: True
    )
    first = _si2_dialog()
    grid = potential_grid(first._field)
    span = float(np.ptp(grid.potential_total_ev))
    first_level = 0.35 * span
    second_level = 0.7 * span
    first._apply_potential_grid_cache(grid)
    _show_level(first, first_level)
    first._set_isosurface_color("#dc2626", emit_state=False)

    first._add_isosurface_layer()

    assert len(first._isosurface_layers) == 2
    assert first._active_isosurface_index == 1
    assert first._isosurface_layer_tabs.isHidden() is False
    assert first._isosurface_delete_button.isEnabled() is True

    first._set_isosurface_color("#12ab34", emit_state=False)
    _show_level(first, second_level)

    assert len(first._potential_viewer._isosurface_specs) == 2
    assert first._potential_viewer._isosurface_specs[0].color == "#dc2626"
    assert first._potential_viewer._isosurface_specs[1].color == "#12ab34"
    assert first._isosurface_layer_tabs._layer_colors[1].name() == "#12ab34"
    assert first._isosurface_legend_entries_layout.count() >= 2

    first._on_isosurface_tab_changed(0)

    assert first._active_isosurface_index == 0
    assert first._active_isosurface_layer().color == "#dc2626"
    assert first._current_isosurface_level() == pytest.approx(first_level, abs=1e-3)
    first_legend_row = first._isosurface_legend_entries_layout.itemAt(0).widget()
    assert first_legend_row is not None
    legend_texts = [
        label.text() for label in first_legend_row.findChildren(type(first._isosurface_min_label))
    ]
    assert any(text.endswith(" eV") for text in legend_texts)

    snapshot = first.state_snapshot()
    first.close()

    _run_isosurface_threads_inline(monkeypatch)
    restored = _si2_dialog(initial_state=snapshot)
    restored.show()
    restored._tabs.setCurrentWidget(restored._potential_tab)
    qapp.processEvents()

    assert len(restored._isosurface_layers) == 2
    assert restored._active_isosurface_index == 0
    assert restored._isosurface_layer_tabs.count() == 2
    assert len(restored._potential_viewer._isosurface_specs) == 2
    assert restored._isosurface_layers[0].color == "#dc2626"
    assert restored._isosurface_layers[1].color == "#12ab34"
    assert restored._isosurface_legend_entries_layout.count() >= 2

    restored._on_isosurface_tab_changed(1)
    assert restored._active_isosurface_index == 1
    assert restored._current_isosurface_level() == pytest.approx(second_level, abs=1e-3)

    restored._delete_current_isosurface_layer()
    assert len(restored._isosurface_layers) == 1
    assert restored._isosurface_layer_tabs.isHidden() is False
    assert len(restored._potential_viewer._isosurface_specs) == 1
    restored.close()


def test_uep_wizard_switching_isosurface_tabs_does_not_rebuild_tab_bar(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    _show_level(dialog, 5.0)
    dialog._add_isosurface_layer()
    _show_level(dialog, 10.0)

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


@pytest.mark.parametrize("confirmed", [False, True])
def test_removing_an_isosurface_asks_first_and_keeps_the_window_open(
    qapp, monkeypatch, confirmed: bool
) -> None:
    asked = _answer_confirmations(monkeypatch, answer=confirmed)
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    _show_level(dialog, 5.0)
    dialog._add_isosurface_layer()
    _show_level(dialog, 10.0)
    dialog.show()
    qapp.processEvents()

    dialog._isosurface_delete_button.click()
    qapp.processEvents()

    assert asked == ["Remove isosurface"]
    assert dialog.isVisible() is True
    assert len(dialog._isosurface_layers) == (1 if confirmed else 2)
    assert dialog._isosurface_layer_tabs.count() == (1 if confirmed else 2)
    dialog.close()


def test_uep_wizard_isosurface_mesh_busy_keeps_level_controls_enabled(qapp) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._isosurface_mesh_thread = object()

    dialog._refresh_isosurface_controls()

    assert dialog._isosurface_slider.isEnabled() is True
    assert dialog._isosurface_level_input.isEnabled() is True
    assert dialog._isosurface_export_button.isEnabled() is False
    dialog._isosurface_mesh_thread = None
    dialog.close()


def test_uep_wizard_slider_schedule_triggers_live_preview_and_debounced_request(
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


def test_uep_wizard_live_preview_does_not_update_legend_until_refined(qapp) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    settled_level = 5.0
    preview_level = 10.0
    _show_level(dialog, settled_level)

    legend_row = dialog._isosurface_legend_entries_layout.itemAt(0).widget()
    assert legend_row is not None
    settled_texts = [
        label.text() for label in legend_row.findChildren(type(dialog._isosurface_min_label))
    ]
    assert any(f"{settled_level:.6f} eV" == text for text in settled_texts)

    dialog._render_live_isosurface_preview(preview_level)

    legend_row_after = dialog._isosurface_legend_entries_layout.itemAt(0).widget()
    assert legend_row_after is not None
    after_texts = [
        label.text() for label in legend_row_after.findChildren(type(dialog._isosurface_min_label))
    ]
    assert any(f"{settled_level:.6f} eV" == text for text in after_texts)
    assert all(f"{preview_level:.6f} eV" != text for text in after_texts)
    dialog.close()


def test_uep_wizard_slider_sync_does_not_rebuild_legend(
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
    dialog._set_isosurface_level(dialog._current_isosurface_level() or 0.0)

    calls = {"legend": 0}
    monkeypatch.setattr(
        dialog,
        "_refresh_isosurface_legend",
        lambda: calls.__setitem__("legend", calls["legend"] + 1),
    )

    dialog._isosurface_slider.setValue(
        min(dialog._isosurface_slider.maximum(), dialog._isosurface_slider.value() + 25)
    )

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
    assert dialog._potential_status.text() == "Exported uep_export.xsf"
    assert dialog._potential_status.property("statusKind") == "exported"
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
    assert dialog._potential_viewer._spins is not None
    assert dialog._potential_viewer._spins.tolist() == pytest.approx([1.5, -1.5])
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


def test_uep_wizard_adds_slice_plane_and_updates_viewer(
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
    assert len(dialog._potential_viewer._slice_specs) == 1
    assert dialog._potential_viewer._slice_specs[0].cmap == "coolwarm"
    assert dialog._potential_viewer._slice_specs[0].alpha == pytest.approx(0.72)
    assert dialog._slice_plane_list.count() == 1
    assert dialog._slice_plane_list.item(0).text() == "Plane 1"
    dialog.close()


def test_uep_wizard_adding_slice_plane_preserves_viewer_camera(
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
    plane = _slice_plane_for_dialog(dialog)
    dialog._potential_viewer._camera_state = _CameraState(17.0, 38.0, 9.0)
    dialog._potential_viewer._axis.set_from_state(dialog._potential_viewer._camera_state)
    dialog._potential_viewer._limits = ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))

    class _FakeDialog:
        def __init__(self, *args, **kwargs) -> None:
            _ = (args, kwargs)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def plane(self):
            return plane

    monkeypatch.setattr("mufinder.app.uep_wizard._AddSlicePlaneDialog", _FakeDialog)

    dialog._open_add_slice_plane_dialog()

    assert dialog._potential_viewer._camera_state == _CameraState(17.0, 38.0, 9.0)
    assert dialog._potential_viewer._limits == ((-1.0, 1.0), (-2.0, 2.0), (-3.0, 3.0))
    dialog.close()


def test_uep_wizard_sync_slice_viewer_restores_live_pyvista_snapshot(
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
    # The live camera shows the scene only once the viewer has drawn it, on show.
    dialog.show()
    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    viewer = dialog._potential_viewer
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

    monkeypatch.setattr(
        viewer, "set_slice_planes", lambda _surfaces: setattr(camera, "parallel_scale", 0.5)
    )
    render_calls = {"count": 0}
    monkeypatch.setattr(
        viewer, "_render", lambda: render_calls.__setitem__("count", render_calls["count"] + 1)
    )

    dialog._sync_slice_viewer()

    assert render_calls["count"] == 1
    assert viewer._pyvista_camera_snapshot is not None
    position, focal_point, up, parallel_scale = viewer._pyvista_camera_snapshot
    assert position == pytest.approx(live_position)
    assert focal_point == pytest.approx(live_focal_point)
    assert up == pytest.approx(live_up)
    assert parallel_scale == pytest.approx(2.75)
    dialog.close()


def test_uep_wizard_passes_viewer_direction_into_add_slice_plane_dialog(
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
    monkeypatch.setattr(
        dialog._potential_viewer, "current_view_direction", lambda: (0.0, 0.0, -1.0)
    )
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
        cartesian_normal_to_lattice_plane_components(
            dialog._source.structure.lattice_vectors, (0.0, 0.0, -1.0)
        )
    )
    dialog.close()


def test_uep_wizard_add_slice_plane_dialog_treats_selected_muons_like_atoms(
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
    dialog._add_custom_muon_to_viewer((0.125, 0.25, 0.375))
    host_count = len(dialog._viewer_host_structure.species)
    dialog._potential_viewer.select_atoms([host_count])
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
        [
            fractional_point_to_cartesian(
                dialog._source.structure.lattice_vectors, expected_fractional
            )
        ]
    )
    dialog.close()


def test_uep_wizard_deleting_slice_plane_preserves_viewer_camera(
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
    dialog._potential_viewer._camera_state = _CameraState(21.0, 57.0, -11.0)
    dialog._potential_viewer._axis.set_from_state(dialog._potential_viewer._camera_state)
    dialog._potential_viewer._limits = ((-4.0, 4.0), (-5.0, 5.0), (-6.0, 6.0))

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
    assert dialog._potential_viewer._camera_state == _CameraState(21.0, 57.0, -11.0)
    assert dialog._potential_viewer._limits == ((-4.0, 4.0), (-5.0, 5.0), (-6.0, 6.0))
    dialog.close()


def test_uep_wizard_slice_colorbar_marks_clipped_bounds(qapp) -> None:
    dialog = _si2_dialog()
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
    dialog._set_slice_bounds(
        sample.minimum_potential_ev + 0.05, sample.maximum_potential_ev - 0.05, emit_state=False
    )

    assert len(dialog._slice_colorbar_figure.axes) == 1
    colorbar_axis = dialog._slice_colorbar_figure.axes[0]
    texts = [label.get_text() for label in colorbar_axis.xaxis.get_ticklabels()]
    assert texts[0].startswith("≤ ")
    assert texts[1].startswith("≥ ")
    assert colorbar_axis.get_xlabel() == "Potential above minimum (eV)"
    assert dialog._slice_colorbar_figure.get_layout_engine() is not None
    dialog.close()


def test_uep_wizard_slice_slider_defers_colorbar_refresh_until_release(
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
    full_shape = dialog._potential_viewer._slice_specs[0].x_coordinates.shape
    calls = {"count": 0}
    original_refresh = dialog._refresh_slice_colorbar

    def _counting_refresh() -> None:
        calls["count"] += 1
        original_refresh()

    monkeypatch.setattr(dialog, "_refresh_slice_colorbar", _counting_refresh)

    baseline = calls["count"]
    dialog._on_slice_slider_changed(2500, 7500)
    after_drag = calls["count"]
    drag_shape = dialog._potential_viewer._slice_specs[0].x_coordinates.shape
    dialog._dispatch_pending_slice_preview()
    preview_shape = dialog._potential_viewer._slice_specs[0].x_coordinates.shape
    dialog._on_slice_slider_released()
    released_shape = dialog._potential_viewer._slice_specs[0].x_coordinates.shape

    assert after_drag == baseline
    assert calls["count"] == baseline + 1
    assert drag_shape == full_shape
    assert preview_shape[0] < full_shape[0]
    assert preview_shape[1] < full_shape[1]
    assert released_shape == full_shape
    dialog.close()


def test_uep_wizard_restores_slice_state_and_hide_atoms(qapp) -> None:
    dialog = _si2_dialog()
    grid = potential_grid(dialog._field)
    dialog._apply_potential_grid_cache(grid)
    plane = _slice_plane_for_dialog(dialog)

    dialog._slice_planes = [_SlicePlaneState(label="Plane 1", plane=plane)]
    dialog._resample_all_slice_planes()
    dialog._slice_hide_atoms.setChecked(True)
    snapshot = dialog.state_snapshot()
    dialog.close()

    restored = _si2_dialog(initial_state=snapshot)

    assert len(restored._slice_planes) == 1
    assert restored._slice_plane_list.count() == 1
    assert restored._slice_hide_atoms.isChecked() is True
    assert restored._potential_viewer._hide_atoms_in_front_of_planes is True
    assert restored._slice_planes[0].sample is None
    restored._apply_potential_grid_cache(grid)
    assert restored._slice_planes[0].sample is not None
    assert len(restored._potential_viewer._slice_specs) == 1
    restored.close()


def test_uep_wizard_close_shuts_down_viewers_and_threads(
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
    assert dialog.testAttribute(Qt.WidgetAttribute.WA_DeleteOnClose) is True
    emitted_states: list[dict[str, object]] = []
    dialog.state_changed.connect(lambda state: emitted_states.append(state))

    viewer_calls = {"potential": 0, "batch": 0}
    monkeypatch.setattr(
        dialog._potential_viewer,
        "shutdown",
        lambda: viewer_calls.__setitem__("potential", viewer_calls["potential"] + 1),
    )
    monkeypatch.setattr(
        dialog._batch_viewer,
        "shutdown",
        lambda: viewer_calls.__setitem__("batch", viewer_calls["batch"] + 1),
    )

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
    assert viewer_calls == {"potential": 1, "batch": 1}
    assert grid_thread.interrupt_calls == 1
    assert grid_thread.quit_calls == 1
    assert grid_thread.wait_calls == 1
    assert mesh_thread.interrupt_calls == 1
    assert mesh_thread.quit_calls == 1
    assert mesh_thread.wait_calls == 1
    assert batch_thread.interrupt_calls == 1
    assert batch_thread.quit_calls == 1
    assert batch_thread.wait_calls == 1


def _noisy_images_of_one_site(host: Structure, count: int, noise: float) -> list:
    """``count`` optimisation endpoints on the symmetry images of one site, with optimiser scatter."""

    images = symmetry_equivalent_positions(
        (0.11, 0.17, 0.23),
        lattice_vectors=host.lattice_vectors,
        host_species=host.species,
        host_positions_frac=host.fractional_positions,
    )
    generator = np.random.default_rng(3)
    endpoints = []
    for index in range(count):
        fractional = np.mod(
            np.asarray(images[index % len(images)]) + generator.normal(0.0, noise, 3), 1.0
        )
        cartesian = fractional @ np.asarray(host.lattice_vectors)
        endpoints.append(
            UepOptimizationResult(
                start_position_ang=tuple(cartesian.tolist()),
                optimized_position_ang=tuple(cartesian.tolist()),
                optimized_fractional_position=tuple(fractional.tolist()),
                classical_energy_ev=-15.976111 + generator.normal(0.0, 1e-11),
                zero_point_energy_ev=None,
                total_energy_ev=None,
                success=True,
                message="converged",
                iterations=3,
                classification=StationaryPointClassification(
                    label="minimum",
                    eigenvalues=(1.0, 1.0, 1.0),
                    positive_count=3,
                    negative_count=0,
                    near_zero_count=0,
                ),
            )
        )
    return endpoints


@pytest.mark.parametrize("noise", [1e-9, 1e-7, 1e-6])
def test_a_batch_search_whose_starts_all_reach_one_site_lists_one_group_when_it_ends(
    qapp, monkeypatch, noise: float
) -> None:
    host = _source_structure_with_spins()
    endpoints = iter(_noisy_images_of_one_site(host, 12, noise))
    monkeypatch.setattr(
        "mufinder.app.uep_wizard.optimize_muon_position", lambda *_a, **_k: next(endpoints)
    )
    _run_batch_search_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=host,
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )

    dialog._batch_count.setValue(12)
    dialog._batch_min_muon_muon.setValue(0.1)
    dialog._run_batch_search()

    assert dialog._batch_table.topLevelItemCount() == 1
    assert dialog._batch_table.topLevelItem(0).childCount() == 12
    dialog.close()


def _dialog_with_endpoints(monkeypatch, endpoints: list, host: Structure) -> UepAnalysisDialog:
    """A window whose batch search returns ``endpoints``, ready to run."""

    remaining = iter(endpoints)
    monkeypatch.setattr(
        "mufinder.app.uep_wizard.optimize_muon_position", lambda *_a, **_k: next(remaining)
    )
    _run_batch_search_inline(monkeypatch)
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=host,
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._batch_count.setValue(len(endpoints))
    dialog._batch_min_muon_muon.setValue(0.1)
    return dialog


def _two_nearby_sites(host: Structure) -> list:
    """Two minima 0.3 Å apart along x at one energy, in a host with no symmetry between them."""

    lattice = host.lattice_vectors
    endpoints = []
    for x in (0.11, 0.11 + 0.3 / lattice[0][0]):
        (endpoint,) = _noisy_images_of_one_site(host, 1, 0.0)
        cartesian = (x * lattice[0][0], 0.17 * lattice[1][1], 0.23 * lattice[2][2])
        endpoints.append(
            replace(
                endpoint,
                optimized_position_ang=cartesian,
                optimized_fractional_position=(x, 0.17, 0.23),
            )
        )
    return endpoints


def _no_symmetry_host() -> Structure:
    return Structure(
        id=uuid4(),
        label="Low symmetry",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 7.0]],
        species=["Si", "Ge", "C"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.31, 0.47, 0.13], [0.71, 0.11, 0.59]],
    )


def test_the_group_row_reports_its_representative_and_the_log_what_was_left_out(
    qapp, monkeypatch
) -> None:
    host = _source_structure_with_spins()
    endpoints = _noisy_images_of_one_site(host, 4, 1e-7)
    saddle = replace(
        endpoints[0],
        classical_energy_ev=-15.0,
        classification=replace(endpoints[0].classification, label="saddle"),
    )
    failed = replace(endpoints[1], success=False)
    dialog = _dialog_with_endpoints(monkeypatch, [*endpoints, saddle, failed], host)

    dialog._run_batch_search()

    item = dialog._batch_table.topLevelItem(0)
    assert dialog._batch_table.topLevelItemCount() == 1
    assert item.text(0) == "Site 1"
    assert item.text(1) == "4"
    expected = dialog._latest_batch_summary.clusters[0].representative_fractional_position
    assert [float(item.text(column)) for column in (2, 3, 4)] == pytest.approx(expected, abs=1e-6)
    assert [item.child(index).text(0) for index in range(item.childCount())] == [
        "Start 1",
        "Start 2",
        "Start 3",
        "Start 4",
    ]
    log = dialog._batch_summary.toPlainText()
    assert "4 minima in 1 site" in log
    assert "Left out 1 failed optimisation and 1 result that is not a minimum." in log
    dialog.close()


def test_the_grouping_tolerance_regroups_live_and_replaces_the_log_line(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    assert dialog._batch_table.topLevelItemCount() == 2

    dialog._batch_grouping_tolerance.setText("0.5")

    assert dialog._batch_table.topLevelItemCount() == 1
    log = dialog._batch_summary.toPlainText()
    assert "2 minima in 1 site, within 0.5 Å" in log
    assert "within 0.25 Å" not in log
    assert dialog._latest_batch_summary.threshold_ang == pytest.approx(0.5)

    dialog._batch_grouping_tolerance.setText("0.25")

    assert dialog._batch_table.topLevelItemCount() == 2
    dialog.close()


def test_the_grouping_tolerance_ignores_a_number_on_its_way_to_a_valid_one(
    qapp, monkeypatch
) -> None:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    before = dialog._latest_batch_summary

    dialog._batch_grouping_tolerance.setText("0.0")

    assert dialog._latest_batch_summary is before
    dialog.close()


def test_regrouping_measures_the_distances_once(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    monkeypatch.setattr(
        "mufinder.science.uep_clustering.symmetry_distance_matrix",
        lambda *_a, **_k: pytest.fail("the distances were measured again"),
    )

    dialog._batch_grouping_tolerance.setText("0.5")

    assert dialog._batch_table.topLevelItemCount() == 1
    dialog.close()


def test_a_regrouped_tolerance_is_saved_once_it_settles_not_on_every_edit(
    qapp, monkeypatch
) -> None:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    emitted: list[float] = []
    dialog.state_changed.connect(
        lambda state: emitted.append(state["batch_summary_data"]["threshold_ang"])
    )

    dialog._batch_grouping_tolerance.setText("0.3")
    dialog._batch_grouping_tolerance.setText("0.35")
    dialog._batch_grouping_tolerance.setText("0.5")

    assert emitted == []
    assert dialog._grouping_settled_timer.isActive()
    dialog._batch_grouping_tolerance.editingFinished.emit()
    assert emitted == [pytest.approx(0.5)]
    assert not dialog._grouping_settled_timer.isActive()
    # Nothing is pending, so leaving the field again saves nothing.
    dialog._batch_grouping_tolerance.editingFinished.emit()
    assert len(emitted) == 1
    dialog.close()


def test_a_zero_point_pass_leaves_the_groups_and_their_order_alone(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    first, second = _two_nearby_sites(host)
    # The ZPE would reverse the two groups if it took part in their order.
    far = replace(
        second,
        classical_energy_ev=second.classical_energy_ev + 0.01,
        optimized_position_ang=(2.0, 3.0, 3.5),
        optimized_fractional_position=(0.4, 0.5, 0.5),
        zero_point_energy_ev=0.0,
        total_energy_ev=second.classical_energy_ev + 0.01,
    )
    near = replace(first, zero_point_energy_ev=1.0, total_energy_ev=first.classical_energy_ev + 1.0)
    dialog = _dialog_with_endpoints(monkeypatch, [far, near], host)

    dialog._run_batch_search()

    summary = dialog._latest_batch_summary
    assert [cluster.representative_index for cluster in summary.clusters] == [1, 0]
    assert [dialog._batch_table.topLevelItem(row).text(6) for row in range(2)] == [
        "1.000000",
        "0.000000",
    ]
    dialog.close()


def test_a_search_that_finds_no_minimum_lists_nothing_and_says_why(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    (endpoint, _other) = _two_nearby_sites(host)
    saddle = replace(endpoint, classification=replace(endpoint.classification, label="saddle"))
    dialog = _dialog_with_endpoints(monkeypatch, [saddle], host)

    dialog._run_batch_search()

    assert dialog._batch_table.topLevelItemCount() == 0
    assert "Grouping: no minima found." in dialog._batch_summary.toPlainText()
    assert dialog._batch_progress.format() == "No minima found"
    dialog.close()


def test_a_saved_search_without_a_grouping_is_grouped_when_the_window_opens(qapp) -> None:
    host = _source_structure_with_spins()
    results = [asdict(result) for result in _noisy_images_of_one_site(host, 12, 1e-7)]

    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=host,
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_state={
            "batch_results": results,
            "batch_summary_text": "Relaxed sites available for clustering: 12",
        },
    )

    assert dialog._batch_table.topLevelItemCount() == 1
    assert dialog._batch_table.topLevelItem(0).childCount() == 12
    assert "available for clustering" not in dialog._batch_summary.toPlainText()
    dialog.close()


def test_a_saved_hierarchical_grouping_is_regrouped_when_the_window_opens(qapp) -> None:
    host = _source_structure_with_spins()
    endpoints = _noisy_images_of_one_site(host, 12, 1e-7)

    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=host,
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_state={
            "batch_results": [asdict(result) for result in endpoints],
            "batch_summary_data": {
                "clustering_method": "hier",
                "threshold_ang": 0.3,
                "clusters": [{"cluster_index": 1, "representative_index": 0}],
            },
            "clustering_window_state": {"method": "kmeans"},
        },
    )

    assert dialog._batch_table.topLevelItemCount() == 1
    # The hierarchical t was no length, so the tolerance stays at its default.
    assert dialog._batch_grouping_tolerance.value() == pytest.approx(0.25)
    assert "clustering_window_state" not in dialog.state_snapshot()
    dialog.close()


def test_a_failed_measurement_keeps_the_finished_optimisations(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)

    def _fail(*_args, **_kwargs):
        raise ValueError("no symmetry today")

    monkeypatch.setattr("mufinder.app.uep_wizard.measure_uep_minima", _fail)

    dialog._run_batch_search()

    assert len(dialog._latest_batch_results) == 2
    assert dialog._batch_progress.format() == "Batch search complete"
    assert "Grouping failed: no symmetry today" in dialog._batch_summary.toPlainText()
    dialog.close()


def test_a_selection_follows_its_sites_when_the_groups_are_renumbered(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    first, second = _two_nearby_sites(host)
    far = replace(
        second,
        classical_energy_ev=second.classical_energy_ev + 0.01,
        optimized_position_ang=(2.0, 3.0, 3.5),
        optimized_fractional_position=(0.4, 0.5, 0.5),
    )
    dialog = _dialog_with_endpoints(monkeypatch, [first, second, far], host)
    dialog._run_batch_search()
    assert dialog._batch_table.topLevelItemCount() == 3
    (site_index,) = dialog._latest_batch_summary.clusters[1].member_indexes
    group_item = dialog._batch_table.topLevelItem(2)
    group_item.setSelected(True)
    site_item = dialog._batch_table.topLevelItem(1).child(0)
    site_item.setSelected(True)

    dialog._batch_grouping_tolerance.setText("0.5")

    assert dialog._batch_table.topLevelItemCount() == 2
    # The far site is Site 2 now, and the start joined Site 1.
    assert set(dialog._selected_batch_keys()) == {"cluster:2", f"cluster:1:site:{site_index}"}
    dialog.close()


def test_a_muon_in_a_viewer_follows_its_site_into_a_new_group(qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "mufinder.app.workflow_table_utils.confirm_destructive", lambda *_a, **_k: True
    )
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    second = next(
        candidate
        for candidate in dialog._clustered_muon_candidates()
        if candidate.cluster_index == 2
    )
    dialog._add_candidate_muons_to_viewer([second.key], False)
    assert dialog._viewer_muons[0].source_label == "Site 2"

    dialog._batch_grouping_tolerance.setText("0.5")

    assert dialog._viewer_muons[0].source_label == "Site 1"
    assert dialog._viewer_muons[0].cluster_index == 1
    dialog.close()


def _shown_dialog_with_two_sites(qapp, monkeypatch) -> UepAnalysisDialog:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    dialog.show()
    dialog.activateWindow()
    dialog._batch_grouping_tolerance.setFocus()
    dialog._batch_grouping_tolerance.selectAll()
    qapp.processEvents()
    return dialog


@pytest.mark.parametrize("key", [Qt.Key.Key_Tab, Qt.Key.Key_Return])
@pytest.mark.parametrize(
    ("typed", "clamped", "groups"),
    [("100", 10.0, 1), ("0.0001", 0.001, 2), ("0", 0.001, 2)],
    ids=["above", "below", "zero"],
)
def test_an_out_of_range_tolerance_is_clamped_and_grouped_at_on_leaving_the_field(
    qapp, monkeypatch, key: Qt.Key, typed: str, clamped: float, groups: int
) -> None:
    dialog = _shown_dialog_with_two_sites(qapp, monkeypatch)
    searches: list[bool] = []
    dialog._batch_run.clicked.connect(searches.append)
    emitted: list[float] = []
    dialog.state_changed.connect(
        lambda state: emitted.append(state["batch_summary_data"]["threshold_ang"])
    )

    QTest.keyClicks(dialog._batch_grouping_tolerance, typed)
    QTest.keyClick(dialog._batch_grouping_tolerance, key)
    qapp.processEvents()

    assert dialog._batch_grouping_tolerance.value() == pytest.approx(clamped)
    assert float(dialog._batch_grouping_tolerance.text()) == pytest.approx(clamped)
    assert dialog._latest_batch_summary.threshold_ang == pytest.approx(clamped)
    assert dialog._batch_table.topLevelItemCount() == groups
    assert f"within {clamped:g} Å" in dialog._batch_summary.toPlainText()
    assert emitted[-1] == pytest.approx(clamped)
    # Return in the field does not press a button and run another search.
    assert searches == []
    dialog.close()


def test_typing_the_minimum_tolerance_regroups_without_leaving_the_field(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    first, second = _two_nearby_sites(host)
    # 0.0005 Å apart: one site at any tolerance but the smallest.
    close = replace(
        second,
        optimized_position_ang=(
            first.optimized_position_ang[0] + 0.0005,
            *first.optimized_position_ang[1:],
        ),
    )
    dialog = _dialog_with_endpoints(monkeypatch, [first, close], host)
    dialog._batch_grouping_tolerance.setValue(0.0005 * 2 + 0.0001)
    dialog._run_batch_search()
    dialog.show()
    dialog.activateWindow()
    dialog._batch_grouping_tolerance.setFocus()
    dialog._batch_grouping_tolerance.selectAll()
    qapp.processEvents()

    QTest.keyClicks(dialog._batch_grouping_tolerance, "0.001")

    assert dialog._latest_batch_summary.threshold_ang == pytest.approx(0.001)
    dialog.close()


def test_a_tolerance_edit_keeps_expanded_groups_expanded(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    first, second = _two_nearby_sites(host)
    far = replace(
        second,
        classical_energy_ev=second.classical_energy_ev + 0.01,
        optimized_position_ang=(2.0, 3.0, 3.5),
        optimized_fractional_position=(0.4, 0.5, 0.5),
    )
    dialog = _dialog_with_endpoints(monkeypatch, [first, second, far], host)
    dialog._run_batch_search()
    dialog._batch_table.topLevelItem(0).setExpanded(True)
    dialog._batch_table.topLevelItem(2).setExpanded(True)

    # The same groups.
    dialog._batch_grouping_tolerance.setText("0.28")

    assert [dialog._batch_table.topLevelItem(row).isExpanded() for row in range(3)] == [
        True,
        False,
        True,
    ]

    # Sites 1 and 2 merge: the far site is Site 2 now, and stays open.
    dialog._batch_grouping_tolerance.setText("0.5")

    assert [dialog._batch_table.topLevelItem(row).isExpanded() for row in range(2)] == [True, True]
    dialog.close()


def test_a_search_where_every_start_fails_says_no_minima_were_found(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    endpoints = [replace(entry, success=False) for entry in _two_nearby_sites(host)]
    dialog = _dialog_with_endpoints(monkeypatch, endpoints, host)

    dialog._run_batch_search()

    assert dialog._batch_table.topLevelItemCount() == 0
    assert dialog._batch_progress.format() == "No minima found"
    assert "Grouping: no minima found. Left out 2 failed optimisations." in (
        dialog._batch_summary.toPlainText()
    )
    snapshot = dialog.state_snapshot()
    dialog.close()

    restored = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=host,
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        initial_state=snapshot,
    )
    assert restored._batch_progress.format() == "No minima found"
    restored.close()


def test_flushing_writes_a_pending_regroup_and_nothing_otherwise(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    emitted: list[float] = []
    dialog.state_changed.connect(
        lambda state: emitted.append(state["batch_summary_data"]["threshold_ang"])
    )

    dialog.flush_pending_state()
    assert emitted == []

    dialog._batch_grouping_tolerance.setText("0.5")
    dialog.flush_pending_state()

    assert emitted == [pytest.approx(0.5)]
    assert not dialog._grouping_settled_timer.isActive()
    dialog.close()


def _supercell_job_dialog(monkeypatch) -> tuple[UepAnalysisDialog, list, Structure]:
    """A source job on a cell twice as long as its input structure along the first vector."""

    job_lattice = np.asarray(
        load_uep_source(
            den_fmt_path=_fixture_path("Si2.den_fmt"),
            castep_path=_fixture_path("Si2.castep"),
            cell_path=_fixture_path("Si2.cell"),
        ).structure.lattice_vectors,
        dtype=float,
    )
    host_lattice = job_lattice.copy()
    host_lattice[0] /= 2.0
    host = Structure(
        id=uuid4(),
        label="Input cell",
        lattice_vectors=host_lattice.tolist(),
        species=["Si", "Ge", "C"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.31, 0.47, 0.13], [0.71, 0.11, 0.59]],
    )
    (template, _other) = _two_nearby_sites(_no_symmetry_host())
    endpoints = []
    for job_fractional in ((0.6, 0.2, 0.3), (0.1, 0.7, 0.4)):
        cartesian = tuple((np.asarray(job_fractional) @ job_lattice).tolist())
        endpoints.append(
            replace(
                template,
                optimized_position_ang=cartesian,
                optimized_fractional_position=job_fractional,
                classical_energy_ev=template.classical_energy_ev + job_fractional[1],
            )
        )
    return _dialog_with_endpoints(monkeypatch, endpoints, host), endpoints, host


def test_every_row_of_a_supercell_job_reads_in_the_input_cell(qapp, monkeypatch) -> None:
    dialog, _endpoints, _host = _supercell_job_dialog(monkeypatch)

    dialog._run_batch_search()

    first = dialog._batch_table.topLevelItem(0)
    second = dialog._batch_table.topLevelItem(1)
    # (0.6, 0.2, 0.3) in the job is (1.2 → 0.2, 0.2, 0.3) in the input cell.
    assert [float(first.child(0).text(column)) for column in (2, 3, 4)] == pytest.approx(
        (0.2, 0.2, 0.3), abs=1e-6
    )
    assert [float(second.child(0).text(column)) for column in (2, 3, 4)] == pytest.approx(
        (0.2, 0.7, 0.4), abs=1e-6
    )
    dialog.close()


def test_the_muons_of_a_supercell_job_stay_where_the_job_cell_has_them(qapp, monkeypatch) -> None:
    dialog, endpoints, _host = _supercell_job_dialog(monkeypatch)
    dialog._run_batch_search()
    dialog.show()

    # Selecting a start overlays that start's muon, at its position in the job.
    start = dialog._batch_table.topLevelItem(1).child(0)
    start.setSelected(True)
    dialog._batch_table.setCurrentItem(start)
    qapp.processEvents()
    (overlay,) = dialog._batch_viewer._point_overlay_specs
    assert np.asarray(overlay.positions).reshape(-1, 3)[0] == pytest.approx(
        endpoints[1].optimized_position_ang
    )

    # A muon in a viewer sits at its job position; its dialogs read in the input cell.
    candidate = next(
        entry for entry in dialog._clustered_muon_candidates() if entry.result_index == 0
    )
    assert candidate.fractional_position == pytest.approx((0.6, 0.2, 0.3))
    assert candidate.display_fractional_position == pytest.approx((0.2, 0.2, 0.3), abs=1e-6)
    dialog._add_candidate_muons_to_viewer([candidate.key], False)
    assert dialog._viewer_structure.fractional_positions[-1] == pytest.approx([0.6, 0.2, 0.3])
    assert dialog._viewer_muons[0].display_fractional_position == pytest.approx(
        (0.2, 0.2, 0.3), abs=1e-6
    )
    dialog.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Tab, Qt.Key.Key_Return])
def test_a_blank_tolerance_restores_the_value_in_use_on_leaving_the_field(
    qapp, monkeypatch, key: Qt.Key
) -> None:
    dialog = _shown_dialog_with_two_sites(qapp, monkeypatch)
    QTest.keyClicks(dialog._batch_grouping_tolerance, "0.3")
    summary = dialog._latest_batch_summary
    dialog._batch_grouping_tolerance.selectAll()
    QTest.keyClick(dialog._batch_grouping_tolerance, Qt.Key.Key_Delete)
    assert dialog._batch_grouping_tolerance.text() == ""

    QTest.keyClick(dialog._batch_grouping_tolerance, key)
    qapp.processEvents()

    assert dialog._batch_grouping_tolerance.text() == "0.3"
    assert dialog._latest_batch_summary is summary
    dialog.close()


def test_a_saved_viewer_muon_with_a_default_label_reads_in_the_new_words(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    candidate = dialog._clustered_muon_candidates()[0]
    dialog._add_candidate_muons_to_viewer([candidate.key], False)
    entry = dialog._viewer_muons[0]
    entry.label, entry.source_label = "Site 1", "Cluster 1"

    dialog._relabel_viewer_muons()

    assert (entry.label, entry.source_label) == ("Start 1", "Site 1")
    entry.label = "My muon"
    dialog._relabel_viewer_muons()
    assert entry.label == "My muon"
    dialog.close()


def test_a_gaussian_width_of_zero_then_return_keeps_the_batch_results(qapp, monkeypatch) -> None:
    dialog = _shown_dialog_with_two_sites(qapp, monkeypatch)
    results = list(dialog._latest_batch_results)
    width = dialog._source.gw_factor
    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    field = dialog._gw_factor_input
    field.setFocus()
    field.selectAll()
    qapp.processEvents()

    QTest.keyClicks(field, "0")
    QTest.keyClick(field, Qt.Key.Key_Return)
    qapp.processEvents()

    assert dialog._source.gw_factor == width
    assert dialog._latest_batch_results == results
    assert dialog._batch_table.topLevelItemCount() == 2
    dialog.close()


def test_the_muon_dialog_says_which_cell_its_positions_are_in(qapp) -> None:
    dialog = UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
    )
    dialog._open_viewer_muon_dialog()
    muon_dialog = dialog._viewer_muon_dialog

    assert muon_dialog._cell_note.text() == (
        "Start rows are in the input structure's cell, Custom rows in the job's cell."
    )
    muon_dialog.close()
    dialog.close()


def _si2_dialog(**overrides: object) -> UepAnalysisDialog:
    return UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        source_structure=_source_structure_with_spins(),
        den_fmt_path=_fixture_path("Si2.den_fmt"),
        castep_path=_fixture_path("Si2.castep"),
        cell_path=_fixture_path("Si2.cell"),
        **overrides,
    )


def _uep_si8_dialog() -> UepAnalysisDialog:
    return UepAnalysisDialog(
        source_job_label="uep_source_row1",
        source_structure_id="11111111-1111-1111-1111-111111111111",
        den_fmt_path=_fixture_path("Si8.den_fmt"),
        castep_path=_fixture_path("Si8.castep"),
        cell_path=_fixture_path("Si8.cell"),
    )


def _answer_confirmations(monkeypatch, *, answer: bool) -> list[str]:
    asked: list[str] = []

    def _confirm(_parent, title, *_args, **_kwargs) -> bool:
        asked.append(title)
        return answer

    monkeypatch.setattr("mufinder.app.workflow_table_utils.confirm_destructive", _confirm)
    return asked


def test_clicking_run_fills_the_table_with_zero_point_energies_and_nothing_else_to_click(
    qapp, monkeypatch
) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = _si2_dialog()

    dialog._batch_run.click()

    assert dialog._batch_table.topLevelItemCount() >= 1
    for row in range(dialog._batch_table.topLevelItemCount()):
        cluster = dialog._latest_batch_summary.clusters[row]
        representative = dialog._latest_batch_results[cluster.representative_index]
        item = dialog._batch_table.topLevelItem(row)
        assert representative.zero_point_energy_ev > 0.0
        assert representative.total_energy_ev == pytest.approx(
            representative.classical_energy_ev + representative.zero_point_energy_ev
        )
        assert item.text(2) == f"{cluster.representative_fractional_position[0]:.6f}"
        assert item.text(5) == f"{representative.classical_energy_ev:.6f}"
        assert item.text(6) == f"{representative.zero_point_energy_ev:.6f}"
        assert item.text(7) == f"{representative.total_energy_ev:.6f}"
        assert item.childCount() == cluster.size
    assert not any(
        button.text() in {"Calculate ZPE", "Clustering"}
        for button in dialog.findChildren(QPushButton)
    )
    dialog.close()


def test_the_search_log_counts_the_starts_and_how_they_ended(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    first, second = _two_nearby_sites(host)
    saddle = replace(second, classification=replace(second.classification, label="saddle"))
    failed = replace(first, success=False)
    dialog = _dialog_with_endpoints(monkeypatch, [first, saddle, failed], host)

    dialog._batch_run.click()

    log = dialog._batch_summary.toPlainText()
    assert "Requested random starts: 3" in log
    assert "Initial starts generated: 3" in log
    assert "Converged: 1 minimum, 1 saddle point. Failed to converge: 1." in log
    assert "Grouping: 1 minimum in 1 site" in log
    dialog.close()


def test_a_new_search_asks_first_and_cancel_keeps_the_results(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    asked = _answer_confirmations(monkeypatch, answer=False)
    dialog = _si2_dialog()

    dialog._batch_run.click()
    assert asked == []
    results = list(dialog._latest_batch_results)
    rows = dialog._batch_table.topLevelItemCount()

    dialog._batch_run.click()

    assert asked == ["Run a new search"]
    assert dialog._latest_batch_results == results
    assert dialog._batch_table.topLevelItemCount() == rows
    assert dialog._batch_progress.format() == "Batch search complete"
    dialog.close()


def test_a_new_search_replaces_the_results_when_confirmed(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    first, second = _two_nearby_sites(host)
    remaining = iter([first, second, first])
    monkeypatch.setattr(
        "mufinder.app.uep_wizard.optimize_muon_position", lambda *_a, **_k: next(remaining)
    )
    asked = _answer_confirmations(monkeypatch, answer=True)
    _run_batch_search_inline(monkeypatch)
    dialog = _si2_dialog()
    dialog._batch_count.setValue(2)
    dialog._batch_min_muon_muon.setValue(0.1)
    dialog._batch_run.click()
    dialog._batch_count.setValue(1)

    dialog._batch_run.click()

    assert asked == ["Run a new search"]
    assert len(dialog._latest_batch_results) == 1
    dialog.close()


def test_a_failed_new_search_leaves_no_results_behind_it(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    _answer_confirmations(monkeypatch, answer=True)
    monkeypatch.setattr("mufinder.app.uep_wizard.QMessageBox.warning", lambda *_a, **_k: None)
    dialog = _si2_dialog()
    dialog._batch_run.click()

    emitted: list[dict] = []
    dialog.state_changed.connect(emitted.append)

    def _fail(**_kwargs):
        raise ValueError("no room")

    monkeypatch.setattr("mufinder.app.uep_wizard.generate_random_sites", _fail)
    dialog._batch_run.click()

    assert dialog._latest_batch_results == []
    assert dialog._batch_table.topLevelItemCount() == 0
    assert dialog._batch_progress.format() == "Batch search failed"
    assert dialog._batch_summary.toPlainText() == "Batch search failed: no room"
    assert dialog._batch_run.isEnabled()
    # The project hears that the results went, so Generation stops offering them.
    assert emitted[-1]["batch_results"] == []
    dialog.close()


@pytest.mark.parametrize("input_name", ["_gw_factor_input", "_gw_factor_input"])
def test_a_width_change_asks_first_and_cancel_keeps_the_results_and_the_field(
    qapp, monkeypatch, input_name: str
) -> None:
    _run_batch_search_inline(monkeypatch)
    asked = _answer_confirmations(monkeypatch, answer=False)
    dialog = _si2_dialog()
    monkeypatch.setattr(dialog, "_start_isosurface_grid_calculation", lambda: None)
    dialog._batch_run.click()
    results = list(dialog._latest_batch_results)
    field = getattr(dialog, input_name)

    field.setValue(7.0)
    field.returnPressed.emit()

    assert asked == ["Change the Gaussian width factor"]
    assert dialog._source.gw_factor == pytest.approx(5.0)
    assert dialog._gw_factor_input.value() == pytest.approx(5.0)
    assert dialog._latest_batch_results == results
    assert dialog._batch_table.topLevelItemCount() >= 1
    dialog.close()


def test_a_confirmed_width_change_discards_the_results(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    asked = _answer_confirmations(monkeypatch, answer=True)
    dialog = _si2_dialog()
    monkeypatch.setattr(dialog, "_start_isosurface_grid_calculation", lambda: None)
    dialog._batch_run.click()

    dialog._gw_factor_input.setValue(7.0)
    dialog._gw_factor_input.returnPressed.emit()

    assert asked == ["Change the Gaussian width factor"]
    assert dialog._source.gw_factor == pytest.approx(7.0)
    assert dialog._latest_batch_results == []
    assert dialog._batch_table.topLevelItemCount() == 0
    assert dialog._batch_progress.format() == "Ready"
    dialog.close()


def test_a_width_change_with_no_results_or_no_change_asks_nothing(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    asked = _answer_confirmations(monkeypatch, answer=False)
    dialog = _si2_dialog()
    monkeypatch.setattr(dialog, "_start_isosurface_grid_calculation", lambda: None)

    dialog._gw_factor_input.setValue(7.0)
    dialog._gw_factor_input.returnPressed.emit()
    assert dialog._source.gw_factor == pytest.approx(7.0)

    dialog._batch_run.click()
    dialog._gw_factor_input.returnPressed.emit()

    assert asked == []
    assert dialog._latest_batch_results
    dialog.close()


def test_the_tolerance_says_it_is_a_force_and_the_advanced_settings_start_hidden(qapp) -> None:
    dialog = _si2_dialog()
    form = dialog._batch_form
    label_of = lambda control: form.labelForField(control).text()  # noqa: E731

    assert label_of(dialog._batch_force_tolerance) == "Force tolerance (eV/Å)"
    assert label_of(dialog._batch_max_attempts) == "Max attempts"
    assert label_of(dialog._batch_max_steps) == "Optimizer max steps"
    assert "Force threshold" in dialog._batch_force_tolerance.toolTip()
    assert "pymuon" not in dialog._batch_force_tolerance.toolTip()
    assert dialog._batch_force_tolerance.value() == pytest.approx(
        DEFAULT_FORCE_TOLERANCE_EV_PER_ANG
    )
    assert dialog._batch_max_steps.value() == DEFAULT_OPTIMIZER_MAX_STEPS

    advanced = (dialog._batch_max_attempts, dialog._batch_force_tolerance, dialog._batch_max_steps)
    assert all(control.isHidden() for control in advanced)
    assert not dialog._batch_count.isHidden()
    dialog._batch_advanced_toggle.click()
    assert not any(control.isHidden() for control in advanced)
    assert dialog._batch_advanced_toggle.arrowType() == Qt.ArrowType.DownArrow
    dialog._batch_advanced_toggle.click()
    assert all(control.isHidden() for control in advanced)
    dialog.close()


def test_a_search_saved_without_zero_point_energies_says_how_to_get_them(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    first = _si2_dialog()
    first._batch_run.click()
    snapshot = first.state_snapshot()
    first.close()
    for entry in snapshot["batch_results"]:
        entry["zero_point_energy_ev"] = None
        entry["total_energy_ev"] = None
    snapshot["batch_zpe_completed"] = False

    restored = _si2_dialog(initial_state=snapshot)

    item = restored._batch_table.topLevelItem(0)
    assert item.text(5) != ""
    assert item.text(6) == ""
    assert item.text(7) == ""
    assert "saved without zero-point energies" in restored._batch_summary.toPlainText()
    saved_again = restored.state_snapshot()
    restored.close()

    reopened = _si2_dialog(initial_state=saved_again)
    assert reopened._batch_summary.toPlainText().count("saved without zero-point energies") == 1
    reopened.close()


def test_the_controls_and_the_log_fit_the_minimum_window_with_advanced_open(
    qapp, monkeypatch
) -> None:
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._batch_run.click()
    dialog._batch_advanced_toggle.click()
    dialog.resize(dialog.minimumSize())
    dialog.show()
    qapp.processEvents()

    card = dialog._batch_controls_card
    assert dialog._batch_run.geometry().bottom() < card.height()
    assert dialog._batch_progress.geometry().bottom() < card.height()
    assert dialog._batch_summary.height() >= dialog._batch_summary.minimumHeight()
    assert dialog._batch_table.isVisible()
    dialog.close()


def test_a_window_saved_mid_search_reopens_with_an_empty_log(qapp) -> None:
    restored = _si2_dialog(
        initial_state={"batch_results": [], "batch_summary_text": "Optimizing starts…"}
    )

    assert restored._batch_summary.toPlainText() == ""
    assert restored._batch_progress.format() == "Ready"
    restored.close()


def test_advanced_opens_on_a_saved_value_that_is_not_the_default_and_the_log_records_it(
    qapp, monkeypatch
) -> None:
    _run_batch_search_inline(monkeypatch)
    old = _si2_dialog(initial_state={"batch_max_steps": 100})

    assert old._batch_advanced_toggle.isChecked()
    assert old._batch_max_steps.value() == 100
    assert not old._batch_max_steps.isHidden()
    old._batch_run.click()
    assert "Optimizer max steps: 100" in old._batch_summary.toPlainText()
    old.close()

    plain = _si2_dialog(initial_state={"batch_max_steps": DEFAULT_OPTIMIZER_MAX_STEPS})
    assert not plain._batch_advanced_toggle.isChecked()
    plain.close()


def test_the_force_tolerance_field_shows_its_smallest_value(qapp) -> None:
    dialog = _si2_dialog()

    dialog._batch_force_tolerance.setValue(5e-7)

    assert dialog._batch_force_tolerance.hasAcceptableInput()
    assert dialog._batch_force_tolerance.value() == pytest.approx(5e-7)
    dialog.close()


def test_the_log_alone_reproduces_a_search(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = _si2_dialog()
    dialog._batch_min_muon_atom.setValue(1.3)
    dialog._batch_max_attempts.setValue(4000)

    dialog._batch_run.click()

    log = dialog._batch_summary.toPlainText()
    assert "Min muon-muon distance (Å): 0.800" in log
    assert "Min muon-atom distance (Å): 1.300" in log
    assert "Max attempts: 4000" in log
    assert "Force tolerance (eV/Å): 1e-05" in log
    assert "Optimizer max steps: 300" in log
    dialog.close()


def test_the_force_tolerance_tooltip_states_its_unit_on_the_label_and_the_field(qapp) -> None:
    dialog = _si2_dialog()
    field = dialog._batch_force_tolerance

    assert "eV/Å" in field.toolTip()
    assert dialog._batch_form.labelForField(field).toolTip() == field.toolTip()
    dialog.close()


@pytest.mark.parametrize("checked", [False, True])
def test_the_advanced_toggle_draws_nothing_behind_its_text(qapp, checked: bool) -> None:
    previous_sheet = qapp.styleSheet()
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    dialog = _si2_dialog()
    try:
        dialog.show()
        qapp.processEvents()
        toggle = dialog._batch_advanced_toggle
        toggle.setChecked(checked)
        qapp.processEvents()
        image = dialog._batch_controls_card.grab().toImage()
        ratio = image.width() / dialog._batch_controls_card.width()
        corner = toggle.mapTo(dialog._batch_controls_card, toggle.rect().topLeft())
        background = image.pixelColor(int((corner.x() - 2) * ratio), int(corner.y() * ratio))

        for point in (toggle.rect().topLeft(), toggle.rect().bottomRight()):
            inside = toggle.mapTo(dialog._batch_controls_card, point)
            assert image.pixelColor(int(inside.x() * ratio), int(inside.y() * ratio)) == background
    finally:
        dialog.close()
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
        qapp.setStyleSheet(previous_sheet)


def test_a_search_with_no_minima_does_not_read_as_a_success(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    endpoints = [replace(entry, success=False) for entry in _two_nearby_sites(host)]
    dialog = _dialog_with_endpoints(monkeypatch, endpoints, host)

    dialog._batch_run.click()

    assert dialog._batch_progress.format() == "No minima found"
    assert dialog._batch_progress.property("generationState") == "warning"
    snapshot = dialog.state_snapshot()
    dialog.close()

    restored = _si2_dialog(initial_state=snapshot)
    assert restored._batch_progress.property("generationState") == "warning"
    restored.close()


def test_a_search_over_no_sites_replaces_them_without_asking(qapp, monkeypatch) -> None:
    host = _no_symmetry_host()
    failed = [replace(entry, success=False) for entry in _two_nearby_sites(host)]
    dialog = _dialog_with_endpoints(monkeypatch, [*failed, *failed], host)
    asked = _answer_confirmations(monkeypatch, answer=False)
    dialog._batch_count.setValue(2)
    dialog._batch_run.click()
    assert dialog._batch_table.topLevelItemCount() == 0

    dialog._batch_run.click()

    assert asked == []
    dialog.close()


def test_the_discard_question_counts_what_exists_and_names_muons_only_when_there_are_some(
    qapp, monkeypatch
) -> None:
    _run_batch_search_inline(monkeypatch)
    details: list[str] = []

    def _confirm(_parent, _title, _text, detail, *_args, **_kwargs) -> bool:
        details.append(detail)
        return False

    monkeypatch.setattr("mufinder.app.workflow_table_utils.confirm_destructive", _confirm)
    dialog = _si2_dialog()
    dialog._batch_run.click()
    sites = len(dialog._latest_batch_summary.clusters)
    starts = len(dialog._latest_batch_results)

    dialog._batch_run.click()

    assert details[-1] == f"This discards {plural(sites, 'site')} from {plural(starts, 'start')}."

    candidate = dialog._clustered_muon_candidates()[0]
    dialog._add_candidate_muons_to_viewer([candidate.key], False)
    dialog._batch_run.click()

    assert details[-1] == (
        f"This discards {plural(sites, 'site')} from {plural(starts, 'start')} "
        "and removes 1 muon from the viewer."
    )
    dialog.close()


def test_the_log_shows_how_a_search_ended_when_it_is_too_long_to_fit(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = _si2_dialog()
    dialog.resize(dialog.minimumSize())
    dialog.show()
    qapp.processEvents()

    dialog._batch_run.click()
    qapp.processEvents()

    log = dialog._batch_summary
    assert log.verticalScrollBar().maximum() > 0
    assert log.textCursor().atEnd()
    assert 0 <= log.cursorRect().top() and log.cursorRect().bottom() <= log.viewport().height()
    dialog.close()


def test_the_log_records_the_gaussian_width_the_search_ran_with(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    dialog = _si2_dialog(initial_state={"gw_factor": 3.0})

    dialog._batch_run.click()

    assert "Gaussian width factor: 3" in dialog._batch_summary.toPlainText().splitlines()
    dialog.close()

    dialog = _si2_dialog()
    dialog._batch_run.click()

    assert "Gaussian width factor: 5" in dialog._batch_summary.toPlainText().splitlines()
    dialog.close()


def test_opening_a_search_an_older_build_saved_changes_nothing_once_it_has_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run_batch_search_inline(monkeypatch)
    first = _si2_dialog()
    first._batch_run.click()
    older = first.state_snapshot()
    first.close()
    for entry in older["batch_results"]:
        entry["zero_point_energy_ev"] = None
        entry["total_energy_ev"] = None
    older["batch_zpe_completed"] = False

    document = ProjectService().create_project(tmp_path / "older.mufinder", "Older")
    document.ui_state[UEP_ANALYSIS_STATES_KEY] = {"job": to_jsonable(older)}
    regroup_saved_uep_searches(document)
    loaded = document.ui_state[UEP_ANALYSIS_STATES_KEY]["job"]

    restored = _si2_dialog(initial_state=loaded)
    snapshot = to_jsonable(restored.state_snapshot())
    restored.close()

    assert "batch_zpe_completed" not in loaded
    assert snapshot == loaded


def test_the_grid_is_computed_the_first_time_the_potential_tab_is_shown(qapp, monkeypatch) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = _si2_dialog()
    dialog.show()
    qapp.processEvents()

    assert dialog._tabs.currentWidget() is dialog._batch_tab
    assert dialog._potential_grid_cache is None

    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    qapp.processEvents()

    assert dialog._potential_grid_cache is not None
    assert dialog._active_isosurface_layer().mesh is not None
    assert dialog._isosurface_slider.isEnabled() is True
    assert dialog._isosurface_export_button.isEnabled() is True
    assert not dialog._potential_status.isVisibleTo(dialog)
    dialog.close()


def test_a_window_saved_on_the_potential_tab_computes_its_grid_when_shown(
    qapp, monkeypatch
) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = _si2_dialog(initial_state={"current_tab": 1})

    assert dialog._potential_grid_cache is None

    dialog.show()
    qapp.processEvents()

    assert dialog._potential_grid_cache is not None
    dialog.close()


def test_the_tab_says_so_when_the_grid_cannot_be_computed_and_tries_again_on_return(
    qapp, monkeypatch
) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = _si2_dialog()

    def _fail(_field):
        raise ValueError("no memory left")

    monkeypatch.setattr("mufinder.app.uep_wizard.potential_grid", _fail)
    dialog.show()
    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    qapp.processEvents()

    assert "no memory left" in dialog._potential_status.text()
    assert dialog._potential_status.property("statusKind") == "error"
    assert dialog._isosurface_slider.isEnabled() is False

    monkeypatch.undo()
    _run_isosurface_threads_inline(monkeypatch)
    dialog._tabs.setCurrentWidget(dialog._batch_tab)
    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    qapp.processEvents()

    assert dialog._potential_grid_cache is not None
    assert not dialog._potential_status.text()
    dialog.close()


def test_one_width_factor_drives_the_search_and_the_potential(qapp, monkeypatch) -> None:
    _run_batch_search_inline(monkeypatch)
    _run_isosurface_threads_inline(monkeypatch)
    dialog = _si2_dialog()
    dialog.show()

    dialog._gw_factor_input.setValue(4.0)
    dialog._gw_factor_input.returnPressed.emit()
    dialog._batch_count.setValue(1)
    dialog._batch_run.click()
    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    qapp.processEvents()

    assert "Gaussian width factor: 4" in dialog._batch_summary.toPlainText()
    assert dialog._potential_grid_raw_cache.gw_factor == pytest.approx(4.0)
    dialog.close()


def test_a_saved_width_factor_outside_the_range_stays_in_use_and_says_so(qapp) -> None:
    # The search a project holds ran on this factor: the window must not change it unasked.
    dialog = _si2_dialog(initial_state={"gw_factor": 20.0})

    assert dialog._source.gw_factor == pytest.approx(20.0)
    assert dialog._gw_factor_input.value() == pytest.approx(20.0)
    assert dialog.state_snapshot()["gw_factor"] == pytest.approx(20.0)
    assert "saved with a width factor of 20" in dialog._gw_factor_warning.text()

    dialog._on_gw_factor_editing_finished()
    assert dialog._source.gw_factor == pytest.approx(20.0)
    dialog.close()


@pytest.mark.parametrize("typed", ["0.5", "20"])
def test_a_width_factor_outside_the_range_is_not_applied_and_says_why(qapp, typed: str) -> None:
    dialog = _si2_dialog()
    dialog.show()
    field = dialog._gw_factor_input
    field.setFocus()
    field.selectAll()
    qapp.processEvents()

    QTest.keyClicks(field, typed)
    QTest.keyClick(field, Qt.Key.Key_Return)
    qapp.processEvents()

    assert dialog._source.gw_factor == pytest.approx(5.0)
    assert "Enter a width factor from 1 to 10." in dialog._gw_factor_warning.text()
    dialog.close()


def test_the_width_factor_warns_when_the_grid_does_not_resolve_the_gaussians(qapp) -> None:
    dialog = _si2_dialog()
    assert not dialog._gw_factor_warning.isVisibleTo(dialog)

    dialog._gw_factor_input.setValue(10.0)
    dialog._gw_factor_input.returnPressed.emit()

    assert dialog._source.gw_factor == pytest.approx(10.0)
    assert "does not resolve the ionic Gaussians" in dialog._gw_factor_warning.text()
    assert dialog._gw_factor_warning.isVisibleTo(dialog)

    dialog._gw_factor_input.setValue(5.0)
    dialog._gw_factor_input.returnPressed.emit()
    assert not dialog._gw_factor_warning.isVisibleTo(dialog)
    dialog.close()


def test_saved_levels_on_the_potentials_own_scale_keep_their_surface_above_the_minimum(
    qapp,
) -> None:
    baseline = _si2_dialog()
    grid = potential_grid(baseline._field)
    baseline.close()
    minimum = float(grid.potential_total_ev.min())
    state = {
        "isosurface_zero_minimum": False,
        "isosurface_layers": [
            {"label": "Isosurface 1", "key": "a", "level_ev": minimum + 3.0, "color": "#123456"}
        ],
        "slice_saturation_bounds": [minimum + 1.0, minimum + 2.0],
    }

    dialog = _si2_dialog(initial_state=state)

    # Before the grid there is no minimum to measure from, so the state keeps its own scale.
    snapshot = dialog.state_snapshot()
    assert snapshot["isosurface_zero_minimum"] is False
    assert snapshot["isosurface_layers"][0]["level_ev"] == pytest.approx(minimum + 3.0)

    dialog._apply_potential_grid_cache(grid)

    assert dialog._isosurface_layers[0].level_ev == pytest.approx(3.0)
    assert dialog._slice_saturation_bounds == pytest.approx((1.0, 2.0))
    snapshot = dialog.state_snapshot()
    assert snapshot["isosurface_zero_minimum"] is True
    assert snapshot["isosurface_layers"][0]["level_ev"] == pytest.approx(3.0)
    dialog.close()


def test_saved_levels_above_the_minimum_are_read_as_they_are(qapp) -> None:
    state = {
        "isosurface_zero_minimum": True,
        "isosurface_layers": [{"label": "Isosurface 1", "key": "a", "level_ev": 3.0}],
        "slice_saturation_bounds": [1.0, 2.0],
    }

    dialog = _si2_dialog(initial_state=state)
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))

    assert dialog._isosurface_layers[0].level_ev == pytest.approx(3.0)
    assert dialog._slice_saturation_bounds == pytest.approx((1.0, 2.0))
    dialog.close()


def test_a_saved_whole_range_saturation_becomes_the_clipped_default(qapp) -> None:
    baseline = _si2_dialog()
    grid = potential_grid(baseline._field)
    baseline.close()
    values = grid.potential_total_ev

    dialog = _si2_dialog(
        initial_state={"slice_saturation_bounds": [float(values.min()), float(values.max())]}
    )
    dialog._apply_potential_grid_cache(grid)

    assert dialog._slice_saturation_bounds == pytest.approx(
        (0.0, float(np.percentile(values - values.min(), 95)))
    )
    dialog.close()


def test_a_single_mesh_saved_at_the_top_level_gives_its_level(qapp) -> None:
    dialog = _si2_dialog(
        initial_state={"isosurface_zero_minimum": True, "isosurface_mesh": {"level_ev": 3.0}}
    )

    assert dialog._isosurface_layers[0].level_ev == pytest.approx(3.0)
    assert "isosurface_mesh" not in dialog.state_snapshot()
    dialog.close()


def test_a_window_saved_with_two_viewers_shows_both_lists_of_muons_once(qapp) -> None:
    def _muon(key: str, position: tuple[float, float, float], **extra) -> dict:
        return {
            "key": key,
            "label": key,
            "source_kind": "custom",
            "fractional_position": list(position),
            "cartesian_position_ang": [0.0, 0.0, 0.0],
            **extra,
        }

    state = {
        "isosurface_viewer_muons": [
            _muon("both", (0.1, 0.2, 0.3)),
            _muon("site", (0.4, 0.4, 0.4), source_kind="clustered", result_index=2),
        ],
        "slice_viewer_muons": [
            _muon("both-again", (0.1, 0.2, 0.3)),
            _muon("site-again", (0.4, 0.4, 0.4), source_kind="clustered", result_index=2),
            _muon("only-slice", (0.7, 0.2, 0.3)),
        ],
        "isosurface_selected_muon_keys": ["both"],
        "slice_selected_muon_keys": ["only-slice", "both-again"],
    }

    dialog = _si2_dialog(initial_state=state)

    assert [entry.key for entry in dialog._viewer_muons] == ["both", "site", "only-slice"]
    assert dialog._selected_viewer_muon_keys == ["both", "only-slice"]
    assert dialog._potential_viewer._structure.species.count("X") == 3
    snapshot = dialog.state_snapshot()
    assert [entry["key"] for entry in snapshot["viewer_muons"]] == ["both", "site", "only-slice"]
    assert "isosurface_viewer_muons" not in snapshot
    assert "slice_viewer_muons" not in snapshot
    dialog.close()


def test_two_custom_muons_at_one_position_both_survive_a_reopen(qapp) -> None:
    first = _si2_dialog()
    first._add_custom_muon_to_viewer((0.1, 0.2, 0.3))
    first._add_custom_muon_to_viewer((0.1, 0.2, 0.3))
    snapshot = first.state_snapshot()
    first.close()

    restored = _si2_dialog(initial_state=snapshot)

    assert [entry.label for entry in restored._viewer_muons] == ["Custom 1", "Custom 2"]
    restored.close()


def test_two_custom_muons_at_one_position_in_one_old_list_both_survive(qapp) -> None:
    first = _si2_dialog()
    first._add_custom_muon_to_viewer((0.1, 0.2, 0.3))
    first._add_custom_muon_to_viewer((0.1, 0.2, 0.3))
    saved = first.state_snapshot()["viewer_muons"]
    first.close()

    restored = _si2_dialog(initial_state={"isosurface_viewer_muons": saved})

    assert len(restored._viewer_muons) == 2
    restored.close()


def test_the_saved_state_holds_what_the_grid_cannot_give_back(qapp) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    _show_level(dialog, 5.0)
    dialog._slice_planes = [
        _SlicePlaneState(label="Plane 1", plane=_slice_plane_for_dialog(dialog))
    ]
    dialog._resample_all_slice_planes()

    snapshot = dialog.state_snapshot()

    assert set(snapshot["isosurface_layers"][0]) == {
        "label",
        "key",
        "level_ev",
        "color",
        "managed_muon_key",
    }
    assert set(snapshot["slice_planes"][0]) == {"label", "plane"}
    assert "potential_grid" not in snapshot
    assert not hasattr(dialog, "potential_grid_ready")
    dialog.close()


def test_the_settled_surface_comes_from_the_whole_grid(qapp, monkeypatch) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = _uep_si8_dialog()
    dialog.show()
    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    qapp.processEvents()

    level = dialog._active_isosurface_layer().level_ev
    whole_grid = extract_isosurface(dialog._potential_grid_cache, level_ev=level)
    assert len(dialog._active_isosurface_layer().mesh.faces) == len(whole_grid.faces)
    live = dialog._potential_grid_live_preview_cache
    assert live.grid[0] < dialog._potential_grid_cache.grid[0]
    dialog.close()


def test_xsf_export_writes_the_potential_on_its_own_scale(qapp, monkeypatch, tmp_path) -> None:
    dialog = _si2_dialog()
    grid = potential_grid(dialog._field)
    dialog._apply_potential_grid_cache(grid)
    exported = tmp_path / "from_window.xsf"
    monkeypatch.setattr(
        "mufinder.app.uep_wizard.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(exported), "XSF Files (*.xsf)"),
    )

    dialog._isosurface_export_button.click()

    reference = tmp_path / "reference.xsf"
    write_potential_grid_xsf(reference, structure=dialog._source.structure, potential=grid)
    assert exported.read_bytes() == reference.read_bytes()
    dialog.close()


def _dialog_with_a_plane(confirmed: bool, monkeypatch) -> tuple[UepAnalysisDialog, list[str]]:
    asked = _answer_confirmations(monkeypatch, answer=confirmed)
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._slice_planes = [
        _SlicePlaneState(label="Plane 1", plane=_slice_plane_for_dialog(dialog))
    ]
    dialog._resample_all_slice_planes()
    dialog._slice_plane_list.setCurrentRow(0)
    return dialog, asked


@pytest.mark.parametrize("confirmed", [False, True])
def test_the_remove_button_asks_before_it_removes_a_plane(
    qapp, monkeypatch, confirmed: bool
) -> None:
    dialog, asked = _dialog_with_a_plane(confirmed, monkeypatch)

    dialog._slice_remove_plane_button.click()

    assert asked == ["Remove slice plane"]
    assert len(dialog._slice_planes) == (0 if confirmed else 1)
    assert dialog._slice_plane_list.count() == (0 if confirmed else 1)
    dialog.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
@pytest.mark.parametrize("confirmed", [False, True])
def test_both_delete_keys_ask_before_they_remove_a_plane(
    qapp, monkeypatch, key, confirmed: bool
) -> None:
    dialog, asked = _dialog_with_a_plane(confirmed, monkeypatch)
    dialog.show()
    dialog._tabs.setCurrentWidget(dialog._potential_tab)
    dialog.activateWindow()
    dialog._slice_plane_list.setFocus()
    qapp.processEvents()

    QTest.keyClick(dialog._slice_plane_list, key)
    qapp.processEvents()

    assert asked == ["Remove slice plane"]
    assert len(dialog._slice_planes) == (0 if confirmed else 1)
    dialog.close()


def test_a_delete_key_with_no_plane_selected_asks_nothing(qapp, monkeypatch) -> None:
    dialog, asked = _dialog_with_a_plane(True, monkeypatch)
    dialog._slice_plane_list.clearSelection()

    dialog._remove_selected_slice_plane()

    assert asked == []
    assert len(dialog._slice_planes) == 1
    dialog.close()


def _dialog_with_a_muon(confirmed: bool, monkeypatch):
    asked = _answer_confirmations(monkeypatch, answer=confirmed)
    dialog = _si2_dialog()
    dialog._add_custom_muon_to_viewer((0.125, 0.25, 0.375))
    dialog._open_viewer_muon_dialog()
    muon_dialog = dialog._viewer_muon_dialog
    table = muon_dialog._current_table
    table.setCurrentItem(table.topLevelItem(0))
    table.topLevelItem(0).setSelected(True)
    return dialog, muon_dialog, asked


@pytest.mark.parametrize("confirmed", [False, True])
def test_remove_selected_muons_asks_first(qapp, monkeypatch, confirmed: bool) -> None:
    dialog, muon_dialog, asked = _dialog_with_a_muon(confirmed, monkeypatch)

    muon_dialog._remove_selected_button.click()

    assert asked == ["Remove muons"]
    assert len(dialog._viewer_muons) == (0 if confirmed else 1)
    dialog.close()


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
@pytest.mark.parametrize("confirmed", [False, True])
def test_both_delete_keys_ask_before_they_remove_a_muon(
    qapp, monkeypatch, key, confirmed: bool
) -> None:
    dialog, muon_dialog, asked = _dialog_with_a_muon(confirmed, monkeypatch)
    muon_dialog.activateWindow()
    muon_dialog._current_table.setFocus()
    qapp.processEvents()

    QTest.keyClick(muon_dialog._current_table, key)
    qapp.processEvents()

    assert asked == ["Remove muons"]
    assert len(dialog._viewer_muons) == (0 if confirmed else 1)
    dialog.close()


def test_deleting_text_while_renaming_a_muon_does_not_remove_it(qapp, monkeypatch) -> None:
    dialog, muon_dialog, asked = _dialog_with_a_muon(True, monkeypatch)
    muon_dialog.activateWindow()
    table = muon_dialog._current_table
    table.editItem(table.topLevelItem(0), 0)
    qapp.processEvents()
    editor = table.findChild(QLineEdit)
    assert editor is not None
    editor.setFocus()
    qapp.processEvents()

    QTest.keyClick(editor, Qt.Key.Key_Backspace)
    qapp.processEvents()

    assert asked == []
    assert len(dialog._viewer_muons) == 1
    dialog.close()


def test_closing_during_the_rebuilds_starts_no_further_mesh(qapp, monkeypatch) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._add_isosurface_layer()
    started: list[float] = []
    monkeypatch.setattr(
        dialog, "_start_isosurface_mesh_calculation", lambda level, **_: started.append(level)
    )
    dialog._pending_isosurface_rebuilds = [(dialog._isosurface_layers[1].key, 5.0)]

    dialog.close()
    dialog._on_isosurface_mesh_thread_finished()

    assert started == []


def test_a_mesh_follows_its_isosurface_when_another_is_removed(qapp, monkeypatch) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    _show_level(dialog, 5.0)
    dialog._add_isosurface_layer()
    dialog._add_isosurface_layer()
    keys = [layer.key for layer in dialog._isosurface_layers]
    monkeypatch.setattr("mufinder.app.uep_wizard._IsosurfaceMeshThread.start", lambda _thread: None)
    dialog._start_isosurface_mesh_calculation(7.0, layer_index=2)
    dialog._isosurface_mesh_thread = None
    request_id = dialog._isosurface_request_id
    dialog._pending_isosurface_rebuilds = []

    dialog._remove_isosurface_layer_by_key(keys[0])
    mesh = extract_isosurface(dialog._potential_grid_cache, level_ev=7.0)
    dialog._on_isosurface_mesh_ready(request_id, 2, mesh)

    assert [layer.key for layer in dialog._isosurface_layers] == keys[1:]
    assert dialog._isosurface_layers[1].mesh is mesh
    assert dialog._isosurface_layers[1].level_ev == pytest.approx(7.0)
    dialog.close()


def test_a_level_asked_for_while_a_mesh_runs_is_drawn_after_it(qapp, monkeypatch) -> None:
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    monkeypatch.setattr("mufinder.app.uep_wizard._IsosurfaceMeshThread.start", lambda _thread: None)
    dialog._start_isosurface_mesh_calculation(5.0)
    running = dialog._isosurface_request_id

    dialog._isosurface_level_input.setValue(9.0)
    dialog._isosurface_level_input.editingFinished.emit()
    dialog._on_isosurface_mesh_ready(
        running, 0, extract_isosurface(dialog._potential_grid_cache, level_ev=5.0)
    )

    assert dialog._isosurface_level_input.value() == pytest.approx(9.0)
    started: list[float] = []
    monkeypatch.setattr(
        dialog, "_start_isosurface_mesh_calculation", lambda level, **_: started.append(level)
    )
    dialog._on_isosurface_mesh_thread_finished()
    assert started == [pytest.approx(9.0)]
    dialog._isosurface_mesh_thread = None
    dialog.close()


def test_the_next_level_clears_an_earlier_message(qapp, monkeypatch) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = _si2_dialog()
    dialog._apply_potential_grid_cache(potential_grid(dialog._field))
    dialog._set_potential_status("Exported x.xsf", kind="exported")

    dialog._isosurface_level_input.setValue(4.0)
    dialog._isosurface_level_input.editingFinished.emit()

    assert dialog._potential_status.text() == ""
    dialog.close()


def test_a_level_above_a_new_grids_top_is_drawn_at_the_top(qapp, monkeypatch) -> None:
    _run_isosurface_threads_inline(monkeypatch)
    dialog = _si2_dialog()
    grid = potential_grid(dialog._field)
    dialog._isosurface_layers[0].level_ev = 10_000.0

    dialog._on_isosurface_grid_ready(grid)
    qapp.processEvents()

    assert dialog._isosurface_layers[0].level_ev == pytest.approx(
        float(np.ptp(grid.potential_total_ev))
    )
    assert dialog._isosurface_layers[0].mesh is None
    dialog.close()


def test_the_potential_tabs_legend_follows_a_regrouping(qapp, monkeypatch) -> None:
    monkeypatch.setattr(
        "mufinder.app.workflow_table_utils.confirm_destructive", lambda *_a, **_k: True
    )
    host = _no_symmetry_host()
    dialog = _dialog_with_endpoints(monkeypatch, _two_nearby_sites(host), host)
    dialog._run_batch_search()
    second = next(
        candidate
        for candidate in dialog._clustered_muon_candidates()
        if candidate.cluster_index == 2
    )
    dialog._add_candidate_muons_to_viewer([second.key], False)

    dialog._batch_grouping_tolerance.setText("0.5")

    legend = [label.text() for label in dialog._viewer_muon_legend_entries.findChildren(QLabel)]
    assert any("Site 1" in text for text in legend)
    assert not any("Site 2" in text for text in legend)
    dialog.close()
