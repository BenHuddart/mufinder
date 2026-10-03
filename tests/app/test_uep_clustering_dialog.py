"""Tests for the separate UEP clustering dialog."""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy

from mufinder.app.atomic_structure_viewer import MUON_VIEWER_RADIUS_ANG, _CameraState
from mufinder.app.uep_clustering_dialog import UepClusteringDialog
from mufinder.domain.models import Structure
from mufinder.science.uep import StationaryPointClassification, UepOptimizationResult, relaxed_site_number_map


def _host_structure() -> Structure:
    return Structure(
        id=uuid4(),
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"pbc": [True, True, True]},
    )


def _classification() -> StationaryPointClassification:
    return StationaryPointClassification(
        label="minimum",
        eigenvalues=(1.0, 1.1, 1.2),
        positive_count=3,
        negative_count=0,
        near_zero_count=0,
    )


def _optimization(frac, energy: float) -> UepOptimizationResult:
    position = tuple(float(value) * 5.0 for value in frac)
    return UepOptimizationResult(
        start_position_ang=position,
        optimized_position_ang=position,
        optimized_fractional_position=tuple(float(value) for value in frac),
        classical_energy_ev=float(energy),
        zero_point_energy_ev=0.05,
        total_energy_ev=float(energy),
        success=True,
        message="ok",
        iterations=4,
        classification=_classification(),
    )


def _optimizations() -> list[UepOptimizationResult]:
    return [
        _optimization((0.10, 0.10, 0.10), 0.01),
        _optimization((0.12, 0.09, 0.11), 0.02),
        _optimization((0.65, 0.65, 0.65), 0.50),
        _optimization((0.66, 0.67, 0.64), 0.52),
    ]


def _split_ratio(sizes: list[int]) -> float:
    total = sum(int(value) for value in sizes)
    if total <= 0:
        return 0.0
    return float(sizes[0]) / float(total)


def test_uep_clustering_dialog_hierarchical_view_keeps_entry_and_overlay_in_sync(qapp) -> None:
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )

    assert dialog.current_result() is not None
    assert dialog._method_key() == "hier"
    assert dialog._left_panel_title.text() == "Hierarchical clustering"
    assert dialog._controls_summary_label.text().startswith("Relaxed sites:")
    assert "Clusters:" in dialog._controls_summary_label.text()
    assert dialog._hier_t_input.maximumWidth() <= 126
    assert dialog._hier_controls.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Maximum
    assert dialog._hier_t_input.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert dialog._structure_viewer._point_overlay_specs
    assert np.allclose(dialog._structure_viewer._point_overlay_specs[0].radii, MUON_VIEWER_RADIUS_ANG)
    assert dialog._hier_toolbar.isHidden() is False
    assert dialog._hier_toolbar._home_button.text() == "Reset"
    assert dialog._hier_toolbar._pan_button.text() == "Pan"
    assert dialog._hier_toolbar._zoom_button.text() == "Zoom"
    assert dialog._hier_toolbar._pan_button.isCheckable() is True
    assert dialog._hier_toolbar._zoom_button.isCheckable() is True
    assert dialog._hier_cut_line_artist is not None
    assert not hasattr(dialog, "_hier_t_slider")
    assert dialog._bubble_figure.axes
    finite_clusters = [
        cluster
        for cluster in dialog.current_result().summary.clusters
        if cluster.average_energy_ev is not None and cluster.standard_deviation_ev is not None
    ]
    assert len(dialog._bubble_figure.axes[0].collections) >= 2 * len(finite_clusters)

    dialog._hier_t_input.setValue(0.18)
    dialog._on_hier_input_changed()

    assert dialog._hier_t_input.value() == pytest.approx(0.18, rel=0.0, abs=1e-3)
    assert list(dialog._hier_cut_line_artist.get_ydata()) == pytest.approx([0.18, 0.18], rel=0.0, abs=1e-3)
    assert dialog._left_figure.axes
    dialog.close()


def test_uep_clustering_dialog_kmeans_view_projects_centroids(qapp) -> None:
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
        initial_state={"method": "kmeans", "kmeans_k": 2},
    )

    dialog._method_combo.setCurrentIndex(1)
    dialog._kmeans_k_spin.setValue(2)

    result = dialog.current_result()
    assert result is not None
    assert result.config.clustering_method == "kmeans"
    assert len(result.kmeans_centroids) == 2
    assert dialog._left_panel_title.text() == "K-means clustering"
    assert dialog._controls_summary_label.text().startswith("Relaxed sites:")
    assert not hasattr(dialog, "_kmeans_k_slider")
    assert dialog._kmeans_controls.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Maximum
    assert dialog._kmeans_k_spin.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Fixed
    assert dialog._hier_toolbar.isHidden() is True
    assert dialog._left_figure.axes
    assert len(dialog._left_figure.axes[0].collections) >= 2
    dialog.close()


def test_uep_clustering_dialog_compact_layout_keeps_plots_side_by_side(qapp) -> None:
    """Compact mode reflows only the controls; the dendrogram keeps a usable height.

    Stacking the plot panels vertically on a small screen used to leave the
    dendrogram about 10 px tall once the viewer and energy-spread card took
    their minimum heights.
    """

    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )

    dialog.show()
    qapp.processEvents()
    dialog.resize(960, 700)
    for _ in range(4):
        qapp.processEvents()

    assert dialog._main_splitter.orientation() == Qt.Orientation.Horizontal
    assert dialog._controls_layout.itemAtPosition(1, 0) is not None
    assert dialog._controls_layout.itemAtPosition(2, 0) is not None
    assert dialog._right_splitter.orientation() == Qt.Orientation.Vertical
    assert 0.4 <= _split_ratio(dialog._right_splitter.sizes()) <= 0.6
    assert dialog._left_canvas.height() >= 200
    assert dialog._bubble_canvas.minimumHeight() >= 150
    assert dialog._left_figure.get_layout_engine() is not None
    assert dialog._bubble_figure.get_layout_engine() is not None
    dialog.close()


def test_uep_clustering_dialog_default_layout_balances_three_panels_evenly(qapp) -> None:
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )

    dialog.show()
    dialog.resize(1280, 860)
    for _ in range(4):
        qapp.processEvents()

    assert dialog._main_splitter.orientation() == Qt.Orientation.Horizontal
    assert dialog._right_splitter.orientation() == Qt.Orientation.Vertical
    assert dialog._pending_main_splitter_sizes is not None
    assert dialog._pending_right_splitter_sizes is not None
    assert 0.45 <= _split_ratio(dialog._main_splitter.sizes()) <= 0.55
    assert 0.4 <= _split_ratio(dialog._right_splitter.sizes()) <= 0.6
    dialog.close()


def test_uep_clustering_dialog_refresh_preserves_structure_viewer_state(qapp) -> None:
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
        initial_state={"method": "kmeans", "kmeans_k": 2},
    )

    dialog._method_combo.setCurrentIndex(1)
    dialog._structure_viewer._show_spins.setChecked(False)
    dialog._structure_viewer._camera_state = _CameraState(23.0, 41.0, -7.0)
    dialog._structure_viewer._axis.set_from_state(dialog._structure_viewer._camera_state)
    dialog._structure_viewer._limits = ((-1.0, 2.0), (-3.0, 4.0), (-5.0, 6.0))

    dialog._kmeans_k_spin.setValue(3)

    assert dialog._structure_viewer._show_spins.isChecked() is False
    assert dialog._structure_viewer._camera_state == _CameraState(23.0, 41.0, -7.0)
    assert dialog._structure_viewer._limits == ((-1.0, 2.0), (-3.0, 4.0), (-5.0, 6.0))
    dialog.close()


def test_uep_clustering_dialog_restores_persisted_window_and_viewer_state(qapp) -> None:
    host = _host_structure()
    first = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )
    first.show()
    qapp.processEvents()
    first.resize(1180, 860)
    qapp.processEvents()
    first._method_combo.setCurrentIndex(1)
    first._kmeans_k_spin.setValue(2)
    first._main_splitter.setSizes([480, 700])
    first._right_splitter.setSizes([260, 260])
    first._structure_viewer._show_cell.setChecked(False)
    first._structure_viewer._show_bonds.setChecked(True)
    first._structure_viewer._camera_state = _CameraState(31.0, 64.0, 12.0)
    first._structure_viewer._axis.set_from_state(first._structure_viewer._camera_state)
    first._structure_viewer._limits = ((-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0))
    snapshot = first.state_snapshot()
    first.close()

    restored = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
        initial_state=snapshot,
    )
    restored.show()
    qapp.processEvents()

    assert restored._method_key() == "kmeans"
    assert restored._kmeans_k_spin.value() == 2
    assert restored._pending_main_splitter_sizes == snapshot["splitter_sizes"]
    assert restored._pending_right_splitter_sizes == snapshot["right_splitter_sizes"]
    # SC-3: restored geometry is now clamped to the current screen (clamp_to_screen),
    # so it round-trips exactly only when it already fits; on a smaller screen
    # (e.g. this test's offscreen platform) it is shrunk to fit instead of
    # reproducing the saved size verbatim. clamp_to_screen never shrinks below
    # the dialog's own minimum size, so the floor is whichever is larger.
    restored_geometry = restored.state_snapshot()["window_geometry"]
    avail = restored.screen().availableGeometry()
    expected_max_width = max(min(snapshot["window_geometry"][2], avail.width()), restored.minimumWidth())
    expected_max_height = max(min(snapshot["window_geometry"][3], avail.height()), restored.minimumHeight())
    assert restored_geometry[2] <= expected_max_width
    assert restored_geometry[3] <= expected_max_height
    assert restored._structure_viewer._show_cell.isChecked() is False
    assert restored._structure_viewer._show_bonds.isChecked() is True
    assert restored._structure_viewer._camera_state == _CameraState(31.0, 64.0, 12.0)
    assert restored._structure_viewer._limits == ((-2.0, 2.0), (-3.0, 3.0), (-4.0, 4.0))
    restored.close()


def test_uep_clustering_dialog_close_shuts_down_structure_viewer(qapp, monkeypatch) -> None:
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )

    calls: list[str] = []

    def _shutdown() -> None:
        calls.append("shutdown")

    monkeypatch.setattr(dialog._structure_viewer, "shutdown", _shutdown)

    dialog.close()
    qapp.processEvents()

    assert calls == ["shutdown"]


def test_uep_clustering_dialog_dragging_cut_line_updates_entry_and_refreshes_on_release(qapp, monkeypatch) -> None:
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )

    calls: list[bool] = []
    monkeypatch.setattr(dialog, "refresh_clustering", lambda *args, **kwargs: calls.append(True))

    axis = dialog._hier_axis
    assert axis is not None
    start_t = dialog._current_hier_t()
    x_centre = 0.5 * sum(axis.get_xlim())
    start_x, start_y = axis.transData.transform((x_centre, start_t))
    target_t = min(start_t + 0.08, dialog._hierarchical_distance_max)
    drag_x, drag_y = axis.transData.transform((x_centre, target_t))

    dialog._on_left_canvas_button_press(
        SimpleNamespace(button=1, inaxes=axis, x=float(start_x), y=float(start_y), ydata=float(start_t))
    )
    dialog._on_left_canvas_motion(
        SimpleNamespace(inaxes=axis, x=float(drag_x), y=float(drag_y), ydata=float(target_t))
    )

    assert calls == []
    assert dialog._hier_t_input.value() == pytest.approx(target_t, rel=0.0, abs=1e-3)

    dialog._on_left_canvas_button_release(
        SimpleNamespace(button=1, inaxes=axis, x=float(drag_x), y=float(drag_y), ydata=float(target_t))
    )

    assert calls == [True]
    dialog.close()


def test_uep_clustering_dialog_clicking_cut_line_does_not_refresh_or_open_editor(qapp, monkeypatch) -> None:
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )

    calls: list[bool] = []
    monkeypatch.setattr(dialog, "refresh_clustering", lambda *args, **kwargs: calls.append(True))

    axis = dialog._hier_axis
    assert axis is not None
    current_t = dialog._current_hier_t()
    x_centre = 0.5 * sum(axis.get_xlim())
    click_x, click_y = axis.transData.transform((x_centre, current_t))

    event = SimpleNamespace(button=1, inaxes=axis, x=float(click_x), y=float(click_y), ydata=float(current_t))
    dialog._on_left_canvas_button_press(event)
    dialog._on_left_canvas_button_release(event)

    assert dialog._hier_t_input.value() == pytest.approx(current_t, rel=0.0, abs=1e-3)
    assert calls == []
    dialog.close()


def test_uep_clustering_dialog_dendrogram_labels_use_global_energy_rank(qapp, monkeypatch) -> None:
    host = _host_structure()
    optimizations = [
        _optimization((0.65, 0.65, 0.65), 0.50),
        _optimization((0.10, 0.10, 0.10), 0.01),
        _optimization((0.66, 0.67, 0.64), 0.52),
        _optimization((0.12, 0.09, 0.11), 0.02),
    ]
    captured: dict[str, object] = {}

    from scipy.cluster import hierarchy

    original_dendrogram = hierarchy.dendrogram

    def _capture_dendrogram(*args, **kwargs):
        captured["labels"] = list(kwargs.get("labels", ()))
        return original_dendrogram(*args, **kwargs)

    monkeypatch.setattr(hierarchy, "dendrogram", _capture_dendrogram)

    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=optimizations,
    )

    expected_labels = [str(relaxed_site_number_map(optimizations)[index]) for index in range(len(optimizations))]
    assert captured["labels"] == expected_labels
    dialog.close()


def test_uep_clustering_dialog_has_no_unlaid_label_over_the_viewer_toolbar(qapp) -> None:
    """The structure summary lives in the viewer tooltip, not a stray label.

    A label parented to the right panel but never added to its layout used to
    paint at the panel's top-left corner, on top of the viewer's Cell/Bonds
    checkboxes.
    """

    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )
    dialog.show()
    qapp.processEvents()

    assert "Showing 4 relaxed sites" in dialog._structure_viewer.toolTip()
    assert not [
        label
        for label in dialog.findChildren(QLabel)
        if label.isVisibleTo(dialog) and label.text().startswith(("Showing", "Relaxed muon sites are colored"))
    ]
    dialog.close()


def test_uep_clustering_cut_line_ignores_drags_while_a_popped_out_plot_pans(qapp, monkeypatch) -> None:
    """A pop-out window's Pan/Zoom owns the drag; the cut line must not move with it."""

    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )
    dialog.show()
    qapp.processEvents()
    monkeypatch.setattr(dialog, "refresh_clustering", lambda *args, **kwargs: None)

    dialog._left_frame.pop_out()
    window = dialog._left_frame.popped_out_window
    # The dialog's own dendrogram toolbar is paused while the plot is popped out.
    assert not dialog._hier_toolbar.isEnabled()
    window.toolbar.pan()
    assert window.toolbar.mode.name == "PAN"

    axis = dialog._hier_axis
    start_t = dialog._current_hier_t()
    x_centre = 0.5 * sum(axis.get_xlim())
    start_x, start_y = axis.transData.transform((x_centre, start_t))
    target_t = min(start_t + 0.08, dialog._hierarchical_distance_max)
    drag_x, drag_y = axis.transData.transform((x_centre, target_t))
    dialog._on_left_canvas_button_press(
        SimpleNamespace(button=1, inaxes=axis, x=float(start_x), y=float(start_y), ydata=float(start_t))
    )
    dialog._on_left_canvas_motion(SimpleNamespace(inaxes=axis, x=float(drag_x), y=float(drag_y), ydata=float(target_t)))

    assert dialog._hier_t_input.value() == pytest.approx(start_t, rel=0.0, abs=1e-9)

    dialog._left_frame.show_here()
    assert dialog._hier_toolbar.isEnabled()
    dialog.close()
