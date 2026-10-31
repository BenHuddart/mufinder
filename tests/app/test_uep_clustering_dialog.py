"""Tests for the separate UEP clustering dialog."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

from matplotlib import colors as mcolors
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QSize, Qt
from PySide6.QtGui import QMoveEvent, QResizeEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QSizePolicy

from mufinder.app.atomic_structure_viewer import MUON_VIEWER_RADIUS_ANG, _CameraState
from mufinder.app.theme import cluster_color_map, current_palette, plot_series_colours
from mufinder.app.uep_clustering_dialog import _GEOMETRY_STATE_EMIT_DELAY_MS, UepClusteringDialog
from mufinder.science.uep import StationaryPointClassification, UepOptimizationResult
from mufinder.science.uep_clustering import relaxed_site_number_map
from tests.support.structures import QUARTER_DIAGONAL, host_structure


_host_structure = partial(
    host_structure, positions=QUARTER_DIAGONAL, properties={"pbc": [True, True, True]}
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
    assert np.allclose(
        dialog._structure_viewer._point_overlay_specs[0].radii, MUON_VIEWER_RADIUS_ANG
    )
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
    palette = current_palette()
    for figure in (dialog._left_figure, dialog._bubble_figure):
        axis = figure.axes[0]
        assert axis.xaxis.label.get_color() == palette.text_primary
        assert axis.yaxis.label.get_color() == palette.text_primary
    # Each cluster number sits beside its bubble, clear of the series colour.
    bubble_labels = dialog._bubble_figure.axes[0].texts
    assert len(bubble_labels) == len(finite_clusters)
    assert all(label.get_position()[0] > 0.0 for label in bubble_labels)

    dialog._hier_t_input.setValue(0.18)
    dialog._on_hier_input_changed()

    assert dialog._hier_t_input.value() == pytest.approx(0.18, rel=0.0, abs=1e-3)
    assert list(dialog._hier_cut_line_artist.get_ydata()) == pytest.approx(
        [0.18, 0.18], rel=0.0, abs=1e-3
    )
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
    expected_max_width = max(
        min(snapshot["window_geometry"][2], avail.width()), restored.minimumWidth()
    )
    expected_max_height = max(
        min(snapshot["window_geometry"][3], avail.height()), restored.minimumHeight()
    )
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


def test_uep_clustering_dialog_survives_move_and_resize_while_building_viewer(
    qapp, monkeypatch
) -> None:
    # On Windows the viewer's VTK interactor calls winId() mid-build, which
    # creates the dialog's native window and delivers move/resize events before
    # the splitters exist. Replay that ordering on every platform.
    import mufinder.app.uep_clustering_dialog as dialog_module

    real_viewer = dialog_module.AtomicStructureViewer

    def _viewer_sending_native_events(parent):
        dialog = parent.window()
        QApplication.sendEvent(dialog, QMoveEvent(QPoint(40, 40), QPoint(0, 0)))
        QApplication.sendEvent(dialog, QResizeEvent(QSize(900, 700), QSize(640, 480)))
        return real_viewer(parent)

    monkeypatch.setattr(dialog_module, "AtomicStructureViewer", _viewer_sending_native_events)

    host = _host_structure()
    snapshots: list[dict[str, object]] = []
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )
    dialog.state_changed.connect(snapshots.append)
    QApplication.sendEvent(dialog, QMoveEvent(QPoint(80, 80), QPoint(40, 40)))
    QTest.qWait(_GEOMETRY_STATE_EMIT_DELAY_MS + 150)

    assert snapshots and "right_splitter_sizes" in snapshots[-1]
    dialog.close()


def test_uep_clustering_dialog_coalesces_geometry_state_emissions(qapp) -> None:
    # Each emission makes the UEP wizard re-serialize its whole state, so a
    # window or splitter drag must not emit once per event.
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_optimizations(),
    )
    snapshots: list[dict[str, object]] = []
    dialog.state_changed.connect(snapshots.append)

    for step in range(20):
        QApplication.sendEvent(dialog, QMoveEvent(QPoint(step, step), QPoint(step - 1, step - 1)))
        QApplication.sendEvent(dialog, QResizeEvent(QSize(900 + step, 700), QSize(899 + step, 700)))
        dialog._main_splitter.splitterMoved.emit(300 + step, 1)
        dialog._structure_viewer.view_state_changed.emit()
    assert snapshots == []

    QTest.qWait(_GEOMETRY_STATE_EMIT_DELAY_MS + 150)
    assert len(snapshots) == 1

    dialog._method_combo.setCurrentIndex(1)
    assert len(snapshots) >= 2, "semantic changes still emit immediately"
    dialog.close()


def test_uep_clustering_dialog_dragging_cut_line_updates_entry_and_refreshes_on_release(
    qapp, monkeypatch
) -> None:
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
        SimpleNamespace(
            button=1, inaxes=axis, x=float(start_x), y=float(start_y), ydata=float(start_t)
        )
    )
    dialog._on_left_canvas_motion(
        SimpleNamespace(inaxes=axis, x=float(drag_x), y=float(drag_y), ydata=float(target_t))
    )

    assert calls == []
    assert dialog._hier_t_input.value() == pytest.approx(target_t, rel=0.0, abs=1e-3)

    dialog._on_left_canvas_button_release(
        SimpleNamespace(
            button=1, inaxes=axis, x=float(drag_x), y=float(drag_y), ydata=float(target_t)
        )
    )

    assert calls == [True]
    dialog.close()


def test_uep_clustering_dialog_clicking_cut_line_does_not_refresh_or_open_editor(
    qapp, monkeypatch
) -> None:
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

    event = SimpleNamespace(
        button=1, inaxes=axis, x=float(click_x), y=float(click_y), ydata=float(current_t)
    )
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

    expected_labels = [
        str(relaxed_site_number_map(optimizations)[index]) for index in range(len(optimizations))
    ]
    assert captured["labels"] == expected_labels
    dialog.close()


def test_uep_clustering_dialog_dendrogram_links_take_the_cluster_colours(qapp) -> None:
    host = _host_structure()
    dialog = UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=[
            _optimization((0.65, 0.65, 0.65), 0.50),
            _optimization((0.10, 0.10, 0.10), 0.01),
            _optimization((0.66, 0.67, 0.64), 0.52),
            _optimization((0.12, 0.09, 0.11), 0.02),
        ],
    )
    colours = {
        mcolors.to_hex(collection.get_color()[0]) for collection in dialog._hier_axis.collections
    }
    labels = dialog._current_result.cluster_labels
    assert len(set(labels)) == 2
    assert colours == {
        *(mcolors.to_hex(colour) for colour in cluster_color_map(labels).values()),
        mcolors.to_hex(plot_series_colours().reference),
    }
    dialog.close()


def _energy_ordered_against_labels() -> list[UepOptimizationResult]:
    """Three pairs of sites listed in an order other than their energy order."""

    return [
        _optimization((0.65, 0.65, 0.65), 0.50),
        _optimization((0.66, 0.67, 0.64), 0.52),
        _optimization((0.10, 0.10, 0.10), 0.01),
        _optimization((0.12, 0.09, 0.11), 0.02),
        _optimization((0.35, 0.90, 0.35), 0.25),
        _optimization((0.36, 0.91, 0.34), 0.26),
    ]


def _clustering_dialog_for_colour_checks(initial_state: dict[str, object]) -> UepClusteringDialog:
    host = _host_structure()
    return UepClusteringDialog(
        source_job_label="uep_source_row1",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=_energy_ordered_against_labels(),
        initial_state=initial_state,
    )


def _bubble_colours_by_site(dialog: UepClusteringDialog) -> dict[int, str]:
    """Return the colour of the energy bubble each site's cluster draws."""

    # Each cluster draws its bubble and then the marker at the bubble's centre.
    bubbles = dialog._bubble_figure.axes[0].collections[::2]
    clusters = dialog.current_result().summary.clusters
    assert len(bubbles) == len(clusters)
    return {
        site: mcolors.to_hex(bubble.get_facecolor()[0])
        for cluster, bubble in zip(clusters, bubbles, strict=True)
        for site in cluster.member_indexes
    }


def _structure_colours_by_site(dialog: UepClusteringDialog) -> dict[int, str]:
    spec = dialog._structure_viewer._point_overlay_specs[0]
    return {site: mcolors.to_hex(colour) for site, colour in enumerate(spec.colors)}


def _dendrogram_colours_by_site(dialog: UepClusteringDialog) -> dict[int, str]:
    """Return the colour of the link rising from each site's dendrogram leaf."""

    site_by_number = {number: site for site, number in dialog._site_number_map.items()}
    axis = dialog._hier_axis
    site_by_leaf_x = {
        float(x): site_by_number[int(label.get_text())]
        for x, label in zip(axis.get_xticks(), axis.get_xticklabels(), strict=True)
    }
    return {
        site_by_leaf_x[float(x)]: mcolors.to_hex(collection.get_color()[0])
        for collection in axis.collections
        for segment in collection.get_segments()
        for x, y in segment
        if y == 0.0
    }


def test_uep_clustering_dialog_gives_each_cluster_one_colour_in_every_hierarchical_view(
    qapp,
) -> None:
    dialog = _clustering_dialog_for_colour_checks({"method": "hier", "hierarchical_t": 0.3})
    result = dialog.current_result()
    assert len(result.summary.clusters) == 3

    bubbles = _bubble_colours_by_site(dialog)
    assert bubbles == _structure_colours_by_site(dialog)
    assert bubbles == _dendrogram_colours_by_site(dialog)
    assert len(set(bubbles.values())) == 3
    dialog.close()


def test_uep_clustering_dialog_gives_each_cluster_one_colour_in_every_kmeans_view(qapp) -> None:
    dialog = _clustering_dialog_for_colour_checks({"method": "kmeans", "kmeans_k": 3})
    result = dialog.current_result()
    assert result.config.clustering_method == "kmeans"
    assert len(result.summary.clusters) == 3

    bubbles = _bubble_colours_by_site(dialog)
    points = dialog._left_figure.axes[0].collections[0]
    assert bubbles == _structure_colours_by_site(dialog)
    assert bubbles == {
        site: mcolors.to_hex(colour) for site, colour in enumerate(points.get_facecolor())
    }
    # Each centroid carries the number and the colour of its cluster's bubble.
    centroids = dialog._left_figure.axes[0].collections[1]
    centroid_labels = dialog._left_figure.axes[0].texts
    assert len(centroid_labels) == 3
    for row, label in enumerate(centroid_labels):
        cluster = result.summary.clusters[row]
        assert label.get_text().strip() == f"C{cluster.cluster_index}"
        assert (
            mcolors.to_hex(centroids.get_facecolor()[row]) == (bubbles[cluster.member_indexes[0]])
        )
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
        if label.isVisibleTo(dialog)
        and label.text().startswith(("Showing", "Relaxed muon sites are colored"))
    ]
    dialog.close()


def test_uep_clustering_cut_line_ignores_drags_while_a_popped_out_plot_pans(
    qapp, monkeypatch
) -> None:
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
        SimpleNamespace(
            button=1, inaxes=axis, x=float(start_x), y=float(start_y), ydata=float(start_t)
        )
    )
    dialog._on_left_canvas_motion(
        SimpleNamespace(inaxes=axis, x=float(drag_x), y=float(drag_y), ydata=float(target_t))
    )

    assert dialog._hier_t_input.value() == pytest.approx(start_t, rel=0.0, abs=1e-9)

    dialog._left_frame.show_here()
    assert dialog._hier_toolbar.isEnabled()
    dialog.close()
