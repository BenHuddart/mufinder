"""The structure viewer's controls must fit narrow panes without clipping.

The viewer sits in a splitter pane beside a plot in several analysis dialogs.
Its controls used to sit on one fixed row with a 610 px minimum width, so the
neighbouring plot only got what was left (about 260 px at the dialogs' minimum
width) and the viewer's own buttons were clipped whenever a pane was narrower.
"""

from __future__ import annotations

from uuid import uuid4

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QPushButton, QWidget

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.relaxation_analysis_widget import RelaxationClusteringDialog
from mufinder.app.workflow_table_utils import table_natural_width
from mufinder.domain.models import Structure
from mufinder.science.relaxation_analysis import RelaxedMuonSite
from tests.support.structures import host_structure

# Narrowest viewer the analysis dialogs should be able to give it.
_NARROW_VIEWER_WIDTH = 300


def _shown_viewer(qapp, width: int) -> AtomicStructureViewer:
    viewer = AtomicStructureViewer()
    viewer.set_structure(host_structure(lattice=4.0, species=("Fe", "O")))
    viewer.show()
    viewer.resize(width, 480)
    for _ in range(3):
        qapp.processEvents()
    return viewer


def _right_edge_in(widget: QWidget, ancestor: QWidget) -> int:
    return widget.mapTo(ancestor, QPoint(widget.width(), 0)).x()


def _top_in(widget: QWidget, ancestor: QWidget) -> int:
    return widget.mapTo(ancestor, QPoint(0, 0)).y()


def test_viewer_can_shrink_to_a_narrow_pane(qapp) -> None:
    viewer = AtomicStructureViewer()

    assert viewer.minimumSizeHint().width() <= _NARROW_VIEWER_WIDTH
    viewer.close()


def test_view_direction_buttons_are_compact(qapp) -> None:
    viewer = AtomicStructureViewer()

    for button in (viewer._view_x_button, viewer._view_y_button, viewer._view_z_button):
        assert button.width() < 60
        assert button.minimumWidth() == button.maximumWidth()
    viewer.close()


def test_view_direction_buttons_are_named_after_their_axis(qapp) -> None:
    viewer = AtomicStructureViewer()

    names = [
        button.accessibleName()
        for button in (viewer._view_x_button, viewer._view_y_button, viewer._view_z_button)
    ]

    assert names == ["Look Along a", "Look Along b", "Look Along c"]
    viewer.close()


def test_controls_share_one_line_when_wide(qapp) -> None:
    viewer = _shown_viewer(qapp, 720)

    assert _top_in(viewer._reset_view_button, viewer) <= _top_in(viewer._show_cell, viewer) + 4
    assert _right_edge_in(viewer._reset_view_button, viewer) == viewer.width()
    viewer.close()


def test_controls_wrap_onto_a_second_line_instead_of_clipping_when_narrow(qapp) -> None:
    viewer = _shown_viewer(qapp, _NARROW_VIEWER_WIDTH)

    for control in viewer.findChildren(QPushButton) + [
        viewer._show_cell,
        viewer._show_bonds,
        viewer._show_spins,
    ]:
        if control.isVisibleTo(viewer):
            assert _right_edge_in(control, viewer) <= viewer.width(), (
                f"{control.text()!r} is clipped"
            )
    assert _top_in(viewer._reset_view_button, viewer) > _top_in(viewer._show_cell, viewer) + 10
    # The second control line pushes the hint and the 3D canvas down rather than overlapping them.
    assert (
        _top_in(viewer._interaction_hint, viewer)
        >= _top_in(viewer._reset_view_button, viewer) + viewer._reset_view_button.height()
    )
    viewer.close()


def test_long_selection_text_elides_instead_of_widening_the_viewer(qapp) -> None:
    viewer = _shown_viewer(qapp, _NARROW_VIEWER_WIDTH)
    minimum_before = viewer.minimumSizeHint().width()

    viewer.select_atom(1)
    assert viewer._selection_label.toolTip() == viewer._selection_label.text()

    # A multi-atom summary on a large structure is far wider than a narrow pane.
    long_text = "Selected atoms: 12 | " + ", ".join(f"#{index} Fe" for index in range(1, 40))
    viewer._selection_label.setText(long_text)
    for _ in range(2):
        qapp.processEvents()

    assert viewer._selection_label.fontMetrics().horizontalAdvance(long_text) > viewer.width()
    assert viewer._selection_label.text() == long_text
    assert viewer.minimumSizeHint().width() == minimum_before
    assert viewer.width() == _NARROW_VIEWER_WIDTH
    viewer.close()


def _relaxation_clustering_dialog() -> RelaxationClusteringDialog:
    host = Structure(
        id=uuid4(),
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    fractions = [(0.10, 0.10, 0.10), (0.12, 0.10, 0.10), (0.30, 0.30, 0.30), (0.60, 0.20, 0.40)]
    sites = [
        RelaxedMuonSite(
            job_id=uuid4(),
            job_label="Site A",
            charge_state="positive_muon",
            structure_id=uuid4(),
            site_id=None,
            folded_fractional_position=fraction,
            cartesian_position_ang=tuple(5.0 * value for value in fraction),
            total_energy_ev=-10.0 + 0.1 * index,
        )
        for index, fraction in enumerate(fractions)
    ]
    dialog = RelaxationClusteringDialog(
        workspace={"name": "Workspace"}, host_structure=host, sites=sites
    )
    dialog._method_combo.setCurrentIndex(2)
    return dialog


def _resize_and_settle(qapp, dialog, width: int, height: int) -> None:
    dialog.resize(width, height)
    for _ in range(5):
        qapp.processEvents()


def test_relaxation_clustering_split_keeps_the_summary_table_whole_on_small_windows(qapp) -> None:
    dialog = _relaxation_clustering_dialog()
    try:
        dialog.show()
        _resize_and_settle(qapp, dialog, 900, 620)
        plot_pane, summary_pane = dialog._main_splitter.sizes()

        assert summary_pane >= table_natural_width(dialog._summary_table)
        assert plot_pane >= dialog._main_splitter.widget(0).minimumSizeHint().width()
        # Freed from the viewer's old 610 px minimum, the plot gets more than the ~260 px it had.
        assert plot_pane > 300

        _resize_and_settle(qapp, dialog, 1400, 820)
        plot_pane, summary_pane = dialog._main_splitter.sizes()
        assert abs(plot_pane - summary_pane) <= 2
    finally:
        dialog.close()


def test_relaxation_clustering_split_stops_rebalancing_once_the_user_drags_it(qapp) -> None:
    dialog = _relaxation_clustering_dialog()
    try:
        dialog.show()
        _resize_and_settle(qapp, dialog, 1240, 760)
        dialog._main_splitter.splitterMoved.emit(400, 1)
        dialog._main_splitter.setSizes([400, 800])
        _resize_and_settle(qapp, dialog, 1400, 820)

        plot_pane, summary_pane = dialog._main_splitter.sizes()
        assert plot_pane < summary_pane
    finally:
        dialog.close()
