from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox, QPushButton

from mufinder.app.theme import status_brush
from mufinder.app.energy_barriers_widget import (
    EnergyBarrierEndpointRow,
    EnergyBarrierNebResultRow,
    EnergyBarrierPreviewImageRow,
    EnergyBarrierSourceRow,
    EnergyBarriersWidget,
)
from mufinder.app.quantum_corrections_widget import QuantumCorrectionsWidget, QuantumResultRow, QuantumSourceRow
from mufinder.domain.models import Structure


def _sample_structure() -> Structure:
    return Structure(
        label="preview",
        origin="energy_barriers_neb_preview",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )


def test_quantum_widget_emits_configure_for_source_row(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(
                key="site-a",
                label="Site A",
                charge_state="Muonium",
                relative_energy="0.0 eV",
                displacements="9",
                source_job="relax_a",
                stage="singlepoint",
                num_farms_display="",
                status="ready",
            )
        ]
    )

    configured: list[str] = []
    widget.configure_requested.connect(configured.append)

    button = widget._sources_table.cellWidget(0, 8)
    assert isinstance(button, QPushButton)
    assert button.text() == "Configure..."

    button.click()

    assert configured == [["site-a"]]


def test_quantum_widget_batch_configure_emits_selected_source_keys(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(
                key="site-a",
                label="Site A",
                charge_state="Muonium",
                relative_energy="0.0 eV",
                displacements="9",
                source_job="relax_a",
                stage="singlepoint",
                num_farms_display="",
                status="ready",
            ),
            QuantumSourceRow(
                key="site-b",
                label="Site B",
                charge_state="Muonium",
                relative_energy="0.1 eV",
                displacements="12",
                source_job="relax_b",
                stage="singlepoint",
                num_farms_display="",
                status="ready",
            ),
        ]
    )
    configured: list[object] = []
    widget.configure_requested.connect(configured.append)

    widget._sources_table.selectRow(0)
    widget._sources_table.selectRow(1)
    widget._emit_batch_configure()

    assert configured == [["site-b"]]


def test_quantum_widget_uses_shared_status_text_colors(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(
                key="site-a",
                label="Site A",
                charge_state="Muonium",
                relative_energy="0.0 eV",
                displacements="9",
                source_job="relax_a",
                stage="phonon",
                num_farms_display="4",
                status="running",
            )
        ]
    )

    status_item = widget._sources_table.item(0, 7)

    assert status_item is not None
    assert status_item.foreground().color().name() == status_brush("running").color().name()


def test_quantum_widget_emits_reset_to_stage_one_for_selected_rows(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(
                key="site-a",
                label="Site A",
                charge_state="Muonium",
                relative_energy="0.0 eV",
                displacements="9",
                source_job="relax_a",
                stage="phonon",
                num_farms_display="4",
                status="ready",
                resettable=True,
            )
        ]
    )
    widget._sources_table.selectRow(0)

    reset: list[list[str]] = []
    widget.reset_stage_requested.connect(reset.append)

    widget._reset_stage_button.click()

    assert reset == [["site-a"]]


def test_quantum_widget_emits_analysis_for_completed_result(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_results(
        [
            QuantumResultRow(
                key="result-a",
                label="phonon_a",
                zpe="0.1 eV",
                eigenvalues="1:100.0, 2:400.0",
                witness="-0.1",
            )
        ]
    )

    launched: list[str] = []
    widget.analysis_requested.connect(launched.append)

    assert widget._results_table.horizontalHeaderItem(2).text() == "Eigenvalues (cm⁻¹)"
    assert widget._results_table.columnWidth(2) >= widget._results_table.columnWidth(3)

    button = widget._results_table.cellWidget(0, 4)
    assert isinstance(button, QPushButton)
    assert button.text() == "Launch Analysis"

    button.click()

    assert launched == ["result-a"]


def test_energy_barriers_widget_emits_configure_and_view_for_source_row(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="ready",
            )
        ]
    )

    configured: list[str] = []
    viewed: list[str] = []
    widget.configure_result_requested.connect(configured.append)
    widget.view_source_requested.connect(viewed.append)

    configure_button = widget._sources_table.cellWidget(0, 5)
    view_button = widget._sources_table.cellWidget(0, 6)
    assert isinstance(configure_button, QPushButton)
    assert isinstance(view_button, QPushButton)
    assert configure_button.text() == "Configure..."
    assert view_button.text() == "View..."

    configure_button.click()
    view_button.click()

    assert configured == [["path-a"]]
    assert viewed == ["path-a"]


def test_energy_barriers_widget_uses_shared_status_text_colors(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="failed",
            )
        ]
    )

    # NEB Status is column 4 in the Diffusion Paths table; it uses the shared per-status palette.
    status_item = widget._sources_table.item(0, 4)
    assert status_item is not None
    assert status_item.foreground().color().name() == status_brush("failed").color().name()


def test_energy_barriers_widget_batch_configure_emits_selected_source_keys(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="ready",
            ),
            EnergyBarrierSourceRow(
                key="path-b",
                reactant="B",
                product="C",
                charge_state="Muonium",
                path_length="1.4 ang",
                neb_status="ready",
            ),
        ]
    )
    configured: list[object] = []
    widget.configure_result_requested.connect(configured.append)

    widget._sources_table.selectRow(0)
    widget._emit_batch_configure_results()

    assert configured == [["path-a"]]


def test_energy_barriers_widget_emits_generate_paths_with_max_length(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_endpoint_options(
        [
            EnergyBarrierEndpointRow("site-a", "A", "Muonium", "relax_a", "0.0 eV"),
            EnergyBarrierEndpointRow("site-b", "B", "Muonium", "relax_b", "0.1 eV"),
        ],
        reactant_key="site-a",
        product_key="site-b",
        max_path_length_text="2.5",
    )
    widget._max_path_length_input.setText("2.5")

    generated: list[tuple[str, str, str]] = []
    widget.generate_paths_requested.connect(lambda reactant, product, max_length: generated.append((reactant, product, max_length)))

    widget._emit_generate_paths()

    assert generated == [("site-a", "site-b", "2.5")]


def test_energy_barriers_widget_delete_shortcuts_include_child_focus(qapp) -> None:
    widget = EnergyBarriersWidget()

    assert widget._delete_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert widget._backspace_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut


def test_energy_barriers_widget_delete_key_removes_selected_source_from_child_focus(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="A",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="ready",
            ),
            EnergyBarrierSourceRow(
                key="path-b",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.8 ang",
                neb_status="ready",
            ),
        ]
    )
    removed: list[list[str]] = []
    widget.remove_sources_requested.connect(lambda keys: removed.append(list(keys)))
    configure_button = widget._sources_table.cellWidget(0, 5)
    assert isinstance(configure_button, QPushButton)

    widget.show()
    widget._sources_table.selectRow(0)
    configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(configure_button, Qt.Key.Key_Delete)

    assert removed == [["path-a"]]
    assert widget._sources_table.rowCount() == 1
    assert widget._source_rows[0].key == "path-b"


def test_energy_barriers_widget_marks_configured_path_row(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="ready",
                configured=True,
            )
        ]
    )

    configured: list[str] = []
    widget.configure_result_requested.connect(configured.append)

    configure_button = widget._sources_table.cellWidget(0, 5)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured..."

    configure_button.click()

    assert configured == [["path-a"]]


def test_energy_barriers_widget_switches_preview_splitter_for_narrow_widths(qapp) -> None:
    widget = EnergyBarriersWidget()

    widget.resize(900, 800)
    widget._apply_responsive_layout()
    assert widget._preview_splitter.orientation() == Qt.Orientation.Vertical

    widget.resize(1500, 800)
    widget._apply_responsive_layout()
    assert widget._preview_splitter.orientation() == Qt.Orientation.Horizontal


def _sample_preview_rows() -> list[EnergyBarrierPreviewImageRow]:
    return [
        EnergyBarrierPreviewImageRow("REA", "0", "-1 eV", 0.0, -1.0),
        EnergyBarrierPreviewImageRow("TST", "0.5", "-0.8 eV", 0.5, -0.8),
        EnergyBarrierPreviewImageRow("PRO", "1", "-0.9 eV", 1.0, -0.9),
    ]


def test_energy_barriers_widget_stepper_and_plot_navigate_neb_preview(qapp) -> None:
    widget = EnergyBarriersWidget()

    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )

    assert not hasattr(widget, "_preview_slider")
    assert widget._preview_position_label.text() == "Image 1 / 3"
    assert not widget._preview_prev_button.isEnabled()
    assert widget._preview_next_button.isEnabled()

    widget._preview_next_button.click()

    assert "TST" in widget._preview_summary.text()
    assert widget._preview_position_label.text() == "Image 2 / 3"
    assert widget._profile_plot._selected_index == 1

    QTest.keyClick(widget._profile_plot, Qt.Key.Key_Right)

    assert widget._preview_position_label.text() == "Image 3 / 3"
    assert not widget._preview_next_button.isEnabled()

    QTest.keyClick(widget._profile_plot, Qt.Key.Key_Left)

    assert widget._preview_position_label.text() == "Image 2 / 3"
    widget.close()


def test_energy_barriers_widget_preserves_viewer_camera_while_scrubbing(qapp) -> None:
    from mufinder.app.atomic_structure_viewer import _CameraState

    widget = EnergyBarriersWidget()
    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    viewer = widget._preview_viewer
    assert viewer is not None
    viewer._camera_state = _CameraState(17.0, 38.0, 9.0)
    viewer._axis.set_from_state(viewer._camera_state)
    viewer._view = (17.0, 38.0)
    viewer._view_roll = 9.0

    widget._preview_next_button.click()

    assert "TST" in widget._preview_summary.text()
    assert viewer._camera_state == _CameraState(17.0, 38.0, 9.0)
    widget.close()


def test_energy_barriers_widget_neb_results_show_barriers_not_trajectory_paths(qapp) -> None:
    widget = EnergyBarriersWidget()
    table = widget._neb_results_table
    headers = [table.horizontalHeaderItem(index).text() for index in range(table.columnCount())]
    assert headers == ["Path", "Charge", "Length", "Barrier (fwd)", "Barrier (rev)", "ΔE", "Images"]
    assert not table.wordWrap()

    selected: list[str] = []
    widget.neb_result_selected.connect(selected.append)
    trajectory = "/projects/very/long/output/path/neb_job/neb_job.ts"
    widget.set_neb_results(
        [
            EnergyBarrierNebResultRow(
                key="result-a",
                label="site A → site B",
                charge_state="positive muon",
                path_length="1.2 Å",
                barrier_forward="0.2 eV",
                barrier_reverse="0.1 eV",
                delta_energy="0.1 eV",
                images="11",
                transition_state_file=trajectory,
            )
        ]
    )

    assert table.rowCount() == 1
    assert widget._neb_results_group.title() == "Completed NEB Calculations (1)"
    assert table.item(0, 0).text() == "site A → site B"
    assert table.item(0, 3).text() == "0.2 eV"
    assert table.item(0, 4).text() == "0.1 eV"
    row_texts = [table.item(0, column).text() for column in range(table.columnCount())]
    assert trajectory not in row_texts
    assert trajectory in table.item(0, 0).toolTip()
    assert table.maximumHeight() < 16777215
    assert selected == ["result-a"]
    widget.close()


def test_energy_barriers_widget_viewer_popout_and_restore(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    viewer = widget._preview_viewer
    assert viewer is not None
    assert widget._popout_viewer_button.text() == "Pop Out Viewer"

    widget._popout_viewer_button.click()

    window = widget._viewer_popout_window
    assert window is not None
    assert viewer.window() is window
    # Tool windows float above the main window, so scrubbing from the main
    # window never buries the popped-out viewer.
    assert window.windowFlags() & Qt.WindowType.Tool
    assert widget._preview_splitter.widget(1) is widget._viewer_popout_placeholder
    assert widget._popout_viewer_button.text() == "Restore Viewer"

    widget._preview_next_button.click()
    assert "TST" in widget._preview_summary.text()
    assert window._position_label.text() == "Image 2 / 3"

    window._next_button.click()
    assert "PRO" in widget._preview_summary.text()
    assert widget._preview_position_label.text() == "Image 3 / 3"
    assert not window._next_button.isEnabled()

    widget._popout_viewer_button.click()

    assert widget._viewer_popout_window is None
    assert widget._viewer_popout_placeholder is None
    assert widget._preview_splitter.widget(1) is viewer
    assert widget._popout_viewer_button.text() == "Pop Out Viewer"
    widget.close()


def test_reaction_coordinate_profile_renders_ticks_and_barrier_annotation(qapp) -> None:
    from PySide6.QtGui import QPixmap

    from mufinder.app.energy_barriers_widget import _ReactionCoordinateProfile, _nice_ticks

    assert _nice_ticks(0.0, 1.0, 5) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert _nice_ticks(0.0, 0.0, 4) == [0.0]

    plot = _ReactionCoordinateProfile()
    plot.resize(480, 240)
    plot.set_profile(_sample_preview_rows(), selected_index=1)
    pixmap = QPixmap(plot.size())
    plot.render(pixmap)

    geometry = plot._plot_geometry()
    assert geometry is not None
    bounds, points, x_ticks, y_ticks = geometry
    assert len(points) == 3
    assert len(y_ticks) >= 2
    # The y-axis title and tick labels must fit inside the left margin.
    assert bounds.left() > 30.0

    empty = _ReactionCoordinateProfile()
    empty.resize(300, 200)
    empty.render(QPixmap(empty.size()))


def test_reaction_coordinate_profile_plots_energies_relative_to_first_image(qapp) -> None:
    from mufinder.app.energy_barriers_widget import _ReactionCoordinateProfile

    plot = _ReactionCoordinateProfile()
    plot.resize(480, 240)
    # Absolute CASTEP energies (~1.8e5 eV) must not leak into the axis ticks.
    plot.set_profile(
        [
            EnergyBarrierPreviewImageRow("REA", "0", "+0 eV", 0.0, -181556.5),
            EnergyBarrierPreviewImageRow("TST", "0.5", "+0.6 eV", 0.5, -181555.9),
            EnergyBarrierPreviewImageRow("PRO", "1", "+0.2 eV", 1.0, -181556.3),
        ]
    )

    geometry = plot._plot_geometry()
    assert geometry is not None
    _bounds, _points, _x_ticks, y_ticks = geometry
    assert max(abs(tick) for tick in y_ticks) < 10.0


def test_energy_barriers_widget_stepper_stays_below_plot_in_short_panes(qapp) -> None:
    from PySide6.QtCore import QPoint

    widget = EnergyBarriersWidget()
    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    widget._tabs.setCurrentIndex(1)
    widget.resize(1900, 520)
    widget.show()
    qapp.processEvents()

    plot_bottom = widget._profile_plot.mapTo(widget, QPoint(0, widget._profile_plot.height())).y()
    stepper_top = widget._preview_prev_button.mapTo(widget, QPoint(0, 0)).y()
    # The plot must shrink in short panes instead of overpainting the stepper
    # (1px adjacency is tolerated when the layout is in deficit).
    assert stepper_top >= plot_bottom - 1
    widget.close()


def test_reaction_coordinate_profile_click_selects_nearest_point(qapp) -> None:
    from PySide6.QtCore import QPoint

    from mufinder.app.energy_barriers_widget import _ReactionCoordinateProfile

    plot = _ReactionCoordinateProfile()
    plot.resize(480, 240)
    plot.set_profile(_sample_preview_rows(), selected_index=0)

    selected: list[int] = []
    plot.point_selected.connect(selected.append)
    geometry = plot._plot_geometry()
    assert geometry is not None
    target = geometry[1][2]

    QTest.mouseClick(
        plot,
        Qt.MouseButton.LeftButton,
        pos=QPoint(int(target[0]), int(target[1])),
    )

    assert selected == [2]


def test_energy_barriers_source_configure_applies_to_snapshot_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(key="path-a", reactant="A", product="B", charge_state="M", path_length="1.0", neb_status="ready"),
            EnergyBarrierSourceRow(key="path-b", reactant="B", product="C", charge_state="M", path_length="1.2", neb_status="ready"),
        ]
    )
    configured: list[object] = []
    widget.configure_result_requested.connect(configured.append)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))

    btn = widget._sources_table.cellWidget(0, 5)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0, 1)
    btn.clicked.emit()

    assert len(configured) == 1
    assert set(configured[0]) == {"path-a", "path-b"}


def test_energy_barriers_source_configure_single_row_no_prompt(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = EnergyBarriersWidget()
    widget.set_sources(
        [EnergyBarrierSourceRow(key="path-a", reactant="A", product="B", charge_state="M", path_length="1.0", neb_status="ready")]
    )
    configured: list[object] = []
    widget.configure_result_requested.connect(configured.append)
    prompted: list[bool] = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: prompted.append(True) or QMessageBox.StandardButton.Yes))

    btn = widget._sources_table.cellWidget(0, 5)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0,)
    btn.clicked.emit()

    assert configured == [["path-a"]]
    assert prompted == []


def test_energy_barriers_view_button_opens_clicked_row_without_prompt(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(key="path-a", reactant="A", product="B", charge_state="M", path_length="1.0", neb_status="ready"),
            EnergyBarrierSourceRow(key="path-b", reactant="B", product="C", charge_state="M", path_length="1.2", neb_status="ready"),
        ]
    )
    viewed: list[str] = []
    widget.view_source_requested.connect(viewed.append)
    prompted: list[bool] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **kw: prompted.append(True) or QMessageBox.StandardButton.Yes),
    )
    widget._sources_table.selectAll()

    btn = widget._sources_table.cellWidget(0, 6)
    assert isinstance(btn, QPushButton)
    btn.click()

    assert viewed == ["path-a"]
    assert prompted == []


def test_quantum_source_configure_applies_to_snapshot_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(key="site-a", label="A", charge_state="M", relative_energy="0 eV", displacements="9", source_job="j", stage="sp", num_farms_display="", status="ready"),
            QuantumSourceRow(key="site-b", label="B", charge_state="M", relative_energy="0.1 eV", displacements="9", source_job="j", stage="sp", num_farms_display="", status="ready"),
        ]
    )
    configured: list[object] = []
    widget.configure_requested.connect(configured.append)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))

    btn = widget._sources_table.cellWidget(0, 8)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0, 1)
    btn.clicked.emit()

    assert len(configured) == 1
    assert set(configured[0]) == {"site-a", "site-b"}
