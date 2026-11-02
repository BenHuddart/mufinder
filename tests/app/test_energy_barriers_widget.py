from __future__ import annotations

from functools import partial
from io import BytesIO
from pathlib import Path

from matplotlib.colors import to_hex
import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QSize, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QTabWidget

from mufinder.app import energy_barriers_widget as energy_barriers_widget_module
from mufinder.app import workflow_table_utils
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.theme import (
    THEME_DARK,
    THEME_LIGHT,
    apply_theme,
    current_palette,
    status_brush,
)
from mufinder.app.energy_barriers_widget import (
    _BandProfilePlot,
    EnergyBarrierEndpointRow,
    EnergyBarrierPreviewImageRow,
    EnergyBarrierSourceRow,
    EnergyBarriersWidget,
)
from mufinder.castep_io.ts_parser import parse_transition_state_file
from mufinder.science.energy_barriers import band_distances_ang
from tests.support.reset_menu import install_fake_menu, open_reset_menu
from tests.support.structures import host_structure


_sample_structure = partial(
    host_structure,
    label="preview",
    origin="energy_barriers_neb_preview",
    species=("Si", "Si", "H:mu"),
    positions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
)


def _path_row(key: str = "path-a", hop: str = "A → B", **fields) -> EnergyBarrierSourceRow:
    fields.setdefault("path_length", "1.2 Å")
    fields.setdefault("status", "ready")
    return EnergyBarrierSourceRow(key=key, hop=hop, **fields)


def _completed_row(key: str = "path-a", hop: str = "A → B", **fields) -> EnergyBarrierSourceRow:
    return _path_row(
        key,
        hop,
        status="completed",
        barrier_forward="495 meV",
        barrier_reverse="213 meV",
        delta_energy="+283 meV",
        band_key="result-a",
        transition_state_file="/projects/neb_job/neb_job.ts",
        **fields,
    )


def _path_table_cells(widget: EnergyBarriersWidget, row: int) -> list[str]:
    table = widget._sources_table
    return [table.item(row, column).text() for column in range(table.columnCount())]


def test_energy_barriers_page_is_one_page_with_barriers_in_the_path_table(qapp) -> None:
    widget = EnergyBarriersWidget()
    table = widget._sources_table
    assert widget.findChildren(QTabWidget) == []
    assert [table.horizontalHeaderItem(index).text() for index in range(table.columnCount())] == [
        "Hop",
        "Length",
        "Farms",
        "Status",
        "Forward",
        "Reverse",
        "ΔE",
    ]

    widget.set_sources([_path_row("path-b"), _completed_row(farms="4")])

    assert widget._sources_group.title() == "Paths (2)"
    assert _path_table_cells(widget, 0) == ["A → B", "1.2 Å", "Off", "ready", "–", "–", "–"]
    assert _path_table_cells(widget, 1) == [
        "A → B",
        "1.2 Å",
        "4",
        "completed",
        "495 meV",
        "213 meV",
        "+283 meV",
    ]
    assert table.item(1, 3).foreground().color().name() == status_brush("completed").color().name()
    assert "/projects/neb_job/neb_job.ts" in table.item(1, 0).toolTip()
    widget.close()


def test_energy_barriers_configure_button_acts_on_the_selection(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources([_path_row("path-a"), _path_row("path-b")])
    configured: list[object] = []
    widget.configure_result_requested.connect(configured.append)
    assert widget._batch_configure_button.text() == "Configure…"
    assert not widget._batch_configure_button.isEnabled()

    widget._sources_table.selectRow(0)
    widget._batch_configure_button.click()

    assert configured == [["path-a"]]
    widget.close()


def test_energy_barriers_widget_emits_generate_paths_with_max_length(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_endpoint_options(
        [
            EnergyBarrierEndpointRow("site-a", "A"),
            EnergyBarrierEndpointRow("site-b", "B (+100 meV)"),
        ],
        reactant_key="site-a",
        product_key="site-b",
        max_path_length_text="2.5",
    )
    assert widget._product_combo.currentText() == "B (+100 meV)"

    generated: list[tuple[str, str, str]] = []
    widget.generate_paths_requested.connect(
        lambda reactant, product, max_length: generated.append((reactant, product, max_length))
    )

    widget._emit_generate_paths()

    assert generated == [("site-a", "site-b", "2.5")]
    widget.close()


def test_energy_barriers_widget_shows_the_path_symmetry_note_only_when_set(qapp) -> None:
    widget = EnergyBarriersWidget()

    assert widget._path_symmetry_note.isHidden()
    widget.set_path_symmetry_note("Paths use the crystal symmetry.")
    assert not widget._path_symmetry_note.isHidden()
    assert widget._path_symmetry_note.text() == "Paths use the crystal symmetry."
    widget.set_path_symmetry_note("")
    assert widget._path_symmetry_note.isHidden()


def test_energy_barriers_widget_delete_shortcuts_include_child_focus(qapp) -> None:
    widget = EnergyBarriersWidget()

    assert widget._delete_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert widget._backspace_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut


@pytest.mark.parametrize("key", [Qt.Key.Key_Delete, Qt.Key.Key_Backspace])
def test_energy_barriers_widget_delete_keys_remove_the_selected_path_after_confirming(
    qapp, monkeypatch: pytest.MonkeyPatch, key: Qt.Key
) -> None:
    monkeypatch.setattr(
        energy_barriers_widget_module, "confirm_row_removal", lambda *_args, **_kwargs: True
    )
    widget = EnergyBarriersWidget()
    widget.set_sources([_path_row("path-a"), _path_row("path-b")])
    removed: list[list[str]] = []
    widget.remove_sources_requested.connect(lambda keys: removed.append(list(keys)))

    widget.show()
    widget._sources_table.selectRow(0)
    widget._batch_configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(widget._batch_configure_button, key)

    assert removed == [["path-a"]]
    assert widget._sources_table.rowCount() == 1
    assert widget._source_rows[0].key == "path-b"
    widget.close()


def test_energy_barriers_inspector_follows_the_table_selection(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources([_path_row("path-a"), _path_row("path-b")])
    selected: list[str] = []
    widget.path_selected.connect(selected.append)

    widget._sources_table.selectRow(1)
    widget._sources_table.selectRow(0)
    widget._sources_table.clearSelection()

    assert selected == ["path-b", "path-a", ""]
    widget.close()


def test_energy_barriers_refresh_keeps_the_selection_and_reloads_only_a_new_band(qapp) -> None:
    widget = EnergyBarriersWidget()
    rows = [_path_row("path-a"), _path_row("path-b")]
    widget.set_sources(rows)
    widget._sources_table.selectRow(1)
    selected: list[str] = []
    widget.path_selected.connect(selected.append)

    # A dashboard flush with the same rows changes nothing.
    item = widget._sources_table.item(0, 0)
    widget.set_sources(list(rows))
    assert widget._sources_table.item(0, 0) is item
    assert selected == []

    # A new row above keeps the same path selected, with no reload.
    widget.set_sources([_path_row("path-new"), *rows])
    assert widget.selected_source_keys() == ["path-b"]
    assert selected == []

    # The selected path's band finishes: the inspector loads it.
    widget.set_sources([_path_row("path-new"), rows[0], _completed_row("path-b")])
    assert selected == ["path-b"]

    # The selected path's name changes: the inspector takes the new name.
    widget.set_sources([_path_row("path-new"), rows[0], _completed_row("path-b", hop="A → C")])
    assert selected == ["path-b", "path-b"]
    selected.clear()

    # The selected path goes away: the inspector empties.
    widget.set_sources([_path_row("path-new")])
    assert widget.selected_source_keys() == []
    assert selected == [""]
    widget.close()


def test_energy_barriers_inspector_switches_between_hop_and_band(qapp) -> None:
    widget = EnergyBarriersWidget()
    assert widget._preview_title.text() == "No path selected"
    assert widget._band_controls.isHidden()
    assert not widget._popout_viewer_button.isEnabled()

    widget.set_path_preview(
        title="A → B",
        summary="1.2 Å  ·  positive muon  ·  pristine supercell",
        structure=_sample_structure(),
        start_cartesian=(0.0, 0.0, 0.0),
        end_cartesian=(1.0, 0.0, 0.0),
    )

    viewer = widget._preview_viewer
    assert viewer is not None
    assert widget._preview_title.text() == "A → B"
    assert widget._band_controls.isHidden()
    assert widget._popout_viewer_button.isEnabled()
    assert len(viewer._point_overlay_specs) == 2
    assert len(viewer._line_overlay_specs) == 1

    widget.set_neb_preview(
        title="A → B  ·  3 images",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )

    assert widget._preview_viewer is viewer
    assert not widget._band_controls.isHidden()
    assert widget._profile_plot.point_count() == 3
    assert list(viewer._point_overlay_specs) == []
    assert list(viewer._line_overlay_specs) == []

    widget.clear_neb_preview()

    assert widget._preview_title.text() == "No path selected"
    assert widget._band_controls.isHidden()
    assert widget._profile_plot.point_count() == 0
    widget.close()


def _viewer_with_its_real_size_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give the viewer the size hint it has on a real display; offscreen it asks for less."""

    monkeypatch.setattr(AtomicStructureViewer, "sizeHint", lambda _self: QSize(640, 496))


def _show_band_at_default_size(qapp, widget: EnergyBarriersWidget, height: int = 674) -> None:
    widget.set_neb_preview(
        title="A → B  ·  3 images",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    widget.resize(1089, height)
    widget.show()
    for _ in range(3):
        qapp.processEvents()


def _rows_shown_without_scrolling(table) -> int:
    return table.viewport().height() // table.rowHeight(0)


@pytest.mark.parametrize("height", [674, 700, 814])
def test_energy_barriers_table_holds_three_rows_beside_a_real_viewer(
    qapp, monkeypatch: pytest.MonkeyPatch, height: int
) -> None:
    _viewer_with_its_real_size_hint(monkeypatch)
    widget = EnergyBarriersWidget()
    widget.set_sources([_path_row(f"path-{index}") for index in range(3)])
    _show_band_at_default_size(qapp, widget, height)

    table = widget._sources_table
    assert table.verticalScrollBar().maximum() == 0
    assert _rows_shown_without_scrolling(table) == 3
    widget.hide()
    widget.close()


def test_energy_barriers_long_table_holds_three_rows_and_grows_to_five_then_scrolls(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    _viewer_with_its_real_size_hint(monkeypatch)
    widget = EnergyBarriersWidget()
    widget.set_sources([_path_row(f"path-{index}") for index in range(10)])
    _show_band_at_default_size(qapp, widget)

    table = widget._sources_table
    assert _rows_shown_without_scrolling(table) >= 3
    assert table.verticalScrollBar().maximum() > 0

    widget.resize(1089, 900)
    qapp.processEvents()

    assert _rows_shown_without_scrolling(table) == 5
    assert table.verticalScrollBar().maximum() > 0
    widget.hide()
    widget.close()


def test_energy_barriers_page_fits_the_default_window(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page's minimum fits what a 1,333 × 864 window leaves it, with a full table."""

    _viewer_with_its_real_size_hint(monkeypatch)
    widget = EnergyBarriersWidget()
    widget.set_sources([_path_row(f"path-{index}") for index in range(10)])
    _show_band_at_default_size(qapp, widget)

    minimum = widget.minimumSizeHint()
    assert minimum.width() <= 1089
    assert minimum.height() <= 674
    assert widget._preview_splitter.height() >= widget._preview_viewer.minimumSizeHint().height()
    widget.hide()
    widget.close()


def test_energy_barriers_placeholder_defers_to_the_title_that_says_why(qapp) -> None:
    """A first selection that cannot be shown, before any viewer exists."""

    widget = EnergyBarriersWidget()
    hint = widget._preview_viewer_placeholder.text()
    assert hint.startswith("Select a path")

    widget.clear_neb_preview(title="A → B: the band file was not found")

    assert widget._preview_viewer is None
    assert widget._preview_title.text() == "A → B: the band file was not found"
    assert widget._preview_viewer_placeholder.text() == ""

    # Deselecting asks for a selection again.
    widget.clear_neb_preview()
    assert widget._preview_viewer_placeholder.text() == hint
    widget.close()


def test_energy_barriers_clearing_after_the_viewer_replaced_the_placeholder(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_path_preview(
        title="A → B",
        summary="1.2 Å",
        structure=_sample_structure(),
        start_cartesian=(0.0, 0.0, 0.0),
        end_cartesian=(1.0, 0.0, 0.0),
    )
    qapp.processEvents()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)

    widget.clear_neb_preview(title="A → B: the band file was not found")
    widget.clear_neb_preview()

    assert widget._preview_title.text() == "No path selected"
    widget.close()


def test_energy_barriers_empty_viewer_says_something_true(qapp) -> None:
    """Deselecting never tells a project with a host to load one."""

    widget = EnergyBarriersWidget()
    widget.set_path_preview(
        title="A → B",
        summary="1.2 Å",
        structure=_sample_structure(),
        start_cartesian=(0.0, 0.0, 0.0),
        end_cartesian=(1.0, 0.0, 0.0),
    )
    viewer = widget._preview_viewer

    widget.clear_neb_preview()

    assert widget._preview_title.text() == "No path selected"
    assert viewer._structure is None
    assert viewer._empty_message.startswith("Select a path")

    # A selected path that cannot be shown says why in the title; the viewer adds nothing.
    widget.clear_neb_preview(title="A → B: the band file was not found")
    assert viewer._empty_message == ""
    widget.close()


def _sample_preview_rows() -> list[EnergyBarrierPreviewImageRow]:
    return [
        EnergyBarrierPreviewImageRow("Image 1", "0.000 Å", "0 meV", 0.0, -1.0),
        EnergyBarrierPreviewImageRow("Image 2", "0.500 Å", "+200 meV", 0.5, -0.8),
        EnergyBarrierPreviewImageRow("Image 3", "1.250 Å", "+100 meV", 1.25, -0.9),
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

    assert widget._preview_summary.text() == "0.500 Å along the band  ·  E − E₀ +200 meV"
    assert widget._preview_position_label.text() == "Image 2 / 3"
    assert widget._profile_plot.selected_index() == 1

    QTest.keyClick(widget._profile_plot._canvas, Qt.Key.Key_Right)

    assert widget._preview_position_label.text() == "Image 3 / 3"
    assert not widget._preview_next_button.isEnabled()

    QTest.keyClick(widget._profile_plot._canvas, Qt.Key.Key_Right)

    assert widget._preview_position_label.text() == "Image 3 / 3"

    QTest.keyClick(widget._profile_plot._canvas, Qt.Key.Key_Left)

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

    assert "0.500 Å" in widget._preview_summary.text()
    assert viewer._camera_state == _CameraState(17.0, 38.0, 9.0)
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
    assert "0.500 Å" in widget._preview_summary.text()
    assert window._position_label.text() == "Image 2 / 3"

    window._next_button.click()
    assert "1.250 Å" in widget._preview_summary.text()
    assert widget._preview_position_label.text() == "Image 3 / 3"
    assert not window._next_button.isEnabled()

    widget._popout_viewer_button.click()

    assert widget._viewer_popout_window is None
    assert widget._viewer_popout_placeholder is None
    assert widget._preview_splitter.widget(1) is viewer
    assert widget._popout_viewer_button.text() == "Pop Out Viewer"
    widget.close()


def _delete_deferred(qapp) -> None:
    qapp.processEvents()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    qapp.processEvents()


def test_energy_barriers_viewer_survives_popping_out_and_restoring_twice(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    viewer = widget._preview_viewer
    widget.show()

    for _cycle in range(2):
        widget._popout_viewer_button.click()
        assert widget._viewer_popout_window is not None
        assert viewer.window() is widget._viewer_popout_window
        widget._popout_viewer_button.click()
        assert widget._viewer_popout_window is None
        _delete_deferred(qapp)

        assert viewer.window() is widget
        # The controls' layout items outlive the window that held the viewer.
        controls = viewer._controls_layout
        assert [controls.itemAt(index).isEmpty() for index in range(controls.count())]
        assert controls.heightForWidth(400) > 0
        viewer.resize(viewer.width() + 40, viewer.height() + 40)
        qapp.processEvents()
    widget.hide()
    widget.close()


def test_energy_barriers_popout_stepper_shows_only_for_a_band(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_path_preview(
        title="A → B",
        summary="1.2 Å",
        structure=_sample_structure(),
        start_cartesian=(0.0, 0.0, 0.0),
        end_cartesian=(1.0, 0.0, 0.0),
    )
    widget._popout_viewer_button.click()
    window = widget._viewer_popout_window
    assert window._prev_button.isHidden()
    assert window._next_button.isHidden()

    widget.set_neb_preview(
        title="A → B  ·  3 images",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    assert not window._next_button.isHidden()
    assert window._position_label.text() == "Image 1 / 3"

    widget.set_path_preview(
        title="A → B",
        summary="1.2 Å",
        structure=_sample_structure(),
        start_cartesian=(0.0, 0.0, 0.0),
        end_cartesian=(1.0, 0.0, 0.0),
    )
    assert window._next_button.isHidden()
    widget.close()


def _profile_plot(qapp, rows: list[EnergyBarrierPreviewImageRow]) -> _BandProfilePlot:
    """A shown and fully drawn profile plot of ``rows``."""

    plot = _BandProfilePlot()
    plot.resize(480, 300)
    plot.set_profile(rows)
    plot.show()
    plot._canvas.draw()
    qapp.processEvents()
    return plot


def _image_pixel(plot: _BandProfilePlot, index: int) -> QPoint:
    """Where image ``index`` is on the canvas, in the logical pixels Qt's events use."""

    canvas = plot._canvas
    x, y = plot._axes.transData.transform(
        (plot._rows[index].distance_value, plot._energies_mev[index])
    )
    ratio = canvas.device_pixel_ratio
    return QPoint(round(x / ratio), round(canvas.height() - y / ratio))


def _annotations(plot: _BandProfilePlot) -> list[str]:
    return [text.get_text() for text in plot._axes.texts]


def test_band_profile_plots_energy_in_mev_against_distance_in_angstrom(qapp) -> None:
    # Absolute CASTEP energies (~1.8e5 eV) must not leak into the axis.
    plot = _profile_plot(
        qapp,
        [
            EnergyBarrierPreviewImageRow("Image 1", "0.000 Å", "0 meV", 0.0, -181556.5),
            EnergyBarrierPreviewImageRow("Image 2", "0.700 Å", "+600 meV", 0.7, -181555.9),
            EnergyBarrierPreviewImageRow("Image 3", "1.900 Å", "+200 meV", 1.9, -181556.3),
        ],
    )

    series = plot._axes.lines[0]
    assert list(series.get_xdata()) == [0.0, 0.7, 1.9]
    assert series.get_ydata() == pytest.approx([0.0, 600.0, 200.0], abs=1e-6)
    assert plot._axes.get_xlabel() == "Distance along band (Å)"
    assert plot._axes.get_ylabel() == "E − E₀ (meV)"
    plot.close()


def test_band_profile_marks_the_transition_state_with_the_forward_barrier(qapp) -> None:
    plot = _profile_plot(qapp, _sample_preview_rows())

    assert _annotations(plot) == ["Forward 200 meV"]
    transition_state = next(line for line in plot._axes.lines if line.get_marker() == "D")
    assert list(transition_state.get_xdata()) == [0.5]
    assert transition_state.get_ydata()[0] == pytest.approx(200.0)
    plot.close()


def _assert_label_inside_axes_and_clear_of_curve(plot: _BandProfilePlot) -> None:
    label = plot._axes.texts[0].get_window_extent(plot._canvas.get_renderer())
    box = plot._axes.bbox
    assert box.x0 <= label.x0 and label.x1 <= box.x1
    assert box.y0 <= label.y0 and label.y1 <= box.y1
    series = plot._axes.lines[0]
    points = plot._axes.transData.transform(
        np.column_stack((series.get_xdata(), series.get_ydata()))
    )
    # Densified, so a segment crossing the label between two images counts too.
    along = np.linspace(0.0, len(points) - 1.0, 2000)
    curve_x = np.interp(along, np.arange(len(points)), points[:, 0])
    curve_y = np.interp(along, np.arange(len(points)), points[:, 1])
    assert not np.any(
        (label.x0 <= curve_x)
        & (curve_x <= label.x1)
        & (label.y0 <= curve_y)
        & (curve_y <= label.y1)
    )


@pytest.mark.parametrize(
    "energies_mev",
    [
        (0.0, 1.0, 0.0, 0.0, 0.0),  # Peak early.
        (0.0, 0.0, 1.0, 0.0, 0.0),  # Peak central.
        (0.0, 0.0, 0.0, 1.0, 0.0),  # Peak late.
        (0.0, 20.0, -100.0, -400.0, -800.0),  # A small rise before a deep drop.
        (0.0, 5.0, -500.0, -1000.0, -1000.0),
        (0.0, 100.0, 200.0, 300.0, 400.0),  # Uphill to the last image.
    ],
)
def test_band_profile_keeps_the_barrier_label_readable(qapp, energies_mev) -> None:
    plot = _profile_plot(
        qapp,
        [
            EnergyBarrierPreviewImageRow(f"Image {n}", "", "", 0.25 * n, energy / 1000.0)
            for n, energy in enumerate(energies_mev)
        ],
    )

    _assert_label_inside_axes_and_clear_of_curve(plot)

    # The smallest the inspector's splitter can make the plot.
    plot.resize(plot._canvas.minimumSize())
    qapp.processEvents()
    plot._canvas.draw()
    assert plot._canvas.width() == plot._canvas.minimumWidth()
    _assert_label_inside_axes_and_clear_of_curve(plot)
    plot.close()


@pytest.mark.parametrize("peak", [0.76, 0.74, 0.5, 0.26, 0.24])
def test_band_profile_barrier_label_fits_the_minimum_width_wherever_the_peak_is(
    qapp, peak: float
) -> None:
    plot = _profile_plot(
        qapp,
        [
            EnergyBarrierPreviewImageRow("Image 1", "", "", 0.0, 0.0),
            EnergyBarrierPreviewImageRow("Image 2", "", "", peak, 0.495),
            EnergyBarrierPreviewImageRow("Image 3", "", "", 1.0, 0.283),
        ],
    )
    plot.resize(plot._canvas.minimumSize())
    qapp.processEvents()
    plot._canvas.draw()

    label = plot._axes.texts[0].get_window_extent(plot._canvas.get_renderer())
    assert plot._axes.bbox.x0 <= label.x0 and label.x1 <= plot._axes.bbox.x1
    plot.close()


@pytest.mark.parametrize("file_format", ["png", "svg", "pdf"])
def test_band_profile_saves_and_keeps_stepping_afterwards(qapp, file_format: str) -> None:
    plot = _profile_plot(qapp, _sample_preview_rows())
    background = plot._background

    saved = BytesIO()
    plot._figure.savefig(saved, format=file_format)

    assert saved.getvalue()
    assert plot._background is background
    plot.set_selected_index(2)
    qapp.processEvents()
    assert list(plot._selected_marker.get_xdata()) == [1.25]
    plot.close()


def test_band_profile_barrier_label_stays_clear_of_the_real_band(qapp) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "gdru2si2" / "GdRu2Si2_12_neb.ts"
    images = parse_transition_state_file(fixture).images
    plot = _profile_plot(
        qapp,
        [
            EnergyBarrierPreviewImageRow(f"Image {number}", "", "", distance, image.energy)
            for number, (distance, image) in enumerate(
                zip(band_distances_ang(images), images), start=1
            )
        ],
    )

    assert _annotations(plot) == ["Forward 495 meV"]
    _assert_label_inside_axes_and_clear_of_curve(plot)
    plot.close()


def test_band_profile_marks_no_transition_state_on_a_band_that_only_falls(qapp) -> None:
    plot = _profile_plot(
        qapp,
        [
            EnergyBarrierPreviewImageRow("Image 1", "0 Å", "0 meV", 0.0, -1.0),
            EnergyBarrierPreviewImageRow("Image 2", "1 Å", "-200 meV", 1.0, -1.2),
        ],
    )

    assert _annotations(plot) == []
    assert all(line.get_marker() != "D" for line in plot._axes.lines)
    plot.close()


def test_band_profile_shows_a_band_with_a_diverged_image(qapp) -> None:
    plot = _profile_plot(
        qapp,
        [
            EnergyBarrierPreviewImageRow("Image 1", "0 Å", "0 meV", 0.0, -1.0),
            EnergyBarrierPreviewImageRow("Image 2", "1 Å", "–", 1.0, float("nan")),
            EnergyBarrierPreviewImageRow("Image 3", "2 Å", "+100 meV", 2.0, -0.9),
        ],
    )
    selected: list[int] = []
    plot.point_selected.connect(selected.append)

    assert _annotations(plot) == ["Forward 100 meV"]
    QTest.mouseClick(plot._canvas, Qt.MouseButton.LeftButton, pos=_image_pixel(plot, 2))

    assert selected == [2]
    plot.close()


def test_band_profile_click_selects_the_nearest_image(qapp) -> None:
    plot = _profile_plot(qapp, _sample_preview_rows())
    selected: list[int] = []
    plot.point_selected.connect(selected.append)

    QTest.mouseClick(
        plot._canvas, Qt.MouseButton.LeftButton, pos=_image_pixel(plot, 2) + QPoint(-4, 3)
    )

    assert selected == [2]
    plot.close()


def test_band_profile_ignores_other_mouse_buttons(qapp) -> None:
    plot = _profile_plot(qapp, _sample_preview_rows())
    selected: list[int] = []
    plot.point_selected.connect(selected.append)

    QTest.mouseClick(plot._canvas, Qt.MouseButton.RightButton, pos=_image_pixel(plot, 2))

    assert selected == []
    plot.close()


def test_band_profile_drag_follows_the_pointer_along_the_band(qapp) -> None:
    plot = _profile_plot(qapp, _sample_preview_rows())
    plot.point_selected.connect(plot.set_selected_index)
    canvas = plot._canvas

    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=_image_pixel(plot, 0))
    # A drag goes by the distance alone: this point is far below image 2.
    QTest.mouseMove(canvas, QPoint(_image_pixel(plot, 1).x(), canvas.height() - 2))
    qapp.processEvents()
    assert plot.selected_index() == 1
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=_image_pixel(plot, 1))

    # Released: the pointer moving on selects nothing.
    QTest.mouseMove(canvas, _image_pixel(plot, 2))
    qapp.processEvents()
    assert plot.selected_index() == 1
    plot.close()


def test_band_profile_hover_points_at_an_image_only_close_to_it(qapp) -> None:
    plot = _profile_plot(qapp, _sample_preview_rows())
    canvas = plot._canvas

    QTest.mouseMove(canvas, _image_pixel(plot, 1))
    qapp.processEvents()
    assert canvas.cursor().shape() == Qt.CursorShape.PointingHandCursor

    QTest.mouseMove(canvas, QPoint(canvas.width() // 2, canvas.height() - 1))
    qapp.processEvents()
    assert canvas.cursor().shape() != Qt.CursorShape.PointingHandCursor
    plot.close()


def test_band_profile_stepping_repaints_only_the_selected_marker(qapp) -> None:
    plot = _profile_plot(qapp, _sample_preview_rows())
    canvas = plot._canvas
    full_draws: list[object] = []
    canvas.mpl_connect("draw_event", full_draws.append)
    before = np.array(canvas.buffer_rgba())

    plot.set_selected_index(2)
    qapp.processEvents()

    assert full_draws == []
    assert list(plot._selected_marker.get_xdata()) == [1.25]
    after = np.array(canvas.buffer_rgba())
    changed_rows, changed_columns = np.nonzero(np.any(before != after, axis=2))
    assert changed_rows.size
    # The change is two marker-sized spots, inside the plot area and nowhere else.
    box = plot._axes.bbox
    height = canvas.get_renderer().height
    assert changed_columns.min() >= box.x0 - 1 and changed_columns.max() <= box.x1 + 1
    assert height - changed_rows.max() >= box.y0 - 1 and height - changed_rows.min() <= box.y1 + 1

    # The same rows again, as a repeated selection sends them, draw nothing either.
    plot.set_profile(_sample_preview_rows(), selected_index=1)
    qapp.processEvents()
    assert full_draws == []
    assert plot.selected_index() == 1

    # A full draw keeps the marker where the steps left it.
    canvas.draw()
    assert len(full_draws) == 1
    assert list(plot._selected_marker.get_xdata()) == [0.5]
    plot.close()


def test_band_page_stepping_does_not_redraw_the_plot(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    canvas = widget._profile_plot._canvas
    # The draw the new band asked for.
    qapp.processEvents()
    full_draws: list[object] = []
    canvas.mpl_connect("draw_event", full_draws.append)

    widget._preview_next_button.click()
    QTest.keyClick(canvas, Qt.Key.Key_Right)
    qapp.processEvents()

    assert widget._profile_plot.selected_index() == 2
    assert full_draws == []
    widget.close()


def test_band_profile_click_on_the_page_moves_the_stepper(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    widget.resize(1100, 800)
    widget.show()
    qapp.processEvents()
    plot = widget._profile_plot
    plot._canvas.draw()

    QTest.mouseClick(plot._canvas, Qt.MouseButton.LeftButton, pos=_image_pixel(plot, 2))

    assert widget._preview_position_label.text() == "Image 3 / 3"
    assert plot.selected_index() == 2
    widget.close()


def test_band_profile_popped_out_steps_and_leaves_the_mouse_to_its_tools(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    widget.resize(1100, 800)
    widget.show()
    qapp.processEvents()
    plot = widget._profile_plot
    plot._frame.pop_out()
    window = plot._frame.popped_out_window
    assert window is not None
    plot._canvas.draw()
    qapp.processEvents()

    QTest.keyClick(plot._canvas, Qt.Key.Key_Right)
    assert widget._preview_position_label.text() == "Image 2 / 3"

    window.toolbar.zoom()
    QTest.mouseClick(plot._canvas, Qt.MouseButton.LeftButton, pos=_image_pixel(plot, 2))
    assert widget._preview_position_label.text() == "Image 2 / 3"
    # Moving the pointer leaves the zoom tool's cursor alone.
    zoom_cursor = plot._canvas.cursor().shape()
    QTest.mouseMove(plot._canvas, _image_pixel(plot, 2))
    qapp.processEvents()
    assert plot._canvas.cursor().shape() == zoom_cursor
    window.toolbar.zoom()

    # A drag cut short by the plot going home scrubs nothing afterwards.
    QTest.mousePress(plot._canvas, Qt.MouseButton.LeftButton, pos=_image_pixel(plot, 1))
    plot._frame.show_here()
    QTest.mouseMove(plot._canvas, _image_pixel(plot, 2))
    qapp.processEvents()
    assert widget._preview_position_label.text() == "Image 2 / 3"

    # A path without a band brings the plot home rather than leaving an empty window.
    widget.clear_neb_preview()
    assert not plot._frame.is_popped_out()
    widget.close()


def test_band_profile_follows_the_theme(qapp) -> None:
    plot = _profile_plot(qapp, _sample_preview_rows())
    previous_mode = current_palette().mode
    other_mode = THEME_LIGHT if previous_mode == THEME_DARK else THEME_DARK
    try:
        apply_theme(qapp, other_mode)
        qapp.processEvents()
        palette = current_palette()
        assert to_hex(plot._figure.get_facecolor()) == palette.bg_primary.lower()
        assert to_hex(plot._selected_marker.get_markeredgecolor()) == palette.bg_primary.lower()
        assert plot.point_count() == 3
    finally:
        apply_theme(qapp, previous_mode)
        plot.close()


def test_energy_barriers_widget_stepper_stays_below_plot_in_short_panes(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_neb_preview(
        title="NEB preview",
        image_rows=_sample_preview_rows(),
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )
    widget.resize(1900, 520)
    widget.show()
    qapp.processEvents()

    plot_bottom = widget._profile_plot.mapTo(widget, QPoint(0, widget._profile_plot.height())).y()
    stepper_top = widget._preview_prev_button.mapTo(widget, QPoint(0, 0)).y()
    # The plot must shrink in short panes instead of overpainting the stepper
    # (1px adjacency is tolerated when the layout is in deficit).
    assert stepper_top >= plot_bottom - 1
    widget.close()


def test_energy_barriers_widget_shows_task_farm_column(qapp) -> None:
    """The path table surfaces each path's configured farm count.

    Farming is a per-path setting that changes both the runtime cost and the
    output filenames, so it needs to be visible without opening Configure."""

    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                hop="A → B",
                path_length="1.2 ang",
                status="ready",
                farms="3",
            ),
            EnergyBarrierSourceRow(
                key="path-b",
                hop="A → C",
                path_length="1.4 ang",
                status="ready",
            ),
        ]
    )

    assert widget._sources_table.item(0, 2).text() == "3"
    # Farming defaults to off, and reads as such rather than as a bare "1".
    assert widget._sources_table.item(1, 2).text() == "Off"


def _neb_source(key: str, status: str) -> EnergyBarrierSourceRow:
    return EnergyBarrierSourceRow(
        key=key,
        hop="A → B",
        path_length="1.2 ang",
        status=status,
    )


def test_energy_barriers_widget_context_menu_emits_reset_for_failed_source(
    qapp, monkeypatch
) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources([_neb_source("path-a", "failed")])

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    open_reset_menu(widget._sources_table, row=0)

    assert payloads == [["path-a"]]


def test_energy_barriers_widget_context_menu_disables_reset_for_running_source(
    qapp, monkeypatch
) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources([_neb_source("path-a", "running")])

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    menu = open_reset_menu(widget._sources_table, row=0)

    assert menu is not None and menu.reset_action().enabled is False
    assert payloads == []


def test_energy_barriers_widget_context_menu_resets_a_multi_row_selection(
    qapp, monkeypatch
) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources([_neb_source("path-a", "failed"), _neb_source("path-b", "cancelled")])
    widget._sources_table.selectAll()

    payloads: list[object] = []
    widget.reset_failed_requested.connect(payloads.append)
    install_fake_menu(monkeypatch)

    open_reset_menu(widget._sources_table, row=0)

    assert payloads == [["path-a", "path-b"]]


def test_energy_barriers_remove_selected_confirms_with_path_wording(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                hop="A → B",
                path_length="1.0",
                status="ready",
            ),
            EnergyBarrierSourceRow(
                key="path-b",
                hop="A → C",
                path_length="1.0",
                status="completed",
            ),
        ]
    )
    removed: list[list[str]] = []
    widget.remove_sources_requested.connect(lambda keys: removed.append(list(keys)))
    prompts: list[tuple[str, ...]] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt: prompts.append(prompt) or answers.pop(0),
    )
    widget._sources_table.selectAll()

    widget._remove_sources_button.click()
    assert removed == []
    assert widget._sources_table.rowCount() == 2

    widget._delete_shortcut.activated.emit()
    assert removed == [["path-a", "path-b"]]
    assert prompts[1] == (
        "Remove paths",
        "Remove 2 paths?",
        "1 path already has a submitted job.",
        "Remove 2 paths",
    )


def test_energy_barriers_run_all_when_nothing_selected_and_shared_menu(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.support.reset_menu import install_fake_menu, open_reset_menu

    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                hop="A → B",
                path_length="1.0",
                status="failed",
            ),
            EnergyBarrierSourceRow(
                key="path-b",
                hop="A → C",
                path_length="1.0",
                status="failed",
            ),
        ]
    )
    submitted: list[list[str]] = []
    inspected: list[object] = []
    widget.submit_neb_local_requested.connect(submitted.append)
    widget.inspect_failed_requested.connect(inspected.append)

    assert widget._remove_sources_button.text() == "Remove Selected"
    assert not widget._remove_sources_button.isEnabled()
    assert widget._submit_neb_local_button.text() == "Run All Locally"
    widget._submit_neb_local_button.click()
    assert submitted == [["path-a", "path-b"]]

    install_fake_menu(monkeypatch)
    widget._sources_table.selectAll()
    assert widget._selection_count_label.text() == "2 selected"
    assert widget._submit_neb_cluster_button.text() == "Run Selected on Cluster"
    menu = open_reset_menu(widget._sources_table)
    assert [action.text for action in menu.actions] == [
        "Run Local",
        "Run on Cluster",
        "Cancel",
        "Inspect Failure",
        "Reset Failed Jobs…",
        "Configure…",
        "Remove",
        "Open Containing Folder",
        "Export Inputs/Outputs…",
    ]
    menu.action("Inspect Failure").triggered.emit()
    assert inspected == [["path-a", "path-b"]]


def _path_source(key: str) -> EnergyBarrierSourceRow:
    return EnergyBarrierSourceRow(
        key=key,
        hop="A → B",
        path_length="1.2 ang",
        status="ready",
    )


@pytest.mark.parametrize(
    ("widget_type", "make_row", "configure_signal", "run_signal", "run_button"),
    [
        (
            EnergyBarriersWidget,
            _path_source,
            "configure_result_requested",
            "submit_neb_local_requested",
            "_submit_neb_local_button",
        ),
    ],
)
def test_source_table_widgets_share_targeting_and_action_state(
    qapp, widget_type, make_row, configure_signal, run_signal, run_button
) -> None:
    widget = widget_type()
    widget.set_sources([make_row("first"), make_row("second")])
    configured: list[object] = []
    run: list[object] = []
    getattr(widget, configure_signal).connect(configured.append)
    getattr(widget, run_signal).connect(run.append)

    assert widget._batch_configure_button.isEnabled() is False
    getattr(widget, run_button).click()

    widget._sources_table.selectRow(1)
    assert widget.selected_source_keys() == ["second"]
    assert widget._batch_configure_button.isEnabled() is True
    widget._batch_configure_button.click()

    assert run == [["first", "second"]]
    assert configured == [["second"]]
