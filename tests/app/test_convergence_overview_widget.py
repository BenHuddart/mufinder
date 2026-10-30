"""Tests for the parent convergence overview widget."""

from __future__ import annotations

from matplotlib.colors import to_hex
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QLabel, QScrollArea
import pytest

from mufinder.app import theme as theme_module
from mufinder.app.convergence_display import format_default_criteria
from mufinder.app.convergence_overview_widget import ConvergenceOverviewWidget, _PinnedPlotWidget
from mufinder.app.main_window import MainWindow

# The size of the page inside the main window at its default size on a 1280 x 864 desktop.
_DEFAULT_PAGE_SIZE = QSize(964, 677)


def test_convergence_overview_widget_shows_placeholder_and_emits_navigation(qapp) -> None:
    widget = ConvergenceOverviewWidget()

    widget.set_cutoff_panel(
        summary_rows=[
            ("Project default", "Not set"),
            ("Criteria", "Not set"),
        ],
        snapshot=None,
        placeholder_message="Adopt a cutoff analysis to pin it here.",
    )
    widget.set_kpoint_panel(
        summary_rows=[
            ("Project default", "Not set"),
            ("Cutoff default", "Not set"),
            ("Criteria", "Not set"),
        ],
        snapshot=None,
        placeholder_message="Adopt a k-point analysis to pin it here.",
    )

    requested: list[str] = []
    widget.open_cutoff_requested.connect(lambda: requested.append("cutoff"))
    widget.open_kpoint_requested.connect(lambda: requested.append("kpoint"))

    assert isinstance(widget._scroll, QScrollArea)
    assert (
        widget._cutoff_card._plot._placeholder.text() == "Adopt a cutoff analysis to pin it here."
    )
    assert (
        widget._kpoint_card._plot._placeholder.text() == "Adopt a k-point analysis to pin it here."
    )
    assert "Project default:" in widget._cutoff_card._summary_labels["Project default"].text()
    assert "Not set" in widget._cutoff_card._summary_labels["Project default"].text()
    assert "Cutoff default:" in widget._kpoint_card._summary_labels["Cutoff default"].text()

    widget._cutoff_card._action_button.click()
    widget._kpoint_card._action_button.click()

    assert requested == ["cutoff", "kpoint"]
    widget.close()


def test_convergence_overview_widget_renders_pinned_snapshots(qapp) -> None:
    widget = ConvergenceOverviewWidget()

    widget.set_cutoff_panel(
        summary_rows=[
            ("Project default", "550 eV"),
            ("Criteria", "dE <= 0.001 eV/atom, dF <= 0.05 eV/Ang"),
        ],
        snapshot={
            "default_value": 550.0,
            "plot_data": {
                "cutoffs": [300.0, 400.0, 550.0],
                "energies": [-10.0, -10.1, -10.15],
                "forces": [0.30, 0.18, 0.05],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        },
        placeholder_message="unused",
    )
    widget.set_kpoint_panel(
        summary_rows=[
            ("Project default", "2 x 2 x 2"),
            ("Cutoff default", "550 eV"),
            ("Criteria", "dE <= 0.001 eV/atom, dF <= 0.05 eV/Ang"),
        ],
        snapshot={
            "default_value": [2, 2, 2],
            "point_labels": {"8.0": "Grid: 2 x 2 x 2", "27.0": "Grid: 3 x 3 x 3"},
            "plot_data": {
                "cutoffs": [8.0, 27.0],
                "energies": [-10.0, -10.01],
                "forces": [0.10, 0.04],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        },
        placeholder_message="unused",
    )

    assert widget._cutoff_card._plot._placeholder.isHidden() is True
    assert widget._cutoff_card._plot._canvas.isHidden() is False
    assert len(widget._cutoff_card._plot._figure.axes) == 2

    assert widget._kpoint_card._plot._placeholder.isHidden() is True
    assert widget._kpoint_card._plot._canvas.isHidden() is False
    assert len(widget._kpoint_card._plot._figure.axes) == 2
    assert "2 x 2 x 2" in widget._kpoint_card._summary_labels["Project default"].text()
    assert "&lt;=" in widget._cutoff_card._summary_labels["Criteria"].text()
    widget.close()


_CUBIC_LATTICE = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]


def _kpoint_snapshot(**extra: object) -> dict[str, object]:
    return {
        "default_value": [3, 3, 3],
        "point_labels": {
            "1.0": "Grid: 1 x 1 x 1",
            "8.0": "Grid: 2 x 2 x 2",
            "27.0": "Grid: 3 x 3 x 3",
            "64.0": "Grid: 4 x 4 x 4",
        },
        "plot_data": {
            "cutoffs": [1.0, 8.0, 27.0, 64.0],
            "energies": [-10.0, -10.01, -10.011, -10.0112],
            "forces": [0.30, 0.10, 0.04, 0.03],
            "energy_threshold": 0.001,
            "force_threshold": 0.05,
            "point_spacings": {"8.0": 0.1, "27.0": 0.067, "64.0": 0.05},
        },
        **extra,
    }


def test_convergence_overview_widget_pinned_kpoint_plot_uses_the_examiner_axis(qapp) -> None:
    widget = ConvergenceOverviewWidget()
    widget.set_kpoint_panel(
        summary_rows=[("Project default", "3 x 3 x 3")],
        snapshot=_kpoint_snapshot(lattice_vectors=_CUBIC_LATTICE),
        placeholder_message="unused",
    )

    energy_ax, force_ax, spacing_ax = widget._kpoint_card._plot._figure.axes
    # k-point count on a log axis along the bottom, without minor ticks.
    assert force_ax.get_xscale() == "log"
    assert force_ax.get_xlabel() == "Number of k-points"
    assert force_ax.xaxis.get_minorticklocs().size == 0
    # Round spacings along the top, at fixed positions.
    assert spacing_ax.get_xlabel() == "K-point spacing (1/Ang)"
    spacings = [float(tick.get_text()) for tick in spacing_ax.get_xticklabels()]
    assert spacings and spacings == sorted(spacings, reverse=True)
    # The adopted grid is named once; no point carries a grid label.
    texts = [text.get_text() for text in energy_ax.texts]
    assert texts.count("3 x 3 x 3") == 1
    assert not any("2 x 2 x 2" in text or "2x2x2" in text for text in texts)
    tick_texts = [tick.get_text() for tick in force_ax.get_xticklabels()]
    assert not any("x" in text for text in tick_texts)
    widget.close()


def test_convergence_overview_widget_pinned_kpoint_plot_without_lattice_has_no_spacing_axis(
    qapp,
) -> None:
    widget = ConvergenceOverviewWidget()
    widget.set_kpoint_panel(
        summary_rows=[("Project default", "3 x 3 x 3")],
        snapshot=_kpoint_snapshot(),
        placeholder_message="unused",
    )

    energy_ax, force_ax = widget._kpoint_card._plot._figure.axes
    assert force_ax.get_xscale() == "log"
    assert [text.get_text() for text in energy_ax.texts].count("3 x 3 x 3") == 1
    widget.close()


def test_convergence_overview_widget_redraws_a_card_only_when_its_snapshot_changes(
    qapp, monkeypatch
) -> None:
    from mufinder.app import convergence_overview_widget as overview_module

    draws: list[str] = []
    real_draw = overview_module.draw_convergence_plot

    def counting_draw(figure, data, **kwargs):
        draws.append(kwargs["kind"])
        return real_draw(figure, data, **kwargs)

    monkeypatch.setattr(overview_module, "draw_convergence_plot", counting_draw)
    widget = ConvergenceOverviewWidget()
    snapshot = _kpoint_snapshot(lattice_vectors=_CUBIC_LATTICE)

    widget.set_kpoint_panel(summary_rows=[], snapshot=snapshot, placeholder_message="unused")
    widget.set_kpoint_panel(summary_rows=[], snapshot=dict(snapshot), placeholder_message="unused")
    assert draws == ["kpoint"]

    changed = _kpoint_snapshot(lattice_vectors=_CUBIC_LATTICE, default_value=[4, 4, 4])
    widget.set_kpoint_panel(summary_rows=[], snapshot=changed, placeholder_message="unused")
    assert draws == ["kpoint", "kpoint"]

    # A placeholder in between means the plot is drawn again when it returns.
    widget.set_kpoint_panel(summary_rows=[], snapshot=None, placeholder_message="none")
    widget.set_kpoint_panel(summary_rows=[], snapshot=changed, placeholder_message="unused")
    assert draws == ["kpoint", "kpoint", "kpoint"]

    # So is a theme change.
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        widget.set_kpoint_panel(summary_rows=[], snapshot=changed, placeholder_message="unused")
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    assert len(draws) == 4
    widget.close()


def test_convergence_overview_widget_skips_stress_when_data_absent(qapp) -> None:
    widget = ConvergenceOverviewWidget()

    widget.set_cutoff_panel(
        summary_rows=[
            ("Project default", "550 eV"),
            ("Criteria", "dE <= 0.001 eV/atom, dF <= 0.05 eV/Ang"),
        ],
        snapshot={
            "default_value": 550.0,
            "plot_data": {
                "cutoffs": [300.0, 400.0, 550.0],
                "energies": [-10.0, -10.1, -10.15],
                "forces": [0.30, 0.18, 0.05],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
            },
        },
        placeholder_message="unused",
    )

    # Two axes only (energy + force) — no twinx for stress.
    assert len(widget._cutoff_card._plot._figure.axes) == 2

    widget.close()


def test_convergence_overview_widget_renders_stress_overlay_when_present(qapp) -> None:
    widget = ConvergenceOverviewWidget()

    widget.set_cutoff_panel(
        summary_rows=[
            ("Project default", "550 eV"),
            ("Criteria", "dE <= 0.001 eV/atom, dF <= 0.05 eV/Ang, dS <= 0.1 GPa"),
        ],
        snapshot={
            "default_value": 550.0,
            "stress_threshold": 0.1,
            "plot_data": {
                "cutoffs": [300.0, 400.0, 550.0],
                "energies": [-10.0, -10.1, -10.15],
                "forces": [0.30, 0.18, 0.05],
                "stresses": [1.50, 0.40, 0.08],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
                "stress_threshold": 0.1,
            },
        },
        placeholder_message="unused",
    )

    # Three axes: energy, force, and the twinx stress axis sharing the force subplot.
    axes = widget._cutoff_card._plot._figure.axes
    assert len(axes) == 3
    stress_ax = axes[-1]
    assert "Max stress" in stress_ax.get_ylabel()
    assert stress_ax.yaxis.label.get_color() == theme_module.plot_series_colours().stress

    # The stress threshold is drawn on the stress axis, and the Criteria row names it.
    assert 0.1 in [line.get_ydata()[0] for line in stress_ax.lines]
    assert "dS <= 0.1 GPa" in format_default_criteria(
        method=None, energy_threshold=0.001, force_threshold=0.05, stress_threshold=0.1
    )
    widget.close()


def test_convergence_overview_widget_ignores_partial_stress_payload(qapp) -> None:
    widget = ConvergenceOverviewWidget()

    # stresses provided but stress_threshold missing → must be skipped.
    widget.set_cutoff_panel(
        summary_rows=[
            ("Project default", "550 eV"),
            ("Criteria", "…"),
        ],
        snapshot={
            "default_value": 550.0,
            "plot_data": {
                "cutoffs": [300.0, 400.0, 550.0],
                "energies": [-10.0, -10.1, -10.15],
                "forces": [0.30, 0.18, 0.05],
                "stresses": [1.50, 0.40, 0.08],
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
                # stress_threshold intentionally omitted
            },
        },
        placeholder_message="unused",
    )
    assert len(widget._cutoff_card._plot._figure.axes) == 2

    # stresses length mismatched with cutoffs → must be skipped.
    widget.set_cutoff_panel(
        summary_rows=[
            ("Project default", "550 eV"),
            ("Criteria", "…"),
        ],
        snapshot={
            "default_value": 550.0,
            "plot_data": {
                "cutoffs": [300.0, 400.0, 550.0],
                "energies": [-10.0, -10.1, -10.15],
                "forces": [0.30, 0.18, 0.05],
                "stresses": [1.50, 0.40],  # length 2 vs 3
                "energy_threshold": 0.001,
                "force_threshold": 0.05,
                "stress_threshold": 0.1,
            },
        },
        placeholder_message="unused",
    )
    assert len(widget._cutoff_card._plot._figure.axes) == 2
    widget.close()


def test_convergence_overview_widget_plot_colors_follow_active_theme(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        palette = theme_module.current_palette()

        widget = ConvergenceOverviewWidget()
        widget.set_cutoff_panel(
            summary_rows=[
                ("Project default", "550 eV"),
                ("Criteria", "dE <= 0.001 eV/atom, dF <= 0.05 eV/Ang"),
            ],
            snapshot={
                "default_value": 550.0,
                "plot_data": {
                    "cutoffs": [300.0, 400.0, 550.0],
                    "energies": [-10.0, -10.1, -10.15],
                    "forces": [0.30, 0.18, 0.05],
                    "energy_threshold": 0.001,
                    "force_threshold": 0.05,
                },
            },
            placeholder_message="unused",
        )

        energy_ax, force_ax = widget._cutoff_card._plot._figure.axes
        assert (
            to_hex(widget._cutoff_card._plot._figure.get_facecolor()).lower()
            == palette.bg_surface_alt.lower()
        )
        assert to_hex(energy_ax.get_facecolor()).lower() == palette.bg_surface.lower()
        assert energy_ax.yaxis.label.get_color().lower() == palette.text_primary.lower()
        assert force_ax.xaxis.label.get_color().lower() == palette.text_primary.lower()
        assert energy_ax.lines[0].get_color() == theme_module.plot_series_colours().energy
        widget.close()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def _cutoff_snapshot(*, with_stress: bool = False) -> dict[str, object]:
    plot_data: dict[str, object] = {
        "cutoffs": [300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "energies": [-5620.0, -5621.5, -5621.8, -5621.85, -5621.86, -5621.86],
        "forces": [0.61, 0.21, 0.08, 0.03, 0.01, 0.01],
        "energy_threshold": 0.001,
        "force_threshold": 0.05,
    }
    if with_stress:
        plot_data["stresses"] = [1.5, 0.4, 0.2, 0.08, 0.05, 0.04]
        plot_data["stress_threshold"] = 0.1
    return {"default_value": 600.0, "plot_data": plot_data}


_FULL_SUMMARY_ROWS = [
    ("Project default", "4 x 4 x 6"),
    ("Cutoff default", "600 eV"),
    ("Functional", "PBE"),
    ("Criteria", "dE <= 0.001 eV/atom, dF <= 0.05 eV/Ang"),
]


def _pin_both_plots(widget: ConvergenceOverviewWidget) -> None:
    widget.set_cutoff_panel(
        summary_rows=_FULL_SUMMARY_ROWS[:1] + _FULL_SUMMARY_ROWS[2:],
        snapshot=_cutoff_snapshot(),
        placeholder_message="unused",
    )
    widget.set_kpoint_panel(
        summary_rows=_FULL_SUMMARY_ROWS,
        snapshot=_kpoint_snapshot(lattice_vectors=_CUBIC_LATTICE),
        placeholder_message="unused",
    )


def _pinned_page(qapp, *, size: QSize = _DEFAULT_PAGE_SIZE) -> ConvergenceOverviewWidget:
    widget = ConvergenceOverviewWidget()
    _pin_both_plots(widget)
    widget.resize(size)
    widget.show()
    qapp.processEvents()
    return widget


def _assert_fits_without_scrolling(widget: ConvergenceOverviewWidget) -> None:
    bar = widget._scroll.verticalScrollBar()
    assert not bar.isVisible()
    assert bar.maximum() == 0
    assert widget._content.height() <= widget._scroll.viewport().height()
    for card in (widget._cutoff_card, widget._kpoint_card):
        canvas = card._plot._canvas
        assert card.geometry().bottom() < widget._content.height()
        assert canvas.height() >= card._plot.minimumHeight()
        # The x axis label of the force panel is inside the visible canvas.
        canvas.draw()
        force_ax = card._plot._figure.axes[1]
        assert force_ax.xaxis.label.get_window_extent(canvas.get_renderer()).y0 >= 0


def test_convergence_overview_keeps_one_line_subtitle_and_no_page_hint(qapp) -> None:
    widget = _pinned_page(qapp)

    subtitle = widget._subtitle
    assert subtitle.height() <= subtitle.fontMetrics().lineSpacing() + 2
    page_text = " ".join(label.text() for label in widget._content.findChildren(QLabel))
    assert "Each pinned plot" not in page_text
    widget.close()


def test_convergence_overview_fits_the_default_window_with_both_plots_pinned(qapp) -> None:
    widget = _pinned_page(qapp)

    _assert_fits_without_scrolling(widget)
    widget.close()


def test_convergence_overview_fits_the_main_window_at_its_default_size(qapp) -> None:
    window = MainWindow()
    window.show()
    page = window._convergence_overview_view
    window._main_stack.setCurrentWidget(page)
    _pin_both_plots(page)
    qapp.processEvents()

    assert page.isVisible()
    _assert_fits_without_scrolling(page)
    window.close()


def test_convergence_overview_plots_shrink_back_after_a_larger_window(qapp) -> None:
    # A canvas's size hint follows its current size; that must not hold the page tall.
    widget = _pinned_page(qapp, size=QSize(1600, 1400))
    assert widget._cutoff_card._plot._canvas.height() > 500

    widget.resize(_DEFAULT_PAGE_SIZE)
    qapp.processEvents()

    _assert_fits_without_scrolling(widget)
    assert not widget._scroll.horizontalScrollBar().isVisible()
    widget.close()


def test_convergence_overview_scrolls_only_when_the_window_is_too_short(qapp) -> None:
    widget = _pinned_page(qapp, size=QSize(964, 400))

    assert widget._scroll.verticalScrollBar().maximum() > 0
    assert widget._cutoff_card._plot._canvas.height() >= widget._cutoff_card._plot.minimumHeight()
    widget.close()


def test_convergence_overview_empty_cards_fit_the_default_window(qapp) -> None:
    widget = ConvergenceOverviewWidget()
    for setter in (widget.set_cutoff_panel, widget.set_kpoint_panel):
        setter(summary_rows=_FULL_SUMMARY_ROWS, snapshot=None, placeholder_message="Adopt one.")
    widget.resize(_DEFAULT_PAGE_SIZE)
    widget.show()
    qapp.processEvents()

    assert not widget._scroll.verticalScrollBar().isVisible()
    assert widget._cutoff_card._plot._placeholder.isVisible()
    widget.close()


def _plot_texts_under_pop_out_button(plot: _PinnedPlotWidget) -> list[str]:
    canvas, figure = plot._canvas, plot._figure
    canvas.draw()
    renderer = canvas.get_renderer()
    button = plot._canvas_frame.pop_out_button
    top_left = canvas.mapFromGlobal(button.mapToGlobal(button.rect().topLeft()))
    scale = canvas.devicePixelRatioF()
    left, right = top_left.x() * scale, (top_left.x() + button.width()) * scale
    top = figure.bbox.height - top_left.y() * scale
    bottom = top - button.height() * scale

    hidden = []
    for axes in figure.axes:
        ticks = [
            tick.label1 if tick.label1.get_visible() else tick.label2
            for axis in (axes.xaxis, axes.yaxis)
            for tick in axis.get_major_ticks()
        ]
        for text in [*axes.texts, axes.xaxis.label, axes.yaxis.label, *ticks]:
            if not text.get_text() or not text.get_visible():
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < right and box.x1 > left and box.y0 < top and box.y1 > bottom:
                hidden.append(text.get_text())
    return hidden


@pytest.mark.parametrize(
    "kind, snapshot",
    [
        ("cutoff", _cutoff_snapshot()),
        ("cutoff", _cutoff_snapshot(with_stress=True)),
        # The 0.05 1/Ang spacing tick falls on the densest grid, at the right of the axis.
        ("kpoint", _kpoint_snapshot(lattice_vectors=_CUBIC_LATTICE)),
        ("kpoint", _kpoint_snapshot()),
    ],
)
def test_convergence_overview_keeps_plot_text_clear_of_the_pop_out_button(
    qapp, kind: str, snapshot: dict[str, object]
) -> None:
    widget = ConvergenceOverviewWidget()
    card = widget._cutoff_card if kind == "cutoff" else widget._kpoint_card
    setter = widget.set_cutoff_panel if kind == "cutoff" else widget.set_kpoint_panel
    setter(summary_rows=_FULL_SUMMARY_ROWS, snapshot=snapshot, placeholder_message="unused")
    widget.resize(_DEFAULT_PAGE_SIZE)
    widget.show()
    qapp.processEvents()

    assert _plot_texts_under_pop_out_button(card._plot) == []
    widget.close()
