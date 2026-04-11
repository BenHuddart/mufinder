"""Tests for the parent convergence overview widget."""

from __future__ import annotations

from matplotlib.colors import to_hex
from PySide6.QtWidgets import QScrollArea

from mufinder.app import theme as theme_module
from mufinder.app.convergence_overview_widget import ConvergenceOverviewWidget


def test_convergence_overview_widget_shows_placeholder_and_emits_navigation(qapp) -> None:
    widget = ConvergenceOverviewWidget()

    widget.set_cutoff_panel(
        summary_rows=[
            ("Project default", "Not set"),
            ("Method", "Not set"),
            ("Criteria", "Not set"),
        ],
        snapshot=None,
        placeholder_message="Adopt a cutoff analysis to pin it here.",
    )
    widget.set_kpoint_panel(
        summary_rows=[
            ("Project default", "Not set"),
            ("Cutoff default", "Not set"),
            ("Method", "Not set"),
            ("Criteria", "Not set"),
        ],
        snapshot=None,
        placeholder_message="Adopt a k-point analysis to pin it here.",
    )

    requested: list[str] = []
    widget.open_cutoff_requested.connect(lambda: requested.append("cutoff"))
    widget.open_kpoint_requested.connect(lambda: requested.append("kpoint"))

    assert isinstance(widget._scroll, QScrollArea)
    assert widget._cutoff_card._plot._placeholder.text() == "Adopt a cutoff analysis to pin it here."
    assert widget._kpoint_card._plot._placeholder.text() == "Adopt a k-point analysis to pin it here."
    assert "Project default:" in widget._cutoff_card._summary_labels["Project default"].text()
    assert "Not set" in widget._cutoff_card._summary_labels["Project default"].text()
    assert "Cutoff default:" in widget._kpoint_card._summary_labels["Cutoff default"].text()
    assert widget._cutoff_card._pinned_zone.minimumHeight() >= 300

    widget._cutoff_card._action_button.click()
    widget._kpoint_card._action_button.click()

    assert requested == ["cutoff", "kpoint"]
    widget.close()


def test_convergence_overview_widget_renders_pinned_snapshots(qapp) -> None:
    widget = ConvergenceOverviewWidget()

    widget.set_cutoff_panel(
        summary_rows=[
            ("Project default", "550 eV"),
            ("Method", "Asymptotic Extrapolation"),
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
            ("Method", "Delta-Threshold"),
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


def test_convergence_overview_widget_plot_colors_follow_active_theme(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        palette = theme_module.current_palette()

        widget = ConvergenceOverviewWidget()
        widget.set_cutoff_panel(
            summary_rows=[
                ("Project default", "550 eV"),
                ("Method", "Delta-Threshold"),
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
        assert to_hex(widget._cutoff_card._plot._figure.get_facecolor()).lower() == palette.bg_surface_alt.lower()
        assert to_hex(energy_ax.get_facecolor()).lower() == palette.bg_surface.lower()
        assert energy_ax.yaxis.label.get_color().lower() == palette.text_primary.lower()
        assert force_ax.xaxis.label.get_color().lower() == palette.text_primary.lower()
        assert energy_ax.texts
        assert energy_ax.texts[-1].get_color().lower() == palette.text_muted.lower()
        widget.close()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
