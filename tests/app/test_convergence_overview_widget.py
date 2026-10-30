"""Tests for the parent convergence overview widget."""

from __future__ import annotations

from matplotlib.colors import to_hex
from PySide6.QtWidgets import QScrollArea
from pytest import approx as pytest_approx

from mufinder.app import theme as theme_module
from mufinder.app.convergence_overview_widget import ConvergenceOverviewWidget


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

    # The dF badge should sit in the historical top-right corner (x≈0.99, ha="right").
    energy_ax, force_ax = widget._cutoff_card._plot._figure.axes
    badges = [t for t in force_ax.texts if "dF" in t.get_text()]
    assert badges, "expected a dF badge on the force subplot"
    assert badges[0].get_position()[0] == pytest_approx(0.99)
    assert badges[0].get_ha() == "right"
    # Badge colours match their respective series for visual association.
    assert badges[0].get_color().lower() == "#b91c1c"
    dE_badges = [t for t in energy_ax.texts if "dE" in t.get_text()]
    assert dE_badges and dE_badges[0].get_color().lower() == "#1d4ed8"
    assert not any("dS" in t.get_text() for t in force_ax.texts)
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
    assert stress_ax.yaxis.label.get_color().lower() == "#d97706"

    # Force subplot now has both dF and dS badges, both right-anchored.
    force_ax = axes[1]
    dF_badges = [t for t in force_ax.texts if "dF" in t.get_text()]
    dS_badges = [t for t in force_ax.texts if "dS" in t.get_text()]
    assert dF_badges and dS_badges
    assert dS_badges[0].get_position()[0] == pytest_approx(0.99)
    assert dS_badges[0].get_ha() == "right"
    assert dF_badges[0].get_ha() == "right"
    # dF should sit strictly to the left of dS once stress is enabled.
    assert dF_badges[0].get_position()[0] < dS_badges[0].get_position()[0]
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
        assert energy_ax.texts
        # The dE badge takes the energy series colour of the active theme.
        assert (
            energy_ax.texts[-1].get_color().lower()
            == theme_module.convergence_plot_colors().energy.lower()
        )
        widget.close()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
