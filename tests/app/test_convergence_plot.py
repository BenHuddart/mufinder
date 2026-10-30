"""Tests for the convergence plot the examiner and the overview cards share."""

from __future__ import annotations

from matplotlib.figure import Figure
import pytest

from mufinder.app import theme as theme_module
from mufinder.app.convergence_plot import (
    ConvergencePlotData,
    draw_convergence_plot,
    spacing_axis_ticks,
    spacing_tick_label,
)
from mufinder.science.kpoint_grids import kpoint_count_for_spacing

_CUBIC_LATTICE = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
_PLOT_DATA = {
    "cutoffs": [300.0, 400.0, 500.0],
    "energies": [-10.0, -10.01, -10.0105],
    "forces": [0.30, 0.20, 0.19],
    "stresses": [1.5, 0.5, 0.4],
    "energy_threshold": 0.001,
    "force_threshold": 0.05,
    "stress_threshold": 0.1,
    "point_spacings": {"400.0": 0.05},
}


def test_plot_data_round_trips_the_saved_mapping() -> None:
    data = ConvergencePlotData.from_mapping(_PLOT_DATA)

    assert data is not None
    assert data.sweep_values == (300.0, 400.0, 500.0)
    assert data.stresses == (1.5, 0.5, 0.4)
    assert data.point_spacings == {400.0: 0.05}
    assert data.to_mapping() == {**_PLOT_DATA, "point_spacings": {400.0: 0.05}}


@pytest.mark.parametrize(
    "change",
    [
        {"cutoffs": []},
        {"forces": [0.3, 0.2]},
        {"energies": ["not a number", -1.0, -2.0]},
        {"force_threshold": "x"},
    ],
)
def test_plot_data_rejects_an_unplottable_mapping(change: dict) -> None:
    assert ConvergencePlotData.from_mapping({**_PLOT_DATA, **change}) is None
    assert ConvergencePlotData.from_mapping(None) is None


@pytest.mark.parametrize(
    "change", [{"stress_threshold": None}, {"stresses": [1.0, 2.0]}, {"stresses": None}]
)
def test_plot_data_drops_stress_without_a_full_series_and_threshold(change: dict) -> None:
    data = ConvergencePlotData.from_mapping({**_PLOT_DATA, **change})

    assert data is not None
    assert data.stresses == ()
    assert data.stress_threshold is None


def test_plot_data_falls_back_to_the_given_default_thresholds() -> None:
    raw = {key: value for key, value in _PLOT_DATA.items() if not key.endswith("_threshold")}

    data = ConvergencePlotData.from_mapping(
        raw, default_energy_threshold=0.002, default_force_threshold=0.03
    )

    assert data is not None
    assert (data.energy_threshold, data.force_threshold) == (0.002, 0.03)


def test_spacing_axis_ticks_sit_where_castep_places_each_spacing() -> None:
    positions, labels = spacing_axis_ticks(_CUBIC_LATTICE, 1.0, 1000.0)

    assert positions == sorted(positions)
    assert [float(label) for label in labels] == sorted(map(float, labels), reverse=True)
    for position, label in zip(positions, labels, strict=True):
        assert position == pytest.approx(kpoint_count_for_spacing(_CUBIC_LATTICE, float(label)))
    assert all(left != right for left, right in zip(labels, labels[1:], strict=False))
    assert spacing_axis_ticks(_CUBIC_LATTICE, 10.0, 10.0) == ([], [])
    assert spacing_tick_label(0.05) == "0.05"


def test_full_plot_titles_legends_and_stress_axis(qapp) -> None:
    figure = Figure()
    data = ConvergencePlotData.from_mapping(_PLOT_DATA)

    axes = draw_convergence_plot(figure, data, kind="cutoff", compact=False, recommended=400.0)

    colors = theme_module.convergence_plot_colors()
    assert axes.stress is not None and axes.spacing is None
    assert axes.force.get_title() == (
        "Force / stress convergence (thresholds 0.05 eV/Ang, 0.1 GPa)"
    )
    assert axes.energy.get_legend() is not None
    assert axes.energy.lines[0].get_color() == colors.energy
    assert axes.stress.yaxis.label.get_color() == colors.stress
    assert not axes.energy.texts


def test_compact_plot_shows_thresholds_as_badges(qapp) -> None:
    figure = Figure()
    data = ConvergencePlotData.from_mapping(_PLOT_DATA)

    axes = draw_convergence_plot(figure, data, kind="cutoff", compact=True)

    assert axes.energy.get_legend() is None
    assert axes.energy.get_title() == ""
    badges = {text.get_text().split(" ")[0] for text in axes.force.texts}
    assert badges == {"dF", "dS"}
    assert [text.get_text() for text in axes.energy.texts] == ["dE <= 0.001 eV/atom"]


def test_kpoint_plot_uses_a_log_count_axis_and_names_only_the_recommended_grid(qapp) -> None:
    figure = Figure()
    data = ConvergencePlotData(
        sweep_values=(8.0, 27.0, 64.0, 125.0),
        energies=(-10.0, -10.01, -10.011, -10.0111),
        forces=(0.2, 0.1, 0.04, 0.03),
        energy_threshold=0.001,
        force_threshold=0.05,
    )

    axes = draw_convergence_plot(
        figure,
        data,
        kind="kpoint",
        compact=True,
        recommended=64.0,
        recommended_label="4 x 4 x 4",
        lattice_vectors=_CUBIC_LATTICE,
    )
    figure.canvas.draw()

    assert axes.force.get_xscale() == "log"
    assert axes.force.xaxis.get_minorticklocs().size == 0
    x_min, x_max = axes.force.get_xlim()
    visible = [
        tick.get_text()
        for tick in axes.force.get_xticklabels()
        if x_min <= tick.get_position()[0] <= x_max
    ]
    assert visible == ["10", "20", "50", "100"]
    assert axes.spacing is not None
    assert axes.spacing.xaxis.get_minorticklocs().size == 0
    assert [text.get_text() for text in axes.energy.texts].count("4 x 4 x 4") == 1
    # The energy offset text would collide with the spacing labels, so none is used.
    assert axes.energy.yaxis.get_offset_text().get_text() == ""


def test_draw_rejects_an_unknown_kind(qapp) -> None:
    data = ConvergencePlotData.from_mapping(_PLOT_DATA)
    with pytest.raises(ValueError, match="kind"):
        draw_convergence_plot(Figure(), data, kind="phonon", compact=True)


def test_plot_data_without_thresholds_round_trips_with_the_defaults() -> None:
    data = ConvergencePlotData(sweep_values=(1.0, 2.0), energies=(-1.0, -2.0), forces=(0.2, 0.1))

    restored = ConvergencePlotData.from_mapping(data.to_mapping())

    assert restored is not None
    assert (restored.energy_threshold, restored.force_threshold) == (0.001, 0.05)


def test_spacing_axis_ticks_keep_one_tick_per_kpoint_count() -> None:
    # Every spacing coarser than the cell's Gamma-only limit gives one k-point.
    positions, labels = spacing_axis_ticks(_CUBIC_LATTICE, 0.5, 1000.0)

    assert len(positions) == len(set(positions))
    assert positions == sorted(positions)
    assert len(labels) == len(positions)


def test_spacing_axis_follows_a_zoom_of_the_kpoint_axis(qapp) -> None:
    figure = Figure()
    data = ConvergencePlotData(
        sweep_values=(8.0, 27.0, 64.0, 125.0),
        energies=(-10.0, -10.01, -10.011, -10.0111),
        forces=(0.2, 0.1, 0.04, 0.03),
    )
    axes = draw_convergence_plot(
        figure, data, kind="kpoint", compact=False, lattice_vectors=_CUBIC_LATTICE
    )

    axes.force.set_xlim(20.0, 80.0)

    assert axes.spacing is not None
    assert axes.spacing.get_xlim() == pytest.approx((20.0, 80.0))


def test_full_kpoint_plot_names_the_recommended_grid_in_the_legend(qapp) -> None:
    figure = Figure()
    data = ConvergencePlotData(
        sweep_values=(8.0, 27.0, 64.0),
        energies=(-10.0, -10.01, -10.011),
        forces=(0.2, 0.1, 0.04),
        energy_threshold=0.001,
        force_threshold=0.05,
    )

    axes = draw_convergence_plot(
        figure, data, kind="kpoint", compact=False, recommended=27.0, recommended_label="3 x 3 x 3"
    )

    legend = [text.get_text() for text in axes.energy.get_legend().get_texts()]
    assert "Recommended 3 x 3 x 3" in legend
    assert not axes.energy.texts
