"""Tests for the convergence plot the examiner and the overview cards share."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import math
import warnings

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
import numpy as np
import pytest

from mufinder.app import theme as theme_module
from mufinder.app.convergence_plot import (
    ConvergencePlotData,
    clear_convergence_plot,
    draw_convergence_plot,
    offset_tick_label,
    spacing_axis_ticks,
    spacing_tick_label,
    threshold_text,
)
from mufinder.domain.constants import MEV_PER_EV
from mufinder.science.convergence_analysis import (
    ConvergenceAnalysis,
    analyze_convergence,
    analyze_kpoint_convergence,
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
    "num_atoms": 2,
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

    colors = theme_module.plot_series_colours()
    assert axes.stress is not None and axes.spacing is None
    assert axes.energy.get_title() == "Energy per atom relative to 500 eV"
    assert axes.force.get_title() == "Max force and stress relative to 500 eV"
    assert axes.energy.get_legend() is axes.energy_legend is not None
    assert axes.force.get_legend() is axes.force_legend is not None
    assert axes.energy.lines[0].get_color() == colors.energy
    assert axes.energy.get_ylabel() == "ΔE (meV/atom)"
    assert axes.stress.yaxis.label.get_color() == colors.stress
    assert not axes.energy.texts


def test_compact_plot_leaves_thresholds_to_the_cards_criteria_row(qapp) -> None:
    figure = Figure()
    data = ConvergencePlotData.from_mapping(_PLOT_DATA)

    axes = draw_convergence_plot(figure, data, kind="cutoff", compact=True)

    assert axes.energy.get_legend() is None
    assert axes.energy.get_title() == ""
    assert not axes.energy.texts
    assert not axes.force.texts


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


# A cutoff sweep long enough for the power-law fit: per-atom distance from the
# limit decays as Ecut**-2, 1.8 meV/atom at 600 eV, with four atoms.
_FIT_CUTOFFS = (300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
_FIT_ENERGIES = tuple(-5621.0 + 4 * 0.0018 * (cutoff / 600.0) ** -2 for cutoff in _FIT_CUTOFFS)
_FIT_FORCES = (0.62, 0.21, 0.083, 0.034, 0.013, 0.011)
_FIT_STRESSES = (2.1, 0.9, 0.35, 0.12, 0.07, 0.05)


def _sweep(
    *,
    cutoffs=_FIT_CUTOFFS,
    energies=_FIT_ENERGIES,
    forces=_FIT_FORCES,
    stresses=_FIT_STRESSES,
    energy_threshold: float = 0.001,
    force_threshold: float = 0.05,
    stress_threshold: float = 0.1,
) -> tuple[ConvergencePlotData, ConvergenceAnalysis]:
    """The plot data and the criterion's analysis of one cutoff sweep of four atoms."""

    analysis = analyze_convergence(
        cutoffs=cutoffs,
        energies=energies,
        max_forces=forces,
        max_stresses=stresses or None,
        num_atoms=4,
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
        stress_threshold=stress_threshold if stresses else None,
    )
    data = ConvergencePlotData(
        sweep_values=tuple(cutoffs),
        energies=tuple(energies),
        forces=tuple(forces),
        stresses=tuple(stresses),
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
        stress_threshold=stress_threshold if stresses else None,
        num_atoms=4,
    )
    return data, analysis


def _draw_analysis(
    data: ConvergencePlotData, analysis: ConvergenceAnalysis, *, compact: bool = False, **options
):
    figure = Figure(layout="constrained")
    FigureCanvasAgg(figure)
    axes = draw_convergence_plot(
        figure,
        data,
        kind=options.pop("kind", "cutoff"),
        compact=compact,
        recommended=analysis.recommended,
        energy_result=analysis.energy,
        force_result=analysis.force,
        stress_result=analysis.stress,
        **options,
    )
    figure.canvas.draw()
    return figure, axes


def _line(axes, label: str):
    return next(line for line in axes.get_lines() if line.get_label() == label)


def _bands(axes) -> list[tuple[float, float]]:
    bands = []
    for patch in axes.patches:
        if patch.get_label().startswith("Threshold"):
            bottom = patch.get_y()
            bands.append((bottom, bottom + patch.get_height()))
    return bands


def test_energy_panel_plots_the_criterions_per_atom_distances_in_mev(qapp) -> None:
    data, analysis = _sweep()

    _figure, axes = _draw_analysis(data, analysis)

    plotted = list(_line(axes.energy, "Energy").get_ydata())
    # The same numbers the criterion compares, in meV/atom and with their sign.
    assert [abs(value) for value in plotted] == [
        deviation * MEV_PER_EV for _point, deviation in analysis.energy.deviations
    ]
    finest = _FIT_ENERGIES[-1]
    assert plotted == pytest.approx(
        [(energy - finest) / 4 * MEV_PER_EV for energy in _FIT_ENERGIES]
    )
    assert plotted[-1] == 0.0
    assert dict(axes.energy_offsets) == dict(zip(_FIT_CUTOFFS, plotted, strict=True))


def test_each_panel_shades_its_threshold_about_zero(qapp) -> None:
    data, analysis = _sweep()

    _figure, axes = _draw_analysis(data, analysis)

    assert _bands(axes.energy) == [pytest.approx((-1.0, 1.0))]
    assert _bands(axes.force) == [pytest.approx((-0.05, 0.05))]
    # The stress axis shares the force band: 0.1 GPa sits at the height of 0.05 eV/Ang.
    for force, stress in ((0.05, 0.1), (-0.05, -0.1), (0.0, 0.0)):
        assert axes.force.transData.transform((0.0, force))[1] == pytest.approx(
            axes.stress.transData.transform((0.0, stress))[1]
        )
    force_plotted = list(_line(axes.force, "Max force").get_ydata())
    assert [abs(value) for value in force_plotted] == [
        deviation for _point, deviation in analysis.force.deviations
    ]
    stress_plotted = list(_line(axes.stress, "Max stress").get_ydata())
    assert [abs(value) for value in stress_plotted] == [
        deviation for _point, deviation in analysis.stress.deviations
    ]
    legend = [text.get_text() for text in axes.energy_legend.get_texts()]
    assert legend[0] == "Threshold ±1 meV/atom,\nlog scale beyond"


def test_fit_and_its_limit_are_drawn_in_the_panels_units(qapp) -> None:
    data, analysis = _sweep()
    assert analysis.energy.fitted_curve and analysis.energy.extrapolated_energy is not None

    _figure, axes = _draw_analysis(data, analysis)

    finest = _FIT_ENERGIES[-1]
    fit = _line(axes.energy, "Power-law fit")
    assert list(fit.get_ydata()) == pytest.approx(
        [(energy - finest) / 4 * MEV_PER_EV for _x, energy in analysis.energy.fitted_curve]
    )
    limit = next(
        line for line in axes.energy.get_lines() if line.get_label().startswith("Fitted limit")
    )
    assert limit.get_ydata()[0] == pytest.approx(
        (analysis.energy.extrapolated_energy - finest) / 4 * MEV_PER_EV
    )


@pytest.mark.parametrize("kind", ["cutoff", "kpoint"])
@pytest.mark.parametrize("threshold", [0.001, 0.0005])
def test_no_axis_shows_offset_text_or_an_exponent(qapp, kind, threshold) -> None:
    data, analysis = _sweep(
        energy_threshold=threshold, force_threshold=threshold, stress_threshold=threshold
    )
    if kind == "kpoint":
        data = replace(data, sweep_values=(1.0, 8.0, 27.0, 64.0, 125.0, 216.0))

    figure, _axes = _draw_analysis(
        data, analysis, kind=kind, lattice_vectors=_CUBIC_LATTICE if kind == "kpoint" else None
    )

    for axis in figure.axes:
        assert axis.yaxis.get_offset_text().get_text() == ""
        assert axis.xaxis.get_offset_text().get_text() == ""
        labels = [label.get_text() for label in axis.get_yticklabels() if label.get_text()]
        assert all("e" not in label for label in labels), labels


def test_tick_labels_are_plain_numbers() -> None:
    assert offset_tick_label(0.0005) == "0.0005"
    assert offset_tick_label(-0.05) == "\N{MINUS SIGN}0.05"
    assert offset_tick_label(100.0) == "100"
    assert offset_tick_label(0.0) == "0"


def test_coinciding_converged_and_recommended_points_are_each_identifiable(qapp) -> None:
    data, analysis = _sweep()
    assert analysis.recommended == analysis.energy.converged_value == 600.0
    assert analysis.force.converged_value == analysis.stress.converged_value == 600.0

    _figure, axes = _draw_analysis(data, analysis)

    def vertical_lines(axis) -> list:
        return [
            line
            for line in axis.get_lines()
            if len(line.get_xdata()) == 2 and len(set(line.get_xdata())) == 1
        ]

    # One vertical line per panel, the recommendation; each criterion rings its own point.
    assert len(vertical_lines(axes.energy)) == 1
    assert len(vertical_lines(axes.force)) == 1
    assert vertical_lines(axes.stress) == []
    for axis, label, offsets in (
        (axes.energy, "Energy converged at 600 eV", axes.energy_offsets),
        (axes.force, "Force converged at 600 eV", axes.force_offsets),
        (axes.stress, "Stress converged at 600 eV", axes.stress_offsets),
    ):
        ring = _line(axis, label)
        assert list(ring.get_xdata()) == [600.0]
        assert list(ring.get_ydata()) == [offsets[600.0]]
        assert ring.get_markerfacecolor() == "none"
    energy_legend = [text.get_text() for text in axes.energy_legend.get_texts()]
    assert {"Energy converged at 600 eV", "Recommended 600 eV"} <= set(energy_legend)
    force_legend = [text.get_text() for text in axes.force_legend.get_texts()]
    assert {"Force converged at 600 eV", "Stress converged at 600 eV"} <= set(force_legend)


def test_kpoint_legends_and_titles_name_grids(qapp) -> None:
    counts = (8.0, 27.0, 64.0, 125.0)
    energies = (-10.0, -10.01, -10.0101, -10.0101)
    forces = (0.2, 0.03, 0.02, 0.02)
    analysis = analyze_kpoint_convergence(
        kpoint_counts=counts,
        energies=energies,
        max_forces=forces,
        num_atoms=2,
        energy_threshold=0.001,
        force_threshold=0.05,
    )
    data = ConvergencePlotData(
        sweep_values=counts,
        energies=energies,
        forces=forces,
        energy_threshold=0.001,
        force_threshold=0.05,
        num_atoms=2,
    )
    labels = {8.0: "2 x 2 x 2", 27.0: "3 x 3 x 3", 64.0: "4 x 4 x 4", 125.0: "5 x 5 x 5"}

    _figure, axes = _draw_analysis(data, analysis, kind="kpoint", point_labels=labels)

    assert axes.energy.get_title() == "Energy per atom relative to 5 x 5 x 5"
    energy_legend = [text.get_text() for text in axes.energy_legend.get_texts()]
    assert "Energy converged at 3 x 3 x 3" in energy_legend
    assert "Recommended 3 x 3 x 3" in energy_legend


@pytest.mark.parametrize(
    "sweep",
    [
        {"cutoffs": (300.0,), "energies": (-10.0,), "forces": (0.3,), "stresses": ()},
        {
            "cutoffs": (300.0, 400.0),
            "energies": (-10.0, -10.01),
            "forces": (0.3, 0.2),
            "stresses": (),
        },
        {"forces": (0.62, math.nan, 0.083, 0.034, 0.013, 0.011)},
        {"stresses": ()},
        # Energy that rises and falls: the power-law fit is refused.
        {"energies": (-10.0, -10.1, -9.9, -10.3, -10.0, -10.2)},
    ],
    ids=["one point", "two points", "missing force", "no stress", "failed fit"],
)
@pytest.mark.parametrize("compact", [False, True], ids=["full", "compact"])
def test_awkward_sweeps_still_draw(qapp, sweep, compact) -> None:
    data, analysis = _sweep(**sweep)

    figure, axes = _draw_analysis(data, analysis, compact=compact)

    assert axes.energy.get_yscale() == "symlog"
    assert (axes.stress is None) == (not data.stresses)
    if analysis.energy.fitted_curve is None:
        assert not [line for line in axes.energy.get_lines() if line.get_label() == "Power-law fit"]
    for axis in figure.axes:
        assert axis.yaxis.get_offset_text().get_text() == ""


def test_failed_fit_draws_no_fit(qapp) -> None:
    data, analysis = _sweep(energies=(-10.0, -10.1, -9.9, -10.3, -10.0, -10.2))
    assert analysis.energy.fitted_curve is None

    _figure, axes = _draw_analysis(data, analysis)

    labels = [line.get_label() for line in axes.energy.get_lines()]
    assert "Power-law fit" not in labels and "Fitted limit" not in labels


def test_plot_data_without_an_atom_count_shows_the_cells_energy(qapp) -> None:
    raw = {key: value for key, value in _PLOT_DATA.items() if key != "num_atoms"}
    data = ConvergencePlotData.from_mapping(raw)
    assert data is not None and data.num_atoms is None

    figure = Figure(layout="constrained")
    FigureCanvasAgg(figure)
    axes = draw_convergence_plot(figure, data, kind="cutoff", compact=False)
    figure.canvas.draw()

    assert axes.energy.get_ylabel() == "ΔE of the cell (meV)"
    assert axes.energy.get_title() == "Energy of the cell relative to 500 eV"
    assert list(_line(axes.energy, "Energy").get_ydata()) == pytest.approx([10.5, 0.5, 0.0])
    # The threshold is per atom, so a panel of the cell's energy has no band.
    assert _bands(axes.energy) == []
    assert axes.energy.yaxis.get_offset_text().get_text() == ""


@pytest.mark.parametrize("raw", [0, -3, 2.5, "x", float("nan"), float("inf"), True])
def test_plot_data_ignores_an_unusable_atom_count(raw) -> None:
    data = ConvergencePlotData.from_mapping({**_PLOT_DATA, "num_atoms": raw})

    assert data is not None
    assert data.num_atoms is None


@pytest.mark.parametrize("raw", [12, 12.0, "12", "12.0"])
def test_plot_data_reads_a_whole_atom_count_in_any_spelling(raw) -> None:
    data = ConvergencePlotData.from_mapping({**_PLOT_DATA, "num_atoms": raw})

    assert data is not None
    assert data.num_atoms == 12


def test_legends_follow_their_panels_through_a_resize(qapp) -> None:
    data, analysis = _sweep()
    figure, axes = _draw_analysis(data, analysis)

    for size in ((7.0, 4.0), (12.0, 8.0), (9.0, 6.0)):
        figure.set_size_inches(*size)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        energy_legend = axes.energy_legend.get_window_extent(renderer)
        force_legend = axes.force_legend.get_window_extent(renderer)
        stress_axis = axes.stress.yaxis.get_tightbbox(renderer)
        gap = 6.0 * figure.dpi / 72.0
        assert energy_legend.x0 == pytest.approx(axes.energy.bbox.x1 + gap, abs=1.0)
        assert energy_legend.y1 == pytest.approx(axes.energy.bbox.y1, abs=1.0)
        assert force_legend.x0 == pytest.approx(stress_axis.x1 + gap, abs=1.0)
        assert force_legend.y0 == pytest.approx(axes.force.bbox.y0, abs=1.0)
        assert max(energy_legend.x1, force_legend.x1) <= figure.bbox.x1 + 1.0


def test_titles_name_the_point_each_series_is_measured_from(qapp) -> None:
    # Before analysis the finest cutoff may have a force but no energy yet.
    data = ConvergencePlotData(
        sweep_values=(300.0, 400.0, 500.0),
        energies=(-10.0, -10.01, math.nan),
        forces=(0.3, 0.2, 0.19),
        num_atoms=2,
    )
    figure = Figure(layout="constrained")
    FigureCanvasAgg(figure)

    axes = draw_convergence_plot(figure, data, kind="cutoff", compact=False)

    assert axes.energy.get_title() == "Energy per atom relative to 400 eV"
    assert axes.force.get_title() == "Max force relative to 500 eV"


def test_threshold_legend_keeps_every_digit_of_the_threshold(qapp) -> None:
    data, analysis = _sweep(energy_threshold=0.0012345, force_threshold=0.0375)

    _figure, axes = _draw_analysis(data, analysis)

    energy_legend = [text.get_text() for text in axes.energy_legend.get_texts()]
    force_legend = [text.get_text() for text in axes.force_legend.get_texts()]
    assert energy_legend[0] == "Threshold ±1.2345 meV/atom,\nlog scale beyond"
    assert force_legend[0] == "Thresholds ±0.0375 eV/Ang,\n±0.1 GPa, log scale beyond"


def test_an_infinite_threshold_draws_without_a_band(qapp) -> None:
    raw = {**_PLOT_DATA, "force_threshold": float("inf"), "energy_threshold": float("inf")}
    data = ConvergencePlotData.from_mapping(raw)
    figure = Figure(layout="constrained")
    FigureCanvasAgg(figure)

    axes = draw_convergence_plot(figure, data, kind="cutoff", compact=False)
    figure.canvas.draw()

    assert _bands(axes.energy) == [] and _bands(axes.force) == []


# --- The CoF2 docs cutoff sweep (six atoms) ---------------------------------

_COF2_CUTOFFS = (300.0, 400.0, 500.0, 600.0, 700.0, 800.0)
_COF2_ENERGIES = tuple(-5621.8743 + above for above in (1.84, 0.37, 0.061, 0.0049, 0.0007, 0.0))
_COF2_FORCES = (0.62, 0.21, 0.083, 0.034, 0.013, 0.011)


def _cof2(energy_threshold: float = 0.001, force_threshold: float = 0.05):
    analysis = analyze_convergence(
        cutoffs=_COF2_CUTOFFS,
        energies=_COF2_ENERGIES,
        max_forces=_COF2_FORCES,
        num_atoms=6,
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
    )
    data = ConvergencePlotData(
        sweep_values=_COF2_CUTOFFS,
        energies=_COF2_ENERGIES,
        forces=_COF2_FORCES,
        energy_threshold=energy_threshold,
        force_threshold=force_threshold,
        num_atoms=6,
    )
    return data, analysis


@pytest.mark.parametrize("threshold", [0.001, 0.002, 0.01])
def test_fitted_limit_sits_at_e_inf_and_names_the_sweep_length_gap(qapp, threshold) -> None:
    data, analysis = _cof2(threshold)
    check = analysis.sweep_length
    assert check is not None and check.extrapolated_energy is not None

    _figure, axes = _draw_analysis(data, analysis)

    finest = _COF2_ENERGIES[-1]
    limit_line = next(
        line for line in axes.energy.get_lines() if line.get_label().startswith("Fitted limit")
    )
    # Where E_inf really is relative to the finest point.
    assert limit_line.get_ydata()[0] == pytest.approx(
        (check.extrapolated_energy - finest) / 6 * MEV_PER_EV
    )
    # The text's number is the fit's distance from the limit at the finest cutoff: the
    # plot shows it as the gap between the dashed fit's end and the dotted limit.
    fit = _line(axes.energy, "Power-law fit")
    gap = fit.get_ydata()[-1] - limit_line.get_ydata()[0]
    assert gap == pytest.approx(check.finest_gap * MEV_PER_EV, rel=1e-6)
    gap_text = np.format_float_positional(
        check.finest_gap * MEV_PER_EV, precision=2, unique=True, fractional=False, trim="-"
    )
    assert limit_line.get_label() == (
        f"Fitted limit, {gap_text} meV/atom\nbelow the fitted curve at 800 eV"
    )
    # The limit is labelled on the plot, so it stays visible on the zero gridline.
    assert "E\N{INFINITY}" in [text.get_text() for text in axes.energy.texts]


def test_fitted_limit_is_not_hidden_by_the_zero_gridline(qapp) -> None:
    data, analysis = _cof2(0.001)

    _figure, axes = _draw_analysis(data, analysis)

    limit_line = next(
        line for line in axes.energy.get_lines() if line.get_label().startswith("Fitted limit")
    )
    assert limit_line.get_linewidth() > axes.energy.yaxis.get_gridlines()[0].get_linewidth()
    assert limit_line.get_zorder() > axes.energy.yaxis.get_gridlines()[0].get_zorder()


def _labelled_ticks(axis) -> list[tuple[float, str]]:
    low, high = sorted(axis.get_ylim())
    return [
        (label.get_position()[1], label.get_text())
        for label in axis.get_yticklabels()
        if label.get_text() and low <= label.get_position()[1] <= high
    ]


@pytest.mark.parametrize("kind", ["cutoff", "kpoint"])
@pytest.mark.parametrize("compact", [False, True], ids=["full", "compact"])
def test_every_panel_labels_two_ticks_beyond_the_band_when_there_is_room(
    qapp, kind, compact
) -> None:
    data, analysis = _cof2()
    if kind == "kpoint":
        data = replace(data, sweep_values=(12.0, 36.0, 96.0, 175.0, 288.0, 500.0))

    figure, axes = _draw_analysis(
        data,
        analysis,
        kind=kind,
        compact=compact,
        lattice_vectors=_CUBIC_LATTICE if kind == "kpoint" else None,
    )
    figure.set_size_inches(6.0, 5.0)
    figure.canvas.draw()

    for axis, threshold in ((axes.energy, 1.0), (axes.force, 0.05)):
        ticks = _labelled_ticks(axis)
        values = [value for value, _text in ticks]
        assert 0.0 in values
        assert any(value == pytest.approx(threshold) for value in values)
        assert any(value == pytest.approx(-threshold) for value in values)
        beyond = [value for value in values if value > threshold * 1.0001]
        assert len(beyond) >= 2, (axis.get_ylabel(), ticks)
        assert all("e" not in text for _value, text in ticks)
        assert axis.yaxis.get_offset_text().get_text() == ""


def test_band_label_says_the_scale_turns_logarithmic_beyond_it(qapp) -> None:
    data, analysis = _cof2()

    _figure, axes = _draw_analysis(data, analysis)

    energy_legend = [text.get_text() for text in axes.energy_legend.get_texts()]
    assert energy_legend[0] == "Threshold ±1 meV/atom,\nlog scale beyond"


def _threshold_values() -> list[float]:
    # What the energy field holds after typing each value in 1e-5 steps.
    return [float(f"{step * 1e-5:.5f}") for step in range(1, 2000)]


def test_threshold_labels_carry_no_float_noise(qapp) -> None:
    for value in _threshold_values():
        text = threshold_text(value, MEV_PER_EV)
        expected = format((Decimal(repr(value)) * 1000).normalize(), "f")
        assert text == expected, (value, text)
    assert threshold_text(0.00012, MEV_PER_EV) == "0.12"
    assert threshold_text(0.00007, MEV_PER_EV) == "0.07"
    assert threshold_text(0.05) == "0.05"


def test_threshold_label_in_the_legend_has_no_float_noise(qapp) -> None:
    data, analysis = _cof2(0.00012)

    _figure, axes = _draw_analysis(data, analysis)

    energy_legend = [text.get_text() for text in axes.energy_legend.get_texts()]
    assert energy_legend[0] == "Threshold ±0.12 meV/atom,\nlog scale beyond"
    labels = [text for _value, text in _labelled_ticks(axes.energy)]
    assert "0.12" in labels


def _drawn_colour(artist, face: str) -> str:
    """The colour ``artist`` paints over ``face``, its alpha applied."""

    rgba = to_rgba(artist.get_color() if hasattr(artist, "get_color") else artist)
    alpha = rgba[3] * (artist.get_alpha() if artist.get_alpha() is not None else 1.0)
    return theme_module._mixed_colour(to_hex(rgba[:3]), face, alpha)


@pytest.mark.parametrize("mode", [theme_module.THEME_LIGHT, theme_module.THEME_DARK])
@pytest.mark.parametrize("compact", [False, True], ids=["examiner", "card"])
def test_band_edges_and_every_series_reach_three_to_one(qapp, mode, compact) -> None:
    theme_module.apply_theme(qapp, mode)
    try:
        data, analysis = _sweep()
        _figure, axes = _draw_analysis(data, analysis, compact=compact)
        face = to_hex(axes.force.get_facecolor())

        for axis in (axes.energy, axes.force):
            (band,) = [patch for patch in axis.patches if patch.get_label().startswith("Threshold")]
            fill = theme_module._mixed_colour(
                to_hex(band.get_facecolor()[:3]), face, band.get_facecolor()[3]
            )
            edge = theme_module._mixed_colour(
                to_hex(band.get_edgecolor()[:3]), face, band.get_edgecolor()[3]
            )
            assert band.get_linewidth() > 0
            assert theme_module._contrast_ratio(edge, face) >= 3.0
            lines = [
                line
                for candidate in (axis, axes.stress if axis is axes.force else None)
                if candidate is not None
                for line in candidate.get_lines()
            ]
            assert lines
            for line in lines:
                colour = line.get_markeredgecolor() if line.get_linestyle() == "None" else None
                drawn = _drawn_colour(line, face) if colour is None else to_hex(colour)
                assert theme_module._contrast_ratio(drawn, fill) >= 3.0, line.get_label()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_a_card_saved_without_an_atom_count_says_it_shows_the_cell(qapp) -> None:
    raw = {key: value for key, value in _PLOT_DATA.items() if key != "num_atoms"}
    data = ConvergencePlotData.from_mapping(raw)
    figure = Figure(layout="constrained")
    FigureCanvasAgg(figure)

    axes = draw_convergence_plot(figure, data, kind="cutoff", compact=True)

    assert axes.energy.get_ylabel() == "ΔE of the cell (meV)"


def test_redrawing_a_kpoint_plot_warns_nothing(qapp) -> None:
    data = ConvergencePlotData(
        sweep_values=(8.0, 27.0, 64.0, 125.0),
        energies=(-10.0, -10.01, -10.011, -10.0111),
        forces=(0.2, 0.1, 0.04, 0.03),
        num_atoms=2,
    )
    figure = Figure(layout="constrained")
    FigureCanvasAgg(figure)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for _ in range(2):
            draw_convergence_plot(
                figure, data, kind="kpoint", compact=False, lattice_vectors=_CUBIC_LATTICE
            )
            figure.canvas.draw()
        # An overview card clears its plot for a placeholder.
        clear_convergence_plot(figure)
        assert figure.axes == []
