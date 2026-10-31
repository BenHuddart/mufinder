"""The convergence plot the examiner dialog and the pinned overview cards share.

Both draw a sweep's ``plot_data`` mapping (the one the examiner saves and an
overview snapshot carries) with :func:`draw_convergence_plot`: an energy panel
above a force panel, with the force threshold, the recommended value and, when
stress was analysed, a stress axis on the force panel. A k-point sweep puts
the k-point count on a logarithmic axis along the bottom and round k-point
spacings along the top, and names only the recommended grid on the plot.

The examiner draws the full plot (``compact=False``): titles, legends and the
analysis marks it passes in. A pinned card draws it compactly
(``compact=True``): smaller text, with the thresholds left to the card's
``Criteria`` row and the top-right corner left to the card's pop-out button.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

from mufinder.app.theme import (
    plot_series_colours,
    current_palette,
    plot_badge_box,
    style_plot_axes,
    style_plot_legend,
)
from mufinder.science.convergence_analysis import ConvergenceResult
from mufinder.science.kpoint_grids import kpoint_count_for_spacing

#: Round spacings (1/Ang) offered to the k-point spacing axis: each mantissa
#: in each decade. The axis keeps those that fall inside the plotted range.
_SPACING_TICK_MANTISSAS = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)
_SPACING_TICK_DECADES = (-3, -2, -1, 0)

#: Major ticks of the logarithmic k-point count axis: 1, 2 and 5 per decade.
_KPOINT_COUNT_TICK_SUBS = (1.0, 2.0, 5.0)

SPACING_AXIS_LABEL = "K-point spacing (1/Ang)"
_SWEEP_KINDS = ("cutoff", "kpoint")


@dataclass(frozen=True)
class ConvergencePlotData:
    """The points and thresholds of one convergence plot.

    ``sweep_values`` are the cutoffs (eV) or k-point counts. ``stresses`` is
    empty unless every point has a stress and a stress threshold was used.
    ``point_spacings`` maps a k-point count to its grid's spacing.
    """

    sweep_values: tuple[float, ...]
    energies: tuple[float, ...]
    forces: tuple[float, ...]
    energy_threshold: float | None = None
    force_threshold: float | None = None
    stresses: tuple[float, ...] = ()
    stress_threshold: float | None = None
    point_spacings: Mapping[float, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        raw: object,
        *,
        default_energy_threshold: float = 0.001,
        default_force_threshold: float = 0.05,
    ) -> ConvergencePlotData | None:
        """Read a saved ``plot_data`` mapping; ``None`` when it holds no plottable sweep.

        The sweep, energy and force series must be non-empty and of equal
        length. Stresses are kept only when they match the sweep one to one
        and a stress threshold is present.
        """

        if not isinstance(raw, Mapping):
            return None
        try:
            sweep_values = tuple(float(value) for value in raw.get("cutoffs", []))
            energies = tuple(float(value) for value in raw.get("energies", []))
            forces = tuple(float(value) for value in raw.get("forces", []))
            raw_energy_threshold = raw.get("energy_threshold")
            raw_force_threshold = raw.get("force_threshold")
            energy_threshold = float(
                default_energy_threshold if raw_energy_threshold is None else raw_energy_threshold
            )
            force_threshold = float(
                default_force_threshold if raw_force_threshold is None else raw_force_threshold
            )
        except (TypeError, ValueError):
            return None
        if not sweep_values or not (len(sweep_values) == len(energies) == len(forces)):
            return None

        try:
            stresses = tuple(float(value) for value in raw.get("stresses") or [])
            raw_stress_threshold = raw.get("stress_threshold")
            stress_threshold = (
                float(raw_stress_threshold) if raw_stress_threshold is not None else None
            )
        except (TypeError, ValueError):
            stresses, stress_threshold = (), None
        if len(stresses) != len(sweep_values) or stress_threshold is None:
            stresses, stress_threshold = (), None

        point_spacings: dict[float, float] = {}
        raw_spacings = raw.get("point_spacings")
        if isinstance(raw_spacings, Mapping):
            for key, value in raw_spacings.items():
                try:
                    point_spacings[float(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        return cls(
            sweep_values=sweep_values,
            energies=energies,
            forces=forces,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            stresses=stresses,
            stress_threshold=stress_threshold,
            point_spacings=point_spacings,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the ``plot_data`` mapping the examiner saves and a snapshot carries."""

        return {
            "cutoffs": list(self.sweep_values),
            "energies": list(self.energies),
            "forces": list(self.forces),
            "stresses": list(self.stresses),
            "energy_threshold": self.energy_threshold,
            "force_threshold": self.force_threshold,
            "stress_threshold": self.stress_threshold,
            "point_spacings": dict(self.point_spacings),
        }


@dataclass(frozen=True)
class ConvergencePlotAxes:
    """The axes :func:`draw_convergence_plot` created.

    ``spacing`` is the k-point spacing axis along the top of the energy panel
    and ``stress`` the stress axis of the force panel, each ``None`` when not
    drawn.
    """

    energy: Axes
    force: Axes
    spacing: Axes | None = None
    stress: Axes | None = None


def spacing_tick_label(spacing: float) -> str:
    """Return a spacing as the spacing axis labels it (``0.05``, not ``0.0500``)."""

    text = f"{spacing:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _candidate_spacings() -> list[float]:
    candidates = {
        round(mantissa * 10.0**decade, 12)
        for decade in _SPACING_TICK_DECADES
        for mantissa in _SPACING_TICK_MANTISSAS
    }
    return sorted(value for value in candidates if value > 0.0)


def spacing_axis_ticks(
    lattice_vectors: Sequence[Sequence[float]], x_min: float, x_max: float
) -> tuple[list[float], list[str]]:
    """Return the k-point spacing axis ticks between k-point counts ``x_min`` and ``x_max``.

    Each round spacing sits at the k-point count CASTEP's grid for that
    spacing has on this cell (:func:`kpoint_count_for_spacing`). Positions
    read left to right (fewer k-points, larger spacing, first). Spacings that
    land on the same count (every spacing coarser than the cell's Gamma-only
    limit gives one k-point) keep only the first, largest one.
    """

    if x_max <= x_min:
        return [], []
    ticks: list[tuple[float, str]] = []
    for spacing in reversed(_candidate_spacings()):
        try:
            position = float(kpoint_count_for_spacing(lattice_vectors, spacing))
        except ValueError:
            continue
        if not x_min <= position <= x_max:
            continue
        if ticks and position <= ticks[-1][0]:
            continue
        ticks.append((position, spacing_tick_label(spacing)))
    return [position for position, _ in ticks], [label for _, label in ticks]


def _kpoint_count_label(value: float, _position: object) -> str:
    return f"{value:g}"


def _use_kpoint_count_axis(axis: Axes) -> None:
    """Put the k-point count on a log axis with 1-2-5 ticks and no minor ticks."""

    axis.set_xscale("log")
    # set_xscale only requests a rescale; autoscale now so the limits are
    # positive before a twin axis copies them.
    axis.relim()
    axis.autoscale_view()
    axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=_KPOINT_COUNT_TICK_SUBS))
    axis.xaxis.set_major_formatter(FuncFormatter(_kpoint_count_label))
    axis.xaxis.set_minor_locator(NullLocator())


def _clear(figure: Figure) -> None:
    # Clearing resets each axis to (0, 1), which a log axis rejects with a
    # warning; return log axes to linear first.
    for axis in figure.axes:
        if axis.get_xscale() != "linear":
            axis.set_xscale("linear")
    figure.clear()


def draw_convergence_plot(
    figure: Figure,
    data: ConvergencePlotData,
    *,
    kind: str,
    compact: bool,
    recommended: float | None = None,
    recommended_label: str | None = None,
    lattice_vectors: Sequence[Sequence[float]] | None = None,
    energy_result: ConvergenceResult | None = None,
    force_result: ConvergenceResult | None = None,
    stress_result: ConvergenceResult | None = None,
) -> ConvergencePlotAxes:
    """Clear ``figure`` and draw ``data`` on it; the caller draws the canvas.

    ``kind`` is ``"cutoff"`` or ``"kpoint"``. ``recommended`` is the
    recommended sweep value, marked on both panels; on a k-point plot
    ``recommended_label`` names its grid, beside the mark on a compact plot
    and in the energy legend otherwise. ``lattice_vectors``
    (k-point plots) adds the spacing axis. The results add the power-law fit,
    its limit and the value where each criterion converged.
    """

    if kind not in _SWEEP_KINDS:
        raise ValueError(f"kind must be one of {_SWEEP_KINDS}, not {kind!r}.")
    palette = current_palette()
    colors = plot_series_colours()
    label_size = 9 if compact else None
    tick_size = 8 if compact else None
    axes_face = palette.bg_surface if compact else palette.bg_surface_alt
    x_values = list(data.sweep_values)

    figure.set_facecolor(palette.bg_surface_alt if compact else palette.bg_surface)
    _clear(figure)
    energy_ax = figure.add_subplot(2, 1, 1)
    force_ax = figure.add_subplot(2, 1, 2, sharex=energy_ax)

    # Energy panel.
    energy_ax.plot(
        x_values,
        data.energies,
        marker="o",
        linestyle="-",
        color=colors.energy,
        label="Total energy",
    )
    if energy_result is not None and energy_result.fitted_curve:
        energy_ax.plot(
            [point[0] for point in energy_result.fitted_curve],
            [point[1] for point in energy_result.fitted_curve],
            linestyle="--",
            color=colors.fit,
            label="Power-law fit",
        )
    if energy_result is not None and energy_result.extrapolated_energy is not None:
        energy_ax.axhline(
            energy_result.extrapolated_energy, color=colors.fit, linestyle=":", label="Fitted limit"
        )
    if energy_result is not None and energy_result.converged_value is not None:
        energy_ax.axvline(
            energy_result.converged_value,
            color=colors.converged,
            linestyle="--",
            label="Energy convergence",
        )
    name_grid = kind == "kpoint" and recommended is not None and bool(recommended_label)
    if recommended is not None:
        energy_ax.axvline(
            recommended,
            color=colors.reference,
            linestyle="-",
            alpha=0.55,
            label=f"Recommended {recommended_label}" if name_grid and not compact else None,
        )
    energy_ax.set_ylabel("Energy (eV)", fontsize=label_size)
    energy_ax.tick_params(axis="x", labelbottom=False)
    style_plot_axes(energy_ax, facecolor=axes_face, tick_labelsize=tick_size)

    spacing_ax: Axes | None = None
    if kind == "kpoint" and x_values:
        _use_kpoint_count_axis(energy_ax)
        if lattice_vectors is not None:
            x_min, x_max = energy_ax.get_xlim()
            positions, labels = spacing_axis_ticks(lattice_vectors, x_min, x_max)
            if positions:
                spacing_ax = energy_ax.twiny()
                spacing_ax.set_xscale("log")
                spacing_ax.set_xlim(x_min, x_max)
                # Keep the spacing ticks over their counts when the plot is zoomed or panned.
                energy_ax.callbacks.connect(
                    "xlim_changed",
                    lambda axis, twin=spacing_ax: twin.set_xlim(axis.get_xlim()),
                )
                spacing_ax.set_xticks(positions, labels)
                spacing_ax.xaxis.set_minor_locator(NullLocator())
                # On a compact plot the label's row is as tall as the pop-out button
                # in the top-right corner, so a tick label at the right stays clear of it.
                spacing_ax.set_xlabel(
                    SPACING_AXIS_LABEL, labelpad=14 if compact else 6, fontsize=label_size
                )
                style_plot_axes(spacing_ax, facecolor=axes_face, tick_labelsize=tick_size)
                # The twin would repeat the energy axis's grid at its own ticks.
                spacing_ax.grid(False)
                # The energy offset text (``-5.621e3``) would sit on the spacing labels.
                energy_ax.ticklabel_format(axis="y", useOffset=False)

    # Force panel.
    force_handles: list[Any] = [
        force_ax.plot(
            x_values, data.forces, marker="o", linestyle="-", color=colors.force, label="Max force"
        )[0]
    ]
    if force_result is not None and force_result.converged_value is not None:
        force_handles.append(
            force_ax.axvline(
                force_result.converged_value,
                color=colors.converged,
                linestyle="--",
                label="Force convergence",
            )
        )
    if data.force_threshold is not None:
        force_handles.append(
            force_ax.axhline(
                data.force_threshold,
                color=colors.threshold,
                linestyle=":",
                alpha=0.85 if compact else 1.0,
                label="Force threshold",
            )
        )
    if recommended is not None:
        force_ax.axvline(recommended, color=colors.reference, linestyle="-", alpha=0.55)
    force_ax.set_xlabel(
        "Cutoff energy (eV)" if kind == "cutoff" else "Number of k-points",
        labelpad=6 if compact else 8,
        fontsize=label_size,
    )
    force_ax.set_ylabel("Max force\n(eV/Ang)", fontsize=label_size)
    style_plot_axes(force_ax, facecolor=axes_face, tick_labelsize=tick_size)
    if not compact:
        force_ax.yaxis.label.set_color(colors.force)
        force_ax.tick_params(axis="y", colors=colors.force, labelcolor=colors.force)

    stress_ax: Axes | None = None
    stress_handles: list[Any] = []
    if data.stresses:
        stress_ax = force_ax.twinx()
        stress_handles.append(
            stress_ax.plot(
                x_values,
                data.stresses,
                marker="s",
                linestyle="-",
                color=colors.stress,
                label="Max stress",
            )[0]
        )
        if stress_result is not None and stress_result.converged_value is not None:
            stress_handles.append(
                stress_ax.axvline(
                    stress_result.converged_value,
                    color=colors.stress_converged,
                    linestyle="--",
                    alpha=0.85,
                    label="Stress convergence",
                )
            )
        if data.stress_threshold is not None:
            stress_handles.append(
                stress_ax.axhline(
                    data.stress_threshold,
                    color=colors.stress,
                    linestyle=":",
                    alpha=0.85,
                    label="Stress threshold",
                )
            )
        stress_ax.set_ylabel("Max stress\n(GPa)", fontsize=label_size)
        stress_ax.yaxis.label.set_color(colors.stress)
        stress_ax.tick_params(
            axis="y", colors=colors.stress, labelcolor=colors.stress, labelsize=tick_size
        )
        for spine in stress_ax.spines.values():
            spine.set_color(palette.border)
        stress_ax.spines["right"].set_color(colors.stress)

    if name_grid and compact:
        energy_ax.annotate(
            recommended_label,
            xy=(recommended, 1.0),
            xycoords=energy_ax.get_xaxis_transform(),
            # Without a spacing axis above it, the panel's top reaches the pop-out button.
            xytext=(4, -4 if spacing_ax is not None else -24),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color=palette.text_primary,
            bbox=plot_badge_box(),
        )

    if not compact and x_values:
        # Small columns keep a full legend inside a panel at the examiner's minimum size.
        energy_ax.legend(loc="best", ncols=2, fontsize="small")
        style_plot_legend(energy_ax.get_legend(), frame_alpha=0.95)
        # Filled by column, so interleaving gives a force row above a stress row.
        handles = [
            handle
            for pair in zip_longest(force_handles, stress_handles)
            for handle in pair
            if handle is not None
        ]
        # The stress axis draws over the force axis, so it holds the legend when present.
        legend_ax = stress_ax if stress_ax is not None else force_ax
        legend_ax.legend(
            handles,
            [handle.get_label() for handle in handles],
            loc="best",
            ncols=3 if stress_handles else 2,
            fontsize="small",
        )
        style_plot_legend(legend_ax.get_legend(), frame_alpha=0.95)
    if not compact:
        _draw_titles(energy_ax, force_ax, data)

    return ConvergencePlotAxes(
        energy=energy_ax, force=force_ax, spacing=spacing_ax, stress=stress_ax
    )


def _draw_titles(energy_ax: Axes, force_ax: Axes, data: ConvergencePlotData) -> None:
    """Title both panels, before any analysis too: the top row keeps the panel off the pop-out button."""

    energy_title = "Energy convergence"
    if data.energy_threshold is not None:
        energy_title += f" (threshold {data.energy_threshold:.3g} eV/atom)"
    energy_ax.set_title(energy_title, pad=12)
    if data.force_threshold is None:
        force_ax.set_title("Force convergence", pad=10)
        return
    if data.stresses and data.stress_threshold is not None:
        force_ax.set_title(
            "Force / stress convergence "
            f"(thresholds {data.force_threshold:.3g} eV/Ang, {data.stress_threshold:.3g} GPa)",
            pad=10,
        )
    else:
        force_ax.set_title(
            f"Force convergence (threshold {data.force_threshold:.3g} eV/Ang)", pad=10
        )
