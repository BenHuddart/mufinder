"""The convergence plot the examiner dialog and the pinned overview cards share.

Both draw a sweep's ``plot_data`` mapping (the one the examiner saves and an
overview snapshot carries) with :func:`draw_convergence_plot`: an energy panel
above a force panel, each showing every point's distance from the finest point
of the sweep with the threshold as a band about zero, the recommended value
as one vertical line and, when stress was analysed, a stress axis on the force
panel that shares the force band. A k-point sweep puts the k-point count on a
logarithmic axis along the bottom and round k-point spacings along the top.

The examiner draws the full plot (``compact=False``): titles, the analysis
marks it passes in, and legends outside the panels at the right. A pinned card
draws it compactly (``compact=True``): smaller text and no legends, with the
thresholds named in the card's ``Criteria`` row, the adopted grid named beside
its line and the top-right corner left to the card's pop-out button.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
import math
from typing import Any

from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.text import Text
from matplotlib.transforms import Bbox, BboxBase, IdentityTransform, ScaledTranslation
from matplotlib.ticker import FuncFormatter, Locator, LogLocator, NullLocator
import numpy as np

from mufinder.app.theme import (
    plot_series_colours,
    current_palette,
    plot_badge_box,
    plot_threshold_band,
    style_plot_axes,
    style_plot_legend,
)
from mufinder.domain.constants import MEV_PER_EV
from mufinder.science.convergence_analysis import (
    ConvergenceResult,
    offset_from_reference,
    reference_offsets,
)
from mufinder.science.kpoint_grids import kpoint_count_for_spacing

#: Round spacings (1/Ang) offered to the k-point spacing axis: each mantissa
#: in each decade. The axis keeps those that fall inside the plotted range.
_SPACING_TICK_MANTISSAS = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)
_SPACING_TICK_DECADES = (-3, -2, -1, 0)

#: Major ticks of the logarithmic k-point count axis: 1, 2 and 5 per decade.
_KPOINT_COUNT_TICK_SUBS = (1.0, 2.0, 5.0)

SPACING_AXIS_LABEL = "K-point spacing (1/Ang)"

# A relative panel shows at least this many thresholds either side of zero,
# pads its range by this fraction in its symmetric-log frame, and labels a
# decade only beyond this many thresholds, at most this many decades.
_BAND_HEADROOM = 1.5
_OFFSET_MARGIN = 0.06
# Tick mantissas beyond a band, coarsest first; a tick must sit this factor
# above the threshold, and tick labels stay this many font heights apart.
_BEYOND_BAND_MANTISSAS = ((1.0,), (2.0, 5.0), (3.0, 4.0, 6.0, 7.0, 8.0, 9.0))
_ALL_MANTISSAS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
# The band is at least this share of the decades the data spans beyond it.
_BAND_SHARE_OF_DECADES = 1.0 / 3.0
_ABOVE_THRESHOLD = 1.0001
_TICK_LABEL_SPACING = 1.2
# The fitted limit is drawn heavier than a gridline and above it; the band
# sits under every series.
_LIMIT_LINE_WIDTH = 1.8
_LIMIT_LINE_ZORDER = 2.5
# The gap between a line label and the panel edge and line, in points.
_LINE_LABEL_PAD_POINTS = 4.0
_BAND_ZORDER = 0.8
# The gap between a panel's right-hand axis and its legend, in points.
_LEGEND_GAP_POINTS = 6.0
# The ring round the point where a criterion converged, in points.
_RING_SIZE = 13
_SWEEP_KINDS = ("cutoff", "kpoint")


@dataclass(frozen=True)
class ConvergencePlotData:
    """The points and thresholds of one convergence plot.

    ``sweep_values`` are the cutoffs (eV) or k-point counts. ``stresses`` is
    empty unless every point has a stress and a stress threshold was used.
    ``point_spacings`` maps a k-point count to its grid's spacing.
    ``num_atoms`` is the atom count of the swept cell, which puts the energy
    panel per atom; without it the panel shows the energy of the cell.
    """

    sweep_values: tuple[float, ...]
    energies: tuple[float, ...]
    forces: tuple[float, ...]
    energy_threshold: float | None = None
    force_threshold: float | None = None
    stresses: tuple[float, ...] = ()
    stress_threshold: float | None = None
    point_spacings: Mapping[float, float] = field(default_factory=dict)
    num_atoms: int | None = None

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

        num_atoms: int | None = None
        try:
            raw_num_atoms = raw.get("num_atoms")
            if raw_num_atoms is not None and not isinstance(raw_num_atoms, bool):
                count = float(raw_num_atoms)
                if count.is_integer() and count > 0:
                    num_atoms = int(count)
        except (TypeError, ValueError, OverflowError):
            num_atoms = None

        return cls(
            sweep_values=sweep_values,
            energies=energies,
            forces=forces,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            stresses=stresses,
            stress_threshold=stress_threshold,
            point_spacings=point_spacings,
            num_atoms=num_atoms,
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
            "num_atoms": self.num_atoms,
        }


@dataclass(frozen=True)
class ConvergencePlotAxes:
    """The axes :func:`draw_convergence_plot` created.

    ``spacing`` is the k-point spacing axis along the top of the energy panel
    and ``stress`` the stress axis of the force panel, each ``None`` when not
    drawn. The legends are the full plot's, ``None`` on a compact plot. The
    offsets map each sweep value to its plotted distance from the finest point
    (energy in the panel's meV unit).
    """

    energy: Axes
    force: Axes
    spacing: Axes | None = None
    stress: Axes | None = None
    energy_legend: Legend | None = None
    force_legend: Legend | None = None
    energy_offsets: Mapping[float, float] = field(default_factory=dict)
    force_offsets: Mapping[float, float] = field(default_factory=dict)
    stress_offsets: Mapping[float, float] = field(default_factory=dict)


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


def clear_convergence_plot(figure: Figure) -> None:
    """Clear a figure the convergence plot drew on, without a warning from its log axes."""

    # Clearing resets each axis to (0, 1), which a log axis rejects with a
    # warning; return log axes to linear first.
    for axis in figure.axes:
        if axis.get_xscale() != "linear":
            axis.set_xscale("linear")
    figure.clear()


def offset_tick_label(value: float, _position: object = None) -> str:
    """Return a tick of a relative panel as plain text: ``0.0005``, ``10``, never an exponent."""

    if value == 0.0:
        return "0"
    text = np.format_float_positional(
        float(value), precision=3, unique=True, fractional=False, trim="-"
    )
    return text.replace("-", "\N{MINUS SIGN}")


def threshold_text(value: float, factor: float = 1.0) -> str:
    """Return ``value * factor`` as typed: ``0.00012`` eV/atom times 1000 is ``0.12``.

    The product is taken in decimal arithmetic on the shortest text of each
    float, so a unit conversion adds no binary noise (``0.12000000000000001``).
    """

    product = Decimal(repr(float(value))) * Decimal(repr(float(factor)))
    text = format(product.normalize(), "f")
    return text.replace("-", "\N{MINUS SIGN}")


def _round_values(threshold: float, edge: float, mantissas: Sequence[float]) -> list[float]:
    """Each ``mantissa`` times a power of ten in ``(threshold, edge]``, ascending."""

    if edge <= threshold:
        return []
    low = math.floor(math.log10(threshold))
    high = math.ceil(math.log10(edge))
    return sorted(
        value
        for decade in range(low, high + 1)
        for mantissa in mantissas
        if threshold * _ABOVE_THRESHOLD < (value := round(mantissa * 10.0**decade, 12)) <= edge
    )


@dataclass(frozen=True)
class OffsetTicks:
    """The ticks of a relative panel: ``values`` in order, ``unlabelled`` the ones left bare."""

    values: tuple[float, ...]
    unlabelled: frozenset[float] = frozenset()


def offset_ticks(
    threshold: float,
    lower: float,
    upper: float,
    *,
    position: Any = None,
    spacing: float = 0.0,
    reach: tuple[float, float] | None = None,
) -> OffsetTicks:
    """Ticks of a relative panel between ``lower`` and ``upper``, in priority order.

    1. No two labels overlap: a label is kept only when its height (from
       ``position``, a value's height in pixels) is ``spacing`` or more from
       every label kept before it. Without ``position`` nothing is thinned.
    2. Zero and the positive threshold are always labelled. The negative
       threshold is labelled when it fits; otherwise its tick stays bare.
    3. Each side the data passes the band on (``reach`` is the lowest and
       highest plotted value) gets one labelled tick at or beyond the decade
       of its outermost value: the nearest round value (a whole multiple of a
       power of ten) at or above that value, or failing that the nearest
       below it within its decade, whichever fits first.
    4. A second labelled tick beyond the band on that side wherever one fits,
       the coarsest first: whole decades, then 2 and 5 times a decade, then
       every multiple of a decade.
    """

    labelled: list[float] = []
    unlabelled: set[float] = set()

    def fits(value: float) -> bool:
        if position is None:
            return True
        height = float(position(value))
        return all(abs(height - float(position(other))) >= spacing for other in labelled)

    for value in (0.0, threshold):
        if lower <= value <= upper:
            labelled.append(value)
    if lower <= -threshold <= upper:
        if fits(-threshold):
            labelled.append(-threshold)
        else:
            unlabelled.add(-threshold)

    low_reach, high_reach = reach if reach is not None else (lower, upper)
    for sign, edge, extent in ((1.0, upper, high_reach), (-1.0, -lower, -low_reach)):
        if extent <= threshold * _ABOVE_THRESHOLD:
            continue
        everything = _round_values(threshold, edge, _ALL_MANTISSAS)
        decade = 10.0 ** math.floor(math.log10(extent))
        above = [value for value in everything if value >= extent]
        below = [value for value in reversed(everything) if decade <= value < extent]
        beyond = 0
        for magnitude in above + below:
            if fits(sign * magnitude):
                labelled.append(sign * magnitude)
                beyond += 1
                break
        for mantissas in _BEYOND_BAND_MANTISSAS:
            if beyond >= 2:
                break
            for magnitude in _round_values(threshold, edge, mantissas):
                value = sign * magnitude
                if value not in labelled and fits(value):
                    labelled.append(value)
                    beyond += 1
    return OffsetTicks(tuple(sorted({*labelled, *unlabelled})), frozenset(unlabelled))


class _OffsetLocator(Locator):
    """Ticks of a relative panel, thinned at draw time to what its height can label.

    ``unlabelled`` holds the ticks of the last call whose labels did not fit,
    which the panel's formatter leaves blank.
    """

    def __init__(self, threshold: float, reach: tuple[float, float]) -> None:
        self._threshold = threshold
        self._reach = reach
        self.unlabelled: frozenset[float] = frozenset()

    def __call__(self) -> list[float]:
        lower, upper = sorted(self.axis.get_view_interval())
        return self.tick_values(lower, upper)

    def tick_values(self, vmin: float, vmax: float) -> list[float]:
        lower, upper = sorted((float(vmin), float(vmax)))
        axes = self.axis.axes
        transform = self.axis.get_transform()
        bottom, top = transform.transform([lower, upper])
        height = axes.bbox.height
        if not (height > 0 and top > bottom):
            ticks = offset_ticks(self._threshold, lower, upper, reach=self._reach)
        else:

            def position(value: float) -> float:
                return (float(transform.transform([value])[0]) - bottom) / (top - bottom) * height

            font_points = self.axis.majorTicks[0].label1.get_fontsize()
            spacing = font_points * axes.figure.dpi / 72.0 * _TICK_LABEL_SPACING
            ticks = offset_ticks(
                self._threshold,
                lower,
                upper,
                position=position,
                spacing=spacing,
                reach=self._reach,
            )
        self.unlabelled = ticks.unlabelled
        return list(ticks.values)


def _band_linscale(normalised: Sequence[float]) -> float:
    """The band's height in decades either side of zero.

    One decade, or a third of the decades the data spans beyond the band, so a
    sweep that reaches many decades past a small threshold still leaves the
    band tall enough to label zero and the threshold apart.
    """

    reach = max((abs(value) for value in normalised), default=1.0)
    decades = math.log10(reach) if reach > 1.0 else 0.0
    return max(1.0, decades * _BAND_SHARE_OF_DECADES)


def _shared_offset_limits(
    series: Sequence[tuple[Sequence[float], float]],
) -> tuple[float, float, float]:
    """Return the panel's limits in units of each series' threshold, and its band height.

    Each series is a list of offsets and its threshold. The limits hold every
    offset and the threshold band, padded in the panel's symmetric-log frame,
    so axes that share them (and the band height, ``linscale``) place every
    threshold at the same height.
    """

    normalised = [
        float(value) / threshold
        for values, threshold in series
        for value in values
        if math.isfinite(float(value))
    ]
    linscale = _band_linscale(normalised)
    lower = min([-_BAND_HEADROOM, *normalised])
    upper = max([_BAND_HEADROOM, *normalised])
    transform = SymmetricalLogTransform(10.0, 1.0, linscale)
    low, high = transform.transform(np.array([lower, upper]))
    pad = _OFFSET_MARGIN * (high - low)
    padded = transform.inverted().transform(np.array([low - pad, high + pad]))
    return float(padded[0]), float(padded[1]), linscale


def _use_offset_axis(
    axis: Axes,
    threshold: float,
    limits: tuple[float, float, float],
    threshold_label: str,
    values: Sequence[float],
) -> None:
    """Draw ``axis`` linear within ``threshold`` of zero and logarithmic beyond it.

    ``limits`` are the lower and upper limits in units of the threshold and the
    band height (see :func:`_shared_offset_limits`). The ticks at the threshold
    read ``threshold_label``, the value as typed; ``values`` are the plotted
    values, which decide the sides that get ticks beyond the band.
    """

    lower, upper = limits[0] * threshold, limits[1] * threshold
    axis.set_yscale("symlog", base=10, linthresh=threshold, linscale=limits[2])
    axis.set_ylim(lower, upper)
    finite = [float(value) for value in values if math.isfinite(float(value))]
    reach = (min(finite, default=0.0), max(finite, default=0.0))
    locator = _OffsetLocator(threshold, reach)
    axis.yaxis.set_major_locator(locator)

    def label(value: float, _position: object = None) -> str:
        if value in locator.unlabelled:
            return ""
        if math.isclose(abs(value), threshold, rel_tol=1e-9):
            return threshold_label if value > 0 else f"\N{MINUS SIGN}{threshold_label}"
        return offset_tick_label(value)

    axis.yaxis.set_major_formatter(FuncFormatter(label))
    axis.yaxis.set_minor_locator(NullLocator())


def _draw_band(axis: Axes, threshold: float, label: str) -> Any:
    """Shade ``-threshold`` to ``threshold`` with a soft fill and edges that mark the change of scale."""

    style = plot_threshold_band()
    return axis.axhspan(
        -threshold,
        threshold,
        facecolor=to_rgba(style.edge, style.fill_alpha),
        edgecolor=style.edge,
        linewidth=style.edge_width,
        zorder=_BAND_ZORDER,
        label=label,
    )


def _gap_text(value: float) -> str:
    """A gap to two significant figures, as the results text rounds its amounts."""

    return np.format_float_positional(
        float(value), precision=2, unique=True, fractional=False, trim="-"
    )


def _follow_count_axis(axis: Axes, twin: Axes) -> None:
    """Keep the spacing axis over the count axis; a non-positive range is not a log range."""

    low, high = axis.get_xlim()
    if low > 0 and high > 0:
        twin.set_xlim(low, high)


def _usable_threshold(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0.0


def _use_plain_axis(axis: Axes) -> None:
    """A linear panel with no offset text, for a series without a threshold."""

    axis.ticklabel_format(axis="y", useOffset=False)


def _offsets_on(
    x_values: Sequence[float], values: Sequence[float], *, scale: float, factor: float = 1.0
) -> tuple[dict[float, float], float | None]:
    """Return each sweep value's offset from the finest point times ``factor``, and the reference.

    The offsets are the criterion's (:func:`reference_offsets`); a sweep value
    without a finite value is left out.
    """

    signed = reference_offsets(x_values, values, scale=scale)
    if signed is None:
        return {}, None
    return {point: offset * factor for point, offset in signed.offsets}, signed.reference_value


def _series(x_values: Sequence[float], offsets: Mapping[float, float]) -> list[float]:
    """The offsets in sweep order, with a gap where a point has no value."""

    return [offsets.get(float(value), math.nan) for value in x_values]


def _default_point_label(kind: str, value: float) -> str:
    if kind == "cutoff":
        return f"{value:.8g} eV"
    return f"{value:g} k-points"


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
    point_labels: Mapping[float, str] | None = None,
) -> ConvergencePlotAxes:
    """Clear ``figure`` and draw ``data`` on it; the caller draws the canvas.

    Every panel plots each point's distance from the finest point of the sweep,
    the distance the criterion compares with its threshold: the energy per atom
    in meV/atom (in meV for the cell when ``data`` has no atom count), the max
    force in eV/Ang and the max stress in GPa. A panel with a threshold shades
    the band within it about zero and is linear inside the band and
    logarithmic beyond it; the stress axis shares the force band.

    ``kind`` is ``"cutoff"`` or ``"kpoint"``. ``recommended`` is the
    recommended sweep value, drawn as one vertical line through both panels;
    on a k-point plot ``recommended_label`` names its grid, beside the line on
    a compact plot. ``lattice_vectors`` (k-point plots) adds the spacing axis.
    The results add the power-law fit and its limit, and ring the point at
    which each criterion converged on its own series. ``point_labels`` names
    sweep values in the full plot's legends (a cutoff in eV, otherwise).
    The full plot puts its legends outside the panels, at the right.
    """

    if kind not in _SWEEP_KINDS:
        raise ValueError(f"kind must be one of {_SWEEP_KINDS}, not {kind!r}.")
    palette = current_palette()
    colors = plot_series_colours()
    label_size = 9 if compact else None
    tick_size = 8 if compact else None
    axes_face = palette.bg_surface if compact else palette.bg_surface_alt
    x_values = list(data.sweep_values)
    labels = {float(key): value for key, value in (point_labels or {}).items()}

    def name(value: float) -> str:
        if kind == "kpoint" and recommended is not None and float(value) == float(recommended):
            if recommended_label:
                return recommended_label
        return labels.get(float(value)) or _default_point_label(kind, float(value))

    figure.set_facecolor(palette.bg_surface_alt if compact else palette.bg_surface)
    clear_convergence_plot(figure)
    energy_ax = figure.add_subplot(2, 1, 1)
    force_ax = figure.add_subplot(2, 1, 2, sharex=energy_ax)

    # Energy panel: per atom in meV/atom, or for the cell in meV without an atom count.
    per_atom = data.num_atoms is not None and data.num_atoms > 0
    energy_scale = float(data.num_atoms) if per_atom else 1.0
    energy_unit = "meV/atom" if per_atom else "meV"
    energy_label_unit = energy_unit if per_atom else "of the cell (meV)"
    energy_offsets, energy_reference = _offsets_on(
        x_values, data.energies, scale=energy_scale, factor=MEV_PER_EV
    )
    energy_threshold = (
        data.energy_threshold * MEV_PER_EV
        if per_atom and _usable_threshold(data.energy_threshold)
        else None
    )
    energy_threshold_text = (
        threshold_text(data.energy_threshold, MEV_PER_EV) if energy_threshold is not None else ""
    )
    energy_handles: list[Any] = []
    if energy_threshold is not None:
        energy_handles.append(
            _draw_band(
                energy_ax,
                energy_threshold,
                f"Threshold \N{PLUS-MINUS SIGN}{energy_threshold_text} {energy_unit},\n"
                "log scale beyond",
            )
        )
    energy_handles.append(
        energy_ax.plot(
            x_values,
            _series(x_values, energy_offsets),
            marker="o",
            linestyle="-",
            color=colors.energy,
            label="Energy",
        )[0]
    )
    fit_values: list[float] = []
    if energy_reference is not None and energy_result is not None:
        if energy_result.fitted_curve:
            curve_x = [point[0] for point in energy_result.fitted_curve]
            curve_y = [
                offset_from_reference(point[1], energy_reference, energy_scale) * MEV_PER_EV
                for point in energy_result.fitted_curve
            ]
            fit_values.extend(curve_y)
            energy_handles.append(
                energy_ax.plot(
                    curve_x, curve_y, linestyle="--", color=colors.fit, label="Power-law fit"
                )[0]
            )
        if energy_result.extrapolated_energy is not None:
            limit = (
                offset_from_reference(
                    energy_result.extrapolated_energy, energy_reference, energy_scale
                )
                * MEV_PER_EV
            )
            fit_values.append(limit)
            label = "Fitted limit"
            if energy_result.fitted_curve:
                # The sweep-length line's number: the fit's height above the limit
                # at the finest cutoff, the gap between the dashed and dotted lines.
                fit_end_x = energy_result.fitted_curve[-1][0]
                gap = curve_y[-1] - limit
                label = (
                    f"Fitted limit, {_gap_text(gap)} {energy_unit}\n"
                    f"below the fitted curve at {name(fit_end_x)}"
                )
            energy_handles.append(
                energy_ax.axhline(
                    limit,
                    color=colors.fit,
                    linestyle=":",
                    linewidth=_LIMIT_LINE_WIDTH,
                    zorder=_LIMIT_LINE_ZORDER,
                    label=label,
                )
            )
            # Named on the plot, so the line reads apart from a gridline it may sit on;
            # placed at draw time where it covers no point, tick label or legend.
            energy_ax.add_artist(
                _FreeLineLabel(
                    energy_ax,
                    limit,
                    "E\N{INFINITY}",
                    fontsize=8 if compact else "small",
                    color=colors.fit,
                )
            )
    energy_handles.extend(
        _ring_converged(energy_ax, energy_result, energy_offsets, colors.converged, "Energy", name)
    )
    energy_ax.set_ylabel(
        f"\N{GREEK CAPITAL LETTER DELTA}E ({energy_unit})"
        if per_atom
        else f"\N{GREEK CAPITAL LETTER DELTA}E {energy_label_unit}",
        fontsize=label_size,
    )
    energy_ax.tick_params(axis="x", labelbottom=False)
    style_plot_axes(energy_ax, facecolor=axes_face, tick_labelsize=tick_size)
    if energy_threshold is not None:
        _use_offset_axis(
            energy_ax,
            energy_threshold,
            _shared_offset_limits([([*energy_offsets.values(), *fit_values], energy_threshold)]),
            energy_threshold_text,
            [*energy_offsets.values(), *fit_values],
        )
    else:
        _use_plain_axis(energy_ax)

    spacing_ax: Axes | None = None
    if kind == "kpoint" and x_values:
        _use_kpoint_count_axis(energy_ax)
        if lattice_vectors is not None:
            x_min, x_max = energy_ax.get_xlim()
            positions, tick_labels = spacing_axis_ticks(lattice_vectors, x_min, x_max)
            if positions:
                spacing_ax = energy_ax.twiny()
                spacing_ax.set_xscale("log")
                spacing_ax.set_xlim(x_min, x_max)
                # Keep the spacing ticks over their counts when the plot is zoomed or panned.
                energy_ax.callbacks.connect(
                    "xlim_changed",
                    lambda axis, twin=spacing_ax: _follow_count_axis(axis, twin),
                )
                spacing_ax.set_xticks(positions, tick_labels)
                spacing_ax.xaxis.set_minor_locator(NullLocator())
                # On a compact plot the label's row is as tall as the pop-out button
                # in the top-right corner, so a tick label at the right stays clear of it.
                spacing_ax.set_xlabel(
                    SPACING_AXIS_LABEL, labelpad=14 if compact else 6, fontsize=label_size
                )
                style_plot_axes(spacing_ax, facecolor=axes_face, tick_labelsize=tick_size)
                # The twin would repeat the energy axis's grid at its own ticks.
                spacing_ax.grid(False)

    # Force panel, with stress on a second axis that shares the force band.
    force_offsets, _ = _offsets_on(x_values, data.forces, scale=1.0)
    stress_offsets: dict[float, float] = {}
    if data.stresses:
        stress_offsets, _ = _offsets_on(x_values, data.stresses, scale=1.0)
    force_threshold = data.force_threshold if _usable_threshold(data.force_threshold) else None
    stress_threshold = (
        data.stress_threshold
        if data.stresses and _usable_threshold(data.stress_threshold)
        else None
    )
    force_handles: list[Any] = []
    if force_threshold is not None:
        band_label = (
            f"Threshold \N{PLUS-MINUS SIGN}{threshold_text(force_threshold)} eV/Ang,\n"
            "log scale beyond"
        )
        if stress_threshold is not None:
            band_label = (
                f"Thresholds \N{PLUS-MINUS SIGN}{threshold_text(force_threshold)} eV/Ang,\n"
                f"\N{PLUS-MINUS SIGN}{threshold_text(stress_threshold)} GPa, log scale beyond"
            )
        force_handles.append(_draw_band(force_ax, force_threshold, band_label))
    force_handles.append(
        force_ax.plot(
            x_values,
            _series(x_values, force_offsets),
            marker="o",
            linestyle="-",
            color=colors.force,
            label="Max force",
        )[0]
    )
    force_handles.extend(
        _ring_converged(force_ax, force_result, force_offsets, colors.converged, "Force", name)
    )
    force_ax.set_xlabel(
        "Cutoff energy (eV)" if kind == "cutoff" else "Number of k-points",
        labelpad=6 if compact else 8,
        fontsize=label_size,
    )
    force_ax.set_ylabel("\N{GREEK CAPITAL LETTER DELTA} max force\n(eV/Ang)", fontsize=label_size)
    style_plot_axes(force_ax, facecolor=axes_face, tick_labelsize=tick_size)
    if not compact:
        force_ax.yaxis.label.set_color(colors.force)
        force_ax.tick_params(axis="y", colors=colors.force, labelcolor=colors.force)

    stress_ax: Axes | None = None
    if data.stresses:
        stress_ax = force_ax.twinx()
        force_handles.append(
            stress_ax.plot(
                x_values,
                _series(x_values, stress_offsets),
                marker="s",
                linestyle="-",
                color=colors.stress,
                label="Max stress",
            )[0]
        )
        force_handles.extend(
            _ring_converged(
                stress_ax, stress_result, stress_offsets, colors.stress_converged, "Stress", name
            )
        )
        stress_ax.set_ylabel(
            "\N{GREEK CAPITAL LETTER DELTA} max stress\n(GPa)", fontsize=label_size
        )
        stress_ax.yaxis.label.set_color(colors.stress)
        stress_ax.tick_params(
            axis="y", colors=colors.stress, labelcolor=colors.stress, labelsize=tick_size
        )
        for spine in stress_ax.spines.values():
            spine.set_color(palette.border)
        stress_ax.spines["right"].set_color(colors.stress)

    if force_threshold is not None:
        shared = [(list(force_offsets.values()), force_threshold)]
        if stress_ax is not None and stress_threshold is not None:
            shared.append((list(stress_offsets.values()), stress_threshold))
        limits = _shared_offset_limits(shared)
        _use_offset_axis(
            force_ax,
            force_threshold,
            limits,
            threshold_text(force_threshold),
            list(force_offsets.values()),
        )
        if stress_ax is not None and stress_threshold is not None:
            _use_offset_axis(
                stress_ax,
                stress_threshold,
                limits,
                threshold_text(stress_threshold),
                list(stress_offsets.values()),
            )
        elif stress_ax is not None:
            _use_plain_axis(stress_ax)
    else:
        _use_plain_axis(force_ax)
        if stress_ax is not None:
            _use_plain_axis(stress_ax)

    if recommended is not None:
        recommended_line = energy_ax.axvline(
            recommended,
            color=colors.reference,
            linestyle="-",
            linewidth=1.5,
            zorder=1.5,
            label=f"Recommended {name(recommended)}",
        )
        energy_handles.append(recommended_line)
        force_ax.axvline(
            recommended, color=colors.reference, linestyle="-", linewidth=1.5, zorder=1.5
        )

    if kind == "kpoint" and compact and recommended is not None and recommended_label:
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

    energy_legend = None
    force_legend = None
    if not compact and x_values:
        # Beside each panel, past its right-hand axis: a legend never covers a point,
        # and the energy legend starts below the title row, clear of a pop-out button.
        energy_legend = energy_ax.legend(
            energy_handles,
            [handle.get_label() for handle in energy_handles],
            loc="upper left",
            bbox_to_anchor=_BesidePanel(energy_ax, None, top=True),
            bbox_transform=IdentityTransform(),
            borderaxespad=0.0,
            fontsize="small",
        )
        style_plot_legend(energy_legend, frame_alpha=0.95)
        force_legend = force_ax.legend(
            force_handles,
            [handle.get_label() for handle in force_handles],
            loc="lower left",
            bbox_to_anchor=_BesidePanel(force_ax, stress_ax, top=False),
            bbox_transform=IdentityTransform(),
            borderaxespad=0.0,
            fontsize="small",
        )
        style_plot_legend(force_legend, frame_alpha=0.95)
    if not compact:
        _draw_titles(energy_ax, force_ax, data, name, energy_unit, energy_offsets, force_offsets)

    return ConvergencePlotAxes(
        energy=energy_ax,
        force=force_ax,
        spacing=spacing_ax,
        stress=stress_ax,
        energy_legend=energy_legend,
        force_legend=force_legend,
        energy_offsets=energy_offsets,
        force_offsets=force_offsets,
        stress_offsets=stress_offsets,
    )


class _BesidePanel(BboxBase):
    """A legend anchor at the right of a panel and of its second axis, found at draw time.

    The anchor is a point at the panel's top (``top``) or bottom, a small gap
    to the right of the panel or, when ``side`` is given, of that axis's tick
    labels and label. Constrained layout then makes room for the legend.

    Matplotlib offers no public anchor that follows another axis's tick
    labels, so this uses two internals: the figure's renderer, to measure
    those labels, and the transform-node ``_invalid`` flag, so that the
    legend's cached copy of the anchor is refreshed whenever constrained
    layout moves the panel. ``tests/app/test_convergence_plot.py`` pins both
    (the legend follows a resize, beside the stress axis).
    """

    def __init__(self, panel: Axes, side: Axes | None, *, top: bool) -> None:
        super().__init__()
        self._panel = panel
        self._side = side
        self._top = top
        # The legend caches its anchor until a child changes: moving the panel does.
        self.set_children(panel.bbox)

    def get_points(self) -> np.ndarray:
        figure = self._panel.figure
        renderer = figure._get_renderer()
        box = self._panel.get_window_extent(renderer)
        right = box.x1
        if self._side is not None:
            decorations = self._side.yaxis.get_tightbbox(renderer)
            if decorations is not None:
                right = max(right, decorations.x1)
        right += _LEGEND_GAP_POINTS * figure.dpi / 72.0
        height = box.y1 if self._top else box.y0
        # Marked valid, so the next move of the panel reaches the legend's cache again.
        self._invalid = 0
        return np.array([[right, height], [right, height]])


class _FreeLineLabel(Text):
    """A label for a horizontal line, placed at draw time where it covers nothing.

    It tries the line's left end, then its right end, above the line and then
    below it, and draws at the first place clear of every data marker and ring,
    tick label and legend of the figure. With no clear place it is not drawn.
    """

    def __init__(self, axes: Axes, height: float, text: str, **style: Any) -> None:
        super().__init__(0.0, height, text, **style)
        self._panel = axes
        self._height = height
        self.set_transform(axes.get_yaxis_transform())

    def draw(self, renderer: Any) -> None:
        self.set_visible(True)
        figure = self.get_figure()
        obstacles = _label_obstacles(figure, renderer, exclude=self)
        frame = self._panel.bbox
        pad = _LINE_LABEL_PAD_POINTS / 72.0
        for x, ha in ((0.0, "left"), (1.0, "right")):
            for va in ("bottom", "top"):
                self.set_position((x, self._height))
                self.set_horizontalalignment(ha)
                self.set_verticalalignment(va)
                offset = ScaledTranslation(
                    pad if ha == "left" else -pad,
                    pad / 2 if va == "bottom" else -pad / 2,
                    figure.dpi_scale_trans,
                )
                self.set_transform(self._panel.get_yaxis_transform() + offset)
                box = self.get_window_extent(renderer)
                inside = (
                    frame.x0 <= box.x0
                    and box.x1 <= frame.x1
                    and frame.y0 <= box.y0
                    and box.y1 <= frame.y1
                )
                if inside and not any(box.overlaps(other) for other in obstacles):
                    super().draw(renderer)
                    return
        self.set_visible(False)


def _label_obstacles(figure: Figure, renderer: Any, *, exclude: Any) -> list[Any]:
    """Boxes a free label must not cover: markers and rings, tick labels and legends."""

    boxes: list[Any] = []
    for axes in figure.axes:
        radius_scale = figure.dpi / 72.0 / 2.0
        for line in axes.get_lines():
            if line.get_marker() in (None, "None", "", " "):
                continue
            radius = line.get_markersize() * radius_scale
            for x, y in zip(line.get_xdata(), line.get_ydata(), strict=False):
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                cx, cy = axes.transData.transform((x, y))
                boxes.append(Bbox([[cx - radius, cy - radius], [cx + radius, cy + radius]]))
        for label in [*axes.get_yticklabels(), *axes.get_xticklabels()]:
            if label.get_text() and label is not exclude:
                boxes.append(label.get_window_extent(renderer))
        legend = axes.get_legend()
        if legend is not None:
            boxes.append(legend.get_window_extent(renderer))
    return boxes


def _ring_converged(
    axis: Axes,
    result: ConvergenceResult | None,
    offsets: Mapping[float, float],
    color: str,
    quantity: str,
    name: Any,
) -> list[Any]:
    """Ring the point where ``result`` converged on its own series; ``[]`` when it did not."""

    if result is None or result.converged_value is None:
        return []
    value = float(result.converged_value)
    if value not in offsets:
        return []
    ring = axis.plot(
        [value],
        [offsets[value]],
        linestyle="none",
        marker="o",
        markersize=_RING_SIZE,
        markerfacecolor="none",
        markeredgewidth=2.0,
        color=color,
        zorder=4,
        label=f"{quantity} converged at {name(value)}",
    )[0]
    return [ring]


def _draw_titles(
    energy_ax: Axes,
    force_ax: Axes,
    data: ConvergencePlotData,
    name: Any,
    energy_unit: str,
    energy_offsets: Mapping[float, float],
    force_offsets: Mapping[float, float],
) -> None:
    """Title each panel with the point its series is measured from, before any analysis too.

    The title row keeps the panel off the pop-out button, so a panel with no
    values is titled as well, without a reference.
    """

    def relative_to(offsets: Mapping[float, float]) -> str:
        return f" relative to {name(max(offsets))}" if offsets else ""

    energy_title = "Energy per atom" if energy_unit == "meV/atom" else "Energy of the cell"
    quantity = "Max force and stress" if data.stresses else "Max force"
    energy_ax.set_title(f"{energy_title}{relative_to(energy_offsets)}", pad=12)
    force_ax.set_title(f"{quantity}{relative_to(force_offsets)}", pad=10)
