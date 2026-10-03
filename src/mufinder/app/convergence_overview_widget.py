"""Overview page for the parent convergence workflow."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.plot_canvas import PlotFrame
from mufinder.app.theme import (
    current_palette,
    make_empty_state_label,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
)


_PINNED_PLOT_TITLES = {"cutoff": "Cut-off convergence", "kpoint": "k-point convergence"}

class _PinnedPlotWidget(QWidget):
    """Render either a pinned convergence plot or an instructional placeholder."""

    def __init__(self, *, sweep_kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sweep_kind = sweep_kind
        self._placeholder = make_empty_state_label(object_name="overviewPlaceholder", parent=self)

        palette = current_palette()
        self._figure = Figure(
            figsize=(5.0, 3.8),
            facecolor=palette.bg_surface_alt,
            constrained_layout=True,
        )
        layout_engine = self._figure.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(w_pad=0.02, h_pad=0.02, hspace=0.04, wspace=0.03)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._placeholder, stretch=1)
        self._canvas_frame = PlotFrame(self._canvas, title=_PINNED_PLOT_TITLES.get(sweep_kind, "Convergence"), parent=self)
        layout.addWidget(self._canvas_frame, stretch=1)

        self.show_placeholder("Open the analysis tool and adopt a result to pin a convergence plot here.")

    def show_placeholder(self, message: str) -> None:
        self._placeholder.setText(message)
        self._placeholder.show()
        self._canvas_frame.show_here()
        self._canvas.hide()
        self._canvas_frame.hide()
        self._figure.clear()
        self._canvas.draw_idle()

    def set_snapshot(
        self,
        snapshot: Mapping[str, object] | None,
        *,
        placeholder_message: str,
    ) -> None:
        if not isinstance(snapshot, Mapping):
            self.show_placeholder(placeholder_message)
            return

        plot_data = snapshot.get("plot_data")
        if not isinstance(plot_data, Mapping):
            self.show_placeholder(placeholder_message)
            return

        try:
            x_values = [float(value) for value in plot_data.get("cutoffs", [])]
            energies = [float(value) for value in plot_data.get("energies", [])]
            forces = [float(value) for value in plot_data.get("forces", [])]
            energy_threshold = float(plot_data.get("energy_threshold", 0.001))
            force_threshold = float(plot_data.get("force_threshold", 0.05))
        except (TypeError, ValueError):
            self.show_placeholder(placeholder_message)
            return

        if not x_values or len(x_values) != len(energies) or len(x_values) != len(forces):
            self.show_placeholder(placeholder_message)
            return

        point_spacings: dict[float, float] = {}
        raw_spacings = plot_data.get("point_spacings")
        if isinstance(raw_spacings, Mapping):
            for key, value in raw_spacings.items():
                try:
                    point_spacings[float(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        stresses: list[float] = []
        stress_threshold: float | None = None
        try:
            raw_stresses = plot_data.get("stresses") or []
            stresses = [float(value) for value in raw_stresses]
            raw_stress_threshold = plot_data.get("stress_threshold")
            if raw_stress_threshold is not None:
                stress_threshold = float(raw_stress_threshold)
        except (TypeError, ValueError):
            stresses = []
            stress_threshold = None

        # Only honour stress when the series matches the x-axis 1:1 and a
        # threshold was actually adopted. The examiner enforces this when
        # building the snapshot; this is a defensive check for partial payloads.
        if not stresses or len(stresses) != len(x_values) or stress_threshold is None:
            stresses = []
            stress_threshold = None

        point_labels_raw = snapshot.get("point_labels")
        point_labels: dict[float, str] = {}
        if isinstance(point_labels_raw, Mapping):
            for key, value in point_labels_raw.items():
                try:
                    point_labels[float(key)] = str(value)
                except (TypeError, ValueError):
                    continue

        recommended_raw = snapshot.get("default_value")
        recommended_value: float | None = None
        if self._sweep_kind == "cutoff":
            if recommended_raw is not None:
                try:
                    recommended_value = float(recommended_raw)
                except (TypeError, ValueError):
                    recommended_value = None
        else:
            if isinstance(recommended_raw, (list, tuple)) and len(recommended_raw) == 3:
                try:
                    grid = (
                        int(recommended_raw[0]),
                        int(recommended_raw[1]),
                        int(recommended_raw[2]),
                    )
                except (TypeError, ValueError):
                    grid = None
                if grid is not None:
                    recommended_value = float(grid[0] * grid[1] * grid[2])

        self._draw_plot(
            x_values=x_values,
            energies=energies,
            forces=forces,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            recommended_value=recommended_value,
            point_labels=point_labels,
            point_spacings=point_spacings,
            stresses=stresses or None,
            stress_threshold=stress_threshold,
        )
        self._placeholder.hide()
        self._canvas_frame.show()
        self._canvas.show()

    def _apply_axis_theme(self, axis) -> None:
        palette = current_palette()
        axis.set_facecolor(palette.bg_surface)
        axis.tick_params(axis="both", colors=palette.text_primary, labelcolor=palette.text_primary)
        axis.xaxis.label.set_color(palette.text_primary)
        axis.yaxis.label.set_color(palette.text_primary)
        axis.title.set_color(palette.text_primary)
        for spine in axis.spines.values():
            spine.set_color(palette.border)
        axis.grid(True, color=palette.border, alpha=0.35)

    def _draw_plot(
        self,
        *,
        x_values: list[float],
        energies: list[float],
        forces: list[float],
        energy_threshold: float,
        force_threshold: float,
        recommended_value: float | None,
        point_labels: Mapping[float, str],
        point_spacings: Mapping[float, float] | None = None,
        stresses: list[float] | None = None,
        stress_threshold: float | None = None,
    ) -> None:
        palette = current_palette()
        self._figure.set_facecolor(palette.bg_surface_alt)
        self._figure.clear()
        energy_ax = self._figure.add_subplot(2, 1, 1)
        force_ax = self._figure.add_subplot(2, 1, 2, sharex=energy_ax)

        energy_ax.plot(x_values, energies, marker="o", linestyle="-", color="#1d4ed8")
        force_ax.plot(x_values, forces, marker="o", linestyle="-", color="#b91c1c")

        if recommended_value is not None:
            energy_ax.axvline(recommended_value, color=palette.text_muted, linestyle="-", alpha=0.55)
            force_ax.axvline(recommended_value, color=palette.text_muted, linestyle="-", alpha=0.55)
        force_ax.axhline(force_threshold, color=palette.warning, linestyle=":", alpha=0.85)

        energy_ax.set_ylabel("Energy (eV)", fontsize=9)
        energy_ax.tick_params(axis="x", labelbottom=False)
        energy_ax.tick_params(axis="y", labelsize=8)
        force_ax.set_ylabel("Max force\n(eV/Ang)", fontsize=9)
        force_ax.tick_params(axis="both", labelsize=8)

        # Stress overlay: secondary y-axis on the force subplot, mirroring the
        # examiner dialog (see _STRESS_COLOR / amber #d97706 in convergence_examiner.py).
        # The colour literal is intentional and theme-independent so force (red) and
        # stress (amber) remain visually distinct regardless of palette.
        show_stress = bool(stresses) and stress_threshold is not None
        if show_stress:
            stress_ax = force_ax.twinx()
            stress_ax.plot(x_values, stresses, marker="s", linestyle="-", color="#d97706")
            stress_ax.axhline(stress_threshold, color="#d97706", linestyle=":", alpha=0.85)
            stress_ax.set_ylabel("Max stress\n(GPa)", fontsize=9)
            stress_ax.yaxis.label.set_color("#d97706")
            stress_ax.tick_params(axis="y", colors="#d97706", labelcolor="#d97706", labelsize=8)
            for spine in stress_ax.spines.values():
                spine.set_color(palette.border)
            stress_ax.spines["right"].set_color("#d97706")

        if self._sweep_kind == "cutoff":
            force_ax.set_xlabel("Cutoff energy (eV)", labelpad=6, fontsize=9)
        else:
            grid_labels = [
                point_labels.get(value, str(int(value)))
                .replace("Grid:", "")
                .replace(" x ", "x")
                .strip()
                for value in x_values
            ]
            # Gamma-only (1x1x1) has no unique k-point spacing; treat those
            # ticks as exempt from the "all grids have spacings" requirement
            # and suppress their sub-label so the overview is consistent with
            # the examiner's tooltip which also omits the spacing for 1x1x1.
            has_spacings = bool(point_spacings) and all(
                grid == "1x1x1" or value in point_spacings
                for grid, value in zip(grid_labels, x_values, strict=True)
            ) and any(
                grid != "1x1x1" and value in point_spacings
                for grid, value in zip(grid_labels, x_values, strict=True)
            )
            if has_spacings:
                ordered_labels = [
                    f"{grid}\n{point_spacings[value]:.3g}"
                    if grid != "1x1x1" and value in point_spacings
                    else grid
                    for grid, value in zip(grid_labels, x_values, strict=True)
                ]
                force_ax.set_xlabel(
                    "K-point grid (spacing, 1/Ang)", labelpad=6, fontsize=9
                )
            else:
                ordered_labels = grid_labels
                force_ax.set_xlabel("K-point grid", labelpad=6, fontsize=9)
            force_ax.set_xticks(x_values, ordered_labels, rotation=15, ha="right")

        badge_box = {
            "boxstyle": "round,pad=0.18",
            "fc": palette.bg_surface,
            "ec": palette.border,
            "alpha": 0.95,
        }
        energy_ax.text(
            0.99,
            1.01,
            f"dE <= {energy_threshold:.3g} eV/atom",
            transform=energy_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            # Match the energy line/marker colour so each badge maps visually
            # onto the series it describes (parallels the amber dS badge).
            color="#1d4ed8",
            bbox=badge_box,
        )
        # Force-subplot badges: when stress is absent, dF sits in its historical
        # top-right corner. When stress is present, dF moves left to make room
        # for dS on the right (so each badge sits above the axis it describes,
        # with both right-anchored and separated by a small measured gap).
        if show_stress:
            dF_text = f"dF <= {force_threshold:.3g} eV/Ang"
            dS_text = f"dS <= {stress_threshold:.3g} GPa"
            dS_badge = force_ax.text(
                0.99,
                1.01,
                dS_text,
                transform=force_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#d97706",
                bbox=badge_box,
            )
            # Measure dS extent so dF can sit immediately to its left with a small gap.
            self._canvas.draw_idle()
            renderer = self._canvas.get_renderer()
            try:
                bbox_pixels = dS_badge.get_window_extent(renderer=renderer)
                bbox_axes = force_ax.transAxes.inverted().transform_bbox(bbox_pixels)
                dS_width_axes = float(bbox_axes.width)
            except Exception:
                # Fall back to a fixed offset if the renderer can't size the text yet.
                dS_width_axes = 0.30
            # ~5% of axes width gives a clear visual separation between the
            # two right-anchored badges; tighten if it pushes dF off-axis.
            gap = 0.05
            dF_anchor_x = max(0.0, 0.99 - dS_width_axes - gap)
            force_ax.text(
                dF_anchor_x,
                1.01,
                dF_text,
                transform=force_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#b91c1c",
                bbox=badge_box,
            )
        else:
            force_ax.text(
                0.99,
                1.01,
                f"dF <= {force_threshold:.3g} eV/Ang",
                transform=force_ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                # Match the force line/marker colour for visual association.
                color="#b91c1c",
                bbox=badge_box,
            )
        self._apply_axis_theme(energy_ax)
        self._apply_axis_theme(force_ax)
        self._canvas.draw_idle()


class _OverviewCard(QFrame):
    """Notebook-style card for a single convergence sub-workflow."""

    def __init__(
        self,
        *,
        title: str,
        guidance: str,
        action_text: str,
        sweep_kind: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("overviewCard")

        self._title = QLabel(title, self)
        self._title.setObjectName("overviewCardTitle")
        self._action_button = QPushButton(action_text, self)
        self._action_button.setObjectName("overviewActionButton")

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(12)
        header_row.addWidget(self._title)
        header_row.addStretch(1)
        header_row.addWidget(self._action_button, alignment=Qt.AlignmentFlag.AlignTop)

        self._guidance = QLabel(guidance, self)
        self._guidance.setWordWrap(True)
        style_hint_text(self._guidance)

        self._summary_layout = QVBoxLayout()
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setSpacing(6)
        self._summary_labels: dict[str, QLabel] = {}

        self._pinned_zone = QFrame(self)
        self._pinned_zone.setObjectName("overviewPinnedZone")
        self._pinned_zone.setMinimumHeight(300)
        pinned_layout = QVBoxLayout(self._pinned_zone)
        pinned_layout.setContentsMargins(12, 12, 12, 12)
        self._plot = _PinnedPlotWidget(sweep_kind=sweep_kind, parent=self._pinned_zone)
        pinned_layout.addWidget(self._plot)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addLayout(header_row)
        layout.addWidget(self._guidance)
        layout.addLayout(self._summary_layout)
        layout.addWidget(self._pinned_zone, stretch=1)

    def set_summary_rows(self, rows: list[tuple[str, str]]) -> None:
        seen_keys = {key for key, _value in rows}
        for key in list(self._summary_labels):
            if key in seen_keys:
                continue
            label = self._summary_labels.pop(key)
            self._summary_layout.removeWidget(label)
            label.deleteLater()

        for row_index, (key, value) in enumerate(rows):
            label = self._summary_labels.get(key)
            if label is None:
                label = QLabel(self)
                label.setWordWrap(True)
                label.setObjectName("overviewSummaryValue")
                self._summary_labels[key] = label
            label.setText(f"<b>{escape(key)}:</b> {escape(value)}")
            self._summary_layout.addWidget(label)


class ConvergenceOverviewWidget(QWidget):
    """Parent workflow overview for cutoff and k-point convergence."""

    open_cutoff_requested = Signal()
    open_kpoint_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._content = QWidget(self._scroll)
        self._content.setObjectName("convergenceOverviewContent")
        self._scroll.setWidget(self._content)

        self._title = QLabel("Convergence Testing", self._content)
        style_section_title(self._title)

        self._subtitle = QLabel(
            "Production DFT settings should be converged with respect to both plane-wave cutoff energy "
            "and reciprocal-space sampling before the later muon workflows are trusted.",
            self._content,
        )
        self._subtitle.setWordWrap(True)
        style_section_subtitle(self._subtitle)

        self._hint = QLabel(
            "Each pinned plot is created when you adopt a project default from the corresponding analysis dialog.",
            self._content,
        )
        self._hint.setWordWrap(True)
        style_hint_text(self._hint)

        self._cutoff_card = _OverviewCard(
            title="Cutoff Convergence",
            guidance="Set a stable plane-wave cutoff and pin the adopted analysis here.",
            action_text="Open Cutoff Convergence",
            sweep_kind="cutoff",
            parent=self._content,
        )
        self._kpoint_card = _OverviewCard(
            title="K-Point Convergence",
            guidance="Set a stable reciprocal-space grid and pin the adopted analysis here.",
            action_text="Open K-Point Convergence",
            sweep_kind="kpoint",
            parent=self._content,
        )
        self._cutoff_card._action_button.clicked.connect(self.open_cutoff_requested.emit)
        self._kpoint_card._action_button.clicked.connect(self.open_kpoint_requested.emit)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(18)
        cards_row.setAlignment(Qt.AlignmentFlag.AlignTop)
        cards_row.addWidget(self._cutoff_card, stretch=1)
        cards_row.addWidget(self._kpoint_card, stretch=1)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(12)
        content_layout.addWidget(self._title)
        content_layout.addWidget(self._subtitle)
        content_layout.addWidget(self._hint)
        content_layout.addLayout(cards_row)
        content_layout.addStretch(1)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self._scroll)

    def set_cutoff_panel(
        self,
        *,
        summary_rows: list[tuple[str, str]],
        snapshot: Mapping[str, object] | None,
        placeholder_message: str,
    ) -> None:
        self._cutoff_card.set_summary_rows(summary_rows)
        self._cutoff_card._plot.set_snapshot(snapshot, placeholder_message=placeholder_message)

    def set_kpoint_panel(
        self,
        *,
        summary_rows: list[tuple[str, str]],
        snapshot: Mapping[str, object] | None,
        placeholder_message: str,
    ) -> None:
        self._kpoint_card.set_summary_rows(summary_rows)
        self._kpoint_card._plot.set_snapshot(snapshot, placeholder_message=placeholder_message)
