"""Overview page for the parent convergence workflow."""

from __future__ import annotations

from collections.abc import Mapping
from html import escape
import json

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

from mufinder.app.convergence_plot import ConvergencePlotData, draw_convergence_plot
from mufinder.app.plot_canvas import PlotFrame, create_plot_canvas
from mufinder.app.theme import (
    current_palette,
    make_empty_state_label,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
)
from mufinder.science.kpoint_grids import format_kpoint_grid, kpoint_count


_PINNED_PLOT_TITLES = {"cutoff": "Cut-off convergence", "kpoint": "k-point convergence"}


class _PinnedPlotWidget(QWidget):
    """Render either a pinned convergence plot or an instructional placeholder."""

    def __init__(self, *, sweep_kind: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sweep_kind = sweep_kind
        self._placeholder = make_empty_state_label(object_name="overviewPlaceholder", parent=self)
        # What the canvas shows (the snapshot and the theme it was drawn in),
        # so that an unchanged card is not drawn again on every refresh.
        self._drawn_fingerprint: str | None = None

        self._figure, self._canvas = create_plot_canvas(
            figsize=(5.0, 3.8), facecolor=current_palette().bg_surface_alt
        )
        layout_engine = self._figure.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(w_pad=0.02, h_pad=0.02, hspace=0.04, wspace=0.03)
        self.setMinimumHeight(280)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._placeholder, stretch=1)
        self._canvas_frame = PlotFrame(
            self._canvas, title=_PINNED_PLOT_TITLES.get(sweep_kind, "Convergence"), parent=self
        )
        # A canvas's size hint is its current size, which would hold the page at
        # the largest size it has had; the plot takes the room the card has left.
        self._canvas_frame.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self._canvas_frame, stretch=1)

        self.show_placeholder(
            "Open the analysis tool and adopt a result to pin a convergence plot here."
        )

    def show_placeholder(self, message: str) -> None:
        self._placeholder.setText(message)
        self._placeholder.show()
        self._canvas_frame.show_here()
        self._canvas.hide()
        self._canvas_frame.hide()
        self._drawn_fingerprint = None
        if self._figure.axes:
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
        data = ConvergencePlotData.from_mapping(snapshot.get("plot_data"))
        if data is None:
            self.show_placeholder(placeholder_message)
            return

        try:
            fingerprint: str | None = json.dumps(
                [current_palette().mode, snapshot], sort_keys=True, default=str
            )
        except TypeError:
            # Keys of mixed types cannot be sorted; draw the card every time.
            fingerprint = None
        if fingerprint is None or fingerprint != self._drawn_fingerprint:
            recommended, recommended_label = self._recommended_point(snapshot.get("default_value"))
            draw_convergence_plot(
                self._figure,
                data,
                kind=self._sweep_kind,
                compact=True,
                recommended=recommended,
                recommended_label=recommended_label,
                lattice_vectors=self._lattice_vectors(snapshot.get("lattice_vectors")),
            )
            self._canvas.draw_idle()
            self._drawn_fingerprint = fingerprint
        self._placeholder.hide()
        self._canvas_frame.show()
        self._canvas.show()

    def _recommended_point(self, raw: object) -> tuple[float | None, str | None]:
        """Return the adopted value on the plot axis and, for a grid, its name."""

        if self._sweep_kind == "cutoff":
            try:
                return float(raw), None  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None, None
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None, None
        try:
            grid = (int(raw[0]), int(raw[1]), int(raw[2]))
        except (TypeError, ValueError):
            return None, None
        return float(kpoint_count(grid)), format_kpoint_grid(grid)

    @staticmethod
    def _lattice_vectors(raw: object) -> list[list[float]] | None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return None
        try:
            vectors = [[float(component) for component in vector] for vector in raw]
        except (TypeError, ValueError):
            return None
        return vectors if all(len(vector) == 3 for vector in vectors) else None


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
            "Converge the plane-wave cutoff and the k-point grid before the muon workflows.",
            self._content,
        )
        self._subtitle.setWordWrap(True)
        style_section_subtitle(self._subtitle)

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
        cards_row.addWidget(self._cutoff_card, stretch=1)
        cards_row.addWidget(self._kpoint_card, stretch=1)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(12)
        content_layout.addWidget(self._title)
        content_layout.addWidget(self._subtitle)
        content_layout.addLayout(cards_row, stretch=1)

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
