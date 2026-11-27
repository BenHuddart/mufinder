"""The project home's cell survey: the host cell in 3D with the muon positions found.

``CellSurveyWidget`` holds an :class:`~mufinder.app.atomic_structure_viewer.AtomicStructureViewer`
showing the active host with one muon sphere per marker of a
:class:`~mufinder.services.project_survey.SurveyLayer`: the pinned sites, else
the most refined positions found so far. One marker can be highlighted, with a
translucent halo around its sphere. The positions come from
:mod:`mufinder.services.project_survey`.

The viewer is created after the card's first paint, so opening a project shows
the home page before PyVista is imported and the VTK render windows are made.
With no host no viewer is created; if the viewer cannot be created, the card
says so and the rest of the page keeps working.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer, MUON_VIEWER_RADIUS_ANG
from mufinder.app.structure_render_guard import (
    ViewerStructureCheck,
    structure_geometry_fingerprint,
)
from mufinder.app.theme import (
    SPACING,
    STRUCTURE_OVERLAY_COLOURS,
    card_margins,
    form_spacing,
    make_empty_state_label,
    style_hint_text,
)
from mufinder.domain.models import Structure
from mufinder.science.lattice import LatticeError, lattice_frac_to_cart
from mufinder.services.project_survey import SurveyLayer

_LOGGER = logging.getLogger(__name__)

#: Radius (Å) and opacity of the halo around the highlighted marker's muon sphere.
HIGHLIGHT_RADIUS_ANG = 2.0 * MUON_VIEWER_RADIUS_ANG
HIGHLIGHT_ALPHA = 0.45
#: Smallest height of the card's 3D view, so the controls, triad and cell all fit.
VIEW_MINIMUM_HEIGHT = 320

NO_HOST_TEXT = "Load a host structure to survey its cell."
VIEWER_UNAVAILABLE_TEXT = "The 3D view is unavailable."


class _ViewArea(QWidget):
    """Where the 3D view sits: every widget added fills it, and it asks for one height.

    It has no layout of its own, so the viewer's height-for-width (larger than the
    room the page leaves) never reaches the card: the card keeps its height when
    the viewer arrives.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(VIEW_MINIMUM_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def add_filling(self, widget: QWidget) -> None:
        """Make ``widget`` a child that always covers the whole area."""

        widget.setParent(self)
        widget.setGeometry(self.rect())

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return QSize(VIEW_MINIMUM_HEIGHT, VIEW_MINIMUM_HEIGHT)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        for child in self.findChildren(QWidget, options=Qt.FindChildOption.FindDirectChildrenOnly):
            child.setGeometry(self.rect())


class CellSurveyWidget(QFrame):
    """The cell survey card: a caption, the 3D view and the layer's title and note."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("overviewCard")
        self._host: Structure | None = None
        self._layer: SurveyLayer | None = None
        self._highlighted: int | None = None
        self._viewer: AtomicStructureViewer | None = None
        self._viewer_frame: QFrame | None = None
        self._viewer_failed = False
        self._creation_scheduled = False
        # What the viewer was last given, so an unchanged refresh leaves it alone.
        self._viewer_check = ViewerStructureCheck()
        self._viewer_overlays: list[dict[str, object]] | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(card_margins())
        layout.setSpacing(form_spacing())

        caption = QLabel("Survey of the cell", self)
        style_hint_text(caption)
        layout.addWidget(caption)

        self.view_area = _ViewArea(self)
        self.view_area.setObjectName("cellSurveyViewArea")
        self.placeholder_label = make_empty_state_label(parent=self.view_area)
        self.view_area.add_filling(self.placeholder_label)
        layout.addWidget(self.view_area, 1)

        self.layer_label = QLabel(self)
        self.layer_label.setObjectName("cardBody")
        self.layer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layer_label.setWordWrap(True)
        layout.addWidget(self.layer_label)

        self.note_label = QLabel(self)
        style_hint_text(self.note_label)
        self.note_label.setWordWrap(True)
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.note_label)
        self._sync()

    # --- public surface -------------------------------------------------------------

    @property
    def viewer(self) -> AtomicStructureViewer | None:
        """The 3D viewer, or ``None`` until it is created (or when it could not be)."""

        return self._viewer

    def set_survey(self, host: Structure | None, layer: SurveyLayer | None) -> None:
        """Show ``layer`` over ``host``.

        The highlight is kept while the host and the layer are unchanged and
        cleared otherwise. The viewer reloads (and its camera resets) only when
        the host's geometry changes; a new layer on the same host is updated in
        place.
        """

        if host is not None and not host.species:
            # Nothing to draw: the card asks for a host as it does with none.
            host, layer = None, None
        unchanged = layer == self._layer and structure_geometry_fingerprint(
            host
        ) == structure_geometry_fingerprint(self._host)
        self._host = host
        self._layer = layer
        if not unchanged:
            self._highlighted = None
        self._sync()

    def current_layer(self) -> SurveyLayer | None:
        return self._layer

    def highlighted_marker(self) -> int | None:
        return self._highlighted

    def highlight_marker(self, index: int | None) -> None:
        """Draw a halo around marker ``index`` of the layer, or around none for ``None``."""

        layer = self._layer
        if index is not None and (layer is None or not 0 <= index < len(layer.markers)):
            index = None
        if index == self._highlighted:
            return
        self._highlighted = index
        self._sync_viewer()

    def create_viewer(self) -> AtomicStructureViewer | None:
        """Create the 3D viewer now, if the survey is showing a host and has none yet.

        The card's first paint schedules this; it does nothing while the card is
        hidden, so a viewer is only ever created to be shown. Creating and first
        showing the viewer imports PyVista and makes its render windows; if that
        fails the card says the 3D view is unavailable and no viewer is kept.
        """

        if (
            self._viewer is not None
            or self._viewer_failed
            or self._host is None
            or not self.isVisible()
        ):
            return self._viewer
        try:
            # Parentless until it is built, so a failed build leaves no half-made child.
            viewer = AtomicStructureViewer()
        except Exception:  # noqa: BLE001 - any backend failure means no 3D view.
            self._viewer_unavailable(None)
            return None
        viewer.setAccessibleName("Cell Survey")
        viewer.set_compact_chrome(True)
        # An inset of the page colour, which the viewer's controls and 3D area paint.
        frame = QFrame()
        frame.setObjectName("cellSurveyViewerFrame")
        frame_layout = QVBoxLayout(frame)
        inset = SPACING.sm
        frame_layout.setContentsMargins(inset, inset, inset, inset)
        frame_layout.addWidget(viewer)
        frame.hide()
        self.view_area.add_filling(frame)
        self._viewer_frame = frame
        self._viewer = viewer
        self._load_viewer()
        try:
            # The first show imports PyVista, creates the render windows and draws.
            self._show_viewer()
        except Exception:  # noqa: BLE001 - no working OpenGL, for one.
            self._viewer_unavailable(viewer)
            return None
        if not viewer.backend_ready():
            # A Qt binding that reports a failed show event rather than raising it.
            self._viewer_unavailable(viewer)
            return None
        self._sync()
        return viewer

    # --- internals ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        if (
            self._viewer is None
            and not self._viewer_failed
            and self._host is not None
            and not self._creation_scheduled
        ):
            # After this paint reaches the screen: the page shows before the viewer's cost.
            self._creation_scheduled = True
            QTimer.singleShot(0, self, self._create_viewer_after_paint)

    def _create_viewer_after_paint(self) -> None:
        self._creation_scheduled = False
        self.create_viewer()

    def _viewer_unavailable(self, viewer: AtomicStructureViewer | None) -> None:
        _LOGGER.warning(
            "The project home's 3D view could not be created",
            exc_info=sys.exc_info()[0] is not None,
        )
        self._viewer_failed = True
        self._viewer = None
        self._viewer_check.forget()
        self._viewer_overlays = None
        if viewer is not None:
            viewer.hide()
            try:
                viewer.shutdown()
            except Exception:  # noqa: BLE001 - it is being discarded either way.
                _LOGGER.debug("Discarded 3D view did not shut down cleanly", exc_info=True)
            viewer.setParent(None)
            viewer.deleteLater()
        frame, self._viewer_frame = self._viewer_frame, None
        if frame is not None:
            frame.hide()
            frame.setParent(None)
            frame.deleteLater()
        self._sync()

    def _sync(self) -> None:
        layer = self._layer
        self.layer_label.setText(layer.title if layer is not None else "")
        self.layer_label.setVisible(layer is not None)
        if layer is not None:
            self.note_label.setText(layer.note)
        else:
            self.note_label.setText(
                "No muon positions found yet." if self._host is not None else ""
            )
        self.note_label.setVisible(bool(self.note_label.text()))
        if self._host is None:
            placeholder = NO_HOST_TEXT
        elif self._viewer_failed:
            placeholder = VIEWER_UNAVAILABLE_TEXT
        else:
            # The view fills in once the viewer exists; until then the area stays blank.
            placeholder = ""
        self.placeholder_label.setText(placeholder)
        self.placeholder_label.setVisible(self._viewer is None or self._host is None)
        self._sync_viewer()
        self.update()

    def _sync_viewer(self) -> None:
        viewer = self._viewer
        if viewer is None:
            return
        if self._host is None:
            if self._viewer_frame is not None:
                self._viewer_frame.hide()
            return
        self._load_viewer()
        self._show_viewer()

    def _load_viewer(self) -> None:
        """Give the viewer the host and markers, when either changed since it was last given them."""

        viewer = self._viewer
        host = self._host
        if viewer is None or host is None:
            return
        change = self._viewer_check.change(host)
        overlays = self._point_overlays(host)
        if change.unchanged and overlays == self._viewer_overlays:
            return
        with viewer.batch_update():
            if change.geometry:
                viewer.set_structure(host)
            elif change.properties:
                # Spins or atom colours edited on the same host: redraw them, keeping the view.
                viewer.set_structure_preserving_view(host)
            viewer.set_point_overlays(overlays)
        self._viewer_check.remember(host)
        self._viewer_overlays = overlays

    def _show_viewer(self) -> None:
        frame = self._viewer_frame
        if frame is not None and frame.isHidden():
            frame.show()

    def _point_overlays(self, host: Structure) -> list[dict[str, object]]:
        """One muon sphere per marker, then a halo around the highlighted one."""

        layer = self._layer
        if layer is None:
            return []
        overlays: list[dict[str, object]] = []
        halo: dict[str, object] | None = None
        for index, marker in enumerate(layer.markers):
            try:
                position = lattice_frac_to_cart(marker.position, host.lattice_vectors)
            except LatticeError:
                continue
            overlays.append(
                {
                    "positions": [position],
                    "colors": STRUCTURE_OVERLAY_COLOURS.muon,
                    "radii": [MUON_VIEWER_RADIUS_ANG],
                    "alpha": 0.96,
                }
            )
            if index == self._highlighted:
                halo = {
                    "positions": [position],
                    "colors": STRUCTURE_OVERLAY_COLOURS.highlight,
                    "radii": [HIGHLIGHT_RADIUS_ANG],
                    "alpha": HIGHLIGHT_ALPHA,
                }
        if halo is not None:
            overlays.append(halo)
        return overlays
