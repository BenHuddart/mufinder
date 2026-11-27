"""The hierarchical clustering dendrogram the clustering dialogs share."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton
from scipy.cluster import hierarchy

from mufinder.app.theme import (
    cluster_color_map,
    clustering_toolbar_stylesheet,
    plot_series_colours,
)
from mufinder.science.feature_clustering import FeatureClusteringResult, dendrogram_node_clusters


class ClusterDendrogramToolbar(NavigationToolbar2QT):
    """Compact toolbar exposing only the dendrogram navigation tools we use."""

    toolitems = (
        ("Reset", "Reset the dendrogram view", "home", "home"),
        ("Pan", "Pan the dendrogram", "move", "pan"),
        ("Zoom", "Zoom to a rectangle", "zoom_to_rect", "zoom"),
    )

    def __init__(self, canvas, parent=None) -> None:
        super().__init__(canvas, parent, coordinates=False)
        self._button_widgets: dict[str, QPushButton] = {}
        for action in tuple(self.actions()):
            self.removeAction(action)

        self.setStyleSheet(clustering_toolbar_stylesheet())
        self._home_button = self._add_button("home", "Reset", checkable=False)
        self._pan_button = self._add_button("pan", "Pan", checkable=True)
        self._zoom_button = self._add_button("zoom", "Zoom", checkable=True)
        self._sync_button_states()

    def _add_button(self, callback_name: str, label: str, *, checkable: bool) -> QPushButton:
        button = QPushButton(label, self)
        button.setObjectName("clusteringToolbarButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setCheckable(checkable)
        button.setProperty("active", False)
        button.clicked.connect(getattr(self, callback_name))
        self.addWidget(button)
        self._button_widgets[callback_name] = button
        return button

    def _sync_button_states(self) -> None:
        active_mode = getattr(self.mode, "name", "")
        for key, active in (("pan", active_mode == "PAN"), ("zoom", active_mode == "ZOOM")):
            button = self._button_widgets.get(key)
            if button is None:
                continue
            blocked = button.blockSignals(True)
            button.setChecked(active)
            button.blockSignals(blocked)
            button.setProperty("active", active)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _update_buttons_checked(self) -> None:
        super()._update_buttons_checked()
        self._sync_button_states()


def draw_cluster_dendrogram(
    axis: Axes, result: FeatureClusteringResult, *, leaf_labels: Sequence[str]
) -> None:
    """Draw a hierarchical clustering's dendrogram on ``axis``, its links coloured by cluster.

    A link inside one cluster takes that cluster's :func:`cluster_color_map`
    colour, the one its sites have in the structure view. A link joining two
    clusters lies above the cut and keeps the ``reference`` colour.
    """

    above_cut = plot_series_colours().reference
    colours = cluster_color_map(result.cluster_labels)
    node_clusters = dendrogram_node_clusters(result.cluster_labels, result.linkage_matrix)
    hierarchy.dendrogram(
        np.asarray(result.linkage_matrix, dtype=float),
        ax=axis,
        labels=list(leaf_labels),
        above_threshold_color=above_cut,
        link_color_func=lambda node_id: colours.get(node_clusters[int(node_id)], above_cut),
    )
