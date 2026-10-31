"""The hierarchical clustering dendrogram the clustering dialogs share."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes
import numpy as np
from scipy.cluster import hierarchy

from mufinder.app.theme import cluster_color_map, plot_series_colours
from mufinder.science.uep_clustering import UepClusteringResult, dendrogram_node_clusters


def draw_cluster_dendrogram(
    axis: Axes, result: UepClusteringResult, *, leaf_labels: Sequence[str]
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
