"""How a relaxation analysis workspace groups its relaxed muon sites.

A workspace is grouped when it is created and again whenever relaxations join
it, with the method and settings it stores; a new workspace starts with
connected components within 0.25 Å. Applying a grouping from the
``Clustering...`` dialog replaces those settings for that workspace. Both paths
store the same ``clustering`` mapping, which records how the sites were grouped
(:func:`~mufinder.services.muon_sites_summary.relaxed_site_grouping`) when the
grouping is made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging

from mufinder.domain.models import Structure
from mufinder.science.relaxation_analysis import (
    RelaxationClusterSummary,
    RelaxedMuonSite,
    cluster_connected_components,
    cluster_soprano_style,
    dataclass_to_plain,
    normalize_site_energies,
    uep_summary_to_relaxation_clusters,
)
from mufinder.science.wyckoff_analysis import (
    ClusterWyckoffAssignment,
    assign_cluster_wyckoff_sites,
)
from mufinder.services.muon_sites_summary import relaxed_site_grouping

_LOG = logging.getLogger(__name__)

METHOD_CONNECTED = "connected"
METHOD_HIERARCHICAL = "hier"
METHOD_KMEANS = "kmeans"

#: The grouping a new workspace starts with: connected components within 0.25 Å.
DEFAULT_GROUPING_SETTINGS: dict[str, object] = {
    "method": METHOD_CONNECTED,
    "connected_tolerance_ang": 0.25,
    "hierarchical_t": 0.3,
    "kmeans_k": 4,
    "wyckoff_match_tolerance_ang": 0.25,
}


@dataclass(frozen=True, slots=True)
class SiteGrouping:
    """One grouping of a workspace's relaxed sites, in energy order."""

    method: str
    cluster_labels: tuple[int, ...]
    clusters: tuple[RelaxationClusterSummary, ...]
    wyckoff_assignments: tuple[ClusterWyckoffAssignment, ...]
    #: The ``k`` k-means used, at most the number of distinct sites; ``None`` for other methods.
    kmeans_k_used: int | None = None


def grouping_settings(clustering: object) -> dict[str, object]:
    """Return a workspace's stored grouping method and settings, defaults filling any gap."""

    settings = dict(DEFAULT_GROUPING_SETTINGS)
    if not isinstance(clustering, Mapping):
        return settings
    method = str(clustering.get("method", "") or "")
    if method in (METHOD_CONNECTED, METHOD_HIERARCHICAL, METHOD_KMEANS):
        settings["method"] = method
    for key, cast in (
        ("connected_tolerance_ang", float),
        ("hierarchical_t", float),
        ("kmeans_k", int),
        ("wyckoff_match_tolerance_ang", float),
    ):
        try:
            value = cast(clustering[key])  # type: ignore[operator]
        except (KeyError, TypeError, ValueError):
            continue
        # Tolerances may be zero; t and k must be positive.
        if value > 0 or (value == 0 and key.endswith("_ang")):
            settings[key] = value
    return settings


def group_relaxed_sites(
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
    settings: Mapping[str, object],
) -> SiteGrouping:
    """Group ``sites`` with the method and settings in ``settings``.

    Raises :class:`ValueError` (a
    :class:`~mufinder.science.relaxation_analysis.RelaxationAnalysisError` or
    :class:`~mufinder.science.uep.UepError`) when the method cannot run, for
    instance without the optional Soprano dependency.
    """

    resolved = grouping_settings(settings)
    method = str(resolved["method"])
    ordered = normalize_site_energies(sites)
    kmeans_k_used: int | None = None
    if method == METHOD_CONNECTED:
        result = cluster_connected_components(
            ordered,
            host_structure,
            tolerance_ang=float(resolved["connected_tolerance_ang"]),  # type: ignore[arg-type]
        )
        labels, clusters = result.cluster_labels, result.clusters
    else:
        k = max(min(int(resolved["kmeans_k"]), len(ordered)), 1)  # type: ignore[call-overload]
        soprano = cluster_soprano_style(
            ordered,
            host_structure,
            method=method,
            hierarchical_t=float(resolved["hierarchical_t"]),  # type: ignore[arg-type]
            kmeans_k=k,
        )
        labels = soprano.cluster_labels
        clusters = uep_summary_to_relaxation_clusters(ordered, soprano.summary)
        if method == METHOD_KMEANS:
            kmeans_k_used = soprano.summary.clustering_kmeans_k
    assignments = assign_cluster_wyckoff_sites(
        ordered,
        clusters,
        host_structure,
        match_tolerance_ang=float(resolved["wyckoff_match_tolerance_ang"]),  # type: ignore[arg-type]
    )
    return SiteGrouping(
        method=method,
        cluster_labels=tuple(int(label) for label in labels),
        clusters=tuple(clusters),
        wyckoff_assignments=tuple(assignments),
        kmeans_k_used=kmeans_k_used,
    )


def grouped_clustering_state(
    settings: Mapping[str, object],
    grouping: SiteGrouping,
    host_structure: Structure | None,
    *,
    tolerance_scan: Sequence[Sequence[float]] = (),
) -> dict[str, object]:
    """Return the ``clustering`` mapping a workspace stores for ``grouping``.

    It holds the method and settings (k-means with the ``k`` asked for), the
    result, the tolerance scan, and how the sites were grouped, recorded now
    against ``host_structure`` (k-means with the ``k`` it used).
    """

    resolved = grouping_settings(settings)
    resolved["method"] = grouping.method
    clustering: dict[str, object] = {
        **resolved,
        "tolerance_scan": [[float(row[0]), int(row[1])] for row in tolerance_scan],
        "result": {
            "cluster_labels": list(grouping.cluster_labels),
            "clusters": [dataclass_to_plain(cluster) for cluster in grouping.clusters],
            "wyckoff_assignments": [
                dataclass_to_plain(assignment) for assignment in grouping.wyckoff_assignments
            ],
        },
    }
    record = relaxed_site_grouping(clustering, host_structure)
    if grouping.kmeans_k_used is not None:
        # The stored k stays the one asked for, so it can grow with the sites;
        # the record states the k this grouping used.
        record["kmeans_k"] = int(grouping.kmeans_k_used)
    clustering["grouping"] = record
    return clustering


def regroup_workspace(
    workspace: Mapping[str, object],
    sites: Sequence[RelaxedMuonSite],
    host_structure: Structure,
) -> dict[str, object]:
    """Return the ``clustering`` mapping for ``workspace`` grouped over ``sites``.

    The workspace's stored method and settings are used (connected components
    within 0.25 Å for a new workspace). When the stored method cannot run, the
    sites are grouped with the default instead. Should that fail too, the
    workspace is left ungrouped (``result`` is ``None``) rather than raising.
    The tolerance scan is cleared: it described the previous sites.
    """

    settings = grouping_settings(workspace.get("clustering"))
    ungrouped = {**settings, "tolerance_scan": [], "result": None}
    if not sites:
        return ungrouped
    for attempt in (settings, dict(DEFAULT_GROUPING_SETTINGS)):
        try:
            grouping = group_relaxed_sites(sites, host_structure, attempt)
        except ValueError as exc:
            _LOG.warning("Grouping with %s failed: %s", attempt["method"], exc)
            continue
        return grouped_clustering_state(attempt, grouping, host_structure)
    return ungrouped
