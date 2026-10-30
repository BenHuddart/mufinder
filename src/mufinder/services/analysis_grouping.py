"""How a relaxation analysis workspace stores the grouping of its relaxed muon sites.

A workspace starts ungrouped. Applying a grouping from the ``Clustering...``
dialog stores the method, its settings and the result
(:func:`grouped_clustering_state`), with how the sites were grouped
(:func:`~mufinder.services.muon_sites_summary.relaxed_site_grouping`) recorded
when it is applied. Nothing regroups on its own: relaxations that join a
grouped workspace keep the existing groups and stay ungrouped themselves
(:func:`extend_grouping_with_new_sites`) until a grouping is applied again,
and the grouping is out of date meanwhile (:func:`ungrouped_site_count`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.relaxation_analysis import (
    RelaxationClusterSummary,
    RelaxedMuonSite,
    dataclass_to_plain,
    normalize_site_energies,
)
from mufinder.science.wyckoff_analysis import ClusterWyckoffAssignment
from mufinder.services.muon_sites_summary import relaxed_site_grouping

METHOD_CONNECTED = "connected"
METHOD_HIERARCHICAL = "hier"
METHOD_KMEANS = "kmeans"

#: The settings a new workspace's ``Clustering...`` dialog opens with.
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


def grouped_clustering_state(
    settings: Mapping[str, object],
    grouping: SiteGrouping,
    host_structure: Structure | None,
    *,
    tolerance_scan: Sequence[Sequence[float]] = (),
) -> dict[str, object]:
    """Return the ``clustering`` mapping a workspace stores for an applied ``grouping``.

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
        # The stored k stays the one asked for; the record states the k this grouping used.
        record["kmeans_k"] = int(grouping.kmeans_k_used)
    clustering["grouping"] = record
    return clustering


def extend_grouping_with_new_sites(
    clustering: object,
    previous_sites: Sequence[RelaxedMuonSite],
    sites: Sequence[RelaxedMuonSite],
) -> object:
    """Return ``clustering`` for ``sites`` after relaxations joined ``previous_sites``.

    Nothing is regrouped. A grouped workspace keeps its groups: each earlier
    site keeps its cluster label, moved to its place in the new energy order,
    and each new site gets label 0 (ungrouped). The clusters' member and
    representative indexes move the same way, their sizes and relative
    energies follow the new set (whose lowest energy may have moved), and the
    tolerance scan, which counted the earlier sites, is cleared. The grouping
    record still describes the grouping as applied. A workspace without a grouping,
    or whose stored labels no longer match ``previous_sites``, is returned as
    it is.
    """

    if not isinstance(clustering, Mapping):
        return clustering
    result = clustering.get("result")
    labels = result.get("cluster_labels") if isinstance(result, Mapping) else None
    previous = normalize_site_energies(previous_sites)
    if not isinstance(labels, list) or len(labels) != len(previous):
        return clustering
    ordered = normalize_site_energies(sites)
    new_index = {site.job_id: index for index, site in enumerate(ordered)}
    moved = {
        old_index: new_index[site.job_id]
        for old_index, site in enumerate(previous)
        if site.job_id in new_index
    }
    new_labels = [0] * len(ordered)
    for old_index, label in enumerate(labels):
        if old_index in moved:
            new_labels[moved[old_index]] = int(label)
    clusters: list[object] = []
    for cluster in result.get("clusters", []) or []:  # type: ignore[union-attr]
        if not isinstance(cluster, Mapping):
            clusters.append(cluster)
            continue
        updated = dict(cluster)
        members = cluster.get("member_indexes")
        if isinstance(members, list):
            kept = [moved[int(i)] for i in members if int(i) in moved]
            updated["member_indexes"] = kept
            updated["size"] = len(kept)
            # Relative energies follow the workspace's new lowest energy.
            relative = [
                float(ordered[i].relative_energy_ev)  # type: ignore[arg-type]
                for i in kept
                if ordered[i].relative_energy_ev is not None
            ]
            updated["minimum_relative_energy_ev"] = min(relative) if relative else None
            updated["average_relative_energy_ev"] = float(np.mean(relative)) if relative else None
            updated["standard_deviation_ev"] = float(np.std(relative)) if relative else None
        representative = cluster.get("representative_index")
        if isinstance(representative, int) and representative in moved:
            updated["representative_index"] = moved[representative]
        clusters.append(updated)
    return {
        **clustering,
        # The scan counted the earlier sites only.
        "tolerance_scan": [],
        "result": {**result, "cluster_labels": new_labels, "clusters": clusters},  # type: ignore[dict-item]
    }


def ungrouped_site_count(clustering: object, labels: Sequence[int]) -> int:
    """Return how many sites joined a grouped workspace after its grouping was applied.

    The sites with no cluster (label 0) of a workspace that holds a grouping
    with at least one cluster; 0 for an ungrouped workspace.
    """

    result = clustering.get("result") if isinstance(clustering, Mapping) else None
    if not isinstance(result, Mapping) or not any(int(label) for label in labels):
        return 0
    return sum(1 for label in labels if not int(label))
