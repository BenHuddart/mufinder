"""The muon sites the Magnetic Fields page offers, and their relaxed structures.

A site comes from one of three sources.
:data:`~mufinder.services.magnetic_field_results.UEP_SITE_SOURCE` sites are UEP
cluster sites generated into the project and the cluster representatives of
each UEP batch analysis; they are unrelaxed, so fields there are estimates
(:data:`~mufinder.services.magnetic_field_results.UNRELAXED_SITE_NOTE`). :data:`RELAXATION_SITE_SOURCE` sites are the
cluster representatives of the relaxation-analysis workspaces, by default only
those pinned to the Muon Sites summary; each can carry the relaxed structure
of its representative for the muon-induced distortion correction.
:data:`CUSTOM_SITE_SOURCE` is a fractional position typed in by the user.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import UUID

import numpy as np

from mufinder.domain.models import Structure
from mufinder.persistence.codec import (
    float_triplet,
    float_triplet_or_none,
    int_or_none,
    uuid_or_none,
)
from mufinder.science.lattice import fractional_to_cartesian
from mufinder.services.magnetic_field_results import UEP_SITE_SOURCE
from mufinder.services.muon_sites_summary import pinned_sites_by_representative
from mufinder.services.project_document import ProjectDocument
from mufinder.services.project_summary import relaxation_analysis_workspaces
from mufinder.services.project_survey import UEP_ANALYSIS_STATES_KEY

#: The source of a relaxation-analysis cluster representative.
RELAXATION_SITE_SOURCE = "relaxation_clustering"
#: The source of a fractional position typed in by the user.
CUSTOM_SITE_SOURCE = "custom"

#: Generation methods of the UEP cluster sites stored in a project's sites.
UEP_CLUSTER_SITE_METHODS = frozenset({"uep_cluster", "uep_cluster_member", "uep_batch_cluster"})

_SOURCE_LABELS = {
    UEP_SITE_SOURCE: "UEP clustering (unrelaxed estimate)",
    RELAXATION_SITE_SOURCE: "Structural relaxation clustering",
    CUSTOM_SITE_SOURCE: "Custom",
}


@dataclass(frozen=True, slots=True)
class MagneticMuonSiteOption:
    """Muon-site option selected for magnetic-field calculation."""

    source: str
    label: str
    fractional_position: tuple[float, float, float]
    cartesian_position: tuple[float, float, float] | None = None
    cluster_label: str | None = None
    source_label: str | None = None
    distortion_structure_id: UUID | None = None
    properties: dict[str, object] | None = None
    include_distortion: bool = False


def site_source_label(source: str) -> str:
    """Return the name of a site source as the page shows it."""

    return _SOURCE_LABELS.get(source, source.replace("_", " ").title())


def same_muon_site(left: MagneticMuonSiteOption, right: MagneticMuonSiteOption) -> bool:
    """Whether two options name the same site: same source and label, and the same position."""

    return (
        left.source == right.source
        and left.label == right.label
        and np.allclose(left.fractional_position, right.fractional_position, atol=1e-8)
    )


def site_cartesian_position(
    fractional: tuple[float, float, float], structure: Structure
) -> tuple[float, float, float]:
    """Return a fractional site position of ``structure`` in Cartesian ångström."""

    cart = fractional_to_cartesian(fractional, structure.lattice_vectors)
    return (float(cart[0]), float(cart[1]), float(cart[2]))


def distortion_structure(
    document: ProjectDocument | None, site: MagneticMuonSiteOption
) -> Structure | None:
    """Return the relaxed structure a site's distortion correction uses, or ``None`` when it is gone."""

    if document is None or site.distortion_structure_id is None:
        return None
    return document.structures.get(site.distortion_structure_id)


def muon_site_options(
    document: ProjectDocument | None,
    host_structure: Structure | None,
    source: str,
    *,
    pinned_only: bool = False,
) -> list[MagneticMuonSiteOption]:
    """Return the sites of one source; ``pinned_only`` applies to relaxation clusters."""

    if document is None or host_structure is None:
        return []
    if source == UEP_SITE_SOURCE:
        return uep_cluster_muon_site_options(document, host_structure)
    if source == RELAXATION_SITE_SOURCE:
        return relaxation_cluster_muon_site_options(
            document, host_structure, pinned_only=pinned_only
        )
    return []


def default_muon_site(
    document: ProjectDocument | None, host_structure: Structure | None
) -> MagneticMuonSiteOption | None:
    """Return the site a fresh page starts at, or ``None`` when the project has none.

    The lowest-energy site pinned to the Muon Sites summary comes first, then
    the first relaxation cluster, then the first UEP site.
    """

    if document is None or host_structure is None:
        return None
    for options in (
        relaxation_cluster_muon_site_options(document, host_structure, pinned_only=True),
        relaxation_cluster_muon_site_options(document, host_structure),
        uep_cluster_muon_site_options(document, host_structure),
    ):
        if options:
            return options[0]
    return None


def uep_cluster_muon_site_options(
    document: ProjectDocument,
    host_structure: Structure,
) -> list[MagneticMuonSiteOption]:
    """Return the project's UEP cluster sites and every UEP batch cluster representative.

    A batch representative takes its optimised position, else the cluster's
    representative position. Sorted by source, cluster and label.
    """

    options: list[MagneticMuonSiteOption] = []
    seen: set[str] = set()

    for site in document.sites.values():
        if site.generation_method not in UEP_CLUSTER_SITE_METHODS:
            continue
        key = f"site:{site.id}"
        seen.add(key)
        options.append(
            MagneticMuonSiteOption(
                source=UEP_SITE_SOURCE,
                label=site.label or "UEP cluster site",
                source_label="Generated UEP site",
                cluster_label=_cluster_label_from_properties(site.properties),
                fractional_position=float_triplet(site.initial_fractional_position),
                cartesian_position=float_triplet(site.initial_cartesian_position),
                properties=dict(site.properties),
            )
        )

    states = document.ui_state.get(UEP_ANALYSIS_STATES_KEY, {})
    if isinstance(states, dict):
        for raw_job_id, state in states.items():
            if not isinstance(state, dict):
                continue
            source_job_label = _job_label(document, raw_job_id, fallback="UEP source")
            batch_results = state.get("batch_results")
            summary = state.get("batch_summary_data")
            if not isinstance(batch_results, list) or not isinstance(summary, dict):
                continue
            clusters = summary.get("clusters")
            if not isinstance(clusters, (list, tuple)):
                continue
            for cluster in clusters:
                if not isinstance(cluster, dict):
                    continue
                representative_index = int_or_none(cluster.get("representative_index"))
                cluster_index = int_or_none(cluster.get("cluster_index"))
                if representative_index is None or not (
                    0 <= representative_index < len(batch_results)
                ):
                    continue
                result = batch_results[representative_index]
                if not isinstance(result, dict):
                    continue
                fractional = float_triplet_or_none(
                    result.get("optimized_fractional_position")
                    or cluster.get("representative_fractional_position")
                )
                if fractional is None:
                    continue
                cartesian = float_triplet_or_none(
                    result.get("optimized_position_ang")
                ) or site_cartesian_position(fractional, host_structure)
                key = f"uep:{raw_job_id}:{cluster_index}:{representative_index}"
                if key in seen:
                    continue
                seen.add(key)
                options.append(
                    MagneticMuonSiteOption(
                        source=UEP_SITE_SOURCE,
                        label=f"UEP cluster {cluster_index if cluster_index is not None else len(options) + 1}",
                        source_label=str(source_job_label),
                        cluster_label="-" if cluster_index is None else f"Cluster {cluster_index}",
                        fractional_position=fractional,
                        cartesian_position=cartesian,
                        properties={
                            "source_job_id": str(raw_job_id),
                            "cluster": dict(cluster),
                            "representative": dict(result),
                        },
                    )
                )
    return sorted(
        options,
        key=lambda option: (option.source_label or "", option.cluster_label or "", option.label),
    )


def relaxation_cluster_muon_site_options(
    document: ProjectDocument,
    host_structure: Structure,
    *,
    pinned_only: bool = False,
) -> list[MagneticMuonSiteOption]:
    """Return the relaxation-analysis cluster representatives as muon-site options.

    With ``pinned_only`` only the clusters pinned to the Muon Sites summary are
    returned, labelled by Wyckoff label and cluster, in charge-state then
    energy-rank order; a live cluster matches a pin by workspace and
    representative relaxation job. Otherwise every workspace's clusters, sorted
    by name.
    """

    pinned = pinned_sites_by_representative(document) if pinned_only else {}
    ranks = {key: order for order, key in enumerate(pinned)}
    pinned_options: list[tuple[int, MagneticMuonSiteOption]] = []
    options: list[MagneticMuonSiteOption] = []
    for workspace in relaxation_analysis_workspaces(document):
        workspace_name = (
            str(workspace.get("name", "Relaxation analysis")).strip() or "Relaxation analysis"
        )
        clustering = workspace.get("clustering")
        result = clustering.get("result") if isinstance(clustering, dict) else None
        if not isinstance(result, dict):
            continue
        clusters = result.get("clusters")
        if not isinstance(clusters, (list, tuple)):
            continue
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            fractional = float_triplet_or_none(cluster.get("representative_fractional_position"))
            if fractional is None:
                continue
            cluster_index = int_or_none(cluster.get("cluster_index"))
            job_id = uuid_or_none(cluster.get("representative_job_id"))
            # A pin names its cluster's representative relaxation, which survives
            # re-clustering; the cluster index does not.
            pinned_key = (str(workspace.get("id", "")), job_id)
            if pinned_only and pinned_key not in pinned:
                continue
            structure_id = _distorted_structure_id_for_job(document, job_id)
            cartesian = float_triplet_or_none(
                cluster.get("representative_cartesian_position_ang")
            ) or site_cartesian_position(fractional, host_structure)
            option = MagneticMuonSiteOption(
                source=RELAXATION_SITE_SOURCE,
                label=f"{workspace_name} cluster {cluster_index if cluster_index is not None else len(options) + 1}",
                source_label=workspace_name,
                cluster_label="-" if cluster_index is None else f"Cluster {cluster_index}",
                fractional_position=fractional,
                cartesian_position=cartesian,
                distortion_structure_id=structure_id,
                properties={"workspace": dict(workspace), "cluster": dict(cluster)},
            )
            if pinned_only:
                # The label stays the workspace/cluster name so a saved selection
                # still matches; the Cluster column carries the summary label.
                option = replace(option, cluster_label=pinned[pinned_key].label)
                pinned_options.append((ranks[pinned_key], option))
            options.append(option)
    if pinned_only:
        return [option for _rank, option in sorted(pinned_options, key=lambda item: item[0])]
    return sorted(
        options,
        key=lambda option: (option.source_label or "", option.cluster_label or "", option.label),
    )


def _cluster_label_from_properties(properties: dict[str, object]) -> str | None:
    cluster = properties.get("cluster_index")
    if cluster is None:
        return None
    return f"Cluster {cluster}"


def _distorted_structure_id_for_job(document: ProjectDocument, job_id: UUID | None) -> UUID | None:
    if job_id is None:
        return None
    for run in document.runs.values():
        if run.job_id != job_id or run.result_id is None:
            continue
        result = document.results.get(run.result_id)
        if result is None:
            continue
        for structure_id in result.derived_structure_ids:
            if structure_id in document.structures:
                return structure_id
    return None


def _job_label(document: ProjectDocument, raw_job_id: object, *, fallback: str) -> str:
    job_id = uuid_or_none(raw_job_id)
    if job_id is None:
        return fallback
    job = document.jobs.get(job_id)
    return job.label if job is not None and job.label else fallback


__all__ = [
    "CUSTOM_SITE_SOURCE",
    "MagneticMuonSiteOption",
    "RELAXATION_SITE_SOURCE",
    "UEP_CLUSTER_SITE_METHODS",
    "default_muon_site",
    "distortion_structure",
    "muon_site_options",
    "relaxation_cluster_muon_site_options",
    "same_muon_site",
    "site_cartesian_position",
    "site_source_label",
    "uep_cluster_muon_site_options",
]
