"""The pinned Muon Sites summary and the downstream sources it selects.

The Muon Sites summary holds, per muon charge state, a snapshot of one relaxation
analysis workspace's distinct clustered sites, pinned with "Pin to Summary".
This module builds, stores and validates those snapshots, and resolves each
pinned site to the relaxed :class:`~mufinder.domain.models.MuonSite` that
quantum corrections and energy barriers start from, through the same
eligibility rules as :func:`~mufinder.services.source_candidates.eligible_quantum_source_candidates`.

It also owns the per-workflow "Show every relaxed site" choice: the downstream
workflows list the pinned sites by default and every eligible relaxed site when
that choice is on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.charge_states import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    canonical_charge_state,
    muon_charge_state_label,
)
from mufinder.persistence.codec import to_jsonable
from mufinder.services.project_document import ProjectDocument
from mufinder.services.project_summary import (
    MUON_SITES_SUMMARY_SNAPSHOTS_KEY,
    relaxation_analysis_workspaces,
)
from mufinder.services.source_candidates import (
    QuantumSourceCandidate,
    charge_state_for_job,
    eligible_quantum_source_candidates,
)

#: Charge states the summary keeps a panel (and a snapshot) for, in display order.
SUMMARY_CHARGE_STATES = (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE)

#: ``ui_state`` key of the per-workflow "Show every relaxed site" choices.
SHOW_EVERY_RELAXED_SITE_KEY = "downstream_show_every_relaxed_site"

#: Workflow identifiers of the downstream pages that read the summary.
QUANTUM_CORRECTIONS_SCOPE = "quantum_corrections"
ENERGY_BARRIERS_SCOPE = "energy_barriers"
MAGNETIC_FIELDS_SCOPE = "magnetic_fields"

#: Notice a downstream page shows when no charge state has a pin.
NO_PINNED_SITES_NOTICE = (
    "No muon sites are pinned to the Muon Sites summary yet. Pin the distinct "
    "sites from a clustered analysis, or show every relaxed site."
)


# --- snapshots ------------------------------------------------------------------


def muon_sites_summary_snapshots(document: ProjectDocument | None) -> dict[str, object]:
    """Return the stored snapshots keyed by charge state (empty when none)."""

    if document is None:
        return {}
    raw = document.ui_state.get(MUON_SITES_SUMMARY_SNAPSHOTS_KEY, {})
    return raw if isinstance(raw, dict) else {}


def store_muon_sites_summary_snapshot(
    document: ProjectDocument | None,
    charge_state: str,
    snapshot: Mapping[str, object] | None,
) -> None:
    """Store ``snapshot`` as ``charge_state``'s pin, or remove the pin for ``None``."""

    if document is None:
        return
    snapshots = dict(muon_sites_summary_snapshots(document))
    if snapshot is None:
        snapshots.pop(charge_state, None)
    else:
        serialized = to_jsonable(dict(snapshot))
        if isinstance(serialized, dict):
            snapshots[charge_state] = serialized
    document.ui_state[MUON_SITES_SUMMARY_SNAPSHOTS_KEY] = snapshots


def validated_muon_sites_summary_snapshot(
    document: ProjectDocument | None,
    charge_state: str,
) -> dict[str, object] | None:
    """Return ``charge_state``'s snapshot while it describes the active host.

    The snapshot is a decoupled copy of fractional positions in the analysed
    host's cell, so it is hidden (``None``) once a different host is active, as
    well as when it is missing or holds no sites.
    """

    if document is None:
        return None
    snapshot = muon_sites_summary_snapshots(document).get(charge_state)
    if not isinstance(snapshot, dict):
        return None
    host_id = document.project.host_structure_id
    if str(snapshot.get("host_structure_id", "")) != ("" if host_id is None else str(host_id)):
        return None
    sites = snapshot.get("sites")
    if not isinstance(sites, list) or not sites:
        return None
    return snapshot


def muon_site_wyckoff_labels(result: Mapping[str, object]) -> dict[int, str]:
    """Return the Wyckoff label per cluster index of a clustering result."""

    labels: dict[int, str] = {}
    raw = result.get("wyckoff_assignments")
    if not isinstance(raw, list):
        return labels
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        try:
            cluster_index = int(entry.get("cluster_index", 0))
        except (TypeError, ValueError):
            continue
        match = entry.get("match")
        if cluster_index > 0 and isinstance(match, Mapping):
            label = str(match.get("label", "")).strip()
            if label:
                labels[cluster_index] = label
    return labels


def build_muon_sites_summary_snapshot(workspace: Mapping[str, object]) -> dict[str, object] | None:
    """Return the snapshot "Pin to Summary" stores for a clustered workspace.

    Each site records its cluster index, representative fractional position,
    relative energy, Wyckoff label and the representative relaxation's job id,
    which is what resolves the site to a relaxed muon site downstream. Sites are
    ranked by relative energy (sites without one last, cluster index breaking
    ties). Returns ``None`` when the workspace has no host or no clusters.
    """

    # Key the snapshot to the workspace's own host, not the active project host:
    # the cluster positions are fractional coordinates in that host's cell, so
    # validation must hide the pin whenever another host is active.
    host_structure_id = str(workspace.get("host_structure_id", "")).strip()
    if not host_structure_id:
        return None
    clustering = workspace.get("clustering")
    result = clustering.get("result") if isinstance(clustering, Mapping) else None
    clusters = result.get("clusters") if isinstance(result, Mapping) else None
    if not isinstance(result, Mapping) or not isinstance(clusters, list) or not clusters:
        return None
    wyckoff_labels = muon_site_wyckoff_labels(result)
    sites: list[dict[str, object]] = []
    for cluster in clusters:
        if not isinstance(cluster, Mapping):
            continue
        try:
            cluster_index = int(cluster.get("cluster_index", 0))
        except (TypeError, ValueError):
            continue
        position = cluster.get("representative_fractional_position")
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            continue
        try:
            fractional = [float(position[0]), float(position[1]), float(position[2])]
        except (TypeError, ValueError):
            continue
        raw_energy = cluster.get("minimum_relative_energy_ev")
        try:
            energy: float | None = None if raw_energy is None else float(raw_energy)
        except (TypeError, ValueError):
            energy = None
        representative = _uuid_or_none(cluster.get("representative_job_id"))
        sites.append(
            {
                "cluster_index": cluster_index,
                "fractional_position": fractional,
                "relative_energy_ev": energy,
                "wyckoff_label": wyckoff_labels.get(cluster_index, ""),
                "representative_job_id": None if representative is None else str(representative),
            }
        )
    if not sites:
        return None
    sites.sort(
        key=lambda site: (
            site["relative_energy_ev"] is None,
            site["relative_energy_ev"] if site["relative_energy_ev"] is not None else 0.0,
            site["cluster_index"],
        )
    )
    charge_state = canonical_charge_state(workspace.get("charge_state")) or POSITIVE_MUON_CHARGE_STATE
    return {
        "charge_state": charge_state,
        "workspace_id": str(workspace.get("id", "")),
        "workspace_name": str(workspace.get("name", "")).strip() or "Analysis workspace",
        "host_structure_id": host_structure_id,
        "sites": sites,
    }


# --- pinned sites ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PinnedMuonSite:
    """One distinct site of a validated pinned snapshot, in energy-rank order."""

    charge_state: str
    rank: int
    cluster_index: int
    wyckoff_label: str
    relative_energy_ev: float | None
    fractional_position: tuple[float, float, float]
    workspace_id: str
    workspace_name: str
    representative_job_id: UUID | None

    @property
    def label(self) -> str:
        """Return the site's display label: ``"4a · cluster 2"`` or ``"Cluster 2"``."""

        if self.wyckoff_label:
            return f"{self.wyckoff_label} · cluster {self.cluster_index}"
        return f"Cluster {self.cluster_index}"


@dataclass(frozen=True, slots=True)
class PinnedQuantumSource:
    """A pinned site and the eligible relaxed source it resolves to."""

    pinned: PinnedMuonSite
    candidate: QuantumSourceCandidate


@dataclass(frozen=True, slots=True)
class PinnedSourceSelection:
    """The pinned sites the downstream workflows list by default.

    ``pinned_charge_states`` are the charge states with a validated pin;
    ``sources`` the pinned sites that resolve to an eligible relaxed site, in
    charge-state then energy-rank order; ``unresolved`` the pinned sites that no
    longer do (the representative's job was reset or deleted, or a legacy
    snapshot's workspace is gone).
    """

    pinned_charge_states: tuple[str, ...]
    sources: tuple[PinnedQuantumSource, ...]
    unresolved: tuple[PinnedMuonSite, ...]

    def keys(self) -> list[str]:
        return [source.candidate.key for source in self.sources]

    def labels_by_key(self) -> dict[str, str]:
        return {source.candidate.key: source.pinned.label for source in self.sources}


def pinned_muon_sites(document: ProjectDocument | None, charge_state: str) -> list[PinnedMuonSite]:
    """Return the sites of ``charge_state``'s validated pin, in energy-rank order.

    A snapshot pinned before sites recorded ``representative_job_id`` takes it
    from the pinned workspace's cluster with the same index, when that workspace
    still exists; otherwise the site's job id stays ``None``.
    """

    snapshot = validated_muon_sites_summary_snapshot(document, charge_state)
    if snapshot is None or document is None:
        return []
    workspace_id = str(snapshot.get("workspace_id", ""))
    legacy_job_ids = _workspace_representative_job_ids(document, workspace_id)
    sites: list[PinnedMuonSite] = []
    for raw in snapshot.get("sites", []):
        if not isinstance(raw, Mapping):
            continue
        try:
            cluster_index = int(raw.get("cluster_index", 0))
        except (TypeError, ValueError):
            continue
        position = raw.get("fractional_position")
        try:
            fractional = (float(position[0]), float(position[1]), float(position[2]))  # type: ignore[index]
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        raw_energy = raw.get("relative_energy_ev")
        try:
            energy = None if raw_energy is None else float(raw_energy)
        except (TypeError, ValueError):
            energy = None
        if "representative_job_id" in raw:
            job_id = _uuid_or_none(raw.get("representative_job_id"))
        else:
            job_id = legacy_job_ids.get(cluster_index)
        sites.append(
            PinnedMuonSite(
                charge_state=charge_state,
                rank=len(sites) + 1,
                cluster_index=cluster_index,
                wyckoff_label=str(raw.get("wyckoff_label", "")).strip(),
                relative_energy_ev=energy,
                fractional_position=fractional,
                workspace_id=workspace_id,
                workspace_name=str(snapshot.get("workspace_name", "")).strip() or "Analysis workspace",
                representative_job_id=job_id,
            )
        )
    return sites


def pinned_source_selection(
    document: ProjectDocument | None,
    candidates: Sequence[QuantumSourceCandidate] | None = None,
) -> PinnedSourceSelection:
    """Resolve every validated pin to the eligible relaxed sites it names.

    A pinned site resolves to the eligible candidate whose relaxation job is the
    cluster's representative job and whose charge state matches the pin, so a
    pinned site is offered downstream exactly when
    :func:`~mufinder.services.source_candidates.eligible_quantum_source_candidates`
    would offer its relaxed site. ``candidates`` defaults to that list.
    """

    if candidates is None:
        candidates = eligible_quantum_source_candidates(document)
    by_job_id = {candidate.job.id: candidate for candidate in candidates}
    pinned_charge_states: list[str] = []
    sources: list[PinnedQuantumSource] = []
    unresolved: list[PinnedMuonSite] = []
    seen: set[str] = set()
    for charge_state in SUMMARY_CHARGE_STATES:
        sites = pinned_muon_sites(document, charge_state)
        if not sites:
            continue
        pinned_charge_states.append(charge_state)
        for site in sites:
            candidate = by_job_id.get(site.representative_job_id) if site.representative_job_id else None
            if (
                candidate is None
                or candidate.key in seen
                or _candidate_charge_state(candidate) != charge_state
            ):
                unresolved.append(site)
                continue
            seen.add(candidate.key)
            sources.append(PinnedQuantumSource(pinned=site, candidate=candidate))
    return PinnedSourceSelection(
        pinned_charge_states=tuple(pinned_charge_states),
        sources=tuple(sources),
        unresolved=tuple(unresolved),
    )


def has_pinned_sites(document: ProjectDocument | None) -> bool:
    """Return whether any charge state has a validated pin."""

    return any(pinned_muon_sites(document, charge_state) for charge_state in SUMMARY_CHARGE_STATES)


def pinned_sites_by_representative(
    document: ProjectDocument | None,
) -> dict[tuple[str, UUID], PinnedMuonSite]:
    """Return every validated pinned site keyed by ``(workspace_id, representative_job_id)``.

    The representative relaxation job, not the cluster index, identifies a
    pinned site: re-clustering a workspace renumbers its clusters, so matching
    a live cluster by index could offer a different physical site than the pin.
    Sites with no resolvable job id are left out. Keys follow charge-state then
    energy-rank order.
    """

    pinned: dict[tuple[str, UUID], PinnedMuonSite] = {}
    for charge_state in SUMMARY_CHARGE_STATES:
        for site in pinned_muon_sites(document, charge_state):
            if site.representative_job_id is not None:
                pinned.setdefault((site.workspace_id, site.representative_job_id), site)
    return pinned


# --- "Show every relaxed site" --------------------------------------------------


def show_every_relaxed_site(document: ProjectDocument | None, workflow: str) -> bool:
    """Return whether ``workflow`` lists every eligible relaxed site (default off)."""

    if document is None:
        return False
    raw = document.ui_state.get(SHOW_EVERY_RELAXED_SITE_KEY)
    return bool(raw.get(workflow, False)) if isinstance(raw, Mapping) else False


def set_show_every_relaxed_site(document: ProjectDocument | None, workflow: str, enabled: bool) -> None:
    """Store ``workflow``'s "Show every relaxed site" choice in the project."""

    if document is None:
        return
    raw = document.ui_state.get(SHOW_EVERY_RELAXED_SITE_KEY)
    choices = dict(raw) if isinstance(raw, Mapping) else {}
    choices[workflow] = bool(enabled)
    document.ui_state[SHOW_EVERY_RELAXED_SITE_KEY] = choices


def pinned_sites_notice(selection: PinnedSourceSelection, charge_states: Sequence[str]) -> str:
    """Return the notice a downstream page shows above its pinned-site list.

    Empty when every charge state in ``charge_states`` (those with eligible
    relaxed sites) has a pin and every pinned site resolves.
    """

    missing = [
        state
        for state in SUMMARY_CHARGE_STATES
        if state in charge_states and state not in selection.pinned_charge_states
    ]
    parts: list[str] = []
    if not selection.pinned_charge_states:
        parts.append(NO_PINNED_SITES_NOTICE)
    elif missing:
        labels = " or ".join(muon_charge_state_label(state) for state in missing)
        parts.append(f"No {labels} sites are pinned; show every relaxed site to include them.")
    if selection.unresolved:
        count = len(selection.unresolved)
        noun = "site is" if count == 1 else "sites are"
        parts.append(
            f"{count} pinned {noun} no longer an eligible relaxed site (its relaxation "
            "was reset or removed); re-pin the analysis to refresh the summary."
        )
    return " ".join(parts)


@dataclass(frozen=True, slots=True)
class DownstreamSiteScope:
    """Which relaxed sites one downstream workflow lists, and why.

    By default the workflow lists the pinned sites (``selection``) plus any
    source it already tracks, so a configured or submitted row never vanishes;
    with ``show_every_site`` it lists every eligible relaxed site.
    """

    show_every_site: bool
    selection: PinnedSourceSelection
    notice: str

    def scoped_candidates(
        self,
        candidates: Sequence[QuantumSourceCandidate],
        *,
        keep_keys: Sequence[str] = (),
    ) -> list[QuantumSourceCandidate]:
        """Return the listed sources: pinned ones first, then kept extras."""

        if self.show_every_site:
            return list(candidates)
        eligible = {candidate.key for candidate in candidates}
        pinned = [source.candidate for source in self.selection.sources if source.candidate.key in eligible]
        pinned_keys = {candidate.key for candidate in pinned}
        keep = set(keep_keys)
        extras = [
            candidate
            for candidate in candidates
            if candidate.key in keep and candidate.key not in pinned_keys
        ]
        return pinned + extras

    def pinned_label(self, key: str) -> str:
        """Return ``key``'s summary label while pinned sites are listed, else ``""``."""

        if self.show_every_site:
            return ""
        return self.selection.labels_by_key().get(key, "")


def downstream_site_scope(
    document: ProjectDocument | None,
    workflow: str,
    candidates: Sequence[QuantumSourceCandidate] | None = None,
) -> DownstreamSiteScope:
    """Return ``workflow``'s site scope over ``candidates`` (default: every eligible site)."""

    if candidates is None:
        candidates = eligible_quantum_source_candidates(document)
    selection = pinned_source_selection(document, candidates)
    charge_states = {_candidate_charge_state(candidate) for candidate in candidates}
    return DownstreamSiteScope(
        show_every_site=show_every_relaxed_site(document, workflow),
        selection=selection,
        notice=pinned_sites_notice(selection, [state for state in charge_states if state]),
    )


# --- helpers ----------------------------------------------------------------------


def _workspace_representative_job_ids(document: ProjectDocument, workspace_id: str) -> dict[int, UUID]:
    for workspace in relaxation_analysis_workspaces(document):
        if str(workspace.get("id", "")) != workspace_id:
            continue
        clustering = workspace.get("clustering")
        result = clustering.get("result") if isinstance(clustering, Mapping) else None
        clusters = result.get("clusters") if isinstance(result, Mapping) else None
        job_ids: dict[int, UUID] = {}
        for cluster in clusters if isinstance(clusters, list) else []:
            if not isinstance(cluster, Mapping):
                continue
            try:
                cluster_index = int(cluster.get("cluster_index", 0))
            except (TypeError, ValueError):
                continue
            job_id = _uuid_or_none(cluster.get("representative_job_id"))
            if job_id is not None:
                job_ids[cluster_index] = job_id
        return job_ids
    return {}


def _candidate_charge_state(candidate: QuantumSourceCandidate) -> str | None:
    return charge_state_for_job(candidate.job)


def _uuid_or_none(value: object) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None


__all__ = [
    "DownstreamSiteScope",
    "downstream_site_scope",
    "ENERGY_BARRIERS_SCOPE",
    "MAGNETIC_FIELDS_SCOPE",
    "MUON_SITES_SUMMARY_SNAPSHOTS_KEY",
    "NO_PINNED_SITES_NOTICE",
    "QUANTUM_CORRECTIONS_SCOPE",
    "SHOW_EVERY_RELAXED_SITE_KEY",
    "SUMMARY_CHARGE_STATES",
    "PinnedMuonSite",
    "PinnedQuantumSource",
    "PinnedSourceSelection",
    "build_muon_sites_summary_snapshot",
    "muon_site_wyckoff_labels",
    "muon_sites_summary_snapshots",
    "has_pinned_sites",
    "pinned_sites_by_representative",
    "pinned_muon_sites",
    "pinned_source_selection",
    "pinned_sites_notice",
    "set_show_every_relaxed_site",
    "show_every_relaxed_site",
    "store_muon_sites_summary_snapshot",
    "validated_muon_sites_summary_snapshot",
]
