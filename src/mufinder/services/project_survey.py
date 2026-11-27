"""The muon sites found so far, in the active host cell, for the project home's cell survey.

``pinned_sites_layer`` returns the sites pinned to the Muon Sites summary, in
the order the project home's table lists them. ``discovered_sites_layer``
returns the most refined positions found so far: the pinned sites, else the
distinct sites of an analysis workspace, the relaxed positions per charge
state, the candidate sites or the UEP minima.

Every position is host fractional, wrapped into ``[0, 1)``. Results tied to a
different host are left out and counted in the layer's ``note``. UEP minima can
live in a supercell, so they are carried through Cartesian coordinates into the
host cell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
import math
from uuid import UUID

from mufinder.domain.charge_states import muon_charge_state_label
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Job, Structure
from mufinder.science.lattice import LatticeError, lattice_cart_to_frac, wrap_fractional
from mufinder.services.job_monitoring import job_status_word
from mufinder.services.job_results import total_energy_for_job
from mufinder.services.muon_sites_summary import all_pinned_muon_sites
from mufinder.services.project_document import (
    ProjectDocument,
    active_host_structure,
    definitions_for_stage,
)
from mufinder.services.project_summary import (
    MUON_SITES_SUMMARY_SNAPSHOTS_KEY,
    latest_run_for_job,
    muon_relaxation_row_jobs,
    relaxation_analysis_workspaces,
)
from mufinder.services.uep_grouping import uep_source_structure_id
from mufinder.services.uep_results import uep_batch_searches
from mufinder.services.source_candidates import (
    coerce_fractional_triplet,
    completed_muon_relaxation_sources,
)
from mufinder.workflow.muon_site import MUON_RELAXATION_DEFINITION_CALC_TYPE


Fractional = tuple[float, float, float]


class SurveyMarkerKind(StrEnum):
    """What a marker on the survey stands for."""

    UEP_MINIMUM = "uep_minimum"
    CANDIDATE = "candidate"
    RELAXED = "relaxed"
    SITE = "site"


@dataclass(frozen=True, slots=True)
class SurveyMarker:
    """One position on the survey.

    ``start`` is where a relaxed muon began, so the survey can draw how far it
    moved. ``weight`` is the number of relaxations a distinct site gathers.
    ``label`` is a short tag drawn beside the marker; ``detail`` is the tooltip.
    """

    position: Fractional
    kind: SurveyMarkerKind
    label: str = ""
    detail: str = ""
    start: Fractional | None = None
    weight: int = 1


@dataclass(frozen=True, slots=True)
class SurveyLayer:
    """One set of positions found by one stage, shown in the survey."""

    stage_key: str
    title: str
    markers: tuple[SurveyMarker, ...] = ()
    note: str = ""


def pinned_sites_layer(document: ProjectDocument) -> SurveyLayer | None:
    """Return the sites pinned to the Muon Sites summary, or ``None`` when none are.

    The markers follow
    :func:`~mufinder.services.muon_sites_summary.all_pinned_muon_sites`, so
    marker ``i`` is row ``i`` of the project home's table.
    """

    markers = []
    for site in all_pinned_muon_sites(document):
        tag = site.wyckoff_label or f"Site {site.rank}"
        detail = f"Site {site.rank} · {muon_charge_state_label(site.charge_state)}"
        if site.wyckoff_label:
            detail += f" · {site.wyckoff_label}"
        if site.relative_energy_ev is not None:
            detail += f" · {site.relative_energy_ev:+.3f} eV"
        markers.append(
            SurveyMarker(_wrap(site.fractional_position), SurveyMarkerKind.SITE, tag, detail)
        )
    if not markers:
        return None
    return SurveyLayer("muon_analysis", "Pinned muon sites", tuple(markers))


def discovered_sites_layer(document: ProjectDocument) -> SurveyLayer | None:
    """Return the most refined positions found so far, or ``None`` when there are none.

    In order of preference: the pinned distinct sites, the clustered analysis
    sites, the relaxed positions, the candidate sites and the UEP minima. A
    stage with several result sets (workspaces, charge states, source jobs)
    gives its first.
    """

    host = active_host_structure(document)
    if host is None:
        return None
    pinned = pinned_sites_layer(document)
    if pinned is not None:
        return pinned
    for builder in (_analysis_layers, _relaxation_layers, _candidate_layers, _uep_layers):
        for layer in builder(document, host):
            if layer.markers:
                return layer
    return None


# --- Per-stage layers ---------------------------------------------------------


def _uep_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    layers: list[SurveyLayer] = []
    for search in uep_batch_searches(document):
        raw_job_id = search.source_job_id
        job = document.jobs.get(_uuid(raw_job_id)) if _uuid(raw_job_id) else None
        summary = search.grouping
        clusters = summary.get("clusters") if isinstance(summary, Mapping) else None
        if job is None or not isinstance(clusters, list):
            continue
        source_id = uep_source_structure_id(document, job.id)
        if (
            source_id is not None
            and source_id != host.id
            and _derived_from(document, source_id) != host.id
        ):
            continue
        markers = []
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            position = _host_fractional_from_cartesian(
                cluster.get("representative_position_ang"), host
            )
            if position is None:
                continue
            members = _int(cluster.get("size")) or len(_list(cluster.get("member_indexes"))) or 1
            energy = _float(cluster.get("minimum_energy_ev"))
            detail = f"UEP site {_int(cluster.get('cluster_index')) or len(markers) + 1}"
            if energy is not None:
                detail += f" · {energy:.3f} eV"
            markers.append(
                SurveyMarker(
                    position,
                    SurveyMarkerKind.UEP_MINIMUM,
                    detail=f"{detail} · {members} optimisation{'s' if members != 1 else ''}",
                    weight=members,
                )
            )
        layers.append(
            SurveyLayer("uep", f"UEP minima · {job.label or 'source job'}", tuple(markers))
        )
    return layers


def _candidate_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    sites = [
        site for site in document.sites.values() if site.source_structure_id in (None, host.id)
    ]
    markers = tuple(
        SurveyMarker(
            _wrap(site.initial_fractional_position),
            SurveyMarkerKind.CANDIDATE,
            detail=f"{site.label or 'Candidate'} · {_method_label(site.generation_method)} · {site.status}",
        )
        for site in sites
    )
    hidden = len(document.sites) - len(sites)
    note = f"{hidden} from a previous host not shown" if hidden else ""
    return [SurveyLayer("muon_candidates", f"{len(markers)} candidate sites", markers, note=note)]


def _relaxation_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    rows: list[tuple[str, str, UUID | None, Job | None]] = []
    for definition in definitions_for_stage(
        document,
        WorkflowStage.MUON_RELAXATION,
        calc_type=MUON_RELAXATION_DEFINITION_CALC_TYPE,
    ):
        if host.id in definition.input_structure_ids:
            rows.extend(muon_relaxation_row_jobs(document, definition))
    if not rows:
        return []
    completed = completed_muon_relaxation_sources(
        document,
        rows,
        run_for_job=partial(latest_run_for_job, document),
        total_energy_for_job=partial(total_energy_for_job, document),
    )
    completed_by_job = {source.job_id: source for source in completed}
    energies = [
        source.total_energy_ev for source in completed if source.total_energy_ev is not None
    ]
    lowest = min(energies) if energies else None
    by_charge_state: dict[str, list[SurveyMarker]] = {}
    for label, charge_state, site_id, job in rows:
        site = document.sites.get(site_id) if site_id is not None else None
        start = _wrap(site.initial_fractional_position) if site is not None else None
        source = completed_by_job.get(job.id) if job is not None else None
        markers = by_charge_state.setdefault(charge_state, [])
        if source is not None:
            energy = source.total_energy_ev
            detail = label
            if energy is not None and lowest is not None:
                detail += f" · {energy - lowest:+.3f} eV"
            markers.append(
                SurveyMarker(
                    _wrap(source.folded_fractional_position),
                    SurveyMarkerKind.RELAXED,
                    detail=detail,
                    start=start,
                )
            )
        elif start is not None:
            status = job_status_word(job) if job is not None else "not submitted"
            markers.append(
                SurveyMarker(start, SurveyMarkerKind.CANDIDATE, detail=f"{label} · {status}")
            )
    layers = []
    for charge_state, markers in by_charge_state.items():
        relaxed = sum(marker.kind == SurveyMarkerKind.RELAXED for marker in markers)
        title = f"{muon_charge_state_label(charge_state).capitalize()} · {relaxed} of {len(markers)} relaxed"
        layers.append(SurveyLayer("muon_relaxations", title, tuple(markers)))
    return layers


def _analysis_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    pinned_ids = _pinned_workspace_ids(document, host)
    layers = []
    for workspace in relaxation_analysis_workspaces(document):
        if str(workspace.get("host_structure_id", "")) != str(host.id):
            continue
        clustering = workspace.get("clustering")
        result = clustering.get("result") if isinstance(clustering, Mapping) else None
        clusters = result.get("clusters") if isinstance(result, Mapping) else None
        if not isinstance(clusters, list):
            continue
        labels = _wyckoff_labels(result)
        markers = []
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            position = _triplet(cluster.get("representative_fractional_position"))
            if position is None:
                continue
            index = _int(cluster.get("cluster_index")) or len(markers) + 1
            size = _int(cluster.get("size")) or 1
            energy = _float(cluster.get("minimum_relative_energy_ev"))
            tag = labels.get(index) or f"Site {index}"
            detail = f"{tag} · {size} relaxation{'s' if size != 1 else ''}"
            if energy is not None:
                detail += f" · {energy:+.3f} eV"
            markers.append(
                SurveyMarker(_wrap(position), SurveyMarkerKind.SITE, tag, detail, weight=size)
            )
        name = str(workspace.get("name", "")).strip() or "Analysis workspace"
        pinned = " (pinned)" if str(workspace.get("id", "")) in pinned_ids else ""
        layers.append(SurveyLayer("muon_analysis", f"{name}{pinned}", tuple(markers)))
    return layers


# --- Helpers ------------------------------------------------------------------


def _pinned_workspace_ids(document: ProjectDocument, host: Structure) -> set[str]:
    snapshots = document.ui_state.get(MUON_SITES_SUMMARY_SNAPSHOTS_KEY)
    if not isinstance(snapshots, Mapping):
        return set()
    return {
        str(snapshot.get("workspace_id", ""))
        for snapshot in snapshots.values()
        if isinstance(snapshot, Mapping)
        and str(snapshot.get("host_structure_id", "")) == str(host.id)
    }


def _wyckoff_labels(result: Mapping[str, object]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for entry in _list(result.get("wyckoff_assignments")):
        if not isinstance(entry, Mapping):
            continue
        match = entry.get("match")
        label = str(match.get("label", "")).strip() if isinstance(match, Mapping) else ""
        index = _int(entry.get("cluster_index"))
        if index and label:
            labels[index] = label
    return labels


def _derived_from(document: ProjectDocument, structure_id: UUID) -> UUID | None:
    structure = document.structures.get(structure_id)
    return structure.parent_structure_id if structure is not None else None


def _host_fractional_from_cartesian(value: object, host: Structure) -> Fractional | None:
    cartesian = _triplet(value)
    if cartesian is None:
        return None
    try:
        return _wrap(lattice_cart_to_frac(cartesian, host.lattice_vectors))
    except LatticeError:
        return None


def _method_label(method: str | None) -> str:
    return {
        "manual": "placed by hand",
        "random": "random",
        "near_element": "near an element",
        "uep_cluster": "UEP site",
        "uep_cluster_member": "UEP site member",
        "uep_batch_cluster": "UEP site",
        "imported": "imported",
        "imported_legacy": "imported",
    }.get(method or "", method or "unknown method")


def _wrap(position: Sequence[float]) -> Fractional:
    wrapped = wrap_fractional(position)
    return tuple(0.0 if value > 1.0 - 1e-9 else round(value, 9) for value in wrapped)  # type: ignore[return-value]


def _triplet(value: object) -> Fractional | None:
    return coerce_fractional_triplet(value)


def _float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _uuid(value: object) -> UUID | None:
    if not value:
        return None
    try:
        return UUID(str(value))
    except ValueError:
        return None
