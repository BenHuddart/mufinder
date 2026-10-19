"""Positions each workflow stage has found, for the project home's cell survey.

``survey_layers`` returns, for one stage, the result sets that have positions
in the active host cell: UEP minima per source job, the candidate sites, the
relaxations per charge state (start and end), the distinct sites per analysis
workspace, the sites with zero-point energies, the diffusion paths and the
sites with magnetic-field results. ``discovered_sites_layer`` returns the best
positions found so far, for the stages that have none of their own.

Every position is host fractional, wrapped into ``[0, 1)``. Results tied to a
different host are left out and counted in the layer's ``note``. UEP minima and
diffusion paths can live in a supercell, so they are carried through Cartesian
coordinates into the host cell.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
import math
from uuid import UUID

from mufinder.domain.charge_states import muon_charge_state_label
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job, Structure
from mufinder.science.lattice import LatticeError, lattice_cart_to_frac, lattice_frac_to_cart, wrap_fractional
from mufinder.services.job_results import latest_result_for_job, total_energy_for_job
from mufinder.services.project_document import ProjectDocument, active_host_structure, definitions_for_stage
from mufinder.services.project_summary import (
    MUON_SITES_SUMMARY_SNAPSHOTS_KEY,
    latest_run_for_job,
    magnetic_field_workspaces,
    muon_relaxation_row_jobs,
    relaxation_analysis_workspaces,
)
from mufinder.services.source_candidates import (
    coerce_fractional_triplet,
    completed_muon_relaxation_sources,
    energy_barrier_generated_sources,
    energy_barrier_neb_sources_by_job_id,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_SET_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
)
from mufinder.workflow.muon_site import MUON_RELAXATION_DEFINITION_CALC_TYPE
from mufinder.workflow.quantum import QUANTUM_PHONON_SET_CALC_TYPE

UEP_ANALYSIS_STATES_KEY = "uep_analysis_states"

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
class SurveyPath:
    """A diffusion path between two sites, drawn as a line."""

    start: Fractional
    end: Fractional
    label: str = ""


@dataclass(frozen=True, slots=True)
class SurveyLayer:
    """One result set of one stage, shown on its own in the survey."""

    stage_key: str
    title: str
    markers: tuple[SurveyMarker, ...] = ()
    paths: tuple[SurveyPath, ...] = ()
    note: str = ""


def survey_layers(document: ProjectDocument, stage_key: str) -> tuple[SurveyLayer, ...]:
    """Return the result sets of ``stage_key`` that have positions in the active host cell."""

    host = active_host_structure(document)
    builder = _LAYER_BUILDERS.get(stage_key)
    if host is None or builder is None:
        return ()
    return tuple(layer for layer in builder(document, host) if layer.markers or layer.paths)


def discovered_sites_layer(document: ProjectDocument) -> SurveyLayer | None:
    """Return the most refined positions found so far, or ``None`` when there are none.

    In order of preference: the pinned distinct sites, the clustered analysis
    sites, the relaxed positions, the candidate sites and the UEP minima.
    """

    host = active_host_structure(document)
    if host is None:
        return None
    pinned = _pinned_sites_layer(document, host)
    if pinned is not None:
        return pinned
    for stage_key in ("muon_analysis", "muon_relaxations", "muon_candidates", "uep"):
        layers = survey_layers(document, stage_key)
        if layers:
            return layers[0]
    return None


# --- Per-stage layers ---------------------------------------------------------


def _uep_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    states = document.ui_state.get(UEP_ANALYSIS_STATES_KEY)
    if not isinstance(states, Mapping):
        return []
    layers: list[SurveyLayer] = []
    for raw_job_id, state in states.items():
        job = document.jobs.get(_uuid(raw_job_id)) if _uuid(raw_job_id) else None
        summary = state.get("batch_summary_data") if isinstance(state, Mapping) else None
        clusters = summary.get("clusters") if isinstance(summary, Mapping) else None
        if job is None or not isinstance(clusters, list):
            continue
        source_id = _uep_source_structure_id(document, job)
        if source_id is not None and source_id != host.id and _derived_from(document, source_id) != host.id:
            continue
        markers = []
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            position = _host_fractional_from_cartesian(cluster.get("representative_position_ang"), host)
            if position is None:
                continue
            members = _int(cluster.get("size")) or len(_list(cluster.get("member_indexes"))) or 1
            energy = _float(cluster.get("minimum_energy_ev"))
            detail = f"UEP minimum {_int(cluster.get('cluster_index')) or len(markers) + 1}"
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
        layers.append(SurveyLayer("uep", f"UEP minima · {job.label or 'source job'}", tuple(markers)))
    return layers


def _candidate_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    sites = [site for site in document.sites.values() if site.source_structure_id in (None, host.id)]
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
    energies = [source.total_energy_ev for source in completed if source.total_energy_ev is not None]
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
                SurveyMarker(_wrap(source.folded_fractional_position), SurveyMarkerKind.RELAXED, detail=detail, start=start)
            )
        elif start is not None:
            status = job.status.value if job is not None else "not submitted"
            markers.append(SurveyMarker(start, SurveyMarkerKind.CANDIDATE, detail=f"{label} · {status}"))
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
            markers.append(SurveyMarker(_wrap(position), SurveyMarkerKind.SITE, tag, detail, weight=size))
        name = str(workspace.get("name", "")).strip() or "Analysis workspace"
        pinned = " (pinned)" if str(workspace.get("id", "")) in pinned_ids else ""
        layers.append(SurveyLayer("muon_analysis", f"{name}{pinned}", tuple(markers)))
    return layers


def _quantum_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    markers = []
    for definition in definitions_for_stage(
        document,
        WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_SET_CALC_TYPE,
    ):
        for site_key, job in _mapped_jobs(document, definition, "phonon_job_ids_by_label", "job_ids_by_label"):
            site = document.sites.get(_uuid(site_key)) if _uuid(site_key) else None
            result = latest_result_for_job(document, job.id)
            if site is None or result is None or not isinstance(result.data, Mapping):
                continue
            zpe = _float(result.data.get("muon_zpe_total_eV"))
            position = _wrap(site.current_fractional_position())
            if zpe is None:
                continue
            markers.append(
                SurveyMarker(
                    position,
                    SurveyMarkerKind.SITE,
                    f"ZPE {zpe:.2f} eV",
                    f"{site.label or 'Site'} · zero-point energy {zpe:.3f} eV",
                )
            )
    return [SurveyLayer("quantum_corrections", f"{len(markers)} sites with zero-point energies", tuple(markers))]


def _barrier_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    definitions = [
        definition
        for definition in document.calculations.values()
        if definition.workflow_stage == WorkflowStage.ENERGY_BARRIERS
        and definition.calc_type in {NUDGED_ELASTIC_BAND_SET_CALC_TYPE, TRANSITION_STATE_SEARCH_SET_CALC_TYPE}
    ]
    heights: dict[str, float] = {}
    for job_id, source in energy_barrier_neb_sources_by_job_id(document, definitions).items():
        result = latest_result_for_job(document, job_id)
        height = _float(result.data.get("barrier_height_eV")) if result is not None and isinstance(result.data, Mapping) else None
        if height is not None:
            heights[str(source.get("key", ""))] = height
    paths = []
    markers: dict[Fractional, SurveyMarker] = {}
    for source in energy_barrier_generated_sources(document):
        cell = document.structures.get(_uuid(source.get("reactant_structure_id"))) if _uuid(source.get("reactant_structure_id")) else None
        start = _host_fractional_from_cell(source.get("path_start_fractional"), cell, host)
        end = _host_fractional_from_cell(source.get("path_end_fractional"), cell, host)
        if start is None or end is None:
            continue
        height = heights.get(str(source.get("key", "")))
        paths.append(SurveyPath(start, end, f"{height:.2f} eV" if height is not None else ""))
        for point in (start, end):
            markers.setdefault(point, SurveyMarker(point, SurveyMarkerKind.SITE, detail="Path endpoint"))
    computed = sum(bool(path.label) for path in paths)
    title = f"{len(paths)} diffusion path{'s' if len(paths) != 1 else ''}, {computed} with barriers"
    return [SurveyLayer("energy_barriers", title, tuple(markers.values()), tuple(paths))]


def _magnetic_layers(document: ProjectDocument, host: Structure) -> list[SurveyLayer]:
    layers = []
    for workspace in magnetic_field_workspaces(document):
        form = workspace.get("form")
        results = form.get("magnetic_field_results") if isinstance(form, Mapping) else None
        if not isinstance(results, Mapping) or results.get("kind") != "commensurate":
            continue
        markers = []
        for entry in _list(results.get("results")):
            if not isinstance(entry, Mapping):
                continue
            position = _triplet(entry.get("muon_position"))
            magnitude = _float(entry.get("magnitude_T"))
            if position is None or magnitude is None:
                continue
            site_label = str(entry.get("site_label", "")).strip() or "Site"
            markers.append(
                SurveyMarker(
                    _wrap(position),
                    SurveyMarkerKind.SITE,
                    f"{magnitude:.3g} T",
                    f"{site_label} · |B| = {magnitude:.4g} T",
                )
            )
        title = str(workspace.get("title", "")).strip() or "Magnetic fields"
        layers.append(SurveyLayer("magnetic_fields", f"{title} · local fields", tuple(markers)))
    return layers


_LAYER_BUILDERS = {
    "uep": _uep_layers,
    "muon_candidates": _candidate_layers,
    "muon_relaxations": _relaxation_layers,
    "muon_analysis": _analysis_layers,
    "quantum_corrections": _quantum_layers,
    "energy_barriers": _barrier_layers,
    "magnetic_fields": _magnetic_layers,
}


def _pinned_sites_layer(document: ProjectDocument, host: Structure) -> SurveyLayer | None:
    snapshots = document.ui_state.get(MUON_SITES_SUMMARY_SNAPSHOTS_KEY)
    if not isinstance(snapshots, Mapping):
        return None
    markers = []
    for charge_state, snapshot in snapshots.items():
        if not isinstance(snapshot, Mapping) or str(snapshot.get("host_structure_id", "")) != str(host.id):
            continue
        for site in _list(snapshot.get("sites")):
            position = _triplet(site.get("fractional_position")) if isinstance(site, Mapping) else None
            if position is None:
                continue
            tag = str(site.get("wyckoff_label", "")).strip() or f"Site {_int(site.get('cluster_index')) or len(markers) + 1}"
            energy = _float(site.get("relative_energy_ev"))
            detail = f"{tag} · {muon_charge_state_label(str(charge_state))}"
            if energy is not None:
                detail += f" · {energy:+.3f} eV"
            markers.append(SurveyMarker(_wrap(position), SurveyMarkerKind.SITE, tag, detail))
    if not markers:
        return None
    return SurveyLayer("muon_analysis", "Pinned distinct sites", tuple(markers))


# --- Helpers ------------------------------------------------------------------


def _pinned_workspace_ids(document: ProjectDocument, host: Structure) -> set[str]:
    snapshots = document.ui_state.get(MUON_SITES_SUMMARY_SNAPSHOTS_KEY)
    if not isinstance(snapshots, Mapping):
        return set()
    return {
        str(snapshot.get("workspace_id", ""))
        for snapshot in snapshots.values()
        if isinstance(snapshot, Mapping) and str(snapshot.get("host_structure_id", "")) == str(host.id)
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


def _uep_source_structure_id(document: ProjectDocument, job: Job) -> UUID | None:
    for definition in document.calculations.values():
        if definition.workflow_stage != WorkflowStage.UEP:
            continue
        mapping = definition.parameters.get("job_ids_by_label")
        if isinstance(mapping, Mapping) and str(job.id) in {str(value) for value in mapping.values()}:
            return definition.input_structure_ids[0] if definition.input_structure_ids else None
    return None


def _derived_from(document: ProjectDocument, structure_id: UUID) -> UUID | None:
    structure = document.structures.get(structure_id)
    return structure.parent_structure_id if structure is not None else None


def _mapped_jobs(
    document: ProjectDocument,
    definition: CalculationDefinition,
    *mapping_names: str,
) -> Iterable[tuple[str, Job]]:
    for name in mapping_names:
        mapping = definition.parameters.get(name)
        if not isinstance(mapping, Mapping) or not mapping:
            continue
        for key, raw_job_id in mapping.items():
            job = document.jobs.get(_uuid(raw_job_id)) if _uuid(raw_job_id) else None
            if job is not None:
                yield str(key), job
        return


def _host_fractional_from_cartesian(value: object, host: Structure) -> Fractional | None:
    cartesian = _triplet(value)
    if cartesian is None:
        return None
    try:
        return _wrap(lattice_cart_to_frac(cartesian, host.lattice_vectors))
    except LatticeError:
        return None


def _host_fractional_from_cell(value: object, cell: Structure | None, host: Structure) -> Fractional | None:
    fractional = _triplet(value)
    if fractional is None:
        return None
    if cell is None or cell.id == host.id:
        return _wrap(fractional)
    try:
        cartesian = lattice_frac_to_cart(fractional, cell.lattice_vectors)
    except LatticeError:
        return None
    return _host_fractional_from_cartesian(cartesian, host)


def _method_label(method: str | None) -> str:
    return {
        "manual": "placed by hand",
        "random": "random",
        "near_element": "near an element",
        "uep_cluster": "UEP cluster",
        "uep_cluster_member": "UEP cluster member",
        "uep_batch_cluster": "UEP cluster",
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
