"""The UEP batch searches a project holds, one durable result per source job.

A finished search is stored as a ``Result`` of type :data:`UEP_BATCH_SEARCH_RESULT_TYPE`
and saved with the project by the next autosave, like any other result. Its ``data``
holds:

- ``source_job_id``: the UEP source job the search ran on;
- ``settings``: the search's settings (:data:`UEP_SEARCH_SETTING_KEYS`);
- ``results``: every optimisation, with its energies, zero-point energy and
  classification, as :class:`~mufinder.science.uep.UepOptimizationResult` fields;
- ``grouping``: the grouping of the minima and its tolerance, as
  :class:`~mufinder.science.uep_clustering.UepBatchSummary` fields, or ``None``;
- ``log``: the search log the analysis window shows.

The record's ``calculation_run_id`` is the source job's run when the search was
stored, so removing the job removes its search. Readers find a search by
``source_job_id``. The analysis window's layout and view stay in ``ui_state`` under
:data:`UEP_ANALYSIS_STATES_KEY`, keyed by job id; they hold no results.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, dataclass
import logging
from typing import Any
from uuid import UUID

from mufinder.domain.models import Job, Result
from mufinder.persistence.codec import to_jsonable, uuid_or_none
from mufinder.persistence.migrations import (
    UEP_ANALYSIS_STATES_KEY,
    UEP_BATCH_SEARCH_RESULT_TYPE,
)
from mufinder.science.site_generation import SiteGenerationError
from mufinder.science.uep import optimization_result_from_dict
from mufinder.science.uep_clustering import (
    UEP_GROUPING_METHOD,
    group_uep_minima,
    saved_uep_search_log,
    uep_batch_summary_from_dict,
    uep_search_log,
)
from mufinder.services.project_document import ProjectDocument
from mufinder.services.site_generation_service import uep_generation_selection
from mufinder.services.study_findings import record_uep_minima
from mufinder.services.uep_grouping import uep_source_structure_id

_LOG = logging.getLogger(__name__)

#: The settings a search records: the potential's Gaussian width factor and the form's
#: starts, distances, start attempts, force tolerance and optimiser steps.
UEP_SEARCH_SETTING_KEYS = (
    "gw_factor",
    "batch_count",
    "batch_min_muon_muon",
    "batch_min_muon_atom",
    "batch_max_attempts",
    "batch_optimization_tolerance_ev",
    "batch_max_steps",
)
#: Display-state keys that select groups, which no longer exist once a search is regrouped.
_SELECTION_KEYS = ("batch_selected_keys", "batch_selected_row")


@dataclass(frozen=True, slots=True)
class UepBatchSearch:
    """One stored search, read from its result record."""

    result_id: UUID
    source_job_id: str
    settings: Mapping[str, Any]
    results: list[Any]
    grouping: Mapping[str, Any] | None
    log: str

    def payload(self) -> dict[str, Any]:
        """The search as the analysis window takes it back."""

        return {
            "settings": dict(self.settings),
            "results": list(self.results),
            "grouping": None if self.grouping is None else dict(self.grouping),
            "log": self.log,
        }


def _search_from_result(result: Result) -> UepBatchSearch | None:
    data = result.data if isinstance(result.data, Mapping) else {}
    job_id = data.get("source_job_id")
    if not isinstance(job_id, str) or not job_id:
        return None
    settings = data.get("settings")
    results = data.get("results")
    grouping = data.get("grouping")
    log = data.get("log")
    return UepBatchSearch(
        result_id=result.id,
        source_job_id=job_id,
        settings=settings if isinstance(settings, Mapping) else {},
        results=results if isinstance(results, list) else [],
        grouping=grouping if isinstance(grouping, Mapping) else None,
        log=log if isinstance(log, str) else "",
    )


def _search_records(document: ProjectDocument) -> list[Result]:
    records = [
        result
        for result in document.results.values()
        if result.result_type == UEP_BATCH_SEARCH_RESULT_TYPE
    ]
    return sorted(records, key=lambda result: (result.created_at, str(result.id)))


def uep_batch_searches(document: ProjectDocument) -> list[UepBatchSearch]:
    """Return every stored search, oldest first."""

    return [
        search
        for record in _search_records(document)
        if (search := _search_from_result(record)) is not None
    ]


def _search_record(document: ProjectDocument, job_id: UUID) -> Result | None:
    target = str(job_id)
    return next(
        (
            record
            for record in _search_records(document)
            if isinstance(record.data, Mapping) and record.data.get("source_job_id") == target
        ),
        None,
    )


def uep_batch_search(document: ProjectDocument, job_id: UUID) -> UepBatchSearch | None:
    """Return the search stored for ``job_id``, or ``None``."""

    record = _search_record(document, job_id)
    return None if record is None else _search_from_result(record)


def _latest_run_id(document: ProjectDocument, job_id: UUID) -> UUID | None:
    runs = [run for run in document.runs.values() if run.job_id == job_id]
    return max(runs, key=lambda run: run.run_index).id if runs else None


def _search_data(job_id: UUID, search: Mapping[str, Any]) -> dict[str, Any]:
    settings = search.get("settings")
    grouping = search.get("grouping")
    log = search.get("log")
    return to_jsonable(
        {
            "source_job_id": str(job_id),
            "settings": {
                key: settings[key]
                for key in UEP_SEARCH_SETTING_KEYS
                if isinstance(settings, Mapping) and key in settings
            },
            "results": list(search.get("results") or []),
            "grouping": grouping if isinstance(grouping, Mapping) else None,
            "log": log if isinstance(log, str) else "",
        }
    )


def store_uep_batch_search(
    document: ProjectDocument, job: Job, search: Mapping[str, Any] | None
) -> bool:
    """Store ``search`` as ``job``'s search, replacing the one before; return whether it changed.

    A search without results removes the stored one. New groups are also recorded in
    the Journal as a finding.
    """

    existing = _search_record(document, job.id)
    if search is None or not search.get("results"):
        if existing is None:
            return False
        del document.results[existing.id]
        return True
    data = _search_data(job.id, search)
    previous = existing.data if existing is not None else None
    if existing is not None and existing.data == data:
        return False
    if existing is None:
        existing = Result(result_type=UEP_BATCH_SEARCH_RESULT_TYPE)
        document.results[existing.id] = existing
    existing.calculation_run_id = _latest_run_id(document, job.id)
    existing.data = data
    record_uep_minima(document, job, previous, data)
    return True


# --- Analysis window state ----------------------------------------------------


def uep_analysis_state(document: ProjectDocument, job_id: UUID) -> dict[str, Any]:
    """Return the analysis window's saved layout and view for ``job_id``, or ``{}``."""

    states = document.ui_state.get(UEP_ANALYSIS_STATES_KEY, {})
    if not isinstance(states, dict):
        states = {}
    state = states.get(str(job_id), {})
    return state if isinstance(state, dict) else {}


def store_uep_analysis_state(
    document: ProjectDocument, job_id: UUID, state: Mapping[str, Any]
) -> bool:
    """Keep the analysis window's layout and view for ``job_id``; return whether it was kept.

    ``state`` is stored in its JSON form. One that is not a mapping in that form
    is not kept.
    """

    serialized_state = to_jsonable(state)
    if not isinstance(serialized_state, dict):
        return False
    states = document.ui_state.get(UEP_ANALYSIS_STATES_KEY, {})
    if not isinstance(states, dict):
        states = {}
    document.ui_state[UEP_ANALYSIS_STATES_KEY] = {**states, str(job_id): serialized_state}
    return True


# --- Generation rows ----------------------------------------------------------


def _classification_label(result: Mapping[str, Any]) -> str:
    classification = result.get("classification")
    if isinstance(classification, Mapping):
        return str(classification.get("label", ""))
    return str(classification or "")


def _group_row(
    search: UepBatchSearch,
    job: Job,
    source_structure_id: UUID | None,
    cluster: Mapping[str, Any],
) -> dict[str, object] | None:
    grouping = search.grouping or {}
    results = search.results
    try:
        cluster_index = int(cluster.get("cluster_index"))
        cluster_size = int(cluster.get("size", 0))
        representative_index = int(cluster.get("representative_index"))
    except (TypeError, ValueError):
        return None
    if not (0 <= representative_index < len(results)):
        return None
    result = results[representative_index]
    if not isinstance(result, Mapping):
        return None
    # The group reports its representative in the asymmetric unit.
    fractional = cluster.get(
        "representative_fractional_position", result.get("optimized_fractional_position")
    )
    cartesian = cluster.get("representative_position_ang", result.get("optimized_position_ang"))
    if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
        return None
    if not isinstance(cartesian, (list, tuple)) or len(cartesian) != 3:
        return None
    classification = _classification_label(result)
    energies = {
        "classical_energy_ev": result.get("classical_energy_ev"),
        "zero_point_energy_ev": result.get("zero_point_energy_ev"),
        "total_energy_ev": result.get("total_energy_ev"),
    }
    properties = {
        "clustering_method": grouping.get("clustering_method"),
        "threshold_ang": grouping.get("threshold_ang"),
        "energy_tolerance_ev": grouping.get("energy_tolerance_ev"),
        "source_job_label": job.label,
        "source_job_id": str(job.id),
        "cluster_index": cluster_index,
        "cluster_size": cluster_size,
        "representative_index": representative_index,
        "member_index": representative_index,
        "selection_kind": "cluster",
        "classification": classification,
        **energies,
    }
    return {
        "source_key": f"{job.id}:cluster:{cluster_index}:cluster:{representative_index}",
        "source_job_id": str(job.id),
        "source_job_label": job.label,
        "source_structure_id": "" if source_structure_id is None else str(source_structure_id),
        "cluster_index": cluster_index,
        "cluster_size": cluster_size,
        "selection_kind": "cluster",
        "classification": classification,
        **energies,
        "fractional_position": [float(value) for value in fractional],
        "cartesian_position": [float(value) for value in cartesian],
        "generation_method": "uep_cluster",
        "label": f"UEP Site {cluster_index}",
        "properties": properties,
    }


def uep_cluster_source_rows(
    document: ProjectDocument, active_job_ids: Collection[UUID]
) -> list[dict[str, object]]:
    """Return the Generation page's UEP rows: one per group of each active job's search.

    Each row holds the group's representative in the active host cell, as
    :func:`~mufinder.services.site_generation_service.uep_generation_selection` gives it.
    A group whose representative cannot be placed in the host is left out.
    """

    rows: list[dict[str, object]] = []
    for search in uep_batch_searches(document):
        job_id = uuid_or_none(search.source_job_id)
        if job_id is None or job_id not in active_job_ids:
            continue
        job = document.jobs.get(job_id)
        clusters = None if search.grouping is None else search.grouping.get("clusters")
        if job is None or not isinstance(clusters, (list, tuple)):
            continue
        source_structure_id = uep_source_structure_id(document, job_id)
        for cluster in clusters:
            if not isinstance(cluster, Mapping):
                continue
            row = _group_row(search, job, source_structure_id, cluster)
            if row is None:
                continue
            try:
                rows.append(uep_generation_selection(document, row))
            except SiteGenerationError:
                continue
    rows.sort(
        key=lambda row: (
            str(row.get("source_job_label", "")),
            int(row.get("cluster_index", 0)),
            str(row.get("source_key", "")),
        )
    )
    return rows


# --- Loading ------------------------------------------------------------------


def _source_structure(document: ProjectDocument, job_id: UUID | None) -> Any:
    if job_id is None:
        return None
    structure_id = uep_source_structure_id(document, job_id)
    if structure_id is None:
        structure_id = document.project.host_structure_id
    return document.structures.get(structure_id) if structure_id is not None else None


def _needs_regrouping(data: Mapping[str, Any]) -> bool:
    results = data.get("results")
    grouping = data.get("grouping")
    method = grouping.get("clustering_method") if isinstance(grouping, Mapping) else None
    return isinstance(results, list) and bool(results) and method != UEP_GROUPING_METHOD


def _regrouped(data: dict[str, Any], structure: Any) -> dict[str, Any] | None:
    """Return ``data`` with its stored results grouped, ``None`` when they cannot be."""

    results = [optimization_result_from_dict(entry) for entry in data["results"]]
    if any(result is None for result in results):
        return None
    summary = group_uep_minima(results, source_structure=structure)
    log = data.get("log")
    return {
        **data,
        "grouping": to_jsonable(asdict(summary)),
        "log": uep_search_log(log if isinstance(log, str) else "", summary),
    }


def _with_restored_log(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` with the log the analysis window shows on restoring it."""

    raw_results = data.get("results")
    log = data.get("log")
    if not raw_results or not isinstance(raw_results, list) or not isinstance(log, str):
        return data
    results = [optimization_result_from_dict(entry) for entry in raw_results]
    summary = uep_batch_summary_from_dict(data.get("grouping"))
    if summary is None or any(result is None for result in results):
        return data
    if not all(0 <= cluster.representative_index < len(results) for cluster in summary.clusters):
        return data
    restored = saved_uep_search_log(log, results, summary)
    return data if restored == log else {**data, "log": restored}


def _guarded(job_id: object, what: str, step: Callable[[], dict[str, Any] | None]) -> Any:
    try:
        return step()
    except Exception as exc:  # noqa: BLE001 - one search must not stop a project opening.
        _LOG.warning("Could not %s the UEP search of job %s: %s", what, job_id, exc)
        return None


def _drop_selections(document: ProjectDocument, job_id: str) -> None:
    states = document.ui_state.get(UEP_ANALYSIS_STATES_KEY)
    state = states.get(job_id) if isinstance(states, dict) else None
    if isinstance(state, dict) and not set(_SELECTION_KEYS).isdisjoint(state):
        states[job_id] = {key: value for key, value in state.items() if key not in _SELECTION_KEYS}


def regroup_saved_uep_searches(document: ProjectDocument) -> set[UUID]:
    """Regroup every stored search with a retired or missing grouping.

    Also rewrites each log as the analysis window does on restoring it, so opening
    the window changes neither. A regrouped search's saved row selections are dropped,
    as they name groups that may no longer exist. A search that cannot be regrouped,
    because its results are malformed or its structure is gone, keeps what it has.
    Nothing is recorded in the Journal: the search was not run again. Returns the ids
    of the records that changed.
    """

    changed: set[UUID] = set()
    for record in _search_records(document):
        data = record.data if isinstance(record.data, dict) else None
        if data is None:
            continue
        current = data
        job_text = str(data.get("source_job_id", ""))
        if _needs_regrouping(current):
            structure = _source_structure(document, uuid_or_none(job_text))
            regrouped = (
                None
                if structure is None
                else _guarded(job_text, "regroup", lambda: _regrouped(current, structure))
            )
            if regrouped is None:
                _LOG.warning("The UEP search of job %s was not regrouped", job_text)
            else:
                current = regrouped
                _drop_selections(document, job_text)
        restored = _guarded(job_text, "tidy the log of", lambda: _with_restored_log(current))
        if restored is not None:
            current = restored
        if current is not data:
            record.data = current
            changed.add(record.id)
    return changed


__all__ = [
    "UEP_ANALYSIS_STATES_KEY",
    "UEP_BATCH_SEARCH_RESULT_TYPE",
    "UEP_SEARCH_SETTING_KEYS",
    "UepBatchSearch",
    "regroup_saved_uep_searches",
    "store_uep_analysis_state",
    "store_uep_batch_search",
    "uep_analysis_state",
    "uep_batch_search",
    "uep_batch_searches",
    "uep_cluster_source_rows",
]
