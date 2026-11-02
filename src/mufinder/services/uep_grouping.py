"""The groups of the UEP batch searches a project holds.

A search's results and groups are saved in the project's UEP analysis states.
:func:`regroup_saved_uep_searches` brings a project saved with a retired clustering
up to date when it opens, from the results it stored, so everything that reads the
groups sees one kind. Muon sites generated earlier keep the source job, cluster
index and member index they recorded. It also brings each search to the form the
analysis wizard writes, so opening the wizard and changing nothing is no change.
"""

from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.persistence.codec import to_jsonable
from mufinder.science.uep import optimization_result_from_dict
from mufinder.science.uep_clustering import (
    UEP_GROUPING_METHOD,
    group_uep_minima,
    saved_uep_search_log,
    uep_batch_summary_from_dict,
    uep_search_log,
)
from mufinder.services.project_document import ProjectDocument

_LOG = logging.getLogger(__name__)

#: The ``ui_state`` key of the UEP analysis state of each source job.
UEP_ANALYSIS_STATES_KEY = "uep_analysis_states"
#: State keys nothing reads any more: the clustering window, and whether a search
#: finished its zero-point energies, which its results now say.
_RETIRED_STATE_KEYS = frozenset({"clustering_window_state", "batch_zpe_completed"})
#: Selections name groups that no longer exist once a search is regrouped.
_SELECTION_KEYS = ("batch_selected_keys", "batch_selected_row")


def uep_source_structure_id(document: ProjectDocument, job_id: UUID) -> UUID | None:
    """Return the structure a UEP source job was configured on, ``None`` when it has none."""

    for run in document.runs.values():
        if run.job_id != job_id or run.definition_id is None:
            continue
        definition = document.calculations.get(run.definition_id)
        if definition is not None and definition.input_structure_ids:
            return definition.input_structure_ids[0]
    for definition in document.calculations.values():
        if definition.workflow_stage != WorkflowStage.UEP:
            continue
        mapping = definition.parameters.get("job_ids_by_label")
        if (
            isinstance(mapping, dict)
            and str(job_id) in {str(value) for value in mapping.values()}
            and definition.input_structure_ids
        ):
            return definition.input_structure_ids[0]
    return None


def _source_structure(document: ProjectDocument, raw_job_id: str) -> Any:
    try:
        structure_id = uep_source_structure_id(document, UUID(raw_job_id))
    except ValueError:
        return None
    if structure_id is None:
        structure_id = document.project.host_structure_id
    return document.structures.get(structure_id) if structure_id is not None else None


def _needs_regrouping(state: dict[str, Any]) -> bool:
    results = state.get("batch_results")
    summary = state.get("batch_summary_data")
    method = summary.get("clustering_method") if isinstance(summary, dict) else None
    return isinstance(results, list) and bool(results) and method != UEP_GROUPING_METHOD


def _regrouped(state: dict[str, Any], structure: Any) -> dict[str, Any] | None:
    """Return ``state`` with its stored results grouped, ``None`` when they cannot be."""

    results = [optimization_result_from_dict(entry) for entry in state["batch_results"]]
    if any(result is None for result in results):
        return None
    summary = group_uep_minima(results, source_structure=structure)
    regrouped = {key: value for key, value in state.items() if key not in _SELECTION_KEYS}
    regrouped["batch_summary_data"] = to_jsonable(asdict(summary))
    text = state.get("batch_summary_text")
    regrouped["batch_summary_text"] = uep_search_log(text if isinstance(text, str) else "", summary)
    return regrouped


def _with_restored_log(state: dict[str, Any]) -> dict[str, Any]:
    """Return ``state`` with the log the wizard shows on restoring it."""

    raw_results = state.get("batch_results")
    text = state.get("batch_summary_text")
    if not raw_results or not isinstance(raw_results, list) or not isinstance(text, str):
        return state
    results = [optimization_result_from_dict(entry) for entry in raw_results]
    summary = uep_batch_summary_from_dict(state.get("batch_summary_data"))
    if summary is None or any(result is None for result in results):
        return state
    if not all(0 <= cluster.representative_index < len(results) for cluster in summary.clusters):
        return state
    log = saved_uep_search_log(text, results, summary)
    return state if log == text else {**state, "batch_summary_text": log}


def regroup_saved_uep_searches(document: ProjectDocument) -> bool:
    """Regroup every saved search with a retired or missing grouping, in ``ui_state``.

    Also drops retired state keys and rewrites each log as the wizard does on restoring
    it, so opening the wizard changes neither. A search that cannot be
    regrouped, because its results are malformed or its structure is gone, keeps what
    it has. Nothing is recorded in the Journal: the search was not run again. Returns
    whether anything changed.
    """

    states = document.ui_state.get(UEP_ANALYSIS_STATES_KEY)
    if not isinstance(states, dict):
        return False
    changed = False
    updated = dict(states)
    for raw_job_id, state in states.items():
        if not isinstance(state, dict):
            continue
        current = state
        if not _RETIRED_STATE_KEYS.isdisjoint(current):
            current = {k: v for k, v in current.items() if k not in _RETIRED_STATE_KEYS}
        if _needs_regrouping(current):
            structure = _source_structure(document, str(raw_job_id))
            try:
                regrouped = None if structure is None else _regrouped(current, structure)
            except Exception as exc:  # noqa: BLE001 - one search must not stop a project opening.
                _LOG.warning("Could not regroup the UEP search of job %s: %s", raw_job_id, exc)
                regrouped = None
            if regrouped is None:
                _LOG.warning("The UEP search of job %s was not regrouped", raw_job_id)
            else:
                current = regrouped
        try:
            current = _with_restored_log(current)
        except Exception as exc:  # noqa: BLE001 - one search must not stop a project opening.
            _LOG.warning("Could not tidy the UEP search log of job %s: %s", raw_job_id, exc)
        if current is not state:
            updated[raw_job_id] = current
            changed = True
    if changed:
        document.ui_state[UEP_ANALYSIS_STATES_KEY] = updated
    return changed


__all__ = ["UEP_ANALYSIS_STATES_KEY", "regroup_saved_uep_searches", "uep_source_structure_id"]
