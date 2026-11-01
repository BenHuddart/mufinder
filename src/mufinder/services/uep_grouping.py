"""The groups of the UEP batch searches a project holds.

A search's results and groups are saved in the project's UEP analysis states.
:func:`regroup_saved_uep_searches` brings a project saved with a retired clustering
up to date when it opens, from the results it stored, so everything that reads the
groups sees one kind. Muon sites generated earlier keep the source job, cluster
index and member index they recorded.
"""

from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.services.project_document import ProjectDocument

_LOG = logging.getLogger(__name__)

_UEP_ANALYSIS_STATES_KEY = "uep_analysis_states"
#: The method of a current grouping, and the state key of the retired clustering window.
_CURRENT_METHOD = "symmetry_distance"
_RETIRED_WINDOW_STATE_KEY = "clustering_window_state"
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
    return isinstance(results, list) and bool(results) and method != _CURRENT_METHOD


def _regrouped(state: dict[str, Any], structure: Any) -> dict[str, Any] | None:
    """Return ``state`` with its stored results grouped, ``None`` when they cannot be."""

    # Deferred: only a project saved with a retired clustering pays for these imports.
    from mufinder.science.uep import optimization_result_from_dict
    from mufinder.science.uep_clustering import group_uep_minima, uep_search_log

    results = [optimization_result_from_dict(entry) for entry in state["batch_results"]]
    if any(result is None for result in results):
        return None
    summary = group_uep_minima(results, source_structure=structure)
    regrouped = {key: value for key, value in state.items() if key not in _SELECTION_KEYS}
    regrouped["batch_summary_data"] = asdict(summary)
    text = state.get("batch_summary_text")
    regrouped["batch_summary_text"] = uep_search_log(text if isinstance(text, str) else "", summary)
    return regrouped


def regroup_saved_uep_searches(document: ProjectDocument) -> bool:
    """Regroup every saved search with a retired or missing grouping, in ``ui_state``.

    Also drops the retired clustering window's state. A search that cannot be
    regrouped, because its results are malformed or its structure is gone, keeps what
    it has. Nothing is recorded in the Journal: the search was not run again. Returns
    whether anything changed.
    """

    states = document.ui_state.get(_UEP_ANALYSIS_STATES_KEY)
    if not isinstance(states, dict):
        return False
    changed = False
    updated = dict(states)
    for raw_job_id, state in states.items():
        if not isinstance(state, dict):
            continue
        current = state
        if _RETIRED_WINDOW_STATE_KEY in current:
            current = {k: v for k, v in current.items() if k != _RETIRED_WINDOW_STATE_KEY}
        if _needs_regrouping(current):
            structure = _source_structure(document, str(raw_job_id))
            try:
                regrouped = None if structure is None else _regrouped(current, structure)
            except ValueError as exc:
                _LOG.warning("Could not regroup the UEP search of job %s: %s", raw_job_id, exc)
                regrouped = None
            if regrouped is None:
                _LOG.warning("The UEP search of job %s was not regrouped", raw_job_id)
            else:
                current = regrouped
        if current is not state:
            updated[raw_job_id] = current
            changed = True
    if changed:
        document.ui_state[_UEP_ANALYSIS_STATES_KEY] = updated
    return changed


__all__ = ["regroup_saved_uep_searches", "uep_source_structure_id"]
