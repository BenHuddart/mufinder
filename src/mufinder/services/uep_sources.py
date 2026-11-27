"""The structures a UEP workflow can run on, and the one the UEP page has selected.

A UEP source job runs on the active host or on one of the host's completed
pristine relaxations. The page's choice is kept in the page's form state,
``ui_state["uep_form"]``, under ``selected_source_structure_id``. A choice
counts only while it names one of those structures; otherwise the page uses
the host.
"""

from __future__ import annotations

from uuid import UUID

from mufinder.domain.models import Structure
from mufinder.services.project_document import ProjectDocument, active_host_structure

_FORM_KEY = "uep_form"
_SELECTED_KEY = "selected_source_structure_id"


def uep_source_structures(document: ProjectDocument) -> list[Structure]:
    """Return the active host, then its pristine relaxations sorted by label.

    Empty when the project has no active host.
    """

    host = active_host_structure(document)
    if host is None:
        return []
    derived = [
        structure
        for structure in document.structures.values()
        if structure.origin == "pristine_relaxation" and structure.parent_structure_id == host.id
    ]
    derived.sort(key=lambda structure: (structure.label.casefold(), str(structure.id)))
    return [host, *derived]


def _offered(document: ProjectDocument, raw_id: object) -> Structure | None:
    """Return the structure of :func:`uep_source_structures` that ``raw_id`` names."""

    if not isinstance(raw_id, str):
        return None
    try:
        structure_id = UUID(raw_id)
    except ValueError:
        return None
    for structure in uep_source_structures(document):
        if structure.id == structure_id:
            return structure
    return None


def _stored_choice(document: ProjectDocument) -> object:
    raw_state = document.ui_state.get(_FORM_KEY, {})
    return raw_state.get(_SELECTED_KEY) if isinstance(raw_state, dict) else None


def selected_uep_source_structure(document: ProjectDocument) -> Structure | None:
    """Return the structure the UEP page has selected, else the first available one.

    A saved selection wins while it names one of :func:`uep_source_structures`;
    otherwise the first of them (the active host), or ``None`` when there is none.
    A structure of a host that is no longer active is not a choice.
    """

    selected = _offered(document, _stored_choice(document))
    if selected is not None:
        return selected
    available = uep_source_structures(document)
    return available[0] if available else None


def select_uep_source_structure(document: ProjectDocument, structure_id: str) -> None:
    """Keep ``structure_id`` as the UEP page's selection; an empty id clears it."""

    state = dict(document.ui_state.get(_FORM_KEY, {}))
    if structure_id:
        state[_SELECTED_KEY] = structure_id
    else:
        state.pop(_SELECTED_KEY, None)
    document.ui_state[_FORM_KEY] = state


def store_uep_form_state(document: ProjectDocument, state: dict) -> None:
    """Keep the UEP page's form ``state``, but never a source choice it cannot show.

    The page reports the structure its picker shows. While the picker is not yet
    filled for this project (empty, or still listing another project's
    structures, as when a saved form is restored on opening) that names no
    structure of :func:`uep_source_structures`. Such a report is never stored:
    the stored choice stays if it names one of those structures, and otherwise
    no choice is stored.
    """

    form = dict(state)
    if _SELECTED_KEY in form and _offered(document, form[_SELECTED_KEY]) is None:
        stored = _stored_choice(document)
        form[_SELECTED_KEY] = stored if _offered(document, stored) is not None else None
    document.ui_state[_FORM_KEY] = form


__all__ = [
    "select_uep_source_structure",
    "selected_uep_source_structure",
    "store_uep_form_state",
    "uep_source_structures",
]
