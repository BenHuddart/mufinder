"""The relaxation analysis workspaces of a project, and pinning one to the Muon Sites summary.

A workspace groups completed muon relaxations of one charge state for clustering.
The project keeps them in ``ui_state["relaxation_analysis_workspaces"]`` as the
dicts the Relaxation Analysis page builds and edits. This module reads and
writes that list, adds a workspace with its Journal record, deletes one with
any Muon Sites summary pin taken from it, records an applied grouping as a
finding, and pins a workspace's distinct sites to the summary after checking
that it can be shown there. Removing a pin is
:func:`~mufinder.services.muon_sites_summary.unpin_muon_sites`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
import logging

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.models import Structure
from mufinder.persistence.codec import uuid_or_none
from mufinder.services.muon_sites_summary import (
    analysis_workspace_created_record,
    build_muon_sites_summary_snapshot,
    muon_sites_summary_snapshots,
    relaxed_sites_grouped_record,
    store_muon_sites_summary_snapshot,
)
from mufinder.services.project_document import ProjectDocument, append_provenance
from mufinder.services.project_summary import RELAXATION_ANALYSIS_WORKSPACES_KEY
from mufinder.services.study_findings import record_finding

LOGGER = logging.getLogger(__name__)


def analysis_workspaces(document: ProjectDocument | None) -> list[dict[str, object]]:
    """Return shallow copies of the stored workspaces, skipping entries that are not mappings.

    A caller may add, remove or replace a workspace's keys without changing the
    project until it stores the list with :func:`set_analysis_workspaces`; the
    nested values (``job_ids``, ``clustering``) are still the stored ones.
    """

    if document is None:
        return []
    raw = document.ui_state.get(RELAXATION_ANALYSIS_WORKSPACES_KEY, [])
    if not isinstance(raw, list):
        return []
    return [dict(workspace) for workspace in raw if isinstance(workspace, dict)]


def set_analysis_workspaces(
    document: ProjectDocument | None, workspaces: Sequence[Mapping[str, object]]
) -> None:
    """Store ``workspaces`` as the project's list, copying each one."""

    if document is None:
        return
    document.ui_state[RELAXATION_ANALYSIS_WORKSPACES_KEY] = [
        dict(workspace) for workspace in workspaces
    ]


def find_analysis_workspace(
    document: ProjectDocument | None, workspace_id: object
) -> dict[str, object] | None:
    """Return a copy of the first workspace whose id is ``workspace_id``, else ``None``."""

    return next(
        (
            workspace
            for workspace in analysis_workspaces(document)
            if str(workspace.get("id", "")) == str(workspace_id)
        ),
        None,
    )


def add_analysis_workspace(document: ProjectDocument, workspace: Mapping[str, object]) -> bool:
    """Append ``workspace`` to the project and record its creation in the Journal.

    The workspace is stored first and its record written after, so the Journal
    never names a workspace the project does not hold. A record that cannot be
    written is logged and the workspace kept, as a finding's is: the caller
    still saves and shows it. Returns whether the record was written.
    """

    record = analysis_workspace_created_record(workspace)
    workspaces = analysis_workspaces(document)
    workspaces.append(dict(workspace))
    set_analysis_workspaces(document, workspaces)
    try:
        append_provenance(document, record)
    except OSError:
        LOGGER.exception("Could not write the record of analysis workspace %s", workspace.get("id"))
        return False
    return True


def pinned_charge_states(document: ProjectDocument | None, workspace_id: object) -> list[str]:
    """Return the charge states whose Muon Sites summary pin came from ``workspace_id``."""

    return [
        str(charge_state)
        for charge_state, snapshot in muon_sites_summary_snapshots(document).items()
        if isinstance(snapshot, dict) and str(snapshot.get("workspace_id", "")) == str(workspace_id)
    ]


def delete_analysis_workspace(document: ProjectDocument, workspace_id: object) -> list[str] | None:
    """Delete a workspace and any summary pin taken from it.

    Only the workspace goes: the relaxations it drew on stay in the project.
    Returns the charge states whose pin was removed, or ``None`` when no
    workspace has that id.
    """

    workspaces = analysis_workspaces(document)
    index = next(
        (
            position
            for position, workspace in enumerate(workspaces)
            if str(workspace.get("id", "")) == str(workspace_id)
        ),
        None,
    )
    if index is None:
        return None
    unpinned = pinned_charge_states(document, workspace_id)
    del workspaces[index]
    set_analysis_workspaces(document, workspaces)
    for charge_state in unpinned:
        store_muon_sites_summary_snapshot(document, charge_state, None)
    return unpinned


def workspace_host(
    document: ProjectDocument | None, workspace: Mapping[str, object]
) -> Structure | None:
    """Return the host structure a workspace was made for, if the project still has it."""

    if document is None:
        return None
    host_id = uuid_or_none(workspace.get("host_structure_id"))
    return document.structures.get(host_id) if host_id is not None else None


def record_grouping_applied(document: ProjectDocument, workspace_id: object) -> bool:
    """Record the grouping applied to a workspace as a finding.

    A repeat of the latest finding the same day adds no record. Returns whether
    the workspace has a grouping to record, repeat or not, so the caller saves
    the grouping either way.
    """

    workspace = find_analysis_workspace(document, workspace_id)
    if workspace is None:
        return False
    record = relaxed_sites_grouped_record(workspace, workspace_host(document, workspace))
    if record is None:
        return False
    record_finding(document, record)
    return True


class PinRefusal(Enum):
    """Why a workspace's sites cannot be pinned to the Muon Sites summary."""

    #: The workspace has no clustering result, so it has no distinct sites.
    NOT_CLUSTERED = "not_clustered"
    #: The workspace was made for a host structure that is not the active host.
    OTHER_HOST = "other_host"
    #: The summary has no panel for the workspace's charge state.
    UNSUPPORTED_CHARGE_STATE = "unsupported_charge_state"


@dataclass(frozen=True)
class PinResult:
    """The outcome of pinning a workspace: the charge state pinned, or why not."""

    charge_state: str | None = None
    refusal: PinRefusal | None = None


def pin_analysis_workspace(document: ProjectDocument, workspace_id: object) -> PinResult | None:
    """Pin a workspace's distinct sites to the Muon Sites summary.

    The pin replaces the charge state's earlier one. It is refused when the
    workspace has no clustering result, when it was made for a host other than
    the active one (its sites are fractional coordinates in that host's cell),
    or when the summary has no panel for its charge state. Returns ``None``
    when no workspace has that id.
    """

    workspace = find_analysis_workspace(document, workspace_id)
    if workspace is None:
        return None
    snapshot = build_muon_sites_summary_snapshot(workspace, workspace_host(document, workspace))
    if snapshot is None:
        return PinResult(refusal=PinRefusal.NOT_CLUSTERED)
    host_id = document.project.host_structure_id
    if str(snapshot.get("host_structure_id", "")) != ("" if host_id is None else str(host_id)):
        return PinResult(refusal=PinRefusal.OTHER_HOST)
    charge_state = str(snapshot.get("charge_state", POSITIVE_MUON_CHARGE_STATE))
    if charge_state not in (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE):
        return PinResult(refusal=PinRefusal.UNSUPPORTED_CHARGE_STATE)
    store_muon_sites_summary_snapshot(document, charge_state, snapshot)
    return PinResult(charge_state=charge_state)


__all__ = [
    "PinRefusal",
    "PinResult",
    "add_analysis_workspace",
    "analysis_workspaces",
    "delete_analysis_workspace",
    "find_analysis_workspace",
    "pin_analysis_workspace",
    "pinned_charge_states",
    "record_grouping_applied",
    "set_analysis_workspaces",
    "workspace_host",
]
