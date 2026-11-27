"""The question asked before the active host changes.

:func:`confirm_host_switch` asks it when the project holds work for the
current host (:meth:`~mufinder.services.project_service.ProjectService.host_work`):
a job, a site or a saved supercell that belongs to it. In a project without
such work the switch needs no question.
"""

from __future__ import annotations

from uuid import UUID

from PySide6.QtWidgets import QWidget

from mufinder.app import workflow_table_utils
from mufinder.services.project_service import (
    ProjectDocument,
    ProjectService,
    active_host_entry,
    host_entry,
)

#: The answers to the question; keeping the current host is the default.
SWITCH = "switch"
KEEP = "keep"


def confirm_host_switch(
    parent: QWidget,
    service: ProjectService,
    document: ProjectDocument,
    structure_id: UUID,
) -> bool:
    """Return whether to make ``structure_id`` the active host, asking first if there is work.

    Returns ``True`` without asking when there is no active host, when
    ``structure_id`` is already the active host, or when the project holds no
    work for the current host. The question names the new host; its detail
    mentions the current host's sites only when it has some.
    """

    current = active_host_entry(document)
    if current is None or current.structure_id == structure_id:
        return True
    work = service.host_work(document, current.structure_id)
    if not work:
        return True
    new = host_entry(document, structure_id)
    new_label = new.label if new is not None else "the new host"
    detail = [
        "Every later stage will use the new host.",
        "Spins, the propagation vector and saved supercells are kept per host, "
        "so each host keeps its own.",
    ]
    if work.sites:
        detail.append(
            f"Sites generated for '{current.label}' stay in the project and are "
            "marked as belonging to that host."
        )
    answer = workflow_table_utils.confirm_choice(
        parent,
        "Switch active host",
        f"Make '{new_label}' the active host?",
        ((SWITCH, "Switch host"), (KEEP, "Cancel")),
        default=KEEP,
        escape=KEEP,
        detail=" ".join(detail),
    )
    return answer == SWITCH


__all__ = ["KEEP", "SWITCH", "confirm_host_switch"]
