"""Comparison-table controller: generate, rename and delete saved comparison tables.

A comparison table is a saved document item under Pristine Relaxation in the
navigator, with no job of its own, so this controller is not a
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`. It keeps
the rest of the controller contract: no reference to ``MainWindow``, no dialogs
of its own, and everything the window owns arrives through
:class:`ComparisonTablesDeps`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from mufinder.services.source_candidates import CompletedPristineJob
from mufinder.services.structure_comparison import build_comparison_table, comparison_inputs

from mufinder.app.comparison_table_widget import ComparisonTablesPage
from mufinder.app.workflow_controllers.base import WorkflowControllerContext

_ROUTE_PREFIX = "pristine_relaxation/comparison/"
_PARENT_ROUTE = "pristine_relaxation"


def comparison_route(table_id: UUID) -> str:
    """The navigator route of one comparison table."""

    return f"{_ROUTE_PREFIX}{table_id}"


def comparison_table_id_from_route(route: object) -> UUID | None:
    """The table a navigator route names, or ``None`` for any other route."""

    if not isinstance(route, str) or not route.startswith(_ROUTE_PREFIX):
        return None
    try:
        return UUID(route[len(_ROUTE_PREFIX) :].strip())
    except ValueError:
        return None


@dataclass(frozen=True)
class ComparisonTablesDeps:
    """What the controller needs from the window and the Pristine Relaxation page.

    ``completed_jobs`` lists the relaxations a table can compare.
    ``choose_jobs`` opens the job dialog with these jobs and a default label and
    returns the chosen job ids and label, or ``None`` when it is cancelled.
    ``show_warning`` shows a message box. ``refresh_project_tree``,
    ``refresh_journal_view``, ``active_route`` and ``open_route`` keep the
    navigator and journal in step. ``shutting_down`` is true while the window
    closes, when a title edit losing focus must not save the project.
    """

    completed_jobs: Callable[[], list[CompletedPristineJob]]
    choose_jobs: Callable[[list[CompletedPristineJob], str], tuple[list[UUID], str] | None]
    show_warning: Callable[[str, str], None]
    refresh_project_tree: Callable[[], None]
    refresh_journal_view: Callable[[], None]
    active_route: Callable[[], object]
    open_route: Callable[[str], None]
    shutting_down: Callable[[], bool]


class ComparisonTablesController:
    """Owns the saved comparison tables of the open project and their page."""

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: ComparisonTablesDeps,
        view: ComparisonTablesPage,
    ) -> None:
        self._context = context
        self._deps = deps
        self._view = view

    def sync_view(self) -> None:
        """Rebuild the page from the tables of the open project."""

        document = self._context.document()
        self._view.set_tables(() if document is None else document.comparison_tables.values())

    def generate_table(self) -> None:
        """Ask for the jobs to compare, then save, open and record the new table."""

        document = self._context.document()
        if document is None:
            self._context.show_info("No project", "Create or open a project first.")
            return

        service = self._context.project_service
        host = service.active_host_structure(document)
        if host is None:
            self._deps.show_warning(
                "No host structure",
                "Load a host structure before generating a comparison table.",
            )
            return

        completed_jobs = self._deps.completed_jobs()
        if not completed_jobs:
            self._context.show_info(
                "No completed pristine relaxations",
                "Run and complete one or more pristine-relaxation jobs first.",
            )
            return

        choice = self._deps.choose_jobs(
            completed_jobs, f"Comparison {len(document.comparison_tables) + 1}"
        )
        if choice is None:
            return
        job_ids, label = choice
        selected = set(job_ids)
        jobs = [job for job in completed_jobs if job.job_id in selected]

        inputs = comparison_inputs(document, jobs)
        if not inputs:
            self._deps.show_warning(
                "Comparison unavailable",
                "The selected jobs did not provide relaxed structures for comparison.",
            )
            return

        try:
            table = build_comparison_table(label=label, host=host, inputs=inputs, jobs=jobs)
        except ValueError as exc:
            self._deps.show_warning("Comparison failed", str(exc))
            return

        service.add_comparison_table(document, table)
        service.record_comparison_table_created(document, table=table)

        self.sync_view()
        self._deps.refresh_project_tree()
        self._deps.open_route(comparison_route(table.id))
        self._deps.refresh_journal_view()
        self._context.status_message(f"Generated comparison table '{table.label}'")

    def rename_table(self, table_id: UUID, title: str) -> None:
        """Store a new title, keep the table open and save the project."""

        document = self._context.document()
        if document is None or self._deps.shutting_down():
            return
        service = self._context.project_service
        table = service.rename_comparison_table(document, table_id, title)
        if table is None:
            return
        self._view.set_table_label(table_id, table.label)
        self._deps.refresh_project_tree()
        self._deps.open_route(comparison_route(table_id))
        try:
            service.save_project(document)
        except ValueError:
            self._context.status_message("Comparison table title was updated, but autosave failed")
            return
        self._context.status_message(f"Renamed comparison table to '{table.label}'")

    def delete_table(self, table_id: UUID) -> None:
        """Delete a table the user has already confirmed deleting.

        The table's own Delete button is the only entry point, and the table
        widget asks for the confirmation before it emits ``delete_requested``.
        """

        document = self._context.document()
        if document is None or table_id not in document.comparison_tables:
            return

        self._context.project_service.remove_comparison_table(document, table_id)
        self.sync_view()
        self._deps.refresh_project_tree()

        route = self._deps.active_route()
        if route is None or comparison_table_id_from_route(route) is not None:
            self._deps.open_route(_PARENT_ROUTE)

        self._context.status_message("Deleted comparison table")


__all__ = [
    "ComparisonTablesController",
    "ComparisonTablesDeps",
    "comparison_route",
    "comparison_table_id_from_route",
]
