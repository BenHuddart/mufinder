"""Analysis-workspace controller: the Relaxation Analysis workspaces and the Muon Sites summary pin.

A relaxation analysis workspace groups completed muon relaxations of one charge
state for clustering. This controller creates one from the rows chosen on the
Relaxations page, adds relaxations to one, deletes one, stores the page's edits,
records an applied grouping, and pins a workspace's distinct sites to the Muon
Sites summary or removes that pin. It also opens the analysis a pinned site came
from. The project state changes are made in
``mufinder.services.analysis_workspaces`` and
``mufinder.services.muon_sites_summary``; this controller asks the questions,
shows the messages and keeps the pages in step.

A workspace has no job of its own, so this controller is not a
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`. It keeps
the rest of the controller contract: no reference to ``MainWindow``, and
everything the window owns arrives through :class:`AnalysisWorkspacesDeps`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtWidgets import QDialog, QInputDialog, QLineEdit, QMessageBox, QWidget

from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE, muon_charge_state_label
from mufinder.domain.routes import relaxation_analysis_workspace_route
from mufinder.domain.text import plural
from mufinder.services.analysis_workspaces import (
    PinRefusal,
    add_analysis_workspace,
    delete_analysis_workspace,
    find_analysis_workspace,
    pin_analysis_workspace,
    pinned_charge_states,
    record_grouping_applied,
    set_analysis_workspaces,
)
from mufinder.services.muon_sites_summary import unpin_muon_sites, workspace_job_ids
from mufinder.services.source_candidates import (
    CompletedRelaxationAnalysisSource,
    mixes_calculations,
    relaxation_sources_for_labels,
)

from mufinder.app import workflow_table_utils
from mufinder.app.relaxation_analysis_widget import (
    RelaxationAnalysisJobDialog,
    RelaxationAnalysisWidget,
    default_workspace_state,
)
from mufinder.app.workflow_controllers.base import WorkflowControllerContext

#: Navigator route of the Relaxation Analysis page.
_ANALYSIS_ROUTE = "muon_analysis"
#: Navigator route of the Muon Sites summary.
_SUMMARY_ROUTE = "muon_sites"

#: Why a pin was refused, as the message box's title and text.
_PIN_REFUSALS: dict[PinRefusal, tuple[str, str]] = {
    PinRefusal.NOT_CLUSTERED: (
        "Nothing to pin",
        "Run clustering on this workspace first — the Muon Sites summary shows the "
        "distinct clustered muon sites.",
    ),
    PinRefusal.OTHER_HOST: (
        "Cannot pin",
        "This analysis was run against a host structure that is not the active "
        "host. Switch to that host before pinning it to the Muon Sites summary.",
    ),
    PinRefusal.UNSUPPORTED_CHARGE_STATE: (
        "Cannot pin",
        "This workspace's charge state is not supported by the Muon Sites summary.",
    ),
}


@dataclass(frozen=True)
class AnalysisWorkspacesDeps:
    """What the controller needs from the window.

    ``dialog_parent`` parents the name prompt, the relaxation picker and the
    message boxes. ``completed_sources`` lists the completed muon relaxations of
    the active host a workspace can use. ``refresh_analysis_page`` rebuilds the
    Relaxation Analysis page (or marks it stale while hidden).
    ``refresh_navigator`` rebuilds the navigator's rows and seals.
    ``refresh_journal_view``,
    ``refresh_muon_sites_summary`` and ``refresh_pinned_site_consumers`` bring the
    Journal, the summary and the pages that list pinned sites in step.
    ``active_route`` and ``open_route`` read and change the navigator's page.
    ``shutting_down`` is true while the window closes, when the page's edits are
    no longer stored.
    """

    dialog_parent: Callable[[], QWidget]
    completed_sources: Callable[[], list[CompletedRelaxationAnalysisSource]]
    refresh_analysis_page: Callable[[], None]
    refresh_navigator: Callable[[], None]
    refresh_journal_view: Callable[[], None]
    refresh_muon_sites_summary: Callable[[], None]
    refresh_pinned_site_consumers: Callable[[], None]
    active_route: Callable[[], object]
    open_route: Callable[[str], None]
    shutting_down: Callable[[], bool]


class AnalysisWorkspacesController:
    """Owns the relaxation analysis workspaces of the open project and their summary pins."""

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: AnalysisWorkspacesDeps,
        view: RelaxationAnalysisWidget,
    ) -> None:
        self._context = context
        self._deps = deps
        self._view = view

    # --- workspaces -----------------------------------------------------------

    def create_workspace(self, labels: object) -> None:
        """Create a workspace from the completed relaxations of the rows ``labels`` names."""

        document = self._context.document()
        if document is None:
            self._context.show_info("No project", "Create or open a project first.")
            return
        parent = self._deps.dialog_parent()
        host = self._context.project_service.active_host_structure(document)
        if host is None:
            QMessageBox.warning(
                parent,
                "No host structure",
                "Load a host structure before creating an analysis workspace.",
            )
            return
        label_list = [str(label) for label in labels] if isinstance(labels, list) else [str(labels)]
        sources = relaxation_sources_for_labels(self._deps.completed_sources(), label_list)
        if not sources:
            self._context.show_info(
                "No completed relaxations",
                "Select completed, converged structural relaxation rows before creating an analysis workspace.",
            )
            return
        charge_states = {source.charge_state for source in sources}
        if len(charge_states) > 1:
            QMessageBox.warning(
                parent,
                "Mixed charge states",
                "Create separate analysis workspaces for positive muon and muonium relaxation results.",
            )
            return

        if not self._accept_mixed_calculations(sources):
            return

        charge_state = next(iter(charge_states))
        default_name = f"{muon_charge_state_label(charge_state)} analysis"
        name, accepted = QInputDialog.getText(
            parent,
            "Create Analysis Workspace",
            "Workspace name",
            QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not accepted:
            return

        workspace = default_workspace_state(
            name=name,
            charge_state=charge_state,
            host_structure_id=host.id,
            job_ids=[source.job_id for source in sources],
        )
        recorded = add_analysis_workspace(document, workspace)
        # The Journal now names the workspace, so save it with its record. A
        # record that could not be written does not cost the workspace: it is
        # saved and opened all the same, and the user told.
        self._context.persist_ui_state(
            failure_message="The analysis workspace was created, but autosave failed."
        )
        self._deps.refresh_analysis_page()
        self._deps.refresh_navigator()
        self._deps.refresh_journal_view()
        self._deps.open_route(relaxation_analysis_workspace_route(str(workspace["id"])))
        if not recorded:
            QMessageBox.warning(
                parent,
                "Journal not updated",
                f"The analysis workspace '{workspace['name']}' was created, but its Journal "
                "record could not be written.",
            )
        self._context.status_message(f"Created analysis workspace '{workspace['name']}'")

    def store_workspaces(self, workspaces: list) -> None:
        """Store the page's edited workspaces and refresh the navigator's labels and seals."""

        document = self._context.document()
        if document is None or self._deps.shutting_down():
            return
        cleaned = [dict(workspace) for workspace in workspaces if isinstance(workspace, dict)]
        set_analysis_workspaces(document, cleaned)
        self._deps.refresh_navigator()

    def delete_workspace(self, workspace_id: str) -> None:
        """Delete one analysis workspace, keeping the relaxations it drew on.

        A preliminary analysis made before every relaxation finished goes stale
        once the full set is in; deleting it only drops the workspace and its
        clustering. A Muon Sites summary pin taken from it is removed as well,
        so the summary never keeps showing sites from an analysis that is gone.
        """

        document = self._context.document()
        if document is None:
            return
        target = find_analysis_workspace(document, workspace_id)
        if target is None:
            return
        name = str(target.get("name", "")).strip() or "Relaxation analysis"
        detail = (
            "Its clustering results are discarded. The structural relaxations it "
            "uses are kept and can be added to another analysis."
        )
        if pinned_charge_states(document, workspace_id):
            detail += " Its sites are pinned to the Muon Sites summary; that pin is removed too."
        if not workflow_table_utils.confirm_destructive(
            self._deps.dialog_parent(),
            "Delete analysis",
            f"Delete analysis '{name}'?",
            detail,
            accept_label="Delete",
        ):
            return
        unpinned = delete_analysis_workspace(document, workspace_id)
        if self._deps.active_route() == relaxation_analysis_workspace_route(str(workspace_id)):
            # Step up to the Analysis node first: rebuilding the tree without
            # the selected node would otherwise fall back to the home page.
            self._deps.open_route(_ANALYSIS_ROUTE)
        self._deps.refresh_analysis_page()
        self._deps.refresh_navigator()
        if unpinned:
            self._deps.refresh_muon_sites_summary()
            self._deps.refresh_pinned_site_consumers()
        self._context.persist_ui_state(
            failure_message="The analysis was deleted, but autosave failed."
        )
        self._context.status_message(f"Deleted analysis '{name}'")

    def add_relaxations(self, workspace_id: str) -> None:
        """Ask for completed relaxations of the workspace's charge state and add them to it."""

        document = self._context.document()
        if document is None:
            return
        workspace = find_analysis_workspace(document, workspace_id)
        if workspace is None:
            return
        charge_state = str(workspace.get("charge_state", POSITIVE_MUON_CHARGE_STATE))
        existing_ids = set(workspace_job_ids(workspace))
        sources = [
            source
            for source in self._deps.completed_sources()
            if source.charge_state == charge_state
        ]
        if not sources:
            self._context.show_info(
                "No matching relaxations",
                f"No completed {muon_charge_state_label(charge_state)} relaxation results are available to add.",
            )
            return
        dialog = RelaxationAnalysisJobDialog(
            title="Add Structure to Analysis",
            sources=sources,
            excluded_job_ids=existing_ids,
            reference_job_ids=existing_ids,
            parent=self._deps.dialog_parent(),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_job_ids()
        if not selected:
            return
        joined = [
            source
            for source in sources
            if source.job_id in existing_ids or source.job_id in set(selected)
        ]
        if not self._accept_mixed_calculations(joined):
            return
        self._view.add_job_ids_to_workspace(str(workspace_id), selected)
        self._context.status_message(
            f"Added {plural(len(selected), 'relaxation result')} to the analysis workspace"
        )

    def record_grouping(self, workspace_id: str) -> None:
        """Record the grouping just applied to a workspace in the Journal and save it."""

        document = self._context.document()
        if document is None:
            return
        if not record_grouping_applied(document, workspace_id):
            return
        # A repeat adds no record, but the grouping it applied is still saved.
        self._context.persist_ui_state(
            failure_message="The grouping was applied, but autosave failed."
        )
        self._deps.refresh_journal_view()

    def _accept_mixed_calculations(self, sources: list[CompletedRelaxationAnalysisSource]) -> bool:
        """Return whether to go on with relaxations from more than one calculation.

        Total energies are only comparable between relaxations in the same
        supercell with the same settings, so mixing them asks first, with
        ``Cancel`` the default.
        """

        if not mixes_calculations(sources):
            return True
        answer = workflow_table_utils.confirm_choice(
            self._deps.dialog_parent(),
            "Different calculations",
            "These relaxations ran in different supercells or with different settings "
            "(cutoff, k-points, functional, Hubbard U or spin), so their energies are not comparable "
            "and the workspace's dE values would be meaningless.",
            (("add", "Use them anyway"), ("cancel", "Cancel")),
            default="cancel",
            escape="cancel",
            icon=QMessageBox.Icon.Warning,
        )
        return answer == "add"

    # --- Muon Sites summary pins -----------------------------------------------

    def pin_to_summary(self, workspace_id: str) -> None:
        """Pin a workspace's distinct sites to the Muon Sites summary and open it."""

        document = self._context.document()
        if document is None:
            return
        result = pin_analysis_workspace(document, workspace_id)
        if result is None:
            return
        if result.refusal is not None:
            title, text = _PIN_REFUSALS[result.refusal]
            self._context.show_info(title, text)
            return
        charge_state = str(result.charge_state)
        self._deps.refresh_muon_sites_summary()
        self._deps.refresh_pinned_site_consumers()
        self._deps.open_route(_SUMMARY_ROUTE)
        self._context.persist_ui_state(
            failure_message="The muon sites summary was pinned, but autosave failed."
        )
        self._context.status_message(
            f"Pinned {muon_charge_state_label(charge_state)} sites to the Muon Sites summary"
        )

    def unpin(self, charge_state: str) -> None:
        """Remove ``charge_state``'s pin from the Muon Sites summary."""

        document = self._context.document()
        if document is None:
            return
        key = str(charge_state)
        if not unpin_muon_sites(document, key):
            return
        self._deps.refresh_muon_sites_summary()
        self._deps.refresh_pinned_site_consumers()
        self._context.persist_ui_state(
            failure_message="The muon sites summary was updated, but autosave failed."
        )
        self._context.status_message(
            f"Unpinned {muon_charge_state_label(key)} sites from the Muon Sites summary"
        )

    def open_pinned_analysis(self, workspace_id: str) -> None:
        """Open the analysis workspace a pin was taken from."""

        self._open_pinned(str(workspace_id), None)

    def focus_pinned_site(self, workspace_id: str, cluster_index: int) -> None:
        """Open the analysis workspace a pinned site came from, on that site's cluster."""

        self._open_pinned(str(workspace_id), int(cluster_index))

    def _open_pinned(self, workspace_id: str, cluster_index: int | None) -> None:
        self._deps.open_route(_ANALYSIS_ROUTE)
        resolved = self._view.focus_workspace(str(workspace_id), cluster_index=cluster_index)
        if not resolved:
            self._context.status_message(
                "The pinned analysis has changed — opened the Analysis tab instead"
            )


__all__ = ["AnalysisWorkspacesController", "AnalysisWorkspacesDeps"]
