"""Site-generation controller: the Generation page's muon site starts.

``Generate Sites`` adds starts by the chosen method. Manual and UEP starts are
generated at once; random and element-attached starts are computed by a
:class:`~mufinder.app.muon_generation_support.MuonGenerationThread` and added
when it finishes, unless it was cancelled, failed or the project it was started
in is no longer open. ``Remove Selected`` deletes starts and ``Cluster
Positions…`` moves starts to their images nearest an anchor; both are refused
while a run computes sites, from the buttons and the Delete and Backspace keys
alike. The project state
changes are made by ``mufinder.services.site_generation_service`` through
:class:`~mufinder.services.project_service.ProjectService`; this controller asks
the questions, runs and stops the worker thread, shows the messages and keeps
the page in step. It also stores the page's form state in ``ui_state`` and
refuses a host change while a run computes sites for the current host.

A start has no job of its own, so this controller is not a
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`. It keeps
the rest of the controller contract: no reference to ``MainWindow``, and
everything the window owns arrives through :class:`SiteGenerationDeps`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import shiboken6
from PySide6.QtCore import QEventLoop, QObject, Qt, Slot
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from mufinder.domain.models import MuonSite
from mufinder.domain.text import plural
from mufinder.persistence.codec import uuid_list
from mufinder.science.site_generation import SiteGenerationError
from mufinder.services.project_document import ProjectDocument
from mufinder.services.site_generation_service import WORKER_GENERATION_METHODS

from mufinder.app import workflow_table_utils
from mufinder.app.muon_generation_support import MuonGenerationThread
from mufinder.app.muon_generation_widget import MuonGenerationWidget
from mufinder.app.workflow_controllers.base import WorkflowControllerContext


@dataclass(frozen=True)
class SiteGenerationDeps:
    """What the controller needs from the window.

    ``dialog_parent`` parents the message boxes and the closing dialog, and the
    worker thread and its receiver, so both end with the window.
    ``refresh_generation_table`` rebuilds the Generation page's table and
    ``refresh_journal_view`` the Journal after a change. ``shutting_down``
    says whether the window is closing, when a finishing run changes nothing.
    """

    dialog_parent: Callable[[], QWidget]
    refresh_generation_table: Callable[[], None]
    refresh_journal_view: Callable[[], None]
    shutting_down: Callable[[], bool]


class _GenerationReceiver(QObject):
    """Receives one generation worker's signals on the GUI thread.

    Qt picks the connection type from the receiver's thread affinity. A slot on
    this ``QObject``, which lives on the GUI thread, is delivered queued, so the
    handlers never depend on which object Qt would take as the context of a
    plain callable. It holds the run's thread, method and selection mode for
    the controller's handlers. Parented to the window, so a run that ends after
    the window is destroyed finds its connections gone instead of a deleted
    window.
    """

    def __init__(
        self,
        controller: SiteGenerationController,
        thread: MuonGenerationThread,
        method: str,
        mode: str,
        parent: QObject | None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._thread = thread
        self._method = method
        self._mode = mode

    @Slot(object)
    def generated(self, outcome: object) -> None:
        self._controller.generation_finished(self._thread, outcome)

    @Slot(str)
    def failed(self, message: str) -> None:
        self._controller.generation_failed(self._method, message)

    @Slot(str)
    def failed_unexpectedly(self, reason: str) -> None:
        self._controller.generation_failed_unexpectedly(self._method, reason)

    @Slot()
    def cancelled(self) -> None:
        self._controller.generation_cancelled()

    @Slot(int, int)
    def progressed(self, generated: int, expected: int) -> None:
        self._controller.generation_progressed(self._method, generated, expected, self._mode)

    @Slot()
    def thread_finished(self) -> None:
        self._controller.thread_finished()


class SiteGenerationController:
    """Owns generating, removing and clustering the Generation page's starts."""

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: SiteGenerationDeps,
        view: MuonGenerationWidget,
    ) -> None:
        self._context = context
        self._deps = deps
        self._view = view
        self._thread: MuonGenerationThread | None = None
        self._receiver: _GenerationReceiver | None = None
        # The document a running generation was prepared from; its sites are
        # applied only to that document.
        self._document: ProjectDocument | None = None
        self._failed = False

    @property
    def thread(self) -> MuonGenerationThread | None:
        """Return the running generation's worker thread, or ``None``."""

        return self._thread

    def generate(self, method: str, parameters: dict) -> None:
        """Generate starts by ``method``, on a worker thread for random and element-attached."""

        parent = self._deps.dialog_parent()
        document = self._context.document()
        if document is None:
            QMessageBox.information(parent, "No project", "Create or open a project first.")
            return
        if self._thread is not None:
            QMessageBox.information(
                parent, "Generation in progress", "Wait for the current generation task to finish."
            )
            return

        if method in WORKER_GENERATION_METHODS:
            self._start_thread(method, parameters)
            return

        try:
            created = self._context.project_service.generate_candidate_sites(
                document,
                method=method,
                parameters=parameters,
            )
        except (SiteGenerationError, ValueError) as exc:
            QMessageBox.warning(parent, "Generation blocked", str(exc))
            return

        self._context.persist_ui_state(
            failure_message="Muon sites were generated, but autosave failed."
        )
        self._deps.refresh_generation_table()
        self._deps.refresh_journal_view()
        self._finish_status(method, parameters, created)

    def _start_thread(self, method: str, parameters: dict) -> None:
        document = self._context.document()
        if document is None:
            return
        parent = self._deps.dialog_parent()
        try:
            request = self._context.project_service.prepare_candidate_sites(
                document, method=method, parameters=parameters
            )
        except (SiteGenerationError, ValueError) as exc:
            QMessageBox.warning(parent, "Generation blocked", str(exc))
            return
        thread = MuonGenerationThread(self._context.project_service, request, parent=parent)
        self._thread = thread
        self._document = document
        self._failed = False
        random_mode = (
            str(parameters.get("selection_mode", parameters.get("mode", ""))).strip().lower()
        )
        receiver = _GenerationReceiver(self, thread, method, random_mode, parent)
        self._receiver = receiver
        thread.generated.connect(receiver.generated)
        thread.failed.connect(receiver.failed)
        thread.failed_unexpectedly.connect(receiver.failed_unexpectedly)
        thread.cancelled.connect(receiver.cancelled)
        thread.progress_changed.connect(receiver.progressed)
        thread.finished.connect(receiver.thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(receiver.deleteLater)
        method_label = "random" if method == "random" else "near-element"
        self._view.set_generation_busy(True, f"Preparing {method_label} generation…")
        self._context.status_message(f"Generating {method_label} muon sites…")
        thread.start()

    def cancel_generation(self) -> None:
        """Cancel the running generation once the user confirms it."""

        thread = self._thread
        if thread is None or thread.cancel_requested:
            return
        if not workflow_table_utils.confirm_destructive(
            self._deps.dialog_parent(),
            "Cancel generation",
            "Cancel the running muon site generation?",
            "The sites it has generated so far are discarded.",
            accept_label="Cancel generation",
            reject_label="Keep generating",
        ):
            return
        # The run may have finished while the question was open.
        if self._thread is thread:
            thread.cancel()
            self._view.set_generation_progress(0, 1, "Cancelling…")

    def blocks_host_change(self) -> bool:
        """Refuse a host change while a generation computes sites for the current host."""

        return self._refused_while_generating(
            "Muon sites are being generated for the current host structure. Wait for the "
            "generation to finish, or cancel it, before changing the host."
        )

    def _refused_while_generating(self, text: str) -> bool:
        """Say ``text`` and return ``True`` while a run computes sites, else ``False``.

        A failed run computes nothing more, so it refuses nothing while its
        thread winds down; its page controls are back on by then.
        """

        if self._thread is None or self._failed:
            return False
        QMessageBox.information(self._deps.dialog_parent(), "Generation in progress", text)
        return True

    def generation_progressed(self, method: str, generated: int, expected: int, mode: str) -> None:
        """Show a running generation's progress on the page."""

        if method == "random" and mode in {"separation", "distance", "spacing"}:
            message = "Generated %v sites"
        else:
            message = "Generated %v/%m sites"
        self._view.set_generation_progress(
            generated,
            expected,
            message,
        )

    def generation_finished(self, thread: MuonGenerationThread, outcome: object) -> None:
        """Apply a finished worker's sites to the document, on the GUI thread."""

        if self._deps.shutting_down():
            return
        document = self._document
        if document is None or document is not self._context.document():
            self.generation_discarded()
            return
        if thread.cancel_requested:
            self.generation_cancelled()
            return
        method = thread.method
        parameters = thread.parameters
        try:
            sites = self._context.project_service.apply_generated_sites(
                document, thread.request, outcome
            )
        except SiteGenerationError as exc:
            self.generation_failed(method, str(exc))
            return
        self._context.persist_ui_state(
            failure_message="Muon sites were generated, but autosave failed."
        )
        used_spacing: object | None = None
        if method == "random" and sites:
            used_spacing = sites[0].properties.get(
                "min_muon_muon",
                sites[0].properties.get("estimated_min_muon_muon"),
            )
        self._view.set_generation_progress(
            len(sites),
            max(len(sites), 1),
            f"Generated {plural(len(sites), 'site')}",
        )
        self._deps.refresh_generation_table()
        if used_spacing is not None:
            self._view.set_random_muon_muon_distance(float(used_spacing))
        self._deps.refresh_journal_view()
        self._finish_status(method, parameters, sites)

    def generation_cancelled(self) -> None:
        """Report a cancelled run, which added no site."""

        if self._deps.shutting_down():
            return
        if self._document is not self._context.document():
            self.generation_discarded()
            return
        self._view.set_generation_busy(False, "Generation cancelled")
        self._context.status_message("Muon site generation cancelled; no sites were added")

    def generation_discarded(self) -> None:
        """Drop a run started in a project that is no longer open."""

        self._view.set_generation_busy(False, "Ready")
        self._context.status_message(
            "Discarded the muon site generation started in the previously open project"
        )

    def generation_failed(
        self, method: str, message: str, *, title: str = "Generation blocked"
    ) -> None:
        """Say why a run failed, clearing the page's busy state and turning its controls on."""

        self._failed = True
        if self._deps.shutting_down():
            return
        if self._document is not self._context.document():
            self.generation_discarded()
            return
        self._view.set_generation_busy(False, "Generation failed")
        QMessageBox.warning(self._deps.dialog_parent(), title, message)
        method_label = "Random" if method == "random" else "Near-element"
        self._context.status_message(f"{method_label} site generation failed")

    def generation_failed_unexpectedly(self, method: str, reason: str) -> None:
        """Report a run stopped by an error the request's checks did not foresee.

        ``reason`` is the plain text of the error; its traceback is in the log.
        """

        if reason:
            message = f"Muon site generation stopped: {reason}. No sites were added."
        else:
            message = (
                "Muon site generation stopped because of an unexpected error. No sites were added."
            )
        self.generation_failed(method, message, title="Generation failed")

    def thread_finished(self) -> None:
        """Forget the ended run and turn the page's controls back on unless it failed."""

        self._thread = None
        self._receiver = None
        self._document = None
        if not self._failed:
            self._view.set_generation_controls_enabled(True)

    def stop_for_close(self) -> None:
        """Cancel a running generation and wait for its thread while the window closes."""

        thread = self._thread
        if thread is None:
            return
        # The worker only computes points and never touches the document,
        # so closing cancels it and discards its sites. It notices at its
        # next progress report, which a run that rejects many trial
        # positions in a row can take a while to reach, so the event loop
        # runs behind a busy dialog meanwhile; a bare wait() would show
        # the window as not responding. The window modality prevents
        # re-entrant close clicks.
        thread.cancel()
        if not thread.isFinished():
            progress = QProgressDialog(
                "Cancelling muon site generation…", None, 0, 0, self._deps.dialog_parent()
            )
            progress.setWindowTitle("Closing")
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(0)
            progress.show()
            loop = QEventLoop()
            thread.finished.connect(loop.quit)
            if not thread.isFinished():
                loop.exec()
            progress.close()
        # The finished thread's deleteLater may already have run in the loop above;
        # deleting a finishing QThread waits for it, so a deleted one has ended.
        if shiboken6.isValid(thread):
            thread.wait()
        self._thread = None
        self._receiver = None
        self._document = None

    def _finish_status(self, method: str, parameters: dict, created: list[MuonSite]) -> None:
        document = self._context.document()
        requested = (
            self._context.project_service.requested_site_count(
                document, method=method, parameters=parameters
            )
            if document is not None
            else None
        )
        suffix = ""
        if requested is not None and len(created) < requested:
            suffix = f" (requested {requested}; constraints prevented full count)"
        if (
            method == "random"
            and created
            and created[0].properties.get("estimated_min_muon_muon") is not None
        ):
            estimated = float(created[0].properties["estimated_min_muon_muon"])
            used = float(created[0].properties.get("min_muon_muon", estimated))
            if abs(used - estimated) > 1e-9:
                suffix += f" (estimated spacing {estimated:.2f} A; used {used:.2f} A)"
            else:
                suffix += f" (estimated spacing {estimated:.2f} A)"
        self._context.status_message(f"Generated {plural(len(created), 'muon site')}{suffix}")

    def delete_sites(self, raw_site_ids: object) -> None:
        """Remove the starts the page names; other sites are left alone."""

        document = self._context.document()
        if document is None:
            QMessageBox.information(
                self._deps.dialog_parent(), "No project", "Create or open a project first."
            )
            return

        site_ids = uuid_list(raw_site_ids)
        if not site_ids:
            return
        if self._refused_while_generating(
            "Muon sites are being generated. Wait for the generation to finish, or cancel it, "
            "before removing sites."
        ):
            return

        removed = self._context.project_service.remove_candidate_sites(document, site_ids)
        if removed:
            self._context.persist_ui_state(
                failure_message="Muon sites were deleted, but autosave failed."
            )
        self._deps.refresh_generation_table()
        self._deps.refresh_journal_view()
        self._context.status_message(f"Deleted {plural(len(removed), 'generated muon site')}")

    def cluster_positions(self, payload: object) -> None:
        """Move the starts the page names to their images nearest its anchor."""

        parent = self._deps.dialog_parent()
        document = self._context.document()
        if document is None:
            QMessageBox.information(parent, "No project", "Create or open a project first.")
            return
        if not isinstance(payload, dict):
            return
        if self._refused_while_generating(
            "Muon sites are being generated. Wait for the generation to finish, or cancel it, "
            "before clustering positions."
        ):
            return

        site_ids = uuid_list(payload.get("site_ids", []))
        anchor = payload.get("anchor_fractional", [])

        try:
            clustered = self._context.project_service.cluster_candidate_positions(
                document,
                site_ids,
                anchor_fractional=anchor,
            )
        except (SiteGenerationError, ValueError, TypeError) as exc:
            QMessageBox.warning(parent, "Clustering blocked", str(exc))
            return

        clustered_ids = [str(site.id) for site in clustered]
        self._context.persist_ui_state(
            failure_message="Muon positions were clustered, but autosave failed."
        )
        self._deps.refresh_generation_table()
        self._view.select_site_ids(clustered_ids)
        self._deps.refresh_journal_view()
        self._context.status_message(f"Clustered {plural(len(clustered), 'muon position')}")

    def store_form_state(self, state: dict) -> None:
        """Keep the page's form in ``ui_state`` so the project reopens with it."""

        document = self._context.document()
        if document is None or self._deps.shutting_down():
            return
        document.ui_state["muon_generation_form"] = state
