"""Imported-relaxations controller: the File menu's import and removal of CASTEP relaxations.

``File > Import CASTEP Relaxations…`` scans a folder of finished CASTEP geometry
optimisations on a worker thread, reviews the runs in
:class:`~mufinder.app.relaxation_import_dialog.RelaxationImportDialog` and
commits the chosen ones as read-only muon-relaxation rows.
``File > Remove Imported Relaxations…`` removes one import batch, or explains
why it cannot. The services build and remove the entities and files
(``mufinder.services.relaxation_import_commit`` and
``mufinder.services.relaxation_import_removal``); this controller asks the
questions, attaches the rows to the muon-relaxation definition through the
live widget and reports the outcome.

An import has no job of its own to submit, so this controller is not a
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`. It keeps
the rest of the controller contract: no reference to ``MainWindow``, and
everything the window owns arrives through :class:`ImportedRelaxationsDeps`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import TypeVar

from PySide6.QtCore import QObject, Slot
from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox, QWidget

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Structure
from mufinder.domain.text import KeptFolder, folder_outcome_status, plain_error, plural
from mufinder.services.definition_mappings import ensure_job_mapping
from mufinder.services.job_file_cleanup import KEPT_AS_NOT_ITS_OWN
from mufinder.services.job_paths import job_directory_label
from mufinder.services.relaxation_import import RelaxationFolderScan
from mufinder.services.relaxation_import_commit import (
    RelaxationImportCancelled,
    RelaxationImportFailed,
    StagedRelaxationImport,
    build_relaxation_import,
    discard_staged_import,
    stage_relaxation_import,
)
from mufinder.services.relaxation_import_removal import (
    NO_JOB,
    NOT_IMPORTED,
    ImportedRelaxationRef,
    LeftSite,
    batch_contents,
    detach_rows_from_definition,
    import_batches,
    plan_removal,
    remove_imported_relaxations,
)
from mufinder.workflow.muon_site import ACTIVE_MUON_RELAXATION_DEFINITION_KEY

from mufinder.app import blocking_call, workflow_table_utils
from mufinder.app.cluster_ui_utils import show_detailed_warning
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.relaxation_adapter import muon_charge_state_prefix
from mufinder.app.relaxation_import_dialog import (
    RelaxationImportDialog,
    choose_import_batch,
    start_relaxation_scan,
    stop_relaxation_scans,
)
from mufinder.app.workflow_config_codecs import deserialize_muon_relaxation_config
from mufinder.app.workflow_controllers.base import WorkflowControllerContext

#: Why a batch's folder stays, for one folder and for several.
_KEPT_FOR_ANOTHER_IMPORT = ("another import uses it", "other imports use them")
#: How long a cancelled import may take to stop and remove what it copied.
_CANCEL_CLEAN_UP_SECONDS = 120.0
#: How many paths a message about an undone import names before it gives a count.
_LEFT_BEHIND_SHOWN = 5
#: How many relaxations a removal's confirmation names in its text; Details lists them all.
_NAMES_INLINE = 3
#: Why a removal leaves a site of its batch, for one site and for several.
_WHY_LEFT = {
    NOT_IMPORTED: (
        "its job is no longer an imported one",
        "their jobs are no longer imported ones",
    ),
    NO_JOB: ("it has no job", "they have no job"),
}

_Named = TypeVar("_Named", ImportedRelaxationRef, LeftSite)


def _inline_names(labels: Sequence[str]) -> str:
    """Name up to a few labels in a sentence, counting the rest (``A, B, C and 9 more``)."""

    if len(labels) > _NAMES_INLINE:
        return f"{', '.join(labels[:_NAMES_INLINE])} and {len(labels) - _NAMES_INLINE} more"
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def import_failure_text(error: RelaxationImportFailed, folder: Path) -> str:
    """Say why an import's file step failed: reading a run's file, or writing into the project."""

    reason = plain_error(error.error)
    if error.source is not None:
        try:
            name = error.source.relative_to(folder).as_posix()
        except ValueError:
            name = error.source.name
        return f"Could not read '{name}' from the chosen folder: {reason}. Nothing was imported."
    return f"Could not write the imported files into the project: {reason}. Nothing was imported."


@dataclass(frozen=True)
class ImportedRelaxationsDeps:
    """What the controller needs from the window and the Muon Relaxation page.

    ``dialog_parent`` parents the folder picker, the review dialog and the
    message boxes. ``set_import_enabled`` enables or disables the File menu's
    import action, which stays off while a scan runs. ``clear_status`` clears
    the status bar. ``active_muon_relaxation_definition`` is the definition the
    imported rows join. ``refresh_muon_relaxation_rows``,
    ``refresh_jobs_table`` and ``refresh_muon_sites_summary`` bring the pages
    in step after an import or a removal. ``show_kept_folders`` says why
    folders stayed after a removal.
    """

    dialog_parent: Callable[[], QWidget]
    set_import_enabled: Callable[[bool], None]
    clear_status: Callable[[], None]
    active_muon_relaxation_definition: Callable[[], CalculationDefinition | None]
    refresh_muon_relaxation_rows: Callable[[], None]
    refresh_jobs_table: Callable[[], None]
    refresh_muon_sites_summary: Callable[[], None]
    show_kept_folders: Callable[[Sequence[KeptFolder]], None]


class _ScanReceiver(QObject):
    """Receives the scan worker's signals on the GUI thread.

    Qt picks the connection type from the receiver's thread affinity. A slot on
    this ``QObject``, which lives on the GUI thread, is delivered queued; a
    plain bound method of the controller has no receiver object and would run
    on the worker thread. Parented to the window, so a scan that ends after the
    window is destroyed finds its connections gone instead of a deleted window.
    """

    def __init__(self, controller: ImportedRelaxationsController, parent: QObject | None) -> None:
        super().__init__(parent)
        self._controller = controller

    @Slot(int, int)
    def progressed(self, done: int, total: int) -> None:
        self._controller.scan_progressed(done, total)

    @Slot(str)
    def failed(self, message: str) -> None:
        self._controller.scan_failed(message)

    @Slot(object)
    def completed(self, scan: object) -> None:
        self._controller.scan_completed(scan)


@dataclass(frozen=True)
class _ScanOrigin:
    """The project and host a scan was started for, kept to check its result is still wanted.

    ``document`` is the open project's document object itself: reopening the
    same folder loads a new one, which is a different project to this scan.
    ``lattice`` and ``species`` are copies of what the scan compared each run
    against, so an edit to the host made while it ran is noticed too.
    """

    document: object
    host: Structure
    lattice: tuple[tuple[float, ...], ...]
    species: tuple[str, ...]


def _host_signature(host: Structure) -> tuple[tuple[tuple[float, ...], ...], tuple[str, ...]]:
    return (
        tuple(tuple(float(value) for value in vector) for vector in host.lattice_vectors),
        tuple(host.species),
    )


class ImportedRelaxationsController:
    """Owns importing CASTEP relaxations into the open project and removing them."""

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: ImportedRelaxationsDeps,
        view: MuonRelaxationWidget,
    ) -> None:
        self._context = context
        self._deps = deps
        self._view = view
        self._scan_root: Path | None = None
        self._scan_origin: _ScanOrigin | None = None
        # While the window is closing, a scan's ending waits here: shown if the
        # close is cancelled, dropped with the window otherwise.
        self._closing = False
        self._held_ending: Callable[[], None] | None = None
        # Set once the close went ahead and stopped the scan: nothing ends after that.
        self._stopped = False
        self._receiver = _ScanReceiver(self, deps.dialog_parent())

    @property
    def scan_in_progress(self) -> bool:
        """Return whether a folder scan is running."""

        return self._scan_root is not None

    def import_relaxations(self) -> None:
        """Scan a folder for CASTEP relaxations and import the ones chosen from it."""

        parent = self._deps.dialog_parent()
        document = self._context.document()
        if document is None:
            QMessageBox.information(
                parent,
                "No project open",
                "Open or create a project before importing CASTEP relaxations.",
            )
            return

        host = self._context.project_service.active_host_structure(document)
        if host is None:
            # Deliberately not offered as "import anyway": every imported
            # structure must hang off the active host, or its downstream jobs
            # fail host resolution at submission time.
            QMessageBox.information(
                parent,
                "No host structure",
                "Import a host structure first. Imported relaxations are anchored to "
                "the project's active host so the sites they create can be used "
                "downstream.",
            )
            return

        directory = QFileDialog.getExistingDirectory(
            parent, "Select a folder containing CASTEP relaxations"
        )
        if not directory:
            return

        self._begin_scan(Path(directory), host)

    def _begin_scan(self, root: Path, host: Structure) -> None:
        """Scan ``root`` on a worker thread, then review and commit."""

        self._context.status_message(f"Scanning {root.name} for CASTEP relaxations…")
        self._deps.set_import_enabled(False)
        self._remember_scan(root, host)
        start_relaxation_scan(
            self._receiver,
            root,
            [list(vector) for vector in host.lattice_vectors],
            list(host.species),
            self._receiver.completed,
            self._receiver.failed,
            self._receiver.progressed,
        )

    def _remember_scan(self, root: Path, host: Structure) -> None:
        """Record the folder, the open project and the host a scan is for."""

        lattice, species = _host_signature(host)
        self._scan_root = root
        self._scan_origin = _ScanOrigin(self._context.document(), host, lattice, species)

    def scan_progressed(self, done: int, total: int) -> None:
        """Show how far the scan has got in the status bar."""

        if self._closing:
            return
        root = self._scan_root
        name = root.name if root is not None else "folder"
        self._context.status_message(f"Scanning {name}: relaxation {done} of {total}…")

    def scan_failed(self, message: str) -> None:
        """End the scan and say why it failed."""

        if self._closing:
            if not self._stopped:
                self._held_ending = lambda: self.scan_failed(message)
            return
        origin, root = self._scan_origin, self._scan_root
        self.finish_scan()
        if self._told_discarded(origin, root):
            return
        QMessageBox.warning(
            self._deps.dialog_parent(), "Import failed", f"Could not scan the folder:\n{message}"
        )

    def scan_completed(self, scan: object) -> None:
        """End the scan, review its runs and commit the ones chosen.

        A scan whose project or host changed while it ran is discarded, with a
        word to the user: it compared the runs against a host that is no longer
        the one they would be anchored to, and the import would land in
        whichever project is open now.
        """

        if self._closing:
            if not self._stopped:
                self._held_ending = lambda: self.scan_completed(scan)
            return
        origin, root = self._scan_origin, self._scan_root
        self.finish_scan()
        if self._told_discarded(origin, root):
            return
        parent = self._deps.dialog_parent()
        if origin is None or not isinstance(scan, RelaxationFolderScan):
            return
        host = origin.host

        if not scan.runs:
            if scan.unreadable:
                QMessageBox.information(parent, "Could not read folder", scan.unreadable_note)
                return
            QMessageBox.information(
                parent,
                "Nothing to import",
                "No CASTEP outputs (.castep files) were found in or under that folder.",
            )
            return

        dialog = RelaxationImportDialog(scan, parent=parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_runs()
        if not selected:
            return
        # The review is modal, so this only guards a change made some other way.
        if self._told_discarded(origin, root):
            return

        self.commit_import(scan, selected, host)

    def finish_scan(self) -> None:
        """Forget the running scan and turn the import action back on if a project is open."""

        self._forget_scan()
        self._deps.set_import_enabled(self._context.document() is not None)
        self._deps.clear_status()

    def hold_for_close(self) -> None:
        """Hold back a scan's ending while the window closes.

        Called before the close asks whether to save, because that question
        runs an event loop in which the scan could end and open its review over
        a closing project. A held ending is shown if the close is cancelled
        (:meth:`resume_after_cancelled_close`); if the close goes ahead,
        :meth:`stop_for_close` stops the scan, waits for its thread and drops it.
        """

        self._closing = True

    def stop_for_close(self) -> None:
        """Stop a running scan once the window is closing for good, and wait for its thread.

        Called after the save question was answered. The scan stops before the
        next run it would read; anything it already reported is dropped with the
        held ending. See :func:`stop_relaxation_scans` for what bounds the wait.
        """

        self._stopped = True
        stop_relaxation_scans(self._deps.dialog_parent())
        self._held_ending = None
        self._forget_scan()

    def resume_after_cancelled_close(self) -> None:
        """Let a scan end as usual again, showing one that ended while the close was asked."""

        self._closing = False
        ending, self._held_ending = self._held_ending, None
        if ending is not None:
            ending()

    def _forget_scan(self) -> None:
        self._scan_root = None
        self._scan_origin = None

    def _told_discarded(self, origin: _ScanOrigin | None, root: Path | None) -> bool:
        """Say once why a scan's result is dropped, and return whether it was."""

        if origin is None or root is None:
            return False
        reason = self._discard_reason(origin)
        if reason is None:
            return False
        QMessageBox.information(
            self._deps.dialog_parent(),
            "Scan discarded",
            f"The scan of '{root.name}' was discarded because {reason}",
        )
        return True

    def _discard_reason(self, origin: _ScanOrigin) -> str | None:
        """Return why a scan no longer fits the open project, or ``None`` when it still does."""

        document = self._context.document()
        if document is None:
            return "the project was closed while it ran. Nothing was imported."
        if document is not origin.document:
            return (
                "a different project was opened while it ran. Nothing was imported. Start the "
                "import again from the project you want to import into."
            )
        host = self._context.project_service.active_host_structure(document)
        if (
            host is None
            or host.id != origin.host.id
            or _host_signature(host) != (origin.lattice, origin.species)
        ):
            return (
                "the project's host structure changed while it ran, and the runs were checked "
                "against the previous host. Nothing was imported. Start the import again to "
                "check them against the current host."
            )
        return None

    def commit_import(self, scan: RelaxationFolderScan, runs: list, host: Structure) -> None:
        """Copy the files off the GUI thread, then build the entities, attach the rows and save.

        Nothing reaches the document until every copy has finished; a copy that
        fails or is cancelled leaves no file, entity, row or provenance record.
        """

        parent = self._deps.dialog_parent()
        document = self._context.document()
        if document is None:
            return

        output_root = document.project_root / "files" / WorkflowStage.MUON_RELAXATION.value
        definition = self._deps.active_muon_relaxation_definition()
        existing_labels: list[str] = []
        if definition is not None:
            config = deserialize_muon_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if config is not None:
                existing_labels = [row.label for row in config.rows]

        staged = self._copy_files(scan, runs, output_root, existing_labels)
        if staged is None:
            return
        try:
            outcome = build_relaxation_import(document, staged, host_structure=host)
        except BaseException:
            # Building adds every entity or none, so only the files remain to undo.
            discard_staged_import(staged)
            raise

        if not self.merge_imported_relaxation_rows(outcome, output_directory=output_root):
            QMessageBox.warning(
                parent,
                "Import incomplete",
                "The relaxations were imported but could not be attached to the relaxations table.",
            )
            return

        # Only once the merge has succeeded: append_provenance writes straight to
        # provenance.jsonl, so recording the import before the rows were attached
        # would leave a durable claim about a state the project never reached.
        if outcome.provenance is not None:
            self._context.project_service.append_provenance(document, outcome.provenance)

        self._deps.refresh_muon_relaxation_rows()
        self._deps.refresh_jobs_table()
        self._deps.refresh_muon_sites_summary()
        self._context.persist_ui_state(
            failure_message="CASTEP relaxations were imported, but autosave failed.",
        )

        messages = [f"Imported {plural(len(outcome.imported), 'relaxation')}."]
        if outcome.skipped:
            messages.append(f"{plural(len(outcome.skipped), 'run was', 'runs were')} skipped.")
        row_warnings = [note for item in outcome.imported for note in item.warnings]
        if row_warnings:
            messages.append("")
            messages.extend(f"- {note}" for note in dict.fromkeys(row_warnings))
        QMessageBox.information(parent, "Import complete", "\n".join(messages))
        self._context.status_message(
            f"Imported {plural(len(outcome.imported), 'CASTEP relaxation')}"
        )

    def _copy_files(
        self,
        scan: RelaxationFolderScan,
        runs: list,
        output_root: Path,
        existing_labels: list[str],
    ) -> StagedRelaxationImport | None:
        """Copy the runs' files into the project off the GUI thread.

        Returns the staged import, or ``None`` when nothing was imported: the
        copy failed or was cancelled, and either way the files it had copied
        are gone again and the user has been told.
        """

        parent = self._deps.dialog_parent()
        cancel = threading.Event()
        finished = threading.Event()
        # What the worker returned, kept here too: run_blocking_call drops the
        # result of a call cancelled just as it finished, and its files must go.
        delivered: list[StagedRelaxationImport] = []
        # What a cancelled worker could not remove: run_blocking_call drops its error.
        left_behind: list[Path] = []

        def _stage() -> StagedRelaxationImport:
            try:
                staged = stage_relaxation_import(
                    scan=scan,
                    runs=runs,
                    output_root=output_root,
                    existing_labels=existing_labels,
                    should_cancel=cancel.is_set,
                )
                if cancel.is_set():
                    # Cancelled during the last file, perhaps after the wait
                    # below gave up: nobody will take these files.
                    raise RelaxationImportCancelled(left_behind=discard_staged_import(staged))
                delivered.append(staged)
                return staged
            except RelaxationImportCancelled as exc:
                left_behind.extend(exc.left_behind)
                raise
            finally:
                finished.set()

        try:
            return blocking_call.run_blocking_call(
                parent,
                _stage,
                title="Import CASTEP relaxations",
                description=f"Copying {plural(len(runs), 'relaxation')} into the project…",
                cancellable=True,
                cancel_event=cancel,
            )
        except blocking_call.BlockingCallCancelled:
            # The worker stops before its next file and removes what it copied.
            stopped = blocking_call.wait_for_event(
                parent,
                finished,
                _CANCEL_CLEAN_UP_SECONDS,
                title="Import CASTEP relaxations",
                description="Removing the copied files…",
            )
            for staged in delivered:
                left_behind.extend(discard_staged_import(staged))
            message = "Nothing was imported."
            if not stopped:
                message += (
                    " The files copied so far are removed once the file being copied is done."
                )
            QMessageBox.information(
                parent,
                "Import cancelled",
                message + self._left_behind_text(left_behind, output_root),
            )
        except RelaxationImportFailed as exc:
            QMessageBox.warning(
                parent,
                "Import failed",
                import_failure_text(exc, scan.root)
                + self._left_behind_text(exc.left_behind, output_root),
            )
        return None

    def _left_behind_text(self, paths: Sequence[Path], output_root: Path) -> str:
        """Say what an undone import could not remove, or nothing when it removed everything."""

        if not paths:
            return ""
        document = self._context.document()
        root = document.project_root if document is not None else output_root
        names = [job_directory_label(root, path) for path in sorted(paths)]
        shown = names[:_LEFT_BEHIND_SHOWN]
        lines = "\n".join(f"- {name}" for name in shown)
        more = len(names) - len(shown)
        if more:
            lines += f"\n- and {more} more ({len(names)} in all)"
        return (
            "\n\nMuFinder could not remove everything this import had created, so "
            f"{'this is' if len(names) == 1 else 'these are'} still in the project:\n{lines}"
        )

    def remove_imported_relaxations(self) -> None:
        """Remove one import batch, or explain why it cannot be removed.

        An imported relaxation owns the only copy of its results, so this is
        irreversible in a way that removing a natively-run relaxation is not:
        that one can be re-run. The reference check therefore refuses rather
        than warns, and the confirmation names what will be deleted.
        """

        parent = self._deps.dialog_parent()
        document = self._context.document()
        if document is None:
            QMessageBox.information(parent, "No project open", "Open a project first.")
            return

        batches = import_batches(document)
        if not batches:
            QMessageBox.information(
                parent,
                "Nothing imported",
                "This project contains no imported relaxations.",
            )
            return

        labels = [
            f"{label} ({plural(count, 'relaxation') if count else 'nothing to remove'})"
            for label, count, _batch_id in batches
        ]
        chosen = choose_import_batch(parent, labels)
        if chosen is None:
            return
        batch_id = batches[chosen][2]

        contents = batch_contents(document, batch_id)
        refs = list(contents.targets)
        left = self._in_row_order(contents.left)
        if not refs:
            self._warn_with_lines(
                "Nothing to remove",
                "Nothing in this import can be removed, so nothing was removed.",
                [f"{site.label}: {_WHY_LEFT[site.reason][0]}" for site in left],
            )
            return
        # A relaxation whose job is gone has no folder MuFinder can call its own.
        directories = {
            ref.job_id: path
            for ref in refs
            if (job := document.jobs.get(ref.job_id)) is not None
            and (path := self._context.job_output_directory(job)) is not None
        }
        others = {
            job.id: path
            for job in document.jobs.values()
            if job.id not in directories
            and (path := self._context.job_output_directory(job)) is not None
        }
        plan = plan_removal(
            document, refs, job_directories=directories, other_job_directories=others
        )

        if plan.unsafe:
            self._warn_with_lines(
                "Cannot remove safely",
                f"MuFinder will not delete the folders of {plural(len(plan.unsafe), 'relaxation')}, "
                "so nothing was removed.",
                plan.unsafe,
            )
            return

        if plan.blockers:
            self._warn_with_lines(
                "Still in use",
                "These imported relaxations cannot be removed because other work uses them. "
                "Remove that first.",
                plan.blockers,
            )
            return

        orphaned = self._in_row_order(
            [ref for ref in plan.targets if ref.job_id in plan.orphaned_job_ids]
        )
        if not workflow_table_utils.confirm_destructive(
            parent,
            "Remove imported relaxations?",
            self._removal_question(plan)
            + self._orphaned_text([ref.label for ref in orphaned])
            + self._left_in_batch_text(left),
            "MuFinder did not run these calculations and cannot recreate them.",
            accept_label=(f"Remove {plural(len(plan.targets), 'relaxation', 'relaxations')}"),
            detailed_text=self._names_in_details(orphaned, left),
        ):
            return

        outcome = remove_imported_relaxations(document, plan, batch_id=batch_id)
        removed, errors = outcome.removed, outcome.errors
        if not removed:
            self._warn_with_lines("Cannot remove safely", "Nothing was removed.", errors)
            return
        # Only what actually went: a row whose relaxation stayed keeps its binding.
        gone = [ref for ref in plan.targets if ref.site_id not in document.sites]
        definition = self._deps.active_muon_relaxation_definition()
        if definition is not None:
            detach_rows_from_definition(definition, gone)
            self._view.set_form_state(self._form_state_without(gone))
            document.ui_state["muon_relaxation_form"] = self._view.collect_state()

        # Appended once the document edits are done, because append_provenance
        # writes straight to provenance.jsonl: recording the removal before the
        # rows were detached would leave a durable claim about a state the
        # project never reached.
        if outcome.provenance is not None:
            self._context.project_service.append_provenance(document, outcome.provenance)

        self._deps.refresh_muon_relaxation_rows()
        self._deps.refresh_jobs_table()
        self._deps.refresh_muon_sites_summary()
        self._context.persist_ui_state(
            failure_message="Imported relaxations were removed, but autosave failed.",
        )

        if errors:
            self._warn_with_lines("Removal issues", "Some files could not be deleted.", errors)
            self._context.status_message(
                f"Removed {plural(removed, 'imported relaxation')}; some files could not be deleted"
            )
            return
        kept = [
            KeptFolder(job_directory_label(document.project_root, path), because)
            for paths, because in (
                (outcome.kept_directories, _KEPT_FOR_ANOTHER_IMPORT),
                (outcome.misplaced_directories, KEPT_AS_NOT_ITS_OWN),
            )
            for path in paths
        ]
        self._context.status_message(
            folder_outcome_status(
                f"Removed {plural(removed, 'imported relaxation')}",
                deleted=len(outcome.deleted_directories),
                kept=len(kept),
            )
        )
        self._deps.show_kept_folders(kept)

    def _warn_with_lines(self, title: str, summary: str, lines: Sequence[str]) -> None:
        """Warn with ``summary`` and one line per item, kept on screen however many there are."""

        parent = self._deps.dialog_parent()
        if len(lines) == 1:
            QMessageBox.warning(parent, title, f"{summary}\n\n- {lines[0]}")
            return
        show_detailed_warning(
            parent,
            title,
            summary + "\n" + "\n".join(f"- {line}" for line in lines),
            informative_text="Open Details to see each one.",
        )

    @staticmethod
    def _removal_question(plan) -> str:
        """Return what removing a batch will do with its folders, in plain words."""

        count = plural(len(plan.targets), "imported relaxation")
        stays = (
            "another import uses the same folder. The folder is deleted when the last import "
            "using it is removed."
        )
        elsewhere = (
            "the location saved for {them} is not {their} own folder. Nothing there is touched."
        )
        single = len(plan.targets) == 1
        whose = "Its" if single else "Their"
        if not plan.directories and plan.kept_job_ids and not plan.misplaced_job_ids:
            return f"Remove {count} from the project? {whose} files stay, because {stays}"
        if not plan.directories and plan.misplaced_job_ids and not plan.kept_job_ids:
            why = (
                elsewhere.format(them="it", their="its")
                if single
                else (elsewhere.format(them="them", their="their"))
            )
            return f"Remove {count} from the project? {whose} files stay, because {why}"
        if plan.directories:
            directories = "its job directory" if single else "their job directories"
            if plan.orphaned_job_ids:
                # Only some of them have a folder MuFinder can call theirs.
                directories = plural(len(plan.directories), "job directory", "job directories")
            question = (
                f"Permanently remove {count} and delete {directories}, including the "
                "archived original input files?"
            )
        else:
            question = f"Remove {count} from the project?"
        if plan.kept_job_ids:
            kept = plural(len(plan.kept_job_ids), "relaxation")
            question += f"\n\nThe files of {kept} stay, because {stays}"
        if plan.misplaced_job_ids:
            misplaced = plural(len(plan.misplaced_job_ids), "relaxation")
            why = (
                elsewhere.format(them="it", their="its")
                if len(plan.misplaced_job_ids) == 1
                else elsewhere.format(them="them", their="their")
            )
            question += f"\n\nThe files of {misplaced} stay, because {why}"
        return question

    def _in_row_order(self, items: Sequence[_Named]) -> list[_Named]:
        """Return ``items`` in the order their rows show in the relaxations table."""

        definition = self._deps.active_muon_relaxation_definition()
        configuration = definition.parameters.get("configuration") if definition else None
        rows = configuration.get("rows", []) if isinstance(configuration, dict) else []
        position = {
            str(row.get("site_id", "")): index
            for index, row in enumerate(rows)
            if isinstance(row, dict)
        }
        return sorted(
            items, key=lambda item: (position.get(str(item.site_id), len(position)), item.label)
        )

    @staticmethod
    def _orphaned_text(labels: Sequence[str]) -> str:
        """Say whose records go while their files stay, because their job is already gone."""

        if not labels:
            return ""
        if len(labels) == 1:
            return (
                f"\n\nThe project no longer holds the job of {labels[0]}, so MuFinder cannot "
                "tell which files are its own: its records are removed and its files stay "
                "on disk."
            )
        return (
            f"\n\nThe project no longer holds the jobs of {plural(len(labels), 'relaxation')} "
            f"({_inline_names(labels)}), so MuFinder cannot tell which files are theirs: their "
            "records are removed and their files stay on disk."
        )

    @staticmethod
    def _left_in_batch_text(left: Sequence[LeftSite]) -> str:
        """Say which of the batch's sites stay, and why, one sentence for each reason."""

        text = ""
        for reason, (one, many) in _WHY_LEFT.items():
            labels = [site.label for site in left if site.reason == reason]
            if not labels:
                continue
            single = len(labels) == 1
            text += (
                f"\n\n{plural(len(labels), 'other relaxation')} from this import "
                f"{'stays' if single else 'stay'} in the project, because "
                f"{one if single else many}: {_inline_names(labels)}."
            )
        return text

    @staticmethod
    def _names_in_details(
        orphaned: Sequence[ImportedRelaxationRef], left: Sequence[LeftSite]
    ) -> str:
        """List every name under Details when the message names only some of them."""

        groups = [
            [ref.label for ref in orphaned],
            *([site.label for site in left if site.reason == reason] for reason in _WHY_LEFT),
        ]
        if all(len(group) <= _NAMES_INLINE for group in groups):
            return ""
        sections = []
        if orphaned:
            sections.append(
                "Records removed, files kept on disk:\n"
                + "\n".join(f"- {ref.label}" for ref in orphaned)
            )
        if left:
            sections.append(
                "Staying in the project:\n"
                + "\n".join(f"- {site.label}: {_WHY_LEFT[site.reason][0]}" for site in left)
            )
        return "\n\n".join(sections)

    def _form_state_without(self, targets) -> dict[str, object]:
        """Return the widget's form state with ``targets``' rows dropped."""

        state = self._view.collect_state()
        rows = state.get("rows", []) if isinstance(state, dict) else []
        removed = {str(ref.site_id) for ref in targets}
        return {"rows": [row for row in rows if str(row.get("site_id", "")) not in removed]}

    def merge_imported_relaxation_rows(
        self,
        outcome,
        *,
        output_directory: Path,
    ) -> bool:
        """Attach an import batch's rows to the muon-relaxation definition.

        The app-layer half of an import. ``services`` builds the entities and
        files but cannot finish the job: the relaxations table renders the
        *intersection* of the widget's form state and the active definition's
        rows, matched on a signature that includes normalised parameter values,
        so the rows have to be pushed through the live widget and read back.
        A services-only import would appear to succeed while every imported row
        rendered "ready" with no energy.

        The active definition is mutated **in place**. Re-registering it would
        arrive with an empty ``job_ids_by_label``, and the synchronisation pass
        that normally re-adopts those bindings only matches rows whose
        signatures already agree — which an import's never do — so every
        existing row would lose its job.

        Returns ``False`` when there is no document or no host structure to
        anchor a new definition to.
        """

        document = self._context.document()
        if document is None:
            return False
        payloads = list(outcome.row_payloads)
        if not payloads:
            return True

        definition = self._deps.active_muon_relaxation_definition()
        if definition is None:
            host_id = document.project.host_structure_id
            if host_id is None:
                return False
            definition = self._context.project_service.register_muon_relaxation_definition(
                document,
                input_structure_id=host_id,
                parameters={"rows": []},
                output_directory=output_directory,
            )
            document.ui_state[ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)

        configuration = definition.parameters.setdefault("configuration", {"rows": []})
        if not isinstance(configuration, dict):
            configuration = {"rows": []}
            definition.parameters["configuration"] = configuration
        rows = configuration.setdefault("rows", [])
        if not isinstance(rows, list):
            rows = []
            configuration["rows"] = rows

        mapping = ensure_job_mapping(definition)
        seed_mapping = definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(seed_mapping, dict):
            seed_mapping = {}
            definition.parameters["job_seed_names_by_label"] = seed_mapping

        # Mirror first, then build the definition rows from what the widget gives
        # back. The row signature is computed over both sides, so anything the
        # widget normalises on ingest has to be what the definition stores too;
        # appending the raw payload here would leave the two describing the same
        # row differently, and a signature mismatch drops the imported row to a
        # fallback that only accepts jobs in an active status — rendering a
        # completed import as "ready" with no energy.
        normalized_rows = self._mirror_imported_rows_into_form(payloads)
        if normalized_rows is None:
            return False

        for item, payload in zip(outcome.imported, payloads, strict=False):
            row = normalized_rows.get(str(item.site_id))
            if row is None:
                # The widget did not take the row; adding it to the definition
                # anyway would create exactly the mismatch described above.
                continue
            rows.append(row)
            key = f"{muon_charge_state_prefix(item.charge_state)}:{item.label}"
            mapping[key] = str(item.job_id)
            seed_mapping[key] = item.seed_name
            if item.site_id not in definition.input_site_ids:
                definition.input_site_ids.append(item.site_id)

        return True

    def _mirror_imported_rows_into_form(
        self, payloads: list[dict[str, object]]
    ) -> dict[str, dict[str, object]] | None:
        """Add imported rows to the widget's form state and store the result.

        The widget's state shape is **not** the definition's configuration shape
        — it names the position ``fractional_position`` where the configuration
        says ``initial_fractional_position`` — so the payloads are translated
        rather than passed through. Feeding the configuration shape straight in
        parses without error and silently zeroes every position.

        Returns the widget's own normalised rows, in the *configuration* shape,
        keyed by site id — the caller stores these rather than the payloads it
        passed in, because the row signature is computed over them. Returns
        ``None`` when there is no document, so the caller can decline to touch
        the definition at all rather than leave it holding rows the form never
        received.
        """

        document = self._context.document()
        if document is None:
            return None
        state = self._view.collect_state()
        rows = list(state.get("rows", [])) if isinstance(state, dict) else []
        for payload in payloads:
            entry = dict(payload)
            entry["fractional_position"] = list(
                entry.pop("initial_fractional_position", [0.0, 0.0, 0.0])
            )
            rows.append(entry)
        self._view.set_form_state({"rows": rows})
        normalized = self._view.collect_state()
        if not isinstance(normalized, dict):
            return None
        document.ui_state["muon_relaxation_form"] = normalized

        wanted = {str(payload.get("site_id", "")) for payload in payloads}
        by_site: dict[str, dict[str, object]] = {}
        for row in normalized.get("rows", []):
            if not isinstance(row, dict):
                continue
            site_id = str(row.get("site_id", ""))
            if site_id not in wanted:
                continue
            entry = dict(row)
            # Back into the configuration shape the definition stores.
            entry["initial_fractional_position"] = list(
                entry.pop("fractional_position", [0.0, 0.0, 0.0])
            )
            by_site[site_id] = entry
        return by_site


__all__ = ["ImportedRelaxationsController", "ImportedRelaxationsDeps"]
