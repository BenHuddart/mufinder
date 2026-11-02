"""Plan and run the clean-up of jobs' files and records.

Removing a workflow row, removing a convergence cutoff and deleting finished
jobs from the dashboard all run one sequence: check that a job directory
belongs only to the jobs being removed, delete or archive it, then delete the
job records.
:class:`JobFileCleanup` owns that sequence.

- :meth:`JobFileCleanup.plan_item` turns a group of jobs into a
  :class:`JobCleanupItem`: the directory on disk, where an archived copy goes and,
  when asked, why the directory may not be deleted.
- :meth:`JobFileCleanup.execute` runs a plan. A ``choose_disposition`` callable
  says, per item with files, whether to delete or archive them; an
  ``*_all`` answer applies to every later item. A folder that is not the item's
  alone is not offered: its records go and the folder stays. The outcome lists the
  items cleaned, what happened to each folder and the failures as data.
- :meth:`JobFileCleanup.directory_ownership_refusals` says why deleting a job's
  directories would be unsafe; the failed-job reset and the fresh-start path use
  it with :meth:`JobFileCleanup.clear_directory_contents`.
- :meth:`JobFileCleanup.delete_job_records` deletes a job with its runs and
  results and unbinds it from every definition mapping.

The service never shows a dialog: the window asks the questions and reports the
outcome. Filesystem operations run through ``run_file_op`` so the window can run
them off the GUI thread.

Layering: this lives in ``mufinder.services`` and imports ``domain`` and
``job_paths``, whose ``job_directory_refusal`` is the last check before any
folder goes.

Two answers to "this folder may not go", because they leave the user in different
places. A link is *refused*: the item stays, since the user can remove the link and
try again. Any other folder that is not the item's alone or not its own (another
job's, or one a saved name leads out to) is *kept*: the item's records go and the
folder stays, because nothing in the app can correct a saved name, and waiting for the
other job to go first could wait for ever (an imported batch waits for any native job
in its folder).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import posixpath
import shutil
import stat
import time
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job
from mufinder.domain.text import KeptFolder, plain_error
from mufinder.services.job_paths import (
    job_directory_foreign,
    job_directory_label,
    job_directory_link_refusal,
    job_directory_on_disk,
    job_directory_refusal,
    resolved_path,
)

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument

#: What to do with an item's files: ``delete`` or ``archive`` them, or the
#: ``*_all`` form, which also applies to every later item of the plan.
FileDisposition = Literal["delete", "archive", "delete_all", "archive_all"]

#: Runs one blocking filesystem operation, given the operation and a short
#: description of it, and returns its result or raises its exception.
FileOpRunner = Callable[[Callable[[], object], str], object]

#: The folder, inside an item's archive root, that archived job directories move into.
ARCHIVE_FOLDER_NAME = "removed_from_workflow"

#: The job-mapping parameters a definition binds jobs through.
_JOB_MAPPING_NAMES = (
    "job_ids_by_cutoff",
    "job_ids_by_grid",
    "job_ids_by_label",
    "singlepoint_job_ids_by_label",
    "phonon_job_ids_by_label",
)

#: Why a folder stays, for one folder and for several.
KEPT_FOR_ANOTHER_JOB = ("another job uses it", "other jobs use them")
KEPT_AS_NOT_ITS_OWN = (
    "its saved location is not this job's own folder",
    "their saved locations are not these jobs' own folders",
)
KEPT_IN_ANOTHER_PLACE = (
    "it belongs to another workflow or charge state",
    "they belong to other workflows or charge states",
)

_DELETE_DISPOSITIONS = frozenset({"delete", "delete_all"})
_APPLY_TO_ALL_DISPOSITIONS = frozenset({"delete_all", "archive_all"})


def _run_directly(op: Callable[[], object], _description: str) -> object:
    return op()


@dataclass(frozen=True, slots=True)
class JobCleanupItem:
    """One unit of a clean-up plan: a directory and the job records removed with it.

    ``directory`` is ``None`` when there are no files on disk to deal with.
    Archived files move into ``<archive_root>/removed_from_workflow``, or next to
    ``directory`` when ``archive_root`` is ``None``. An item with a ``refusal``
    is never cleaned: its files and records are kept. ``label`` names the item
    in messages and ``key`` is the caller's handle on it (a row key or a cutoff).
    ``releases`` names jobs that stay in the project but stop using the folder when
    the item goes, so they do not keep it.
    """

    label: str
    jobs: tuple[Job, ...] = ()
    directory: Path | None = None
    archive_root: Path | None = None
    refusal: str | None = None
    key: object = None
    releases: frozenset[UUID] = frozenset()


@dataclass(frozen=True, slots=True)
class JobCleanupFailure:
    """Why an item was not cleaned: an ownership ``refusal`` or a filesystem ``error``."""

    item: JobCleanupItem
    refusal: str | None = None
    error: OSError | None = None

    @property
    def reason(self) -> str:
        """Return why, in words, whichever of the two it was."""

        if self.refusal is not None:
            return self.refusal.removeprefix(f"- {self.item.label}: ")
        return plain_error(self.error) if self.error is not None else "it was not cleaned"


@dataclass(frozen=True, slots=True)
class JobCleanupKept:
    """A folder left in place while its item's records went, and why, for one and for several."""

    item: JobCleanupItem
    directory: Path
    because: tuple[str, str]
    label: str


@dataclass(frozen=True, slots=True)
class JobCleanupOutcome:
    """What :meth:`JobFileCleanup.execute` did, in the order it did it.

    ``cleaned`` holds every item whose records went, ``kept`` those of them whose
    folder stayed. ``cancelled`` means ``choose_disposition`` returned ``None``:
    that item and every later one were left alone.
    """

    cleaned: tuple[JobCleanupItem, ...] = ()
    failures: tuple[JobCleanupFailure, ...] = ()
    cancelled: bool = False
    deleted: tuple[Path, ...] = ()
    archived: tuple[Path, ...] = ()
    kept: tuple[JobCleanupKept, ...] = ()

    @property
    def kept_folders(self) -> tuple[KeptFolder, ...]:
        """Return each folder that stayed, with why."""

        folders = {entry.directory: entry for entry in self.kept}
        return tuple(KeptFolder(entry.label, entry.because) for entry in folders.values())

    @property
    def completed(self) -> bool:
        """Whether every item of the plan was cleaned."""

        return not self.cancelled and not self.failures


class JobFileCleanup:
    """Plans and runs job-file clean-up for the current project document."""

    #: Back-off, in seconds, between retries of a filesystem operation that
    #: another process holds a transient lock on.
    FILESYSTEM_RETRY_DELAYS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0)

    def __init__(
        self,
        *,
        document_getter: Callable[[], ProjectDocument | None],
        job_output_directory: Callable[[Job], Path | None],
        job_seed_name: Callable[[Job], str],
        active_run_by_job: MutableMapping[UUID, UUID] | None = None,
        run_file_op: FileOpRunner = _run_directly,
    ) -> None:
        self._document_getter = document_getter
        self._job_output_directory = job_output_directory
        self._job_seed_name = job_seed_name
        self._active_run_by_job = active_run_by_job if active_run_by_job is not None else {}
        self._run_file_op = run_file_op

    # --- planning ---

    def plan_item(
        self,
        jobs: Sequence[Job],
        *,
        label: str | None = None,
        archive_root: Path | None = None,
        check_ownership: bool = False,
        key: object = None,
    ) -> JobCleanupItem:
        """Return the clean-up item for ``jobs``, whose directory is that of the first job.

        The directory is kept only when it exists. With ``check_ownership``, the
        item is refused when another job still uses the directory; the jobs
        given are cleaned together, so they may share it.
        """

        jobs = tuple(jobs)
        owner = jobs[0] if jobs else None
        directory = self._job_output_directory(owner) if owner is not None else None
        refusal = None
        if check_ownership and owner is not None:
            refusal = self.directory_ownership_refusals(
                [owner],
                shared_with={job.id for job in jobs},
                check_remote=False,
            ).get(owner.id)
        return JobCleanupItem(
            label=label if label is not None else (owner.label if owner is not None else ""),
            jobs=jobs,
            directory=directory if self.directory_on_disk(directory) else None,
            archive_root=archive_root,
            refusal=refusal,
            key=key,
        )

    def directory_on_disk(self, directory: Path | None) -> bool:
        """Return whether a clean-up has anything to deal with at ``directory``.

        True for a folder that exists, and for a dangling link or a missing folder
        behind a link, which have to reach the refusal instead of reading as absent.
        """

        document = self._document_getter()
        if directory is None or document is None:
            return directory is not None and directory.exists()
        return job_directory_on_disk(document.project_root, directory)

    def directory_ownership_refusals(
        self,
        jobs: Sequence[Job],
        *,
        shared_with: Collection[UUID] = (),
        check_remote: bool = True,
    ) -> dict[UUID, str]:
        """Return, per job id, why deleting that job's directories would be unsafe.

        A job is refused when its local directory is not strictly inside
        ``<project>/files/<stage>/``, is reached through a link, is not where a
        job of its workflow belongs, is not named after the job's seed, or is
        used by another job (the same directory, or one inside it); and, with
        ``check_remote``, when its remote directory is not named after the seed
        or another job on the same cluster uses it or a directory inside it.
        Jobs in ``shared_with`` may share a directory without causing a
        refusal: they are being cleaned together, or the caller knows they
        share it by design (a Quantum Corrections phonon job with the legacy
        singlepoint an older project ran in its directory).
        """

        document = self._document_getter()
        if document is None or not jobs:
            return {}
        files_root = (document.project_root / "files").resolve()

        def local_directory(job: Job) -> Path | None:
            directory = self._job_output_directory(job)
            return resolved_path(directory) if directory is not None else None

        def remote_directory(job: Job) -> tuple[str, str] | None:
            target = (job.execution_target or "").strip().casefold()
            directory = (job.remote_directory or "").strip()
            if not target or target == "local" or not directory:
                return None
            return target, posixpath.normpath(directory)

        others = [
            (other.id, local_directory(other), remote_directory(other))
            for other in document.jobs.values()
        ]
        refusals: dict[UUID, str] = {}
        for job in jobs:
            seed = self._job_seed_name(job)
            local = local_directory(job)
            remote = remote_directory(job)
            reason = ""
            if local is not None:
                if (
                    not local.is_relative_to(files_root)
                    or len(local.relative_to(files_root).parts) < 2
                ):
                    reason = f"its output folder is outside the project's job folders ({local})"
                elif unsafe := job_directory_refusal(
                    document.project_root, job, self._job_output_directory(job)
                ):
                    reason = unsafe
                elif foreign := job_directory_foreign(
                    document.project_root, job, self._job_output_directory(job)
                ):
                    reason = foreign
                elif local.name != seed:
                    reason = f"its output folder is not named after the job ({local})"
                elif any(
                    other_id != job.id
                    and other_id not in shared_with
                    and other_local is not None
                    and other_local.is_relative_to(local)
                    for other_id, other_local, _other_remote in others
                ):
                    reason = f"another job uses the same output folder ({local})"
            if not reason and check_remote and remote is not None:
                cluster, path = remote
                if posixpath.basename(path) != seed:
                    reason = f"its cluster directory is not named after the job ({path})"
                elif any(
                    other_id != job.id
                    and other_id not in shared_with
                    and other_remote is not None
                    and other_remote[0] == cluster
                    and (
                        other_remote[1] == path
                        or other_remote[1].startswith(path.rstrip("/") + "/")
                    )
                    for other_id, _other_local, other_remote in others
                ):
                    reason = f"another job uses the same cluster directory ({path})"
            if reason:
                refusals[job.id] = f"- {job.label}: {reason}"
        return refusals

    # --- execution ---

    def execute(
        self,
        plan: Sequence[JobCleanupItem],
        *,
        choose_disposition: Callable[[JobCleanupItem], FileDisposition | None],
        stop_on_error: bool,
    ) -> JobCleanupOutcome:
        """Clean each item of ``plan`` and return what happened.

        In order, except that items whose folders overlap are taken one after another,
        so no prompt can stop the plan between a shared folder going and the last item
        that used it.

        For an item whose files are still on disk when its turn comes,
        ``choose_disposition`` says whether to delete or
        archive them, until it answers with an ``*_all`` disposition, which
        applies to every later item. ``None`` cancels: that item and the rest
        are left alone. A refused item, or one whose files cannot be updated,
        keeps its files and records and is reported; with ``stop_on_error`` the
        plan stops there. An item whose folder is not its alone is not offered
        to ``choose_disposition``: the folder stays and is reported as kept, and
        jobs of other items in ``plan`` do not count unless they are refused, since
        the rest go together. Every other item's job records are deleted.
        """

        cleaned: list[JobCleanupItem] = []
        failures: list[JobCleanupFailure] = []
        deleted: list[Path] = []
        archived: list[Path] = []
        kept: list[JobCleanupKept] = []
        together = {
            job_id
            for item in plan
            if item.refusal is None and self._folder_refusal(item) is None
            for job_id in ({job.id for job in item.jobs} | item.releases)
        }
        bound = self._bound_job_ids()
        shared_disposition: FileDisposition | None = None

        def outcome(*, cancelled: bool = False) -> JobCleanupOutcome:
            return JobCleanupOutcome(
                tuple(cleaned),
                tuple(failures),
                cancelled,
                tuple(deleted),
                tuple(archived),
                # A later item of the plan may have taken a folder an earlier one left.
                tuple(entry for entry in kept if entry.directory.exists()),
            )

        for item in _overlapping_folders_together(plan):
            if item.refusal is not None:
                failures.append(JobCleanupFailure(item, refusal=item.refusal))
                if stop_on_error:
                    break
                continue
            # Checked again here: an earlier item may already have deleted or
            # archived a directory this one shares (a reconfigured relaxation
            # row's old and replacement jobs use one folder).
            if item.directory is not None and self.directory_on_disk(item.directory):
                refusal = self._folder_refusal(item)
                if refusal is not None:
                    failures.append(JobCleanupFailure(item, refusal=refusal))
                    together -= {job.id for job in item.jobs} | item.releases
                    if stop_on_error:
                        break
                    continue
                because = self._folder_kept_because(item, together, bound)
                if because is not None:
                    document = self._document_getter()
                    label = (
                        job_directory_label(document.project_root, item.directory)
                        if document is not None
                        else item.directory.name
                    )
                    kept.append(JobCleanupKept(item, item.directory, because, label))
                else:
                    disposition = shared_disposition or choose_disposition(item)
                    if disposition is None:
                        return outcome(cancelled=True)
                    try:
                        if disposition in _DELETE_DISPOSITIONS:
                            self.delete_path(item.directory)
                            deleted.append(item.directory)
                        else:
                            self.archive_directory(item.directory, item.archive_root)
                            archived.append(item.directory)
                    except OSError as exc:
                        failures.append(JobCleanupFailure(item, error=exc))
                        # Staying now, so a later item must not delete what holds it.
                        together -= {job.id for job in item.jobs} | item.releases
                        if stop_on_error:
                            break
                        continue
                    if disposition in _APPLY_TO_ALL_DISPOSITIONS:
                        shared_disposition = disposition
            for job in item.jobs:
                self.delete_job_records(job.id)
            cleaned.append(item)
        return outcome()

    def _folder_refusal(self, item: JobCleanupItem) -> str | None:
        """Return why ``item``'s folder has to stay with the item, whichever route planned it."""

        document = self._document_getter()
        if document is None or item.directory is None:
            return None
        reason = job_directory_link_refusal(document.project_root, item.directory)
        return f"- {item.label}: {reason}" if reason is not None else None

    def _folder_kept_because(
        self, item: JobCleanupItem, together: Collection[UUID], bound: Collection[UUID]
    ) -> tuple[str, str] | None:
        """Return why ``item``'s sound folder is not its to delete, or ``None``.

        Checked on every item, since not every route asks for the ownership check.
        Another job shares it when their folders are the same or one holds the other.
        The one job that does not count is a detached cutoff job in this very folder: a
        reset leaves it in the project, unbound and ``defined``, finding by its name the
        folder of whichever job later ran that cutoff, with nothing there of its own.
        """

        document = self._document_getter()
        if document is None or item.directory is None:
            return None
        owner = item.jobs[0] if item.jobs else None
        if job_directory_refusal(document.project_root, owner, item.directory):
            return KEPT_AS_NOT_ITS_OWN
        folder = resolved_path(item.directory)
        for other in document.jobs.values():
            if other.id in together:
                continue
            directory = self._job_output_directory(other)
            if directory is None:
                continue
            # A saved name leading out of its place (``..`` reads as a whole workflow's
            # folder) says nothing about who uses what. A link really does point here.
            if job_directory_refusal(
                document.project_root, other, directory
            ) and not job_directory_link_refusal(document.project_root, directory):
                continue
            resolved = resolved_path(directory)
            if resolved == folder and (
                other.workflow_stage == WorkflowStage.CONVERGENCE_CUTOFF
                and other.status == JobStatus.DEFINED
                and other.id not in bound
                and IMPORTED_JOB_TAG not in (other.tags or ())
            ):
                continue
            if resolved.is_relative_to(folder) or folder.is_relative_to(resolved):
                return KEPT_FOR_ANOTHER_JOB
        if job_directory_foreign(document.project_root, owner, item.directory):
            return KEPT_IN_ANOTHER_PLACE
        return None

    def _bound_job_ids(self) -> set[UUID]:
        """Return the jobs some definition's mappings bind."""

        document = self._document_getter()
        if document is None:
            return set()
        return {
            document_job.id
            for definition in document.calculations.values()
            for mapping_name in _JOB_MAPPING_NAMES
            if isinstance(mapping := definition.parameters.get(mapping_name, {}), dict)
            for value in mapping.values()
            if (document_job := _job_by_text_id(document.jobs, value)) is not None
        }

    def delete_job_records(self, job_id: UUID) -> None:
        """Delete a job with its runs and results, and unbind it from every definition.

        A site relaxed by one of the job's runs loses that relaxation.
        """

        document = self._document_getter()
        if document is None:
            return

        run_ids = [run.id for run in document.runs.values() if run.job_id == job_id]
        result_ids = {
            result.id
            for result in document.results.values()
            if result.calculation_run_id in run_ids
        }
        result_ids.update(
            run.result_id
            for run in document.runs.values()
            if run.job_id == job_id and run.result_id is not None
        )

        run_id_set = set(run_ids)
        for site in document.sites.values():
            if site.relaxation_run_id not in run_id_set:
                continue
            site.clear_relaxation()

        for result_id in result_ids:
            document.results.pop(result_id, None)
        for run_id in run_ids:
            document.runs.pop(run_id, None)

        self._active_run_by_job.pop(job_id, None)
        target = str(job_id)
        for definition in document.calculations.values():
            for mapping_name in _JOB_MAPPING_NAMES:
                mapping = definition.parameters.get(mapping_name, {})
                if not isinstance(mapping, dict):
                    continue
                stale_keys = [key for key, value in mapping.items() if str(value) == target]
                for key in stale_keys:
                    mapping.pop(key, None)
        document.jobs.pop(job_id, None)

    # --- filesystem operations ---

    def delete_path(self, path: Path) -> None:
        """Delete a file or directory tree, retrying through transient locks.

        Raises ``OSError`` when it still cannot be deleted.
        """

        if path.is_dir():
            op = lambda: self.retry_filesystem_op(  # noqa: E731
                lambda: shutil.rmtree(path, onexc=_force_writable_and_retry)
            )
        else:
            op = lambda: self.retry_filesystem_op(path.unlink)  # noqa: E731
        self._run_file_op(op, f"Deleting {path.name}…")

    def archive_directory(self, source: Path, archive_root: Path | None) -> Path:
        """Move ``source`` into ``<archive_root>/removed_from_workflow`` and return where it went.

        ``archive_root`` defaults to the folder holding ``source``. A name
        already taken there gets a numeric suffix.
        """

        archive_folder = (
            archive_root if archive_root is not None else source.parent
        ) / ARCHIVE_FOLDER_NAME
        archive_folder.mkdir(parents=True, exist_ok=True)
        destination = _unique_destination(archive_folder / source.name)
        self._run_file_op(
            lambda: self.retry_filesystem_op(lambda: source.rename(destination)),
            f"Archiving {source.name}…",
        )
        return destination

    @classmethod
    def retry_filesystem_op(cls, op: Callable[[], object]) -> None:
        """Run ``op``, retrying with short back-off while it raises ``PermissionError``.

        OneDrive sync, antivirus scanners and the Windows search indexer hold
        transient handles that produce ``PermissionError`` ([WinError 5]); the
        retries (about 2 s in all) clear the common case. A missing path counts
        as done. The last attempt's error propagates.
        """

        for delay in cls.FILESYSTEM_RETRY_DELAYS:
            try:
                op()
                return
            except FileNotFoundError:
                return
            except PermissionError:
                time.sleep(delay)
        try:
            op()
        except FileNotFoundError:
            pass

    @staticmethod
    def clear_directory_contents(directory: Path) -> None:
        """Delete a job directory's contents except its ``.cell``/``.param`` inputs.

        A job directory is named after the job's seed, so CASTEP's relaxed
        ``{seed}-out.cell`` is recognised by exact name and deleted with the
        rest of the run's output; every other ``.cell``/``.param`` is kept, so a
        seed that itself ends in ``-out`` keeps its input. A symlink is removed
        as a link, never followed into what it points at.
        """

        output_cell = f"{directory.name}-out.cell".casefold()
        for child in directory.iterdir():
            if (
                child.is_file()
                and child.suffix.lower() in {".cell", ".param"}
                and child.name.casefold() != output_cell
            ):
                continue
            if child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def _job_by_text_id(jobs: Mapping[UUID, Job], value: object) -> Job | None:
    try:
        return jobs.get(UUID(str(value)))
    except ValueError:
        return None


def _overlapping_folders_together(plan: Sequence[JobCleanupItem]) -> list[JobCleanupItem]:
    """Return ``plan`` with each item followed by all those linked to it by overlapping folders.

    Followed through: ``outer/a`` brings ``outer``, and ``outer`` brings ``outer/b``.
    """

    folders = [
        resolved_path(item.directory) if item.directory is not None else None for item in plan
    ]
    ordered: list[JobCleanupItem] = []
    placed: set[int] = set()
    for first in range(len(plan)):
        waiting = [first] if first not in placed else []
        placed.update(waiting)
        while waiting:
            index = waiting.pop(0)
            ordered.append(plan[index])
            folder = folders[index]
            if folder is None:
                continue
            for other_index, other in enumerate(folders):
                if (
                    other_index not in placed
                    and other is not None
                    and (other.is_relative_to(folder) or folder.is_relative_to(other))
                ):
                    placed.add(other_index)
                    waiting.append(other_index)
    return ordered


def _force_writable_and_retry(func, path, exc_info) -> None:
    # shutil.rmtree onexc handler: Windows refuses to delete read-only files
    # (CASTEP often marks output read-only) and raises PermissionError. Clear
    # the read-only attribute and retry the failed primitive once. Anything
    # else, including a second failure, propagates to the retry loop.
    exc = exc_info[1] if isinstance(exc_info, tuple) else exc_info
    if not isinstance(exc, PermissionError):
        raise exc
    try:
        os.chmod(path, stat.S_IWRITE)
    except OSError:
        raise exc
    func(path)


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.parent / f"{path.name}_{suffix}"
        if not candidate.exists():
            return candidate
        suffix += 1
