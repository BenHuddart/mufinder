"""Snapshot a window's project around a submission, and answer the submission dialog with Cancel.

A cancelled submission is meant to change nothing. A test saves and reopens the
project as File > Open leaves it (:func:`reopen_saved`), takes a
:class:`ProjectSnapshot`, runs the case with the dialog cancelled
(:func:`cancel_submission`) and compares a second snapshot with the first.
:func:`without_ids` writes a project's files and model with every id and time
replaced, so two runs of one case compare equal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from mufinder.persistence.codec import to_jsonable
from mufinder.services.project_service import ProjectDocument, UnsavedChanges

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow


def project_files(root: Path) -> dict[str, bytes]:
    """Return every file under ``root`` by its relative POSIX path, with its bytes."""

    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class ProjectSnapshot:
    """What a submission may change in the window's project.

    ``journal`` holds each record's action and description, in memory;
    ``journal_on_disk`` the bytes of the project's provenance stream; ``model``
    the whole in-memory project as JSON.
    """

    definitions: frozenset[UUID]
    journal: tuple[tuple[str, str], ...]
    journal_on_disk: bytes
    unsaved: UnsavedChanges
    files: dict[str, bytes]
    model: str

    @classmethod
    def take(cls, window: MainWindow) -> ProjectSnapshot:
        """Snapshot the project open in ``window``."""

        document = window._document
        assert document is not None
        provenance = document.project_root / "provenance.jsonl"
        return cls(
            definitions=frozenset(document.calculations),
            journal=tuple((record.action, record.description) for record in document.provenance),
            journal_on_disk=provenance.read_bytes() if provenance.exists() else b"",
            unsaved=window._project_service.unsaved_changes(document),
            files=project_files(document.project_root),
            model=document_model(document),
        )

    def new_journal_records(self, later: ProjectSnapshot) -> tuple[tuple[str, str], ...]:
        """Return the records ``later`` holds after this snapshot's."""

        assert later.journal[: len(self.journal)] == self.journal
        return later.journal[len(self.journal) :]


def document_model(document: ProjectDocument) -> str:
    """Return the whole in-memory ``document`` as sorted JSON."""

    return json.dumps(to_jsonable(document), sort_keys=True)


_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(\+00:00|Z)?")


def without_ids(text: str) -> str:
    """Return ``text`` with every UUID written ``<id>`` and every ISO time ``<time>``."""

    return _TIME.sub("<time>", _ID.sub("<id>", text))


def _canonical(value: object) -> object:
    if isinstance(value, dict):
        if value and all(_ID.fullmatch(str(key)) for key in value):
            # Keyed by ids: the order of random ids says nothing, so sort the entries.
            return sorted(json.dumps(_canonical(item), sort_keys=True) for item in value.values())
        return {without_ids(str(key)): _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, str):
        return without_ids(value)
    return value


def model_without_ids(document: ProjectDocument) -> str:
    """Return :func:`document_model` with every id and time replaced, in an order ids do not set."""

    payload = to_jsonable(document)
    payload.pop("project_root", None)
    return json.dumps(_canonical(payload), sort_keys=True)


def files_without_ids(root: Path) -> list[tuple[str, str]]:
    """Return every text file under ``root`` as ``(path, content)``, ids and times replaced, sorted.

    A JSON file (or JSON-lines file) is read and written back through the same
    canonical form as :func:`model_without_ids`, so the random order of ids
    sets no order in it.
    """

    def canonical(name: str, text: str) -> str:
        if name.endswith(".json"):
            return json.dumps(_canonical(json.loads(text)), sort_keys=True)
        if name.endswith(".jsonl"):
            return "\n".join(
                json.dumps(_canonical(json.loads(line)), sort_keys=True)
                for line in text.splitlines()
                if line.strip()
            )
        return without_ids(text)

    return sorted(
        (without_ids(name), canonical(name, data.decode("utf-8")))
        for name, data in project_files(root).items()
    )


def reopen_saved(window: MainWindow, document: ProjectDocument) -> ProjectDocument:
    """Save ``document``, open it in ``window`` as File > Open does and save it; return it."""

    window._project_service.save_project(document)
    assert window.open_project(document.project_root)
    # Saved once more as opened: the pages tidy some saved forms as they fill.
    window._project_service.save_project(window._document)
    assert window._project_service.unsaved_changes(window._document) is UnsavedChanges.NONE
    return window._document


class _RejectingClusterDialog:
    """The cluster submission dialog, cancelled."""

    class DialogCode:
        Accepted = 1

    def __init__(self, **_kwargs) -> None:
        pass

    def exec(self) -> int:
        return 0


def cancel_submission(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, target: str
) -> list[str]:
    """Answer the submission dialog of ``target`` (``"local"`` or a cluster name) with Cancel.

    Returns the dialogs opened, ``"local"`` or ``"cluster"``, so a test knows
    the submission reached its dialog; nothing may reach the runner.
    """

    import mufinder.app.jobs_controller as jobs_controller_module
    from tests.support.main_window import fake_local_submission, patch_continuation_cluster_dialog

    fake_local_submission(window, monkeypatch, accept=False)
    monkeypatch.setattr(
        window._job_execution, "submit_local_jobs", lambda *_a, **_k: pytest.fail("submitted")
    )
    opened: list[str] = []
    local_dialog = jobs_controller_module.LocalSubmissionDialog
    monkeypatch.setattr(
        jobs_controller_module,
        "LocalSubmissionDialog",
        lambda **kwargs: opened.append("local") or local_dialog(**kwargs),
    )
    if target != "local":
        patch_continuation_cluster_dialog(monkeypatch)
        monkeypatch.setattr(
            jobs_controller_module,
            "ClusterSubmissionDialog",
            lambda **kwargs: opened.append("cluster") or _RejectingClusterDialog(**kwargs),
        )
    return opened


def quiet_messages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Answer every information and warning box at once, and return the titles shown."""

    import mufinder.app.jobs_controller as jobs_controller_module
    import mufinder.app.main_window as main_window_module

    shown: list[str] = []
    for module in (jobs_controller_module, main_window_module):
        for name in ("information", "warning"):
            monkeypatch.setattr(
                module.QMessageBox,
                name,
                lambda _parent, title, *_a, **_k: shown.append(title) or None,
            )
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, title, *_a, **_k: shown.append(title),
    )
    return shown


def accept_cluster_dialog(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, *, credentials: bool = True
) -> None:
    """Accept the cluster submission dialog for cluster ``alpha``, with no reusable connection.

    Without ``credentials`` the dialog gives none.
    """

    import mufinder.app.jobs_controller as jobs_controller_module
    from tests.support.main_window import patch_continuation_cluster_dialog

    patch_continuation_cluster_dialog(monkeypatch)
    monkeypatch.setattr(window._keyword_catalog_guard, "confirm_submission", lambda: True)
    if not credentials:

        class _NoCredentials(jobs_controller_module.ClusterSubmissionDialog):
            def credentials(self):
                return None

        monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _NoCredentials)
    monkeypatch.setattr(
        jobs_controller_module, "can_reuse_cluster_connection", lambda *_a, **_k: False
    )


def reach_the_remote_check(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, *, answer: str, existing: bool = True
) -> list[list[dict]]:
    """Accept the cluster dialog and connect; with ``existing`` every remote folder is taken.

    Each overwrite prompt is answered ``answer``. The batch never runs: the
    returned list collects the items each batch would have sent.
    """

    accept_cluster_dialog(window, monkeypatch)
    monkeypatch.setattr(window._jobs, "_connect_for_cluster_submission", lambda *_a: True)
    monkeypatch.setattr(
        window._jobs,
        "_await_remote_dir_check",
        lambda targets: {
            "existing": [target for _id, target in targets] if existing else [],
            "errors": {},
        },
    )
    monkeypatch.setattr(window._jobs, "_prompt_remote_directory_overwrite", lambda *_a: answer)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    batches: list[list[dict]] = []
    monkeypatch.setattr(
        window._jobs,
        "_run_cluster_submit_batch",
        lambda items, _prepared: (batches.append(list(items)), ([], []))[1],
    )
    return batches


def assert_unchanged(before: ProjectSnapshot, window: MainWindow) -> None:
    """Assert the window's project is as ``before`` found it, and had nothing unsaved."""

    after = ProjectSnapshot.take(window)
    assert after.files == before.files
    assert after.model == before.model
    assert after.journal == before.journal
    assert after.unsaved is before.unsaved is UnsavedChanges.NONE


#: What each accepted submission of the planned-jobs tests leaves: its project's
#: file names, Journal actions and jobs, and the cluster batch it sends.
ACCEPTED_SUBMISSIONS = Path(__file__).resolve().parents[1] / "data" / "accepted_submissions.json"


def accepted_summary(window: MainWindow, batches: Sequence[Sequence[dict]] = ()) -> dict:
    """Return what an accepted submission left in ``window``'s project, ids and times replaced.

    The project's file names, the contents of its ``.cell`` and ``.param``
    files, its Journal actions in order, its jobs as ``[label, status]`` and the
    cluster batch items as ``[label, seed name, remote directory]``, each sorted
    where order says nothing.
    """

    document = window._document
    assert document is not None
    files = files_without_ids(document.project_root)
    return {
        "files": [name for name, _text in files],
        "inputs": {name: text for name, text in files if name.endswith((".cell", ".param"))},
        "journal": [record.action for record in document.provenance],
        "jobs": sorted([job.label, job.status.value] for job in document.jobs.values()),
        "sent": sorted(
            [
                item["label"],
                str(item["submit_kwargs"].get("seed_name")),
                str(item["submit_kwargs"].get("remote_directory")),
            ]
            for batch in batches
            for item in batch
        ),
    }


def expected_accepted(name: str) -> dict:
    """Return the stored :func:`accepted_summary` of the accepted submission ``name``."""

    return json.loads(ACCEPTED_SUBMISSIONS.read_text(encoding="utf-8"))[name]
