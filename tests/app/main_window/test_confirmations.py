"""MainWindow confirmations go through the shared helpers with action-named buttons.

``confirm_destructive`` guarantees the safe choice is the default and the
Escape answer (tested in ``tests/app/test_workflow_table_utils.py``); these
tests pin that each prompt uses it, with the label its action button shows,
and that every ``confirm_choice`` question defaults to and escapes with the
safe answer.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.app import workflow_table_utils
from mufinder.domain.enums import JobStatus
from mufinder.services.project_service import (
    ProjectService,
    SiteHostMismatch,
    StructureHostMismatch,
)
from tests.support.jobs import make_job
from tests.support.main_window import open_project_with_host
from tests.support.structures import host_structure


def test_close_prompt_offers_save_dont_save_and_cancel(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered_choices
) -> None:
    window = main_window
    window._document = ProjectService().create_project(tmp_path / "close.mufinder", "Close")
    monkeypatch.setattr(window._project_service, "save_project", lambda _doc: None)

    window.show()
    window.close()

    (prompt,) = answered_choices
    assert prompt["title"] == "Save project?"
    assert prompt["choices"] == [("save", "Save"), ("discard", "Don't save"), ("cancel", "Cancel")]
    assert (prompt["default"], prompt["escape"]) == ("save", "cancel")
    assert prompt["destructive"] == ("discard",)


def test_close_without_saving_skips_the_save(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    window._document = ProjectService().create_project(tmp_path / "close.mufinder", "Close")
    saves: list[bool] = []
    monkeypatch.setattr(window._project_service, "save_project", lambda _doc: saves.append(True))
    monkeypatch.setattr(workflow_table_utils, "confirm_choice", lambda *_a, **_k: "discard")

    window.show()
    window.close()

    assert saves == []
    assert window.isVisible() is False


def test_magnetic_field_workspace_delete_names_its_action(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(window, tmp_path)
    document.ui_state["magnetic_field_workspaces"] = [{"id": "ws-1", "title": "Helix"}]
    answers = [False, True]
    prompts: list[tuple[str, str, str]] = []

    def _confirm(_parent, title, text, detail="", accept_label="Remove", *_args, **_kwargs):
        prompts.append((title, text, accept_label))
        return answers.pop(0)

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _confirm)
    window._delete_magnetic_field_workspace("ws-1")
    assert document.ui_state["magnetic_field_workspaces"] != []

    window._delete_magnetic_field_workspace("ws-1")

    assert document.ui_state["magnetic_field_workspaces"] == []
    assert (
        prompts
        == [
            (
                "Delete magnetic field workspace",
                "Delete magnetic field workspace 'Helix'?",
                "Delete workspace",
            )
        ]
        * 2
    )


def test_dashboard_job_delete_names_how_many_jobs_it_deletes(
    main_window, tmp_path: Path, confirmed_destructive_actions
) -> None:
    window = main_window
    service, document, _host = open_project_with_host(window, tmp_path)
    jobs = [make_job(f"job-{index}", service=service, document=document) for index in range(2)]
    for job in jobs:
        job.status = JobStatus.COMPLETED

    window._delete_dashboard_jobs(jobs)

    (prompt,) = confirmed_destructive_actions
    assert prompt["title"] == "Delete jobs"
    assert prompt["accept_label"] == "Delete 2 jobs"
    assert all(job.id not in document.jobs for job in jobs)


@pytest.mark.parametrize("mismatch_type", [SiteHostMismatch, StructureHostMismatch])
def test_host_mismatch_prompt_defaults_to_cancel(
    main_window, tmp_path: Path, answered_choices, mismatch_type
) -> None:
    window = main_window
    _service, document, host = open_project_with_host(window, tmp_path)
    other = host_structure()
    other.label = "Other host"
    mismatch = mismatch_type(
        active_host_id=host.id,
        active_host_label=host.label,
        offenders=[(uuid4(), "item", other.id, other.label)],
    )
    handler = (
        window._handle_site_host_mismatch
        if mismatch_type is SiteHostMismatch
        else window._handle_structure_host_mismatch
    )

    # The stub answers with the default, Cancel, so the host does not change.
    assert handler(mismatch) is False
    assert document.project.host_structure_id == host.id
    (prompt,) = answered_choices
    assert prompt["choices"] == [("switch", "Switch host"), ("cancel", "Cancel")]
    assert (prompt["default"], prompt["escape"]) == ("cancel", "cancel")
