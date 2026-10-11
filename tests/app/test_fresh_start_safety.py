"""Starting a relaxation again deletes only files that belong to that job.

Start again empties the job's local output directory and deletes its remote
directory. These tests pin the guards that keep those deletions scoped: a
directory shared with another job (reconfiguring a row can leave an old job and
its replacement in one directory), a directory outside the project's job
folders or not named after the job, symlinks, inputs that look like output,
and every way of backing out before anything is deleted.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import FailureReason, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition
from mufinder.services.project_service import ProjectService
from tests.support.main_window import (
    patch_continuation_cluster_dialog,
    patch_continuation_cluster_runner,
)

SEED = "pristine_relax_unit"


def _project(tmp_path: Path):
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / "safety.mufinder", "Safety")
    window._document = doc
    return window, service, doc


def _pristine_job(service, doc, status: JobStatus, *, row_key: str = "unit"):
    """Add a pristine relaxation job bound to its own definition, as Configure does."""

    job = service.add_job(
        doc,
        label=SEED,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    job.status = status
    job.failure_reason = FailureReason.WALLTIME if status == JobStatus.STOPPED else None
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
        parameters={
            "configuration": {},
            "output_directory": "files/pristine_relaxation",
            "job_ids_by_label": {row_key: str(job.id)},
        },
        input_structure_ids=[],
    )
    doc.calculations[definition.id] = definition
    return job


def _job_directory(doc, seed: str = SEED) -> Path:
    directory = doc.project_root / "files" / "pristine_relaxation" / seed
    directory.mkdir(parents=True, exist_ok=True)
    for name in (f"{seed}.cell", f"{seed}.param", f"{seed}.castep", f"{seed}-out.cell", f"{seed}.check"):
        (directory / name).write_text("data\n", encoding="utf-8")
    return directory


def _on_cluster(job, remote_directory: str = f"/remote/work/{SEED}", remote_job_id: str = "4242"):
    job.execution_target = "alpha"
    job.remote_directory = remote_directory
    job.remote_job_id = remote_job_id


ALL_FILES = sorted(
    [f"{SEED}.cell", f"{SEED}.param", f"{SEED}.castep", f"{SEED}-out.cell", f"{SEED}.check"]
)


# --- Ownership --------------------------------------------------------------


def test_a_job_in_its_own_directory_is_cleaned(qapp, tmp_path: Path) -> None:
    window, service, doc = _project(tmp_path)
    job = _pristine_job(service, doc, JobStatus.STOPPED)
    directory = _job_directory(doc)

    assert window._fresh_start_refusals([job]) == {}
    cleaned, errors = window._clear_outputs_for_fresh_start([job])

    assert cleaned == [job] and errors == []
    assert sorted(path.name for path in directory.iterdir()) == [f"{SEED}.cell", f"{SEED}.param"]
    window.close()


def test_a_directory_shared_with_another_job_is_never_cleaned(qapp, tmp_path: Path) -> None:
    window, service, doc = _project(tmp_path)
    old = _pristine_job(service, doc, JobStatus.STOPPED)
    _replacement = _pristine_job(service, doc, JobStatus.COMPLETED)
    directory = _job_directory(doc)

    cleaned, errors = window._clear_outputs_for_fresh_start([old])

    assert cleaned == []
    assert errors == [f"- {SEED}: another job uses the same output folder ({directory.resolve()})"]
    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    window.close()


def test_a_remote_directory_shared_with_another_job_is_refused(qapp, tmp_path: Path) -> None:
    window, service, doc = _project(tmp_path)
    job = _pristine_job(service, doc, JobStatus.STOPPED)
    _job_directory(doc)
    _on_cluster(job)
    other = service.add_job(
        doc, label="elsewhere", workflow_stage=WorkflowStage.MUON_RELAXATION, calc_type="castep"
    )
    _on_cluster(other, remote_directory=f"/remote/work/{SEED}/nested")

    assert window._fresh_start_refusals([job]) == {
        job.id: f"- {SEED}: another job uses the same cluster directory (/remote/work/{SEED})"
    }
    window.close()


def test_the_same_remote_path_on_another_cluster_is_not_shared(qapp, tmp_path: Path) -> None:
    window, service, doc = _project(tmp_path)
    job = _pristine_job(service, doc, JobStatus.STOPPED)
    _job_directory(doc)
    _on_cluster(job)
    other = service.add_job(
        doc, label="elsewhere", workflow_stage=WorkflowStage.MUON_RELAXATION, calc_type="castep"
    )
    _on_cluster(other)
    other.execution_target = "beta"

    assert window._fresh_start_refusals([job]) == {}
    window.close()


def test_directories_not_named_after_the_job_are_refused(qapp, tmp_path: Path) -> None:
    window, service, doc = _project(tmp_path)
    job = _pristine_job(service, doc, JobStatus.STOPPED)
    _job_directory(doc)

    window._job_seed_name = lambda _job: SEED
    _on_cluster(job, remote_directory="/remote/work")
    remote_refusal = window._fresh_start_refusals([job])[job.id]

    _on_cluster(job)
    window._job_seed_name = lambda _job: "another_seed"
    local_refusal = window._fresh_start_refusals([job])[job.id]

    assert remote_refusal == f"- {SEED}: its cluster directory is not named after the job (/remote/work)"
    assert local_refusal.startswith(f"- {SEED}: its output folder is not named after the job")
    window.close()


@pytest.mark.parametrize("location", ["outside", "stage_root"])
def test_directories_outside_the_job_folders_are_refused(qapp, tmp_path: Path, location: str) -> None:
    window, service, doc = _project(tmp_path)
    job = _pristine_job(service, doc, JobStatus.STOPPED)
    directory = (
        tmp_path / "elsewhere" / SEED
        if location == "outside"
        else doc.project_root / "files" / "pristine_relaxation"
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "keep.castep").write_text("data", encoding="utf-8")
    window._job_output_directory = lambda _job: directory
    window._job_seed_name = lambda _job: directory.name

    cleaned, errors = window._clear_outputs_for_fresh_start([job])

    assert cleaned == []
    assert "outside the project's job folders" in errors[0]
    assert (directory / "keep.castep").exists()
    window.close()


def test_dashboard_start_again_never_touches_a_running_replacements_directory(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, service, doc = _project(tmp_path)
    old = _pristine_job(service, doc, JobStatus.STOPPED)
    replacement = _pristine_job(service, doc, JobStatus.RUNNING)
    directory = _job_directory(doc)
    _on_cluster(old)
    _on_cluster(replacement, remote_job_id="5000")

    monkeypatch.setattr(
        window,
        "_choose_relaxation_resubmit_mode",
        lambda _jobs: pytest.fail("a refused job is never offered the choice"),
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox, "warning", lambda _parent, _title, text: warnings.append(text)
    )
    submissions: list = []
    monkeypatch.setattr(window, "_submit_jobs_on_cluster", lambda jobs, **kwargs: submissions.append(jobs))

    window._on_resubmit_from_dashboard([old.id])

    # Neither started again nor continued in the replacement's directory.
    assert submissions == []
    assert f"- {SEED}: another job uses the same output folder" in warnings[0]
    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    window.close()


def test_two_fresh_starts_sharing_a_remote_directory_delete_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, service, doc = _project(tmp_path)
    first = _pristine_job(service, doc, JobStatus.STOPPED)
    second = _pristine_job(service, doc, JobStatus.FAILED)
    _job_directory(doc)
    _on_cluster(first)
    _on_cluster(second, remote_job_id="4243")

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(main_window_module, "show_detailed_warning", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster(
        [first, second],
        eligible_statuses={JobStatus.STOPPED, JobStatus.FAILED},
        reuse_existing_remote_files=True,
        fresh_start_job_ids={first.id, second.id},
    )

    assert recorded["deleted"] == []
    assert recorded["submitted"] == []
    window.close()


def test_a_fresh_start_into_a_protected_directory_is_dropped(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Defence in depth: never submit a fresh start whose directory survives."""

    window, service, doc = _project(tmp_path)
    fresh = _pristine_job(service, doc, JobStatus.STOPPED)
    _job_directory(doc)
    _on_cluster(fresh)
    continuing = _pristine_job(service, doc, JobStatus.DEFINED, row_key="other")
    continuing.execution_target = "alpha"
    continuing.remote_directory = fresh.remote_directory
    # Bypass the ownership check, which would refuse these two first.
    monkeypatch.setattr(window, "_fresh_start_refusals", lambda _jobs: {})

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    failures: list[str] = []
    monkeypatch.setattr(
        main_window_module, "show_detailed_warning", lambda _parent, _title, text, **_kw: failures.append(text)
    )

    window._submit_jobs_on_cluster(
        [fresh, continuing],
        eligible_statuses={JobStatus.STOPPED, JobStatus.DEFINED},
        reuse_existing_remote_files=True,
        fresh_start_job_ids={fresh.id},
    )

    assert recorded["deleted"] == []
    assert [item["remote_directory"] for item in recorded["submitted"]] == [continuing.remote_directory]
    assert "so it was not started again" in failures[0]
    window.close()


# --- Backing out deletes nothing ---------------------------------------------


def test_cancelling_the_cluster_dialog_deletes_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, service, doc = _project(tmp_path)
    job = _pristine_job(service, doc, JobStatus.STOPPED)
    directory = _job_directory(doc)
    _on_cluster(job)

    patch_continuation_cluster_dialog(monkeypatch)
    monkeypatch.setattr(main_window_module.ClusterSubmissionDialog, "exec", lambda _self: 0)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)

    window._submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.STOPPED},
        reuse_existing_remote_files=True,
        fresh_start_job_ids={job.id},
    )

    assert recorded["deleted"] == [] and recorded["submitted"] == []
    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    window.close()


def test_a_failed_cluster_connection_deletes_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, service, doc = _project(tmp_path)
    job = _pristine_job(service, doc, JobStatus.STOPPED)
    directory = _job_directory(doc)
    _on_cluster(job)

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)

    def _refuse(*_args, **_kwargs):
        raise ConnectionError("host unreachable")

    monkeypatch.setattr(window, "_connect_to_cluster_with_trust", _refuse)
    monkeypatch.setattr(main_window_module, "show_detailed_warning", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.STOPPED},
        reuse_existing_remote_files=True,
        fresh_start_job_ids={job.id},
    )

    assert recorded["deleted"] == [] and recorded["submitted"] == []
    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    window.close()


def test_cancelling_the_local_dialog_deletes_nothing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, service, doc = _project(tmp_path)
    job = _pristine_job(service, doc, JobStatus.STOPPED)
    job.execution_target = "local"
    directory = _job_directory(doc)

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: SimpleNamespace(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _CancelledDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _CancelledDialog)

    window._submit_jobs_locally(
        [job], eligible_statuses={JobStatus.STOPPED}, fresh_start_job_ids={job.id}
    )

    assert sorted(path.name for path in directory.iterdir()) == ALL_FILES
    window.close()


# --- What the clean removes --------------------------------------------------


def test_clean_keeps_the_input_of_a_seed_ending_in_out(tmp_path: Path) -> None:
    directory = tmp_path / "pristine_relax_relax-out"
    directory.mkdir()
    for name in ("pristine_relax_relax-out.cell", "pristine_relax_relax-out.param", "pristine_relax_relax-out-out.cell"):
        (directory / name).write_text("data", encoding="utf-8")

    MainWindow._clear_directory_contents(directory)

    assert sorted(path.name for path in directory.iterdir()) == [
        "pristine_relax_relax-out.cell",
        "pristine_relax_relax-out.param",
    ]


def test_clean_removes_symlinks_without_following_them(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "H.usp").write_text("pseudopotential", encoding="utf-8")
    outside_file = tmp_path / "notes.txt"
    outside_file.write_text("keep", encoding="utf-8")
    directory = tmp_path / "seed"
    directory.mkdir()
    (directory / "seed.cell").write_text("input", encoding="utf-8")
    (directory / "seed.check").write_text("checkpoint", encoding="utf-8")
    os.symlink(shared, directory / "pseudos")
    os.symlink(outside_file, directory / "notes.txt")

    MainWindow._clear_directory_contents(directory)

    assert sorted(path.name for path in directory.iterdir()) == ["seed.cell"]
    assert (shared / "H.usp").read_text(encoding="utf-8") == "pseudopotential"
    assert outside_file.read_text(encoding="utf-8") == "keep"
