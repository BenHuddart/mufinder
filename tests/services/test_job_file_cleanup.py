"""Behaviour of the shared job-file clean-up service."""

from __future__ import annotations

from pathlib import Path

import pytest

import mufinder.services.job_file_cleanup as cleanup_module
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result
from mufinder.services.job_file_cleanup import JobCleanupItem, JobFileCleanup
from mufinder.services.project_service import ProjectDocument, ProjectService


def _project(tmp_path: Path) -> tuple[ProjectService, ProjectDocument]:
    service = ProjectService()
    return service, service.create_project(tmp_path / "cleanup.mufinder", "Cleanup")


def _job(service: ProjectService, document: ProjectDocument, label: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_geometry_optimization",
    )


def _cleanup(
    document: ProjectDocument,
    directories: dict[str, Path],
    *,
    active_run_by_job: dict | None = None,
    file_ops: list[str] | None = None,
) -> JobFileCleanup:
    def run_file_op(op, description):
        if file_ops is not None:
            file_ops.append(description)
        return op()

    return JobFileCleanup(
        document_getter=lambda: document,
        job_output_directory=lambda job: directories.get(job.label),
        job_seed_name=lambda job: job.label,
        active_run_by_job=active_run_by_job,
        run_file_op=run_file_op,
    )


def _job_directory(document: ProjectDocument, label: str) -> Path:
    directory = document.project_root / "files" / "pristine_relaxation" / label
    directory.mkdir(parents=True)
    (directory / f"{label}.castep").write_text("output", encoding="utf-8")
    return directory


def test_execute_deletes_files_and_records_through_the_file_op_runner(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    directory = _job_directory(document, job.label)
    file_ops: list[str] = []
    cleanup = _cleanup(document, {job.label: directory}, file_ops=file_ops)
    asked: list[JobCleanupItem] = []

    outcome = cleanup.execute(
        [cleanup.plan_item([job])],
        choose_disposition=lambda item: asked.append(item) or "delete",
        stop_on_error=True,
    )

    assert outcome.completed
    assert [item.label for item in outcome.cleaned] == ["relax_a"]
    assert asked[0].directory == directory
    assert not directory.exists()
    assert job.id not in document.jobs
    assert file_ops == ["Deleting relax_a…"]


def test_execute_archives_into_the_item_archive_root_with_a_unique_name(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    directory = _job_directory(document, job.label)
    archive_root = directory.parent
    (archive_root / "removed_from_workflow" / "relax_a").mkdir(parents=True)
    cleanup = _cleanup(document, {job.label: directory})

    outcome = cleanup.execute(
        [cleanup.plan_item([job], archive_root=archive_root)],
        choose_disposition=lambda _item: "archive",
        stop_on_error=True,
    )

    assert outcome.completed
    assert not directory.exists()
    assert (archive_root / "removed_from_workflow" / "relax_a_1" / "relax_a.castep").exists()
    assert job.id not in document.jobs


def test_execute_applies_an_apply_to_all_answer_to_later_items(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    jobs = [_job(service, document, label) for label in ("relax_a", "relax_b", "relax_c")]
    directories = {job.label: _job_directory(document, job.label) for job in jobs}
    cleanup = _cleanup(document, directories)
    answers = iter(["delete", "delete_all"])
    asked: list[str] = []

    def choose(item):
        asked.append(item.label)
        return next(answers)

    outcome = cleanup.execute([cleanup.plan_item([job]) for job in jobs], choose_disposition=choose, stop_on_error=True)

    assert outcome.completed
    assert asked == ["relax_a", "relax_b"]
    assert not any(directory.exists() for directory in directories.values())
    assert document.jobs == {}


def test_execute_skips_the_prompt_for_an_item_without_files(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    cleanup = _cleanup(document, {job.label: tmp_path / "missing"})

    item = cleanup.plan_item([job])
    outcome = cleanup.execute([item], choose_disposition=pytest.fail, stop_on_error=True)

    assert item.directory is None
    assert outcome.completed
    assert job.id not in document.jobs


def test_execute_stops_on_cancel_and_keeps_the_rest(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    first, second = _job(service, document, "relax_a"), _job(service, document, "relax_b")
    directories = {job.label: _job_directory(document, job.label) for job in (first, second)}
    cleanup = _cleanup(document, directories)
    answers = iter(["delete", None])

    outcome = cleanup.execute(
        [cleanup.plan_item([first]), cleanup.plan_item([second])],
        choose_disposition=lambda _item: next(answers),
        stop_on_error=True,
    )

    assert outcome.cancelled
    assert not outcome.completed
    assert [item.label for item in outcome.cleaned] == ["relax_a"]
    assert directories["relax_b"].exists()
    assert second.id in document.jobs


@pytest.mark.parametrize("stop_on_error", [True, False])
def test_execute_reports_a_file_error_and_keeps_that_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_on_error: bool
) -> None:
    service, document = _project(tmp_path)
    first, second = _job(service, document, "relax_a"), _job(service, document, "relax_b")
    directories = {job.label: _job_directory(document, job.label) for job in (first, second)}
    cleanup = _cleanup(document, directories)
    real_delete = cleanup.delete_path

    def delete_path(path: Path) -> None:
        if path.name == "relax_a":
            raise OSError("busy")
        real_delete(path)

    monkeypatch.setattr(cleanup, "delete_path", delete_path)

    outcome = cleanup.execute(
        [cleanup.plan_item([first]), cleanup.plan_item([second])],
        choose_disposition=lambda _item: "delete",
        stop_on_error=stop_on_error,
    )

    assert [(failure.item.label, str(failure.error)) for failure in outcome.failures] == [("relax_a", "busy")]
    assert first.id in document.jobs
    assert (second.id in document.jobs) is stop_on_error
    assert directories["relax_b"].exists() is stop_on_error


def test_plan_item_refuses_a_directory_another_job_uses(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    owner, other = _job(service, document, "relax_a"), _job(service, document, "relax_b")
    directory = _job_directory(document, owner.label)
    cleanup = _cleanup(document, {"relax_a": directory, "relax_b": directory / "nested"})

    refused = cleanup.plan_item([owner], label="Row A", check_ownership=True)
    together = cleanup.plan_item([owner, other], check_ownership=True)
    outcome = cleanup.execute([refused], choose_disposition=pytest.fail, stop_on_error=False)

    assert refused.label == "Row A"
    assert refused.refusal == f"- relax_a: another job uses the same output folder ({directory.resolve()})"
    assert together.refusal is None
    assert outcome.failures[0].refusal == refused.refusal
    assert directory.exists()
    assert owner.id in document.jobs


def test_directory_ownership_refusals_checks_the_folder_and_cluster_directory(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    outside, misnamed, remote = (_job(service, document, label) for label in ("job_out", "job_name", "job_remote"))
    remote.execution_target = "alpha"
    remote.remote_directory = "/work/other_name"
    directories = {
        "job_out": tmp_path / "elsewhere" / "job_out",
        "job_name": document.project_root / "files" / "pristine_relaxation" / "something_else",
    }
    cleanup = _cleanup(document, directories)

    refusals = cleanup.directory_ownership_refusals([outside, misnamed, remote])

    assert "outside the project's job folders" in refusals[outside.id]
    assert "not named after the job" in refusals[misnamed.id]
    assert refusals[remote.id] == "- job_remote: its cluster directory is not named after the job (/work/other_name)"
    assert cleanup.directory_ownership_refusals([remote], check_remote=False) == {}


def test_delete_job_records_removes_runs_results_mappings_and_relaxations(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    job, kept = _job(service, document, "relax_a"), _job(service, document, "relax_b")
    run = CalculationRun(job_id=job.id)
    result = Result(calculation_run_id=run.id)
    document.runs[run.id] = run
    document.results[result.id] = result
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        parameters={"job_ids_by_label": {"row_a": str(job.id), "row_b": str(kept.id)}},
    )
    document.calculations[definition.id] = definition
    active_run_by_job = {job.id: run.id}
    cleanup = _cleanup(document, {}, active_run_by_job=active_run_by_job)

    cleanup.delete_job_records(job.id)

    assert job.id not in document.jobs
    assert run.id not in document.runs
    assert result.id not in document.results
    assert definition.parameters["job_ids_by_label"] == {"row_b": str(kept.id)}
    assert active_run_by_job == {}


def test_delete_path_retries_a_transient_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "transient_dir"
    target.mkdir()
    (target / "inner.txt").write_text("x", encoding="utf-8")
    real_rmtree = cleanup_module.shutil.rmtree
    calls = {"n": 0}

    def flaky_rmtree(path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "Access is denied", str(path))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(cleanup_module.shutil, "rmtree", flaky_rmtree)
    monkeypatch.setattr(cleanup_module.time, "sleep", lambda _delay: None)

    JobFileCleanup(
        document_getter=lambda: None, job_output_directory=lambda _job: None, job_seed_name=lambda job: job.label
    ).delete_path(target)

    assert calls["n"] >= 2
    assert not target.exists()
