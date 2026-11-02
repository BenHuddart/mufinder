"""Behaviour of the shared job-file clean-up service."""

from __future__ import annotations

from pathlib import Path

import pytest

import mufinder.services.job_file_cleanup as cleanup_module
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result
from mufinder.domain.text import KeptFolder
from mufinder.services.job_file_cleanup import (
    ARCHIVE_FOLDER_NAME,
    KEPT_FOR_ANOTHER_JOB,
    JobCleanupItem,
    JobFileCleanup,
)
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

    outcome = cleanup.execute(
        [cleanup.plan_item([job]) for job in jobs], choose_disposition=choose, stop_on_error=True
    )

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


def test_execute_does_not_ask_again_for_a_directory_an_earlier_item_removed(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    old, replacement = _job(service, document, "relax_a"), _job(service, document, "relax_b")
    directory = _job_directory(document, "relax_a")
    cleanup = _cleanup(document, {"relax_a": directory, "relax_b": directory})
    plan = [cleanup.plan_item([old]), cleanup.plan_item([replacement])]
    asked: list[str] = []

    # Cancel would be the answer for a second prompt; there must not be one.
    answers = iter(["delete", None])
    outcome = cleanup.execute(
        plan,
        choose_disposition=lambda item: asked.append(item.label) or next(answers),
        stop_on_error=True,
    )

    assert asked == ["relax_a"]
    assert outcome.completed
    assert not directory.exists()
    assert document.jobs == {}


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

    assert [(failure.item.label, str(failure.error)) for failure in outcome.failures] == [
        ("relax_a", "busy")
    ]
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
    assert (
        refused.refusal
        == f"- relax_a: another job uses the same output folder ({directory.resolve()})"
    )
    assert together.refusal is None
    assert outcome.failures[0].refusal == refused.refusal
    assert directory.exists()
    assert owner.id in document.jobs


def test_directory_ownership_refusals_checks_the_folder_and_cluster_directory(
    tmp_path: Path,
) -> None:
    service, document = _project(tmp_path)
    outside, misnamed, remote = (
        _job(service, document, label) for label in ("job_out", "job_name", "job_remote")
    )
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
    assert (
        refusals[remote.id]
        == "- job_remote: its cluster directory is not named after the job (/work/other_name)"
    )
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


def test_delete_path_retries_a_transient_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        document_getter=lambda: None,
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
    ).delete_path(target)

    assert calls["n"] >= 2
    assert not target.exists()


def test_a_folder_another_job_uses_is_not_offered_and_only_the_records_go(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    other = _job(service, document, "relax_b")
    other.status = JobStatus.COMPLETED
    directory = _job_directory(document, job.label)
    cleanup = _cleanup(document, {job.label: directory, other.label: directory})
    asked: list[JobCleanupItem] = []

    outcome = cleanup.execute(
        [cleanup.plan_item([job])],
        choose_disposition=lambda item: asked.append(item) or "archive",
        stop_on_error=True,
    )

    assert outcome.completed
    assert asked == []
    assert directory.is_dir()
    assert not (directory.parent / ARCHIVE_FOLDER_NAME).exists()
    assert job.id not in document.jobs
    assert other.id in document.jobs
    assert [entry.directory for entry in outcome.kept] == [directory]
    assert outcome.kept_folders == (
        KeptFolder("files/pristine_relaxation/relax_a", KEPT_FOR_ANOTHER_JOB),
    )


def test_a_folder_holding_or_inside_another_jobs_folder_is_kept(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    outer = _job(service, document, "outer")
    inner = _job(service, document, "inner")
    inner.status = JobStatus.COMPLETED
    outer.status = JobStatus.COMPLETED
    outer_directory = _job_directory(document, outer.label)
    inner_directory = outer_directory / inner.label
    inner_directory.mkdir()
    directories = {outer.label: outer_directory, inner.label: inner_directory}

    for job in (outer, inner):
        cleanup = _cleanup(document, directories)
        document.jobs[job.id] = job
        outcome = cleanup.execute(
            [cleanup.plan_item([job])],
            choose_disposition=lambda _item: "delete",
            stop_on_error=True,
        )
        assert len(outcome.kept) == 1
        document.jobs[job.id] = job

    assert inner_directory.is_dir()


def test_jobs_cleaned_together_may_share_a_folder(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    old = _job(service, document, "relax_a")
    replacement = _job(service, document, "relax_b")
    replacement.status = JobStatus.COMPLETED
    directory = _job_directory(document, old.label)
    cleanup = _cleanup(document, {old.label: directory, replacement.label: directory})

    outcome = cleanup.execute(
        [cleanup.plan_item([old]), cleanup.plan_item([replacement])],
        choose_disposition=lambda _item: "delete",
        stop_on_error=True,
    )

    assert outcome.completed
    assert outcome.kept == ()
    assert outcome.deleted == (directory,)
    assert not directory.exists()


def test_an_outcome_lists_the_folders_of_each_fate(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    mine = _job(service, document, "relax_a")
    moved = _job(service, document, "relax_b")
    shared = _job(service, document, "relax_c")
    other = _job(service, document, "relax_d")
    other.status = JobStatus.COMPLETED
    mine_directory = _job_directory(document, mine.label)
    moved_directory = _job_directory(document, moved.label)
    shared_directory = _job_directory(document, shared.label)
    cleanup = _cleanup(
        document,
        {
            mine.label: mine_directory,
            moved.label: moved_directory,
            shared.label: shared_directory,
            other.label: shared_directory,
        },
    )
    answers = {mine.label: "delete", moved.label: "archive"}

    outcome = cleanup.execute(
        [cleanup.plan_item([mine]), cleanup.plan_item([moved]), cleanup.plan_item([shared])],
        choose_disposition=lambda item: answers[item.label],
        stop_on_error=True,
    )

    assert outcome.deleted == (mine_directory,)
    assert outcome.archived == (moved_directory,)
    assert [folder.name for folder in outcome.kept_folders] == ["files/pristine_relaxation/relax_c"]


def test_an_outcome_with_nothing_kept_lists_no_kept_folder(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    directory = _job_directory(document, job.label)
    cleanup = _cleanup(document, {job.label: directory})

    outcome = cleanup.execute(
        [cleanup.plan_item([job])], choose_disposition=lambda _item: "delete", stop_on_error=True
    )

    assert outcome.kept_folders == ()


def _cutoff_job(service: ProjectService, document: ProjectDocument, label: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )


def test_a_detached_cutoff_job_in_the_same_folder_does_not_hold_it(tmp_path: Path) -> None:
    """A reset cutoff job stays in the project with nothing of its own in the folder."""

    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    detached = _cutoff_job(service, document, "relax_b")
    directory = _job_directory(document, job.label)
    cleanup = _cleanup(document, {job.label: directory, detached.label: directory})

    outcome = cleanup.execute(
        [cleanup.plan_item([job])], choose_disposition=lambda _item: "delete", stop_on_error=True
    )

    assert outcome.kept == ()
    assert not directory.exists()


def test_an_unbound_defined_job_of_another_workflow_holds_its_folder(tmp_path: Path) -> None:
    """An older definition can bind a job by label, so only a detached cutoff is let off."""

    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    legacy = _job(service, document, "relax_b")
    directory = _job_directory(document, job.label)
    cleanup = _cleanup(document, {job.label: directory, legacy.label: directory})

    outcome = cleanup.execute(
        [cleanup.plan_item([job])], choose_disposition=lambda _item: "delete", stop_on_error=True
    )

    assert len(outcome.kept) == 1
    assert directory.is_dir()


def test_an_unbound_defined_cutoff_job_inside_the_folder_holds_it(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    legacy = _cutoff_job(service, document, "relax_b")
    directory = _job_directory(document, job.label)
    inner = directory / legacy.label
    inner.mkdir()
    cleanup = _cleanup(document, {job.label: directory, legacy.label: inner})

    outcome = cleanup.execute(
        [cleanup.plan_item([job])], choose_disposition=lambda _item: "delete", stop_on_error=True
    )

    assert len(outcome.kept) == 1
    assert inner.is_dir()


def test_a_defined_job_bound_to_a_definition_does_hold_its_folder(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    bound = _job(service, document, "relax_b")
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_geometry_optimization",
        parameters={"job_ids_by_label": {"row": str(bound.id)}},
    )
    document.calculations[definition.id] = definition
    directory = _job_directory(document, job.label)
    cleanup = _cleanup(document, {job.label: directory, bound.label: directory})

    outcome = cleanup.execute(
        [cleanup.plan_item([job])], choose_disposition=lambda _item: "delete", stop_on_error=True
    )

    assert len(outcome.kept) == 1
    assert directory.is_dir()


def test_a_cancel_between_two_items_cannot_part_a_shared_folder_from_its_jobs(
    tmp_path: Path,
) -> None:
    """Items sharing a folder run one after another, whatever the plan's order."""

    service, document = _project(tmp_path)
    first = _job(service, document, "relax_a")
    between = _job(service, document, "relax_b")
    last = _job(service, document, "relax_c")
    shared = _job_directory(document, first.label)
    own = _job_directory(document, between.label)
    cleanup = _cleanup(document, {first.label: shared, between.label: own, last.label: shared})
    answers = {first.label: "delete", between.label: None}

    outcome = cleanup.execute(
        [cleanup.plan_item([first]), cleanup.plan_item([between]), cleanup.plan_item([last])],
        choose_disposition=lambda item: answers[item.label],
        stop_on_error=True,
    )

    assert outcome.cancelled
    assert not shared.exists()
    assert first.id not in document.jobs
    assert last.id not in document.jobs
    assert own.is_dir()
    assert between.id in document.jobs


def test_a_refused_item_keeps_a_folder_it_lives_inside(tmp_path: Path) -> None:
    service, document = _project(tmp_path)
    outer = _job(service, document, "relax_a")
    inner = _job(service, document, "relax_b")
    outer_directory = _job_directory(document, outer.label)
    inner_directory = outer_directory / inner.label
    inner_directory.mkdir()
    cleanup = _cleanup(document, {outer.label: outer_directory, inner.label: inner_directory})
    refused = JobCleanupItem(
        label=inner.label, jobs=(inner,), directory=inner_directory, refusal="- no"
    )
    inner.status = JobStatus.COMPLETED

    outcome = cleanup.execute(
        [cleanup.plan_item([outer]), refused],
        choose_disposition=lambda _item: "delete",
        stop_on_error=False,
    )

    assert inner_directory.is_dir()
    assert inner.id in document.jobs
    assert [entry.directory for entry in outcome.kept] == [outer_directory]


def test_a_job_an_item_releases_does_not_keep_its_folder_and_keeps_its_record(
    tmp_path: Path,
) -> None:
    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    released = _job(service, document, "relax_b")
    released.status = JobStatus.COMPLETED
    directory = _job_directory(document, job.label)
    cleanup = _cleanup(document, {job.label: directory, released.label: directory})
    item = JobCleanupItem(
        label=job.label, jobs=(job,), directory=directory, releases=frozenset({released.id})
    )

    outcome = cleanup.execute([item], choose_disposition=lambda _item: "delete", stop_on_error=True)

    assert outcome.kept == ()
    assert not directory.exists()
    assert released.id in document.jobs


def test_a_folder_holding_one_that_failed_to_clean_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, document = _project(tmp_path)
    inner_job = _job(service, document, "relax_a")
    outer_job = _job(service, document, "relax_b")
    outer = _job_directory(document, outer_job.label)
    inner = outer / inner_job.label
    inner.mkdir()
    (inner / "kept.castep").write_text("kept", encoding="utf-8")
    cleanup = _cleanup(document, {inner_job.label: inner, outer_job.label: outer})

    def _refuse_inner(path: Path) -> None:
        raise PermissionError(13, "denied")

    monkeypatch.setattr(cleanup, "delete_path", _refuse_inner)

    outcome = cleanup.execute(
        [cleanup.plan_item([inner_job]), cleanup.plan_item([outer_job])],
        choose_disposition=lambda _item: "delete",
        stop_on_error=False,
    )

    assert (inner / "kept.castep").is_file()
    assert inner_job.id in document.jobs
    assert [failure.item.label for failure in outcome.failures] == [inner_job.label]
    assert [entry.directory for entry in outcome.kept] == [outer]


def test_a_prompt_cannot_come_between_folders_linked_through_a_third(tmp_path: Path) -> None:
    """``outer/a`` brings ``outer``, which brings ``outer/b``, before anything unrelated."""

    service, document = _project(tmp_path)
    a_job = _job(service, document, "a")
    unrelated_job = _job(service, document, "unrelated")
    outer_job = _job(service, document, "outer")
    b_job = _job(service, document, "b")
    outer = _job_directory(document, outer_job.label)
    (outer / "a").mkdir()
    (outer / "b").mkdir()
    unrelated = _job_directory(document, unrelated_job.label)
    cleanup = _cleanup(
        document,
        {
            a_job.label: outer / "a",
            unrelated_job.label: unrelated,
            outer_job.label: outer,
            b_job.label: outer / "b",
        },
    )
    asked: list[str] = []

    def _answer(item: JobCleanupItem):
        asked.append(item.label)
        return None if item.label == unrelated_job.label else "delete"

    outcome = cleanup.execute(
        [cleanup.plan_item([job]) for job in (a_job, unrelated_job, outer_job, b_job)],
        choose_disposition=_answer,
        stop_on_error=True,
    )

    assert outcome.cancelled
    assert asked[-1] == unrelated_job.label
    assert not outer.exists()
    assert b_job.id not in document.jobs
    assert unrelated.is_dir()
    assert unrelated_job.id in document.jobs


def test_a_folder_holding_one_refused_at_its_turn_is_kept(tmp_path: Path) -> None:
    """A folder can turn into a link while an earlier prompt is open."""

    service, document = _project(tmp_path)
    first_job = _job(service, document, "relax_a")
    inner_job = _job(service, document, "relax_b")
    outer_job = _job(service, document, "relax_c")
    first = _job_directory(document, first_job.label)
    outer = _job_directory(document, outer_job.label)
    inner = outer / inner_job.label
    inner.mkdir()
    (outer / "real").mkdir()
    (outer / "real" / "kept.castep").write_text("kept", encoding="utf-8")
    cleanup = _cleanup(
        document, {first_job.label: first, inner_job.label: inner, outer_job.label: outer}
    )

    def _answer(item: JobCleanupItem):
        if item.label == first_job.label:
            inner.rmdir()
            inner.symlink_to(outer / "real", target_is_directory=True)
        return "delete"

    outcome = cleanup.execute(
        [cleanup.plan_item([job]) for job in (first_job, inner_job, outer_job)],
        choose_disposition=_answer,
        stop_on_error=False,
    )

    assert (outer / "real" / "kept.castep").is_file()
    assert inner_job.id in document.jobs
    assert [entry.directory for entry in outcome.kept] == [outer]


def test_a_job_whose_saved_name_leaves_its_place_does_not_hold_a_folder(tmp_path: Path) -> None:
    """``..`` reads as a whole workflow's folder, which would otherwise hold every job's."""

    service, document = _project(tmp_path)
    job = _job(service, document, "relax_a")
    stray = _job(service, document, "relax_b")
    stray.status = JobStatus.COMPLETED
    directory = _job_directory(document, job.label)
    cleanup = _cleanup(
        document, {job.label: directory, stray.label: directory.parent.parent / ".." / "files"}
    )

    outcome = cleanup.execute(
        [cleanup.plan_item([job])], choose_disposition=lambda _item: "delete", stop_on_error=True
    )

    assert outcome.kept == ()
    assert not directory.exists()


def test_a_folder_a_later_item_takes_is_not_reported_as_kept(tmp_path: Path) -> None:
    """A saved name leads the first item into the folder that is the second's own."""

    service, document = _project(tmp_path)
    stray = service.add_job(
        document,
        label="stray",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
    )
    owner = _job(service, document, "relax_a")
    directory = _job_directory(document, owner.label)
    cleanup = _cleanup(document, {stray.label: directory, owner.label: directory})

    outcome = cleanup.execute(
        [cleanup.plan_item([stray]), cleanup.plan_item([owner])],
        choose_disposition=lambda _item: "delete",
        stop_on_error=True,
    )

    assert not directory.exists()
    assert outcome.deleted == (directory,)
    assert outcome.kept == ()


def test_a_folder_holding_a_folder_just_moved_is_not_offered(tmp_path: Path) -> None:
    """The default archive place of a nested folder is inside the folder around it."""

    service, document = _project(tmp_path)
    inner_job = _job(service, document, "relax_a")
    outer_job = _job(service, document, "relax_b")
    outer = _job_directory(document, outer_job.label)
    inner = outer / inner_job.label
    inner.mkdir()
    (inner / "moved.castep").write_text("kept", encoding="utf-8")
    cleanup = _cleanup(document, {inner_job.label: inner, outer_job.label: outer})
    asked: list[str] = []

    def _answer(item: JobCleanupItem):
        asked.append(item.label)
        return "archive"

    outcome = cleanup.execute(
        [cleanup.plan_item([inner_job]), cleanup.plan_item([outer_job])],
        choose_disposition=_answer,
        stop_on_error=True,
    )

    assert asked == [inner_job.label]
    assert (outer / ARCHIVE_FOLDER_NAME / inner_job.label / "moved.castep").is_file()
    assert [entry.directory for entry in outcome.kept] == [outer]
    assert outcome.archived == (inner,)
