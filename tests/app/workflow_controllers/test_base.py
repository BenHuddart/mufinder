"""Tests for the per-workflow controller base class.

The base :class:`WorkflowJobController` is exercised through a *fake* controller
so that the template-method ``submit`` flow, the no-ready path, the grouped
submission override, and the ``merge_controller_result`` result-dispatch hook are
proven independently of any concrete workflow. The module-level helpers in
``mufinder.app.workflow_controllers.base`` (``dedupe_jobs``,
``apply_row_parameters_to_form_state``) are tested here too.
"""

from __future__ import annotations

from pathlib import Path

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.project_service import ProjectService

from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    dedupe_jobs,
    merge_controller_result,
    validate_controller_artifacts,
)

from tests.support.jobs import make_job
from tests.support.controllers import (
    FakeRowState,
    RecordingBackend,
    completed_castep_result,
    make_context,
    no_artifacts,
)


# --------------------------------------------------------------------------- #
# Fake controller
# --------------------------------------------------------------------------- #


class _FakeController(WorkflowJobController):
    """Fake controller exercising the base template method + result hook."""

    calc_types = frozenset({"fake_calc"})
    no_ready_message = "No ready fake rows."

    def __init__(
        self,
        context: WorkflowControllerContext,
        *,
        states: dict[str, FakeRowState],
        existing: dict[str, Job] | None = None,
        materialized: dict[str, Job] | None = None,
        group_by_size: bool = False,
        pre_submit_ok: bool = True,
        build_result_payload: dict[str, object] | None = None,
    ) -> None:
        super().__init__(context)
        self._states = states
        self._existing = existing or {}
        self._materialized = materialized or {}
        self._group_by_size = group_by_size
        self._pre_submit_ok = pre_submit_ok
        self._build_result_payload = build_result_payload
        self.refreshed = 0
        self.materialize_calls: list[list[object]] = []

    def normalize(self, value: object) -> str:
        return str(value)

    def build_states(self) -> dict[str, FakeRowState]:
        return self._states

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        return [self._existing[k] for k in keys if k in self._existing]

    def materialize(self, keys: list[object]) -> list[Job]:
        self.materialize_calls.append(list(keys))
        return [self._materialized[k] for k in keys if k in self._materialized]

    def pre_submit_check(self, keys: list[object]) -> bool:
        return self._pre_submit_ok

    def submission_groups(self, jobs: list[Job]):
        if not self._group_by_size:
            return super().submission_groups(jobs)
        # Split into per-job groups, each carrying an index kwarg, to prove the
        # override path forwards extra kwargs to the backend.
        return [([job], {"index": i}) for i, job in enumerate(jobs)]

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        if self._build_result_payload is None:
            return super().build_result(job, parsed, run, artifacts)
        return ControllerResult(
            payload=dict(self._build_result_payload),
            output_files=["extra.out"],
            warnings=["controller warning"],
        )

    def refresh_view(self) -> None:
        self.refreshed += 1


# --------------------------------------------------------------------------- #
# Base template-method: submit()
# --------------------------------------------------------------------------- #


def test_submit_local_dispatches_planned_jobs_to_local_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    existing = make_job("existing", service=service, document=document, calc_type="fake_calc")
    fresh = make_job("fresh", service=service, document=document, calc_type="fake_calc")

    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(local_backend=local, cluster_backend=cluster)
    controller = _FakeController(
        context,
        states={"a": FakeRowState("ready"), "b": FakeRowState("ready")},
        existing={"a": existing},
        materialized={"b": fresh},
    )

    controller.submit(["a", "b"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted_jobs, extras = local.calls[0]
    assert [j.id for j in submitted_jobs] == [existing.id, fresh.id]
    assert extras == {}


def test_submit_cluster_dispatches_to_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    fresh = make_job("fresh", service=service, document=document, calc_type="fake_calc")

    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(local_backend=local, cluster_backend=cluster)
    controller = _FakeController(
        context,
        states={"b": FakeRowState("ready")},
        materialized={"b": fresh},
    )

    controller.submit(["b"], target="cluster")

    assert local.calls == []
    assert [j.id for j in cluster.calls[0][0]] == [fresh.id]


def test_submit_shows_info_when_no_ready_jobs(tmp_path: Path) -> None:
    service = ProjectService()
    service.create_project(tmp_path / "p.mufinder", "P")
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(local_backend=local, infos=infos)
    controller = _FakeController(
        context,
        # No 'ready' rows and no jobs → empty plan.
        states={"a": FakeRowState("failed")},
    )

    controller.submit(["a"], target="local")

    assert local.calls == []
    assert infos == [("No ready jobs", "No ready fake rows.")]


def test_submit_pre_submit_check_short_circuits(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    fresh = make_job("fresh", service=service, document=document, calc_type="fake_calc")
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(local_backend=local, infos=infos)
    controller = _FakeController(
        context,
        states={"b": FakeRowState("ready")},
        materialized={"b": fresh},
        pre_submit_ok=False,
    )

    controller.submit(["b"], target="local")

    assert local.calls == []
    assert infos == []  # pre-submit refusal is silent (no "no ready jobs")


def test_submit_grouped_override_forwards_extra_kwargs(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job_a = make_job("a", service=service, document=document, calc_type="fake_calc")
    job_b = make_job("b", service=service, document=document, calc_type="fake_calc")
    local = RecordingBackend()
    context = make_context(local_backend=local)
    controller = _FakeController(
        context,
        states={"a": FakeRowState("ready"), "b": FakeRowState("ready")},
        materialized={"a": job_a, "b": job_b},
        group_by_size=True,
    )

    controller.submit(["a", "b"], target="local")

    assert [[j.id for j in jobs] for jobs, _ in local.calls] == [[job_a.id], [job_b.id]]
    assert [extras for _, extras in local.calls] == [{"index": 0}, {"index": 1}]


def test_submit_resets_dismissed_job_before_dispatch(tmp_path: Path) -> None:
    """A dismissed terminal job is handed to the reset hook, then submitted.

    Its row projects "ready" (see ``effective_row_status``) even though the job
    itself is still FAILED, so without the pre-dispatch reset it would be
    dropped by the ``{DEFINED}`` gate in the submit backends.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    dismissed = make_job("dismissed", service=service, document=document, calc_type="fake_calc")
    service.submit_job(dismissed)
    service.fail_job(dismissed, "boom")
    service.dismiss_job(dismissed)

    prepared: list[list[Job]] = []

    def _prepare(jobs) -> bool:
        prepared.append(list(jobs))
        for job in jobs:
            service.reset_failed_job(job)
        return True

    local = RecordingBackend()
    context = make_context(
        local_backend=local,
        prepare_jobs_for_resubmission=_prepare,
    )
    controller = _FakeController(
        context,
        states={"a": FakeRowState("ready")},
        existing={"a": dismissed},
        # The row keeps its bound job: materialize reuses it, it does not mint
        # a second job for the same row.
        materialized={"a": dismissed},
    )

    controller.submit(["a"], target="local")

    assert [[job.id for job in jobs] for jobs in prepared] == [[dismissed.id]]
    assert [[job.id for job in jobs] for jobs, _ in local.calls] == [[dismissed.id]]
    assert dismissed.status is JobStatus.DEFINED
    assert dismissed.dismissed_at is None
    assert len(document.jobs) == 1
    # Materialize is re-run after the reset so the now-DEFINED job's input
    # directory is regenerated from the current configuration.
    assert controller.materialize_calls == [["a"], ["a"]]


def test_submit_aborts_when_resubmission_preparation_is_cancelled(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    dismissed = make_job("dismissed", service=service, document=document, calc_type="fake_calc")
    service.submit_job(dismissed)
    service.fail_job(dismissed, "boom")
    service.dismiss_job(dismissed)

    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(
        local_backend=local,
        infos=infos,
        prepare_jobs_for_resubmission=lambda _jobs: False,
    )
    controller = _FakeController(
        context,
        states={"a": FakeRowState("ready")},
        existing={"a": dismissed},
    )

    controller.submit(["a"], target="local")

    assert local.calls == []
    assert infos == []  # a cancelled cleanup prompt is a silent abort
    assert dismissed.status is JobStatus.FAILED
    assert dismissed.dismissed_at is not None
    assert controller.materialize_calls == [["a"]]  # no second pass after an abort


def test_submit_never_prepares_a_non_dismissed_failed_job(tmp_path: Path) -> None:
    """A mixed selection must not sweep a non-dismissed failure into the reset.

    ``submission_plan`` calls ``jobs_for_values`` over *every* selected key, so
    the plan holds the non-dismissed failed job too; only the dismissed one may
    be reset, and the other keeps being dropped by the backend status gate.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    dismissed = make_job("dismissed", service=service, document=document, calc_type="fake_calc")
    service.submit_job(dismissed)
    service.fail_job(dismissed, "boom")
    service.dismiss_job(dismissed)
    still_failed = make_job("still-failed", service=service, document=document, calc_type="fake_calc")
    service.submit_job(still_failed)
    service.fail_job(still_failed, "boom")

    prepared: list[list[Job]] = []
    local = RecordingBackend()
    context = make_context(
        local_backend=local,
        prepare_jobs_for_resubmission=lambda jobs: (prepared.append(list(jobs)), True)[1],
    )
    controller = _FakeController(
        context,
        states={"a": FakeRowState("ready"), "b": FakeRowState("failed")},
        existing={"a": dismissed, "b": still_failed},
    )

    controller.submit(["a", "b"], target="local")

    assert [[job.id for job in jobs] for jobs in prepared] == [[dismissed.id]]
    assert still_failed.status is JobStatus.FAILED
    assert still_failed.dismissed_at is None


def test_submit_reports_rows_left_out_of_a_mixed_selection(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    ready_job = make_job("ready", service=service, document=document, calc_type="fake_calc")
    running_job = make_job("running", service=service, document=document, calc_type="fake_calc")
    running_job.status = JobStatus.RUNNING
    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(local_backend=local, cluster_backend=cluster)
    controller = _FakeController(
        context,
        # "a" is ready, "b" has a running job, "c" is not ready and has no job.
        states={"a": FakeRowState("ready"), "b": FakeRowState("running"), "c": FakeRowState("completed")},
        existing={"b": running_job},
        materialized={"a": ready_job},
    )

    controller.submit(["a", "b", "c"], target="local")
    controller.submit(["a", "b", "c"], target="cluster")

    (_jobs, local_extras), (_cluster_jobs, cluster_extras) = local.calls[0], cluster.calls[0]
    assert local_extras["success_message"](1) == (
        "Submitted 1 local job(s); skipped 2 selected rows that were not ready"
    )
    assert cluster_extras["success_message"](1, "hpc") == (
        "Submitted 1 cluster job(s) on hpc; skipped 2 selected rows that were not ready"
    )


def test_submit_reports_a_single_skipped_row_in_the_singular(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    ready_job = make_job("ready", service=service, document=document, calc_type="fake_calc")
    local = RecordingBackend()
    controller = _FakeController(
        make_context(local_backend=local),
        states={"a": FakeRowState("ready"), "b": FakeRowState("completed")},
        materialized={"a": ready_job},
    )

    controller.submit(["a", "b"], target="local")

    assert local.calls[0][1]["success_message"](1) == (
        "Submitted 1 local job(s); skipped 1 selected row that was not ready"
    )


def test_submit_skipped_report_goes_on_the_last_group_only(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job_a = make_job("a", service=service, document=document, calc_type="fake_calc")
    job_b = make_job("b", service=service, document=document, calc_type="fake_calc")
    local = RecordingBackend()
    controller = _FakeController(
        make_context(local_backend=local),
        states={"a": FakeRowState("ready"), "b": FakeRowState("ready"), "c": FakeRowState("completed")},
        materialized={"a": job_a, "b": job_b},
        group_by_size=True,
    )

    controller.submit(["a", "b", "c"], target="local")

    first_extras, last_extras = (extras for _jobs, extras in local.calls)
    assert first_extras == {"index": 0}
    assert set(last_extras) == {"index", "success_message"}


def test_default_submission_groups_is_single_group() -> None:
    context = make_context()
    controller = _FakeController(context, states={})
    jobs: list[Job] = []
    assert controller.submission_groups(jobs) == [([], {})]


def test_default_is_ready_uses_workflow_row_predicate() -> None:
    context = make_context()
    controller = _FakeController(context, states={})
    # Base default accepts the wider workflow-row status set.
    assert controller.is_ready({"k": FakeRowState("not generated")}, "k") is True
    assert controller.is_ready({"k": FakeRowState("failed")}, "k") is False


class _StagedFakeController(_FakeController):
    stage = WorkflowStage.UEP
    active_key = "active_fake_definition_id"


def test_base_definition_lookups_read_the_context_document(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from mufinder.domain.models import CalculationDefinition

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    older = CalculationDefinition(
        workflow_stage=WorkflowStage.UEP, created_at=datetime(2025, 1, 1, tzinfo=UTC)
    )
    newer = CalculationDefinition(
        workflow_stage=WorkflowStage.UEP, created_at=datetime(2025, 1, 2, tzinfo=UTC)
    )
    for definition in (older, newer):
        document.calculations[definition.id] = definition
    job = service.add_job(document, label="row", workflow_stage=WorkflowStage.UEP, calc_type="fake_calc")
    older.parameters["job_ids_by_label"] = {"row": str(job.id)}
    controller = _StagedFakeController(make_context(document=document), states={})

    assert controller.definitions() == [newer, older]
    assert controller.active_definition() is newer
    document.ui_state["active_fake_definition_id"] = str(older.id)
    assert controller.active_definition() is older
    assert controller.job_for_key(older, "row") is job
    assert controller.job_for_key(older, "row", calc_type="other_calc") is None
    assert controller.job_for_key(newer, "row") is None


def test_base_definitions_keep_creation_order_when_the_clock_ties(tmp_path: Path, monkeypatch) -> None:
    from datetime import UTC, datetime

    from mufinder.domain import models
    from mufinder.domain.models import Structure

    class _FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=tz or UTC)

    # A frozen clock gives every definition registered in this loop the same reading.
    monkeypatch.setattr(models, "datetime", _FrozenClock)
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(label="host", lattice_vectors=[[3.0, 0, 0], [0, 3.0, 0], [0, 0, 3.0]], species=["Fe"], fractional_positions=[[0, 0, 0]])
    service.add_structure(document, host)
    created = [
        service.register_uep_definition(
            document, input_structure_id=host.id, parameters={"row": index}, output_directory=tmp_path / "uep"
        )
        for index in range(4)
    ]
    newest_first = [definition.id for definition in reversed(created)]

    controller = _StagedFakeController(make_context(document=document), states={})
    assert [definition.id for definition in controller.definitions()] == newest_first

    service.save_project(document)
    reloaded = service.load_project(document.project_root)
    controller = _StagedFakeController(make_context(document=reloaded), states={})
    assert [definition.id for definition in controller.definitions()] == newest_first


def test_base_definitions_sharing_a_timestamp_are_ordered_by_id_whatever_the_load_order(tmp_path: Path) -> None:
    from datetime import UTC, datetime
    from uuid import UUID

    from mufinder.domain.models import CalculationDefinition

    instant = datetime(2025, 1, 1, tzinfo=UTC)
    orders = []
    for ids in ((1, 2, 3), (3, 1, 2)):
        document = ProjectService().create_project(tmp_path / f"p{ids[0]}.mufinder", "P")
        for value in ids:
            definition = CalculationDefinition(id=UUID(int=value), workflow_stage=WorkflowStage.UEP, created_at=instant)
            document.calculations[definition.id] = definition
        controller = _StagedFakeController(make_context(document=document), states={})
        orders.append([definition.id.int for definition in controller.definitions()])

    assert orders == [[3, 2, 1], [3, 2, 1]]


def test_base_definition_lookups_without_a_document_or_stage(tmp_path: Path) -> None:
    staged = _StagedFakeController(make_context(document=None), states={})
    assert staged.definitions() == []
    assert staged.active_definition() is None
    assert staged.job_for_key(None, "row") is None

    document = ProjectService().create_project(tmp_path / "p.mufinder", "P")
    unstaged = _FakeController(make_context(document=document), states={})
    assert unstaged.definitions() == []
    assert unstaged.active_definition() is None


# --------------------------------------------------------------------------- #
# Base result dispatch: merge_controller_result
# --------------------------------------------------------------------------- #


def test_merge_controller_result_no_matching_controller_is_noop(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = make_job("j", service=service, document=document, calc_type="unhandled_calc")
    run = CalculationRun(job_id=job.id)
    controllers = [_FakeController(make_context(), states={})]

    payload = {"total_energy": -1.0}
    merged = merge_controller_result(
        controllers,
        job,
        completed_castep_result(),
        run,
        no_artifacts(),
        payload=payload,
        output_files=["seed.castep"],
        derived_structure_ids=[],
        warnings=["parse warning"],
    )

    assert merged == ({"total_energy": -1.0}, ["seed.castep"], [], ["parse warning"])
    # The caller's inputs were not mutated.
    assert payload == {"total_energy": -1.0}


def test_merge_controller_result_empty_result_is_noop(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = make_job("j", service=service, document=document, calc_type="fake_calc")
    run = CalculationRun(job_id=job.id)
    # Fake controller with no build_result payload returns the empty default.
    controllers = [_FakeController(make_context(), states={})]

    merged_payload, files, structures, warnings = merge_controller_result(
        controllers,
        job,
        completed_castep_result(),
        run,
        no_artifacts(),
        payload={"total_energy": -1.0},
        output_files=["seed.castep"],
        derived_structure_ids=[],
        warnings=["parse warning"],
    )

    assert merged_payload == {"total_energy": -1.0}
    assert files == ["seed.castep"]
    assert structures == []
    assert warnings == ["parse warning"]


def test_merge_controller_result_additively_merges_matching_controller(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = make_job("j", service=service, document=document, calc_type="fake_calc")
    run = CalculationRun(job_id=job.id)
    controller = _FakeController(
        make_context(),
        states={},
        build_result_payload={"custom_metric": 42, "total_energy": -2.0},
    )

    merged_payload, files, structures, warnings = merge_controller_result(
        [controller],
        job,
        completed_castep_result(),
        run,
        no_artifacts(),
        payload={"total_energy": -1.0, "completed": True},
        output_files=["seed.castep"],
        derived_structure_ids=[],
        warnings=["parse warning"],
    )

    # Controller payload updates over the base; files/warnings are appended.
    assert merged_payload == {"total_energy": -2.0, "completed": True, "custom_metric": 42}
    assert files == ["seed.castep", "extra.out"]
    assert structures == []
    assert warnings == ["parse warning", "controller warning"]


def test_merge_controller_result_selects_first_handling_controller(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = make_job("j", service=service, document=document, calc_type="fake_calc")
    run = CalculationRun(job_id=job.id)

    class _OtherController(_FakeController):
        calc_types = frozenset({"other_calc"})

    other = _OtherController(make_context(), states={}, build_result_payload={"x": 1})
    handler = _FakeController(make_context(), states={}, build_result_payload={"custom_metric": 7})

    merged_payload, *_ = merge_controller_result(
        [other, handler],
        job,
        completed_castep_result(),
        run,
        no_artifacts(),
        payload={},
        output_files=[],
        derived_structure_ids=[],
        warnings=[],
    )

    assert merged_payload == {"custom_metric": 7}


def test_validate_controller_artifacts_asks_the_owning_controller(tmp_path: Path) -> None:
    """Dispatch uses the same owner lookup as merge_controller_result: no owner
    and the base default both accept the completion, and an override's message
    is returned verbatim."""

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = make_job("j", service=service, document=document, calc_type="fake_calc")
    run = CalculationRun(job_id=job.id)

    class _RejectingController(_FakeController):
        def validate_artifacts(self, job, parsed, run, artifacts):
            return f"rejected {job.label}"

    default = _FakeController(make_context(), states={})
    rejecting = _RejectingController(make_context(), states={})
    unhandled = make_job("u", service=service, document=document, calc_type="unhandled_calc")

    assert validate_controller_artifacts([default], job, completed_castep_result(), run, no_artifacts()) is None
    assert (
        validate_controller_artifacts([rejecting], unhandled, completed_castep_result(), run, no_artifacts())
        is None
    )
    # The first handling controller wins, exactly as for result dispatch.
    assert (
        validate_controller_artifacts([rejecting, default], job, completed_castep_result(), run, no_artifacts())
        == "rejected j"
    )
    assert (
        validate_controller_artifacts([default, rejecting], job, completed_castep_result(), run, no_artifacts())
        is None
    )


def test_dedupe_jobs_keeps_the_first_of_each_job_in_order() -> None:
    first = make_job("a")
    second = make_job("b")

    assert dedupe_jobs([first, second, first, second]) == [first, second]
    assert dedupe_jobs([]) == []


def test_apply_row_parameters_to_form_state_updates_one_row_and_its_alias() -> None:
    from mufinder.app.workflow_controllers.base import (
        apply_row_parameters_to_form_state,
    )

    state = {
        "name_prefix": "pristine_relax",
        "rows": [
            {"label": "row1", "params": {"a": 1}, "parameter_values": {"a": 1}},
            {"label": "row2", "params": {"a": 2}},
            "not a row",
        ],
    }

    updated = apply_row_parameters_to_form_state(
        state,
        row_matches=lambda row: row.get("label") == "row1",
        parameter_values={"a": 9},
        added_advanced=["extra"],
        parameter_key="params",
    )

    assert updated is not None
    assert updated["rows"][0]["params"] == {"a": 9}
    # Both spellings move together: the deserializer prefers "parameter_values",
    # so a stale alias would win.
    assert updated["rows"][0]["parameter_values"] == {"a": 9}
    assert updated["rows"][0]["added_advanced"] == ["extra"]
    assert updated["rows"][1] == {"label": "row2", "params": {"a": 2}}
    assert updated["rows"][2] == "not a row"
    # The input is untouched.
    assert state["rows"][0]["params"] == {"a": 1}
    assert "added_advanced" not in state["rows"][0]

    # No matching row, and malformed states, are reported as "not mirrored".
    assert (
        apply_row_parameters_to_form_state(
            state,
            row_matches=lambda _row: False,
            parameter_values={"a": 9},
            added_advanced=None,
            parameter_key="params",
        )
        is None
    )
    for malformed in ([], {"rows": "nope"}):
        assert (
            apply_row_parameters_to_form_state(
                malformed,
                row_matches=lambda _row: True,
                parameter_values={"a": 9},
                added_advanced=None,
                parameter_key="params",
            )
            is None
        )
