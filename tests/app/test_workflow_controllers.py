"""Tests for the per-workflow controller base class and cutoff migration.

The base :class:`WorkflowJobController` is exercised through a *fake* controller
so that the template-method ``submit`` flow, the no-ready path, the grouped
submission override, and the ``merge_controller_result`` result-dispatch hook are
proven independently of any concrete workflow (the six future migrations depend
on these). :class:`ConvergenceCutoffController` is then tested against a real
``ProjectService`` document with fake submit backends, pinning that it reaches
the right backend with the right jobs and writes the ``job_ids_by_cutoff``
mapping identically to the former ``MainWindow`` behavior.

Both use narrow fakes for the :class:`WorkflowControllerContext` callables — no
``MainWindow`` is constructed — which is exactly the decoupling the controller
layer is meant to provide.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import pytest

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    merge_controller_result,
)
from mufinder.app.workflow_controllers.convergence_cutoff import (
    ConvergenceCutoffController,
    ConvergenceCutoffDeps,
)
from mufinder.app.workflow_controllers.convergence_kpoint import (
    ConvergenceKpointController,
    ConvergenceKpointDeps,
)
from mufinder.app.workflow_controllers.muon_relaxation import (
    MuonRelaxationController,
    MuonRelaxationDeps,
)
from mufinder.app.workflow_controllers.pristine_relaxation import (
    PristineRelaxationController,
    PristineRelaxationDeps,
)
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.app.workflow_controllers.uep import (
    UepController,
    UepDeps,
)
from mufinder.app.workflow_submission import WorkflowSubmissionCoordinator
from tests.support.jobs import make_job


# --------------------------------------------------------------------------- #
# Shared fakes
# --------------------------------------------------------------------------- #


def _no_artifacts() -> ParsedJobArtifacts:
    """An empty artifacts carrier for controllers that consume no secondary output."""

    return ParsedJobArtifacts()


class _RecordingBackend:
    """Records ``(jobs, kwargs)`` for each submit call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[Job], dict[str, object]]] = []

    def __call__(self, jobs: list[Job], **kwargs: object) -> None:
        self.calls.append((list(jobs), dict(kwargs)))


# The relaxation controllers serialize/deserialize their definition configuration
# through required deps. These mirror the shape ``MainWindow`` stores (the real
# pair is exercised end to end in ``tests/app/test_main_window_actions.py``);
# structured CASTEP value restoration is not reproduced here.


def _serialize_pristine_config(config) -> dict[str, object]:
    return {
        "name_prefix": config.name_prefix,
        "rows": [
            {
                "label": row.label,
                "fix_all_cell": bool(row.fix_all_cell),
                "supercell_matrix": (
                    [list(line) for line in row.supercell_matrix]
                    if row.supercell_matrix is not None
                    else None
                ),
                "parameter_values": dict(row.parameter_values),
                "added_advanced": list(row.added_advanced),
            }
            for row in config.rows
        ],
    }


def _deserialize_pristine_config(payload: object):
    from mufinder.app.relaxation_types import (
        PristineRelaxationConfig,
        PristineRelaxationRow,
    )

    if not isinstance(payload, dict):
        return None
    rows = [
        PristineRelaxationRow(
            label=str(entry.get("label", "")),
            fix_all_cell=bool(entry.get("fix_all_cell", False)),
            parameter_values=dict(
                entry.get("parameter_values", entry.get("params", {}))
            ),
            added_advanced=tuple(str(name) for name in entry.get("added_advanced", ())),
        )
        for entry in payload.get("rows", [])
        if isinstance(entry, dict)
    ]
    return PristineRelaxationConfig(
        name_prefix=str(payload.get("name_prefix", "pristine_relax")),
        rows=tuple(rows),
    )


def _serialize_muon_config(config) -> dict[str, object]:
    return {
        "rows": [
            {
                "site_id": str(row.site_id),
                "label": row.label,
                "initial_fractional_position": list(row.initial_fractional_position),
                "charge_state": row.charge_state,
                "supercell_matrix": (
                    [list(line) for line in row.supercell_matrix]
                    if row.supercell_matrix is not None
                    else None
                ),
                "parameter_values": dict(row.parameter_values),
                "added_advanced": list(row.added_advanced),
            }
            for row in config.rows
        ]
    }


def _deserialize_muon_config(payload: object):
    from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow

    if not isinstance(payload, dict):
        return None
    rows = []
    for entry in payload.get("rows", []):
        if not isinstance(entry, dict):
            continue
        position = entry.get("initial_fractional_position", (0.0, 0.0, 0.0))
        rows.append(
            MuonRelaxationRow(
                site_id=UUID(str(entry.get("site_id"))),
                label=str(entry.get("label", "")),
                initial_fractional_position=(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                ),
                charge_state=str(entry.get("charge_state", "positive_muon")),
                parameter_values=dict(entry.get("parameter_values", {})),
                added_advanced=tuple(
                    str(name) for name in entry.get("added_advanced", ())
                ),
            )
        )
    return MuonRelaxationConfig(rows=tuple(rows))


def _make_context(
    *,
    document: ProjectDocument | None = None,
    project_service: ProjectService | None = None,
    local_backend: _RecordingBackend | None = None,
    cluster_backend: _RecordingBackend | None = None,
    infos: list[tuple[str, str]] | None = None,
    prepare_jobs_for_resubmission: Callable[[object], bool] | None = None,
    definition_output_directory: Callable[[object], object] = lambda _d: None,
    job_output_directory: Callable[[object], object] = lambda _j: None,
    job_seed_name: Callable[[object], str] = lambda j: j.label,
    run_for_job: Callable[[object], object] = lambda _r: None,
    total_energy_for_job: Callable[[object], object] = lambda _j: None,
    max_force_for_job: Callable[[object], object] = lambda _j: None,
) -> WorkflowControllerContext:
    service = project_service or ProjectService()
    return WorkflowControllerContext(
        document=lambda: document,
        project_service=service,
        job_execution=None,  # type: ignore[arg-type]  # unused by these tests
        submission_coordinator=WorkflowSubmissionCoordinator(),
        submit_jobs_locally=local_backend or _RecordingBackend(),
        submit_jobs_on_cluster=cluster_backend or _RecordingBackend(),
        prepare_jobs_for_resubmission=prepare_jobs_for_resubmission or (lambda _jobs: True),
        show_info=(lambda title, text: infos.append((title, text))) if infos is not None else (lambda *_: None),
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=lambda **_: None,
        status_message=lambda _message: None,
        definition_output_directory=definition_output_directory,
        job_output_directory=job_output_directory,
        job_seed_name=job_seed_name,
        run_for_job=run_for_job,
        total_energy_for_job=total_energy_for_job,
        max_force_for_job=max_force_for_job,
    )


class _State:
    """Minimal state object with a ``status`` and optional ``job_id``."""

    def __init__(self, status: str, job_id: object | None = None) -> None:
        self.status = status
        self.job_id = job_id


class _FakeController(WorkflowJobController):
    """Fake controller exercising the base template method + result hook."""

    calc_types = frozenset({"fake_calc"})
    no_ready_message = "No ready fake rows."

    def __init__(
        self,
        context: WorkflowControllerContext,
        *,
        states: dict[str, _State],
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

    def build_states(self) -> dict[str, _State]:
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

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(local_backend=local, cluster_backend=cluster)
    controller = _FakeController(
        context,
        states={"a": _State("ready"), "b": _State("ready")},
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

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(local_backend=local, cluster_backend=cluster)
    controller = _FakeController(
        context,
        states={"b": _State("ready")},
        materialized={"b": fresh},
    )

    controller.submit(["b"], target="cluster")

    assert local.calls == []
    assert [j.id for j in cluster.calls[0][0]] == [fresh.id]


def test_submit_shows_info_when_no_ready_jobs(tmp_path: Path) -> None:
    service = ProjectService()
    service.create_project(tmp_path / "p.mufinder", "P")
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(local_backend=local, infos=infos)
    controller = _FakeController(
        context,
        # No 'ready' rows and no jobs → empty plan.
        states={"a": _State("failed")},
    )

    controller.submit(["a"], target="local")

    assert local.calls == []
    assert infos == [("No ready jobs", "No ready fake rows.")]


def test_submit_pre_submit_check_short_circuits(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    fresh = make_job("fresh", service=service, document=document, calc_type="fake_calc")
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(local_backend=local, infos=infos)
    controller = _FakeController(
        context,
        states={"b": _State("ready")},
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
    local = _RecordingBackend()
    context = _make_context(local_backend=local)
    controller = _FakeController(
        context,
        states={"a": _State("ready"), "b": _State("ready")},
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

    local = _RecordingBackend()
    context = _make_context(
        local_backend=local,
        prepare_jobs_for_resubmission=_prepare,
    )
    controller = _FakeController(
        context,
        states={"a": _State("ready")},
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

    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(
        local_backend=local,
        infos=infos,
        prepare_jobs_for_resubmission=lambda _jobs: False,
    )
    controller = _FakeController(
        context,
        states={"a": _State("ready")},
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
    local = _RecordingBackend()
    context = _make_context(
        local_backend=local,
        prepare_jobs_for_resubmission=lambda jobs: (prepared.append(list(jobs)), True)[1],
    )
    controller = _FakeController(
        context,
        states={"a": _State("ready"), "b": _State("failed")},
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
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(local_backend=local, cluster_backend=cluster)
    controller = _FakeController(
        context,
        # "a" is ready, "b" has a running job, "c" is not ready and has no job.
        states={"a": _State("ready"), "b": _State("running"), "c": _State("completed")},
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
    local = _RecordingBackend()
    controller = _FakeController(
        _make_context(local_backend=local),
        states={"a": _State("ready"), "b": _State("completed")},
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
    local = _RecordingBackend()
    controller = _FakeController(
        _make_context(local_backend=local),
        states={"a": _State("ready"), "b": _State("ready"), "c": _State("completed")},
        materialized={"a": job_a, "b": job_b},
        group_by_size=True,
    )

    controller.submit(["a", "b", "c"], target="local")

    first_extras, last_extras = (extras for _jobs, extras in local.calls)
    assert first_extras == {"index": 0}
    assert set(last_extras) == {"index", "success_message"}


def test_default_submission_groups_is_single_group() -> None:
    context = _make_context()
    controller = _FakeController(context, states={})
    jobs: list[Job] = []
    assert controller.submission_groups(jobs) == [([], {})]


def test_default_is_ready_uses_workflow_row_predicate() -> None:
    context = _make_context()
    controller = _FakeController(context, states={})
    # Base default accepts the wider workflow-row status set.
    assert controller.is_ready({"k": _State("not generated")}, "k") is True
    assert controller.is_ready({"k": _State("failed")}, "k") is False


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
    controller = _StagedFakeController(_make_context(document=document), states={})

    assert controller.definitions() == [newer, older]
    assert controller.active_definition() is newer
    document.ui_state["active_fake_definition_id"] = str(older.id)
    assert controller.active_definition() is older
    assert controller.job_for_key(older, "row") is job
    assert controller.job_for_key(older, "row", calc_type="other_calc") is None
    assert controller.job_for_key(newer, "row") is None


def test_base_definition_lookups_without_a_document_or_stage(tmp_path: Path) -> None:
    staged = _StagedFakeController(_make_context(document=None), states={})
    assert staged.definitions() == []
    assert staged.active_definition() is None
    assert staged.job_for_key(None, "row") is None

    document = ProjectService().create_project(tmp_path / "p.mufinder", "P")
    unstaged = _FakeController(_make_context(document=document), states={})
    assert unstaged.definitions() == []
    assert unstaged.active_definition() is None


# --------------------------------------------------------------------------- #
# Base result dispatch: merge_controller_result
# --------------------------------------------------------------------------- #


def _parsed() -> CastepResult:
    return CastepResult(completed=True, total_energy=-1.0, warnings=("parse warning",))


def test_merge_controller_result_no_matching_controller_is_noop(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = make_job("j", service=service, document=document, calc_type="unhandled_calc")
    run = CalculationRun(job_id=job.id)
    controllers = [_FakeController(_make_context(), states={})]

    payload = {"total_energy": -1.0}
    merged = merge_controller_result(
        controllers,
        job,
        _parsed(),
        run,
        _no_artifacts(),
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
    controllers = [_FakeController(_make_context(), states={})]

    merged_payload, files, structures, warnings = merge_controller_result(
        controllers,
        job,
        _parsed(),
        run,
        _no_artifacts(),
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
        _make_context(),
        states={},
        build_result_payload={"custom_metric": 42, "total_energy": -2.0},
    )

    merged_payload, files, structures, warnings = merge_controller_result(
        [controller],
        job,
        _parsed(),
        run,
        _no_artifacts(),
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

    other = _OtherController(_make_context(), states={}, build_result_payload={"x": 1})
    handler = _FakeController(_make_context(), states={}, build_result_payload={"custom_metric": 7})

    merged_payload, *_ = merge_controller_result(
        [other, handler],
        job,
        _parsed(),
        run,
        _no_artifacts(),
        payload={},
        output_files=[],
        derived_structure_ids=[],
        warnings=[],
    )

    assert merged_payload == {"custom_metric": 7}


# --------------------------------------------------------------------------- #
# ConvergenceCutoffController
# --------------------------------------------------------------------------- #


class _FakeConvergenceView:
    """Minimal ConvergenceWidget stand-in for controller tests."""

    def __init__(self, form_state: dict | None = None) -> None:
        self._form_state = form_state or {}
        self.runtime_states: list[dict] = []

    def collect_state(self) -> dict:
        return self._form_state

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


def _stub_deps() -> ConvergenceCutoffDeps:
    """Deps whose helpers all report 'nothing found' — for tests that never
    reach the definition-backed paths."""

    return ConvergenceCutoffDeps(
        deserialize_convergence_config=lambda _payload: None,
        sync_equivalent_convergence_cutoff_mappings=lambda _definition: None,
        definition_for_convergence_cutoff=lambda _cutoff: (None, None),
        cutoff_key=lambda value: f"{float(value):.12g}",
        max_stress_for_job=lambda _job_id: None,
    )


def _cutoff_context_and_deps(
    service: ProjectService,
    document: ProjectDocument,
    *,
    local_backend: _RecordingBackend,
    cluster_backend: _RecordingBackend,
) -> tuple[WorkflowControllerContext, ConvergenceCutoffDeps]:
    """Shared context + cutoff deps wired with fake helpers.

    The cutoff-specific helpers below mirror the (unchanged) MainWindow versions
    closely enough to drive materialize/build_states without a MainWindow.
    """

    active_definition = next(iter(document.calculations.values()), None)

    def deserialize(payload):
        # The test seeds a ConvergenceConfig directly on the definition.
        return document.ui_state.get("_test_config")

    def definition_for_cutoff(cutoff):
        return active_definition, document.ui_state.get("_test_config")

    context = WorkflowControllerContext(
        document=lambda: document,
        project_service=service,
        job_execution=None,  # type: ignore[arg-type]
        submission_coordinator=WorkflowSubmissionCoordinator(),
        submit_jobs_locally=local_backend,
        submit_jobs_on_cluster=cluster_backend,
        prepare_jobs_for_resubmission=lambda _jobs: True,
        show_info=lambda *_: None,
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=lambda **_: None,
        status_message=lambda _m: None,
        definition_output_directory=lambda d: document.project_root / "files" / "convergence_cutoff",
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
        run_for_job=lambda _job_id: None,
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
    )
    deps = ConvergenceCutoffDeps(
        deserialize_convergence_config=deserialize,
        sync_equivalent_convergence_cutoff_mappings=lambda _d: None,
        definition_for_convergence_cutoff=definition_for_cutoff,
        cutoff_key=lambda value: f"{float(value):.12g}",
        max_stress_for_job=lambda _job_id: None,
    )
    return context, deps


def _seed_convergence_document(service: ProjectService, document: ProjectDocument):
    from mufinder.app.convergence_types import ConvergenceConfig
    from mufinder.domain.models import Structure

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    definition = service.register_convergence_cutoff_definition(
        document,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=document.project_root / "files" / "convergence_cutoff",
    )
    config = ConvergenceConfig(
        name_prefix="conv",
        cutoff_energies=(300.0,),
        shared_params={"task": "SINGLEPOINT"},
    )
    document.ui_state["_test_config"] = config
    return definition, config


def test_convergence_controller_materialize_writes_job_ids_by_cutoff(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    definition, _config = _seed_convergence_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceCutoffController(context, deps, view)

    jobs = controller.materialize([300.0])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.label == "conv_ecut_300"
    assert job.calc_type == "castep_convergence_cutoff"
    mapping = definition.parameters["job_ids_by_cutoff"]
    assert mapping[f"{300.0:.12g}"] == str(job.id)
    # Rendered inputs were written.
    root = document.project_root / "files" / "convergence_cutoff" / "conv_ecut_300"
    assert (root / "conv_ecut_300.cell").exists()
    assert (root / "conv_ecut_300.param").exists()


def test_convergence_controller_materialize_rewrites_inputs_for_defined_job(
    tmp_path: Path,
) -> None:
    """A defined job's inputs are regenerated on re-materialize.

    This is what makes the post-reset ``materialize`` pass in
    ``WorkflowJobController.submit`` effective for convergence rows: a job reset
    back to ``defined`` after dashboard dismissal must not resubmit against the
    stale ``.cell``/``.param`` files its failed run left behind.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=_RecordingBackend(), cluster_backend=_RecordingBackend()
    )
    controller = ConvergenceCutoffController(context, deps, view)

    (job,) = controller.materialize([300.0])
    param_path = (
        document.project_root
        / "files"
        / "convergence_cutoff"
        / "conv_ecut_300"
        / "conv_ecut_300.param"
    )
    param_path.write_text("STALE")

    again = controller.materialize([300.0])

    assert [j.id for j in again] == [job.id]
    assert len(document.jobs) == 1
    assert param_path.read_text() != "STALE"


def test_convergence_controller_materialize_keeps_non_defined_job_inputs(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=_RecordingBackend(), cluster_backend=_RecordingBackend()
    )
    controller = ConvergenceCutoffController(context, deps, view)

    (job,) = controller.materialize([300.0])
    job.status = JobStatus.RUNNING
    param_path = (
        document.project_root
        / "files"
        / "convergence_cutoff"
        / "conv_ecut_300"
        / "conv_ecut_300.param"
    )
    param_path.write_text("STALE")

    again = controller.materialize([300.0])

    assert [j.id for j in again] == [job.id]
    assert param_path.read_text() == "STALE"


def test_convergence_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceCutoffController(context, deps, view)

    controller.submit([300.0], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert [j.label for j in submitted] == ["conv_ecut_300"]


def test_convergence_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    _seed_convergence_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    view = _FakeConvergenceView(form_state={"name_prefix": "conv", "cutoff_energies": [300.0]})
    context, deps = _cutoff_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceCutoffController(context, deps, view)

    controller.submit([300.0], target="cluster")

    assert local.calls == []
    assert [j.label for j in cluster.calls[0][0]] == ["conv_ecut_300"]


def test_convergence_controller_uses_shared_ready_predicate() -> None:
    controller = ConvergenceCutoffController(_make_context(), _stub_deps(), _FakeConvergenceView())
    # Cutoff readiness now uses the shared workflow-row predicate, so the same
    # states that are submittable for every other workflow ("ready", "defined",
    # "not generated") are submittable here too; active states are not.
    from mufinder.app.convergence_types import SweepPointRunState

    for ready_status in ("ready", "defined", "not generated"):
        states = {300.0: SweepPointRunState(status=ready_status)}
        assert controller.is_ready(states, 300.0) is True
    states = {300.0: SweepPointRunState(status="running")}
    assert controller.is_ready(states, 300.0) is False


def test_convergence_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeConvergenceView()
    controller = ConvergenceCutoffController(_make_context(document=None), _stub_deps(), view)

    controller.refresh_view()

    assert view.runtime_states == [{}]


def test_convergence_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "conv.mufinder", "Conv")
    controller = ConvergenceCutoffController(
        _make_context(document=document), _stub_deps(), _FakeConvergenceView()
    )
    handled = make_job("j", service=service, document=document, calc_type="castep_convergence_cutoff")
    other = make_job("k", service=service, document=document, calc_type="castep_uep_source")
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False


# --------------------------------------------------------------------------- #
# ConvergenceKpointController
# --------------------------------------------------------------------------- #


class _FakeKpointView:
    """Minimal KPointConvergenceWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.runtime_states: list[dict] = []

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


def _grid_key(grid: tuple[int, int, int]) -> str:
    return f"{int(grid[0])}x{int(grid[1])}x{int(grid[2])}"


def _stub_kpoint_deps() -> ConvergenceKpointDeps:
    """Deps whose helpers all report 'nothing found' — for tests that never
    reach the definition-backed paths."""

    return ConvergenceKpointDeps(
        deserialize_convergence_config=lambda _payload: None,
        sync_equivalent_convergence_kpoint_mappings=lambda _definition: None,
        grid_key=_grid_key,
        max_stress_for_job=lambda _job_id: None,
    )


def _kpoint_context_and_deps(
    service: ProjectService,
    document: ProjectDocument,
    *,
    local_backend: _RecordingBackend,
    cluster_backend: _RecordingBackend,
) -> tuple[WorkflowControllerContext, ConvergenceKpointDeps]:
    """Shared context + kpoint deps wired with fake helpers.

    The kpoint-specific helpers below mirror the (unchanged) MainWindow versions
    closely enough to drive materialize/build_states without a MainWindow.
    """

    next(iter(document.calculations.values()), None)

    def deserialize(payload):
        return document.ui_state.get("_test_config")

    context = WorkflowControllerContext(
        document=lambda: document,
        project_service=service,
        job_execution=None,  # type: ignore[arg-type]
        submission_coordinator=WorkflowSubmissionCoordinator(),
        submit_jobs_locally=local_backend,
        submit_jobs_on_cluster=cluster_backend,
        prepare_jobs_for_resubmission=lambda _jobs: True,
        show_info=lambda *_: None,
        request_dashboard_flush=lambda **_: None,
        persist_ui_state=lambda **_: None,
        status_message=lambda _m: None,
        definition_output_directory=lambda d: document.project_root / "files" / "convergence_kpoint",
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
        run_for_job=lambda _job_id: None,
        total_energy_for_job=lambda _job_id: None,
        max_force_for_job=lambda _job_id: None,
    )
    deps = ConvergenceKpointDeps(
        deserialize_convergence_config=deserialize,
        sync_equivalent_convergence_kpoint_mappings=lambda _d: None,
        grid_key=_grid_key,
        max_stress_for_job=lambda _job_id: None,
    )
    return context, deps


def _seed_kpoint_document(service: ProjectService, document: ProjectDocument):
    from mufinder.app.convergence_types import ConvergenceConfig
    from mufinder.domain.models import Structure

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = ConvergenceConfig(
        name_prefix="kconv",
        sweep_parameter="kpoint_mp_grid",
        sweep_values=((2, 2, 2),),
        shared_params={"task": "SINGLEPOINT"},
    )
    definition = service.register_convergence_kpoint_definition(
        document,
        input_structure_id=host.id,
        parameters={
            "name_prefix": "kconv",
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2]],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=document.project_root / "files" / "convergence_kpoint",
    )
    document.ui_state["_test_config"] = config
    # Persist the form so _kpoint_form_snapshot agrees with the active definition.
    document.ui_state["kpoint_convergence_form"] = {
        "name_prefix": "kconv",
        "kpoint_grids": [[2, 2, 2]],
    }
    return definition, config


def test_kpoint_controller_materialize_writes_job_ids_by_grid(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    definition, _config = _seed_kpoint_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    view = _FakeKpointView()
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceKpointController(context, deps, view)

    jobs = controller.materialize([(2, 2, 2)])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.calc_type == "castep_convergence_kpoint"
    mapping = definition.parameters["job_ids_by_grid"]
    assert mapping[_grid_key((2, 2, 2))] == str(job.id)
    # Rendered inputs were written.
    root = document.project_root / "files" / "convergence_kpoint" / job.label
    assert (root / f"{job.label}.cell").exists()
    assert (root / f"{job.label}.param").exists()


def test_kpoint_controller_materialize_rewrites_inputs_for_defined_job(
    tmp_path: Path,
) -> None:
    """Same invariant as the cutoff variant: defined jobs get fresh inputs."""

    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=_RecordingBackend(), cluster_backend=_RecordingBackend()
    )
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())

    (job,) = controller.materialize([(2, 2, 2)])
    param_path = (
        document.project_root / "files" / "convergence_kpoint" / job.label / f"{job.label}.param"
    )
    param_path.write_text("STALE")

    again = controller.materialize([(2, 2, 2)])

    assert [j.id for j in again] == [job.id]
    assert len(document.jobs) == 1
    assert param_path.read_text() != "STALE"


def test_kpoint_controller_materialize_keeps_non_defined_job_inputs(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=_RecordingBackend(), cluster_backend=_RecordingBackend()
    )
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())

    (job,) = controller.materialize([(2, 2, 2)])
    job.status = JobStatus.RUNNING
    param_path = (
        document.project_root / "files" / "convergence_kpoint" / job.label / f"{job.label}.param"
    )
    param_path.write_text("STALE")

    again = controller.materialize([(2, 2, 2)])

    assert [j.id for j in again] == [job.id]
    assert param_path.read_text() == "STALE"


def test_kpoint_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    view = _FakeKpointView()
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceKpointController(context, deps, view)

    controller.submit([(2, 2, 2)], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_convergence_kpoint"


def test_kpoint_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    view = _FakeKpointView()
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=local, cluster_backend=cluster
    )
    controller = ConvergenceKpointController(context, deps, view)

    controller.submit([(2, 2, 2)], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == "castep_convergence_kpoint"


def test_kpoint_controller_uses_shared_ready_predicate() -> None:
    controller = ConvergenceKpointController(_make_context(), _stub_kpoint_deps(), _FakeKpointView())
    from mufinder.app.convergence_types import SweepPointRunState

    for ready_status in ("ready", "defined", "not generated"):
        states = {(2, 2, 2): SweepPointRunState(status=ready_status)}
        assert controller.is_ready(states, (2, 2, 2)) is True
    states = {(2, 2, 2): SweepPointRunState(status="running")}
    assert controller.is_ready(states, (2, 2, 2)) is False


def test_kpoint_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeKpointView()
    controller = ConvergenceKpointController(
        _make_context(document=None), _stub_kpoint_deps(), view
    )

    controller.refresh_view()

    assert view.runtime_states == [{}]


def test_kpoint_controller_build_states_reflects_form_ready_grids(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    _seed_kpoint_document(service, document)
    context, deps = _kpoint_context_and_deps(
        service, document, local_backend=_RecordingBackend(), cluster_backend=_RecordingBackend()
    )
    controller = ConvergenceKpointController(context, deps, _FakeKpointView())

    states = controller.build_states()

    assert (2, 2, 2) in states
    assert states[(2, 2, 2)].status == "ready"


def test_kpoint_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "kconv.mufinder", "KConv")
    controller = ConvergenceKpointController(
        _make_context(document=document), _stub_kpoint_deps(), _FakeKpointView()
    )
    handled = make_job("j", service=service, document=document, calc_type="castep_convergence_kpoint")
    other = make_job("k", service=service, document=document, calc_type="castep_convergence_cutoff")
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False


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


# --------------------------------------------------------------------------- #
# PristineRelaxationController
# --------------------------------------------------------------------------- #
#
# This is the first migration whose ``build_result`` does real work: it parses
# the relaxed ``-out.cell``, contributes the ``lattice_a/b/c`` payload, and
# creates the derived relaxed Structure on the document. The tests below drive
# it with narrow fakes (no MainWindow) and a REAL ``-out.cell`` fixture.


class _FakeRelaxationView:
    """Minimal PristineRelaxationWidget stand-in for controller tests.

    ``collect_state``/``set_form_state`` store the form state verbatim: the real
    widget also normalizes it (stripping implicit project defaults), and that
    round trip is pinned against the real widget in
    ``tests/app/test_main_window_actions.py``.
    """

    def __init__(self, form_state: dict | None = None) -> None:
        self.runtime_states: list[dict] = []
        self.form_state: dict = form_state if form_state is not None else {"rows": []}

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))

    def collect_state(self) -> dict:
        return copy.deepcopy(self.form_state)

    def set_form_state(self, state: dict) -> None:
        self.form_state = copy.deepcopy(state)


_OUT_CELL_TEXT = """
%BLOCK LATTICE_CART
ang
5.1 0.0 0.0
0.0 5.2 0.0
0.0 0.0 5.3
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
"""


def _stub_pristine_deps(**overrides) -> PristineRelaxationDeps:
    base = dict(
        deserialize_pristine_relaxation_config=lambda _payload: None,
        serialize_pristine_relaxation_config=_serialize_pristine_config,
        sync_equivalent_pristine_relaxation_mappings=lambda _definition: None,
        pristine_relaxation_form_config=lambda: None,
        pristine_relaxation_row_signature=lambda _row: (),
        expand_pristine_relaxation_jobs=lambda _definition, _config: None,
    )
    base.update(overrides)
    return PristineRelaxationDeps(**base)


def _pristine_job(service: ProjectService, document: ProjectDocument, label: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )


def test_pristine_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    controller = PristineRelaxationController(
        _make_context(document=document), _stub_pristine_deps(), _FakeRelaxationView()
    )
    handled = _pristine_job(service, document, "pristine_relax_row1")
    muon = service.add_job(
        document,
        label="site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muonium_relaxation",
    )
    # The still-inline muon-relaxation branch must NOT be claimed by this
    # controller — it is dispatched by the inline code, not build_result.
    assert controller.handles_job(handled) is True
    assert controller.handles_job(muon) is False


def test_pristine_controller_uses_workflow_row_ready_predicate() -> None:
    controller = PristineRelaxationController(
        _make_context(), _stub_pristine_deps(), _FakeRelaxationView()
    )
    # Pristine relaxation uses the base-default (wider) workflow-row predicate:
    # "not generated" / "defined" rows are ready, only "failed" is excluded.
    assert controller.is_ready({"r": _State("not generated")}, "r") is True
    assert controller.is_ready({"r": _State("ready")}, "r") is True
    assert controller.is_ready({"r": _State("failed")}, "r") is False


def test_pristine_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeRelaxationView()
    controller = PristineRelaxationController(
        _make_context(document=None), _stub_pristine_deps(), view
    )
    controller.refresh_view()
    assert view.runtime_states == [{}]


def test_pristine_controller_build_result_parses_lattice_and_creates_structure(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)

    service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "pristine_relax", "rows": []},
        output_directory=document.project_root / "files" / "pristine_relaxation",
    )

    job = _pristine_job(service, document, "pristine_relax_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # A REAL -out.cell fixture on disk, fed through the controller's parse helper.
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(_OUT_CELL_TEXT, encoding="utf-8")

    deps = _stub_pristine_deps(
    )
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = PristineRelaxationController(context, deps, _FakeRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())
    artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    result = controller.build_result(job, parsed, run, artifacts)

    # Payload gains the first three lattice parameters from the relaxed cell.
    assert result.payload["lattice_a"] == pytest.approx(5.1)
    assert result.payload["lattice_b"] == pytest.approx(5.2)
    assert result.payload["lattice_c"] == pytest.approx(5.3)

    # No output_files contributed — the call site already passes them generically.
    assert result.output_files == []

    # Exactly one derived structure created on the document, with the right
    # origin and parentage.
    assert len(result.derived_structure_ids) == 1
    derived_id = result.derived_structure_ids[0]
    derived = document.structures[derived_id]
    assert derived.origin == "pristine_relaxation"
    assert derived.parent_structure_id == host.id
    assert derived.properties["source_job_id"] == str(job.id)
    assert derived.properties["source_workflow_stage"] == WorkflowStage.PRISTINE_RELAXATION.value
    assert derived.label == "pristine_relax_row1 relaxed"


def test_pristine_controller_build_result_missing_out_cell_is_empty(tmp_path: Path) -> None:
    """build_result no longer parses or raises on a missing ``-out.cell``: it now
    consumes the pre-parsed carrier and returns an empty ControllerResult when
    ``artifacts.relaxed_cell`` is ``None``. In production the missing-file failure
    is detected by the inline handler guard from the carrier (the ``cell_missing``
    flag surfaced on :class:`ParsedJobArtifacts`), not by build_result — so
    build_result is never reached with a missing file, and its own contract when
    handed an empty carrier is simply to contribute nothing."""

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _pristine_job(service, document, "pristine_relax_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    deps = _stub_pristine_deps()
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: empty_dir,
    )
    controller = PristineRelaxationController(context, deps, _FakeRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())
    result = controller.build_result(job, parsed, run, ParsedJobArtifacts())
    assert result.payload == {}
    assert result.derived_structure_ids == []
    # No derived structure was created.
    assert document.structures == {} or all(
        s.origin != "pristine_relaxation" for s in document.structures.values()
    )


_OUT_CELL_TEXT_CONTINUED = """
%BLOCK LATTICE_CART
ang
5.4 0.0 0.0
0.0 5.5 0.0
0.0 0.0 5.6
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.01 0.02 0.03
%ENDBLOCK POSITIONS_FRAC
"""


def test_pristine_controller_recompletion_updates_derived_structure_in_place(
    tmp_path: Path,
) -> None:
    """A continued pristine job completes twice. The second completion must
    refresh the SAME derived structure rather than mint a duplicate: a
    comparison table holds ``dft_structure_ids``, so a replacement structure
    would strand that reference on a superseded geometry."""

    from mufinder.domain.models import ComparisonTable, Result, Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "pristine_relax", "rows": []},
        output_directory=document.project_root / "files" / "pristine_relaxation",
    )

    job = _pristine_job(service, document, "pristine_relax_row1")
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    out_cell = output_dir / f"{job.label}-out.cell"

    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = PristineRelaxationController(
        context,
        _stub_pristine_deps(),
        _FakeRelaxationView(),
    )
    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())

    def complete(run_index: int, cell_text: str):
        run = CalculationRun(job_id=job.id, run_index=run_index)
        document.runs[run.id] = run
        out_cell.write_text(cell_text, encoding="utf-8")
        artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
        controller_result = controller.build_result(job, parsed, run, artifacts)
        result = Result(
            calculation_run_id=run.id,
            result_type=job.calc_type,
            derived_structure_ids=list(controller_result.derived_structure_ids),
        )
        document.results[result.id] = result
        run.result_id = result.id
        return run, result

    _first_run, first_result = complete(1, _OUT_CELL_TEXT)
    first_id = first_result.derived_structure_ids[0]
    structure_count = len(document.structures)

    # A comparison table built from the first completion — the reference the
    # update-in-place behaviour exists to protect.
    table = ComparisonTable(
        label="Comparison 1",
        experimental_structure_id=host.id,
        dft_structure_ids=[first_id],
        dft_labels=[job.label],
    )
    service.add_comparison_table(document, table)

    _second_run, second_result = complete(2, _OUT_CELL_TEXT_CONTINUED)

    # Same structure id, no duplicate structure, both Results converge on it.
    assert second_result.derived_structure_ids == [first_id]
    assert len(document.structures) == structure_count
    assert first_result.derived_structure_ids == second_result.derived_structure_ids

    # The structure now carries the SECOND run's geometry.
    derived = document.structures[first_id]
    assert derived.lattice_vectors == [[5.4, 0.0, 0.0], [0.0, 5.5, 0.0], [0.0, 0.0, 5.6]]
    assert derived.fractional_positions == [pytest.approx([0.01, 0.02, 0.03])]
    assert len(derived.species) == len(derived.fractional_positions)
    assert derived.label == "pristine_relax_row1 relaxed"
    assert derived.properties["source_job_id"] == str(job.id)
    assert derived.parent_structure_id == host.id

    # Every reference — the comparison table's and both Results' — still resolves.
    assert table.dft_structure_ids == [first_id]
    assert service.validate_document(document) == []


def test_pristine_controller_recompletion_refreshes_result_payload(
    tmp_path: Path,
) -> None:
    """The re-completion's payload comes from the new ``-out.cell``, so the
    workflow row reports the continued run's lattice, not the first run's."""

    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)

    job = _pristine_job(service, document, "pristine_relax_row1")
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    out_cell = output_dir / f"{job.label}-out.cell"
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = PristineRelaxationController(
        context, _stub_pristine_deps(), _FakeRelaxationView()
    )
    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())

    out_cell.write_text(_OUT_CELL_TEXT, encoding="utf-8")
    run_one = CalculationRun(job_id=job.id, run_index=1)
    document.runs[run_one.id] = run_one
    controller.build_result(
        job, parsed, run_one, collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    )

    out_cell.write_text(_OUT_CELL_TEXT_CONTINUED, encoding="utf-8")
    run_two = CalculationRun(job_id=job.id, run_index=2)
    document.runs[run_two.id] = run_two
    second = controller.build_result(
        job, parsed, run_two, collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    )

    assert second.payload["lattice_a"] == pytest.approx(5.4)
    assert second.payload["lattice_b"] == pytest.approx(5.5)
    assert second.payload["lattice_c"] == pytest.approx(5.6)


def test_find_job_derived_structure_falls_back_to_earlier_run_results(
    tmp_path: Path,
) -> None:
    """A derived structure written before ``properties["source_job_id"]`` existed
    is still found — through the ``derived_structure_ids`` of a Result reachable
    from one of the job's own runs — so it is refreshed rather than duplicated."""

    from mufinder.app.workflow_controllers.base import find_job_derived_structure
    from mufinder.domain.models import Result, Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _pristine_job(service, document, "pristine_relax_row1")

    legacy = Structure(
        label="pristine_relax_row1 relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[5.1, 0.0, 0.0], [0.0, 5.2, 0.0], [0.0, 0.0, 5.3]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, legacy)
    assert "source_job_id" not in legacy.properties

    run = CalculationRun(job_id=job.id, run_index=1)
    document.runs[run.id] = run
    result = Result(calculation_run_id=run.id, derived_structure_ids=[legacy.id])
    document.results[result.id] = result
    run.result_id = result.id

    found = find_job_derived_structure(document, job, origin="pristine_relaxation")
    assert found is legacy

    # A structure belonging to a DIFFERENT job is never reached by the fallback.
    other_job = _pristine_job(service, document, "pristine_relax_row2")
    assert find_job_derived_structure(document, other_job, origin="pristine_relaxation") is None


def test_find_job_derived_structure_prefers_the_first_of_a_duplicate_pair(
    tmp_path: Path,
) -> None:
    """A job that completed twice before update-in-place existed left a duplicate
    pair. The lookup resolves to the FIRST in document order — the one any
    comparison table built in the meantime already references."""

    from mufinder.app.workflow_controllers.base import find_job_derived_structure
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _pristine_job(service, document, "pristine_relax_row1")

    first = Structure(
        label="pristine_relax_row1 relaxed",
        origin="pristine_relaxation",
        properties={"source_job_id": str(job.id)},
    )
    second = Structure(
        label="pristine_relax_row1 relaxed",
        origin="pristine_relaxation",
        properties={"source_job_id": str(job.id)},
    )
    service.add_structure(document, first)
    service.add_structure(document, second)

    assert find_job_derived_structure(document, job, origin="pristine_relaxation") is first
    # Origin is part of the key: a muon-origin structure is never returned for a
    # pristine lookup even when it names the same job.
    assert find_job_derived_structure(document, job, origin="muon_relaxation") is None


def test_pristine_controller_materialize_writes_job_ids_by_label(tmp_path: Path) -> None:
    from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
    from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(PristineRelaxationRow(label="row1"),),
    )
    output_root = document.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "pristine_relax",
            "rows": [{"label": "row1", "parameter_values": {}}],
        },
        output_directory=output_root,
    )

    def expand(defn, cfg):
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=cfg,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )

    deps = _stub_pristine_deps(
        deserialize_pristine_relaxation_config=lambda _payload: config,
        expand_pristine_relaxation_jobs=expand,
    )
    context = _make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = PristineRelaxationController(context, deps, _FakeRelaxationView())

    jobs = controller.materialize(["row1"])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.calc_type == "castep_pristine_relaxation"
    mapping = definition.parameters["job_ids_by_label"]
    assert mapping["row1"] == str(job.id)
    # Rendered inputs were written under the output root.
    job_dir = output_root / job.label
    assert (job_dir / f"{job.label}.cell").exists()
    assert (job_dir / f"{job.label}.param").exists()


def test_pristine_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
    from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(PristineRelaxationRow(label="row1"),),
    )
    output_root = document.project_root / "files" / "pristine_relaxation"
    service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "pristine_relax", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    def expand(defn, cfg):
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=cfg,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )

    deps = _stub_pristine_deps(
        deserialize_pristine_relaxation_config=lambda _payload: config,
        pristine_relaxation_form_config=lambda: config,
        pristine_relaxation_row_signature=lambda row: (row.label,),
        expand_pristine_relaxation_jobs=expand,
    )

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = PristineRelaxationController(context, deps, _FakeRelaxationView())

    controller.submit(["row1"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_pristine_relaxation"


def _pristine_parameter_edit_fixture(service: ProjectService, tmp_path: Path):
    """Definition with two rows, ``row1`` bound to a job, and a matching form.

    This is the Continue-relaxation precondition: a submitted row whose
    ``job_ids_by_label`` entry must survive a parameter edit.
    """

    from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
    from mufinder.app.relaxation_types import (
        PristineRelaxationConfig,
        PristineRelaxationRow,
    )
    from mufinder.domain.models import Structure

    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(label="row1", parameter_values={"geom_max_iter": 120}),
            PristineRelaxationRow(label="row2", parameter_values={"geom_max_iter": 400}),
        ),
    )
    output_root = document.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters=_serialize_pristine_config(config),
        output_directory=output_root,
    )
    job = _pristine_job(service, document, "pristine_relax_row1")
    definition.parameters["job_ids_by_label"] = {"row1": str(job.id)}

    view = _FakeRelaxationView(
        {
            "name_prefix": "pristine_relax",
            "rows": [
                {
                    "label": "row1",
                    "fix_all_cell": False,
                    "supercell_matrix": None,
                    "params": {"geom_max_iter": 120},
                    "added_advanced": [],
                },
                {
                    "label": "row2",
                    "fix_all_cell": False,
                    "supercell_matrix": None,
                    "params": {"geom_max_iter": 400},
                    "added_advanced": [],
                },
            ],
        }
    )

    def expand(_defn, cfg):
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=cfg,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )

    deps = _stub_pristine_deps(
        deserialize_pristine_relaxation_config=_deserialize_pristine_config,
        pristine_relaxation_form_config=lambda: _deserialize_pristine_config(
            view.collect_state()
        ),
        pristine_relaxation_row_signature=lambda row: (
            tuple(sorted(row.parameter_values.items())),
            tuple(row.added_advanced),
        ),
        expand_pristine_relaxation_jobs=expand,
    )
    return document, definition, job, deps, view, output_root


def _pristine_edit_controller(service, document, deps, view, output_root, persists):
    import dataclasses

    context = dataclasses.replace(
        _make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _defn: output_root,
        ),
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    return PristineRelaxationController(context, deps, view)


def test_pristine_controller_update_row_parameters_preserves_row_job_binding(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    definitions_before = set(document.calculations)
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is True

    # No new definition was registered: the binding lives on this very object.
    assert set(document.calculations) == definitions_before
    assert document.calculations[definition.id] is definition
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    assert controller.jobs_for_keys(["row1"]) == [job]

    config = _deserialize_pristine_config(definition.parameters["configuration"])
    rows = {row.label: row for row in config.rows}
    assert rows["row1"].parameter_values == {"geom_max_iter": 500}
    assert rows["row2"].parameter_values == {"geom_max_iter": 400}

    # The form state was mirrored so the row signatures still match.
    form_rows = {row["label"]: row for row in view.form_state["rows"]}
    assert form_rows["row1"]["params"] == {"geom_max_iter": 500}
    assert form_rows["row2"]["params"] == {"geom_max_iter": 400}
    assert document.ui_state["pristine_relaxation_form"] == view.form_state
    assert persists == [
        {
            "failure_message": (
                "Pristine relaxation parameters were updated, but autosave failed."
            )
        }
    ]


def test_pristine_controller_update_row_parameters_rejects_unknown_row(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = (
        _pristine_parameter_edit_fixture(service, tmp_path)
    )
    before = copy.deepcopy(definition.parameters["configuration"])
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.update_row_parameters("row9", {"geom_max_iter": 500}) is False

    assert definition.parameters["configuration"] == before
    assert persists == []


def test_pristine_controller_update_row_parameters_edits_a_later_unbound_row(
    tmp_path: Path,
) -> None:
    """Editing the second (jobless) row leaves the first row — and its job
    binding — alone in both the definition and the form."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.update_row_parameters("row2", {"geom_max_iter": 700}) is True

    config = _deserialize_pristine_config(definition.parameters["configuration"])
    rows = {row.label: row for row in config.rows}
    assert rows["row1"].parameter_values == {"geom_max_iter": 120}
    assert rows["row2"].parameter_values == {"geom_max_iter": 700}
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    form_rows = {row["label"]: row for row in view.form_state["rows"]}
    assert form_rows["row1"]["params"] == {"geom_max_iter": 120}
    assert form_rows["row2"]["params"] == {"geom_max_iter": 700}


def test_pristine_controller_update_row_parameters_without_definition_or_config(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = (
        _pristine_parameter_edit_fixture(service, tmp_path)
    )
    persists: list[dict] = []

    no_definition = _stub_pristine_deps()
    controller = _pristine_edit_controller(
        service, document, no_definition, view, output_root, persists
    )
    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is False

    unparseable = _stub_pristine_deps(
        deserialize_pristine_relaxation_config=lambda _payload: None,
    )
    controller = _pristine_edit_controller(
        service, document, unparseable, view, output_root, persists
    )
    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is False
    assert persists == []


def test_pristine_controller_update_row_parameters_without_the_row_in_the_form(
    tmp_path: Path,
) -> None:
    """With no form row to mirror into, the caller's own values are stored: the
    row is not projected from the form either, so no signature can desync."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    view.form_state = {"name_prefix": "pristine_relax", "rows": []}
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is True

    config = _deserialize_pristine_config(definition.parameters["configuration"])
    rows = {row.label: row for row in config.rows}
    assert rows["row1"].parameter_values == {"geom_max_iter": 500}
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    assert view.form_state["rows"] == []
    assert "pristine_relaxation_form" not in document.ui_state


def test_pristine_controller_materialize_after_edit_rewrites_same_directory(
    tmp_path: Path,
) -> None:
    """The Continue cycle: render, finish, edit parameters, return to DEFINED,
    re-materialize. The row keeps its job and its directory; only the rendered
    ``.param`` changes."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.materialize(["row1"]) == [job]
    param_path = output_root / job.label / f"{job.label}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")

    job.status = JobStatus.COMPLETED
    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is True
    # WP1's prepare_job_continuation returns the job to DEFINED (with its
    # remote_directory preserved) before the directory is rewritten.
    job.status = JobStatus.DEFINED

    assert controller.materialize(["row1"]) == [job]

    assert list(document.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered


def test_pristine_controller_build_states_keeps_job_after_parameter_edit(
    tmp_path: Path,
) -> None:
    """Finding 5: pristine relaxation has no active-status escape hatch, so a
    form/definition signature mismatch would drop the row to a jobless "ready"."""

    service = ProjectService()
    document, _definition, job, deps, view, output_root = (
        _pristine_parameter_edit_fixture(service, tmp_path)
    )
    job.status = JobStatus.RUNNING
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = controller.build_states()
    assert before["row1"].job_id == job.id

    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is True

    after = controller.build_states()
    assert after["row1"].job_id == job.id
    assert after["row1"].status == JobStatus.RUNNING.value


def test_pristine_controller_row_is_continuable_reflects_raw_job_status(
    tmp_path: Path,
) -> None:
    """Finding 8: gating reads the job's real status, never displayed text —
    a dismissed FAILED job (which *displays* "ready") is still continuable."""

    from mufinder.domain.models import utc_now

    service = ProjectService()
    document, _definition, job, deps, view, output_root = (
        _pristine_parameter_edit_fixture(service, tmp_path)
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )

    job.status = JobStatus.RUNNING
    assert controller.row_is_continuable("row1") is False

    for status in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.STOPPED,
        JobStatus.CANCELLED,
    ):
        job.status = status
        assert controller.row_is_continuable("row1") is True

    job.status = JobStatus.FAILED
    job.dismissed_at = utc_now()
    assert controller.row_is_continuable("row1") is True

    # row2 has no bound job; an unknown key resolves to no job either.
    assert controller.row_is_continuable("row2") is False
    assert controller.row_is_continuable("unknown") is False


def test_pristine_controller_render_row_param_text_previews_without_persisting(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = (
        _pristine_parameter_edit_fixture(service, tmp_path)
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = copy.deepcopy(definition.parameters["configuration"])

    text = controller.render_row_param_text("row1", {"geom_max_iter": 777}, ())

    assert "GEOM_MAX_ITER : 777" in text
    # Nothing was persisted: this is a pure preview.
    assert definition.parameters["configuration"] == before

    assert controller.render_row_param_text("unknown", {}, ()) == ""


def test_pristine_controller_continue_job_returns_to_defined_and_rewrites_directory(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import utc_now

    service = ProjectService()
    document, definition, job, deps, view, output_root = (
        _pristine_parameter_edit_fixture(service, tmp_path)
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )
    controller.materialize(["row1"])
    param_path = output_root / job.label / f"{job.label}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    job.execution_target = "myhpc"
    job.remote_directory = "/scratch/user/row1"
    job.remote_job_id = "12345"
    job.error_message = "boom"
    job.dismissed_at = utc_now()

    continued = controller.continue_job("row1", {"geom_max_iter": 500})

    assert continued is job
    assert job.status == JobStatus.DEFINED
    # remote_directory is deliberately preserved; everything else the previous
    # run stamped is cleared.
    assert job.remote_directory == "/scratch/user/row1"
    assert job.remote_job_id is None
    assert job.error_message is None
    assert job.dismissed_at is None

    # Same directory rewritten, no duplicate job minted.
    assert list(document.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered

    continuation = next(
        record for record in document.provenance if record.action == "job_continuation"
    )
    assert continuation.job_id == job.id
    assert continuation.parameters["remote_directory"] == "/scratch/user/row1"
    assert continuation.parameters["preserved_remote_directory"] is True


def test_pristine_controller_continue_job_rejects_non_continuable_job_without_mutating(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = (
        _pristine_parameter_edit_fixture(service, tmp_path)
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = copy.deepcopy(definition.parameters["configuration"])

    job.status = JobStatus.RUNNING
    assert controller.continue_job("row1", {"geom_max_iter": 500}) is None

    # Untouched: the row's parameters were never mutated, since the job wasn't
    # continuable in the first place.
    assert job.status == JobStatus.RUNNING
    assert definition.parameters["configuration"] == before
    assert not any(record.action == "job_continuation" for record in document.provenance)


def test_pristine_controller_continue_job_unknown_row_returns_none(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, _definition, _job, deps, view, output_root = (
        _pristine_parameter_edit_fixture(service, tmp_path)
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.continue_job("unknown", {"geom_max_iter": 500}) is None


def test_pristine_controller_continue_job_returns_none_when_materialize_fails(
    tmp_path: Path,
) -> None:
    """Audit finding 1: materialize() can swallow an expansion error and return
    ``[]`` without raising. If continue_job ignored that, the job would be left
    DEFINED with a preserved remote_directory while the on-disk .param still
    held the PREVIOUS run's parameters -- and the caller (which only checks
    "is the return value truthy") would submit that stale directory as if the
    new parameters had been written."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(
        service, document, deps, view, output_root, persists
    )
    controller.materialize(["row1"])  # write the real initial directory contents
    param_path = output_root / job.label / f"{job.label}.param"
    original_rendered = param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    job.remote_directory = "/scratch/user/row1"

    # Force the continuation's own materialize call to fail exactly the way
    # the real one does on a caught ValueError/StructureBridgeError: return []
    # without raising.
    controller.materialize = lambda keys: []

    result = controller.continue_job("row1", {"geom_max_iter": 500})

    assert result is None
    # The row's stored parameters were still mutated (update_row_parameters ran
    # before the failed materialize) and the job was still moved to DEFINED
    # (prepare_job_continuation ran) -- it stays recoverable -- but the ON-DISK
    # .param is untouched: this is exactly why the caller must not submit.
    config = _deserialize_pristine_config(definition.parameters["configuration"])
    assert {row.label: row.parameter_values for row in config.rows}["row1"] == {
        "geom_max_iter": 500
    }
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == "/scratch/user/row1"
    assert param_path.read_text(encoding="utf-8") == original_rendered
    # No provenance milestone either: the journal must not claim a continuation
    # that was never rendered and (because continue_job returned None) is never
    # submitted. record_job_continuation therefore runs only after materialize.
    assert not any(record.action == "job_continuation" for record in document.provenance)


def test_pristine_controller_continue_job_leaves_row_untouched_when_document_disappears(
    tmp_path: Path,
) -> None:
    """Audit finding 5: the document lookup must run BEFORE
    update_row_parameters. ``jobs_for_keys`` reads the document once to resolve
    the job (call #1, succeeds); this context then returns ``None`` on the
    NEXT call (call #2), simulating the document disappearing between menu-open
    and dialog-accept (e.g. the project was closed/switched). If
    update_row_parameters ran first, the row's stored parameters would be
    mutated before continue_job discovered there was no document to record
    provenance against."""

    import dataclasses

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    job.status = JobStatus.FAILED
    before_config = copy.deepcopy(definition.parameters["configuration"])

    calls = {"n": 0}

    def flaky_document():
        calls["n"] += 1
        return document if calls["n"] == 1 else None

    persists: list[dict] = []
    context = dataclasses.replace(
        _make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _defn: output_root,
        ),
        document=flaky_document,
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    controller = PristineRelaxationController(context, deps, view)

    result = controller.continue_job("row1", {"geom_max_iter": 500})

    assert result is None
    assert calls["n"] >= 2
    assert definition.parameters["configuration"] == before_config
    assert job.status == JobStatus.FAILED


# --------------------------------------------------------------------------- #
# MuonRelaxationController
# --------------------------------------------------------------------------- #
#
# The first migration exercising the two base hooks beyond their defaults:
# ``pre_submit_check`` (staged-site/host validation runs BEFORE any submission
# plan is built) and ``submission_groups`` (a mixed job list is split into
# per-charge-state groups, each dispatched separately through the base template
# loop). ``build_result`` carries the richest result logic so far — muon
# position payload (supercell + folded host coordinates), derived structure,
# and the sanctioned source-site mutation — driven below with a REAL
# ``-out.cell`` fixture containing an ``H:mu`` muon.


class _FakeMuonRelaxationView:
    """Minimal MuonRelaxationWidget stand-in for controller tests.

    ``collect_state``/``set_form_state`` store the form state verbatim; see
    :class:`_FakeRelaxationView`.
    """

    def __init__(self, form_state: dict | None = None) -> None:
        self.runtime_states: list[dict] = []
        self.form_state: dict = form_state if form_state is not None else {"rows": []}

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))

    def collect_state(self) -> dict:
        return copy.deepcopy(self.form_state)

    def set_form_state(self, state: dict) -> None:
        self.form_state = copy.deepcopy(state)


_MUON_OUT_CELL_TEXT = """
%BLOCK LATTICE_CART
ang
8.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
H:mu 0.3 0.4 0.5
%ENDBLOCK POSITIONS_FRAC
"""


def _real_muon_position_from_cell(cell):
    """Extract the H:mu position exactly as MainWindow._muon_fractional_position_from_cell does."""

    species = getattr(cell, "species", ())
    positions = getattr(cell, "fractional_positions", ())
    for atom_species, position in reversed(list(zip(species, positions, strict=False))):
        if str(atom_species).strip().casefold() != "h:mu":
            continue
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            return None
        return (float(position[0]), float(position[1]), float(position[2]))
    return None


def _real_coerce_fractional_triplet(value):
    """Coerce exactly as MainWindow._coerce_fractional_triplet does."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _stub_muon_deps(**overrides) -> MuonRelaxationDeps:
    from mufinder.app.relaxation_adapter import muon_charge_state_prefix

    base = dict(
        deserialize_muon_relaxation_config=lambda _payload: None,
        serialize_muon_relaxation_config=_serialize_muon_config,
        sync_equivalent_muon_relaxation_mappings=lambda _definition: None,
        sync_active_muon_relaxation_dashboard_jobs=lambda _definition: None,
        muon_relaxation_form_config=lambda: None,
        muon_relaxation_row_signature=lambda row: (row.label, row.charge_state),
        muon_relaxation_row_key=(
            lambda row: f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"
        ),
        handle_site_host_mismatch=lambda _mismatch: False,
        muon_relaxation_definition_and_row_for_job=lambda _job: (None, None),
        muon_fractional_position_from_cell=_real_muon_position_from_cell,
        coerce_fractional_triplet=_real_coerce_fractional_triplet,
    )
    base.update(overrides)
    return MuonRelaxationDeps(**base)


def _muon_job(
    service: ProjectService, document: ProjectDocument, label: str, calc_type: str
) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=calc_type,
    )


def _muon_host_document(service: ProjectService, tmp_path: Path):
    from mufinder.domain.models import Structure

    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    return document, host


def test_muon_controller_job_for_label_checks_charge_state_calc_types(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters={"rows": []},
        output_directory=document.project_root / "files" / "muon_relaxation",
    )
    positive = _muon_job(service, document, "Site 1", "castep_positivemuon_relaxation")
    muonium = _muon_job(service, document, "Site 2", "castep_muonium_relaxation")
    pristine = _muon_job(service, document, "Site 3", "castep_pristine_relaxation")
    definition.parameters["job_ids_by_label"] = {
        "positivemuon:Site 1": str(positive.id),
        # A charge-qualified key bound to the wrong charge state's job.
        "muonium:Site 1": str(positive.id),
        # A bare legacy key accepts any muon-relaxation calc type.
        "Site 2": str(muonium.id),
        "Site 3": str(pristine.id),
    }
    controller = MuonRelaxationController(
        _make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )

    assert controller.active_definition() is definition
    assert controller.job_for_label(definition, "Site 1", "positive_muon") is positive
    assert controller.job_for_label(definition, "Site 1", "muonium") is None
    assert controller.job_for_label(definition, "Site 2", "positive_muon") is muonium
    assert controller.job_for_label(definition, "Site 2") is muonium
    assert controller.job_for_label(definition, "Site 3") is None


def test_muon_controller_handles_both_charge_state_calc_types(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = _muon_host_document(service, tmp_path)
    controller = MuonRelaxationController(
        _make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    positive = _muon_job(service, document, "s1", "castep_positivemuon_relaxation")
    muonium = _muon_job(service, document, "s2", "castep_muonium_relaxation")
    pristine = service.add_job(
        document,
        label="p",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    assert controller.handles_job(positive) is True
    assert controller.handles_job(muonium) is True
    # Pristine relaxation is owned by its own controller, not this one.
    assert controller.handles_job(pristine) is False


def test_muon_controller_ready_predicate_matches_bare_and_qualified_labels() -> None:
    controller = MuonRelaxationController(
        _make_context(), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    states = {
        "muonium:Site 1": _State("ready"),
        "positivemuon:Site 2": _State("failed"),
    }
    # Charge-qualified keys match their own state only.
    assert controller.is_ready(states, "muonium:Site 1") is True
    assert controller.is_ready(states, "positivemuon:Site 2") is False
    # A bare label matches any charge-qualified key ending in ":{label}".
    assert controller.is_ready(states, "Site 1") is True
    assert controller.is_ready(states, "Site 2") is False
    # "defined" and "not generated" also count as ready (unlike the sweep predicate).
    assert controller.is_ready({"s": _State("defined")}, "s") is True
    assert controller.is_ready({"s": _State("not generated")}, "s") is True


def test_muon_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeMuonRelaxationView()
    controller = MuonRelaxationController(
        _make_context(document=None), _stub_muon_deps(), view
    )
    controller.refresh_view()
    assert view.runtime_states == [{}]


def test_muon_controller_pre_submit_check_blocks_submission(tmp_path: Path) -> None:
    """An unresolved staged-site/host mismatch silently refuses the submission:
    no submission plan is built, no jobs are materialized or dispatched, and no
    'no ready jobs' dialog appears."""

    from mufinder.domain.models import CalculationDefinition, MuonSite, Structure

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    other = Structure(
        label="Other",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, other)
    # The staged site was generated for a different host than the active one.
    site = MuonSite(
        label="Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=other.id,
    )
    document.sites[site.id] = site
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    mismatches: list[object] = []
    form_config_calls: list[bool] = []
    deps = _stub_muon_deps(
        handle_site_host_mismatch=lambda mismatch: (mismatches.append(mismatch), False)[1],
        muon_relaxation_form_config=lambda: (form_config_calls.append(True), None)[1],
    )
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1"], target="local")

    assert len(mismatches) == 1  # the host check ran and surfaced the mismatch
    assert local.calls == []  # nothing dispatched
    assert infos == []  # the refusal is silent (no "no ready jobs" dialog)
    assert form_config_calls == []  # no submission plan was even built
    assert document.jobs == {}  # nothing materialized


def test_muon_controller_pre_submit_check_passes_when_sites_match_host(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    mismatches: list[object] = []
    deps = _stub_muon_deps(
        handle_site_host_mismatch=lambda mismatch: (mismatches.append(mismatch), False)[1],
    )
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1"], target="local")

    # No mismatch dialog; the check passed and the plan ran (and found no
    # configured rows, hence the standard no-ready dialog).
    assert mismatches == []
    assert infos == [("No ready jobs", MuonRelaxationController.no_ready_message)]
    assert local.calls == []


def test_muon_controller_submission_groups_split_by_charge_state(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = _muon_host_document(service, tmp_path)
    controller = MuonRelaxationController(
        _make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    pos_a = _muon_job(service, document, "a", "castep_positivemuon_relaxation")
    mu_b = _muon_job(service, document, "b", "castep_muonium_relaxation")
    pos_c = _muon_job(service, document, "c", "castep_positivemuon_relaxation")

    # Mixed list splits into positive-muon group first, then muonium, no extra
    # kwargs. ``submission_groups`` no longer dedupes by id — the submission
    # coordinator's plan already does that before this hook runs — so a duplicate
    # job id passed straight to this hook is preserved within its charge-state
    # group rather than collapsed.
    groups = controller.submission_groups([pos_a, mu_b, pos_c, pos_a])
    assert groups == [([pos_a, pos_c, pos_a], {}), ([mu_b], {})]

    # A single-charge-state list yields one group (no empty groups).
    assert controller.submission_groups([mu_b]) == [([mu_b], {})]


def _muon_two_row_submission_fixture(service: ProjectService, tmp_path: Path):
    """Document + config with one positive-muon and one muonium row, wired for
    real materialization through expand_muon_relaxation_jobs."""


    from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow
    from mufinder.domain.models import MuonSite

    document, host = _muon_host_document(service, tmp_path)
    site1 = MuonSite(
        label="Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    site2 = MuonSite(
        label="Site 2",
        status="staged",
        fractional_position=[0.4, 0.1, 0.2],
        source_structure_id=host.id,
    )
    document.sites[site1.id] = site1
    document.sites[site2.id] = site2

    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=site1.id,
                label="Site 1",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="positive_muon",
            ),
            MuonRelaxationRow(
                site_id=site2.id,
                label="Site 2",
                initial_fractional_position=(0.4, 0.1, 0.2),
                charge_state="muonium",
            ),
        )
    )
    output_root = document.project_root / "files" / "muon_relaxation"
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters={"configuration": {}},
        output_directory=output_root,
    )

    deps = _stub_muon_deps(
        deserialize_muon_relaxation_config=lambda _payload: config,
        muon_relaxation_form_config=lambda: config,
    )
    return document, definition, deps, output_root


def test_muon_controller_materialize_writes_charge_prefixed_mappings_and_persists(
    tmp_path: Path,
) -> None:
    import dataclasses

    service = ProjectService()
    document, definition, deps, output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    context = dataclasses.replace(
        _make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _defn: output_root,
        ),
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    jobs = controller.materialize(["muonium:Site 2"])

    assert [job.calc_type for job in jobs] == ["castep_muonium_relaxation"]
    assert [job.label for job in jobs] == ["Site 2"]
    mapping = definition.parameters["job_ids_by_label"]
    seed_mapping = definition.parameters["job_seed_names_by_label"]
    assert mapping == {"muonium:Site 2": str(jobs[0].id)}
    assert set(seed_mapping) == {"muonium:Site 2"}
    # Rendered inputs were written under the charge-state root.
    seed = seed_mapping["muonium:Site 2"]
    job_dir = output_root / "muonium" / seed
    assert (job_dir / f"{seed}.cell").exists()
    assert (job_dir / f"{seed}.param").exists()
    # The document was persisted with the muon-specific failure message.
    assert persists == [
        {"failure_message": "Muon relaxation jobs were generated, but autosave failed."}
    ]


def test_muon_controller_submit_local_splits_groups_across_backend_calls(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, deps, output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1", "muonium:Site 2"], target="local")

    assert cluster.calls == []
    # One backend call per charge-state group: positive muon first, then muonium.
    assert len(local.calls) == 2
    first_group, first_extras = local.calls[0]
    second_group, second_extras = local.calls[1]
    assert first_extras == {} and second_extras == {}
    assert [job.calc_type for job in first_group] == ["castep_positivemuon_relaxation"]
    assert [job.label for job in first_group] == ["Site 1"]
    assert [job.calc_type for job in second_group] == ["castep_muonium_relaxation"]
    assert [job.label for job in second_group] == ["Site 2"]
    # Both jobs were recorded under charge-prefixed mapping keys.
    mapping = definition.parameters["job_ids_by_label"]
    assert set(mapping) == {"positivemuon:Site 1", "muonium:Site 2"}


def _muon_parameter_edit_fixture(service: ProjectService, tmp_path: Path):
    """One site staged in BOTH charge states under the same label.

    The two rows share a label, so only the charge-qualified key distinguishes
    them; this is the fixture that proves an edit reaches exactly one row.
    """

    from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow
    from mufinder.domain.models import MuonSite

    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Site 1",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site

    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=site.id,
                label="Site 1",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="positive_muon",
                parameter_values={"geom_max_iter": 120},
            ),
            MuonRelaxationRow(
                site_id=site.id,
                label="Site 1",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="muonium",
                parameter_values={"geom_max_iter": 400},
            ),
        )
    )
    output_root = document.project_root / "files" / "muon_relaxation"
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters=_serialize_muon_config(config),
        output_directory=output_root,
    )
    job = _muon_job(service, document, "Site 1", "castep_positivemuon_relaxation")
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}

    view = _FakeMuonRelaxationView(_serialize_muon_config(config))

    deps = _stub_muon_deps(
        deserialize_muon_relaxation_config=_deserialize_muon_config,
        muon_relaxation_form_config=lambda: _deserialize_muon_config(
            view.collect_state()
        ),
        muon_relaxation_row_signature=lambda row: (
            tuple(sorted(row.parameter_values.items())),
            tuple(row.added_advanced),
        ),
    )
    return document, definition, job, deps, view, output_root


def _muon_edit_controller(service, document, deps, view, output_root, persists):
    import dataclasses

    context = dataclasses.replace(
        _make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _defn: output_root,
        ),
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    return MuonRelaxationController(context, deps, view)


def test_muon_controller_update_row_parameters_touches_only_the_charge_state_row(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    definitions_before = set(document.calculations)
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is True
    )

    assert set(document.calculations) == definitions_before
    assert document.calculations[definition.id] is definition
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    assert controller.jobs_for_keys(["positivemuon:Site 1"]) == [job]

    config = _deserialize_muon_config(definition.parameters["configuration"])
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values == {"geom_max_iter": 500}
    # The muonium row shares the label and is left untouched.
    assert by_charge["muonium"].parameter_values == {"geom_max_iter": 400}

    form_by_charge = {row["charge_state"]: row for row in view.form_state["rows"]}
    assert form_by_charge["positive_muon"]["parameter_values"] == {"geom_max_iter": 500}
    assert form_by_charge["muonium"]["parameter_values"] == {"geom_max_iter": 400}
    assert document.ui_state["muon_relaxation_form"] == view.form_state
    assert persists == [
        {
            "failure_message": (
                "Muon relaxation parameters were updated, but autosave failed."
            )
        }
    ]


def test_muon_controller_update_row_parameters_rejects_bare_label(
    tmp_path: Path,
) -> None:
    """A bare label is ambiguous when two charge states share it, so it is
    refused rather than applied to whichever row matches first."""

    service = ProjectService()
    document, definition, _job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    before = copy.deepcopy(definition.parameters["configuration"])
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.update_row_parameters("Site 1", {"geom_max_iter": 500}) is False

    assert definition.parameters["configuration"] == before
    assert persists == []


def test_muon_controller_update_row_parameters_edits_the_second_charge_state_row(
    tmp_path: Path,
) -> None:
    """The mirror reaches the muonium row (second in both the definition and the
    form) without disturbing the bound positive-muon row."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert (
        controller.update_row_parameters("muonium:Site 1", {"geom_max_iter": 700})
        is True
    )

    config = _deserialize_muon_config(definition.parameters["configuration"])
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values == {"geom_max_iter": 120}
    assert by_charge["muonium"].parameter_values == {"geom_max_iter": 700}
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    form_by_charge = {row["charge_state"]: row for row in view.form_state["rows"]}
    assert form_by_charge["positive_muon"]["parameter_values"] == {"geom_max_iter": 120}
    assert form_by_charge["muonium"]["parameter_values"] == {"geom_max_iter": 700}


def test_muon_controller_update_row_parameters_without_definition_or_config(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []

    no_definition = _stub_muon_deps()
    controller = _muon_edit_controller(
        service, document, no_definition, view, output_root, persists
    )
    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is False
    )

    unparseable = _stub_muon_deps(
        deserialize_muon_relaxation_config=lambda _payload: None,
    )
    controller = _muon_edit_controller(
        service, document, unparseable, view, output_root, persists
    )
    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is False
    )
    assert persists == []


def test_muon_controller_update_row_parameters_without_the_row_in_the_form(
    tmp_path: Path,
) -> None:
    """With no form row to mirror into, the caller's own values are stored; see
    the pristine equivalent."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    view.form_state = {"rows": []}
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is True
    )

    config = _deserialize_muon_config(definition.parameters["configuration"])
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values == {"geom_max_iter": 500}
    assert by_charge["muonium"].parameter_values == {"geom_max_iter": 400}
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    assert view.form_state["rows"] == []
    assert "muon_relaxation_form" not in document.ui_state


def test_muon_controller_materialize_after_edit_rewrites_same_directory(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.materialize(["positivemuon:Site 1"]) == [job]
    seed_mapping_before = dict(definition.parameters["job_seed_names_by_label"])
    seed = seed_mapping_before["positivemuon:Site 1"]
    param_path = output_root / "positivemuon" / seed / f"{seed}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is True
    )
    job.status = JobStatus.DEFINED

    assert controller.materialize(["positivemuon:Site 1"]) == [job]

    assert list(document.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    # A changed seed name would relocate the directory and orphan the preserved
    # remote directory.
    assert definition.parameters["job_seed_names_by_label"] == seed_mapping_before
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered


def test_muon_controller_build_states_keeps_job_after_parameter_edit(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, _definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    job.status = JobStatus.COMPLETED
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = controller.build_states()
    assert before["positivemuon:Site 1"].job_id == job.id

    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is True
    )

    after = controller.build_states()
    assert after["positivemuon:Site 1"].job_id == job.id
    assert after["positivemuon:Site 1"].status == JobStatus.COMPLETED.value


def test_muon_controller_row_is_continuable_uses_charge_qualified_key(
    tmp_path: Path,
) -> None:
    """Finding 8: gating reads the job's real status, never displayed text —
    a dismissed FAILED job (which *displays* "ready") is still continuable.
    The charge-qualified key (finding 7) is what actually resolves to the
    right row's job here; see ``test_muon_controller_continue_job_rejects_bare_label``
    for the strict charge-qualification check ``continue_job`` performs (a
    bare label happens to still resolve to *a* job through the pre-existing
    ``jobs_for_keys`` label-matching fallback, so ``row_is_continuable`` alone
    is not that safety net)."""

    from mufinder.domain.models import utc_now

    service = ProjectService()
    document, _definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    job.status = JobStatus.RUNNING
    assert controller.row_is_continuable("positivemuon:Site 1") is False

    for status in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.STOPPED,
        JobStatus.CANCELLED,
    ):
        job.status = status
        assert controller.row_is_continuable("positivemuon:Site 1") is True

    job.status = JobStatus.FAILED
    job.dismissed_at = utc_now()
    assert controller.row_is_continuable("positivemuon:Site 1") is True

    # The muonium row shares the label but has no bound job of its own.
    assert controller.row_is_continuable("muonium:Site 1") is False


def test_muon_controller_render_row_param_text_previews_without_persisting(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = copy.deepcopy(definition.parameters["configuration"])

    text = controller.render_row_param_text(
        "positivemuon:Site 1", {"geom_max_iter": 777}, ()
    )

    assert "GEOM_MAX_ITER : 777" in text
    assert definition.parameters["configuration"] == before
    assert controller.render_row_param_text("unknown:Site 1", {}, ()) == ""


def test_muon_controller_continue_job_returns_to_defined_and_rewrites_directory(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import utc_now

    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    controller.materialize(["positivemuon:Site 1"])
    seed_mapping_before = dict(definition.parameters["job_seed_names_by_label"])
    seed = seed_mapping_before["positivemuon:Site 1"]
    param_path = output_root / "positivemuon" / seed / f"{seed}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    job.execution_target = "myhpc"
    job.remote_directory = "/scratch/user/site1"
    job.remote_job_id = "999"
    job.error_message = "boom"
    job.dismissed_at = utc_now()

    continued = controller.continue_job("positivemuon:Site 1", {"geom_max_iter": 500})

    assert continued is job
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == "/scratch/user/site1"
    assert job.remote_job_id is None
    assert job.error_message is None
    assert job.dismissed_at is None

    assert list(document.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    # A changed seed name would relocate the directory and orphan the
    # preserved remote directory.
    assert definition.parameters["job_seed_names_by_label"] == seed_mapping_before
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered

    continuation = next(
        record for record in document.provenance if record.action == "job_continuation"
    )
    assert continuation.job_id == job.id
    assert continuation.parameters["remote_directory"] == "/scratch/user/site1"


def test_muon_controller_continue_job_rejects_non_continuable_job_without_mutating(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = copy.deepcopy(definition.parameters["configuration"])

    job.status = JobStatus.RUNNING
    assert controller.continue_job("positivemuon:Site 1", {"geom_max_iter": 500}) is None

    assert job.status == JobStatus.RUNNING
    assert definition.parameters["configuration"] == before
    assert not any(record.action == "job_continuation" for record in document.provenance)


def test_muon_controller_continue_job_rejects_bare_label(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = copy.deepcopy(definition.parameters["configuration"])

    job.status = JobStatus.FAILED
    assert controller.continue_job("Site 1", {"geom_max_iter": 500}) is None

    assert job.status == JobStatus.FAILED
    assert definition.parameters["configuration"] == before


def test_muon_controller_continue_job_returns_none_when_materialize_fails(
    tmp_path: Path,
) -> None:
    """Muon counterpart of the pristine "materialize can fail silently" check
    above (audit finding 1)."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    controller.materialize(["positivemuon:Site 1"])
    seed_mapping_before = dict(definition.parameters["job_seed_names_by_label"])
    seed = seed_mapping_before["positivemuon:Site 1"]
    param_path = output_root / "positivemuon" / seed / f"{seed}.param"
    original_rendered = param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    job.remote_directory = "/scratch/user/site1"
    controller.materialize = lambda keys: []

    result = controller.continue_job("positivemuon:Site 1", {"geom_max_iter": 500})

    assert result is None
    config = _deserialize_muon_config(definition.parameters["configuration"])
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values == {"geom_max_iter": 500}
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == "/scratch/user/site1"
    assert param_path.read_text(encoding="utf-8") == original_rendered
    assert not any(record.action == "job_continuation" for record in document.provenance)


def test_muon_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, deps, output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1", "muonium:Site 2"], target="cluster")

    assert local.calls == []
    assert [
        [job.calc_type for job in group] for group, _extras in cluster.calls
    ] == [["castep_positivemuon_relaxation"], ["castep_muonium_relaxation"]]


def test_muon_controller_build_result_records_position_structure_and_site_mutation(
    tmp_path: Path,
) -> None:
    """build_result with a REAL -out.cell containing an H:mu muon contributes
    the lattice + folded-position payload, creates the derived structure, and
    mutates the source site (the sanctioned document side effect)."""

    from mufinder.app.relaxation_types import MuonRelaxationRow
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site

    row = MuonRelaxationRow(
        site_id=site.id,
        label="Shared Site",
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state="muonium",
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # A REAL -out.cell fixture (2x2x1 supercell of the 4 A host) on disk, fed
    # through the controller's parse + position-extraction helpers.
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(_MUON_OUT_CELL_TEXT, encoding="utf-8")

    deps = _stub_muon_deps(
        muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
    )
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-12.0, warnings=())
    artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    result = controller.build_result(job, parsed, run, artifacts)

    # Relaxed supercell lattice payload.
    assert result.payload["lattice_a"] == pytest.approx(8.0)
    assert result.payload["lattice_b"] == pytest.approx(8.0)
    assert result.payload["lattice_c"] == pytest.approx(4.0)
    # Muon position: raw supercell fraction, and folded back to host coordinates
    # through the row's 2x2x1 supercell matrix.
    assert result.payload["muon_final_fractional_position_supercell"] == pytest.approx(
        [0.3, 0.4, 0.5]
    )
    assert result.payload["muon_final_fractional_position_host"] == pytest.approx(
        [0.6, 0.8, 0.5]
    )
    # No output_files contributed — the call site already passes them generically.
    assert result.output_files == []

    # Exactly one derived structure created on the document, with the right
    # origin, parentage, and the muon present.
    assert len(result.derived_structure_ids) == 1
    derived = document.structures[result.derived_structure_ids[0]]
    assert derived.origin == "muon_relaxation"
    assert derived.parent_structure_id == host.id
    assert derived.label == "Shared Site relaxed"
    assert derived.properties["source_job_id"] == str(job.id)
    assert derived.properties["source_workflow_stage"] == WorkflowStage.MUON_RELAXATION.value
    assert any(str(s).strip().casefold() == "h:mu" for s in derived.species)

    # Sanctioned document side effect: the source site is marked relaxed and
    # linked to this run with the job's total energy.
    assert site.status == "relaxed"
    assert site.relaxation_run_id == run.id
    assert site.total_energy_eV == pytest.approx(-12.0)


_MUON_OUT_CELL_TEXT_CONTINUED = """
%BLOCK LATTICE_CART
ang
8.2 0.0 0.0
0.0 8.2 0.0
0.0 0.0 4.1
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
H:mu 0.35 0.45 0.55
%ENDBLOCK POSITIONS_FRAC
"""


def _muon_row_and_definition(document, host, site, charge_state: str):
    from mufinder.app.relaxation_types import MuonRelaxationRow
    from mufinder.domain.models import CalculationDefinition

    row = MuonRelaxationRow(
        site_id=site.id,
        label=site.label,
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state=charge_state,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition
    return row, definition


def test_muon_controller_recompletion_updates_structure_and_site_to_new_run(
    tmp_path: Path,
) -> None:
    """A continued muon job's re-completion refreshes the same derived structure
    and re-points the site's ``relaxation_run_id``/``total_energy_eV`` at the
    new run, so the site and the structure never describe different runs."""

    from mufinder.domain.models import MuonSite, Result

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site
    row, definition = _muon_row_and_definition(document, host, site, "muonium")

    job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    out_cell = output_dir / f"{job.label}-out.cell"

    deps = _stub_muon_deps(
        muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
    )
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    def complete(run_index: int, cell_text: str, total_energy: float):
        run = CalculationRun(job_id=job.id, run_index=run_index)
        document.runs[run.id] = run
        out_cell.write_text(cell_text, encoding="utf-8")
        artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
        parsed = CastepResult(completed=True, total_energy=total_energy, warnings=())
        controller_result = controller.build_result(job, parsed, run, artifacts)
        result = Result(
            calculation_run_id=run.id,
            result_type=job.calc_type,
            derived_structure_ids=list(controller_result.derived_structure_ids),
        )
        document.results[result.id] = result
        run.result_id = result.id
        return run, result, controller_result

    run_one, first_result, _first = complete(1, _MUON_OUT_CELL_TEXT, -12.0)
    first_id = first_result.derived_structure_ids[0]
    structure_count = len(document.structures)
    assert site.relaxation_run_id == run_one.id

    run_two, second_result, second = complete(2, _MUON_OUT_CELL_TEXT_CONTINUED, -12.5)

    # Stable structure id, no duplicate, both Results converge on it.
    assert second_result.derived_structure_ids == [first_id]
    assert len(document.structures) == structure_count
    assert first_result.derived_structure_ids == second_result.derived_structure_ids

    derived = document.structures[first_id]
    assert derived.lattice_vectors == [[8.2, 0.0, 0.0], [0.0, 8.2, 0.0], [0.0, 0.0, 4.1]]
    assert derived.fractional_positions[-1] == pytest.approx([0.35, 0.45, 0.55])
    assert derived.parent_structure_id == host.id
    # The refresh replaces the geometry as a set: species and positions must not
    # be able to drift out of step with each other.
    assert len(derived.species) == len(derived.fractional_positions)
    assert derived.species[-1].strip().casefold() == "h:mu"

    # Payload follows the new run, and so do the site fields.
    assert second.payload["muon_final_fractional_position_supercell"] == pytest.approx(
        [0.35, 0.45, 0.55]
    )
    assert site.status == "relaxed"
    assert site.relaxation_run_id == run_two.id
    assert site.relaxation_run_id != run_one.id
    assert site.total_energy_eV == pytest.approx(-12.5)
    assert service.validate_document(document) == []


def test_muon_controller_recompletion_keeps_charge_states_separate(
    tmp_path: Path,
) -> None:
    """One site's two charge states are two jobs with the SAME ``job.label`` and
    two derived structures with the same ``"{label} relaxed"`` label. Each job's
    re-completion must refresh its own structure — the reason the lookup keys on
    the job id and never on the label."""

    from mufinder.domain.models import MuonSite

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site

    positive_row, definition = _muon_row_and_definition(document, host, site, "positive_muon")
    muonium_row, _ = _muon_row_and_definition(document, host, site, "muonium")

    positive_job = _muon_job(
        service, document, "Shared Site", "castep_positivemuon_relaxation"
    )
    muonium_job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    assert positive_job.label == muonium_job.label

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    out_cell = output_dir / f"{positive_job.label}-out.cell"
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )

    def controller_for(row):
        deps = _stub_muon_deps(
            muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
        )
        return MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    def complete(controller, job, run_index: int, cell_text: str):
        run = CalculationRun(job_id=job.id, run_index=run_index)
        document.runs[run.id] = run
        out_cell.write_text(cell_text, encoding="utf-8")
        artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
        parsed = CastepResult(completed=True, total_energy=-12.0, warnings=())
        return controller.build_result(job, parsed, run, artifacts)

    positive_controller = controller_for(positive_row)
    muonium_controller = controller_for(muonium_row)

    positive_id = complete(
        positive_controller, positive_job, 1, _MUON_OUT_CELL_TEXT
    ).derived_structure_ids[0]
    muonium_id = complete(
        muonium_controller, muonium_job, 1, _MUON_OUT_CELL_TEXT
    ).derived_structure_ids[0]
    assert positive_id != muonium_id
    assert (
        document.structures[positive_id].label == document.structures[muonium_id].label
    )

    # Continue only the positive-muon job: its structure moves, the muonium
    # structure is untouched.
    second_positive = complete(
        positive_controller, positive_job, 2, _MUON_OUT_CELL_TEXT_CONTINUED
    )
    assert second_positive.derived_structure_ids == [positive_id]
    assert document.structures[positive_id].lattice_vectors == [
        [8.2, 0.0, 0.0],
        [0.0, 8.2, 0.0],
        [0.0, 0.0, 4.1],
    ]
    assert document.structures[muonium_id].lattice_vectors == [
        [8.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 4.0],
    ]


def test_muon_controller_build_result_fails_loudly_on_muonless_cell(tmp_path: Path) -> None:
    """The success-path invariant is self-contained: if build_result is ever
    reached with a relaxed cell that lacks the muon (the inline guard's job to
    prevent), it raises a descriptive error instead of a deep TypeError."""

    from mufinder.app.relaxation_types import MuonRelaxationRow
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = _muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site
    row = MuonRelaxationRow(
        site_id=site.id,
        label="Shared Site",
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state="muonium",
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition
    job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # Same fixture with the muon line removed: extraction returns None.
    muonless_text = "\n".join(
        line for line in _MUON_OUT_CELL_TEXT.splitlines() if "h:mu" not in line.casefold()
    )
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(muonless_text, encoding="utf-8")

    deps = _stub_muon_deps(
        muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
    )
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-12.0, warnings=())
    # The muonless cell still parses (lattice + Si present), so the carrier's
    # ``relaxed_cell``/``lattice_abc`` are populated — build_result proceeds past
    # the empty-carrier guard and only then raises on the missing H:mu position.
    artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    with pytest.raises(ValueError, match="relaxed cell containing"):
        controller.build_result(job, parsed, run, artifacts)
    # No partial side effects: site untouched, no derived structure added.
    assert site.status == "staged"
    assert site.relaxation_run_id is None


# --------------------------------------------------------------------------- #
# UepController
# --------------------------------------------------------------------------- #
#
# One of the simpler migrations: no pre_submit_check, no submission_groups
# override (single group, normalize=str, base-default is_ready), and a small
# build_result contributing only the ``density_available`` boolean flag — no
# derived structures and no document mutation. The tests drive it with narrow
# fakes (no MainWindow) and a real ``.den_fmt`` file on disk to exercise the
# present-vs-absent flag branches.


class _FakeUepView:
    """Minimal UEPWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.runtime_states: list[dict] = []

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))


def _stub_uep_deps(**overrides) -> UepDeps:
    base = dict(
        deserialize_uep_config=lambda _payload: None,
        sync_equivalent_uep_mappings=lambda _definition: None,
        uep_potential_grid_result_for_job=lambda _job_id: None,
    )
    base.update(overrides)
    return UepDeps(**base)


def _uep_job(service: ProjectService, document: ProjectDocument, label: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )


def test_uep_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    controller = UepController(
        _make_context(document=document), _stub_uep_deps(), _FakeUepView()
    )
    handled = _uep_job(service, document, "uep_row1")
    other = service.add_job(
        document,
        label="row1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    assert controller.handles_job(handled) is True
    assert controller.handles_job(other) is False


def test_uep_controller_uses_workflow_row_ready_predicate() -> None:
    controller = UepController(_make_context(), _stub_uep_deps(), _FakeUepView())
    # UEP uses the base-default (wider) workflow-row predicate: "not generated"
    # / "defined" / "ready" rows are ready, only "failed" is excluded.
    assert controller.is_ready({"r": _State("not generated")}, "r") is True
    assert controller.is_ready({"r": _State("ready")}, "r") is True
    assert controller.is_ready({"r": _State("failed")}, "r") is False


def test_uep_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeUepView()
    controller = UepController(
        _make_context(document=None), _stub_uep_deps(), view
    )
    controller.refresh_view()
    assert view.runtime_states == [{}]


def test_uep_controller_build_result_flag_true_when_den_fmt_present(tmp_path: Path) -> None:
    """build_result reports ``density_available: True`` when the completed job's
    ``{seed}.den_fmt`` output exists on disk — byte-for-byte parity with the
    former inline ``result_payload["density_available"] = True`` on the success
    path (where the inline guard has already guaranteed the file)."""

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _uep_job(service, document, "uep_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}.den_fmt").write_text("density", encoding="utf-8")

    deps = _stub_uep_deps()
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = UepController(context, deps, _FakeUepView())

    parsed = CastepResult(completed=True, total_energy=-20.5, warnings=())
    result = controller.build_result(job, parsed, run, _no_artifacts())

    assert result.payload == {"density_available": True}
    # No output_files contributed — the call site already passes them generically.
    assert result.output_files == []
    assert result.derived_structure_ids == []


def test_uep_controller_build_result_flag_false_when_den_fmt_absent(tmp_path: Path) -> None:
    """With no ``{seed}.den_fmt`` on disk, build_result reports
    ``density_available: False``. In production the inline guard fails the job
    before build_result is ever reached, so this branch is only exercised by a
    direct unit call — mirroring the pristine missing-out.cell unit test."""

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _uep_job(service, document, "uep_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()  # deliberately empty — no den_fmt written

    deps = _stub_uep_deps()
    context = _make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = UepController(context, deps, _FakeUepView())

    parsed = CastepResult(completed=True, total_energy=-20.5, warnings=())
    result = controller.build_result(job, parsed, run, _no_artifacts())

    assert result.payload == {"density_available": False}


def test_uep_controller_materialize_writes_job_ids_by_label(tmp_path: Path) -> None:
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = UepConfig(name_prefix="uep", rows=(UepSourceRow(label="row1"),))
    output_root = document.project_root / "files" / "uep"
    definition = service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps(
        deserialize_uep_config=lambda _payload: config,
    )
    context = _make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = UepController(context, deps, _FakeUepView())

    jobs = controller.materialize(["row1"])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.calc_type == "castep_uep_source"
    mapping = definition.parameters["job_ids_by_label"]
    assert mapping["row1"] == str(job.id)
    # Rendered inputs were written under the output root.
    job_dir = output_root / job.label
    assert (job_dir / f"{job.label}.cell").exists()
    assert (job_dir / f"{job.label}.param").exists()

    # Re-materializing an existing DEFINED job returns the same job (no dup).
    again = controller.materialize(["row1"])
    assert len(again) == 1
    assert again[0].id == job.id


def test_uep_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = UepConfig(name_prefix="uep", rows=(UepSourceRow(label="row1"),))
    output_root = document.project_root / "files" / "uep"
    service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps(
        deserialize_uep_config=lambda _payload: config,
    )

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = UepController(context, deps, _FakeUepView())

    controller.submit(["row1"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_uep_source"


def test_uep_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = UepConfig(name_prefix="uep", rows=(UepSourceRow(label="row1"),))
    output_root = document.project_root / "files" / "uep"
    service.register_uep_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "uep", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    deps = _stub_uep_deps(
        deserialize_uep_config=lambda _payload: config,
    )

    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = UepController(context, deps, _FakeUepView())

    controller.submit(["row1"], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    submitted, extras = cluster.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_uep_source"


# --------------------------------------------------------------------------- #
# QuantumCorrectionsController
# --------------------------------------------------------------------------- #
#
# The two-stage (singlepoint → phonon) migration. Its ``build_result`` carries the
# richest result logic — parsing a real ``.phonon`` file into the muon ZPE,
# eigenvalues, mode participations, and the ``w1`` entanglement witness — for a
# quantum-*phonon* job, while a quantum-*singlepoint* job contributes an empty
# result (but is still ``handles_job``-owned so the two-stage pair is coherent).
# ``pre_submit_check`` runs the shared host guard; ``materialize`` dispatches each
# selected source to its current stage. Driven below with narrow fakes (no
# MainWindow) and a REAL ``.phonon`` fixture containing an ``H:mu`` muon.

from dataclasses import dataclass as _dataclass

from mufinder.workflow.quantum import (
    QUANTUM_PHONON_CALC_TYPE,
    QUANTUM_SINGLEPOINT_CALC_TYPE,
)
from mufinder.workflow.types import InputTemplate
from mufinder.app.workflow_controllers.quantum_corrections import (
    QUANTUM_STAGE_PHONON,
    QUANTUM_STAGE_SINGLEPOINT,
    QuantumCorrectionsController,
    QuantumCorrectionsDeps,
)


# A minimal, valid CASTEP ``.phonon`` file with one Gamma q-point and an H:mu
# atom (mirrors the fixture used by tests/app/test_main_window.py so build_result
# exercises the same real science path the inline guard validates).
_QUANTUM_PHONON_TEXT = """ BEGIN header
 Number of ions         2
 Number of branches     6
 Number of wavevectors  1
 Frequencies in         cm-1
 Unit cell vectors (A)
    1.000000    0.000000    0.000000
    0.000000    1.000000    0.000000
    0.000000    0.000000    1.000000
 Fractional Co-ordinates
     1     0.000000    0.000000    0.000000   N        14.006740
     2     0.500000    0.500000    0.500000   H:mu      0.113429
 END header
     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000
       1     100.000000
       2     400.000000
       3     900.000000
       4    1600.000000
       5    2500.000000
       6    3600.000000
                        Phonon Eigenvectors
Mode Ion                X                                   Y                                   Z
   1   1  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   1   2  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   2   1 -0.336650000000  0.000000000000      0.019803000000  0.000000000000      0.138621000000  0.000000000000
   2   2 -0.693103000000  0.000000000000      0.376256000000  0.000000000000      0.495074000000  0.000000000000
   3   1 -0.080432200000  0.000000000000     -0.350117000000  0.000000000000      0.230257000000  0.000000000000
   3   2  0.189252000000  0.000000000000     -0.619801000000  0.000000000000      0.407510000000  0.000000000000
   4   1  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   4   2  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   5   1 -0.336650000000  0.000000000000      0.019803000000  0.000000000000      0.138621000000  0.000000000000
   5   2 -0.693103000000  0.000000000000      0.376256000000  0.000000000000      0.495074000000  0.000000000000
   6   1 -0.080432200000  0.000000000000     -0.350117000000  0.000000000000      0.230257000000  0.000000000000
   6   2  0.189252000000  0.000000000000     -0.619801000000  0.000000000000      0.407510000000  0.000000000000
"""


class _FakeQuantumView:
    """Minimal QuantumCorrectionsWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.sources: list[list] = []
        self.results: list[list] = []

    def set_sources(self, rows) -> None:
        self.sources.append(list(rows))

    def set_results(self, rows) -> None:
        self.results.append(list(rows))


@_dataclass
class _FakeQuantumStageState:
    stage: str
    status: str
    current_job: Job | None = None


@_dataclass
class _FakeQuantumCandidate:
    key: str
    site: object
    run: object
    job: object
    structure: object


def _stub_quantum_deps(**overrides) -> QuantumCorrectionsDeps:
    base = dict(
        ensure_quantum_corrections_definition=lambda _keys: None,
        eligible_quantum_source_candidates=lambda: [],
        quantum_source_relative_energies=lambda: {},
        quantum_source_candidate_by_key=lambda _key: None,
        quantum_stage_state=lambda _definition, _candidate: _FakeQuantumStageState(
            stage=QUANTUM_STAGE_SINGLEPOINT, status="ready"
        ),
        build_quantum_corrections_states=lambda: {},
        jobs_for_quantum_source_keys=lambda _keys: [],
        singlepoint_job_for_quantum_source_key=lambda _definition, _key: None,
        phonon_job_for_quantum_source_key=lambda _definition, _key: None,
        quantum_seed_name_for_key=lambda _definition, key: f"quantum_{key}",
        quantum_runtime_configuration=lambda _definition, _key: ({}, "unknown"),
        workflow_row_configuration=lambda _definition, _key: ({}, (), {}),
        workflow_row_configuration_exists=lambda _definition, _key: False,
        quantum_num_farms_from_extras=lambda _extras: 1,
        quantum_source_label=lambda candidate: candidate.site.label,
        quantum_method_from_gui_value=lambda _value: "finite_displacement",
        charge_state_for_job=lambda _job: None,
        charge_state_label_for_job=lambda _job: "",
        quantum_can_reset_to_stage_one=lambda _definition, _candidate: False,
        input_template_from_keyword_values=lambda _values: InputTemplate(),
        write_quantum_job_directory=lambda _spec, _root, *, seed_name: Path("/tmp") / seed_name,
        dedupe_jobs=lambda jobs: jobs,
        check_definition_host=lambda _definition: True,
        format_energy=_format_energy_stub,
        confirm=lambda _title, _text: True,
    )
    base.update(overrides)
    return QuantumCorrectionsDeps(**base)


def _format_energy_stub(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6g} eV"
    except (TypeError, ValueError):
        return ""


def _quantum_job(service: ProjectService, document: ProjectDocument, label: str, calc_type: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=calc_type,
    )


def test_quantum_controller_handles_both_stage_calc_types(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    controller = QuantumCorrectionsController(
        _make_context(document=document), _stub_quantum_deps(), _FakeQuantumView()
    )
    singlepoint = _quantum_job(service, document, "quantum_singlepoint_s", QUANTUM_SINGLEPOINT_CALC_TYPE)
    phonon = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    other = service.add_job(
        document,
        label="u",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_uep_source",
    )
    # Both stage calc types are owned so the two-stage pair is coherent.
    assert controller.handles_job(singlepoint) is True
    assert controller.handles_job(phonon) is True
    assert controller.handles_job(other) is False


def test_quantum_controller_refresh_view_without_document_clears_tables() -> None:
    view = _FakeQuantumView()
    controller = QuantumCorrectionsController(
        _make_context(document=None), _stub_quantum_deps(), view
    )
    controller.refresh_view()
    assert view.sources == [[]]
    assert view.results == [[]]


def test_quantum_controller_refresh_view_populates_result_row_status(tmp_path: Path) -> None:
    """A completed phonon result becomes a result row whose ``status`` mirrors the
    producing job's status, so the results table can colour it with the shared palette."""

    from mufinder.domain.models import Result

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(
        result_type=QUANTUM_PHONON_CALC_TYPE,
        calculation_run_id=run.id,
        data={"muon_zpe_total_eV": 0.1},
    )
    document.results[result.id] = result

    view = _FakeQuantumView()
    controller = QuantumCorrectionsController(
        _make_context(document=document, project_service=service), _stub_quantum_deps(), view
    )
    controller.refresh_view()

    assert len(view.results[-1]) == 1
    result_row = view.results[-1][0]
    assert result_row.label == "quantum_phonon_s"
    assert result_row.status == "completed"


def test_quantum_controller_pre_submit_check_blocks_submission(tmp_path: Path) -> None:
    """An unresolved host mismatch silently refuses the submission: the host guard
    returns False, so no submission plan is built and nothing is dispatched."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    materialize_calls: list[list] = []
    deps = _stub_quantum_deps(
        check_definition_host=lambda _definition: False,
    )
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())
    # Prove materialize is never reached by making it record + explode if called.
    controller.materialize = lambda keys: materialize_calls.append(list(keys)) or []  # type: ignore[assignment]

    controller.submit(["site-key"], target="local")

    assert local.calls == []  # nothing dispatched
    assert infos == []  # the refusal is silent (no "no ready jobs" dialog)
    assert materialize_calls == []  # no submission plan was even built


def test_quantum_controller_pre_submit_check_passes_when_host_matches(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    deps = _stub_quantum_deps(check_definition_host=lambda _definition: True)
    controller = QuantumCorrectionsController(
        _make_context(document=document, project_service=service), deps, _FakeQuantumView()
    )
    assert controller.pre_submit_check(["site-key"]) is True


def test_quantum_controller_build_result_singlepoint_is_empty_but_handled(tmp_path: Path) -> None:
    """A quantum-singlepoint job contributes an empty ControllerResult (base
    payload only), yet is still owned by the controller so the two-stage pair is
    coherent for result dispatch."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_singlepoint_s", QUANTUM_SINGLEPOINT_CALC_TYPE)
    run = CalculationRun(job_id=job.id)
    controller = QuantumCorrectionsController(
        _make_context(document=document, project_service=service), _stub_quantum_deps(), _FakeQuantumView()
    )

    assert controller.handles_job(job) is True
    result = controller.build_result(job, _parsed(), run, _no_artifacts())
    assert result.payload == {}
    assert result.output_files == []
    assert result.derived_structure_ids == []
    assert result.warnings == []


def test_quantum_controller_build_result_phonon_parses_zpe_and_witness(tmp_path: Path) -> None:
    """A quantum-phonon job's build_result parses the real ``.phonon`` file and
    contributes the ZPE / eigenvalue / mode / witness payload — matching the
    fields the former inline result block assigned."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    phonon_path = output_dir / "quantum_s.phonon"
    phonon_path.write_text(_QUANTUM_PHONON_TEXT, encoding="utf-8")

    deps = _stub_quantum_deps()
    controller = QuantumCorrectionsController(
        _make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: output_dir,
            job_seed_name=lambda _job: "quantum_s",
        ),
        deps,
        _FakeQuantumView(),
    )

    artifacts = collect_job_artifacts(output_dir, "quantum_s", want_phonon=True)
    result = controller.build_result(job, _parsed(), run, artifacts)

    payload = result.payload
    assert payload["phonon_file"] == str(phonon_path)
    assert payload["gamma_only"] is True
    assert payload["phonon_mode_eigenvalues_cm"] == [100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0]
    assert isinstance(payload["muon_zpe_total_eV"], float)
    assert payload["muon_zpe_total_eV"] > 0.0
    assert len(payload["muon_zpe_modes"]) == 3
    assert all("mode_index" in mode and "frequency_cm" in mode for mode in payload["muon_zpe_modes"])
    assert isinstance(payload["w1_entanglement_witness"], float)
    # No output_files or derived structures contributed by build_result.
    assert result.output_files == []
    assert result.derived_structure_ids == []
    # The muon mass CASTEP reported is the muon mass, so no warning is raised.
    assert result.warnings == []


def test_quantum_controller_build_result_warns_on_a_hydrogen_mass_muon(tmp_path: Path) -> None:
    """A phonon run CASTEP performed with the default hydrogen mass for H:mu is
    reported as a run warning (dashboard-visible), not only in the payload."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / "quantum_s.phonon").write_text(
        _QUANTUM_PHONON_TEXT.replace("H:mu      0.113429", "H:mu      1.007940"),
        encoding="utf-8",
    )

    controller = QuantumCorrectionsController(
        _make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: output_dir,
            job_seed_name=lambda _job: "quantum_s",
        ),
        _stub_quantum_deps(),
        _FakeQuantumView(),
    )

    artifacts = collect_job_artifacts(output_dir, "quantum_s", want_phonon=True)
    result = controller.build_result(job, _parsed(), run, artifacts)

    assert len(result.warnings) == 1
    assert "not the muon mass" in result.warnings[0]
    assert any(
        "not the muon mass" in warning for warning in result.payload["phonon_warnings"]
    )


def test_quantum_controller_build_result_phonon_missing_dir_is_empty(tmp_path: Path) -> None:
    """If the job's output directory can't be resolved, build_result yields an
    empty result rather than crashing (the inline guard fails such jobs first)."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)
    deps = _stub_quantum_deps()
    controller = QuantumCorrectionsController(
        _make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: None,
        ),
        deps,
        _FakeQuantumView(),
    )
    assert controller.build_result(job, _parsed(), run, ParsedJobArtifacts()).payload == {}


def _seed_quantum_two_stage_document(service: ProjectService, tmp_path: Path):
    """Seed a document with two eligible relaxed sources sharing one definition."""

    from mufinder.domain.models import MuonSite, Structure

    document = service.create_project(tmp_path / "q2.mufinder", "Q2")
    output_root = document.project_root / "files" / "quantum_corrections"

    candidates = {}
    site_ids = []
    structure_ids = []
    for label in ("siteA", "siteB"):
        relaxed = Structure(
            label=f"{label} relaxed",
            origin="muon_relaxation",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
            properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
        )
        document.structures[relaxed.id] = relaxed
        source_job = service.add_job(
            document,
            label=f"mu-{label}",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_positivemuon_relaxation",
        )
        source_job.status = JobStatus.COMPLETED
        run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
        document.runs[run.id] = run
        site = MuonSite(label=label, status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
        document.sites[site.id] = site
        site_ids.append(site.id)
        structure_ids.append(relaxed.id)
        candidates[str(site.id)] = _FakeQuantumCandidate(
            key=str(site.id), site=site, run=run, job=source_job, structure=relaxed
        )

    definition = service.register_quantum_phonon_definition(
        document,
        input_structure_ids=structure_ids,
        input_site_ids=site_ids,
        parameters={
            "method": "finite_displacement",
            "pseudopotential_family": "unknown",
            "singlepoint_job_ids_by_label": {},
            "phonon_job_ids_by_label": {},
            "job_seed_names_by_label": {},
        },
        output_directory=output_root,
    )
    return document, definition, candidates, output_root


def test_quantum_controller_materialize_two_stage_mixed_keys(tmp_path: Path) -> None:
    """Mixed selection where one source is at the singlepoint stage and another at
    the phonon stage: materialize writes each key into the correct per-stage
    mapping with the correct calc_type, in one pass."""

    from mufinder.app.relaxation_types import SupercellMatrix  # noqa: F401  (import parity)

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    keys = list(candidates.keys())
    singlepoint_key, phonon_key = keys[0], keys[1]

    # siteA is fresh (singlepoint stage); siteB already has a COMPLETED singlepoint
    # job, so its stage has advanced to phonon.
    completed_singlepoint = _quantum_job(
        service, document, "quantum_singlepoint_siteB", QUANTUM_SINGLEPOINT_CALC_TYPE
    )
    completed_singlepoint.status = JobStatus.COMPLETED
    definition.parameters["singlepoint_job_ids_by_label"][phonon_key] = str(completed_singlepoint.id)

    stage_by_key = {
        singlepoint_key: _FakeQuantumStageState(stage=QUANTUM_STAGE_SINGLEPOINT, status="ready"),
        phonon_key: _FakeQuantumStageState(stage=QUANTUM_STAGE_PHONON, status="ready"),
    }

    def write_dir(spec, root, *, seed_name):
        job_dir = root / seed_name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def singlepoint_job_for_key(_definition, key):
        job_id = _definition.parameters.get("singlepoint_job_ids_by_label", {}).get(key)
        return document.jobs.get(UUID(job_id)) if isinstance(job_id, str) else None

    def phonon_job_for_key(_definition, key):
        job_id = _definition.parameters.get("phonon_job_ids_by_label", {}).get(key)
        return document.jobs.get(UUID(job_id)) if isinstance(job_id, str) else None

    deps = _stub_quantum_deps(
        ensure_quantum_corrections_definition=lambda _keys: definition,
        quantum_source_candidate_by_key=lambda key: candidates.get(key),
        quantum_stage_state=lambda _definition, candidate: stage_by_key[candidate.key],
        singlepoint_job_for_quantum_source_key=singlepoint_job_for_key,
        phonon_job_for_quantum_source_key=phonon_job_for_key,
        write_quantum_job_directory=write_dir,
    )
    context = _make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())

    jobs = controller.materialize([singlepoint_key, phonon_key])

    assert len(jobs) == 2
    calc_types = {job.calc_type for job in jobs}
    assert calc_types == {QUANTUM_SINGLEPOINT_CALC_TYPE, QUANTUM_PHONON_CALC_TYPE}

    singlepoint_mapping = definition.parameters["singlepoint_job_ids_by_label"]
    phonon_mapping = definition.parameters["phonon_job_ids_by_label"]
    # siteA got a fresh singlepoint job; siteB got a fresh phonon continuation.
    assert singlepoint_key in singlepoint_mapping
    new_singlepoint_id = singlepoint_mapping[singlepoint_key]
    assert document.jobs[UUID(new_singlepoint_id)].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE
    assert phonon_key in phonon_mapping
    new_phonon_id = phonon_mapping[phonon_key]
    assert document.jobs[UUID(new_phonon_id)].calc_type == QUANTUM_PHONON_CALC_TYPE
    # The pre-existing completed singlepoint for siteB is untouched (not re-created).
    assert singlepoint_mapping.get(phonon_key) == str(completed_singlepoint.id)
    # Both seed names were recorded.
    seed_mapping = definition.parameters["job_seed_names_by_label"]
    assert singlepoint_key in seed_mapping and phonon_key in seed_mapping
    # The phonon mapping stays aliased into job_ids_by_label (load-bearing fallback).
    assert definition.parameters["job_ids_by_label"] is phonon_mapping


def _write_stage_one_cell(output_root: Path, seed_name: str, *, with_species_mass: bool) -> Path:
    """Write a stage-1 singlepoint ``.cell`` into the shared seed directory."""

    job_dir = output_root / seed_name
    job_dir.mkdir(parents=True, exist_ok=True)
    cell_path = job_dir / f"{seed_name}.cell"
    species_mass = (
        "\n%BLOCK SPECIES_MASS\nAMU\nH:mu 0.11342892566\n%ENDBLOCK SPECIES_MASS\n"
        if with_species_mass
        else ""
    )
    cell_path.write_text(
        "%BLOCK LATTICE_CART\nang\n5 0 0\n0 5 0\n0 0 5\n%ENDBLOCK LATTICE_CART\n"
        "\n%BLOCK POSITIONS_FRAC\nSi 0 0 0\nH:mu 0.25 0.25 0.25\n%ENDBLOCK POSITIONS_FRAC\n"
        f"{species_mass}",
        encoding="utf-8",
    )
    return cell_path


def _quantum_stage_two_guard_setup(service: ProjectService, tmp_path: Path):
    """Return (deps-kwargs, context-kwargs, phonon_key, output_root) for the guard."""

    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    phonon_key = list(candidates)[0]
    deps_kwargs = dict(
        quantum_source_candidate_by_key=lambda key: candidates.get(key),
        quantum_stage_state=lambda _definition, _candidate: _FakeQuantumStageState(
            stage=QUANTUM_STAGE_PHONON, status="ready"
        ),
    )
    context_kwargs = dict(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
    )
    return deps_kwargs, context_kwargs, phonon_key, output_root


def test_quantum_controller_refuses_stage_two_when_stage_one_omits_species_mass(
    tmp_path: Path,
) -> None:
    """A phonon continuation whose stage-1 cell has no SPECIES_MASS is refused.

    CASTEP restores species masses from the stage-1 checkpoint, so continuing from
    a singlepoint that ran without the muon mass would silently produce phonons at
    the default hydrogen mass. The refusal explains itself rather than being silent.
    """

    service = ProjectService()
    deps_kwargs, context_kwargs, phonon_key, output_root = _quantum_stage_two_guard_setup(
        service, tmp_path
    )
    _write_stage_one_cell(output_root, f"quantum_{phonon_key}", with_species_mass=False)
    infos: list[tuple[str, str]] = []
    controller = QuantumCorrectionsController(
        _make_context(infos=infos, **context_kwargs),
        _stub_quantum_deps(**deps_kwargs),
        _FakeQuantumView(),
    )

    assert controller.pre_submit_check([phonon_key]) is False
    assert len(infos) == 1
    title, message = infos[0]
    assert title == "Stage-1 muon mass missing"
    assert "SPECIES_MASS" in message
    assert "siteA" in message


def test_quantum_controller_allows_stage_two_when_stage_one_declares_species_mass(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    deps_kwargs, context_kwargs, phonon_key, output_root = _quantum_stage_two_guard_setup(
        service, tmp_path
    )
    _write_stage_one_cell(output_root, f"quantum_{phonon_key}", with_species_mass=True)
    infos: list[tuple[str, str]] = []
    controller = QuantumCorrectionsController(
        _make_context(infos=infos, **context_kwargs),
        _stub_quantum_deps(**deps_kwargs),
        _FakeQuantumView(),
    )

    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_stage_two_guard_fails_open_without_a_stage_one_cell(
    tmp_path: Path,
) -> None:
    """No readable stage-1 cell is no evidence: the guard must not block."""

    service = ProjectService()
    deps_kwargs, context_kwargs, phonon_key, output_root = _quantum_stage_two_guard_setup(
        service, tmp_path
    )
    unreadable = output_root / f"quantum_{phonon_key}" / f"quantum_{phonon_key}.cell"
    unreadable.parent.mkdir(parents=True, exist_ok=True)
    unreadable.write_text("this is not a cell file", encoding="utf-8")
    infos: list[tuple[str, str]] = []
    controller = QuantumCorrectionsController(
        _make_context(infos=infos, **context_kwargs),
        _stub_quantum_deps(**deps_kwargs),
        _FakeQuantumView(),
    )

    # Unparseable cell, and (below) no cell at all: both allow submission.
    assert controller.pre_submit_check([phonon_key]) is True
    unreadable.unlink()
    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_stage_two_guard_ignores_stage_one_rows(tmp_path: Path) -> None:
    """A row still on stage 1 is never blocked: it has no checkpoint to inherit."""

    service = ProjectService()
    deps_kwargs, context_kwargs, phonon_key, output_root = _quantum_stage_two_guard_setup(
        service, tmp_path
    )
    _write_stage_one_cell(output_root, f"quantum_{phonon_key}", with_species_mass=False)
    deps_kwargs["quantum_stage_state"] = lambda _definition, _candidate: _FakeQuantumStageState(
        stage=QUANTUM_STAGE_SINGLEPOINT, status="ready"
    )
    infos: list[tuple[str, str]] = []
    controller = QuantumCorrectionsController(
        _make_context(infos=infos, **context_kwargs),
        _stub_quantum_deps(**deps_kwargs),
        _FakeQuantumView(),
    )

    assert controller.pre_submit_check([phonon_key]) is True
    assert infos == []


def test_quantum_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    """End-to-end submit wiring: build_states → is_ready (base default) →
    jobs_for_keys → materialize → local backend, dispatched as a single group with
    no extra kwargs (task farming lives in the shared backend, not the controller)."""

    from mufinder.app.relaxation_types import RelaxationRunState

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    key = next(iter(candidates))

    stage_state = _FakeQuantumStageState(stage=QUANTUM_STAGE_SINGLEPOINT, status="ready")

    def write_dir(spec, root, *, seed_name):
        job_dir = root / seed_name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    deps = _stub_quantum_deps(
        ensure_quantum_corrections_definition=lambda _keys: definition,
        eligible_quantum_source_candidates=lambda: list(candidates.values()),
        quantum_source_candidate_by_key=lambda k: candidates.get(k),
        quantum_stage_state=lambda _definition, _candidate: stage_state,
        build_quantum_corrections_states=lambda: {key: RelaxationRunState(status="ready")},
        jobs_for_quantum_source_keys=lambda _keys: [],
        singlepoint_job_for_quantum_source_key=lambda _definition, _key: None,
        phonon_job_for_quantum_source_key=lambda _definition, _key: None,
        write_quantum_job_directory=write_dir,
    )
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())

    controller.submit([key], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE


def test_quantum_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    from mufinder.app.relaxation_types import RelaxationRunState

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_two_stage_document(
        service, tmp_path
    )
    key = next(iter(candidates))
    stage_state = _FakeQuantumStageState(stage=QUANTUM_STAGE_SINGLEPOINT, status="ready")

    def write_dir(spec, root, *, seed_name):
        job_dir = root / seed_name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    deps = _stub_quantum_deps(
        ensure_quantum_corrections_definition=lambda _keys: definition,
        eligible_quantum_source_candidates=lambda: list(candidates.values()),
        quantum_source_candidate_by_key=lambda k: candidates.get(k),
        quantum_stage_state=lambda _definition, _candidate: stage_state,
        build_quantum_corrections_states=lambda: {key: RelaxationRunState(status="ready")},
        jobs_for_quantum_source_keys=lambda _keys: [],
        singlepoint_job_for_quantum_source_key=lambda _definition, _key: None,
        phonon_job_for_quantum_source_key=lambda _definition, _key: None,
        write_quantum_job_directory=write_dir,
    )
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())

    controller.submit([key], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE


# --------------------------------------------------------------------------- #
# Energy barriers (NEB): the LAST of the seven migrations. Distinguished by
# source-key provenance, IDPP intermediate interpolation in materialize, a ``.ts``
# transition-state ``build_result`` payload, and a LAZILY-CREATED view reached via
# a ``view_getter`` (the only sanctioned constructor-shape deviation). These tests
# pin: the view_getter no-ops before creation (refresh and result dispatch both
# work with no view), the ``.ts`` payload, and source-key materialize.
# --------------------------------------------------------------------------- #

from uuid import uuid4 as _uuid4  # noqa: E402

from mufinder.domain.models import CalculationDefinition as _CalculationDefinition  # noqa: E402
from mufinder.domain.models import MuonSite as _MuonSite  # noqa: E402
from mufinder.domain.models import Structure as _Structure  # noqa: E402
from mufinder.workflow.energy_barriers import (  # noqa: E402
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
)
from mufinder.app.workflow_controllers.energy_barriers import (  # noqa: E402
    EnergyBarriersController,
    EnergyBarriersDeps,
)

_NEB_TS_FIXTURE = Path(__file__).parent.parent / "fixtures" / "neb_path.ts"


class _FakeEnergyBarriersView:
    """Minimal EnergyBarriersWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.endpoint_calls: list[tuple] = []
        self.sources: list[list] = []
        self.neb_results: list[list] = []

    def set_endpoint_options(self, rows, **kwargs) -> None:
        self.endpoint_calls.append((list(rows), kwargs))

    def set_sources(self, rows) -> None:
        self.sources.append(list(rows))

    def set_neb_results(self, rows) -> None:
        self.neb_results.append(list(rows))


@_dataclass
class _FakeNebCandidate:
    key: str
    source: dict


def _stub_energy_barriers_deps(**overrides) -> EnergyBarriersDeps:
    base = dict(
        build_energy_barrier_neb_states=lambda: {},
        ensure_energy_barrier_neb_definition=lambda _keys: None,
        energy_barrier_neb_source_candidates=lambda: [],
        eligible_energy_barrier_source_candidates=lambda: [],
        energy_barrier_neb_row_configuration_with_source_defaults=(
            lambda _definition, _key, **_kwargs: ({}, (), {})
        ),
        handle_structure_host_mismatch=lambda _mismatch: True,
        handle_site_host_mismatch=lambda _mismatch: True,
        uuid_from_source=lambda source, key: _uuid_from_source_stub(source, key),
        float_from_source=lambda source, key: _float_from_source_stub(source, key),
        fractional_triplet_from_source=lambda source, key: tuple(
            source.get(key, (0.0, 0.0, 0.0))
        ),
        fractional_position_block_from_structure=lambda structure: structure,
        dedupe_jobs=lambda jobs: jobs,
        project_convergence_input_defaults=lambda: None,
        input_template_from_keyword_values=lambda _values: InputTemplate(),
        workflow_row_configuration_exists=lambda _definition, _key: False,
        num_farms_from_extras=lambda extras: max(1, int(extras.get("num_farms", 1))),
        energy_barrier_neb_row_extras=lambda _definition, _key: {},
        charge_state_label_for_job=lambda _job: "",
        energy_barrier_endpoint_rows=lambda: [],
        selected_energy_barrier_endpoint_keys=lambda: ("", ""),
        energy_barrier_max_path_length_text=lambda: "",
        completed_energy_barrier_neb_candidates=lambda: [],
        energy_barrier_neb_sources_by_job_id=lambda: {},
        energy_barrier_neb_result_path_label=lambda _candidate, _source: "",
        neb_barrier_summaries=lambda _data: ("", "", ""),
        muon_charge_state_label=lambda value: value,
    )
    base.update(overrides)
    return EnergyBarriersDeps(**base)


def _uuid_from_source_stub(source, key):
    raw = source.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return UUID(raw.strip())
    except ValueError:
        return None


def _float_from_source_stub(source, key):
    raw = source.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _neb_job(service: ProjectService, document: ProjectDocument, label: str, calc_type: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=calc_type,
    )


def _neb_frac_block(structure):
    """Mirror ``MainWindow._fractional_position_block_from_structure`` for tests."""

    from mufinder.castep_io.types import AtomicPosition, PositionBlock

    rows = [
        AtomicPosition(str(species), tuple(float(v) for v in fractional))
        for species, fractional in zip(
            structure.species, structure.fractional_positions, strict=True
        )
    ]
    return PositionBlock(positions=tuple(rows), unit="frac")


def _neb_materialize_deps(definition, candidate, output_root, **overrides) -> EnergyBarriersDeps:
    """Deps wired so ``materialize`` runs the real science stack and creates a job
    for ``candidate`` — shared by the source-key and submit tests."""

    base = dict(
        ensure_energy_barrier_neb_definition=lambda _keys: definition,
        energy_barrier_neb_source_candidates=lambda: [candidate],
        eligible_energy_barrier_source_candidates=lambda: [],
        fractional_position_block_from_structure=_neb_frac_block,
        build_energy_barrier_neb_states=lambda: {
            candidate.key: _State("ready")
        },
    )
    base.update(overrides)
    return _stub_energy_barriers_deps(**base)


def test_energy_barriers_controller_handles_both_ts_and_neb_calc_types(tmp_path: Path) -> None:
    """Both the NEB and transition-state-search calc types are owned so the
    ``.ts``-producing pair is coherent for result dispatch."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    controller = EnergyBarriersController(
        _make_context(document=document),
        _stub_energy_barriers_deps(),
        lambda: _FakeEnergyBarriersView(),
    )
    neb = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    tss = _neb_job(service, document, "barrier_tss", TRANSITION_STATE_SEARCH_CALC_TYPE)
    other = service.add_job(
        document, label="u", workflow_stage=WorkflowStage.MUON_RELAXATION, calc_type="castep_uep_source"
    )
    assert controller.handles_job(neb) is True
    assert controller.handles_job(tss) is True
    assert controller.handles_job(other) is False


def test_energy_barriers_controller_refresh_view_noops_when_view_never_created() -> None:
    """The sanctioned lazy-view deviation: when the getter returns ``None`` (the
    view was never opened), refresh_view is a strict no-op — it never touches any
    view. This is what lets refresh run safely before the workflow route opens."""

    calls: list[str] = []

    def _getter():
        calls.append("getter")
        return None

    controller = EnergyBarriersController(
        _make_context(document=None), _stub_energy_barriers_deps(), _getter
    )
    controller.refresh_view()  # must not raise, must not build any rows
    assert calls == ["getter"]  # getter consulted, then no-op


def test_energy_barriers_controller_refresh_view_clears_when_document_none() -> None:
    """With a real (lazily-created) view but no document, refresh clears all three
    tables — distinct from the getter-None no-op above."""

    view = _FakeEnergyBarriersView()
    controller = EnergyBarriersController(
        _make_context(document=None), _stub_energy_barriers_deps(), lambda: view
    )
    controller.refresh_view()
    assert view.endpoint_calls[-1][0] == []
    assert view.sources == [[]]
    assert view.neb_results == [[]]


def test_energy_barriers_controller_refresh_view_renders_sources_and_results(
    tmp_path: Path,
) -> None:
    """With a view and a document, refresh renders endpoint options, one source
    row, and one completed-NEB result row — exercising the full render body
    (source-row build + result-row build with barrier summaries / charge label)."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    job.status = JobStatus.COMPLETED

    @_dataclass
    class _Endpoint:
        site: object
        job: object

    @_dataclass
    class _Named:
        label: str

    @_dataclass
    class _Path:
        length_ang: float

    @_dataclass
    class _SourceCandidate:
        key: str
        reactant: object
        product: object
        path: object

    @_dataclass
    class _CompletedResult:
        data: dict

    @_dataclass
    class _CompletedCandidate:
        key: str
        job: object
        result: object

    source_candidate = _SourceCandidate(
        key="source-key",
        reactant=_Endpoint(site=_Named("siteA"), job=_Named("jobA")),
        product=_Endpoint(site=_Named("siteB"), job=_Named("jobB")),
        path=_Path(length_ang=2.5),
    )
    completed = _CompletedCandidate(
        key="result-key",
        job=job,
        result=_CompletedResult(
            data={
                "num_images": 7,
                "transition_state_file": "/tmp/barrier.ts",
                "image_energies_eV": [-1.0, -0.8, -0.9],
                "charge_state": "positive",
            }
        ),
    )

    view = _FakeEnergyBarriersView()
    deps = _stub_energy_barriers_deps(
        eligible_energy_barrier_source_candidates=lambda: [source_candidate],
        build_energy_barrier_neb_states=lambda: {"source-key": _State("ready")},
        completed_energy_barrier_neb_candidates=lambda: [completed],
        energy_barrier_neb_sources_by_job_id=lambda: {job.id: {"charge_state": "positive", "path_length_ang": 2.5}},
        energy_barrier_neb_result_path_label=lambda _c, _s: "siteA → siteB",
        neb_barrier_summaries=lambda _data: ("0.2 eV", "0.1 eV", "-0.1 eV"),
        muon_charge_state_label=lambda value: f"mu+({value})",
    )
    controller = EnergyBarriersController(
        _make_context(document=document, project_service=service), deps, lambda: view
    )

    controller.refresh_view()

    assert len(view.endpoint_calls) == 1
    assert len(view.sources[-1]) == 1
    assert view.sources[-1][0].reactant == "siteA"
    assert len(view.neb_results[-1]) == 1
    result_row = view.neb_results[-1][0]
    assert result_row.label == "siteA → siteB"
    assert result_row.images == "7"
    assert result_row.charge_state == "mu+(positive)"
    assert result_row.path_length == "2.5 Å"
    # The result row carries the producing job's status so the results table can colour it.
    assert result_row.status == "completed"


def test_energy_barriers_controller_build_result_ts_payload(tmp_path: Path) -> None:
    """build_result parses the completed ``.ts`` file and contributes the
    transition-state payload — the fields the former inline result block assigned.
    Uses the real ``neb_path.ts`` fixture so the science path matches the inline
    guard's parse."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    run = CalculationRun(job_id=job.id)

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    ts_path = output_dir / "barrier.ts"
    ts_path.write_text(_NEB_TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    deps = _stub_energy_barriers_deps()
    controller = EnergyBarriersController(
        _make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: output_dir,
            job_seed_name=lambda _job: "barrier",
        ),
        deps,
        lambda: None,  # view never created: result dispatch must still work
    )

    artifacts = collect_job_artifacts(output_dir, "barrier", want_ts=True)
    result = controller.build_result(job, _parsed(), run, artifacts)

    payload = result.payload
    assert payload["transition_state_file"] == str(ts_path)
    assert payload["num_images"] == 7
    assert payload["image_labels"] == ["REA", "TST", "TST", "TST", "TST", "TST", "PRO"]
    assert payload["barrier_height_eV"] == pytest.approx(0.1214, abs=2e-3)
    assert len(payload["reaction_coordinates"]) == 7
    assert len(payload["image_energies_eV"]) == 7
    # build_result contributes no output files or derived structures.
    assert result.output_files == []
    assert result.derived_structure_ids == []


def test_energy_barriers_controller_build_result_missing_dir_is_empty(tmp_path: Path) -> None:
    """If the job's output directory can't be resolved, build_result yields an
    empty result rather than crashing (the inline ``.ts`` guard fails such jobs
    first)."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    run = CalculationRun(job_id=job.id)
    deps = _stub_energy_barriers_deps()
    controller = EnergyBarriersController(
        _make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: None,
        ),
        deps,
        lambda: None,
    )
    assert controller.build_result(job, _parsed(), run, ParsedJobArtifacts()).payload == {}


def test_energy_barriers_controller_pre_submit_check_blocks_on_structure_mismatch(
    tmp_path: Path,
) -> None:
    """An unresolved structure host mismatch refuses the submission: the guard
    returns False, so no submission plan is built and nothing is dispatched."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    structure_id = _uuid4()

    class _StubService:
        def validate_structures_match_active_host(self, _doc, _ids):
            return object()  # a mismatch

        def validate_sites_match_active_host(self, _doc, _ids):
            return None

    candidate = _FakeNebCandidate(
        key="source-key",
        source={
            "reactant_structure_id": str(structure_id),
            "product_structure_id": str(structure_id),
        },
    )
    deps = _stub_energy_barriers_deps(
        energy_barrier_neb_source_candidates=lambda: [candidate],
        handle_structure_host_mismatch=lambda _mismatch: False,
    )
    local = _RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = _make_context(
        document=document, project_service=_StubService(), local_backend=local, infos=infos
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    assert controller.pre_submit_check(["source-key"]) is False
    controller.submit(["source-key"], target="local")
    assert local.calls == []  # nothing dispatched
    assert infos == []  # silent refusal


def test_energy_barriers_controller_pre_submit_check_passes_without_document() -> None:
    controller = EnergyBarriersController(
        _make_context(document=None), _stub_energy_barriers_deps(), lambda: None
    )
    assert controller.pre_submit_check(["source-key"]) is True


def _seed_energy_barriers_document(service: ProjectService, tmp_path: Path):
    """Seed a document with a host, two relaxed endpoints, and a NEB definition
    carrying one generated-path source keyed ``source-key``."""

    document = service.create_project(tmp_path / "eb_mat.mufinder", "EBMat")
    host = _Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    document.structures[host.id] = host
    document.project.host_structure_id = host.id

    relaxed_a = _Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )
    relaxed_b = _Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )
    document.structures[relaxed_a.id] = relaxed_a
    document.structures[relaxed_b.id] = relaxed_b

    site_a = _MuonSite(label="siteA", status="relaxed")
    site_b = _MuonSite(label="siteB", status="relaxed")
    document.sites[site_a.id] = site_a
    document.sites[site_b.id] = site_b

    source = {
        "key": "source-key",
        "label": "siteA_to_siteB_path1",
        "reactant_site_id": str(site_a.id),
        "product_site_id": str(site_b.id),
        "reactant_run_id": str(_uuid4()),
        "product_run_id": str(_uuid4()),
        "reactant_job_id": str(_uuid4()),
        "product_job_id": str(_uuid4()),
        "reactant_structure_id": str(relaxed_a.id),
        "product_structure_id": str(relaxed_b.id),
        "path_start_fractional": [0.25, 0.25, 0.25],
        "path_end_fractional": [0.75, 0.25, 0.25],
        "path_length_ang": 2.5,
    }
    output_root = document.project_root / "files" / "energy_barriers_neb"
    definition = _CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_nudged_elastic_band_set",
        parameters={
            "output_directory": "files/energy_barriers_neb",
            "job_ids_by_label": {},
            "configuration": {"method": "NEB", "sources": [source]},
        },
    )
    document.calculations[definition.id] = definition
    return document, definition, source, output_root


def test_energy_barriers_controller_materialize_source_key_provenance(tmp_path: Path) -> None:
    """materialize writes ``job_ids_by_label`` keyed by SOURCE KEY and stamps each
    spec with ``provenance={"source_key": key}`` — the NEB provenance contract.

    Uses the real science stack (``prepare_transition_state_structures`` /
    ``idpp_intermediate_structure`` / ``expand_nudged_elastic_band_jobs``) with
    only the MainWindow-side helpers stubbed via deps, so this pins the controller's
    materialize behavior in isolation."""

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)

    import dataclasses

    persisted: list[str] = []
    deps = _neb_materialize_deps(definition, candidate, output_root)
    context = dataclasses.replace(
        _make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        persist_ui_state=lambda **_: persisted.append("persist"),
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    jobs = controller.materialize(["source-key"])

    assert len(jobs) == 1
    created = jobs[0]
    assert created.calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE
    # job_ids_by_label is keyed by the SOURCE KEY, mapping to the new job id.
    assert definition.parameters["job_ids_by_label"] == {"source-key": str(created.id)}
    # Each expanded spec carried provenance={"source_key": key}; verify it round-trips
    # by re-reading the mapping keyed by that same source key.
    assert set(definition.parameters["job_ids_by_label"].keys()) == {"source-key"}
    assert persisted == ["persist"]


def test_energy_barriers_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    """The full submit path (pre_submit_check → plan → materialize → backend)
    dispatches the newly-created NEB job to the LOCAL backend — the flow the
    lazy-view submit lambda invokes."""

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(definition, candidate, output_root)
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    controller.submit(["source-key"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    assert len(local.calls[0][0]) == 1
    assert local.calls[0][0][0].calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE


def test_energy_barriers_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(definition, candidate, output_root)
    local = _RecordingBackend()
    cluster = _RecordingBackend()
    context = _make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    controller.submit(["source-key"], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE


def test_energy_barriers_controller_materialize_writes_num_farms(tmp_path: Path) -> None:
    """The per-row task-farm count reaches the generated ``.param`` as NUM_FARMS.

    The farm count is stored in the row's configuration extras by the NEB
    configure dialog (it is not a CASTEP keyword row), so it has to be read from
    extras at materialize time -- the point at which the job directory is
    written."""

    import dataclasses

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(
        definition,
        candidate,
        output_root,
        # materialize reads the row's extras from the same call that resolves the
        # inherited cut-off/k-points, which is what MainWindow wires in production.
        energy_barrier_neb_row_configuration_with_source_defaults=(
            lambda _definition, _key, **_kwargs: ({}, (), {"num_farms": 3})
        ),
    )
    context = dataclasses.replace(
        _make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        persist_ui_state=lambda **_: None,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    jobs = controller.materialize(["source-key"])

    assert len(jobs) == 1
    param_text = (output_root / jobs[0].label / f"{jobs[0].label}.param").read_text(
        encoding="utf-8"
    )
    assert "NUM_FARMS : 3" in param_text


def test_energy_barriers_controller_materialize_omits_num_farms_when_off(
    tmp_path: Path,
) -> None:
    import dataclasses

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(definition, candidate, output_root)
    context = dataclasses.replace(
        _make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        persist_ui_state=lambda **_: None,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    jobs = controller.materialize(["source-key"])

    param_text = (output_root / jobs[0].label / f"{jobs[0].label}.param").read_text(
        encoding="utf-8"
    )
    assert "NUM_FARMS" not in param_text


def test_energy_barriers_controller_materialize_ignores_num_farms_parameter_row(
    tmp_path: Path,
) -> None:
    """A persisted ``num_farms`` keyword row must not override the row's extras.

    The extras value is what ``_task_farming_for_job`` reports to the runners
    (deciding whether to read ``<seed>_farm001.*``) and what submission
    validates the rank count against. If a keyword row could win in the rendered
    .param, CASTEP would farm one way while MuFinder assumed another -- so the
    dedicated control stays authoritative and the keyword row is dropped."""

    import dataclasses

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(
        definition,
        candidate,
        output_root,
        energy_barrier_neb_row_configuration_with_source_defaults=(
            lambda _definition, _key, **_kwargs: (
                {"num_farms": 7},
                (),
                {"num_farms": 3},
            )
        ),
    )
    context = dataclasses.replace(
        _make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        persist_ui_state=lambda **_: None,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    jobs = controller.materialize(["source-key"])

    param_text = (output_root / jobs[0].label / f"{jobs[0].label}.param").read_text(
        encoding="utf-8"
    )
    assert "NUM_FARMS : 3" in param_text
    assert "NUM_FARMS : 7" not in param_text


def _stage_two_target_controller(
    service: ProjectService,
    tmp_path: Path,
    *,
    stage_one_targets: list[str],
    infos: list[tuple[str, str]],
    local: _RecordingBackend | None = None,
    cluster: _RecordingBackend | None = None,
):
    """Build a controller whose rows are all ready for stage 2.

    Row ``i`` (sites ``siteA``, ``siteB``) has a completed stage-1 singlepoint
    that ran on ``stage_one_targets[i]``. Materialize is stubbed to hand back one
    defined phonon job per row, so an allowed submission reaches the backend.
    """

    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    keys = list(candidates)[: len(stage_one_targets)]
    singlepoints: dict[str, Job] = {}
    for key, stage_one in zip(keys, stage_one_targets, strict=True):
        singlepoint = _quantum_job(service, document, f"quantum_singlepoint_{key}", QUANTUM_SINGLEPOINT_CALC_TYPE)
        singlepoint.status = JobStatus.COMPLETED
        singlepoint.execution_target = stage_one
        singlepoints[key] = singlepoint
    ready = _FakeQuantumStageState(stage=QUANTUM_STAGE_PHONON, status="ready")
    deps = _stub_quantum_deps(
        quantum_source_candidate_by_key=lambda key: candidates.get(key),
        quantum_stage_state=lambda _definition, _candidate: ready,
        build_quantum_corrections_states=lambda: {key: ready for key in keys},
        singlepoint_job_for_quantum_source_key=lambda _definition, key: singlepoints.get(key),
    )
    context = _make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
        local_backend=local,
        cluster_backend=cluster,
        infos=infos,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())
    phonon_jobs = {
        key: _quantum_job(service, document, f"quantum_phonon_{key}", QUANTUM_PHONON_CALC_TYPE) for key in keys
    }
    controller.materialize = lambda values: [phonon_jobs[str(value)] for value in values]  # type: ignore[assignment]
    return controller, keys


def test_quantum_controller_refuses_local_stage_two_after_a_cluster_singlepoint(tmp_path: Path) -> None:
    """The stage-1 checkpoint is never downloaded, so a local continuation would fail."""

    infos: list[tuple[str, str]] = []
    local = _RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["archer2"], infos=infos, local=local
    )

    controller.submit(keys, target="local")

    assert local.calls == []
    assert len(infos) == 1
    title, message = infos[0]
    assert title == "Phonon stage must run where stage 1 ran"
    assert "siteA (stage 1 ran on archer2)" in message


def test_quantum_controller_refuses_cluster_stage_two_after_a_local_singlepoint(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    cluster = _RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["local"], infos=infos, cluster=cluster
    )

    controller.submit(keys, target="cluster")

    assert cluster.calls == []
    assert len(infos) == 1
    assert "ran on this computer" in infos[0][1]
    assert "siteA" in infos[0][1]


def test_quantum_controller_refuses_stage_two_rows_from_different_clusters(tmp_path: Path) -> None:
    """One cluster submission cannot reach two checkpoints on two clusters."""

    infos: list[tuple[str, str]] = []
    cluster = _RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["archer2", "young"], infos=infos, cluster=cluster
    )

    controller.submit(keys, target="cluster")

    assert cluster.calls == []
    assert infos[0][0] == "Submit each cluster's rows separately"
    assert "archer2: siteA" in infos[0][1]
    assert "young: siteB" in infos[0][1]


def test_quantum_controller_preselects_the_stage_one_cluster_for_stage_two(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    cluster = _RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["archer2", "archer2"], infos=infos, cluster=cluster
    )

    controller.submit(keys, target="cluster")

    assert infos == []
    assert len(cluster.calls) == 1
    jobs, kwargs = cluster.calls[0]
    assert len(jobs) == 2
    assert kwargs["default_cluster_name"] == "archer2"
    # The preselection does not leak into the next submission.
    assert controller.submission_groups(jobs) == [(jobs, {})]


def test_quantum_controller_runs_local_stage_two_after_a_local_singlepoint(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    local = _RecordingBackend()
    controller, keys = _stage_two_target_controller(
        ProjectService(), tmp_path, stage_one_targets=["local"], infos=infos, local=local
    )

    controller.submit(keys, target="local")

    assert infos == []
    assert len(local.calls) == 1
    assert local.calls[0][1] == {}


_GUI_TO_METHOD = {
    "FINITEDISPLACEMENT": "finite_displacement",
    "DFPT": "dfpt",
    "LINEARRESPONSE": "linear_response",
}


def _phonon_method_controller(
    service: ProjectService,
    tmp_path: Path,
    *,
    rows: dict[str, tuple[str, str, str]],
    infos: list[tuple[str, str]],
    questions: list[tuple[str, str]],
    answer: bool,
    local: _RecordingBackend,
):
    """Build a controller whose rows ``siteA``/``siteB`` are ready to submit.

    ``rows`` maps a site label to ``(stage, GUI phonon method, pseudopotential
    family)``; materialize hands back one defined job per row so an allowed
    submission reaches the local backend.
    """

    document, definition, candidates, output_root = _seed_quantum_two_stage_document(service, tmp_path)
    by_label = {candidate.site.label: key for key, candidate in candidates.items()}
    config = {by_label[label]: spec for label, spec in rows.items()}
    keys = list(config)
    deps = _stub_quantum_deps(
        quantum_source_candidate_by_key=lambda key: candidates.get(key),
        quantum_stage_state=lambda _definition, candidate: _FakeQuantumStageState(
            stage=config[candidate.key][0], status="ready"
        ),
        build_quantum_corrections_states=lambda: {
            key: _FakeQuantumStageState(stage=config[key][0], status="ready") for key in keys
        },
        quantum_runtime_configuration=lambda _definition, key: (
            {"phonon_method": config[key][1]},
            config[key][2],
        ),
        quantum_method_from_gui_value=lambda value: _GUI_TO_METHOD[str(value).upper()],
        confirm=lambda title, text: questions.append((title, text)) or answer,
    )
    context = _make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
        local_backend=local,
        infos=infos,
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())
    jobs = {key: _quantum_job(service, document, f"quantum_{key}", QUANTUM_SINGLEPOINT_CALC_TYPE) for key in keys}
    controller.materialize = lambda values: [jobs[str(value)] for value in values]  # type: ignore[assignment]
    return controller, keys


def test_quantum_controller_refuses_dfpt_with_ultrasoft_pseudopotentials_by_name(tmp_path: Path) -> None:
    """The refusal names the rows instead of ending in a generic "No ready jobs"."""

    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = _RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={
            "siteA": (QUANTUM_STAGE_SINGLEPOINT, "DFPT", "ultrasoft"),
            "siteB": (QUANTUM_STAGE_PHONON, "LINEARRESPONSE", "ultrasoft"),
        },
        infos=infos,
        questions=questions,
        answer=True,
        local=local,
    )

    controller.submit(keys, target="local")

    assert local.calls == []
    assert questions == []
    assert len(infos) == 1
    title, message = infos[0]
    assert title == "Phonon method needs norm-conserving pseudopotentials"
    assert "siteA (DFPT, stage 1)" in message
    assert "siteB (LINEARRESPONSE, stage 2)" in message
    assert "No ready jobs" not in title


def test_quantum_controller_asks_before_dfpt_with_an_unknown_pseudopotential_family(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = _RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={"siteA": (QUANTUM_STAGE_SINGLEPOINT, "DFPT", "unknown")},
        infos=infos,
        questions=questions,
        answer=False,
        local=local,
    )

    controller.submit(keys, target="local")

    assert local.calls == []
    assert infos == []
    assert len(questions) == 1
    title, message = questions[0]
    assert title == "Check pseudopotentials for DFPT"
    assert "siteA (DFPT, stage 1)" in message
    assert "norm-conserving" in message


def test_quantum_controller_submits_dfpt_with_an_unknown_family_once_confirmed(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = _RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={"siteA": (QUANTUM_STAGE_SINGLEPOINT, "DFPT", "unknown")},
        infos=infos,
        questions=questions,
        answer=True,
        local=local,
    )

    controller.submit(keys, target="local")

    assert len(questions) == 1
    assert len(local.calls) == 1


def test_quantum_controller_submits_finite_displacement_without_asking(tmp_path: Path) -> None:
    infos: list[tuple[str, str]] = []
    questions: list[tuple[str, str]] = []
    local = _RecordingBackend()
    controller, keys = _phonon_method_controller(
        ProjectService(),
        tmp_path,
        rows={
            "siteA": (QUANTUM_STAGE_SINGLEPOINT, "FINITEDISPLACEMENT", "ultrasoft"),
            "siteB": (QUANTUM_STAGE_SINGLEPOINT, "DFPT", "norm_conserving"),
        },
        infos=infos,
        questions=questions,
        answer=False,
        local=local,
    )

    controller.submit(keys, target="local")

    assert infos == []
    assert questions == []
    assert len(local.calls) == 1
