"""Submitting a job never writes into a folder another job's records use.

Each test builds a scratch project and hashes its tree before and after, through the
controller's own ``submit``, so the plan, ``materialize`` and the submit backend all run.
The local submission dialog is cancelled, which is where a job would start running.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
from mufinder.app.workflow_config_codecs import serialize_pristine_relaxation_config
from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job, utc_now

from tests.app.test_job_folder_removal_guards import _Project, host  # noqa: F401
from tests.app.test_relaxation_import_wizard import _document

ROW_KEY = "positivemuon:Site_native"
SEED = "positivemuon_Site_native"


def _hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class _Submission(_Project):
    """A scratch project whose submissions stop at the local dialog, recording who got there."""

    def __init__(self, window: MainWindow, tmp_path: Path, host, monkeypatch) -> None:  # noqa: F811
        super().__init__(window, tmp_path, host, monkeypatch)
        self.controller = window._muon_relaxation_controller
        self.started: list[str] = []
        self.infos: list[str] = []
        project = self

        class _CancelledDialog:
            class DialogCode:
                Accepted = 1

            def __init__(self, *, job_labels, **kwargs) -> None:
                project.started.extend(job_labels)

            def exec(self) -> int:
                return 0

        monkeypatch.setattr("mufinder.app.main_window.LocalSubmissionDialog", _CancelledDialog)
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information",
            lambda *args, **kwargs: self.infos.append(args[2]),
        )
        monkeypatch.setattr(window, "_castep_executable_ready", lambda: True)
        monkeypatch.setattr(window._keyword_catalog_guard, "confirm_submission", lambda: True)

    def add_row(self, label: str = "Site_native") -> None:
        rows = self.controller.active_definition().parameters["configuration"]["rows"]
        site_id = str(uuid4())
        normalized = self.window._mirror_imported_rows_into_form(
            [
                {
                    **rows[0],
                    "label": label,
                    "imported": False,
                    "import_batch_id": None,
                    "site_id": site_id,
                }
            ]
        )
        rows.append(normalized[site_id])

    def job(self, key: str = ROW_KEY) -> Job:
        mapping = self.controller.active_definition().parameters["job_ids_by_label"]
        return self.document.jobs[UUID(mapping[key])]

    def seed(self, key: str = ROW_KEY) -> str:
        return self.controller.active_definition().parameters["job_seed_names_by_label"][key]

    def submit(self, *keys: str) -> None:
        self.controller.submit(list(keys or (ROW_KEY,)), target="local")

    def run_from_page(self, *keys: str) -> None:
        """Submit as the page's Run does: the form's configuration first, then the rows."""

        self.window._on_muon_relaxation_configured(self.window._muon_relaxation_form_config())
        self.submit(*keys)

    def native_jobs(self, label: str = "Site_native") -> list[Job]:
        return [
            job
            for job in self.document.jobs.values()
            if job.label == label and IMPORTED_JOB_TAG not in (job.tags or ())
        ]


@pytest.fixture()
def project(main_window: MainWindow, tmp_path: Path, host, monkeypatch) -> _Submission:  # noqa: F811
    return _Submission(main_window, tmp_path, host, monkeypatch)


class TestAnImportsFolder:
    def test_a_row_whose_seed_is_an_imports_gets_a_folder_of_its_own(
        self, project: _Submission
    ) -> None:
        item = project.import_batch("runs", (1,), seed=SEED).imported[0]
        project.add_row()
        before = _hashes(item.job_directory)

        project.submit()

        assert _hashes(item.job_directory) == before
        job = project.job()
        assert project.seed() == f"{SEED}_2"
        assert project.window._job_output_directory(job) == project.charge / f"{SEED}_2"
        assert project.window._job_seed_name(job) == f"{SEED}_2"
        assert {path.name for path in (project.charge / f"{SEED}_2").iterdir()} == {
            f"{SEED}_2.cell",
            f"{SEED}_2.param",
        }
        assert project.started == [job.label]

    def test_a_native_job_that_came_first_keeps_its_folder_and_the_import_gets_another(
        self, project: _Submission
    ) -> None:
        project.import_batch("runs", (1,), seed="first")
        project.add_row()
        project.submit()
        assert project.seed() == SEED

        item = project.import_batch("more", (2,), seed=SEED).imported[0]
        assert item.seed_name == f"{SEED}_2"
        imported = _hashes(item.job_directory)
        native_cell = project.charge / SEED / f"{SEED}.cell"
        native_cell.write_text("stale inputs")

        project.submit()

        assert _hashes(item.job_directory) == imported
        assert project.seed() == SEED
        assert "stale inputs" not in native_cell.read_text()

    def test_a_row_in_its_free_folder_stays_there_when_submitted_again(
        self, project: _Submission
    ) -> None:
        outcome = project.import_batch("runs", (1,), seed=SEED)
        item = outcome.imported[0]
        project.add_row()
        project.submit()
        before = _hashes(item.job_directory)
        cell = project.charge / f"{SEED}_2" / f"{SEED}_2.cell"
        cell.write_text("stale inputs")

        project.submit()

        assert "stale inputs" not in cell.read_text()
        assert _hashes(item.job_directory) == before
        assert project.seed() == f"{SEED}_2"
        assert len(project.document.jobs) == 2

    def test_the_new_folder_goes_with_the_row_and_the_import_stays(
        self, project: _Submission
    ) -> None:
        item = project.import_batch("runs", (1,), seed=SEED).imported[0]
        project.add_row()
        project.submit()
        job = project.job()
        before = _hashes(item.job_directory)

        project.remove_row("Site_native")

        assert not (project.charge / f"{SEED}_2").exists()
        assert _hashes(item.job_directory) == before
        assert job.id not in project.document.jobs
        assert item.job_id in project.document.jobs

    def test_a_row_of_the_other_charge_state_is_not_affected(self, project: _Submission) -> None:
        item = project.import_batch("runs", (1,), seed="muonium_Site_native").imported[0]
        project.add_row()

        project.submit()

        assert project.seed() == SEED
        assert item.job_directory.parent == project.charge


class TestRunFromThePage:
    """The page's Run configures the form again before it submits, as a new definition."""

    def test_a_defined_row_run_again_keeps_its_job_and_free_folder(
        self, project: _Submission
    ) -> None:
        item = project.import_batch("runs", (1,), seed=SEED).imported[0]
        project.add_row()
        before = _hashes(item.job_directory)
        project.run_from_page()
        (job,) = project.native_jobs()

        project.run_from_page()
        project.run_from_page()

        assert project.native_jobs() == [job]
        assert project.job().id == job.id
        assert project.seed() == f"{SEED}_2"
        assert sorted(path.name for path in project.charge.iterdir()) == [SEED, f"{SEED}_2"]
        assert _hashes(item.job_directory) == before
        assert project.started == ["Site_native"] * 3

    def test_a_row_with_no_clash_keeps_its_job_and_plain_folder(self, project: _Submission) -> None:
        project.import_batch("runs", (1,), seed="first")
        project.add_row()
        project.run_from_page()
        (job,) = project.native_jobs()

        project.run_from_page()

        assert project.native_jobs() == [job]
        assert project.seed() == SEED
        assert not (project.charge / f"{SEED}_2").exists()

    def test_removing_a_row_run_twice_asks_about_one_job(self, project: _Submission) -> None:
        project.import_batch("runs", (1,), seed=SEED)
        project.add_row()
        project.run_from_page()
        project.run_from_page()

        project.remove_row("Site_native")

        assert len(project.asked) == 1
        assert project.native_jobs() == []
        assert not (project.charge / f"{SEED}_2").exists()

    def test_the_page_runs_its_rows_through_the_same_job(self, project: _Submission) -> None:
        project.import_batch("runs", (1,), seed=SEED)
        project.add_row()
        view = project.window._muon_relaxation_view

        view._run_local_selected_rows()
        view._run_local_selected_rows()

        assert len(project.native_jobs()) == 1
        assert project.seed() == f"{SEED}_2"
        assert not (project.charge / f"{SEED}_3").exists()

    def test_a_defined_row_in_an_imports_folder_is_refused(self, project: _Submission) -> None:
        item = project.import_batch("runs", (1,), seed=SEED).imported[0]
        project.add_row()
        job = project.native(label="Site_native", seed=SEED)
        job.status = JobStatus.DEFINED
        before = _hashes(project.files)

        project.run_from_page()

        assert _hashes(project.files) == before
        assert project.native_jobs() == [job]
        assert project.job().id == job.id
        assert project.started == []
        assert job.status == JobStatus.DEFINED
        assert project.window._job_output_directory(job) == item.job_directory
        assert len(project.warned) == 1
        assert (
            "'Site_native' was not started because another job uses its folder."
            in (project.warned[0])
        )

    def test_only_the_row_in_an_imports_folder_is_refused(self, project: _Submission) -> None:
        project.import_batch("runs", (1,), seed=SEED)
        project.add_row()
        job = project.native(label="Site_native", seed=SEED)
        job.status = JobStatus.DEFINED
        project.add_row("Site_other")

        project.run_from_page(ROW_KEY, "positivemuon:Site_other")

        assert project.started == ["Site_other"]
        assert len(project.warned) == 1
        assert "'Site_native'" in project.warned[0]
        assert project.native_jobs() == [job]


class TestTwoNativeRows:
    def test_a_row_whose_seed_another_row_just_took_as_its_free_name_gets_the_next(
        self, project: _Submission
    ) -> None:
        project.import_batch("runs", (1,), seed=SEED)
        project.add_row()
        project.add_row("Site_native_2")
        other = "positivemuon:Site_native_2"

        project.submit(ROW_KEY, other)

        assert project.seed() == f"{SEED}_2"
        assert project.seed(other) == f"{SEED}_2_2"
        for seed in (project.seed(), project.seed(other)):
            assert (project.charge / seed / f"{seed}.cell").is_file()

    def test_a_reconfigured_row_stays_in_its_folder(self, project: _Submission) -> None:
        project.import_batch("runs", (1,), seed="first")
        project.add_row()
        project.submit()
        old = project.job()
        config = project.window._muon_relaxation_form_config()
        config = replace(
            config,
            rows=tuple(
                replace(row, initial_fractional_position=(0.3, 0.3, 0.3))
                if row.label == "Site_native"
                else row
                for row in config.rows
            ),
        )

        project.window._on_muon_relaxation_configured(config)
        project.submit()

        assert project.job().id != old.id
        assert project.seed() == SEED
        assert project.window._job_output_directory(project.job()) == project.charge / SEED


class TestAProjectAlreadyHoldingBothInOneFolder:
    """Nothing saved changes and nothing is written: the job is refused and the user decides."""

    def _shared(self, project: _Submission, status: JobStatus = JobStatus.DEFINED):
        item = project.import_batch("runs", (1,), seed=SEED).imported[0]
        project.add_row()
        job = project.native(label="Site_native", seed=SEED)
        job.status = status
        return item, job

    def test_a_defined_job_is_not_written_or_started(self, project: _Submission) -> None:
        item, job = self._shared(project)
        before = _hashes(project.files)

        project.submit()

        assert _hashes(project.files) == before
        assert project.started == []
        assert project.seed() == SEED
        assert job.status == JobStatus.DEFINED
        assert project.window._job_output_directory(job) == item.job_directory
        assert (
            "'Site_native' was not started because another job uses its folder."
            in (project.warned[0])
        )
        assert "muon_relaxation/positivemuon/" + SEED in project.warned[0]

    def test_a_finished_job_is_left_alone(self, project: _Submission) -> None:
        _item, job = self._shared(project, JobStatus.COMPLETED)
        before = _hashes(project.files)

        project.submit()

        assert _hashes(project.files) == before
        assert project.started == []
        assert job.status == JobStatus.COMPLETED

    def test_a_dismissed_failed_job_is_not_written_or_started(
        self, project: _Submission, monkeypatch
    ) -> None:
        _item, job = self._shared(project, JobStatus.FAILED)
        job.dismissed_at = utc_now()
        before = _hashes(project.files)

        project.submit()

        assert _hashes(project.files) == before
        assert project.started == []
        assert job.status == JobStatus.FAILED
        assert project.asked == []
        assert len(project.warned) == 1

    def test_continue_relaxation_is_refused_before_anything_changes(
        self, project: _Submission, monkeypatch
    ) -> None:
        _item, job = self._shared(project, JobStatus.FAILED)
        before = _hashes(project.files)
        monkeypatch.setattr(
            project.controller,
            "continue_job",
            lambda *args, **kwargs: pytest.fail("continued a refused job"),
        )

        project.window._continue_relaxation_row(
            controller=project.controller,
            view=project.window._muon_relaxation_view,
            key=ROW_KEY,
            autosave_failure_message="",
        )

        assert _hashes(project.files) == before
        assert job.status == JobStatus.FAILED
        assert project.seed() == SEED
        assert len(project.warned) == 1

    def test_a_defined_job_is_not_sent_to_a_cluster(
        self, project: _Submission, monkeypatch
    ) -> None:
        _item, _job = self._shared(project)
        before = _hashes(project.files)
        asked: list[object] = []
        monkeypatch.setattr(
            project.window,
            "_choose_cluster_submission",
            lambda *args, **kwargs: asked.append(args),
        )

        project.controller.submit([ROW_KEY], target="cluster")

        assert _hashes(project.files) == before
        assert asked == []
        assert project.warned

    def test_dashboard_resubmit_refuses_it_with_the_same_box(self, project: _Submission) -> None:
        _item, job = self._shared(project, JobStatus.CANCELLED)
        job.execution_target = "local"
        before = _hashes(project.files)

        project.window._on_resubmit_from_dashboard([job.id])

        assert _hashes(project.files) == before
        assert project.started == []
        assert job.status == JobStatus.CANCELLED
        assert len(project.warned) == 1
        assert (
            "'Site_native' was not started because another job uses its folder."
            in (project.warned[0])
        )

    def test_the_dashboard_refuses_it_too(self, project: _Submission) -> None:
        _item, job = self._shared(project)
        before = _hashes(project.files)

        project.window._on_submit_local_from_dashboard([job.id])

        assert _hashes(project.files) == before
        assert project.started == []
        assert project.warned

    def test_the_other_rows_of_a_submission_still_start(self, project: _Submission) -> None:
        _item, _job = self._shared(project)
        project.add_row("Site_other")
        other = "positivemuon:Site_other"

        project.submit(ROW_KEY, other)

        assert project.started == ["Site_other"]
        assert len(project.warned) == 1

    def test_removing_and_adding_the_row_gives_it_a_folder_of_its_own(
        self, project: _Submission
    ) -> None:
        item, _job = self._shared(project)
        before = _hashes(item.job_directory)

        project.remove_row("Site_native")
        project.add_row()
        project.submit()

        assert _hashes(item.job_directory) == before
        assert project.seed() == f"{SEED}_2"
        assert project.started == ["Site_native"]


class TestPristineRelaxation:
    def test_a_row_whose_folder_another_job_uses_gets_its_own(
        self,
        main_window: MainWindow,
        tmp_path: Path,
        host,  # noqa: F811
    ) -> None:
        window = main_window
        window._document = _document(tmp_path, host)
        document = window._document
        output_root = document.project_root / "files" / "pristine_relaxation"
        controller = window._pristine_relaxation_controller
        config = PristineRelaxationConfig(
            name_prefix="pristine", rows=(PristineRelaxationRow(label="Host"),)
        )
        definition = window._project_service.register_pristine_relaxation_definition(
            document,
            input_structure_id=host.id,
            parameters=serialize_pristine_relaxation_config(config),
            output_directory=output_root,
        )
        document.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
        other = window._project_service.add_job(
            document,
            label="pristine_Host",
            workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
            calc_type="castep_pristine_relaxation",
        )
        folder = output_root / other.label
        folder.mkdir(parents=True)
        (folder / f"{other.label}.cell").write_text("the other job's cell")
        before = _hashes(folder)

        (job,) = controller.materialize(["Host"])
        (again,) = controller.materialize(["Host"])

        assert _hashes(folder) == before
        assert job.id == again.id
        assert job.label == "pristine_Host_2"
        assert (output_root / job.label / f"{job.label}.cell").is_file()
