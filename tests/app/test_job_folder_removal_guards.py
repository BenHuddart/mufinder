"""Removing or cleaning a job never takes a folder another job needs, on any route.

Each test builds a scratch project and lists its tree before and after, through the
window's own handlers: batch Remove, native row removal, cutoff removal and the
clearing of a job's previous output when it starts again.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.app import workflow_table_utils
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import EndReason, JobStatus, WorkflowStage
from mufinder.domain.models import Job, Structure
from mufinder.domain.text import KeptFolder, folder_outcome_status
from mufinder.services.journal import build_journal
from mufinder.services.project_service import ProjectService
from mufinder.services.relaxation_import_commit import commit_relaxation_import
from mufinder.services.relaxation_import_removal import import_batches

from tests.app.test_relaxation_import_wizard import _document, _scan, _tree
from tests.services.test_relaxation_import_scan import HOST_LATTICE, build_folder


@pytest.fixture()
def host() -> Structure:
    return Structure(
        label="Host",
        origin="imported",
        lattice_vectors=[list(row) for row in HOST_LATTICE],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


class _Project:
    """A scratch project with one import batch, driven through the window's handlers."""

    def __init__(self, window: MainWindow, tmp_path: Path, host: Structure, monkeypatch) -> None:
        self.window = window
        self.tmp_path = tmp_path
        self.host = host
        self.monkeypatch = monkeypatch
        window._document = _document(tmp_path, host)
        self.document = window._document
        self.files = self.document.project_root / "files"
        self.output_root = self.files / "muon_relaxation"
        self.charge = self.output_root / "positivemuon"
        self.batches: list = []
        self.warned: list[str] = []
        self.asked: list[str] = []
        self.boxes: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.warning",
            lambda *args, **kwargs: self.warned.append(args[2]),
        )
        monkeypatch.setattr(
            "mufinder.app.main_window.show_detailed_warning",
            lambda _parent, title, text, **kwargs: (
                self.boxes if title == "Folders kept" else self.warned
            ).append(f"{text}\n{kwargs.get('details', '')}"),
        )
        monkeypatch.setattr(
            "mufinder.app.jobs_controller.show_detailed_warning",
            lambda _parent, title, text, **kwargs: (
                self.boxes if title == "Folders kept" else self.warned
            ).append(f"{text}\n{kwargs.get('details', '')}"),
        )
        monkeypatch.setattr(
            "mufinder.app.workflow_controllers.imported_relaxations.show_detailed_warning",
            lambda _parent, title, text, **kwargs: (
                self.boxes if title == "Folders kept" else self.warned
            ).append(f"{text}\n{kwargs.get('details', '')}"),
        )
        monkeypatch.setattr(
            workflow_table_utils, "confirm_destructive", lambda *args, **kwargs: True
        )
        monkeypatch.setattr(window, "_persist_document_ui_state", lambda **kwargs: None)
        monkeypatch.setattr(window, "_refresh_muon_sites_summary", lambda **kwargs: None)

        def _delete(*, title, text, keep_text, include_apply_all=False):
            self.asked.append(text)
            return "delete"

        monkeypatch.setattr(window._jobs, "prompt_job_file_disposition", _delete)

    def import_batch(self, name: str, sites: tuple[int, ...], seed: str | None = None):
        scan = _scan(build_folder(self.tmp_path / name, sites=sites))
        runs = list(scan.importable_runs)
        if seed is not None:
            runs = [replace(runs[0], seed_name=seed)]
        outcome = commit_relaxation_import(
            self.document,
            scan=scan,
            runs=runs,
            host_structure=self.host,
            output_root=self.output_root,
            existing_labels=[item.label for done in self.batches for item in done.imported],
        )
        self.window._imported_relaxations_controller.merge_imported_relaxation_rows(
            outcome, output_directory=self.output_root
        )
        self.window._project_service.append_provenance(self.document, outcome.provenance)
        self.batches.append(outcome)
        return outcome

    def native(
        self, label: str = "native", seed: str | None = "own", charge: str = "positivemuon"
    ) -> Job:
        job = Job(
            label=label,
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type=f"castep_{charge}_relaxation",
            status=JobStatus.COMPLETED,
            execution_target="local",
        )
        self.document.jobs[job.id] = job
        definition = self.window._muon_relaxation_controller.active_definition()
        definition.parameters["job_ids_by_label"][f"{charge}:{label}"] = str(job.id)
        if seed is not None:
            definition.parameters["job_seed_names_by_label"][f"{charge}:{label}"] = seed
        return job

    def remove_row(self, label: str = "native") -> None:
        self.window._on_remove_muon_relaxation_rows([f"positivemuon:{label}"])

    def remove_batch(self, outcome) -> None:
        index = [
            position
            for position, batch in enumerate(import_batches(self.document))
            if batch[2] == outcome.batch_id
        ][0]
        self.monkeypatch.setattr(
            "mufinder.app.workflow_controllers.imported_relaxations.choose_import_batch",
            lambda *args, **kwargs: index,
        )
        self.window._imported_relaxations_controller.remove_imported_relaxations()

    def point_at(self, item, seed: str) -> None:
        """Make an imported job read as a saved project with this seed would."""

        definition = self.window._muon_relaxation_controller.active_definition()
        definition.parameters["job_seed_names_by_label"][f"positivemuon:{item.label}"] = seed
        site = self.document.sites[item.site_id]
        result = self.document.results[self.document.runs[site.relaxation_run_id].result_id]
        result.output_files = [
            str(Path(Path(seed).name) / Path(name).name) for name in result.output_files
        ]

    def status(self) -> str:
        return self.window.statusBar().currentMessage()

    def journal_bodies(self) -> list[str]:
        journal = build_journal("Test", self.document.provenance)
        return [entry.body for day in journal.days for entry in day.entries]


@pytest.fixture()
def project(main_window: MainWindow, tmp_path: Path, host: Structure, monkeypatch) -> _Project:
    return _Project(main_window, tmp_path, host, monkeypatch)


class TestNativeRowInASharedFolder:
    """The row goes with its records; the folder stays, because it is not the row's alone."""

    def test_a_saved_seed_naming_an_imports_folder_keeps_the_folder(
        self, project: _Project
    ) -> None:
        item = project.import_batch("runs", (1,)).imported[0]
        job = project.native(seed=item.seed_name)
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs
        assert project.asked == []
        assert project.warned == []
        assert item.job_id in project.document.jobs
        assert project.status() == "Removed 1 job; kept 1 folder"
        assert len(project.boxes) == 1
        assert "because another job uses it." in project.boxes[0]
        assert f"positivemuon/{item.seed_name}'" in project.boxes[0]

    def test_no_saved_seed_and_a_default_name_matching_an_import_keeps_the_folder(
        self, project: _Project
    ) -> None:
        """The job an older project might hold: its folder is inferred from its label."""

        item = project.import_batch("runs", (1,), seed="positivemuon_Site_native").imported[0]
        job = project.native(label="Site_native", seed=None)
        assert project.window._job_output_directory(job) == item.job_directory
        before = _tree(project.files)

        project.remove_row("Site_native")

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs
        assert item.job_id in project.document.jobs
        assert project.status() == "Removed 1 job; kept 1 folder"

    def test_a_normal_submit_never_lands_in_an_imports_folder(self, project: _Project) -> None:
        """A native row whose default seed is an import's seed, through the real submit code."""

        item = project.import_batch("runs", (1,), seed="positivemuon_Site_native").imported[0]
        controller = project.window._muon_relaxation_controller
        definition = controller.active_definition()
        rows = definition.parameters["configuration"]["rows"]
        rows.append(
            {
                **rows[0],
                "label": "Site_native",
                "imported": False,
                "import_batch_id": None,
                "site_id": str(uuid4()),
            }
        )
        job = controller.materialize(["positivemuon:Site_native"])[0]
        own_folder = project.window._job_output_directory(job)
        assert own_folder == project.charge / "positivemuon_Site_native_2"
        before = _tree(item.job_directory)

        project.remove_row("Site_native")

        assert _tree(item.job_directory) == before
        assert not own_folder.exists()
        assert job.id not in project.document.jobs
        assert item.job_id in project.document.jobs

    def test_the_import_can_then_go_and_takes_its_folder(self, project: _Project) -> None:
        """Neither side is left waiting for the other."""

        outcome = project.import_batch("runs", (1,))
        item = outcome.imported[0]
        project.native(seed=item.seed_name)

        project.remove_batch(outcome)
        assert item.job_id in project.document.jobs
        assert item.job_directory.is_dir()
        assert project.warned and "nothing was removed" in project.warned[0]
        project.warned.clear()

        project.remove_row()
        project.remove_batch(outcome)

        assert item.job_id not in project.document.jobs
        assert not item.job_directory.exists()
        assert project.warned == []

    def test_an_absolute_seed_into_another_jobs_folder_in_another_workflow_keeps_it(
        self, project: _Project
    ) -> None:
        project.import_batch("runs", (1,))
        service = project.window._project_service
        other = service.add_job(
            project.document,
            label="uep_job",
            workflow_stage=WorkflowStage.UEP,
            calc_type="castep_uep",
        )
        folder = project.files / "uep" / other.label
        folder.mkdir(parents=True)
        (folder / "uep_job.castep").write_text("kept")
        job = project.native(seed=str(folder))
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs
        assert other.id in project.document.jobs

    def test_an_absolute_seed_into_another_workflows_unowned_folder_keeps_it(
        self, project: _Project
    ) -> None:
        project.import_batch("runs", (1,))
        folder = project.files / "uep" / "not_a_job"
        folder.mkdir(parents=True)
        (folder / "notes.txt").write_text("kept")
        job = project.native(seed=str(folder))
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs
        assert project.status() == "Removed 1 job; kept 1 folder"
        assert "'files/uep/not_a_job'" in project.boxes[0]
        assert "another workflow" in project.boxes[0]

    def test_a_seed_inside_an_imports_folder_keeps_it(self, project: _Project) -> None:
        item = project.import_batch("runs", (1,)).imported[0]
        inner = item.job_directory / "original_inputs"
        assert inner.is_dir()
        job = project.native(seed=f"{item.seed_name}/original_inputs")
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs


class TestNativeRowInItsOwnFolder:
    def test_its_own_folder_still_deletes(self, project: _Project) -> None:
        item = project.import_batch("runs", (1,)).imported[0]
        job = project.native(seed="own")
        (project.charge / "own").mkdir()
        (project.charge / "own" / "own.castep").write_text("gone")
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == [
            path
            for path in before
            if path.split("/")[:3] != ["muon_relaxation", "positivemuon", "own"]
        ]
        assert job.id not in project.document.jobs
        assert len(project.asked) == 1
        assert item.job_directory.is_dir()
        assert project.status() == "Removed 1 job; deleted 1 folder"

    def test_a_reconfigured_rows_two_jobs_delete_their_one_folder(self, project: _Project) -> None:
        project.import_batch("runs", (1,))
        old = project.native(seed="own")
        (project.charge / "own").mkdir()
        (project.charge / "own" / "own.castep").write_text("gone")
        controller = project.window._muon_relaxation_controller
        replacement_definition = (
            project.window._project_service.register_muon_relaxation_definition(
                project.document,
                input_structure_id=project.host.id,
                parameters={"rows": []},
                output_directory=project.output_root,
            )
        )
        replacement = Job(
            label="native",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_positivemuon_relaxation",
            status=JobStatus.COMPLETED,
            execution_target="local",
        )
        project.document.jobs[replacement.id] = replacement
        replacement_definition.parameters["job_ids_by_label"] = {
            "positivemuon:native": str(replacement.id)
        }
        replacement_definition.parameters["job_seed_names_by_label"] = {
            "positivemuon:native": "own"
        }
        assert {job.id for job in controller.jobs_for_removed_rows({"positivemuon:native"})} == {
            old.id,
            replacement.id,
        }

        project.remove_row()

        assert not (project.charge / "own").exists()
        assert old.id not in project.document.jobs
        assert replacement.id not in project.document.jobs

    def test_one_row_kept_and_one_deleted_says_which_folder_went_where(
        self, project: _Project
    ) -> None:
        item = project.import_batch("runs", (1,)).imported[0]
        project.native(label="shared", seed=item.seed_name)
        project.native(label="mine", seed="own")
        (project.charge / "own").mkdir()
        (project.charge / "own" / "own.castep").write_text("gone")

        project.window._on_remove_muon_relaxation_rows(["positivemuon:shared", "positivemuon:mine"])

        assert not (project.charge / "own").exists()
        assert item.job_directory.is_dir()
        assert project.status() == "Removed 2 jobs; deleted 1 folder; kept 1"
        assert f"positivemuon/{item.seed_name}': another job uses it." in project.boxes[0]


class TestLinks:
    """A link anywhere below files/ is never looked through, on any route."""

    def _alias(self, project: _Project) -> tuple[Path, Path]:
        """A real folder and a link to it, both inside the charge folder."""

        real = project.charge / "real"
        (real / "target").mkdir(parents=True)
        (real / "target" / "target.castep").write_text("kept")
        alias = project.charge / "alias"
        alias.symlink_to(real, target_is_directory=True)
        return real, alias

    def test_native_a_link_in_a_parent_inside_the_project_is_refused(
        self, project: _Project
    ) -> None:
        project.import_batch("runs", (1,))
        self._alias(project)
        job = project.native(seed="alias/target")
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert (project.charge / "real" / "target" / "target.castep").is_file()
        assert job.id in project.document.jobs
        assert len(project.warned) == 1
        assert "link" in project.warned[0]

    def test_batch_a_link_in_a_parent_of_the_same_charge_is_refused(
        self, project: _Project
    ) -> None:
        outcome = project.import_batch("runs", (1,))
        item = outcome.imported[0]
        (project.charge / "real").mkdir()
        (project.charge / "alias").symlink_to(project.charge, target_is_directory=True)
        project.point_at(item, f"alias/{item.seed_name}")
        before = _tree(project.files)

        project.remove_batch(outcome)

        assert _tree(project.files) == before
        assert item.job_directory.is_dir()
        assert item.job_id in project.document.jobs
        assert project.warned and "link" in project.warned[0]

    def test_batch_a_link_to_a_non_job_folder_with_the_matching_name_is_refused(
        self, project: _Project
    ) -> None:
        outcome = project.import_batch("runs", (1,))
        item = outcome.imported[0]
        real = project.charge / "real" / "lookalike"
        real.mkdir(parents=True)
        (real / "notes.txt").write_text("kept")
        (project.charge / "alias").symlink_to(project.charge / "real", target_is_directory=True)
        project.point_at(item, "alias/lookalike")
        before = _tree(project.files)

        project.remove_batch(outcome)

        assert _tree(project.files) == before
        assert (real / "notes.txt").is_file()
        assert item.job_id in project.document.jobs

    def test_batch_a_dangling_link_is_refused(self, project: _Project) -> None:
        outcome = project.import_batch("runs", (1,))
        item = outcome.imported[0]
        (project.charge / "dangling").symlink_to(project.charge / "gone", target_is_directory=True)
        project.point_at(item, "dangling")
        before = _tree(project.files)

        project.remove_batch(outcome)

        assert _tree(project.files) == before
        assert (project.charge / "dangling").is_symlink()
        assert item.job_id in project.document.jobs
        assert project.warned and "link" in project.warned[0]

    def test_native_a_dangling_link_is_refused_and_the_row_stays(self, project: _Project) -> None:
        project.import_batch("runs", (1,))
        (project.charge / "dangling").symlink_to(project.charge / "gone", target_is_directory=True)
        job = project.native(seed="dangling")
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert (project.charge / "dangling").is_symlink()
        assert job.id in project.document.jobs
        assert len(project.warned) == 1
        assert "link" in project.warned[0]

    def test_native_a_dangling_link_in_a_parent_is_refused(self, project: _Project) -> None:
        project.import_batch("runs", (1,))
        (project.charge / "dangling").symlink_to(project.charge / "gone", target_is_directory=True)
        job = project.native(seed="dangling/inner")
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert job.id in project.document.jobs
        assert project.warned and "link" in project.warned[0]

    def test_a_link_cycle_is_refused_on_each_route_and_does_not_stop_others(
        self, project: _Project
    ) -> None:
        outcome = project.import_batch("runs", (1,))
        looped = project.native(label="looped", seed="loop")
        looped.status = JobStatus.ENDED
        looped.end_reason = EndReason.UNKNOWN
        (project.charge / "loop").symlink_to(project.charge / "loop")
        mine = project.native(label="mine", seed="own")
        (project.charge / "own").mkdir()
        (project.charge / "own" / "own.castep").write_text("gone")
        (project.charge / "own" / "own.cell").write_text("input")

        cleaned, errors = project.window._jobs._clear_outputs_for_fresh_start([looped])
        assert cleaned == [] and "link" in errors[0]
        assert (project.charge / "loop").is_symlink()

        project.remove_row("looped")
        assert looped.id in project.document.jobs
        assert (project.charge / "loop").is_symlink()
        assert project.warned and "link" in project.warned[0]

        project.remove_row("mine")
        assert mine.id not in project.document.jobs
        assert not (project.charge / "own").exists()

        project.remove_batch(outcome)
        assert not outcome.imported[0].job_directory.exists()

    def test_native_a_missing_folder_with_no_link_lets_the_row_go(self, project: _Project) -> None:
        """Nothing on disk and no link: nothing to protect, so nothing to refuse."""

        project.import_batch("runs", (1,))
        job = project.native(seed="never_written")
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs
        assert project.warned == []
        assert project.status() == "Removed 1 job"


class TestBatchStatusAndJournal:
    def _partly_shared(self, project: _Project):
        """A batch of three, one of whose folders another import also uses."""

        outcome = project.import_batch("runs", (1, 2, 3))
        other = project.import_batch("other", (4,))
        shared = outcome.imported[0]
        project.point_at(other.imported[0], shared.seed_name)
        import shutil

        shutil.rmtree(other.imported[0].job_directory)
        return outcome, shared, other.imported[0]

    def test_a_partly_shared_batch_says_what_happened_to_each_folder(
        self, project: _Project
    ) -> None:
        outcome, shared, kept_item = self._partly_shared(project)
        gone = [item for item in outcome.imported if item is not shared]
        shared_tree = _tree(shared.job_directory)

        project.remove_batch(outcome)

        assert _tree(shared.job_directory) == shared_tree
        assert all(not item.job_directory.exists() for item in gone)
        assert kept_item.job_id in project.document.jobs
        assert project.status() == "Removed 3 imported relaxations; deleted 2 folders; kept 1"
        assert "MuFinder kept 1 folder because another import uses it." in project.boxes[0]
        assert f"positivemuon/{shared.seed_name}'" in project.boxes[0]

        body = next(
            text for text in project.journal_bodies() if text.startswith("Removed 3 imported")
        )
        assert f"Kept folder '{shared.seed_name}'" in body
        for item in gone:
            assert item.seed_name in body
        assert "Their job directories were deleted" not in body
        assert "Their files stay" not in body


class TestCutoffRoute:
    def _sweep(self, project: _Project, tmp_path: Path):
        service = ProjectService()
        document = service.create_project(tmp_path / "conv.mufinder", "Conv")
        host = Structure(
            label="Host",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
        service.add_structure(document, host)
        document.project.host_structure_id = host.id
        document.ui_state["convergence_form"] = {
            "name_prefix": "conv",
            "cutoff_energies": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
        }
        output_root = document.project_root / "files" / "convergence_cutoff"
        job = service.add_job(
            document,
            label="conv_ecut_300",
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        job.status = JobStatus.ENDED
        job.end_reason = EndReason.UNKNOWN
        definition = service.register_convergence_cutoff_definition(
            document,
            host_structure_id=host.id,
            parameters={
                "name_prefix": "conv",
                "sweep_values": [300.0],
                "shared_params": {"task": "SINGLEPOINT"},
                "overrides": [],
            },
            output_directory=output_root,
        )
        definition.parameters["job_ids_by_cutoff"] = {"300": str(job.id)}
        window = project.window
        window._document = document
        window._refresh_ui_after_project_change()
        project.document = document
        project.files = document.project_root / "files"
        asked: list[Path] = []

        def _delete(_cutoff, path, **kwargs):
            asked.append(path)
            return "delete"

        project.monkeypatch.setattr(window, "_prompt_cutoff_file_disposition", _delete)
        return window, job, output_root, asked

    def test_its_own_folder_still_deletes(self, project: _Project, tmp_path: Path) -> None:
        window, job, output_root, asked = self._sweep(project, tmp_path)
        (output_root / job.label).mkdir(parents=True)
        (output_root / job.label / "out.castep").write_text("gone")
        (output_root / "neighbour").mkdir()
        before = _tree(project.files)

        window._on_remove_cutoffs_from_convergence([300.0])

        assert _tree(project.files) == [path for path in before if job.label not in path]
        assert job.id not in project.document.jobs
        assert len(asked) == 1
        assert "kept" not in project.status()

    def test_a_job_reset_and_detached_earlier_does_not_hold_the_folder(
        self, project: _Project, tmp_path: Path
    ) -> None:
        """A reset leaves its detached job behind, with the sweep point's folder name."""

        window, job, output_root, asked = self._sweep(project, tmp_path)
        (output_root / job.label).mkdir(parents=True)
        (output_root / job.label / "out.castep").write_text("gone")
        detached = Job(
            label=job.label,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
            status=JobStatus.DEFINED,
        )
        project.document.jobs[detached.id] = detached
        assert window._job_output_directory(detached) == output_root / job.label

        window._on_remove_cutoffs_from_convergence([300.0])

        assert not (output_root / job.label).exists()
        assert job.id not in project.document.jobs
        assert len(asked) == 1

    def test_a_reconfigured_sweeps_earlier_job_gives_up_the_folder_with_the_cutoff(
        self, project: _Project, tmp_path: Path
    ) -> None:
        """Both definitions lose the cutoff, so the earlier job cannot keep the folder."""

        window, job, output_root, asked = self._sweep(project, tmp_path)
        (output_root / job.label).mkdir(parents=True)
        (output_root / job.label / "out.castep").write_text("gone")
        first = window._convergence_controller.definitions()[0]
        earlier = Job(
            label=job.label,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
            status=JobStatus.COMPLETED,
        )
        project.document.jobs[earlier.id] = earlier
        service = window._project_service
        second = service.register_convergence_cutoff_definition(
            project.document,
            host_structure_id=first.input_structure_ids[0],
            parameters=dict(first.parameters["configuration"]),
            output_directory=output_root,
        )
        second.parameters["job_ids_by_cutoff"] = {"300": str(earlier.id)}

        window._on_remove_cutoffs_from_convergence([300.0])

        assert len(asked) == 1
        assert not (output_root / job.label).exists()
        assert "kept" not in project.status()

    def test_a_folder_another_job_uses_stays_and_the_cutoff_goes(
        self, project: _Project, tmp_path: Path
    ) -> None:
        window, job, output_root, asked = self._sweep(project, tmp_path)
        (output_root / job.label).mkdir(parents=True)
        (output_root / job.label / "out.castep").write_text("kept")
        other = Job(
            label=job.label,
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
            status=JobStatus.COMPLETED,
        )
        project.document.jobs[other.id] = other
        before = _tree(project.files)

        window._on_remove_cutoffs_from_convergence([300.0])

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs
        assert other.id in project.document.jobs
        assert asked == []
        assert project.status() == "Removed 1 convergence cutoff entry; kept 1 folder"
        assert f"convergence_cutoff/{job.label}'" in project.boxes[0]

    def test_a_link_in_a_parent_is_refused(self, project: _Project, tmp_path: Path) -> None:
        window, job, output_root, asked = self._sweep(project, tmp_path)
        elsewhere = project.files / "kept_here"
        (elsewhere / job.label).mkdir(parents=True)
        (elsewhere / job.label / "out.castep").write_text("kept")
        output_root.parent.mkdir(parents=True, exist_ok=True)
        output_root.symlink_to(elsewhere, target_is_directory=True)
        before = _tree(project.files)

        window._on_remove_cutoffs_from_convergence([300.0])

        assert _tree(project.files) == before
        assert job.id in project.document.jobs
        assert asked == []
        assert project.warned and "link" in project.warned[0]

    def test_a_dangling_link_is_refused(self, project: _Project, tmp_path: Path) -> None:
        window, job, output_root, asked = self._sweep(project, tmp_path)
        output_root.mkdir(parents=True)
        (output_root / job.label).symlink_to(output_root / "gone", target_is_directory=True)
        before = _tree(project.files)

        window._on_remove_cutoffs_from_convergence([300.0])

        assert _tree(project.files) == before
        assert (output_root / job.label).is_symlink()
        assert job.id in project.document.jobs
        assert project.warned and "link" in project.warned[0]


class TestStartAgain:
    """A job that starts again has its previous output cleared only from a folder of its own."""

    def _failed_native(self, project: _Project, label: str, seed: str) -> Job:
        job = project.native(label=label, seed=seed)
        job.status = JobStatus.ENDED
        job.end_reason = EndReason.UNKNOWN
        return job

    def _folder(self, project: _Project, name: str) -> Path:
        folder = project.charge / name
        folder.mkdir(parents=True)
        (folder / f"{name}.cell").write_text("input")
        (folder / f"{name}.castep").write_text("output")
        return folder

    def _start_again(self, project: _Project, jobs: list[Job]) -> tuple[list[Job], list[str]]:
        return project.window._jobs._clear_outputs_for_fresh_start(jobs)

    def test_its_own_folder_is_cleared(self, project: _Project) -> None:
        project.import_batch("runs", (1,))
        job = self._failed_native(project, "native", "own")
        folder = self._folder(project, "own")

        cleaned, errors = self._start_again(project, [job])

        assert _tree(folder) == ["own.cell"]
        assert cleaned == [job] and errors == []

    def test_a_folder_an_import_uses_is_not_cleared(self, project: _Project) -> None:
        item = project.import_batch("runs", (1,)).imported[0]
        job = self._failed_native(project, "native", item.seed_name)
        before = _tree(project.files)

        cleaned, errors = self._start_again(project, [job])

        assert _tree(project.files) == before
        assert cleaned == [] and "another job uses" in errors[0]

    @pytest.mark.parametrize("seed", ["alias/target", "dangling", "dangling/inner"])
    def test_a_link_is_not_looked_through(self, project: _Project, seed: str) -> None:
        project.import_batch("runs", (1,))
        real = project.charge / "real"
        (real / "target").mkdir(parents=True)
        (real / "target" / "target.castep").write_text("kept")
        (project.charge / "alias").symlink_to(real, target_is_directory=True)
        (project.charge / "dangling").symlink_to(project.charge / "gone", target_is_directory=True)
        job = self._failed_native(project, "native", seed)
        before = _tree(project.files)

        cleaned, errors = self._start_again(project, [job])

        assert _tree(project.files) == before
        assert cleaned == [] and "link" in errors[0]

    def test_an_absolute_seed_into_another_workflow_is_not_cleared(self, project: _Project) -> None:
        project.import_batch("runs", (1,))
        folder = project.files / "uep" / "not_a_job"
        folder.mkdir(parents=True)
        (folder / "notes.txt").write_text("kept")
        job = self._failed_native(project, "native", str(folder))
        before = _tree(project.files)

        cleaned, errors = self._start_again(project, [job])

        assert _tree(project.files) == before
        assert cleaned == [] and "another workflow" in errors[0]

    def test_a_mix_clears_only_the_folder_of_its_own(self, project: _Project) -> None:
        item = project.import_batch("runs", (1,)).imported[0]
        own = self._failed_native(project, "mine", "own")
        self._folder(project, "own")
        shared = self._failed_native(project, "shared", item.seed_name)
        imported_tree = _tree(item.job_directory)

        cleaned, errors = self._start_again(project, [own, shared])

        assert _tree(project.charge / "own") == ["own.cell"]
        assert _tree(item.job_directory) == imported_tree
        assert cleaned == [own] and len(errors) == 1

    def test_folders_sharing_a_name_in_two_charge_folders_are_both_cleared(
        self, project: _Project
    ) -> None:
        project.import_batch("runs", (1,))
        positive = self._failed_native(project, "positive", "same")
        muonium = project.native(label="neutral", seed="same", charge="muonium")
        muonium.status = JobStatus.ENDED
        muonium.end_reason = EndReason.UNKNOWN
        self._folder(project, "same")
        other = project.output_root / "muonium" / "same"
        other.mkdir(parents=True)
        (other / "same.cell").write_text("input")
        (other / "same.castep").write_text("output")

        cleaned, errors = self._start_again(project, [positive, muonium])

        assert _tree(project.charge / "same") == ["same.cell"]
        assert _tree(other) == ["same.cell"]
        assert cleaned == [positive, muonium] and errors == []


class TestEnergyBarrierRoute:
    def test_the_status_says_a_folder_stayed(self, project: _Project) -> None:
        """The path's own status message must not hide that its folder was kept."""

        item = project.import_batch("runs", (1,)).imported[0]
        job = Job(
            label="neb",
            workflow_stage=WorkflowStage.ENERGY_BARRIERS,
            calc_type="castep_ts_neb",
            status=JobStatus.COMPLETED,
        )
        project.document.jobs[job.id] = job
        controller = project.window._energy_barriers_controller
        project.monkeypatch.setattr(
            controller, "jobs_for_removed_sources", lambda _keys: [(None, job)]
        )
        project.monkeypatch.setattr(controller, "remove_rows", lambda _keys: None)
        project.monkeypatch.setattr(
            project.window._job_file_cleanup,
            "_job_output_directory",
            lambda candidate: (
                item.job_directory
                if candidate.id == job.id
                else project.window._job_output_directory(candidate)
            ),
        )
        before = _tree(project.files)

        controller.remove_paths(["path"])

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs
        assert project.status() == "Removed 1 energy-barrier path; kept 1 folder"
        assert f"positivemuon/{item.seed_name}'" in project.boxes[0]


class TestUnsafeSavedSeed:
    """Nothing in the app can correct a saved name, so the row goes and the disk is untouched."""

    @pytest.mark.parametrize("seed", [".", "..", "../..", "../../.."])
    def test_a_name_leading_out_lets_the_row_go(self, project: _Project, seed: str) -> None:
        project.import_batch("runs", (1,))
        job = project.native(seed=seed)
        before = _tree(project.files)

        project.remove_row()

        assert _tree(project.files) == before
        assert job.id not in project.document.jobs
        assert project.asked == []
        assert project.warned == []
        assert project.status() == "Removed 1 job; kept 1 folder"
        assert "saved location" in project.boxes[0]

    def test_an_absolute_name_outside_the_project_lets_the_row_go(
        self, project: _Project, tmp_path: Path
    ) -> None:
        project.import_batch("runs", (1,))
        victim = tmp_path / "elsewhere" / "thesis"
        victim.mkdir(parents=True)
        (victim / "thesis.txt").write_text("kept")
        job = project.native(seed=str(victim))

        project.remove_row()

        assert (victim / "thesis.txt").is_file()
        assert job.id not in project.document.jobs
        assert "saved location" in project.boxes[0]

    def test_it_does_not_hold_another_rows_own_folder(self, project: _Project) -> None:
        project.import_batch("runs", (1,))
        project.native(label="stray", seed="..")
        mine = project.native(label="mine", seed="own")
        (project.charge / "own").mkdir()
        (project.charge / "own" / "own.castep").write_text("gone")

        project.remove_row("mine")

        assert not (project.charge / "own").exists()
        assert mine.id not in project.document.jobs
        assert project.boxes == []


class TestALinkIsNoDeadlock:
    def test_removing_the_link_frees_the_row_and_then_the_batch(self, project: _Project) -> None:
        """A native row linked into an import's folder: each refuses until the link is gone."""

        outcome = project.import_batch("runs", (1,))
        item = outcome.imported[0]
        (project.charge / "linked").symlink_to(item.job_directory, target_is_directory=True)
        job = project.native(seed="linked")

        project.remove_row()
        assert job.id in project.document.jobs
        project.remove_batch(outcome)
        assert item.job_id in project.document.jobs
        assert item.job_directory.is_dir()

        (project.charge / "linked").unlink()
        project.remove_row()
        project.remove_batch(outcome)

        assert job.id not in project.document.jobs
        assert item.job_id not in project.document.jobs
        assert not item.job_directory.exists()


class TestTheBoxAndTheBarFit:
    """Measured limits from a GUI check at the default window width and a 1051 px screen."""

    FREE_STATUS_WIDTH = 1028
    SCREEN_HEIGHT = 1051

    @pytest.mark.parametrize(
        "text",
        [
            folder_outcome_status("Removed 999 imported relaxations", deleted=999, kept=999),
            folder_outcome_status("Removed 999 jobs", deleted=999, archived=999, kept=999),
            folder_outcome_status(
                "Removed 999 convergence cutoff entries", deleted=999, archived=999, kept=999
            ),
            folder_outcome_status("Removed 999 energy-barrier paths", deleted=999, kept=999),
        ],
    )
    def test_the_longest_status_fits(self, main_window: MainWindow, text: str) -> None:
        metrics = main_window.statusBar().fontMetrics()

        assert metrics.horizontalAdvance(text) <= self.FREE_STATUS_WIDTH

    @pytest.mark.parametrize("title", ["Folders kept"])
    def test_many_kept_folders_stay_on_screen_with_their_names_under_details(
        self, main_window: MainWindow, monkeypatch, title: str
    ) -> None:
        from PySide6.QtWidgets import QMessageBox

        shown: list[QMessageBox] = []
        monkeypatch.setattr(QMessageBox, "exec", lambda box: shown.append(box) or 0)
        names = [f"positivemuon_Site_{index}" for index in range(15)]
        reason = ("another job uses it", "other jobs use them")

        main_window._show_kept_folders([KeptFolder(name, reason) for name in names], title=title)

        box = shown[0]
        assert box.text() == "MuFinder kept 15 folders because other jobs use them."
        assert box.informativeText() == "Open Details to see which ones."
        assert box.sizeHint().height() < self.SCREEN_HEIGHT
        assert box.text() not in box.detailedText()
        assert all(name in box.detailedText() for name in names)
        assert box.detailedText().index("Site_2") < box.detailedText().index("Site_10")


class TestBatchWithAnUnsafeSavedSeed:
    """As for a row: nothing in the app can correct a saved name, so the records can go."""

    def _question(self, project: _Project) -> list[str]:
        asked: list[str] = []

        def _confirm(*args, **kwargs) -> bool:
            asked.append(" ".join(str(part) for part in args))
            return True

        project.monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _confirm)
        return asked

    @pytest.mark.parametrize(
        "seed", [".", "..", "../..", "../positivemuon", "../muonium/native_job"]
    )
    def test_the_records_go_and_the_disk_is_untouched(self, project: _Project, seed: str) -> None:
        outcome = project.import_batch("runs", (1,))
        item = outcome.imported[0]
        (project.output_root / "muonium" / "native_job").mkdir(parents=True)
        project.point_at(item, seed)
        asked = self._question(project)
        before = _tree(project.files)

        project.remove_batch(outcome)

        assert _tree(project.files) == before
        assert item.job_id not in project.document.jobs
        assert item.site_id not in project.document.sites
        assert project.warned == []
        assert "Its files stay" in asked[0]
        assert "delete" not in asked[0].split("MuFinder did not")[0]
        assert project.status() == "Removed 1 imported relaxation; kept 1 folder"
        assert "saved location" in project.boxes[0]
        body = next(text for text in project.journal_bodies() if "Removed 1 imported" in text)
        assert "nothing there was touched" in body
        assert "was deleted" not in body

    def test_an_absolute_name_outside_the_project(self, project: _Project, tmp_path: Path) -> None:
        outcome = project.import_batch("runs", (1,))
        item = outcome.imported[0]
        victim = tmp_path / "elsewhere" / "positivemuon" / item.seed_name
        victim.mkdir(parents=True)
        (victim / "thesis.txt").write_text("kept")
        project.point_at(item, str(victim))
        before = _tree(project.files)

        project.remove_batch(outcome)

        assert (victim / "thesis.txt").is_file()
        assert _tree(project.files) == before
        assert item.job_id not in project.document.jobs

    def test_a_mixed_batch_deletes_only_the_sound_folders(self, project: _Project) -> None:
        outcome = project.import_batch("runs", (1, 2))
        stray, sound = outcome.imported
        stray_tree = _tree(stray.job_directory)
        project.point_at(stray, "..")
        asked = self._question(project)

        project.remove_batch(outcome)

        assert _tree(stray.job_directory) == stray_tree
        assert not sound.job_directory.exists()
        assert stray.job_id not in project.document.jobs
        assert sound.job_id not in project.document.jobs
        assert "delete its job directory" not in asked[0]
        assert (
            "The files of 1 relaxation stay, because the location saved for it is not its own "
            "folder."
        ) in asked[0]
        assert project.status() == "Removed 2 imported relaxations; deleted 1 folder; kept 1"
        body = next(text for text in project.journal_bodies() if "Removed 2 imported" in text)
        assert f"Deleted folder '{sound.seed_name}'" in body
        assert (
            "Left folder 'files/muon_relaxation' untouched, because its saved location is not "
            "its own."
        ) in body
        assert "Their" not in body

    def test_a_link_still_refuses_and_nothing_goes(self, project: _Project) -> None:
        outcome = project.import_batch("runs", (1,))
        item = outcome.imported[0]
        (project.charge / "alias").symlink_to(project.charge, target_is_directory=True)
        project.point_at(item, f"alias/{item.seed_name}")

        project.remove_batch(outcome)

        assert item.job_id in project.document.jobs
        assert project.warned and "link" in project.warned[0]


class TestDetailsTellFoldersApart:
    def test_one_name_in_two_charge_folders(self, project: _Project) -> None:
        project.import_batch("runs", (1,))
        other = project.native(label="other", seed="same")
        neutral_other = project.native(label="neutral_other", seed="same", charge="muonium")
        for charge in ("positivemuon", "muonium"):
            (project.output_root / charge / "same").mkdir(parents=True)
        first = project.native(label="a", seed="same")
        second = project.native(label="b", seed="same", charge="muonium")

        project.window._on_remove_muon_relaxation_rows(["positivemuon:a", "muonium:b"])

        assert "files/muon_relaxation/muonium/same" in project.boxes[0]
        assert "files/muon_relaxation/positivemuon/same" in project.boxes[0]
        assert first.id not in project.document.jobs
        assert second.id not in project.document.jobs
        assert other.id in project.document.jobs
        assert neutral_other.id in project.document.jobs

    def test_two_names_leading_out_by_different_amounts(self, project: _Project) -> None:
        project.import_batch("runs", (1,))
        project.native(label="a", seed="..")
        project.native(label="b", seed="../../../..")

        project.window._on_remove_muon_relaxation_rows(["positivemuon:a", "positivemuon:b"])

        assert "- '..':" in project.boxes[0]
        assert "- 'files/muon_relaxation':" in project.boxes[0]


class TestABatchWithMissingRecords:
    """A batch some of whose runs the project no longer fully holds.

    A run whose job is gone loses its records and keeps its files: with no job,
    nothing says which folder is its own. A site with no run, or whose job is
    no longer an imported one, stays as it is and the confirmation says so. The
    chooser, the confirmation and the outcome all count the same relaxations.
    """

    def _remove(self, project: _Project, outcome, *, accept: bool = True):
        """Remove ``outcome``'s batch, recording the chooser's labels and the confirmation."""

        listed: list[list[str]] = []
        asked: list[dict[str, str]] = []
        index = [
            position
            for position, batch in enumerate(import_batches(project.document))
            if batch[2] == outcome.batch_id
        ][0]

        def choose(_parent, labels):
            listed.append(list(labels))
            return index

        def confirm(
            _parent, title, text, detail="", accept_label="Remove", *, detailed_text="", **_kw
        ):
            asked.append(
                {
                    "title": title,
                    "text": text,
                    "accept_label": accept_label,
                    "details": detailed_text,
                }
            )
            return accept

        project.monkeypatch.setattr(
            "mufinder.app.workflow_controllers.imported_relaxations.choose_import_batch", choose
        )
        project.monkeypatch.setattr(workflow_table_utils, "confirm_destructive", confirm)
        project.window._imported_relaxations_controller.remove_imported_relaxations()
        return listed, asked

    @staticmethod
    def _forget_job(project: _Project, item) -> None:
        """Drop an imported run's job from the project, as a hand-edited save might."""

        del project.document.jobs[item.job_id]

    @staticmethod
    def _records_gone(project: _Project, item) -> bool:
        document = project.document
        return (
            item.site_id not in document.sites
            and item.run_id not in document.runs
            and item.result_id not in document.results
            and item.structure_id not in document.structures
        )

    def test_a_run_whose_job_is_gone_loses_its_records_and_keeps_its_files(
        self, project: _Project
    ) -> None:
        outcome = project.import_batch("runs", (1, 2))
        orphan, healthy = outcome.imported
        self._forget_job(project, orphan)
        orphan_tree = _tree(orphan.job_directory)

        listed, asked = self._remove(project, outcome)

        assert len(listed) == 1 and len(listed[0]) == 1
        assert listed[0][0].endswith("(2 relaxations)")
        (prompt,) = asked
        assert prompt["accept_label"] == "Remove 2 relaxations"
        assert prompt["text"].startswith(
            "Permanently remove 2 imported relaxations and delete 1 job directory, including "
            "the archived original input files?"
        )
        assert prompt["details"] == ""
        assert prompt["text"].endswith(
            f"\n\nThe project no longer holds the job of {orphan.label}, so MuFinder cannot "
            "tell which files are its own: its records are removed and its files stay on disk."
        )
        assert project.warned == []
        assert self._records_gone(project, healthy)
        assert healthy.job_id not in project.document.jobs
        assert not healthy.job_directory.exists()
        assert self._records_gone(project, orphan)
        assert _tree(orphan.job_directory) == orphan_tree
        assert project.status() == "Removed 2 imported relaxations; deleted 1 folder"
        record = project.document.provenance[-1]
        assert record.parameters["labels"] == [orphan.label, healthy.label]
        assert record.parameters["directories"] == [str(healthy.job_directory)]
        definition = project.window._muon_relaxation_controller.active_definition()
        assert definition.parameters["configuration"]["rows"] == []
        assert definition.parameters["job_ids_by_label"] == {}

    def test_a_batch_whose_jobs_are_all_gone_touches_no_file(self, project: _Project) -> None:
        outcome = project.import_batch("runs", (1, 2))
        for item in outcome.imported:
            self._forget_job(project, item)
        before = _tree(project.files)

        listed, asked = self._remove(project, outcome)

        assert listed[0][0].endswith("(2 relaxations)")
        labels = " and ".join(item.label for item in outcome.imported)
        assert asked[0]["text"] == (
            "Remove 2 imported relaxations from the project?\n\nThe project no longer holds "
            f"the jobs of 2 relaxations ({labels}), so MuFinder cannot tell which files are "
            "theirs: their records are removed and their files stay on disk."
        )
        assert _tree(project.files) == before
        assert all(self._records_gone(project, item) for item in outcome.imported)
        assert project.status() == "Removed 2 imported relaxations"

    def test_a_tagged_site_without_a_run_is_counted_the_same_everywhere(
        self, project: _Project
    ) -> None:
        outcome = project.import_batch("runs", (1, 2))
        bare, healthy = outcome.imported
        project.document.sites[bare.site_id].relaxation_run_id = None

        listed, asked = self._remove(project, outcome)

        assert listed[0][0].endswith("(1 relaxation)")
        (prompt,) = asked
        assert prompt["accept_label"] == "Remove 1 relaxation"
        assert prompt["text"].endswith(
            "\n\n1 other relaxation from this import stays in the project, because it has no "
            f"job: {bare.label}."
        )
        assert bare.site_id in project.document.sites
        assert healthy.site_id not in project.document.sites
        assert project.status() == "Removed 1 imported relaxation; deleted 1 folder"

    def test_a_run_whose_job_is_no_longer_imported_is_counted_as_staying(
        self, project: _Project
    ) -> None:
        outcome = project.import_batch("runs", (1, 2, 3))
        native = outcome.imported[0]
        project.document.jobs[native.job_id].tags = ["muon_relaxation"]
        native_tree = _tree(native.job_directory)

        listed, asked = self._remove(project, outcome)

        assert listed[0][0].endswith("(2 relaxations)")
        (prompt,) = asked
        assert prompt["accept_label"] == "Remove 2 relaxations"
        assert prompt["text"].endswith(
            "\n\n1 other relaxation from this import stays in the project, because its job "
            f"is no longer an imported one: {native.label}."
        )
        assert native.job_id in project.document.jobs
        assert native.site_id in project.document.sites
        assert _tree(native.job_directory) == native_tree
        assert all(item.job_id not in project.document.jobs for item in outcome.imported[1:])
        assert project.status() == "Removed 2 imported relaxations; deleted 2 folders"
        assert project.warned == []

    def test_a_batch_with_nothing_to_remove_says_so_and_changes_nothing(
        self, project: _Project
    ) -> None:
        outcome = project.import_batch("runs", (1, 2))
        for item in outcome.imported:
            project.document.jobs[item.job_id].tags = []
        before = _tree(project.files)
        sites_before = set(project.document.sites)

        listed, asked = self._remove(project, outcome)

        assert listed[0][0].endswith("(nothing to remove)")
        assert asked == []
        assert project.warned == [
            "Nothing in this import can be removed, so nothing was removed.\n"
            + "\n".join(
                f"- {item.label}: its job is no longer an imported one" for item in outcome.imported
            )
            + "\n"
        ]
        assert _tree(project.files) == before
        assert set(project.document.sites) == sites_before

    def test_a_single_site_with_nothing_to_remove_says_why_inline(self, project: _Project) -> None:
        outcome = project.import_batch("runs", (1,))
        (bare,) = outcome.imported
        project.document.sites[bare.site_id].relaxation_run_id = None

        _listed, asked = self._remove(project, outcome)

        assert asked == []
        assert project.warned == [
            "Nothing in this import can be removed, so nothing was removed.\n\n"
            f"- {bare.label}: it has no job"
        ]

    @staticmethod
    def _inline(labels: list[str]) -> str:
        if len(labels) <= 3:
            return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} and {labels[-1]}"
        return f"{', '.join(labels[:3])} and {len(labels) - 3} more"

    @pytest.mark.parametrize("count", [1, 3, 12])
    def test_orphaned_runs_are_named_in_row_order_and_listed_in_full_under_details(
        self, project: _Project, count: int
    ) -> None:
        outcome = project.import_batch("runs", tuple(range(1, count + 2)))
        orphans = outcome.imported[:count]
        for item in orphans:
            self._forget_job(project, item)
        labels = [item.label for item in orphans]

        _listed, asked = self._remove(project, outcome, accept=False)

        (prompt,) = asked
        assert prompt["accept_label"] == f"Remove {count + 1} relaxations"
        assert prompt["text"].startswith(
            f"Permanently remove {count + 1} imported relaxations and delete 1 job directory"
        )
        if count == 1:
            assert prompt["text"].endswith(
                f"The project no longer holds the job of {labels[0]}, "
                "so MuFinder cannot tell which files are its own: its records are removed and "
                "its files stay on disk."
            )
        else:
            assert (
                f"The project no longer holds the jobs of {count} relaxations "
                f"({self._inline(labels)}), so MuFinder cannot tell which files are theirs: "
                "their records are removed and their files stay on disk."
            ) in prompt["text"]
        if count > 3:
            assert prompt["details"] == (
                "Records removed, files kept on disk:\n"
                + "\n".join(f"- {label}" for label in labels)
            )
            assert labels[3] not in prompt["text"]
        else:
            assert prompt["details"] == ""
        assert all(item.site_id in project.document.sites for item in outcome.imported)

    @pytest.mark.parametrize("count", [1, 3, 12])
    def test_sites_that_stay_are_named_in_row_order_and_listed_in_full_under_details(
        self, project: _Project, count: int
    ) -> None:
        outcome = project.import_batch("runs", tuple(range(1, count + 2)))
        staying = outcome.imported[:count]
        for item in staying:
            project.document.jobs[item.job_id].tags = ["muon_relaxation"]
        labels = [item.label for item in staying]

        _listed, asked = self._remove(project, outcome, accept=False)

        (prompt,) = asked
        assert prompt["accept_label"] == "Remove 1 relaxation"
        if count == 1:
            sentence = (
                "1 other relaxation from this import stays in the project, because its job "
                f"is no longer an imported one: {labels[0]}."
            )
        else:
            sentence = (
                f"{count} other relaxations from this import stay in the project, because "
                f"their jobs are no longer imported ones: {self._inline(labels)}."
            )
        assert prompt["text"].endswith("\n\n" + sentence)
        if count > 3:
            assert prompt["details"] == "Staying in the project:\n" + "\n".join(
                f"- {label}: its job is no longer an imported one" for label in labels
            )
        else:
            assert prompt["details"] == ""

    def test_twelve_sites_with_nothing_to_remove_list_under_details(
        self, project: _Project
    ) -> None:
        outcome = project.import_batch("runs", tuple(range(1, 13)))
        for item in outcome.imported:
            project.document.jobs[item.job_id].tags = []

        self._remove(project, outcome)

        (warning,) = project.warned
        summary, *lines = [line for line in warning.split("\n") if line]
        assert summary == "Nothing in this import can be removed, so nothing was removed."
        assert lines == [
            f"- {item.label}: its job is no longer an imported one" for item in outcome.imported
        ]

    def test_a_project_saved_with_a_job_file_missing_can_lose_the_batch_and_save_again(
        self, project: _Project
    ) -> None:
        """The state a hand-edited project reaches, through a real save and load.

        Loading warns about the run whose job is missing and saving refuses
        while it stays, so removing the batch is also what lets the project save.
        """

        outcome = project.import_batch("runs", (1, 2))
        orphan, healthy = outcome.imported
        service = ProjectService()
        service.save_project(project.document)
        (job_file,) = project.document.project_root.rglob(f"*{orphan.job_id}*.json")
        job_file.unlink()
        reloaded = service.load_project(project.document.project_root)
        assert any(str(orphan.job_id) in issue for issue in service.validate_document(reloaded))
        project.window._document = reloaded
        project.document = reloaded
        orphan_tree = _tree(orphan.job_directory)

        self._remove(project, outcome)
        service.save_project(reloaded)
        again = service.load_project(reloaded.project_root)

        assert service.validate_document(again) == []
        assert orphan.site_id not in again.sites
        assert healthy.site_id not in again.sites
        assert import_batches(again) == []
        assert _tree(orphan.job_directory) == orphan_tree
        assert not healthy.job_directory.exists()

    def test_the_integrity_warning_goes_once_the_removal_has_mended_the_project(
        self, project: _Project
    ) -> None:
        """Through File > Open Project: warned once on opening, never again after the removal.

        The removal's save must not carry the warning into the saved project.
        """

        outcome = project.import_batch("runs", (1, 2))
        orphan = outcome.imported[0]
        service = project.window._project_service
        root = project.document.project_root
        service.save_project(project.document)
        (job_file,) = root.rglob(f"*{orphan.job_id}*.json")
        job_file.unlink()

        assert project.window.open_project(root)
        integrity = [text for text in project.warned if "references missing job" in text]
        assert len(integrity) == 1
        project.warned.clear()
        project.document = project.window._document
        self._remove(project, outcome)
        service.save_project(project.document)

        assert project.window.open_project(root)
        assert project.warned == []
        assert "_integrity_warnings" not in json.loads((root / "state.json").read_text())
        assert project.window.open_project(root)
        assert project.warned == []
