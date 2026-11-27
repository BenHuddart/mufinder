"""Removing imported relaxations, and refusing to.

An imported relaxation owns the only copy of its results, so the interesting
behaviour is the *refusal*: a reference the scan misses does not become a
warning, it becomes a dangling id in a saved project. Each referrer is therefore
tested individually rather than through one composite case, so a scan that stops
covering one of them fails a test that names it.

Fixtures are hand-authored: see ``docs/relaxation_import.rst``.
"""

from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, ComparisonTable, Project, Structure
from mufinder.services.relaxation_import import scan_relaxation_folder
from mufinder.services.relaxation_import_commit import commit_relaxation_import
from mufinder.services.relaxation_import_removal import (
    NO_JOB,
    NOT_IMPORTED,
    REMOVAL_ACTION,
    LeftSite,
    batch_contents,
    detach_rows_from_definition,
    find_references,
    import_batches,
    imported_relaxations_in_batch,
    plan_removal,
    remove_imported_relaxations,
)
from mufinder.services.project_service import ProjectDocument

from tests.services.test_relaxation_import_scan import HOST_LATTICE, build_folder


@pytest.fixture()
def host_structure() -> Structure:
    return Structure(
        label="Host",
        origin="imported",
        lattice_vectors=[list(row) for row in HOST_LATTICE],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


@pytest.fixture()
def imported(tmp_path: Path, host_structure: Structure):
    """A project with one committed import batch of two relaxations."""

    project = Project(name="T")
    project.host_structure_id = host_structure.id
    document = ProjectDocument(project_root=tmp_path / "project", project=project)
    document.structures[host_structure.id] = host_structure

    scan = scan_relaxation_folder(
        build_folder(tmp_path / "runs", sites=(1, 2)), host_lattice=HOST_LATTICE
    )
    outcome = commit_relaxation_import(
        document,
        scan=scan,
        runs=scan.importable_runs,
        host_structure=host_structure,
        output_root=document.project_root / "files" / "muon_relaxation",
    )
    return document, outcome


class TestResolvingABatch:
    def test_the_batch_resolves_to_its_relaxations(self, imported) -> None:
        document, outcome = imported

        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert len(refs) == 2
        assert {ref.site_id for ref in refs} == {item.site_id for item in outcome.imported}
        assert all(ref.structure_id is not None and ref.run_id is not None for ref in refs)

    def test_an_unknown_batch_resolves_to_nothing(self, imported) -> None:
        document, _outcome = imported

        assert imported_relaxations_in_batch(document, uuid4()) == []

    def test_a_batch_is_listed_by_its_folder_with_its_size(self, imported) -> None:
        document, outcome = imported
        document.provenance.append(outcome.provenance)

        ((label, count, batch_id),) = import_batches(document)

        assert label.startswith("runs, imported ")
        assert count == 2
        assert batch_id == outcome.batch_id

    def test_a_batch_without_its_record_is_listed_last_by_a_generic_name(
        self, imported, tmp_path: Path, host_structure: Structure
    ) -> None:
        document, first = imported
        scan = scan_relaxation_folder(
            build_folder(tmp_path / "later", sites=(3,)), host_lattice=HOST_LATTICE
        )
        second = commit_relaxation_import(
            document,
            scan=scan,
            runs=scan.importable_runs,
            host_structure=host_structure,
            output_root=document.project_root / "files" / "muon_relaxation",
            existing_labels=[item.label for item in first.imported],
        )
        document.provenance.append(second.provenance)

        batches = import_batches(document)

        assert [(label.split(",")[0], count) for label, count, _id in batches] == [
            ("later", 1),
            ("imported batch", 2),
        ]
        assert [batch_id for _label, _count, batch_id in batches] == [
            second.batch_id,
            first.batch_id,
        ]


class TestWhatABatchRemovalTakes:
    """The chooser's count, the confirmation and the deletion all read ``batch_contents``."""

    def test_a_healthy_batch_takes_every_relaxation(self, imported) -> None:
        document, outcome = imported

        contents = batch_contents(document, outcome.batch_id)

        assert list(contents.targets) == imported_relaxations_in_batch(document, outcome.batch_id)
        assert contents.left == ()

    def test_a_run_whose_job_is_gone_is_taken_and_counted(self, imported) -> None:
        document, outcome = imported
        document.provenance.append(outcome.provenance)
        orphan = outcome.imported[0]
        del document.jobs[orphan.job_id]

        contents = batch_contents(document, outcome.batch_id)

        assert orphan.site_id in {ref.site_id for ref in contents.targets}
        assert contents.left == ()
        assert [count for _label, count, _id in import_batches(document)] == [2]

    @pytest.mark.parametrize("damage", ["no run", "no job id", "not imported"])
    def test_a_site_without_an_imported_job_is_left_and_not_counted(
        self, imported, damage: str
    ) -> None:
        document, outcome = imported
        document.provenance.append(outcome.provenance)
        item = outcome.imported[0]
        if damage == "no run":
            document.sites[item.site_id].relaxation_run_id = None
        elif damage == "no job id":
            document.runs[item.run_id].job_id = None
        else:
            document.jobs[item.job_id].tags = ["muon_relaxation"]

        contents = batch_contents(document, outcome.batch_id)

        assert [ref.site_id for ref in contents.targets] == [outcome.imported[1].site_id]
        reason = NOT_IMPORTED if damage == "not imported" else NO_JOB
        assert contents.left == (LeftSite(item.site_id, item.label, reason),)
        assert [count for _label, count, _id in import_batches(document)] == [1]

    def test_a_batch_with_nothing_to_take_is_still_listed(self, imported) -> None:
        document, outcome = imported
        document.provenance.append(outcome.provenance)
        for item in outcome.imported:
            document.jobs[item.job_id].tags = []

        ((_label, count, batch_id),) = import_batches(document)

        assert (count, batch_id) == (0, outcome.batch_id)

    def test_an_orphaned_run_loses_its_records_and_no_folder(self, imported) -> None:
        document, outcome = imported
        orphan, healthy = outcome.imported
        del document.jobs[orphan.job_id]
        orphan_files = sorted(orphan.job_directory.rglob("*"))
        targets = batch_contents(document, outcome.batch_id).targets
        plan = plan_removal(
            document, targets, job_directories={healthy.job_id: healthy.job_directory}
        )
        assert plan.orphaned_job_ids == frozenset({orphan.job_id})

        result = remove_imported_relaxations(document, plan, batch_id=outcome.batch_id)

        assert result.removed == 2
        assert result.errors == []
        assert result.deleted_directories == (healthy.job_directory,)
        for item in outcome.imported:
            assert item.site_id not in document.sites
            assert item.run_id not in document.runs
            assert item.result_id not in document.results
            assert item.structure_id not in document.structures
        assert sorted(orphan.job_directory.rglob("*")) == orphan_files

    def test_a_job_gone_only_after_the_plan_is_not_treated_as_orphaned(self, imported) -> None:
        """Only a job already missing when the plan was made, and so shown as such, loses its
        records without its folder."""

        document, outcome = imported
        refs = imported_relaxations_in_batch(document, outcome.batch_id)
        plan = plan_removal(
            document,
            refs,
            job_directories={item.job_id: item.job_directory for item in outcome.imported},
        )
        gone = outcome.imported[0]
        del document.jobs[gone.job_id]

        result = remove_imported_relaxations(document, plan)

        assert result.removed == 1
        assert gone.site_id in document.sites
        assert gone.job_directory.is_dir()

    def test_an_orphan_whose_job_came_back_is_not_removed(self, imported) -> None:
        document, outcome = imported
        orphan = outcome.imported[0]
        job = document.jobs.pop(orphan.job_id)
        plan = plan_removal(document, batch_contents(document, outcome.batch_id).targets)
        document.jobs[orphan.job_id] = job

        result = remove_imported_relaxations(document, plan)

        assert result.errors == [
            f"'{orphan.label}' was not removed: its job changed after the check."
        ]
        assert orphan.site_id in document.sites
        assert orphan.job_id in document.jobs


class TestReferencesBlockRemoval:
    def test_a_clean_project_has_no_references(self, imported) -> None:
        document, outcome = imported
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert find_references(document, refs) == []

    def test_the_owning_relaxation_definition_is_not_a_referrer(self, imported) -> None:
        """The muon-relaxation definition holds the imported rows. It is the
        thing being removed from, not a dependant, or removal could never
        proceed at all."""

        document, outcome = imported
        document.calculations[uuid4()] = CalculationDefinition(
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_muon_relaxation",
            input_site_ids=[item.site_id for item in outcome.imported],
        )
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert find_references(document, refs) == []

    def test_a_quantum_corrections_definition_blocks_removal(self, imported) -> None:
        document, outcome = imported
        document.calculations[uuid4()] = CalculationDefinition(
            workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type="castep_quantum_phonon",
            input_site_ids=[outcome.imported[0].site_id],
        )
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        blockers = find_references(document, refs)
        assert any("quantum_corrections" in item for item in blockers)

    def test_an_energy_barriers_definition_blocks_removal(self, imported) -> None:
        document, outcome = imported
        document.calculations[uuid4()] = CalculationDefinition(
            workflow_stage=WorkflowStage.ENERGY_BARRIERS,
            calc_type="castep_neb",
            input_site_ids=[outcome.imported[1].site_id],
        )
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert any("energy_barriers" in item for item in find_references(document, refs))

    def test_a_definition_using_the_relaxed_structure_blocks_removal(self, imported) -> None:
        document, outcome = imported
        document.calculations[uuid4()] = CalculationDefinition(
            workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type="castep_quantum_singlepoint",
            input_structure_ids=[outcome.imported[0].structure_id],
        )
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert find_references(document, refs)

    def test_a_comparison_table_blocks_removal(self, imported) -> None:
        document, outcome = imported
        table = ComparisonTable(
            label="Pristine vs relaxed",
            dft_structure_ids=[outcome.imported[0].structure_id],
        )
        document.comparison_tables[table.id] = table
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert any("comparison table" in item for item in find_references(document, refs))

    def test_an_energy_barrier_path_in_interface_state_blocks_removal(self, imported) -> None:
        """Paths are generated into ui_state before any definition exists, so a
        scan that only looked at definitions would miss them."""

        document, outcome = imported
        document.ui_state["energy_barriers_form"] = {
            "generated_sources": [
                {
                    "key": "path-1",
                    "label": "A to B",
                    "reactant_site_id": str(outcome.imported[0].site_id),
                    "product_site_id": str(uuid4()),
                }
            ]
        }
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert any("energy-barrier path" in item for item in find_references(document, refs))

    def test_a_relaxation_analysis_workspace_blocks_removal(self, imported) -> None:
        """Workspaces reference jobs, not sites or structures."""

        document, outcome = imported
        document.ui_state["relaxation_analysis_workspaces"] = [
            {
                "id": str(uuid4()),
                "name": "positive muon analysis",
                "job_ids": [str(outcome.imported[0].job_id)],
            }
        ]
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert any("workspace" in item for item in find_references(document, refs))

    def test_references_are_reported_once_each(self, imported) -> None:
        document, outcome = imported
        document.ui_state["relaxation_analysis_workspaces"] = [
            {
                "id": str(uuid4()),
                "name": "analysis",
                "job_ids": [str(item.job_id) for item in outcome.imported],
            }
        ]
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        assert len(find_references(document, refs)) == 1


class TestRemoval:
    def _plan(self, document, outcome):
        refs = imported_relaxations_in_batch(document, outcome.batch_id)
        return plan_removal(
            document,
            refs,
            job_directories={item.job_id: item.job_directory for item in outcome.imported},
        )

    def test_a_clean_plan_is_allowed(self, imported) -> None:
        document, outcome = imported

        plan = self._plan(document, outcome)

        assert plan.allowed
        assert len(plan.directories) == 2

    def test_a_blocked_plan_is_not_allowed(self, imported) -> None:
        document, outcome = imported
        document.calculations[uuid4()] = CalculationDefinition(
            workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type="castep_quantum_phonon",
            input_site_ids=[outcome.imported[0].site_id],
        )

        assert not self._plan(document, outcome).allowed

    def test_removal_deletes_every_entity_and_its_files(self, imported) -> None:
        document, outcome = imported
        plan = self._plan(document, outcome)
        directories = list(plan.directories)
        assert all(path.is_dir() for path in directories)

        result = remove_imported_relaxations(document, plan, batch_id=outcome.batch_id)

        assert result.removed == 2
        assert result.errors == []
        for item in outcome.imported:
            assert item.site_id not in document.sites
            assert item.job_id not in document.jobs
            assert item.run_id not in document.runs
            assert item.result_id not in document.results
            assert item.structure_id not in document.structures
        assert not any(path.exists() for path in directories)
        # The host survives.
        assert document.project.host_structure_id in document.structures

    def test_removal_can_keep_the_files(self, imported) -> None:
        document, outcome = imported
        plan = self._plan(document, outcome)

        result = remove_imported_relaxations(document, plan, delete_files=False)

        assert result.removed == 2
        assert all(path.is_dir() for path in plan.directories)

    def test_the_outcome_and_record_name_only_the_folders_that_went(
        self, imported, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        document, outcome = imported
        plan = self._plan(document, outcome)
        stuck, gone = plan.directories
        real_rmtree = shutil.rmtree

        def _rmtree(path, *args, **kwargs):
            if Path(path) == stuck:
                raise PermissionError(13, "denied")
            real_rmtree(path, *args, **kwargs)

        monkeypatch.setattr("mufinder.services.relaxation_import_removal.shutil.rmtree", _rmtree)

        result = remove_imported_relaxations(document, plan)

        assert result.deleted_directories == (gone,)
        assert result.provenance.parameters["directories"] == [str(gone)]
        assert result.provenance.parameters["undeleted_directories"] == [str(stuck)]

    def test_a_target_that_stopped_being_imported_keeps_its_folder(self, imported) -> None:
        document, outcome = imported
        plan = self._plan(document, outcome)
        stayed = outcome.imported[0]
        document.jobs[stayed.job_id].tags = []
        tree = sorted(stayed.job_directory.rglob("*"))

        result = remove_imported_relaxations(document, plan)

        assert result.removed == 1
        assert stayed.job_id in document.jobs
        assert sorted(stayed.job_directory.rglob("*")) == tree
        assert not outcome.imported[1].job_directory.exists()
        assert result.deleted_directories == (outcome.imported[1].job_directory,)

    def test_keeping_the_files_records_no_folder_as_deleted(self, imported) -> None:
        document, outcome = imported

        result = remove_imported_relaxations(
            document, self._plan(document, outcome), delete_files=False
        )

        assert result.deleted_directories == ()
        assert result.provenance.parameters["directories"] == []

    def test_a_blocked_plan_removes_nothing(self, imported) -> None:
        """The guard is on the removal itself, not only on the caller, so a
        hand-built plan cannot bypass the reference check."""

        document, outcome = imported
        document.calculations[uuid4()] = CalculationDefinition(
            workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type="castep_quantum_phonon",
            input_site_ids=[outcome.imported[0].site_id],
        )
        plan = self._plan(document, outcome)

        result = remove_imported_relaxations(document, plan)

        assert result.removed == 0
        assert result.errors
        assert result.provenance is None
        assert outcome.imported[0].site_id in document.sites

    def test_a_job_that_is_no_longer_imported_is_left_alone(self, imported) -> None:
        """Plan ids are resolved before any confirmation dialog. If something
        changed underneath, removing nothing beats deleting a native job."""

        document, outcome = imported
        plan = self._plan(document, outcome)
        document.jobs[outcome.imported[0].job_id].tags = ["muon_relaxation"]

        result = remove_imported_relaxations(document, plan)

        assert result.removed == 1
        assert result.errors and "no longer an imported job" in result.errors[0]
        assert outcome.imported[0].site_id in document.sites
        assert outcome.imported[1].site_id not in document.sites

    def test_the_removal_is_returned_for_the_caller_to_persist(self, imported) -> None:
        """Returned, not appended to the document.

        ``provenance.jsonl`` is append-only through
        ``ProjectService.append_provenance`` and ``save_project`` never rewrites
        it, so a record this function appended itself would never reach disk.
        """

        document, outcome = imported

        result = remove_imported_relaxations(
            document, self._plan(document, outcome), batch_id=outcome.batch_id
        )

        assert not [item for item in document.provenance if item.action == REMOVAL_ACTION]
        record = result.provenance
        assert record is not None
        assert record.action == REMOVAL_ACTION
        assert record.parameters["batch_id"] == str(outcome.batch_id)
        assert len(record.parameters["labels"]) == 2

    def test_the_removal_appears_in_the_journal(self, imported) -> None:
        from mufinder.services.journal import build_journal

        document, outcome = imported
        result = remove_imported_relaxations(document, self._plan(document, outcome))
        assert result.provenance is not None

        entries = [
            entry for day in build_journal("T", [result.provenance]).days for entry in day.entries
        ]
        removal = [entry for entry in entries if entry.title == "Imported relaxations removed"]

        assert len(removal) == 1
        assert removal[0].stamp_kind == "warning"
        assert "archived original" in removal[0].body


class TestDetachingRows:
    def test_rows_and_bindings_are_dropped_together(self, imported) -> None:
        document, outcome = imported
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_muon_relaxation",
            parameters={
                "configuration": {"rows": list(outcome.row_payloads)},
                "job_ids_by_label": {
                    f"positivemuon:{item.label}": str(item.job_id) for item in outcome.imported
                },
                "job_seed_names_by_label": {
                    f"positivemuon:{item.label}": item.seed_name for item in outcome.imported
                },
            },
            input_site_ids=[item.site_id for item in outcome.imported],
        )
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        dropped = detach_rows_from_definition(definition, refs)

        assert dropped == 2
        assert definition.parameters["configuration"]["rows"] == []
        assert definition.parameters["job_ids_by_label"] == {}
        assert definition.parameters["job_seed_names_by_label"] == {}
        assert definition.input_site_ids == []

    def test_unrelated_rows_survive(self, imported) -> None:
        """Rows are matched on site id, not label: a label can be edited or
        suffixed for uniqueness, but the site id is the row's identity."""

        document, outcome = imported
        native_site = uuid4()
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_muon_relaxation",
            parameters={
                "configuration": {
                    "rows": [
                        *outcome.row_payloads,
                        {"site_id": str(native_site), "label": "Site 1", "imported": False},
                    ]
                },
                "job_ids_by_label": {
                    **{f"positivemuon:{item.label}": str(item.job_id) for item in outcome.imported},
                    "positivemuon:Site 1": str(uuid4()),
                },
            },
            input_site_ids=[*(item.site_id for item in outcome.imported), native_site],
        )
        refs = imported_relaxations_in_batch(document, outcome.batch_id)

        detach_rows_from_definition(definition, refs)

        remaining = definition.parameters["configuration"]["rows"]
        assert [row["label"] for row in remaining] == ["Site 1"]
        assert list(definition.parameters["job_ids_by_label"]) == ["positivemuon:Site 1"]
        assert definition.input_site_ids == [native_site]
