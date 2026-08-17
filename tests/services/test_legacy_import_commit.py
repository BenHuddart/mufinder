"""Committing detected legacy relaxations into project entities and files.

The two assertions that matter most are not "did we create records" but the two
gates a real import has to clear: the downstream eligibility gate, and host
resolution at submission time. A site can satisfy the first and still fail the
second, which is exactly the trap this feature was designed around.

Every fixture is hand-authored: see ``docs/legacy_import.rst`` (Research data
boundary).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Project, Structure
from mufinder.services.legacy_import import scan_legacy_folder
from mufinder.services.legacy_import_commit import (
    IMPORT_ACTION,
    LEGACY_MANIFEST_NAME,
    LEGACY_SOURCE_DIRNAME,
    commit_legacy_import,
)
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.quantum import quantum_source_eligibility

from tests.services.test_legacy_import_scan import HOST_LATTICE, build_folder


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
def document(tmp_path: Path, host_structure: Structure) -> ProjectDocument:
    project = Project(name="Test")
    project.host_structure_id = host_structure.id
    doc = ProjectDocument(project_root=tmp_path / "project", project=project)
    doc.structures[host_structure.id] = host_structure
    return doc


def _commit(document: ProjectDocument, folder: Path, host_structure: Structure, **kwargs):
    scan = scan_legacy_folder(folder, host_lattice=HOST_LATTICE)
    return scan, commit_legacy_import(
        document,
        scan=scan,
        runs=scan.importable_runs,
        host_structure=host_structure,
        output_root=document.project_root / "files" / "muon_relaxation",
        **kwargs,
    )


class TestEligibility:
    def test_an_imported_site_passes_the_downstream_gate(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The whole point: Quantum Corrections and Energy Barriers accept it."""

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        assert len(outcome.imported) == 1
        item = outcome.imported[0]
        site = document.sites[item.site_id]
        run = document.runs[item.run_id]
        job = document.jobs[item.job_id]
        structure = document.structures[item.structure_id]

        verdict = quantum_source_eligibility(
            site=site, run=run, job=job, relaxed_structure=structure
        )

        assert verdict.eligible, verdict.reason

    def test_the_run_result_and_structure_resolve_from_each_other(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The gate's caller walks run -> job -> result -> structure and skips
        a candidate on any missing link."""

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        site = document.sites[item.site_id]
        run = document.runs[site.relaxation_run_id]

        assert run.job_id == item.job_id
        assert run.result_id == item.result_id
        result = document.results[run.result_id]
        assert result.derived_structure_ids == [item.structure_id]

    def test_the_structure_resolves_to_the_active_host(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """Passing eligibility is not enough to *submit*.

        Submission separately walks ``parent_structure_id`` for an ancestor with
        a host origin. Without it the walk returns ``None``, which compares
        unequal to the host and raises a host-mismatch dialog -- after the site
        has passed every visible check.
        """

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        structure_id = outcome.imported[0].structure_id

        service = ProjectService()
        assert service.resolve_underlying_host(document, structure_id) == host_structure.id
        assert service.validate_structures_match_active_host(document, [structure_id]) is None

    def test_the_relaxed_structure_carries_the_muon_species(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        structure = document.structures[outcome.imported[0].structure_id]
        assert "H:mu" in structure.species

    def test_a_bare_hydrogen_muon_is_relabelled(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The gate hard-requires ``H:mu``; legacy accepted a bare ``H``."""

        folder = build_folder(tmp_path / "legacy", sites=(1,), muon_label="H")
        _scan, outcome = _commit(document, folder, host_structure)

        structure = document.structures[outcome.imported[0].structure_id]
        assert "H:mu" in structure.species
        assert "H" not in structure.species


class TestEntities:
    def test_the_job_is_completed_and_marked_imported(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        job = document.jobs[outcome.imported[0].job_id]
        assert job.status is JobStatus.COMPLETED
        assert job.workflow_stage is WorkflowStage.MUON_RELAXATION
        assert "imported" in job.tags
        assert job.calc_type == "castep_positivemuon_relaxation"

    def test_a_muonium_run_gets_the_muonium_calc_type(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "legacy", sites=(1,), castep_kwargs={"charge": " 0.000"})
        _scan, outcome = _commit(document, folder, host_structure)

        assert outcome.imported[0].charge_state == MUONIUM_CHARGE_STATE
        job = document.jobs[outcome.imported[0].job_id]
        assert job.calc_type == "castep_muonium_relaxation"

    def test_the_result_payload_carries_the_total_energy(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The relaxations table reads its energy column from here, and the
        completed-source list sorts on it."""

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        data = document.results[outcome.imported[0].result_id].data
        assert data["total_energy"] == pytest.approx(-1234.5678)
        assert data["geom_converged"] is True
        assert data["imported"] is True
        assert data["lattice_a"] == pytest.approx(8.0)
        assert "muon_final_fractional_position_host" in data

    def test_the_site_records_its_energy_and_relaxed_status(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        site = document.sites[outcome.imported[0].site_id]
        assert site.status == "relaxed"
        assert site.generation_method == "imported_legacy"
        assert site.relative_energy_eV == pytest.approx(-1234.5678)
        assert site.source_structure_id == host_structure.id

    def test_labels_do_not_collide_with_existing_rows(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """Row keys are ``{prefix}:{label}``, so a collision would silently
        overwrite an existing row's job mapping."""

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(
            document, folder, host_structure, existing_labels=["Legacy Site 1"]
        )

        assert outcome.imported[0].label != "Legacy Site 1"

    def test_every_imported_row_payload_is_marked(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "legacy", sites=(1, 2))
        _scan, outcome = _commit(document, folder, host_structure)

        assert len(outcome.row_payloads) == 2
        for payload in outcome.row_payloads:
            assert payload["imported"] is True
            assert payload["import_batch_id"] == str(outcome.batch_id)
            assert payload["supercell_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


class TestFiles:
    def test_the_job_directory_carries_what_downstream_reads(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """Downstream inheritance re-reads ``<seed>.cell`` and ``<seed>.param``
        from disk, so both must exist under the resolved job directory."""

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        job_dir = outcome.imported[0].job_directory
        seed = outcome.imported[0].seed_name

        assert (job_dir / f"{seed}.cell").is_file()
        assert (job_dir / f"{seed}.param").is_file()
        assert (job_dir / f"{seed}.castep").is_file()
        assert (job_dir / f"{seed}-out.cell").is_file()

    def test_the_rendered_cell_states_the_grid_the_run_used(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The regression this whole layout exists to prevent.

        A legacy input cell carries no k-point grid, so a verbatim copy would
        leave a downstream phonon job silently falling back to project defaults.
        """

        from mufinder.castep_io.cell_parser import parse_cell_file

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        source_grid = parse_cell_file(
            folder / "input" / f"{item.seed_name}.cell"
        ).cell_file.kpoint_mp_grid
        rendered = parse_cell_file(item.job_directory / f"{item.seed_name}.cell")

        assert source_grid is None
        assert rendered.cell_file.kpoint_mp_grid == (2, 2, 2)

    def test_the_rendered_param_states_the_cutoff_the_run_used(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """Legacy used ``BASIS_PRECISION``; the cutoff exists only in the echo."""

        from mufinder.castep_io.param_parser import parse_param_file

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        parsed = parse_param_file(item.job_directory / f"{item.seed_name}.param")
        values = {item_.name: item_.value for item_ in parsed.param_file.keywords}

        assert values["cut_off_energy"].value == pytest.approx(600.0)
        assert values["charge"] == 1
        assert values["task"] == "GEOMETRYOPTIMIZATION"

    def test_the_rendered_cell_holds_the_initial_geometry(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """Initial-versus-relaxed comparison must stay truthful."""

        from mufinder.castep_io.cell_parser import parse_cell_file

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        rendered = parse_cell_file(item.job_directory / f"{item.seed_name}.cell")
        # The fixture's muon starts at 0.1/0.2/0.3 and relaxes to 0.11/0.21/0.31.
        assert rendered.fractional_positions[-1] == pytest.approx((0.1, 0.2, 0.3))

    def test_the_originals_are_archived_byte_exact_with_digests(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        import json

        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]
        archive = item.job_directory / LEGACY_SOURCE_DIRNAME

        archived_cell = archive / f"{item.seed_name}.cell"
        assert archived_cell.read_bytes() == (
            folder / "input" / f"{item.seed_name}.cell"
        ).read_bytes()

        manifest = json.loads((archive / LEGACY_MANIFEST_NAME).read_text())
        assert manifest["version"] == 1
        assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
        assert any(entry["file"].endswith(".castep") for entry in manifest["files"])

    def test_the_copied_castep_is_byte_exact(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "legacy", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        assert (item.job_directory / f"{item.seed_name}.castep").read_bytes() == (
            folder / "results" / f"{item.seed_name}.castep"
        ).read_bytes()


class TestProvenance:
    def test_one_record_describes_the_batch(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "legacy", sites=(1, 2))
        _scan, outcome = _commit(document, folder, host_structure)

        records = [item for item in document.provenance if item.action == IMPORT_ACTION]
        assert len(records) == 1
        record = records[0]
        assert record.parameters["batch_id"] == str(outcome.batch_id)
        assert len(record.parameters["imported_sites"]) == 2
        assert set(record.output_entity_ids) == {item.site_id for item in outcome.imported}

    def test_the_record_names_each_value_source_and_conflict(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """A stale local input tree is the headline legacy hazard; the import
        must say what it saw and which value it took."""

        from tests.services.test_legacy_import_scan import _legacy_param

        folder = build_folder(tmp_path / "legacy", sites=(1,), param_text=_legacy_param(xc="LDA"))
        _scan, outcome = _commit(document, folder, host_structure)

        site_record = [
            item for item in document.provenance if item.action == IMPORT_ACTION
        ][0].parameters["imported_sites"][0]

        assert site_record["parameter_sources"]["cut_off_energy"] == "castep"
        assert any("XC_FUNCTIONAL" in item for item in site_record["conflicts"])
        assert site_record["castep_version"] == "26.11"

    def test_skipped_runs_are_recorded_with_their_reason(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "legacy", sites=(1, 2))
        next(folder.glob("results/*_2-out.cell")).unlink()

        scan = scan_legacy_folder(folder, host_lattice=HOST_LATTICE)
        outcome = commit_legacy_import(
            document,
            scan=scan,
            runs=scan.runs,
            host_structure=host_structure,
            output_root=document.project_root / "files" / "muon_relaxation",
        )

        assert len(outcome.imported) == 1
        assert outcome.skipped and outcome.skipped[0][0] == 2
        record = [item for item in document.provenance if item.action == IMPORT_ACTION][0]
        assert record.parameters["skipped"][0]["site_index"] == 2


class TestJournal:
    def test_the_import_appears_in_the_journal(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """An action with no builder renders nothing at all, so the entry has to
        be added alongside the provenance record."""

        from mufinder.services.journal import build_journal

        folder = build_folder(tmp_path / "legacy", sites=(1, 2))
        _scan, _outcome = _commit(document, folder, host_structure)

        journal = build_journal("Test", document.provenance)
        entries = [entry for day in journal.days for entry in day.entries]

        assert len(entries) == 1
        assert entries[0].title == "Legacy relaxations imported"
        assert "read-only" in entries[0].body
        assert "2 muon relaxation(s)" in entries[0].body

    def test_the_entry_flags_skipped_runs_and_conflicts(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        from mufinder.services.journal import build_journal
        from tests.services.test_legacy_import_scan import _legacy_param

        folder = build_folder(
            tmp_path / "legacy", sites=(1, 2), param_text=_legacy_param(xc="LDA")
        )
        next(folder.glob("results/*_2-out.cell")).unlink()

        scan = scan_legacy_folder(folder, host_lattice=HOST_LATTICE)
        commit_legacy_import(
            document,
            scan=scan,
            runs=scan.runs,
            host_structure=host_structure,
            output_root=document.project_root / "files" / "muon_relaxation",
        )

        entry = [e for day in build_journal("T", document.provenance).days for e in day.entries][0]

        assert entry.stamp_kind == "warning"
        assert "skipped" in entry.body
        assert "disagreement" in entry.body
