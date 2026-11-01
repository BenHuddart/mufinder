"""Committing detected CASTEP relaxations into project entities and files.

The two assertions that matter most are not "did we create records" but the two
gates a real import has to clear: the downstream eligibility gate, and host
resolution at submission time. A site can satisfy the first and still fail the
second, which is exactly the trap this feature was designed around.

Every fixture is hand-authored: see ``docs/relaxation_import.rst`` (Research data
boundary).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Project, Structure
from mufinder.services.relaxation_import import scan_relaxation_folder
from mufinder.services.relaxation_import_commit import (
    SOURCE_DIRNAME,
    SOURCE_MANIFEST_NAME,
    IMPORT_ACTION,
    commit_relaxation_import,
)
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.quantum import quantum_source_eligibility

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
def document(tmp_path: Path, host_structure: Structure) -> ProjectDocument:
    project = Project(name="Test")
    project.host_structure_id = host_structure.id
    doc = ProjectDocument(project_root=tmp_path / "project", project=project)
    doc.structures[host_structure.id] = host_structure
    return doc


def _commit(document: ProjectDocument, folder: Path, host_structure: Structure, **kwargs):
    scan = scan_relaxation_folder(folder, host_lattice=HOST_LATTICE)
    return scan, commit_relaxation_import(
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

        folder = build_folder(tmp_path / "runs", sites=(1,))
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

        folder = build_folder(tmp_path / "runs", sites=(1,))
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

        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        structure_id = outcome.imported[0].structure_id

        service = ProjectService()
        assert service.resolve_underlying_host(document, structure_id) == host_structure.id
        assert service.validate_structures_match_active_host(document, [structure_id]) is None

    def test_the_relaxed_structure_carries_the_muon_species(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        structure = document.structures[outcome.imported[0].structure_id]
        assert "H:mu" in structure.species

    def test_an_out_cell_with_calculation_keywords_is_committed(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The relaxed geometry is read past keywords outside the built-in set."""

        folder = build_folder(tmp_path / "runs", sites=(1,))
        out_cell = next(folder.glob("results/*-out.cell"))
        out_cell.write_text(
            out_cell.read_text(encoding="utf-8")
            + "\n%BLOCK EXTERNAL_PRESSURE\nGPa\n0.5 0.0 0.0\n0.5 0.0\n0.5\n%ENDBLOCK EXTERNAL_PRESSURE\n",
            encoding="utf-8",
        )
        _scan, outcome = _commit(document, folder, host_structure)

        assert outcome.skipped == ()
        structure = document.structures[outcome.imported[0].structure_id]
        assert "H:mu" in structure.species

    def test_a_bare_hydrogen_muon_is_relabelled(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The gate hard-requires ``H:mu``; a run may have used a bare ``H``."""

        folder = build_folder(tmp_path / "runs", sites=(1,), muon_label="H")
        _scan, outcome = _commit(document, folder, host_structure)

        structure = document.structures[outcome.imported[0].structure_id]
        assert "H:mu" in structure.species
        assert "H" not in structure.species


class TestEntities:
    def test_the_job_is_completed_and_marked_imported(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        job = document.jobs[outcome.imported[0].job_id]
        assert job.status is JobStatus.COMPLETED
        assert job.workflow_stage is WorkflowStage.MUON_RELAXATION
        assert "imported" in job.tags
        assert job.calc_type == "castep_positivemuon_relaxation"

    def test_a_muonium_run_gets_the_muonium_calc_type(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1,), castep_kwargs={"charge": " 0.000"})
        _scan, outcome = _commit(document, folder, host_structure)

        assert outcome.imported[0].charge_state == MUONIUM_CHARGE_STATE
        job = document.jobs[outcome.imported[0].job_id]
        assert job.calc_type == "castep_muonium_relaxation"

    def test_the_result_payload_carries_the_total_energy(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The relaxations table reads its energy column from here, and the
        completed-source list sorts on it."""

        folder = build_folder(tmp_path / "runs", sites=(1,))
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
        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        site = document.sites[outcome.imported[0].site_id]
        assert site.status == "relaxed"
        assert site.generation_method == "imported"
        assert site.total_energy_eV == pytest.approx(-1234.5678)
        assert site.source_structure_id == host_structure.id

    def test_labels_do_not_collide_with_existing_rows(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """Row keys are ``{prefix}:{label}``, so a collision would silently
        overwrite an existing row's job mapping."""

        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(
            document, folder, host_structure, existing_labels=["Imported Host_1"]
        )

        assert outcome.imported[0].label == "Imported Host_1 (2)"

    def test_every_imported_row_payload_is_marked(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1, 2))
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

        folder = build_folder(tmp_path / "runs", sites=(1,))
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

        An input cell may carry no k-point grid, so a verbatim copy would
        leave a downstream phonon job silently falling back to project defaults.
        """

        from mufinder.castep_io.cell_parser import parse_cell_file

        folder = build_folder(tmp_path / "runs", sites=(1,))
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
        """The run used ``BASIS_PRECISION``; the cutoff exists only in the echo."""

        from mufinder.castep_io.param_parser import parse_param_file

        folder = build_folder(tmp_path / "runs", sites=(1,))
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

        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        rendered = parse_cell_file(item.job_directory / f"{item.seed_name}.cell")
        # The fixture's muon starts at 0.1/0.2/0.3 and relaxes to 0.11/0.21/0.31.
        assert rendered.fractional_positions[-1] == pytest.approx((0.1, 0.2, 0.3))

    def test_the_rendered_cell_writes_the_codata_muon_gamma(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        cell_text = (item.job_directory / f"{item.seed_name}.cell").read_text()
        gamma_block = cell_text.split("%BLOCK SPECIES_GAMMA")[1].split("%ENDBLOCK SPECIES_GAMMA")[0]
        unit, species, gamma = gamma_block.split()
        assert (unit, species) == ("radsectesla", "H:mu")
        # CODATA gamma_mu = 2 |mu_mu| / hbar (2 pi x 135.538 81 MHz/T), not the old 851586494.1.
        assert float(gamma) == pytest.approx(851_615_456.6, rel=1e-8)

    def test_the_originals_are_archived_byte_exact_with_digests(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        import json

        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]
        archive = item.job_directory / SOURCE_DIRNAME

        archived_cell = archive / f"{item.seed_name}.cell"
        assert (
            archived_cell.read_bytes() == (folder / "input" / f"{item.seed_name}.cell").read_bytes()
        )

        manifest = json.loads((archive / SOURCE_MANIFEST_NAME).read_text())
        assert manifest["version"] == 1
        assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
        assert any(entry["file"].endswith(".castep") for entry in manifest["files"])

    def test_the_copied_castep_is_byte_exact(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        assert (item.job_directory / f"{item.seed_name}.castep").read_bytes() == (
            folder / "results" / f"{item.seed_name}.castep"
        ).read_bytes()


class TestProvenance:
    def test_one_record_describes_the_batch(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1, 2))
        _scan, outcome = _commit(document, folder, host_structure)

        # Returned, not appended to the document: provenance.jsonl is
        # append-only through ProjectService.append_provenance and save_project
        # never rewrites it, so a record appended here would never reach disk.
        assert not [item for item in document.provenance if item.action == IMPORT_ACTION]
        record = outcome.provenance
        assert record is not None
        assert record.action == IMPORT_ACTION
        assert record.parameters["batch_id"] == str(outcome.batch_id)
        assert len(record.parameters["imported_sites"]) == 2
        assert set(record.output_entity_ids) == {item.site_id for item in outcome.imported}

    def test_the_record_names_each_value_source_and_conflict(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """A stale local input tree is the headline hazard; the import
        must say what it saw and which value it took."""

        from tests.services.test_relaxation_import_scan import _run_param

        folder = build_folder(tmp_path / "runs", sites=(1,), param_text=_run_param(xc="LDA"))
        _scan, outcome = _commit(document, folder, host_structure)

        assert outcome.provenance is not None
        site_record = outcome.provenance.parameters["imported_sites"][0]

        assert site_record["parameter_sources"]["cut_off_energy"] == "castep"
        assert any("XC_FUNCTIONAL" in item for item in site_record["conflicts"])
        assert site_record["castep_version"] == "26.11"

    def test_skipped_runs_are_recorded_with_their_reason(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1, 2))
        next(folder.glob("results/*_2-out.cell")).unlink()

        scan = scan_relaxation_folder(folder, host_lattice=HOST_LATTICE)
        outcome = commit_relaxation_import(
            document,
            scan=scan,
            runs=scan.runs,
            host_structure=host_structure,
            output_root=document.project_root / "files" / "muon_relaxation",
        )

        assert len(outcome.imported) == 1
        assert outcome.skipped and outcome.skipped[0][0] == "results/Host_2"
        assert outcome.provenance is not None
        assert outcome.provenance.parameters["skipped"][0]["run"] == "results/Host_2"


class TestJournal:
    def test_the_import_appears_in_the_journal(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """An action with no builder renders nothing at all, so the entry has to
        be added alongside the provenance record."""

        from mufinder.services.journal import build_journal

        folder = build_folder(tmp_path / "runs", sites=(1, 2))
        _scan, outcome = _commit(document, folder, host_structure)
        assert outcome.provenance is not None

        journal = build_journal("Test", [outcome.provenance])
        entries = [entry for day in journal.days for entry in day.entries]

        assert len(entries) == 1
        assert entries[0].title == "CASTEP relaxations imported"
        assert outcome.provenance.action == "import_castep_relaxations"
        assert "read-only" in entries[0].body
        assert "2 muon relaxations" in entries[0].body

    def test_the_entry_flags_skipped_runs_and_conflicts(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        from mufinder.services.journal import build_journal
        from tests.services.test_relaxation_import_scan import _run_param

        folder = build_folder(tmp_path / "runs", sites=(1, 2), param_text=_run_param(xc="LDA"))
        next(folder.glob("results/*_2-out.cell")).unlink()

        scan = scan_relaxation_folder(folder, host_lattice=HOST_LATTICE)
        outcome = commit_relaxation_import(
            document,
            scan=scan,
            runs=scan.runs,
            host_structure=host_structure,
            output_root=document.project_root / "files" / "muon_relaxation",
        )
        assert outcome.provenance is not None

        entry = [e for day in build_journal("T", [outcome.provenance]).days for e in day.entries][0]

        assert entry.stamp_kind == "warning"
        assert "skipped" in entry.body
        assert "disagreement" in entry.body


class TestCoordinateFrames:
    """A row and a site both record host-frame coordinates, like native ones.

    An input cell states the muon position in the *supercell* frame. Every other
    relaxation row takes this field from ``MuonSite.fractional_position`` in host
    coordinates, and job expansion converts host to supercell on the way out, so
    storing supercell coordinates would both display wrong numbers and, if the
    row were ever re-expanded, transform them a second time.
    """

    def test_the_row_position_is_in_the_host_frame(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        payload = outcome.row_payloads[0]
        # The fixture muon sits at 0.1/0.2/0.3 in a 2x2x2 supercell, which is
        # 0.2/0.4/0.6 in the host cell.
        assert payload["initial_fractional_position"] == pytest.approx([0.2, 0.4, 0.6])

    def test_the_site_keeps_its_initial_position_and_copies_the_relaxed_one(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """Native completion never rewrites the site's initial position and copies
        the host-cell relaxed position from the result, so an imported site
        must do the same."""

        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        site = document.sites[outcome.imported[0].site_id]
        assert site.initial_fractional_position == pytest.approx([0.2, 0.4, 0.6])
        data = document.results[outcome.imported[0].result_id].data
        assert data["muon_final_fractional_position_host"] == pytest.approx([0.22, 0.42, 0.62])
        assert site.relaxed_fractional_position == pytest.approx([0.22, 0.42, 0.62])
        assert site.current_fractional_position() == pytest.approx([0.22, 0.42, 0.62])

    def test_the_site_carries_a_cartesian_position(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """Left at the model default, every imported site would read as sitting
        at the origin to any code preferring Cartesian coordinates."""

        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        site = document.sites[outcome.imported[0].site_id]
        # 0.2/0.4/0.6 of a 4 A cubic host.
        assert site.initial_cartesian_position == pytest.approx([0.8, 1.6, 2.4])
        assert site.initial_cartesian_position != [0.0, 0.0, 0.0]


class TestResolvedParameters:
    """One resolution feeds both the rendered ``.param`` and the definition row.

    The row is what the parameter dialog shows; the file is what downstream jobs
    inherit. Two descriptions of the same historical run that disagree would be
    worse than either being wrong alone, so the rule lives in one place.
    """

    def test_the_cutoff_wins_over_the_precision_that_implied_it(self) -> None:
        """A run states BASIS_PRECISION while the output echo reports the
        CUT_OFF_ENERGY CASTEP derived from it. CASTEP treats them as exclusive,
        so a merged set holding both has to be resolved, not passed on."""

        from mufinder.services.relaxation_import_commit import resolve_recovered_parameters

        resolved, dropped = resolve_recovered_parameters(
            {"basis_precision": "FINE", "cut_off_energy": 900.0, "xc_functional": "PBE"}
        )

        assert resolved["cut_off_energy"] == 900.0
        assert "basis_precision" not in resolved
        assert dropped == ["basis_precision"]

    def test_resolution_does_not_depend_on_sort_order(self) -> None:
        """The priority pass exists because 'basis_precision' sorts before
        'cut_off_energy'; without it the wrong half would win."""

        from mufinder.services.relaxation_import_commit import resolve_recovered_parameters

        first, _ = resolve_recovered_parameters(
            {"basis_precision": "FINE", "cut_off_energy": 900.0}
        )
        second, _ = resolve_recovered_parameters(
            {"cut_off_energy": 900.0, "basis_precision": "FINE"}
        )

        assert first == second == {"cut_off_energy": 900.0}

    def test_cell_keywords_stay_out_of_the_param_set(self) -> None:
        """Emitting them in a ``.param`` would produce a file CASTEP rejects."""

        from mufinder.services.relaxation_import_commit import resolve_recovered_parameters

        resolved, _ = resolve_recovered_parameters(
            {"kpoint_mp_grid": (2, 2, 2), "fix_all_cell": True, "cut_off_energy": 900.0}
        )

        assert set(resolved) == {"cut_off_energy"}

    def test_the_row_and_the_rendered_param_agree(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)
        item = outcome.imported[0]

        param_text = (item.job_directory / f"{item.seed_name}.param").read_text(encoding="utf-8")
        row_values = outcome.row_payloads[0]["parameter_values"]

        for name, value in row_values.items():
            if name == "kpoint_mp_grid":
                # Carried by the .cell, never the .param.
                assert name.upper() not in param_text
                continue
            assert name.upper() in param_text, f"{name} is in the row but not the .param"
        assert "BASIS_PRECISION" not in param_text
        assert float(getattr(row_values["cut_off_energy"], "value", 0.0)) == 600.0

    def test_the_row_carries_the_grid_the_run_used(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        """The grid is frequently recoverable only from the output echo: the
        input cell does not state it and the ``.param`` cannot."""

        folder = build_folder(tmp_path / "runs", sites=(1,))
        _scan, outcome = _commit(document, folder, host_structure)

        assert outcome.row_payloads[0]["parameter_values"]["kpoint_mp_grid"] == [2, 2, 2]

    def test_a_run_with_no_recoverable_parameters_yields_an_empty_set(self) -> None:
        """Empty means recovery found nothing — not that the run used whatever
        the workflow currently defaults to."""

        from mufinder.services.relaxation_import import DetectedRun
        from mufinder.services.relaxation_import_commit import row_parameter_values

        assert row_parameter_values(DetectedRun(seed_name="s_1")) == {}


class TestWhatCastepWrote:
    """Imported energies and positions are the numbers in the run's own files."""

    def test_energy_and_positions_match_the_castep_and_cell_files(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        import re

        from tests.services.test_relaxation_import_scan import write_run

        root = tmp_path / "anywhere" / "at" / "all"
        write_run(root, "site")
        castep = (root / "site.castep").read_text(encoding="utf-8")
        written_energy = float(re.search(r"Final energy, E\s+=\s+(\S+)", castep).group(1))

        _scan, outcome = _commit(document, root, host_structure)

        item = outcome.imported[0]
        site = document.sites[item.site_id]
        result = document.results[document.runs[site.relaxation_run_id].result_id]
        assert written_energy == -1234.5678
        assert site.total_energy_eV == written_energy
        assert result.data["total_energy"] == written_energy
        # The 2x2x2 cell states (0.11, 0.21, 0.31) and started at (0.1, 0.2, 0.3).
        assert result.data["muon_final_fractional_position_supercell"] == [0.11, 0.21, 0.31]
        assert site.relaxed_fractional_position == pytest.approx([0.22, 0.42, 0.62])
        assert site.initial_fractional_position == pytest.approx([0.2, 0.4, 0.6])
        structure = document.structures[item.structure_id]
        assert structure.species[-1] == "H:mu"
        assert structure.fractional_positions[-1] == [0.11, 0.21, 0.31]

    def test_the_outputs_are_copied_byte_for_byte(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        from tests.services.test_relaxation_import_scan import write_run

        root = write_run(tmp_path / "run", "site")

        _scan, outcome = _commit(document, root, host_structure)

        directory = outcome.imported[0].job_directory
        for name in ("site.castep", "site-out.cell"):
            assert (directory / name).read_bytes() == (root / name).read_bytes()
        assert (directory / SOURCE_DIRNAME / "site.cell").read_bytes() == (
            root / "site.cell"
        ).read_bytes()
        manifest = json.loads((directory / SOURCE_DIRNAME / SOURCE_MANIFEST_NAME).read_text())
        assert {entry["file"] for entry in manifest["files"]} >= {"site.castep", "site-out.cell"}


class TestNamedFromTheSeed:
    def test_rows_are_named_from_the_seed(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        from tests.services.test_relaxation_import_scan import write_run

        root = write_run(tmp_path / "run", "fluorine_site")

        _scan, outcome = _commit(document, root, host_structure)

        assert outcome.imported[0].label == "Imported fluorine_site"
        job = document.jobs[outcome.imported[0].job_id]
        assert job.label == "Imported fluorine_site"
        assert job.tags == ["muon_relaxation", job.calc_type, "imported"]

    def test_one_seed_in_two_folders_keeps_two_jobs_and_two_sets_of_files(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        from tests.services.test_relaxation_import_scan import _out_cell, write_run

        write_run(tmp_path / "runs" / "site1", out_text=_out_cell(muon=(0.11, 0.21, 0.31)))
        write_run(
            tmp_path / "runs" / "site2",
            out_text=_out_cell(muon=(0.41, 0.51, 0.61)),
            castep_kwargs={"cutoff": "  700.0000   eV"},
        )

        _scan, outcome = _commit(document, tmp_path / "runs", host_structure)

        first, second = outcome.imported
        assert (first.label, second.label) == ("Imported mu", "Imported mu (2)")
        assert (first.seed_name, second.seed_name) == ("mu", "mu_2")
        assert first.job_directory != second.job_directory
        assert (second.job_directory / "mu_2.castep").is_file()
        assert (second.job_directory / "mu_2-out.cell").is_file()
        assert (second.job_directory / "mu_2.cell").is_file()
        assert (second.job_directory / SOURCE_DIRNAME / "mu.cell").is_file()
        assert (first.job_directory / "mu-out.cell").read_bytes() == (
            tmp_path / "runs" / "site1" / "mu-out.cell"
        ).read_bytes()

    def test_importing_a_folder_twice_does_not_overwrite_the_first_batch(
        self, tmp_path: Path, document: ProjectDocument, host_structure: Structure
    ) -> None:
        from tests.services.test_relaxation_import_scan import write_run

        root = write_run(tmp_path / "run")

        _scan, first = _commit(document, root, host_structure)
        _scan, second = _commit(
            document, root, host_structure, existing_labels=[first.imported[0].label]
        )

        assert first.imported[0].job_directory != second.imported[0].job_directory
