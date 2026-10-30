"""Tests for journal derivation and export from provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Project, ProvenanceRecord
from mufinder.services import site_generation_service
from mufinder.services.project_document import ProjectDocument
from tests.support.symmetry_hosts import cof2_host
from tests.support.journal import generation_record
from mufinder.services.journal import (
    build_journal,
    export_journal,
    extend_journal,
    render_journal_markdown,
    render_journal_text,
    stamp_label,
)


def test_build_journal_groups_entries_by_day_and_maps_actions() -> None:
    first = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action="import_structure",
        workflow_stage=WorkflowStage.MUON_GENERATION,
        parameters={"source_file": "/tmp/host.cif"},
        output_entity_ids=[uuid4()],
    )
    second = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 15, 0, tzinfo=UTC),
        action="set_recommended_cutoff",
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        parameters={
            "recommended_cutoff_energy": 600.0,
            "recommended_cutoff_method": "energy_window",
        },
    )
    third = ProvenanceRecord(
        timestamp=datetime(2026, 4, 22, 10, 0, tzinfo=UTC),
        action="job_completed",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        parameters={
            "job_label": "relax_row_1",
            "execution_target": "local",
            "result_total_energy": -123.45,
            "result_max_force": 0.02,
            "derived_structure_count": 1,
        },
        job_id=uuid4(),
    )

    journal = build_journal("JournalTest", [second, third, first])

    assert [day.date_label for day in journal.days] == [
        "Tuesday 21 April 2026",
        "Wednesday 22 April 2026",
    ]
    assert [entry.title for entry in journal.days[0].entries] == [
        "Host structure imported",
        "Recommended cutoff adopted",
    ]
    assert journal.days[0].entries[1].stamp_kind == "success"
    assert journal.days[1].entries[0].title == "Calculation completed"
    assert journal.days[1].entries[0].workflow_label == "Pristine Relaxation"


def test_rendered_journal_omits_clock_times() -> None:
    record = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 45, 12, tzinfo=UTC),
        action="submit_jobs",
        workflow_stage=WorkflowStage.UEP,
        parameters={
            "execution_target": "local",
            "job_count": 2,
            "job_labels": ["uep_a", "uep_b"],
        },
    )
    journal = build_journal("NoTimes", [record])

    text = render_journal_text(journal)
    markdown = render_journal_markdown(journal)

    assert "09:45" not in text
    assert "09:45" not in markdown
    assert "Tuesday 21 April 2026" in text
    assert "## Tuesday 21 April 2026" in markdown


def test_job_continuation_entry_names_the_reused_directory() -> None:
    record = ProvenanceRecord(
        timestamp=datetime(2026, 8, 1, 11, 0, tzinfo=UTC),
        action="job_continuation",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        parameters={
            "job_label": "relax_row_1",
            "calc_type": "castep_pristine_relaxation",
            "execution_target": "scarf",
            "preserved_remote_directory": True,
            "remote_directory": "/scratch/user/relax_row_1",
        },
        job_id=uuid4(),
    )

    journal = build_journal("ContinueTest", [record])
    entry = journal.days[0].entries[0]

    assert entry.title == "Calculation continued"
    assert entry.stamp_kind == "milestone"
    assert "relax_row_1" in entry.body
    assert "/scratch/user/relax_row_1" in entry.body
    assert entry.workflow_label == "Pristine Relaxation"
    assert entry.linked_job_id == record.job_id


def test_job_continuation_entry_omits_directory_for_a_local_job() -> None:
    record = ProvenanceRecord(
        timestamp=datetime(2026, 8, 1, 11, 0, tzinfo=UTC),
        action="job_continuation",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        parameters={
            "job_label": "mu_row_1",
            "execution_target": "local",
            "preserved_remote_directory": False,
            "remote_directory": None,
        },
    )

    journal = build_journal("ContinueLocal", [record])
    entry = journal.days[0].entries[0]

    assert entry.title == "Calculation continued"
    assert "reusing" not in entry.body


def test_export_journal_writes_text_and_markdown(tmp_path: Path) -> None:
    record = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
        action="create_comparison_table",
        parameters={"table_label": "Comparison 1", "dft_structure_count": 2},
        output_entity_ids=[uuid4()],
    )
    journal = build_journal("ExportMe", [record])

    text_path = tmp_path / "journal.txt"
    md_path = tmp_path / "journal.md"
    export_journal(journal, text_path, format_name="text")
    export_journal(journal, md_path, format_name="markdown")

    text = text_path.read_text(encoding="utf-8")
    markdown = md_path.read_text(encoding="utf-8")

    assert "ExportMe Journal" in text
    assert "Comparison table created" in text
    assert "# ExportMe Journal" in markdown
    assert "**[Completed] Comparison table created**" in markdown


def test_journal_describes_project_xc_functional_defaults() -> None:
    records = [
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
            action="set_recommended_cutoff",
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            parameters={
                "recommended_cutoff_energy": 600.0,
                "recommended_cutoff_xc_functional": "PBE",
            },
        ),
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
            action="set_recommended_kpoint_grid",
            workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
            parameters={
                "recommended_kpoint_grid": [4, 4, 4],
                "recommended_kpoint_xc_functional": "PBE",
            },
        ),
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 11, 0, tzinfo=UTC),
            action="set_recommended_xc_functional",
            parameters={"recommended_xc_functional": "PBESOL"},
        ),
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 12, 0, tzinfo=UTC),
            action="set_recommended_xc_functional",
            parameters={"recommended_xc_functional": None},
        ),
    ]

    entries = build_journal("XC", records).days[0].entries

    assert entries[0].body == (
        "Set the project default cutoff to 600 eV (converged with the PBE functional)."
    )
    assert entries[1].body == (
        "Set the project default k-point grid to 4x4x4 (converged with the PBE functional)."
    )
    assert entries[2].title == "Default XC functional set"
    assert "PBESOL" in entries[2].body
    assert entries[3].title == "Default XC functional cleared"


def test_journal_distinguishes_manual_and_cleared_defaults() -> None:
    def record(hour: int, action: str, **parameters: object) -> ProvenanceRecord:
        return ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, hour, 0, tzinfo=UTC),
            action=action,
            parameters=parameters,
        )

    entries = (
        build_journal(
            "Manual",
            [
                record(
                    9,
                    "set_recommended_cutoff",
                    recommended_cutoff_energy=800.0,
                    recommended_cutoff_method="manual",
                ),
                record(
                    10,
                    "set_recommended_kpoint_grid",
                    recommended_kpoint_grid=[6, 6, 4],
                    recommended_kpoint_method="manual",
                ),
                record(11, "clear_recommended_cutoff"),
                record(12, "clear_recommended_kpoint_grid"),
            ],
        )
        .days[0]
        .entries
    )

    assert [entry.title for entry in entries] == [
        "Default cutoff set manually",
        "Default k-point grid set manually",
        "Default cutoff cleared",
        "Default k-point grid cleared",
    ]
    assert "800 eV by hand" in entries[0].body
    assert "6x6x4 by hand" in entries[1].body


@pytest.mark.parametrize("method", ["converged", "delta-threshold", "asymptotic", None])
def test_journal_adopted_defaults_name_no_convergence_method(method: str | None) -> None:
    def record(hour: int, action: str, **parameters: object) -> ProvenanceRecord:
        return ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, hour, 0, tzinfo=UTC),
            action=action,
            parameters=parameters,
        )

    entries = (
        build_journal(
            "Adopted",
            [
                record(
                    9,
                    "set_recommended_cutoff",
                    recommended_cutoff_energy=600.0,
                    recommended_cutoff_method=method,
                    recommended_cutoff_xc_functional="PBE",
                ),
                record(
                    10,
                    "set_recommended_kpoint_grid",
                    recommended_kpoint_grid=[4, 4, 6],
                    recommended_kpoint_method=method,
                ),
            ],
        )
        .days[0]
        .entries
    )

    assert [entry.title for entry in entries] == [
        "Recommended cutoff adopted",
        "Recommended k-point grid adopted",
    ]
    assert entries[0].body.startswith("Set the project default cutoff to 600 eV")
    assert entries[1].body.startswith("Set the project default k-point grid to 4x4x6")
    for entry in entries:
        assert " using " not in entry.body


def test_journal_counts_cutoffs_under_either_configuration_key() -> None:
    # Provenance is never migrated: records from before schema 2 may hold the
    # sweep only under the legacy ``cutoff_energies`` key.
    def configured(configuration: dict[str, object]) -> ProvenanceRecord:
        return ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
            action="configure_convergence_cutoff",
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            parameters={"configuration": configuration},
        )

    records = [
        configured({"cutoff_energies": [300.0, 360.0]}),
        configured(
            {"sweep_values": [300.0, 360.0, 420.0], "cutoff_energies": [300.0, 360.0, 420.0]}
        ),
        configured({"sweep_values": [300.0, 360.0, 420.0, 480.0]}),
    ]

    bodies = [entry.body for entry in build_journal("Cutoffs", records).days[0].entries]

    assert bodies == [
        f"Prepared a cutoff convergence sweep with {count} cutoff values." for count in (2, 3, 4)
    ]


def test_symmetry_unavailable_entry_warns_with_the_recorded_reason() -> None:
    record = ProvenanceRecord(
        timestamp=datetime(2026, 10, 29, 9, 0, tzinfo=UTC),
        action="symmetry_unavailable",
        workflow_stage=WorkflowStage.MUON_GENERATION,
        actor="system",
        description="Host symmetry was not detected: species 'Xx' name no element.",
    )

    entry = build_journal("Symmetry", [record]).days[0].entries[0]

    assert entry.title == "Host symmetry not detected"
    assert entry.stamp_kind == "warning"
    assert entry.body == record.description


def test_extend_journal_matches_a_full_build_of_the_grown_stream() -> None:
    records = [generation_record(9, day=21), generation_record(10, day=21)]
    journal = build_journal("Stream", records)
    added = [
        generation_record(11, day=21),
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 22, 9, 0, tzinfo=UTC), action="not_journalled"
        ),
        generation_record(10, day=22),
    ]

    extended = extend_journal(journal, added)

    assert extended == build_journal("Stream", records + added)
    assert extend_journal(journal, []) is journal


def test_extend_journal_refuses_a_record_older_than_the_last_entry() -> None:
    journal = build_journal("Stream", [generation_record(9, day=22)])

    assert extend_journal(journal, [generation_record(9, day=21)]) is None


def _muon_sites_record(action: str, **fields: object) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp=datetime(2026, 10, 30, 9, 0, tzinfo=UTC),
        action=action,
        workflow_stage=WorkflowStage.MUON_GENERATION,
        actor="user",
        **fields,  # type: ignore[arg-type]
    )


def test_a_generation_entry_prints_the_run_line_and_seed(tmp_path: Path) -> None:
    document = ProjectDocument(project_root=tmp_path, project=Project(name="Runs"))
    host = cof2_host()
    document.structures[host.id] = host
    document.project.host_structure_id = host.id
    outcome = site_generation_service.generate_candidate_sites(
        document,
        method="random",
        parameters={"selection_mode": "count", "count": 3, "min_muon_atom": 1.0, "seed": 42},
    )

    entry = build_journal("Runs", outcome.provenance).days[0].entries[0]

    assert entry.title == "Muon sites generated"
    assert entry.body.startswith("Random, 3 starts, ≥ ")
    assert entry.body.endswith(", ≥ 1.0 Å from atoms, P4₂/mnm (16 ops), seed 42.")


def test_a_generation_entry_without_a_run_keeps_the_older_sentence() -> None:
    entry = build_journal("Old", [generation_record(9, count=2)]).days[0].entries[0]

    assert entry.body == "Generated 0 muon sites with 'Random generation'."


def test_a_generation_entry_without_a_run_never_shows_an_unknown_method_id() -> None:
    record = generation_record(9)
    record.parameters["method"] = "legacy_grid"

    entry = build_journal("Old", [record]).days[0].entries[0]

    assert entry.body == "Generated 0 muon sites."


def test_journal_describes_cluster_positions_staging_and_removal() -> None:
    ids = [uuid4(), uuid4()]
    labels = {"site_labels": ["Random Site 1", "Random Site 2"]}
    records = [
        _muon_sites_record(
            "cluster_candidate_positions",
            parameters={**labels, "anchor_fractional": [0.25, 0.25, 0.25]},
            input_entity_ids=ids,
            output_entity_ids=ids,
        ),
        _muon_sites_record(
            "stage_muon_relaxation_sites",
            parameters=labels,
            input_entity_ids=ids,
            output_entity_ids=ids,
        ),
        _muon_sites_record("delete_candidates", parameters=labels, input_entity_ids=ids),
    ]

    entries = build_journal("Starts", records).days[0].entries

    assert [(entry.title, entry.body) for entry in entries] == [
        (
            "Muon starts clustered",
            "Moved 2 generated muon sites to the symmetry-equivalent positions nearest the "
            "anchor (0.2500, 0.2500, 0.2500): Random Site 1, Random Site 2. The original "
            "positions are kept in the project's provenance.",
        ),
        (
            "Muon sites staged for relaxation",
            "Staged 2 generated muon sites for relaxation: Random Site 1, Random Site 2.",
        ),
        (
            "Generated muon sites removed",
            "Removed 2 generated muon sites: Random Site 1, Random Site 2.",
        ),
    ]


def test_journal_describes_workspace_creation_and_grouping() -> None:
    records = [
        _muon_sites_record(
            "create_analysis_workspace",
            parameters={
                "workspace_name": "Mu+ sites",
                "charge_state_label": "positive muon",
                "relaxation_count": 6,
            },
        ),
        _muon_sites_record(
            "group_relaxed_sites",
            parameters={
                "workspace_name": "Mu+ sites",
                "site_count": 6,
                "distinct_site_count": 2,
                "grouping_summary": "Connected components within 0.25 Å",
            },
        ),
    ]

    entries = build_journal("Analysis", records).days[0].entries

    assert [(entry.title, entry.body) for entry in entries] == [
        ("Analysis workspace created", "Created 'Mu+ sites' with 6 positive muon relaxations."),
        (
            "Relaxed sites grouped",
            "Grouped 6 relaxed sites in 'Mu+ sites' into 2 distinct sites: Connected "
            "components within 0.25 Å.",
        ),
    ]


def test_a_cluster_positions_entry_names_the_run_only_when_the_starts_had_one() -> None:
    record = _muon_sites_record(
        "cluster_candidate_positions",
        parameters={
            "site_labels": ["Random Site 1"],
            "anchor_fractional": [0.5, 0.5, 0.5],
            "generation_run_ids": [str(uuid4())],
        },
        output_entity_ids=[uuid4()],
    )

    body = build_journal("Starts", [record]).days[0].entries[0].body

    assert body.endswith(
        "The original positions are kept in the project's provenance and with the generation run."
    )


def _job_record(action: str, stage: WorkflowStage, hour: int = 9, **parameters) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, hour, 0, tzinfo=UTC),
        action=action,
        workflow_stage=stage,
        parameters=parameters,
        job_id=None if action == "submit_jobs" else uuid4(),
    )


def _batch(charge_state: str, hour: int) -> list[ProvenanceRecord]:
    """A muon-relaxation submission of two sites and one completion, in ``charge_state``."""

    jobs = [
        {"label": label, "row_label": label, "charge_state": charge_state}
        for label in ("Attached-F Site 1", "Random Site 1")
    ]
    return [
        _job_record(
            "submit_jobs",
            WorkflowStage.MUON_RELAXATION,
            hour,
            execution_target="local",
            job_count=2,
            job_labels=[job["label"] for job in jobs],
            charge_state=charge_state,
            jobs=jobs,
        ),
        _job_record(
            "job_completed",
            WorkflowStage.MUON_RELAXATION,
            hour + 1,
            job_label="Random Site 1",
            row_label="Random Site 1",
            charge_state=charge_state,
            execution_target="local",
        ),
    ]


def test_positive_muon_and_muonium_batches_read_differently() -> None:
    entries = build_journal("Mu", [*_batch("positive_muon", 9), *_batch("muonium", 12)])
    texts = [(entry.title, entry.body) for entry in entries.days[0].entries]

    assert texts == [
        (
            "Muon relaxations launched: positive muon",
            "Submitted 2 jobs to run on this computer: Attached-F Site 1, Random Site 1.",
        ),
        (
            "Positive muon relaxation completed",
            "Positive muon relaxation Random Site 1 finished on this computer.",
        ),
        (
            "Muon relaxations launched: muonium",
            "Submitted 2 jobs to run on this computer: Attached-F Site 1, Random Site 1.",
        ),
        (
            "Muonium relaxation completed",
            "Muonium relaxation Random Site 1 finished on this computer.",
        ),
    ]


def test_a_batch_of_both_charge_states_names_both() -> None:
    record = _job_record(
        "submit_jobs",
        WorkflowStage.MUON_RELAXATION,
        execution_target="SCARF",
        job_count=2,
        job_labels=["Site 1", "Site 1"],
        jobs=[
            {"label": "Site 1", "row_label": "Site 1", "charge_state": "positive_muon"},
            {"label": "Site 1", "row_label": "Site 1", "charge_state": "muonium"},
        ],
    )

    entry = build_journal("Mu", [record]).days[0].entries[0]

    assert entry.title == "Muon relaxations launched: positive muon and muonium"
    assert (
        entry.body == "Submitted 2 jobs to run on SCARF: Site 1 (positive muon), Site 1 (muonium)."
    )


def test_a_batch_from_several_workflows_keeps_the_general_title() -> None:
    record = _job_record(
        "submit_jobs",
        None,
        execution_target="local",
        job_count=2,
        job_labels=["pristine_relax_relax_1", "Site 1"],
        jobs=[
            {"label": "pristine_relax_relax_1", "row_label": "relax_1"},
            {"label": "Site 1", "row_label": "Site 1", "charge_state": "positive_muon"},
        ],
    )

    entry = build_journal("Mixed", [record]).days[0].entries[0]

    assert entry.title == "Calculations launched"
    assert entry.body == "Submitted 2 jobs to run on this computer: relax_1, Site 1."


def test_an_unknown_charge_state_is_not_named_and_a_nameless_job_starts_a_sentence() -> None:
    record = _job_record(
        "job_completed",
        WorkflowStage.MUON_RELAXATION,
        charge_state="negative",
        execution_target="local",
    )

    entry = build_journal("Odd", [record]).days[0].entries[0]

    assert entry.title == "Calculation completed"
    assert entry.body == "A calculation finished on this computer."


@pytest.mark.parametrize(
    ("action", "title"),
    [
        ("job_failed", "Muonium relaxation failed"),
        ("job_stopped", "Muonium relaxation stopped"),
        ("job_cancelled", "Muonium relaxation cancelled"),
    ],
)
def test_every_outcome_of_a_muon_relaxation_names_its_charge_state(action: str, title: str) -> None:
    record = _job_record(
        action,
        WorkflowStage.MUON_RELAXATION,
        job_label="Random Site 2",
        row_label="Random Site 2",
        charge_state="muonium",
        execution_target="SCARF",
    )

    entry = build_journal("Mu", [record]).days[0].entries[0]

    assert entry.title == title
    assert entry.body == "Muonium relaxation Random Site 2 finished on SCARF."


def test_a_job_is_described_by_its_workflow_and_row_label() -> None:
    records = [
        _job_record(
            "job_completed",
            WorkflowStage.PRISTINE_RELAXATION,
            job_label="pristine_relax_relax_1",
            row_label="relax_1",
            execution_target="local",
        ),
        _job_record(
            "job_reset",
            WorkflowStage.UEP,
            10,
            job_label="uep_source_01",
            row_label="source_01",
            cleaned_output_directory=False,
        ),
        _job_record(
            "job_continuation",
            WorkflowStage.QUANTUM_CORRECTIONS,
            11,
            job_label="quantum_phonon_Attached-F_Site_1",
            row_label="Attached-F Site 1",
        ),
    ]

    bodies = [entry.body for entry in build_journal("Rows", records).days[0].entries]

    assert bodies == [
        "Pristine relaxation relax_1 finished on this computer.",
        "Reset UEP calculation source_01 back to the ready-to-submit state.",
        "Queued phonon calculation for Attached-F Site 1 for a continuation run.",
    ]


def test_old_job_records_without_a_charge_state_or_row_label_read_as_before() -> None:
    records = [
        _job_record(
            "submit_jobs",
            WorkflowStage.MUON_RELAXATION,
            execution_target="local",
            job_count=2,
            job_labels=["Attached-F Site 1", "Random Site 1"],
        ),
        _job_record(
            "job_completed",
            WorkflowStage.PRISTINE_RELAXATION,
            10,
            job_label="pristine_relax_relax_1",
            calc_type="castep_pristine_relaxation",
            execution_target="local",
        ),
    ]

    entries = build_journal("Old", records).days[0].entries

    assert [(entry.title, entry.body) for entry in entries] == [
        (
            "Calculations launched",
            "Submitted 2 jobs to run on this computer: Attached-F Site 1, Random Site 1.",
        ),
        ("Calculation completed", "pristine_relax_relax_1 finished on this computer."),
    ]


def test_a_single_job_quotes_its_energy_to_a_tenth_of_a_mev_and_its_force_to_three_figures() -> (
    None
):
    record = _job_record(
        "job_completed",
        WorkflowStage.PRISTINE_RELAXATION,
        job_label="relax_1",
        execution_target="local",
        result_total_energy=-5621.874312345,
        result_max_force=0.012300682907871417,
        warnings_count=2,
    )

    body = build_journal("Precision", [record]).days[0].entries[0].body

    assert body == (
        "relax_1 finished on this computer. Total energy -5621.8743 eV. Max force 0.0123 eV/Å. "
        "Recorded 2 warnings."
    )


def test_an_adopted_default_names_the_criterion_it_converged_under() -> None:
    records = [
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
            action="set_recommended_cutoff",
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            parameters={
                "recommended_cutoff_energy": 600.0,
                "recommended_cutoff_method": "converged",
                "recommended_cutoff_energy_threshold": 0.001,
                "recommended_cutoff_force_threshold": 0.05,
                "recommended_cutoff_xc_functional": "PBE",
            },
        ),
        ProvenanceRecord(
            timestamp=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
            action="set_recommended_kpoint_grid",
            workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
            parameters={
                "recommended_kpoint_grid": [4, 4, 6],
                "recommended_kpoint_method": "delta-threshold",
                "recommended_kpoint_energy_threshold": 0.001,
                "recommended_kpoint_force_threshold": 0.05,
                "recommended_kpoint_stress_threshold": 0.1,
            },
        ),
    ]

    bodies = [entry.body for entry in build_journal("Defaults", records).days[0].entries]

    assert bodies == [
        "Set the project default cutoff to 600 eV (converged with the PBE functional): energy "
        "within 0.001 eV/atom and max force within 0.05 eV/Å of the finest cutoff.",
        "Set the project default k-point grid to 4x4x6: energy within 0.001 eV/atom, max force "
        "within 0.05 eV/Å and max stress within 0.1 GPa of the densest grid.",
    ]
    assert not any("delta" in body for body in bodies)


def _configure_record(action: str, stage: WorkflowStage, configuration: dict) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action=action,
        workflow_stage=stage,
        parameters={"configuration": configuration},
    )


def test_journal_shows_muon_relaxation_quantum_and_barrier_configuration() -> None:
    rows = [{"label": f"Site {n}", "charge_state": "positive_muon"} for n in range(3)]
    rows.append({"label": "Site 0", "charge_state": "muonium"})
    records = [
        _configure_record(
            "configure_muon_relaxation", WorkflowStage.MUON_RELAXATION, {"rows": rows}
        ),
        _configure_record(
            "configure_quantum_corrections",
            WorkflowStage.QUANTUM_CORRECTIONS,
            {"sources": [{"key": "a"}]},
        ),
        _configure_record(
            "configure_energy_barriers",
            WorkflowStage.ENERGY_BARRIERS,
            {"method": "NEB", "sources": [{"key": "a"}, {"key": "b"}]},
        ),
    ]

    entries = build_journal("Configured", records).days[0].entries

    assert [(entry.title, entry.body, entry.workflow_label) for entry in entries] == [
        (
            "Muon relaxations configured",
            "Prepared 4 muon relaxations for later submission (3 positive muon, 1 muonium).",
            "Muon Relaxations",
        ),
        (
            "Quantum Corrections configured",
            "Prepared phonon calculations for 1 relaxed muon site.",
            "Quantum Corrections",
        ),
        (
            "Energy Barriers configured",
            "Prepared NEB calculations for 2 paths.",
            "Energy Barriers",
        ),
    ]


def test_journal_names_the_new_active_host_and_reads_older_records_from_their_description() -> None:
    new = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action="set_active_host_structure",
        parameters={
            "new_host_structure_label": "CoF2 relaxed",
            "previous_host_structure_label": "CoF2",
        },
    )
    old = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 10, 0, tzinfo=UTC),
        action="set_active_host_structure",
        description="Changed active host from 'CoF2 relaxed' to 'CoF2'",
        parameters={"new_host_structure_id": str(uuid4())},
    )

    entries = build_journal("Host", [new, old]).days[0].entries

    assert [(entry.title, entry.body) for entry in entries] == [
        (
            "Active host changed",
            "Made 'CoF2 relaxed' the active host structure in place of 'CoF2'.",
        ),
        ("Active host changed", "Changed active host from 'CoF2 relaxed' to 'CoF2'."),
    ]


def test_journal_shows_a_calculation_dismissed_from_the_dashboard() -> None:
    record = _job_record(
        "job_dismissed",
        WorkflowStage.MUON_RELAXATION,
        job_label="Random Site 3",
        row_label="Random Site 3",
        charge_state="positive_muon",
        execution_target="local",
    )

    entry = build_journal("Dismissed", [record]).days[0].entries[0]

    assert (entry.title, entry.body, entry.stamp_kind) == (
        "Calculation dismissed",
        "Dismissed positive muon relaxation Random Site 3 from the Job Dashboard.",
        "milestone",
    )


def test_stamp_label_names_each_stamp_kind() -> None:
    assert [stamp_label(kind) for kind in ("milestone", "success", "warning", "danger", "x")] == [
        "Milestone",
        "Completed",
        "Attention",
        "Failed",
        "Entry",
    ]
