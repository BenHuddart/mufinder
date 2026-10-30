"""Tests for journal derivation and export from provenance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
    entry_details,
    entry_meta_text,
    export_journal,
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
    ("action", "title", "body"),
    [
        ("job_failed", "Muonium relaxation failed", "failed on SCARF"),
        ("job_stopped", "Muonium relaxation stopped", "stopped on SCARF"),
        ("job_cancelled", "Muonium relaxation cancelled", "was cancelled on SCARF"),
    ],
)
def test_every_outcome_of_a_muon_relaxation_names_its_charge_state(
    action: str, title: str, body: str
) -> None:
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
    assert entry.body == f"Muonium relaxation Random Site 2 {body}."


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


def _at(hour: int, day: int = 21) -> datetime:
    return datetime(2026, 4, day, hour, 0, tzinfo=UTC)


def _submission(job_ids: list, *, hour: int = 9, day: int = 21) -> ProvenanceRecord:
    """A positive-muon relaxation batch of ``Site n``, as the service records it."""

    return ProvenanceRecord(
        timestamp=_at(hour, day),
        action="submit_jobs",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        parameters={
            "execution_target": "local",
            "job_count": len(job_ids),
            "job_labels": [f"Site {n}" for n in range(1, len(job_ids) + 1)],
            "charge_state": "positive_muon",
            "jobs": [
                {
                    "job_id": str(job_id),
                    "label": f"Site {n}",
                    "row_label": f"Site {n}",
                    "charge_state": "positive_muon",
                }
                for n, job_id in enumerate(job_ids, start=1)
            ],
        },
        output_entity_ids=list(job_ids),
    )


def _outcome(
    action: str, job_id, *, label: str = "Site 1", hour: int = 10, day: int = 21, **parameters
) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp=_at(hour, day),
        action=action,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        parameters={
            "job_label": label,
            "row_label": label,
            "charge_state": "positive_muon",
            "execution_target": "local",
            **parameters,
        },
        job_id=job_id,
    )


def _titles(journal) -> list[str]:
    return [entry.title for day in journal.days for entry in day.entries]


def test_a_batch_of_ten_completions_is_one_entry_with_a_row_per_job() -> None:
    job_ids = [uuid4() for _ in range(10)]
    records = [_submission(job_ids)] + [
        _outcome(
            "job_completed",
            job_id,
            label=f"Site {n}",
            result_total_energy=-100.0 - n,
            result_max_force=0.01,
            warnings_count=0,
        )
        for n, job_id in enumerate(job_ids, start=1)
    ]

    journal = build_journal("Batch", records)
    entries = journal.days[0].entries

    assert len(journal.days) == 1
    assert len(entries) == 1
    assert entries[0].title == "Muon relaxations launched: positive muon"
    assert entries[0].stamp_kind == "milestone"
    assert entries[0].body == (
        "Submitted 10 jobs to run on this computer: Site 1, Site 2, Site 3, and 7 more. "
        "10 of 10 completed."
    )
    assert [row.label for row in entries[0].job_rows] == [f"Site {n}" for n in range(1, 11)]
    assert {row.status for row in entries[0].job_rows} == {"completed"}
    assert entries[0].job_rows[1].total_energy == -102.0
    assert entries[0].job_rows[1].max_force == 0.01
    assert entries[0].job_rows[1].warnings_count == 0
    assert entries[0].job_rows[1].failure_reason is None


def test_a_mixed_batch_keeps_a_card_for_each_failure_with_its_reason() -> None:
    job_ids = [uuid4() for _ in range(3)]
    records = [
        _submission(job_ids),
        _outcome("job_completed", job_ids[0], label="Site 1"),
        _outcome(
            "job_failed",
            job_ids[1],
            label="Site 2",
            hour=11,
            failure_reason="walltime",
            error_message="Slurm reported TIMEOUT",
        ),
        _outcome("job_completed", job_ids[2], label="Site 3", hour=12),
    ]

    entries = build_journal("Mixed", records).days[0].entries

    assert [entry.title for entry in entries] == [
        "Muon relaxations launched: positive muon",
        "Positive muon relaxation failed",
    ]
    assert entries[0].body.endswith("Site 1, Site 2, Site 3. 2 completed, 1 failed.")
    assert entries[1].stamp_kind == "danger"
    assert entries[1].linked_job_id == job_ids[1]
    assert entries[1].body == (
        "Positive muon relaxation Site 2 failed on this computer. "
        "The scheduler stopped the job when it reached its walltime limit. "
        "Slurm reported TIMEOUT."
    )
    failed_row = entries[0].job_rows[1]
    assert failed_row.status == "failed"
    assert failed_row.failure_reason == (
        "The scheduler stopped the job when it reached its walltime limit. Slurm reported TIMEOUT."
    )


@pytest.mark.parametrize(
    ("action", "title", "stamp"),
    [
        ("job_stopped", "Positive muon relaxation stopped", "warning"),
        ("job_cancelled", "Positive muon relaxation cancelled", "warning"),
    ],
)
def test_stopped_and_cancelled_jobs_keep_their_own_cards(
    action: str, title: str, stamp: str
) -> None:
    job_id = uuid4()

    entries = (
        build_journal("Own", [_submission([job_id]), _outcome(action, job_id)]).days[0].entries
    )

    assert [entry.title for entry in entries] == [
        "Muon relaxations launched: positive muon",
        title,
    ]
    assert entries[1].stamp_kind == stamp
    assert entries[0].body.endswith(f"1 of 1 {title.rsplit(' ', 1)[1]}.")


def test_a_batch_still_running_says_how_far_it_has_got() -> None:
    job_ids = [uuid4() for _ in range(4)]
    submitted = _submission(job_ids)

    def body(*outcomes: ProvenanceRecord) -> str:
        return build_journal("Running", [submitted, *outcomes]).days[0].entries[0].body

    assert body().endswith("Site 3, and 1 more.")
    assert body(_outcome("job_completed", job_ids[0])).endswith(" 1 of 4 completed.")
    assert body(
        _outcome("job_completed", job_ids[0]),
        _outcome("job_failed", job_ids[1], label="Site 2"),
    ).endswith(" 2 of 4 finished: 1 completed, 1 failed.")

    rows = build_journal("Running", [submitted]).days[0].entries[0].job_rows
    assert {row.status for row in rows} == {"submitted"}


def test_a_batch_finishing_on_a_later_day_adds_a_finished_line_under_that_day() -> None:
    job_ids = [uuid4(), uuid4()]
    records = [
        _submission(job_ids),
        _outcome("job_completed", job_ids[0], hour=22),
        ProvenanceRecord(
            timestamp=_at(8, 22),
            action="set_recommended_xc_functional",
            parameters={"recommended_xc_functional": "PBE"},
        ),
        _outcome("job_completed", job_ids[1], label="Site 2", hour=9, day=22),
    ]

    journal = build_journal("Later", records)

    assert [day.date_label for day in journal.days] == [
        "Tuesday 21 April 2026",
        "Wednesday 22 April 2026",
    ]
    assert _titles(journal) == [
        "Muon relaxations launched: positive muon",
        "Default XC functional set",
        "Finished: 2 of 2 completed",
    ]
    launch = journal.days[0].entries[0]
    assert launch.body.endswith("2 of 2 completed.")
    line = journal.days[1].entries[-1]
    assert line.stamp_kind == "success"
    assert line.workflow_label == "Muon Relaxations"
    assert line.body == "Muon relaxations launched: positive muon on Tuesday 21 April 2026."
    assert line.job_rows == ()


def test_a_finished_line_with_failures_asks_for_attention_after_their_card() -> None:
    job_ids = [uuid4(), uuid4()]
    records = [
        _submission(job_ids),
        _outcome("job_completed", job_ids[0], hour=12),
        _outcome("job_failed", job_ids[1], label="Site 2", hour=9, day=22),
    ]

    entries = build_journal("Later", records).days[1].entries

    assert [(entry.title, entry.stamp_kind) for entry in entries] == [
        ("Positive muon relaxation failed", "danger"),
        ("Finished: 1 completed, 1 failed", "warning"),
    ]


def test_no_finished_line_while_a_job_is_outstanding_or_on_the_launch_day() -> None:
    job_ids = [uuid4(), uuid4()]

    unfinished = build_journal(
        "Open", [_submission(job_ids), _outcome("job_completed", job_ids[0], day=22)]
    )
    same_day = build_journal(
        "Same",
        [
            _submission(job_ids),
            _outcome("job_completed", job_ids[0]),
            _outcome("job_completed", job_ids[1], label="Site 2", hour=11),
        ],
    )

    assert _titles(unfinished) == ["Muon relaxations launched: positive muon"]
    assert _titles(same_day) == ["Muon relaxations launched: positive muon"]


def test_an_orphan_completion_renders_on_its_own() -> None:
    orphan = _outcome("job_completed", uuid4(), result_total_energy=-1.5)

    entry = build_journal("Orphan", [_submission([uuid4()]), orphan]).days[0].entries[1]

    assert entry.title == "Positive muon relaxation completed"
    assert entry.stamp_kind == "success"
    assert "Total energy -1.5000 eV." in entry.body


def test_an_old_submission_without_job_ids_folds_nothing() -> None:
    job_id = uuid4()
    old_submission = ProvenanceRecord(
        timestamp=_at(9),
        action="submit_jobs",
        workflow_stage=WorkflowStage.UEP,
        parameters={"execution_target": "local", "job_count": 1, "job_labels": ["uep_a"]},
        output_entity_ids=[job_id],
    )
    old_completion = ProvenanceRecord(
        timestamp=_at(10),
        action="job_completed",
        workflow_stage=WorkflowStage.UEP,
        parameters={"job_label": "uep_a", "execution_target": "local"},
        job_id=job_id,
    )

    entries = build_journal("Old", [old_submission, old_completion]).days[0].entries

    assert [(entry.title, entry.body) for entry in entries] == [
        ("Calculations launched", "Submitted 1 job to run on this computer: uep_a."),
        ("Calculation completed", "uep_a finished on this computer."),
    ]
    assert entries[0].job_rows == ()


def test_a_recorded_failure_reason_the_journal_does_not_know_is_left_out() -> None:
    record = _outcome("job_failed", uuid4(), failure_reason="gremlins")

    body = build_journal("Odd", [record]).days[0].entries[0].body

    assert body == "Positive muon relaxation Site 1 failed on this computer."


def test_a_resubmitted_job_counts_towards_its_latest_batch() -> None:
    job_ids = [uuid4(), uuid4()]
    records = [
        _submission(job_ids),
        _outcome("job_completed", job_ids[0]),
        _outcome("job_failed", job_ids[1], label="Site 2", hour=11),
        _outcome("job_reset", job_ids[1], label="Site 2", hour=12),
        _submission([job_ids[1]], hour=13),
        _outcome("job_completed", job_ids[1], label="Site 2", hour=14),
    ]

    entries = build_journal("Retry", records).days[0].entries

    assert [entry.title for entry in entries] == [
        "Muon relaxations launched: positive muon",
        "Positive muon relaxation failed",
        "Calculation reset",
        "Muon relaxations launched: positive muon",
    ]
    assert entries[0].body.endswith("1 completed, 1 failed.")
    assert entries[3].body.endswith("1 of 1 completed.")


def test_an_outcome_after_a_continuation_with_no_new_submission_stands_alone() -> None:
    job_id = uuid4()
    records = [
        _submission([job_id]),
        _outcome("job_stopped", job_id),
        _outcome("job_continuation", job_id, hour=11),
        _outcome("job_completed", job_id, hour=12),
    ]

    entries = build_journal("Continued", records).days[0].entries

    assert [entry.title for entry in entries] == [
        "Muon relaxations launched: positive muon",
        "Positive muon relaxation stopped",
        "Calculation continued",
        "Positive muon relaxation completed",
    ]
    assert entries[0].body.endswith("1 of 1 stopped.")


def test_a_stopped_job_that_reappears_and_completes_reads_as_completed() -> None:
    job_id = uuid4()
    records = [
        _submission([job_id]),
        _outcome("job_stopped", job_id),
        _outcome("job_completed", job_id, hour=11),
    ]

    entries = build_journal("Reappeared", records).days[0].entries

    assert [entry.title for entry in entries] == [
        "Muon relaxations launched: positive muon",
        "Positive muon relaxation stopped",
    ]
    assert entries[0].body.endswith("1 of 1 completed.")
    assert entries[0].job_rows[0].status == "completed"


def test_a_late_failure_keeps_its_card_but_does_not_reopen_a_completed_job() -> None:
    job_id = uuid4()
    records = [
        _submission([job_id]),
        _outcome("job_completed", job_id),
        _outcome("job_failed", job_id, hour=11, error_message="late failure"),
    ]

    entries = build_journal("Late", records).days[0].entries

    assert [entry.title for entry in entries] == [
        "Muon relaxations launched: positive muon",
        "Positive muon relaxation failed",
    ]
    assert entries[0].body.endswith("1 of 1 completed.")
    assert entries[0].job_rows[0].status == "completed"


def test_a_walltime_stop_does_not_say_its_reason_twice() -> None:
    summary = "The scheduler stopped the job when it reached its walltime limit."
    record = _outcome("job_stopped", uuid4(), failure_reason="walltime", error_message=summary)

    body = build_journal("Walltime", [record]).days[0].entries[0].body

    assert body == f"Positive muon relaxation Site 1 stopped on this computer. {summary}"


def test_a_submission_with_a_job_lacking_an_id_forms_no_batch() -> None:
    job_id = uuid4()
    submission = _submission([job_id, uuid4()])
    del submission.parameters["jobs"][1]["job_id"]
    records = [submission, _outcome("job_completed", job_id)]

    entries = build_journal("Partial", records).days[0].entries

    assert [entry.title for entry in entries] == [
        "Muon relaxations launched: positive muon",
        "Positive muon relaxation completed",
    ]
    assert entries[0].job_rows == ()


def test_a_batch_of_both_charge_states_labels_its_rows_with_them() -> None:
    job_ids = [uuid4(), uuid4()]
    record = _job_record(
        "submit_jobs",
        WorkflowStage.MUON_RELAXATION,
        execution_target="local",
        job_count=2,
        job_labels=["Site 1", "Site 1"],
        jobs=[
            {"job_id": str(job_ids[0]), "row_label": "Site 1", "charge_state": "positive_muon"},
            {"job_id": str(job_ids[1]), "row_label": "Site 1", "charge_state": "muonium"},
        ],
    )

    entry = build_journal("Mu", [record]).days[0].entries[0]

    assert [row.label for row in entry.job_rows] == ["Site 1 (positive muon)", "Site 1 (muonium)"]


def test_exports_match_the_grouped_entries(tmp_path: Path) -> None:
    job_ids = [uuid4(), uuid4()]
    journal = build_journal(
        "Exported",
        [
            _submission(job_ids),
            _outcome("job_completed", job_ids[0], result_total_energy=-12.3456),
            _outcome(
                "job_failed", job_ids[1], label="Site 2", error_message="SCF did not converge"
            ),
        ],
    )
    text_path = tmp_path / "journal.txt"
    markdown_path = tmp_path / "journal.md"

    export_journal(journal, text_path, format_name="text")
    export_journal(journal, markdown_path, format_name="markdown")

    text = text_path.read_text(encoding="utf-8")
    markdown = markdown_path.read_text(encoding="utf-8")
    assert text == render_journal_text(journal)
    assert markdown == render_journal_markdown(journal)
    for content in (text, markdown):
        assert "1 completed, 1 failed." in content
        assert "Positive muon relaxation failed" in content
        assert "SCF did not converge." in content
        assert "Positive muon relaxation completed" not in content
        assert "-12.3456" not in content


def _dismissal(job_id, label: str, when: datetime, **parameters) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp=when,
        action="job_dismissed",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        parameters={"job_label": label, "row_label": label, **parameters},
        output_entity_ids=[job_id],
        job_id=job_id,
    )


def test_dismissals_in_a_row_fold_into_one_card() -> None:
    job_ids = [uuid4() for _ in range(5)]
    records = [
        _dismissal(job_id, f"Site {n}", _at(9) + timedelta(milliseconds=n))
        for n, job_id in enumerate(job_ids, start=1)
    ]

    entries = build_journal("Dismissed", records).days[0].entries

    assert len(entries) == 1
    assert entries[0].title == "Calculations dismissed"
    assert entries[0].stamp_kind == "milestone"
    assert entries[0].workflow_label == "Muon Relaxations"
    assert entries[0].body == (
        "Dismissed 5 calculations from the Job Dashboard: Site 1, Site 2, Site 3, and 2 more."
    )
    assert entries[0].linked_entity_ids == tuple(job_ids)
    assert entries[0].linked_job_id is None


def test_dismissals_split_at_another_entry_and_at_a_new_day() -> None:
    other = ProvenanceRecord(
        timestamp=_at(10),
        action="set_recommended_xc_functional",
        parameters={"recommended_xc_functional": "PBE"},
    )
    records = [
        _dismissal(uuid4(), "Site 1", _at(9)),
        other,
        _dismissal(uuid4(), "Site 2", _at(11)),
        _dismissal(uuid4(), "Site 3", _at(9, day=22)),
    ]

    journal = build_journal("Split", records)

    assert _titles(journal) == [
        "Calculation dismissed",
        "Default XC functional set",
        "Calculation dismissed",
        "Calculation dismissed",
    ]
    assert len(journal.days) == 2


def test_dismissals_from_several_workflows_name_no_single_workflow() -> None:
    first = _dismissal(uuid4(), "Site 1", _at(9), charge_state="positive_muon")
    second = _dismissal(uuid4(), "Site 1", _at(9), charge_state="muonium")
    third = _dismissal(uuid4(), "relax_1", _at(9))
    third.workflow_stage = WorkflowStage.PRISTINE_RELAXATION

    entry = build_journal("Mixed", [first, second, third]).days[0].entries[0]

    assert entry.workflow_label is None
    assert entry.body == (
        "Dismissed 3 calculations from the Job Dashboard: Site 1 (positive muon), "
        "Site 1 (muonium), relax_1."
    )


def _only_entry(*records: ProvenanceRecord):
    return build_journal("Details", records).days[0].entries[0]


def _details_record(action: str, stage: WorkflowStage | None = None, **parameters: object):
    return ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 30, 15, tzinfo=UTC),
        action=action,
        workflow_stage=stage,
        parameters=parameters,
    )


def test_every_entry_details_give_the_exact_local_time() -> None:
    entry = _only_entry(_details_record("import_structure", WorkflowStage.MUON_GENERATION))
    local = entry.timestamp.astimezone()

    assert local.strftime("%A %d %B %Y at %H:%M:%S") in entry_details(entry).exact_time
    assert entry_meta_text(entry) == f"Muon Sites · {local.strftime('%H:%M')}"


def test_an_entry_without_a_workflow_has_only_its_time_in_the_header() -> None:
    entry = _only_entry(_details_record("set_recommended_xc_functional"))

    assert entry_meta_text(entry) == entry.timestamp.astimezone().strftime("%H:%M")
    assert entry.route is None
    assert entry.open_label is None


def test_convergence_details_give_shared_parameters_and_the_sweep() -> None:
    entry = _only_entry(
        _details_record(
            "configure_convergence_cutoff",
            WorkflowStage.CONVERGENCE_CUTOFF,
            configuration={
                "shared_params": {"spin_polarized": False, "xc_functional": "PBE"},
                "sweep_values": [300.0, 400.0, 800.0],
            },
            simulation_cell_mode="unit_cell",
        )
    )

    assert entry_details(entry).lines == (
        "Unit cell · not spin polarized · PBE",
        "3 cutoff values: 300 eV to 800 eV",
    )
    assert entry.route == "convergence_cutoff"
    assert entry.open_label == "Open in Cutoff Convergence"


def test_kpoint_sweep_details_name_the_grids_and_legacy_cutoff_records_still_read() -> None:
    kpoint = _only_entry(
        _details_record(
            "configure_convergence_kpoint",
            WorkflowStage.CONVERGENCE_KPOINT,
            configuration={"sweep_values": [[1, 1, 1], [4, 4, 6]]},
        )
    )
    legacy = _only_entry(
        _details_record(
            "configure_convergence_cutoff",
            WorkflowStage.CONVERGENCE_CUTOFF,
            configuration={"cutoff_energies": [500.0]},
        )
    )

    assert entry_details(kpoint).lines == ("2 grid values: 1×1×1 to 4×4×6",)
    assert entry_details(legacy).lines == ("1 cutoff value: 500 eV",)


def test_configuration_details_collapse_a_long_row_list_and_keep_only_shared_values() -> None:
    rows = [
        {
            "label": f"Site {n}",
            "charge_state": "positive_muon",
            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            "parameter_values": {
                "spin_polarized": True,
                "xc_functional": "PBE",
                "cut_off_energy": {"unit": "eV", "value": 600.0},
                "kpoint_mp_grid": [4, 4, 6],
                "task": "GEOMETRYOPTIMIZATION" if n else "SINGLEPOINT",
            },
        }
        for n in range(5)
    ]
    entry = _only_entry(
        _details_record(
            "configure_muon_relaxation",
            WorkflowStage.MUON_RELAXATION,
            configuration={"rows": rows},
        )
    )

    assert entry_details(entry).lines == (
        "Supercell 2×2×2 · spin polarized · PBE · 600 eV · 4×4×6",
        "5 rows: Site 0, Site 1, Site 2, and 2 more",
    )


def test_configuration_details_omit_a_value_the_rows_disagree_on_and_show_a_skewed_cell() -> None:
    entry = _only_entry(
        _details_record(
            "configure_pristine_relaxation",
            WorkflowStage.PRISTINE_RELAXATION,
            configuration={
                "rows": [
                    {
                        "label": "a",
                        "supercell_matrix": [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                        "parameter_values": {"xc_functional": "PBE"},
                    },
                    {
                        "label": "b",
                        "supercell_matrix": [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
                        "parameter_values": {"xc_functional": "LDA"},
                    },
                ]
            },
        )
    )

    assert entry_details(entry).lines == ("Supercell [1 1 0; 0 1 0; 0 0 1]", "2 rows: a, b")


def test_quantum_and_barrier_details_count_their_sources() -> None:
    entry = _only_entry(
        _details_record(
            "configure_quantum_corrections",
            WorkflowStage.QUANTUM_CORRECTIONS,
            configuration={"sources": [{"label": "Site 1"}]},
        )
    )

    assert entry_details(entry).lines == ("1 source: Site 1",)
    assert entry.route == "quantum_corrections"


def test_adopted_default_details_give_thresholds_and_functional() -> None:
    entry = _only_entry(
        _details_record(
            "set_recommended_cutoff",
            WorkflowStage.CONVERGENCE_CUTOFF,
            recommended_cutoff_energy=600.0,
            recommended_cutoff_method="converged",
            recommended_cutoff_energy_threshold=0.001,
            recommended_cutoff_force_threshold=0.05,
            recommended_cutoff_xc_functional="PBE",
        )
    )

    assert entry_details(entry).lines == (
        "Energy within 0.001 eV/atom · Max force within 0.05 eV/Å · Functional PBE",
    )


def test_generation_details_give_inputs_and_seed() -> None:
    entry = _only_entry(
        _details_record(
            "generate_candidates",
            WorkflowStage.MUON_GENERATION,
            generation_run={
                "id": str(uuid4()),
                "method": "random",
                "parameters": {"min_muon_muon": 1.5, "min_muon_atom": 1.0},
                "seed": 418273,
                "site_ids": [str(uuid4())],
            },
        )
    )

    (line,) = entry_details(entry).lines
    assert "at least 1.5 Å from the other starts" in line
    assert "The random seed was 418273." in line
    assert entry.route == "muon_candidates"


def test_generation_details_of_an_older_record_are_only_the_time() -> None:
    entry = _only_entry(generation_record(9))

    assert entry_details(entry).lines == ()


def test_legacy_import_details_list_skipped_runs_and_conflicts() -> None:
    entry = _only_entry(
        _details_record(
            "import_legacy_relaxations",
            WorkflowStage.MUON_RELAXATION,
            source_folder="/data/old",
            imported_sites=[{"label": "Site 1", "conflicts": ["CUT_OFF_ENERGY: two values."]}],
            skipped=[{"site_index": 3, "reason": "No output file."}],
        )
    )

    assert entry_details(entry).lines == (
        "Source folder /data/old",
        "Skipped run 3: No output file.",
        "Site 1: CUT_OFF_ENERGY: two values.",
    )


def _muon_batch(job_ids: list, *, num_processes: list[int]) -> list[ProvenanceRecord]:
    timestamp = datetime(2026, 4, 21, 9, 0, tzinfo=UTC)
    return [
        ProvenanceRecord(
            timestamp=timestamp,
            action="configure_muon_relaxation",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            parameters={
                "configuration": {
                    "rows": [
                        {
                            "label": "Site 1",
                            "charge_state": "positive_muon",
                            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                            "parameter_values": {"spin_polarized": True},
                        },
                        {
                            "label": "Site 2",
                            "charge_state": "positive_muon",
                            "supercell_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                            "parameter_values": {"spin_polarized": True},
                        },
                        {
                            "label": "Site 1",
                            "charge_state": "muonium",
                            "supercell_matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                            "parameter_values": {"spin_polarized": False},
                        },
                    ]
                }
            },
        ),
        ProvenanceRecord(
            timestamp=timestamp + timedelta(minutes=1),
            action="submit_jobs",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            parameters={
                "execution_target": "local",
                "job_count": len(job_ids),
                "charge_state": "positive_muon",
                "jobs": [
                    {
                        "job_id": str(job_id),
                        "label": f"Site {n + 1}",
                        "row_label": f"Site {n + 1}",
                        "charge_state": "positive_muon",
                        "num_processes": processes,
                    }
                    for n, (job_id, processes) in enumerate(zip(job_ids, num_processes))
                ],
            },
        ),
    ]


def _batch_outcome(action: str, job_id, minute: int, **parameters: object) -> ProvenanceRecord:
    return ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, minute, tzinfo=UTC),
        action=action,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        parameters=parameters,
        job_id=job_id,
    )


def test_batch_details_give_shared_parameters_and_energies_relative_to_the_minimum() -> None:
    job_ids = [uuid4(), uuid4()]
    records = [
        *_muon_batch(job_ids, num_processes=[8, 8]),
        _batch_outcome(
            "job_completed",
            job_ids[0],
            5,
            result_total_energy=-100.0,
            result_max_force=0.01234,
            warnings_count=2,
        ),
        _batch_outcome("job_completed", job_ids[1], 6, result_total_energy=-100.25),
    ]
    entry = build_journal("Batch", records).days[0].entries[1]

    assert entry.card_summary == "2 sites on this computer, 8 processes each · 2 of 2 completed"
    details = entry_details(entry)
    # The muonium row of Site 1 is not in this positive-muon batch.
    assert details.lines == ("Supercell 2×2×2 · spin polarized",)
    assert details.job_table.headers[0] == "Site"
    assert details.job_table.rows == (
        ("Site 1", "Completed", "250.0", "0.0123", "2"),
        ("Site 2", "Completed", "0.0", "—", "—"),
    )
    assert entry.open_label == "Open in Muon Relaxations"


def test_batch_details_of_a_failure_and_an_unfinished_job() -> None:
    job_ids = [uuid4(), uuid4()]
    records = [
        *_muon_batch(job_ids, num_processes=[8, 16]),
        _batch_outcome("job_failed", job_ids[0], 5, error_message="SCF did not converge"),
    ]
    entry = build_journal("Batch", records).days[0].entries[1]

    assert entry.card_summary == "2 sites on this computer, 8 to 16 processes · 1 of 2 failed"
    details = entry_details(entry)
    assert "Site 1: SCF did not converge." in details.lines
    assert details.job_table.rows == (
        ("Site 1", "Failed", "—", "—", "—"),
        ("Site 2", "Submitted", "—", "—", "—"),
    )


def test_a_batch_says_nothing_about_configuration_rows_it_did_not_run() -> None:
    job_id = uuid4()
    records = _muon_batch([job_id], num_processes=[8])
    records[1].parameters["jobs"][0]["row_label"] = "Another site"

    entry = build_journal("Batch", records).days[0].entries[1]

    assert entry_details(entry).lines == ()


def test_a_batch_without_a_configuration_has_only_its_table() -> None:
    job_id = uuid4()
    entry = _only_entry(
        _details_record(
            "submit_jobs",
            WorkflowStage.UEP,
            execution_target="local",
            jobs=[{"job_id": str(job_id), "label": "uep_1", "num_processes": 1}],
        )
    )

    assert entry.card_summary == "1 job on this computer, 1 process"
    assert entry_details(entry).lines == ()
    assert entry_details(entry).job_table.headers[0] == "Calculation"


def test_comparison_table_entry_opens_the_table_and_older_records_have_no_link() -> None:
    table_id = uuid4()
    linked = ProvenanceRecord(
        timestamp=datetime(2026, 4, 21, 9, 0, tzinfo=UTC),
        action="create_comparison_table",
        parameters={"table_label": "Comparison 1", "dft_structure_count": 1},
        output_entity_ids=[table_id, uuid4()],
    )
    entry = _only_entry(linked)

    assert entry.route == f"pristine_relaxation/comparison/{table_id}"
    assert entry.open_label == "Open in 'Comparison 1'"
    assert _only_entry(_details_record("create_comparison_table")).route is None


def test_an_imported_structure_opens_the_host_structure_page() -> None:
    entry = _only_entry(_details_record("import_structure", WorkflowStage.MUON_GENERATION))

    assert entry.route == "host_structure"
    assert entry.open_label == "Open in Host Structure"


def test_dismissals_from_several_workflows_have_no_link() -> None:
    records = [
        _details_record("job_dismissed", WorkflowStage.UEP, job_label="a"),
        _details_record("job_dismissed", WorkflowStage.PRISTINE_RELAXATION, job_label="b"),
    ]

    assert _only_entry(*records).route is None
    assert _only_entry(records[0]).route == "uep"


def test_exports_stay_the_short_summary(tmp_path: Path) -> None:
    job_id = uuid4()
    journal = build_journal(
        "Export",
        [
            *_muon_batch([job_id], num_processes=[8]),
            _batch_outcome("job_completed", job_id, 5, result_total_energy=-100.0),
        ],
    )

    for text in (render_journal_text(journal), render_journal_markdown(journal)):
        assert "Submitted 1 job to run on this computer: Site 1. 1 of 1 completed." in text
        assert "Supercell" not in text
        assert "meV" not in text
        assert "processes" not in text
