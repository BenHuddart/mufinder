"""Tests for journal derivation and export from provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import ProvenanceRecord
from mufinder.services.journal import (
    build_journal,
    export_journal,
    render_journal_markdown,
    render_journal_text,
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
        f"Prepared a cutoff convergence sweep with {count} cutoff value(s)." for count in (2, 3, 4)
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
