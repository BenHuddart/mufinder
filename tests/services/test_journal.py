"""Tests for journal derivation and export from provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import ProvenanceRecord
from mufinder.services.journal import build_journal, export_journal, render_journal_markdown, render_journal_text


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
        parameters={"recommended_cutoff_energy": 600.0, "recommended_cutoff_method": "energy_window"},
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

    assert [day.date_label for day in journal.days] == ["Tuesday 21 April 2026", "Wednesday 22 April 2026"]
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
