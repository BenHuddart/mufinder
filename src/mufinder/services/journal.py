"""Journal transformation and export helpers for provenance records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import ProvenanceRecord


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """One user-facing journal entry derived from provenance."""

    timestamp: datetime
    stamp_kind: str
    title: str
    body: str
    workflow_label: str | None
    linked_entity_ids: tuple[UUID, ...] = ()
    linked_job_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class JournalDay:
    """Journal entries grouped under one calendar-day heading."""

    date_label: str
    entries: tuple[JournalEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class JournalDocument:
    """Stable journal model for UI rendering and export."""

    project_name: str
    days: tuple[JournalDay, ...] = ()


_WORKFLOW_LABELS: dict[WorkflowStage, str] = {
    WorkflowStage.CONVERGENCE_CUTOFF: "Cutoff Convergence",
    WorkflowStage.CONVERGENCE_KPOINT: "K-Point Convergence",
    WorkflowStage.PRISTINE_RELAXATION: "Pristine Relaxation",
    WorkflowStage.UEP: "UEP",
    WorkflowStage.MUON_GENERATION: "Muon Sites",
    WorkflowStage.MUON_RELAXATION: "Muon Relaxations",
    WorkflowStage.SITE_ANALYSIS: "Site Analysis",
    WorkflowStage.MAGNETIC_FIELD: "Magnetic Fields",
    WorkflowStage.QUANTUM_CORRECTIONS: "Quantum Corrections",
    WorkflowStage.ENERGY_BARRIERS: "Energy Barriers",
}

_STAMP_LABELS = {
    "milestone": "Milestone",
    "success": "Completed",
    "warning": "Attention",
    "danger": "Failed",
}


def build_journal(project_name: str, records: Iterable[ProvenanceRecord]) -> JournalDocument:
    """Convert structured provenance into a curated journal view model."""

    entries = [entry for record in sorted(records, key=lambda item: item.timestamp) if (entry := _entry_from_record(record))]
    grouped: dict[str, list[JournalEntry]] = {}
    day_order: list[str] = []
    for entry in entries:
        day_key = entry.timestamp.astimezone().strftime("%A %d %B %Y")
        if day_key not in grouped:
            grouped[day_key] = []
            day_order.append(day_key)
        grouped[day_key].append(entry)

    return JournalDocument(
        project_name=project_name,
        days=tuple(JournalDay(date_label=day_key, entries=tuple(grouped[day_key])) for day_key in day_order),
    )


def render_journal_text(journal: JournalDocument) -> str:
    """Render a journal document as plain text."""

    sections = [f"{journal.project_name} Journal", ""]
    if not journal.days:
        sections.append("No journal entries have been recorded for this project yet.")
        return "\n".join(sections).strip() + "\n"

    for index, day in enumerate(journal.days):
        if index:
            sections.append("")
        sections.append(day.date_label)
        sections.append("=" * len(day.date_label))
        for entry in day.entries:
            prefix = _stamp_prefix(entry.stamp_kind)
            sections.append(f"{prefix} {entry.title}")
            if entry.workflow_label:
                sections.append(f"Workflow: {entry.workflow_label}")
            sections.append(entry.body)
            sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def render_journal_markdown(journal: JournalDocument) -> str:
    """Render a journal document as Markdown."""

    lines = [f"# {journal.project_name} Journal", ""]
    if not journal.days:
        lines.append("No journal entries have been recorded for this project yet.")
        return "\n".join(lines).strip() + "\n"

    for day in journal.days:
        lines.append(f"## {day.date_label}")
        lines.append("")
        for entry in day.entries:
            prefix = _stamp_prefix(entry.stamp_kind)
            workflow_suffix = f" ({entry.workflow_label})" if entry.workflow_label else ""
            lines.append(f"- **{prefix} {entry.title}**{workflow_suffix}")
            lines.append(f"  {entry.body}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_journal(journal: JournalDocument, output_path: Path, *, format_name: str) -> None:
    """Write the journal to disk in the requested format."""

    normalized = format_name.strip().lower()
    if normalized == "text":
        content = render_journal_text(journal)
    elif normalized == "markdown":
        content = render_journal_markdown(journal)
    else:
        raise ValueError(f"Unsupported journal export format '{format_name}'.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _entry_from_record(record: ProvenanceRecord) -> JournalEntry | None:
    mapping = {
        "import_structure": _import_structure_entry,
        "configure_convergence_cutoff": _configure_cutoff_entry,
        "configure_convergence_kpoint": _configure_kpoint_entry,
        "configure_pristine_relaxation": _configure_pristine_relaxation_entry,
        "configure_uep": _configure_uep_entry,
        "set_recommended_cutoff": _recommended_cutoff_entry,
        "set_recommended_kpoint_grid": _recommended_kpoint_entry,
        "generate_candidates": _generate_candidates_entry,
        "generation_partial": _generation_partial_entry,
        "promote_uep_candidate": _promote_uep_candidate_entry,
        "submit_jobs": _submit_jobs_entry,
        "job_completed": _job_completed_entry,
        "job_failed": _job_failed_entry,
        "job_stopped": _job_stopped_entry,
        "job_cancelled": _job_cancelled_entry,
        "job_reset": _job_reset_entry,
        "create_comparison_table": _create_comparison_table_entry,
    }
    builder = mapping.get(record.action)
    if builder is None:
        return None
    return builder(record)


def _entry(
    record: ProvenanceRecord,
    *,
    stamp_kind: str,
    title: str,
    body: str,
) -> JournalEntry:
    linked_entities = tuple(record.input_entity_ids) + tuple(record.output_entity_ids)
    return JournalEntry(
        timestamp=record.timestamp,
        stamp_kind=stamp_kind,
        title=title,
        body=body,
        workflow_label=_workflow_label(record.workflow_stage),
        linked_entity_ids=linked_entities,
        linked_job_id=record.job_id,
    )


def _workflow_label(workflow_stage: WorkflowStage | None) -> str | None:
    if workflow_stage is None:
        return None
    return _WORKFLOW_LABELS.get(workflow_stage, workflow_stage.value.replace("_", " ").title())


def _int_param(record: ProvenanceRecord, key: str) -> int | None:
    value = record.parameters.get(key)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _text_param(record: ProvenanceRecord, key: str) -> str | None:
    value = record.parameters.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _import_structure_entry(record: ProvenanceRecord) -> JournalEntry:
    source_file = _text_param(record, "source_file")
    file_name = Path(source_file).name if source_file else "structure file"
    return _entry(
        record,
        stamp_kind="milestone",
        title="Host structure imported",
        body=f"Imported {file_name} into the project and recorded a portable project copy.",
    )


def _configure_cutoff_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _int_param(record, "configuration_count") or _count_sequence(record, "configuration", "cutoff_energies")
    return _entry(
        record,
        stamp_kind="milestone",
        title="Cutoff convergence configured",
        body=f"Prepared a cutoff convergence sweep with {count} cutoff value(s).",
    )


def _configure_kpoint_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _int_param(record, "configuration_count") or _count_sequence(record, "configuration", "sweep_values")
    return _entry(
        record,
        stamp_kind="milestone",
        title="K-point convergence configured",
        body=f"Prepared a k-point convergence sweep with {count} grid value(s).",
    )


def _configure_pristine_relaxation_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _int_param(record, "configuration_count") or _count_sequence(record, "configuration", "rows")
    return _entry(
        record,
        stamp_kind="milestone",
        title="Pristine relaxation configured",
        body=f"Prepared {count} pristine relaxation job row(s) for later submission.",
    )


def _configure_uep_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _int_param(record, "configuration_count") or _count_sequence(record, "configuration", "rows")
    return _entry(
        record,
        stamp_kind="milestone",
        title="UEP source workflow configured",
        body=f"Prepared {count} UEP source job row(s) for later submission.",
    )


def _recommended_cutoff_entry(record: ProvenanceRecord) -> JournalEntry:
    value = record.parameters.get("recommended_cutoff_energy")
    method = _text_param(record, "recommended_cutoff_method")
    method_text = f" using {method}" if method else ""
    return _entry(
        record,
        stamp_kind="success",
        title="Recommended cutoff adopted",
        body=f"Set the project default cutoff to {float(value):.8g} eV{method_text}.",
    )


def _recommended_kpoint_entry(record: ProvenanceRecord) -> JournalEntry:
    grid = record.parameters.get("recommended_kpoint_grid")
    grid_text = "x".join(str(int(component)) for component in grid) if isinstance(grid, list) and len(grid) == 3 else "unknown"
    method = _text_param(record, "recommended_kpoint_method")
    method_text = f" using {method}" if method else ""
    return _entry(
        record,
        stamp_kind="success",
        title="Recommended k-point grid adopted",
        body=f"Set the project default k-point grid to {grid_text}{method_text}.",
    )


def _generate_candidates_entry(record: ProvenanceRecord) -> JournalEntry:
    count = len(record.output_entity_ids)
    method = _text_param(record, "method") or "configured method"
    return _entry(
        record,
        stamp_kind="milestone",
        title="Muon candidates generated",
        body=f"Generated {count} candidate muon site(s) using {method}.",
    )


def _generation_partial_entry(record: ProvenanceRecord) -> JournalEntry:
    requested = _int_param(record, "requested") or 0
    generated = _int_param(record, "generated") or len(record.output_entity_ids)
    return _entry(
        record,
        stamp_kind="warning",
        title="Candidate generation stopped short",
        body=f"Requested {requested} candidate site(s) but only generated {generated}.",
    )


def _promote_uep_candidate_entry(record: ProvenanceRecord) -> JournalEntry:
    source_job_label = _text_param(record, "source_job_label")
    source_text = f" from {source_job_label}" if source_job_label else ""
    return _entry(
        record,
        stamp_kind="success",
        title="UEP point promoted to candidate site",
        body=f"Promoted a UEP analysis point{source_text} into the candidate muon-site list.",
    )


def _submit_jobs_entry(record: ProvenanceRecord) -> JournalEntry:
    labels = record.parameters.get("job_labels")
    count = len(labels) if isinstance(labels, list) else _int_param(record, "job_count") or 0
    target = _text_param(record, "execution_target") or "local"
    label_text = _label_summary(labels)
    body = f"Submitted {count} job(s) to {target}."
    if label_text:
        body += f" Jobs: {label_text}."
    return _entry(
        record,
        stamp_kind="milestone",
        title="Calculations launched",
        body=body,
    )


def _job_completed_entry(record: ProvenanceRecord) -> JournalEntry:
    return _terminal_job_entry(record, stamp_kind="success", title="Calculation completed")


def _job_failed_entry(record: ProvenanceRecord) -> JournalEntry:
    return _terminal_job_entry(record, stamp_kind="danger", title="Calculation failed")


def _job_stopped_entry(record: ProvenanceRecord) -> JournalEntry:
    return _terminal_job_entry(record, stamp_kind="warning", title="Calculation stopped")


def _job_cancelled_entry(record: ProvenanceRecord) -> JournalEntry:
    return _terminal_job_entry(record, stamp_kind="warning", title="Calculation cancelled")


def _job_reset_entry(record: ProvenanceRecord) -> JournalEntry:
    job_label = _text_param(record, "job_label") or "job"
    clean_text = " and cleaned its directory" if bool(record.parameters.get("cleaned_output_directory")) else ""
    return _entry(
        record,
        stamp_kind="milestone",
        title="Calculation reset",
        body=f"Reset {job_label} back to the ready-to-submit state{clean_text}.",
    )


def _create_comparison_table_entry(record: ProvenanceRecord) -> JournalEntry:
    table_label = _text_param(record, "table_label") or "comparison table"
    count = _int_param(record, "dft_structure_count") or 0
    return _entry(
        record,
        stamp_kind="success",
        title="Comparison table created",
        body=f"Created '{table_label}' using {count} relaxed DFT structure(s).",
    )


def _terminal_job_entry(record: ProvenanceRecord, *, stamp_kind: str, title: str) -> JournalEntry:
    job_label = _text_param(record, "job_label") or "job"
    target = _text_param(record, "execution_target") or "local"
    summary = _job_summary_text(record)
    body = f"{job_label} finished on {target}."
    if summary:
        body += f" {summary}"
    return _entry(record, stamp_kind=stamp_kind, title=title, body=body)


def _job_summary_text(record: ProvenanceRecord) -> str:
    parts: list[str] = []
    total_energy = record.parameters.get("result_total_energy")
    if total_energy is not None:
        try:
            parts.append(f"Total energy {float(total_energy):.8g} eV.")
        except (TypeError, ValueError):
            pass
    max_force = record.parameters.get("result_max_force")
    if max_force is not None:
        try:
            parts.append(f"Max force {float(max_force):.8g} eV/Ang.")
        except (TypeError, ValueError):
            pass
    derived_count = _int_param(record, "derived_structure_count")
    if derived_count:
        parts.append(f"Produced {derived_count} derived structure(s).")
    warnings_count = _int_param(record, "warnings_count")
    if warnings_count:
        parts.append(f"Recorded {warnings_count} warning(s).")
    error_message = _text_param(record, "error_message")
    if error_message:
        parts.append(error_message.rstrip(".") + ".")
    return " ".join(parts)


def _count_sequence(record: ProvenanceRecord, parent_key: str, child_key: str) -> int:
    parent = record.parameters.get(parent_key)
    if not isinstance(parent, dict):
        return 0
    values = parent.get(child_key)
    return len(values) if isinstance(values, list) else 0


def _label_summary(value: object) -> str:
    if not isinstance(value, list):
        return ""
    labels = [str(item).strip() for item in value if str(item).strip()]
    if not labels:
        return ""
    if len(labels) <= 3:
        return ", ".join(labels)
    return ", ".join(labels[:3]) + f", and {len(labels) - 3} more"


def _stamp_prefix(stamp_kind: str) -> str:
    return f"[{_STAMP_LABELS.get(stamp_kind, 'Entry')}]"
