"""Journal transformation and export helpers for provenance records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import UUID

from mufinder.domain.charge_states import (
    MUON_RELAXATION_CALC_TYPES,
    canonical_charge_state,
    muon_charge_state_label,
)
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import (
    GENERATION_METHOD_LABELS,
    MANUAL_DEFAULT_METHOD,
    ProvenanceRecord,
)
from mufinder.domain.text import execution_target_text, plural
from mufinder.services.generation_runs import generation_run_from_provenance, run_journal_line


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
    WorkflowStage.HYPERFINE_COUPLING: "Hyperfine Coupling",
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

#: How a job of each workflow is named before its row label (``pristine
#: relaxation relax_1``). A muon relaxation is named by its charge state when
#: the record has one (``muonium relaxation Random Site 1``).
_JOB_NOUNS: dict[WorkflowStage, str] = {
    WorkflowStage.PRISTINE_RELAXATION: "pristine relaxation",
    WorkflowStage.UEP: "UEP calculation",
    WorkflowStage.MUON_RELAXATION: "muon relaxation",
    WorkflowStage.QUANTUM_CORRECTIONS: "phonon calculation for",
}


def stamp_label(stamp_kind: str) -> str:
    """Return the word a journal entry's stamp shows for ``stamp_kind``."""

    return _STAMP_LABELS.get(stamp_kind, "Entry")


def build_journal(project_name: str, records: Iterable[ProvenanceRecord]) -> JournalDocument:
    """Convert structured provenance into a curated journal view model."""

    entries = _entries(records)
    return JournalDocument(project_name=project_name, days=_append_days((), entries))


def extend_journal(
    journal: JournalDocument, records: Iterable[ProvenanceRecord]
) -> JournalDocument | None:
    """Return ``journal`` with the entries for ``records`` appended after its last entry.

    The provenance stream is append-only, so a refresh only needs entries for the
    records added since the journal was built. Returns ``None`` when a new entry
    would sort before the journal's last entry (a record with an older timestamp),
    because :func:`build_journal` would then place it earlier than the end; the
    caller rebuilds the whole journal instead.
    """

    entries = _entries(records)
    if not entries:
        return journal
    if journal.days and entries[0].timestamp < journal.days[-1].entries[-1].timestamp:
        return None
    return JournalDocument(
        project_name=journal.project_name, days=_append_days(journal.days, entries)
    )


def _entries(records: Iterable[ProvenanceRecord]) -> list[JournalEntry]:
    """Journal entries for ``records`` in time order; records without one are skipped."""

    return [
        entry
        for record in sorted(records, key=lambda item: item.timestamp)
        if (entry := _entry_from_record(record))
    ]


def _day_label(entry: JournalEntry) -> str:
    return entry.timestamp.astimezone().strftime("%A %d %B %Y")


def _append_days(
    days: tuple[JournalDay, ...], entries: Iterable[JournalEntry]
) -> tuple[JournalDay, ...]:
    """Group time-ordered ``entries`` under day headings, continuing ``days``."""

    grouped: dict[str, list[JournalEntry]] = {day.date_label: list(day.entries) for day in days}
    day_order = [day.date_label for day in days]
    for entry in entries:
        day_key = _day_label(entry)
        if day_key not in grouped:
            grouped[day_key] = []
            day_order.append(day_key)
        grouped[day_key].append(entry)
    return tuple(
        JournalDay(date_label=day_key, entries=tuple(grouped[day_key])) for day_key in day_order
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
        "import_legacy_relaxations": _import_legacy_relaxations_entry,
        "remove_imported_relaxations": _remove_imported_relaxations_entry,
        "configure_convergence_cutoff": _configure_cutoff_entry,
        "configure_convergence_kpoint": _configure_kpoint_entry,
        "configure_pristine_relaxation": _configure_pristine_relaxation_entry,
        "configure_uep": _configure_uep_entry,
        "configure_muon_relaxation": _configure_muon_relaxation_entry,
        "configure_quantum_corrections": _configure_quantum_corrections_entry,
        "configure_energy_barriers": _configure_energy_barriers_entry,
        "set_active_host_structure": _set_active_host_entry,
        "set_recommended_cutoff": _recommended_cutoff_entry,
        "set_recommended_kpoint_grid": _recommended_kpoint_entry,
        "set_recommended_xc_functional": _recommended_xc_functional_entry,
        "clear_recommended_cutoff": _cleared_cutoff_entry,
        "clear_recommended_kpoint_grid": _cleared_kpoint_entry,
        "generate_candidates": _generate_candidates_entry,
        "generation_partial": _generation_partial_entry,
        "symmetry_unavailable": _symmetry_unavailable_entry,
        "cluster_candidate_positions": _cluster_positions_entry,
        "stage_muon_relaxation_sites": _stage_sites_entry,
        "delete_candidates": _delete_candidates_entry,
        "create_analysis_workspace": _create_analysis_workspace_entry,
        "group_relaxed_sites": _group_relaxed_sites_entry,
        "promote_uep_candidate": _promote_uep_candidate_entry,
        "submit_jobs": _submit_jobs_entry,
        "job_completed": _job_completed_entry,
        "job_failed": _job_failed_entry,
        "job_stopped": _job_stopped_entry,
        "job_cancelled": _job_cancelled_entry,
        "job_reset": _job_reset_entry,
        "job_continuation": _job_continuation_entry,
        "job_dismissed": _job_dismissed_entry,
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


def _import_legacy_relaxations_entry(record: ProvenanceRecord) -> JournalEntry:
    """Describe one legacy-import batch.

    The journal is where a reader learns that some relaxations in this project
    were not run here, so the entry leads with that and surfaces the two things
    most likely to matter later: runs that were skipped, and parameter sources
    that disagreed.
    """

    sites = record.parameters.get("imported_sites")
    imported = len(sites) if isinstance(sites, list) else 0
    skipped_raw = record.parameters.get("skipped")
    skipped = len(skipped_raw) if isinstance(skipped_raw, list) else 0
    source = _text_param(record, "source_folder")
    folder = Path(source).name if source else "a legacy folder"

    conflicts = 0
    if isinstance(sites, list):
        for entry in sites:
            if isinstance(entry, dict) and isinstance(entry.get("conflicts"), list):
                conflicts += len(entry["conflicts"])

    body = (
        f"Imported {plural(imported, 'muon relaxation')} from {folder}. These were run "
        "outside MuFinder; their rows are read-only."
    )
    if skipped:
        body += f" {plural(skipped, 'run was', 'runs were')} skipped."
    if conflicts:
        body += (
            f" {plural(conflicts, 'parameter disagreement was', 'parameter disagreements were')} "
            "resolved in favour of the "
            "value the calculation actually used."
        )

    return _entry(
        record,
        stamp_kind="warning" if (skipped or conflicts) else "milestone",
        title="Legacy relaxations imported",
        body=body,
    )


def _remove_imported_relaxations_entry(record: ProvenanceRecord) -> JournalEntry:
    """Describe one removal of imported relaxations.

    Stamped as a warning rather than a milestone: an imported relaxation holds
    the only copy of its results, so its removal is not recoverable from within
    the project the way re-running a native relaxation is.
    """

    labels = record.parameters.get("labels")
    count = len(labels) if isinstance(labels, list) else 0
    body = f"Removed {plural(count, 'imported muon relaxation')} from the project."
    if record.parameters.get("deleted_files"):
        body += " Their job directories were deleted, including the archived original input files."
    return _entry(
        record,
        stamp_kind="warning",
        title="Imported relaxations removed",
        body=body,
    )


def _configure_cutoff_entry(record: ProvenanceRecord) -> JournalEntry:
    # Records written before the schema 2 migration may hold the sweep only
    # under the legacy ``cutoff_energies`` key; provenance is never rewritten.
    count = (
        _int_param(record, "configuration_count")
        or _count_sequence(record, "configuration", "sweep_values")
        or _count_sequence(record, "configuration", "cutoff_energies")
    )
    return _entry(
        record,
        stamp_kind="milestone",
        title="Cutoff convergence configured",
        body=f"Prepared a cutoff convergence sweep with {plural(count, 'cutoff value')}.",
    )


def _configure_kpoint_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _int_param(record, "configuration_count") or _count_sequence(
        record, "configuration", "sweep_values"
    )
    return _entry(
        record,
        stamp_kind="milestone",
        title="K-point convergence configured",
        body=f"Prepared a k-point convergence sweep with {plural(count, 'grid value')}.",
    )


def _configure_pristine_relaxation_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _int_param(record, "configuration_count") or _count_sequence(
        record, "configuration", "rows"
    )
    return _entry(
        record,
        stamp_kind="milestone",
        title="Pristine relaxation configured",
        body=f"Prepared {plural(count, 'pristine relaxation job row')} for later submission.",
    )


def _configure_uep_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _int_param(record, "configuration_count") or _count_sequence(
        record, "configuration", "rows"
    )
    return _entry(
        record,
        stamp_kind="milestone",
        title="UEP source workflow configured",
        body=f"Prepared {plural(count, 'UEP source job row')} for later submission.",
    )


def _configure_muon_relaxation_entry(record: ProvenanceRecord) -> JournalEntry:
    configuration = record.parameters.get("configuration")
    rows = configuration.get("rows") if isinstance(configuration, dict) else None
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    charge_counts: dict[str, int] = {}
    for row in rows:
        label = muon_charge_state_label(str(row.get("charge_state") or ""))
        charge_counts[label] = charge_counts.get(label, 0) + 1
    body = f"Prepared {plural(len(rows), 'muon relaxation')} for later submission"
    if charge_counts:
        body += " (" + ", ".join(f"{count} {label}" for label, count in charge_counts.items()) + ")"
    return _entry(
        record, stamp_kind="milestone", title="Muon relaxations configured", body=f"{body}."
    )


def _configure_quantum_corrections_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _count_sequence(record, "configuration", "sources")
    return _entry(
        record,
        stamp_kind="milestone",
        title="Quantum Corrections configured",
        body=f"Prepared phonon calculations for {plural(count, 'relaxed muon site')}.",
    )


def _configure_energy_barriers_entry(record: ProvenanceRecord) -> JournalEntry:
    count = _count_sequence(record, "configuration", "sources")
    return _entry(
        record,
        stamp_kind="milestone",
        title="Energy Barriers configured",
        body=f"Prepared NEB calculations for {plural(count, 'path')}.",
    )


def _set_active_host_entry(record: ProvenanceRecord) -> JournalEntry:
    label = _text_param(record, "new_host_structure_label")
    if label is None:
        # Records written before the labels were recorded keep them in the description.
        body = record.description.rstrip(".") + "."
    else:
        previous = _text_param(record, "previous_host_structure_label")
        body = f"Made '{label}' the active host structure"
        body += f" in place of '{previous}'." if previous else "."
    return _entry(record, stamp_kind="milestone", title="Active host changed", body=body)


def _recommended_cutoff_entry(record: ProvenanceRecord) -> JournalEntry:
    value = record.parameters.get("recommended_cutoff_energy")
    method = _text_param(record, "recommended_cutoff_method")
    if method == MANUAL_DEFAULT_METHOD:
        return _entry(
            record,
            stamp_kind="milestone",
            title="Default cutoff set manually",
            body=f"Set the project default cutoff to {float(value):.8g} eV by hand.",
        )
    return _entry(
        record,
        stamp_kind="success",
        title="Recommended cutoff adopted",
        body=(
            f"Set the project default cutoff to {float(value):.8g} eV"
            f"{_converged_with_text(record, 'recommended_cutoff_xc_functional')}"
            f"{_criterion_text(record, 'recommended_cutoff', 'the finest cutoff')}."
        ),
    )


def _recommended_kpoint_entry(record: ProvenanceRecord) -> JournalEntry:
    grid = record.parameters.get("recommended_kpoint_grid")
    grid_text = (
        "x".join(str(int(component)) for component in grid)
        if isinstance(grid, list) and len(grid) == 3
        else "unknown"
    )
    method = _text_param(record, "recommended_kpoint_method")
    if method == MANUAL_DEFAULT_METHOD:
        return _entry(
            record,
            stamp_kind="milestone",
            title="Default k-point grid set manually",
            body=f"Set the project default k-point grid to {grid_text} by hand.",
        )
    return _entry(
        record,
        stamp_kind="success",
        title="Recommended k-point grid adopted",
        body=(
            f"Set the project default k-point grid to {grid_text}"
            f"{_converged_with_text(record, 'recommended_kpoint_xc_functional')}"
            f"{_criterion_text(record, 'recommended_kpoint', 'the densest grid')}."
        ),
    )


def _criterion_text(record: ProvenanceRecord, prefix: str, reference: str) -> str:
    """Name the convergence criterion a default was adopted under, from its tolerances.

    Each quantity in use is converged within its tolerance of the sweep's
    reference point (the finest cutoff, or the densest k-point grid). Empty when
    the record holds no tolerance.
    """

    parts: list[str] = []
    for key, quantity, unit in (
        ("energy_threshold", "energy", "eV/atom"),
        ("force_threshold", "max force", "eV/Å"),
        ("stress_threshold", "max stress", "GPa"),
    ):
        value = _float_param(record, f"{prefix}_{key}")
        if value is not None:
            parts.append(f"{quantity} within {value:g} {unit}")
    if not parts:
        return ""
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f": {joined} of {reference}"


def _float_param(record: ProvenanceRecord, key: str) -> float | None:
    value = record.parameters.get(key)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _cleared_cutoff_entry(record: ProvenanceRecord) -> JournalEntry:
    return _entry(
        record,
        stamp_kind="milestone",
        title="Default cutoff cleared",
        body="Cleared the project default cutoff energy.",
    )


def _cleared_kpoint_entry(record: ProvenanceRecord) -> JournalEntry:
    return _entry(
        record,
        stamp_kind="milestone",
        title="Default k-point grid cleared",
        body="Cleared the project default k-point grid.",
    )


def _converged_with_text(record: ProvenanceRecord, key: str) -> str:
    functional = _text_param(record, key)
    return f" (converged with the {functional} functional)" if functional else ""


def _recommended_xc_functional_entry(record: ProvenanceRecord) -> JournalEntry:
    functional = _text_param(record, "recommended_xc_functional")
    if functional is None:
        return _entry(
            record,
            stamp_kind="milestone",
            title="Default XC functional cleared",
            body="Cleared the project default exchange-correlation functional.",
        )
    return _entry(
        record,
        stamp_kind="milestone",
        title="Default XC functional set",
        body=f"Set the project default exchange-correlation functional to {functional}.",
    )


def _generate_candidates_entry(record: ProvenanceRecord) -> JournalEntry:
    run = generation_run_from_provenance(record.parameters)
    if run is not None:
        body = run_journal_line(run)
    else:
        # Records written before generation runs were recorded keep only the request.
        count = len(record.output_entity_ids)
        method = GENERATION_METHOD_LABELS.get(_text_param(record, "method") or "")
        method_text = f" with '{method}'" if method else ""
        body = f"Generated {plural(count, 'muon site')}{method_text}."
    return _entry(record, stamp_kind="milestone", title="Muon sites generated", body=body)


def _site_labels_text(record: ProvenanceRecord) -> str:
    labels = _label_summary(record.parameters.get("site_labels"))
    return f": {labels}" if labels else ""


def _cluster_positions_entry(record: ProvenanceRecord) -> JournalEntry:
    count = len(record.output_entity_ids)
    anchor = record.parameters.get("anchor_fractional")
    anchor_text = (
        " (" + ", ".join(f"{float(value):.4f}" for value in anchor) + ")"
        if isinstance(anchor, list) and len(anchor) == 3
        else ""
    )
    return _entry(
        record,
        stamp_kind="milestone",
        title="Muon starts clustered",
        body=(
            f"Moved {plural(count, 'generated muon site')} to the symmetry-equivalent "
            f"positions nearest the anchor{anchor_text}{_site_labels_text(record)}."
            f"{_original_positions_text(record)}"
        ),
    )


def _original_positions_text(record: ProvenanceRecord) -> str:
    """Say where the moved starts' original positions are kept.

    The record keeps every one; starts made before generation runs were
    recorded have no run to keep them as well.
    """

    runs = record.parameters.get("generation_run_ids")
    run_count = len(runs) if isinstance(runs, list) else 0
    text = " The original positions are kept in the project's provenance"
    if run_count == 1:
        text += " and with the generation run"
    elif run_count > 1:
        text += f" and with their {run_count} generation runs"
    return f"{text}."


def _stage_sites_entry(record: ProvenanceRecord) -> JournalEntry:
    count = len(record.output_entity_ids)
    return _entry(
        record,
        stamp_kind="milestone",
        title="Muon sites staged for relaxation",
        body=f"Staged {plural(count, 'generated muon site')} for relaxation{_site_labels_text(record)}.",
    )


def _delete_candidates_entry(record: ProvenanceRecord) -> JournalEntry:
    count = len(record.input_entity_ids)
    return _entry(
        record,
        stamp_kind="milestone",
        title="Generated muon sites removed",
        body=f"Removed {plural(count, 'generated muon site')}{_site_labels_text(record)}.",
    )


def _create_analysis_workspace_entry(record: ProvenanceRecord) -> JournalEntry:
    name = _text_param(record, "workspace_name") or "analysis workspace"
    count = _int_param(record, "relaxation_count") or 0
    charge = _text_param(record, "charge_state_label")
    relaxations = f"{charge} relaxation" if charge else "relaxation"
    body = f"Created '{name}' with {plural(count, relaxations)}"
    distinct = _int_param(record, "distinct_site_count")
    summary = _text_param(record, "grouping_summary")
    if distinct is not None:
        body += f", grouped into {plural(distinct, 'distinct site')}"
        body += f": {summary}." if summary else "."
    else:
        body += "."
    return _entry(record, stamp_kind="milestone", title="Analysis workspace created", body=body)


def _group_relaxed_sites_entry(record: ProvenanceRecord) -> JournalEntry:
    name = _text_param(record, "workspace_name") or "analysis workspace"
    site_count = _int_param(record, "site_count") or 0
    distinct = _int_param(record, "distinct_site_count") or 0
    summary = _text_param(record, "grouping_summary")
    body = (
        f"Grouped {plural(site_count, 'relaxed site')} in '{name}' into "
        f"{plural(distinct, 'distinct site')}"
    )
    body += f": {summary}." if summary else "."
    return _entry(record, stamp_kind="success", title="Relaxed sites grouped", body=body)


def _generation_partial_entry(record: ProvenanceRecord) -> JournalEntry:
    requested = _int_param(record, "requested") or 0
    generated = _int_param(record, "generated") or len(record.output_entity_ids)
    return _entry(
        record,
        stamp_kind="warning",
        title="Site generation stopped short",
        body=f"Requested {plural(requested, 'site')} but only generated {generated}.",
    )


def _symmetry_unavailable_entry(record: ProvenanceRecord) -> JournalEntry:
    return _entry(
        record,
        stamp_kind="warning",
        title="Host symmetry not detected",
        body=record.description,
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


def _charge_state_text(value: object) -> str | None:
    """Return a recorded muon charge state as text names it, or ``None`` when there is none."""

    charge_state = canonical_charge_state(value) if isinstance(value, str) else None
    if charge_state not in MUON_RELAXATION_CALC_TYPES:
        return None
    return muon_charge_state_label(charge_state)


def _job_description(record: ProvenanceRecord, *, sentence_start: bool = False) -> str:
    """Name a job by its workflow and row label (``pristine relaxation relax_1``).

    A muon relaxation is named by its charge state. Records without a row label
    (older records, and workflows whose rows are not named) give the job's
    label, which keeps its own spelling at the start of a sentence.
    """

    row_label = _text_param(record, "row_label")
    if row_label is None:
        job_label = _text_param(record, "job_label")
        if job_label is not None:
            return job_label
        description = "a calculation"
    else:
        charge = _charge_state_text(record.parameters.get("charge_state"))
        if charge is not None:
            noun = f"{charge} relaxation"
        elif record.workflow_stage is not None:
            noun = _JOB_NOUNS.get(record.workflow_stage, "calculation")
        else:
            noun = "calculation"
        description = f"{noun} {row_label}"
    return description[:1].upper() + description[1:] if sentence_start else description


def _target_text(record: ProvenanceRecord) -> str:
    return execution_target_text(_text_param(record, "execution_target"))


def _submit_jobs_entry(record: ProvenanceRecord) -> JournalEntry:
    raw_jobs = record.parameters.get("jobs")
    jobs = [job for job in raw_jobs if isinstance(job, dict)] if isinstance(raw_jobs, list) else []
    labels = record.parameters.get("job_labels")
    count = len(labels) if isinstance(labels, list) else _int_param(record, "job_count") or 0
    charges: list[str] = []
    for value in [
        record.parameters.get("charge_state"),
        *(job.get("charge_state") for job in jobs),
    ]:
        charge = _charge_state_text(value)
        if charge is not None and charge not in charges:
            charges.append(charge)
    if jobs:
        labels = [str(job.get("row_label") or job.get("label") or "") for job in jobs]
        if len(charges) > 1:
            # One site can be relaxed in both charge states in one batch.
            labels = [
                f"{label} ({charge})"
                if (charge := _charge_state_text(job.get("charge_state")))
                else label
                for label, job in zip(labels, jobs, strict=True)
            ]
    label_text = _label_summary(labels)
    body = f"Submitted {plural(count, 'job')} to run on {_target_text(record)}"
    body += f": {label_text}." if label_text else "."

    # A batch from several workflows (no workflow stage) keeps the general title.
    title = (
        f"Muon relaxations launched: {' and '.join(charges)}"
        if charges and record.workflow_stage == WorkflowStage.MUON_RELAXATION
        else "Calculations launched"
    )
    return _entry(record, stamp_kind="milestone", title=title, body=body)


def _job_completed_entry(record: ProvenanceRecord) -> JournalEntry:
    return _terminal_job_entry(record, stamp_kind="success", outcome="completed")


def _job_failed_entry(record: ProvenanceRecord) -> JournalEntry:
    return _terminal_job_entry(record, stamp_kind="danger", outcome="failed")


def _job_stopped_entry(record: ProvenanceRecord) -> JournalEntry:
    return _terminal_job_entry(record, stamp_kind="warning", outcome="stopped")


def _job_cancelled_entry(record: ProvenanceRecord) -> JournalEntry:
    return _terminal_job_entry(record, stamp_kind="warning", outcome="cancelled")


def _job_reset_entry(record: ProvenanceRecord) -> JournalEntry:
    clean_text = (
        " and cleaned its directory"
        if bool(record.parameters.get("cleaned_output_directory"))
        else ""
    )
    return _entry(
        record,
        stamp_kind="milestone",
        title="Calculation reset",
        body=f"Reset {_job_description(record)} back to the ready-to-submit state{clean_text}.",
    )


def _job_continuation_entry(record: ProvenanceRecord) -> JournalEntry:
    directory = _text_param(record, "remote_directory")
    preserved = bool(record.parameters.get("preserved_remote_directory")) and bool(directory)
    reuse_text = f" reusing {directory}" if preserved else ""
    return _entry(
        record,
        stamp_kind="milestone",
        title="Calculation continued",
        body=f"Queued {_job_description(record)} for a continuation run{reuse_text}.",
    )


def _job_dismissed_entry(record: ProvenanceRecord) -> JournalEntry:
    return _entry(
        record,
        stamp_kind="milestone",
        title="Calculation dismissed",
        body=f"Dismissed {_job_description(record)} from the Job Dashboard.",
    )


def _create_comparison_table_entry(record: ProvenanceRecord) -> JournalEntry:
    table_label = _text_param(record, "table_label") or "comparison table"
    count = _int_param(record, "dft_structure_count") or 0
    return _entry(
        record,
        stamp_kind="success",
        title="Comparison table created",
        body=f"Created '{table_label}' using {plural(count, 'relaxed DFT structure')}.",
    )


def _terminal_job_entry(record: ProvenanceRecord, *, stamp_kind: str, outcome: str) -> JournalEntry:
    charge = _charge_state_text(record.parameters.get("charge_state"))
    title = f"{charge.capitalize()} relaxation {outcome}" if charge else f"Calculation {outcome}"
    summary = _job_summary_text(record)
    body = f"{_job_description(record, sentence_start=True)} finished on {_target_text(record)}."
    if summary:
        body += f" {summary}"
    return _entry(record, stamp_kind=stamp_kind, title=title, body=body)


def _job_summary_text(record: ProvenanceRecord) -> str:
    """The job's results: its total energy to 0.1 meV and its max force to 3 significant figures."""

    parts: list[str] = []
    total_energy = _float_param(record, "result_total_energy")
    if total_energy is not None:
        parts.append(f"Total energy {total_energy:.4f} eV.")
    max_force = _float_param(record, "result_max_force")
    if max_force is not None:
        parts.append(f"Max force {max_force:.3g} eV/Å.")
    derived_count = _int_param(record, "derived_structure_count")
    if derived_count:
        parts.append(f"Produced {plural(derived_count, 'derived structure')}.")
    warnings_count = _int_param(record, "warnings_count")
    if warnings_count:
        parts.append(f"Recorded {plural(warnings_count, 'warning')}.")
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
    return f"[{stamp_label(stamp_kind)}]"
