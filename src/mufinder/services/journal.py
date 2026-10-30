"""Journal transformation and export helpers for provenance records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable
from uuid import UUID

from mufinder.domain.charge_states import (
    MUON_RELAXATION_CALC_TYPES,
    canonical_charge_state,
    muon_charge_state_label,
)
from mufinder.domain.constants import MEV_PER_EV
from mufinder.domain.enums import FailureReason, WorkflowStage
from mufinder.domain.models import (
    GENERATION_METHOD_LABELS,
    MANUAL_DEFAULT_METHOD,
    ProvenanceRecord,
)
from mufinder.domain.routes import comparison_route
from mufinder.domain.text import execution_target_text, plural
from mufinder.services.failure_diagnosis import FAILURE_REASON_SUMMARIES
from mufinder.services.generation_runs import (
    generation_run_from_provenance,
    run_journal_line,
    run_methods_sentence,
)


@dataclass(frozen=True, slots=True)
class JournalJobRow:
    """What the Journal knows about one job of a submitted batch.

    ``status`` is ``submitted`` until a terminal record arrives, then that
    record's outcome: ``completed``, ``failed``, ``stopped`` or ``cancelled``.
    """

    label: str
    status: str
    total_energy: float | None = None
    max_force: float | None = None
    warnings_count: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """One user-facing journal entry derived from provenance.

    ``body`` is the curated text exports print. ``summary`` is the shorter line a
    collapsed card shows when it differs from the body (see :attr:`card_summary`).
    ``detail_lines`` and ``job_rows`` feed :func:`entry_details`, ``job_column``
    heads the first column of the job table, and ``route`` is the navigator route
    of the page ``open_label`` opens.
    """

    timestamp: datetime
    stamp_kind: str
    title: str
    body: str
    workflow_label: str | None
    linked_entity_ids: tuple[UUID, ...] = ()
    linked_job_id: UUID | None = None
    job_rows: tuple[JournalJobRow, ...] = ()
    summary: str = ""
    detail_lines: tuple[str, ...] = ()
    job_column: str = "Calculation"
    route: str | None = None
    open_label: str | None = None

    @property
    def card_summary(self) -> str:
        """The one line a collapsed card shows under its title."""

        return self.summary or self.body


@dataclass(frozen=True, slots=True)
class JournalJobTable:
    """A batch's per-job table as display text, in column order."""

    headers: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class JournalDetails:
    """What an expanded card shows: the exact time, logged details and any job table."""

    exact_time: str
    lines: tuple[str, ...] = ()
    job_table: JournalJobTable | None = None


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

#: The navigator route of each workflow's page.
_WORKFLOW_ROUTES: dict[WorkflowStage, str] = {
    WorkflowStage.CONVERGENCE_CUTOFF: "convergence_cutoff",
    WorkflowStage.CONVERGENCE_KPOINT: "convergence_kpoint",
    WorkflowStage.PRISTINE_RELAXATION: "pristine_relaxation",
    WorkflowStage.UEP: "uep",
    WorkflowStage.MUON_GENERATION: "muon_candidates",
    WorkflowStage.MUON_RELAXATION: "muon_relaxations",
    WorkflowStage.SITE_ANALYSIS: "muon_analysis",
    WorkflowStage.HYPERFINE_COUPLING: "hyperfine_coupling",
    WorkflowStage.MAGNETIC_FIELD: "magnetic_fields",
    WorkflowStage.QUANTUM_CORRECTIONS: "quantum_corrections",
    WorkflowStage.ENERGY_BARRIERS: "energy_barriers",
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


#: The terminal job records and the outcome each one reports, in the order a
#: batch's outcome lists them.
_TERMINAL_OUTCOMES: dict[str, str] = {
    "job_completed": "completed",
    "job_failed": "failed",
    "job_stopped": "stopped",
    "job_cancelled": "cancelled",
}

#: How a card about one job says how it ended.
_OUTCOME_VERBS: dict[str, str] = {
    "completed": "finished",
    "failed": "failed",
    "stopped": "stopped",
    "cancelled": "was cancelled",
}

#: Records after which a job's next outcome no longer belongs to its submission.
_FOLD_BREAKING_ACTIONS = frozenset({"job_reset", "job_continuation"})


def stamp_label(stamp_kind: str) -> str:
    """Return the word a journal entry's stamp shows for ``stamp_kind``."""

    return _STAMP_LABELS.get(stamp_kind, "Entry")


def entry_meta_text(entry: JournalEntry) -> str:
    """The right of a card's header: the workflow, if any, and the local time."""

    time_text = entry.timestamp.astimezone().strftime("%H:%M")
    return f"{entry.workflow_label} · {time_text}" if entry.workflow_label else time_text


def entry_details(entry: JournalEntry) -> JournalDetails:
    """The details block of an expanded card, built when the card is first expanded."""

    local = entry.timestamp.astimezone()
    failures = tuple(
        f"{row.label}: {row.failure_reason}" for row in entry.job_rows if row.failure_reason
    )
    return JournalDetails(
        exact_time=f"{_day_label(local)} at {local.strftime('%H:%M:%S %Z').strip()}",
        lines=(*entry.detail_lines, *failures),
        job_table=_job_table(entry) if entry.job_rows else None,
    )


def _job_table(entry: JournalEntry) -> JournalJobTable:
    """Per-job rows, with each energy relative to the batch's lowest in meV."""

    energies = [row.total_energy for row in entry.job_rows if row.total_energy is not None]
    lowest = min(energies, default=0.0)
    rows = []
    for row in entry.job_rows:
        relative = (
            f"{(row.total_energy - lowest) * MEV_PER_EV:.1f}"
            if row.total_energy is not None
            else "—"
        )
        rows.append(
            (
                row.label,
                row.status.capitalize(),
                relative,
                f"{row.max_force:.3g}" if row.max_force is not None else "—",
                str(row.warnings_count) if row.warnings_count else "—",
            )
        )
    return JournalJobTable(
        headers=(entry.job_column, "Status", "ΔE (meV)", "Max force (eV/Å)", "Warnings"),
        rows=tuple(rows),
    )


def build_journal(project_name: str, records: Iterable[ProvenanceRecord]) -> JournalDocument:
    """Convert structured provenance into a curated journal view model."""

    return JournalDocument(project_name=project_name, days=_group_days(_entries(records)))


def _entries(records: Iterable[ProvenanceRecord]) -> list[JournalEntry]:
    """Journal entries for ``records`` in time order; records without one are skipped.

    Completions fold into the card of the submission that launched them, and a
    run of dismissals on one day folds into one card.
    """

    ordered = sorted(records, key=lambda item: item.timestamp)
    batches = _submitted_batches(ordered)
    by_submission = {id(batch.record): batch for batch in batches}
    by_last_outcome = {id(batch.last_outcome): batch for batch in batches if batch.last_outcome}
    folded = {id(record) for batch in batches for record in batch.completions}

    entries: list[JournalEntry] = []
    dismissals: list[ProvenanceRecord] = []
    configurations: dict[WorkflowStage | None, ProvenanceRecord] = {}

    def flush_dismissals() -> None:
        if dismissals:
            entries.append(_dismissals_entry(dismissals))
            dismissals.clear()

    for record in ordered:
        if record.action == "job_dismissed":
            if dismissals and _record_day(dismissals[-1]) != _record_day(record):
                flush_dismissals()
            dismissals.append(record)
            continue
        if record.action in _CONFIGURE_ACTIONS:
            configurations[record.workflow_stage] = record
        new: list[JournalEntry] = []
        batch = by_submission.get(id(record))
        if batch is not None:
            new.append(_batch_entry(batch, configurations.get(record.workflow_stage)))
        elif id(record) not in folded and (entry := _entry_from_record(record)):
            new.append(entry)
        finished = by_last_outcome.get(id(record))
        if finished is not None and (line := _finished_entry(finished)):
            new.append(line)
        if new:
            flush_dismissals()
            entries.extend(new)
    flush_dismissals()
    return entries


def _day_label(timestamp: datetime) -> str:
    return timestamp.astimezone().strftime("%A %d %B %Y")


def _record_day(record: ProvenanceRecord) -> str:
    return _day_label(record.timestamp)


def _group_days(entries: Iterable[JournalEntry]) -> tuple[JournalDay, ...]:
    """Group time-ordered ``entries`` under day headings."""

    grouped: dict[str, list[JournalEntry]] = {}
    for entry in entries:
        grouped.setdefault(_day_label(entry.timestamp), []).append(entry)
    return tuple(
        JournalDay(date_label=day_key, entries=tuple(day_entries))
        for day_key, day_entries in grouped.items()
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
    entry = builder(record)
    details = _DETAIL_BUILDERS.get(record.action)
    return replace(entry, detail_lines=details(record)) if details else entry


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
        route=_WORKFLOW_ROUTES.get(record.workflow_stage),
        open_label=_open_label(record.workflow_stage),
    )


def _open_label(workflow_stage: WorkflowStage | None) -> str | None:
    label = _workflow_label(workflow_stage)
    return f"Open in {label}" if label else None


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
    entry = _entry(
        record,
        stamp_kind="milestone",
        title="Host structure imported",
        body=f"Imported {file_name} into the project and recorded a portable project copy.",
    )
    # The record is stamped with the generation workflow; the structure has its own page.
    return replace(entry, route="host_structure", open_label="Open in Host Structure")


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

    parts = _threshold_parts(record, prefix)
    if not parts:
        return ""
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f": {joined} of {reference}"


def _threshold_parts(record: ProvenanceRecord, prefix: str) -> list[str]:
    """Each tolerance a default was adopted under: ``energy within 0.001 eV/atom``."""

    parts: list[str] = []
    for key, quantity, unit in (
        ("energy_threshold", "energy", "eV/atom"),
        ("force_threshold", "max force", "eV/Å"),
        ("stress_threshold", "max stress", "GPa"),
    ):
        value = _float_param(record, f"{prefix}_{key}")
        if value is not None:
            parts.append(f"{quantity} within {value:g} {unit}")
    return parts


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
    return _entry(
        record,
        stamp_kind="milestone",
        title="Analysis workspace created",
        body=f"Created '{name}' with {plural(count, relaxations)}.",
    )


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


def _submitted_jobs(record: ProvenanceRecord) -> list[Mapping[str, object]]:
    raw_jobs = record.parameters.get("jobs")
    return [job for job in raw_jobs if isinstance(job, dict)] if isinstance(raw_jobs, list) else []


def _job_list_labels(jobs: Sequence[Mapping[str, object]], charges: Sequence[str]) -> list[str]:
    """Name each job by its row. When ``charges`` holds both, each label says its own."""

    labels = [
        str(job.get("row_label") or job.get("label") or job.get("job_label") or "") for job in jobs
    ]
    if len(charges) <= 1:
        return labels
    # One site can be relaxed in both charge states in one batch.
    return [
        f"{label} ({charge})" if (charge := _charge_state_text(job.get("charge_state"))) else label
        for label, job in zip(labels, jobs, strict=True)
    ]


def _charge_states(values: Iterable[object]) -> list[str]:
    """The distinct muon charge states named by ``values``, in first-seen order."""

    charges: list[str] = []
    for value in values:
        charge = _charge_state_text(value)
        if charge is not None and charge not in charges:
            charges.append(charge)
    return charges


def _submission_charges(record: ProvenanceRecord) -> list[str]:
    return _charge_states(
        [
            record.parameters.get("charge_state"),
            *(job.get("charge_state") for job in _submitted_jobs(record)),
        ]
    )


def _submit_jobs_entry(record: ProvenanceRecord) -> JournalEntry:
    jobs = _submitted_jobs(record)
    labels = record.parameters.get("job_labels")
    count = len(labels) if isinstance(labels, list) else _int_param(record, "job_count") or 0
    charges = _submission_charges(record)
    if jobs:
        labels = _job_list_labels(jobs, charges)
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


@dataclass(slots=True)
class _SubmittedBatch:
    """One ``submit_jobs`` record and the latest outcome recorded for each of its jobs."""

    record: ProvenanceRecord
    jobs: dict[str, Mapping[str, object]]
    outcomes: dict[str, ProvenanceRecord] = field(default_factory=dict)
    completions: list[ProvenanceRecord] = field(default_factory=list)
    last_outcome: ProvenanceRecord | None = None


def _submitted_batches(ordered: Iterable[ProvenanceRecord]) -> list[_SubmittedBatch]:
    """Assign time-ordered terminal records to the submission that launched each job.

    A job belongs to its latest submission. A reset or continuation ends that
    until the job is submitted again, so an outcome with no submission behind it
    stands alone. A later outcome replaces an earlier one, as when a stopped
    cluster job reappears and completes, except that a completed job stays
    completed: a late failure keeps its own card and does not reopen the job.
    Records without job ids (older projects, imports) form no batch.
    """

    batches: list[_SubmittedBatch] = []
    current: dict[str, _SubmittedBatch] = {}
    for record in ordered:
        if record.action == "submit_jobs":
            listed = _submitted_jobs(record)
            if listed and all(job.get("job_id") for job in listed):
                jobs = {str(job["job_id"]): job for job in listed}
                batch = _SubmittedBatch(record=record, jobs=jobs)
                batches.append(batch)
                current.update(dict.fromkeys(jobs, batch))
        elif record.job_id is None:
            continue
        elif record.action in _FOLD_BREAKING_ACTIONS:
            current.pop(str(record.job_id), None)
        elif record.action in _TERMINAL_OUTCOMES:
            batch = current.get(str(record.job_id))
            if batch is None:
                continue
            if record.action == "job_completed":
                batch.completions.append(record)
            else:
                earlier = batch.outcomes.get(str(record.job_id))
                if earlier is not None and earlier.action == "job_completed":
                    continue
            batch.outcomes[str(record.job_id)] = record
            batch.last_outcome = record
    return batches


def _batch_outcome_text(batch: _SubmittedBatch) -> str:
    """Say how a batch stands: ``10 of 10 completed``, ``8 completed, 2 failed``."""

    counts = Counter(_TERMINAL_OUTCOMES[record.action] for record in batch.outcomes.values())
    if not counts:
        return ""
    total = len(batch.jobs)
    finished = sum(counts.values())
    if len(counts) == 1:
        outcome, count = counts.popitem()
        return f"{count} of {total} {outcome}"
    text = ", ".join(
        f"{counts[outcome]} {outcome}" for outcome in _TERMINAL_OUTCOMES.values() if counts[outcome]
    )
    return text if finished == total else f"{finished} of {total} finished: {text}"


def _batch_summary(batch: _SubmittedBatch, outcome: str) -> str:
    """The card's line: ``10 sites on this computer, 8 processes each · 10 of 10 completed``."""

    jobs = list(batch.jobs.values())
    if batch.record.workflow_stage == WorkflowStage.MUON_RELAXATION:
        # One site can be relaxed in both charge states in one batch.
        noun = "site" if len(_submission_charges(batch.record)) == 1 else "relaxation"
    else:
        noun = "job"
    text = f"{plural(len(jobs), noun)} on {_target_text(batch.record)}"
    processes = sorted(
        {job["num_processes"] for job in jobs if isinstance(job.get("num_processes"), int)}
    )
    if len(processes) == 1:
        each = " each" if len(jobs) > 1 else ""
        text += f", {plural(processes[0], 'process', 'processes')}{each}"
    elif processes:
        text += f", {processes[0]} to {processes[-1]} processes"
    return f"{text} · {outcome}" if outcome else text


def _batch_entry(batch: _SubmittedBatch, configuration: ProvenanceRecord | None) -> JournalEntry:
    """The submission's card, with the batch's outcome and one row per job.

    ``configuration`` is the workflow's latest configuration at the submission;
    the card's details give the parameters its rows for these jobs share.
    """

    entry = _submit_jobs_entry(batch.record)
    outcome = _batch_outcome_text(batch)
    labels = _job_list_labels(list(batch.jobs.values()), _submission_charges(batch.record))
    rows = tuple(
        _job_row(label, batch.outcomes.get(job_id))
        for label, job_id in zip(labels, batch.jobs, strict=True)
    )
    body = f"{entry.body} {outcome}." if outcome else entry.body
    row_keys = {(str(job.get("row_label")), job.get("charge_state")) for job in batch.jobs.values()}
    line = _configuration_parameters_line(configuration, row_keys) if configuration else ""
    stage = batch.record.workflow_stage
    return replace(
        entry,
        body=body,
        job_rows=rows,
        summary=_batch_summary(batch, outcome),
        detail_lines=(line,) if line else (),
        job_column="Site" if stage == WorkflowStage.MUON_RELAXATION else "Calculation",
    )


def _job_row(label: str, outcome: ProvenanceRecord | None) -> JournalJobRow:
    if outcome is None:
        return JournalJobRow(label=label, status="submitted")
    return JournalJobRow(
        label=label,
        status=_TERMINAL_OUTCOMES[outcome.action],
        total_energy=_float_param(outcome, "result_total_energy"),
        max_force=_float_param(outcome, "result_max_force"),
        warnings_count=_int_param(outcome, "warnings_count"),
        failure_reason=_failure_text(outcome) or None,
    )


def _finished_entry(batch: _SubmittedBatch) -> JournalEntry | None:
    """The line under a later day saying a batch finished; ``None`` if it has not."""

    last = batch.last_outcome
    if (
        last is None
        or len(batch.outcomes) < len(batch.jobs)
        or _record_day(last) == _record_day(batch.record)
    ):
        return None
    launch = _submit_jobs_entry(batch.record)
    all_completed = all(record.action == "job_completed" for record in batch.outcomes.values())
    return replace(
        launch,
        timestamp=last.timestamp,
        stamp_kind="success" if all_completed else "warning",
        title=f"Finished: {_batch_outcome_text(batch)}",
        body=f"{launch.title} on {_record_day(batch.record)}.",
    )


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


def _dismissals_entry(records: Sequence[ProvenanceRecord]) -> JournalEntry:
    """One card for a run of dismissals; a single dismissal keeps its own wording."""

    if len(records) == 1:
        return _job_dismissed_entry(records[0])
    parameters = [record.parameters for record in records]
    charges = _charge_states(job.get("charge_state") for job in parameters)
    labels = _label_summary(_job_list_labels(parameters, charges))
    stages = {record.workflow_stage for record in records}
    entry = _entry(
        records[0],
        stamp_kind="milestone",
        title="Calculations dismissed",
        body=(
            f"Dismissed {plural(len(records), 'calculation')} from the Job Dashboard"
            f"{f': {labels}' if labels else ''}."
        ),
    )
    stage = records[0].workflow_stage if len(stages) == 1 else None
    return replace(
        entry,
        workflow_label=_workflow_label(stage),
        route=_WORKFLOW_ROUTES.get(stage),
        open_label=_open_label(stage),
        linked_entity_ids=tuple(
            entity
            for record in records
            for entity in (*record.input_entity_ids, *record.output_entity_ids)
        ),
        linked_job_id=None,
    )


def _create_comparison_table_entry(record: ProvenanceRecord) -> JournalEntry:
    table_label = _text_param(record, "table_label") or "comparison table"
    count = _int_param(record, "dft_structure_count") or 0
    entry = _entry(
        record,
        stamp_kind="success",
        title="Comparison table created",
        body=f"Created '{table_label}' using {plural(count, 'relaxed DFT structure')}.",
    )
    if not record.output_entity_ids:
        return entry
    # The record lists the table itself first, then the structures it compares.
    return replace(
        entry,
        route=comparison_route(record.output_entity_ids[0]),
        open_label=f"Open in '{table_label}'",
    )


def _terminal_job_entry(record: ProvenanceRecord, *, stamp_kind: str, outcome: str) -> JournalEntry:
    charge = _charge_state_text(record.parameters.get("charge_state"))
    title = f"{charge.capitalize()} relaxation {outcome}" if charge else f"Calculation {outcome}"
    summary = _job_summary_text(record)
    verb = _OUTCOME_VERBS[outcome]
    body = f"{_job_description(record, sentence_start=True)} {verb} on {_target_text(record)}."
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
    failure = _failure_text(record)
    if failure:
        parts.append(failure)
    return " ".join(parts)


def _failure_text(record: ProvenanceRecord) -> str:
    """Why a job did not complete: its diagnosed reason and the recorded message."""

    parts: list[str] = []
    try:
        parts.append(FAILURE_REASON_SUMMARIES[FailureReason(_text_param(record, "failure_reason"))])
    except (KeyError, ValueError):
        pass
    error_message = _text_param(record, "error_message")
    if error_message:
        sentence = error_message.rstrip(".") + "."
        # A walltime stop records the reason's own summary as its message.
        if sentence not in parts:
            parts.append(sentence)
    return " ".join(parts)


def _quantity_text(value: object, default_unit: str) -> str | None:
    """A recorded quantity, either a number or ``{"value": ..., "unit": ...}``, as text."""

    unit = default_unit
    if isinstance(value, dict):
        unit = str(value.get("unit") or default_unit)
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return f"{value:g} {unit}"


def _grid_text(value: object) -> str | None:
    """A k-point grid of three whole numbers as ``4×4×6``."""

    if not isinstance(value, list) or len(value) != 3:
        return None
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return None
    return "×".join(str(item) for item in value)


def _supercell_text(matrix: object) -> str | None:
    """A supercell matrix as ``Supercell 2×2×2``, or by its rows when it is not diagonal."""

    if not isinstance(matrix, list) or len(matrix) != 3:
        return None
    if not all(isinstance(row, list) and len(row) == 3 for row in matrix):
        return None
    if all(matrix[i][j] == 0 for i in range(3) for j in range(3) if i != j):
        return f"Supercell {matrix[0][0]}×{matrix[1][1]}×{matrix[2][2]}"
    rows = "; ".join(" ".join(str(item) for item in row) for row in matrix)
    return f"Supercell [{rows}]"


def _shared_value(
    rows: Sequence[Mapping[str, object]], read: Callable[[Mapping[str, object]], object]
) -> object:
    """What ``read`` gives for every row, or ``None`` when the rows differ or there are none."""

    values = [read(row) for row in rows]
    return values[0] if values and all(value == values[0] for value in values) else None


def _row_parameter(row: Mapping[str, object], key: str) -> object:
    # Older records keep a row's settings under ``params``; provenance is never rewritten.
    values = row.get("parameter_values", row.get("params"))
    return values.get(key) if isinstance(values, dict) else None


def _configuration_parameters_line(
    record: ProvenanceRecord, row_keys: set[tuple[str, object]] | None = None
) -> str:
    """The supercell and the shared parameters a configuration recorded, on one line.

    A row's parameter counts when every row (of the ``(label, charge state)`` pairs
    in ``row_keys``, when given) has the same value. Parameters left at the project
    default are not in the record.
    """

    configuration = record.parameters.get("configuration")
    if not isinstance(configuration, dict):
        return ""
    raw_rows = configuration.get("rows")
    rows = [row for row in raw_rows if isinstance(row, dict)] if isinstance(raw_rows, list) else []
    if row_keys is not None and rows:
        rows = [row for row in rows if (str(row.get("label")), row.get("charge_state")) in row_keys]
        if not rows:
            # These rows are not what the batch ran, so they say nothing about it.
            return ""
    shared = configuration.get("shared_params")
    parameters = dict(shared) if isinstance(shared, dict) else {}
    for key in ("spin_polarized", "xc_functional", "cut_off_energy", "kpoint_mp_grid"):
        value = _shared_value(rows, lambda row, key=key: _row_parameter(row, key))
        if value is not None:
            parameters[key] = value

    parts: list[str | None] = []
    if rows:
        parts.append(_supercell_text(_shared_value(rows, lambda row: row.get("supercell_matrix"))))
    elif record.parameters.get("simulation_cell_mode") == "unit_cell":
        parts.append("Unit cell")
    spin = parameters.get("spin_polarized")
    if isinstance(spin, bool):
        parts.append("spin polarized" if spin else "not spin polarized")
    functional = parameters.get("xc_functional")
    parts.append(str(functional) if functional else None)
    parts.append(_quantity_text(parameters.get("cut_off_energy"), "eV"))
    parts.append(_grid_text(parameters.get("kpoint_mp_grid")))
    return " · ".join(part for part in parts if part)


def _sweep_line(record: ProvenanceRecord) -> str:
    """A convergence sweep's range: ``6 cutoff values: 300 eV to 800 eV``."""

    configuration = record.parameters.get("configuration")
    if not isinstance(configuration, dict):
        return ""
    # Records written before the schema 2 migration keep the cutoffs under their legacy key.
    values = configuration.get("sweep_values") or configuration.get("cutoff_energies")
    if not isinstance(values, list) or not values:
        return ""
    if record.action == "configure_convergence_kpoint":
        noun, first, last = "grid value", _grid_text(values[0]), _grid_text(values[-1])
    else:
        noun = "cutoff value"
        first, last = _quantity_text(values[0], "eV"), _quantity_text(values[-1], "eV")
    if first is None or last is None:
        return ""
    span = first if len(values) == 1 else f"{first} to {last}"
    return f"{plural(len(values), noun)}: {span}"


def _configuration_detail_lines(record: ProvenanceRecord) -> tuple[str, ...]:
    """Shared parameters and the supercell; a long row list collapses to a count and three names."""

    lines = [_configuration_parameters_line(record), _sweep_line(record)]
    configuration = record.parameters.get("configuration")
    if isinstance(configuration, dict):
        for key, noun in (("rows", "row"), ("sources", "source")):
            items = configuration.get(key)
            if isinstance(items, list) and items:
                labels = _label_summary(
                    [item.get("label") for item in items if isinstance(item, dict)]
                )
                lines.append(
                    f"{plural(len(items), noun)}: {labels}" if labels else plural(len(items), noun)
                )
    return tuple(line for line in lines if line)


def _default_detail_lines(prefix: str) -> Callable[[ProvenanceRecord], tuple[str, ...]]:
    """Details of an adopted default: the tolerances it met and its functional."""

    def build(record: ProvenanceRecord) -> tuple[str, ...]:
        parts = [part[:1].upper() + part[1:] for part in _threshold_parts(record, prefix)]
        functional = _text_param(record, f"{prefix}_xc_functional")
        if functional:
            parts.append(f"Functional {functional}")
        return (" · ".join(parts),) if parts else ()

    return build


def _generation_detail_lines(record: ProvenanceRecord) -> tuple[str, ...]:
    """The run's inputs and seed, in the words of its methods sentence."""

    run = generation_run_from_provenance(record.parameters)
    return (run_methods_sentence(run),) if run is not None else ()


def _legacy_import_detail_lines(record: ProvenanceRecord) -> tuple[str, ...]:
    """The source folder, every skipped run with its reason, and every conflict."""

    lines: list[str] = []
    source = _text_param(record, "source_folder")
    if source:
        lines.append(f"Source folder {source}")
    skipped = record.parameters.get("skipped")
    for item in skipped if isinstance(skipped, list) else []:
        if isinstance(item, dict):
            lines.append(f"Skipped run {item.get('site_index')}: {item.get('reason')}")
    sites = record.parameters.get("imported_sites")
    for site in sites if isinstance(sites, list) else []:
        if isinstance(site, dict) and isinstance(site.get("conflicts"), list):
            lines.extend(f"{site.get('label')}: {conflict}" for conflict in site["conflicts"])
    return tuple(lines)


_CONFIGURE_ACTIONS = frozenset(
    {
        "configure_convergence_cutoff",
        "configure_convergence_kpoint",
        "configure_pristine_relaxation",
        "configure_uep",
        "configure_muon_relaxation",
        "configure_quantum_corrections",
        "configure_energy_barriers",
    }
)

#: What each kind of record adds to its expanded card, beyond the exact time.
_DETAIL_BUILDERS: dict[str, Callable[[ProvenanceRecord], tuple[str, ...]]] = {
    **dict.fromkeys(_CONFIGURE_ACTIONS, _configuration_detail_lines),
    "set_recommended_cutoff": _default_detail_lines("recommended_cutoff"),
    "set_recommended_kpoint_grid": _default_detail_lines("recommended_kpoint"),
    "generate_candidates": _generation_detail_lines,
    "import_legacy_relaxations": _legacy_import_detail_lines,
}


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
