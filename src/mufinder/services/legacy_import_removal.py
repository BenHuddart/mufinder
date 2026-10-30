"""Remove an imported legacy relaxation, or refuse because something needs it.

An imported relaxation owns the only copy of its results: the ``.castep``, the
relaxed cell, and the archive of the originals all live in its job directory and
exist nowhere else in the project. Removing one is therefore irreversible in a
way that removing a natively-run relaxation is not — that one can be re-run.

So removal is refused while anything still points at the site, run, job or
structure. The scan behind that refusal is the substance of this module, and it
has to be **wider than the two obvious workflows**: a reference it misses does
not become a warning, it becomes a dangling id in a saved project.

Referrers checked, and where each keeps its ids:

===============================  =========================================
Referrer                         Stored as
===============================  =========================================
Quantum Corrections, TS, NEB     ``CalculationDefinition.input_site_ids``
Any definition's inputs          ``CalculationDefinition.input_structure_ids``
Comparison tables                ``ComparisonTable.dft_structure_ids`` and
                                 ``experimental_structure_id``
Energy-barrier path sources      ``ui_state["energy_barriers_form"]``,
                                 ``generated_sources`` entries
Relaxation-analysis workspaces   ``ui_state["relaxation_analysis_workspaces"]``,
                                 ``job_ids``
===============================  =========================================

The muon-relaxation definition that *owns* the imported rows is deliberately not
treated as a referrer: it is the thing being removed from, not a dependant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shutil
from typing import Any, Sequence
from uuid import UUID

from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import ProvenanceRecord
from mufinder.domain.text import plural
from mufinder.persistence.codec import uuid_or_none

#: Provenance action for one removal. ``services.journal`` renders nothing for
#: an action it does not know, so a matching journal builder is required.
REMOVAL_ACTION = "remove_imported_relaxations"

_ENERGY_BARRIER_FORM_KEY = "energy_barriers_form"
_ANALYSIS_WORKSPACES_KEY = "relaxation_analysis_workspaces"


@dataclass(frozen=True, slots=True)
class ImportedRelaxationRef:
    """The entities one imported relaxation owns."""

    site_id: UUID
    job_id: UUID
    run_id: UUID | None = None
    result_id: UUID | None = None
    structure_id: UUID | None = None
    label: str = ""

    def entity_ids(self) -> set[UUID]:
        """Return every id another entity could point at."""

        return {
            value
            for value in (self.site_id, self.job_id, self.run_id, self.result_id, self.structure_id)
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class RemovalPlan:
    """What removing a set of imported relaxations would do."""

    targets: tuple[ImportedRelaxationRef, ...] = ()
    #: User-facing descriptions of what still needs these entities. Non-empty
    #: means the removal must be refused.
    blockers: tuple[str, ...] = ()
    directories: tuple[Path, ...] = ()

    @property
    def allowed(self) -> bool:
        """Return whether the removal may proceed."""

        return bool(self.targets) and not self.blockers


@dataclass(frozen=True, slots=True)
class RemovalOutcome:
    """What one removal actually did."""

    removed: int = 0
    errors: list[str] = field(default_factory=list)
    #: The removal's provenance record, for the caller to persist. Returned
    #: rather than appended to the document, because appending only reaches
    #: memory: ``provenance.jsonl`` is append-only through
    #: ``ProjectService.append_provenance`` and ``save_project`` never rewrites
    #: it, so a record appended here would vanish on reload. ``None`` when
    #: nothing was removed.
    provenance: ProvenanceRecord | None = None


def imported_relaxations_in_batch(document: Any, batch_id: UUID) -> list[ImportedRelaxationRef]:
    """Return the imported relaxations belonging to one import batch."""

    refs: list[ImportedRelaxationRef] = []
    for site in document.sites.values():
        if str(site.properties.get("import_batch_id", "")) != str(batch_id):
            continue
        run = document.runs.get(site.relaxation_run_id) if site.relaxation_run_id else None
        result = document.results.get(run.result_id) if run and run.result_id else None
        structure_id = (
            result.derived_structure_ids[0]
            if result is not None and result.derived_structure_ids
            else None
        )
        if run is None or run.job_id is None:
            continue
        refs.append(
            ImportedRelaxationRef(
                site_id=site.id,
                job_id=run.job_id,
                run_id=run.id,
                result_id=result.id if result is not None else None,
                structure_id=structure_id,
                label=site.label,
            )
        )
    return refs


def _definition_owns_targets(definition: Any, targets: set[UUID]) -> bool:
    """Return whether ``definition`` is the muon-relaxation owner, not a referrer."""

    return definition.workflow_stage == WorkflowStage.MUON_RELAXATION


def find_references(document: Any, targets: Sequence[ImportedRelaxationRef]) -> list[str]:
    """Return user-facing descriptions of everything still needing ``targets``.

    An empty list means removal is safe.
    """

    site_ids = {ref.site_id for ref in targets}
    structure_ids = {ref.structure_id for ref in targets if ref.structure_id is not None}
    job_ids = {ref.job_id for ref in targets}
    run_ids = {ref.run_id for ref in targets if ref.run_id is not None}
    all_ids = set().union(*(ref.entity_ids() for ref in targets)) if targets else set()

    references: list[str] = []

    for definition in document.calculations.values():
        if _definition_owns_targets(definition, all_ids):
            continue
        if site_ids.intersection(definition.input_site_ids) or structure_ids.intersection(
            definition.input_structure_ids
        ):
            references.append(
                f"{definition.workflow_stage.value} calculation '{definition.calc_type}'"
            )

    for table in document.comparison_tables.values():
        if structure_ids.intersection(table.dft_structure_ids) or (
            table.experimental_structure_id in structure_ids
        ):
            references.append(f"comparison table '{table.label}'")

    form = document.ui_state.get(_ENERGY_BARRIER_FORM_KEY)
    if isinstance(form, dict):
        sources = form.get("generated_sources")
        if isinstance(sources, list):
            for entry in sources:
                if not isinstance(entry, dict):
                    continue
                referenced = {
                    uuid_or_none(entry.get(key))
                    for key in (
                        "reactant_site_id",
                        "product_site_id",
                        "reactant_run_id",
                        "product_run_id",
                        "reactant_job_id",
                        "product_job_id",
                        "reactant_structure_id",
                        "product_structure_id",
                    )
                }
                if referenced.intersection(site_ids | run_ids | job_ids | structure_ids):
                    references.append(
                        f"energy-barrier path '{entry.get('label', entry.get('key', 'path'))}'"
                    )

    workspaces = document.ui_state.get(_ANALYSIS_WORKSPACES_KEY)
    if isinstance(workspaces, list):
        for workspace in workspaces:
            if not isinstance(workspace, dict):
                continue
            raw_jobs = workspace.get("job_ids")
            if not isinstance(raw_jobs, list):
                continue
            if {uuid_or_none(value) for value in raw_jobs}.intersection(job_ids):
                references.append(
                    f"relaxation analysis workspace '{workspace.get('name', 'workspace')}'"
                )

    # Stable, de-duplicated: the same workspace can reference several targets.
    return list(dict.fromkeys(references))


def plan_removal(
    document: Any,
    targets: Sequence[ImportedRelaxationRef],
    *,
    job_directories: dict[UUID, Path] | None = None,
) -> RemovalPlan:
    """Describe what removing ``targets`` would do, or why it cannot happen."""

    directories = tuple(
        path for ref in targets if (path := (job_directories or {}).get(ref.job_id)) is not None
    )
    return RemovalPlan(
        targets=tuple(targets),
        blockers=tuple(find_references(document, targets)),
        directories=directories,
    )


def remove_imported_relaxations(
    document: Any,
    plan: RemovalPlan,
    *,
    batch_id: UUID | None = None,
    delete_files: bool = True,
) -> RemovalOutcome:
    """Delete the entities and files in ``plan``.

    Refuses outright when the plan is not :attr:`~RemovalPlan.allowed`, so a
    caller cannot skip the reference check by constructing a plan by hand.

    Only jobs still carrying the imported tag are removed. That guard matters
    because the ids in a plan are resolved before any confirmation dialog: if
    something has changed underneath in the meantime, the honest outcome is to
    remove nothing rather than to delete a job MuFinder itself ran.
    """

    if not plan.allowed:
        return RemovalOutcome(
            removed=0,
            errors=["Removal was refused because something still references these relaxations."],
        )

    errors: list[str] = []
    removed = 0
    removed_labels: list[str] = []

    for ref in plan.targets:
        job = document.jobs.get(ref.job_id)
        if job is None or IMPORTED_JOB_TAG not in (job.tags or ()):
            errors.append(f"'{ref.label}' was not removed: its job is no longer an imported job.")
            continue

        document.jobs.pop(ref.job_id, None)
        document.sites.pop(ref.site_id, None)
        if ref.run_id is not None:
            document.runs.pop(ref.run_id, None)
        if ref.result_id is not None:
            document.results.pop(ref.result_id, None)
        if ref.structure_id is not None:
            document.structures.pop(ref.structure_id, None)
        removed += 1
        removed_labels.append(ref.label)

    if delete_files:
        for directory in plan.directories:
            if not directory.exists():
                continue
            try:
                shutil.rmtree(directory)
            except OSError as exc:
                errors.append(f"Could not delete {directory.name}: {exc}")

    record: ProvenanceRecord | None = None
    if removed:
        record = ProvenanceRecord(
            action=REMOVAL_ACTION,
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            actor="user",
            description=f"Removed {plural(removed, 'imported muon relaxation')}",
            parameters={
                "batch_id": str(batch_id) if batch_id is not None else None,
                "labels": removed_labels,
                "deleted_files": bool(delete_files),
                "directories": [str(path) for path in plan.directories],
            },
            input_entity_ids=[ref.site_id for ref in plan.targets],
        )

    return RemovalOutcome(removed=removed, errors=errors, provenance=record)


def detach_rows_from_definition(definition: Any, targets: Sequence[ImportedRelaxationRef]) -> int:
    """Drop removed rows and their job bindings from a muon-relaxation definition.

    Rows are matched on ``site_id`` rather than label: a label can be edited or
    suffixed for uniqueness, but the site id is the row's stable identity.
    """

    configuration = definition.parameters.get("configuration")
    if not isinstance(configuration, dict):
        return 0
    rows = configuration.get("rows")
    if not isinstance(rows, list):
        return 0

    site_ids = {str(ref.site_id) for ref in targets}
    keep = [row for row in rows if str(row.get("site_id", "")) not in site_ids]
    dropped_labels = {
        str(row.get("label", "")) for row in rows if str(row.get("site_id", "")) in site_ids
    }
    configuration["rows"] = keep

    job_ids = {str(ref.job_id) for ref in targets}
    for mapping_name in ("job_ids_by_label", "job_seed_names_by_label"):
        mapping = definition.parameters.get(mapping_name)
        if not isinstance(mapping, dict):
            continue
        for key in [
            key
            for key, value in mapping.items()
            if (mapping_name == "job_ids_by_label" and str(value) in job_ids)
            or (str(key).partition(":")[2] in dropped_labels)
        ]:
            mapping.pop(key, None)

    site_uuids = {ref.site_id for ref in targets}
    definition.input_site_ids = [
        site_id for site_id in definition.input_site_ids if site_id not in site_uuids
    ]
    return len(rows) - len(keep)


__all__ = [
    "REMOVAL_ACTION",
    "ImportedRelaxationRef",
    "RemovalOutcome",
    "RemovalPlan",
    "detach_rows_from_definition",
    "find_references",
    "imported_relaxations_in_batch",
    "plan_removal",
    "remove_imported_relaxations",
]
