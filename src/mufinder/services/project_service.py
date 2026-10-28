"""Project-level orchestration for create/load/save workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import shutil
from typing import Any
from uuid import UUID

from mufinder.domain.enums import ACTIVE_JOB_STATUSES, FailureReason, JobStatus, WorkflowStage
from mufinder.domain.magnetic_properties import propagation_vector_from_structure
from mufinder.domain.models import (
    CalculationDefinition,
    ComparisonTable,
    Job,
    MuonSite,
    Project,
    ProvenanceRecord,
    Result,
    Structure,
    next_definition_created_at,
    next_job_created_at,
    utc_now,
)
from mufinder.persistence import project_codec
from mufinder.persistence.codec import to_jsonable
from mufinder.persistence.json_store import ProjectStore
from mufinder.persistence.migrations import migrate_project_payloads
from mufinder.persistence.project_codec import normalized_xc_functional
from mufinder.services import site_generation_service, supercell_assets
from mufinder.services.perf_trace import PerfTrace

# The document aggregate, its read-only queries and the supercell records are
# re-exported: callers import them from this module.
from mufinder.services.project_document import (
    ProjectDocument,
    active_definition as active_definition,
    active_host_structure,
    definitions_for_stage as definitions_for_stage,
    job_for_mapping_key as job_for_mapping_key,
)
from mufinder.services.supercell_assets import (
    WORKFLOW_SUPERCELL_UNIT_CELL,
    WorkflowStructureResolution,
    WorkflowSupercellAsset,
    WorkflowSupercellSelection,
)
from mufinder.science.structure_io import load_structure_file
from mufinder.workflow.muon_site import MUON_RELAXATION_DEFINITION_CALC_TYPE

_LOG = logging.getLogger(__name__)


def _project_relative_path(path: Path, project_root: Path) -> str:
    """Return ``path`` relative to the project root, or as given when outside it."""

    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def _known_input_structure(
    document: ProjectDocument,
    input_structure_id: UUID | None,
    host_structure_id: UUID | None,
) -> UUID:
    """Resolve a definition's single input structure, falling back to the host id."""

    if input_structure_id is None:
        input_structure_id = host_structure_id
    if input_structure_id is None or input_structure_id not in document.structures:
        raise ValueError("input_structure_id must reference a known structure.")
    return input_structure_id


def _require_known_inputs(
    document: ProjectDocument,
    input_structure_ids: Sequence[UUID],
    input_site_ids: Sequence[UUID],
) -> None:
    """Reject a multi-input definition with no structure or an unknown structure or site."""

    if not input_structure_ids:
        raise ValueError("At least one input structure is required.")
    if any(structure_id not in document.structures for structure_id in input_structure_ids):
        raise ValueError("input_structure_ids must reference known structures.")
    if any(site_id not in document.sites for site_id in input_site_ids):
        raise ValueError("input_site_ids must reference known sites.")


#: Terminal job statuses a job can be recovered from without losing its record:
#: :meth:`ProjectService.reset_failed_job` accepts exactly these, and
#: :meth:`ProjectService.dismiss_job` hides exactly these from the dashboard.
#: The two must stay identical — a dismissible job has to stay resettable, which
#: is what makes a dismissed row resubmittable from its workflow table. See
#: ``mufinder.app.workflow_submission.RESUBMITTABLE_AFTER_DISMISSAL_STATUSES``
#: (kept in lockstep by a test) and docs/job_lifecycle.rst.
TERMINAL_RECOVERABLE_JOB_STATUSES = frozenset(
    {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}
)

#: Job statuses a calculation can be *continued* from:
#: :meth:`ProjectService.prepare_job_continuation` accepts exactly these. The set
#: is deliberately wider than :data:`TERMINAL_RECOVERABLE_JOB_STATUSES` because a
#: COMPLETED relaxation is continuable — running more geometry-optimisation steps
#: from its checkpoint is a normal outcome, not a recovery from failure.
#: Widening it does not weaken the dismissed-row reasoning above: continuation is
#: a distinct primitive with its own entry point, and the reset/dismiss sets stay
#: identical to each other. See docs/job_lifecycle.rst.
CONTINUABLE_JOB_STATUSES = frozenset(
    {
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.STOPPED,
        JobStatus.CANCELLED,
    }
)

#: Message stamped on a local job that was still in progress when MuFinder was
#: closed and is reconciled to STOPPED on the next project open. See
#: ``ProjectService.reconcile_orphaned_local_jobs`` and docs/job_lifecycle.rst.
ORPHANED_LOCAL_JOB_STOP_MESSAGE = (
    "MuFinder was closed while this job was in progress locally; reset or resubmit it."
)


@dataclass(slots=True, frozen=True)
class PromotableHost:
    """A structure eligible to serve as the project's active host."""

    structure_id: UUID
    label: str
    provenance_hint: str
    kind: str  # "imported" or "relaxed"
    created_at: datetime | None


@dataclass(slots=True)
class SiteHostMismatch:
    """Sites whose source structure differs from the active host."""

    active_host_id: UUID
    active_host_label: str
    offenders: list[
        tuple[UUID, str, UUID | None, str]
    ]  # (site_id, label, source_host_id, source_host_label)


@dataclass(slots=True)
class StructureHostMismatch:
    """Structures whose provenance chain doesn't resolve to the active host."""

    active_host_id: UUID
    active_host_label: str
    offenders: list[
        tuple[UUID, str, UUID | None, str]
    ]  # (struct_id, label, resolved_host_id, host_label)


class ProjectService:
    """Creates, loads, and saves projects using ProjectStore.

    Dirty tracking
    --------------
    ``save_project`` rewrites every entity JSON file on every call, which is
    expensive on mature projects (hundreds of ``json.dumps`` + file writes per
    dashboard flush, all on the GUI thread). To avoid that, the service keeps a
    private content-hash cache mapping ``(entity_type, entity_id)`` to the hash
    of the canonical serialized payload that was last written to (or loaded
    from) disk. On the next save an entity whose payload hashes identically and
    whose file still exists is skipped.

    The cache is keyed by the resolved project root (:meth:`_resolved_root`) so
    that a single long-lived ``ProjectService`` instance (see
    ``MainWindow._project_service``) reused across projects never applies one
    project's hashes to another. It is reset/reseeded on :meth:`load_project`
    and :meth:`create_project`. This cache is purely a save-time optimization;
    it holds no persisted state and never changes the on-disk format.

    Concurrency: ``save_project`` runs on the GUI thread only, so the cache is
    accessed single-threaded and needs no locking. It is kept private to the
    service and is not exposed to callers.
    """

    #: Cache sentinel identifying the singleton ui-state document. It is not a
    #: real entity directory; ``ui_state`` is written via ``write_json`` to
    #: ``state.json`` rather than ``write_entity``.
    _UI_STATE_ENTITY_TYPE = "ui_state"
    _UI_STATE_ENTITY_ID = "ui_state"

    def __init__(self) -> None:
        # (project_root, entity_type, entity_id) -> hash of the canonical
        # serialized payload last persisted for that entity. "entity_id" is the
        # entity UUID string for real entities, and a sentinel ("ui_state") for
        # the singleton ui-state document.
        self._content_hashes: dict[tuple[str, str, str], str] = {}

    @staticmethod
    def _entity_groups(
        document: ProjectDocument,
    ) -> list[tuple[str, list[tuple[str, Any]]]]:
        """Return the per-type ``(entity_type, [(entity_id, entity), ...])`` groups.

        This single ordered description drives both cache seeding on load and
        the prune/write loop on save, so the two can never disagree about which
        entities exist or which directory they live in.
        """
        return [
            ("structures", [(str(s.id), s) for s in document.structures.values()]),
            ("sites", [(str(s.id), s) for s in document.sites.values()]),
            ("calculations", [(str(c.id), c) for c in document.calculations.values()]),
            ("runs", [(str(r.id), r) for r in document.runs.values()]),
            ("results", [(str(r.id), r) for r in document.results.values()]),
            ("comparison_tables", [(str(t.id), t) for t in document.comparison_tables.values()]),
            ("jobs", [(str(j.id), j) for j in document.jobs.values()]),
        ]

    @staticmethod
    def _resolved_root(project_root: Path) -> str:
        """Return a stable cache key for a project root.

        Resolving guards against the same project being addressed via different
        but equivalent paths (e.g. relative vs absolute) within one session.
        """
        try:
            return str(Path(project_root).resolve())
        except OSError:
            return str(project_root)

    @staticmethod
    def _payload_hash(payload: dict[str, Any]) -> str:
        """Hash the exact canonical string ``ProjectStore`` would write.

        Using :meth:`ProjectStore.canonical_json` (the same helper
        ``write_json`` uses) guarantees the hash tracks precisely the bytes on
        disk, so the two can never drift.
        """
        canonical = ProjectStore.canonical_json(payload)
        return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()

    def _reset_content_hashes(self, project_root: Path) -> None:
        """Drop all cached hashes for one project root.

        Called when a project is (re)loaded or created so that stale hashes
        from a previous in-memory document for the same root cannot cause a
        genuinely-changed entity to be skipped.
        """
        root_key = self._resolved_root(project_root)
        for key in [key for key in self._content_hashes if key[0] == root_key]:
            del self._content_hashes[key]

    def create_project(self, project_root: Path, name: str) -> ProjectDocument:
        """Create a new project document and persist baseline files."""
        self._reset_content_hashes(project_root)
        project = Project(name=name, modified_at=utc_now())
        document = ProjectDocument(project_root=project_root, project=project)
        self.save_project(document)
        return document

    def load_project(self, project_root: Path) -> ProjectDocument:
        """Load a project from disk into a ProjectDocument."""
        store = ProjectStore(project_root)
        manifest = store.read_json(store.project_manifest_path)
        payloads = {
            entity_type: store.load_entities(entity_type)
            for entity_type in (
                "structures",
                "sites",
                "calculations",
                "runs",
                "results",
                "comparison_tables",
                "jobs",
            )
        }
        ui_state = store.read_json(store.state_path) if store.state_path.exists() else {}
        # Projects saved at an older schema are upgraded in memory; the entity
        # files (and ui_state) a migration changed are left out of the
        # dirty-tracking cache below so the next save writes them with the new
        # manifest version.
        migrated_keys = migrate_project_payloads(manifest, payloads, ui_state)

        project = project_codec.project_from_dict(manifest)
        # Each entity directory is named after the ``ProjectDocument`` mapping it fills.
        entities = {
            entity_type: {entity.id: entity for entity in map(decode, payloads[entity_type])}
            for entity_type, decode in project_codec.ENTITY_DECODERS.items()
        }
        provenance = [project_codec.provenance_from_dict(p) for p in store.load_provenance()]
        job_payloads = payloads["jobs"]
        jobs = project_codec.jobs_in_creation_order(job_payloads, provenance)
        # Jobs saved before ``created_at`` existed carry a backfilled value that
        # is not on disk yet; leaving them out of the cache makes the next save
        # write it, so later loads use the persisted value.
        unsaved_keys = migrated_keys | {
            ("jobs", payload["id"]) for payload in job_payloads if payload.get("created_at") is None
        }

        document = ProjectDocument(
            project_root=project_root,
            project=project,
            **entities,
            jobs=jobs,
            provenance=provenance,
            ui_state=ui_state,
        )

        # Reset then seed the dirty-tracking cache from the payloads just loaded
        # off disk so the first save after opening a project does not rewrite
        # every entity. Seeding uses ``to_jsonable(entity)`` (what a save would
        # write); a regression test asserts this round-trips to the exact
        # loaded payload for representative entities. The ui_state seeded here
        # is the raw on-disk value; ``_integrity_warnings`` injected below is a
        # transient in-memory annotation and is intentionally not part of the
        # seeded hash, so it will be written on the first save if present.
        self._reset_content_hashes(project_root)
        root_key = self._resolved_root(project_root)
        for entity_type, entities in self._entity_groups(document):
            for entity_id, entity in entities:
                if (entity_type, entity_id) in unsaved_keys:
                    continue
                self._content_hashes[(root_key, entity_type, entity_id)] = self._payload_hash(
                    to_jsonable(entity)
                )
        if (self._UI_STATE_ENTITY_TYPE, self._UI_STATE_ENTITY_ID) not in unsaved_keys:
            self._content_hashes[
                (root_key, self._UI_STATE_ENTITY_TYPE, self._UI_STATE_ENTITY_ID)
            ] = self._payload_hash(to_jsonable(ui_state))

        issues = self.validate_document(document)
        if issues:
            document.ui_state["_integrity_warnings"] = issues
        return document

    def save_project(self, document: ProjectDocument) -> None:
        """Persist all mutable project entities to disk."""
        issues = self.validate_document(document)
        if issues:
            issue_text = "\n".join(f"- {issue}" for issue in issues)
            raise ValueError(f"Project integrity check failed:\n{issue_text}")

        trace = PerfTrace(label="save_project", logger=_LOG)
        document.project.modified_at = utc_now()
        store = ProjectStore(document.project_root)
        store.initialize_layout()
        root_key = self._resolved_root(document.project_root)

        # The manifest carries ``modified_at`` (bumped every save above), so it
        # is always rewritten and deliberately not hash-tracked.
        store.write_json(store.project_manifest_path, to_jsonable(document.project))

        written = 0
        skipped = 0

        for entity_type, entities in self._entity_groups(document):
            # Prune first so externally-orphaned files (and files for entities
            # deleted in memory) are removed even when nothing else changed.
            store.prune_entities(entity_type, {entity_id for entity_id, _ in entities})
            for entity_id, entity in entities:
                payload = to_jsonable(entity)
                digest = self._payload_hash(payload)
                cache_key = (root_key, entity_type, entity_id)
                target = store.entities_root / entity_type / f"{entity_id}.json"
                # Skip only when the content is unchanged AND the file still
                # exists on disk; the ``exists()`` guard forces a rewrite if the
                # file was deleted externally since the cached hash was taken.
                if self._content_hashes.get(cache_key) == digest and target.exists():
                    skipped += 1
                    continue
                store.write_entity(entity_type, entity_id, payload)
                self._content_hashes[cache_key] = digest
                written += 1

        # Drop cached hashes for entities that no longer exist in memory so the
        # cache cannot grow unbounded across a long session. Their files were
        # already removed by ``prune_entities`` above.
        live_keys = {
            (root_key, entity_type, entity_id)
            for entity_type, entities in self._entity_groups(document)
            for entity_id, _ in entities
        }
        for key in [
            key
            for key in self._content_hashes
            if key[0] == root_key and key[1] != self._UI_STATE_ENTITY_TYPE and key not in live_keys
        ]:
            del self._content_hashes[key]

        # ui_state gets the same hash-skip treatment as entities.
        ui_payload = to_jsonable(document.ui_state)
        ui_digest = self._payload_hash(ui_payload)
        ui_key = (root_key, self._UI_STATE_ENTITY_TYPE, self._UI_STATE_ENTITY_ID)
        if self._content_hashes.get(ui_key) == ui_digest and store.state_path.exists():
            skipped += 1
        else:
            store.write_json(store.state_path, ui_payload)
            self._content_hashes[ui_key] = ui_digest
            written += 1

        trace.log_summary(extra=f"entity_files_written={written} entity_files_skipped={skipped}")

    def append_provenance(self, document: ProjectDocument, record: ProvenanceRecord) -> None:
        """Append a provenance record in-memory and to disk."""
        document.provenance.append(record)
        store = ProjectStore(document.project_root)
        store.append_provenance(to_jsonable(record))

    def add_structure(self, document: ProjectDocument, structure: Structure) -> None:
        """Add a structure and update host structure if currently unset."""
        document.structures[structure.id] = structure
        if document.project.host_structure_id is None:
            document.project.host_structure_id = structure.id

    def add_comparison_table(self, document: ProjectDocument, table: ComparisonTable) -> None:
        """Register one persisted comparison table in the current document."""
        document.comparison_tables[table.id] = table

    def rename_comparison_table(
        self, document: ProjectDocument, table_id: UUID, label: str
    ) -> ComparisonTable | None:
        """Update the display label for one persisted comparison table."""
        table = document.comparison_tables.get(table_id)
        if table is None:
            return None
        table.label = str(label).strip() or "Comparison Table"
        return table

    def remove_comparison_table(self, document: ProjectDocument, table_id: UUID) -> None:
        """Delete one persisted comparison table from the current document."""
        document.comparison_tables.pop(table_id, None)

    def remove_candidate_sites(
        self, document: ProjectDocument, site_ids: Sequence[UUID]
    ) -> list[MuonSite]:
        """Remove generated muon sites by id and record the edit in provenance."""
        outcome = site_generation_service.remove_candidate_sites(document, site_ids)
        return self._record_site_change(document, outcome)

    def cluster_candidate_positions(
        self,
        document: ProjectDocument,
        site_ids: Sequence[UUID],
        *,
        anchor_fractional: Sequence[float],
    ) -> list[MuonSite]:
        """Move selected generated muon sites to symmetry representatives near an anchor."""
        outcome = site_generation_service.cluster_candidate_positions(
            document, site_ids, anchor_fractional=anchor_fractional
        )
        return self._record_site_change(document, outcome)

    def stage_generated_sites(
        self, document: ProjectDocument, site_ids: Sequence[UUID]
    ) -> list[MuonSite]:
        """Mark generated muon sites as staged for structural relaxation."""
        outcome = site_generation_service.stage_generated_sites(document, site_ids)
        return self._record_site_change(document, outcome)

    def record_job_submission(
        self,
        document: ProjectDocument,
        *,
        jobs: Sequence[Job],
        execution_target: str,
    ) -> None:
        """Append one provenance milestone for a user submission action."""

        submitted_jobs = [
            job for job in jobs if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}
        ]
        if not submitted_jobs:
            return

        workflow_stage = (
            submitted_jobs[0].workflow_stage
            if len({job.workflow_stage for job in submitted_jobs}) == 1
            else None
        )
        self.append_provenance(
            document,
            ProvenanceRecord(
                action="submit_jobs",
                workflow_stage=workflow_stage,
                actor="user",
                description=f"Submitted {len(submitted_jobs)} job(s) to {execution_target}",
                parameters={
                    "execution_target": execution_target,
                    "job_count": len(submitted_jobs),
                    "job_labels": [job.label for job in submitted_jobs],
                    "jobs": [
                        {
                            "job_id": str(job.id),
                            "label": job.label,
                            "status": job.status.value,
                            "execution_target": job.execution_target,
                            "remote_job_id": job.remote_job_id,
                            "remote_directory": job.remote_directory,
                            "num_processes": job.num_processes,
                            "submitted_at": (
                                job.submitted_at.isoformat() if job.submitted_at else None
                            ),
                        }
                        for job in submitted_jobs
                    ],
                },
                output_entity_ids=[job.id for job in submitted_jobs],
            ),
        )

    def record_job_terminal_event(
        self,
        document: ProjectDocument,
        *,
        action: str,
        job: Job,
        result: Result | None = None,
        error_message: str | None = None,
    ) -> None:
        """Append one provenance milestone for a terminal job outcome."""

        parameters: dict[str, Any] = {
            "job_label": job.label,
            "calc_type": job.calc_type,
            "execution_target": job.execution_target,
        }
        if result is not None:
            data = result.data if isinstance(result.data, dict) else {}
            total_energy = data.get("total_energy")
            max_force = data.get("max_force")
            warnings = data.get("warnings", [])
            if total_energy is not None:
                parameters["result_total_energy"] = total_energy
            if max_force is not None:
                parameters["result_max_force"] = max_force
            if isinstance(warnings, list):
                parameters["warnings_count"] = len(warnings)
            if result.derived_structure_ids:
                parameters["derived_structure_count"] = len(result.derived_structure_ids)
        if error_message:
            parameters["error_message"] = error_message
        if job.failure_reason is not None and action in {"job_failed", "job_stopped"}:
            parameters["failure_reason"] = job.failure_reason.value

        output_entity_ids: list[UUID] = [job.id]
        if result is not None:
            output_entity_ids.append(result.id)
            output_entity_ids.extend(result.derived_structure_ids)

        descriptions = {
            "job_completed": f"Completed calculation {job.label}",
            "job_failed": f"Calculation {job.label} failed",
            "job_stopped": f"Calculation {job.label} stopped",
            "job_cancelled": f"Cancelled calculation {job.label}",
        }
        self.append_provenance(
            document,
            ProvenanceRecord(
                action=action,
                workflow_stage=job.workflow_stage,
                actor="system",
                description=descriptions.get(action, job.label),
                parameters=parameters,
                output_entity_ids=output_entity_ids,
                job_id=job.id,
            ),
        )

    def record_job_reset(
        self,
        document: ProjectDocument,
        *,
        job: Job,
        cleaned_output_directory: bool,
    ) -> None:
        """Append one provenance milestone when a job is reset for resubmission."""

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="job_reset",
                workflow_stage=job.workflow_stage,
                actor="user",
                description=f"Reset calculation {job.label} for resubmission",
                parameters={
                    "job_label": job.label,
                    "calc_type": job.calc_type,
                    "execution_target": job.execution_target,
                    "cleaned_output_directory": cleaned_output_directory,
                },
                output_entity_ids=[job.id],
                job_id=job.id,
            ),
        )

    def record_job_continuation(
        self,
        document: ProjectDocument,
        *,
        job: Job,
        remote_directory: str | None,
    ) -> None:
        """Append one provenance milestone when a job is prepared for continuation.

        ``remote_directory`` is the directory the continuation run will reuse
        (see :meth:`prepare_job_continuation`); ``None`` for a local job or a
        job that never reached a cluster.
        """

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="job_continuation",
                workflow_stage=job.workflow_stage,
                actor="user",
                description=f"Continued calculation {job.label}",
                parameters={
                    "job_label": job.label,
                    "calc_type": job.calc_type,
                    "execution_target": job.execution_target,
                    "preserved_remote_directory": remote_directory is not None,
                    "remote_directory": remote_directory,
                },
                output_entity_ids=[job.id],
                job_id=job.id,
            ),
        )

    def record_job_dismissed(
        self,
        document: ProjectDocument,
        *,
        job: Job,
    ) -> None:
        """Append one provenance milestone when a terminal job is dismissed.

        Dismissal hides the job from the active dashboard without deleting its
        record, runs, results, or provenance (see :meth:`dismiss_job`). The
        milestone lets the journal show that the row was intentionally removed
        from view.
        """

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="job_dismissed",
                workflow_stage=job.workflow_stage,
                actor="user",
                description=f"Dismissed calculation {job.label} from the dashboard",
                parameters={
                    "job_label": job.label,
                    "calc_type": job.calc_type,
                    "execution_target": job.execution_target,
                },
                output_entity_ids=[job.id],
                job_id=job.id,
            ),
        )

    def record_comparison_table_created(
        self,
        document: ProjectDocument,
        *,
        table: ComparisonTable,
    ) -> None:
        """Append one provenance milestone when a comparison table is created."""

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="create_comparison_table",
                actor="user",
                description=f"Created comparison table '{table.label}'",
                parameters={
                    "table_label": table.label,
                    "dft_structure_count": len(table.dft_structure_ids),
                },
                input_entity_ids=[table.experimental_structure_id]
                if table.experimental_structure_id is not None
                else [],
                output_entity_ids=[table.id, *table.dft_structure_ids],
            ),
        )

    # Workflow supercell assets and selection live in
    # ``mufinder.services.supercell_assets``; these methods delegate to it.

    def _supercell_asset(self, structure: Structure) -> WorkflowSupercellAsset | None:
        # ``mufinder.app.supercell_dialog`` reads asset metadata through this.
        return supercell_assets.supercell_asset(structure)

    def get_workflow_supercell_selection(
        self,
        document: ProjectDocument,
        workflow_stage: WorkflowStage,
    ) -> WorkflowSupercellSelection:
        """Return ``workflow_stage``'s persisted simulation-cell selection."""
        return supercell_assets.get_workflow_supercell_selection(document, workflow_stage)

    def set_workflow_supercell_selection(
        self,
        document: ProjectDocument,
        workflow_stage: WorkflowStage,
        selection: WorkflowSupercellSelection,
    ) -> None:
        """Persist ``workflow_stage``'s simulation-cell selection."""
        supercell_assets.set_workflow_supercell_selection(document, workflow_stage, selection)

    def list_workflow_supercell_assets(
        self,
        document: ProjectDocument,
        *,
        host_structure_id: UUID | None,
    ) -> list[WorkflowSupercellAsset]:
        """Return the saved supercell assets built from ``host_structure_id``."""
        return supercell_assets.list_workflow_supercell_assets(
            document, host_structure_id=host_structure_id
        )

    def repair_supercell_asset_if_needed(
        self, document: ProjectDocument, *, structure_id: UUID
    ) -> bool:
        """Rebuild an asset that lacks its magnetic host's spins; return whether it was rebuilt."""
        return supercell_assets.repair_supercell_asset_if_needed(
            document, structure_id=structure_id
        )

    def structure_atomic_spins(self, structure: Structure) -> list[float]:
        """Return numeric per-atom spins for a structure, defaulting missing values to zero."""
        return supercell_assets.structure_atomic_spins(structure)

    def structure_propagation_vector(self, structure: Structure) -> list[float]:
        """Return the structure propagation vector, defaulting to [0, 0, 0]."""
        return list(propagation_vector_from_structure(structure))

    def create_workflow_supercell_asset(
        self,
        document: ProjectDocument,
        *,
        workflow_stage: WorkflowStage,
        host_structure_id: UUID,
        matrix: list[list[int]],
        label: str | None = None,
        source_asset_structure_id: UUID | None = None,
        reuse_existing: bool = True,
    ) -> Structure:
        """Build and register a supercell asset, reusing a matching one by default."""
        return supercell_assets.create_workflow_supercell_asset(
            document,
            workflow_stage=workflow_stage,
            host_structure_id=host_structure_id,
            matrix=matrix,
            label=label,
            source_asset_structure_id=source_asset_structure_id,
            reuse_existing=reuse_existing,
        )

    def supercell_asset_references(
        self, document: ProjectDocument, *, asset_structure_id: UUID
    ) -> list[str]:
        """Return user-facing references to a saved supercell asset."""
        return supercell_assets.supercell_asset_references(
            document, asset_structure_id=asset_structure_id
        )

    def supercell_asset_is_frozen(
        self, document: ProjectDocument, *, asset_structure_id: UUID
    ) -> bool:
        """Return whether the asset is frozen into a calculation definition."""
        return supercell_assets.supercell_asset_is_frozen(
            document, asset_structure_id=asset_structure_id
        )

    def rename_workflow_supercell_asset(
        self,
        document: ProjectDocument,
        *,
        asset_structure_id: UUID,
        label: str,
    ) -> Structure:
        """Rename a saved supercell asset."""
        return supercell_assets.rename_workflow_supercell_asset(
            document, asset_structure_id=asset_structure_id, label=label
        )

    def update_workflow_supercell_asset_matrix(
        self,
        document: ProjectDocument,
        *,
        asset_structure_id: UUID,
        matrix: list[list[int]],
        label: str | None = None,
    ) -> Structure:
        """Regenerate an unfrozen saved supercell asset with a new matrix."""
        return supercell_assets.update_workflow_supercell_asset_matrix(
            document, asset_structure_id=asset_structure_id, matrix=matrix, label=label
        )

    def delete_workflow_supercell_asset(
        self, document: ProjectDocument, *, asset_structure_id: UUID
    ) -> Structure:
        """Delete an unused saved supercell asset and clear stale active selections."""
        return supercell_assets.delete_workflow_supercell_asset(
            document, asset_structure_id=asset_structure_id
        )

    def use_workflow_supercell_asset(
        self,
        document: ProjectDocument,
        *,
        workflow_stage: WorkflowStage,
        asset_structure_id: UUID,
    ) -> Structure:
        """Select a saved supercell asset for ``workflow_stage``."""
        return supercell_assets.use_workflow_supercell_asset(
            document, workflow_stage=workflow_stage, asset_structure_id=asset_structure_id
        )

    def clear_workflow_supercell_selection(
        self, document: ProjectDocument, workflow_stage: WorkflowStage
    ) -> None:
        """Reset ``workflow_stage`` to simulate the single unit cell."""
        supercell_assets.clear_workflow_supercell_selection(document, workflow_stage)

    def resolve_workflow_structure(
        self,
        document: ProjectDocument,
        workflow_stage: WorkflowStage,
        *,
        normalize_state: bool = False,
    ) -> WorkflowStructureResolution:
        """Return the structure ``workflow_stage`` simulates and its selection state."""
        return supercell_assets.resolve_workflow_structure(
            document, workflow_stage, normalize_state=normalize_state
        )

    def import_structure_file(
        self,
        document: ProjectDocument,
        source_path: Path,
        *,
        label: str | None = None,
        make_active: bool = True,
    ) -> Structure:
        """Load a structure from disk and register it in the project.

        The input file is copied to the project's imports directory to preserve
        provenance and portability of loaded projects.
        """
        loaded = load_structure_file(source_path)
        copied_path = self._copy_import_file(document.project_root, source_path)
        project_copy = str(copied_path.relative_to(document.project_root))

        structure = Structure(
            label=label or source_path.stem,
            origin="imported",
            lattice_vectors=loaded.lattice_vectors,
            species=loaded.species,
            fractional_positions=loaded.fractional_positions,
            source_file=project_copy,
            properties=loaded.properties,
        )
        document.structures[structure.id] = structure
        if make_active:
            document.project.host_structure_id = structure.id
            document.project.magnetic_structure = loaded.properties.get("magnetic_structure")

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="import_structure",
                workflow_stage=WorkflowStage.MUON_GENERATION,
                actor="user",
                description=f"Imported structure from {source_path.name}",
                parameters={
                    "source_file": str(source_path),
                    "project_copy": project_copy,
                    "file_extension": source_path.suffix.lower(),
                    "make_active": make_active,
                    "has_magnetic_structure": bool(loaded.properties.get("magnetic_structure")),
                },
                output_entity_ids=[structure.id],
            ),
        )
        return structure

    def add_job(
        self,
        document: ProjectDocument,
        *,
        label: str,
        workflow_stage: WorkflowStage,
        calc_type: str,
        execution_target: str = "local",
    ) -> Job:
        """Create an in-memory job record for the dashboard."""
        job = Job(
            label=label,
            workflow_stage=workflow_stage,
            calc_type=calc_type,
            status=JobStatus.DEFINED,
            execution_target=execution_target,
            created_at=next_job_created_at(document.jobs.values()),
            tags=[workflow_stage.value, calc_type],
        )
        document.jobs[job.id] = job
        return job

    def register_convergence_cutoff_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_id: UUID | None = None,
        host_structure_id: UUID | None = None,
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_cutoff: dict[str, str] | None = None,
        simulation_cell_mode: str = WORKFLOW_SUPERCELL_UNIT_CELL,
        supercell_source_workflow: str | None = None,
    ) -> CalculationDefinition:
        """Persist one convergence-cutoff configuration as a reusable definition.

        This records the configured parameter snapshot as a first-class entity,
        separate from ephemeral UI form state.
        """

        structure_id = _known_input_structure(document, input_structure_id, host_structure_id)
        snapshot = to_jsonable(parameters)
        return self._register_definition(
            document,
            stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
            action="configure_convergence_cutoff",
            description=(
                "Configured cutoff convergence with "
                f"{len(snapshot.get('sweep_values', []))} cutoff value(s)"
            ),
            configuration=snapshot,
            output_directory=output_directory,
            extra_parameters={
                "job_ids_by_cutoff": dict(job_ids_by_cutoff or {}),
                "simulation_cell_mode": simulation_cell_mode,
                "supercell_source_workflow": supercell_source_workflow,
            },
            input_structure_ids=[structure_id],
        )

    def register_convergence_kpoint_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_id: UUID | None = None,
        host_structure_id: UUID | None = None,
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_grid: dict[str, str] | None = None,
        simulation_cell_mode: str = WORKFLOW_SUPERCELL_UNIT_CELL,
        supercell_source_workflow: str | None = None,
    ) -> CalculationDefinition:
        """Persist one convergence-kpoint configuration as a reusable definition."""

        structure_id = _known_input_structure(document, input_structure_id, host_structure_id)
        snapshot = to_jsonable(parameters)
        return self._register_definition(
            document,
            stage=WorkflowStage.CONVERGENCE_KPOINT,
            calc_type="castep_convergence_kpoint",
            action="configure_convergence_kpoint",
            description=(
                "Configured k-point convergence with "
                f"{len(snapshot.get('sweep_values', []))} grid value(s)"
            ),
            configuration=snapshot,
            output_directory=output_directory,
            extra_parameters={
                "job_ids_by_grid": dict(job_ids_by_grid or {}),
                "simulation_cell_mode": simulation_cell_mode,
                "supercell_source_workflow": supercell_source_workflow,
            },
            input_structure_ids=[structure_id],
        )

    def register_pristine_relaxation_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_id: UUID | None = None,
        host_structure_id: UUID | None = None,
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_label: dict[str, str] | None = None,
    ) -> CalculationDefinition:
        """Persist one pristine-relaxation configuration as a reusable definition."""

        structure_id = _known_input_structure(document, input_structure_id, host_structure_id)
        snapshot = to_jsonable(parameters)
        return self._register_definition(
            document,
            stage=WorkflowStage.PRISTINE_RELAXATION,
            calc_type="castep_pristine_relaxation",
            action="configure_pristine_relaxation",
            description=f"Configured pristine relaxation with {len(snapshot.get('rows', []))} job row(s)",
            configuration=snapshot,
            output_directory=output_directory,
            extra_parameters={"job_ids_by_label": dict(job_ids_by_label or {})},
            input_structure_ids=[structure_id],
        )

    def register_muon_relaxation_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_id: UUID | None = None,
        host_structure_id: UUID | None = None,
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_label: dict[str, str] | None = None,
    ) -> CalculationDefinition:
        """Persist one muon-relaxation configuration.

        The definition's input sites are the rows' parseable ``site_id`` values.
        """

        structure_id = _known_input_structure(document, input_structure_id, host_structure_id)
        snapshot = to_jsonable(parameters)
        input_site_ids: list[UUID] = []
        for row in snapshot.get("rows", []):
            if not isinstance(row, dict) or not row.get("site_id"):
                continue
            try:
                input_site_ids.append(UUID(str(row["site_id"])))
            except ValueError:
                continue
        return self._register_definition(
            document,
            stage=WorkflowStage.MUON_RELAXATION,
            calc_type=MUON_RELAXATION_DEFINITION_CALC_TYPE,
            action="configure_muon_relaxation",
            description=f"Configured muon relaxation with {len(snapshot.get('rows', []))} job row(s)",
            configuration=snapshot,
            output_directory=output_directory,
            extra_parameters={"job_ids_by_label": dict(job_ids_by_label or {})},
            input_structure_ids=[structure_id],
            input_site_ids=input_site_ids,
        )

    def register_uep_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_id: UUID | None = None,
        host_structure_id: UUID | None = None,
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_label: dict[str, str] | None = None,
    ) -> CalculationDefinition:
        """Persist one UEP source-job configuration as a reusable definition."""

        structure_id = _known_input_structure(document, input_structure_id, host_structure_id)
        snapshot = to_jsonable(parameters)
        return self._register_definition(
            document,
            stage=WorkflowStage.UEP,
            calc_type="castep_uep_source",
            action="configure_uep",
            description=f"Configured UEP source workflow with {len(snapshot.get('rows', []))} job row(s)",
            configuration=snapshot,
            output_directory=output_directory,
            extra_parameters={"job_ids_by_label": dict(job_ids_by_label or {})},
            input_structure_ids=[structure_id],
        )

    def register_quantum_phonon_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_ids: list[UUID],
        input_site_ids: list[UUID],
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_label: dict[str, str] | None = None,
    ) -> CalculationDefinition:
        """Persist one Quantum Corrections phonon configuration.

        The configuration's phonon-job and seed-name maps are also lifted to
        top-level definition parameters; the provenance record omits them.
        """

        _require_known_inputs(document, input_structure_ids, input_site_ids)
        snapshot = to_jsonable(parameters)
        lifted_maps: dict[str, Any] = {}
        for key in ("phonon_job_ids_by_label", "job_seed_names_by_label"):
            value = snapshot.get(key, {})
            lifted_maps[key] = dict(value) if isinstance(value, dict) else {}
        return self._register_definition(
            document,
            stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type="castep_quantum_phonon_set",
            action="configure_quantum_corrections",
            description=f"Configured Quantum Corrections with {len(input_site_ids)} relaxed site(s)",
            configuration=snapshot,
            output_directory=output_directory,
            extra_parameters={"job_ids_by_label": dict(job_ids_by_label or {})},
            definition_only_parameters=lifted_maps,
            input_structure_ids=input_structure_ids,
            input_site_ids=input_site_ids,
        )

    def register_hyperfine_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_ids: list[UUID],
        input_site_ids: list[UUID],
        parameters: dict[str, Any],
        output_directory: Path,
    ) -> CalculationDefinition:
        """Persist one Hyperfine Coupling configuration (one magres job per relaxed site)."""

        _require_known_inputs(document, input_structure_ids, input_site_ids)
        return self._register_definition(
            document,
            stage=WorkflowStage.HYPERFINE_COUPLING,
            calc_type="castep_hyperfine_set",
            action="configure_hyperfine_coupling",
            description=f"Configured Hyperfine Coupling with {len(input_site_ids)} relaxed site(s)",
            configuration=to_jsonable(parameters),
            output_directory=output_directory,
            extra_parameters={"job_ids_by_label": {}},
            input_structure_ids=input_structure_ids,
            input_site_ids=input_site_ids,
        )

    def register_nudged_elastic_band_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_ids: list[UUID],
        input_site_ids: list[UUID],
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_label: dict[str, str] | None = None,
    ) -> CalculationDefinition:
        """Persist one Energy Barriers NEB configuration."""

        _require_known_inputs(document, input_structure_ids, input_site_ids)
        return self._register_definition(
            document,
            stage=WorkflowStage.ENERGY_BARRIERS,
            calc_type="castep_nudged_elastic_band_set",
            action="configure_energy_barriers",
            description=f"Configured Energy Barriers NEB with {len(input_site_ids)} relaxed site(s)",
            configuration=to_jsonable(parameters),
            output_directory=output_directory,
            extra_parameters={"job_ids_by_label": dict(job_ids_by_label or {})},
            input_structure_ids=input_structure_ids,
            input_site_ids=input_site_ids,
        )

    def _register_definition(
        self,
        document: ProjectDocument,
        *,
        stage: WorkflowStage,
        calc_type: str,
        action: str,
        description: str,
        configuration: Any,
        output_directory: Path,
        extra_parameters: dict[str, Any],
        input_structure_ids: Sequence[UUID],
        input_site_ids: Sequence[UUID] = (),
        definition_only_parameters: dict[str, Any] | None = None,
    ) -> CalculationDefinition:
        """Store one calculation definition and append its ``configure_*`` record.

        The definition's parameters are ``configuration`` (the JSON snapshot of
        the configured form), the project-relative ``output_directory``, then
        ``extra_parameters`` and ``definition_only_parameters`` in that order.
        The provenance record repeats every parameter except the
        definition-only ones, behind the new definition's id; its mapping
        values are copies, so later edits to the definition's job maps leave
        the recorded event unchanged.
        """

        relative_output = _project_relative_path(output_directory, document.project_root)
        definition = CalculationDefinition(
            workflow_stage=stage,
            calc_type=calc_type,
            parameters={
                "configuration": configuration,
                "output_directory": relative_output,
                **extra_parameters,
                **(definition_only_parameters or {}),
            },
            input_structure_ids=list(input_structure_ids),
            input_site_ids=list(input_site_ids),
            created_at=next_definition_created_at(document.calculations.values()),
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action=action,
                workflow_stage=stage,
                actor="user",
                description=description,
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": configuration,
                    "output_directory": relative_output,
                    **{
                        key: dict(value) if isinstance(value, dict) else value
                        for key, value in extra_parameters.items()
                    },
                },
                input_entity_ids=[*definition.input_structure_ids, *definition.input_site_ids],
                output_entity_ids=[definition.id],
            ),
        )
        return definition

    def set_recommended_cutoff(
        self,
        document: ProjectDocument,
        cutoff_energy: float,
        *,
        method: str | None = None,
        energy_threshold: float | None = None,
        force_threshold: float | None = None,
        stress_threshold: float | None = None,
        cell_id: str | None = None,
        xc_functional: str | None = None,
    ) -> None:
        """Persist a project-wide recommended cutoff energy for future workflows.

        ``xc_functional`` is the functional the converged runs used. It is
        recorded against the cutoff and, when given, also becomes the project
        default functional — a converged cutoff is only meaningful alongside it.
        """
        document.project.recommended_cutoff_energy = float(cutoff_energy)
        document.project.recommended_cutoff_method = (
            str(method).strip() if method is not None and str(method).strip() else None
        )
        document.project.recommended_cutoff_energy_threshold = (
            float(energy_threshold) if energy_threshold is not None else None
        )
        document.project.recommended_cutoff_force_threshold = (
            float(force_threshold) if force_threshold is not None else None
        )
        document.project.recommended_cutoff_stress_threshold = (
            float(stress_threshold) if stress_threshold is not None else None
        )
        document.project.recommended_cutoff_cell_id = (
            str(cell_id).strip() if cell_id is not None and str(cell_id).strip() else None
        )
        document.project.recommended_cutoff_xc_functional = normalized_xc_functional(xc_functional)
        if document.project.recommended_cutoff_xc_functional is not None:
            document.project.recommended_xc_functional = (
                document.project.recommended_cutoff_xc_functional
            )

        method_fragment = (
            f" using method '{document.project.recommended_cutoff_method}'"
            if document.project.recommended_cutoff_method
            else ""
        )
        self.append_provenance(
            document,
            ProvenanceRecord(
                action="set_recommended_cutoff",
                workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
                actor="user",
                description=(
                    f"Set project recommended cutoff to {float(cutoff_energy):.8g} eV"
                    f"{method_fragment}"
                ),
                parameters={
                    "recommended_cutoff_energy": float(cutoff_energy),
                    "recommended_cutoff_method": document.project.recommended_cutoff_method,
                    "recommended_cutoff_energy_threshold": document.project.recommended_cutoff_energy_threshold,
                    "recommended_cutoff_force_threshold": document.project.recommended_cutoff_force_threshold,
                    "recommended_cutoff_stress_threshold": document.project.recommended_cutoff_stress_threshold,
                    "recommended_cutoff_cell_id": document.project.recommended_cutoff_cell_id,
                    "recommended_cutoff_xc_functional": (
                        document.project.recommended_cutoff_xc_functional
                    ),
                },
            ),
        )

    def clear_recommended_cutoff(self, document: ProjectDocument) -> None:
        """Remove the project default cutoff energy and how it was established."""

        project = document.project
        project.recommended_cutoff_energy = None
        project.recommended_cutoff_method = None
        project.recommended_cutoff_energy_threshold = None
        project.recommended_cutoff_force_threshold = None
        project.recommended_cutoff_stress_threshold = None
        project.recommended_cutoff_cell_id = None
        project.recommended_cutoff_xc_functional = None
        self.append_provenance(
            document,
            ProvenanceRecord(
                action="clear_recommended_cutoff",
                workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
                actor="user",
                description="Cleared project recommended cutoff",
            ),
        )

    def clear_recommended_kpoint_grid(self, document: ProjectDocument) -> None:
        """Remove the project default k-point grid and how it was established."""

        project = document.project
        project.recommended_kpoint_grid = None
        project.recommended_kpoint_lattice = None
        project.recommended_kpoint_method = None
        project.recommended_kpoint_energy_threshold = None
        project.recommended_kpoint_force_threshold = None
        project.recommended_kpoint_stress_threshold = None
        project.recommended_kpoint_cell_id = None
        project.recommended_kpoint_xc_functional = None
        self.append_provenance(
            document,
            ProvenanceRecord(
                action="clear_recommended_kpoint_grid",
                workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
                actor="user",
                description="Cleared project recommended k-point grid",
            ),
        )

    def set_recommended_kpoint_grid(
        self,
        document: ProjectDocument,
        kpoint_grid: tuple[int, int, int],
        *,
        lattice_vectors: list[list[float]] | None = None,
        method: str | None = None,
        energy_threshold: float | None = None,
        force_threshold: float | None = None,
        stress_threshold: float | None = None,
        cell_id: str | None = None,
        xc_functional: str | None = None,
    ) -> None:
        """Persist a project-wide recommended k-point grid for future workflows.

        ``xc_functional`` is the functional the converged runs used; see
        :meth:`set_recommended_cutoff`.
        """

        grid = (int(kpoint_grid[0]), int(kpoint_grid[1]), int(kpoint_grid[2]))
        document.project.recommended_kpoint_grid = grid
        document.project.recommended_kpoint_lattice = (
            [[float(row[0]), float(row[1]), float(row[2])] for row in lattice_vectors]
            if isinstance(lattice_vectors, list) and len(lattice_vectors) == 3
            else None
        )
        document.project.recommended_kpoint_method = (
            str(method).strip() if method is not None and str(method).strip() else None
        )
        document.project.recommended_kpoint_energy_threshold = (
            float(energy_threshold) if energy_threshold is not None else None
        )
        document.project.recommended_kpoint_force_threshold = (
            float(force_threshold) if force_threshold is not None else None
        )
        document.project.recommended_kpoint_stress_threshold = (
            float(stress_threshold) if stress_threshold is not None else None
        )
        document.project.recommended_kpoint_cell_id = (
            str(cell_id).strip() if cell_id is not None and str(cell_id).strip() else None
        )
        document.project.recommended_kpoint_xc_functional = normalized_xc_functional(xc_functional)
        if document.project.recommended_kpoint_xc_functional is not None:
            document.project.recommended_xc_functional = (
                document.project.recommended_kpoint_xc_functional
            )

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="set_recommended_kpoint_grid",
                workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
                actor="user",
                description=(
                    f"Set project recommended k-point grid to {grid[0]}x{grid[1]}x{grid[2]}"
                ),
                parameters={
                    "recommended_kpoint_grid": [grid[0], grid[1], grid[2]],
                    "recommended_kpoint_lattice": document.project.recommended_kpoint_lattice,
                    "recommended_kpoint_method": document.project.recommended_kpoint_method,
                    "recommended_kpoint_energy_threshold": document.project.recommended_kpoint_energy_threshold,
                    "recommended_kpoint_force_threshold": document.project.recommended_kpoint_force_threshold,
                    "recommended_kpoint_stress_threshold": document.project.recommended_kpoint_stress_threshold,
                    "recommended_kpoint_cell_id": document.project.recommended_kpoint_cell_id,
                    "recommended_kpoint_xc_functional": (
                        document.project.recommended_kpoint_xc_functional
                    ),
                },
            ),
        )

    def set_recommended_xc_functional(
        self,
        document: ProjectDocument,
        xc_functional: str | None,
    ) -> None:
        """Persist (or clear, with ``None``) the project default XC functional.

        The functionals the cutoff and k-point defaults were converged with are
        left untouched, so a mismatch stays visible rather than being silently
        discarded.
        """

        functional = normalized_xc_functional(xc_functional)
        document.project.recommended_xc_functional = functional
        self.append_provenance(
            document,
            ProvenanceRecord(
                action="set_recommended_xc_functional",
                actor="user",
                description=(
                    f"Set project default XC functional to {functional}"
                    if functional is not None
                    else "Cleared project default XC functional"
                ),
                parameters={"recommended_xc_functional": functional},
            ),
        )

    def submit_job(self, job: Job) -> None:
        """Move a job from DEFINED to SUBMITTING and stamp submission time."""
        self._assert_transition(job.status, JobStatus.SUBMITTING)
        job.status = JobStatus.SUBMITTING
        job.submitted_at = utc_now()
        job.started_at = None
        job.completed_at = None
        job.error_message = None
        job.failure_reason = None

    def mark_job_queued(self, job: Job, remote_job_id: str | None = None) -> None:
        """Mark an in-flight submission as queued."""
        self._assert_transition(job.status, JobStatus.QUEUED)
        job.status = JobStatus.QUEUED
        if remote_job_id:
            job.remote_job_id = remote_job_id

    def start_job(self, job: Job) -> None:
        """Mark a queued job as running."""
        self._assert_transition(job.status, JobStatus.RUNNING)
        job.status = JobStatus.RUNNING
        job.started_at = utc_now()

    def complete_job(self, job: Job) -> None:
        """Mark a running or retrieving job as completed."""
        self._assert_transition(job.status, JobStatus.COMPLETED)
        job.status = JobStatus.COMPLETED
        job.completed_at = utc_now()
        job.error_message = None
        job.failure_reason = None

    def stop_job(
        self,
        job: Job,
        message: str | None = None,
        *,
        reason: FailureReason | None = None,
    ) -> None:
        """Mark a running or retrieving job as stopped (for example walltime reached)."""
        self._assert_transition(job.status, JobStatus.STOPPED)
        job.status = JobStatus.STOPPED
        job.completed_at = utc_now()
        job.error_message = message
        job.failure_reason = reason

    def fail_job(
        self,
        job: Job,
        message: str,
        *,
        reason: FailureReason | None = None,
    ) -> None:
        """Mark a job as failed with an error message and its classified cause."""
        self._assert_transition(job.status, JobStatus.FAILED)
        job.status = JobStatus.FAILED
        job.completed_at = utc_now()
        job.error_message = message
        job.failure_reason = reason

    def cancel_job(self, job: Job) -> None:
        """Cancel a non-terminal job."""
        self._assert_transition(job.status, JobStatus.CANCELLED)
        job.status = JobStatus.CANCELLED
        job.completed_at = utc_now()
        job.failure_reason = None

    def reset_failed_job(self, job: Job) -> None:
        """Reset a failed, stopped, or cancelled job to defined for resubmission."""

        if job.status not in TERMINAL_RECOVERABLE_JOB_STATUSES:
            raise ValueError("Only failed, stopped, or cancelled jobs can be reset.")

        job.status = JobStatus.DEFINED
        job.submitted_at = None
        job.started_at = None
        job.completed_at = None
        job.remote_job_id = None
        job.remote_directory = None
        job.num_processes = None
        job.error_message = None
        job.failure_reason = None
        job.last_checked_at = None
        job.last_status_source = None
        job.last_status_message = None
        job.dismissed_at = None

    def prepare_job_continuation(self, job: Job) -> None:
        """Return a finished job to defined for a continuation run, keeping its directory.

        Accepts :data:`CONTINUABLE_JOB_STATUSES` (completed, failed, stopped, or
        cancelled). Like :meth:`reset_failed_job` this direct-assigns
        ``DEFINED`` instead of going through :meth:`_assert_transition`: DEFINED
        is not a target in the validated transition table, and COMPLETED has no
        validated exit at all.

        Every field the previous run stamped is cleared, but
        ``remote_directory`` is **deliberately preserved** — that is the whole
        point of the primitive. A DEFINED job carrying a non-null
        ``remote_directory`` is the durable marker the cluster submission path
        uses to detect a continuation: it resubmits into the existing remote
        directory (where CASTEP's ``.check`` file lives) instead of recomputing
        a target, checking for an existing directory, and offering to delete it.

        No other code path can produce that status/field combination. The
        writers of ``remote_directory`` are: ``JobExecutionController``, which
        sets it while moving a job to SUBMITTING and clears it on local
        submission; :meth:`reset_failed_job`, which clears it; and the main
        window's cluster-tracking sync, which only touches active jobs.
        """

        if job.status not in CONTINUABLE_JOB_STATUSES:
            raise ValueError("Only completed, failed, stopped, or cancelled jobs can be continued.")

        job.status = JobStatus.DEFINED
        job.submitted_at = None
        job.started_at = None
        job.completed_at = None
        job.remote_job_id = None
        job.num_processes = None
        job.error_message = None
        job.failure_reason = None
        job.last_checked_at = None
        job.last_status_source = None
        job.last_status_message = None
        job.dismissed_at = None

    def mark_job_retrieving(self, job: Job) -> None:
        """Set job to retrieval stage before completion for remote runs."""
        self._assert_transition(job.status, JobStatus.RETRIEVING)
        job.status = JobStatus.RETRIEVING

    def retry_job_retrieval(self, job: Job) -> None:
        """Re-enter the RETRIEVING stage to re-attempt output retrieval.

        This primitive exists for the dashboard "Retry retrieval" action: a
        cluster job whose scheduler run finished but whose output retrieval or
        parsing failed keeps its remote metadata (``remote_job_id`` /
        ``remote_directory``), so retrieval can be re-attempted without
        resubmitting a duplicate scheduler job. The FAILED/STOPPED -> RETRIEVING
        edges are validated by :meth:`_assert_transition`; the caller
        (``JobExecutionController.prepare_retry_retrieval``) enforces the
        failed/stopped-only and remote-metadata preconditions before calling
        this. The remote metadata is deliberately preserved so the runner can
        re-download from the same remote directory.
        """

        self._assert_transition(job.status, JobStatus.RETRIEVING)
        job.status = JobStatus.RETRIEVING
        job.error_message = None
        job.failure_reason = None

    def resume_reappeared_job(self, job: Job, status: JobStatus) -> None:
        """Return a failed job to ``status`` because the scheduler lists it again.

        This primitive backs the recovery of a cluster job failed for a missing
        scheduler entry (the queue and accounting both stopped listing it)
        that the scheduler later reports as queued, running, or completed:
        ``status`` is ``QUEUED``, ``RUNNING``, or ``RETRIEVING`` for a completed
        job whose output is still to be fetched. The FAILED -> QUEUED / RUNNING
        / RETRIEVING edges are validated by :meth:`_assert_transition`; the
        caller (``JobExecutionController.resume_reappeared_cluster_job``)
        enforces that the job failed for a missing scheduler entry. The failure
        stamps are cleared and the remote metadata is kept, so polling resumes
        against the same scheduler job.
        """

        if status not in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.RETRIEVING}:
            raise ValueError(f"A reappeared job cannot resume as {status.value}.")
        self._assert_transition(job.status, status)
        job.status = status
        job.completed_at = None
        job.error_message = None
        job.failure_reason = None
        if status == JobStatus.RUNNING:
            job.started_at = utc_now()

    def dismiss_job(self, job: Job) -> None:
        """Hide a terminal job from the active dashboard without deleting it.

        Dismissal is **not** a status transition: the job keeps its FAILED /
        STOPPED / CANCELLED status and its record, runs, results, and provenance
        all remain in the document. Only ``dismissed_at`` is stamped so
        ``_refresh_jobs_table`` can filter the row out of the active dashboard.
        Already-dismissed jobs are rejected so the milestone is recorded once.

        Dismissal is nonetheless *acknowledgement* of the failure: the app layer
        derives a submittable workflow-row status from the stamp (see
        ``mufinder.app.workflow_submission.effective_row_status``) so the row the
        job belongs to can be resubmitted once its dashboard controls are gone.
        That projection is derived at read time and changes nothing here.
        """

        if job.status not in TERMINAL_RECOVERABLE_JOB_STATUSES:
            raise ValueError("Only failed, stopped, or cancelled jobs can be dismissed.")
        if job.dismissed_at is not None:
            raise ValueError("Only failed, stopped, or cancelled jobs can be dismissed.")
        job.dismissed_at = utc_now()

    def reconcile_orphaned_local_jobs(self, document: ProjectDocument) -> list[Job]:
        """Stop local jobs left mid-flight when MuFinder closed.

        A local job left in an active state (submitting/queued/running/
        retrieving) when the app exits has no live process behind it on the next
        launch, yet the dashboard would keep showing it as running. Cluster jobs
        are re-attached separately (see
        ``MainWindow._sync_cluster_tracking_from_document``) and are left
        untouched here.

        Selection matches the local semantics pinned by the cluster
        reattachment characterization tests: an ``execution_target`` that is
        blank/empty or equals ``"local"`` (case-insensitively). Each selected
        job is moved to STOPPED with ``ORPHANED_LOCAL_JOB_STOP_MESSAGE`` and its
        latest non-terminal run is closed out to keep run/job status consistent.
        A ``job_stopped`` provenance milestone is recorded per job.

        :param document: Loaded project document to reconcile in place.
        :returns: The jobs that were stopped, for the caller's status message.
        """

        reconciled: list[Job] = []
        for job in document.jobs.values():
            if job.status not in ACTIVE_JOB_STATUSES:
                continue
            target = (job.execution_target or "").strip()
            if target and target.casefold() != "local":
                continue

            self.stop_job(job, ORPHANED_LOCAL_JOB_STOP_MESSAGE, reason=FailureReason.INTERRUPTED)
            # Monitor metadata is bookkeeping, not a status transition: record
            # that reconciliation was the last observer of this job.
            job.last_checked_at = utc_now()
            job.last_status_source = "reconciliation"
            job.last_status_message = ORPHANED_LOCAL_JOB_STOP_MESSAGE
            self._close_orphaned_run_for_job(document, job)
            self.record_job_terminal_event(
                document,
                action="job_stopped",
                job=job,
                error_message=job.error_message,
            )
            reconciled.append(job)

        return reconciled

    @staticmethod
    def _close_orphaned_run_for_job(document: ProjectDocument, job: Job) -> None:
        """Mark the job's latest in-flight run as stopped, mirroring walltime stops."""

        active_runs = [
            run
            for run in document.runs.values()
            if run.job_id == job.id and run.status in ACTIVE_JOB_STATUSES
        ]
        if not active_runs:
            return
        run = max(active_runs, key=lambda run: run.run_index)
        run.status = JobStatus.STOPPED
        run.completed_at = utc_now()

    def generate_candidate_sites(
        self,
        document: ProjectDocument,
        *,
        method: str,
        parameters: dict[str, Any],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[MuonSite]:
        """Create generated muon sites from a generation request.

        See :func:`mufinder.services.site_generation_service.generate_candidate_sites`.
        """
        outcome = site_generation_service.generate_candidate_sites(
            document, method=method, parameters=parameters, progress_callback=progress_callback
        )
        return self._record_site_change(document, outcome)

    def _record_site_change(
        self,
        document: ProjectDocument,
        outcome: site_generation_service.SiteGenerationOutcome,
    ) -> list[MuonSite]:
        """Append a generation step's provenance and return the sites it touched."""
        for record in outcome.provenance:
            self.append_provenance(document, record)
        return outcome.sites

    def active_host_structure(self, document: ProjectDocument) -> Structure | None:
        """Return the current host structure, if one is selected."""
        return active_host_structure(document)

    def list_promotable_host_structures(self, document: ProjectDocument) -> list[PromotableHost]:
        """Return structures eligible to be the active host, ordered for display.

        Includes imported structures and successfully completed pristine relaxation
        results. Excludes supercell assets and all other derived structure origins.
        """
        # Build set of structure IDs produced by a completed pristine relaxation run.
        completed_ids: set[UUID] = set()
        result_created_at: dict[UUID, datetime | None] = {}
        for result in document.results.values():
            if not result.calculation_run_id:
                continue
            run = document.runs.get(result.calculation_run_id)
            if (
                run is None
                or run.status != JobStatus.COMPLETED
                or run.convergence_achieved is False
            ):
                continue
            for struct_id in result.derived_structure_ids:
                completed_ids.add(struct_id)
                result_created_at[struct_id] = result.created_at

        promotable: list[PromotableHost] = []
        for structure in document.structures.values():
            if supercell_assets.supercell_asset(structure) is not None:
                continue
            if structure.origin == "imported":
                hint = (
                    f"Imported from {structure.source_file}"
                    if structure.source_file
                    else "Imported structure"
                )
                promotable.append(
                    PromotableHost(
                        structure_id=structure.id,
                        label=f"Original — {structure.label}",
                        provenance_hint=hint,
                        kind="imported",
                        created_at=None,
                    )
                )
            elif structure.origin == "pristine_relaxation" and structure.id in completed_ids:
                created_at = result_created_at.get(structure.id)
                date_str = created_at.strftime("%Y-%m-%d") if created_at else ""
                job_label = structure.label
                source_job_id_str = structure.properties.get("source_job_id")
                if source_job_id_str:
                    try:
                        job = document.jobs.get(UUID(str(source_job_id_str)))
                        if job:
                            job_label = job.label
                    except (ValueError, AttributeError):
                        pass
                label = f"Relaxed — {job_label}"
                if date_str:
                    label += f" ({date_str})"
                hint = f"Result of pristine relaxation '{job_label}'"
                if date_str:
                    hint += f" completed {date_str}"
                promotable.append(
                    PromotableHost(
                        structure_id=structure.id,
                        label=label,
                        provenance_hint=hint,
                        kind="relaxed",
                        created_at=created_at,
                    )
                )

        def _sort_key(h: PromotableHost) -> tuple:
            if h.created_at is None:
                return (1, 0.0)
            return (0, -h.created_at.timestamp())

        imported = sorted(
            [h for h in promotable if h.kind == "imported"],
            key=lambda h: h.label.casefold(),
        )
        relaxed = sorted(
            [h for h in promotable if h.kind == "relaxed"],
            key=_sort_key,
        )
        return imported + relaxed

    def set_active_host_structure(
        self,
        document: ProjectDocument,
        structure_id: UUID,
        *,
        actor: str = "user",
    ) -> Structure:
        """Switch the project's active host to a promotable structure.

        Updates host_structure_id and rebinds magnetic_structure to the new
        host's properties (pristine-relaxation outputs typically carry no spins,
        so the field becomes None and the user re-enters spins as needed).
        """
        promotable_ids = {h.structure_id for h in self.list_promotable_host_structures(document)}
        if structure_id not in promotable_ids:
            raise ValueError("Structure is not eligible as an active host.")

        structure = document.structures[structure_id]
        previous_id = document.project.host_structure_id
        if previous_id == structure_id:
            return structure

        previous = document.structures.get(previous_id) if previous_id else None
        prev_label = previous.label if previous else "(none)"

        document.project.host_structure_id = structure_id
        document.project.magnetic_structure = structure.properties.get("magnetic_structure")

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="set_active_host_structure",
                actor=actor,
                description=f"Changed active host from '{prev_label}' to '{structure.label}'",
                parameters={
                    "previous_host_structure_id": str(previous_id) if previous_id else None,
                    "new_host_structure_id": str(structure_id),
                },
                input_entity_ids=[previous_id] if previous_id else [],
                output_entity_ids=[structure_id],
            ),
        )
        return structure

    def resolve_underlying_host(self, document: ProjectDocument, structure_id: UUID) -> UUID | None:
        """Walk the parent_structure_id chain to find the host root.

        Returns the id of the first ancestor (inclusive) whose origin is
        'imported' or 'pristine_relaxation', or None if the chain is broken
        or contains a cycle.
        """
        _HOST_ORIGINS = {"imported", "pristine_relaxation"}
        visited: set[UUID] = set()
        current_id: UUID | None = structure_id
        while current_id is not None:
            if current_id in visited:
                return None
            visited.add(current_id)
            structure = document.structures.get(current_id)
            if structure is None:
                return None
            if structure.origin in _HOST_ORIGINS:
                return structure.id
            current_id = structure.parent_structure_id
        return None

    def validate_sites_match_active_host(
        self,
        document: ProjectDocument,
        site_ids: Sequence[UUID],
    ) -> SiteHostMismatch | None:
        """Return mismatch info if any site was generated for a different host.

        Sites carry source_structure_id stamped at generation time; this check
        ensures none of them target a structure other than the active host.
        Returns None when all sites are consistent with the active host.
        """
        host = active_host_structure(document)
        if host is None:
            return None
        offenders: list[tuple[UUID, str, UUID | None, str]] = []
        for site_id in site_ids:
            site = document.sites.get(site_id)
            if site is None:
                continue
            if site.source_structure_id != host.id:
                source = (
                    document.structures.get(site.source_structure_id)
                    if site.source_structure_id
                    else None
                )
                source_label = source.label if source else "(unknown)"
                offenders.append((site.id, site.label, site.source_structure_id, source_label))
        if not offenders:
            return None
        return SiteHostMismatch(
            active_host_id=host.id,
            active_host_label=host.label,
            offenders=offenders,
        )

    def validate_structures_match_active_host(
        self,
        document: ProjectDocument,
        structure_ids: Sequence[UUID],
    ) -> StructureHostMismatch | None:
        """Return mismatch info if any structure's provenance chain leads to a different host.

        Used for downstream workflows (quantum corrections, energy barriers, NEB) that
        consume muoniated relaxed structures rather than pristine sites. Walks the
        parent_structure_id chain to find the underlying host for each structure.
        Returns None when all structures resolve to the active host.
        """
        host = active_host_structure(document)
        if host is None:
            return None
        offenders: list[tuple[UUID, str, UUID | None, str]] = []
        for struct_id in structure_ids:
            structure = document.structures.get(struct_id)
            if structure is None:
                continue
            resolved_host_id = self.resolve_underlying_host(document, struct_id)
            if resolved_host_id != host.id:
                resolved_host = (
                    document.structures.get(resolved_host_id) if resolved_host_id else None
                )
                resolved_label = resolved_host.label if resolved_host else "(unknown)"
                offenders.append((struct_id, structure.label, resolved_host_id, resolved_label))
        if not offenders:
            return None
        return StructureHostMismatch(
            active_host_id=host.id,
            active_host_label=host.label,
            offenders=offenders,
        )

    def _copy_import_file(self, project_root: Path, source_path: Path) -> Path:
        """Copy imported source file into the project imports directory."""
        store = ProjectStore(project_root)
        store.initialize_layout()

        imports_dir = project_root / "imports"
        imports_dir.mkdir(parents=True, exist_ok=True)

        candidate = imports_dir / source_path.name
        if not candidate.exists():
            shutil.copy2(source_path, candidate)
            return candidate

        stem = source_path.stem
        suffix = source_path.suffix
        counter = 1
        while True:
            renamed = imports_dir / f"{stem}_{counter}{suffix}"
            if not renamed.exists():
                shutil.copy2(source_path, renamed)
                return renamed
            counter += 1

    def validate_document(self, document: ProjectDocument) -> list[str]:
        """Validate cross-entity references and return detected issues."""
        issues: list[str] = []

        if (
            document.project.host_structure_id
            and document.project.host_structure_id not in document.structures
        ):
            issues.append("project.host_structure_id does not reference a known structure")

        for structure in document.structures.values():
            if (
                structure.parent_structure_id
                and structure.parent_structure_id not in document.structures
            ):
                issues.append(
                    f"structure {structure.id} references missing parent structure {structure.parent_structure_id}"
                )
            asset = supercell_assets.supercell_asset(structure)
            if asset is None:
                continue
            if structure.parent_structure_id is None:
                issues.append(f"supercell asset {structure.id} is missing parent_structure_id")
            if (
                asset.source_asset_structure_id is not None
                and asset.source_asset_structure_id not in document.structures
            ):
                issues.append(
                    "supercell asset "
                    f"{structure.id} references missing source asset {asset.source_asset_structure_id}"
                )

        for calc in document.calculations.values():
            missing_structures = [
                sid for sid in calc.input_structure_ids if sid not in document.structures
            ]
            if missing_structures:
                issues.append(
                    f"calculation {calc.id} references missing structures: {', '.join(str(s) for s in missing_structures)}"
                )

        for run in document.runs.values():
            if run.definition_id and run.definition_id not in document.calculations:
                issues.append(
                    f"run {run.id} references missing calculation definition {run.definition_id}"
                )
            if run.job_id and run.job_id not in document.jobs:
                issues.append(f"run {run.id} references missing job {run.job_id}")
            if run.result_id and run.result_id not in document.results:
                issues.append(f"run {run.id} references missing result {run.result_id}")

        for result in document.results.values():
            if result.calculation_run_id and result.calculation_run_id not in document.runs:
                issues.append(
                    f"result {result.id} references missing calculation run {result.calculation_run_id}"
                )
            missing_structures = [
                sid for sid in result.derived_structure_ids if sid not in document.structures
            ]
            if missing_structures:
                issues.append(
                    f"result {result.id} references missing derived structures: {', '.join(str(s) for s in missing_structures)}"
                )

        for site in document.sites.values():
            if site.source_structure_id and site.source_structure_id not in document.structures:
                issues.append(
                    f"site {site.id} references missing structure {site.source_structure_id}"
                )
            if site.relaxation_run_id and site.relaxation_run_id not in document.runs:
                issues.append(f"site {site.id} references missing run {site.relaxation_run_id}")

        for table in document.comparison_tables.values():
            if (
                table.experimental_structure_id is not None
                and table.experimental_structure_id not in document.structures
            ):
                issues.append(
                    "comparison table "
                    f"{table.id} references missing experimental structure {table.experimental_structure_id}"
                )
            missing_structures = [
                sid for sid in table.dft_structure_ids if sid not in document.structures
            ]
            if missing_structures:
                issues.append(
                    "comparison table "
                    f"{table.id} references missing DFT structures: {', '.join(str(s) for s in missing_structures)}"
                )

        return issues

    def _assert_transition(self, current: JobStatus, target: JobStatus) -> None:
        allowed: dict[JobStatus, set[JobStatus]] = {
            JobStatus.DEFINED: {JobStatus.SUBMITTING, JobStatus.CANCELLED},
            JobStatus.SUBMITTING: {
                JobStatus.QUEUED,
                JobStatus.RUNNING,
                JobStatus.STOPPED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            },
            JobStatus.QUEUED: {
                JobStatus.RUNNING,
                JobStatus.STOPPED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            },
            JobStatus.RUNNING: {
                JobStatus.RETRIEVING,
                JobStatus.STOPPED,
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            },
            JobStatus.RETRIEVING: {
                JobStatus.STOPPED,
                JobStatus.COMPLETED,
                JobStatus.FAILED,
                JobStatus.CANCELLED,
            },
            # FAILED/STOPPED -> RETRIEVING supports the dashboard
            # "Retry retrieval" action: a cluster job whose scheduler run
            # finished but whose output retrieval failed keeps its remote
            # metadata, so retrieval can be re-attempted without resubmitting.
            # FAILED -> QUEUED/RUNNING/RETRIEVING also resumes a job failed for
            # a missing scheduler entry that the scheduler lists again
            # (``resume_reappeared_job``).
            JobStatus.STOPPED: {
                JobStatus.SUBMITTING,
                JobStatus.RETRIEVING,
                JobStatus.CANCELLED,
            },
            JobStatus.COMPLETED: set(),
            JobStatus.FAILED: {
                JobStatus.SUBMITTING,
                JobStatus.QUEUED,
                JobStatus.RUNNING,
                JobStatus.RETRIEVING,
                JobStatus.CANCELLED,
            },
            JobStatus.CANCELLED: {JobStatus.SUBMITTING},
        }
        if target not in allowed[current]:
            raise ValueError(f"Invalid job transition: {current.value} -> {target.value}")
