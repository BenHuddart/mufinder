"""Project-level orchestration for create/load/save workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import UUID

from mufinder._version import current_version
from mufinder.domain.enums import ACTIVE_JOB_STATUSES, JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    ComparisonTable,
    Job,
    MuonSite,
    Project,
    ProvenanceRecord,
    Result,
    Structure,
    utc_now,
)
from mufinder.persistence.codec import parse_datetime, parse_uuid, to_jsonable
from mufinder.persistence.json_store import ProjectStore
from mufinder.services.perf_trace import PerfTrace
from mufinder.science.site_generation import (
    SiteGenerationError,
    closest_symmetry_equivalent_position,
    estimate_random_site_count_from_separation,
    estimate_muon_muon_separation_for_count,
    generate_attached_element_sites,
    generate_random_sites,
    generate_random_sites_by_distance,
    lattice_cart_to_frac,
    lattice_frac_to_cart,
)
from mufinder.science.structure_io import load_structure_file
from mufinder.science.supercell import build_supercell_structure

_LOG = logging.getLogger(__name__)

_WORKFLOW_SUPERCELLS_STATE_KEY = "workflow_supercells"
_WORKFLOW_SUPERCELL_UNIT_CELL = "unit_cell"
_WORKFLOW_SUPERCELL_SAVED = "saved_supercell"
_IDENTITY_SUPERCELL_MATRIX = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
_GENERATED_SITE_STATUS = "generated"
_LEGACY_GENERATED_SITE_STATUSES = {_GENERATED_SITE_STATUS, "candidate"}
_STAGEABLE_SITE_STATUSES = {*_LEGACY_GENERATED_SITE_STATUSES, "staged"}

#: Message stamped on a local job that was still in progress when MuFinder was
#: closed and is reconciled to STOPPED on the next project open. See
#: ``ProjectService.reconcile_orphaned_local_jobs`` and docs/job_lifecycle.rst.
ORPHANED_LOCAL_JOB_STOP_MESSAGE = (
    "MuFinder was closed while this job was in progress locally; reset or resubmit it."
)


@dataclass(slots=True)
class ProjectDocument:
    """In-memory aggregate for all project content currently loaded."""

    project_root: Path
    project: Project
    structures: dict[UUID, Structure] = field(default_factory=dict)
    sites: dict[UUID, MuonSite] = field(default_factory=dict)
    calculations: dict[UUID, CalculationDefinition] = field(default_factory=dict)
    runs: dict[UUID, CalculationRun] = field(default_factory=dict)
    results: dict[UUID, Result] = field(default_factory=dict)
    comparison_tables: dict[UUID, ComparisonTable] = field(default_factory=dict)
    jobs: dict[UUID, Job] = field(default_factory=dict)
    provenance: list[ProvenanceRecord] = field(default_factory=list)
    ui_state: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class WorkflowSupercellSelection:
    """Persisted workflow simulation-cell selection."""

    mode: str = _WORKFLOW_SUPERCELL_UNIT_CELL
    structure_id: UUID | None = None


@dataclass(slots=True, frozen=True)
class WorkflowSupercellAsset:
    """Metadata for one saved reusable supercell structure."""

    structure_id: UUID
    label: str
    parent_structure_id: UUID | None
    owner_workflow_stage: WorkflowStage
    matrix: list[list[int]]
    source_asset_structure_id: UUID | None = None


@dataclass(slots=True)
class WorkflowStructureResolution:
    """Resolved workflow structure plus UI-facing selection state."""

    structure: Structure | None
    available_assets: list[WorkflowSupercellAsset]
    selection: WorkflowSupercellSelection
    normalized_selection: WorkflowSupercellSelection
    selected_asset: WorkflowSupercellAsset | None = None
    status_message: str = ""
    source_workflow_stage: WorkflowStage | None = None


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
    offenders: list[tuple[UUID, str, UUID | None, str]]  # (site_id, label, source_host_id, source_host_label)


@dataclass(slots=True)
class StructureHostMismatch:
    """Structures whose provenance chain doesn't resolve to the active host."""

    active_host_id: UUID
    active_host_label: str
    offenders: list[tuple[UUID, str, UUID | None, str]]  # (struct_id, label, resolved_host_id, host_label)


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

        project = self._project_from_dict(manifest)
        structures = {
            entity.id: entity for entity in (self._structure_from_dict(p) for p in store.load_entities("structures"))
        }
        sites = {entity.id: entity for entity in (self._site_from_dict(p) for p in store.load_entities("sites"))}
        calculations = {
            entity.id: entity
            for entity in (self._calculation_from_dict(p) for p in store.load_entities("calculations"))
        }
        runs = {entity.id: entity for entity in (self._run_from_dict(p) for p in store.load_entities("runs"))}
        results = {
            entity.id: entity for entity in (self._result_from_dict(p) for p in store.load_entities("results"))
        }
        comparison_tables = {
            entity.id: entity
            for entity in (self._comparison_table_from_dict(p) for p in store.load_entities("comparison_tables"))
        }
        jobs = {entity.id: entity for entity in (self._job_from_dict(p) for p in store.load_entities("jobs"))}
        provenance = [self._provenance_from_dict(p) for p in store.load_provenance()]
        ui_state = store.read_json(store.state_path) if store.state_path.exists() else {}

        document = ProjectDocument(
            project_root=project_root,
            project=project,
            structures=structures,
            sites=sites,
            calculations=calculations,
            runs=runs,
            results=results,
            comparison_tables=comparison_tables,
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
                self._content_hashes[(root_key, entity_type, entity_id)] = self._payload_hash(
                    to_jsonable(entity)
                )
        self._content_hashes[(root_key, self._UI_STATE_ENTITY_TYPE, self._UI_STATE_ENTITY_ID)] = (
            self._payload_hash(to_jsonable(ui_state))
        )

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
            if key[0] == root_key
            and key[1] != self._UI_STATE_ENTITY_TYPE
            and key not in live_keys
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

    def rename_comparison_table(self, document: ProjectDocument, table_id: UUID, label: str) -> ComparisonTable | None:
        """Update the display label for one persisted comparison table."""
        table = document.comparison_tables.get(table_id)
        if table is None:
            return None
        table.label = str(label).strip() or "Comparison Table"
        return table

    def remove_comparison_table(self, document: ProjectDocument, table_id: UUID) -> None:
        """Delete one persisted comparison table from the current document."""
        document.comparison_tables.pop(table_id, None)

    def remove_candidate_sites(self, document: ProjectDocument, site_ids: Sequence[UUID]) -> list[MuonSite]:
        """Remove generated muon sites by id and record the edit in provenance."""
        removed: list[MuonSite] = []
        for site_id in site_ids:
            site = document.sites.get(site_id)
            if site is None or site.status not in _LEGACY_GENERATED_SITE_STATUSES:
                continue
            removed.append(document.sites.pop(site_id))

        if removed:
            self.append_provenance(
                document,
                ProvenanceRecord(
                    action="delete_candidates",
                    workflow_stage=WorkflowStage.MUON_GENERATION,
                    actor="user",
                    description=f"Deleted {len(removed)} generated muon site(s)",
                    parameters={"site_labels": [site.label for site in removed]},
                    input_entity_ids=[site.id for site in removed],
                ),
            )
        return removed

    def cluster_candidate_positions(
        self,
        document: ProjectDocument,
        site_ids: Sequence[UUID],
        *,
        anchor_fractional: Sequence[float],
    ) -> list[MuonSite]:
        """Move selected generated muon sites to symmetry representatives near an anchor."""

        host = self._active_host_structure(document)
        if host is None:
            raise SiteGenerationError("Select a host structure before clustering generated muon positions.")
        if not site_ids:
            raise SiteGenerationError("Select one or more generated muon positions to cluster.")
        if len(anchor_fractional) != 3:
            raise SiteGenerationError("Enter a three-component fractional anchor position.")

        lattice = host.lattice_vectors
        host_species = host.species
        host_positions = host.fractional_positions
        anchor = [float(value) % 1.0 for value in anchor_fractional]
        updated: list[MuonSite] = []
        for site_id in site_ids:
            site = document.sites.get(site_id)
            if site is None or site.status not in _LEGACY_GENERATED_SITE_STATUSES:
                continue
            if len(site.fractional_position) != 3:
                raise SiteGenerationError(f"Muon site '{site.label}' is missing a valid fractional position.")
            clustered = closest_symmetry_equivalent_position(
                site.fractional_position,
                anchor_frac=anchor,
                lattice_vectors=lattice,
                host_species=host_species,
                host_positions_frac=host_positions,
            )
            site.fractional_position = clustered
            site.cartesian_position = lattice_frac_to_cart(clustered, lattice)
            updated.append(site)

        if not updated:
            raise SiteGenerationError("Selected rows did not contain any generated muon positions.")

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="cluster_candidate_positions",
                workflow_stage=WorkflowStage.MUON_GENERATION,
                actor="user",
                description=f"Clustered {len(updated)} generated muon site(s) near an anchor position",
                parameters={
                    "anchor_fractional": anchor,
                    "site_labels": [site.label for site in updated],
                },
                input_entity_ids=[site.id for site in updated],
                output_entity_ids=[site.id for site in updated],
            ),
        )
        return updated

    def stage_generated_sites(self, document: ProjectDocument, site_ids: Sequence[UUID]) -> list[MuonSite]:
        """Mark generated muon sites as staged for structural relaxation."""

        staged: list[MuonSite] = []
        for site_id in site_ids:
            site = document.sites.get(site_id)
            if site is None or site.status not in _STAGEABLE_SITE_STATUSES:
                continue
            site.status = "staged"
            staged.append(site)

        if staged:
            self.append_provenance(
                document,
                ProvenanceRecord(
                    action="stage_muon_relaxation_sites",
                    workflow_stage=WorkflowStage.MUON_RELAXATION,
                    actor="user",
                    description=f"Staged {len(staged)} generated muon site(s) for relaxation",
                    parameters={"site_labels": [site.label for site in staged]},
                    input_entity_ids=[site.id for site in staged],
                    output_entity_ids=[site.id for site in staged],
                ),
            )
        return staged

    def record_job_submission(
        self,
        document: ProjectDocument,
        *,
        jobs: Sequence[Job],
        execution_target: str,
    ) -> None:
        """Append one provenance milestone for a user submission action."""

        submitted_jobs = [job for job in jobs if job.status in {JobStatus.SUBMITTING, JobStatus.QUEUED}]
        if not submitted_jobs:
            return

        workflow_stage = submitted_jobs[0].workflow_stage if len({job.workflow_stage for job in submitted_jobs}) == 1 else None
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
                            "submitted_at": job.submitted_at.isoformat() if job.submitted_at else None,
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
                input_entity_ids=[table.experimental_structure_id] if table.experimental_structure_id is not None else [],
                output_entity_ids=[table.id, *table.dft_structure_ids],
            ),
        )

    @staticmethod
    def _identity_supercell_matrix() -> list[list[int]]:
        return [[int(value) for value in row] for row in _IDENTITY_SUPERCELL_MATRIX]

    @staticmethod
    def _coerce_supercell_matrix(matrix: Any) -> list[list[int]] | None:
        if not isinstance(matrix, list) or len(matrix) != 3:
            return None
        parsed: list[list[int]] = []
        for row in matrix:
            if not isinstance(row, list) or len(row) != 3:
                return None
            parsed.append([int(row[0]), int(row[1]), int(row[2])])
        return parsed

    @staticmethod
    def _parse_optional_uuid(value: Any) -> UUID | None:
        if value is None:
            return None
        try:
            return UUID(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _supercell_asset_label(workflow_stage: WorkflowStage, matrix: list[list[int]]) -> str:
        if (
            matrix[0][1] == 0
            and matrix[0][2] == 0
            and matrix[1][0] == 0
            and matrix[1][2] == 0
            and matrix[2][0] == 0
            and matrix[2][1] == 0
        ):
            token = f"{matrix[0][0]}x{matrix[1][1]}x{matrix[2][2]}"
        else:
            token = (
                f"[{matrix[0][0]} {matrix[0][1]} {matrix[0][2]}; "
                f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]}; "
                f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]}]"
            )
        return f"{workflow_stage.value} {token}"

    def _workflow_supercells_state(self, document: ProjectDocument) -> dict[str, Any]:
        raw_state = document.ui_state.get(_WORKFLOW_SUPERCELLS_STATE_KEY)
        if isinstance(raw_state, dict):
            return raw_state
        state: dict[str, Any] = {}
        document.ui_state[_WORKFLOW_SUPERCELLS_STATE_KEY] = state
        return state

    def get_workflow_supercell_selection(
        self,
        document: ProjectDocument,
        workflow_stage: WorkflowStage,
    ) -> WorkflowSupercellSelection:
        raw = self._workflow_supercells_state(document).get(workflow_stage.value, {})
        if not isinstance(raw, dict):
            return WorkflowSupercellSelection()

        mode = str(raw.get("mode", _WORKFLOW_SUPERCELL_UNIT_CELL)).strip().lower()
        if mode != _WORKFLOW_SUPERCELL_SAVED:
            return WorkflowSupercellSelection()

        structure_id = self._parse_optional_uuid(raw.get("structure_id"))
        if structure_id is None:
            return WorkflowSupercellSelection()
        return WorkflowSupercellSelection(mode=_WORKFLOW_SUPERCELL_SAVED, structure_id=structure_id)

    def set_workflow_supercell_selection(
        self,
        document: ProjectDocument,
        workflow_stage: WorkflowStage,
        selection: WorkflowSupercellSelection,
    ) -> None:
        state = self._workflow_supercells_state(document)
        if selection.mode == _WORKFLOW_SUPERCELL_SAVED and selection.structure_id is not None:
            state[workflow_stage.value] = {
                "mode": _WORKFLOW_SUPERCELL_SAVED,
                "structure_id": str(selection.structure_id),
            }
            return

        state[workflow_stage.value] = {"mode": _WORKFLOW_SUPERCELL_UNIT_CELL}

    def _supercell_asset(self, structure: Structure) -> WorkflowSupercellAsset | None:
        raw = structure.properties.get("supercell_asset")
        if not isinstance(raw, dict):
            return None

        matrix = self._coerce_supercell_matrix(raw.get("matrix"))
        if matrix is None:
            return None

        owner = raw.get("owner_workflow_stage")
        try:
            owner_stage = WorkflowStage(str(owner))
        except ValueError:
            return None

        return WorkflowSupercellAsset(
            structure_id=structure.id,
            label=structure.label,
            parent_structure_id=structure.parent_structure_id,
            owner_workflow_stage=owner_stage,
            matrix=matrix,
            source_asset_structure_id=self._parse_optional_uuid(raw.get("source_asset_structure_id")),
        )

    def list_workflow_supercell_assets(
        self,
        document: ProjectDocument,
        *,
        host_structure_id: UUID | None,
    ) -> list[WorkflowSupercellAsset]:
        if host_structure_id is None:
            return []

        assets: list[WorkflowSupercellAsset] = []
        for structure in document.structures.values():
            if structure.parent_structure_id != host_structure_id:
                continue
            asset = self._supercell_asset(structure)
            if asset is None:
                continue
            assets.append(asset)

        assets.sort(
            key=lambda asset: (
                asset.owner_workflow_stage.value,
                asset.label.casefold(),
                str(asset.structure_id),
            )
        )
        return assets

    def _find_matching_workflow_supercell_asset(
        self,
        document: ProjectDocument,
        *,
        host_structure_id: UUID,
        workflow_stage: WorkflowStage,
        matrix: list[list[int]],
    ) -> Structure | None:
        for structure in document.structures.values():
            if structure.parent_structure_id != host_structure_id:
                continue
            asset = self._supercell_asset(structure)
            if asset is None:
                continue
            if asset.owner_workflow_stage != workflow_stage:
                continue
            if asset.matrix != matrix:
                continue
            return structure
        return None

    @staticmethod
    def _has_nonzero_spin(values: Sequence[float], *, tolerance: float = 1e-12) -> bool:
        return any(abs(float(value)) > tolerance for value in values)

    @staticmethod
    def _spins_from_magnetic_structure(structure: Structure, *, atom_count: int) -> list[float] | None:
        magnetic = structure.properties.get("magnetic_structure")
        if not isinstance(magnetic, dict):
            return None

        moments = magnetic.get("moments")
        if not isinstance(moments, list) or not moments:
            return None

        spins = [0.0] * atom_count
        saw_numeric = False
        for index, moment in enumerate(moments[:atom_count]):
            if not isinstance(moment, dict):
                continue
            vector = moment.get("vector")
            if not isinstance(vector, list) or len(vector) < 3:
                continue
            try:
                spins[index] = float(vector[2])
                saw_numeric = True
            except (TypeError, ValueError):
                continue
        return spins if saw_numeric else None

    @staticmethod
    def _structure_atomic_spins(structure: Structure) -> list[float]:
        raw = structure.properties.get("atomic_spins")
        if raw is None:
            raw = structure.properties.get("initial_magnetic_moments")

        spins = [0.0] * len(structure.species)
        saw_numeric = False
        if isinstance(raw, list):
            for index, value in enumerate(raw[: len(spins)]):
                try:
                    spins[index] = float(value)
                    saw_numeric = True
                except (TypeError, ValueError):
                    spins[index] = 0.0

        fallback = ProjectService._spins_from_magnetic_structure(structure, atom_count=len(spins))
        if saw_numeric:
            if isinstance(structure.properties.get("supercell_asset"), dict):
                return spins
            if (
                fallback is not None
                and not ProjectService._has_nonzero_spin(spins)
                and ProjectService._has_nonzero_spin(fallback)
            ):
                return fallback
            return spins

        if fallback is not None:
            return fallback
        return spins

    def _repair_legacy_supercell_asset_spins(
        self,
        *,
        host: Structure,
        structure: Structure,
        asset: WorkflowSupercellAsset,
    ) -> bool:
        host_spins = self._structure_atomic_spins(host)
        if not self._has_nonzero_spin(host_spins):
            return False

        raw = structure.properties.get("atomic_spins")
        raw_spins = [0.0] * len(structure.species)
        saw_numeric = False
        if isinstance(raw, list):
            for index, value in enumerate(raw[: len(raw_spins)]):
                try:
                    raw_spins[index] = float(value)
                    saw_numeric = True
                except (TypeError, ValueError):
                    raw_spins[index] = 0.0
        if saw_numeric and self._has_nonzero_spin(raw_spins):
            return False

        try:
            repaired, expanded_spins = build_supercell_structure(
                host,
                asset.matrix,
                atomic_spins=host_spins,
                propagation_vector=self._structure_propagation_vector(host),
            )
        except ValueError:
            return False

        if not self._has_nonzero_spin(expanded_spins):
            return False

        structure.lattice_vectors = [list(vector) for vector in repaired.lattice_vectors]
        structure.species = list(repaired.species)
        structure.fractional_positions = [list(position) for position in repaired.fractional_positions]
        repaired_properties = dict(repaired.properties)
        repaired_properties["supercell_asset"] = {
            "matrix": [list(row) for row in asset.matrix],
            "owner_workflow_stage": asset.owner_workflow_stage.value,
            "source_asset_structure_id": (
                str(asset.source_asset_structure_id) if asset.source_asset_structure_id is not None else None
            ),
        }
        structure.properties = repaired_properties
        return True

    def repair_supercell_asset_if_needed(
        self,
        document: ProjectDocument,
        *,
        structure_id: UUID,
    ) -> bool:
        structure = document.structures.get(structure_id)
        if structure is None:
            return False
        asset = self._supercell_asset(structure)
        if asset is None:
            return False
        parent_id = structure.parent_structure_id
        if parent_id is None:
            return False
        host = document.structures.get(parent_id)
        if host is None:
            return False
        return self._repair_legacy_supercell_asset_spins(host=host, structure=structure, asset=asset)

    def _refresh_supercell_asset_if_needed(
        self,
        *,
        host: Structure,
        structure: Structure,
    ) -> None:
        if structure.parent_structure_id != host.id:
            return
        asset = self._supercell_asset(structure)
        if asset is None:
            return
        self._repair_legacy_supercell_asset_spins(host=host, structure=structure, asset=asset)

    @staticmethod
    def _structure_propagation_vector(structure: Structure) -> list[float]:
        source = structure.properties.get("propagation_vector")
        if source is None:
            magnetic = structure.properties.get("magnetic_structure")
            if isinstance(magnetic, dict):
                source = magnetic.get("propagation_vector")

        vector = [0.0, 0.0, 0.0]
        if isinstance(source, list):
            for index, value in enumerate(source[:3]):
                try:
                    vector[index] = float(value)
                except (TypeError, ValueError):
                    vector[index] = 0.0
        return vector

    def structure_atomic_spins(self, structure: Structure) -> list[float]:
        """Return numeric per-atom spins for a structure, defaulting missing values to zero."""
        return self._structure_atomic_spins(structure)

    def structure_propagation_vector(self, structure: Structure) -> list[float]:
        """Return the structure propagation vector, defaulting to [0, 0, 0]."""
        return self._structure_propagation_vector(structure)

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
        host = document.structures.get(host_structure_id)
        if host is None:
            raise ValueError("host_structure_id must reference a known structure.")

        parsed_matrix = self._coerce_supercell_matrix(matrix)
        if parsed_matrix is None:
            raise ValueError("Supercell matrix must be a 3x3 integer matrix.")

        if reuse_existing:
            existing = self._find_matching_workflow_supercell_asset(
                document,
                host_structure_id=host_structure_id,
                workflow_stage=workflow_stage,
                matrix=parsed_matrix,
            )
            if existing is not None:
                self._refresh_supercell_asset_if_needed(host=host, structure=existing)
                return existing

        structure, _expanded_spins = build_supercell_structure(
            host,
            parsed_matrix,
            atomic_spins=self._structure_atomic_spins(host),
            propagation_vector=self._structure_propagation_vector(host),
        )
        structure.label = str(label or "").strip() or self._supercell_asset_label(workflow_stage, parsed_matrix)
        structure.parent_structure_id = host_structure_id
        structure.properties["supercell_asset"] = {
            "matrix": [list(row) for row in parsed_matrix],
            "owner_workflow_stage": workflow_stage.value,
            "source_asset_structure_id": (
                str(source_asset_structure_id) if source_asset_structure_id is not None else None
            ),
        }
        document.structures[structure.id] = structure
        return structure

    def supercell_asset_references(
        self,
        document: ProjectDocument,
        *,
        asset_structure_id: UUID,
    ) -> list[str]:
        """Return user-facing references to a saved supercell asset."""

        references: list[str] = []
        for definition in document.calculations.values():
            if asset_structure_id in definition.input_structure_ids:
                references.append(
                    f"{definition.workflow_stage.value} calculation '{definition.calc_type}'"
                )

        for stage_value, raw in self._workflow_supercells_state(document).items():
            if not isinstance(raw, dict):
                continue
            raw_id = self._parse_optional_uuid(raw.get("structure_id"))
            if raw_id == asset_structure_id:
                references.append(f"{stage_value} active selection")

        return references

    def supercell_asset_is_frozen(
        self,
        document: ProjectDocument,
        *,
        asset_structure_id: UUID,
    ) -> bool:
        """Return whether the asset is frozen into a calculation definition."""

        return any(asset_structure_id in definition.input_structure_ids for definition in document.calculations.values())

    def rename_workflow_supercell_asset(
        self,
        document: ProjectDocument,
        *,
        asset_structure_id: UUID,
        label: str,
    ) -> Structure:
        """Rename a saved supercell asset."""

        structure = document.structures.get(asset_structure_id)
        if structure is None or self._supercell_asset(structure) is None:
            raise ValueError("Selected structure is not a saved supercell asset.")
        cleaned = str(label).strip()
        if not cleaned:
            raise ValueError("Supercell name must not be empty.")
        structure.label = cleaned
        return structure

    def update_workflow_supercell_asset_matrix(
        self,
        document: ProjectDocument,
        *,
        asset_structure_id: UUID,
        matrix: list[list[int]],
        label: str | None = None,
    ) -> Structure:
        """Regenerate an unfrozen saved supercell asset with a new matrix."""

        structure = document.structures.get(asset_structure_id)
        if structure is None:
            raise ValueError("Selected supercell does not exist in this project.")
        asset = self._supercell_asset(structure)
        if asset is None:
            raise ValueError("Selected structure is not a saved supercell asset.")
        if self.supercell_asset_is_frozen(document, asset_structure_id=asset_structure_id):
            raise ValueError("This supercell is already used by configured calculations; save a new copy instead.")
        if structure.parent_structure_id is None:
            raise ValueError("Saved supercell is missing its source structure.")
        host = document.structures.get(structure.parent_structure_id)
        if host is None:
            raise ValueError("Saved supercell source structure is no longer available.")

        parsed_matrix = self._coerce_supercell_matrix(matrix)
        if parsed_matrix is None:
            raise ValueError("Supercell matrix must be a 3x3 integer matrix.")

        regenerated, _expanded_spins = build_supercell_structure(
            host,
            parsed_matrix,
            atomic_spins=self._structure_atomic_spins(host),
            propagation_vector=self._structure_propagation_vector(host),
        )
        structure.lattice_vectors = [list(vector) for vector in regenerated.lattice_vectors]
        structure.species = list(regenerated.species)
        structure.fractional_positions = [list(position) for position in regenerated.fractional_positions]
        structure.label = str(label or "").strip() or structure.label or self._supercell_asset_label(
            asset.owner_workflow_stage,
            parsed_matrix,
        )
        structure.properties = dict(regenerated.properties)
        structure.properties["supercell_asset"] = {
            "matrix": [list(row) for row in parsed_matrix],
            "owner_workflow_stage": asset.owner_workflow_stage.value,
            "source_asset_structure_id": (
                str(asset.source_asset_structure_id) if asset.source_asset_structure_id is not None else None
            ),
        }
        structure.parent_structure_id = host.id
        return structure

    def delete_workflow_supercell_asset(
        self,
        document: ProjectDocument,
        *,
        asset_structure_id: UUID,
    ) -> Structure:
        """Delete an unused saved supercell asset and clear stale active selections."""

        structure = document.structures.get(asset_structure_id)
        if structure is None or self._supercell_asset(structure) is None:
            raise ValueError("Selected structure is not a saved supercell asset.")
        if self.supercell_asset_is_frozen(document, asset_structure_id=asset_structure_id):
            raise ValueError("This supercell is already used by configured calculations and cannot be deleted.")

        removed = document.structures.pop(asset_structure_id)
        state = self._workflow_supercells_state(document)
        for stage_value, raw in list(state.items()):
            if not isinstance(raw, dict):
                continue
            if self._parse_optional_uuid(raw.get("structure_id")) == asset_structure_id:
                state[stage_value] = {"mode": _WORKFLOW_SUPERCELL_UNIT_CELL}
        return removed

    def use_workflow_supercell_asset(
        self,
        document: ProjectDocument,
        *,
        workflow_stage: WorkflowStage,
        asset_structure_id: UUID,
    ) -> Structure:
        host = self._active_host_structure(document)
        if host is None:
            raise ValueError("Load a host structure before selecting a saved supercell.")

        asset_structure = document.structures.get(asset_structure_id)
        if asset_structure is None:
            raise ValueError("Selected supercell does not exist in this project.")

        asset = self._supercell_asset(asset_structure)
        if asset is None:
            raise ValueError("Selected structure is not a saved supercell asset.")
        if asset_structure.parent_structure_id != host.id:
            raise ValueError("Selected supercell does not belong to the active host structure.")

        selected = asset_structure
        if asset.owner_workflow_stage != workflow_stage:
            selected = self._find_matching_workflow_supercell_asset(
                document,
                host_structure_id=host.id,
                workflow_stage=workflow_stage,
                matrix=asset.matrix,
            )
            if selected is None:
                selected = self.create_workflow_supercell_asset(
                    document,
                    workflow_stage=workflow_stage,
                    host_structure_id=host.id,
                    matrix=asset.matrix,
                    source_asset_structure_id=asset.structure_id,
                    reuse_existing=False,
                )
        self._refresh_supercell_asset_if_needed(host=host, structure=selected)

        self.set_workflow_supercell_selection(
            document,
            workflow_stage,
            WorkflowSupercellSelection(
                mode=_WORKFLOW_SUPERCELL_SAVED,
                structure_id=selected.id,
            ),
        )
        return selected

    def clear_workflow_supercell_selection(
        self,
        document: ProjectDocument,
        workflow_stage: WorkflowStage,
    ) -> None:
        self.set_workflow_supercell_selection(document, workflow_stage, WorkflowSupercellSelection())

    def resolve_workflow_structure(
        self,
        document: ProjectDocument,
        workflow_stage: WorkflowStage,
        *,
        normalize_state: bool = False,
    ) -> WorkflowStructureResolution:
        host = self._active_host_structure(document)
        selection = self.get_workflow_supercell_selection(document, workflow_stage)
        normalized = selection
        available_assets = self.list_workflow_supercell_assets(
            document,
            host_structure_id=host.id if host is not None else None,
        )

        if host is None:
            normalized = WorkflowSupercellSelection()
            if normalize_state:
                self.set_workflow_supercell_selection(document, workflow_stage, normalized)
            return WorkflowStructureResolution(
                structure=None,
                available_assets=[],
                selection=selection,
                normalized_selection=normalized,
                status_message="Load a host structure to choose a simulation cell.",
            )

        if selection.mode == _WORKFLOW_SUPERCELL_SAVED and selection.structure_id is not None:
            selected_structure = document.structures.get(selection.structure_id)
            selected_asset = (
                self._supercell_asset(selected_structure)
                if selected_structure is not None
                else None
            )
            if (
                selected_structure is not None
                and selected_asset is not None
                and selected_structure.parent_structure_id == host.id
            ):
                self._refresh_supercell_asset_if_needed(host=host, structure=selected_structure)
                selected_asset = self._supercell_asset(selected_structure)
                if selected_asset is None:
                    normalized = WorkflowSupercellSelection()
                    if normalize_state:
                        self.set_workflow_supercell_selection(document, workflow_stage, normalized)
                    return WorkflowStructureResolution(
                        structure=host,
                        available_assets=available_assets,
                        selection=selection,
                        normalized_selection=normalized,
                        status_message="Saved supercell unavailable for the active host; using single unit cell.",
                    )
                source_workflow_stage = selected_asset.owner_workflow_stage
                if selected_asset.source_asset_structure_id is not None:
                    source_structure = document.structures.get(selected_asset.source_asset_structure_id)
                    source_asset = (
                        self._supercell_asset(source_structure)
                        if source_structure is not None
                        else None
                    )
                    if source_asset is not None:
                        source_workflow_stage = source_asset.owner_workflow_stage
                return WorkflowStructureResolution(
                    structure=selected_structure,
                    available_assets=available_assets,
                    selection=selection,
                    normalized_selection=selection,
                    selected_asset=selected_asset,
                    status_message=f"Using saved supercell: {selected_structure.label}",
                    source_workflow_stage=source_workflow_stage,
                )

            normalized = WorkflowSupercellSelection()
            if normalize_state:
                self.set_workflow_supercell_selection(document, workflow_stage, normalized)
            return WorkflowStructureResolution(
                structure=host,
                available_assets=available_assets,
                selection=selection,
                normalized_selection=normalized,
                status_message="Saved supercell unavailable for the active host; using single unit cell.",
            )

        if normalize_state and normalized != selection:
            self.set_workflow_supercell_selection(document, workflow_stage, normalized)

        return WorkflowStructureResolution(
            structure=host,
            available_assets=available_assets,
            selection=selection,
            normalized_selection=normalized,
            status_message="Using single unit cell.",
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

        structure = Structure(
            label=label or source_path.stem,
            origin="imported",
            lattice_vectors=loaded.lattice_vectors,
            species=loaded.species,
            fractional_positions=loaded.fractional_positions,
            source_file=str(copied_path.relative_to(document.project_root)),
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
                    "project_copy": str(copied_path.relative_to(document.project_root)),
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
        simulation_cell_mode: str = _WORKFLOW_SUPERCELL_UNIT_CELL,
        supercell_source_workflow: str | None = None,
    ) -> CalculationDefinition:
        """Persist one convergence-cutoff configuration as a reusable definition.

        This records the configured parameter snapshot as a first-class entity,
        separate from ephemeral UI form state.
        """

        if input_structure_id is None:
            input_structure_id = host_structure_id

        if input_structure_id is None or input_structure_id not in document.structures:
            raise ValueError("input_structure_id must reference a known structure.")

        relative_output = str(output_directory)
        try:
            relative_output = str(output_directory.relative_to(document.project_root))
        except ValueError:
            relative_output = str(output_directory)

        snapshot = to_jsonable(parameters)
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
            parameters={
                "configuration": snapshot,
                "output_directory": relative_output,
                "job_ids_by_cutoff": dict(job_ids_by_cutoff or {}),
                "simulation_cell_mode": simulation_cell_mode,
                "supercell_source_workflow": supercell_source_workflow,
            },
            input_structure_ids=[input_structure_id],
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="configure_convergence_cutoff",
                workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
                actor="user",
                description=(
                    "Configured cutoff convergence with "
                    f"{len(snapshot.get('cutoff_energies', []))} cutoff value(s)"
                ),
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": snapshot,
                    "output_directory": relative_output,
                    "job_ids_by_cutoff": dict(job_ids_by_cutoff or {}),
                    "simulation_cell_mode": simulation_cell_mode,
                    "supercell_source_workflow": supercell_source_workflow,
                },
                input_entity_ids=[input_structure_id],
                output_entity_ids=[definition.id],
            ),
        )

        return definition

    def register_convergence_kpoint_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_id: UUID | None = None,
        host_structure_id: UUID | None = None,
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_grid: dict[str, str] | None = None,
        simulation_cell_mode: str = _WORKFLOW_SUPERCELL_UNIT_CELL,
        supercell_source_workflow: str | None = None,
    ) -> CalculationDefinition:
        """Persist one convergence-kpoint configuration as a reusable definition."""

        if input_structure_id is None:
            input_structure_id = host_structure_id

        if input_structure_id is None or input_structure_id not in document.structures:
            raise ValueError("input_structure_id must reference a known structure.")

        relative_output = str(output_directory)
        try:
            relative_output = str(output_directory.relative_to(document.project_root))
        except ValueError:
            relative_output = str(output_directory)

        snapshot = to_jsonable(parameters)
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
            calc_type="castep_convergence_kpoint",
            parameters={
                "configuration": snapshot,
                "output_directory": relative_output,
                "job_ids_by_grid": dict(job_ids_by_grid or {}),
                "simulation_cell_mode": simulation_cell_mode,
                "supercell_source_workflow": supercell_source_workflow,
            },
            input_structure_ids=[input_structure_id],
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="configure_convergence_kpoint",
                workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
                actor="user",
                description=(
                    "Configured k-point convergence with "
                    f"{len(snapshot.get('sweep_values', []))} grid value(s)"
                ),
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": snapshot,
                    "output_directory": relative_output,
                    "job_ids_by_grid": dict(job_ids_by_grid or {}),
                    "simulation_cell_mode": simulation_cell_mode,
                    "supercell_source_workflow": supercell_source_workflow,
                },
                input_entity_ids=[input_structure_id],
                output_entity_ids=[definition.id],
            ),
        )

        return definition

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

        if input_structure_id is None:
            input_structure_id = host_structure_id

        if input_structure_id is None or input_structure_id not in document.structures:
            raise ValueError("input_structure_id must reference a known structure.")

        relative_output = str(output_directory)
        try:
            relative_output = str(output_directory.relative_to(document.project_root))
        except ValueError:
            relative_output = str(output_directory)

        snapshot = to_jsonable(parameters)
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
            calc_type="castep_pristine_relaxation",
            parameters={
                "configuration": snapshot,
                "output_directory": relative_output,
                "job_ids_by_label": dict(job_ids_by_label or {}),
            },
            input_structure_ids=[input_structure_id],
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="configure_pristine_relaxation",
                workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
                actor="user",
                description=(
                    "Configured pristine relaxation with "
                    f"{len(snapshot.get('rows', []))} job row(s)"
                ),
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": snapshot,
                    "output_directory": relative_output,
                    "job_ids_by_label": dict(job_ids_by_label or {}),
                },
                input_entity_ids=[input_structure_id],
                output_entity_ids=[definition.id],
            ),
        )

        return definition

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
        """Persist one muon-relaxation configuration."""

        if input_structure_id is None:
            input_structure_id = host_structure_id

        if input_structure_id is None or input_structure_id not in document.structures:
            raise ValueError("input_structure_id must reference a known structure.")

        relative_output = str(output_directory)
        try:
            relative_output = str(output_directory.relative_to(document.project_root))
        except ValueError:
            relative_output = str(output_directory)

        snapshot = to_jsonable(parameters)
        input_site_ids: list[UUID] = []
        for row in snapshot.get("rows", []):
            if not isinstance(row, dict) or not row.get("site_id"):
                continue
            try:
                input_site_ids.append(UUID(str(row["site_id"])))
            except ValueError:
                continue

        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_muon_relaxation",
            parameters={
                "configuration": snapshot,
                "output_directory": relative_output,
                "job_ids_by_label": dict(job_ids_by_label or {}),
            },
            input_structure_ids=[input_structure_id],
            input_site_ids=input_site_ids,
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="configure_muon_relaxation",
                workflow_stage=WorkflowStage.MUON_RELAXATION,
                actor="user",
                description=f"Configured muon relaxation with {len(snapshot.get('rows', []))} job row(s)",
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": snapshot,
                    "output_directory": relative_output,
                    "job_ids_by_label": dict(job_ids_by_label or {}),
                },
                input_entity_ids=[input_structure_id, *definition.input_site_ids],
                output_entity_ids=[definition.id],
            ),
        )

        return definition

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

        if input_structure_id is None:
            input_structure_id = host_structure_id

        if input_structure_id is None or input_structure_id not in document.structures:
            raise ValueError("input_structure_id must reference a known structure.")

        relative_output = str(output_directory)
        try:
            relative_output = str(output_directory.relative_to(document.project_root))
        except ValueError:
            relative_output = str(output_directory)

        snapshot = to_jsonable(parameters)
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.UEP,
            calc_type="castep_uep_source",
            parameters={
                "configuration": snapshot,
                "output_directory": relative_output,
                "job_ids_by_label": dict(job_ids_by_label or {}),
            },
            input_structure_ids=[input_structure_id],
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="configure_uep",
                workflow_stage=WorkflowStage.UEP,
                actor="user",
                description=(
                    "Configured UEP source workflow with "
                    f"{len(snapshot.get('rows', []))} job row(s)"
                ),
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": snapshot,
                    "output_directory": relative_output,
                    "job_ids_by_label": dict(job_ids_by_label or {}),
                },
                input_entity_ids=[input_structure_id],
                output_entity_ids=[definition.id],
            ),
        )

        return definition

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
        """Persist one Quantum Corrections phonon configuration."""

        if not input_structure_ids:
            raise ValueError("At least one input structure is required.")
        missing_structures = [structure_id for structure_id in input_structure_ids if structure_id not in document.structures]
        if missing_structures:
            raise ValueError("input_structure_ids must reference known structures.")
        missing_sites = [site_id for site_id in input_site_ids if site_id not in document.sites]
        if missing_sites:
            raise ValueError("input_site_ids must reference known sites.")

        relative_output = str(output_directory)
        try:
            relative_output = str(output_directory.relative_to(document.project_root))
        except ValueError:
            relative_output = str(output_directory)

        snapshot = to_jsonable(parameters)
        singlepoint_job_ids_by_label = snapshot.get("singlepoint_job_ids_by_label", {})
        phonon_job_ids_by_label = snapshot.get("phonon_job_ids_by_label", {})
        job_seed_names_by_label = snapshot.get("job_seed_names_by_label", {})
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type="castep_quantum_phonon_set",
            parameters={
                "configuration": snapshot,
                "output_directory": relative_output,
                "job_ids_by_label": dict(job_ids_by_label or {}),
                "singlepoint_job_ids_by_label": (
                    dict(singlepoint_job_ids_by_label) if isinstance(singlepoint_job_ids_by_label, dict) else {}
                ),
                "phonon_job_ids_by_label": (
                    dict(phonon_job_ids_by_label) if isinstance(phonon_job_ids_by_label, dict) else {}
                ),
                "job_seed_names_by_label": (
                    dict(job_seed_names_by_label) if isinstance(job_seed_names_by_label, dict) else {}
                ),
            },
            input_structure_ids=list(input_structure_ids),
            input_site_ids=list(input_site_ids),
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="configure_quantum_corrections",
                workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
                actor="user",
                description=f"Configured Quantum Corrections with {len(input_site_ids)} relaxed site(s)",
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": snapshot,
                    "output_directory": relative_output,
                    "job_ids_by_label": dict(job_ids_by_label or {}),
                },
                input_entity_ids=[*input_structure_ids, *input_site_ids],
                output_entity_ids=[definition.id],
            ),
        )

        return definition

    def register_transition_state_search_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_ids: list[UUID],
        input_site_ids: list[UUID],
        parameters: dict[str, Any],
        output_directory: Path,
        job_ids_by_label: dict[str, str] | None = None,
    ) -> CalculationDefinition:
        """Persist one Energy Barriers transition-state-search configuration."""

        if not input_structure_ids:
            raise ValueError("At least one input structure is required.")
        missing_structures = [structure_id for structure_id in input_structure_ids if structure_id not in document.structures]
        if missing_structures:
            raise ValueError("input_structure_ids must reference known structures.")
        missing_sites = [site_id for site_id in input_site_ids if site_id not in document.sites]
        if missing_sites:
            raise ValueError("input_site_ids must reference known sites.")

        relative_output = str(output_directory)
        try:
            relative_output = str(output_directory.relative_to(document.project_root))
        except ValueError:
            relative_output = str(output_directory)

        snapshot = to_jsonable(parameters)
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.ENERGY_BARRIERS,
            calc_type="castep_transition_state_search_set",
            parameters={
                "configuration": snapshot,
                "output_directory": relative_output,
                "job_ids_by_label": dict(job_ids_by_label or {}),
            },
            input_structure_ids=list(input_structure_ids),
            input_site_ids=list(input_site_ids),
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="configure_energy_barriers",
                workflow_stage=WorkflowStage.ENERGY_BARRIERS,
                actor="user",
                description=f"Configured Energy Barriers with {len(input_site_ids)} relaxed site(s)",
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": snapshot,
                    "output_directory": relative_output,
                    "job_ids_by_label": dict(job_ids_by_label or {}),
                },
                input_entity_ids=[*input_structure_ids, *input_site_ids],
                output_entity_ids=[definition.id],
            ),
        )

        return definition

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

        if not input_structure_ids:
            raise ValueError("At least one input structure is required.")
        missing_structures = [structure_id for structure_id in input_structure_ids if structure_id not in document.structures]
        if missing_structures:
            raise ValueError("input_structure_ids must reference known structures.")
        missing_sites = [site_id for site_id in input_site_ids if site_id not in document.sites]
        if missing_sites:
            raise ValueError("input_site_ids must reference known sites.")

        relative_output = str(output_directory)
        try:
            relative_output = str(output_directory.relative_to(document.project_root))
        except ValueError:
            relative_output = str(output_directory)

        snapshot = to_jsonable(parameters)
        definition = CalculationDefinition(
            workflow_stage=WorkflowStage.ENERGY_BARRIERS,
            calc_type="castep_nudged_elastic_band_set",
            parameters={
                "configuration": snapshot,
                "output_directory": relative_output,
                "job_ids_by_label": dict(job_ids_by_label or {}),
            },
            input_structure_ids=list(input_structure_ids),
            input_site_ids=list(input_site_ids),
        )
        document.calculations[definition.id] = definition

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="configure_energy_barriers",
                workflow_stage=WorkflowStage.ENERGY_BARRIERS,
                actor="user",
                description=f"Configured Energy Barriers NEB with {len(input_site_ids)} relaxed site(s)",
                parameters={
                    "calculation_definition_id": str(definition.id),
                    "configuration": snapshot,
                    "output_directory": relative_output,
                    "job_ids_by_label": dict(job_ids_by_label or {}),
                },
                input_entity_ids=[*input_structure_ids, *input_site_ids],
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
    ) -> None:
        """Persist a project-wide recommended cutoff energy for future workflows."""
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
                },
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
    ) -> None:
        """Persist a project-wide recommended k-point grid for future workflows."""

        grid = (int(kpoint_grid[0]), int(kpoint_grid[1]), int(kpoint_grid[2]))
        document.project.recommended_kpoint_grid = grid
        document.project.recommended_kpoint_lattice = (
            [
                [float(row[0]), float(row[1]), float(row[2])]
                for row in lattice_vectors
            ]
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

        self.append_provenance(
            document,
            ProvenanceRecord(
                action="set_recommended_kpoint_grid",
                workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
                actor="user",
                description=(
                    "Set project recommended k-point grid to "
                    f"{grid[0]}x{grid[1]}x{grid[2]}"
                ),
                parameters={
                    "recommended_kpoint_grid": [grid[0], grid[1], grid[2]],
                    "recommended_kpoint_lattice": document.project.recommended_kpoint_lattice,
                    "recommended_kpoint_method": document.project.recommended_kpoint_method,
                    "recommended_kpoint_energy_threshold": document.project.recommended_kpoint_energy_threshold,
                    "recommended_kpoint_force_threshold": document.project.recommended_kpoint_force_threshold,
                    "recommended_kpoint_stress_threshold": document.project.recommended_kpoint_stress_threshold,
                    "recommended_kpoint_cell_id": document.project.recommended_kpoint_cell_id,
                },
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

    def stop_job(self, job: Job, message: str | None = None) -> None:
        """Mark a running or retrieving job as stopped (for example walltime reached)."""
        self._assert_transition(job.status, JobStatus.STOPPED)
        job.status = JobStatus.STOPPED
        job.completed_at = utc_now()
        job.error_message = message

    def fail_job(self, job: Job, message: str) -> None:
        """Mark a job as failed with an error message."""
        self._assert_transition(job.status, JobStatus.FAILED)
        job.status = JobStatus.FAILED
        job.completed_at = utc_now()
        job.error_message = message

    def cancel_job(self, job: Job) -> None:
        """Cancel a non-terminal job."""
        self._assert_transition(job.status, JobStatus.CANCELLED)
        job.status = JobStatus.CANCELLED
        job.completed_at = utc_now()

    def reset_failed_job(self, job: Job) -> None:
        """Reset a failed, stopped, or cancelled job to defined for resubmission."""

        if job.status not in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}:
            raise ValueError("Only failed, stopped, or cancelled jobs can be reset.")

        job.status = JobStatus.DEFINED
        job.submitted_at = None
        job.started_at = None
        job.completed_at = None
        job.remote_job_id = None
        job.remote_directory = None
        job.num_processes = None
        job.error_message = None
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

    def dismiss_job(self, job: Job) -> None:
        """Hide a terminal job from the active dashboard without deleting it.

        Dismissal is **not** a status transition: the job keeps its FAILED /
        STOPPED / CANCELLED status and its record, runs, results, and provenance
        all remain in the document. Only ``dismissed_at`` is stamped so
        ``_refresh_jobs_table`` can filter the row out of the active dashboard.
        Already-dismissed jobs are rejected so the milestone is recorded once.
        """

        if job.status not in {JobStatus.FAILED, JobStatus.STOPPED, JobStatus.CANCELLED}:
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

            self.stop_job(job, ORPHANED_LOCAL_JOB_STOP_MESSAGE)
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

        Random strategies enforce periodic muon-muon and muon-atom distance
        constraints. Element-attached generation ranks alternative starts for
        each selected host atom by local clearance from the surrounding host.
        """
        host = self._active_host_structure(document)
        host_positions = host.fractional_positions if host else []
        host_species = host.species if host else []
        lattice = host.lattice_vectors if host and host.lattice_vectors else None

        created_sites: list[MuonSite] = []
        random_selection_mode = self._random_selection_mode(parameters) if method == "random" else ""
        if method == "manual":
            requested_count: int | None = 1
        elif method == "random" and random_selection_mode == "separation":
            requested_count = None
        elif method == "uep_clusters" and isinstance(parameters.get("selections"), list):
            requested_count = len(parameters.get("selections", []))
        elif method == "near_element":
            target = str(parameters.get("element", "")).strip().lower()
            muons_per_atom_for_count = max(
                1,
                int(
                    parameters.get(
                        "muons_per_atom",
                        parameters.get("max_candidates_per_atom", parameters.get("count", 1)),
                    )
                ),
            )
            matching_atoms = sum(1 for species in host_species if str(species).strip().lower() == target)
            requested_count = matching_atoms * muons_per_atom_for_count if target and matching_atoms else None
        else:
            requested_count = int(parameters.get("count", 1))
        attempts_used = 1

        if method == "manual":
            coord_system = str(parameters.get("coord_system", "fractional"))
            coords = [float(parameters.get("x", 0.0)), float(parameters.get("y", 0.0)), float(parameters.get("z", 0.0))]
            if coord_system == "cartesian":
                fractional = lattice_cart_to_frac(coords, lattice or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                fractional = [v % 1.0 for v in fractional]
                cartesian = lattice_frac_to_cart(
                    fractional,
                    lattice or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                )
            else:
                fractional = [v % 1.0 for v in coords]
                cartesian = lattice_frac_to_cart(
                    fractional,
                    lattice or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                )

            label_prefix = "Manual Site"
            site = MuonSite(
                label=f"{label_prefix} {self._next_site_label_number(document, label_prefix)}",
                status=_GENERATED_SITE_STATUS,
                fractional_position=fractional,
                cartesian_position=cartesian,
                generation_method=method,
                source_structure_id=host.id if host else None,
                properties={"coordinate_system": coord_system},
            )
            created_sites.append(site)
        elif method == "random":
            seed = self._parse_optional_int(parameters.get("seed"), field_name="seed")
            min_muon_atom = float(parameters.get("min_muon_atom", 1.0))
            include_symmetry_equivalents = bool(parameters.get("include_symmetry_equivalents", True))
            estimated_min_muon_muon: float | None = None
            if random_selection_mode == "separation":
                min_muon_muon = float(parameters.get("min_muon_muon", 0.8))
                predicted_count = max(
                    1,
                    int(
                        estimate_random_site_count_from_separation(
                            lattice_vectors=lattice,
                            host_positions_frac=host_positions,
                            host_species=host_species,
                            min_muon_muon=min_muon_muon,
                            min_muon_atom=min_muon_atom,
                            include_symmetry_equivalents=include_symmetry_equivalents,
                        )
                        + 0.999
                    ),
                )
                points, attempts_used = generate_random_sites_by_distance(
                    lattice_vectors=lattice,
                    host_positions_frac=host_positions,
                    host_species=host_species,
                    min_muon_muon=min_muon_muon,
                    min_muon_atom=min_muon_atom,
                    max_attempts=int(parameters.get("max_attempts", 20000)),
                    seed=seed,
                    include_symmetry_equivalents=include_symmetry_equivalents,
                    expected_count=predicted_count,
                    progress_callback=progress_callback,
                )
            else:
                target_count = max(1, int(requested_count or parameters.get("count", 1)))
                estimated_min_muon_muon = estimate_muon_muon_separation_for_count(
                    lattice_vectors=lattice,
                    host_positions_frac=host_positions,
                    host_species=host_species,
                    target_count=target_count,
                    min_muon_atom=min_muon_atom,
                    include_symmetry_equivalents=include_symmetry_equivalents,
                )
                min_muon_muon = estimated_min_muon_muon
                last_error: SiteGenerationError | None = None
                for _attempt in range(6):
                    try:
                        points, attempts_used = generate_random_sites(
                            lattice_vectors=lattice,
                            host_positions_frac=host_positions,
                            host_species=host_species,
                            count=target_count,
                            min_muon_muon=min_muon_muon,
                            min_muon_atom=min_muon_atom,
                            max_attempts=int(parameters.get("max_attempts", 20000)),
                            seed=seed,
                            include_symmetry_equivalents=include_symmetry_equivalents,
                            progress_callback=progress_callback,
                        )
                        break
                    except SiteGenerationError as exc:
                        last_error = exc
                        min_muon_muon *= 0.9
                else:
                    assert last_error is not None
                    raise last_error
            label_prefix = "Random Site"
            start_index = self._next_site_label_number(document, label_prefix)
            for idx, point in enumerate(points, start=1):
                site = MuonSite(
                    label=f"{label_prefix} {start_index + idx - 1}",
                    status=_GENERATED_SITE_STATUS,
                    fractional_position=point.fractional,
                    cartesian_position=point.cartesian,
                    generation_method=method,
                    source_structure_id=host.id if host else None,
                    properties={
                        "selection_mode": random_selection_mode,
                        "min_muon_muon": min_muon_muon,
                        "min_muon_atom": min_muon_atom,
                        "include_symmetry_equivalents": include_symmetry_equivalents,
                        **({"target_count": int(requested_count or 0)} if random_selection_mode == "count" else {}),
                        **(
                            {"estimated_min_muon_muon": estimated_min_muon_muon}
                            if estimated_min_muon_muon is not None
                            else {}
                        ),
                    },
                )
                created_sites.append(site)
        elif method == "near_element":
            muons_per_atom = max(
                1,
                int(
                    parameters.get(
                        "muons_per_atom",
                        parameters.get("max_candidates_per_atom", parameters.get("count", 1)),
                    )
                ),
            )
            points, attempts_used = generate_attached_element_sites(
                lattice_vectors=lattice,
                host_species=host_species,
                host_positions_frac=host_positions,
                target_element=str(parameters.get("element", "")).strip(),
                muons_per_atom=muons_per_atom,
                bond_distance=float(parameters.get("bond_distance", 1.0)),
                min_muon_atom=float(parameters.get("min_muon_atom", 0.8)),
                progress_callback=progress_callback,
            )
            label_prefix = f"Attached-{parameters.get('element', '').strip() or 'Element'} Site"
            start_index = self._next_site_label_number(document, label_prefix)
            for idx, point in enumerate(points, start=1):
                site = MuonSite(
                    label=f"{label_prefix} {start_index + idx - 1}",
                    status=_GENERATED_SITE_STATUS,
                    fractional_position=point.fractional,
                    cartesian_position=point.cartesian,
                    generation_method=method,
                    source_structure_id=host.id if host else None,
                    properties={
                        "target_element": str(parameters.get("element", "")).strip(),
                        "strategy": "attached",
                        "muons_per_atom": muons_per_atom,
                        "bond_distance": float(parameters.get("bond_distance", 1.0)),
                        "min_muon_atom": float(parameters.get("min_muon_atom", 0.8)),
                    },
                )
                created_sites.append(site)
        elif method == "uep_clusters":
            selections = parameters.get("selections", [])
            if not isinstance(selections, list) or not selections:
                raise SiteGenerationError("Select one or more UEP cluster rows to add as generated sites.")

            skipped_duplicates = 0
            for selection in selections:
                if not isinstance(selection, dict):
                    continue
                fractional_raw = selection.get("fractional_position", selection.get("fractional", []))
                cartesian_raw = selection.get("cartesian_position", selection.get("cartesian", []))
                if (
                    not isinstance(fractional_raw, (list, tuple))
                    or len(fractional_raw) != 3
                    or not isinstance(cartesian_raw, (list, tuple))
                    or len(cartesian_raw) != 3
                ):
                    raise SiteGenerationError("Selected UEP cluster row is missing valid coordinates.")

                fractional = [float(value) % 1.0 for value in fractional_raw]
                cartesian = [float(value) for value in cartesian_raw]
                source_key = str(selection.get("source_key", "")).strip()
                if source_key and self._has_duplicate_uep_cluster_site(document, source_key, fractional):
                    skipped_duplicates += 1
                    continue

                properties = selection.get("properties", {})
                if not isinstance(properties, dict):
                    properties = {}
                method_name = str(selection.get("generation_method", "")).strip()
                if method_name not in {"uep_cluster", "uep_cluster_member"}:
                    method_name = "uep_cluster_member" if properties.get("selection_kind") == "member" else "uep_cluster"
                merged_properties = dict(properties)
                if source_key:
                    merged_properties["uep_source_key"] = source_key

                source_structure_id = None
                raw_source_structure_id = selection.get("source_structure_id")
                if raw_source_structure_id:
                    try:
                        source_structure_id = UUID(str(raw_source_structure_id))
                    except ValueError:
                        source_structure_id = host.id if host else None
                elif host is not None:
                    source_structure_id = host.id

                label_hint = str(selection.get("label", "")).strip()
                if not label_hint:
                    label_hint = "UEP Cluster Site" if method_name == "uep_cluster" else "UEP Cluster Member"
                label = (
                    label_hint
                    if str(selection.get("label", "")).strip()
                    else f"{label_hint} {len(document.sites) + len(created_sites) + 1}"
                )
                site = MuonSite(
                    label=label,
                    status=_GENERATED_SITE_STATUS,
                    fractional_position=fractional,
                    cartesian_position=cartesian,
                    generation_method=method_name,
                    source_structure_id=source_structure_id,
                    properties=merged_properties,
                )
                created_sites.append(site)

            if not created_sites and skipped_duplicates:
                raise SiteGenerationError("Selected UEP cluster rows are already present in the generated-site list.")
            if not created_sites:
                raise SiteGenerationError("Selected UEP cluster rows did not contain any usable generated sites.")
        else:
            raise SiteGenerationError(f"Unknown generation method '{method}'.")

        for site in created_sites:
            document.sites[site.id] = site

        warning: str | None = None
        if requested_count is not None and method != "manual" and len(created_sites) < requested_count:
            reason = (
                "Some selected UEP rows were already present."
                if method == "uep_clusters"
                else "Increase max attempts, reduce distance limits, or use a larger supercell."
            )
            warning = f"Requested {requested_count} sites but generated {len(created_sites)}. {reason}"

        record = ProvenanceRecord(
            action="generate_candidates",
            workflow_stage=WorkflowStage.MUON_GENERATION,
            actor="user",
            description=f"Generated {len(created_sites)} site(s) via {method}",
            parameters={"method": method, **parameters},
            output_entity_ids=[site.id for site in created_sites],
        )
        self.append_provenance(document, record)

        if warning:
            self.append_provenance(
                document,
                ProvenanceRecord(
                    action="generation_partial",
                    workflow_stage=WorkflowStage.MUON_GENERATION,
                    actor="system",
                    description=warning,
                    parameters={"requested": requested_count, "generated": len(created_sites), "attempts": attempts_used},
                    output_entity_ids=[site.id for site in created_sites],
                ),
            )
        return created_sites

    @staticmethod
    def _has_duplicate_uep_cluster_site(document: ProjectDocument, source_key: str, fractional: Sequence[float]) -> bool:
        for site in document.sites.values():
            if site.properties.get("uep_source_key") == source_key:
                return True
            if site.generation_method not in {"uep_cluster", "uep_cluster_member", "uep_batch_cluster"}:
                continue
            if not site.fractional_position or len(site.fractional_position) != 3:
                continue
            if all(abs((float(a) - float(b) + 0.5) % 1.0 - 0.5) <= 1e-6 for a, b in zip(site.fractional_position, fractional)):
                return True
        return False

    @staticmethod
    def _next_site_label_number(document: ProjectDocument, prefix: str) -> int:
        pattern = re.compile(rf"^{re.escape(prefix.strip())}\s+(\d+)$")
        used_numbers: list[int] = []
        for site in document.sites.values():
            match = pattern.match(str(site.label).strip())
            if match is not None:
                used_numbers.append(int(match.group(1)))
        return max(used_numbers, default=0) + 1

    @staticmethod
    def _random_selection_mode(parameters: dict[str, Any]) -> str:
        mode = str(parameters.get("selection_mode", parameters.get("mode", ""))).strip().lower()
        if mode in {"separation", "distance", "spacing"}:
            return "separation"
        if mode in {"count", "number", "target_count"}:
            return "count"
        return "count" if "count" in parameters else "separation"

    def _active_host_structure(self, document: ProjectDocument) -> Structure | None:
        """Return the currently selected host structure if available."""
        host_id = document.project.host_structure_id
        if host_id is None:
            return None
        return document.structures.get(host_id)

    def active_host_structure(self, document: ProjectDocument) -> Structure | None:
        """Return the current host structure, if one is selected."""
        return self._active_host_structure(document)

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
            if run is None or run.status != JobStatus.COMPLETED or run.convergence_achieved is False:
                continue
            for struct_id in result.derived_structure_ids:
                completed_ids.add(struct_id)
                result_created_at[struct_id] = result.created_at

        promotable: list[PromotableHost] = []
        for structure in document.structures.values():
            if self._supercell_asset(structure) is not None:
                continue
            if structure.origin == "imported":
                hint = f"Imported from {structure.source_file}" if structure.source_file else "Imported structure"
                promotable.append(PromotableHost(
                    structure_id=structure.id,
                    label=f"Original — {structure.label}",
                    provenance_hint=hint,
                    kind="imported",
                    created_at=None,
                ))
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
                promotable.append(PromotableHost(
                    structure_id=structure.id,
                    label=label,
                    provenance_hint=hint,
                    kind="relaxed",
                    created_at=created_at,
                ))

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
        host = self._active_host_structure(document)
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
        host = self._active_host_structure(document)
        if host is None:
            return None
        offenders: list[tuple[UUID, str, UUID | None, str]] = []
        for struct_id in structure_ids:
            structure = document.structures.get(struct_id)
            if structure is None:
                continue
            resolved_host_id = self.resolve_underlying_host(document, struct_id)
            if resolved_host_id != host.id:
                resolved_host = document.structures.get(resolved_host_id) if resolved_host_id else None
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

    def _parse_optional_int(self, value: Any, *, field_name: str) -> int | None:
        """Parse optional integer-like input from UI payloads."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError as exc:
            raise SiteGenerationError(f"{field_name} must be an integer if provided.") from exc

    def validate_document(self, document: ProjectDocument) -> list[str]:
        """Validate cross-entity references and return detected issues."""
        issues: list[str] = []

        if document.project.host_structure_id and document.project.host_structure_id not in document.structures:
            issues.append("project.host_structure_id does not reference a known structure")

        for structure in document.structures.values():
            if structure.parent_structure_id and structure.parent_structure_id not in document.structures:
                issues.append(
                    f"structure {structure.id} references missing parent structure {structure.parent_structure_id}"
                )
            asset = self._supercell_asset(structure)
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
            missing_structures = [sid for sid in calc.input_structure_ids if sid not in document.structures]
            if missing_structures:
                issues.append(
                    f"calculation {calc.id} references missing structures: {', '.join(str(s) for s in missing_structures)}"
                )

        for run in document.runs.values():
            if run.definition_id and run.definition_id not in document.calculations:
                issues.append(f"run {run.id} references missing calculation definition {run.definition_id}")
            if run.job_id and run.job_id not in document.jobs:
                issues.append(f"run {run.id} references missing job {run.job_id}")
            if run.result_id and run.result_id not in document.results:
                issues.append(f"run {run.id} references missing result {run.result_id}")

        for result in document.results.values():
            if result.calculation_run_id and result.calculation_run_id not in document.runs:
                issues.append(
                    f"result {result.id} references missing calculation run {result.calculation_run_id}"
                )
            missing_structures = [sid for sid in result.derived_structure_ids if sid not in document.structures]
            if missing_structures:
                issues.append(
                    f"result {result.id} references missing derived structures: {', '.join(str(s) for s in missing_structures)}"
                )

        for site in document.sites.values():
            if site.source_structure_id and site.source_structure_id not in document.structures:
                issues.append(f"site {site.id} references missing structure {site.source_structure_id}")
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
            missing_structures = [sid for sid in table.dft_structure_ids if sid not in document.structures]
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
            JobStatus.STOPPED: {
                JobStatus.SUBMITTING,
                JobStatus.RETRIEVING,
                JobStatus.CANCELLED,
            },
            JobStatus.COMPLETED: set(),
            JobStatus.FAILED: {
                JobStatus.SUBMITTING,
                JobStatus.RETRIEVING,
                JobStatus.CANCELLED,
            },
            JobStatus.CANCELLED: {JobStatus.SUBMITTING},
        }
        if target not in allowed[current]:
            raise ValueError(f"Invalid job transition: {current.value} -> {target.value}")

    def _project_from_dict(self, payload: dict[str, Any]) -> Project:
        raw_recommended_cutoff = payload.get("recommended_cutoff_energy")
        try:
            recommended_cutoff = float(raw_recommended_cutoff) if raw_recommended_cutoff is not None else None
        except (TypeError, ValueError):
            recommended_cutoff = None

        raw_recommended_method = payload.get("recommended_cutoff_method")
        recommended_method = None
        if isinstance(raw_recommended_method, str) and raw_recommended_method.strip():
            recommended_method = raw_recommended_method.strip()

        raw_energy_threshold = payload.get("recommended_cutoff_energy_threshold")
        try:
            recommended_energy_threshold = (
                float(raw_energy_threshold) if raw_energy_threshold is not None else None
            )
        except (TypeError, ValueError):
            recommended_energy_threshold = None

        raw_force_threshold = payload.get("recommended_cutoff_force_threshold")
        try:
            recommended_force_threshold = (
                float(raw_force_threshold) if raw_force_threshold is not None else None
            )
        except (TypeError, ValueError):
            recommended_force_threshold = None

        raw_stress_threshold = payload.get("recommended_cutoff_stress_threshold")
        try:
            recommended_stress_threshold = (
                float(raw_stress_threshold) if raw_stress_threshold is not None else None
            )
        except (TypeError, ValueError):
            recommended_stress_threshold = None

        raw_recommended_cutoff_cell_id = payload.get("recommended_cutoff_cell_id")
        recommended_cutoff_cell_id = None
        if isinstance(raw_recommended_cutoff_cell_id, str) and raw_recommended_cutoff_cell_id.strip():
            recommended_cutoff_cell_id = raw_recommended_cutoff_cell_id.strip()

        raw_recommended_kpoint_grid = payload.get("recommended_kpoint_grid")
        recommended_kpoint_grid: tuple[int, int, int] | None = None
        if isinstance(raw_recommended_kpoint_grid, list) and len(raw_recommended_kpoint_grid) == 3:
            try:
                recommended_kpoint_grid = (
                    int(raw_recommended_kpoint_grid[0]),
                    int(raw_recommended_kpoint_grid[1]),
                    int(raw_recommended_kpoint_grid[2]),
                )
            except (TypeError, ValueError):
                recommended_kpoint_grid = None

        raw_recommended_kpoint_lattice = payload.get("recommended_kpoint_lattice")
        recommended_kpoint_lattice: list[list[float]] | None = None
        if isinstance(raw_recommended_kpoint_lattice, list) and len(raw_recommended_kpoint_lattice) == 3:
            parsed_lattice: list[list[float]] = []
            valid_lattice = True
            for row in raw_recommended_kpoint_lattice:
                if not isinstance(row, list) or len(row) != 3:
                    valid_lattice = False
                    break
                try:
                    parsed_lattice.append([float(row[0]), float(row[1]), float(row[2])])
                except (TypeError, ValueError):
                    valid_lattice = False
                    break
            if valid_lattice:
                recommended_kpoint_lattice = parsed_lattice

        raw_recommended_kpoint_method = payload.get("recommended_kpoint_method")
        recommended_kpoint_method = None
        if isinstance(raw_recommended_kpoint_method, str) and raw_recommended_kpoint_method.strip():
            recommended_kpoint_method = raw_recommended_kpoint_method.strip()

        raw_kpoint_energy_threshold = payload.get("recommended_kpoint_energy_threshold")
        try:
            recommended_kpoint_energy_threshold = (
                float(raw_kpoint_energy_threshold) if raw_kpoint_energy_threshold is not None else None
            )
        except (TypeError, ValueError):
            recommended_kpoint_energy_threshold = None

        raw_kpoint_force_threshold = payload.get("recommended_kpoint_force_threshold")
        try:
            recommended_kpoint_force_threshold = (
                float(raw_kpoint_force_threshold) if raw_kpoint_force_threshold is not None else None
            )
        except (TypeError, ValueError):
            recommended_kpoint_force_threshold = None

        raw_kpoint_stress_threshold = payload.get("recommended_kpoint_stress_threshold")
        try:
            recommended_kpoint_stress_threshold = (
                float(raw_kpoint_stress_threshold) if raw_kpoint_stress_threshold is not None else None
            )
        except (TypeError, ValueError):
            recommended_kpoint_stress_threshold = None

        raw_recommended_kpoint_cell_id = payload.get("recommended_kpoint_cell_id")
        recommended_kpoint_cell_id = None
        if isinstance(raw_recommended_kpoint_cell_id, str) and raw_recommended_kpoint_cell_id.strip():
            recommended_kpoint_cell_id = raw_recommended_kpoint_cell_id.strip()

        return Project(
            id=UUID(payload["id"]),
            name=payload["name"],
            schema_version=payload["schema_version"],
            app_version=payload.get("app_version", current_version()),
            created_at=parse_datetime(payload["created_at"]) or utc_now(),
            modified_at=parse_datetime(payload["modified_at"]) or utc_now(),
            host_structure_id=parse_uuid(payload.get("host_structure_id")),
            magnetic_structure=payload.get("magnetic_structure"),
            supercell_matrix=payload.get("supercell_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            recommended_cutoff_energy=recommended_cutoff,
            recommended_cutoff_method=recommended_method,
            recommended_cutoff_energy_threshold=recommended_energy_threshold,
            recommended_cutoff_force_threshold=recommended_force_threshold,
            recommended_cutoff_stress_threshold=recommended_stress_threshold,
            recommended_cutoff_cell_id=recommended_cutoff_cell_id,
            recommended_kpoint_grid=recommended_kpoint_grid,
            recommended_kpoint_lattice=recommended_kpoint_lattice,
            recommended_kpoint_method=recommended_kpoint_method,
            recommended_kpoint_energy_threshold=recommended_kpoint_energy_threshold,
            recommended_kpoint_force_threshold=recommended_kpoint_force_threshold,
            recommended_kpoint_stress_threshold=recommended_kpoint_stress_threshold,
            recommended_kpoint_cell_id=recommended_kpoint_cell_id,
            notes=payload.get("notes", ""),
        )

    def _structure_from_dict(self, payload: dict[str, Any]) -> Structure:
        return Structure(
            id=UUID(payload["id"]),
            label=payload.get("label", ""),
            origin=payload.get("origin", "imported"),
            lattice_vectors=payload.get("lattice_vectors", []),
            species=payload.get("species", []),
            fractional_positions=payload.get("fractional_positions", []),
            source_file=payload.get("source_file"),
            parent_structure_id=parse_uuid(payload.get("parent_structure_id")),
            properties=payload.get("properties", {}),
        )

    def _calculation_from_dict(self, payload: dict[str, Any]) -> CalculationDefinition:
        return CalculationDefinition(
            id=UUID(payload["id"]),
            workflow_stage=WorkflowStage(payload["workflow_stage"]),
            calc_type=payload.get("calc_type", ""),
            parameters=payload.get("parameters", {}),
            input_structure_ids=[UUID(v) for v in payload.get("input_structure_ids", [])],
            input_site_ids=[UUID(v) for v in payload.get("input_site_ids", [])],
            custom_param_text=payload.get("custom_param_text"),
            created_at=parse_datetime(payload.get("created_at")) or utc_now(),
        )

    def _run_from_dict(self, payload: dict[str, Any]) -> CalculationRun:
        return CalculationRun(
            id=UUID(payload["id"]),
            definition_id=parse_uuid(payload.get("definition_id")),
            job_id=parse_uuid(payload.get("job_id")),
            run_index=int(payload.get("run_index", 1)),
            status=JobStatus(payload.get("status", JobStatus.DEFINED.value)),
            started_at=parse_datetime(payload.get("started_at")),
            completed_at=parse_datetime(payload.get("completed_at")),
            result_id=parse_uuid(payload.get("result_id")),
            warnings=payload.get("warnings", []),
            convergence_achieved=payload.get("convergence_achieved"),
        )

    def _result_from_dict(self, payload: dict[str, Any]) -> Result:
        return Result(
            id=UUID(payload["id"]),
            calculation_run_id=parse_uuid(payload.get("calculation_run_id")),
            result_type=payload.get("result_type", ""),
            data=payload.get("data", {}),
            output_files=payload.get("output_files", []),
            derived_structure_ids=[UUID(v) for v in payload.get("derived_structure_ids", [])],
            created_at=parse_datetime(payload.get("created_at")) or utc_now(),
        )

    def _comparison_table_from_dict(self, payload: dict[str, Any]) -> ComparisonTable:
        return ComparisonTable(
            id=UUID(payload["id"]),
            label=payload.get("label", ""),
            experimental_structure_id=parse_uuid(payload.get("experimental_structure_id")),
            dft_structure_ids=[UUID(v) for v in payload.get("dft_structure_ids", [])],
            dft_labels=[str(label) for label in payload.get("dft_labels", [])],
            result_data=payload.get("result_data", {}),
            created_at=parse_datetime(payload.get("created_at")) or utc_now(),
        )

    def _job_from_dict(self, payload: dict[str, Any]) -> Job:
        raw_num_processes = payload.get("num_processes")
        num_processes = None
        if raw_num_processes is not None:
            try:
                parsed_num_processes = int(raw_num_processes)
            except (TypeError, ValueError):
                parsed_num_processes = 0
            if parsed_num_processes > 0:
                num_processes = parsed_num_processes
        return Job(
            id=UUID(payload["id"]),
            label=payload.get("label", ""),
            workflow_stage=WorkflowStage(payload["workflow_stage"]),
            calc_type=payload.get("calc_type", ""),
            status=JobStatus(payload["status"]),
            execution_target=payload.get("execution_target", "local"),
            submitted_at=parse_datetime(payload.get("submitted_at")),
            started_at=parse_datetime(payload.get("started_at")),
            completed_at=parse_datetime(payload.get("completed_at")),
            remote_job_id=payload.get("remote_job_id"),
            remote_directory=payload.get("remote_directory"),
            num_processes=num_processes,
            tags=payload.get("tags", []),
            error_message=payload.get("error_message"),
            last_checked_at=parse_datetime(payload.get("last_checked_at")),
            last_status_source=payload.get("last_status_source"),
            last_status_message=payload.get("last_status_message"),
            dismissed_at=parse_datetime(payload.get("dismissed_at")),
        )

    def _site_from_dict(self, payload: dict[str, Any]) -> MuonSite:
        return MuonSite(
            id=UUID(payload["id"]),
            label=payload.get("label", ""),
            status=self._site_status_from_payload(payload),
            fractional_position=payload.get("fractional_position", [0.0, 0.0, 0.0]),
            cartesian_position=payload.get("cartesian_position", [0.0, 0.0, 0.0]),
            generation_method=payload.get("generation_method"),
            source_generation_batch_id=parse_uuid(payload.get("source_generation_batch_id")),
            source_structure_id=parse_uuid(payload.get("source_structure_id")),
            relaxation_run_id=parse_uuid(payload.get("relaxation_run_id")),
            relative_energy_eV=payload.get("relative_energy_eV"),
            host_displacement_max=payload.get("host_displacement_max"),
            cluster_id=parse_uuid(payload.get("cluster_id")),
            symmetry_label=payload.get("symmetry_label"),
            properties=payload.get("properties", {}),
        )

    @staticmethod
    def _site_status_from_payload(payload: dict[str, Any]) -> str:
        status = str(payload.get("status", _GENERATED_SITE_STATUS) or _GENERATED_SITE_STATUS)
        if status == "candidate":
            return _GENERATED_SITE_STATUS
        return status

    def _provenance_from_dict(self, payload: dict[str, Any]) -> ProvenanceRecord:
        workflow_stage = payload.get("workflow_stage")
        return ProvenanceRecord(
            id=UUID(payload["id"]),
            timestamp=parse_datetime(payload.get("timestamp")) or utc_now(),
            action=payload.get("action", ""),
            workflow_stage=WorkflowStage(workflow_stage) if workflow_stage else None,
            actor=payload.get("actor", "system"),
            description=payload.get("description", ""),
            parameters=payload.get("parameters", {}),
            input_entity_ids=[UUID(v) for v in payload.get("input_entity_ids", [])],
            output_entity_ids=[UUID(v) for v in payload.get("output_entity_ids", [])],
            job_id=parse_uuid(payload.get("job_id")),
        )
