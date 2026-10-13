"""UEP-source workflow controller.

Owns the submit → materialize → result → refresh cycle for the CASTEP UEP
(unperturbed electrostatic potential) source jobs. This is one of the simpler
migrations onto
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`: there is
no ``pre_submit_check`` and no ``submission_groups`` override (a single group,
``normalize=str``, and the base-default ``is_ready`` exactly mirror the former
``_on_submit_*_from_uep`` handlers), and
:meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.build_result`
contributes a single boolean payload flag (``density_available``) — no derived
structures and no document mutation.

The logic here is a behavior-preserving move of the former ``MainWindow`` methods
``_on_submit_local_from_uep`` / ``_on_submit_cluster_from_uep`` (submission),
``_materialize_uep_jobs``, ``_jobs_for_uep_labels``, ``_build_uep_states``, and
``_refresh_uep_rows``, plus the ``density_available`` *flag assignment* from the
``castep_uep_source`` branch of ``_on_local_job_finished``'s result block.

Note on the den_fmt check: :meth:`UepController.validate_artifacts` fails a job
whose ``{seed}.den_fmt`` output is missing. The density file is not parsed on
completion, so this is a plain existence check rather than a carrier read.
:meth:`UepController.build_result` re-checks the file on the validated success
path, where it is guaranteed to exist, so the flag is ``True``. The den_fmt file
itself is **not** added to ``output_files`` here — the call site already passes
``_output_files_for_job(job)``, which includes it.

Shared helpers that non-migrating code still calls (row removal, the analysis
wizard, dashboard sync, cluster-source rows) remain on ``MainWindow`` and are
reached via :class:`UepDeps`, following the shared-context + per-controller-deps
+ view pattern documented in
:mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result
from mufinder.services.definition_mappings import (
    convergence_mapping_job_priority,
    remove_labels_from_definition,
    sync_equivalent_mappings,
)
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.workflow import StructureBridgeError, write_job_directory

from mufinder.app.uep_adapter import expand_uep_jobs
from mufinder.app.uep_types import UepConfig, UepRunState, UepSourceRow
from mufinder.app.uep_widget import UEPWidget
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    dedupe_jobs,
)
from mufinder.app.workflow_submission import effective_row_status


@dataclass(frozen=True)
class UepDeps:
    """UEP-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating code still calls
    them there: the configuration codec and ``uep_row_signature`` are shared
    with the UEP configure path, the analysis wizard, and dashboard
    cluster-source sync; and ``uep_potential_grid_result_for_job`` still feeds
    the analysis wizard's "already analysed" detection. The controller reaches
    them only through these required callables. Workflow-specific dependencies
    like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`;
    cross-workflow job/definition readers (``definition_output_directory``,
    ``job_output_directory``, ``job_seed_name``, ``run_for_job``) live there
    instead.
    """

    deserialize_uep_config: Callable[[object], UepConfig | None]
    serialize_uep_config: Callable[[UepConfig], dict[str, object]]
    uep_row_signature: Callable[[UepSourceRow], tuple[object, ...]]
    uep_potential_grid_result_for_job: Callable[[UUID], Result | None]


class UepController(WorkflowJobController):
    """Controller for the UEP source-job workflow."""

    calc_types = frozenset({"castep_uep_source"})
    stage = WorkflowStage.UEP
    active_key = "active_uep_definition_id"
    no_ready_message = (
        "Only configured UEP source rows that have not yet been submitted can be submitted."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: UepDeps,
        view: UEPWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- definition maintenance ---

    def sync_mappings(self, definition: CalculationDefinition) -> None:
        """Copy job bindings into ``definition`` from equivalent UEP definitions.

        A row carries over when its signature is unchanged; candidates rank by
        :func:`~mufinder.services.definition_mappings.convergence_mapping_job_priority`.
        """

        document = self._context.document()
        if document is None:
            return
        sync_equivalent_mappings(
            document,
            self.definitions(),
            definition,
            deserialize=self._deps.deserialize_uep_config,
            signatures=lambda config: {
                row.label: self._deps.uep_row_signature(row) for row in config.rows
            },
            priority=convergence_mapping_job_priority,
        )

    def remove_labels_from_definition(
        self, definition: CalculationDefinition, labels: Collection[str]
    ) -> bool:
        document = self._context.document()
        if document is None:
            return False
        return remove_labels_from_definition(
            document,
            definition,
            labels,
            deserialize=self._deps.deserialize_uep_config,
            serialize=self._deps.serialize_uep_config,
            active_key=self.active_key,
        )

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready)
    # exactly mirror the former ``_on_submit_*_from_uep`` handlers. There is no
    # pre_submit_check and submission is a single group.

    def normalize(self, value: object) -> str:
        return str(value)

    def build_states(self) -> dict[str, UepRunState]:
        states: dict[str, UepRunState] = {}
        document = self._context.document()
        if document is None:
            return states
        definition = self.active_definition()
        if definition is None:
            return states
        self.sync_mappings(definition)
        config = self._deps.deserialize_uep_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return states

        for row in config.rows:
            job = self.job_for_key(definition, row.label)
            if job is None:
                states[row.label] = UepRunState(status="ready")
                continue
            run = self._context.run_for_job(job.id)
            result = (
                document.results.get(run.result_id)
                if run is not None and run.result_id is not None
                else None
            )
            data = result.data if result is not None and isinstance(result.data, dict) else {}
            potential_grid_available = (
                self._deps.uep_potential_grid_result_for_job(job.id) is not None
            )
            states[row.label] = UepRunState(
                status=(
                    "ready" if job.status == JobStatus.DEFINED else effective_row_status(job)
                ),
                total_energy=float(data["total_energy"])
                if isinstance(data.get("total_energy"), (int, float))
                else None,
                potential_grid_available=potential_grid_available,
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
                density_available=bool(data.get("density_available"))
                if data.get("density_available") is not None
                else None,
            )
        return states

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        definition = self.active_definition()
        if definition is None:
            return []
        self.sync_mappings(definition)
        selected: list[Job] = []
        seen: set[UUID] = set()
        for label in keys:
            job = self.job_for_key(definition, str(label))
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def materialize(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        definition = self.active_definition()
        if definition is None or not definition.input_structure_ids:
            return []
        self.sync_mappings(definition)
        config = self._deps.deserialize_uep_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return []
        source_structure = document.structures.get(definition.input_structure_ids[0])
        output_root = self._context.definition_output_directory(definition)
        if source_structure is None or output_root is None:
            return []

        try:
            expansion = expand_uep_jobs(
                host_structure=source_structure,
                config=config,
                project_cutoff_energy=document.project.recommended_cutoff_energy,
                project_kpoint_grid=document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=document.project.recommended_kpoint_lattice,
                project_xc_functional=document.project.recommended_xc_functional,
            )
        except (ValueError, StructureBridgeError):
            return []

        # The configure path already reported ``expansion.kpoint_adjustments``;
        # materialising does not report them again.

        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping

        spec_by_label = {
            str(spec.provenance.get("label", "")): spec for spec in expansion.jobs
        }
        created: list[Job] = []
        for label in keys:
            label = str(label)
            spec = spec_by_label.get(label)
            existing = self.job_for_key(definition, label)
            if existing is not None:
                if existing.status == JobStatus.DEFINED and spec is not None:
                    write_job_directory(spec, output_root)
                created.append(existing)
                continue
            if spec is None:
                continue
            write_job_directory(spec, output_root)
            job = self._context.project_service.add_job(
                document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type="castep_uep_source",
            )
            mapping[label] = str(job.id)
            created.append(job)

        return dedupe_jobs(created)

    # --- result contract ---

    def validate_artifacts(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> str | None:
        """Fail a UEP source job whose ``{seed}.den_fmt`` was not written.

        The formatted density is not parsed on completion, so it has no carrier
        field; this is a plain existence check in the job's output directory.
        """

        output_dir = self._context.job_output_directory(job)
        seed_name = self._context.job_seed_name(job)
        density_path = None if output_dir is None else output_dir / f"{seed_name}.den_fmt"
        if density_path is None or not density_path.exists():
            return f"Expected formatted density output was not found: {seed_name}.den_fmt"
        return None

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute the ``density_available`` flag for a completed UEP job.

        :meth:`validate_artifacts` has already failed the job when
        ``{seed}.den_fmt`` is missing, so on the validated success path this
        re-check finds the file present and the flag is ``True``. The den_fmt
        file is **not** contributed to ``output_files`` here: the call site
        already passes ``_output_files_for_job(job)``, which includes it.
        """

        output_dir = self._context.job_output_directory(job)
        seed_name = self._context.job_seed_name(job)
        density_available = (
            output_dir is not None and (output_dir / f"{seed_name}.den_fmt").exists()
        )
        return ControllerResult(payload={"density_available": density_available})

    # --- view contract ---

    def refresh_view(self) -> None:
        self._view.set_runtime_states(self.build_states())


__all__ = ["UepController", "UepDeps"]
