"""Pristine-relaxation workflow controller.

Owns the submit → materialize → result → refresh cycle for the CASTEP pristine
structural relaxation. This is the first migration onto
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController` that gives
:meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.build_result`
its first genuine implementation: a completed ``castep_pristine_relaxation`` job
contributes the relaxed lattice parameters to the result payload and creates the
derived relaxed :class:`~mufinder.domain.models.Structure` on the document.

The logic here is a behavior-preserving move of the former ``MainWindow`` methods
``_materialize_pristine_relaxation_jobs``, ``_build_pristine_relaxation_states``,
``_jobs_for_pristine_relaxation_labels``, ``_refresh_pristine_relaxation_rows``,
and the ``castep_pristine_relaxation`` portion of ``_on_local_job_finished``'s
result block. The muon-relaxation branch of that result block does **not** migrate
in this PR and stays inline on ``MainWindow``.

Shared helpers that non-migrating code still calls (including the muon-relaxation
branch, which reuses the same ``-out.cell`` parse and the same
``pristine_relaxation``/``muon_relaxation`` definition/job lookups) remain on
``MainWindow`` and are reached via :class:`PristineRelaxationDeps` — the
per-controller deps dataclass this module defines, following the
shared-context + per-controller-deps + view pattern documented in
:mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
    Structure,
)
from mufinder.workflow import StructureBridgeError, write_job_directory

from mufinder.app.adapter_common import AdapterExpansion
from mufinder.app.relaxation_types import PristineRelaxationConfig, RelaxationRunState
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
)


@dataclass(frozen=True)
class PristineRelaxationDeps:
    """Pristine-relaxation-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating code still calls
    them there. The controller reaches them only through these required
    callables. Workflow-specific dependencies like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`;
    cross-workflow job/definition readers (``job_output_directory``,
    ``job_seed_name``, ``run_for_job``, ``definition_output_directory``, the
    energy/force readers) live there instead. The relaxed ``-out.cell`` is parsed
    once on the runner worker thread and delivered to ``build_result`` via
    :class:`~mufinder.services.job_artifacts.ParsedJobArtifacts`, so no parse dep
    is needed here.
    """

    active_pristine_relaxation_definition: Callable[[], CalculationDefinition | None]
    deserialize_pristine_relaxation_config: Callable[
        [object], PristineRelaxationConfig | None
    ]
    sync_equivalent_pristine_relaxation_mappings: Callable[
        [CalculationDefinition], None
    ]
    pristine_relaxation_form_config: Callable[[], PristineRelaxationConfig | None]
    pristine_relaxation_row_signature: Callable[[object], tuple[object, ...]]
    job_for_pristine_relaxation_label: Callable[
        [CalculationDefinition, str], Job | None
    ]
    expand_pristine_relaxation_jobs: Callable[
        [CalculationDefinition, PristineRelaxationConfig], AdapterExpansion | None
    ]


class PristineRelaxationController(WorkflowJobController):
    """Controller for the pristine structural-relaxation workflow."""

    calc_types = frozenset({"castep_pristine_relaxation"})
    no_ready_message = (
        "Only configured pristine-relaxation rows that have not yet been submitted can be submitted."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: PristineRelaxationDeps,
        view: PristineRelaxationWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready)
    # exactly mirror the former ``_on_submit_*_from_pristine_relaxation`` handlers.
    # There is no pre_submit_check and submission is a single group.

    def normalize(self, value: object) -> str:
        return str(value)

    def build_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        document = self._context.document()
        if document is None:
            return states

        form_config = self._deps.pristine_relaxation_form_config()
        if form_config is not None:
            for row in form_config.rows:
                states[row.label] = RelaxationRunState(status="ready")

        definition = self._deps.active_pristine_relaxation_definition()
        if definition is None:
            return states
        self._deps.sync_equivalent_pristine_relaxation_mappings(definition)
        config = self._deps.deserialize_pristine_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return states

        if form_config is not None:
            definition_rows_by_label = {row.label: row for row in config.rows}
            rows_for_status = [
                definition_row
                for row in form_config.rows
                if (definition_row := definition_rows_by_label.get(row.label)) is not None
                and self._deps.pristine_relaxation_row_signature(row)
                == self._deps.pristine_relaxation_row_signature(definition_row)
            ]
        else:
            rows_for_status = list(config.rows)

        for row in rows_for_status:
            job = self._deps.job_for_pristine_relaxation_label(definition, row.label)
            if job is None:
                states[row.label] = RelaxationRunState(status="ready")
                continue

            status_value = "ready" if job.status == JobStatus.DEFINED else job.status.value
            run = self._context.run_for_job(job.id)
            if (
                status_value == JobStatus.COMPLETED.value
                and run is not None
                and run.convergence_achieved is False
            ):
                status_value = "not converged"

            lattice_a = None
            lattice_b = None
            lattice_c = None
            if run is not None and run.result_id is not None:
                result = document.results.get(run.result_id)
                if result is not None and isinstance(result.data, dict):
                    lattice_a = result.data.get("lattice_a")
                    lattice_b = result.data.get("lattice_b")
                    lattice_c = result.data.get("lattice_c")

            states[row.label] = RelaxationRunState(
                status=status_value,
                total_energy=self._context.total_energy_for_job(job.id),
                max_force=self._context.max_force_for_job(job.id),
                lattice_a=float(lattice_a) if isinstance(lattice_a, (int, float)) else None,
                lattice_b=float(lattice_b) if isinstance(lattice_b, (int, float)) else None,
                lattice_c=float(lattice_c) if isinstance(lattice_c, (int, float)) else None,
                job_id=job.id,
                job_label=job.label,
                geom_converged=(run.convergence_achieved if run is not None else None),
                error_message=job.error_message,
            )

        return states

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        definition = self._deps.active_pristine_relaxation_definition()
        if definition is None:
            return []
        self._deps.sync_equivalent_pristine_relaxation_mappings(definition)

        selected: list[Job] = []
        seen: set[UUID] = set()
        for label in keys:
            job = self._deps.job_for_pristine_relaxation_label(definition, str(label))
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def materialize(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []

        definition = self._deps.active_pristine_relaxation_definition()
        if definition is None:
            return []
        self._deps.sync_equivalent_pristine_relaxation_mappings(definition)

        config = self._deps.deserialize_pristine_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return []
        if not definition.input_structure_ids:
            return []

        output_root = self._context.definition_output_directory(definition)
        if output_root is None:
            return []

        try:
            expansion = self._deps.expand_pristine_relaxation_jobs(definition, config)
        except (ValueError, StructureBridgeError):
            return []
        if expansion is None:
            return []

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
            existing_job = self._deps.job_for_pristine_relaxation_label(definition, label)
            if existing_job is not None:
                if existing_job.status == JobStatus.DEFINED and spec is not None:
                    write_job_directory(spec, output_root)
                created.append(existing_job)
                continue

            if spec is None:
                continue

            write_job_directory(spec, output_root)
            job = self._context.project_service.add_job(
                document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type="castep_pristine_relaxation",
            )
            mapping[label] = str(job.id)
            created.append(job)

        deduped: list[Job] = []
        seen: set[UUID] = set()
        for job in created:
            if job.id in seen:
                continue
            seen.add(job.id)
            deduped.append(job)
        return deduped

    # --- result contract ---

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute the relaxed lattice parameters and derived structure.

        The inline ``_on_local_job_finished`` prologue has already validated that
        the ``-out.cell`` exists and parses (it aborts the job otherwise, shared
        with the muon branch), so the carried ``artifacts.relaxed_cell`` is
        present on this success path. The file is parsed **once** on the runner
        worker thread; this controller consumes the carried cell rather than
        re-reading disk.

        Output files are **not** contributed here: the ``_on_local_job_finished``
        call site already passes ``_output_files_for_job(job)`` for every job, and
        that list already includes ``{seed_name}-out.cell``. Adding them again
        would duplicate entries.
        """

        document = self._context.document()
        if document is None:
            return ControllerResult()

        relaxed = artifacts.relaxed_cell
        if relaxed is None or artifacts.lattice_abc is None:
            return ControllerResult()
        lattice_a, lattice_b, lattice_c = artifacts.lattice_abc

        payload: dict[str, object] = {
            "lattice_a": lattice_a,
            "lattice_b": lattice_b,
            "lattice_c": lattice_c,
        }

        definition = self._deps.active_pristine_relaxation_definition()
        parent_structure_id = (
            definition.input_structure_ids[0]
            if definition is not None and definition.input_structure_ids
            else None
        )
        relaxed_structure = Structure(
            label=f"{job.label} relaxed",
            origin="pristine_relaxation",
            lattice_vectors=[list(vector) for vector in relaxed.lattice_vectors],
            species=list(relaxed.species),
            fractional_positions=[list(pos) for pos in relaxed.fractional_positions],
            parent_structure_id=parent_structure_id,
            properties={
                "source_job_id": str(job.id),
                "source_workflow_stage": WorkflowStage.PRISTINE_RELAXATION.value,
            },
        )
        self._context.project_service.add_structure(document, relaxed_structure)

        return ControllerResult(
            payload=payload,
            derived_structure_ids=[relaxed_structure.id],
        )

    # --- view contract ---

    def refresh_view(self) -> None:
        self._view.set_runtime_states(self.build_states())


__all__ = ["PristineRelaxationController", "PristineRelaxationDeps"]
