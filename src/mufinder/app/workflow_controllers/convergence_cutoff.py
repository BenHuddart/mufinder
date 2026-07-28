"""Cutoff-convergence workflow controller.

Owns the submit → materialize → refresh cycle for the CASTEP cutoff-energy
convergence sweep. This is the first workflow migrated onto
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`; the logic
here is a behavior-preserving move of the former ``MainWindow`` methods
``_materialize_convergence_jobs``, ``_build_convergence_cutoff_states``,
``_jobs_for_convergence_cutoffs``, ``_refresh_convergence_cutoff_rows``, and
``_convergence_form_snapshot``. Shared convergence helpers that non-migrating
code still calls remain on ``MainWindow`` and are reached via
:class:`ConvergenceCutoffDeps` — the per-controller deps dataclass this module
defines, following the shared-context + per-controller-deps + view pattern
documented in :mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationDefinition, Job
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory

from mufinder.app.convergence_adapter import convergence_job_name, expand_convergence_jobs
from mufinder.app.convergence_types import ConvergenceConfig, SweepPointRunState
from mufinder.app.convergence_widget import ConvergenceWidget
from mufinder.app.workflow_controllers.base import (
    WorkflowControllerContext,
    WorkflowJobController,
)
from mufinder.app.workflow_submission import effective_row_status


@dataclass(frozen=True)
class ConvergenceCutoffDeps:
    """Cutoff-convergence-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating convergence code
    still calls them there; the controller reaches them only through these
    required callables. Workflow-specific dependencies like these never go in
    the shared :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`.
    """

    active_convergence_definition: Callable[[], CalculationDefinition | None]
    deserialize_convergence_config: Callable[[object], ConvergenceConfig | None]
    sync_equivalent_convergence_cutoff_mappings: Callable[[CalculationDefinition], None]
    definition_for_convergence_cutoff: Callable[
        [float], tuple[CalculationDefinition | None, ConvergenceConfig | None]
    ]
    job_for_definition_cutoff: Callable[[CalculationDefinition, float], Job | None]
    cutoff_key: Callable[[float], str]
    max_stress_for_job: Callable[[UUID], float | None]


class ConvergenceCutoffController(WorkflowJobController):
    """Controller for the cutoff-energy convergence sweep."""

    calc_types = frozenset({"castep_convergence_cutoff"})
    no_ready_message = (
        "Only configured cutoffs that have not yet been submitted can be submitted."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: ConvergenceCutoffDeps,
        view: ConvergenceWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- submission contract ---

    def normalize(self, value: object) -> float:
        return float(value)  # type: ignore[arg-type]

    # ``is_ready`` uses the shared base default
    # (:func:`~mufinder.app.workflow_submission.workflow_row_state_is_ready`),
    # so convergence rows use the same submit-eligibility predicate as every
    # other workflow.

    def build_states(self) -> dict[float, SweepPointRunState]:
        document = self._context.document()
        if document is None:
            return {}

        states: dict[float, SweepPointRunState] = {}
        form_prefix, form_cutoffs = self._convergence_form_snapshot()
        for cutoff in form_cutoffs:
            states[cutoff] = SweepPointRunState(
                status="ready",
                job_label=convergence_job_name(
                    form_prefix,
                    cutoff,
                    sweep_parameter="cut_off_energy",
                ),
            )

        definition = self._deps.active_convergence_definition()
        if definition is None:
            return states

        self._deps.sync_equivalent_convergence_cutoff_mappings(definition)
        config = self._deps.deserialize_convergence_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return states

        config_cutoffs = tuple(dict.fromkeys(float(value) for value in config.cutoff_energies))
        # If the editable convergence form has diverged from the active definition,
        # do not project old runtime results into the regenerated cutoff table.
        if form_cutoffs and (form_prefix != config.name_prefix or form_cutoffs != config_cutoffs):
            return states

        for cutoff in config.cutoff_energies:
            job = self._deps.job_for_definition_cutoff(definition, cutoff)
            if job is None:
                states[cutoff] = SweepPointRunState(
                    status="ready",
                    job_label=convergence_job_name(
                        config.name_prefix,
                        cutoff,
                        sweep_parameter="cut_off_energy",
                    ),
                )
                continue

            status_value = (
                "ready" if job.status == JobStatus.DEFINED else effective_row_status(job)
            )

            states[cutoff] = SweepPointRunState(
                status=status_value,
                total_energy=self._context.total_energy_for_job(job.id),
                max_force=self._context.max_force_for_job(job.id),
                max_stress=self._deps.max_stress_for_job(job.id),
                job_id=job.id,
                job_label=job.label,
            )

        return states

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []

        states = self.build_states()
        selected: list[Job] = []
        seen: set[UUID] = set()
        for cutoff in keys:
            state = states.get(float(cutoff))  # type: ignore[arg-type]
            if state is None:
                continue
            job = document.jobs.get(state.job_id) if state.job_id is not None else None
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def materialize(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []

        created: list[Job] = []
        grouped: dict[UUID, tuple[object, ConvergenceConfig, set[float]]] = {}
        for cutoff in keys:
            definition, config = self._deps.definition_for_convergence_cutoff(cutoff)
            if definition is None or config is None:
                continue
            existing = self._deps.job_for_definition_cutoff(definition, cutoff)
            if existing is not None:
                created.append(existing)
                if existing.status != JobStatus.DEFINED:
                    continue
                # Fall through for defined jobs so their inputs are rewritten
                # from the current configuration, matching the other workflows.
            record = grouped.get(definition.id)
            if record is None:
                grouped[definition.id] = (definition, config, {float(cutoff)})
            else:
                record[2].add(float(cutoff))

        for definition, config, cutoffs in grouped.values():
            if not definition.input_structure_ids:
                continue
            structure = document.structures.get(definition.input_structure_ids[0])
            output_root = self._context.definition_output_directory(definition)
            if structure is None or output_root is None:
                continue
            self._context.project_service.repair_supercell_asset_if_needed(
                document,
                structure_id=structure.id,
            )

            try:
                structure_spec = structure_to_spec(structure)
                all_jobs = expand_convergence_jobs(structure=structure_spec, config=config).jobs
            except (ValueError, StructureBridgeError):
                continue

            desired_labels = {
                convergence_job_name(
                    config.name_prefix,
                    cutoff,
                    sweep_parameter="cut_off_energy",
                ): cutoff
                for cutoff in cutoffs
            }
            mapping = definition.parameters.setdefault("job_ids_by_cutoff", {})
            if not isinstance(mapping, dict):
                mapping = {}
                definition.parameters["job_ids_by_cutoff"] = mapping

            for job_spec in all_jobs:
                cutoff = desired_labels.get(job_spec.name)
                if cutoff is None:
                    continue
                existing = self._deps.job_for_definition_cutoff(definition, cutoff)
                if existing is not None:
                    if existing.status == JobStatus.DEFINED:
                        write_job_directory(job_spec, output_root)
                    continue
                write_job_directory(job_spec, output_root)
                job = self._context.project_service.add_job(
                    document,
                    label=job_spec.name,
                    workflow_stage=job_spec.workflow_stage,
                    calc_type="castep_convergence_cutoff",
                )
                mapping[self._deps.cutoff_key(cutoff)] = str(job.id)
                created.append(job)

        deduped: list[Job] = []
        seen: set[UUID] = set()
        for job in created:
            if job.id in seen:
                continue
            seen.add(job.id)
            deduped.append(job)
        return deduped

    # --- view contract ---

    def refresh_view(self) -> None:
        document = self._context.document()
        if document is None:
            self._view.set_runtime_states({})
            return
        self._view.set_runtime_states(self.build_states())

    # --- private helpers moved from MainWindow (cutoff-only) ---

    def _convergence_form_snapshot(self) -> tuple[str, tuple[float, ...]]:
        document = self._context.document()
        if document is None:
            return "cutoff_convergence", ()

        form_sources: list[dict] = []
        widget_state = self._view.collect_state()
        if isinstance(widget_state, dict):
            form_sources.append(widget_state)

        persisted = document.ui_state.get("convergence_form", {})
        if isinstance(persisted, dict):
            form_sources.append(persisted)

        for form_state in form_sources:
            name_prefix = (
                str(form_state.get("name_prefix", "cutoff_convergence")).strip()
                or "cutoff_convergence"
            )

            cutoffs: list[float] = []
            raw_cutoffs = form_state.get("cutoff_energies", [])
            if isinstance(raw_cutoffs, list):
                for value in raw_cutoffs:
                    try:
                        cutoffs.append(float(value))
                    except (TypeError, ValueError):
                        continue

            unique = tuple(dict.fromkeys(cutoffs))
            if unique:
                return name_prefix, unique

        return "cutoff_convergence", ()


__all__ = ["ConvergenceCutoffController", "ConvergenceCutoffDeps"]
