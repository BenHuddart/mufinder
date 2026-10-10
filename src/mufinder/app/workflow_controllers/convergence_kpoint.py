"""K-point-convergence workflow controller.

Owns the submit → materialize → refresh cycle for the CASTEP k-point-grid
convergence sweep. This is the sibling of
:class:`~mufinder.app.workflow_controllers.convergence_cutoff.ConvergenceCutoffController`
migrated onto
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`; the logic
here is a behavior-preserving move of the former ``MainWindow`` methods
``_materialize_kpoint_jobs``, ``_build_convergence_kpoint_states``,
``_jobs_for_convergence_kpoint_grids``, ``_refresh_kpoint_convergence_rows``, and
``_kpoint_form_snapshot``, plus (as of the PR 4.8 cleanup) the private
``_definition_for_kpoint_grid`` / ``_job_for_definition_grid`` helpers inlined
from ``MainWindow`` once no other caller needed them. Shared convergence helpers
that non-migrating code still calls remain on ``MainWindow`` and are reached via
:class:`ConvergenceKpointDeps` — the per-controller deps dataclass this module
defines, following the shared-context + per-controller-deps + view pattern
documented in :mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory

from mufinder.app.convergence_adapter import convergence_job_name, expand_convergence_jobs
from mufinder.app.convergence_types import ConvergenceConfig, SweepPointRunState
from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget
from mufinder.app.workflow_controllers.base import (
    WorkflowControllerContext,
    WorkflowJobController,
)
from mufinder.app.workflow_submission import effective_row_status

_Grid = tuple[int, int, int]


@dataclass(frozen=True)
class ConvergenceKpointDeps:
    """K-point-convergence-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating convergence code
    still calls them there; the controller reaches them only through these
    required callables. Workflow-specific dependencies like these never go in
    the shared :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`.

    This deliberately mirrors ``ConvergenceCutoffDeps`` but is its own type — the
    two sweeps do not share a deps instance even where fields overlap.
    """

    deserialize_convergence_config: Callable[[object], ConvergenceConfig | None]
    sync_equivalent_convergence_kpoint_mappings: Callable[[CalculationDefinition], None]
    grid_key: Callable[[_Grid], str]
    max_stress_for_job: Callable[[UUID], float | None]


class ConvergenceKpointController(WorkflowJobController):
    """Controller for the k-point-grid convergence sweep."""

    calc_types = frozenset({"castep_convergence_kpoint"})
    stage = WorkflowStage.CONVERGENCE_KPOINT
    active_key = "active_convergence_kpoint_definition_id"
    no_ready_message = (
        "Only configured k-point grids that have not yet been submitted can be submitted."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: ConvergenceKpointDeps,
        view: KPointConvergenceWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- submission contract ---

    def normalize(self, value: object) -> _Grid:
        grid = value  # type: ignore[assignment]
        return (int(grid[0]), int(grid[1]), int(grid[2]))

    # ``is_ready`` uses the shared base default
    # (:func:`~mufinder.app.workflow_submission.workflow_row_state_is_ready`),
    # so convergence rows use the same submit-eligibility predicate as every
    # other workflow.

    def build_states(self) -> dict[_Grid, SweepPointRunState]:
        document = self._context.document()
        if document is None:
            return {}

        states: dict[_Grid, SweepPointRunState] = {}
        form_prefix, form_grids = self._kpoint_form_snapshot()
        for grid in form_grids:
            states[grid] = SweepPointRunState(
                status="ready",
                job_label=convergence_job_name(
                    form_prefix,
                    grid,
                    sweep_parameter="kpoint_mp_grid",
                ),
            )

        definition = self.active_definition()
        if definition is None:
            return states

        self._deps.sync_equivalent_convergence_kpoint_mappings(definition)
        config = self._deps.deserialize_convergence_config(
            definition.parameters.get("configuration", {})
        )
        if config is None or config.sweep_parameter != "kpoint_mp_grid":
            return states

        config_grids = tuple(
            tuple(int(component) for component in value)
            for value in config.sweep_values
            if isinstance(value, tuple)
        )
        if form_grids and (form_prefix != config.name_prefix or form_grids != config_grids):
            return states

        for grid in config_grids:
            job = self._job_for_definition_grid(definition, grid)
            if job is None:
                states[grid] = SweepPointRunState(
                    status="ready",
                    job_label=convergence_job_name(
                        config.name_prefix,
                        grid,
                        sweep_parameter="kpoint_mp_grid",
                    ),
                )
                continue

            status_value = (
                "ready" if job.status == JobStatus.DEFINED else effective_row_status(job)
            )
            states[grid] = SweepPointRunState(
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
        for grid in keys:
            state = states.get(tuple(grid))  # type: ignore[arg-type]
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
        grouped: dict[UUID, tuple[object, ConvergenceConfig, set[_Grid]]] = {}
        for grid in keys:
            definition, config = self._definition_for_kpoint_grid(grid)  # type: ignore[arg-type]
            if definition is None or config is None:
                continue
            existing = self._job_for_definition_grid(definition, grid)  # type: ignore[arg-type]
            if existing is not None:
                created.append(existing)
                if existing.status != JobStatus.DEFINED:
                    continue
                # Fall through for defined jobs so their inputs are rewritten
                # from the current configuration, matching the other workflows.
            record = grouped.get(definition.id)
            if record is None:
                grouped[definition.id] = (definition, config, {tuple(grid)})  # type: ignore[arg-type]
            else:
                record[2].add(tuple(grid))  # type: ignore[arg-type]

        for definition, config, grouped_grids in grouped.values():
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
                    grid,
                    sweep_parameter="kpoint_mp_grid",
                ): grid
                for grid in grouped_grids
            }
            mapping = definition.parameters.setdefault("job_ids_by_grid", {})
            if not isinstance(mapping, dict):
                mapping = {}
                definition.parameters["job_ids_by_grid"] = mapping

            for job_spec in all_jobs:
                grid = desired_labels.get(job_spec.name)
                if grid is None:
                    continue
                existing = self._job_for_definition_grid(definition, grid)  # type: ignore[arg-type]
                if existing is not None:
                    if existing.status == JobStatus.DEFINED:
                        write_job_directory(job_spec, output_root)
                    continue
                write_job_directory(job_spec, output_root)
                job = self._context.project_service.add_job(
                    document,
                    label=job_spec.name,
                    workflow_stage=job_spec.workflow_stage,
                    calc_type="castep_convergence_kpoint",
                )
                mapping[self._deps.grid_key(grid)] = str(job.id)
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

    # --- private helpers moved from MainWindow (kpoint-only) ---

    def _definition_for_kpoint_grid(
        self, grid: _Grid
    ) -> tuple[CalculationDefinition | None, ConvergenceConfig | None]:
        """Return the active k-point definition/config that owns ``grid``.

        Inlined from the former ``MainWindow._definition_for_kpoint_grid`` (PR 4.8
        cleanup): the wrapped helpers already live in this controller's deps and
        no other caller needed the MainWindow original.
        """

        active_definition = self.active_definition()
        if active_definition is None:
            return None, None

        self._deps.sync_equivalent_convergence_kpoint_mappings(active_definition)
        config = self._deps.deserialize_convergence_config(
            active_definition.parameters.get("configuration", {})
        )
        if config is None or config.sweep_parameter != "kpoint_mp_grid":
            return None, None

        if grid in config.sweep_values:
            return active_definition, config
        return None, None

    def _job_for_definition_grid(
        self, definition: CalculationDefinition, grid: _Grid
    ) -> Job | None:
        """Return the existing job for ``definition``'s ``grid`` sweep point.

        Inlined from the former ``MainWindow._job_for_definition_grid`` (PR 4.8
        cleanup); reads the document via the shared context and the grid-key
        helper via this controller's deps.
        """

        document = self._context.document()
        if document is None:
            return None

        if isinstance(definition.parameters.get("job_ids_by_grid"), dict):
            return self.job_for_key(
                definition, self._deps.grid_key(grid), mapping_name="job_ids_by_grid"
            )

        label = convergence_job_name(
            definition.parameters.get("configuration", {}).get(
                "name_prefix", "kpoint_convergence"
            ),
            grid,
            sweep_parameter="kpoint_mp_grid",
        )
        for job in reversed(tuple(document.jobs.values())):
            if job.label == label:
                return job
        return None

    def _kpoint_form_snapshot(self) -> tuple[str, tuple[_Grid, ...]]:
        document = self._context.document()
        if document is None:
            return "kpoint_convergence", ()

        persisted = document.ui_state.get("kpoint_convergence_form", {})
        if not isinstance(persisted, dict):
            return "kpoint_convergence", ()

        name_prefix = str(persisted.get("name_prefix", "kpoint_convergence")).strip() or "kpoint_convergence"
        raw_grids = persisted.get("kpoint_grids", [])
        grids: list[_Grid] = []
        if isinstance(raw_grids, list):
            for entry in raw_grids:
                if not isinstance(entry, list) or len(entry) != 3:
                    continue
                try:
                    grid = (int(entry[0]), int(entry[1]), int(entry[2]))
                except (TypeError, ValueError):
                    continue
                grids.append(grid)
        return name_prefix, tuple(dict.fromkeys(grids))


__all__ = ["ConvergenceKpointController", "ConvergenceKpointDeps"]
