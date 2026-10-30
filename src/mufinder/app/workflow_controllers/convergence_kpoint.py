"""K-point-convergence workflow controller.

The k-point-grid sweep of
:class:`~mufinder.app.workflow_controllers.sweep_convergence.SweepConvergenceController`:
rows are keyed by Monkhorst-Pack grid, jobs are bound in ``job_ids_by_grid``,
and a definition written before that mapping existed is read through the job
label its configuration's name prefix gives the grid. Shared convergence
helpers that other ``MainWindow`` code still calls are reached through
:class:`ConvergenceKpointDeps`, the per-controller deps dataclass, following
the shared-context + per-controller-deps + view pattern documented in
:mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job, Structure
from mufinder.science.kpoint_grids import kpoint_count, most_uniform_grid

from mufinder.services.convergence_defaults import (
    KPOINT_EXAMINER_STATE_KEY,
    KPOINT_SNAPSHOT,
    build_kpoint_overview_snapshot,
    store_convergence_overview_snapshot,
)
from mufinder.services.definition_mappings import remove_grid_keys_from_definition
from mufinder.services.project_document import ProjectDocument

from mufinder.app.convergence_types import (
    KPOINT_GRID_CODEC,
    ConvergenceConfig,
    KPointGrid,
    SweepPointRunState,
    format_kpoint_grid,
    parse_grid_mapping_key,
)
from mufinder.app.kpoint_convergence_widget import KPointConvergenceWidget
from mufinder.app.workflow_controllers.sweep_convergence import SweepConvergenceController


@dataclass(frozen=True)
class ConvergenceKpointDeps:
    """K-point-convergence-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because other convergence code still
    calls them there; the controller reaches them only through these required
    callables. Workflow-specific dependencies like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`.

    This deliberately mirrors ``ConvergenceCutoffDeps`` but is its own type — the
    two sweeps do not share a deps instance even where fields overlap.

    ``run_examiner`` opens the examiner dialog modally and returns its final
    state; ``show_warning`` and ``confirm_choice`` show message boxes; and
    ``refresh_after_default_adopted`` refreshes the window views an adopted
    default changes; ``select_simulation_cell_in_editor`` opens the Supercell
    Editor for the sweep's stage and calls back with the cell picked there;
    ``refresh_jobs_table`` and ``refresh_journal_view`` refresh the job
    dashboard and the journal after a configuration is stored.
    """

    deserialize_convergence_config: Callable[[object], ConvergenceConfig | None]
    grid_key: Callable[[KPointGrid], str]
    max_stress_for_job: Callable[[UUID], float | None]
    run_examiner: Callable[..., dict[str, object]]
    show_warning: Callable[[str, str], None]
    confirm_choice: Callable[..., str]
    refresh_after_default_adopted: Callable[[], None]
    select_simulation_cell_in_editor: Callable[..., None]
    refresh_jobs_table: Callable[[], None]
    refresh_journal_view: Callable[[], None]


class ConvergenceKpointController(SweepConvergenceController):
    """Controller for the k-point-grid convergence sweep."""

    calc_types = frozenset({"castep_convergence_kpoint"})
    stage = WorkflowStage.CONVERGENCE_KPOINT
    active_key = "active_convergence_kpoint_definition_id"
    no_ready_message = (
        "Only configured k-point grids that have not yet been submitted can be submitted."
    )

    key_codec = KPOINT_GRID_CODEC
    sweep_parameter = "kpoint_mp_grid"
    job_calc_type = "castep_convergence_kpoint"
    mapping_name = "job_ids_by_grid"
    form_state_key = "kpoint_convergence_form"
    keys_state_field = "kpoint_grids"
    default_name_prefix = "kpoint_convergence"
    examiner_state_key = KPOINT_EXAMINER_STATE_KEY
    snapshot_persist_failure_message = (
        "Could not persist k-point convergence overview snapshot to disk"
    )
    simulation_cell_name = "k-point"
    output_folder = "convergence_kpoint"
    job_noun = "k-point"
    generated_jobs_name = "k-point convergence"

    _deps: ConvergenceKpointDeps
    _view: KPointConvergenceWidget

    def register_configured_definition(
        self,
        document: ProjectDocument,
        *,
        input_structure_id: UUID,
        parameters: dict[str, object],
        output_directory: Path,
        simulation_cell_mode: str,
        supercell_source_workflow: str | None,
    ) -> CalculationDefinition:
        """Register a new k-point sweep definition and copy equivalent job bindings into it."""

        definition = self._context.project_service.register_convergence_kpoint_definition(
            document,
            input_structure_id=input_structure_id,
            parameters=parameters,
            output_directory=output_directory,
            simulation_cell_mode=simulation_cell_mode,
            supercell_source_workflow=supercell_source_workflow,
        )
        self.sync_mappings(definition)
        return definition

    def show_simulation_cell_structure(self, structure: Structure | None) -> None:
        """Give the widget the sweep cell's id and lattice for its generation warning."""

        self._view.set_generation_context(
            structure_id=str(structure.id) if structure is not None else None,
            lattice_vectors=structure.lattice_vectors if structure is not None else None,
        )

    def mapping_key(self, key: KPointGrid) -> str:
        return self._deps.grid_key(key)

    def parse_mapping_key(self, raw: object) -> KPointGrid | None:
        return parse_grid_mapping_key(raw) if isinstance(raw, str) else None

    def sweep_point_signatures(self, config: ConvergenceConfig) -> dict[KPointGrid, object]:
        """Map each grid to its job-name prefix and its effective parameters.

        A grid's effective parameters are the shared parameters updated by
        every override group listing it.
        """

        if config.sweep_parameter != self.sweep_parameter:
            return {}
        signatures: dict[KPointGrid, object] = {}
        for value in config.sweep_values:
            if not isinstance(value, tuple) or len(value) != 3:
                continue
            grid = (int(value[0]), int(value[1]), int(value[2]))
            params = dict(config.shared_params)
            for members, override in config.overrides.items():
                member_grids = {
                    tuple(int(component) for component in member)
                    for member in members
                    if isinstance(member, tuple) and len(member) == 3
                }
                if grid in member_grids:
                    params.update(override)
            signatures[grid] = (config.name_prefix, params)
        return signatures

    def remove_labels_from_definition(
        self, definition: CalculationDefinition, labels: Collection[str]
    ) -> bool:
        """Drop the grids whose mapping keys are ``labels``; the definition is always kept."""

        remove_grid_keys_from_definition(
            definition, labels, mapping_name=self.mapping_name, mapping_key=self.mapping_key
        )
        return False

    def legacy_job_for_key(self, definition: CalculationDefinition, key: KPointGrid) -> Job | None:
        """Return the newest job labelled as the grid's job under the definition's prefix."""

        configuration = definition.parameters.get("configuration", {})
        name_prefix = (
            configuration.get("name_prefix", self.default_name_prefix)
            if isinstance(configuration, dict)
            else self.default_name_prefix
        )
        return self.newest_job_labelled(self.job_name(name_prefix, key))

    # --- examine and adopt hooks ---

    @staticmethod
    def examiner_points(
        states: Mapping[KPointGrid, SweepPointRunState],
        lattice_vectors: Sequence[Sequence[float]],
    ) -> tuple[dict[float, SweepPointRunState], dict[float, str], list[str]]:
        """Key k-point sweep states by k-point count for the examiner's axis.

        Two different grids with the same count (only possible with hand-entered
        grids) cannot share a point on that axis. One is kept per count: a grid
        with a completed energy is preferred, then the most uniformly sampled one
        for the cell (see :func:`~mufinder.science.kpoint_grids.most_uniform_grid`).
        Each grid left out is described in the returned notes.
        """

        by_count: dict[int, list[KPointGrid]] = {}
        for grid in states:
            by_count.setdefault(kpoint_count(grid), []).append(tuple(int(n) for n in grid))  # type: ignore[arg-type]
        counts_map: dict[float, SweepPointRunState] = {}
        labels: dict[float, str] = {}
        notes: list[str] = []
        for count, grids in by_count.items():
            kept = grids[0]
            if len(grids) > 1:
                completed = [grid for grid in grids if states[grid].total_energy is not None]
                if len(completed) == 1:
                    kept = completed[0]
                    reason = "it is the only one with a completed result"
                else:
                    kept = most_uniform_grid(lattice_vectors, completed or grids)
                    reason = "it samples the cell most uniformly" + (
                        " of those with a completed result"
                        if completed and len(completed) < len(grids)
                        else ""
                    )
                excluded = ", ".join(format_kpoint_grid(grid) for grid in grids if grid != kept)
                notes.append(
                    f"{excluded} left out: same k-point count ({count}) as {format_kpoint_grid(kept)}, "
                    f"which is examined instead because {reason}. "
                    "Remove the extra grid from the sweep to examine it."
                )
            counts_map[float(count)] = states[kept]
            labels[float(count)] = f"Grid: {format_kpoint_grid(kept)}"
        return counts_map, labels, notes

    def examiner_input(
        self, states: Mapping[KPointGrid, SweepPointRunState], cell: Structure
    ) -> tuple[dict[float, SweepPointRunState], dict[float, str], dict[str, object]]:
        """Examine one grid per k-point count, labelled with its grid, on ``cell``'s lattice."""

        counts_map, labels, notes = self.examiner_points(states, cell.lattice_vectors)
        return (
            counts_map,
            labels,
            {
                "sweep_kind": "kpoint",
                "point_labels": labels,
                "lattice_vectors": cell.lattice_vectors,
                "notes": notes,
            },
        )

    def resolve_adopted_key(
        self, value: float, point_labels: Mapping[float, str]
    ) -> KPointGrid | None:
        """Return the grid the adopted point's ``Grid:`` label names."""

        label = point_labels.get(float(value), "")
        if not label.startswith("Grid:"):
            return None
        payload = label.split(":", maxsplit=1)[1].strip().replace(" ", "")
        return parse_grid_mapping_key(payload)

    def unresolved_adopt_message(self) -> str:
        return "Could not resolve the recommended k-point grid from the plotted data."

    def record_adopted_default(
        self,
        document: ProjectDocument,
        key: KPointGrid,
        definition: CalculationDefinition | None,
        *,
        method: str,
        energy_threshold: float,
        force_threshold: float,
        stress_threshold: float | None,
        cell_id: str | None,
        definition_cell_id: str | None,
        xc_functional: str | None,
        point_labels: Mapping[float, str],
    ) -> str:
        """Set the grid with the lattice it was converged on, and pin its snapshot.

        Later workflows rescale the grid to their own cell with
        ``equivalent_grid_for_supercell``, which needs this reference lattice:
        the sweep's cell, else the active host.
        """

        reference_cell = self.sweep_cell(definition)
        if reference_cell is None and document.project.host_structure_id is not None:
            reference_cell = document.structures.get(document.project.host_structure_id)

        self._context.project_service.set_recommended_kpoint_grid(
            document,
            key,
            lattice_vectors=(
                [list(row) for row in reference_cell.lattice_vectors]
                if reference_cell is not None
                else None
            ),
            method=method,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            stress_threshold=stress_threshold,
            cell_id=cell_id,
            xc_functional=xc_functional,
        )
        store_convergence_overview_snapshot(
            document,
            KPOINT_SNAPSHOT,
            build_kpoint_overview_snapshot(
                document,
                grid=key,
                energy_threshold=float(energy_threshold),
                force_threshold=float(force_threshold),
                simulation_cell_structure_id=definition_cell_id,
                labels=point_labels,
                stress_threshold=float(stress_threshold) if stress_threshold is not None else None,
            ),
        )
        return f"Project default k-point grid set to {key[0]}x{key[1]}x{key[2]}"


__all__ = ["ConvergenceKpointController", "ConvergenceKpointDeps"]
