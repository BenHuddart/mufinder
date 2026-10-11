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

from collections.abc import Callable, Collection
from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job

from mufinder.services.definition_mappings import remove_grid_keys_from_definition

from mufinder.app.convergence_types import (
    KPOINT_GRID_CODEC,
    ConvergenceConfig,
    KPointGrid,
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
    """

    deserialize_convergence_config: Callable[[object], ConvergenceConfig | None]
    grid_key: Callable[[KPointGrid], str]
    max_stress_for_job: Callable[[UUID], float | None]


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

    _deps: ConvergenceKpointDeps
    _view: KPointConvergenceWidget

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


__all__ = ["ConvergenceKpointController", "ConvergenceKpointDeps"]
