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

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job

from mufinder.app.convergence_types import KPOINT_GRID_CODEC, ConvergenceConfig, KPointGrid
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
    sync_equivalent_convergence_kpoint_mappings: Callable[[CalculationDefinition], None]
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

    def sync_mappings(self, definition: CalculationDefinition) -> None:
        self._deps.sync_equivalent_convergence_kpoint_mappings(definition)

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
