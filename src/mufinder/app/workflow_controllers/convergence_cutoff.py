"""Cutoff-convergence workflow controller.

The cutoff-energy sweep of
:class:`~mufinder.app.workflow_controllers.sweep_convergence.SweepConvergenceController`:
rows are keyed by cutoff energy (eV), jobs are bound in ``job_ids_by_cutoff``,
and a definition written before that mapping existed is read through its
``job_labels``. Shared convergence helpers that other ``MainWindow`` code still
calls are reached through :class:`ConvergenceCutoffDeps`, the per-controller
deps dataclass, following the shared-context + per-controller-deps + view
pattern documented in :mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job

from mufinder.services.definition_mappings import remove_cutoff_from_definition

from mufinder.app.convergence_types import CUTOFF_KEY_CODEC, ConvergenceConfig
from mufinder.app.convergence_widget import CutoffConvergenceWidget
from mufinder.app.workflow_controllers.sweep_convergence import SweepConvergenceController


@dataclass(frozen=True)
class ConvergenceCutoffDeps:
    """Cutoff-convergence-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because other convergence code still
    calls them there; the controller reaches them only through these required
    callables. Workflow-specific dependencies like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`.
    """

    deserialize_convergence_config: Callable[[object], ConvergenceConfig | None]
    cutoff_key: Callable[[float], str]
    max_stress_for_job: Callable[[UUID], float | None]


class ConvergenceCutoffController(SweepConvergenceController):
    """Controller for the cutoff-energy convergence sweep."""

    calc_types = frozenset({"castep_convergence_cutoff"})
    stage = WorkflowStage.CONVERGENCE_CUTOFF
    active_key = "active_convergence_cutoff_definition_id"
    no_ready_message = (
        "Only configured cutoffs that have not yet been submitted can be submitted."
    )

    key_codec = CUTOFF_KEY_CODEC
    sweep_parameter = "cut_off_energy"
    job_calc_type = "castep_convergence_cutoff"
    mapping_name = "job_ids_by_cutoff"
    form_state_key = "convergence_form"
    keys_state_field = "cutoff_energies"
    default_name_prefix = "cutoff_convergence"

    _deps: ConvergenceCutoffDeps
    _view: CutoffConvergenceWidget

    def mapping_key(self, key: float) -> str:
        return self._deps.cutoff_key(key)

    def parse_mapping_key(self, raw: object) -> float | None:
        try:
            return float(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    def sweep_point_signatures(self, config: ConvergenceConfig) -> dict[float, object]:
        """Map each cutoff to its job-name prefix and its effective parameters.

        A cutoff's effective parameters are the shared parameters updated by
        every override group listing it.
        """

        if config.sweep_parameter != self.sweep_parameter:
            return {}
        signatures: dict[float, object] = {}
        for value in config.cutoff_energies:
            cutoff = float(value)
            params = dict(config.shared_params)
            for members, override in config.overrides.items():
                if cutoff in {float(member) for member in members}:  # type: ignore[arg-type]
                    params.update(override)
            signatures[cutoff] = (config.name_prefix, params)
        return signatures

    def remove_cutoff_from_definition(self, definition: CalculationDefinition, cutoff: float) -> bool:
        """Drop ``cutoff`` from ``definition``; return whether the definition was deleted.

        See :func:`~mufinder.services.definition_mappings.remove_cutoff_from_definition`.
        """

        document = self._context.document()
        if document is None:
            return False
        return remove_cutoff_from_definition(
            document,
            definition,
            cutoff,
            mapping_name=self.mapping_name,
            mapping_key=self.mapping_key,
            active_key=self.active_key,
        )

    def legacy_job_for_key(self, definition: CalculationDefinition, key: float) -> Job | None:
        """Return the newest job labelled as the cutoff's entry in ``job_labels``."""

        labels = definition.parameters.get("job_labels", [])
        configuration = self._deps.deserialize_convergence_config(
            definition.parameters.get("configuration", {})
        )
        if not isinstance(labels, list) or configuration is None:
            return None
        try:
            index = list(configuration.cutoff_energies).index(float(key))
        except ValueError:
            return None
        if index >= len(labels):
            return None
        return self.newest_job_labelled(str(labels[index]))

    def job_for_cutoff(self, definition: CalculationDefinition, cutoff: float) -> Job | None:
        """Return the job ``definition`` binds to its ``cutoff`` sweep point."""

        return self.job_for_sweep_key(definition, cutoff)


__all__ = ["ConvergenceCutoffController", "ConvergenceCutoffDeps"]
