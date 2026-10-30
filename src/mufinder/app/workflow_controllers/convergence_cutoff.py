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

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Job, Structure

from mufinder.services.convergence_defaults import (
    CUTOFF_EXAMINER_STATE_KEY,
    CUTOFF_SNAPSHOT,
    build_cutoff_overview_snapshot,
    store_convergence_overview_snapshot,
)
from mufinder.services.definition_mappings import remove_cutoff_from_definition
from mufinder.services.project_document import ProjectDocument

from mufinder.app.convergence_types import CUTOFF_KEY_CODEC, ConvergenceConfig, SweepPointRunState
from mufinder.app.convergence_widget import CutoffConvergenceWidget
from mufinder.app.workflow_controllers.sweep_convergence import SweepConvergenceController


@dataclass(frozen=True)
class ConvergenceCutoffDeps:
    """Cutoff-convergence-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because other convergence code still
    calls them there; the controller reaches them only through these required
    callables. Workflow-specific dependencies like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`.

    ``run_examiner`` opens the examiner dialog modally and returns its final
    state; ``show_warning`` and ``confirm_choice`` show message boxes; and
    ``refresh_after_default_adopted`` refreshes the window views an adopted
    default changes.
    """

    deserialize_convergence_config: Callable[[object], ConvergenceConfig | None]
    cutoff_key: Callable[[float], str]
    max_stress_for_job: Callable[[UUID], float | None]
    run_examiner: Callable[..., dict[str, object]]
    show_warning: Callable[[str, str], None]
    confirm_choice: Callable[..., str]
    refresh_after_default_adopted: Callable[[], None]


class ConvergenceCutoffController(SweepConvergenceController):
    """Controller for the cutoff-energy convergence sweep."""

    calc_types = frozenset({"castep_convergence_cutoff"})
    stage = WorkflowStage.CONVERGENCE_CUTOFF
    active_key = "active_convergence_cutoff_definition_id"
    no_ready_message = "Only configured cutoffs that have not yet been submitted can be submitted."

    key_codec = CUTOFF_KEY_CODEC
    sweep_parameter = "cut_off_energy"
    job_calc_type = "castep_convergence_cutoff"
    mapping_name = "job_ids_by_cutoff"
    form_state_key = "convergence_form"
    keys_state_field = "cutoff_energies"
    default_name_prefix = "cutoff_convergence"
    examiner_state_key = CUTOFF_EXAMINER_STATE_KEY
    snapshot_persist_failure_message = "Could not persist convergence overview snapshot to disk"

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
        for value in config.sweep_values:
            cutoff = float(value)
            params = dict(config.shared_params)
            for members, override in config.overrides.items():
                if cutoff in {float(member) for member in members}:  # type: ignore[arg-type]
                    params.update(override)
            signatures[cutoff] = (config.name_prefix, params)
        return signatures

    def remove_cutoff_from_definition(
        self, definition: CalculationDefinition, cutoff: float
    ) -> bool:
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
        if (
            not isinstance(labels, list)
            or configuration is None
            or configuration.sweep_parameter != self.sweep_parameter
        ):
            return None
        try:
            index = list(configuration.sweep_values).index(float(key))
        except ValueError:
            return None
        if index >= len(labels):
            return None
        return self.newest_job_labelled(str(labels[index]))

    def job_for_cutoff(self, definition: CalculationDefinition, cutoff: float) -> Job | None:
        """Return the job ``definition`` binds to its ``cutoff`` sweep point."""

        return self.job_for_sweep_key(definition, cutoff)

    # --- examine and adopt hooks ---

    def examiner_input(
        self, states: Mapping[float, SweepPointRunState], cell: Structure
    ) -> tuple[dict[float, SweepPointRunState], dict[float, str], dict[str, object]]:
        """Examine every cutoff as it is; the dialog's options keep their cutoff defaults."""

        return dict(states), {}, {}

    def resolve_adopted_key(self, value: float, point_labels: Mapping[float, str]) -> float:
        return float(value)

    def record_adopted_default(
        self,
        document: ProjectDocument,
        key: float,
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
        self._context.project_service.set_recommended_cutoff(
            document,
            key,
            method=method,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            stress_threshold=stress_threshold,
            cell_id=cell_id,
            xc_functional=xc_functional,
        )
        store_convergence_overview_snapshot(
            document,
            CUTOFF_SNAPSHOT,
            build_cutoff_overview_snapshot(
                document,
                cutoff_energy=float(key),
                energy_threshold=float(energy_threshold),
                force_threshold=float(force_threshold),
                simulation_cell_structure_id=definition_cell_id,
                stress_threshold=float(stress_threshold) if stress_threshold is not None else None,
            ),
        )
        return f"Project default cutoff set to {float(key):.8g} eV"


__all__ = ["ConvergenceCutoffController", "ConvergenceCutoffDeps"]
