"""Shared controller for the two convergence sweeps (cutoff energy and k-point grid).

A sweep stages one CASTEP singlepoint per sweep value. Its definition's
configuration is a serialised :class:`~mufinder.app.convergence_types.ConvergenceConfig`,
its jobs are bound in a ``job_ids_by_<key>`` mapping, and its rows are keyed by
the sweep value itself. :class:`SweepConvergenceController` implements the
submit, materialize and refresh cycle once; the concrete controllers declare
the sweep parameter, the job calc type, the mapping name and the form-state key
as class attributes and supply four hooks: :meth:`mapping_key` (the key's
spelling in the job mapping), :meth:`parse_mapping_key` (its inverse),
:meth:`sweep_point_signatures` (what makes two definitions' sweep points
equivalent, for :meth:`sync_mappings`) and :meth:`legacy_job_for_key` (the
lookup for a definition written before the mapping existed).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationDefinition, Job
from mufinder.services.definition_mappings import (
    convergence_mapping_job_priority,
    sweep_definition_context_matches,
    sync_equivalent_mappings,
)
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory

from mufinder.app.convergence_adapter import convergence_job_name, expand_convergence_jobs
from mufinder.app.convergence_types import ConvergenceConfig, SweepKeyCodec, SweepPointRunState
from mufinder.app.workflow_controllers.base import (
    WorkflowControllerContext,
    WorkflowJobController,
    dedupe_jobs,
)
from mufinder.app.workflow_submission import effective_row_status


class SweepConvergenceView(Protocol):
    """What a sweep controller needs from its widget."""

    def collect_state(self) -> dict[str, Any]: ...

    def set_runtime_states(self, states: dict[Any, SweepPointRunState]) -> None: ...


class SweepConvergenceDeps(Protocol):
    """The deps fields the shared sweep cycle reads; each sweep's deps add their own."""

    @property
    def deserialize_convergence_config(self) -> Callable[[object], ConvergenceConfig | None]: ...

    @property
    def max_stress_for_job(self) -> Callable[[UUID], float | None]: ...


class SweepConvergenceController(WorkflowJobController):
    """Submit, materialize and refresh cycle for one convergence sweep.

    Class attributes a subclass sets, besides the base controller's
    ``calc_types``, ``stage``, ``active_key`` and ``no_ready_message``:

    - ``key_codec``: the sweep's :class:`~mufinder.app.convergence_types.SweepKeyCodec`;
    - ``sweep_parameter``: ``"cut_off_energy"`` or ``"kpoint_mp_grid"``;
    - ``job_calc_type``: the calc type its jobs are created with;
    - ``mapping_name``: the definition's ``job_ids_by_<key>`` mapping;
    - ``form_state_key`` / ``keys_state_field`` / ``default_name_prefix``: where
      the widget's form state lives in ``ui_state``, its key list and its
      default job-name prefix.
    """

    key_codec: SweepKeyCodec[Any]
    sweep_parameter = ""
    job_calc_type = ""
    mapping_name = ""
    form_state_key = ""
    keys_state_field = ""
    default_name_prefix = ""

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: SweepConvergenceDeps,
        view: SweepConvergenceView,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- hooks ---

    def mapping_key(self, key: Any) -> str:
        """Return ``key``'s spelling in the definition's job mapping."""

        raise NotImplementedError

    def parse_mapping_key(self, raw: object) -> Any | None:
        """Return the sweep key a job-mapping key spells, or ``None``."""

        raise NotImplementedError

    def sweep_point_signatures(self, config: ConvergenceConfig) -> dict[Any, object]:
        """Map each of ``config``'s sweep keys to what a matching point must share.

        Empty when ``config`` sweeps another parameter.
        """

        raise NotImplementedError

    def sync_mappings(self, definition: CalculationDefinition) -> None:
        """Copy job bindings into ``definition`` from equivalent sweep definitions.

        A definition is equivalent when it shares ``definition``'s structure,
        output directory and simulation cell
        (:func:`~mufinder.services.definition_mappings.sweep_definition_context_matches`);
        a sweep point carries over when both configurations list it with the
        same :meth:`sweep_point_signatures` entry. Candidates rank by
        :func:`~mufinder.services.definition_mappings.convergence_mapping_job_priority`.
        """

        document = self._context.document()
        if document is None or not isinstance(definition.parameters.get("configuration"), dict):
            return
        sync_equivalent_mappings(
            document,
            self.definitions(),
            definition,
            deserialize=self._deps.deserialize_convergence_config,
            signatures=self.sweep_point_signatures,
            priority=convergence_mapping_job_priority,
            mapping_name=self.mapping_name,
            context_matches=sweep_definition_context_matches,
            parse_key=self.parse_mapping_key,
            format_key=self.mapping_key,
            unlisted_keys_match=False,
        )

    def legacy_job_for_key(self, definition: CalculationDefinition, key: Any) -> Job | None:
        """Return the job for ``key`` of a definition with no job mapping."""

        raise NotImplementedError

    # --- definition and job lookups ---

    def active_config(self) -> tuple[CalculationDefinition | None, ConvergenceConfig | None]:
        """Return the active definition and its sweep configuration.

        The definition's job mapping is synced first. The configuration is
        ``None`` when it does not deserialise or sweeps another parameter.
        """

        definition = self.active_definition()
        if definition is None:
            return None, None
        self.sync_mappings(definition)
        config = self._deps.deserialize_convergence_config(
            definition.parameters.get("configuration", {})
        )
        if config is None or config.sweep_parameter != self.sweep_parameter:
            return definition, None
        return definition, config

    def config_keys(self, config: ConvergenceConfig) -> tuple[Any, ...]:
        """Return ``config``'s sweep values as keys, in order and without repeats."""

        return tuple(
            dict.fromkeys(
                key
                for raw in config.sweep_values
                if (key := self.key_codec.coerce(raw)) is not None
            )
        )

    def definition_for_sweep_key(
        self, key: Any
    ) -> tuple[CalculationDefinition | None, ConvergenceConfig | None]:
        """Return the active definition and configuration when they sweep ``key``."""

        definition, config = self.active_config()
        if (
            definition is None
            or config is None
            or self.normalize(key) not in self.config_keys(config)
        ):
            return None, None
        return definition, config

    def job_for_sweep_key(self, definition: CalculationDefinition, key: Any) -> Job | None:
        """Return the job ``definition`` binds to its ``key`` sweep point.

        A definition carrying the ``mapping_name`` mapping answers from it
        alone; one written before the mapping existed falls back to
        :meth:`legacy_job_for_key`.
        """

        if isinstance(definition.parameters.get(self.mapping_name), dict):
            return self.job_for_key(
                definition, self.mapping_key(key), mapping_name=self.mapping_name
            )
        return self.legacy_job_for_key(definition, key)

    def newest_job_labelled(self, label: str) -> Job | None:
        document = self._context.document()
        if document is None:
            return None
        for job in reversed(tuple(document.jobs.values())):
            if job.label == label:
                return job
        return None

    def job_name(self, name_prefix: str, key: Any) -> str:
        return convergence_job_name(name_prefix, key, sweep_parameter=self.sweep_parameter)

    # --- submission contract ---

    def normalize(self, value: object) -> Any:
        return self.key_codec.normalize(value)

    # ``is_ready`` uses the shared base default
    # (:func:`~mufinder.app.workflow_submission.workflow_row_state_is_ready`),
    # so convergence rows use the same submit-eligibility predicate as every
    # other workflow.

    def build_states(self) -> dict[Any, SweepPointRunState]:
        document = self._context.document()
        if document is None:
            return {}

        form_prefix, form_keys = self.form_snapshot()
        states: dict[Any, SweepPointRunState] = {
            key: SweepPointRunState(status="ready", job_label=self.job_name(form_prefix, key))
            for key in form_keys
        }

        definition, config = self.active_config()
        if definition is None or config is None:
            return states

        config_keys = self.config_keys(config)
        # If the editable form has diverged from the active definition, do not
        # project old runtime results into the regenerated table.
        if form_keys and (form_prefix != config.name_prefix or form_keys != config_keys):
            return states

        for key in config_keys:
            job = self.job_for_sweep_key(definition, key)
            if job is None:
                states[key] = SweepPointRunState(
                    status="ready", job_label=self.job_name(config.name_prefix, key)
                )
                continue
            states[key] = SweepPointRunState(
                status="ready" if job.status == JobStatus.DEFINED else effective_row_status(job),
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
        jobs: list[Job] = []
        for raw in keys:
            key = self.key_codec.coerce(raw)
            state = states.get(key) if key is not None else None
            job = (
                document.jobs.get(state.job_id)
                if state is not None and state.job_id is not None
                else None
            )
            if job is not None:
                jobs.append(job)
        return dedupe_jobs(jobs)

    def materialize(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []

        created: list[Job] = []
        grouped: dict[UUID, tuple[CalculationDefinition, ConvergenceConfig, set[Any]]] = {}
        for raw in keys:
            key = self.normalize(raw)
            definition, config = self.definition_for_sweep_key(key)
            if definition is None or config is None:
                continue
            existing = self.job_for_sweep_key(definition, key)
            if existing is not None:
                created.append(existing)
                if existing.status != JobStatus.DEFINED:
                    continue
                # Fall through for defined jobs so their inputs are rewritten
                # from the current configuration, matching the other workflows.
            grouped.setdefault(definition.id, (definition, config, set()))[2].add(key)

        for definition, config, grouped_keys in grouped.values():
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

            desired_labels = {self.job_name(config.name_prefix, key): key for key in grouped_keys}
            mapping = definition.parameters.setdefault(self.mapping_name, {})
            if not isinstance(mapping, dict):
                mapping = {}
                definition.parameters[self.mapping_name] = mapping

            for job_spec in all_jobs:
                key = desired_labels.get(job_spec.name)
                if key is None:
                    continue
                existing = self.job_for_sweep_key(definition, key)
                if existing is not None:
                    if existing.status == JobStatus.DEFINED:
                        write_job_directory(job_spec, output_root)
                    continue
                write_job_directory(job_spec, output_root)
                job = self._context.project_service.add_job(
                    document,
                    label=job_spec.name,
                    workflow_stage=job_spec.workflow_stage,
                    calc_type=self.job_calc_type,
                )
                mapping[self.mapping_key(key)] = str(job.id)
                created.append(job)

        return dedupe_jobs(created)

    # --- view contract ---

    def refresh_view(self) -> None:
        document = self._context.document()
        if document is None:
            self._view.set_runtime_states({})
            return
        self._view.set_runtime_states(self.build_states())

    def form_snapshot(self) -> tuple[str, tuple[Any, ...]]:
        """Return the form's job-name prefix and keys.

        The live widget state is read first and the persisted
        ``ui_state[form_state_key]`` second; the first holding any key wins.
        """

        document = self._context.document()
        if document is None:
            return self.default_name_prefix, ()

        sources: list[dict[str, Any]] = []
        widget_state = self._view.collect_state()
        if isinstance(widget_state, dict):
            sources.append(widget_state)
        persisted = document.ui_state.get(self.form_state_key, {})
        if isinstance(persisted, dict):
            sources.append(persisted)

        for form_state in sources:
            raw_keys = form_state.get(self.keys_state_field, [])
            keys = tuple(
                dict.fromkeys(
                    key
                    for raw in (raw_keys if isinstance(raw_keys, list) else [])
                    if (key := self.key_codec.coerce(raw)) is not None
                )
            )
            if keys:
                name_prefix = (
                    str(form_state.get("name_prefix", self.default_name_prefix)).strip()
                    or self.default_name_prefix
                )
                return name_prefix, keys

        return self.default_name_prefix, ()


__all__ = ["SweepConvergenceController"]
