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

The controller also examines a sweep and adopts its result as the project
default (:meth:`~SweepConvergenceController.examine`,
:meth:`~SweepConvergenceController.adopt`). Two more hooks carry what
differs between the sweeps: ``examiner_input`` (the examiner's points and its
sweep-specific options, among them the k-point grid of each point) and
``record_adopted_default`` (the project-default setter and the pinned overview
snapshot). The examiner emits the adopted sweep key itself. The examiner
dialog, the confirmation and the window refreshes come in through the deps.

It stores the configuration the widget's Configure dialog accepts
(:meth:`~SweepConvergenceController.configure`): the configuration is checked
by expanding it on the sweep's cell, then registered as a new definition, or,
for a sweep that reuses one (``reuse_matching_definition``), matched to an
existing definition with the same configuration and context
(:func:`~mufinder.services.definition_mappings.matching_sweep_definition`).
The ``register_configured_definition`` hook calls the sweep's project-service
registration.

It also keeps the sweep's ``Simulation cell`` selector in step with the
project (:meth:`~SweepConvergenceController.refresh_simulation_cell`) and
handles its three requests: pick a saved supercell
(:meth:`~SweepConvergenceController.select_simulation_cell`), go back to the
unit cell (:meth:`~SweepConvergenceController.use_unit_cell`), and pick or
create one in the Supercell Editor
(:meth:`~SweepConvergenceController.create_simulation_cell`), which the window
opens through the ``select_simulation_cell_in_editor`` dep. The
``show_simulation_cell_structure`` hook passes the resolved cell to a widget
that needs more of it than the selector does.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationDefinition, Job, Structure
from mufinder.domain.text import plural
from mufinder.persistence.codec import to_jsonable
from mufinder.services.project_document import ProjectDocument
from mufinder.services.definition_mappings import (
    convergence_mapping_job_priority,
    matching_sweep_definition,
    sweep_definition_context_matches,
    sync_equivalent_mappings,
)
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory

from mufinder.app.convergence_adapter import convergence_job_name, expand_convergence_jobs
from mufinder.app.convergence_types import (
    ConvergenceConfig,
    KPointGrid,
    SweepKeyCodec,
    SweepPointRunState,
    convergence_config_xc_functional,
)
from mufinder.app.adapter_common import IDENTITY_SUPERCELL
from mufinder.app.workflow_config_codecs import serialize_convergence_config
from mufinder.app.workflow_parameters import structure_species_options
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

    def set_simulation_cell_options(
        self,
        *,
        summary: str,
        assets: list[dict[str, str]],
        selected_structure_id: str | None,
        enabled: bool,
    ) -> None: ...

    def set_parameter_species_options(self, species_options: Sequence[str]) -> None: ...


class SweepConvergenceDeps(Protocol):
    """The deps fields the shared sweep cycle reads; each sweep's deps add their own."""

    @property
    def deserialize_convergence_config(self) -> Callable[[object], ConvergenceConfig | None]: ...

    @property
    def max_stress_for_job(self) -> Callable[[UUID], float | None]: ...

    @property
    def run_examiner(self) -> Callable[..., dict[str, object]]: ...

    @property
    def show_warning(self) -> Callable[[str, str], None]: ...

    @property
    def confirm_choice(self) -> Callable[..., str]: ...

    @property
    def refresh_after_default_adopted(self) -> Callable[[], None]: ...

    @property
    def select_simulation_cell_in_editor(self) -> Callable[..., None]: ...

    @property
    def refresh_jobs_table(self) -> Callable[[], None]: ...

    @property
    def refresh_journal_view(self) -> Callable[[], None]: ...


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
      default job-name prefix;
    - ``examiner_state_key``: where the examiner's state lives in ``ui_state``;
    - ``snapshot_persist_failure_message``: the status shown when saving an
      adopted default's overview snapshot fails;
    - ``simulation_cell_name``: the sweep's name in the status shown when a
      simulation cell is selected;
    - ``output_folder``: the folder under the project's ``files`` its
      definitions write to;
    - ``job_noun`` / ``generated_jobs_name``: the sweep's name in the status
      shown when a configuration is stored, and in the warning shown when
      there is no host structure to generate its jobs on;
    - ``reuse_matching_definition``: whether :meth:`configure` reuses an
      existing definition with the same configuration and context instead of
      registering a new one.
    """

    key_codec: SweepKeyCodec[Any]
    sweep_parameter = ""
    job_calc_type = ""
    mapping_name = ""
    form_state_key = ""
    keys_state_field = ""
    default_name_prefix = ""
    examiner_state_key = ""
    snapshot_persist_failure_message = ""
    simulation_cell_name = ""
    output_folder = ""
    job_noun = ""
    generated_jobs_name = ""
    reuse_matching_definition = False

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

    # --- examine and adopt ---

    def examiner_input(
        self, states: Mapping[Any, SweepPointRunState], cell: Structure
    ) -> tuple[dict[float, SweepPointRunState], dict[str, object]]:
        """Return the examiner's points by axis value and its options.

        The options are the sweep-specific keyword arguments of the examiner
        dialog; a k-point sweep's ``point_grids`` (the grid of each point) are
        also pinned with an adopted default. ``cell`` is the structure the sweep
        ran on.
        """

        raise NotImplementedError

    def record_adopted_default(
        self,
        document: ProjectDocument,
        key: Any,
        definition: CalculationDefinition | None,
        *,
        method: str,
        energy_threshold: float,
        force_threshold: float,
        stress_threshold: float | None,
        cell_id: str | None,
        definition_cell_id: str | None,
        xc_functional: str | None,
        point_grids: Mapping[float, KPointGrid],
    ) -> str:
        """Set ``key`` as the project default and pin its overview snapshot.

        ``cell_id`` is the cell recorded with the default (the sweep's cell,
        else the active host); ``definition_cell_id`` is the sweep's own cell
        only. Returns the status-bar message.
        """

        raise NotImplementedError

    def sweep_cell(self, definition: CalculationDefinition | None) -> Structure | None:
        """Return the cell a convergence sweep ran on: its definition's input structure.

        That is the unit cell or the saved supercell chosen under ``Simulation
        Cell``. Per-atom energies and k-point spacings must use this cell, not the
        host unit cell, or a supercell sweep is normalized by the wrong atom count
        and its grids are related to the wrong lattice.
        """

        document = self._context.document()
        if document is None or definition is None or not definition.input_structure_ids:
            return None
        return document.structures.get(definition.input_structure_ids[0])

    def examine(self) -> None:
        """Open the examiner on the active sweep and keep its state when it closes."""

        document = self._context.document()
        if document is None:
            self._context.show_info("No project", "Create or open a project first.")
            return

        host_id = document.project.host_structure_id
        host = document.structures.get(host_id) if host_id is not None else None
        if host is None:
            self._deps.show_warning(
                "No host structure",
                "Load a host structure before examining convergence.",
            )
            return

        cell = self.sweep_cell(self.active_definition()) or host
        points, options = self.examiner_input(self.build_states(), cell)
        grids = options.get("point_grids", {})

        persisted_state = document.ui_state.get(self.examiner_state_key, {})
        if not isinstance(persisted_state, dict):
            persisted_state = {}

        final_state = self._deps.run_examiner(
            points,
            num_atoms=len(cell.species),
            initial_state=persisted_state,
            on_adopted=lambda key, method, et, ft, st: self.adopt(
                key, method, et, ft, st, point_grids=grids
            ),
            on_state_changed=self.store_examiner_state,
            **options,
        )
        self.store_examiner_state(final_state)

    def store_examiner_state(self, state: dict) -> None:
        """Keep the examiner's state in ``ui_state``; the project is not saved."""

        document = self._context.document()
        if document is None:
            return

        serialized_state = to_jsonable(state)
        if not isinstance(serialized_state, dict):
            return

        document.ui_state[self.examiner_state_key] = serialized_state

    def adopt(
        self,
        key: Any,
        method: str,
        energy_threshold: float,
        force_threshold: float,
        stress_threshold: float | None = None,
        *,
        point_grids: Mapping[float, KPointGrid],
    ) -> None:
        """Make the examiner's adopted sweep ``key`` the project default.

        ``key`` is the cutoff in eV or the k-point grid the examiner emitted;
        ``point_grids`` are the grids it was opened with, pinned with a k-point
        default. A result converged with a functional other than the project's
        is confirmed first, since adopting it also changes the project
        functional.
        """

        document = self._context.document()
        if document is None:
            return

        definition = self.active_definition()
        definition_cell_id = None
        if definition is not None and definition.input_structure_ids:
            definition_cell_id = str(definition.input_structure_ids[0])
        xc_functional = self.definition_xc_functional(definition)
        if not self.confirm_adopted_xc_functional(xc_functional):
            return
        host_id = document.project.host_structure_id
        status = self.record_adopted_default(
            document,
            key,
            definition,
            method=method,
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            stress_threshold=stress_threshold,
            cell_id=(
                definition_cell_id
                if definition_cell_id is not None
                else (str(host_id) if host_id else None)
            ),
            definition_cell_id=definition_cell_id,
            xc_functional=xc_functional,
            point_grids=point_grids,
        )
        self._deps.refresh_after_default_adopted()
        self._context.persist_ui_state(failure_message=self.snapshot_persist_failure_message)
        self._context.status_message(status)

    def definition_xc_functional(self, definition: CalculationDefinition | None) -> str | None:
        """Return the one functional a convergence definition's sweep ran with."""

        if definition is None:
            return None
        config = self._deps.deserialize_convergence_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return None
        return convergence_config_xc_functional(config)

    def confirm_adopted_xc_functional(self, xc_functional: str | None) -> bool:
        """Confirm adopting a result converged with a non-default functional.

        Adopting a convergence result also makes the functional it was
        converged with the project default, so switching functional is
        confirmed rather than done silently.
        """

        document = self._context.document()
        if document is None or xc_functional is None:
            return True
        current = document.project.recommended_xc_functional
        if current is None or current == xc_functional:
            return True
        answer = self._deps.confirm_choice(
            "Change project functional?",
            f"These convergence calculations used the {xc_functional} functional, "
            f"but the project default functional is {current}.\n\n"
            f"Adopting this result will also change the project default "
            f"functional to {xc_functional}. Continue?",
            [("change", "Change functional"), ("cancel", "Cancel")],
            default="cancel",
            escape="cancel",
        )
        return answer == "change"

    # --- configure ---

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
        """Register a new definition for a configuration :meth:`configure` accepted.

        Returns the definition :meth:`configure` makes active.
        """

        raise NotImplementedError

    def configure(self, config: ConvergenceConfig) -> None:
        """Store ``config``, accepted in the widget's Configure dialog, as the active sweep.

        It is checked by expanding it on the sweep's resolved cell; no job is
        created and no input is written until the sweep is submitted.
        """

        document = self._context.document()
        if document is None:
            self._context.show_info("No project", "Create or open a project first.")
            return

        resolution = self._context.project_service.resolve_workflow_structure(
            document,
            self.stage,
            normalize_state=True,
        )
        structure = resolution.structure
        if structure is None:
            self._deps.show_warning(
                "No host structure",
                f"Load a host structure before generating {self.generated_jobs_name} jobs.",
            )
            return

        try:
            structure_spec = structure_to_spec(structure)
            expand_convergence_jobs(structure=structure_spec, config=config)
        except (ValueError, StructureBridgeError) as exc:
            self._deps.show_warning("Convergence setup invalid", str(exc))
            return

        input_structure_id = structure.id
        parameters = serialize_convergence_config(config)
        output_directory = document.project_root / "files" / self.output_folder
        simulation_cell_mode = resolution.normalized_selection.mode
        supercell_source_workflow = (
            resolution.source_workflow_stage.value
            if resolution.source_workflow_stage is not None
            else None
        )
        count = len(config.sweep_values)

        definition = (
            matching_sweep_definition(
                document,
                (self.active_definition(), *self.definitions()),
                parameters=parameters,
                input_structure_id=input_structure_id,
                output_directory=output_directory,
                simulation_cell_mode=simulation_cell_mode,
                supercell_source_workflow=supercell_source_workflow,
            )
            if self.reuse_matching_definition
            else None
        )
        if definition is not None:
            self.sync_mappings(definition)
            status = (
                f"Using existing {self.job_noun} setup with {plural(count, f'{self.job_noun} job')}"
            )
        else:
            definition = self.register_configured_definition(
                document,
                input_structure_id=input_structure_id,
                parameters=parameters,
                output_directory=output_directory,
                simulation_cell_mode=simulation_cell_mode,
                supercell_source_workflow=supercell_source_workflow,
            )
            status = (
                f"Configured {count} {self.job_noun} jobs; "
                "input files will be generated on submission"
            )
        document.ui_state[self.active_key] = str(definition.id)

        self._deps.refresh_jobs_table()
        self.refresh_view()
        self._deps.refresh_journal_view()
        self._context.status_message(status)

    # --- simulation cell ---

    def show_simulation_cell_structure(self, structure: Structure | None) -> None:
        """Pass the sweep's resolved cell to the widget; ``None`` when there is none.

        The selector itself needs only the options
        :meth:`refresh_simulation_cell` sets, so this does nothing unless a
        sweep's widget needs the cell too.
        """

    def refresh_simulation_cell(self) -> None:
        """Show the sweep's simulation-cell choices and the species its cell holds."""

        document = self._context.document()
        if document is None:
            self._view.set_simulation_cell_options(
                summary="Load a host structure to choose a simulation cell.",
                assets=[],
                selected_structure_id=None,
                enabled=False,
            )
            self._view.set_parameter_species_options(())
            self.show_simulation_cell_structure(None)
            return

        resolution = self._context.project_service.resolve_workflow_structure(
            document,
            self.stage,
            normalize_state=True,
        )
        assets = [
            {
                "structure_id": str(asset.structure_id),
                "label": asset.label,
            }
            for asset in resolution.available_assets
        ]
        selected_structure_id = (
            str(resolution.normalized_selection.structure_id)
            if resolution.normalized_selection.structure_id is not None
            else None
        )
        self._view.set_simulation_cell_options(
            summary=resolution.status_message,
            assets=assets,
            selected_structure_id=selected_structure_id,
            enabled=resolution.structure is not None,
        )
        self._view.set_parameter_species_options(structure_species_options(resolution.structure))
        self.show_simulation_cell_structure(resolution.structure)

    def supercell_initial_matrix(self) -> list[list[int]]:
        """Return the matrix the Supercell Editor opens on for this sweep.

        That is the selected saved supercell's matrix, else the project's
        supercell matrix, else the identity when no project is open.
        """

        document = self._context.document()
        if document is None:
            return [list(row) for row in IDENTITY_SUPERCELL]

        resolution = self._context.project_service.resolve_workflow_structure(
            document,
            self.stage,
            normalize_state=True,
        )
        if resolution.selected_asset is not None:
            return [list(row) for row in resolution.selected_asset.matrix]
        return [list(row) for row in document.project.supercell_matrix]

    def create_simulation_cell(self) -> None:
        """Pick or create the sweep's simulation cell in the Supercell Editor."""

        self._deps.select_simulation_cell_in_editor(
            self.stage,
            initial_matrix=self.supercell_initial_matrix,
            on_selected=self._use_simulation_cell_from_editor,
        )

    def _use_simulation_cell_from_editor(self, structure_id: UUID) -> None:
        document = self._context.document()
        if document is None:
            return
        selected = self._context.project_service.use_workflow_supercell_asset(
            document,
            workflow_stage=self.stage,
            asset_structure_id=structure_id,
        )
        self._show_selected_simulation_cell(selected)

    def _show_selected_simulation_cell(self, selected: Structure) -> None:
        self.refresh_simulation_cell()
        self._context.status_message(
            f"Selected {self.simulation_cell_name} simulation cell '{selected.label}'"
        )

    def select_simulation_cell(self, payload: object) -> None:
        """Use the saved supercell ``payload`` names, or the unit cell when it names none."""

        document = self._context.document()
        if document is None:
            return

        structure_id = None
        if payload is not None:
            try:
                structure_id = UUID(str(payload))
            except ValueError:
                structure_id = None

        if structure_id is None:
            self._context.project_service.clear_workflow_supercell_selection(document, self.stage)
            self.refresh_simulation_cell()
            return

        try:
            selected = self._context.project_service.use_workflow_supercell_asset(
                document,
                workflow_stage=self.stage,
                asset_structure_id=structure_id,
            )
        except ValueError as exc:
            self._deps.show_warning("Simulation cell unavailable", str(exc))
            self.refresh_simulation_cell()
            return

        self._show_selected_simulation_cell(selected)

    def use_unit_cell(self) -> None:
        """Run the sweep on the unit cell again."""

        self.select_simulation_cell(None)


__all__ = ["SweepConvergenceController"]
