"""Hyperfine Coupling workflow controller.

Owns the hyperfine workflow: each relaxed muon source runs as one CASTEP
``TASK : MAGRES`` / ``MAGRES_TASK : HYPERFINE`` job.

- **Sources and rows.** Rows are the eligible relaxed muon sites from
  :mod:`mufinder.services.source_candidates`, scoped to the pinned sites like
  Quantum Corrections. A row is bound to its job through the definition's
  ``job_ids_by_label``; its state is that job's projected status, or ``ready``.
- **Materialize.** :meth:`HyperfineCouplingController.materialize` builds a
  config per ready row, expands them through ``expand_hyperfine_jobs`` and
  writes each job into ``<output>/<job label>``.
- **Result.** :meth:`HyperfineCouplingController.validate_artifacts` fails a job
  whose ``.magres`` is missing or unreadable;
  :meth:`HyperfineCouplingController.build_result` stores the coupling, the
  pseudo spin density (a warning when the ``.den_fmt`` could not be read) and
  the contact-shell inputs Magnetic Fields needs, from the source relaxation's
  Mulliken spins.
- **Configure.** The Configure dialog edits the row's electronic settings; the
  workflow-controlled parameters (task, magres task, spin polarisation, density
  output) are not offered. It is editable unless a selected row's job is
  active, or is a completed job whose inputs are missing.
- **Changing a completed row.** A completed row runs again with other
  parameters through the shared route of
  :class:`~mufinder.app.workflow_controllers.completed_row_edits.CompletedRowEditsController`:
  changing it asks first, the row reads ``edited`` and keeps its hyperfine
  result until it runs again, and the result goes once that submission is
  accepted. A coupling already handed to a Magnetic Fields form is that form's
  input and stays as it is.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from uuid import UUID

from mufinder.castep_io.input_diff import JobInputTexts

from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.charge_states import muon_charge_state_label
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result
from mufinder.services.hyperfine_results import (
    NO_MULLIKEN_SPINS_WARNING,
    contact_shell_inputs,
    handoff_unavailable_reason,
    hyperfine_result_payload,
    hyperfine_results,
    mulliken_spins_in_structure_order,
)
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.job_results import parse_numeric
from mufinder.services.muon_sites_summary import (
    HYPERFINE_COUPLING_SCOPE,
    DownstreamSiteScope,
    downstream_site_scope,
    set_show_every_relaxed_site,
)
from mufinder.services.source_candidates import (
    QuantumSourceCandidate,
    charge_state_for_job,
    charge_state_label_for_job,
    eligible_quantum_source_candidates,
    quantum_source_candidate_by_key,
    quantum_source_key,
    quantum_source_label,
    serialize_quantum_sources,
    source_relaxation_parameter_values,
)
from mufinder.workflow import StructureBridgeError, render_job, structure_to_spec
from mufinder.workflow.hyperfine import (
    HYPERFINE_CALC_TYPE,
    HYPERFINE_CONTROLLED_PARAM_KEYWORDS,
    HYPERFINE_SET_CALC_TYPE,
    HyperfineJobConfig,
    expand_hyperfine_jobs,
)
from mufinder.workflow.quantum import QuantumPhononSource
from mufinder.workflow.types import JobSpec

from mufinder.app.adapter_common import build_workflow_settings_template
from mufinder.app.hyperfine_coupling_widget import (
    HyperfineCouplingWidget,
    HyperfineResultRow,
    HyperfineSourceRow,
)
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.app.workflow_config_codecs import (
    source_row_parameter_values,
    store_workflow_row_configuration,
    workflow_row_configuration,
    workflow_row_configuration_exists,
)
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    confirm_active_host,
    dedupe_jobs,
    output_predates_run,
    locked_configuration_reason,
)
from mufinder.app.workflow_controllers.completed_row_edits import CompletedRowEditsController
from mufinder.app.workflow_dialogs import (
    HyperfineCouplingConfigDialog,
    SourceConfigureRequest,
    SourceConfigureResult,
    run_source_configure_dialog,
)
from mufinder.app.workflow_parameters import (
    build_workflow_parameter_surface,
    merge_species_options,
    workflow_basic_parameter_names,
)
from mufinder.app.workflow_submission import (
    EDITED_ROW_STATUS,
    effective_row_status,
)

_CONFIG_KEYWORD_NAMES = workflow_basic_parameter_names()
_DEFAULT_ELEC_ENERGY_TOL = ValueWithUnit(1e-9, "eV")
_JOB_MAPPING = "job_ids_by_label"
_JOB_NAME_PREFIX = "hyperfine"


@dataclass(frozen=True)
class HyperfineCouplingDeps:
    """The window-side handles the hyperfine controller needs.

    The host-mismatch handlers ask whether to switch the active host (shared
    with every host-guarded workflow).
    """

    handle_structure_host_mismatch: Callable[[object], bool]
    handle_site_host_mismatch: Callable[[object], bool]


class HyperfineCouplingController(CompletedRowEditsController):
    """Controller for the Hyperfine Coupling workflow: one magres job per row."""

    calc_types = frozenset({HYPERFINE_CALC_TYPE})
    planned_calc_type = HYPERFINE_CALC_TYPE
    output_folder = "hyperfine_coupling"
    stage = WorkflowStage.HYPERFINE_COUPLING
    definition_calc_type = HYPERFINE_SET_CALC_TYPE
    no_ready_message = (
        "Only eligible muon sites that are ready, failed, stopped or cancelled can be submitted."
    )
    fixed_keyword_names = HYPERFINE_CONTROLLED_PARAM_KEYWORDS
    site_scope = HYPERFINE_COUPLING_SCOPE
    result_noun = "hyperfine result"
    result_noun_plural = "hyperfine results"
    replaced_result_detail = (
        "Running again deletes the hyperfine result and its row in Completed Results, and "
        "clears the job's folder. Until that run's submission is accepted, the row keeps its "
        "current result, and setting the parameters back keeps it for good. A coupling already "
        "used in Magnetic Fields stays as it is until 'Use in Magnetic Fields' is run again on "
        "the new result."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: HyperfineCouplingDeps,
        view: HyperfineCouplingWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- sources and jobs ---

    def candidates(self) -> list[QuantumSourceCandidate]:
        """Return the eligible relaxed muon sources, in table order."""

        return eligible_quantum_source_candidates(self._context.document())

    def candidate_by_key(self, key: str) -> QuantumSourceCandidate | None:
        return quantum_source_candidate_by_key(self._context.document(), key)

    def row_job(self, definition: CalculationDefinition | None, key: str) -> Job | None:
        """Return the row's magres job, if it has one."""

        return self.job_for_key(definition, key, calc_type=HYPERFINE_CALC_TYPE)

    def result_job_for_key(self, definition: CalculationDefinition | None, key: str) -> Job | None:
        """Return the row's magres job: running the row again replaces its result."""

        return self.row_job(definition, key)

    def definition_and_key_for_job(
        self, job: Job
    ) -> tuple[CalculationDefinition, str] | tuple[None, None]:
        """Return the definition and row key whose job mapping binds ``job``."""

        for definition in self.definitions():
            mapping = definition.parameters.get(_JOB_MAPPING, {})
            if not isinstance(mapping, dict):
                continue
            for key, value in mapping.items():
                if str(value) == str(job.id):
                    return definition, str(key)
        return None, None

    # --- configuration ---

    @staticmethod
    def parameter_specs() -> list[KeywordSpec]:
        """Return the Configure dialog's surface, without the workflow-controlled parameters."""

        specs = build_workflow_parameter_surface(
            workflow_stage="hyperfine_coupling",
            basic_names=_CONFIG_KEYWORD_NAMES,
            excluded_names=HYPERFINE_CONTROLLED_PARAM_KEYWORDS,
        )
        return [
            replace(spec, default_value=_DEFAULT_ELEC_ENERGY_TOL)
            if spec.name == "elec_energy_tol"
            else spec
            for spec in specs
        ]

    def runtime_configuration(
        self, definition: CalculationDefinition | None, key: str
    ) -> dict[str, object]:
        """Return a row's effective parameter values.

        The row's stored values sit over its source relaxation's settings; the
        parameters the workflow sets itself are dropped from what it inherits.
        """

        candidate = self.candidate_by_key(key)
        return source_row_parameter_values(
            definition,
            key,
            source_relaxation_parameter_values(candidate.job, self._context.job_output_directory)
            if candidate is not None
            else {},
            dropped=HYPERFINE_CONTROLLED_PARAM_KEYWORDS,
            defaults={"elec_energy_tol": _DEFAULT_ELEC_ENERGY_TOL},
        )

    def ensure_definition(
        self, source_keys: Sequence[str], *, hold_record: bool = False
    ) -> CalculationDefinition | None:
        """Return the active definition with the selected sources added, creating one if needed.

        With ``hold_record`` a new definition's "Configured …" record is held
        until the submission is accepted, as a Run's is.
        """

        document = self._context.document()
        if document is None:
            return None
        candidates = [
            candidate
            for key in source_keys
            if (candidate := self.candidate_by_key(key)) is not None
        ]
        if not candidates:
            return None
        definition = self.active_definition()
        if definition is None:
            return self._context.project_service.register_hyperfine_definition(
                document,
                input_structure_ids=list(
                    dict.fromkeys(candidate.structure.id for candidate in candidates)
                ),
                input_site_ids=list(dict.fromkeys(candidate.site.id for candidate in candidates)),
                parameters={
                    "sources": serialize_quantum_sources(
                        document, [candidate.key for candidate in candidates]
                    )
                },
                output_directory=document.project_root / "files" / "hyperfine_coupling",
                **({"hold_record": self._held_records.append} if hold_record else {}),
            )
        for candidate in candidates:
            if candidate.structure.id not in definition.input_structure_ids:
                definition.input_structure_ids.append(candidate.structure.id)
            if candidate.site.id not in definition.input_site_ids:
                definition.input_site_ids.append(candidate.site.id)
        config = definition.parameters.setdefault("configuration", {})
        if isinstance(config, dict):
            stored = config.get("sources")
            stored_keys = [
                str(entry.get("key"))
                for entry in (stored if isinstance(stored, list) else [])
                if isinstance(entry, dict) and entry.get("key")
            ]
            config["sources"] = serialize_quantum_sources(
                document, [*stored_keys, *(candidate.key for candidate in candidates)]
            )
        definition.parameters.setdefault(_JOB_MAPPING, {})
        return definition

    def configure_request(self, keys: Sequence[str]) -> SourceConfigureRequest | None:
        """Return what the Configure dialog opens with for the rows ``keys`` name."""

        if self._context.document() is None:
            return None
        candidates = [
            candidate for key in keys if (candidate := self.candidate_by_key(key)) is not None
        ]
        if not candidates:
            return None
        definition = self.active_definition()
        states = self.build_states()
        values_by_key: list[dict[str, object]] = []
        advanced_names: set[str] = set()
        for candidate in candidates:
            values_by_key.append(self.runtime_configuration(definition, candidate.key))
            _params, row_advanced_names, _extras = workflow_row_configuration(
                definition, candidate.key
            )
            advanced_names.update(row_advanced_names)
        configurable = {
            candidate.key: self.row_is_configurable(states, definition, candidate.key)
            for candidate in candidates
        }
        read_only = not all(configurable.values())
        locked_reason = (
            locked_configuration_reason(
                [
                    None if configurable[candidate.key] else self.row_job(definition, candidate.key)
                    for candidate in candidates
                ],
                [
                    str(getattr(states.get(candidate.key), "status", "ready"))
                    for candidate in candidates
                ],
                job_noun="magres job",
            )
            if read_only
            else ""
        )
        return SourceConfigureRequest(
            keys=tuple(candidate.key for candidate in candidates),
            row_values=tuple(values_by_key),
            advanced_names=frozenset(advanced_names),
            title=(
                "Inspect Hyperfine Coupling Parameters"
                if read_only
                else "Configure Hyperfine Coupling Parameters"
            ),
            intro_text=(
                f"Selected sources: {len(candidates)}\n"
                "Each runs as one spin-polarised TASK : MAGRES job with MAGRES_TASK : HYPERFINE."
                + (f"\n\n{locked_reason}" if locked_reason else "")
            ),
            read_only=read_only,
            num_farms=1,
            species_options=merge_species_options(candidate.structure for candidate in candidates),
        )

    def dialog_parent(self) -> object:
        """Return the window the Configure dialog and its question open over."""

        return self._view.window()

    def run_configure_dialog(self, request: SourceConfigureRequest) -> SourceConfigureResult | None:
        """Run the Hyperfine Coupling Configure dialog on ``request``."""

        return run_source_configure_dialog(
            partial(
                HyperfineCouplingConfigDialog,
                specs=self.parameter_specs(),
                intro_text=request.intro_text,
                read_only=request.read_only,
                species_options=request.species_options,
                parent=self.dialog_parent(),
            ),
            request,
        )

    def apply_configuration(
        self,
        keys: Sequence[str],
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> None:
        """Store an accepted Configure dialog on every row ``keys`` name.

        A completed row the dialog leaves as it was stores nothing; when no row
        changes, nothing is saved either.
        """

        unchanged = self.rows_unchanged_by(keys, result, extras)
        if unchanged >= set(keys):
            # Nothing to store, so nothing is written: the project stays byte for byte.
            return
        definition = self.ensure_definition(keys)
        if definition is None:
            return
        for key in keys:
            if key in unchanged:
                continue
            store_workflow_row_configuration(
                definition, key, self._row_payload(definition, key, result, extras)
            )
        self.refresh_view()
        self._context.persist_ui_state(
            failure_message="Hyperfine Coupling configuration was updated, but autosave failed."
        )

    def rename_source(self, key: str, label: str) -> bool:
        """Rename a source's muon site from its editable label cell; return whether it changed."""

        candidate = self.candidate_by_key(key)
        if candidate is None:
            return False
        normalized = str(label).strip()
        if not normalized or normalized == candidate.site.label:
            self.refresh_view()
            return False
        candidate.site.label = normalized
        self.refresh_view()
        return True

    def hyperfine_result(self, result_id: str) -> Result | None:
        """Return the completed hyperfine result ``result_id`` names, if any."""

        document = self._context.document()
        if document is None:
            return None
        try:
            parsed = UUID(str(result_id))
        except ValueError:
            return None
        result = document.results.get(parsed)
        if result is None or result.result_type != HYPERFINE_CALC_TYPE:
            return None
        return result

    def set_show_every_site(self, enabled: bool) -> None:
        """List every eligible relaxed site (``True``) or only the pinned ones."""

        document = self._context.document()
        if document is None:
            return
        set_show_every_relaxed_site(document, HYPERFINE_COUPLING_SCOPE, enabled)
        self.refresh_view()
        self._context.persist_ui_state(
            failure_message="The Hyperfine Coupling site list was updated, but autosave failed."
        )

    # --- submission contract ---

    def normalize(self, value: object) -> str:
        return str(value)

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Validate the selection before anything is materialized.

        A selection of only ``completed`` rows has nothing to run, and one
        notice says how to run such a row again (:meth:`nothing_to_run`).
        Otherwise the active definition's structures and sites must belong to
        the active host.
        """

        if self.nothing_to_run(keys):
            return False
        document = self._context.document()
        definition = self.active_definition()
        if document is None or definition is None:
            return True
        return confirm_active_host(
            self._context.project_service,
            document,
            structure_ids=definition.input_structure_ids,
            site_ids=definition.input_site_ids,
            handle_structure_mismatch=self._deps.handle_structure_host_mismatch,
            handle_site_mismatch=self._deps.handle_site_host_mismatch,
        )

    def build_states(self) -> dict[str, RelaxationRunState]:
        """Return each eligible row's state: its job's projected status, else ``ready``.

        A completed row whose parameters changed since its run
        (:meth:`parameters_changed_since_run`) reads ``edited``.
        """

        states: dict[str, RelaxationRunState] = {}
        definition = self.active_definition()
        for candidate in self.candidates():
            job = self.row_job(definition, candidate.key)
            if job is None:
                states[candidate.key] = RelaxationRunState(status="ready")
                continue
            edited = self.completed_row_is_edited(definition, candidate.key, job)
            states[candidate.key] = RelaxationRunState(
                status=EDITED_ROW_STATUS if edited else effective_row_status(job),
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
            )
        return states

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        """Return each row's job across every definition, once each."""

        if self._context.document() is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        for definition in self.definitions():
            for key in (str(key) for key in keys):
                job = self.row_job(definition, key)
                if job is None or job.id in seen:
                    continue
                selected.append(job)
                seen.add(job.id)
        return selected

    def describe_rows(
        self, keys: list[object], *, also: Collection[UUID] = ()
    ) -> tuple[CalculationDefinition | None, Path, list[tuple[str, JobSpec, Job | None]]] | None:
        """Return the active definition (``None`` before the first), the output folder and each row.

        Each row is ``(key, spec, magres job)``; see
        :meth:`~mufinder.app.workflow_controllers.completed_row_edits.CompletedRowEditsController.describe_source_rows`.
        """

        return self.describe_source_rows(
            keys,
            also=also,
            job_for_row=self.row_job,
            job_config=self._hyperfine_job_config,
            expand=lambda configs: expand_hyperfine_jobs(
                jobs=configs, name_prefix=_JOB_NAME_PREFIX
            ),
        )

    def plan_existing(self, job: Job, spec: JobSpec | None, output_root: Path) -> None:
        """Plan where a defined job is rewritten: under its source's label when that is free.

        A source's label can change as others come and go; the job follows it
        unless another job has that name, and then stays where it is. The
        window reads the new folder until the submission ends, so the dialogs
        and the cluster plan use the one the job runs in.
        """

        if job.status != JobStatus.DEFINED or spec is None or spec.name == job.label:
            return
        if spec.name not in self.reserved_names(output_root) and self.job_may_write_folder(
            job, output_root / spec.name
        ):
            self.plan_move(job, output_root / spec.name)

    def write_jobs(
        self,
        keys: list[object],
        *,
        going: Collection[UUID],
        run_again_ids: Collection[UUID] = (),
    ) -> list[Job]:
        """Create the planned magres jobs and rewrite the defined ones, of the rows ``going`` names.

        See :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.write_source_rows`.
        A rewritten job follows its source's label unless another job has that
        name, and then stays where it is; its new name is taken once it goes
        ahead (:meth:`when_going`).
        """

        written = self.write_source_rows(
            keys,
            going=going,
            run_again_ids=run_again_ids,
            rewrite_names=lambda job, spec: (spec.name, job.label),
        )
        if written is None:
            return []
        _definition, collected = written
        for _key, _spec, existing, name in collected.rewritten:
            if name != existing.label:
                self.when_going(existing, partial(setattr, existing, "label", name))
        self.refresh_view()
        return dedupe_jobs(collected.jobs)

    def materialize(self, keys: list[object]) -> list[Job]:
        """Create the rows' magres jobs now and save.

        See :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.materialize`.
        """

        jobs = super().materialize(keys)
        self.save_materialized(
            jobs, failure_message="Hyperfine Coupling jobs were generated, but autosave failed."
        )
        return jobs

    def _hyperfine_job_config(
        self, definition: CalculationDefinition | None, key: str
    ) -> HyperfineJobConfig | None:
        """Return the magres job config row ``key`` builds from its settings, if it builds."""

        document = self._context.document()
        candidate = self.candidate_by_key(key)
        if document is None or candidate is None:
            return None
        try:
            structure = structure_to_spec(candidate.structure)
            overrides = build_workflow_settings_template(
                {
                    name: value
                    for name, value in self.runtime_configuration(definition, key).items()
                    if name != "task"
                },
                workflow_controlled=HYPERFINE_CONTROLLED_PARAM_KEYWORDS,
            )
        except (StructureBridgeError, ValueError):
            return None
        return HyperfineJobConfig(
            source=QuantumPhononSource(
                label=quantum_source_label(document, candidate),
                structure=structure,
                source_site_id=candidate.site.id,
                source_relaxation_run_id=candidate.run.id,
                source_relaxation_job_id=candidate.job.id,
                charge_state=charge_state_for_job(candidate.job),
                total_energy_eV=candidate.total_energy_eV,
            ),
            cell_overrides=overrides.cell,
            param_overrides=overrides.param,
        )

    def row_inputs(
        self, definition: CalculationDefinition | None, key: str
    ) -> JobInputTexts | None:
        """Render the ``.cell`` and ``.param`` row ``key``'s job would be written with now.

        A pure preview of :meth:`materialize` for one row: nothing is stored or
        written. ``None`` without an eligible source or when the row's settings
        do not build.
        """

        config = self._hyperfine_job_config(definition, key)
        if config is None:
            return None
        try:
            (spec,) = expand_hyperfine_jobs(jobs=[config], name_prefix=_JOB_NAME_PREFIX)
        except (ValueError, StructureBridgeError):
            return None
        cell_text, param_text = render_job(spec)
        return JobInputTexts(cell=cell_text, param=param_text)

    # --- result contract ---

    def validate_artifacts(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> str | None:
        """Fail a hyperfine job without a readable ``.magres`` from this run.

        When CASTEP stopped before writing one, its own error (for example an
        unconverged ground state) is the reason given. A ``.magres`` older than
        the run's start was left by an earlier run and fails the job too.
        """

        if job.calc_type != HYPERFINE_CALC_TYPE:
            return None
        seed_name = self._context.job_seed_name(job)
        if artifacts.magres_missing:
            if parsed.error_message:
                return parsed.error_message
            return f"Expected CASTEP magres output was not found: {seed_name}.magres"
        if artifacts.magres_error is not None or artifacts.magres is None:
            return f"Failed to read CASTEP magres output: {artifacts.magres_error}"
        if output_predates_run(artifacts.magres_path, run.started_at or job.started_at):
            return (
                f"{seed_name}.magres predates this run: CASTEP wrote no hyperfine "
                "tensors, so the previous run's are still in place."
            )
        return None

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute the hyperfine payload for a completed magres job."""

        if job.calc_type != HYPERFINE_CALC_TYPE or artifacts.magres is None:
            return ControllerResult()
        warnings: list[str] = []
        pseudo_spin_density = artifacts.pseudo_spin_density_angstrom3
        if artifacts.spin_density_error is not None:
            warnings.append(
                "The pseudo spin density at the muon could not be read: "
                f"{artifacts.spin_density_error}"
            )
        density_path = (
            None if artifacts.magres_path is None else artifacts.magres_path.with_suffix(".den_fmt")
        )
        if pseudo_spin_density is not None and output_predates_run(
            density_path, run.started_at or job.started_at
        ):
            # A density left by an earlier run would not match this run's tensor.
            pseudo_spin_density = None
            warnings.append(
                f"{density_path.name} predates this run, so the pseudo spin density "
                "cross-check is left empty."
            )
        candidate = self._source_candidate_for_job(job)
        shell = None
        source: dict[str, str | None] = {
            "source_site_id": None,
            "source_relaxation_job_id": None,
            "relaxed_structure_id": None,
            "charge_state": None,
        }
        if candidate is not None:
            source = {
                "source_site_id": str(candidate.site.id),
                "source_relaxation_job_id": str(candidate.job.id),
                "relaxed_structure_id": str(candidate.structure.id),
                "charge_state": charge_state_for_job(candidate.job),
            }
            output_dir = self._context.job_output_directory(candidate.job)
            castep_path = (
                None
                if output_dir is None
                else output_dir / f"{self._context.job_seed_name(candidate.job)}.castep"
            )
            shell = contact_shell_inputs(
                candidate.structure,
                mulliken_spins_in_structure_order(candidate.structure, castep_path),
            )
        if shell is None:
            warnings.append(NO_MULLIKEN_SPINS_WARNING)
        payload = hyperfine_result_payload(
            artifacts.magres,
            pseudo_spin_density_angstrom3=pseudo_spin_density,
            shell=shell,
            source=source,
        )
        if artifacts.magres_path is not None:
            payload["magres_file"] = str(artifacts.magres_path)
        return ControllerResult(payload=payload, warnings=warnings)

    def _source_candidate_for_job(self, job: Job) -> QuantumSourceCandidate | None:
        _definition, key = self.definition_and_key_for_job(job)
        return None if key is None else self.candidate_by_key(key)

    # --- view contract ---

    def refresh_view(self) -> None:
        """Rebuild the hyperfine source and result tables."""

        document = self._context.document()
        if document is None:
            self._view.set_site_scope(show_every_site=False)
            self._view.set_sources([])
            self._view.set_results([])
            return

        states = self.build_states()
        candidates = self.candidates()
        definition = self.active_definition()
        # The pinned list keeps every row this workflow already tracks, so a
        # configured or submitted site never drops out of the table.
        tracked_keys = [
            candidate.key
            for candidate in candidates
            if self.row_job(definition, candidate.key) is not None
            or workflow_row_configuration_exists(definition, candidate.key)
        ]
        scope = downstream_site_scope(document, HYPERFINE_COUPLING_SCOPE, candidates)
        self._view.set_site_scope(show_every_site=scope.show_every_site, notice=scope.notice)
        source_rows = [
            HyperfineSourceRow(
                key=candidate.key,
                label=candidate.site.label or candidate.job.label,
                charge_state=charge_state_label_for_job(candidate.job),
                source_job=candidate.job.label,
                status=getattr(states.get(candidate.key), "status", "ready") or "ready",
                configured=workflow_row_configuration_exists(definition, candidate.key),
                pinned_label=scope.pinned_label(candidate.key),
            )
            for candidate in scope.scoped_candidates(candidates, keep_keys=tracked_keys)
        ]
        self._view.set_sources(source_rows)
        self._view.set_results(
            [_result_row(document, result, scope) for result in hyperfine_results(document)]
        )


def _document_entity(entities: Mapping[UUID, object], raw_id: object):
    """Return the entity ``raw_id`` names in ``entities``, or ``None``."""

    if not raw_id:
        return None
    try:
        return entities.get(UUID(str(raw_id)))
    except ValueError:
        return None


def _result_row(document, result: Result, scope: DownstreamSiteScope) -> HyperfineResultRow:
    """Return ``result``'s row, naming its site as the jobs table names the source."""

    data = result.data if isinstance(result.data, Mapping) else {}
    site_label = ""
    site = _document_entity(document.sites, data.get("source_site_id"))
    if site is not None:
        source_job = _document_entity(document.jobs, data.get("source_relaxation_job_id"))
        site_label = (
            scope.pinned_label(quantum_source_key(document, site, source_job)) or site.label
        )
    if not site_label:
        run = (
            document.runs.get(result.calculation_run_id)
            if result.calculation_run_id is not None
            else None
        )
        job = document.jobs.get(run.job_id) if run is not None and run.job_id else None
        site_label = job.label if job is not None else "Hyperfine"
    charge_state = data.get("charge_state")
    contact = parse_numeric(data.get("contact_field_T"))
    pseudo = parse_numeric(data.get("pseudo_contact_field_T"))
    dipolar = data.get("dipolar_principal_mhz")
    return HyperfineResultRow(
        key=str(result.id),
        site=site_label,
        charge_state=(
            muon_charge_state_label(charge_state) if isinstance(charge_state, str) else ""
        ),
        isotropic=_number(data.get("isotropic_mhz"), "{:.2f}"),
        contact_field=_number(contact, "{:.4f}"),
        pseudo_contact_field=_number(pseudo, "{:.4f}"),
        ratio=(
            ""
            if contact is None or pseudo is None or abs(pseudo) < 1e-12
            else f"{contact / pseudo:.3f}"
        ),
        dipolar=(
            ", ".join(_number(value, "{:.2f}") for value in dipolar)
            if isinstance(dipolar, list)
            else ""
        ),
        handoff_reason=handoff_unavailable_reason(result),
    )


def _number(value: object, template: str) -> str:
    number = parse_numeric(value)
    return "" if number is None else template.format(number)


__all__ = ["HyperfineCouplingController", "HyperfineCouplingDeps"]
