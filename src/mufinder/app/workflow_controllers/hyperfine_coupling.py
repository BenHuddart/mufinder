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
  output) are not offered.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import UUID

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
    downstream_site_scope,
    set_show_every_relaxed_site,
)
from mufinder.services.source_candidates import (
    QuantumSourceCandidate,
    charge_state_for_job,
    charge_state_label_for_job,
    eligible_quantum_source_candidates,
    quantum_source_candidate_by_key,
    quantum_source_label,
    serialize_quantum_sources,
    source_relaxation_parameter_values,
)
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory
from mufinder.workflow.hyperfine import (
    HYPERFINE_CALC_TYPE,
    HYPERFINE_CONTROLLED_PARAM_KEYWORDS,
    HYPERFINE_SET_CALC_TYPE,
    HyperfineJobConfig,
    expand_hyperfine_jobs,
)
from mufinder.workflow.quantum import QuantumPhononSource

from mufinder.app.adapter_common import build_workflow_settings_template
from mufinder.app.hyperfine_coupling_widget import (
    HyperfineCouplingWidget,
    HyperfineResultRow,
    HyperfineSourceRow,
)
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.app.workflow_config_codecs import (
    batch_row_payload,
    row_settings_payload,
    store_workflow_row_configuration,
    workflow_row_configuration,
    workflow_row_configuration_exists,
)
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    confirm_active_host,
    dedupe_jobs,
    output_predates_run,
    locked_configuration_reason,
)
from mufinder.app.workflow_dialogs import SourceConfigureRequest, SourceConfigureResult
from mufinder.app.workflow_parameters import (
    build_workflow_parameter_surface,
    merge_species_options,
    workflow_basic_parameter_names,
)
from mufinder.app.workflow_submission import effective_row_status, workflow_row_state_is_ready

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


class HyperfineCouplingController(WorkflowJobController):
    """Controller for the Hyperfine Coupling workflow: one magres job per row."""

    calc_types = frozenset({HYPERFINE_CALC_TYPE})
    stage = WorkflowStage.HYPERFINE_COUPLING
    definition_calc_type = HYPERFINE_SET_CALC_TYPE
    no_ready_message = "Only eligible muon sites that have not yet been submitted can be submitted."

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

        params, _advanced, _extras = workflow_row_configuration(definition, key)
        candidate = self.candidate_by_key(key)
        values = (
            source_relaxation_parameter_values(candidate.job, self._context.job_output_directory)
            if candidate is not None
            else {}
        )
        values.update(params)
        for name in HYPERFINE_CONTROLLED_PARAM_KEYWORDS:
            values.pop(name, None)
        values.setdefault("elec_energy_tol", _DEFAULT_ELEC_ENERGY_TOL)
        return values

    def ensure_definition(self, source_keys: Sequence[str]) -> CalculationDefinition | None:
        """Return the active definition with the selected sources added, creating one if needed."""

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
        read_only = any(
            not workflow_row_state_is_ready(states, candidate.key) for candidate in candidates
        )
        locked_reason = (
            locked_configuration_reason(
                [self.row_job(definition, candidate.key) for candidate in candidates],
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

    def apply_configuration(self, keys: Sequence[str], result: SourceConfigureResult) -> None:
        """Store an accepted Configure dialog on every row ``keys`` name."""

        payload = row_settings_payload(result.values, result.advanced_names, {})
        definition = self.ensure_definition(keys)
        if definition is None:
            return
        for key in keys:
            store_workflow_row_configuration(
                definition, key, batch_row_payload(definition, key, payload, result.untouched_mixed)
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
        """Check the active definition's structures and sites belong to the active host."""

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
        """Return each eligible row's state: its job's projected status, else ``ready``."""

        states: dict[str, RelaxationRunState] = {}
        definition = self.active_definition()
        for candidate in self.candidates():
            job = self.row_job(definition, candidate.key)
            if job is None:
                states[candidate.key] = RelaxationRunState(status="ready")
                continue
            states[candidate.key] = RelaxationRunState(
                status=effective_row_status(job),
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

    def materialize(self, keys: list[object]) -> list[Job]:
        """Write the magres job of each ready row ``keys`` names, creating it when needed.

        A row's existing job is reused and, while it is ``defined``, its inputs
        are rewritten from the row's configuration.
        """

        document = self._context.document()
        if document is None:
            return []
        str_keys = [str(key) for key in keys]
        definition = self.ensure_definition(str_keys)
        if definition is None:
            return []
        output_root = self._context.definition_output_directory(definition)
        if output_root is None:
            return []

        states = self.build_states()
        configs: list[HyperfineJobConfig] = []
        config_keys: list[str] = []
        for key in str_keys:
            candidate = self.candidate_by_key(key)
            if candidate is None or not workflow_row_state_is_ready(states, key):
                continue
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
                continue
            configs.append(
                HyperfineJobConfig(
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
            )
            config_keys.append(key)
        if not configs:
            return []
        try:
            specs = expand_hyperfine_jobs(jobs=configs, name_prefix=_JOB_NAME_PREFIX)
        except (ValueError, StructureBridgeError):
            return []

        mapping = definition.parameters.setdefault(_JOB_MAPPING, {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters[_JOB_MAPPING] = mapping
        created: list[Job] = []
        for key, spec in zip(config_keys, specs, strict=True):
            existing = self.row_job(definition, key)
            if existing is not None:
                if existing.status == JobStatus.DEFINED:
                    existing.label = spec.name
                    write_job_directory(spec, output_root)
                created.append(existing)
                continue
            write_job_directory(spec, output_root)
            job = self._context.project_service.add_job(
                document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type=HYPERFINE_CALC_TYPE,
            )
            mapping[key] = str(job.id)
            created.append(job)

        jobs = dedupe_jobs(created)
        if jobs:
            self.refresh_view()
            self._context.persist_ui_state(
                failure_message="Hyperfine Coupling jobs were generated, but autosave failed."
            )
        return jobs

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
        if artifacts.spin_density_error is not None:
            warnings.append(
                "The pseudo spin density at the muon could not be read: "
                f"{artifacts.spin_density_error}"
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
            pseudo_spin_density_angstrom3=artifacts.pseudo_spin_density_angstrom3,
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
            [_result_row(document, result) for result in hyperfine_results(document)]
        )


def _result_row(document, result: Result) -> HyperfineResultRow:
    data = result.data if isinstance(result.data, Mapping) else {}
    site_label = ""
    site_id = data.get("source_site_id")
    if site_id:
        try:
            site = document.sites.get(UUID(str(site_id)))
        except ValueError:
            site = None
        site_label = site.label if site is not None else ""
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
