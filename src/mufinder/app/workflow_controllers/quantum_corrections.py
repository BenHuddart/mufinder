"""Quantum-corrections workflow controller.

Owns the submit → materialize → result → refresh cycle for the two-stage CASTEP
quantum-corrections workflow. Each relaxed muon source runs as a
*singlepoint → phonon* pair: a stage-1 singlepoint job, then a stage-2 phonon
continuation. This is the richest migration onto
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`:

- **Two-stage materialize.** :meth:`QuantumCorrectionsController.materialize`
  dispatches each selected source to its current stage
  (:data:`QUANTUM_STAGE_SINGLEPOINT` / :data:`QUANTUM_STAGE_PHONON`) via the
  deps-provided ``quantum_stage_state`` helper, builds singlepoint *and* phonon
  configs, expands them through the workflow-layer
  ``expand_quantum_singlepoint_jobs`` / ``expand_quantum_phonon_jobs``, and writes
  the two per-stage job-id mappings (``singlepoint_job_ids_by_label`` /
  ``phonon_job_ids_by_label``) plus the shared ``job_seed_names_by_label`` — a
  behavior-preserving move of the former
  ``MainWindow._materialize_quantum_corrections_jobs``.
- **Pre-submit host check.**
  :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.pre_submit_check`
  validates the active definition against the current host structure/sites (the
  former ``_check_definition_host`` guard, shared with the NEB / energy-barriers
  workflow and therefore reached via :class:`QuantumCorrectionsDeps`).
- **Richest result parsing.** :meth:`QuantumCorrectionsController.build_result`
  parses the completed ``.phonon`` file and contributes the muon ZPE, phonon
  eigenvalues, mode participations, and the ``w1`` entanglement witness for a
  quantum-*phonon* job; a quantum-*singlepoint* job contributes nothing (an empty
  :class:`~mufinder.app.workflow_controllers.base.ControllerResult`, a strict
  no-op). Following the success-path-only convention, the failure guards
  (missing/unparseable ``.phonon``, plus any downstream
  :class:`~mufinder.science.quantum_phonons.PhononParseError` from the ZPE / Gamma
  / witness computation) stay inline in ``MainWindow._on_local_job_finished`` and
  abort the job before this controller's ``build_result`` runs; ``build_result``
  therefore re-parses and recomputes on the validated success path.

- **Task farming is NOT this controller's concern.** The stage-2 phonon jobs are
  task-farmed, but that is applied by ``MainWindow._submit_jobs_locally`` in the
  shared submit backend (``task_farming_for_job=self._quantum_task_farming_for_job``,
  which filters by job) for *all* workflows. This controller uses the base
  single-group :meth:`submission_groups` default and does not touch task farming.

Both quantum calc types are registered on :attr:`calc_types` so
:meth:`handles_job` is coherent across the two-stage pair; the singlepoint branch
of ``build_result`` returns an empty result, so "handling" a singlepoint job is
byte-identical to not handling it.

Shared helpers that non-migrating code still calls (the definition/config/stage
helpers, shared with the configure path, source-label rename, stage reset, and the
NEB / energy-barriers workflow) remain on ``MainWindow`` and are reached via
:class:`QuantumCorrectionsDeps`, following the shared-context + per-controller-deps
+ view pattern documented in :mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job
from mufinder.science.quantum_phonons import (
    muon_zpe,
    parse_phonon_path,
    w1_entanglement_witness,
)
from mufinder.workflow import StructureBridgeError, structure_to_spec
from mufinder.workflow.quantum import (
    QUANTUM_PHONON_CALC_TYPE,
    QUANTUM_SINGLEPOINT_CALC_TYPE,
    QuantumPhononJobConfig,
    QuantumPhononSource,
    expand_quantum_phonon_jobs,
    expand_quantum_singlepoint_jobs,
)
from mufinder.workflow.types import InputTemplate, JobSpec

from mufinder.app.quantum_corrections_widget import (
    QuantumCorrectionsWidget,
    QuantumResultRow,
    QuantumSourceRow,
)
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
)

#: Stage identifiers for a two-stage quantum-corrections source row. Defined here
#: as the single source of truth; ``main_window`` imports them back (app→app is
#: allowed) so the controller never needs to import ``main_window``.
QUANTUM_STAGE_SINGLEPOINT = "singlepoint"
QUANTUM_STAGE_PHONON = "phonon"


def _format_phonon_eigenvalues(value: object) -> str:
    """Format a list of gamma-point eigenvalues as ``"1:v1, 2:v2, ..."``.

    Behavior-preserving move of the former module-level
    ``main_window._format_phonon_eigenvalues`` (its only caller was the quantum
    view refresh, which now lives in :meth:`QuantumCorrectionsController.refresh_view`).
    """

    if not isinstance(value, list):
        return ""
    entries: list[str] = []
    for mode_index, eigenvalue in enumerate(value, start=1):
        try:
            entries.append(f"{mode_index}:{float(eigenvalue):.1f}")
        except (TypeError, ValueError):
            continue
    return ", ".join(entries)


@dataclass(frozen=True)
class QuantumCorrectionsDeps:
    """Quantum-corrections-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating code still calls
    them: the definition/stage/config helpers are shared with the quantum
    configure path, source-label rename, and stage-one reset; ``check_definition_host``
    is shared with the NEB / energy-barriers workflow; the source-candidate and
    row-configuration helpers are shared across the quantum surface and the energy
    barriers path; and ``job_output_directory`` / ``job_seed_name`` /
    ``definition_output_directory`` / ``dedupe_jobs`` / ``format_energy`` are shared
    across workflows. ``input_template_from_keyword_values`` wraps the shared
    ``main_window`` module-level keyword→template helper. The controller reaches
    them only through these required callables. Workflow-specific dependencies like
    these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`.
    """

    active_quantum_corrections_definition: Callable[[], CalculationDefinition | None]
    ensure_quantum_corrections_definition: Callable[
        [list[str]], CalculationDefinition | None
    ]
    definition_output_directory: Callable[[CalculationDefinition], Path | None]
    eligible_quantum_source_candidates: Callable[[], list[object]]
    quantum_source_candidate_by_key: Callable[[str], object | None]
    quantum_stage_state: Callable[[CalculationDefinition | None, object], object]
    build_quantum_corrections_states: Callable[[], Mapping[str, object]]
    jobs_for_quantum_source_keys: Callable[[list[str]], list[Job]]
    singlepoint_job_for_quantum_source_key: Callable[
        [CalculationDefinition, str], Job | None
    ]
    phonon_job_for_quantum_source_key: Callable[
        [CalculationDefinition, str], Job | None
    ]
    quantum_seed_name_for_key: Callable[[CalculationDefinition, str], str]
    quantum_runtime_configuration: Callable[
        [CalculationDefinition, str], tuple[dict[str, object], str]
    ]
    workflow_row_configuration: Callable[
        [CalculationDefinition | None, str],
        tuple[dict[str, object], tuple[str, ...], dict[str, object]],
    ]
    workflow_row_configuration_exists: Callable[
        [CalculationDefinition | None, str], bool
    ]
    quantum_num_farms_from_extras: Callable[[Mapping[str, object]], int]
    quantum_source_label: Callable[[object], str]
    quantum_method_from_gui_value: Callable[[object], str]
    charge_state_for_job: Callable[[Job | None], str | None]
    charge_state_label_for_job: Callable[[Job | None], str]
    quantum_can_reset_to_stage_one: Callable[
        [CalculationDefinition | None, object], bool
    ]
    input_template_from_keyword_values: Callable[[Mapping[str, object]], InputTemplate]
    write_quantum_job_directory: Callable[..., Path]
    dedupe_jobs: Callable[[list[Job]], list[Job]]
    check_definition_host: Callable[[CalculationDefinition | None], bool]
    job_output_directory: Callable[[Job], Path | None]
    job_seed_name: Callable[[Job], str]
    format_energy: Callable[[object], str]


class QuantumCorrectionsController(WorkflowJobController):
    """Controller for the two-stage quantum-corrections workflow."""

    calc_types = frozenset(
        {QUANTUM_SINGLEPOINT_CALC_TYPE, QUANTUM_PHONON_CALC_TYPE}
    )
    no_ready_message = (
        "Only eligible relaxed muon sites in the 'ready' state can be submitted."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: QuantumCorrectionsDeps,
        view: QuantumCorrectionsWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready)
    # exactly mirror the former ``_on_submit_*_from_quantum_corrections`` handlers.
    # There is no submission_groups override: task farming is applied by the shared
    # submit backend, not here.

    def normalize(self, value: object) -> str:
        return str(value)

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Validate the active definition against the current host.

        Behavior-preserving move of the ``_check_definition_host`` guard the
        former quantum submit handlers ran before building a submission plan. The
        guard is shared with the NEB / energy-barriers workflow, so it stays on
        ``MainWindow`` and is reached via the deps. A refusal is silent.
        """

        return self._deps.check_definition_host(
            self._deps.active_quantum_corrections_definition()
        )

    def build_states(self) -> Mapping[str, object]:
        return self._deps.build_quantum_corrections_states()

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        return self._deps.jobs_for_quantum_source_keys([str(key) for key in keys])

    def materialize(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        str_keys = [str(key) for key in keys]
        definition = self._deps.ensure_quantum_corrections_definition(str_keys)
        if definition is None:
            return []
        output_root = self._deps.definition_output_directory(definition)
        if output_root is None:
            return []

        singlepoint_configs: list[QuantumPhononJobConfig] = []
        phonon_configs: list[QuantumPhononJobConfig] = []
        for key in str_keys:
            candidate = self._deps.quantum_source_candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self._deps.quantum_stage_state(definition, candidate)
            if stage_state.status != "ready" and (
                stage_state.current_job is None
                or stage_state.current_job.status != JobStatus.DEFINED
            ):
                continue
            try:
                structure = structure_to_spec(candidate.structure)
            except StructureBridgeError:
                continue
            values, pseudopotential_family = self._deps.quantum_runtime_configuration(
                definition, key
            )
            _params, _advanced, extras = self._deps.workflow_row_configuration(
                definition, key
            )
            values = {name: value for name, value in values.items() if name != "task"}
            method_value = values.pop("phonon_method", "FINITEDISPLACEMENT")
            try:
                method = self._deps.quantum_method_from_gui_value(method_value)
                overrides = self._deps.input_template_from_keyword_values(values)
            except ValueError:
                continue
            config = QuantumPhononJobConfig(
                source=QuantumPhononSource(
                    label=self._deps.quantum_source_label(candidate),
                    structure=structure,
                    source_site_id=candidate.site.id,
                    source_relaxation_run_id=candidate.run.id,
                    source_relaxation_job_id=candidate.job.id,
                    charge_state=self._deps.charge_state_for_job(candidate.job),
                    relative_energy_eV=candidate.site.relative_energy_eV,
                ),
                method=method,
                pseudopotential_family=str(pseudopotential_family),
                num_farms=self._deps.quantum_num_farms_from_extras(extras),
                cell_overrides=overrides.cell,
                param_overrides=overrides.param,
            )
            if stage_state.stage == QUANTUM_STAGE_SINGLEPOINT:
                singlepoint_configs.append(config)
            else:
                phonon_configs.append(config)
        if not singlepoint_configs and not phonon_configs:
            return []

        singlepoint_specs: list[JobSpec] = []
        phonon_specs: list[JobSpec] = []
        try:
            if singlepoint_configs:
                singlepoint_specs = expand_quantum_singlepoint_jobs(
                    jobs=singlepoint_configs,
                    name_prefix="quantum_singlepoint",
                )
            if phonon_configs:
                phonon_specs = expand_quantum_phonon_jobs(
                    jobs=phonon_configs,
                    name_prefix="quantum_phonon",
                )
        except (ValueError, StructureBridgeError):
            return []

        singlepoint_mapping = definition.parameters.setdefault(
            "singlepoint_job_ids_by_label", {}
        )
        if not isinstance(singlepoint_mapping, dict):
            singlepoint_mapping = {}
            definition.parameters["singlepoint_job_ids_by_label"] = singlepoint_mapping
        phonon_mapping = definition.parameters.setdefault("phonon_job_ids_by_label", {})
        if not isinstance(phonon_mapping, dict):
            phonon_mapping = {}
            definition.parameters["phonon_job_ids_by_label"] = phonon_mapping
        seed_mapping = definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(seed_mapping, dict):
            seed_mapping = {}
            definition.parameters["job_seed_names_by_label"] = seed_mapping
        # Load-bearing: ``_phonon_job_for_quantum_source_key`` falls back to
        # reading ``job_ids_by_label``, so keep the phonon mapping aliased there.
        definition.parameters["job_ids_by_label"] = phonon_mapping

        singlepoint_specs_by_key = {
            str(spec.provenance.get("source_site_id", "")): spec
            for spec in singlepoint_specs
        }
        phonon_specs_by_key = {
            str(spec.provenance.get("source_site_id", "")): spec
            for spec in phonon_specs
        }
        created: list[Job] = []
        for key in str_keys:
            candidate = self._deps.quantum_source_candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self._deps.quantum_stage_state(definition, candidate)
            seed_name = self._deps.quantum_seed_name_for_key(definition, key)
            seed_mapping[key] = seed_name
            if stage_state.stage == QUANTUM_STAGE_SINGLEPOINT:
                spec = singlepoint_specs_by_key.get(key)
                existing = self._deps.singlepoint_job_for_quantum_source_key(
                    definition, key
                )
                mapping = singlepoint_mapping
                calc_type = QUANTUM_SINGLEPOINT_CALC_TYPE
            else:
                spec = phonon_specs_by_key.get(key)
                existing = self._deps.phonon_job_for_quantum_source_key(definition, key)
                mapping = phonon_mapping
                calc_type = QUANTUM_PHONON_CALC_TYPE
            if spec is None:
                continue
            if existing is not None:
                if existing.status == JobStatus.DEFINED:
                    existing.label = spec.name
                    self._deps.write_quantum_job_directory(
                        spec, output_root, seed_name=seed_name
                    )
                created.append(existing)
                continue
            self._deps.write_quantum_job_directory(
                spec, output_root, seed_name=seed_name
            )
            job = self._context.project_service.add_job(
                document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type=calc_type,
            )
            mapping[key] = str(job.id)
            created.append(job)

        jobs = self._deps.dedupe_jobs(created)
        if jobs:
            self.refresh_view()
            self._context.persist_ui_state(
                failure_message=(
                    "Quantum Corrections jobs were generated, but autosave failed."
                )
            )
        return jobs

    # --- result contract ---

    def build_result(
        self, job: Job, parsed: CastepResult, run: CalculationRun
    ) -> ControllerResult:
        """Contribute the phonon/ZPE payload for a completed quantum-phonon job.

        A quantum-*singlepoint* job contributes nothing (an empty
        :class:`~mufinder.app.workflow_controllers.base.ControllerResult`, a strict
        no-op): its result is the base payload only, and stage advancement to the
        phonon continuation happens in ``quantum_stage_state`` during the next
        materialize/refresh, not here.

        For a quantum-*phonon* job, the inline ``_on_local_job_finished`` guard has
        already verified the ``.phonon`` file exists and that the full ZPE / Gamma /
        witness computation succeeds (aborting the job on any
        :class:`~mufinder.science.quantum_phonons.PhononParseError`), so this
        re-parse and recompute on the validated success path cannot hit those
        failure branches. The ``.phonon`` file itself is **not** contributed to
        ``output_files``: the call site already passes ``_output_files_for_job(job)``.
        """

        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return ControllerResult()

        output_dir = self._deps.job_output_directory(job)
        seed_name = self._deps.job_seed_name(job)
        if output_dir is None:
            return ControllerResult()
        phonon_path = output_dir / f"{seed_name}.phonon"

        phonon = parse_phonon_path(phonon_path)
        zpe = muon_zpe(phonon)
        payload: dict[str, object] = {
            "phonon_file": str(phonon_path),
            "gamma_only": len(phonon.qpoints) == 1 and phonon.qpoints[0].is_gamma,
            "phonon_mode_eigenvalues_cm": [
                float(value) for value in phonon.gamma_point().frequencies_cm
            ],
            "muon_zpe_total_eV": zpe.total_eV,
            "muon_zpe_modes": [
                {
                    "mode_index": mode.mode_index,
                    "frequency_cm": mode.frequency_cm,
                    "participation": mode.participation,
                    "zpe_eV": mode.zpe_eV,
                }
                for mode in zpe.modes
            ],
            "w1_entanglement_witness": w1_entanglement_witness(phonon),
        }
        if zpe.warnings:
            payload["phonon_warnings"] = list(zpe.warnings)
        return ControllerResult(payload=payload)

    # --- view contract ---

    def refresh_view(self) -> None:
        """Rebuild the quantum-corrections source and result tables.

        Behavior-preserving move of ``MainWindow._refresh_quantum_corrections_view``.
        Unlike the sibling controllers whose widgets build rows from
        ``set_runtime_states``, the quantum widget takes pre-built
        :class:`~mufinder.app.quantum_corrections_widget.QuantumSourceRow` /
        :class:`~mufinder.app.quantum_corrections_widget.QuantumResultRow` lists, so
        the whole row-building body lives here.
        """

        document = self._context.document()
        if document is None:
            self._view.set_sources([])
            self._view.set_results([])
            return

        source_rows: list[QuantumSourceRow] = []
        quantum_states = self._deps.build_quantum_corrections_states()
        definition = self._deps.active_quantum_corrections_definition()
        for candidate in self._deps.eligible_quantum_source_candidates():
            state = quantum_states.get(candidate.key)
            stage_state = self._deps.quantum_stage_state(definition, candidate)
            _params, _advanced, extras = self._deps.workflow_row_configuration(
                definition, candidate.key
            )
            source_rows.append(
                QuantumSourceRow(
                    key=candidate.key,
                    label=candidate.site.label or candidate.job.label,
                    charge_state=self._deps.charge_state_label_for_job(candidate.job),
                    relative_energy=self._deps.format_energy(
                        candidate.site.relative_energy_eV
                    ),
                    displacements=str(3 * len(candidate.structure.species)),
                    source_job=candidate.job.label,
                    stage=stage_state.stage,
                    num_farms_display=(
                        str(self._deps.quantum_num_farms_from_extras(extras))
                        if stage_state.stage == QUANTUM_STAGE_PHONON
                        else ""
                    ),
                    status=getattr(state, "status", "ready") or "ready",
                    configured=self._deps.workflow_row_configuration_exists(
                        definition, candidate.key
                    ),
                    resettable=self._deps.quantum_can_reset_to_stage_one(
                        definition, candidate
                    ),
                )
            )

        result_rows: list[QuantumResultRow] = []
        for result in document.results.values():
            if result.result_type != QUANTUM_PHONON_CALC_TYPE:
                continue
            job_label = ""
            run = (
                document.runs.get(result.calculation_run_id)
                if result.calculation_run_id is not None
                else None
            )
            if run is not None and run.job_id is not None:
                job = document.jobs.get(run.job_id)
                if job is not None:
                    job_label = job.label
            eigenvalue_text = (
                _format_phonon_eigenvalues(result.data.get("phonon_mode_eigenvalues_cm"))
                if isinstance(result.data, dict)
                else ""
            )
            if not eigenvalue_text and isinstance(result.data, dict):
                modes = result.data.get("muon_zpe_modes", [])
                if isinstance(modes, list):
                    eigenvalue_text = ", ".join(
                        f"{mode.get('mode_index')}:{float(mode.get('frequency_cm', 0.0)):.1f}"
                        for mode in modes
                        if isinstance(mode, dict)
                    )
            result_rows.append(
                QuantumResultRow(
                    key=str(result.id),
                    label=job_label or "Quantum phonon",
                    zpe=self._deps.format_energy(
                        result.data.get("muon_zpe_total_eV")
                        if isinstance(result.data, dict)
                        else None
                    ),
                    eigenvalues=eigenvalue_text,
                    witness=(
                        f"{float(result.data['w1_entanglement_witness']):.4g}"
                        if isinstance(result.data, dict)
                        and "w1_entanglement_witness" in result.data
                        else ""
                    ),
                )
            )

        self._view.set_sources(source_rows)
        self._view.set_results(result_rows)


__all__ = [
    "QUANTUM_STAGE_PHONON",
    "QUANTUM_STAGE_SINGLEPOINT",
    "QuantumCorrectionsController",
    "QuantumCorrectionsDeps",
]
