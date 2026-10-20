"""Quantum-corrections workflow controller.

Owns the quantum-corrections workflow: each relaxed muon source runs as a
*singlepoint → phonon* pair, a stage-1 singlepoint job and then a stage-2 phonon
continuation that shares its seed directory.

- **Sources and stages.** Rows are the eligible relaxed muon sites from
  :mod:`mufinder.services.source_candidates`. :meth:`QuantumCorrectionsController.stage_state`
  resolves a row's current stage from its two jobs, bound through the
  definition's ``singlepoint_job_ids_by_label`` / ``phonon_job_ids_by_label``.
- **Two-stage materialize.** :meth:`QuantumCorrectionsController.materialize`
  builds a config per selected row for its current stage, expands them through
  ``expand_quantum_singlepoint_jobs`` / ``expand_quantum_phonon_jobs`` and writes
  both stages into the row's shared seed directory.
- **Submission guards.** :meth:`QuantumCorrectionsController.pre_submit_check`
  checks the active definition against the active host and refuses a stage-2
  row whose stage-1 cell omits the muon mass; :meth:`QuantumCorrectionsController.submit`
  checks each row's phonon method against its pseudopotentials and refuses a
  stage-2 row sent to a target that does not hold its stage-1 checkpoint.
- **Result.** :meth:`QuantumCorrectionsController.validate_artifacts` fails a
  phonon job whose ``.phonon`` is missing or unparseable;
  :meth:`QuantumCorrectionsController.build_result` contributes the muon ZPE,
  phonon eigenvalues, mode participations and the ``w1`` entanglement witness.
- **Configure and reset.** :meth:`QuantumCorrectionsController.configure_request`
  and :meth:`QuantumCorrectionsController.apply_configuration` back the Configure
  dialog, which explains why it opened read-only and which reset unlocks it.
  :meth:`QuantumCorrectionsController.reset_to_stage_one` rewinds a row to a
  fresh stage 1; :meth:`QuantumCorrectionsController.reset_to_stage_two` returns
  a failed, stopped or cancelled phonon job to ``defined`` and keeps stage 1.

Task farming of the phonon stage is applied by the shared submit backend, which
reads each row's farm count from its configuration extras.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import UUID

from mufinder.castep_io.cell_parser import CellParseError, parse_cell_file
from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.enums import ACTIVE_JOB_STATUSES, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.project_service import TERMINAL_RECOVERABLE_JOB_STATUSES
from mufinder.services.source_candidates import (
    QUANTUM_STAGE_PHONON,
    QUANTUM_STAGE_SINGLEPOINT,
    QuantumSourceCandidate,
    QuantumStageState,
    charge_state_for_job,
    charge_state_label_for_job,
    eligible_quantum_source_candidates,
    quantum_seed_name_for_key,
    quantum_source_candidate_by_key,
    quantum_source_label,
    quantum_source_relative_energies,
    quantum_stage_state,
    serialize_quantum_sources,
    source_relaxation_parameter_values,
)
from mufinder.science.quantum_phonons import (
    muon_mass_warning,
    muon_zpe,
    w1_entanglement_witness,
)
from mufinder.workflow import StructureBridgeError, render_job, structure_to_spec
from mufinder.workflow.quantum import (
    QUANTUM_PHONON_CALC_TYPE,
    QUANTUM_PHONON_SET_CALC_TYPE,
    QUANTUM_SINGLEPOINT_CALC_TYPE,
    QuantumPhononJobConfig,
    QuantumPhononSource,
    expand_quantum_phonon_jobs,
    expand_quantum_singlepoint_jobs,
    phonon_method_compatibility,
)
from mufinder.workflow.types import JobSpec

from mufinder.app.adapter_common import build_workflow_settings_template
from mufinder.app.quantum_corrections_widget import (
    QuantumCorrectionsWidget,
    QuantumResultRow,
    QuantumSourceRow,
)
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.app.workflow_config_codecs import (
    batch_row_payload,
    num_farms_from_extras,
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
)
from mufinder.app.workflow_dialogs import SourceConfigureRequest, SourceConfigureResult
from mufinder.app.workflow_parameters import (
    build_workflow_parameter_surface,
    merge_species_options,
    workflow_basic_parameter_names,
)
from mufinder.app.workflow_submission import effective_row_status
from mufinder.app.workflow_table_utils import format_energy_ev

_CONFIG_KEYWORD_NAMES = workflow_basic_parameter_names()
_GUI_TO_METHOD = {
    "FINITEDISPLACEMENT": "finite_displacement",
    "DFPT": "dfpt",
    "LINEARRESPONSE": "linear_response",
}
_METHOD_TO_GUI = {value: key for key, value in _GUI_TO_METHOD.items()}
_DEFAULT_ELEC_ENERGY_TOL = ValueWithUnit(1e-9, "eV")


def _method_from_gui_value(value: object) -> str:
    normalized = str(value).strip().upper()
    try:
        return _GUI_TO_METHOD[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported phonon method '{value}'.") from exc


def _method_to_gui_value(value: object) -> str:
    return _METHOD_TO_GUI.get(str(value).strip().lower(), "FINITEDISPLACEMENT")


def _write_job_directory(job_spec: JobSpec, output_root: Path, *, seed_name: str) -> Path:
    """Write a stage's ``.cell`` / ``.param`` into the row's shared seed directory."""

    job_dir = output_root / seed_name
    job_dir.mkdir(parents=True, exist_ok=True)
    cell_text, param_text = render_job(job_spec)
    (job_dir / f"{seed_name}.cell").write_text(cell_text, encoding="utf-8", newline="\n")
    (job_dir / f"{seed_name}.param").write_text(param_text, encoding="utf-8", newline="\n")
    return job_dir


def _format_phonon_eigenvalues(value: object) -> str:
    """Format a list of gamma-point eigenvalues as ``"1:v1, 2:v2, ..."``."""

    if not isinstance(value, list):
        return ""
    entries: list[str] = []
    for mode_index, eigenvalue in enumerate(value, start=1):
        try:
            entries.append(f"{mode_index}:{float(eigenvalue):.1f}")
        except (TypeError, ValueError):
            continue
    return ", ".join(entries)


def _format_relative_energy(value: float | None) -> str:
    """Format a source's energy relative to its comparable minimum for the ``dE (eV)`` column."""

    return "" if value is None else f"{float(value):.5f}"


def _is_local_target(target: str | None) -> bool:
    """Return whether a job's ``execution_target`` names this computer.

    Matches the local semantics used elsewhere for job reattachment: blank or
    ``"local"`` (case-insensitively).
    """

    value = (target or "").strip()
    return not value or value.casefold() == "local"


_DFPT_LIMITATIONS = (
    "CASTEP DFPT phonons need norm-conserving pseudopotentials and are not "
    "available with DFT+U, meta-GGA or hybrid functionals, non-collinear "
    "magnetism or spin-orbit coupling, or the newer dispersion corrections."
)


def _method_rows(rows: list[tuple[str, str, str]]) -> str:
    return "\n".join(f"  - {label} ({method}, {stage})" for label, method, stage in rows)


def _incompatible_method_message(rows: list[tuple[str, str, str]]) -> tuple[str, str]:
    """Return the refusal for rows using DFPT or linear response with ultrasoft pseudopotentials."""

    return (
        "Phonon method needs norm-conserving pseudopotentials",
        "These rows use a DFPT phonon method with ultrasoft pseudopotentials, which "
        "CASTEP does not support:\n\n"
        f"{_method_rows(rows)}\n\n"
        "Configure a stage-1 row to use FINITEDISPLACEMENT, or set Pseudopotentials to "
        "Norm-conserving if that is what the calculation uses. A stage-2 row's settings "
        "are fixed by its completed singlepoint: reset it to stage 1 to change them.",
    )


def _unverified_method_question(rows: list[tuple[str, str, str]]) -> tuple[str, str]:
    """Return the confirmation for rows using DFPT with an unknown pseudopotential family."""

    return (
        "Check pseudopotentials for DFPT",
        "These rows use a DFPT phonon method, but their pseudopotential family is not "
        "set, so MuFinder cannot check it:\n\n"
        f"{_method_rows(rows)}\n\n"
        f"{_DFPT_LIMITATIONS} The phonon method is fixed once stage 1 has been "
        "submitted. Submit anyway?",
    )


def _stage_is_open(state: QuantumStageState) -> bool:
    """Return whether a row's current stage can still be configured and submitted.

    That is a ready stage (nothing staged yet, or a dismissed terminal job) or
    a job still ``defined``: submitting either rewrites the stage's inputs from
    the row's stored configuration.
    """

    job = state.current_job
    return state.status == "ready" or (job is not None and job.status == JobStatus.DEFINED)


def _locked_reason(states: Sequence[QuantumStageState]) -> str:
    """Return why the Configure dialog cannot edit every parameter of ``states``.

    The reason names the action that unlocks the rows. It is empty when every
    row is an open stage-1 row, the only fully editable case.
    """

    for state in states:
        job = state.current_job
        if job is not None and job.status in ACTIVE_JOB_STATUSES:
            return (
                f"Parameters are locked while a selected row's job is {job.status.value}. "
                "Cancel the job first, then use Reset to Stage 1 to change them."
            )
    if all(_stage_is_open(state) for state in states):
        stages = {state.stage for state in states}
        if stages == {QUANTUM_STAGE_SINGLEPOINT}:
            return ""
        if stages == {QUANTUM_STAGE_PHONON}:
            return (
                "Stage 1 has completed, so only Stage 2 farms can change. "
                "Use Reset to Stage 1 to change the other parameters."
            )
        return "The selection mixes stage-1 and stage-2 rows. Configure them separately."
    for state in states:
        job = state.current_job
        if (
            state.stage == QUANTUM_STAGE_PHONON
            and job is not None
            and job.status in TERMINAL_RECOVERABLE_JOB_STATUSES
        ):
            return (
                f"A selected row's phonon job is {job.status.value}. Use Reset to Stage 2 to "
                "change Stage 2 farms and rerun the phonon stage, or Reset to Stage 1 to "
                "change every parameter."
            )
    for state in states:
        job = state.current_job
        if state.stage == QUANTUM_STAGE_SINGLEPOINT and job is not None and not _stage_is_open(state):
            return (
                f"A selected row's stage-1 singlepoint is {job.status.value}. "
                "Use Reset to Stage 1 to change the parameters and run it again."
            )
    return (
        "A selected row has completed both stages. "
        "Use Reset to Stage 1 to rerun it with different parameters."
    )


def _cell_declares_species_mass(cell_path: Path) -> bool:
    """Return whether a written ``.cell`` file carries a ``SPECIES_MASS`` block.

    Deliberately fails *open*: a missing or unparseable file is no evidence that
    the muon mass was omitted, and this backs a submission guard that must never
    block a user over an unrelated read or parse problem.
    """

    if not cell_path.is_file():
        return True
    try:
        parsed = parse_cell_file(cell_path)
    except (CellParseError, OSError):
        return True
    return any(block.name.strip().lower() == "species_mass" for block in parsed.cell_file.blocks)


@dataclass(frozen=True)
class QuantumCorrectionsDeps:
    """The window-side handles the quantum-corrections controller needs.

    The dialogs: the host-mismatch handlers ask whether to switch the active
    host (shared with every host-guarded workflow), ``confirm`` asks a yes/no
    question and ``warn`` reports a problem. The rest are the window's job-reset
    and refresh primitives the two stage resets drive; the stage-1 reset cleans
    files through the shared context's ``job_file_cleanup``. Everything else the
    controller reads from the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`
    or the service layer.
    """

    handle_structure_host_mismatch: Callable[[object], bool]
    handle_site_host_mismatch: Callable[[object], bool]
    #: Ask a yes/no question (title, text); ``True`` means proceed. The safe
    #: answer, "no", is the default.
    confirm: Callable[[str, str], bool]
    #: Show a warning (title, text).
    warn: Callable[[str, str], None]
    #: Return terminal jobs to ``defined`` without touching their files, and
    #: return how many were reset and any problems to report.
    reset_jobs_keeping_files: Callable[[Sequence[Job]], tuple[int, list[str]]]
    #: Repaint the jobs dashboard and every workflow table after a reset.
    refresh_after_reset: Callable[[], None]


@dataclass(frozen=True)
class QuantumConfigureRequest(SourceConfigureRequest):
    """The quantum Configure dialog's seed: the shared request plus its stage rules.

    ``fully_editable`` rows are all open on stage 1; ``farms_only`` rows are all
    open on stage 2, where only the farm count may still change. A stage is open
    when it is ready or its job is still ``defined``. Any other selection opens
    read-only. ``locked_reason`` says why the dialog cannot change everything
    and names the reset that unlocks it; it is empty when ``fully_editable``.
    """

    pseudopotential_family: str
    farms_only: bool
    fully_editable: bool
    locked_reason: str


@dataclass(frozen=True)
class StageOneReset:
    """A row "Reset to Stage 1" may rewind: its key, label and staged jobs.

    ``jobs`` lists the phonon job (when there is one) before the singlepoint, so
    ``jobs[0]`` owns the shared seed directory.
    """

    key: str
    label: str
    jobs: tuple[Job, ...]


class QuantumCorrectionsController(WorkflowJobController):
    """Controller for the two-stage quantum-corrections workflow."""

    calc_types = frozenset(
        {QUANTUM_SINGLEPOINT_CALC_TYPE, QUANTUM_PHONON_CALC_TYPE}
    )
    stage = WorkflowStage.QUANTUM_CORRECTIONS
    definition_calc_type = QUANTUM_PHONON_SET_CALC_TYPE
    no_ready_message = (
        "Only eligible muon sites that have not yet been submitted can be submitted."
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
        self._preferred_cluster_name: str | None = None

    # --- sources and stages ---

    def candidates(self) -> list[QuantumSourceCandidate]:
        """Return the eligible relaxed muon sources, in table order."""

        return eligible_quantum_source_candidates(self._context.document())

    def candidate_by_key(self, key: str) -> QuantumSourceCandidate | None:
        return quantum_source_candidate_by_key(self._context.document(), key)

    def source_label(self, candidate: QuantumSourceCandidate) -> str:
        return quantum_source_label(self._context.document(), candidate)

    def seed_name_for_key(self, definition: CalculationDefinition | None, key: str) -> str:
        return quantum_seed_name_for_key(self._context.document(), definition, key)

    def singlepoint_job_for_key(self, definition: CalculationDefinition | None, key: str) -> Job | None:
        return self.job_for_key(
            definition,
            key,
            mapping_name="singlepoint_job_ids_by_label",
            calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
        )

    def phonon_job_for_key(self, definition: CalculationDefinition | None, key: str) -> Job | None:
        """Return the row's phonon job, falling back to the ``job_ids_by_label`` alias."""

        job = self.job_for_key(
            definition,
            key,
            mapping_name="phonon_job_ids_by_label",
            calc_type=QUANTUM_PHONON_CALC_TYPE,
        )
        if job is not None:
            return job
        return self.job_for_key(definition, key, calc_type=QUANTUM_PHONON_CALC_TYPE)

    def stage_state(
        self, definition: CalculationDefinition | None, candidate: QuantumSourceCandidate
    ) -> QuantumStageState:
        return quantum_stage_state(
            singlepoint_job=self.singlepoint_job_for_key(definition, candidate.key),
            phonon_job=self.phonon_job_for_key(definition, candidate.key),
            row_status=effective_row_status,
        )

    def can_reset_to_stage_one(
        self, definition: CalculationDefinition | None, candidate: QuantumSourceCandidate
    ) -> bool:
        """Return whether a row can be rewound to stage 1.

        Quantum corrections is the one workflow whose recovery is a *stage*
        reset rather than a job reset: a row stages a singlepoint job and then a
        phonon job into a shared output directory, so the reset deletes both job
        records instead of calling ``reset_failed_job``. That makes this
        predicate the workflow's whole in-table escape hatch.

        A row qualifies once it has staged a singlepoint job and its current
        job (the phonon job when there is one, else the singlepoint) is not
        active. That covers a job still ``defined``, whether never submitted or
        returned there by the Job Dashboard's Reset, and every finished status,
        so no inactive row is left without a way back to editable parameters.
        An active job has to be cancelled first.
        """

        singlepoint_job = self.singlepoint_job_for_key(definition, candidate.key)
        if singlepoint_job is None:
            return False
        current_job = self.phonon_job_for_key(definition, candidate.key) or singlepoint_job
        return current_job.status not in ACTIVE_JOB_STATUSES

    def can_reset_to_stage_two(
        self, definition: CalculationDefinition | None, candidate: QuantumSourceCandidate
    ) -> bool:
        """Return whether a row's phonon job can be returned to ``defined``.

        Only a failed, stopped or cancelled phonon job whose stage-1 singlepoint
        completed qualifies: the phonon stage then reruns as a continuation from
        the stage-1 checkpoint, which this reset keeps. A completed phonon job
        does not, because its run has rewritten that checkpoint; Reset to
        Stage 1 reruns both stages instead.
        """

        phonon_job = self.phonon_job_for_key(definition, candidate.key)
        if phonon_job is None or phonon_job.status not in TERMINAL_RECOVERABLE_JOB_STATUSES:
            return False
        singlepoint_job = self.singlepoint_job_for_key(definition, candidate.key)
        return singlepoint_job is not None and singlepoint_job.status == JobStatus.COMPLETED

    def definition_and_key_for_job(
        self, job: Job
    ) -> tuple[CalculationDefinition, str] | tuple[None, None]:
        """Return the definition and row key whose job mappings bind ``job``."""

        for definition in self.definitions():
            for mapping_name in ("singlepoint_job_ids_by_label", "phonon_job_ids_by_label", "job_ids_by_label"):
                mapping = definition.parameters.get(mapping_name, {})
                if not isinstance(mapping, dict):
                    continue
                for key, value in mapping.items():
                    if str(value) == str(job.id):
                        return definition, str(key)
        return None, None

    def _completed_stage_one_job(self, job: Job) -> Job | None:
        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return None
        definition, key = self.definition_and_key_for_job(job)
        if definition is None or key is None:
            return None
        source_job = self.singlepoint_job_for_key(definition, key)
        if source_job is None or source_job.status != JobStatus.COMPLETED:
            return None
        return source_job

    def stage_one_target(self, job: Job) -> str | None:
        """Return where a phonon job's stage-1 singlepoint ran.

        ``None`` when ``job`` is not a quantum phonon job or its completed
        singlepoint cannot be found; otherwise the singlepoint's
        ``execution_target`` (``"local"`` or a cluster name).
        """

        source_job = self._completed_stage_one_job(job)
        if source_job is None:
            return None
        return (source_job.execution_target or "").strip() or "local"

    def continuation_remote_target(self, job: Job, *, cluster_name: str) -> str:
        """Return the remote directory a phonon job continues in on ``cluster_name``.

        That is its completed stage-1 singlepoint's remote directory when the
        singlepoint ran on that cluster, and ``""`` otherwise.
        """

        source_job = self._completed_stage_one_job(job)
        if source_job is None:
            return ""
        if source_job.execution_target.strip().casefold() != cluster_name.strip().casefold():
            return ""
        return (source_job.remote_directory or "").strip()

    # --- configuration ---

    @staticmethod
    def parameter_specs() -> list[KeywordSpec]:
        """Return the Configure dialog's parameter surface (tight default ``elec_energy_tol``)."""

        specs = build_workflow_parameter_surface(
            workflow_stage="quantum_corrections",
            basic_names=_CONFIG_KEYWORD_NAMES,
        )
        return [
            replace(spec, default_value=_DEFAULT_ELEC_ENERGY_TOL)
            if spec.name == "elec_energy_tol"
            else spec
            for spec in specs
        ]

    def _row_configuration(
        self, definition: CalculationDefinition | None, key: str
    ) -> tuple[dict[str, object], tuple[str, ...], dict[str, object]]:
        """Return a row's stored configuration over its source relaxation's settings."""

        params, advanced, extras = workflow_row_configuration(definition, key)
        candidate = self.candidate_by_key(key)
        values = (
            source_relaxation_parameter_values(candidate.job, self._context.job_output_directory)
            if candidate is not None
            else {}
        )
        legacy_method = extras.get("phonon_method")
        if legacy_method is not None:
            values.setdefault("phonon_method", _method_to_gui_value(legacy_method))
        values.update(params)
        return values, advanced, extras

    def runtime_configuration(
        self, definition: CalculationDefinition | None, key: str
    ) -> tuple[dict[str, object], str]:
        """Return a row's effective parameter values and its pseudopotential family.

        The row's stored values sit over its source relaxation's settings; a
        definition-level phonon method or pseudopotential family written by an
        older configuration fills in what the row leaves unset.
        """

        params, _advanced, extras = self._row_configuration(definition, key)
        config = definition.parameters.get("configuration", {}) if definition is not None else {}
        legacy_method = "finite_displacement"
        legacy_pseudopotential = "unknown"
        if isinstance(config, dict):
            legacy_method = str(config.get("method", legacy_method)).strip() or legacy_method
            legacy_pseudopotential = (
                str(config.get("pseudopotential_family", legacy_pseudopotential)).strip() or legacy_pseudopotential
            )
        values = dict(params)
        values.setdefault("phonon_method", _method_to_gui_value(legacy_method))
        values.setdefault("elec_energy_tol", _DEFAULT_ELEC_ENERGY_TOL)
        pseudopotential_family = str(extras.get("pseudopotential_family", legacy_pseudopotential)).strip() or "unknown"
        return values, pseudopotential_family

    def ensure_definition(self, source_keys: Sequence[str]) -> CalculationDefinition | None:
        """Return the active definition with the selected sources added, creating one if needed.

        Returns ``None`` when there is no document or none of ``source_keys`` is
        an eligible source.
        """

        document = self._context.document()
        if document is None:
            return None
        candidates = [
            candidate for key in source_keys if (candidate := self.candidate_by_key(key)) is not None
        ]
        if not candidates:
            return None
        definition = self.active_definition()
        if definition is not None:
            for candidate in candidates:
                if candidate.structure.id not in definition.input_structure_ids:
                    definition.input_structure_ids.append(candidate.structure.id)
                if candidate.site.id not in definition.input_site_ids:
                    definition.input_site_ids.append(candidate.site.id)
            config = definition.parameters.setdefault("configuration", {})
            if isinstance(config, dict):
                config.setdefault("method", "finite_displacement")
                config.setdefault("pseudopotential_family", "unknown")
                config["sources"] = serialize_quantum_sources(document, definition.input_site_ids)
        else:
            definition = self._context.project_service.register_quantum_phonon_definition(
                document,
                input_structure_ids=list(dict.fromkeys(candidate.structure.id for candidate in candidates)),
                input_site_ids=list(dict.fromkeys(candidate.site.id for candidate in candidates)),
                parameters={
                    "method": "finite_displacement",
                    "pseudopotential_family": "unknown",
                    "sources": serialize_quantum_sources(document, [candidate.site.id for candidate in candidates]),
                    "singlepoint_job_ids_by_label": {},
                    "phonon_job_ids_by_label": {},
                    "job_seed_names_by_label": {},
                },
                output_directory=document.project_root / "files" / "quantum_corrections",
            )
        definition.parameters.setdefault("singlepoint_job_ids_by_label", {})
        definition.parameters.setdefault("phonon_job_ids_by_label", {})
        definition.parameters.setdefault("job_seed_names_by_label", {})
        return definition

    def configure_request(self, keys: Sequence[str]) -> QuantumConfigureRequest | None:
        """Return what the Configure dialog opens with for the rows ``keys`` name.

        ``None`` when there is no document or no eligible row among ``keys``.
        """

        if self._context.document() is None:
            return None
        candidates = [
            candidate for key in keys if (candidate := self.candidate_by_key(key)) is not None
        ]
        if not candidates:
            return None
        definition = self.active_definition()
        values_by_key: list[dict[str, object]] = []
        advanced_names: set[str] = set()
        stage_states: list[QuantumStageState] = []
        pseudopotential_families: list[str] = []
        num_farms_values: list[int] = []
        for candidate in candidates:
            values, pseudopotential_family = self.runtime_configuration(definition, candidate.key)
            _params, row_advanced_names, extras = workflow_row_configuration(definition, candidate.key)
            values_by_key.append(values)
            advanced_names.update(row_advanced_names)
            stage_states.append(self.stage_state(definition, candidate))
            pseudopotential_families.append(pseudopotential_family)
            num_farms_values.append(num_farms_from_extras(extras))
        farms_only = all(
            state.stage == QUANTUM_STAGE_PHONON and _stage_is_open(state) for state in stage_states
        )
        fully_editable = all(
            state.stage == QUANTUM_STAGE_SINGLEPOINT and _stage_is_open(state) for state in stage_states
        )
        read_only = not fully_editable and not farms_only
        locked_reason = _locked_reason(stage_states)
        if read_only:
            title = "Inspect Quantum Corrections Parameters"
        elif farms_only:
            title = "Configure Quantum Corrections Stage 2 Farms"
        else:
            title = "Configure Quantum Corrections Parameters"
        stage_text = stage_states[0].stage if len(candidates) == 1 else "mixed/selected"
        return QuantumConfigureRequest(
            keys=tuple(candidate.key for candidate in candidates),
            row_values=tuple(values_by_key),
            advanced_names=frozenset(advanced_names),
            title=title,
            intro_text=(
                f"Selected sources: {len(candidates)}\n"
                f"Current stage: {stage_text}\n"
                f"Displacements required for the phonon stage: {3 * len(candidates[0].structure.species)}"
                + (f"\n\n{locked_reason}" if locked_reason else "")
            ),
            read_only=read_only,
            num_farms=_agreed(num_farms_values, default=1),
            species_options=merge_species_options(candidate.structure for candidate in candidates),
            pseudopotential_family=_agreed(pseudopotential_families, default="") or "unknown",
            farms_only=farms_only,
            fully_editable=fully_editable,
            locked_reason=locked_reason,
        )

    def apply_configuration(
        self,
        keys: Sequence[str],
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> None:
        """Store an accepted Configure dialog on every row ``keys`` name.

        ``extras`` are the dialog's workflow values (pseudopotential family and
        farm count). A mixed parameter the user left alone keeps each row's own
        stored value.
        """

        payload = row_settings_payload(result.values, result.advanced_names, extras)
        definition = self.ensure_definition(keys)
        if definition is None:
            return
        for key in keys:
            store_workflow_row_configuration(
                definition, key, batch_row_payload(definition, key, payload, result.untouched_mixed)
            )
        self.refresh_view()
        self._context.persist_ui_state(
            failure_message="Quantum Corrections configuration was updated, but autosave failed."
        )

    def rename_source(self, key: str, label: str) -> bool:
        """Rename a source's muon site from its editable label cell.

        Returns whether the label changed. A blank or unchanged label restores
        the cell; an unknown key does nothing.
        """

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

    def stage_one_resets(self, keys: Iterable[str]) -> list[StageOneReset] | None:
        """Return the rows among ``keys`` that can be rewound to stage 1.

        ``None`` when there is no active definition (nothing has been staged).
        """

        definition = self.active_definition()
        if definition is None:
            return None
        resets: list[StageOneReset] = []
        for key in keys:
            candidate = self.candidate_by_key(key)
            if candidate is None or not self.can_reset_to_stage_one(definition, candidate):
                continue
            staged = (self.phonon_job_for_key(definition, key), self.singlepoint_job_for_key(definition, key))
            resets.append(
                StageOneReset(
                    key=key,
                    label=candidate.site.label or candidate.job.label,
                    jobs=tuple(job for job in staged if job is not None),
                )
            )
        return resets

    def reset_to_stage_one(self, keys: Iterable[str]) -> None:
        """Rewind the rows ``keys`` name to a fresh stage 1, after confirmation.

        Each row's staged job records and its shared seed directory are
        deleted and its seed name is forgotten; its stored configuration is
        kept. The row then reads ``singlepoint`` / ``ready`` with editable
        parameters. A row whose directory another job also uses is left alone
        and reported.
        """

        resets = self.stage_one_resets(keys)
        if resets is None:
            return
        if not resets:
            self._context.show_info(
                "No resettable rows",
                "Only Quantum Corrections rows that have staged a job, and whose job is not "
                "submitting, queued, running or being retrieved, can be reset to stage 1. "
                "Cancel an active job first.",
            )
            return
        if not self._deps.confirm(
            "Reset Quantum Corrections",
            f"Reset {len(resets)} Quantum Corrections row(s) to stage 1?\n\n"
            "This removes the staged Quantum Corrections job records and clears the shared "
            "output directory for each selected row so stage 1 can be rerun with updated "
            "parameters.",
        ):
            return

        cleanup = self._context.job_file_cleanup
        # The phonon job, when there is one, owns the shared seed directory,
        # which is deleted whole, so it must hold only this row's two stages.
        plan = [
            cleanup.plan_item(reset.jobs, label=reset.label, check_ownership=True, key=reset.key)
            for reset in resets
        ]
        outcome = cleanup.execute(plan, choose_disposition=lambda _item: "delete", stop_on_error=False)
        for item in outcome.cleaned:
            self.forget_seed_name(str(item.key))
        reset_count = len(outcome.cleaned)
        cleanup_errors = [
            f"{failure.refusal}; the row was not reset"
            if failure.refusal is not None
            else f"- {failure.item.label}: {failure.error}"
            for failure in outcome.failures
        ]

        self._deps.refresh_after_reset()
        if cleanup_errors:
            self._deps.warn(
                "Directory cleanup issues",
                "Some Quantum Corrections rows could not be reset cleanly:\n" + "\n".join(cleanup_errors),
            )
        if reset_count:
            self._context.persist_ui_state(
                failure_message="Quantum Corrections stage reset was recorded, but autosave failed.",
            )
            self._context.status_message(f"Reset {reset_count} Quantum Corrections row(s) to stage 1")

    def stage_two_resets(self, keys: Iterable[str]) -> list[Job]:
        """Return the phonon jobs of the rows among ``keys`` that can be reset to stage 2."""

        definition = self.active_definition()
        if definition is None:
            return []
        jobs: list[Job] = []
        for key in keys:
            candidate = self.candidate_by_key(key)
            if candidate is None or not self.can_reset_to_stage_two(definition, candidate):
                continue
            phonon_job = self.phonon_job_for_key(definition, key)
            if phonon_job is not None:
                jobs.append(phonon_job)
        return dedupe_jobs(jobs)

    def reset_to_stage_two(self, keys: Iterable[str]) -> None:
        """Return the phonon jobs of the rows ``keys`` name to ``defined``, after confirmation.

        Each job keeps its files: the completed stage-1 singlepoint and the
        checkpoint the continuation reads stay in place. The row then reads
        ``phonon`` / ``defined``, so Stage 2 farms can change before it is
        submitted again.
        """

        jobs = self.stage_two_resets(keys)
        if not jobs:
            self._context.show_info(
                "No rows to reset to stage 2",
                "Only Quantum Corrections rows whose stage-1 singlepoint completed and whose "
                "phonon job failed, was stopped or was cancelled can be reset to stage 2. "
                "Use Reset to Stage 1 to rerun both stages.",
            )
            return
        if not self._deps.confirm(
            "Reset Quantum Corrections to stage 2",
            f"Reset {len(jobs)} Quantum Corrections row(s) to stage 2?\n\n"
            "Each row's phonon job returns to 'defined' so Stage 2 farms can be changed "
            "before it is submitted again. The completed stage-1 singlepoint and the shared "
            "output directory, including its checkpoint, are kept.",
        ):
            return
        reset_count, errors = self._deps.reset_jobs_keeping_files(jobs)
        self._deps.refresh_after_reset()
        if errors:
            self._deps.warn(
                "Stage 2 reset issues",
                "Some Quantum Corrections rows could not be reset to stage 2:\n" + "\n".join(errors),
            )
        if reset_count:
            self._context.persist_ui_state(
                failure_message="Quantum Corrections stage reset was recorded, but autosave failed.",
            )
            self._context.status_message(f"Reset {reset_count} Quantum Corrections row(s) to stage 2")

    def forget_seed_name(self, key: str) -> None:
        """Drop a reset row's recorded seed name so stage 1 starts from a fresh one."""

        definition = self.active_definition()
        if definition is None:
            return
        seed_mapping = definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(seed_mapping, dict):
            seed_mapping = {}
            definition.parameters["job_seed_names_by_label"] = seed_mapping
        seed_mapping.pop(key, None)

    def phonon_result(self, result_id: str) -> Result | None:
        """Return the completed quantum-phonon result ``result_id`` names, if any."""

        document = self._context.document()
        if document is None:
            return None
        try:
            parsed_result_id = UUID(str(result_id))
        except ValueError:
            return None
        result = document.results.get(parsed_result_id)
        if result is None or result.result_type != QUANTUM_PHONON_CALC_TYPE:
            return None
        return result

    def phonon_analysis_input(self, result: Result) -> tuple[Path, str]:
        """Return the ``.phonon`` file and job label the analysis window opens on.

        Raises ``ValueError`` with a user-facing reason when the result names no
        phonon file or the file is gone. A relative path is read from the
        project root.
        """

        document = self._context.document()
        phonon_file = result.data.get("phonon_file") if isinstance(result.data, dict) else None
        if not isinstance(phonon_file, str) or not phonon_file.strip():
            raise ValueError("The completed result does not reference a phonon file.")
        phonon_path = Path(phonon_file)
        if not phonon_path.is_absolute() and document is not None:
            phonon_path = document.project_root / phonon_path
        if not phonon_path.exists():
            raise ValueError(f"Could not find phonon file:\n{phonon_path}")
        job_label = "Quantum phonon"
        if document is not None:
            run = document.runs.get(result.calculation_run_id) if result.calculation_run_id is not None else None
            job = document.jobs.get(run.job_id) if run is not None and run.job_id is not None else None
            if job is not None:
                job_label = job.label
        return phonon_path, job_label

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready)
    # apply. submission_groups only preselects the stage-1 cluster (see
    # ``submit``); task farming is applied by the shared submit backend, not here.

    def normalize(self, value: object) -> str:
        return str(value)

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Validate the active definition against the host and the stage-1 inputs.

        Two guards run here:

        - the active definition's structures and sites must belong to the
          active host (:func:`~mufinder.app.workflow_controllers.base.confirm_active_host`);
          a mismatch the user does not resolve refuses silently.
        - :meth:`_stage_two_sources_missing_muon_mass`, which refuses a stage-2
          phonon continuation whose stage-1 singlepoint cell has no
          ``SPECIES_MASS`` block. That refusal explains itself via ``show_info``.
        """

        document = self._context.document()
        definition = self.active_definition()
        if (
            document is not None
            and definition is not None
            and not confirm_active_host(
                self._context.project_service,
                document,
                structure_ids=definition.input_structure_ids,
                site_ids=definition.input_site_ids,
                handle_structure_mismatch=self._deps.handle_structure_host_mismatch,
                handle_site_mismatch=self._deps.handle_site_host_mismatch,
            )
        ):
            return False

        missing = self._stage_two_sources_missing_muon_mass(keys)
        if not missing:
            return True
        labels = "\n".join(f"  - {label}" for label in missing)
        self._context.show_info(
            "Stage-1 muon mass missing",
            "These rows cannot run their phonon continuation:\n\n"
            f"{labels}\n\n"
            "Their stage-1 singlepoint cell has no SPECIES_MASS block, so CASTEP "
            "wrote the default hydrogen mass for H:mu into the checkpoint. A "
            "continuation restores species masses from that checkpoint, so the "
            "phonon run would use the wrong mass and report frequencies and a ZPE "
            "roughly a factor of three too low. Reset each row to stage 1 and rerun "
            "the singlepoint before submitting the phonon stage.",
        )
        return False

    def _stage_two_sources_missing_muon_mass(self, keys: list[object]) -> list[str]:
        """Return labels of selected stage-2 rows whose stage-1 cell omits the mass.

        The evidence is the stage-1 ``.cell`` file still on disk in the shared seed
        directory. That check has one blind spot: both stages write ``{seed}.cell``
        to the same path, so once a stage-2 job has been materialized the stage-1
        cell is gone and this returns nothing. It runs before ``materialize`` in
        :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.submit`,
        which keeps the first stage-2 submission of a legacy row catchable; a row
        that slips through is still caught after the fact by the ``.phonon`` mass
        check in :func:`~mufinder.science.quantum_phonons.muon_mass_warning`.
        """

        definition = self.active_definition()
        if definition is None:
            return []
        output_root = self._context.definition_output_directory(definition)
        if output_root is None:
            return []
        missing: list[str] = []
        for key in [str(value) for value in keys]:
            candidate = self.candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self.stage_state(definition, candidate)
            if stage_state.stage != QUANTUM_STAGE_PHONON:
                continue
            seed_name = self.seed_name_for_key(definition, key)
            cell_path = output_root / seed_name / f"{seed_name}.cell"
            if _cell_declares_species_mass(cell_path):
                continue
            missing.append(self.source_label(candidate))
        return missing

    def submit(
        self, values: list[object], *, target: Literal["local", "cluster"]
    ) -> None:
        """Refuse stage-2 rows sent to a target without their stage-1 checkpoint.

        The phonon stage is a CASTEP continuation of the stage-1 singlepoint and
        reads its ``.check`` file, which exists only where the singlepoint ran:
        MuFinder neither downloads checkpoints from a cluster nor uploads them to
        one. A stage-2 row is therefore refused, with an explanation, when it is
        sent locally after a cluster singlepoint or to a cluster after a local
        one. A cluster submission goes to a single cluster, so stage-2 rows whose
        singlepoints ran on different clusters are refused too; otherwise the
        submission dialog opens with the stage-1 cluster preselected. The
        cluster backend still refuses a phonon job if a different cluster is
        picked in that dialog.

        Before any of that, each submittable row's phonon method is checked
        against its pseudopotential family (see :meth:`_phonon_method_issues`):
        DFPT or linear response with ultrasoft pseudopotentials is refused with
        an explanation naming the rows, and with an unknown family the user is
        asked to confirm. Both checks cover stage-1 rows too, because the method
        is frozen once the singlepoint has been submitted.
        """

        incompatible, unverified = self._phonon_method_issues(values)
        if incompatible:
            self._context.show_info(*_incompatible_method_message(incompatible))
            return
        refusal = self._stage_two_target_refusal(values, target=target)
        if refusal is not None:
            title, message = refusal
            self._context.show_info(title, message)
            return
        if unverified and not self._deps.confirm(*_unverified_method_question(unverified)):
            return
        self._preferred_cluster_name = (
            self._stage_two_cluster_name(values) if target == "cluster" else None
        )
        try:
            super().submit(values, target=target)
        finally:
            self._preferred_cluster_name = None

    def submission_groups(
        self, jobs: list[Job]
    ) -> list[tuple[list[Job], dict[str, object]]]:
        """Preselect the stage-1 cluster for a cluster submission of stage-2 rows."""

        if self._preferred_cluster_name:
            return [(jobs, {"default_cluster_name": self._preferred_cluster_name})]
        return [(jobs, {})]

    def _submittable_rows(
        self, keys: list[object]
    ) -> list[tuple[CalculationDefinition, str, object, object]]:
        """Return ``(definition, key, candidate, stage state)`` for rows submittable now.

        A row counts when its current stage is ready, or its current job is
        still ``defined``. Rows already running or finished are left to the
        shared submission plan, which skips them.
        """

        definition = self.active_definition()
        if definition is None:
            return []
        rows: list[tuple[CalculationDefinition, str, object, object]] = []
        for key in [str(value) for value in keys]:
            candidate = self.candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self.stage_state(definition, candidate)
            if not _stage_is_open(stage_state):
                continue
            rows.append((definition, key, candidate, stage_state))
        return rows

    def _phonon_method_issues(
        self, keys: list[object]
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
        """Split submittable rows' phonon methods into incompatible and unverified.

        Each entry is ``(label, CASTEP method, stage)``. A method is incompatible
        when :func:`~mufinder.workflow.quantum.phonon_method_compatibility`
        rejects it for the row's pseudopotential family (DFPT or linear response
        with ultrasoft pseudopotentials), and unverified when it is accepted with
        a caveat (the family is not set).
        """

        incompatible: list[tuple[str, str, str]] = []
        unverified: list[tuple[str, str, str]] = []
        for definition, key, candidate, stage_state in self._submittable_rows(keys):
            values, pseudopotential_family = self.runtime_configuration(definition, key)
            gui_method = str(values.get("phonon_method", "FINITEDISPLACEMENT")).strip().upper()
            try:
                method = _method_from_gui_value(gui_method)
            except ValueError:
                continue
            compatible, message = phonon_method_compatibility(
                method,  # type: ignore[arg-type]
                pseudopotential_family=pseudopotential_family,  # type: ignore[arg-type]
            )
            entry = (
                self.source_label(candidate),
                gui_method,
                "stage 2" if getattr(stage_state, "stage", None) == QUANTUM_STAGE_PHONON else "stage 1",
            )
            if not compatible:
                incompatible.append(entry)
            elif message:
                unverified.append(entry)
        return incompatible, unverified

    def _stage_two_stage_one_targets(self, keys: list[object]) -> list[tuple[str, str]]:
        """Return ``(label, stage-1 execution target)`` for each selected stage-2 row.

        Only rows that could be submitted now count (see :meth:`_submittable_rows`).
        """

        rows: list[tuple[str, str]] = []
        for definition, key, candidate, stage_state in self._submittable_rows(keys):
            if getattr(stage_state, "stage", None) != QUANTUM_STAGE_PHONON:
                continue
            singlepoint = self.singlepoint_job_for_key(definition, key)
            if singlepoint is None:
                continue
            rows.append(
                (
                    self.source_label(candidate),
                    (singlepoint.execution_target or "").strip(),
                )
            )
        return rows

    def _stage_two_cluster_name(self, keys: list[object]) -> str | None:
        clusters = {
            stage_one
            for _label, stage_one in self._stage_two_stage_one_targets(keys)
            if not _is_local_target(stage_one)
        }
        return next(iter(clusters)) if len(clusters) == 1 else None

    def _stage_two_target_refusal(
        self, keys: list[object], *, target: Literal["local", "cluster"]
    ) -> tuple[str, str] | None:
        """Return ``(title, message)`` when stage-2 rows cannot run on ``target``."""

        rows = self._stage_two_stage_one_targets(keys)
        if target == "local":
            remote = [(label, stage_one) for label, stage_one in rows if not _is_local_target(stage_one)]
            if not remote:
                return None
            listing = "\n".join(f"  - {label} (stage 1 ran on {cluster})" for label, cluster in remote)
            return (
                "Phonon stage must run where stage 1 ran",
                "These rows cannot run their phonon continuation on this computer:\n\n"
                f"{listing}\n\n"
                "The phonon stage continues from the stage-1 checkpoint (.check), which "
                "is only on the cluster that ran the singlepoint; MuFinder does not "
                "download checkpoint files. Submit these rows on that cluster, or reset "
                "them to stage 1 to rerun both stages here.",
            )

        local = [label for label, stage_one in rows if _is_local_target(stage_one)]
        if local:
            listing = "\n".join(f"  - {label}" for label in local)
            return (
                "Phonon stage must run where stage 1 ran",
                "These rows cannot run their phonon continuation on a cluster:\n\n"
                f"{listing}\n\n"
                "Their stage-1 singlepoint ran on this computer, so its checkpoint "
                "(.check) is only here; MuFinder does not upload checkpoint files. "
                "Run these rows locally, or reset them to stage 1 to rerun both stages "
                "on a cluster.",
            )
        by_cluster: dict[str, list[str]] = {}
        for label, stage_one in rows:
            by_cluster.setdefault(stage_one, []).append(label)
        if len(by_cluster) <= 1:
            return None
        listing = "\n".join(
            f"  - {cluster}: {', '.join(labels)}" for cluster, labels in sorted(by_cluster.items())
        )
        return (
            "Submit each cluster's rows separately",
            "The selected rows' stage-1 singlepoints ran on different clusters:\n\n"
            f"{listing}\n\n"
            "Each phonon continuation must run on the cluster that holds its "
            "checkpoint, and a cluster submission goes to one cluster. Select the rows "
            "for one cluster at a time.",
        )

    def build_states(self) -> dict[str, RelaxationRunState]:
        """Return each eligible row's state on its current stage."""

        states: dict[str, RelaxationRunState] = {}
        definition = self.active_definition()
        for candidate in self.candidates():
            stage_state = self.stage_state(definition, candidate)
            job = stage_state.current_job
            if job is None:
                states[candidate.key] = RelaxationRunState(status=stage_state.status)
                continue
            states[candidate.key] = RelaxationRunState(
                status=stage_state.status,
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
            )
        return states

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        """Return each row's current-stage job across every definition, once each."""

        if self._context.document() is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        str_keys = [str(key) for key in keys]
        for definition in self.definitions():
            for key in str_keys:
                candidate = self.candidate_by_key(key)
                if candidate is None:
                    continue
                job = self.stage_state(definition, candidate).current_job
                if job is None or job.id in seen:
                    continue
                selected.append(job)
                seen.add(job.id)
        return selected

    def materialize(self, keys: list[object]) -> list[Job]:
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

        singlepoint_configs: list[QuantumPhononJobConfig] = []
        phonon_configs: list[QuantumPhononJobConfig] = []
        for key in str_keys:
            candidate = self.candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self.stage_state(definition, candidate)
            if not _stage_is_open(stage_state):
                continue
            try:
                structure = structure_to_spec(candidate.structure)
            except StructureBridgeError:
                continue
            values, pseudopotential_family = self.runtime_configuration(definition, key)
            _params, _advanced, extras = workflow_row_configuration(definition, key)
            values = {name: value for name, value in values.items() if name != "task"}
            method_value = values.pop("phonon_method", "FINITEDISPLACEMENT")
            try:
                method = _method_from_gui_value(method_value)
                overrides = build_workflow_settings_template(values)
            except ValueError:
                continue
            config = QuantumPhononJobConfig(
                source=QuantumPhononSource(
                    label=self.source_label(candidate),
                    structure=structure,
                    source_site_id=candidate.site.id,
                    source_relaxation_run_id=candidate.run.id,
                    source_relaxation_job_id=candidate.job.id,
                    charge_state=charge_state_for_job(candidate.job),
                    total_energy_eV=candidate.site.total_energy_eV,
                ),
                method=method,
                pseudopotential_family=str(pseudopotential_family),
                num_farms=num_farms_from_extras(extras),
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
        # Load-bearing: ``phonon_job_for_key`` falls back to reading
        # ``job_ids_by_label``, so keep the phonon mapping aliased there.
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
            candidate = self.candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self.stage_state(definition, candidate)
            seed_name = self.seed_name_for_key(definition, key)
            seed_mapping[key] = seed_name
            if stage_state.stage == QUANTUM_STAGE_SINGLEPOINT:
                spec = singlepoint_specs_by_key.get(key)
                existing = self.singlepoint_job_for_key(definition, key)
                mapping = singlepoint_mapping
                calc_type = QUANTUM_SINGLEPOINT_CALC_TYPE
            else:
                spec = phonon_specs_by_key.get(key)
                existing = self.phonon_job_for_key(definition, key)
                mapping = phonon_mapping
                calc_type = QUANTUM_PHONON_CALC_TYPE
            if spec is None:
                continue
            if existing is not None:
                if existing.status == JobStatus.DEFINED:
                    existing.label = spec.name
                    _write_job_directory(spec, output_root, seed_name=seed_name)
                created.append(existing)
                continue
            _write_job_directory(spec, output_root, seed_name=seed_name)
            job = self._context.project_service.add_job(
                document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type=calc_type,
            )
            mapping[key] = str(job.id)
            created.append(job)

        jobs = dedupe_jobs(created)
        if jobs:
            self.refresh_view()
            self._context.persist_ui_state(
                failure_message=(
                    "Quantum Corrections jobs were generated, but autosave failed."
                )
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
        """Fail a quantum-phonon job whose ``.phonon`` is missing or unusable.

        Reads the carrier: ``phonon_missing`` for an absent file, and
        ``phonon_error`` for a file that does not parse or fails the ZPE / Gamma /
        witness computation during collection. A singlepoint job is accepted.
        """

        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return None
        if artifacts.phonon_missing:
            seed_name = self._context.job_seed_name(job)
            return f"Expected CASTEP phonon output was not found: {seed_name}.phonon"
        if artifacts.phonon_error is not None:
            return f"Failed to parse CASTEP phonon output: {artifacts.phonon_error}"
        return None

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute the phonon/ZPE payload for a completed quantum-phonon job.

        A quantum-*singlepoint* job contributes nothing (an empty
        :class:`~mufinder.app.workflow_controllers.base.ControllerResult`, a strict
        no-op): its result is the base payload only, and the row moves on to the
        phonon stage through :meth:`stage_state` on the next materialize or
        refresh, not here.

        For a quantum-*phonon* job, :meth:`validate_artifacts` has already
        verified the ``.phonon`` file exists and that the full ZPE / Gamma /
        witness computation succeeds (failing the job on any
        :class:`~mufinder.science.quantum_phonons.PhononParseError`). The file is
        parsed **once** on the runner worker thread and carried on
        ``artifacts.phonon``; this controller derives the ZPE / eigenvalue /
        witness payload from the carried calculation with no second disk read. The
        ``.phonon`` file itself is **not** contributed to ``output_files``: the
        call site already passes ``_output_files_for_job(job)``.
        """

        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return ControllerResult()

        phonon = artifacts.phonon
        if phonon is None:
            return ControllerResult()
        phonon_path = artifacts.phonon_path
        if phonon_path is None:
            output_dir = self._context.job_output_directory(job)
            seed_name = self._context.job_seed_name(job)
            phonon_path = (
                None if output_dir is None else output_dir / f"{seed_name}.phonon"
            )

        zpe = muon_zpe(phonon)
        payload: dict[str, object] = {
            "phonon_file": str(phonon_path) if phonon_path is not None else "",
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
        # The muon-mass check is also raised to the run's warnings so it reaches the
        # jobs dashboard, not only the durable payload: a run CASTEP performed with
        # the default hydrogen mass produces plausible-looking numbers that are
        # roughly a factor of three off, and nothing downstream can repair them.
        mass_warning = muon_mass_warning(phonon)
        warnings = [] if mass_warning is None else [mass_warning]
        return ControllerResult(payload=payload, warnings=warnings)

    # --- view contract ---

    def refresh_view(self) -> None:
        """Rebuild the quantum-corrections source and result tables.

        Unlike the row-table widgets, which build rows from
        ``set_runtime_states``, the quantum widget takes pre-built
        :class:`~mufinder.app.quantum_corrections_widget.QuantumSourceRow` /
        :class:`~mufinder.app.quantum_corrections_widget.QuantumResultRow` lists, so
        the row building lives here.
        """

        document = self._context.document()
        if document is None:
            self._view.set_sources([])
            self._view.set_results([])
            return

        source_rows: list[QuantumSourceRow] = []
        quantum_states = self.build_states()
        candidates = self.candidates()
        relative_energies = quantum_source_relative_energies(candidates)
        definition = self.active_definition()
        for candidate in candidates:
            state = quantum_states.get(candidate.key)
            stage_state = self.stage_state(definition, candidate)
            _params, _advanced, extras = workflow_row_configuration(definition, candidate.key)
            source_rows.append(
                QuantumSourceRow(
                    key=candidate.key,
                    label=candidate.site.label or candidate.job.label,
                    charge_state=charge_state_label_for_job(candidate.job),
                    relative_energy=_format_relative_energy(
                        relative_energies.get(candidate.key)
                    ),
                    displacements=str(3 * len(candidate.structure.species)),
                    source_job=candidate.job.label,
                    stage=stage_state.stage,
                    num_farms_display=(
                        str(num_farms_from_extras(extras))
                        if stage_state.stage == QUANTUM_STAGE_PHONON
                        else ""
                    ),
                    status=getattr(state, "status", "ready") or "ready",
                    configured=workflow_row_configuration_exists(definition, candidate.key),
                    resettable=self.can_reset_to_stage_one(definition, candidate),
                    stage_two_resettable=self.can_reset_to_stage_two(definition, candidate),
                )
            )

        result_rows: list[QuantumResultRow] = []
        for result in document.results.values():
            if result.result_type != QUANTUM_PHONON_CALC_TYPE:
                continue
            job_label = ""
            job_status = "completed"
            run = (
                document.runs.get(result.calculation_run_id)
                if result.calculation_run_id is not None
                else None
            )
            if run is not None and run.job_id is not None:
                job = document.jobs.get(run.job_id)
                if job is not None:
                    job_label = job.label
                    job_status = job.status.value
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
                    zpe=format_energy_ev(
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
                    status=job_status,
                )
            )

        self._view.set_sources(source_rows)
        self._view.set_results(result_rows)



def _agreed(values: Sequence[object], *, default: object) -> object:
    """Return the value every entry of ``values`` shares, else ``default``."""

    if values and all(value == values[0] for value in values):
        return values[0]
    return default


__all__ = [
    "QUANTUM_STAGE_PHONON",
    "QUANTUM_STAGE_SINGLEPOINT",
    "QuantumConfigureRequest",
    "QuantumCorrectionsController",
    "QuantumCorrectionsDeps",
    "StageOneReset",
]
