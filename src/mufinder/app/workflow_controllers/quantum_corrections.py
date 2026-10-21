"""Quantum-corrections workflow controller.

Owns the quantum-corrections workflow: each relaxed muon source runs as one
CASTEP ``TASK : PHONON`` job, which converges the ground state and computes the
Gamma-point phonons in the same run.

- **Sources and rows.** Rows are the eligible relaxed muon sites from
  :mod:`mufinder.services.source_candidates`. A row is bound to its phonon job
  through the definition's ``phonon_job_ids_by_label`` (aliased as
  ``job_ids_by_label``), and its state is that job's projected status, or
  ``ready`` without one.
- **Materialize.** :meth:`QuantumCorrectionsController.materialize` builds a
  config per ready row, expands them through ``expand_quantum_phonon_jobs``
  and writes each job into the row's seed directory. A new job whose seed
  directory already exists gets a fresh seed, so it never starts from another
  run's checkpoint.
- **Submission guards.** :meth:`QuantumCorrectionsController.pre_submit_check`
  checks the active definition against the active host, refuses rows whose
  legacy singlepoint is still active, and checks each row's phonon method
  against its pseudopotentials.
- **Result.** :meth:`QuantumCorrectionsController.validate_artifacts` fails a
  phonon job whose ``.phonon`` is missing or unparseable;
  :meth:`QuantumCorrectionsController.build_result` contributes the muon ZPE,
  phonon eigenvalues, mode participations and the ``w1`` entanglement witness.
- **Configure.** :meth:`QuantumCorrectionsController.configure_request` and
  :meth:`QuantumCorrectionsController.apply_configuration` back the Configure
  dialog, which is editable while every selected row is ready and otherwise
  opens read-only with the reason.
- **Recovery.** A failed, stopped or cancelled phonon job goes back to
  ``defined`` through the shared failed-job reset.
  :meth:`QuantumCorrectionsController.directory_companions` lets that reset
  clean a directory the job shares with a legacy singlepoint.

Older projects ran a singlepoint job before the phonon job. Those
``castep_quantum_singlepoint`` jobs, bound through
``singlepoint_job_ids_by_label``, are still recognised: they collect cleanly,
keep their own directory, and a row whose singlepoint is still active is
refused submission until it finishes or is cancelled.

Task farming is applied by the shared submit backend, which reads each row's
farm count from its configuration extras.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.enums import ACTIVE_JOB_STATUSES, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.job_results import parse_numeric
from mufinder.services.quantum_results import (
    W1_DEFINITION,
    mode_widths_from_result_data,
    w1_from_result_data,
)
from mufinder.services.job_paths import QUANTUM_SINGLEPOINT_JOB_MAPPING, QUANTUM_SINGLEPOINT_SEED_MAPPING
from mufinder.services.muon_sites_summary import (
    QUANTUM_CORRECTIONS_SCOPE,
    downstream_site_scope,
    set_show_every_relaxed_site,
)
from mufinder.services.source_candidates import (
    QuantumSourceCandidate,
    charge_state_for_job,
    charge_state_label_for_job,
    eligible_quantum_source_candidates,
    quantum_seed_name_for_key,
    quantum_source_candidate_by_key,
    quantum_source_label,
    quantum_source_relative_energies,
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
from mufinder.app.workflow_submission import effective_row_status, workflow_row_state_is_ready
from mufinder.app.workflow_table_utils import RESET_FAILED_ACTION_TEXT

_CONFIG_KEYWORD_NAMES = workflow_basic_parameter_names()
_GUI_TO_METHOD = {
    "FINITEDISPLACEMENT": "finite_displacement",
    "DFPT": "dfpt",
    "LINEARRESPONSE": "linear_response",
}
_METHOD_TO_GUI = {value: key for key, value in _GUI_TO_METHOD.items()}
_DEFAULT_ELEC_ENERGY_TOL = ValueWithUnit(1e-9, "eV")
_PHONON_JOB_MAPPING = "phonon_job_ids_by_label"
_SEED_MAPPING = "job_seed_names_by_label"


def _method_from_gui_value(value: object) -> str:
    normalized = str(value).strip().upper()
    try:
        return _GUI_TO_METHOD[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported phonon method '{value}'.") from exc


def _method_to_gui_value(value: object) -> str:
    return _METHOD_TO_GUI.get(str(value).strip().lower(), "FINITEDISPLACEMENT")


def _write_job_directory(job_spec: JobSpec, output_root: Path, *, seed_name: str) -> Path:
    """Write a phonon job's ``.cell`` / ``.param`` into its seed directory."""

    job_dir = output_root / seed_name
    job_dir.mkdir(parents=True, exist_ok=True)
    cell_text, param_text = render_job(job_spec)
    (job_dir / f"{seed_name}.cell").write_text(cell_text, encoding="utf-8", newline="\n")
    (job_dir / f"{seed_name}.param").write_text(param_text, encoding="utf-8", newline="\n")
    return job_dir


def _fresh_seed_name(seed_name: str, *, output_root: Path, taken: set[str]) -> str:
    """Return a seed derived from ``seed_name`` that no directory or other row uses."""

    candidate = f"{seed_name}_phonon"
    index = 2
    while candidate in taken or (output_root / candidate).exists():
        candidate = f"{seed_name}_phonon_{index}"
        index += 1
    return candidate


def _format_zpe_ev(value: object) -> str:
    """Format the muon ZPE for the ``ZPE (eV)`` column: three decimals, no unit."""

    number = parse_numeric(value)
    return "" if number is None else f"{number:.3f}"


def _muon_mode_texts(data: Mapping[str, object]) -> tuple[str, str]:
    """Return the ``Muon modes (cm⁻¹)`` and ``Width (Å)`` texts of a phonon payload.

    The selected ``muon_zpe_modes`` frequencies (one decimal) and their
    zero-point widths (two decimals, from
    :func:`~mufinder.services.quantum_results.mode_widths_from_result_data`)
    are listed comma-separated, both in ascending frequency order.
    """

    modes = data.get("muon_zpe_modes")
    if not isinstance(modes, list):
        return "", ""
    frequencies = [
        parse_numeric(mode.get("frequency_cm")) for mode in modes if isinstance(mode, Mapping)
    ]
    widths = mode_widths_from_result_data(data)
    pairs = sorted(
        (frequency, width)
        for frequency, width in zip(frequencies, widths, strict=True)
        if frequency is not None
    )
    return (
        ", ".join(f"{frequency:.1f}" for frequency, _width in pairs),
        ", ".join(f"{width:.2f}" for _frequency, width in pairs),
    )


def _format_relative_energy(value: float | None) -> str:
    """Format a source's energy relative to its comparable minimum for the ``dE (eV)`` column."""

    return "" if value is None else f"{float(value):.5f}"


_DFPT_LIMITATIONS = (
    "CASTEP DFPT phonons need norm-conserving pseudopotentials and are not "
    "available with DFT+U, meta-GGA or hybrid functionals, non-collinear "
    "magnetism or spin-orbit coupling, or the newer dispersion corrections."
)


def _method_rows(rows: list[tuple[str, str]]) -> str:
    return "\n".join(f"  - {label} ({method})" for label, method in rows)


def _incompatible_method_message(rows: list[tuple[str, str]]) -> tuple[str, str]:
    """Return the refusal for rows using DFPT or linear response with ultrasoft pseudopotentials."""

    return (
        "Phonon method needs norm-conserving pseudopotentials",
        "These rows use a DFPT phonon method with ultrasoft pseudopotentials, which "
        "CASTEP does not support:\n\n"
        f"{_method_rows(rows)}\n\n"
        "Configure them to use FINITEDISPLACEMENT, or set Pseudopotentials to "
        "Norm-conserving if that is what the calculation uses.",
    )


def _unverified_method_question(rows: list[tuple[str, str]]) -> tuple[str, str]:
    """Return the confirmation for rows using DFPT with an unknown pseudopotential family."""

    return (
        "Check pseudopotentials for DFPT",
        "These rows use a DFPT phonon method, but their pseudopotential family is not "
        "set, so MuFinder cannot check it:\n\n"
        f"{_method_rows(rows)}\n\n"
        f"{_DFPT_LIMITATIONS} Submit anyway?",
    )


def _locked_reason(jobs: Sequence[Job | None], statuses: Sequence[str]) -> str:
    """Return why the Configure dialog opens read-only for rows in ``statuses``.

    ``jobs`` are the rows' jobs and ``statuses`` their projected row statuses,
    in the same order. The reason is one line naming what unlocks the first
    locked row; it is empty when every row is ready.
    """

    for job, status in zip(jobs, statuses, strict=True):
        if job is None or status in {"ready", "defined"}:
            continue
        if job.calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE:
            return (
                f"A selected row's singlepoint from an older MuFinder version is {status}; "
                "wait for it to finish or cancel it to edit the parameters."
            )
        if job.status in ACTIVE_JOB_STATUSES:
            return (
                f"Parameters are read-only while a selected row's phonon job is {status}; "
                f"cancel it, then use {RESET_FAILED_ACTION_TEXT} to edit them."
            )
        if job.status == JobStatus.COMPLETED:
            return (
                "A selected row's phonon job has completed; delete it from the Job Dashboard "
                "to run the row again with other parameters."
            )
        return f"A selected row's phonon job is {status}; use {RESET_FAILED_ACTION_TEXT} to edit its parameters."
    return ""


@dataclass(frozen=True)
class QuantumCorrectionsDeps:
    """The window-side handles the quantum-corrections controller needs.

    The host-mismatch handlers ask whether to switch the active host (shared
    with every host-guarded workflow) and ``confirm`` asks a yes/no question.
    Everything else the controller reads from the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`
    or the service layer.
    """

    handle_structure_host_mismatch: Callable[[object], bool]
    handle_site_host_mismatch: Callable[[object], bool]
    #: Ask a yes/no question (title, text); ``True`` means proceed. The safe
    #: answer, "no", is the default.
    confirm: Callable[[str, str], bool]


@dataclass(frozen=True)
class QuantumConfigureRequest(SourceConfigureRequest):
    """The quantum Configure dialog's seed: the shared request plus its own values.

    The dialog is editable when every selected row is ready (no job, or its job
    still ``defined``) and read-only otherwise; ``locked_reason`` says why it
    opened read-only and is empty when it is editable.
    """

    pseudopotential_family: str
    locked_reason: str


class QuantumCorrectionsController(WorkflowJobController):
    """Controller for the quantum-corrections workflow: one phonon job per row."""

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

    # --- sources and jobs ---

    def candidates(self) -> list[QuantumSourceCandidate]:
        """Return the eligible relaxed muon sources, in table order."""

        return eligible_quantum_source_candidates(self._context.document())

    def candidate_by_key(self, key: str) -> QuantumSourceCandidate | None:
        return quantum_source_candidate_by_key(self._context.document(), key)

    def source_label(self, candidate: QuantumSourceCandidate) -> str:
        return quantum_source_label(self._context.document(), candidate)

    def seed_name_for_key(self, definition: CalculationDefinition | None, key: str) -> str:
        return quantum_seed_name_for_key(self._context.document(), definition, key)

    def phonon_job_for_key(self, definition: CalculationDefinition | None, key: str) -> Job | None:
        """Return the row's phonon job, falling back to the ``job_ids_by_label`` alias."""

        job = self.job_for_key(
            definition,
            key,
            mapping_name=_PHONON_JOB_MAPPING,
            calc_type=QUANTUM_PHONON_CALC_TYPE,
        )
        if job is not None:
            return job
        return self.job_for_key(definition, key, calc_type=QUANTUM_PHONON_CALC_TYPE)

    def legacy_singlepoint_job_for_key(
        self, definition: CalculationDefinition | None, key: str
    ) -> Job | None:
        """Return the singlepoint an older MuFinder version ran for the row, if any."""

        return self.job_for_key(
            definition,
            key,
            mapping_name=QUANTUM_SINGLEPOINT_JOB_MAPPING,
            calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
        )

    def row_job(self, definition: CalculationDefinition | None, key: str) -> Job | None:
        """Return the job a row's state, Cancel and Inspect act on.

        That is the row's phonon job. A row without one whose legacy
        singlepoint is still active shows that singlepoint instead, so it is
        not ready and can be cancelled from the table.
        """

        phonon_job = self.phonon_job_for_key(definition, key)
        if phonon_job is not None:
            return phonon_job
        singlepoint = self.legacy_singlepoint_job_for_key(definition, key)
        if singlepoint is not None and singlepoint.status in ACTIVE_JOB_STATUSES:
            return singlepoint
        return None

    def definition_and_key_for_job(
        self, job: Job
    ) -> tuple[CalculationDefinition, str] | tuple[None, None]:
        """Return the definition and row key whose job mappings bind ``job``."""

        for definition in self.definitions():
            for mapping_name in (QUANTUM_SINGLEPOINT_JOB_MAPPING, _PHONON_JOB_MAPPING, "job_ids_by_label"):
                mapping = definition.parameters.get(mapping_name, {})
                if not isinstance(mapping, dict):
                    continue
                for key, value in mapping.items():
                    if str(value) == str(job.id):
                        return definition, str(key)
        return None, None

    def directory_companions(self, job: Job) -> list[Job]:
        """Return the legacy singlepoint that shares a phonon job's directory.

        An older project ran each row's singlepoint and phonon job in one seed
        directory. Resetting the phonon job with directory clean-up then empties
        that directory, singlepoint files included, so the row reruns as a fresh
        job; the singlepoint stays in the Job Dashboard as a record. An active
        singlepoint is never a companion.
        """

        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return []
        definition, key = self.definition_and_key_for_job(job)
        if definition is None or key is None:
            return []
        singlepoint = self.legacy_singlepoint_job_for_key(definition, key)
        if singlepoint is None or singlepoint.status in ACTIVE_JOB_STATUSES:
            return []
        directory = self._context.job_output_directory(job)
        if directory is None or self._context.job_output_directory(singlepoint) != directory:
            return []
        return [singlepoint]

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
                config["sources"] = serialize_quantum_sources(
                    document,
                    [*_stored_source_keys(config, definition), *(candidate.key for candidate in candidates)],
                )
        else:
            definition = self._context.project_service.register_quantum_phonon_definition(
                document,
                input_structure_ids=list(dict.fromkeys(candidate.structure.id for candidate in candidates)),
                input_site_ids=list(dict.fromkeys(candidate.site.id for candidate in candidates)),
                parameters={
                    "method": "finite_displacement",
                    "pseudopotential_family": "unknown",
                    "sources": serialize_quantum_sources(document, [candidate.key for candidate in candidates]),
                    "phonon_job_ids_by_label": {},
                    "job_seed_names_by_label": {},
                },
                output_directory=document.project_root / "files" / "quantum_corrections",
            )
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
        states = self.build_states()
        values_by_key: list[dict[str, object]] = []
        advanced_names: set[str] = set()
        pseudopotential_families: list[str] = []
        num_farms_values: list[int] = []
        for candidate in candidates:
            values, pseudopotential_family = self.runtime_configuration(definition, candidate.key)
            _params, row_advanced_names, extras = workflow_row_configuration(definition, candidate.key)
            values_by_key.append(values)
            advanced_names.update(row_advanced_names)
            pseudopotential_families.append(pseudopotential_family)
            num_farms_values.append(num_farms_from_extras(extras))
        read_only = any(not workflow_row_state_is_ready(states, candidate.key) for candidate in candidates)
        locked_reason = (
            _locked_reason(
                [self.row_job(definition, candidate.key) for candidate in candidates],
                [str(getattr(states.get(candidate.key), "status", "ready")) for candidate in candidates],
            )
            if read_only
            else ""
        )
        return QuantumConfigureRequest(
            keys=tuple(candidate.key for candidate in candidates),
            row_values=tuple(values_by_key),
            advanced_names=frozenset(advanced_names),
            title=(
                "Inspect Quantum Corrections Parameters" if read_only else "Configure Quantum Corrections Parameters"
            ),
            intro_text=(
                f"Selected sources: {len(candidates)}\n"
                f"Displacements required for the phonon run: {3 * len(candidates[0].structure.species)}"
                + (f"\n\n{locked_reason}" if locked_reason else "")
            ),
            read_only=read_only,
            num_farms=_agreed(num_farms_values, default=1),
            species_options=merge_species_options(candidate.structure for candidate in candidates),
            pseudopotential_family=_agreed(pseudopotential_families, default="") or "unknown",
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
    # apply. There is no submission_groups override: the rows submit as one
    # group, and task farming is applied by the shared submit backend.

    def normalize(self, value: object) -> str:
        return str(value)

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Validate the selection before anything is materialized.

        In order:

        - the active definition's structures and sites must belong to the
          active host (:func:`~mufinder.app.workflow_controllers.base.confirm_active_host`);
          a mismatch the user does not resolve refuses silently;
        - a row whose legacy singlepoint is still active is refused, with an
          explanation naming it, until that job finishes or is cancelled;
        - each ready row's phonon method is checked against its
          pseudopotential family (see :meth:`_phonon_method_issues`): DFPT or
          linear response with ultrasoft pseudopotentials is refused with an
          explanation naming the rows, and with an unknown family the user is
          asked to confirm.
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

        active_legacy = self._rows_with_active_legacy_singlepoint(keys)
        if active_legacy:
            listing = "\n".join(f"  - {label} ({status})" for label, status in active_legacy)
            self._context.show_info(
                "Singlepoint still in progress",
                "These rows still have a singlepoint job from an older MuFinder version in progress:\n\n"
                f"{listing}\n\n"
                "Wait for it to finish, or cancel it, before submitting the row. The row then "
                "runs as a single phonon job in a fresh directory.",
            )
            return False

        incompatible, unverified = self._phonon_method_issues(keys)
        if incompatible:
            self._context.show_info(*_incompatible_method_message(incompatible))
            return False
        if unverified and not self._deps.confirm(*_unverified_method_question(unverified)):
            return False
        return True

    def _rows_with_active_legacy_singlepoint(self, keys: list[object]) -> list[tuple[str, str]]:
        """Return ``(label, status)`` for selected rows whose legacy singlepoint is active."""

        definition = self.active_definition()
        if definition is None:
            return []
        rows: list[tuple[str, str]] = []
        for key in [str(value) for value in keys]:
            candidate = self.candidate_by_key(key)
            if candidate is None:
                continue
            job = self.row_job(definition, key)
            if job is not None and job.calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE:
                rows.append((self.source_label(candidate), job.status.value))
        return rows

    def _phonon_method_issues(
        self, keys: list[object]
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Split ready rows' phonon methods into incompatible and unverified.

        Each entry is ``(label, CASTEP method)``. Only rows that could be
        submitted now count. A method is incompatible when
        :func:`~mufinder.workflow.quantum.phonon_method_compatibility` rejects
        it for the row's pseudopotential family (DFPT or linear response with
        ultrasoft pseudopotentials), and unverified when it is accepted with a
        caveat (the family is not set).
        """

        definition = self.active_definition()
        states = self.build_states()
        incompatible: list[tuple[str, str]] = []
        unverified: list[tuple[str, str]] = []
        for key in [str(value) for value in keys]:
            candidate = self.candidate_by_key(key)
            if candidate is None or not workflow_row_state_is_ready(states, key):
                continue
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
            entry = (self.source_label(candidate), gui_method)
            if not compatible:
                incompatible.append(entry)
            elif message:
                unverified.append(entry)
        return incompatible, unverified

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
        """Return each row's job (see :meth:`row_job`) across every definition, once each."""

        if self._context.document() is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        str_keys = [str(key) for key in keys]
        for definition in self.definitions():
            for key in str_keys:
                job = self.row_job(definition, key)
                if job is None or job.id in seen:
                    continue
                selected.append(job)
                seen.add(job.id)
        return selected

    def materialize(self, keys: list[object]) -> list[Job]:
        """Write the phonon job of each ready row ``keys`` names, creating it when needed.

        A row's existing job is reused and, while it is ``defined``, its inputs
        are rewritten from the row's configuration. A new job starts in a
        directory of its own: when the row's seed directory already exists (a
        legacy singlepoint ran there), the row gets a fresh seed, and the
        legacy singlepoint's seed is recorded so it keeps its directory.
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
        configs: list[QuantumPhononJobConfig] = []
        config_keys: list[str] = []
        for key in str_keys:
            candidate = self.candidate_by_key(key)
            if candidate is None or not workflow_row_state_is_ready(states, key):
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
            configs.append(
                QuantumPhononJobConfig(
                    source=QuantumPhononSource(
                        label=self.source_label(candidate),
                        structure=structure,
                        source_site_id=candidate.site.id,
                        source_relaxation_run_id=candidate.run.id,
                        source_relaxation_job_id=candidate.job.id,
                        charge_state=charge_state_for_job(candidate.job),
                        total_energy_eV=candidate.total_energy_eV,
                    ),
                    method=method,
                    pseudopotential_family=str(pseudopotential_family),
                    num_farms=num_farms_from_extras(extras),
                    cell_overrides=overrides.cell,
                    param_overrides=overrides.param,
                )
            )
            config_keys.append(key)
        if not configs:
            return []

        try:
            specs = expand_quantum_phonon_jobs(jobs=configs, name_prefix="quantum_phonon")
        except (ValueError, StructureBridgeError):
            return []

        phonon_mapping = _definition_mapping(definition, _PHONON_JOB_MAPPING)
        seed_mapping = _definition_mapping(definition, _SEED_MAPPING)
        # Load-bearing: ``phonon_job_for_key`` falls back to reading
        # ``job_ids_by_label``, so keep the phonon mapping aliased there.
        definition.parameters["job_ids_by_label"] = phonon_mapping

        # One spec per config, in order; a site relaxed in both charge states
        # has two keys with the same source site id.
        specs_by_key = dict(zip(config_keys, specs, strict=True))
        created: list[Job] = []
        for key in str_keys:
            spec = specs_by_key.get(key)
            if spec is None:
                continue
            existing = self.phonon_job_for_key(definition, key)
            if existing is not None:
                if existing.status == JobStatus.DEFINED:
                    existing.label = spec.name
                    _write_job_directory(spec, output_root, seed_name=self.seed_name_for_key(definition, key))
                created.append(existing)
                continue
            seed_name = self._seed_for_new_job(definition, key, output_root=output_root)
            seed_mapping[key] = seed_name
            _write_job_directory(spec, output_root, seed_name=seed_name)
            job = self._context.project_service.add_job(
                document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type=QUANTUM_PHONON_CALC_TYPE,
            )
            phonon_mapping[key] = str(job.id)
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

    def _seed_for_new_job(self, definition: CalculationDefinition, key: str, *, output_root: Path) -> str:
        """Return the seed a row's new phonon job is written under.

        The row's seed, unless its directory already exists: then a fresh seed,
        after recording the row's legacy singlepoint (when it has one) under the
        old seed so that job keeps its directory.
        """

        seed_name = self.seed_name_for_key(definition, key)
        if not (output_root / seed_name).exists():
            return seed_name
        if self.legacy_singlepoint_job_for_key(definition, key) is not None:
            legacy_seeds = _definition_mapping(definition, QUANTUM_SINGLEPOINT_SEED_MAPPING)
            legacy_seeds.setdefault(key, seed_name)
        seed_mapping = _definition_mapping(definition, _SEED_MAPPING)
        taken = {str(value) for other, value in seed_mapping.items() if other != key}
        return _fresh_seed_name(seed_name, output_root=output_root, taken=taken)

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
        witness computation during collection. A legacy singlepoint job is
        accepted.
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

        A legacy quantum-*singlepoint* job, which an older project may still be
        running, contributes nothing (an empty
        :class:`~mufinder.app.workflow_controllers.base.ControllerResult`, a strict
        no-op): its result is the base payload only.

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
                    "width_A": mode.width_A,
                }
                for mode in zpe.modes
            ],
            "w1_entanglement_witness": w1_entanglement_witness(phonon),
            "w1_definition": W1_DEFINITION,
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

    def set_show_every_site(self, enabled: bool) -> None:
        """List every eligible relaxed site (``True``) or only the pinned ones."""

        document = self._context.document()
        if document is None:
            return
        set_show_every_relaxed_site(document, QUANTUM_CORRECTIONS_SCOPE, enabled)
        self.refresh_view()
        self._context.persist_ui_state(
            failure_message="The Quantum Corrections site list was updated, but autosave failed."
        )

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
            self._view.set_site_scope(show_every_site=False)
            self._view.set_sources([])
            self._view.set_results([])
            return

        source_rows: list[QuantumSourceRow] = []
        quantum_states = self.build_states()
        candidates = self.candidates()
        relative_energies = quantum_source_relative_energies(candidates)
        definition = self.active_definition()
        # The pinned list keeps every row this workflow already tracks, so a
        # configured or submitted site never drops out of the table.
        tracked_keys = [
            candidate.key
            for candidate in candidates
            if self.phonon_job_for_key(definition, candidate.key) is not None
            or self.legacy_singlepoint_job_for_key(definition, candidate.key) is not None
            or workflow_row_configuration_exists(definition, candidate.key)
        ]
        scope = downstream_site_scope(document, QUANTUM_CORRECTIONS_SCOPE, candidates)
        self._view.set_site_scope(show_every_site=scope.show_every_site, notice=scope.notice)
        for candidate in scope.scoped_candidates(candidates, keep_keys=tracked_keys):
            state = quantum_states.get(candidate.key)
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
                    num_farms_display=str(num_farms_from_extras(extras)),
                    status=getattr(state, "status", "ready") or "ready",
                    configured=workflow_row_configuration_exists(definition, candidate.key),
                    pinned_label=scope.pinned_label(candidate.key),
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
            data = result.data if isinstance(result.data, dict) else {}
            muon_modes, widths = _muon_mode_texts(data)
            # A payload stored before ``w1_definition`` holds the witness in an
            # earlier form; ``w1_from_result_data`` recomputes it as ``P - 3``.
            witness = w1_from_result_data(data)
            result_rows.append(
                QuantumResultRow(
                    key=str(result.id),
                    label=job_label or "Quantum phonon",
                    zpe=_format_zpe_ev(data.get("muon_zpe_total_eV")),
                    muon_modes=muon_modes,
                    widths=widths,
                    witness="" if witness is None else f"{witness:.4g}",
                    status=job_status,
                )
            )

        self._view.set_sources(source_rows)
        self._view.set_results(result_rows)



def _stored_source_keys(config: dict, definition: CalculationDefinition) -> list[str]:
    """Return the source keys a definition's stored ``sources`` already select.

    An entry names its ``key``; an entry saved before entries carried one is
    keyed by its ``site_id``. A definition with no stored ``sources`` list falls
    back to its input site ids.
    """

    stored = config.get("sources")
    if not isinstance(stored, list):
        return [str(site_id) for site_id in definition.input_site_ids]
    return [
        str(entry.get("key") or entry.get("site_id") or "")
        for entry in stored
        if isinstance(entry, dict) and (entry.get("key") or entry.get("site_id"))
    ]


def _definition_mapping(definition: CalculationDefinition, name: str) -> dict[str, object]:
    """Return ``definition``'s ``name`` mapping, replacing a missing or malformed one."""

    mapping = definition.parameters.setdefault(name, {})
    if not isinstance(mapping, dict):
        mapping = {}
        definition.parameters[name] = mapping
    return mapping


def _agreed(values: Sequence[object], *, default: object) -> object:
    """Return the value every entry of ``values`` shares, else ``default``."""

    if values and all(value == values[0] for value in values):
        return values[0]
    return default


__all__ = [
    "QuantumConfigureRequest",
    "QuantumCorrectionsController",
    "QuantumCorrectionsDeps",
]
