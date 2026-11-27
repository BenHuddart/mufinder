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
  legacy singlepoint is still active, and tells the user when a row an older
  version configured for DFPT or linear response will run as finite
  displacement, the only phonon method the workflow writes.
- **Result.** :meth:`QuantumCorrectionsController.validate_artifacts` fails a
  phonon job whose ``.phonon`` is missing or unparseable;
  :meth:`QuantumCorrectionsController.build_result` contributes the muon ZPE,
  phonon eigenvalues, mode participations and the ``w1`` entanglement witness.
  :meth:`QuantumCorrectionsController.open_analysis` opens the analysis window.
- **Configure.** :meth:`QuantumCorrectionsController.configure_sources` runs the
  Configure dialog from :meth:`QuantumCorrectionsController.configure_request`
  and stores it through :meth:`QuantumCorrectionsController.apply_configuration`;
  it is editable unless a selected row's job is active, and then opens read-only
  with the reason.
- **Changing a completed row.** A completed row stays configurable and runs
  again with other parameters through the shared route of
  :class:`~mufinder.app.workflow_controllers.completed_row_edits.CompletedRowEditsController`:
  changing it asks first, the row reads ``edited`` and keeps its phonon result
  until it runs again, and the result goes once that submission is accepted.
- **Recovery.** A failed, stopped or cancelled phonon job runs again from its
  row or the dashboard's ``Resubmit…``.
  :meth:`QuantumCorrectionsController.directory_companions` lets starting it
  again clean a directory the job shares with a legacy singlepoint.

Older projects ran a singlepoint job before the phonon job. Those
``castep_quantum_singlepoint`` jobs, bound through
``singlepoint_job_ids_by_label``, are still recognised: they collect cleanly,
keep their own directory, and a row whose singlepoint is still active is
refused submission until it finishes or is cancelled.

Task farming is applied by the shared submit backend, which reads each row's
farm count from its configuration extras.
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
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
    Result,
    Structure,
)
from mufinder.domain.text import natural_sort_key
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.job_monitoring import job_status_word
from mufinder.services.job_results import parse_numeric
from mufinder.services.quantum_results import (
    W1_DEFINITION,
    mode_widths_from_result_data,
    phonon_warnings_from_result_data,
    w1_from_result_data,
)
from mufinder.services.job_paths import (
    QUANTUM_SINGLEPOINT_JOB_MAPPING,
    QUANTUM_SINGLEPOINT_SEED_MAPPING,
)
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
from mufinder.services.zpe_ranking import (
    PHONON_JOB_MAPPING,
    PHONON_SOURCE_RUN_MAPPING,
    phonon_job_for_row,
    zpe_ranking,
)
from mufinder.castep_io.param_parser import ParamParseError, parse_param_assignments
from mufinder.science.quantum_phonons import (
    PhononParseError,
    muon_mass_warning,
    muon_zpe,
    w1_entanglement_witness,
)
from mufinder.workflow import StructureBridgeError, render_job, structure_to_spec
from mufinder.workflow.quantum import (
    QUANTUM_PHONON_CALC_TYPE,
    QUANTUM_PHONON_METHOD,
    QUANTUM_PHONON_SET_CALC_TYPE,
    QUANTUM_SINGLEPOINT_CALC_TYPE,
    QuantumPhononJobConfig,
    QuantumPhononSource,
    expand_quantum_phonon_jobs,
)
from mufinder.workflow.types import JobSpec

from mufinder.app.adapter_common import build_workflow_settings_template
from mufinder.app.quantum_analysis_dialog import QuantumPhononAnalysisDialog
from mufinder.app.quantum_corrections_widget import (
    NO_VALUE_TEXT,
    QuantumCorrectionsWidget,
    QuantumSourceRow,
)
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.app.workflow_config_codecs import (
    num_farms_from_extras,
    source_row_parameter_values,
    store_workflow_row_configuration,
    workflow_row_configuration,
    workflow_row_configuration_exists,
)
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    PlannedJob,
    WorkflowControllerContext,
    confirm_active_host,
    dedupe_jobs,
    locked_configuration_reason,
)
from mufinder.app.workflow_controllers.completed_row_edits import (
    CompletedRowEditsController,
)
from mufinder.app.workflow_dialogs import (
    QuantumCorrectionsConfigDialog,
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
    CONFIGURABLE_WORKFLOW_ROW_STATUSES,
    EDITED_ROW_STATUS,
    effective_row_status,
    workflow_row_state_is_ready,
)

_CONFIG_KEYWORD_NAMES = workflow_basic_parameter_names()
#: ``PHONON_METHOD`` is fixed at finite displacement, so the Configure dialog
#: neither shows it nor offers it under ``Add Parameter``.
_EXCLUDED_KEYWORD_NAMES = ("task", "phonon_method")
#: Stored phonon-method spellings an older configuration wrote, mapped to the
#: CASTEP value: the dialog's ``phonon_method`` row stored the CASTEP value and
#: older row extras and definitions stored the lower-case internal name.
_LEGACY_METHOD_TO_CASTEP = {
    "finite_displacement": "FINITEDISPLACEMENT",
    "dfpt": "DFPT",
    "linear_response": "LINEARRESPONSE",
}
_DEFAULT_ELEC_ENERGY_TOL = ValueWithUnit(1e-9, "eV")
_SEED_MAPPING = "job_seed_names_by_label"
#: What an edited row's tooltip adds below its result.
EDITED_ROW_NOTE = "Parameters changed since this result; running the row again replaces it."


def _legacy_method_value(value: object) -> str:
    """Return the CASTEP spelling of a phonon method an older configuration stored."""

    text = str(value).strip()
    return _LEGACY_METHOD_TO_CASTEP.get(text.lower(), text.upper())


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


def _format_three_decimals(value: float | None, *, signed: bool = False) -> str:
    """Format a number to three decimals: ``–`` without one, and an unsigned ``0.000`` at zero."""

    if value is None:
        return NO_VALUE_TEXT
    if round(value, 3) == 0:
        return "0.000"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _muon_mode_texts(data: Mapping[str, object]) -> tuple[str, str]:
    """Return the muon-mode frequencies and ``Widths (Å)`` texts of a phonon payload.

    The selected ``muon_zpe_modes`` frequencies (one decimal, comma-separated) and
    their zero-point widths (two decimals, space-separated, from
    :func:`~mufinder.services.quantum_results.mode_widths_from_result_data`), both
    in ascending frequency order.
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
        " ".join(f"{width:.2f}" for _frequency, width in pairs),
    )


def _result_cells(result: Result) -> dict[str, object]:
    """Return a row's fields read from its stored phonon result, ``dE + ZPE`` aside."""

    data = result.data if isinstance(result.data, dict) else {}
    muon_modes, widths = _muon_mode_texts(data)
    return {
        "zpe": _format_three_decimals(parse_numeric(data.get("muon_zpe_total_eV"))),
        "widths": widths or NO_VALUE_TEXT,
        # A payload stored before ``w1_definition`` holds the witness in an earlier
        # form; ``w1_from_result_data`` recomputes it as ``P - 3``.
        "witness": _format_three_decimals(w1_from_result_data(data)),
        "muon_modes": muon_modes,
        "result_id": str(result.id),
        "warnings": phonon_warnings_from_result_data(data),
    }


def _legacy_method_notice(rows: list[tuple[str, str]]) -> tuple[str, str]:
    """Return the one-line notice for rows an older version configured for another method."""

    listing = ", ".join(f"{label} ({method})" for label, method in rows)
    return (
        "Phonon method changed",
        "Running as finite displacement (FINITEDISPLACEMENT), the only method "
        f"Quantum Corrections uses: {listing}.",
    )


def _locked_reason(jobs: Sequence[Job | None], statuses: Sequence[str]) -> str:
    """Return why the Configure dialog opens read-only for rows in ``statuses``.

    A row whose only job is a legacy singlepoint names that job; every other
    row gets the shared reason for its phonon job (see
    :func:`~mufinder.app.workflow_controllers.base.locked_configuration_reason`).
    """

    for job, status in zip(jobs, statuses, strict=True):
        if job is None or status in CONFIGURABLE_WORKFLOW_ROW_STATUSES:
            continue
        if job.calc_type == QUANTUM_SINGLEPOINT_CALC_TYPE:
            return (
                f"A selected row's singlepoint from an older MuFinder version is {status}; "
                "wait for it to finish or cancel it to edit the parameters."
            )
        return locked_configuration_reason([job], [status], job_noun="phonon job")
    return ""


@dataclass(frozen=True)
class QuantumCorrectionsDeps:
    """The window-side handles the quantum-corrections controller needs.

    The host-mismatch handlers ask whether to switch the active host (shared
    with every host-guarded workflow). ``refresh_site_label_views`` refreshes the
    other pages that list a renamed site, ``show_warning`` says why an analysis window
    cannot open, and ``shutting_down`` stops a rename saving during teardown. Everything else the controller reads from
    the shared :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`
    or the service layer.
    """

    handle_structure_host_mismatch: Callable[[object], bool]
    handle_site_host_mismatch: Callable[[object], bool]
    refresh_site_label_views: Callable[[], None]
    show_warning: Callable[[str, str], object]
    shutting_down: Callable[[], bool]


@dataclass(frozen=True)
class QuantumConfigureRequest(SourceConfigureRequest):
    """The quantum Configure dialog's seed: the shared request plus its own values.

    The dialog is editable unless a selected row's job is active, or is a
    completed job whose inputs cannot be read; ``locked_reason`` says why it
    opened read-only and is empty when it is editable.
    """

    locked_reason: str


class QuantumCorrectionsController(CompletedRowEditsController):
    """Controller for the quantum-corrections workflow: one phonon job per row."""

    calc_types = frozenset({QUANTUM_SINGLEPOINT_CALC_TYPE, QUANTUM_PHONON_CALC_TYPE})
    output_folder = "quantum_corrections"
    stage = WorkflowStage.QUANTUM_CORRECTIONS
    definition_calc_type = QUANTUM_PHONON_SET_CALC_TYPE
    no_ready_message = (
        "Only eligible muon sites that are ready, failed, stopped or cancelled can be submitted."
    )
    fixed_keyword_names = _EXCLUDED_KEYWORD_NAMES
    site_scope = QUANTUM_CORRECTIONS_SCOPE
    result_noun = "phonon result"
    result_noun_plural = "phonon results"
    replaced_result_detail = (
        "Running again deletes the phonon result, its ZPE and what the page derives "
        "from it (dE + ZPE and the ranking), and clears the job's folder. Until that "
        "run's submission is accepted, the row keeps its current result, and setting "
        "the parameters back keeps it for good."
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
        self._analysis_dialogs: dict[UUID, QuantumPhononAnalysisDialog] = {}

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

        document = self._context.document()
        return None if document is None else phonon_job_for_row(document, definition, key)

    def result_job_for_key(self, definition: CalculationDefinition | None, key: str) -> Job | None:
        """Return the row's phonon job: running the row again replaces its result."""

        return self.phonon_job_for_key(definition, key)

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
        if singlepoint is not None and singlepoint.status == JobStatus.ACTIVE:
            return singlepoint
        return None

    def definition_and_key_for_job(
        self, job: Job
    ) -> tuple[CalculationDefinition, str] | tuple[None, None]:
        """Return the definition and row key whose job mappings bind ``job``."""

        for definition in self.definitions():
            for mapping_name in (
                QUANTUM_SINGLEPOINT_JOB_MAPPING,
                PHONON_JOB_MAPPING,
                "job_ids_by_label",
            ):
                mapping = definition.parameters.get(mapping_name, {})
                if not isinstance(mapping, dict):
                    continue
                for key, value in mapping.items():
                    if str(value) == str(job.id):
                        return definition, str(key)
        return None, None

    mapping_name = PHONON_JOB_MAPPING

    def directory_companions(self, job: Job) -> list[Job]:
        """Return the legacy singlepoint that shares a phonon job's directory.

        An older project ran each row's singlepoint and phonon job in one seed
        directory. Starting the phonon job again empties that directory,
        singlepoint files included, so the row reruns as a fresh job; the singlepoint stays in the Job Dashboard as a record. An active
        singlepoint is never a companion.
        """

        if job.calc_type != QUANTUM_PHONON_CALC_TYPE:
            return []
        definition, key = self.definition_and_key_for_job(job)
        if definition is None or key is None:
            return []
        singlepoint = self.legacy_singlepoint_job_for_key(definition, key)
        if singlepoint is None or singlepoint.status == JobStatus.ACTIVE:
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
            excluded_names=_EXCLUDED_KEYWORD_NAMES,
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

        The row's stored values sit over its source relaxation's settings, with
        the workflow's tight ``elec_energy_tol`` default. The keywords the
        workflow writes itself are left out, including a ``phonon_method`` an
        older configuration stored: every row runs finite displacement (see
        :meth:`legacy_phonon_method`).
        """

        candidate = self.candidate_by_key(key)
        return source_row_parameter_values(
            definition,
            key,
            source_relaxation_parameter_values(candidate.job, self._context.job_output_directory)
            if candidate is not None
            else {},
            dropped=_EXCLUDED_KEYWORD_NAMES,
            defaults={"elec_energy_tol": _DEFAULT_ELEC_ENERGY_TOL},
        )

    @staticmethod
    def legacy_phonon_method(definition: CalculationDefinition | None, key: str) -> str | None:
        """Return the phonon method an older configuration chose for a row, unless it is ours.

        Older versions offered DFPT and linear response. The row's stored
        ``phonon_method`` parameter wins, then a ``phonon_method`` in its
        extras, then the definition's ``configuration["method"]``. Returns the
        CASTEP spelling, or ``None`` when nothing is stored or the stored
        method is finite displacement.
        """

        params, _advanced, extras = workflow_row_configuration(definition, key)
        config = definition.parameters.get("configuration", {}) if definition is not None else {}
        for stored in (
            params.get("phonon_method"),
            extras.get("phonon_method"),
            config.get("method") if isinstance(config, dict) else None,
        ):
            if stored is None or not str(stored).strip():
                continue
            method = _legacy_method_value(stored)
            return None if method == QUANTUM_PHONON_METHOD else method
        return None

    def ensure_definition(
        self, source_keys: Sequence[str], *, hold_record: bool = False
    ) -> CalculationDefinition | None:
        """Return the active definition with the selected sources added, creating one if needed.

        Returns ``None`` when there is no document or none of ``source_keys`` is
        an eligible source. With ``hold_record`` a new definition's "Configured …"
        record is held until the submission is accepted, as a Run's is.
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
        if definition is not None:
            for candidate in candidates:
                if candidate.structure.id not in definition.input_structure_ids:
                    definition.input_structure_ids.append(candidate.structure.id)
                if candidate.site.id not in definition.input_site_ids:
                    definition.input_site_ids.append(candidate.site.id)
            config = definition.parameters.setdefault("configuration", {})
            if isinstance(config, dict):
                config["sources"] = serialize_quantum_sources(
                    document,
                    [
                        *_stored_source_keys(config, definition),
                        *(candidate.key for candidate in candidates),
                    ],
                )
        else:
            definition = self._context.project_service.register_quantum_phonon_definition(
                document,
                input_structure_ids=list(
                    dict.fromkeys(candidate.structure.id for candidate in candidates)
                ),
                input_site_ids=list(dict.fromkeys(candidate.site.id for candidate in candidates)),
                parameters={
                    "sources": serialize_quantum_sources(
                        document, [candidate.key for candidate in candidates]
                    ),
                    "phonon_job_ids_by_label": {},
                    "job_seed_names_by_label": {},
                },
                output_directory=document.project_root / "files" / "quantum_corrections",
                **({"hold_record": self._held_records.append} if hold_record else {}),
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
        num_farms_values: list[int] = []
        for candidate in candidates:
            _params, row_advanced_names, extras = workflow_row_configuration(
                definition, candidate.key
            )
            values_by_key.append(self.runtime_configuration(definition, candidate.key))
            advanced_names.update(
                name for name in row_advanced_names if name not in _EXCLUDED_KEYWORD_NAMES
            )
            num_farms_values.append(num_farms_from_extras(extras))
        configurable = {
            candidate.key: self.row_is_configurable(states, definition, candidate.key)
            for candidate in candidates
        }
        read_only = not all(configurable.values())
        locked_reason = (
            _locked_reason(
                [
                    None if configurable[candidate.key] else self.row_job(definition, candidate.key)
                    for candidate in candidates
                ],
                [
                    str(getattr(states.get(candidate.key), "status", "ready"))
                    for candidate in candidates
                ],
            )
            if read_only
            else ""
        )
        return QuantumConfigureRequest(
            keys=tuple(candidate.key for candidate in candidates),
            row_values=tuple(values_by_key),
            advanced_names=frozenset(advanced_names),
            title=(
                "Inspect Quantum Corrections Parameters"
                if read_only
                else "Configure Quantum Corrections Parameters"
            ),
            intro_text=(
                f"Selected sources: {len(candidates)}\n"
                f"Displacements required for the phonon run: {3 * len(candidates[0].structure.species)}"
                + (f"\n\n{locked_reason}" if locked_reason else "")
            ),
            read_only=read_only,
            num_farms=_agreed(num_farms_values, default=1),
            species_options=merge_species_options(candidate.structure for candidate in candidates),
            locked_reason=locked_reason,
        )

    def apply_configuration(
        self,
        keys: Sequence[str],
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> None:
        """Store an accepted Configure dialog on every row ``keys`` name.

        ``extras`` are the dialog's workflow values; without ``num_farms``, the farm
        count was left alone and each row keeps its own. A mixed parameter the user left
        alone keeps each row's own stored value. A completed row the dialog leaves as it
        was stores nothing; when no row changes, nothing is saved either.
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
            failure_message="Quantum Corrections configuration was updated, but autosave failed."
        )

    def dialog_parent(self) -> object:
        """Return the window the Configure dialog and its question open over."""

        return self._view.window()

    def run_configure_dialog(self, request: SourceConfigureRequest) -> SourceConfigureResult | None:
        """Run the Quantum Corrections Configure dialog, with its farm count, on ``request``."""

        return run_source_configure_dialog(
            partial(
                QuantumCorrectionsConfigDialog,
                specs=self.parameter_specs(),
                num_farms=request.num_farms,
                intro_text=request.intro_text,
                read_only=request.read_only,
                species_options=request.species_options,
                parent=self.dialog_parent(),
            ),
            request,
            extras=lambda dialog: (
                {"num_farms": dialog.num_farms()} if dialog.num_farms_edited() else {}
            ),
        )

    def row_payload_extras(
        self, definition: CalculationDefinition, key: str, extras: Mapping[str, object]
    ) -> dict[str, object]:
        """Return the dialog's extras over row ``key``'s own farm count, which it keeps otherwise."""

        own_farms = num_farms_from_extras(workflow_row_configuration(definition, key)[2])
        return {"num_farms": own_farms, **extras}

    def row_inputs(
        self, definition: CalculationDefinition | None, key: str
    ) -> JobInputTexts | None:
        """Render the ``.cell`` and ``.param`` row ``key``'s job would be written with now.

        A pure preview of :meth:`materialize` for one row: nothing is stored or
        written. ``None`` without an eligible source or when the row's settings
        do not build.
        """

        config = self._phonon_job_config(definition, key)
        if config is None:
            return None
        try:
            (spec,) = expand_quantum_phonon_jobs(jobs=[config], name_prefix="quantum_phonon")
        except (ValueError, StructureBridgeError):
            return None
        cell_text, param_text = render_job(spec)
        return JobInputTexts(cell=cell_text, param=param_text)

    def rename_source(self, key: str, label: str) -> bool:
        """Rename a source's muon site from its editable label cell.

        Returns whether the label changed. A blank or unchanged label restores
        the cell; an unknown key does nothing. A new label also reaches the other
        pages that list the site, and is saved.
        """

        if self._deps.shutting_down():
            return False
        candidate = self.candidate_by_key(key)
        if candidate is None:
            return False
        normalized = str(label).strip()
        if not normalized or normalized == candidate.site.label:
            self.refresh_view()
            return False
        candidate.site.label = normalized
        self.refresh_view()
        self._deps.refresh_site_label_views()
        self._context.persist_ui_state(
            failure_message="Quantum Corrections site label was updated, but autosave failed."
        )
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
        """Return the ``.phonon`` file and the name the analysis window opens on.

        The name is the site's and charge state's (see :meth:`site_title`), or the
        job's label for a result whose site is gone. Raises ``ValueError`` with a
        user-facing reason when the result names no phonon file or the file is
        gone. A relative path is read from the project root.
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
        title = "Quantum phonon"
        if document is not None:
            run = (
                document.runs.get(result.calculation_run_id)
                if result.calculation_run_id is not None
                else None
            )
            job = (
                document.jobs.get(run.job_id)
                if run is not None and run.job_id is not None
                else None
            )
            if job is not None:
                _definition, key = self.definition_and_key_for_job(job)
                title = (self.site_title(key) if key is not None else "") or job.label
        return phonon_path, title

    def _analysis_site_context(
        self, result: Result
    ) -> tuple[Structure | None, Structure | None, str, str]:
        """Return the host, the site's relaxed cell, and the table's ZPE mark and note.

        The analysis window names the principal axes on the relaxed cell's image of the
        site, and marks its numbers as the table marks the row's ``ZPE``. A result with
        no listed site gives no relaxed cell and no mark.
        """

        document = self._context.document()
        if document is None:
            return None, None, "", ""
        host_id = document.project.host_structure_id
        host = document.structures.get(host_id) if host_id is not None else None
        run = (
            document.runs.get(result.calculation_run_id)
            if result.calculation_run_id is not None
            else None
        )
        job = document.jobs.get(run.job_id) if run is not None and run.job_id else None
        key = self.definition_and_key_for_job(job)[1] if job is not None else None
        candidate = self.candidate_by_key(key) if key else None
        if candidate is None:
            return host, None, "", ""
        ranking = zpe_ranking(document, self.candidates())
        if ranking.results.get(candidate.key) is not result:
            return host, candidate.structure, "", ""
        mark = ranking.mark(candidate.key)
        note = ranking.notes(candidate.key)[0] if mark and ranking.notes(candidate.key) else ""
        return host, candidate.structure, mark, note

    def analysis_dialog(self, result_id: UUID) -> QuantumPhononAnalysisDialog | None:
        """Return the analysis window open on the result ``result_id``, if any."""

        return self._analysis_dialogs.get(result_id)

    def open_analysis(self, result_id: str) -> None:
        """Open the analysis window of a completed result, or raise the one already open."""

        result = self.phonon_result(result_id)
        if result is None:
            return
        existing = self._analysis_dialogs.get(result.id)
        if existing is not None:
            if existing.isVisible():
                existing.raise_()
                existing.activateWindow()
                return
            self._analysis_dialogs.pop(result.id, None)
            existing.deleteLater()

        try:
            phonon_path, title = self.phonon_analysis_input(result)
            host, relaxed_site, mark, note = self._analysis_site_context(result)
            dialog = QuantumPhononAnalysisDialog(
                phonon_path=str(phonon_path),
                site_label=title,
                result_data=result.data if isinstance(result.data, dict) else {},
                host=host,
                relaxed_site=relaxed_site,
                zpe_mark=mark,
                zpe_note=note,
                parent=self._view.window(),
            )
        except (ValueError, PhononParseError) as exc:
            self._deps.show_warning("Phonon analysis unavailable", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self._deps.show_warning(
                "Phonon analysis unavailable", f"Could not open the analysis window: {exc}"
            )
            return

        # A closed window is deleted later, by when the result may have a new one.
        dialog.destroyed.connect(
            lambda *_args, opened=result.id, window=dialog: (
                self._analysis_dialogs.pop(opened, None)
                if self._analysis_dialogs.get(opened) is window
                else None
            )
        )
        self._analysis_dialogs[result.id] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

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

        - a selection of only ``completed`` rows (their parameters unchanged
          since their run) has nothing to run: one notice says so and how to run
          such a row again (:meth:`nothing_to_run`), instead of the
          backends' generic "No eligible jobs";
        - the active definition's structures and sites must belong to the
          active host (:func:`~mufinder.app.workflow_controllers.base.confirm_active_host`);
          a mismatch the user does not resolve refuses silently;
        - a row whose legacy singlepoint is still active is refused, with an
          explanation naming it, until that job finishes or is cancelled;
        - a ready row an older version configured for DFPT or linear
          response (see :meth:`legacy_phonon_method`) is not refused: a
          one-line notice names it and says it runs as finite displacement,
          and the submission goes ahead. The notice is left out only when the
          row's phonon job already has finite displacement in its written
          ``.param`` (see :meth:`_job_runs_finite_displacement`), so a job an
          older version wrote with another method is always announced.
        """

        if self.nothing_to_run(keys):
            return False

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

        legacy_methods = self._rows_with_legacy_phonon_method(keys)
        if legacy_methods:
            self._context.show_info(*_legacy_method_notice(legacy_methods))
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
                rows.append((self.source_label(candidate), job_status_word(job)))
        return rows

    def _rows_with_legacy_phonon_method(self, keys: list[object]) -> list[tuple[str, str]]:
        """Return ``(label, CASTEP method)`` for ready rows stored with another phonon
        method whose job is not already written as finite displacement."""

        definition = self.active_definition()
        if definition is None:
            return []
        states = self.build_states()
        rows: list[tuple[str, str]] = []
        for key in [str(value) for value in keys]:
            candidate = self.candidate_by_key(key)
            if (
                candidate is None
                or not workflow_row_state_is_ready(states, key)
                or self._job_runs_finite_displacement(self.phonon_job_for_key(definition, key))
            ):
                continue
            method = self.legacy_phonon_method(definition, key)
            if method is not None:
                rows.append((self.source_label(candidate), method))
        return rows

    def _job_runs_finite_displacement(self, job: Job | None) -> bool:
        """Return whether ``job``'s written ``.param`` already asks for finite displacement.

        ``False`` without a job, without a readable ``.param``, or when it
        names another method (a job an older version wrote).
        """

        if job is None:
            return False
        directory = self._context.job_output_directory(job)
        if directory is None:
            return False
        param_path = directory / f"{self._context.job_seed_name(job)}.param"
        try:
            assignments = parse_param_assignments(
                param_path.read_text(encoding="utf-8"), strict=False
            )
        except (OSError, UnicodeDecodeError, ParamParseError, ValueError):
            return False
        method = assignments.get("phonon_method")
        return method is not None and _legacy_method_value(method) == QUANTUM_PHONON_METHOD

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
            edited = job.calc_type == QUANTUM_PHONON_CALC_TYPE and self.completed_row_is_edited(
                definition, candidate.key, job
            )
            states[candidate.key] = RelaxationRunState(
                status=EDITED_ROW_STATUS if edited else effective_row_status(job),
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

    def describe_rows(
        self, keys: list[object], *, also: Collection[UUID] = ()
    ) -> tuple[CalculationDefinition | None, Path, list[tuple[str, JobSpec, Job | None]]] | None:
        """Return the active definition (``None`` before the first), the output folder and each row.

        Each row is ``(key, spec, phonon job)``; see
        :meth:`~mufinder.app.workflow_controllers.completed_row_edits.CompletedRowEditsController.describe_source_rows`.
        """

        return self.describe_source_rows(
            keys,
            also=also,
            job_for_row=self.phonon_job_for_key,
            job_config=self._phonon_job_config,
            expand=lambda configs: expand_quantum_phonon_jobs(
                jobs=configs, name_prefix="quantum_phonon"
            ),
        )

    def _planned_seed(
        self, definition: CalculationDefinition | None, key: str, output_root: Path
    ) -> str:
        """Return the seed a row's new phonon job is planned under, changing nothing.

        The row's seed, unless its directory already exists or another planned
        job took it: then a fresh seed that no directory, other row or planned
        job uses.
        """

        seed_name = self.seed_name_for_key(definition, key)
        reserved = self.reserved_names(output_root)
        if not (output_root / seed_name).exists() and seed_name not in reserved:
            return seed_name
        stored = definition.parameters.get(_SEED_MAPPING, {}) if definition is not None else {}
        taken = {
            str(value)
            for other, value in (stored if isinstance(stored, dict) else {}).items()
            if other != key
        }
        return _fresh_seed_name(seed_name, output_root=output_root, taken=taken | reserved)

    def plan_jobs(self, keys: list[object]) -> list[Job]:
        """Return each ready row's phonon job, planning one for a row that has none.

        Nothing changes: a new job is planned under its row's seed, or a fresh
        one when that folder exists, and is labelled with its spec's name.
        """

        try:
            described = self.describe_rows(keys)
        except (ValueError, StructureBridgeError):
            return []
        if described is None:
            return []
        definition, output_root, rows = described
        jobs: list[Job] = []
        for key, spec, existing in rows:
            if existing is not None:
                jobs.append(existing)
                continue
            jobs.append(
                self.plan_new_job(
                    spec,
                    output_root,
                    key,
                    calc_type=QUANTUM_PHONON_CALC_TYPE,
                    definition=definition,
                    label=spec.name,
                    seed=self._planned_seed(definition, key, output_root),
                )
            )
        return dedupe_jobs(jobs)

    def write_jobs(
        self,
        keys: list[object],
        *,
        going: Collection[UUID],
        run_again_ids: Collection[UUID] = (),
    ) -> list[Job]:
        """Create the planned phonon jobs and rewrite the defined ones, of the rows ``going`` names.

        See :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.write_jobs`.
        Once every input is written the definition gains the sources of the
        rows going ahead (:meth:`ensure_definition`, which registers one when
        there is none, its "Configured …" record held until the submission is
        accepted), a rewritten job takes its spec's name and its source run
        once it goes ahead (:meth:`when_going`), and the seed, source-run and
        legacy singlepoint maps follow for a new job, whose seed is its planned
        folder's name.
        """

        document = self._context.document()
        str_keys = [str(key) for key in keys]
        described = self.describe_rows(str_keys, also=run_again_ids)
        if document is None or described is None:
            if any(job_id in going for job_id in self._planned):
                raise ValueError("the rows' sources can no longer be read")
            return []
        active, output_root, rows = described
        writes: list[tuple[JobSpec, Path]] = []
        rewritten: list[tuple[str, JobSpec, Job]] = []
        adopted: list[tuple[PlannedJob, JobSpec]] = []
        jobs: list[Job] = []
        going_keys: list[str] = []
        for key, spec, existing in rows:
            if existing is not None:
                if existing.id not in going:
                    continue
                jobs.append(existing)
                going_keys.append(key)
                rewrite = existing.status == JobStatus.DEFINED or existing.id in run_again_ids
                seed_name = self.seed_name_for_key(active, key)
                if rewrite and self.job_may_write_folder(existing, output_root / seed_name):
                    writes.append((replace(spec, name=seed_name), output_root))
                    rewritten.append((key, spec, existing))
                continue
            planned = self.planned_for_key(key)
            if planned is None or planned.job.id not in going:
                continue
            writes.append((planned.spec, output_root))
            adopted.append((planned, spec))
            jobs.append(planned.job)
            going_keys.append(key)
        if not jobs:
            return []
        self.write_inputs(writes)
        definition = self.ensure_definition(going_keys, hold_record=True)
        if definition is None:
            raise ValueError("the rows' sources are no longer eligible")
        phonon_mapping = _definition_mapping(definition, PHONON_JOB_MAPPING)
        seed_mapping = _definition_mapping(definition, _SEED_MAPPING)
        source_runs = _definition_mapping(definition, PHONON_SOURCE_RUN_MAPPING)
        # Load-bearing: ``phonon_job_for_key`` falls back to reading
        # ``job_ids_by_label``, so keep the phonon mapping aliased there.
        definition.parameters["job_ids_by_label"] = phonon_mapping
        for key, spec, existing in rewritten:
            self.when_going(existing, partial(setattr, existing, "label", spec.name))
            self.when_going(
                existing,
                partial(source_runs.__setitem__, key, spec.provenance["source_relaxation_run_id"]),
            )
        for planned, spec in adopted:
            row_seed = self.seed_name_for_key(definition, planned.key)
            if (
                planned.folder.name != row_seed
                and self.legacy_singlepoint_job_for_key(definition, planned.key) is not None
            ):
                # The legacy singlepoint keeps the folder the row's seed named.
                legacy_seeds = _definition_mapping(definition, QUANTUM_SINGLEPOINT_SEED_MAPPING)
                legacy_seeds.setdefault(planned.key, row_seed)
            seed_mapping[planned.key] = planned.folder.name
            source_runs[planned.key] = spec.provenance["source_relaxation_run_id"]
            phonon_mapping[planned.key] = str(self.adopt_planned(planned).id)
        jobs = dedupe_jobs(jobs)
        if jobs:
            self.refresh_view()
        return jobs

    def materialize(self, keys: list[object]) -> list[Job]:
        """Create the rows' phonon jobs now and save.

        See :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.materialize`.
        """

        jobs = super().materialize(keys)
        self.save_materialized(
            jobs, failure_message="Quantum Corrections jobs were generated, but autosave failed."
        )
        return jobs

    def _phonon_job_config(
        self, definition: CalculationDefinition | None, key: str
    ) -> QuantumPhononJobConfig | None:
        """Return the phonon job config row ``key`` builds from its settings, if it builds."""

        candidate = self.candidate_by_key(key)
        if candidate is None:
            return None
        try:
            structure = structure_to_spec(candidate.structure)
        except StructureBridgeError:
            return None
        _params, _advanced, extras = workflow_row_configuration(definition, key)
        values = {
            name: value
            for name, value in self.runtime_configuration(definition, key).items()
            if name != "task"
        }
        try:
            overrides = build_workflow_settings_template(values)
        except ValueError:
            return None
        return QuantumPhononJobConfig(
            source=QuantumPhononSource(
                label=self.source_label(candidate),
                structure=structure,
                source_site_id=candidate.site.id,
                source_relaxation_run_id=candidate.run.id,
                source_relaxation_job_id=candidate.job.id,
                charge_state=charge_state_for_job(candidate.job),
                total_energy_eV=candidate.total_energy_eV,
            ),
            num_farms=num_farms_from_extras(extras),
            cell_overrides=overrides.cell,
            param_overrides=overrides.param,
        )

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
            phonon_path = None if output_dir is None else output_dir / f"{seed_name}.phonon"

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
        """Rebuild the table: one row per listed site, with its result's columns.

        Unlike the row-table widgets, which build rows from
        ``set_runtime_states``, the quantum widget takes pre-built
        :class:`~mufinder.app.quantum_corrections_widget.QuantumSourceRow` lists, so
        the row building lives here.
        """

        document = self._context.document()
        self._close_analysis_of_removed_results()
        if document is None:
            self._view.set_site_scope(show_every_site=False)
            self._view.set_sources([])
            return

        source_rows: list[QuantumSourceRow] = []
        listed_job_ids: set[UUID] = set()
        quantum_states = self.build_states()
        candidates = self.candidates()
        definition = self.active_definition()
        ranking = zpe_ranking(document, candidates)
        relative_energies = quantum_source_relative_energies(candidates)
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
            row = QuantumSourceRow(
                key=candidate.key,
                label=candidate.site.label or candidate.job.label,
                charge_state=charge_state_label_for_job(candidate.job),
                relative_energy=_format_three_decimals(
                    relative_energies.get(candidate.key), signed=True
                ),
                displacements=str(3 * len(candidate.structure.species)),
                source_job=candidate.job.label,
                num_farms_display=str(num_farms_from_extras(extras)),
                status=getattr(state, "status", "ready") or "ready",
                pinned_label=scope.pinned_label(candidate.key),
            )
            result = ranking.results.get(candidate.key)
            if result is not None:
                row = replace(
                    row,
                    corrected_energy=_format_three_decimals(
                        ranking.corrected_eV(candidate.key), signed=True
                    ),
                    zpe_mark=ranking.mark(candidate.key),
                    notes=(
                        *ranking.notes(candidate.key),
                        *((EDITED_ROW_NOTE,) if row.status == EDITED_ROW_STATUS else ()),
                    ),
                    **_result_cells(result),
                )
            source_rows.append(row)
            job = self.phonon_job_for_key(definition, candidate.key)
            if job is not None:
                listed_job_ids.add(job.id)
        self._view.set_sources(
            source_rows + self._rows_without_a_site(ranking.newest_results, listed_job_ids)
        )

    def _close_analysis_of_removed_results(self) -> None:
        """Close each open analysis window whose result is no longer in the project.

        Running an edited row again removes its result once the submission is
        accepted; its window would otherwise go on showing it as current.
        """

        document = self._context.document()
        for result_id, dialog in list(self._analysis_dialogs.items()):
            if document is not None and result_id in document.results:
                continue
            self._analysis_dialogs.pop(result_id, None)
            dialog.close()
            dialog.deleteLater()

    def _rows_without_a_site(
        self, newest: Mapping[UUID, Result], listed_job_ids: set[UUID]
    ) -> list[QuantumSourceRow]:
        """Return a read-only row for each result that no listed site's job stands behind.

        It has no energy to rank; it keeps its numbers and its analysis window. A job
        that is running again or was reset no longer stands behind its result, and the
        Job Dashboard shows it, so it gets no row.
        """

        document = self._context.document()
        if document is None:
            return []
        rows: list[tuple[tuple[object, ...], QuantumSourceRow]] = []
        for owner, result in newest.items():
            job = document.jobs.get(owner)
            if owner in listed_job_ids or (job is not None and job.status != JobStatus.COMPLETED):
                continue
            label = (job.label if job is not None else "") or "Quantum phonon"
            row = QuantumSourceRow(
                key="",
                label=f"{label} (no eligible site)",
                charge_state=NO_VALUE_TEXT,
                relative_energy=NO_VALUE_TEXT,
                displacements="",
                source_job="",
                num_farms_display=NO_VALUE_TEXT,
                status="completed",
                **_result_cells(result),
            )
            rows.append(((job is None, natural_sort_key(label), str(result.id)), row))
        return [row for _order, row in sorted(rows, key=lambda pair: pair[0])]


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
