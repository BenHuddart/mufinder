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
  shared submit backend (``task_farming_for_job=self._task_farming_for_job``,
  which filters by job) for *all* workflows. This controller's single-group
  :meth:`submission_groups` only adds a preselected cluster and does not touch
  task farming.
- **Stage-2 target guard.** :meth:`QuantumCorrectionsController.submit` refuses a
  stage-2 phonon row sent to a target that does not hold its stage-1 checkpoint
  (local after a cluster singlepoint, a cluster after a local one, or rows whose
  singlepoints ran on different clusters) and preselects the stage-1 cluster.

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
from typing import Literal

from mufinder.castep_io.cell_parser import CellParseError, parse_cell_file
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.science.quantum_phonons import (
    muon_mass_warning,
    muon_zpe,
    w1_entanglement_witness,
)
from mufinder.workflow import StructureBridgeError, structure_to_spec
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
    """Quantum-corrections-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating code still calls
    them: the definition/stage/config helpers are shared with the quantum
    configure path, source-label rename, and stage-one reset; ``check_definition_host``
    is shared with the NEB / energy-barriers workflow; the source-candidate and
    row-configuration helpers are shared across the quantum surface and the energy
    barriers path; and ``dedupe_jobs`` / ``format_energy`` are shared across
    workflows. ``input_template_from_keyword_values`` wraps the shared
    ``main_window`` module-level keyword→template helper. The controller reaches
    them only through these required callables. Workflow-specific dependencies like
    these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`;
    cross-workflow job/definition readers (``definition_output_directory``,
    ``job_output_directory``, ``job_seed_name``) live there instead.
    """

    ensure_quantum_corrections_definition: Callable[
        [list[str]], CalculationDefinition | None
    ]
    eligible_quantum_source_candidates: Callable[[], list[object]]
    quantum_source_relative_energies: Callable[[], Mapping[str, float | None]]
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
    format_energy: Callable[[object], str]
    #: Ask a yes/no question (title, text); ``True`` means proceed. The safe
    #: answer, "no", is the default.
    confirm: Callable[[str, str], bool]


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

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready)
    # exactly mirror the former ``_on_submit_*_from_quantum_corrections`` handlers.
    # submission_groups only preselects the stage-1 cluster (see ``submit``); task
    # farming is applied by the shared submit backend, not here.

    def normalize(self, value: object) -> str:
        return str(value)

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Validate the active definition against the host and the stage-1 inputs.

        Two guards run here:

        - the ``_check_definition_host`` guard the former quantum submit handlers
          ran before building a submission plan (shared with the NEB /
          energy-barriers workflow, so it stays on ``MainWindow`` and is reached
          via the deps). A refusal is silent.
        - :meth:`_stage_two_sources_missing_muon_mass`, which refuses a stage-2
          phonon continuation whose stage-1 singlepoint cell has no
          ``SPECIES_MASS`` block. That refusal explains itself via ``show_info``.
        """

        if not self._deps.check_definition_host(
            self.active_definition()
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
            candidate = self._deps.quantum_source_candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self._deps.quantum_stage_state(definition, candidate)
            if getattr(stage_state, "stage", None) != QUANTUM_STAGE_PHONON:
                continue
            seed_name = self._deps.quantum_seed_name_for_key(definition, key)
            cell_path = output_root / seed_name / f"{seed_name}.cell"
            if _cell_declares_species_mass(cell_path):
                continue
            missing.append(self._deps.quantum_source_label(candidate))
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
            candidate = self._deps.quantum_source_candidate_by_key(key)
            if candidate is None:
                continue
            stage_state = self._deps.quantum_stage_state(definition, candidate)
            current_job = getattr(stage_state, "current_job", None)
            if getattr(stage_state, "status", None) != "ready" and (
                current_job is None or current_job.status != JobStatus.DEFINED
            ):
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
            values, pseudopotential_family = self._deps.quantum_runtime_configuration(
                definition, key
            )
            gui_method = str(values.get("phonon_method", "FINITEDISPLACEMENT")).strip().upper()
            try:
                method = self._deps.quantum_method_from_gui_value(gui_method)
            except ValueError:
                continue
            compatible, message = phonon_method_compatibility(
                method,  # type: ignore[arg-type]
                pseudopotential_family=pseudopotential_family,  # type: ignore[arg-type]
            )
            entry = (
                self._deps.quantum_source_label(candidate),
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
            singlepoint = self._deps.singlepoint_job_for_quantum_source_key(
                definition, key
            )
            if singlepoint is None:
                continue
            rows.append(
                (
                    self._deps.quantum_source_label(candidate),
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
        output_root = self._context.definition_output_directory(definition)
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
                    total_energy_eV=candidate.site.total_energy_eV,
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
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
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
        relative_energies = self._deps.quantum_source_relative_energies()
        definition = self.active_definition()
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
                    relative_energy=_format_relative_energy(
                        relative_energies.get(candidate.key)
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
                    status=job_status,
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
