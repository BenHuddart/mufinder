"""Muon-relaxation workflow controller.

Owns the submit → materialize → result → refresh cycle for the CASTEP muon
structural relaxations (both charge states: positive muon and muonium). This
migration is the first to exercise two base hooks beyond their defaults:

- :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.pre_submit_check`
  validates that every staged site matches the active host structure (the
  former ``MainWindow._check_staged_muon_relaxation_host`` guard) before any
  submission plan is built. A refusal is silent — no dialogs beyond the
  mismatch-resolution dialog itself.
- :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.submission_groups`
  splits a mixed job list into per-charge-state groups (the former
  ``MainWindow._submit_muon_relaxation_jobs_grouped``), submitted positive-muon
  group first, then muonium, each through the same local/cluster backend with
  no extra kwargs.

The logic here is a behavior-preserving move of the former ``MainWindow``
methods ``_on_submit_local_from_muon_relaxation`` /
``_on_submit_cluster_from_muon_relaxation`` (submission),
``_submit_muon_relaxation_jobs_grouped`` (grouping),
``_materialize_muon_relaxation_jobs``, ``_jobs_for_muon_relaxation_labels``,
``_build_muon_relaxation_states``, ``_muon_relaxation_label_is_ready``,
``_check_staged_muon_relaxation_host``, and the muon-relaxation portion of
``_on_local_job_finished``'s result block.

:meth:`MuonRelaxationController.build_result` carries the richest result logic
so far: relaxed lattice payload, the muon's final fractional position in both
supercell and host coordinates, the derived relaxed
:class:`~mufinder.domain.models.Structure`, and — as a sanctioned document side
effect (see the :class:`~mufinder.app.workflow_controllers.base.ControllerResult`
docstring) — mutation of the source :class:`~mufinder.domain.models.MuonSite`
(``status``/``relaxation_run_id``/``total_energy_eV``/``relaxed_fractional_position``).
:meth:`MuonRelaxationController.validate_artifacts` fails the job before any
result is built: it applies the relaxation checks shared with the pristine
controller (``geom_converged``, a missing, unparseable or stale ``-out.cell``)
and then requires an ``H:mu`` position in the relaxed cell.

The controller also owns the page's form and rows outside submission: the
configuration the form holds (:meth:`MuonRelaxationController.form_config`),
binding active dashboard jobs to unmapped rows, staging generated sites as rows
(:meth:`MuonRelaxationController.stage_sites`), and the row-removal steps that
read or change its rows: which sites the removed rows held, dropping them from
the saved form, restoring the form from the definition and returning sites no
row lists any more to ``generated``. What it needs from the main window arrives
through :class:`MuonRelaxationDeps`, following the shared-context +
per-controller-deps + view pattern documented in
:mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from uuid import UUID

from PySide6.QtWidgets import QDialog, QMessageBox, QWidget

from mufinder.castep_io.input_diff import JobInputTexts
from mufinder.castep_io.output_parser import CastepResult
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
)
from mufinder.domain.text import plural
from mufinder.persistence.codec import uuid_list, uuid_or_none
from mufinder.services.definition_mappings import (
    LABEL_MAPPING,
    ensure_job_mapping,
    relaxation_mapping_job_priority,
    remove_labels_from_definition,
    sync_equivalent_mappings,
)
from mufinder.services.project_service import (
    CONTINUABLE_JOB_STATUSES,
    SiteHostMismatch,
    muon_relaxation_input_site_ids,
)
from mufinder.services.site_generation_service import unstage_sites
from mufinder.services.site_relaxations import record_site_relaxation
from mufinder.services.source_candidates import coerce_fractional_triplet
from mufinder.science.lattice import lattice_frac_to_cart, wrap_fractional
from mufinder.science.relaxation_analysis import muon_fractional_position_from_cell
from mufinder.services.relaxation_import import seed_is_usable
from mufinder.workflow import StructureBridgeError, render_job
from mufinder.workflow.types import JobSpec
from mufinder.workflow.muon_site import (
    ACTIVE_MUON_RELAXATION_DEFINITION_KEY,
    MUON_RELAXATION_DEFINITION_CALC_TYPE,
)

from mufinder.app.muon_generation_support import MuonChargeStateDialog
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.relaxation_adapter import (
    MUON_RELAXATION_CALC_TYPES,
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    expand_muon_relaxation_jobs,
    is_muon_relaxation_calc_type,
    muon_charge_state_label,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
)
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    RelaxationRunState,
    SupercellMatrix,
)
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    PlannedJob,
    WorkflowControllerContext,
    WorkflowJobController,
    dedupe_jobs,
    apply_row_parameters_to_form_state,
    relaxation_artifact_failure,
    upsert_job_derived_structure,
)
from mufinder.app.workflow_config_codecs import (
    deserialize_muon_relaxation_config,
    muon_relaxation_row_key,
    muon_relaxation_row_signature,
    serialize_muon_relaxation_config,
    stored_row_keys,
)
from mufinder.app.workflow_submission import effective_row_status, workflow_row_state_is_ready


#: The definition mapping that holds each row's job seed name.
SEED_NAME_MAPPING = "job_seed_names_by_label"


def _without_imported_rows(config: MuonRelaxationConfig) -> MuonRelaxationConfig:
    """Return ``config`` with imported rows dropped.

    Job expansion must never see them. An imported row carries the parameters
    its run actually used, recovered from third-party output rather than
    authored against the current schema, so expansion can legitimately reject a
    value — and both callers turn that into a silent empty result. Since
    expansion is all-or-nothing over the whole configuration, one such row would
    take every *native* row's submission down with it.

    Dropping them costs nothing: an imported row already owns a COMPLETED job
    and is barred from submission, so it has no job to materialise and no
    ``.param`` worth previewing.
    """

    if not any(row.imported for row in config.rows):
        return config
    return replace(config, rows=tuple(row for row in config.rows if not row.imported))


def _muon_relaxation_selection_matches(row: MuonRelaxationRow, selection: str) -> bool:
    """Return whether a raw selection value targets ``row``.

    Selections are either charge-qualified (``"{prefix}:{label}"``) or bare
    labels; a bare label matches the row regardless of charge state.
    """

    selection = str(selection)
    prefix, separator, label = selection.partition(":")
    if separator:
        return prefix == muon_charge_state_prefix(row.charge_state) and label == row.label
    return selection == row.label


def _fold_supercell_fractional_to_host(
    position: tuple[float, float, float],
    matrix: SupercellMatrix | None,
) -> tuple[float, float, float]:
    """Fold a supercell fractional position back into host-cell coordinates."""

    # A supercell fractional row vector times the supercell matrix is the host position.
    host_position = position if matrix is None else lattice_frac_to_cart(position, matrix)
    return tuple(wrap_fractional(host_position))  # type: ignore[return-value]


def _names_a_folder_here(seed: object) -> bool:
    """Return whether a saved seed names a folder inside the charge folder it is read from.

    A saved seed can hold anything, including ``..`` or an absolute path, and inputs are never
    written where such a name leads.
    """

    return isinstance(seed, str) and seed_is_usable(seed) and Path(seed).name == seed


def _job_submission_sort_key(job: Job) -> float:
    """The time a job was first sent off: submitted, else started, else completed."""

    for stamp in (job.submitted_at, job.started_at, job.completed_at):
        if stamp is not None:
            return stamp.timestamp()
    return 0.0


@dataclass(frozen=True)
class MuonRelaxationDeps:
    """What the muon-relaxation controller needs from the main window.

    ``handle_site_host_mismatch`` asks whether to switch the active host when
    sites were generated for another one; the other site-based workflows share
    it. ``dialog_parent`` parents the charge-state dialog and the staging
    messages. ``show_staged_sites`` refreshes the pages that list sites and
    rows after sites are staged, then shows the Relaxations page.
    Workflow-specific dependencies like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`;
    cross-workflow job/definition readers (``job_output_directory``,
    ``job_seed_name``, ``run_for_job``, ``definition_output_directory``, the
    energy/force readers) live there instead.
    """

    handle_site_host_mismatch: Callable[[SiteHostMismatch], bool]
    dialog_parent: Callable[[], QWidget]
    show_staged_sites: Callable[[], None]


class MuonRelaxationController(WorkflowJobController):
    """Controller for the muon structural-relaxation workflow."""

    calc_types = frozenset(MUON_RELAXATION_CALC_TYPES.values())
    stage = WorkflowStage.MUON_RELAXATION
    definition_calc_type = MUON_RELAXATION_DEFINITION_CALC_TYPE
    active_key = ACTIVE_MUON_RELAXATION_DEFINITION_KEY
    no_ready_message = "Only configured muon-relaxation rows that are ready, failed, stopped or cancelled can be submitted."

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: MuonRelaxationDeps,
        view: MuonRelaxationWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    def job_for_label(
        self,
        definition: CalculationDefinition | None,
        label: str,
        charge_state: str | None = None,
    ) -> Job | None:
        """Return the muon-relaxation job ``definition`` binds to one row.

        With a ``charge_state`` the charge-qualified ``"{prefix}:{label}"`` key
        is tried first and must hold a job of that charge state's calc type.
        The bare ``label`` key, kept by definitions written before rows were
        charge-qualified, accepts a job of any muon-relaxation calc type.
        """

        if charge_state:
            job = self.job_for_key(
                definition,
                f"{muon_charge_state_prefix(charge_state)}:{label}",
                calc_type=muon_relaxation_calc_type(charge_state),
            )
            if job is not None:
                return job
        job = self.job_for_key(definition, label)
        if job is None or not is_muon_relaxation_calc_type(job.calc_type):
            return None
        return job

    def job_for_mapping_value(self, key: str, raw_job_id: str) -> Job | None:
        """Return the muon-relaxation job ``raw_job_id`` names, when it fits ``key``.

        A charge-qualified ``key`` (``"{prefix}:{label}"``) needs a job of that
        charge state's calc type and an unknown prefix fits none; a bare label
        fits a job of any muon-relaxation calc type.
        """

        document = self._context.document()
        if document is None:
            return None
        try:
            job_id = UUID(str(raw_job_id))
        except ValueError:
            return None
        job = document.jobs.get(job_id)
        if job is None or not is_muon_relaxation_calc_type(job.calc_type):
            return None
        prefix, separator, _label = str(key).partition(":")
        if not separator:
            return job
        if prefix == muon_charge_state_prefix(POSITIVE_MUON_CHARGE_STATE):
            expected = muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE)
        elif prefix == muon_charge_state_prefix(MUONIUM_CHARGE_STATE):
            expected = muon_relaxation_calc_type(MUONIUM_CHARGE_STATE)
        else:
            return None
        return job if job.calc_type == expected else None

    # --- definition maintenance ---

    def sync_mappings(self, definition: CalculationDefinition) -> None:
        """Copy job bindings and seed names into ``definition`` from equivalent definitions.

        Rows are keyed ``"{prefix}:{label}"``; a source's bare-label key, from
        before rows were charge-qualified, is read as its row's qualified key.
        A row carries over when its signature is unchanged and its job exists and
        fits the key's charge state (:meth:`job_for_mapping_value`), whatever its
        status: the page's Run configures a new definition before it submits when
        its configuration changed, so
        a ``defined`` row keeps its job and folder rather than gaining another.
        Candidates rank by
        :func:`~mufinder.services.definition_mappings.relaxation_mapping_job_priority`,
        and a winner's ``job_seed_names_by_label`` entry travels with it.
        """

        document = self._context.document()
        if document is None:
            return

        def accept_job(key: str, raw_job_id: str) -> bool:
            return self.job_for_mapping_value(key, raw_job_id) is not None

        sync_equivalent_mappings(
            document,
            self.definitions(),
            definition,
            deserialize=deserialize_muon_relaxation_config,
            signatures=lambda config: {
                muon_relaxation_row_key(row): muon_relaxation_row_signature(row)
                for row in config.rows
            },
            priority=relaxation_mapping_job_priority,
            accept_job=accept_job,
            legacy_keys=lambda config: {
                row.label: muon_relaxation_row_key(row)
                for row in config.rows
                if ":" not in row.label
            },
            seed_mapping_name=SEED_NAME_MAPPING,
        )

    def row_key_matcher(self, labels: Collection[str]) -> Callable[[object], bool]:
        """Match a mapping key to removed rows across both key spellings.

        A key matches when it is one of ``labels``, when it is charge-qualified
        and its label is a bare entry of ``labels``, or when it is a bare label
        and ``labels`` holds a charge-qualified key for it.
        """

        unqualified_labels = {label for label in labels if ":" not in label}
        full_key_labels = {label.partition(":")[2] for label in labels if ":" in label}

        def matches(key: object) -> bool:
            if not isinstance(key, str):
                return False
            return (
                key in labels
                or (":" in key and key.partition(":")[2] in unqualified_labels)
                or (":" not in key and key in full_key_labels)
            )

        return matches

    def remove_labels_from_definition(
        self, definition: CalculationDefinition, labels: Collection[str]
    ) -> bool:
        """Drop the rows ``labels`` name, and their job and seed-name bindings.

        A row goes when its qualified key is in ``labels`` or its label is a
        bare entry of ``labels``; mapping keys go by :meth:`row_key_matcher`.
        """

        document = self._context.document()
        if document is None:
            return False
        unqualified_labels = {label for label in labels if ":" not in label}
        return remove_labels_from_definition(
            document,
            definition,
            labels,
            deserialize=deserialize_muon_relaxation_config,
            serialize=serialize_muon_relaxation_config,
            active_key=self.active_key,
            row_matches=lambda row: (
                muon_relaxation_row_key(row) in labels or row.label in unqualified_labels
            ),
            key_matches=self.row_key_matcher(labels),
            mapping_names=(LABEL_MAPPING, SEED_NAME_MAPPING),
        )

    def configured_site_ids(self, configuration: Mapping[str, object]) -> list[UUID]:
        """Return each row's site, as the definition registered from ``configuration`` records."""

        return muon_relaxation_input_site_ids(configuration)

    def sync_active_dashboard_jobs(self, definition: CalculationDefinition) -> None:
        """Bind each row of ``definition`` with no usable job to its active job, if any.

        A row whose mapped job still fits its key keeps it. Otherwise an
        ``active`` muon-relaxation job with the row's label and charge state's
        calc type is bound, the highest by
        :func:`~mufinder.services.definition_mappings.relaxation_mapping_job_priority`
        and then the latest sent off, so a job running from the dashboard stays
        on its row.
        """

        document = self._context.document()
        if document is None:
            return
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return

        mapping = ensure_job_mapping(definition)
        for row in config.rows:
            key = muon_relaxation_row_key(row)
            existing = mapping.get(key)
            if isinstance(existing, str) and self.job_for_mapping_value(key, existing) is not None:
                continue

            expected_calc_type = muon_relaxation_calc_type(row.charge_state)
            candidates = [
                job
                for job in document.jobs.values()
                if job.workflow_stage == WorkflowStage.MUON_RELAXATION
                and job.calc_type == expected_calc_type
                and job.label == row.label
                and job.status == JobStatus.ACTIVE
            ]
            if not candidates:
                continue

            job = max(
                candidates,
                key=lambda candidate: (
                    relaxation_mapping_job_priority(document, str(candidate.id)),
                    _job_submission_sort_key(candidate),
                ),
            )
            mapping[key] = str(job.id)

    def definition_and_row_for_job(
        self, job: Job
    ) -> tuple[CalculationDefinition | None, MuonRelaxationRow | None]:
        """Return the definition and row ``job`` is bound to, or ``(None, None)``."""

        if self._context.document() is None:
            return None, None
        for definition in self.definitions():
            config = deserialize_muon_relaxation_config(
                definition.parameters.get("configuration", {})
            )
            if config is None:
                continue
            for row in config.rows:
                candidate = self.job_for_label(definition, row.label, row.charge_state)
                if candidate is not None and candidate.id == job.id:
                    return definition, row
        return None, None

    # --- the page's form and rows ---

    def form_config(self) -> MuonRelaxationConfig | None:
        """Return the configuration the page's form holds.

        The live form is read first, then the form saved in ``ui_state``; the
        first that holds a row wins. ``None`` with no project or no row.
        """

        document = self._context.document()
        if document is None:
            return None

        form_sources: list[dict] = []
        widget_state = self._view.collect_state()
        if isinstance(widget_state, dict):
            form_sources.append(widget_state)

        persisted = document.ui_state.get("muon_relaxation_form", {})
        if isinstance(persisted, dict):
            form_sources.append(persisted)

        for form_state in form_sources:
            config = deserialize_muon_relaxation_config(form_state)
            if config is not None and config.rows:
                return config
        return None

    def stage_sites(self, raw_site_ids: object) -> None:
        """Stage generated sites as relaxation rows in the charge states the user picks.

        The sites must match the active host; a mismatch asks to switch hosts
        first. The charge-state dialog then picks the charge states, the sites
        are staged through the project service and added to the page, the
        project is saved and the Relaxations page is shown.
        """

        parent = self._deps.dialog_parent()
        document = self._context.document()
        if document is None:
            QMessageBox.information(parent, "No project", "Create or open a project first.")
            return
        site_ids = uuid_list(raw_site_ids)
        if not site_ids:
            return
        project_service = self._context.project_service
        mismatch = project_service.validate_sites_match_active_host(document, site_ids)
        if mismatch is not None:
            if not self._deps.handle_site_host_mismatch(mismatch):
                return
            mismatch = project_service.validate_sites_match_active_host(document, site_ids)
            if mismatch is not None:
                return
        dialog = MuonChargeStateDialog(site_count=len(site_ids), parent=parent)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        charge_states = dialog.selected_charge_states()
        staged = project_service.stage_generated_sites(document, site_ids)
        if not staged:
            QMessageBox.information(
                parent,
                "No generated sites",
                "Select one or more generated muon sites that have not already been staged.",
            )
            return
        rows = [
            {
                "id": str(site.id),
                "label": site.label,
                "fractional_position": list(site.initial_fractional_position),
                "method": site.generation_method or "",
            }
            for site in staged
        ]
        self._view.add_staged_sites(rows, charge_states=charge_states)
        self._context.persist_ui_state(
            failure_message="Muon sites were staged, but autosave failed."
        )
        self._deps.show_staged_sites()
        charge_label = " and ".join(
            muon_charge_state_label(charge_state) for charge_state in charge_states
        )
        self._context.status_message(
            f"Staged {plural(len(staged), 'muon relaxation position')} for {charge_label}"
        )

    def site_ids_for_removed_rows(
        self, definitions: Sequence[CalculationDefinition], labels: Collection[str]
    ) -> set[UUID]:
        """Return the sites the rows ``labels`` name hold, in the saved form and ``definitions``."""

        document = self._context.document()
        if document is None:
            return set()

        key_matches = self.row_key_matcher(labels)
        site_ids: set[UUID] = set()

        def collect_from_rows(rows: object) -> None:
            if not isinstance(rows, list):
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if not any(key_matches(key) for key in stored_row_keys(row)):
                    continue
                site_id = uuid_or_none(row.get("site_id"))
                if site_id is not None:
                    site_ids.add(site_id)

        state = document.ui_state.get("muon_relaxation_form", {})
        if isinstance(state, dict):
            collect_from_rows(state.get("rows", []))

        for definition in definitions:
            config = definition.parameters.get("configuration", {})
            if isinstance(config, dict):
                collect_from_rows(config.get("rows", []))

        return site_ids

    def remove_labels_from_form_state(self, labels: Collection[str]) -> None:
        """Drop the rows ``labels`` name from the form saved in ``ui_state``."""

        document = self._context.document()
        if document is None:
            return
        state = document.ui_state.get("muon_relaxation_form", {})
        if not isinstance(state, dict):
            return
        rows = state.get("rows", [])
        if not isinstance(rows, list):
            return
        key_matches = self.row_key_matcher(labels)
        kept: list[object] = []
        for row in rows:
            if not isinstance(row, dict):
                kept.append(row)
                continue
            if any(key_matches(key) for key in stored_row_keys(row)):
                continue
            kept.append(row)
        updated = dict(state)
        updated["rows"] = kept
        document.ui_state["muon_relaxation_form"] = updated

    def restore_form_from_definition(self) -> bool:
        """Put the active definition's rows back in the form and the saved form.

        Returns whether it did: ``False`` with no active definition or one whose
        configuration does not read.
        """

        definition = self.active_definition()
        if definition is None:
            return False
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return False
        state = serialize_muon_relaxation_config(config)
        for row in state.get("rows", []):
            if isinstance(row, dict) and "fractional_position" not in row:
                row["fractional_position"] = list(
                    row.get("initial_fractional_position", (0.0, 0.0, 0.0))
                )
        self._view.set_form_state(state)
        document = self._context.document()
        if document is not None:
            document.ui_state["muon_relaxation_form"] = state
        return True

    def listed_site_ids(self) -> set[UUID]:
        """Return every site a row lists, in the saved form or any muon-relaxation definition."""

        document = self._context.document()
        if document is None:
            return set()
        site_ids: set[UUID] = set()

        def collect(rows: object) -> None:
            if not isinstance(rows, list):
                return
            for row in rows:
                if not isinstance(row, dict):
                    continue
                site_id = uuid_or_none(row.get("site_id"))
                if site_id is not None:
                    site_ids.add(site_id)

        state = document.ui_state.get("muon_relaxation_form", {})
        if isinstance(state, dict):
            collect(state.get("rows", []))
        for definition in self.definitions():
            config = definition.parameters.get("configuration", {})
            if isinstance(config, dict):
                collect(config.get("rows", []))
        return site_ids

    def unstage_unlisted_sites(self, site_ids: Collection[UUID]) -> bool:
        """Return the staged ``site_ids`` no row lists any more to ``generated``.

        Returns whether any site changed; see
        :func:`~mufinder.services.site_generation_service.unstage_sites`.
        """

        document = self._context.document()
        if document is None:
            return False
        return unstage_sites(document, site_ids, still_listed=self.listed_site_ids())

    # --- submission contract ---

    def normalize(self, value: object) -> str:
        return str(value)

    def is_ready(self, states: Mapping[object, object], key: object) -> bool:
        """A row is ready when its own key — or, for a bare label, any
        charge-qualified key ending in ``":{label}"`` — is ready by the shared
        :func:`~mufinder.app.workflow_submission.workflow_row_state_is_ready`.
        """

        if workflow_row_state_is_ready(states, key):
            return True
        if ":" in str(key):
            return False
        suffix = f":{key}"
        return any(
            workflow_row_state_is_ready(states, candidate_key)
            for candidate_key in states
            if str(candidate_key).endswith(suffix)
        )

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Validate all staged muon-relaxation sites match the active host.

        Returns True if the submission should proceed (no mismatch, or the
        user resolved the mismatch by switching hosts). Returns False if the
        user cancelled or the mismatch could not be resolved. Behavior-preserving
        move of ``MainWindow._check_staged_muon_relaxation_host``.
        """

        document = self._context.document()
        if document is None:
            return True
        definition = self.active_definition()
        if definition is None or not definition.input_site_ids:
            return True
        mismatch = self._context.project_service.validate_sites_match_active_host(
            document, definition.input_site_ids
        )
        return mismatch is None or self._deps.handle_site_host_mismatch(mismatch)

    def submission_groups(self, jobs: list[Job]) -> list[tuple[list[Job], dict[str, object]]]:
        """Split jobs into per-charge-state groups, positive muon first.

        Behavior-preserving move of
        ``MainWindow._submit_muon_relaxation_jobs_grouped``: jobs are grouped by
        charge-state calc_type, and each non-empty group is submitted separately
        (the base template loop calls the backend once per group, with no extra
        kwargs — exactly the former per-group ``_submit_jobs_locally`` /
        ``_submit_jobs_on_cluster`` calls). Jobs with a non-muon calc_type are
        dropped, as before.
        """

        # The submission coordinator's plan already dedupes ``jobs`` by id
        # (``submission_plan`` returns ``tuple(self._dedupe_jobs(jobs))``), so no
        # id-dedupe is repeated here.
        grouped: dict[str, list[Job]] = {}
        for job in jobs:
            grouped.setdefault(job.calc_type, []).append(job)
        return [
            (grouped[calc_type], {})
            for calc_type in (
                MUON_RELAXATION_CALC_TYPES["positive_muon"],
                MUON_RELAXATION_CALC_TYPES["muonium"],
            )
            if grouped.get(calc_type)
        ]

    def build_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        document = self._context.document()
        if document is None:
            return states
        form_config = self.form_config()
        if form_config is not None:
            for row in form_config.rows:
                states[muon_relaxation_row_key(row)] = RelaxationRunState(status="ready")

        definition = self.active_definition()
        if definition is None:
            return states
        self.sync_mappings(definition)
        self.sync_active_dashboard_jobs(definition)
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return states

        if form_config is not None:
            definition_rows_by_key = {muon_relaxation_row_key(row): row for row in config.rows}
            rows_for_status = []
            for row in form_config.rows:
                key = muon_relaxation_row_key(row)
                definition_row = definition_rows_by_key.get(key)
                if definition_row is None:
                    continue
                if muon_relaxation_row_signature(row) == muon_relaxation_row_signature(
                    definition_row
                ):
                    rows_for_status.append(definition_row)
                    continue
                job = self.job_for_label(
                    definition,
                    definition_row.label,
                    definition_row.charge_state,
                )
                if job is not None and job.status == JobStatus.ACTIVE:
                    rows_for_status.append(definition_row)
        else:
            rows_for_status = list(config.rows)

        for row in rows_for_status:
            state_key = muon_relaxation_row_key(row)
            job = self.job_for_label(definition, row.label, row.charge_state)
            if job is None:
                states[state_key] = RelaxationRunState(status="ready")
                continue
            status_value = effective_row_status(job)
            run = self._context.run_for_job(job.id)
            if (
                status_value == JobStatus.COMPLETED.value
                and run is not None
                and run.convergence_achieved is False
            ):
                status_value = "not converged"
            lattice_a = lattice_b = lattice_c = None
            final_fractional_position = None
            if run is not None and run.result_id is not None:
                result = document.results.get(run.result_id)
                if result is not None and isinstance(result.data, dict):
                    lattice_a = result.data.get("lattice_a")
                    lattice_b = result.data.get("lattice_b")
                    lattice_c = result.data.get("lattice_c")
                    final_fractional_position = coerce_fractional_triplet(
                        result.data.get("muon_final_fractional_position_host")
                    )
            states[state_key] = RelaxationRunState(
                status=status_value,
                total_energy=self._context.total_energy_for_job(job.id),
                max_force=self._context.max_force_for_job(job.id),
                lattice_a=float(lattice_a) if isinstance(lattice_a, (int, float)) else None,
                lattice_b=float(lattice_b) if isinstance(lattice_b, (int, float)) else None,
                lattice_c=float(lattice_c) if isinstance(lattice_c, (int, float)) else None,
                final_fractional_position=final_fractional_position,
                job_id=job.id,
                job_label=job.label,
                geom_converged=(run.convergence_achieved if run is not None else None),
                error_message=job.error_message,
            )
        return states

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        definition = self.active_definition()
        if definition is None:
            return []
        self.sync_mappings(definition)
        self.sync_active_dashboard_jobs(definition)
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        wanted = {str(key) for key in keys}
        for row in config.rows:
            if not any(_muon_relaxation_selection_matches(row, selection) for selection in wanted):
                continue
            job = self.job_for_label(definition, row.label, row.charge_state)
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def describe_rows(
        self, keys: list[object], *, also: Collection[UUID] = ()
    ) -> (
        tuple[
            CalculationDefinition,
            Path,
            list[tuple[str, MuonRelaxationRow, JobSpec | None, Job | None]],
        ]
        | None
    ):
        """Return the active definition, its output folder and each selected row's spec and job.

        Imported rows are left out, and every selected row counts (``also``
        adds nothing). A job running from the dashboard is bound to its row
        first, as every read of the rows does (``jobs_for_keys``), so no row
        plans a second one. See
        :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.active_expansion`.
        """

        document = self._context.document()
        definition = self.active_definition() if document is not None else None
        if document is None or definition is None:
            return None
        self.sync_mappings(definition)
        self.sync_active_dashboard_jobs(definition)

        def deserialize(payload: object) -> MuonRelaxationConfig | None:
            config = deserialize_muon_relaxation_config(payload)
            return _without_imported_rows(config) if config is not None else None

        def expand(definition: CalculationDefinition, config: object):
            host = document.structures.get(definition.input_structure_ids[0])
            if host is None:
                return None
            return expand_muon_relaxation_jobs(
                host_structure=host,
                config=config,
                project_cutoff_energy=document.project.recommended_cutoff_energy,
                project_kpoint_grid=document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=document.project.recommended_kpoint_lattice,
                project_xc_functional=document.project.recommended_xc_functional,
            )

        prepared = self.active_expansion(deserialize, expand)
        if prepared is None:
            return None
        definition, output_root, config, expansion = prepared
        spec_by_key = {
            f"{muon_charge_state_prefix(str(spec.provenance.get('charge_state', '')))}:{spec.provenance.get('label', '')}": spec
            for spec in expansion.jobs
        }
        wanted = {str(key) for key in keys}
        rows = [
            (
                muon_relaxation_row_key(row),
                row,
                spec_by_key.get(muon_relaxation_row_key(row)),
                self.job_for_label(definition, row.label, row.charge_state),
            )
            for row in config.rows
            if any(_muon_relaxation_selection_matches(row, selection) for selection in wanted)
        ]
        return definition, output_root, rows

    def plan_jobs(self, keys: list[object]) -> list[Job]:
        """Return each selected row's job, planning one for a row that has none; nothing changes.

        A new row's job is labelled with the row's label and planned in its
        charge-state folder, under the spec's name or a free one.
        """

        try:
            described = self.describe_rows(keys)
        except (ValueError, StructureBridgeError):
            return []
        if described is None:
            return []
        definition, output_root, rows = described
        jobs: list[Job] = []
        for key, row, spec, existing in rows:
            if existing is not None:
                jobs.append(existing)
            elif spec is not None:
                jobs.append(
                    self.plan_new_job(
                        spec,
                        output_root / muon_charge_state_prefix(row.charge_state),
                        key,
                        calc_type=muon_relaxation_calc_type(row.charge_state),
                        definition=definition,
                        label=row.label,
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
        """Create the planned jobs and rewrite the defined ones, of the rows ``going`` names.

        See :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.write_jobs`.
        A row's existing job is rewritten in its saved seed folder (or the
        spec's name) unless another job uses it, and the seed map follows once
        the job goes ahead (:meth:`when_going`); a new job's seed is its planned
        folder's name.
        """

        document = self._context.document()
        described = self.describe_rows(keys)
        if document is None or described is None:
            if any(job_id in going for job_id in self._planned):
                raise ValueError("the rows' configuration can no longer be read")
            return []
        definition, output_root, rows = described
        stored_seeds = definition.parameters.get("job_seed_names_by_label", {})
        if not isinstance(stored_seeds, dict):
            stored_seeds = {}
        writes: list[tuple[JobSpec, Path]] = []
        seeds: list[tuple[Job, str, str]] = []
        adopted: list[PlannedJob] = []
        jobs: list[Job] = []
        for key, row, spec, existing in rows:
            charge_root = output_root / muon_charge_state_prefix(row.charge_state)
            if existing is not None:
                if existing.id not in going:
                    continue
                jobs.append(existing)
                rewrite = existing.status == JobStatus.DEFINED or existing.id in run_again_ids
                if not rewrite or spec is None:
                    continue
                saved_seed = stored_seeds.get(key, stored_seeds.get(row.label))
                if isinstance(saved_seed, str):
                    saved_seed = saved_seed.strip()
                name = saved_seed if _names_a_folder_here(saved_seed) else spec.name
                if self.job_may_write_folder(existing, charge_root / name):
                    writes.append((replace(spec, name=name), charge_root))
                    seeds.append((existing, key, name))
                continue
            planned = self.planned_for_key(key)
            if planned is None or planned.job.id not in going:
                continue
            writes.append((planned.spec, planned.folder.parent))
            adopted.append(planned)
            jobs.append(planned.job)
        self.write_inputs(writes)
        mapping = ensure_job_mapping(definition)
        seed_mapping = ensure_job_mapping(definition, "job_seed_names_by_label")
        for existing, key, name in seeds:
            self.when_going(existing, partial(seed_mapping.__setitem__, key, name))
        for planned in adopted:
            mapping[planned.key] = str(self.adopt_planned(planned).id)
            seed_mapping[planned.key] = planned.folder.name
        return dedupe_jobs(jobs)

    def materialize(self, keys: list[object]) -> list[Job]:
        """Create the rows' jobs now and save, as continuing a relaxation needs.

        See :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.materialize`.
        """

        jobs = super().materialize(keys)
        self.save_materialized(
            jobs, failure_message="Muon relaxation jobs were generated, but autosave failed."
        )
        return jobs

    # --- in-place row parameter mutation ---

    def update_row_parameters(
        self,
        key: object,
        parameter_values: Mapping[str, object],
        added_advanced: Sequence[str] | None = None,
    ) -> bool:
        """Rewrite one row's CASTEP parameters inside the ACTIVE definition.

        ``key`` must be the charge-qualified row key (``"{prefix}:{label}"``) —
        one row is one job, and two rows can share a label, so a bare label is
        ambiguous and is refused. Returns ``False`` when there is no active
        definition or it holds no such row.

        The edit must never go through ``register_muon_relaxation_definition``:
        a new definition arrives with an empty ``job_ids_by_label``, which severs
        the row→job binding, so ``materialize`` would mint a duplicate ``Job``
        onto the row's existing directory. This mutates the very definition
        ``materialize`` reads, leaving ``job_ids_by_label``,
        ``job_seed_names_by_label`` and every other row untouched, and rewrites
        the row through the serialize/deserialize pair so the stored shape stays
        canonical.

        The parameters actually stored are the ones the *widget* reports after the
        mirror, not the caller's raw mapping: the tables strip implicit project
        defaults, and :meth:`build_states` matches a row's form signature against
        its definition signature.
        """

        definition = self.active_definition()
        if definition is None:
            return False
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return False
        row_key = str(key)
        if not any(muon_relaxation_row_key(row) == row_key for row in config.rows):
            return False

        values = dict(parameter_values)
        advanced = (
            tuple(str(name) for name in added_advanced) if added_advanced is not None else None
        )
        mirrored = self._mirror_row_parameters_into_form(row_key, values, advanced)
        if mirrored is not None:
            values, advanced = mirrored

        rows = tuple(
            replace(
                row,
                parameter_values=dict(values),
                added_advanced=(advanced if advanced is not None else row.added_advanced),
            )
            if muon_relaxation_row_key(row) == row_key
            else row
            for row in config.rows
        )
        definition.parameters["configuration"] = serialize_muon_relaxation_config(
            replace(config, rows=rows)
        )

        self._context.persist_ui_state(
            failure_message="Muon relaxation parameters were updated, but autosave failed."
        )
        self.refresh_view()
        return True

    def _mirror_row_parameters_into_form(
        self,
        row_key: str,
        parameter_values: Mapping[str, object],
        added_advanced: tuple[str, ...] | None,
    ) -> tuple[dict[str, object], tuple[str, ...] | None] | None:
        """Apply the edit to the widget form state and read the result back.

        Returns the widget's normalized ``(parameter_values, added_advanced)`` for
        the row, or ``None`` when the form does not hold it (in which case the
        caller stores its own values — the row is not projected from the form
        anyway).
        """

        def matches(row: Mapping[str, object]) -> bool:
            label = str(row.get("label", "")).strip()
            prefix = muon_charge_state_prefix(str(row.get("charge_state", "")))
            return f"{prefix}:{label}" == row_key

        updated = apply_row_parameters_to_form_state(
            self._view.collect_state(),
            row_matches=matches,
            parameter_values=parameter_values,
            added_advanced=added_advanced,
            parameter_key="parameter_values",
        )
        if updated is None:
            return None

        self._view.set_form_state(updated)
        normalized = self._view.collect_state()
        document = self._context.document()
        if document is not None and isinstance(normalized, dict):
            document.ui_state["muon_relaxation_form"] = normalized
        rows = normalized.get("rows", []) if isinstance(normalized, dict) else []
        for row in rows:
            if not isinstance(row, dict) or not matches(row):
                continue
            raw_params = row.get("parameter_values", {})
            raw_advanced = row.get("added_advanced", [])
            return (
                dict(raw_params) if isinstance(raw_params, dict) else {},
                tuple(str(name) for name in raw_advanced) if isinstance(raw_advanced, list) else (),
            )
        return None

    # --- continuation ("Continue relaxation…") ---

    def row_is_continuable(self, key: object) -> bool:
        """Return whether ``key``'s row has a job in :data:`CONTINUABLE_JOB_STATUSES`.

        ``key`` must be the charge-qualified row key (``"{prefix}:{label}"`` —
        see :meth:`jobs_for_keys`/:func:`_muon_relaxation_selection_matches`);
        one label can be shared by two rows (one per charge state), so a bare
        label would be ambiguous here. Resolves the row's job through
        :meth:`jobs_for_keys` — never from a table's *displayed* status text,
        which reads "not converged" for a completed-but-unconverged job. Called
        live at context-menu-open time (not cached), so a status change between
        refreshes (e.g. a run started from the dashboard) is always reflected.
        """

        jobs = self.jobs_for_keys([key])
        if not jobs:
            return False
        # An imported job is COMPLETED, so it would otherwise qualify. Continuing
        # one rebuilds the cell from the *initial* muon position and relies on a
        # checkpoint file an import does not have, so it would silently restart
        # from scratch while presenting itself as a warm start -- and discard the
        # relaxed result the import exists to preserve.
        if IMPORTED_JOB_TAG in (jobs[0].tags or ()):
            return False
        return jobs[0].status in CONTINUABLE_JOB_STATUSES

    def row_settings(self, key: object) -> tuple[dict[str, object], tuple[str, ...]] | None:
        """Return the ``(parameter_values, added_advanced)`` the row ``key`` stores.

        ``key`` is the charge-qualified row key. The row's own settings, without
        the project defaults it inherits. ``None`` when there is no active
        definition or no such row.
        """

        definition = self.active_definition()
        if definition is None:
            return None
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None:
            return None
        row = next((row for row in config.rows if muon_relaxation_row_key(row) == str(key)), None)
        if row is None:
            return None
        return dict(row.parameter_values), tuple(row.added_advanced)

    def render_row_inputs(
        self,
        key: object,
        parameter_values: Mapping[str, object],
        added_advanced: Sequence[str],
    ) -> JobInputTexts | None:
        """Render the ``.cell`` and ``.param`` one row would write with ``parameter_values``.

        A pure preview of :meth:`continue_job`: the values go through the same
        normalisation the row applies when it stores an edit, then the row's
        ``JobSpec`` is expanded with them and rendered, without persisting
        anything or touching the filesystem. Mirrors :meth:`materialize`'s own
        expansion call (same host structure and project k-point/cutoff
        defaults). Powers the Continue-relaxation dialog's ``Parameter
        changes`` table. ``None`` when there is no active definition, no host
        structure, no matching row, or expansion fails.
        """

        document = self._context.document()
        if document is None:
            return None
        definition = self.active_definition()
        if definition is None:
            return None
        config = deserialize_muon_relaxation_config(definition.parameters.get("configuration", {}))
        if config is None or not definition.input_structure_ids:
            return None
        # Also drops an imported row's own key, so previewing one yields None —
        # correct, since the run it describes is already finished and its real
        # inputs are archived in its job directory.
        config = _without_imported_rows(config)
        host = document.structures.get(definition.input_structure_ids[0])
        if host is None:
            return None
        row_key = str(key)
        if not any(muon_relaxation_row_key(row) == row_key for row in config.rows):
            return None

        stored = self._view.row_values_to_store(row_key, parameter_values)
        values = dict(parameter_values) if stored is None else stored
        rows = tuple(
            replace(
                row,
                parameter_values=values,
                added_advanced=tuple(str(name) for name in added_advanced),
            )
            if muon_relaxation_row_key(row) == row_key
            else row
            for row in config.rows
        )
        try:
            expansion = expand_muon_relaxation_jobs(
                host_structure=host,
                config=replace(config, rows=rows),
                project_cutoff_energy=document.project.recommended_cutoff_energy,
                project_kpoint_grid=document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=document.project.recommended_kpoint_lattice,
                project_xc_functional=document.project.recommended_xc_functional,
            )
        except (ValueError, StructureBridgeError):
            return None
        for spec in expansion.jobs:
            spec_key = (
                f"{muon_charge_state_prefix(str(spec.provenance.get('charge_state', '')))}"
                f":{spec.provenance.get('label', '')}"
            )
            if spec_key == row_key:
                cell_text, param_text = render_job(spec)
                return JobInputTexts(cell=cell_text, param=param_text)
        return None

    def continue_job(
        self,
        key: object,
        parameter_values: Mapping[str, object],
        added_advanced: Sequence[str] | None = None,
    ) -> Job | None:
        """Mutate the row in place, prepare its job for continuation, and re-render.

        ``key`` must be the charge-qualified row key. Orchestrates the
        controller-owned steps of the Continue-relaxation accept flow:
        :meth:`update_row_parameters` mutates the row and mirrors the edit into
        the widget form; ``ProjectService.prepare_job_continuation`` returns
        the row's job to ``DEFINED`` while preserving ``remote_directory``;
        :meth:`materialize` rewrites the job's directory in place from the new
        parameters (the existing DEFINED path — no new ``Job`` is minted,
        ``job_ids_by_label``/``job_seed_names_by_label`` are untouched,
        matching finding 1 of the warm-start plan); ``record_job_continuation``
        then appends the provenance milestone. This deliberately does not
        route through :meth:`submit`/:meth:`pre_submit_check`: the row's site
        already matched the active host when the job it is continuing was
        first submitted, and the host-mismatch dialog has no sensible role in
        resubmitting the very same site. Submission and view refresh are the
        caller's responsibility.

        The job's status is checked with :meth:`row_is_continuable` *before*
        ``update_row_parameters`` runs, so a row whose job is not continuable
        (e.g. it started running between the menu opening and the dialog being
        accepted) is left completely untouched rather than having its stored
        parameters mutated with no corresponding directory rewrite. The
        document lookup is also hoisted above ``update_row_parameters`` for the
        same reason: a missing document must not leave the definition row
        mutated while the job itself is never touched. Returns ``None`` in
        either case, or when there is no active definition or no matching row.

        :meth:`materialize` can itself return an empty list (a caught
        ``ValueError``/``StructureBridgeError`` from expansion, or a missing
        host structure/output root) without raising — in that case the job is
        left ``DEFINED`` with its remote directory preserved (so it stays
        recoverable and a later submit re-runs materialize), but the directory
        on disk still holds the *previous* run's ``.param``, not the
        just-edited values. This method returns ``None`` in that case too, so
        the caller never submits a job whose on-disk inputs disagree with the
        diff it just showed the user. The provenance milestone is appended only
        *after* that check passes: a continuation that was never rendered (and
        therefore never submitted) must not leave a journal entry claiming it
        happened.
        """

        jobs = self.jobs_for_keys([key])
        if not jobs or jobs[0].status not in CONTINUABLE_JOB_STATUSES:
            return None
        job = jobs[0]
        document = self._context.document()
        if document is None:
            return None
        if not self.update_row_parameters(key, parameter_values, added_advanced):
            return None
        self._context.project_service.prepare_job_continuation(job)
        if not self.materialize([key]):
            return None
        self._context.project_service.record_job_continuation(
            document, job=job, remote_directory=job.remote_directory
        )
        return job

    # --- result contract ---

    def validate_artifacts(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> str | None:
        """Fail a muon relaxation with no fresh ``-out.cell`` or no muon in it.

        Applies :func:`~mufinder.app.workflow_controllers.base.relaxation_artifact_failure`
        first, then looks for the ``H:mu`` position in the carried relaxed cell
        (no second disk read).
        """

        seed_name = self._context.job_seed_name(job)
        failure = relaxation_artifact_failure(job, parsed, run, artifacts, seed_name=seed_name)
        if failure is not None:
            return failure
        relaxed = artifacts.relaxed_cell
        if relaxed is None or muon_fractional_position_from_cell(relaxed) is None:
            return f"Expected H:mu muon position was not found in {seed_name}-out.cell."
        return None

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute lattice/muon-position payload, derived structure, and the
        source-site mutation for a completed muon-relaxation job.

        :meth:`validate_artifacts` has already accepted the completion (the
        ``-out.cell`` exists, parses, belongs to this run and contains an
        ``H:mu`` position), so the carried ``artifacts.relaxed_cell`` is present
        and muonated on this success path.
        The file is parsed **once** on the runner worker thread; this controller
        consumes the carried cell rather than re-reading disk.

        Mutating the source :class:`~mufinder.domain.models.MuonSite`
        (``status``/``relaxation_run_id``/``total_energy_eV`` and the host-cell
        ``relaxed_fractional_position``) is a
        sanctioned document side effect (see the ``ControllerResult``
        docstring). Output files are **not** contributed here: the call site
        already passes ``_output_files_for_job(job)`` for every job.

        The derived structure is created on first completion and **refreshed in
        place** on every later one, so a continued job keeps a single, stable
        structure id — see
        :func:`~mufinder.app.workflow_controllers.base.upsert_job_derived_structure`.
        The site fields are rewritten from the same completion, so they and the
        structure always describe the same run.
        """

        document = self._context.document()
        if document is None:
            return ControllerResult()

        relaxed = artifacts.relaxed_cell
        if relaxed is None or artifacts.lattice_abc is None:
            return ControllerResult()
        lattice_a, lattice_b, lattice_c = artifacts.lattice_abc

        payload: dict[str, object] = {
            "lattice_a": lattice_a,
            "lattice_b": lattice_b,
            "lattice_c": lattice_c,
        }

        definition, row = self.definition_and_row_for_job(job)
        if definition is None:
            definition = self.active_definition()
        final_supercell_position = muon_fractional_position_from_cell(relaxed)
        if final_supercell_position is None:
            # build_result runs only on the validated success path:
            # validate_artifacts fails muon-less cells before dispatch reaches
            # build_result. Fail loudly if that contract is violated.
            raise ValueError(
                "Muon-relaxation build_result requires a relaxed cell containing "
                "the muon; validate_artifacts must reject muon-less cells first."
            )
        final_host_position = _fold_supercell_fractional_to_host(
            final_supercell_position,
            row.supercell_matrix if row is not None else None,
        )
        payload["muon_final_fractional_position_supercell"] = list(final_supercell_position)
        payload["muon_final_fractional_position_host"] = list(final_host_position)

        parent_structure_id = (
            definition.input_structure_ids[0]
            if definition is not None and definition.input_structure_ids
            else None
        )
        relaxed_structure = upsert_job_derived_structure(
            self._context.project_service,
            document,
            job,
            origin="muon_relaxation",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            relaxed=relaxed,
            parent_structure_id=parent_structure_id,
        )

        # ``run`` is the run this completion belongs to — every submission mints
        # a fresh CalculationRun (``JobExecutionController._create_active_run``),
        # so a continued job's re-completion advances the site's link to the new
        # run rather than leaving it on the superseded one.
        # A site relaxed in both charge states stays linked to the charge state
        # it was first relaxed in; the other relaxation is found through its row.
        site = document.sites.get(row.site_id) if row is not None else None
        if site is not None:
            record_site_relaxation(
                document,
                site,
                run,
                job,
                total_energy_eV=parsed.total_energy,
                fractional_position=final_host_position,
            )

        return ControllerResult(
            payload=payload,
            derived_structure_ids=[relaxed_structure.id],
        )

    # --- view contract ---

    def refresh_view(self) -> None:
        self._view.set_runtime_states(self.build_states())


__all__ = ["MuonRelaxationController", "MuonRelaxationDeps"]
