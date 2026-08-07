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
(``status``/``relaxation_run_id``/``relative_energy_eV``). Following the
success-path-only convention, the failure guards (``geom_converged``,
missing/unparseable ``-out.cell``, and the missing ``H:mu`` position) stay
inline in ``MainWindow._on_local_job_finished`` and abort the job before this
controller's ``build_result`` runs.

Shared helpers that non-migrating code still calls remain on ``MainWindow`` and
are reached via :class:`MuonRelaxationDeps`, following the shared-context +
per-controller-deps + view pattern documented in
:mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.domain.enums import ACTIVE_JOB_STATUSES, JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
    Structure,
)
from mufinder.services.project_service import SiteHostMismatch
from mufinder.workflow import StructureBridgeError, write_job_directory

from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.relaxation_adapter import (
    MUON_RELAXATION_CALC_TYPES,
    expand_muon_relaxation_jobs,
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
    WorkflowControllerContext,
    WorkflowJobController,
    apply_row_parameters_to_form_state,
)
from mufinder.app.workflow_submission import effective_row_status


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

    if matrix is None:
        return tuple(float(component) % 1.0 for component in position)
    x, y, z = position
    host_position = (
        x * matrix[0][0] + y * matrix[1][0] + z * matrix[2][0],
        x * matrix[0][1] + y * matrix[1][1] + z * matrix[2][1],
        x * matrix[0][2] + y * matrix[1][2] + z * matrix[2][2],
    )
    return tuple(float(component) % 1.0 for component in host_position)


@dataclass(frozen=True)
class MuonRelaxationDeps:
    """Muon-relaxation-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating code still calls
    them there: the definition/config/mapping helpers are shared with row
    removal, staging, and dashboard sync; ``muon_fractional_position_from_cell``
    is still called by the inline ``H:mu`` failure guard in
    ``_on_local_job_finished`` (against the carried, already-parsed cell); and
    ``coerce_fractional_triplet`` and ``muon_relaxation_row_key`` have other
    view-refresh callers. The relaxed ``-out.cell`` is parsed once on the runner
    worker thread and delivered to ``build_result`` via
    :class:`~mufinder.services.job_artifacts.ParsedJobArtifacts`, so no parse dep
    is needed here. The controller reaches these helpers only through required
    callables.
    Workflow-specific dependencies like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`;
    cross-workflow job/definition readers (``job_output_directory``,
    ``job_seed_name``, ``run_for_job``, ``definition_output_directory``, the
    energy/force readers) live there instead. Active-job membership uses the
    domain-layer :data:`~mufinder.domain.enums.ACTIVE_JOB_STATUSES` constant
    directly (``app`` → ``domain`` is a legal import).
    """

    active_muon_relaxation_definition: Callable[[], CalculationDefinition | None]
    deserialize_muon_relaxation_config: Callable[[object], MuonRelaxationConfig | None]
    serialize_muon_relaxation_config: Callable[
        [MuonRelaxationConfig], dict[str, object]
    ]
    sync_equivalent_muon_relaxation_mappings: Callable[[CalculationDefinition], None]
    sync_active_muon_relaxation_dashboard_jobs: Callable[[CalculationDefinition], None]
    muon_relaxation_form_config: Callable[[], MuonRelaxationConfig | None]
    muon_relaxation_row_signature: Callable[[MuonRelaxationRow], tuple[object, ...]]
    muon_relaxation_row_key: Callable[[MuonRelaxationRow], str]
    job_for_muon_relaxation_label: Callable[
        [CalculationDefinition, str, str | None], Job | None
    ]
    handle_site_host_mismatch: Callable[[SiteHostMismatch], bool]
    # Shared with the still-inline result-block failure guards:
    muon_relaxation_definition_and_row_for_job: Callable[
        [Job], tuple[CalculationDefinition | None, MuonRelaxationRow | None]
    ]
    muon_fractional_position_from_cell: Callable[
        [object], tuple[float, float, float] | None
    ]
    coerce_fractional_triplet: Callable[[object], tuple[float, float, float] | None]


class MuonRelaxationController(WorkflowJobController):
    """Controller for the muon structural-relaxation workflow."""

    calc_types = frozenset(MUON_RELAXATION_CALC_TYPES.values())
    no_ready_message = (
        "Only configured muon-relaxation rows that have not yet been submitted can be submitted."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: MuonRelaxationDeps,
        view: MuonRelaxationWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- submission contract ---

    def normalize(self, value: object) -> str:
        return str(value)

    def is_ready(self, states: Mapping[object, object], key: object) -> bool:
        """A row is ready when its own key — or, for a bare label, any
        charge-qualified key ending in ``":{label}"`` — is in a ready status.

        Behavior-preserving move of ``MainWindow._muon_relaxation_label_is_ready``.
        """

        ready_statuses = {"ready", "defined", "not generated"}
        state = states.get(key)
        if state is not None and state.status in ready_statuses:
            return True
        if ":" in str(key):
            return False
        suffix = f":{key}"
        return any(
            candidate.status in ready_statuses
            for candidate_key, candidate in states.items()
            if candidate_key.endswith(suffix)
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
        definition = self._deps.active_muon_relaxation_definition()
        if definition is None or not definition.input_site_ids:
            return True
        mismatch = self._context.project_service.validate_sites_match_active_host(
            document, definition.input_site_ids
        )
        return mismatch is None or self._deps.handle_site_host_mismatch(mismatch)

    def submission_groups(
        self, jobs: list[Job]
    ) -> list[tuple[list[Job], dict[str, object]]]:
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
        form_config = self._deps.muon_relaxation_form_config()
        if form_config is not None:
            for row in form_config.rows:
                states[self._deps.muon_relaxation_row_key(row)] = RelaxationRunState(
                    status="ready"
                )

        definition = self._deps.active_muon_relaxation_definition()
        if definition is None:
            return states
        self._deps.sync_equivalent_muon_relaxation_mappings(definition)
        self._deps.sync_active_muon_relaxation_dashboard_jobs(definition)
        config = self._deps.deserialize_muon_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return states

        if form_config is not None:
            definition_rows_by_key = {
                self._deps.muon_relaxation_row_key(row): row for row in config.rows
            }
            rows_for_status = []
            for row in form_config.rows:
                key = self._deps.muon_relaxation_row_key(row)
                definition_row = definition_rows_by_key.get(key)
                if definition_row is None:
                    continue
                if self._deps.muon_relaxation_row_signature(
                    row
                ) == self._deps.muon_relaxation_row_signature(definition_row):
                    rows_for_status.append(definition_row)
                    continue
                job = self._deps.job_for_muon_relaxation_label(
                    definition,
                    definition_row.label,
                    definition_row.charge_state,
                )
                if job is not None and job.status in ACTIVE_JOB_STATUSES:
                    rows_for_status.append(definition_row)
        else:
            rows_for_status = list(config.rows)

        for row in rows_for_status:
            state_key = self._deps.muon_relaxation_row_key(row)
            job = self._deps.job_for_muon_relaxation_label(
                definition, row.label, row.charge_state
            )
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
                    final_fractional_position = self._deps.coerce_fractional_triplet(
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
        definition = self._deps.active_muon_relaxation_definition()
        if definition is None:
            return []
        self._deps.sync_equivalent_muon_relaxation_mappings(definition)
        self._deps.sync_active_muon_relaxation_dashboard_jobs(definition)
        config = self._deps.deserialize_muon_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        wanted = {str(key) for key in keys}
        for row in config.rows:
            if not any(
                _muon_relaxation_selection_matches(row, selection) for selection in wanted
            ):
                continue
            job = self._deps.job_for_muon_relaxation_label(
                definition, row.label, row.charge_state
            )
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def materialize(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        definition = self._deps.active_muon_relaxation_definition()
        if definition is None:
            return []
        self._deps.sync_equivalent_muon_relaxation_mappings(definition)
        self._deps.sync_active_muon_relaxation_dashboard_jobs(definition)
        config = self._deps.deserialize_muon_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None or not definition.input_structure_ids:
            return []
        host = document.structures.get(definition.input_structure_ids[0])
        output_root = self._context.definition_output_directory(definition)
        if host is None or output_root is None:
            return []
        try:
            expansion = expand_muon_relaxation_jobs(
                host_structure=host,
                config=config,
                project_cutoff_energy=document.project.recommended_cutoff_energy,
                project_kpoint_grid=document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=document.project.recommended_kpoint_lattice,
            )
        except (ValueError, StructureBridgeError):
            return []
        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping
        seed_mapping = definition.parameters.setdefault("job_seed_names_by_label", {})
        if not isinstance(seed_mapping, dict):
            seed_mapping = {}
            definition.parameters["job_seed_names_by_label"] = seed_mapping
        spec_by_key = {
            f"{muon_charge_state_prefix(str(spec.provenance.get('charge_state', '')))}:{spec.provenance.get('label', '')}": spec
            for spec in expansion.jobs
        }
        wanted = {str(key) for key in keys}
        rows_by_label = [
            row
            for row in config.rows
            if any(
                _muon_relaxation_selection_matches(row, selection) for selection in wanted
            )
        ]
        created: list[Job] = []
        for row in rows_by_label:
            key = f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"
            spec = spec_by_key.get(key)
            existing_job = self._deps.job_for_muon_relaxation_label(
                definition, row.label, row.charge_state
            )
            if existing_job is not None:
                if existing_job.status == JobStatus.DEFINED and spec is not None:
                    seed_mapping[key] = spec.name
                    write_job_directory(
                        spec, output_root / muon_charge_state_prefix(row.charge_state)
                    )
                created.append(existing_job)
                continue
            if spec is None:
                continue
            charge_root = output_root / muon_charge_state_prefix(row.charge_state)
            write_job_directory(spec, charge_root)
            job = self._context.project_service.add_job(
                document,
                label=row.label,
                workflow_stage=spec.workflow_stage,
                calc_type=muon_relaxation_calc_type(row.charge_state),
            )
            mapping[key] = str(job.id)
            seed_mapping[key] = spec.name
            created.append(job)
        deduped: list[Job] = []
        seen: set[UUID] = set()
        for job in created:
            if job.id in seen:
                continue
            seen.add(job.id)
            deduped.append(job)
        if deduped:
            self._context.persist_ui_state(
                failure_message="Muon relaxation jobs were generated, but autosave failed."
            )
        return deduped

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

        definition = self._deps.active_muon_relaxation_definition()
        if definition is None:
            return False
        config = self._deps.deserialize_muon_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return False
        row_key = str(key)
        if not any(
            self._deps.muon_relaxation_row_key(row) == row_key for row in config.rows
        ):
            return False

        values = dict(parameter_values)
        advanced = (
            tuple(str(name) for name in added_advanced)
            if added_advanced is not None
            else None
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
            if self._deps.muon_relaxation_row_key(row) == row_key
            else row
            for row in config.rows
        )
        definition.parameters["configuration"] = (
            self._deps.serialize_muon_relaxation_config(replace(config, rows=rows))
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
                tuple(str(name) for name in raw_advanced)
                if isinstance(raw_advanced, list)
                else (),
            )
        return None

    # --- result contract ---

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute lattice/muon-position payload, derived structure, and the
        source-site mutation for a completed muon-relaxation job.

        The inline ``_on_local_job_finished`` prologue has already validated
        that the ``-out.cell`` exists, parses, and contains an ``H:mu``
        position (it aborts the job otherwise), so the carried
        ``artifacts.relaxed_cell`` is present and muonated on this success path.
        The file is parsed **once** on the runner worker thread; this controller
        consumes the carried cell rather than re-reading disk.

        Mutating the source :class:`~mufinder.domain.models.MuonSite`
        (``status``/``relaxation_run_id``/``relative_energy_eV``) is a
        sanctioned document side effect (see the ``ControllerResult``
        docstring). Output files are **not** contributed here: the call site
        already passes ``_output_files_for_job(job)`` for every job.
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

        definition, row = self._deps.muon_relaxation_definition_and_row_for_job(job)
        if definition is None:
            definition = self._deps.active_muon_relaxation_definition()
        final_supercell_position = self._deps.muon_fractional_position_from_cell(relaxed)
        if final_supercell_position is None:
            # build_result runs only on the validated success path: the inline
            # H:mu guard in the job-finished handler fails muon-less cells
            # before dispatch reaches this controller. Fail loudly if violated.
            raise ValueError(
                "Muon-relaxation build_result requires a relaxed cell containing "
                "the muon; the inline guard must reject muon-less cells first."
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
        relaxed_structure = Structure(
            label=f"{job.label} relaxed",
            origin="muon_relaxation",
            lattice_vectors=[list(vector) for vector in relaxed.lattice_vectors],
            species=list(relaxed.species),
            fractional_positions=[list(pos) for pos in relaxed.fractional_positions],
            parent_structure_id=parent_structure_id,
            properties={
                "source_job_id": str(job.id),
                "source_workflow_stage": WorkflowStage.MUON_RELAXATION.value,
            },
        )
        self._context.project_service.add_structure(document, relaxed_structure)

        site = document.sites.get(row.site_id) if row is not None else None
        if site is not None:
            site.status = "relaxed"
            site.relaxation_run_id = run.id
            site.relative_energy_eV = parsed.total_energy

        return ControllerResult(
            payload=payload,
            derived_structure_ids=[relaxed_structure.id],
        )

    # --- view contract ---

    def refresh_view(self) -> None:
        self._view.set_runtime_states(self.build_states())


__all__ = ["MuonRelaxationController", "MuonRelaxationDeps"]
