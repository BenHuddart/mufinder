"""Pristine-relaxation workflow controller.

Owns the submit → materialize → result → refresh cycle for the CASTEP pristine
structural relaxation. This is the first migration onto
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController` that gives
:meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.build_result`
its first genuine implementation: a completed ``castep_pristine_relaxation`` job
contributes the relaxed lattice parameters to the result payload and creates the
derived relaxed :class:`~mufinder.domain.models.Structure` on the document.

The logic here is a behavior-preserving move of the former ``MainWindow`` methods
``_materialize_pristine_relaxation_jobs``, ``_build_pristine_relaxation_states``,
``_jobs_for_pristine_relaxation_labels``, ``_refresh_pristine_relaxation_rows``,
and the ``castep_pristine_relaxation`` portion of ``_on_local_job_finished``'s
result block. :meth:`PristineRelaxationController.validate_artifacts` fails a job
whose geometry did not converge or whose ``-out.cell`` is missing, unparseable or
older than the current run, through the
:func:`~mufinder.app.workflow_controllers.base.relaxation_artifact_failure` checks
it shares with the muon-relaxation controller.

Definition and job lookups come from the base class (:attr:`stage` and
:attr:`active_key` drive them). Shared helpers that non-migrating code still
calls remain on ``MainWindow`` and are reached via
:class:`PristineRelaxationDeps` — the per-controller deps dataclass this module
defines, following the shared-context + per-controller-deps + view pattern
documented in :mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
)
from mufinder.services.project_service import CONTINUABLE_JOB_STATUSES
from mufinder.workflow import StructureBridgeError, render_job, write_job_directory

from mufinder.app.adapter_common import AdapterExpansion
from mufinder.app.relaxation_types import PristineRelaxationConfig, RelaxationRunState
from mufinder.app.relaxation_widget import PristineRelaxationWidget
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    dedupe_jobs,
    apply_row_parameters_to_form_state,
    relaxation_artifact_failure,
    upsert_job_derived_structure,
)
from mufinder.app.workflow_submission import effective_row_status


@dataclass(frozen=True)
class PristineRelaxationDeps:
    """Pristine-relaxation-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating code still calls
    them there. The controller reaches them only through these required
    callables. Workflow-specific dependencies like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`;
    cross-workflow job/definition readers (``job_output_directory``,
    ``job_seed_name``, ``run_for_job``, ``definition_output_directory``, the
    energy/force readers) live there instead. The relaxed ``-out.cell`` is parsed
    once on the runner worker thread and delivered to ``build_result`` via
    :class:`~mufinder.services.job_artifacts.ParsedJobArtifacts`, so no parse dep
    is needed here.
    """

    deserialize_pristine_relaxation_config: Callable[
        [object], PristineRelaxationConfig | None
    ]
    serialize_pristine_relaxation_config: Callable[
        [PristineRelaxationConfig], dict[str, object]
    ]
    sync_equivalent_pristine_relaxation_mappings: Callable[
        [CalculationDefinition], None
    ]
    pristine_relaxation_form_config: Callable[[], PristineRelaxationConfig | None]
    pristine_relaxation_row_signature: Callable[[object], tuple[object, ...]]
    expand_pristine_relaxation_jobs: Callable[
        [CalculationDefinition, PristineRelaxationConfig], AdapterExpansion | None
    ]


class PristineRelaxationController(WorkflowJobController):
    """Controller for the pristine structural-relaxation workflow."""

    calc_types = frozenset({"castep_pristine_relaxation"})
    stage = WorkflowStage.PRISTINE_RELAXATION
    active_key = "active_pristine_relaxation_definition_id"
    no_ready_message = (
        "Only configured pristine-relaxation rows that have not yet been submitted can be submitted."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: PristineRelaxationDeps,
        view: PristineRelaxationWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready)
    # exactly mirror the former ``_on_submit_*_from_pristine_relaxation`` handlers.
    # There is no pre_submit_check and submission is a single group.

    def normalize(self, value: object) -> str:
        return str(value)

    def build_states(self) -> dict[str, RelaxationRunState]:
        states: dict[str, RelaxationRunState] = {}
        document = self._context.document()
        if document is None:
            return states

        form_config = self._deps.pristine_relaxation_form_config()
        if form_config is not None:
            for row in form_config.rows:
                states[row.label] = RelaxationRunState(status="ready")

        definition = self.active_definition()
        if definition is None:
            return states
        self._deps.sync_equivalent_pristine_relaxation_mappings(definition)
        config = self._deps.deserialize_pristine_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return states

        if form_config is not None:
            definition_rows_by_label = {row.label: row for row in config.rows}
            rows_for_status = [
                definition_row
                for row in form_config.rows
                if (definition_row := definition_rows_by_label.get(row.label)) is not None
                and self._deps.pristine_relaxation_row_signature(row)
                == self._deps.pristine_relaxation_row_signature(definition_row)
            ]
        else:
            rows_for_status = list(config.rows)

        for row in rows_for_status:
            job = self.job_for_key(definition, row.label)
            if job is None:
                states[row.label] = RelaxationRunState(status="ready")
                continue

            status_value = (
                "ready" if job.status == JobStatus.DEFINED else effective_row_status(job)
            )
            run = self._context.run_for_job(job.id)
            if (
                status_value == JobStatus.COMPLETED.value
                and run is not None
                and run.convergence_achieved is False
            ):
                status_value = "not converged"

            lattice_a = None
            lattice_b = None
            lattice_c = None
            if run is not None and run.result_id is not None:
                result = document.results.get(run.result_id)
                if result is not None and isinstance(result.data, dict):
                    lattice_a = result.data.get("lattice_a")
                    lattice_b = result.data.get("lattice_b")
                    lattice_c = result.data.get("lattice_c")

            states[row.label] = RelaxationRunState(
                status=status_value,
                total_energy=self._context.total_energy_for_job(job.id),
                max_force=self._context.max_force_for_job(job.id),
                lattice_a=float(lattice_a) if isinstance(lattice_a, (int, float)) else None,
                lattice_b=float(lattice_b) if isinstance(lattice_b, (int, float)) else None,
                lattice_c=float(lattice_c) if isinstance(lattice_c, (int, float)) else None,
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
        self._deps.sync_equivalent_pristine_relaxation_mappings(definition)

        selected: list[Job] = []
        seen: set[UUID] = set()
        for label in keys:
            job = self.job_for_key(definition, str(label))
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def materialize(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []

        definition = self.active_definition()
        if definition is None:
            return []
        self._deps.sync_equivalent_pristine_relaxation_mappings(definition)

        config = self._deps.deserialize_pristine_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return []
        if not definition.input_structure_ids:
            return []

        output_root = self._context.definition_output_directory(definition)
        if output_root is None:
            return []

        try:
            expansion = self._deps.expand_pristine_relaxation_jobs(definition, config)
        except (ValueError, StructureBridgeError):
            return []
        if expansion is None:
            return []

        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping

        spec_by_label = {
            str(spec.provenance.get("label", "")): spec for spec in expansion.jobs
        }

        created: list[Job] = []
        for label in keys:
            label = str(label)
            spec = spec_by_label.get(label)
            existing_job = self.job_for_key(definition, label)
            if existing_job is not None:
                if existing_job.status == JobStatus.DEFINED and spec is not None:
                    write_job_directory(spec, output_root)
                created.append(existing_job)
                continue

            if spec is None:
                continue

            write_job_directory(spec, output_root)
            job = self._context.project_service.add_job(
                document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type="castep_pristine_relaxation",
            )
            mapping[label] = str(job.id)
            created.append(job)

        return dedupe_jobs(created)

    # --- in-place row parameter mutation ---

    def update_row_parameters(
        self,
        key: object,
        parameter_values: Mapping[str, object],
        added_advanced: Sequence[str] | None = None,
    ) -> bool:
        """Rewrite one row's CASTEP parameters inside the ACTIVE definition.

        ``key`` is the row label. Returns ``False`` when there is no active
        definition or it holds no such row.

        The edit must never go through ``register_pristine_relaxation_definition``:
        a new definition arrives with an empty ``job_ids_by_label``, which severs
        the row→job binding, so ``materialize`` would mint a duplicate ``Job``
        onto the row's existing directory. This mutates the very definition
        ``materialize`` reads, leaving ``job_ids_by_label`` and every other row
        untouched, and rewrites the row through the serialize/deserialize pair so
        the stored shape stays canonical.

        The parameters actually stored are the ones the *widget* reports after the
        mirror, not the caller's raw mapping: the table strips implicit project
        defaults, and :meth:`build_states` drops a row whose form signature differs
        from its definition signature (this workflow has no active-job escape
        hatch, so a desynced row falls back to a jobless "ready").
        """

        definition = self.active_definition()
        if definition is None:
            return False
        config = self._deps.deserialize_pristine_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return False
        label = str(key)
        if not any(row.label == label for row in config.rows):
            return False

        values = dict(parameter_values)
        advanced = (
            tuple(str(name) for name in added_advanced)
            if added_advanced is not None
            else None
        )
        mirrored = self._mirror_row_parameters_into_form(label, values, advanced)
        if mirrored is not None:
            values, advanced = mirrored

        rows = tuple(
            replace(
                row,
                parameter_values=dict(values),
                added_advanced=(advanced if advanced is not None else row.added_advanced),
            )
            if row.label == label
            else row
            for row in config.rows
        )
        definition.parameters["configuration"] = (
            self._deps.serialize_pristine_relaxation_config(replace(config, rows=rows))
        )

        self._context.persist_ui_state(
            failure_message="Pristine relaxation parameters were updated, but autosave failed."
        )
        self.refresh_view()
        return True

    def _mirror_row_parameters_into_form(
        self,
        label: str,
        parameter_values: Mapping[str, object],
        added_advanced: tuple[str, ...] | None,
    ) -> tuple[dict[str, object], tuple[str, ...] | None] | None:
        """Apply the edit to the widget form state and read the result back.

        Returns the widget's normalized ``(parameter_values, added_advanced)`` for
        the row, or ``None`` when the form does not hold it (in which case the
        caller stores its own values — the row is not projected from the form
        anyway).
        """

        updated = apply_row_parameters_to_form_state(
            self._view.collect_state(),
            row_matches=lambda row: str(row.get("label", "")).strip() == label,
            parameter_values=parameter_values,
            added_advanced=added_advanced,
            parameter_key="params",
        )
        if updated is None:
            return None

        self._view.set_form_state(updated)
        normalized = self._view.collect_state()
        document = self._context.document()
        if document is not None and isinstance(normalized, dict):
            document.ui_state["pristine_relaxation_form"] = normalized
        rows = normalized.get("rows", []) if isinstance(normalized, dict) else []
        for row in rows:
            if not isinstance(row, dict) or str(row.get("label", "")).strip() != label:
                continue
            raw_params = row.get("params", {})
            raw_advanced = row.get("added_advanced", [])
            return (
                dict(raw_params) if isinstance(raw_params, dict) else {},
                tuple(str(name) for name in raw_advanced)
                if isinstance(raw_advanced, list)
                else (),
            )
        return None

    # --- continuation ("Continue relaxation…") ---

    def row_is_continuable(self, key: object) -> bool:
        """Return whether ``key``'s row has a job in :data:`CONTINUABLE_JOB_STATUSES`.

        Resolves the row's job through :meth:`jobs_for_keys` — never from a
        table's *displayed* status text, which reads "not converged" for a
        completed-but-unconverged job and "ready" for a dismissed failed job
        (see ``mufinder.app.workflow_submission.effective_row_status``). Both
        of those rows must still gate on the job's real, durable status. Called
        live at context-menu-open time (not cached), so a status change between
        refreshes (e.g. a dashboard dismissal) is always reflected.
        """

        jobs = self.jobs_for_keys([key])
        return bool(jobs) and jobs[0].status in CONTINUABLE_JOB_STATUSES

    def render_row_param_text(
        self,
        key: object,
        parameter_values: Mapping[str, object],
        added_advanced: Sequence[str],
    ) -> str:
        """Render the ``.param`` text one row would produce with ``parameter_values``.

        A pure preview: expands the row's ``JobSpec`` with ``parameter_values``/
        ``added_advanced`` substituted for its currently-stored values and
        renders it, without persisting anything or touching the filesystem.
        Powers the Continue-relaxation dialog's live keyword diff — there is no
        ``.param`` reader in ``castep_io``, so the diff is computed by
        expanding and re-rendering rather than parsing (see the warm-start
        plan's diff-by-re-render note). Returns ``""`` when there is no active
        definition, no matching row, or expansion fails.
        """

        definition = self.active_definition()
        if definition is None:
            return ""
        config = self._deps.deserialize_pristine_relaxation_config(
            definition.parameters.get("configuration", {})
        )
        if config is None:
            return ""
        label = str(key)
        if not any(row.label == label for row in config.rows):
            return ""

        rows = tuple(
            replace(
                row,
                parameter_values=dict(parameter_values),
                added_advanced=tuple(str(name) for name in added_advanced),
            )
            if row.label == label
            else row
            for row in config.rows
        )
        try:
            expansion = self._deps.expand_pristine_relaxation_jobs(
                definition, replace(config, rows=rows)
            )
        except (ValueError, StructureBridgeError):
            return ""
        if expansion is None:
            return ""
        for spec in expansion.jobs:
            if str(spec.provenance.get("label", "")) == label:
                return render_job(spec)[1]
        return ""

    def continue_job(
        self,
        key: object,
        parameter_values: Mapping[str, object],
        added_advanced: Sequence[str] | None = None,
    ) -> Job | None:
        """Mutate the row in place, prepare its job for continuation, and re-render.

        Orchestrates the controller-owned steps of the Continue-relaxation
        accept flow: :meth:`update_row_parameters` mutates the row and mirrors
        the edit into the widget form; ``ProjectService.prepare_job_continuation``
        returns the row's job to ``DEFINED`` while preserving
        ``remote_directory``; :meth:`materialize` rewrites the job's directory
        in place from the new parameters (the existing DEFINED path — no new
        ``Job`` is minted, ``job_ids_by_label`` is untouched, matching finding 1
        of the warm-start plan); ``record_job_continuation`` then appends the
        provenance milestone. Submission and view refresh are the caller's
        responsibility: they need the job's original execution target and
        cluster-connection/dialog state this controller does not have.

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
        output root) without raising — in that case the job is left ``DEFINED``
        with its remote directory preserved (so it stays recoverable and a
        later submit re-runs materialize), but the directory on disk still
        holds the *previous* run's ``.param``, not the just-edited values. This
        method returns ``None`` in that case too, so the caller never submits a
        job whose on-disk inputs disagree with the diff it just showed the user.
        The provenance milestone is appended only *after* that check passes: a
        continuation that was never rendered (and therefore never submitted)
        must not leave a journal entry claiming it happened.
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
        """Fail a relaxation that did not converge or left no fresh ``-out.cell``.

        See :func:`~mufinder.app.workflow_controllers.base.relaxation_artifact_failure`
        for the checks and their messages; the carrier is read, not disk.
        """

        return relaxation_artifact_failure(
            job, parsed, run, artifacts, seed_name=self._context.job_seed_name(job)
        )

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute the relaxed lattice parameters and derived structure.

        :meth:`validate_artifacts` has already accepted the completion (the
        ``-out.cell`` exists, parses and belongs to this run), so the carried
        ``artifacts.relaxed_cell`` is present on this success path. The file is parsed **once** on the runner
        worker thread; this controller consumes the carried cell rather than
        re-reading disk.

        Output files are **not** contributed here: the ``_on_local_job_finished``
        call site already passes ``_output_files_for_job(job)`` for every job, and
        that list already includes ``{seed_name}-out.cell``. Adding them again
        would duplicate entries.

        The derived structure is created on first completion and **refreshed in
        place** on every later one, so a continued job keeps a single, stable
        structure id — see :func:`~mufinder.app.workflow_controllers.base.upsert_job_derived_structure`.
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

        definition = self.active_definition()
        parent_structure_id = (
            definition.input_structure_ids[0]
            if definition is not None and definition.input_structure_ids
            else None
        )
        relaxed_structure = upsert_job_derived_structure(
            self._context.project_service,
            document,
            job,
            origin="pristine_relaxation",
            workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
            relaxed=relaxed,
            parent_structure_id=parent_structure_id,
        )

        return ControllerResult(
            payload=payload,
            derived_structure_ids=[relaxed_structure.id],
        )

    # --- view contract ---

    def refresh_view(self) -> None:
        self._view.set_runtime_states(self.build_states())


__all__ = ["PristineRelaxationController", "PristineRelaxationDeps"]
