"""UEP-source workflow controller: the UEP page and its analysis windows.

Owns the submit → materialize → result → refresh cycle for the CASTEP UEP
(unperturbed electrostatic potential) source jobs on
:class:`~mufinder.app.workflow_controllers.base.WorkflowJobController`. There is
no ``pre_submit_check`` and no ``submission_groups`` override (a single group,
``normalize=str`` and the base-default ``is_ready``), and
:meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.build_result`
contributes a single boolean payload flag (``density_available``): no derived
structures and no document mutation.

Note on the den_fmt check: :meth:`UepController.validate_artifacts` fails a job
whose ``{seed}.den_fmt`` output is missing. The density file is not parsed on
completion, so this is a plain existence check rather than a carrier read.
:meth:`UepController.build_result` re-checks the file on the validated success
path, where it is guaranteed to exist, so the flag is ``True``. The den_fmt file
itself is **not** added to ``output_files`` here — the call site already passes
``_output_files_for_job(job)``, which includes it.

The controller also owns the UEP page's configuration: the source-structure
choice (:mod:`mufinder.services.uep_sources`), the page's form state, Configure
and row removal. And it owns the analysis window of each completed source job:
opening one or raising the one already open, keeping its layout and view
(:func:`~mufinder.services.uep_results.store_uep_analysis_state`) and its
search (:func:`~mufinder.services.uep_results.store_uep_batch_search`), and
flushing and closing the open windows when the project is saved or swapped or
the window closes. What it needs from the window arrives through
:class:`UepDeps`.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QDialog, QMessageBox, QWidget

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Structure
from mufinder.domain.text import plural
from mufinder.science.uep import UepError
from mufinder.services.definition_mappings import (
    convergence_mapping_job_priority,
    remove_labels_from_definition,
    sync_equivalent_mappings,
)
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.uep_results import (
    store_uep_analysis_state,
    store_uep_batch_search,
    uep_analysis_state,
    uep_batch_search,
)
from mufinder.services.uep_sources import (
    select_uep_source_structure,
    selected_uep_source_structure,
    store_uep_form_state,
    uep_source_structures,
)
from mufinder.workflow import StructureBridgeError
from mufinder.workflow.types import JobSpec

from mufinder.app.adapter_common import AdapterExpansion
from mufinder.app.uep_adapter import expand_uep_jobs
from mufinder.app.uep_types import UepConfig, UepRunState
from mufinder.app.uep_widget import UEPWidget
from mufinder.app.uep_wizard import UepAnalysisDialog
from mufinder.app.workflow_config_codecs import (
    deserialize_uep_config,
    serialize_uep_config,
    uep_row_signature,
)
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
)
from mufinder.app.workflow_parameters import structure_species_options
from mufinder.app.workflow_submission import effective_row_status


@dataclass(frozen=True)
class UepDeps:
    """What the UEP controller needs from the window.

    ``dialog_parent`` parents the message boxes and the analysis windows, and is
    searched for analysis windows when they close.
    ``refresh_workflow_defaults`` pushes the project defaults and the selected
    source structure to the UEP page. ``note_kpoint_adjustments`` keeps the rows
    whose project k-point grid was rescaled for the coming submission.
    ``refresh_journal_view`` rebuilds the Journal and
    ``refresh_generation_table`` the Generation page, whose UEP rows read the
    stored searches. ``shutting_down`` says whether the window is closing, when
    nothing is stored.
    """

    dialog_parent: Callable[[], QWidget]
    refresh_workflow_defaults: Callable[[], None]
    note_kpoint_adjustments: Callable[[AdapterExpansion], None]
    refresh_journal_view: Callable[[], None]
    refresh_generation_table: Callable[[], None]
    shutting_down: Callable[[], bool]


class UepController(WorkflowJobController):
    """Controller for the UEP source-job workflow."""

    calc_types = frozenset({"castep_uep_source"})
    stage = WorkflowStage.UEP
    active_key = "active_uep_definition_id"
    planned_calc_type = "castep_uep_source"
    no_ready_message = "Only configured UEP source rows that are ready, failed, stopped or cancelled can be submitted."

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: UepDeps,
        view: UEPWidget,
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view = view
        self._analysis_dialogs: dict[UUID, UepAnalysisDialog] = {}
        self._handling_form_state = False

    # --- definition maintenance ---

    def sync_mappings(self, definition: CalculationDefinition) -> None:
        """Copy job bindings into ``definition`` from equivalent UEP definitions.

        A row carries over when its signature is unchanged; candidates rank by
        :func:`~mufinder.services.definition_mappings.convergence_mapping_job_priority`.
        """

        document = self._context.document()
        if document is None:
            return
        sync_equivalent_mappings(
            document,
            self.definitions(),
            definition,
            deserialize=deserialize_uep_config,
            signatures=lambda config: {row.label: uep_row_signature(row) for row in config.rows},
            priority=convergence_mapping_job_priority,
        )

    def remove_labels_from_definition(
        self, definition: CalculationDefinition, labels: Collection[str]
    ) -> bool:
        document = self._context.document()
        if document is None:
            return False
        return remove_labels_from_definition(
            document,
            definition,
            labels,
            deserialize=deserialize_uep_config,
            serialize=serialize_uep_config,
            active_key=self.active_key,
        )

    def active_job_ids(self) -> set[UUID]:
        """Return the ids of the jobs the active definition's rows are bound to.

        Syncs the active definition's mappings first.
        """

        definition = self.active_definition()
        if definition is None:
            return set()
        self.sync_mappings(definition)
        mapping = definition.parameters.get("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            return set()
        job_ids: set[UUID] = set()
        for raw_job_id in mapping.values():
            try:
                job_ids.add(UUID(str(raw_job_id)))
            except ValueError:
                continue
        return job_ids

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready).
    # There is no pre_submit_check and submission is a single group.

    def normalize(self, value: object) -> str:
        return str(value)

    def build_states(self) -> dict[str, UepRunState]:
        states: dict[str, UepRunState] = {}
        document = self._context.document()
        if document is None:
            return states
        definition = self.active_definition()
        if definition is None:
            return states
        self.sync_mappings(definition)
        config = deserialize_uep_config(definition.parameters.get("configuration", {}))
        if config is None:
            return states

        for row in config.rows:
            job = self.job_for_key(definition, row.label)
            if job is None:
                states[row.label] = UepRunState(status="ready")
                continue
            run = self._context.run_for_job(job.id)
            result = (
                document.results.get(run.result_id)
                if run is not None and run.result_id is not None
                else None
            )
            data = result.data if result is not None and isinstance(result.data, dict) else {}
            states[row.label] = UepRunState(
                status=effective_row_status(job),
                total_energy=float(data["total_energy"])
                if isinstance(data.get("total_energy"), (int, float))
                else None,
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
                density_available=bool(data.get("density_available"))
                if data.get("density_available") is not None
                else None,
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
        selected: list[Job] = []
        seen: set[UUID] = set()
        for label in keys:
            job = self.job_for_key(definition, str(label))
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected

    def describe_rows(
        self, keys: list[object], *, also: Collection[UUID] = ()
    ) -> tuple[CalculationDefinition, Path, list[tuple[str, JobSpec | None, Job | None]]] | None:
        """Return the active definition, its output folder and each row's spec and job.

        Every selected row counts (``also`` adds nothing). The rows expand on
        the definition's source structure. See
        :meth:`~mufinder.app.workflow_controllers.base.WorkflowJobController.active_expansion`.
        """

        document = self._context.document()
        if document is None:
            return None

        def expand(definition: CalculationDefinition, config: object):
            source_structure = document.structures.get(definition.input_structure_ids[0])
            if source_structure is None:
                return None
            return expand_uep_jobs(
                host_structure=source_structure,
                config=config,
                project_cutoff_energy=document.project.recommended_cutoff_energy,
                project_kpoint_grid=document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=document.project.recommended_kpoint_lattice,
                project_xc_functional=document.project.recommended_xc_functional,
            )

        # The configure path already reported ``expansion.kpoint_adjustments``;
        # planning does not report them again.
        prepared = self.active_expansion(deserialize_uep_config, expand)
        if prepared is None:
            return None
        definition, output_root, _config, expansion = prepared
        spec_by_label = {str(spec.provenance.get("label", "")): spec for spec in expansion.jobs}
        rows = [
            (str(key), spec_by_label.get(str(key)), self.job_for_key(definition, str(key)))
            for key in keys
        ]
        return definition, output_root, rows

    # --- result contract ---

    def validate_artifacts(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> str | None:
        """Fail a UEP source job whose ``{seed}.den_fmt`` was not written.

        The formatted density is not parsed on completion, so it has no carrier
        field; this is a plain existence check in the job's output directory.
        """

        output_dir = self._context.job_output_directory(job)
        seed_name = self._context.job_seed_name(job)
        density_path = None if output_dir is None else output_dir / f"{seed_name}.den_fmt"
        if density_path is None or not density_path.exists():
            return f"Expected formatted density output was not found: {seed_name}.den_fmt"
        return None

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute the ``density_available`` flag for a completed UEP job.

        :meth:`validate_artifacts` has already failed the job when
        ``{seed}.den_fmt`` is missing, so on the validated success path this
        re-check finds the file present and the flag is ``True``. The den_fmt
        file is **not** contributed to ``output_files`` here: the call site
        already passes ``_output_files_for_job(job)``, which includes it.
        """

        output_dir = self._context.job_output_directory(job)
        seed_name = self._context.job_seed_name(job)
        density_available = (
            output_dir is not None and (output_dir / f"{seed_name}.den_fmt").exists()
        )
        return ControllerResult(payload={"density_available": density_available})

    # --- view contract ---

    def refresh_view(self) -> None:
        self._view.set_runtime_states(self.build_states())

    # --- the UEP page ---

    def selected_source_structure(self) -> Structure | None:
        """Return the structure the UEP page's rows are built on, or ``None``."""

        document = self._context.document()
        if document is None:
            return None
        return selected_uep_source_structure(document)

    def refresh_source_structures(self) -> None:
        """Offer the page the structures a UEP job can run on, with the one selected."""

        document = self._context.document()
        if document is None:
            self._view.set_source_structure_options(
                [], selected_id=None, summary="Create or open a project first."
            )
            self._view.set_parameter_species_options(())
            return

        structures = uep_source_structures(document)
        selected = selected_uep_source_structure(document)
        selected_id = str(selected.id) if selected is not None else None
        if not structures:
            self._view.set_source_structure_options(
                [],
                selected_id=None,
                summary="Load a host structure before configuring the UEP workflow.",
            )
            self._view.set_parameter_species_options(())
            return

        summary = (
            f"Using source structure: {selected.label}"
            if selected is not None
            else "Choose the host structure or a completed pristine-relaxed structure as the UEP source."
        )
        self._view.set_source_structure_options(
            [{"id": str(structure.id), "label": structure.label} for structure in structures],
            selected_id=selected_id,
            summary=summary,
        )
        self._view.set_parameter_species_options(structure_species_options(selected))

    def select_source_structure(self, payload: object) -> None:
        """Select the source structure the page asked for; an empty id clears the choice."""

        document = self._context.document()
        if document is None:
            return
        select_uep_source_structure(document, str(payload).strip())
        self.refresh_source_structures()
        self._deps.refresh_workflow_defaults()

    def store_form_state(self, state: dict) -> None:
        """Keep the page's form state and push the project defaults back to the page.

        A source choice the page's picker cannot show yet leaves the stored one
        in place (:func:`~mufinder.services.uep_sources.store_uep_form_state`).
        """

        document = self._context.document()
        if document is None or self._deps.shutting_down():
            return
        store_uep_form_state(document, state)
        if self._handling_form_state:
            return
        self._handling_form_state = True
        try:
            self._deps.refresh_workflow_defaults()
        finally:
            self._handling_form_state = False

    def configure(self, config: UepConfig) -> None:
        """Register ``config`` as the active UEP definition on the selected source structure.

        A ``config`` the active definition already stores for that structure
        registers nothing (:meth:`store_configuration`).
        """

        parent = self._deps.dialog_parent()
        document = self._context.document()
        if document is None:
            QMessageBox.information(parent, "No project", "Create or open a project first.")
            return
        source_structure = selected_uep_source_structure(document)
        if source_structure is None:
            QMessageBox.warning(
                parent,
                "No source structure",
                "Load a host structure or select a pristine-relaxed structure first.",
            )
            return
        try:
            expansion = expand_uep_jobs(
                host_structure=source_structure,
                config=config,
                project_cutoff_energy=document.project.recommended_cutoff_energy,
                project_kpoint_grid=document.project.recommended_kpoint_grid,
                project_kpoint_reference_lattice=document.project.recommended_kpoint_lattice,
                project_xc_functional=document.project.recommended_xc_functional,
            )
        except (ValueError, StructureBridgeError) as exc:
            QMessageBox.warning(parent, "UEP setup invalid", str(exc))
            return
        self._deps.note_kpoint_adjustments(expansion)

        output_root = document.project_root / "files" / "uep"
        self.store_configuration(
            serialize_uep_config(config),
            input_structure_id=source_structure.id,
            output_directory=output_root,
            register=self._context.project_service.register_uep_definition,
        )
        self.refresh_view()
        self._deps.refresh_journal_view()
        self._context.status_message(
            f"Configured {plural(len(config.rows), 'UEP source job')}; input files will be generated on submission"
        )

    def remove_labelled_rows(self, labels: Collection[str]) -> None:
        """Drop the rows ``labels`` name and refresh the page."""

        if self._context.document() is None or not labels:
            return
        self.remove_rows(labels)
        self.refresh_view()

    # --- the analysis windows ---

    @property
    def analysis_dialogs(self) -> dict[UUID, UepAnalysisDialog]:
        """The open analysis windows by source job id (the live mapping)."""

        return self._analysis_dialogs

    def open_analysis(self, label: object) -> None:
        """Open the analysis window of row ``label``'s completed source job.

        Raises the window already open for that job instead of opening another.
        """

        document = self._context.document()
        if document is None:
            return
        row_label = str(label).strip()
        if not row_label:
            return

        parent = self._deps.dialog_parent()
        definition = self.active_definition()
        if definition is None:
            QMessageBox.information(
                parent, "No UEP workflow", "Configure and complete a UEP source job first."
            )
            return

        job = self.job_for_key(definition, row_label)
        if job is None or job.status != JobStatus.COMPLETED:
            QMessageBox.information(
                parent,
                "UEP source incomplete",
                "Only completed UEP source jobs can be analysed.",
            )
            return

        output_dir = self._context.job_output_directory(job)
        if output_dir is None:
            QMessageBox.warning(
                parent, "UEP files unavailable", "Could not locate the UEP job output directory."
            )
            return

        seed_name = self._context.job_seed_name(job)
        den_fmt_path = output_dir / f"{seed_name}.den_fmt"
        castep_path = output_dir / f"{seed_name}.castep"
        cell_path = output_dir / f"{seed_name}.cell"
        if not den_fmt_path.exists() or not castep_path.exists():
            QMessageBox.warning(
                parent,
                "UEP files unavailable",
                "UEP analysis requires both `.den_fmt` and `.castep` files from the completed source job.",
            )
            return

        source_structure_id = (
            definition.input_structure_ids[0] if definition.input_structure_ids else None
        )
        if source_structure_id is None:
            QMessageBox.warning(
                parent, "UEP source unavailable", "The source structure reference is missing."
            )
            return

        existing_dialog = self._analysis_dialogs.get(job.id)
        if existing_dialog is not None:
            try:
                is_visible = getattr(existing_dialog, "isVisible", None)
                if callable(is_visible) and not is_visible():
                    self._drop_analysis_dialog(job.id)
                else:
                    existing_dialog.show()
                    existing_dialog.raise_()
                    existing_dialog.activateWindow()
                    return
            except RuntimeError:
                self._drop_analysis_dialog(job.id)

        try:
            initial_state = uep_analysis_state(document, job.id)
            search = uep_batch_search(document, job.id)
            dialog = UepAnalysisDialog(
                source_job_label=job.label,
                source_structure_id=source_structure_id,
                source_structure=document.structures.get(source_structure_id),
                den_fmt_path=den_fmt_path,
                castep_path=castep_path,
                cell_path=cell_path if cell_path.exists() else None,
                initial_state=initial_state,
                initial_search=None if search is None else search.payload(),
                parent=parent,
            )
        except (UepError, OSError, ValueError) as exc:
            QMessageBox.warning(parent, "UEP analysis unavailable", str(exc))
            return

        dialog.state_changed.connect(
            lambda state, job_id=job.id: self.analysis_state_changed(job_id, state)
        )
        dialog.search_changed.connect(
            lambda search, job_id=job.id: self.search_changed(job_id, search)
        )
        dialog.setModal(False)
        dialog.destroyed.connect(lambda *_args, job_id=job.id: self._drop_analysis_dialog(job_id))
        self._analysis_dialogs[job.id] = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def analysis_state_changed(self, job_id: UUID, state: dict) -> None:
        """Keep an analysis window's layout and view, then raise the window again."""

        document = self._context.document()
        if document is None or self._deps.shutting_down():
            return
        if not store_uep_analysis_state(document, job_id, state):
            return
        self._queue_raise_visible_analysis_dialog(job_id)

    def search_changed(self, job_id: UUID, search: object) -> None:
        """Store a finished, regrouped or discarded search and save it with the next autosave."""

        document = self._context.document()
        if document is None or self._deps.shutting_down():
            return
        job = document.jobs.get(job_id)
        if job is None:
            return
        if not store_uep_batch_search(document, job, search if isinstance(search, dict) else None):
            return
        # Refreshed here too: a Save Project in progress takes the pending autosave.
        self._deps.refresh_journal_view()
        QTimer.singleShot(0, self._deps.refresh_generation_table)
        self._context.request_dashboard_flush(
            failure_message="The UEP search was kept, but autosave failed."
        )
        self._queue_raise_visible_analysis_dialog(job_id)

    def flush_analysis_dialogs(self) -> None:
        """Bring the document up to date with a regroup or a view change that has not settled."""

        for dialog in tuple(self._analysis_dialogs.values()):
            try:
                dialog.flush_pending_state()
            except RuntimeError:
                continue

    def close_analysis_dialogs(self) -> None:
        """Close every analysis window, those tracked and those parented to the window."""

        dialogs: list[QDialog] = []
        seen: set[int] = set()
        for dialog in tuple(self._analysis_dialogs.values()):
            dialog_id = id(dialog)
            if dialog_id in seen:
                continue
            seen.add(dialog_id)
            dialogs.append(dialog)
        for dialog in self._deps.dialog_parent().findChildren(UepAnalysisDialog):
            dialog_id = id(dialog)
            if dialog_id in seen:
                continue
            seen.add(dialog_id)
            dialogs.append(dialog)

        for dialog in dialogs:
            try:
                dialog.close()
            except RuntimeError:
                continue
        self._analysis_dialogs.clear()

    def _drop_analysis_dialog(self, job_id: UUID) -> None:
        self._analysis_dialogs.pop(job_id, None)

    @staticmethod
    def _raise_existing_analysis_dialog(dialog: object) -> None:
        try:
            is_visible = getattr(dialog, "isVisible", None)
            if callable(is_visible) and not is_visible():
                return
            show = getattr(dialog, "show", None)
            if callable(show):
                show()
            raise_window = getattr(dialog, "raise_", None)
            if callable(raise_window):
                raise_window()
            activate = getattr(dialog, "activateWindow", None)
            if callable(activate):
                activate()
        except RuntimeError:
            return

    def _queue_raise_visible_analysis_dialog(self, job_id: UUID) -> None:
        dialog = self._analysis_dialogs.get(job_id)
        if dialog is None:
            return
        QTimer.singleShot(0, lambda dialog=dialog: self._raise_existing_analysis_dialog(dialog))


__all__ = ["UepController", "UepDeps"]
