"""Running a completed source row again with other parameters.

:class:`CompletedRowEditsController` is the shared home of the route a
source-row workflow (one job per relaxed muon site) takes when the user changes
a completed row's parameters:

- **Configure stays available.** A completed row is configurable while its
  job's ``.cell`` and ``.param`` are there to compare it with
  (:meth:`CompletedRowEditsController.row_is_configurable`).
- **Changing it asks first.** :meth:`CompletedRowEditsController.configure_sources`
  runs the workflow's dialog, and one ``confirm_destructive`` question asks
  before a change that would make a completed row differ from what its job ran
  with (:meth:`CompletedRowEditsController.confirm_configuration_change`).
- **Nothing new is stored.** Only the row's stored configuration changes. A row
  reads ``edited`` while its stored parameters differ from its job's inputs
  (:meth:`CompletedRowEditsController.parameters_changed_since_run`), and
  setting them back makes it ``completed`` again.
- **Running it replaces the result.** :meth:`CompletedRowEditsController.completed_jobs_to_run_again`
  names the edited rows' completed jobs, which the shared submit cycle starts
  again; their results go once the submission is accepted. A run of only
  unchanged completed rows stops with one notice
  (:meth:`CompletedRowEditsController.nothing_to_run`).

A workflow subclass supplies the hooks: its row's job, the inputs a row would
write now, the keywords it fixes, and the words its messages use.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from uuid import UUID

from mufinder.castep_io.input_diff import JobInputTexts, diff_job_inputs, read_job_inputs
from mufinder.castep_io.schema import KeywordSpec, canonical_keyword_name
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationDefinition, Job
from mufinder.workflow.types import JobSpec
from mufinder.domain.text import plural
from mufinder.services.muon_sites_summary import downstream_site_scope
from mufinder.services.source_candidates import (
    QuantumSourceCandidate,
    charge_state_label_for_job,
)

from mufinder.app import workflow_table_utils
from mufinder.app.workflow_config_codecs import (
    batch_row_payload,
    num_farms_from_extras,
    row_settings_payload,
    store_workflow_row_configuration,
    workflow_row_configuration,
)
from mufinder.app.workflow_controllers.base import (
    WorkflowControllerContext,
    WorkflowJobController,
)
from mufinder.app.workflow_dialogs import SourceConfigureRequest, SourceConfigureResult
from mufinder.app.workflow_parameters import workflow_basic_parameter_names
from mufinder.app.workflow_submission import (
    EDITED_ROW_STATUS,
    WorkflowSubmissionPlan,
    workflow_row_state_is_configurable,
    workflow_row_state_is_ready,
)


class CompletedRowEditsController(WorkflowJobController):
    """A source-row workflow whose completed rows can run again with other parameters."""

    #: Keywords the workflow writes on every job itself; a difference in them is never the user's.
    fixed_keyword_names: tuple[str, ...] = ()
    #: Keywords a row's inputs can lose without the user removing them: the
    #: settings inherited from the source relaxation (gone when its files are),
    #: the shared rows the dialog always stores, and the farm count, which
    #: changes how the job runs but not its result.
    not_user_removed_names: frozenset[str] = frozenset(
        {*workflow_basic_parameter_names(), "xc_functional", "spin_polarized", "spin", "num_farms"}
    )
    #: The pinned-site scope whose labels name the workflow's rows.
    site_scope: str = ""
    #: What running a changed completed row again replaces, singular and plural.
    result_noun: str = "result"
    result_noun_plural: str = "results"
    #: The question's detail: what running again deletes, and when.
    replaced_result_detail: str = ""

    def __init__(self, context: WorkflowControllerContext) -> None:
        super().__init__(context)
        #: :meth:`parameters_changed_since_run`'s answers, by what they depend on.
        self._changed_since_run: dict[tuple[object, ...], bool | None] = {}

    # --- workflow hooks ---

    def describe_source_rows(
        self,
        keys: list[object],
        *,
        also: Collection[UUID],
        job_for_row: Callable[[CalculationDefinition | None, str], Job | None],
        job_config: Callable[[CalculationDefinition | None, str], object | None],
        expand: Callable[[list[object]], Sequence[JobSpec]],
    ) -> tuple[CalculationDefinition | None, Path, list[tuple[str, JobSpec, Job | None]]] | None:
        """Return the active definition (``None`` before the first), the output folder and each row.

        Each row is ``(key, spec, job_for_row(...))`` for a selected row that is
        ready, or whose job ``also`` names (an ended or edited row running
        again), and whose ``job_config`` builds. Changes nothing: the definition
        gains the selected sources only once a submission is accepted. ``None``
        when no selected row is an eligible source. A ``ValueError`` or
        ``StructureBridgeError`` from ``expand`` reaches the caller.
        """

        if self._context.document() is None:
            return None
        str_keys = [str(key) for key in keys]
        if not any(self.candidate_by_key(key) is not None for key in str_keys):
            return None
        definition = self.active_definition()
        output_root = self.planned_output_root(definition)
        if output_root is None:
            return None
        states = self.build_states()
        configs: list[object] = []
        config_keys: list[str] = []
        for key in str_keys:
            existing = job_for_row(definition, key)
            if not workflow_row_state_is_ready(states, key) and (
                existing is None or existing.id not in also
            ):
                continue
            config = job_config(definition, key)
            if config is None:
                continue
            configs.append(config)
            config_keys.append(key)
        if not configs:
            return definition, output_root, []
        specs = expand(configs)
        return (
            definition,
            output_root,
            [
                (key, spec, job_for_row(definition, key))
                for key, spec in zip(config_keys, specs, strict=True)
            ],
        )

    def candidates(self) -> list[QuantumSourceCandidate]:
        """Return the eligible relaxed muon sources, in table order."""

        raise NotImplementedError

    def candidate_by_key(self, key: str) -> QuantumSourceCandidate | None:
        """Return the eligible source row ``key`` names, if any."""

        raise NotImplementedError

    def result_job_for_key(self, definition: CalculationDefinition | None, key: str) -> Job | None:
        """Return the job whose result running row ``key`` again replaces."""

        raise NotImplementedError

    def row_inputs(
        self, definition: CalculationDefinition | None, key: str
    ) -> JobInputTexts | None:
        """Render the ``.cell`` and ``.param`` row ``key``'s job would be written with now."""

        raise NotImplementedError

    def parameter_specs(self) -> list[KeywordSpec]:
        """Return the Configure dialog's parameter surface."""

        raise NotImplementedError

    def runtime_configuration(
        self, definition: CalculationDefinition | None, key: str
    ) -> dict[str, object]:
        """Return row ``key``'s effective parameter values."""

        raise NotImplementedError

    def site_title(self, key: str) -> str:
        """Return how the row ``key`` names its site and charge state, ``""`` for no such row.

        The site's :meth:`~mufinder.services.muon_sites_summary.DownstreamSiteScope.site_name`
        in :attr:`site_scope`.
        """

        candidate = self.candidate_by_key(key)
        if candidate is None:
            return ""
        document = self._context.document()
        scope = downstream_site_scope(document, self.site_scope, self.candidates())
        label = scope.site_name(candidate)
        charge = charge_state_label_for_job(candidate.job)
        return f"{label}, {charge}" if charge else label

    def dialog_parent(self) -> object:
        """Return the window the workflow's dialogs and questions open over."""

        raise NotImplementedError

    def configure_request(self, keys: Sequence[str]) -> SourceConfigureRequest | None:
        """Return what the Configure dialog opens with for the rows ``keys`` name."""

        raise NotImplementedError

    def run_configure_dialog(self, request: SourceConfigureRequest) -> SourceConfigureResult | None:
        """Run the workflow's Configure dialog on ``request``; ``None`` when nothing is accepted."""

        raise NotImplementedError

    def apply_configuration(
        self,
        keys: Sequence[str],
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> None:
        """Store an accepted Configure dialog on every row ``keys`` name."""

        raise NotImplementedError

    def row_payload_extras(
        self, definition: CalculationDefinition, key: str, extras: Mapping[str, object]
    ) -> dict[str, object]:
        """Return the workflow values row ``key`` stores beside an accepted dialog's parameters.

        Default: the dialog's ``extras`` as they are.
        """

        return dict(extras)

    # --- Configure ---

    def configure_sources(self, keys: Sequence[str]) -> None:
        """Run the Configure dialog on the rows ``keys`` name and store what it accepts.

        When the accepted parameters would change a completed row from what its
        job ran with, one question asks first (:meth:`confirm_configuration_change`);
        ``Cancel`` keeps every row as it was. Accepting it changes only the rows'
        stored configuration.
        """

        request = self.configure_request([str(key) for key in keys])
        if request is None:
            return
        result = self.run_configure_dialog(request)
        if result is None:
            return
        if not self.confirm_configuration_change(request.keys, result, result.extras):
            return
        self.apply_configuration(request.keys, result, result.extras)

    def confirm_configuration_change(
        self,
        keys: Sequence[str],
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> bool:
        """Return whether storing ``result`` may go ahead, asking when it replaces results.

        Asks once, naming how many results running the rows again replaces
        (:meth:`rows_whose_result_a_change_replaces`); nothing is asked when it
        replaces none.
        """

        replaced = self.rows_whose_result_a_change_replaces(keys, result, extras)
        return not replaced or self._confirm_replacing_results(replaced)

    def rows_whose_result_a_change_replaces(
        self,
        keys: Sequence[str],
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> list[str]:
        """Return the completed rows among ``keys`` that storing ``result`` would make edited.

        A row already edited is left out (replacing its result was confirmed
        when it was first changed), and so is a completed row the change leaves
        running what its job ran (:meth:`parameters_changed_since_run`). A row
        whose new settings cannot be compared is counted, so the question is
        asked rather than skipped.
        """

        definition = self.active_definition()
        if definition is None:
            return []
        states = self.build_states()
        replaced: list[str] = []
        for key in keys:
            if row_status(states, key) != "completed":
                continue
            job = self.result_job_for_key(definition, key)
            if job is None or not self._configuration_changes_row(key, result, extras):
                continue
            preview = definition_with_row_payload(
                definition, key, self._row_payload(definition, key, result, extras)
            )
            if self.parameters_changed_since_run(preview, key, job) is not False:
                replaced.append(key)
        return replaced

    def rows_unchanged_by(
        self,
        keys: Sequence[str],
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> set[str]:
        """Return the completed rows among ``keys`` that storing ``result`` leaves as they are."""

        states = self.build_states()
        return {
            key
            for key in keys
            if row_status(states, key) == "completed"
            and not self._configuration_changes_row(key, result, extras)
        }

    def _confirm_replacing_results(self, keys: Sequence[str]) -> bool:
        """Ask whether to change parameters that replace completed results; ``Cancel`` is the default."""

        if len(keys) == 1:
            text = (
                f"Change the parameters of '{self.site_title(keys[0]) or keys[0]}'? Running "
                f"the row again replaces its {self.result_noun}."
            )
        else:
            text = (
                f"Change the parameters of {plural(len(keys), 'completed site')}? Running "
                f"the rows again replaces their {self.result_noun_plural}."
            )
        return workflow_table_utils.confirm_destructive(
            self.dialog_parent(),
            "Change parameters",
            text,
            self.replaced_result_detail,
            "Change parameters",
        )

    def _row_payload(
        self,
        definition: CalculationDefinition,
        key: str,
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> dict[str, object] | None:
        """Return the ``row_settings`` entry an accepted dialog stores for row ``key``.

        For a row whose job has completed, a value the dialog filled in by
        itself (left as the dialog opened it, and neither stored, inherited nor
        a default of the row) is left out: it was not what the job ran with, and
        the user did not set it, so storing it would change the row's inputs
        behind an unchanged dialog.
        """

        values = result.values
        job = self.result_job_for_key(definition, key)
        if job is not None and job.status == JobStatus.COMPLETED:
            own = self.runtime_configuration(definition, key)
            values = {
                name: value
                for name, value in values.items()
                if name in own
                or name in result.advanced_names
                or name not in result.opening_values
                or result.opening_values[name] != value
            }
        payload = row_settings_payload(
            values,
            result.advanced_names,
            self.row_payload_extras(definition, key, extras),
        )
        return batch_row_payload(definition, key, payload, result.untouched_mixed)

    def _configuration_changes_row(
        self, key: str, result: SourceConfigureResult, extras: Mapping[str, object]
    ) -> bool:
        """Return whether storing ``result`` would change row ``key``'s parameters or farm count."""

        definition = self.active_definition()
        if definition is None:
            return True
        preview = definition_with_row_payload(
            definition, key, self._row_payload(definition, key, result, extras)
        )
        before = workflow_row_configuration(definition, key)
        after = workflow_row_configuration(preview, key)
        return (
            self.runtime_configuration(definition, key) != self.runtime_configuration(preview, key)
            or set(before[1]) != set(after[1])
            or num_farms_from_extras(before[2]) != num_farms_from_extras(after[2])
        )

    def row_is_configurable(
        self,
        states: Mapping[str, object],
        definition: CalculationDefinition | None,
        key: str,
    ) -> bool:
        """Return whether row ``key``'s parameters can change.

        As on every workflow table (no job, a ``defined`` or ended one, or an
        ``edited`` row), and also a completed row whose job's ``.cell`` and
        ``.param`` are there to compare the row with
        (:meth:`parameters_changed_since_run`).
        """

        if workflow_row_state_is_configurable(states, key):
            return True
        if row_status(states, key) != "completed":
            return False
        job = self.result_job_for_key(definition, key)
        return job is not None and self._job_input_paths(job) is not None

    # --- comparing a row with its run ---

    def completed_row_is_edited(
        self, definition: CalculationDefinition | None, key: str, job: Job
    ) -> bool:
        """Return whether completed ``job``'s row ``key`` reads ``edited``."""

        return (
            job.status == JobStatus.COMPLETED
            and self.parameters_changed_since_run(definition, key, job) is True
        )

    def parameters_changed_since_run(
        self, definition: CalculationDefinition | None, key: str, job: Job
    ) -> bool | None:
        """Return whether row ``key``'s stored parameters differ from what ``job`` ran with.

        The job's own ``.cell`` and ``.param`` are compared, keyword by keyword
        and leaving the structure out (:func:`~mufinder.castep_io.input_diff.diff_job_inputs`),
        with what the row would write now (:meth:`row_inputs`). Only what the
        user sets counts:

        - a keyword the row stores that is written differently, added or
          dropped;
        - a keyword the run had that the row no longer writes at all and that
          the Configure dialog offers (a parameter the user removed).

        A keyword the workflow fixes (:attr:`fixed_keyword_names`), one the row
        inherits or always stores, and the farm count (:attr:`not_user_removed_names`)
        never count by themselves, and neither does a keyword the row does not
        store whose value changed or that only the new inputs have: that is a
        changed default or a value derived from one. So a changed default, a
        job an older version wrote, or source relaxation files deleted since,
        do not make a row edited. A row that stores nothing has not changed.
        ``None`` when the job's inputs cannot be read or the row's settings do
        not build. The answer is kept until the job's input files, the row's
        settings or its source relaxation change.
        """

        params, advanced, _extras = workflow_row_configuration(definition, key)
        files = self._job_input_paths(job)
        if files is None:
            return None
        candidate = self.candidate_by_key(key)
        # Every page refresh asks this of every completed row, so the answer is
        # kept until the job's inputs, the row's settings or its source change.
        cache_key = (
            job.id,
            key,
            *(_file_stamp(path) for path in files),
            _settings_fingerprint(definition, key),
            None if candidate is None else candidate.run.id,
        )
        if cache_key in self._changed_since_run:
            return self._changed_since_run[cache_key]
        answer = self._compare_with_run(definition, key, files, params, advanced)
        self._changed_since_run[cache_key] = answer
        return answer

    def _compare_with_run(
        self,
        definition: CalculationDefinition | None,
        key: str,
        files: tuple[Path, Path],
        params: Mapping[str, object],
        advanced: Sequence[str],
    ) -> bool | None:
        """Compare row ``key``'s inputs with its job's; see :meth:`parameters_changed_since_run`."""

        old = read_job_inputs(files[0].parent, files[0].stem)
        if old is None:
            return None
        if not params and not advanced:
            return False
        new = self.row_inputs(definition, key)
        if new is None:
            return None
        stored = {
            canonical_keyword_name(file, str(name))
            for name in (*params, *advanced)
            for file in ("cell", "param")
        }
        removable: set[str] | None = None
        for change in diff_job_inputs(old, new):
            name = change.keyword.lower()
            if name in self.fixed_keyword_names:
                continue
            if name in stored:
                return True
            if change.change != "Removed" or name in self.not_user_removed_names:
                continue
            # Only a parameter the dialog offers can have been the user's: a
            # keyword an older version wrote by itself is not.
            if removable is None:
                removable = {spec.name for spec in self.parameter_specs()}
            if name in removable:
                return True
        return False

    def _job_input_paths(self, job: Job) -> tuple[Path, Path] | None:
        """Return ``job``'s ``.cell`` and ``.param`` paths, or ``None`` when either is missing."""

        directory = self._context.job_output_directory(job)
        if directory is None:
            return None
        seed = self._context.job_seed_name(job)
        paths = (directory / f"{seed}.cell", directory / f"{seed}.param")
        return paths if all(path.is_file() for path in paths) else None

    # --- submission ---

    def nothing_to_run(self, keys: Sequence[object]) -> bool:
        """Return whether every listed row in ``keys`` is ``completed``, saying so when it is.

        Their parameters are unchanged since their run, so there is nothing to
        run: one notice says so and how to run such a row again, instead of the
        backends' generic "No eligible jobs".
        """

        listed = [str(key) for key in keys if self.candidate_by_key(str(key)) is not None]
        states = self.build_states()
        if not listed or not all(row_status(states, key) == "completed" for key in listed):
            return False
        self._context.show_info(*self._only_completed_notice(listed))
        return True

    def _only_completed_notice(self, keys: Sequence[str]) -> tuple[str, str]:
        """Return the notice for a run whose rows have all completed with their parameters unchanged."""

        if len(keys) == 1:
            subject = f"'{self.site_title(keys[0]) or keys[0]}' has"
            pronoun, owner = "it", "its"
        else:
            subject = f"These {len(keys)} sites have"
            pronoun, owner = "one", "its"
        return (
            "Nothing to run",
            f"{subject} completed. To run {pronoun} again, change one of {owner} parameters "
            f"with 'Configure…' first; running it then replaces its {self.result_noun}.",
        )

    def completed_jobs_to_run_again(self, plan: WorkflowSubmissionPlan) -> list[Job]:
        """Return the completed jobs of the plan's ``edited`` rows: running them replaces their results."""

        states = self.build_states()
        definition = self.active_definition()
        jobs: list[Job] = []
        for key in plan.ready_values:
            if row_status(states, str(key)) != EDITED_ROW_STATUS:
                continue
            job = self.result_job_for_key(definition, str(key))
            if job is not None and job.status == JobStatus.COMPLETED:
                jobs.append(job)
        return jobs


def row_status(states: Mapping[str, object], key: str) -> str:
    """Return the status row ``key`` reads in ``states``; ``ready`` without one."""

    return str(getattr(states.get(key), "status", "ready"))


def definition_with_row_payload(
    definition: CalculationDefinition, key: str, payload: Mapping[str, object] | None
) -> CalculationDefinition:
    """Return a copy of ``definition`` with row ``key``'s settings replaced, for a preview.

    ``definition`` itself is left as it is; ``None`` drops the row's settings.
    """

    config = definition.parameters.get("configuration", {})
    preview_config = dict(config) if isinstance(config, dict) else {}
    rows = preview_config.get("row_settings", {})
    preview_config["row_settings"] = dict(rows) if isinstance(rows, dict) else {}
    preview = replace(
        definition, parameters={**definition.parameters, "configuration": preview_config}
    )
    store_workflow_row_configuration(preview, key, None if payload is None else dict(payload))
    return preview


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """Return ``path``'s modification time and size, ``None`` when it cannot be read."""

    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_mtime_ns, info.st_size


def _settings_fingerprint(definition: CalculationDefinition | None, key: str) -> str:
    """Return row ``key``'s stored settings as text that changes whenever they do."""

    params, advanced, extras = workflow_row_configuration(definition, key)
    return repr(
        (sorted(params.items(), key=str), sorted(advanced), sorted(extras.items(), key=str))
    )


__all__ = [
    "CompletedRowEditsController",
    "definition_with_row_payload",
    "row_status",
]
