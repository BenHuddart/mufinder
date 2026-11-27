"""A continued relaxation keeps the inputs its run used.

Through the real ``Continue relaxation…`` dialog and through ``Run`` or the Job
Dashboard's ``Resubmit…`` answered ``Continue``: an untouched continuation
writes the job's own ``.cell`` and ``.param`` again, whatever the project's
defaults are now; only an edit in the dialog, or a setting the row itself
stores, changes what is written.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import mufinder.app.jobs_controller as jobs_controller_module
import mufinder.app.main_window as main_window_module
import mufinder.app.workflow_table_utils as workflow_table_utils
from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
)
from mufinder.app.workflow_config_codecs import (
    deserialize_muon_relaxation_config,
    serialize_muon_relaxation_config,
    serialize_pristine_relaxation_config,
)
from mufinder.castep_io.input_diff import JobInputTexts, diff_job_inputs, read_job_inputs
from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.charge_states import muon_relaxation_calc_type
from mufinder.domain.enums import EndReason, JobStatus, WorkflowStage
from mufinder.domain.models import Job, MuonSite, Structure
from mufinder.services.project_service import ProjectService
from tests.support.main_window import fake_local_submission

_SUPERCELL_2X2X2 = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
_CHARGE_PREFIX = {"positive_muon": "positivemuon", "muonium": "muonium"}


@dataclass
class _Ran:
    """One relaxation row whose job ran and ended with a checkpoint."""

    window: MainWindow
    service: ProjectService
    doc: object
    workflow: str
    key: str
    job: Job
    job_dir: Path
    seed: str

    def inputs(self) -> JobInputTexts:
        inputs = read_job_inputs(self.job_dir, self.seed)
        assert inputs is not None
        return inputs

    def stored(self) -> tuple[dict[str, object], tuple[str, ...]]:
        settings = self.controller().row_settings(self.key)
        assert settings is not None
        return settings

    def controller(self):
        if self.workflow == "pristine":
            return self.window._pristine_relaxation_controller
        return self.window._muon_relaxation_controller


def _host() -> Structure:
    return Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _adopt_defaults(window: MainWindow, service: ProjectService, doc, cutoff: float, grid) -> None:
    host = doc.structures[doc.project.host_structure_id]
    service.set_recommended_cutoff(doc, cutoff, cell_id=str(host.id))
    service.set_recommended_kpoint_grid(
        doc, grid, lattice_vectors=host.lattice_vectors, cell_id=str(host.id)
    )
    window._refresh_ui_after_project_change()


def _end_with_checkpoint(service: ProjectService, job: Job, job_dir: Path, seed: str) -> None:
    service.submit_job(job)
    service.end_job(job, EndReason.WALLTIME, "walltime reached")
    (job_dir / f"{seed}.castep").write_text("previous run\n", encoding="utf-8")
    (job_dir / f"{seed}.check").write_bytes(b"checkpoint")


def _run_rows(
    tmp_path: Path,
    workflow: str,
    *,
    parameter_values: dict[str, object] | None = None,
    added_advanced: tuple[str, ...] = (),
    defaults: tuple[float, tuple[int, int, int]] | None = None,
) -> dict[str, _Ran]:
    """Build a project, write each row's inputs as its first run did, and end the run.

    ``workflow`` is ``pristine`` (one row) or ``muon`` (one site as a positive
    muon and as muonium, in a 2x2x2 supercell). ``defaults`` adopts a cutoff and
    a k-point grid on the project before the run; without it nothing is adopted.
    """

    values = dict(parameter_values or {})
    window = MainWindow()
    service = ProjectService()
    doc = service.create_project(tmp_path / f"{workflow}.mufinder", "Continue")
    host = _host()
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc
    if defaults is not None:
        _adopt_defaults(window, service, doc, *defaults)

    ran: dict[str, _Ran] = {}
    if workflow == "pristine":
        row = PristineRelaxationRow(
            label="row_1", parameter_values=values, added_advanced=added_advanced
        )
        output_root = doc.project_root / "files" / "pristine_relaxation"
        config = PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        definition = service.register_pristine_relaxation_definition(
            doc,
            input_structure_id=host.id,
            parameters=serialize_pristine_relaxation_config(config),
            output_directory=output_root,
        )
        job = service.add_job(
            doc,
            label="pristine_relax_row_1",
            workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
            calc_type="castep_pristine_relaxation",
        )
        definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
        doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
        doc.ui_state["pristine_relaxation_form"] = {
            "name_prefix": "pristine_relax",
            "rows": [
                {
                    "label": "row_1",
                    "fix_all_cell": False,
                    "params": serialize_pristine_relaxation_config(config)["rows"][0][
                        "parameter_values"
                    ],
                    "added_advanced": list(added_advanced),
                }
            ],
        }
        window._refresh_ui_after_project_change()
        assert window._pristine_relaxation_controller.materialize(["row_1"]) == [job]
        job_dir = output_root / job.label
        _end_with_checkpoint(service, job, job_dir, job.label)
        ran["row_1"] = _Ran(window, service, doc, workflow, "row_1", job, job_dir, job.label)
    else:
        site = MuonSite(
            label="Site 1",
            initial_fractional_position=[0.1, 0.2, 0.3],
            status="staged",
            source_structure_id=host.id,
        )
        doc.sites[site.id] = site
        rows = tuple(
            MuonRelaxationRow(
                site_id=site.id,
                label="Site 1",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state=charge_state,
                supercell_matrix=_SUPERCELL_2X2X2,
                # A muonium row always stores its spin polarisation, as the page does.
                parameter_values=(
                    {**values, "spin_polarized": True}
                    if charge_state == "muonium"
                    else dict(values)
                ),
                added_advanced=added_advanced,
            )
            for charge_state in ("positive_muon", "muonium")
        )
        output_root = doc.project_root / "files" / "muon_relaxation"
        config = MuonRelaxationConfig(rows=rows)
        definition = service.register_muon_relaxation_definition(
            doc,
            input_structure_id=host.id,
            parameters=serialize_muon_relaxation_config(config),
            output_directory=output_root,
        )
        jobs = {}
        for row in rows:
            key = f"{_CHARGE_PREFIX[row.charge_state]}:Site 1"
            jobs[key] = service.add_job(
                doc,
                label="Site 1",
                workflow_stage=WorkflowStage.MUON_RELAXATION,
                calc_type=muon_relaxation_calc_type(row.charge_state),
            )
        definition.parameters["job_ids_by_label"] = {key: str(job.id) for key, job in jobs.items()}
        doc.ui_state[window._ACTIVE_MUON_RELAXATION_DEFINITION_KEY] = str(definition.id)
        doc.ui_state["muon_relaxation_form"] = serialize_muon_relaxation_config(config)
        window._refresh_ui_after_project_change()
        materialized = window._muon_relaxation_controller.materialize(list(jobs))
        assert {job.id for job in materialized} == {job.id for job in jobs.values()}
        for key, job in jobs.items():
            seed = definition.parameters["job_seed_names_by_label"][key]
            job_dir = output_root / key.split(":")[0] / seed
            _end_with_checkpoint(service, job, job_dir, seed)
            ran[key] = _Ran(window, service, doc, workflow, key, job, job_dir, seed)
    window._refresh_ui_after_project_change()
    return ran


def _continue_through_dialog(
    monkeypatch: pytest.MonkeyPatch,
    ran: _Ran,
    edit: Callable[[object], None] = lambda _dialog: None,
    *,
    accept: bool = True,
) -> dict[str, object]:
    """Open the real ``Continue relaxation…`` dialog, apply ``edit``, and accept it.

    Returns the table's rows and what the dialog showed. Submission is recorded,
    not run, so the files on disk are exactly what continuing wrote.
    """

    seen: dict[str, object] = {"submitted": []}

    def exec_dialog(dialog) -> int:
        edit(dialog)
        table = dialog._diff_table
        seen["table"] = (
            []
            if table.isHidden()
            else [
                tuple(table.item(row, column).text() for column in range(table.columnCount()))
                for row in range(table.rowCount())
            ]
        )
        seen["label"] = dialog._diff_no_changes_label.text()
        seen["dialog"] = dialog
        seen["values"] = dict(dialog.collect_values())
        return (
            main_window_module.QDialog.DialogCode.Accepted
            if accept
            else main_window_module.QDialog.DialogCode.Rejected
        )

    monkeypatch.setattr(jobs_controller_module.ContinueRelaxationDialog, "exec", exec_dialog)
    monkeypatch.setattr(
        ran.window._jobs,
        "submit_jobs_locally",
        lambda jobs, **_kwargs: seen["submitted"].append(list(jobs)),
    )
    if ran.workflow == "pristine":
        ran.window._jobs.on_continue_relaxation_requested_from_pristine_relaxation(ran.key)
    else:
        ran.window._jobs.on_continue_relaxation_requested_from_muon_relaxation(ran.key)
    return seen


def _continue_through_run(monkeypatch: pytest.MonkeyPatch, ran: _Ran, entry: str) -> list:
    """Run the ended row again from the row (``Run``) or the dashboard, answering ``Continue``."""

    asked: list[str] = []

    def answer(_parent, _title, _text, _choices, **kwargs) -> str:
        asked.append(str(kwargs.get("detail", "")))
        return "continue"

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", answer)
    submitted = fake_local_submission(ran.window, monkeypatch)
    if entry == "row":
        ran.controller().submit([ran.key], target="local")
    else:
        ran.window._jobs._on_resubmit_from_dashboard([ran.job.id])
    assert len(asked) == 1
    return submitted


def _changes(before: JobInputTexts, after: JobInputTexts) -> list[tuple[str, str, str]]:
    return [
        (change.file, change.keyword, change.change) for change in diff_job_inputs(before, after)
    ]


def _table_changes(seen: dict[str, object]) -> list[tuple[str, str, str]]:
    files = {".cell": "cell", ".param": "param"}
    return [(files[row[0]], row[1], row[2]) for row in seen["table"]]


_PRISTINE_KEYS = ("row_1",)
_MUON_KEYS = ("positivemuon:Site 1", "muonium:Site 1")
_CUTOFF_AND_SPACING = {
    "cut_off_energy": ValueWithUnit(500.0, "eV"),
    "kpoint_mp_spacing": ValueWithUnit(0.04, "1/ang"),
}


def _case(name: str) -> dict[str, object]:
    """The findings' cases: how the run's row and project were set up."""

    return {
        # a, d, h: nothing adopted, row never configured.
        "nothing-adopted": {},
        # b: cutoff and grid adopted on the project.
        "defaults-adopted": {"defaults": (600.0, (4, 4, 6))},
        # c: explicit cutoff and spacing on the row.
        "cutoff-and-spacing": {"parameter_values": dict(_CUTOFF_AND_SPACING)},
        # e: an advanced parameter added on the row.
        "advanced-added": {
            "parameter_values": {"smearing_width": ValueWithUnit(0.15, "eV")},
            "added_advanced": ("smearing_width",),
        },
    }[name]


_CASES = ("nothing-adopted", "defaults-adopted", "cutoff-and-spacing", "advanced-added")
_WORKFLOWS = [("pristine", key) for key in _PRISTINE_KEYS] + [("muon", key) for key in _MUON_KEYS]


@pytest.mark.parametrize(("workflow", "key"), _WORKFLOWS)
@pytest.mark.parametrize("case", _CASES)
def test_an_untouched_continue_dialog_writes_the_runs_inputs_again(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, key: str, case: str
) -> None:
    ran = _run_rows(tmp_path, workflow, **_case(case))[key]
    before = ran.inputs()
    stored_before = ran.stored()

    seen = _continue_through_dialog(monkeypatch, ran)

    assert seen["table"] == []
    assert seen["label"] == "No parameter changes."
    assert seen["submitted"] == [[ran.job]]
    after = ran.inputs()
    assert _changes(before, after) == []
    assert after == before
    # The row stores what it stored before.
    assert ran.stored() == stored_before
    # A second untouched Continue still shows no changes.
    ran.job.status = JobStatus.ENDED
    ran.job.end_reason = EndReason.UNKNOWN
    second = _continue_through_dialog(monkeypatch, ran, accept=False)
    assert second["table"] == []
    ran.window.close()


@pytest.mark.parametrize(("workflow", "key"), _WORKFLOWS)
@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_run_or_resubmit_answered_continue_writes_the_runs_inputs_again(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    key: str,
    case: str,
    entry: str,
) -> None:
    ran = _run_rows(tmp_path, workflow, **_case(case))[key]
    before = ran.inputs()

    submitted = _continue_through_run(monkeypatch, ran, entry)

    assert [ran.job] in submitted
    assert ran.job.status is JobStatus.DEFINED
    assert (ran.job_dir / f"{ran.seed}.check").is_file()
    after = ran.inputs()
    assert _changes(before, after) == []
    assert after == before
    ran.window.close()


@pytest.mark.parametrize(("workflow", "key"), _WORKFLOWS)
def test_a_continuation_keeps_the_runs_values_after_the_project_defaults_change(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, key: str
) -> None:
    """Case f: run with 600 eV and 4 4 6, then the project adopts 900 eV and 7 7 9."""

    ran = _run_rows(tmp_path, workflow, defaults=(600.0, (4, 4, 6)))[key]
    before = ran.inputs()
    _adopt_defaults(ran.window, ran.service, ran.doc, 900.0, (7, 7, 9))

    seen = _continue_through_dialog(monkeypatch, ran)

    assert seen["table"] == []
    # The editors show the values the run used, not the new defaults.
    assert seen["values"]["cut_off_energy"] == ValueWithUnit(600.0, "eV")
    assert ran.inputs() == before
    ran.window.close()


@pytest.mark.parametrize("entry", ["row", "dashboard"])
@pytest.mark.parametrize(("workflow", "key"), _WORKFLOWS)
def test_run_answered_continue_keeps_the_runs_values_after_the_project_defaults_change(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, key: str, entry: str
) -> None:
    ran = _run_rows(tmp_path, workflow, defaults=(600.0, (4, 4, 6)))[key]
    before = ran.inputs()
    _adopt_defaults(ran.window, ran.service, ran.doc, 900.0, (7, 7, 9))

    _continue_through_run(monkeypatch, ran, entry)

    after = ran.inputs()
    assert _changes(before, after) == []
    assert "600" in after.param.split("CUT_OFF_ENERGY", 1)[1].splitlines()[0]
    ran.window.close()


def test_starting_again_after_a_continuation_picks_up_the_current_defaults(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continue leaves the row as it was, so start again uses today's defaults."""

    ran = _run_rows(tmp_path, "pristine", defaults=(600.0, (4, 4, 6)))["row_1"]
    _adopt_defaults(ran.window, ran.service, ran.doc, 900.0, (7, 7, 9))
    _continue_through_dialog(monkeypatch, ran)
    # Back to the real submission path, which runs the run-again plan.
    monkeypatch.delattr(ran.window._jobs, "submit_jobs_locally")
    ran.service.submit_job(ran.job)
    ran.service.end_job(ran.job, EndReason.WALLTIME, "walltime reached")
    ran.window._refresh_ui_after_project_change()
    before = ran.inputs()
    monkeypatch.setattr(
        workflow_table_utils, "confirm_choice", lambda *_args, **_kwargs: "start_again"
    )
    fake_local_submission(ran.window, monkeypatch)

    ran.controller().submit([ran.key], target="local")

    assert ("param", "CUT_OFF_ENERGY", "Changed") in _changes(before, ran.inputs())
    assert ("cell", "KPOINT_MP_GRID", "Changed") in _changes(before, ran.inputs())
    ran.window.close()


def _choose_cutoff(dialog) -> None:
    row = dialog._rows["cut_off_energy"]
    row.radio.setChecked(True)
    row.editor.set_value(ValueWithUnit(700.0, "eV"))


def _set_max_iterations(dialog) -> None:
    dialog._rows["geom_max_iter"].editor.set_value(250)


def _choose_spacing(dialog) -> None:
    row = dialog._rows["kpoint_mp_spacing"]
    row.radio.setChecked(True)
    row.editor.set_value(ValueWithUnit(0.05, "1/ang"))


@pytest.mark.parametrize(("workflow", "key"), _WORKFLOWS)
@pytest.mark.parametrize(
    ("case", "edit", "expected"),
    [
        ("nothing-adopted", _set_max_iterations, [("param", "GEOM_MAX_ITER", "Changed")]),
        # Choosing the cutoff over the basis precision writes the cutoff only.
        ("nothing-adopted", _choose_cutoff, [("param", "CUT_OFF_ENERGY", "Added")]),
        (
            "defaults-adopted",
            _choose_spacing,
            [("cell", "KPOINT_MP_GRID", "Removed"), ("cell", "KPOINT_MP_SPACING", "Added")],
        ),
    ],
    ids=["max-iterations", "cutoff-over-basis", "spacing-over-grid"],
)
def test_an_edit_in_the_continue_dialog_is_written_and_only_that_edit(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workflow: str,
    key: str,
    case: str,
    edit,
    expected,
) -> None:
    ran = _run_rows(tmp_path, workflow, **_case(case))[key]
    before = ran.inputs()

    seen = _continue_through_dialog(monkeypatch, ran, edit)

    assert _changes(before, ran.inputs()) == expected
    assert _table_changes(seen) == expected
    ran.window.close()


def test_an_edit_in_the_continue_dialog_is_stored_on_the_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = _run_rows(tmp_path, "pristine")["row_1"]

    _continue_through_dialog(monkeypatch, ran, _set_max_iterations)

    values, _advanced = ran.stored()
    assert values == {"geom_max_iter": 250}
    ran.window.close()


@pytest.mark.parametrize(("workflow", "key"), _WORKFLOWS)
def test_resetting_an_edit_returns_to_the_runs_value(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, key: str
) -> None:
    ran = _run_rows(tmp_path, workflow, defaults=(600.0, (4, 4, 6)))[key]
    before = ran.inputs()
    _adopt_defaults(ran.window, ran.service, ran.doc, 900.0, (7, 7, 9))

    def edit_then_reset(dialog) -> None:
        dialog._rows["cut_off_energy"].editor.set_value(ValueWithUnit(700.0, "eV"))
        dialog._reset_parameter("cut_off_energy")

    seen = _continue_through_dialog(monkeypatch, ran, edit_then_reset)

    assert seen["values"]["cut_off_energy"] == ValueWithUnit(600.0, "eV")
    assert seen["table"] == []
    assert ran.inputs() == before
    ran.window.close()


def _capture_warnings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, title, text, **_kwargs: warnings.append((title, text)),
    )
    return warnings


def test_a_continuation_whose_inputs_cannot_be_written_is_not_submitted(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = _run_rows(tmp_path, "pristine")["row_1"]
    warnings = _capture_warnings(monkeypatch)

    def refuse(*_args, **_kwargs) -> None:
        raise OSError("read-only folder")

    monkeypatch.setattr(jobs_controller_module, "write_job_inputs", refuse)

    seen = _continue_through_dialog(monkeypatch, ran)

    assert seen["submitted"] == []
    assert [title for title, _text in warnings] == ["Relaxation not continued"]
    ran.window.close()


def test_run_answered_continue_reports_inputs_it_could_not_write_back(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = _run_rows(tmp_path, "pristine")["row_1"]
    warnings = _capture_warnings(monkeypatch)

    def refuse(*_args, **_kwargs) -> None:
        raise OSError("read-only folder")

    monkeypatch.setattr(jobs_controller_module, "write_job_inputs", refuse)

    _continue_through_run(monkeypatch, ran, "row")

    assert [title for title, _text in warnings] == ["Relaxation inputs not kept"]
    ran.window.close()


@pytest.mark.parametrize(("workflow", "key"), _WORKFLOWS)
def test_continue_is_refused_when_the_jobs_inputs_are_missing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workflow: str, key: str
) -> None:
    ran = _run_rows(tmp_path, workflow)[key]
    (ran.job_dir / f"{ran.seed}.param").unlink()
    warnings = _capture_warnings(monkeypatch)
    monkeypatch.setattr(
        jobs_controller_module.ContinueRelaxationDialog,
        "exec",
        lambda _dialog: pytest.fail("the dialog must not open without the run's inputs"),
    )
    monkeypatch.setattr(
        ran.window._jobs, "submit_jobs_locally", lambda *_a, **_k: pytest.fail("must not submit")
    )

    handler = ran.window._jobs.continue_relaxation_for_job(ran.job)
    assert handler is not None
    handler()

    assert warnings == [
        (
            "Relaxation not continued",
            f"'{ran.job.label}' was not continued because its input files could not be read.",
        )
    ]
    assert ran.job.status is JobStatus.ENDED
    assert not (ran.job_dir / f"{ran.seed}.param").exists()
    ran.window.close()


@pytest.mark.parametrize("entry", ["row", "dashboard"])
def test_run_answered_continue_is_refused_when_the_jobs_inputs_are_missing(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    ran = _run_rows(tmp_path, "pristine")["row_1"]
    (ran.job_dir / f"{ran.seed}.cell").unlink()
    warnings = _capture_warnings(monkeypatch)
    monkeypatch.setattr(workflow_table_utils, "confirm_choice", lambda *_a, **_k: "continue")
    submitted = fake_local_submission(ran.window, monkeypatch)

    if entry == "row":
        ran.controller().submit([ran.key], target="local")
    else:
        ran.window._jobs._on_resubmit_from_dashboard([ran.job.id])

    assert [title for title, _text in warnings] == ["Relaxation not continued"]
    assert submitted == []
    assert ran.job.status is JobStatus.ENDED
    assert not (ran.job_dir / f"{ran.seed}.cell").exists()
    ran.window.close()


def test_a_setting_changed_on_the_row_since_the_run_applies_when_it_continues(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The row's own settings are taken from the row; the defaults it inherits are not."""

    ran = _run_rows(tmp_path, "pristine", defaults=(600.0, (4, 4, 6)))["row_1"]
    before = ran.inputs()
    _adopt_defaults(ran.window, ran.service, ran.doc, 900.0, (7, 7, 9))
    ran.controller().update_row_parameters("row_1", {"geom_max_iter": 250}, ())

    _continue_through_run(monkeypatch, ran, "row")

    assert _changes(before, ran.inputs()) == [("param", "GEOM_MAX_ITER", "Changed")]
    ran.window.close()


def test_the_run_again_question_says_a_continued_relaxation_keeps_its_inputs(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ran = _run_rows(tmp_path, "pristine")["row_1"]
    details: list[str] = []

    def answer(_parent, _title, _text, _choices, **kwargs) -> str:
        details.append(str(kwargs.get("detail", "")))
        return "cancel"

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", answer)
    fake_local_submission(ran.window, monkeypatch)
    ran.controller().submit([ran.key], target="local")

    (detail,) = details
    assert "Continue keeps the previous run's files and inputs" in detail
    assert "a changed project default does not" in detail
    ran.window.close()


def test_row_settings_reads_the_rows_own_values(qapp, tmp_path: Path) -> None:
    ran = _run_rows(tmp_path, "muon", parameter_values={"geom_max_iter": 80})["muonium:Site 1"]

    assert ran.stored() == ({"geom_max_iter": 80, "spin_polarized": True}, ())
    assert ran.controller().row_settings("muonium:Nowhere") is None
    config = deserialize_muon_relaxation_config(
        ran.controller().active_definition().parameters["configuration"]
    )
    assert {row.charge_state for row in config.rows} == {"positive_muon", "muonium"}
    ran.window.close()
