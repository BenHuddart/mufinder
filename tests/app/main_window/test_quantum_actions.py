"""Quantum Corrections actions in the main window: configuring rows and running their jobs again."""

from __future__ import annotations

from mufinder.services.job_monitoring import job_status_word
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication, QDialog

from mufinder.app import workflow_table_utils
import mufinder.app.workflow_controllers.quantum_corrections as quantum_controller_module
from mufinder.domain.enums import EndReason, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Result
from mufinder.services.project_service import ProjectService
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE
from tests.support.main_window import fake_local_submission, stage_quantum_row


def _failed_phonon_row(main_window, tmp_path: Path, *, legacy: bool):
    """Stage a row whose phonon job failed, with its seed directory on disk.

    ``legacy`` adds the completed singlepoint an older MuFinder version ran in
    the same directory first.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_again.mufinder", "QuantumAgain")
    window._document = doc
    definition, candidate, singlepoint = stage_quantum_row(
        window, service, doc, singlepoint_status=JobStatus.COMPLETED if legacy else None
    )
    definition.parameters["job_seed_names_by_label"] = {candidate.key: "quantum_siteA"}
    phonon = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    phonon.status = JobStatus.ENDED
    phonon.end_reason = EndReason.UNKNOWN
    phonon.execution_target = "archer2"
    phonon.remote_directory = "/work/quantum_siteA"
    definition.parameters["phonon_job_ids_by_label"] = {candidate.key: str(phonon.id)}
    directory = doc.project_root / "files" / "quantum_corrections" / "quantum_siteA"
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in (".cell", ".param", ".castep", ".check"):
        (directory / f"quantum_siteA{suffix}").write_text("data\n", encoding="utf-8")
    window._refresh_quantum_corrections_view()
    return window, doc, definition, candidate, singlepoint, phonon, directory


def _answer_run_again(monkeypatch: pytest.MonkeyPatch, answer: str) -> list[str]:
    """Answer the run-again question with ``answer`` and record its title."""

    asked: list[str] = []

    def _answer(_parent, title, _text, _choices, **_kwargs) -> str:
        asked.append(title)
        return answer

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", _answer)
    return asked


@pytest.mark.parametrize("legacy", [False, True])
def test_starting_a_failed_phonon_row_again_reruns_it_as_a_fresh_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    window, doc, _definition, candidate, singlepoint, phonon, directory = _failed_phonon_row(
        main_window, tmp_path, legacy=legacy
    )
    asked = _answer_run_again(monkeypatch, "start_again")
    submitted = fake_local_submission(window, monkeypatch)

    window._quantum_corrections_controller.submit([candidate.key], target="local")

    assert asked == ["Run again"]
    assert submitted == [[phonon]]
    assert phonon.status == JobStatus.DEFINED
    # The remote directory is forgotten, so a cluster run goes to a fresh target.
    assert phonon.remote_directory is None
    # The legacy singlepoint's files in the shared folder go with the phonon output.
    assert sorted(path.name for path in directory.iterdir()) == [
        "quantum_siteA.cell",
        "quantum_siteA.param",
    ]
    if singlepoint is not None:
        assert doc.jobs[singlepoint.id].status == JobStatus.COMPLETED
    controller = window._quantum_corrections_controller
    request = controller.configure_request([candidate.key])
    assert request is not None and request.read_only is False
    window.close()


def test_cancelling_the_run_again_question_leaves_a_failed_phonon_row_as_it_was(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, _doc, _definition, candidate, _singlepoint, phonon, directory = _failed_phonon_row(
        main_window, tmp_path, legacy=True
    )
    asked = _answer_run_again(monkeypatch, "cancel")
    submitted = fake_local_submission(window, monkeypatch)

    window._quantum_corrections_controller.submit([candidate.key], target="local")

    assert asked == ["Run again"]
    assert submitted == []
    assert job_status_word(phonon) == "failed"
    assert phonon.remote_directory == "/work/quantum_siteA"
    assert (directory / "quantum_siteA.check").exists()
    window.close()


class _FakeConfigDialog:
    """Records how a source Configure dialog was opened and accepts it."""

    def __init__(self, opened: list[dict[str, object]], *, num_farms: int, **kwargs) -> None:
        self._opened = opened
        self._num_farms = num_farms
        opened.append({"num_farms": num_farms, **kwargs})

    def setWindowTitle(self, title: str) -> None:  # noqa: N802 - Qt API naming.
        self._opened[-1]["title"] = title

    def exec(self):
        from PySide6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted

    def deleteLater(self) -> None:  # noqa: N802 - Qt API naming.
        self._opened[-1]["deleted"] = True

    def collect_values(self) -> dict[str, object]:
        return {}

    def advanced_names(self) -> tuple[str, ...]:
        return ()

    def untouched_mixed_names(self) -> set[str]:
        return set()

    def num_farms(self) -> int:
        return 4

    def num_farms_edited(self) -> bool:
        return True


def test_configuring_a_ready_row_edits_every_parameter(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.workflow_config_codecs import workflow_row_configuration

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_farms.mufinder", "QuantumFarms")
    window._document = doc
    definition, candidate, _singlepoint = stage_quantum_row(window, service, doc)
    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        quantum_controller_module,
        "QuantumCorrectionsConfigDialog",
        lambda **kwargs: _FakeConfigDialog(opened, **kwargs),
    )

    window._quantum_corrections_view.configure_requested.emit([candidate.key])

    assert len(opened) == 1
    assert opened[0]["title"] == "Configure Quantum Corrections Parameters"
    assert opened[0]["read_only"] is False
    assert not {"editable_param_names", "num_farms_editable", "pseudopotential_editable"} & set(
        opened[0]
    )
    assert workflow_row_configuration(definition, candidate.key)[2] == {"num_farms": 4}
    assert "pseudopotential_family" not in opened[0]
    # Seeded through the constructor and deleted once it closes.
    assert {"values", "advanced_names", "mixed_names"} <= set(opened[0])
    assert opened[0]["deleted"] is True


@pytest.mark.parametrize("legacy", [False, True])
def test_configuring_a_failed_row_is_editable_without_a_reset(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    """A failed row's Configure dialog opens editable: its next run rebuilds the inputs."""

    window, _doc, _definition, candidate, _singlepoint, phonon, _directory = _failed_phonon_row(
        main_window, tmp_path, legacy=legacy
    )
    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        quantum_controller_module,
        "QuantumCorrectionsConfigDialog",
        lambda **kwargs: _FakeConfigDialog(opened, **kwargs),
    )

    window._quantum_corrections_view.configure_requested.emit([candidate.key])

    assert opened[-1]["read_only"] is False
    assert "Reset" not in str(opened[-1]["intro_text"])
    assert phonon.status == JobStatus.ENDED
    window.close()


def _completed_quantum_row(window, tmp_path: Path):
    """Stage a row whose phonon job completed, with a result pointing at a phonon file."""

    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_result.mufinder", "QuantumResult")
    window._document = doc
    definition, candidate, _singlepoint = stage_quantum_row(window, service, doc)
    phonon = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    phonon.status = JobStatus.COMPLETED
    definition.parameters["phonon_job_ids_by_label"] = {candidate.key: str(phonon.id)}
    definition.parameters["phonon_source_run_ids_by_label"] = {candidate.key: str(candidate.run.id)}
    phonon_path = doc.project_root / "site.phonon"
    phonon_path.write_text("phonon\n", encoding="utf-8")
    run = CalculationRun(job_id=phonon.id, status=JobStatus.COMPLETED)
    result = Result(
        calculation_run_id=run.id,
        result_type=QUANTUM_PHONON_CALC_TYPE,
        data={"phonon_file": str(phonon_path), "muon_zpe_total_eV": 0.5},
    )
    doc.runs[run.id] = run
    doc.results[result.id] = result
    window._refresh_ui_after_project_change()
    window._refresh_quantum_corrections_view()
    return result, phonon_path


def test_open_analysis_opens_the_selected_site_titled_by_site_and_charge(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    result, phonon_path = _completed_quantum_row(window, tmp_path)
    constructed: list[dict[str, object]] = []

    class _FakeDialog(QDialog):
        def __init__(self, **kwargs) -> None:
            constructed.append(kwargs)
            super().__init__(kwargs["parent"])

    monkeypatch.setattr(quantum_controller_module, "QuantumPhononAnalysisDialog", _FakeDialog)
    view = window._quantum_corrections_view

    view._sources_table.selectRow(0)
    view._open_analysis_button.click()
    view._open_analysis_button.click()

    assert len(constructed) == 1
    assert constructed[0]["phonon_path"] == str(phonon_path)
    assert constructed[0]["site_label"] == "siteA, positive muon"
    assert constructed[0]["parent"] is window
    dialog = window._quantum_corrections_controller.analysis_dialog(result.id)
    assert dialog.isVisible()
    dialog.close()


def test_reopening_a_closed_analysis_keeps_track_of_the_new_window(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    result, _phonon_path = _completed_quantum_row(window, tmp_path)
    constructed: list[QDialog] = []

    class _FakeDialog(QDialog):
        def __init__(self, **kwargs) -> None:
            super().__init__(kwargs["parent"])
            constructed.append(self)

    monkeypatch.setattr(quantum_controller_module, "QuantumPhononAnalysisDialog", _FakeDialog)
    controller = window._quantum_corrections_controller

    controller.open_analysis(str(result.id))
    constructed[0].close()
    controller.open_analysis(str(result.id))
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    controller.open_analysis(str(result.id))

    assert len(constructed) == 2
    assert controller.analysis_dialog(result.id) is constructed[1]
    constructed[1].close()


def test_a_result_whose_site_is_gone_still_opens_its_analysis(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    result, phonon_path = _completed_quantum_row(window, tmp_path)
    doc = window._document
    for site in doc.sites.values():
        site.clear_relaxation()
    window._refresh_quantum_corrections_view()
    constructed: list[dict[str, object]] = []

    class _FakeDialog(QDialog):
        def __init__(self, **kwargs) -> None:
            constructed.append(kwargs)
            super().__init__(kwargs["parent"])

    monkeypatch.setattr(quantum_controller_module, "QuantumPhononAnalysisDialog", _FakeDialog)
    view = window._quantum_corrections_view

    assert view._sources_table.rowCount() == 1
    assert view._sources_table.item(0, 0).text() == ("quantum_phonon_siteA (no eligible site)")
    assert view._sources_table.item(0, 3).text() == "0.500"
    view._sources_table.selectRow(0)
    view._open_analysis_button.click()

    assert constructed[0]["phonon_path"] == str(phonon_path)
    assert constructed[0]["site_label"] == "quantum_phonon_siteA"
    window._quantum_corrections_controller.analysis_dialog(result.id).close()


#: A two-atom Gamma-point ``.phonon`` file with the muon as atom 2.
_TWO_ATOM_PHONON = """ BEGIN header
 Number of ions         2
 Number of branches     6
 Number of wavevectors  1
 Frequencies in         cm-1
 Unit cell vectors (A)
    1.000000    0.000000    0.000000
    0.000000    1.000000    0.000000
    0.000000    0.000000    1.000000
 Fractional Co-ordinates
     1     0.000000    0.000000    0.000000   N        14.006740
     2     0.500000    0.500000    0.500000   H:mu      0.113429
 END header
     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000
       1     100.000000
       2     400.000000
       3     900.000000
       4    1600.000000
       5    2500.000000
       6    3600.000000
                        Phonon Eigenvectors
Mode Ion                X                                   Y                                   Z
""" + "".join(
    f"   {mode}   {ion}  {'0.408248000000  0.000000000000      ' * 3}\n"
    for mode in range(1, 7)
    for ion in (1, 2)
)


@pytest.mark.parametrize(
    ("contents", "refusal"),
    [
        (b"phonon\n", "Expected CASTEP phonon header."),
        (b"\xff\xfe not text\n", "can't decode"),
        (
            _TWO_ATOM_PHONON.replace(
                "0.000000  0.000000  0.000000", "0.500000  0.000000  0.000000"
            ).encode(),
            "Gamma-point phonon data were not found.",
        ),
        (
            _TWO_ATOM_PHONON.replace("H:mu      0.113429", "H         1.007940").encode(),
            "Expected exactly one atom labelled 'H:mu'; found 0.",
        ),
    ],
    ids=["not-a-phonon-file", "not-text", "no-gamma-point", "no-muon"],
)
def test_a_refused_phonon_file_leaves_no_analysis_window_behind(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: bytes,
    refusal: str,
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from mufinder.app.quantum_analysis_dialog import QuantumPhononAnalysisDialog

    window = main_window
    result, phonon_path = _completed_quantum_row(window, tmp_path)
    phonon_path.write_bytes(contents)
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, text, *_args: warnings.append((title, text)),
    )
    view = window._quantum_corrections_view

    view._sources_table.selectRow(0)
    view._open_analysis_button.click()

    assert [title for title, _text in warnings] == ["Phonon analysis unavailable"]
    assert refusal in warnings[0][1]
    assert window.findChildren(QuantumPhononAnalysisDialog) == []
    assert window._quantum_corrections_controller.analysis_dialog(result.id) is None


def test_next_step_goes_to_energy_barriers(main_window, tmp_path: Path) -> None:
    window = main_window
    _completed_quantum_row(window, tmp_path)
    view = window._quantum_corrections_view
    assert not view._next_button.isHidden()

    view._next_button.click()

    assert window._main_stack.currentWidget() is window._energy_barriers_view


def test_page_fits_the_default_window_width(main_window, tmp_path: Path) -> None:
    window = main_window
    _completed_quantum_row(window, tmp_path)
    window._set_active_route("quantum_corrections")
    window.resize(1333, 864)
    window.show()
    QApplication.processEvents()

    view = window._quantum_corrections_view
    assert window._main_stack.currentWidget() is view
    assert window.width() == 1333
    assert view.minimumSizeHint().width() <= view.width()
    window.close()


def test_running_a_failed_phonon_row_again_asks_to_continue_from_its_checkpoint(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered_choices
) -> None:
    """A phonon job's inputs resume from a checkpoint, so its Run asks the run-again question.

    Continue (the default) keeps the job's folder and its remote folder, where
    the checkpoint is, and rebuilds the inputs from the row.
    """

    window, doc, _definition, candidate, _singlepoint, phonon, directory = _failed_phonon_row(
        main_window, tmp_path, legacy=False
    )
    submitted: list[tuple[list[object], dict[str, object]]] = []
    monkeypatch.setattr(
        window._jobs,
        "submit_jobs_on_cluster",
        # Stands for an accepted cluster submission, which calls the accept step. The
        # job continues, so no output is cleared.
        lambda jobs, **kwargs: (
            submitted.append((list(jobs), kwargs)),
            kwargs["on_accepted"](jobs, set(), lambda restarting: (list(restarting), [])),
        ),
    )

    window._quantum_corrections_controller.submit([candidate.key], target="cluster")

    (question,) = answered_choices
    assert question["title"] == "Run again"
    assert "could not be checked (Not connected to archer2)" in question["detail"]
    assert [jobs for jobs, _kwargs in submitted] == [[phonon]]
    assert phonon.status == JobStatus.DEFINED
    assert phonon.remote_directory == "/work/quantum_siteA"
    assert (directory / "quantum_siteA.check").is_file()
    assert "CONTINUATION : DEFAULT" in (directory / "quantum_siteA.param").read_text(
        encoding="utf-8"
    )
    window.close()


# --------------------------------------------------------------------------- #
# Running a completed row again with other parameters
# --------------------------------------------------------------------------- #


class _ChangingConfigDialog(_FakeConfigDialog):
    """Accepts the Configure dialog with a looser SCF tolerance than the run used."""

    def __init__(self, opened: list[dict[str, object]], **kwargs) -> None:
        super().__init__(opened, **kwargs)
        self._seed = dict(kwargs["values"])

    def collect_values(self) -> dict[str, object]:
        from mufinder.castep_io.types import ValueWithUnit

        return {**self._seed, "elec_energy_tol": ValueWithUnit(1e-8, "eV")}

    def num_farms_edited(self) -> bool:
        return False


class _UnchangedConfigDialog(_ChangingConfigDialog):
    """Accepts the Configure dialog exactly as it opened."""

    def collect_values(self) -> dict[str, object]:
        return dict(self._seed)


def _project_bytes(doc) -> dict[str, bytes]:
    return {
        str(path.relative_to(doc.project_root)): path.read_bytes()
        for path in sorted(doc.project_root.rglob("*"))
        if path.is_file()
    }


def test_accepting_a_completed_row_unchanged_writes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    window = main_window
    doc, candidate, _phonon, _result, _directory = _completed_phonon_run(window, tmp_path)
    window._save_project()
    before = _project_bytes(doc)
    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        quantum_controller_module,
        "QuantumCorrectionsConfigDialog",
        lambda **kwargs: _UnchangedConfigDialog(opened, **kwargs),
    )

    window._quantum_corrections_view.configure_requested.emit([candidate.key])

    assert opened and opened[-1]["read_only"] is False
    assert confirmed_destructive_actions == []
    assert _project_bytes(doc) == before
    window.close()


@pytest.mark.parametrize("signal", ["submit_local_requested", "submit_cluster_requested"])
def test_running_an_unchanged_completed_row_says_how_to_run_it_again(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signal: str
) -> None:
    window = main_window
    _doc, candidate, phonon, result, _directory = _completed_phonon_run(window, tmp_path)
    shown: list[tuple[str, str]] = []
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(
        QMessageBox, "information", lambda _parent, title, text, *_a: shown.append((title, text))
    )
    submitted = fake_local_submission(window, monkeypatch)

    # The page's one site: Run Selected and Run All both hand over its key.
    getattr(window._quantum_corrections_view, signal).emit([candidate.key])

    assert shown == [
        (
            "Nothing to run",
            "'siteA, positive muon' has completed. To run it again, change one of its "
            "parameters with 'Configure…' first; running it then replaces its phonon result.",
        )
    ]
    assert submitted == []
    assert phonon.status == JobStatus.COMPLETED
    assert result.id in _doc.results
    window.close()


class _ShownDialog(QDialog):
    """Stands for the analysis window."""

    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs["parent"])


def _complete_with_result(doc, phonon, directory: Path, *, zpe: float) -> Result:
    """Stand for a phonon job's collection: completed, with its result and output files."""

    phonon.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=phonon.id, status=JobStatus.COMPLETED)
    phonon_path = directory / f"{directory.name}.phonon"
    phonon_path.write_text("phonon\n", encoding="utf-8")
    (directory / f"{directory.name}.castep").write_text("castep\n", encoding="utf-8")
    result = Result(
        calculation_run_id=run.id,
        result_type=QUANTUM_PHONON_CALC_TYPE,
        data={"phonon_file": str(phonon_path), "muon_zpe_total_eV": zpe},
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    return result


def _completed_phonon_run(window, tmp_path: Path):
    """Run a staged row's phonon job to a result, its inputs written as a real run writes them."""

    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_rerun.mufinder", "QuantumRerun")
    window._document = doc
    _definition, candidate, _singlepoint = stage_quantum_row(window, service, doc)
    (phonon,) = window._quantum_corrections_controller.materialize([candidate.key])
    directory = window._job_output_directory(phonon)
    result = _complete_with_result(doc, phonon, directory, zpe=0.25)
    window._refresh_ui_after_project_change()
    window._refresh_quantum_corrections_view()
    return doc, candidate, phonon, result, directory


def _edit_completed_row(window, candidate, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        quantum_controller_module,
        "QuantumCorrectionsConfigDialog",
        lambda **kwargs: _ChangingConfigDialog(opened, **kwargs),
    )
    window._quantum_corrections_view.configure_requested.emit([candidate.key])
    assert opened[-1]["read_only"] is False


def _row(window, key: str):
    return next(row for row in window._quantum_corrections_view._source_rows if row.key == key)


def _files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_a_confirmed_change_then_a_cancelled_submission_keeps_the_result_and_files(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    window = main_window
    doc, candidate, phonon, result, directory = _completed_phonon_run(window, tmp_path)
    files = _files(directory)
    _edit_completed_row(window, candidate, monkeypatch)
    assert [prompt["title"] for prompt in confirmed_destructive_actions] == ["Change parameters"]
    assert _row(window, candidate.key).status == "edited"
    # The edited row still holds its completed result.
    assert window._quantum_corrections_view._run_status_label.text() == "1 of 1 completed"
    submitted = fake_local_submission(window, monkeypatch, accept=False)

    window._quantum_corrections_view.submit_local_requested.emit([candidate.key])

    assert submitted == []
    assert phonon.status == JobStatus.COMPLETED
    assert result.id in doc.results
    assert _files(directory) == files
    row = _row(window, candidate.key)
    assert (row.status, row.result_id, row.zpe) == ("edited", str(result.id), "0.250")
    assert not any(record.action == "job_reset" for record in doc.provenance)
    window.close()


def test_running_an_edited_row_replaces_its_result_once_the_new_run_completes(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered_choices
) -> None:
    window = main_window
    doc, candidate, phonon, result, directory = _completed_phonon_run(window, tmp_path)
    controller = window._quantum_corrections_controller
    monkeypatch.setattr(quantum_controller_module, "QuantumPhononAnalysisDialog", _ShownDialog)
    controller.open_analysis(str(result.id))
    analysis = controller.analysis_dialog(result.id)
    assert analysis is not None and analysis.isVisible()
    _edit_completed_row(window, candidate, monkeypatch)
    submitted = fake_local_submission(window, monkeypatch)

    window._quantum_corrections_view.submit_local_requested.emit([candidate.key])

    # Accepted: the job starts again in its own, cleared folder with the new inputs, and the
    # run-again question is not asked, since replacing the result was confirmed already.
    assert answered_choices == []
    assert submitted == [[phonon]]
    assert phonon.status == JobStatus.DEFINED
    assert result.id not in doc.results
    assert all(run.result_id is None for run in doc.runs.values() if run.job_id == phonon.id)
    assert sorted(_files(directory)) == [f"{directory.name}.cell", f"{directory.name}.param"]
    assert "ELEC_ENERGY_TOL : 1e-08 eV" in (directory / f"{directory.name}.param").read_text(
        encoding="utf-8"
    )
    (reset,) = [record for record in doc.provenance if record.action == "job_reset"]
    assert reset.parameters["removed_result"] is True
    assert reset.parameters["cleaned_output_directory"] is True
    # The analysis window of the removed result closes rather than show it as current.
    assert controller.analysis_dialog(result.id) is None
    assert not analysis.isVisible()
    row = _row(window, candidate.key)
    assert (row.status, row.result_id, row.zpe) == ("ready", "", "–")

    new_result = _complete_with_result(doc, phonon, directory, zpe=0.3)
    window._refresh_quantum_corrections_view()

    row = _row(window, candidate.key)
    assert (row.status, row.result_id, row.zpe) == ("completed", str(new_result.id), "0.300")
    window.close()


def test_a_failed_rerun_of_an_edited_row_leaves_a_failed_row_without_a_result(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once accepted the old result is gone, so a failed rerun reads like any failed row."""

    window = main_window
    doc, candidate, phonon, result, _directory = _completed_phonon_run(window, tmp_path)
    _edit_completed_row(window, candidate, monkeypatch)
    fake_local_submission(window, monkeypatch)
    window._quantum_corrections_view.submit_local_requested.emit([candidate.key])

    phonon.status = JobStatus.ENDED
    phonon.end_reason = EndReason.UNKNOWN
    window._refresh_quantum_corrections_view()

    row = _row(window, candidate.key)
    assert (row.status, row.result_id) == ("failed", "")
    assert result.id not in doc.results
    request = window._quantum_corrections_controller.configure_request([candidate.key])
    assert request is not None and request.read_only is False
    window.close()
