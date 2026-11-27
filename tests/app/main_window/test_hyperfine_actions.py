"""Hyperfine Coupling actions in the main window: handing a coupling to Magnetic Fields."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app import workflow_table_utils
from mufinder.castep_io.magres import hyperfine_tensor_for_species, parse_magres_hyperfine
from mufinder.domain.models import Result, Structure
from mufinder.services.hyperfine_results import contact_shell_inputs, hyperfine_result_payload
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE
from tests.support.main_window import open_project_with_host

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "hyperfine"


def _hyperfine_result(document, relaxation_job_id: str, *, tied: bool = False) -> Result:
    # ``tied`` puts the muon halfway between the two Fe, whose moments differ.
    relaxed = Structure(
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Fe", "Fe", "H:mu"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0] if tied else [0.1, 0.0, 0.0],
            [0.25, 0.0, 0.0] if tied else [0.25, 0.25, 0.25],
        ],
    )
    tensor = hyperfine_tensor_for_species(
        parse_magres_hyperfine(FIXTURES / "fe_tetrahedral_muon.magres"), "H:mu"
    )
    result = Result(
        result_type=HYPERFINE_CALC_TYPE,
        data=hyperfine_result_payload(
            tensor,
            pseudo_spin_density_angstrom3=None,
            shell=contact_shell_inputs(relaxed, (2.2, 2.0, 0.0)),
            source={
                "source_site_id": None,
                "source_relaxation_job_id": relaxation_job_id,
                "relaxed_structure_id": str(relaxed.id),
                "charge_state": "positive_muon",
            },
        ),
    )
    document.results[result.id] = result
    return result


def _selected_site(relaxation_job_id: str) -> dict:
    return {
        "source": "relaxation_clustering",
        "label": "Analysis cluster 1",
        "fractional_position": [0.25, 0.25, 0.25],
        "properties": {"cluster": {"representative_job_id": relaxation_job_id}},
    }


def test_use_in_magnetic_fields_confirms_before_replacing_a_coupling(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(window, tmp_path)
    relaxation_job_id = str(uuid4())
    result = _hyperfine_result(document, relaxation_job_id)
    document.ui_state["magnetic_field_form"] = {
        "selected_muon_site": _selected_site(relaxation_job_id),
        "hyperfine_coupling": "-0.5",
        "hyperfine_coupling_unit": "T_per_muB",
    }
    prompts: list[tuple] = []
    answers = [False, True]
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt, **kwargs: prompts.append((*prompt, kwargs)) or answers.pop(0),
    )
    messages: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    assert len(prompts) == 1
    assert document.ui_state["magnetic_field_form"]["hyperfine_coupling"] == "-0.5"
    assert messages == []

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    form = document.ui_state["magnetic_field_form"]
    assert form["hyperfine_coupling_unit"] == "angstrom^-3"
    assert float(form["hyperfine_coupling"]) < 0
    assert messages and messages[-1].startswith(
        "Filled the hyperfine coupling of 1 Magnetic Fields form."
    )
    assert (
        window._magnetic_field_view.collect_state()["hyperfine_coupling"]
        == (form["hyperfine_coupling"])
    )


def test_use_in_magnetic_fields_without_a_matching_form_says_so(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(window, tmp_path)
    result = _hyperfine_result(document, str(uuid4()))
    document.ui_state["magnetic_field_form"] = {"selected_muon_site": None}
    messages: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    assert len(messages) == 1
    assert messages[0].startswith("No Magnetic Fields form shows this muon site.")
    assert "hyperfine_coupling" not in document.ui_state["magnetic_field_form"]


def test_use_in_magnetic_fields_sees_the_site_an_unedited_form_shows(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(window, tmp_path)
    relaxation_job_id = str(uuid4())
    result = _hyperfine_result(document, relaxation_job_id)
    # A form never edited stores nothing, though its page shows a default site.
    document.ui_state.pop("magnetic_field_form", None)
    shown = {
        **window._magnetic_field_view.collect_state(),
        "selected_muon_site": _selected_site(relaxation_job_id),
    }
    monkeypatch.setattr(window._magnetic_field_view, "collect_state", lambda: dict(shown))
    prompts: list[tuple] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda _parent, *prompt, **kwargs: prompts.append(prompt) or True,
    )
    messages: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    form = document.ui_state["magnetic_field_form"]
    assert float(form["hyperfine_coupling"]) < 0
    # The widget's default 0.0 coupling is empty, not a value to confirm replacing.
    assert prompts == []
    assert messages and messages[-1].startswith("Filled the hyperfine coupling of 1")


def test_use_in_magnetic_fields_warns_when_the_form_shell_splits_a_tie(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    _service, document, _host = open_project_with_host(window, tmp_path)
    relaxation_job_id = str(uuid4())
    result = _hyperfine_result(document, relaxation_job_id, tied=True)
    document.ui_state["magnetic_field_form"] = {
        "selected_muon_site": _selected_site(relaxation_job_id),
        "contact_nnn": 1,
    }
    messages: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: messages.append(text),
    )

    window._on_use_hyperfine_in_magnetic_fields(str(result.id))

    (message,) = messages
    assert message.startswith("Filled the hyperfine coupling of 1 Magnetic Fields form.")
    assert "\n\nWarning:\nMagnetic Fields: Contact neighbours (nnn) = 1 keeps 1 of the 2" in message
    assert "nnn = 2 includes them all" in message


# --- running a completed row again with other parameters ---


class _ChangingConfigDialog:
    """The Configure dialog, accepted with a looser SCF tolerance than the run used."""

    def __init__(self, opened: list[dict[str, object]], *, change: bool = True, **kwargs) -> None:
        self._opened = opened
        self._seed = dict(kwargs["values"])
        self._change = change
        opened.append(kwargs)

    def setWindowTitle(self, title: str) -> None:  # noqa: N802 - Qt API naming.
        self._opened[-1]["title"] = title

    def exec(self):
        from PySide6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted

    def deleteLater(self) -> None:  # noqa: N802 - Qt API naming.
        pass

    def collect_values(self) -> dict[str, object]:
        from mufinder.castep_io.types import ValueWithUnit

        if not self._change:
            return dict(self._seed)
        return {**self._seed, "elec_energy_tol": ValueWithUnit(1e-8, "eV")}

    def advanced_names(self) -> tuple[str, ...]:
        return ()

    def untouched_mixed_names(self) -> set[str]:
        return set()


def _complete_with_result(doc, job, directory: Path, *, isotropic: float) -> Result:
    """Stand for a magres job's collection: completed, with its result and output files."""

    from mufinder.domain.enums import JobStatus
    from mufinder.domain.models import CalculationRun

    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    (directory / f"{directory.name}.magres").write_text("magres\n", encoding="utf-8")
    (directory / f"{directory.name}.castep").write_text("castep\n", encoding="utf-8")
    result = Result(
        calculation_run_id=run.id,
        result_type=HYPERFINE_CALC_TYPE,
        data={"isotropic_mhz": isotropic, "charge_state": "positive_muon"},
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    return result


def _completed_hyperfine_row(window, tmp_path: Path):
    """A completed Hyperfine Coupling row whose coupling is already in a Magnetic Fields form.

    The form holds the coupling "Use in Magnetic Fields" wrote and a field
    result computed with it.
    """

    from mufinder.services.magnetic_field_results import (
        MAGNETIC_FIELD_PAGE_OWNER,
        MAGNETIC_FIELD_RESULT_TYPE,
    )
    from mufinder.services.project_service import ProjectService

    from tests.support.main_window import stage_quantum_row

    service = ProjectService()
    doc = service.create_project(tmp_path / "hyperfine_rerun.mufinder", "HyperfineRerun")
    window._document = doc
    _definition, candidate, _singlepoint = stage_quantum_row(window, service, doc)
    controller = window._hyperfine_coupling_controller
    (job,) = controller.materialize([candidate.key])
    directory = window._job_output_directory(job)
    result = _complete_with_result(doc, job, directory, isotropic=12.5)
    doc.ui_state["magnetic_field_form"] = {
        "hyperfine_coupling": "0.123",
        "hyperfine_coupling_unit": "angstrom^-3",
    }
    field = Result(
        result_type=MAGNETIC_FIELD_RESULT_TYPE, data={"owner": MAGNETIC_FIELD_PAGE_OWNER}
    )
    doc.results[field.id] = field
    window._refresh_ui_after_project_change()
    window._refresh_hyperfine_coupling_view()
    return doc, candidate, job, result, directory, field


def _magnetic_fields_untouched(doc, field) -> bool:
    form = doc.ui_state["magnetic_field_form"]
    return (
        form["hyperfine_coupling"] == "0.123"
        and form["hyperfine_coupling_unit"] == "angstrom^-3"
        and field.id in doc.results
    )


def _source_row(window, key: str):
    return next(row for row in window._hyperfine_coupling_view._source_rows if row.key == key)


def _result_keys(window) -> list[str]:
    return [row.key for row in window._hyperfine_coupling_view._result_rows]


def _files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def _edit(window, candidate, monkeypatch: pytest.MonkeyPatch, *, change: bool = True) -> list:
    import mufinder.app.workflow_controllers.hyperfine_coupling as module

    opened: list[dict[str, object]] = []
    monkeypatch.setattr(
        module,
        "HyperfineCouplingConfigDialog",
        lambda **kwargs: _ChangingConfigDialog(opened, change=change, **kwargs),
    )
    window._hyperfine_coupling_view.configure_requested.emit([candidate.key])
    return opened


def test_configure_on_a_completed_row_opens_editable_and_unchanged_writes_nothing(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    window = main_window
    doc, candidate, _job, _result, _directory, _field = _completed_hyperfine_row(window, tmp_path)
    window._save_project()
    before = {
        str(path.relative_to(doc.project_root)): path.read_bytes()
        for path in sorted(doc.project_root.rglob("*"))
        if path.is_file()
    }

    opened = _edit(window, candidate, monkeypatch, change=False)

    assert opened[-1]["read_only"] is False
    assert opened[-1]["title"] == "Configure Hyperfine Coupling Parameters"
    assert confirmed_destructive_actions == []
    after = {
        str(path.relative_to(doc.project_root)): path.read_bytes()
        for path in sorted(doc.project_root.rglob("*"))
        if path.is_file()
    }
    assert after == before
    window.close()


def test_a_confirmed_change_then_a_cancelled_submission_keeps_the_result_and_magnetic_fields(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    from mufinder.domain.enums import JobStatus

    from tests.support.main_window import fake_local_submission

    window = main_window
    doc, candidate, job, result, directory, field = _completed_hyperfine_row(window, tmp_path)
    files = _files(directory)

    _edit(window, candidate, monkeypatch)

    (prompt,) = confirmed_destructive_actions
    assert prompt["title"] == "Change parameters"
    assert prompt["detail"].endswith(
        "A coupling already used in Magnetic Fields stays as it is until 'Use in Magnetic "
        "Fields' is run again on the new result."
    )
    assert _source_row(window, candidate.key).status == "edited"
    assert _result_keys(window) == [str(result.id)]
    assert _magnetic_fields_untouched(doc, field)
    submitted = fake_local_submission(window, monkeypatch, accept=False)

    window._hyperfine_coupling_view.submit_local_requested.emit([candidate.key])

    assert submitted == []
    assert job.status == JobStatus.COMPLETED
    assert result.id in doc.results
    assert _files(directory) == files
    assert _source_row(window, candidate.key).status == "edited"
    assert _result_keys(window) == [str(result.id)]
    assert not any(record.action == "job_reset" for record in doc.provenance)
    assert _magnetic_fields_untouched(doc, field)
    window.close()


def test_running_an_edited_row_replaces_its_result_once_the_new_run_completes(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered_choices
) -> None:
    from mufinder.domain.enums import JobStatus

    from tests.support.main_window import fake_local_submission

    window = main_window
    doc, candidate, job, result, directory, field = _completed_hyperfine_row(window, tmp_path)
    _edit(window, candidate, monkeypatch)
    submitted = fake_local_submission(window, monkeypatch)

    window._hyperfine_coupling_view.submit_local_requested.emit([candidate.key])

    # Accepted: the job starts again in its own, cleared folder with the new inputs, and the
    # run-again question is not asked, since replacing the result was confirmed already.
    assert answered_choices == []
    assert submitted == [[job]]
    assert job.status == JobStatus.DEFINED
    assert result.id not in doc.results
    assert all(run.result_id is None for run in doc.runs.values() if run.job_id == job.id)
    assert sorted(_files(directory)) == [f"{directory.name}.cell", f"{directory.name}.param"]
    assert "ELEC_ENERGY_TOL : 1e-08 eV" in (directory / f"{directory.name}.param").read_text(
        encoding="utf-8"
    )
    (reset,) = [record for record in doc.provenance if record.action == "job_reset"]
    assert reset.parameters["removed_result"] is True
    # Nothing shows the old result while the new run goes.
    assert _source_row(window, candidate.key).status == "ready"
    assert _result_keys(window) == []
    assert _magnetic_fields_untouched(doc, field)

    new_result = _complete_with_result(doc, job, directory, isotropic=13.0)
    window._refresh_hyperfine_coupling_view()

    assert _source_row(window, candidate.key).status == "completed"
    assert _result_keys(window) == [str(new_result.id)]
    assert _magnetic_fields_untouched(doc, field)
    window.close()


def test_a_failed_rerun_of_an_edited_row_reads_failed_without_a_result(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.domain.enums import EndReason, JobStatus

    from tests.support.main_window import fake_local_submission

    window = main_window
    doc, candidate, job, result, _directory, field = _completed_hyperfine_row(window, tmp_path)
    _edit(window, candidate, monkeypatch)
    fake_local_submission(window, monkeypatch)
    window._hyperfine_coupling_view.submit_local_requested.emit([candidate.key])

    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    window._refresh_hyperfine_coupling_view()

    assert _source_row(window, candidate.key).status == "failed"
    assert _result_keys(window) == []
    assert result.id not in doc.results
    request = window._hyperfine_coupling_controller.configure_request([candidate.key])
    assert request is not None and request.read_only is False
    assert _magnetic_fields_untouched(doc, field)
    window.close()


@pytest.mark.parametrize("signal", ["submit_local_requested", "submit_cluster_requested"])
def test_running_an_unchanged_completed_row_says_how_to_run_it_again(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signal: str
) -> None:
    from PySide6.QtWidgets import QMessageBox

    from tests.support.main_window import fake_local_submission

    window = main_window
    doc, candidate, _job, result, _directory, _field = _completed_hyperfine_row(window, tmp_path)
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox, "information", lambda _parent, title, text, *_a: shown.append((title, text))
    )
    submitted = fake_local_submission(window, monkeypatch)

    getattr(window._hyperfine_coupling_view, signal).emit([candidate.key])

    assert shown == [
        (
            "Nothing to run",
            "'siteA, positive muon' has completed. To run it again, change one of its "
            "parameters with 'Configure…' first; running it then replaces its hyperfine result.",
        )
    ]
    assert submitted == []
    assert result.id in doc.results
    window.close()
