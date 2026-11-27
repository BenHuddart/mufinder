"""Tests for :class:`HyperfineCouplingController`.

Covers one magres job per relaxed-site row, ``validate_artifacts`` on the
``.magres`` carrier, ``build_result`` on a real CASTEP ``.magres`` with a
synthetic spin density and the source relaxation's Mulliken spins, and the
Configure request.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import UUID

import pytest

from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Result
from mufinder.services.hyperfine_results import NO_MULLIKEN_SPINS_WARNING
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectService
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE

from mufinder.app.workflow_config_codecs import workflow_row_configuration
from mufinder.app.workflow_controllers.hyperfine_coupling import (
    HyperfineCouplingController,
    HyperfineCouplingDeps,
)

from tests.support.controllers import RecordingBackend, completed_castep_result, make_context
from tests.support.muon_sites import add_relaxed_muon_site, relaxation_job_id

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "hyperfine"

_MULLIKEN = """
     Atomic Populations (Mulliken)
     -----------------------------
Species          Ion Spin      s       p       d       f      Total   Charge (e)  Spin (hbar/2)
==============================================================================================
  Fe              1   up:     1.0     1.0     4.0     0.0     6.0     0.00     2.20
                      dn:     1.0     1.0     2.0     0.0     4.0
  Fe              2   up:     1.0     1.0     4.0     0.0     6.0     0.00     2.10
                      dn:     1.0     1.0     2.0     0.0     4.0
  H:mu            1   up:     0.5     0.0     0.0     0.0     0.5    -0.10    -0.02
                      dn:     0.5     0.0     0.0     0.0     0.5
==============================================================================================
"""


class _FakeView:
    def __init__(self) -> None:
        self.sources: list[list] = []
        self.results: list[list] = []

    def set_site_scope(self, **_kwargs) -> None:
        pass

    def set_sources(self, rows) -> None:
        self.sources.append(list(rows))

    def set_results(self, rows) -> None:
        self.results.append(list(rows))

    def window(self) -> None:
        return None


def _setup(tmp_path: Path, *, mulliken: bool = True, **context):
    service = ProjectService()
    document = service.create_project(tmp_path / "h.mufinder", "H")
    site = add_relaxed_muon_site(document, label="siteA", species=("Fe", "Fe", "H:mu"))
    relaxation = document.jobs[relaxation_job_id(document, site)]
    relaxation_dir = tmp_path / "relax"
    relaxation_dir.mkdir()
    if mulliken:
        (relaxation_dir / f"{relaxation.label}.castep").write_text(_MULLIKEN, encoding="utf-8")
    output_root = document.project_root / "files" / "hyperfine_coupling"

    def job_output_directory(job):
        return relaxation_dir if job.id == relaxation.id else output_root / job.label

    view = _FakeView()
    controller = HyperfineCouplingController(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
            job_output_directory=job_output_directory,
            **context,
        ),
        HyperfineCouplingDeps(
            handle_structure_host_mismatch=lambda _m: True,
            handle_site_host_mismatch=lambda _m: True,
        ),
        view,  # type: ignore[arg-type]
    )
    return document, controller, view, str(site.id), output_root


def _write_spin_density(path: Path, *, edge: float = 5.0, value: float = -0.01) -> None:
    """Write a uniform spin-polarised ``.den_fmt`` of ``value`` Å^-3 on a 2x2x2 grid."""

    volume = edge**3
    header = (
        " BEGIN header\n\n"
        "           Real Lattice(A)\n"
        f"     {edge:.7f}     0.0000000     0.0000000\n"
        f"     0.0000000     {edge:.7f}     0.0000000\n"
        f"     0.0000000     0.0000000     {edge:.7f}\n\n"
        "   2   F                        ! nspins, non-collinear spin\n"
        "   2     2     2                ! fine FFT grid along <a,b,c>\n"
        " END header: data is charge spin\n\n"
    )
    rows = "".join(
        f"  {i} {j} {k}  1.0  {value * volume}\n" for k in (1, 2) for j in (1, 2) for i in (1, 2)
    )
    path.write_text(header + rows, encoding="utf-8")


def test_materialize_writes_one_spin_polarised_magres_job_per_row(tmp_path: Path) -> None:
    _document, controller, _view, key, output_root = _setup(tmp_path)

    (job,) = controller.materialize([key])

    assert job.calc_type == HYPERFINE_CALC_TYPE
    assert job.label == "hyperfine_siteA"
    assert job.status == JobStatus.DEFINED
    definition = controller.active_definition()
    assert definition is not None
    assert definition.parameters["job_ids_by_label"] == {key: str(job.id)}
    param = (output_root / "hyperfine_siteA" / "hyperfine_siteA.param").read_text(encoding="utf-8")
    cell = (output_root / "hyperfine_siteA" / "hyperfine_siteA.cell").read_text(encoding="utf-8")
    assert "TASK : MAGRES" in param
    assert "MAGRES_TASK : HYPERFINE" in param
    assert "SPIN_POLARIZED : TRUE" in param
    assert "WRITE_FORMATTED_DENSITY : TRUE" in param
    assert "ELEC_ENERGY_TOL : 1e-09 eV" in param
    assert "%BLOCK SPECIES_GAMMA" in cell
    # The row's job is reused, not duplicated.
    assert controller.materialize([key]) == [job]


def test_configure_request_hides_workflow_controlled_parameters(tmp_path: Path) -> None:
    _document, controller, _view, key, _output_root = _setup(tmp_path)

    request = controller.configure_request([key])
    names = {spec.name for spec in controller.parameter_specs()}

    assert request is not None
    assert not request.read_only
    assert "spin_polarized" not in request.row_values[0]
    assert {
        "task",
        "magres_task",
        "spin_polarized",
        "write_formatted_density",
        "charge",
    }.isdisjoint(names)


def _run(document, job) -> CalculationRun:
    run = CalculationRun(job_id=job.id, status=JobStatus.ACTIVE)
    document.runs[run.id] = run
    return run


def test_validate_artifacts_fails_a_missing_or_unreadable_magres(tmp_path: Path) -> None:
    document, controller, _view, key, _output_root = _setup(tmp_path)
    (job,) = controller.materialize([key])
    run = _run(document, job)
    parsed = completed_castep_result()

    missing = controller.validate_artifacts(
        job, parsed, run, ParsedJobArtifacts(magres_missing=True)
    )
    broken = controller.validate_artifacts(job, parsed, run, ParsedJobArtifacts(magres_error="bad"))

    assert missing == "Expected CASTEP magres output was not found: hyperfine_siteA.magres"
    assert broken == "Failed to read CASTEP magres output: bad"


def test_validate_artifacts_reports_castep_error_and_stale_magres(tmp_path: Path) -> None:
    from dataclasses import replace
    from datetime import UTC, datetime, timedelta
    import os

    from mufinder.castep_io.magres import parse_magres_hyperfine

    document, controller, _view, key, _output_root = _setup(tmp_path)
    (job,) = controller.materialize([key])
    run = _run(document, job)
    unconverged = replace(
        completed_castep_result(),
        error_message="electronic_minimisation of initial cell failed.",
    )

    # CASTEP stopped before writing a .magres: its own error is the reason.
    assert (
        controller.validate_artifacts(
            job, unconverged, run, ParsedJobArtifacts(magres_missing=True)
        )
        == "electronic_minimisation of initial cell failed."
    )

    # A .magres left by an earlier run is not this run's result.
    stale = tmp_path / "stale.magres"
    shutil.copy(FIXTURES / "muonium_vacuum.magres", stale)
    old = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
    os.utime(stale, (old, old))
    run.started_at = datetime.now(UTC)
    (tensor,) = parse_magres_hyperfine(stale)
    artifacts = ParsedJobArtifacts(magres=tensor, magres_path=stale)
    message = controller.validate_artifacts(job, completed_castep_result(), run, artifacts)
    assert message is not None and "predates this run" in message


def test_build_result_stores_the_coupling_pseudo_density_and_contact_shell(
    tmp_path: Path,
) -> None:
    document, controller, view, key, output_root = _setup(tmp_path)
    (job,) = controller.materialize([key])
    job_dir = output_root / job.label
    shutil.copy(FIXTURES / "fe_tetrahedral_muon.magres", job_dir / f"{job.label}.magres")
    _write_spin_density(job_dir / f"{job.label}.den_fmt")
    artifacts = collect_job_artifacts(job_dir, job.label, want_magres=True)
    run = _run(document, job)

    assert controller.validate_artifacts(job, completed_castep_result(), run, artifacts) is None
    contribution = controller.build_result(job, completed_castep_result(), run, artifacts)

    payload = contribution.payload
    assert contribution.warnings == []
    assert payload["isotropic_mhz"] == pytest.approx(-324.2122)
    assert payload["contact_field_T"] == pytest.approx(-1.1946, abs=1e-3)
    assert payload["pseudo_spin_density_bohr3"] == pytest.approx(-0.01 * 0.529177**3, rel=1e-4)
    assert payload["pseudo_contact_field_T"] < 0
    shell = payload["contact_shell"]
    assert shell["host_moments_muB"] == [2.2, 2.1]
    assert shell["muon_cartesian_ang"] == pytest.approx([1.25, 1.25, 1.25])
    assert payload["source_site_id"] == key
    assert payload["charge_state"] == "positive_muon"

    from mufinder.domain.models import Result

    result = Result(calculation_run_id=run.id, result_type=job.calc_type, data=payload)
    document.results[result.id] = result
    controller.refresh_view()
    (row,) = view.results[-1]
    assert row.site == "siteA"
    assert row.isotropic == "-324.21"
    assert row.contact_field == "-1.1946"
    assert row.handoff_reason == ""


def test_build_result_warns_without_mulliken_spins_or_density(tmp_path: Path) -> None:
    document, controller, _view, key, output_root = _setup(tmp_path, mulliken=False)
    (job,) = controller.materialize([key])
    job_dir = output_root / job.label
    shutil.copy(FIXTURES / "fe_tetrahedral_muon.magres", job_dir / f"{job.label}.magres")
    artifacts = collect_job_artifacts(job_dir, job.label, want_magres=True)

    contribution = controller.build_result(
        job, completed_castep_result(), _run(document, job), artifacts
    )

    assert contribution.payload["contact_shell"] is None
    assert contribution.payload["pseudo_contact_field_T"] is None
    assert NO_MULLIKEN_SPINS_WARNING in contribution.warnings
    assert any("pseudo spin density" in warning for warning in contribution.warnings)


def test_materialize_starts_from_the_relaxed_structure_spins(tmp_path: Path) -> None:
    document, controller, _view, key, output_root = _setup(tmp_path)
    (relaxed,) = (
        structure
        for structure in document.structures.values()
        if structure.properties.get("source_workflow_stage") == "muon_relaxation"
    )
    relaxed.properties["atomic_spins"] = [2.2, -2.2, 0.0]

    controller.materialize([key])

    cell = (output_root / "hyperfine_siteA" / "hyperfine_siteA.cell").read_text(encoding="utf-8")
    fe_lines = [line for line in cell.splitlines() if line.startswith("Fe ")]
    assert [line.rsplit(" ", 1)[-1] for line in fe_lines] == ["SPIN=2.2", "SPIN=-2.2"]


def test_results_name_a_pinned_site_as_the_jobs_table_does(tmp_path: Path) -> None:
    from mufinder.domain.models import Result
    from mufinder.services.muon_sites_summary import (
        build_muon_sites_summary_snapshot,
        store_muon_sites_summary_snapshot,
    )

    from tests.support.muon_sites import ClusterSpec, clustered_workspace, store_workspaces
    from tests.support.structures import host_structure

    document, controller, view, key, _output_root = _setup(tmp_path)
    host = host_structure()
    document.structures[host.id] = host
    document.project.host_structure_id = host.id
    relaxation = document.jobs[relaxation_job_id(document, document.sites[UUID(key)])]
    workspace = clustered_workspace(
        host,
        [ClusterSpec(cluster_index=1, representative_job_id=relaxation.id, wyckoff_label="12d")],
    )
    store_workspaces(document, workspace)
    snapshot = build_muon_sites_summary_snapshot(workspace)
    store_muon_sites_summary_snapshot(document, str(snapshot["charge_state"]), snapshot)
    result = Result(
        result_type=HYPERFINE_CALC_TYPE,
        data={
            "source_site_id": key,
            "source_relaxation_job_id": str(relaxation.id),
            "charge_state": "positive_muon",
        },
    )
    document.results[result.id] = result

    controller.refresh_view()

    (source,) = view.sources[-1]
    (row,) = view.results[-1]
    assert source.pinned_label
    assert row.site == source.pinned_label


def test_build_result_ignores_a_density_left_by_an_earlier_run(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta
    import os

    document, controller, _view, key, output_root = _setup(tmp_path)
    (job,) = controller.materialize([key])
    job_dir = output_root / job.label
    shutil.copy(FIXTURES / "fe_tetrahedral_muon.magres", job_dir / f"{job.label}.magres")
    density = job_dir / f"{job.label}.den_fmt"
    _write_spin_density(density)
    old = (datetime.now(UTC) - timedelta(hours=2)).timestamp()
    os.utime(density, (old, old))
    artifacts = collect_job_artifacts(job_dir, job.label, want_magres=True)
    run = _run(document, job)
    run.started_at = datetime.now(UTC)

    contribution = controller.build_result(job, completed_castep_result(), run, artifacts)

    assert artifacts.pseudo_spin_density_angstrom3 is not None
    assert contribution.payload["pseudo_spin_density_bohr3"] is None
    assert contribution.payload["pseudo_contact_field_T"] is None
    assert any("predates this run" in warning for warning in contribution.warnings)


def test_materialize_gives_a_row_its_own_folder_when_another_job_has_that_name(
    tmp_path: Path,
) -> None:
    """A source's label is numbered among the others, so a new row can get a finished job's name."""

    document, controller, _view, key, output_root = _setup(tmp_path)
    taken = output_root / "hyperfine_siteA"
    taken.mkdir(parents=True)
    (taken / "hyperfine_siteA.magres").write_text("another job's result", encoding="utf-8")
    ProjectService().add_job(
        document,
        label="hyperfine_siteA",
        workflow_stage=WorkflowStage.HYPERFINE_COUPLING,
        calc_type=HYPERFINE_CALC_TYPE,
    )

    (job,) = controller.materialize([key])
    (again,) = controller.materialize([key])

    assert job.id == again.id
    assert job.label == "hyperfine_siteA_2"
    assert (output_root / job.label / f"{job.label}.param").is_file()
    assert sorted(path.name for path in taken.iterdir()) == ["hyperfine_siteA.magres"]


# --- running a completed row again with other parameters ---

_LOOSER_TOLERANCE = {"elec_energy_tol": ValueWithUnit(1e-8, "eV")}

_REPLACE_DETAIL = (
    "Running again deletes the hyperfine result and its row in Completed Results, and "
    "clears the job's folder. Until that run's submission is accepted, the row keeps its "
    "current result, and setting the parameters back keeps it for good. A coupling already "
    "used in Magnetic Fields stays as it is until 'Use in Magnetic Fields' is run again on "
    "the new result."
)


def _completed_row(tmp_path: Path, **context):
    """Run a row's magres job to a result, its inputs written as a real run writes them."""

    document, controller, view, key, output_root = _setup(tmp_path, **context)
    (job,) = controller.materialize([key])
    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    result = Result(
        calculation_run_id=run.id,
        result_type=HYPERFINE_CALC_TYPE,
        data={"source_site_id": key, "charge_state": "positive_muon", "isotropic_mhz": 12.5},
    )
    run.result_id = result.id
    document.runs[run.id] = run
    document.results[result.id] = result
    return document, controller, view, key, job, result, output_root / job.label


class _AcceptingConfigDialog:
    """A Configure dialog accepted with its seed values, ``changes`` applied on top."""

    def __init__(self, changes: dict[str, object], **kwargs) -> None:
        self._values = {**kwargs["values"], **changes}
        self._advanced = kwargs["advanced_names"]

    def setWindowTitle(self, _title: str) -> None:  # noqa: N802 - Qt API naming.
        pass

    def exec(self):
        from PySide6.QtWidgets import QDialog

        return QDialog.DialogCode.Accepted

    def deleteLater(self) -> None:  # noqa: N802 - Qt API naming.
        pass

    def collect_values(self) -> dict[str, object]:
        return dict(self._values)

    def advanced_names(self) -> tuple[str, ...]:
        return tuple(self._advanced)

    def untouched_mixed_names(self) -> set[str]:
        return set()


@pytest.fixture
def configure_with(monkeypatch: pytest.MonkeyPatch):
    """Make ``configure_sources`` accept its dialog with the given changes."""

    import mufinder.app.workflow_controllers.hyperfine_coupling as module

    def accept(changes: dict[str, object]) -> None:
        monkeypatch.setattr(
            module,
            "HyperfineCouplingConfigDialog",
            lambda **kwargs: _AcceptingConfigDialog(changes, **kwargs),
        )

    return accept


@pytest.fixture
def replace_questions(monkeypatch: pytest.MonkeyPatch):
    """Record each replace-result question and answer it with ``answer[0]``."""

    from mufinder.app import workflow_table_utils

    asked: list[dict[str, object]] = []
    answer = [True]

    def _confirm(_parent, title, text, detail="", accept_label="Remove", reject_label="Cancel"):
        asked.append(
            {
                "title": title,
                "text": text,
                "detail": detail,
                "accept": accept_label,
                "reject": reject_label,
            }
        )
        return answer[0]

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", _confirm)
    return asked, answer


def _files(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_configure_opens_editable_on_a_completed_row(tmp_path: Path) -> None:
    """A completed row can be configured again, from its current parameters."""

    _document, controller, _view, key, _job, _result, _directory = _completed_row(tmp_path)

    request = controller.configure_request([key])

    assert request is not None
    assert request.read_only is False
    assert request.title == "Configure Hyperfine Coupling Parameters"
    assert controller.build_states()[key].status == "completed"


def test_a_completed_row_whose_inputs_are_missing_opens_read_only(tmp_path: Path) -> None:
    _document, controller, _view, key, job, _result, directory = _completed_row(tmp_path)
    (directory / f"{job.label}.cell").unlink()

    request = controller.configure_request([key])

    assert request.read_only is True
    assert request.title == "Inspect Hyperfine Coupling Parameters"
    assert request.intro_text.endswith(
        "The input files of a selected row's completed magres job are missing, so its "
        "parameters cannot be compared with its result and stay as they are."
    )
    assert "Job Dashboard" not in request.intro_text


def test_an_active_row_stays_locked(tmp_path: Path) -> None:
    from tests.support.jobs import set_job_state

    _document, controller, _view, key, _output_root = _setup(tmp_path)
    (job,) = controller.materialize([key])
    set_job_state(job, "running")

    request = controller.configure_request([key])

    assert request.read_only is True
    assert request.intro_text.endswith(
        "Parameters are read-only while a selected row's magres job is running; "
        "cancel it to edit them."
    )


def test_accepting_a_completed_row_unchanged_asks_nothing_and_stores_nothing(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    persisted: list[str] = []
    document, controller, _view, key, _job, result, directory = _completed_row(
        tmp_path, persisted=persisted
    )
    asked, _answer = replace_questions
    files = _files(directory)
    persisted.clear()
    configure_with({})

    controller.configure_sources([key])

    assert asked == []
    assert persisted == []
    assert workflow_row_configuration(controller.active_definition(), key) == ({}, (), {})
    assert controller.build_states()[key].status == "completed"
    assert result.id in document.results
    assert _files(directory) == files


def _real_dialog(monkeypatch: pytest.MonkeyPatch, changes: dict[str, object]) -> None:
    """Accept the real Configure dialog, with ``changes`` made over what it opened with."""

    import mufinder.app.workflow_controllers.hyperfine_coupling as module
    from mufinder.app.workflow_dialogs import HyperfineCouplingConfigDialog

    class _Accepted(HyperfineCouplingConfigDialog):
        _accepted = False

        def exec(self):
            self._accepted = True
            return 1

        def collect_values(self):
            values = dict(super().collect_values())
            return {**values, **changes} if self._accepted else values

    monkeypatch.setattr(module, "HyperfineCouplingConfigDialog", _Accepted)


def test_the_real_dialog_accepted_unchanged_leaves_a_completed_row_as_it_was(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_questions
) -> None:
    """Defaults the dialog fills in for keywords the run never had are not a change."""

    document, controller, _view, key, _job, result, directory = _completed_row(tmp_path)
    asked, _answer = replace_questions
    files = _files(directory)
    _real_dialog(monkeypatch, {})

    controller.configure_sources([key])

    assert asked == []
    assert workflow_row_configuration(controller.active_definition(), key) == ({}, (), {})
    assert controller.build_states()[key].status == "completed"
    assert result.id in document.results
    assert _files(directory) == files


def test_the_real_dialog_stores_only_the_change_on_a_completed_row(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_questions
) -> None:
    _document, controller, _view, key, _job, _result, _directory = _completed_row(tmp_path)
    asked, _answer = replace_questions
    _real_dialog(monkeypatch, _LOOSER_TOLERANCE)

    controller.configure_sources([key])

    assert len(asked) == 1
    params, _advanced, _extras = workflow_row_configuration(controller.active_definition(), key)
    assert set(params) == {"elec_energy_tol"}
    assert controller.build_states()[key].status == "edited"


def test_changing_a_completed_row_asks_first_and_cancel_keeps_everything(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    document, controller, _view, key, _job, result, directory = _completed_row(tmp_path)
    asked, answer = replace_questions
    answer[0] = False
    files = _files(directory)
    configure_with(_LOOSER_TOLERANCE)

    controller.configure_sources([key])

    assert asked == [
        {
            "title": "Change parameters",
            "text": "Change the parameters of 'siteA, positive muon'? Running the row again "
            "replaces its hyperfine result.",
            "detail": _REPLACE_DETAIL,
            "accept": "Change parameters",
            "reject": "Cancel",
        }
    ]
    assert workflow_row_configuration(controller.active_definition(), key) == ({}, (), {})
    assert controller.build_states()[key].status == "completed"
    assert result.id in document.results
    assert _files(directory) == files


def test_a_confirmed_change_keeps_the_result_until_the_row_runs_again(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    document, controller, view, key, _job, result, directory = _completed_row(tmp_path)
    asked, _answer = replace_questions
    files = _files(directory)
    configure_with(_LOOSER_TOLERANCE)

    controller.configure_sources([key])

    assert len(asked) == 1
    assert controller.build_states()[key].status == "edited"
    (source,) = view.sources[-1]
    assert source.status == "edited"
    # The result keeps its row in Completed Results until the row runs again.
    assert [row.key for row in view.results[-1]] == [str(result.id)]
    request = controller.configure_request([key])
    assert request.read_only is False
    assert request.row_values[0]["elec_energy_tol"] == ValueWithUnit(1e-8, "eV")
    assert _files(directory) == files

    # Changing an edited row again asks nothing more; setting it back makes it completed.
    configure_with({"elec_energy_tol": ValueWithUnit(1e-7, "eV")})
    controller.configure_sources([key])
    assert controller.build_states()[key].status == "edited"
    configure_with({"elec_energy_tol": ValueWithUnit(1e-9, "eV")})
    controller.configure_sources([key])
    assert len(asked) == 1
    assert controller.build_states()[key].status == "completed"
    assert result.id in document.results


def test_an_edited_row_survives_a_save_and_reopen(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    document, controller, _view, key, _job, result, _directory = _completed_row(tmp_path)
    configure_with(_LOOSER_TOLERANCE)
    controller.configure_sources([key])
    service = ProjectService()
    service.save_project(document)

    loaded = service.load_project(document.project_root)
    output_root = document.project_root / "files" / "hyperfine_coupling"
    relaxation_dir = tmp_path / "relax"
    relaxation_ids = {relaxation_job_id(loaded, site) for site in loaded.sites.values()}
    reopened = HyperfineCouplingController(
        make_context(
            document=loaded,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
            job_output_directory=lambda job: (
                relaxation_dir if job.id in relaxation_ids else output_root / job.label
            ),
        ),
        HyperfineCouplingDeps(
            handle_structure_host_mismatch=lambda _m: True,
            handle_site_host_mismatch=lambda _m: True,
        ),
        _FakeView(),  # type: ignore[arg-type]
    )

    assert reopened.build_states()[key].status == "edited"
    assert result.id in loaded.results


def test_running_only_an_unchanged_completed_row_says_how_to_run_it_again(
    tmp_path: Path,
) -> None:
    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    document, controller, _view, key, job, result, _directory = _completed_row(
        tmp_path, infos=infos, local_backend=local
    )

    controller.submit([key], target="local")

    assert infos == [
        (
            "Nothing to run",
            "'siteA, positive muon' has completed. To run it again, change one of its "
            "parameters with 'Configure…' first; running it then replaces its hyperfine result.",
        )
    ]
    assert local.calls == []
    assert job.status == JobStatus.COMPLETED
    assert result.id in document.results


def test_an_edited_rows_job_starts_again_and_a_cancelled_submission_changes_nothing(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    """The edited row's completed job goes to the run-again plan; a cancel changes nothing.

    The accepted submission, with the real jobs controller, is covered in
    ``tests/app/main_window/test_hyperfine_actions.py``.
    """

    local = RecordingBackend(accept=False)
    document, controller, _view, key, job, result, directory = _completed_row(
        tmp_path, local_backend=local
    )
    configure_with(_LOOSER_TOLERANCE)
    controller.configure_sources([key])
    files = _files(directory)

    controller.submit([key], target="local")

    ((submitted, kwargs),) = local.calls
    assert submitted == [job]
    assert job.id in kwargs["run_again"].job_ids
    assert job.status == JobStatus.COMPLETED
    assert result.id in document.results
    assert _files(directory) == files
    assert controller.build_states()[key].status == "edited"
