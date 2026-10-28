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

from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationRun
from mufinder.services.hyperfine_results import NO_MULLIKEN_SPINS_WARNING
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectService
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE

from mufinder.app.workflow_controllers.hyperfine_coupling import (
    HyperfineCouplingController,
    HyperfineCouplingDeps,
)

from tests.support.controllers import completed_castep_result, make_context
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


def _setup(tmp_path: Path, *, mulliken: bool = True):
    service = ProjectService()
    document = service.create_project(tmp_path / "h.mufinder", "H")
    site = add_relaxed_muon_site(document, label="siteA", species=("Fe", "Fe", "H:mu"))
    relaxation = document.jobs[relaxation_job_id(document, site)]
    relaxation_dir = tmp_path / "relax"
    relaxation_dir.mkdir()
    if mulliken:
        (relaxation_dir / f"{relaxation.label}.castep").write_text(_MULLIKEN, encoding="utf-8")
    output_root = tmp_path / "files" / "hyperfine_coupling"

    def job_output_directory(job):
        return relaxation_dir if job.id == relaxation.id else output_root / job.label

    view = _FakeView()
    controller = HyperfineCouplingController(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
            job_output_directory=job_output_directory,
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
    run = CalculationRun(job_id=job.id, status=JobStatus.RUNNING)
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
