"""Tests for :class:`PristineRelaxationController`.

Covers submission, ``build_result`` on a real ``-out.cell``, derived-structure
lookups, row parameter edits, continuation, ``validate_artifacts`` and
definition maintenance.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.pristine_relaxation import PristineRelaxationController
from mufinder.app.workflow_config_codecs import (
    deserialize_pristine_relaxation_config,
    serialize_pristine_relaxation_config,
)

from tests.support.controllers import (
    FakeRelaxationView,
    FakeRowState,
    RecordingBackend,
    make_context,
    muon_host_document,
    pristine_sync_definition,
    stub_pristine_deps,
)


# --------------------------------------------------------------------------- #
# PristineRelaxationController
# --------------------------------------------------------------------------- #
#
# This is the first migration whose ``build_result`` does real work: it parses
# the relaxed ``-out.cell``, contributes the ``lattice_a/b/c`` payload, and
# creates the derived relaxed Structure on the document. The tests below drive
# it with narrow fakes (no MainWindow) and a REAL ``-out.cell`` fixture.


_OUT_CELL_TEXT = """
%BLOCK LATTICE_CART
ang
5.1 0.0 0.0
0.0 5.2 0.0
0.0 0.0 5.3
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
"""


def _pristine_job(service: ProjectService, document: ProjectDocument, label: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )


def test_pristine_controller_handles_only_its_calc_type(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    controller = PristineRelaxationController(
        make_context(document=document), stub_pristine_deps(), FakeRelaxationView()
    )
    handled = _pristine_job(service, document, "pristine_relax_row1")
    muon = service.add_job(
        document,
        label="site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muonium_relaxation",
    )
    # The still-inline muon-relaxation branch must NOT be claimed by this
    # controller — it is dispatched by the inline code, not build_result.
    assert controller.handles_job(handled) is True
    assert controller.handles_job(muon) is False


def test_pristine_controller_uses_workflow_row_ready_predicate() -> None:
    controller = PristineRelaxationController(
        make_context(), stub_pristine_deps(), FakeRelaxationView()
    )
    # Pristine relaxation uses the base-default (wider) workflow-row predicate:
    # "not generated" / "defined" rows are ready, only "failed" is excluded.
    assert controller.is_ready({"r": FakeRowState("not generated")}, "r") is True
    assert controller.is_ready({"r": FakeRowState("ready")}, "r") is True
    assert controller.is_ready({"r": FakeRowState("failed")}, "r") is False


def test_pristine_controller_refresh_view_without_document_clears_rows() -> None:
    view = FakeRelaxationView()
    controller = PristineRelaxationController(
        make_context(document=None), stub_pristine_deps(), view
    )
    controller.refresh_view()
    assert view.runtime_states == [{}]


def test_pristine_controller_build_result_parses_lattice_and_creates_structure(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)

    service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "pristine_relax", "rows": []},
        output_directory=document.project_root / "files" / "pristine_relaxation",
    )

    job = _pristine_job(service, document, "pristine_relax_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # A REAL -out.cell fixture on disk, fed through the controller's parse helper.
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(_OUT_CELL_TEXT, encoding="utf-8")

    deps = stub_pristine_deps()
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = PristineRelaxationController(context, deps, FakeRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())
    artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    result = controller.build_result(job, parsed, run, artifacts)

    # Payload gains the first three lattice parameters from the relaxed cell.
    assert result.payload["lattice_a"] == pytest.approx(5.1)
    assert result.payload["lattice_b"] == pytest.approx(5.2)
    assert result.payload["lattice_c"] == pytest.approx(5.3)

    # No output_files contributed — the call site already passes them generically.
    assert result.output_files == []

    # Exactly one derived structure created on the document, with the right
    # origin and parentage.
    assert len(result.derived_structure_ids) == 1
    derived_id = result.derived_structure_ids[0]
    derived = document.structures[derived_id]
    assert derived.origin == "pristine_relaxation"
    assert derived.parent_structure_id == host.id
    assert derived.properties["source_job_id"] == str(job.id)
    assert derived.properties["source_workflow_stage"] == WorkflowStage.PRISTINE_RELAXATION.value
    assert derived.label == "pristine_relax_row1 relaxed"


def test_pristine_controller_build_result_missing_out_cell_is_empty(tmp_path: Path) -> None:
    """build_result no longer parses or raises on a missing ``-out.cell``: it now
    consumes the pre-parsed carrier and returns an empty ControllerResult when
    ``artifacts.relaxed_cell`` is ``None``. In production the missing-file failure
    is detected by ``validate_artifacts`` from the carrier (the ``cell_missing``
    flag surfaced on :class:`ParsedJobArtifacts`), not by build_result — so
    build_result is never reached with a missing file, and its own contract when
    handed an empty carrier is simply to contribute nothing."""

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _pristine_job(service, document, "pristine_relax_row1")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    deps = stub_pristine_deps()
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: empty_dir,
    )
    controller = PristineRelaxationController(context, deps, FakeRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())
    result = controller.build_result(job, parsed, run, ParsedJobArtifacts())
    assert result.payload == {}
    assert result.derived_structure_ids == []
    # No derived structure was created.
    assert document.structures == {} or all(
        s.origin != "pristine_relaxation" for s in document.structures.values()
    )


_OUT_CELL_TEXT_CONTINUED = """
%BLOCK LATTICE_CART
ang
5.4 0.0 0.0
0.0 5.5 0.0
0.0 0.0 5.6
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.01 0.02 0.03
%ENDBLOCK POSITIONS_FRAC
"""


def test_pristine_controller_recompletion_updates_derived_structure_in_place(
    tmp_path: Path,
) -> None:
    """A continued pristine job completes twice. The second completion must
    refresh the SAME derived structure rather than mint a duplicate: a
    comparison table holds ``dft_structure_ids``, so a replacement structure
    would strand that reference on a superseded geometry."""

    from mufinder.domain.models import ComparisonTable, Result, Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "pristine_relax", "rows": []},
        output_directory=document.project_root / "files" / "pristine_relaxation",
    )

    job = _pristine_job(service, document, "pristine_relax_row1")
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    out_cell = output_dir / f"{job.label}-out.cell"

    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = PristineRelaxationController(
        context,
        stub_pristine_deps(),
        FakeRelaxationView(),
    )
    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())

    def complete(run_index: int, cell_text: str):
        run = CalculationRun(job_id=job.id, run_index=run_index)
        document.runs[run.id] = run
        out_cell.write_text(cell_text, encoding="utf-8")
        artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
        controller_result = controller.build_result(job, parsed, run, artifacts)
        result = Result(
            calculation_run_id=run.id,
            result_type=job.calc_type,
            derived_structure_ids=list(controller_result.derived_structure_ids),
        )
        document.results[result.id] = result
        run.result_id = result.id
        return run, result

    _first_run, first_result = complete(1, _OUT_CELL_TEXT)
    first_id = first_result.derived_structure_ids[0]
    structure_count = len(document.structures)

    # A comparison table built from the first completion — the reference the
    # update-in-place behaviour exists to protect.
    table = ComparisonTable(
        label="Comparison 1",
        experimental_structure_id=host.id,
        dft_structure_ids=[first_id],
        dft_labels=[job.label],
    )
    service.add_comparison_table(document, table)

    _second_run, second_result = complete(2, _OUT_CELL_TEXT_CONTINUED)

    # Same structure id, no duplicate structure, both Results converge on it.
    assert second_result.derived_structure_ids == [first_id]
    assert len(document.structures) == structure_count
    assert first_result.derived_structure_ids == second_result.derived_structure_ids

    # The structure now carries the SECOND run's geometry.
    derived = document.structures[first_id]
    assert derived.lattice_vectors == [[5.4, 0.0, 0.0], [0.0, 5.5, 0.0], [0.0, 0.0, 5.6]]
    assert derived.fractional_positions == [pytest.approx([0.01, 0.02, 0.03])]
    assert len(derived.species) == len(derived.fractional_positions)
    assert derived.label == "pristine_relax_row1 relaxed"
    assert derived.properties["source_job_id"] == str(job.id)
    assert derived.parent_structure_id == host.id

    # Every reference — the comparison table's and both Results' — still resolves.
    assert table.dft_structure_ids == [first_id]
    assert service.validate_document(document) == []


def test_pristine_controller_recompletion_refreshes_result_payload(
    tmp_path: Path,
) -> None:
    """The re-completion's payload comes from the new ``-out.cell``, so the
    workflow row reports the continued run's lattice, not the first run's."""

    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)

    job = _pristine_job(service, document, "pristine_relax_row1")
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    out_cell = output_dir / f"{job.label}-out.cell"
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = PristineRelaxationController(context, stub_pristine_deps(), FakeRelaxationView())
    parsed = CastepResult(completed=True, total_energy=-11.0, warnings=())

    out_cell.write_text(_OUT_CELL_TEXT, encoding="utf-8")
    run_one = CalculationRun(job_id=job.id, run_index=1)
    document.runs[run_one.id] = run_one
    controller.build_result(
        job, parsed, run_one, collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    )

    out_cell.write_text(_OUT_CELL_TEXT_CONTINUED, encoding="utf-8")
    run_two = CalculationRun(job_id=job.id, run_index=2)
    document.runs[run_two.id] = run_two
    second = controller.build_result(
        job, parsed, run_two, collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    )

    assert second.payload["lattice_a"] == pytest.approx(5.4)
    assert second.payload["lattice_b"] == pytest.approx(5.5)
    assert second.payload["lattice_c"] == pytest.approx(5.6)


def test_find_job_derived_structure_falls_back_to_earlier_run_results(
    tmp_path: Path,
) -> None:
    """A derived structure written before ``properties["source_job_id"]`` existed
    is still found — through the ``derived_structure_ids`` of a Result reachable
    from one of the job's own runs — so it is refreshed rather than duplicated."""

    from mufinder.services.job_results import find_job_derived_structure
    from mufinder.domain.models import Result, Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _pristine_job(service, document, "pristine_relax_row1")

    legacy = Structure(
        label="pristine_relax_row1 relaxed",
        origin="pristine_relaxation",
        lattice_vectors=[[5.1, 0.0, 0.0], [0.0, 5.2, 0.0], [0.0, 0.0, 5.3]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, legacy)
    assert "source_job_id" not in legacy.properties

    run = CalculationRun(job_id=job.id, run_index=1)
    document.runs[run.id] = run
    result = Result(calculation_run_id=run.id, derived_structure_ids=[legacy.id])
    document.results[result.id] = result
    run.result_id = result.id

    found = find_job_derived_structure(document, job, origin="pristine_relaxation")
    assert found is legacy

    # A structure belonging to a DIFFERENT job is never reached by the fallback.
    other_job = _pristine_job(service, document, "pristine_relax_row2")
    assert find_job_derived_structure(document, other_job, origin="pristine_relaxation") is None


def test_find_job_derived_structure_prefers_the_first_of_a_duplicate_pair(
    tmp_path: Path,
) -> None:
    """A job that completed twice before update-in-place existed left a duplicate
    pair. The lookup resolves to the FIRST in document order — the one any
    comparison table built in the meantime already references."""

    from mufinder.services.job_results import find_job_derived_structure
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _pristine_job(service, document, "pristine_relax_row1")

    first = Structure(
        label="pristine_relax_row1 relaxed",
        origin="pristine_relaxation",
        properties={"source_job_id": str(job.id)},
    )
    second = Structure(
        label="pristine_relax_row1 relaxed",
        origin="pristine_relaxation",
        properties={"source_job_id": str(job.id)},
    )
    service.add_structure(document, first)
    service.add_structure(document, second)

    assert find_job_derived_structure(document, job, origin="pristine_relaxation") is first
    # Origin is part of the key: a muon-origin structure is never returned for a
    # pristine lookup even when it names the same job.
    assert find_job_derived_structure(document, job, origin="muon_relaxation") is None


def test_pristine_controller_materialize_writes_job_ids_by_label(tmp_path: Path) -> None:
    from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
    from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(PristineRelaxationRow(label="row1"),),
    )
    output_root = document.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "pristine_relax",
            "rows": [{"label": "row1", "parameter_values": {}}],
        },
        output_directory=output_root,
    )

    def expand(defn, cfg):
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=cfg,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )

    deps = stub_pristine_deps(
        deserialize_pristine_relaxation_config=lambda _payload: config,
        expand_pristine_relaxation_jobs=expand,
    )
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = PristineRelaxationController(context, deps, FakeRelaxationView())

    jobs = controller.materialize(["row1"])

    assert len(jobs) == 1
    job = jobs[0]
    assert job.calc_type == "castep_pristine_relaxation"
    mapping = definition.parameters["job_ids_by_label"]
    assert mapping["row1"] == str(job.id)
    # Rendered inputs were written under the output root.
    job_dir = output_root / job.label
    assert (job_dir / f"{job.label}.cell").exists()
    assert (job_dir / f"{job.label}.param").exists()


def test_pristine_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
    from mufinder.app.relaxation_types import PristineRelaxationConfig, PristineRelaxationRow
    from mufinder.domain.models import Structure

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(PristineRelaxationRow(label="row1"),),
    )
    output_root = document.project_root / "files" / "pristine_relaxation"
    service.register_pristine_relaxation_definition(
        document,
        host_structure_id=host.id,
        parameters={"name_prefix": "pristine_relax", "rows": [{"label": "row1"}]},
        output_directory=output_root,
    )

    def expand(defn, cfg):
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=cfg,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )

    deps = stub_pristine_deps(
        deserialize_pristine_relaxation_config=lambda _payload: config,
        pristine_relaxation_form_config=lambda: config,
        pristine_relaxation_row_signature=lambda row: (row.label,),
        expand_pristine_relaxation_jobs=expand,
    )

    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = PristineRelaxationController(context, deps, FakeRelaxationView())

    controller.submit(["row1"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    submitted, extras = local.calls[0]
    assert extras == {}
    assert len(submitted) == 1
    assert submitted[0].calc_type == "castep_pristine_relaxation"


def _pristine_parameter_edit_fixture(service: ProjectService, tmp_path: Path):
    """Definition with two rows, ``row1`` bound to a job, and a matching form.

    This is the Continue-relaxation precondition: a submitted row whose
    ``job_ids_by_label`` entry must survive a parameter edit.
    """

    from mufinder.app.relaxation_adapter import expand_pristine_relaxation_jobs
    from mufinder.app.relaxation_types import (
        PristineRelaxationConfig,
        PristineRelaxationRow,
    )
    from mufinder.domain.models import Structure

    document = service.create_project(tmp_path / "p.mufinder", "P")
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, host)
    document.project.host_structure_id = host.id

    config = PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(label="row1", parameter_values={"geom_max_iter": 120}),
            PristineRelaxationRow(label="row2", parameter_values={"geom_max_iter": 400}),
        ),
    )
    output_root = document.project_root / "files" / "pristine_relaxation"
    definition = service.register_pristine_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters=serialize_pristine_relaxation_config(config),
        output_directory=output_root,
    )
    job = _pristine_job(service, document, "pristine_relax_row1")
    definition.parameters["job_ids_by_label"] = {"row1": str(job.id)}

    view = FakeRelaxationView(
        {
            "name_prefix": "pristine_relax",
            "rows": [
                {
                    "label": "row1",
                    "fix_all_cell": False,
                    "supercell_matrix": None,
                    "params": {"geom_max_iter": 120},
                    "added_advanced": [],
                },
                {
                    "label": "row2",
                    "fix_all_cell": False,
                    "supercell_matrix": None,
                    "params": {"geom_max_iter": 400},
                    "added_advanced": [],
                },
            ],
        }
    )

    def expand(_defn, cfg):
        return expand_pristine_relaxation_jobs(
            host_structure=host,
            config=cfg,
            project_cutoff_energy=None,
            project_kpoint_grid=None,
            project_kpoint_reference_lattice=None,
        )

    deps = stub_pristine_deps(
        deserialize_pristine_relaxation_config=deserialize_pristine_relaxation_config,
        pristine_relaxation_form_config=lambda: deserialize_pristine_relaxation_config(
            view.collect_state()
        ),
        pristine_relaxation_row_signature=lambda row: (
            tuple(sorted(row.parameter_values.items())),
            tuple(row.added_advanced),
        ),
        expand_pristine_relaxation_jobs=expand,
    )
    return document, definition, job, deps, view, output_root


def _pristine_edit_controller(service, document, deps, view, output_root, persists):
    import dataclasses

    context = dataclasses.replace(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _defn: output_root,
        ),
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    return PristineRelaxationController(context, deps, view)


def test_pristine_controller_update_row_parameters_preserves_row_job_binding(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    definitions_before = set(document.calculations)
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)

    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is True

    # No new definition was registered: the binding lives on this very object.
    assert set(document.calculations) == definitions_before
    assert document.calculations[definition.id] is definition
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    assert controller.jobs_for_keys(["row1"]) == [job]

    config = deserialize_pristine_relaxation_config(definition.parameters["configuration"])
    rows = {row.label: row for row in config.rows}
    assert rows["row1"].parameter_values == {"geom_max_iter": 500}
    assert rows["row2"].parameter_values == {"geom_max_iter": 400}

    # The form state was mirrored so the row signatures still match.
    form_rows = {row["label"]: row for row in view.form_state["rows"]}
    assert form_rows["row1"]["params"] == {"geom_max_iter": 500}
    assert form_rows["row2"]["params"] == {"geom_max_iter": 400}
    assert document.ui_state["pristine_relaxation_form"] == view.form_state
    assert persists == [
        {"failure_message": ("Pristine relaxation parameters were updated, but autosave failed.")}
    ]


def test_pristine_controller_update_row_parameters_rejects_unknown_row(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    before = copy.deepcopy(definition.parameters["configuration"])
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)

    assert controller.update_row_parameters("row9", {"geom_max_iter": 500}) is False

    assert definition.parameters["configuration"] == before
    assert persists == []


def test_pristine_controller_update_row_parameters_edits_a_later_unbound_row(
    tmp_path: Path,
) -> None:
    """Editing the second (jobless) row leaves the first row — and its job
    binding — alone in both the definition and the form."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)

    assert controller.update_row_parameters("row2", {"geom_max_iter": 700}) is True

    config = deserialize_pristine_relaxation_config(definition.parameters["configuration"])
    rows = {row.label: row for row in config.rows}
    assert rows["row1"].parameter_values == {"geom_max_iter": 120}
    assert rows["row2"].parameter_values == {"geom_max_iter": 700}
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    form_rows = {row["label"]: row for row in view.form_state["rows"]}
    assert form_rows["row1"]["params"] == {"geom_max_iter": 120}
    assert form_rows["row2"]["params"] == {"geom_max_iter": 700}


def test_pristine_controller_update_row_parameters_without_definition_or_config(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []

    no_definition = stub_pristine_deps()
    controller = _pristine_edit_controller(
        service, document, no_definition, view, output_root, persists
    )
    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is False

    unparseable = stub_pristine_deps(
        deserialize_pristine_relaxation_config=lambda _payload: None,
    )
    controller = _pristine_edit_controller(
        service, document, unparseable, view, output_root, persists
    )
    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is False
    assert persists == []


def test_pristine_controller_update_row_parameters_without_the_row_in_the_form(
    tmp_path: Path,
) -> None:
    """With no form row to mirror into, the caller's own values are stored: the
    row is not projected from the form either, so no signature can desync."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    view.form_state = {"name_prefix": "pristine_relax", "rows": []}
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)

    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is True

    config = deserialize_pristine_relaxation_config(definition.parameters["configuration"])
    rows = {row.label: row for row in config.rows}
    assert rows["row1"].parameter_values == {"geom_max_iter": 500}
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    assert view.form_state["rows"] == []
    assert "pristine_relaxation_form" not in document.ui_state


def test_pristine_controller_materialize_after_edit_rewrites_same_directory(
    tmp_path: Path,
) -> None:
    """The Continue cycle: render, finish, edit parameters, return to DEFINED,
    re-materialize. The row keeps its job and its directory; only the rendered
    ``.param`` changes."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)

    assert controller.materialize(["row1"]) == [job]
    param_path = output_root / job.label / f"{job.label}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")

    job.status = JobStatus.COMPLETED
    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is True
    # WP1's prepare_job_continuation returns the job to DEFINED (with its
    # remote_directory preserved) before the directory is rewritten.
    job.status = JobStatus.DEFINED

    assert controller.materialize(["row1"]) == [job]

    assert list(document.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered


def test_pristine_controller_build_states_keeps_job_after_parameter_edit(
    tmp_path: Path,
) -> None:
    """Finding 5: pristine relaxation has no active-status escape hatch, so a
    form/definition signature mismatch would drop the row to a jobless "ready"."""

    service = ProjectService()
    document, _definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    job.status = JobStatus.RUNNING
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)
    before = controller.build_states()
    assert before["row1"].job_id == job.id

    assert controller.update_row_parameters("row1", {"geom_max_iter": 500}) is True

    after = controller.build_states()
    assert after["row1"].job_id == job.id
    assert after["row1"].status == JobStatus.RUNNING.value


def test_pristine_controller_row_is_continuable_reflects_raw_job_status(
    tmp_path: Path,
) -> None:
    """Finding 8: gating reads the job's real status, never displayed text —
    a dismissed FAILED job (which *displays* "ready") is still continuable."""

    from mufinder.domain.models import utc_now

    service = ProjectService()
    document, _definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)

    job.status = JobStatus.RUNNING
    assert controller.row_is_continuable("row1") is False

    for status in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.STOPPED,
        JobStatus.CANCELLED,
    ):
        job.status = status
        assert controller.row_is_continuable("row1") is True

    job.status = JobStatus.FAILED
    job.dismissed_at = utc_now()
    assert controller.row_is_continuable("row1") is True

    # row2 has no bound job; an unknown key resolves to no job either.
    assert controller.row_is_continuable("row2") is False
    assert controller.row_is_continuable("unknown") is False


def test_pristine_controller_relaxed_structure_export_follows_the_rows_job(
    tmp_path: Path,
) -> None:
    """The row menu's export resolves the row's job and its derived structure."""

    from mufinder.domain.models import Structure
    from mufinder.services.structure_export import NO_STRUCTURE_REASON, NOT_COMPLETED_REASON

    service = ProjectService()
    document, _definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    controller = _pristine_edit_controller(service, document, deps, view, output_root, [])

    job.status = JobStatus.RUNNING
    assert controller.relaxed_structure_export("row1").unavailable_reason == NOT_COMPLETED_REASON

    job.status = JobStatus.COMPLETED
    assert controller.relaxed_structure_export("row1").unavailable_reason == NO_STRUCTURE_REASON

    relaxed = Structure(
        label="row1 relaxed",
        origin="pristine_relaxation",
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
        properties={"source_job_id": str(job.id)},
    )
    service.add_structure(document, relaxed)
    export = controller.relaxed_structure_export("row1")
    assert export.structure is relaxed
    assert export.default_stem == f"{job.label}_relaxed".lower()

    # A row with no bound job has nothing to export.
    assert controller.relaxed_structure_export("row2").structure is None


def test_pristine_controller_job_files_exports_lists_the_rows_job_files(tmp_path: Path) -> None:
    """Open Containing Folder / Export Inputs/Outputs resolve the row's job folder."""

    import dataclasses

    from mufinder.services.job_file_export import NO_FOLDER_REASON

    service = ProjectService()
    document, _definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    job_dir = tmp_path / "jobs" / job.label
    controller = _pristine_edit_controller(service, document, deps, view, output_root, [])
    controller._context = dataclasses.replace(
        controller._context, job_output_directory=lambda _job: job_dir
    )

    [missing] = controller.job_files_exports(["row1"])
    assert missing.job is job
    assert missing.unavailable_reason == NO_FOLDER_REASON

    job_dir.mkdir(parents=True)
    for suffix in (".cell", ".castep", ".check"):
        (job_dir / f"{job.label}{suffix}").write_text(suffix, encoding="utf-8")
    [export] = controller.job_files_exports(["row1"])
    assert export.directory == job_dir
    assert [path.name for path in export.files] == [f"{job.label}.castep", f"{job.label}.cell"]

    # A row with no bound job contributes nothing.
    assert controller.job_files_exports(["row2"]) == []


def test_pristine_controller_render_row_param_text_previews_without_persisting(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)
    before = copy.deepcopy(definition.parameters["configuration"])

    text = controller.render_row_param_text("row1", {"geom_max_iter": 777}, ())

    assert "GEOM_MAX_ITER : 777" in text
    # Nothing was persisted: this is a pure preview.
    assert definition.parameters["configuration"] == before

    assert controller.render_row_param_text("unknown", {}, ()) == ""


def test_pristine_controller_continue_job_returns_to_defined_and_rewrites_directory(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import utc_now

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)
    controller.materialize(["row1"])
    param_path = output_root / job.label / f"{job.label}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    job.execution_target = "myhpc"
    job.remote_directory = "/scratch/user/row1"
    job.remote_job_id = "12345"
    job.error_message = "boom"
    job.dismissed_at = utc_now()

    continued = controller.continue_job("row1", {"geom_max_iter": 500})

    assert continued is job
    assert job.status == JobStatus.DEFINED
    # remote_directory is deliberately preserved; everything else the previous
    # run stamped is cleared.
    assert job.remote_directory == "/scratch/user/row1"
    assert job.remote_job_id is None
    assert job.error_message is None
    assert job.dismissed_at is None

    # Same directory rewritten, no duplicate job minted.
    assert list(document.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {"row1": str(job.id)}
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered

    continuation = next(
        record for record in document.provenance if record.action == "job_continuation"
    )
    assert continuation.job_id == job.id
    assert continuation.parameters["remote_directory"] == "/scratch/user/row1"
    assert continuation.parameters["preserved_remote_directory"] is True


def test_pristine_controller_continue_job_rejects_non_continuable_job_without_mutating(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)
    before = copy.deepcopy(definition.parameters["configuration"])

    job.status = JobStatus.RUNNING
    assert controller.continue_job("row1", {"geom_max_iter": 500}) is None

    # Untouched: the row's parameters were never mutated, since the job wasn't
    # continuable in the first place.
    assert job.status == JobStatus.RUNNING
    assert definition.parameters["configuration"] == before
    assert not any(record.action == "job_continuation" for record in document.provenance)


def test_pristine_controller_continue_job_unknown_row_returns_none(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, _definition, _job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)

    assert controller.continue_job("unknown", {"geom_max_iter": 500}) is None


def test_pristine_controller_continue_job_returns_none_when_materialize_fails(
    tmp_path: Path,
) -> None:
    """Audit finding 1: materialize() can swallow an expansion error and return
    ``[]`` without raising. If continue_job ignored that, the job would be left
    DEFINED with a preserved remote_directory while the on-disk .param still
    held the PREVIOUS run's parameters -- and the caller (which only checks
    "is the return value truthy") would submit that stale directory as if the
    new parameters had been written."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _pristine_edit_controller(service, document, deps, view, output_root, persists)
    controller.materialize(["row1"])  # write the real initial directory contents
    param_path = output_root / job.label / f"{job.label}.param"
    original_rendered = param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    job.remote_directory = "/scratch/user/row1"

    # Force the continuation's own materialize call to fail exactly the way
    # the real one does on a caught ValueError/StructureBridgeError: return []
    # without raising.
    controller.materialize = lambda keys: []

    result = controller.continue_job("row1", {"geom_max_iter": 500})

    assert result is None
    # The row's stored parameters were still mutated (update_row_parameters ran
    # before the failed materialize) and the job was still moved to DEFINED
    # (prepare_job_continuation ran) -- it stays recoverable -- but the ON-DISK
    # .param is untouched: this is exactly why the caller must not submit.
    config = deserialize_pristine_relaxation_config(definition.parameters["configuration"])
    assert {row.label: row.parameter_values for row in config.rows}["row1"] == {
        "geom_max_iter": 500
    }
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == "/scratch/user/row1"
    assert param_path.read_text(encoding="utf-8") == original_rendered
    # No provenance milestone either: the journal must not claim a continuation
    # that was never rendered and (because continue_job returned None) is never
    # submitted. record_job_continuation therefore runs only after materialize.
    assert not any(record.action == "job_continuation" for record in document.provenance)


def test_pristine_controller_continue_job_leaves_row_untouched_when_document_disappears(
    tmp_path: Path,
) -> None:
    """Audit finding 5: the document lookup must run BEFORE
    update_row_parameters. ``jobs_for_keys`` reads the document once to resolve
    the job (call #1, succeeds); this context then returns ``None`` on the
    NEXT call (call #2), simulating the document disappearing between menu-open
    and dialog-accept (e.g. the project was closed/switched). If
    update_row_parameters ran first, the row's stored parameters would be
    mutated before continue_job discovered there was no document to record
    provenance against."""

    import dataclasses

    service = ProjectService()
    document, definition, job, deps, view, output_root = _pristine_parameter_edit_fixture(
        service, tmp_path
    )
    job.status = JobStatus.FAILED
    before_config = copy.deepcopy(definition.parameters["configuration"])

    calls = {"n": 0}

    def flaky_document():
        calls["n"] += 1
        return document if calls["n"] == 1 else None

    persists: list[dict] = []
    context = dataclasses.replace(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _defn: output_root,
        ),
        document=flaky_document,
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    controller = PristineRelaxationController(context, deps, view)

    result = controller.continue_job("row1", {"geom_max_iter": 500})

    assert result is None
    assert calls["n"] >= 2
    assert definition.parameters["configuration"] == before_config
    assert job.status == JobStatus.FAILED


# --------------------------------------------------------------------------- #
# validate_artifacts: per-controller completion failures
# --------------------------------------------------------------------------- #
#
# Each controller rejects a completion whose outputs cannot carry a result
# before build_result runs. The checks read the pre-parsed carrier; these tests
# pin every failure message the job-finished handler records.


def test_pristine_controller_validate_artifacts_messages(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    service = ProjectService()
    document = service.create_project(tmp_path / "p.mufinder", "P")
    job = _pristine_job(service, document, "pristine_relax_row1")
    run = CalculationRun(job_id=job.id)
    controller = PristineRelaxationController(
        make_context(document=document), stub_pristine_deps(), FakeRelaxationView()
    )
    converged = CastepResult(completed=True, total_energy=-1.0, geom_converged=True)
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(_OUT_CELL_TEXT, encoding="utf-8")
    fresh = collect_job_artifacts(output_dir, job.label, want_out_cell=True)

    # Geometry did not converge: CASTEP's own message, else the generic one.
    assert (
        controller.validate_artifacts(
            job,
            CastepResult(completed=True, geom_converged=False, error_message="BFGS gave up"),
            run,
            fresh,
        )
        == "BFGS gave up"
    )
    assert (
        controller.validate_artifacts(
            job, CastepResult(completed=True, geom_converged=False), run, fresh
        )
        == "Geometry optimization did not satisfy CASTEP convergence criteria."
    )
    assert (
        controller.validate_artifacts(job, converged, run, ParsedJobArtifacts(cell_missing=True))
        == "Expected relaxed structure output was not found: pristine_relax_row1-out.cell"
    )
    assert (
        controller.validate_artifacts(
            job, converged, run, ParsedJobArtifacts(cell_error="bad block")
        )
        == "Failed to parse relaxed CELL output: bad block"
    )

    # Fails open with no start time; a file a whole run older than the start is stale.
    assert controller.validate_artifacts(job, converged, run, fresh) is None
    run.started_at = datetime.now(UTC) + timedelta(minutes=5)
    assert controller.validate_artifacts(job, converged, run, fresh) == (
        "pristine_relax_row1-out.cell predates this run: no relaxed structure "
        "was written, so the previous run's geometry is still in place."
    )


def test_pristine_controller_remove_rows_reports_deleted_definitions(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    emptied = pristine_sync_definition(service, document, host, job_ids_by_label={"row_1": "1"})
    kept = pristine_sync_definition(
        service,
        document,
        host,
        rows=("row_1", "row_2"),
        job_ids_by_label={"row_1": "1", "row_2": "2"},
    )
    controller = PristineRelaxationController(
        make_context(document=document),
        stub_pristine_deps(
            deserialize_pristine_relaxation_config=deserialize_pristine_relaxation_config
        ),
        FakeRelaxationView(),
    )
    document.ui_state[controller.active_key] = str(emptied.id)

    assert controller.remove_rows(["row_1"]) == (emptied.id,)

    assert emptied.id not in document.calculations
    assert controller.active_key not in document.ui_state
    assert [row["label"] for row in kept.parameters["configuration"]["rows"]] == ["row_2"]
    assert kept.parameters["job_ids_by_label"] == {"row_2": "2"}


def test_remove_rows_without_a_document_removes_nothing() -> None:
    controller = PristineRelaxationController(
        make_context(), stub_pristine_deps(), FakeRelaxationView()
    )

    assert controller.remove_rows(["row_1"]) == ()
