"""Tests for :class:`EnergyBarriersController`.

Covers the lazily created view, the ``.ts`` result payload, source-key
materialization, ``NUM_FARMS`` rendering and ``validate_artifacts``.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest

from mufinder.services.muon_sites_summary import ENERGY_BARRIERS_SCOPE, set_show_every_relaxed_site
from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.domain.models import CalculationDefinition as _CalculationDefinition
from mufinder.domain.models import MuonSite as _MuonSite
from mufinder.domain.models import Structure as _Structure
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
)

from mufinder.app.workflow_controllers.energy_barriers import (
    EnergyBarriersController,
    EnergyBarriersDeps,
)

from tests.support.controllers import (
    configure_result,
    RecordingBackend,
    completed_castep_result,
    make_context,
    no_artifacts,
)


# --------------------------------------------------------------------------- #
# Energy barriers (NEB): the LAST of the seven migrations. Distinguished by
# source-key provenance, IDPP intermediate interpolation in materialize, a ``.ts``
# transition-state ``build_result`` payload, and a LAZILY-CREATED view reached via
# a ``view_getter`` (the only sanctioned constructor-shape deviation). These tests
# pin: the view_getter no-ops before creation (refresh and result dispatch both
# work with no view), the ``.ts`` payload, and source-key materialize.
# --------------------------------------------------------------------------- #

_NEB_TS_FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "neb_path.ts"


class _FakeEnergyBarriersView:
    """Minimal EnergyBarriersWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.endpoint_calls: list[tuple] = []
        self.sources: list[list] = []
        self.neb_results: list[list] = []
        self.site_scopes: list[dict] = []
        self.neb_previews: list[dict] = []
        self.cleared_previews: list[str | None] = []

    def set_site_scope(self, **kwargs) -> None:
        self.site_scopes.append(kwargs)

    def set_endpoint_options(self, rows, **kwargs) -> None:
        self.endpoint_calls.append((list(rows), kwargs))

    def set_sources(self, rows) -> None:
        self.sources.append(list(rows))

    def set_neb_results(self, rows) -> None:
        self.neb_results.append(list(rows))

    def set_neb_preview(self, **kwargs) -> None:
        self.neb_previews.append(kwargs)

    def clear_neb_preview(self, title: str | None = None) -> None:
        self.cleared_previews.append(title)


def _stub_energy_barriers_deps(**overrides) -> EnergyBarriersDeps:
    base = dict(
        handle_structure_host_mismatch=lambda _mismatch: True,
        handle_site_host_mismatch=lambda _mismatch: True,
        show_warning=lambda _title, _text: None,
    )
    base.update(overrides)
    return EnergyBarriersDeps(**base)


def _neb_job(service: ProjectService, document: ProjectDocument, label: str, calc_type: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=calc_type,
    )


def test_energy_barriers_controller_handles_both_ts_and_neb_calc_types(tmp_path: Path) -> None:
    """Both the NEB and transition-state-search calc types are owned so the
    ``.ts``-producing pair is coherent for result dispatch."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    controller = EnergyBarriersController(
        make_context(document=document),
        _stub_energy_barriers_deps(),
        lambda: _FakeEnergyBarriersView(),
    )
    neb = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    tss = _neb_job(service, document, "barrier_tss", TRANSITION_STATE_SEARCH_CALC_TYPE)
    other = service.add_job(
        document,
        label="u",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_uep_source",
    )
    assert controller.handles_job(neb) is True
    assert controller.handles_job(tss) is True
    assert controller.handles_job(other) is False


def test_energy_barriers_controller_refresh_view_noops_when_view_never_created() -> None:
    """The sanctioned lazy-view deviation: when the getter returns ``None`` (the
    view was never opened), refresh_view is a strict no-op — it never touches any
    view. This is what lets refresh run safely before the workflow route opens."""

    calls: list[str] = []

    def _getter():
        calls.append("getter")
        return None

    controller = EnergyBarriersController(
        make_context(document=None), _stub_energy_barriers_deps(), _getter
    )
    controller.refresh_view()  # must not raise, must not build any rows
    assert calls == ["getter"]  # getter consulted, then no-op


def test_energy_barriers_controller_refresh_view_clears_when_document_none() -> None:
    """With a real (lazily-created) view but no document, refresh clears all three
    tables — distinct from the getter-None no-op above."""

    view = _FakeEnergyBarriersView()
    controller = EnergyBarriersController(
        make_context(document=None), _stub_energy_barriers_deps(), lambda: view
    )
    controller.refresh_view()
    assert view.endpoint_calls[-1][0] == []
    assert view.sources == [[]]
    assert view.neb_results == [[]]


def test_energy_barriers_controller_refresh_view_renders_sources_and_results(
    tmp_path: Path,
) -> None:
    """With a view and a document, refresh renders endpoint options, one source
    row, and one completed-NEB result row — exercising the full render body
    (source-row build + result-row build with barrier summaries / charge label)."""

    from mufinder.domain.models import Result as _Result

    service = ProjectService()
    document, definition, _source, output_root = _seed_energy_barriers_document(
        service, tmp_path, row_settings={"num_farms": 3}
    )
    set_show_every_relaxed_site(document, ENERGY_BARRIERS_SCOPE, True)
    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"]["source-key"] = str(job.id)
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    result = _Result(
        calculation_run_id=run.id,
        result_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
        data={
            "num_images": 7,
            "transition_state_file": "/tmp/barrier.ts",
            "image_energies_eV": [-1.0, -0.8, -0.9],
        },
    )
    run.result_id = result.id
    document.runs[run.id] = run
    document.results[result.id] = result

    view = _FakeEnergyBarriersView()
    controller = _neb_controller(service, document, output_root, view_getter=lambda: view)

    controller.refresh_view()

    endpoint_rows, endpoint_options = view.endpoint_calls[-1]
    assert [row.label for row in endpoint_rows] == ["siteA", "siteB"]
    assert endpoint_options["reactant_key"] == endpoint_rows[0].key
    assert endpoint_options["product_key"] == endpoint_rows[1].key
    source_row = view.sources[-1][0]
    assert (source_row.reactant, source_row.product) == ("siteA", "siteB")
    assert source_row.charge_state == "positive muon"
    assert source_row.path_length == "2.5 Å"
    assert source_row.farms == "3"
    assert source_row.configured is True
    assert source_row.status == "completed"
    result_row = view.neb_results[-1][0]
    assert result_row.label == "siteA → siteB"
    assert result_row.images == "7"
    assert result_row.charge_state == "positive muon"
    assert result_row.path_length == "2.5 Å"
    assert (result_row.barrier_forward, result_row.barrier_reverse, result_row.delta_energy) == (
        "0.2 eV",
        "0.1 eV",
        "0.1 eV",
    )
    # The result row carries the producing job's status so the results table can colour it.
    assert result_row.status == "completed"


def test_energy_barriers_controller_build_result_ts_payload(tmp_path: Path) -> None:
    """build_result parses the completed ``.ts`` file and contributes the
    transition-state payload — the fields the former inline result block assigned.
    Uses the real ``neb_path.ts`` fixture so the science path matches the inline
    guard's parse."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    run = CalculationRun(job_id=job.id)

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    ts_path = output_dir / "barrier.ts"
    ts_path.write_text(_NEB_TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    deps = _stub_energy_barriers_deps()
    controller = EnergyBarriersController(
        make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: output_dir,
            job_seed_name=lambda _job: "barrier",
        ),
        deps,
        lambda: None,  # view never created: result dispatch must still work
    )

    artifacts = collect_job_artifacts(output_dir, "barrier", want_ts=True)
    result = controller.build_result(job, completed_castep_result(), run, artifacts)

    payload = result.payload
    assert payload["transition_state_file"] == str(ts_path)
    assert payload["num_images"] == 7
    assert payload["image_labels"] == ["REA", "TST", "TST", "TST", "TST", "TST", "PRO"]
    assert payload["barrier_height_eV"] == pytest.approx(0.1214, abs=2e-3)
    assert len(payload["reaction_coordinates"]) == 7
    assert len(payload["image_energies_eV"]) == 7
    # build_result contributes no output files or derived structures.
    assert result.output_files == []
    assert result.derived_structure_ids == []


def test_energy_barriers_controller_show_neb_preview_structures_are_in_angstrom(
    tmp_path: Path,
) -> None:
    """show_neb_preview hands the view one ``Structure`` per image, with the
    ``.ts`` geometry already converted from Bohr to Å: the fixture's 6.8431966 Bohr
    fcc Cu cell must reach the viewer as ~3.6213 Å, not 1.89 times too large."""

    from mufinder.domain.models import Result as _Result

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    job.status = JobStatus.COMPLETED
    ts_path = tmp_path / "barrier.ts"
    ts_path.write_text(_NEB_TS_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    result = _Result(
        calculation_run_id=run.id,
        result_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
        data={"num_images": 7, "transition_state_file": str(ts_path)},
    )
    run.result_id = result.id
    document.runs[run.id] = run
    document.results[result.id] = result

    view = _FakeEnergyBarriersView()
    controller = EnergyBarriersController(
        make_context(document=document, project_service=service),
        _stub_energy_barriers_deps(),
        lambda: view,
    )

    controller.show_neb_preview(str(result.id))

    assert view.cleared_previews == []
    preview = view.neb_previews[-1]
    assert preview["title"] == "barrier_neb (7 images)"
    structures = preview["structures"]
    assert [structure.label for structure in structures] == [
        f"barrier_neb {label}" for label in ("REA", "TST", "TST", "TST", "TST", "TST", "PRO")
    ]
    reactant = structures[0]
    assert reactant.lattice_vectors[0] == pytest.approx([3.6213, 0.0, 0.0], abs=1e-4)
    assert reactant.lattice_vectors[1] == pytest.approx([0.0, 3.6213, 0.0], abs=1e-4)
    assert reactant.lattice_vectors[2] == pytest.approx([0.0, 0.0, 3.6213], abs=1e-4)
    assert reactant.species == ["Cu", "Cu", "Cu", "Cu", "H:mu"]
    # Fractional coordinates are unit-free, so they only come out right when the
    # lattice and the Cartesian positions were converted together.
    assert reactant.fractional_positions[1] == pytest.approx([0.5, 0.5, 0.0])
    assert reactant.fractional_positions[4] == pytest.approx([0.5, 0.5, 0.5])
    assert structures[-1].fractional_positions[4] == pytest.approx([0.25, 0.25, 0.25])


def test_energy_barriers_controller_build_result_missing_dir_is_empty(tmp_path: Path) -> None:
    """If the job's output directory can't be resolved, build_result yields an
    empty result rather than crashing (validate_artifacts fails such jobs
    first)."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    run = CalculationRun(job_id=job.id)
    deps = _stub_energy_barriers_deps()
    controller = EnergyBarriersController(
        make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: None,
        ),
        deps,
        lambda: None,
    )
    assert (
        controller.build_result(job, completed_castep_result(), run, ParsedJobArtifacts()).payload
        == {}
    )


def test_energy_barriers_controller_pre_submit_check_blocks_on_structure_mismatch(
    tmp_path: Path,
) -> None:
    """An unresolved structure host mismatch refuses the submission: the guard
    returns False, so no submission plan is built and nothing is dispatched."""

    service = ProjectService()
    document, _definition, _source, output_root = _seed_energy_barriers_document(service, tmp_path)
    other_host = _Structure(
        label="other host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    document.structures[other_host.id] = other_host
    document.project.host_structure_id = other_host.id
    mismatches: list[object] = []
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    controller = _neb_controller(
        service,
        document,
        output_root,
        deps=_stub_energy_barriers_deps(
            handle_structure_host_mismatch=lambda mismatch: mismatches.append(mismatch) or False
        ),
        local_backend=local,
        infos=infos,
    )

    assert controller.pre_submit_check(["source-key"]) is False
    controller.submit(["source-key"], target="local")
    assert len(mismatches) == 2  # once per check
    assert local.calls == []  # nothing dispatched
    assert infos == []  # silent refusal


def test_energy_barriers_controller_pre_submit_check_passes_without_document() -> None:
    controller = EnergyBarriersController(
        make_context(document=None), _stub_energy_barriers_deps(), lambda: None
    )
    assert controller.pre_submit_check(["source-key"]) is True


def _seed_energy_barriers_document(service: ProjectService, tmp_path: Path, *, row_settings=None):
    """Seed a document with a host, two relaxed endpoints, one generated path keyed
    ``source-key`` in the Energy Barriers form state, and a NEB definition.

    ``row_settings`` is stored as the definition's per-path configuration.
    """

    from mufinder.domain.models import Result as _Result

    document = service.create_project(tmp_path / "eb_mat.mufinder", "EBMat")
    host = _Structure(
        label="host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    document.structures[host.id] = host
    document.project.host_structure_id = host.id

    ids: dict[str, dict[str, str]] = {}
    for side, label, muon in (("reactant", "siteA", 0.25), ("product", "siteB", 0.75)):
        relaxed = _Structure(
            label=f"{label} relaxed",
            origin="muon_relaxation",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si", "Si", "H:mu"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [muon, 0.25, 0.25]],
            parent_structure_id=host.id,
            properties={
                "atomic_spins": [1.0, -1.0, 0.0],
                "source_workflow_stage": "muon_relaxation",
            },
        )
        document.structures[relaxed.id] = relaxed
        job = service.add_job(
            document,
            label=f"mu-{label}",
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type="castep_positivemuon_relaxation",
        )
        job.status = JobStatus.COMPLETED
        run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
        result = _Result(calculation_run_id=run.id, derived_structure_ids=[relaxed.id])
        run.result_id = result.id
        document.runs[run.id] = run
        document.results[result.id] = result
        site = _MuonSite(
            label=label, status="relaxed", relaxation_run_id=run.id, source_structure_id=host.id
        )
        document.sites[site.id] = site
        ids[side] = {
            "site": str(site.id),
            "run": str(run.id),
            "job": str(job.id),
            "structure": str(relaxed.id),
        }

    source = {
        "key": "source-key",
        "label": "siteA_to_siteB_path1",
        "path_index": 1,
        **{
            f"{side}_{field}_id": value
            for side, fields in ids.items()
            for field, value in fields.items()
        },
        "path_start_fractional": [0.25, 0.25, 0.25],
        "path_end_fractional": [0.75, 0.25, 0.25],
        "path_length_ang": 2.5,
        "charge_state": "positive_muon",
    }
    document.ui_state["energy_barriers_form"] = {"generated_sources": [dict(source)]}
    output_root = document.project_root / "files" / "energy_barriers_neb"
    configuration: dict[str, object] = {"method": "NEB", "sources": [source]}
    if row_settings is not None:
        configuration["row_settings"] = {"source-key": row_settings}
    definition = _CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_nudged_elastic_band_set",
        parameters={
            "output_directory": "files/energy_barriers_neb",
            "job_ids_by_label": {},
            "configuration": configuration,
        },
    )
    document.calculations[definition.id] = definition
    return document, definition, source, output_root


def _neb_controller(
    service: ProjectService,
    document,
    output_root,
    *,
    deps=None,
    view_getter=lambda: None,
    **context_kwargs,
):
    import dataclasses

    persisted = context_kwargs.pop("persisted", None)
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
        **context_kwargs,
    )
    if persisted is not None:
        context = dataclasses.replace(
            context, persist_ui_state=lambda **_: persisted.append("persist")
        )
    return EnergyBarriersController(context, deps or _stub_energy_barriers_deps(), view_getter)


def test_energy_barriers_controller_materialize_source_key_provenance(tmp_path: Path) -> None:
    """materialize writes ``job_ids_by_label`` keyed by SOURCE KEY and stamps each
    spec with ``provenance={"source_key": key}`` — the NEB provenance contract.

    Uses the real science stack (``prepare_transition_state_structures`` /
    ``idpp_intermediate_structure`` / ``expand_nudged_elastic_band_jobs``)."""

    service = ProjectService()
    document, definition, _source, output_root = _seed_energy_barriers_document(service, tmp_path)
    persisted: list[str] = []
    controller = _neb_controller(service, document, output_root, persisted=persisted)

    jobs = controller.materialize(["source-key"])

    assert len(jobs) == 1
    created = jobs[0]
    assert created.calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE
    # job_ids_by_label is keyed by the SOURCE KEY, mapping to the new job id.
    assert definition.parameters["job_ids_by_label"] == {"source-key": str(created.id)}
    assert persisted == ["persist"]
    param_text = (output_root / created.label / f"{created.label}.param").read_text(
        encoding="utf-8"
    )
    # An unconfigured path runs the default (odd) image count.
    assert "TSSEARCH_MAX_PATH_POINTS : 11" in param_text


def test_energy_barriers_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    """The full submit path (pre_submit_check → plan → materialize → backend)
    dispatches the newly-created NEB job to the LOCAL backend — the flow the
    lazy-view submit lambda invokes."""

    service = ProjectService()
    document, _definition, _source, output_root = _seed_energy_barriers_document(service, tmp_path)
    local = RecordingBackend()
    cluster = RecordingBackend()
    controller = _neb_controller(
        service, document, output_root, local_backend=local, cluster_backend=cluster
    )

    controller.submit(["source-key"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    assert len(local.calls[0][0]) == 1
    assert local.calls[0][0][0].calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE


def test_energy_barriers_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, _source, output_root = _seed_energy_barriers_document(service, tmp_path)
    local = RecordingBackend()
    cluster = RecordingBackend()
    controller = _neb_controller(
        service, document, output_root, local_backend=local, cluster_backend=cluster
    )

    controller.submit(["source-key"], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE


def _materialized_neb_param(tmp_path: Path, row_settings) -> str:
    service = ProjectService()
    document, _definition, _source, output_root = _seed_energy_barriers_document(
        service, tmp_path, row_settings=row_settings
    )
    jobs = _neb_controller(service, document, output_root).materialize(["source-key"])
    assert len(jobs) == 1
    return (output_root / jobs[0].label / f"{jobs[0].label}.param").read_text(encoding="utf-8")


def test_energy_barriers_controller_materialize_writes_num_farms(tmp_path: Path) -> None:
    """The per-row task-farm count reaches the generated ``.param`` as NUM_FARMS.

    The farm count is stored in the row's configuration extras by the NEB
    configure dialog (it is not a CASTEP keyword row), so it has to be read from
    extras at materialize time -- the point at which the job directory is
    written."""

    assert "NUM_FARMS : 3" in _materialized_neb_param(tmp_path, {"num_farms": 3})


def test_energy_barriers_controller_materialize_omits_num_farms_when_off(
    tmp_path: Path,
) -> None:
    assert "NUM_FARMS" not in _materialized_neb_param(tmp_path, None)


def test_energy_barriers_controller_materialize_ignores_num_farms_parameter_row(
    tmp_path: Path,
) -> None:
    """A persisted ``num_farms`` keyword row must not override the row's extras.

    The extras value is what ``_task_farming_for_job`` reports to the runners
    (deciding whether to read ``<seed>_farm001.*``) and what submission
    validates the rank count against. If a keyword row could win in the rendered
    .param, CASTEP would farm one way while MuFinder assumed another -- so the
    dedicated control stays authoritative and the keyword row is dropped."""

    param_text = _materialized_neb_param(
        tmp_path, {"parameter_values": {"num_farms": 7}, "num_farms": 3}
    )

    assert "NUM_FARMS : 3" in param_text
    assert "NUM_FARMS : 7" not in param_text


def test_energy_barriers_controller_generate_paths_validates_its_inputs(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, source, output_root = _seed_energy_barriers_document(service, tmp_path)
    infos: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    controller = _neb_controller(
        service,
        document,
        output_root,
        deps=_stub_energy_barriers_deps(
            show_warning=lambda title, text: warnings.append((title, text))
        ),
        infos=infos,
    )
    reactant, product = source["reactant_site_id"], source["product_site_id"]

    controller.generate_paths(reactant, "")
    assert infos[-1][0] == "Select endpoints"
    controller.generate_paths(reactant, product, "-1")
    assert warnings == [("Invalid path length", "Maximum path length must be greater than zero.")]
    document.jobs[UUID(source["product_job_id"])].calc_type = "castep_muonium_relaxation"
    controller.generate_paths(reactant, product)
    assert infos[-1][0] == "Charge-state mismatch"
    assert document.ui_state["energy_barriers_form"]["generated_sources"] == [source]


def test_energy_barriers_controller_generates_and_removes_paths(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(service, tmp_path)
    persisted: list[str] = []
    view = _FakeEnergyBarriersView()
    controller = _neb_controller(
        service, document, output_root, view_getter=lambda: view, persisted=persisted
    )
    reactant, product = source["reactant_site_id"], source["product_site_id"]

    controller.select_endpoints(product, reactant)
    state = document.ui_state["energy_barriers_form"]
    assert (state["selected_reactant_key"], state["selected_product_key"]) == (product, reactant)

    controller.generate_paths(reactant, product, "4.0")

    state = document.ui_state["energy_barriers_form"]
    assert (state["selected_reactant_key"], state["selected_product_key"]) == (reactant, product)
    assert state["max_path_length_ang"] == 4.0
    keys = [generated["key"] for generated in state["generated_sources"]]
    assert keys and all(key.startswith(f"{reactant}:{product}:") for key in keys)
    assert persisted == ["persist"]
    assert view.endpoint_calls[-1][1]["max_path_length_text"] == "4"

    # Removing the original path drops it from the form state and from the
    # definition, which is deleted once it holds no path.
    assert controller.remove_rows(["source-key"]) == (definition.id,)
    assert "source-key" not in [generated["key"] for generated in state["generated_sources"]]
    assert definition.id not in document.calculations


def test_energy_barriers_controller_configure_request_and_apply(tmp_path: Path) -> None:
    from mufinder.app.workflow_config_codecs import workflow_row_configuration

    service = ProjectService()
    document, definition, _source, output_root = _seed_energy_barriers_document(
        service, tmp_path, row_settings={"num_farms": 2}
    )
    controller = _neb_controller(service, document, output_root)

    request = controller.configure_request(["source-key", "missing"])

    assert request is not None
    assert request.keys == ("source-key",)
    assert request.title == "Configure NEB Parameters"
    assert request.intro_text == "Path: siteA_to_siteB_path1"
    assert request.read_only is False
    assert request.num_farms == 2
    assert request.species_options == ("Si", "H:mu")
    assert request.row_values[0]["tssearch_max_path_points"] == 11

    controller.apply_configuration(
        request.keys, configure_result({"tssearch_max_path_points": 8}), {"num_farms": 4}
    )

    params, _advanced, extras = workflow_row_configuration(definition, "source-key")
    assert extras == {"num_farms": 4}
    assert params == {"tssearch_max_path_points": 8}
    # An even image count is rounded up for climbing-image NEB.
    assert (
        controller.row_configuration(definition, "source-key")[0]["tssearch_max_path_points"] == 9
    )

    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    job.status = JobStatus.RUNNING
    definition.parameters["job_ids_by_label"]["source-key"] = str(job.id)
    inspected = controller.configure_request(["source-key"])
    assert inspected is not None
    assert (inspected.title, inspected.read_only) == ("Inspect NEB Parameters", True)
    assert controller.configure_request(["missing"]) is None


@pytest.mark.parametrize(("entered", "effective"), [(40, 41), (41, 41), (7, 7)])
def test_energy_barriers_row_configuration_stores_the_image_count_castep_runs(
    tmp_path: Path, entered: int, effective: int
) -> None:
    """``TSSEARCH_MAX_PATH_POINTS`` counts intermediate images; climbing is locked on.

    With climbing on, CASTEP adds one image to an even count, so the effective
    count is the next odd number and an odd count is kept as entered.
    """

    service = ProjectService()
    document, definition, _source, output_root = _seed_energy_barriers_document(service, tmp_path)
    controller = _neb_controller(service, document, output_root)
    request = controller.configure_request(["source-key"])
    assert request is not None

    controller.apply_configuration(
        request.keys, configure_result({"tssearch_max_path_points": entered}), {}
    )

    values = controller.row_configuration(definition, "source-key")[0]
    assert values["tssearch_max_path_points"] == effective


def test_energy_barriers_controller_path_preview(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, source, output_root = _seed_energy_barriers_document(service, tmp_path)
    controller = _neb_controller(service, document, output_root)

    preview = controller.path_preview("source-key")

    assert preview is not None
    assert preview.title == "siteA_to_siteB_path1 path preview"
    assert preview.structure.origin == "energy_barriers_path_preview"
    assert preview.start_fractional == tuple(source["path_start_fractional"])
    assert preview.end_fractional == tuple(source["path_end_fractional"])
    assert controller.path_preview("missing") is None
    document.project.host_structure_id = None
    with pytest.raises(ValueError, match="pristine host structure is required"):
        controller.path_preview("source-key")


@pytest.mark.parametrize(
    "calc_type", [NUDGED_ELASTIC_BAND_CALC_TYPE, TRANSITION_STATE_SEARCH_CALC_TYPE]
)
def test_energy_barriers_controller_validate_artifacts_checks_ts(
    tmp_path: Path, calc_type: str
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "e.mufinder", "E")
    controller = EnergyBarriersController(
        make_context(document=document), _stub_energy_barriers_deps(), lambda: None
    )
    job = _neb_job(service, document, "neb_path", calc_type)
    run = CalculationRun(job_id=job.id)
    parsed = CastepResult(completed=True, total_energy=-1.0)

    assert (
        controller.validate_artifacts(job, parsed, run, ParsedJobArtifacts(ts_missing=True))
        == "Expected CASTEP path output was not found: neb_path.ts"
    )
    # Under task farming the collector looked for the farmed name; report that.
    assert (
        controller.validate_artifacts(
            job,
            parsed,
            run,
            ParsedJobArtifacts(ts_missing=True, ts_path=tmp_path / "neb_path_farm001.ts"),
        )
        == "Expected CASTEP path output was not found: neb_path_farm001.ts"
    )
    assert (
        controller.validate_artifacts(
            job, parsed, run, ParsedJobArtifacts(ts_error="truncated band")
        )
        == "Failed to parse CASTEP path output: truncated band"
    )
    assert controller.validate_artifacts(job, parsed, run, no_artifacts()) is None
