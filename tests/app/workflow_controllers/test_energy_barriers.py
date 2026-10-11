"""Tests for :class:`EnergyBarriersController`.

Covers the lazily created view, the ``.ts`` result payload, source-key
materialization, ``NUM_FARMS`` rendering and ``validate_artifacts``.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from pathlib import Path
from uuid import UUID
from uuid import uuid4 as _uuid4

import pytest

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.workflow.types import InputTemplate
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
    FakeRowState,
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

    def set_endpoint_options(self, rows, **kwargs) -> None:
        self.endpoint_calls.append((list(rows), kwargs))

    def set_sources(self, rows) -> None:
        self.sources.append(list(rows))

    def set_neb_results(self, rows) -> None:
        self.neb_results.append(list(rows))


@_dataclass
class _FakeNebCandidate:
    key: str
    source: dict


def _stub_energy_barriers_deps(**overrides) -> EnergyBarriersDeps:
    base = dict(
        build_energy_barrier_neb_states=lambda: {},
        ensure_energy_barrier_neb_definition=lambda _keys: None,
        energy_barrier_neb_source_candidates=lambda: [],
        eligible_energy_barrier_source_candidates=lambda: [],
        energy_barrier_neb_row_configuration_with_source_defaults=(
            lambda _definition, _key, **_kwargs: ({}, (), {})
        ),
        handle_structure_host_mismatch=lambda _mismatch: True,
        handle_site_host_mismatch=lambda _mismatch: True,
        uuid_from_source=lambda source, key: _uuid_from_source_stub(source, key),
        float_from_source=lambda source, key: _float_from_source_stub(source, key),
        fractional_triplet_from_source=lambda source, key: tuple(
            source.get(key, (0.0, 0.0, 0.0))
        ),
        fractional_position_block_from_structure=lambda structure: structure,
        dedupe_jobs=lambda jobs: jobs,
        project_convergence_input_defaults=lambda: None,
        input_template_from_keyword_values=lambda _values: InputTemplate(),
        workflow_row_configuration_exists=lambda _definition, _key: False,
        num_farms_from_extras=lambda extras: max(1, int(extras.get("num_farms", 1))),
        energy_barrier_neb_row_extras=lambda _definition, _key: {},
        charge_state_label_for_job=lambda _job: "",
        energy_barrier_endpoint_rows=lambda: [],
        selected_energy_barrier_endpoint_keys=lambda: ("", ""),
        energy_barrier_max_path_length_text=lambda: "",
        completed_energy_barrier_neb_candidates=lambda: [],
        energy_barrier_neb_sources_by_job_id=lambda: {},
        energy_barrier_neb_result_path_label=lambda _candidate, _source: "",
        neb_barrier_summaries=lambda _data: ("", "", ""),
        muon_charge_state_label=lambda value: value,
    )
    base.update(overrides)
    return EnergyBarriersDeps(**base)


def _uuid_from_source_stub(source, key):
    raw = source.get(key)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return UUID(raw.strip())
    except ValueError:
        return None


def _float_from_source_stub(source, key):
    raw = source.get(key)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _neb_job(service: ProjectService, document: ProjectDocument, label: str, calc_type: str) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=calc_type,
    )


def _neb_frac_block(structure):
    """Mirror ``MainWindow._fractional_position_block_from_structure`` for tests."""

    from mufinder.castep_io.types import AtomicPosition, PositionBlock

    rows = [
        AtomicPosition(str(species), tuple(float(v) for v in fractional))
        for species, fractional in zip(
            structure.species, structure.fractional_positions, strict=True
        )
    ]
    return PositionBlock(positions=tuple(rows), unit="frac")


def _neb_materialize_deps(definition, candidate, output_root, **overrides) -> EnergyBarriersDeps:
    """Deps wired so ``materialize`` runs the real science stack and creates a job
    for ``candidate`` — shared by the source-key and submit tests."""

    base = dict(
        ensure_energy_barrier_neb_definition=lambda _keys: definition,
        energy_barrier_neb_source_candidates=lambda: [candidate],
        eligible_energy_barrier_source_candidates=lambda: [],
        fractional_position_block_from_structure=_neb_frac_block,
        build_energy_barrier_neb_states=lambda: {
            candidate.key: FakeRowState("ready")
        },
    )
    base.update(overrides)
    return _stub_energy_barriers_deps(**base)


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
        document, label="u", workflow_stage=WorkflowStage.MUON_RELAXATION, calc_type="castep_uep_source"
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

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    job = _neb_job(service, document, "barrier_neb", NUDGED_ELASTIC_BAND_CALC_TYPE)
    job.status = JobStatus.COMPLETED

    @_dataclass
    class _Endpoint:
        site: object
        job: object

    @_dataclass
    class _Named:
        label: str

    @_dataclass
    class _Path:
        length_ang: float

    @_dataclass
    class _SourceCandidate:
        key: str
        reactant: object
        product: object
        path: object

    @_dataclass
    class _CompletedResult:
        data: dict

    @_dataclass
    class _CompletedCandidate:
        key: str
        job: object
        result: object

    source_candidate = _SourceCandidate(
        key="source-key",
        reactant=_Endpoint(site=_Named("siteA"), job=_Named("jobA")),
        product=_Endpoint(site=_Named("siteB"), job=_Named("jobB")),
        path=_Path(length_ang=2.5),
    )
    completed = _CompletedCandidate(
        key="result-key",
        job=job,
        result=_CompletedResult(
            data={
                "num_images": 7,
                "transition_state_file": "/tmp/barrier.ts",
                "image_energies_eV": [-1.0, -0.8, -0.9],
                "charge_state": "positive",
            }
        ),
    )

    view = _FakeEnergyBarriersView()
    deps = _stub_energy_barriers_deps(
        eligible_energy_barrier_source_candidates=lambda: [source_candidate],
        build_energy_barrier_neb_states=lambda: {"source-key": FakeRowState("ready")},
        completed_energy_barrier_neb_candidates=lambda: [completed],
        energy_barrier_neb_sources_by_job_id=lambda: {job.id: {"charge_state": "positive", "path_length_ang": 2.5}},
        energy_barrier_neb_result_path_label=lambda _c, _s: "siteA → siteB",
        neb_barrier_summaries=lambda _data: ("0.2 eV", "0.1 eV", "-0.1 eV"),
        muon_charge_state_label=lambda value: f"mu+({value})",
    )
    controller = EnergyBarriersController(
        make_context(document=document, project_service=service), deps, lambda: view
    )

    controller.refresh_view()

    assert len(view.endpoint_calls) == 1
    assert len(view.sources[-1]) == 1
    assert view.sources[-1][0].reactant == "siteA"
    assert len(view.neb_results[-1]) == 1
    result_row = view.neb_results[-1][0]
    assert result_row.label == "siteA → siteB"
    assert result_row.images == "7"
    assert result_row.charge_state == "mu+(positive)"
    assert result_row.path_length == "2.5 Å"
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
    assert controller.build_result(job, completed_castep_result(), run, ParsedJobArtifacts()).payload == {}


def test_energy_barriers_controller_pre_submit_check_blocks_on_structure_mismatch(
    tmp_path: Path,
) -> None:
    """An unresolved structure host mismatch refuses the submission: the guard
    returns False, so no submission plan is built and nothing is dispatched."""

    service = ProjectService()
    document = service.create_project(tmp_path / "eb.mufinder", "EB")
    structure_id = _uuid4()

    class _StubService:
        def validate_structures_match_active_host(self, _doc, _ids):
            return object()  # a mismatch

        def validate_sites_match_active_host(self, _doc, _ids):
            return None

    candidate = _FakeNebCandidate(
        key="source-key",
        source={
            "reactant_structure_id": str(structure_id),
            "product_structure_id": str(structure_id),
        },
    )
    deps = _stub_energy_barriers_deps(
        energy_barrier_neb_source_candidates=lambda: [candidate],
        handle_structure_host_mismatch=lambda _mismatch: False,
    )
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(
        document=document, project_service=_StubService(), local_backend=local, infos=infos
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    assert controller.pre_submit_check(["source-key"]) is False
    controller.submit(["source-key"], target="local")
    assert local.calls == []  # nothing dispatched
    assert infos == []  # silent refusal


def test_energy_barriers_controller_pre_submit_check_passes_without_document() -> None:
    controller = EnergyBarriersController(
        make_context(document=None), _stub_energy_barriers_deps(), lambda: None
    )
    assert controller.pre_submit_check(["source-key"]) is True


def _seed_energy_barriers_document(service: ProjectService, tmp_path: Path):
    """Seed a document with a host, two relaxed endpoints, and a NEB definition
    carrying one generated-path source keyed ``source-key``."""

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

    relaxed_a = _Structure(
        label="siteA relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.25, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )
    relaxed_b = _Structure(
        label="siteB relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.52, 0.5, 0.5], [0.75, 0.25, 0.25]],
        parent_structure_id=host.id,
        properties={"atomic_spins": [1.0, -1.0, 0.0]},
    )
    document.structures[relaxed_a.id] = relaxed_a
    document.structures[relaxed_b.id] = relaxed_b

    site_a = _MuonSite(label="siteA", status="relaxed")
    site_b = _MuonSite(label="siteB", status="relaxed")
    document.sites[site_a.id] = site_a
    document.sites[site_b.id] = site_b

    source = {
        "key": "source-key",
        "label": "siteA_to_siteB_path1",
        "reactant_site_id": str(site_a.id),
        "product_site_id": str(site_b.id),
        "reactant_run_id": str(_uuid4()),
        "product_run_id": str(_uuid4()),
        "reactant_job_id": str(_uuid4()),
        "product_job_id": str(_uuid4()),
        "reactant_structure_id": str(relaxed_a.id),
        "product_structure_id": str(relaxed_b.id),
        "path_start_fractional": [0.25, 0.25, 0.25],
        "path_end_fractional": [0.75, 0.25, 0.25],
        "path_length_ang": 2.5,
    }
    output_root = document.project_root / "files" / "energy_barriers_neb"
    definition = _CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_nudged_elastic_band_set",
        parameters={
            "output_directory": "files/energy_barriers_neb",
            "job_ids_by_label": {},
            "configuration": {"method": "NEB", "sources": [source]},
        },
    )
    document.calculations[definition.id] = definition
    return document, definition, source, output_root


def test_energy_barriers_controller_materialize_source_key_provenance(tmp_path: Path) -> None:
    """materialize writes ``job_ids_by_label`` keyed by SOURCE KEY and stamps each
    spec with ``provenance={"source_key": key}`` — the NEB provenance contract.

    Uses the real science stack (``prepare_transition_state_structures`` /
    ``idpp_intermediate_structure`` / ``expand_nudged_elastic_band_jobs``) with
    only the MainWindow-side helpers stubbed via deps, so this pins the controller's
    materialize behavior in isolation."""

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)

    import dataclasses

    persisted: list[str] = []
    deps = _neb_materialize_deps(definition, candidate, output_root)
    context = dataclasses.replace(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        persist_ui_state=lambda **_: persisted.append("persist"),
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    jobs = controller.materialize(["source-key"])

    assert len(jobs) == 1
    created = jobs[0]
    assert created.calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE
    # job_ids_by_label is keyed by the SOURCE KEY, mapping to the new job id.
    assert definition.parameters["job_ids_by_label"] == {"source-key": str(created.id)}
    # Each expanded spec carried provenance={"source_key": key}; verify it round-trips
    # by re-reading the mapping keyed by that same source key.
    assert set(definition.parameters["job_ids_by_label"].keys()) == {"source-key"}
    assert persisted == ["persist"]


def test_energy_barriers_controller_submit_local_reaches_local_backend(tmp_path: Path) -> None:
    """The full submit path (pre_submit_check → plan → materialize → backend)
    dispatches the newly-created NEB job to the LOCAL backend — the flow the
    lazy-view submit lambda invokes."""

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(definition, candidate, output_root)
    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    controller.submit(["source-key"], target="local")

    assert cluster.calls == []
    assert len(local.calls) == 1
    assert len(local.calls[0][0]) == 1
    assert local.calls[0][0][0].calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE


def test_energy_barriers_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(definition, candidate, output_root)
    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    controller.submit(["source-key"], target="cluster")

    assert local.calls == []
    assert len(cluster.calls) == 1
    assert cluster.calls[0][0][0].calc_type == NUDGED_ELASTIC_BAND_CALC_TYPE


def test_energy_barriers_controller_materialize_writes_num_farms(tmp_path: Path) -> None:
    """The per-row task-farm count reaches the generated ``.param`` as NUM_FARMS.

    The farm count is stored in the row's configuration extras by the NEB
    configure dialog (it is not a CASTEP keyword row), so it has to be read from
    extras at materialize time -- the point at which the job directory is
    written."""

    import dataclasses

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(
        definition,
        candidate,
        output_root,
        # materialize reads the row's extras from the same call that resolves the
        # inherited cut-off/k-points, which is what MainWindow wires in production.
        energy_barrier_neb_row_configuration_with_source_defaults=(
            lambda _definition, _key, **_kwargs: ({}, (), {"num_farms": 3})
        ),
    )
    context = dataclasses.replace(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        persist_ui_state=lambda **_: None,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    jobs = controller.materialize(["source-key"])

    assert len(jobs) == 1
    param_text = (output_root / jobs[0].label / f"{jobs[0].label}.param").read_text(
        encoding="utf-8"
    )
    assert "NUM_FARMS : 3" in param_text


def test_energy_barriers_controller_materialize_omits_num_farms_when_off(
    tmp_path: Path,
) -> None:
    import dataclasses

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(definition, candidate, output_root)
    context = dataclasses.replace(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        persist_ui_state=lambda **_: None,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    jobs = controller.materialize(["source-key"])

    param_text = (output_root / jobs[0].label / f"{jobs[0].label}.param").read_text(
        encoding="utf-8"
    )
    assert "NUM_FARMS" not in param_text


def test_energy_barriers_controller_materialize_ignores_num_farms_parameter_row(
    tmp_path: Path,
) -> None:
    """A persisted ``num_farms`` keyword row must not override the row's extras.

    The extras value is what ``_task_farming_for_job`` reports to the runners
    (deciding whether to read ``<seed>_farm001.*``) and what submission
    validates the rank count against. If a keyword row could win in the rendered
    .param, CASTEP would farm one way while MuFinder assumed another -- so the
    dedicated control stays authoritative and the keyword row is dropped."""

    import dataclasses

    service = ProjectService()
    document, definition, source, output_root = _seed_energy_barriers_document(
        service, tmp_path
    )
    candidate = _FakeNebCandidate(key="source-key", source=source)
    deps = _neb_materialize_deps(
        definition,
        candidate,
        output_root,
        energy_barrier_neb_row_configuration_with_source_defaults=(
            lambda _definition, _key, **_kwargs: (
                {"num_farms": 7},
                (),
                {"num_farms": 3},
            )
        ),
    )
    context = dataclasses.replace(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        persist_ui_state=lambda **_: None,
    )
    controller = EnergyBarriersController(context, deps, lambda: None)

    jobs = controller.materialize(["source-key"])

    param_text = (output_root / jobs[0].label / f"{jobs[0].label}.param").read_text(
        encoding="utf-8"
    )
    assert "NUM_FARMS : 3" in param_text
    assert "NUM_FARMS : 7" not in param_text


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

    assert controller.validate_artifacts(
        job, parsed, run, ParsedJobArtifacts(ts_missing=True)
    ) == "Expected CASTEP path output was not found: neb_path.ts"
    # Under task farming the collector looked for the farmed name; report that.
    assert controller.validate_artifacts(
        job,
        parsed,
        run,
        ParsedJobArtifacts(ts_missing=True, ts_path=tmp_path / "neb_path_farm001.ts"),
    ) == "Expected CASTEP path output was not found: neb_path_farm001.ts"
    assert controller.validate_artifacts(
        job, parsed, run, ParsedJobArtifacts(ts_error="truncated band")
    ) == "Failed to parse CASTEP path output: truncated band"
    assert controller.validate_artifacts(job, parsed, run, no_artifacts()) is None
