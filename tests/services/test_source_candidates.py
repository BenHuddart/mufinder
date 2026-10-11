"""Source-candidate queries over a project document."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mufinder.domain.charge_states import (
    MUONIUM_CHARGE_STATE,
    POSITIVE_MUON_CHARGE_STATE,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
)
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
    MuonSite,
    Result,
    Structure,
)
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.source_candidates import (
    QUANTUM_STAGE_PHONON,
    QUANTUM_STAGE_SINGLEPOINT,
    CompletedRelaxationAnalysisSource,
    charge_state_for_job,
    coerce_fractional_triplet,
    completed_energy_barrier_neb_candidates,
    completed_muon_relaxation_sources,
    completed_pristine_relaxation_jobs,
    eligible_energy_barrier_source_candidates,
    eligible_quantum_source_candidates,
    energy_barrier_generated_sources,
    energy_barrier_neb_source_candidates,
    enumerate_energy_barrier_source_candidates,
    float_from_source,
    fractional_triplet_from_source,
    quantum_source_candidate_by_key,
    quantum_source_labels_by_key,
    quantum_source_relative_energies,
    quantum_stage_state,
    relaxation_sources_for_labels,
    uuid_from_source,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
)
from tests.support.structures import host_structure


@pytest.fixture
def document(tmp_path: Path) -> ProjectDocument:
    doc = ProjectService().create_project(tmp_path / "sources.mufinder", "Sources")
    host = host_structure(label="host")
    doc.structures[host.id] = host
    doc.project.host_structure_id = host.id
    return doc


def _host(doc: ProjectDocument) -> Structure:
    return doc.structures[doc.project.host_structure_id]


def _add_relaxed_source(
    doc: ProjectDocument,
    label: str,
    *,
    energy: float | None,
    muon: tuple[float, float, float] = (0.25, 0.25, 0.25),
    charge_state: str = POSITIVE_MUON_CHARGE_STATE,
    extra_species: tuple[str, ...] = (),
    job_label: str | None = None,
) -> MuonSite:
    host = _host(doc)
    species = ["Si", "Si", *extra_species, "H:mu"]
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]] + [[0.1, 0.1, 0.1]] * len(extra_species) + [list(muon)]
    relaxed = Structure(
        label=f"{label} relaxed",
        origin="muon_relaxation",
        lattice_vectors=[list(vector) for vector in host.lattice_vectors],
        species=species,
        fractional_positions=positions,
        parent_structure_id=host.id,
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed.id] = relaxed
    job = Job(
        label=job_label if job_label is not None else f"mu-{label}",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=muon_relaxation_calc_type(charge_state),
        status=JobStatus.COMPLETED,
    )
    doc.jobs[job.id] = job
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
    result = Result(calculation_run_id=run.id, derived_structure_ids=[relaxed.id])
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label=label, status="relaxed", relaxation_run_id=run.id, total_energy_eV=energy)
    doc.sites[site.id] = site
    return site


# --- serialized source fields -------------------------------------------------


def test_uuid_from_source_parses_valid_ids_only() -> None:
    value = uuid4()
    assert uuid_from_source({"id": str(value)}, "id") == value
    assert uuid_from_source({"id": "not-a-uuid"}, "id") is None
    assert uuid_from_source({"id": "  "}, "id") is None
    assert uuid_from_source({"id": 12}, "id") is None
    assert uuid_from_source({}, "id") is None


def test_fractional_triplet_from_source_requires_three_numbers() -> None:
    assert fractional_triplet_from_source({"p": [0, "0.5", 1]}, "p") == (0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="three fractional coordinates for p"):
        fractional_triplet_from_source({"p": [0, 1]}, "p")
    with pytest.raises(ValueError):
        fractional_triplet_from_source({"p": (0, 0, 0)}, "p")


def test_float_from_source_and_coerce_fractional_triplet() -> None:
    assert float_from_source({"x": "2.5"}, "x") == 2.5
    assert float_from_source({"x": "abc"}, "x") is None
    assert float_from_source({}, "x") is None
    assert coerce_fractional_triplet((1, 2, 3)) == (1.0, 2.0, 3.0)
    assert coerce_fractional_triplet([1, "x", 3]) is None
    assert coerce_fractional_triplet([1, 2]) is None


def test_charge_state_for_job_reads_muon_relaxation_calc_types() -> None:
    assert charge_state_for_job(None) is None
    assert charge_state_for_job(
        Job(calc_type=muon_relaxation_calc_type(MUONIUM_CHARGE_STATE))
    ) == MUONIUM_CHARGE_STATE
    assert charge_state_for_job(
        Job(calc_type=muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE))
    ) == POSITIVE_MUON_CHARGE_STATE
    assert charge_state_for_job(Job(calc_type="castep_cutoff")) is None


# --- quantum-corrections sources ----------------------------------------------


def test_queries_return_nothing_without_a_document() -> None:
    assert eligible_quantum_source_candidates(None) == []
    assert quantum_source_candidate_by_key(None, "x") is None
    assert energy_barrier_generated_sources(None) == []
    assert eligible_energy_barrier_source_candidates(None) == []
    assert enumerate_energy_barrier_source_candidates(None, "a", "b") == []
    assert energy_barrier_neb_source_candidates(None) == []
    assert completed_energy_barrier_neb_candidates(None) == []
    assert completed_muon_relaxation_sources(
        None, [], run_for_job=lambda _id: None, total_energy_for_job=lambda _id: None
    ) == []
    assert completed_pristine_relaxation_jobs(None, [], {}, run_for_job=lambda _id: None) == []


def test_eligible_quantum_sources_skip_ineligible_sites(document: ProjectDocument) -> None:
    eligible = _add_relaxed_source(document, "A", energy=-10.0)
    unrelaxed = MuonSite(label="generated", status="generated")
    document.sites[unrelaxed.id] = unrelaxed
    orphan = MuonSite(label="orphan", status="relaxed", relaxation_run_id=uuid4())
    document.sites[orphan.id] = orphan

    candidates = eligible_quantum_source_candidates(document)

    assert [candidate.key for candidate in candidates] == [str(eligible.id)]
    candidate = candidates[0]
    assert candidate.site is eligible
    assert candidate.run.id == eligible.relaxation_run_id
    assert "H:mu" in candidate.structure.species
    assert quantum_source_candidate_by_key(document, str(eligible.id)) == candidate
    assert quantum_source_candidate_by_key(document, "missing") is None


def test_eligible_quantum_sources_order_by_charge_state_then_relative_energy(
    document: ProjectDocument,
) -> None:
    high = _add_relaxed_source(document, "high", energy=-9.0)
    low = _add_relaxed_source(document, "low", energy=-10.0)
    unknown = _add_relaxed_source(document, "unknown", energy=None)
    muonium = _add_relaxed_source(document, "mu", energy=-50.0, charge_state=MUONIUM_CHARGE_STATE)

    ordered = [candidate.site for candidate in eligible_quantum_source_candidates(document)]

    # "muonium" sorts before "positive_muon"; within a charge state the lowest
    # relative energy comes first and sources without an energy go last.
    assert ordered == [muonium, low, high, unknown]


def test_quantum_source_relative_energies_group_by_charge_state_and_cell(
    document: ProjectDocument,
) -> None:
    a = _add_relaxed_source(document, "A", energy=-10.0)
    b = _add_relaxed_source(document, "B", energy=-9.5)
    supercell = _add_relaxed_source(document, "S", energy=-40.0, extra_species=("Si", "Si"))
    muonium = _add_relaxed_source(document, "M", energy=-8.0, charge_state=MUONIUM_CHARGE_STATE)

    relative = quantum_source_relative_energies(eligible_quantum_source_candidates(document))

    assert relative[str(a.id)] == pytest.approx(0.0)
    assert relative[str(b.id)] == pytest.approx(0.5)
    assert relative[str(supercell.id)] == pytest.approx(0.0)
    assert relative[str(muonium.id)] == pytest.approx(0.0)


def test_quantum_source_labels_deduplicate_in_candidate_order(document: ProjectDocument) -> None:
    first = _add_relaxed_source(document, "site", energy=-10.0)
    second = _add_relaxed_source(document, "site", energy=-9.0)
    unlabelled = _add_relaxed_source(document, "", energy=-8.0, job_label="")

    labels = quantum_source_labels_by_key(eligible_quantum_source_candidates(document))

    assert labels == {str(first.id): "site", str(second.id): "site_2", str(unlabelled.id): "site_3"}


def test_quantum_stage_state_resolves_each_stage() -> None:
    def row_status(job: Job) -> str:
        return f"row:{job.status.value}"

    ready =quantum_stage_state(singlepoint_job=None, phonon_job=None, row_status=row_status)
    assert (ready.stage, ready.status, ready.current_job) == (QUANTUM_STAGE_SINGLEPOINT, "ready", None)

    running = Job(status=JobStatus.RUNNING)
    stage_one = quantum_stage_state(singlepoint_job=running, phonon_job=None, row_status=row_status)
    assert (stage_one.stage, stage_one.status, stage_one.current_job) == (
        QUANTUM_STAGE_SINGLEPOINT,
        "row:running",
        running,
    )

    completed = Job(status=JobStatus.COMPLETED)
    stage_two_ready = quantum_stage_state(singlepoint_job=completed, phonon_job=None, row_status=row_status)
    assert (stage_two_ready.stage, stage_two_ready.status, stage_two_ready.current_job) == (
        QUANTUM_STAGE_PHONON,
        "ready",
        None,
    )

    phonon = Job(status=JobStatus.FAILED)
    stage_two = quantum_stage_state(singlepoint_job=completed, phonon_job=phonon, row_status=row_status)
    assert (stage_two.stage, stage_two.status, stage_two.current_job) == (
        QUANTUM_STAGE_PHONON,
        "row:failed",
        phonon,
    )


# --- energy-barrier sources ---------------------------------------------------


def _serialized_path(reactant: MuonSite, product: MuonSite, doc: ProjectDocument, **overrides) -> dict:
    def ids(site: MuonSite) -> tuple[str, str, str]:
        run = doc.runs[site.relaxation_run_id]
        structure_id = doc.results[run.result_id].derived_structure_ids[0]
        return str(run.id), str(run.job_id), str(structure_id)

    reactant_run, reactant_job, reactant_structure = ids(reactant)
    product_run, product_job, product_structure = ids(product)
    source = {
        "key": f"{reactant.id}:{product.id}:1",
        "path_index": 1,
        "reactant_site_id": str(reactant.id),
        "product_site_id": str(product.id),
        "reactant_run_id": reactant_run,
        "product_run_id": product_run,
        "reactant_job_id": reactant_job,
        "product_job_id": product_job,
        "reactant_structure_id": reactant_structure,
        "product_structure_id": product_structure,
        "path_start_fractional": [0.25, 0.25, 0.25],
        "path_end_fractional": [0.75, 0.25, 0.25],
        "path_length_ang": 2.5,
    }
    source.update(overrides)
    return source


def test_generated_sources_merge_form_state_and_definitions(document: ProjectDocument) -> None:
    document.ui_state["energy_barriers_form"] = {
        "generated_sources": [{"key": "a", "label": "form"}, {"key": "b"}, "junk"],
    }
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
        parameters={"configuration": {"sources": [{"key": "a", "label": "definition"}, {"key": "c"}]}},
    )
    other = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_nudged_elastic_band_set",
        parameters={"configuration": {"sources": [{"key": "ignored"}]}},
    )
    document.calculations[definition.id] = definition
    document.calculations[other.id] = other

    sources = energy_barrier_generated_sources(document)

    assert [source["key"] for source in sources] == ["a", "b", "c"]
    assert sources[0]["label"] == "definition"
    assert [candidate.key for candidate in energy_barrier_neb_source_candidates(document)] == ["a", "b", "c"]


def test_eligible_energy_barrier_sources_rebuild_endpoints(document: ProjectDocument) -> None:
    reactant = _add_relaxed_source(document, "A", energy=-10.0)
    product = _add_relaxed_source(document, "B", energy=-9.0, muon=(0.75, 0.25, 0.25))
    valid = _serialized_path(reactant, product, document)
    missing_site = _serialized_path(reactant, product, document, key="gone", product_site_id=str(uuid4()))
    bad_id = _serialized_path(reactant, product, document, key="bad", reactant_run_id="nope")
    bad_path = _serialized_path(reactant, product, document, key="path", path_end_fractional=[1, 2])
    document.ui_state["energy_barriers_form"] = {
        "generated_sources": [valid, missing_site, bad_id, bad_path],
    }

    candidates = eligible_energy_barrier_source_candidates(document)

    assert [candidate.key for candidate in candidates] == [valid["key"]]
    candidate = candidates[0]
    assert candidate.reactant.site is reactant
    assert candidate.product.site is product
    assert candidate.reactant.key == str(reactant.id)
    assert candidate.path.end_fractional == (0.75, 0.25, 0.25)
    assert candidate.path.length_ang == 2.5
    assert candidate.path_index == 1


def test_enumerate_energy_barrier_paths_between_two_sources(document: ProjectDocument) -> None:
    reactant = _add_relaxed_source(document, "A", energy=-10.0)
    product = _add_relaxed_source(document, "B", energy=-9.0, muon=(0.75, 0.25, 0.25))
    muonium = _add_relaxed_source(document, "M", energy=-9.0, charge_state=MUONIUM_CHARGE_STATE)

    candidates = enumerate_energy_barrier_source_candidates(document, str(reactant.id), str(product.id))

    assert candidates
    assert [candidate.path_index for candidate in candidates] == list(range(1, len(candidates) + 1))
    assert candidates[0].key == f"{reactant.id}:{product.id}:1"
    assert candidates[0].reactant.site is reactant
    assert enumerate_energy_barrier_source_candidates(document, str(reactant.id), str(muonium.id)) == []
    assert enumerate_energy_barrier_source_candidates(document, str(reactant.id), "missing") == []


def test_completed_neb_candidates_need_run_and_job(document: ProjectDocument) -> None:
    job_b = Job(label="b-neb", calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE)
    job_a = Job(label="a-neb", calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE)
    candidates = []
    for job in (job_b, job_a):
        run = CalculationRun(job_id=job.id)
        result = Result(calculation_run_id=run.id, result_type=NUDGED_ELASTIC_BAND_CALC_TYPE)
        document.jobs[job.id] = job
        document.runs[run.id] = run
        document.results[result.id] = result
        candidates.append(result)
    orphan = Result(calculation_run_id=uuid4(), result_type=NUDGED_ELASTIC_BAND_CALC_TYPE)
    other = Result(calculation_run_id=next(iter(document.runs)), result_type="castep_phonon")
    document.results[orphan.id] = orphan
    document.results[other.id] = other

    completed = completed_energy_barrier_neb_candidates(document)

    assert [candidate.job.label for candidate in completed] == ["a-neb", "b-neb"]
    assert completed[0].result is candidates[1]


# --- completed relaxations ----------------------------------------------------


def _completed_relaxation(
    doc: ProjectDocument,
    *,
    muon: tuple[float, float, float] | None = (0.25, 0.25, 0.25),
    data: dict | None = None,
    converged: bool | None = True,
    status: JobStatus = JobStatus.COMPLETED,
) -> tuple[Job, CalculationRun]:
    species = ["Si", "Si"] + (["H:mu"] if muon is not None else [])
    positions = [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]] + ([list(muon)] if muon is not None else [])
    structure = Structure(
        label="relaxed",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=species,
        fractional_positions=positions,
    )
    doc.structures[structure.id] = structure
    job = Job(label="job", status=status)
    run = CalculationRun(job_id=job.id, convergence_achieved=converged)
    result = Result(calculation_run_id=run.id, data=data or {}, derived_structure_ids=[structure.id])
    run.result_id = result.id
    doc.jobs[job.id] = job
    doc.runs[run.id] = run
    doc.results[result.id] = result
    return job, run


def test_completed_muon_relaxation_sources_filter_and_sort(document: ProjectDocument) -> None:
    runs: dict[UUID, CalculationRun] = {}
    energies: dict[UUID, float | None] = {}

    def add(label: str, energy: float | None, **kwargs) -> tuple[str, str, UUID | None, Job]:
        job, run = _completed_relaxation(document, **kwargs)
        runs[job.id] = run
        energies[job.id] = energy
        return (label, POSITIVE_MUON_CHARGE_STATE, uuid4(), job)

    high = add("high", -1.0, data={"muon_final_fractional_position_host": [0.1, 0.2, 0.3]})
    low = add("Low", -2.0)
    unknown = add("unknown", None)
    unconverged = add("unconverged", -5.0, converged=False)
    running = add("running", -5.0, status=JobStatus.RUNNING)
    no_muon = add("no-muon", -5.0, muon=None)
    rows = [high, low, unknown, unconverged, running, no_muon, high, ("none", "x", None, None)]

    sources = completed_muon_relaxation_sources(
        document,
        rows,
        run_for_job=runs.get,
        total_energy_for_job=energies.get,
    )

    assert [source.job_label for source in sources] == ["Low", "high", "unknown"]
    assert sources[1].folded_fractional_position == (0.1, 0.2, 0.3)
    assert sources[0].folded_fractional_position == pytest.approx((0.25, 0.25, 0.25))
    assert sources[0].site_id == low[2]
    assert sources[0].total_energy_ev == -2.0


def test_relaxation_sources_for_labels_accepts_keyed_and_bare_labels() -> None:
    def source(label: str, charge_state: str) -> CompletedRelaxationAnalysisSource:
        return CompletedRelaxationAnalysisSource(
            job_id=uuid4(),
            job_label=label,
            charge_state=charge_state,
            structure_id=uuid4(),
            site_id=None,
            folded_fractional_position=(0.0, 0.0, 0.0),
            total_energy_ev=None,
        )

    plus = source("a", POSITIVE_MUON_CHARGE_STATE)
    mu = source("a", MUONIUM_CHARGE_STATE)
    other = source("b", POSITIVE_MUON_CHARGE_STATE)
    keyed = f"{muon_charge_state_prefix(MUONIUM_CHARGE_STATE)}:a"

    assert relaxation_sources_for_labels([plus, mu, other], [keyed]) == [mu]
    assert relaxation_sources_for_labels([plus, mu, other], ["b"]) == [other]


@dataclass
class _Row:
    label: str
    parameter_values: dict = field(default_factory=dict)
    supercell_matrix: tuple | None = None


@dataclass
class _State:
    status: str
    job_id: UUID | None
    total_energy: float | None = None
    lattice_a: float | None = None
    lattice_b: float | None = None
    lattice_c: float | None = None


def test_completed_pristine_relaxation_jobs_describe_completed_rows(document: ProjectDocument) -> None:
    job_unit, run_unit = _completed_relaxation(document, muon=None)
    job_super, run_super = _completed_relaxation(document, muon=None)
    runs = {job_unit.id: run_unit, job_super.id: run_super}
    rows = [
        _Row("unit", {"cut_off_energy": 500}),
        _Row("super", supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1))),
    ]
    states = {
        "unit": _State(JobStatus.COMPLETED.value, job_unit.id, total_energy=-3.0, lattice_a=5.1),
        "super": _State(JobStatus.COMPLETED.value, job_super.id),
        "queued": _State(JobStatus.QUEUED.value, uuid4()),
        "unbound": _State(JobStatus.COMPLETED.value, None),
        "lost": _State(JobStatus.COMPLETED.value, uuid4()),
    }

    completed = completed_pristine_relaxation_jobs(document, rows, states, run_for_job=runs.get)

    assert [job.job_label for job in completed] == ["super", "unit"]
    super_job, unit_job = completed
    assert super_job.supercell_label == "[2,0,0; 0,2,0; 0,0,1]"
    assert unit_job.supercell_label == "Unit cell"
    assert unit_job.parameter_values == {"cut_off_energy": 500}
    assert unit_job.total_energy == -3.0
    assert unit_job.lattice_a == 5.1
    assert unit_job.structure_id == document.results[run_unit.result_id].derived_structure_ids[0]
