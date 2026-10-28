"""Source-candidate queries over a project document."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
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
from mufinder.services.project_summary import latest_run_for_job
from mufinder.services.source_candidates import (
    CompletedRelaxationAnalysisSource,
    charge_state_for_job,
    charge_state_label_for_job,
    coerce_fractional_triplet,
    completed_energy_barrier_neb_candidates,
    completed_muon_relaxation_sources,
    completed_pristine_relaxation_jobs,
    eligible_energy_barrier_source_candidates,
    eligible_quantum_source_candidates,
    energy_barrier_form_state,
    energy_barrier_generated_sources,
    energy_barrier_max_path_length_text,
    energy_barrier_neb_result_path_label,
    energy_barrier_neb_source_candidates,
    energy_barrier_neb_sources_by_job_id,
    energy_barrier_path_preview_structure,
    energy_barrier_source_label,
    enumerate_energy_barrier_source_candidates,
    float_from_source,
    fractional_triplet_from_source,
    quantum_seed_name_for_key,
    quantum_source_candidate_by_key,
    quantum_source_key,
    quantum_source_label,
    quantum_source_labels_by_key,
    quantum_source_relative_energies,
    relaxation_sources_for_labels,
    selected_energy_barrier_endpoint_keys,
    serialize_energy_barrier_neb_sources,
    serialize_energy_barrier_source_candidates,
    serialize_quantum_sources,
    site_charge_state_relaxation_jobs,
    source_relaxation_parameter_values,
    uuid_from_source,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
)
from tests.support.muon_sites import bind_muon_relaxation_row
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
    positions = (
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]] + [[0.1, 0.1, 0.1]] * len(extra_species) + [list(muon)]
    )
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
    assert (
        charge_state_for_job(Job(calc_type=muon_relaxation_calc_type(MUONIUM_CHARGE_STATE)))
        == MUONIUM_CHARGE_STATE
    )
    assert (
        charge_state_for_job(Job(calc_type=muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE)))
        == POSITIVE_MUON_CHARGE_STATE
    )
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
    assert (
        completed_muon_relaxation_sources(
            None, [], run_for_job=lambda _id: None, total_energy_for_job=lambda _id: None
        )
        == []
    )
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


def test_charge_state_label_for_job_names_muon_relaxations_only() -> None:
    assert (
        charge_state_label_for_job(Job(calc_type=muon_relaxation_calc_type(MUONIUM_CHARGE_STATE)))
        == "muonium"
    )
    assert charge_state_label_for_job(
        Job(calc_type=muon_relaxation_calc_type(POSITIVE_MUON_CHARGE_STATE))
    ) == ("positive muon")
    assert charge_state_label_for_job(Job(calc_type="castep_uep_source")) == ""
    assert charge_state_label_for_job(None) == ""


def test_quantum_source_labels_seed_names_and_stored_sources(document: ProjectDocument) -> None:
    first = _add_relaxed_source(document, "site", energy=-10.0)
    second = _add_relaxed_source(
        document, "site", energy=-9.0, charge_state=POSITIVE_MUON_CHARGE_STATE
    )
    second_candidate = quantum_source_candidate_by_key(document, str(second.id))
    assert second_candidate is not None
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type="castep_quantum_phonon_set",
        parameters={"job_seed_names_by_label": {str(first.id): " quantum_kept "}},
    )

    assert quantum_source_label(document, second_candidate) == "site_2"
    assert quantum_seed_name_for_key(document, definition, str(first.id)) == "quantum_kept"
    assert quantum_seed_name_for_key(document, definition, str(second.id)) == "quantum_site_2"
    assert quantum_seed_name_for_key(document, None, "missing") == "quantum"
    stored = serialize_quantum_sources(document, [second.id, uuid4()])
    assert stored == [
        {
            "key": str(second.id),
            "site_id": str(second.id),
            "source_relaxation_run_id": str(second_candidate.run.id),
            "source_relaxation_job_id": str(second_candidate.job.id),
            "relaxed_structure_id": str(second_candidate.structure.id),
            "label": "site_2",
            "charge_state": POSITIVE_MUON_CHARGE_STATE,
        }
    ]


def _relaxation_job_with_inputs(
    tmp_path: Path, *, param: str | None, cell: str | None = None, **job_fields
) -> Job:
    job = Job(label="relax_site_1", **job_fields)
    output_dir = tmp_path / job.label
    output_dir.mkdir()
    if param is not None:
        (output_dir / f"{job.label}.param").write_text(param, encoding="utf-8")
    if cell is not None:
        (output_dir / f"{job.label}.cell").write_text(cell, encoding="utf-8")
    return job


def test_follow_on_rows_inherit_the_source_relaxation_functional(tmp_path: Path) -> None:
    job = _relaxation_job_with_inputs(
        tmp_path,
        param="TASK : GEOMETRYOPTIMIZATION\nXC_FUNCTIONAL : PBESOL\nCUT_OFF_ENERGY : 650 eV\n",
    )

    values = source_relaxation_parameter_values(job, lambda _job: tmp_path / job.label)

    assert values["xc_functional"] == "PBESOL"
    assert values["cut_off_energy"].value == pytest.approx(650.0)


_SPIN_SOURCE_CELL = (
    "%BLOCK LATTICE_CART\n2.87 0 0\n0 2.87 0\n0 0 2.87\n%ENDBLOCK LATTICE_CART\n"
    "%BLOCK POSITIONS_FRAC\nFe 0 0 0{spin}\nFe 0.5 0.5 0.5{spin}\n%ENDBLOCK POSITIONS_FRAC\n"
    "KPOINT_MP_GRID 4 4 2\n"
)


@pytest.mark.parametrize(
    ("param_lines", "cell_spin", "expected"),
    [
        pytest.param("SPIN_POLARISED : TRUE\n", "", True, id="explicit-english-spelling"),
        pytest.param("SPIN_POLARIZED : FALSE\n", " SPIN=2", False, id="explicit-false-wins"),
        pytest.param("", " SPIN=2", True, id="implied-by-atomic-spins"),
        pytest.param("SPIN : -2\n", "", True, id="implied-by-net-spin"),
        pytest.param("", "", False, id="not-spin-polarised"),
    ],
)
def test_follow_on_rows_inherit_the_source_relaxation_spin_treatment(
    tmp_path: Path, param_lines: str, cell_spin: str, expected: bool
) -> None:
    job = _relaxation_job_with_inputs(
        tmp_path,
        param="TASK : GEOMETRYOPTIMIZATION\n" + param_lines,
        cell=_SPIN_SOURCE_CELL.format(spin=cell_spin),
    )

    values = source_relaxation_parameter_values(job, lambda _job: tmp_path / job.label)

    assert values["spin_polarized"] is expected
    assert values["kpoint_mp_grid"] == (4, 4, 2)


@pytest.mark.parametrize(
    ("charge_state", "expected"),
    [(MUONIUM_CHARGE_STATE, {"spin_polarized": True}), (POSITIVE_MUON_CHARGE_STATE, {})],
)
def test_unreadable_source_param_defaults_only_muonium_to_spin_polarised(
    tmp_path: Path, charge_state: str, expected: dict[str, object]
) -> None:
    job = _relaxation_job_with_inputs(
        tmp_path, param=None, calc_type=muon_relaxation_calc_type(charge_state)
    )

    assert source_relaxation_parameter_values(job, lambda _job: tmp_path / job.label) == expected
    assert source_relaxation_parameter_values(job, lambda _job: None) == {}


# --- energy-barrier sources ---------------------------------------------------


def _serialized_path(
    reactant: MuonSite, product: MuonSite, doc: ProjectDocument, **overrides
) -> dict:
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
        parameters={
            "configuration": {"sources": [{"key": "a", "label": "definition"}, {"key": "c"}]}
        },
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
    assert [candidate.key for candidate in energy_barrier_neb_source_candidates(document)] == [
        "a",
        "b",
        "c",
    ]


def test_eligible_energy_barrier_sources_rebuild_endpoints(document: ProjectDocument) -> None:
    reactant = _add_relaxed_source(document, "A", energy=-10.0)
    product = _add_relaxed_source(document, "B", energy=-9.0, muon=(0.75, 0.25, 0.25))
    valid = _serialized_path(reactant, product, document)
    missing_site = _serialized_path(
        reactant, product, document, key="gone", product_site_id=str(uuid4())
    )
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

    candidates = enumerate_energy_barrier_source_candidates(
        document, str(reactant.id), str(product.id)
    )

    assert candidates
    assert [candidate.path_index for candidate in candidates] == list(range(1, len(candidates) + 1))
    assert candidates[0].key == f"{reactant.id}:{product.id}:1"
    assert candidates[0].reactant.site is reactant
    assert (
        enumerate_energy_barrier_source_candidates(document, str(reactant.id), str(muonium.id))
        == []
    )
    assert enumerate_energy_barrier_source_candidates(document, str(reactant.id), "missing") == []


def test_energy_barrier_form_state_readers(document: ProjectDocument) -> None:
    first = _add_relaxed_source(document, "A", energy=-10.0)
    second = _add_relaxed_source(document, "B", energy=-9.0)

    assert energy_barrier_form_state(document) == {}
    assert energy_barrier_max_path_length_text(document) == ""
    assert selected_energy_barrier_endpoint_keys(document) == (str(first.id), str(second.id))

    document.ui_state["energy_barriers_form"] = {
        "max_path_length_ang": 4.25,
        "selected_product_key": str(first.id),
    }
    state = energy_barrier_form_state(document)
    state["changed"] = True
    assert "changed" not in document.ui_state["energy_barriers_form"]  # a copy
    assert energy_barrier_max_path_length_text(document) == "4.25"
    assert selected_energy_barrier_endpoint_keys(document) == (str(first.id), str(first.id))
    # A stored endpoint the pickers do not offer falls back to the offered keys.
    document.ui_state["energy_barriers_form"] = {
        "selected_reactant_key": str(first.id),
        "selected_product_key": str(second.id),
    }
    assert selected_energy_barrier_endpoint_keys(document, [str(second.id)]) == (
        str(second.id),
        str(second.id),
    )
    assert selected_energy_barrier_endpoint_keys(document, [str(second.id), str(first.id)]) == (
        str(first.id),
        str(second.id),
    )
    assert selected_energy_barrier_endpoint_keys(document, []) == ("", "")
    document.ui_state["energy_barriers_form"] = {"max_path_length_ang": "far"}
    assert energy_barrier_max_path_length_text(document) == ""
    assert energy_barrier_form_state(None) == {}


def test_generated_paths_serialize_label_and_preview(document: ProjectDocument) -> None:
    reactant = _add_relaxed_source(document, "A", energy=-10.0)
    product = _add_relaxed_source(document, "B", energy=-9.0, muon=(0.75, 0.25, 0.25))
    candidates = enumerate_energy_barrier_source_candidates(
        document, str(reactant.id), str(product.id)
    )
    first = candidates[0]

    assert energy_barrier_source_label(first) == "A_to_B_path1"
    stored = serialize_energy_barrier_source_candidates(candidates)[0]
    assert stored["key"] == first.key
    assert stored["label"] == "A_to_B_path1"
    assert stored["charge_state"] == POSITIVE_MUON_CHARGE_STATE
    assert stored["path_start_fractional"] == list(first.path.start_fractional)

    preview = energy_barrier_path_preview_structure(document, first)
    assert preview.label == "A_to_B_path1 preview"
    assert preview.origin == "energy_barriers_path_preview"
    assert preview.parent_structure_id == document.project.host_structure_id
    with pytest.raises(ValueError, match="No project document"):
        energy_barrier_path_preview_structure(None, first)
    document.project.host_structure_id = None
    with pytest.raises(ValueError, match="pristine host structure is required"):
        energy_barrier_path_preview_structure(document, first)


def test_neb_sources_map_jobs_back_to_their_paths(document: ProjectDocument) -> None:
    reactant = _add_relaxed_source(document, "A", energy=-10.0)
    product = _add_relaxed_source(document, "B", energy=-9.0, muon=(0.75, 0.25, 0.25))
    path = _serialized_path(reactant, product, document)
    document.ui_state["energy_barriers_form"] = {"generated_sources": [path, {"key": "other"}]}
    job = Job(label="barrier_neb", calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE)
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_nudged_elastic_band_set",
        parameters={
            "job_ids_by_label": {
                path["key"]: str(job.id),
                "other": "not-a-uuid",
                "gone": str(uuid4()),
            }
        },
    )

    assert serialize_energy_barrier_neb_sources(document, ["missing", path["key"]]) == [path]
    by_job_id = energy_barrier_neb_sources_by_job_id(document, [definition])
    assert by_job_id == {job.id: path}

    completed = SimpleNamespace(job=job)
    assert energy_barrier_neb_result_path_label(document, completed, path) == "A → B"
    assert energy_barrier_neb_result_path_label(document, completed, None) == "barrier_neb"
    reactant.label = ""
    assert energy_barrier_neb_result_path_label(document, completed, path) == "barrier_neb"


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
    result = Result(
        calculation_run_id=run.id, data=data or {}, derived_structure_ids=[structure.id]
    )
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


def test_completed_pristine_relaxation_jobs_describe_completed_rows(
    document: ProjectDocument,
) -> None:
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


# --- one site relaxed in both charge states -----------------------------------


def _relax_in_other_charge_state(
    doc: ProjectDocument,
    site: MuonSite,
    charge_state: str,
    *,
    energy: float,
    muon: tuple[float, float, float] = (0.25, 0.25, 0.25),
) -> Job:
    """Relax ``site`` again in ``charge_state`` and bind the job to it by a muon-relaxation row."""

    scratch = _add_relaxed_source(
        doc, site.label, energy=energy, muon=muon, charge_state=charge_state
    )
    run = doc.runs[scratch.relaxation_run_id]
    doc.results[run.result_id].data = {"total_energy": energy}
    del doc.sites[scratch.id]
    job = doc.jobs[run.job_id]
    bind_muon_relaxation_row(doc, site, job, charge_state)
    return job


def test_a_site_relaxed_in_both_charge_states_is_a_source_in_each(
    document: ProjectDocument,
) -> None:
    site = _add_relaxed_source(document, "O1", energy=-10.0)
    muonium_job = _relax_in_other_charge_state(document, site, MUONIUM_CHARGE_STATE, energy=-12.5)

    candidates = eligible_quantum_source_candidates(document)

    by_state = {charge_state_for_job(candidate.job): candidate for candidate in candidates}
    assert set(by_state) == {POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE}
    linked, muonium = by_state[POSITIVE_MUON_CHARGE_STATE], by_state[MUONIUM_CHARGE_STATE]
    # The site's own link keeps the bare site id as its key.
    assert linked.key == str(site.id)
    assert muonium.key == f"{site.id}:muonium"
    assert muonium.job is muonium_job
    assert (linked.total_energy_eV, muonium.total_energy_eV) == (-10.0, -12.5)
    assert quantum_source_key(document, site, muonium_job) == muonium.key
    assert quantum_source_labels_by_key(candidates) == {linked.key: "O1", muonium.key: "O1_muonium"}
    assert quantum_seed_name_for_key(document, None, muonium.key) == "quantum_O1_muonium"
    stored = serialize_quantum_sources(document, [muonium.key, site.id, muonium.key])
    assert [(entry["key"], entry["charge_state"]) for entry in stored] == [
        (muonium.key, MUONIUM_CHARGE_STATE),
        (str(site.id), POSITIVE_MUON_CHARGE_STATE),
    ]
    assert site_charge_state_relaxation_jobs(document) == {
        site.id: {MUONIUM_CHARGE_STATE: muonium_job}
    }


def test_the_other_charge_state_needs_a_completed_relaxation(document: ProjectDocument) -> None:
    site = _add_relaxed_source(document, "O1", energy=-10.0)
    muonium_job = _relax_in_other_charge_state(document, site, MUONIUM_CHARGE_STATE, energy=-12.5)

    muonium_job.status = JobStatus.RUNNING
    assert [candidate.key for candidate in eligible_quantum_source_candidates(document)] == [
        str(site.id)
    ]

    muonium_job.status = JobStatus.COMPLETED
    for run in [run for run in document.runs.values() if run.job_id == muonium_job.id]:
        del document.runs[run.id]
    assert [candidate.key for candidate in eligible_quantum_source_candidates(document)] == [
        str(site.id)
    ]


def test_the_other_charge_state_stays_a_source_when_the_site_link_is_cleared(
    document: ProjectDocument,
) -> None:
    site = _add_relaxed_source(document, "O1", energy=-10.0)
    _relax_in_other_charge_state(document, site, MUONIUM_CHARGE_STATE, energy=-12.5)

    site.clear_relaxation()

    candidates = eligible_quantum_source_candidates(document)
    assert [(candidate.key, charge_state_for_job(candidate.job)) for candidate in candidates] == [
        (f"{site.id}:muonium", MUONIUM_CHARGE_STATE)
    ]


def test_paths_between_two_sites_are_kept_apart_per_charge_state(document: ProjectDocument) -> None:
    first = _add_relaxed_source(document, "A", energy=-10.0, muon=(0.25, 0.25, 0.25))
    second = _add_relaxed_source(document, "B", energy=-9.0, muon=(0.75, 0.25, 0.25))
    _relax_in_other_charge_state(
        document, first, MUONIUM_CHARGE_STATE, energy=-12.0, muon=(0.25, 0.25, 0.25)
    )
    _relax_in_other_charge_state(
        document, second, MUONIUM_CHARGE_STATE, energy=-11.0, muon=(0.75, 0.25, 0.25)
    )

    positive = enumerate_energy_barrier_source_candidates(document, str(first.id), str(second.id))
    muonium = enumerate_energy_barrier_source_candidates(
        document, f"{first.id}:muonium", f"{second.id}:muonium"
    )

    assert positive and muonium
    assert positive[0].key == f"{first.id}:{second.id}:1"
    assert muonium[0].key == f"{first.id}:muonium:{second.id}:muonium:1"
    assert energy_barrier_source_label(muonium[0]) == "A_muonium_to_B_muonium_path1"
    # Stored paths rebuild their endpoints under the same keys.
    document.ui_state["energy_barriers_form"] = {
        "generated_sources": serialize_energy_barrier_source_candidates([*positive, *muonium])
    }
    rebuilt = {
        candidate.key: candidate
        for candidate in eligible_energy_barrier_source_candidates(document)
    }
    assert rebuilt[muonium[0].key].reactant.key == f"{first.id}:muonium"
    assert rebuilt[muonium[0].key].reactant.total_energy_eV == -12.0
    assert rebuilt[positive[0].key].reactant.key == str(first.id)


def test_the_bare_key_stays_with_its_charge_state_after_a_reset(document: ProjectDocument) -> None:
    from mufinder.services.site_relaxations import record_site_relaxation

    site = _add_relaxed_source(document, "O1", energy=-10.0)
    linked_run = document.runs[site.relaxation_run_id]
    record_site_relaxation(
        document,
        site,
        linked_run,
        document.jobs[linked_run.job_id],
        total_energy_eV=-10.0,
        fractional_position=None,
    )
    muonium_job = _relax_in_other_charge_state(document, site, MUONIUM_CHARGE_STATE, energy=-12.5)

    site.clear_relaxation()
    muonium_run = latest_run_for_job(document, muonium_job.id)
    assert not record_site_relaxation(
        document, site, muonium_run, muonium_job, total_energy_eV=-12.5, fractional_position=None
    )

    assert [candidate.key for candidate in eligible_quantum_source_candidates(document)] == [
        f"{site.id}:muonium"
    ]
