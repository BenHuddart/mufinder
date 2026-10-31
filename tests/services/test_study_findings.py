"""Findings: one record per distinct result, written by the service that holds it."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.charge_states import MUON_RELAXATION_CALC_TYPES, MUONIUM_CHARGE_STATE
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationDefinition,
    CalculationRun,
    Job,
    ProvenanceRecord,
    Result,
)
from mufinder.persistence.json_store import ProjectStore
from mufinder.services.job_execution_controller import JobExecutionController
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    store_magnetic_field_results,
)
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.source_candidates import eligible_quantum_source_candidates
from mufinder.services.study_findings import (
    FINGERPRINT_KEY,
    SUBJECT_KEY,
    input_fingerprint,
    record_finding,
    record_uep_minima,
)
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    NUDGED_ELASTIC_BAND_SET_CALC_TYPE,
)
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_PHONON_SET_CALC_TYPE
from tests.support.muon_sites import add_relaxed_muon_site


@pytest.fixture
def service() -> ProjectService:
    return ProjectService()


@pytest.fixture
def document(service: ProjectService, tmp_path: Path) -> ProjectDocument:
    return service.create_project(tmp_path / "findings.mufinder", "Findings")


def _actions(document: ProjectDocument, action: str) -> list[ProvenanceRecord]:
    return [record for record in document.provenance if record.action == action]


# --- the fingerprint -----------------------------------------------------------


def _finding(fingerprint: str, subject: str = "the subject", **fields: object) -> ProvenanceRecord:
    return ProvenanceRecord(
        action="a_finding",
        parameters={SUBJECT_KEY: subject, FINGERPRINT_KEY: fingerprint},
        **fields,
    )


def test_a_repeat_on_the_same_day_is_skipped_and_a_later_day_records_it(document) -> None:
    first = _finding("same")

    assert record_finding(document, first) is True
    assert record_finding(document, _finding("same", timestamp=first.timestamp)) is False
    later = first.timestamp + timedelta(days=1)
    assert record_finding(document, _finding("same", timestamp=later)) is True

    assert len(_actions(document, "a_finding")) == 2


def test_going_back_to_an_earlier_result_is_a_new_finding_so_the_last_is_current(document) -> None:
    now = _finding("A").timestamp

    assert record_finding(document, _finding("A", timestamp=now)) is True
    assert record_finding(document, _finding("B", timestamp=now)) is True
    assert record_finding(document, _finding("A", timestamp=now)) is True
    # A straight repeat of the latest still adds nothing.
    assert record_finding(document, _finding("A", timestamp=now)) is False

    assert [record.parameters[FINGERPRINT_KEY] for record in _actions(document, "a_finding")] == [
        "A",
        "B",
        "A",
    ]


def test_only_the_latest_finding_about_the_same_subject_and_action_counts(document) -> None:
    now = _finding("A").timestamp

    assert record_finding(document, _finding("A", "one", timestamp=now)) is True
    # Another subject's finding in between does not make a repeat new.
    assert record_finding(document, _finding("B", "two", timestamp=now)) is True
    assert record_finding(document, _finding("A", "one", timestamp=now)) is False
    # The same fingerprint about another subject is its own finding.
    assert record_finding(document, _finding("A", "two", timestamp=now)) is True
    other_action = _finding("A", "one", timestamp=now)
    other_action.action = "another_finding"
    assert record_finding(document, other_action) is True


def test_a_finding_reaches_the_project_stream_on_disk(service, document) -> None:
    record_finding(document, _finding("same"))

    reopened = service.load_project(document.project_root)

    assert [record.action for record in reopened.provenance][-1] == "a_finding"


def test_the_fingerprint_is_stable_and_follows_the_inputs() -> None:
    job_id = uuid4()

    assert input_fingerprint(job_id, {"a": 1, "b": [2]}) == input_fingerprint(
        job_id, {"b": [2], "a": 1}
    )
    assert input_fingerprint(job_id, {"a": 1}) != input_fingerprint(job_id, {"a": 2})


# --- UEP batch search ----------------------------------------------------------


def _uep_state(
    *energies: float, starts: tuple[float, ...] = (0.0,), total: float | None = None
) -> dict:
    return {
        "current_tab": 1,
        "batch_results": [
            {"start_position_ang": [start, 0.0, 0.0], "total_energy_ev": total} for start in starts
        ],
        "batch_summary_data": {
            "threshold_ang": 0.2,
            "clustering_method": "legacy_cartesian_hier",
            "clusters": [
                {
                    "cluster_index": index,
                    "size": 2,
                    "member_indexes": [index, index + 1],
                    "representative_fractional_position": [0.0, 0.5, index / 10],
                    "minimum_energy_ev": energy,
                }
                for index, energy in enumerate(energies, start=1)
            ],
        },
    }


def _uep_job(document: ProjectDocument) -> Job:
    job = Job(label="uep_source", workflow_stage=WorkflowStage.UEP, calc_type="uep")
    document.jobs[job.id] = job
    return job


def test_a_new_uep_grouping_records_its_minima_lowest_first(document) -> None:
    job = _uep_job(document)

    assert record_uep_minima(document, job, {}, _uep_state(-1.0, -1.5, starts=(0.0, 1.0)))

    record = _actions(document, "uep_minima_found")[0]
    assert record.workflow_stage == WorkflowStage.UEP
    assert record.job_id == job.id
    assert record.parameters["start_count"] == 2
    assert [row["minimum_energy_eV"] for row in record.parameters["minima"]] == [-1.5, -1.0]
    assert record.parameters["minima"][0]["fractional_position"] == [0.0, 0.5, 0.2]
    assert record.parameters["includes_zpe"] is False


def test_a_uep_state_without_a_new_grouping_records_nothing(document) -> None:
    job = _uep_job(document)
    state = _uep_state(-1.0)

    assert not record_uep_minima(document, job, {}, {"batch_results": []})
    assert record_uep_minima(document, job, {}, state)
    # A turned camera emits the same grouping again.
    assert not record_uep_minima(document, job, state, {**state, "current_tab": 2})


def test_a_changed_uep_grouping_setting_is_new_even_with_the_same_clusters(document) -> None:
    job = _uep_job(document)
    state = _uep_state(-1.0)
    regrouped = _uep_state(-1.0)
    regrouped["batch_summary_data"]["threshold_ang"] = 0.3

    assert record_uep_minima(document, job, {}, state)
    assert record_uep_minima(document, job, state, regrouped)

    thresholds = [
        record.parameters["grouping"]["threshold_ang"]
        for record in _actions(document, "uep_minima_found")
    ]
    assert thresholds == [0.2, 0.3]


def test_a_uep_summary_saved_without_a_method_is_not_a_new_grouping_when_restored(document) -> None:
    job = _uep_job(document)
    state = _uep_state(-1.0)
    saved = _uep_state(-1.0)
    del saved["batch_summary_data"]["clustering_method"]

    assert not record_uep_minima(document, job, saved, state)


def test_repeating_a_uep_search_from_the_same_starts_records_it_once(document) -> None:
    job = _uep_job(document)

    assert record_uep_minima(document, job, {}, _uep_state(-1.0))
    # The grouping was cleared and made again.
    assert not record_uep_minima(document, job, {}, _uep_state(-1.0))
    # Adding the ZPE gives new energies from the same starts.
    assert record_uep_minima(document, job, _uep_state(-1.0), _uep_state(-0.8, total=-0.8))
    assert _actions(document, "uep_minima_found")[-1].parameters["includes_zpe"] is True
    # New random starts are a new search.
    assert record_uep_minima(document, job, {}, _uep_state(-1.0, starts=(2.0,)))

    assert len(_actions(document, "uep_minima_found")) == 3


# --- magnetic fields -----------------------------------------------------------


def _field_form(tesla: float, *, site_source: str | None = None) -> dict:
    field = {
        "site_label": "Mu 1",
        "muon_position": [0.0, 0.5, 0.0],
        "dipolar_T": [0.0, 0.0, tesla],
        "lorentz_T": [0.0, 0.0, 0.0],
        "contact_T": [0.0, 0.1, 0.0],
        "demag_T": [0.0, 0.0, 0.0],
        "applied_T": [0.0, 0.0, 0.0],
        "total_T": [0.0, 0.1, tesla],
        "magnitude_T": tesla,
    }
    value = {"kind": "commensurate", "results": [field]}
    if site_source:
        value["site_source"] = site_source
    return {
        "magnetic_field_results": value,
        "selected_muon_site": {"label": "4c · cluster 1"},
        "applied_field_T": [0.0, 0.0, 0.0],
    }


def _store(document, form: dict, *, owner: str = MAGNETIC_FIELD_PAGE_OWNER) -> None:
    store_magnetic_field_results(document, owner, form, journal_title="Magnetic Fields")


def test_a_field_calculation_records_its_site_size_and_terms(document) -> None:
    _store(document, _field_form(0.42))

    record = _actions(document, "field_calculated")[0]
    assert record.parameters["site_label"] == "4c · cluster 1"
    assert record.parameters["terms"] == ["dipolar", "contact"]
    assert record.parameters["results"][0]["magnitude_T"] == 0.42
    assert record.parameters["estimate"] is False


def test_a_field_record_keeps_the_settings_the_calculation_used(document) -> None:
    form = _field_form(0.42)
    form["magnetic_field_results"]["results"][0]["calculation_parameters"] = {
        "contact_coupling": 1.2,
        "nnn": 2,
        "rcont": 4.0,
    }
    form["supercell"] = [10, 10, 10]
    form["include_equivalent_sites"] = True

    _store(document, form)

    settings = _actions(document, "field_calculated")[0].parameters["settings"]
    assert settings["contact_coupling_per_A3"] == 1.2
    assert settings["contact_neighbours"] == 2
    assert settings["contact_radius_A"] == 4.0
    assert settings["supercell"] == [10, 10, 10]
    assert settings["equivalent_sites"] is True
    assert settings["distortion_corrected"] is False


def test_each_distinct_field_result_is_recorded_once(document) -> None:
    _store(document, _field_form(0.42))
    _store(document, _field_form(0.42))
    # Choosing another site or setting clears the results; calculating again gives the same values.
    _store(document, {"magnetic_field_results": None})
    _store(document, _field_form(0.42))
    _store(document, _field_form(0.5))

    assert [
        record.parameters["results"][0]["magnitude_T"]
        for record in _actions(document, "field_calculated")
    ] == [0.42, 0.5]


def test_going_back_to_an_earlier_field_adds_a_third_entry(document) -> None:
    for tesla in (0.42, 0.5, 0.42, 0.42):
        _store(document, _field_form(tesla))

    assert [
        record.parameters["results"][0]["magnitude_T"]
        for record in _actions(document, "field_calculated")
    ] == [0.42, 0.5, 0.42]


def test_a_field_source_added_to_a_stored_record_is_not_a_calculation(document) -> None:
    _store(document, _field_form(0.42))
    _store(document, _field_form(0.42, site_source="muon_relaxation"))

    assert len(_actions(document, "field_calculated")) == 1


def test_results_copied_to_a_new_workspace_record_nothing(document) -> None:
    store_magnetic_field_results(document, "workspace", _field_form(0.42))

    assert _actions(document, "field_calculated") == []


def test_a_distribution_record_leaves_out_the_samples(document) -> None:
    form = {
        "magnetic_field_results": {
            "kind": "distribution",
            "results": [
                {
                    "site_label": "Mu 1",
                    "muon_position": [0.0, 0.5, 0.0],
                    "fields_T": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
                    "mean_T": [0.0, 0.0, 0.0],
                    "total_rms_T": 0.1,
                    "terms": {"dipolar": {"mean_T": [0.0, 0.0, 0.0], "rms_T": 0.1}},
                    "convergence": [{"converged": False, "message": "Too few samples."}],
                }
            ],
        }
    }

    _store(document, form)

    row = _actions(document, "field_calculated")[0].parameters["results"][0]
    assert "fields_T" not in row
    assert row["total_rms_T"] == 0.1
    assert row["terms"] == {"dipolar": 0.1}
    assert row["not_converged"] == ["Too few samples."]


# --- jobs: quantum corrections and energy barriers -------------------------------


def _controller(document: ProjectDocument, service: ProjectService) -> JobExecutionController:
    return JobExecutionController(
        document_getter=lambda: document,
        project_service=service,
        local_runner=None,
        cluster_runner=None,
        job_output_directory=lambda _job: None,
        job_seed_name=lambda job: job.label,
    )


def _finish(
    document: ProjectDocument, service: ProjectService, job: Job, data: dict, calc_type: str
) -> None:
    """Complete ``job`` with a result, as the shared completion path does."""

    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, status=JobStatus.RUNNING)
    document.runs[run.id] = run
    result = Result(calculation_run_id=run.id, result_type=calc_type, data=data)
    document.results[result.id] = result
    _controller(document, service).complete_job_with_result(
        job, result=result, warnings=[], convergence_achieved=None
    )


def _phonon_rows(document: ProjectDocument) -> dict[str, Job]:
    """One phonon row per eligible relaxed site, by site label."""

    jobs: dict[str, Job] = {}
    mapping: dict[str, str] = {}
    for candidate in eligible_quantum_source_candidates(document):
        job = Job(
            label=f"phonon_{candidate.site.label}",
            workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
            calc_type=QUANTUM_PHONON_CALC_TYPE,
        )
        document.jobs[job.id] = job
        jobs[candidate.site.label] = job
        mapping[candidate.key] = str(job.id)
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_SET_CALC_TYPE,
        parameters={"phonon_job_ids_by_label": mapping},
    )
    document.calculations[definition.id] = definition
    return jobs


def _zpe_data(zpe: float, **extra: object) -> dict:
    return {
        "muon_zpe_total_eV": zpe,
        "muon_zpe_modes": [
            {"mode_index": 1, "frequency_cm": 3000.0, "participation": 0.9, "width_A": 0.2}
        ],
        "w1_entanglement_witness": -0.01,
        "w1_definition": "participation_minus_3",
        **extra,
    }


def _finish_phonon(document, service, job: Job, zpe: float, **extra: object) -> None:
    _finish(document, service, job, _zpe_data(zpe, **extra), QUANTUM_PHONON_CALC_TYPE)


def test_a_phonon_result_records_the_site_zpe_and_its_modes(document, service) -> None:
    add_relaxed_muon_site(document, label="A", total_energy=-100.0)
    job = _phonon_rows(document)["A"]

    _finish_phonon(document, service, job, 0.68, phonon_warnings=["Check the muon mass."])

    record = _actions(document, "quantum_corrected")[0]
    assert record.job_id == job.id
    assert record.parameters["site_label"] == "A"
    assert record.parameters["charge_state_label"] == "positive muon"
    assert record.parameters["muon_zpe_total_eV"] == 0.68
    assert record.parameters["mode_frequencies_cm"] == [3000.0]
    assert record.parameters["warnings"] == ["Check the muon mass."]
    # One site has nothing to be ordered against.
    assert record.parameters["ordering_changed"] is None


@pytest.mark.parametrize(
    ("zpe_a", "zpe_b", "changed", "lowest"),
    [(0.6, 0.6, False, "A"), (0.9, 0.5, True, "B")],
    ids=["unchanged", "swapped"],
)
def test_a_phonon_result_says_whether_the_zpe_reorders_the_sites(
    document, service, zpe_a: float, zpe_b: float, changed: bool, lowest: str
) -> None:
    add_relaxed_muon_site(document, label="A", total_energy=-100.0)
    add_relaxed_muon_site(document, label="B", total_energy=-99.9)
    jobs = _phonon_rows(document)

    _finish_phonon(document, service, jobs["A"], zpe_a)
    _finish_phonon(document, service, jobs["B"], zpe_b)

    first, second = _actions(document, "quantum_corrected")
    assert first.parameters["ordering_changed"] is None
    assert second.parameters["ordering_changed"] is changed
    comparison = second.parameters["comparison"]
    assert comparison[0]["site_label"] == lowest
    assert comparison[0]["corrected_energy_eV"] == 0.0
    by_label = {row["site_label"]: row for row in comparison}
    assert by_label["A"]["relative_energy_eV"] == pytest.approx(0.0)
    assert by_label["B"]["relative_energy_eV"] == pytest.approx(0.1)


def test_sites_in_another_charge_state_are_not_compared(document, service) -> None:
    add_relaxed_muon_site(document, label="A", total_energy=-100.0)
    add_relaxed_muon_site(
        document,
        label="B",
        total_energy=-101.0,
        charge_calc_type=MUON_RELAXATION_CALC_TYPES[MUONIUM_CHARGE_STATE],
    )
    jobs = _phonon_rows(document)

    _finish_phonon(document, service, jobs["A"], 0.6)
    _finish_phonon(document, service, jobs["B"], 0.5)

    assert [
        record.parameters["ordering_changed"] for record in _actions(document, "quantum_corrected")
    ] == [None, None]


def test_running_a_phonon_job_again_with_the_same_inputs_records_it_once(document, service) -> None:
    add_relaxed_muon_site(document, label="A", total_energy=-100.0)
    job = _phonon_rows(document)["A"]

    _finish_phonon(document, service, job, 0.68)
    _finish_phonon(document, service, job, 0.68)

    assert len(_actions(document, "quantum_corrected")) == 1


def test_a_phonon_job_run_with_changed_settings_is_a_new_finding(document, service) -> None:
    add_relaxed_muon_site(document, label="A", total_energy=-100.0)
    job = _phonon_rows(document)["A"]
    _finish_phonon(document, service, job, 0.68)

    definition = next(iter(document.calculations.values()))
    key = next(iter(definition.parameters["phonon_job_ids_by_label"]))
    definition.parameters["configuration"] = {
        "row_settings": {key: {"parameter_values": {"cut_off_energy": 900.0}}}
    }
    _finish_phonon(document, service, job, 0.7)

    assert len(_actions(document, "quantum_corrected")) == 2


def test_a_phonon_result_without_a_zpe_records_nothing(document, service) -> None:
    add_relaxed_muon_site(document, label="A", total_energy=-100.0)
    job = _phonon_rows(document)["A"]

    _finish(document, service, job, {}, QUANTUM_PHONON_CALC_TYPE)

    assert _actions(document, "quantum_corrected") == []


def _neb_job(document: ProjectDocument, charge_state: str = "muon_plus") -> Job:
    """A NEB job bound to a generated path between two relaxed sites."""

    reactant = add_relaxed_muon_site(document, label="4c")
    product = add_relaxed_muon_site(document, label="T_A")
    job = Job(
        label="barrier_neb",
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
    )
    document.jobs[job.id] = job
    key = f"{reactant.id}:{product.id}:1"
    document.ui_state["energy_barriers_form"] = {
        "generated_sources": [
            {
                "key": key,
                "reactant_site_id": str(reactant.id),
                "product_site_id": str(product.id),
                "path_length_ang": 2.5,
                "charge_state": charge_state,
            }
        ]
    }
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type=NUDGED_ELASTIC_BAND_SET_CALC_TYPE,
        parameters={"job_ids_by_label": {key: str(job.id)}},
    )
    document.calculations[definition.id] = definition
    return job


def _neb_data() -> dict:
    return {"barrier_height_eV": 0.31, "num_images": 3, "image_energies_eV": [-5.0, -4.69, -4.9]}


def test_a_neb_result_records_the_path_and_its_barriers_once(document, service) -> None:
    job = _neb_job(document)

    _finish(document, service, job, _neb_data(), NUDGED_ELASTIC_BAND_CALC_TYPE)
    _finish(document, service, job, _neb_data(), NUDGED_ELASTIC_BAND_CALC_TYPE)

    records = _actions(document, "barrier_calculated")
    assert len(records) == 1
    parameters = records[0].parameters
    assert (parameters["start_label"], parameters["end_label"]) == ("4c", "T_A")
    assert parameters["charge_state_label"] is not None
    assert parameters["forward_barrier_eV"] == pytest.approx(0.31)
    assert parameters["reverse_barrier_eV"] == pytest.approx(0.21)
    assert parameters["endpoint_delta_eV"] == pytest.approx(0.1)
    assert parameters["num_images"] == 3
    assert parameters["path_length_ang"] == 2.5


def test_a_neb_result_with_no_energies_records_nothing(document, service) -> None:
    job = _neb_job(document)

    _finish(document, service, job, {"num_images": 0}, NUDGED_ELASTIC_BAND_CALC_TYPE)

    assert _actions(document, "barrier_calculated") == []


def test_other_results_record_no_finding(document, service) -> None:
    job = Job(label="relax", calc_type="castep_geometry_optimization")
    document.jobs[job.id] = job

    _finish(document, service, job, {"total_energy": -1.0}, job.calc_type)

    assert _actions(document, "quantum_corrected") == []
    assert _actions(document, "barrier_calculated") == []


def test_a_finding_that_cannot_be_written_keeps_the_result(document, monkeypatch) -> None:
    def refuse(_store, _payload) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(ProjectStore, "append_provenance", refuse)

    assert record_finding(document, _finding("same")) is False
    assert _actions(document, "a_finding") == []

    # Nothing was written, so the same finding can still be recorded once the disk recovers.
    monkeypatch.undo()
    assert record_finding(document, _finding("same")) is True


def test_a_failing_finding_does_not_undo_the_job_completion(document, service, monkeypatch) -> None:
    import mufinder.services.job_execution_controller as controller_module

    def explode(*_args: object) -> bool:
        raise RuntimeError("boom")

    monkeypatch.setattr(controller_module, "record_job_finding", explode)
    job = Job(label="relax", calc_type="castep_geometry_optimization")
    document.jobs[job.id] = job

    _finish(document, service, job, {}, job.calc_type)

    assert job.status == JobStatus.COMPLETED


def test_a_row_run_in_a_newer_definition_replaces_its_older_zpe(document, service) -> None:
    add_relaxed_muon_site(document, label="A", total_energy=-100.0)
    add_relaxed_muon_site(document, label="B", total_energy=-99.9)
    jobs = _phonon_rows(document)
    _finish_phonon(document, service, jobs["B"], 0.1)
    # Row B is bound to a new job, which has no result yet.
    definition = next(iter(document.calculations.values()))
    newer = CalculationDefinition(
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_SET_CALC_TYPE,
        created_at=definition.created_at + timedelta(seconds=1),
        parameters={
            "phonon_job_ids_by_label": {
                key: (str(uuid4()) if job_id == str(jobs["B"].id) else job_id)
                for key, job_id in definition.parameters["phonon_job_ids_by_label"].items()
            }
        },
    )
    document.calculations[newer.id] = newer

    _finish_phonon(document, service, jobs["A"], 0.6)

    record = _actions(document, "quantum_corrected")[-1]
    assert [row["site_label"] for row in record.parameters["comparison"]] == ["A"]
    assert record.parameters["ordering_changed"] is None
