"""Tests for :class:`QuantumCorrectionsController`.

Covers the one-phonon-job-per-row submission and row lifecycle, ``build_result``
on a real ``.phonon`` file, the Configure dialog's editable and read-only
states, the phonon-method checks, ``validate_artifacts`` and how rows of a
project that ran the legacy singlepoint job first are read.
"""

from __future__ import annotations

from mufinder.services.job_monitoring import job_status_word
import math
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.enums import JobActivity, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, Result, Structure
from mufinder.app.quantum_corrections_widget import QuantumCorrectionsWidget
from mufinder.domain.constants import MUON_MASS_AMU
from mufinder.science.quantum_phonons import NO_REAL_FREQUENCY_ALONG_AXIS, mode_width_angstrom
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.job_findings import record_job_finding
from mufinder.services.job_paths import job_output_directory
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.source_candidates import quantum_seed_name_for_key
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE

from mufinder.app.workflow_config_codecs import (
    store_workflow_row_configuration,
    workflow_row_configuration,
)
from mufinder.app.workflow_controllers.quantum_corrections import (
    QuantumCorrectionsController,
    QuantumCorrectionsDeps,
)

from tests.support.jobs import set_job_state
from tests.support.quantum_corrections import (
    QUANTUM_PHONON_TEXT as _QUANTUM_PHONON_TEXT,
    bind_phonon as _bind_phonon,
    quantum_job as _quantum_job,
    seed_quantum_document as _seed_quantum_document,
    store_phonon_result as _store_phonon_result,
)
from tests.support.phonon_fixtures import (
    phonon_calculation_from_dynamical_matrix,
    phonon_calculation_text,
    uncoupled_muon_dynamical_matrix,
)
from tests.support.controllers import (
    start_again_by_reset,
    configure_result,
    RecordingBackend,
    completed_castep_result,
    make_context,
    no_artifacts,
)


# --------------------------------------------------------------------------- #
# QuantumCorrectionsController
# --------------------------------------------------------------------------- #
#
# Each row runs one CASTEP phonon job. ``build_result`` carries the richest
# result logic — parsing a real ``.phonon`` file into the muon ZPE, eigenvalues,
# mode participations, and the ``w1`` entanglement witness — while a legacy
# quantum-*singlepoint* job an older project still holds contributes an empty
# result. Driven below with narrow fakes (no MainWindow) and a REAL ``.phonon``
# fixture containing an ``H:mu`` muon.


class _FakeQuantumView:
    """Minimal QuantumCorrectionsWidget stand-in for controller tests."""

    def __init__(self) -> None:
        self.sources: list[list] = []
        self.site_scopes: list[dict] = []

    def set_site_scope(self, **kwargs) -> None:
        self.site_scopes.append(kwargs)

    def set_sources(self, rows) -> None:
        self.sources.append(list(rows))

    def window(self) -> None:
        return None


def _stub_quantum_deps(**overrides) -> QuantumCorrectionsDeps:
    base = dict(
        handle_structure_host_mismatch=lambda _mismatch: True,
        handle_site_host_mismatch=lambda _mismatch: True,
        refresh_site_label_views=lambda: None,
        show_warning=lambda _title, _text: None,
        shutting_down=lambda: False,
    )
    base.update(overrides)
    return QuantumCorrectionsDeps(**base)


def test_quantum_controller_handles_phonon_and_legacy_singlepoint_calc_types(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    controller = QuantumCorrectionsController(
        make_context(document=document), _stub_quantum_deps(), _FakeQuantumView()
    )
    singlepoint = _quantum_job(
        service, document, "quantum_singlepoint_s", QUANTUM_SINGLEPOINT_CALC_TYPE
    )
    phonon = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    other = service.add_job(
        document,
        label="u",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_uep_source",
    )
    # The legacy singlepoint type stays owned so an older project's job still collects.
    assert controller.handles_job(singlepoint) is True
    assert controller.handles_job(phonon) is True
    assert controller.handles_job(other) is False


def test_quantum_controller_refresh_view_without_document_clears_tables() -> None:
    view = _FakeQuantumView()
    controller = QuantumCorrectionsController(
        make_context(document=None), _stub_quantum_deps(), view
    )
    controller.refresh_view()
    assert view.sources == [[]]


def _displayed_result_row(tmp_path: Path, data: dict):
    """Return the row ``refresh_view`` shows for one site with this stored phonon payload."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    job = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    _store_phonon_result(document, job, data)
    controller = _quantum_document_controller(service, document, output_root)
    controller.refresh_view()
    return next(row for row in controller._view.sources[-1] if row.key == key)


def _displayed_witness(tmp_path: Path, data: dict) -> str:
    """Return the witness text ``refresh_view`` shows for one stored phonon payload."""

    return _displayed_result_row(tmp_path, data).witness


def test_quantum_controller_shows_the_corrected_energy_of_sites_with_a_result(
    tmp_path: Path,
) -> None:
    """Hand-checked on the GdRu2Si2 numbers: site B is dE +0.257 with ZPE 0.654 against
    site A's 0.558, so dE + ZPE is 0.257 + 0.654 - 0.558 = +0.353 eV, and a site with no
    result shows a dash."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key_a, key_b = list(candidates)
    document.sites[candidates[key_a].site.id].total_energy_eV = -100.0
    document.sites[candidates[key_b].site.id].total_energy_eV = -99.743
    controller = _quantum_document_controller(service, document, output_root)
    controller.set_show_every_site(True)
    rows = {row.key: row for row in controller._view.sources[-1]}
    assert rows[key_a].relative_energy == "0.000"
    assert rows[key_b].relative_energy == "+0.257"
    assert rows[key_b].corrected_energy == "–"
    assert rows[key_b].result_id == ""

    job_a = _bind_phonon(service, document, definition, key_a, status=JobStatus.COMPLETED)
    _store_phonon_result(document, job_a, {"muon_zpe_total_eV": 0.558})
    controller.refresh_view()
    rows = {row.key: row for row in controller._view.sources[-1]}
    assert (rows[key_a].zpe, rows[key_a].corrected_energy) == ("0.558", "0.000")
    assert (rows[key_b].zpe, rows[key_b].corrected_energy) == ("–", "–")

    job_b = _bind_phonon(service, document, definition, key_b, status=JobStatus.COMPLETED)
    result_b = _store_phonon_result(document, job_b, {"muon_zpe_total_eV": 0.654})
    controller.refresh_view()
    rows = {row.key: row for row in controller._view.sources[-1]}
    assert (rows[key_a].zpe, rows[key_a].corrected_energy) == ("0.558", "0.000")
    assert (rows[key_b].zpe, rows[key_b].corrected_energy) == ("0.654", "+0.353")
    assert rows[key_b].result_id == str(result_b.id)
    assert rows[key_b].status == "completed"


def test_quantum_controller_flags_a_stored_warning_without_changing_the_numbers(
    tmp_path: Path,
) -> None:
    """A saddle's non-positive muon mode is left out of the ZPE and warned about when stored.

    Hand-checked: 0.257 + 0.654 - 0.558 = +0.353 still reads +0.353, with the warning beside it.
    """

    warning = "Selected muon-dominated modes include non-positive frequencies."
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key_a, key_b = list(candidates)
    document.sites[candidates[key_a].site.id].total_energy_eV = -100.0
    document.sites[candidates[key_b].site.id].total_energy_eV = -99.743
    job_a = _bind_phonon(service, document, definition, key_a, status=JobStatus.COMPLETED)
    job_b = _bind_phonon(service, document, definition, key_b, status=JobStatus.COMPLETED)
    _store_phonon_result(document, job_a, {"muon_zpe_total_eV": 0.558})
    _store_phonon_result(
        document, job_b, {"muon_zpe_total_eV": 0.654, "phonon_warnings": [warning, 7, ""]}
    )
    controller = _quantum_document_controller(service, document, output_root)
    controller.refresh_view()

    rows = {row.key: row for row in controller._view.sources[-1]}
    assert rows[key_a].warnings == ()
    assert rows[key_b].warnings == (warning,)
    assert (rows[key_b].zpe, rows[key_b].corrected_energy) == ("0.654", "+0.353")
    assert rows[key_a].corrected_energy == "0.000"


def _two_sites_with_results(service, tmp_path, *, data_b=None, recorded_b=True):
    """Sites A and B at dE 0 and +0.257, with ZPEs 0.558 and 0.654 unless ``data_b`` replaces B's."""

    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key_a, key_b = list(candidates)
    document.sites[candidates[key_a].site.id].total_energy_eV = -100.0
    document.sites[candidates[key_b].site.id].total_energy_eV = -99.743
    job_a = _bind_phonon(service, document, definition, key_a, status=JobStatus.COMPLETED)
    job_b = _bind_phonon(
        service,
        document,
        definition,
        key_b,
        status=JobStatus.COMPLETED,
        recorded_source=recorded_b,
    )
    _store_phonon_result(document, job_a, {"muon_zpe_total_eV": 0.558})
    _store_phonon_result(document, job_b, data_b or {"muon_zpe_total_eV": 0.654})
    controller = _quantum_document_controller(service, document, output_root)
    return document, definition, controller, key_a, key_b, job_b


def _rows_by_key(controller) -> dict:
    controller.refresh_view()
    return {row.key: row for row in controller._view.sources[-1]}


def test_a_zpe_from_an_earlier_relaxation_run_is_marked_and_cannot_set_the_zero(
    tmp_path: Path,
) -> None:
    """A's run moves on, so A leaves the ranking and B, at +0.353 from A before, becomes the zero."""

    service = ProjectService()
    document, definition, controller, key_a, key_b, _job = _two_sites_with_results(
        service, tmp_path
    )
    assert _rows_by_key(controller)[key_b].corrected_energy == "+0.353"

    definition.parameters["phonon_source_run_ids_by_label"][key_a] = str(uuid4())
    rows = _rows_by_key(controller)

    assert (rows[key_a].zpe, rows[key_a].zpe_mark) == ("0.558", "earlier")
    assert rows[key_a].corrected_energy == "–"
    assert rows[key_a].result_id != ""
    assert any("earlier geometry" in note for note in rows[key_a].notes)
    assert rows[key_b].corrected_energy == "0.000"
    assert rows[key_b].zpe_mark == ""


def test_a_zpe_whose_source_run_cannot_be_told_stays_out_of_the_ranking(tmp_path: Path) -> None:
    """An old record with no stamps: the sum might mix geometries, so it states no ranking."""

    service = ProjectService()
    _document, _definition, controller, key_a, key_b, _job = _two_sites_with_results(
        service, tmp_path, recorded_b=False
    )

    rows = _rows_by_key(controller)

    assert (rows[key_b].zpe, rows[key_b].zpe_mark) == ("0.654", "unconfirmed")
    assert rows[key_b].corrected_energy == "–"
    assert any("does not record" in note for note in rows[key_b].notes)
    assert rows[key_a].corrected_energy == "0.000"


def test_an_old_record_is_read_from_its_timestamps(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, controller, _key_a, key_b, job_b = _two_sites_with_results(
        service, tmp_path, recorded_b=False
    )
    candidate = next(item for item in controller.candidates() if item.key == key_b)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    document.runs[candidate.run.id].completed_at = start
    job_b.created_at = start + timedelta(hours=1)
    job_b.submitted_at = start + timedelta(hours=2)

    row = _rows_by_key(controller)[key_b]
    assert (row.zpe_mark, row.corrected_energy) == ("", "+0.353")

    document.runs[candidate.run.id].completed_at = start + timedelta(hours=3)
    row = _rows_by_key(controller)[key_b]
    assert (row.zpe_mark, row.corrected_energy) == ("earlier", "–")


def test_materialize_records_the_run_it_read_and_keeps_it_across_a_later_run(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    run_id = candidates[key].run.id
    controller = _quantum_document_controller(service, document, output_root)

    (job,) = controller.materialize([key])

    assert definition.parameters["phonon_source_run_ids_by_label"] == {key: str(run_id)}
    job.status = JobStatus.COMPLETED
    _store_phonon_result(document, job, {"muon_zpe_total_eV": 0.5})
    # Another row's Configure must not move this row's record.
    controller.ensure_definition(list(candidates))
    assert definition.parameters["phonon_source_run_ids_by_label"] == {key: str(run_id)}
    assert _rows_by_key(controller)[key].zpe_mark == ""


def test_a_saddle_reads_from_the_zero_of_the_unflagged_sites(tmp_path: Path) -> None:
    """Hand-checked: A's sum -99.442 is the zero, so saddle B at -99.743 + 0.100 reads -0.201.

    Before the rule B would have been the zero, and A would have read +0.201.
    """

    service = ProjectService()
    _document, _definition, controller, key_a, key_b, _job = _two_sites_with_results(
        service,
        tmp_path,
        data_b={"muon_zpe_total_eV": 0.1, "muon_zpe_modes": [{"frequency_cm": -300.0}]},
    )

    rows = _rows_by_key(controller)

    assert rows[key_a].corrected_energy == "0.000"
    assert rows[key_b].corrected_energy == "-0.201"
    assert rows[key_b].zpe == "0.100"
    assert not any("zero is the lowest of them" in note for note in rows[key_b].notes)


def test_a_group_of_only_saddles_falls_back_to_them_and_says_so(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key_a, key_b = list(candidates)
    document.sites[candidates[key_a].site.id].total_energy_eV = -100.0
    document.sites[candidates[key_b].site.id].total_energy_eV = -99.743
    for key, zpe in ((key_a, 0.558), (key_b, 0.654)):
        job = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
        _store_phonon_result(
            document,
            job,
            {"muon_zpe_total_eV": zpe, "muon_zpe_modes": [{"frequency_cm": -300.0}]},
        )
    controller = _quantum_document_controller(service, document, output_root)

    rows = _rows_by_key(controller)

    assert rows[key_a].corrected_energy == "0.000"
    assert rows[key_b].corrected_energy == "+0.353"
    for key in (key_a, key_b):
        assert any("zero is the lowest of them" in note for note in rows[key].notes)


def _ranked_in_each_view(document, controller, job) -> tuple[dict, dict]:
    """Each site's ``dE + ZPE`` as the table and the Journal state it."""

    table = {
        row.label: float(row.corrected_energy)
        for row in _rows_by_key(controller).values()
        if row.corrected_energy != "–"
    }
    result = next(
        result
        for result in document.results.values()
        if result.result_type == QUANTUM_PHONON_CALC_TYPE
        and document.runs[result.calculation_run_id].job_id == job.id
    )
    assert record_job_finding(document, job, result)
    record = next(r for r in reversed(document.provenance) if r.action == "quantum_corrected")
    journal = {
        row["site_label"]: round(row["corrected_energy_eV"], 3)
        for row in record.parameters["comparison"]
    }
    return table, journal


def test_a_stale_zpe_gives_the_same_ranking_in_the_table_and_the_journal(
    tmp_path: Path,
) -> None:
    """Hand-checked: A's run moves on, so B alone is ranked, and is the zero."""

    service = ProjectService()
    document, definition, controller, key_a, _key_b, job_b = _two_sites_with_results(
        service, tmp_path
    )
    definition.parameters["phonon_source_run_ids_by_label"][key_a] = str(uuid4())

    table, journal = _ranked_in_each_view(document, controller, job_b)

    assert table == journal == {"siteB": 0.0}


def test_a_flagged_saddle_gives_the_same_ranking_in_the_table_and_the_journal(
    tmp_path: Path,
) -> None:
    """Hand-checked: A's sum -99.442 is the zero, so saddle B at -99.743 + 0.100 reads -0.201."""

    service = ProjectService()
    document, _definition, controller, _key_a, _key_b, job_b = _two_sites_with_results(
        service,
        tmp_path,
        data_b={"muon_zpe_total_eV": 0.1, "muon_zpe_modes": [{"frequency_cm": -300.0}]},
    )

    table, journal = _ranked_in_each_view(document, controller, job_b)

    assert table == journal == {"siteA": 0.0, "siteB": -0.201}


def test_quantum_controller_keeps_a_row_for_a_result_whose_site_is_gone(tmp_path: Path) -> None:
    """Hand-checked: the sites rank 0.000 and +0.353 as if the 0.100 eV orphan were not there."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key_a, key_b = list(candidates)
    document.sites[candidates[key_a].site.id].total_energy_eV = -100.0
    document.sites[candidates[key_b].site.id].total_energy_eV = -99.743
    job_a = _bind_phonon(service, document, definition, key_a, status=JobStatus.COMPLETED)
    job_b = _bind_phonon(service, document, definition, key_b, status=JobStatus.COMPLETED)
    _store_phonon_result(document, job_a, {"muon_zpe_total_eV": 0.558})
    _store_phonon_result(document, job_b, {"muon_zpe_total_eV": 0.654})
    orphan_job = _quantum_job(service, document, "quantum_phonon_old", QUANTUM_PHONON_CALC_TYPE)
    orphan_job.status = JobStatus.COMPLETED
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    _store_phonon_result(document, orphan_job, {"muon_zpe_total_eV": 0.9}, start)
    orphan = _store_phonon_result(
        document,
        orphan_job,
        {"muon_zpe_total_eV": 0.1, "w1_entanglement_witness": -0.25, "w1_definition": "p"},
        start + timedelta(minutes=1),
    )
    jobless_run = CalculationRun(status=JobStatus.COMPLETED)
    document.runs[jobless_run.id] = jobless_run
    jobless = Result(
        result_type=QUANTUM_PHONON_CALC_TYPE,
        calculation_run_id=jobless_run.id,
        data={"muon_zpe_total_eV": 0.2},
    )
    document.results[jobless.id] = jobless
    controller = _quantum_document_controller(service, document, output_root)
    controller.set_show_every_site(True)

    rows = controller._view.sources[-1]
    assert [row.corrected_energy for row in rows[:2]] == ["0.000", "+0.353"]
    first, second = rows[2:]
    assert first.label == "quantum_phonon_old (no eligible site)"
    assert (first.key, first.result_id) == ("", str(orphan.id))
    assert (first.zpe, first.witness) == ("0.100", "-0.250")
    assert (first.relative_energy, first.corrected_energy) == ("–", "–")
    assert first.status == "completed"
    assert second.label == "Quantum phonon (no eligible site)"
    assert second.result_id == str(jobless.id)
    assert controller.phonon_result(first.result_id) is orphan


def test_quantum_controller_leaves_an_earlier_run_of_a_listed_job_out_of_the_table(
    tmp_path: Path,
) -> None:
    """A row's own job running again shows on that row; its old result gets no second row."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    job = _bind_phonon(service, document, definition, key, status="running")
    _store_phonon_result(document, job, {"muon_zpe_total_eV": 0.5})
    controller = _quantum_document_controller(service, document, output_root)
    controller.refresh_view()

    assert all(row.key for row in controller._view.sources[-1])


def test_quantum_controller_gives_a_row_only_to_a_completed_job_with_no_site(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, _definition, _candidates, output_root = _seed_quantum_document(service, tmp_path)
    job = _quantum_job(service, document, "quantum_phonon_old", QUANTUM_PHONON_CALC_TYPE)
    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    _store_phonon_result(document, job, {"muon_zpe_total_eV": 0.5})
    controller = _quantum_document_controller(service, document, output_root)
    controller.refresh_view()
    assert all(row.key for row in controller._view.sources[-1])

    job.status = JobStatus.COMPLETED
    controller.refresh_view()
    assert [row.zpe for row in controller._view.sources[-1] if not row.key] == ["0.500"]


def test_quantum_controller_ignores_a_phonon_result_with_no_payload(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    job = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    result = _store_phonon_result(document, job, {})
    result.data = None
    controller = _quantum_document_controller(service, document, output_root)
    controller.refresh_view()

    row = next(row for row in controller._view.sources[-1] if row.key == key)
    assert row.result_id == ""


def test_quantum_controller_shows_no_result_while_the_job_is_not_completed(
    tmp_path: Path,
) -> None:
    """A job running again no longer stands behind an earlier run's numbers."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    job = _bind_phonon(service, document, definition, key, status="running")
    _store_phonon_result(document, job, {"muon_zpe_total_eV": 0.5})
    controller = _quantum_document_controller(service, document, output_root)
    controller.refresh_view()

    row = next(row for row in controller._view.sources[-1] if row.key == key)
    assert (row.zpe, row.corrected_energy, row.result_id) == ("–", "–", "")


def test_quantum_controller_reads_the_newest_result_with_a_naive_stamp_as_utc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun result stamped without a time zone, as an older project may hold, is read
    as UTC whatever the machine's time zone, so the newest result of the job shows."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    job = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    later = _store_phonon_result(document, job, {}, datetime(2026, 1, 1, 12, 0))
    _store_phonon_result(document, job, {}, datetime(2026, 1, 1, 11, 30, tzinfo=timezone.utc))
    controller = _quantum_document_controller(service, document, output_root)
    # Read as local time at UTC+05:30, the naive 12:00 would be 06:30 UTC.
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    try:
        controller.refresh_view()
    finally:
        monkeypatch.undo()
        time.tzset()

    row = next(row for row in controller._view.sources[-1] if row.key == key)
    assert row.result_id == str(later.id)


_STORED_MODES = [
    {"mode_index": 4, "frequency_cm": 1600.0, "participation": 0.9, "zpe_eV": 0.099},
    {"mode_index": 5, "frequency_cm": 2500.0, "participation": 0.8, "zpe_eV": 0.155},
    {"mode_index": 6, "frequency_cm": 3600.0, "participation": 0.7, "zpe_eV": 0.223},
]


def test_quantum_controller_refresh_view_shows_a_current_witness_as_stored(tmp_path: Path) -> None:
    data = {
        "muon_zpe_modes": _STORED_MODES,
        "w1_entanglement_witness": -0.25,
        "w1_definition": "participation_minus_3",
    }

    assert _displayed_witness(tmp_path, data) == "-0.250"


def test_quantum_controller_refresh_view_recomputes_a_witness_stored_before_its_definition(
    tmp_path: Path,
) -> None:
    """A payload without ``w1_definition`` holds the earlier ``-sqrt(3 - P)``
    form; the row shows ``P - 3`` recomputed from the stored participations."""

    old_form = -math.sqrt(3.0 - 2.4)
    data = {"muon_zpe_modes": _STORED_MODES, "w1_entanglement_witness": old_form}

    assert _displayed_witness(tmp_path, data) == "-0.600"


def test_quantum_controller_refresh_view_shows_an_old_witness_without_modes_as_stored(
    tmp_path: Path,
) -> None:
    assert _displayed_witness(tmp_path, {"w1_entanglement_witness": -0.5}) == "-0.500"
    assert _displayed_witness(tmp_path, {"muon_zpe_total_eV": 0.1}) == "–"


def test_quantum_controller_refresh_view_shows_muon_modes_and_stored_widths(
    tmp_path: Path,
) -> None:
    """The row lists the three muon modes and their stored ``width_A`` in
    frequency order, and the ZPE as a bare number (the header carries eV)."""

    data = {
        "muon_zpe_total_eV": 0.5786,
        "muon_zpe_modes": [
            {"mode_index": 9, "frequency_cm": 3226.0, "participation": 0.9, "width_A": 0.2149},
            {"mode_index": 7, "frequency_cm": 2742.0, "participation": 0.95, "width_A": 0.2331},
            {"mode_index": 8, "frequency_cm": 3226.0, "participation": 0.9, "width_A": 0.2149},
        ],
        "w1_entanglement_witness": -0.14,
        "w1_definition": "participation_minus_3",
        "phonon_mode_eigenvalues_cm": [100.0, 400.0, 900.0, 2742.0, 3226.0, 3226.0],
    }

    row = _displayed_result_row(tmp_path, data)

    assert row.zpe == "0.579"
    assert row.muon_modes == "2742.0, 3226.0, 3226.0"
    assert row.widths == "0.23 0.21 0.21"
    assert row.witness == "-0.140"


def test_quantum_controller_refresh_view_computes_widths_for_a_payload_without_width(
    tmp_path: Path,
) -> None:
    """A payload stored without ``width_A`` shows widths computed from each
    mode's frequency with the muon mass."""

    row = _displayed_result_row(tmp_path, {"muon_zpe_modes": _STORED_MODES})

    assert row.muon_modes == "1600.0, 2500.0, 3600.0"
    assert row.widths == " ".join(
        f"{mode_width_angstrom(mode['frequency_cm']):.2f}" for mode in _STORED_MODES
    )


def test_quantum_controller_refresh_view_leaves_mode_cells_blank_without_modes(
    tmp_path: Path,
) -> None:
    row = _displayed_result_row(tmp_path, {"muon_zpe_total_eV": "not a number"})

    assert (row.zpe, row.muon_modes, row.widths) == ("–", "", "–")


def test_quantum_controller_pre_submit_check_blocks_submission(tmp_path: Path) -> None:
    """An unresolved host mismatch silently refuses the submission: the host guard
    returns False, so no submission plan is built and nothing is dispatched."""

    service = ProjectService()
    document, _definition, candidates, _output_root = _seed_quantum_document(service, tmp_path)
    other_host = Structure(
        label="other host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    document.structures[other_host.id] = other_host
    document.project.host_structure_id = other_host.id
    mismatches: list[object] = []
    deps = _stub_quantum_deps(
        handle_structure_host_mismatch=lambda mismatch: mismatches.append(mismatch) or False,
    )
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = QuantumCorrectionsController(context, deps, _FakeQuantumView())
    materialize_calls: list[list] = []
    # Prove materialize is never reached by making it record if called.
    controller.materialize = lambda keys: materialize_calls.append(list(keys)) or []  # type: ignore[assignment]

    controller.submit(list(candidates), target="local")

    assert len(mismatches) == 1  # the relaxed structures descend from the other host
    assert local.calls == []  # nothing dispatched
    assert infos == []  # the refusal is silent (no "no ready jobs" dialog)
    assert materialize_calls == []  # no submission plan was even built


def test_quantum_controller_pre_submit_check_passes_when_host_matches(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, _output_root = _seed_quantum_document(service, tmp_path)
    mismatches: list[object] = []
    deps = _stub_quantum_deps(
        handle_structure_host_mismatch=lambda mismatch: mismatches.append(mismatch) or False,
        handle_site_host_mismatch=lambda mismatch: mismatches.append(mismatch) or False,
    )
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), deps, _FakeQuantumView()
    )
    assert controller.pre_submit_check(list(candidates)) is True
    assert mismatches == []


def test_quantum_controller_build_result_singlepoint_is_empty_but_handled(tmp_path: Path) -> None:
    """A legacy quantum-singlepoint job contributes an empty ControllerResult (base
    payload only), yet is still owned by the controller so it collects cleanly."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_singlepoint_s", QUANTUM_SINGLEPOINT_CALC_TYPE)
    run = CalculationRun(job_id=job.id)
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service),
        _stub_quantum_deps(),
        _FakeQuantumView(),
    )

    assert controller.handles_job(job) is True
    result = controller.build_result(job, completed_castep_result(), run, no_artifacts())
    assert result.payload == {}
    assert result.output_files == []
    assert result.derived_structure_ids == []
    assert result.warnings == []


def test_quantum_controller_build_result_phonon_parses_zpe_and_witness(tmp_path: Path) -> None:
    """A quantum-phonon job's build_result parses the real ``.phonon`` file and
    contributes the ZPE / eigenvalue / mode / witness payload — matching the
    fields the former inline result block assigned."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    phonon_path = output_dir / "quantum_s.phonon"
    phonon_path.write_text(_QUANTUM_PHONON_TEXT, encoding="utf-8")

    deps = _stub_quantum_deps()
    controller = QuantumCorrectionsController(
        make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: output_dir,
            job_seed_name=lambda _job: "quantum_s",
        ),
        deps,
        _FakeQuantumView(),
    )

    artifacts = collect_job_artifacts(output_dir, "quantum_s", want_phonon=True)
    result = controller.build_result(job, completed_castep_result(), run, artifacts)

    payload = result.payload
    assert payload["phonon_file"] == str(phonon_path)
    assert payload["gamma_only"] is True
    assert payload["phonon_mode_eigenvalues_cm"] == [100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0]
    assert isinstance(payload["muon_zpe_total_eV"], float)
    assert payload["muon_zpe_total_eV"] > 0.0
    assert len(payload["muon_zpe_modes"]) == 3
    assert all(
        "mode_index" in mode and "frequency_cm" in mode for mode in payload["muon_zpe_modes"]
    )
    # Each selected mode carries its muon zero-point width, from its frequency.
    for mode in payload["muon_zpe_modes"]:
        assert mode["width_A"] == pytest.approx(mode_width_angstrom(mode["frequency_cm"]))
        assert mode["width_A"] > 0.0
    # The witness is the paper's ``P - 3`` and the payload names that definition.
    assert payload["w1_definition"] == "participation_minus_3"
    assert payload["w1_entanglement_witness"] == pytest.approx(
        sum(mode["participation"] for mode in payload["muon_zpe_modes"]) - 3.0
    )
    # No output_files or derived structures contributed by build_result.
    assert result.output_files == []
    assert result.derived_structure_ids == []
    # The muon mass CASTEP reported is the muon mass, so no warning is raised.
    assert result.warnings == []


def test_quantum_controller_build_result_warns_on_a_hydrogen_mass_muon(tmp_path: Path) -> None:
    """A phonon run CASTEP performed with the default hydrogen mass for H:mu is
    reported as a run warning (dashboard-visible), not only in the payload."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / "quantum_s.phonon").write_text(
        _QUANTUM_PHONON_TEXT.replace("H:mu      0.113429", "H:mu      1.007940"),
        encoding="utf-8",
    )

    controller = QuantumCorrectionsController(
        make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: output_dir,
            job_seed_name=lambda _job: "quantum_s",
        ),
        _stub_quantum_deps(),
        _FakeQuantumView(),
    )

    artifacts = collect_job_artifacts(output_dir, "quantum_s", want_phonon=True)
    result = controller.build_result(job, completed_castep_result(), run, artifacts)

    assert len(result.warnings) == 1
    assert "not the muon mass" in result.warnings[0]
    assert any("not the muon mass" in warning for warning in result.payload["phonon_warnings"])


def test_quantum_controller_build_result_phonon_missing_dir_is_empty(tmp_path: Path) -> None:
    """If the job's output directory can't be resolved, build_result yields an
    empty result rather than crashing (validate_artifacts fails such jobs first)."""

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id)
    deps = _stub_quantum_deps()
    controller = QuantumCorrectionsController(
        make_context(
            document=document,
            project_service=service,
            job_output_directory=lambda _job: None,
        ),
        deps,
        _FakeQuantumView(),
    )
    assert (
        controller.build_result(job, completed_castep_result(), run, ParsedJobArtifacts()).payload
        == {}
    )


def _quantum_document_controller(
    service: ProjectService, document, output_root, *, deps=None, **context_kwargs
):
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _definition: output_root,
        **context_kwargs,
    )
    return QuantumCorrectionsController(context, deps or _stub_quantum_deps(), _FakeQuantumView())


def _bind_legacy_singlepoint(
    service: ProjectService,
    document: ProjectDocument,
    definition,
    key: str,
    *,
    status: JobStatus = JobStatus.COMPLETED,
) -> Job:
    """Bind the singlepoint an older MuFinder version ran first (``singlepoint_job_ids_by_label``)."""

    singlepoint = _quantum_job(
        service, document, f"quantum_singlepoint_{key}", QUANTUM_SINGLEPOINT_CALC_TYPE
    )
    set_job_state(singlepoint, status)
    definition.parameters.setdefault("singlepoint_job_ids_by_label", {})[key] = str(singlepoint.id)
    return singlepoint


def _write_legacy_seed_directory(output_root: Path, seed_name: str) -> Path:
    """Leave the files a legacy singlepoint wrote in the row's seed directory."""

    job_dir = output_root / seed_name
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{seed_name}.param").write_text("TASK : SINGLEPOINT\n", encoding="utf-8")
    (job_dir / f"{seed_name}.check").write_bytes(b"legacy checkpoint")
    return job_dir


def test_quantum_controller_materialize_writes_one_phonon_job_per_row(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    store_workflow_row_configuration(definition, second, {"num_farms": 3})
    controller = _quantum_document_controller(service, document, output_root)

    jobs = controller.materialize([first, second])

    assert [job.calc_type for job in jobs] == [QUANTUM_PHONON_CALC_TYPE, QUANTUM_PHONON_CALC_TYPE]
    assert [job.label for job in jobs] == ["quantum_phonon_siteA", "quantum_phonon_siteB"]
    assert all(job.status == JobStatus.DEFINED for job in jobs)
    phonon_mapping = definition.parameters["phonon_job_ids_by_label"]
    assert phonon_mapping == {first: str(jobs[0].id), second: str(jobs[1].id)}
    # The phonon mapping stays aliased into job_ids_by_label (load-bearing fallback).
    assert definition.parameters["job_ids_by_label"] is phonon_mapping
    assert "singlepoint_job_ids_by_label" not in definition.parameters
    assert definition.parameters["job_seed_names_by_label"] == {
        first: "quantum_siteA",
        second: "quantum_siteB",
    }

    param_text = (output_root / "quantum_siteA" / "quantum_siteA.param").read_text(encoding="utf-8")
    cell_text = (output_root / "quantum_siteA" / "quantum_siteA.cell").read_text(encoding="utf-8")
    assert "TASK : PHONON" in param_text
    assert "PHONON_METHOD : FINITEDISPLACEMENT" in param_text
    assert "CONTINUATION : DEFAULT" in param_text
    assert "CHARGE : 1" in param_text
    assert "NUM_FARMS" not in param_text
    assert "SINGLEPOINT" not in param_text
    assert "%BLOCK SPECIES_MASS" in cell_text
    assert "%BLOCK SPECIES_GAMMA" in cell_text
    assert "PHONON_KPOINT_MP_GRID 1 1 1" in cell_text
    farmed = (output_root / "quantum_siteB" / "quantum_siteB.param").read_text(encoding="utf-8")
    assert "NUM_FARMS : 3" in farmed


def test_quantum_controller_materialize_runs_one_site_in_both_charge_states(tmp_path: Path) -> None:
    from mufinder.services.source_candidates import eligible_quantum_source_candidates

    from tests.support.muon_sites import (
        MUONIUM_RELAXATION_CALC_TYPE,
        add_relaxed_muon_site,
        bind_muon_relaxation_row,
        relaxation_job_id,
    )

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    site_a = next(
        candidate.site for candidate in candidates.values() if candidate.site.label == "siteA"
    )
    scratch = add_relaxed_muon_site(
        document, label="siteA", charge_calc_type=MUONIUM_RELAXATION_CALC_TYPE, total_energy=-2.0
    )
    muonium_job = document.jobs[relaxation_job_id(document, scratch)]
    del document.sites[scratch.id]
    bind_muon_relaxation_row(document, site_a, muonium_job, "muonium")
    keys = [str(site_a.id), f"{site_a.id}:muonium"]
    assert set(keys) <= {
        candidate.key for candidate in eligible_quantum_source_candidates(document)
    }
    controller = _quantum_document_controller(service, document, output_root)

    jobs = controller.materialize(keys)

    assert [job.label for job in jobs] == ["quantum_phonon_siteA", "quantum_phonon_siteA_muonium"]
    assert definition.parameters["phonon_job_ids_by_label"] == {
        keys[0]: str(jobs[0].id),
        keys[1]: str(jobs[1].id),
    }
    assert definition.parameters["job_seed_names_by_label"] == {
        keys[0]: "quantum_siteA",
        keys[1]: "quantum_siteA_muonium",
    }
    muonium_param = (
        output_root / "quantum_siteA_muonium" / "quantum_siteA_muonium.param"
    ).read_text(encoding="utf-8")
    assert "CHARGE : 0" in muonium_param
    stored = definition.parameters["configuration"]["sources"]
    assert [
        (entry["key"], entry["charge_state"])
        for entry in stored
        if entry["site_id"] == str(site_a.id)
    ] == [
        (keys[0], "positive_muon"),
        (keys[1], "muonium"),
    ]


def test_quantum_controller_keeps_an_unselected_charge_state_out_of_the_sources(
    tmp_path: Path,
) -> None:
    from tests.support.muon_sites import (
        MUONIUM_RELAXATION_CALC_TYPE,
        add_relaxed_muon_site,
        bind_muon_relaxation_row,
        relaxation_job_id,
    )

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    site_a = next(
        candidate.site for candidate in candidates.values() if candidate.site.label == "siteA"
    )
    scratch = add_relaxed_muon_site(
        document, label="siteA", charge_calc_type=MUONIUM_RELAXATION_CALC_TYPE, total_energy=-2.0
    )
    muonium_job = document.jobs[relaxation_job_id(document, scratch)]
    del document.sites[scratch.id]
    bind_muon_relaxation_row(document, site_a, muonium_job, "muonium")
    # The definition so far holds only siteB; select siteA's muonium relaxation alone.
    site_b = next(
        candidate.site for candidate in candidates.values() if candidate.site.label == "siteB"
    )
    definition.input_site_ids = [site_b.id]
    definition.parameters["configuration"] = {"sources": [{"site_id": str(site_b.id)}]}
    controller = _quantum_document_controller(service, document, output_root)
    muonium_key = f"{site_a.id}:muonium"

    controller.ensure_definition([muonium_key])
    controller.ensure_definition([muonium_key])

    stored = definition.parameters["configuration"]["sources"]
    assert [entry["key"] for entry in stored] == [str(site_b.id), muonium_key]


def test_quantum_controller_materialize_reuses_a_rows_job(tmp_path: Path) -> None:
    """A defined job is rewritten from the row's configuration; an ended one waits for its reset.

    Both rows are ready, but a controller writes a job directory only for a
    defined job: the submit cycle resets an ended job, then materializes again.
    """

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    defined = _bind_phonon(service, document, definition, first)
    failed = _bind_phonon(service, document, definition, second, status="failed")
    store_workflow_row_configuration(definition, first, {"num_farms": 2})
    controller = _quantum_document_controller(service, document, output_root)

    jobs = controller.materialize([first, second])

    # Both rows keep their own job; only the defined one is written now.
    assert jobs == [defined, failed]
    assert len(document.jobs) == 2 + 2  # two source relaxations, two phonon jobs
    assert "NUM_FARMS : 2" in (output_root / "quantum_siteA" / "quantum_siteA.param").read_text(
        encoding="utf-8"
    )
    assert not (output_root / "quantum_siteB").exists()
    assert job_status_word(failed) == "failed"


def test_quantum_controller_row_lifecycle_from_ready_to_a_completed_result(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    local = RecordingBackend()
    view = _FakeQuantumView()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        definition_output_directory=lambda _definition: output_root,
    )
    controller = QuantumCorrectionsController(context, _stub_quantum_deps(), view)

    assert controller.build_states()[key].status == "ready"
    assert controller.jobs_for_keys([key]) == []

    controller.submit([key], target="local")

    ((submitted, extras),) = local.calls
    assert list(extras) == ["on_accepted"]
    (job,) = submitted
    assert job.calc_type == QUANTUM_PHONON_CALC_TYPE
    assert controller.build_states()[key].status == "ready"
    assert controller.jobs_for_keys([key]) == [job]

    job.status = JobStatus.ACTIVE
    job.activity = JobActivity.RUNNING
    assert controller.build_states()[key].status == "running"
    # Submitting again leaves a running row alone: nothing new is created.
    controller.submit([key], target="local")
    assert controller.jobs_for_keys([key]) == [job]
    assert (
        sum(1 for other in document.jobs.values() if other.calc_type == QUANTUM_PHONON_CALC_TYPE)
        == 1
    )

    job.status = JobStatus.COMPLETED
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(
        result_type=QUANTUM_PHONON_CALC_TYPE,
        calculation_run_id=run.id,
        data={"muon_zpe_total_eV": 0.2},
    )
    document.results[result.id] = result
    controller.refresh_view()

    source_row = next(row for row in view.sources[-1] if row.key == key)
    assert source_row.status == "completed"
    assert source_row.result_id == str(result.id)
    assert source_row.zpe == "0.200"


def test_quantum_controller_refresh_view_always_shows_the_farm_count(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    store_workflow_row_configuration(definition, second, {"num_farms": 4})
    view = _FakeQuantumView()
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service), _stub_quantum_deps(), view
    )
    controller.set_show_every_site(True)

    rows = {row.key: row for row in view.sources[-1]}
    assert (rows[first].num_farms_display, rows[first].status) == ("1", "ready")
    assert rows[second].num_farms_display == "4"


def test_quantum_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    local = RecordingBackend()
    cluster = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, cluster_backend=cluster
    )

    controller.submit([key], target="cluster")

    assert local.calls == []
    ((submitted, extras),) = cluster.calls
    extras.pop("on_accepted")
    # No preselected cluster: a phonon job can run on any target.
    assert extras == {}
    assert [job.calc_type for job in submitted] == [QUANTUM_PHONON_CALC_TYPE]


def test_quantum_controller_dismissed_failed_job_resubmits_without_a_new_job(
    tmp_path: Path,
) -> None:
    """A dismissed failed row reads failed; the shared submit cycle resets and rewrites its job."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    phonon = _bind_phonon(service, document, definition, key, status="failed")
    phonon.dismissed_at = datetime.now(timezone.utc)
    reset: list[list[Job]] = []

    start_again = start_again_by_reset(service)

    def _reset(jobs):
        reset.append(list(jobs))
        return start_again(jobs)

    local = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, prepare_jobs_for_resubmission=_reset
    )

    assert controller.build_states()[key].status == "failed"
    controller.submit([key], target="local")

    assert reset == [[phonon]]
    assert local.calls[0][0] == [phonon]
    assert phonon.status == JobStatus.DEFINED
    assert (output_root / "quantum_siteA" / "quantum_siteA.param").is_file()


def test_quantum_controller_configure_request_opens_editable_ready_rows(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    controller = _quantum_document_controller(service, document, output_root)

    request = controller.configure_request([*candidates, "not-a-source"])

    assert request is not None
    assert request.keys == tuple(candidates)
    assert request.title == "Configure Quantum Corrections Parameters"
    assert request.read_only is False
    assert request.locked_reason == ""
    assert request.num_farms == 1
    assert request.species_options == ("Si", "H:mu")
    assert "Selected sources: 2" in request.intro_text
    assert "Displacements required for the phonon run: 6" in request.intro_text
    assert "stage" not in request.intro_text.lower()
    # No row carries a phonon method: every row runs finite displacement.
    assert all("phonon_method" not in values for values in request.row_values)
    assert controller.configure_request(["not-a-source"]) is None
    assert (
        QuantumCorrectionsController(
            make_context(document=None), _stub_quantum_deps(), _FakeQuantumView()
        ).configure_request(list(candidates))
        is None
    )


def test_quantum_controller_configure_request_edits_a_defined_job(tmp_path: Path) -> None:
    """A job still defined is rewritten on submit, so its row stays editable."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    _bind_phonon(service, document, definition, key)
    controller = _quantum_document_controller(service, document, output_root)

    request = controller.configure_request([key])

    assert request is not None
    assert request.read_only is False
    assert request.title == "Configure Quantum Corrections Parameters"
    assert request.locked_reason == ""


@pytest.mark.parametrize("status", ["failed", "stopped", "cancelled"])
@pytest.mark.parametrize("dismissed", [False, True])
def test_quantum_controller_farm_count_edit_on_an_ended_row_takes_effect_on_the_rerun(
    tmp_path: Path, status: str, dismissed: bool
) -> None:
    """An ended row is editable, and the next run rebuilds its inputs from the edit.

    Every run of an ended row, from the row or from the dashboard's Resubmit,
    goes through the shared submit cycle, which rewrites the job's ``.param``
    from the row. So a changed farm count reaches ``NUM_FARMS`` and the farm
    count the runners read the output under, and the two never disagree.
    """

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    phonon = _bind_phonon(service, document, definition, key, status=status)
    if dismissed:
        phonon.dismissed_at = datetime.now(timezone.utc)

    _reset = start_again_by_reset(service)

    local = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, prepare_jobs_for_resubmission=_reset
    )

    request = controller.configure_request([key])

    assert request is not None
    assert request.read_only is False
    assert request.locked_reason == ""
    assert request.title == "Configure Quantum Corrections Parameters"

    controller.apply_configuration(request.keys, configure_result(), {"num_farms": 3})
    controller.submit([key], target="local")

    assert local.calls[0][0] == [phonon]
    assert workflow_row_configuration(definition, key)[2] == {"num_farms": 3}
    param = (output_root / "quantum_siteA" / "quantum_siteA.param").read_text(encoding="utf-8")
    assert "NUM_FARMS : 3" in param


def _resolve_quantum_directory(document: ProjectDocument):
    """Resolve a job's folder the way the main window does, through the row's seed."""

    def resolve(job: Job) -> Path | None:
        return job_output_directory(
            document,
            job,
            quantum_seed_name=lambda definition, row: quantum_seed_name_for_key(
                document, definition, row
            ),
        )

    return resolve


def _complete_phonon_job(document: ProjectDocument, job: Job, *, zpe: float = 0.25) -> Result:
    """Complete ``job`` with a phonon result whose ``.phonon`` sits in its folder."""

    job.status = JobStatus.COMPLETED
    job.completed_at = datetime.now(timezone.utc)
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    output_root = document.project_root / "files" / "quantum_corrections"
    seed = "quantum_siteA" if job.label.endswith("siteA") else "quantum_siteB"
    phonon_path = output_root / seed / f"{seed}.phonon"
    phonon_path.write_text(_QUANTUM_PHONON_TEXT, encoding="utf-8")
    (output_root / seed / f"{seed}.castep").write_text("castep output\n", encoding="utf-8")
    result = Result(
        calculation_run_id=run.id,
        result_type=QUANTUM_PHONON_CALC_TYPE,
        data={"phonon_file": str(phonon_path), "muon_zpe_total_eV": zpe},
    )
    run.result_id = result.id
    document.runs[run.id] = run
    document.results[result.id] = result
    return result


def _completed_rows(tmp_path: Path, *, completed: int = 1, **context_kwargs):
    """Seed two rows and run the first ``completed`` of them to a phonon result.

    The inputs are written by ``materialize`` as a real run writes them, so the
    row's parameters can be compared with what the job ran. Returns ``(service,
    document, definition, keys, controller, jobs, results, output_root)``.
    """

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    keys = list(candidates)
    controller = _quantum_document_controller(
        service,
        document,
        output_root,
        job_output_directory=_resolve_quantum_directory(document),
        job_seed_name=lambda job: _resolve_quantum_directory(document)(job).name,
        **context_kwargs,
    )
    jobs = controller.materialize(keys[:completed])
    results = [
        _complete_phonon_job(document, job, zpe=0.25 + index) for index, job in enumerate(jobs)
    ]
    return service, document, definition, keys, controller, jobs, results, output_root


def test_quantum_controller_configure_opens_editable_on_a_completed_row(tmp_path: Path) -> None:
    """A completed row can be configured again, from its current parameters."""

    _service, _document, _definition, keys, controller, _jobs, _results, _root = _completed_rows(
        tmp_path
    )

    request = controller.configure_request([keys[0]])

    assert request is not None
    assert request.read_only is False
    assert request.locked_reason == ""
    assert request.title == "Configure Quantum Corrections Parameters"
    assert controller.build_states()[keys[0]].status == "completed"


class _AcceptingConfigDialog:
    """A Configure dialog accepted with its seed values, ``changes`` applied on top."""

    def __init__(self, changes: dict[str, object], advanced: tuple[str, ...] | None, **kwargs):
        self._values = {**kwargs["values"], **changes}
        self._advanced = kwargs["advanced_names"] if advanced is None else advanced
        self._num_farms = kwargs["num_farms"]

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

    def num_farms(self) -> int:
        return self._num_farms

    def num_farms_edited(self) -> bool:
        return False


_LOOSER_TOLERANCE = {"elec_energy_tol": ValueWithUnit(1e-8, "eV")}


@pytest.fixture
def configure_with(monkeypatch: pytest.MonkeyPatch):
    """Make ``configure_sources`` accept its dialog with the given changes."""

    import mufinder.app.workflow_controllers.quantum_corrections as module

    def accept(changes: dict[str, object], *, advanced: tuple[str, ...] | None = None) -> None:
        monkeypatch.setattr(
            module,
            "QuantumCorrectionsConfigDialog",
            lambda **kwargs: _AcceptingConfigDialog(changes, advanced, **kwargs),
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


def _job_files(job_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(job_dir.iterdir())}


def test_accepting_a_completed_row_unchanged_asks_nothing_and_changes_nothing(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    _service, _document, definition, keys, controller, _jobs, results, root = _completed_rows(
        tmp_path
    )
    asked, _answer = replace_questions
    files = _job_files(root / "quantum_siteA")
    configure_with({})

    controller.configure_sources([keys[0]])

    assert asked == []
    assert workflow_row_configuration(definition, keys[0]) == ({}, (), {})
    assert controller.build_states()[keys[0]].status == "completed"
    assert results[0].id in _document.results
    assert _job_files(root / "quantum_siteA") == files


def test_changing_a_completed_row_asks_first_and_cancel_keeps_everything(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    _service, document, definition, keys, controller, _jobs, results, root = _completed_rows(
        tmp_path
    )
    asked, answer = replace_questions
    answer[0] = False
    files = _job_files(root / "quantum_siteA")
    configure_with(_LOOSER_TOLERANCE)

    controller.configure_sources([keys[0]])

    assert asked == [
        {
            "title": "Change parameters",
            "text": "Change the parameters of 'siteA, positive muon'? Running the row again "
            "replaces its phonon result.",
            "detail": "Running again deletes the phonon result, its ZPE and what the page "
            "derives from it (dE + ZPE and the ranking), and clears the job's folder. Until "
            "that run's submission is accepted, the row keeps its current result, and setting "
            "the parameters back keeps it for good.",
            "accept": "Change parameters",
            "reject": "Cancel",
        }
    ]
    assert workflow_row_configuration(definition, keys[0]) == ({}, (), {})
    assert controller.build_states()[keys[0]].status == "completed"
    assert results[0].id in document.results
    assert _job_files(root / "quantum_siteA") == files


def test_a_confirmed_change_keeps_the_result_until_the_row_runs_again(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    _service, document, definition, keys, controller, _jobs, results, root = _completed_rows(
        tmp_path
    )
    asked, _answer = replace_questions
    files = _job_files(root / "quantum_siteA")
    configure_with(_LOOSER_TOLERANCE)

    controller.configure_sources([keys[0]])

    assert len(asked) == 1
    assert controller.build_states()[keys[0]].status == "edited"
    row = next(row for row in controller._view.sources[-1] if row.key == keys[0])
    assert (row.status, row.result_id, row.zpe) == ("edited", str(results[0].id), "0.250")
    assert row.notes[-1] == (
        "Parameters changed since this result; running the row again replaces it."
    )
    # The Configure dialog shows the new parameters, editable.
    request = controller.configure_request([keys[0]])
    assert request.read_only is False
    assert request.row_values[0]["elec_energy_tol"] == ValueWithUnit(1e-8, "eV")
    assert results[0].id in document.results
    assert _job_files(root / "quantum_siteA") == files

    # Changing an edited row again asks nothing more.
    configure_with({"elec_energy_tol": ValueWithUnit(1e-7, "eV")})
    controller.configure_sources([keys[0]])
    assert len(asked) == 1
    assert controller.build_states()[keys[0]].status == "edited"


def test_setting_the_parameters_back_makes_the_row_completed_again(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    _service, _document, _definition, keys, controller, _jobs, _results, _root = _completed_rows(
        tmp_path
    )
    asked, _answer = replace_questions
    configure_with(_LOOSER_TOLERANCE)
    controller.configure_sources([keys[0]])
    assert controller.build_states()[keys[0]].status == "edited"

    configure_with({"elec_energy_tol": ValueWithUnit(1e-9, "eV")})
    controller.configure_sources([keys[0]])

    assert len(asked) == 1
    assert controller.build_states()[keys[0]].status == "completed"


def test_a_selection_of_completed_and_ready_rows_asks_once_naming_the_count(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    _service, _document, definition, keys, controller, _jobs, _results, _root = _completed_rows(
        tmp_path, completed=2
    )
    asked, _answer = replace_questions
    configure_with(_LOOSER_TOLERANCE)

    controller.configure_sources(keys)

    assert [question["text"] for question in asked] == [
        "Change the parameters of 2 completed sites? Running the rows again replaces their "
        "phonon results."
    ]
    assert {controller.build_states()[key].status for key in keys} == {"edited"}

    # One completed row beside one never run: the question names the completed one only.
    _s, _d, _def, keys, controller, _j, _r, _root = _completed_rows(tmp_path / "mixed")
    asked.clear()
    controller.configure_sources(keys)
    assert [question["text"] for question in asked] == [
        "Change the parameters of 'siteA, positive muon'? Running the row again replaces "
        "its phonon result."
    ]
    states = controller.build_states()
    assert (states[keys[0]].status, states[keys[1]].status) == ("edited", "ready")


def test_a_farm_count_alone_does_not_replace_a_result(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    """The farm count changes how the job runs, not its result, so it asks nothing."""

    _service, _document, definition, keys, controller, _jobs, _results, _root = _completed_rows(
        tmp_path
    )
    asked, _answer = replace_questions
    configure_with({})

    controller.apply_configuration([keys[0]], configure_result(), {"num_farms": 3})

    assert (
        controller.rows_whose_result_a_change_replaces(
            [keys[0]], configure_result(), {"num_farms": 4}
        )
        == []
    )
    assert asked == []
    assert workflow_row_configuration(definition, keys[0])[2] == {"num_farms": 3}
    assert controller.build_states()[keys[0]].status == "completed"


def test_adding_or_removing_an_advanced_parameter_edits_a_completed_row(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    _service, _document, definition, keys, controller, jobs, _results, root = _completed_rows(
        tmp_path
    )
    asked, _answer = replace_questions
    configure_with({"max_scf_cycles": 200}, advanced=("max_scf_cycles",))
    controller.configure_sources([keys[0]])
    assert len(asked) == 1
    assert controller.build_states()[keys[0]].status == "edited"

    # A run that had the parameter, which the row then drops, is edited too.
    (job,) = jobs
    param = root / "quantum_siteA" / "quantum_siteA.param"
    param.write_text(param.read_text(encoding="utf-8") + "MAX_SCF_CYCLES : 200\n", encoding="utf-8")
    store_workflow_row_configuration(
        definition,
        keys[0],
        {"parameter_values": {"max_scf_cycles": 200}, "added_advanced": ["max_scf_cycles"]},
    )
    assert controller.build_states()[keys[0]].status == "completed"
    configure_with({}, advanced=())
    removed = controller.rows_whose_result_a_change_replaces(
        [keys[0]], configure_result({"elec_energy_tol": ValueWithUnit(1e-9, "eV")}), {}
    )
    assert removed == [keys[0]]
    assert job.status == JobStatus.COMPLETED


def test_a_changed_default_or_missing_source_files_never_edit_a_row(tmp_path: Path) -> None:
    """Only what the row stores counts: what it inherits or defaults to is left out."""

    _service, _document, definition, keys, controller, _jobs, _results, root = _completed_rows(
        tmp_path
    )
    param = root / "quantum_siteA" / "quantum_siteA.param"
    # The run was written with an older default and a setting inherited from a
    # source relaxation whose files have gone since.
    text = param.read_text(encoding="utf-8")
    param.write_text(text.replace("1e-09", "1e-08") + "XC_FUNCTIONAL : PBE\n", encoding="utf-8")

    assert controller.build_states()[keys[0]].status == "completed"
    # The row stores only a farm count, which is not a parameter.
    store_workflow_row_configuration(definition, keys[0], {"num_farms": 2})
    assert controller.build_states()[keys[0]].status == "completed"
    # A row an older version configured for DFPT, whose job ran it, is not edited either:
    # the workflow fixes the method.
    param.write_text(
        param.read_text(encoding="utf-8").replace("FINITEDISPLACEMENT", "DFPT"), encoding="utf-8"
    )
    store_workflow_row_configuration(
        definition, keys[0], {"parameter_values": {"phonon_method": "DFPT"}}
    )
    assert controller.build_states()[keys[0]].status == "completed"


def test_a_keyword_an_older_version_wrote_by_itself_does_not_edit_a_row(tmp_path: Path) -> None:
    """A keyword the run had that the Configure dialog does not offer was not the user's."""

    _service, _document, definition, keys, controller, _jobs, _results, root = _completed_rows(
        tmp_path
    )
    param = root / "quantum_siteA" / "quantum_siteA.param"
    param.write_text(param.read_text(encoding="utf-8") + "TASK_LEGACY_MARKER : 1\n")
    store_workflow_row_configuration(
        definition, keys[0], {"parameter_values": {"elec_energy_tol": "1e-09 eV"}}
    )

    assert controller.build_states()[keys[0]].status == "completed"


def test_a_completed_row_whose_settings_do_not_build_stays_editable(tmp_path: Path) -> None:
    """Settings that cannot be written are left for the user to correct."""

    _service, _document, definition, keys, controller, _jobs, _results, _root = _completed_rows(
        tmp_path
    )
    store_workflow_row_configuration(
        definition, keys[0], {"parameter_values": {"kpoint_mp_grid": "not a grid"}}
    )

    assert controller.row_inputs(definition, keys[0]) is None
    assert controller.build_states()[keys[0]].status == "completed"
    request = controller.configure_request([keys[0]])
    assert (request.read_only, request.locked_reason) == (False, "")


def test_a_completed_row_whose_inputs_are_missing_opens_read_only(tmp_path: Path) -> None:
    _service, _document, _definition, keys, controller, _jobs, _results, root = _completed_rows(
        tmp_path
    )
    (root / "quantum_siteA" / "quantum_siteA.param").unlink()

    request = controller.configure_request([keys[0]])

    assert request.read_only is True
    assert request.locked_reason == (
        "The input files of a selected row's completed phonon job are missing, so its "
        "parameters cannot be compared with its result and stay as they are."
    )
    assert controller.build_states()[keys[0]].status == "completed"


def test_an_edited_row_survives_a_save_and_reopen(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    service, document, _definition, keys, controller, _jobs, results, root = _completed_rows(
        tmp_path
    )
    configure_with(_LOOSER_TOLERANCE)
    controller.configure_sources([keys[0]])
    service.save_project(document)

    loaded = service.load_project(document.project_root)
    reopened = _quantum_document_controller(
        service,
        loaded,
        root,
        job_output_directory=_resolve_quantum_directory(loaded),
        job_seed_name=lambda job: _resolve_quantum_directory(loaded)(job).name,
    )

    assert reopened.build_states()[keys[0]].status == "edited"
    assert results[0].id in loaded.results
    reopened.refresh_view()
    row = next(row for row in reopened._view.sources[-1] if row.key == keys[0])
    assert (row.status, row.zpe) == ("edited", "0.250")


def test_running_an_edited_row_starts_its_completed_job_again(
    tmp_path: Path, configure_with, replace_questions
) -> None:
    """The edited row's completed job goes to the run-again plan; a cancel changes nothing."""

    asked_plans: list[list[Job]] = []
    local = RecordingBackend()

    def _cancel(jobs):
        asked_plans.append(list(jobs))
        return None

    _service, document, _definition, keys, controller, jobs, results, root = _completed_rows(
        tmp_path, completed=2, local_backend=local, prepare_jobs_for_resubmission=_cancel
    )
    configure_with(_LOOSER_TOLERANCE)
    controller.configure_sources([keys[0]])

    # Run All: only the edited row's job runs again; the other completed row is left out.
    controller.submit(keys, target="local")

    assert asked_plans == [[jobs[0]]]
    assert local.calls == []
    assert jobs[0].status == JobStatus.COMPLETED
    assert results[0].id in document.results


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("running", "phonon job is running; cancel it to edit them."),
        ("queued", "phonon job is queued; cancel it to edit them."),
    ],
)
def test_quantum_controller_read_only_configure_gives_one_reason(
    tmp_path: Path, status: JobStatus, expected: str
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    _bind_phonon(service, document, definition, first, status=status)
    controller = _quantum_document_controller(service, document, output_root)

    # One locked row makes the whole selection read-only.
    request = controller.configure_request([second, first])

    assert request is not None
    assert request.read_only is True
    assert request.title == "Inspect Quantum Corrections Parameters"
    assert expected in request.locked_reason
    assert "\n" not in request.locked_reason
    assert request.intro_text.endswith(request.locked_reason)


def test_quantum_controller_parameter_surface_offers_no_phonon_method() -> None:
    """``PHONON_METHOD`` is fixed, so it is neither a row nor under ``Add Parameter``."""

    names = [spec.name for spec in QuantumCorrectionsController.parameter_specs()]

    assert "phonon_method" not in names
    assert "task" not in names


def _store_old_phonon_methods(definition, by_label: dict[str, str]) -> list[str]:
    """Store the DFPT / linear-response rows an older MuFinder version configured.

    ``siteA`` keeps the dialog's ``phonon_method`` row value and ``siteB`` the
    lower-case name older row extras held; both carry the retired
    ``pseudopotential_family`` extra.
    """

    store_workflow_row_configuration(
        definition,
        by_label["siteA"],
        {"parameter_values": {"phonon_method": "DFPT"}, "pseudopotential_family": "ultrasoft"},
    )
    store_workflow_row_configuration(
        definition,
        by_label["siteB"],
        {"phonon_method": "linear_response", "pseudopotential_family": "norm_conserving"},
    )
    return [by_label["siteA"], by_label["siteB"]]


def test_quantum_controller_resubmits_an_old_dfpt_row_as_finite_displacement_with_a_notice(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    by_label = {candidate.site.label: key for key, candidate in candidates.items()}
    keys = _store_old_phonon_methods(definition, by_label)
    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, infos=infos
    )

    # The old rows load: their Configure dialog shows no phonon method.
    request = controller.configure_request(keys)
    assert request is not None
    assert all("phonon_method" not in values for values in request.row_values)

    controller.submit(keys, target="local")

    ((title, message),) = infos
    assert title == "Phonon method changed"
    assert "\n" not in message
    assert "siteA (DFPT)" in message
    assert "siteB (LINEARRESPONSE)" in message
    assert "FINITEDISPLACEMENT" in message
    assert len(local.calls) == 1
    for seed in ("quantum_siteA", "quantum_siteB"):
        param_text = (output_root / seed / f"{seed}.param").read_text(encoding="utf-8")
        assert "PHONON_METHOD : FINITEDISPLACEMENT" in param_text
        assert "DFPT" not in param_text
        assert "LINEARRESPONSE" not in param_text


@pytest.mark.parametrize(
    ("written_method", "notice"),
    [("FINITEDISPLACEMENT", False), ("DFPT", True), (None, True)],
    ids=["written-finite-displacement", "written-dfpt", "no-param-file"],
)
def test_quantum_controller_announces_an_existing_job_unless_it_is_finite_displacement(
    tmp_path: Path, written_method: str | None, notice: bool
) -> None:
    """A defined job an older version wrote as DFPT is announced before it is rewritten;
    one already written as finite displacement (a reset row) is not announced again."""

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    by_label = {candidate.site.label: key for key, candidate in candidates.items()}
    keys = _store_old_phonon_methods(definition, by_label)
    phonon = _bind_phonon(service, document, definition, keys[0], status=JobStatus.DEFINED)
    job_dir = output_root / phonon.label
    if written_method is not None:
        job_dir.mkdir(parents=True)
        (job_dir / f"{phonon.label}.param").write_text(
            f"TASK : PHONON\nPHONON_METHOD : {written_method}\n", encoding="utf-8"
        )
    infos: list[tuple[str, str]] = []
    controller = _quantum_document_controller(
        service,
        document,
        output_root,
        infos=infos,
        job_output_directory=lambda job: job_dir if job.id == phonon.id else None,
    )

    assert controller.pre_submit_check(keys) is True

    ((_title, message),) = infos
    assert "siteB (LINEARRESPONSE)" in message
    assert ("siteA (DFPT)" in message) is notice


def test_quantum_controller_reads_an_old_definition_level_method(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    definition.parameters["configuration"] = {"method": "dfpt", "pseudopotential_family": "unknown"}
    store_workflow_row_configuration(
        definition, second, {"parameter_values": {"phonon_method": "FINITEDISPLACEMENT"}}
    )
    controller = _quantum_document_controller(service, document, output_root)

    # The row's own value wins over the definition's.
    assert controller.legacy_phonon_method(definition, first) == "DFPT"
    assert controller.legacy_phonon_method(definition, second) is None
    assert controller.legacy_phonon_method(None, first) is None


def test_quantum_controller_submits_without_a_notice_when_no_old_method_is_stored(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, infos=infos
    )

    controller.submit(list(candidates), target="local")

    assert infos == []
    assert len(local.calls) == 1


def test_quantum_controller_keeps_a_completed_dfpt_rows_result(tmp_path: Path) -> None:
    """A row an older version ran with DFPT keeps its job and result, and no notice."""

    from mufinder.domain.models import Result

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    by_label = {candidate.site.label: key for key, candidate in candidates.items()}
    keys = _store_old_phonon_methods(definition, by_label)
    phonon = _bind_phonon(service, document, definition, keys[0], status=JobStatus.COMPLETED)
    run = CalculationRun(job_id=phonon.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(
        result_type=QUANTUM_PHONON_CALC_TYPE,
        calculation_run_id=run.id,
        data={"phonon_method": "dfpt", "muon_zpe_total_eV": 0.42},
    )
    document.results[result.id] = result
    infos: list[tuple[str, str]] = []
    controller = _quantum_document_controller(service, document, output_root, infos=infos)
    view = controller._view

    controller.refresh_view()

    result_row = next(row for row in view.sources[-1] if row.key == keys[0])
    assert result_row.result_id == str(result.id)
    assert result_row.status == "completed"
    assert controller.build_states()[keys[0]].status == "completed"
    assert controller.phonon_result(str(result.id)) is result
    # Only the not-yet-run row is submittable, and only it gets the notice.
    assert controller.pre_submit_check(keys) is True
    ((_title, message),) = infos
    assert "siteB (LINEARRESPONSE)" in message
    assert "siteA" not in message


# --- projects that ran the legacy singlepoint job first ----------------------


def test_quantum_controller_keeps_a_completed_legacy_pairs_phonon_result(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    _bind_legacy_singlepoint(service, document, definition, key)
    phonon = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    run = CalculationRun(job_id=phonon.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(
        result_type=QUANTUM_PHONON_CALC_TYPE,
        calculation_run_id=run.id,
        data={"muon_zpe_total_eV": 0.3},
    )
    document.results[result.id] = result
    view = _FakeQuantumView()
    context = make_context(
        document=document,
        project_service=service,
        definition_output_directory=lambda _d: output_root,
    )
    controller = QuantumCorrectionsController(context, _stub_quantum_deps(), view)

    assert controller.build_states()[key].status == "completed"
    assert controller.jobs_for_keys([key]) == [phonon]
    assert controller.materialize([key]) == []
    controller.refresh_view()
    assert [row.result_id for row in view.sources[-1] if row.key == key] == [str(result.id)]
    assert controller.phonon_result(str(result.id)) is result


@pytest.mark.parametrize("legacy_status", [JobStatus.COMPLETED, "failed"])
def test_quantum_controller_singlepoint_only_row_is_ready_with_a_fresh_directory(
    tmp_path: Path, legacy_status: JobStatus
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _bind_legacy_singlepoint(service, document, definition, key, status=legacy_status)
    definition.parameters["job_seed_names_by_label"][key] = "quantum_siteA"
    legacy_dir = _write_legacy_seed_directory(output_root, "quantum_siteA")
    controller = _quantum_document_controller(service, document, output_root)

    assert controller.build_states()[key].status == "ready"
    assert controller.jobs_for_keys([key]) == []

    (job,) = controller.materialize([key])

    assert job.calc_type == QUANTUM_PHONON_CALC_TYPE
    assert definition.parameters["job_seed_names_by_label"][key] == "quantum_siteA_phonon"
    assert definition.parameters["singlepoint_seed_names_by_label"] == {key: "quantum_siteA"}
    assert (output_root / "quantum_siteA_phonon" / "quantum_siteA_phonon.param").is_file()
    assert not (output_root / "quantum_siteA_phonon" / "quantum_siteA_phonon.check").exists()
    # The legacy singlepoint keeps its directory and its record.
    assert (legacy_dir / "quantum_siteA.check").read_bytes() == b"legacy checkpoint"
    assert document.jobs[singlepoint.id] is singlepoint
    assert (
        job_output_directory(
            document,
            singlepoint,
            quantum_seed_name=lambda definition, row: quantum_seed_name_for_key(
                document, definition, row
            ),
        )
        == output_root / "quantum_siteA"
    )
    assert (
        job_output_directory(
            document,
            job,
            quantum_seed_name=lambda definition, row: quantum_seed_name_for_key(
                document, definition, row
            ),
        )
        == output_root / "quantum_siteA_phonon"
    )


def test_quantum_controller_fresh_seed_skips_names_already_taken(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    definition.parameters["job_seed_names_by_label"].update(
        {first: "quantum_siteA", second: "quantum_siteA_phonon"}
    )
    _write_legacy_seed_directory(output_root, "quantum_siteA")
    (output_root / "quantum_siteA_phonon_2").mkdir()
    controller = _quantum_document_controller(service, document, output_root)

    controller.materialize([first])

    assert definition.parameters["job_seed_names_by_label"][first] == "quantum_siteA_phonon_3"
    # No legacy singlepoint is bound, so none is recorded.
    assert "singlepoint_seed_names_by_label" not in definition.parameters


@pytest.mark.parametrize("status", ["submitting", "queued", "running", "retrieving"])
def test_quantum_controller_refuses_a_row_whose_legacy_singlepoint_is_active(
    tmp_path: Path, status: JobStatus
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _bind_legacy_singlepoint(service, document, definition, key, status=status)
    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    controller = _quantum_document_controller(
        service, document, output_root, local_backend=local, infos=infos
    )

    # The row shows the singlepoint, so it can be cancelled from the table.
    assert controller.build_states()[key].status == status
    assert controller.jobs_for_keys([key]) == [singlepoint]
    request = controller.configure_request([key])
    assert request is not None and request.read_only is True
    assert "older MuFinder version" in request.locked_reason

    controller.submit([key], target="local")

    assert local.calls == []
    ((title, message),) = infos
    assert title == "Singlepoint still in progress"
    assert f"siteA ({status})" in message
    assert "phonon_job_ids_by_label" in definition.parameters
    assert definition.parameters["phonon_job_ids_by_label"] == {}


def test_quantum_controller_directory_companions_name_the_legacy_singlepoint(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    definition.parameters["job_seed_names_by_label"].update(
        {first: "quantum_siteA", second: "quantum_siteB"}
    )
    singlepoint = _bind_legacy_singlepoint(service, document, definition, first)
    phonon = _bind_phonon(service, document, definition, first, status="failed")
    lone_phonon = _bind_phonon(service, document, definition, second, status="failed")

    def resolve(job: Job) -> Path | None:
        return job_output_directory(
            document,
            job,
            quantum_seed_name=lambda definition, row: quantum_seed_name_for_key(
                document, definition, row
            ),
        )

    controller = _quantum_document_controller(
        service, document, output_root, job_output_directory=resolve
    )

    assert controller.definition_and_key_for_job(phonon) == (definition, first)
    assert controller.directory_companions(phonon) == [singlepoint]
    assert controller.directory_companions(singlepoint) == []
    assert controller.directory_companions(lone_phonon) == []
    singlepoint.status = JobStatus.ACTIVE
    singlepoint.activity = JobActivity.RUNNING
    assert controller.directory_companions(phonon) == []
    singlepoint.status = JobStatus.COMPLETED
    # Once the singlepoint records its own seed, it no longer shares the directory.
    definition.parameters["singlepoint_seed_names_by_label"] = {first: "quantum_siteA_legacy"}
    assert controller.directory_companions(phonon) == []


def test_quantum_controller_apply_configuration_stores_every_row(tmp_path: Path) -> None:
    import dataclasses

    from mufinder.app.workflow_config_codecs import workflow_row_configuration

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    first, second = list(candidates)
    store_workflow_row_configuration(
        definition, first, {"parameter_values": {"max_scf_cycles": 80}}
    )
    persisted: list[str] = []
    view = _FakeQuantumView()
    context = make_context(document=document, project_service=service)
    controller = QuantumCorrectionsController(
        dataclasses.replace(
            context, persist_ui_state=lambda **kwargs: persisted.append(kwargs["failure_message"])
        ),
        _stub_quantum_deps(),
        view,
    )

    controller.apply_configuration(
        [first, second],
        configure_result({"task": "PHONON", "xc_functional": "PBE"}, untouched={"max_scf_cycles"}),
        {"num_farms": 2},
    )

    first_params, _advanced, first_extras = workflow_row_configuration(definition, first)
    second_params, _advanced, second_extras = workflow_row_configuration(definition, second)
    assert first_params == {"xc_functional": "PBE", "max_scf_cycles": 80}
    assert second_params == {"xc_functional": "PBE"}
    assert first_extras == second_extras == {"num_farms": 2}
    assert definition.parameters["configuration"][
        "sources"
    ]  # the definition now records its sources
    assert persisted == ["Quantum Corrections configuration was updated, but autosave failed."]
    assert view.sources  # refreshed


def test_quantum_controller_renames_a_source_site(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key, candidate = next(iter(candidates.items()))
    refreshed: list[str] = []
    persisted: list[str] = []
    controller = _quantum_document_controller(
        service,
        document,
        output_root,
        deps=_stub_quantum_deps(refresh_site_label_views=lambda: refreshed.append("other pages")),
        persisted=persisted,
    )

    assert controller.rename_source(key, "  ") is False
    assert controller.rename_source(key, "siteA") is False
    assert controller.rename_source("missing", "renamed") is False
    assert candidate.site.label == "siteA"
    assert refreshed == []
    assert controller.rename_source(key, " renamed ") is True
    assert document.sites[candidate.site.id].label == "renamed"
    assert refreshed == ["other pages"]
    assert persisted == ["Quantum Corrections site label was updated, but autosave failed."]


def test_quantum_controller_resolves_the_phonon_analysis_input(tmp_path: Path) -> None:
    from mufinder.domain.models import Result

    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    job = _quantum_job(service, document, "quantum_phonon_siteA", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=job.id, status=JobStatus.COMPLETED)
    document.runs[run.id] = run
    result = Result(calculation_run_id=run.id, result_type=QUANTUM_PHONON_CALC_TYPE, data={})
    document.results[result.id] = result
    controller = QuantumCorrectionsController(
        make_context(document=document, project_service=service),
        _stub_quantum_deps(),
        _FakeQuantumView(),
    )

    assert controller.phonon_result("not-a-uuid") is None
    assert controller.phonon_result(str(uuid4())) is None
    assert controller.phonon_result(str(result.id)) is result
    with pytest.raises(ValueError, match="does not reference a phonon file"):
        controller.phonon_analysis_input(result)
    result.data["phonon_file"] = "files/site.phonon"
    with pytest.raises(ValueError, match="Could not find phonon file"):
        controller.phonon_analysis_input(result)
    phonon_path = document.project_root / "files" / "site.phonon"
    phonon_path.parent.mkdir(parents=True, exist_ok=True)
    phonon_path.write_text(_QUANTUM_PHONON_TEXT, encoding="utf-8")
    assert controller.phonon_analysis_input(result) == (phonon_path, "quantum_phonon_siteA")


def test_quantum_controller_names_the_analysis_window_after_the_site_and_charge_state(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    job = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    phonon_path = output_root / "site.phonon"
    phonon_path.parent.mkdir(parents=True, exist_ok=True)
    phonon_path.write_text(_QUANTUM_PHONON_TEXT, encoding="utf-8")
    result = _store_phonon_result(document, job, {"phonon_file": str(phonon_path)})
    controller = _quantum_document_controller(service, document, output_root)

    assert controller.phonon_analysis_input(result) == (phonon_path, "siteA, positive muon")


def test_quantum_controller_rename_does_nothing_while_shutting_down(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key, candidate = next(iter(candidates.items()))
    controller = _quantum_document_controller(
        service, document, output_root, deps=_stub_quantum_deps(shutting_down=lambda: True)
    )

    assert controller.rename_source(key, "renamed") is False
    assert candidate.site.label == "siteA"


@pytest.mark.parametrize(
    ("extras", "expected"),
    [
        ({}, {"a": 4, "b": 1}),
        ({"num_farms": 1}, {"a": 1, "b": 1}),
        ({"num_farms": 2}, {"a": 2, "b": 2}),
    ],
    ids=["left-alone", "set-to-one", "set-to-two"],
)
def test_quantum_controller_configure_keeps_differing_farm_counts_unless_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extras: dict, expected: dict[str, int]
) -> None:
    from mufinder.app.workflow_config_codecs import workflow_row_configuration
    from mufinder.app.workflow_controllers import quantum_corrections as module

    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key_a, key_b = list(candidates)
    store_workflow_row_configuration(definition, key_a, {"num_farms": 4})
    store_workflow_row_configuration(definition, key_b, {"num_farms": 1})
    monkeypatch.setattr(
        module,
        "run_source_configure_dialog",
        lambda _build, _request, **_kwargs: replace(configure_result(), extras=extras),
    )
    persisted: list[str] = []
    controller = _quantum_document_controller(service, document, output_root, persisted=persisted)

    controller.configure_sources([key_a, key_b])

    assert len(persisted) == 1

    farms = {
        name: workflow_row_configuration(definition, key)[2]["num_farms"]
        for name, key in (("a", key_a), ("b", key_b))
    }
    assert farms == expected


def test_quantum_controller_says_why_an_analysis_cannot_open(qapp, tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    job = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    phonon_path = output_root / "broken.phonon"
    phonon_path.parent.mkdir(parents=True, exist_ok=True)
    phonon_path.write_text("not a phonon file", encoding="utf-8")
    missing = _store_phonon_result(document, job, {})
    broken = _store_phonon_result(document, job, {"phonon_file": str(phonon_path)})
    infos: list[tuple[str, str]] = []
    controller = _quantum_document_controller(
        service,
        document,
        output_root,
        deps=_stub_quantum_deps(show_warning=lambda title, text: infos.append((title, text))),
    )

    controller.open_analysis("not-a-uuid")
    assert infos == []
    controller.open_analysis(str(missing.id))
    controller.open_analysis(str(broken.id))

    assert [title for title, _text in infos] == ["Phonon analysis unavailable"] * 2
    assert "does not reference a phonon file" in infos[0][1]
    assert controller.analysis_dialog(broken.id) is None


def test_quantum_controller_validate_artifacts_checks_phonon_jobs_only(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "q.mufinder", "Q")
    controller = QuantumCorrectionsController(
        make_context(document=document), _stub_quantum_deps(), _FakeQuantumView()
    )
    singlepoint = _quantum_job(
        service, document, "quantum_singlepoint_s", QUANTUM_SINGLEPOINT_CALC_TYPE
    )
    phonon = _quantum_job(service, document, "quantum_phonon_s", QUANTUM_PHONON_CALC_TYPE)
    run = CalculationRun(job_id=phonon.id)
    parsed = CastepResult(completed=True, total_energy=-1.0)

    assert (
        controller.validate_artifacts(
            singlepoint, parsed, run, ParsedJobArtifacts(phonon_missing=True)
        )
        is None
    )
    assert (
        controller.validate_artifacts(phonon, parsed, run, ParsedJobArtifacts(phonon_missing=True))
        == "Expected CASTEP phonon output was not found: quantum_phonon_s.phonon"
    )
    assert (
        controller.validate_artifacts(
            phonon, parsed, run, ParsedJobArtifacts(phonon_error="no Gamma point")
        )
        == "Failed to parse CASTEP phonon output: no Gamma point"
    )
    assert controller.validate_artifacts(phonon, parsed, run, no_artifacts()) is None


def test_quantum_controller_fresh_seed_and_legacy_binding_survive_a_reload(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key = next(iter(candidates))
    singlepoint = _bind_legacy_singlepoint(service, document, definition, key)
    definition.parameters["job_seed_names_by_label"][key] = "quantum_siteA"
    _write_legacy_seed_directory(output_root, "quantum_siteA")
    (job,) = _quantum_document_controller(service, document, output_root).materialize([key])
    service.save_project(document)

    loaded = service.load_project(document.project_root)
    controller = _quantum_document_controller(service, loaded, output_root)
    loaded_definition = controller.active_definition()

    assert loaded_definition is not None
    assert controller.phonon_job_for_key(loaded_definition, key).id == job.id
    assert controller.legacy_singlepoint_job_for_key(loaded_definition, key).id == singlepoint.id
    assert loaded_definition.parameters["job_seed_names_by_label"][key] == "quantum_siteA_phonon"
    assert loaded_definition.parameters["singlepoint_seed_names_by_label"] == {key: "quantum_siteA"}
    assert controller.build_states()[key].status == "ready"

    def seed(owner, row: str) -> str:
        return quantum_seed_name_for_key(loaded, owner, row)

    assert job_output_directory(loaded, loaded.jobs[singlepoint.id], quantum_seed_name=seed) == (
        output_root / "quantum_siteA"
    )


def test_running_only_unchanged_completed_rows_says_how_to_run_them_again(
    tmp_path: Path,
) -> None:
    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    _service, _document, _definition, keys, controller, jobs, _results, _root = _completed_rows(
        tmp_path, completed=2, infos=infos, local_backend=local
    )

    controller.submit(keys, target="local")

    assert infos == [
        (
            "Nothing to run",
            "These 2 sites have completed. To run one again, change one of its parameters "
            "with 'Configure…' first; running it then replaces its phonon result.",
        )
    ]
    assert local.calls == []

    # With another reason in the selection (a running row), the shared path answers.
    infos.clear()
    set_job_state(jobs[1], "running")
    controller.submit(keys, target="local")
    assert all(title != "Nothing to run" for title, _text in infos)


def test_a_completed_row_beside_a_ready_one_runs_the_ready_one_without_the_notice(
    tmp_path: Path,
) -> None:
    infos: list[tuple[str, str]] = []
    local = RecordingBackend()
    _service, _document, _definition, keys, controller, jobs, _results, _root = _completed_rows(
        tmp_path, infos=infos, local_backend=local
    )

    controller.submit(keys, target="local")

    assert infos == []
    ((submitted, _extras),) = local.calls
    # The backend runs the ready row's new job; the completed one is not run again.
    assert [job.status for job in submitted if job is not jobs[0]] == [JobStatus.DEFINED]
    assert jobs[0].status == JobStatus.COMPLETED


# --- The principal ZPEs in the analysis window --------------------------------------


def _site_a_with_toy_phonons(
    service: ProjectService, tmp_path: Path, muon_block_cm2, *, recorded_run: str | None = None
):
    """siteA with a completed phonon job whose muon block is ``muon_block_cm2``, on its own cell."""

    document, definition, candidates, output_root = _seed_quantum_document(service, tmp_path)
    key, candidate = next(iter(candidates.items()))
    job = _bind_phonon(service, document, definition, key, status=JobStatus.COMPLETED)
    if recorded_run is not None:
        definition.parameters["phonon_source_run_ids_by_label"][key] = recorded_run
    structure = candidate.structure
    calculation = phonon_calculation_from_dynamical_matrix(
        uncoupled_muon_dynamical_matrix(np.asarray(muon_block_cm2), (300.0,), muon_atom=1),
        lattice_vectors=structure.lattice_vectors,
        species=structure.species,
        fractional_positions=structure.fractional_positions,
        masses_amu=(28.0855, MUON_MASS_AMU),
    )
    phonon_path = output_root / "siteA.phonon"
    phonon_path.parent.mkdir(parents=True, exist_ok=True)
    phonon_path.write_text(phonon_calculation_text(calculation), encoding="utf-8")
    result = _store_phonon_result(
        document, job, {"phonon_file": str(phonon_path), "muon_zpe_total_eV": 0.5}
    )
    return document, output_root, result


def _open_analysis_by_clicking(qapp, service, document, output_root):
    """Select the row with a result on the real page and click ``Open Analysis``."""

    view = QuantumCorrectionsWidget()
    controller = QuantumCorrectionsController(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _definition: output_root,
        ),
        _stub_quantum_deps(),
        view,
    )
    view.analysis_requested.connect(controller.open_analysis)
    view.resize(1100, 500)
    view.show()
    controller.refresh_view()
    qapp.processEvents()
    table = view._sources_table
    row = next(index for index, source in enumerate(view._source_rows) if source.result_id)
    QTest.mouseClick(
        table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=table.visualItemRect(table.item(row, 0)).center(),
    )
    QTest.mouseClick(view._open_analysis_button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    dialog = next(iter(controller._analysis_dialogs.values()))
    return view, dialog


def test_the_analysis_window_gives_the_principal_zpes_in_the_hosts_axes(
    qapp, tmp_path: Path
) -> None:
    service = ProjectService()
    document, output_root, _result = _site_a_with_toy_phonons(
        service, tmp_path, np.diag([2000.0, 3000.0, 4000.0]) ** 2
    )

    view, dialog = _open_analysis_by_clicking(qapp, service, document, output_root)

    assert dialog._principal_zpe_label.text() == (
        "Muon ZPE along the principal axes of its well: "
        "124 meV along a · 186 meV along b · 248 meV along c"
    )
    tooltip = dialog._principal_zpe_label.toolTip()
    assert "every other atom held fixed" in tooltip
    assert "host crystal's axes" in tooltip
    assert "barrier" not in dialog._principal_zpe_label.text() + tooltip
    assert "stable" not in dialog._principal_zpe_label.text() + tooltip
    dialog.close()
    view.close()


def test_an_axis_of_negative_curvature_reads_a_dash_and_why(qapp, tmp_path: Path) -> None:
    service = ProjectService()
    document, output_root, _result = _site_a_with_toy_phonons(
        service, tmp_path, np.diag([2000.0**2, 3000.0**2, -(900.0**2)])
    )

    view, dialog = _open_analysis_by_clicking(qapp, service, document, output_root)

    assert dialog._principal_zpe_label.text() == (
        "Muon ZPE along the principal axes of its well: "
        "– along c (curvature not positive) · 124 meV along a · 186 meV along b"
    )
    assert NO_REAL_FREQUENCY_ALONG_AXIS in dialog._principal_zpe_label.toolTip()
    dialog.close()
    view.close()


def test_equal_principal_zpes_name_a_plane_rather_than_an_axis(qapp, tmp_path: Path) -> None:
    service = ProjectService()
    document, output_root, _result = _site_a_with_toy_phonons(
        service, tmp_path, np.diag([2000.0, 2000.0, 4000.0]) ** 2
    )

    view, dialog = _open_analysis_by_clicking(qapp, service, document, output_root)

    assert dialog._principal_zpe_label.text() == (
        "Muon ZPE along the principal axes of its well: "
        "124 meV twice, in the plane normal to c · 248 meV along c"
    )
    dialog.close()
    view.close()


def test_principal_zpes_from_an_earlier_relaxation_are_marked(qapp, tmp_path: Path) -> None:
    service = ProjectService()
    document, output_root, _result = _site_a_with_toy_phonons(
        service, tmp_path, np.diag([2000.0, 3000.0, 4000.0]) ** 2, recorded_run=str(uuid4())
    )

    view, dialog = _open_analysis_by_clicking(qapp, service, document, output_root)

    assert dialog._principal_zpe_label.text().startswith(
        "Muon ZPE along the principal axes of its well (earlier): 124 meV"
    )
    assert "earlier geometry" in dialog._principal_zpe_label.toolTip()
    dialog.close()
    view.close()


def test_an_edited_rows_window_keeps_its_principal_zpes_until_the_result_goes(
    qapp, tmp_path: Path, configure_with, replace_questions
) -> None:
    """The line reads the result the window shows; replacing the result closes the window."""

    _service, document, _definition, keys, controller, _jobs, results, _root = _completed_rows(
        tmp_path
    )
    controller.open_analysis(str(results[0].id))
    dialog = controller.analysis_dialog(results[0].id)
    line = dialog.principal_zpe_text()
    assert line.startswith("Muon ZPE along the principal axes of its well: ")

    configure_with(_LOOSER_TOLERANCE)
    controller.configure_sources([keys[0]])
    controller.refresh_view()

    assert controller.build_states()[keys[0]].status == "edited"
    assert controller.analysis_dialog(results[0].id) is dialog
    assert dialog.principal_zpe_text() == line

    # Running the edited row again removes its result: the window closes, raising nothing.
    del document.results[results[0].id]
    controller.refresh_view()

    assert controller.analysis_dialog(results[0].id) is None
    controller.open_analysis(str(results[0].id))
    assert controller.analysis_dialog(results[0].id) is None


def test_a_window_sorted_on_muon_coupling_closes_cleanly_when_its_result_goes(
    qapp, tmp_path: Path
) -> None:
    from PySide6.QtWidgets import QTabWidget

    from tests.support.qt import click_header_section

    _service, document, _definition, _keys, controller, _jobs, results, _root = _completed_rows(
        tmp_path
    )
    controller.open_analysis(str(results[0].id))
    dialog = controller.analysis_dialog(results[0].id)
    tabs = dialog.findChild(QTabWidget, "quantumAnalysisTabs")
    tabs.setCurrentIndex(tabs.count() - 1)
    table = dialog._neighbour_table
    click_header_section(table, 3)
    assert table.horizontalHeader().sortIndicatorSection() == 3

    del document.results[results[0].id]
    controller.refresh_view()

    assert controller.analysis_dialog(results[0].id) is None
