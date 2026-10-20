"""Tests for Magnetic Fields results stored as project ``Result`` records."""

from __future__ import annotations

from pathlib import Path

from mufinder.services.magnetic_field_results import (
    MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE,
    MAGNETIC_FIELD_PAGE_OWNER,
    MAGNETIC_FIELD_RESULT_TYPE,
    NUCLEAR_RELAXATION_RESULT_TYPE,
    attach_magnetic_field_results,
    commensurate_field_entries,
    magnetic_field_result,
    magnetic_field_results,
    remove_magnetic_field_results,
    store_magnetic_field_results,
)
from mufinder.services.project_service import ProjectService


_FIELD = {
    "kind": "commensurate",
    "results": [{"site_label": "site 1", "muon_position": [0.25, 0.25, 0.25], "total_T": [0.1, 0.2, 0.3]}],
}
_TENSORS = [{"site_label": "site 1", "tensor": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]]}]
_NUCLEAR = {"mode": "zero_field_powder", "sigma_us_inv": 0.4}


def _form(**overrides: object) -> dict[str, object]:
    return {
        "histogram_bins": 512,
        "magnetic_field_results": _FIELD,
        "dipolar_tensor_results": _TENSORS,
        "nuclear_relaxation_results": _NUCLEAR,
        **overrides,
    }


def _document(tmp_path: Path):
    return ProjectService().create_project(tmp_path / "magnetic.mufinder", "Magnetic")


def test_store_records_each_result_and_returns_the_form_without_them(tmp_path: Path) -> None:
    document = _document(tmp_path)

    stored = store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form())

    assert stored == {"histogram_bins": 512}
    records = {record.result_type: record for record in magnetic_field_results(document)}
    assert set(records) == {
        MAGNETIC_FIELD_RESULT_TYPE,
        MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE,
        NUCLEAR_RELAXATION_RESULT_TYPE,
    }
    assert all(record.calculation_run_id is None for record in records.values())
    assert records[MAGNETIC_FIELD_RESULT_TYPE].data == {"owner": MAGNETIC_FIELD_PAGE_OWNER, **_FIELD}
    assert records[MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE].data == {
        "owner": MAGNETIC_FIELD_PAGE_OWNER,
        "results": _TENSORS,
    }
    assert records[NUCLEAR_RELAXATION_RESULT_TYPE].data == {"owner": MAGNETIC_FIELD_PAGE_OWNER, "result": _NUCLEAR}


def test_attach_restores_the_form_the_page_stored(tmp_path: Path) -> None:
    document = _document(tmp_path)
    stored = store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form())

    assert attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, stored) == _form()
    assert attach_magnetic_field_results(document, "other-workspace", None) == {}


def test_attach_gives_every_missing_result_an_explicit_empty_value(tmp_path: Path) -> None:
    # A page reused across projects must clear results the owner has no record of.
    document = _document(tmp_path)
    store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, {"magnetic_field_results": _FIELD})

    assert attach_magnetic_field_results(document, "other-workspace", {"histogram_bins": 512}) == {
        "histogram_bins": 512,
        "magnetic_field_results": None,
        "dipolar_tensor_results": [],
        "nuclear_relaxation_results": {},
    }
    assert attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, {}) == {
        "magnetic_field_results": _FIELD,
        "dipolar_tensor_results": [],
        "nuclear_relaxation_results": {},
    }


def test_unchanged_results_keep_their_record_and_changed_results_replace_it(tmp_path: Path) -> None:
    document = _document(tmp_path)
    store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form())
    before = {record.result_type: record.id for record in magnetic_field_results(document)}

    store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form(histogram_bins=1024))
    assert {record.result_type: record.id for record in magnetic_field_results(document)} == before

    recalculated = {"kind": "commensurate", "results": [{"site_label": "site 1", "total_T": [0.0, 0.0, 1.0]}]}
    store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form(magnetic_field_results=recalculated))

    after = {record.result_type: record.id for record in magnetic_field_results(document)}
    assert len(document.results) == 3
    assert after[MAGNETIC_FIELD_RESULT_TYPE] != before[MAGNETIC_FIELD_RESULT_TYPE]
    assert after[NUCLEAR_RELAXATION_RESULT_TYPE] == before[NUCLEAR_RELAXATION_RESULT_TYPE]
    assert magnetic_field_result(document, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_FIELD_RESULT_TYPE).data["results"] == (
        recalculated["results"]
    )


def test_cleared_or_empty_results_remove_their_record(tmp_path: Path) -> None:
    document = _document(tmp_path)
    store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form())

    store_magnetic_field_results(
        document,
        MAGNETIC_FIELD_PAGE_OWNER,
        _form(
            magnetic_field_results=None,
            dipolar_tensor_results=[],
            nuclear_relaxation_results={},
        ),
    )

    assert document.results == {}


def test_owners_keep_separate_records(tmp_path: Path) -> None:
    document = _document(tmp_path)
    store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form())
    store_magnetic_field_results(document, "workspace-1", _form())

    remove_magnetic_field_results(document, "workspace-1")

    assert magnetic_field_results(document, "workspace-1") == []
    assert len(magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER)) == 3


def test_commensurate_field_entries_skip_distributions_and_malformed_entries(tmp_path: Path) -> None:
    document = _document(tmp_path)
    store_magnetic_field_results(
        document,
        "commensurate",
        {"magnetic_field_results": {"kind": "commensurate", "results": [{"site_label": "a"}, "bad"]}},
    )
    store_magnetic_field_results(
        document,
        "distribution",
        {"magnetic_field_results": {"kind": "distribution", "results": [{"site_label": "b"}]}},
    )

    commensurate = magnetic_field_result(document, "commensurate", MAGNETIC_FIELD_RESULT_TYPE)
    distribution = magnetic_field_result(document, "distribution", MAGNETIC_FIELD_RESULT_TYPE)
    assert list(commensurate_field_entries(commensurate)) == [{"site_label": "a"}]
    assert list(commensurate_field_entries(distribution)) == []


def test_records_survive_a_save_and_reload(tmp_path: Path) -> None:
    service = ProjectService()
    document = _document(tmp_path)
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, _form()
    )
    service.save_project(document)

    reloaded = service.load_project(document.project_root)

    assert attach_magnetic_field_results(
        reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
    ) == _form()
