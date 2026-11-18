"""Tests for Magnetic Fields results stored as project ``Result`` records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mufinder.science.magnetic_fields import (
    FieldTermStatistics,
    MagneticFieldDistributionResult,
    MagneticFieldResult,
    regenerate_phase_samples,
)
from mufinder.science.polarisation import calculate_polarisation
from mufinder.services.magnetic_field_results import (
    CALCULATION_METHOD_VERSION,
    CALCULATION_VERSION_KEY,
    MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE,
    MAGNETIC_FIELD_PAGE_OWNER,
    MAGNETIC_FIELD_RESULT_TYPE,
    MUON_POLARISATION_RESULT_TYPE,
    NUCLEAR_RELAXATION_RESULT_TYPE,
    attach_magnetic_field_results,
    commensurate_field_entries,
    field_result_site_source,
    is_unrelaxed_site_source,
    magnetic_field_result,
    magnetic_field_results,
    polarisation_result,
    remove_magnetic_field_results,
    store_magnetic_field_results,
    store_polarisation_result,
    uses_older_method,
    PLATE_METHOD_VERSION,
    PLATE_METHOD_VERSION_KEY,
    TRANSLATION_METHOD_NOT_USED,
    TRANSLATION_METHOD_VERSION,
    TRANSLATION_METHOD_VERSION_KEY,
    recorded_translation_method_version,
    field_result_uses_older_method,
    magnetic_field_owner_form,
    magnetic_field_owner_model,
    translation_method_applies,
)
from mufinder.services.project_service import ProjectService
from tests.support.magnetic_fields import collinear_waves_model


_FIELD = {
    "kind": "commensurate",
    "results": [
        {"site_label": "site 1", "muon_position": [0.25, 0.25, 0.25], "total_T": [0.1, 0.2, 0.3]}
    ],
}
_TENSORS = [
    {"site_label": "site 1", "tensor": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]]}
]
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
    assert records[MAGNETIC_FIELD_RESULT_TYPE].data == {
        "owner": MAGNETIC_FIELD_PAGE_OWNER,
        **_FIELD,
    }
    assert records[MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE].data == {
        "owner": MAGNETIC_FIELD_PAGE_OWNER,
        "results": _TENSORS,
    }
    assert records[NUCLEAR_RELAXATION_RESULT_TYPE].data == {
        "owner": MAGNETIC_FIELD_PAGE_OWNER,
        "result": _NUCLEAR,
    }


def test_attach_restores_the_form_the_page_stored(tmp_path: Path) -> None:
    document = _document(tmp_path)
    stored = store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form())

    assert attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, stored) == _form()
    assert attach_magnetic_field_results(document, "other-workspace", None) == {}


@pytest.mark.parametrize(
    ("result", "older"),
    [
        ({"sigma": 0.2}, True),
        ({"sigma": 0.2, CALCULATION_VERSION_KEY: None}, True),
        ({"sigma": 0.2, CALCULATION_VERSION_KEY: "1"}, True),
        ({"sigma": 0.2, CALCULATION_VERSION_KEY: True}, True),
        ({"sigma": 0.2, CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION - 1}, True),
        ({"sigma": 0.2, CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION}, False),
        ({"sigma": 0.2, CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION + 1}, False),
        ({}, False),
        (["not", "a", "result"], False),
    ],
)
def test_a_result_is_older_by_its_stored_version(result, older) -> None:
    assert uses_older_method(result) is older


_CURRENT = {"results": [1], CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION}


@pytest.mark.parametrize(
    ("form", "older"),
    [
        ({"demag_shape": "image_plate"}, True),
        ({"demag_shape": "none"}, False),
        ({"demag_shape": "sphere"}, False),
        ({"demag_shape": "infinite_plate"}, False),
        ({"demag_shape": "rectangular_prism"}, False),
        ({"demag_shape": "voxel_grid", "demag_sample": {"padding": 2.5, "smoothing": 0.0}}, False),
        ({"demag_shape": "voxel_grid", "demag_sample": {}}, False),
        ({"demag_shape": "voxel_grid", "demag_sample": {"padding": 3.0}}, True),
        ({"demag_shape": "voxel_grid", "demag_sample": {"smoothing": 0.5}}, True),
        # Not what the form holds now: the advanced sample is kept but not selected.
        ({"demag_shape": "sphere", "demag_sample": {"shape": "image_plate"}}, False),
        (None, False),
    ],
)
def test_a_result_with_no_plate_version_is_decided_by_its_saved_sample(form, older) -> None:
    assert field_result_uses_older_method(_CURRENT, form) is older


@pytest.mark.parametrize(
    ("recorded", "older"),
    [
        (PLATE_METHOD_VERSION, False),
        (PLATE_METHOD_VERSION + 1, False),
        (PLATE_METHOD_VERSION - 1, True),
        # What the plate from a photo recorded at a padding of 2.5.
        (2, True),
        ("2", True),
        (True, True),
    ],
)
def test_a_recorded_plate_version_decides_whatever_the_form_holds(recorded, older) -> None:
    result = {**_CURRENT, PLATE_METHOD_VERSION_KEY: recorded}

    assert field_result_uses_older_method(result, {"demag_shape": "image_plate"}) is older
    assert field_result_uses_older_method(result, {"demag_shape": "none"}) is older


def test_an_older_calculation_version_is_older_with_or_without_a_plate() -> None:
    assert field_result_uses_older_method({"results": [1]}, {"demag_shape": "none"})
    assert field_result_uses_older_method(
        {"results": [1], PLATE_METHOD_VERSION_KEY: PLATE_METHOD_VERSION}, None
    )
    assert not field_result_uses_older_method({}, {"demag_shape": "image_plate"})


def test_a_plate_version_is_stored_and_comes_back(tmp_path: Path) -> None:
    document = _document(tmp_path)
    fields = {
        **_form()["magnetic_field_results"],
        CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION,
        PLATE_METHOD_VERSION_KEY: PLATE_METHOD_VERSION,
    }

    store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, {"magnetic_field_results": fields}
    )

    record = magnetic_field_result(document, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_FIELD_RESULT_TYPE)
    assert record.data[PLATE_METHOD_VERSION_KEY] == PLATE_METHOD_VERSION
    attached = attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, {})
    assert attached["magnetic_field_results"][PLATE_METHOD_VERSION_KEY] == PLATE_METHOD_VERSION


def test_the_owner_form_comes_from_the_page_or_its_workspace(tmp_path: Path) -> None:
    document = _document(tmp_path)
    document.ui_state["magnetic_field_form"] = {"demag_shape": "sphere"}
    document.ui_state["magnetic_field_workspaces"] = [
        {"id": "w1", "form": {"demag_shape": "image_plate"}}
    ]

    assert magnetic_field_owner_form(document, MAGNETIC_FIELD_PAGE_OWNER)["demag_shape"] == "sphere"
    assert magnetic_field_owner_form(document, "w1")["demag_shape"] == "image_plate"
    assert magnetic_field_owner_form(document, "gone") is None


def test_each_record_keeps_the_version_its_result_was_calculated_with(tmp_path: Path) -> None:
    document = _document(tmp_path)
    version = {CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION}
    form = _form(
        magnetic_field_results={**_FIELD, **version},
        dipolar_tensor_results=[{**_TENSORS[0], **version}],
        nuclear_relaxation_results={**_NUCLEAR, **version},
    )

    stored = store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, form)

    for record in magnetic_field_results(document):
        assert record.data[CALCULATION_VERSION_KEY] == CALCULATION_METHOD_VERSION
    assert attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, stored) == form
    tensors = magnetic_field_result(
        document, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE
    )
    assert tensors.data["results"] == _TENSORS
    nuclear = magnetic_field_result(
        document, MAGNETIC_FIELD_PAGE_OWNER, NUCLEAR_RELAXATION_RESULT_TYPE
    )
    assert nuclear.data["result"] == _NUCLEAR


def test_storing_an_older_result_again_never_marks_it_current(tmp_path: Path) -> None:
    document = _document(tmp_path)
    stored = store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, _form())
    before = {record.id: dict(record.data) for record in magnetic_field_results(document)}

    attached = attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, stored)
    store_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, attached)

    assert {record.id: dict(record.data) for record in magnetic_field_results(document)} == before
    assert all(uses_older_method(record.data) for record in magnetic_field_results(document))
    # A copy of older results, as a pinned workspace takes, stays older too.
    store_magnetic_field_results(document, "workspace", attached)
    assert all(
        uses_older_method(record.data) for record in magnetic_field_results(document, "workspace")
    )


def test_tensor_rows_of_mixed_versions_keep_their_own_and_read_as_older(tmp_path: Path) -> None:
    document = _document(tmp_path)
    rows = [
        {**_TENSORS[0], CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION},
        dict(_TENSORS[0]),
    ]

    store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, {"dipolar_tensor_results": rows}
    )

    attached = attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, {})
    assert attached["dipolar_tensor_results"] == rows
    assert any(uses_older_method(row) for row in attached["dipolar_tensor_results"])


def test_attach_gives_every_missing_result_an_explicit_empty_value(tmp_path: Path) -> None:
    # A page reused across projects must clear results the owner has no record of.
    document = _document(tmp_path)
    store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, {"magnetic_field_results": _FIELD}
    )

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

    recalculated = {
        "kind": "commensurate",
        "results": [{"site_label": "site 1", "total_T": [0.0, 0.0, 1.0]}],
    }
    store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, _form(magnetic_field_results=recalculated)
    )

    after = {record.result_type: record.id for record in magnetic_field_results(document)}
    assert len(document.results) == 3
    assert after[MAGNETIC_FIELD_RESULT_TYPE] != before[MAGNETIC_FIELD_RESULT_TYPE]
    assert after[NUCLEAR_RELAXATION_RESULT_TYPE] == before[NUCLEAR_RELAXATION_RESULT_TYPE]
    assert (
        magnetic_field_result(document, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_FIELD_RESULT_TYPE).data[
            "results"
        ]
        == (recalculated["results"])
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


def test_commensurate_field_entries_skip_distributions_and_malformed_entries(
    tmp_path: Path,
) -> None:
    document = _document(tmp_path)
    store_magnetic_field_results(
        document,
        "commensurate",
        {
            "magnetic_field_results": {
                "kind": "commensurate",
                "results": [{"site_label": "a"}, "bad"],
            }
        },
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

    assert (
        attach_magnetic_field_results(
            reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
        )
        == _form()
    )


def test_field_records_keep_the_site_source_through_a_save_and_reload(tmp_path: Path) -> None:
    service = ProjectService()
    document = _document(tmp_path)
    form = _form(magnetic_field_results={**_FIELD, "site_source": "uep_clustering"})
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, form
    )
    service.save_project(document)

    reloaded = service.load_project(document.project_root)

    record = magnetic_field_result(reloaded, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_FIELD_RESULT_TYPE)
    assert record.data["site_source"] == "uep_clustering"
    attached = attach_magnetic_field_results(
        reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
    )
    assert attached == form


def test_attach_takes_a_missing_site_source_from_the_selected_site(tmp_path: Path) -> None:
    document = _document(tmp_path)
    selected = {
        "source": "uep_clustering",
        "label": "UEP cluster 1",
        "fractional_position": [0.1, 0.2, 0.3],
    }
    stored = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, _form(selected_muon_site=selected)
    )

    attached = attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, stored)

    assert attached["magnetic_field_results"] == {**_FIELD, "site_source": "uep_clustering"}
    # Without a selected site nothing is inferred.
    other = store_magnetic_field_results(document, "other", _form())
    assert (
        "site_source"
        not in attach_magnetic_field_results(document, "other", other)["magnetic_field_results"]
    )


def test_tensor_and_nuclear_records_keep_the_site_source_through_a_save_and_reload(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document = _document(tmp_path)
    form = _form(
        dipolar_tensor_results=[{**row, "site_source": "uep_clustering"} for row in _TENSORS],
        nuclear_relaxation_results={**_NUCLEAR, "site_source": "relaxation_clustering"},
    )
    document.ui_state["magnetic_field_form"] = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, form
    )
    service.save_project(document)

    reloaded = service.load_project(document.project_root)

    tensor = magnetic_field_result(
        reloaded, MAGNETIC_FIELD_PAGE_OWNER, MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE
    )
    assert tensor.data == {
        "owner": MAGNETIC_FIELD_PAGE_OWNER,
        "results": _TENSORS,
        "site_source": "uep_clustering",
    }
    nuclear = magnetic_field_result(
        reloaded, MAGNETIC_FIELD_PAGE_OWNER, NUCLEAR_RELAXATION_RESULT_TYPE
    )
    assert nuclear.data == {
        "owner": MAGNETIC_FIELD_PAGE_OWNER,
        "result": _NUCLEAR,
        "site_source": "relaxation_clustering",
    }
    attached = attach_magnetic_field_results(
        reloaded, MAGNETIC_FIELD_PAGE_OWNER, reloaded.ui_state["magnetic_field_form"]
    )
    assert attached == form


def test_tensor_and_nuclear_records_without_a_source_take_none_from_the_selected_site(
    tmp_path: Path,
) -> None:
    # These results survive a change of muon site, so the selected site says nothing about them.
    document = _document(tmp_path)
    selected = {
        "source": "uep_clustering",
        "label": "UEP cluster 1",
        "fractional_position": [0.1, 0.2, 0.3],
    }
    stored = store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, _form(selected_muon_site=selected)
    )

    attached = attach_magnetic_field_results(document, MAGNETIC_FIELD_PAGE_OWNER, stored)

    assert attached["dipolar_tensor_results"] == _TENSORS
    assert attached["nuclear_relaxation_results"] == _NUCLEAR


def test_only_uep_sites_are_unrelaxed_estimates() -> None:
    assert is_unrelaxed_site_source("uep_clustering")
    assert not is_unrelaxed_site_source("relaxation_clustering")
    assert not is_unrelaxed_site_source("custom")
    assert not is_unrelaxed_site_source(None)
    assert (
        field_result_site_source(
            {"site_source": "custom"}, {"selected_muon_site": {"source": "uep_clustering"}}
        )
        == "custom"
    )
    assert (
        field_result_site_source({}, {"selected_muon_site": {"source": "uep_clustering"}})
        == "uep_clustering"
    )
    assert field_result_site_source({}, None) is None


def test_a_saved_distribution_keeps_its_term_breakdown_and_phase_sampling(tmp_path: Path) -> None:
    distribution = MagneticFieldDistributionResult(
        site_label="site 1",
        muon_position=(0.25, 0.25, 0.25),
        fields_T=np.asarray([[0.1, 0.0, 0.05], [-0.1, 0.0, 0.05]], dtype=float),
        histogram_centres_T=np.asarray([0.1118], dtype=float),
        histogram_counts=np.asarray([2.0], dtype=float),
        calculation_parameters={
            "phase_basis_rlu": [[0.12345, 0.0, 0.0]],
            "phase_sample_count": 2,
            "resolved_phase_sampling": "montecarlo",
            "phase_sampling_seed": 2**62 + 7,
            "component_phase_coefficients": [[1.0]],
        },
        terms={
            "dipolar": FieldTermStatistics(mean_T=(0.0, 0.0, 0.0), rms_T=0.1),
            "lorentz": FieldTermStatistics(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "contact": FieldTermStatistics(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "demag": FieldTermStatistics(mean_T=(0.0, 0.0, 0.0), rms_T=0.0),
            "applied": FieldTermStatistics(mean_T=(0.0, 0.0, 0.05), rms_T=0.05),
        },
        total_rms_T=0.1118,
        applied_T=(0.0, 0.0, 0.05),
        demag_T=(0.0, 0.0, 0.0),
    )
    service = ProjectService()
    document = _document(tmp_path)
    field = {"kind": "distribution", "results": [distribution.to_json()]}
    store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, _form(magnetic_field_results=field)
    )
    service.save_project(document)

    reloaded = service.load_project(document.project_root)
    attached = attach_magnetic_field_results(reloaded, MAGNETIC_FIELD_PAGE_OWNER, {})
    restored = MagneticFieldDistributionResult.from_json(
        attached["magnetic_field_results"]["results"][0]
    )

    assert restored.terms == distribution.terms
    assert restored.applied_T == (0.0, 0.0, 0.05)
    assert restored.calculation_parameters["phase_sampling_seed"] == 2**62 + 7
    np.testing.assert_array_equal(
        regenerate_phase_samples(restored).phases_turns,
        regenerate_phase_samples(distribution).phases_turns,
    )


def _polarisation(lambda_z: float = 0.0):
    field = MagneticFieldResult(
        "site 1", (0.0, 0.0, 0.0), (0.05, 0.0, 0.1), *([(0.0, 0.0, 0.0)] * 4)
    )
    return calculate_polarisation(field, np.linspace(0.0, 5.0, 51), lambda_z=lambda_z)


def test_a_polarisation_record_is_stored_replaced_and_removed_per_owner(tmp_path: Path) -> None:
    document = _document(tmp_path)

    first = store_polarisation_result(document, "workspace-1", _polarisation())
    again = store_polarisation_result(document, "workspace-1", _polarisation())
    store_polarisation_result(document, MAGNETIC_FIELD_PAGE_OWNER, _polarisation(0.5))
    changed = store_polarisation_result(document, "workspace-1", _polarisation(0.2))

    assert again is first
    assert changed is not None and changed.id != first.id
    assert first.id not in document.results
    assert changed.result_type == MUON_POLARISATION_RESULT_TYPE
    assert polarisation_result(document, "workspace-1").parameters["lambda_z"] == 0.2
    assert polarisation_result(document, MAGNETIC_FIELD_PAGE_OWNER).parameters["lambda_z"] == 0.5
    # The form-state round trip leaves the polarisation record alone.
    store_magnetic_field_results(document, "workspace-1", _form())
    assert polarisation_result(document, "workspace-1") is not None

    assert store_polarisation_result(document, "workspace-1", None) is None
    assert polarisation_result(document, "workspace-1") is None
    assert store_polarisation_result(document, "workspace-1", None) is None
    store_polarisation_result(document, "workspace-1", _polarisation())
    remove_magnetic_field_results(document, "workspace-1")
    assert polarisation_result(document, "workspace-1") is None
    assert polarisation_result(document, MAGNETIC_FIELD_PAGE_OWNER) is not None


def test_a_polarisation_record_survives_a_save_and_reload(tmp_path: Path) -> None:
    service = ProjectService()
    document = _document(tmp_path)
    result = _polarisation(0.3)
    store_polarisation_result(document, MAGNETIC_FIELD_PAGE_OWNER, result)
    service.save_project(document)

    restored = polarisation_result(
        service.load_project(document.project_root), MAGNETIC_FIELD_PAGE_OWNER
    )

    assert restored is not None
    np.testing.assert_allclose(restored.polarisation, result.polarisation)
    assert restored.site_labels == ("site 1",)


def test_a_malformed_polarisation_record_reads_as_none(tmp_path: Path) -> None:
    document = _document(tmp_path)
    record = store_polarisation_result(document, "workspace-1", _polarisation())
    record.data["result"] = {"times_us": "broken"}

    assert polarisation_result(document, "workspace-1") is None
    record.data["result"] = None
    assert polarisation_result(document, "workspace-1") is None


def _field_rows(*positions, modulated: bool = True) -> list[dict[str, object]]:
    return [
        {"muon_position": list(position), "calculation_parameters": {"modulated": modulated}}
        for position in positions
    ]


_HOME = (0.25, 0.5, 0.75)
_IMAGE = (0.75, 0.5, 0.25)
_TRANSLATED = (1.25, 0.5, 0.75)


def _current_field_result(kind: str, rows: list[dict[str, object]], **extra) -> dict[str, object]:
    return {
        "kind": kind,
        "results": rows,
        CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION,
        **extra,
    }


@pytest.mark.parametrize(
    ("result", "older", "recorded"),
    [
        # A q != 0 result from before: the field at T = 0 alone, or a period above 12 as one row.
        (_current_field_result("commensurate", _field_rows(_HOME)), True, 1),
        (_current_field_result("commensurate", _field_rows(_HOME, _IMAGE)), True, 1),
        # q = 0 had nothing to change, however many rows it holds.
        (
            _current_field_result("commensurate", _field_rows(_HOME, modulated=False)),
            False,
            TRANSLATION_METHOD_NOT_USED,
        ),
        # A distribution sampled every phase.
        (
            _current_field_result("distribution", _field_rows(_HOME)),
            False,
            TRANSLATION_METHOD_NOT_USED,
        ),
        # The list of an approximant named its translations.
        (
            _current_field_result("commensurate", _field_rows(_HOME, _TRANSLATED)),
            False,
            TRANSLATION_METHOD_NOT_USED,
        ),
    ],
    ids=["one-row", "orbit", "uniform", "distribution", "approximant"],
)
def test_a_result_with_no_translation_version_is_decided_by_its_rows(
    result, older, recorded
) -> None:
    assert field_result_uses_older_method(result) is older
    assert recorded_translation_method_version(result) == recorded


@pytest.mark.parametrize(
    ("recorded", "older"),
    [
        (TRANSLATION_METHOD_VERSION, False),
        (TRANSLATION_METHOD_VERSION + 1, False),
        (2, False),
        (1, True),
        (TRANSLATION_METHOD_NOT_USED, False),
        ("2", True),
        (True, True),
    ],
)
def test_a_recorded_translation_version_marks_only_affected_commensurate_rows(
    recorded, older
) -> None:
    result = _current_field_result(
        "commensurate", _field_rows(_HOME), **{TRANSLATION_METHOD_VERSION_KEY: recorded}
    )

    assert field_result_uses_older_method(result) is older


@pytest.mark.parametrize("version", [None, 2, TRANSLATION_METHOD_VERSION])
@pytest.mark.parametrize(
    ("vectors", "affected"),
    [
        (((0.22, 0, 0), (0.78, 0, 0)), True),
        (((0.22, 0, 0), (0.44, 0, 0), (0.56, 0, 0)), True),
        (((0.22, 0, 0), (1.22, 1, -1)), True),
        (((0.22, 0, 0), (0.44, 0, 0)), False),
        (((0.22, 0, 0), (-0.22, 0, 0)), False),
        (((0.22, 0, 0), (0, 0.78, 0)), False),
        (((0.25, 0, 0), (0.75, 0, 0)), False),
    ],
    ids=[
        "complement",
        "dependent-complement",
        "shift",
        "harmonic",
        "negative",
        "independent",
        "short",
    ],
)
def test_only_aliased_distributions_predate_the_phase_basis_correction(
    version, vectors, affected
) -> None:
    rows = _field_rows(_HOME)
    rows[0]["calculation_parameters"]["commensurate_max_period"] = 12
    result = _current_field_result("distribution", rows)
    if version is not None:
        result[TRANSLATION_METHOD_VERSION_KEY] = version
    model = _model_json(*vectors)
    older = version != TRANSLATION_METHOD_VERSION and affected
    # Keyless short-period distributions also predate exact translation sampling.
    if version is None and vectors[0][0] == 0.25:
        older = True
    assert field_result_uses_older_method(result, None, model) is older
    effective = recorded_translation_method_version(result, model)
    assert (
        field_result_uses_older_method(
            {
                CALCULATION_VERSION_KEY: CALCULATION_METHOD_VERSION,
                TRANSLATION_METHOD_VERSION_KEY: effective,
            }
        )
        is older
    )


def test_a_translation_version_is_stored_and_comes_back(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "translation.mufinder", "Translation")
    value = _current_field_result(
        "commensurate",
        _field_rows(_HOME, _TRANSLATED),
        **{TRANSLATION_METHOD_VERSION_KEY: TRANSLATION_METHOD_VERSION},
    )

    store_magnetic_field_results(
        document, MAGNETIC_FIELD_PAGE_OWNER, {"magnetic_field_results": value}
    )
    service.save_project(document)
    reloaded = service.load_project(document.project_root)

    restored = attach_magnetic_field_results(reloaded, MAGNETIC_FIELD_PAGE_OWNER, {})[
        "magnetic_field_results"
    ]
    assert restored[TRANSLATION_METHOD_VERSION_KEY] == TRANSLATION_METHOD_VERSION
    assert not field_result_uses_older_method(restored)


def _model_json(*q_vectors: tuple[float, float, float]) -> dict[str, object]:
    return collinear_waves_model(*q_vectors).to_json()


_MNSI_Q = (0.0147, 0.0147, 0.0147)
_INCOMMENSURATE_Q = (0.123456, 0.0, 0.0)


@pytest.mark.parametrize(
    ("max_period", "model", "older"),
    [
        # Saved at period 12 or less listed: a short period beside the helix stayed at T = 0.
        (12, _model_json((0.22, 0.0, 0.0), (0.25, 0.0, 0.0)), True),
        (12, _model_json((0.22, 0.0, 0.0), (0.0, 0.0, 0.5)), True),
        # Period 13 and up was already sampled there, as was a helix alone or a uniform part.
        (12, _model_json((0.22, 0.0, 0.0)), False),
        (12, _model_json((0.22, 0.0, 0.0), (1.0 / 13.0, 0.0, 0.0)), False),
        (12, _model_json((0.22, 0.0, 0.0), (1.0, 0.0, 0.0)), False),
        # Saved when periods up to 64 counted as commensurate: those stayed at T = 0 too.
        (64, _model_json(_INCOMMENSURATE_Q, (0.22, 0.0, 0.0)), True),
        (64, _model_json(_INCOMMENSURATE_Q, (1.0 / 13.0, 0.0, 0.0)), True),
        (64, _model_json(_INCOMMENSURATE_Q, (0.0, 0.0, 0.5)), True),
        (64, _model_json(_MNSI_Q), False),
        (64, _model_json(_INCOMMENSURATE_Q), False),
        # Saved before the period was recorded: the limit was 64.
        (None, _model_json(_INCOMMENSURATE_Q, (0.22, 0.0, 0.0)), True),
        (None, _model_json(_MNSI_Q), False),
        # No model to read: not marked.
        (64, None, False),
        (64, {"components": "malformed"}, False),
    ],
    ids=[
        "12-quarter",
        "12-half",
        "12-helix",
        "12-thirteen",
        "12-uniform",
        "64-fifty",
        "64-thirteen",
        "64-half",
        "64-mnsi",
        "64-helix",
        "unrecorded-fifty",
        "unrecorded-mnsi",
        "no-model",
        "malformed",
    ],
)
def test_a_distribution_with_no_translation_version_is_decided_by_its_owners_model(
    max_period, model, older
) -> None:
    rows = _field_rows(_HOME)
    if max_period is not None:
        rows[0]["calculation_parameters"]["commensurate_max_period"] = max_period
    result = _current_field_result("distribution", rows)

    assert field_result_uses_older_method(result, None, model) is older
    assert recorded_translation_method_version(result, model) == (
        1 if older else TRANSLATION_METHOD_NOT_USED
    )


def test_a_distribution_records_the_translation_version_and_keeps_its_status() -> None:
    model = _model_json((0.22, 0.0, 0.0), (0.25, 0.0, 0.0))
    result = _current_field_result(
        "distribution",
        _field_rows(_HOME),
        **{TRANSLATION_METHOD_VERSION_KEY: TRANSLATION_METHOD_VERSION},
    )

    assert translation_method_applies(result)
    assert not field_result_uses_older_method(result, None, model)
    assert recorded_translation_method_version(result, model) == TRANSLATION_METHOD_VERSION


def test_the_owner_model_is_the_page_structure_or_the_workspaces_own(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "owners.mufinder", "Owners")
    page = _model_json((0.25, 0.0, 0.0))
    workspace = _model_json((0.22, 0.0, 0.0))
    document.project.magnetic_structure = page
    document.ui_state["magnetic_field_workspaces"] = [
        {"id": "workspace-1", "magnetic_structure": workspace, "form": {}}
    ]

    assert magnetic_field_owner_model(document, MAGNETIC_FIELD_PAGE_OWNER) == page
    assert magnetic_field_owner_model(document, "workspace-1") == workspace
    assert magnetic_field_owner_model(document, "gone") is None
