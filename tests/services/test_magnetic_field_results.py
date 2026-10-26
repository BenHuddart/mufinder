"""Tests for Magnetic Fields results stored as project ``Result`` records."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mufinder.science.magnetic_fields import (
    FieldTermStatistics,
    MagneticFieldDistributionResult,
    MagneticFieldResult,
    regenerate_phase_samples,
)
from mufinder.science.polarisation import calculate_polarisation
from mufinder.services.magnetic_field_results import (
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
)
from mufinder.services.project_service import ProjectService


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
