"""Tests for the saved-project schema migrations."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.domain.models import PROJECT_SCHEMA_VERSION, Project
from mufinder.persistence.migrations import UI_STATE_KEY, migrate_project_payloads, stored_schema_version
from mufinder.science.magnetic_fields import DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3


def _definition(stage: str, configuration: object, definition_id: str = "def-1") -> dict[str, object]:
    return {
        "id": definition_id,
        "workflow_stage": stage,
        "parameters": {"configuration": configuration, "output_directory": "files/out"},
    }


def test_new_projects_are_written_at_the_current_schema_version() -> None:
    assert PROJECT_SCHEMA_VERSION == 5
    assert Project().schema_version == PROJECT_SCHEMA_VERSION


def test_schema_1_cutoff_definition_keeps_its_sweep_in_sweep_values_only() -> None:
    manifest = {"schema_version": 1}
    cutoff = _definition(
        "convergence_cutoff",
        {
            "sweep_parameter": "cut_off_energy",
            "sweep_values": [300.0, 400.0],
            "cutoff_energies": [300.0, 400.0],
            "overrides": [{"cutoffs": [400.0], "params": {"max_scf_cycles": 120}}],
        },
    )
    entities = {"calculations": [cutoff]}

    changed = migrate_project_payloads(manifest, entities)

    assert manifest["schema_version"] == PROJECT_SCHEMA_VERSION
    assert changed == {("calculations", "def-1")}
    assert cutoff["parameters"]["configuration"] == {
        "sweep_parameter": "cut_off_energy",
        "sweep_values": [300.0, 400.0],
        "overrides": [{"cutoffs": [400.0], "params": {"max_scf_cycles": 120}}],
    }


def test_schema_1_cutoff_values_win_over_a_disagreeing_sweep_values_list() -> None:
    # Schema 1 read a cutoff sweep from ``cutoff_energies`` first, so those are
    # the values the project was running with.
    legacy_only = _definition("convergence_cutoff", {"cutoff_energies": [300.0, 360.0]}, "legacy")
    disagreeing = _definition(
        "convergence_cutoff",
        {"sweep_values": [999.0], "cutoff_energies": [500.0]},
        "disagreeing",
    )
    entities = {"calculations": [legacy_only, disagreeing]}

    changed = migrate_project_payloads({"schema_version": 1}, entities)

    assert changed == {("calculations", "legacy"), ("calculations", "disagreeing")}
    assert legacy_only["parameters"]["configuration"] == {"sweep_values": [300.0, 360.0]}
    assert disagreeing["parameters"]["configuration"] == {"sweep_values": [500.0]}


def test_schema_1_kpoint_definition_drops_the_empty_cutoff_energies_key() -> None:
    kpoint = _definition(
        "convergence_kpoint",
        {
            "sweep_parameter": "kpoint_mp_grid",
            "sweep_values": [[2, 2, 2], [4, 4, 4]],
            "cutoff_energies": [],
        },
    )

    changed = migrate_project_payloads({"schema_version": 1}, {"calculations": [kpoint]})

    assert changed == {("calculations", "def-1")}
    assert kpoint["parameters"]["configuration"] == {
        "sweep_parameter": "kpoint_mp_grid",
        "sweep_values": [[2, 2, 2], [4, 4, 4]],
    }


def test_migration_leaves_other_payloads_alone() -> None:
    other_stage = _definition("pristine_relaxation", {"cutoff_energies": [300.0]}, "relax")
    already_current = _definition("convergence_cutoff", {"sweep_values": [300.0]}, "current")
    malformed = {"id": "bad", "workflow_stage": "convergence_cutoff", "parameters": {"configuration": []}}
    entities = {"calculations": [other_stage, already_current, malformed], "jobs": [{"id": "job"}]}

    changed = migrate_project_payloads({"schema_version": 1}, entities)

    assert changed == set()
    assert other_stage["parameters"]["configuration"] == {"cutoff_energies": [300.0]}
    assert already_current["parameters"]["configuration"] == {"sweep_values": [300.0]}


def _relaxed_site_payloads() -> dict[str, list[dict[str, object]]]:
    """A schema 2 relaxed site, its run and result, and an unrelaxed site."""

    return {
        "sites": [
            {
                "id": "relaxed",
                "status": "relaxed",
                "fractional_position": [0.1, 0.2, 0.3],
                "cartesian_position": [0.4, 0.8, 1.2],
                "relaxation_run_id": "run-1",
            },
            {
                "id": "staged",
                "status": "staged",
                "fractional_position": [0.5, 0.5, 0.5],
                "cartesian_position": [2.0, 2.0, 2.0],
                "relaxation_run_id": None,
            },
        ],
        "runs": [{"id": "run-1", "result_id": "result-1"}],
        "results": [
            {
                "id": "result-1",
                "calculation_run_id": "run-1",
                "data": {
                    "muon_final_fractional_position_host": [0.15, 0.25, 0.35],
                    "muon_final_fractional_position_supercell": [0.075, 0.125, 0.175],
                },
            }
        ],
    }


def test_schema_2_site_positions_become_initial_positions() -> None:
    entities = _relaxed_site_payloads()

    changed = migrate_project_payloads({"schema_version": 2}, entities)

    assert changed == {("sites", "relaxed"), ("sites", "staged")}
    staged = entities["sites"][1]
    assert staged["initial_fractional_position"] == [0.5, 0.5, 0.5]
    assert staged["initial_cartesian_position"] == [2.0, 2.0, 2.0]
    assert "fractional_position" not in staged
    assert "cartesian_position" not in staged
    assert "relaxed_fractional_position" not in staged


def test_schema_2_relaxed_site_is_backfilled_from_its_run_result() -> None:
    entities = _relaxed_site_payloads()

    migrate_project_payloads({"schema_version": 2}, entities)

    relaxed = entities["sites"][0]
    assert relaxed["initial_fractional_position"] == [0.1, 0.2, 0.3]
    assert relaxed["relaxed_fractional_position"] == [0.15, 0.25, 0.35]


def test_schema_2_backfill_finds_a_result_by_its_run_when_the_run_has_no_result_id() -> None:
    entities = _relaxed_site_payloads()
    entities["runs"][0]["result_id"] = None

    migrate_project_payloads({"schema_version": 2}, entities)

    assert entities["sites"][0]["relaxed_fractional_position"] == [0.15, 0.25, 0.35]


@pytest.mark.parametrize(
    "breakage",
    [
        lambda entities: entities["runs"].clear(),
        lambda entities: entities["results"].clear(),
        lambda entities: entities["results"][0]["data"].pop("muon_final_fractional_position_host"),
        lambda entities: entities["results"][0]["data"].update(muon_final_fractional_position_host=[0.1, 0.2]),
        lambda entities: entities["results"][0]["data"].update(muon_final_fractional_position_host=["a", 0.2, 0.3]),
    ],
    ids=["missing-run", "missing-result", "no-final-position", "short-position", "non-numeric-position"],
)
def test_schema_2_relaxed_site_without_a_usable_result_position_is_not_backfilled(breakage) -> None:
    entities = _relaxed_site_payloads()
    breakage(entities)

    changed = migrate_project_payloads({"schema_version": 2}, entities)

    relaxed = entities["sites"][0]
    assert ("sites", "relaxed") in changed
    assert relaxed["initial_fractional_position"] == [0.1, 0.2, 0.3]
    assert "relaxed_fractional_position" not in relaxed


def test_schema_2_site_already_using_the_new_keys_is_left_alone() -> None:
    site = {"id": "current", "initial_fractional_position": [0.1, 0.2, 0.3], "relaxation_run_id": None}

    changed = migrate_project_payloads({"schema_version": 2}, {"sites": [site]})

    assert changed == set()
    assert site == {"id": "current", "initial_fractional_position": [0.1, 0.2, 0.3], "relaxation_run_id": None}


def test_current_and_newer_projects_are_not_migrated() -> None:
    configuration = {"sweep_values": [300.0], "cutoff_energies": [300.0]}
    for version in (PROJECT_SCHEMA_VERSION, PROJECT_SCHEMA_VERSION + 1):
        manifest = {"schema_version": version}
        definition = _definition("convergence_cutoff", dict(configuration))

        assert migrate_project_payloads(manifest, {"calculations": [definition]}) == set()
        assert manifest["schema_version"] == version
        assert definition["parameters"]["configuration"] == configuration


def test_manifest_without_a_schema_version_is_read_as_schema_1() -> None:
    manifest: dict[str, object] = {}
    definition = _definition("convergence_cutoff", {"cutoff_energies": [300.0]})

    assert stored_schema_version(manifest) == 1
    migrate_project_payloads(manifest, {"calculations": [definition]})

    assert manifest["schema_version"] == PROJECT_SCHEMA_VERSION
    assert definition["parameters"]["configuration"] == {"sweep_values": [300.0]}


def _schema_3_magnetic_ui_state() -> dict[str, object]:
    """Schema 3 Magnetic Fields form state with results kept inline."""

    return {
        "magnetic_field_form": {
            "histogram_bins": 512,
            "magnetic_field_results": {"kind": "commensurate", "results": [{"site_label": "a", "total_T": [0, 0, 1]}]},
            "dipolar_tensor_results": [],
            "nuclear_relaxation_results": {"mode": "zero_field_powder", "sigma_us_inv": 0.4},
        },
        "magnetic_field_workspaces": [
            {
                "id": "ws-1",
                "title": "Helix",
                "form": {
                    "magnetic_field_results": {"kind": "distribution", "results": [{"site_label": "b"}]},
                    "dipolar_tensor_results": [{"site_label": "b", "tensor": [[1, 0, 0], [0, 1, 0], [0, 0, -2]]}],
                },
            },
            {"id": "ws-2", "title": "Empty", "form": {"magnetic_field_results": None}},
            {"title": "No id", "form": {"magnetic_field_results": {"kind": "commensurate", "results": [{}]}}},
        ],
    }


def test_schema_3_magnetic_field_results_become_result_entities() -> None:
    ui_state = _schema_3_magnetic_ui_state()
    entities: dict[str, list[dict[str, object]]] = {}

    changed = migrate_project_payloads({"schema_version": 3}, entities, ui_state)

    results = entities["results"]
    assert changed == {UI_STATE_KEY} | {("results", str(result["id"])) for result in results}
    by_owner_and_type = {(result["data"]["owner"], result["result_type"]): result for result in results}
    assert set(by_owner_and_type) == {
        ("magnetic_fields", "magnetic_field"),
        ("magnetic_fields", "nuclear_relaxation"),
        ("ws-1", "magnetic_field"),
        ("ws-1", "magnetic_dipolar_tensor"),
    }
    page_field = by_owner_and_type[("magnetic_fields", "magnetic_field")]
    assert page_field["calculation_run_id"] is None
    assert page_field["output_files"] == []
    assert page_field["data"] == {
        "owner": "magnetic_fields",
        "kind": "commensurate",
        "results": [{"site_label": "a", "total_T": [0, 0, 1]}],
    }
    assert by_owner_and_type[("magnetic_fields", "nuclear_relaxation")]["data"]["result"] == {
        "mode": "zero_field_powder",
        "sigma_us_inv": 0.4,
    }
    assert by_owner_and_type[("ws-1", "magnetic_dipolar_tensor")]["data"]["results"][0]["site_label"] == "b"
    assert ui_state["magnetic_field_form"] == {"histogram_bins": 512}
    assert ui_state["magnetic_field_workspaces"][0]["form"] == {}
    assert ui_state["magnetic_field_workspaces"][1]["form"] == {}
    # A workspace without an id has no owner, so its form is left alone.
    assert "magnetic_field_results" in ui_state["magnetic_field_workspaces"][2]["form"]


def test_schema_3_project_without_magnetic_results_changes_nothing() -> None:
    ui_state: dict[str, object] = {"magnetic_field_form": {"histogram_bins": 512}, "other": {"x": 1}}
    entities: dict[str, list[dict[str, object]]] = {"results": []}

    assert migrate_project_payloads({"schema_version": 3}, entities, ui_state) == set()
    assert ui_state == {"magnetic_field_form": {"histogram_bins": 512}, "other": {"x": 1}}
    assert entities == {"results": []}


def _tensor_record(record_id: str, entries: list[object], result_type: str = "magnetic_dipolar_tensor") -> dict[str, object]:
    return {
        "id": record_id,
        "calculation_run_id": None,
        "result_type": result_type,
        "data": {"owner": "magnetic_fields", "results": entries},
    }


def test_schema_4_dipolar_tensors_are_converted_from_inverse_cubic_angstrom_to_tesla_per_mu_b() -> None:
    raw = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]]
    tensor = _tensor_record("tensor", [{"site_label": "a", "dipolar_tensor_T_per_muB": raw}, {"site_label": "b"}])
    field = _tensor_record("field", [{"dipolar_tensor_T_per_muB": raw}], result_type="magnetic_field")
    entities = {"results": [tensor, field]}

    changed = migrate_project_payloads({"schema_version": 4}, entities)

    assert changed == {("results", "tensor")}
    np.testing.assert_allclose(
        tensor["data"]["results"][0]["dipolar_tensor_T_per_muB"], DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3 * np.asarray(raw)
    )
    assert tensor["data"]["results"][1] == {"site_label": "b"}
    assert field["data"]["results"][0]["dipolar_tensor_T_per_muB"] == raw


@pytest.mark.parametrize(
    "tensor",
    [None, "1 0 0", [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, "x"]], [[True] * 3] * 3],
    ids=["missing", "string", "two-by-two", "non-numeric", "boolean"],
)
def test_schema_4_migration_leaves_malformed_tensors_alone(tensor) -> None:
    record = _tensor_record("tensor", [{"dipolar_tensor_T_per_muB": tensor}, "not a mapping"])
    entities = {"results": [record]}

    assert migrate_project_payloads({"schema_version": 4}, entities) == set()
    assert record["data"]["results"] == [{"dipolar_tensor_T_per_muB": tensor}, "not a mapping"]


def test_schema_3_inline_dipolar_tensor_is_converted_after_becoming_a_record() -> None:
    ui_state = {
        "magnetic_field_form": {
            "dipolar_tensor_results": [{"site_label": "a", "dipolar_tensor_T_per_muB": [[0, 0, 0], [0, 0, 0], [0, 0, 2]]}]
        }
    }
    entities: dict[str, list[dict[str, object]]] = {}

    migrate_project_payloads({"schema_version": 3}, entities, ui_state)

    (record,) = entities["results"]
    assert record["data"]["results"][0]["dipolar_tensor_T_per_muB"][2][2] == pytest.approx(
        2.0 * DIPOLAR_TENSOR_T_PER_MUB_PER_ANGSTROM_MINUS3
    )
