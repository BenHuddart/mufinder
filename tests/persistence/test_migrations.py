"""Tests for the saved-project schema migrations."""

from __future__ import annotations

import pytest

from mufinder.domain.models import PROJECT_SCHEMA_VERSION, Project
from mufinder.persistence.migrations import migrate_project_payloads, stored_schema_version


def _definition(stage: str, configuration: object, definition_id: str = "def-1") -> dict[str, object]:
    return {
        "id": definition_id,
        "workflow_stage": stage,
        "parameters": {"configuration": configuration, "output_directory": "files/out"},
    }


def test_new_projects_are_written_at_the_current_schema_version() -> None:
    assert PROJECT_SCHEMA_VERSION == 3
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
