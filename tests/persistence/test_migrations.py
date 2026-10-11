"""Tests for the saved-project schema migrations."""

from __future__ import annotations

from mufinder.domain.models import PROJECT_SCHEMA_VERSION, Project
from mufinder.persistence.migrations import migrate_project_payloads, stored_schema_version


def _definition(stage: str, configuration: object, definition_id: str = "def-1") -> dict[str, object]:
    return {
        "id": definition_id,
        "workflow_stage": stage,
        "parameters": {"configuration": configuration, "output_directory": "files/out"},
    }


def test_new_projects_are_written_at_the_current_schema_version() -> None:
    assert PROJECT_SCHEMA_VERSION == 2
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

    assert manifest["schema_version"] == 2
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
