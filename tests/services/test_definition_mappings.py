"""Tests for syncing job mappings between definitions and removing definition rows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, Project
from mufinder.services.definition_mappings import (
    convergence_mapping_job_priority,
    definition_workflow_context_matches,
    ensure_job_mapping,
    relaxation_mapping_job_priority,
    remove_cutoff_from_definition,
    remove_definition_if_no_rows,
    remove_grid_keys_from_definition,
    remove_labels_from_definition,
    remove_source_keys_from_definition,
    sweep_definition_context_matches,
    sync_equivalent_mappings,
)
from mufinder.services.project_service import ProjectDocument
from tests.support.jobs import make_job

STAGE = WorkflowStage.PRISTINE_RELAXATION
CALC_TYPE = "castep_pristine_relaxation"
STRUCTURE_ID = uuid4()


@dataclass(frozen=True)
class _Row:
    label: str
    signature: str = "same"


@dataclass(frozen=True)
class _Config:
    rows: tuple[_Row, ...] = ()


def _serialize(config: _Config) -> dict[str, object]:
    return {"rows": [{"label": row.label, "signature": row.signature} for row in config.rows]}


def _deserialize(payload: object) -> _Config | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        return None
    return _Config(rows=tuple(_Row(str(row["label"]), str(row["signature"])) for row in payload["rows"]))


def _row_signatures(config: _Config) -> dict[str, object]:
    return {row.label: row.signature for row in config.rows}


@pytest.fixture
def document(tmp_path: Path) -> ProjectDocument:
    root = tmp_path / "project"
    root.mkdir()
    return ProjectDocument(project_root=root, project=Project())


def _definition(
    document: ProjectDocument,
    rows: tuple[_Row, ...] = (),
    *,
    output_directory: str = "files/relax",
    **parameters: object,
) -> CalculationDefinition:
    definition = CalculationDefinition(
        workflow_stage=STAGE,
        calc_type=CALC_TYPE,
        input_structure_ids=[STRUCTURE_ID],
        parameters={
            "output_directory": output_directory,
            "configuration": _serialize(_Config(rows=rows)),
            **parameters,
        },
    )
    document.calculations[definition.id] = definition
    return definition


def _job_id(document: ProjectDocument, status: JobStatus, *, calc_type: str = CALC_TYPE) -> str:
    job = make_job("job", status=status, calc_type=calc_type, workflow_stage=STAGE)
    document.jobs[job.id] = job
    return str(job.id)


def _sync(document: ProjectDocument, target: CalculationDefinition, **options: object) -> None:
    options.setdefault("priority", relaxation_mapping_job_priority)
    options.setdefault("signatures", _row_signatures)
    sync_equivalent_mappings(
        document,
        list(document.calculations.values()),
        target,
        deserialize=_deserialize,
        **options,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Job priority
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status", "convergence", "relaxation"),
    [
        (JobStatus.SUBMITTING, 50, 50),
        (JobStatus.QUEUED, 50, 50),
        (JobStatus.RUNNING, 50, 50),
        (JobStatus.RETRIEVING, 50, 50),
        (JobStatus.COMPLETED, 40, 40),
        (JobStatus.FAILED, 30, 30),
        (JobStatus.STOPPED, 30, 30),
        (JobStatus.CANCELLED, 30, 30),
        (JobStatus.DEFINED, 10, 35),
    ],
)
def test_priorities_rank_job_statuses(
    document: ProjectDocument, status: JobStatus, convergence: int, relaxation: int
) -> None:
    job_id = _job_id(document, status)

    assert convergence_mapping_job_priority(document, job_id) == convergence
    assert relaxation_mapping_job_priority(document, job_id) == relaxation


def test_priorities_rank_unknown_jobs_lowest(document: ProjectDocument) -> None:
    for raw in ("not-a-uuid", str(uuid4())):
        assert convergence_mapping_job_priority(document, raw) == 0
        assert relaxation_mapping_job_priority(document, raw) == 0


# --------------------------------------------------------------------------- #
# Context matching
# --------------------------------------------------------------------------- #


def test_workflow_context_needs_stage_calc_type_structures_and_output(document: ProjectDocument) -> None:
    target = _definition(document)
    same = _definition(document)
    elsewhere = _definition(document, output_directory="files/other")
    other_type = _definition(document)
    other_type.calc_type = "other"

    assert definition_workflow_context_matches(document, same, target)
    assert not definition_workflow_context_matches(document, elsewhere, target)
    assert not definition_workflow_context_matches(document, other_type, target)


def test_sweep_context_needs_target_structure_output_and_cell(document: ProjectDocument) -> None:
    target = _definition(document, simulation_cell_mode="unit_cell")
    implicit_cell = _definition(document)
    supercell = _definition(document, simulation_cell_mode="supercell")
    no_structure = _definition(document)
    no_structure.input_structure_ids = []

    assert sweep_definition_context_matches(document, implicit_cell, target)
    assert not sweep_definition_context_matches(document, supercell, target)
    assert not sweep_definition_context_matches(document, target, no_structure)


# --------------------------------------------------------------------------- #
# sync_equivalent_mappings — row workflows
# --------------------------------------------------------------------------- #


def test_sync_copies_a_binding_for_an_unchanged_row(document: ProjectDocument) -> None:
    completed = _job_id(document, JobStatus.COMPLETED)
    _definition(document, (_Row("a"),), job_ids_by_label={"a": completed})
    target = _definition(document, (_Row("a"),))

    _sync(document, target)

    assert target.parameters["job_ids_by_label"] == {"a": completed}


def test_sync_skips_a_row_whose_signature_changed(document: ProjectDocument) -> None:
    _definition(document, (_Row("a", "old"),), job_ids_by_label={"a": _job_id(document, JobStatus.COMPLETED)})
    target = _definition(document, (_Row("a", "new"),))

    _sync(document, target)

    assert "job_ids_by_label" not in target.parameters


def test_sync_ignores_definitions_in_another_context(document: ProjectDocument) -> None:
    _definition(
        document,
        (_Row("a"),),
        output_directory="files/other",
        job_ids_by_label={"a": _job_id(document, JobStatus.COMPLETED)},
    )
    target = _definition(document, (_Row("a"),))

    _sync(document, target)

    assert "job_ids_by_label" not in target.parameters


def test_sync_carries_a_key_neither_configuration_lists(document: ProjectDocument) -> None:
    orphan = _job_id(document, JobStatus.COMPLETED)
    _definition(document, (), job_ids_by_label={"gone": orphan})
    target = _definition(document, ())

    _sync(document, target)

    assert target.parameters["job_ids_by_label"] == {"gone": orphan}


def test_sync_keeps_a_target_binding_that_ranks_as_high(document: ProjectDocument) -> None:
    own = _job_id(document, JobStatus.COMPLETED)
    _definition(document, (_Row("a"),), job_ids_by_label={"a": _job_id(document, JobStatus.COMPLETED)})
    target = _definition(document, (_Row("a"),), job_ids_by_label={"a": own})

    _sync(document, target)

    assert target.parameters["job_ids_by_label"] == {"a": own}


def test_sync_replaces_a_lower_ranked_target_binding(document: ProjectDocument) -> None:
    running = _job_id(document, JobStatus.RUNNING)
    _definition(document, (_Row("a"),), job_ids_by_label={"a": running})
    target = _definition(document, (_Row("a"),), job_ids_by_label={"a": _job_id(document, JobStatus.FAILED)})

    _sync(document, target)

    assert target.parameters["job_ids_by_label"] == {"a": running}


def test_sync_priority_decides_between_a_defined_and_a_failed_job(document: ProjectDocument) -> None:
    """The relaxation order prefers a re-staged (defined) job; the convergence order a failed one."""

    defined = _job_id(document, JobStatus.DEFINED)
    failed = _job_id(document, JobStatus.FAILED)
    _definition(document, (_Row("a"),), job_ids_by_label={"a": failed})
    _definition(document, (_Row("a"),), job_ids_by_label={"a": defined})
    relaxation_target = _definition(document, (_Row("a"),))
    convergence_target = _definition(document, (_Row("a"),))

    _sync(document, relaxation_target, priority=relaxation_mapping_job_priority)
    _sync(document, convergence_target, priority=convergence_mapping_job_priority)

    assert relaxation_target.parameters["job_ids_by_label"] == {"a": defined}
    assert convergence_target.parameters["job_ids_by_label"] == {"a": failed}


def test_sync_leaves_the_target_alone_without_a_candidate(document: ProjectDocument) -> None:
    target = _definition(document, (_Row("a"),), job_ids_by_label="broken")

    _sync(document, target)

    assert target.parameters["job_ids_by_label"] == "broken"


def test_sync_replaces_a_broken_target_mapping_when_it_copies(document: ProjectDocument) -> None:
    completed = _job_id(document, JobStatus.COMPLETED)
    _definition(document, (_Row("a"),), job_ids_by_label={"a": completed, "b": 3})
    target = _definition(document, (_Row("a"),), job_ids_by_label="broken")

    _sync(document, target)

    assert target.parameters["job_ids_by_label"] == {"a": completed}


def test_sync_does_nothing_for_an_unreadable_target(document: ProjectDocument) -> None:
    _definition(document, (_Row("a"),), job_ids_by_label={"a": _job_id(document, JobStatus.COMPLETED)})
    target = _definition(document, (_Row("a"),))
    target.parameters["configuration"] = "broken"

    _sync(document, target)

    assert "job_ids_by_label" not in target.parameters


# --------------------------------------------------------------------------- #
# sync_equivalent_mappings — sweep keys
# --------------------------------------------------------------------------- #


def _parse_int(raw: object) -> int | None:
    try:
        return int(str(raw))
    except ValueError:
        return None


def test_sync_respells_parsed_keys_and_needs_both_configurations_to_list_them(
    document: ProjectDocument,
) -> None:
    completed = _job_id(document, JobStatus.COMPLETED)
    unlisted = _job_id(document, JobStatus.COMPLETED)
    _definition(
        document,
        (_Row("1"), _Row("2")),
        job_ids_by_point={"01": completed, "7": unlisted, "bad": completed},
    )
    target = _definition(document, (_Row("1"),))

    _sync(
        document,
        target,
        signatures=lambda config: {int(row.label): row.signature for row in config.rows},
        mapping_name="job_ids_by_point",
        parse_key=_parse_int,
        format_key=lambda key: f"point-{key}",
        unlisted_keys_match=False,
    )

    assert target.parameters["job_ids_by_point"] == {"point-1": completed}


# --------------------------------------------------------------------------- #
# sync_equivalent_mappings — legacy keys, job filter and seed names
# --------------------------------------------------------------------------- #


def _muon_like_sync(document: ProjectDocument, target: CalculationDefinition, **options: object) -> None:
    _sync(
        document,
        target,
        signatures=lambda config: {f"q:{row.label}": row.signature for row in config.rows},
        legacy_keys=lambda config: {row.label: f"q:{row.label}" for row in config.rows},
        seed_mapping_name="job_seed_names_by_label",
        **options,
    )


def test_sync_translates_a_legacy_key_and_brings_its_seed_name(document: ProjectDocument) -> None:
    completed = _job_id(document, JobStatus.COMPLETED)
    _definition(
        document,
        (_Row("a"),),
        job_ids_by_label={"a": completed},
        job_seed_names_by_label={"a": "  seed_a  "},
    )
    target = _definition(document, (_Row("a"),))

    _muon_like_sync(document, target)

    assert target.parameters["job_ids_by_label"] == {"q:a": completed}
    assert target.parameters["job_seed_names_by_label"] == {"q:a": "seed_a"}


def test_sync_reads_a_seed_name_under_the_translated_key(document: ProjectDocument) -> None:
    completed = _job_id(document, JobStatus.COMPLETED)
    _definition(
        document,
        (_Row("a"),),
        job_ids_by_label={"a": completed},
        job_seed_names_by_label={"q:a": "seed_q"},
    )
    target = _definition(document, (_Row("a"),))

    _muon_like_sync(document, target)

    assert target.parameters["job_seed_names_by_label"] == {"q:a": "seed_q"}


def test_sync_creates_the_seed_mapping_but_ignores_a_blank_seed(document: ProjectDocument) -> None:
    completed = _job_id(document, JobStatus.COMPLETED)
    _definition(
        document,
        (_Row("a"),),
        job_ids_by_label={"q:a": completed},
        job_seed_names_by_label={"q:a": "   "},
    )
    target = _definition(document, (_Row("a"),))

    _muon_like_sync(document, target)

    assert target.parameters["job_ids_by_label"] == {"q:a": completed}
    assert target.parameters["job_seed_names_by_label"] == {}


def test_sync_drops_candidates_the_job_filter_rejects(document: ProjectDocument) -> None:
    rejected = _job_id(document, JobStatus.COMPLETED)
    accepted = _job_id(document, JobStatus.FAILED)
    _definition(document, (_Row("a"), _Row("b")), job_ids_by_label={"q:a": rejected, "q:b": accepted})
    target = _definition(document, (_Row("a"), _Row("b")))
    seen: list[tuple[str, str]] = []

    def accept_job(key: str, raw_job_id: str) -> bool:
        seen.append((key, raw_job_id))
        return raw_job_id != rejected

    _muon_like_sync(document, target, accept_job=accept_job)

    assert target.parameters["job_ids_by_label"] == {"q:b": accepted}
    assert ("q:a", rejected) in seen


# --------------------------------------------------------------------------- #
# Row removal
# --------------------------------------------------------------------------- #


def _remove(document: ProjectDocument, definition: CalculationDefinition, labels: set[str], **options):
    options.setdefault("active_key", "active_id")
    return remove_labels_from_definition(
        document,
        definition,
        labels,
        deserialize=_deserialize,
        serialize=_serialize,
        **options,
    )


def test_remove_labels_drops_rows_and_bindings(document: ProjectDocument) -> None:
    definition = _definition(document, (_Row("a"), _Row("b")), job_ids_by_label={"a": "1", "b": "2"})

    assert _remove(document, definition, {"a"}) is False

    assert definition.parameters["configuration"] == _serialize(_Config(rows=(_Row("b"),)))
    assert definition.parameters["job_ids_by_label"] == {"b": "2"}
    assert definition.id in document.calculations


def test_remove_labels_deletes_an_emptied_definition_and_its_active_key(document: ProjectDocument) -> None:
    definition = _definition(document, (_Row("a"),))
    document.ui_state["active_id"] = str(definition.id)

    assert _remove(document, definition, {"a"}) is True

    assert definition.id not in document.calculations
    assert "active_id" not in document.ui_state


def test_remove_labels_uses_custom_matchers_across_mappings(document: ProjectDocument) -> None:
    definition = _definition(
        document,
        (_Row("a"), _Row("b")),
        job_ids_by_label={"q:a": "1", "a": "2", "q:b": "3"},
        job_seed_names_by_label={"q:a": "s"},
    )

    _remove(
        document,
        definition,
        {"a"},
        row_matches=lambda row: row.label == "a",
        key_matches=lambda key: key.endswith("a"),
        mapping_names=("job_ids_by_label", "job_seed_names_by_label"),
    )

    assert definition.parameters["job_ids_by_label"] == {"q:b": "3"}
    assert definition.parameters["job_seed_names_by_label"] == {}


def test_remove_definition_if_no_rows_keeps_definitions_it_cannot_read(document: ProjectDocument) -> None:
    unreadable = _definition(document)
    unreadable.parameters["configuration"] = {"rows": "broken"}
    other_active = _definition(document)
    document.ui_state["active_id"] = "someone-else"

    assert remove_definition_if_no_rows(document, unreadable, "active_id") is False
    assert remove_definition_if_no_rows(document, other_active, "active_id") is True
    assert document.ui_state["active_id"] == "someone-else"


# --------------------------------------------------------------------------- #
# Sweep and source removal
# --------------------------------------------------------------------------- #


def test_remove_grid_keys_keeps_the_definition_and_unreadable_values(document: ProjectDocument) -> None:
    definition = _definition(document, job_ids_by_grid={"2x2x2": "1", "3x3x3": "2"})
    definition.parameters["configuration"] = {"sweep_values": [[2, 2, 2], [3, 3, 3], "bad", [1, "x", 1]]}

    remove_grid_keys_from_definition(
        definition,
        {"2x2x2"},
        mapping_name="job_ids_by_grid",
        mapping_key=lambda grid: "x".join(str(value) for value in grid),
    )

    assert definition.parameters["configuration"]["sweep_values"] == [[3, 3, 3], "bad", [1, "x", 1]]
    assert definition.parameters["job_ids_by_grid"] == {"3x3x3": "2"}
    assert definition.id in document.calculations


def test_remove_last_cutoff_deletes_the_definition(document: ProjectDocument) -> None:
    definition = _definition(document, job_ids_by_cutoff={"300": "1"})
    definition.parameters["configuration"] = {"cutoff_energies": [300.0], "sweep_values": [300.0]}
    document.ui_state["active_cutoff"] = str(definition.id)

    deleted = remove_cutoff_from_definition(
        document,
        definition,
        300.0,
        mapping_name="job_ids_by_cutoff",
        mapping_key=lambda value: f"{float(value):.12g}",
        active_key="active_cutoff",
    )

    assert deleted is True
    assert definition.parameters["job_ids_by_cutoff"] == {}
    assert definition.id not in document.calculations
    assert "active_cutoff" not in document.ui_state


def test_remove_source_keys_drops_sources_settings_and_bindings(document: ProjectDocument) -> None:
    definition = _definition(document, job_ids_by_label={"p1": "1", "p2": "2"})
    definition.parameters["configuration"] = {
        "sources": [{"key": " p1 "}, {"key": "p2"}, "junk"],
        "row_settings": {"p1": {"num_farms": 2}},
    }

    assert remove_source_keys_from_definition(document, definition, {"p1"}) is False
    assert definition.parameters["configuration"] == {"sources": [{"key": "p2"}, "junk"]}
    assert definition.parameters["job_ids_by_label"] == {"p2": "2"}

    definition.parameters["configuration"]["sources"] = [{"key": "p2"}]
    assert remove_source_keys_from_definition(document, definition, {"p2"}) is True
    assert definition.id not in document.calculations


def test_ensure_job_mapping_replaces_a_broken_mapping() -> None:
    definition = CalculationDefinition(parameters={"job_ids_by_label": ["broken"]})

    mapping = ensure_job_mapping(definition)
    mapping["a"] = "1"

    assert definition.parameters["job_ids_by_label"] == {"a": "1"}
