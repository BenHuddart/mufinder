"""Tests for saved workflow supercell assets and simulation-cell selection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition, Project, Structure
from mufinder.services import supercell_assets
from mufinder.services.project_document import ProjectDocument
from mufinder.services.supercell_assets import WorkflowSupercellSelection
from tests.support.structures import host_structure, silicon_oxide_host

_STAGE = WorkflowStage.CONVERGENCE_CUTOFF
_DOUBLE_A = [[2, 0, 0], [0, 1, 0], [0, 0, 1]]


def _document_with_host(tmp_path: Path) -> tuple[ProjectDocument, Structure]:
    host = silicon_oxide_host()
    document = ProjectDocument(
        project_root=tmp_path, project=Project(name="Assets", host_structure_id=host.id)
    )
    document.structures[host.id] = host
    return document, host


@pytest.mark.parametrize("matrix", [[[1, 0, 0]], [[1, 0, 0], [0, 1], [0, 0, 1]], "2x2x2"])
def test_create_asset_rejects_a_matrix_that_is_not_3x3(tmp_path: Path, matrix: object) -> None:
    document, host = _document_with_host(tmp_path)

    with pytest.raises(ValueError, match="3x3 integer matrix"):
        supercell_assets.create_workflow_supercell_asset(
            document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=matrix
        )


def test_create_asset_rejects_an_unknown_host(tmp_path: Path) -> None:
    document, _host = _document_with_host(tmp_path)

    with pytest.raises(ValueError, match="host_structure_id"):
        supercell_assets.create_workflow_supercell_asset(
            document, workflow_stage=_STAGE, host_structure_id=uuid4(), matrix=_DOUBLE_A
        )


def test_create_asset_labels_diagonal_and_general_matrices(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)

    diagonal = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    general = supercell_assets.create_workflow_supercell_asset(
        document,
        workflow_stage=_STAGE,
        host_structure_id=host.id,
        matrix=[[1, 1, 0], [0, 1, 0], [0, 0, 1]],
    )

    assert diagonal.label == "2x1x1 supercell"
    assert general.label == "[1 1 0; 0 1 0; 0 0 1] supercell"
    assert len(diagonal.species) == 2 * len(host.species)
    assert diagonal.parent_structure_id == host.id


def test_create_asset_reuses_a_matching_asset_unless_told_not_to(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)

    first = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    again = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    fresh = supercell_assets.create_workflow_supercell_asset(
        document,
        workflow_stage=_STAGE,
        host_structure_id=host.id,
        matrix=_DOUBLE_A,
        reuse_existing=False,
    )

    assert again is first
    assert fresh.id != first.id


@pytest.mark.parametrize(
    "raw",
    [
        "bad-state",
        {"convergence_cutoff": "bad"},
        {"convergence_cutoff": {"mode": "saved_supercell", "structure_id": "not-a-uuid"}},
        {"convergence_cutoff": {"mode": "something_else", "structure_id": str(uuid4())}},
    ],
)
def test_malformed_selection_state_reads_as_unit_cell(tmp_path: Path, raw: object) -> None:
    document, _host = _document_with_host(tmp_path)
    document.ui_state["workflow_supercells"] = raw

    selection = supercell_assets.get_workflow_supercell_selection(document, _STAGE)

    assert selection == WorkflowSupercellSelection()
    assert isinstance(document.ui_state["workflow_supercells"], dict)


def test_supercell_asset_ignores_invalid_metadata() -> None:
    structure = silicon_oxide_host(
        parent_structure_id=uuid4(),
        properties={"supercell_asset": {"matrix": "bad", "owner_workflow_stage": "bad"}},
    )
    unknown_stage = silicon_oxide_host(
        properties={
            "supercell_asset": {"matrix": _DOUBLE_A, "owner_workflow_stage": "not_a_stage"}
        },
    )

    assert supercell_assets.supercell_asset(structure) is None
    assert supercell_assets.supercell_asset(unknown_stage) is None
    assert supercell_assets.supercell_asset(silicon_oxide_host()) is None


def test_list_assets_needs_a_host_and_skips_other_hosts(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    asset = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )

    assert supercell_assets.list_workflow_supercell_assets(document, host_structure_id=None) == []
    assert (
        supercell_assets.list_workflow_supercell_assets(document, host_structure_id=uuid4()) == []
    )
    listed = supercell_assets.list_workflow_supercell_assets(document, host_structure_id=host.id)
    assert [item.structure_id for item in listed] == [asset.id]


def test_use_asset_from_another_stage_keeps_one_library_entry(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    original = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )

    selected = supercell_assets.use_workflow_supercell_asset(
        document, workflow_stage=WorkflowStage.UEP, asset_structure_id=original.id
    )
    resolution = supercell_assets.resolve_workflow_structure(document, WorkflowStage.UEP)

    assert selected is original
    assert resolution.structure is original
    assert resolution.source_workflow_stage == _STAGE
    assert [
        asset.structure_id
        for asset in supercell_assets.list_workflow_supercell_assets(
            document, host_structure_id=host.id
        )
    ] == [original.id]


def test_create_asset_reuses_a_matching_asset_from_another_stage(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    original = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=None, host_structure_id=host.id, matrix=_DOUBLE_A
    )

    again = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=WorkflowStage.UEP, host_structure_id=host.id, matrix=_DOUBLE_A
    )

    assert again is original


def test_asset_saved_with_no_workflow_has_no_owner_and_can_be_selected(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)

    saved = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=None, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    supercell_assets.use_workflow_supercell_asset(
        document, workflow_stage=_STAGE, asset_structure_id=saved.id
    )
    resolution = supercell_assets.resolve_workflow_structure(document, _STAGE)

    assert saved.label == "2x1x1 supercell"
    assert saved.properties["supercell_asset"]["owner_workflow_stage"] is None
    assert resolution.structure is saved
    assert resolution.selected_asset is not None
    assert resolution.selected_asset.owner_workflow_stage is None
    assert resolution.source_workflow_stage is None


def test_asset_without_an_owner_survives_a_matrix_update_and_a_spin_repair(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    saved = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=None, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    double_b = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]

    supercell_assets.update_workflow_supercell_asset_matrix(
        document, asset_structure_id=saved.id, matrix=double_b
    )
    host.properties["atomic_spins"] = [1.0] * len(host.species)
    assert supercell_assets.repair_supercell_asset_if_needed(document, structure_id=saved.id)

    asset = supercell_assets.supercell_asset(saved)
    assert asset is not None
    assert asset.owner_workflow_stage is None
    assert asset.matrix == double_b


def test_legacy_clone_keeps_resolving_to_its_source_workflow(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    source = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    clone = supercell_assets.create_workflow_supercell_asset(
        document,
        workflow_stage=WorkflowStage.UEP,
        host_structure_id=host.id,
        matrix=_DOUBLE_A,
        label="uep 2x1x1",
        source_asset_structure_id=source.id,
        reuse_existing=False,
    )

    selected = supercell_assets.use_workflow_supercell_asset(
        document, workflow_stage=WorkflowStage.UEP, asset_structure_id=clone.id
    )
    resolution = supercell_assets.resolve_workflow_structure(document, WorkflowStage.UEP)

    assert selected is clone
    assert resolution.source_workflow_stage == _STAGE


def test_assets_are_listed_by_label_whichever_stage_saved_them(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    for label, stage, matrix in (
        ("Zinc", WorkflowStage.CONVERGENCE_CUTOFF, _DOUBLE_A),
        ("Alpha", WorkflowStage.UEP, [[1, 0, 0], [0, 2, 0], [0, 0, 1]]),
        ("beta", None, [[1, 0, 0], [0, 1, 0], [0, 0, 2]]),
    ):
        supercell_assets.create_workflow_supercell_asset(
            document, workflow_stage=stage, host_structure_id=host.id, matrix=matrix, label=label
        )

    assets = supercell_assets.list_workflow_supercell_assets(document, host_structure_id=host.id)

    assert [asset.label for asset in assets] == ["Alpha", "beta", "Zinc"]


def _dated_asset(document, host, matrix, day: int) -> Structure:
    structure = supercell_assets.create_workflow_supercell_asset(
        document,
        workflow_stage=None,
        host_structure_id=host.id,
        matrix=matrix,
        reuse_existing=False,
    )
    structure.properties["supercell_asset"]["saved_at"] = datetime(
        2026, 1, day, tzinfo=UTC
    ).isoformat()
    return structure


def test_latest_asset_is_the_newest_and_can_be_narrowed_to_a_matrix(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    double_b = [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
    newest_a = _dated_asset(document, host, _DOUBLE_A, day=3)
    _dated_asset(document, host, _DOUBLE_A, day=1)
    newest_b = _dated_asset(document, host, double_b, day=2)

    latest = supercell_assets.latest_workflow_supercell_asset(document, host_structure_id=host.id)
    of_b = supercell_assets.latest_workflow_supercell_asset(
        document, host_structure_id=host.id, matrix=double_b
    )
    of_a = supercell_assets.latest_workflow_supercell_asset(
        document, host_structure_id=host.id, matrix=_DOUBLE_A
    )

    assert latest is not None and latest.structure_id == newest_a.id
    assert of_b is not None and of_b.structure_id == newest_b.id
    assert of_a is not None and of_a.structure_id == newest_a.id
    assert (
        supercell_assets.latest_workflow_supercell_asset(
            document, host_structure_id=host.id, matrix=[[5, 0, 0], [0, 5, 0], [0, 0, 5]]
        )
        is None
    )
    assert (
        supercell_assets.latest_workflow_supercell_asset(document, host_structure_id=None) is None
    )


def test_latest_asset_prefers_any_recorded_save_time_over_none(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    dated = _dated_asset(document, host, _DOUBLE_A, day=1)
    undated = _dated_asset(document, host, [[1, 0, 0], [0, 2, 0], [0, 0, 1]], day=2)
    del undated.properties["supercell_asset"]["saved_at"]
    garbled = _dated_asset(document, host, [[1, 0, 0], [0, 1, 0], [0, 0, 2]], day=3)
    garbled.properties["supercell_asset"]["saved_at"] = "not a date"

    latest = supercell_assets.latest_workflow_supercell_asset(document, host_structure_id=host.id)

    assert latest is not None and latest.structure_id == dated.id


def test_latest_asset_ignores_a_save_time_without_a_timezone(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    dated = _dated_asset(document, host, _DOUBLE_A, day=1)
    naive = _dated_asset(document, host, [[1, 0, 0], [0, 2, 0], [0, 0, 1]], day=2)
    naive.properties["supercell_asset"]["saved_at"] = "2026-06-01T12:00:00"

    latest = supercell_assets.latest_workflow_supercell_asset(document, host_structure_id=host.id)

    assert latest is not None and latest.structure_id == dated.id


def test_copy_of_a_cell_without_an_owner_keeps_its_own_source_workflow(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    source = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=None, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    copy = supercell_assets.create_workflow_supercell_asset(
        document,
        workflow_stage=_STAGE,
        host_structure_id=host.id,
        matrix=[[3, 0, 0], [0, 1, 0], [0, 0, 1]],
        source_asset_structure_id=source.id,
        reuse_existing=False,
    )
    supercell_assets.use_workflow_supercell_asset(
        document, workflow_stage=_STAGE, asset_structure_id=copy.id
    )

    resolution = supercell_assets.resolve_workflow_structure(document, _STAGE)

    assert resolution.source_workflow_stage == _STAGE


def test_saving_an_asset_again_makes_it_the_latest(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    first = _dated_asset(document, host, _DOUBLE_A, day=1)
    second = _dated_asset(document, host, [[1, 0, 0], [0, 2, 0], [0, 0, 1]], day=2)

    def latest() -> object:
        found = supercell_assets.latest_workflow_supercell_asset(
            document, host_structure_id=host.id
        )
        return found.structure_id

    assert latest() == second.id
    supercell_assets.rename_workflow_supercell_asset(
        document, asset_structure_id=first.id, label="Renamed"
    )
    assert latest() == first.id
    supercell_assets.update_workflow_supercell_asset_matrix(
        document, asset_structure_id=second.id, matrix=[[1, 0, 0], [0, 3, 0], [0, 0, 1]]
    )
    assert latest() == second.id


def test_spin_repair_keeps_the_asset_metadata(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    saved = _dated_asset(document, host, _DOUBLE_A, day=1)
    before = dict(saved.properties["supercell_asset"])
    host.properties["atomic_spins"] = [1.0] * len(host.species)

    assert supercell_assets.repair_supercell_asset_if_needed(document, structure_id=saved.id)

    assert saved.properties["supercell_asset"] == before


def test_workflows_selecting_an_asset_are_listed_in_workflow_order(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    shared = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=None, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    other = supercell_assets.create_workflow_supercell_asset(
        document,
        workflow_stage=None,
        host_structure_id=host.id,
        matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 1]],
    )
    for stage in (WorkflowStage.CONVERGENCE_KPOINT, _STAGE):
        supercell_assets.use_workflow_supercell_asset(
            document, workflow_stage=stage, asset_structure_id=shared.id
        )
    supercell_assets.use_workflow_supercell_asset(
        document, workflow_stage=WorkflowStage.UEP, asset_structure_id=other.id
    )

    assert supercell_assets.workflows_selecting_supercell_asset(
        document, asset_structure_id=shared.id
    ) == [_STAGE, WorkflowStage.CONVERGENCE_KPOINT]


def test_distinct_label_adds_copy_only_when_the_name_is_taken(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=None, host_structure_id=host.id, matrix=_DOUBLE_A, label="Cell"
    )

    def distinct(label: str) -> str:
        return supercell_assets.distinct_supercell_asset_label(
            document, host_structure_id=host.id, label=label
        )

    assert distinct("Other") == "Other"
    assert distinct("cell") == "cell copy"
    assert distinct("Cell") == "Cell copy"
    supercell_assets.create_workflow_supercell_asset(
        document,
        workflow_stage=None,
        host_structure_id=host.id,
        matrix=[[1, 0, 0], [0, 2, 0], [0, 0, 1]],
        label="Cell copy",
    )
    assert distinct("Cell") == "Cell copy 2"


def test_editor_seed_is_the_identity_for_a_new_project(tmp_path: Path) -> None:
    document, _host = _document_with_host(tmp_path)

    assert supercell_assets.supercell_editor_seed_matrix(document) == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_editor_seed_reads_the_legacy_project_matrix_without_a_saved_cell(tmp_path: Path) -> None:
    document, _host = _document_with_host(tmp_path)
    document.project.supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]

    assert supercell_assets.supercell_editor_seed_matrix(document) == [
        [3, 0, 0],
        [0, 3, 0],
        [0, 0, 1],
    ]


def test_editor_seed_prefers_the_newest_saved_cell_over_the_legacy_matrix(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    document.project.supercell_matrix = [[3, 0, 0], [0, 3, 0], [0, 0, 1]]
    _dated_asset(document, host, _DOUBLE_A, day=2)
    _dated_asset(document, host, [[1, 0, 0], [0, 2, 0], [0, 0, 1]], day=1)

    assert supercell_assets.supercell_editor_seed_matrix(document) == _DOUBLE_A


def test_editor_seed_ignores_a_malformed_legacy_matrix(tmp_path: Path) -> None:
    document, _host = _document_with_host(tmp_path)
    document.project.supercell_matrix = [[1, 0], [0, 1]]

    assert supercell_assets.supercell_editor_seed_matrix(document) == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_frozen_asset_cannot_be_regenerated_or_deleted(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    asset = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    definition = CalculationDefinition(
        workflow_stage=_STAGE, calc_type="castep_cutoff", input_structure_ids=[asset.id]
    )
    document.calculations[definition.id] = definition

    assert supercell_assets.supercell_asset_is_frozen(document, asset_structure_id=asset.id)
    assert supercell_assets.supercell_asset_references(document, asset_structure_id=asset.id) == [
        "convergence_cutoff calculation 'castep_cutoff'"
    ]
    with pytest.raises(ValueError, match="save a new copy"):
        supercell_assets.update_workflow_supercell_asset_matrix(
            document, asset_structure_id=asset.id, matrix=[[3, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
    with pytest.raises(ValueError, match="cannot be deleted"):
        supercell_assets.delete_workflow_supercell_asset(document, asset_structure_id=asset.id)


def test_delete_asset_resets_selections_that_pointed_at_it(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    asset = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )
    supercell_assets.use_workflow_supercell_asset(
        document, workflow_stage=_STAGE, asset_structure_id=asset.id
    )

    supercell_assets.delete_workflow_supercell_asset(document, asset_structure_id=asset.id)

    assert asset.id not in document.structures
    assert (
        supercell_assets.get_workflow_supercell_selection(document, _STAGE)
        == WorkflowSupercellSelection()
    )


def test_structure_atomic_spins_prefers_moments_over_stale_zero_spins() -> None:
    stale = host_structure(
        species=("Fe",),
        positions=((0.0, 0.0, 0.0),),
        properties={
            "atomic_spins": [0.0],
            "magnetic_structure": {"moments": [{"vector": [0.0, 0.0, 1.75]}]},
        },
    )

    assert supercell_assets.structure_atomic_spins(stale) == pytest.approx([1.75])


def test_structure_atomic_spins_ignores_malformed_asset_metadata() -> None:
    structure = host_structure(
        species=("Fe",),
        positions=((0.0, 0.0, 0.0),),
        properties={
            "atomic_spins": [0.0],
            "magnetic_structure": {"moments": [{"vector": [0.0, 0.0, 1.75]}]},
            "supercell_asset": {"matrix": "not a matrix"},
        },
    )

    assert supercell_assets.supercell_asset(structure) is None
    assert supercell_assets.structure_atomic_spins(structure) == pytest.approx([1.75])


def test_structure_atomic_spins_keeps_a_valid_assets_stored_spins() -> None:
    structure = host_structure(
        species=("Fe",),
        positions=((0.0, 0.0, 0.0),),
        properties={
            "atomic_spins": [0.0],
            "magnetic_structure": {"moments": [{"vector": [0.0, 0.0, 1.75]}]},
            "supercell_asset": {
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "owner_workflow_stage": _STAGE.value,
            },
        },
    )

    assert supercell_assets.supercell_asset(structure) is not None
    assert supercell_assets.structure_atomic_spins(structure) == pytest.approx([0.0])
