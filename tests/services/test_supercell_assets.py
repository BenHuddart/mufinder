"""Tests for saved workflow supercell assets and simulation-cell selection."""

from __future__ import annotations

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

    assert diagonal.label == "convergence_cutoff 2x1x1"
    assert general.label == "convergence_cutoff [1 1 0; 0 1 0; 0 0 1]"
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


def test_use_asset_from_another_stage_selects_this_stages_copy(tmp_path: Path) -> None:
    document, host = _document_with_host(tmp_path)
    original = supercell_assets.create_workflow_supercell_asset(
        document, workflow_stage=_STAGE, host_structure_id=host.id, matrix=_DOUBLE_A
    )

    selected = supercell_assets.use_workflow_supercell_asset(
        document, workflow_stage=WorkflowStage.UEP, asset_structure_id=original.id
    )
    resolution = supercell_assets.resolve_workflow_structure(document, WorkflowStage.UEP)

    assert selected.id != original.id
    assert resolution.structure is selected
    assert resolution.selected_asset is not None
    assert resolution.selected_asset.source_asset_structure_id == original.id
    assert resolution.source_workflow_stage == _STAGE


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
