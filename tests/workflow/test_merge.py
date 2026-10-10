"""Tests for workflow template merge and precedence behavior."""

from __future__ import annotations

import pytest

from mufinder.castep_io.types import AtomicPosition, KPoint, KeywordAssignment, LatticeABC, PositionBlock, ValueWithUnit
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.workflow.bridge import structure_to_spec
from mufinder.workflow.merge import merge_with_precedence, resolve_job
from mufinder.workflow.types import CellTemplate, InputTemplate, ParamTemplate


def _keyword_value(keywords: tuple[KeywordAssignment, ...], name: str) -> object:
    for assignment in keywords:
        if assignment.name == name:
            return assignment.value
    raise AssertionError(f"Keyword {name} not found")


def test_merge_with_precedence_prefers_higher_layers() -> None:
    merged = merge_with_precedence(
        global_defaults=InputTemplate(
            cell=CellTemplate(kpoint_mp_spacing=ValueWithUnit(0.06, "1/ang")),
            param=ParamTemplate(
                keywords=(
                    KeywordAssignment("task", "SINGLEPOINT"),
                    KeywordAssignment("cut_off_energy", ValueWithUnit(250.0, "eV")),
                )
            ),
        ),
        workflow_defaults=InputTemplate(
            cell=CellTemplate(kpoint_mp_grid=(2, 2, 2)),
            param=ParamTemplate(
                keywords=(
                    KeywordAssignment("cut_off_energy", ValueWithUnit(300.0, "eV")),
                )
            ),
        ),
        sweep_overrides=InputTemplate(
            param=ParamTemplate(
                keywords=(
                    KeywordAssignment("cut_off_energy", ValueWithUnit(350.0, "eV")),
                )
            )
        ),
        user_overrides=InputTemplate(
            cell=CellTemplate(
                kpoints_list=(
                    KPoint((0.0, 0.0, 0.0), 1.0),
                )
            ),
            param=ParamTemplate(
                keywords=(
                    KeywordAssignment("cut_off_energy", ValueWithUnit(400.0, "eV")),
                )
            ),
        ),
    )

    assert merged.cell is not None
    assert merged.cell.kpoints_list == (KPoint((0.0, 0.0, 0.0), 1.0),)
    assert merged.cell.kpoint_mp_grid is None
    assert merged.cell.kpoint_mp_spacing is None

    assert merged.param is not None
    value = _keyword_value(merged.param.keywords, "cut_off_energy")
    assert isinstance(value, ValueWithUnit)
    assert value.value == 400.0


def test_merge_with_precedence_keeps_explicit_kpoint_grid_from_same_layer() -> None:
    merged = merge_with_precedence(
        user_overrides=InputTemplate(
            cell=CellTemplate(
                kpoint_mp_grid=(3, 3, 3),
            )
        ),
    )

    assert merged.cell is not None
    assert merged.cell.kpoint_mp_grid == (3, 3, 3)
    assert merged.cell.kpoints_list == ()
    assert merged.cell.kpoint_mp_spacing is None


def test_merge_with_precedence_handles_structural_block_overrides() -> None:
    merged = merge_with_precedence(
        global_defaults=InputTemplate(
            cell=CellTemplate(
                lattice_abc=LatticeABC(
                    lengths=(5.43, 5.43, 5.43),
                    angles=(90.0, 90.0, 90.0),
                    unit="ang",
                )
            )
        ),
        user_overrides=InputTemplate(
            cell=CellTemplate(
                positions_abs=PositionBlock(
                    positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),),
                    unit="ang",
                )
            )
        ),
    )

    assert merged.cell is not None
    assert merged.cell.lattice_abc is not None
    assert merged.cell.lattice_cart is None
    assert merged.cell.positions_abs is not None
    assert merged.cell.positions_frac is None


def test_resolve_job_includes_atomic_spins_only_for_spin_polarized() -> None:
    structure = Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe", "Fe"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        properties={"atomic_spins": [2.0, -2.0]},
    )
    spec = structure_to_spec(structure)

    spin_job = resolve_job(
        name="spin_on",
        base_structure=spec,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        workflow_defaults=InputTemplate(
            param=ParamTemplate(
                keywords=(
                    KeywordAssignment("task", "SINGLEPOINT"),
                    KeywordAssignment("spin_polarized", True),
                )
            )
        ),
    )
    non_spin_job = resolve_job(
        name="spin_off",
        base_structure=spec,
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        workflow_defaults=InputTemplate(
            param=ParamTemplate(
                keywords=(KeywordAssignment("task", "SINGLEPOINT"),)
            )
        ),
    )

    assert spin_job.cell_file.atomic_spins == pytest.approx((2.0, -2.0))
    assert non_spin_job.cell_file.atomic_spins is None
