"""Characterization tests for the 5-layer template precedence in resolve_job.

Pins the current override order (global -> workflow -> structure -> sweep/patch
-> user), the kpoint grid/spacing mutual-exclusivity behavior, and the
conditional atomic_spins / spin_polarized behavior in
``mufinder.workflow.merge.resolve_job``.
"""

from __future__ import annotations

from mufinder.castep_io.types import (
    AtomicPosition,
    KeywordAssignment,
    LatticeCart,
    PositionBlock,
    ValueWithUnit,
)
from mufinder.domain.enums import WorkflowStage
from mufinder.workflow.merge import MERGE_PRECEDENCE_ORDER, resolve_job
from mufinder.workflow.types import (
    CellTemplate,
    InputTemplate,
    JobPatch,
    ParamTemplate,
    StructureSpec,
)


def _structure() -> StructureSpec:
    return StructureSpec(
        lattice_cart=LatticeCart(
            vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Fe", (0.0, 0.0, 0.0)),
                AtomicPosition("Fe", (0.5, 0.5, 0.5)),
            )
        ),
        species=("Fe",),
        atomic_spins=(2.0, -2.0),
        label="Fe",
    )


def _param_value(job, name: str):
    for keyword in job.param_file.keywords:
        if keyword.name == name:
            return keyword.value
    raise AssertionError(f"missing param keyword {name}")


def test_precedence_order_is_stable_and_recorded() -> None:
    assert MERGE_PRECEDENCE_ORDER == (
        "global_defaults",
        "workflow_defaults",
        "structure_defaults",
        "sweep_overrides",
        "user_overrides",
    )
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
    )
    assert job.provenance["merge_precedence"] == list(MERGE_PRECEDENCE_ORDER)


def test_higher_precedence_layer_wins_for_same_param_keyword() -> None:
    # Each layer sets max_scf_cycles to a distinct value; the highest-precedence
    # present layer must win.
    def layer(value: int) -> InputTemplate:
        return InputTemplate(
            param=ParamTemplate(keywords=(KeywordAssignment("max_scf_cycles", value),))
        )

    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        global_defaults=layer(10),
        workflow_defaults=layer(20),
        structure_defaults=layer(30),
        patch=JobPatch(
            label_suffix="s",
            param=ParamTemplate(keywords=(KeywordAssignment("max_scf_cycles", 40),)),
        ),
        user_overrides=layer(50),
    )
    assert _param_value(job, "max_scf_cycles") == 50


def test_patch_sweep_layer_overrides_structure_but_loses_to_user() -> None:
    # Structure < patch(sweep) < user, proven with all three present.
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        structure_defaults=InputTemplate(
            param=ParamTemplate(keywords=(KeywordAssignment("max_scf_cycles", 30),))
        ),
        patch=JobPatch(
            label_suffix="s",
            param=ParamTemplate(keywords=(KeywordAssignment("max_scf_cycles", 40),)),
        ),
        user_overrides=InputTemplate(
            param=ParamTemplate(keywords=(KeywordAssignment("max_scf_cycles", 50),))
        ),
    )
    assert _param_value(job, "max_scf_cycles") == 50


def test_each_adjacent_precedence_boundary_is_pinned() -> None:
    # Distinct value per layer; peel off the top layer and re-resolve, asserting
    # the next-highest wins. This pins every adjacent boundary
    # (global<workflow<structure<patch<user), not just "user beats all".
    def param(value: int) -> InputTemplate:
        return InputTemplate(
            param=ParamTemplate(keywords=(KeywordAssignment("max_scf_cycles", value),))
        )

    layers = {
        "global_defaults": param(10),
        "workflow_defaults": param(20),
        "structure_defaults": param(30),
    }
    patch = JobPatch(
        label_suffix="s", param=ParamTemplate(keywords=(KeywordAssignment("max_scf_cycles", 40),))
    )
    user = param(50)

    # All five present -> user (50).
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        **layers,
        patch=patch,
        user_overrides=user,
    )
    assert _param_value(job, "max_scf_cycles") == 50

    # Drop user -> patch (40).
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        **layers,
        patch=patch,
    )
    assert _param_value(job, "max_scf_cycles") == 40

    # Drop patch -> structure (30).
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        **layers,
    )
    assert _param_value(job, "max_scf_cycles") == 30

    # Drop structure -> workflow (20).
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        global_defaults=layers["global_defaults"],
        workflow_defaults=layers["workflow_defaults"],
    )
    assert _param_value(job, "max_scf_cycles") == 20

    # Drop workflow -> global (10).
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        global_defaults=layers["global_defaults"],
    )
    assert _param_value(job, "max_scf_cycles") == 10


def test_kpoint_grid_in_higher_layer_clears_lower_spacing() -> None:
    # Lower layer sets spacing; higher layer sets grid. Grid must win and
    # spacing must be cleared (mutual exclusivity).
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
        global_defaults=InputTemplate(
            cell=CellTemplate(kpoint_mp_spacing=ValueWithUnit(0.05, "1/ang"))
        ),
        user_overrides=InputTemplate(cell=CellTemplate(kpoint_mp_grid=(4, 4, 4))),
    )
    assert job.cell_file.kpoint_mp_grid == (4, 4, 4)
    assert job.cell_file.kpoint_mp_spacing is None


def test_kpoint_spacing_in_higher_layer_clears_lower_grid() -> None:
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_KPOINT,
        global_defaults=InputTemplate(cell=CellTemplate(kpoint_mp_grid=(4, 4, 4))),
        user_overrides=InputTemplate(
            cell=CellTemplate(kpoint_mp_spacing=ValueWithUnit(0.05, "1/ang"))
        ),
    )
    assert job.cell_file.kpoint_mp_spacing == ValueWithUnit(0.05, "1/ang")
    assert job.cell_file.kpoint_mp_grid is None


def test_atomic_spins_dropped_when_not_spin_polarized() -> None:
    # characterization: resolve_job emits atomic_spins ONLY when the param layer
    # says spin_polarized is truthy; otherwise the structure's spins are dropped
    # from the rendered cell even though the structure carries them.
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
    )
    assert job.cell_file.atomic_spins is None


def test_atomic_spins_emitted_from_structure_when_spin_polarized() -> None:
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        user_overrides=InputTemplate(
            param=ParamTemplate(keywords=(KeywordAssignment("spin_polarized", True),))
        ),
    )
    assert job.cell_file.atomic_spins == (2.0, -2.0)


def test_atomic_spins_template_override_wins_when_spin_polarized() -> None:
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        user_overrides=InputTemplate(
            cell=CellTemplate(atomic_spins=(1.0, 1.0)),
            param=ParamTemplate(keywords=(KeywordAssignment("spin_polarized", True),)),
        ),
    )
    assert job.cell_file.atomic_spins == (1.0, 1.0)


def test_spin_polarized_false_drops_template_atomic_spins() -> None:
    # Even an explicit cell-layer atomic_spins is dropped when spin is off.
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        user_overrides=InputTemplate(
            cell=CellTemplate(atomic_spins=(1.0, 1.0)),
            param=ParamTemplate(keywords=(KeywordAssignment("spin_polarized", False),)),
        ),
    )
    assert job.cell_file.atomic_spins is None


def test_patch_structure_replaces_base_structure_lattice_and_positions() -> None:
    replacement = StructureSpec(
        lattice_cart=LatticeCart(
            vectors=((6.0, 0.0, 0.0), (0.0, 6.0, 0.0), (0.0, 0.0, 6.0)),
            unit="ang",
        ),
        positions_frac=PositionBlock(positions=(AtomicPosition("Si", (0.1, 0.1, 0.1)),)),
        species=("Si",),
        label="replacement",
    )
    job = resolve_job(
        name="j",
        base_structure=_structure(),
        workflow_stage=WorkflowStage.CONVERGENCE_CUTOFF,
        patch=JobPatch(label_suffix="s", structure=replacement),
    )
    assert job.cell_file.lattice_cart == replacement.lattice_cart
    assert job.cell_file.positions_frac == replacement.positions_frac
    assert job.structure is replacement
