"""Validation tests for CASTEP input models."""

from __future__ import annotations

import pytest

from mufinder.castep_io import (
    AtomicPosition,
    CellFile,
    ConflictingKeywordError,
    KPoint,
    KeywordAssignment,
    LatticeABC,
    LatticeCart,
    MissingRequiredKeywordError,
    ParamFile,
    PositionBlock,
    ValueWithUnit,
    validate_cell_file,
    validate_param_file,
)


def test_validate_cell_rejects_conflicting_lattice_blocks() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        lattice_abc=LatticeABC(lengths=(1.0, 1.0, 1.0), angles=(90.0, 90.0, 90.0)),
        positions_frac=PositionBlock(positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
    )

    with pytest.raises(ConflictingKeywordError, match="LATTICE_CART cannot be used together with LATTICE_ABC"):
        validate_cell_file(cell_file)


def test_validate_cell_requires_positions() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    )

    with pytest.raises(MissingRequiredKeywordError, match="requires either POSITIONS_FRAC or POSITIONS_ABS"):
        validate_cell_file(cell_file)


def test_validate_param_rejects_conflicting_spin_spellings() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("task", "SINGLEPOINT"),
            KeywordAssignment("spin_polarized", True),
            KeywordAssignment("spin_polarised", False),
        )
    )

    with pytest.raises(ConflictingKeywordError, match="SPIN_POLARIZED cannot be used together with SPIN_POLARISED"):
        validate_param_file(param_file)


def test_validate_param_rejects_task_mismatch() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("task", "SINGLEPOINT"),
            KeywordAssignment("md_delta_t", ValueWithUnit(1.0, "fs")),
        )
    )

    with pytest.raises(ValueError, match="TASK=SINGLEPOINT"):
        validate_param_file(param_file)


def test_validate_param_rejects_basis_precision_and_cutoff_combination() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("task", "SINGLEPOINT"),
            KeywordAssignment("basis_precision", "FINE"),
            KeywordAssignment("cut_off_energy", ValueWithUnit(340.0, "eV")),
        )
    )

    with pytest.raises(ConflictingKeywordError, match="BASIS_PRECISION cannot be used together with CUT_OFF_ENERGY"):
        validate_param_file(param_file)


def test_validate_kpoints_list_requires_weights_to_sum_to_one() -> None:
    cell_file = CellFile(
        lattice_abc=LatticeABC(lengths=(5.0, 5.0, 5.0), angles=(90.0, 90.0, 90.0)),
        positions_frac=PositionBlock(positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
        kpoints_list=(
            KPoint((0.0, 0.0, 0.0), 0.4),
            KPoint((0.5, 0.5, 0.5), 0.4),
        ),
    )

    with pytest.raises(ValueError, match="weights must sum to 1.0"):
        validate_cell_file(cell_file)


def test_validate_param_rejects_elec_energy_tol_per_atom_units() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("task", "SINGLEPOINT"),
            KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-6, "eV/atom")),
        )
    )

    with pytest.raises(ValueError, match="not per-atom units"):
        validate_param_file(param_file)


def test_validate_cell_rejects_atomic_spin_count_mismatch() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0))),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Fe", (0.0, 0.0, 0.0)),
                AtomicPosition("Fe", (0.5, 0.5, 0.5)),
            )
        ),
        atomic_spins=(2.0,),
    )

    with pytest.raises(ValueError, match="Atomic spin count must match POSITIONS_FRAC row count"):
        validate_cell_file(cell_file)


def test_validate_cell_rejects_atomic_spins_without_positions_frac() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0))),
        positions_abs=PositionBlock(
            positions=(AtomicPosition("Fe", (0.0, 0.0, 0.0)),),
            unit="ang",
        ),
        atomic_spins=(2.0,),
    )

    with pytest.raises(ValueError, match="Per-atom spins require POSITIONS_FRAC"):
        validate_cell_file(cell_file)
