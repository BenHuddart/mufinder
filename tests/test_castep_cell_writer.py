"""Tests for CASTEP ``.cell`` serialization."""

from __future__ import annotations

from pathlib import Path

from mufinder.castep_io import (
    AtomicPosition,
    CellFile,
    KPoint,
    LatticeABC,
    LatticeCart,
    PositionBlock,
    SpeciesPotential,
    ValueWithUnit,
    write_cell,
)


def test_write_cell_serializes_basic_lattice_positions_and_scalars() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=((5.43, 0.0, 0.0), (0.0, 5.43, 0.0), (0.0, 0.0, 5.43)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.25, 0.25, 0.25)),
            )
        ),
        species_pot=(SpeciesPotential("Si", "Si_00.recpot"),),
        kpoint_mp_grid=(4, 4, 4),
        symmetry_generate=True,
    )

    rendered = write_cell(cell_file)

    assert "%BLOCK LATTICE_CART" in rendered
    assert "%BLOCK POSITIONS_FRAC" in rendered
    assert "KPOINT_MP_GRID 4 4 4" in rendered
    assert "SYMMETRY_GENERATE" in rendered


def test_write_cell_supports_alternative_blocks() -> None:
    cell_file = CellFile(
        lattice_abc=LatticeABC(lengths=(5.43, 5.43, 5.43), angles=(90.0, 90.0, 90.0), unit="ang"),
        positions_abs=PositionBlock(
            positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),),
            unit="ang",
        ),
        kpoint_mp_spacing=ValueWithUnit(0.05, "1/ang"),
    )

    rendered = write_cell(cell_file)

    assert "%BLOCK LATTICE_ABC" in rendered
    assert "%BLOCK POSITIONS_ABS" in rendered
    assert "KPOINT_MP_SPACING 0.05 1/ang" in rendered


def test_write_cell_matches_golden_file() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=((5.43, 0.0, 0.0), (0.0, 5.43, 0.0), (0.0, 0.0, 5.43)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.25, 0.25, 0.25)),
            )
        ),
        species_pot=(SpeciesPotential("Si", "Si_00.recpot"),),
        kpoint_mp_grid=(4, 4, 4),
        symmetry_generate=True,
    )

    rendered = write_cell(cell_file)
    golden = Path("tests/golden/si_geom_opt.cell").read_text(encoding="utf-8")
    assert rendered == golden


def test_write_cell_supports_explicit_kpoints_list() -> None:
    cell_file = CellFile(
        lattice_abc=LatticeABC(lengths=(5.43, 5.43, 5.43), angles=(90.0, 90.0, 90.0), unit="ang"),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Si", (0.0, 0.0, 0.0)),
                AtomicPosition("Si", (0.25, 0.25, 0.25)),
            )
        ),
        kpoints_list=(
            KPoint((0.25, 0.25, 0.25), 0.25),
            KPoint((0.25, -0.25, 0.25), 0.25),
            KPoint((0.25, 0.25, -0.25), 0.25),
            KPoint((0.25, -0.25, -0.25), 0.25),
        ),
    )

    rendered = write_cell(cell_file)

    assert "%BLOCK KPOINTS_LIST" in rendered
    assert rendered.count("0.25") >= 8


def test_write_cell_serializes_inline_spins_in_positions_frac() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=((5.43, 0.0, 0.0), (0.0, 5.43, 0.0), (0.0, 0.0, 5.43)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Fe", (0.0, 0.0, 0.0)),
                AtomicPosition("Fe", (0.5, 0.5, 0.5)),
            )
        ),
        atomic_spins=(2.0, -2.0),
    )

    rendered = write_cell(cell_file)

    assert "Fe 0 0 0 SPIN=2" in rendered
    assert "Fe 0.5 0.5 0.5 SPIN=-2" in rendered


def test_write_cell_omits_zero_spins_and_matches_golden() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Fe", (0.0, 0.0, 0.0)),
                AtomicPosition("Fe", (0.5, 0.5, 0.5)),
            )
        ),
        atomic_spins=(2.0, 0.0),
    )

    rendered = write_cell(cell_file)
    golden = Path("tests/golden/fe_spin.cell").read_text(encoding="utf-8")
    assert rendered == golden
