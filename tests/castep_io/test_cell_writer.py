"""Tests for CASTEP ``.cell`` serialization."""

from __future__ import annotations

from pathlib import Path

from mufinder.castep_io import (
    AtomicPosition,
    BlockSection,
    CellFile,
    KeywordAssignment,
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
    assert "SYMMETRY_GENERATE : TRUE" not in rendered


def test_write_cell_serializes_bool_scalars_explicitly() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=((5.43, 0.0, 0.0), (0.0, 5.43, 0.0), (0.0, 0.0, 5.43)),
            unit="ang",
        ),
        positions_frac=PositionBlock(positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
        scalars=(
            # FIX_ALL_CELL is a valued boolean keyword, unlike presence-only
            # flags such as SYMMETRY_GENERATE.
            KeywordAssignment("fix_all_cell", True),
        ),
    )

    rendered = write_cell(cell_file)

    assert "FIX_ALL_CELL : TRUE" in rendered
    assert "FIX_CELL" not in rendered


def test_write_cell_keeps_presence_flags_bare_when_extra_scalar() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=((5.43, 0.0, 0.0), (0.0, 5.43, 0.0), (0.0, 0.0, 5.43)),
            unit="ang",
        ),
        positions_frac=PositionBlock(positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
        scalars=(KeywordAssignment("symmetry_generate", True),),
    )

    rendered = write_cell(cell_file)

    assert "SYMMETRY_GENERATE" in rendered
    assert "SYMMETRY_GENERATE : TRUE" not in rendered


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


def test_write_cell_serializes_hubbard_u_block() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(
                AtomicPosition("Sm", (0.0, 0.0, 0.0)),
                AtomicPosition("Ni", (0.5, 0.5, 0.5)),
            )
        ),
        blocks=(
            BlockSection(
                "hubbard_u",
                (
                    ("Sm", 1, "f:", 6.1),
                    ("Ni", "d:", 2.4),
                ),
                unit="eV",
            ),
        ),
    )

    rendered = write_cell(cell_file)

    assert "%BLOCK HUBBARD_U" in rendered
    assert "eV" in rendered
    assert "Sm 1 f: 6.1" in rendered
    assert "Ni d: 2.4" in rendered


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
