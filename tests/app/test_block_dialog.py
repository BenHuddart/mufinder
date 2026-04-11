"""Tests for structured CASTEP block editors."""

from __future__ import annotations

import pytest

from mufinder.app.block_dialog import BlockParameterDialog, summarize_block_value
from mufinder.castep_io.schema import get_keyword_spec
from mufinder.castep_io.types import BlockSection


@pytest.mark.parametrize(
    ("name", "initial_value", "summary"),
    [
        (
            "lattice_cart",
            BlockSection(
                name="lattice_cart",
                rows=((5.43, 0.0, 0.0), (0.0, 5.43, 0.0), (0.0, 0.0, 5.43)),
                unit="ang",
            ),
            "ang 3x3 matrix",
        ),
        (
            "positions_frac",
            BlockSection(
                name="positions_frac",
                rows=(("Si", 0.0, 0.0, 0.0), ("Si", 0.25, 0.25, 0.25)),
            ),
            "2 rows",
        ),
        (
            "species_pot",
            BlockSection(
                name="species_pot",
                rows=(("Si", "Si_00PBE.usp"),),
            ),
            "1 row",
        ),
        (
            "kpoints_list",
            BlockSection(
                name="kpoints_list",
                rows=((0.0, 0.0, 0.0, 1.0),),
            ),
            "1 row",
        ),
        (
            "ionic_constraints",
            BlockSection(
                name="ionic_constraints",
                rows=((1, "Si", 1, 1.0, 0.0, 0.0),),
            ),
            "1 row",
        ),
    ],
)
def test_block_dialog_roundtrips_supported_block_families(
    qapp,
    name: str,
    initial_value: BlockSection,
    summary: str,
) -> None:
    spec = get_keyword_spec("cell", name)
    assert spec is not None

    dialog = BlockParameterDialog(spec, initial_value=initial_value)

    assert dialog.block_value() == initial_value
    assert summarize_block_value(spec, initial_value) == summary

    dialog.close()

