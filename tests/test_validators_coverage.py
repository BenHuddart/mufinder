"""Branch coverage for previously-untested CASTEP validator paths.

This module deliberately targets the scalar/block validation branches and
error messages that are *not* already exercised by
``tests/test_castep_validators_internal.py`` or
``tests/test_castep_validators_coverage.py``:

* the whole ``_validate_hubbard_u_row`` token grammar,
* block-schema row-count rules (``fixed_row_count`` / ``min_rows`` /
  ``row_group_size``) reached through real schema entries,
* ``_validate_block_cell`` ``int`` / ``float`` / ``bool`` / ``allowed_values``
  branches (synthetic column specs, since the fallback schema has no bool or
  enumerated columns),
* scalar dispatch error branches (``int_triplet`` / ``real_vector`` non-tuple,
  the unsupported-validator-type fallthrough, and block-keyword-as-scalar),
* the ``_validate_block_section`` "not a block keyword" branch,
* the ``validate_keyword_value`` flag branch,
* the standalone ``_validate_kpoints_list`` / ``_validate_lattice_cart`` /
  ``_validate_lattice_abc`` / ``_validate_positions`` /
  ``_validate_atomic_position`` helpers.

Everything is deterministic: no network, GUI, filesystem, sleeps, or
randomness. The in-tree fallback catalog is pinned so the assertions do not
depend on an externally-installed keyword catalog.
"""

from __future__ import annotations

import dataclasses

import pytest

import mufinder.castep_io.schema as schema_module
import mufinder.castep_io.validators as validators
from mufinder.castep_io.schema import BlockColumnSpec, get_keyword_spec
from mufinder.castep_io.types import (
    AtomicPosition,
    BlockSection,
    KeywordAssignment,
    KPoint,
    LatticeABC,
    LatticeCart,
    PositionBlock,
)
from mufinder.castep_io.validators import (
    CastepValidationError,
    UnknownKeywordError,
    _validate_atomic_position,
    _validate_block_cell,
    _validate_block_section,
    _validate_hubbard_u_row,
    _validate_kpoints_list,
    _validate_lattice_abc,
    _validate_lattice_cart,
    _validate_positions,
    _validate_scalar_value,
    validate_keyword_value,
)


@pytest.fixture(autouse=True)
def fallback_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the deterministic in-tree fallback catalog for every test."""

    original_sources = schema_module._candidate_catalog_sources
    monkeypatch.setattr(schema_module, "_candidate_catalog_sources", lambda: ())
    schema_module.refresh_keyword_catalog(allow_fallback=True)
    yield
    monkeypatch.setattr(schema_module, "_candidate_catalog_sources", original_sources)
    schema_module.refresh_keyword_catalog(allow_fallback=True)


# --------------------------------------------------------------------------- #
# _validate_hubbard_u_row — full token grammar.
# --------------------------------------------------------------------------- #
def test_hubbard_u_row_accepts_inline_and_separate_values() -> None:
    # Inline "orbital: value" token.
    _validate_hubbard_u_row(("Fe", "d: 2.4"))
    # Explicit atom index followed by an inline orbital token.
    _validate_hubbard_u_row(("Fe", 2, "d: 2.4"))
    # Orbital token with the value supplied in the following cell.
    _validate_hubbard_u_row(("Fe", "d:", 2.4))
    # Multiple distinct orbitals in one row.
    _validate_hubbard_u_row(("Fe", "d: 2.4", "p: 1.0"))


def test_hubbard_u_row_species_label_required() -> None:
    with pytest.raises(CastepValidationError, match="must start with a species label"):
        _validate_hubbard_u_row((1, "d: 2.4"))  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="must start with a species label"):
        _validate_hubbard_u_row(("   ", "d: 2.4"))


def test_hubbard_u_row_atom_index_must_be_positive_int() -> None:
    with pytest.raises(CastepValidationError, match="atom indices must be positive integers"):
        _validate_hubbard_u_row(("Fe", 0, "d: 2.4"))

    with pytest.raises(CastepValidationError, match="atom indices must be positive integers"):
        _validate_hubbard_u_row(("Fe", -3, "d: 2.4"))

    # bool subclasses int but is not a valid atom index.
    with pytest.raises(CastepValidationError, match="atom indices must be positive integers"):
        _validate_hubbard_u_row(("Fe", True, "d: 2.4"))


def test_hubbard_u_row_orbital_token_rules() -> None:
    # A non-colon token after the species is treated as an atom index and rejected.
    with pytest.raises(CastepValidationError, match="atom indices must be positive integers"):
        _validate_hubbard_u_row(("Fe", "nocolon"))

    with pytest.raises(CastepValidationError, match="must be one of s, p, d, or f"):
        _validate_hubbard_u_row(("Fe", "x: 2.4"))

    with pytest.raises(CastepValidationError, match="defined more than once"):
        _validate_hubbard_u_row(("Fe", "d: 2.4", "d: 1.0"))


def test_hubbard_u_row_missing_and_non_numeric_values() -> None:
    with pytest.raises(CastepValidationError, match="is missing a U value"):
        _validate_hubbard_u_row(("Fe", "d:"))

    with pytest.raises(CastepValidationError, match="values must be numeric"):
        _validate_hubbard_u_row(("Fe", "d: abc"))

    with pytest.raises(CastepValidationError, match="must define at least one orbital"):
        _validate_hubbard_u_row(("Fe",))


def test_hubbard_u_block_unit_and_row_dispatch() -> None:
    # A valid HUBBARD_U block routes each row through _validate_hubbard_u_row.
    _validate_block_section("cell", BlockSection("hubbard_u", (("Fe", "d: 2.4"),)))

    with pytest.raises(CastepValidationError, match="HUBBARD_U units must be eV or Ha"):
        _validate_block_section(
            "cell", BlockSection("hubbard_u", (("Fe", "d: 2.4"),), unit="K")
        )

    # eV / Ha units are accepted (case-insensitive).
    _validate_block_section(
        "cell", BlockSection("hubbard_u", (("Fe", "d: 2.4"),), unit="eV")
    )


# --------------------------------------------------------------------------- #
# Block-schema row-count rules: fixed_row_count, min_rows, row_group_size.
# --------------------------------------------------------------------------- #
def test_block_fixed_row_count_enforced() -> None:
    # lattice_cart has a fixed row count of 3.
    with pytest.raises(CastepValidationError, match="requires exactly 3 rows"):
        _validate_block_section(
            "cell", BlockSection("lattice_cart", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        )


def test_block_min_rows_and_row_group_size_enforced() -> None:
    # symmetry_ops requires at least 4 rows in groups of 4.
    good = tuple((1.0, 0.0, 0.0) for _ in range(4))
    _validate_block_section("cell", BlockSection("symmetry_ops", good))

    with pytest.raises(CastepValidationError, match="requires at least 4 rows"):
        _validate_block_section(
            "cell", BlockSection("symmetry_ops", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
        )

    not_multiple = tuple((1.0, 0.0, 0.0) for _ in range(6))
    with pytest.raises(CastepValidationError, match="must be a multiple of 4"):
        _validate_block_section("cell", BlockSection("symmetry_ops", not_multiple))


def test_block_empty_row_rejected() -> None:
    with pytest.raises(CastepValidationError, match="block rows may not be empty"):
        _validate_block_section("cell", BlockSection("positions_frac", ((),)))


def test_block_int_column_validation() -> None:
    # ionic_constraints has integer 'Constraint' / 'Ion Index' columns.
    _validate_block_section(
        "cell", BlockSection("ionic_constraints", ((1, "Si", 1, 0.0, 0.0, 0.0),))
    )

    with pytest.raises(CastepValidationError, match="column 'Constraint' expects integers"):
        _validate_block_section(
            "cell", BlockSection("ionic_constraints", ((1.5, "Si", 1, 0.0, 0.0, 0.0),))
        )


def test_block_section_rejects_scalar_keyword() -> None:
    with pytest.raises(CastepValidationError, match="CONTINUATION is not a block keyword"):
        _validate_block_section("param", BlockSection("continuation", (("x",),)))


# --------------------------------------------------------------------------- #
# _validate_block_cell — int / float / bool / allowed_values branches.
# The fallback schema has no bool or enumerated columns, so synthetic
# BlockColumnSpec objects are used to reach those branches directly.
# --------------------------------------------------------------------------- #
def test_block_cell_int_branch() -> None:
    column = BlockColumnSpec("Index", "int", ())
    _validate_block_cell(3, column, "LBL")

    with pytest.raises(CastepValidationError, match="column 'Index' expects integers"):
        _validate_block_cell(3.0, column, "LBL")

    # bool is rejected because it subclasses int.
    with pytest.raises(CastepValidationError, match="column 'Index' expects integers"):
        _validate_block_cell(True, column, "LBL")


def test_block_cell_float_branch() -> None:
    column = BlockColumnSpec("Coord", "float", ())
    _validate_block_cell(2, column, "LBL")
    _validate_block_cell(2.5, column, "LBL")

    with pytest.raises(CastepValidationError, match="column 'Coord' expects numeric values"):
        _validate_block_cell("x", column, "LBL")

    with pytest.raises(CastepValidationError, match="column 'Coord' expects numeric values"):
        _validate_block_cell(True, column, "LBL")


def test_block_cell_bool_branch() -> None:
    column = BlockColumnSpec("Flag", "bool", ())
    _validate_block_cell(True, column, "LBL")

    with pytest.raises(CastepValidationError, match="column 'Flag' expects booleans"):
        _validate_block_cell(1, column, "LBL")


def test_block_cell_text_branch() -> None:
    column = BlockColumnSpec("Name", "text", ())
    _validate_block_cell("Si", column, "LBL")

    with pytest.raises(CastepValidationError, match="column 'Name' expects text"):
        _validate_block_cell(5, column, "LBL")


def test_block_cell_allowed_values_branch() -> None:
    column = BlockColumnSpec("Mode", "text", (), allowed_values=("a", "b"))
    _validate_block_cell("a", column, "LBL")

    with pytest.raises(CastepValidationError, match="column 'Mode' must be one of: a, b"):
        _validate_block_cell("c", column, "LBL")


# --------------------------------------------------------------------------- #
# Scalar dispatch: int_triplet / real_vector non-tuple, block-as-scalar,
# and the unsupported-validator-type fallthrough.
# --------------------------------------------------------------------------- #
def test_scalar_int_triplet_requires_tuple() -> None:
    with pytest.raises(CastepValidationError, match="expects a tuple of three integers"):
        _validate_scalar_value("cell", KeywordAssignment("kpoint_mp_grid", 4))


def test_scalar_real_vector_requires_tuple() -> None:
    with pytest.raises(CastepValidationError, match="expects a numeric vector value"):
        _validate_scalar_value("cell", KeywordAssignment("kpoint_mp_offset", 0.5))


def test_scalar_block_keyword_rejected() -> None:
    with pytest.raises(CastepValidationError, match="must be provided as a block"):
        _validate_scalar_value("cell", KeywordAssignment("hubbard_u", "Fe"))


def test_scalar_unknown_keyword_rejected() -> None:
    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP param keyword"):
        _validate_scalar_value("param", KeywordAssignment("definitely_not_real", 1))


def test_scalar_unsupported_validator_type_fallthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_spec = get_keyword_spec("param", "continuation")
    assert real_spec is not None
    bogus = dataclasses.replace(real_spec, value_type="weirdtype")

    def fake_get_keyword_spec(file, name):  # type: ignore[no-untyped-def]
        if name == "continuation":
            return bogus
        return get_keyword_spec(file, name)

    monkeypatch.setattr(validators, "get_keyword_spec", fake_get_keyword_spec)

    with pytest.raises(CastepValidationError, match="Unsupported validator type 'weirdtype'"):
        _validate_scalar_value("param", KeywordAssignment("continuation", "x"))


# --------------------------------------------------------------------------- #
# validate_keyword_value — flag branch.
# --------------------------------------------------------------------------- #
def test_validate_keyword_value_flag_branch_accepts_presence_states() -> None:
    # A flag keyword: falsey / empty / None values short-circuit to acceptance.
    validate_keyword_value("cell", "symmetry_generate", False)
    validate_keyword_value("cell", "symmetry_generate", None)
    validate_keyword_value("cell", "symmetry_generate", "")
    # A truthy presence marker is also accepted.
    validate_keyword_value("cell", "symmetry_generate", True)


def test_validate_keyword_value_unknown_keyword() -> None:
    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP cell keyword"):
        validate_keyword_value("cell", "definitely_not_real", 1)


# --------------------------------------------------------------------------- #
# Standalone geometry / k-point helper error branches.
# --------------------------------------------------------------------------- #
def test_validate_kpoints_list_branches() -> None:
    with pytest.raises(CastepValidationError, match="must contain at least one k-point row"):
        _validate_kpoints_list(())

    with pytest.raises(CastepValidationError, match="must contain three coordinates"):
        _validate_kpoints_list((KPoint((0.0, 0.0), 1.0),))  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="weights must sum to 1.0"):
        _validate_kpoints_list((KPoint((0.0, 0.0, 0.0), 0.5),))

    # Two rows whose weights sum to exactly 1.0 are accepted.
    _validate_kpoints_list(
        (KPoint((0.0, 0.0, 0.0), 0.5), KPoint((0.5, 0.5, 0.5), 0.5))
    )


def test_validate_lattice_cart_vector_component_count() -> None:
    with pytest.raises(CastepValidationError, match="vector 1 must contain three components"):
        _validate_lattice_cart(
            LatticeCart(vectors=((1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        )


def test_validate_lattice_abc_length_and_angle_counts() -> None:
    with pytest.raises(CastepValidationError, match="requires three lengths and three angles"):
        _validate_lattice_abc(
            LatticeABC(lengths=(4.0, 4.0), angles=(90.0, 90.0, 90.0))
        )


def test_validate_positions_empty_block() -> None:
    with pytest.raises(CastepValidationError, match="must contain at least one position row"):
        _validate_positions(
            PositionBlock(()), allow_unit=False, block_name="POSITIONS_FRAC"
        )


def test_validate_atomic_position_branches() -> None:
    with pytest.raises(CastepValidationError, match="entries require a species label"):
        _validate_atomic_position(
            AtomicPosition("  ", (0.0, 0.0, 0.0)), "POSITIONS_FRAC"
        )

    with pytest.raises(CastepValidationError, match="coordinates must contain three components"):
        _validate_atomic_position(AtomicPosition("Si", (0.0, 0.0)), "POSITIONS_FRAC")

    # A well-formed row validates without error.
    _validate_atomic_position(AtomicPosition("Si", (0.0, 0.0, 0.0)), "POSITIONS_FRAC")
