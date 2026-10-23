"""Validation tests for CASTEP input models."""

from __future__ import annotations

from collections.abc import Iterator
import dataclasses

import pytest

from mufinder.castep_io import (
    AtomicPosition,
    BlockSection,
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
from mufinder.castep_io.schema import BlockColumnSpec, get_keyword_spec
from mufinder.castep_io.types import SpeciesPotential
import mufinder.castep_io.validators as validators
from mufinder.castep_io.validators import (
    CastepValidationError,
    DuplicateKeywordError,
    UnknownKeywordError,
    _find_param_value,
    _ordered_canonical_names,
    _validate_block_cell,
    _validate_block_section,
    _validate_bounds,
    _validate_kpoints_list,
    _validate_scalar_value,
    validate_keyword_value,
)
from tests.support.golden_inputs import pinned_starter_catalog


@pytest.fixture(autouse=True)
def _starter_catalog() -> Iterator[None]:
    """Validate against the bundled starter catalog, whatever is installed locally."""

    with pinned_starter_catalog():
        yield


def _cell(**extra: object) -> CellFile:
    fields: dict[str, object] = {
        "lattice_cart": LatticeCart(
            vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
            unit="ang",
        ),
        "positions_frac": PositionBlock(positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
    }
    fields.update(extra)
    return CellFile(**fields)  # type: ignore[arg-type]


def _block(name: str, rows: tuple[tuple[object, ...], ...], unit: str | None = None) -> None:
    validate_keyword_value("cell", name, BlockSection(name, rows, unit=unit))  # type: ignore[arg-type]


# --- Cell geometry and required blocks ----------------------------------------


def test_validate_cell_rejects_conflicting_lattice_blocks() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        lattice_abc=LatticeABC(lengths=(1.0, 1.0, 1.0), angles=(90.0, 90.0, 90.0)),
        positions_frac=PositionBlock(positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
    )

    with pytest.raises(
        ConflictingKeywordError, match="LATTICE_CART cannot be used together with LATTICE_ABC"
    ):
        validate_cell_file(cell_file)


def test_validate_cell_requires_a_lattice() -> None:
    with pytest.raises(
        MissingRequiredKeywordError, match="requires either LATTICE_CART or LATTICE_ABC"
    ):
        validate_cell_file(CellFile())

    # Positions without a lattice fail the keyword requirement rule first.
    with pytest.raises(MissingRequiredKeywordError):
        validate_cell_file(
            CellFile(
                positions_frac=PositionBlock(positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),))
            )
        )


def test_validate_cell_requires_positions() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    )

    with pytest.raises(
        MissingRequiredKeywordError, match="requires either POSITIONS_FRAC or POSITIONS_ABS"
    ):
        validate_cell_file(cell_file)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        pytest.param(
            {"lattice_cart": LatticeCart(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))},  # type: ignore[arg-type]
            "requires exactly three lattice vectors",
            id="two-lattice-vectors",
        ),
        pytest.param(
            {"lattice_cart": LatticeCart(vectors=((1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))},  # type: ignore[arg-type]
            "vector 1 must contain three components",
            id="short-lattice-vector",
        ),
        pytest.param(
            {
                "lattice_cart": LatticeCart(
                    vectors=((1.0, "bad", 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                )
            },  # type: ignore[arg-type]
            "expects numeric values",
            id="non-numeric-lattice-vector",
        ),
        pytest.param(
            {
                "lattice_cart": None,
                "lattice_abc": LatticeABC(lengths=(4.0, 4.0), angles=(90.0, 90.0, 90.0)),
            },  # type: ignore[arg-type]
            "requires three lengths and three angles",
            id="two-abc-lengths",
        ),
        pytest.param(
            {
                "lattice_cart": None,
                "lattice_abc": LatticeABC(lengths=(4.0, 0.0, 4.0), angles=(90.0, 90.0, 90.0)),
            },
            "lengths values must be greater than zero",
            id="zero-abc-length",
        ),
        pytest.param(
            {
                "lattice_cart": None,
                "lattice_abc": LatticeABC(lengths=(4.0, 4.0, 4.0), angles=(90.0, 90.0, 180.0)),
            },
            "angles must be less than 180",
            id="straight-abc-angle",
        ),
        pytest.param(
            {"positions_frac": PositionBlock(())},
            "must contain at least one position row",
            id="empty-positions",
        ),
        pytest.param(
            {"positions_frac": PositionBlock((AtomicPosition("Si", (0.0, 0.0, 0.0)),), unit="ang")},
            "POSITIONS_FRAC does not support a unit line",
            id="unit-on-fractional-positions",
        ),
        pytest.param(
            {"positions_frac": PositionBlock((AtomicPosition("  ", (0.0, 0.0, 0.0)),))},
            "entries require a species label",
            id="blank-species",
        ),
        pytest.param(
            {"positions_frac": PositionBlock((AtomicPosition("Si", (0.0, 0.0)),))},  # type: ignore[arg-type]
            "coordinates must contain three components",
            id="short-coordinates",
        ),
        pytest.param(
            {"species_pot": (SpeciesPotential("", ""),)},
            "require both species and pseudopotential name",
            id="blank-species-pot",
        ),
        pytest.param(
            {"symmetry_generate": "yes"},
            "must be a boolean",
            id="non-boolean-symmetry-generate",
        ),
    ],
)
def test_validate_cell_rejects_malformed_geometry(fields: dict[str, object], message: str) -> None:
    with pytest.raises(CastepValidationError, match=message):
        validate_cell_file(_cell(**fields))


def test_validate_cell_rejects_unknown_scalars_and_blocks() -> None:
    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP cell keyword"):
        validate_cell_file(_cell(scalars=(KeywordAssignment("not_real", 1),)))

    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP cell keyword"):
        validate_cell_file(_cell(blocks=(BlockSection("not_real", ((1,),)),)))


# --- K-points -------------------------------------------------------------------


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


def test_validate_kpoints_list_rows() -> None:
    validate_cell_file(
        _cell(kpoints_list=(KPoint((0.0, 0.0, 0.0), 0.5), KPoint((0.5, 0.5, 0.5), 0.5)))
    )

    with pytest.raises(CastepValidationError, match="must contain three coordinates"):
        validate_cell_file(_cell(kpoints_list=(KPoint((0.0, 0.0), 1.0),)))  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="weights must sum to 1.0"):
        _block("kpoints_list", ((0.0, 0.0, 0.0, 0.5),))


def test_kpoint_grid_and_spacing_are_validated_and_mutually_exclusive() -> None:
    validate_cell_file(_cell(kpoint_mp_grid=(2, 2, 2)))
    validate_cell_file(_cell(kpoint_mp_spacing=ValueWithUnit(0.05, "1/ang")))

    with pytest.raises(ConflictingKeywordError, match="cannot be used together"):
        validate_cell_file(
            _cell(kpoint_mp_grid=(2, 2, 2), kpoint_mp_spacing=ValueWithUnit(0.05, "1/ang"))
        )

    with pytest.raises(
        CastepValidationError, match="KPOINT_MP_GRID values must be greater than zero"
    ):
        validate_cell_file(_cell(kpoint_mp_grid=(1, 0, 1)))

    with pytest.raises(CastepValidationError, match="KPOINT_MP_SPACING expects a numeric value"):
        validate_cell_file(_cell(kpoint_mp_spacing="bad"))


# --- Per-atom spins -------------------------------------------------------------


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


def test_validate_cell_rejects_non_numeric_atomic_spins() -> None:
    with pytest.raises(CastepValidationError, match="must be numeric"):
        validate_cell_file(_cell(atomic_spins=("bad",)))


# --- Param files ----------------------------------------------------------------


def test_validate_param_rejects_conflicting_spin_spellings() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("task", "SINGLEPOINT"),
            KeywordAssignment("spin_polarized", True),
            KeywordAssignment("spin_polarised", False),
        )
    )

    with pytest.raises(
        ConflictingKeywordError, match="SPIN_POLARIZED cannot be used together with SPIN_POLARISED"
    ):
        validate_param_file(param_file)


def test_validate_param_rejects_duplicate_keywords() -> None:
    with pytest.raises(DuplicateKeywordError, match="TASK may only appear once"):
        validate_param_file(
            ParamFile(
                keywords=(
                    KeywordAssignment("task", "SINGLEPOINT"),
                    KeywordAssignment("task", "GEOMETRYOPTIMIZATION"),
                )
            )
        )


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

    with pytest.raises(
        ConflictingKeywordError, match="BASIS_PRECISION cannot be used together with CUT_OFF_ENERGY"
    ):
        validate_param_file(param_file)


def test_validate_param_rejects_elec_energy_tol_per_atom_units() -> None:
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("task", "SINGLEPOINT"),
            KeywordAssignment("elec_energy_tol", ValueWithUnit(1e-6, "eV/atom")),
        )
    )

    with pytest.raises(ValueError, match="not per-atom units"):
        validate_param_file(param_file)


def test_validate_param_rejects_keywords_given_in_the_wrong_form() -> None:
    with pytest.raises(CastepValidationError, match="requires an explicit value"):
        validate_param_file(ParamFile(keywords=(KeywordAssignment("cut_off_energy", None),)))

    with pytest.raises(CastepValidationError, match="DEVEL_CODE must be provided as a block"):
        validate_param_file(ParamFile(keywords=(KeywordAssignment("devel_code", "x"),)))

    with pytest.raises(CastepValidationError, match="CONTINUATION is not a block keyword"):
        validate_param_file(ParamFile(blocks=(BlockSection("continuation", (("x",),)),)))


# --- Scalar keyword values ------------------------------------------------------


@pytest.mark.parametrize("file_kind", ["cell", "param"])
def test_validate_keyword_value_rejects_unknown_keywords(file_kind: str) -> None:
    with pytest.raises(UnknownKeywordError, match=f"Unsupported CASTEP {file_kind} keyword"):
        validate_keyword_value(file_kind, "definitely_not_real", 1)  # type: ignore[arg-type]


def test_flag_keywords_are_presence_only() -> None:
    # Falsey / empty / None values mean "absent"; a truthy marker means "present".
    for value in (False, None, "", True):
        validate_keyword_value("cell", "symmetry_generate", value)
    validate_cell_file(_cell(scalars=(KeywordAssignment("symmetry_generate", None),)))

    with pytest.raises(CastepValidationError, match="presence-only flag"):
        validate_cell_file(_cell(scalars=(KeywordAssignment("symmetry_generate", "value"),)))


def test_int_value_type_accepts_integer_and_rejects_non_integers() -> None:
    # Valid integer (including negative, since charge has no lower bound).
    validate_keyword_value("param", "charge", 3)
    validate_keyword_value("param", "charge", -2)

    # bool is a subclass of int but must be rejected.
    for value in (1.5, True, "5"):
        with pytest.raises(CastepValidationError, match="CHARGE expects an integer value"):
            validate_keyword_value("param", "charge", value)


def test_positive_int_value_type_branches() -> None:
    validate_keyword_value("param", "max_scf_cycles", 30)

    for value in (0, -4, True, 2.0):
        with pytest.raises(CastepValidationError, match="expects an integer greater than zero"):
            validate_keyword_value("param", "max_scf_cycles", value)


def test_non_negative_int_value_type_branches() -> None:
    validate_keyword_value("param", "geom_max_iter", 0)
    validate_keyword_value("param", "geom_max_iter", 50)

    for value in (-1, True, 3.5):
        with pytest.raises(CastepValidationError, match="greater than or equal to zero"):
            validate_keyword_value("param", "geom_max_iter", value)


def test_number_value_type_accepts_numbers_and_rejects_others() -> None:
    validate_keyword_value("param", "spin", 2)
    validate_keyword_value("param", "spin", 1.5)

    for value in ("up", True):
        with pytest.raises(CastepValidationError, match="SPIN expects a numeric value"):
            validate_keyword_value("param", "spin", value)


def test_text_value_type_branches() -> None:
    validate_keyword_value("param", "continuation", "default")
    validate_keyword_value("cell", "unit_cell", "anything")

    with pytest.raises(CastepValidationError, match="CONTINUATION expects text"):
        validate_keyword_value("param", "continuation", 42)

    with pytest.raises(CastepValidationError, match="UNIT_CELL expects text"):
        validate_keyword_value("cell", "unit_cell", (1, 2, 3))


def test_real_vector_value_type_branches() -> None:
    validate_keyword_value("cell", "kpoint_mp_offset", (0.0, 0.5, 0.25))
    # Integers are acceptable real components.
    validate_keyword_value("cell", "kpoint_mp_offset", (0, 1, 0))

    with pytest.raises(CastepValidationError, match="expects a numeric vector value"):
        validate_keyword_value("cell", "kpoint_mp_offset", 0.5)

    for value in ((0.0, 0.5), (0.0, 0.5, 0.25, 0.1)):
        with pytest.raises(CastepValidationError, match="expects exactly 3 components"):
            validate_keyword_value("cell", "kpoint_mp_offset", value)

    # bool components are rejected even though bool subclasses int.
    for value in ((0.0, "x", 0.25), (True, 0.0, 0.0)):
        with pytest.raises(CastepValidationError, match="expects numeric vector components"):
            validate_keyword_value("cell", "kpoint_mp_offset", value)


def test_enum_value_type_branches() -> None:
    # Canonicalization upper-cases, so lower case input is accepted.
    validate_keyword_value("param", "task", "singlepoint")
    validate_keyword_value("param", "task", "GEOMETRYOPTIMIZATION")

    with pytest.raises(CastepValidationError, match="is not supported by the starter schema"):
        validate_keyword_value("param", "task", "NOTATASK")

    with pytest.raises(CastepValidationError, match="Allowed values"):
        validate_keyword_value("param", "basis_precision", "INVALID")


def test_bool_value_type_branches() -> None:
    validate_keyword_value("param", "spin_polarized", True)
    validate_keyword_value("param", "spin_polarized", False)

    for value in ("yes", 1):
        with pytest.raises(CastepValidationError, match="expects TRUE or FALSE"):
            validate_keyword_value("param", "spin_polarized", value)


def test_positive_float_with_unit_branches() -> None:
    validate_keyword_value("param", "cut_off_energy", 400.0)
    validate_keyword_value("param", "cut_off_energy", ValueWithUnit(400.0, "eV"))
    validate_keyword_value("param", "elec_energy_tol", ValueWithUnit(1e-6, "eV"))

    # Tuple is explicitly rejected for a scalar physical quantity.
    with pytest.raises(CastepValidationError, match="expects a scalar numeric value"):
        validate_keyword_value("param", "cut_off_energy", (1.0, 2.0))

    with pytest.raises(CastepValidationError, match="expects a numeric value"):
        validate_keyword_value("param", "cut_off_energy", "lots")

    with pytest.raises(CastepValidationError, match="greater than zero"):
        validate_keyword_value("param", "cut_off_energy", -5.0)

    with pytest.raises(CastepValidationError, match="units may not be empty"):
        validate_keyword_value("param", "cut_off_energy", ValueWithUnit(400.0, "   "))


def test_int_triplet_value_type_branches() -> None:
    validate_keyword_value("cell", "kpoint_mp_grid", (2, 2, 2))

    with pytest.raises(CastepValidationError, match="expects a tuple of three integers"):
        validate_keyword_value("cell", "kpoint_mp_grid", 4)

    with pytest.raises(CastepValidationError, match="expects exactly 3 values"):
        validate_keyword_value("cell", "kpoint_mp_grid", (1, 2))

    with pytest.raises(CastepValidationError, match="expects integer values"):
        validate_keyword_value("cell", "kpoint_mp_grid", (1, 2, 3.5))

    with pytest.raises(CastepValidationError, match="values must be greater than zero"):
        validate_keyword_value("cell", "kpoint_mp_grid", (2, 0, 2))


def test_unsupported_value_type_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    real_spec = get_keyword_spec("param", "continuation")
    assert real_spec is not None
    bogus = dataclasses.replace(real_spec, value_type="weirdtype")
    monkeypatch.setattr(
        validators,
        "get_keyword_spec",
        lambda file, name: bogus if name == "continuation" else get_keyword_spec(file, name),
    )

    with pytest.raises(CastepValidationError, match="Unsupported validator type 'weirdtype'"):
        validate_keyword_value("param", "continuation", "x")


# --- Block keyword values -------------------------------------------------------


def test_block_keywords_require_block_values_and_rows() -> None:
    with pytest.raises(CastepValidationError, match="requires a block value"):
        validate_keyword_value("cell", "species_pot", "Si")

    # The block name is taken from the keyword when the section is unnamed or aliased.
    validate_keyword_value(
        "cell", "species_pot", BlockSection("not_canonical", (("Si", "Si.recpot"),))
    )

    with pytest.raises(CastepValidationError, match="must contain at least one row"):
        _block("species_pot", ())

    with pytest.raises(CastepValidationError, match="block rows may not be empty"):
        _block("positions_frac", ((),))

    with pytest.raises(CastepValidationError, match="must be provided as a block"):
        validate_cell_file(_cell(scalars=(KeywordAssignment("hubbard_u", "Fe"),)))


def test_block_row_count_rules() -> None:
    # lattice_cart has a fixed row count of 3.
    with pytest.raises(CastepValidationError, match="requires exactly 3 rows"):
        _block("lattice_cart", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    # symmetry_ops requires at least 4 rows in groups of 4.
    _block("symmetry_ops", tuple((1.0, 0.0, 0.0) for _ in range(4)))

    with pytest.raises(CastepValidationError, match="requires at least 4 rows"):
        _block("symmetry_ops", ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    with pytest.raises(CastepValidationError, match="must be a multiple of 4"):
        _block("symmetry_ops", tuple((1.0, 0.0, 0.0) for _ in range(6)))


def test_block_columns_check_value_count_and_kind() -> None:
    # ionic_constraints has integer 'Constraint' / 'Ion Index' columns.
    _block("ionic_constraints", ((1, "Si", 1, 0.0, 0.0, 0.0),))

    with pytest.raises(CastepValidationError, match="column 'Constraint' expects integers"):
        _block("ionic_constraints", ((1.5, "Si", 1, 0.0, 0.0, 0.0),))

    # bool is rejected because it subclasses int.
    with pytest.raises(CastepValidationError, match="column 'Constraint' expects integers"):
        _block("ionic_constraints", ((True, "Si", 1, 0.0, 0.0, 0.0),))

    for bad in ("x", True):
        with pytest.raises(CastepValidationError, match="expects numeric values"):
            _block("lattice_cart", ((bad, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))

    with pytest.raises(CastepValidationError, match="rows must contain 2 values"):
        _block("species_pot", (("Si",),))

    with pytest.raises(CastepValidationError, match="column 'Species' expects text"):
        _block("species_pot", ((1, "Si.recpot"),))


def test_block_cell_bool_and_allowed_value_columns() -> None:
    # The starter catalog has no boolean or enumerated block columns; a full
    # CASTEP catalog can, so check those column rules directly.
    flag_column = BlockColumnSpec("Flag", "bool", ())
    _validate_block_cell(True, flag_column, "LBL")
    with pytest.raises(CastepValidationError, match="column 'Flag' expects booleans"):
        _validate_block_cell(1, flag_column, "LBL")

    mode_column = BlockColumnSpec("Mode", "text", (), allowed_values=("a", "b"))
    _validate_block_cell("a", mode_column, "LBL")
    with pytest.raises(CastepValidationError, match="column 'Mode' must be one of: a, b"):
        _validate_block_cell("c", mode_column, "LBL")


# --- HUBBARD_U ------------------------------------------------------------------


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(("Fe", "d: 2.4"), id="inline-value"),
        pytest.param(("Fe", 2, "d: 2.4"), id="atom-index"),
        pytest.param(("Fe", "d:", 2.4), id="separate-value"),
        pytest.param(("Fe", "d: 2.4", "p: 1.0"), id="two-orbitals"),
    ],
)
def test_hubbard_u_rows_accept_castep_token_layouts(row: tuple[object, ...]) -> None:
    _block("hubbard_u", (row,))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        pytest.param((1, "d: 2.4"), "must start with a species label", id="numeric-species"),
        pytest.param(("   ", "d: 2.4"), "must start with a species label", id="blank-species"),
        pytest.param(
            ("Fe", 0, "d: 2.4"), "atom indices must be positive integers", id="zero-index"
        ),
        pytest.param(
            ("Fe", -3, "d: 2.4"), "atom indices must be positive integers", id="negative-index"
        ),
        # bool subclasses int but is not a valid atom index.
        pytest.param(
            ("Fe", True, "d: 2.4"), "atom indices must be positive integers", id="bool-index"
        ),
        # A non-colon token after the species is treated as an atom index.
        pytest.param(("Fe", "nocolon"), "atom indices must be positive integers", id="text-index"),
        pytest.param(("Fe", 1, 2.4), "must use tokens such as 'd: 2.4'", id="bare-value"),
        pytest.param(("Fe", "d: 2.4", "d: 1.0"), "defined more than once", id="repeated-orbital"),
        pytest.param(("Fe", "d:"), "is missing a U value", id="missing-value"),
        pytest.param(("Fe", "d: abc"), "values must be numeric", id="non-numeric-value"),
        pytest.param(("Fe",), "must define at least one orbital", id="no-orbitals"),
    ],
)
def test_hubbard_u_rows_reject_malformed_tokens(row: tuple[object, ...], message: str) -> None:
    with pytest.raises(CastepValidationError, match=message):
        _block("hubbard_u", (row,))


def test_validate_cell_rejects_invalid_hubbard_u_rows() -> None:
    cell_file = CellFile(
        lattice_cart=LatticeCart(vectors=((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0))),
        positions_frac=PositionBlock(positions=(AtomicPosition("Ni", (0.0, 0.0, 0.0)),)),
        blocks=(BlockSection("hubbard_u", (("Ni", "x:", 2.4),), unit="eV"),),
    )

    with pytest.raises(ValueError, match="orbitals must be one of s, p, d, or f"):
        validate_cell_file(cell_file)


def test_hubbard_u_units_must_be_energies() -> None:
    _block("hubbard_u", (("Fe", "d: 2.4"),), unit="eV")
    _block("hubbard_u", (("Fe", "d: 2.4"),), unit="ha")

    with pytest.raises(CastepValidationError, match="HUBBARD_U units must be eV or Ha"):
        _block("hubbard_u", (("Fe", "d: 2.4"),), unit="K")


# --- Defensive guards behind earlier checks -------------------------------------


def test_internal_guards_reject_inputs_the_public_checks_already_exclude() -> None:
    # Each guard sits behind an earlier public check (unknown names, empty
    # k-point lists, type checks, duplicate rejection), so exercise it directly.
    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP cell block"):
        _validate_block_section("cell", BlockSection("not_real", ((1,),)))

    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP param keyword"):
        _validate_scalar_value("param", KeywordAssignment("definitely_not_real", 1))

    with pytest.raises(CastepValidationError, match="must contain at least one k-point row"):
        _validate_kpoints_list(())

    spec = get_keyword_spec("param", "cut_off_energy")
    assert spec is not None
    with pytest.raises(CastepValidationError, match="must be greater than or equal to"):
        _validate_bounds(spec, -1.0)

    param = ParamFile(keywords=(KeywordAssignment("cut_off_energy", ValueWithUnit(400.0, "eV")),))
    assert _find_param_value(param, "cut_off_energy") == pytest.approx(400.0)
    assert _ordered_canonical_names("param", ["task", "TASK", "basis_precision"]) == [
        "task",
        "basis_precision",
    ]
