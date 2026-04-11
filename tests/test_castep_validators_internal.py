"""Additional coverage for CASTEP validator internals."""

from __future__ import annotations

import pytest

import mufinder.castep_io.schema as schema_module
from mufinder.castep_io.schema import get_keyword_spec
from mufinder.castep_io.types import (
    AtomicPosition,
    BlockSection,
    CellFile,
    KeywordAssignment,
    KPoint,
    LatticeABC,
    LatticeCart,
    ParamFile,
    PositionBlock,
    SpeciesPotential,
    ValueWithUnit,
)
from mufinder.castep_io.validators import (
    CastepValidationError,
    ConflictingKeywordError,
    DuplicateKeywordError,
    MissingRequiredKeywordError,
    UnknownKeywordError,
    _find_param_value,
    _ordered_canonical_names,
    _validate_block_section,
    _validate_bounds,
    _validate_numeric_tuple,
    _validate_triplet,
    validate_cell_file,
    validate_keyword_value,
    validate_param_file,
)


@pytest.fixture(autouse=True)
def fallback_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    original_sources = schema_module._candidate_catalog_sources
    monkeypatch.setattr(schema_module, "_candidate_catalog_sources", lambda: ())
    schema_module.refresh_keyword_catalog(allow_fallback=True)
    yield
    monkeypatch.setattr(schema_module, "_candidate_catalog_sources", original_sources)
    schema_module.refresh_keyword_catalog(allow_fallback=True)


def _valid_cell_file() -> CellFile:
    return CellFile(
        lattice_cart=LatticeCart(
            vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),),
        ),
    )


def test_validate_cell_file_covers_requirement_and_geometry_errors() -> None:
    with pytest.raises(MissingRequiredKeywordError, match="requires either LATTICE_CART or LATTICE_ABC"):
        validate_cell_file(CellFile())

    with pytest.raises(MissingRequiredKeywordError, match="requires either POSITIONS_FRAC or POSITIONS_ABS"):
        validate_cell_file(CellFile(lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0),) * 3)))

    with pytest.raises(CastepValidationError, match="requires exactly three lattice vectors"):
        validate_cell_file(
            CellFile(
                lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
                positions_frac=PositionBlock((AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
            )
        )

    with pytest.raises(CastepValidationError, match="angles must be less than 180"):
        validate_cell_file(
            CellFile(
                lattice_abc=LatticeABC(lengths=(4.0, 4.0, 4.0), angles=(90.0, 90.0, 180.0)),
                positions_frac=PositionBlock((AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
            )
        )

    with pytest.raises(CastepValidationError, match="POSITIONS_FRAC does not support a unit line"):
        validate_cell_file(
            CellFile(
                lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0),) * 3),
                positions_frac=PositionBlock((AtomicPosition("Si", (0.0, 0.0, 0.0)),), unit="ang"),
            )
        )

    with pytest.raises(CastepValidationError, match="Per-atom spins require POSITIONS_FRAC"):
        validate_cell_file(
            CellFile(
                lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0),) * 3),
                positions_abs=PositionBlock((AtomicPosition("Si", (0.0, 0.0, 0.0)),), unit="ang"),
                atomic_spins=(1.0,),
            )
        )

    with pytest.raises(CastepValidationError, match="Atomic spin count must match POSITIONS_FRAC row count"):
        validate_cell_file(
            CellFile(
                lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0),) * 3),
                positions_frac=PositionBlock((AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
                atomic_spins=(1.0, 2.0),
            )
        )

    with pytest.raises(CastepValidationError, match="must be numeric"):
        validate_cell_file(
            CellFile(
                lattice_cart=LatticeCart(vectors=((1.0, 0.0, 0.0),) * 3),
                positions_frac=PositionBlock((AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
                atomic_spins=("bad",),  # type: ignore[arg-type]
            )
        )

    valid = _valid_cell_file()
    with pytest.raises(CastepValidationError, match="require both species and pseudopotential name"):
        validate_cell_file(
            CellFile(
                lattice_cart=valid.lattice_cart,
                positions_frac=valid.positions_frac,
                species_pot=(SpeciesPotential("", ""),),
            )
        )


def test_validate_cell_file_covers_scalar_block_and_weight_errors() -> None:
    base = _valid_cell_file()

    with pytest.raises(CastepValidationError, match="weights must sum to 1.0"):
        validate_cell_file(
            CellFile(
                lattice_cart=base.lattice_cart,
                positions_frac=base.positions_frac,
                kpoints_list=(KPoint((0.0, 0.0, 0.0), 0.4),),
            )
        )

    with pytest.raises(CastepValidationError, match="must be a boolean"):
        validate_cell_file(
            CellFile(
                lattice_cart=base.lattice_cart,
                positions_frac=base.positions_frac,
                symmetry_generate="yes",  # type: ignore[arg-type]
            )
        )

    with pytest.raises(CastepValidationError, match="KPOINT_MP_GRID values must be greater than zero"):
        validate_cell_file(
            CellFile(
                lattice_cart=base.lattice_cart,
                positions_frac=base.positions_frac,
                kpoint_mp_grid=(1, 0, 1),
            )
        )

    with pytest.raises(CastepValidationError, match="KPOINT_MP_SPACING expects a numeric value"):
        validate_cell_file(
            CellFile(
                lattice_cart=base.lattice_cart,
                positions_frac=base.positions_frac,
                kpoint_mp_spacing="bad",  # type: ignore[arg-type]
            )
        )

    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP cell keyword"):
        validate_cell_file(
            CellFile(
                lattice_cart=base.lattice_cart,
                positions_frac=base.positions_frac,
                scalars=(KeywordAssignment("not_real", 1),),
            )
        )

    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP cell keyword"):
        validate_cell_file(
            CellFile(
                lattice_cart=base.lattice_cart,
                positions_frac=base.positions_frac,
                blocks=(BlockSection("not_real", ((1,),)),),
            )
        )


def test_validate_param_file_covers_duplicates_conflicts_and_task_restrictions() -> None:
    with pytest.raises(DuplicateKeywordError, match="TASK may only appear once"):
        validate_param_file(
            ParamFile(
                keywords=(
                    KeywordAssignment("task", "SINGLEPOINT"),
                    KeywordAssignment("task", "GEOMETRYOPTIMIZATION"),
                )
            )
        )

    with pytest.raises(ConflictingKeywordError, match="CUT_OFF_ENERGY cannot be used together with BASIS_PRECISION"):
        validate_param_file(
            ParamFile(
                keywords=(
                    KeywordAssignment("task", "SINGLEPOINT"),
                    KeywordAssignment("cut_off_energy", ValueWithUnit(300.0, "eV")),
                    KeywordAssignment("basis_precision", "FINE"),
                )
            )
        )

    with pytest.raises(CastepValidationError, match="only supported by this starter implementation for task"):
        validate_param_file(
            ParamFile(
                keywords=(
                    KeywordAssignment("task", "SINGLEPOINT"),
                    KeywordAssignment("geom_force_tol", ValueWithUnit(0.05, "eV/ang")),
                    KeywordAssignment("geom_max_iter", 20),
                )
            )
        )


def test_validate_keyword_value_and_block_helpers_cover_remaining_branches() -> None:
    with pytest.raises(CastepValidationError, match="requires a block value"):
        validate_keyword_value("cell", "species_pot", "Si")  # type: ignore[arg-type]

    validate_keyword_value("cell", "species_pot", BlockSection("not_canonical", (("Si", "Si.recpot"),)))

    with pytest.raises(CastepValidationError, match="must contain at least one row"):
        _validate_block_section("cell", BlockSection("species_pot", ()))

    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP cell block"):
        _validate_block_section("cell", BlockSection("not_real", ((1,),)))

    with pytest.raises(CastepValidationError, match="rows must contain 2 values"):
        _validate_block_section("cell", BlockSection("species_pot", (("Si",),)))

    with pytest.raises(CastepValidationError, match="column 'Species' expects text"):
        _validate_block_section("cell", BlockSection("species_pot", ((1, "Si.recpot"),)))

    with pytest.raises(CastepValidationError, match="weights must sum to 1.0"):
        _validate_block_section("cell", BlockSection("kpoints_list", ((0.0, 0.0, 0.0, 0.5),)))

    with pytest.raises(CastepValidationError, match="expects exactly 3 values"):
        _validate_triplet((1, 2), "TRIPLET", positive_only=True, expected_length=3)  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="expects integer values"):
        _validate_triplet((1, 2, 3.5), "TRIPLET", positive_only=False, expected_length=3)  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="values must be greater than zero"):
        _validate_triplet((1, 0, 3), "TRIPLET", positive_only=True, expected_length=3)

    with pytest.raises(CastepValidationError, match="expects numeric values"):
        _validate_numeric_tuple((1.0, "bad", 3.0), "VECTOR")  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="values must be greater than zero"):
        _validate_numeric_tuple((1.0, 0.0, 3.0), "VECTOR", positive_only=True)

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
