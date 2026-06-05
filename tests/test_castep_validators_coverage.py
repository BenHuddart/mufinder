"""Additional coverage for scalar value-type validation branches.

These tests exercise the per-``value_type`` branches inside
``_validate_scalar_value`` (reached through the public ``validate_keyword_value``
and ``validate_param_file`` entry points) plus the mutual-exclusion and
requirement helpers. They deliberately do not overlap with
``tests/test_castep_validators_internal.py``.
"""

from __future__ import annotations

import pytest

import mufinder.castep_io.schema as schema_module
from mufinder.castep_io.types import (
    AtomicPosition,
    CellFile,
    KeywordAssignment,
    LatticeCart,
    PositionBlock,
    ValueWithUnit,
)
from mufinder.castep_io.validators import (
    CastepValidationError,
    ConflictingKeywordError,
    MissingRequiredKeywordError,
    UnknownKeywordError,
    validate_cell_file,
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
# int value_type (param: charge)
# --------------------------------------------------------------------------- #
def test_int_value_type_accepts_integer_and_rejects_non_integers() -> None:
    # Valid integer (including negative, since charge has no lower bound).
    validate_keyword_value("param", "charge", 3)
    validate_keyword_value("param", "charge", -2)

    with pytest.raises(CastepValidationError, match="CHARGE expects an integer value"):
        validate_keyword_value("param", "charge", 1.5)

    # bool is a subclass of int but must be rejected.
    with pytest.raises(CastepValidationError, match="CHARGE expects an integer value"):
        validate_keyword_value("param", "charge", True)

    with pytest.raises(CastepValidationError, match="CHARGE expects an integer value"):
        validate_keyword_value("param", "charge", "5")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# positive_int value_type (param: max_scf_cycles, min_value == 1)
# --------------------------------------------------------------------------- #
def test_positive_int_value_type_branches() -> None:
    validate_keyword_value("param", "max_scf_cycles", 30)

    with pytest.raises(CastepValidationError, match="expects an integer greater than zero"):
        validate_keyword_value("param", "max_scf_cycles", 0)

    with pytest.raises(CastepValidationError, match="expects an integer greater than zero"):
        validate_keyword_value("param", "max_scf_cycles", -4)

    with pytest.raises(CastepValidationError, match="expects an integer greater than zero"):
        validate_keyword_value("param", "max_scf_cycles", True)

    with pytest.raises(CastepValidationError, match="expects an integer greater than zero"):
        validate_keyword_value("param", "max_scf_cycles", 2.0)


# --------------------------------------------------------------------------- #
# non_negative_int value_type (param: geom_max_iter, min_value == 0)
# --------------------------------------------------------------------------- #
def test_non_negative_int_value_type_branches() -> None:
    validate_keyword_value("param", "geom_max_iter", 0)
    validate_keyword_value("param", "geom_max_iter", 50)

    with pytest.raises(CastepValidationError, match="greater than or equal to zero"):
        validate_keyword_value("param", "geom_max_iter", -1)

    with pytest.raises(CastepValidationError, match="greater than or equal to zero"):
        validate_keyword_value("param", "geom_max_iter", True)

    with pytest.raises(CastepValidationError, match="greater than or equal to zero"):
        validate_keyword_value("param", "geom_max_iter", 3.5)


# --------------------------------------------------------------------------- #
# number / float value_type (param: spin)
# --------------------------------------------------------------------------- #
def test_number_value_type_accepts_numbers_and_rejects_others() -> None:
    validate_keyword_value("param", "spin", 2)
    validate_keyword_value("param", "spin", 1.5)

    with pytest.raises(CastepValidationError, match="SPIN expects a numeric value"):
        validate_keyword_value("param", "spin", "up")  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="SPIN expects a numeric value"):
        validate_keyword_value("param", "spin", True)


# --------------------------------------------------------------------------- #
# text value_type (cell: unit_cell, param: continuation)
# --------------------------------------------------------------------------- #
def test_text_value_type_branches() -> None:
    validate_keyword_value("param", "continuation", "default")
    validate_keyword_value("cell", "unit_cell", "anything")

    with pytest.raises(CastepValidationError, match="CONTINUATION expects text"):
        validate_keyword_value("param", "continuation", 42)  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="UNIT_CELL expects text"):
        validate_keyword_value("cell", "unit_cell", (1, 2, 3))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# real_vector value_type (cell: kpoint_mp_offset, vector_length == 3)
# --------------------------------------------------------------------------- #
def test_real_vector_value_type_branches() -> None:
    validate_keyword_value("cell", "kpoint_mp_offset", (0.0, 0.5, 0.25))
    # Integers are acceptable real components.
    validate_keyword_value("cell", "kpoint_mp_offset", (0, 1, 0))

    with pytest.raises(CastepValidationError, match="expects a numeric vector value"):
        validate_keyword_value("cell", "kpoint_mp_offset", 0.5)  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="expects exactly 3 components"):
        validate_keyword_value("cell", "kpoint_mp_offset", (0.0, 0.5))

    with pytest.raises(CastepValidationError, match="expects exactly 3 components"):
        validate_keyword_value("cell", "kpoint_mp_offset", (0.0, 0.5, 0.25, 0.1))

    with pytest.raises(CastepValidationError, match="expects numeric vector components"):
        validate_keyword_value("cell", "kpoint_mp_offset", (0.0, "x", 0.25))  # type: ignore[arg-type]

    # bool components are rejected even though bool subclasses int.
    with pytest.raises(CastepValidationError, match="expects numeric vector components"):
        validate_keyword_value("cell", "kpoint_mp_offset", (True, 0.0, 0.0))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# enum value_type (param: task)
# --------------------------------------------------------------------------- #
def test_enum_value_type_branches() -> None:
    # Canonicalization upper-cases, so lower case input is accepted.
    validate_keyword_value("param", "task", "singlepoint")
    validate_keyword_value("param", "task", "GEOMETRYOPTIMIZATION")

    with pytest.raises(CastepValidationError, match="is not supported by the starter schema"):
        validate_keyword_value("param", "task", "NOTATASK")


# --------------------------------------------------------------------------- #
# bool value_type (param: spin_polarized)
# --------------------------------------------------------------------------- #
def test_bool_value_type_branches() -> None:
    validate_keyword_value("param", "spin_polarized", True)
    validate_keyword_value("param", "spin_polarized", False)

    with pytest.raises(CastepValidationError, match="expects TRUE or FALSE"):
        validate_keyword_value("param", "spin_polarized", "yes")  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="expects TRUE or FALSE"):
        validate_keyword_value("param", "spin_polarized", 1)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# positive_float_with_optional_unit value_type (param: cut_off_energy,
# elec_energy_tol) — including the ValueWithUnit and unit-specific branches.
# --------------------------------------------------------------------------- #
def test_positive_float_with_unit_branches() -> None:
    validate_keyword_value("param", "cut_off_energy", 400.0)
    validate_keyword_value("param", "cut_off_energy", ValueWithUnit(400.0, "eV"))

    # Tuple is explicitly rejected for a scalar physical quantity.
    with pytest.raises(CastepValidationError, match="expects a scalar numeric value"):
        validate_keyword_value("param", "cut_off_energy", (1.0, 2.0))  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="expects a numeric value"):
        validate_keyword_value("param", "cut_off_energy", "lots")  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="greater than zero"):
        validate_keyword_value("param", "cut_off_energy", -5.0)

    with pytest.raises(CastepValidationError, match="units may not be empty"):
        validate_keyword_value("param", "cut_off_energy", ValueWithUnit(400.0, "   "))


def test_elec_energy_tol_rejects_per_atom_units() -> None:
    validate_keyword_value("param", "elec_energy_tol", ValueWithUnit(1e-6, "eV"))

    with pytest.raises(CastepValidationError, match="not per-atom units"):
        validate_keyword_value("param", "elec_energy_tol", ValueWithUnit(1e-6, "eV/atom"))


# --------------------------------------------------------------------------- #
# int_triplet value_type (cell: kpoint_mp_grid) through validate_keyword_value.
# --------------------------------------------------------------------------- #
def test_int_triplet_value_type_branches() -> None:
    validate_keyword_value("cell", "kpoint_mp_grid", (2, 2, 2))

    with pytest.raises(CastepValidationError, match="expects a tuple of three integers"):
        validate_keyword_value("cell", "kpoint_mp_grid", 4)  # type: ignore[arg-type]

    with pytest.raises(CastepValidationError, match="values must be greater than zero"):
        validate_keyword_value("cell", "kpoint_mp_grid", (2, 0, 2))


# --------------------------------------------------------------------------- #
# Scalar dispatch error paths: None value, flag misuse, block-as-scalar.
# --------------------------------------------------------------------------- #
def test_scalar_dispatch_error_paths() -> None:
    from mufinder.castep_io.validators import _validate_scalar_value

    with pytest.raises(UnknownKeywordError, match="Unsupported CASTEP param keyword"):
        validate_keyword_value("param", "definitely_not_real", 1)

    # A block keyword cannot be supplied as a scalar assignment.
    with pytest.raises(CastepValidationError, match="must be provided as a block"):
        _validate_scalar_value("cell", KeywordAssignment("species_pot", "Si"))


def test_value_required_for_typed_keyword() -> None:
    # A typed (non-flag) keyword with value None must be rejected.
    from mufinder.castep_io.validators import _validate_scalar_value

    with pytest.raises(CastepValidationError, match="requires an explicit value"):
        _validate_scalar_value("param", KeywordAssignment("cut_off_energy", None))


def test_flag_value_type_rejects_explicit_value() -> None:
    # A presence-only flag may not carry a value.
    from mufinder.castep_io.validators import _validate_scalar_value
    from mufinder.castep_io.schema import SCHEMA

    flag_entry = next(
        (
            (file, name)
            for file in ("cell", "param")
            for name, spec in SCHEMA[file].items()
            if spec.kind == "flag"
        ),
        None,
    )
    if flag_entry is None:
        pytest.skip("fallback schema has no flag-kind keyword")

    file_kind, flag_name = flag_entry
    # A bare flag (value None or True) is accepted.
    _validate_scalar_value(file_kind, KeywordAssignment(flag_name, None))
    _validate_scalar_value(file_kind, KeywordAssignment(flag_name, True))

    with pytest.raises(CastepValidationError, match="presence-only flag"):
        _validate_scalar_value(file_kind, KeywordAssignment(flag_name, "value"))


# --------------------------------------------------------------------------- #
# Mutual exclusion (cell: kpoint_mp_spacing vs kpoint_mp_grid).
# --------------------------------------------------------------------------- #
def _cell_with_geometry(**extra: object) -> CellFile:
    return CellFile(
        lattice_cart=LatticeCart(
            vectors=((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),),
        ),
        **extra,
    )


def test_mutual_exclusion_kpoint_grid_and_spacing() -> None:
    with pytest.raises(ConflictingKeywordError, match="cannot be used together"):
        validate_cell_file(
            _cell_with_geometry(
                kpoint_mp_grid=(2, 2, 2),
                kpoint_mp_spacing=ValueWithUnit(0.05, "1/ang"),
            )
        )


def test_no_conflict_when_only_one_kpoint_mode_present() -> None:
    # Grid alone is valid; spacing alone is valid. No exception expected.
    validate_cell_file(_cell_with_geometry(kpoint_mp_grid=(2, 2, 2)))
    validate_cell_file(
        _cell_with_geometry(kpoint_mp_spacing=ValueWithUnit(0.05, "1/ang"))
    )


# --------------------------------------------------------------------------- #
# Requirements (cell: positions_frac requires_any_of lattice_*).
# --------------------------------------------------------------------------- #
def test_requirements_any_of_missing_lattice() -> None:
    with pytest.raises(MissingRequiredKeywordError):
        validate_cell_file(
            CellFile(
                positions_frac=PositionBlock(
                    positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),),
                )
            )
        )
