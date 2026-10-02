"""Validation logic for CASTEP input objects."""

from __future__ import annotations

from collections import Counter
from math import isclose
from typing import Iterable

from .schema import (
    keyword_catalog_status,
    BlockColumnSpec,
    FileKind,
    canonical_keyword_name,
    get_keyword_spec,
)
from .types import (
    AtomicPosition,
    BlockSection,
    CellFile,
    KeywordAssignment,
    KPoint,
    LatticeABC,
    LatticeCart,
    ParamFile,
    PositionBlock,
    ScalarPrimitive,
    ScalarValue,
    SpeciesPotential,
    ValueWithUnit,
)


class CastepValidationError(ValueError):
    """Base class for CASTEP schema and content validation errors."""


class UnknownKeywordError(CastepValidationError):
    """Raised when a keyword is not in the implemented starter schema."""


class DuplicateKeywordError(CastepValidationError):
    """Raised when a non-repeatable keyword appears more than once."""


class ConflictingKeywordError(CastepValidationError):
    """Raised when mutually exclusive keywords are present together."""


class MissingRequiredKeywordError(CastepValidationError):
    """Raised when a required keyword or block is missing."""


def validate_cell_file(cell_file: CellFile) -> None:
    """Validate a structured ``.cell`` payload."""

    present_names = _cell_present_names(cell_file)
    _validate_known_names("cell", present_names)
    _validate_duplicates("cell", present_names)
    _validate_mutual_exclusion("cell", present_names)
    _validate_requirements("cell", present_names)

    if cell_file.lattice_cart is None and cell_file.lattice_abc is None:
        raise MissingRequiredKeywordError(
            "A CASTEP .cell file requires either LATTICE_CART or LATTICE_ABC."
        )
    if cell_file.positions_frac is None and cell_file.positions_abs is None:
        raise MissingRequiredKeywordError(
            "A CASTEP .cell file requires either POSITIONS_FRAC or POSITIONS_ABS."
        )

    if cell_file.lattice_cart is not None:
        _validate_lattice_cart(cell_file.lattice_cart)
    if cell_file.lattice_abc is not None:
        _validate_lattice_abc(cell_file.lattice_abc)
    if cell_file.positions_frac is not None:
        _validate_positions(cell_file.positions_frac, allow_unit=False, block_name="POSITIONS_FRAC")
    if cell_file.positions_abs is not None:
        _validate_positions(cell_file.positions_abs, allow_unit=True, block_name="POSITIONS_ABS")
    if cell_file.species_pot:
        _validate_species_pot(cell_file.species_pot)
    if cell_file.kpoints_list:
        _validate_kpoints_list(cell_file.kpoints_list)
    if cell_file.kpoint_mp_grid is not None:
        _validate_triplet(
            cell_file.kpoint_mp_grid,
            "KPOINT_MP_GRID",
            positive_only=True,
            expected_length=3,
        )
    if cell_file.kpoint_mp_spacing is not None:
        _validate_scalar_value(
            "cell",
            KeywordAssignment(name="kpoint_mp_spacing", value=cell_file.kpoint_mp_spacing),
        )
    if cell_file.atomic_spins is not None:
        _validate_atomic_spins(cell_file)
    if cell_file.symmetry_generate not in (False, True):
        raise CastepValidationError("SYMMETRY_GENERATE must be a boolean.")

    for assignment in cell_file.scalars:
        _validate_scalar_value("cell", assignment)
    for block in cell_file.blocks:
        _validate_block_section("cell", block)


def validate_param_file(param_file: ParamFile) -> None:
    """Validate a structured ``.param`` payload."""

    present_names = _param_present_names(param_file)
    _validate_known_names("param", present_names)
    _validate_duplicates("param", present_names)
    _validate_mutual_exclusion("param", present_names)
    _validate_requirements("param", present_names)

    for assignment in param_file.keywords:
        _validate_scalar_value("param", assignment)
    for block in param_file.blocks:
        _validate_block_section("param", block)

    task_name = _find_param_value(param_file, "task")
    if task_name is None:
        return

    normalized_task = str(task_name).strip().lower()
    for name in present_names:
        spec = get_keyword_spec("param", name)
        if spec and spec.allowed_tasks and normalized_task.upper() not in {
            task.upper() for task in spec.allowed_tasks
        }:
            raise CastepValidationError(
                f"{name.upper()} is only supported by this starter implementation for "
                f"task(s) {', '.join(task.upper() for task in spec.allowed_tasks)}; "
                f"received TASK={normalized_task.upper()}."
            )


def validate_keyword_value(file: FileKind, name: str, value: object) -> None:
    """Validate one canonicalized keyword value outside a full file object."""

    spec = get_keyword_spec(file, name)
    if spec is None:
        raise UnknownKeywordError(f"Unsupported CASTEP {file} keyword '{name}'.")

    if spec.kind == "block":
        if not isinstance(value, BlockSection):
            raise CastepValidationError(f"{spec.name.upper()} requires a block value.")
        block_name = value.name.strip().lower() or spec.name
        block = value if block_name == spec.name else BlockSection(spec.name, value.rows, value.unit)
        _validate_block_section(file, block)
        return

    if spec.kind == "flag":
        if value in (False, None, ""):
            return
        _validate_scalar_value(file, KeywordAssignment(spec.name, None))
        return

    _validate_scalar_value(file, KeywordAssignment(spec.name, value))


def _cell_present_names(cell_file: CellFile) -> list[str]:
    names: list[str] = []
    if cell_file.lattice_cart is not None:
        names.append("lattice_cart")
    if cell_file.lattice_abc is not None:
        names.append("lattice_abc")
    if cell_file.positions_frac is not None:
        names.append("positions_frac")
    if cell_file.positions_abs is not None:
        names.append("positions_abs")
    if cell_file.species_pot:
        names.append("species_pot")
    if cell_file.kpoints_list:
        names.append("kpoints_list")
    if cell_file.kpoint_mp_grid is not None:
        names.append("kpoint_mp_grid")
    if cell_file.kpoint_mp_spacing is not None:
        names.append("kpoint_mp_spacing")
    if cell_file.symmetry_generate:
        names.append("symmetry_generate")
    names.extend(assignment.name.strip().lower() for assignment in cell_file.scalars)
    names.extend(block.name.strip().lower() for block in cell_file.blocks)
    return names


def _param_present_names(param_file: ParamFile) -> list[str]:
    names = [assignment.name.strip().lower() for assignment in param_file.keywords]
    names.extend(block.name.strip().lower() for block in param_file.blocks)
    return names


def _validate_known_names(file: FileKind, names: Iterable[str]) -> None:
    for name in names:
        if get_keyword_spec(file, name) is None:
            raise UnknownKeywordError(_unknown_keyword_message(file, name))


def _unknown_keyword_message(file: FileKind, name: str) -> str:
    """Explain an unknown keyword in terms the person seeing it can act on.

    On the bundled starter catalog the usual cause is not a bad file but a
    missing ``castep_keywords.json``: CASTEP writes keywords into its own output
    (``QUANTISATION_AXIS`` after a Hubbard U run, for example) that the starter
    catalog does not list. Users need pointing at CASTEP setup; only with a full
    catalog loaded is this a contributor-facing gap in the schema.
    """

    prefix = f"Unsupported CASTEP {file} keyword '{name}'. "
    if keyword_catalog_status().is_starter:
        return prefix + (
            "MuFinder is running on its limited built-in keyword set because no CASTEP keyword catalog "
            "(castep_keywords.json) is loaded. Generate or import one in Setup > CASTEP so MuFinder "
            "recognises every keyword your CASTEP version writes."
        )
    return prefix + "Add it to schema.py and docs/castep/keyword_catalog.yaml before using it."


def _validate_duplicates(file: FileKind, names: Iterable[str]) -> None:
    counter = Counter(name.strip().lower() for name in names)
    for name, count in counter.items():
        spec = get_keyword_spec(file, name)
        if spec is None:
            continue
        if count > 1 and not spec.repeatable:
            raise DuplicateKeywordError(f"{name.upper()} may only appear once in a {file} file.")


def _validate_mutual_exclusion(file: FileKind, names: Iterable[str]) -> None:
    normalized_names = [name.strip().lower() for name in names]
    normalized_set = set(normalized_names)
    present_canonical = {canonical_keyword_name(file, name) for name in normalized_names}
    raw_names_by_canonical: dict[str, set[str]] = {}
    for name in normalized_names:
        raw_names_by_canonical.setdefault(canonical_keyword_name(file, name), set()).add(name)
    for canonical, raw_names in raw_names_by_canonical.items():
        if len(raw_names) > 1:
            ordered = sorted(raw_names, key=lambda item: (item != canonical, item))
            raise ConflictingKeywordError(
                f"{ordered[0].upper()} cannot be used together with {ordered[1].upper()}."
            )
    for name in normalized_names:
        spec = get_keyword_spec(file, name)
        if spec is None:
            continue
        for other in spec.mutually_exclusive_with:
            if other in normalized_set or canonical_keyword_name(file, other) in present_canonical:
                raise ConflictingKeywordError(
                    f"{name.upper()} cannot be used together with {other.upper()}."
                )


def _validate_requirements(file: FileKind, names: Iterable[str]) -> None:
    ordered_present = _ordered_canonical_names(file, names)
    present = set(ordered_present)
    for name in ordered_present:
        spec = get_keyword_spec(file, name)
        if spec is None:  # pragma: no cover - name came from the live schema above
            continue
        for required in spec.requires:
            if required not in present:
                raise MissingRequiredKeywordError(
                    f"{name.upper()} requires {required.upper()}."
                )
        if spec.requires_any_of and not any(required in present for required in spec.requires_any_of):
            joined = " or ".join(required.upper() for required in spec.requires_any_of)
            raise MissingRequiredKeywordError(f"{name.upper()} requires {joined}.")


def _validate_lattice_cart(lattice: LatticeCart) -> None:
    if len(lattice.vectors) != 3:
        raise CastepValidationError("LATTICE_CART requires exactly three lattice vectors.")
    for index, vector in enumerate(lattice.vectors, start=1):
        if len(vector) != 3:
            raise CastepValidationError(f"LATTICE_CART vector {index} must contain three components.")
        _validate_numeric_tuple(vector, f"LATTICE_CART vector {index}")


def _validate_lattice_abc(lattice: LatticeABC) -> None:
    if len(lattice.lengths) != 3 or len(lattice.angles) != 3:
        raise CastepValidationError("LATTICE_ABC requires three lengths and three angles.")
    _validate_numeric_tuple(lattice.lengths, "LATTICE_ABC lengths", positive_only=True)
    _validate_numeric_tuple(lattice.angles, "LATTICE_ABC angles", positive_only=True)
    for angle in lattice.angles:
        if angle >= 180:
            raise CastepValidationError("LATTICE_ABC angles must be less than 180 degrees.")


def _validate_positions(block: PositionBlock, *, allow_unit: bool, block_name: str) -> None:
    if not block.positions:
        raise CastepValidationError(f"{block_name} must contain at least one position row.")
    if block.unit and not allow_unit:
        raise CastepValidationError(f"{block_name} does not support a unit line.")
    for row in block.positions:
        _validate_atomic_position(row, block_name)


def _validate_atomic_spins(cell_file: CellFile) -> None:
    if cell_file.positions_frac is None:
        raise CastepValidationError(
            "Per-atom spins require POSITIONS_FRAC because SPIN values are rendered inline."
        )

    if len(cell_file.atomic_spins) != len(cell_file.positions_frac.positions):
        raise CastepValidationError(
            "Atomic spin count must match POSITIONS_FRAC row count."
        )

    for index, value in enumerate(cell_file.atomic_spins, start=1):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CastepValidationError(
                f"Atomic spin value at row {index} must be numeric."
            )


def _validate_atomic_position(position: AtomicPosition, block_name: str) -> None:
    if not position.species.strip():
        raise CastepValidationError(f"{block_name} entries require a species label.")
    if len(position.coordinates) != 3:
        raise CastepValidationError(f"{block_name} coordinates must contain three components.")
    _validate_numeric_tuple(position.coordinates, f"{block_name} coordinates")


def _validate_species_pot(rows: tuple[SpeciesPotential, ...]) -> None:
    for row in rows:
        if not row.species.strip() or not row.potential.strip():
            raise CastepValidationError("SPECIES_POT entries require both species and pseudopotential name.")


def _validate_kpoints_list(rows: tuple[KPoint, ...]) -> None:
    if not rows:
        raise CastepValidationError("KPOINTS_LIST must contain at least one k-point row.")
    total_weight = 0.0
    for row in rows:
        if len(row.coordinates) != 3:
            raise CastepValidationError("Each KPOINTS_LIST row must contain three coordinates.")
        _validate_numeric_tuple(row.coordinates, "KPOINTS_LIST coordinates")
        total_weight += row.weight
    if not isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise CastepValidationError(
            f"KPOINTS_LIST weights must sum to 1.0; received {total_weight:.12g}."
        )


def _validate_block_section(file: FileKind, block: BlockSection) -> None:
    spec = get_keyword_spec(file, block.name)
    if spec is None:
        raise UnknownKeywordError(f"Unsupported CASTEP {file} block '{block.name}'.")
    if spec.kind != "block":
        raise CastepValidationError(f"{spec.name.upper()} is not a block keyword.")
    if not block.rows:
        raise CastepValidationError(f"{spec.name.upper()} blocks must contain at least one row.")
    schema = spec.block_schema
    if schema is not None:
        if schema.fixed_row_count is not None and len(block.rows) != schema.fixed_row_count:
            raise CastepValidationError(
                f"{spec.name.upper()} requires exactly {schema.fixed_row_count} rows."
            )
        if len(block.rows) < schema.min_rows:
            raise CastepValidationError(
                f"{spec.name.upper()} requires at least {schema.min_rows} rows."
            )
        if schema.row_group_size is not None and len(block.rows) % schema.row_group_size != 0:
            raise CastepValidationError(
                f"{spec.name.upper()} row count must be a multiple of {schema.row_group_size}."
            )
    if spec.name == "hubbard_u" and block.unit is not None and block.unit.strip().lower() not in {"ev", "ha"}:
        raise CastepValidationError("HUBBARD_U units must be eV or Ha.")
    for row in block.rows:
        if not row:
            raise CastepValidationError(f"{spec.name.upper()} block rows may not be empty.")
        if spec.name == "hubbard_u":
            _validate_hubbard_u_row(row)
            continue
        if schema is not None and schema.columns:
            if len(row) != len(schema.columns):
                raise CastepValidationError(
                    f"{spec.name.upper()} rows must contain {len(schema.columns)} values."
                )
            for value, column in zip(row, schema.columns):
                _validate_block_cell(value, column, spec.name.upper())

    if spec.name == "kpoints_list":
        total_weight = sum(float(row[-1]) for row in block.rows)
        if not isclose(total_weight, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise CastepValidationError(
                f"KPOINTS_LIST weights must sum to 1.0; received {total_weight:.12g}."
            )


def _validate_hubbard_u_row(row: tuple[ScalarPrimitive, ...]) -> None:
    if not isinstance(row[0], str) or not row[0].strip():
        raise CastepValidationError("HUBBARD_U rows must start with a species label.")

    index = 1
    if len(row) > 1 and ":" not in str(row[1]):
        atom_index = row[1]
        if not isinstance(atom_index, int) or isinstance(atom_index, bool) or atom_index <= 0:
            raise CastepValidationError("HUBBARD_U atom indices must be positive integers.")
        index = 2

    seen_orbitals: set[str] = set()
    while index < len(row):
        token = row[index]
        if not isinstance(token, str) or ":" not in token:
            raise CastepValidationError("HUBBARD_U orbital entries must use tokens such as 'd: 2.4'.")
        orbital, inline_value = token.split(":", maxsplit=1)
        orbital = orbital.strip().lower()
        if orbital not in {"s", "p", "d", "f"}:
            raise CastepValidationError("HUBBARD_U orbitals must be one of s, p, d, or f.")
        if orbital in seen_orbitals:
            raise CastepValidationError(f"HUBBARD_U orbital '{orbital}' is defined more than once in a row.")
        seen_orbitals.add(orbital)

        if inline_value.strip():
            raw_value: object = inline_value.strip()
        else:
            index += 1
            if index >= len(row):
                raise CastepValidationError(f"HUBBARD_U orbital '{orbital}' is missing a U value.")
            raw_value = row[index]

        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            try:
                float(str(raw_value))
            except ValueError as exc:
                raise CastepValidationError("HUBBARD_U values must be numeric.") from exc
        index += 1

    if not seen_orbitals:
        raise CastepValidationError("HUBBARD_U rows must define at least one orbital U value.")

def _validate_scalar_value(file: FileKind, assignment: KeywordAssignment) -> None:
    spec = get_keyword_spec(file, assignment.name)
    if spec is None:
        raise UnknownKeywordError(f"Unsupported CASTEP {file} keyword '{assignment.name}'.")
    if spec.kind == "block":
        raise CastepValidationError(f"{spec.name.upper()} must be provided as a block, not a scalar.")

    value = assignment.value
    if spec.kind == "flag":
        if value not in (None, True):
            raise CastepValidationError(f"{spec.name.upper()} is a presence-only flag and may not take a value.")
        return

    if value is None:
        raise CastepValidationError(f"{spec.name.upper()} requires an explicit value.")

    if spec.value_type == "bool":
        if not isinstance(value, bool):
            raise CastepValidationError(f"{spec.name.upper()} expects TRUE or FALSE.")
        return
    if spec.value_type == "enum":
        normalized = str(value).strip().upper()
        allowed = {item.upper() for item in spec.allowed_values}
        if allowed and normalized not in allowed:
            allowed = ", ".join(spec.allowed_values)
            raise CastepValidationError(
                f"{spec.name.upper()}='{value}' is not supported by the starter schema. "
                f"Allowed values: {allowed}."
            )
        return
    if spec.value_type == "int_triplet":
        if not isinstance(value, tuple):
            raise CastepValidationError(f"{spec.name.upper()} expects a tuple of three integers.")
        _validate_triplet(
            value,
            spec.name.upper(),
            positive_only=bool(spec.min_value is not None and float(spec.min_value) > 0),
            expected_length=spec.vector_length or 3,
        )
        return
    if spec.value_type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise CastepValidationError(f"{spec.name.upper()} expects an integer value.")
        _validate_bounds(spec, float(value))
        return
    if spec.value_type == "positive_int":
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise CastepValidationError(f"{spec.name.upper()} expects an integer greater than zero.")
        _validate_bounds(spec, float(value))
        return
    if spec.value_type == "non_negative_int":
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise CastepValidationError(f"{spec.name.upper()} expects an integer greater than or equal to zero.")
        _validate_bounds(spec, float(value))
        return
    if spec.value_type in {"float", "number"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CastepValidationError(f"{spec.name.upper()} expects a numeric value.")
        _validate_bounds(spec, float(value))
        return
    if spec.value_type in {"physical", "positive_float_with_optional_unit"}:
        numeric = value.value if isinstance(value, ValueWithUnit) else value
        if isinstance(numeric, tuple):
            raise CastepValidationError(f"{spec.name.upper()} expects a scalar numeric value.")
        if not isinstance(numeric, (int, float)) or isinstance(numeric, bool):
            raise CastepValidationError(f"{spec.name.upper()} expects a numeric value.")
        if spec.value_type == "positive_float_with_optional_unit" and numeric <= 0:
            raise CastepValidationError(f"{spec.name.upper()} expects a numeric value greater than zero.")
        _validate_bounds(spec, float(numeric))
        if isinstance(value, ValueWithUnit) and not value.unit.strip():
            raise CastepValidationError(f"{spec.name.upper()} units may not be empty.")
        if isinstance(value, ValueWithUnit) and spec.name == "elec_energy_tol":
            normalized_unit = value.unit.strip().lower().replace(" ", "")
            if "/atom" in normalized_unit:
                raise CastepValidationError(
                    "ELEC_ENERGY_TOL must use energy units (for example 'eV'), "
                    "not per-atom units such as 'eV/atom'."
                )
        return
    if spec.value_type == "real_vector":
        if not isinstance(value, tuple):
            raise CastepValidationError(f"{spec.name.upper()} expects a numeric vector value.")
        expected = spec.vector_length or 3
        if len(value) != expected:
            raise CastepValidationError(f"{spec.name.upper()} expects exactly {expected} components.")
        for item in value:
            if not isinstance(item, (int, float)) or isinstance(item, bool):
                raise CastepValidationError(f"{spec.name.upper()} expects numeric vector components.")
        return
    if spec.value_type == "text":
        if not isinstance(value, str):
            raise CastepValidationError(f"{spec.name.upper()} expects text.")
        return

    raise CastepValidationError(
        f"Unsupported validator type '{spec.value_type}' for {spec.name.upper()}."
    )


def _validate_triplet(
    value: tuple[int, int, int] | tuple[ScalarPrimitive, ...],
    label: str,
    *,
    positive_only: bool,
    expected_length: int,
) -> None:
    if len(value) != expected_length:
        raise CastepValidationError(f"{label} expects exactly {expected_length} values.")
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise CastepValidationError(f"{label} expects integer values.")
        if positive_only and item <= 0:
            raise CastepValidationError(f"{label} values must be greater than zero.")


def _validate_numeric_tuple(
    values: tuple[float, float, float] | tuple[float, float, float, float],
    label: str,
    *,
    positive_only: bool = False,
) -> None:
    for item in values:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise CastepValidationError(f"{label} expects numeric values.")
        if positive_only and item <= 0:
            raise CastepValidationError(f"{label} values must be greater than zero.")


def _find_param_value(param_file: ParamFile, name: str) -> ScalarValue | ValueWithUnit | None:
    for assignment in param_file.keywords:
        if assignment.name.strip().lower() == name:
            value = assignment.value
            if isinstance(value, ValueWithUnit):
                return value.value
            return value
    return None


def _validate_bounds(spec, value: float) -> None:
    if spec.min_value is not None and value < float(spec.min_value):
        raise CastepValidationError(
            f"{spec.name.upper()} must be greater than or equal to {spec.min_value}."
        )
    if spec.max_value is not None and value > float(spec.max_value):
        raise CastepValidationError(
            f"{spec.name.upper()} must be less than or equal to {spec.max_value}."
        )


def _validate_block_cell(value: object, column: BlockColumnSpec, label: str) -> None:
    if column.allowed_values and str(value) not in column.allowed_values:
        allowed = ", ".join(column.allowed_values)
        raise CastepValidationError(f"{label} column '{column.name}' must be one of: {allowed}.")
    if column.value_kind == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            raise CastepValidationError(f"{label} column '{column.name}' expects integers.")
    elif column.value_kind == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CastepValidationError(f"{label} column '{column.name}' expects numeric values.")
    elif column.value_kind == "bool":
        if not isinstance(value, bool):
            raise CastepValidationError(f"{label} column '{column.name}' expects booleans.")
    else:
        if not isinstance(value, str):
            raise CastepValidationError(f"{label} column '{column.name}' expects text.")


def _ordered_canonical_names(file: FileKind, names: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        spec = get_keyword_spec(file, name)
        if spec is None or spec.name in seen:
            continue
        ordered.append(spec.name)
        seen.add(spec.name)
    return ordered
