"""Parser for CASTEP ``.cell`` and ``-out.cell`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
from typing import Any

import numpy as np
from ase.geometry import cellpar_to_cell

from .schema import canonical_keyword_name, get_keyword_spec
from .types import (
    AtomicPosition,
    BlockSection,
    CellFile,
    KPoint,
    LatticeABC,
    LatticeCart,
    PositionBlock,
    SpeciesPotential,
    ValueWithUnit,
)
from .validators import validate_cell_file


_CELL_UNITS = {
    "bohr": 0.529177210903,
    "ang": 1.0,
    "m": 1e10,
    "cm": 1e8,
    "nm": 10.0,
    "pm": 1e-2,
}
# A wrapped ``POSITIONS_ABS`` coordinate this close below 1.0 is an atom on a
# cell face and is placed at 0.0. Far above the round-off of the conversion
# (~1e-16) and far below any physical displacement (1e-12 of a lattice vector).
_CELL_FACE_TOLERANCE = 1e-12
_BLOCK_START_RE = re.compile(r"^\s*%BLOCK\s+([A-Za-z0-9_]+)\s*$", re.IGNORECASE)
_BLOCK_END_RE = re.compile(r"^\s*%ENDBLOCK\s+([A-Za-z0-9_]+)\s*$", re.IGNORECASE)
_STRUCTURED_BLOCKS = frozenset(
    {"lattice_cart", "lattice_abc", "positions_frac", "positions_abs", "species_pot", "kpoints_list"}
)
_GEOMETRY_BLOCKS = frozenset({"lattice_cart", "lattice_abc", "positions_frac", "positions_abs"})


class CellParseError(ValueError):
    """Raised when a CASTEP CELL file cannot be parsed."""


@dataclass(frozen=True, slots=True)
class ParsedCell:
    """Structured parsed CELL payload with geometry convenience fields."""

    cell_file: CellFile
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    species: tuple[str, ...]
    fractional_positions: tuple[tuple[float, float, float], ...]
    atomic_spins: tuple[float, ...] | None = None


def parse_cell_file(path: str | Path, *, geometry_only: bool = False) -> ParsedCell:
    """Parse a CASTEP ``.cell`` or ``-out.cell`` file from disk.

    ``geometry_only`` has the meaning documented on :func:`parse_cell_text`.
    """

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise CellParseError(f"Failed to read CELL file '{source}': {exc}") from exc
    return parse_cell_text(text, geometry_only=geometry_only)


def parse_cell_text(text: str, *, geometry_only: bool = False) -> ParsedCell:
    """Parse CASTEP CELL text into structured dataclasses.

    By default every scalar keyword and block is parsed and the whole file is
    validated against the loaded keyword catalog, so a keyword outside the
    catalog raises :class:`CellParseError`.

    With ``geometry_only=True`` only the lattice (``LATTICE_CART`` /
    ``LATTICE_ABC``) and position (``POSITIONS_FRAC`` / ``POSITIONS_ABS``)
    blocks are read and validated. Scalar keywords and every other block are
    skipped without being parsed or checked against the catalog, and the
    returned ``cell_file`` carries the geometry only. Block delimiters are
    still checked for the whole text. Structure import uses this mode: it
    needs the crystal, not the calculation settings a CASTEP ``-out.cell``
    carries alongside it.
    """

    known_blocks, unknown_blocks, scalar_lines = _partition_cell_text(
        text, geometry_only=geometry_only
    )

    lattice_cart = _parse_lattice_cart(known_blocks.get("lattice_cart"))
    lattice_abc = _parse_lattice_abc(known_blocks.get("lattice_abc"))
    if lattice_cart is not None and lattice_abc is not None:
        raise CellParseError("CELL text may not contain both LATTICE_CART and LATTICE_ABC.")

    if lattice_cart is None and lattice_abc is None:
        raise CellParseError("CELL text must contain LATTICE_CART or LATTICE_ABC.")

    lattice_vectors = lattice_cart.vectors if lattice_cart is not None else _lattice_from_abc(lattice_abc)

    positions_frac, positions_abs, spins = _parse_positions(
        positions_frac_lines=known_blocks.get("positions_frac"),
        positions_abs_lines=known_blocks.get("positions_abs"),
        lattice_vectors=lattice_vectors,
    )

    species_pot = _parse_species_pot(known_blocks.get("species_pot"))
    kpoints_list = _parse_kpoints_list(known_blocks.get("kpoints_list"))

    kpoint_mp_grid: tuple[int, int, int] | None = None
    kpoint_mp_spacing: float | ValueWithUnit | None = None
    symmetry_generate = False
    extra_scalars: list[Any] = []
    for line in scalar_lines:
        assignment = _parse_scalar_assignment(line)
        if assignment is None:
            continue
        canonical = canonical_keyword_name("cell", assignment["name"])
        value = assignment["value"]
        if canonical == "kpoint_mp_grid":
            if not isinstance(value, tuple) or len(value) != 3:
                raise CellParseError("KPOINT_MP_GRID requires three integer values.")
            kpoint_mp_grid = (int(value[0]), int(value[1]), int(value[2]))
            continue
        if canonical == "kpoint_mp_spacing":
            if isinstance(value, tuple):
                raise CellParseError("KPOINT_MP_SPACING requires one numeric value.")
            kpoint_mp_spacing = value
            continue
        if canonical == "symmetry_generate":
            symmetry_generate = True if value is None else bool(value)
            continue
        extra_scalars.append(assignment)

    cell_file = CellFile(
        lattice_cart=lattice_cart,
        lattice_abc=lattice_abc,
        positions_frac=positions_frac,
        positions_abs=positions_abs,
        atomic_spins=spins,
        species_pot=species_pot,
        kpoints_list=kpoints_list,
        kpoint_mp_grid=kpoint_mp_grid,
        kpoint_mp_spacing=kpoint_mp_spacing,
        symmetry_generate=symmetry_generate,
        scalars=tuple(
            _to_keyword_assignment(item["name"], item["value"])
            for item in extra_scalars
        ),
        blocks=tuple(
            BlockSection(name=name, rows=rows, unit=unit)
            for name, rows, unit in unknown_blocks
        ),
    )

    try:
        validate_cell_file(cell_file)
    except ValueError as exc:
        raise CellParseError(str(exc)) from exc

    if positions_frac is not None:
        species = tuple(position.species for position in positions_frac.positions)
        fractional = tuple(position.coordinates for position in positions_frac.positions)
    else:
        assert positions_abs is not None
        species = tuple(position.species for position in positions_abs.positions)
        fractional = _fractional_from_cartesian(
            [position.coordinates for position in positions_abs.positions],
            lattice_vectors,
        )

    return ParsedCell(
        cell_file=cell_file,
        lattice_vectors=lattice_vectors,
        species=species,
        fractional_positions=fractional,
        atomic_spins=spins,
    )


def _fractional_from_cartesian(
    coordinates: list[tuple[float, float, float]],
    lattice_vectors: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ],
) -> tuple[tuple[float, float, float], ...]:
    """Convert absolute positions to wrapped fractional coordinates.

    Deliberately arithmetic rather than a round trip through ``ase.Atoms``:
    ASE resolves every symbol against the periodic table, so a CASTEP species
    *label* such as ``H:mu`` raises ``KeyError`` before any conversion happens.
    That made muonated cells written with ``POSITIONS_ABS`` unparseable, even
    though nothing about the conversion needs chemical identity.

    Raises:
        CellParseError: If the lattice is singular and cannot be inverted.
    """

    if not coordinates:
        return ()

    lattice = np.asarray(lattice_vectors, dtype=float)
    cartesian = np.asarray(coordinates, dtype=float)
    try:
        fractional = cartesian @ np.linalg.inv(lattice)
    except np.linalg.LinAlgError as exc:
        raise CellParseError(f"CELL lattice vectors are singular and cannot be inverted: {exc}") from exc

    # An atom on a cell face of a non-orthogonal lattice converts to a
    # coordinate a rounding error below an integer, which wraps to 1.0 or to
    # just under it. How far below depends on the platform's ``cos`` and LAPACK,
    # so snap within a tolerance rather than relying on a second ``mod``.
    wrapped = fractional - np.floor(fractional)
    wrapped[wrapped > 1.0 - _CELL_FACE_TOLERANCE] = 0.0
    return tuple(tuple(float(value) for value in row) for row in wrapped.tolist())


def _partition_cell_text(
    text: str,
    *,
    geometry_only: bool = False,
) -> tuple[
    dict[str, list[str]],
    list[tuple[str, tuple[tuple[Any, ...], ...], str | None]],
    list[str],
]:
    known_blocks: dict[str, list[str]] = {}
    unknown_blocks: list[tuple[str, tuple[tuple[Any, ...], ...], str | None]] = []
    scalar_lines: list[str] = []

    current_name: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        block_start = _BLOCK_START_RE.match(line)
        if block_start is not None:
            if current_name is not None:
                raise CellParseError(f"Nested %BLOCK is not allowed ({current_name}).")
            current_name = canonical_keyword_name("cell", block_start.group(1).strip())
            current_lines = []
            continue

        block_end = _BLOCK_END_RE.match(line)
        if block_end is not None:
            if current_name is None:
                raise CellParseError("Found %ENDBLOCK without matching %BLOCK.")
            end_name = canonical_keyword_name("cell", block_end.group(1).strip())
            if end_name != current_name:
                raise CellParseError(
                    f"Mismatched block terminator %ENDBLOCK {end_name} for %BLOCK {current_name}."
                )
            kept_blocks = _GEOMETRY_BLOCKS if geometry_only else _STRUCTURED_BLOCKS
            if current_name in kept_blocks:
                if current_name in known_blocks:
                    raise CellParseError(f"Duplicate block {current_name.upper()} is not supported.")
                known_blocks[current_name] = list(current_lines)
            elif not geometry_only:
                unit, rows = _parse_generic_block_rows(current_lines)
                unknown_blocks.append((current_name, rows, unit))
            current_name = None
            current_lines = []
            continue

        if current_name is not None:
            stripped = _strip_comment(line)
            if stripped:
                current_lines.append(stripped)
            continue

        stripped = _strip_comment(line)
        if stripped and not geometry_only:
            scalar_lines.append(stripped)

    if current_name is not None:
        raise CellParseError(f"Unterminated block %BLOCK {current_name}.")

    return known_blocks, unknown_blocks, scalar_lines


def _split_length_unit(lines: list[str], block_name: str) -> tuple[float, str | None, list[str]]:
    """Split an optional leading length-unit line off a block.

    Returns the scale to Angstrom, the unit token as written (``None`` when the
    block has no unit line) and the remaining data rows. A single non-numeric
    token that is not a supported unit raises instead of being read as a row.
    """

    if not lines or len(lines[0].split()) != 1:
        return 1.0, None, list(lines)
    token = lines[0]
    scale = _CELL_UNITS.get(token.lower())
    if scale is not None:
        return scale, token, list(lines[1:])
    try:
        float(token)
    except ValueError:
        raise CellParseError(f"{block_name} uses unsupported length unit '{token}'.") from None
    return 1.0, None, list(lines)


def _parse_lattice_cart(lines: list[str] | None) -> LatticeCart | None:
    if lines is None:
        return None
    scale, unit, data_lines = _split_length_unit(lines, "LATTICE_CART")
    if len(data_lines) < 3:
        raise CellParseError("LATTICE_CART must contain three lattice vectors.")
    vectors: list[tuple[float, float, float]] = []
    for line in data_lines[:3]:
        parts = line.split()
        if len(parts) < 3:
            raise CellParseError("LATTICE_CART rows must contain three values.")
        vectors.append((float(parts[0]) * scale, float(parts[1]) * scale, float(parts[2]) * scale))
    return LatticeCart(vectors=(vectors[0], vectors[1], vectors[2]), unit=unit)


def _parse_lattice_abc(lines: list[str] | None) -> LatticeABC | None:
    if lines is None:
        return None
    scale, unit, data_lines = _split_length_unit(lines, "LATTICE_ABC")
    if len(data_lines) < 2:
        raise CellParseError("LATTICE_ABC must contain lengths and angles rows.")
    lengths_parts = data_lines[0].split()
    angles_parts = data_lines[1].split()
    if len(lengths_parts) < 3 or len(angles_parts) < 3:
        raise CellParseError("LATTICE_ABC rows must contain three values.")
    lengths = (
        float(lengths_parts[0]) * scale,
        float(lengths_parts[1]) * scale,
        float(lengths_parts[2]) * scale,
    )
    angles = (float(angles_parts[0]), float(angles_parts[1]), float(angles_parts[2]))
    return LatticeABC(lengths=lengths, angles=angles, unit=unit)


def _lattice_from_abc(lattice: LatticeABC | None) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    if lattice is None:
        raise CellParseError("Missing LATTICE_ABC definition.")
    cell = cellpar_to_cell([
        float(lattice.lengths[0]),
        float(lattice.lengths[1]),
        float(lattice.lengths[2]),
        float(lattice.angles[0]),
        float(lattice.angles[1]),
        float(lattice.angles[2]),
    ])
    return (
        (float(cell[0][0]), float(cell[0][1]), float(cell[0][2])),
        (float(cell[1][0]), float(cell[1][1]), float(cell[1][2])),
        (float(cell[2][0]), float(cell[2][1]), float(cell[2][2])),
    )


def _parse_positions(
    *,
    positions_frac_lines: list[str] | None,
    positions_abs_lines: list[str] | None,
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
) -> tuple[PositionBlock | None, PositionBlock | None, tuple[float, ...] | None]:
    if positions_frac_lines is not None and positions_abs_lines is not None:
        raise CellParseError("CELL text may not contain both POSITIONS_FRAC and POSITIONS_ABS.")
    if positions_frac_lines is None and positions_abs_lines is None:
        raise CellParseError("CELL text must contain POSITIONS_FRAC or POSITIONS_ABS.")

    if positions_frac_lines is not None:
        positions: list[AtomicPosition] = []
        spins: list[float] = []
        saw_spin = False
        for line in positions_frac_lines:
            parts = shlex.split(line)
            if len(parts) < 4:
                raise CellParseError("POSITIONS_FRAC rows must contain species and three coordinates.")
            species = parts[0]
            coordinates = (float(parts[1]), float(parts[2]), float(parts[3]))
            positions.append(AtomicPosition(species=species, coordinates=coordinates))

            spin_value = 0.0
            tail = parts[4:]
            for index, token in enumerate(tail):
                if token.lower().startswith("spin="):
                    spin_raw = token.split("=", maxsplit=1)[1].strip()
                    if spin_raw == "" and index + 1 < len(tail):
                        spin_raw = tail[index + 1].strip()
                    if spin_raw == "":
                        # CASTEP can emit an empty SPIN marker for non-spin-polarized rows.
                        # Treat this as "no explicit spin" rather than a hard parse failure.
                        break
                    try:
                        spin_value = float(spin_raw)
                    except ValueError as exc:
                        raise CellParseError(f"Invalid SPIN value '{spin_raw}' in POSITIONS_FRAC row.") from exc
                    saw_spin = True
                    break
            spins.append(spin_value)

        return (
            PositionBlock(positions=tuple(positions)),
            None,
            tuple(spins) if saw_spin else None,
        )

    assert positions_abs_lines is not None
    scale, unit, data_lines = _split_length_unit(positions_abs_lines, "POSITIONS_ABS")

    positions_abs: list[AtomicPosition] = []
    for line in data_lines:
        parts = shlex.split(line)
        if len(parts) < 4:
            raise CellParseError("POSITIONS_ABS rows must contain species and three coordinates.")
        positions_abs.append(
            AtomicPosition(
                species=parts[0],
                coordinates=(
                    float(parts[1]) * scale,
                    float(parts[2]) * scale,
                    float(parts[3]) * scale,
                ),
            )
        )

    return (None, PositionBlock(positions=tuple(positions_abs), unit=unit), None)


def _parse_species_pot(lines: list[str] | None) -> tuple[SpeciesPotential, ...]:
    if lines is None:
        return ()
    rows: list[SpeciesPotential] = []
    for line in lines:
        parts = shlex.split(line)
        if len(parts) < 2:
            raise CellParseError("SPECIES_POT rows must contain species and potential tokens.")
        rows.append(SpeciesPotential(species=parts[0], potential=parts[1]))
    return tuple(rows)


def _parse_kpoints_list(lines: list[str] | None) -> tuple[KPoint, ...]:
    if lines is None:
        return ()
    rows: list[KPoint] = []
    for line in lines:
        parts = shlex.split(line)
        if len(parts) < 4:
            raise CellParseError("KPOINTS_LIST rows must contain kx ky kz and weight.")
        rows.append(
            KPoint(
                coordinates=(float(parts[0]), float(parts[1]), float(parts[2])),
                weight=float(parts[3]),
            )
        )
    return tuple(rows)


def _parse_scalar_assignment(line: str) -> dict[str, object] | None:
    if ":" in line:
        name_token, value_token = line.split(":", maxsplit=1)
        raw_name = name_token.strip()
        raw_value = value_token.strip()
    else:
        parts = line.split(maxsplit=1)
        if not parts:
            return None
        raw_name = parts[0].strip()
        raw_value = parts[1].strip() if len(parts) > 1 else ""

    if not raw_name:
        return None

    canonical = canonical_keyword_name("cell", raw_name)
    spec = get_keyword_spec("cell", canonical)
    if raw_value == "":
        value: object | None = None
    else:
        value = _parse_scalar_value(raw_value, spec)

    return {"name": canonical, "value": value}


def _parse_scalar_value(raw_value: str, spec) -> object:
    if spec is not None and spec.value_type == "text":
        return raw_value.strip()

    tokens = shlex.split(raw_value)
    if not tokens:
        return None

    if spec is not None and spec.value_type == "bool":
        token = tokens[0].strip().lower()
        if token in {"true", "t", ".true.", "yes", "1"}:
            return True
        if token in {"false", "f", ".false.", "no", "0"}:
            return False
        raise CellParseError(f"Invalid boolean value '{tokens[0]}' for {spec.name.upper()}.")

    if spec is not None and spec.value_type == "int_triplet":
        if len(tokens) < 3:
            raise CellParseError(f"{spec.name.upper()} requires three integer values.")
        return (int(tokens[0]), int(tokens[1]), int(tokens[2]))

    if spec is not None and spec.value_type in {"physical", "positive_float_with_optional_unit"}:
        if len(tokens) == 1:
            return float(tokens[0])
        if len(tokens) >= 2:
            numeric = float(tokens[0])
            unit = tokens[1]
            return ValueWithUnit(value=numeric, unit=unit)

    if spec is not None and spec.value_type == "int":
        return int(tokens[0])

    if spec is not None and spec.value_type in {"float", "number"}:
        return float(tokens[0])

    if len(tokens) == 1:
        return _parse_generic_token(tokens[0])

    return tuple(_parse_generic_token(token) for token in tokens)


def _parse_generic_block_rows(lines: list[str]) -> tuple[str | None, tuple[tuple[Any, ...], ...]]:
    unit: str | None = None
    data_lines = list(lines)
    if data_lines and len(data_lines[0].split()) == 1 and _looks_like_unit(data_lines[0]):
        unit = data_lines[0]
        data_lines = data_lines[1:]

    rows: list[tuple[Any, ...]] = []
    for line in data_lines:
        tokens = shlex.split(line)
        if not tokens:
            continue
        rows.append(tuple(_parse_generic_token(token) for token in tokens))
    return unit, tuple(rows)


def _to_keyword_assignment(name: str, value: object):
    from .types import KeywordAssignment

    return KeywordAssignment(name=name, value=value)


def _parse_generic_token(token: str) -> object:
    lowered = token.strip().lower()
    if lowered in {"true", ".true.", "yes"}:
        return True
    if lowered in {"false", ".false.", "no"}:
        return False

    try:
        if re.fullmatch(r"[+-]?\d+", token):
            return int(token)
        return float(token)
    except ValueError:
        return token


def _looks_like_unit(token: str) -> bool:
    lowered = token.strip().lower()
    return lowered in _CELL_UNITS or lowered in {"ev", "ha"} or "/" in lowered


def _strip_comment(line: str) -> str:
    for marker in ("!", "#"):
        if marker in line:
            line = line.split(marker, maxsplit=1)[0]
    return line.strip()
