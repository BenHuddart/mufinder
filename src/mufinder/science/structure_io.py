"""Input structure loading helpers backed by ASE."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex
from typing import Any

from ase import Atoms
from ase.geometry import cellpar_to_cell
from ase.io import read as ase_read


class StructureLoadError(ValueError):
    """Raised when a structure file cannot be parsed."""


@dataclass(slots=True)
class LoadedStructure:
    """Parsed structure payload from an input file."""

    lattice_vectors: list[list[float]]
    species: list[str]
    fractional_positions: list[list[float]]
    properties: dict[str, Any] = field(default_factory=dict)


def load_structure_file(path: str | Path) -> LoadedStructure:
    """Load structure data from CIF, MCIF, or CASTEP CELL input.

    Supported extensions:
    - .cif
    - .mcif
    - .cell
    """
    source = Path(path)
    if not source.exists():
        raise StructureLoadError(f"Structure file does not exist: {source}")

    suffix = source.suffix.lower()
    read_format = _resolve_ase_format(suffix)
    if read_format is None:
        raise StructureLoadError(
            f"Unsupported structure extension '{suffix}'. Expected .cif, .mcif, or .cell."
        )

    if read_format == "castep-cell":
        return _load_castep_cell_file(source)

    try:
        atoms = ase_read(str(source), format=read_format)
    except Exception as exc:  # noqa: BLE001 - normalize parser exceptions.
        raise StructureLoadError(f"Failed to parse structure file '{source.name}': {exc}") from exc

    if isinstance(atoms, list):
        if not atoms:
            raise StructureLoadError(f"No structures found in file '{source.name}'.")
        atoms = atoms[0]

    if not isinstance(atoms, Atoms):
        raise StructureLoadError(f"Unsupported structure payload in '{source.name}'.")

    symbols = atoms.get_chemical_symbols()
    if not symbols:
        raise StructureLoadError(f"Structure '{source.name}' does not contain any atoms.")

    lattice_vectors = atoms.cell.array.tolist()
    fractional_positions = atoms.get_scaled_positions(wrap=True).tolist()

    magmoms = atoms.get_initial_magnetic_moments().tolist()
    properties: dict[str, Any] = {
        "source_extension": suffix,
        "atom_count": len(symbols),
        "pbc": [bool(v) for v in atoms.pbc.tolist()],
    }
    if any(abs(value) > 1e-12 for value in magmoms):
        properties["initial_magnetic_moments"] = magmoms
        properties["atomic_spins"] = [float(value) for value in magmoms]

    if suffix == ".mcif":
        magnetic_structure = _extract_mcif_magnetic_structure(source)
        if magnetic_structure:
            properties["magnetic_structure"] = magnetic_structure
            propagation_vector = magnetic_structure.get("propagation_vector")
            if isinstance(propagation_vector, list) and len(propagation_vector) == 3:
                properties["propagation_vector"] = [float(value) for value in propagation_vector]

            spins_from_moments = _spins_from_magnetic_structure(
                magnetic_structure=magnetic_structure,
                atom_count=len(symbols),
            )
            if spins_from_moments is not None:
                properties["atomic_spins"] = spins_from_moments

    return LoadedStructure(
        lattice_vectors=lattice_vectors,
        species=symbols,
        fractional_positions=fractional_positions,
        properties=properties,
    )


def _resolve_ase_format(suffix: str) -> str | None:
    if suffix == ".cif":
        return "cif"
    if suffix == ".mcif":
        # MCIF files use CIF grammar with magnetic metadata extensions.
        return "cif"
    if suffix == ".cell":
        return "castep-cell"
    return None


_CELL_UNITS = {
    "bohr": 0.529177210903,
    "ang": 1.0,
    "m": 1e10,
    "cm": 1e8,
    "nm": 10.0,
    "pm": 1e-2,
}


def _load_castep_cell_file(source: Path) -> LoadedStructure:
    text = source.read_text(encoding="utf-8")
    blocks = _extract_castep_blocks(text)

    cell = _parse_cell_lattice(blocks)
    species, scaled_positions, spins = _parse_cell_positions(blocks, cell)

    properties: dict[str, Any] = {
        "source_extension": ".cell",
        "atom_count": len(species),
        "pbc": [True, True, True],
    }
    if spins is not None:
        properties["atomic_spins"] = [float(value) for value in spins]

    return LoadedStructure(
        lattice_vectors=[[float(value) for value in row] for row in cell],
        species=species,
        fractional_positions=[[float(value) for value in row] for row in scaled_positions],
        properties=properties,
    )


def _extract_castep_blocks(text: str) -> dict[str, list[str]]:
    pattern = re.compile(
        r"%BLOCK\s+([A-Za-z0-9_]+)\s*\n(.*?)\n%ENDBLOCK\s+\1",
        re.IGNORECASE | re.DOTALL,
    )
    blocks: dict[str, list[str]] = {}
    for match in pattern.finditer(text):
        name = match.group(1).strip().lower()
        lines = [line.strip() for line in match.group(2).splitlines() if line.strip()]
        blocks[name] = lines
    return blocks


def _parse_cell_lattice(blocks: dict[str, list[str]]) -> list[list[float]]:
    if "lattice_cart" in blocks:
        return _parse_lattice_cart_block(blocks["lattice_cart"])
    if "lattice_abc" in blocks:
        return _parse_lattice_abc_block(blocks["lattice_abc"])
    raise StructureLoadError("CASTEP CELL file must contain LATTICE_CART or LATTICE_ABC.")


def _parse_lattice_cart_block(lines: list[str]) -> list[list[float]]:
    scale = 1.0
    data_lines = lines
    if len(lines) >= 4 and len(lines[0].split()) == 1:
        scale = _parse_length_unit(lines[0], block_name="LATTICE_CART")
        data_lines = lines[1:]
    if len(data_lines) < 3:
        raise StructureLoadError("LATTICE_CART must contain three lattice vectors.")
    vectors: list[list[float]] = []
    for index, line in enumerate(data_lines[:3], start=1):
        parts = line.split()
        if len(parts) < 3:
            raise StructureLoadError(f"LATTICE_CART vector {index} must contain three components.")
        vectors.append([float(parts[0]) * scale, float(parts[1]) * scale, float(parts[2]) * scale])
    return vectors


def _parse_lattice_abc_block(lines: list[str]) -> list[list[float]]:
    scale = 1.0
    data_lines = lines
    if len(lines) >= 3 and len(lines[0].split()) == 1:
        scale = _parse_length_unit(lines[0], block_name="LATTICE_ABC")
        data_lines = lines[1:]
    if len(data_lines) < 2:
        raise StructureLoadError("LATTICE_ABC must contain lengths and angles rows.")
    lengths = [float(value) * scale for value in data_lines[0].split()[:3]]
    angles = [float(value) for value in data_lines[1].split()[:3]]
    if len(lengths) != 3 or len(angles) != 3:
        raise StructureLoadError("LATTICE_ABC must provide three lengths and three angles.")
    cell = cellpar_to_cell(lengths + angles)
    return [[float(component) for component in row] for row in cell]


def _parse_cell_positions(
    blocks: dict[str, list[str]],
    cell: list[list[float]],
) -> tuple[list[str], list[list[float]], list[float] | None]:
    if "positions_frac" in blocks:
        species, frac, spins = _parse_positions_frac_block(blocks["positions_frac"])
        return species, frac, spins
    if "positions_abs" in blocks:
        species, cart = _parse_positions_abs_block(blocks["positions_abs"])
        atoms = Atoms(symbols=species, positions=cart, cell=cell, pbc=True)
        scaled = atoms.get_scaled_positions(wrap=True).tolist()
        return species, [[float(value) for value in row] for row in scaled], None
    raise StructureLoadError("CASTEP CELL file must contain POSITIONS_FRAC or POSITIONS_ABS.")


def _parse_positions_frac_block(lines: list[str]) -> tuple[list[str], list[list[float]], list[float] | None]:
    species: list[str] = []
    positions: list[list[float]] = []
    spins: list[float] = []
    saw_spin_token = False
    for line in lines:
        parts = line.split()
        if len(parts) < 4:
            raise StructureLoadError("POSITIONS_FRAC rows must contain species and three coordinates.")
        species.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

        spin_value = 0.0
        for token in parts[4:]:
            normalized = token.strip()
            if normalized.lower().startswith("spin="):
                raw_value = normalized.split("=", maxsplit=1)[1].strip()
                try:
                    spin_value = float(raw_value)
                except ValueError as exc:
                    raise StructureLoadError(
                        f"Invalid SPIN value '{raw_value}' in POSITIONS_FRAC row '{line}'."
                    ) from exc
                saw_spin_token = True
                break
        spins.append(spin_value)
    return species, positions, spins if saw_spin_token else None


def _parse_positions_abs_block(lines: list[str]) -> tuple[list[str], list[list[float]]]:
    scale = 1.0
    data_lines = lines
    if lines and len(lines[0].split()) == 1 and lines[0].lower() in _CELL_UNITS:
        scale = _parse_length_unit(lines[0], block_name="POSITIONS_ABS")
        data_lines = lines[1:]
    species: list[str] = []
    positions: list[list[float]] = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 4:
            raise StructureLoadError("POSITIONS_ABS rows must contain species and three coordinates.")
        species.append(parts[0])
        positions.append([float(parts[1]) * scale, float(parts[2]) * scale, float(parts[3]) * scale])
    return species, positions


def _parse_length_unit(token: str, *, block_name: str) -> float:
    unit = token.strip().lower()
    try:
        return _CELL_UNITS[unit]
    except KeyError as exc:
        raise StructureLoadError(f"{block_name} uses unsupported length unit '{token}'.") from exc


def _extract_mcif_magnetic_structure(source: Path) -> dict[str, Any] | None:
    """Extract magnetic metadata from MCIF text when present."""
    text = source.read_text(encoding="utf-8")
    tags, loops = _parse_cif_tags_and_loops(text)

    propagation = _extract_propagation_vector(tags)
    moments = _extract_moments_from_loops(loops)

    if not propagation and not moments:
        return None

    payload: dict[str, Any] = {}
    if propagation:
        payload["propagation_vector"] = propagation
    if moments:
        payload["moments"] = moments
    return payload


def _spins_from_magnetic_structure(
    *,
    magnetic_structure: dict[str, Any],
    atom_count: int,
) -> list[float] | None:
    moments = magnetic_structure.get("moments")
    if not isinstance(moments, list) or not moments:
        return None

    spins = [0.0] * atom_count
    saw_moment = False
    for index, moment in enumerate(moments):
        if index >= atom_count or not isinstance(moment, dict):
            break
        vector = moment.get("vector")
        if not isinstance(vector, list) or len(vector) < 3:
            continue
        try:
            spins[index] = float(vector[2])
            saw_moment = True
        except (TypeError, ValueError):
            continue

    return spins if saw_moment else None


def _parse_cif_tags_and_loops(text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Parse simple CIF/MCIF tags and loops from text.

    This parser intentionally targets the subset needed for magnetic metadata.
    """
    lines = text.splitlines()
    tags: dict[str, str] = {}
    loops: list[dict[str, Any]] = []

    idx = 0
    while idx < len(lines):
        line = _normalize_cif_line(lines[idx])
        idx += 1
        if not line:
            continue

        lower = line.lower()
        if lower == "loop_":
            headers: list[str] = []
            while idx < len(lines):
                probe = _normalize_cif_line(lines[idx])
                if not probe:
                    idx += 1
                    continue
                if probe.startswith("_"):
                    headers.append(probe.split()[0].lower())
                    idx += 1
                    continue
                break

            if not headers:
                continue

            rows: list[dict[str, str]] = []
            pending: list[str] = []
            while idx < len(lines):
                probe = _normalize_cif_line(lines[idx])
                if not probe:
                    idx += 1
                    continue
                lower_probe = probe.lower()
                if lower_probe == "loop_" or lower_probe.startswith("data_") or probe.startswith("_"):
                    break

                pending.extend(_tokenize_cif_line(probe))
                idx += 1

                while len(pending) >= len(headers):
                    row_tokens = pending[: len(headers)]
                    pending = pending[len(headers) :]
                    rows.append({headers[col]: row_tokens[col] for col in range(len(headers))})

            loops.append({"headers": headers, "rows": rows})
            continue

        if line.startswith("_"):
            parts = _tokenize_cif_line(line)
            key = parts[0].lower()
            value = parts[1] if len(parts) > 1 else ""
            tags[key] = value

    return tags, loops


def _normalize_cif_line(line: str) -> str:
    """Strip comments and surrounding whitespace from CIF line."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return ""
    if "#" in stripped:
        stripped = stripped.split("#", maxsplit=1)[0].strip()
    return stripped


def _tokenize_cif_line(line: str) -> list[str]:
    """Tokenize one CIF line while preserving quoted strings."""
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def _extract_propagation_vector(tags: dict[str, str]) -> list[float] | None:
    """Extract propagation vector if present in MCIF tags."""
    candidates = [
        ("_cell_wave_vector_x", "_cell_wave_vector_y", "_cell_wave_vector_z"),
        ("_propagation_vector_x", "_propagation_vector_y", "_propagation_vector_z"),
        (
            "_parent_propagation_vector.kx",
            "_parent_propagation_vector.ky",
            "_parent_propagation_vector.kz",
        ),
    ]

    for kx, ky, kz in candidates:
        if kx in tags and ky in tags and kz in tags:
            x = _parse_cif_float(tags[kx])
            y = _parse_cif_float(tags[ky])
            z = _parse_cif_float(tags[kz])
            if x is not None and y is not None and z is not None:
                return [x, y, z]
    return None


def _extract_moments_from_loops(loops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract site moment vectors from MCIF loops."""
    extracted: list[dict[str, Any]] = []

    for loop in loops:
        headers: list[str] = loop.get("headers", [])
        if not any(header.startswith("_atom_site_moment") for header in headers):
            continue

        label_key = _first_matching_header(
            headers,
            [
                "_atom_site_moment.label",
                "_atom_site_moment.atom_site_label",
            ],
        )
        x_key = _first_header_with_suffix(headers, [".crystalaxis_x", ".cartn_x", ".x"])
        y_key = _first_header_with_suffix(headers, [".crystalaxis_y", ".cartn_y", ".y"])
        z_key = _first_header_with_suffix(headers, [".crystalaxis_z", ".cartn_z", ".z"])
        if not (x_key and y_key and z_key):
            continue

        coordinate_system = "cartesian" if ".cartn_" in x_key else "crystalaxis"
        for row_index, row in enumerate(loop.get("rows", []), start=1):
            x = _parse_cif_float(str(row.get(x_key, "")))
            y = _parse_cif_float(str(row.get(y_key, "")))
            z = _parse_cif_float(str(row.get(z_key, "")))
            if x is None or y is None or z is None:
                continue

            label = str(row.get(label_key, f"moment_{row_index}")) if label_key else f"moment_{row_index}"
            extracted.append(
                {
                    "label": label,
                    "vector": [x, y, z],
                    "coordinate_system": coordinate_system,
                }
            )

    return extracted


def _first_matching_header(headers: list[str], options: list[str]) -> str | None:
    for option in options:
        if option in headers:
            return option
    return None


def _first_header_with_suffix(headers: list[str], suffixes: list[str]) -> str | None:
    for header in headers:
        for suffix in suffixes:
            if header.endswith(suffix):
                return header
    return None


def _parse_cif_float(token: str) -> float | None:
    """Parse numeric CIF token, removing uncertainty notation when present."""
    text = token.strip().strip("'\"")
    if text in {"", ".", "?"}:
        return None

    if "(" in text:
        text = text.split("(", maxsplit=1)[0]

    if not re.match(r"^[+-]?(\d+\.?\d*|\d*\.\d+)([eE][+-]?\d+)?$", text):
        return None
    return float(text)
