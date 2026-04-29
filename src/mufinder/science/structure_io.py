"""Input structure loading helpers backed by ASE."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
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

    atom_labels = _extract_atom_labels(tags, loops)
    propagation_vectors = _extract_propagation_vectors(tags, loops)
    propagation = propagation_vectors[0] if propagation_vectors else None
    moments = _attach_moment_atom_indices(
        _extract_moments_from_loops(loops),
        atom_labels=atom_labels,
    )

    if not propagation_vectors and not moments:
        return None

    payload: dict[str, Any] = {"source": "mcif"}
    if atom_labels:
        payload["atom_labels"] = atom_labels
    if propagation:
        payload["propagation_vector"] = propagation
    if propagation_vectors:
        payload["propagation_vectors_rlu"] = propagation_vectors
    if moments:
        payload["moments"] = moments
        payload["site_moments"] = {
            str(moment["label"]): list(moment["vector"])
            for moment in moments
            if isinstance(moment.get("label"), str) and isinstance(moment.get("vector"), list)
        }
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
        if not isinstance(moment, dict):
            continue
        atom_index = index
        if "atom_index" in moment:
            try:
                atom_index = int(moment["atom_index"])
            except (TypeError, ValueError):
                continue
        if atom_index < 0 or atom_index >= atom_count:
            continue
        vector = moment.get("vector")
        if not isinstance(vector, list) or len(vector) < 3:
            continue
        try:
            spins[atom_index] = float(vector[2])
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


def _extract_atom_labels(tags: dict[str, str], loops: list[dict[str, Any]]) -> list[str]:
    """Extract MCIF atom labels in structure order when present."""
    labels: list[str] = []
    if "_atom_site_label" in tags:
        labels.append(str(tags["_atom_site_label"]))

    for loop in loops:
        headers: list[str] = loop.get("headers", [])
        if "_atom_site_label" not in headers:
            continue
        for row in loop.get("rows", []):
            label = str(row.get("_atom_site_label", "")).strip()
            if label:
                labels.append(label)

    return _dedupe_strings(labels)


def _extract_propagation_vectors(tags: dict[str, str], loops: list[dict[str, Any]]) -> list[list[float]]:
    """Extract all known MCIF propagation vectors in reciprocal-lattice units."""
    vectors: list[list[float]] = []

    for key in ("_magnetic_propagation_vector_kxkykz", "_propagation_vector_kxkykz"):
        if key in tags:
            triplet = _parse_cif_triplet(tags[key])
            if triplet is not None:
                vectors.append(triplet)

    for loop in loops:
        headers: list[str] = loop.get("headers", [])
        key = _first_matching_header(
            headers,
            [
                "_magnetic_propagation_vector_kxkykz",
                "_propagation_vector_kxkykz",
            ],
        )
        if key is None:
            continue
        for row in loop.get("rows", []):
            triplet = _parse_cif_triplet(str(row.get(key, "")))
            if triplet is not None:
                vectors.append(triplet)

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
                vectors.append([x, y, z])
    return _dedupe_vectors(vectors)


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
                "_atom_site_moment_label",
                "_atom_site_moment_site_label",
            ],
        )
        x_key = _first_header_with_suffix(
            headers,
            [".crystalaxis_x", ".cartn_x", ".x", "_crystalaxis_x", "_cartn_x", "_x"],
        )
        y_key = _first_header_with_suffix(
            headers,
            [".crystalaxis_y", ".cartn_y", ".y", "_crystalaxis_y", "_cartn_y", "_y"],
        )
        z_key = _first_header_with_suffix(
            headers,
            [".crystalaxis_z", ".cartn_z", ".z", "_crystalaxis_z", "_cartn_z", "_z"],
        )
        if not (x_key and y_key and z_key):
            continue

        coordinate_system = "cartesian" if "cartn" in x_key else "crystalaxis"
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


def _attach_moment_atom_indices(
    moments: list[dict[str, Any]],
    *,
    atom_labels: list[str],
) -> list[dict[str, Any]]:
    """Attach MCIF atom indices using exact or unique prefix label matches."""
    if not atom_labels:
        return moments

    mapped: list[dict[str, Any]] = []
    for moment in moments:
        updated = dict(moment)
        label = str(updated.get("label", "")).strip()
        atom_index = _find_atom_label_index(atom_labels, label)
        if atom_index is not None:
            updated["atom_index"] = atom_index
            updated["atom_label"] = atom_labels[atom_index]
        mapped.append(updated)
    return mapped


def _find_atom_label_index(atom_labels: list[str], label: str) -> int | None:
    """Return the atom index for an MCIF label using mag2spectra-style matching."""
    if not label:
        return None
    if label in atom_labels:
        return atom_labels.index(label)
    prefix_matches = [index for index, candidate in enumerate(atom_labels) if candidate.startswith(f"{label}_")]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    return None


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


def _parse_cif_triplet(token: str) -> list[float] | None:
    """Parse a CIF triplet token such as ``0, 0, 1/4`` or ``0 0 0.25``."""
    text = token.strip().strip("'\"").strip("[]()")
    if text in {"", ".", "?"}:
        return None
    parts = [part.strip() for part in text.split(",")] if "," in text else text.split()
    if len(parts) != 3:
        return None
    values = [_parse_cif_float(part) for part in parts]
    if any(value is None for value in values):
        return None
    return [float(value) for value in values if value is not None]


def _parse_cif_float(token: str) -> float | None:
    """Parse numeric CIF token, removing uncertainty notation when present."""
    text = token.strip().strip("'\"")
    if text in {"", ".", "?"}:
        return None

    if "(" in text:
        text = text.split("(", maxsplit=1)[0]

    if "/" in text:
        try:
            return float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            return None

    if not re.match(r"^[+-]?(\d+\.?\d*|\d*\.\d+)([eE][+-]?\d+)?$", text):
        return None
    return float(text)


def _dedupe_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dedupe_vectors(vectors: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for vector in vectors:
        if any(all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(vector, existing, strict=True)) for existing in out):
            continue
        out.append([float(value) for value in vector])
    return out
