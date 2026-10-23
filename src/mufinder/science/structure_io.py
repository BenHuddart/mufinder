"""Structure file loading (CIF and MCIF through ASE, CASTEP CELL through ``castep_io``) and CIF export."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
import io
from pathlib import Path
import re
from typing import Any

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write

from mufinder.castep_io.cell_parser import parse_cell_file
from mufinder.domain.magnetic_properties import atomic_spins_from_magnetic_moments
from mufinder.domain.models import Structure
from mufinder.science.lattice import nearest_image_distance


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
        # ASE decodes CIF bytes as Latin-1, so this round trip is lossless.
        cif_text = _cif_text_for_ase(source.read_bytes().decode("latin-1"))
        atoms = ase_read(io.BytesIO(cif_text.encode("latin-1")), format=read_format)
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

            spins_from_moments = atomic_spins_from_magnetic_moments(
                magnetic_structure,
                atom_count=len(symbols),
                honour_atom_index=True,
            )
            if spins_from_moments is not None:
                properties["atomic_spins"] = spins_from_moments

    return LoadedStructure(
        lattice_vectors=lattice_vectors,
        species=symbols,
        fractional_positions=fractional_positions,
        properties=properties,
    )


@dataclass(frozen=True, slots=True)
class CifMuonSite:
    """One muon site to render into an exported CIF alongside its host."""

    label: str
    fractional_position: tuple[float, float, float]
    relative_energy_ev: float | None = None
    wyckoff_label: str | None = None


#: Below this separation, a muon site is coincident with a host atom rather
#: than merely nearby. Real muon stopping sites sit roughly 1 Angstrom from
#: the ion they associate with (e.g. Mu-F, Mu-O); a sub-0.5 Angstrom contact
#: indicates a folding/provenance problem upstream rather than a real site,
#: and most CIF viewers will render such an overlap as a single merged,
#: fractionally-occupied site (e.g. "Sr1/Mu1") rather than two atoms.
_MIN_PLAUSIBLE_MUON_HOST_DISTANCE_ANG = 0.5


def export_structure_with_muon_sites(
    host_structure: Structure,
    muon_sites: list[CifMuonSite],
    output_path: str | Path,
) -> list[str]:
    """Write a host structure plus muon sites to a CIF file.

    Muons are written as hydrogen (``H``) atoms - the closest standard CIF
    element - with distinct ``_atom_site_label`` values (``site.label``, e.g.
    ``Mu1``) so they remain identifiable from the host's own hydrogen atoms.
    Muon sites generally break the host's symmetry, so the file is written in
    P1 with every atom listed explicitly rather than compressing to a smaller
    asymmetric unit.

    Returns a list of human-readable warnings for any muon site written
    implausibly close to (or coincident with) a host atom. The file is still
    written in that case: many downstream CIF viewers will display such a
    pair as a single merged, fractionally-occupied site (e.g. "Sr1/Mu1")
    even though this file assigns each atom full occupancy.
    """
    host_species = list(host_structure.species)
    host_positions = [
        [float(value) for value in row] for row in host_structure.fractional_positions
    ]
    lattice_vectors = host_structure.lattice_vectors

    species = host_species + ["H"] * len(muon_sites)
    positions = host_positions + [
        [float(value) for value in site.fractional_position] for site in muon_sites
    ]

    atoms = Atoms(
        symbols=species,
        scaled_positions=positions,
        cell=[[float(value) for value in row] for row in lattice_vectors],
        pbc=True,
    )

    species_counts: dict[str, int] = {}
    host_labels: list[str] = []
    for symbol in host_species:
        species_counts[symbol] = species_counts.get(symbol, 0) + 1
        host_labels.append(f"{symbol}{species_counts[symbol]}")
    labels = host_labels + [site.label for site in muon_sites]

    warnings: list[str] = []
    for site in muon_sites:
        for host_index, (host_symbol, host_label, host_fractional) in enumerate(
            zip(host_species, host_labels, host_positions, strict=True)
        ):
            distance = nearest_image_distance(
                site.fractional_position, host_fractional, lattice_vectors
            )
            if distance < _MIN_PLAUSIBLE_MUON_HOST_DISTANCE_ANG:
                warnings.append(
                    f"{site.label} is only {distance:.3f} Å from host atom {host_label} "
                    f"({host_symbol}); many CIF viewers will render this as a single merged, "
                    f"fractionally-occupied site rather than two distinct atoms."
                )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The _atom_site loop is left as plain, standard CIF (no extra columns):
    # mixing a MuFinder-specific column into it makes some readers (observed
    # with VESTA) miscount that loop's column width, silently shifting every
    # atom's fields by one and corrupting the whole file. Per-muon metadata
    # instead goes in its own standalone loop below, keyed by
    # _mufinder_muon_site_label, which a reader that does not recognise it can
    # simply skip without touching atom parsing.
    ase_write(destination, atoms, format="cif", labels=[labels])
    if muon_sites:
        with destination.open("a", encoding="utf-8") as handle:
            handle.write("\nloop_\n")
            handle.write("  _mufinder_muon_site_label\n")
            handle.write("  _mufinder_relative_energy_eV\n")
            handle.write("  _mufinder_wyckoff_label\n")
            for site in muon_sites:
                energy = (
                    f"{site.relative_energy_ev:.5f}" if site.relative_energy_ev is not None else "?"
                )
                wyckoff = site.wyckoff_label or "?"
                handle.write(f"  {site.label}  {energy}  {wyckoff}\n")
    return warnings


def _resolve_ase_format(suffix: str) -> str | None:
    if suffix == ".cif":
        return "cif"
    if suffix == ".mcif":
        # MCIF files use CIF grammar with magnetic metadata extensions.
        return "cif"
    if suffix == ".cell":
        return "castep-cell"
    return None


def _load_castep_cell_file(source: Path) -> LoadedStructure:
    """Load the geometry of a CASTEP CELL file through :mod:`mufinder.castep_io.cell_parser`.

    Only the lattice and position blocks are read, so calculation keywords a
    CASTEP ``-out.cell`` carries (or keywords outside the loaded catalog) do
    not block the import. ``atomic_spins`` is present only when a row carried
    a ``SPIN=`` value.
    """
    try:
        parsed = parse_cell_file(source, geometry_only=True)
    except ValueError as exc:  # CellParseError, malformed numbers, undecodable text.
        raise StructureLoadError(f"Failed to parse structure file '{source.name}': {exc}") from exc

    properties: dict[str, Any] = {
        "source_extension": ".cell",
        "atom_count": len(parsed.species),
        "pbc": [True, True, True],
    }
    if parsed.atomic_spins is not None:
        properties["atomic_spins"] = [float(value) for value in parsed.atomic_spins]

    return LoadedStructure(
        lattice_vectors=[[float(value) for value in row] for row in parsed.lattice_vectors],
        species=list(parsed.species),
        fractional_positions=[
            [float(value) for value in row] for row in parsed.fractional_positions
        ],
        properties=properties,
    )


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


def _parse_cif_tags_and_loops(text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Parse simple CIF/MCIF tags and loops from text.

    This parser intentionally targets the subset needed for magnetic metadata.
    """
    tokens = list(_iter_cif_tokens(text))
    tags: dict[str, str] = {}
    loops: list[dict[str, Any]] = []

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        idx += 1
        if token.is_tag:
            value = ""
            if idx < len(tokens) and tokens[idx].is_value:
                value = tokens[idx].text
                idx += 1
            tags[token.text.lower()] = value
            continue
        if token.quoted or token.text.lower() != "loop_":
            continue

        headers: list[str] = []
        while idx < len(tokens) and tokens[idx].is_tag:
            headers.append(tokens[idx].text.lower())
            idx += 1
        values: list[str] = []
        while idx < len(tokens) and tokens[idx].is_value:
            values.append(tokens[idx].text)
            idx += 1
        if not headers:
            continue

        width = len(headers)
        rows = [
            dict(zip(headers, values[start : start + width], strict=True))
            for start in range(0, len(values) - width + 1, width)
        ]
        loops.append({"headers": headers, "rows": rows})

    return tags, loops


_CIF_WHITESPACE = " \t"
_CIF_RESERVED_WORDS = frozenset({"loop_", "global_", "stop_"})
_CIF_RESERVED_PREFIXES = ("data_", "save_")


@dataclass(frozen=True, slots=True)
class _CifToken:
    """One CIF token; ``quoted`` marks quoted strings and text fields, which are always values."""

    text: str
    quoted: bool = False

    @property
    def is_tag(self) -> bool:
        return not self.quoted and self.text.startswith("_")

    @property
    def is_value(self) -> bool:
        if self.quoted:
            return True
        lower = self.text.lower()
        return not (
            self.text.startswith("_")
            or lower in _CIF_RESERVED_WORDS
            or lower.startswith(_CIF_RESERVED_PREFIXES)
        )


def _split_cif_lines(text: str) -> list[str]:
    """Split CIF text into lines, accepting LF, CRLF and CR line terminators."""
    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _lex_cif_line(line: str) -> tuple[list[_CifToken], int | None]:
    """Split one CIF line outside a text field into tokens.

    Returns the tokens and the column where a comment starts, or ``None``. As
    in the CIF grammar, ``#`` starts a comment only where a token could start:
    a hash inside a quoted string or within an unquoted value (``C#1``) is
    data. A quoted string closes at its delimiter when that is followed by
    whitespace or the end of the line, so ``'O'1'`` is one value.
    """
    tokens: list[_CifToken] = []
    idx = 0
    while idx < len(line):
        char = line[idx]
        if char in _CIF_WHITESPACE:
            idx += 1
        elif char == "#":
            return tokens, idx
        elif char in "'\"":
            end = idx + 1
            while end < len(line) and not (
                line[end] == char and (end + 1 == len(line) or line[end + 1] in _CIF_WHITESPACE)
            ):
                end += 1
            tokens.append(_CifToken(line[idx + 1 : end], quoted=True))
            idx = end + 1
        else:
            end = idx
            while end < len(line) and line[end] not in _CIF_WHITESPACE:
                end += 1
            tokens.append(_CifToken(line[idx:end]))
            idx = end
    return tokens, None


def _iter_cif_tokens(text: str) -> Iterator[_CifToken]:
    """Yield the tokens of CIF text, with each semicolon text field as one quoted token."""
    lines = _split_cif_lines(text)
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if line.startswith(";"):
            field_lines = [line[1:]]
            while idx < len(lines) and not lines[idx].startswith(";"):
                field_lines.append(lines[idx])
                idx += 1
            yield _CifToken("\n".join(field_lines), quoted=True)
            if idx >= len(lines):
                return
            # Text after the closing semicolon is ordinary CIF.
            line = lines[idx][1:]
            idx += 1
        yield from _lex_cif_line(line)[0]


def _cif_text_for_ase(text: str) -> str:
    """Return CIF text whose loop headings ASE's built-in CIF reader accepts.

    ASE's reader takes a loop heading only from a line holding that one tag
    and nothing else, so a comment beside a heading, or a comment or blank
    line between headings, ends the heading list early and the next line is
    rejected as an unexpected entry. CIF allows all three. Within a loop's
    heading list this strips trailing comments and drops comment-only and
    blank lines; every other line, including all text-field content, is kept
    as written. Line terminators become LF, the only one ASE splits on.
    """
    lines: list[str] = []
    in_text_field = False
    in_loop_headings = False
    for line in _split_cif_lines(text):
        if line.startswith(";"):
            in_text_field = not in_text_field
            in_loop_headings = False
            lines.append(line)
            continue
        if in_text_field:
            lines.append(line)
            continue

        tokens, comment_start = _lex_cif_line(line)
        if in_loop_headings:
            if not tokens:
                continue
            if len(tokens) == 1 and tokens[0].is_tag:
                lines.append(line if comment_start is None else line[:comment_start].rstrip())
                continue
            in_loop_headings = False
        lines.append(line)
        if len(tokens) == 1 and not tokens[0].quoted and tokens[0].text.lower() == "loop_":
            in_loop_headings = True
    return "\n".join(lines)


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


def _extract_propagation_vectors(
    tags: dict[str, str], loops: list[dict[str, Any]]
) -> list[list[float]]:
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

            label = (
                str(row.get(label_key, f"moment_{row_index}"))
                if label_key
                else f"moment_{row_index}"
            )
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
    prefix_matches = [
        index for index, candidate in enumerate(atom_labels) if candidate.startswith(f"{label}_")
    ]
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
        if any(
            all(abs(float(a) - float(b)) <= 1e-12 for a, b in zip(vector, existing, strict=True))
            for existing in out
        ):
            continue
        out.append([float(value) for value in vector])
    return out
