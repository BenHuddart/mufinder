"""Structure file loading (CIF and MCIF through ASE, CASTEP CELL through ``castep_io``) and export.

Export writes a host plus muon sites as CIF (:func:`export_structure_with_muon_sites`)
and one complete structure, such as a relaxed supercell, as CIF or CASTEP CELL
(:func:`export_structure`).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import io
from pathlib import Path
import re
from typing import Any

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
import numpy as np

from mufinder.castep_io.cell_parser import parse_cell_file
from mufinder.castep_io.cell_writer import write_cell
from mufinder.castep_io.types import AtomicPosition, CellFile, LatticeCart, PositionBlock
from mufinder.domain.magnetic_properties import atomic_spins_from_magnetic_moments
from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import element_symbol, is_muon_label
from mufinder.science.lattice import (
    crystal_axis_to_cartesian,
    greedy_unique_indexes,
    minimum_image_distance_matrix,
    nearest_image_distance,
)


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

    raw_bytes = source.read_bytes()
    try:
        # ASE decodes CIF bytes as Latin-1, so this round trip is lossless.
        cif_text = _cif_text_for_ase(raw_bytes.decode("latin-1"))
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

    magnetic_structure: dict[str, Any] | None = None
    magnetic_cell: _MagneticCell | None = None
    # Only a magnetic CIF needs the second parse: a .cif without magnetic operations
    # is read by ASE alone.
    if suffix == ".mcif" or b"magn_operation" in raw_bytes.lower():
        tags, loops = _parse_cif_tags_and_loops(raw_bytes.decode("utf-8", errors="replace"))
        try:
            operations = _extract_magnetic_operations(tags, loops)
            if suffix == ".mcif" or operations:
                magnetic_cell = _magnetic_cell(
                    tags,
                    loops,
                    operations,
                    lattice_vectors=lattice_vectors,
                    species=symbols,
                    fractional_positions=fractional_positions,
                )
        except StructureLoadError as exc:
            raise StructureLoadError(f"Failed to read '{source.name}': {exc}") from exc
    if magnetic_cell is not None:
        symbols = magnetic_cell.species
        fractional_positions = magnetic_cell.fractional_positions
        magmoms = [magmoms[parent] for parent in magnetic_cell.parents]
        magnetic_structure = magnetic_cell.magnetic_structure

    properties: dict[str, Any] = {
        "source_extension": suffix,
        "atom_count": len(symbols),
        "pbc": [bool(v) for v in atoms.pbc.tolist()],
    }
    if any(abs(value) > 1e-12 for value in magmoms):
        properties["initial_magnetic_moments"] = magmoms
        properties["atomic_spins"] = [float(value) for value in magmoms]

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
    *,
    header_lines: Sequence[str] = (),
) -> list[str]:
    """Write a host structure plus muon sites to a CIF file.

    ``header_lines`` are written as ``#`` comments at the top of the file, as
    in :func:`export_structure`.

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

    host_labels = _host_atom_labels(host_species)
    labels = host_labels + [site.label for site in muon_sites]

    warnings: list[str] = []
    for site in muon_sites:
        for host_symbol, host_label, host_fractional in zip(
            host_species, host_labels, host_positions, strict=True
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
    _write_p1_cif(
        destination, lattice_vectors, species, positions, labels, header_lines=header_lines
    )
    if muon_sites:
        with destination.open("a", encoding=_CIF_ENCODING, errors="replace") as handle:
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


#: Text encoding of every exported CIF, matching ASE's writer and :func:`load_structure_file`.
_CIF_ENCODING = "latin-1"

#: File extensions :func:`export_structure` writes, in the order a save dialog offers them.
STRUCTURE_EXPORT_SUFFIXES = (".cif", ".cell")


def export_structure(
    structure: Structure,
    output_path: str | Path,
    *,
    header_lines: Sequence[str] = (),
) -> None:
    """Write one complete structure, such as a relaxed supercell, to CIF or CASTEP CELL.

    The format follows the extension of ``output_path`` (see
    :data:`STRUCTURE_EXPORT_SUFFIXES`); any other extension raises
    :class:`ValueError`. ``header_lines`` are written as comments at the top of
    the file (``#`` in CIF, ``!`` in CELL), one per line.

    CIF is written in P1 with every atom listed. A muon (any label
    :func:`~mufinder.science.ase_bridge.is_muon_label` accepts, such as ``H:mu``)
    is written as hydrogen labelled ``Mu1``, ``Mu2``…, and host atoms are
    labelled by element and count (``Fe1``, ``Fe2``…), as in
    :func:`export_structure_with_muon_sites`, so the muon stays distinct from
    any host hydrogen. CELL keeps the species labels as stored (the muon stays
    ``H:mu``) with ``LATTICE_CART`` in Å, ``POSITIONS_FRAC`` and, when the
    structure carries non-zero per-atom spins, their ``SPIN=`` values.
    """

    destination = Path(output_path)
    suffix = destination.suffix.lower()
    if suffix not in STRUCTURE_EXPORT_SUFFIXES:
        raise ValueError(
            f"Unsupported export extension '{destination.suffix}'. "
            f"Expected {' or '.join(STRUCTURE_EXPORT_SUFFIXES)}."
        )
    lattice_vectors = [[float(value) for value in row] for row in structure.lattice_vectors]
    positions = [[float(value) for value in row] for row in structure.fractional_positions]
    species = [str(label) for label in structure.species]

    if suffix == ".cif":
        symbols = [
            "H" if is_muon_label(label) else element_symbol(label, error=ValueError)
            for label in species
        ]
        host_labels = iter(
            _host_atom_labels(
                [
                    symbol
                    for symbol, label in zip(symbols, species, strict=True)
                    if not is_muon_label(label)
                ]
            )
        )
        muon_count = 0
        labels: list[str] = []
        for label in species:
            if is_muon_label(label):
                muon_count += 1
                labels.append(f"Mu{muon_count}")
            else:
                labels.append(next(host_labels))
        _write_p1_cif(
            destination, lattice_vectors, symbols, positions, labels, header_lines=header_lines
        )
        return

    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=tuple(tuple(row) for row in lattice_vectors),  # type: ignore[arg-type]
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=tuple(
                AtomicPosition(species=label, coordinates=tuple(position))  # type: ignore[arg-type]
                for label, position in zip(species, positions, strict=True)
            )
        ),
        atomic_spins=_structure_spins(structure),
    )
    body = write_cell(cell_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(_comment_block(header_lines, "!") + body, encoding="utf-8")


def _host_atom_labels(species: Sequence[str]) -> list[str]:
    """Label each atom by its element and running count (``Fe1``, ``Fe2``, ``O1``…)."""

    counts: dict[str, int] = {}
    labels: list[str] = []
    for symbol in species:
        counts[symbol] = counts.get(symbol, 0) + 1
        labels.append(f"{symbol}{counts[symbol]}")
    return labels


def _write_p1_cif(
    destination: Path,
    lattice_vectors: Sequence[Sequence[float]],
    symbols: Sequence[str],
    fractional_positions: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    header_lines: Sequence[str] = (),
) -> None:
    """Write a P1 CIF with one ``_atom_site`` row per atom, preceded by ``header_lines``.

    The file is Latin-1 throughout (:data:`_CIF_ENCODING`): ASE writes CIF in
    Latin-1 and :func:`load_structure_file` reads it back the same way. A header
    character outside Latin-1 (``CoF₂``) is written as ``?``.
    """

    atoms = Atoms(
        symbols=list(symbols),
        scaled_positions=[list(row) for row in fractional_positions],
        cell=[[float(value) for value in row] for row in lattice_vectors],
        pbc=True,
    )
    # The _atom_site loop is left as plain, standard CIF (no extra columns):
    # mixing a MuFinder-specific column into it makes some readers (observed
    # with VESTA) miscount that loop's column width, silently shifting every
    # atom's fields by one and corrupting the whole file. Per-muon metadata
    # instead goes in its own standalone loop, keyed by
    # _mufinder_muon_site_label, which a reader that does not recognise it can
    # simply skip without touching atom parsing.
    buffer = io.BytesIO()
    ase_write(buffer, atoms, format="cif", labels=[list(labels)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    header = _comment_block(header_lines, "#").encode(_CIF_ENCODING, errors="replace")
    destination.write_bytes(header + buffer.getvalue())


def _comment_block(lines: Sequence[str], marker: str) -> str:
    if not lines:
        return ""
    return "".join(f"{marker} {line}".rstrip() + "\n" for line in lines) + "\n"


def _structure_spins(structure: Structure) -> tuple[float, ...] | None:
    """Return the stored per-atom spins when they cover every atom and are not all zero."""

    raw = structure.properties.get("atomic_spins")
    if not isinstance(raw, list) or len(raw) != len(structure.species):
        return None
    try:
        spins = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    return spins if any(abs(value) > 1e-12 for value in spins) else None


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


#: Images of one atom closer than this (Angstrom) are one atom. Coordinates in a
#: magnetic CIF are rounded, so an image of a site on a special position lands a
#: rounding error away from it; distinct atoms are never this close.
_MAGNETIC_IMAGE_TOLERANCE_ANG = 0.01

#: Tag pairs (positions with time reversal, moments) for magnetic operations and
#: centerings: the MAGNDATA spelling first, then the DDLm magCIF spelling.
_MAGNETIC_OPERATION_TAGS = (
    ("_space_group_symop.magn_operation_xyz", "_space_group_symop.magn_operation_mxmymz"),
    ("_space_group_symop_magn_operation.xyz", "_space_group_symop_magn_operation.mxmymz"),
)
_MAGNETIC_CENTERING_TAGS = (
    ("_space_group_symop.magn_centering_xyz", "_space_group_symop.magn_centering_mxmymz"),
    ("_space_group_symop_magn_centering.xyz", "_space_group_symop_magn_centering.mxmymz"),
)


@dataclass(frozen=True, slots=True)
class _MagneticOperation:
    """One magnetic symmetry operation in the lattice basis.

    ``rotation`` and ``translation`` act on fractional positions; ``moment_matrix``
    acts on a moment's components along the lattice vectors (m = sum_i u_i a_i).
    """

    rotation: np.ndarray
    translation: np.ndarray
    time_reversal: int
    moment_matrix: np.ndarray

    def after(self, first: _MagneticOperation) -> _MagneticOperation:
        """Return this operation applied after ``first``."""
        return _MagneticOperation(
            rotation=self.rotation @ first.rotation,
            translation=self.rotation @ first.translation + self.translation,
            time_reversal=self.time_reversal * first.time_reversal,
            moment_matrix=self.moment_matrix @ first.moment_matrix,
        )


@dataclass(slots=True)
class _MagneticCell:
    """Atoms of a magnetic CIF after its magnetic symmetry, with their magnetic metadata.

    ``parents[i]`` is the index, in the file's atom-site order, of the site atom ``i``
    was generated from.
    """

    species: list[str]
    fractional_positions: list[list[float]]
    parents: list[int]
    magnetic_structure: dict[str, Any] | None


def _magnetic_cell(
    tags: dict[str, str],
    loops: list[dict[str, Any]],
    operations: list[_MagneticOperation],
    *,
    lattice_vectors: list[list[float]],
    species: list[str],
    fractional_positions: list[list[float]],
) -> _MagneticCell:
    """Apply a magnetic CIF's operations to its sites and read its magnetic metadata.

    Moments are returned as Cartesian vectors in Bohr magnetons. Without magnetic
    operations the atoms are kept as read.
    """
    site_labels = _atom_site_labels(tags, loops)
    moments = _cartesian_moments(_extract_moments_from_loops(loops), lattice_vectors)
    parents = list(range(len(species)))
    if operations:
        if len(site_labels) != len(species):
            raise StructureLoadError(
                f"The CIF reader gave {len(species)} atoms for the {len(site_labels)} atom "
                "sites the file lists, through ordinary symmetry operations or merged "
                "partially occupied sites; magnetic operations apply only to the listed sites."
            )
        fractional_positions, parents, images = _apply_magnetic_operations(
            fractional_positions, site_labels, operations, lattice_vectors
        )
        species = [species[parent] for parent in parents]
        atom_labels = [site_labels[parent] for parent in parents]
        moments = _moment_images(moments, site_labels, parents, images, lattice_vectors)
    else:
        atom_labels = _dedupe_strings(site_labels)
        moments = _attach_moment_atom_indices(moments, atom_labels=atom_labels)

    propagation_vectors = _extract_propagation_vectors(tags, loops)
    magnetic_structure: dict[str, Any] | None = None
    if propagation_vectors or moments:
        magnetic_structure = {"source": "mcif"}
        if atom_labels:
            magnetic_structure["atom_labels"] = atom_labels
        if propagation_vectors:
            magnetic_structure["propagation_vector"] = propagation_vectors[0]
            magnetic_structure["propagation_vectors_rlu"] = propagation_vectors
        if moments:
            magnetic_structure["moments"] = moments
            site_moments: dict[str, list[float]] = {}
            for moment in moments:
                # The first entry for a label is the site as written, before any image.
                site_moments.setdefault(str(moment["label"]), list(moment["vector"]))
            magnetic_structure["site_moments"] = site_moments
    return _MagneticCell(
        species=list(species),
        fractional_positions=fractional_positions,
        parents=parents,
        magnetic_structure=magnetic_structure,
    )


def _extract_magnetic_operations(
    tags: dict[str, str], loops: list[dict[str, Any]]
) -> list[_MagneticOperation]:
    """Return every magnetic operation composed with every magnetic centering.

    Empty when the file has no magnetic operations. The identity comes first when
    both lists start with it, as magnetic CIF files write them.
    """
    operations = _magnetic_operation_rows(tags, loops, _MAGNETIC_OPERATION_TAGS)
    if not operations:
        return []
    centerings = _magnetic_operation_rows(tags, loops, _MAGNETIC_CENTERING_TAGS)
    if not centerings:
        return operations
    return [centering.after(operation) for operation in operations for centering in centerings]


def _magnetic_operation_rows(
    tags: dict[str, str],
    loops: list[dict[str, Any]],
    tag_pairs: tuple[tuple[str, str], ...],
) -> list[_MagneticOperation]:
    for xyz_tag, moment_tag in tag_pairs:
        for loop in loops:
            headers: list[str] = loop.get("headers", [])
            if xyz_tag in headers:
                return [
                    _parse_magnetic_operation(str(row[xyz_tag]), row.get(moment_tag))
                    for row in loop.get("rows", [])
                ]
        if xyz_tag in tags:
            return [_parse_magnetic_operation(tags[xyz_tag], tags.get(moment_tag))]
    return []


def _parse_magnetic_operation(xyz: str, mxmymz: str | None) -> _MagneticOperation:
    """Parse ``x',y',z',theta`` and, when given, the explicit moment part ``mx',my',mz'``.

    Without a moment part, an axial moment transforms as det(R) * theta * R.
    """
    parts = [part.strip() for part in xyz.split(",")]
    if len(parts) != 4:
        raise StructureLoadError(
            f"Magnetic symmetry operation '{xyz}' must have four comma-separated parts: "
            "x, y, z and a time-reversal sign."
        )
    rows = [_parse_symmetry_expression(part, ("x", "y", "z"), xyz) for part in parts[:3]]
    rotation = np.array([row[0] for row in rows], dtype=float)
    translation = np.array([row[1] for row in rows], dtype=float)
    if parts[3] not in {"+1", "1", "-1"}:
        raise StructureLoadError(
            f"Magnetic symmetry operation '{xyz}' has time-reversal sign '{parts[3]}'; "
            "expected +1 or -1."
        )
    time_reversal = -1 if parts[3] == "-1" else 1
    determinant = float(np.linalg.det(rotation))
    if abs(determinant) < 1e-9:
        raise StructureLoadError(f"Magnetic symmetry operation '{xyz}' has a singular rotation.")

    moment_text = "" if mxmymz is None else str(mxmymz).strip()
    if moment_text in {"", ".", "?"}:
        moment_matrix = np.sign(determinant) * time_reversal * rotation
    else:
        moment_parts = [part.strip() for part in moment_text.split(",")]
        if len(moment_parts) != 3:
            raise StructureLoadError(
                f"Moment part '{moment_text}' of a magnetic symmetry operation must have "
                "three comma-separated parts."
            )
        moment_rows = [
            _parse_symmetry_expression(part, ("mx", "my", "mz"), moment_text)
            for part in moment_parts
        ]
        if any(constant != 0 for _, constant in moment_rows):
            raise StructureLoadError(
                f"Moment part '{moment_text}' of a magnetic symmetry operation must be "
                "linear in mx, my, mz."
            )
        moment_matrix = np.array([row[0] for row in moment_rows], dtype=float)
    return _MagneticOperation(
        rotation=rotation,
        translation=translation,
        time_reversal=time_reversal,
        moment_matrix=moment_matrix,
    )


def _parse_symmetry_expression(
    text: str, variables: tuple[str, str, str], operation: str
) -> tuple[list[Fraction], Fraction]:
    """Parse one affine component such as ``-x+y+1/2`` into coefficients and a constant."""
    compact = text.replace(" ", "").lower()
    terms = re.findall(r"[+-]?[^+-]+", compact)
    if not terms or "".join(terms) != compact:
        raise StructureLoadError(f"Cannot read '{text}' in symmetry operation '{operation}'.")
    coefficients = [Fraction(0)] * 3
    constant = Fraction(0)
    for term in terms:
        sign = -1 if term.startswith("-") else 1
        body = term.lstrip("+-")
        try:
            for axis, variable in enumerate(variables):
                if body.endswith(variable):
                    factor = body[: -len(variable)].rstrip("*")
                    coefficients[axis] += sign * (Fraction(factor) if factor else Fraction(1))
                    break
            else:
                constant += sign * Fraction(body)
        except (ValueError, ZeroDivisionError) as exc:
            raise StructureLoadError(
                f"Cannot read '{text}' in symmetry operation '{operation}'."
            ) from exc
    return coefficients, constant


def _apply_magnetic_operations(
    fractional_positions: list[list[float]],
    site_labels: list[str],
    operations: list[_MagneticOperation],
    lattice_vectors: list[list[float]],
) -> tuple[list[list[float]], list[int], list[_MagneticOperation]]:
    """Generate every distinct image of each site, grouped by site in the file's order.

    Returns the wrapped positions, the site each came from, and the operation that
    generated it. Each site comes first in its group, so a site keeps its position as
    written. Raises when an image of one site lands on another site.
    """
    positions: list[list[float]] = []
    parents: list[int] = []
    images: list[_MagneticOperation] = []
    identity = _MagneticOperation(np.eye(3), np.zeros(3), 1, np.eye(3))
    candidate_operations = [identity, *operations]
    for parent, site in enumerate(fractional_positions):
        point = np.asarray(site, dtype=float)
        candidates = (
            np.array(
                [
                    operation.rotation @ point + operation.translation
                    for operation in candidate_operations
                ]
            )
            % 1.0
        )
        # A coordinate a rounding error below zero wraps to exactly 1.0.
        candidates[candidates >= 1.0] = 0.0
        if positions:
            clashes = (
                minimum_image_distance_matrix(candidates, positions, lattice_vectors)
                <= _MAGNETIC_IMAGE_TOLERANCE_ANG
            )
            if clashes.any():
                other = parents[int(np.argwhere(clashes)[0][1])]
                raise StructureLoadError(
                    f"Magnetic symmetry places an image of site '{site_labels[parent]}' on "
                    f"site '{site_labels[other]}'."
                )
        within = (
            minimum_image_distance_matrix(candidates, candidates, lattice_vectors)
            <= _MAGNETIC_IMAGE_TOLERANCE_ANG
        )
        for index in greedy_unique_indexes(within):
            positions.append([float(value) for value in candidates[index]])
            parents.append(parent)
            images.append(candidate_operations[index])
    return positions, parents, images


def _cartesian_moments(
    moments: list[dict[str, Any]], lattice_vectors: list[list[float]]
) -> list[dict[str, Any]]:
    """Convert moment components to Cartesian vectors and check any stated magnitude.

    ``crystalaxis`` components lie along the unit lattice directions; ``cartn``
    components pass through. The components as written are kept.
    """
    converted: list[dict[str, Any]] = []
    for moment in moments:
        components = [float(value) for value in moment["vector"]]
        source_system = moment["coordinate_system"]
        if source_system == "crystalaxis":
            vector = crystal_axis_to_cartesian(
                components, lattice_vectors, error=StructureLoadError
            )
        else:
            vector = list(components)
        magnitude = moment.pop("magnitude", None)
        if magnitude is not None:
            length = float(np.linalg.norm(vector))
            if abs(length - magnitude) > max(0.02, 0.01 * abs(magnitude)):
                raise StructureLoadError(
                    f"Moment '{moment['label']}' has |m| = {length:.2f} Bohr magnetons from its "
                    f"{source_system} components but the file gives a magnitude of "
                    f"{magnitude:.2f}."
                )
        converted.append(
            {
                **moment,
                "vector": [float(value) for value in vector],
                "coordinate_system": "cartesian",
                "source_coordinate_system": source_system,
                "source_components": components,
            }
        )
    return converted


def _moment_images(
    moments: list[dict[str, Any]],
    site_labels: list[str],
    parents: list[int],
    images: list[_MagneticOperation],
    lattice_vectors: list[list[float]],
) -> list[dict[str, Any]]:
    """Give every image of a site with a moment its transformed moment.

    The moment's lattice-basis components u (m = sum_i u_i a_i) transform with each
    operation's moment matrix. A moment whose label matches no site belongs to the
    site at its row position, as it would without magnetic operations; one beyond
    the last site is dropped.
    """
    lattice = np.asarray(lattice_vectors, dtype=float)
    expanded: list[dict[str, Any]] = []
    for row, moment in enumerate(moments):
        site = _find_atom_label_index(site_labels, str(moment.get("label", "")).strip())
        if site is None:
            if row >= len(site_labels):
                continue
            site = row
        lattice_components = np.linalg.solve(lattice.T, np.asarray(moment["vector"]))
        for atom_index, (parent, operation) in enumerate(zip(parents, images, strict=True)):
            if parent != site:
                continue
            vector = (operation.moment_matrix @ lattice_components) @ lattice
            expanded.append(
                {
                    **moment,
                    "vector": [float(value) for value in vector],
                    "atom_index": atom_index,
                    "atom_label": site_labels[site],
                }
            )
    return expanded


def _parse_cif_tags_and_loops(text: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Parse the simple tags and loops of the CIF data block ASE reads.

    ASE reads the last ``data_`` block that holds a structure (atom symbols and
    positions), so the magnetic metadata comes from that block alone; without such
    a block the last block is used. This parser intentionally targets the subset
    needed for magnetic metadata.
    """
    blocks: list[list[_CifToken]] = [[]]
    for token in _iter_cif_tokens(text):
        if not token.quoted and token.text.lower().startswith("data_"):
            blocks.append([])
            continue
        blocks[-1].append(token)
    parsed = [_parse_cif_block(block) for block in blocks if block] or [({}, [])]
    for tags, loops in reversed(parsed):
        if _cif_block_has_structure(tags, loops):
            return tags, loops
    return parsed[-1]


def _cif_block_has_structure(tags: dict[str, str], loops: list[dict[str, Any]]) -> bool:
    """Whether a parsed block has atom symbols and positions, as ASE requires of a structure."""
    names = set(tags) | {header for loop in loops for header in loop.get("headers", [])}
    has_symbols = bool(names & {"_atom_site_type_symbol", "_atom_site_label"})
    has_positions = bool(names & {"_atom_site_fract_x", "_atom_site_cartn_x"})
    return has_symbols and has_positions


def _parse_cif_block(tokens: list[_CifToken]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Parse the tags and loops of one data block's tokens."""
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


def _atom_site_labels(tags: dict[str, str], loops: list[dict[str, Any]]) -> list[str]:
    """Return the CIF atom-site labels in file order, one per site row."""
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

    return labels


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

        magnitude_key = _first_matching_header(
            headers, ["_atom_site_moment.magnitude", "_atom_site_moment_magnitude"]
        )
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
            moment: dict[str, Any] = {
                "label": label,
                "vector": [x, y, z],
                "coordinate_system": coordinate_system,
            }
            magnitude = _parse_cif_float(str(row.get(magnitude_key, ""))) if magnitude_key else None
            if magnitude is not None:
                moment["magnitude"] = magnitude
            extracted.append(moment)

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
    """Return the atom index for an MCIF label: an exact match, else the one atom labelled ``<label>_*``."""
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
