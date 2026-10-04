"""Shared app-layer helpers for turning GUI value dicts into workflow templates.

This module consolidates the near-identical logic that the convergence,
relaxation, and UEP adapters used to triplicate: parsing a GUI ``values`` dict
into a workflow :class:`~mufinder.workflow.types.InputTemplate`, the standard
expansion-result shape, and the supercell k-point-grid rescaling helper.

It stays free of PySide imports so it can live alongside the adapters in the
``app`` layer while remaining a pure workflow/castep_io/domain consumer.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mufinder.castep_io.gui_metadata import make_block_section, make_keyword_assignment
from mufinder.castep_io.schema import canonical_keyword_name, get_keyword_spec
from mufinder.castep_io.types import (
    AtomicPosition,
    BlockSection,
    KeywordAssignment,
    KPoint,
    LatticeABC,
    LatticeCart,
    PositionBlock,
    SpeciesPotential,
)
from mufinder.science.kpoint_grids import equivalent_grid_for_supercell
from mufinder.science.supercell_matching import detect_supercell_matrix_from_lattices
from mufinder.workflow.types import (
    CellTemplate,
    InputTemplate,
    JobSpec,
    ParamTemplate,
)

# Cell keywords whose values always come from the selected structure and must
# never be overridden through a GUI parameter dict.
STRUCTURE_CONTROLLED_CELL_KEYWORDS = frozenset(
    {
        "lattice_cart",
        "lattice_abc",
        "positions_frac",
        "positions_abs",
    }
)


@dataclass(frozen=True, slots=True)
class AdapterExpansion:
    """Expanded workflow jobs plus any UI-facing adjustment messages.

    Every app-layer ``expand_*`` function returns this shape. ``messages``
    carries human-readable notes (currently k-point rescaling adjustments) that
    the UI can surface; it is empty when nothing needed adjusting.
    """

    jobs: tuple[JobSpec, ...]
    messages: tuple[str, ...] = ()

    @property
    def kpoint_messages(self) -> tuple[str, ...]:
        """Backward-compatible alias for :attr:`messages`.

        The adjustment messages were originally named ``kpoint_messages`` on the
        per-workflow expansion dataclasses. The field is not persisted and no
        code outside the adapters and their tests read it, so it was renamed to
        the generic ``messages`` here; this alias keeps any lingering readers
        working.
        """

        return self.messages


def find_keyword_spec(name: str):
    """Return the CASTEP keyword spec for ``name`` from param or cell metadata."""

    for file_kind in ("param", "cell"):
        spec = get_keyword_spec(file_kind, name)
        if spec is not None:
            return spec
    return None


def row_values_has_keyword(row_values: Mapping[str, object], target: str) -> bool:
    """Return true if ``row_values`` include the given canonical keyword."""

    for raw_name in row_values:
        spec = find_keyword_spec(raw_name)
        if spec is not None and canonical_keyword_name(spec.file, spec.name) == target:
            return True
    return False


def _lattice_cart_from_block(block: BlockSection) -> LatticeCart:
    return LatticeCart(
        vectors=tuple(tuple(float(component) for component in row) for row in block.rows),  # type: ignore[arg-type]
        unit=block.unit,
    )


def _lattice_abc_from_block(block: BlockSection) -> LatticeABC:
    return LatticeABC(
        lengths=tuple(float(component) for component in block.rows[0]),  # type: ignore[arg-type]
        angles=tuple(float(component) for component in block.rows[1]),  # type: ignore[arg-type]
        unit=block.unit,
    )


def _positions_from_block(block: BlockSection) -> PositionBlock:
    return PositionBlock(
        positions=tuple(
            AtomicPosition(
                species=str(row[0]),
                coordinates=tuple(float(component) for component in row[1:4]),  # type: ignore[arg-type]
            )
            for row in block.rows
        ),
        unit=block.unit,
    )


def _species_pot_from_block(block: BlockSection) -> tuple[SpeciesPotential, ...]:
    return tuple(SpeciesPotential(species=str(row[0]), potential=str(row[1])) for row in block.rows)


def _kpoints_list_from_block(block: BlockSection) -> tuple[KPoint, ...]:
    return tuple(
        KPoint(
            coordinates=tuple(float(component) for component in row[0:3]),  # type: ignore[arg-type]
            weight=float(row[3]),
        )
        for row in block.rows
    )


def build_input_template(
    values: Mapping[str, object],
    *,
    context_label: str,
    forbidden: Mapping[str, str] | None = None,
    kpoints_list_reset: tuple[KPoint, ...] | None = None,
) -> InputTemplate:
    """Parse a GUI ``values`` dict into a workflow :class:`InputTemplate`.

    Parameters
    ----------
    values:
        Mapping of raw GUI keyword names to values.
    context_label:
        Phrase inserted into the "Unsupported CASTEP keyword '{name}' in
        {context_label}." error so each adapter keeps its own wording.
    forbidden:
        Mapping of canonical keyword name to the exact ``ValueError`` message to
        raise when a caller supplies a keyword controlled elsewhere (sweep
        list, selected structure, row controls). Structure-controlled cell
        keywords are supplied here through this mapping so each adapter keeps
        its exact validation wording.
    kpoints_list_reset:
        Sentinel that a scalar ``kpoint_mp_grid`` / ``kpoint_mp_spacing``
        assignment writes into ``kpoints_list`` to clear any inherited list.
        Convergence passes ``()`` ("explicitly clear the list layer"); the
        relaxation/UEP adapters pass ``None`` ("inherit"). This difference is
        preserved verbatim because it changes merge semantics.
    """

    forbidden = forbidden or {}

    lattice_cart = None
    lattice_abc = None
    positions_frac = None
    positions_abs = None
    cell_scalars: list[KeywordAssignment] = []
    cell_blocks: list[BlockSection] = []
    param_keywords: list[KeywordAssignment] = []
    param_blocks: list[BlockSection] = []

    species_pot = None
    kpoint_mp_grid = None
    kpoint_mp_spacing = None
    kpoints_list = None
    symmetry_generate = None

    for raw_name, raw_value in values.items():
        spec = find_keyword_spec(raw_name)
        if spec is None:
            raise ValueError(f"Unsupported CASTEP keyword '{raw_name}' in {context_label}.")

        canonical_name = canonical_keyword_name(spec.file, spec.name)
        if canonical_name in forbidden:
            raise ValueError(forbidden[canonical_name])

        if spec.kind == "block":
            block = make_block_section(spec, raw_value)
            if block is None:
                continue
            if spec.file == "cell":
                if canonical_name == "lattice_cart":
                    lattice_cart = _lattice_cart_from_block(block)
                    lattice_abc = None
                    continue
                if canonical_name == "lattice_abc":
                    lattice_abc = _lattice_abc_from_block(block)
                    lattice_cart = None
                    continue
                if canonical_name == "positions_frac":
                    positions_frac = _positions_from_block(block)
                    positions_abs = None
                    continue
                if canonical_name == "positions_abs":
                    positions_abs = _positions_from_block(block)
                    positions_frac = None
                    continue
                if canonical_name == "species_pot":
                    species_pot = _species_pot_from_block(block)
                    continue
                if canonical_name == "kpoints_list":
                    kpoints_list = _kpoints_list_from_block(block)
                    kpoint_mp_grid = None
                    kpoint_mp_spacing = None
                    continue
                cell_blocks.append(block)
                continue
            param_blocks.append(block)
            continue

        if spec.file == "cell":
            if canonical_name == "symmetry_generate":
                symmetry_generate = bool(raw_value)
                continue
            if canonical_name == "kpoint_mp_grid":
                assignment = make_keyword_assignment(spec, raw_value)
                if assignment is None:
                    continue
                triplet = tuple(int(component) for component in assignment.value)  # type: ignore[arg-type]
                if len(triplet) != 3:
                    raise ValueError("kpoint_mp_grid requires three integer values.")
                kpoint_mp_grid = (triplet[0], triplet[1], triplet[2])
                kpoint_mp_spacing = None
                kpoints_list = kpoints_list_reset
                continue
            if canonical_name == "kpoint_mp_spacing":
                assignment = make_keyword_assignment(spec, raw_value)
                if assignment is None:
                    continue
                kpoint_mp_spacing = assignment.value  # type: ignore[assignment]
                kpoint_mp_grid = None
                kpoints_list = kpoints_list_reset
                continue

        assignment = make_keyword_assignment(spec, raw_value)
        if assignment is None:
            continue

        if spec.file == "param":
            param_keywords.append(assignment)
        else:
            cell_scalars.append(assignment)

    cell_template = None
    if (
        lattice_cart is not None
        or lattice_abc is not None
        or positions_frac is not None
        or positions_abs is not None
        or species_pot is not None
        or kpoint_mp_grid is not None
        or kpoint_mp_spacing is not None
        or kpoints_list is not None
        or symmetry_generate is not None
        or cell_scalars
        or cell_blocks
    ):
        cell_template = CellTemplate(
            lattice_cart=lattice_cart,
            lattice_abc=lattice_abc,
            positions_frac=positions_frac,
            positions_abs=positions_abs,
            species_pot=species_pot,
            kpoint_mp_grid=kpoint_mp_grid,
            kpoint_mp_spacing=kpoint_mp_spacing,
            kpoints_list=kpoints_list,
            symmetry_generate=symmetry_generate,
            extra_scalars=tuple(cell_scalars),
            extra_blocks=tuple(cell_blocks),
        )

    param_template = (
        ParamTemplate(keywords=tuple(param_keywords), extra_blocks=tuple(param_blocks))
        if param_keywords or param_blocks
        else None
    )
    return InputTemplate(cell=cell_template, param=param_template)


def template_has_explicit_kpoints(template: InputTemplate) -> bool:
    """Return true if ``template`` already pins a k-point grid, spacing, or list."""

    cell = template.cell
    return cell is not None and (
        cell.kpoint_mp_grid is not None
        or cell.kpoint_mp_spacing is not None
        or cell.kpoints_list is not None
    )


def reference_is_supercell_of(
    reference_lattice: list[list[float]] | None,
    lattice: list[list[float]],
) -> bool:
    """Return whether a project grid's reference cell is a genuine supercell of ``lattice``.

    A project k-point grid converged on a saved supercell is recorded with that
    supercell's lattice, so a unit-cell row must rescale it. A reference that is
    merely a different unit cell (for example the host before it was replaced or
    re-relaxed) is not an integer supercell, and the grid is used exactly.
    """

    if reference_lattice is None:
        return False
    detection = detect_supercell_matrix_from_lattices(lattice, reference_lattice)
    return detection.is_supercell and detection.determinant > 1


def rescale_kpoint_grid(
    *,
    project_kpoint_grid: tuple[int, int, int] | None,
    reference_lattice: list[list[float]],
    row_lattice: list[list[float]],
    is_identity_supercell: bool,
    has_supercell: bool,
    target_description: str,
) -> tuple[tuple[int, int, int] | None, str | None]:
    """Return the supercell-adjusted k-point grid and an optional UI message.

    With no project default grid there is nothing to return. A supercell row
    rescales the grid from ``reference_lattice`` (the cell it was converged on)
    to its own cell at equivalent reciprocal-space density. A unit-cell row uses
    the grid exactly unless it was converged on a supercell of that unit cell
    (see :func:`reference_is_supercell_of`), in which case it is rescaled too.
    When the grid changed, a message describing the adjustment is returned.

    The message core (``K-point grid adjusted from AxBxC to DxExF for``) is
    shared; ``target_description`` supplies the trailing phrase so each adapter
    keeps its own wording (e.g. ``"2x2x1 (equivalent density)."`` for
    relaxation vs ``"row 'foo' (equivalent reciprocal-space density)."`` for
    UEP).
    """

    if project_kpoint_grid is None:
        return None, None

    if is_identity_supercell and not reference_is_supercell_of(reference_lattice, row_lattice):
        kpoint_grid = project_kpoint_grid
    else:
        kpoint_grid = equivalent_grid_for_supercell(
            reference_lattice,
            project_kpoint_grid,
            row_lattice,
        )

    message = None
    if kpoint_grid != project_kpoint_grid:
        message = (
            "K-point grid adjusted from "
            f"{project_kpoint_grid[0]}x{project_kpoint_grid[1]}x{project_kpoint_grid[2]} to "
            f"{kpoint_grid[0]}x{kpoint_grid[1]}x{kpoint_grid[2]} for "
            f"{target_description}"
        )
    return kpoint_grid, message
