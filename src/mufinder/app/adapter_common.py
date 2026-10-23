"""Shared app-layer helpers for turning GUI value dicts into workflow templates.

This module consolidates the near-identical logic that the convergence,
relaxation, and UEP adapters used to triplicate: parsing a GUI ``values`` dict
into a workflow :class:`~mufinder.workflow.types.InputTemplate`, the standard
expansion-result shape, the supercell k-point-grid rescaling helper, the
expansion prologue every supercell workflow adapter runs (project-default
seeding, the k-point reference lattice, and the per-row label, structure and
k-point preparation), and the supercell-matrix coercion and labelling shared
with the workflow widgets.

It stays free of PySide imports so it can live alongside the adapters in the
``app`` layer while remaining a pure workflow/castep_io/domain consumer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from mufinder.app.relaxation_types import SupercellMatrix
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
    ValueWithUnit,
)
from mufinder.domain.models import Project, Structure
from mufinder.science.kpoint_grids import equivalent_grid_for_supercell
from mufinder.science.supercell_matching import detect_supercell_matrix_from_lattices
from mufinder.workflow.bridge import StructureBridgeError, structure_to_spec
from mufinder.workflow.types import (
    CellTemplate,
    InputTemplate,
    JobSpec,
    ParamTemplate,
    StructureSpec,
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


def format_grid(grid: tuple[int, int, int]) -> str:
    """Return a k-point grid as ``AxBxC``."""

    return f"{grid[0]}x{grid[1]}x{grid[2]}"


@dataclass(frozen=True, slots=True)
class KpointAdjustment:
    """One row whose project k-point grid was rescaled to its own cell.

    ``cell_label`` names the row's cell (``unit cell``, ``2x2x2`` or a
    spelled-out custom matrix); ``project_grid`` is the project default and
    ``adjusted_grid`` the grid at equivalent reciprocal-space density.
    """

    row_label: str
    cell_label: str
    project_grid: tuple[int, int, int]
    adjusted_grid: tuple[int, int, int]

    @property
    def cell_description(self) -> str:
        """Return ``unit cell`` or ``<cell_label> supercell``."""

        return self.cell_label if self.cell_label == "unit cell" else f"{self.cell_label} supercell"

    @property
    def change(self) -> str:
        """Return the grid change as ``AxBxC → DxExF``."""

        return f"{format_grid(self.project_grid)} → {format_grid(self.adjusted_grid)}"

    @property
    def message(self) -> str:
        """Describe this row's adjustment on one short line."""

        return f"{self.row_label}: {self.change} ({self.cell_description})"


@dataclass(frozen=True, slots=True)
class AdapterExpansion:
    """Expanded workflow jobs plus any k-point adjustments made on the way.

    Every app-layer ``expand_*`` function returns this shape.
    ``kpoint_adjustments`` holds one record per row whose project k-point grid
    was rescaled; the UI summarises them with
    :func:`summarize_kpoint_adjustments`. It is empty when nothing changed.
    """

    jobs: tuple[JobSpec, ...]
    kpoint_adjustments: tuple[KpointAdjustment, ...] = ()


#: Most distinct adjustments listed in a k-point summary before the rest are
#: left to the details.
KPOINT_SUMMARY_MAX_GROUPS = 5


@dataclass(frozen=True, slots=True)
class KpointAdjustmentSummary:
    """A k-point adjustment report sized for a message box.

    ``headline`` is one short sentence, ``body`` the grouped changes and
    ``details`` one line per row.
    """

    headline: str
    body: str
    details: str


def summarize_kpoint_adjustments(
    adjustments: Iterable[KpointAdjustment],
    *,
    max_groups: int = KPOINT_SUMMARY_MAX_GROUPS,
) -> KpointAdjustmentSummary | None:
    """Summarise ``adjustments`` for display, or return ``None`` when there are none.

    Rows with the same cell and the same grid change share one body line with a
    row count, so the body grows with the number of distinct cells, not the
    number of rows. At most ``max_groups`` lines are listed, the most common
    first; a final line counts the rest. ``details`` lists every row.
    """

    records = list(adjustments)
    if not records:
        return None
    groups: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.cell_description, record.change)
        groups[key] = groups.get(key, 0) + 1
    ordered = sorted(groups.items(), key=lambda item: -item[1])

    lines = ["Each grid keeps the project's reciprocal-space density.", ""]
    for (cell, change), count in ordered[:max_groups]:
        lines.append(f"• {cell}: {change} ({_rows(count)})")
    hidden = len(ordered) - max_groups
    if hidden > 0:
        lines.append(
            f"• and {hidden} more {'change' if hidden == 1 else 'changes'}; see Show Details."
        )
    return KpointAdjustmentSummary(
        headline=f"K-point grid rescaled for {_rows(len(records))}.",
        body="\n".join(lines),
        details="\n".join(record.message for record in records),
    )


def _rows(count: int) -> str:
    return f"{count} {'row' if count == 1 else 'rows'}"


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


def build_workflow_settings_template(values: Mapping[str, object]) -> InputTemplate:
    """Parse quantum-corrections and energy-barriers row values into a template.

    These workflows take their structure from the relaxed source, so the
    structure-controlled cell keywords are refused, and a scalar
    ``kpoint_mp_grid`` / ``kpoint_mp_spacing`` clears any inherited
    ``kpoints_list`` (``kpoints_list_reset=()``).
    """

    return build_input_template(
        values,
        context_label="workflow settings",
        forbidden={
            name: f"{name} is controlled by the workflow structure and cannot be overridden."
            for name in STRUCTURE_CONTROLLED_CELL_KEYWORDS
        },
        kpoints_list_reset=(),
    )


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
) -> tuple[int, int, int] | None:
    """Return the project k-point grid rescaled to a row's cell.

    With no project default grid there is nothing to return. A supercell row
    rescales the grid from ``reference_lattice`` (the cell it was converged on)
    to its own cell at equivalent reciprocal-space density. A unit-cell row uses
    the grid exactly unless it was converged on a supercell of that unit cell
    (see :func:`reference_is_supercell_of`), in which case it is rescaled too.
    """

    if project_kpoint_grid is None:
        return None
    if is_identity_supercell and not reference_is_supercell_of(reference_lattice, row_lattice):
        return project_kpoint_grid
    return equivalent_grid_for_supercell(
        reference_lattice,
        project_kpoint_grid,
        row_lattice,
    )


IDENTITY_SUPERCELL: SupercellMatrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def coerce_supercell_matrix(value: object) -> SupercellMatrix | None:
    """Return ``value`` as a 3x3 integer supercell matrix, or ``None``.

    Accepts the nested lists that persisted widget state and supercell options
    carry as well as tuples. Anything that is not three rows of three integers
    (or integer-convertible values) yields ``None``.
    """

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (
            (int(value[0][0]), int(value[0][1]), int(value[0][2])),
            (int(value[1][0]), int(value[1][1]), int(value[1][2])),
            (int(value[2][0]), int(value[2][1]), int(value[2][2])),
        )
    except (TypeError, ValueError, IndexError):
        return None


def is_identity_supercell(matrix: SupercellMatrix | None) -> bool:
    """Return whether ``matrix`` selects the unit cell (``None`` or identity)."""

    return matrix is None or matrix == IDENTITY_SUPERCELL


def supercell_label(
    matrix: SupercellMatrix | None,
    *,
    unit_cell: str = "Unit cell",
    spell_out_custom: bool = False,
) -> str:
    """Return a short label for a supercell matrix.

    The unit cell reads ``unit_cell``, a diagonal matrix reads ``AxBxC``, and
    any other matrix reads ``Custom 3x3`` or, with ``spell_out_custom``, its
    rows as ``[a b c; d e f; g h i]``.
    """

    if matrix is None or is_identity_supercell(matrix):
        return unit_cell
    if all(matrix[i][j] == 0 for i in range(3) for j in range(3) if i != j):
        return f"{matrix[0][0]}x{matrix[1][1]}x{matrix[2][2]}"
    if not spell_out_custom:
        return "Custom 3x3"
    return "[" + "; ".join(" ".join(str(value) for value in row) for row in matrix) + "]"


def seed_workflow_defaults(
    values: Mapping[str, object],
    *,
    project_xc_functional: str | None,
    project_cutoff_energy: float | None,
) -> dict[str, object]:
    """Return a workflow's default GUI values with the project defaults applied.

    The project exchange-correlation functional and cut-off energy, when set,
    are layered over the workflow's own defaults.
    """

    seeded = dict(values)
    if project_xc_functional is not None:
        seeded["xc_functional"] = project_xc_functional
    if project_cutoff_energy is not None:
        seeded["cut_off_energy"] = float(project_cutoff_energy)
    return seeded


def project_convergence_parameter_values(
    project: Project,
    *,
    target_lattice: list[list[float]] | None = None,
) -> dict[str, object]:
    """Return the project defaults as GUI parameter values for a follow-on row.

    The functional, the cut-off energy (in eV) and the k-point grid, each only
    when set. With ``target_lattice`` and a recorded reference lattice, the grid
    is rescaled to the target cell at equivalent reciprocal-space density.
    """

    values: dict[str, object] = {}
    if project.recommended_xc_functional is not None:
        values["xc_functional"] = project.recommended_xc_functional
    cutoff = project.recommended_cutoff_energy
    if cutoff is not None:
        values["cut_off_energy"] = ValueWithUnit(value=float(cutoff), unit="eV")
    grid = project.recommended_kpoint_grid
    if grid is not None:
        resolved_grid = tuple(int(component) for component in grid)
        reference_lattice = project.recommended_kpoint_lattice
        if (
            target_lattice is not None
            and isinstance(reference_lattice, list)
            and len(reference_lattice) == 3
        ):
            resolved_grid = equivalent_grid_for_supercell(
                reference_lattice, resolved_grid, target_lattice
            )
        values["kpoint_mp_grid"] = resolved_grid
    return values


def kpoint_reference_lattice(
    project_kpoint_reference_lattice: list[list[float]] | None,
    host_structure: Structure,
) -> list[list[float]]:
    """Return the lattice the project k-point grid was converged on.

    A recorded three-vector project reference lattice wins; otherwise the grid
    is taken to belong to the host cell.
    """

    if (
        isinstance(project_kpoint_reference_lattice, list)
        and len(project_kpoint_reference_lattice) == 3
    ):
        return project_kpoint_reference_lattice
    return host_structure.lattice_vectors


def require_row_label(raw_label: str, *, workflow: str) -> str:
    """Return the stripped row label, rejecting an empty one.

    ``workflow`` names the rows in the error, for example
    ``"pristine-relaxation"`` or ``"UEP source"``.
    """

    label = raw_label.strip()
    if not label:
        raise ValueError(f"{workflow[:1].upper()}{workflow[1:]} row labels must not be empty.")
    return label


def claim_unique_name(name: str, seen: set[str], *, workflow: str, noun: str = "row label") -> None:
    """Record ``name`` in ``seen``, rejecting a name an earlier row already claimed."""

    if name in seen:
        raise ValueError(f"Duplicate {workflow} {noun} '{name}'.")
    seen.add(name)


@dataclass(frozen=True, slots=True)
class PreparedRow:
    """One adapter row's resolved inputs, ready for its workflow expander.

    ``values`` are the row's GUI values after the workflow-controlled names
    were dropped or forced; ``template`` is built from them. ``kpoint_grid``
    is the supercell-rescaled project grid, or ``None`` when the row pins its
    own k-points or the project has no grid. ``kpoint_adjustment`` records the
    rescaling when the grid differs from the project grid.
    """

    structure: StructureSpec
    values: dict[str, object]
    template: InputTemplate
    kpoint_grid: tuple[int, int, int] | None
    kpoint_adjustment: KpointAdjustment | None


def prepare_row(
    *,
    label: str,
    structure: Structure,
    parameter_values: Mapping[str, object],
    template_from_values: Callable[[Mapping[str, object]], InputTemplate],
    supercell_matrix: SupercellMatrix | None,
    project_kpoint_grid: tuple[int, int, int] | None,
    reference_lattice: list[list[float]],
    drop: Iterable[str] = (),
    force: Mapping[str, object] | None = None,
) -> PreparedRow:
    """Run the per-row prologue shared by the supercell workflow adapters.

    Converts the row's ``structure`` to a workflow structure spec, removes the
    ``drop`` names the workflow controls and applies the ``force`` values,
    builds the row template, and, unless the row pins its own k-points,
    rescales the project grid to the row's cell with :func:`rescale_kpoint_grid`
    and records a :class:`KpointAdjustment` when the grid changed.
    """

    try:
        structure_spec = structure_to_spec(structure)
    except StructureBridgeError as exc:
        raise ValueError(f"Could not convert row '{label}' structure: {exc}") from exc

    values = dict(parameter_values)
    for name in drop:
        values.pop(name, None)
    values.update(force or {})
    template = template_from_values(values)

    kpoint_grid = None
    adjustment = None
    if not template_has_explicit_kpoints(template):
        kpoint_grid = rescale_kpoint_grid(
            project_kpoint_grid=project_kpoint_grid,
            reference_lattice=reference_lattice,
            row_lattice=structure.lattice_vectors,
            is_identity_supercell=is_identity_supercell(supercell_matrix),
        )
        if (
            project_kpoint_grid is not None
            and kpoint_grid is not None
            and kpoint_grid != project_kpoint_grid
        ):
            adjustment = KpointAdjustment(
                row_label=label,
                cell_label=supercell_label(
                    supercell_matrix, unit_cell="unit cell", spell_out_custom=True
                ),
                project_grid=tuple(project_kpoint_grid),  # type: ignore[arg-type]
                adjusted_grid=tuple(kpoint_grid),  # type: ignore[arg-type]
            )
    return PreparedRow(
        structure=structure_spec,
        values=values,
        template=template,
        kpoint_grid=kpoint_grid,
        kpoint_adjustment=adjustment,
    )
