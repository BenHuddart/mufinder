"""Serialization for CASTEP ``.cell`` files."""

from __future__ import annotations

from .schema import canonical_keyword_name, get_keyword_spec, serialization_order
from .types import BlockSection, CellFile, KeywordAssignment, ValueWithUnit
from .validators import validate_cell_file


def write_cell(cell_file: CellFile) -> str:
    """Serialize a :class:`CellFile` into CASTEP ``.cell`` text."""

    validate_cell_file(cell_file)

    sections: list[str] = []
    for name in serialization_order("cell"):
        rendered = _render_known_section(cell_file, name)
        if rendered is not None:
            sections.append(rendered)

    sections.extend(_render_extra_blocks(cell_file.blocks))
    sections.extend(_render_extra_scalars(cell_file.scalars))
    return "\n\n".join(sections) + "\n"


def _render_known_section(cell_file: CellFile, name: str) -> str | None:
    if name == "lattice_cart" and cell_file.lattice_cart is not None:
        rows = tuple(tuple(vector) for vector in cell_file.lattice_cart.vectors)
        return _render_block("LATTICE_CART", rows, unit=cell_file.lattice_cart.unit)
    if name == "lattice_abc" and cell_file.lattice_abc is not None:
        rows = (tuple(cell_file.lattice_abc.lengths), tuple(cell_file.lattice_abc.angles))
        return _render_block("LATTICE_ABC", rows, unit=cell_file.lattice_abc.unit)
    if name == "positions_frac" and cell_file.positions_frac is not None:
        rows = []
        for index, row in enumerate(cell_file.positions_frac.positions):
            values: tuple[object, ...] = (row.species, *row.coordinates)
            if (
                cell_file.atomic_spins is not None
                and index < len(cell_file.atomic_spins)
                and abs(float(cell_file.atomic_spins[index])) > 1e-12
            ):
                values = (*values, f"SPIN={_format_atom(float(cell_file.atomic_spins[index]))}")
            rows.append(values)
        return _render_block("POSITIONS_FRAC", rows, unit=None)
    if name == "positions_abs" and cell_file.positions_abs is not None:
        rows = tuple((row.species, *row.coordinates) for row in cell_file.positions_abs.positions)
        return _render_block("POSITIONS_ABS", rows, unit=cell_file.positions_abs.unit)
    if name == "species_pot" and cell_file.species_pot:
        rows = tuple((row.species, row.potential) for row in cell_file.species_pot)
        return _render_block("SPECIES_POT", rows)
    if name == "kpoints_list" and cell_file.kpoints_list:
        rows = tuple((*row.coordinates, row.weight) for row in cell_file.kpoints_list)
        return _render_block("KPOINTS_LIST", rows)
    if name == "kpoint_mp_grid" and cell_file.kpoint_mp_grid is not None:
        return _render_scalar(
            KeywordAssignment(name="kpoint_mp_grid", value=cell_file.kpoint_mp_grid)
        )
    if name == "kpoint_mp_spacing" and cell_file.kpoint_mp_spacing is not None:
        return _render_scalar(
            KeywordAssignment(name="kpoint_mp_spacing", value=cell_file.kpoint_mp_spacing)
        )
    if name == "symmetry_generate" and cell_file.symmetry_generate:
        return "SYMMETRY_GENERATE"
    return None


def _render_extra_blocks(blocks: tuple[BlockSection, ...]) -> list[str]:
    return [_render_block(block.name.upper(), block.rows, unit=block.unit) for block in blocks]


def _render_extra_scalars(assignments: tuple[KeywordAssignment, ...]) -> list[str]:
    return [_render_scalar(assignment) for assignment in assignments]


def _render_scalar(assignment: KeywordAssignment) -> str:
    canonical_name = canonical_keyword_name("cell", assignment.name)
    name = canonical_name.upper()
    spec = get_keyword_spec("cell", canonical_name)
    if assignment.value is None:
        return name
    if spec is not None and spec.kind == "flag" and assignment.value is True:
        return name
    if isinstance(assignment.value, bool):
        return f"{name} : {_render_value(assignment.value)}"
    return f"{name} {_render_value(assignment.value)}"


def _render_block(
    name: str,
    rows: tuple[tuple[object, ...], ...] | list[tuple[object, ...]],
    *,
    unit: str | None = None,
) -> str:
    lines = [f"%BLOCK {name}"]
    if unit:
        lines.append(unit)
    lines.extend(" ".join(_format_atom(value) for value in row) for row in rows)
    lines.append(f"%ENDBLOCK {name}")
    return "\n".join(lines)


def _render_value(value: object) -> str:
    if isinstance(value, ValueWithUnit):
        rendered = _render_value(value.value)
        return f"{rendered} {value.unit}"
    if isinstance(value, tuple):
        return " ".join(_format_atom(item) for item in value)
    return _format_atom(value)


def _format_atom(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)
