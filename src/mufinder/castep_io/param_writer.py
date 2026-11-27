"""Serialization for CASTEP ``.param`` files."""

from __future__ import annotations

from .schema import canonical_keyword_name, serialization_order
from .types import BlockSection, KeywordAssignment, ParamFile, ValueWithUnit
from .validators import validate_param_file


def write_param(param_file: ParamFile) -> str:
    """Serialize a :class:`ParamFile` into CASTEP ``.param`` text."""

    validate_param_file(param_file)

    ordered_names = list(serialization_order("param"))
    seen_indexes: set[int] = set()
    lines: list[str] = []

    for canonical_name in ordered_names:
        for index, assignment in enumerate(param_file.keywords):
            if index in seen_indexes:
                continue
            if canonical_keyword_name("param", assignment.name) == canonical_name:
                lines.append(_render_scalar(assignment))
                seen_indexes.add(index)

    for index, assignment in enumerate(param_file.keywords):
        if index not in seen_indexes:
            lines.append(_render_scalar(assignment))

    if param_file.blocks:
        if lines:
            lines.append("")
        for index, block in enumerate(param_file.blocks):
            if index:
                lines.append("")
            lines.append(_render_block(block))

    return "\n".join(lines) + "\n"


def _render_scalar(assignment: KeywordAssignment) -> str:
    name = canonical_keyword_name("param", assignment.name).upper()
    return f"{name} : {_render_value(assignment.value)}"


def _render_block(block: BlockSection) -> str:
    name = canonical_keyword_name("param", block.name).upper()
    lines = [f"%BLOCK {name}"]
    if block.unit:
        lines.append(block.unit)
    lines.extend(" ".join(_format_atom(value) for value in row) for row in block.rows)
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
    if value is None:
        raise ValueError("CASTEP param scalar values may not be None.")
    return str(value)
