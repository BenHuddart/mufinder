"""Keyword-level comparison of a job's CASTEP ``.cell`` and ``.param`` inputs.

The Continue-relaxation dialog shows what rewriting a job's inputs will change.
:func:`diff_job_inputs` compares two :class:`JobInputTexts` keyword by keyword
and block by block, in both files. Formatting that CASTEP reads the same way
(case, spacing, comments, ``500 eV`` against ``500.0 eV``) is not a change.

The structure blocks (``LATTICE_CART``, ``LATTICE_ABC``, ``POSITIONS_FRAC``,
``POSITIONS_ABS``) are left out: they describe the structure, not the run's
parameters.

A continued relaxation keeps some of the inputs its run used.
:func:`overlay_job_inputs` builds them: one version of the files with only the
named keywords taken from another.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .cell_parser import GEOMETRY_BLOCKS
from .gui_metadata import exclusive_partner_names
from .param_parser import _split_assignment, _strip_comment, parse_param_value
from .schema import canonical_keyword_name, get_keyword_spec
from .types import ValueWithUnit

InputFileKind = Literal["cell", "param"]
KeywordChangeKind = Literal["Added", "Changed", "Removed"]


@dataclass(frozen=True, slots=True)
class JobInputTexts:
    """The text of one job's ``.cell`` and ``.param`` files."""

    cell: str
    param: str


@dataclass(frozen=True, slots=True)
class InputSections:
    """The keywords and blocks of one CASTEP input file, by canonical name.

    A scalar keeps its value text as written (empty for a bare flag such as
    ``SYMMETRY_GENERATE``); a block keeps its non-empty lines.
    """

    scalars: dict[str, str]
    blocks: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class InputKeywordChange:
    """One keyword or block that differs between two versions of an input file.

    ``old_lines``/``new_lines`` hold a scalar's value as a single line and a
    block's content line by line; the side where the keyword is absent is empty.
    """

    file: InputFileKind
    keyword: str
    change: KeywordChangeKind
    is_block: bool
    old_lines: tuple[str, ...]
    new_lines: tuple[str, ...]


def split_input_sections(text: str, file: InputFileKind) -> InputSections:
    """Split CASTEP input text into its scalar keywords and blocks.

    Lenient by design: it reads a file only to compare it, so an unterminated
    block keeps the lines it gathered and a stray ``%ENDBLOCK`` is ignored
    rather than raising. Use :func:`~mufinder.castep_io.cell_parser.parse_cell_text`
    or :func:`~mufinder.castep_io.param_parser.parse_param_text` to validate.
    """

    scalars: dict[str, str] = {}
    blocks: dict[str, tuple[str, ...]] = {}
    block_name: str | None = None
    block_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line.strip())
        if not line:
            continue
        words = line.split()
        marker = words[0].upper()
        if marker == "%BLOCK":
            if block_name is not None:
                blocks[block_name] = tuple(block_lines)
            # A header with no name opens nothing; anything after the name is ignored.
            block_name = canonical_keyword_name(file, words[1]) if len(words) > 1 else None
            block_lines = []
            continue
        if marker == "%ENDBLOCK":
            if block_name is not None:
                blocks[block_name] = tuple(block_lines)
            block_name = None
            continue
        if block_name is not None:
            block_lines.append(" ".join(line.split()))
            continue
        split = _split_assignment(line)
        if split is None:
            continue
        raw_name, raw_value = split
        scalars[canonical_keyword_name(file, raw_name)] = " ".join(raw_value.split())
    if block_name is not None:
        blocks[block_name] = tuple(block_lines)
    return InputSections(scalars=scalars, blocks=blocks)


def _comparable_scalar(file: InputFileKind, name: str, value: str) -> object:
    spec = get_keyword_spec(file, name)
    try:
        parsed = parse_param_value(value, spec)
    except ValueError:
        return value.casefold()
    # A value with no unit is in the keyword's default unit, and CASTEP reads
    # unit names in any case.
    if isinstance(parsed, ValueWithUnit):
        return (parsed.value, parsed.unit.casefold())
    if isinstance(parsed, (int, float)) and not isinstance(parsed, bool):
        if spec is not None and spec.unit:
            return (float(parsed), spec.unit.casefold())
    return parsed


def _comparable_token(token: str) -> object:
    try:
        return float(token)
    except ValueError:
        return token.casefold()


def _comparable_block(lines: tuple[str, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(tuple(_comparable_token(token) for token in line.split()) for line in lines)


def diff_input_text(
    old_text: str,
    new_text: str,
    file: InputFileKind,
    *,
    ignored_blocks: frozenset[str] = frozenset(),
) -> tuple[InputKeywordChange, ...]:
    """Return the keywords and blocks added, changed or removed in one input file.

    Changes are ordered by keyword name. Blocks named in ``ignored_blocks``
    (canonical lowercase names) are left out.
    """

    old = split_input_sections(old_text, file)
    new = split_input_sections(new_text, file)
    changes: list[InputKeywordChange] = []

    for name in sorted(set(old.scalars) | set(new.scalars) | set(old.blocks) | set(new.blocks)):
        if name in ignored_blocks:
            continue
        is_block = name in old.blocks or name in new.blocks
        if is_block:
            old_lines = old.blocks.get(name)
            new_lines = new.blocks.get(name)
            same = (
                old_lines is not None
                and new_lines is not None
                and _comparable_block(old_lines) == _comparable_block(new_lines)
            )
        else:
            old_value = old.scalars.get(name)
            new_value = new.scalars.get(name)
            old_lines = None if old_value is None else (old_value,)
            new_lines = None if new_value is None else (new_value,)
            same = (
                old_value is not None
                and new_value is not None
                and _comparable_scalar(file, name, old_value)
                == _comparable_scalar(file, name, new_value)
            )
        if same:
            continue
        if old_lines is None:
            change: KeywordChangeKind = "Added"
        elif new_lines is None:
            change = "Removed"
        else:
            change = "Changed"
        changes.append(
            InputKeywordChange(
                file=file,
                keyword=name.upper(),
                change=change,
                is_block=is_block,
                old_lines=old_lines or (),
                new_lines=new_lines or (),
            )
        )
    return tuple(changes)


def diff_job_inputs(old: JobInputTexts, new: JobInputTexts) -> tuple[InputKeywordChange, ...]:
    """Return every keyword and block change across a job's ``.cell`` and ``.param``.

    ``.cell`` changes come first, then ``.param`` changes, each ordered by
    keyword name. The ``.cell`` structure blocks are left out.
    """

    return diff_input_text(
        old.cell, new.cell, "cell", ignored_blocks=GEOMETRY_BLOCKS
    ) + diff_input_text(old.param, new.param, "param")


def read_job_inputs(directory: Path, seed_name: str) -> JobInputTexts | None:
    """Read ``{seed_name}.cell`` and ``{seed_name}.param`` from ``directory``.

    Returns ``None`` when either file is missing or unreadable.
    """

    try:
        return JobInputTexts(
            cell=(directory / f"{seed_name}.cell").read_text(encoding="utf-8"),
            param=(directory / f"{seed_name}.param").read_text(encoding="utf-8"),
        )
    except (OSError, UnicodeDecodeError):
        return None


def write_job_inputs(directory: Path, seed_name: str, inputs: JobInputTexts) -> None:
    """Write ``inputs`` as ``{seed_name}.cell`` and ``{seed_name}.param`` in ``directory``.

    Written as the workflow writes a job's inputs (UTF-8, ``\\n`` line ends), so the
    ``.param`` is fresh, as a rewritten input always is.
    """

    (directory / f"{seed_name}.cell").write_text(inputs.cell, encoding="utf-8", newline="\n")
    (directory / f"{seed_name}.param").write_text(inputs.param, encoding="utf-8", newline="\n")


def _input_entries(text: str, file: InputFileKind) -> list[tuple[str | None, list[str]]]:
    """Split ``text`` into entries ``(canonical name, raw lines)``, in file order.

    A scalar keyword is one line; a block runs from its ``%BLOCK`` line to its
    ``%ENDBLOCK`` line. Blank and comment lines, and anything else that names no
    keyword, are entries with no name.
    """

    entries: list[tuple[str | None, list[str]]] = []
    block: tuple[str | None, list[str]] | None = None
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line.strip())
        words = line.split()
        marker = words[0].upper() if words else ""
        if block is not None:
            block[1].append(raw_line)
            if marker == "%ENDBLOCK":
                entries.append(block)
                block = None
            continue
        if marker == "%BLOCK":
            name = canonical_keyword_name(file, words[1]) if len(words) > 1 else None
            block = (name, [raw_line])
            continue
        split = _split_assignment(line) if line else None
        if split is None or marker == "%ENDBLOCK":
            entries.append((None, [raw_line]))
            continue
        entries.append((canonical_keyword_name(file, split[0]), [raw_line]))
    if block is not None:
        entries.append(block)
    return entries


def with_exclusive_partners(names: Collection[str]) -> frozenset[str]:
    """Return ``names`` with every keyword that cannot be set together with one of them.

    ``names`` are keyword names in any case, of either file; the result holds
    canonical names (``cut_off_energy`` brings ``basis_precision``,
    ``kpoint_mp_grid`` brings ``kpoint_mp_spacing``).
    """

    result: set[str] = set()
    for file in ("cell", "param"):
        for name in names:
            canonical = canonical_keyword_name(file, str(name))
            result.add(canonical)
            spec = get_keyword_spec(file, canonical)
            if spec is not None:
                result.update(exclusive_partner_names(spec))
    return frozenset(result)


def _overlay_text(base: str, new: str, names: frozenset[str], file: InputFileKind) -> str:
    new_entries = {name: lines for name, lines in _input_entries(new, file) if name in names}
    lines: list[str] = []
    placed: set[str] = set()
    for name, entry_lines in _input_entries(base, file):
        if name is None or name not in names:
            lines.extend(entry_lines)
            continue
        if name in new_entries and name not in placed:
            lines.extend(new_entries[name])
            placed.add(name)
    for name, entry_lines in new_entries.items():
        if name not in placed:
            lines.extend(entry_lines)
    text = "\n".join(lines)
    return f"{text}\n" if text and (base.endswith("\n") or not base) else text


def overlay_job_inputs(
    base: JobInputTexts, new: JobInputTexts, names: Collection[str]
) -> JobInputTexts:
    """Return ``base`` with the keywords and blocks ``names`` taken from ``new``.

    Every other line of ``base`` is kept exactly as written. A named keyword or
    block replaces the one in ``base`` where it stands, is appended when ``base``
    lacks it, and is dropped when ``new`` lacks it. A name carries its exclusive
    partners with it (:func:`with_exclusive_partners`), so the result never
    holds both members of an either-or pair. ``names`` are keyword names in any
    case, of either file.
    """

    taken = with_exclusive_partners(names)
    return JobInputTexts(
        cell=_overlay_text(base.cell, new.cell, taken, "cell"),
        param=_overlay_text(base.param, new.param, taken, "param"),
    )


__all__ = [
    "InputKeywordChange",
    "InputSections",
    "JobInputTexts",
    "diff_input_text",
    "diff_job_inputs",
    "overlay_job_inputs",
    "read_job_inputs",
    "split_input_sections",
    "with_exclusive_partners",
    "write_job_inputs",
]
