"""Utilities for parsing CASTEP ``.param`` files.

MuFinder historically wrote ``.param`` files without ever reading them back.
That asymmetry is why the Continue-relaxation keyword diff is computed by
re-expanding and re-rendering a job rather than by parsing what is on disk, and
why importing calculations performed outside MuFinder had no way to recover the
settings a run actually used.

This module closes the gap. It is deliberately the mirror image of
:mod:`mufinder.castep_io.cell_parser`: :func:`parse_param_text` returns a
:class:`~mufinder.castep_io.types.ParamFile` that
:func:`~mufinder.castep_io.param_writer.write_param` can render again, while
:func:`parse_param_assignments` returns the flat ``{canonical name: value}``
mapping that callers wanting a single keyword prefer.

**Values are coerced through the runtime keyword schema, never guessed.** A
keyword's :class:`~mufinder.castep_io.schema.KeywordSpec` decides whether
``1`` means the integer one, the boolean true, or the string ``"1"``. Keywords
with no spec — the runtime catalog is environment-dependent, see
:func:`~mufinder.castep_io.schema.keyword_catalog_source_info` — fall back to a
conservative literal inference and are reported by
:func:`parse_param_observations` as unmapped, so a caller can record them
instead of silently dropping them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex

from .schema import KeywordSpec, canonical_keyword_name, get_keyword_spec
from .types import BlockSection, KeywordAssignment, ParamFile, ValueWithUnit


class ParamParseError(Exception):
    """Raised when CASTEP ``.param`` text cannot be parsed."""


_COMMENT_MARKERS = ("!", "#")
_TRUE_TOKENS = frozenset({"true", "t", ".true.", "yes", "1"})
_FALSE_TOKENS = frozenset({"false", "f", ".false.", "no", "0"})
#: Tokens accepted as booleans when no spec pins the type. Deliberately narrower
#: than the spec-driven sets above: without a spec, a bare ``1`` is far more
#: likely to be a number than a boolean, so digits are excluded here.
_UNTYPED_TRUE_TOKENS = frozenset({"true", "t", ".true.", "yes"})
_UNTYPED_FALSE_TOKENS = frozenset({"false", "f", ".false.", "no"})

#: CASTEP accepts several spellings of a ``TASK`` value and normalises them on
#: read. Transcribed from ``Source/Fundamental/parameters.f90``, which rewrites
#: the British spelling with the comment "English spelling" and folds the short
#: forms in the same place. Legacy MuFinder wrote ``geometryoptimisation``, so
#: without this table a legacy file parses to a value MuFinder's own validator
#: then rejects.
#:
#: Only value aliases CASTEP itself performs appear here. This is transcription,
#: not inference: a spelling CASTEP does not fold must not be folded here.
_TASK_VALUE_ALIASES: dict[str, str] = {
    "GEOMETRYOPTIMISATION": "GEOMETRYOPTIMIZATION",
    "GEOMOPT": "GEOMETRYOPTIMIZATION",
    "GEOM": "GEOMETRYOPTIMIZATION",
    "SPE": "SINGLEPOINT",
    "ENERGY": "SINGLEPOINT",
    "MD": "MOLECULARDYNAMICS",
    "DYNAMICS": "MOLECULARDYNAMICS",
    "SPECTRAL": "ELECTRONICSPECTROSCOPY",
    "EPC": "EPCOUPLING",
    "EPME": "EPCOUPLING",
    "SOCKET_DRIVER": "SOCKETDRIVER",
    "ELASTICS": "ELASTIC",
    "ELASTIC_CONSTANTS": "ELASTIC",
    "ELASTIC_CONSTANT": "ELASTIC",
    "ELASTICCONSTANTS": "ELASTIC",
    "ELASTICCONSTANT": "ELASTIC",
    "KOOPMAN": "KOOPMANS",
}

#: Value-alias tables by canonical keyword name. Only ``TASK`` has one today.
_VALUE_ALIASES: dict[str, dict[str, str]] = {"task": _TASK_VALUE_ALIASES}


def canonical_enum_value(keyword: str, raw_value: str, spec: KeywordSpec | None) -> str:
    """Return the canonical CASTEP spelling of one enum value.

    CASTEP uppercases enum values on read, so a file may legitimately use any
    case. It also folds a handful of documented value aliases. Applying both
    here means a parsed file carries the value CASTEP would have used, which is
    what MuFinder's validators and writers expect.

    A value that matches nothing is returned unchanged rather than coerced, so
    the caller still sees exactly what the file said and validation can report
    it.
    """

    candidate = raw_value.strip().upper()
    if not candidate:
        return raw_value.strip()
    candidate = _VALUE_ALIASES.get(keyword, {}).get(candidate, candidate)
    if spec is not None and spec.allowed_values:
        allowed = {item.upper(): item for item in spec.allowed_values}
        if candidate in allowed:
            return allowed[candidate]
        return raw_value.strip()
    return candidate


@dataclass(frozen=True, slots=True)
class ParamObservation:
    """One assignment read from a ``.param`` file, mapped or not.

    ``keyword`` is the canonical runtime name when the schema recognises the
    assignment, and ``None`` when it does not. An unmapped observation keeps its
    ``raw_name`` and ``raw_value`` so callers can record exactly what the file
    said without inventing a keyword for it.
    """

    raw_name: str
    raw_value: str
    #: The name lowercased and alias-resolved. Always present, even when the
    #: schema does not recognise it, so a caller can key on something stable.
    canonical_name: str = ""
    keyword: str | None = None
    value: object | None = None
    error: str | None = None

    @property
    def mapped(self) -> bool:
        """Return whether the schema recognised this assignment."""

        return self.keyword is not None and self.error is None


@dataclass(frozen=True, slots=True)
class ParsedParam:
    """Structured parse of a ``.param`` file plus its unmapped remainder."""

    param_file: ParamFile
    observations: tuple[ParamObservation, ...] = field(default_factory=tuple)

    @property
    def unmapped(self) -> tuple[ParamObservation, ...]:
        """Return observations the runtime schema did not recognise."""

        return tuple(item for item in self.observations if not item.mapped)


def parse_param_value(raw_value: str, spec: KeywordSpec | None) -> object:
    """Coerce one raw ``.param`` value using its keyword spec.

    Behaviour-preserving extraction of the former
    ``MainWindow._parse_param_scalar_value``. With a ``spec`` the value type
    drives coercion and a malformed value raises :class:`ValueError`; without
    one the token is inferred conservatively (bool, then int, then float, then
    left as text), and multi-token values become a tuple.

    Raises:
        ValueError: If ``spec`` pins a type the value cannot satisfy.
    """

    if spec is not None and spec.value_type == "text":
        return raw_value.strip()

    if spec is not None and spec.value_type == "enum":
        return canonical_enum_value(spec.name, raw_value, spec)

    tokens = shlex.split(raw_value)
    if not tokens:
        return None

    if spec is not None and spec.value_type == "bool":
        token = tokens[0].strip().lower()
        if token in _TRUE_TOKENS:
            return True
        if token in _FALSE_TOKENS:
            return False
        raise ValueError(f"Invalid boolean value '{tokens[0]}' for {spec.name.upper()}.")

    if spec is not None and spec.value_type == "int_triplet":
        if len(tokens) < 3:
            raise ValueError(f"{spec.name.upper()} requires three integer values.")
        return (int(tokens[0]), int(tokens[1]), int(tokens[2]))

    if spec is not None and spec.value_type in {"physical", "positive_float_with_optional_unit"}:
        if len(tokens) == 1:
            return float(tokens[0])
        if len(tokens) >= 2:
            return ValueWithUnit(value=float(tokens[0]), unit=tokens[1])

    if spec is not None and spec.value_type in {"int", "positive_int", "non_negative_int"}:
        return int(tokens[0])

    if spec is not None and spec.value_type in {"float", "number"}:
        return float(tokens[0])

    if len(tokens) == 1:
        token = tokens[0]
        lowered = token.strip().lower()
        if lowered in _UNTYPED_TRUE_TOKENS:
            return True
        if lowered in _UNTYPED_FALSE_TOKENS:
            return False
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            return token

    values: list[object] = []
    for token in tokens:
        try:
            values.append(int(token))
            continue
        except ValueError:
            pass
        try:
            values.append(float(token))
            continue
        except ValueError:
            values.append(token)
    return tuple(values)


def _strip_comment(line: str) -> str:
    """Remove trailing ``!`` and ``#`` comments from one line."""

    for marker in _COMMENT_MARKERS:
        if marker in line:
            line = line.split(marker, maxsplit=1)[0]
    return line.strip()


def _split_assignment(line: str) -> tuple[str, str] | None:
    """Split one ``.param`` line into a raw name and raw value.

    CASTEP accepts both ``NAME : VALUE`` and ``NAME VALUE``. Returns ``None``
    for a line that carries no name at all.
    """

    if ":" in line:
        raw_name, raw_value = line.split(":", maxsplit=1)
        return raw_name.strip(), raw_value
    parts = line.split(maxsplit=1)
    if not parts:
        return None
    return parts[0], parts[1] if len(parts) > 1 else ""


def parse_param_observations(text: str, *, strict: bool = True) -> tuple[ParamObservation, ...]:
    """Read every scalar assignment in ``.param`` text, mapped or not.

    Block contents are skipped; see :func:`parse_param_text` for the block-aware
    parse. With ``strict`` a value that fails coercion raises; otherwise the
    failure is captured on the observation's ``error`` field and the raw text is
    preserved, so one malformed line cannot discard an entire file.

    Raises:
        ValueError: If ``strict`` and a value cannot be coerced to its spec type.
    """

    observations: list[ParamObservation] = []
    in_block = False
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line.strip())
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("%BLOCK "):
            in_block = True
            continue
        if upper.startswith("%ENDBLOCK "):
            in_block = False
            continue
        if in_block:
            continue

        split = _split_assignment(line)
        if split is None:
            continue
        raw_name, raw_value = split

        canonical = canonical_keyword_name("param", raw_name)
        spec = get_keyword_spec("param", canonical)
        if spec is None:
            # No spec: the runtime catalog does not know this keyword. Keep the
            # observation with a conservative literal value so a caller can
            # record what the file said, but leave ``keyword`` unset so it is
            # never mistaken for a supported assignment.
            observations.append(
                ParamObservation(
                    raw_name=raw_name,
                    raw_value=raw_value.strip(),
                    canonical_name=canonical,
                    value=parse_param_value(raw_value.strip(), None),
                )
            )
            continue

        try:
            value = parse_param_value(raw_value.strip(), spec)
        except ValueError as exc:
            if strict:
                raise
            observations.append(
                ParamObservation(
                    raw_name=raw_name,
                    raw_value=raw_value.strip(),
                    canonical_name=canonical,
                    error=str(exc),
                )
            )
            continue

        observations.append(
            ParamObservation(
                raw_name=raw_name,
                raw_value=raw_value.strip(),
                canonical_name=canonical,
                keyword=canonical,
                value=value,
            )
        )
    return tuple(observations)


def parse_param_assignments(text: str, *, strict: bool = True) -> dict[str, object]:
    """Return ``{canonical name: value}`` for every scalar assignment in ``text``.

    This is a faithful view of the file: keywords the runtime schema does not
    recognise are **included**, under their canonical (lowercased,
    alias-resolved) name and with a conservatively inferred value, because a
    caller asking for "the file as a dict" wants what the file said. Use
    :func:`parse_param_observations` when the mapped/unmapped split matters, and
    :func:`parse_param_text` when only schema-backed keywords should survive.

    Raises:
        ValueError: If ``strict`` and a value cannot be coerced to its spec type.
    """

    return {
        observation.canonical_name: observation.value
        for observation in parse_param_observations(text, strict=strict)
        if observation.canonical_name and observation.error is None
    }


def _parse_block_rows(lines: list[str]) -> tuple[str | None, tuple[tuple[str, ...], ...]]:
    """Split gathered block lines into an optional unit and whitespace rows."""

    unit: str | None = None
    rows: list[tuple[str, ...]] = []
    for index, line in enumerate(lines):
        tokens = tuple(line.split())
        if not tokens:
            continue
        if index == 0 and len(tokens) == 1 and not _looks_numeric(tokens[0]):
            unit = tokens[0]
            continue
        rows.append(tokens)
    return unit, tuple(rows)


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def parse_param_text(text: str, *, strict: bool = True) -> ParsedParam:
    """Parse ``.param`` text into a :class:`ParsedParam`.

    The returned :attr:`ParsedParam.param_file` holds recognised scalars as
    :class:`~mufinder.castep_io.types.KeywordAssignment` values and every block
    as a :class:`~mufinder.castep_io.types.BlockSection`, so it can be handed
    back to :func:`~mufinder.castep_io.param_writer.write_param`. Blocks are
    preserved verbatim as string rows: ``.param`` blocks have no shared row
    schema, so imposing one here would mean inventing syntax.

    Raises:
        ParamParseError: If a block is unterminated or mismatched.
        ValueError: If ``strict`` and a value cannot be coerced to its spec type.
    """

    blocks: list[BlockSection] = []
    current_name: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line.strip())
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("%BLOCK "):
            if current_name is not None:
                raise ParamParseError(f"Nested %BLOCK is not allowed ({current_name}).")
            current_name = line.split(maxsplit=1)[1].strip().lower()
            current_lines = []
            continue
        if upper.startswith("%ENDBLOCK "):
            if current_name is None:
                raise ParamParseError("Found %ENDBLOCK without matching %BLOCK.")
            end_name = line.split(maxsplit=1)[1].strip().lower()
            if end_name != current_name:
                raise ParamParseError(
                    f"Mismatched block terminator %ENDBLOCK {end_name} for %BLOCK {current_name}."
                )
            unit, rows = _parse_block_rows(current_lines)
            blocks.append(BlockSection(name=current_name, rows=rows, unit=unit))
            current_name = None
            current_lines = []
            continue
        if current_name is not None:
            current_lines.append(line)

    if current_name is not None:
        raise ParamParseError(f"Unterminated block %BLOCK {current_name}.")

    observations = parse_param_observations(text, strict=strict)
    keywords = tuple(
        KeywordAssignment(name=observation.keyword, value=observation.value)
        for observation in observations
        if observation.mapped and observation.keyword is not None
    )
    return ParsedParam(
        param_file=ParamFile(keywords=keywords, blocks=tuple(blocks)),
        observations=observations,
    )


def parse_param_file(path: str | Path, *, strict: bool = True) -> ParsedParam:
    """Parse a CASTEP ``.param`` file from disk.

    Raises:
        ParamParseError: If the file cannot be read, or its blocks are malformed.
        ValueError: If ``strict`` and a value cannot be coerced to its spec type.
    """

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ParamParseError(f"Failed to read PARAM file '{source}': {exc}") from exc
    return parse_param_text(text, strict=strict)


__all__ = [
    "ParamObservation",
    "canonical_enum_value",
    "ParamParseError",
    "ParsedParam",
    "parse_param_assignments",
    "parse_param_file",
    "parse_param_observations",
    "parse_param_text",
    "parse_param_value",
]
