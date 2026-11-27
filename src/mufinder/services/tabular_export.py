"""Write numeric results as CSV or whitespace-separated ``.dat`` text with a commented header.

The numeric data exports (the field distribution and the polarisation
simulation) go through :func:`write_table_export`; the Journal and the
comparison-table CSV stay plain files for text readers and spreadsheets. A
:class:`TableExport` holds a title, where the numbers came from
(:class:`ExportProvenance`: the project name and the source jobs), free-form
notes and one or more :class:`TableSection` blocks of equal-length
:class:`TableColumn` columns. The file reads::

    # MuFinder field distribution export
    # Project: MnSi
    # Source jobs: mu_relax_01 (id 3f2c…)
    # Exported: 2026-11-23T10:15:00+00:00
    # <notes, one comment line each>
    # <section title, when it has one>
    # Columns: B_T p_site_1
    # Units: T -
    0.1 0.25
    …

Every header line starts with ``#``, so ``numpy.loadtxt`` (with
``delimiter=","`` for CSV) and a :mod:`csv` reader that skips comment lines read
the numbers alone. The ``Columns`` and ``Units`` lines are rows in the file's
own format after ``# Columns: `` and ``# Units: ``: comma-separated with CSV
quoting for CSV, space-separated with double quotes around a name that holds
whitespace or a quote (doubled inside) for ``.dat``. A dimensionless column's
unit is ``-``. Sections are separated by one blank line. Numbers are written
with twelve significant digits, or, for an export with ``full_precision``, as
the shortest text that reads back as the same double (a whole number without
``.0``); NaN and infinities as ``nan``, ``inf`` and ``-inf``. The output is
deterministic apart from the ``Exported`` line, whose time is a parameter.

Layering: this lives in ``mufinder.services``; :func:`export_provenance` reads
the project document, everything else is plain text.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import io
from pathlib import Path
from uuid import UUID

import numpy as np

from mufinder.services.project_document import ProjectDocument

#: The unit a dimensionless column shows in the ``Units`` line.
DIMENSIONLESS_UNIT = "-"
#: What the ``Project`` and ``Source jobs`` lines say when there is nothing to name.
_NONE_TEXT = "none"


class TableFormat(StrEnum):
    """The text layout of a table export."""

    CSV = "csv"
    DAT = "dat"

    @classmethod
    def for_path(cls, path: str | Path) -> TableFormat:
        """CSV for a ``.csv`` path (any case), whitespace-separated ``.dat`` text otherwise."""

        return cls.CSV if Path(path).suffix.lower() == ".csv" else cls.DAT


@dataclass(frozen=True, slots=True)
class TableColumn:
    """One column: its name (``B_T``), its values and its unit (``T``; empty when dimensionless)."""

    name: str
    values: Sequence[float] | np.ndarray
    unit: str = ""


@dataclass(frozen=True, slots=True)
class TableSection:
    """A block of equal-length columns, with an optional title line above its column names."""

    columns: Sequence[TableColumn]
    title: str = ""


@dataclass(frozen=True, slots=True)
class ExportProvenance:
    """Where an export's numbers came from: the project's name and its source jobs, as text."""

    project_name: str = ""
    source_jobs: Sequence[str] = ()


@dataclass(frozen=True, slots=True)
class TableExport:
    """A numeric export: its title line, provenance, notes and sections.

    ``full_precision`` writes every number so that it reads back as the same
    double, for an export whose numbers must equal the ones shown.
    """

    title: str
    sections: Sequence[TableSection]
    provenance: ExportProvenance = ExportProvenance()
    notes: Sequence[str] = ()
    full_precision: bool = False


def export_provenance(
    document: ProjectDocument | None, job_ids: Iterable[UUID | str | None] = ()
) -> ExportProvenance:
    """The project name and the jobs ``job_ids`` name, each once, in order.

    A job reads ``label (id <uuid>)``; one no longer in the project reads
    ``id <uuid>``. Ids that are not UUIDs are skipped. Only the project's name
    is used, never its folder.
    """

    if document is None:
        return ExportProvenance()
    seen: set[UUID] = set()
    jobs: list[str] = []
    for raw in job_ids:
        try:
            job_id = raw if isinstance(raw, UUID) else UUID(str(raw))
        except ValueError:
            continue
        if job_id in seen:
            continue
        seen.add(job_id)
        job = document.jobs.get(job_id)
        jobs.append(
            f"{job.label} (id {job_id})" if job is not None and job.label else f"id {job_id}"
        )
    return ExportProvenance(project_name=document.project.name, source_jobs=tuple(jobs))


def render_table_export(export: TableExport, fmt: TableFormat, *, written_at: datetime) -> str:
    """Return ``export`` as text in ``fmt``, stamped with ``written_at``.

    Raises ``ValueError`` when the columns of a section differ in length.
    """

    provenance = export.provenance
    lines = _comment_lines(export.title)
    lines += _comment_lines(f"Project: {provenance.project_name.strip() or _NONE_TEXT}")
    lines += _comment_lines(f"Source jobs: {'; '.join(provenance.source_jobs) or _NONE_TEXT}")
    lines.append(f"# Exported: {written_at.isoformat(timespec='seconds')}")
    for note in export.notes:
        lines += _comment_lines(note)
    for index, section in enumerate(export.sections):
        if index:
            lines.append("")
        lines += _section_lines(section, fmt, full_precision=export.full_precision)
    return "\n".join(lines) + "\n"


def write_table_export(
    export: TableExport,
    path: str | Path,
    *,
    fmt: TableFormat | None = None,
    written_at: datetime | None = None,
) -> Path:
    """Write ``export`` to ``path`` and return it.

    ``fmt`` defaults to :meth:`TableFormat.for_path`, and ``written_at`` to now,
    in local time.
    """

    target = Path(path)
    text = render_table_export(
        export,
        fmt or TableFormat.for_path(target),
        written_at=written_at or datetime.now().astimezone(),
    )
    target.write_text(text, encoding="utf-8")
    return target


def _section_lines(
    section: TableSection, fmt: TableFormat, *, full_precision: bool = False
) -> list[str]:
    columns = [np.asarray(column.values, dtype=float).ravel() for column in section.columns]
    lengths = {len(values) for values in columns}
    if len(lengths) > 1:
        raise ValueError(
            "the columns of a table section differ in length ("
            + ", ".join(
                f"{column.name}: {len(values)}"
                for column, values in zip(section.columns, columns, strict=True)
            )
            + ")"
        )
    lines = _comment_lines(section.title) if section.title else []
    names = [_one_line(column.name) for column in section.columns]
    units = [_one_line(column.unit) or DIMENSIONLESS_UNIT for column in section.columns]
    lines.append("# Columns: " + _text_row(names, fmt))
    lines.append("# Units: " + _text_row(units, fmt))
    separator = "," if fmt is TableFormat.CSV else " "
    for row in zip(*columns, strict=True):
        lines.append(separator.join(_number_text(value, full_precision) for value in row))
    return lines


def _number_text(value: float, full_precision: bool) -> str:
    number = float(value)
    if not full_precision:
        return f"{number:.12g}"
    text = repr(number)
    # A whole number reads back the same without its ".0", as an index column should.
    return text[:-2] if text.endswith(".0") else text


def _text_row(fields: Sequence[str], fmt: TableFormat) -> str:
    if fmt is TableFormat.CSV:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerow(fields)
        return buffer.getvalue().rstrip("\n")
    return " ".join(_dat_field(field) for field in fields)


def _dat_field(text: str) -> str:
    if text and not any(character.isspace() or character == '"' for character in text):
        return text
    return '"' + text.replace('"', '""') + '"'


def _one_line(text: str) -> str:
    return " ".join(str(text).splitlines()).strip()


def _comment_lines(text: str) -> list[str]:
    """``text`` as comment lines, one per line it holds, so no line of it escapes the header."""

    return [f"# {line}".rstrip() for line in str(text).splitlines() or [""]]


__all__ = [
    "DIMENSIONLESS_UNIT",
    "ExportProvenance",
    "TableColumn",
    "TableExport",
    "TableFormat",
    "TableSection",
    "export_provenance",
    "render_table_export",
    "write_table_export",
]
