"""The shared tabular writer: header, units, escaping, non-finite values and round trips."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from mufinder.services.project_service import ProjectService
from mufinder.services.tabular_export import (
    ExportProvenance,
    TableColumn,
    TableExport,
    TableFormat,
    TableSection,
    export_provenance,
    render_table_export,
    write_table_export,
)
from tests.support.jobs import make_job

WRITTEN_AT = datetime(2026, 11, 23, 10, 15, 0, tzinfo=UTC)


def _export(*sections: TableSection, notes: tuple[str, ...] = ()) -> TableExport:
    return TableExport(
        title="MuFinder test export",
        sections=sections
        or (
            TableSection(
                (
                    TableColumn("E", [-1.5, 0.25], unit="eV"),
                    TableColumn("d", [0.0, 1.75], unit="Å"),
                    TableColumn("weight", [1.0, 0.5]),
                )
            ),
        ),
        provenance=ExportProvenance("MnSi", ("mu_relax_01 (id 1)", "uep_01 (id 2)")),
        notes=notes,
    )


def _data_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line and not line.startswith("#")]


def test_dat_export_starts_with_the_commented_header() -> None:
    text = render_table_export(
        _export(notes=("Simulation: ZF powder",)), TableFormat.DAT, written_at=WRITTEN_AT
    )

    assert text.splitlines() == [
        "# MuFinder test export",
        "# Project: MnSi",
        "# Source jobs: mu_relax_01 (id 1); uep_01 (id 2)",
        "# Exported: 2026-11-23T10:15:00+00:00",
        "# Simulation: ZF powder",
        "# Columns: E d weight",
        "# Units: eV Å -",
        "-1.5 0 1",
        "0.25 1.75 0.5",
    ]


def test_csv_export_separates_names_units_and_values_by_commas() -> None:
    text = render_table_export(_export(), TableFormat.CSV, written_at=WRITTEN_AT)

    assert text.splitlines()[4:] == [
        "# Columns: E,d,weight",
        "# Units: eV,Å,-",
        "-1.5,0,1",
        "0.25,1.75,0.5",
    ]


def test_exported_time_is_the_given_time_with_its_offset() -> None:
    local = datetime(2026, 1, 2, 3, 4, 5, 678, tzinfo=timezone(timedelta(hours=1)))

    text = render_table_export(_export(), TableFormat.DAT, written_at=local)

    assert "# Exported: 2026-01-02T03:04:05+01:00" in text.splitlines()
    # Everything else is the same whenever it is written.
    other = render_table_export(_export(), TableFormat.DAT, written_at=WRITTEN_AT)
    assert [line for line in text.splitlines() if not line.startswith("# Exported")] == [
        line for line in other.splitlines() if not line.startswith("# Exported")
    ]


def test_missing_project_and_jobs_read_none() -> None:
    export = TableExport(title="T", sections=())

    lines = render_table_export(export, TableFormat.DAT, written_at=WRITTEN_AT).splitlines()

    assert lines[1:3] == ["# Project: none", "# Source jobs: none"]


def test_sections_are_titled_and_separated_by_one_blank_line() -> None:
    export = _export(
        TableSection((TableColumn("t_us", [0.0, 0.1], unit="μs"),), title="Time domain"),
        TableSection((TableColumn("f_MHz", [0.0], unit="MHz"),), title="Spectrum"),
    )

    lines = render_table_export(export, TableFormat.DAT, written_at=WRITTEN_AT).splitlines()

    assert lines[4:] == [
        "# Time domain",
        "# Columns: t_us",
        "# Units: μs",
        "0",
        "0.1",
        "",
        "# Spectrum",
        "# Columns: f_MHz",
        "# Units: MHz",
        "0",
    ]


@pytest.mark.parametrize(
    ("fmt", "columns_line"),
    [
        (TableFormat.DAT, '# Columns: "B, total" "the ""p"" value" "spin up"'),
        (TableFormat.CSV, '# Columns: "B, total","the ""p"" value",spin up'),
    ],
)
def test_names_with_commas_quotes_and_spaces_are_quoted(fmt, columns_line) -> None:
    export = _export(
        TableSection(
            (
                TableColumn("B, total", [1.0], unit="T"),
                TableColumn('the "p" value', [0.5]),
                TableColumn("spin up", [2.0], unit="μ_B per atom"),
            )
        )
    )

    text = render_table_export(export, fmt, written_at=WRITTEN_AT)

    assert columns_line in text.splitlines()
    names_line = next(line for line in text.splitlines() if line.startswith("# Columns: "))
    delimiter = "," if fmt is TableFormat.CSV else " "
    names = next(
        csv.reader([names_line.removeprefix("# Columns: ")], delimiter=delimiter, quotechar='"')
    )
    assert names == ["B, total", 'the "p" value', "spin up"]
    units_line = next(line for line in text.splitlines() if line.startswith("# Units: "))
    units = next(csv.reader([units_line.removeprefix("# Units: ")], delimiter=delimiter))
    assert units == ["T", "-", "μ_B per atom"]


def test_line_breaks_in_header_text_stay_in_comment_lines() -> None:
    export = TableExport(
        title="Title\nsecond line",
        sections=(TableSection((TableColumn("a\nb", [1.0], unit="e\nV"),)),),
        provenance=ExportProvenance("Two\nlines"),
        notes=("note\r\nmore",),
    )

    text = render_table_export(export, TableFormat.DAT, written_at=WRITTEN_AT)

    assert _data_rows(text) == ["1"]
    assert '# Columns: "a b"' in text.splitlines()
    assert '# Units: "e V"' in text.splitlines()
    assert "# more" in text.splitlines()


def test_non_finite_values_are_written_as_nan_and_inf() -> None:
    export = _export(TableSection((TableColumn("x", [np.nan, np.inf, -np.inf, 1e-300]),)))

    text = render_table_export(export, TableFormat.DAT, written_at=WRITTEN_AT)

    assert _data_rows(text) == ["nan", "inf", "-inf", "1e-300"]


def test_values_keep_twelve_significant_digits() -> None:
    export = _export(TableSection((TableColumn("x", [1 / 3, 123456789.123456789, -0.0]),)))

    text = render_table_export(export, TableFormat.DAT, written_at=WRITTEN_AT)

    assert _data_rows(text) == ["0.333333333333", "123456789.123", "-0"]


def test_full_precision_values_read_back_as_the_same_doubles(tmp_path: Path) -> None:
    values = [1 / 3, -69185.30055728927, 2.0, -0.0, np.nan, np.inf, 1e-300]
    export = TableExport(
        title="MuFinder test export",
        sections=(TableSection((TableColumn("x", values),)),),
        full_precision=True,
    )

    text = render_table_export(export, TableFormat.DAT, written_at=WRITTEN_AT)
    path = write_table_export(export, tmp_path / "x.dat", written_at=WRITTEN_AT)

    assert _data_rows(text) == [
        "0.3333333333333333",
        "-69185.30055728927",
        "2",
        "-0",
        "nan",
        "inf",
        "1e-300",
    ]
    assert np.array_equal(np.loadtxt(path), values, equal_nan=True)


def test_columns_of_different_lengths_are_refused() -> None:
    export = _export(TableSection((TableColumn("a", [1.0, 2.0]), TableColumn("b", [1.0]))))

    with pytest.raises(ValueError, match="a: 2, b: 1"):
        render_table_export(export, TableFormat.DAT, written_at=WRITTEN_AT)


def _round_trip_table() -> tuple[TableExport, np.ndarray]:
    values = np.array(
        [[0.0, -2.5, np.nan], [0.1, 1e-12, np.inf], [0.2, 3.25e7, -np.inf]], dtype=float
    )
    export = _export(
        TableSection(
            (
                TableColumn("E, relative", values[:, 0], unit="eV"),
                TableColumn('"d"', values[:, 1], unit="Å"),
                TableColumn("with space", values[:, 2]),
            )
        ),
        notes=("a note, with a comma", 'and "quotes"'),
    )
    return export, values


@pytest.mark.parametrize(("suffix", "delimiter"), [(".dat", None), (".txt", None), (".csv", ",")])
def test_written_file_reads_back_with_numpy_loadtxt(tmp_path: Path, suffix, delimiter) -> None:
    export, values = _round_trip_table()
    path = write_table_export(export, tmp_path / f"table{suffix}", written_at=WRITTEN_AT)

    loaded = np.loadtxt(path, delimiter=delimiter, encoding="utf-8")

    np.testing.assert_array_equal(loaded, values)


@pytest.mark.parametrize(("suffix", "delimiter"), [(".dat", " "), (".csv", ",")])
def test_written_file_reads_back_with_the_csv_module(tmp_path: Path, suffix, delimiter) -> None:
    export, values = _round_trip_table()
    path = write_table_export(export, tmp_path / f"table{suffix}", written_at=WRITTEN_AT)

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(
            csv.reader((line for line in handle if not line.startswith("#")), delimiter=delimiter)
        )

    np.testing.assert_array_equal(np.array(rows, dtype=float), values)


def test_format_follows_the_path_suffix(tmp_path: Path) -> None:
    assert TableFormat.for_path("a.CSV") is TableFormat.CSV
    assert TableFormat.for_path("a.dat") is TableFormat.DAT
    assert TableFormat.for_path("a.txt") is TableFormat.DAT

    path = write_table_export(_export(), tmp_path / "a.csv", written_at=WRITTEN_AT)
    assert path.read_text(encoding="utf-8") == render_table_export(
        _export(), TableFormat.CSV, written_at=WRITTEN_AT
    )


def test_written_file_is_stamped_with_the_current_local_time(tmp_path: Path) -> None:
    before = datetime.now().astimezone().replace(microsecond=0)

    path = write_table_export(_export(), tmp_path / "a.dat")

    stamp = next(
        line.removeprefix("# Exported: ")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("# Exported: ")
    )
    written = datetime.fromisoformat(stamp)
    assert written.tzinfo is not None
    assert before <= written <= datetime.now().astimezone()


def test_writing_to_a_missing_folder_raises_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        write_table_export(_export(), tmp_path / "missing" / "a.dat", written_at=WRITTEN_AT)


def test_export_provenance_names_the_project_and_each_job_once(tmp_path: Path) -> None:
    service = ProjectService()
    document = service.create_project(tmp_path / "private-folder" / "p.mufinder", "MnSi")
    job = make_job("mu_relax_01", service=service, document=document)
    gone = uuid4()

    provenance = export_provenance(document, [job.id, str(job.id), None, "not-a-uuid", gone])

    assert provenance == ExportProvenance(
        project_name="MnSi", source_jobs=(f"mu_relax_01 (id {job.id})", f"id {gone}")
    )
    text = render_table_export(
        TableExport("T", (), provenance=provenance), TableFormat.DAT, written_at=WRITTEN_AT
    )
    assert "private-folder" not in text
    assert str(tmp_path) not in text


def test_export_provenance_without_a_project_is_empty() -> None:
    assert export_provenance(None, [uuid4()]) == ExportProvenance()
