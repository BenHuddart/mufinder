"""Tests for the keyword-level comparison of a job's ``.cell`` and ``.param``."""

from __future__ import annotations

from pathlib import Path

from mufinder.castep_io.input_diff import (
    InputKeywordChange,
    JobInputTexts,
    diff_input_text,
    diff_job_inputs,
    overlay_job_inputs,
    read_job_inputs,
    split_input_sections,
    with_exclusive_partners,
    write_job_inputs,
)

_CELL = """\
%BLOCK LATTICE_CART
ang
5 0 0
0 5 0
0 0 5
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0 0 0
%ENDBLOCK POSITIONS_FRAC

SYMMETRY_GENERATE

FIX_ALL_CELL : FALSE
"""

_PARAM = """\
TASK : GEOMETRYOPTIMIZATION
CUT_OFF_ENERGY : 500 eV
GEOM_MAX_ITER : 120
"""


def _summary(changes: tuple[InputKeywordChange, ...]) -> list[tuple[str, str, str]]:
    return [(change.file, change.keyword, change.change) for change in changes]


def test_split_reads_scalars_flags_and_blocks() -> None:
    sections = split_input_sections(_CELL, "cell")

    assert sections.scalars == {"symmetry_generate": "", "fix_all_cell": "FALSE"}
    assert sections.blocks["positions_frac"] == ("Si 0 0 0",)
    assert sections.blocks["lattice_cart"] == ("ang", "5 0 0", "0 5 0", "0 0 5")


def test_split_is_lenient_about_a_malformed_block() -> None:
    sections = split_input_sections("%ENDBLOCK STRAY\n%BLOCK SPECIES_POT\nSi Si_00.usp\n", "cell")

    assert sections.blocks == {"species_pot": ("Si Si_00.usp",)}


def test_split_ignores_a_block_header_without_a_name() -> None:
    sections = split_input_sections("%BLOCK\nFIX_ALL_CELL : TRUE\n", "cell")

    assert sections.scalars == {"fix_all_cell": "TRUE"}
    assert sections.blocks == {}


def test_unit_case_and_the_default_unit_are_not_a_change() -> None:
    assert diff_input_text("CUT_OFF_ENERGY : 500 eV", "cut_off_energy : 500 EV", "param") == ()
    assert diff_input_text("CUT_OFF_ENERGY : 500", "CUT_OFF_ENERGY : 500 eV", "param") == ()


def test_numbers_in_a_block_compare_by_value() -> None:
    old = "%BLOCK SPECIES_MASS\nAMU\nH:mu 0.1134\n%ENDBLOCK SPECIES_MASS"
    same = "%BLOCK SPECIES_MASS\namu\nH:mu 0.11340\n%ENDBLOCK SPECIES_MASS"
    other = "%BLOCK SPECIES_MASS\nAMU\nH:mu 0.1135\n%ENDBLOCK SPECIES_MASS"

    assert diff_input_text(old, same, "cell") == ()
    assert [change.change for change in diff_input_text(old, other, "cell")] == ["Changed"]


def test_identical_inputs_have_no_changes() -> None:
    inputs = JobInputTexts(cell=_CELL, param=_PARAM)

    assert diff_job_inputs(inputs, inputs) == ()


def test_formatting_alone_is_not_a_change() -> None:
    reformatted = (
        "task: geometryoptimization\ncut_off_energy   500.0 eV ! comment\nGEOM_MAX_ITER : 120\n"
    )

    assert diff_input_text(_PARAM, reformatted, "param") == ()


def test_a_removed_cell_flag_is_listed() -> None:
    old = JobInputTexts(cell=_CELL, param=_PARAM)
    new = JobInputTexts(cell=_CELL.replace("SYMMETRY_GENERATE\n", ""), param=_PARAM)

    changes = diff_job_inputs(old, new)

    assert changes == (
        InputKeywordChange(
            file="cell",
            keyword="SYMMETRY_GENERATE",
            change="Removed",
            is_block=False,
            old_lines=("",),
            new_lines=(),
        ),
    )


def test_added_and_changed_keywords_across_both_files() -> None:
    old = JobInputTexts(cell=_CELL, param=_PARAM)
    new = JobInputTexts(
        cell=_CELL + "KPOINT_MP_GRID 3 3 3\n",
        param=_PARAM.replace("GEOM_MAX_ITER : 120", "GEOM_MAX_ITER : 250"),
    )

    changes = diff_job_inputs(old, new)

    assert _summary(changes) == [
        ("cell", "KPOINT_MP_GRID", "Added"),
        ("param", "GEOM_MAX_ITER", "Changed"),
    ]
    assert changes[0].new_lines == ("3 3 3",)
    assert (changes[1].old_lines, changes[1].new_lines) == (("120",), ("250",))


def test_a_changed_block_is_listed_with_its_lines() -> None:
    old = _CELL + "%BLOCK SPECIES_POT\nSi Si_00.usp\n%ENDBLOCK SPECIES_POT\n"
    new = _CELL + "%BLOCK SPECIES_POT\nSi Si_C19.usp\nH H_00.usp\n%ENDBLOCK SPECIES_POT\n"

    changes = diff_input_text(old, new, "cell")

    assert changes == (
        InputKeywordChange(
            file="cell",
            keyword="SPECIES_POT",
            change="Changed",
            is_block=True,
            old_lines=("Si Si_00.usp",),
            new_lines=("Si Si_C19.usp", "H H_00.usp"),
        ),
    )


def test_structure_blocks_are_not_parameter_changes() -> None:
    old = JobInputTexts(cell=_CELL, param=_PARAM)
    moved = _CELL.replace("Si 0 0 0", "Si 0.01 0 0").replace("5 0 0", "5.1 0 0")

    assert diff_job_inputs(old, JobInputTexts(cell=moved, param=_PARAM)) == ()


def test_read_job_inputs_reads_both_files(tmp_path: Path) -> None:
    (tmp_path / "relax.cell").write_text(_CELL, encoding="utf-8")
    (tmp_path / "relax.param").write_text(_PARAM, encoding="utf-8")

    assert read_job_inputs(tmp_path, "relax") == JobInputTexts(cell=_CELL, param=_PARAM)


def test_read_job_inputs_is_none_when_a_file_is_missing(tmp_path: Path) -> None:
    (tmp_path / "relax.param").write_text(_PARAM, encoding="utf-8")

    assert read_job_inputs(tmp_path, "relax") is None


def test_overlay_keeps_every_other_line_of_the_base_as_written() -> None:
    base = JobInputTexts(cell=_CELL, param=_PARAM.replace("120", "120 ! the run's own"))
    new = JobInputTexts(
        cell=_CELL + "KPOINT_MP_GRID : 4 4 4\n",
        param="TASK : GEOMETRYOPTIMIZATION\nCUT_OFF_ENERGY : 900 eV\nGEOM_MAX_ITER : 300\n",
    )

    assert overlay_job_inputs(base, new, ()) == base


def test_overlay_takes_the_named_keywords_where_they_stand() -> None:
    base = JobInputTexts(cell=_CELL, param=_PARAM)
    new = JobInputTexts(
        cell=_CELL + "KPOINT_MP_SPACING : 0.05 1/ang\n",
        param="TASK : GEOMETRYOPTIMIZATION\nCUT_OFF_ENERGY : 900 eV\nGEOM_MAX_ITER : 300\n",
    )

    result = overlay_job_inputs(base, new, ("CUT_OFF_ENERGY", "kpoint_mp_spacing"))

    assert result.param == _PARAM.replace("500 eV", "900 eV")
    assert result.cell == _CELL + "KPOINT_MP_SPACING : 0.05 1/ang\n"
    assert _summary(diff_job_inputs(base, result)) == [
        ("cell", "KPOINT_MP_SPACING", "Added"),
        ("param", "CUT_OFF_ENERGY", "Changed"),
    ]


def test_overlay_takes_a_named_keywords_partners_so_a_pair_never_holds_both() -> None:
    base = JobInputTexts(cell=_CELL + "KPOINT_MP_GRID : 4 4 4\n", param=_PARAM)
    new = JobInputTexts(cell=_CELL + "KPOINT_MP_SPACING : 0.05 1/ang\n", param=_PARAM)

    result = overlay_job_inputs(base, new, ("kpoint_mp_spacing",))

    assert "KPOINT_MP_GRID" not in result.cell
    assert "KPOINT_MP_SPACING : 0.05 1/ang" in result.cell


def test_overlay_replaces_and_drops_whole_blocks() -> None:
    block = "%BLOCK SPECIES_POT\nSi Si_00.usp\n%ENDBLOCK SPECIES_POT\n"
    base = JobInputTexts(cell=_CELL + block, param=_PARAM)
    replaced = JobInputTexts(cell=_CELL + block.replace("00", "C19"), param=_PARAM)

    assert overlay_job_inputs(base, replaced, ("species_pot",)).cell == replaced.cell
    assert overlay_job_inputs(base, JobInputTexts(cell=_CELL, param=_PARAM), ("species_pot",)) == (
        JobInputTexts(cell=_CELL, param=_PARAM)
    )


def test_with_exclusive_partners_adds_each_pairs_other_member() -> None:
    assert {"basis_precision", "cut_off_energy"} <= with_exclusive_partners({"CUT_OFF_ENERGY"})
    assert {"kpoint_mp_grid", "kpoint_mp_spacing"} <= with_exclusive_partners({"kpoint_mp_grid"})
    assert with_exclusive_partners({"geom_max_iter"}) == {"geom_max_iter"}


def test_write_job_inputs_writes_what_read_job_inputs_reads(tmp_path: Path) -> None:
    inputs = JobInputTexts(cell=_CELL, param=_PARAM)

    write_job_inputs(tmp_path, "relax", inputs)

    assert read_job_inputs(tmp_path, "relax") == inputs
