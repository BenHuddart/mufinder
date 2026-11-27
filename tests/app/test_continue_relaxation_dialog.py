"""Tests for the Continue Relaxation dialog (parameter editor + live diff)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from mufinder.app.continue_relaxation_dialog import ContinueRelaxationDialog, RenderJobInputs
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.workflow_parameters import kpoint_exclusive_hint, reorder_workflow_parameter_specs
from mufinder.castep_io.input_diff import JobInputTexts
from mufinder.castep_io.schema import KeywordSpec
from mufinder.services.cluster_runner import CheckpointProbeResult
from tests.support.qt import assert_tab_leaves_table


def _continue_specs() -> list[KeywordSpec]:
    # Deliberately out of the required shared-first order, so ordering
    # assertions actually exercise reorder_workflow_parameter_specs rather
    # than passing by accident.
    return [
        KeywordSpec(
            name="task",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("SINGLEPOINT", "GEOMETRYOPTIMIZATION"),
            widget_hint="dropdown",
            category="workflow",
            basic_workflows=("pristine_relaxation",),
            default_value="SINGLEPOINT",
        ),
        KeywordSpec(
            name="geom_max_iter",
            file="param",
            kind="scalar",
            value_type="int",
            widget_hint="spinbox_int",
            category="geometry",
            basic_workflows=("pristine_relaxation",),
            default_value=100,
        ),
        KeywordSpec(
            name="kpoint_mp_grid",
            file="cell",
            kind="scalar",
            value_type="int_triplet",
            widget_hint="int_triplet",
            category="kpoints",
            basic_workflows=("pristine_relaxation",),
            default_value=(2, 2, 2),
        ),
        KeywordSpec(
            name="elec_energy_tol",
            file="param",
            kind="scalar",
            value_type="physical",
            unit="eV",
            widget_hint="unit_float",
            category="electronic",
            basic_workflows=("pristine_relaxation",),
            default_value=1e-5,
        ),
        KeywordSpec(
            name="cut_off_energy",
            file="param",
            kind="scalar",
            value_type="positive_float_with_optional_unit",
            unit="eV",
            widget_hint="unit_float",
            category="electronic",
            basic_workflows=("pristine_relaxation",),
            default_value=340.0,
        ),
        KeywordSpec(
            name="basis_precision",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("FINE", "PRECISE"),
            widget_hint="dropdown",
            category="electronic",
            basic_workflows=("pristine_relaxation",),
            default_value="FINE",
        ),
        KeywordSpec(
            name="num_dump_cycles",
            file="param",
            kind="scalar",
            value_type="int",
            widget_hint="spinbox_int",
            category="io",
            default_value=5,
            # basic_workflows left empty -> advanced-only for this stage.
        ),
    ]


_LOCKED_VALUES = {"task": "GEOMETRYOPTIMIZATION"}
_SPEC_FILES = {spec.name: spec.file for spec in _continue_specs()}


def _render_inputs(values: Mapping[str, object], advanced: tuple[str, ...]) -> JobInputTexts:
    """Deterministic fake renderer: one ``KEY : value`` line per entry, in its file."""

    lines: dict[str, list[str]] = {"cell": [], "param": []}
    for name, value in sorted(values.items()):
        if isinstance(value, tuple):
            value = " ".join(str(item) for item in value)
        lines[_SPEC_FILES.get(name, "param")].append(f"{name.upper()} : {value}")
    return JobInputTexts(cell="\n".join(lines["cell"]), param="\n".join(lines["param"]))


def _effective_inputs(
    render: RenderJobInputs,
    *,
    parameter_values: Mapping[str, object],
    added_advanced: Iterable[str] = (),
) -> JobInputTexts | None:
    """Render the row's *fully resolved* inputs (defaults + overrides).

    Builds the same editor the dialog builds and renders its unedited values,
    standing in for the files on disk of a run that used exactly those values.
    """

    reference = ParameterDialog(
        specs=reorder_workflow_parameter_specs(_continue_specs()),
        workflow_stage="pristine_relaxation",
        locked_values=_LOCKED_VALUES,
        values=dict(parameter_values),
        preferred_exclusive=kpoint_exclusive_hint([parameter_values]),
        advanced_names=tuple(added_advanced),
    )
    try:
        values = dict(reference.collect_values())
        values.pop("task", None)
        return render(values, reference.advanced_names())
    finally:
        reference.close()


def _make_dialog(**overrides: object) -> ContinueRelaxationDialog:
    parameter_values = overrides.pop(
        "parameter_values", {"basis_precision": "FINE", "cut_off_energy": 340.0}
    )
    added_advanced = overrides.pop("added_advanced", ())
    render = overrides.pop("render_inputs", _render_inputs)

    kwargs: dict[str, object] = {
        "specs": _continue_specs(),
        "workflow_stage": "pristine_relaxation",
        "parameter_values": parameter_values,
        "added_advanced": added_advanced,
        "locked_values": _LOCKED_VALUES,
        "old_inputs": _effective_inputs(
            render,  # type: ignore[arg-type]
            parameter_values=parameter_values,  # type: ignore[arg-type]
            added_advanced=added_advanced,  # type: ignore[arg-type]
        ),
        "render_inputs": render,
        "checkpoint_probe": CheckpointProbeResult.found(),
        "execution_target": "local",
        "reused_directory": "/work/jobs/relax_unit",
    }
    kwargs.update(overrides)
    return ContinueRelaxationDialog(**kwargs)  # type: ignore[arg-type]


def _diff_rows(dialog: ContinueRelaxationDialog) -> list[tuple[str, ...]]:
    table = dialog._diff_table
    return [
        tuple(table.item(row, column).text() for column in range(table.columnCount()))
        for row in range(table.rowCount())
    ]


def test_seeds_editor_values_from_row_parameters(qapp) -> None:
    dialog = _make_dialog(
        parameter_values={"basis_precision": "PRECISE", "geom_max_iter": 250},
    )

    assert dialog._rows["basis_precision"].editor.value() == "PRECISE"
    assert dialog._rows["geom_max_iter"].editor.value() == 250
    # Untouched fields keep their spec defaults.
    assert dialog._rows["cut_off_energy"].editor.value().value == 340.0


def test_shared_params_come_first_in_the_stable_order(qapp) -> None:
    dialog = _make_dialog()

    visible_basic_order = [
        name
        for name in dialog._rows
        if name
        in {
            "basis_precision",
            "cut_off_energy",
            "elec_energy_tol",
            "kpoint_mp_grid",
            "geom_max_iter",
        }
    ]

    assert visible_basic_order == [
        "basis_precision",
        "cut_off_energy",
        "elec_energy_tol",
        "kpoint_mp_grid",
        "geom_max_iter",
    ]


def test_diff_reports_no_changes_when_identical(qapp) -> None:
    dialog = _make_dialog()

    assert dialog._diff_table.rowCount() == 0
    assert dialog._diff_no_changes_label.text() == "No parameter changes."
    assert dialog._diff_no_changes_label.isHidden() is False
    assert dialog._diff_table.isHidden() is True


def test_diff_table_names_the_file_of_each_change(qapp) -> None:
    dialog = _make_dialog()

    headers = [
        dialog._diff_table.horizontalHeaderItem(column).text()
        for column in range(dialog._diff_table.columnCount())
    ]
    assert headers == ["File", "Keyword", "Change", "Value"]


def test_diff_shows_changed_row_after_editing_a_value(qapp) -> None:
    dialog = _make_dialog(parameter_values={"basis_precision": "FINE"})

    dialog._rows["basis_precision"].editor.set_value("PRECISE")

    assert dialog._diff_no_changes_label.isHidden() is True
    assert _diff_rows(dialog) == [(".param", "BASIS_PRECISION", "Changed", "FINE → PRECISE")]


def test_diff_shows_a_changed_cell_keyword(qapp) -> None:
    dialog = _make_dialog(parameter_values={"kpoint_mp_grid": (2, 2, 2)})

    dialog._rows["kpoint_mp_grid"].editor.set_value((3, 3, 3))

    assert _diff_rows(dialog) == [(".cell", "KPOINT_MP_GRID", "Changed", "2 2 2 → 3 3 3")]


def test_diff_lists_cell_changes_before_param_changes(qapp) -> None:
    dialog = _make_dialog(parameter_values={"basis_precision": "FINE", "kpoint_mp_grid": (2, 2, 2)})

    dialog._rows["basis_precision"].editor.set_value("PRECISE")
    dialog._rows["kpoint_mp_grid"].editor.set_value((3, 3, 3))

    assert [row[:2] for row in _diff_rows(dialog)] == [
        (".cell", "KPOINT_MP_GRID"),
        (".param", "BASIS_PRECISION"),
    ]


def test_diff_shows_added_row_after_adding_advanced_parameter(qapp) -> None:
    dialog = _make_dialog(parameter_values={"basis_precision": "FINE"}, added_advanced=())

    dialog.set_advanced_names(("num_dump_cycles",))

    assert (".param", "NUM_DUMP_CYCLES", "Added", "5") in _diff_rows(dialog)


def test_diff_shows_removed_row_after_removing_advanced_parameter(qapp) -> None:
    dialog = _make_dialog(
        parameter_values={"basis_precision": "FINE", "num_dump_cycles": 7},
        added_advanced=("num_dump_cycles",),
    )
    assert dialog._diff_no_changes_label.isHidden() is False

    dialog._remove_advanced("num_dump_cycles")

    assert (".param", "NUM_DUMP_CYCLES", "Removed", "7") in _diff_rows(dialog)

    # The thing WP6 actually consumes: the removed keyword must not reappear
    # in the accessor's result.
    values, advanced = dialog.edited_parameters()
    assert "num_dump_cycles" not in values
    assert "num_dump_cycles" not in advanced


def _render_with_species_pot(rows: int):
    def render(values: Mapping[str, object], advanced: tuple[str, ...]) -> JobInputTexts:
        inputs = _render_inputs(values, advanced)
        count = rows if values.get("basis_precision") == "PRECISE" else 1
        block = ["%BLOCK SPECIES_POT"]
        block += [f"Si{index} Si_00.usp" for index in range(count)]
        block.append("%ENDBLOCK SPECIES_POT")
        return JobInputTexts(cell="\n".join([inputs.cell, *block]), param=inputs.param)

    return render


def test_diff_shows_a_short_block_change_in_full(qapp) -> None:
    def render(values: Mapping[str, object], advanced: tuple[str, ...]) -> JobInputTexts:
        potential = "Si_C19.usp" if values.get("basis_precision") == "PRECISE" else "Si_00.usp"
        return JobInputTexts(
            cell=f"%BLOCK SPECIES_POT\nSi {potential}\n%ENDBLOCK SPECIES_POT", param=""
        )

    dialog = _make_dialog(parameter_values={"basis_precision": "FINE"}, render_inputs=render)

    dialog._rows["basis_precision"].editor.set_value("PRECISE")

    assert _diff_rows(dialog) == [
        (".cell", "SPECIES_POT", "Changed", "Si Si_00.usp → Si Si_C19.usp")
    ]


def test_diff_summarises_a_long_block_by_its_line_count(qapp) -> None:
    dialog = _make_dialog(
        parameter_values={"basis_precision": "FINE"}, render_inputs=_render_with_species_pot(3)
    )

    dialog._rows["basis_precision"].editor.set_value("PRECISE")

    assert (".cell", "SPECIES_POT", "Changed", "Si0 Si_00.usp → 3 lines") in _diff_rows(dialog)


def test_diff_says_contents_changed_when_a_block_keeps_its_length(qapp) -> None:
    def render(values: Mapping[str, object], advanced: tuple[str, ...]) -> JobInputTexts:
        mass = "2.0" if values.get("basis_precision") == "PRECISE" else "1.0"
        return JobInputTexts(
            cell=f"%BLOCK SPECIES_MASS\nAMU\nH:mu {mass}\n%ENDBLOCK SPECIES_MASS", param=""
        )

    dialog = _make_dialog(parameter_values={"basis_precision": "FINE"}, render_inputs=render)

    dialog._rows["basis_precision"].editor.set_value("PRECISE")

    assert _diff_rows(dialog) == [(".cell", "SPECIES_MASS", "Changed", "Contents changed")]


def test_diff_leaves_out_the_structure_blocks(qapp) -> None:
    calls = {"count": 0}

    def render_with_positions(
        values: Mapping[str, object], advanced: tuple[str, ...]
    ) -> JobInputTexts:
        calls["count"] += 1
        inputs = _render_inputs(values, advanced)
        positions = f"%BLOCK POSITIONS_FRAC\nSi 0.0 0.0 {calls['count']}\n%ENDBLOCK POSITIONS_FRAC"
        return JobInputTexts(cell=f"{inputs.cell}\n{positions}", param=inputs.param)

    dialog = _make_dialog(
        parameter_values={"basis_precision": "FINE"},
        render_inputs=render_with_positions,
    )

    # The positions changed between the caller-computed "old" render and the
    # dialog's own refresh (the counter advanced on each call), but the
    # keywords did not -- the table shows no rows.
    assert calls["count"] >= 2
    assert dialog._diff_table.rowCount() == 0
    assert dialog._diff_no_changes_label.isHidden() is False


def test_diff_says_when_the_inputs_could_not_be_compared(qapp) -> None:
    dialog = _make_dialog(old_inputs=None)

    dialog._rows["basis_precision"].editor.set_value("PRECISE")

    assert dialog._diff_table.isHidden() is True
    assert dialog._diff_no_changes_label.text() == "Could not compare the input files."


def test_diff_says_when_the_new_inputs_could_not_be_rendered(qapp) -> None:
    dialog = _make_dialog(
        render_inputs=lambda values, advanced: None, old_inputs=JobInputTexts("", "")
    )

    assert dialog._diff_table.isHidden() is True
    assert dialog._diff_no_changes_label.text() == "Could not compare the input files."


def test_checkpoint_status_line_found(qapp) -> None:
    dialog = _make_dialog(checkpoint_probe=CheckpointProbeResult.found())

    assert "resume" in dialog._checkpoint_label.text().lower()


def test_checkpoint_status_line_absent(qapp) -> None:
    dialog = _make_dialog(checkpoint_probe=CheckpointProbeResult.absent())

    text = dialog._checkpoint_label.text().lower()
    assert "no checkpoint found" in text
    assert "initial positions" in text


def test_checkpoint_status_line_unavailable(qapp) -> None:
    dialog = _make_dialog(
        checkpoint_probe=CheckpointProbeResult.unavailable("SSH connection refused")
    )

    text = dialog._checkpoint_label.text()
    assert text == "Could not verify checkpoint: SSH connection refused."


def test_submit_summary_reports_target_and_directory(qapp) -> None:
    dialog = _make_dialog(execution_target="scarf", reused_directory="/scratch/user/relax_unit")

    text = dialog._summary_label.text()
    assert "scarf" in text
    assert "/scratch/user/relax_unit" in text


def test_edited_parameters_reflects_current_editor_state(qapp) -> None:
    dialog = _make_dialog(parameter_values={"basis_precision": "FINE"})

    dialog._rows["basis_precision"].editor.set_value("PRECISE")
    values, advanced = dialog.edited_parameters()

    assert values["basis_precision"] == "PRECISE"
    assert "task" not in values
    assert advanced == ()


def test_edited_parameters_includes_added_advanced_names(qapp) -> None:
    dialog = _make_dialog(parameter_values={"basis_precision": "FINE"})

    dialog.set_advanced_names(("num_dump_cycles",))
    values, advanced = dialog.edited_parameters()

    assert "num_dump_cycles" in values
    assert advanced == ("num_dump_cycles",)


def test_cancel_does_not_prevent_reading_the_accessor_and_input_is_not_mutated(qapp) -> None:
    original_values = {"basis_precision": "FINE"}
    dialog = _make_dialog(parameter_values=original_values)

    dialog._rows["basis_precision"].editor.set_value("PRECISE")
    dialog.reject()

    # The dialog is pure: it never mutates the caller's input dict, and the
    # accessor keeps reporting current widget state regardless of how the
    # dialog was closed -- callers are expected to gate on the result code
    # (dialog.result() == QDialog.DialogCode.Accepted) before applying it.
    assert original_values == {"basis_precision": "FINE"}
    values, _advanced = dialog.edited_parameters()
    assert values["basis_precision"] == "PRECISE"


def test_ok_button_is_labelled_continue(qapp) -> None:
    from PySide6.QtWidgets import QDialogButtonBox

    dialog = _make_dialog()

    root = dialog.layout()
    button_box = root.itemAt(root.count() - 1).widget()
    assert isinstance(button_box, QDialogButtonBox)
    ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
    assert ok_button is not None
    assert ok_button.text() == "Continue"


def test_builds_each_editor_once(qapp, monkeypatch) -> None:
    from mufinder.app import param_dialog as param_dialog_module
    from mufinder.app.param_widgets import create_param_editor

    editor_parents: list[object] = []

    def counting_create_param_editor(spec, parent=None, **kwargs):
        editor_parents.append(parent)
        return create_param_editor(spec, parent, **kwargs)

    monkeypatch.setattr(param_dialog_module, "create_param_editor", counting_create_param_editor)

    dialog = _make_dialog(added_advanced=("num_dump_cycles",))

    table_editors = [parent for parent in editor_parents if parent is dialog._table_widget]
    row_editors = [row.editor for row in dialog._rows.values() if row.editor is not None]
    assert row_editors
    assert len(table_editors) == len(row_editors)
    # The seed still reaches the single build.
    assert "num_dump_cycles" in dialog.advanced_names()
    assert dialog._diff_table.rowCount() == 0


def test_tab_leaves_the_changes_table(qapp) -> None:
    dialog = _make_dialog()
    try:
        assert_tab_leaves_table(dialog._diff_table)
    finally:
        dialog.close()


def test_an_unedited_dialog_writes_the_kept_inputs(qapp) -> None:
    kept = JobInputTexts(cell="", param="BASIS_PRECISION : FINE\nCUT_OFF_ENERGY : 340.0\n")
    dialog = _make_dialog(old_inputs=kept, kept_inputs=kept)

    assert dialog.edited_names() == frozenset()
    assert dialog.continuation_inputs() == kept
    assert dialog._diff_table.rowCount() == 0


def test_an_edit_takes_only_the_edited_keyword_from_the_rendering(qapp) -> None:
    kept = JobInputTexts(cell="", param="GEOM_MAX_ITER : 120 ! kept as written\n")
    dialog = _make_dialog(
        parameter_values={"geom_max_iter": 120}, old_inputs=kept, kept_inputs=kept
    )

    dialog._rows["geom_max_iter"].editor.set_value(250)

    assert dialog.edited_names() == {"geom_max_iter"}
    assert dialog.continuation_inputs() == JobInputTexts(cell="", param="GEOM_MAX_ITER : 250\n")
    assert _diff_rows(dialog) == [(".param", "GEOM_MAX_ITER", "Changed", "120 → 250")]


def test_kept_names_open_on_the_values_the_run_used(qapp) -> None:
    kept = JobInputTexts(cell="KPOINT_MP_GRID : 4 4 6\n", param="CUT_OFF_ENERGY : 600 eV\n")
    dialog = _make_dialog(
        parameter_values={"cut_off_energy": 900.0, "kpoint_mp_grid": (7, 7, 9)},
        old_inputs=kept,
        kept_inputs=kept,
        kept_names={"cut_off_energy", "basis_precision", "kpoint_mp_grid", "kpoint_mp_spacing"},
    )

    assert dialog._rows["cut_off_energy"].editor.value().value == 600.0
    assert dialog._rows["kpoint_mp_grid"].editor.value() == (4, 4, 6)
    assert dialog.continuation_inputs() == kept


def test_resetting_an_editor_returns_it_to_the_value_it_opened_with(qapp) -> None:
    dialog = _make_dialog(parameter_values={"geom_max_iter": 120})

    dialog._rows["geom_max_iter"].editor.set_value(250)
    dialog._reset_parameter("geom_max_iter")

    assert dialog._rows["geom_max_iter"].editor.value() == 120
    assert dialog.edited_names() == frozenset()


def test_edited_parameters_leaves_the_row_as_it_was_when_nothing_is_edited(qapp) -> None:
    dialog = _make_dialog(parameter_values={"geom_max_iter": 120})

    values, advanced = dialog.edited_parameters()

    # The editors the dialog fills in (basis precision, k-point grid) are not
    # stored on the row unless they are edited.
    assert values == {"geom_max_iter": 120}
    assert advanced == ()


def test_a_kept_name_the_run_left_to_castep_does_not_open_on_the_project_default(qapp) -> None:
    kept = JobInputTexts(cell="", param="GEOM_MAX_ITER : 120\n")
    dialog = _make_dialog(
        parameter_values={"geom_max_iter": 120, "cut_off_energy": 900.0},
        old_inputs=kept,
        kept_inputs=kept,
        kept_names={"cut_off_energy", "basis_precision"},
    )

    assert dialog._rows["cut_off_energy"].editor.value().value != 900.0
    assert dialog.continuation_inputs() == kept
    # The row keeps what it stored.
    assert dialog.edited_parameters()[0]["cut_off_energy"] == 900.0
