"""Tests for the Continue Relaxation dialog (parameter editor + live diff)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from mufinder.app.continue_relaxation_dialog import ContinueRelaxationDialog, RenderParamText
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.workflow_parameters import kpoint_exclusive_hint, reorder_workflow_parameter_specs
from mufinder.castep_io.schema import KeywordSpec
from mufinder.services.cluster_runner import CheckpointProbeResult


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


def _render_param_text(values: Mapping[str, object], advanced: tuple[str, ...]) -> str:
    """Deterministic fake renderer: one ``KEY : value`` line per entry."""

    lines = [
        f"{name.upper()} : {value}"
        for name, value in sorted(values.items(), key=lambda item: item[0])
    ]
    return "\n".join(lines) + "\n"


def _effective_param_text(
    render: RenderParamText,
    *,
    parameter_values: Mapping[str, object],
    added_advanced: Iterable[str] = (),
) -> str:
    """Render the row's *fully resolved* text (defaults + overrides).

    Mirrors exactly what ``ContinueRelaxationDialog`` does internally for its
    live "new" render (same reorder + seed sequence, same
    ``collect_values()``/``advanced_names()`` read-back), so a caller-computed
    ``old_param_text`` build this way is guaranteed to agree with the dialog's
    own unedited state -- the contract documented on
    ``ContinueRelaxationDialog``.
    """

    reference = ParameterDialog(
        specs=reorder_workflow_parameter_specs(_continue_specs()),
        workflow_stage="pristine_relaxation",
        locked_values=_LOCKED_VALUES,
    )
    try:
        reference.set_values(
            dict(parameter_values),
            preferred_exclusive=kpoint_exclusive_hint([parameter_values]),
        )
        reference.set_advanced_names(tuple(added_advanced))
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
    render = overrides.pop("render_param_text", _render_param_text)

    kwargs: dict[str, object] = {
        "specs": _continue_specs(),
        "workflow_stage": "pristine_relaxation",
        "parameter_values": parameter_values,
        "added_advanced": added_advanced,
        "locked_values": _LOCKED_VALUES,
        "old_param_text": _effective_param_text(
            render,  # type: ignore[arg-type]
            parameter_values=parameter_values,  # type: ignore[arg-type]
            added_advanced=added_advanced,  # type: ignore[arg-type]
        ),
        "render_param_text": render,
        "checkpoint_probe": CheckpointProbeResult.found(),
        "execution_target": "local",
        "reused_directory": "/work/jobs/relax_unit",
    }
    kwargs.update(overrides)
    return ContinueRelaxationDialog(**kwargs)  # type: ignore[arg-type]


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
    assert dialog._diff_no_changes_label.isHidden() is False
    assert dialog._diff_table.isHidden() is True


def test_diff_shows_changed_row_after_editing_a_value(qapp) -> None:
    dialog = _make_dialog(parameter_values={"basis_precision": "FINE"})

    dialog._rows["basis_precision"].editor.set_value("PRECISE")

    assert dialog._diff_no_changes_label.isHidden() is True
    rows = {
        dialog._diff_table.item(row, 0).text(): (
            dialog._diff_table.item(row, 1).text(),
            dialog._diff_table.item(row, 2).text(),
        )
        for row in range(dialog._diff_table.rowCount())
    }
    assert rows["BASIS_PRECISION"] == ("Changed", "FINE → PRECISE")


def test_diff_shows_added_row_after_adding_advanced_parameter(qapp) -> None:
    dialog = _make_dialog(parameter_values={"basis_precision": "FINE"}, added_advanced=())

    dialog.set_advanced_names(("num_dump_cycles",))

    rows = {
        dialog._diff_table.item(row, 0).text(): dialog._diff_table.item(row, 1).text()
        for row in range(dialog._diff_table.rowCount())
    }
    assert rows["NUM_DUMP_CYCLES"] == "Added"
    value_by_keyword = {
        dialog._diff_table.item(row, 0).text(): dialog._diff_table.item(row, 2).text()
        for row in range(dialog._diff_table.rowCount())
    }
    assert value_by_keyword["NUM_DUMP_CYCLES"] == "5"


def test_diff_shows_removed_row_after_removing_advanced_parameter(qapp) -> None:
    dialog = _make_dialog(
        parameter_values={"basis_precision": "FINE", "num_dump_cycles": 7},
        added_advanced=("num_dump_cycles",),
    )
    assert dialog._diff_no_changes_label.isHidden() is False

    dialog._remove_advanced("num_dump_cycles")

    rows = {
        dialog._diff_table.item(row, 0).text(): dialog._diff_table.item(row, 1).text()
        for row in range(dialog._diff_table.rowCount())
    }
    assert rows["NUM_DUMP_CYCLES"] == "Removed"

    # The thing WP6 actually consumes: the removed keyword must not reappear
    # in the accessor's result.
    values, advanced = dialog.edited_parameters()
    assert "num_dump_cycles" not in values
    assert "num_dump_cycles" not in advanced


def test_diff_excludes_block_content(qapp) -> None:
    calls = {"count": 0}

    def render_with_block(values: Mapping[str, object], advanced: tuple[str, ...]) -> str:
        calls["count"] += 1
        lines = [f"{name.upper()} : {value}" for name, value in sorted(values.items())]
        lines.append("%BLOCK POSITIONS_FRAC")
        lines.append(f"Si 0.0 0.0 {calls['count']}")
        lines.append("%ENDBLOCK POSITIONS_FRAC")
        return "\n".join(lines) + "\n"

    dialog = _make_dialog(
        parameter_values={"basis_precision": "FINE"},
        render_param_text=render_with_block,
    )

    # The renderer's block content changed between the caller-computed "old"
    # render (used to build old_param_text) and the dialog's own initial
    # diff refresh (the counter advanced on each call), but the keyword
    # values did not -- the diff must show no rows.
    assert calls["count"] >= 2
    assert dialog._diff_table.rowCount() == 0
    assert dialog._diff_no_changes_label.isHidden() is False


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
