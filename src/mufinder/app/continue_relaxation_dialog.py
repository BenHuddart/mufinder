"""Dialog for continuing a terminal relaxation job with an in-place parameter edit.

``ContinueRelaxationDialog`` is a :class:`~mufinder.app.param_dialog.ParameterDialog`
subclass (the same base every row parameter dialog uses -- see
``mufinder.app.workflow_dialogs.run_row_parameter_dialog``) that adds a live
keyword-level comparison of the job's ``.cell`` and ``.param`` as they are on
disk against what continuing will write, an informational checkpoint-probe
line, and a submit summary. It performs no I/O and knows nothing about
``Job``/``WorkflowDefinition``/SSH: everything it needs (the row's current
parameter values, a probe result, the current input texts and a rendering
callable) is injected by the caller, which owns every side effect. The
comparison itself is :func:`mufinder.castep_io.input_diff.diff_job_inputs`.

Continuing keeps the run's inputs: an unedited dialog writes ``kept_inputs``
(the row's inputs with the project defaults it inherits kept as the run used
them), and an edit takes only the edited keywords from a fresh rendering
(:func:`mufinder.castep_io.input_diff.overlay_job_inputs`).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.table_keyboard import TabStopTable
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.theme import style_hint_text, style_warning_text, track_item_view_corners
from mufinder.app.workflow_parameters import kpoint_exclusive_hint, reorder_workflow_parameter_specs
from mufinder.castep_io.gui_metadata import normalize_gui_value
from mufinder.castep_io.input_diff import (
    InputKeywordChange,
    JobInputTexts,
    diff_job_inputs,
    overlay_job_inputs,
    split_input_sections,
)
from mufinder.castep_io.param_parser import parse_param_value
from mufinder.castep_io.schema import KeywordSpec
from mufinder.domain.text import plural
from mufinder.persistence.codec import to_jsonable
from mufinder.services.cluster_runner import CheckpointProbeResult, CheckpointProbeStatus

#: Renders the ``.cell`` and ``.param`` one row would write from its structured
#: parameter values and the set of advanced keyword names currently added to the
#: row -- the same shape ``ParameterDialog.collect_values()``/``advanced_names()``
#: produce -- or ``None`` when the row cannot be rendered. The dialog calls it
#: for the live ("new") side on every edit, with its own fully resolved value set.
RenderJobInputs = Callable[[Mapping[str, object], tuple[str, ...]], JobInputTexts | None]

_CHECKPOINT_MESSAGES: dict[CheckpointProbeStatus, str] = {
    CheckpointProbeStatus.FOUND: "Checkpoint found — the relaxation will resume from it.",
    CheckpointProbeStatus.ABSENT: "No checkpoint found — the run will start from the initial positions.",
}

_NO_CHANGES_TEXT = "No parameter changes."
_NOT_COMPARED_TEXT = "Could not compare the input files."

#: A block of one line up to this long is shown in full; longer or multi-line
#: blocks are summarised by their line count.
_INLINE_BLOCK_LENGTH = 40

_FILE_LABELS = {"cell": ".cell", "param": ".param"}
_MISSING = object()


def _run_values(
    inputs: JobInputTexts,
    specs: Sequence[KeywordSpec],
    values: Mapping[str, object],
    names: Collection[str],
) -> dict[str, object]:
    """Return ``values`` with ``names`` shown as ``inputs`` hold them.

    ``names`` are the project defaults the row inherits, which continuing keeps
    from the run. Each one ``inputs`` hold replaces the row's current value (a
    default adopted or changed since the run). One they do not hold, which the
    run left to CASTEP or to an either-or partner, is dropped, so its editor
    opens on its starting value rather than on a project default the run never
    used. A value that does not parse is left as it is.
    """

    sections = {
        "cell": split_input_sections(inputs.cell, "cell").scalars,
        "param": split_input_sections(inputs.param, "param").scalars,
    }
    specs_by_name = {spec.name: spec for spec in specs}
    result = dict(values)
    held: set[str] = set()
    for name in names:
        spec = specs_by_name.get(name)
        if spec is None or spec.kind == "block":
            continue
        text = sections.get(spec.file, {}).get(name)
        if text is None:
            continue
        try:
            parsed = parse_param_value(text, spec)
            if parsed is not None:
                result[name] = normalize_gui_value(spec, parsed)
                held.add(name)
        except (TypeError, ValueError):
            continue
    for name in set(names) - held:
        result.pop(name, None)
    return result


def _checkpoint_status_text(probe: CheckpointProbeResult) -> str:
    if probe.status == CheckpointProbeStatus.UNAVAILABLE:
        return f"Could not verify checkpoint: {probe.reason}."
    return _CHECKPOINT_MESSAGES[probe.status]


def _block_summary(lines: tuple[str, ...]) -> str:
    if len(lines) == 1 and len(lines[0]) <= _INLINE_BLOCK_LENGTH:
        return lines[0]
    return plural(len(lines), "line")


def _value_text(change: InputKeywordChange) -> str:
    """Return the table's ``Value`` cell for one change."""

    if change.is_block:
        old = _block_summary(change.old_lines)
        new = _block_summary(change.new_lines)
    else:
        old = change.old_lines[0] if change.old_lines else ""
        new = change.new_lines[0] if change.new_lines else ""
    if change.change == "Added":
        return new
    if change.change == "Removed":
        return old
    if old == new:
        return "Contents changed"
    return f"{old} → {new}"


class ContinueRelaxationDialog(ParameterDialog):
    """Edit a terminal relaxation row's parameters and preview what continuing changes.

    In addition to the inherited parameter table (search, advanced-parameter
    add/remove, exclusive kpoint grid/spacing choice, locked task, etc.), this
    dialog adds:

    * a fixed informational line stating the ``.cell`` is regenerated from the
      definition's structure (positions are not edited here);
    * a checkpoint-probe status line;
    * a submit summary (execution target + reused directory);
    * a live ``Parameter changes`` table listing every keyword and block, in
      the ``.cell`` and the ``.param``, that continuing would add, change or
      remove, refreshed on every parameter edit. The ``.cell`` structure
      blocks are not listed.

    ``old_inputs`` is the job's current ``.cell`` and ``.param`` (the files on
    disk, or ``None`` when they cannot be read); ``render_inputs`` renders the
    row's inputs for the dialog's values, exactly as the accept path renders
    them. ``kept_inputs`` is what continuing writes when nothing is edited: the
    row's inputs with ``kept_names``, the project defaults the row inherits,
    kept as the run used them.

    :meth:`continuation_inputs` is ``kept_inputs`` with only the
    :meth:`edited_names` taken from ``render_inputs``, so on a row unchanged
    since its run an unedited dialog reads "No parameter changes" and writes
    the files the run used. The ``kept_names`` editors open on the run's values
    (see :func:`_run_values`), and resetting an editor returns it to the value
    it opened with.

    The dialog performs no I/O. Use :meth:`edited_parameters` to read back the
    result; it is a pure accessor, safe to call regardless of how the dialog
    was closed -- gate on ``dialog.exec() == QDialog.DialogCode.Accepted``
    first, exactly as every other ``ParameterDialog``-based workflow surface
    in this app does.
    """

    def __init__(
        self,
        *,
        specs: list[KeywordSpec],
        workflow_stage: str,
        parameter_values: Mapping[str, object],
        added_advanced: Iterable[str] = (),
        locked_values: dict[str, str] | None = None,
        species_options: Sequence[str] = (),
        old_inputs: JobInputTexts | None,
        render_inputs: RenderJobInputs,
        kept_inputs: JobInputTexts | None = None,
        kept_names: Iterable[str] = (),
        checkpoint_probe: CheckpointProbeResult,
        execution_target: str,
        reused_directory: str,
        parent: QWidget | None = None,
    ) -> None:
        # Task is always fixed by locked_values for these workflows; drop it
        # from the seeded values so it is excluded from the live render the
        # same way collect_values() excludes it in _render_current_inputs.
        row_values: dict[str, object] = {
            str(name): value for name, value in parameter_values.items() if str(name) != "task"
        }
        kept_inputs = kept_inputs if kept_inputs is not None else old_inputs
        initial_values = (
            _run_values(kept_inputs, specs, row_values, frozenset(map(str, kept_names)))
            if kept_inputs is not None
            else dict(row_values)
        )
        initial_advanced = tuple(str(name) for name in added_advanced)

        super().__init__(
            specs=reorder_workflow_parameter_specs(list(specs)),
            workflow_stage=workflow_stage,
            locked_values=locked_values,
            intro_text=(
                "The .cell file will be regenerated from the definition's structure "
                "on acceptance; atomic positions are not changed by this dialog."
            ),
            species_options=species_options,
            # Seeded through the constructor, as every row parameter dialog is
            # (see workflow_dialogs.run_row_parameter_dialog), so the rows are
            # built once and show exactly what the row was continuing from.
            values=dict(initial_values),
            preferred_exclusive=kpoint_exclusive_hint([initial_values]),
            advanced_names=initial_advanced,
            parent=parent,
        )
        self.setWindowTitle("Continue Relaxation")

        self._render_inputs = render_inputs
        self._old_inputs = old_inputs
        self._kept_inputs = kept_inputs
        self._row_values = row_values
        # What the dialog opened with: an edit is a difference from these, and
        # resetting an editor returns it to its opening value, the run's.
        self._opening_values = self._current_values()
        self._opening_advanced = frozenset(self.advanced_names())
        self._opening_editor_values = {
            name: row.editor.value() for name, row in self._rows.items() if row.editor is not None
        }
        self._row_default_values.update(self._opening_editor_values)
        self._opening_render = render_inputs(dict(self._opening_values), self.advanced_names())

        self._checkpoint_label = QLabel(_checkpoint_status_text(checkpoint_probe), self)
        self._checkpoint_label.setWordWrap(True)
        if checkpoint_probe.status == CheckpointProbeStatus.UNAVAILABLE:
            style_warning_text(self._checkpoint_label)
        else:
            style_hint_text(self._checkpoint_label)

        self._summary_label = QLabel(
            f"Will resubmit to {execution_target}, reusing the existing job directory: {reused_directory}",
            self,
        )
        self._summary_label.setWordWrap(True)
        style_hint_text(self._summary_label)

        self._diff_group = QGroupBox("Parameter changes", self)
        diff_layout = QVBoxLayout(self._diff_group)
        self._diff_table = TabStopTable(0, 4, self._diff_group)
        self._diff_table.setAccessibleName("Parameter changes")
        track_item_view_corners(self._diff_table)
        self._diff_table.setHorizontalHeaderLabels(["File", "Keyword", "Change", "Value"])
        self._diff_table.verticalHeader().setVisible(False)
        self._diff_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._diff_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._diff_table.horizontalHeader().setStretchLastSection(True)
        self._diff_no_changes_label = QLabel(_NO_CHANGES_TEXT, self._diff_group)
        style_hint_text(self._diff_no_changes_label)
        diff_layout.addWidget(self._diff_table)
        diff_layout.addWidget(self._diff_no_changes_label)

        root = self.layout()
        assert root is not None
        # The button box is always the last item ParameterDialog.__init__
        # adds; grab it before inserting anything so "Continue" replaces "OK"
        # regardless of how many rows we insert above it.
        button_box = root.itemAt(root.count() - 1).widget()
        if isinstance(button_box, QDialogButtonBox):
            ok_button = button_box.button(QDialogButtonBox.StandardButton.Ok)
            if ok_button is not None:
                ok_button.setText("Continue")

        insert_index = root.count() - 1
        root.insertWidget(insert_index, self._checkpoint_label)
        root.insertWidget(insert_index + 1, self._summary_label)
        root.insertWidget(insert_index + 2, self._diff_group)
        self._order_tab_stops()

        self._refresh_diff()

    # -- live diff panel --------------------------------------------------
    #
    # ParameterDialog mutates parameter/advanced state from several private
    # entry points. All of them funnel through one of the three methods
    # overridden below, so overriding those three is enough to keep the diff
    # panel live without duplicating ParameterDialog's internals:
    #   * direct edits and _reset_parameter -> _on_editor_changed
    #   * exclusive-group radio toggles      -> _on_exclusive_toggled
    #   * task change / add-advanced / remove-advanced / set_values /
    #     set_advanced_names                 -> _rebuild_rows

    def _rebuild_rows(self, *, preserve_current: bool = True) -> None:
        super()._rebuild_rows(preserve_current=preserve_current)
        opening = getattr(self, "_opening_editor_values", None)
        if opening is not None:
            self._row_default_values.update(
                {name: value for name, value in opening.items() if name in self._rows}
            )
        self._maybe_refresh_diff()

    def _on_editor_changed(self) -> None:
        super()._on_editor_changed()
        self._maybe_refresh_diff()

    def _on_exclusive_toggled(self, group_key: tuple[str, ...], name: str, checked: bool) -> None:
        super()._on_exclusive_toggled(group_key, name, checked)
        self._maybe_refresh_diff()

    def _maybe_refresh_diff(self) -> None:
        # The overrides above fire during ParameterDialog.__init__, before the
        # diff widgets exist yet -- skip until they've been built.
        if getattr(self, "_diff_table", None) is None:
            return
        self._refresh_diff()

    def _current_values(self) -> dict[str, object]:
        values = dict(self.collect_values())
        values.pop("task", None)
        return values

    def edited_names(self) -> frozenset[str]:
        """Return the parameters edited in the dialog: changed, chosen, added or removed.

        A value set back to the one the dialog opened with, by hand or with its
        reset button, is not an edit. Choosing the other member of an either-or
        pair edits both.
        """

        current = self._current_values()
        opening = self._opening_values
        names = {
            name
            for name in set(current) | set(opening)
            if current.get(name, _MISSING) != opening.get(name, _MISSING)
        }
        names |= set(self.advanced_names()) ^ self._opening_advanced
        return frozenset(names)

    def continuation_inputs(self) -> JobInputTexts | None:
        """Return the ``.cell`` and ``.param`` continuing writes, or ``None`` if not rendered.

        ``kept_inputs`` with the dialog's edits taken from the row's fresh
        rendering: each edited parameter, and any other keyword or block the
        edits change in that rendering.
        """

        if self._kept_inputs is None or self._opening_render is None:
            return None
        edited = self.edited_names()
        if not edited:
            return self._kept_inputs
        rendered = self._render_inputs(self._current_values(), self.advanced_names())
        if rendered is None:
            return None
        changed = {change.keyword for change in diff_job_inputs(self._opening_render, rendered)}
        return overlay_job_inputs(self._kept_inputs, rendered, edited | changed)

    def _refresh_diff(self) -> None:
        old_inputs = self._old_inputs
        new_inputs = self.continuation_inputs()
        compared = old_inputs is not None and new_inputs is not None
        changes = diff_job_inputs(old_inputs, new_inputs) if compared else ()

        self._diff_table.setRowCount(len(changes))
        for row_index, change in enumerate(changes):
            cells = (_FILE_LABELS[change.file], change.keyword, change.change, _value_text(change))
            for column, text in enumerate(cells):
                self._diff_table.setItem(row_index, column, QTableWidgetItem(text))
        self._diff_table.resizeColumnsToContents()

        has_changes = bool(changes)
        self._diff_table.setVisible(has_changes)
        self._diff_no_changes_label.setText(_NO_CHANGES_TEXT if compared else _NOT_COMPARED_TEXT)
        self._diff_no_changes_label.setVisible(not has_changes)

    # -- result -------------------------------------------------------------

    def edited_parameters(self) -> tuple[dict[str, object], tuple[str, ...]]:
        """Return what the row stores: ``(parameter_values, added_advanced)``.

        The row's values as the dialog received them, with only the
        :meth:`edited_names` changed, so an unedited dialog leaves the row as it
        was. Pure accessor with no side effects, safe to call regardless of how
        the dialog was closed. Callers gate on ``dialog.exec() ==
        QDialog.DialogCode.Accepted`` (or ``dialog.result()``) before applying
        the result -- this dialog never mutates caller state itself.

        ``task`` is never included in the returned ``parameter_values``: both
        relaxation workflows lock it via ``locked_values``, so the caller
        already knows it and re-supplies it when re-materializing the row.
        """

        current = self._current_values()
        values = dict(self._row_values)
        for name in self.edited_names():
            if name in current:
                values[name] = current[name]
            else:
                values.pop(name, None)
        normalized = to_jsonable(values)
        parameter_values = normalized if isinstance(normalized, dict) else {}
        return parameter_values, self.advanced_names()
