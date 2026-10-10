"""Dialog for continuing a terminal relaxation job with an in-place parameter edit.

``ContinueRelaxationDialog`` is a :class:`~mufinder.app.param_dialog.ParameterDialog`
subclass (the same base every row parameter dialog uses -- see
``mufinder.app.workflow_dialogs.run_row_parameter_dialog``) that adds a live
keyword-level diff of the row's old vs. new rendered ``.param`` text, an
informational checkpoint-probe line, and a submit summary. It performs no I/O
and knows nothing about ``Job``/``WorkflowDefinition``/SSH: everything it needs
(the row's current parameter values, a probe result, and a text-rendering
callable) is injected by the caller, so the controller wiring that constructs
it (WP6) owns every side effect.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.theme import style_hint_text, style_warning_text
from mufinder.app.workflow_parameters import kpoint_exclusive_hint, reorder_workflow_parameter_specs
from mufinder.castep_io.schema import KeywordSpec
from mufinder.persistence.codec import to_jsonable
from mufinder.services.cluster_runner import CheckpointProbeResult, CheckpointProbeStatus

#: Renders one row's ``.param`` text from its structured parameter values and
#: the set of advanced keyword names currently added to the row -- the same
#: shape ``ParameterDialog.collect_values()``/``advanced_names()`` produce.
#: WP6 binds this to the workflow's JobSpec-expansion + renderer path. The
#: dialog calls it only for the *live* ("new") side, on every edit, with its
#: own fully-resolved value set: every currently visible row's value, whether
#: the user edited it or it is still sitting at the GUI catalog's
#: ``default_value``. It does NOT call this for the "old" baseline -- see
#: ``old_param_text`` on :class:`ContinueRelaxationDialog` for why. The
#: dialog does not parse CASTEP files or touch the filesystem/network itself
#: -- it only diffs the strings it is given/computes.
RenderParamText = Callable[[Mapping[str, object], tuple[str, ...]], str]

_CHECKPOINT_MESSAGES: dict[CheckpointProbeStatus, str] = {
    CheckpointProbeStatus.FOUND: "Checkpoint found — the relaxation will resume from it.",
    CheckpointProbeStatus.ABSENT: "No checkpoint found — the run will start from the initial positions.",
}


def _checkpoint_status_text(probe: CheckpointProbeResult) -> str:
    if probe.status == CheckpointProbeStatus.UNAVAILABLE:
        return f"Could not verify checkpoint: {probe.reason}."
    return _CHECKPOINT_MESSAGES[probe.status]


def _parse_param_keywords(text: str) -> dict[str, str]:
    """Parse ``KEY : value`` lines out of rendered CASTEP ``.param`` text.

    This is presentation-only parsing of text the app itself just rendered
    (via the caller-supplied :data:`RenderParamText` callable) -- it is not a
    general CASTEP file reader (there is deliberately no ``.param`` reader in
    ``castep_io``; see the diff-by-re-render note in the warm-start plan).
    ``%BLOCK``/``%ENDBLOCK`` sections are skipped entirely, so block contents
    (e.g. positions) never enter the keyword-level diff.
    """

    keywords: dict[str, str] = {}
    in_block = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("%BLOCK"):
            in_block = True
            continue
        if upper.startswith("%ENDBLOCK"):
            in_block = False
            continue
        if in_block or ":" not in line:
            continue
        name, _, value = line.partition(":")
        keywords[name.strip().upper()] = value.strip()
    return keywords


@dataclass(frozen=True, slots=True)
class _KeywordDiffRow:
    keyword: str
    change: str  # "Added" | "Changed" | "Removed"
    value_text: str


def _diff_param_text(old_text: str, new_text: str) -> list[_KeywordDiffRow]:
    """Return keyword-level Added/Changed/Removed rows between two renders."""

    old_keywords = _parse_param_keywords(old_text)
    new_keywords = _parse_param_keywords(new_text)

    rows: list[_KeywordDiffRow] = []
    for name in sorted(set(old_keywords) | set(new_keywords)):
        old_value = old_keywords.get(name)
        new_value = new_keywords.get(name)
        if old_value == new_value:
            continue
        if old_value is None:
            rows.append(_KeywordDiffRow(name, "Added", new_value or ""))
        elif new_value is None:
            rows.append(_KeywordDiffRow(name, "Removed", old_value))
        else:
            rows.append(_KeywordDiffRow(name, "Changed", f"{old_value} → {new_value}"))
    return rows


class ContinueRelaxationDialog(ParameterDialog):
    """Edit a terminal relaxation row's parameters and preview the warm-start diff.

    In addition to the inherited parameter table (search, advanced-parameter
    add/remove, exclusive kpoint grid/spacing choice, locked task, etc.), this
    dialog adds:

    * a fixed informational line stating the ``.cell`` is regenerated from the
      definition's structure (positions are not edited here);
    * a checkpoint-probe status line;
    * a submit summary (execution target + reused directory);
    * a live keyword-level diff table of the old vs. new rendered ``.param``
      text, refreshed on every parameter edit.

    The dialog performs no I/O: the checkpoint probe result, the render
    callable, and the execution target/directory strings are all supplied by
    the caller. Use :meth:`edited_parameters` to read back the result; it is a
    pure accessor, safe to call regardless of how the dialog was closed --
    gate on ``dialog.exec() == QDialog.DialogCode.Accepted`` first, exactly as
    every other ``ParameterDialog``-based workflow surface in this app does.

    **Why ``old_param_text`` is a plain string, not something this dialog
    renders itself:** a row's stored ``parameter_values`` is normally a
    *partial* set of overrides on top of workflow-template defaults (see
    ``_strip_implicit_row_defaults`` in ``relaxation_widget.py``), so the text
    actually on disk depends on how the real JobSpec-expansion path resolves
    keywords the row never mentions. This dialog has no visibility into that
    resolution -- only WP6's controller wiring does. Passing a pre-rendered
    ``old_param_text`` keeps the dialog honest: it diffs two strings, one
    fixed at construction, one re-rendered live via ``render_param_text`` on
    every edit. For the *unedited* dialog to correctly show "No parameter
    changes", ``old_param_text`` and the dialog's live render must agree
    keyword-for-keyword; the safest way to guarantee that is for the caller
    to render ``old_param_text`` with this exact same ``render_param_text``
    callable, applied to the row's pre-edit values, rather than through a
    separate code path that might resolve implicit defaults differently.
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
        old_param_text: str,
        render_param_text: RenderParamText,
        checkpoint_probe: CheckpointProbeResult,
        execution_target: str,
        reused_directory: str,
        parent: QWidget | None = None,
    ) -> None:
        # Task is always fixed by locked_values for these workflows; drop it
        # from the seeded values so it is excluded from the live render the
        # same way collect_values() excludes it in _render_current_param_text.
        initial_values: dict[str, object] = {
            str(name): value for name, value in parameter_values.items() if str(name) != "task"
        }
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
            parent=parent,
        )
        self.setWindowTitle("Continue Relaxation")

        self._render_param_text = render_param_text
        self._old_param_text = old_param_text
        self._new_param_text = old_param_text

        # Mirrors the call sequence of every row parameter dialog (see
        # workflow_dialogs.run_row_parameter_dialog): seed values first, then the advanced
        # set, so the shown editors reflect exactly what the row was
        # continuing from before any edits are made.
        self.set_values(dict(initial_values), preferred_exclusive=kpoint_exclusive_hint([initial_values]))
        self.set_advanced_names(initial_advanced)

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
        self._diff_table = QTableWidget(0, 3, self._diff_group)
        self._diff_table.setHorizontalHeaderLabels(["Keyword", "Change", "Value"])
        self._diff_table.verticalHeader().setVisible(False)
        self._diff_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._diff_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._diff_table.horizontalHeader().setStretchLastSection(True)
        self._diff_no_changes_label = QLabel("No parameter changes.", self._diff_group)
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
        self._maybe_refresh_diff()

    def _on_editor_changed(self) -> None:
        super()._on_editor_changed()
        self._maybe_refresh_diff()

    def _on_exclusive_toggled(self, group_key: tuple[str, ...], name: str, checked: bool) -> None:
        super()._on_exclusive_toggled(group_key, name, checked)
        self._maybe_refresh_diff()

    def _maybe_refresh_diff(self) -> None:
        # The overrides above fire during ParameterDialog.__init__ (and again
        # from set_values/set_advanced_names in our own __init__), before the
        # diff widgets exist yet -- skip until they've been built.
        if getattr(self, "_diff_table", None) is None:
            return
        self._refresh_diff()

    def _render_current_param_text(self) -> str:
        values = dict(self.collect_values())
        values.pop("task", None)
        return self._render_param_text(values, self.advanced_names())

    def _refresh_diff(self) -> None:
        self._new_param_text = self._render_current_param_text()
        rows = _diff_param_text(self._old_param_text, self._new_param_text)

        self._diff_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self._diff_table.setItem(row_index, 0, QTableWidgetItem(row.keyword))
            self._diff_table.setItem(row_index, 1, QTableWidgetItem(row.change))
            self._diff_table.setItem(row_index, 2, QTableWidgetItem(row.value_text))

        has_changes = bool(rows)
        self._diff_table.setVisible(has_changes)
        self._diff_no_changes_label.setVisible(not has_changes)

    # -- result -------------------------------------------------------------

    def edited_parameters(self) -> tuple[dict[str, object], tuple[str, ...]]:
        """Return the dialog's current ``(parameter_values, added_advanced)``.

        Pure accessor with no side effects, safe to call regardless of how the
        dialog was closed. Callers gate on ``dialog.exec() ==
        QDialog.DialogCode.Accepted`` (or ``dialog.result()``) before applying
        the result -- this dialog never mutates caller state itself.

        ``task`` is never included in the returned ``parameter_values``: both
        relaxation workflows lock it via ``locked_values``, so the caller
        already knows it and re-supplies it when re-materializing the row.
        """

        collected = dict(self.collect_values())
        collected.pop("task", None)
        normalized = to_jsonable(collected)
        parameter_values = normalized if isinstance(normalized, dict) else {}
        return parameter_values, self.advanced_names()
