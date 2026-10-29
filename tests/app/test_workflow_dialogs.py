from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QDialog, QWidget

from mufinder.app import param_dialog as param_dialog_module
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.param_widgets import create_param_editor
from mufinder.app.workflow_dialogs import (
    EnergyBarrierPathViewerDialog,
    QuantumCorrectionsConfigDialog,
    RowParameterRequest,
    SourceConfigureRequest,
    run_row_parameter_dialog,
    run_source_configure_dialog,
)
from mufinder.app.workflow_parameters import (
    build_workflow_parameter_surface,
    workflow_basic_parameter_names,
)
from mufinder.castep_io.schema import KeywordSpec
from tests.support.structures import host_structure


def _quantum_specs() -> list[KeywordSpec]:
    return [
        KeywordSpec(
            name="task",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("SINGLEPOINT", "PHONON"),
            widget_hint="dropdown",
            category="workflow",
            basic_workflows=("quantum_corrections",),
            default_value="SINGLEPOINT",
        ),
        KeywordSpec(
            name="phonon_method",
            file="param",
            kind="scalar",
            value_type="enum",
            allowed_values=("FINITEDISPLACEMENT", "DFPT", "LINEARRESPONSE"),
            widget_hint="dropdown",
            category="phonon",
            basic_workflows=("quantum_corrections",),
            default_value="FINITEDISPLACEMENT",
            allowed_tasks=("PHONON",),
        ),
    ]


def test_quantum_corrections_config_dialog_locks_task_and_tracks_pseudopotentials(qapp) -> None:
    dialog = QuantumCorrectionsConfigDialog(
        specs=_quantum_specs(),
        pseudopotential_family="norm_conserving",
        num_farms=6,
    )

    assert dialog.collect_values()["task"] == "PHONON"
    assert dialog.pseudopotential_family() == "norm_conserving"
    assert dialog.num_farms() == 6


def test_quantum_corrections_config_dialog_labels_the_farm_count(qapp) -> None:
    from PySide6.QtWidgets import QLabel

    dialog = QuantumCorrectionsConfigDialog(specs=_quantum_specs(), num_farms=5)

    labels = [label.text() for label in dialog.findChildren(QLabel)]
    assert "Farms" in labels
    assert "A value of 1 disables task farming." in labels
    assert not any("stage" in text.lower() for text in labels)
    assert dialog._num_farms_spin.isEnabled() is True
    assert dialog._pseudo_combo.isEnabled() is True


@pytest.mark.parametrize("read_only", [False, True])
def test_quantum_corrections_config_dialog_locks_everything_or_nothing(
    qapp, read_only: bool
) -> None:
    dialog = QuantumCorrectionsConfigDialog(
        specs=_quantum_specs(), num_farms=5, read_only=read_only
    )

    assert dialog._rows["phonon_method"].editor is not None
    assert dialog._rows["phonon_method"].editor.isEnabled() is not read_only
    assert dialog._pseudo_combo.isEnabled() is not read_only
    assert dialog._num_farms_spin.isEnabled() is not read_only


def test_atomic_structure_viewer_line_overlays_extend_scene(qapp) -> None:
    viewer = AtomicStructureViewer()
    viewer.set_structure(host_structure(label="supercell"))
    viewer.set_line_overlays(
        [
            {
                "start": (1.0, 1.0, 1.0),
                "end": (6.0, 1.0, 1.0),
                "arrow": True,
            }
        ]
    )

    _positions, _numbers, _symbols, _colors, _radii, extra_points = viewer._scene_geometry()

    assert viewer._line_overlay_specs
    assert any(np.allclose(point, [1.0, 1.0, 1.0]) for point in extra_points)
    assert any(np.allclose(point, [6.0, 1.0, 1.0]) for point in extra_points)
    viewer.close()


def test_energy_barrier_path_viewer_dialog_toggles_arrow_overlay(qapp) -> None:
    dialog = EnergyBarrierPathViewerDialog(
        structure=host_structure(label="supercell"),
        start_fractional=(0.2, 0.2, 0.2),
        end_fractional=(0.8, 0.2, 0.2),
        title="Path Preview",
    )

    assert len(dialog._viewer._line_overlay_specs) == 1
    assert len(dialog._viewer._point_overlay_specs) == 2

    dialog._show_arrow.setChecked(False)

    assert dialog._viewer._line_overlay_specs == []
    dialog.close()


def _row_request(
    row_values: Mapping[int, Mapping[str, object]],
    *,
    row_advanced: Mapping[int, set[str]] | None = None,
    values: Mapping[str, object] | None = None,
    read_only: bool = False,
    sanitize: Callable[[Mapping[str, object]], dict[str, object]] = dict,
) -> RowParameterRequest[int]:
    return RowParameterRequest(
        specs=build_workflow_parameter_surface(
            workflow_stage="pristine_relaxation",
            basic_names=workflow_basic_parameter_names(),
        ),
        workflow_stage="pristine_relaxation",
        subject="Relaxation Parameters",
        values=dict(next(iter(row_values.values()), {})) if values is None else values,
        row_values=row_values,
        row_advanced=row_advanced or {},
        sanitize=sanitize,
        locked_values={"task": "GEOMETRYOPTIMIZATION"},
        read_only=read_only,
        intro_text="Included rows: a, b",
    )


def _capture_exec(
    monkeypatch: pytest.MonkeyPatch,
    result: QDialog.DialogCode,
    edit: Callable[[ParameterDialog], None] | None = None,
) -> list[ParameterDialog]:
    opened: list[ParameterDialog] = []

    def fake_exec(dialog: ParameterDialog) -> QDialog.DialogCode:
        opened.append(dialog)
        if edit is not None:
            edit(dialog)
        return result

    monkeypatch.setattr(ParameterDialog, "exec", fake_exec)
    return opened


def test_run_row_parameter_dialog_merges_batch_edit_into_each_row(qapp, monkeypatch) -> None:
    rows = {
        1: {"max_scf_cycles": 80, "xc_functional": "PBE"},
        2: {"max_scf_cycles": 120, "xc_functional": "PBE"},
    }
    sanitized: list[dict[str, object]] = []

    def sanitize(values: Mapping[str, object]) -> dict[str, object]:
        sanitized.append(dict(values))
        return {**values, "sanitized": True}

    opened = _capture_exec(
        monkeypatch,
        QDialog.DialogCode.Accepted,
        lambda dialog: dialog._rows["xc_functional"].editor.set_value("LDA"),
    )

    result = run_row_parameter_dialog(
        _row_request(
            rows,
            values={"xc_functional": "PBE"},
            row_advanced={1: {"max_scf_cycles"}, 2: set()},
            sanitize=sanitize,
        )
    )

    assert result is not None
    (dialog,) = opened
    assert dialog.windowTitle() == "Configure Relaxation Parameters"
    assert dialog.untouched_mixed_names() == {"max_scf_cycles"}
    assert len(sanitized) == 1 and result.values["sanitized"] is True
    # The differing parameter the user left alone stays per row; the edit applies to both.
    assert {key: values["max_scf_cycles"] for key, values in result.row_values.items()} == {
        1: 80,
        2: 120,
    }
    assert {values["xc_functional"] for values in result.row_values.values()} == {"LDA"}
    assert result.row_values[1]["sanitized"] is True
    assert set(result.row_values) == set(result.row_advanced) == {1, 2}


def test_run_row_parameter_dialog_seeds_advanced_names_from_every_row(qapp, monkeypatch) -> None:
    opened = _capture_exec(monkeypatch, QDialog.DialogCode.Rejected)

    run_row_parameter_dialog(
        _row_request(
            {1: {"xc_functional": "PBE"}, 2: {"xc_functional": "PBE"}},
            row_advanced={1: {"max_scf_cycles"}, 2: {"spin_polarized"}},
        )
    )

    (dialog,) = opened
    assert {"max_scf_cycles", "spin_polarized"} <= set(dialog.advanced_names())


def test_run_row_parameter_dialog_prefers_spacing_when_every_row_uses_it(qapp, monkeypatch) -> None:
    collected: list[dict[str, object]] = []
    _capture_exec(
        monkeypatch,
        QDialog.DialogCode.Rejected,
        lambda dialog: collected.append(dialog.collect_values()),
    )
    rows = {1: {"kpoint_mp_spacing": 0.05}, 2: {"kpoint_mp_spacing": 0.05}}

    run_row_parameter_dialog(
        _row_request(rows, values={"kpoint_mp_spacing": 0.05, "kpoint_mp_grid": (4, 4, 4)})
    )

    assert "kpoint_mp_spacing" in collected[0]
    assert "kpoint_mp_grid" not in collected[0]


@pytest.mark.parametrize("exec_result", [QDialog.DialogCode.Accepted, QDialog.DialogCode.Rejected])
def test_run_row_parameter_dialog_inspects_locked_rows_without_a_result(
    qapp, monkeypatch, exec_result
) -> None:
    opened = _capture_exec(monkeypatch, exec_result)

    result = run_row_parameter_dialog(_row_request({1: {"xc_functional": "PBE"}}, read_only=True))

    assert result is None
    (dialog,) = opened
    assert dialog.windowTitle() == "Inspect Relaxation Parameters"
    assert dialog._read_only is True


def test_run_row_parameter_dialog_returns_none_when_cancelled(qapp, monkeypatch) -> None:
    sanitized: list[object] = []
    _capture_exec(monkeypatch, QDialog.DialogCode.Rejected)

    request = _row_request(
        {1: {"xc_functional": "PBE"}}, sanitize=lambda v: sanitized.append(v) or {}
    )

    assert run_row_parameter_dialog(request) is None
    assert sanitized == []


def test_run_row_parameter_dialog_without_rows_opens_nothing(qapp, monkeypatch) -> None:
    opened = _capture_exec(monkeypatch, QDialog.DialogCode.Accepted)

    assert run_row_parameter_dialog(_row_request({}, values={})) is None
    assert opened == []


def test_run_row_parameter_dialog_deletes_the_dialog_after_it_closes(qapp, monkeypatch) -> None:
    _capture_exec(monkeypatch, QDialog.DialogCode.Rejected)
    parent = QWidget()

    for _ in range(5):
        run_row_parameter_dialog(_row_request({1: {"xc_functional": "PBE"}}), parent=parent)
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    assert parent.findChildren(ParameterDialog) == []
    parent.deleteLater()


def test_run_row_parameter_dialog_builds_each_editor_once(qapp, monkeypatch) -> None:
    editor_parents: list[object] = []

    def counting_create_param_editor(spec, parent=None, **kwargs):
        editor_parents.append(parent)
        return create_param_editor(spec, parent, **kwargs)

    monkeypatch.setattr(param_dialog_module, "create_param_editor", counting_create_param_editor)
    opened = _capture_exec(monkeypatch, QDialog.DialogCode.Rejected)

    run_row_parameter_dialog(
        _row_request(
            {1: {"max_scf_cycles": 80, "xc_functional": "PBE"}, 2: {"max_scf_cycles": 120}},
            values={"xc_functional": "PBE"},
            row_advanced={1: {"max_scf_cycles"}, 2: set()},
        )
    )

    (dialog,) = opened
    table_editors = [parent for parent in editor_parents if parent is dialog._table_widget]
    row_editors = [row.editor for row in dialog._rows.values() if row.editor is not None]
    assert row_editors
    assert len(table_editors) == len(row_editors)
    # The batch seed still reaches the single build.
    assert "max_scf_cycles" in dialog.advanced_names()
    assert "max_scf_cycles" in dialog.untouched_mixed_names()


def _source_request(*, read_only: bool = False) -> SourceConfigureRequest:
    return SourceConfigureRequest(
        keys=("a", "b"),
        row_values=({"phonon_method": "DFPT"}, {"phonon_method": "LINEARRESPONSE"}),
        advanced_names=frozenset(),
        title="Configure Things",
        intro_text="Selected: 2",
        read_only=read_only,
        num_farms=1,
        species_options=(),
    )


def test_run_source_configure_dialog_seeds_shared_values_and_returns_the_edit(
    qapp, monkeypatch
) -> None:
    opened = _capture_exec(monkeypatch, QDialog.DialogCode.Accepted)
    dialog = ParameterDialog(specs=_quantum_specs(), workflow_stage="quantum_corrections")

    result = run_source_configure_dialog(dialog, _source_request())

    assert opened == [dialog]
    assert dialog.windowTitle() == "Configure Things"
    assert result is not None
    # The rows disagree on the phonon method, so the dialog marks it mixed and,
    # left untouched, reports it so each row keeps its own.
    assert result.untouched_mixed == frozenset({"phonon_method"})
    assert "phonon_method" not in result.values
    assert result.values == dialog.collect_values()


def test_run_source_configure_dialog_returns_nothing_when_cancelled_or_read_only(
    qapp, monkeypatch
) -> None:
    _capture_exec(monkeypatch, QDialog.DialogCode.Rejected)
    dialog = ParameterDialog(specs=_quantum_specs(), workflow_stage="quantum_corrections")
    assert run_source_configure_dialog(dialog, _source_request()) is None

    _capture_exec(monkeypatch, QDialog.DialogCode.Accepted)
    dialog = ParameterDialog(
        specs=_quantum_specs(), workflow_stage="quantum_corrections", read_only=True
    )
    assert run_source_configure_dialog(dialog, _source_request(read_only=True)) is None
