from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pytest
from PySide6.QtWidgets import QDialog

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.workflow_dialogs import (
    EnergyBarrierPathViewerDialog,
    QuantumCorrectionsConfigDialog,
    RowParameterRequest,
    run_row_parameter_dialog,
)
from mufinder.app.workflow_parameters import build_workflow_parameter_surface, workflow_basic_parameter_names
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


def test_quantum_corrections_config_dialog_can_lock_parameters_but_keep_farms_editable(qapp) -> None:
    dialog = QuantumCorrectionsConfigDialog(
        specs=_quantum_specs(),
        num_farms=5,
        editable_param_names=(),
        pseudopotential_editable=False,
        num_farms_editable=True,
    )

    assert dialog._rows["phonon_method"].editor is not None
    assert dialog._rows["phonon_method"].editor.isEnabled() is False
    assert dialog._pseudo_combo.isEnabled() is False
    assert dialog._num_farms_spin.isEnabled() is True
    assert dialog._add_advanced_button.isEnabled() is False


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
    assert {key: values["max_scf_cycles"] for key, values in result.row_values.items()} == {1: 80, 2: 120}
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

    request = _row_request({1: {"xc_functional": "PBE"}}, sanitize=lambda v: sanitized.append(v) or {})

    assert run_row_parameter_dialog(request) is None
    assert sanitized == []


def test_run_row_parameter_dialog_without_rows_opens_nothing(qapp, monkeypatch) -> None:
    opened = _capture_exec(monkeypatch, QDialog.DialogCode.Accepted)

    assert run_row_parameter_dialog(_row_request({}, values={})) is None
    assert opened == []
