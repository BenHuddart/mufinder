from __future__ import annotations

import numpy as np

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.workflow_dialogs import EnergyBarrierPathViewerDialog, QuantumCorrectionsConfigDialog
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
