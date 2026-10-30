"""The shared dialog base that shuts its structure viewers down once."""

from __future__ import annotations

import pytest

from mufinder.app.magnetic_field_widget import MagneticStructureBuilderDialog
from mufinder.app.quantum_analysis_dialog import QuantumPhononAnalysisDialog
from mufinder.app.relaxation_analysis_widget import (
    DisplacementAnalysisDialog,
    RelaxationClusteringDialog,
)
from mufinder.app.supercell_dialog import SupercellEditorDialog
from mufinder.app.viewer_dialog import StructureViewerDialog
from mufinder.app.workflow_dialogs import EnergyBarrierPathViewerDialog


class _Viewer:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls = 0
        self._error = error

    def shutdown(self) -> None:
        self.calls += 1
        if self._error is not None:
            raise self._error


@pytest.mark.parametrize("finish", ["accept", "reject", "close"])
def test_every_held_viewer_is_shut_down_once_however_the_dialog_ends(qapp, finish) -> None:
    dialog = StructureViewerDialog()
    first, second = _Viewer(), _Viewer()
    assert dialog.hold_viewer(first) is first
    dialog.hold_viewer(second)
    dialog.show()

    getattr(dialog, finish)()
    dialog.done(0)
    dialog.close()

    assert (first.calls, second.calls) == (1, 1)
    assert dialog.viewers_shut_down() is True


def test_a_viewer_already_destroyed_by_qt_does_not_stop_the_others(qapp) -> None:
    dialog = StructureViewerDialog()
    gone = dialog.hold_viewer(_Viewer(RuntimeError("Internal C++ object already deleted.")))
    alive = dialog.hold_viewer(_Viewer())

    dialog.done(0)

    assert (gone.calls, alive.calls) == (1, 1)


def test_other_errors_are_raised_after_every_viewer_is_shut_down(qapp) -> None:
    dialog = StructureViewerDialog()
    broken = dialog.hold_viewer(_Viewer(ValueError("broken")))
    after = dialog.hold_viewer(_Viewer())

    with pytest.raises(ValueError, match="broken"):
        dialog.shutdown_viewers()

    assert (broken.calls, after.calls) == (1, 1)
    dialog.shutdown_viewers()
    assert (broken.calls, after.calls) == (1, 1)


def test_nothing_is_shut_down_before_the_dialog_ends(qapp) -> None:
    dialog = StructureViewerDialog()
    viewer = dialog.hold_viewer(_Viewer())

    assert viewer.calls == 0
    assert dialog.viewers_shut_down() is False
    dialog.done(0)


@pytest.mark.parametrize(
    "dialog_class",
    [
        EnergyBarrierPathViewerDialog,
        SupercellEditorDialog,
        RelaxationClusteringDialog,
        DisplacementAnalysisDialog,
        MagneticStructureBuilderDialog,
        QuantumPhononAnalysisDialog,
    ],
)
def test_the_viewer_dialogs_share_the_shutdown(dialog_class) -> None:
    assert issubclass(dialog_class, StructureViewerDialog)
    for name in ("done", "closeEvent", "_shutdown_viewer"):
        assert name not in vars(dialog_class)
