"""Every dialog and workflow page meets the keyboard baseline.

The baseline (``docs/architecture.rst``, Keyboard baseline):

1. Tab and Shift+Tab reach every enabled, shown control.
2. Tab follows the screen panel by panel, as ``screen_tab_order`` reads it: a panel
   is finished before the next, and inside one, top to bottom and then left to right.
3. Enter presses the default button, where a dialog has one, and Escape cancels.

Each dialog is built with test content, shown and walked with real Tab and
Shift+Tab presses; Enter and Escape are pressed from where it opens and from every
stop, with the dialog's accept and reject recorded instead of run. Each page is
walked inside the main window of a project with a row on every page that lists rows.
``DIALOGS`` must hold every ``QDialog`` subclass in ``mufinder.app``, and
``KNOWN_EXCEPTIONS`` the places that keep a deliberate difference, each with its
reason. An exception that starts passing fails, so the list cannot go stale.
"""

from __future__ import annotations

from collections.abc import Callable
import importlib
from pathlib import Path
import pkgutil
from uuid import uuid4

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QWidget,
)
import shiboken6

import mufinder.app
from tests.support.qt import open_and_take_focus
from tests.support.keyboard import (
    DialogKeys,
    address,
    cell_table,
    controls_to_reach,
    describe,
    reverse_faults,
    screen_order_faults,
    tab_walk,
    unreached,
)

# (target, rule): (the dialog attribute holding the control it is excused on, or None
# for every control, and why it keeps its difference). The rules are "reach", "order",
# "enter" and "escape".
KNOWN_EXCEPTIONS: dict[tuple[str, str], tuple[str | None, str]] = {
    ("ClusterSubmissionDialog", "enter"): (
        "_queue_combo",
        "Enter in the editable queue box keeps the typed queue name instead of pressing OK.",
    ),
    ("MagneticStructureBuilderDialog", "enter"): (
        None,
        "Enter reaches the dialog after a preview-repeat spin box commits its value, so it "
        "presses Use Structure only from a focused button and never closes the builder.",
    ),
    ("GeometryProgressDialog", "reach"): (
        "_table",
        "The criteria table is display-only and is not a Tab stop.",
    ),
}

# Dialog classes in ``mufinder.app`` that the registry does not build, with the reason.
NOT_BUILT: dict[str, str] = {}


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """No message box, file dialog, input dialog or nested ``exec`` may block a walk."""

    def _answer(answer):
        return staticmethod(lambda *args, **kwargs: answer)

    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, _answer(QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "question", _answer(QMessageBox.StandardButton.No))
    for name in ("getOpenFileName", "getSaveFileName"):
        monkeypatch.setattr(QFileDialog, name, _answer(("", "")))
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", _answer(([], "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _answer(""))
    monkeypatch.setattr(QInputDialog, "getText", _answer(("", False)))
    monkeypatch.setattr(QInputDialog, "getItem", _answer(("", False)))
    monkeypatch.setattr(QDialog, "exec", lambda self, *args: 0)
    monkeypatch.setattr(QMessageBox, "exec", lambda self, *args: 0)


class _Context:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp = tmp_path
        self.monkeypatch = monkeypatch
        # Parents a dialog is opened from, closed with it.
        self.keep: list[QWidget] = []


# ------------------------------------------------------------------- dialog builders


def _hubbard_u(ctx):
    from tests.app.test_table_keyboard import _hubbard_u_dialog

    return _hubbard_u_dialog()


def _new_project(ctx):
    from mufinder.app.new_project_dialog import NewProjectDialog

    dialog = NewProjectDialog(str(ctx.tmp))
    dialog._name_edit.setText("Keyboard")
    return dialog


def _parameter_help(ctx):
    from mufinder.app.help_dialog import ParameterHelpDialog
    from mufinder.castep_io.schema import get_keyword_spec

    return ParameterHelpDialog(get_keyword_spec("param", "cut_off_energy"))


def _supercell_editor(ctx):
    from tests.app.test_supercell_dialog import _make_editor

    return _make_editor(ctx.tmp)


def _kpoint_spacing(ctx):
    from mufinder.app.kpoint_convergence_widget import KPointSpacingDialog

    return KPointSpacingDialog({"spacing": 0.25, "replace_existing": True})


def _kpoint_grid(ctx):
    from mufinder.app.kpoint_convergence_widget import KPointGridDialog

    return KPointGridDialog((2, 3, 4))


def _advanced_selection(ctx):
    from mufinder.app.param_dialog import _AdvancedSelectionDialog
    from tests.app.test_param_dialog import _picker_dialog

    parent = _picker_dialog()
    ctx.keep.append(parent)
    return _AdvancedSelectionDialog(parent._available_advanced_specs(), parent)


def _parameter(ctx):
    from mufinder.app.param_dialog import ParameterDialog
    from tests.app.test_param_dialog import _dialog_specs

    return ParameterDialog(specs=_dialog_specs(), workflow_stage="convergence_cutoff")


def _parameter_tall(ctx):
    from tests.app.test_param_dialog import _tall_parameter_dialog

    return _tall_parameter_dialog(read_only=False)


def _cluster_setup(ctx):
    from mufinder.app.cluster_setup_dialog import ClusterSetupDialog
    from tests.app.test_cluster_setup_dialog import _FakeSettings

    return ClusterSetupDialog(settings=_FakeSettings())


def _magnetic_texture(ctx):
    from tests.app.test_table_keyboard import _magnetic_texture_dialog

    return _magnetic_texture_dialog()


def _continue_relaxation(ctx):
    from tests.app.test_continue_relaxation_dialog import _make_dialog

    return _make_dialog()


def _relaxation_import(ctx):
    from mufinder.app.relaxation_import_dialog import RelaxationImportDialog
    from mufinder.services.relaxation_import import scan_relaxation_folder
    from tests.services.test_relaxation_import_scan import HOST_LATTICE, build_folder

    folder = build_folder(ctx.tmp, sites=(1, 2))
    return RelaxationImportDialog(scan_relaxation_folder(folder, host_lattice=HOST_LATTICE))


def _import_batch_chooser(ctx):
    from mufinder.app.relaxation_import_dialog import ImportBatchChooser

    return ImportBatchChooser(["Batch 1 (2 relaxations)", "Batch 2 (3 relaxations)"])


def _cluster_positions(ctx):
    from mufinder.app.muon_generation_widget import _ClusterPositionsDialog

    return _ClusterPositionsDialog([{"label": "Site A", "fractional": [0.1, 0.2, 0.3]}])


def _quantum_corrections_config(ctx):
    from mufinder.app.workflow_dialogs import QuantumCorrectionsConfigDialog
    from tests.app.test_workflow_dialogs import _quantum_specs

    return QuantumCorrectionsConfigDialog(specs=_quantum_specs(), num_farms=6)


def _hyperfine_config(ctx):
    from mufinder.app.workflow_dialogs import HyperfineCouplingConfigDialog
    from tests.app.test_workflow_dialogs import _quantum_specs

    return HyperfineCouplingConfigDialog(
        specs=_quantum_specs(), intro_text="One magres job per site."
    )


def _neb_config(ctx):
    from mufinder.app.workflow_dialogs import NudgedElasticBandConfigDialog
    from tests.app.test_param_dialog import _neb_dialog_specs

    return NudgedElasticBandConfigDialog(specs=_neb_dialog_specs(), num_farms=3)


def _quantum_analysis(ctx):
    from tests.app.test_quantum_analysis_dialog import _open_dialog

    return _open_dialog(ctx.tmp)


def _coupling_pair(ctx):
    from mufinder.app.quantum_analysis_dialog import MuonCouplingPairDialog
    from tests.app.test_quantum_analysis_dialog import _open_dialog

    parent = _open_dialog(ctx.tmp)
    ctx.keep.append(parent)
    return MuonCouplingPairDialog(parent._calculation)


def _quantum_help(ctx):
    from mufinder.app.quantum_analysis_dialog import QuantumAnalysisHelpDialog

    return QuantumAnalysisHelpDialog(
        {
            "title": "Equation help",
            "paragraphs": ("Plain-language context.",),
            "equations": (r"E_0=\frac{1}{2}\hbar\omega",),
            "symbols": ((r"E_0", "Ground-state energy."),),
        }
    )


def _geometry_progress(ctx):
    from mufinder.app.geometry_progress_dialog import GeometryProgressDialog
    from mufinder.castep_io.geometry_progress import parse_geometry_progress

    fixture = Path(__file__).parents[1] / "fixtures" / "geometry" / "Si2_bfgs.castep"
    return GeometryProgressDialog("Si2", parse_geometry_progress(fixture))


def _charge_state(ctx):
    from mufinder.app.muon_generation_support import MuonChargeStateDialog

    return MuonChargeStateDialog(site_count=2)


def _cutoff_range(ctx):
    from mufinder.app.convergence_widget import CutoffRangeDialog

    return CutoffRangeDialog({"start": 300.0, "stop": 360.0, "step": 30.0})


def _failed_job(ctx):
    from tests.app.test_failed_job_dialog import _dialog, _report

    reports = [
        _report(log_excerpt="line 1\nline 2", output_directory=ctx.tmp),
        _report("Muon_relax_0009", notes=("A note.",)),
    ]
    return _dialog(reports, can_resubmit=True, can_continue=True)


def _site_selection(ctx):
    from mufinder.app.magnetic_site_dialog import MuonSiteSelectionDialog
    from tests.support.magnetic_field_sites import feo_host_structure, uep_cluster_document

    return MuonSiteSelectionDialog(uep_cluster_document(ctx.tmp), feo_host_structure())


def _structure_viewer(ctx):
    from mufinder.app.viewer_dialog import StructureViewerDialog

    return StructureViewerDialog()


def _structure_builder(ctx):
    from tests.app.test_table_keyboard import _magnetic_structure_builder

    return _magnetic_structure_builder(ctx.tmp)


def _sample_photo(ctx):
    from mufinder.app.sample_photo_dialog import SamplePhotoDialog
    from tests.app.test_sample_photo_dialog import _plate_photo

    dialog = SamplePhotoDialog()
    dialog.load_photo(_plate_photo(ctx.tmp))
    return dialog


_LATTICE = ((5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 0.0, 5.0))


def _add_slice_plane(ctx):
    from mufinder.app.uep_wizard import _AddSlicePlaneDialog

    return _AddSlicePlaneDialog(
        selected_positions_ang=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        selected_positions_fractional=[(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)],
        lattice_vectors=_LATTICE,
    )


def _edit_slice_plane(ctx):
    from mufinder.app.uep_wizard import _EditSlicePlaneDialog
    from mufinder.science.uep import plane_from_fractional_anchor_and_miller

    plane = plane_from_fractional_anchor_and_miller(
        anchor_fractional=(0.25, 0.5, 0.75),
        miller_indices=(1.0, 0.0, 1.0),
        lattice_vectors=_LATTICE,
        points_u=21,
        points_v=21,
    )
    return _EditSlicePlaneDialog(plane_number=1, plane=plane, lattice_vectors=_LATTICE)


def _manage_viewer_muons(ctx):
    from tests.app.test_uep_wizard import _si2_dialog

    parent = _si2_dialog()
    ctx.keep.append(parent)
    parent._open_viewer_muon_dialog()
    return parent._viewer_muon_dialog


def _uep_analysis(ctx):
    from tests.app.test_uep_wizard import _si2_dialog

    return _si2_dialog()


def _nuclear_relaxation(ctx):
    from tests.app.test_table_keyboard import _nuclear_relaxation_dialog

    return _nuclear_relaxation_dialog()


def _dipolar_tensor(ctx):
    from tests.app.test_table_keyboard import _dipolar_tensor_dialog

    return _dipolar_tensor_dialog()


def _local_submission(ctx):
    from mufinder.app.submission_dialog import LocalSubmissionDialog

    return LocalSubmissionDialog(
        job_labels=["conv_ecut_300", "conv_ecut_330"], mpirun_configured=True
    )


def _castep_setup(ctx):
    from mufinder.app.castep_setup_dialog import CastepSetupDialog
    from tests.app.test_castep_setup_dialog import _FakeSettings

    return CastepSetupDialog(settings=_FakeSettings())


def _relaxation_analysis_job(ctx):
    from mufinder.app.relaxation_analysis_widget import RelaxationAnalysisJobDialog
    from mufinder.services.source_candidates import CompletedRelaxationAnalysisSource

    ids = [uuid4(), uuid4()]
    sources = [
        CompletedRelaxationAnalysisSource(
            job_id=job_id,
            job_label=label,
            charge_state="positive_muon",
            structure_id=uuid4(),
            site_id=None,
            folded_fractional_position=position,
            total_energy_ev=energy,
        )
        for job_id, label, position, energy in (
            (ids[0], "Included", (0.1, 0.2, 0.3), -10.0),
            (ids[1], "Lower candidate", (0.4, 0.5, 0.6), -10.25),
        )
    ]
    return RelaxationAnalysisJobDialog(
        title="Add Structure to Analysis",
        sources=sources,
        excluded_job_ids={ids[0]},
        reference_job_ids={ids[0]},
    )


def _relaxation_clustering(ctx):
    from tests.app.test_plot_canvas_layout import _relaxation_clustering

    return _relaxation_clustering(1)


def _displacement_analysis(ctx):
    from tests.app.test_plot_canvas_layout import _displacements

    return _displacements()


def _supercell_suggestion(ctx):
    from mufinder.app.supercell_suggestion_dialog import SupercellSuggestionDialog
    from tests.app.test_supercell_suggestion_dialog import _three_candidate_suggestion

    return SupercellSuggestionDialog(suggestion=_three_candidate_suggestion())


def _comparison_job(ctx):
    from mufinder.app.comparison_job_dialog import ComparisonJobDialog
    from tests.app.test_comparison_job_dialog import _job

    return ComparisonJobDialog(jobs=[_job("job-a"), _job("job-b")], default_label="Comparison 1")


def _field_settings(ctx):
    from mufinder.app.magnetic_field_widget import (
        MagneticFieldCalculationSettings,
        MagneticFieldCalculationSettingsDialog,
    )

    return MagneticFieldCalculationSettingsDialog(
        MagneticFieldCalculationSettings(supercell=(12, 24, 36), nangles=4096)
    )


def _block_parameter(ctx):
    from tests.app.test_table_keyboard import _block_dialog

    return _block_dialog()


def _convergence_examiner(ctx):
    from tests.app.test_plot_canvas_layout import _convergence_examiner

    return _convergence_examiner()


def _convergence_examiner_narrow(ctx):
    dialog = _convergence_examiner(ctx)
    dialog.resize(dialog.minimumSizeHint().width(), dialog.sizeHint().height())
    return dialog


def _cluster_submission(ctx):
    from mufinder.app.cluster_submission_dialog import ClusterSubmissionDialog
    from tests.app.test_cluster_submission_dialog import _configs

    return ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())


def _magnetic_component(ctx):
    from tests.app.test_table_keyboard import _magnetic_component_dialog

    return _magnetic_component_dialog()


def _atom_phase(ctx):
    from tests.app.test_table_keyboard import _atom_phase_dialog

    return _atom_phase_dialog()


def _magnetic_preset(ctx):
    from mufinder.app.magnetic_component_dialogs import MagneticPresetDialog
    from mufinder.science.magnetic_structure import MagneticStructureModel
    from tests.support.magnetic_field_sites import feo_host_structure

    return MagneticPresetDialog(feo_host_structure(), model=MagneticStructureModel())


def _demagnetizing(ctx):
    from tests.app.test_demagnetizing_dialogs import _demag_dialog

    return _demag_dialog(ctx.monkeypatch)


def _field_distribution(ctx):
    from tests.app.test_plot_canvas_layout import _field_distribution

    return _field_distribution()


def _project_defaults(ctx):
    from mufinder.app.project_defaults_widget import ProjectDefaultsDialog
    from tests.app.test_project_defaults_widget import _summary

    return ProjectDefaultsDialog(_summary())


# Target name -> builder. A name is its dialog's class name, with a variant in brackets.
DIALOGS: dict[str, Callable[[_Context], QDialog]] = {
    "AtomPhaseDialog": _atom_phase,
    "BlockParameterDialog": _block_parameter,
    "CastepSetupDialog": _castep_setup,
    "ClusterSetupDialog": _cluster_setup,
    "ClusterSubmissionDialog": _cluster_submission,
    "ComparisonJobDialog": _comparison_job,
    "ContinueRelaxationDialog": _continue_relaxation,
    "ConvergenceExaminerDialog": _convergence_examiner,
    "ConvergenceExaminerDialog (narrow)": _convergence_examiner_narrow,
    "CutoffRangeDialog": _cutoff_range,
    "DemagnetizingFieldAdvancedDialog": _demagnetizing,
    "DipolarTensorDialog": _dipolar_tensor,
    "DisplacementAnalysisDialog": _displacement_analysis,
    "FailedJobDialog": _failed_job,
    "GeometryProgressDialog": _geometry_progress,
    "HubbardUDialog": _hubbard_u,
    "HyperfineCouplingConfigDialog": _hyperfine_config,
    "ImportBatchChooser": _import_batch_chooser,
    "KPointGridDialog": _kpoint_grid,
    "KPointSpacingDialog": _kpoint_spacing,
    "LocalSubmissionDialog": _local_submission,
    "MagneticComponentDialog": _magnetic_component,
    "MagneticFieldCalculationSettingsDialog": _field_settings,
    "MagneticFieldDistributionDialog": _field_distribution,
    "MagneticPresetDialog": _magnetic_preset,
    "MagneticStructureBuilderDialog": _structure_builder,
    "MagneticTextureDialog": _magnetic_texture,
    "MuonChargeStateDialog": _charge_state,
    "MuonCouplingPairDialog": _coupling_pair,
    "MuonSiteSelectionDialog": _site_selection,
    "NewProjectDialog": _new_project,
    "NuclearRelaxationDialog": _nuclear_relaxation,
    "NudgedElasticBandConfigDialog": _neb_config,
    "ParameterDialog": _parameter,
    "ParameterDialog (tall)": _parameter_tall,
    "ParameterHelpDialog": _parameter_help,
    "ProjectDefaultsDialog": _project_defaults,
    "QuantumAnalysisHelpDialog": _quantum_help,
    "QuantumCorrectionsConfigDialog": _quantum_corrections_config,
    "QuantumPhononAnalysisDialog": _quantum_analysis,
    "RelaxationAnalysisJobDialog": _relaxation_analysis_job,
    "RelaxationClusteringDialog": _relaxation_clustering,
    "RelaxationImportDialog": _relaxation_import,
    "SamplePhotoDialog": _sample_photo,
    "StructureViewerDialog": _structure_viewer,
    "SupercellEditorDialog": _supercell_editor,
    "SupercellSuggestionDialog": _supercell_suggestion,
    "UepAnalysisDialog": _uep_analysis,
    "_AddSlicePlaneDialog": _add_slice_plane,
    "_AdvancedSelectionDialog": _advanced_selection,
    "_ClusterPositionsDialog": _cluster_positions,
    "_EditSlicePlaneDialog": _edit_slice_plane,
    "_ManageViewerMuonsDialog": _manage_viewer_muons,
}


# ------------------------------------------------------------------- checks


# A fault: the control it is on, when there is one, and what went wrong.
Fault = tuple[QWidget | None, str]


def _check(target: str, faults: dict[str, list[Fault]], owner: QWidget | None = None) -> None:
    """Fail on a fault that no known exception excuses, and on an exception that excuses none."""

    def excused(rule: str, widget: QWidget | None) -> bool:
        if (target, rule) not in KNOWN_EXCEPTIONS:
            return False
        attribute = KNOWN_EXCEPTIONS[(target, rule)][0]
        if attribute is None:
            return True
        control = getattr(owner, attribute)
        return widget is not None and (widget is control or control.isAncestorOf(widget))

    unexpected = {
        rule: [text for widget, text in found if not excused(rule, widget)]
        for rule, found in faults.items()
    }
    unexpected = {rule: texts for rule, texts in unexpected.items() if texts}
    stale = [
        rule
        for (name, rule) in KNOWN_EXCEPTIONS
        if name == target and not any(excused(rule, widget) for widget, _ in faults.get(rule, ()))
    ]
    assert not unexpected, f"{target} breaks the keyboard baseline: {unexpected}"
    assert not stale, f"{target} now meets {stale}: take it out of KNOWN_EXCEPTIONS"


def _walk_faults(root: QWidget, window: QWidget) -> tuple[dict[str, list[Fault]], list[QWidget]]:
    """Rules 1 and 2 for ``root``, walked round ``window`` from the control that has focus.

    A part of a window (``root`` is not ``window``) must be walked from its first
    control on screen; its stops must then come first, in one run.
    """

    start = QApplication.focusWidget()
    forward = tab_walk(window)
    backward = tab_walk(window, backward=True)
    faults: dict[str, list[Fault]] = {"reach": [], "order": []}
    for walk in (forward, backward):
        if walk.fault:
            faults["reach"].append((None, walk.fault))
    whole = root is window
    stops = forward.stops
    if not whole and stops and start is not None:
        # The walk ends back on the control it started from: put that first.
        stops = [stops[-1], *stops[:-1]]
        inside = [root.isAncestorOf(stop) for stop in stops]
        leaves = inside.index(False) if False in inside else len(inside)
        if any(inside[leaves:]):
            back_at = stops[leaves + inside[leaves:].index(True)]
            faults["order"].append(
                (
                    back_at,
                    f"Tab leaves it for {describe(stops[leaves])} and comes back at "
                    f"{describe(back_at)}",
                )
            )
    ahead = [stop for stop in stops if whole or root.isAncestorOf(stop)]
    back = [stop for stop in backward.stops if whole or root.isAncestorOf(stop)]
    required = controls_to_reach(root)
    faults["reach"] += [(w, f"Tab misses {describe(w)}") for w in unreached(required, ahead)]
    faults["reach"] += [
        (w, f"Shift+Tab misses {describe(w)}")
        for w in unreached([e for e in required if cell_table(e[0]) is None], back)
    ]
    faults["order"] += [(None, text) for text in screen_order_faults(ahead, root, cycle=whole)]
    reverse = reverse_faults(ahead, back)
    if reverse:
        faults["order"].append((None, reverse))
    return faults, ahead


def _key_faults(dialog: QDialog, opening: QWidget | None, stops: list[QWidget]) -> dict:
    """Rule 3: Enter presses the default button and Escape cancels, from every stop."""

    keys = DialogKeys(dialog)
    defaults = keys.default_buttons()
    faults: dict[str, list[Fault]] = {"enter": [], "escape": []}
    starts = [opening] if opening is not None else []
    starts += [stop for stop in stops if stop is not opening]
    for start in starts:
        if not shiboken6.isValid(start) or not start.isVisible():
            continue
        # Enter on a focused push button presses that button; it is checked where the
        # dialog opens, and every other stop must press the default button.
        if defaults and (start is opening or not isinstance(start, QPushButton)):
            result = keys.press(Qt.Key.Key_Return, start)
            wanted = [describe(start)] if isinstance(start, QPushButton) else defaults[:1]
            if result.pressed[:1] != wanted:
                faults["enter"].append((start, f"{describe(start)}: {result.pressed or 'nothing'}"))
        result = keys.press(Qt.Key.Key_Escape, start)
        if result.closed not in ("reject", "done(0)"):
            faults["escape"].append((start, f"{describe(start)}: {result.closed or 'nothing'}"))
    return faults


@pytest.mark.parametrize("target", sorted(DIALOGS))
def test_dialog_meets_the_keyboard_baseline(target, qapp, tmp_path, monkeypatch) -> None:
    ctx = _Context(tmp_path, monkeypatch)
    dialog = DIALOGS[target](ctx)
    try:
        assert type(dialog).__name__ == target.split(" (")[0]
        opening = open_and_take_focus(dialog)
        faults, stops = _walk_faults(dialog, dialog)
        faults.update(_key_faults(dialog, opening, stops))
        _check(target, faults, dialog)
    finally:
        for widget in (dialog, *ctx.keep):
            if shiboken6.isValid(widget):
                widget.hide()
                widget.deleteLater()
        QApplication.processEvents()


def _dialog_classes() -> set[str]:
    for module in pkgutil.walk_packages(mufinder.app.__path__, "mufinder.app."):
        importlib.import_module(module.name)
    found: set[str] = set()
    pending = list(QDialog.__subclasses__())
    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())
        if cls.__module__.startswith("mufinder.app."):
            found.add(cls.__name__)
    return found


def test_every_dialog_class_is_walked() -> None:
    registered = {target.split(" (")[0] for target in DIALOGS}
    classes = _dialog_classes()

    assert classes - registered - set(NOT_BUILT) == set(), (
        "Add each new dialog to DIALOGS in this file"
    )
    assert (registered | set(NOT_BUILT)) - classes == set(), "Remove dialogs that are gone"


def test_every_known_exception_names_a_target_and_a_rule() -> None:
    targets = set(DIALOGS) | {f"page:{route}" for route in _page_routes()}

    for target, rule in KNOWN_EXCEPTIONS:
        assert target in targets, target
        assert rule in {"reach", "order", "enter", "escape"}, rule


# ------------------------------------------------------------------- pages


def _page_routes() -> tuple[str, ...]:
    from tests.app.test_accessible_names import PAGE_ROUTES

    return PAGE_ROUTES


@pytest.fixture
def staged_window(project_with_host, qapp):
    """The main window of a project with a row on every page that lists rows, and jobs."""

    from tests.app.test_accessible_names import _stage_rows
    from tests.app.test_job_dashboard_tray import WINDOW_SIZE
    from tests.support.dashboard import add_dashboard_job

    window, service, document = project_with_host
    _stage_rows(window, document)
    for index, status in enumerate(("failed", "running", "completed")):
        add_dashboard_job(service, document, f"keyboard_{index}", status)
    window._refresh_jobs_table()
    window.resize(*WINDOW_SIZE)
    try:
        open_and_take_focus(window)
        QApplication.processEvents()
        yield window
    finally:
        window.hide()


def _focus_first_stop(root: QWidget) -> None:
    from mufinder.app.table_keyboard import screen_tab_order

    for stop in screen_tab_order(root):
        if stop.isVisible() and stop.isEnabled() and stop.focusPolicy() & Qt.FocusPolicy.TabFocus:
            stop.setFocus(Qt.FocusReason.OtherFocusReason)
            QApplication.processEvents()
            return


@pytest.mark.parametrize("route", _page_routes())
def test_page_meets_the_keyboard_baseline(route, staged_window) -> None:
    from tests.app.test_accessible_names import _SHOW_ROWS

    window = staged_window
    window._set_active_route(route)
    if route in _SHOW_ROWS:
        _SHOW_ROWS[route](window)
    QApplication.processEvents()
    page = window._main_stack.currentWidget()
    _focus_first_stop(page)
    faults, _stops = _walk_faults(page, window)
    _check(f"page:{route}", faults)


def test_the_job_dashboard_meets_the_keyboard_baseline(staged_window) -> None:
    window = staged_window
    window._set_active_route("host_structure")
    QApplication.processEvents()
    dock = window._job_tray.dock
    _focus_first_stop(dock)
    faults, stops = _walk_faults(dock, window)
    _check("job dashboard", faults)
    assert address(stops[0]) == address(window._job_tray.header.toggle_button)
