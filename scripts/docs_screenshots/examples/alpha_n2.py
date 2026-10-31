"""Solid α-nitrogen: the Quantum Corrections worked example.

α-N2 is a molecular crystal in space group Pa-3: the N2 molecule centres sit
on the fcc 4a sites and the N atoms at 8c (x, x, x), so each N≡N bond points
along one of the four <111> directions. The positive muon stops midway between
two neighbouring molecules, at the (0, 1/4, 1/4)-type site, and binds them into
a linear [N2-mu-N2]+ complex along [011] (Gomilšek et al. 2023).
``docs/user_guide/examples/alpha_n2.rst`` tells the story these scenes
illustrate.

The CASTEP outputs are the paper's own, kept under ``tests/fixtures/alpha_n2``
(see its ``README.md``): the relaxed muon supercell and its Gamma-point phonon
run. The host is the pristine cell of that calculation, its lattice edge half
the supercell's and its x from the molecules far from the muon.

:func:`results_project` drives a project in a ``MainWindow`` as the buttons
do: a 2x2x2 supercell (64 N and the muon), the paper's 1200 eV cutoff and
2x2x2 grid set by hand, one manual start at (0, 1/4, 1/4) relaxed as a positive
muon, one analysis workspace, clustered and pinned, and the phonon job of the
pinned site, configured with a 1e-10 eV SCF tolerance. No scene runs CASTEP: the
relaxation's ``-out.cell`` and the phonon job's ``.phonon`` and ``.castep`` are
the fixture files (see :func:`~docs_screenshots.render.synthetic_runs`). The
data set holds no log of the relaxation, so its ``.castep`` is a short one
ending on the dispersion-corrected energy of the undisplaced cell.

:func:`configured_project` stops before the phonon job is submitted, for the
configuration dialog of a row that is still editable.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    PROJECT_ROOT,
    XC_FUNCTIONAL,
    Dialog,
    Floating,
    Page,
    Scene,
    cluster_and_pin,
    focus_default_button,
    frame_viewers,
    hide_catalog_notice,
    new_project,
    opened,
    refuse_message_boxes,
    settle,
    synthetic_runs,
    when_modal,
)
from ..synthetic import castep_text

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtWidgets import QApplication, QWidget

from mufinder.domain.charge_states import POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectDocument, ProjectService

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "alpha_n2"
SEED = "aN2_mu_phonon"

# --- The host -------------------------------------------------------------------

# The calculation's cell: half the 11.3270639 Å edge of the relaxed supercell,
# which the paper held at the experimental volume (Venables and English 1974
# give a = 5.660 Å and x = 0.056).
N2_A = 5.66353193
# N at 8c (x, x, x), for the 1.1037 Å N≡N bond of the 22 molecules more than
# 4.5 Å from the muon in the relaxed supercell. Their bonds no longer lie along
# <111>: the charged complex turns every molecule of the supercell.
N2_X = 0.05626
# Each molecule's centre (4a) and the <111> direction its bond points along:
# the 8c orbit of (x, x, x) under the Pa-3 generators.
N2_MOLECULES = (
    ((0.0, 0.0, 0.0), (1, 1, 1)),
    ((0.5, 0.0, 0.5), (-1, -1, 1)),
    ((0.0, 0.5, 0.5), (-1, 1, -1)),
    ((0.5, 0.5, 0.0), (1, -1, -1)),
)
SUPERCELL_2X2X2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
SUPERCELL_LABEL = "alpha-N2 2x2x2"


def n2_host() -> Structure:
    """The eight-atom cubic cell of α-N2, each molecule's two atoms in turn."""

    positions = [
        [float(value) for value in (np.asarray(centre) + sign * N2_X * np.asarray(axis)) % 1.0]
        for centre, axis in N2_MOLECULES
        for sign in (1.0, -1.0)
    ]
    return Structure(
        label="alpha-N2",
        lattice_vectors=[[N2_A, 0.0, 0.0], [0.0, N2_A, 0.0], [0.0, 0.0, N2_A]],
        species=["N"] * len(positions),
        fractional_positions=positions,
    )


# --- The results project ------------------------------------------------------------

CUTOFF_EV = 1200.0
SUPERCELL_GRID = (2, 2, 2)
START = (0.0, 0.25, 0.25)  # midway between the molecules at (0, 0, 0) and (0, 1/2, 1/2)
# The dispersion-corrected energy of the undisplaced cell, the phonon run's
# first SCF solve.
RELAXED_ENERGY_EV = -17718.38090479
SCF_TOLERANCE_EV = 1e-10


def _fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _job_species(job_dir: Path, seed: str) -> list[str]:
    from mufinder.castep_io.cell_parser import parse_cell_file

    return list(parse_cell_file(job_dir / f"{seed}.cell").species)


def _relaxation_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The paper's relaxed supercell, and a log ending on its real energy."""

    (job_dir / f"{seed}-out.cell").write_text(_fixture_text(f"{SEED}-out.cell"), encoding="utf-8")
    text = castep_text(
        energy=RELAXED_ENERGY_EV,
        species=_job_species(job_dir, seed),
        geometry=True,
        wall_time=48210.7,
    )
    (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
    return True


def _phonon_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The paper's Gamma-point ``.phonon`` file and the log of its run."""

    for suffix in (".phonon", ".castep"):
        (job_dir / f"{seed}{suffix}").write_text(_fixture_text(f"{SEED}{suffix}"), encoding="utf-8")
    return True


def _fixture(directory: Path) -> tuple[ProjectService, ProjectDocument]:
    """The host, the supercell, the paper's cutoff and grid, and the one start."""

    host = n2_host()
    service, document = new_project(directory, "alpha-N2", host)
    service.set_recommended_xc_functional(document, XC_FUNCTIONAL)
    service.create_workflow_supercell_asset(
        document,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        host_structure_id=host.id,
        matrix=SUPERCELL_2X2X2,
        label=SUPERCELL_LABEL,
    )
    # The paper's settings, set by hand: no convergence sweep ran in this project.
    supercell_lattice = (
        np.asarray(SUPERCELL_2X2X2, dtype=float) @ np.asarray(host.lattice_vectors)
    ).tolist()
    service.set_recommended_cutoff(
        document, CUTOFF_EV, method="manual", xc_functional=XC_FUNCTIONAL
    )
    service.set_recommended_kpoint_grid(
        document,
        SUPERCELL_GRID,
        lattice_vectors=supercell_lattice,
        method="manual",
        xc_functional=XC_FUNCTIONAL,
    )
    x, y, z = START
    service.generate_candidate_sites(
        document,
        method="manual",
        parameters={"coord_system": "fractional", "x": x, "y": y, "z": z},
    )
    return service, document


def host_project(directory: Path) -> Path:
    """The α-N2 host with its supercell and its one start, before any job runs."""

    service, document = _fixture(directory)
    service.save_project(document)
    return document.project_root


def _configure_phonon_row(dialog: QWidget) -> None:
    """A 1e-10 eV SCF tolerance; everything else is inherited from the relaxation."""

    from mufinder.castep_io.types import ValueWithUnit

    dialog.set_values({"elec_energy_tol": ValueWithUnit(SCF_TOLERANCE_EV, "eV")})
    dialog._accept_if_valid()


def pinned_key(window: MainWindow) -> str:
    """The key of the Quantum Corrections page's one row, the pinned site."""

    window._quantum_corrections_controller.refresh_view()
    (row,) = window._quantum_corrections_view._source_rows
    return row.key


def _build_results(directory: Path, *, submit_phonon: bool) -> Path:
    """Build and save the results project; see the module docstring."""

    app = QApplication.instance()
    service, document = _fixture(directory)
    service.save_project(document)

    with opened(app, document.project_root) as window, refuse_message_boxes(app):
        document = window._document
        relaxations = window._muon_relaxation_view
        when_modal(
            "MuonChargeStateDialog",
            lambda dialog: (dialog._positive.setChecked(True), dialog.accept()),
        )
        window._on_create_muon_relaxation_jobs_requested(
            [str(site.id) for site in document.sites.values()]
        )
        for entry in list(relaxations._row_entries()):
            relaxations.set_row_supercell(entry.row_id, [list(row) for row in SUPERCELL_2X2X2])
        relaxations._charge_tabs.setCurrentWidget(
            relaxations._table_for_charge_state(POSITIVE_MUON_CHARGE_STATE)
        )
        relaxations._active_table().clearSelection()
        with synthetic_runs(window, _relaxation_outputs):
            relaxations._run_local_selected_rows()
        cluster_and_pin(window, POSITIVE_MUON_CHARGE_STATE, "Positive muon sites")

        # Quantum corrections: configure the pinned site's row, then run it.
        window._set_active_route("quantum_corrections")
        key = pinned_key(window)
        when_modal("QuantumCorrectionsConfigDialog", _configure_phonon_row)
        window._on_configure_quantum_corrections_source([key])
        if submit_phonon:
            with synthetic_runs(window, _phonon_outputs):
                window._quantum_corrections_controller.submit([key], target="local")
        window._save_project()
        return document.project_root


def results_project(directory: Path) -> Path:
    """The α-N2 project with the pinned site's phonon job finished; see the module docstring."""

    return _build_results(directory, submit_phonon=True)


def configured_project(directory: Path) -> Path:
    """The α-N2 project with the pinned site's phonon row configured but not submitted."""

    return _build_results(directory, submit_phonon=False)


# --- Scene steps --------------------------------------------------------------


def open_workspace(window: MainWindow) -> None:
    """Open the results project's analysis workspace in the navigator."""

    workspace_id = str(window._relaxation_analysis_workspaces()[0]["id"])
    window._set_active_route(window._relaxation_analysis_workspace_route(workspace_id))


def open_displacements(window: MainWindow) -> None:
    """Click ``Analyse Displacements`` in the analysis workspace."""

    open_workspace(window)
    window._relaxation_analysis_view._open_displacements(
        str(window._relaxation_analysis_workspaces()[0]["id"])
    )


def show_molecules(axis: str) -> Callable[[QWidget], None]:
    """A step: draw the bonds in every 3D viewer, so each N2 molecule shows as one, and view down ``axis``.

    The relaxed complex lies along [011], in the plane of a view down a.
    """

    def show(widget: QWidget) -> None:
        from mufinder.app.atomic_structure_viewer import AtomicStructureViewer

        frame_viewers(widget)
        for viewer in widget.findChildren(AtomicStructureViewer):
            if viewer.isVisible():
                viewer._show_bonds.setChecked(True)
                viewer._set_named_view(axis)
        settle(QApplication.instance())

    return show


def open_configuration(window: MainWindow) -> None:
    """Click ``Configure...`` on the pinned site's row."""

    window._on_configure_quantum_corrections_source([pinned_key(window)])


def open_analysis(window: MainWindow) -> QWidget:
    """Click ``Launch Analysis`` on the completed phonon result."""

    result = next(
        result
        for result in window._document.results.values()
        if isinstance(result.data, dict) and "muon_zpe_total_eV" in result.data
    )
    window._on_launch_quantum_phonon_analysis(str(result.id))
    return window._quantum_analysis_dialogs[result.id]


def _show_tab(analysis: QWidget, title: str) -> None:
    from PySide6.QtWidgets import QTabWidget

    tabs = analysis.findChild(QTabWidget, "quantumAnalysisTabs")
    index = next(index for index in range(tabs.count()) if tabs.tabText(index) == title)
    tabs.setCurrentIndex(index)
    settle(QApplication.instance())


def show_muon_modes(analysis: QWidget) -> None:
    """Show the L mode's vectors, with the table scrolled to the muon modes above the N2 stretch band."""

    analysis._phonon_show_vectors_checkbox.setChecked(True)
    analysis._phonon_mode_table.scrollToBottom()
    settle(QApplication.instance())
    show_molecules("a")(analysis)


def nearest_nitrogen(analysis: QWidget) -> int:
    """The one-based index of a nitrogen nearest the muon in the phonon structure."""

    lattice = analysis._phonon_lattice
    fractional = analysis._phonon_fractional_positions
    muon = analysis._muon_atom_index
    separation = (fractional - fractional[muon] + 0.5) % 1.0 - 0.5
    distances = np.linalg.norm(separation @ lattice, axis=1)
    distances[muon] = np.inf
    return int(np.argmin(distances)) + 1


def show_muon_coupling(analysis: QWidget) -> None:
    """Open ``Muon Coupling`` with Atom A the muon and Atom B its nearest nitrogen, and click ``Calculate``."""

    _show_tab(analysis, "Muon Coupling")
    combo = analysis._atom_b_combo
    combo.setCurrentIndex(combo.findData(nearest_nitrogen(analysis)))
    analysis._calculate_button.click()


def open_coupling_help(window: MainWindow) -> None:
    """Open the analysis window and click the ``?`` of ``Muon Coupling``."""

    analysis = open_analysis(window)
    _show_tab(analysis, "Muon Coupling")
    analysis._show_correlation_help()


# --- Scenes -------------------------------------------------------------------

_ANALYSIS_SIZE = (1180, 780)

SCENES: dict[str, Scene] = {
    "n2_host_structure": Scene(
        host_project,
        "host_structure",
        Page("_host_structure_view", (1100, 700), prepare=show_molecules("a")),
    ),
    "n2_relaxation_workspace": Scene(
        results_project,
        None,
        Page("_relaxation_analysis_view", (1100, 720), prepare=frame_viewers),
        setup=open_workspace,
    ),
    "n2_displacement_analysis": Scene(
        results_project,
        None,
        Dialog(
            open_displacements,
            kind="DisplacementAnalysisDialog",
            size=(1180, 720),
            prepare=show_molecules("a"),
        ),
    ),
    "n2_muon_sites_summary": Scene(
        results_project,
        "muon_sites",
        Page("_muon_sites_summary_view", (1100, 800), prepare=frame_viewers),
    ),
    "n2_quantum_corrections_page": Scene(
        results_project, "quantum_corrections", Page("_quantum_corrections_view", (1100, 520))
    ),
    "n2_quantum_configuration_dialog": Scene(
        configured_project,
        "quantum_corrections",
        Dialog(
            open_configuration,
            kind="QuantumCorrectionsConfigDialog",
            size=(900, 580),
            prepare=hide_catalog_notice,
        ),
    ),
    "n2_phonon_modes": Scene(
        results_project,
        "quantum_corrections",
        Floating(open_analysis, size=_ANALYSIS_SIZE, prepare=show_muon_modes),
    ),
    "n2_muon_coupling": Scene(
        results_project,
        "quantum_corrections",
        Floating(open_analysis, size=(1180, 560), prepare=show_muon_coupling),
    ),
    "n2_coupling_help": Scene(
        results_project,
        "quantum_corrections",
        Dialog(
            open_coupling_help,
            kind="QuantumAnalysisHelpDialog",
            size=(660, 640),
            prepare=focus_default_button,
        ),
    ),
}
