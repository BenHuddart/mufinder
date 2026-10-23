"""Solid α-nitrogen: the Quantum Corrections worked example.

α-N2 is a molecular crystal (Pa-3, a = 5.660 Å; Venables and English 1974):
the N2 molecule centres sit on the fcc 4a sites and the N atoms at 8c
(x, x, x) with x = 0.056, so each N≡N bond is 1.098 Å long and points along
one of the four <111> directions. The positive muon stops midway between two
neighbouring molecules, at the (0, 1/4, 1/4)-type site, and binds them into a
linear [N2-mu-N2]+ complex along [011] (Gomilšek et al. 2023).
``docs/user_guide/examples/alpha_n2.rst`` tells the story these scenes
illustrate.

:func:`results_project` drives a project in a ``MainWindow`` as the buttons
do: a 2x2x2 supercell (64 N and the muon), an adopted 1200 eV cutoff and a
unit-cell grid that becomes 2x2x2 in the supercell, seeded random positive-muon
starts relaxed in the supercell, one analysis workspace, clustered and pinned,
and the phonon job of the pinned site, configured with ultrasoft
pseudopotentials, ``FINITEDISPLACEMENT`` and a 1e-10 eV SCF tolerance. No
scene runs CASTEP: each job's outputs are synthetic (see
:func:`~docs_screenshots.render.synthetic_runs`):

- every start relaxes to the nearest (0, 1/4, 1/4)-type site, where the two
  molecules either side turn to point at the muon, 1.29 Å from their near
  nitrogens, with their bonds stretched to 1.115 Å, and the molecules around
  them move 0.03 Å towards it (``COMPLEX``);
- the phonon run embeds the five-atom chain spring model of
  ``tests/support/phonon_fixtures.py``, tuned to the paper's harmonic numbers,
  in the relaxed supercell, with every other molecule vibrating on its own
  (``N2_BANDS_CM``).

:func:`configured_project` stops before the phonon job is submitted, for the
configuration dialog of a row that is still editable.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    XC_FUNCTIONAL,
    Dialog,
    Floating,
    Outputs,
    Page,
    Scene,
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
from ..synthetic import bridged_diatomic_cell, bridged_diatomic_phonon_text, castep_text, cell_text

from collections.abc import Callable
from functools import cache
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

# --- The host -------------------------------------------------------------------

N2_A = 5.660  # Å, Venables and English 1974
N2_X = 0.056  # N at 8c (x, x, x): an N≡N bond of 1.098 Å
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
UNIT_CELL_GRID = (4, 4, 4)  # 2x2x2 in the supercell
RANDOM_STARTS = 6
RANDOM_SEED = "205"
SITE = (0.0, 0.25, 0.25)  # midway between the molecules at (0, 0, 0) and (0, 1/2, 1/2)
# The relaxed complex: the muon to each near nitrogen, the stretched N≡N bond
# of the two complex molecules, and how far (Å) every other molecule whose
# centre lies within NEIGHBOUR_RADIUS of the muon moves towards it.
COMPLEX = {
    "muon_bond": 1.29,
    "molecule_bond": 1.115,
    "neighbour_shift": 0.03,
    "neighbour_radius": 4.5,
}
POSITIVE_ENERGY_EV = -17318.4271  # the relaxed supercell
# The phonon bands (cm^-1) of the molecules outside the complex: the N≡N
# stretch, the librations and the translations.
N2_BANDS_CM = {
    "stretch_band_cm": (2330.0, 2350.0),
    "libration_band_cm": (30.0, 70.0),
    "translation_band_cm": (15.0, 95.0),
}
SCF_TOLERANCE_EV = 1e-10


@cache
def _site_orbit() -> tuple[tuple[float, float, float], ...]:
    """The 24 positions equivalent to ``SITE`` in the host cell."""

    from mufinder.science.spglib_compat import spglib_cell, symmetry_operations, symmetry_orbit

    host = n2_host()
    operations = symmetry_operations(spglib_cell(host), symprec=1e-5)
    orbit = symmetry_orbit(SITE, operations, lattice_vectors=host.lattice_vectors, tolerance=1e-4)
    return tuple(tuple(float(value) for value in image) for image in orbit)


def nearest_site(start: Any) -> np.ndarray:
    """The member of ``SITE``'s orbit nearest ``start``, both in host fractional coordinates."""

    lattice = np.asarray(n2_host().lattice_vectors, dtype=float)
    shifts = [(np.asarray(image) - np.asarray(start) + 0.5) % 1.0 - 0.5 for image in _site_orbit()]
    shift = min(shifts, key=lambda value: float(np.linalg.norm(value @ lattice)))
    return np.asarray(start, dtype=float) + shift


def _job_cell(job_dir: Path, seed: str):
    from mufinder.castep_io.cell_parser import parse_cell_file

    return parse_cell_file(job_dir / f"{seed}.cell")


def _muon_outputs(window: MainWindow) -> Outputs:
    """The relaxed complex and a CASTEP output for each start; later rows sit a little higher."""

    def write(job: Any, job_dir: Path, seed: str) -> bool:
        from types import SimpleNamespace

        _definition, row = window._muon_relaxation_definition_and_row_for_job(job)
        start = np.asarray(row.initial_fractional_position, dtype=float)
        cell = _job_cell(job_dir, seed)
        fractional = np.asarray(cell.fractional_positions, dtype=float)
        muon = list(cell.species).index("H:mu")
        fractional[muon] += (nearest_site(start) - start) / 2.0
        moved = SimpleNamespace(
            lattice_vectors=cell.lattice_vectors,
            species=list(cell.species),
            fractional_positions=fractional % 1.0,
        )
        relaxed = bridged_diatomic_cell(moved, **COMPLEX)
        (job_dir / f"{seed}-out.cell").write_text(
            cell_text(cell.lattice_vectors, cell.species, relaxed), encoding="utf-8"
        )
        number = int("".join(character for character in row.label if character.isdigit()) or 1)
        energy = POSITIVE_ENERGY_EV + 0.0005 * (number - 1)
        text = castep_text(
            energy=energy,
            species=list(cell.species),
            geometry=True,
            wall_time=31204.6 + 1213.0 * number,
        )
        (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
        return True

    return write


def _phonon_outputs(_job: Any, job_dir: Path, seed: str) -> bool:
    """The phonon run of the pinned site: the chain model of the complex in the relaxed supercell."""

    cell = _job_cell(job_dir, seed)
    (job_dir / f"{seed}.phonon").write_text(
        bridged_diatomic_phonon_text(cell, **N2_BANDS_CM), encoding="utf-8"
    )
    text = castep_text(
        energy=POSITIVE_ENERGY_EV - 0.0003,
        species=list(cell.species),
        geometry=False,
        wall_time=172418.9,
    )
    (job_dir / f"{seed}.castep").write_text(text, encoding="utf-8")
    return True


def _fixture(directory: Path) -> tuple[ProjectService, ProjectDocument]:
    """The host, the supercell, the adopted cutoff and grid, and the random starts."""

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
    document.project.supercell_matrix = [list(row) for row in SUPERCELL_2X2X2]
    criteria = {
        "method": "delta-threshold",
        "cell_id": str(host.id),
        "xc_functional": XC_FUNCTIONAL,
    }
    service.set_recommended_cutoff(document, CUTOFF_EV, **criteria)
    service.set_recommended_kpoint_grid(
        document, UNIT_CELL_GRID, lattice_vectors=host.lattice_vectors, **criteria
    )
    service.generate_candidate_sites(
        document,
        method="random",
        parameters={
            "selection_mode": "count",
            "count": RANDOM_STARTS,
            "min_muon_atom": 1.0,
            "max_attempts": 50000,
            "seed": RANDOM_SEED,
            "include_symmetry_equivalents": False,
        },
    )
    return service, document


def host_project(directory: Path) -> Path:
    """The α-N2 host with its supercell and random starts, before any job runs."""

    service, document = _fixture(directory)
    service.save_project(document)
    return document.project_root


def _configure_phonon_row(dialog: QWidget) -> None:
    """Ultrasoft pseudopotentials and a 1e-10 eV SCF tolerance; the phonon method stays FINITEDISPLACEMENT."""

    from mufinder.castep_io.types import ValueWithUnit

    dialog.set_pseudopotential_family("ultrasoft")
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

    # The relaxations run in the 2x2x2 supercell, so configuring them rescales
    # the adopted unit-cell k-point grid and says so.
    with (
        opened(app, document.project_root) as window,
        refuse_message_boxes(app, expected=("K-point grids adjusted",)),
    ):
        document = window._document
        relaxations = window._muon_relaxation_view
        when_modal(
            "MuonChargeStateDialog",
            lambda dialog: (dialog._positive.setChecked(True), dialog.accept()),
        )
        sites = sorted(
            document.sites.values(), key=lambda site: window._natural_label_sort_key(site.label)
        )
        window._on_create_muon_relaxation_jobs_requested([str(site.id) for site in sites])
        for entry in list(relaxations._row_entries()):
            relaxations.set_row_supercell(entry.row_id, [list(row) for row in SUPERCELL_2X2X2])
        relaxations._charge_tabs.setCurrentWidget(
            relaxations._table_for_charge_state(POSITIVE_MUON_CHARGE_STATE)
        )
        relaxations._active_table().clearSelection()
        with synthetic_runs(window, _muon_outputs(window)):
            relaxations._run_local_selected_rows()

        # One analysis workspace, clustered and pinned.
        window._set_active_route("muon_relaxations")
        relaxations._active_table().selectAll()
        when_modal(
            "QInputDialog",
            lambda dialog: (dialog.setTextValue("Positive muon sites"), dialog.accept()),
        )
        relaxations._create_analysis_button.click()
        workspace_id = str(window._relaxation_analysis_workspaces()[-1]["id"])
        when_modal("RelaxationClusteringDialog", lambda dialog: dialog._apply_button.click())
        window._relaxation_analysis_view._open_clustering(workspace_id)
        window._on_muon_sites_pin_requested(workspace_id)

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

    The pinned site's representative relaxation has its complex along [110],
    so the complex lies in the plane of a view down c.
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
    show_molecules("c")(analysis)


def show_longitudinal_well(analysis: QWidget) -> None:
    """Open ``Mode wells and widths`` with the L well, the first selected mode, alone."""

    _show_tab(analysis, "Mode wells and widths")
    analysis._single_particle_table.selectRow(0)


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
            prepare=show_molecules("c"),
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
    "n2_mode_wells": Scene(
        results_project,
        "quantum_corrections",
        Floating(open_analysis, size=_ANALYSIS_SIZE, prepare=show_longitudinal_well),
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
