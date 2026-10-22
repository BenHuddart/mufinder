"""Body-centred cubic iron: the Magnetic Fields worked example.

bcc Fe (Im-3m, a = 2.868 Å) is a ferromagnet with 2.2 μB per atom, here along
c. The positive muon stops in an interstitial site of the bcc lattice, either
the octahedral site (1/2, 0, 0) (Wyckoff 6b) or the tetrahedral site
(1/2, 1/4, 0) (12d). The scenes calculate the field at each with the real
``muesr`` calculation, with every equivalent site listed: the contact field is
set to the −1.11 T measured at low temperature (Nishida et al. 1977). The
project is a host structure and a Magnetic Fields form: no CASTEP job is
involved. ``docs/user_guide/examples/fe.rst`` tells the story these scenes
illustrate.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import (
    Dialog,
    Page,
    ProjectBuilder,
    Scene,
    Window,
    calculate_field,
    current_page,
    focus_default_button,
    new_project,
    save_magnetic_workspace,
    settle,
    show_host_with_spins,
    when_modal,
)

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication, QGroupBox, QWidget

from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import AtomSelector, MagneticStructureModel, build_collinear_component

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow

# --- The host -------------------------------------------------------------------

FE_A = 2.868  # Å
FE_MOMENT_MU_B = 2.2  # per Fe, along c
FE_SPIN = 2.2  # ħ/2 per Fe: one unpaired electron per μB
OCTAHEDRAL_SITE = (0.5, 0.0, 0.0)  # 6b
TETRAHEDRAL_SITE = (0.5, 0.25, 0.0)  # 12d
# The contact field measured at low temperature is -1.11 T (Nishida et al.
# 1977). The two nearest Fe each carry 2.2 μB, so the contact term is the
# coupling times 2.2 μB: -1.11 T / 2.2 μB.
CONTACT_COUPLING_T_PER_MU_B = "-0.5045455"
# An 80 x 80 x 80 lattice sum keeps the cubic symmetry of the Lorentz sphere
# exactly, so the components that vanish by symmetry print as 0; the default
# 100 x 100 x 100 sum leaves 1e-7 T of numerical noise in them.
SUPERCELL = [80, 80, 80]
MEASURED_MAGNETISATION_MU_B = 2.0  # μB per Fe: a sample short of saturation, illustrative


def fe_host() -> Structure:
    """The conventional bcc cell with a ferromagnetic spin on each Fe."""

    return Structure(
        label="Fe",
        lattice_vectors=[[FE_A, 0.0, 0.0], [0.0, FE_A, 0.0], [0.0, 0.0, FE_A]],
        species=["Fe", "Fe"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        properties={"atomic_spins": [FE_SPIN, FE_SPIN]},
    )


def fe_magnetic_model() -> MagneticStructureModel:
    """The ferromagnet: 2.2 μB along c on every Fe."""

    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Fe moment",
                selector=AtomSelector.all(),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, FE_MOMENT_MU_B),
            ),
        ),
    )


def custom_site(fractional: tuple[float, float, float]) -> dict[str, object]:
    return {
        "source": "custom",
        "label": "Custom site",
        "fractional_position": list(fractional),
        "cartesian_position": [FE_A * value for value in fractional],
    }


def fe_project(name: str, form: dict[str, object]) -> ProjectBuilder:
    """A builder for the Fe host, its ferromagnetic model and a Magnetic Fields ``form``.

    ``form`` extends the common one: the contact coupling, every equivalent
    site and a spherical sample.
    """

    settings: dict[str, object] = {
        "demag_shape": "sphere",
        "include_equivalent_sites": True,
        "supercell": SUPERCELL,
        "hyperfine_coupling": CONTACT_COUPLING_T_PER_MU_B,
        "hyperfine_coupling_unit": "T_per_muB",
        **form,
    }

    def build(directory: Path) -> Path:
        service, document = new_project(directory, name, fe_host())
        document.project.magnetic_structure = fe_magnetic_model().to_json()
        document.ui_state["magnetic_field_form"] = settings
        service.save_project(document)
        return document.project_root

    return build


# --- Scene steps --------------------------------------------------------------


def calculation_card(window: MainWindow) -> QWidget:
    """The Magnetic Fields page's ``Calculation`` card."""

    for group in window._magnetic_field_view.findChildren(QGroupBox):
        if group.title() == "Calculation":
            return group
    raise RuntimeError("The Magnetic Fields page has no Calculation card")


def show_every_result_row(page: QWidget) -> None:
    """Make the results table tall enough to list every row, so the grab does not scroll it."""

    table = page._result_table
    rows = sum(table.rowHeight(row) for row in range(table.rowCount()))
    table.setMinimumHeight(table.horizontalHeader().height() + rows + 2 * table.frameWidth() + 2)


def calculate_and_list(window: MainWindow) -> None:
    """Calculate the field and show every result row."""

    page = window._magnetic_field_view
    calculate_field(page)
    show_every_result_row(page)


def select_parallel_site(page: QWidget) -> None:
    """Select the first result row whose dipolar field points along the moment."""

    table = page._result_table
    for row in range(table.rowCount()):
        if float(table.item(row, 3).text()) > 0:
            table.selectRow(row)
            return
    raise RuntimeError("No result row has a field along the moment")


def select_infinite_plate(window: MainWindow) -> None:
    """Calculate with the sample an infinite plate normal to c, along the moment."""

    page = window._magnetic_field_view
    page._plate_radio.setChecked(True)
    calculate_and_list(window)


def save_both_workspaces(window: MainWindow) -> None:
    """Save the octahedral field as a workspace, move to the tetrahedral site and save that too."""

    page = window._magnetic_field_view
    save_magnetic_workspace("Octahedral site")(window)
    settle(QApplication.instance())

    def choose_tetrahedral(dialog: QWidget) -> None:
        for edit, value in zip((dialog._custom_x, dialog._custom_y, dialog._custom_z), TETRAHEDRAL_SITE, strict=True):
            edit.setText(f"{value:g}")
        dialog.accept()

    when_modal("MuonSiteSelectionDialog", choose_tetrahedral)
    page._select_site_button.click()
    save_magnetic_workspace("Tetrahedral site")(window)
    # Show the octahedral workspace, whose six rows fit the window.
    workspace_id = window._document.ui_state["magnetic_field_workspaces"][0]["id"]
    window._set_active_route(window._magnetic_field_workspace_route(workspace_id))
    show_every_result_row(current_page(window))


# --- Scenes -------------------------------------------------------------------

_OCTAHEDRAL = {"selected_muon_site": custom_site(OCTAHEDRAL_SITE)}
_TETRAHEDRAL = {"selected_muon_site": custom_site(TETRAHEDRAL_SITE)}
octahedral_project = fe_project("Iron", _OCTAHEDRAL)

SCENES: dict[str, Scene] = {
    "fe_host_structure": Scene(
        octahedral_project, "host_structure", Page("_host_structure_view", (1100, 760), prepare=show_host_with_spins)
    ),
    "fe_magnetic_fields_page": Scene(
        octahedral_project,
        "magnetic_fields",
        Page("_magnetic_field_view", (1100, 1260)),
        setup=calculate_and_list,
    ),
    "fe_tetrahedral_sites": Scene(
        fe_project("Iron tetrahedral", _TETRAHEDRAL),
        "magnetic_fields",
        Page("_magnetic_field_view", (1100, 1250)),
        setup=calculate_and_list,
    ),
    "fe_equivalent_sites": Scene(
        fe_project("Iron without demagnetizing field", {**_OCTAHEDRAL, "demag_shape": "none"}),
        "magnetic_fields",
        Page(calculation_card, (1100, 900)),
        setup=lambda window: (calculate_and_list(window), select_parallel_site(window._magnetic_field_view)),
    ),
    "fe_demagnetizing_shapes": Scene(
        octahedral_project,
        "magnetic_fields",
        Page(calculation_card, (1100, 900)),
        setup=select_infinite_plate,
    ),
    "fe_measured_magnetisation": Scene(
        fe_project(
            "Iron measured magnetisation",
            {
                **_OCTAHEDRAL,
                "use_measured_magnetization": True,
                "measured_magnetization": ["0", "0", str(MEASURED_MAGNETISATION_MU_B)],
            },
        ),
        "magnetic_fields",
        Page(calculation_card, (1100, 900)),
        setup=calculate_and_list,
    ),
    "fe_select_muon_site": Scene(
        octahedral_project,
        "magnetic_fields",
        Dialog(
            lambda window: window._magnetic_field_view._select_site_button.click(),
            kind="MuonSiteSelectionDialog",
            size=(760, 230),
            prepare=focus_default_button,
        ),
    ),
    "fe_magnetic_workspaces": Scene(octahedral_project, "magnetic_fields", Window((1280, 1140)), setup=save_both_workspaces),
}
