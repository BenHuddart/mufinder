"""Simple cubic magnets: the Magnetic Fields scenes.

The models are the worked examples in ``docs/user_guide/magnetic_fields.rst``:
one 1 μB ion in a simple cubic cell with a = 3 Å, as a ferromagnet or a helix.
Each scene's builder from :func:`magnetic_project` saves the host with its
model and a Magnetic Fields form; the scenes run the real ``muesr`` field
calculation.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import Dialog, Page, ProjectBuilder, Scene, Window, current_page, new_project, when_modal

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtWidgets import QWidget

from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
    build_helix_component,
)

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow


def simple_cubic_host() -> Structure:
    return Structure(
        label="Fe",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def ferromagnet() -> MagneticStructureModel:
    return MagneticStructureModel(
        components=(
            build_collinear_component(
                label="Fe moment",
                selector=AtomSelector.all(),
                q_coeff=(0.0, 0.0, 0.0),
                moment=(0.0, 0.0, 1.0),
            ),
        ),
    )


def helix(q: float) -> MagneticStructureModel:
    return MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, q), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(
            build_helix_component(
                label="Fe helix",
                selector=AtomSelector.all(),
                q_coeff=(1.0, 0.0, 0.0),
                real_axis_vector=(1.0, 0.0, 0.0),
                imaginary_axis_vector=(0.0, 1.0, 0.0),
            ),
        ),
    )


def custom_site(fractional: tuple[float, float, float]) -> dict[str, object]:
    return {
        "source": "custom",
        "label": "Custom site",
        "fractional_position": list(fractional),
        "cartesian_position": [3.0 * value for value in fractional],
    }


def magnetic_project(name: str, model: MagneticStructureModel, form: dict[str, object]) -> ProjectBuilder:
    """A builder for the simple cubic host with ``model`` and a Magnetic Fields ``form``."""

    def build(directory: Path) -> Path:
        service, document = new_project(directory, name, simple_cubic_host())
        document.project.magnetic_structure = model.to_json()
        document.ui_state["magnetic_field_form"] = form
        service.save_project(document)
        return document.project_root

    return build


# --- Scene steps --------------------------------------------------------------


def calculate(page: Any) -> None:
    """Click ``Calculate Dipolar Field`` and select the first result row."""

    page._calculate_button.click()
    if page._result_table.rowCount() == 0:
        # A screenshot of a failed calculation must not reach the docs.
        raise RuntimeError(f"The field calculation gave no result: {page._calculation_status.text()}")
    page._result_table.selectRow(0)


def show_magnetic_cell(dialog: QWidget) -> None:
    """One magnetic cell of the q = 1/4 helix is four host cells along c."""

    dialog._repeat_c.setValue(4)


def open_distribution_plot(window: MainWindow) -> None:
    page = window._magnetic_field_view
    calculate(page)
    page._open_distribution_button.click()


def save_magnetic_workspace(window: MainWindow) -> None:
    """Calculate the field, then click ``Save Workspace`` and name it."""

    page = window._magnetic_field_view
    calculate(page)
    when_modal("QInputDialog", lambda dialog: (dialog.setTextValue("Ferromagnet"), dialog.accept()))
    page._pin_button.click()


# --- Scenes -------------------------------------------------------------------

_FERROMAGNET_FORM = {"demag_shape": "sphere", "selected_muon_site": custom_site((0.5, 0.5, 0.0))}
ferromagnet_project = magnetic_project("Simple cubic ferromagnet", ferromagnet(), _FERROMAGNET_FORM)

SCENES: dict[str, Scene] = {
    "magnetic_fields_page": Scene(
        ferromagnet_project,
        "magnetic_fields",
        Window(),
        setup=lambda window: calculate(window._magnetic_field_view),
    ),
    "magnetic_structure_builder": Scene(
        magnetic_project("Commensurate helix", helix(0.25), {"selected_muon_site": custom_site((0.5, 0.0, 0.0))}),
        "magnetic_fields",
        Dialog(
            lambda window: window._magnetic_field_view._edit_structure_button.click(),
            size=(1180, 900),
            prepare=show_magnetic_cell,
        ),
    ),
    "magnetic_field_distribution": Scene(
        magnetic_project(
            "Incommensurate helix",
            helix(0.2861),
            {"demag_shape": "none", "selected_muon_site": custom_site((0.5, 0.0, 0.0))},
        ),
        "magnetic_fields",
        Dialog(open_distribution_plot),
    ),
    "magnetic_field_workspace": Scene(
        ferromagnet_project,
        "magnetic_fields",
        Page(current_page, (1100, 860)),
        setup=save_magnetic_workspace,
    ),
}
