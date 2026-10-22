"""Simple cubic helices: the Magnetic Fields scenes of the theory page's helix examples.

The models are the worked examples in ``docs/user_guide/magnetic_fields_theory.rst``:
one 1 μB ion in a simple cubic cell with a = 3 Å, as a commensurate or an
incommensurate helix. Each scene's builder from :func:`magnetic_project` saves
the host with its model and a Magnetic Fields form; the scenes run the real
``muesr`` field calculation.
"""

from __future__ import annotations

# The render module configures Qt and VTK, so it is imported before them.
from ..render import Dialog, ProjectBuilder, Scene, calculate_field, new_project

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QWidget

from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
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


def show_magnetic_cell(dialog: QWidget) -> None:
    """One magnetic cell of the q = 1/4 helix is four host cells along c."""

    dialog._repeat_c.setValue(4)


def open_distribution_plot(window: MainWindow) -> None:
    page = window._magnetic_field_view
    calculate_field(page)
    page._open_distribution_button.click()


# --- Scenes -------------------------------------------------------------------

SCENES: dict[str, Scene] = {
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
}
