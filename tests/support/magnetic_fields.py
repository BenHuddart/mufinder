"""Magnetic models, structures and form helpers shared by the Magnetic Fields tests."""

from __future__ import annotations

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.demagnetization import DemagnetizingSample
from mufinder.science.magnetic_structure import (
    AtomSelector,
    ComplexVector,
    FourierTerm,
    MagneticComponent,
    MagneticStructureModel,
    build_helix_component,
)
from mufinder.services.magnetic_field_sites import MagneticMuonSiteOption


def coo_structure() -> Structure:
    return Structure(
        label="CoO",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Co", "O", "Co"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )


def z_moment_component(component_id: str, label: str, atom_index: int) -> MagneticComponent:
    return MagneticComponent(
        component_id=component_id,
        label=label,
        kind="test",
        selector=AtomSelector.indices([atom_index]),
        terms=(
            FourierTerm(
                q_coeff=(0.0, 0.0, 0.0),
                coefficient=ComplexVector.from_vector([0.0, 0.0, 1.0]),
            ),
        ),
    )


def set_direction_inputs_text(inputs, values: tuple[float, float, float]) -> None:
    for edit, value in zip(inputs, values, strict=True):
        edit.setText(f"{value:g}")


def custom_muon_site(
    position: tuple[float, float, float],
) -> MagneticMuonSiteOption:
    return MagneticMuonSiteOption(
        source="custom",
        label="Custom site",
        fractional_position=position,
        cartesian_position=tuple(3.0 * value for value in position),
    )


def photo_plate_sample(**fields):
    x = (np.arange(40) + 0.5) / 40 * 2.0 - 1.0
    outline = x[:, None] ** 2 + x[None, :] ** 2 <= 1.0
    return DemagnetizingSample.image_plate(
        outline, (5.0, 5.0), 0.2, fourier_grid=32, padding=2.0, **fields
    )


def two_row_helix_model() -> MagneticStructureModel:
    helix = build_helix_component(
        label="Fe helix",
        selector=AtomSelector.species("Fe"),
        q_coeff=(1.0, 0.0, 0.0),
        real_axis_vector=(1.0, 0.0, 0.0),
        imaginary_axis_vector=(0.0, 1.0, 0.0),
    )
    return MagneticStructureModel(
        q_basis_rlu=((0.0, 0.0, 0.25), (0.5, 0.0, 0.0), (0.0, 0.0, 0.0)),
        components=(helix,),
    )


def select_combo_data(combo, data) -> None:
    index = combo.findData(data)
    assert index >= 0
    combo.setCurrentIndex(index)
