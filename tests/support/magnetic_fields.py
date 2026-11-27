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


def host_with_spin_record(
    species: tuple[str, ...] = ("F", "Co"),
    spins: tuple[float, ...] = (0.0, 3.0),
    *,
    atom_index: bool = False,
    mcif: bool = False,
) -> Structure:
    """A host whose spins are stored as the Host Structure page's moment record.

    Each non-zero spin is a moment along z labelled by species and one-based atom
    number (``Co2``). Without ``atom_index`` the record is the one saved before
    moments carried their atom, which placed ``Co2`` of ``F, Co`` on the F atom.
    ``mcif`` marks the record as an MCIF import's.
    """

    moments = [
        {
            "label": f"{name}{index + 1}",
            **({"atom_index": index} if atom_index else {}),
            "vector": [0.0, 0.0, spin],
            "coordinate_system": "cartesian",
        }
        for index, (name, spin) in enumerate(zip(species, spins))
        if spin
    ]
    record: dict[str, object] = {"propagation_vector": [0.0, 0.0, 0.0], "moments": moments}
    if mcif:
        record["source"] = "mcif"
    return Structure(
        label="Spin host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=list(species),
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5]][: len(species)],
        properties={
            "atomic_spins": list(spins),
            "propagation_vector": [0.0, 0.0, 0.0],
            "magnetic_structure": record,
        },
    )


def spin_host_project(tmp_path, host: Structure | None = None):
    """A new project whose active host is ``host`` (by default :func:`host_with_spin_record`).

    The project's ``magnetic_structure`` is the host's spin record, as a typed spin leaves it,
    so the Magnetic Fields page builds its model from that record. Returns the service and
    the document.
    """

    from mufinder.services.project_service import ProjectService

    service = ProjectService()
    document = service.create_project(tmp_path / "spins.mufinder", "Spins")
    host = host_with_spin_record() if host is None else host
    service.add_structure(document, host)
    document.project.magnetic_structure = host.properties["magnetic_structure"]
    return service, document


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
    return DemagnetizingSample.image_plate(outline, (5.0, 5.0), 0.2, **fields)


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


def collinear_waves_model(*q_vectors: tuple[float, float, float]) -> MagneticStructureModel:
    """One collinear z component on every atom per wavevector, each on its own q-basis row."""

    rows = [tuple(q) for q in q_vectors] + [(0.0, 0.0, 0.0)] * (3 - len(q_vectors))
    return MagneticStructureModel(
        q_basis_rlu=tuple(rows),
        components=tuple(
            MagneticComponent(
                component_id=f"wave {index + 1}",
                label=f"Wave {index + 1}",
                kind="custom",
                selector=AtomSelector.all(),
                terms=(
                    FourierTerm(
                        tuple(1.0 if axis == index else 0.0 for axis in range(3)),
                        ComplexVector.from_vector([0.0, 0.0, 1.0]),
                    ),
                ),
            )
            for index in range(len(q_vectors))
        ),
    )
