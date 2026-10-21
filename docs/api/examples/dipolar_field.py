"""Local field at the face centre of a simple cubic ferromagnet.

Needs the ``scientific-backends`` extra (muesr and muLFC).

    python docs/api/examples/dipolar_field.py
"""

from mufinder.domain.models import Structure
from mufinder.science.magnetic_fields import calculate_commensurate_fields
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
)

# One magnetic ion in a 3 Angstrom cube.
host = Structure(
    label="Fe",
    lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
    species=["Fe"],
    fractional_positions=[[0.0, 0.0, 0.0]],
)

# A 1 muB moment along c on every atom, with q = 0 (a ferromagnet).
model = MagneticStructureModel(
    components=(
        build_collinear_component(
            label="Ferromagnet",
            selector=AtomSelector.all(),
            q_coeff=(0.0, 0.0, 0.0),
            moment=(0.0, 0.0, 1.0),
        ),
    ),
)

(result,) = calculate_commensurate_fields(host, model, [(0.5, 0.5, 0.0)])

print(f"Muon site: {result.muon_position}")
print(f"Dipolar Bz: {result.dipolar[2]:.4f} T")
print(f"Lorentz Bz: {result.lorentz[2]:.4f} T")
print(f"Total Bz: {result.total[2]:.4f} T")
print(f"|B|: {result.magnitude:.4f} T")
