"""Local field at the octahedral interstitial site of bcc iron.

Needs the ``scientific-backends`` extra (muesr and muLFC).

    python docs/api/examples/dipolar_field.py
"""

from mufinder.domain.models import Structure
from mufinder.science.magnetic_fields import MagneticFieldOptions, calculate_commensurate_fields
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
)

# The conventional bcc cell of iron: two Fe in a 2.868 Angstrom cube.
host = Structure(
    label="Fe",
    lattice_vectors=[[2.868, 0.0, 0.0], [0.0, 2.868, 0.0], [0.0, 0.0, 2.868]],
    species=["Fe", "Fe"],
    fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# A 2.2 muB moment along c on every atom, with q = 0 (a ferromagnet).
model = MagneticStructureModel(
    components=(
        build_collinear_component(
            label="Ferromagnet",
            selector=AtomSelector.all(),
            q_coeff=(0.0, 0.0, 0.0),
            moment=(0.0, 0.0, 2.2),
        ),
    ),
)

# The octahedral site (1/2, 0, 0), summed over an 80 x 80 x 80 block of cells.
options = MagneticFieldOptions(supercell=(80, 80, 80))
(result,) = calculate_commensurate_fields(host, model, [(0.5, 0.0, 0.0)], options=options)

print(f"Muon site: {result.muon_position}")
print(f"Dipolar Bz: {result.dipolar[2]:.4f} T")
print(f"Lorentz Bz: {result.lorentz[2]:.4f} T")
print(f"Total Bz: {result.total[2]:.4f} T")
print(f"|B|: {result.magnitude:.4f} T")
