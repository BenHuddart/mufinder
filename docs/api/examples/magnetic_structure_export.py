"""A sublattice antiferromagnet from one component, exported to ASE.

Needs only the core install.

    python docs/api/examples/magnetic_structure_export.py
"""

from dataclasses import replace

from mufinder.domain.models import Structure
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
    magnetic_structure_to_ase_atoms,
    wavevector_rlu_from_inverse_angstrom,
)

# The conventional bcc cell of iron: two Fe in a 2.868 Angstrom cube.
host = Structure(
    label="Fe",
    lattice_vectors=[[2.868, 0.0, 0.0], [0.0, 2.868, 0.0], [0.0, 0.0, 2.868]],
    species=["Fe", "Fe"],
    fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)

# One q = 0 component on both atoms; half a turn on atom 1 reverses its moment.
antiferromagnet = replace(
    build_collinear_component(
        label="Antiferromagnet",
        selector=AtomSelector.all(),
        q_coeff=(0.0, 0.0, 0.0),
        moment=(0.0, 0.0, 2.2),
    ),
    atom_phase_turns={1: 0.5},
)
model = MagneticStructureModel(components=(antiferromagnet,))

# A 2 x 2 x 2 supercell as ASE Atoms, moments in initial_magnetic_moments.
atoms = magnetic_structure_to_ase_atoms(host, model, (2, 2, 2))
moments_z = atoms.get_initial_magnetic_moments()[:, 2]
print(f"Atoms: {len(atoms)} {atoms.get_chemical_formula()}")
print(f"Moments along c: {sorted(set(moments_z.round(3).tolist()))} muB")

# The MnSi helix: |k| = 0.035 / Angstrom along [111] of a 4.558 Angstrom cube.
mnsi = [[4.558, 0.0, 0.0], [0.0, 4.558, 0.0], [0.0, 0.0, 4.558]]
q = wavevector_rlu_from_inverse_angstrom(mnsi, 0.035, (1.0, 1.0, 1.0))
print("MnSi q (r.l.u.): (" + ", ".join(f"{value:.4f}" for value in q) + ")")
