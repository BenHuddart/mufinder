"""Parse a CASTEP geometry optimisation and print its summary.

Run from the repository root so the fixture path resolves:

    python docs/api/examples/parse_castep_output.py
"""

from pathlib import Path

from mufinder.castep_io.output_parser import parse_castep_output

result = parse_castep_output(Path("tests/fixtures/geometry/Si2_bfgs.castep"))

print(f"Completed: {result.completed}")
print(f"Final energy: {result.total_energy:.6f} eV")
print(f"Final enthalpy: {result.enthalpy:.6f} eV")
print(f"Max force: {result.max_force:.5f} eV/A")
print(f"Max stress: {result.max_stress:.6f} GPa")
print(f"Geometry converged: {result.geom_converged}")
print("Forces (eV/A):")
for species, fx, fy, fz in result.forces:
    print(f"  {species:<3} {fx:10.5f} {fy:10.5f} {fz:10.5f}")
