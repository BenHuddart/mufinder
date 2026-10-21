"""Generate random candidate muon sites in a silicon host.

    python docs/api/examples/generate_muon_sites.py
"""

from pathlib import Path
import tempfile

from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectService

# Conventional diamond-cubic silicon, a = 5.431 Angstrom, eight atoms.
A = 5.431
silicon = Structure(
    label="Si",
    lattice_vectors=[[A, 0.0, 0.0], [0.0, A, 0.0], [0.0, 0.0, A]],
    species=["Si"] * 8,
    fractional_positions=[
        [0.00, 0.00, 0.00],
        [0.50, 0.50, 0.00],
        [0.50, 0.00, 0.50],
        [0.00, 0.50, 0.50],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ],
)

service = ProjectService()
with tempfile.TemporaryDirectory() as project_root:
    # A throwaway project: the first structure added becomes its host.
    document = service.create_project(Path(project_root), "Si")
    service.add_structure(document, silicon)

    sites = service.generate_candidate_sites(
        document,
        method="random",
        parameters={
            "selection_mode": "count",
            "count": 5,
            "min_muon_atom": 1.0,
            "max_attempts": 20000,
            "include_symmetry_equivalents": True,
            "seed": 42,
        },
    )

print(f"Generated {len(sites)} sites")
for site in sites:
    x, y, z = site.initial_fractional_position
    print(f"  {site.label:<14} {x:.4f} {y:.4f} {z:.4f}")
