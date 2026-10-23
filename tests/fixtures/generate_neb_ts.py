"""Generate a faithful synthetic CASTEP NEB ``.ts`` fixture.

The byte layout mirrors ``tssearch_neb_write_ts_image`` /
``tssearch_neb_write_ts_header`` in CASTEP's ``Source/Functional/tssearch_neb.f90``:

* line 1 (fresh run, not a continuation): `` TSTYPE TSConfirmation``
* ``REA`` (reaction coordinate 0.0) and ``PRO`` (1.0) written ONCE at start,
* ``TST`` blocks for every image, repeated for each NEB iteration (the ``i4``
  field is the iteration number; the final iteration is the converged band).

Per-block Fortran formats reproduced here:
  header   (1x, a3, 1x, i4, 3x, es18.8e3)
  energy   (9x, 2(3x, es18.8e3), 21x, '  <-- E')   # energy written twice
  lattice  (9x, 3(3x, es18.8e3), '  <-- h')        # x3
  position (1x, a8, 1x, i4, 3(3x, es18.8e3), '  <-- R')
  force    (1x, a8, 1x, i4, 3(3x, es18.8e3), '  <-- F')
  separator: a near-blank line (list-directed write of ' ')

Energies/positions/forces are in CASTEP internal atomic units (Hartree, Bohr,
Hartree/Bohr) — exactly as the real ``.ts`` files store them.
"""

from __future__ import annotations

from dataclasses import dataclass

HARTREE_TO_EV = 27.211386245988


def _es(value: float) -> str:
    """Format like Fortran ``es18.8e3`` (1 digit before point, 3-digit exp)."""
    s = f"{value:.8E}"  # e.g. '-1.17704630E+00'
    mantissa, exp = s.split("E")
    sign = "+" if int(exp) >= 0 else "-"
    field = f"{mantissa}E{sign}{abs(int(exp)):03d}"
    return field.rjust(18)


def _field(value: float) -> str:
    return "   " + _es(value)  # the leading 3x in every numeric column


def _block(
    tag: str,
    iteration: int,
    reaction_coordinate: float,
    energy_ha: float,
    lattice: list[tuple[float, float, float]],
    species: list[str],
    positions: list[tuple[float, float, float]],
    forces: list[tuple[float, float, float]],
) -> list[str]:
    lines: list[str] = []
    # header: 1x, a3, 1x, i4, 3x, es18.8e3
    lines.append(" " + tag.ljust(3) + " " + f"{iteration:4d}" + "   " + _es(reaction_coordinate))
    # energy written twice, then 21x, then marker
    lines.append(" " * 9 + _field(energy_ha) + _field(energy_ha) + " " * 21 + "  <-- E")
    for row in lattice:
        lines.append(" " * 9 + "".join(_field(c) for c in row) + "  <-- h")
    for elem, pos in zip(species, positions):
        lines.append(
            " " + elem.ljust(8) + " " + f"{1:4d}" + "".join(_field(c) for c in pos) + "  <-- R"
        )
    for elem, frc in zip(species, forces):
        lines.append(
            " " + elem.ljust(8) + " " + f"{1:4d}" + "".join(_field(c) for c in frc) + "  <-- F"
        )
    lines.append("  ")  # separator (list-directed write of ' ')
    return lines


@dataclass(frozen=True)
class Endpoint:
    energy_ha: float
    species: list[str]
    positions: list[tuple[float, float, float]]


def build_neb_ts(
    *,
    lattice: list[tuple[float, float, float]],
    reactant: Endpoint,
    product: Endpoint,
    n_images: int = 5,
    iterations: int = 3,
    barrier_eV: float = 0.12,
    initial_overshoot_eV: float = 0.05,
) -> str:
    """Build a multi-iteration NEB ``.ts`` with a realistic relaxing band.

    ``n_images`` interior beads (odd suits climbing-image), ``iterations`` NEB
    steps whose peak relaxes from ``barrier_eV + initial_overshoot_eV`` down to
    ``barrier_eV`` so the final iteration is the converged minimum-energy path.
    """
    assert len(reactant.positions) == len(product.positions) == len(reactant.species)
    species = reactant.species
    natoms = len(species)
    zero_forces = [(0.0, 0.0, 0.0)] * natoms

    lines: list[str] = [" TSTYPE TSConfirmation"]
    # REA / PRO written once.
    lines += _block(
        "REA", 1, 0.0, reactant.energy_ha, lattice, species, reactant.positions, zero_forces
    )
    lines += _block(
        "PRO", 1, 1.0, product.energy_ha, lattice, species, product.positions, zero_forces
    )

    coords = [(i + 1) / (n_images + 1) for i in range(n_images)]  # interior beads
    for it in range(1, iterations + 1):
        # peak height relaxes toward barrier_eV as iterations proceed
        frac = (it - 1) / max(iterations - 1, 1)
        peak_eV = barrier_eV + initial_overshoot_eV * (1.0 - frac)
        peak_ha = peak_eV / HARTREE_TO_EV
        for rc in coords:
            # linear interpolation of geometry between endpoints
            pos = [
                tuple(r + (p - r) * rc for r, p in zip(rp, pp))
                for rp, pp in zip(reactant.positions, product.positions)
            ]
            # smooth (sin^2) energy profile from reactant baseline up to the peak
            import math

            base = reactant.energy_ha + (product.energy_ha - reactant.energy_ha) * rc
            bump = peak_ha * math.sin(math.pi * rc) ** 2
            energy = base + bump
            lines += _block("TST", it, rc, energy, lattice, species, pos, zero_forces)
    return "\n".join(lines) + "\n"


def _demo_endpoints() -> tuple[list[tuple[float, float, float]], Endpoint, Endpoint]:
    """Small, self-contained Cu+muon-style system (Bohr / Hartree)."""
    a = 6.8431966  # Bohr (~3.62 Ang fcc Cu lattice parameter)
    lattice = [(a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)]
    species = ["Cu", "Cu", "Cu", "Cu", "H:mu"]
    cu = [
        (0.0, 0.0, 0.0),
        (0.5 * a, 0.5 * a, 0.0),
        (0.5 * a, 0.0, 0.5 * a),
        (0.0, 0.5 * a, 0.5 * a),
    ]
    e_reac = -6.67206998e3  # Hartree, taken to match the real Cu .ts scale
    reactant = Endpoint(e_reac, species, [*cu, (0.5 * a, 0.5 * a, 0.5 * a)])  # octahedral
    product = Endpoint(
        e_reac + 1.0e-4, species, [*cu, (0.25 * a, 0.25 * a, 0.25 * a)]
    )  # tetrahedral
    return lattice, reactant, product


if __name__ == "__main__":
    import pathlib

    lattice, reactant, product = _demo_endpoints()
    text = build_neb_ts(
        lattice=lattice, reactant=reactant, product=product, n_images=5, iterations=3
    )
    out = pathlib.Path(__file__).with_name("neb_path.ts")
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.splitlines())} lines)")
