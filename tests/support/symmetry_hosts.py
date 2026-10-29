"""Representative host structures for symmetry-sensitive science tests.

Each host exercises a different corner of the symmetry code: many operations
(BCC), none (triclinic), non-orthogonal axes (HCP), screw/glide operations
without fixed points (rutile), and rotations that are not signed permutation
matrices (primitive FCC).
"""

from __future__ import annotations

from mufinder.domain.models import Structure


def bcc_host() -> Structure:
    # Two Si atoms in a 5 Å cube: spglib finds Im-3m with 96 operations.
    return Structure(
        label="BCC",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def triclinic_host() -> Structure:
    return Structure(
        label="Triclinic",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.1, 9.0, 0.0], [0.2, 0.3, 8.0]],
        species=["Si", "O", "C"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.31, 0.23, 0.19], [0.67, 0.41, 0.83]],
    )


def hcp_host() -> Structure:
    a, c = 3.21, 5.21
    return Structure(
        label="HCP",
        lattice_vectors=[[a, 0.0, 0.0], [-a / 2.0, a * 3.0**0.5 / 2.0, 0.0], [0.0, 0.0, c]],
        species=["Mg", "Mg"],
        fractional_positions=[[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]],
    )


def rutile_host() -> Structure:
    # P4_2/mnm has screw and glide operations without fixed points.
    a, c, u = 4.594, 2.959, 0.305
    return Structure(
        label="Rutile",
        lattice_vectors=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]],
        species=["Ti", "Ti", "O", "O", "O", "O"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [u, u, 0.0],
            [1.0 - u, 1.0 - u, 0.0],
            [0.5 + u, 0.5 - u, 0.5],
            [0.5 - u, 0.5 + u, 0.5],
        ],
    )


def fcc_primitive_host() -> Structure:
    # A primitive FCC cell: rotations are not signed permutation matrices.
    h = 3.61 / 2.0
    return Structure(
        label="FCC",
        lattice_vectors=[[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        species=["Cu"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def diamond_si_host() -> Structure:
    # Conventional diamond Si: Fd-3m with 192 operations (48 point operations x 4 centring translations).
    a = 5.43
    fcc = [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    positions = [
        list(point) for corner in fcc for point in (corner, tuple(value + 0.25 for value in corner))
    ]
    return Structure(
        label="Diamond Si",
        lattice_vectors=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]],
        species=["Si"] * len(positions),
        fractional_positions=positions,
    )


def zno_host(*, cif_rounded: bool = False) -> Structure:
    # Wurtzite ZnO, P6_3mc (12 operations). ``cif_rounded`` writes 1/3 and 2/3 to four
    # decimals, as most CIF files do.
    a, c, u = 3.25, 5.207, 0.382
    third, two_thirds = (0.3333, 0.6667) if cif_rounded else (1.0 / 3.0, 2.0 / 3.0)
    return Structure(
        label="ZnO",
        lattice_vectors=[[a, 0.0, 0.0], [-a / 2.0, a * 3.0**0.5 / 2.0, 0.0], [0.0, 0.0, c]],
        species=["Zn", "Zn", "O", "O"],
        fractional_positions=[
            [third, two_thirds, 0.0],
            [two_thirds, third, 0.5],
            [third, two_thirds, u],
            [two_thirds, third, 0.5 + u],
        ],
    )


def si_primitive_host() -> Structure:
    # Diamond Si in its primitive FCC cell: Fd-3m with 48 operations, axes at 60 degrees.
    h = 5.43 / 2.0
    return Structure(
        label="Si primitive",
        lattice_vectors=[[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def cof2_host(*, cobalt_label: str = "Co") -> Structure:
    # Rutile CoF2, P4_2/mnm (16 operations); every F is on one 4f orbit.
    a, c, u = 4.695, 3.18, 0.303
    return Structure(
        label="CoF2",
        lattice_vectors=[[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, c]],
        species=[cobalt_label, cobalt_label, "F", "F", "F", "F"],
        fractional_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [u, u, 0.0],
            [1.0 - u, 1.0 - u, 0.0],
            [0.5 + u, 0.5 - u, 0.5],
            [0.5 - u, 0.5 + u, 0.5],
        ],
    )
