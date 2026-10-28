"""Muon hyperfine coupling from a CASTEP magres run and its spin density.

The Hyperfine Coupling workflow runs one ``TASK : MAGRES`` /
``MAGRES_TASK : HYPERFINE`` job per muon site. CASTEP reconstructs the
all-electron spin density at the muon with its PAW/GIPAW projectors and reports
the hyperfine tensor in MHz. The same job writes the pseudo spin density on its
FFT grid (``.den_fmt``); evaluated at the muon it gives the contact field without
the reconstruction, the quantity Onuorah, Bonfà and De Renzi compare against the
reconstructed one (Phys. Rev. B 97, 174414 (2018)).

Conventions, checked against CASTEP 26.11 runs:

- CASTEP weights the first spin channel by -1 when it accumulates the
  hyperfine tensor (``magres_hyperfine.f90``), so its sign is opposite to the
  physical one: it prints -4628 MHz for muonium in vacuum, whose coupling is
  +4463 MHz. :data:`CASTEP_HYPERFINE_SIGN` undoes this.
- The isotropic coupling and the contact field are related by
  ``A = g_e (gamma_mu / 2 pi) B_c``, and the contact field and spin density
  ``rho_s = rho_up - rho_down`` at the muon by ``B_c = (2 mu_0 / 3) mu_B rho_s``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mufinder.castep_io.charge_density import FormattedDensity
from mufinder.castep_io.magres import HyperfineTensor
from mufinder.domain.constants import (
    BOHR_RADIUS_ANGSTROM,
    ELECTRON_G_FACTOR,
    MUON_GAMMA_OVER_2PI_MHZ_T,
)
from mufinder.science.lattice import cell_volume
from mufinder.science.magnetic_fields import (
    CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
    contact_shell_mean,
    contact_shell_reach_limit,
)

#: Factor turning a CASTEP hyperfine tensor into the physical sign convention.
CASTEP_HYPERFINE_SIGN = -1.0

#: Isotropic muon hyperfine coupling, in MHz, per tesla of contact field.
HYPERFINE_MHZ_PER_TESLA = ELECTRON_G_FACTOR * MUON_GAMMA_OVER_2PI_MHZ_T

#: A host ion counts as magnetic in the contact shell when its Mulliken spin
#: moment is at least this large, in mu_B.
MAGNETIC_MOMENT_THRESHOLD_MUB = 0.1


class HyperfineError(ValueError):
    """Raised when hyperfine inputs cannot give a coupling."""


@dataclass(frozen=True, slots=True)
class MuonHyperfine:
    """The muon's hyperfine coupling in the physical sign convention.

    ``tensor_mhz`` is the full tensor, ``isotropic_mhz`` its Fermi-contact part
    and ``dipolar_principal_mhz`` the principal values of the traceless rest,
    sorted ascending.
    """

    tensor_mhz: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    isotropic_mhz: float
    dipolar_principal_mhz: tuple[float, float, float]

    @property
    def contact_field_t(self) -> float:
        return contact_field_from_isotropic(self.isotropic_mhz)

    @property
    def spin_density_angstrom3(self) -> float:
        return spin_density_from_contact_field(self.contact_field_t)


def muon_hyperfine_from_castep(tensor: HyperfineTensor) -> MuonHyperfine:
    """Return the muon's coupling from CASTEP's tensor, with the sign corrected."""

    matrix = CASTEP_HYPERFINE_SIGN * np.asarray(tensor.tensor_mhz, dtype=float)
    isotropic = CASTEP_HYPERFINE_SIGN * float(tensor.isotropic_mhz)
    dipolar = matrix - isotropic * np.eye(3)
    principal = np.sort(np.linalg.eigvalsh(0.5 * (dipolar + dipolar.T)))
    rows = tuple(tuple(float(value) for value in row) for row in matrix)
    return MuonHyperfine(
        tensor_mhz=rows,  # type: ignore[arg-type]
        isotropic_mhz=isotropic,
        dipolar_principal_mhz=(float(principal[0]), float(principal[1]), float(principal[2])),
    )


def contact_field_from_isotropic(isotropic_mhz: float) -> float:
    """Return the contact field, in T, of an isotropic muon coupling in MHz."""

    return float(isotropic_mhz) / HYPERFINE_MHZ_PER_TESLA


def contact_field_from_spin_density(spin_density_angstrom3: float) -> float:
    """Return the contact field, in T, of a spin density at the muon in Å^-3."""

    return float(spin_density_angstrom3) * CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB


def spin_density_from_contact_field(contact_field_t: float) -> float:
    """Return the spin density at the muon, in Å^-3, that gives ``contact_field_t``."""

    return float(contact_field_t) / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB


def per_angstrom3_to_per_bohr3(density_angstrom3: float) -> float:
    """Convert a density in Å^-3 to atomic units, a0^-3."""

    return float(density_angstrom3) * BOHR_RADIUS_ANGSTROM**3


def per_bohr3_to_per_angstrom3(density_bohr3: float) -> float:
    """Convert a density in atomic units, a0^-3, to Å^-3."""

    return float(density_bohr3) / BOHR_RADIUS_ANGSTROM**3


def spin_density_at(density: FormattedDensity, fractional: Sequence[float]) -> float:
    """Return the pseudo spin density, in Å^-3, at a fractional position.

    The grid is interpolated by its Fourier series, which is exact for the
    plane-wave density CASTEP writes; trilinear interpolation reads several per
    cent low near the muon, where the density is sharply peaked.
    """

    if density.nspins != 2:
        raise HyperfineError(
            "The formatted density is not spin polarised; the hyperfine job must run with "
            "SPIN_POLARIZED : TRUE."
        )
    grid = np.asarray(density.values[:, :, :, 1], dtype=float)
    coefficients = np.fft.fftn(grid) / grid.size
    phases = [
        np.exp(2j * np.pi * np.fft.fftfreq(size, 1.0 / size) * float(fraction))
        for size, fraction in zip(grid.shape, fractional, strict=True)
    ]
    value = np.einsum("ijk,i,j,k->", coefficients, *phases).real
    return float(value) / cell_volume(density.lattice_vectors)


def contact_coupling_from_moments(
    spin_density_angstrom3: float,
    *,
    muon_cartesian: Sequence[float],
    ion_cartesian: Sequence[Sequence[float]],
    ion_moments_mub: Sequence[float],
    lattice_vectors: Sequence[Sequence[float]],
    nnn: int,
    rcont: float,
) -> float:
    """Return the contact coupling, in Å^-3 per mu_B, for the Magnetic Fields page.

    Magnetic Fields takes the contact field as the coupling times the
    ``1/r^3``-weighted mean moment of the ``nnn`` nearest magnetic ions within
    ``rcont`` (muesr's contact rule). Dividing the computed spin density by that
    mean of the DFT moments of the same shell lets Magnetic Fields rescale the
    contact field by the moments of its own magnetic structure. Ions are
    unwrapped to their nearest image of the muon in the periodic cell.
    """

    lattice = np.asarray(lattice_vectors, dtype=float)
    ions = np.asarray(ion_cartesian, dtype=float).reshape(-1, 3)
    moments = np.asarray(ion_moments_mub, dtype=float)
    if moments.shape[0] != ions.shape[0]:
        raise HyperfineError("Every host ion needs one moment for the contact shell.")
    inverse = np.linalg.inv(lattice)
    delta = (ions - np.asarray(muon_cartesian, dtype=float)) @ inverse
    # Rounding gives the nearest image of every ion within half the smallest
    # plane spacing, the reach contact_shell_mean allows, in any cell shape.
    delta -= np.round(delta)
    vectors = delta @ lattice
    magnetic = np.abs(moments) >= MAGNETIC_MOMENT_THRESHOLD_MUB
    if not np.any(magnetic):
        raise HyperfineError(
            f"No host ion carries a spin moment of {MAGNETIC_MOMENT_THRESHOLD_MUB} μB or more, "
            "so there is no contact shell to divide the spin density by."
        )
    coefficients = np.zeros((ions.shape[0], 3), dtype=complex)
    coefficients[:, 2] = moments
    mean = contact_shell_mean(
        vectors,
        coefficients,
        magnetic,
        nnn=int(nnn),
        rcont=float(rcont),
        reach_limit=contact_shell_reach_limit(lattice),
    )
    mean_moment = float(mean[2].real)
    if abs(mean_moment) < MAGNETIC_MOMENT_THRESHOLD_MUB:
        raise HyperfineError(
            "The contact shell has no net moment, so the spin density cannot be expressed per μB."
        )
    return float(spin_density_angstrom3) / mean_moment
