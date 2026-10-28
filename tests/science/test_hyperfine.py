from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mufinder.castep_io.charge_density import FormattedDensity
from mufinder.castep_io.magres import hyperfine_tensor_for_species, parse_magres_hyperfine
from mufinder.domain.constants import BOHR_RADIUS_ANGSTROM
from mufinder.science.hyperfine import (
    HyperfineError,
    per_angstrom3_to_per_bohr3,
    per_bohr3_to_per_angstrom3,
    contact_coupling_from_moments,
    contact_field_from_spin_density,
    muon_hyperfine_from_castep,
    spin_density_at,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hyperfine"


def _muon(name: str):
    return muon_hyperfine_from_castep(
        hyperfine_tensor_for_species(parse_magres_hyperfine(FIXTURES / name), "H:mu")
    )


def test_muonium_coupling_is_positive_after_the_castep_sign_flip() -> None:
    # CASTEP prints -4628 MHz for muonium in vacuum; the physical coupling is positive.
    muonium = _muon("muonium_vacuum.magres")

    assert muonium.isotropic_mhz == pytest.approx(4628.4054)
    assert muonium.dipolar_principal_mhz == pytest.approx((0.0, 0.0, 0.0), abs=1e-3)
    # A = g_e (gamma_mu / 2 pi) B_c, with gamma_mu / 2 pi = 135.5388 MHz/T.
    assert muonium.contact_field_t == pytest.approx(4628.4054 / (2.00231930 * 135.5388), rel=1e-6)


def test_iron_contact_field_matches_the_reconstructed_spin_density() -> None:
    # bcc Fe, 2x2x2, muon at the tetrahedral site: -1.195 T, against -1.25 T for the PAW
    # reconstructed density of Onuorah et al. (PRB 97, 174414) and -1.11 T measured.
    iron = _muon("fe_tetrahedral_muon.magres")

    assert iron.isotropic_mhz == pytest.approx(-324.2122)
    assert iron.contact_field_t == pytest.approx(-1.1946, abs=1e-3)
    assert per_angstrom3_to_per_bohr3(iron.spin_density_angstrom3) == pytest.approx(
        -0.0228, abs=1e-4
    )
    assert per_bohr3_to_per_angstrom3(per_angstrom3_to_per_bohr3(-0.15)) == pytest.approx(-0.15)
    # The traceless dipolar part, axial along y.
    assert iron.dipolar_principal_mhz == pytest.approx((-137.61, 68.80, 68.81), abs=0.02)
    assert iron.tensor_mhz[1][1] == pytest.approx(-461.8211)


def test_contact_field_from_spin_density_reproduces_onuorah_table() -> None:
    rho_bohr3 = -0.0238  # Fe, PAW reconstructed, Onuorah et al. Table I
    rho_angstrom3 = rho_bohr3 / BOHR_RADIUS_ANGSTROM**3

    assert contact_field_from_spin_density(rho_angstrom3) == pytest.approx(-1.25, abs=0.005)


def _density(spin: np.ndarray, *, nspins: int = 2, edge: float = 4.0) -> FormattedDensity:
    volume = edge**3
    values = np.zeros((*spin.shape, nspins))
    values[..., 0] = 1.0
    if nspins == 2:
        values[..., 1] = spin * volume
    return FormattedDensity(
        lattice_vectors=((edge, 0.0, 0.0), (0.0, edge, 0.0), (0.0, 0.0, edge)),
        grid=spin.shape,
        nspins=nspins,
        values=values,
    )


def test_spin_density_is_fourier_interpolated_between_grid_points() -> None:
    size = 12
    axis = np.arange(size) / size
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")

    def smooth(fx, fy, fz):
        return (
            0.3
            + 0.2 * np.cos(2 * np.pi * fx) * np.cos(2 * np.pi * fy)
            - 0.1 * np.sin(4 * np.pi * fz)
        )

    density = _density(smooth(x, y, z))
    point = (0.137, 0.402, 0.861)

    assert spin_density_at(density, point) == pytest.approx(float(smooth(*point)), rel=1e-10)
    assert spin_density_at(density, (0.25, 0.5, 0.0)) == pytest.approx(
        float(smooth(0.25, 0.5, 0.0)), rel=1e-10
    )


def test_spin_density_needs_a_spin_polarised_grid() -> None:
    with pytest.raises(HyperfineError, match="not spin polarised"):
        spin_density_at(_density(np.zeros((2, 2, 2)), nspins=1), (0.0, 0.0, 0.0))


_EDGE = 5.66
_LATTICE = ((_EDGE, 0.0, 0.0), (0.0, _EDGE, 0.0), (0.0, 0.0, _EDGE))


def _bcc_ions() -> np.ndarray:
    corners = [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]
    fractional = [tuple(c / 2 for c in corner) for corner in corners]
    fractional += [tuple((c + 0.5) / 2 for c in corner) for corner in corners]
    return np.asarray(fractional) * _EDGE


def test_contact_coupling_divides_by_the_shell_mean_moment() -> None:
    ions = _bcc_ions()
    moments = np.full(len(ions), 2.2)
    muon = np.array([0.25, 0.125, 0.0]) * _EDGE

    coupling = contact_coupling_from_moments(
        -0.154,
        muon_cartesian=muon,
        ion_cartesian=ions,
        ion_moments_mub=moments,
        lattice_vectors=_LATTICE,
        nnn=2,
        rcont=2.5,
    )

    assert coupling == pytest.approx(-0.154 / 2.2)


def test_contact_coupling_ignores_weak_moments_and_uses_nearest_images() -> None:
    ions = np.array([[0.1, 0.0, 0.0], [_EDGE - 1.0, 0.0, 0.0], [2.8, 2.8, 2.8]])
    # The second ion is 1.0 Å away through the periodic boundary; the first is
    # nearer but carries almost no moment, so it is not magnetic.
    coupling = contact_coupling_from_moments(
        0.1,
        muon_cartesian=(0.0, 0.0, 0.0),
        ion_cartesian=ions,
        ion_moments_mub=(0.05, 2.0, -3.0),
        lattice_vectors=_LATTICE,
        nnn=1,
        rcont=2.0,
    )

    assert coupling == pytest.approx(0.05)


def test_contact_coupling_failure_modes() -> None:
    kwargs = dict(
        muon_cartesian=(0.0, 0.0, 0.0),
        ion_cartesian=[[1.0, 0.0, 0.0]],
        lattice_vectors=_LATTICE,
        nnn=1,
        rcont=2.0,
    )
    with pytest.raises(HyperfineError, match="No host ion carries"):
        contact_coupling_from_moments(0.1, ion_moments_mub=(0.0,), **kwargs)
    with pytest.raises(HyperfineError, match="one moment"):
        contact_coupling_from_moments(0.1, ion_moments_mub=(1.0, 2.0), **kwargs)
    with pytest.raises(HyperfineError, match="no net moment"):
        contact_coupling_from_moments(
            0.1,
            muon_cartesian=(0.0, 0.0, 0.0),
            ion_cartesian=[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            ion_moments_mub=(2.0, -2.0),
            lattice_vectors=_LATTICE,
            nnn=2,
            rcont=2.0,
        )
