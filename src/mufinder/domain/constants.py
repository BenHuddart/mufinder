"""Physical constants and unit conversions, the one home for every value MuFinder uses.

Every constant is taken from ``scipy.constants``, which carries CODATA 2022 from SciPy 1.15
(the floor in ``pyproject.toml``), or is an exact SI definition. Each name carries its
units. This module is the only place that imports ``scipy.constants`` or ``ase.units``;
``tests/test_agent_harness.py`` fails on a constant typed out anywhere else in
``src/mufinder``.
"""

from __future__ import annotations

import math

from scipy.constants import angstrom, c, e, epsilon_0, h, hbar, k, mu_0, physical_constants

#: Planck constant, in J s. Exact in the SI.
PLANCK_CONSTANT_J_S: float = h

#: Reduced Planck constant, h / 2 pi, in J s.
HBAR_J_S: float = hbar

#: Speed of light in vacuum, in m s^-1. Exact in the SI.
LIGHT_SPEED_M_S: float = c

#: Elementary charge, in C; also the number of joules in one electronvolt. Exact in the SI.
ELEMENTARY_CHARGE_C: float = e

#: Boltzmann constant, in J K^-1. Exact in the SI.
BOLTZMANN_J_K: float = k

#: Vacuum magnetic permeability mu_0, in T m A^-1. Source: CODATA (no longer exactly
#: 4 pi 1e-7 since the 2019 SI).
MU0_T_M_PER_A: float = mu_0

#: mu_0 / 4 pi, in T m A^-1, the prefactor of the dipolar field.
MU0_OVER_4PI_T_M_PER_A: float = mu_0 / (4.0 * math.pi)

#: Vacuum electric permittivity epsilon_0, in F m^-1. Source: CODATA.
VACUUM_PERMITTIVITY_F_PER_M: float = epsilon_0

#: Bohr magneton, in J T^-1. Source: CODATA.
BOHR_MAGNETON_J_PER_T: float = physical_constants["Bohr magneton"][0]

#: Nuclear magneton, in J T^-1. Source: CODATA.
NUCLEAR_MAGNETON_J_PER_T: float = physical_constants["nuclear magneton"][0]

#: Atomic mass constant (one dalton), in kg. Source: CODATA.
ATOMIC_MASS_KG: float = physical_constants["atomic mass constant"][0]

#: Metres in one angstrom. Exact.
METRES_PER_ANGSTROM: float = angstrom

#: Bohr radius, in angstrom. Source: CODATA.
BOHR_RADIUS_ANGSTROM: float = physical_constants["Bohr radius"][0] / angstrom

#: Hartree energy, in eV. Source: CODATA.
HARTREE_EV: float = physical_constants["Hartree energy in eV"][0]

#: Muon mass, in unified atomic mass units: the value MuFinder writes to CASTEP
#: ``SPECIES_MASS``. Source: CODATA.
MUON_MASS_AMU: float = physical_constants["muon mass in u"][0]

#: Muon mass, in kg. Source: CODATA.
MUON_MASS_KG: float = physical_constants["muon mass"][0]

#: Muon gyromagnetic ratio, ``gamma_mu = 2 * abs(mu_mu) / hbar``, in rad s^-1 T^-1.
#:
#: Source: CODATA (the muon magnetic moment and hbar), the same expression pymuon-suite
#: writes to CASTEP ``SPECIES_GAMMA``. It agrees with the CODATA 2018
#: gamma_mu / 2 pi = 135.538 8094 MHz T^-1 (gamma_mu ~ 8.516 155e8) to better than 1e-8.
MUON_GAMMA_RAD_S_T: float = abs(2.0 * physical_constants["muon mag. mom."][0] / hbar)

#: Muon gyromagnetic ratio over 2 pi, gamma_mu / 2 pi, in MHz T^-1.
MUON_GAMMA_OVER_2PI_MHZ_T: float = MUON_GAMMA_RAD_S_T / (2.0 * math.pi) / 1e6

#: Tag marking a job that records a calculation performed outside MuFinder.
#:
#: Carried on the ``Job`` rather than looked up through its definition row,
#: because every guard that must respect it — dashboard reset, resubmission,
#: continuation — already holds the job and may not hold the row. A job wearing
#: this tag must never be re-run or have its directory cleaned: the directory
#: holds the only copy of the imported results and their archived originals.
IMPORTED_JOB_TAG = "imported"
