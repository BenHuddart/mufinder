"""Physical constants for the muon and related particles."""

from __future__ import annotations

import math

from scipy.constants import hbar, physical_constants

MUON_MASS_AMU: float = 0.11342892566

#: Muon gyromagnetic ratio, ``gamma_mu = 2 * abs(mu_mu) / hbar``, in rad s^-1 T^-1.
#:
#: Source: CODATA through ``scipy.constants`` (the muon magnetic moment and hbar),
#: the same expression pymuon-suite writes to CASTEP ``SPECIES_GAMMA``. It agrees with
#: the CODATA 2018 gamma_mu / 2 pi = 135.538 8094 MHz T^-1 (gamma_mu ~ 8.516 155e8)
#: to better than 1e-8.
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
