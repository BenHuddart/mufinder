"""CODATA 2022 reference values for tests, typed independently of the code under test.

``mufinder.domain.constants`` takes its values from ``scipy.constants``. These are the
same CODATA 2022 recommended values typed out by hand, so a test that checks a formula
against them also catches a constant the code reads from the wrong place.
"""

from __future__ import annotations

import math

#: Vacuum magnetic permeability, in T m A^-1 (CODATA 2022).
MU0_T_M_PER_A = 1.256_637_061_27e-6

#: mu_0 / 4 pi, in T m A^-1.
MU0_OVER_4PI_T_M_PER_A = MU0_T_M_PER_A / (4.0 * math.pi)

#: Bohr magneton, in J T^-1 (CODATA 2022).
BOHR_MAGNETON_J_PER_T = 9.274_010_065_7e-24

#: Muon mass, in unified atomic mass units (CODATA 2022).
MUON_MASS_AMU = 0.113_428_925_7
