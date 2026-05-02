"""Tests for CASTEP phonon quantum-correction analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mufinder.science.quantum_phonons import (
    CM_INV_ZPE_TO_EV,
    PhononParseError,
    correlation_matrix,
    covariance_matrix,
    muon_zpe,
    parse_phonon_text,
    select_muon_modes,
    w1_entanglement_witness,
    w2_entanglement_witness,
    zpe_from_frequency_cm,
)


PHONON_TEXT = """ BEGIN header
 Number of ions         2
 Number of branches     6
 Number of wavevectors  1
 Frequencies in         cm-1
 Unit cell vectors (A)
    1.000000    0.000000    0.000000
    0.000000    1.000000    0.000000
    0.000000    0.000000    1.000000
 Fractional Co-ordinates
     1     0.000000    0.000000    0.000000   N        14.006740
     2     0.500000    0.500000    0.500000   H:mu      0.113429
 END header
     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000
       1     100.000000
       2     400.000000
       3     900.000000
       4    1600.000000
       5    2500.000000
       6    3600.000000
                        Phonon Eigenvectors
Mode Ion                X                                   Y                                   Z
   1   1  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   1   2  0.408248000000  0.000000000000      0.408248000000  0.000000000000      0.408248000000  0.000000000000
   2   1 -0.336650000000  0.000000000000      0.019803000000  0.000000000000      0.138621000000  0.000000000000
   2   2 -0.693103000000  0.000000000000      0.376256000000  0.000000000000      0.495074000000  0.000000000000
   3   1 -0.080432200000  0.000000000000     -0.350117000000  0.000000000000      0.230257000000  0.000000000000
   3   2  0.189252000000  0.000000000000     -0.619801000000  0.000000000000      0.630841000000  0.000000000000
   4   1 -0.501570000000  0.000000000000     -0.324961000000  0.000000000000     -0.339089000000  0.000000000000
   4   2  0.536891000000  0.000000000000      0.459183000000  0.000000000000      0.169545000000  0.000000000000
   5   1  0.583042000000  0.000000000000     -0.144467000000  0.000000000000     -0.701203000000  0.000000000000
   5   2 -0.145760000000  0.000000000000      0.057786700000  0.000000000000      0.350601000000  0.000000000000
   6   1 -0.349315000000  0.000000000000      0.764127000000  0.000000000000     -0.392979000000  0.000000000000
   6   2  0.087328800000  0.000000000000     -0.305651000000  0.000000000000      0.196490000000  0.000000000000
"""


def test_parse_phonon_text_reads_header_gamma_and_eigenvectors() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    assert len(calculation.atoms) == 2
    assert calculation.atoms[1].species == "H:mu"
    assert calculation.gamma_point().frequencies_cm == (100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0)
    assert calculation.gamma_point().eigenvectors.shape == (6, 2, 3)


def test_muon_modes_and_zpe_use_half_quantum_conversion() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    modes = select_muon_modes(calculation)
    assert [mode.mode_index for mode in modes] == [2, 3, 4]
    assert [mode.frequency_cm for mode in modes] == [400.0, 900.0, 1600.0]
    result = muon_zpe(calculation)

    assert result.total_eV == pytest.approx((400.0 + 900.0 + 1600.0) * CM_INV_ZPE_TO_EV)
    assert zpe_from_frequency_cm(1000.0) == pytest.approx(0.06199209915)


def test_entanglement_witnesses_and_covariance_are_finite() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    assert math.isfinite(w1_entanglement_witness(calculation))
    covariance = covariance_matrix(calculation, 1, 2)
    correlation = correlation_matrix(calculation, 1, 2)
    assert covariance.shape == (3, 3)
    assert correlation.shape == (3, 3)
    assert np.isfinite(covariance).all()
    assert np.isfinite(correlation).all()
    assert math.isfinite(w2_entanglement_witness(calculation, 1, 2))


def test_parser_rejects_missing_gamma_point() -> None:
    shifted = PHONON_TEXT.replace("0.000000  0.000000  0.000000", "0.500000  0.000000  0.000000", 1)
    calculation = parse_phonon_text(shifted)

    with pytest.raises(PhononParseError, match="Gamma"):
        calculation.gamma_point()
