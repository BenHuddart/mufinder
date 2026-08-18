"""Tests for CASTEP phonon quantum-correction analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mufinder.science.quantum_phonons import (
    CM_INV_ZPE_TO_EV,
    MUON_MASS_AMU,
    PhononParseError,
    correlation_matrix,
    covariance_matrix,
    muon_mass_warning,
    muon_zpe,
    muon_mode_participations,
    parse_phonon_text,
    real_mode_displacements,
    scaled_mode_displacements,
    select_muon_modes,
    structure_from_phonon_calculation,
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


def test_structure_from_phonon_calculation_builds_muonated_structure() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    structure = structure_from_phonon_calculation(calculation, label="Gamma cell")

    assert structure.label == "Gamma cell"
    assert structure.origin == "phonon"
    assert structure.species == ["N", "H:mu"]
    assert structure.fractional_positions == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert structure.properties["pbc"] == [True, True, True]


def test_muon_modes_and_zpe_use_half_quantum_conversion() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    modes = select_muon_modes(calculation)
    assert [mode.mode_index for mode in modes] == [2, 3, 4]
    assert [mode.frequency_cm for mode in modes] == [400.0, 900.0, 1600.0]
    result = muon_zpe(calculation)

    assert result.total_eV == pytest.approx((400.0 + 900.0 + 1600.0) * CM_INV_ZPE_TO_EV)
    assert zpe_from_frequency_cm(1000.0) == pytest.approx(0.06199209915)


def test_muon_mode_helpers_return_participation_real_shape_and_scaled_amplitude() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    participation = muon_mode_participations(calculation)
    assert len(participation) == 6
    assert all(value >= 0.0 for value in participation)
    assert sorted(index + 1 for index in np.argsort(participation)[-3:]) == [2, 3, 4]

    mode_shape = real_mode_displacements(calculation, 2)
    assert mode_shape.shape == (2, 3)
    assert np.isrealobj(mode_shape)
    assert np.linalg.norm(mode_shape[1]) > 0.0

    scaled = scaled_mode_displacements(calculation, 2, 0.2)
    assert scaled.shape == (2, 3)
    assert float(np.max(np.linalg.norm(scaled, axis=1))) == pytest.approx(0.2)
    assert np.allclose(scaled / 0.2, mode_shape / np.max(np.linalg.norm(mode_shape, axis=1)))


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


#: The same phonon file as ``PHONON_TEXT`` but with the CASTEP default hydrogen
#: mass on the muon — what a stage-2 continuation produces when the stage-1
#: singlepoint was written without a ``SPECIES_MASS`` block.
HYDROGEN_MASS_PHONON_TEXT = PHONON_TEXT.replace("H:mu      0.113429", "H:mu      1.007940")


def test_muon_mass_warning_is_silent_for_a_correct_muon_mass() -> None:
    assert muon_mass_warning(parse_phonon_text(PHONON_TEXT)) is None


def test_muon_mass_warning_reports_a_hydrogen_mass_muon() -> None:
    warning = muon_mass_warning(parse_phonon_text(HYDROGEN_MASS_PHONON_TEXT))

    assert warning is not None
    assert "1.00794" in warning
    assert f"{MUON_MASS_AMU:.6g}" in warning
    assert "SPECIES_MASS" in warning


def test_muon_zpe_carries_the_mass_warning() -> None:
    result = muon_zpe(parse_phonon_text(HYDROGEN_MASS_PHONON_TEXT))

    assert any("not the muon mass" in warning for warning in result.warnings)
    # The ZPE itself is still computed from the reported frequencies: the run has
    # to be repeated, and nothing here can rescale it after the fact.
    assert result.total_eV == pytest.approx((400.0 + 900.0 + 1600.0) * CM_INV_ZPE_TO_EV)


def test_muon_mass_warning_is_silent_when_the_muon_is_absent() -> None:
    """A warning helper must not raise where its callers raise for themselves."""

    calculation = parse_phonon_text(PHONON_TEXT)
    assert muon_mass_warning(calculation, muon_species="Xx") is None
