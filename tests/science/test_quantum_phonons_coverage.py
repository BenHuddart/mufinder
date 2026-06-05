"""Coverage-focused tests for CASTEP phonon helpers and validation branches.

These tests deliberately exercise the error paths, parser edge cases, and
numeric helper branches that the primary suite in
``tests/science/test_quantum_phonons.py`` does not reach. They must not
duplicate the happy-path assertions made there.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mufinder.science.quantum_phonons import (
    PhononParseError,
    correlation_matrix,
    covariance_matrix,
    muon_zpe,
    parse_phonon_text,
    real_mode_displacements,
    scaled_mode_displacements,
    select_muon_modes,
    w2_entanglement_witness,
    zpe_from_frequency_cm,
)


# Single Gamma q-point, two atoms, six modes. The H:mu eigenvector for the
# first mode has a zero displacement so the pivot-fallback branch in
# ``real_mode_displacements`` is reachable for mode 1.
GAMMA_TEXT = """ BEGIN header
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
     q-pt=    1    0.000000  0.000000  0.000000      1.0000000000    1.000000  0.000000  0.000000
       1    -100.000000
       2     400.000000
       3     900.000000
       4    1600.000000
       5    2500.000000
       6    3600.000000
                        Phonon Eigenvectors
Mode Ion                X                                   Y                                   Z
   1   1  0.700000000000  0.000000000000      0.000000000000  0.000000000000      0.000000000000  0.000000000000
   1   2  0.000000000000  0.000000000000      0.000000000000  0.000000000000      0.000000000000  0.000000000000
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


def _calc():
    return parse_phonon_text(GAMMA_TEXT)


# --------------------------------------------------------------------------
# Header / top-level parser validation branches
# --------------------------------------------------------------------------


def test_parse_phonon_text_requires_begin_header() -> None:
    with pytest.raises(PhononParseError, match="Expected CASTEP phonon header"):
        parse_phonon_text("not a header\nEND header\n")


def test_parse_phonon_text_empty_input_is_rejected() -> None:
    with pytest.raises(PhononParseError, match="Expected CASTEP phonon header"):
        parse_phonon_text("")


def test_missing_end_header_marker_raises() -> None:
    text = GAMMA_TEXT.replace(" END header", " NOT THE END")
    with pytest.raises(PhononParseError, match="END header"):
        parse_phonon_text(text)


def test_missing_unit_cell_vectors_raises() -> None:
    text = GAMMA_TEXT.replace(" Unit cell vectors (A)", " Unit cell vectors X")
    with pytest.raises(PhononParseError, match="Unit cell vectors were not found"):
        parse_phonon_text(text)


def test_missing_fractional_coordinates_raises() -> None:
    text = GAMMA_TEXT.replace(" Fractional Co-ordinates", " Fractional X")
    with pytest.raises(PhononParseError, match="Fractional coordinates were not found"):
        parse_phonon_text(text)


def test_unsupported_frequency_units_raises() -> None:
    text = GAMMA_TEXT.replace("Frequencies in         cm-1", "Frequencies in         meV")
    with pytest.raises(PhononParseError, match="Unsupported phonon frequency units"):
        parse_phonon_text(text)


def test_short_unit_cell_vector_triplet_raises() -> None:
    text = GAMMA_TEXT.replace(
        "    1.000000    0.000000    0.000000\n",
        "    1.000000    0.000000\n",
        1,
    )
    with pytest.raises(PhononParseError, match="three values for unit cell vector"):
        parse_phonon_text(text)


def test_no_qpoints_present_raises() -> None:
    header_only = GAMMA_TEXT.split(" END header")[0] + " END header\n"
    with pytest.raises(PhononParseError, match="No q-points were found"):
        parse_phonon_text(header_only)


def test_direction_is_parsed_from_qpoint_rest() -> None:
    # The GAMMA_TEXT q-pt line carries a trailing direction triplet.
    qpoint = _calc().qpoints[0]
    assert qpoint.direction == (1.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# q-point block validation branches
# --------------------------------------------------------------------------


def test_missing_eigenvector_block_raises() -> None:
    # Drop everything from the "Phonon Eigenvectors" marker onward so the
    # frequency-scan loop runs off the end of the file.
    text = GAMMA_TEXT.split("                        Phonon Eigenvectors")[0]
    with pytest.raises(PhononParseError, match="Missing eigenvector block"):
        parse_phonon_text(text)


def test_wrong_eigenvector_row_count_raises() -> None:
    # Remove one eigenvector data row -> rows_seen != n_modes * n_atoms.
    text = GAMMA_TEXT.replace(
        "   6   2  0.087328800000  0.000000000000     -0.305651000000"
        "  0.000000000000      0.196490000000  0.000000000000\n",
        "",
    )
    with pytest.raises(PhononParseError, match="eigenvector rows for q-point"):
        parse_phonon_text(text)


# --------------------------------------------------------------------------
# PhononCalculation / atom lookup helpers
# --------------------------------------------------------------------------


def test_atom_index_for_species_requires_unique_match() -> None:
    calc = _calc()
    with pytest.raises(PhononParseError, match="Expected exactly one atom labelled"):
        calc.atom_index_for_species("does-not-exist")


def test_atom_by_index_via_covariance_rejects_unknown_atom() -> None:
    calc = _calc()
    # _atom_by_index is reached through covariance_matrix; atom index 99 absent.
    with pytest.raises(PhononParseError, match="Atom index 99 is not present"):
        covariance_matrix(calc, 1, 99)


# --------------------------------------------------------------------------
# real_mode_displacements branches
# --------------------------------------------------------------------------


def test_real_mode_displacements_rejects_out_of_range_mode() -> None:
    calc = _calc()
    with pytest.raises(PhononParseError, match="Mode index 0 is not present"):
        real_mode_displacements(calc, 0)
    with pytest.raises(PhononParseError, match="Mode index 99 is not present"):
        real_mode_displacements(calc, 99)


def test_real_mode_displacements_uses_global_pivot_when_muon_is_zero() -> None:
    # Mode 1's muon (atom 2) eigenvector is all zeros, so the global-pivot
    # fallback branch is taken. The result must still be real and finite.
    calc = _calc()
    vectors = real_mode_displacements(calc, 1)
    assert vectors.shape == (2, 3)
    assert np.isrealobj(vectors)
    assert np.isfinite(vectors).all()
    # Muon row is zero; nitrogen row carries the (real, aligned) displacement.
    assert np.linalg.norm(vectors[1]) == pytest.approx(0.0)
    assert np.linalg.norm(vectors[0]) > 0.0


# --------------------------------------------------------------------------
# scaled_mode_displacements validation branches
# --------------------------------------------------------------------------


def test_scaled_mode_displacements_rejects_non_numeric_amplitude() -> None:
    calc = _calc()
    with pytest.raises(ValueError, match="Amplitude must be a finite number"):
        scaled_mode_displacements(calc, 2, "not-a-number")  # type: ignore[arg-type]


def test_scaled_mode_displacements_rejects_non_finite_amplitude() -> None:
    calc = _calc()
    with pytest.raises(ValueError, match="Amplitude must be finite"):
        scaled_mode_displacements(calc, 2, float("inf"))


def test_scaled_mode_displacements_rejects_negative_amplitude() -> None:
    calc = _calc()
    with pytest.raises(ValueError, match="Amplitude must be non-negative"):
        scaled_mode_displacements(calc, 2, -0.5)


def test_scaled_mode_displacements_returns_zeros_for_zero_displacement_mode() -> None:
    # Mode 1's largest-norm atom still moves (nitrogen), so to hit the
    # all-zero branch we build a mode where every atom is stationary by
    # zeroing mode 1 entirely via a tailored text.
    zero_mode_text = GAMMA_TEXT.replace(
        "   1   1  0.700000000000  0.000000000000      0.000000000000"
        "  0.000000000000      0.000000000000  0.000000000000\n",
        "   1   1  0.000000000000  0.000000000000      0.000000000000"
        "  0.000000000000      0.000000000000  0.000000000000\n",
    )
    calc = parse_phonon_text(zero_mode_text)
    scaled = scaled_mode_displacements(calc, 1, 0.3)
    assert scaled.shape == (2, 3)
    assert np.all(scaled == 0.0)


# --------------------------------------------------------------------------
# select_muon_modes / muon_zpe / zpe helper branches
# --------------------------------------------------------------------------


def test_select_muon_modes_rejects_non_positive_count() -> None:
    calc = _calc()
    with pytest.raises(ValueError, match="At least one muon mode"):
        select_muon_modes(calc, count=0)
    with pytest.raises(ValueError, match="At least one muon mode"):
        select_muon_modes(calc, count=-2)


def test_muon_zpe_warns_on_non_positive_frequencies() -> None:
    # GAMMA_TEXT mode 1 has a negative frequency and large muon-independent
    # weight; request enough modes that the unstable one is selected.
    calc = _calc()
    result = muon_zpe(calc, count=6)
    assert any("non-positive frequencies" in w for w in result.warnings)
    # The negative-frequency mode must contribute zero ZPE to the total.
    positive_total = sum(m.zpe_eV for m in result.modes if m.frequency_cm > 0.0)
    assert result.total_eV == pytest.approx(positive_total)
    assert result.total_eV > 0.0


def test_zpe_from_frequency_cm_zero_and_negative_return_zero() -> None:
    assert zpe_from_frequency_cm(0.0) == 0.0
    assert zpe_from_frequency_cm(-250.0) == 0.0


# --------------------------------------------------------------------------
# correlation / w2 zero-denominator branches and temperature path
# --------------------------------------------------------------------------


def test_w2_witness_returns_zero_when_self_covariance_vanishes() -> None:
    # A purely Gamma-point file: covariance is built from positive-frequency
    # modes. By zeroing every nitrogen eigenvector component, cov_aa for atom 1
    # vanishes, forcing the denom<=0 branch to return 0.0.
    text = GAMMA_TEXT
    for mode in range(1, 7):
        # Replace each "   <mode>   1  ..." nitrogen row with zeros.
        for line in GAMMA_TEXT.splitlines():
            stripped = line.split()
            if len(stripped) >= 8 and stripped[0] == str(mode) and stripped[1] == "1":
                zeroed = (
                    f"   {mode}   1  0.000000000000  0.000000000000"
                    "      0.000000000000  0.000000000000"
                    "      0.000000000000  0.000000000000"
                )
                text = text.replace(line, zeroed)
                break
    calc = parse_phonon_text(text)
    assert w2_entanglement_witness(calc, 1, 2) == 0.0


def test_correlation_matrix_zero_denominator_yields_zeros() -> None:
    # Same construction: atom 1 has zero self-covariance, so the normalized
    # correlation against it must be all zeros via the np.divide where-guard.
    text = GAMMA_TEXT
    for mode in range(1, 7):
        for line in GAMMA_TEXT.splitlines():
            stripped = line.split()
            if len(stripped) >= 8 and stripped[0] == str(mode) and stripped[1] == "1":
                zeroed = (
                    f"   {mode}   1  0.000000000000  0.000000000000"
                    "      0.000000000000  0.000000000000"
                    "      0.000000000000  0.000000000000"
                )
                text = text.replace(line, zeroed)
                break
    calc = parse_phonon_text(text)
    corr = correlation_matrix(calc, 1, 2)
    assert corr.shape == (3, 3)
    assert np.all(corr == 0.0)


def test_covariance_temperature_branch_scales_up_displacements() -> None:
    # temperature_K > 0 activates the thermal coth factor in
    # _weighted_displacement_vectors; a hotter cell must broaden the muon
    # self-covariance trace relative to the 0 K result.
    calc = _calc()
    cold = covariance_matrix(calc, 2, 2, temperature_K=0.0)
    hot = covariance_matrix(calc, 2, 2, temperature_K=300.0)
    assert np.isfinite(hot).all()
    assert np.trace(hot) >= np.trace(cold)
    assert np.trace(hot) > 0.0


def test_weighted_displacement_all_negative_frequencies_gives_zero_covariance() -> None:
    # If every Gamma frequency is non-positive, the "not np.any(positive)"
    # branch returns zero vectors, so the covariance is exactly zero.
    text = GAMMA_TEXT
    replacements = {
        "       2     400.000000": "       2    -400.000000",
        "       3     900.000000": "       3    -900.000000",
        "       4    1600.000000": "       4   -1600.000000",
        "       5    2500.000000": "       5   -2500.000000",
        "       6    3600.000000": "       6   -3600.000000",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    calc = parse_phonon_text(text)
    cov = covariance_matrix(calc, 2, 2, temperature_K=0.0)
    assert np.all(cov == 0.0)
    # w2 also collapses to 0.0 because both self-covariances vanish.
    assert w2_entanglement_witness(calc, 1, 2) == 0.0


def test_covariance_matrices_are_real_and_finite() -> None:
    calc = _calc()
    cov = covariance_matrix(calc, 1, 2, temperature_K=0.0)
    assert cov.dtype == np.float64
    assert math.isfinite(float(np.trace(cov)))
