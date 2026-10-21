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
    mode_width_angstrom,
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


#: The same phonon file as ``PHONON_TEXT`` but with the CASTEP default hydrogen
#: mass on the muon — what a stage-2 continuation produces when the stage-1
#: singlepoint was written without a ``SPECIES_MASS`` block.
HYDROGEN_MASS_PHONON_TEXT = PHONON_TEXT.replace("H:mu      0.113429", "H:mu      1.007940")


#: A Gamma-only file whose first mode is unstable (negative frequency) and
#: leaves the muon at rest, and whose q-point line carries a direction.
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


def _static_nitrogen_text() -> str:
    """``GAMMA_TEXT`` with every nitrogen eigenvector row zeroed."""

    lines = []
    for line in GAMMA_TEXT.splitlines():
        fields = line.split()
        if len(fields) >= 8 and fields[0].isdigit() and fields[1] == "1":
            line = f"   {fields[0]}   1" + "  0.000000000000" * 6
        lines.append(line)
    return "\n".join(lines) + "\n"


# Parsing .phonon files


def test_parse_phonon_text_reads_header_gamma_and_eigenvectors() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    assert len(calculation.atoms) == 2
    assert calculation.atoms[1].species == "H:mu"
    assert calculation.gamma_point().frequencies_cm == (100.0, 400.0, 900.0, 1600.0, 2500.0, 3600.0)
    assert calculation.gamma_point().eigenvectors.shape == (6, 2, 3)


def test_direction_is_parsed_from_qpoint_rest() -> None:
    # The GAMMA_TEXT q-pt line carries a trailing direction triplet.
    qpoint = _calc().qpoints[0]
    assert qpoint.direction == (1.0, 0.0, 0.0)


def test_structure_from_phonon_calculation_builds_muonated_structure() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    structure = structure_from_phonon_calculation(calculation, label="Gamma cell")

    assert structure.label == "Gamma cell"
    assert structure.origin == "phonon"
    assert structure.species == ["N", "H:mu"]
    assert structure.fractional_positions == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert structure.properties["pbc"] == [True, True, True]


def test_parser_rejects_missing_gamma_point() -> None:
    shifted = PHONON_TEXT.replace("0.000000  0.000000  0.000000", "0.500000  0.000000  0.000000", 1)
    calculation = parse_phonon_text(shifted)

    with pytest.raises(PhononParseError, match="Gamma"):
        calculation.gamma_point()


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


def test_atom_index_for_species_requires_unique_match() -> None:
    calc = _calc()
    with pytest.raises(PhononParseError, match="Expected exactly one atom labelled"):
        calc.atom_index_for_species("does-not-exist")


# Muon modes and zero-point energy


def test_muon_modes_and_zpe_use_half_quantum_conversion() -> None:
    calculation = parse_phonon_text(PHONON_TEXT)

    modes = select_muon_modes(calculation)
    assert [mode.mode_index for mode in modes] == [2, 3, 4]
    assert [mode.frequency_cm for mode in modes] == [400.0, 900.0, 1600.0]
    result = muon_zpe(calculation)

    assert result.total_eV == pytest.approx((400.0 + 900.0 + 1600.0) * CM_INV_ZPE_TO_EV)
    assert zpe_from_frequency_cm(1000.0) == pytest.approx(0.06199209915)


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


# Mode shapes


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


# Covariance, correlation and entanglement witnesses


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


@pytest.mark.parametrize("text", [PHONON_TEXT, GAMMA_TEXT], ids=["phonon", "gamma"])
def test_w1_is_the_top_three_muon_participation_minus_three(text: str) -> None:
    calculation = parse_phonon_text(text)
    top_three = sorted(muon_mode_participations(calculation))[-3:]

    w1 = w1_entanglement_witness(calculation)

    assert w1 == pytest.approx(sum(top_three) - 3.0)
    # It sums the participations of the same modes the ZPE uses.
    selected = select_muon_modes(calculation)
    assert w1 == pytest.approx(sum(mode.participation for mode in selected) - 3.0)
    assert -3.0 <= w1 <= 0.0


def _muon_localised_text() -> str:
    """Six modes: three move only nitrogen, three move only the muon along x, y, z."""

    rows = []
    for mode in range(1, 7):
        for ion in (1, 2):
            vector = [0.0, 0.0, 0.0]
            moving_ion = 1 if mode <= 3 else 2
            if ion == moving_ion:
                vector[(mode - 1) % 3] = 1.0
            columns = "      ".join(f"{value:.12f}  0.000000000000" for value in vector)
            rows.append(f"   {mode}   {ion}  {columns}")
    header, _eigenvectors = PHONON_TEXT.split("Mode Ion")
    return header + "Mode Ion                X                                   Y                                   Z\n" + "\n".join(rows) + "\n"


def test_w1_is_zero_when_the_muon_motion_is_fully_in_three_modes() -> None:
    calculation = parse_phonon_text(_muon_localised_text())

    assert w1_entanglement_witness(calculation) == pytest.approx(0.0)
    assert [mode.mode_index for mode in select_muon_modes(calculation)] == [4, 5, 6]


@pytest.mark.parametrize(
    ("frequency_cm", "zpe_eV", "width_A"),
    [(2742.0, 0.17, 0.23), (3226.0, 0.20, 0.21)],
    ids=["zpe-0.17eV", "zpe-0.20eV"],
)
def test_mode_width_angstrom_is_the_muon_zero_point_spread(
    frequency_cm: float, zpe_eV: float, width_A: float
) -> None:
    assert zpe_from_frequency_cm(frequency_cm) == pytest.approx(zpe_eV, abs=5e-4)
    assert round(mode_width_angstrom(frequency_cm), 2) == width_A


def test_mode_width_angstrom_scales_with_mass_and_is_zero_for_unstable_modes() -> None:
    muon = mode_width_angstrom(3000.0)

    assert mode_width_angstrom(3000.0, mass_amu=4.0 * MUON_MASS_AMU) == pytest.approx(muon / 2.0)
    assert mode_width_angstrom(-100.0) == 0.0
    assert mode_width_angstrom(0.0) == 0.0


def test_select_muon_modes_carries_each_modes_width() -> None:
    modes = select_muon_modes(parse_phonon_text(PHONON_TEXT))

    for mode in modes:
        assert mode.width_A == pytest.approx(mode_width_angstrom(mode.frequency_cm))
        assert mode.width_A > 0.0


def test_covariance_matrices_are_real_and_finite() -> None:
    calc = _calc()
    cov = covariance_matrix(calc, 1, 2, temperature_K=0.0)
    assert cov.dtype == np.float64
    assert math.isfinite(float(np.trace(cov)))


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


def test_atom_by_index_via_covariance_rejects_unknown_atom() -> None:
    calc = _calc()
    # _atom_by_index is reached through covariance_matrix; atom index 99 absent.
    with pytest.raises(PhononParseError, match="Atom index 99 is not present"):
        covariance_matrix(calc, 1, 99)


def test_w2_witness_returns_zero_when_self_covariance_vanishes() -> None:
    # A static nitrogen has zero self-covariance, so the witness denominator vanishes.
    assert w2_entanglement_witness(parse_phonon_text(_static_nitrogen_text()), 1, 2) == 0.0


def test_correlation_matrix_zero_denominator_yields_zeros() -> None:
    # Correlation against an atom with zero self-covariance is defined as zero.
    corr = correlation_matrix(parse_phonon_text(_static_nitrogen_text()), 1, 2)

    assert corr.shape == (3, 3)
    assert np.all(corr == 0.0)
