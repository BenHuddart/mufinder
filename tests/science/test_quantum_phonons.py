"""Tests for CASTEP phonon quantum-correction analysis."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from mufinder.domain.constants import ATOMIC_MASS_KG, HBAR_J_S, LIGHT_SPEED_M_S, MUON_MASS_AMU
from mufinder.science.quantum_phonons import (
    CM_INV_ZPE_TO_EV,
    NO_REAL_FREQUENCY_ALONG_AXIS,
    NO_REAL_FREQUENCY_ALONG_DIRECTION,
    PhononParseError,
    correlation_matrix,
    covariance_matrix,
    mode_width_angstrom,
    muon_mass_warning,
    MuonNeighbour,
    muon_nearest_atoms,
    muon_dynamical_block_cm2,
    muon_principal_zpes,
    muon_zpe,
    muon_zpe_along,
    muon_mode_participations,
    parse_phonon_path,
    order_muon_neighbours,
    parse_phonon_text,
    real_mode_displacements,
    scaled_mode_displacements,
    select_muon_modes,
    structure_from_phonon_calculation,
    w1_entanglement_witness,
    w2_entanglement_witness,
    zpe_from_frequency_cm,
)
from mufinder.science.lattice import nearest_image_distance
from tests.support.phonon_fixtures import (
    BOX_EDGE_ANG,
    MUON_ATOM_INDEX,
    MUON_NITROGEN_DISTANCE_ANG,
    NEAREST_NITROGEN_INDEX,
    n2_mu_n2_phonon_text,
    phonon_calculation_from_dynamical_matrix,
    uncoupled_muon_dynamical_matrix,
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
#: mass on the muon — what a phonon run produces when it continues from a
#: checkpoint written without the muon's ``SPECIES_MASS`` block.
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


def test_muon_mass_warning_accepts_a_rounded_muon_mass_within_one_percent() -> None:
    # 0.1138 AMU is a common rounded muon mass, 0.33 % above MuFinder's; 0.1150 AMU
    # is 1.4 % above it and is still flagged.
    rounded = PHONON_TEXT.replace("H:mu      0.113429", "H:mu      0.113800")
    too_heavy = PHONON_TEXT.replace("H:mu      0.113429", "H:mu      0.115000")

    assert muon_mass_warning(parse_phonon_text(rounded)) is None
    warning = muon_mass_warning(parse_phonon_text(too_heavy))
    assert warning is not None
    assert "0.115" in warning


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
    return (
        header
        + "Mode Ion                X                                   Y                                   Z\n"
        + "\n".join(rows)
        + "\n"
    )


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


# A toy model: the [N2-mu-N2]+ complex of solid nitrogen
#
# The numbers below are the harmonic (weakly-bound) row of Gomilsek et al.,
# Commun. Phys. 6, 142 (2023), Table 1. The fixture is a five-atom toy model
# tuned to reproduce them (see tests/support/phonon_fixtures.py); the worked
# example of the documentation reads the paper's own phonon run instead (see
# tests/docs/test_docs_alpha_n2.py).


def _n2_mu_n2():
    return parse_phonon_text(n2_mu_n2_phonon_text())


def test_n2_mu_n2_fixture_parses_as_a_gamma_only_muon_run() -> None:
    calculation = _n2_mu_n2()

    assert [atom.species for atom in calculation.atoms] == ["N", "N", "H:mu", "N", "N"]
    assert calculation.atom_index_for_species("H:mu") == MUON_ATOM_INDEX
    gamma = calculation.gamma_point()
    assert len(gamma.frequencies_cm) == 15
    assert all(frequency > 0.0 for frequency in gamma.frequencies_cm)
    assert list(gamma.frequencies_cm) == sorted(gamma.frequencies_cm)
    assert muon_mass_warning(calculation) is None
    # Mass-weighted eigenvectors are orthonormal, so the muon's participation
    # summed over every mode is exactly its three degrees of freedom.
    assert sum(muon_mode_participations(calculation)) == pytest.approx(3.0, abs=1e-9)


def test_n2_mu_n2_zero_point_energies_match_the_paper() -> None:
    result = muon_zpe(_n2_mu_n2())

    assert result.warnings == ()
    assert [round(mode.zpe_eV, 2) for mode in result.modes] == [0.17, 0.20, 0.20]
    assert round(result.total_eV, 2) == 0.58
    # The muon modes are the stiffest modes here, so the largest-participation
    # rule picks the same three modes as the paper's highest-frequency rule.
    assert [mode.mode_index for mode in result.modes] == [13, 14, 15]


def test_n2_mu_n2_selected_modes_carry_most_but_not_all_muon_motion() -> None:
    calculation = _n2_mu_n2()
    modes = select_muon_modes(calculation)

    summed = sum(mode.participation for mode in modes)
    assert summed == pytest.approx(2.86, abs=0.01)
    # The longitudinal mode hybridises with the odd N2 stretch just below it,
    # which takes most of the muon weight the three selected modes miss.
    participation = muon_mode_participations(calculation)
    assert modes[0].participation == pytest.approx(0.87, abs=0.01)
    unselected = [
        value for index, value in enumerate(participation) if index + 1 not in {13, 14, 15}
    ]
    assert max(unselected) == participation[10] == pytest.approx(0.13, abs=0.01)


def test_n2_mu_n2_harmonic_widths_match_the_paper() -> None:
    modes = select_muon_modes(_n2_mu_n2())

    muon_mass_kg = ATOMIC_MASS_KG * MUON_MASS_AMU
    widths = [
        1e10
        * math.sqrt(
            HBAR_J_S
            / (2.0 * muon_mass_kg * 2.0 * math.pi * LIGHT_SPEED_M_S * 100.0 * mode.frequency_cm)
        )
        for mode in modes
    ]
    assert [round(width, 2) for width in widths] == [0.23, 0.21, 0.21]


def test_n2_mu_n2_muon_correlates_with_its_nearest_nitrogens() -> None:
    calculation = _n2_mu_n2()

    nearest = w2_entanglement_witness(calculation, MUON_ATOM_INDEX, NEAREST_NITROGEN_INDEX)
    assert -1.0 <= nearest <= 0.0
    assert abs(nearest) == pytest.approx(0.26, abs=0.03)
    # Both nearest nitrogens are equivalent, and the correlation decays with distance.
    assert w2_entanglement_witness(calculation, MUON_ATOM_INDEX, 4) == pytest.approx(nearest)
    assert abs(w2_entanglement_witness(calculation, MUON_ATOM_INDEX, 1)) < abs(nearest)
    # The muon and its nearest nitrogen move together, not against each other.
    assert np.trace(covariance_matrix(calculation, MUON_ATOM_INDEX, NEAREST_NITROGEN_INDEX)) > 0.0


def test_the_muon_self_covariance_is_in_square_angstrom() -> None:
    # Each mode adds its muon participation times its zero-point width
    # squared, so the trace is the muon's mean-square displacement in Å².
    calculation = _n2_mu_n2()
    gamma = calculation.gamma_point()
    participation = muon_mode_participations(calculation)

    expected = sum(
        share * mode_width_angstrom(frequency) ** 2
        for share, frequency in zip(participation, gamma.frequencies_cm, strict=True)
    )
    trace = np.trace(covariance_matrix(calculation, MUON_ATOM_INDEX, MUON_ATOM_INDEX))
    assert trace == pytest.approx(expected, rel=1e-6)
    assert 0.1 < trace < 0.2


def test_the_nearest_atoms_are_ranked_by_distance_with_ties_in_index_order() -> None:
    # The muon sits midway along the chain, so each shell holds two nitrogens.
    calculation = _n2_mu_n2()

    neighbours = muon_nearest_atoms(calculation, 12)

    assert [neighbour.atom_index for neighbour in neighbours] == [2, 4, 1, 5]
    assert {neighbour.species for neighbour in neighbours} == {"N"}
    assert neighbours[0].distance_ang == pytest.approx(MUON_NITROGEN_DISTANCE_ANG, abs=1e-6)
    assert neighbours[1].distance_ang == pytest.approx(MUON_NITROGEN_DISTANCE_ANG, abs=1e-6)
    for neighbour in neighbours:
        assert neighbour.correlation == -w2_entanglement_witness(
            calculation, MUON_ATOM_INDEX, neighbour.atom_index
        )


def test_the_nearest_atom_list_stops_at_the_count_even_inside_a_shell() -> None:
    neighbours = muon_nearest_atoms(_n2_mu_n2(), 3)

    assert [neighbour.atom_index for neighbour in neighbours] == [2, 4, 1]


def test_a_tie_is_judged_to_a_thousandth_of_an_angstrom() -> None:
    calculation = _n2_mu_n2()
    # Move atom 2 along the chain, 0.0003 Å further from the muon than atom 4 is.
    shifted = []
    for atom in calculation.atoms:
        x, y, z = atom.fractional_coordinates
        step = 0.0003 / BOX_EDGE_ANG if atom.index == 2 else 0.0
        shifted.append(replace(atom, fractional_coordinates=(x - step, y, z)))

    neighbours = muon_nearest_atoms(replace(calculation, atoms=tuple(shifted)), 2)

    assert [neighbour.atom_index for neighbour in neighbours] == [2, 4]
    assert neighbours[0].distance_ang > neighbours[1].distance_ang


def test_a_tie_does_not_depend_on_where_the_distances_fall_against_a_rounding_edge() -> None:
    calculation = _n2_mu_n2()
    distance = {atom.index: atom for atom in calculation.atoms}
    muon_x = distance[MUON_ATOM_INDEX].fractional_coordinates[0]
    # Atoms 2 and 4 either side of 1.2905 Å, a hair apart, with atom 2 the further.
    shifted = []
    for atom in calculation.atoms:
        x, y, z = atom.fractional_coordinates
        if atom.index == 2:
            x = muon_x - 1.29050001 / BOX_EDGE_ANG
        elif atom.index == 4:
            x = muon_x + 1.29049999 / BOX_EDGE_ANG
        shifted.append(replace(atom, fractional_coordinates=(x, y, z)))

    neighbours = muon_nearest_atoms(replace(calculation, atoms=tuple(shifted)), 2)

    assert [neighbour.atom_index for neighbour in neighbours] == [2, 4]


def test_atoms_more_than_the_tolerance_apart_are_not_tied() -> None:
    calculation = _n2_mu_n2()
    # Atom 2 moved 0.002 Å further from the muon than atom 4 is.
    shifted = []
    for atom in calculation.atoms:
        x, y, z = atom.fractional_coordinates
        step = 0.002 / BOX_EDGE_ANG if atom.index == 2 else 0.0
        shifted.append(replace(atom, fractional_coordinates=(x - step, y, z)))

    neighbours = muon_nearest_atoms(replace(calculation, atoms=tuple(shifted)), 2)

    assert [neighbour.atom_index for neighbour in neighbours] == [4, 2]


def test_the_nearest_atoms_measure_to_the_nearest_periodic_image() -> None:
    # In this sheared cell the shortest image is (-a + b + c) / 2. Folding each
    # fractional component separately finds (a + b + c) / 2, which is 1.18 Å long.
    lattice = ((1.0, 0.0, 0.0), (0.9, 1.0, 0.0), (0.0, 0.0, 1.0))
    sheared = PHONON_TEXT.replace(
        "    0.000000    1.000000    0.000000", "    0.900000    1.000000    0.000000"
    )
    calculation = parse_phonon_text(sheared)
    assert calculation.lattice_vectors == lattice

    neighbour = muon_nearest_atoms(calculation, 1)[0]

    atom, muon = calculation.atoms
    assert neighbour.distance_ang == pytest.approx(
        nearest_image_distance(muon.fractional_coordinates, atom.fractional_coordinates, lattice)
    )
    assert neighbour.distance_ang == pytest.approx(0.5 * math.sqrt(0.1**2 + 1.0 + 1.0))


def test_the_nearest_atom_count_must_be_positive() -> None:
    with pytest.raises(ValueError, match="count must be positive"):
        muon_nearest_atoms(_n2_mu_n2(), 0)


# --- The ZPE along a direction: the muon block D(0) of the dynamical matrix ---------

_GDRU2SI2 = Path(__file__).parents[1] / "fixtures" / "gdru2si2"
_TOY_LATTICE = ((6.0, 0.0, 0.0), (0.0, 6.0, 0.0), (0.0, 0.0, 6.0))
_TOY_SPECIES = ("Si", "H:mu", "Si")
_TOY_POSITIONS = ((0.0, 0.0, 0.0), (0.25, 0.25, 0.25), (0.5, 0.5, 0.5))
_TOY_MASSES = (28.0855, MUON_MASS_AMU, 28.0855)


def _rotation(seed: int) -> np.ndarray:
    """A proper rotation, the Q of a random matrix's QR decomposition."""

    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _toy(muon_block_cm2: np.ndarray, *, coupling_cm2: float = 0.0):
    """The three-atom toy: the muon's own block, the host's, and an optional muon-host spring."""

    matrix = uncoupled_muon_dynamical_matrix(muon_block_cm2, (300.0, 450.0), muon_atom=1)
    matrix[3:6, 0:3] = matrix[0:3, 3:6] = -coupling_cm2 * np.eye(3)
    return phonon_calculation_from_dynamical_matrix(
        matrix,
        lattice_vectors=_TOY_LATTICE,
        species=_TOY_SPECIES,
        fractional_positions=_TOY_POSITIONS,
        masses_amu=_TOY_MASSES,
    ), matrix


def _rotated(calculation, rotation: np.ndarray):
    """The same calculation with its cell and every eigenvector turned by ``rotation``."""

    gamma = calculation.gamma_point()
    turned = np.einsum("ab,mnb->mna", rotation, gamma.eigenvectors)
    return replace(
        calculation,
        lattice_vectors=tuple(
            tuple(row) for row in np.asarray(calculation.lattice_vectors) @ rotation.T
        ),
        qpoints=(replace(gamma, eigenvectors=turned),),
    )


def test_the_gdru2si2_site_2_zpe_along_the_1_to_1_prime_line_is_the_papers() -> None:
    """Huddart et al., PRB 111, 054440, supplement: 0.25 eV along (-1, 1, 0) at site 2."""

    calculation = parse_phonon_path(_GDRU2SI2 / "GdRu2Si2_site2.phonon")

    along = muon_zpe_along(calculation, np.array([-1.0, 1.0, 0.0]) / math.sqrt(2.0))

    assert along.reason == ""
    assert along.zpe_eV == pytest.approx(0.2491, abs=5e-5)
    assert round(along.zpe_eV, 2) == 0.25
    assert along.frequency_cm == pytest.approx(4018.3, abs=0.1)


def test_d0_of_an_uncoupled_toy_gives_the_muon_only_frequencies() -> None:
    frequencies = np.array([1500.0, 2700.0, 4100.0])
    axes = _rotation(3)
    block = axes @ np.diag(frequencies**2) @ axes.T
    calculation, _matrix = _toy(block)

    d0 = muon_dynamical_block_cm2(calculation)

    assert d0 == pytest.approx(block, rel=1e-9, abs=1e-6)
    assert np.sqrt(np.linalg.eigvalsh(d0)) == pytest.approx(frequencies, rel=1e-9)
    for axis, frequency in zip(axes.T, frequencies, strict=True):
        along = muon_zpe_along(calculation, axis)
        assert along.frequency_cm == pytest.approx(frequency, rel=1e-9)
        assert along.zpe_eV == pytest.approx(zpe_from_frequency_cm(frequency), rel=1e-9)
    # The toy's muon modes are pure: each of three modes is all muon.
    assert sorted(muon_mode_participations(calculation))[-3:] == pytest.approx([1.0] * 3)


def test_d0_is_the_muon_block_of_a_coupled_dynamical_matrix_summed_over_every_mode() -> None:
    calculation, matrix = _toy(np.diag([1500.0, 2700.0, 4100.0]) ** 2, coupling_cm2=2.0e6)

    assert muon_dynamical_block_cm2(calculation) == pytest.approx(
        matrix[3:6, 3:6], rel=1e-9, abs=1e-6
    )
    # Coupled, no mode is all muon, so the stored three-mode total differs from D(0).
    assert max(muon_mode_participations(calculation)) < 0.99


def test_the_stored_phonon_files_hold_complete_muon_eigenvector_rows() -> None:
    """D(0) needs sum_n e_mu,n e_mu,n^dagger = 1: CASTEP's orthonormal, mass-weighted modes."""

    for site in (1, 2):
        calculation = parse_phonon_path(_GDRU2SI2 / f"GdRu2Si2_site{site}.phonon")
        muon = calculation.atom_index_for_species("H:mu") - 1
        vectors = calculation.gamma_point().eigenvectors[:, muon, :]
        assert np.einsum("na,nb->ab", vectors, vectors.conj()).real == pytest.approx(
            np.eye(3), abs=1e-6
        )


def test_eigenvectors_that_are_not_complete_give_an_error() -> None:
    calculation, _matrix = _toy(np.diag([1500.0, 2700.0, 4100.0]) ** 2)
    gamma = calculation.gamma_point()
    halved = replace(calculation, qpoints=(replace(gamma, eigenvectors=gamma.eigenvectors * 0.5),))

    with pytest.raises(PhononParseError, match="complete orthonormal"):
        muon_dynamical_block_cm2(halved)


def test_turning_the_cell_and_the_direction_together_keeps_the_zpe() -> None:
    calculation = parse_phonon_path(_GDRU2SI2 / "GdRu2Si2_site2.phonon")
    direction = np.array([0.3, -0.8, 0.5])
    expected = muon_zpe_along(calculation, direction).zpe_eV

    for seed in (1, 2, 5):
        rotation = _rotation(seed)
        turned = muon_zpe_along(_rotated(calculation, rotation), rotation @ direction)
        assert turned.zpe_eV == pytest.approx(expected, rel=1e-9)


def test_the_direction_is_normalised() -> None:
    calculation = parse_phonon_path(_GDRU2SI2 / "GdRu2Si2_site1.phonon")
    unit = np.array([0.0, 1.0, 0.0])

    assert muon_zpe_along(calculation, 3.7 * unit).zpe_eV == pytest.approx(
        muon_zpe_along(calculation, unit).zpe_eV, rel=1e-12
    )
    assert muon_zpe_along(calculation, 1e-3 * unit).zpe_eV == pytest.approx(
        muon_zpe_along(calculation, unit).zpe_eV, rel=1e-12
    )


@pytest.mark.parametrize("site", [1, 2])
def test_the_principal_zpes_of_d0_agree_with_its_eigenvalues(site: int) -> None:
    calculation = parse_phonon_path(_GDRU2SI2 / f"GdRu2Si2_site{site}.phonon")
    d0 = muon_dynamical_block_cm2(calculation)
    eigenvalues, axes = np.linalg.eigh(d0)

    principal = [muon_zpe_along(calculation, axis).zpe_eV for axis in axes.T]

    assert principal == pytest.approx(
        [CM_INV_ZPE_TO_EV * math.sqrt(value) for value in eigenvalues], rel=1e-9
    )
    # Over any orthonormal basis the squared frequencies sum to the trace of D(0).
    basis = _rotation(site)
    squares = [muon_zpe_along(calculation, axis).frequency_cm ** 2 for axis in basis.T]
    assert sum(squares) == pytest.approx(float(np.trace(d0)), rel=1e-9)
    # And no direction lies outside the principal range.
    for axis in _rotation(site + 10).T:
        assert principal[0] - 1e-12 <= muon_zpe_along(calculation, axis).zpe_eV
        assert muon_zpe_along(calculation, axis).zpe_eV <= principal[-1] + 1e-12


def test_a_direction_of_negative_curvature_gives_no_value() -> None:
    """An imaginary muon mode: the muon block has a negative eigenvalue along z."""

    calculation, _matrix = _toy(np.diag([1500.0**2, 2700.0**2, -(900.0**2)]))

    along_z = muon_zpe_along(calculation, (0.0, 0.0, 1.0))
    along_x = muon_zpe_along(calculation, (1.0, 0.0, 0.0))

    assert (along_z.zpe_eV, along_z.frequency_cm) == (None, None)
    assert along_z.reason == NO_REAL_FREQUENCY_ALONG_DIRECTION
    assert along_x.frequency_cm == pytest.approx(1500.0, rel=1e-9)
    # A direction mixing the two gives its own sign: here x dominates.
    assert muon_zpe_along(calculation, (1.0, 0.0, 0.5)).zpe_eV is not None


@pytest.mark.parametrize("direction", [(0.0, 0.0, 0.0), (1.0, float("nan"), 0.0), (1.0, 0.0)])
def test_a_zero_or_malformed_direction_is_rejected(direction) -> None:
    calculation, _matrix = _toy(np.diag([1500.0, 2700.0, 4100.0]) ** 2)

    with pytest.raises(ValueError):
        muon_zpe_along(calculation, direction)


# --- The principal axes of the muon's well -------------------------------------------


@pytest.mark.parametrize(
    ("site", "expected_meV"),
    [(1, (70.2, 199.9, 287.1)), (2, (131.6, 249.1, 272.5))],
)
def test_the_gdru2si2_principal_zpes(site: int, expected_meV) -> None:
    calculation = parse_phonon_path(_GDRU2SI2 / f"GdRu2Si2_site{site}.phonon")

    principal = muon_principal_zpes(calculation)

    assert [entry.zpe_eV * 1000 for entry in principal] == pytest.approx(expected_meV, abs=0.05)
    assert [entry.degenerate_with for entry in principal] == [(), (), ()]
    for entry in principal:
        assert np.linalg.norm(entry.axis) == pytest.approx(1.0)
        assert muon_zpe_along(calculation, entry.axis).zpe_eV == pytest.approx(entry.zpe_eV)


def test_site_2s_249_mev_axis_is_the_papers_direction() -> None:
    calculation = parse_phonon_path(_GDRU2SI2 / "GdRu2Si2_site2.phonon")

    middle = muon_principal_zpes(calculation)[1]

    assert np.abs(np.dot(middle.axis, np.array([-1.0, 1.0, 0.0]) / math.sqrt(2.0))) == (
        pytest.approx(1.0, abs=1e-6)
    )
    # An axis has no sense: its first component that is not zero is positive.
    assert middle.axis[0] > 0.0


def test_the_principal_zpes_turn_with_the_cell() -> None:
    calculation = parse_phonon_path(_GDRU2SI2 / "GdRu2Si2_site2.phonon")
    rotation = _rotation(4)

    before = muon_principal_zpes(calculation)
    after = muon_principal_zpes(_rotated(calculation, rotation))

    assert [entry.zpe_eV for entry in after] == pytest.approx([entry.zpe_eV for entry in before])
    for old, new in zip(before, after, strict=True):
        assert abs(float(np.dot(rotation @ np.asarray(old.axis), new.axis))) == pytest.approx(1.0)


def test_equal_principal_values_are_reported_as_one_level_without_an_axis() -> None:
    uniaxial, _matrix = _toy(np.diag([2000.0**2, 2000.0**2, 4000.0**2]))
    isotropic, _matrix = _toy(np.eye(3) * 3000.0**2)

    pair = muon_principal_zpes(uniaxial)
    level = muon_principal_zpes(isotropic)

    assert [entry.degenerate_with for entry in pair] == [(1,), (0,), ()]
    assert (pair[0].axis, pair[1].axis) == (None, None)
    assert pair[2].axis == pytest.approx((0.0, 0.0, 1.0))
    assert [entry.degenerate_with for entry in level] == [(1, 2), (0, 2), (0, 1)]
    assert [entry.axis for entry in level] == [None, None, None]
    assert [entry.frequency_cm for entry in level] == pytest.approx([3000.0] * 3)


def test_a_principal_axis_of_negative_curvature_gives_no_value() -> None:
    calculation, _matrix = _toy(np.diag([1500.0**2, 2700.0**2, -(900.0**2)]))

    lowest, middle, highest = muon_principal_zpes(calculation)

    assert (lowest.zpe_eV, lowest.frequency_cm) == (None, None)
    assert lowest.reason == NO_REAL_FREQUENCY_ALONG_AXIS
    assert lowest.curvature_cm2 == pytest.approx(-(900.0**2))
    assert lowest.axis == pytest.approx((0.0, 0.0, 1.0))
    assert (middle.frequency_cm, highest.frequency_cm) == pytest.approx((1500.0, 2700.0))
    assert (middle.reason, highest.reason) == ("", "")


def test_close_principal_values_chain_into_one_level() -> None:
    """Each neighbour is within the tolerance though the ends are not: one level of three."""

    step = 0.0009 * 4000.0**2
    calculation, _matrix = _toy(np.diag([4000.0**2 - 2 * step, 4000.0**2 - step, 4000.0**2]))

    principal = muon_principal_zpes(calculation)

    assert [entry.degenerate_with for entry in principal] == [(1, 2), (0, 2), (0, 1)]


def test_values_of_opposite_sign_are_never_one_level() -> None:
    calculation, _matrix = _toy(np.diag([-200.0, 300.0, 1000.0**2]))

    principal = muon_principal_zpes(calculation)

    assert [entry.degenerate_with for entry in principal] == [(), (), ()]
    assert principal[0].zpe_eV is None
    assert principal[1].frequency_cm == pytest.approx(math.sqrt(300.0))


def _neighbour(
    atom: int, species: str, distance: float, correlation: float, shell: int
) -> MuonNeighbour:
    return MuonNeighbour(
        atom_index=atom,
        species=species,
        distance_ang=distance,
        correlation=correlation,
        distance_shell=shell,
    )


# Out of index order, with a distance shell (1, 3, 9, 0.0004 Å apart), a |c| tie (2, 7),
# a species tie and a missing |c| (5).
NEIGHBOURS = (
    _neighbour(9, "N", 2.0004, 0.30, 1),
    _neighbour(2, "Si", 3.5, 0.10, 2),
    _neighbour(5, "N", 1.0, math.nan, 0),
    _neighbour(7, "gd", 4.0, 0.10, 3),
    _neighbour(3, "N", 2.0, 0.20, 1),
    _neighbour(1, "Si", 2.0002, 0.05, 1),
)


def _indices(neighbours) -> list[int]:
    return [neighbour.atom_index for neighbour in neighbours]


@pytest.mark.parametrize(
    ("key", "ascending", "descending"),
    [
        ("atom", [1, 2, 3, 5, 7, 9], [9, 7, 5, 3, 2, 1]),
        # Names ignore case; atoms of one species keep index order both ways.
        ("species", [7, 3, 5, 9, 1, 2], [1, 2, 3, 5, 9, 7]),
        # Atoms 1, 3 and 9 are one shell, so they stay in index order both ways.
        ("distance", [5, 1, 3, 9, 2, 7], [7, 2, 1, 3, 9, 5]),
        # Atoms 2 and 7 tie; the missing |c| of atom 5 goes last both ways.
        ("correlation", [1, 2, 7, 3, 9, 5], [9, 3, 2, 7, 1, 5]),
    ],
)
def test_the_neighbours_order_by_a_column_with_ties_in_index_order(
    key: str, ascending: list[int], descending: list[int]
) -> None:
    assert _indices(order_muon_neighbours(NEIGHBOURS, key)) == ascending
    assert _indices(order_muon_neighbours(NEIGHBOURS, key, descending=True)) == descending


def test_ordering_the_neighbours_changes_no_value() -> None:
    for key in ("atom", "species", "distance", "correlation"):
        for descending in (False, True):
            ordered = order_muon_neighbours(NEIGHBOURS, key, descending=descending)
            assert sorted(ordered, key=lambda n: n.atom_index) == sorted(
                NEIGHBOURS, key=lambda n: n.atom_index
            )


def test_ordering_by_distance_gives_the_nearest_atom_order() -> None:
    calculation = _n2_mu_n2()
    neighbours = muon_nearest_atoms(calculation, 12)

    assert order_muon_neighbours(reversed(neighbours), "distance") == neighbours


def _shell_cut_by_the_count():
    """Atoms 4, 5 and 2 at 1.2900, 1.2907 and 1.2912 Å: one shell, chained in 0.0007 and
    0.0005 Å steps, though 4 and 2 are 0.0012 Å apart. Atom 1 is far off."""

    calculation = _n2_mu_n2()
    muon = next(atom for atom in calculation.atoms if atom.index == MUON_ATOM_INDEX)
    mx, my, mz = muon.fractional_coordinates
    place = {
        2: (mx - 1.2912 / BOX_EDGE_ANG, my, mz),
        4: (mx + 1.2900 / BOX_EDGE_ANG, my, mz),
        5: (mx, my + 1.2907 / BOX_EDGE_ANG, mz),
        1: (mx, my - 3.0 / BOX_EDGE_ANG, mz),
    }
    return replace(
        calculation,
        atoms=tuple(
            replace(atom, fractional_coordinates=place.get(atom.index, atom.fractional_coordinates))
            for atom in calculation.atoms
        ),
    )


def test_ordering_by_distance_keeps_the_shells_of_every_atom_when_the_count_cuts_one() -> None:
    calculation = _shell_cut_by_the_count()
    assert _indices(muon_nearest_atoms(calculation, 4)) == [2, 4, 5, 1]
    reversed_shells = {1: [2], 2: [2, 4], 3: [2, 4, 5], 4: [1, 2, 4, 5]}

    for count in range(1, 5):
        nearest = muon_nearest_atoms(calculation, count)
        # Atoms 2 and 4 are 0.0012 Å apart, but atom 5, cut off at a count of 2,
        # ties them: they stay one shell, in index order, both ways.
        assert order_muon_neighbours(reversed(nearest), "distance") == nearest
        assert (
            _indices(order_muon_neighbours(nearest, "distance", descending=True))
            == (reversed_shells[count])
        )


def test_the_neighbours_cannot_be_ordered_by_an_unknown_column() -> None:
    with pytest.raises(ValueError, match="Cannot order"):
        order_muon_neighbours(NEIGHBOURS, "mass")
