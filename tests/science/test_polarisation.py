"""Tests for muon-spin polarisation functions, powder averages and spectra."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.magnetic_fields import (
    MagneticFieldDistributionResult,
    MagneticFieldResult,
    calculate_incommensurate_field_distributions,
)
from mufinder.science.nuclear_relaxation import calculate_nuclear_width_tensors
from mufinder.science.polarisation import (
    MAX_MAGNITUDE_BINS,
    PolarisationError,
    PolarisationResult,
    calculate_lf_polarisation,
    calculate_polarisation,
    calculate_spectrum,
    calculate_tf_polarisation,
    fibonacci_sphere,
)
from tests.support.codata import MUON_GAMMA_RAD_S_T
from tests.support.mnsi import MNSI_MUON, mnsi_helix, mnsi_options, mnsi_structure

GAMMA = MUON_GAMMA_RAD_S_T * 1e-6  # rad μs⁻¹ T⁻¹
ZERO = (0.0, 0.0, 0.0)
TIMES = np.linspace(0.0, 10.0, 1001)


def _field(total, applied=ZERO, label: str = "site 1") -> MagneticFieldResult:
    """A field result whose total is ``total`` (T), carrying ``applied`` within it."""

    internal = tuple(float(t) - float(a) for t, a in zip(total, applied, strict=True))
    return MagneticFieldResult(label, ZERO, internal, ZERO, ZERO, ZERO, tuple(applied))


def _distribution(fields, applied=None, label: str = "site 1") -> MagneticFieldDistributionResult:
    return MagneticFieldDistributionResult(
        site_label=label,
        muon_position=ZERO,
        fields_T=np.asarray(fields, dtype=float),
        histogram_centres_T=np.zeros(1),
        histogram_counts=np.zeros(1),
        applied_T=applied,
    )


def _samples(count: int = 10_000, seed: int = 254) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(count, 3)) * 0.02 + np.array([0.01, 0.05, 0.03])


def _direct(fields, times, spin, detector, *, lambda_z=0.0, sigma=0.0) -> np.ndarray:
    """The kernel summed sample by sample, the reference for the histogram evaluation."""

    spin = np.asarray(spin, dtype=float) / np.linalg.norm(spin)
    detector = np.asarray(detector, dtype=float) / np.linalg.norm(detector)
    magnitudes = np.linalg.norm(fields, axis=1)
    directions = fields / magnitudes[:, None]
    constant = (directions @ detector) * (directions @ spin)
    cosine = spin @ detector - constant
    sine = np.einsum("j,ij->i", detector, np.cross(spin, directions))
    phase = np.outer(GAMMA * magnitudes, times)
    oscillation = np.mean(cosine[:, None] * np.cos(phase) + sine[:, None] * np.sin(phase), axis=0)
    return np.mean(constant) * np.exp(-lambda_z * times) + oscillation * np.exp(
        -0.5 * sigma**2 * times**2
    )


# --- The single-field kernel -----------------------------------------------------------


def test_a_field_along_the_spin_only_relaxes_longitudinally() -> None:
    result = calculate_polarisation(_field((0.0, 0.0, 0.1)), TIMES, lambda_z=0.3, sigma_nuclear=0.5)

    np.testing.assert_allclose(result.polarisation, np.exp(-0.3 * TIMES), atol=1e-6)
    assert result.label == "P"


def test_a_field_across_the_spin_gives_a_damped_cosine_at_the_larmor_frequency() -> None:
    result = calculate_polarisation(_field((0.1, 0.0, 0.0)), TIMES, lambda_z=0.3, sigma_nuclear=0.2)
    spectrum = calculate_spectrum(TIMES, result.polarisation, padding_factor=8)

    expected = np.cos(GAMMA * 0.1 * TIMES) * np.exp(-0.5 * 0.2**2 * TIMES**2)
    np.testing.assert_allclose(result.polarisation, expected, atol=1e-6)
    peak = spectrum.frequencies_MHz[np.argmax(spectrum.amplitudes)]
    assert peak == pytest.approx(GAMMA * 0.1 / (2.0 * np.pi), abs=0.02)
    assert spectrum.fields_T[np.argmax(spectrum.amplitudes)] == pytest.approx(0.1, abs=2e-4)


def test_a_detector_across_the_spin_and_the_field_sees_a_pure_sine() -> None:
    result = calculate_polarisation(
        _field((0.0, 0.0, 0.1)),
        TIMES,
        polarisation_direction=(1.0, 0.0, 0.0),
        detector_direction=(0.0, 1.0, 0.0),
    )

    # dS/dt = γ S × B turns a spin along x towards −y in a field along z.
    np.testing.assert_allclose(result.polarisation, -np.sin(GAMMA * 0.1 * TIMES), atol=1e-6)


def test_a_vanishing_field_leaves_the_projection_of_the_spin_on_the_detector() -> None:
    result = calculate_polarisation(
        _field(ZERO), TIMES, polarisation_direction=(1.0, 0.0, 1.0), sigma_nuclear=0.1
    )

    np.testing.assert_allclose(result.polarisation, np.exp(-0.005 * TIMES**2), atol=1e-6)


# --- Distributions, sites and frames ---------------------------------------------------


@pytest.mark.parametrize("times", [TIMES, np.linspace(0.0, 30.0, 601)])
def test_histogram_evaluation_matches_the_direct_sum(times: np.ndarray) -> None:
    fields = _samples()
    result = calculate_polarisation(
        _distribution(fields),
        times,
        polarisation_direction=(1.0, 1.0, 0.0),
        detector_direction=(0.0, 1.0, 1.0),
        lambda_z=0.1,
        sigma_nuclear=0.1,
    )
    direct = _direct(fields, times, (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), lambda_z=0.1, sigma=0.1)

    np.testing.assert_allclose(result.polarisation, direct, atol=1e-4)
    assert result.parameters["magnitude_bins"] >= 2048
    assert result.parameters["max_in_bin_phase_rad"] <= 0.1


def test_the_bin_count_stops_at_its_limit_and_records_the_phase_it_allows() -> None:
    # 1 T of spread over 100 μs would need about 425 000 bins for 0.1 rad.
    fields = np.array([[0.0, 0.0, 0.01], [0.0, 0.0, 1.01]])
    times = np.linspace(0.0, 100.0, 11)

    result = calculate_polarisation(_distribution(fields), times, polarisation_direction=(1, 0, 0))

    assert result.parameters["magnitude_bins"] == MAX_MAGNITUDE_BINS
    assert result.parameters["max_in_bin_phase_rad"] > 0.1


def test_sites_are_averaged_with_their_weights() -> None:
    sites = [_field((0.0, 0.0, 0.1), label="along"), _field((0.1, 0.0, 0.0), label="across")]

    equal = calculate_polarisation(sites, TIMES)
    weighted = calculate_polarisation(sites, TIMES, site_weights=(3.0, 1.0))

    np.testing.assert_allclose(equal.polarisation, equal.site_polarisations.mean(axis=0))
    np.testing.assert_allclose(
        weighted.polarisation, 0.75 + 0.25 * np.cos(GAMMA * 0.1 * TIMES), atol=1e-6
    )
    assert weighted.site_labels == ("along", "across")
    np.testing.assert_allclose(weighted.site_weights, (0.75, 0.25))


def test_the_crystal_rotation_turns_the_fields_into_the_laboratory_frame() -> None:
    fields = _samples(2000)
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    options = {"polarisation_direction": (1.0, 0.0, 0.2), "detector_direction": (0.0, 1.0, 0.0)}

    rotated = calculate_polarisation(
        _distribution(fields), TIMES, crystal_rotation=rotation, **options
    )
    in_lab = calculate_polarisation(_distribution(fields @ rotation.T), TIMES, **options)

    np.testing.assert_allclose(rotated.polarisation, in_lab.polarisation, atol=1e-6)


def test_auto_nuclear_width_damps_each_field_direction_with_its_own_tf_width() -> None:
    host = Structure(
        label="H",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]],
        species=["H"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    tensors = calculate_nuclear_width_tensors(host, (0.3, 0.4, 0.2), radius=10.0)
    sites = [_field((0.1, 0.0, 0.0), label="x"), _field((0.0, 0.1, 0.0), label="y")]

    result = calculate_polarisation(
        sites, TIMES, sigma_nuclear="auto", nuclear_width_tensors=[tensors] * 2
    )

    for row, axis in zip(result.site_polarisations, np.eye(3)[:2], strict=True):
        sigma_squared = tensors.tf_sigma_squared(axis)
        # Both fields lie across the spin, so each site is a damped cosine.
        expected = np.cos(GAMMA * 0.1 * TIMES) * np.exp(-0.5 * sigma_squared * TIMES**2)
        np.testing.assert_allclose(row, expected, atol=1e-5)
    assert result.parameters["sigma_nuclear"] == "auto"


def test_auto_nuclear_width_damps_each_sample_even_within_one_magnitude_bin() -> None:
    # Fields of one size in every direction share a |B| bin but not a nuclear width.
    host = Structure(
        label="H",
        lattice_vectors=[[3.4, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 5.5]],
        species=["H"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    # Midway between two protons 3.4 Å apart: σ² runs from 0.0005 to 0.48 μs⁻².
    tensors = calculate_nuclear_width_tensors(host, (0.5, 0.0, 0.0), radius=10.0)
    directions = fibonacci_sphere(4000)
    fields = 0.1 * directions
    times = np.linspace(0.0, 4.0, 401)
    widths = tensors.tf_sigma_squared(directions)
    assert widths.max() > 100.0 * widths.min()

    result = calculate_polarisation(
        _distribution(fields), times, sigma_nuclear="auto", nuclear_width_tensors=[tensors]
    )

    # The kernel sample by sample, each with its own width.
    along = directions[:, 2]
    oscillation = (
        (1.0 - along**2)[:, None]
        * np.cos(GAMMA * 0.1 * times)
        * np.exp(-0.5 * np.outer(widths, times**2))
    )
    direct = np.mean(along**2) + np.mean(oscillation, axis=0)
    np.testing.assert_allclose(result.polarisation, direct, atol=1e-4)
    # One mean width per bin would damp the trace differently.
    lumped = np.mean(along**2) + np.mean(1.0 - along**2) * np.cos(GAMMA * 0.1 * times) * np.exp(
        -0.5 * np.average(widths, weights=1.0 - along**2) * times**2
    )
    assert np.max(np.abs(lumped - direct)) > 1e-2
    assert result.parameters["width_levels"] > 1


# --- Powder averages -----------------------------------------------------------------


def test_zero_field_powder_is_one_third_plus_two_thirds_per_sample() -> None:
    fields = _samples(5000)
    magnitudes = np.linalg.norm(fields, axis=1)

    powder = calculate_polarisation(
        _distribution(fields), TIMES, polycrystalline=True, lambda_z=0.2, sigma_nuclear=0.1
    )
    crossed = calculate_polarisation(
        _distribution(fields),
        TIMES,
        polycrystalline=True,
        polarisation_direction=(1.0, 0.0, 0.0),
        detector_direction=(0.0, 1.0, 0.0),
    )

    oscillation = np.mean(np.cos(np.outer(GAMMA * magnitudes, TIMES)), axis=0)
    expected = np.exp(-0.2 * TIMES) / 3.0 + 2.0 / 3.0 * oscillation * np.exp(-0.005 * TIMES**2)
    np.testing.assert_allclose(powder.polarisation, expected, atol=1e-5)
    # With the detector across the spin a powder shows nothing: the sine term averages away.
    np.testing.assert_allclose(crossed.polarisation, 0.0, atol=1e-6)
    assert powder.polycrystalline


def test_longitudinal_field_powder_tends_to_zero_field_and_to_full_decoupling() -> None:
    fields = _samples(2000)
    zero_field = calculate_polarisation(_distribution(fields), TIMES, polycrystalline=True)

    weak = calculate_lf_polarisation(
        _distribution(fields), TIMES, applied_field=(0.0, 0.0, 1e-7), polycrystalline=True
    )
    strong = calculate_lf_polarisation(
        _distribution(fields),
        TIMES,
        applied_field=(0.0, 0.0, 100.0),
        polycrystalline=True,
        lambda_z=0.2,
    )

    np.testing.assert_allclose(weak.polarisation, zero_field.polarisation, atol=2e-3)
    np.testing.assert_allclose(strong.polarisation, np.exp(-0.2 * TIMES), atol=1e-3)
    assert strong.label == "P_z"
    assert strong.polarisation_direction == pytest.approx((0.0, 0.0, 1.0))
    assert strong.applied_field_T == (0.0, 0.0, 100.0)


def test_transverse_field_powder_precesses_at_the_applied_field() -> None:
    weak_internal = _samples(500) * 1e-6

    result = calculate_tf_polarisation(
        _distribution(weak_internal), TIMES, applied_field=(0.0, 0.0, 0.02), polycrystalline=True
    )

    np.testing.assert_allclose(result.polarisation, np.cos(GAMMA * 0.02 * TIMES), atol=1e-4)
    assert result.label == "P_x"
    assert result.polarisation_direction == pytest.approx((1.0, 0.0, 0.0))
    assert result.detector_direction == result.polarisation_direction


def test_the_applied_field_in_a_result_equals_one_passed_with_a_zero_field_result() -> None:
    fields = _samples(1000)
    applied = (0.0, 0.0, 0.05)

    carried = calculate_lf_polarisation(
        _distribution(fields + np.asarray(applied), applied=applied),
        TIMES,
        polycrystalline=True,
        sphere_points=100,
    )
    added = calculate_lf_polarisation(
        _distribution(fields), TIMES, applied_field=applied, polycrystalline=True, sphere_points=100
    )

    np.testing.assert_allclose(carried.polarisation, added.polarisation, atol=1e-6)


def test_a_transverse_spin_may_be_chosen_across_the_field() -> None:
    result = calculate_tf_polarisation(
        _field((0.1, 0.0, 0.0), applied=(0.1, 0.0, 0.0)),
        TIMES,
        polarisation_direction=(0.0, 0.0, 2.0),
    )

    np.testing.assert_allclose(result.polarisation, np.cos(GAMMA * 0.1 * TIMES), atol=1e-6)
    assert result.polarisation_direction == (0.0, 0.0, 1.0)


def test_fibonacci_sphere_points_are_unit_vectors_averaging_to_zero() -> None:
    points = fibonacci_sphere(600)

    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0)
    np.testing.assert_allclose(points.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(points.T @ points / 600, np.eye(3) / 3.0, atol=1e-3)
    with pytest.raises(PolarisationError, match="sphere_points"):
        fibonacci_sphere(0)


# --- Validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"applied_field": (0.0, 0.0, 0.1)}, "already include an applied field"),
        ({"lambda_z": -1.0}, "lambda_z"),
        ({"sigma_nuclear": -1.0}, "sigma_nuclear"),
        ({"sigma_nuclear": "zf"}, "sigma_nuclear"),
        ({"sigma_nuclear": "auto"}, "nuclear_width_tensors"),
        ({"polarisation_direction": ZERO}, "non-zero"),
        ({"detector_direction": (1.0, 0.0)}, "three finite"),
        ({"crystal_rotation": np.diag((1.0, 1.0, -1.0))}, "rotation"),
        ({"crystal_rotation": np.ones((3, 3))}, "rotation"),
        ({"site_weights": (1.0, 2.0)}, "one weight per"),
        ({"site_weights": (-1.0,)}, "non-negative"),
        ({"magnitude_bins": 0}, "magnitude_bins"),
        ({"sphere_points": 0}, "sphere_points"),
    ],
)
def test_bad_options_are_refused(options: dict, message: str) -> None:
    with pytest.raises(PolarisationError, match=message):
        calculate_polarisation(_field((0.1, 0.0, 0.1), applied=(0.0, 0.0, 0.1)), TIMES, **options)


def test_bad_inputs_are_refused() -> None:
    with pytest.raises(PolarisationError, match="no field results"):
        calculate_polarisation([], TIMES)
    with pytest.raises(PolarisationError, match="expected magnetic-field results"):
        calculate_polarisation([{"total_T": [0, 0, 1]}], TIMES)
    with pytest.raises(PolarisationError, match="times_us"):
        calculate_polarisation(_field((0.0, 0.0, 0.1)), np.zeros((2, 2)))
    with pytest.raises(PolarisationError, match="no field samples"):
        calculate_polarisation(_distribution(np.zeros((0, 3))), TIMES)
    with pytest.raises(PolarisationError, match="different applied fields"):
        calculate_polarisation(
            [_field((0.0, 0.0, 0.1), applied=(0.0, 0.0, 0.1)), _field((0.0, 0.0, 0.1))], TIMES
        )
    with pytest.raises(PolarisationError, match="three finite"):
        calculate_polarisation(_field((0.0, 0.0, 0.1)), TIMES, applied_field=(0.0, 0.1))
    with pytest.raises(PolarisationError, match="needs an applied field"):
        calculate_lf_polarisation(_field((0.0, 0.0, 0.1)), TIMES)
    with pytest.raises(PolarisationError, match="P_z sets detector_direction"):
        calculate_lf_polarisation(
            _field((0.0, 0.0, 0.1), applied=(0.0, 0.0, 0.1)),
            TIMES,
            detector_direction=(1.0, 0.0, 0.0),
        )
    with pytest.raises(PolarisationError, match="P_x sets label"):
        calculate_tf_polarisation(
            _field((0.0, 0.0, 0.1), applied=(0.0, 0.0, 0.1)), TIMES, label="P"
        )
    with pytest.raises(PolarisationError, match="perpendicular"):
        calculate_tf_polarisation(
            _field((0.0, 0.0, 0.1)),
            TIMES,
            applied_field=(0.0, 0.0, 0.1),
            polarisation_direction=(0.0, 1.0, 1.0),
        )


def test_a_distribution_saved_without_its_applied_field_counts_as_zero_field() -> None:
    result = calculate_polarisation(
        _distribution([(0.1, 0.0, 0.0)]), TIMES, applied_field=(0.0, 0.0, 0.0)
    )

    np.testing.assert_allclose(result.polarisation, np.cos(GAMMA * 0.1 * TIMES), atol=1e-6)


# --- Persistence ---------------------------------------------------------------------


def test_a_result_round_trips_through_json() -> None:
    result = calculate_polarisation(
        [_field((0.0, 0.05, 0.1), label="I"), _distribution(_samples(100), label="II")],
        TIMES,
        site_weights=(1.0, 3.0),
        lambda_z=0.1,
    )

    restored = PolarisationResult.from_json(json.loads(json.dumps(result.to_json())))

    np.testing.assert_allclose(restored.polarisation, result.polarisation)
    np.testing.assert_allclose(restored.site_polarisations, result.site_polarisations)
    np.testing.assert_allclose(restored.site_weights, result.site_weights)
    assert restored.site_labels == ("I", "II")
    assert restored.parameters == result.parameters
    assert restored.polarisation_direction == result.polarisation_direction
    with pytest.raises(PolarisationError, match="malformed"):
        PolarisationResult.from_json({"times_us": [0.0]})


# --- Spectra -------------------------------------------------------------------------


def test_the_spectrum_integrates_to_the_oscillating_fraction() -> None:
    trace = 1.0 / 3.0 + 2.0 / 3.0 * np.cos(2.0 * np.pi * 5.0 * TIMES) * np.exp(-0.05 * TIMES)

    spectrum = calculate_spectrum(TIMES, trace, apodisation_sigma_us=4.0, padding_factor=4)
    raw = calculate_spectrum(TIMES, trace, subtract_baseline=False, apodisation_lambda_us=3.0)

    step = spectrum.frequencies_MHz[1] - spectrum.frequencies_MHz[0]
    baseline = np.mean(trace[-101:])
    assert np.sum(spectrum.amplitudes) * step == pytest.approx(trace[0] - baseline)
    assert spectrum.frequencies_MHz[np.argmax(spectrum.amplitudes)] == pytest.approx(5.0, abs=0.03)
    assert spectrum.frequencies_MHz.size == 4 * 1001 // 2 + 1
    np.testing.assert_allclose(
        spectrum.fields_T * GAMMA / (2.0 * np.pi), spectrum.frequencies_MHz, rtol=1e-8
    )
    # Without the baseline subtracted the constant third shows as a line at zero frequency.
    assert np.argmax(raw.amplitudes) == 0
    assert np.iscomplexobj(raw.spectrum)


def test_a_flat_trace_has_an_empty_spectrum() -> None:
    spectrum = calculate_spectrum(TIMES, np.ones_like(TIMES))

    np.testing.assert_array_equal(spectrum.amplitudes, 0.0)


@pytest.mark.parametrize(
    ("times", "values", "options", "message"),
    [
        (TIMES, TIMES[:-1], {}, "same length"),
        (TIMES**2, TIMES, {}, "uniform"),
        (TIMES[::-1], TIMES, {}, "uniform"),
        (TIMES, np.where(TIMES > 5.0, np.nan, TIMES), {}, "finite"),
        (TIMES, TIMES, {"padding_factor": 0}, "padding_factor"),
        (TIMES, TIMES, {"padding_factor": 1.5}, "padding_factor"),
        (TIMES, TIMES, {"apodisation_sigma_us": 0.0}, "apodisation_sigma_us"),
        (TIMES, TIMES, {"apodisation_lambda_us": -1.0}, "apodisation_lambda_us"),
    ],
)
def test_bad_spectrum_inputs_are_refused(times, values, options: dict, message: str) -> None:
    with pytest.raises(PolarisationError, match=message):
        calculate_spectrum(times, values, **options)


# --- MnSi, the incommensurate-helix worked example (tests/support/mnsi.py) ------------


def test_worked_example_mnsi_spectrum_has_its_two_edges_at_90_and_207_mT(real_muesr) -> None:
    sites = calculate_incommensurate_field_distributions(
        mnsi_structure(), mnsi_helix(), [MNSI_MUON], options=mnsi_options()
    )
    times = np.linspace(0.0, 16.0, 4001)

    result = calculate_polarisation(sites, times, polycrystalline=True)
    spectrum = calculate_spectrum(
        times, result.polarisation, apodisation_sigma_us=6.0, padding_factor=4
    )

    fields_mT = spectrum.fields_T * 1e3
    amplitudes = spectrum.amplitudes
    step = spectrum.frequencies_MHz[1] - spectrum.frequencies_MHz[0]
    assert np.sum(amplitudes) * step == pytest.approx(
        result.polarisation[0] - np.mean(result.polarisation[-401:]), rel=1e-9
    )
    # The strongest peak is the lower edge at 90 mT, with site I's line at 88 mT beside it.
    assert fields_mT[np.argmax(amplitudes)] == pytest.approx(89.0, abs=2.0)
    # The upper edge, at 207 mT, is the second horn.
    upper = (fields_mT > 150.0) & (fields_mT < 250.0)
    assert fields_mT[upper][np.argmax(amplitudes[upper])] == pytest.approx(206.8, abs=2.0)
    # Outside the two edges only the window's wings remain.
    outside = (fields_mT < 70.0) | (fields_mT > 230.0)
    assert np.max(amplitudes[outside]) < 0.02 * np.max(amplitudes)
