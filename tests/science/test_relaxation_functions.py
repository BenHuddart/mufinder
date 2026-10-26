"""Tests for the reference muon-spin relaxation functions."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from mufinder.science import relaxation_functions
from mufinder.science.polarisation import PolarisationError
from mufinder.science.relaxation_functions import (
    RELAXATION_FUNCTIONS,
    abragam_relaxation,
    dynamic_lorentzian_kubo_toyabe,
    keren_lf_relaxation,
    static_gaussian_kubo_toyabe,
    static_gaussian_kubo_toyabe_lf,
    static_lorentzian_kubo_toyabe,
)
from tests.support.codata import MUON_GAMMA_RAD_S_T

GAMMA = MUON_GAMMA_RAD_S_T * 1e-6  # rad μs⁻¹ T⁻¹
TIMES = np.linspace(0.0, 20.0, 2001)

# Example arguments for each library function, after the times.
EXAMPLE_ARGUMENTS = {
    "static_gaussian_kubo_toyabe": {"delta_per_us": 0.5},
    "static_gaussian_kubo_toyabe_lf": {"delta_per_us": 0.5, "field_T": 0.002},
    "static_lorentzian_kubo_toyabe": {"a_per_us": 0.5},
    "dynamic_lorentzian_kubo_toyabe": {"a_per_us": 0.5, "collision_rate_per_us": 1.0},
    "keren_lf_relaxation": {"delta_per_us": 0.5, "collision_rate_per_us": 2.0, "field_T": 0.002},
    "abragam_relaxation": {"delta_per_us": 0.5, "collision_rate_per_us": 2.0},
}


def _lf_trapezoid(times: np.ndarray, delta: float, field_T: float) -> np.ndarray:
    omega = GAMMA * field_T
    gaussian = np.exp(-((delta * times) ** 2) / 2.0)
    integral = cumulative_trapezoid(gaussian * np.sin(omega * times), times, initial=0.0)
    return (
        1.0
        - 2.0 * delta**2 / omega**2 * (1.0 - gaussian * np.cos(omega * times))
        + 2.0 * delta**4 / omega**3 * integral
    )


def test_every_library_entry_is_described_and_starts_at_one() -> None:
    assert set(RELAXATION_FUNCTIONS) == set(EXAMPLE_ARGUMENTS)
    for name, entry in RELAXATION_FUNCTIONS.items():
        assert entry.name == name
        assert entry.scenario and entry.assumptions and entry.references
        signature = list(inspect.signature(entry.function).parameters)
        assert tuple(signature[1:]) == entry.parameters
        values = entry.function(TIMES, **EXAMPLE_ARGUMENTS[name])
        assert values.shape == TIMES.shape
        assert values[0] == pytest.approx(1.0, abs=1e-12)


def test_static_gaussian_kubo_toyabe_dips_and_recovers_to_one_third() -> None:
    delta = 0.4
    values = static_gaussian_kubo_toyabe(TIMES, delta)
    dip = static_gaussian_kubo_toyabe(np.sqrt(3.0) / delta, delta)

    assert float(dip) == pytest.approx(1.0 / 3.0 - 4.0 / 3.0 * np.exp(-1.5))
    assert np.min(values) == pytest.approx(float(dip), abs=1e-4)
    assert values[-1] == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_lf_kubo_toyabe_in_zero_field_is_the_zero_field_function() -> None:
    np.testing.assert_allclose(
        static_gaussian_kubo_toyabe_lf(TIMES, 0.5, 0.0),
        static_gaussian_kubo_toyabe(TIMES, 0.5),
        atol=1e-15,
    )


@pytest.mark.parametrize("field_T", [0.001, 0.005, 0.02])
def test_lf_kubo_toyabe_matches_a_fine_trapezoid(field_T: float) -> None:
    times = np.linspace(0.0, 10.0, 200_001)

    np.testing.assert_allclose(
        static_gaussian_kubo_toyabe_lf(times, 1.0, field_T),
        _lf_trapezoid(times, 1.0, field_T),
        atol=1e-8,
    )


def test_lf_kubo_toyabe_is_independent_of_the_time_grid() -> None:
    coarse = np.linspace(0.0, 10.0, 11)
    fine = np.linspace(0.0, 10.0, 10_001)

    np.testing.assert_array_equal(
        static_gaussian_kubo_toyabe_lf(coarse, 1.0, 0.005),
        static_gaussian_kubo_toyabe_lf(fine, 1.0, 0.005)[::1000],
    )


def test_lf_kubo_toyabe_is_continuous_across_its_low_field_series() -> None:
    # Below ω/Δ = 0.05 the function uses its power series in ω/Δ.
    delta = 1.0
    field_T = 0.05 * delta / GAMMA

    below = static_gaussian_kubo_toyabe_lf(TIMES, delta, field_T * (1.0 - 1e-9))
    above = static_gaussian_kubo_toyabe_lf(TIMES, delta, field_T * (1.0 + 1e-9))

    np.testing.assert_allclose(below, above, atol=1e-10)


def test_lf_kubo_toyabe_decouples_in_a_strong_field_and_ignores_the_field_sign() -> None:
    np.testing.assert_allclose(static_gaussian_kubo_toyabe_lf(TIMES, 0.5, 1.0), 1.0, atol=1e-4)
    np.testing.assert_array_equal(
        static_gaussian_kubo_toyabe_lf(TIMES, 0.5, -0.003),
        static_gaussian_kubo_toyabe_lf(TIMES, 0.5, 0.003),
    )


def test_static_lorentzian_kubo_toyabe_dips_at_two_over_a() -> None:
    a = 0.5
    dip = static_lorentzian_kubo_toyabe(2.0 / a, a)

    assert float(dip) == pytest.approx(1.0 / 3.0 - 2.0 / 3.0 * np.exp(-2.0))
    assert static_lorentzian_kubo_toyabe(TIMES, a)[-1] == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_dynamic_lorentzian_tends_to_the_static_function_at_low_rates() -> None:
    np.testing.assert_array_equal(
        dynamic_lorentzian_kubo_toyabe(TIMES, 0.5, 0.0), static_lorentzian_kubo_toyabe(TIMES, 0.5)
    )
    # Just above the static fallback the three-exponential form is still accurate.
    np.testing.assert_allclose(
        dynamic_lorentzian_kubo_toyabe(TIMES, 0.5, 1e-7),
        static_lorentzian_kubo_toyabe(TIMES, 0.5),
        atol=1e-5,
    )


def test_dynamic_lorentzian_decays_to_zero_with_collisions() -> None:
    values = dynamic_lorentzian_kubo_toyabe(np.linspace(0.0, 200.0, 2001), 0.5, 1.0)

    assert abs(values[-1]) < 1e-3
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("ratio", [1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0, 10.0, 1e3])
def test_explicit_cubic_solve_matches_numpy_roots(ratio: float) -> None:
    a, nu = 1.0, ratio
    coefficients = (2.0 * a - nu, a * a - 2.0 * a * nu / 3.0, -nu * a * a / 3.0)

    roots = relaxation_functions._cubic_roots(*coefficients)
    expected = np.roots([1.0, *coefficients])

    np.testing.assert_allclose(
        np.sort_complex(roots), np.sort_complex(expected), rtol=1e-6, atol=1e-9
    )


def test_explicit_cubic_solve_finds_three_real_roots() -> None:
    # (u − 1)(u − 2)(u + 3): the trigonometric branch.
    roots = relaxation_functions._cubic_roots(0.0, -7.0, 6.0)

    np.testing.assert_allclose(np.sort(roots.real), [-3.0, 1.0, 2.0], atol=1e-12)
    np.testing.assert_array_equal(roots.imag, 0.0)


def test_keren_in_zero_field_is_abragam() -> None:
    np.testing.assert_array_equal(
        keren_lf_relaxation(TIMES, 0.5, 2.0, 0.0), abragam_relaxation(TIMES, 0.5, 2.0)
    )


def test_keren_without_fluctuations_is_the_static_lf_gaussian_limit() -> None:
    # ν = 0: Γt = 2Δ²(1 − cos ωt)/ω².
    delta, field_T = 0.5, 0.01
    omega = GAMMA * field_T

    np.testing.assert_allclose(
        keren_lf_relaxation(TIMES, delta, 0.0, field_T),
        np.exp(-2.0 * delta**2 * (1.0 - np.cos(omega * TIMES)) / omega**2),
        rtol=1e-7,
    )


def test_abragam_limits() -> None:
    delta = 0.5
    np.testing.assert_allclose(
        abragam_relaxation(TIMES, delta, 0.0), np.exp(-((delta * TIMES) ** 2)), rtol=1e-12
    )
    late = np.linspace(10.0, 20.0, 11)
    nu = 10.0
    np.testing.assert_allclose(
        abragam_relaxation(late, delta, nu),
        np.exp(-2.0 * delta**2 * late / nu + 2.0 * delta**2 / nu**2),
        rtol=1e-12,
    )


def test_abragam_is_continuous_across_its_short_time_series() -> None:
    # The motional factor switches to a power series below νt = 1e-3.
    nu = 1.0
    times = np.array([1e-3 * (1.0 - 1e-9), 1e-3 * (1.0 + 1e-9)])

    values = abragam_relaxation(times, 50.0, nu)

    assert values[0] == pytest.approx(values[1], rel=1e-9)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: static_gaussian_kubo_toyabe([-1.0, 0.0], 1.0), "times_us"),
        (lambda: static_gaussian_kubo_toyabe([np.nan], 1.0), "times_us"),
        (lambda: static_gaussian_kubo_toyabe(TIMES, -1.0), "delta_per_us"),
        (lambda: static_gaussian_kubo_toyabe_lf(TIMES, 1.0, np.inf), "field_T"),
        (lambda: static_lorentzian_kubo_toyabe(TIMES, -0.1), "a_per_us"),
        (lambda: dynamic_lorentzian_kubo_toyabe(TIMES, 0.5, -1.0), "collision_rate_per_us"),
        (lambda: keren_lf_relaxation(TIMES, 0.5, np.nan, 0.0), "collision_rate_per_us"),
        (lambda: abragam_relaxation(TIMES, 0.5, -2.0), "collision_rate_per_us"),
    ],
)
def test_invalid_arguments_raise(call, message: str) -> None:
    with pytest.raises(PolarisationError, match=message):
        call()


def test_zero_widths_give_no_relaxation() -> None:
    for values in (
        static_gaussian_kubo_toyabe_lf(TIMES, 0.0, 0.01),
        dynamic_lorentzian_kubo_toyabe(TIMES, 0.0, 1.0),
        keren_lf_relaxation(TIMES, 0.0, 1.0, 0.01),
    ):
        np.testing.assert_array_equal(values, 1.0)
