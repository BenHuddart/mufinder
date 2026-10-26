"""Tests for strong-collision dynamic polarisation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mufinder.science.dynamics import (
    DynamicPolarisationResult,
    calculate_dynamic_polarisation,
)
from mufinder.science.polarisation import PolarisationError
from mufinder.science.relaxation_functions import (
    abragam_relaxation,
    dynamic_lorentzian_kubo_toyabe,
    keren_lf_relaxation,
    static_gaussian_kubo_toyabe,
    static_gaussian_kubo_toyabe_lf,
    static_lorentzian_kubo_toyabe,
)

DELTA = 1.0  # μs⁻¹


def _gaussian_kt(times: np.ndarray) -> np.ndarray:
    return static_gaussian_kubo_toyabe(times, DELTA)


def test_zero_rate_returns_the_static_trace() -> None:
    times = np.linspace(0.0, 10.0, 501)
    static = _gaussian_kt(times)

    for method in ("exact", "fft"):
        result = calculate_dynamic_polarisation(static, times, 0.0, method=method)
        np.testing.assert_array_equal(result.polarisation, static)
        np.testing.assert_array_equal(result.static_polarisation, static)


def test_result_records_the_rate_step_and_method() -> None:
    times = np.linspace(0.0, 8.0, 801)

    exact = calculate_dynamic_polarisation(_gaussian_kt, times, 0.5)
    fft = calculate_dynamic_polarisation(_gaussian_kt, times, 0.5, method="fft")

    assert exact.method == "exact"
    assert exact.alpha is None
    assert exact.collision_rate_per_us == 0.5
    assert exact.time_step_us == pytest.approx(0.01)
    assert fft.method == "fft"
    assert fft.alpha == pytest.approx(10.0 / 801)


def test_array_and_callable_static_inputs_agree() -> None:
    times = np.linspace(0.0, 10.0, 1001)

    from_array = calculate_dynamic_polarisation(_gaussian_kt(times), times, 1.0)
    from_callable = calculate_dynamic_polarisation(_gaussian_kt, times, 1.0)

    np.testing.assert_array_equal(from_array.polarisation, from_callable.polarisation)


def test_exact_and_fft_agree_on_4096_points() -> None:
    times = np.linspace(0.0, 20.0, 4096)

    exact = calculate_dynamic_polarisation(_gaussian_kt, times, 0.5)
    fft = calculate_dynamic_polarisation(_gaussian_kt, times, 0.5, method="fft")

    np.testing.assert_allclose(fft.polarisation, exact.polarisation, rtol=0.0, atol=1e-6)


def test_fft_error_falls_as_the_damping_rises_to_ten_over_n() -> None:
    # Allodi and De Renzi, Table 1: the wrap-around error falls from αN = 0 to 10.
    times = np.linspace(0.0, 20.0, 4096)
    exact = calculate_dynamic_polarisation(_gaussian_kt, times, 0.5).polarisation

    errors = [
        np.max(
            np.abs(
                calculate_dynamic_polarisation(
                    _gaussian_kt, times, 0.5, method="fft", alpha=alpha_n / times.size
                ).polarisation
                - exact
            )
        )
        for alpha_n in (0.0, 1.0, 2.0, 5.0, 10.0)
    ]

    assert all(later < earlier for earlier, later in zip(errors, errors[1:], strict=False))
    assert errors[-1] < 1e-9


def test_fast_collisions_on_a_gaussian_distribution_tend_to_exponential_decay() -> None:
    # ν → ∞: G → exp(−2Δ²t/ν), with an O(Δ²/ν²) difference that falls as ν rises.
    times = np.linspace(0.0, 20.0, 20001)

    def deviation(rate: float) -> float:
        dynamic = calculate_dynamic_polarisation(_gaussian_kt, times, rate).polarisation
        return float(np.max(np.abs(dynamic - np.exp(-2.0 * DELTA**2 * times / rate))))

    slow, fast = deviation(10.0), deviation(20.0)

    assert fast < 5e-3
    assert fast < slow / 3.0


def test_dynamic_gaussian_kubo_toyabe_matches_abragam_at_large_rates() -> None:
    times = np.linspace(0.0, 10.0, 10001)

    dynamic = calculate_dynamic_polarisation(_gaussian_kt, times, 20.0).polarisation

    np.testing.assert_allclose(dynamic, abragam_relaxation(times, DELTA, 20.0), atol=1e-3)


@pytest.mark.parametrize("field_T", [0.0, 0.005, 0.02])
def test_dynamic_lf_kubo_toyabe_matches_keren_in_its_regime(field_T: float) -> None:
    times = np.linspace(0.0, 10.0, 10001)

    dynamic = calculate_dynamic_polarisation(
        lambda t: static_gaussian_kubo_toyabe_lf(t, DELTA, field_T), times, 10.0
    ).polarisation

    np.testing.assert_allclose(dynamic, keren_lf_relaxation(times, DELTA, 10.0, field_T), atol=5e-3)


@pytest.mark.parametrize("rate", [0.1, 1.0, 5.0])
def test_dynamic_lorentzian_matches_its_closed_form(rate: float) -> None:
    times = np.linspace(0.0, 10.0, 10001)

    dynamic = calculate_dynamic_polarisation(
        lambda t: static_lorentzian_kubo_toyabe(t, 1.0), times, rate
    ).polarisation

    np.testing.assert_allclose(dynamic, dynamic_lorentzian_kubo_toyabe(times, 1.0, rate), atol=1e-6)


def test_fft_keeps_a_static_trace_that_does_not_start_at_one() -> None:
    times = np.linspace(0.0, 10.0, 2048)
    static = 0.8 * _gaussian_kt(times)

    exact = calculate_dynamic_polarisation(static, times, 1.0)
    fft = calculate_dynamic_polarisation(static, times, 1.0, method="fft")

    assert exact.polarisation[0] == pytest.approx(0.8)
    np.testing.assert_allclose(fft.polarisation, exact.polarisation, atol=1e-6)


@pytest.mark.parametrize("method", ["exact", "fft"])
def test_a_collision_every_step_does_not_overflow(method: str) -> None:
    # ν τ = 1000: e^{ντ} overflows, but the model is just a collision in every step,
    # G_n = G⁽⁰⁾(τ) G_{n−1}.
    times = np.linspace(0.0, 10.0, 11)
    static = _gaussian_kt(times)

    dynamic = calculate_dynamic_polarisation(static, times, 1000.0, method=method).polarisation

    np.testing.assert_allclose(dynamic, static[1] ** np.arange(11), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    ("times", "message"),
    [
        (np.array([0.0, 0.1, 0.3, 0.4]), "uniformly spaced"),
        (np.linspace(0.1, 1.0, 10), "start at zero"),
        (np.linspace(1.0, 0.0, 10), "increase"),
        (np.array([0.0]), "at least two"),
        (np.array([0.0, np.nan]), "finite"),
    ],
)
def test_invalid_grids_raise(times: np.ndarray, message: str) -> None:
    with pytest.raises(PolarisationError, match=message):
        calculate_dynamic_polarisation(np.ones_like(times), times, 1.0)


@pytest.mark.parametrize("rate", [-1.0, np.inf, np.nan])
def test_invalid_rates_raise(rate: float) -> None:
    times = np.linspace(0.0, 1.0, 11)

    with pytest.raises(PolarisationError, match="collision_rate_per_us"):
        calculate_dynamic_polarisation(np.ones(11), times, rate)


def test_invalid_static_traces_and_options_raise() -> None:
    times = np.linspace(0.0, 1.0, 11)

    with pytest.raises(PolarisationError, match="shape"):
        calculate_dynamic_polarisation(np.ones(10), times, 1.0)
    with pytest.raises(PolarisationError, match="shape"):
        calculate_dynamic_polarisation(lambda t: np.ones(3), times, 1.0)
    with pytest.raises(PolarisationError, match="finite"):
        calculate_dynamic_polarisation(np.full(11, np.nan), times, 1.0)
    with pytest.raises(PolarisationError, match="unknown method"):
        calculate_dynamic_polarisation(np.ones(11), times, 1.0, method="laplace")  # type: ignore[arg-type]
    with pytest.raises(PolarisationError, match="alpha applies only"):
        calculate_dynamic_polarisation(np.ones(11), times, 1.0, alpha=0.1)
    with pytest.raises(PolarisationError, match="alpha must be"):
        calculate_dynamic_polarisation(np.ones(11), times, 1.0, method="fft", alpha=-0.1)
    with pytest.raises(PolarisationError, match="overflow"):
        calculate_dynamic_polarisation(np.ones(11), times, 1.0, method="fft", alpha=100.0)


def test_result_round_trips_through_json() -> None:
    times = np.linspace(0.0, 5.0, 101)
    result = calculate_dynamic_polarisation(_gaussian_kt, times, 2.0, method="fft")

    restored = DynamicPolarisationResult.from_json(json.loads(json.dumps(result.to_json())))

    np.testing.assert_array_equal(restored.times_us, result.times_us)
    np.testing.assert_array_equal(restored.polarisation, result.polarisation)
    np.testing.assert_array_equal(restored.static_polarisation, result.static_polarisation)
    assert restored.collision_rate_per_us == result.collision_rate_per_us
    assert restored.time_step_us == result.time_step_us
    assert restored.method == "fft"
    assert restored.alpha == result.alpha


@pytest.mark.parametrize(
    "change",
    [
        {"method": "laplace"},
        {"polarisation": [1.0, 0.5]},
        {"collision_rate_per_us": "fast"},
    ],
)
def test_malformed_json_raises(change: dict[str, object]) -> None:
    times = np.linspace(0.0, 1.0, 11)
    payload = calculate_dynamic_polarisation(np.ones(11), times, 1.0).to_json()
    payload.update(change)

    with pytest.raises(PolarisationError, match="malformed dynamic polarisation result"):
        DynamicPolarisationResult.from_json(payload)


def test_missing_json_field_raises() -> None:
    times = np.linspace(0.0, 1.0, 11)
    payload = calculate_dynamic_polarisation(np.ones(11), times, 1.0).to_json()
    del payload["times_us"]

    with pytest.raises(PolarisationError, match="malformed dynamic polarisation result"):
        DynamicPolarisationResult.from_json(payload)
