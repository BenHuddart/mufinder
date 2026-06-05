"""Tests for cutoff convergence analysis algorithms."""

from __future__ import annotations

import math

import pytest

from mufinder.science.convergence_analysis import (
    ConvergenceResult,
    _apply_kpoint_label,
    analyze_convergence,
    analyze_kpoint_convergence,
    apply_kpoint_label_to_stress_result,
    asymptotic_extrapolation_energy,
    delta_threshold_energy,
    delta_threshold_force,
    delta_threshold_stress,
    derivative_energy,
    recommended_cutoff,
    recommended_value,
)


def test_delta_threshold_energy_converges_on_successive_delta() -> None:
    result = delta_threshold_energy(
        [300.0, 400.0, 500.0],
        [-100.0, -100.05, -100.051],
        num_atoms=10,
        threshold=0.001,
    )

    assert result.converged is True
    assert result.converged_value == pytest.approx(500.0)


def test_delta_threshold_energy_reports_not_converged() -> None:
    result = delta_threshold_energy(
        [300.0, 400.0, 500.0],
        [-100.0, -100.2, -100.35],
        num_atoms=2,
        threshold=0.001,
    )

    assert result.converged is False
    assert result.converged_value is None


def test_derivative_energy_converges_when_rate_flattens() -> None:
    result = derivative_energy(
        [300.0, 400.0, 500.0],
        [-10.0, -10.05, -10.06],
        num_atoms=10,
        threshold=0.001,
    )

    assert result.converged is True
    assert result.converged_value == pytest.approx(500.0)


def test_force_delta_threshold_handles_non_monotonic_series() -> None:
    result = delta_threshold_force(
        [300.0, 400.0, 500.0, 600.0],
        [0.25, 0.20, 0.23, 0.205],
        threshold=0.03,
    )

    assert result.converged is True
    assert result.converged_value == pytest.approx(500.0)


def test_asymptotic_extrapolation_needs_enough_points() -> None:
    result = asymptotic_extrapolation_energy(
        [300.0, 400.0, 500.0],
        [-10.0, -10.05, -10.07],
        num_atoms=4,
        threshold=0.001,
    )

    assert result.converged is False
    assert "at least four" in result.details.lower()
    assert result.fallback_suggestion is not None


def test_asymptotic_extrapolation_converges_on_clean_signal() -> None:
    scipy = pytest.importorskip("scipy")
    assert scipy is not None

    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0]
    energies = [-10.0 + 2.0 * math.exp(-0.01 * cutoff) for cutoff in cutoffs]

    result = asymptotic_extrapolation_energy(
        cutoffs,
        energies,
        num_atoms=2,
        threshold=0.003,
    )

    assert result.converged is True
    assert result.converged_value is not None
    assert result.fitted_curve is not None
    assert result.extrapolated_energy is not None


def test_asymptotic_extrapolation_falls_back_on_noisy_data() -> None:
    scipy = pytest.importorskip("scipy")
    assert scipy is not None

    result = asymptotic_extrapolation_energy(
        [300.0, 400.0, 500.0, 600.0, 700.0],
        [-10.0, -9.2, -10.8, -9.0, -10.7],
        num_atoms=2,
        threshold=0.001,
    )

    assert result.converged is False
    assert result.fallback_suggestion is not None


def test_analyze_convergence_runs_selected_method_and_force() -> None:
    energy_result, force_result = analyze_convergence(
        method="delta-threshold",
        cutoffs=[300.0, 400.0, 500.0],
        energies=[-100.0, -100.05, -100.051],
        max_forces=[0.20, 0.17, 0.165],
        num_atoms=10,
        energy_threshold=0.001,
        force_threshold=0.01,
    )

    assert energy_result.converged is True
    assert force_result.converged is True
    assert recommended_cutoff(energy_result, force_result) == pytest.approx(500.0)


def test_delta_threshold_ignores_nan_and_requires_two_points() -> None:
    result = delta_threshold_energy(
        [300.0, 400.0, 500.0],
        [-10.0, float("nan"), float("nan")],
        num_atoms=2,
        threshold=0.001,
    )

    assert result.converged is False
    assert "at least two" in result.details.lower()


def test_analyze_kpoint_convergence_uses_delta_threshold_only() -> None:
    energy_result, force_result = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0],
        energies=[-100.0, -100.02, -100.0205],
        max_forces=[0.20, 0.16, 0.159],
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.01,
    )

    assert energy_result.method == "delta-threshold"
    assert force_result.method == "force-delta-threshold"
    assert energy_result.converged is True
    assert force_result.converged is True
    assert recommended_cutoff(energy_result, force_result) == pytest.approx(27.0)


def test_analyze_kpoint_convergence_details_use_grid_label_when_labels_provided() -> None:
    energy_result, force_result = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0],
        energies=[-100.0, -100.02, -100.0205],
        max_forces=[0.20, 0.16, 0.159],
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.01,
        point_labels={1.0: "1 x 1 x 1", 8.0: "2 x 2 x 2", 27.0: "3 x 3 x 3"},
    )

    assert "at 3 x 3 x 3" in energy_result.details
    assert "at 3 x 3 x 3" in force_result.details
    # Raw count must not appear as the convergence point
    assert "at 27" not in energy_result.details
    assert "at 27" not in force_result.details


def test_analyze_kpoint_convergence_details_keep_numeric_without_labels() -> None:
    energy_result, force_result = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0],
        energies=[-100.0, -100.02, -100.0205],
        max_forces=[0.20, 0.16, 0.159],
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.01,
    )

    # Without point_labels the original numeric details are preserved unchanged.
    assert "27" in energy_result.details
    assert "27" in force_result.details


def test_delta_threshold_stress_converges_when_delta_below_threshold() -> None:
    result = delta_threshold_stress(
        [300.0, 400.0, 500.0, 600.0],
        [1.5, 0.4, 0.35, 0.34],
        threshold=0.1,
    )

    assert result.converged is True
    assert result.method == "stress-delta-threshold"
    assert result.converged_value == pytest.approx(500.0)


def test_delta_threshold_stress_not_converged_keeps_deltas() -> None:
    result = delta_threshold_stress(
        [300.0, 400.0, 500.0],
        [5.0, 2.0, 0.5],
        threshold=0.1,
    )

    assert result.converged is False
    assert result.deltas is not None
    assert len(result.deltas) == 2


def test_delta_threshold_stress_requires_two_points() -> None:
    result = delta_threshold_stress(
        [300.0],
        [0.05],
        threshold=0.1,
    )

    assert result.converged is False
    assert "at least two" in result.details.lower()


def test_delta_threshold_stress_skips_nan_values() -> None:
    result = delta_threshold_stress(
        [300.0, 400.0, 500.0],
        [float("nan"), float("nan"), 0.4],
        threshold=0.1,
    )

    assert result.converged is False
    assert "at least two" in result.details.lower()


def test_recommended_value_includes_stress_when_converged() -> None:
    energy = ConvergenceResult(converged=True, method="delta-threshold", details="", converged_value=400.0)
    force = ConvergenceResult(converged=True, method="force-delta-threshold", details="", converged_value=500.0)
    stress = ConvergenceResult(converged=True, method="stress-delta-threshold", details="", converged_value=600.0)

    assert recommended_value(energy, force, stress) == pytest.approx(600.0)


def test_recommended_value_ignores_unconverged_stress() -> None:
    energy = ConvergenceResult(converged=True, method="delta-threshold", details="", converged_value=400.0)
    force = ConvergenceResult(converged=True, method="force-delta-threshold", details="", converged_value=500.0)
    stress = ConvergenceResult(converged=False, method="stress-delta-threshold", details="", converged_value=None)

    assert recommended_value(energy, force, stress) == pytest.approx(500.0)


def test_recommended_value_backward_compatible_without_stress() -> None:
    energy = ConvergenceResult(converged=True, method="delta-threshold", details="", converged_value=400.0)
    force = ConvergenceResult(converged=True, method="force-delta-threshold", details="", converged_value=500.0)

    assert recommended_value(energy, force) == pytest.approx(500.0)


def test_apply_kpoint_label_to_stress_result_rewrites_grid_label() -> None:
    stress = delta_threshold_stress(
        [1.0, 8.0, 27.0],
        [1.0, 0.3, 0.25],
        threshold=0.1,
    )
    assert stress.converged is True
    assert stress.converged_value == pytest.approx(27.0)

    labeled = apply_kpoint_label_to_stress_result(
        stress,
        {1.0: "1 x 1 x 1", 8.0: "2 x 2 x 2", 27.0: "3 x 3 x 3"},
    )

    assert "Stress converged by" in labeled.details
    assert "3 x 3 x 3" in labeled.details
    assert "at 27 " not in labeled.details


def test_analyze_kpoint_convergence_details_include_spacing_for_non_gamma_grid() -> None:
    energy_result, force_result = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0],
        energies=[-100.0, -100.02, -100.0205],
        max_forces=[0.20, 0.16, 0.159],
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.01,
        point_labels={1.0: "Grid: 1 x 1 x 1", 8.0: "Grid: 2 x 2 x 2", 27.0: "Grid: 3 x 3 x 3"},
        point_spacings={1.0: 0.25, 8.0: 0.25, 27.0: 0.124},
    )

    assert "Grid: 3 x 3 x 3" in energy_result.details
    assert "spacing 0.124 1/Ang" in energy_result.details
    assert "delta=" in energy_result.details
    assert "Grid: 3 x 3 x 3" in force_result.details
    assert "spacing 0.124 1/Ang" in force_result.details


def test_apply_kpoint_label_omits_spacing_for_gamma_only_grid() -> None:
    # When the converged point is (1,1,1) the spacing is ill-defined, so the
    # details string should not advertise the boundary value as if it were
    # the "right" spacing. The standard delta-threshold algorithm never
    # actually returns (1,1,1) as converged (convergence is always the
    # second-or-later point), but a synthetic result lets us exercise the
    # Gamma-only branch directly.
    result = ConvergenceResult(
        converged=True,
        method="delta-threshold",
        details="",
        converged_value=1.0,
        deltas=((1.0, 1.0e-6),),
    )
    labeled = _apply_kpoint_label(
        result,
        {1.0: "Grid: 1 x 1 x 1"},
        quantity="Energy",
        delta_unit="eV/atom",
        point_spacings={1.0: 0.25},
    )
    assert "Grid: 1 x 1 x 1" in labeled.details
    assert "spacing" not in labeled.details


def test_apply_kpoint_label_to_stress_result_includes_spacing_when_supplied() -> None:
    stress = delta_threshold_stress(
        [1.0, 8.0, 27.0],
        [1.0, 0.3, 0.25],
        threshold=0.1,
    )
    labeled = apply_kpoint_label_to_stress_result(
        stress,
        {1.0: "Grid: 1 x 1 x 1", 8.0: "Grid: 2 x 2 x 2", 27.0: "Grid: 3 x 3 x 3"},
        point_spacings={1.0: 0.25, 8.0: 0.25, 27.0: 0.124},
    )
    assert "Grid: 3 x 3 x 3" in labeled.details
    assert "spacing 0.124 1/Ang" in labeled.details
