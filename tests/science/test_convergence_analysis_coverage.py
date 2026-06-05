"""Supplementary coverage for cutoff/k-point convergence analysis.

These tests exercise error paths, normalization guards, and method-dispatch
branches that the primary suite in ``test_convergence_analysis.py`` does not
reach. They are deterministic: no network, GUI, files, sleeps, or randomness.
"""

from __future__ import annotations

import pytest

from mufinder.science.convergence_analysis import (
    ConvergenceResult,
    _apply_kpoint_label,
    _clean_series,
    _spacing_annotation,
    analyze_convergence,
    asymptotic_extrapolation_energy,
    delta_threshold_energy,
    delta_threshold_force,
    derivative_energy,
    recommended_value,
)


# --------------------------------------------------------------------------
# ConvergenceResult.converged_cutoff alias (line 30)
# --------------------------------------------------------------------------
def test_converged_cutoff_aliases_converged_value() -> None:
    result = ConvergenceResult(
        converged=True,
        method="delta-threshold",
        details="",
        converged_value=512.0,
    )
    assert result.converged_cutoff == pytest.approx(512.0)

    empty = ConvergenceResult(converged=False, method="delta-threshold", details="")
    assert empty.converged_cutoff is None


# --------------------------------------------------------------------------
# _clean_series drops non-coercible entries (lines 39-40)
# --------------------------------------------------------------------------
def test_clean_series_skips_non_numeric_and_sorts_dedupes() -> None:
    cutoffs = [400.0, "not-a-number", 300.0, 300.0, None]
    values = [-10.5, -9.0, -10.0, -10.2, -1.0]

    x, y = _clean_series(cutoffs, values)

    # "not-a-number" and None rows are dropped; duplicate 300.0 keeps the last
    # observed value after a stable sort; output is ascending by cutoff.
    assert x == (300.0, 400.0)
    assert y == (-10.2, -10.5)


# --------------------------------------------------------------------------
# delta_threshold_energy: non-positive atom count (line 73)
# --------------------------------------------------------------------------
def test_delta_threshold_energy_rejects_non_positive_atom_count() -> None:
    result = delta_threshold_energy(
        [300.0, 400.0],
        [-10.0, -10.0001],
        num_atoms=0,
        threshold=0.001,
    )
    assert result.converged is False
    assert "positive" in result.details.lower()


# --------------------------------------------------------------------------
# derivative_energy error paths (lines 122, 130, 138, 154, 162)
# --------------------------------------------------------------------------
def test_derivative_energy_requires_two_points() -> None:
    result = derivative_energy([300.0], [-10.0], num_atoms=4, threshold=0.001)
    assert result.converged is False
    assert "at least two" in result.details.lower()
    assert result.fallback_suggestion is not None


def test_derivative_energy_rejects_non_positive_atom_count() -> None:
    result = derivative_energy(
        [300.0, 400.0],
        [-10.0, -10.05],
        num_atoms=-3,
        threshold=0.001,
    )
    assert result.converged is False
    assert "positive" in result.details.lower()


def test_derivative_energy_requires_increasing_cutoffs() -> None:
    # After dedup, two identical cutoffs collapse to one point (caught by the
    # <2 guard). Use distinct-but-equal-step zero by passing values that, once
    # cleaned, have no positive step: feed cutoffs whose only gap is non-positive.
    # _clean_series dedupes equal cutoffs, so build a case with >=2 distinct
    # points but rely on the step filter by giving a descending-then-handled
    # series is impossible post-sort; instead force the "no positive steps"
    # branch via a single repeated value handled above. To hit line 138 we need
    # >=2 distinct cleaned points yet no positive step — unreachable after sort —
    # so this asserts the dedup path collapses to the <2 branch instead.
    result = derivative_energy(
        [400.0, 400.0],
        [-10.0, -10.5],
        num_atoms=4,
        threshold=0.001,
    )
    assert result.converged is False
    assert "at least two" in result.details.lower()


def test_derivative_energy_reports_when_rate_never_flattens() -> None:
    # Steep, sustained slope so the derivative never falls below threshold.
    result = derivative_energy(
        [300.0, 400.0, 500.0],
        [-10.0, -20.0, -30.0],
        num_atoms=1,
        threshold=0.0001,
    )
    assert result.converged is False
    assert result.converged_value is None
    assert "did not reach" in result.details.lower()
    assert result.deltas is not None and len(result.deltas) == 2
    assert result.fallback_suggestion is not None


def test_derivative_energy_converges_and_reports_rate() -> None:
    result = derivative_energy(
        [300.0, 400.0, 500.0],
        [-10.0, -10.5, -10.5005],
        num_atoms=1,
        threshold=0.01,
    )
    assert result.converged is True
    assert result.converged_value == pytest.approx(500.0)
    assert "|dE/dEcut|" in result.details


# --------------------------------------------------------------------------
# asymptotic_extrapolation_energy guards (lines 208, 245-246, 285)
# --------------------------------------------------------------------------
def test_asymptotic_rejects_non_positive_atom_count() -> None:
    pytest.importorskip("scipy")
    result = asymptotic_extrapolation_energy(
        [300.0, 400.0, 500.0, 600.0],
        [-10.0, -10.5, -10.7, -10.75],
        num_atoms=0,
        threshold=0.001,
    )
    assert result.converged is False
    assert "positive" in result.details.lower()


def test_asymptotic_fit_quality_or_failure_falls_back() -> None:
    pytest.importorskip("scipy")
    # Pathological alternating data: either the fit raises (245-246) or yields a
    # poor-quality result; both paths produce a non-converged fallback result.
    result = asymptotic_extrapolation_energy(
        [300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        [0.0, 100.0, -100.0, 100.0, -100.0, 100.0],
        num_atoms=2,
        threshold=0.001,
    )
    assert result.converged is False
    assert result.fallback_suggestion is not None


def test_asymptotic_good_fit_but_tolerance_unreached(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("scipy")
    import mufinder.science.convergence_analysis as mod

    # A clean exponential fits well (high R^2, low RMSE) so we pass the quality
    # gate, but an absurdly tiny threshold means no sampled cutoff's gap to
    # E_inf clears it -> converged_value stays None (line 285 branch).
    import math

    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0]
    energies = [-10.0 + 2.0 * math.exp(-0.01 * c) for c in cutoffs]

    result = mod.asymptotic_extrapolation_energy(
        cutoffs,
        energies,
        num_atoms=2,
        threshold=1e-15,
    )
    assert result.converged is False
    assert result.converged_value is None
    assert result.extrapolated_energy is not None
    assert result.fitted_curve is not None
    assert "did not reach the requested energy tolerance" in result.details


# --------------------------------------------------------------------------
# delta_threshold_force error paths (lines 321, 339)
# --------------------------------------------------------------------------
def test_delta_threshold_force_requires_two_points() -> None:
    result = delta_threshold_force([300.0], [0.2], threshold=0.01)
    assert result.converged is False
    assert "at least two" in result.details.lower()
    assert result.fallback_suggestion is not None


def test_delta_threshold_force_reports_not_converged() -> None:
    result = delta_threshold_force(
        [300.0, 400.0, 500.0],
        [0.5, 0.1, 0.5],
        threshold=0.001,
    )
    assert result.converged is False
    assert result.converged_value is None
    assert "did not reach the threshold" in result.details
    assert result.deltas is not None and len(result.deltas) == 2


# --------------------------------------------------------------------------
# _spacing_annotation branches (lines 430, 438-439)
# --------------------------------------------------------------------------
def test_spacing_annotation_returns_empty_without_spacings() -> None:
    assert _spacing_annotation("Grid: 3 x 3 x 3", 27.0, None) == ""


def test_spacing_annotation_returns_empty_when_value_absent() -> None:
    assert _spacing_annotation("Grid: 3 x 3 x 3", 27.0, {8.0: 0.25}) == ""


def test_spacing_annotation_handles_unparseable_grid_label() -> None:
    # Three "x"-separated parts that are not integers trigger the ValueError
    # path; since ints is None it is not (1,1,1), so the spacing is emitted.
    text = _spacing_annotation("Grid: a x b x c", 27.0, {27.0: 0.124})
    assert text == "spacing 0.124 1/Ang"


def test_spacing_annotation_emits_for_non_gamma_grid() -> None:
    text = _spacing_annotation("Grid: 3 x 3 x 3", 27.0, {27.0: 0.124})
    assert text == "spacing 0.124 1/Ang"


# --------------------------------------------------------------------------
# _apply_kpoint_label early-return branches (lines 461, 464)
# --------------------------------------------------------------------------
def test_apply_kpoint_label_returns_input_when_not_converged() -> None:
    unconverged = ConvergenceResult(
        converged=False,
        method="delta-threshold",
        details="original",
        converged_value=None,
    )
    out = _apply_kpoint_label(
        unconverged,
        {27.0: "3 x 3 x 3"},
        quantity="Energy",
        delta_unit="eV/atom",
    )
    assert out is unconverged
    assert out.details == "original"


def test_apply_kpoint_label_returns_input_when_label_missing() -> None:
    converged = ConvergenceResult(
        converged=True,
        method="delta-threshold",
        details="original",
        converged_value=27.0,
        deltas=((27.0, 1e-6),),
    )
    # No label for 27.0 -> grid_label is None -> result returned unchanged.
    out = _apply_kpoint_label(
        converged,
        {8.0: "2 x 2 x 2"},
        quantity="Energy",
        delta_unit="eV/atom",
    )
    assert out is converged
    assert out.details == "original"


# --------------------------------------------------------------------------
# analyze_convergence method dispatch (lines 562-567)
# --------------------------------------------------------------------------
def test_analyze_convergence_dispatches_derivative() -> None:
    energy, force = analyze_convergence(
        method="Derivative",
        cutoffs=[300.0, 400.0, 500.0],
        energies=[-10.0, -10.5, -10.5005],
        max_forces=[0.2, 0.17, 0.169],
        num_atoms=1,
        energy_threshold=0.01,
        force_threshold=0.01,
    )
    assert energy.method == "derivative"
    assert force.method == "force-delta-threshold"


def test_analyze_convergence_dispatches_asymptotic() -> None:
    pytest.importorskip("scipy")
    import math

    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0]
    energies = [-10.0 + 2.0 * math.exp(-0.01 * c) for c in cutoffs]
    energy, force = analyze_convergence(
        method="asymptotic",
        cutoffs=cutoffs,
        energies=energies,
        max_forces=[0.2, 0.18, 0.17, 0.169, 0.1689],
        num_atoms=2,
        energy_threshold=0.003,
        force_threshold=0.01,
    )
    assert energy.method == "asymptotic"


def test_analyze_convergence_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unsupported energy convergence method"):
        analyze_convergence(
            method="nonsense-method",
            cutoffs=[300.0, 400.0],
            energies=[-10.0, -10.1],
            max_forces=[0.2, 0.19],
            num_atoms=4,
            energy_threshold=0.001,
            force_threshold=0.01,
        )


# --------------------------------------------------------------------------
# recommended_value with no converged candidates (line 590)
# --------------------------------------------------------------------------
def test_recommended_value_returns_none_when_nothing_converged() -> None:
    energy = ConvergenceResult(converged=False, method="delta-threshold", details="")
    force = ConvergenceResult(converged=False, method="force-delta-threshold", details="")
    assert recommended_value(energy, force) is None
