"""Tests for cutoff and k-point convergence analysis."""

from __future__ import annotations

import math

import pytest

from mufinder.science.convergence_analysis import (
    SWEEP_LONG_ENOUGH,
    SWEEP_NOT_ESTIMATED,
    SWEEP_TOO_SHORT,
    ConvergenceResult,
    analyze_convergence,
    analyze_kpoint_convergence,
    check_sweep_length,
    recommended_value,
    unconverged_criteria,
)

ATOMS = 6
TOL = 0.001  # eV/atom
GRID_LABELS = {
    1.0: "1 x 1 x 1",
    8.0: "2 x 2 x 2",
    27.0: "3 x 3 x 3",
    64.0: "4 x 4 x 4",
    125.0: "5 x 5 x 5",
}


def _power_law(cutoffs, *, exponent: float, gap_at_600: float) -> list[float]:
    """Total energies whose per-atom distance from the limit decays as ``Ecut**exponent``."""

    scale = gap_at_600 * ATOMS / 600.0**exponent
    return [-1000.0 + scale * float(cutoff) ** exponent for cutoff in cutoffs]


def _per_atom(values) -> list[float]:
    return [-1000.0 + value * ATOMS for value in values]


def _forces(cutoffs) -> list[float]:
    return [0.01 + 0.5 * (float(cutoff) / 300.0) ** -4 for cutoff in cutoffs]


def _cutoff(cutoffs, energies, forces=None, **kwargs):
    return analyze_convergence(
        cutoffs=cutoffs,
        energies=energies,
        max_forces=forces if forces is not None else [0.0] * len(cutoffs),
        num_atoms=kwargs.pop("num_atoms", ATOMS),
        energy_threshold=kwargs.pop("energy_threshold", TOL),
        force_threshold=kwargs.pop("force_threshold", 0.05),
        **kwargs,
    )


def _converged_at(value: float) -> ConvergenceResult:
    return ConvergenceResult(
        converged=True,
        quantity="Stress",
        summary=f"converged at {value:g}",
        explanation="",
        converged_value=value,
    )


# The reference criterion


def test_energy_is_judged_against_the_finest_point_not_the_previous_one() -> None:
    # Every successive difference is at most 1 meV/atom, but 300 and 400 eV sit
    # 2.5 and 1.5 meV/atom from the 800 eV value.
    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    analysis = _cutoff(cutoffs, _per_atom([0.0025, 0.0015, 0.0009, 0.0004, 0.0001, 0.0]))

    assert analysis.energy.converged
    assert analysis.energy.converged_value == pytest.approx(500.0)
    assert analysis.energy.reference_value == pytest.approx(800.0)
    assert analysis.energy.deviations[0] == (300.0, pytest.approx(0.0025))


def test_recommendation_does_not_depend_on_the_step() -> None:
    coarse = [300.0 + 100.0 * index for index in range(6)]
    fine = [300.0 + 50.0 * index for index in range(11)]

    def recommend(cutoffs):
        energies = _power_law(cutoffs, exponent=-6.0, gap_at_600=0.004)
        return _cutoff(cutoffs, energies, _forces(cutoffs)).recommended

    assert recommend(coarse) == pytest.approx(700.0)
    assert recommend(fine) == pytest.approx(700.0)


def test_a_point_back_outside_tolerance_restarts_the_search() -> None:
    # 300 eV agrees with 700 eV by accident; 500 eV is outside again.
    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0]
    analysis = _cutoff(cutoffs, _per_atom([0.0, 0.0005, 0.003, 0.0004, 0.0]))

    assert analysis.energy.converged_value == pytest.approx(600.0)


def test_the_finest_point_is_never_converged_on_its_own() -> None:
    analysis = _cutoff([300.0, 400.0, 500.0], [-10.0, -10.1, -10.2])

    assert not analysis.energy.converged
    assert analysis.energy.summary == "not converged within the sweep"
    assert analysis.energy.explanation == "400 eV is 0.017 eV/atom from 500 eV"
    assert analysis.recommended is None
    assert analysis.unconverged == ("energy",)


def test_two_points_are_needed() -> None:
    analysis = _cutoff([300.0, math.nan], [-10.0, -10.0])

    assert not analysis.energy.converged
    assert analysis.energy.explanation == "needs at least two completed points"


def test_series_are_cleaned_sorted_and_deduplicated() -> None:
    analysis = _cutoff(
        [500.0, 300.0, 400.0, 400.0, math.inf],
        [-10.0, -9.0, -10.5, -10.0, -10.0],
    )

    assert analysis.energy.converged_value == pytest.approx(400.0)
    assert [point for point, _ in analysis.energy.deviations] == [300.0, 400.0, 500.0]


def test_force_and_stress_are_judged_against_their_finest_values() -> None:
    analysis = _cutoff(
        [300.0, 400.0, 500.0, 600.0],
        [-10.0] * 4,
        [0.30, 0.12, 0.05, 0.02],
        max_stresses=[2.0, 0.4, 0.25, 0.2],
        stress_threshold=0.1,
    )

    assert analysis.force.converged_value == pytest.approx(500.0)
    assert analysis.force.explanation == "within 0.03 eV/Ang of 600 eV, as is every finer cutoff"
    assert analysis.stress.converged_value == pytest.approx(500.0)
    assert analysis.recommended == pytest.approx(500.0)


def test_converged_wording_names_the_deviation_and_the_reference() -> None:
    analysis = _cutoff([300.0, 400.0, 500.0, 600.0], _per_atom([0.5, 0.01, 0.0008, 0.0]))

    assert analysis.energy.summary == "converged at 500 eV"
    assert analysis.energy.explanation == (
        "within 0.0008 eV/atom of 600 eV, as is every finer cutoff"
    )
    assert analysis.energy.details == (
        "Energy converged at 500 eV: within 0.0008 eV/atom of 600 eV, as is every finer cutoff."
    )
    assert analysis.reference_text == "600 eV, the finest cutoff in the sweep"


def test_non_positive_atom_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="atom count"):
        _cutoff([300.0, 400.0], [-1.0, -1.0], num_atoms=0)


def test_stress_is_only_a_criterion_when_supplied() -> None:
    analysis = _cutoff([300.0, 400.0, 500.0], [-1.0, -1.0, -1.0], [0.1, 0.1, 0.1])

    assert analysis.stress is None
    assert analysis.recommended == pytest.approx(300.0)


# Sweep-length check


def test_a_short_smooth_sweep_warns_and_suggests_a_cutoff() -> None:
    cutoffs = [300.0, 400.0, 500.0, 600.0]
    energies = _power_law(cutoffs, exponent=-1.5, gap_at_600=0.003)

    check = check_sweep_length(cutoffs, energies, num_atoms=ATOMS, threshold=TOL)

    # The tolerance is reached at 600 * 3**(1/1.5) = 1248 eV.
    assert check.status == SWEEP_TOO_SHORT
    assert check.exponent == pytest.approx(-1.5, abs=0.05)
    assert check.finest_gap == pytest.approx(0.003, rel=0.05)
    assert check.required_cutoff == pytest.approx(1248.0, rel=0.05)
    assert check.suggested_cutoff == pytest.approx(1300.0)
    assert check.summary == "too short"
    assert check.explanation == (
        "600 eV is 0.003 eV/atom from the fitted limit; "
        "extend the sweep to 1300 eV and analyse again"
    )


def test_the_suggestion_rounds_up_to_the_last_step() -> None:
    cutoffs = [300.0 + 50.0 * index for index in range(11)]
    energies = _power_law(cutoffs, exponent=-1.5, gap_at_600=0.003)

    check = check_sweep_length(cutoffs, energies, num_atoms=ATOMS, threshold=TOL)

    assert check.status == SWEEP_TOO_SHORT
    assert check.suggested_cutoff == pytest.approx(1250.0)


def test_a_long_enough_sweep_does_not_warn() -> None:
    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    energies = _power_law(cutoffs, exponent=-6.0, gap_at_600=0.004)

    check = check_sweep_length(cutoffs, energies, num_atoms=ATOMS, threshold=TOL)

    assert check.status == SWEEP_LONG_ENOUGH
    assert check.summary == "long enough"
    assert check.explanation == "800 eV is within 0.00071 eV/atom of the fitted limit"
    assert check.suggested_cutoff is None
    assert check.extrapolated_energy < energies[-1]
    assert check.fitted_curve[0][0] == pytest.approx(cutoffs[0])
    assert check.fitted_curve[-1][0] == pytest.approx(cutoffs[-1])


def test_the_energy_result_carries_the_fit_for_the_plot() -> None:
    cutoffs = [300.0, 400.0, 500.0, 600.0]
    analysis = _cutoff(cutoffs, _power_law(cutoffs, exponent=-1.5, gap_at_600=0.003))

    assert analysis.sweep_length.status == SWEEP_TOO_SHORT
    assert analysis.energy.fitted_curve == analysis.sweep_length.fitted_curve
    assert analysis.energy.extrapolated_energy == analysis.sweep_length.extrapolated_energy


def test_fewer_than_four_points_cannot_be_fitted() -> None:
    check = check_sweep_length(
        [300.0, 400.0, 500.0], [-1.0, -1.1, -1.12], num_atoms=1, threshold=TOL
    )

    assert check.status == SWEEP_NOT_ESTIMATED
    assert check.summary == "limit not estimated"
    assert check.explanation == "the fit needs at least four cutoffs"
    assert check.fitted_curve is None


def test_an_energy_that_rises_in_every_tail_is_not_fitted() -> None:
    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0]

    check = check_sweep_length(
        cutoffs, _per_atom([0.5, 0.0, 0.003, 0.0, 0.002]), num_atoms=ATOMS, threshold=TOL
    )

    assert check.status == SWEEP_NOT_ESTIMATED
    assert check.explanation == "the energy does not fall steadily towards the finest cutoff"


def test_a_fit_that_misses_the_points_is_not_used() -> None:
    # A step, not a decay: no power law follows it to a quarter of the tolerance.
    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0]

    check = check_sweep_length(
        cutoffs, _per_atom([0.02, 0.02, 0.02, 0.0, 0.0]), num_atoms=ATOMS, threshold=TOL
    )

    assert check.status == SWEEP_NOT_ESTIMATED
    assert check.explanation in {
        "the power law does not follow the points closely enough",
        "the decay is not a power law",
    }


def test_a_decay_slower_than_the_exponent_bound_is_not_used() -> None:
    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    energies = [-1.0 + 0.02 * ATOMS * (cutoff / 300.0) ** -0.1 for cutoff in cutoffs]

    check = check_sweep_length(cutoffs, energies, num_atoms=ATOMS, threshold=TOL)

    assert check.status == SWEEP_NOT_ESTIMATED
    assert check.explanation == "the decay is not a power law"


def test_the_longest_good_tail_is_fitted() -> None:
    # The 300 eV point is far from the asymptotic regime; the fit drops it.
    cutoffs = [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
    energies = _power_law(cutoffs, exponent=-3.0, gap_at_600=0.003)
    energies[0] += 5.0

    check = check_sweep_length(cutoffs, energies, num_atoms=ATOMS, threshold=TOL)

    assert check.status == SWEEP_TOO_SHORT
    assert check.fitted_curve[0][0] == pytest.approx(400.0)
    assert check.exponent == pytest.approx(-3.0, abs=0.1)


def test_kpoint_sweeps_have_no_sweep_length_check() -> None:
    analysis = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0, 64.0],
        energies=[-100.0, -100.02, -100.0205, -100.021],
        max_forces=[0.20, 0.16, 0.159, 0.158],
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.01,
    )

    assert analysis.sweep_length is None
    assert analysis.energy.fitted_curve is None


# K-point sweeps


def test_kpoint_grids_are_judged_against_the_densest_grid() -> None:
    # 8 and 27 k-points agree with each other, but both sit 2.5 meV/atom from
    # the densest grid.
    analysis = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0, 64.0, 125.0],
        energies=[-100.0, -100.0005, -100.0, -100.02, -100.0205],
        max_forces=[0.2] * 5,
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.01,
        point_labels=GRID_LABELS,
        point_spacings={64.0: 0.08, 125.0: 0.06},
    )

    assert analysis.energy.converged_value == pytest.approx(64.0)
    assert analysis.energy.summary == "converged at 4 x 4 x 4 (spacing 0.08 1/Ang)"
    assert analysis.energy.explanation == (
        "within 0.000063 eV/atom of 5 x 5 x 5, as is every denser grid"
    )
    assert analysis.reference_text == "5 x 5 x 5, the densest grid in the sweep"
    assert analysis.recommended == pytest.approx(64.0)


def test_kpoint_wording_has_no_method_name_or_grid_prefix() -> None:
    analysis = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0],
        energies=[-100.0, -100.02, -100.0205],
        max_forces=[0.20, 0.16, 0.159],
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.05,
        point_labels=GRID_LABELS,
        point_spacings={1.0: 0.5, 8.0: 0.25, 27.0: 0.124},
    )

    for result in (analysis.energy, analysis.force):
        assert "Delta" not in result.details
        assert "Grid:" not in result.details
    assert analysis.energy.summary == "converged at 2 x 2 x 2 (spacing 0.25 1/Ang)"
    # 1 x 1 x 1 has no spacing that reproduces it.
    assert analysis.force.summary == "converged at 1 x 1 x 1"


def test_kpoint_wording_without_labels_names_the_count() -> None:
    analysis = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0],
        energies=[-100.0, -100.02, -100.0205],
        max_forces=[0.20, 0.16, 0.159],
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.05,
    )

    assert analysis.energy.summary == "converged at 8 k-points"
    assert analysis.reference_text == "27 k-points, the densest grid in the sweep"


def test_kpoint_stress_uses_grid_labels() -> None:
    analysis = analyze_kpoint_convergence(
        kpoint_counts=[1.0, 8.0, 27.0],
        energies=[-100.0, -100.0, -100.0],
        max_forces=[0.1, 0.1, 0.1],
        num_atoms=8,
        energy_threshold=0.001,
        force_threshold=0.05,
        max_stresses=[1.0, 0.3, 0.25],
        stress_threshold=0.1,
        point_labels=GRID_LABELS,
    )

    assert analysis.stress.summary == "converged at 2 x 2 x 2"
    assert analysis.stress.explanation == "within 0.05 GPa of 3 x 3 x 3, as is every denser grid"
    assert analysis.recommended == pytest.approx(8.0)


# Combined recommendation


def test_recommended_value_is_the_largest_converged_value() -> None:
    assert recommended_value(
        _converged_at(400.0), _converged_at(500.0), _converged_at(600.0)
    ) == pytest.approx(600.0)
    assert recommended_value(_converged_at(400.0), _converged_at(500.0)) == pytest.approx(500.0)


def test_recommended_value_requires_every_supplied_criterion() -> None:
    unconverged = ConvergenceResult(
        converged=False, quantity="Force", summary="not converged", explanation=""
    )

    assert recommended_value(_converged_at(400.0), _converged_at(500.0), unconverged) is None
    assert unconverged_criteria(_converged_at(400.0), unconverged) == ("force",)
    assert unconverged_criteria(_converged_at(400.0), _converged_at(500.0)) == ()
