"""Tests for grouping UEP minima into distinct sites by symmetry-orbit distance."""

from __future__ import annotations

from dataclasses import asdict
import json
import random

import numpy as np
import pytest

from mufinder.domain.models import Structure
from mufinder.science.uep import UepError, UepOptimizationResult
from mufinder.science.uep_clustering import (
    DEFAULT_GROUPING_TOLERANCE_ANG,
    GROUPING_LOG_PREFIX,
    UEP_GROUPING_METHOD,
    group_uep_minima,
    measure_uep_minima,
    relaxed_site_number_map,
    uep_batch_summary_from_dict,
    uep_grouping_report,
    uep_search_log,
)

from tests.support.uep_results import fake_optimization

_A = 4.695
_C = 3.180
_X = 0.306
_COF2 = Structure(
    label="CoF2",
    lattice_vectors=[[_A, 0.0, 0.0], [0.0, _A, 0.0], [0.0, 0.0, _C]],
    species=["Co", "Co", "F", "F", "F", "F"],
    fractional_positions=[
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [_X, _X, 0.0],
        [1.0 - _X, 1.0 - _X, 0.0],
        [0.5 + _X, 0.5 - _X, 0.5],
        [0.5 - _X, 0.5 + _X, 0.5],
    ],
)
_LATTICE = tuple(tuple(row) for row in _COF2.lattice_vectors)
#: The four images of the 4d site, the minimum of the CoF2 model density.
_IMAGES_4D = ((0.0, 0.5, 0.25), (0.0, 0.5, 0.75), (0.5, 0.0, 0.25), (0.5, 0.0, 0.75))
_ENERGY_EV = -15.976111


def _minimum(
    fractional: tuple[float, float, float],
    energy: float = _ENERGY_EV,
    *,
    label: str = "minimum",
    success: bool = True,
) -> UepOptimizationResult:
    return fake_optimization(
        tuple(float(value) % 1.0 for value in fractional),
        classical=energy,
        label=label,
        success=success,
        lattice_vectors=_LATTICE,
    )


def _noisy_images(
    noise: float, *, count: int = 12, seed: int = 0, energy: float = _ENERGY_EV
) -> list[UepOptimizationResult]:
    generator = np.random.default_rng(seed)
    return [
        _minimum(
            tuple(np.asarray(_IMAGES_4D[index % 4]) + generator.normal(0.0, noise, 3)),
            energy + generator.normal(0.0, 1e-11),
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("noise", [1e-9, 1e-7, 1e-6, 1e-4])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_twelve_noisy_images_of_one_site_are_one_group(noise: float, seed: int) -> None:
    summary = group_uep_minima(_noisy_images(noise, seed=seed), source_structure=_COF2)

    assert [cluster.size for cluster in summary.clusters] == [12]
    assert summary.clusters[0].member_indexes == tuple(range(12))


def test_the_representative_is_the_lowest_member_reported_in_the_asymmetric_unit() -> None:
    results = _noisy_images(1e-7)
    results[7] = _minimum((0.99999999, 0.5, 0.75 + 1e-7), _ENERGY_EV - 1e-6)

    (cluster,) = group_uep_minima(results, source_structure=_COF2).clusters

    assert cluster.representative_index == 7
    assert cluster.representative_fractional_position == pytest.approx((0.0, 0.5, 0.25))
    assert cluster.representative_position_ang == pytest.approx((0.0, _A / 2, _C / 4))
    assert cluster.minimum_energy_ev == pytest.approx(_ENERGY_EV - 1e-6)
    assert cluster.average_energy_ev == pytest.approx(_ENERGY_EV, abs=1e-6)


def test_two_distinct_sites_one_angstrom_apart_are_two_groups() -> None:
    # The second site shares no orbit with the 4d site, and sits about 1 Å along c from it.
    summary = group_uep_minima(
        [_minimum((0.0, 0.5, 0.25)), _minimum((0.2, 0.3, 0.1))],
        source_structure=_COF2,
    )

    assert [cluster.member_indexes for cluster in summary.clusters] == [(0,), (1,)]


def test_equal_positions_more_than_a_millielectronvolt_apart_stay_apart() -> None:
    summary = group_uep_minima(
        [_minimum((0.0, 0.5, 0.25), _ENERGY_EV + 2e-3), _minimum((0.0, 0.5, 0.25), _ENERGY_EV)],
        source_structure=_COF2,
    )

    assert [cluster.member_indexes for cluster in summary.clusters] == [(1,), (0,)]
    assert summary.clusters[0].cluster_index == 1


def test_an_energy_difference_under_the_tolerance_does_not_separate_a_site() -> None:
    summary = group_uep_minima(
        [_minimum((0.0, 0.5, 0.25), _ENERGY_EV + 5e-4), _minimum((0.5, 0.0, 0.75), _ENERGY_EV)],
        source_structure=_COF2,
    )

    assert [cluster.size for cluster in summary.clusters] == [2]


def test_groups_are_ordered_by_energy_and_numbered_from_one() -> None:
    summary = group_uep_minima(
        [
            _minimum((0.2, 0.3, 0.1), _ENERGY_EV + 0.5),
            _minimum((0.0, 0.5, 0.25), _ENERGY_EV),
            _minimum((0.1, 0.1, 0.4), _ENERGY_EV + 0.2),
        ],
        source_structure=_COF2,
    )

    assert [cluster.cluster_index for cluster in summary.clusters] == [1, 2, 3]
    assert [cluster.representative_index for cluster in summary.clusters] == [1, 2, 0]


@pytest.mark.parametrize(
    ("result", "failed", "non_minimum"),
    [
        (_minimum((0.0, 0.5, 0.0), _ENERGY_EV + 0.086, label="saddle"), 0, 1),
        (_minimum((0.5, 0.5, 0.0), _ENERGY_EV + 3.0, label="maximum"), 0, 1),
        (_minimum((0.0, 0.5, 0.25), _ENERGY_EV, label="flat"), 0, 1),
        (_minimum((0.0, 0.5, 0.25), _ENERGY_EV, success=False), 1, 0),
        (_minimum((0.0, 0.5, 0.25), float("nan")), 0, 1),
    ],
    ids=["saddle", "maximum", "flat", "failed", "no-energy"],
)
def test_only_converged_minima_are_grouped_and_the_others_are_counted(
    result: UepOptimizationResult, failed: int, non_minimum: int
) -> None:
    results = [*_noisy_images(1e-7, count=4), result]

    summary = group_uep_minima(results, source_structure=_COF2)

    assert [cluster.member_indexes for cluster in summary.clusters] == [(0, 1, 2, 3)]
    assert summary.failed_count == failed
    assert summary.non_minimum_count == non_minimum
    report = uep_grouping_report(summary)
    assert ("1 failed optimisation" in report) == bool(failed)
    assert ("1 result that is not a minimum" in report) == bool(non_minimum)


def test_a_batch_with_no_minimum_has_no_groups() -> None:
    summary = group_uep_minima(
        [_minimum((0.0, 0.5, 0.0), label="saddle"), _minimum((0.1, 0.1, 0.1), success=False)],
        source_structure=_COF2,
    )

    assert summary.clusters == ()
    assert summary.failed_count == 1
    assert summary.non_minimum_count == 1
    assert uep_grouping_report(summary) == (
        f"{GROUPING_LOG_PREFIX}no minima found. "
        "Left out 1 failed optimisation and 1 result that is not a minimum."
    )


def test_an_empty_batch_has_no_groups() -> None:
    assert group_uep_minima([], source_structure=_COF2).clusters == ()


def _groups_as_positions(results: list[UepOptimizationResult]) -> list[list[tuple[float, ...]]]:
    summary = group_uep_minima(results, source_structure=_COF2)
    return [
        sorted(results[member].optimized_fractional_position for member in cluster.member_indexes)
        for cluster in summary.clusters
    ]


def test_the_groups_do_not_depend_on_the_order_of_the_results() -> None:
    results = [
        *_noisy_images(1e-6, count=8),
        _minimum((0.2, 0.3, 0.1), _ENERGY_EV + 0.4),
        _minimum((0.8, 0.7, 0.1), _ENERGY_EV + 0.4 + 1e-5),
        _minimum((0.0, 0.5, 0.0), _ENERGY_EV + 0.086, label="saddle"),
    ]
    expected = _groups_as_positions(results)

    for seed in range(6):
        shuffled = list(results)
        random.Random(seed).shuffle(shuffled)
        assert _groups_as_positions(shuffled) == expected


def test_a_group_does_not_depend_on_which_other_sites_the_batch_found() -> None:
    alone = group_uep_minima(_noisy_images(1e-6), source_structure=_COF2)
    with_outlier = group_uep_minima(
        [*_noisy_images(1e-6), _minimum((0.1, 0.1, 0.4), _ENERGY_EV + 5.0)],
        source_structure=_COF2,
    )

    assert with_outlier.clusters[0] == alone.clusters[0]


def test_the_tolerance_is_a_length_and_chains_members() -> None:
    # Three minima on a line along c: 0.2 Å and 0.2 Å apart, so 0.4 Å end to end.
    step = 0.2 / _C
    results = [
        _minimum((0.2, 0.3, 0.1)),
        _minimum((0.2, 0.3, 0.1 + step)),
        _minimum((0.2, 0.3, 0.1 + 2 * step)),
    ]

    assert len(group_uep_minima(results, source_structure=_COF2).clusters) == 1
    assert len(group_uep_minima(results, source_structure=_COF2, tolerance_ang=0.1).clusters) == 3


def test_regrouping_with_measured_distances_gives_the_same_groups() -> None:
    results = [*_noisy_images(1e-6), _minimum((0.2, 0.3, 0.1), _ENERGY_EV + 1e-4)]
    distances = measure_uep_minima(results, source_structure=_COF2)

    for tolerance in (0.05, 0.25, 1.0):
        assert group_uep_minima(
            results, source_structure=_COF2, tolerance_ang=tolerance, distances=distances
        ) == group_uep_minima(results, source_structure=_COF2, tolerance_ang=tolerance)


def test_distances_of_other_results_are_measured_again() -> None:
    stale = measure_uep_minima(_noisy_images(1e-6, count=4), source_structure=_COF2)
    results = _noisy_images(1e-6, count=8)

    summary = group_uep_minima(results, source_structure=_COF2, distances=stale)

    assert [cluster.size for cluster in summary.clusters] == [8]


def test_a_supercell_job_groups_on_the_cell_of_its_input_structure() -> None:
    # Positions in a 2x2x2 job's Cartesian frame fold into the unit-cell orbit.
    doubled = tuple(tuple(2.0 * value for value in row) for row in _LATTICE)
    results = [
        fake_optimization(position, classical=_ENERGY_EV, lattice_vectors=doubled)
        for position in ((0.0, 0.25, 0.125), (0.5, 0.75, 0.625), (0.25, 0.5, 0.875))
    ]

    (cluster,) = group_uep_minima(results, source_structure=_COF2).clusters

    assert cluster.size == 3
    assert cluster.representative_fractional_position == pytest.approx((0.0, 0.5, 0.25))


def test_a_host_without_detectable_symmetry_says_so_and_groups_by_position() -> None:
    host = Structure(
        label="Unknown",
        lattice_vectors=_COF2.lattice_vectors,
        species=["Xx"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    summary = group_uep_minima(
        [_minimum((0.0, 0.5, 0.25)), _minimum((0.5, 0.0, 0.75))], source_structure=host
    )

    assert len(summary.clusters) == 2
    assert "treated as having no symmetry (P1)" in uep_grouping_report(summary)


@pytest.mark.parametrize(
    "keywords", [{"tolerance_ang": 0.0}, {"tolerance_ang": -1.0}, {"energy_tolerance_ev": 0.0}]
)
def test_a_non_positive_tolerance_is_refused(keywords: dict[str, float]) -> None:
    with pytest.raises(UepError, match="tolerance must be positive"):
        group_uep_minima(_noisy_images(1e-7, count=2), source_structure=_COF2, **keywords)


def test_a_grouping_round_trips_through_its_saved_form() -> None:
    summary = group_uep_minima(
        [*_noisy_images(1e-7, count=4), _minimum((0.0, 0.5, 0.0), label="saddle")],
        source_structure=_COF2,
    )

    assert uep_batch_summary_from_dict(json.loads(json.dumps(asdict(summary)))) == summary
    assert summary.clustering_method == UEP_GROUPING_METHOD
    assert summary.threshold_ang == DEFAULT_GROUPING_TOLERANCE_ANG


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"clustering_method": "hier", "threshold_ang": 0.3, "clusters": []},
        {"clustering_method": "kmeans", "threshold_ang": 0.0, "clusters": []},
        {"threshold_ang": 0.2, "clusters": []},
        {"clustering_method": UEP_GROUPING_METHOD, "clusters": []},
        {"clustering_method": UEP_GROUPING_METHOD, "threshold_ang": 0.25, "clusters": "none"},
        {"clustering_method": UEP_GROUPING_METHOD, "threshold_ang": 0.25, "clusters": [{}]},
    ],
    ids=["none", "hier", "kmeans", "no-method", "no-tolerance", "no-clusters", "bad-cluster"],
)
def test_a_retired_or_malformed_grouping_is_not_read(payload: object) -> None:
    assert uep_batch_summary_from_dict(payload) is None


def test_the_search_log_replaces_its_account_of_the_grouping() -> None:
    summary = group_uep_minima(_noisy_images(1e-7), source_structure=_COF2)
    old_log = "Requested random starts: 12\nRelaxed sites available for clustering: 12"

    log = uep_search_log(old_log, summary)

    assert log.splitlines() == [
        "Requested random starts: 12",
        "Grouping: 12 minima in 1 site, within 0.25 Å and 1 meV.",
    ]
    assert uep_search_log(log, summary) == log


def test_site_numbers_follow_the_classical_energy_of_the_included_results() -> None:
    results = [
        _minimum((0.1, 0.1, 0.1), 2.0),
        _minimum((0.2, 0.2, 0.2), -1.0),
        _minimum((0.3, 0.3, 0.3), float("nan")),
        _minimum((0.4, 0.4, 0.4), -3.0),
    ]

    assert relaxed_site_number_map(results) == {3: 1, 1: 2, 0: 3, 2: 4}
    assert relaxed_site_number_map(results, included=(0, 1)) == {1: 1, 0: 2}
