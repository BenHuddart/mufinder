"""Tests for the shared lattice algebra in :mod:`mufinder.science.lattice`.

Each migrated caller gets an equivalence test against a copy of the
implementation it used before the helpers were consolidated: bit-for-bit where
the old code used the same arithmetic, within a tight tolerance where it used a
different NumPy route.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mufinder.science.lattice import (
    LatticeError,
    cartesian_to_fractional,
    cell_volume,
    deduplicate_periodic_positions,
    fractional_to_cartesian,
    inverse_lattice,
    lattice_cart_to_frac,
    lattice_frac_to_cart,
    lattice_matrix,
    minimum_image_cartesian_delta,
    minimum_image_delta,
    minimum_image_distance,
    minimum_image_distance_matrix,
    near_minimum_indexes,
    nearest_image_distance,
    reciprocal_to_cartesian,
    vector_length,
    wrap_fractional,
)

_CUBIC = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
_TRICLINIC = [[10.0, 0.0, 0.0], [0.1, 9.0, 0.0], [0.2, 0.3, 8.0]]
_SKEWED = [[4.0, 0.0, 0.0], [3.6, 1.5, 0.0], [0.7, 0.4, 3.9]]
_SINGULAR = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]


def _random_pairs(count: int = 200, seed: int = 7) -> list[tuple[list[float], list[float]]]:
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1.5, 2.5, size=(count, 2, 3))
    return [(pair[0].tolist(), pair[1].tolist()) for pair in points]


# Copies of the implementations the shared helpers replaced.


def _site_generation_frac_to_cart(fractional, lattice):
    x = fractional[0] * lattice[0][0] + fractional[1] * lattice[1][0] + fractional[2] * lattice[2][0]
    y = fractional[0] * lattice[0][1] + fractional[1] * lattice[1][1] + fractional[2] * lattice[2][1]
    z = fractional[0] * lattice[0][2] + fractional[1] * lattice[1][2] + fractional[2] * lattice[2][2]
    return [x, y, z]


def _site_generation_pbc_distance(a, b, lattice):
    delta = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    delta = [d - round(d) for d in delta]
    cart = _site_generation_frac_to_cart(delta, lattice)
    return math.sqrt(cart[0] ** 2 + cart[1] ** 2 + cart[2] ** 2)


def _relaxation_analysis_pbc_distance(left, right, lattice):
    delta = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    delta = delta - np.round(delta)
    return float(np.linalg.norm(np.asarray(_site_generation_frac_to_cart(delta.tolist(), lattice), dtype=float)))


def _structure_io_minimum_periodic_distance(a, b, lattice_vectors):
    lattice = np.asarray(lattice_vectors, dtype=float)
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    delta -= np.round(delta)
    best = float("inf")
    for ox in (-1.0, 0.0, 1.0):
        for oy in (-1.0, 0.0, 1.0):
            for oz in (-1.0, 0.0, 1.0):
                best = min(best, float(np.linalg.norm((delta + np.array([ox, oy, oz])) @ lattice)))
    return best


def _magnetic_fields_fractional_distance(a, b):
    delta = np.mod(np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 0.5, 1.0) - 0.5
    return float(np.linalg.norm(delta))


def _legacy_import_fold_to_host(position, matrix):
    x, y, z = position
    folded = (
        x * matrix[0][0] + y * matrix[1][0] + z * matrix[2][0],
        x * matrix[0][1] + y * matrix[1][1] + z * matrix[2][1],
        x * matrix[0][2] + y * matrix[1][2] + z * matrix[2][2],
    )
    return tuple(float(value) % 1.0 for value in folded)


# Validation


def test_lattice_matrix_and_rows_reject_non_3x3_input_with_the_callers_error() -> None:
    class CallerError(ValueError):
        pass

    with pytest.raises(LatticeError, match="3x3 matrix"):
        lattice_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(CallerError, match="3x3 matrix"):
        lattice_matrix([[1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], error=CallerError)
    with pytest.raises(CallerError, match="3x3 matrix"):
        lattice_frac_to_cart([0.0, 0.0, 0.0], [[1.0, 0.0, 0.0]], error=CallerError)


def test_inverse_rejects_singular_lattices() -> None:
    with pytest.raises(LatticeError, match="singular"):
        inverse_lattice(_SINGULAR)
    with pytest.raises(LatticeError, match="singular"):
        lattice_cart_to_frac([1.0, 2.0, 3.0], _SINGULAR)


def test_broadcast_conversions_reject_points_that_are_not_3_vectors() -> None:
    with pytest.raises(LatticeError, match="Fractional positions must be 3-vectors"):
        fractional_to_cartesian(np.zeros((2, 4)), _CUBIC)
    with pytest.raises(LatticeError, match="Cartesian positions must be 3-vectors"):
        cartesian_to_fractional(np.zeros((2, 2)), _CUBIC)


# Conversions


def test_scalar_and_broadcast_conversions_agree_bit_for_bit() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(-1.0, 2.0, size=(50, 3))

    cartesian = fractional_to_cartesian(points, _TRICLINIC)
    for point, row in zip(points.tolist(), cartesian.tolist(), strict=True):
        assert lattice_frac_to_cart(point, _TRICLINIC) == row
        assert lattice_frac_to_cart(point, _TRICLINIC) == _site_generation_frac_to_cart(point, _TRICLINIC)

    fractional = cartesian_to_fractional(cartesian, _TRICLINIC)
    for point, row in zip(cartesian.tolist(), fractional.tolist(), strict=True):
        assert lattice_cart_to_frac(point, _TRICLINIC) == row
    # The inverse is applied to the Cartesian row vector, so the round trip holds for any lattice.
    assert fractional == pytest.approx(points, abs=1e-12)


def test_cartesian_to_fractional_undoes_a_non_symmetric_lattice() -> None:
    hexagonal = [[3.0, 0.0, 0.0], [-1.5, 2.598, 0.0], [0.0, 0.0, 5.0]]
    cartesian = lattice_frac_to_cart([0.1, 0.2, 0.3], hexagonal)

    assert lattice_cart_to_frac(cartesian, hexagonal) == pytest.approx([0.1, 0.2, 0.3], abs=1e-12)


def test_conversions_match_the_uep_matrix_route() -> None:
    rng = np.random.default_rng(11)
    points = rng.uniform(-1.0, 2.0, size=(50, 3))
    lattice = np.asarray(_SKEWED)

    assert fractional_to_cartesian(points, lattice) == pytest.approx(points @ lattice, abs=1e-12)
    cartesian = points @ lattice
    assert cartesian_to_fractional(cartesian, lattice) == pytest.approx(
        np.linalg.solve(lattice.T, cartesian.T).T, abs=1e-12
    )


def test_cell_volume_and_vector_length() -> None:
    assert cell_volume(_CUBIC) == pytest.approx(125.0)
    assert cell_volume(_TRICLINIC) == pytest.approx(abs(np.linalg.det(np.asarray(_TRICLINIC))))
    for vector in ([3.0, 4.0, 0.0], [0.1, -0.2, 0.3], [1e-9, 2.0, -7.5]):
        assert vector_length(vector) == float(sum(float(value) ** 2 for value in vector) ** 0.5)


def test_folding_a_supercell_position_matches_the_legacy_import_arithmetic() -> None:
    matrix = ((2, 0, 0), (1, 3, 0), (0, -1, 2))
    for position, _ in _random_pairs(50):
        folded = tuple(wrap_fractional(lattice_frac_to_cart(position, matrix)))
        assert folded == _legacy_import_fold_to_host(position, matrix)


# Minimum image


def test_minimum_image_delta_folds_each_component() -> None:
    assert minimum_image_delta([0.9, 0.1, 2.25], [0.1, 0.9, 0.0]) == pytest.approx([-0.2, 0.2, 0.25])
    assert minimum_image_cartesian_delta([0.9, 0.0, 0.0], [0.1, 0.0, 0.0], _CUBIC) == pytest.approx([-1.0, 0.0, 0.0])
    assert minimum_image_cartesian_delta([0.9, 0.0, 0.0], [0.1, 0.0, 0.0]) == pytest.approx([-0.2, 0.0, 0.0])


@pytest.mark.parametrize("lattice", [_CUBIC, _TRICLINIC, _SKEWED], ids=["cubic", "triclinic", "skewed"])
def test_minimum_image_distance_matches_site_generation_bit_for_bit(lattice) -> None:
    for a, b in _random_pairs():
        assert minimum_image_distance(a, b, lattice) == _site_generation_pbc_distance(a, b, lattice)


@pytest.mark.parametrize("lattice", [_CUBIC, _TRICLINIC, _SKEWED], ids=["cubic", "triclinic", "skewed"])
def test_minimum_image_distance_matches_relaxation_analysis(lattice) -> None:
    for a, b in _random_pairs():
        assert minimum_image_distance(a, b, lattice) == pytest.approx(
            _relaxation_analysis_pbc_distance(b, a, lattice), rel=1e-14, abs=1e-14
        )


def test_fractional_minimum_image_distance_matches_magnetic_fields() -> None:
    for a, b in _random_pairs():
        assert minimum_image_distance(a, b) == pytest.approx(_magnetic_fields_fractional_distance(a, b), abs=1e-12)
    # Exactly half a cell apart: both fold to the same length.
    assert minimum_image_distance([0.5, 0.0, 0.0], [0.0, 0.0, 0.0]) == _magnetic_fields_fractional_distance(
        [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]
    )


@pytest.mark.parametrize("lattice", [_CUBIC, _TRICLINIC, _SKEWED], ids=["cubic", "triclinic", "skewed"])
def test_nearest_image_distance_matches_structure_io(lattice) -> None:
    for a, b in _random_pairs():
        assert nearest_image_distance(a, b, lattice) == pytest.approx(
            _structure_io_minimum_periodic_distance(a, b, lattice), abs=1e-12
        )


def test_nearest_image_distance_can_beat_the_per_component_fold_in_a_skewed_cell() -> None:
    a, b = [0.0, 0.0, 0.0], [0.45, 0.45, 0.0]

    assert nearest_image_distance(a, b, _SKEWED) < minimum_image_distance(a, b, _SKEWED)


def test_distance_matrix_matches_the_scalar_helper() -> None:
    rng = np.random.default_rng(5)
    left = rng.uniform(0.0, 1.0, size=(6, 3))
    right = rng.uniform(0.0, 1.0, size=(4, 3))

    for lattice in (_TRICLINIC, None):
        matrix = minimum_image_distance_matrix(left, right, lattice)
        assert matrix.shape == (6, 4)
        for i, j in np.ndindex(matrix.shape):
            assert matrix[i, j] == pytest.approx(minimum_image_distance(right[j], left[i], lattice), rel=1e-14)


def test_near_minimum_indexes_keeps_every_near_tie() -> None:
    distances = np.asarray([2.0, 1.0, 1.0 + 1e-12, 1.5])

    assert near_minimum_indexes(distances).tolist() == [1, 2]


# Deduplication


def test_deduplication_tolerance_is_inclusive_at_the_exact_pair_distance() -> None:
    first, second = [0.1, 0.2, 0.3], [0.1 + 3e-7, 0.2 - 1e-7, 0.3 + 2e-7]
    pair_distance = minimum_image_distance(second, first, _CUBIC)

    def deduplicate(tolerance: float) -> list[list[float]]:
        return deduplicate_periodic_positions([first, second], _CUBIC, tolerance=tolerance)

    # Broadcast norms can be an ulp off, so decisions this close are re-measured with the scalar helper.
    assert deduplicate(pair_distance) == [first]
    assert deduplicate(math.nextafter(pair_distance, 0.0)) == [first, second]
    assert deduplicate(math.nextafter(pair_distance, 1.0)) == [first]
    assert deduplicate_periodic_positions([], _CUBIC, tolerance=1e-5) == []


def test_deduplication_keeps_first_occurrence_and_drops_exact_repeats() -> None:
    positions = [[1.25, -0.5, 0.0], [0.5, 0.5, 0.5], [0.25, 0.5, 0.0], [0.5, 0.5, 0.5 + 1e-7]]

    unique = deduplicate_periodic_positions(positions, _CUBIC, tolerance=1e-5)

    assert unique == [[0.25, 0.5, 0.0], [0.5, 0.5, 0.5]]


def test_deduplication_without_a_lattice_uses_fractional_distances() -> None:
    positions = [[0.0, 0.0, 0.0], [0.9995, 0.0, 0.0], [0.002, 0.0, 0.0]]

    assert deduplicate_periodic_positions(positions, None, tolerance=1e-3) == [[0.0, 0.0, 0.0], [0.002, 0.0, 0.0]]


def test_reciprocal_to_cartesian_gives_the_dual_basis() -> None:
    lattice = [[3.0, 0.0, 0.0], [-1.5, 1.5 * math.sqrt(3.0), 0.0], [0.0, 0.0, 5.0]]

    a_star = reciprocal_to_cartesian((1.0, 0.0, 0.0), lattice)
    q = reciprocal_to_cartesian((0.25, -0.5, 2.0), lattice)

    assert np.allclose(np.asarray(lattice) @ a_star, (1.0, 0.0, 0.0))
    assert np.allclose(np.asarray(lattice) @ q, (0.25, -0.5, 2.0))


def test_reciprocal_to_cartesian_rejects_a_singular_lattice_with_the_callers_error() -> None:
    class _Error(ValueError):
        pass

    with pytest.raises(_Error, match="singular"):
        reciprocal_to_cartesian((1.0, 0.0, 0.0), [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]], error=_Error)
