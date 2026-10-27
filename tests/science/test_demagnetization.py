"""Tests for demagnetizing tensor helpers."""

from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from mufinder.science.demagnetization import (
    DemagnetizationError,
    DemagnetizingSample,
    PlatePhoto,
    VoxelMask,
    demagnetizing_factors_ellipsoid,
    demagnetizing_factors_rectangular_prism,
    demagnetizing_tensor,
    demagnetizing_tensor_field,
    fourier_box_shape,
    fourier_demagnetizing_result,
    fourier_k_grid,
    magnetometric_demagnetizing_tensor,
    magnetometric_rectangular_prism_tensor,
    principal_demagnetizing_tensor,
    rectangular_prism_shape,
    rectangular_prism_tensor,
    sample_fourier_box,
    sample_orientation,
    voxel_demagnetizing_tensor,
)

_CUBIC = np.eye(3) * 4.0
_HEXAGONAL = np.array([[3.0, 0.0, 0.0], [-1.5, 1.5 * np.sqrt(3.0), 0.0], [0.0, 0.0, 5.0]])


def _prolate_closed_form(a: float, c: float) -> tuple[float, float, float]:
    e = np.sqrt(1.0 - (a / c) ** 2)
    nz = (1.0 - e * e) / (2.0 * e**3) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)
    return (0.5 * (1.0 - nz), 0.5 * (1.0 - nz), nz)


def _oblate_closed_form(a: float, c: float) -> tuple[float, float, float]:
    eta = np.sqrt((a / c) ** 2 - 1.0)
    nz = (1.0 + eta * eta) / eta**3 * (eta - np.arctan(eta))
    return (0.5 * (1.0 - nz), 0.5 * (1.0 - nz), nz)


def _rotation_about_z(angle_deg: float) -> np.ndarray:
    t = np.radians(angle_deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])


def _prism(shape: str) -> DemagnetizingSample:
    return DemagnetizingSample(
        shape=shape, prism_lengths=(1.0, 1.0, 2.0), fourier_grid=8, padding=2.5
    )


# demagnetizing_tensor: shape dispatch


def test_demagnetizing_tensor_supports_simple_limits() -> None:
    np.testing.assert_allclose(
        demagnetizing_tensor(DemagnetizingSample(shape="sphere")), np.eye(3) / 3.0
    )
    np.testing.assert_allclose(
        demagnetizing_tensor(DemagnetizingSample(shape="infinite_plate", plate_normal_axis="b")),
        np.diag([0.0, 1.0, 0.0]),
    )


@pytest.mark.parametrize("shape", ["none", "", "OFF", "no demag"])
def test_demagnetizing_tensor_no_demag_aliases_return_zeros(shape: str) -> None:
    np.testing.assert_array_equal(
        demagnetizing_tensor(DemagnetizingSample(shape=shape)), np.zeros((3, 3))
    )


@pytest.mark.parametrize("shape", ["Plate", "infiniteplate", "infinite plate"])
def test_demagnetizing_tensor_plate_aliases_select_infinite_plate(shape: str) -> None:
    np.testing.assert_allclose(
        demagnetizing_tensor(DemagnetizingSample(shape=shape, plate_normal_axis="c")),
        np.diag([0.0, 0.0, 1.0]),
    )


@pytest.mark.parametrize(
    ("axis", "index"),
    [("a", 0), ("x", 0), ("B", 1), ("y", 1), ("c", 2), ("Z", 2)],
)
def test_infinite_plate_normal_accepts_lattice_and_cartesian_axes(axis: str, index: int) -> None:
    expected = np.zeros((3, 3))
    expected[index, index] = 1.0

    tensor = demagnetizing_tensor(DemagnetizingSample(shape="plate", plate_normal_axis=axis))

    np.testing.assert_allclose(tensor, expected)


def test_infinite_plate_rejects_unknown_normal_axis() -> None:
    with pytest.raises(DemagnetizationError, match="plate_normal_axis"):
        demagnetizing_tensor(DemagnetizingSample(shape="plate", plate_normal_axis="q"))


def test_demagnetizing_tensor_ellipsoid_branch_diagonal() -> None:
    sample = DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 1.0, 4.0))
    tensor = demagnetizing_tensor(sample)

    # An aligned ellipsoid has a diagonal tensor of unit trace.
    np.testing.assert_allclose(tensor - np.diag(np.diag(tensor)), 0.0, atol=1e-12)
    assert float(np.trace(tensor)) == pytest.approx(1.0)


def test_demagnetizing_tensor_rejects_ellipsoid_without_three_semi_axes() -> None:
    with pytest.raises(DemagnetizationError, match="semi_axes must contain three numbers"):
        demagnetizing_tensor(DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 2.0)))  # type: ignore[arg-type]


def test_demagnetizing_tensor_rejects_non_positive_semi_axes() -> None:
    with pytest.raises(DemagnetizationError, match="semi_axes values must be positive"):
        demagnetizing_tensor(DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, -2.0, 3.0)))


def test_demagnetizing_tensor_rectangular_prism_branch_is_the_aharoni_diagonal() -> None:
    sample = DemagnetizingSample(shape="rectangular_prism", prism_lengths=(3.0, 2.0, 0.5))
    tensor = demagnetizing_tensor(sample)

    np.testing.assert_allclose(
        tensor, np.diag(demagnetizing_factors_rectangular_prism((3.0, 2.0, 0.5))), atol=1e-15
    )
    assert float(np.trace(tensor)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("shape", ["Box", "cuboid", "Rectangular"])
def test_demagnetizing_tensor_prism_aliases_select_rectangular_prism(shape: str) -> None:
    np.testing.assert_allclose(
        demagnetizing_tensor(_prism(shape)), demagnetizing_tensor(_prism("rectangular_prism"))
    )


def test_demagnetizing_tensor_rejects_unknown_shape() -> None:
    with pytest.raises(DemagnetizationError, match="unknown"):
        demagnetizing_tensor(DemagnetizingSample(shape="unknown_shape"))


# demagnetizing_factors_ellipsoid


def test_ellipsoid_sphere_factors_are_one_third() -> None:
    np.testing.assert_allclose(
        demagnetizing_factors_ellipsoid(2.0, 2.0, 2.0), (1.0 / 3.0,) * 3, rtol=0.0, atol=1e-15
    )


@pytest.mark.parametrize("c", [1.0001, 1.5, 4.0, 100.0])
def test_ellipsoid_prolate_spheroid_matches_closed_form(c: float) -> None:
    np.testing.assert_allclose(
        demagnetizing_factors_ellipsoid(1.0, 1.0, c), _prolate_closed_form(1.0, c), atol=1e-12
    )


@pytest.mark.parametrize("a", [1.0001, 1.5, 5.0, 100.0])
def test_ellipsoid_oblate_spheroid_matches_closed_form(a: float) -> None:
    np.testing.assert_allclose(
        demagnetizing_factors_ellipsoid(a, a, 1.0), _oblate_closed_form(a, 1.0), atol=1e-12
    )


@pytest.mark.parametrize(
    "axes",
    [(1.0, 2.0, 4.0), (3.0, 0.2, 7.0), (1e-6, 1.0, 1e6), (1e9, 2e9, 3e9)],
    ids=["triaxial", "flat-needle", "extreme", "large"],
)
def test_ellipsoid_factors_sum_to_one_without_renormalisation(
    axes: tuple[float, float, float],
) -> None:
    factors = demagnetizing_factors_ellipsoid(*axes)

    assert abs(sum(factors) - 1.0) <= 1e-12
    assert all(0.0 <= f <= 1.0 for f in factors)


def test_ellipsoid_triaxial_factor_is_largest_along_the_shortest_axis() -> None:
    factors = demagnetizing_factors_ellipsoid(1.0, 2.0, 4.0)

    assert factors[0] > factors[1] > factors[2]


def test_ellipsoid_two_infinite_axes_gives_plate_limit() -> None:
    # Two infinite axes -> finite axis carries the full factor of 1.
    factors = demagnetizing_factors_ellipsoid(np.inf, np.inf, 1.0)
    np.testing.assert_allclose(factors, (0.0, 0.0, 1.0))


def test_ellipsoid_single_infinite_axis_raises() -> None:
    with pytest.raises(DemagnetizationError, match="infinite axes"):
        demagnetizing_factors_ellipsoid(np.inf, 1.0, 1.0)


@pytest.mark.parametrize("axis", [0.0, np.nan], ids=["zero", "nan"])
def test_ellipsoid_invalid_axis_raises(axis: float) -> None:
    with pytest.raises(DemagnetizationError, match="positive"):
        demagnetizing_factors_ellipsoid(axis, 1.0, 1.0)


# Sample orientation


def test_aligned_sample_tensor_is_the_principal_tensor() -> None:
    sample = DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 2.0, 4.0))

    assert not sample.is_oriented
    np.testing.assert_array_equal(
        demagnetizing_tensor(sample), np.diag(demagnetizing_factors_ellipsoid(1.0, 2.0, 4.0))
    )


def test_spheroid_along_110_matches_aligned_spheroid_with_direction_rotated_back() -> None:
    aligned = DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 1.0, 3.0))
    oriented = DemagnetizingSample.oriented(
        _CUBIC, (1.0, 1.0, 0.0), (0.0, 0.0, 1.0), shape="ellipsoid", semi_axes=(1.0, 1.0, 3.0)
    )
    rotation = oriented.rotation_matrix()
    np.testing.assert_allclose(rotation[:, 2], np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0))
    np.testing.assert_allclose(rotation[:, 0], (0.0, 0.0, 1.0), atol=1e-15)

    tensor = demagnetizing_tensor(oriented)
    aligned_tensor = demagnetizing_tensor(aligned)
    for m in ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.3, -0.4, 0.8)):
        m_hat = np.asarray(m) / np.linalg.norm(m)
        np.testing.assert_allclose(
            tensor @ m_hat, rotation @ (aligned_tensor @ (rotation.T @ m_hat)), atol=1e-14
        )
    # Along the long axis the effective factor is the small axial one.
    assert oriented.effective_demagnetizing_factor((1.0, 1.0, 0.0)) == pytest.approx(
        demagnetizing_factors_ellipsoid(1.0, 1.0, 3.0)[2], abs=1e-14
    )
    np.testing.assert_allclose(tensor, tensor.T)
    assert tensor[0, 1] != pytest.approx(0.0)


@pytest.mark.parametrize(
    ("normal_hkl", "in_plane_uvw", "reciprocal_row"),
    [((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), 2), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 0)],
    ids=["001", "100"],
)
def test_plate_normal_to_hkl_of_hexagonal_cell_has_factor_one_along_reciprocal_vector(
    normal_hkl: tuple[float, float, float],
    in_plane_uvw: tuple[float, float, float],
    reciprocal_row: int,
) -> None:
    plate = DemagnetizingSample.oriented(
        _HEXAGONAL, normal_hkl, in_plane_uvw, shape="infinite_plate", plate_normal_axis="c"
    )
    reciprocal = np.linalg.inv(_HEXAGONAL).T[reciprocal_row]

    assert plate.effective_demagnetizing_factor(reciprocal) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(np.trace(demagnetizing_tensor(plate)), 1.0)
    # The in-plane lattice direction carries no demagnetizing field.
    in_plane = np.asarray(in_plane_uvw) @ _HEXAGONAL
    assert plate.effective_demagnetizing_factor(in_plane) == pytest.approx(0.0, abs=1e-14)


def test_hexagonal_100_normal_is_not_along_the_a_lattice_vector() -> None:
    plate = DemagnetizingSample.oriented(
        _HEXAGONAL, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), shape="infinite_plate"
    )

    # a* is 30 degrees from a in a hexagonal cell, so N along a is cos^2(30).
    assert plate.effective_demagnetizing_factor(_HEXAGONAL[0]) == pytest.approx(0.75)


def test_sample_orientation_reports_angle_of_non_orthogonal_pair() -> None:
    orientation = sample_orientation(_CUBIC, (1.0, 1.0, 0.0), (1.0, 0.0, 0.0))

    assert orientation.angle_deg == pytest.approx(45.0)
    assert not orientation.orthogonal
    rotation = np.asarray(orientation.rotation)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation[:, 0], np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0))


def test_sample_orientation_of_orthogonal_pair_is_orthogonal() -> None:
    orientation = sample_orientation(_HEXAGONAL, (0.0, 0.0, 1.0), (1.0, 1.0, 0.0))

    assert orientation.angle_deg == pytest.approx(90.0)
    assert orientation.orthogonal


def test_sample_orientation_rejects_parallel_directions() -> None:
    with pytest.raises(DemagnetizationError, match="parallel"):
        sample_orientation(_CUBIC, (1.0, 1.0, 0.0), (2.0, 2.0, 0.0))


@pytest.mark.parametrize(
    ("normal_hkl", "in_plane_uvw", "match"),
    [
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), "normal"),
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), "in-plane"),
        ((0.0, np.nan, 1.0), (1.0, 0.0, 0.0), "finite"),
        ((0.0, 1.0), (1.0, 0.0, 0.0), "three numbers"),
    ],
    ids=["zero-normal", "zero-in-plane", "nan", "two-values"],
)
def test_sample_orientation_rejects_unusable_indices(
    normal_hkl: object, in_plane_uvw: object, match: str
) -> None:
    with pytest.raises(DemagnetizationError, match=match):
        sample_orientation(_CUBIC, normal_hkl, in_plane_uvw)  # type: ignore[arg-type]


def test_sample_orientation_rejects_singular_lattice() -> None:
    with pytest.raises(DemagnetizationError, match="singular"):
        sample_orientation(np.zeros((3, 3)), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))


def test_sample_accepts_an_explicit_proper_rotation() -> None:
    rotation = _rotation_about_z(30.0)
    sample = DemagnetizingSample(
        shape="ellipsoid", semi_axes=(1.0, 2.0, 2.0), rotation=rotation.tolist()
    )

    expected = rotation @ np.diag(demagnetizing_factors_ellipsoid(1.0, 2.0, 2.0)) @ rotation.T
    np.testing.assert_allclose(demagnetizing_tensor(sample), expected, atol=1e-15)


def test_sample_rejects_non_orthogonal_rotation() -> None:
    shear = np.array([[1.0, 0.2, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(DemagnetizationError, match="orthonormal"):
        DemagnetizingSample(shape="sphere", rotation=shear.tolist())


def test_sample_rejects_left_handed_rotation() -> None:
    mirror = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(DemagnetizationError, match="left-handed"):
        DemagnetizingSample(shape="sphere", rotation=mirror.tolist())


@pytest.mark.parametrize(
    "rotation", [[[1.0, 0.0], [0.0, 1.0]], "rotation", np.full((3, 3), np.nan).tolist()]
)
def test_sample_rejects_malformed_rotation(rotation: object) -> None:
    with pytest.raises(DemagnetizationError, match="3x3"):
        DemagnetizingSample(shape="sphere", rotation=rotation)  # type: ignore[arg-type]


def test_sample_rejects_zero_miller_indices() -> None:
    with pytest.raises(DemagnetizationError, match="non-zero"):
        DemagnetizingSample(shape="sphere", normal_hkl=(0, 0, 1), in_plane_uvw=(0, 0, 0))


def test_principal_tensor_ignores_rotation_and_rotate_applies_it() -> None:
    rotation = _rotation_about_z(30.0)
    sample = DemagnetizingSample(
        shape="ellipsoid", semi_axes=(1.0, 2.0, 2.0), rotation=rotation.tolist()
    )
    principal = principal_demagnetizing_tensor(sample)

    np.testing.assert_array_equal(
        principal, np.diag(demagnetizing_factors_ellipsoid(1.0, 2.0, 2.0))
    )
    np.testing.assert_allclose(sample.rotate(principal), demagnetizing_tensor(sample))
    aligned = DemagnetizingSample(shape="sphere")
    assert aligned.rotate(principal) is principal


def test_sample_needs_both_crystallographic_directions() -> None:
    with pytest.raises(DemagnetizationError, match="both"):
        DemagnetizingSample(shape="sphere", normal_hkl=(0.0, 0.0, 1.0))


def test_unresolved_crystallographic_orientation_raises_for_tensor() -> None:
    sample = DemagnetizingSample(
        shape="sphere", normal_hkl=(0.0, 0.0, 1.0), in_plane_uvw=(1.0, 0.0, 0.0)
    )

    with pytest.raises(DemagnetizationError, match="resolved"):
        demagnetizing_tensor(sample)


def test_resolve_orientation_recomputes_rotation_for_a_new_lattice() -> None:
    sample = DemagnetizingSample.oriented(
        _CUBIC, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), shape="infinite_plate"
    )
    resolved = sample.resolve_orientation(_HEXAGONAL)

    np.testing.assert_allclose(sample.rotation_matrix()[:, 2], (1.0, 0.0, 0.0), atol=1e-15)
    np.testing.assert_allclose(
        resolved.rotation_matrix()[:, 2], (np.sqrt(3.0) / 2.0, 0.5, 0.0), atol=1e-15
    )
    aligned = DemagnetizingSample(shape="sphere")
    assert aligned.resolve_orientation(_HEXAGONAL) is aligned


@pytest.mark.parametrize("direction", [(0.0, 0.0, 0.0), (1.0, 2.0), (np.inf, 0.0, 0.0)])
def test_effective_demagnetizing_factor_rejects_unusable_direction(direction: object) -> None:
    with pytest.raises(DemagnetizationError, match="direction"):
        DemagnetizingSample(shape="sphere").effective_demagnetizing_factor(direction)  # type: ignore[arg-type]


# Rectangular prism: Aharoni's closed form


def test_aharoni_cube_factors_are_one_third() -> None:
    np.testing.assert_allclose(
        demagnetizing_factors_rectangular_prism((2.0, 2.0, 2.0)), np.full(3, 1.0 / 3.0), atol=1e-14
    )


def test_aharoni_factors_tend_to_the_plate_limit() -> None:
    factors = demagnetizing_factors_rectangular_prism((1.0, 1.0, 1e-5))

    np.testing.assert_allclose(factors, (0.0, 0.0, 1.0), atol=1e-4)


@pytest.mark.parametrize("ratio", [1e-5, 1e-8, 1e-150])
def test_aharoni_factors_stay_exact_for_extreme_aspect_ratios(ratio: float) -> None:
    # The factors are accurate to about 1e-40 absolute, whatever the ratio.
    needle = demagnetizing_factors_rectangular_prism((1.0, ratio, ratio))
    plate = demagnetizing_factors_rectangular_prism((1.0, 1.0, ratio))

    for factors in (needle, plate):
        assert sum(factors) == pytest.approx(1.0, abs=1e-14)
        assert min(factors) >= 0.0
    np.testing.assert_allclose(needle, (0.0, 0.5, 0.5), atol=10 * ratio + 1e-30)
    assert needle[1] == needle[2]
    np.testing.assert_allclose(plate, (0.0, 0.0, 1.0), atol=10 * ratio + 1e-30)
    assert plate[0] == plate[1]


def test_aharoni_needle_factor_falls_in_proportion_to_its_width() -> None:
    # Along a long square rod the factor is proportional to width / length.
    wide = demagnetizing_factors_rectangular_prism((1.0, 1e-5, 1e-5))[0]
    narrow = demagnetizing_factors_rectangular_prism((1.0, 1e-8, 1e-8))[0]

    assert narrow / wide == pytest.approx(1e-3, rel=1e-3)


def test_aharoni_factors_tend_to_the_needle_limit() -> None:
    factors = demagnetizing_factors_rectangular_prism((1.0, 1.0, 1e5))

    np.testing.assert_allclose(factors, (0.5, 0.5, 0.0), atol=1e-4)


@pytest.mark.parametrize("lengths", [(3.0, 1.0, 0.2), (1.0, 7.0, 2.0), (50.0, 1.0, 1.0)])
def test_aharoni_factors_sum_to_one_and_follow_the_edges(lengths: tuple[float, ...]) -> None:
    factors = demagnetizing_factors_rectangular_prism(lengths)

    assert sum(factors) == pytest.approx(1.0, abs=1e-12)
    # The shortest edge carries the largest factor.
    assert np.argmax(factors) == np.argmin(lengths)
    np.testing.assert_allclose(
        demagnetizing_factors_rectangular_prism(lengths[::-1]), factors[::-1], atol=1e-14
    )


@pytest.mark.parametrize("lengths", [(0.0, 1.0, 1.0), (1.0, -1.0, 1.0), (1.0, np.inf, 1.0)])
def test_aharoni_factors_reject_unusable_lengths(lengths: tuple[float, ...]) -> None:
    with pytest.raises(DemagnetizationError, match="prism_lengths"):
        demagnetizing_factors_rectangular_prism(lengths)


def test_magnetometric_prism_tensor_is_the_diagonal_of_the_factors() -> None:
    np.testing.assert_array_equal(
        magnetometric_rectangular_prism_tensor((10.0, 10.0, 0.25)),
        np.diag(demagnetizing_factors_rectangular_prism((10.0, 10.0, 0.25))),
    )


def test_magnetometric_prism_tensor_warns_that_grid_arguments_are_ignored() -> None:
    with pytest.warns(DeprecationWarning, match="ignored"):
        tensor = magnetometric_rectangular_prism_tensor((1.0, 1.0, 1.0), grid_size=4, padding=1.0)

    np.testing.assert_allclose(tensor, np.eye(3) / 3.0, atol=1e-14)


# rectangular_prism_tensor (local Joseph-Schlomann tensor)


def test_rectangular_prism_local_tensor_is_symmetric() -> None:
    tensor = rectangular_prism_tensor(0.1, 0.2, 0.3, (1.0, 2.0, 3.0))

    assert tensor.shape == (3, 3)
    np.testing.assert_allclose(tensor, tensor.T, atol=1e-12)


def test_rectangular_prism_tensor_outside_x_correction() -> None:
    # A point clearly outside along x only triggers the nxx -1 correction.
    inside = rectangular_prism_tensor(0.0, 0.0, 0.0, (1.0, 1.0, 1.0))
    outside = rectangular_prism_tensor(5.0, 0.0, 0.0, (1.0, 1.0, 1.0))
    assert inside.shape == (3, 3)
    assert outside.shape == (3, 3)
    np.testing.assert_allclose(outside, outside.T, atol=1e-12)


def test_rectangular_prism_tensor_broadcasts_arrays() -> None:
    x = np.linspace(-0.5, 0.5, 3)[:, None, None]
    y = np.linspace(-0.5, 0.5, 3)[None, :, None]
    z = np.linspace(-0.5, 0.5, 3)[None, None, :]
    tensor = rectangular_prism_tensor(x, y, z, (1.0, 1.0, 1.0))
    assert tensor.shape == (3, 3, 3, 3, 3)
    # Symmetric in the tensor indices for every grid point.
    np.testing.assert_allclose(tensor[0, 1], tensor[1, 0], atol=1e-12)


def test_rectangular_prism_tensor_bad_half_lengths_raises() -> None:
    with pytest.raises(DemagnetizationError, match="positive"):
        rectangular_prism_tensor(0.0, 0.0, 0.0, (1.0, -1.0, 1.0))


# rectangular_prism_shape and fourier_k_grid


def test_rectangular_prism_shape_marks_interior() -> None:
    shape = rectangular_prism_shape((9, 9, 9), (0.25, 0.25, 0.25), 0.1)
    assert shape.dtype == bool
    assert shape.shape == (9, 9, 9)
    # Centre voxel (coord 0) is inside; outer voxels (coord 0.4 > 0.25) outside.
    assert bool(shape[4, 4, 4]) is True
    assert bool(shape[0, 0, 0]) is False


def test_rectangular_prism_shape_accepts_per_axis_spacing() -> None:
    scalar = rectangular_prism_shape((5, 5, 5), (1.0, 1.0, 1.0), 0.5)
    vector = rectangular_prism_shape((5, 5, 5), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5))

    assert vector.any()
    np.testing.assert_array_equal(scalar, vector)


def test_rectangular_prism_shape_zero_dim_raises() -> None:
    with pytest.raises(DemagnetizationError, match="positive"):
        rectangular_prism_shape((0, 5, 5), (1.0, 1.0, 1.0), 0.5)


def test_rectangular_prism_shape_rejects_two_component_spacing() -> None:
    with pytest.raises(DemagnetizationError, match="scalar or a three-vector"):
        rectangular_prism_shape((5, 5, 5), (1.0, 1.0, 1.0), (1.0, 2.0))


def test_fourier_k_grid_shapes_and_zero_origin() -> None:
    kx, ky, kz = fourier_k_grid((4, 6, 8), 0.5)
    assert kx.shape == ky.shape == kz.shape == (4, 6, 8)
    # The first element (DC component) is the reciprocal-space origin.
    assert kx[0, 0, 0] == 0.0
    assert ky[0, 0, 0] == 0.0
    assert kz[0, 0, 0] == 0.0


def test_fourier_k_grid_rejects_non_positive_spacing() -> None:
    with pytest.raises(DemagnetizationError, match="spacing values must be positive"):
        fourier_k_grid((4, 4, 4), 0.0)


# magnetometric_demagnetizing_tensor (Fourier path)


def test_magnetometric_demagnetizing_tensor_cube_is_isotropic() -> None:
    n = 16
    spacing = 1.0 / n
    grid_shape = (n, n, n)
    half = 0.25
    shape_grid = rectangular_prism_shape(grid_shape, (half, half, half), spacing)
    amplitude = np.fft.fftn(shape_grid.astype(float)) * (spacing**3)
    kx, ky, kz = fourier_k_grid(grid_shape, spacing)
    volume = (2 * half) ** 3
    tensor = magnetometric_demagnetizing_tensor(
        amplitude,
        kx,
        ky,
        kz,
        sample_volume=volume,
        domain_lengths=(1.0, 1.0, 1.0),
    )
    # With k = 0 on the diagonal the trace is 1 by Parseval's theorem, unnormalised.
    assert float(np.trace(tensor)) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(tensor, np.eye(3) / 3.0, atol=1e-12)


def test_magnetometric_demagnetizing_tensor_matches_the_voxel_path() -> None:
    mask = np.ones((4, 4, 2), dtype=bool)
    mask[0, 0, 0] = False
    spacing = (0.5, 0.75, 0.2)
    box = fourier_box_shape(mask.shape, spacing, 2.0)
    embedded = np.zeros(box)
    embedded[: mask.shape[0], : mask.shape[1], : mask.shape[2]] = mask
    voxel_volume = float(np.prod(spacing))
    kx, ky, kz = fourier_k_grid(box, spacing)
    tensor = magnetometric_demagnetizing_tensor(
        np.fft.fftn(embedded) * voxel_volume,
        kx,
        ky,
        kz,
        sample_volume=float(mask.sum()) * voxel_volume,
        domain_lengths=[n * s for n, s in zip(box, spacing, strict=True)],
    )

    voxel = voxel_demagnetizing_tensor(mask, spacing, padding=2.0).tensor

    # The full-spectrum sum counts each even-length axis's unpaired Nyquist
    # plane in the off-diagonal elements; the voxel path leaves it out.
    np.testing.assert_allclose(np.diag(voxel), np.diag(tensor), atol=1e-12)
    assert voxel[0, 1] == pytest.approx(tensor[0, 1], abs=2e-3)


def test_mirror_symmetric_voxel_shape_has_no_off_diagonal_elements() -> None:
    mask = np.zeros((7, 6, 4), dtype=bool)
    mask[1:6, :, :] = True
    mask[0, 2:4, 1:3] = True
    mask[6, 2:4, 1:3] = True
    tensor = voxel_demagnetizing_tensor(mask, (0.5, 0.75, 0.2), padding=2.0).tensor

    np.testing.assert_allclose(tensor - np.diag(np.diag(tensor)), 0.0, atol=1e-14)


def test_magnetometric_demagnetizing_tensor_zero_amplitude_gives_zero() -> None:
    grid_shape = (4, 4, 4)
    kx, ky, kz = fourier_k_grid(grid_shape, 1.0)
    amplitude = np.zeros(grid_shape, dtype=complex)
    tensor = magnetometric_demagnetizing_tensor(
        amplitude,
        kx,
        ky,
        kz,
        sample_volume=1.0,
        domain_lengths=(1.0, 1.0, 1.0),
    )
    np.testing.assert_array_equal(tensor, np.zeros((3, 3)))


def test_magnetometric_demagnetizing_tensor_nonpositive_volume_raises() -> None:
    grid_shape = (4, 4, 4)
    kx, ky, kz = fourier_k_grid(grid_shape, 1.0)
    amplitude = np.zeros(grid_shape, dtype=complex)
    with pytest.raises(DemagnetizationError, match="sample_volume"):
        magnetometric_demagnetizing_tensor(
            amplitude, kx, ky, kz, sample_volume=0.0, domain_lengths=(1.0, 1.0, 1.0)
        )


def test_magnetometric_demagnetizing_tensor_shape_mismatch_raises() -> None:
    kx, ky, kz = fourier_k_grid((4, 4, 4), 1.0)
    amplitude = np.zeros((4, 4, 3), dtype=complex)
    with pytest.raises(DemagnetizationError, match="identical shapes"):
        magnetometric_demagnetizing_tensor(
            amplitude, kx, ky, kz, sample_volume=1.0, domain_lengths=(1.0, 1.0, 1.0)
        )


# Fourier method for voxel shapes


def _voxel_sphere(n: int) -> np.ndarray:
    x = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    return x[:, None, None] ** 2 + x[None, :, None] ** 2 + x[None, None, :] ** 2 <= 1.0


def test_voxel_sphere_tensor_is_one_third_at_128_cubed() -> None:
    result = voxel_demagnetizing_tensor(_voxel_sphere(128), 1.0 / 64.0, padding=2.0)

    np.testing.assert_allclose(result.tensor, np.eye(3) / 3.0, rtol=0.01, atol=1e-12)
    assert result.box_shape == (256, 256, 256)


@pytest.mark.parametrize(
    ("lengths", "voxels"),
    [
        ((1.0, 1.0, 1.0), (16, 16, 16)),
        ((4.0, 4.0, 1.0), (32, 32, 4)),
        ((20.0, 20.0, 1.0), (64, 64, 2)),
        ((1.0, 1.0, 4.0), (8, 8, 32)),
    ],
    ids=["aspect-1", "plate-4", "plate-20", "needle-4"],
)
def test_voxel_prism_with_anisotropic_voxels_matches_aharoni(
    lengths: tuple[float, float, float], voxels: tuple[int, int, int]
) -> None:
    spacing = tuple(length / count for length, count in zip(lengths, voxels, strict=True))
    result = voxel_demagnetizing_tensor(np.ones(voxels, dtype=bool), spacing, padding=2.0)

    np.testing.assert_allclose(
        np.diag(result.tensor), demagnetizing_factors_rectangular_prism(lengths), rtol=0.02
    )
    np.testing.assert_allclose(result.tensor - np.diag(np.diag(result.tensor)), 0.0, atol=1e-12)


def test_voxel_needle_of_aspect_20_matches_aharoni() -> None:
    lengths = (1.0, 1.0, 20.0)
    result = voxel_demagnetizing_tensor(
        np.ones((6, 6, 60), dtype=bool), (1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0), padding=2.0
    )

    # The small factor along the needle is resolved to 1e-3 rather than 2 %.
    np.testing.assert_allclose(
        np.diag(result.tensor), demagnetizing_factors_rectangular_prism(lengths), atol=1e-3
    )


def test_voxel_tensor_trace_is_one_without_smoothing() -> None:
    result = voxel_demagnetizing_tensor(_voxel_sphere(12), (1.0, 2.0, 0.5), padding=3.0)

    assert result.trace == pytest.approx(1.0, abs=1e-12)


def test_smoothed_trace_approaches_one_with_grid_refinement() -> None:
    traces = [
        voxel_demagnetizing_tensor(
            np.ones((n, n, n // 4), dtype=bool), 4.0 / n, padding=2.0, smoothing=0.5
        ).trace
        for n in (8, 16, 32)
    ]

    assert traces[0] < traces[1] < traces[2] < 1.0
    assert traces[2] > 0.9


def test_padding_sets_the_box_on_every_axis_from_the_longest_edge() -> None:
    # A plate 64 x 64 x 2 voxels of 20 x 20 x 1: the box spans 2 x 20 along every
    # axis, so 80 voxels through the 1-thick plate, not 2 x 2.
    assert fourier_box_shape((64, 64, 2), (20 / 64, 20 / 64, 0.5), 2.0) == (128, 128, 80)
    assert fourier_box_shape((64, 64, 2), (20 / 64, 20 / 64, 0.5), 3.0)[2] == 120


def test_padding_changes_the_tensor_of_a_plate() -> None:
    mask = np.ones((16, 16, 2), dtype=bool)
    spacing = (0.25, 0.25, 0.25)
    loose = voxel_demagnetizing_tensor(mask, spacing, padding=2.0).tensor
    tight = voxel_demagnetizing_tensor(mask, spacing, padding=4.0).tensor

    assert loose[2, 2] != pytest.approx(tight[2, 2], abs=1e-6)
    assert tight[2, 2] == pytest.approx(
        demagnetizing_factors_rectangular_prism((4.0, 4.0, 0.5))[2], rel=0.01
    )


def test_voxel_tensor_ignores_empty_margins() -> None:
    mask = np.ones((4, 4, 2), dtype=bool)
    padded = np.zeros((9, 7, 5), dtype=bool)
    padded[3:7, 1:5, 2:4] = True

    np.testing.assert_allclose(
        voxel_demagnetizing_tensor(padded, 1.0, padding=2.0).tensor,
        voxel_demagnetizing_tensor(mask, 1.0, padding=2.0).tensor,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"padding": 1.5}, "padding must be at least 2"),
        ({"smoothing": -1.0}, "smoothing"),
        ({"spacing": 0.0}, "spacing"),
    ],
)
def test_voxel_tensor_rejects_unusable_settings(kwargs: dict[str, float], match: str) -> None:
    options = {"spacing": 1.0, "padding": 2.0, **kwargs}
    spacing = options.pop("spacing")
    with pytest.raises(DemagnetizationError, match=match):
        voxel_demagnetizing_tensor(np.ones((2, 2, 2), dtype=bool), spacing, **options)


def test_voxel_tensor_rejects_empty_and_flat_masks() -> None:
    with pytest.raises(DemagnetizationError, match="empty"):
        voxel_demagnetizing_tensor(np.zeros((3, 3, 3), dtype=bool), 1.0)
    with pytest.raises(DemagnetizationError, match="3-D"):
        voxel_demagnetizing_tensor(np.ones((3, 3), dtype=bool), 1.0)


def test_sample_fourier_box_matches_the_calculation_and_checks_its_size() -> None:
    plate = DemagnetizingSample.image_plate(
        np.ones((30, 20), dtype=bool), (6.0, 4.0), 0.5, fourier_grid=24, padding=2.0
    )

    assert sample_fourier_box(plate) == fourier_demagnetizing_result(plate).box_shape
    with pytest.raises(DemagnetizationError, match="million voxels"):
        sample_fourier_box(replace(plate, plate_thickness=1e-5, thickness_voxels=64))
    with pytest.raises(DemagnetizationError, match="closed-form"):
        sample_fourier_box(DemagnetizingSample(shape="sphere"))


def test_voxel_tensor_rejects_a_box_too_large_to_hold() -> None:
    with pytest.raises(DemagnetizationError, match="million voxels"):
        voxel_demagnetizing_tensor(np.ones((64, 64, 1), dtype=bool), (1.0, 1.0, 1e-3))


def test_tensor_field_trace_is_one_inside_and_zero_outside() -> None:
    mask = np.zeros((6, 5, 3), dtype=bool)
    mask[0:6, 0:5, 0:2] = True
    mask[2:4, 1:3, 0:2] = False
    field = demagnetizing_tensor_field(mask, (1.0, 1.0, 0.5), padding=2.0)
    trace = np.einsum("ii...->...", field.values)

    np.testing.assert_allclose(trace, field.mask.astype(float), atol=1e-12)
    assert field.values.shape == (3, 3, *field.mask.shape)


def test_tensor_field_matches_the_local_prism_tensor_at_its_centre() -> None:
    n = 24
    field = demagnetizing_tensor_field(np.ones((n, n, n), dtype=bool), 1.0 / n, padding=3.0)
    centre = field.values[:, :, n // 2, n // 2, n // 2]
    # The voxel grid is offset by half a voxel, so compare with the tensor there.
    offset = 0.5 / n
    local = rectangular_prism_tensor(offset, offset, offset, (0.5, 0.5, 0.5))

    np.testing.assert_allclose(centre, local, atol=0.01)


def test_tensor_field_of_a_mirror_symmetric_shape_averages_without_off_diagonals() -> None:
    # A diagonal line of voxels: symmetric under z -> -z, but not axis-aligned.
    mask = np.eye(4, dtype=bool)[:, :, None]
    field = demagnetizing_tensor_field(mask, 1.0, padding=2.0)
    average = field.values[:, :, field.mask].mean(axis=-1)

    np.testing.assert_allclose(average[[0, 1], [2, 2]], 0.0, atol=1e-14)
    np.testing.assert_allclose(
        average, voxel_demagnetizing_tensor(mask, 1.0, padding=2.0).tensor, atol=1e-12
    )


def test_tensor_field_averages_to_the_magnetometric_tensor() -> None:
    mask = np.ones((8, 6, 2), dtype=bool)
    spacing = (1.0, 1.0, 0.5)
    field = demagnetizing_tensor_field(mask, spacing, padding=2.0)
    average = field.values[:, :, field.mask].mean(axis=-1)

    np.testing.assert_allclose(
        average, voxel_demagnetizing_tensor(mask, spacing, padding=2.0).tensor, atol=1e-12
    )


# Voxel-grid and image-plate samples


def _disc(n: int) -> np.ndarray:
    x = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    return x[:, None] ** 2 + x[None, :] ** 2 <= 1.0


def test_voxel_grid_sample_uses_the_fourier_tensor() -> None:
    mask = np.ones((8, 8, 2), dtype=bool)
    sample = DemagnetizingSample.voxel_grid(mask, (0.5, 0.5, 0.25), padding=2.0)

    assert sample.shape == "voxel_grid"
    assert sample.uses_fourier_method
    np.testing.assert_allclose(
        principal_demagnetizing_tensor(sample),
        voxel_demagnetizing_tensor(mask, (0.5, 0.5, 0.25), padding=2.0).tensor,
    )


def test_voxel_grid_sample_is_rotated_like_any_shape() -> None:
    sample = DemagnetizingSample.voxel_grid(
        np.ones((8, 4, 2), dtype=bool), 1.0, rotation=_rotation_about_z(90.0)
    )
    principal = principal_demagnetizing_tensor(sample)

    np.testing.assert_allclose(demagnetizing_tensor(sample)[0, 0], principal[1, 1], atol=1e-12)


def test_image_plate_of_a_thin_disc_is_a_plate_along_its_normal() -> None:
    sample = DemagnetizingSample.image_plate(
        _disc(64), (5.0, 5.0), 0.05, fourier_grid=48, padding=2.0
    )
    tensor = demagnetizing_tensor(sample)

    assert tensor[2, 2] > 0.95
    assert tensor[0, 0] == pytest.approx(tensor[1, 1], abs=1e-3)
    assert fourier_demagnetizing_result(sample).trace == pytest.approx(1.0, abs=1e-12)


def test_image_plate_of_a_rectangle_matches_aharoni() -> None:
    sample = DemagnetizingSample.image_plate(
        np.ones((30, 20), dtype=bool),
        (6.0, 4.0),
        1.0,
        fourier_grid=48,
        thickness_voxels=4,
        padding=2.0,
    )

    np.testing.assert_allclose(
        np.diag(principal_demagnetizing_tensor(sample)),
        demagnetizing_factors_rectangular_prism((6.0, 4.0, 1.0)),
        rtol=0.02,
    )


def test_image_plate_voxels_follow_the_grid_and_thickness() -> None:
    sample = DemagnetizingSample.image_plate(
        np.ones((40, 20), dtype=bool), (8.0, 4.0), 0.3, fourier_grid=16, thickness_voxels=3
    )
    model = sample.voxel_model()

    assert model.mask.shape == (16, 8, 3)
    np.testing.assert_allclose(model.spacing, (0.5, 0.5, 0.1))
    np.testing.assert_allclose(model.lengths, (8.0, 4.0, 0.3))


def test_image_plate_stores_a_large_outline_at_256_cells() -> None:
    sample = DemagnetizingSample.image_plate(np.ones((1024, 512), dtype=bool), (2.0, 1.0), 0.1)

    assert sample.plate_mask is not None
    assert sample.plate_mask.shape == (256, 128)


def test_fourier_tensor_is_cached_per_shape_and_ignores_orientation() -> None:
    sample = DemagnetizingSample.voxel_grid(np.ones((4, 4, 2), dtype=bool), 1.0)
    rotated = DemagnetizingSample.voxel_grid(
        np.ones((4, 4, 2), dtype=bool), 1.0, rotation=_rotation_about_z(30.0)
    )

    assert fourier_demagnetizing_result(sample) is fourier_demagnetizing_result(rotated)
    with pytest.raises(ValueError, match="read-only"):
        fourier_demagnetizing_result(sample).tensor[0, 0] = 1.0


@pytest.mark.parametrize(
    ("sample", "match"),
    [
        (DemagnetizingSample(shape="voxel_grid"), "no voxels"),
        (DemagnetizingSample(shape="image_plate"), "no outline"),
        (
            DemagnetizingSample(
                shape="image_plate",
                plate_mask=VoxelMask.from_array(np.ones((4, 4), dtype=bool)),
                plate_thickness=0.0,
            ),
            "thickness must be positive",
        ),
        (
            DemagnetizingSample(
                shape="image_plate",
                plate_mask=VoxelMask.from_array(np.ones((4, 4), dtype=bool)),
                thickness_voxels=0,
            ),
            "at least one voxel",
        ),
        (
            DemagnetizingSample(
                shape="image_plate",
                plate_mask=VoxelMask.from_array(np.ones((4, 4), dtype=bool)),
                fourier_grid=4,
            ),
            "at least 8",
        ),
    ],
    ids=["no-voxels", "no-outline", "zero-thickness", "zero-layers", "small-grid"],
)
def test_voxel_samples_report_what_is_missing(sample: DemagnetizingSample, match: str) -> None:
    with pytest.raises(DemagnetizationError, match=match):
        demagnetizing_tensor(sample)


def test_image_plate_rejects_an_empty_outline() -> None:
    with pytest.raises(DemagnetizationError, match="sample in it"):
        DemagnetizingSample.image_plate(np.zeros((4, 4), dtype=bool), (1.0, 1.0), 0.1)


def test_closed_form_shape_has_no_fourier_result() -> None:
    with pytest.raises(DemagnetizationError, match="closed-form"):
        fourier_demagnetizing_result(DemagnetizingSample(shape="sphere"))
    with pytest.raises(DemagnetizationError, match="not made of voxels"):
        DemagnetizingSample(shape="sphere").voxel_model()


def test_voxel_mask_round_trips_and_rejects_non_boolean_values() -> None:
    values = np.random.default_rng(3).random((5, 7, 3)) > 0.5
    mask = VoxelMask.from_array(values)

    np.testing.assert_array_equal(mask.array(), values)
    assert VoxelMask.from_json(mask.to_json()) == mask
    assert mask.count == int(values.sum())
    with pytest.raises(DemagnetizationError, match="true and false"):
        VoxelMask.from_array(np.full((2, 2), 0.5))


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {"shape": [2, 2]},
        {"shape": [2, 2], "bits": "not base64!"},
        {"shape": [2, 2, 2, 2], "bits": "AA=="},
        {"shape": [4, 4], "bits": "AA=="},
    ],
    ids=["missing", "no-bits", "bad-base64", "four-dimensional", "wrong-length"],
)
def test_voxel_mask_from_json_reads_unusable_payloads_as_none(payload: object) -> None:
    assert VoxelMask.from_json(payload) is None


# DemagnetizingSample JSON


def test_demagnetizing_sample_round_trips_json() -> None:
    sample = DemagnetizingSample(
        shape="rectangular_prism",
        semi_axes=(1.0, 2.0, 3.0),
        prism_lengths=(4.0, 5.0, 6.0),
        fourier_grid=32,
        padding=3.0,
    )

    assert DemagnetizingSample.from_json(sample.to_json()) == sample


def test_oriented_sample_round_trips_json() -> None:
    sample = DemagnetizingSample.oriented(
        _HEXAGONAL, (1.0, 0.0, 1.0), (0.0, 1.0, 0.0), shape="ellipsoid", semi_axes=(1.0, 2.0, 3.0)
    )
    payload = sample.to_json()

    assert payload["normal_hkl"] == [1.0, 0.0, 1.0]
    assert payload["in_plane_uvw"] == [0.0, 1.0, 0.0]
    restored = DemagnetizingSample.from_json(payload)
    assert restored == sample
    np.testing.assert_array_equal(demagnetizing_tensor(restored), demagnetizing_tensor(sample))


def test_sample_from_json_without_orientation_is_aligned() -> None:
    sample = DemagnetizingSample.from_json({"shape": "ellipsoid", "semi_axes": [1.0, 2.0, 3.0]})

    assert sample.rotation is None
    assert sample.normal_hkl is None
    assert not sample.is_oriented


@pytest.mark.parametrize(
    "orientation",
    [
        {"rotation": np.diag([1.0, 1.0, -1.0]).tolist()},
        {"rotation": "not-a-matrix"},
        {"normal_hkl": [0.0, 0.0, 1.0]},
        {"normal_hkl": [0.0, 0.0, 1.0], "in_plane_uvw": ["x", 0.0, 0.0]},
        {"normal_hkl": [0.0, 0.0, 0.0], "in_plane_uvw": [1.0, 0.0, 0.0]},
    ],
    ids=["left-handed", "unconvertible", "half-spec", "bad-index", "zero-index"],
)
def test_sample_from_json_reads_unusable_orientation_as_aligned(
    orientation: dict[str, object],
) -> None:
    sample = DemagnetizingSample.from_json(
        {"shape": "ellipsoid", "semi_axes": [1.0, 2.0, 3.0], **orientation}
    )

    assert sample == DemagnetizingSample(shape="ellipsoid", semi_axes=(1.0, 2.0, 3.0))


def test_image_plate_sample_round_trips_json() -> None:
    sample = DemagnetizingSample.image_plate(
        _disc(40),
        (5.0, 4.0),
        0.1,
        fourier_grid=32,
        thickness_voxels=2,
        smoothing=0.5,
        photo=PlatePhoto(path="/data/crystal.png", threshold=97.5, invert=True, fill_holes=False),
    )
    restored = DemagnetizingSample.from_json(sample.to_json())

    assert restored == sample
    np.testing.assert_array_equal(demagnetizing_tensor(restored), demagnetizing_tensor(sample))


def test_voxel_grid_sample_round_trips_json() -> None:
    sample = DemagnetizingSample.voxel_grid(np.ones((3, 4, 2), dtype=bool), (1.0, 0.5, 0.25))

    assert DemagnetizingSample.from_json(sample.to_json()) == sample


def test_sample_from_json_without_voxel_fields_reads_defaults() -> None:
    sample = DemagnetizingSample.from_json({"shape": "image_plate"})

    assert sample.plate_mask is None
    assert sample.photo is None
    assert sample.plate_size == (1.0, 1.0)
    assert sample.thickness_voxels == 1
    assert sample.smoothing == 0.0


def test_sample_from_json_reads_unusable_voxel_fields_as_defaults() -> None:
    sample = DemagnetizingSample.from_json(
        {
            "shape": "image_plate",
            "plate_mask": {"shape": [2, 2], "bits": 7},
            "plate_size": [1.0],
            "plate_thickness": "thin",
            "thickness_voxels": -2,
            "smoothing": float("nan"),
            "photo": {"path": "a.png", "threshold": "dark", "invert": 1},
        }
    )

    assert sample.plate_mask is None
    assert sample.plate_size == (1.0, 1.0)
    assert sample.plate_thickness == 0.1
    assert sample.thickness_voxels == 1
    assert sample.smoothing == 0.0
    assert sample.photo == PlatePhoto(path="a.png", threshold=None, invert=True)


def test_summary_json_names_each_mask_by_its_digest() -> None:
    plate = DemagnetizingSample.image_plate(_disc(40), (5.0, 4.0), 0.1)
    voxels = DemagnetizingSample.voxel_grid(np.ones((3, 4, 2), dtype=bool), 1.0)

    summary = plate.summary_json()
    assert summary["plate_mask"] == {
        "shape": [40, 40],
        "sha256": hashlib.sha256(plate.plate_mask.bits).hexdigest(),
    }
    assert summary["plate_size"] == [5.0, 4.0]
    assert voxels.summary_json()["voxels"]["shape"] == [3, 4, 2]
    assert "bits" not in str(summary)
    sphere = DemagnetizingSample(shape="sphere")
    assert sphere.summary_json() == sphere.to_json()


def test_sample_from_json_non_mapping_returns_default() -> None:
    assert DemagnetizingSample.from_json(["not", "a", "mapping"]) == DemagnetizingSample()


def test_sample_from_json_fills_defaults_for_falsy_values() -> None:
    sample = DemagnetizingSample.from_json(
        {"shape": "  ", "plate_normal_axis": "", "fourier_grid": 0, "padding": 0}
    )
    default = DemagnetizingSample()
    assert sample.shape == "none"
    assert sample.plate_normal_axis == "c"
    assert sample.fourier_grid == default.fourier_grid
    assert sample.padding == default.padding


@pytest.mark.parametrize(
    "semi_axes",
    [None, [1.0, 2.0], "not-a-number"],
    ids=["missing", "two-values", "unconvertible"],
)
def test_sample_from_json_falls_back_to_unit_axes_for_unusable_triplets(semi_axes: object) -> None:
    sample = DemagnetizingSample.from_json({"shape": "ellipsoid", "semi_axes": semi_axes})

    assert sample.semi_axes == (1.0, 1.0, 1.0)


def test_sample_from_json_reads_valid_triplets() -> None:
    sample = DemagnetizingSample.from_json(
        {"shape": "ellipsoid", "semi_axes": [1.5, 2.5, 3.5], "prism_lengths": ["4", "5", "6"]}
    )

    assert sample.semi_axes == (1.5, 2.5, 3.5)
    assert sample.prism_lengths == (4.0, 5.0, 6.0)
