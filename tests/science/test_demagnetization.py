"""Tests for demagnetizing tensor helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.science.demagnetization import (
    DemagnetizationError,
    DemagnetizingSample,
    demagnetizing_factors_ellipsoid,
    demagnetizing_tensor,
    fourier_k_grid,
    magnetometric_demagnetizing_tensor,
    magnetometric_rectangular_prism_tensor,
    principal_demagnetizing_tensor,
    rectangular_prism_shape,
    rectangular_prism_tensor,
    sample_orientation,
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


def test_demagnetizing_tensor_rectangular_prism_branch() -> None:
    sample = DemagnetizingSample(
        shape="rectangular_prism",
        prism_lengths=(1.0, 1.0, 1.0),
        fourier_grid=16,
        padding=2.5,
    )
    tensor = demagnetizing_tensor(sample)

    assert float(np.trace(tensor)) == pytest.approx(1.0)
    np.testing.assert_allclose(tensor, tensor.T, atol=1e-12)


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


# magnetometric_rectangular_prism_tensor


def test_rectangular_prism_tensor_is_symmetric_and_normalized_for_cube() -> None:
    tensor = magnetometric_rectangular_prism_tensor((1.0, 1.0, 1.0), grid_size=24)

    np.testing.assert_allclose(tensor, tensor.T, atol=1e-12)
    assert float(np.trace(tensor)) == pytest.approx(1.0)
    np.testing.assert_allclose(np.diag(tensor), np.full(3, 1.0 / 3.0), atol=0.04)


def test_rectangular_prism_tensor_uses_analytic_shape_for_thin_plate() -> None:
    tensor = magnetometric_rectangular_prism_tensor((10.0, 10.0, 0.25), grid_size=18)

    np.testing.assert_allclose(tensor, tensor.T, atol=1e-12)
    assert float(np.trace(tensor)) == pytest.approx(1.0, abs=1e-3)
    assert tensor[2, 2] > 0.85
    assert tensor[0, 0] < 0.1
    assert tensor[1, 1] < 0.1


def test_prism_grid_too_small_raises() -> None:
    with pytest.raises(DemagnetizationError, match="at least 8"):
        magnetometric_rectangular_prism_tensor((1.0, 1.0, 1.0), grid_size=4)


def test_prism_padding_too_small_raises() -> None:
    with pytest.raises(DemagnetizationError, match="padding"):
        magnetometric_rectangular_prism_tensor((1.0, 1.0, 1.0), grid_size=16, padding=1.0)


def test_prism_nonpositive_length_raises() -> None:
    with pytest.raises(DemagnetizationError, match="positive"):
        magnetometric_rectangular_prism_tensor((0.0, 1.0, 1.0), grid_size=16)


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
    assert float(np.trace(tensor)) == pytest.approx(1.0)
    np.testing.assert_allclose(tensor, tensor.T, atol=1e-12)
    # Cube -> near isotropic diagonal.
    np.testing.assert_allclose(np.diag(tensor), np.full(3, 1.0 / 3.0), atol=0.05)


def test_magnetometric_demagnetizing_tensor_zero_amplitude_trace_branch() -> None:
    # All-zero amplitude -> trace == 0 -> skips the normalization branch.
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
