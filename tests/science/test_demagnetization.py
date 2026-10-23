"""Tests for demagnetizing tensor helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.science import demagnetization as demag
from mufinder.science.demagnetization import (
    DemagnetizationError,
    DemagnetizingSample,
    demagnetizing_factors_ellipsoid,
    demagnetizing_tensor,
    fourier_k_grid,
    magnetometric_demagnetizing_tensor,
    magnetometric_rectangular_prism_tensor,
    rectangular_prism_shape,
    rectangular_prism_tensor,
)


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

    # Off-diagonal entries must be zero (np.diag) and trace normalized.
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


def test_ellipsoid_demagnetizing_factors_are_normalized_and_shape_sensitive() -> None:
    # a == b < c: prolate spheroid; the long axis (z) has the smallest factor.
    factors = demagnetizing_factors_ellipsoid(1.0, 1.0, 4.0)

    assert sum(factors) == pytest.approx(1.0)
    assert factors[2] < factors[0]
    assert factors[0] == pytest.approx(factors[1])


def test_ellipsoid_oblate_branch_a_gt_c() -> None:
    # a == b > c: oblate spheroid; short axis (z) has the largest factor.
    factors = demagnetizing_factors_ellipsoid(5.0, 5.0, 1.0)
    assert sum(factors) == pytest.approx(1.0)
    assert factors[0] == pytest.approx(factors[1])
    assert factors[2] > factors[0]


def test_ellipsoid_triaxial_integral_branch() -> None:
    # Fully triaxial: hits the Osborn integral path, factors all distinct.
    factors = demagnetizing_factors_ellipsoid(1.0, 2.0, 4.0)
    assert sum(factors) == pytest.approx(1.0)
    # Largest factor along the shortest axis (a), smallest along longest (c).
    assert factors[0] > factors[1] > factors[2]
    assert all(0.0 <= f <= 1.0 for f in factors)


def test_ellipsoid_sphere_limit_equal_axes() -> None:
    factors = demagnetizing_factors_ellipsoid(2.0, 2.0, 2.0)
    np.testing.assert_allclose(factors, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))


def test_ellipsoid_two_infinite_axes_gives_plate_limit() -> None:
    # Two infinite axes -> finite axis carries the full factor of 1.
    factors = demagnetizing_factors_ellipsoid(np.inf, np.inf, 1.0)
    np.testing.assert_allclose(factors, (0.0, 0.0, 1.0))


def test_ellipsoid_single_infinite_axis_raises() -> None:
    with pytest.raises(DemagnetizationError, match="infinite axes"):
        demagnetizing_factors_ellipsoid(np.inf, 1.0, 1.0)


def test_ellipsoid_n_grid_below_minimum_raises() -> None:
    with pytest.raises(DemagnetizationError, match="at least 2000"):
        demagnetizing_factors_ellipsoid(1.0, 2.0, 4.0, n_grid=1999)


@pytest.mark.parametrize("axis", [0.0, np.nan], ids=["zero", "nan"])
def test_ellipsoid_invalid_axis_raises(axis: float) -> None:
    with pytest.raises(DemagnetizationError, match="positive"):
        demagnetizing_factors_ellipsoid(axis, 1.0, 1.0)


def test_spheroid_closed_forms_are_isotropic_at_zero_eccentricity() -> None:
    # Private helpers: demagnetizing_factors_ellipsoid takes the sphere limit before a zero eccentricity reaches them.
    np.testing.assert_allclose(demag._prolate_factors(1.0, 1.0), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    np.testing.assert_allclose(demag._oblate_factors(1.0, 1.0), (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))


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
