"""Tests for demagnetizing tensor helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mufinder.science.demagnetization import (
    DemagnetizationError,
    DemagnetizingSample,
    demagnetizing_factors_ellipsoid,
    demagnetizing_tensor,
    magnetometric_rectangular_prism_tensor,
    rectangular_prism_tensor,
)


def test_demagnetizing_tensor_supports_simple_limits() -> None:
    np.testing.assert_allclose(demagnetizing_tensor(DemagnetizingSample(shape="sphere")), np.eye(3) / 3.0)
    np.testing.assert_allclose(
        demagnetizing_tensor(DemagnetizingSample(shape="infinite_plate", plate_normal_axis="b")),
        np.diag([0.0, 1.0, 0.0]),
    )


def test_ellipsoid_demagnetizing_factors_are_normalized_and_shape_sensitive() -> None:
    factors = demagnetizing_factors_ellipsoid(1.0, 1.0, 4.0)

    assert sum(factors) == pytest.approx(1.0)
    assert factors[2] < factors[0]
    assert factors[0] == pytest.approx(factors[1])


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


def test_rectangular_prism_local_tensor_is_symmetric() -> None:
    tensor = rectangular_prism_tensor(0.1, 0.2, 0.3, (1.0, 2.0, 3.0))

    assert tensor.shape == (3, 3)
    np.testing.assert_allclose(tensor, tensor.T, atol=1e-12)


def test_demagnetizing_sample_round_trips_json() -> None:
    sample = DemagnetizingSample(
        shape="rectangular_prism",
        semi_axes=(1.0, 2.0, 3.0),
        prism_lengths=(4.0, 5.0, 6.0),
        fourier_grid=32,
        padding=3.0,
    )

    assert DemagnetizingSample.from_json(sample.to_json()) == sample


def test_demagnetizing_tensor_rejects_unknown_shape() -> None:
    with pytest.raises(DemagnetizationError, match="unknown"):
        demagnetizing_tensor(DemagnetizingSample(shape="unknown_shape"))
