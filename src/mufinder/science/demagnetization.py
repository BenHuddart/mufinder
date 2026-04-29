"""Demagnetizing tensor helpers for magnetic-field calculations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_FACTOR_TOL = 1e-12
_DEFAULT_ELLIPSOID_GRID = 20_000
_DEFAULT_PRISM_GRID = 48
_DEFAULT_PRISM_PADDING = 2.5


class DemagnetizationError(ValueError):
    """Raised when demagnetizing-field inputs are invalid."""


@dataclass(frozen=True, slots=True)
class DemagnetizingSample:
    """Description of a sample shape used to calculate a demagnetizing tensor."""

    shape: str = "none"
    semi_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    prism_lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)
    plate_normal_axis: str = "c"
    fourier_grid: int = _DEFAULT_PRISM_GRID
    padding: float = _DEFAULT_PRISM_PADDING

    def to_json(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "semi_axes": list(self.semi_axes),
            "prism_lengths": list(self.prism_lengths),
            "plate_normal_axis": self.plate_normal_axis,
            "fourier_grid": int(self.fourier_grid),
            "padding": float(self.padding),
        }

    @classmethod
    def from_json(cls, payload: object) -> "DemagnetizingSample":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            shape=str(payload.get("shape", "none")).strip() or "none",
            semi_axes=_float_triplet(payload.get("semi_axes"), default=(1.0, 1.0, 1.0)),
            prism_lengths=_float_triplet(payload.get("prism_lengths"), default=(1.0, 1.0, 1.0)),
            plate_normal_axis=str(payload.get("plate_normal_axis", "c")).strip() or "c",
            fourier_grid=int(payload.get("fourier_grid", _DEFAULT_PRISM_GRID) or _DEFAULT_PRISM_GRID),
            padding=float(payload.get("padding", _DEFAULT_PRISM_PADDING) or _DEFAULT_PRISM_PADDING),
        )


def demagnetizing_tensor(sample: DemagnetizingSample, *, n_grid: int = _DEFAULT_ELLIPSOID_GRID) -> np.ndarray:
    """Return the demagnetizing tensor for a supported sample description."""

    shape = _normalize_shape(sample.shape)
    if shape == "none":
        return np.zeros((3, 3), dtype=float)
    if shape == "sphere":
        return np.eye(3, dtype=float) / 3.0
    if shape == "infinite_plate":
        axis = _axis_index(sample.plate_normal_axis)
        tensor = np.zeros((3, 3), dtype=float)
        tensor[axis, axis] = 1.0
        return tensor
    if shape == "ellipsoid":
        a, b, c = _positive_triplet(sample.semi_axes, "semi_axes")
        return np.diag(demagnetizing_factors_ellipsoid(a, b, c, n_grid=n_grid))
    if shape == "rectangular_prism":
        lengths = _positive_triplet(sample.prism_lengths, "prism_lengths")
        return magnetometric_rectangular_prism_tensor(
            lengths,
            grid_size=int(sample.fourier_grid),
            padding=float(sample.padding),
        )
    raise DemagnetizationError(f"unknown demagnetizing sample shape '{sample.shape}'")


def demagnetizing_factors_ellipsoid(
    a: float,
    b: float,
    c: float,
    *,
    n_grid: int = _DEFAULT_ELLIPSOID_GRID,
) -> tuple[float, float, float]:
    """Return principal demagnetizing factors for an ellipsoid.

    This ports the mag2spectra ellipsoid workflow: exact limits are used for
    spheres and infinite plates, spheroids use closed forms, and fully triaxial
    ellipsoids use the standard Osborn integral.
    """

    a, b, c = _validate_axes(a, b, c)
    if int(n_grid) < 2_000:
        raise DemagnetizationError("n_grid must be at least 2000")

    limit = _asymptotic_limit_factors(a, b, c)
    if limit is not None:
        return limit

    if abs(a - b) < _FACTOR_TOL:
        if a < c:
            return _prolate_factors(a, c)
        if a > c:
            return _oblate_factors(a, c)

    factors = np.array(
        [
            _integral_factor(a, a, b, c, int(n_grid)),
            _integral_factor(b, a, b, c, int(n_grid)),
            _integral_factor(c, a, b, c, int(n_grid)),
        ],
        dtype=float,
    )
    total = float(np.sum(factors))
    if total <= 0.0:
        raise DemagnetizationError("failed to compute valid demagnetizing factors")
    factors = np.clip(factors / total, 0.0, 1.0)
    factors /= float(np.sum(factors))
    return (float(factors[0]), float(factors[1]), float(factors[2]))


def magnetometric_rectangular_prism_tensor(
    lengths: Sequence[float],
    *,
    grid_size: int = _DEFAULT_PRISM_GRID,
    padding: float = _DEFAULT_PRISM_PADDING,
) -> np.ndarray:
    """Estimate the magnetometric demagnetizing tensor for a rectangular prism.

    Rectangular prisms use the Joseph-Schlomann analytic tensor averaged over
    midpoint samples inside the prism. This is more robust than a cubic Fourier
    shape grid for high-aspect-ratio plates and needles.
    """

    lx, ly, lz = _positive_triplet(lengths, "prism_lengths")
    n = int(grid_size)
    if n < 8:
        raise DemagnetizationError("fourier_grid must be at least 8")
    if float(padding) <= 1.0:
        raise DemagnetizationError("padding must be greater than 1")

    a, b, c = 0.5 * lx, 0.5 * ly, 0.5 * lz
    dx = 2.0 * a / float(n)
    dy = 2.0 * b / float(n)
    dz = 2.0 * c / float(n)
    x = np.linspace(-a + 0.5 * dx, a - 0.5 * dx, n)
    y = np.linspace(-b + 0.5 * dy, b - 0.5 * dy, n)
    z = np.linspace(-c + 0.5 * dz, c - 0.5 * dz, n)
    tensor = rectangular_prism_tensor(x[:, None, None], y[None, :, None], z[None, None, :], (a, b, c))
    avg = np.mean(tensor, axis=(2, 3, 4))
    return 0.5 * (avg + avg.T)


def rectangular_prism_tensor(
    x: float | np.ndarray,
    y: float | np.ndarray,
    z: float | np.ndarray,
    half_lengths: Sequence[float],
) -> np.ndarray:
    """Return the local demagnetizing tensor for an axis-aligned rectangular prism."""

    a, b, c = _positive_triplet(half_lengths, "half_lengths")
    xb, yb, zb = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float))

    nzz = _rectangular_diagonal_component(xb, yb, zb, a, b, c)
    nxx = _rectangular_diagonal_component(zb, yb, xb, c, b, a)
    nyy = _rectangular_diagonal_component(xb, zb, yb, a, c, b)
    nxy = _rectangular_offdiag_component(xb, yb, zb, a, b, c)
    nxz = _rectangular_offdiag_component(xb, zb, yb, a, c, b)
    nyz = _rectangular_offdiag_component(zb, yb, xb, c, b, a)

    tensor = np.empty((3, 3) + xb.shape, dtype=float)
    tensor[0, 0] = nxx
    tensor[1, 1] = nyy
    tensor[2, 2] = nzz
    tensor[0, 1] = tensor[1, 0] = nxy
    tensor[0, 2] = tensor[2, 0] = nxz
    tensor[1, 2] = tensor[2, 1] = nyz

    tol = 1e-12
    outside_x = np.abs(xb) > (a + tol)
    outside_y = np.abs(yb) > (b + tol)
    outside_z = np.abs(zb) > (c + tol)
    tensor[0, 0] = tensor[0, 0] - (outside_x & (~outside_y) & (~outside_z)).astype(float)
    tensor[1, 1] = tensor[1, 1] - (outside_y & (~outside_x) & (~outside_z)).astype(float)
    tensor[2, 2] = tensor[2, 2] - (outside_z & (~outside_x) & (~outside_y)).astype(float)
    return tensor


def _rectangular_diagonal_component(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    def _atan_term(tx: np.ndarray, ty: np.ndarray, tz: np.ndarray) -> np.ndarray:
        dx = a - tx
        dy = b - ty
        dz = c - tz
        rr = np.sqrt(dx * dx + dy * dy + dz * dz)
        return np.arctan2(dx * dy, dz * rr)

    return (
        _atan_term(x, y, z)
        + _atan_term(-x, y, z)
        + _atan_term(x, -y, z)
        + _atan_term(x, y, -z)
        + _atan_term(-x, -y, z)
        + _atan_term(x, -y, -z)
        + _atan_term(-x, y, -z)
        + _atan_term(-x, -y, -z)
    ) / (4.0 * np.pi)


def _rectangular_g_function(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    return c - z + np.sqrt((a - x) ** 2 + (b - y) ** 2 + (c - z) ** 2)


def _rectangular_offdiag_component(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> np.ndarray:
    p = (
        _rectangular_g_function(x, y, z, a, b, c)
        * _rectangular_g_function(x, y, z, -a, -b, c)
        * _rectangular_g_function(x, y, z, -a, b, -c)
        * _rectangular_g_function(x, y, z, a, -b, -c)
    )
    q = (
        _rectangular_g_function(x, y, z, -a, b, c)
        * _rectangular_g_function(x, y, z, a, -b, c)
        * _rectangular_g_function(x, y, z, a, b, -c)
        * _rectangular_g_function(x, y, z, -a, -b, -c)
    )
    return -(_safe_log(p) - _safe_log(q)) / (4.0 * np.pi)


def _safe_log(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    mask = values > 0.0
    out[mask] = np.log(values[mask])
    return out


def rectangular_prism_shape(
    grid_shape: tuple[int, int, int],
    half_lengths: Sequence[float],
    spacing: float | Sequence[float],
) -> np.ndarray:
    """Return a centered binary shape grid for an axis-aligned rectangular prism."""

    a, b, c = _positive_triplet(half_lengths, "half_lengths")
    sx, sy, sz = _spacing3(spacing)
    nx, ny, nz = (int(value) for value in grid_shape)
    if nx < 1 or ny < 1 or nz < 1:
        raise DemagnetizationError("grid_shape values must be positive")
    x = np.abs(_coords_1d(nx, sx))[:, None, None]
    y = np.abs(_coords_1d(ny, sy))[None, :, None]
    z = np.abs(_coords_1d(nz, sz))[None, None, :]
    return (x <= a) & (y <= b) & (z <= c)


def fourier_k_grid(
    shape: tuple[int, int, int],
    spacing: float | Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an ``ij``-ordered Cartesian reciprocal-space grid."""

    sx, sy, sz = _spacing3(spacing)
    kx = 2.0 * np.pi * np.fft.fftfreq(int(shape[0]), d=sx)
    ky = 2.0 * np.pi * np.fft.fftfreq(int(shape[1]), d=sy)
    kz = 2.0 * np.pi * np.fft.fftfreq(int(shape[2]), d=sz)
    return np.meshgrid(kx, ky, kz, indexing="ij")


def magnetometric_demagnetizing_tensor(
    shape_amplitude_k: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
    kz: np.ndarray,
    *,
    sample_volume: float,
    domain_lengths: Sequence[float],
) -> np.ndarray:
    """Return the volume-averaged tensor from a shape amplitude."""

    if float(sample_volume) <= 0.0:
        raise DemagnetizationError("sample_volume must be positive")
    lx, ly, lz = _positive_triplet(domain_lengths, "domain_lengths")
    d_k = np.asarray(shape_amplitude_k, dtype=complex)
    if d_k.shape != kx.shape or kx.shape != ky.shape or ky.shape != kz.shape:
        raise DemagnetizationError("shape_amplitude_k and k-grid arrays must have identical shapes")

    k2 = kx * kx + ky * ky + kz * kz
    valid = k2 > 1e-24
    prefactor = 1.0 / (float(sample_volume) * lx * ly * lz)
    d2 = np.abs(d_k) ** 2

    tensor = np.zeros((3, 3), dtype=float)
    k_vec = (kx, ky, kz)
    for i in range(3):
        for j in range(i, 3):
            values = np.zeros_like(k2, dtype=float)
            values[valid] = d2[valid] * (k_vec[i][valid] * k_vec[j][valid]) / k2[valid]
            tensor[i, j] = prefactor * float(np.sum(values))
            tensor[j, i] = tensor[i, j]

    trace = float(np.trace(tensor))
    if trace > 0.0:
        tensor /= trace
    return tensor


def _normalize_shape(shape: str) -> str:
    text = str(shape).strip().lower().replace(" ", "_")
    aliases = {
        "": "none",
        "off": "none",
        "no_demag": "none",
        "plate": "infinite_plate",
        "infiniteplate": "infinite_plate",
        "box": "rectangular_prism",
        "cuboid": "rectangular_prism",
        "rectangular": "rectangular_prism",
    }
    return aliases.get(text, text)


def _axis_index(axis: str) -> int:
    index = {"a": 0, "x": 0, "b": 1, "y": 1, "c": 2, "z": 2}.get(str(axis).strip().lower())
    if index is None:
        raise DemagnetizationError("plate_normal_axis must be one of a, b, c, x, y, or z")
    return index


def _validate_axes(a: float, b: float, c: float) -> tuple[float, float, float]:
    axes = np.array([a, b, c], dtype=float)
    if np.any(np.isnan(axes)) or np.any(axes <= 0.0):
        raise DemagnetizationError("semi-axes must be positive")
    return (float(axes[0]), float(axes[1]), float(axes[2]))


def _asymptotic_limit_factors(a: float, b: float, c: float) -> tuple[float, float, float] | None:
    axes = np.array([a, b, c], dtype=float)
    inf_mask = np.isinf(axes)
    n_inf = int(np.count_nonzero(inf_mask))
    if n_inf == 2:
        finite_index = int(np.where(~inf_mask)[0][0])
        values = np.zeros(3, dtype=float)
        values[finite_index] = 1.0
        return (float(values[0]), float(values[1]), float(values[2]))
    if n_inf > 0:
        raise DemagnetizationError("infinite axes are supported only for infinite plates")
    if np.allclose(axes, axes[0], rtol=0.0, atol=_FACTOR_TOL):
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return None


def _prolate_factors(a: float, c: float) -> tuple[float, float, float]:
    e2 = 1.0 - (a * a) / (c * c)
    e = float(np.sqrt(max(e2, 0.0)))
    if e < 1e-15:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    nz = ((1.0 - e * e) / (2.0 * e**3)) * (np.log((1.0 + e) / (1.0 - e)) - 2.0 * e)
    nxy = 0.5 * (1.0 - nz)
    return (float(nxy), float(nxy), float(nz))


def _oblate_factors(a: float, c: float) -> tuple[float, float, float]:
    eta2 = (a * a) / (c * c) - 1.0
    eta = float(np.sqrt(max(eta2, 0.0)))
    if eta < 1e-15:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    nz = ((1.0 + eta * eta) / (eta**3)) * (eta - np.arctan(eta))
    nxy = 0.5 * (1.0 - nz)
    return (float(nxy), float(nxy), float(nz))


def _integral_factor(ai: float, a: float, b: float, c: float, n_grid: int) -> float:
    t = np.linspace(0.0, 1.0 - 1e-10, int(n_grid), dtype=float)
    one_minus_t = 1.0 - t
    s = t / one_minus_t
    denom = (s + ai * ai) * np.sqrt((s + a * a) * (s + b * b) * (s + c * c))
    integrand = 1.0 / (denom * one_minus_t * one_minus_t)
    return 0.5 * a * b * c * float(np.trapezoid(integrand, t))


def _coords_1d(n: int, spacing: float) -> np.ndarray:
    return (np.arange(int(n), dtype=float) - 0.5 * (int(n) - 1)) * float(spacing)


def _spacing3(value: float | Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        out = (float(arr), float(arr), float(arr))
    elif arr.shape == (3,):
        out = (float(arr[0]), float(arr[1]), float(arr[2]))
    else:
        raise DemagnetizationError("spacing must be a scalar or a three-vector")
    return _positive_triplet(out, "spacing")


def _float_triplet(value: object, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return default
    if arr.shape != (3,):
        return default
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _positive_triplet(values: Iterable[float] | Sequence[float], name: str) -> tuple[float, float, float]:
    try:
        a, b, c = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise DemagnetizationError(f"{name} must contain three numbers") from exc
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        raise DemagnetizationError(f"{name} values must be positive")
    return (a, b, c)
