"""Demagnetizing tensor helpers for magnetic-field calculations.

A sample's shape fixes its principal demagnetizing factors along its own axes
``a``, ``b`` and ``c``. An optional proper rotation places those axes in the
crystal Cartesian frame (the frame of the host lattice vectors), so the tensor
used by a calculation is ``R diag(N) R^T`` and is not diagonal in general.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.special import elliprd

from mufinder.science.lattice import lattice_frac_to_cart, reciprocal_to_cartesian

_ROTATION_TOL = 1e-8
_PARALLEL_TOL = 1e-8
_ORTHOGONAL_TOL_DEG = 1e-6
_DEFAULT_PRISM_GRID = 48
_DEFAULT_PRISM_PADDING = 2.5

Triplet = tuple[float, float, float]
Rotation = tuple[Triplet, Triplet, Triplet]


class DemagnetizationError(ValueError):
    """Raised when demagnetizing-field inputs are invalid."""


@dataclass(frozen=True, slots=True)
class SampleOrientation:
    """A sample rotation resolved from crystallographic directions.

    ``rotation`` holds the sample axes ``a``, ``b`` and ``c`` as columns in the
    crystal Cartesian frame. ``angle_deg`` is the angle between the given
    normal and in-plane directions before they were made orthogonal.
    """

    rotation: Rotation
    angle_deg: float

    @property
    def orthogonal(self) -> bool:
        """Whether the given directions were perpendicular to within 1e-6 degrees."""

        return abs(self.angle_deg - 90.0) <= _ORTHOGONAL_TOL_DEG


@dataclass(frozen=True, slots=True)
class DemagnetizingSample:
    """Description of a sample shape used to calculate a demagnetizing tensor.

    ``rotation`` is ``None`` for a sample whose axes ``a``, ``b`` and ``c`` lie
    along the Cartesian ``x``, ``y`` and ``z``; otherwise it is a proper
    rotation whose columns are the sample axes in the crystal Cartesian frame.
    ``normal_hkl`` and ``in_plane_uvw`` record the crystallographic directions
    the rotation was resolved from: the sample ``c`` axis is normal to the
    ``(h k l)`` plane and the ``a`` axis lies along ``[u v w]``.
    """

    shape: str = "none"
    semi_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    prism_lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)
    plate_normal_axis: str = "c"
    fourier_grid: int = _DEFAULT_PRISM_GRID
    padding: float = _DEFAULT_PRISM_PADDING
    rotation: Rotation | None = None
    normal_hkl: Triplet | None = None
    in_plane_uvw: Triplet | None = None

    def __post_init__(self) -> None:
        if self.rotation is not None:
            object.__setattr__(self, "rotation", _validated_rotation(self.rotation))
        if (self.normal_hkl is None) != (self.in_plane_uvw is None):
            raise DemagnetizationError(
                "an oriented sample needs both a normal (h k l) and an in-plane axis [u v w]"
            )
        if self.normal_hkl is not None and self.in_plane_uvw is not None:
            object.__setattr__(
                self, "normal_hkl", _miller_triplet(self.normal_hkl, "normal (h k l)")
            )
            object.__setattr__(
                self, "in_plane_uvw", _miller_triplet(self.in_plane_uvw, "in-plane axis [u v w]")
            )

    @classmethod
    def oriented(
        cls,
        lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
        normal_hkl: Sequence[float],
        in_plane_uvw: Sequence[float],
        **fields: object,
    ) -> "DemagnetizingSample":
        """Return a sample whose ``c`` axis is normal to ``(h k l)`` and ``a`` axis along ``[u v w]``.

        The remaining ``fields`` are passed to the constructor. Use
        :func:`sample_orientation` to read the angle between the directions.
        """

        sample = cls(normal_hkl=normal_hkl, in_plane_uvw=in_plane_uvw, **fields)  # type: ignore[arg-type]
        return sample.resolve_orientation(lattice_vectors)

    @property
    def is_oriented(self) -> bool:
        """Whether the sample axes are rotated away from the Cartesian axes."""

        return self.rotation is not None or self.normal_hkl is not None

    def resolve_orientation(
        self, lattice_vectors: Sequence[Sequence[float]] | np.ndarray
    ) -> "DemagnetizingSample":
        """Return this sample with its rotation recomputed from its directions in ``lattice_vectors``.

        A sample without crystallographic directions is returned unchanged.
        """

        if self.normal_hkl is None or self.in_plane_uvw is None:
            return self
        orientation = sample_orientation(lattice_vectors, self.normal_hkl, self.in_plane_uvw)
        return replace(self, rotation=orientation.rotation)

    def rotation_matrix(self) -> np.ndarray:
        """Return the rotation as a ``(3, 3)`` array; the identity for an aligned sample."""

        if self.rotation is None:
            if self.normal_hkl is not None:
                raise DemagnetizationError(
                    "the sample orientation has not been resolved against a lattice"
                )
            return np.eye(3, dtype=float)
        return np.asarray(self.rotation, dtype=float)

    def rotate(self, tensor: np.ndarray) -> np.ndarray:
        """Return ``R T R^T`` for a tensor ``T`` given along the sample axes."""

        if not self.is_oriented:
            return tensor
        rotation = self.rotation_matrix()
        rotated = rotation @ np.asarray(tensor, dtype=float) @ rotation.T
        return 0.5 * (rotated + rotated.T)

    def effective_demagnetizing_factor(self, direction: Sequence[float]) -> float:
        """Return ``m^T N m`` for the unit vector ``m`` along a Cartesian ``direction``."""

        try:
            vector = np.asarray(direction, dtype=float)
        except (TypeError, ValueError) as exc:
            raise DemagnetizationError("direction must be three numbers") from exc
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise DemagnetizationError("direction must be a finite three-vector")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise DemagnetizationError("direction must be non-zero")
        unit = vector / norm
        return float(unit @ demagnetizing_tensor(self) @ unit)

    def to_json(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "semi_axes": list(self.semi_axes),
            "prism_lengths": list(self.prism_lengths),
            "plate_normal_axis": self.plate_normal_axis,
            "fourier_grid": int(self.fourier_grid),
            "padding": float(self.padding),
            "rotation": None if self.rotation is None else [list(row) for row in self.rotation],
            "normal_hkl": None if self.normal_hkl is None else list(self.normal_hkl),
            "in_plane_uvw": None if self.in_plane_uvw is None else list(self.in_plane_uvw),
        }

    @classmethod
    def from_json(cls, payload: object) -> "DemagnetizingSample":
        """Read a sample; a missing or unusable orientation reads as aligned."""

        if not isinstance(payload, Mapping):
            return cls()
        sample = cls(
            shape=str(payload.get("shape", "none")).strip() or "none",
            semi_axes=_float_triplet(payload.get("semi_axes"), default=(1.0, 1.0, 1.0)),
            prism_lengths=_float_triplet(payload.get("prism_lengths"), default=(1.0, 1.0, 1.0)),
            plate_normal_axis=str(payload.get("plate_normal_axis", "c")).strip() or "c",
            fourier_grid=int(
                payload.get("fourier_grid", _DEFAULT_PRISM_GRID) or _DEFAULT_PRISM_GRID
            ),
            padding=float(payload.get("padding", _DEFAULT_PRISM_PADDING) or _DEFAULT_PRISM_PADDING),
        )
        try:
            return replace(
                sample,
                rotation=_optional_rotation(payload.get("rotation")),
                normal_hkl=_optional_triplet(payload.get("normal_hkl")),
                in_plane_uvw=_optional_triplet(payload.get("in_plane_uvw")),
            )
        except (DemagnetizationError, TypeError, ValueError):
            return sample


def sample_orientation(
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    normal_hkl: Sequence[float],
    in_plane_uvw: Sequence[float],
) -> SampleOrientation:
    """Return the rotation that puts the sample ``c`` axis along the ``(h k l)`` normal.

    The normal is ``h a* + k b* + l c*`` from the reciprocal lattice and the
    in-plane direction ``u a + v b + w c`` from the lattice rows. The normal is
    kept exactly and the in-plane direction is made perpendicular to it
    (Gram-Schmidt) to give the sample ``a`` axis; ``b = c x a`` completes a
    right-handed set. Directions that are not perpendicular are accepted and
    the angle between them is reported.
    """

    lattice = np.asarray(lattice_vectors, dtype=float).tolist()
    hkl = _miller_triplet(normal_hkl, "normal (h k l)")
    uvw = _miller_triplet(in_plane_uvw, "in-plane axis [u v w]")
    normal = np.asarray(
        reciprocal_to_cartesian(hkl, lattice, error=DemagnetizationError), dtype=float
    )
    in_plane = np.asarray(
        lattice_frac_to_cart(uvw, lattice, error=DemagnetizationError), dtype=float
    )
    normal_unit = normal / np.linalg.norm(normal)
    in_plane_unit = in_plane / np.linalg.norm(in_plane)
    cosine = float(np.clip(normal_unit @ in_plane_unit, -1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    perpendicular = in_plane_unit - cosine * normal_unit
    if float(np.linalg.norm(perpendicular)) <= _PARALLEL_TOL:
        raise DemagnetizationError(
            "the in-plane axis [u v w] is parallel to the normal (h k l); choose a direction "
            "in the plane"
        )
    axis_a = perpendicular / np.linalg.norm(perpendicular)
    axis_c = normal_unit
    axis_b = np.cross(axis_c, axis_a)
    rotation = np.column_stack([axis_a, axis_b, axis_c])
    return SampleOrientation(rotation=_validated_rotation(rotation), angle_deg=angle)


def demagnetizing_tensor(sample: DemagnetizingSample) -> np.ndarray:
    """Return the demagnetizing tensor of a sample in the crystal Cartesian frame.

    This is ``R N R^T`` for the principal tensor ``N`` of the shape and the
    sample rotation ``R``.
    """

    return sample.rotate(principal_demagnetizing_tensor(sample))


def principal_demagnetizing_tensor(sample: DemagnetizingSample) -> np.ndarray:
    """Return the demagnetizing tensor of the sample's shape along its own axes ``a``, ``b``, ``c``."""

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
        return np.diag(demagnetizing_factors_ellipsoid(a, b, c))
    if shape == "rectangular_prism":
        lengths = _positive_triplet(sample.prism_lengths, "prism_lengths")
        return magnetometric_rectangular_prism_tensor(
            lengths,
            grid_size=int(sample.fourier_grid),
            padding=float(sample.padding),
        )
    raise DemagnetizationError(f"unknown demagnetizing sample shape '{sample.shape}'")


def demagnetizing_factors_ellipsoid(a: float, b: float, c: float) -> tuple[float, float, float]:
    """Return the principal demagnetizing factors of an ellipsoid with semi-axes ``a, b, c``.

    The factors are exact (J. A. Osborn, Phys. Rev. **67**, 351 (1945)):
    ``N_a = (abc / 3) R_D(b^2, c^2, a^2)`` and cyclically, with Carlson's
    symmetric elliptic integral ``R_D`` (B. C. Carlson, Numer. Algorithms
    **10**, 13 (1995)). They sum to one without renormalisation. Two infinite
    semi-axes give the exact plate limit, factor 1 along the finite axis.
    """

    a, b, c = _validate_axes(a, b, c)
    limit = _infinite_axis_limit(a, b, c)
    if limit is not None:
        return limit
    scale = max(a, b, c)
    a, b, c = a / scale, b / scale, c / scale
    a2, b2, c2 = a * a, b * b, c * c
    prefactor = a * b * c / 3.0
    return (
        float(prefactor * elliprd(b2, c2, a2)),
        float(prefactor * elliprd(c2, a2, b2)),
        float(prefactor * elliprd(a2, b2, c2)),
    )


def magnetometric_rectangular_prism_tensor(
    lengths: Sequence[float],
    *,
    grid_size: int = _DEFAULT_PRISM_GRID,
    padding: float = _DEFAULT_PRISM_PADDING,
) -> np.ndarray:
    """Estimate the magnetometric demagnetizing tensor for a rectangular prism.

    Rectangular prisms use the Joseph-Schlömann analytic tensor averaged over
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
    tensor = rectangular_prism_tensor(
        x[:, None, None], y[None, :, None], z[None, None, :], (a, b, c)
    )
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
    xb, yb, zb = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)
    )

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


def _infinite_axis_limit(a: float, b: float, c: float) -> tuple[float, float, float] | None:
    axes = np.array([a, b, c], dtype=float)
    inf_mask = np.isinf(axes)
    n_inf = int(np.count_nonzero(inf_mask))
    if n_inf == 0:
        return None
    if n_inf == 2:
        values = np.zeros(3, dtype=float)
        values[int(np.where(~inf_mask)[0][0])] = 1.0
        return (float(values[0]), float(values[1]), float(values[2]))
    raise DemagnetizationError("infinite axes are supported only for infinite plates")


def _validated_rotation(value: object) -> Rotation:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DemagnetizationError("rotation must be a 3x3 matrix of numbers") from exc
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise DemagnetizationError("rotation must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=_ROTATION_TOL):
        raise DemagnetizationError(
            "rotation must be orthonormal: its columns, the sample axes, must be "
            "perpendicular unit vectors"
        )
    if float(np.linalg.det(matrix)) < 0.0:
        raise DemagnetizationError(
            "rotation must be proper (determinant +1): the sample axes form a left-handed set"
        )
    return tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[return-value]


def _optional_rotation(value: object) -> Rotation | None:
    return None if value is None else _validated_rotation(value)


def _optional_triplet(value: object) -> Triplet | None:
    return None if value is None else _finite_triplet(value, "orientation direction")  # type: ignore[arg-type]


def _miller_triplet(values: Iterable[float] | Sequence[float], name: str) -> Triplet:
    triplet = _finite_triplet(values, name)
    if not any(triplet):
        raise DemagnetizationError(f"the {name} must have a non-zero index")
    return triplet


def _finite_triplet(values: Iterable[float] | Sequence[float], name: str) -> Triplet:
    try:
        a, b, c = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise DemagnetizationError(f"{name} must contain three numbers") from exc
    if not all(math.isfinite(value) for value in (a, b, c)):
        raise DemagnetizationError(f"{name} must be finite")
    return (a, b, c)


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


def _float_triplet(
    value: object, *, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if value is None:
        return default
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return default
    if arr.shape != (3,):
        return default
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _positive_triplet(
    values: Iterable[float] | Sequence[float], name: str
) -> tuple[float, float, float]:
    try:
        a, b, c = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise DemagnetizationError(f"{name} must contain three numbers") from exc
    if a <= 0.0 or b <= 0.0 or c <= 0.0:
        raise DemagnetizationError(f"{name} values must be positive")
    return (a, b, c)
