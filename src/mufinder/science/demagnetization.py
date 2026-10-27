"""Demagnetizing tensor helpers for magnetic-field calculations.

A sample's shape fixes its principal demagnetizing tensor along its own axes
``a``, ``b`` and ``c``. An optional proper rotation places those axes in the
crystal Cartesian frame (the frame of the host lattice vectors), so the tensor
used by a calculation is ``R N R^T`` and is not diagonal in general.

Ellipsoids and rectangular prisms have closed forms. Any other shape is given
as voxels, directly or as a plate outlined in a photograph, and its tensor is
computed in Fourier space (M. Beleggia and M. De Graef, J. Magn. Magn. Mater.
**263**, L1 (2003)).
"""

from __future__ import annotations

import base64
import binascii
import decimal
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
import math
import warnings

import numpy as np
from scipy import fft as scipy_fft
from scipy.special import elliprd

from mufinder.science.lattice import lattice_frac_to_cart, reciprocal_to_cartesian
from mufinder.science.sample_image import extrude_plate, fit_mask, resample_mask

_ROTATION_TOL = 1e-8
_PARALLEL_TOL = 1e-8
_ORTHOGONAL_TOL_DEG = 1e-6
_DEFAULT_FOURIER_GRID = 48
_DEFAULT_PADDING = 2.5
_MIN_FOURIER_GRID = 8
_MIN_PADDING = 2.0
_MAX_BOX_VOXELS = 1 << 24
_STORED_MASK_EDGE = 256
_DEFAULT_PLATE_THICKNESS = 0.1

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
class VoxelMask:
    """A boolean array stored packed, so a sample holding it stays hashable and saves compactly."""

    shape: tuple[int, ...]
    bits: bytes

    @classmethod
    def from_array(cls, array: Sequence[object] | np.ndarray) -> "VoxelMask":
        values = np.asarray(array)
        if values.dtype != bool:
            if not np.all(np.isin(values, (0, 1))):
                raise DemagnetizationError("a voxel mask must hold only true and false values")
            values = values.astype(bool)
        if values.ndim not in (2, 3) or min(values.shape) < 1:
            raise DemagnetizationError("a voxel mask must be a non-empty 2-D or 3-D array")
        return cls(
            shape=tuple(int(size) for size in values.shape),
            bits=np.packbits(values, axis=None).tobytes(),
        )

    def array(self) -> np.ndarray:
        count = int(np.prod(self.shape))
        flat = np.unpackbits(np.frombuffer(self.bits, dtype=np.uint8), count=count)
        return flat.astype(bool).reshape(self.shape)

    @property
    def count(self) -> int:
        """Return the number of set voxels."""

        return int(np.count_nonzero(self.array()))

    def to_json(self) -> dict[str, object]:
        return {"shape": list(self.shape), "bits": base64.b64encode(self.bits).decode("ascii")}

    @classmethod
    def from_json(cls, payload: object) -> "VoxelMask | None":
        """Read a mask; an unusable payload reads as ``None``."""

        if not isinstance(payload, Mapping):
            return None
        try:
            shape = tuple(int(size) for size in payload["shape"])  # type: ignore[union-attr]
            bits = base64.b64decode(str(payload["bits"]), validate=True)
        except (KeyError, TypeError, ValueError, binascii.Error):
            return None
        if len(shape) not in (2, 3) or min(shape) < 1:
            return None
        if len(bits) != (int(np.prod(shape)) + 7) // 8:
            return None
        return cls(shape=shape, bits=bits)


@dataclass(frozen=True, slots=True)
class PlatePhoto:
    """Where a plate's outline came from: the photograph and the segmentation settings.

    ``threshold`` is ``None`` for the Otsu threshold. The outline itself is
    stored with the sample, so the tensor does not need the file.
    """

    path: str = ""
    threshold: float | None = None
    invert: bool = False
    fill_holes: bool = True

    def to_json(self) -> dict[str, object]:
        return {
            "path": self.path,
            "threshold": self.threshold,
            "invert": self.invert,
            "fill_holes": self.fill_holes,
        }

    @classmethod
    def from_json(cls, payload: object) -> "PlatePhoto | None":
        if not isinstance(payload, Mapping):
            return None
        threshold = payload.get("threshold")
        try:
            level = None if threshold is None else float(threshold)
        except (TypeError, ValueError):
            level = None
        if level is not None and not math.isfinite(level):
            level = None
        return cls(
            path=str(payload.get("path", "") or ""),
            threshold=level,
            invert=bool(payload.get("invert", False)),
            fill_holes=bool(payload.get("fill_holes", True)),
        )


@dataclass(frozen=True, slots=True)
class VoxelModel:
    """A sample's voxels, indexed ``[a, b, c]``, and the voxel edge along each axis."""

    mask: np.ndarray
    spacing: Triplet

    @property
    def lengths(self) -> Triplet:
        a, b, c = (float(n) * s for n, s in zip(self.mask.shape, self.spacing, strict=True))
        return (a, b, c)


@dataclass(frozen=True, slots=True)
class FourierDemagnetizingResult:
    """The magnetometric tensor of a voxel shape and how it was computed.

    ``trace`` is the trace of ``tensor``. It is 1 to rounding for an unsmoothed
    shape, and the Gaussian filter lowers it by the share of the shape's
    spectrum it removed. ``box_shape`` is the padded FFT box in voxels.
    """

    tensor: np.ndarray
    trace: float
    box_shape: tuple[int, int, int]
    voxel_count: int


@dataclass(frozen=True, slots=True)
class DemagnetizingTensorField:
    """The local demagnetizing tensor ``N_ij(r)`` over the padded FFT box.

    ``values`` has shape ``(3, 3, nx, ny, nz)``; ``mask`` marks the sample in
    the box, which starts at the box origin; ``spacing`` is the voxel edge.
    The field from a uniform magnetisation ``M`` is ``-N(r) M``.
    """

    values: np.ndarray
    mask: np.ndarray
    spacing: Triplet


@dataclass(frozen=True, slots=True)
class DemagnetizingSample:
    """Description of a sample shape used to calculate a demagnetizing tensor.

    ``rotation`` is ``None`` for a sample whose axes ``a``, ``b`` and ``c`` lie
    along the Cartesian ``x``, ``y`` and ``z``; otherwise it is a proper
    rotation whose columns are the sample axes in the crystal Cartesian frame.
    ``normal_hkl`` and ``in_plane_uvw`` record the crystallographic directions
    the rotation was resolved from: the sample ``c`` axis is normal to the
    ``(h k l)`` plane and the ``a`` axis lies along ``[u v w]``.

    The ``voxel_grid`` shape is ``voxels`` (a 3-D mask indexed ``[a, b, c]``)
    with ``voxel_spacing``. The ``image_plate`` shape is ``plate_mask`` (an
    in-plane outline indexed ``[a, b]``) stretched over ``plate_size`` and
    extruded through ``plate_thickness`` in ``thickness_voxels`` layers;
    ``fourier_grid`` voxels span its longer in-plane edge and ``photo``
    records where the outline came from. Both use the Fourier method with the
    FFT box ``padding`` times the longest sample edge along every axis and a
    Gaussian filter of ``smoothing`` voxels (0 for none).
    """

    shape: str = "none"
    semi_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    prism_lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)
    plate_normal_axis: str = "c"
    fourier_grid: int = _DEFAULT_FOURIER_GRID
    padding: float = _DEFAULT_PADDING
    rotation: Rotation | None = None
    normal_hkl: Triplet | None = None
    in_plane_uvw: Triplet | None = None
    smoothing: float = 0.0
    voxels: VoxelMask | None = None
    voxel_spacing: Triplet = (1.0, 1.0, 1.0)
    plate_mask: VoxelMask | None = None
    plate_size: tuple[float, float] = (1.0, 1.0)
    plate_thickness: float = _DEFAULT_PLATE_THICKNESS
    thickness_voxels: int = 1
    photo: PlatePhoto | None = field(default=None)

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

    @classmethod
    def voxel_grid(
        cls,
        mask: Sequence[object] | np.ndarray,
        spacing: float | Sequence[float] = 1.0,
        **fields: object,
    ) -> "DemagnetizingSample":
        """Return a ``voxel_grid`` sample from a 3-D boolean array indexed ``[a, b, c]``."""

        values = np.asarray(mask)
        if values.ndim != 3:
            raise DemagnetizationError("a voxel grid must be a 3-D array")
        return cls(
            shape="voxel_grid",
            voxels=VoxelMask.from_array(values),
            voxel_spacing=_spacing3(spacing),
            **fields,  # type: ignore[arg-type]
        )

    @classmethod
    def image_plate(
        cls,
        mask: Sequence[object] | np.ndarray,
        size: Sequence[float],
        thickness: float,
        **fields: object,
    ) -> "DemagnetizingSample":
        """Return an ``image_plate`` sample from an in-plane outline indexed ``[a, b]``.

        The outline is stored at up to 256 cells along its longer side; ``size``
        is the physical width (along ``a``) and height (along ``b``) of its
        bounding box.
        """

        width, height = _positive_pair(size, "plate size")
        return cls(
            shape="image_plate",
            plate_mask=plate_outline_mask(mask),
            plate_size=(width, height),
            plate_thickness=float(thickness),
            **fields,  # type: ignore[arg-type]
        )

    @property
    def uses_fourier_method(self) -> bool:
        """Whether the tensor comes from the Fourier method rather than a closed form."""

        return _normalize_shape(self.shape) in _FOURIER_SHAPES

    def voxel_model(self) -> VoxelModel:
        """Return the voxels the Fourier method uses for a ``voxel_grid`` or ``image_plate`` sample."""

        shape = _normalize_shape(self.shape)
        if shape == "voxel_grid":
            if self.voxels is None or len(self.voxels.shape) != 3:
                raise DemagnetizationError("the voxel-grid sample has no voxels")
            return VoxelModel(mask=self.voxels.array(), spacing=_spacing3(self.voxel_spacing))
        if shape == "image_plate":
            in_plane, layers, spacing = self._plate_layout()
            return VoxelModel(mask=extrude_plate(in_plane, layers), spacing=spacing)
        raise DemagnetizationError(f"the '{self.shape}' shape is not made of voxels")

    def _plate_layout(self) -> tuple[np.ndarray, int, Triplet]:
        """Return a plate's in-plane voxels, its layers and the voxel edges, without extruding."""

        if self.plate_mask is None or len(self.plate_mask.shape) != 2:
            raise DemagnetizationError("the plate has no outline; choose a photograph")
        width, height = _positive_pair(self.plate_size, "plate size")
        thickness = float(self.plate_thickness)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise DemagnetizationError("the plate thickness must be positive")
        grid = _fourier_grid(self.fourier_grid)
        layers = int(self.thickness_voxels)
        if layers < 1:
            raise DemagnetizationError("the plate needs at least one voxel through its thickness")
        longer = max(width, height)
        counts = (
            max(1, int(round(grid * width / longer))),
            max(1, int(round(grid * height / longer))),
        )
        in_plane = resample_mask(self.plate_mask.array(), counts)
        if not np.any(in_plane):
            raise DemagnetizationError(
                "the outline vanishes at this grid; increase the Fourier grid"
            )
        return in_plane, layers, (width / counts[0], height / counts[1], thickness / layers)

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
            "smoothing": float(self.smoothing),
            "voxels": None if self.voxels is None else self.voxels.to_json(),
            "voxel_spacing": list(self.voxel_spacing),
            "plate_mask": None if self.plate_mask is None else self.plate_mask.to_json(),
            "plate_size": list(self.plate_size),
            "plate_thickness": float(self.plate_thickness),
            "thickness_voxels": int(self.thickness_voxels),
            "photo": None if self.photo is None else self.photo.to_json(),
        }

    def summary_json(self) -> dict[str, object]:
        """Return :meth:`to_json` with each voxel mask replaced by its shape and SHA-256 digest.

        Results record the sample they were calculated with this way, so they
        name the outline without carrying a copy of it.
        """

        payload = self.to_json()
        for key in ("voxels", "plate_mask"):
            mask = getattr(self, key)
            if mask is not None:
                payload[key] = {
                    "shape": list(mask.shape),
                    "sha256": hashlib.sha256(mask.bits).hexdigest(),
                }
        return payload

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
            fourier_grid=_json_int(payload.get("fourier_grid"), _DEFAULT_FOURIER_GRID),
            padding=_json_float(payload.get("padding"), _DEFAULT_PADDING),
            smoothing=max(0.0, _json_float(payload.get("smoothing"), 0.0, allow_zero=True)),
            voxels=VoxelMask.from_json(payload.get("voxels")),
            voxel_spacing=_float_triplet(payload.get("voxel_spacing"), default=(1.0, 1.0, 1.0)),
            plate_mask=VoxelMask.from_json(payload.get("plate_mask")),
            plate_size=_float_pair(payload.get("plate_size"), default=(1.0, 1.0)),
            plate_thickness=_json_float(payload.get("plate_thickness"), _DEFAULT_PLATE_THICKNESS),
            thickness_voxels=_json_int(payload.get("thickness_voxels"), 1),
            photo=PlatePhoto.from_json(payload.get("photo")),
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
        return np.diag(demagnetizing_factors_rectangular_prism(sample.prism_lengths))
    if shape in _FOURIER_SHAPES:
        return fourier_demagnetizing_result(sample).tensor.copy()
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


def demagnetizing_factors_rectangular_prism(lengths: Sequence[float]) -> Triplet:
    """Return the magnetometric demagnetizing factors of a rectangular prism with edges ``lengths``.

    The factors are exact (A. Aharoni, J. Appl. Phys. **83**, 3432 (1998)) and
    sum to one. The off-diagonal elements of an axis-aligned prism vanish, so
    these are its whole tensor along the prism edges.
    """

    lx, ly, lz = _positive_triplet(lengths, "prism_lengths")
    if not all(math.isfinite(value) for value in (lx, ly, lz)):
        raise DemagnetizationError("prism_lengths values must be finite")
    scale = max(lx, ly, lz)
    a, b, c = 0.5 * lx / scale, 0.5 * ly / scale, 0.5 * lz / scale
    return (
        _aharoni_factor(b, c, a),
        _aharoni_factor(c, a, b),
        _aharoni_factor(a, b, c),
    )


def _aharoni_factor(a: float, b: float, c: float) -> float:
    """Return Aharoni's factor along the edge of half-length ``c`` (Eq. 1 of the paper).

    The polynomial and logarithmic terms each grow like the square of the
    aspect ratio and cancel to a factor between 0 and 1, so in floating point
    a 1e5:1 needle loses six digits and a 1e8:1 one all of them. They are
    evaluated in decimal arithmetic with enough digits for the aspect ratio;
    the arctangent term is bounded and stays in floating point. The
    logarithms of the paper are written as inverse hyperbolic sines.
    """

    aspect = max(a, b, c) / min(a, b, c)
    digits = 40 + 2 * math.ceil(math.log10(aspect))
    with decimal.localcontext() as context:
        context.prec = digits
        x, y, z = decimal.Decimal(a), decimal.Decimal(b), decimal.Decimal(c)
        r = (x * x + y * y + z * z).sqrt()
        r_xy = (x * x + y * y).sqrt()
        r_yz = (y * y + z * z).sqrt()
        r_xz = (x * x + z * z).sqrt()
        xyz = x * y * z
        three = decimal.Decimal(3)
        value = (
            -(y * y - z * z) / (y * z) * _decimal_asinh(x / r_yz)
            - (x * x - z * z) / (x * z) * _decimal_asinh(y / r_xz)
            + y / z * _decimal_asinh(x / y)
            + x / z * _decimal_asinh(y / x)
            - z / x * _decimal_asinh(y / z)
            - z / y * _decimal_asinh(x / z)
            + (x**3 + y**3 - 2 * z**3) / (three * xyz)
            + (x * x + y * y - 2 * z * z) / (three * xyz) * r
            + z / (x * y) * (r_xz + r_yz)
            - (r_xy**3 + r_yz**3 + r_xz**3) / (three * xyz)
        )
    bounded = 2.0 * math.atan(a * b / (c * math.sqrt(a * a + b * b + c * c)))
    return float(value) / math.pi + bounded / math.pi


def _decimal_asinh(x: decimal.Decimal) -> decimal.Decimal:
    """Return ``asinh(x)`` for ``x >= 0`` at the context's precision."""

    return (x + (x * x + 1).sqrt()).ln()


def magnetometric_rectangular_prism_tensor(
    lengths: Sequence[float],
    *,
    grid_size: int | None = None,
    padding: float | None = None,
) -> np.ndarray:
    """Return the magnetometric demagnetizing tensor of an axis-aligned rectangular prism.

    This is ``diag`` of :func:`demagnetizing_factors_rectangular_prism`. The
    tensor is exact, so ``grid_size`` and ``padding`` are ignored; passing
    either is deprecated.
    """

    if grid_size is not None or padding is not None:
        warnings.warn(
            "grid_size and padding are ignored: the rectangular-prism tensor is exact",
            DeprecationWarning,
            stacklevel=2,
        )
    return np.diag(demagnetizing_factors_rectangular_prism(lengths))


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
    """Return the volume-averaged tensor from a shape amplitude on a periodic box.

    This is Eq. 11 of Beleggia and De Graef as a sum over the box's reciprocal
    lattice, ``<N_ij> = sum_k |D(k)|^2 k_i k_j / k^2 / (V V_box)``, with the
    ``k = 0`` term ``|D(0)|^2 / 3`` on the diagonal. The tensor is not
    normalised, so its trace measures how much of the shape's spectrum the
    grid holds. :func:`voxel_demagnetizing_tensor` is the same sum for a voxel
    shape, computed without full-size ``k`` arrays.
    """

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
    tensor += np.eye(3) * prefactor * float(np.sum(d2[~valid])) / 3.0
    return tensor


def voxel_demagnetizing_tensor(
    mask: Sequence[object] | np.ndarray,
    spacing: float | Sequence[float],
    *,
    padding: float = _DEFAULT_PADDING,
    smoothing: float = 0.0,
) -> FourierDemagnetizingResult:
    """Return the magnetometric demagnetizing tensor of a voxel shape by the Fourier method.

    ``mask`` is a 3-D boolean array of voxels with edges ``spacing`` (a scalar
    or one per axis). Its FFT is the shape amplitude ``D(k)``; the tensor is
    ``<N_ij> = sum_k |D(k)|^2 k_i k_j / k^2`` normalised by the sample and box
    volumes, with ``k = 0`` shared equally between the diagonal elements so
    the trace is the share of ``sum |D(k)|^2`` kept, 1 without smoothing.

    The FFT box is periodic. It is ``padding`` (at least 2) times the longest
    sample edge along every axis, not along each axis separately: a thin plate
    padded only through its thickness would sit in a stack of its images and
    look like bulk. ``smoothing`` is the width, in voxels, of a Gaussian filter
    ``exp(-sigma^2 k^2 / 2)`` on ``D(k)`` that damps Gibbs ringing.
    """

    voxels, step, box = _fourier_box(mask, spacing, padding, smoothing)
    spectrum = scipy_fft.rfftn(_embedded(voxels, box), workers=-1)
    power = spectrum.real**2 + spectrum.imag**2
    del spectrum
    k_axes = _k_axes(box, step)
    if smoothing > 0.0:
        power *= _gaussian_filter(k_axes, step, smoothing, squared=True)
    # rfftn keeps the non-negative kz half; every column but kz = 0 and the
    # even-length Nyquist column stands for itself and its mirror image.
    weights = np.full(k_axes[2].shape, 2.0)
    weights[0] = 1.0
    if box[2] % 2 == 0:
        weights[-1] = 1.0
    power *= weights
    k2 = (
        k_axes[0][:, None, None] ** 2
        + k_axes[1][None, :, None] ** 2
        + k_axes[2][None, None, :] ** 2
    )
    origin = float(power[0, 0, 0])
    k2[0, 0, 0] = 1.0
    power /= k2
    power[0, 0, 0] = 0.0
    del k2
    kx, ky, kz = k_axes
    # The Nyquist plane of an even-length axis has no partner at -k, so it
    # would give a mirror-symmetric shape spurious off-diagonal elements;
    # leave it out of them.
    px, py, pz = (_without_nyquist(k, n) for k, n in zip(k_axes, box, strict=True))
    sum_yz = power.sum(axis=(1, 2))
    sum_xz = power.sum(axis=(0, 2))
    sum_xy = power.sum(axis=(0, 1))
    plane_xy = power.sum(axis=2)
    plane_xz = power.sum(axis=1)
    plane_yz = power.sum(axis=0)
    tensor = np.empty((3, 3), dtype=float)
    tensor[0, 0] = float(kx**2 @ sum_yz)
    tensor[1, 1] = float(ky**2 @ sum_xz)
    tensor[2, 2] = float(kz**2 @ sum_xy)
    tensor[0, 1] = tensor[1, 0] = float(px @ plane_xy @ py)
    tensor[0, 2] = tensor[2, 0] = float(px @ plane_xz @ pz)
    tensor[1, 2] = tensor[2, 1] = float(py @ plane_yz @ pz)
    tensor += np.eye(3) * origin / 3.0
    count = int(np.count_nonzero(voxels))
    tensor /= float(count) * float(np.prod(box))
    return FourierDemagnetizingResult(
        tensor=tensor,
        trace=float(np.trace(tensor)),
        box_shape=box,
        voxel_count=count,
    )


def demagnetizing_tensor_field(
    mask: Sequence[object] | np.ndarray,
    spacing: float | Sequence[float],
    *,
    padding: float = _DEFAULT_PADDING,
    smoothing: float = 0.0,
) -> DemagnetizingTensorField:
    """Return the local tensor ``N_ij(r)`` of a voxel shape over the padded FFT box.

    It is the inverse FFT of ``N_ij(k) = D(k) k_i k_j / k^2`` (Eq. 6 of
    Beleggia and De Graef), with ``D(0) / 3`` on the diagonal at ``k = 0``, so
    without smoothing its trace is the shape itself: 1 inside and 0 outside.
    Its average over the sample is :func:`voxel_demagnetizing_tensor`. The
    arguments are those of that function.
    """

    voxels, step, box = _fourier_box(mask, spacing, padding, smoothing)
    embedded = _embedded(voxels, box)
    spectrum = scipy_fft.rfftn(embedded, workers=-1)
    k_axes = _k_axes(box, step)
    if smoothing > 0.0:
        spectrum *= _gaussian_filter(k_axes, step, smoothing, squared=False)
    grids = np.meshgrid(*k_axes, indexing="ij", sparse=True)
    # As in voxel_demagnetizing_tensor, the unpaired Nyquist planes are left
    # out of the off-diagonal elements.
    paired = np.meshgrid(
        *(_without_nyquist(k, n) for k, n in zip(k_axes, box, strict=True)),
        indexing="ij",
        sparse=True,
    )
    k2 = grids[0] ** 2 + grids[1] ** 2 + grids[2] ** 2
    k2[0, 0, 0] = 1.0
    values = np.empty((3, 3, *box), dtype=float)
    for i in range(3):
        for j in range(i, 3):
            factors = grids if i == j else paired
            component = spectrum * (factors[i] * factors[j] / k2)
            if i == j:
                component[0, 0, 0] = spectrum[0, 0, 0] / 3.0
            values[i, j] = scipy_fft.irfftn(component, s=box, workers=-1)
            values[j, i] = values[i, j]
    return DemagnetizingTensorField(values=values, mask=embedded.astype(bool), spacing=step)


def fourier_demagnetizing_result(sample: DemagnetizingSample) -> FourierDemagnetizingResult:
    """Return the Fourier-method result for a ``voxel_grid`` or ``image_plate`` sample.

    The result is along the sample axes and is cached, since a calculation and
    the dialog ask for the same shape repeatedly.
    """

    if not sample.uses_fourier_method:
        raise DemagnetizationError(f"the '{sample.shape}' shape has a closed-form tensor")
    shape_only = replace(sample, rotation=None, normal_hkl=None, in_plane_uvw=None, photo=None)
    return _cached_fourier_result(shape_only)


@lru_cache(maxsize=8)
def _cached_fourier_result(sample: DemagnetizingSample) -> FourierDemagnetizingResult:
    model = sample.voxel_model()
    result = voxel_demagnetizing_tensor(
        model.mask,
        model.spacing,
        padding=float(sample.padding),
        smoothing=float(sample.smoothing),
    )
    result.tensor.flags.writeable = False
    return result


def fourier_box_shape(
    mask_shape: Sequence[int], spacing: float | Sequence[float], padding: float
) -> tuple[int, int, int]:
    """Return the FFT box, in voxels, for a sample of ``mask_shape`` voxels.

    Every axis spans at least ``padding`` times the longest sample edge,
    rounded up to a size the FFT handles quickly.
    """

    step = _spacing3(spacing)
    pad = float(padding)
    if not math.isfinite(pad) or pad < _MIN_PADDING:
        raise DemagnetizationError(f"padding must be at least {_MIN_PADDING:g}")
    counts = tuple(int(n) for n in mask_shape)
    if len(counts) != 3 or min(counts) < 1:
        raise DemagnetizationError("the voxel shape must be a non-empty 3-D array")
    edge = pad * max(n * s for n, s in zip(counts, step, strict=True))
    box = tuple(
        int(scipy_fft.next_fast_len(max(2 * n, int(math.ceil(edge / s - 1e-9)))))
        for n, s in zip(counts, step, strict=True)
    )
    return box  # type: ignore[return-value]


def plate_outline_mask(mask: Sequence[object] | np.ndarray) -> VoxelMask:
    """Return an in-plane outline as an ``image_plate`` stores it: at most 256 cells along its longer side."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not np.any(values):
        raise DemagnetizationError("a plate outline must be a 2-D mask with the sample in it")
    return VoxelMask.from_array(fit_mask(values, _STORED_MASK_EDGE))


def sample_fourier_box(sample: DemagnetizingSample) -> tuple[int, int, int]:
    """Return the FFT box, in voxels, the Fourier method would use for ``sample``.

    It applies the same checks as the calculation, without the FFT, so a
    dialog can report a box too large to hold before anything is computed:
    :class:`DemagnetizationError` names the settings to lower.
    """

    if not sample.uses_fourier_method:
        raise DemagnetizationError(f"the '{sample.shape}' shape has a closed-form tensor")
    _smoothing_width(sample.smoothing)
    if _normalize_shape(sample.shape) == "image_plate":
        # The layers are identical, so the box follows from the in-plane voxels.
        in_plane, layers, spacing = sample._plate_layout()
        rows = np.flatnonzero(np.any(in_plane, axis=1))
        columns = np.flatnonzero(np.any(in_plane, axis=0))
        extent = (int(rows[-1] - rows[0] + 1), int(columns[-1] - columns[0] + 1), layers)
        return _checked_box(extent, spacing, float(sample.padding))
    model = sample.voxel_model()
    _voxels, _step, box = _fourier_box(
        model.mask, model.spacing, float(sample.padding), float(sample.smoothing)
    )
    return box


def _fourier_box(
    mask: Sequence[object] | np.ndarray,
    spacing: float | Sequence[float],
    padding: float,
    smoothing: float,
) -> tuple[np.ndarray, Triplet, tuple[int, int, int]]:
    voxels = np.asarray(mask)
    if voxels.dtype != bool:
        voxels = voxels.astype(bool)
    if voxels.ndim != 3:
        raise DemagnetizationError("the voxel shape must be a 3-D array")
    if not np.any(voxels):
        raise DemagnetizationError("the voxel shape is empty")
    _smoothing_width(smoothing)
    hits = [np.flatnonzero(np.any(voxels, axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))]
    voxels = voxels[
        hits[0][0] : hits[0][-1] + 1,
        hits[1][0] : hits[1][-1] + 1,
        hits[2][0] : hits[2][-1] + 1,
    ]
    step = _spacing3(spacing)
    return voxels, step, _checked_box(voxels.shape, step, padding)


def _smoothing_width(smoothing: float) -> float:
    sigma = float(smoothing)
    if not math.isfinite(sigma) or sigma < 0.0:
        raise DemagnetizationError("smoothing must be zero or positive")
    return sigma


def _checked_box(
    extent: Sequence[int], spacing: float | Sequence[float], padding: float
) -> tuple[int, int, int]:
    """Return the FFT box for a sample of ``extent`` voxels, refusing one too large to hold."""

    box = fourier_box_shape(extent, spacing, padding)
    total = int(np.prod(box))
    if total > _MAX_BOX_VOXELS:
        raise DemagnetizationError(
            f"the FFT box would hold {total / 1e6:.1f} million voxels (limit "
            f"{_MAX_BOX_VOXELS / 1e6:.1f} million); lower the Fourier grid, the voxels "
            "through the thickness or the padding"
        )
    return box


def _embedded(voxels: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    embedded = np.zeros(box, dtype=float)
    embedded[: voxels.shape[0], : voxels.shape[1], : voxels.shape[2]] = voxels
    return embedded


def _k_axes(box: tuple[int, int, int], step: Triplet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        2.0 * np.pi * np.fft.fftfreq(box[0], d=step[0]),
        2.0 * np.pi * np.fft.fftfreq(box[1], d=step[1]),
        2.0 * np.pi * np.fft.rfftfreq(box[2], d=step[2]),
    )


def _without_nyquist(k: np.ndarray, length: int) -> np.ndarray:
    if length % 2:
        return k
    trimmed = k.copy()
    trimmed[length // 2 if len(k) == length else -1] = 0.0
    return trimmed


def _gaussian_filter(
    k_axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    step: Triplet,
    smoothing: float,
    *,
    squared: bool,
) -> np.ndarray:
    """Return ``exp(-sum (sigma_i k_i)^2 / 2)``, or its square, with ``sigma_i`` in voxels along axis ``i``."""

    factor = 1.0 if squared else 0.5
    parts = [np.exp(-factor * (smoothing * s * k) ** 2) for k, s in zip(k_axes, step, strict=True)]
    return parts[0][:, None, None] * parts[1][None, :, None] * parts[2][None, None, :]


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
        "voxels": "voxel_grid",
        "photo": "image_plate",
        "plate_from_photo": "image_plate",
    }
    return aliases.get(text, text)


_FOURIER_SHAPES = frozenset({"voxel_grid", "image_plate"})


def _fourier_grid(value: object) -> int:
    grid = int(value)  # type: ignore[call-overload]
    if grid < _MIN_FOURIER_GRID:
        raise DemagnetizationError(f"fourier_grid must be at least {_MIN_FOURIER_GRID}")
    return grid


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


def _float_pair(value: object, *, default: tuple[float, float]) -> tuple[float, float]:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return default
    if arr.shape != (2,):
        return default
    return (float(arr[0]), float(arr[1]))


def _json_float(value: object, default: float, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number < 0.0 or (number == 0.0 and not allow_zero):
        return default
    return number


def _json_int(value: object, default: int) -> int:
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError, OverflowError):
        return default
    return number if number > 0 else default


def _positive_pair(values: Iterable[float] | Sequence[float], name: str) -> tuple[float, float]:
    try:
        a, b = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise DemagnetizationError(f"the {name} must contain two numbers") from exc
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0.0 or b <= 0.0:
        raise DemagnetizationError(f"the {name} values must be positive")
    return (a, b)


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
