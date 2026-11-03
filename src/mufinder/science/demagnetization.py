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
#: Edge of the FFT box along every axis, in units of the longest sample edge, for a voxel grid.
FOURIER_PADDING = 2.5
#: The same for a plate outlined in a photograph, whose sum holds only the box's 2-D plane.
PLATE_FOURIER_PADDING = 4.0
#: Voxels along the longest edge, thickness included, of a plate outlined in a photograph.
PLATE_FOURIER_GRID = 256
# How far a factor can be off for an outline this many voxels across; 8 and more go unwarned.
_NARROW_OUTLINE_ERROR = {1: 0.06, 2: 0.04, 3: 0.03, 4: 0.02, 5: 0.02, 6: 0.02, 7: 0.01}
#: How far the trace of a computed tensor may differ from 1 before it is refused.
_TRACE_TOLERANCE = 0.01
_MIN_PADDING = 2.0
_MAX_BOX_VOXELS = 1 << 28
_MAX_FFT_LENGTH = 1 << 40
_STORED_MASK_EDGE = 256
_DEFAULT_PLATE_THICKNESS = 0.1
_THINNEST_PLATE_RATIO = 10_000
# A plate's sum takes a time that follows its box, not its memory. The limit is half as much
# again as the box of a square plate at _THINNEST_PLATE_RATIO, so that rounding an in-plane edge
# to whole voxels does not refuse a plate at that ratio. It is what refuses a long needle.
_MAX_PLATE_BOX_VOXELS = 3 * (
    int(PLATE_FOURIER_PADDING * PLATE_FOURIER_GRID) ** 2
    * int(PLATE_FOURIER_PADDING * _THINNEST_PLATE_RATIO)
    // 2
)
# An in-plane edge this many thicknesses long makes an infinite plate a fair description.
_INFINITE_PLATE_MIN_RATIO = 1_000

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

    ``trace`` is the trace of ``tensor``, 1 to rounding by Parseval's theorem.
    ``box_shape`` is the padded FFT box in voxels.
    """

    tensor: np.ndarray
    trace: float
    box_shape: tuple[int, int, int]
    voxel_count: int


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
    extruded through ``plate_thickness``; ``photo`` records where the outline
    came from. ``PLATE_FOURIER_GRID`` voxels span its longest edge, thickness
    included, and the other edges follow, so its voxels are close to cubes.
    Both use the Fourier method with the FFT box a multiple of the longest sample
    edge along every axis: ``FOURIER_PADDING`` for a voxel grid and
    ``PLATE_FOURIER_PADDING`` for a plate.
    """

    shape: str = "none"
    semi_axes: tuple[float, float, float] = (1.0, 1.0, 1.0)
    prism_lengths: tuple[float, float, float] = (1.0, 1.0, 1.0)
    plate_normal_axis: str = "c"
    rotation: Rotation | None = None
    normal_hkl: Triplet | None = None
    in_plane_uvw: Triplet | None = None
    voxels: VoxelMask | None = None
    voxel_spacing: Triplet = (1.0, 1.0, 1.0)
    plate_mask: VoxelMask | None = None
    plate_size: tuple[float, float] = (1.0, 1.0)
    plate_thickness: float = _DEFAULT_PLATE_THICKNESS
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

    def plate_outline_model(self) -> VoxelModel:
        """Return an ``image_plate`` as one layer spanning its thickness, enough to draw it."""

        in_plane, layers, (step_a, step_b, step_c) = self._plate_layout()
        return VoxelModel(
            mask=extrude_plate(in_plane, 1), spacing=(step_a, step_b, step_c * layers)
        )

    def plate_cells_across(self) -> int:
        """Return how many voxels the narrower side of an ``image_plate`` outline spans."""

        return min(_trimmed_outline(self._plate_layout()[0]).shape)

    def plate_narrow_outline_error(self) -> float | None:
        """Return how far a factor can be off for an outline this narrow, or ``None`` if not narrow.

        The figures are the largest errors measured on rectangles this many voxels across,
        rounded up. The check reads the outline's bounding box only.
        """

        return _NARROW_OUTLINE_ERROR.get(self.plate_cells_across())

    def _plate_layout(self) -> tuple[np.ndarray, int, Triplet]:
        """Return a plate's in-plane voxels, its layers and the voxel edges, without extruding."""

        if self.plate_mask is None or len(self.plate_mask.shape) != 2:
            raise DemagnetizationError("the plate has no outline; choose a photograph")
        width, height = _positive_pair(self.plate_size, "plate size")
        thickness = float(self.plate_thickness)
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise DemagnetizationError("the plate thickness must be positive")
        longest = max(width, height, thickness)
        counts = tuple(
            max(1, int(round(PLATE_FOURIER_GRID * (edge / longest))))
            for edge in (width, height, thickness)
        )
        layers = counts[2]
        in_plane = _resampled_outline(self.plate_mask, counts[:2])
        if not np.any(in_plane):
            raise DemagnetizationError(
                "the outline vanishes at this resolution; outline a larger part of the photo"
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
            "rotation": None if self.rotation is None else [list(row) for row in self.rotation],
            "normal_hkl": None if self.normal_hkl is None else list(self.normal_hkl),
            "in_plane_uvw": None if self.in_plane_uvw is None else list(self.in_plane_uvw),
            "voxels": None if self.voxels is None else self.voxels.to_json(),
            "voxel_spacing": list(self.voxel_spacing),
            "plate_mask": None if self.plate_mask is None else self.plate_mask.to_json(),
            "plate_size": list(self.plate_size),
            "plate_thickness": float(self.plate_thickness),
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
        """Read a sample; a missing or unusable orientation reads as aligned.

        The ``fourier_grid``, ``thickness_voxels``, ``padding`` and ``smoothing``
        keys of a sample saved when they were settings are ignored.
        """

        if not isinstance(payload, Mapping):
            return cls()
        sample = cls(
            shape=str(payload.get("shape", "none")).strip() or "none",
            semi_axes=_float_triplet(payload.get("semi_axes"), default=(1.0, 1.0, 1.0)),
            prism_lengths=_float_triplet(payload.get("prism_lengths"), default=(1.0, 1.0, 1.0)),
            plate_normal_axis=str(payload.get("plate_normal_axis", "c")).strip() or "c",
            voxels=VoxelMask.from_json(payload.get("voxels")),
            voxel_spacing=_float_triplet(payload.get("voxel_spacing"), default=(1.0, 1.0, 1.0)),
            plate_mask=VoxelMask.from_json(payload.get("plate_mask")),
            plate_size=_float_pair(payload.get("plate_size"), default=(1.0, 1.0)),
            plate_thickness=_json_float(payload.get("plate_thickness"), _DEFAULT_PLATE_THICKNESS),
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


def voxel_demagnetizing_tensor(
    mask: Sequence[object] | np.ndarray,
    spacing: float | Sequence[float],
    *,
    padding: float = FOURIER_PADDING,
) -> FourierDemagnetizingResult:
    """Return the magnetometric demagnetizing tensor of a voxel shape by the Fourier method.

    ``mask`` is a 3-D boolean array of voxels with edges ``spacing`` (a scalar
    or one per axis). Its FFT is the shape amplitude ``D(k)``; the tensor is
    ``<N_ij> = sum_k |D(k)|^2 k_i k_j / k^2`` normalised by the sample and box
    volumes, with ``k = 0`` shared equally between the diagonal elements so
    the trace is 1 by Parseval's theorem. A trace more than 1 % from 1 means the
    sum went wrong and raises :class:`DemagnetizationError`, as does running out
    of memory for the box.

    The FFT box is periodic. It is ``padding`` (at least 2) times the longest
    sample edge along every axis, not along each axis separately: a thin plate
    padded only through its thickness would sit in a stack of its images and
    look like bulk.
    """

    voxels, step, box = _fourier_box(mask, spacing, padding)
    try:
        tensor = _voxel_tensor(voxels, step, box)
    except MemoryError as exc:
        raise _out_of_memory(box) from exc
    return _checked_result(tensor, box, int(np.count_nonzero(voxels)))


def _voxel_tensor(voxels: np.ndarray, step: Triplet, box: tuple[int, int, int]) -> np.ndarray:
    """Return the Fourier sum of a voxel shape, by an FFT of the whole box."""

    spectrum = scipy_fft.rfftn(_embedded(voxels, box), workers=-1)
    power = spectrum.real**2 + spectrum.imag**2
    del spectrum
    k_axes = _k_axes(box, step)
    power *= _half_axis_weights(box[2])
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
    tensor /= float(np.count_nonzero(voxels)) * float(np.prod(box))
    return tensor


def _plate_tensor(
    outline: np.ndarray, layers: int, step: Triplet, box: tuple[int, int, int]
) -> np.ndarray:
    """Return the same Fourier sum as :func:`_voxel_tensor` for identical layers of ``outline``.

    The shape amplitude factorises, ``D(kx, ky, kz) = D2(kx, ky) S(kz)``, so the sum takes one
    2-D FFT, one 1-D profile and a loop over ``kz`` whose memory is 2-D. It is the same number
    as the FFT of the whole box, not an approximation, and needs none of the box's memory.
    """

    embedded = np.zeros(box[:2], dtype=float)
    embedded[: outline.shape[0], : outline.shape[1]] = outline
    spectrum = scipy_fft.rfft2(embedded, workers=-1)
    del embedded
    power = spectrum.real**2 + spectrum.imag**2
    del spectrum
    kx = 2.0 * np.pi * np.fft.fftfreq(box[0], d=step[0])
    ky = 2.0 * np.pi * np.fft.rfftfreq(box[1], d=step[1])
    kz = 2.0 * np.pi * np.fft.rfftfreq(box[2], d=step[2])
    profile = np.zeros(box[2], dtype=float)
    profile[:layers] = 1.0
    layer_spectrum = scipy_fft.rfft(profile)
    layer_power = (layer_spectrum.real**2 + layer_spectrum.imag**2) * _half_axis_weights(box[2])
    q2 = kx[:, None] ** 2 + ky[None, :] ** 2
    # kernel is the sum over kz of |S(kz)|^2 / (q^2 + kz^2).
    kernel = np.zeros_like(q2)
    term = np.empty_like(q2)
    for weight, kz_value in zip(layer_power, kz, strict=True):
        np.add(q2, kz_value**2, out=term)
        if kz_value == 0.0:
            term[0, 0] = np.inf
        np.divide(weight, term, out=term)
        kernel += term
    weighted = power * _half_axis_weights(box[1])[None, :]
    origin = float(power[0, 0] * layer_power[0])
    total_layer_power = float(layer_power.sum())
    along_c = total_layer_power - q2 * kernel
    along_c[0, 0] = total_layer_power - float(layer_power[0])
    tensor = np.zeros((3, 3), dtype=float)
    tensor[0, 0] = float(np.sum(weighted * kx[:, None] ** 2 * kernel))
    tensor[1, 1] = float(np.sum(weighted * ky[None, :] ** 2 * kernel))
    tensor[2, 2] = float(np.sum(weighted * along_c))
    # The elements with c vanish: the outline's spectrum is even under (kx, ky) -> (-kx, -ky).
    tensor[0, 1] = tensor[1, 0] = float(
        np.sum(
            weighted
            * _without_nyquist(kx, box[0])[:, None]
            * _without_nyquist(ky, box[1])[None, :]
            * kernel
        )
    )
    tensor += np.eye(3) * origin / 3.0
    tensor /= float(np.count_nonzero(outline) * layers) * float(np.prod(box))
    return tensor


def _checked_result(
    tensor: np.ndarray, box: tuple[int, int, int], count: int
) -> FourierDemagnetizingResult:
    trace = float(np.trace(tensor))
    # Written so that a NaN trace is refused too.
    if not abs(trace - 1.0) <= _TRACE_TOLERANCE:
        raise DemagnetizationError(
            f"the demagnetizing tensor has trace {trace:.4f} rather than 1, so it is not used"
        )
    return FourierDemagnetizingResult(tensor=tensor, trace=trace, box_shape=box, voxel_count=count)


def _out_of_memory(box: tuple[int, int, int]) -> DemagnetizationError:
    size = " x ".join(str(n) for n in box)
    return DemagnetizationError(f"there is not enough memory for the {size} FFT box")


def fourier_demagnetizing_result(sample: DemagnetizingSample) -> FourierDemagnetizingResult:
    """Return the Fourier-method result for a ``voxel_grid`` or ``image_plate`` sample.

    The result is along the sample axes and is cached, since a calculation and
    the dialog ask for the same shape repeatedly.
    """

    if not sample.uses_fourier_method:
        raise DemagnetizationError(f"the '{sample.shape}' shape has a closed-form tensor")
    shape_only = replace(sample, rotation=None, normal_hkl=None, in_plane_uvw=None, photo=None)
    return _cached_fourier_result(shape_only)


@lru_cache(maxsize=4)
def _resampled_outline(outline: VoxelMask, counts: tuple[int, int]) -> np.ndarray:
    """Return a stored outline at ``counts`` voxels; a dialog asks for it on every edit."""

    resampled = resample_mask(outline.array(), counts)
    resampled.flags.writeable = False
    return resampled


@lru_cache(maxsize=8)
def _cached_fourier_result(sample: DemagnetizingSample) -> FourierDemagnetizingResult:
    if _normalize_shape(sample.shape) == "image_plate":
        outline, layers, step, box = _plate_box(sample)
        try:
            tensor = _plate_tensor(outline, layers, step, box)
        except MemoryError as exc:
            raise _out_of_memory(box) from exc
        result = _checked_result(tensor, box, int(np.count_nonzero(outline)) * layers)
    else:
        model = sample.voxel_model()
        result = voxel_demagnetizing_tensor(model.mask, model.spacing)
    result.tensor.flags.writeable = False
    return result


def fourier_box_shape(
    mask_shape: Sequence[int], spacing: float | Sequence[float], padding: float
) -> tuple[int, int, int]:
    """Return the FFT box, in voxels, for a sample of ``mask_shape`` voxels.

    Every axis spans at least ``padding`` times the longest sample edge,
    rounded up to a size the FFT handles quickly. Raises :class:`OverflowError` for a
    length no FFT could take.
    """

    step = _spacing3(spacing)
    if not all(math.isfinite(edge) for edge in step):
        raise DemagnetizationError("spacing must be finite")
    pad = float(padding)
    if not math.isfinite(pad) or pad < _MIN_PADDING:
        raise DemagnetizationError(f"padding must be at least {_MIN_PADDING:g}")
    counts = tuple(int(n) for n in mask_shape)
    if len(counts) != 3 or min(counts) < 1:
        raise DemagnetizationError("the voxel shape must be a non-empty 3-D array")
    edge = pad * max(n * s for n, s in zip(counts, step, strict=True))
    lengths = [
        max(2 * n, int(math.ceil(edge / s - 1e-9))) for n, s in zip(counts, step, strict=True)
    ]
    # next_fast_len has its own limit, and raises ValueError at it or OverflowError beyond.
    if max(lengths) > _MAX_FFT_LENGTH:
        raise OverflowError(f"an FFT length of {max(lengths)} is too large")
    return tuple(int(scipy_fft.next_fast_len(length)) for length in lengths)  # type: ignore[return-value]


def plate_outline_mask(mask: Sequence[object] | np.ndarray) -> VoxelMask:
    """Return an in-plane outline as an ``image_plate`` stores it: at most 256 cells along its longer side."""

    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or not np.any(values):
        raise DemagnetizationError("a plate outline must be a 2-D mask with the sample in it")
    return VoxelMask.from_array(fit_mask(values, _STORED_MASK_EDGE))


def sample_fourier_box(sample: DemagnetizingSample) -> tuple[int, int, int]:
    """Return the FFT box, in voxels, the Fourier method would use for ``sample``.

    It applies the same checks as the calculation, without the FFT, so a
    dialog can report a refusal before anything is computed:
    :class:`DemagnetizationError` says what the sample would have to change.
    """

    if not sample.uses_fourier_method:
        raise DemagnetizationError(f"the '{sample.shape}' shape has a closed-form tensor")
    if _normalize_shape(sample.shape) == "image_plate":
        return _plate_box(sample)[3]
    model = sample.voxel_model()
    _voxels, _step, box = _fourier_box(model.mask, model.spacing, FOURIER_PADDING)
    return box


def _trimmed_outline(in_plane: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(np.any(in_plane, axis=1))
    columns = np.flatnonzero(np.any(in_plane, axis=0))
    return in_plane[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]


def _plate_box(
    sample: DemagnetizingSample,
) -> tuple[np.ndarray, int, Triplet, tuple[int, int, int]]:
    """Return a plate's outline, layers, voxel edges and FFT box, refusing a box too slow to sum."""

    in_plane, layers, step = sample._plate_layout()
    outline = _trimmed_outline(in_plane)
    edges = (*sample.plate_size, float(sample.plate_thickness))
    if max(edges) > _THINNEST_PLATE_RATIO * min(edges):
        raise DemagnetizationError(_plate_refusal(sample))
    box = fourier_box_shape((*outline.shape, layers), step, PLATE_FOURIER_PADDING)
    if math.prod(box) > _MAX_PLATE_BOX_VOXELS:
        raise DemagnetizationError(_plate_refusal(sample))
    return outline, layers, step, box


def _plate_refusal(sample: DemagnetizingSample) -> str:
    width, height = sample.plate_size
    thickness = float(sample.plate_thickness)
    if min(width, height) >= _INFINITE_PLATE_MIN_RATIO * thickness:
        try:
            exact = demagnetizing_factors_rectangular_prism((width, height, thickness))[2]
        except ArithmeticError:
            # Proportions too extreme to evaluate, where the plate is an infinite one to rounding.
            exact = 1.0
        off_by = _rounded_up(1.0 - exact)
        return (
            f"the plate is too thin: its longest edge can be at most {_THINNEST_PLATE_RATIO} "
            f"times its thickness. Choose Infinite plate, which has its normal along c and is "
            f"within {off_by:.1g} for a rectangular plate of this size, or a Rectangular prism, "
            "which keeps the orientation"
        )
    return (
        "the sample's edges are too different in length for the Fourier method. A Rectangular "
        "prism has a closed form at any proportions"
    )


def _fourier_box(
    mask: Sequence[object] | np.ndarray,
    spacing: float | Sequence[float],
    padding: float,
) -> tuple[np.ndarray, Triplet, tuple[int, int, int]]:
    voxels = np.asarray(mask)
    if voxels.dtype != bool:
        voxels = voxels.astype(bool)
    if voxels.ndim != 3:
        raise DemagnetizationError("the voxel shape must be a 3-D array")
    if not np.any(voxels):
        raise DemagnetizationError("the voxel shape is empty")
    hits = [np.flatnonzero(np.any(voxels, axis=axes)) for axes in ((1, 2), (0, 2), (0, 1))]
    voxels = voxels[
        hits[0][0] : hits[0][-1] + 1,
        hits[1][0] : hits[1][-1] + 1,
        hits[2][0] : hits[2][-1] + 1,
    ]
    step = _spacing3(spacing)
    return voxels, step, _checked_box(voxels.shape, step, padding)


def _rounded_up(value: float) -> float:
    """Return ``value`` rounded up to one significant figure, so that "within" it holds."""

    # Below this the closed form's own rounding decides, so it is as far as "within" can say.
    value = max(value, 1e-12)
    scale = 10.0 ** math.floor(math.log10(value))
    return math.ceil(value / scale) * scale


def _checked_box(
    extent: Sequence[int], spacing: float | Sequence[float], padding: float
) -> tuple[int, int, int]:
    """Return the FFT box for a sample of ``extent`` voxels, refusing one too large to hold."""

    try:
        box = fourier_box_shape(extent, spacing, padding)
    except OverflowError as exc:
        raise DemagnetizationError("the FFT box is too large to count; use coarser voxels") from exc
    # Python integers: a box this far over the limit would wrap around in a NumPy product.
    total = math.prod(box)
    if total > _MAX_BOX_VOXELS:
        raise DemagnetizationError(
            f"the FFT box would hold {total / 1e6:.1f} million voxels (limit "
            f"{_MAX_BOX_VOXELS / 1e6:.1f} million); use coarser voxels"
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


def _half_axis_weights(length: int) -> np.ndarray:
    """Return the weights of an axis's non-negative half, as a real FFT of ``length`` keeps it.

    Every column but k = 0 and the even-length Nyquist column stands for itself and its mirror
    image.
    """

    weights = np.full(length // 2 + 1, 2.0)
    weights[0] = 1.0
    if length % 2 == 0:
        weights[-1] = 1.0
    return weights


def _without_nyquist(k: np.ndarray, length: int) -> np.ndarray:
    if length % 2:
        return k
    trimmed = k.copy()
    trimmed[length // 2 if len(k) == length else -1] = 0.0
    return trimmed


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


def _json_float(value: object, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number) or number <= 0.0:
        return default
    return number


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
    # Written so that NaN is refused too.
    if not (a > 0.0 and b > 0.0 and c > 0.0):
        raise DemagnetizationError(f"{name} values must be positive")
    return (a, b, c)
