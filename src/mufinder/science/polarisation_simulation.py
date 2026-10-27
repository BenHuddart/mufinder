"""One polarisation simulation: settings, the traces they give and their spectrum.

:class:`PolarisationSettings` collects every input of a simulation that is not
a field result: the geometry, the crystal orientation, the sample, the
damping, the time grid, the strong-collision dynamics and the spectrum
processing. :func:`simulate_polarisation` runs
:func:`~mufinder.science.polarisation.calculate_polarisation` (or its
longitudinal and transverse forms) on the field results, the dynamics of
:mod:`mufinder.science.dynamics` on the weighted average, and
:func:`~mufinder.science.polarisation.calculate_spectrum` on the trace that is
shown, the dynamic one when there is one.

Strong collisions redraw the field from the whole static distribution, every
site included, so the dynamics act on the site-weighted average and not on
each site. The scalar strong-collision equation holds only where symmetry
leaves the measured component the only one that survives a redraw: a
polycrystal in zero field, measured along the spin, and a polycrystal in a
longitudinal field, whose average is axial about the field. Dynamics are
refused in every other geometry, where they would need the whole spin
evolution rather than one projected trace.

The crystal orientation is given as the two lattice directions ``[uvw]`` that
lie along the laboratory ``z`` axis (the beam and the initial spin in a
longitudinal geometry) and the laboratory ``x`` axis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal

import numpy as np

from mufinder.domain.constants import MUON_GAMMA_OVER_2PI_MHZ_T
from mufinder.science.dynamics import (
    DynamicPolarisationResult,
    DynamicsMethod,
    calculate_dynamic_polarisation,
)
from mufinder.science.magnetic_fields import MagneticFieldResult
from mufinder.science.nuclear_relaxation import NuclearWidthTensors
from mufinder.science.polarisation import (
    FieldResult,
    PolarisationError,
    PolarisationResult,
    SpectrumResult,
    calculate_lf_polarisation,
    calculate_polarisation,
    calculate_spectrum,
    calculate_tf_polarisation,
)

PolarisationGeometry = Literal["zero_field", "longitudinal", "transverse"]

POLARISATION_GEOMETRIES: tuple[PolarisationGeometry, ...] = (
    "zero_field",
    "longitudinal",
    "transverse",
)
DYNAMICS_METHODS: tuple[DynamicsMethod, ...] = ("exact", "fft")

# Largest number of time points a simulation accepts; the exact dynamics cost O(N²).
MAX_TIME_POINTS = 65_536
# Largest number of Fibonacci-sphere orientations of a powder average in a field.
MAX_SPHERE_POINTS = 20_000
# Largest zero-padding factor of the spectrum.
MAX_PADDING_FACTOR = 16
# Directions closer than this (sine of the angle) count as parallel.
_PARALLEL_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class PolarisationSettings:
    """Every input of a polarisation simulation except the field results.

    ``polarisation_direction`` and ``detector_direction`` are laboratory-frame
    directions used by the zero-field geometry; ``detector_direction`` ``None``
    means the same as the spin. The longitudinal and transverse geometries
    take their axis from the applied field. ``crystal_along_z_uvw`` and
    ``crystal_along_x_uvw`` are the lattice directions along the laboratory
    axes; the second is made perpendicular to the first.

    ``added_applied_field_T`` (crystal frame) adds an applied field to field
    results computed without one. ``sigma_from_nuclear_widths`` replaces
    ``sigma_per_us`` with the transverse width along each sample's local
    field. ``collision_rate_per_us`` zero means static; a non-zero rate needs
    a polycrystal in zero field measured along the spin, or in a longitudinal
    field (see the module notes). The apodisation times
    ``None`` mean no window.
    """

    geometry: PolarisationGeometry = "zero_field"
    polarisation_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    detector_direction: tuple[float, float, float] | None = None
    crystal_along_z_uvw: tuple[float, float, float] = (0.0, 0.0, 1.0)
    crystal_along_x_uvw: tuple[float, float, float] = (1.0, 0.0, 0.0)
    added_applied_field_T: tuple[float, float, float] | None = None
    polycrystalline: bool = False
    sphere_points: int = 600
    lambda_z_per_us: float = 0.0
    sigma_per_us: float = 0.0
    sigma_from_nuclear_widths: bool = False
    end_time_us: float = 10.0
    time_points: int = 2001
    collision_rate_per_us: float = 0.0
    dynamics_method: DynamicsMethod = "exact"
    subtract_baseline: bool = True
    apodisation_sigma_us: float | None = None
    apodisation_lambda_us: float | None = None
    padding_factor: int = 1

    def __post_init__(self) -> None:
        if self.geometry not in POLARISATION_GEOMETRIES:
            raise PolarisationError(f"unknown geometry {self.geometry!r}")
        if self.dynamics_method not in DYNAMICS_METHODS:
            raise PolarisationError(f"unknown dynamics method {self.dynamics_method!r}")
        for name in ("polarisation_direction", "crystal_along_z_uvw", "crystal_along_x_uvw"):
            _nonzero_vector(getattr(self, name), name)
        if self.detector_direction is not None:
            _nonzero_vector(self.detector_direction, "detector_direction")
        if self.added_applied_field_T is not None:
            _finite_vector(self.added_applied_field_T, "added_applied_field_T")
        _integer_in_range(self.sphere_points, "sphere_points", 1, MAX_SPHERE_POINTS)
        _integer_in_range(self.time_points, "time_points", 2, MAX_TIME_POINTS)
        _integer_in_range(self.padding_factor, "padding_factor", 1, MAX_PADDING_FACTOR)
        for name in ("lambda_z_per_us", "sigma_per_us", "collision_rate_per_us"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise PolarisationError(f"{name} must be finite and >= 0")
        if not math.isfinite(self.end_time_us) or self.end_time_us <= 0.0:
            raise PolarisationError("end_time_us must be finite and > 0")
        for name in ("apodisation_sigma_us", "apodisation_lambda_us"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise PolarisationError(f"{name} must be finite and > 0")

    @property
    def times_us(self) -> np.ndarray:
        return np.linspace(0.0, float(self.end_time_us), int(self.time_points))

    @property
    def nyquist_MHz(self) -> float:
        """The highest frequency the time grid resolves."""

        step = float(self.end_time_us) / (int(self.time_points) - 1)
        return 0.5 / step

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        for name, value in payload.items():
            if isinstance(value, tuple):
                payload[name] = list(value)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> PolarisationSettings:
        """Rebuild settings from :meth:`to_json` output; missing keys take their defaults."""

        defaults = cls()
        values: dict[str, Any] = {}
        try:
            for name in cls.__dataclass_fields__:
                if name not in payload:
                    continue
                raw = payload[name]
                default = getattr(defaults, name)
                if name in _OPTIONAL_VECTORS:
                    values[name] = None if raw is None else _tuple3(raw)
                elif name in _OPTIONAL_FLOATS:
                    values[name] = None if raw is None else float(raw)
                elif isinstance(default, tuple):
                    values[name] = _tuple3(raw)
                elif isinstance(default, bool):
                    if not isinstance(raw, bool):
                        raise ValueError(f"{name} must be true or false")
                    values[name] = raw
                elif isinstance(default, int):
                    values[name] = _strict_int(raw, name)
                elif isinstance(default, float):
                    values[name] = float(raw)
                else:
                    values[name] = str(raw)
        except (TypeError, ValueError) as exc:
            raise PolarisationError(f"malformed polarisation settings: {exc}") from exc
        return cls(**values)


_OPTIONAL_VECTORS = frozenset({"detector_direction", "added_applied_field_T"})
_OPTIONAL_FLOATS = frozenset({"apodisation_sigma_us", "apodisation_lambda_us"})


@dataclass(frozen=True, slots=True)
class PolarisationSimulation:
    """The traces of one simulation and the spectrum of the trace shown.

    ``dynamic`` is ``None`` for a static simulation. ``spectrum`` is computed
    from ``dynamic.polarisation`` when there is one, from the static weighted
    average otherwise. ``warnings`` lists conditions worth showing, such as a
    time grid too coarse for the largest field.
    """

    static: PolarisationResult
    dynamic: DynamicPolarisationResult | None
    spectrum: SpectrumResult
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def shown_polarisation(self) -> np.ndarray:
        return self.static.polarisation if self.dynamic is None else self.dynamic.polarisation

    def to_json(self) -> dict[str, object]:
        """The traces; the spectrum is recomputed from them and the settings."""

        return {
            "static": self.static.to_json(),
            "dynamic": None if self.dynamic is None else self.dynamic.to_json(),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_json(
        cls, payload: Mapping[str, Any], settings: PolarisationSettings
    ) -> PolarisationSimulation:
        try:
            static = PolarisationResult.from_json(payload["static"])
            raw_dynamic = payload.get("dynamic")
            dynamic = (
                None if raw_dynamic is None else DynamicPolarisationResult.from_json(raw_dynamic)
            )
            raw_warnings = payload.get("warnings") or []
            if not isinstance(raw_warnings, list):
                raise TypeError("warnings must be a list")
            warnings = tuple(str(item) for item in raw_warnings)
        except (KeyError, TypeError) as exc:
            raise PolarisationError(f"malformed polarisation simulation: {exc}") from exc
        shown = static.polarisation if dynamic is None else dynamic.polarisation
        return cls(
            static=static,
            dynamic=dynamic,
            spectrum=_spectrum(static.times_us, shown, settings),
            warnings=warnings,
        )


def crystal_rotation_from_directions(
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    along_z_uvw: Sequence[float],
    along_x_uvw: Sequence[float],
) -> np.ndarray:
    """Return the rotation taking crystal Cartesian vectors to the laboratory frame.

    ``lattice_vectors`` are the rows ``a``, ``b``, ``c`` in Å. The lattice
    direction ``along_z_uvw`` is put along laboratory ``z`` and the part of
    ``along_x_uvw`` perpendicular to it along laboratory ``x``.
    """

    lattice = np.asarray(lattice_vectors, dtype=float)
    if lattice.shape != (3, 3) or abs(float(np.linalg.det(lattice))) < 1e-12:
        raise PolarisationError("lattice_vectors must be three independent vectors")
    z_axis = _nonzero_vector(along_z_uvw, "along_z_uvw") @ lattice
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = _nonzero_vector(along_x_uvw, "along_x_uvw") @ lattice
    x_axis = x_axis - (x_axis @ z_axis) * z_axis
    size = float(np.linalg.norm(x_axis))
    if size < _PARALLEL_TOLERANCE * float(np.linalg.norm(np.asarray(along_x_uvw) @ lattice)):
        raise PolarisationError("the directions along laboratory z and x must not be parallel")
    x_axis = x_axis / size
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack((x_axis, y_axis, z_axis))


def simulate_polarisation(
    settings: PolarisationSettings,
    field_results: Sequence[FieldResult],
    *,
    lattice_vectors: Sequence[Sequence[float]] | np.ndarray,
    site_weights: Sequence[float] | None = None,
    nuclear_width_tensors: Sequence[NuclearWidthTensors] | None = None,
) -> PolarisationSimulation:
    """Run one simulation of ``field_results`` with ``settings``.

    ``site_weights`` weight the field results, equal by default.
    ``nuclear_width_tensors``, one per field result, are needed when
    ``settings.sigma_from_nuclear_widths`` is set.
    """

    results = list(field_results)
    if not results:
        raise PolarisationError("select at least one field result")
    if settings.sigma_from_nuclear_widths and (
        nuclear_width_tensors is None or len(nuclear_width_tensors) != len(results)
    ):
        raise PolarisationError("the nuclear widths need one width tensor per field result")
    added = settings.added_applied_field_T
    if added is not None and not any(value != 0.0 for value in added):
        added = None
    if settings.geometry == "zero_field" and (
        added is not None or any(_carries_applied_field(result) for result in results)
    ):
        raise PolarisationError(
            "the zero-field geometry needs field results without an applied field; "
            "choose the longitudinal or transverse geometry"
        )
    if settings.collision_rate_per_us > 0.0 and not _dynamics_reduce_to_one_component(settings):
        raise PolarisationError(
            "strong-collision dynamics need a polycrystal in zero field with the detector along "
            "the spin, or a polycrystal in a longitudinal field"
        )
    times = settings.times_us
    rotation = crystal_rotation_from_directions(
        lattice_vectors, settings.crystal_along_z_uvw, settings.crystal_along_x_uvw
    )
    options: dict[str, Any] = {
        "crystal_rotation": rotation,
        "applied_field": added,
        "lambda_z": float(settings.lambda_z_per_us),
        "sigma_nuclear": "auto"
        if settings.sigma_from_nuclear_widths
        else float(settings.sigma_per_us),
        "nuclear_width_tensors": nuclear_width_tensors
        if settings.sigma_from_nuclear_widths
        else None,
        "polycrystalline": bool(settings.polycrystalline),
        "sphere_points": int(settings.sphere_points),
        "site_weights": site_weights,
    }
    if settings.geometry == "longitudinal":
        static = calculate_lf_polarisation(results, times, **options)
    elif settings.geometry == "transverse":
        static = calculate_tf_polarisation(results, times, **options)
    else:
        static = calculate_polarisation(
            results,
            times,
            polarisation_direction=settings.polarisation_direction,
            detector_direction=settings.detector_direction,
            **options,
        )

    dynamic = None
    if settings.collision_rate_per_us > 0.0:
        dynamic = calculate_dynamic_polarisation(
            static.polarisation,
            times,
            float(settings.collision_rate_per_us),
            method=settings.dynamics_method,
        )
    shown = static.polarisation if dynamic is None else dynamic.polarisation
    return PolarisationSimulation(
        static=static,
        dynamic=dynamic,
        spectrum=_spectrum(times, shown, settings),
        warnings=_aliasing_warnings(results, added, settings),
    )


def _spectrum(
    times: np.ndarray, polarisation: np.ndarray, settings: PolarisationSettings
) -> SpectrumResult:
    return calculate_spectrum(
        times,
        polarisation,
        subtract_baseline=bool(settings.subtract_baseline),
        apodisation_sigma_us=settings.apodisation_sigma_us,
        apodisation_lambda_us=settings.apodisation_lambda_us,
        padding_factor=int(settings.padding_factor),
    )


def _dynamics_reduce_to_one_component(settings: PolarisationSettings) -> bool:
    """Whether symmetry reduces the strong-collision dynamics to the measured trace alone."""

    if not settings.polycrystalline:
        return False
    if settings.geometry == "longitudinal":
        return True
    if settings.geometry != "zero_field":
        return False
    detector = settings.detector_direction
    if detector is None:
        return True
    spin = np.asarray(settings.polarisation_direction, dtype=float)
    detector_unit = np.asarray(detector, dtype=float)
    cosine = float(spin @ detector_unit) / float(
        np.linalg.norm(spin) * np.linalg.norm(detector_unit)
    )
    return cosine > 1.0 - _PARALLEL_TOLERANCE


def _aliasing_warnings(
    results: Sequence[FieldResult],
    added: Sequence[float] | None,
    settings: PolarisationSettings,
) -> tuple[str, ...]:
    """Warn when the largest field precesses faster than the time grid resolves.

    A powder in an applied field turns the applied field through every
    direction in the crystal, so a grain sees up to ``|B_int| + |B_app|``.
    """

    largest = 0.0
    for result in results:
        applied = np.asarray(added if added is not None else _applied(result), dtype=float)
        samples = _field_samples(result) + (0.0 if added is None else applied)
        if settings.polycrystalline and np.any(applied != 0.0):
            internal = samples - applied
            size = float(np.max(np.linalg.norm(internal, axis=1))) + float(np.linalg.norm(applied))
        else:
            size = float(np.max(np.linalg.norm(samples, axis=1)))
        largest = max(largest, size)
    frequency = largest * MUON_GAMMA_OVER_2PI_MHZ_T
    nyquist = settings.nyquist_MHz
    if frequency <= nyquist:
        return ()
    return (
        f"The largest field, {largest * 1e3:.4g} mT, precesses at {frequency:.4g} MHz, above the "
        f"{nyquist:.4g} MHz the time grid resolves; use more points or a shorter end time.",
    )


def _field_samples(result: FieldResult) -> np.ndarray:
    if isinstance(result, MagneticFieldResult):
        return np.asarray(result.total, dtype=float).reshape(1, 3)
    fields = np.asarray(result.fields_T, dtype=float)
    return fields if fields.size else np.zeros((1, 3))


def _applied(result: FieldResult) -> tuple[float, float, float]:
    applied = result.applied if isinstance(result, MagneticFieldResult) else result.applied_T
    return (0.0, 0.0, 0.0) if applied is None else _tuple3(applied)


def _carries_applied_field(result: FieldResult) -> bool:
    return any(value != 0.0 for value in _applied(result))


def _strict_int(value: object, name: str) -> int:
    number = float(value)  # type: ignore[arg-type]
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _integer_in_range(value: object, name: str, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PolarisationError(f"{name} must be an integer")
    if not minimum <= int(value) <= maximum:
        raise PolarisationError(f"{name} must be between {minimum} and {maximum}")


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise PolarisationError(f"{name} must have three finite components")
    return vector


def _nonzero_vector(values: Sequence[float], name: str) -> np.ndarray:
    vector = _finite_vector(values, name)
    if not np.any(vector != 0.0):
        raise PolarisationError(f"{name} must be non-zero")
    return vector


def _tuple3(values: object) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in values)  # type: ignore[union-attr]
    return (x, y, z)
