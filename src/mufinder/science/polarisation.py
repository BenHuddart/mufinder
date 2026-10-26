"""Muon-spin polarisation functions and spectra from local-field results.

A muon spin that starts along ``p̂`` in a static local field ``B``, with
``n̂ = B/|B|`` and ``ω = γ_μ|B|``, precesses about ``n̂`` (``dS/dt = γ_μ S × B``).
Measured along a detector axis ``d̂`` its polarisation is

    P_d(t) = (d̂·n̂)(p̂·n̂) e^{−λ_z t}
           + {[d̂·p̂ − (d̂·n̂)(p̂·n̂)] cos ωt + d̂·(p̂ × n̂) sin ωt} e^{−σ² t²/2}.

With ``d̂ = p̂`` this is the familiar ``cos²θ e^{−λt} + sin²θ e^{−σ²t²/2} cos ωt``
(Dalmas de Réotier et al., Phys. Rev. B 93, 144419 (2016), Eq. 14). ``λ_z`` is a
phenomenological longitudinal rate and ``σ`` the nuclear transverse-field
width along the local field (:class:`~mufinder.science.nuclear_relaxation.NuclearWidthTensors`).

The trace depends on the field samples only through weighted histograms of
``|B|``, so evaluation costs ``O(N_samples)`` per geometry plus
``O(N_bins × N_times)``, never ``O(N_samples × N_times)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Literal

import numpy as np

from mufinder.domain.constants import MUON_GAMMA_OVER_2PI_MHZ_T, MUON_GAMMA_RAD_S_T
from mufinder.science.magnetic_fields import (
    MagneticFieldDistributionResult,
    MagneticFieldResult,
)
from mufinder.science.nuclear_relaxation import NuclearWidthTensors

FieldResult = MagneticFieldResult | MagneticFieldDistributionResult

# γ_μ in rad μs⁻¹ T⁻¹, so that ω t is in radians for t in μs.
_GAMMA_RAD_US_T = MUON_GAMMA_RAD_S_T * 1e-6
# Largest phase, in radians, a sample may sit from its bin centre at the last time.
_MAX_IN_BIN_PHASE = 0.1
# Upper limit on the number of |B| bins the in-bin phase limit may ask for.
MAX_MAGNITUDE_BINS = 1 << 16
# Occupied bins evaluated together, which bounds the memory of the time evaluation.
_BIN_CHUNK = 512
# Fraction of the trace, at its end, whose mean is the spectrum baseline.
_BASELINE_FRACTION = 0.1
# Tolerance for "perpendicular", "orthogonal" and "uniform" checks.
_TOLERANCE = 1e-6


class PolarisationError(ValueError):
    """Raised when polarisation or spectrum inputs are invalid."""


@dataclass(frozen=True, slots=True)
class PolarisationResult:
    """A muon-spin polarisation trace and the per-site traces it averages.

    ``polarisation`` is the ``site_weights``-weighted mean of the rows of
    ``site_polarisations``. The directions are unit vectors in the laboratory
    frame; ``applied_field_T`` is the applied field in the crystal frame, as the
    field calculation stores it. ``parameters`` records the remaining inputs.
    """

    times_us: np.ndarray
    polarisation: np.ndarray
    site_labels: tuple[str, ...]
    site_polarisations: np.ndarray
    site_weights: np.ndarray
    polarisation_direction: tuple[float, float, float]
    detector_direction: tuple[float, float, float]
    polycrystalline: bool
    applied_field_T: tuple[float, float, float]
    label: str = "P"
    parameters: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "times_us": np.asarray(self.times_us, dtype=float).tolist(),
            "polarisation": np.asarray(self.polarisation, dtype=float).tolist(),
            "site_labels": list(self.site_labels),
            "site_polarisations": np.asarray(self.site_polarisations, dtype=float).tolist(),
            "site_weights": np.asarray(self.site_weights, dtype=float).tolist(),
            "polarisation_direction": list(self.polarisation_direction),
            "detector_direction": list(self.detector_direction),
            "polycrystalline": bool(self.polycrystalline),
            "applied_field_T": list(self.applied_field_T),
            "label": self.label,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> PolarisationResult:
        try:
            times = np.asarray(payload["times_us"], dtype=float)
            sites = np.asarray(payload["site_polarisations"], dtype=float).reshape(-1, times.size)
            raw_parameters = payload.get("parameters")
            return cls(
                times_us=times,
                polarisation=np.asarray(payload["polarisation"], dtype=float),
                site_labels=tuple(str(label) for label in payload["site_labels"]),
                site_polarisations=sites,
                site_weights=np.asarray(payload["site_weights"], dtype=float),
                polarisation_direction=_tuple3(payload["polarisation_direction"]),
                detector_direction=_tuple3(payload["detector_direction"]),
                polycrystalline=bool(payload["polycrystalline"]),
                applied_field_T=_tuple3(payload["applied_field_T"]),
                label=str(payload.get("label", "P")),
                parameters=dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {},
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PolarisationError(f"malformed polarisation result: {exc}") from exc


@dataclass(frozen=True, slots=True)
class SpectrumResult:
    """The Fourier spectrum of a polarisation trace.

    ``fields_T`` is the frequency axis as a field, ``f / (γ_μ/2π)``.
    ``amplitudes`` is ``|spectrum|`` as a density per MHz whose integral over
    frequency equals the oscillating fraction, the processed trace's value at
    its first time. ``spectrum`` is the complex transform in μs.
    """

    frequencies_MHz: np.ndarray
    fields_T: np.ndarray
    amplitudes: np.ndarray
    spectrum: np.ndarray


def calculate_polarisation(
    field_results: FieldResult | Sequence[FieldResult],
    times_us: Sequence[float] | np.ndarray,
    *,
    polarisation_direction: Sequence[float] = (0.0, 0.0, 1.0),
    detector_direction: Sequence[float] | None = None,
    crystal_rotation: Sequence[Sequence[float]] | np.ndarray | None = None,
    applied_field: Sequence[float] | None = None,
    lambda_z: float = 0.0,
    sigma_nuclear: float | Literal["auto"] = 0.0,
    nuclear_width_tensors: Sequence[NuclearWidthTensors] | None = None,
    polycrystalline: bool = False,
    sphere_points: int = 600,
    site_weights: Sequence[float] | None = None,
    magnitude_bins: int = 2048,
) -> PolarisationResult:
    """Return the polarisation along ``detector_direction`` from stored field results.

    Each result is one site: a :class:`MagneticFieldResult` is a single field,
    a :class:`MagneticFieldDistributionResult` a set of equally weighted field
    samples. Sites are averaged with ``site_weights``, equal by default, which
    is right for equivalent sites and approximant translations.

    ``polarisation_direction`` (the initial muon spin) and ``detector_direction``
    (default: the same) are laboratory-frame directions. ``crystal_rotation`` is
    the rotation matrix taking crystal-frame Cartesian vectors, the frame of
    the fields, to the laboratory frame (default: identity).

    The applied field is part of the field results. ``applied_field`` (T, in
    the crystal frame) adds one to results that carry none, so a zero-field
    calculation serves a longitudinal-field scan; it is refused when a result
    already carries an applied field.

    ``lambda_z`` (μs⁻¹) damps the non-oscillating part and ``sigma_nuclear``
    (μs⁻¹) the oscillating part. ``"auto"`` takes σ²_TF along each sample's
    field direction from ``nuclear_width_tensors``, one per field result.

    ``polycrystalline`` averages over grain orientations. In zero applied field
    this is exact: each sample gives ``(d̂·p̂)[⅓ e^{−λt} + ⅔ e^{−σ²t²/2} cos ωt]``.
    With an applied field, the field's direction in the crystal frame runs over
    ``sphere_points`` points of a Fibonacci sphere, and the rotation of the
    instrument about the applied field is averaged analytically.

    The oscillating part is binned in ``|B|`` on at least ``magnitude_bins``
    bins, more if needed to keep each sample within 0.1 rad of its bin centre
    at the last time (up to :data:`MAX_MAGNITUDE_BINS`); offsets within a bin are
    corrected to second order. ``parameters["max_in_bin_phase_rad"]`` records
    the largest in-bin phase the bins allow, which exceeds 0.1 rad only at
    that limit.
    """

    return _polarisation(
        field_results,
        times_us,
        polarisation_direction=polarisation_direction,
        detector_direction=detector_direction,
        crystal_rotation=crystal_rotation,
        applied_field=applied_field,
        lambda_z=lambda_z,
        sigma_nuclear=sigma_nuclear,
        nuclear_width_tensors=nuclear_width_tensors,
        polycrystalline=polycrystalline,
        sphere_points=sphere_points,
        site_weights=site_weights,
        magnitude_bins=magnitude_bins,
        label="P",
    )


def calculate_lf_polarisation(
    field_results: FieldResult | Sequence[FieldResult],
    times_us: Sequence[float] | np.ndarray,
    *,
    crystal_rotation: Sequence[Sequence[float]] | np.ndarray | None = None,
    applied_field: Sequence[float] | None = None,
    **options: Any,
) -> PolarisationResult:
    """Return ``P_z``, the longitudinal-field polarisation: spin and detector along the field.

    The applied field comes from the results or ``applied_field`` and must be
    non-zero. Other keywords are those of :func:`calculate_polarisation`.
    """

    _refuse_fixed_options(options, ("polarisation_direction", "detector_direction", "label"), "P_z")
    results = _as_results(field_results)
    rotation = _rotation(crystal_rotation)
    axis = _applied_axis(results, rotation, applied_field)
    return _polarisation(
        results,
        times_us,
        polarisation_direction=axis,
        detector_direction=axis,
        crystal_rotation=rotation,
        applied_field=applied_field,
        label="P_z",
        **options,
    )


def calculate_tf_polarisation(
    field_results: FieldResult | Sequence[FieldResult],
    times_us: Sequence[float] | np.ndarray,
    *,
    polarisation_direction: Sequence[float] | None = None,
    crystal_rotation: Sequence[Sequence[float]] | np.ndarray | None = None,
    applied_field: Sequence[float] | None = None,
    **options: Any,
) -> PolarisationResult:
    """Return ``P_x``, the transverse-field polarisation: spin and detector across the field.

    ``polarisation_direction`` must be perpendicular to the applied field in
    the laboratory frame; by default it is the laboratory x axis, or y when
    the field lies along x, made perpendicular to the field. Other keywords
    are those of :func:`calculate_polarisation`.
    """

    _refuse_fixed_options(options, ("detector_direction", "label"), "P_x")
    results = _as_results(field_results)
    rotation = _rotation(crystal_rotation)
    axis = _applied_axis(results, rotation, applied_field)
    if polarisation_direction is None:
        seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        spin = seed - (seed @ axis) * axis
        spin = spin / np.linalg.norm(spin)
    else:
        spin = _unit(polarisation_direction, "polarisation_direction")
        if abs(float(spin @ axis)) > _TOLERANCE:
            raise PolarisationError(
                "a transverse-field polarisation_direction must be perpendicular to the applied field"
            )
    return _polarisation(
        results,
        times_us,
        polarisation_direction=spin,
        detector_direction=spin,
        crystal_rotation=rotation,
        applied_field=applied_field,
        label="P_x",
        **options,
    )


def calculate_spectrum(
    times_us: Sequence[float] | np.ndarray,
    polarisation: Sequence[float] | np.ndarray,
    *,
    subtract_baseline: bool = True,
    apodisation_sigma_us: float | None = None,
    apodisation_lambda_us: float | None = None,
    padding_factor: int = 1,
) -> SpectrumResult:
    """Return the Fourier spectrum of ``polarisation`` on a uniform time grid.

    The baseline, the mean of the last 10 % of points, is subtracted unless
    ``subtract_baseline`` is false. The trace is then multiplied by a Gaussian
    window ``exp(−(t−t₀)²/2τ²)`` with ``τ = apodisation_sigma_us`` and an
    exponential window ``exp(−(t−t₀)/τ)`` with ``τ = apodisation_lambda_us``,
    each when given, zero-padded to ``padding_factor`` times its length and
    transformed with a real FFT.

    For a simulated trace the spectrum is the field histogram convolved with
    the window; its value is in processing the simulation the way the data
    are processed.
    """

    times = np.asarray(times_us, dtype=float)
    values = np.asarray(polarisation, dtype=float)
    if times.ndim != 1 or values.shape != times.shape or times.size < 2:
        raise PolarisationError("times and polarisation must be 1D arrays of the same length >= 2")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise PolarisationError("times and polarisation must be finite")
    steps = np.diff(times)
    step = float(steps[0])
    if step <= 0.0 or not np.allclose(steps, step, rtol=_TOLERANCE, atol=0.0):
        raise PolarisationError("the spectrum needs a uniform, increasing time grid")
    if int(padding_factor) != padding_factor or padding_factor < 1:
        raise PolarisationError("padding_factor must be a positive integer")
    for name, value in (
        ("apodisation_sigma_us", apodisation_sigma_us),
        ("apodisation_lambda_us", apodisation_lambda_us),
    ):
        if value is not None and not value > 0.0:
            raise PolarisationError(f"{name} must be > 0")

    signal = values.copy()
    if subtract_baseline:
        tail = max(1, int(math.ceil(_BASELINE_FRACTION * signal.size)))
        signal -= float(np.mean(signal[-tail:]))
    elapsed = times - times[0]
    if apodisation_sigma_us is not None:
        signal *= np.exp(-(elapsed**2) / (2.0 * apodisation_sigma_us**2))
    if apodisation_lambda_us is not None:
        signal *= np.exp(-elapsed / apodisation_lambda_us)

    length = signal.size * int(padding_factor)
    spectrum = np.fft.rfft(signal, n=length) * step
    frequencies = np.fft.rfftfreq(length, d=step)
    magnitude = np.abs(spectrum)
    total = float(np.sum(magnitude)) / (length * step)
    amplitudes = magnitude * (abs(signal[0]) / total) if total > 0.0 else np.zeros_like(magnitude)
    return SpectrumResult(
        frequencies_MHz=frequencies,
        fields_T=frequencies / MUON_GAMMA_OVER_2PI_MHZ_T,
        amplitudes=amplitudes,
        spectrum=spectrum,
    )


def fibonacci_sphere(count: int) -> np.ndarray:
    """Return ``count`` near-uniform unit vectors on the sphere (a Fibonacci lattice)."""

    if count < 1:
        raise PolarisationError("sphere_points must be >= 1")
    index = np.arange(count, dtype=float)
    z = 1.0 - (2.0 * index + 1.0) / count
    radius = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    angle = index * math.pi * (3.0 - math.sqrt(5.0))
    return np.column_stack((radius * np.cos(angle), radius * np.sin(angle), z))


# --- Implementation ---------------------------------------------------------------


@dataclass(slots=True)
class _Site:
    label: str
    # Field samples in the crystal frame (T) without the applied field.
    internal: np.ndarray
    tensors: NuclearWidthTensors | None
    # |internal|² per sample, reused by every powder orientation.
    internal_squared: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.internal_squared = np.einsum("ij,ij->i", self.internal, self.internal)


@dataclass(slots=True)
class _Geometry:
    """Per-sample |B| and kernel amplitudes for one orientation of one site."""

    magnitudes: np.ndarray
    constant: np.ndarray
    cosine: np.ndarray
    sine: np.ndarray
    # The crystal-frame applied field of this orientation, added to the site's samples.
    applied: np.ndarray


def _polarisation(
    field_results: FieldResult | Sequence[FieldResult],
    times_us: Sequence[float] | np.ndarray,
    *,
    polarisation_direction: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
    detector_direction: Sequence[float] | np.ndarray | None = None,
    crystal_rotation: Sequence[Sequence[float]] | np.ndarray | None = None,
    applied_field: Sequence[float] | None = None,
    lambda_z: float = 0.0,
    sigma_nuclear: float | str = 0.0,
    nuclear_width_tensors: Sequence[NuclearWidthTensors] | None = None,
    polycrystalline: bool = False,
    sphere_points: int = 600,
    site_weights: Sequence[float] | None = None,
    magnitude_bins: int = 2048,
    label: str = "P",
) -> PolarisationResult:
    results = _as_results(field_results)
    times = np.asarray(times_us, dtype=float)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise PolarisationError("times_us must be a non-empty 1D array of finite times")
    spin = _unit(polarisation_direction, "polarisation_direction")
    detector = (
        spin if detector_direction is None else _unit(detector_direction, "detector_direction")
    )
    rotation = _rotation(crystal_rotation)
    if not lambda_z >= 0.0:
        raise PolarisationError("lambda_z must be >= 0")
    if int(magnitude_bins) != magnitude_bins or magnitude_bins < 1:
        raise PolarisationError("magnitude_bins must be a positive integer")
    if int(sphere_points) != sphere_points or sphere_points < 1:
        raise PolarisationError("sphere_points must be a positive integer")
    auto_sigma = isinstance(sigma_nuclear, str)
    if auto_sigma and sigma_nuclear != "auto":
        raise PolarisationError("sigma_nuclear must be a width in μs⁻¹ or 'auto'")
    if not auto_sigma and not float(sigma_nuclear) >= 0.0:
        raise PolarisationError("sigma_nuclear must be >= 0")
    if auto_sigma and (nuclear_width_tensors is None or len(nuclear_width_tensors) != len(results)):
        raise PolarisationError(
            "sigma_nuclear='auto' needs nuclear_width_tensors, one per field result"
        )
    weights = _site_weights(site_weights, len(results))
    applied = _applied_field(results, applied_field)
    sites = [
        _Site(
            label=result.site_label,
            internal=_field_samples(result) - _stored_applied(result),
            tensors=nuclear_width_tensors[index] if auto_sigma and nuclear_width_tensors else None,
        )
        for index, result in enumerate(results)
    ]
    fixed_sigma_squared = 0.0 if auto_sigma else float(sigma_nuclear) ** 2

    applied_size = float(np.linalg.norm(applied))
    powder_field = applied_size if polycrystalline else 0.0
    if powder_field > 0.0:
        axis_lab = rotation @ applied / applied_size
        orientations = fibonacci_sphere(int(sphere_points))

        # One orientation at a time, so memory stays at one site's samples.
        def geometries(site: _Site) -> Iterator[_Geometry]:
            for axis in orientations:
                yield _powder_geometry(site, spin, detector, axis_lab, axis, powder_field)

    else:
        # Crystal-frame directions; the rotation is orthogonal, so its inverse is its transpose.
        spin_c, detector_c = rotation.T @ spin, rotation.T @ detector

        orientations = np.zeros((1, 3))

        def geometries(site: _Site) -> Iterator[_Geometry]:
            yield _single_geometry(site, spin_c, detector_c, applied, polycrystalline)

    edges = _magnitude_edges(sites, applied, powder_field, times, int(magnitude_bins))
    centres = 0.5 * (edges[:-1] + edges[1:]) * _GAMMA_RAD_US_T
    histograms = [
        _site_histogram(geometries(site), len(orientations), site, edges, centres) for site in sites
    ]
    traces = _site_traces(histograms, centres, times, lambda_z, fixed_sigma_squared)
    return PolarisationResult(
        times_us=times,
        polarisation=weights @ traces,
        site_labels=tuple(site.label for site in sites),
        site_polarisations=traces,
        site_weights=weights,
        polarisation_direction=_tuple3(spin),
        detector_direction=_tuple3(detector),
        polycrystalline=bool(polycrystalline),
        applied_field_T=_tuple3(applied),
        label=label,
        parameters={
            "lambda_z": float(lambda_z),
            "sigma_nuclear": "auto" if auto_sigma else float(sigma_nuclear),
            "crystal_rotation": rotation.tolist(),
            "sphere_points": int(sphere_points),
            "magnitude_bins": int(edges.size - 1),
            "max_in_bin_phase_rad": _max_in_bin_phase(edges, times),
        },
    )


def _single_geometry(
    site: _Site,
    spin: np.ndarray,
    detector: np.ndarray,
    applied: np.ndarray,
    polycrystalline: bool,
) -> _Geometry:
    fields = site.internal + applied
    magnitudes, directions = _magnitudes_and_directions(fields)
    overlap = float(spin @ detector)
    if polycrystalline:
        # The sphere average of (d̂·n̂)(p̂·n̂) is (d̂·p̂)/3, and of d̂·(p̂ × n̂) zero.
        constant = np.full(magnitudes.shape, overlap / 3.0)
        sine = np.zeros_like(magnitudes)
    else:
        constant = (directions @ detector) * (directions @ spin)
        sine = directions @ np.cross(detector, spin)
    return _Geometry(magnitudes, constant, overlap - constant, sine, applied)


def _powder_geometry(
    site: _Site,
    spin: np.ndarray,
    detector: np.ndarray,
    axis_lab: np.ndarray,
    axis_crystal: np.ndarray,
    applied_size: float,
) -> _Geometry:
    """One grain orientation, with the applied field along ``axis_crystal`` in the crystal.

    Averaging over the rotation of the instrument about the applied field
    axis ``b̂`` keeps, of ``M = p̂ d̂ᵀ``, only ``b̂ᵀMb̂``, half its trace across
    ``b̂`` and the component of ``d̂ × p̂`` along ``b̂``. The amplitudes then
    depend on a sample only through ``u·n̂``, with ``u`` the field axis in
    the crystal.
    """

    # |B|² = |B_int|² + 2 B (B_int·u) + B², so no per-sample vectors are formed.
    projection = site.internal @ axis_crystal
    magnitudes = np.sqrt(
        np.maximum(site.internal_squared + applied_size * (2.0 * projection + applied_size), 0.0)
    )
    safe = np.where(magnitudes > 0.0, magnitudes, 1.0)
    along = np.where(magnitudes > 0.0, (projection + applied_size) / safe, 0.0)
    overlap = float(spin @ detector)
    parallel = float(axis_lab @ spin) * float(axis_lab @ detector)
    across = 0.5 * (overlap - parallel)
    handed = float(np.cross(detector, spin) @ axis_lab)
    constant = parallel * along**2 + across * (1.0 - along**2)
    sine = handed * along
    return _Geometry(magnitudes, constant, overlap - constant, sine, applied_size * axis_crystal)


@dataclass(slots=True)
class _Histogram:
    """One site's kernel amplitudes binned in |B|.

    ``moments[k]`` is the per-bin sum of ``z δᵏ``, with ``z = C − iS`` and
    ``δ`` the sample's frequency offset from its bin centre; ``sigma_squared``
    is the per-bin nuclear width, or ``None`` when one width damps every bin.
    """

    constant: float
    moments: np.ndarray
    sigma_squared: np.ndarray | None


def _site_histogram(
    geometries: Iterable[_Geometry],
    orientation_count: int,
    site: _Site,
    edges: np.ndarray,
    centres: np.ndarray,
) -> _Histogram:
    """Bin one site's samples, over all its orientations, on the shared |B| grid.

    The nuclear width of a bin is the ``|z|``-weighted mean of its samples' ``σ²``.
    """

    bins = centres.size
    constant = 0.0
    moments = np.zeros((2, 3, bins))
    size_sum = np.zeros(bins)
    sigma_sum = np.zeros(bins)
    for geometry in geometries:
        weight = 1.0 / (orientation_count * geometry.magnitudes.size)
        constant += weight * float(np.sum(geometry.constant))
        scaled = (geometry.magnitudes - edges[0]) * (bins / (edges[-1] - edges[0]))
        index = np.clip(scaled.astype(np.intp), 0, bins - 1)
        offset = geometry.magnitudes * _GAMMA_RAD_US_T - centres[index]
        # z = C − iS, so that Re[z e^{iωt}] = C cos ωt + S sin ωt; S often vanishes.
        parts = (geometry.cosine, -geometry.sine) if np.any(geometry.sine) else (geometry.cosine,)
        power = np.ones_like(offset)
        for order in range(3):
            for part, values in enumerate(parts):
                moments[part, order] += weight * np.bincount(
                    index, weights=values * power, minlength=bins
                )
            power = power * offset
        if site.tensors is not None:
            size = np.hypot(geometry.cosine, geometry.sine)
            sigma_squared = site.tensors.tf_sigma_squared(_field_axes(site, geometry))
            size_sum += weight * np.bincount(index, weights=size, minlength=bins)
            sigma_sum += weight * np.bincount(index, weights=size * sigma_squared, minlength=bins)
    per_bin = None
    if site.tensors is not None:
        per_bin = np.divide(sigma_sum, size_sum, out=np.zeros(bins), where=size_sum > 0.0)
    return _Histogram(constant, moments[0] + 1j * moments[1], per_bin)


def _site_traces(
    histograms: Sequence[_Histogram],
    centres: np.ndarray,
    times: np.ndarray,
    lambda_z: float,
    fixed_sigma_squared: float,
) -> np.ndarray:
    """Evaluate every site's trace, ``Σ_b e^{iω_b t} (Z₀ + i t Z₁ − t² Z₂ / 2) × envelope``.

    The bin centres are evenly spaced, so a chunk's phases are one base phase
    times a table of ``e^{ikΔω t}`` computed once. When one width damps every
    bin its envelope leaves the sum, which becomes a matrix product.
    """

    time = times[:, None]
    traces = np.outer([histogram.constant for histogram in histograms], np.exp(-lambda_z * times))
    moments = np.stack([histogram.moments for histogram in histograms])  # (sites, 3, bins)
    occupied = np.any(moments != 0.0, axis=(0, 1))
    step = centres[1] - centres[0] if centres.size > 1 else 0.0
    table = np.exp(1j * time * (step * np.arange(min(_BIN_CHUNK, centres.size))))
    oscillation = np.zeros((times.size, len(histograms)), dtype=complex)
    # Sites with one width for every bin share the phases in a single matrix product.
    fixed = [site for site, histogram in enumerate(histograms) if histogram.sigma_squared is None]
    damped = [
        site for site, histogram in enumerate(histograms) if histogram.sigma_squared is not None
    ]
    for start in range(0, centres.size, _BIN_CHUNK):
        stop = min(start + _BIN_CHUNK, centres.size)
        if not np.any(occupied[start:stop]):
            continue
        phases = np.exp(1j * times * centres[start])[:, None] * table[:, : stop - start]
        if fixed:
            z0, z1, z2 = (moments[fixed, order, start:stop].T for order in range(3))
            oscillation[:, fixed] += (
                phases @ z0 + 1j * time * (phases @ z1) - 0.5 * time**2 * (phases @ z2)
            )
        for site in damped:
            sigma_squared = histograms[site].sigma_squared[start:stop]
            chunk = phases * np.exp(-0.5 * time**2 * sigma_squared)
            z0, z1, z2 = moments[site, :, start:stop]
            oscillation[:, site] += (
                chunk @ z0 + 1j * times * (chunk @ z1) - 0.5 * times**2 * (chunk @ z2)
            )
    envelopes = np.array(
        [
            np.ones_like(times)
            if histogram.sigma_squared is not None
            else np.exp(-0.5 * fixed_sigma_squared * times**2)
            for histogram in histograms
        ]
    )
    return traces + oscillation.real.T * envelopes


def _field_axes(site: _Site, geometry: _Geometry) -> np.ndarray:
    """Directions for the nuclear width: the field direction, or z where the field vanishes."""

    _, axes = _magnitudes_and_directions(site.internal + geometry.applied)
    axes[geometry.magnitudes == 0.0] = (0.0, 0.0, 1.0)
    return axes


def _magnitude_edges(
    sites: Sequence[_Site],
    applied: np.ndarray,
    powder_field: float,
    times: np.ndarray,
    magnitude_bins: int,
) -> np.ndarray:
    """Return |B| bin edges shared by every site and orientation.

    In a powder with an applied field the range covers every orientation of
    the field. The bin count rises above ``magnitude_bins`` until no sample
    sits more than :data:`_MAX_IN_BIN_PHASE` from its bin centre at the last
    time, up to :data:`MAX_MAGNITUDE_BINS`.
    """

    if powder_field > 0.0:
        internal = np.concatenate([np.linalg.norm(site.internal, axis=1) for site in sites])
        low = max(0.0, float(internal.min()) - powder_field)
        high = float(internal.max()) + powder_field
    else:
        magnitudes = np.concatenate(
            [np.linalg.norm(site.internal + applied, axis=1) for site in sites]
        )
        low, high = float(magnitudes.min()), float(magnitudes.max())
    pad = 1e-9 * max(1.0, high)
    if high - low < pad:
        low, high = max(0.0, low - pad), high + pad
    span = _GAMMA_RAD_US_T * (high - low) * float(np.max(np.abs(times)))
    needed = int(math.ceil(span / (2.0 * _MAX_IN_BIN_PHASE)))
    bins = max(magnitude_bins, min(needed, MAX_MAGNITUDE_BINS))
    return np.linspace(low, high, bins + 1)


def _max_in_bin_phase(edges: np.ndarray, times: np.ndarray) -> float:
    """The largest phase, in radians, a sample can sit from its bin centre at the last time.

    It stays below 0.1 rad unless the bin count reached :data:`MAX_MAGNITUDE_BINS`;
    the second-order in-bin correction is accurate while it is well below 1.
    """

    half_width = 0.5 * _GAMMA_RAD_US_T * (edges[1] - edges[0])
    return float(half_width * np.max(np.abs(times)))


def _magnitudes_and_directions(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    magnitudes = np.sqrt(np.einsum("ij,ij->i", fields, fields))
    safe = np.where(magnitudes > 0.0, magnitudes, 1.0)
    return magnitudes, fields / safe[:, None]


def _as_results(field_results: FieldResult | Sequence[FieldResult]) -> list[FieldResult]:
    if isinstance(field_results, (MagneticFieldResult, MagneticFieldDistributionResult)):
        return [field_results]
    results = list(field_results)
    if not results:
        raise PolarisationError("no field results to evaluate")
    for result in results:
        if not isinstance(result, (MagneticFieldResult, MagneticFieldDistributionResult)):
            raise PolarisationError(f"expected magnetic-field results, got {type(result).__name__}")
    return results


def _field_samples(result: FieldResult) -> np.ndarray:
    if isinstance(result, MagneticFieldResult):
        return np.asarray(result.total, dtype=float).reshape(1, 3)
    fields = np.asarray(result.fields_T, dtype=float)
    if fields.ndim != 2 or fields.shape[1] != 3 or fields.shape[0] == 0:
        raise PolarisationError(f"site {result.site_label!r} has no field samples")
    return fields


def _stored_applied(result: FieldResult) -> np.ndarray:
    """The applied field a result carries; none (zero) for a distribution saved without it."""

    if isinstance(result, MagneticFieldResult):
        return np.asarray(result.applied, dtype=float)
    if result.applied_T is None:
        return np.zeros(3)
    return np.asarray(result.applied_T, dtype=float)


def _applied_field(
    results: Sequence[FieldResult], applied_field: Sequence[float] | None
) -> np.ndarray:
    stored = [_stored_applied(result) for result in results]
    if applied_field is not None:
        if any(np.any(vector != 0.0) for vector in stored):
            raise PolarisationError(
                "the field results already include an applied field; "
                "leave applied_field unset or recalculate the fields without it"
            )
        vector = np.asarray(applied_field, dtype=float)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise PolarisationError("applied_field must be three finite components in T")
        return vector
    if any(not np.allclose(vector, stored[0], rtol=0.0, atol=1e-12) for vector in stored[1:]):
        raise PolarisationError("the field results carry different applied fields")
    return stored[0]


def _applied_axis(
    results: Sequence[FieldResult],
    rotation: np.ndarray,
    applied_field: Sequence[float] | None,
) -> np.ndarray:
    """The laboratory-frame direction of the applied field."""

    applied = rotation @ _applied_field(results, applied_field)
    size = float(np.linalg.norm(applied))
    if size == 0.0:
        raise PolarisationError("a longitudinal or transverse geometry needs an applied field")
    return applied / size


def _refuse_fixed_options(options: Mapping[str, Any], fixed: Sequence[str], label: str) -> None:
    clashes = sorted(set(options) & set(fixed))
    if clashes:
        raise PolarisationError(f"{label} sets {', '.join(clashes)} itself; leave them out")


def _rotation(rotation: Sequence[Sequence[float]] | np.ndarray | None) -> np.ndarray:
    if rotation is None:
        return np.eye(3)
    matrix = np.asarray(rotation, dtype=float)
    if (
        matrix.shape != (3, 3)
        or not np.allclose(matrix @ matrix.T, np.eye(3), rtol=0.0, atol=_TOLERANCE)
        or np.linalg.det(matrix) <= 0.0
    ):
        raise PolarisationError("crystal_rotation must be a proper 3 x 3 rotation matrix")
    return matrix


def _site_weights(weights: Sequence[float] | None, count: int) -> np.ndarray:
    if weights is None:
        return np.full(count, 1.0 / count)
    values = np.asarray(weights, dtype=float)
    if values.shape != (count,):
        raise PolarisationError(f"site_weights needs one weight per field result ({count})")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)) or values.sum() <= 0.0:
        raise PolarisationError("site_weights must be non-negative with a positive sum")
    return values / values.sum()


def _unit(vector: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise PolarisationError(f"{name} must have three finite components")
    size = float(np.linalg.norm(values))
    if size == 0.0:
        raise PolarisationError(f"{name} must be non-zero")
    return values / size


def _tuple3(values: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    x, y, z = (float(value) for value in values)
    return (x, y, z)
