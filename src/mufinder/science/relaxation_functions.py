"""Reference muon-spin relaxation functions.

The standard phenomenological polarisation functions used to fit μSR data, in
one tested place. Every function takes times in μs, widths and rates in μs⁻¹
and fields in tesla, evaluates on an array of times and returns ``G(t)`` with
``G(0) = 1``. :data:`RELAXATION_FUNCTIONS` describes each one: the physical
scenario, its assumptions, its parameters and its references.

The Kubo–Toyabe functions describe a static, isotropic distribution of local
fields. The dynamic Lorentzian Kubo–Toyabe function uses the strong-collision
model of :mod:`mufinder.science.dynamics`, in which the whole field is redrawn
at Poisson times; the Keren and Abragam functions use the Gaussian–Markov
model, in which each field component fluctuates continuously and which holds
when the fluctuations are fast compared with the width.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType

import numpy as np
from scipy.special import wofz

from mufinder.domain.constants import MUON_GAMMA_RAD_S_T
from mufinder.science.polarisation import PolarisationError

# γ_μ in rad μs⁻¹ T⁻¹, so that ω t is in radians for t in μs.
_GAMMA_RAD_US_T = MUON_GAMMA_RAD_S_T * 1e-6
# Below this ω/Δ the longitudinal-field Kubo–Toyabe uses its power series in ω/Δ,
# where the closed form's 1/ω² and 1/ω³ terms cancel badly.
_LF_SERIES_LIMIT = 0.05
# Below this ν/a the dynamic Lorentzian Kubo–Toyabe is the static function: the
# cubic's double root at u = −a splits only as √(ν/a).
_STATIC_RATE_LIMIT = 1e-9
# Below this ν t the motional-narrowing exponent uses its power series.
_MOTIONAL_SERIES_LIMIT = 1e-3

TimesLike = Sequence[float] | np.ndarray


@dataclass(frozen=True, slots=True)
class RelaxationFunction:
    """A reference relaxation function and the description of its use.

    ``parameters`` names the keyword arguments after ``times_us``, with units
    in their names.
    """

    name: str
    function: Callable[..., np.ndarray]
    scenario: str
    assumptions: str
    parameters: tuple[str, ...]
    references: tuple[str, ...]


def static_gaussian_kubo_toyabe(times_us: TimesLike, delta_per_us: float) -> np.ndarray:
    """Zero-field static Gaussian Kubo–Toyabe ``⅓ + ⅔(1 − Δ²t²) e^{−Δ²t²/2}``."""

    times = _times(times_us)
    delta = _non_negative(delta_per_us, "delta_per_us")
    x2 = (delta * times) ** 2
    return 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - x2) * np.exp(-x2 / 2.0)


def static_gaussian_kubo_toyabe_lf(
    times_us: TimesLike, delta_per_us: float, field_T: float
) -> np.ndarray:
    """Static Gaussian Kubo–Toyabe in a longitudinal field ``B`` (``ω = γ_μ|B|``).

    ``G = 1 − (2Δ²/ω²)[1 − e^{−Δ²t²/2} cos ωt]
    + (2Δ⁴/ω³) ∫₀ᵗ e^{−Δ²τ²/2} sin ωτ dτ``, with the integral in closed form
    through the Faddeeva function.
    """

    times = _times(times_us)
    delta = _non_negative(delta_per_us, "delta_per_us")
    field = _finite(field_T, "field_T")
    omega = _GAMMA_RAD_US_T * abs(field)
    if delta == 0.0:
        return np.ones_like(times)
    ratio = omega / delta
    x = delta * times
    if ratio < _LF_SERIES_LIMIT:
        return _lf_kubo_toyabe_series(x, ratio)
    # ∫₀ᵗ e^{−Δ²τ²/2} sin ωτ dτ = √(π/2)/Δ · Im[w(b) − e^{−X² + 2ibX} w(b + iX)],
    # X = Δt/√2, b = ω/(√2 Δ); w(b + iX) stays bounded for X ≥ 0.
    big_x = x / math.sqrt(2.0)
    b = ratio / math.sqrt(2.0)
    faddeeva = wofz(b) - np.exp(-(big_x**2) + 2j * b * big_x) * wofz(b + 1j * big_x)
    integral = math.sqrt(math.pi / 2.0) / delta * faddeeva.imag
    gaussian = np.exp(-(x**2) / 2.0)
    return (
        1.0
        - 2.0 / ratio**2 * (1.0 - gaussian * np.cos(omega * times))
        + 2.0 * delta**4 / omega**3 * integral
    )


def static_lorentzian_kubo_toyabe(times_us: TimesLike, a_per_us: float) -> np.ndarray:
    """Zero-field static Lorentzian Kubo–Toyabe ``⅓ + ⅔(1 − at) e^{−at}``."""

    times = _times(times_us)
    a = _non_negative(a_per_us, "a_per_us")
    return 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - a * times) * np.exp(-a * times)


def dynamic_lorentzian_kubo_toyabe(
    times_us: TimesLike, a_per_us: float, collision_rate_per_us: float
) -> np.ndarray:
    """Zero-field Lorentzian Kubo–Toyabe with strong collisions at rate ``ν``, exactly.

    The static Laplace transform ``g̃(u) = 1/(3u) + (2/3) u/(u + a)²`` is
    rational, so ``G̃(s) = g̃(s + ν)/(1 − ν g̃(s + ν))`` is rational with a
    cubic denominator and ``G(t)`` is a sum of three exponentials from its
    roots.
    """

    times = _times(times_us)
    a = _non_negative(a_per_us, "a_per_us")
    nu = _non_negative(collision_rate_per_us, "collision_rate_per_us")
    if a == 0.0:
        return np.ones_like(times)
    if nu <= _STATIC_RATE_LIMIT * a:
        return static_lorentzian_kubo_toyabe(times, a)
    # With u = s + ν, G̃ = N(u)/D(u): N = 3u² + 2au + a²,
    # D = 3u(u + a)² − νN = 3u³ + (6a − 3ν)u² + (3a² − 2aν)u − νa².
    roots = _cubic_roots(2.0 * a - nu, a * a - 2.0 * a * nu / 3.0, -nu * a * a / 3.0)
    numerator = 3.0 * roots**2 + 2.0 * a * roots + a * a
    derivative = 9.0 * roots**2 + 2.0 * (6.0 * a - 3.0 * nu) * roots + 3.0 * a * a - 2.0 * a * nu
    residues = numerator / derivative
    exponents = np.exp(np.outer(times, roots - nu))
    return (exponents @ residues).real


def keren_lf_relaxation(
    times_us: TimesLike, delta_per_us: float, collision_rate_per_us: float, field_T: float
) -> np.ndarray:
    """Keren's Gaussian–Markov relaxation ``exp(−Γ(t) t)`` in a longitudinal field.

    ``Γ(t) t = 2Δ²/(ω² + ν²)² [(ω² + ν²)νt + (ω² − ν²)(1 − e^{−νt} cos ωt)
    − 2νω e^{−νt} sin ωt]`` with ``ω = γ_μ|B|``. It reduces to the Abragam
    function in zero field.
    """

    times = _times(times_us)
    delta = _non_negative(delta_per_us, "delta_per_us")
    nu = _non_negative(collision_rate_per_us, "collision_rate_per_us")
    field = _finite(field_T, "field_T")
    omega = _GAMMA_RAD_US_T * abs(field)
    if omega == 0.0:
        return abragam_relaxation(times, delta, nu)
    total = omega**2 + nu**2
    decay = np.exp(-nu * times)
    exponent = (
        2.0
        * delta**2
        / total**2
        * (
            total * nu * times
            + (omega**2 - nu**2) * (1.0 - decay * np.cos(omega * times))
            - 2.0 * nu * omega * decay * np.sin(omega * times)
        )
    )
    return np.exp(-exponent)


def abragam_relaxation(
    times_us: TimesLike, delta_per_us: float, collision_rate_per_us: float
) -> np.ndarray:
    """Abragam's zero-field motional-narrowing function
    ``exp(−2Δ²/ν² (e^{−νt} − 1 + νt))``.

    It is Gaussian, ``e^{−Δ²t²}``, for ``νt ≪ 1`` and exponential,
    ``e^{−2Δ²t/ν}``, for ``νt ≫ 1``.
    """

    times = _times(times_us)
    delta = _non_negative(delta_per_us, "delta_per_us")
    nu = _non_negative(collision_rate_per_us, "collision_rate_per_us")
    return np.exp(-2.0 * (delta * times) ** 2 * _motional_factor(nu * times))


_KUBO_TOYABE = (
    "Kubo and Toyabe, in Magnetic Resonance and Relaxation, ed. Blinc (North-Holland, 1967)"
)
_HAYANO = "Hayano et al., Phys. Rev. B 20, 850 (1979)"
_UEMURA = "Uemura et al., Phys. Rev. B 31, 546 (1985)"
_KEREN = "Keren, Phys. Rev. B 50, 10039 (1994)"
_ABRAGAM = "Abragam, The Principles of Nuclear Magnetism (Oxford, 1961), Ch. X"

RELAXATION_FUNCTIONS: Mapping[str, RelaxationFunction] = MappingProxyType(
    {
        entry.name: entry
        for entry in (
            RelaxationFunction(
                name="static_gaussian_kubo_toyabe",
                function=static_gaussian_kubo_toyabe,
                scenario="Zero field; a dense array of static moments, typically nuclear dipoles.",
                assumptions="Static, isotropic Gaussian field distribution of width Δ/γ_μ per "
                "component; powder or isotropic site.",
                parameters=("delta_per_us",),
                references=(_KUBO_TOYABE, _HAYANO),
            ),
            RelaxationFunction(
                name="static_gaussian_kubo_toyabe_lf",
                function=static_gaussian_kubo_toyabe_lf,
                scenario="Longitudinal field B along the initial polarisation; decoupling of "
                "a static Gaussian distribution.",
                assumptions="As the zero-field Gaussian Kubo–Toyabe, plus a fixed applied field "
                "along the initial muon spin.",
                parameters=("delta_per_us", "field_T"),
                references=(_HAYANO,),
            ),
            RelaxationFunction(
                name="static_lorentzian_kubo_toyabe",
                function=static_lorentzian_kubo_toyabe,
                scenario="Zero field; dilute static moments, as in spin glasses.",
                assumptions="Static, isotropic Lorentzian field distribution of half-width "
                "a/γ_μ per component.",
                parameters=("a_per_us",),
                references=(_UEMURA,),
            ),
            RelaxationFunction(
                name="dynamic_lorentzian_kubo_toyabe",
                function=dynamic_lorentzian_kubo_toyabe,
                scenario="Zero field; dilute moments whose field is redrawn at rate ν.",
                assumptions="Strong-collision model: the whole field is redrawn from the "
                "static Lorentzian distribution at Poisson times, with no memory.",
                parameters=("a_per_us", "collision_rate_per_us"),
                references=(_UEMURA, _HAYANO),
            ),
            RelaxationFunction(
                name="keren_lf_relaxation",
                function=keren_lf_relaxation,
                scenario="Longitudinal field; fluctuating Gaussian fields, fast enough that the "
                "Kubo–Toyabe dip is gone.",
                assumptions="Gaussian–Markov model: each field component fluctuates with "
                "correlation e^{−νt}; valid for ν ≳ Δ or ω ≫ Δ.",
                parameters=("delta_per_us", "collision_rate_per_us", "field_T"),
                references=(_KEREN,),
            ),
            RelaxationFunction(
                name="abragam_relaxation",
                function=abragam_relaxation,
                scenario="Zero field; motional narrowing of a Gaussian width by fluctuations "
                "or muon diffusion.",
                assumptions="Gaussian–Markov model with correlation e^{−νt}; valid for ν ≳ Δ. "
                "It has no 1/3 tail.",
                parameters=("delta_per_us", "collision_rate_per_us"),
                references=(_ABRAGAM,),
            ),
        )
    }
)


def _lf_kubo_toyabe_series(x: np.ndarray, ratio: float) -> np.ndarray:
    # G(x, r) = Σ_k c_k(x) r^k with x = Δt, r = ω/Δ, through r⁸.
    x2 = x * x
    gaussian = np.exp(-x2 / 2.0)
    tail = -np.expm1(-x2 / 2.0)  # 1 − e^{−x²/2}
    r2 = ratio * ratio
    c0 = 1.0 / 3.0 + 2.0 / 3.0 * (1.0 - x2) * gaussian
    c2 = ((x2 * x2 - x2) * gaussian + 2.0 * tail) / 15.0
    c4 = ((-(x2**3) + x2**2 + 4.0 * x2) * gaussian - 8.0 * tail) / 420.0
    c6 = ((x2**4 - x2**3 - 6.0 * x2**2 - 24.0 * x2) * gaussian + 48.0 * tail) / 22680.0
    c8 = (
        (-(x2**5) + x2**4 + 8.0 * x2**3 + 48.0 * x2**2 + 192.0 * x2) * gaussian - 384.0 * tail
    ) / 1995840.0
    return c0 + r2 * (c2 + r2 * (c4 + r2 * (c6 + r2 * c8)))


def _motional_factor(x: np.ndarray) -> np.ndarray:
    # (e^{−x} − 1 + x)/x², which is ½ at x = 0.
    small = x < _MOTIONAL_SERIES_LIMIT
    safe = np.where(small, 1.0, x)
    exact = (np.expm1(-safe) + safe) / safe**2
    series = 0.5 - x / 6.0 + x**2 / 24.0 - x**3 / 120.0 + x**4 / 720.0
    return np.where(small, series, exact)


def _cubic_roots(b: float, c: float, d: float) -> np.ndarray:
    """The roots of the monic cubic ``u³ + bu² + cu + d``, from the explicit formula."""

    shift = b / 3.0
    p = c - b * b / 3.0
    q = 2.0 * b**3 / 27.0 - b * c / 3.0 + d
    discriminant = (q / 2.0) ** 2 + (p / 3.0) ** 3
    if discriminant < 0.0:
        # Three real roots (p < 0): the trigonometric form.
        radius = 2.0 * math.sqrt(-p / 3.0)
        angle = math.acos(max(-1.0, min(1.0, 3.0 * q / (p * radius)))) / 3.0
        ys = radius * np.cos(angle - 2.0 * math.pi * np.arange(3) / 3.0)
        return ys.astype(complex) - shift
    # One real root and a complex pair: Cardano, with the second cube root
    # taken as −p/(3A) to avoid cancellation.
    root = math.sqrt(discriminant)
    first = float(np.cbrt(-q / 2.0 + root if q <= 0.0 else -q / 2.0 - root))
    second = -p / (3.0 * first) if first != 0.0 else 0.0
    real = first + second
    pair = complex(-real / 2.0, math.sqrt(3.0) / 2.0 * (first - second))
    return np.array([real, pair, pair.conjugate()]) - shift


def _times(times_us: TimesLike) -> np.ndarray:
    times = np.asarray(times_us, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise PolarisationError("times_us must be finite and non-negative")
    return times


def _non_negative(value: float, name: str) -> float:
    number = _finite(value, name)
    if number < 0.0:
        raise PolarisationError(f"{name} must be non-negative")
    return number


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise PolarisationError(f"{name} must be finite")
    return number
