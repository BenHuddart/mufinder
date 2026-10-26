"""Dynamic muon-spin polarisation in the strong-collision model.

In the strong-collision model the whole local field is redrawn from the static
field distribution at Poisson times with rate ``ν``, and the field after a
collision keeps no memory of the field before it. Between collisions the spin
evolves as it does in the static distribution, so the dynamic polarisation
follows from the static one ``G⁽⁰⁾(t)`` alone:

    G(t) = e^{−νt} G⁽⁰⁾(t) + ν ∫₀ᵗ e^{−νt'} G⁽⁰⁾(t') G(t − t') dt'.

The model describes muon hopping between sites and fast paramagnetic
fluctuations, and it keeps a fixed applied field because every sample of the
static distribution contains it. It is not the Gaussian–Markov model behind
the Keren and Abragam formulas in :mod:`mufinder.science.relaxation_functions`,
in which each field component fluctuates continuously; the two agree closely
for Gaussian distributions.

On a uniform grid ``t_n = nτ`` with ``q = e^{−ντ}``, ``λ = 1 − q`` and
``H_n = qⁿ G⁽⁰⁾_n``, a collision in each step with probability ``λ`` gives

    G_n = H_n + (λ/q) Σ_{k=1}^{n} H_k G_{n−k}

(Allodi and De Renzi, Phys. Scr. 89, 115201 (2014), Eq. 13a), which the
``exact`` method evaluates directly. The ``fft`` method solves the same
recursion by a damped discrete Fourier transform (their Eq. 20).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np

from mufinder.science.polarisation import PolarisationError

DynamicsMethod = Literal["exact", "fft"]

# Damping exponent α N of the fft method (Allodi and De Renzi, Table 1).
DEFAULT_ALPHA_N = 10.0
# Largest damping exponent α N the fft method accepts; e^{αN} must not overflow.
_MAX_ALPHA_N = 700.0
# Relative tolerance on the grid step for a "uniform" grid.
_GRID_TOLERANCE = 1e-6


@dataclass(frozen=True, slots=True)
class DynamicPolarisationResult:
    """A static polarisation trace and its strong-collision dynamic trace.

    ``time_step_us`` is the grid step ``τ``. ``alpha`` is the damping exponent
    per step of the ``fft`` method and ``None`` for ``exact``.
    """

    times_us: np.ndarray
    static_polarisation: np.ndarray
    polarisation: np.ndarray
    collision_rate_per_us: float
    time_step_us: float
    method: DynamicsMethod
    alpha: float | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "times_us": np.asarray(self.times_us, dtype=float).tolist(),
            "static_polarisation": np.asarray(self.static_polarisation, dtype=float).tolist(),
            "polarisation": np.asarray(self.polarisation, dtype=float).tolist(),
            "collision_rate_per_us": float(self.collision_rate_per_us),
            "time_step_us": float(self.time_step_us),
            "method": self.method,
            "alpha": None if self.alpha is None else float(self.alpha),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> DynamicPolarisationResult:
        try:
            times = np.asarray(payload["times_us"], dtype=float)
            static = np.asarray(payload["static_polarisation"], dtype=float)
            dynamic = np.asarray(payload["polarisation"], dtype=float)
            if static.shape != times.shape or dynamic.shape != times.shape:
                raise ValueError("traces do not match the time grid")
            method = _method(payload["method"])
            alpha = payload.get("alpha")
            return cls(
                times_us=times,
                static_polarisation=static,
                polarisation=dynamic,
                collision_rate_per_us=float(payload["collision_rate_per_us"]),
                time_step_us=float(payload["time_step_us"]),
                method=method,
                alpha=None if alpha is None else float(alpha),
            )
        except (KeyError, TypeError, ValueError, PolarisationError) as exc:
            raise PolarisationError(f"malformed dynamic polarisation result: {exc}") from exc


def calculate_dynamic_polarisation(
    static_polarisation: Sequence[float] | np.ndarray | Callable[[np.ndarray], np.ndarray],
    times_us: Sequence[float] | np.ndarray,
    collision_rate_per_us: float,
    *,
    method: DynamicsMethod = "exact",
    alpha: float | None = None,
) -> DynamicPolarisationResult:
    """Return the strong-collision dynamic polarisation of a static trace.

    ``static_polarisation`` is the static ``G⁽⁰⁾`` on ``times_us`` or a
    callable that evaluates it there. ``times_us`` must be uniform and start
    at zero. ``collision_rate_per_us`` is ``ν``; zero returns the static trace.

    ``method="exact"`` (the default) evaluates the discrete recursion in
    ``O(N²)`` with one dot product per step; it has no tuning parameter and no
    wrap-around error. ``method="fft"`` costs ``O(N log N)`` and suits very
    long grids; ``alpha`` is its damping exponent per step, ``10/N`` by default.
    """

    times = np.asarray(times_us, dtype=float)
    step = _grid_step(times)
    static = _static_trace(static_polarisation, times)
    rate = float(collision_rate_per_us)
    if not math.isfinite(rate) or rate < 0.0:
        raise PolarisationError("collision_rate_per_us must be finite and non-negative")
    method = _method(method)
    if method == "exact" and alpha is not None:
        raise PolarisationError("alpha applies only to method='fft'")
    if method == "fft":
        alpha = DEFAULT_ALPHA_N / times.size if alpha is None else float(alpha)
        if not math.isfinite(alpha) or alpha < 0.0:
            raise PolarisationError("alpha must be finite and non-negative")
        if alpha * times.size > _MAX_ALPHA_N:
            raise PolarisationError(
                f"alpha × N must not exceed {_MAX_ALPHA_N:g}; the unweighting e^(alpha N) would overflow"
            )

    if rate == 0.0:
        dynamic = static.copy()
    elif method == "exact":
        dynamic = _exact(static, rate * step)
    else:
        assert alpha is not None
        dynamic = _fft(static, rate * step, alpha)
    return DynamicPolarisationResult(
        times_us=times,
        static_polarisation=static,
        polarisation=dynamic,
        collision_rate_per_us=rate,
        time_step_us=step,
        method=method,
        alpha=alpha,
    )


def _exact(static: np.ndarray, rate_step: float) -> np.ndarray:
    damped, kernel = _recursion_terms(static, rate_step)
    dynamic = np.empty_like(damped)
    dynamic[0] = damped[0]
    for n in range(1, damped.size):
        dynamic[n] = damped[n] + np.dot(kernel[1 : n + 1], dynamic[n - 1 :: -1])
    return dynamic


def _fft(static: np.ndarray, rate_step: float, alpha: float) -> np.ndarray:
    # Weighting by e^{−αn} keeps the convolution and damps the wrap-around.
    count = static.size
    steps = np.arange(count)
    damped, kernel = _recursion_terms(static, rate_step)
    weight = np.exp(-alpha * steps)
    # G = H + K ⊛ G, so G = H / (1 − K) in the transform; this is q H / (1 − λ H)
    # when G⁽⁰⁾(0) = 1.
    solved = np.fft.rfft(damped * weight, 2 * count) / (
        1.0 - np.fft.rfft(kernel * weight, 2 * count)
    )
    return np.fft.irfft(solved, 2 * count)[:count] / weight


def _recursion_terms(static: np.ndarray, rate_step: float) -> tuple[np.ndarray, np.ndarray]:
    """``H_n = qⁿ G⁽⁰⁾_n`` and the kernel ``K_k = (λ/q) H_k = λ q^{k−1} G⁽⁰⁾_k`` (``K_0 = 0``).

    Written with ``q^{k−1}`` rather than ``λ/q = e^{ντ} − 1``, so a collision in
    every step (``ντ`` beyond the float range of ``e^{ντ}``) stays finite.
    """

    q = math.exp(-rate_step)
    collision = -math.expm1(-rate_step)  # λ = 1 − q
    steps = np.arange(static.size)
    damped = static * q**steps
    kernel = np.zeros_like(damped)
    kernel[1:] = collision * q ** (steps[1:] - 1) * static[1:]
    return damped, kernel


def _grid_step(times: np.ndarray) -> float:
    if times.ndim != 1 or times.size < 2:
        raise PolarisationError("times_us must be a one-dimensional grid of at least two times")
    if not np.all(np.isfinite(times)):
        raise PolarisationError("times_us must be finite")
    step = float(times[1] - times[0])
    if step <= 0.0:
        raise PolarisationError("times_us must increase")
    if abs(times[0]) > _GRID_TOLERANCE * step:
        raise PolarisationError("times_us must start at zero")
    if not np.allclose(np.diff(times), step, rtol=_GRID_TOLERANCE, atol=0.0):
        raise PolarisationError("times_us must be uniformly spaced")
    return step


def _static_trace(
    static: Sequence[float] | np.ndarray | Callable[[np.ndarray], np.ndarray],
    times: np.ndarray,
) -> np.ndarray:
    values = static(times.copy()) if callable(static) else static
    trace = np.asarray(values, dtype=float)
    if trace.shape != times.shape:
        raise PolarisationError(
            f"static polarisation has shape {trace.shape}; the time grid has {times.shape}"
        )
    if not np.all(np.isfinite(trace)):
        raise PolarisationError("static polarisation must be finite")
    return trace


def _method(method: object) -> DynamicsMethod:
    if method == "exact":
        return "exact"
    if method == "fft":
        return "fft"
    raise PolarisationError(f"unknown method {method!r}; use 'exact' or 'fft'")
