"""Strong-collision dynamics of a Gaussian Kubo-Toyabe function.

Needs only the core install.

    python docs/api/examples/dynamics.py
"""

import numpy as np

from mufinder.science.dynamics import calculate_dynamic_polarisation
from mufinder.science.relaxation_functions import (
    RELAXATION_FUNCTIONS,
    abragam_relaxation,
    static_gaussian_kubo_toyabe,
)

delta = 0.5  # nuclear width Delta, in us^-1
times = np.linspace(0.0, 20.0, 4001)  # uniform, from zero


def static(t):
    return static_gaussian_kubo_toyabe(t, delta)


# Slow fluctuations wash out the 1/3 tail; fast ones narrow the decay.
for rate in (0.0, 0.2, 5.0):
    dynamic = calculate_dynamic_polarisation(static, times, rate)
    print(f"nu = {rate:3.1f} us^-1: G(10 us) = {dynamic.polarisation[2000]:.3f}")

# At nu >> Delta the strong-collision result approaches Abragam's formula.
fast = calculate_dynamic_polarisation(static, times, 5.0)
gap = np.max(np.abs(fast.polarisation - abragam_relaxation(times, delta, 5.0)))
print(f"Largest difference from Abragam at nu = 5 us^-1: {gap:.4f}")

print("Abragam assumes:", RELAXATION_FUNCTIONS["abragam_relaxation"].assumptions)
