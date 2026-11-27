"""The demagnetizing tensor of a voxel shape by the Fourier method.

Needs only the core install.

    python docs/api/examples/demagnetizing_tensor.py
"""

import numpy as np

from mufinder.science.demagnetization import (
    DemagnetizingSample,
    demagnetizing_factors_rectangular_prism,
    fourier_demagnetizing_result,
)

# A 4 x 4 x 1 square plate as 32 x 32 x 4 voxels of 0.125 x 0.125 x 0.25.
plate = DemagnetizingSample.voxel_grid(np.ones((32, 32, 4), dtype=bool), (0.125, 0.125, 0.25))
result = fourier_demagnetizing_result(plate)

print("Fourier N:", np.round(np.diag(result.tensor), 4))
print("Aharoni N:", np.round(demagnetizing_factors_rectangular_prism((4.0, 4.0, 1.0)), 4))
print(f"Tr N: {result.trace:.6f}, FFT box: {result.box_shape}")
