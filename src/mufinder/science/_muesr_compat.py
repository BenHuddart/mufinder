"""Compatibility helpers for the muesr calculation backend."""

from __future__ import annotations

import numpy as np


def patch_numpy_aliases_for_muesr() -> None:
    """Restore NumPy aliases still used by current muesr releases.

    NumPy removed aliases such as ``np.complex`` and ``np.float`` after a long
    deprecation period. muesr still uses them internally when constructing
    magnetic models and local-field objects. This narrow, idempotent patch keeps
    that compatibility workaround at the muesr boundary.
    """
    if "complex" not in np.__dict__:
        setattr(np, "complex", complex)
    if "float" not in np.__dict__:
        setattr(np, "float", float)
