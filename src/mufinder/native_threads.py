"""Thread-pool caps for the native maths libraries MuFinder loads.

OpenBLAS, MKL and OpenMP start one worker thread per core by default. MuFinder's
linear algebra is small (eigensolves of a few hundred rows, matrix products in
matching, symmetry and phonon analysis), and on a many-core machine whose other
cores are busy, for example with a local CASTEP run, those pools spend far longer
waiting for their threads than computing. The entry point caps them once, before
anything imports NumPy, because the libraries read these variables only when they
load.

The cap is MuFinder's own: :func:`capped_thread_variables` names the values it set,
and the local runner removes them, so CASTEP runs see the user's environment
unchanged. This module uses only the standard library, so the entry point can
import it before the splash screen without loading anything heavy.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

#: Environment variables that size the native maths thread pools.
THREAD_POOL_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

#: Threads per pool when the user's environment does not choose a number.
DEFAULT_THREAD_CAP = 4

# The variables this process set itself, so child processes can drop them.
_capped_variables: tuple[str, ...] = ()


def cap_native_thread_pools(
    environ: MutableMapping[str, str] | None = None,
    *,
    threads: int = DEFAULT_THREAD_CAP,
) -> tuple[str, ...]:
    """Cap each native thread pool the user has not sized; return the names set.

    A variable the user already set to a non-empty value is kept as it is. Call
    this before NumPy, SciPy or anything else that loads BLAS is imported.
    """

    global _capped_variables
    target = os.environ if environ is None else environ
    capped = tuple(name for name in THREAD_POOL_VARIABLES if not target.get(name))
    for name in capped:
        target[name] = str(threads)
    if environ is None:
        _capped_variables = tuple(dict.fromkeys((*_capped_variables, *capped)))
    return capped


def capped_thread_variables() -> tuple[str, ...]:
    """Return the thread-pool variables this process set, not the user."""

    return _capped_variables
