"""Compatibility helpers for the muesr calculation backend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import importlib
import sys
from types import ModuleType
import warnings

import numpy as np

_CGI_DEPRECATION_MESSAGE = "'cgi' is deprecated and slated for removal in Python 3.13"


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


@contextmanager
def muesr_imports() -> Iterator[None]:
    """Make ``muesr`` importable on every supported Python version.

    Import muesr modules inside this context. It applies the NumPy alias patch,
    silences the ``cgi`` deprecation warning and, where ``cgi`` no longer exists
    (Python 3.13 removed it under PEP 594), provides an empty ``cgi`` module for
    the duration of the import. muesr's vendored ``muesr.core.prettytable``
    imports ``cgi`` at module level but only calls ``cgi.escape`` from its HTML
    table renderer, which MuFinder never uses and which Python 3.8 already broke
    by removing ``cgi.escape``. The stand-in is removed again on exit, so code
    outside muesr still sees ``cgi`` as missing.
    """
    patch_numpy_aliases_for_muesr()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_CGI_DEPRECATION_MESSAGE, category=DeprecationWarning)
        if _cgi_importable():
            yield
            return
        missing = object()
        previous = sys.modules.get("cgi", missing)
        stand_in = ModuleType("cgi", "Empty stand-in for the removed cgi module, installed for muesr imports.")
        sys.modules["cgi"] = stand_in
        try:
            yield
        finally:
            if sys.modules.get("cgi") is stand_in:
                if previous is missing:
                    del sys.modules["cgi"]
                else:
                    sys.modules["cgi"] = previous


def _cgi_importable() -> bool:
    try:
        importlib.import_module("cgi")
    except ImportError:
        return False
    return True
