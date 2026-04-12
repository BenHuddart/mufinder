"""Helpers for resolving the MuFinder package version."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
import re

_VERSION_RE = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)
_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _ROOT / "pyproject.toml"


@lru_cache(maxsize=1)
def current_version() -> str:
    """Return the installed package version, falling back to pyproject metadata."""

    try:
        return installed_version("mufinder")
    except PackageNotFoundError:
        if _PYPROJECT.is_file():
            match = _VERSION_RE.search(_PYPROJECT.read_text(encoding="utf-8"))
            if match:
                return match.group(1)
        return "0.0.0"
