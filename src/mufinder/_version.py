"""Helpers for resolving the MuFinder package version."""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as installed_version
from pathlib import Path
import re

_VERSION_RE = re.compile(r'^version = "([^"]+)"$', re.MULTILINE)
_ROOT = Path(__file__).resolve().parents[2]
_PYPROJECT = _ROOT / "pyproject.toml"


def _read_pyproject_version() -> str | None:
    """Return the source-tree version when running from a checkout."""

    if not _PYPROJECT.is_file():
        return None
    match = _VERSION_RE.search(_PYPROJECT.read_text(encoding="utf-8"))
    if match:
        return match.group(1)
    return None


@lru_cache(maxsize=1)
def current_version() -> str:
    """Return the installed package version, falling back to pyproject metadata."""

    if (source_version := _read_pyproject_version()) is not None:
        return source_version

    try:
        return installed_version("mufinder")
    except PackageNotFoundError:
        return "0.0.0"
