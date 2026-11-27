"""Bundled branding and other packaged runtime assets for MuFinder."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def starter_keyword_catalog_path() -> Path:
    """Return the packaged starter CASTEP catalog used when local metadata is unavailable."""

    return Path(str(files("mufinder.assets").joinpath("keyword_catalog.yaml")))
