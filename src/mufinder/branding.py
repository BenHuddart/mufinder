"""Helpers for MuFinder branding assets."""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtGui import QIcon


def logo_path() -> str:
    """Return the packaged MuFinder logo path."""

    return str(files("mufinder.assets").joinpath("mufinder_logo.png"))


def app_icon() -> QIcon:
    """Return the packaged MuFinder application icon."""

    return QIcon(logo_path())

