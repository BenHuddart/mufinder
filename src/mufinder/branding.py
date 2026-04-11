"""Helpers for MuFinder branding assets."""

from __future__ import annotations

from importlib.resources import files
import sys

from PySide6.QtGui import QIcon


def logo_path() -> str:
    """Return the packaged MuFinder logo path."""

    return str(files("mufinder.assets").joinpath("mufinder_logo.png"))


def windows_icon_path() -> str:
    """Return the packaged Windows ICO asset path."""

    return str(files("mufinder.assets").joinpath("mufinder_logo.ico"))


def app_icon() -> QIcon:
    """Return the packaged MuFinder application icon."""

    return QIcon(windows_icon_path() if sys.platform.startswith("win") else logo_path())
