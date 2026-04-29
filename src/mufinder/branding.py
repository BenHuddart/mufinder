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


def project_home_landmark_path(name: str) -> str:
    """Return a packaged project-home landmark sprite path."""

    safe_name = name.strip().replace("/", "_").replace("\\", "_")
    return str(files("mufinder.assets").joinpath(f"project_home_{safe_name}.png"))


def project_home_map_path() -> str:
    """Return the packaged muted project-home map artwork path."""

    return str(files("mufinder.assets").joinpath("project_home_map.png"))


def project_home_color_map_path() -> str:
    """Return the packaged color project-home map artwork path."""

    return str(files("mufinder.assets").joinpath("project_home_map_color.png"))


def app_icon() -> QIcon:
    """Return the packaged MuFinder application icon."""

    return QIcon(windows_icon_path() if sys.platform.startswith("win") else logo_path())
