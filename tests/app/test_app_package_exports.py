"""Tests for lazy exports from mufinder.app package."""

from __future__ import annotations

import pytest

import mufinder.app as app_package
from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.app.math_label import MathLabel


def test_package_getattr_loads_and_caches_export() -> None:
    app_package.__dict__.pop("ConvergenceConfig", None)

    loaded = app_package.__getattr__("ConvergenceConfig")

    assert loaded is ConvergenceConfig
    assert app_package.ConvergenceConfig is ConvergenceConfig


def test_package_getattr_loads_math_label_export() -> None:
    app_package.__dict__.pop("MathLabel", None)

    loaded = app_package.__getattr__("MathLabel")

    assert loaded is MathLabel
    assert app_package.MathLabel is MathLabel


def test_package_getattr_raises_for_unknown_export() -> None:
    with pytest.raises(AttributeError):
        app_package.__getattr__("NotARealExport")
