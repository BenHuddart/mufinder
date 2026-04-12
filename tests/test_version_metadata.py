"""Version metadata consistency checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re

from mufinder import __version__
from mufinder._version import current_version
from mufinder.domain.models import Project

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_runtime_version_matches_pyproject() -> None:
    expected = _pyproject_version()

    assert current_version() == expected
    assert __version__ == expected
    assert Project().app_version == expected


def test_docs_release_matches_pyproject() -> None:
    conf_path = ROOT / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("mufinder_docs_conf", conf_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.release == _pyproject_version()
