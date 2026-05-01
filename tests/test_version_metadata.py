"""Version metadata consistency checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

import mufinder._version as version_module
from mufinder import __version__
from mufinder._version import current_version
from mufinder.domain.models import Project

ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None
    return match.group(1)


def _pyproject_requires_python() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^requires-python = "([^"]+)"$', text, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_runtime_version_matches_pyproject() -> None:
    expected = _pyproject_version()

    assert current_version() == expected
    assert __version__ == expected
    assert Project().app_version == expected


def test_python_requirement_matches_supported_syntax() -> None:
    assert _pyproject_requires_python() == ">=3.12"
    assert sys.version_info >= (3, 12)


def test_github_workflows_target_supported_python_version() -> None:
    workflow_dir = ROOT / ".github" / "workflows"
    workflow_texts = [path.read_text(encoding="utf-8") for path in workflow_dir.glob("*.yml")]
    versions = {
        match.group(1)
        for text in workflow_texts
        for match in re.finditer(r'python-version:\s*"([^"]+)"', text)
    }

    assert versions
    assert versions == {"3.12"}


def test_current_version_prefers_source_tree_pyproject(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "9.9.9"\n', encoding="utf-8")

    version_module.current_version.cache_clear()
    monkeypatch.setattr(version_module, "_PYPROJECT", pyproject)
    monkeypatch.setattr(version_module, "installed_version", lambda _name: "0.1.2")

    try:
        assert current_version() == "9.9.9"
    finally:
        version_module.current_version.cache_clear()


def test_docs_release_matches_pyproject() -> None:
    conf_path = ROOT / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("mufinder_docs_conf", conf_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.release == _pyproject_version()
