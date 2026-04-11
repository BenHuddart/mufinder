"""Tests for comprehensive CASTEP schema coverage and alias normalization."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mufinder.assets import starter_keyword_catalog_path
import mufinder.castep_io.schema as schema_module
from mufinder.castep_io.schema import (
    CatalogSourceInfo,
    SOURCE_KEYWORDS,
    canonical_keyword_name,
    get_keyword_spec,
    keyword_catalog_source_info,
    refresh_keyword_catalog,
)


def test_repo_does_not_vendor_castep_keyword_json_copy() -> None:
    assert not Path("src/mufinder/castep_io/base_catalog.json").exists()


def test_packaged_starter_catalog_matches_docs_copy() -> None:
    docs_catalog = Path("docs/castep/keyword_catalog.yaml")
    packaged_catalog = starter_keyword_catalog_path()

    assert packaged_catalog.name == "keyword_catalog.yaml"
    assert packaged_catalog.read_text(encoding="utf-8") == docs_catalog.read_text(encoding="utf-8")


def test_runtime_schema_covers_every_keyword_from_source_catalog() -> None:
    for file_kind, source_names in SOURCE_KEYWORDS.items():
        missing = [name for name in source_names if get_keyword_spec(file_kind, name) is None]
        assert missing == []


def test_aliases_canonicalize_to_single_runtime_keyword() -> None:
    assert canonical_keyword_name("param", "spin_polarised") == "spin_polarized"
    assert canonical_keyword_name("param", "calculate_polarisation") == "calculate_polarization"
    assert canonical_keyword_name("cell", "kpoints_mp_grid") == "kpoint_mp_grid"

    cell_spec = get_keyword_spec("cell", "kpoints_mp_grid")
    assert cell_spec is not None
    assert cell_spec.name == "kpoint_mp_grid"


def test_refresh_keyword_catalog_uses_updated_env_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_mufinder_env = os.environ.get("MUFINDER_CASTEP_KEYWORDS_JSON")
    original_castep_env = os.environ.get("CASTEP_KEYWORDS_JSON")

    custom_catalog = tmp_path / "custom_keywords.json"
    custom_catalog.write_text(
        (
            "{\n"
            '  "castep_version": "test",\n'
            '  "cell": {"abc": {"option_type": "Text", "level": "Basic", "docstring": "Example"}},\n'
            '  "param": {"task": {"option_type": "Text", "level": "Basic", "docstring": "Example"}}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", str(custom_catalog))
    refresh_keyword_catalog()

    source = keyword_catalog_source_info()
    assert source is not None
    assert source.origin == "env_json:MUFINDER_CASTEP_KEYWORDS_JSON"
    assert source.path == custom_catalog

    if original_mufinder_env is None:
        monkeypatch.delenv("MUFINDER_CASTEP_KEYWORDS_JSON", raising=False)
    else:
        monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", original_mufinder_env)
    if original_castep_env is None:
        monkeypatch.delenv("CASTEP_KEYWORDS_JSON", raising=False)
    else:
        monkeypatch.setenv("CASTEP_KEYWORDS_JSON", original_castep_env)
    refresh_keyword_catalog()


def test_refresh_keyword_catalog_falls_back_to_bundled_starter_catalog_when_missing(
    tmp_path: Path,
) -> None:
    original_candidate_sources = schema_module._candidate_catalog_sources

    try:
        schema_module._candidate_catalog_sources = lambda: (
            CatalogSourceInfo(path=tmp_path / "missing.json", origin="test_missing"),
        )

        source = refresh_keyword_catalog(allow_fallback=True)

        assert source is not None
        assert source.origin == "bundled_starter_catalog"
        assert source.path.name == "keyword_catalog.yaml"
        assert get_keyword_spec("param", "task") is not None
        assert get_keyword_spec("cell", "kpoints_mp_grid") is not None
    finally:
        schema_module._candidate_catalog_sources = original_candidate_sources
        refresh_keyword_catalog(allow_fallback=True)


def test_refresh_keyword_catalog_stays_strict_without_fallback_when_missing(
    tmp_path: Path,
) -> None:
    original_candidate_sources = schema_module._candidate_catalog_sources

    try:
        schema_module._candidate_catalog_sources = lambda: (
            CatalogSourceInfo(path=tmp_path / "missing.json", origin="test_missing"),
        )

        with pytest.raises(FileNotFoundError, match="requires a local CASTEP metadata file"):
            refresh_keyword_catalog()
    finally:
        schema_module._candidate_catalog_sources = original_candidate_sources
        refresh_keyword_catalog(allow_fallback=True)
