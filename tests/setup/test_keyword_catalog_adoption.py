"""Startup adopts a keyword catalog saved at a user-chosen path into MuFinder's own copy."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
from PySide6.QtCore import QSettings

import mufinder.castep_io.schema as schema_module
import mufinder.castep_setup as castep_setup_module
from mufinder.castep_io.schema import SETUP_KEYWORDS_JSON_ENV_VAR, keyword_catalog_status
from mufinder.castep_setup import (
    CASTEP_EXECUTABLE_KEY,
    CASTEP_KEYWORDS_JSON_KEY,
    apply_saved_castep_environment,
    load_castep_setup,
    managed_keywords_json_path,
)
from tests.support.keyword_catalogs import isolate_catalog_sources, write_complete_catalog


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> QSettings:
    """Saved settings on a machine with no other catalog source, as at a fresh start."""

    isolate_catalog_sources(monkeypatch)
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def _legacy_setup(settings: QSettings, legacy: Path) -> None:
    settings.setValue(CASTEP_EXECUTABLE_KEY, "/opt/castep/bin/castep")
    settings.setValue(CASTEP_KEYWORDS_JSON_KEY, str(legacy))


def test_startup_adopts_a_legacy_catalog_into_the_managed_copy(
    settings: QSettings, tmp_path: Path
) -> None:
    legacy = write_complete_catalog(tmp_path / "CASTEP-26.11" / "castep_keywords.json")
    _legacy_setup(settings, legacy)

    setup = apply_saved_castep_environment(settings)

    managed = managed_keywords_json_path()
    assert managed.is_file()
    assert setup.keywords_json_path == str(managed)
    assert load_castep_setup(settings).keywords_json_path == str(managed)
    # Other saved settings survive the adoption.
    assert setup.executable_path == "/opt/castep/bin/castep"
    assert os.environ[SETUP_KEYWORDS_JSON_ENV_VAR] == str(managed)
    status = keyword_catalog_status()
    assert (status.origin, status.path, status.castep_version) == ("setup", managed, "26.11")
    assert status.issue is None
    # The legacy file is copied, never moved.
    assert legacy.is_file()


def test_a_second_start_does_not_copy_again(settings: QSettings, tmp_path: Path) -> None:
    legacy = write_complete_catalog(tmp_path / "CASTEP-26.11" / "castep_keywords.json")
    _legacy_setup(settings, legacy)
    apply_saved_castep_environment(settings)

    # The old file changes after adoption; a second start must not pick it up.
    write_complete_catalog(legacy, "25.12")
    os.environ.pop(SETUP_KEYWORDS_JSON_ENV_VAR, None)
    schema_module.refresh_keyword_catalog(allow_fallback=True)

    setup = apply_saved_castep_environment(settings)

    managed = managed_keywords_json_path()
    assert setup.keywords_json_path == str(managed)
    status = keyword_catalog_status()
    assert (status.origin, status.path, status.castep_version) == ("setup", managed, "26.11")


def test_a_missing_legacy_catalog_is_left_configured_and_reported(
    settings: QSettings, tmp_path: Path
) -> None:
    legacy = tmp_path / "CASTEP-26.11" / "castep_keywords.json"
    _legacy_setup(settings, legacy)

    setup = apply_saved_castep_environment(settings)

    assert setup.keywords_json_path == str(legacy)
    assert load_castep_setup(settings).keywords_json_path == str(legacy)
    assert not managed_keywords_json_path().exists()
    status = keyword_catalog_status()
    assert status.is_starter is True
    assert status.issue == f"The configured keyword catalog '{legacy}' no longer exists."


def test_an_invalid_legacy_catalog_is_left_configured_and_reported(
    settings: QSettings, tmp_path: Path
) -> None:
    legacy = tmp_path / "castep_keywords.json"
    legacy.write_text("{not json", encoding="utf-8")
    _legacy_setup(settings, legacy)

    setup = apply_saved_castep_environment(settings)

    assert setup.keywords_json_path == str(legacy)
    assert load_castep_setup(settings).keywords_json_path == str(legacy)
    assert not managed_keywords_json_path().exists()
    status = keyword_catalog_status()
    assert status.is_starter is True
    assert status.issue is not None
    assert status.issue.startswith(f"The keyword catalog '{legacy}' could not be used:")


def test_an_unwritable_data_directory_keeps_reading_the_legacy_catalog(
    settings: QSettings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A regular file where the data directory's parent should be, so creating
    # the directory fails on every platform.
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(castep_setup_module, "_app_data_directory", lambda: blocker / "data")
    legacy = write_complete_catalog(tmp_path / "CASTEP-26.11" / "castep_keywords.json")
    _legacy_setup(settings, legacy)

    with caplog.at_level(logging.WARNING, logger="mufinder.castep_setup"):
        setup = apply_saved_castep_environment(settings)

    assert setup.keywords_json_path == str(legacy)
    assert load_castep_setup(settings).keywords_json_path == str(legacy)
    status = keyword_catalog_status()
    assert (status.origin, status.path, status.castep_version) == ("setup", legacy, "26.11")
    assert any("reading it in place" in record.getMessage() for record in caplog.records)


def test_the_managed_copy_is_not_reinstalled(
    settings: QSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed = write_complete_catalog(managed_keywords_json_path())
    settings.setValue(CASTEP_KEYWORDS_JSON_KEY, str(managed))

    def refuse_install(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the managed copy must not be installed again")

    monkeypatch.setattr(castep_setup_module, "install_keyword_catalog", refuse_install)

    setup = apply_saved_castep_environment(settings)

    assert setup.keywords_json_path == str(managed)
    assert keyword_catalog_status().path == managed
