"""Reading application settings values that are stored as text."""

from __future__ import annotations

from PySide6.QtCore import QSettings
import pytest

from mufinder.app.settings_values import read_bool_setting


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        (" Yes ", True),
        ("1", True),
        ("false", False),
        ("0", False),
        ("not a bool", False),
        (1, True),
        (0, False),
    ],
)
def test_a_saved_flag_reads_as_a_bool(qapp, saved, expected) -> None:
    settings = QSettings("MuFinder", "MuFinder")
    settings.setValue("test/flag", saved)

    assert read_bool_setting(settings, "test/flag") is expected


def test_a_missing_flag_reads_as_its_default(qapp) -> None:
    settings = QSettings("MuFinder", "MuFinder")
    settings.remove("test/absent")

    assert read_bool_setting(settings, "test/absent") is False
    assert read_bool_setting(settings, "test/absent", True) is True
