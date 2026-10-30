"""Guard that tests never resolve the developer's real MuFinder settings file.

``tests/conftest.py`` redirects every file-backed ``QSettings`` store to the
test's ``tmp_path``. These tests only read ``QSettings.fileName()``; they never
write a value, so a regression cannot itself touch the real file.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys

import pytest
from PySide6.QtCore import QSettings

# macOS and Windows keep native settings in preferences and the registry, which
# QSettings.setPath cannot redirect; on Linux the native format is an INI file.
_native_is_a_file = pytest.mark.skipif(
    sys.platform in {"darwin", "win32"}, reason="native settings are not a file here"
)

_SETTINGS_FACTORIES = [
    # The constructor the app uses: native format, user scope.
    pytest.param(
        lambda: QSettings("MuFinder", "MuFinder"), id="native-user", marks=_native_is_a_file
    ),
    pytest.param(
        lambda: QSettings(
            QSettings.Format.NativeFormat, QSettings.Scope.SystemScope, "MuFinder", "MuFinder"
        ),
        id="native-system",
        marks=_native_is_a_file,
    ),
    pytest.param(
        lambda: QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope, "MuFinder", "MuFinder"
        ),
        id="ini-user",
    ),
    pytest.param(
        lambda: QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.SystemScope, "MuFinder", "MuFinder"
        ),
        id="ini-system",
    ),
]


@pytest.mark.parametrize("factory", _SETTINGS_FACTORIES)
def test_settings_resolve_under_the_test_temp_dir(
    factory: Callable[[], QSettings], tmp_path: Path
) -> None:
    resolved = Path(factory().fileName()).resolve()

    assert resolved.is_relative_to(tmp_path.resolve()), resolved
