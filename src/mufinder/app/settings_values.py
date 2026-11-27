"""Reading application settings (``QSettings``) values that are stored as text."""

from __future__ import annotations

from PySide6.QtCore import QSettings


def read_bool_setting(settings: QSettings, key: str, default: bool = False) -> bool:
    """Return the bool saved under ``key``, or ``default`` when nothing is saved.

    ``QSettings`` hands a saved bool back as a bool or as text, depending on
    the platform's store. Text other than ``1``, ``true`` or ``yes`` reads as
    false.
    """

    value = settings.value(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
