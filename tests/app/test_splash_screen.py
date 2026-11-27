"""Tests for splash-screen rendering helpers and progress updates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PySide6.QtGui import QColor, QPixmap

import mufinder.app.splash_screen as splash_module
from mufinder.app.splash_screen import MuFinderSplashScreen


def test_splash_progress_clamps_and_renders(qapp) -> None:
    splash = MuFinderSplashScreen()

    splash.set_progress(150, "  Loading UI…  ")

    assert splash._progress == 100
    assert splash._message == "Loading UI…"
    assert splash.pixmap() is not None
    assert not splash.pixmap().isNull()


def test_splash_progress_fill_takes_the_track_height(qapp) -> None:
    splash = MuFinderSplashScreen()
    splash.set_progress(50, "Loading")
    image = splash.pixmap().toImage()
    ratio = splash.pixmap().devicePixelRatio()
    # The track: x 84 to 596, y 294 to 304.
    fill = QColor("#2C2418")

    def pixel(x: int, y: int) -> QColor:
        return image.pixelColor(round(x * ratio), round(y * ratio))

    for y in (296, 299, 301):
        assert pixel(200, y) == fill
        assert pixel(500, y) != fill


def test_make_canvas_uses_device_pixel_ratio() -> None:
    canvas = MuFinderSplashScreen._make_canvas(2.0)

    assert canvas.width() == MuFinderSplashScreen._WIDTH * 2
    assert canvas.height() == MuFinderSplashScreen._HEIGHT * 2
    assert canvas.devicePixelRatio() == 2.0


def test_best_font_family_falls_back_for_unknown_fonts() -> None:
    chosen = MuFinderSplashScreen._best_font_family(
        ("DefinitelyNotInstalledFont",), fallback="Helvetica"
    )
    assert chosen == "Helvetica"


def test_load_logo_returns_none_for_missing_path(monkeypatch) -> None:
    monkeypatch.setattr(splash_module, "logo_path", lambda: "/tmp/this-file-does-not-exist.png")
    assert MuFinderSplashScreen._load_logo() is None


def test_load_logo_and_scaled_logo_work_with_valid_image(monkeypatch, tmp_path: Path, qapp) -> None:
    logo_path = tmp_path / "logo.png"
    pixmap = QPixmap(32, 32)
    assert pixmap.save(str(logo_path))
    monkeypatch.setattr(splash_module, "logo_path", lambda: str(logo_path))

    loaded = MuFinderSplashScreen._load_logo()
    assert loaded is not None
    assert not loaded.isNull()

    scaled = MuFinderSplashScreen._scaled_logo(SimpleNamespace(_logo=loaded), 2.0)
    assert scaled is not None
    assert not scaled.isNull()
    assert scaled.devicePixelRatio() == 2.0


def test_scaled_logo_returns_none_when_logo_missing() -> None:
    assert MuFinderSplashScreen._scaled_logo(SimpleNamespace(_logo=None), 2.0) is None


def test_device_pixel_ratio_uses_window_handle_screen_ratio() -> None:
    class _Screen:
        def devicePixelRatio(self) -> float:
            return 2.5

    class _Handle:
        def screen(self) -> _Screen:
            return _Screen()

    class _SplashLike:
        def windowHandle(self) -> _Handle:
            return _Handle()

        def _fallback_device_pixel_ratio(self) -> float:
            return 1.0

    dpr = MuFinderSplashScreen._device_pixel_ratio(_SplashLike())
    assert dpr == 2.5


def test_device_pixel_ratio_falls_back_without_window_handle() -> None:
    class _SplashLike:
        def windowHandle(self):
            return None

        def _fallback_device_pixel_ratio(self) -> float:
            return 1.25

    dpr = MuFinderSplashScreen._device_pixel_ratio(_SplashLike())
    assert dpr == 1.25
