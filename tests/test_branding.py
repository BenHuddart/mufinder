from __future__ import annotations

from pathlib import Path

from mufinder import branding


def test_logo_and_windows_icon_paths_point_to_packaged_assets() -> None:
    logo = Path(branding.logo_path())
    windows_icon = Path(branding.windows_icon_path())
    home_map = Path(branding.project_home_map_path())
    home_color_map = Path(branding.project_home_color_map_path())

    assert logo.name == "mufinder_logo.png"
    assert logo.exists()
    assert windows_icon.name == "mufinder_logo.ico"
    assert windows_icon.exists()
    assert home_map.name == "project_home_map.png"
    assert home_map.exists()
    assert home_color_map.name == "project_home_map_color.png"
    assert home_color_map.exists()


def test_app_icon_uses_windows_icon_on_windows(monkeypatch) -> None:
    captured = {}

    class _FakeIcon:
        def __init__(self, path: str):
            captured["path"] = path

    monkeypatch.setattr(branding.sys, "platform", "win32")
    monkeypatch.setattr(branding, "QIcon", _FakeIcon)

    icon = branding.app_icon()

    assert isinstance(icon, _FakeIcon)
    assert captured["path"].endswith("mufinder_logo.ico")
