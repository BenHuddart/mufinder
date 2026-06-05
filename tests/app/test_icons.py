"""Tests for MuFinder workflow marker icon helpers."""

from __future__ import annotations

import mufinder.app.icons as icons_module
from mufinder.app.icons import icon, workflow_stage_icon, workflow_tick_icon


def test_workflow_stage_icon_states_render(qapp) -> None:
    completed = workflow_stage_icon("completed")
    active = workflow_stage_icon("active")
    incomplete = workflow_stage_icon("incomplete")
    fallback = workflow_stage_icon("unknown")

    assert not completed.isNull()
    assert not active.isNull()
    assert not incomplete.isNull()
    assert not fallback.isNull()


def test_workflow_tick_icon_compatibility(qapp) -> None:
    done = workflow_tick_icon(True)
    pending = workflow_tick_icon(False)

    assert not done.isNull()
    assert not pending.isNull()


def test_icon_canvas_scales_with_device_pixel_ratio(monkeypatch, qapp) -> None:
    monkeypatch.setattr(icons_module, "_icon_device_pixel_ratio", lambda: 2.0)

    canvas = icons_module._icon_canvas(16)

    assert canvas.width() == 32
    assert canvas.height() == 32
    assert canvas.devicePixelRatio() == 2.0


def test_semantic_icons_render_even_without_style_icons(monkeypatch, qapp) -> None:
    class _StyleWithoutIcons:
        def standardIcon(self, _standard):
            return icons_module.QIcon()

    monkeypatch.setattr(qapp, "style", lambda: _StyleWithoutIcons())

    for name in (
        "new_project",
        "open_project",
        "project_home",
        "save_project",
        "save_workspace",
        "demo",
        "cluster",
        "settings",
        "delete",
        "appearance_light",
        "appearance_dark",
        "appearance_system",
        "help",
    ):
        rendered = icon(name)
        assert not rendered.isNull(), name
        assert not rendered.pixmap(18, 18).isNull(), name


def test_project_home_icon_uses_mufinder_semantic_mark(monkeypatch, qapp) -> None:
    class _StyleThatShouldNotBeUsed:
        def standardIcon(self, _standard):
            raise AssertionError("project_home should use the MuFinder semantic icon")

    monkeypatch.setattr(qapp, "style", lambda: _StyleThatShouldNotBeUsed())

    rendered = icon("project_home")

    assert not rendered.isNull()
    assert not rendered.pixmap(22, 22).isNull()
