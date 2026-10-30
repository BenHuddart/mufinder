"""Tests for MuFinder's painted icons: lookup, rendering size and theme following."""

from __future__ import annotations

import gc

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPixmap

import mufinder.app.icons as icons_module
from mufinder.app import theme as theme_module
from mufinder.app.icons import GLYPH_NAMES, icon, workflow_stage_icon

STAGE_STATES = ("completed", "active", "incomplete", "unknown")


def _opaque_pixels_near(icon_to_render: QIcon, colour: str, *, size: int = 16) -> int:
    """Count the clearly opaque pixels within a small tolerance of ``colour``."""

    target = QColor(colour)
    image = icon_to_render.pixmap(QSize(size, size), 2.0).toImage()
    image = image.convertToFormat(QImage.Format.Format_ARGB32)
    count = 0
    for x in range(image.width()):
        for y in range(image.height()):
            pixel = image.pixelColor(x, y)
            if pixel.alpha() < 230:
                continue
            if (
                abs(pixel.red() - target.red()) <= 12
                and abs(pixel.green() - target.green()) <= 12
                and abs(pixel.blue() - target.blue()) <= 12
            ):
                count += 1
    return count


@pytest.fixture
def light_theme(qapp):
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    try:
        yield qapp
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_workflow_stage_icon_states_render(qapp) -> None:
    for state in STAGE_STATES:
        rendered = workflow_stage_icon(state)
        assert not rendered.isNull(), state
        assert not rendered.pixmap(16, 16).isNull(), state


@pytest.mark.parametrize("dpr", [1.0, 2.0])
@pytest.mark.parametrize("name", GLYPH_NAMES)
def test_glyph_pixmap_is_painted_at_the_requested_physical_size(qapp, name, dpr) -> None:
    for logical in (16, 22):
        pixmap = icon(name).pixmap(QSize(logical, logical), dpr)

        expected = int(logical * dpr)
        assert (pixmap.width(), pixmap.height()) == (expected, expected), (name, logical)
        assert pixmap.devicePixelRatio() == dpr


@pytest.mark.parametrize("dpr", [1.0, 2.0])
def test_stage_icon_pixmap_is_painted_at_the_requested_physical_size(qapp, dpr) -> None:
    for state in STAGE_STATES:
        pixmap = workflow_stage_icon(state).pixmap(QSize(16, 16), dpr)

        assert (pixmap.width(), pixmap.height()) == (int(16 * dpr), int(16 * dpr)), state
        assert pixmap.devicePixelRatio() == dpr


def _rgba_image(pixmap) -> QImage:
    return pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)


@pytest.mark.parametrize("name", ["settings", "project_home"])
def test_glyph_is_painted_for_the_device_pixel_ratio_not_resampled(qapp, name) -> None:
    # The 2x pixmap of a 16 px icon is the glyph painted straight onto 32
    # physical pixels: identical to a 32 px icon at 1x, and not the 16 px
    # rendering scaled up.
    rendered = icon(name)
    at_2x = _rgba_image(rendered.pixmap(QSize(16, 16), 2.0))
    painted_at_32 = _rgba_image(rendered.pixmap(QSize(32, 32), 1.0))
    upscaled = _rgba_image(rendered.pixmap(QSize(16, 16), 1.0)).scaled(
        32, 32, mode=Qt.TransformationMode.SmoothTransformation
    )

    assert at_2x == painted_at_32
    assert at_2x != upscaled


def test_scaled_pixmap_reads_a_physical_size_before_qt_6_8(monkeypatch, qapp) -> None:
    # Before Qt 6.8, QIcon passes scaledPixmap the size already multiplied by
    # the scale; from 6.8 on it passes the logical size.
    engine = icons_module._GlyphIconEngine(
        icons_module._draw_stage_pending, 16.0, icons_module._stage_colours
    )
    monkeypatch.setattr(icons_module, "_SCALED_PIXMAP_TAKES_LOGICAL_SIZE", False)
    old_style = engine.scaledPixmap(QSize(32, 32), QIcon.Mode.Normal, QIcon.State.Off, 2.0)
    monkeypatch.setattr(icons_module, "_SCALED_PIXMAP_TAKES_LOGICAL_SIZE", True)
    new_style = engine.scaledPixmap(QSize(16, 16), QIcon.Mode.Normal, QIcon.State.Off, 2.0)

    for pixmap in (old_style, new_style):
        assert (pixmap.width(), pixmap.height()) == (32, 32)
        assert pixmap.devicePixelRatio() == 2.0


def test_render_cache_keeps_scales_apart(qapp) -> None:
    rendered = icon("delete")

    at_1x = rendered.pixmap(QSize(32, 32), 1.0)
    at_2x = rendered.pixmap(QSize(16, 16), 2.0)

    assert at_1x.devicePixelRatio() == 1.0
    assert at_2x.devicePixelRatio() == 2.0


def test_copied_icon_can_be_changed_without_touching_the_original(qapp) -> None:
    # Changing a copy makes Qt clone the engine; the clone must outlive the
    # call that created it.
    original = icon("settings")
    copy = QIcon(original)
    extra = QPixmap(8, 8)
    extra.fill(QColor("#123456"))

    copy.addPixmap(extra)
    gc.collect()

    assert not copy.pixmap(QSize(16, 16), 1.0).isNull()
    assert not original.pixmap(QSize(16, 16), 1.0).isNull()
    assert icon("settings").availableSizes() == []
    del copy
    gc.collect()
    assert not icon("settings").pixmap(QSize(16, 16), 2.0).isNull()


def test_existing_icon_follows_apply_theme(light_theme) -> None:
    light = theme_module.light_palette()
    dark = theme_module.dark_palette()
    created_in_light = icon("new_project")
    assert _opaque_pixels_near(created_in_light, light.text_primary) > 0

    theme_module.apply_theme(light_theme, theme_module.THEME_DARK)

    assert _opaque_pixels_near(created_in_light, dark.text_primary) > 0
    assert _opaque_pixels_near(created_in_light, dark.accent) > 0
    assert _opaque_pixels_near(created_in_light, light.text_primary) == 0
    assert _opaque_pixels_near(created_in_light, light.accent) == 0


def test_existing_stage_icon_follows_apply_theme(light_theme) -> None:
    light = theme_module.light_palette()
    dark = theme_module.dark_palette()
    completed = workflow_stage_icon("completed")
    assert _opaque_pixels_near(completed, light.seal_gold) > 0

    theme_module.apply_theme(light_theme, theme_module.THEME_DARK)

    assert _opaque_pixels_near(completed, dark.seal_gold) > 0
    assert _opaque_pixels_near(completed, light.seal_gold) == 0


def test_home_button_and_file_menu_follow_appearance_switch(main_window, light_theme) -> None:
    # The window applies the saved or system theme when it is built, so the
    # light theme is applied after it.
    theme_module.apply_theme(light_theme, theme_module.THEME_LIGHT)
    light = theme_module.light_palette()
    dark = theme_module.dark_palette()

    def submenu_actions(actions, title: str) -> dict:
        menu = next(action.menu() for action in actions if action.text() == title)
        return {action.text(): action for action in menu.actions()}

    menu_bar_actions = main_window.menuBar().actions()
    new_project = submenu_actions(menu_bar_actions, "File")["New Project"]
    view_actions = submenu_actions(menu_bar_actions, "View")
    appearance = submenu_actions(list(view_actions.values()), "Appearance")
    home_button = main_window._project_home_button
    assert _opaque_pixels_near(home_button.icon(), light.accent, size=22) > 0
    assert _opaque_pixels_near(new_project.icon(), light.accent) > 0

    appearance["Dark"].trigger()

    for rendered, size in ((home_button.icon(), 22), (new_project.icon(), 16)):
        assert _opaque_pixels_near(rendered, dark.accent, size=size) > 0
        assert _opaque_pixels_near(rendered, light.accent, size=size) == 0
        assert _opaque_pixels_near(rendered, light.text_primary, size=size) == 0


def test_disabled_icon_is_generated_from_the_current_rendering(qapp) -> None:
    rendered = icon("open_project")
    normal = rendered.pixmap(QSize(16, 16), 1.0, QIcon.Mode.Normal).toImage()
    disabled = rendered.pixmap(QSize(16, 16), 1.0, QIcon.Mode.Disabled).toImage()

    assert disabled.size() == normal.size()
    assert disabled != normal


def test_semantic_icons_render_even_without_style_icons(monkeypatch, qapp) -> None:
    class _StyleWithoutIcons:
        def standardIcon(self, _standard):
            return QIcon()

    monkeypatch.setattr(qapp, "style", lambda: _StyleWithoutIcons())

    for name in GLYPH_NAMES:
        rendered = icon(name)
        assert not rendered.isNull(), name
        assert not rendered.pixmap(18, 18).isNull(), name


def test_branded_icons_prefer_semantic_over_platform_standard(monkeypatch, qapp) -> None:
    # Names with a MuFinder glyph never consult the platform's standard icon,
    # so they look the same on every platform.
    class _StyleThatShouldNotBeUsed:
        def standardIcon(self, _standard):
            raise AssertionError("a branded glyph must be used instead of the platform icon")

    monkeypatch.setattr(qapp, "style", lambda: _StyleThatShouldNotBeUsed())

    for name in ("new_project", "open_project", "save_project", "demo", "cluster", "settings"):
        rendered = icon(name)
        assert not rendered.isNull(), name
        assert not rendered.pixmap(18, 18).isNull(), name


def test_unbranded_name_falls_back_to_platform_standard_icon(monkeypatch, qapp) -> None:
    # Names without a hand-drawn glyph still resolve, via the platform standard
    # icon (which, under the forced Fusion style, is itself cross-platform).
    sentinel = icon("new_project")
    calls: list[object] = []

    class _StyleWithStandardIcon:
        def standardIcon(self, standard):
            calls.append(standard)
            return sentinel

    monkeypatch.setattr(qapp, "style", lambda: _StyleWithStandardIcon())

    rendered = icon("refresh")  # mapped in _ICON_MAP but has no semantic drawing

    assert calls, "fallback to the platform standard icon was expected"
    assert not rendered.isNull()


def test_unknown_name_gives_a_null_icon(qapp) -> None:
    assert icon("no_such_icon").isNull()
    assert "no_such_icon" not in icons_module.GLYPH_NAMES
