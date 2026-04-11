"""Tests for reusable math rendering widgets."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel

from mufinder.app.math_label import MathLabel


def test_math_label_renders_valid_expression_to_pixmap(qapp) -> None:
    widget = MathLabel(r"\alpha = \frac{a}{b}", display_mode=True)

    rendered_lines = widget.findChildren(QLabel, "mathRenderedLine")
    fallback_lines = widget.findChildren(QLabel, "mathFallbackLine")

    assert widget.math() == r"\alpha = \frac{a}{b}"
    assert rendered_lines
    assert not fallback_lines
    assert widget.sizeHint().height() > 0
    assert rendered_lines[0].pixmap() is not None


def test_math_label_falls_back_to_plain_text_for_invalid_expression(qapp) -> None:
    widget = MathLabel(r"\notacommand{", display_mode=True)

    rendered_lines = widget.findChildren(QLabel, "mathRenderedLine")
    fallback_lines = widget.findChildren(QLabel, "mathFallbackLine")

    assert not rendered_lines
    assert len(fallback_lines) == 1
    assert fallback_lines[0].text() == r"\notacommand{"