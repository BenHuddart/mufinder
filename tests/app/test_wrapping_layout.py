"""The row layout that wraps its items when the width runs out."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QWidget

from mufinder.app.wrapping_layout import WrappingRowLayout


def test_spacing_reads_back_what_was_set_and_places_the_items(qapp) -> None:
    page = QWidget()
    layout = WrappingRowLayout(page, spacing=4)
    layout.setContentsMargins(0, 0, 0, 0)
    first, second = QLabel("first", page), QLabel("second", page)
    layout.addWidget(first)
    layout.addWidget(second)

    layout.setSpacing(20)
    page.resize(600, 100)
    page.show()
    qapp.processEvents()

    assert layout.spacing() == 20
    assert second.x() - (first.x() + first.width()) == 20
    page.close()


def test_items_wrap_onto_a_new_line_when_the_width_runs_out(qapp) -> None:
    page = QWidget()
    layout = WrappingRowLayout(page, spacing=4)
    layout.setContentsMargins(0, 0, 0, 0)
    first, second = QLabel("first label", page), QLabel("second label", page)
    layout.addWidget(first)
    layout.addWidget(second)
    page.resize(max(first.sizeHint().width(), second.sizeHint().width()) + 4, 100)
    page.show()
    qapp.processEvents()

    assert second.y() > first.y()
    page.close()
