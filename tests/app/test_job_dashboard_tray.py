"""The Job Dashboard tray: collapsing, fitting its rows, clipping the page and the summary."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from PySide6.QtCore import QByteArray, QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDockWidget, QPushButton, QScrollArea, QWidget
import pytest

from mufinder.app import workflow_table_utils
from mufinder.app.job_dashboard_tray import (
    COLLAPSED_SETTINGS_KEY,
    TRAY_HEIGHT_SHARE,
    ClippablePageArea,
    job_summary_sentence,
)
from mufinder.app.main_window import MainWindow
from mufinder.app.theme import SPACING
from mufinder.domain.models import utc_now
from mufinder.services.project_service import ProjectService
from tests.support.dashboard import add_dashboard_job, open_context_menu
from tests.support.jobs import set_job_state
from tests.support.qt import press_key, pump_until

WINDOW_SIZE = (1280, 1000)


def _settle() -> None:
    for _ in range(4):
        QApplication.processEvents()


@pytest.fixture
def service() -> ProjectService:
    return ProjectService()


@pytest.fixture
def window(
    main_window: MainWindow, tmp_path: Path, service: ProjectService
) -> Iterator[MainWindow]:
    project = service.create_project(tmp_path / "Tray.mufinder", "Tray").project_root
    main_window.resize(*WINDOW_SIZE)
    main_window.show()
    assert main_window.open_project(project)
    _settle()
    yield main_window
    # Hidden, a window closes without asking, so the Qt clean-up shows no real prompt.
    main_window.hide()


def _add_jobs(window: MainWindow, service: ProjectService, statuses, *, start: int = 0):
    jobs = [
        add_dashboard_job(service, window._document, f"j{start + index}", status)
        for index, status in enumerate(statuses)
    ]
    window._refresh_jobs_table()
    _settle()
    return jobs


def _tray(window: MainWindow):
    return window._job_tray


def _dock(window: MainWindow) -> QDockWidget:
    return window.findChild(QDockWidget, "jobDashboardDock")


def _table(window: MainWindow):
    return window._job_dashboard.table


def _toggle(window: MainWindow):
    return _tray(window).header.toggle_button


def _page_area(window: MainWindow) -> ClippablePageArea:
    return window.findChild(ClippablePageArea, "pageClipArea")


def _strip_height(window: MainWindow) -> int:
    return _dock(window).minimumSizeHint().height()


# --- Summary --------------------------------------------------------------------


def test_the_summary_counts_the_listed_jobs_by_state(window, service) -> None:
    summary = _tray(window).header.summary
    assert summary.entry_texts() == ["No jobs"]
    assert summary.accessibleName() == "No jobs"

    _add_jobs(window, service, ["failed", "queued", "running", "queued", "cancelled"])

    assert summary.entry_texts() == ["1 running", "2 queued", "1 failed", "1 cancelled"]
    assert summary.accessibleName() == "Jobs: 1 running, 2 queued, 1 failed and 1 cancelled"


def test_the_summary_counts_dismissed_jobs_only_while_they_are_listed(window, service) -> None:
    failed, _running = _add_jobs(window, service, ["failed", "running"])
    failed.dismissed_at = utc_now()
    window._refresh_jobs_table()
    summary = _tray(window).header.summary
    assert summary.entry_texts() == ["1 running"]

    window._job_dashboard.show_dismissed_box.setChecked(True)
    _settle()

    assert summary.entry_texts() == ["1 running", "1 failed"]


def test_each_summary_count_carries_its_state_word_and_glyph(window, service) -> None:
    _add_jobs(window, service, ["stopped"])
    summary = _tray(window).header.summary

    (button,) = summary.count_buttons()
    assert button.objectName() == "jobSummaryCount"
    assert button.property("jobState") == "stopped"
    assert button.text() == "1 stopped"
    assert not button.icon().isNull()
    assert button.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon


def test_the_summary_sentence_joins_the_counts() -> None:
    assert job_summary_sentence(()) == "No jobs"
    assert job_summary_sentence((("running", 4),)) == "Jobs: 4 running"
    assert job_summary_sentence((("running", 4), ("failed", 2))) == "Jobs: 4 running and 2 failed"


def test_the_summary_sentence_says_what_a_filter_lists() -> None:
    counts = (("running", 4), ("failed", 2), ("stopped", 0))

    assert job_summary_sentence(counts[:2], ("failed",)) == (
        "Jobs: 4 running and 2 failed. Listing failed only."
    )
    assert job_summary_sentence(counts, ("failed", "stopped")) == (
        "Jobs: 4 running, 2 failed and 0 stopped. Listing failed and stopped only."
    )


def test_the_header_strip_holds_the_toggle_summary_hint_and_show_dismissed(window) -> None:
    header = _tray(window).header
    assert _dock(window).titleBarWidget() is header
    for widget in (
        _toggle(window),
        header.summary,
        window._job_dashboard.selection_hint,
        window._job_dashboard.show_dismissed_box,
    ):
        assert header.isAncestorOf(widget)
    assert _toggle(window).text() == "Job Dashboard"


def test_the_page_runs_down_to_the_tray_with_no_band_between(window) -> None:
    central = window.centralWidget()
    assert central.layout().contentsMargins().bottom() == 0
    area = _page_area(window)
    page_bottom = area.mapTo(window, area.rect().bottomLeft()).y()
    gap = _dock(window).geometry().top() - page_bottom - 1
    # Only the themed separator, which reads as the strip's top edge, lies between.
    assert gap == SPACING.xs


def test_the_tray_docks_at_the_bottom_only_and_has_no_close_button(window) -> None:
    dock = _dock(window)
    # Closable for View > Job Dashboard; the custom title bar shows no close button.
    assert dock.features() == QDockWidget.DockWidgetFeature.DockWidgetClosable
    assert dock.titleBarWidget() is _tray(window).header
    assert dock.allowedAreas() == Qt.DockWidgetArea.BottomDockWidgetArea
    assert window.dockWidgetArea(dock) == Qt.DockWidgetArea.BottomDockWidgetArea


# --- Collapsing and expanding -----------------------------------------------------


def test_the_toggle_collapses_to_the_strip_and_expands_again(window, service) -> None:
    _add_jobs(window, service, ["running", "failed"])
    toggle = _toggle(window)
    assert toggle.accessibleName() == "Collapse Job Dashboard"
    assert "Ctrl+J" in toggle.toolTip()

    toggle.click()
    _settle()

    assert _tray(window).is_collapsed()
    assert _table(window).isHidden()
    assert _dock(window).height() == _strip_height(window)
    assert toggle.accessibleName() == "Expand Job Dashboard"
    assert "Ctrl+J" in toggle.toolTip()

    toggle.click()
    _settle()

    assert not _tray(window).is_collapsed()
    assert _table(window).isVisible()
    assert _dock(window).height() == _tray(window).fitted_height()
    assert toggle.accessibleName() == "Collapse Job Dashboard"


def test_a_collapsed_tray_cannot_be_dragged_taller(window) -> None:
    _toggle(window).click()
    _settle()

    window.resizeDocks([_dock(window)], [400], Qt.Orientation.Vertical)
    _settle()

    assert _dock(window).height() == _strip_height(window)


def test_ctrl_j_toggles_and_moves_focus_into_and_out_of_the_table(window, service) -> None:
    _add_jobs(window, service, ["failed"])
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    table = _table(window)
    table.setCurrentCell(0, 0)
    table.setFocus(Qt.FocusReason.OtherFocusReason)
    _settle()

    press_key(Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier)
    _settle()
    assert _tray(window).is_collapsed()
    assert QApplication.focusWidget() is _toggle(window)

    press_key(Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier)
    _settle()
    assert not _tray(window).is_collapsed()
    assert QApplication.focusWidget() is table


def test_collapsing_clears_the_selection_so_the_delete_keys_do_nothing(
    window, service, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_jobs(window, service, ["failed", "running"])
    asked: list[object] = []
    monkeypatch.setattr(
        workflow_table_utils,
        "confirm_destructive",
        lambda *args, **_kwargs: asked.append(args) or True,
    )
    table = _table(window)
    table.selectAll()
    table.setCurrentCell(0, 0)

    _toggle(window).click()
    _settle()

    assert table.selectionModel().selectedRows() == []
    assert table.currentRow() == -1
    window._job_dashboard.delete_shortcut.activated.emit()
    window._job_dashboard.backspace_shortcut.activated.emit()
    assert asked == []


def test_a_collapsed_tray_stops_the_runtime_timer_and_keeps_its_summary(window, service) -> None:
    (running,) = _add_jobs(window, service, ["running"])
    running.started_at = utc_now()
    window._refresh_jobs_table()
    dashboard = window._job_dashboard
    assert dashboard.runtime_timer.isActive()

    _toggle(window).click()
    _settle()
    assert not dashboard.runtime_timer.isActive()

    _add_jobs(window, service, ["failed"], start=1)
    assert not dashboard.runtime_timer.isActive()
    assert _tray(window).header.summary.entry_texts() == ["1 running", "1 failed"]

    _toggle(window).click()
    _settle()
    assert dashboard.runtime_timer.isActive()


def test_show_dismissed_works_while_collapsed(window, service) -> None:
    (failed,) = _add_jobs(window, service, ["failed"])
    failed.dismissed_at = utc_now()
    window._refresh_jobs_table()
    _toggle(window).click()
    _settle()
    box = window._job_dashboard.show_dismissed_box
    assert box.isVisible()

    box.click()
    _settle()

    assert _table(window).rowCount() == 1
    assert _tray(window).header.summary.entry_texts() == ["1 failed"]


def test_the_toggle_does_nothing_while_the_tray_is_closed(window) -> None:
    _dock(window).close()
    _settle()

    _tray(window).toggle()

    assert not _tray(window).is_collapsed()


# --- Fitting ----------------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 3])
def test_an_expanded_tray_shows_every_row_without_a_scroll_bar(window, service, count) -> None:
    _add_jobs(window, service, ["failed"] * count)
    table = _table(window)

    assert pump_until(lambda: _dock(window).height() == _tray(window).fitted_height())
    assert not table.verticalScrollBar().isVisible()
    rows_height = table.verticalHeader().length() or table.verticalHeader().defaultSectionSize()
    assert table.viewport().height() >= rows_height


def test_an_expanded_tray_is_capped_and_its_table_scrolls(window, service) -> None:
    _add_jobs(window, service, ["failed"] * 60)
    tray = _tray(window)
    shared = window.centralWidget().height() + _dock(window).height()

    assert pump_until(lambda: _dock(window).height() == tray.height_cap())
    assert tray.height_cap() == int(shared * TRAY_HEIGHT_SHARE)
    assert tray.fitted_height() > tray.height_cap()
    assert _table(window).verticalScrollBar().isVisible()


def test_the_tray_refits_as_jobs_join_and_leave(window, service) -> None:
    jobs = _add_jobs(window, service, ["failed"])
    one_row = _dock(window).height()

    _add_jobs(window, service, ["failed", "failed"], start=1)
    assert pump_until(lambda: _dock(window).height() > one_row)
    three_rows = _dock(window).height()
    assert three_rows == _tray(window).fitted_height()

    for job in jobs:
        job.dismissed_at = utc_now()
    window._refresh_jobs_table()
    assert pump_until(lambda: _dock(window).height() < three_rows)
    assert _dock(window).height() == _tray(window).fitted_height()


def test_the_tray_refits_when_a_row_is_resized(window, service) -> None:
    _add_jobs(window, service, ["failed"] * 3)
    fitted = _dock(window).height()

    _table(window).verticalHeader().resizeSection(0, 100)

    assert pump_until(lambda: _dock(window).height() > fitted)
    assert _dock(window).height() == _tray(window).fitted_height()
    assert not _table(window).verticalScrollBar().isVisible()


def test_the_tray_refits_when_the_horizontal_scroll_bar_shows(window, service) -> None:
    _add_jobs(window, service, ["failed"] * 2)
    table = _table(window)
    fitted = _dock(window).height()

    table.horizontalHeader().resizeSection(window._job_dashboard.STAGE_COLUMN, 4000)

    assert pump_until(lambda: table.horizontalScrollBar().isVisible())
    assert pump_until(lambda: _dock(window).height() > fitted)
    assert _dock(window).height() == _tray(window).fitted_height()


def _drag_dock_edge_to(window: MainWindow, height: int) -> None:
    """Press on the window, resize the dock, release: what dragging the dock's edge does.

    The press is sent to the window itself, as a press on a dock edge is, but
    away from the edge, so Qt's own edge drag does not undo the resize.
    """

    point = QPointF(window.centralWidget().geometry().center())
    for kind in (QEvent.Type.MouseButtonPress, None, QEvent.Type.MouseButtonRelease):
        if kind is None:
            window.resizeDocks([_dock(window)], [height], Qt.Orientation.Vertical)
            _settle()
            continue
        QApplication.sendEvent(
            window,
            QMouseEvent(
                kind,
                point,
                window.mapToGlobal(point),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton
                if kind == QEvent.Type.MouseButtonPress
                else Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
            ),
        )
    _settle()


def test_a_dragged_height_wins_until_the_next_toggle(window, service) -> None:
    _add_jobs(window, service, ["failed"])
    fitted = _dock(window).height()

    _drag_dock_edge_to(window, fitted + 60)
    dragged = _dock(window).height()
    assert dragged > fitted

    _add_jobs(window, service, ["failed"], start=1)
    _settle()
    assert _dock(window).height() == dragged

    _toggle(window).click()
    _toggle(window).click()
    _settle()
    assert _dock(window).height() == _tray(window).fitted_height()


def test_a_clipping_tray_dragged_short_can_be_dragged_tall_again(window, service) -> None:
    _add_jobs(window, service, ["failed"] * 60)
    tray = _tray(window)
    assert pump_until(lambda: _dock(window).height() == tray.height_cap())
    tall = _dock(window).height()

    _drag_dock_edge_to(window, _strip_height(window) + 120)
    assert _dock(window).height() < tall
    assert not _page_area(window).is_clipped()

    _drag_dock_edge_to(window, tall)
    assert _dock(window).height() == tall
    assert _page_area(window).is_clipped()


def test_a_dragged_tray_can_reach_the_cap_after_many_jobs_join(window, service) -> None:
    _add_jobs(window, service, ["failed"])
    _drag_dock_edge_to(window, _strip_height(window) + 80)
    dragged = _dock(window).height()

    _add_jobs(window, service, ["failed"] * 60, start=1)
    _settle()
    assert _dock(window).height() == dragged

    cap = _tray(window).height_cap()
    _drag_dock_edge_to(window, cap)
    assert _dock(window).height() == cap
    assert _page_area(window).is_clipped()


def test_a_clipping_tray_shown_again_from_the_view_menu_fits_its_rows(window, service) -> None:
    _add_jobs(window, service, ["failed"] * 60)
    tray = _tray(window)
    assert pump_until(lambda: _dock(window).height() == tray.height_cap())
    tall = _dock(window).height()
    action = _dock(window).toggleViewAction()

    action.trigger()
    _settle()
    assert not _dock(window).isVisible()
    assert not _page_area(window).is_height_released()

    action.trigger()
    assert pump_until(lambda: _dock(window).height() == tall)
    assert _page_area(window).is_clipped()


def test_a_window_resize_re_applies_the_cap(window, service) -> None:
    _add_jobs(window, service, ["failed"] * 60)
    tray = _tray(window)
    assert pump_until(lambda: _dock(window).height() == tray.height_cap())
    tall = _dock(window).height()

    window.resize(WINDOW_SIZE[0], WINDOW_SIZE[1] + 200)
    assert pump_until(lambda: _dock(window).height() > tall)
    assert _dock(window).height() == tray.height_cap()


# --- Clipping the page ----------------------------------------------------------


def test_many_rows_clip_the_page_without_resizing_the_window(window, service) -> None:
    area = _page_area(window)
    page_minimum = area.page_minimum_size().height()
    minimum = window.minimumSize()
    size = window.size()

    _add_jobs(window, service, ["failed"] * 60)
    assert pump_until(lambda: area.is_clipped())

    assert area.height() < page_minimum
    assert window._main_stack.height() == page_minimum
    assert window.size() == size
    assert window.minimumSize() == minimum
    assert area.minimumSizeHint().width() == area.page_minimum_size().width()


def _wheel_down_over(widget: QWidget) -> None:
    centre = QPointF(widget.rect().center())
    event = QWheelEvent(
        centre,
        QPointF(widget.mapToGlobal(centre.toPoint())),
        QPoint(0, 0),
        QPoint(0, -240),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)
    _settle()


def test_a_clipped_page_stays_at_its_top_whatever_asks_it_to_scroll(window, service) -> None:
    area = _page_area(window)
    _add_jobs(window, service, ["failed"] * 60)
    assert pump_until(lambda: area.is_clipped())
    assert area.verticalScrollBar().maximum() > 0

    _wheel_down_over(area.viewport())
    assert area.verticalScrollBar().value() == 0

    area.verticalScrollBar().setValue(area.verticalScrollBar().maximum())
    area.ensureVisible(0, window._main_stack.height() - 1)
    _settle()
    assert area.verticalScrollBar().value() == 0
    assert window._main_stack.mapTo(area.viewport(), QPoint(0, 0)).y() == 0


def _focusable_children(window: MainWindow) -> list[QWidget]:
    page = window._main_stack.currentWidget()
    return [
        child
        for child in page.findChildren(QWidget)
        if child.isVisibleTo(page)
        and child.isEnabled()
        and child.focusPolicy() & Qt.FocusPolicy.TabFocus
        and child.height() > 0
    ]


def test_focus_on_a_control_hidden_behind_the_tray_collapses_it(window, service) -> None:
    area = _page_area(window)
    window.show_page("convergence_cutoff")
    _add_jobs(window, service, ["failed"] * 60)
    assert pump_until(lambda: area.is_clipped())
    window.activateWindow()
    _settle()
    shown = [child for child in _focusable_children(window) if not area.hides(child)]
    hidden = [child for child in _focusable_children(window) if area.hides(child)]
    assert shown and hidden

    shown[0].setFocus(Qt.FocusReason.TabFocusReason)
    _settle()
    assert not _tray(window).is_collapsed()

    hidden[0].setFocus(Qt.FocusReason.TabFocusReason)
    _settle()

    assert _tray(window).is_collapsed()
    assert QApplication.focusWidget() is hidden[0]
    assert not area.hides(hidden[0])


def test_focus_inside_the_tray_never_collapses_it(window, service) -> None:
    area = _page_area(window)
    _add_jobs(window, service, ["failed"] * 60)
    assert pump_until(lambda: area.is_clipped())
    window.activateWindow()

    _table(window).setFocus(Qt.FocusReason.TabFocusReason)
    window._job_dashboard.show_dismissed_box.setFocus(Qt.FocusReason.TabFocusReason)
    _settle()

    assert not _tray(window).is_collapsed()


def test_a_tray_expanded_over_the_focused_control_takes_focus_to_its_button(
    window, service
) -> None:
    area = _page_area(window)
    window.show_page("convergence_cutoff")
    _add_jobs(window, service, ["failed"] * 60)
    assert pump_until(lambda: area.is_clipped())
    window.activateWindow()
    _settle()
    hidden = next(child for child in _focusable_children(window) if area.hides(child))
    _toggle(window).click()
    _settle()
    hidden.setFocus(Qt.FocusReason.MouseFocusReason)
    _settle()
    assert QApplication.focusWidget() is hidden

    _toggle(window).click()
    assert pump_until(lambda: area.is_clipped())

    assert not _tray(window).is_collapsed()
    assert QApplication.focusWidget() is _toggle(window)


def test_a_control_scrolled_out_of_a_view_in_the_page_is_not_hidden_by_the_tray(
    window, service
) -> None:
    area = _page_area(window)
    _add_jobs(window, service, ["failed"] * 60)
    assert pump_until(lambda: area.is_clipped())
    page = window._main_stack.currentWidget()
    # A short view at the top of the page whose own content runs far below it.
    view = QScrollArea(page)
    view.setGeometry(0, 0, 200, 40)
    content = QWidget()
    content.resize(180, 4000)
    scrolled_out = QPushButton("far down", content)
    scrolled_out.move(0, 3900)
    view.setWidget(content)
    view.show()
    _settle()

    assert scrolled_out.mapTo(page, QPoint(0, 0)).y() >= area.viewport().height()
    assert not area.hides(scrolled_out)


def _tab_order(window: MainWindow, widgets) -> list[QWidget]:
    """Return the shown ``widgets`` in the order Tab reaches them, from the toggle on."""

    order = []
    start = widget = _toggle(window)
    while True:
        if widget in widgets and widget.isVisible():
            order.append(widget)
        widget = widget.nextInFocusChain()
        if widget is start:
            break
    return order


def test_tab_goes_through_the_header_strip_before_the_table(window, service) -> None:
    _add_jobs(window, service, ["failed", "running"])
    summary = _summary(window)
    show_dismissed = window._job_dashboard.show_dismissed_box
    running, failed = summary.count_buttons()
    strip = [_toggle(window), running, failed, summary.show_all_button, show_dismissed]

    assert _tab_order(window, [*strip, _table(window)]) == [
        _toggle(window),
        running,
        failed,
        show_dismissed,
        _table(window),
    ]

    failed.click()
    _settle()

    assert _tab_order(window, [*strip, _table(window)]) == [*strip, _table(window)]


def test_a_count_that_joins_takes_its_place_in_the_tab_order(window, service) -> None:
    _add_jobs(window, service, ["failed"])
    _add_jobs(window, service, ["running", "stopped"], start=1)
    summary = _summary(window)
    buttons = summary.count_buttons()
    assert [button.text() for button in buttons] == ["1 running", "1 failed", "1 stopped"]

    assert _tab_order(window, [*buttons, window._job_dashboard.show_dismissed_box]) == [
        *buttons,
        window._job_dashboard.show_dismissed_box,
    ]


def test_the_summary_tooltip_reads_every_count(window, service) -> None:
    summary = _tray(window).header.summary
    assert summary.toolTip() == "No jobs"

    _add_jobs(window, service, ["failed", "queued", "running"])

    assert summary.toolTip() == "Jobs: 1 running, 1 queued and 1 failed"


def test_a_collapsed_tray_still_reads_job_progress(window, service) -> None:
    _add_jobs(window, service, ["running"])
    assert window._job_progress._enabled

    _toggle(window).click()
    _settle()

    assert _tray(window).is_collapsed()
    assert not window._job_dashboard.runtime_timer.isActive()
    assert window._job_progress._enabled


def test_rows_changing_during_an_edge_drag_wait_for_the_drag_to_end(window, service) -> None:
    _add_jobs(window, service, ["failed"] * 3)
    dock = _dock(window)
    fitted = dock.height()
    point = QPointF(window.centralWidget().geometry().center())

    def send(kind, buttons) -> None:
        QApplication.sendEvent(
            window,
            QMouseEvent(
                kind,
                point,
                window.mapToGlobal(point),
                Qt.MouseButton.LeftButton,
                buttons,
                Qt.KeyboardModifier.NoModifier,
            ),
        )

    send(QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton)
    _add_jobs(window, service, ["failed"] * 2, start=3)
    assert dock.height() == fitted

    send(QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)
    _settle()

    assert dock.height() == _tray(window).fitted_height() > fitted


def test_collapsing_gives_the_page_its_minimum_back_at_its_top(window, service) -> None:
    area = _page_area(window)
    _add_jobs(window, service, ["failed"] * 60)
    assert pump_until(lambda: area.is_clipped())
    expanded_minimum = window.minimumSize()

    _toggle(window).click()
    _settle()

    assert not area.is_clipped()
    assert area.height() >= area.page_minimum_size().height()
    # The whole page shows: it is as tall as the area, with nothing left to scroll.
    assert window._main_stack.height() == area.viewport().height()
    assert area.verticalScrollBar().maximum() == 0
    assert area.verticalScrollBar().value() == 0
    assert window.minimumSize() == expanded_minimum


def test_the_window_minimum_is_the_same_collapsed_and_expanded(window, service) -> None:
    _add_jobs(window, service, ["failed"] * 3)
    expanded = window.minimumSize()

    _toggle(window).click()
    _settle()

    assert window.minimumSize() == expanded


def _open_with_jobs(tmp_path: Path, count: int) -> MainWindow:
    """Open a project that already holds ``count`` failed jobs in a newly shown window."""

    service = ProjectService()
    document = service.create_project(tmp_path / "Many.mufinder", "Many")
    for index in range(count):
        add_dashboard_job(service, document, f"m{index}", "failed")
    service.save_project(document)
    window = MainWindow()
    window.resize(1280, 945)
    window.show()
    assert window.open_project(document.project_root)
    _settle()
    assert pump_until(lambda: _page_area(window).is_clipped())
    _settle()
    return window


def _collapsed_minimum(window: MainWindow):
    _tray(window).set_collapsed(True)
    _settle()
    minimum = window.minimumSize()
    _tray(window).set_collapsed(False)
    assert pump_until(lambda: _page_area(window).is_clipped())
    _settle()
    return minimum


def test_a_tray_clipping_from_the_first_show_holds_the_true_window_minimum(qapp, tmp_path) -> None:
    window = _open_with_jobs(tmp_path, 40)
    try:
        held = window.minimumSize()

        assert held == _collapsed_minimum(window)
    finally:
        window.hide()


def test_collapsing_after_the_page_minimum_grows_and_a_shrink_keeps_the_window_size(
    qapp, tmp_path
) -> None:
    window = _open_with_jobs(tmp_path, 40)
    try:
        area = _page_area(window)
        before = area.page_minimum_size().height()
        # Another page, whose content then needs more height than any page did.
        window._set_active_route("magnetic_fields")
        _settle()
        window._main_stack.currentWidget().setMinimumHeight(before + 40)
        assert pump_until(lambda: area.page_minimum_size().height() == before + 40)
        _settle()
        assert area.is_clipped()

        window.resize(window.minimumSize())
        _settle()
        size = window.size()
        _tray(window).set_collapsed(True)
        _settle()

        assert window.size() == size
        assert window.minimumSize() == size
    finally:
        window.hide()


def test_the_window_minimum_width_is_the_same_collapsed_and_clipping(qapp, tmp_path) -> None:
    window = _open_with_jobs(tmp_path, 40)
    try:
        clipping = window.minimumSize().width()

        assert _collapsed_minimum(window).width() == clipping
    finally:
        window.hide()


def test_the_page_minimum_follows_the_page(qapp) -> None:
    page = QWidget()
    page.setMinimumSize(300, 400)
    area = ClippablePageArea(page)
    assert area.minimumSizeHint().height() == 400

    area.set_height_released(True)
    assert area.minimumSizeHint().height() == 0
    assert area.minimumSizeHint().width() == 300

    area.set_height_released(False)
    page.setMinimumSize(300, 500)
    assert area.minimumSizeHint().height() == 500
    assert area.page() is page
    area.deleteLater()


def test_a_page_that_trades_height_for_width_is_laid_out_at_the_area_height(qapp) -> None:
    from PySide6.QtWidgets import QLabel, QVBoxLayout

    page = QWidget()
    layout = QVBoxLayout(page)
    label = QLabel("word " * 400)
    label.setWordWrap(True)
    layout.addWidget(label)
    area = ClippablePageArea(page)
    area.resize(300, max(area.minimumSizeHint().height(), 120))
    area.show()
    QApplication.processEvents()

    assert page.height() == area.viewport().height()
    assert area.verticalScrollBar().maximum() == 0
    area.hide()
    area.deleteLater()


# --- Opening a job's page ---------------------------------------------------------


def test_opening_a_job_collapses_the_tray_only_while_it_clips_the_page(window, service) -> None:
    _add_jobs(window, service, ["failed"])
    window._job_dashboard.action_requested.emit("open", [])
    _settle()
    assert not _tray(window).is_collapsed()

    _add_jobs(window, service, ["failed"] * 60, start=1)
    assert pump_until(lambda: _page_area(window).is_clipped())
    window._job_dashboard.action_requested.emit("open", [])
    _settle()

    assert _tray(window).is_collapsed()
    assert not _page_area(window).is_clipped()


# --- Remembering ------------------------------------------------------------------


def _restarted(window: MainWindow, project: Path) -> MainWindow:
    window.close()
    QApplication.processEvents()
    restarted = MainWindow()
    restarted.resize(*WINDOW_SIZE)
    restarted.show()
    assert restarted.open_project(project)
    _settle()
    return restarted


def test_the_collapsed_state_survives_a_restart(window, tmp_path) -> None:
    _toggle(window).click()
    _settle()
    assert QSettings("MuFinder", "MuFinder").value(COLLAPSED_SETTINGS_KEY) in (True, "true")

    restarted = _restarted(window, tmp_path / "Tray.mufinder")
    try:
        assert restarted._job_tray.is_collapsed()
        assert _dock(restarted).height() == _strip_height(restarted)
        assert _toggle(restarted).accessibleName() == "Expand Job Dashboard"
    finally:
        restarted.hide()


def test_the_tray_starts_expanded_with_a_garbled_collapsed_setting(qapp, tmp_path) -> None:
    QSettings("MuFinder", "MuFinder").setValue(COLLAPSED_SETTINGS_KEY, "not a bool")
    project = ProjectService().create_project(tmp_path / "G.mufinder", "G").project_root
    window = MainWindow()
    window.resize(*WINDOW_SIZE)
    window.show()
    try:
        assert window.open_project(project)
        _settle()
        assert not window._job_tray.is_collapsed()
    finally:
        window.hide()


@pytest.mark.parametrize("where", ["floating", "top"])
def test_a_saved_floating_or_top_dock_comes_back_docked_at_the_bottom(
    qapp, tmp_path, where
) -> None:
    project = ProjectService().create_project(tmp_path / "S.mufinder", "S").project_root
    first = MainWindow()
    first.resize(*WINDOW_SIZE)
    first.show()
    assert first.open_project(project)
    _settle()
    dock = _dock(first)
    dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetFloatable)
    if where == "floating":
        dock.setFloating(True)
    else:
        first.addDockWidget(Qt.DockWidgetArea.TopDockWidgetArea, dock)
    _settle()
    QSettings("MuFinder", "MuFinder").setValue(MainWindow._MAIN_WINDOW_STATE_KEY, first.saveState())
    first.hide()

    restarted = MainWindow()
    restarted.resize(*WINDOW_SIZE)
    restarted.show()
    try:
        assert restarted.open_project(project)
        _settle()
        restored = _dock(restarted)
        assert not restored.isFloating()
        assert restarted.dockWidgetArea(restored) == Qt.DockWidgetArea.BottomDockWidgetArea
        assert restored.isVisible()
    finally:
        restarted.hide()


def test_a_saved_window_state_with_no_dock_still_shows_the_tray(qapp, tmp_path) -> None:
    QSettings("MuFinder", "MuFinder").setValue(
        MainWindow._MAIN_WINDOW_STATE_KEY, QByteArray(b"garbage")
    )
    project = ProjectService().create_project(tmp_path / "B.mufinder", "B").project_root
    window = MainWindow()
    window.resize(*WINDOW_SIZE)
    window.show()
    try:
        assert window.open_project(project)
        _settle()
        assert _dock(window).isVisible()
        assert window.dockWidgetArea(_dock(window)) == Qt.DockWidgetArea.BottomDockWidgetArea
        assert _dock(window).height() == window._job_tray.fitted_height()
    finally:
        window.hide()


# --- Filtering by state -------------------------------------------------------------


def _summary(window: MainWindow):
    return _tray(window).header.summary


def _count(window: MainWindow, word: str):
    button = _summary(window).count_button(word)
    assert button is not None, f"no {word} count"
    return button


def _listed_labels(window: MainWindow) -> list[str]:
    table = _table(window)
    return [table.item(row, 0).text() for row in range(table.rowCount())]


def _selected_labels(window: MainWindow) -> list[str]:
    return sorted(job.label for job in window._job_dashboard.selected_jobs())


def _tick(window: MainWindow, word: str) -> None:
    _count(window, word).click()
    _settle()


def test_ticking_a_count_lists_only_the_jobs_in_its_state(window, service) -> None:
    _add_jobs(window, service, ["failed", "running", "failed", "stopped"])

    _tick(window, "failed")

    assert _listed_labels(window) == ["j0", "j2"]
    assert window._job_dashboard.state_filter() == ("failed",)
    assert _count(window, "failed").isChecked()
    assert not _count(window, "running").isChecked()
    # The counts still cover every listed job.
    assert _summary(window).entry_texts() == ["1 running", "2 failed", "1 stopped"]
    assert _summary(window).show_all_button.isVisible()


def test_ticking_several_counts_lists_the_jobs_in_each_state(window, service) -> None:
    _add_jobs(window, service, ["failed", "running", "stopped", "queued"])

    _tick(window, "stopped")
    _tick(window, "failed")

    assert _listed_labels(window) == ["j0", "j2"]
    assert window._job_dashboard.state_filter() == ("failed", "stopped")


def test_unticking_counts_lists_their_jobs_again(window, service) -> None:
    _add_jobs(window, service, ["failed", "running", "stopped"])
    _tick(window, "failed")
    _tick(window, "stopped")

    _tick(window, "failed")
    assert _listed_labels(window) == ["j2"]
    assert not _count(window, "failed").isChecked()

    _tick(window, "stopped")
    assert _listed_labels(window) == ["j0", "j1", "j2"]
    assert window._job_dashboard.state_filter() == ()
    assert not _summary(window).show_all_button.isVisible()


def test_space_ticks_and_unticks_a_focused_count(window, service) -> None:
    _add_jobs(window, service, ["failed", "running"])
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    failed = _count(window, "failed")
    failed.setFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(failed, Qt.Key.Key_Space)
    _settle()
    assert _listed_labels(window) == ["j0"]
    assert failed.isChecked()

    QTest.keyClick(failed, Qt.Key.Key_Space)
    _settle()
    assert _listed_labels(window) == ["j0", "j1"]
    assert not failed.isChecked()


def test_show_all_shows_only_while_a_count_is_ticked_and_clears_the_filter(window, service) -> None:
    _add_jobs(window, service, ["failed", "running", "stopped"])
    show_all = _summary(window).show_all_button
    assert not show_all.isVisible()
    _tick(window, "failed")
    _tick(window, "stopped")
    assert show_all.isVisible()

    show_all.click()
    _settle()

    assert _listed_labels(window) == ["j0", "j1", "j2"]
    assert window._job_dashboard.state_filter() == ()
    assert not any(button.isChecked() for button in _summary(window).count_buttons())
    assert not show_all.isVisible()


def test_show_all_hands_its_focus_to_the_last_count(window, service) -> None:
    _add_jobs(window, service, ["failed", "running"])
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    _tick(window, "running")
    show_all = _summary(window).show_all_button
    show_all.setFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(show_all, Qt.Key.Key_Space)
    _settle()

    assert not show_all.isVisible()
    assert QApplication.focusWidget() is _count(window, "failed")


def test_hidden_rows_leave_the_selection_and_do_not_come_back_selected(window, service) -> None:
    _add_jobs(window, service, ["failed", "running", "failed", "running"])
    _table(window).selectAll()
    assert _selected_labels(window) == ["j0", "j1", "j2", "j3"]

    _tick(window, "failed")
    assert _selected_labels(window) == ["j0", "j2"]

    _summary(window).show_all_button.click()
    _settle()

    # The running rows come back between the selected ones, unselected.
    assert _listed_labels(window) == ["j0", "j1", "j2", "j3"]
    assert _selected_labels(window) == ["j0", "j2"]


def test_rows_shown_again_inside_a_selected_range_are_not_selected(window, service) -> None:
    _add_jobs(window, service, ["failed", "running", "failed"])
    _tick(window, "failed")
    # One range over both shown rows; the running row comes back inside it.
    _table(window).selectAll()

    _tick(window, "failed")

    assert _listed_labels(window) == ["j0", "j1", "j2"]
    assert _selected_labels(window) == ["j0", "j2"]


def test_the_delete_keys_cannot_reach_a_job_the_filter_hides(
    window, service, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed, _stopped = _add_jobs(window, service, ["failed", "stopped"])
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)
    requests: list[tuple[list, list]] = []
    window._job_dashboard.cancel_and_dismiss_requested.connect(
        lambda cancel, dismiss: requests.append((list(cancel), list(dismiss)))
    )
    _table(window).selectAll()

    _tick(window, "failed")
    window._job_dashboard.delete_shortcut.activated.emit()
    window._job_dashboard.backspace_shortcut.activated.emit()

    assert requests[0] == ([], [failed.id])
    assert all(request == ([], [failed.id]) for request in requests)


def test_the_delete_keys_do_nothing_once_the_filter_hides_every_selected_row(
    window, service, monkeypatch: pytest.MonkeyPatch
) -> None:
    _add_jobs(window, service, ["failed", "running", "failed"])
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)
    requests: list[tuple[list, list]] = []
    window._job_dashboard.cancel_and_dismiss_requested.connect(
        lambda cancel, dismiss: requests.append((list(cancel), list(dismiss)))
    )
    _table(window).selectRow(1)

    _tick(window, "failed")
    window._job_dashboard.delete_shortcut.activated.emit()
    window._job_dashboard.backspace_shortcut.activated.emit()

    assert _selected_labels(window) == []
    assert requests == []


def test_the_runtime_timer_ticks_only_while_the_filter_shows_a_running_row(window, service) -> None:
    running, _failed = _add_jobs(window, service, ["running", "failed"])
    running.started_at = utc_now()
    window._refresh_jobs_table()
    dashboard = window._job_dashboard
    assert dashboard.runtime_timer.isActive()

    _tick(window, "failed")
    assert not dashboard.runtime_timer.isActive()

    _tick(window, "running")
    assert dashboard.runtime_timer.isActive()


def test_a_restart_starts_with_no_count_ticked(window, service, tmp_path) -> None:
    _add_jobs(window, service, ["failed", "running"])
    service.save_project(window._document)
    _tick(window, "failed")

    restarted = _restarted(window, tmp_path / "Tray.mufinder")
    try:
        assert restarted._job_dashboard.state_filter() == ()
        assert _listed_labels(restarted) == ["j0", "j1"]
        assert not _count(restarted, "failed").isChecked()
        assert not _summary(restarted).show_all_button.isVisible()
    finally:
        restarted.hide()


def test_the_context_menu_cannot_reach_a_job_the_filter_hides(
    window, service, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed, _stopped = _add_jobs(window, service, ["failed", "stopped"])
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)
    requests: list[tuple[str, list]] = []
    window._job_dashboard.action_requested.connect(
        lambda kind, ids: requests.append((kind, list(ids)))
    )
    _table(window).selectAll()
    _tick(window, "failed")

    menu = open_context_menu(window, monkeypatch, row=0)
    menu.action("Dismiss").triggered.emit()

    assert requests == [("dismiss", [failed.id])]


def test_a_menu_left_open_cannot_reach_a_job_that_left_the_ticked_state(
    window, service, monkeypatch: pytest.MonkeyPatch
) -> None:
    running, other = _add_jobs(window, service, ["running", "running"])
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)
    requests: list[tuple[str, list]] = []
    window._job_dashboard.action_requested.connect(
        lambda kind, ids: requests.append((kind, list(ids)))
    )
    _tick(window, "running")
    _table(window).selectAll()
    menu = open_context_menu(window, monkeypatch, row=0)

    # While the menu is open, a check moves one job out of the ticked state.
    set_job_state(running, "retrieving")
    window._refresh_jobs_table()
    _settle()
    menu.action("Cancel").triggered.emit()

    assert requests == [("cancel", [other.id])]


def test_a_job_leaving_a_ticked_state_loses_its_row(window, service) -> None:
    running, _failed = _add_jobs(window, service, ["running", "failed"])
    _tick(window, "running")
    assert _listed_labels(window) == ["j0"]

    set_job_state(running, "failed")
    window._refresh_jobs_table()
    _settle()

    assert _listed_labels(window) == []
    assert _summary(window).entry_texts() == ["0 running", "2 failed"]
    assert _count(window, "running").isChecked()


def test_a_ticked_state_with_no_jobs_stays_ticked_at_zero(window, service) -> None:
    failed, _running = _add_jobs(window, service, ["failed", "running"])
    _tick(window, "failed")

    failed.dismissed_at = utc_now()
    window._refresh_jobs_table()
    _settle()

    assert _summary(window).entry_texts() == ["1 running", "0 failed"]
    assert _count(window, "failed").isChecked()
    assert _count(window, "failed").accessibleName() == "0 failed jobs"
    assert _table(window).rowCount() == 0
    assert window._job_dashboard.state_filter() == ("failed",)
    assert _summary(window).show_all_button.isVisible()


def test_unticking_a_zero_count_drops_it_and_keeps_focus_in_the_strip(window, service) -> None:
    failed, _running = _add_jobs(window, service, ["failed", "running"])
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    _tick(window, "failed")
    failed.dismissed_at = utc_now()
    window._refresh_jobs_table()
    _settle()
    zero = _count(window, "failed")
    zero.setFocus(Qt.FocusReason.TabFocusReason)

    QTest.keyClick(zero, Qt.Key.Key_Space)
    _settle()

    assert _summary(window).count_button("failed") is None
    assert _summary(window).entry_texts() == ["1 running"]
    assert QApplication.focusWidget() is _count(window, "running")


def test_show_dismissed_under_a_filter_keeps_the_filter(window, service) -> None:
    dismissed, _failed, _running = _add_jobs(window, service, ["failed", "failed", "running"])
    dismissed.dismissed_at = utc_now()
    window._refresh_jobs_table()
    _tick(window, "failed")
    assert _listed_labels(window) == ["j1"]
    box = window._job_dashboard.show_dismissed_box

    box.click()
    _settle()
    assert _listed_labels(window) == ["j0", "j1"]
    assert _summary(window).entry_texts() == ["1 running", "2 failed"]
    assert _count(window, "failed").isChecked()

    box.click()
    _settle()
    assert _listed_labels(window) == ["j1"]
    assert _summary(window).entry_texts() == ["1 running", "1 failed"]
    assert window._job_dashboard.state_filter() == ("failed",)


def test_the_tray_fits_the_rows_the_filter_shows(window, service) -> None:
    _add_jobs(window, service, ["failed"] + ["running"] * 5)
    assert pump_until(lambda: _dock(window).height() == _tray(window).fitted_height())
    every_row = _dock(window).height()

    _tick(window, "failed")

    assert pump_until(lambda: _dock(window).height() < every_row)
    assert _dock(window).height() == _tray(window).fitted_height()

    _summary(window).show_all_button.click()
    assert pump_until(lambda: _dock(window).height() == every_row)


def test_ticking_a_count_while_collapsed_expands_the_tray(window, service) -> None:
    _add_jobs(window, service, ["failed", "running"])
    _toggle(window).click()
    _settle()

    _tick(window, "failed")

    assert not _tray(window).is_collapsed()
    assert _listed_labels(window) == ["j0"]


def test_unticking_or_show_all_while_collapsed_leaves_the_tray_collapsed(window, service) -> None:
    _add_jobs(window, service, ["failed", "running"])
    _tick(window, "failed")
    _tick(window, "running")
    _toggle(window).click()
    _settle()

    _tick(window, "failed")
    assert _tray(window).is_collapsed()
    assert window._job_dashboard.state_filter() == ("running",)

    _summary(window).show_all_button.click()
    _settle()
    assert _tray(window).is_collapsed()
    assert window._job_dashboard.state_filter() == ()


def test_the_filter_is_kept_across_collapsing_and_expanding(window, service) -> None:
    _add_jobs(window, service, ["failed", "running"])
    _tick(window, "failed")

    _toggle(window).click()
    _settle()
    _toggle(window).click()
    _settle()

    assert window._job_dashboard.state_filter() == ("failed",)
    assert _count(window, "failed").isChecked()
    assert _listed_labels(window) == ["j0"]


def test_opening_another_project_clears_the_filter(window, service, tmp_path) -> None:
    _add_jobs(window, service, ["failed", "running"])
    _tick(window, "failed")
    other_document = service.create_project(tmp_path / "Other.mufinder", "Other")
    add_dashboard_job(service, other_document, "k0", "cancelled")
    service.save_project(other_document)
    other = other_document.project_root

    assert window.open_project(other)
    _settle()

    assert window._job_dashboard.state_filter() == ()
    assert not _summary(window).show_all_button.isVisible()
    assert _summary(window).entry_texts() == ["1 cancelled"]
    assert not _count(window, "cancelled").isChecked()
    assert _listed_labels(window) == ["k0"]


def test_each_count_is_a_named_toggle(window, service) -> None:
    _add_jobs(window, service, ["failed", "running", "running"])
    running, failed = _summary(window).count_buttons()

    assert running.isCheckable() and failed.isCheckable()
    assert running.focusPolicy() & Qt.FocusPolicy.TabFocus
    assert running.accessibleName() == "2 running jobs"
    assert failed.accessibleName() == "1 failed job"
    assert failed.toolTip() == (
        "List only the failed jobs. Tick more counts to list several states."
    )
    assert not failed.isChecked()

    _tick(window, "failed")

    assert failed.isChecked()
    assert failed.toolTip() == "Stop filtering by failed jobs."
    show_all = _summary(window).show_all_button
    assert show_all.text() == "Show all"
    assert show_all.accessibleName() == "Show all jobs"
    assert show_all.toolTip() == "Clear the filter and list every job."


def test_focus_stays_on_a_count_whose_count_changes(window, service) -> None:
    _add_jobs(window, service, ["failed"])
    window.activateWindow()
    QTest.qWaitForWindowActive(window)
    failed = _count(window, "failed")
    failed.setFocus(Qt.FocusReason.TabFocusReason)
    _tick(window, "failed")

    _add_jobs(window, service, ["failed", "running"], start=1)

    assert _count(window, "failed") is failed
    assert failed.text() == "2 failed"
    assert failed.isChecked()
    assert QApplication.focusWidget() is failed


def test_the_summary_name_and_tooltip_say_what_the_filter_lists(window, service) -> None:
    _add_jobs(window, service, ["failed", "running", "stopped"])
    _tick(window, "stopped")
    _tick(window, "failed")

    expected = "Jobs: 1 running, 1 failed and 1 stopped. Listing failed and stopped only."
    assert _summary(window).accessibleName() == expected
    assert _summary(window).toolTip() == expected
