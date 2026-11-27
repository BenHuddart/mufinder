"""The project navigator: its rows, seals, current row, keys and context menu."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QLabel, QStyleOptionViewItem

from mufinder.app.project_navigator import (
    HOST_STRUCTURE_ROUTE,
    JOURNAL_ROUTE,
    PROJECT_HOME_ROUTE,
    ProjectNavigator,
    RowKind,
    navigator_rows,
)
from mufinder.domain.models import ComparisonTable
from mufinder.domain.routes import (
    comparison_route,
    magnetic_field_workspace_route,
    relaxation_analysis_workspace_route,
)
from mufinder.services.analysis_workspaces import set_analysis_workspaces
from mufinder.services.project_service import ProjectDocument, ProjectService, active_host_entry
from mufinder.services.project_summary import (
    MAGNETIC_FIELD_WORKSPACES_KEY,
    StageStatus,
    summarize_project,
)
from tests.support.structures import add_relaxed_host, host_structure

#: The rows of a project with no saved workspaces, as (indent, label).
FRESH_ROWS = [
    (0, "Project Home"),
    (0, "Journal"),
    (0, "Prepare"),
    (0, "Host Structure"),
    (0, "Convergence Testing"),
    (1, "Cutoff Energy"),
    (1, "K-Point Grid"),
    (0, "Pristine Relaxation"),
    (0, "Find sites"),
    (0, "UEP"),
    (0, "Muon Sites"),
    (1, "Generation"),
    (1, "Relaxations"),
    (1, "Analysis"),
    (0, "Interpret"),
    (0, "Hyperfine Coupling"),
    (0, "Magnetic Fields"),
    (0, "Polarisation"),
    (0, "Quantum Corrections"),
    (0, "Energy Barriers"),
]


@pytest.fixture
def document(tmp_path: Path) -> ProjectDocument:
    return ProjectService().create_project(tmp_path / "CoF2.mufinder", "CoF2")


def _with_workspaces(document: ProjectDocument) -> ComparisonTable:
    """Give ``document`` a comparison table, an analysis and a magnetic workspace."""

    host = host_structure()
    service = ProjectService()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    table = ComparisonTable(label="Comparison 1", experimental_structure_id=host.id)
    service.add_comparison_table(document, table)
    set_analysis_workspaces(
        document, [{"id": "ana1", "name": "Positive muon sites"}, {"id": " ", "name": "blank"}]
    )
    document.ui_state[MAGNETIC_FIELD_WORKSPACES_KEY] = [{"id": "mag1", "title": "Octahedral site"}]
    return table


@pytest.fixture
def navigator(qapp, document: ProjectDocument) -> ProjectNavigator:
    navigator = ProjectNavigator()
    navigator.refresh(document, summarize_project(document))
    return navigator


def _shown(navigator: ProjectNavigator) -> ProjectNavigator:
    navigator.resize(ProjectNavigator.DEFAULT_WIDTH, 900)
    navigator.show()
    QApplication.processEvents()
    return navigator


def _announcements(navigator: ProjectNavigator) -> list[object]:
    announced: list[object] = []
    navigator.route_selected.connect(announced.append)
    navigator.home_requested.connect(lambda: announced.append(PROJECT_HOME_ROUTE))
    return announced


def test_a_fresh_project_lists_home_journal_and_the_phases_in_order(
    navigator: ProjectNavigator,
) -> None:
    rows = navigator.rows()

    assert [(row.indent, row.label) for row in rows] == FRESH_ROWS
    assert [row.label for row in rows if row.kind is RowKind.HEADING] == [
        "Prepare",
        "Find sites",
        "Interpret",
    ]
    assert [row.route for row in rows[:2]] == [PROJECT_HOME_ROUTE, JOURNAL_ROUTE]
    assert all(row.status is StageStatus.NOT_STARTED for row in rows if row.kind is RowKind.STAGE)
    assert navigator.current_route() == PROJECT_HOME_ROUTE
    assert navigator.active_route() is None


def test_a_full_project_lists_each_saved_item_under_its_stage(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    table = _with_workspaces(document)

    navigator.refresh(document, summarize_project(document))

    labels = [(row.indent, row.label) for row in navigator.rows()]
    assert labels[labels.index((0, "Pristine Relaxation")) + 1] == (1, "Comparison 1")
    assert labels[labels.index((1, "Analysis")) + 1] == (2, "Positive muon sites")
    assert labels[labels.index((0, "Magnetic Fields")) + 1] == (1, "Octahedral site")
    assert len(labels) == len(FRESH_ROWS) + 3
    for route in (
        comparison_route(table.id),
        relaxation_analysis_workspace_route("ana1"),
        magnetic_field_workspace_route("mag1"),
    ):
        row = navigator.row(route)
        assert row.kind is RowKind.WORKSPACE
        assert row.status is None
        assert row.tooltip == row.label
    assert navigator.row("host_structure").status is StageStatus.COMPLETE
    assert not navigator.has_route("supercell_editor")


def test_rails_join_the_seals_of_a_phase_and_of_a_group() -> None:
    rows = {row.route: row for row in navigator_rows(None, None) if row.route}

    # The Prepare rail runs from Host Structure to Pristine Relaxation, past
    # the convergence stages; their own rail joins Cutoff Energy and K-Point Grid.
    assert rows["host_structure"].rails == ((0, False, True),)
    assert rows["convergence"].rails == ((0, True, True),)
    assert rows["convergence_cutoff"].rails == ((0, True, True), (1, False, True))
    assert rows["convergence_kpoint"].rails == ((0, True, True), (1, True, False))
    assert rows["pristine_relaxation"].rails == ((0, True, False),)
    assert rows["muon_analysis"].rails == ((1, True, False),)
    assert rows[PROJECT_HOME_ROUTE].rails == ()


def test_stage_rows_carry_their_goal_and_status_and_say_it_aloud(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    host = navigator.row("host_structure")
    assert host.tooltip == "Complete when: Load a host structure."
    assert navigator.row(JOURNAL_ROUTE).tooltip == ""

    index = navigator.index_of("muon_relaxations")
    assert index.data(Qt.ItemDataRole.AccessibleTextRole) == "Relaxations, not started"
    assert index.data(Qt.ItemDataRole.ToolTipRole).startswith("Complete when: ")

    ProjectService().add_job(
        document, label="mu1", workflow_stage=_relaxation_stage(), calc_type="muon_relaxation"
    )
    navigator.refresh(document, summarize_project(document))

    index = navigator.index_of("muon_relaxations")
    assert index.data(Qt.ItemDataRole.AccessibleTextRole) == "Relaxations, under way"
    assert navigator.row("muon_sites").status is StageStatus.IN_PROGRESS
    assert (
        navigator.index_of(PROJECT_HOME_ROUTE).data(Qt.ItemDataRole.AccessibleTextRole)
        == "Project Home"
    )


def _relaxation_stage():
    from mufinder.domain.enums import WorkflowStage

    return WorkflowStage.MUON_RELAXATION


def test_optional_stages_are_quiet_until_they_start(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    assert navigator.row("uep").quiet
    assert navigator.row("pristine_relaxation").quiet
    assert not navigator.row("host_structure").quiet

    _with_workspaces(document)
    navigator.refresh(document, summarize_project(document))

    assert navigator.row("pristine_relaxation").status is StageStatus.COMPLETE
    assert not navigator.row("pristine_relaxation").quiet


def test_headings_cannot_be_selected_and_the_arrow_keys_step_over_them(
    navigator: ProjectNavigator,
) -> None:
    _shown(navigator)
    heading = navigator.view.model().index(2, 0)
    assert heading.data() == "Prepare"
    assert not heading.flags() & Qt.ItemFlag.ItemIsSelectable
    assert not heading.flags() & Qt.ItemFlag.ItemIsEnabled
    announced = _announcements(navigator)
    navigator.focus_list()

    QTest.keyClick(navigator.view, Qt.Key.Key_Down)
    QTest.keyClick(navigator.view, Qt.Key.Key_Down)
    assert navigator.current_route() == "host_structure"
    QTest.keyClick(navigator.view, Qt.Key.Key_Up)
    assert navigator.current_route() == JOURNAL_ROUTE

    navigator.set_current_route("pristine_relaxation")
    QTest.keyClick(navigator.view, Qt.Key.Key_Down)
    assert navigator.current_route() == "uep"
    QTest.keyClick(navigator.view, Qt.Key.Key_Up)
    assert navigator.current_route() == "pristine_relaxation"
    QTest.keyClick(navigator.view, Qt.Key.Key_End)
    assert navigator.current_route() == "energy_barriers"
    QTest.keyClick(navigator.view, Qt.Key.Key_Home)
    assert navigator.current_route() == PROJECT_HOME_ROUTE

    assert announced == [
        JOURNAL_ROUTE,
        "host_structure",
        JOURNAL_ROUTE,
        "uep",
        "pristine_relaxation",
        "energy_barriers",
        PROJECT_HOME_ROUTE,
    ]


def test_clicking_a_heading_or_below_the_rows_changes_nothing(
    navigator: ProjectNavigator,
) -> None:
    _shown(navigator)
    announced = _announcements(navigator)
    view = navigator.view
    heading = view.model().index(2, 0)
    last = view.visualRect(view.model().index(view.model().rowCount() - 1, 0))

    for point in (view.visualRect(heading).center(), last.center() + last.height() * QPoint(0, 2)):
        QTest.mouseClick(
            view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point
        )

        assert navigator.current_route() == PROJECT_HOME_ROUTE
        assert view.selectionModel().selectedIndexes() == [navigator.index_of(PROJECT_HOME_ROUTE)]
    assert announced == []


def test_a_click_announces_the_route_once(navigator: ProjectNavigator) -> None:
    _shown(navigator)
    announced = _announcements(navigator)
    centre = navigator.view.visualRect(navigator.index_of("uep")).center()

    QTest.mouseClick(
        navigator.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre
    )
    QTest.mouseClick(
        navigator.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, centre
    )

    assert announced == ["uep"]
    assert navigator.active_route() == "uep"


def test_the_highlighted_row_is_always_the_current_one(navigator: ProjectNavigator) -> None:
    _shown(navigator)
    announced = _announcements(navigator)
    view = navigator.view
    navigator.set_current_route("host_structure")
    navigator.focus_list()

    QTest.keyClick(view, Qt.Key.Key_Down, Qt.KeyboardModifier.ControlModifier)
    assert navigator.current_route() == "convergence"
    assert view.selectionModel().selectedIndexes() == [navigator.index_of("convergence")]

    QTest.mouseClick(
        view.viewport(),
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ControlModifier,
        view.visualRect(navigator.index_of("convergence")).center(),
    )
    assert view.selectionModel().selectedIndexes() == [navigator.index_of("convergence")]
    assert announced == ["convergence"]


def test_setting_the_route_from_code_announces_nothing(navigator: ProjectNavigator) -> None:
    announced = _announcements(navigator)

    assert navigator.set_current_route("uep")
    assert navigator.set_current_route(PROJECT_HOME_ROUTE)
    assert not navigator.set_current_route("no_such_page")

    assert announced == []
    assert navigator.current_route() == PROJECT_HOME_ROUTE
    assert navigator.view.selectionModel().selectedIndexes() == [
        navigator.index_of(PROJECT_HOME_ROUTE)
    ]


def test_a_refresh_keeps_the_current_row_unless_told_not_to(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    announced = _announcements(navigator)
    navigator.set_current_route("uep")

    assert navigator.refresh(document, summarize_project(document)) is True
    assert navigator.current_route() == "uep"

    assert navigator.refresh(document, None, keep_current=False) is False
    assert navigator.current_route() == PROJECT_HOME_ROUTE
    assert announced == []


def test_a_changed_seal_repaints_its_row_without_resetting_the_list(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    navigator.resize(ProjectNavigator.DEFAULT_WIDTH, 200)
    navigator.show()
    QApplication.processEvents()
    resets: list[None] = []
    navigator.view.model().modelReset.connect(lambda: resets.append(None))
    navigator.set_current_route("host_structure")
    scroll_bar = navigator.view.verticalScrollBar()
    scroll_bar.setValue(scroll_bar.maximum())
    assert scroll_bar.value() > 0

    navigator.refresh(document, summarize_project(document))
    ProjectService().add_structure(document, host_structure())
    document.project.host_structure_id = next(iter(document.structures))
    navigator.refresh(document, summarize_project(document))

    assert resets == []
    assert navigator.row("host_structure").status is StageStatus.COMPLETE
    assert navigator.current_route() == "host_structure"
    assert scroll_bar.value() == scroll_bar.maximum()

    # A new saved item changes the rows listed, which resets the list.
    _with_workspaces(document)
    navigator.refresh(document, summarize_project(document))

    assert len(resets) == 1
    assert navigator.current_route() == "host_structure"


def test_a_refresh_that_loses_the_current_row_falls_back_to_the_home(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    _with_workspaces(document)
    navigator.refresh(document, summarize_project(document))
    navigator.set_current_route(magnetic_field_workspace_route("mag1"))

    document.ui_state[MAGNETIC_FIELD_WORKSPACES_KEY] = []

    assert navigator.refresh(document, summarize_project(document)) is False
    assert navigator.current_route() == PROJECT_HOME_ROUTE


def test_workspace_rows_offer_their_deletion_and_other_rows_have_no_menu(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    table = _with_workspaces(document)
    navigator.refresh(document, summarize_project(document))
    deleted: list[tuple[str, str]] = []
    navigator.delete_analysis_requested.connect(lambda wid: deleted.append(("analysis", wid)))
    navigator.delete_magnetic_workspace_requested.connect(
        lambda wid: deleted.append(("magnetic", wid))
    )

    analysis_menu = navigator.context_menu(relaxation_analysis_workspace_route("ana1"))
    magnetic_menu = navigator.context_menu(magnetic_field_workspace_route("mag1"))
    assert [action.text() for action in analysis_menu.actions()] == ["Delete Analysis…"]
    assert [action.text() for action in magnetic_menu.actions()] == ["Delete Workspace…"]
    analysis_menu.actions()[0].trigger()
    magnetic_menu.actions()[0].trigger()

    assert deleted == [("analysis", "ana1"), ("magnetic", "mag1")]
    for route in ("uep", "muon_sites", PROJECT_HOME_ROUTE, comparison_route(table.id)):
        assert navigator.context_menu(route) is None, route
    assert navigator.context_menu(magnetic_field_workspace_route("gone")) is None


def test_a_renamed_workspace_relabels_its_row_and_leaves_the_current_row(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    table = _with_workspaces(document)
    navigator.refresh(document, summarize_project(document))
    navigator.set_current_route("uep")
    announced = _announcements(navigator)

    navigator.set_row_label(comparison_route(table.id), "Relaxed cell")

    row = navigator.row(comparison_route(table.id))
    assert (row.label, row.tooltip) == ("Relaxed cell", "Relaxed cell")
    assert navigator.current_route() == "uep"
    assert announced == []

    set_analysis_workspaces(document, [{"id": "ana1", "name": "Renamed sites"}])
    navigator.refresh(document, summarize_project(document))

    assert navigator.row(relaxation_analysis_workspace_route("ana1")).label == "Renamed sites"

    # A blank title keeps the row's label, as the row builder would.
    label = navigator.row(comparison_route(table.id)).label
    navigator.set_row_label(comparison_route(table.id), "  ")
    assert navigator.row(comparison_route(table.id)).label == label


def test_the_header_shows_the_name_alone_with_the_full_name_as_tooltip(
    navigator: ProjectNavigator,
) -> None:
    name = "A project with a very long name that cannot fit the navigator"

    navigator.set_title(name)
    _shown(navigator)

    title = navigator.findChild(QLabel, "projectTreePanelTitle")
    assert title.text() == name
    assert title.toolTip() == name
    assert title.displayed_text().endswith("…")
    assert navigator.title() == name


def _painted_texts(navigator: ProjectNavigator, route: str, monkeypatch) -> list[str]:
    """The texts the delegate draws for ``route``'s row, in the order drawn."""

    import mufinder.app.project_navigator as navigator_module

    drawn: list[str] = []
    original = navigator_module.QPainter.drawText

    def record(painter, *args):
        if len(args) == 3 and isinstance(args[2], str):
            drawn.append(args[2])
        return original(painter, *args)

    view = navigator.view
    index = navigator.index_of(route)
    rect = view.visualRect(index)
    option = QStyleOptionViewItem()
    option.initFrom(view)
    option.rect = rect.translated(-rect.topLeft())
    image = QImage(rect.size(), QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    monkeypatch.setattr(navigator_module.QPainter, "drawText", record)
    try:
        view.itemDelegate().paint(painter, option, index)
    finally:
        monkeypatch.setattr(navigator_module.QPainter, "drawText", original)
        painter.end()
    return drawn


def test_without_a_host_the_host_structure_row_has_one_line(
    navigator: ProjectNavigator, monkeypatch
) -> None:
    _shown(navigator)
    row = navigator.row(HOST_STRUCTURE_ROUTE)
    one_line = navigator.view.visualRect(navigator.index_of("uep")).height()

    assert (row.detail, row.spoken_detail) == ("", "")
    assert row.tooltip == "Complete when: Load a host structure."
    assert navigator.view.visualRect(navigator.index_of(HOST_STRUCTURE_ROUTE)).height() == one_line
    assert _painted_texts(navigator, HOST_STRUCTURE_ROUTE, monkeypatch) == ["Host Structure"]


def test_the_host_structure_row_names_an_imported_host_on_a_second_line(
    navigator: ProjectNavigator, document: ProjectDocument, monkeypatch
) -> None:
    _shown(navigator)
    one_line = navigator.view.visualRect(navigator.index_of("uep")).height()
    host = host_structure(label="CoF2")
    ProjectService().add_structure(document, host)
    document.project.host_structure_id = host.id
    navigator.set_current_route("uep")

    navigator.refresh(document, summarize_project(document))
    QApplication.processEvents()

    index = navigator.index_of(HOST_STRUCTURE_ROUTE)
    assert navigator.row(HOST_STRUCTURE_ROUTE).detail == "Original — CoF2"
    assert index.data(Qt.ItemDataRole.ToolTipRole) == (
        "Active host: Original — CoF2\nComplete when: Load a host structure."
    )
    assert index.data(Qt.ItemDataRole.AccessibleTextRole) == (
        "Host Structure, complete, active host Original — CoF2"
    )
    # The row is laid out taller at once; the other rows and the current row stay.
    assert navigator.view.visualRect(index).height() > one_line
    assert navigator.view.visualRect(navigator.index_of("uep")).height() == one_line
    assert navigator.current_route() == "uep"
    assert _painted_texts(navigator, HOST_STRUCTURE_ROUTE, monkeypatch) == [
        "Host Structure",
        "Original — CoF2",
    ]


def test_the_host_structure_row_names_a_relaxed_host_without_its_date(
    navigator: ProjectNavigator, document: ProjectDocument
) -> None:
    host = host_structure(label="CoF2")
    service = ProjectService()
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    relaxed = add_relaxed_host(document)
    service.set_active_host_structure(document, relaxed.id)
    full = active_host_entry(document).label
    assert full.startswith("Relaxed — relax_1 (")

    navigator.refresh(document, summarize_project(document))

    row = navigator.row(HOST_STRUCTURE_ROUTE)
    assert row.detail == "Relaxed — relax_1"
    assert row.tooltip.startswith(f"Active host: {full}\n")
    assert row.accessible_text == f"Host Structure, complete, active host {full}"

    # Closing the project leaves the row on one line.
    navigator.refresh(None, None)
    assert navigator.row(HOST_STRUCTURE_ROUTE).detail == ""


def test_a_host_name_too_long_for_the_navigator_is_elided_on_the_right(
    navigator: ProjectNavigator, document: ProjectDocument, monkeypatch
) -> None:
    name = "A host structure with a very long name from a long file"
    host = host_structure(label=name)
    ProjectService().add_structure(document, host)
    document.project.host_structure_id = host.id
    navigator.refresh(document, summarize_project(document))
    navigator.resize(ProjectNavigator.MINIMUM_WIDTH, 900)
    navigator.show()
    QApplication.processEvents()

    label, detail = _painted_texts(navigator, HOST_STRUCTURE_ROUTE, monkeypatch)

    assert label == "Host Structure"
    assert detail.endswith("…")
    assert f"Original — {name}".startswith(detail[:-1])
    assert navigator.row(HOST_STRUCTURE_ROUTE).tooltip.startswith(f"Active host: Original — {name}")
