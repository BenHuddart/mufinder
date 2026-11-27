"""The page shown when no workflow page is: the welcome page and the project home.

``HomePage`` shows one of two pages. With no project open it is the welcome
page: the logo and a title over cards to create or open a project and the five
newest recent projects. With a project open it is the project home
(``ProjectHomeWidget``), which has no logo or title of its own beyond the
project's name. The page only asks for things; the main window acts on
``new_requested``, ``open_requested``, ``recent_requested`` and
``route_requested``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.icons import icon
from mufinder.app.project_home import ProjectHomeWidget
from mufinder.app.recent_project_button import RecentProjectButton
from mufinder.app.recent_projects import RecentProjects
from mufinder.branding import logo_path

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument
    from mufinder.services.project_summary import ProjectSummary

#: How many recent projects the welcome page lists.
_WELCOME_RECENT_PROJECTS = 5


class HomePage(QWidget):
    """The welcome page with no project open, the project home with one."""

    new_requested = Signal()
    open_requested = Signal()
    recent_requested = Signal(str)
    route_requested = Signal(str)

    def __init__(self, recent_projects: RecentProjects, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._recent_projects = recent_projects

        self._title = QLabel("Welcome to MuFinder")
        self._title.setObjectName("homeHeroTitle")
        self._title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._subtitle = QLabel(
            "Your compass for muon site discovery.\n"
            "Start by loading a structure, then follow guided convergence and muon-site steps."
        )
        self._subtitle.setObjectName("homeHeroSubtitle")
        self._subtitle.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self.content_stack = QStackedWidget(self)
        self.content_stack.setObjectName("homeContentStack")
        self.welcome_page = self._build_welcome_page()
        self.project_page = self._build_project_page()
        self.content_stack.addWidget(self.welcome_page)
        self.content_stack.addWidget(self.project_page)
        layout.addWidget(self.content_stack, 1)

        self.refresh_recent_projects()
        self.set_document(None)
        self.show_project_page(False)

    def make_card(self, title: str, body: str) -> tuple[QFrame, QVBoxLayout]:
        """Return a welcome card with ``title`` over ``body``, and its layout for buttons."""

        card = QFrame(self)
        card.setObjectName("homeCard")
        layout = QVBoxLayout(card)

        title_label = QLabel(title, card)
        title_label.setObjectName("cardTitle")
        # A card beside a taller one has height to spare. The body takes it, so
        # every title keeps one height and the buttons line up at the bottom.
        title_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        body_label = QLabel(body, card)
        body_label.setObjectName("cardBody")
        body_label.setWordWrap(True)
        body_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        layout.addWidget(title_label)
        layout.addWidget(body_label)
        return card, layout

    @staticmethod
    def _fallback_device_pixel_ratio() -> float:
        app = QGuiApplication.instance()
        if app is None:
            return 1.0
        screen = app.primaryScreen()
        if screen is None:
            return 1.0
        return max(1.0, float(screen.devicePixelRatio()))

    def logo_device_pixel_ratio(self) -> float:
        """Return the device pixel ratio of the screen the window is on, at least 1."""

        handle = self.window().windowHandle()
        if handle is not None and handle.screen() is not None:
            return max(1.0, float(handle.screen().devicePixelRatio()))
        return self._fallback_device_pixel_ratio()

    def scaled_logo(self, logical_size: int = 92) -> QPixmap | None:
        """Return the logo at ``logical_size`` for this screen, or ``None`` without one."""

        logo_pixmap = QPixmap(logo_path())
        if logo_pixmap.isNull():
            return None

        dpr = self.logo_device_pixel_ratio()
        physical_size = max(1, int(round(logical_size * dpr)))
        logo_scaled = logo_pixmap.scaled(
            physical_size,
            physical_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        logo_scaled.setDevicePixelRatio(dpr)
        return logo_scaled

    def _build_welcome_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        hero_row = QHBoxLayout()
        hero_row.setSpacing(14)
        logo_label = QLabel(page)
        logo_label.setObjectName("homeLogo")
        logo_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        logo_pixmap = self.scaled_logo()
        if logo_pixmap is not None:
            logo_label.setPixmap(logo_pixmap)
        hero_row.addWidget(logo_label)
        hero_text = QVBoxLayout()
        hero_text.setSpacing(6)
        hero_text.addWidget(self._title)
        hero_text.addWidget(self._subtitle)
        hero_row.addLayout(hero_text, 1)
        layout.addLayout(hero_row)

        actions_grid = QGridLayout()
        actions_grid.setHorizontalSpacing(12)
        actions_grid.setVerticalSpacing(12)

        new_card, new_layout = self.make_card(
            "New Project",
            "Start a fresh MuFinder workspace with guided defaults.",
        )
        new_button = QPushButton("Create New Project", new_card)
        new_button.setIcon(icon("new_project"))
        new_button.clicked.connect(lambda _checked=False: self.new_requested.emit())
        new_layout.addWidget(new_button)
        self._new_project_button = new_button

        open_card, open_layout = self.make_card(
            "Open Project",
            "Continue from an existing .mufinder folder.",
        )
        open_button = QPushButton("Open Existing Project", open_card)
        open_button.setIcon(icon("open_project"))
        open_button.clicked.connect(lambda _checked=False: self.open_requested.emit())
        open_layout.addWidget(open_button)

        actions_grid.addWidget(new_card, 0, 0)
        actions_grid.addWidget(open_card, 0, 1)
        layout.addLayout(actions_grid)

        recent_card, recent_layout = self.make_card(
            "Recent Projects",
            "Jump back into your recent work.",
        )
        self.recent_projects_layout = QVBoxLayout()
        recent_layout.addLayout(self.recent_projects_layout)
        layout.addWidget(recent_card)
        layout.addStretch(1)
        return page

    def _build_project_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.project_home = ProjectHomeWidget(page)
        self.project_home.route_requested.connect(self.route_requested)
        layout.addWidget(self.project_home, 1)
        return page

    def refresh_recent_projects(self) -> None:
        """Rebuild the welcome page's recent entries, keeping keyboard focus among them."""

        focused_row: int | None = None
        focused_path: str | None = None
        row = 0
        while self.recent_projects_layout.count():
            item = self.recent_projects_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                if widget.hasFocus() and isinstance(widget, RecentProjectButton):
                    focused_row = row
                    focused_path = widget.project_path
                widget.deleteLater()
            row += 1

        recent = self._recent_projects.paths()[:_WELCOME_RECENT_PROJECTS]
        if not recent:
            empty = QLabel("No recent projects yet.")
            empty.setObjectName("cardBody")
            self.recent_projects_layout.addWidget(empty)
            if focused_row is not None:
                self._new_project_button.setFocus(Qt.FocusReason.OtherFocusReason)
            return

        for project_path in recent:
            button = RecentProjectButton(project_path, self)
            button.clicked.connect(
                lambda _checked=False, project_path=project_path: self.recent_requested.emit(
                    project_path
                )
            )
            self.recent_projects_layout.addWidget(button)

        # Focus stays with its project; if that entry has gone, its neighbour takes over.
        if focused_row is not None:
            row = (
                recent.index(focused_path)
                if focused_path in recent
                else min(focused_row, len(recent) - 1)
            )
            successor = self.recent_projects_layout.itemAt(row).widget()
            # The layout shows it only on its next pass, and a hidden widget takes no focus.
            successor.show()
            successor.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_project_name(self, name: str) -> None:
        """Name the open project ``name`` on the project home."""

        self.project_home.set_project_name(name)

    def set_document(
        self, document: ProjectDocument | None, summary: ProjectSummary | None = None
    ) -> None:
        """Show ``document`` on the project home, with its ``summary`` when already built."""

        self.project_home.set_document(document, summary)

    def show_project_page(self, has_project: bool) -> None:
        """Show the project home with a project open, else the welcome page."""

        self.content_stack.setCurrentWidget(self.project_page if has_project else self.welcome_page)
