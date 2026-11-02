"""The "pinned sites / every relaxed site" control shared by downstream workflows.

Quantum corrections, energy barriers and the Magnetic Fields muon-site dialog
list the sites pinned to the Muon Sites summary by default. This bar holds the
"Show every relaxed site" check box that widens the list to every eligible
relaxed site, and a notice with a "Go to Muon Sites →" link for when a charge
state has no pin (or a pinned site no longer resolves). A page that shows its
flow as steps keeps the link in view beside the check box instead, as "Change on
Muon Sites →", with the notice below them. The choice itself is stored per
workflow by :mod:`mufinder.services.muon_sites_summary`.
"""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from mufinder.app.theme import form_spacing, style_hint_text

SHOW_EVERY_RELAXED_SITE_TEXT = "Show every relaxed site"
GO_TO_MUON_SITES_TEXT = "Go to Muon Sites →"
CHANGE_ON_MUON_SITES_TEXT = "Change on Muon Sites →"


class PinnedSitesScopeBar(QWidget):
    """Check box, notice and Muon Sites link above a downstream site list."""

    show_every_site_toggled = Signal(bool)
    open_muon_sites_requested = Signal()

    def __init__(self, parent: QWidget | None = None, *, link_always_shown: bool = False) -> None:
        super().__init__(parent)
        self.setObjectName("pinnedSitesScopeBar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._checkbox = QCheckBox(SHOW_EVERY_RELAXED_SITE_TEXT, self)
        self._checkbox.setToolTip(
            "Off: list the distinct sites pinned to the Muon Sites summary.\n"
            "On: list every eligible relaxed site, including relaxed copies of the same site."
        )
        self._checkbox.toggled.connect(self.show_every_site_toggled)

        notice_row = QHBoxLayout()
        notice_row.setContentsMargins(0, 0, 0, 0)
        notice_row.setSpacing(8)
        self._notice = QLabel("", self)
        self._notice.setWordWrap(True)
        style_hint_text(self._notice)
        notice_row.addWidget(self._notice, 1)
        self._link = QPushButton(
            CHANGE_ON_MUON_SITES_TEXT if link_always_shown else GO_TO_MUON_SITES_TEXT, self
        )
        self._link.clicked.connect(self.open_muon_sites_requested)
        if link_always_shown:
            controls = QHBoxLayout()
            controls.setContentsMargins(0, 0, 0, 0)
            controls.setSpacing(form_spacing())
            controls.addWidget(self._link)
            controls.addWidget(self._checkbox)
            controls.addStretch(1)
            layout.addLayout(controls)
        else:
            self._link.setObjectName("overviewActionButton")
            layout.addWidget(self._checkbox)
            notice_row.addWidget(self._link)
        self._notice_row = QWidget(self)
        self._notice_row.setLayout(notice_row)
        layout.addWidget(self._notice_row)
        self._notice_row.setVisible(False)

    def set_state(self, *, show_every_site: bool, notice: str = "") -> None:
        """Show the stored choice and, when non-empty, the pinned-sites notice.

        The notice only applies to the pinned list, so it is hidden while every
        relaxed site is shown. Setting the state does not emit
        :attr:`show_every_site_toggled`.
        """

        with QSignalBlocker(self._checkbox):
            self._checkbox.setChecked(bool(show_every_site))
        self._notice.setText(notice)
        self._notice_row.setVisible(bool(notice) and not show_every_site)

    def show_every_site(self) -> bool:
        return self._checkbox.isChecked()

    def notice_text(self) -> str:
        return self._notice.text() if self._notice_row.isVisibleTo(self) else ""
