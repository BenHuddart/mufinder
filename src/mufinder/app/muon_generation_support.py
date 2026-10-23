"""Main-window helpers for the muon site generation page.

``MuonGenerationThread`` runs candidate-site generation off the Qt main thread,
and ``MuonChargeStateDialog`` asks which charge states the relaxation jobs
created from the selected sites should use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import style_section_subtitle, style_section_title
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.science.site_generation import SiteGenerationError

if TYPE_CHECKING:
    from mufinder.services.project_service import ProjectDocument, ProjectService


class MuonChargeStateDialog(QDialog):
    """Choose which muon charge state tabs receive generated positions."""

    POSITIVE_ONLY = "positive"
    MUONIUM_ONLY = "muonium"
    BOTH = "both"

    def __init__(self, *, site_count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("muonChargeStateDialog")
        self.setWindowTitle("Create Muon Relaxation Jobs")
        self.resize(480, 280)

        self._button_group = QButtonGroup(self)
        self._button_group.setExclusive(True)
        self._positive = QRadioButton("Positive muon", self)
        self._positive.setProperty("chargeStateOption", "true")
        self._positive.setChecked(True)
        self._muonium = QRadioButton("Muonium", self)
        self._muonium.setProperty("chargeStateOption", "true")
        self._both = QRadioButton("Both charge states", self)
        self._both.setProperty("chargeStateOption", "true")
        for index, button in enumerate((self._positive, self._muonium, self._both)):
            self._button_group.addButton(button, index)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(10)

        title = QLabel("Choose Charge State", self)
        style_section_title(title)
        layout.addWidget(title)

        noun = "position" if site_count == 1 else "positions"
        subtitle = QLabel(
            f"Create relaxation jobs for {site_count} selected initial muon {noun}.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(subtitle)

        card = QWidget(self)
        card.setObjectName("muonChargeStateCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(8)
        card_layout.addWidget(self._positive)
        card_layout.addWidget(self._muonium)
        card_layout.addWidget(self._both)
        layout.addWidget(card)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Create Jobs")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_charge_states(self) -> tuple[str, ...]:
        if self._both.isChecked():
            return (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE)
        if self._muonium.isChecked():
            return (MUONIUM_CHARGE_STATE,)
        return (POSITIVE_MUON_CHARGE_STATE,)


class MuonGenerationThread(QThread):
    """Run site generation away from the Qt main thread."""

    generated = Signal(object)
    failed = Signal(str)
    progress_changed = Signal(int, int)

    def __init__(
        self,
        service: ProjectService,
        document: ProjectDocument,
        *,
        method: str,
        parameters: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._service = service
        self._document = document
        self._method = method
        self._parameters = dict(parameters)

    @property
    def method(self) -> str:
        return self._method

    @property
    def parameters(self) -> dict:
        return dict(self._parameters)

    def run(self) -> None:
        try:
            created = self._service.generate_candidate_sites(
                self._document,
                method=self._method,
                parameters=self._parameters,
                progress_callback=self.progress_changed.emit,
            )
        except (SiteGenerationError, ValueError) as exc:
            self.failed.emit(str(exc))
            return
        self.generated.emit(created)
