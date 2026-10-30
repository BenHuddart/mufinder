"""Magnetic Fields muon-site selection dialog."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QStackedWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.pinned_sites_scope import PinnedSitesScopeBar
from mufinder.app.theme import magnetic_fields_stylesheet, style_hint_text
from mufinder.app.workflow_table_utils import create_workflow_table
from mufinder.domain.models import Structure
from mufinder.persistence.codec import float_triplet
from mufinder.science.magnetic_fields import muon_site_clearance_error
from mufinder.services.magnetic_field_results import (
    UEP_SITE_SOURCE,
    UNRELAXED_SITE_NOTE,
    is_unrelaxed_site_source,
)
from mufinder.services.magnetic_field_sites import (
    CUSTOM_SITE_SOURCE,
    RELAXATION_SITE_SOURCE,
    MagneticMuonSiteOption,
    muon_site_options,
    same_muon_site,
    site_cartesian_position,
    site_source_label,
)
from mufinder.services.muon_sites_summary import (
    MAGNETIC_FIELDS_SCOPE,
    NO_PINNED_SITES_NOTICE,
    has_pinned_sites,
    set_show_every_relaxed_site,
    show_every_relaxed_site,
)
from mufinder.services.project_service import ProjectDocument

_SITE_OPTION_ROLE = int(Qt.ItemDataRole.UserRole) + 41


class MuonSiteSelectionDialog(QDialog):
    """Dialog for choosing the muon site used by magnetic-field calculations.

    Structural-relaxation clusters list the sites pinned to the Muon Sites
    summary unless "Show every relaxed site" is on, which lists every analysis
    workspace's cluster representatives.
    """

    open_muon_sites_requested = Signal()

    def __init__(
        self,
        document: ProjectDocument | None,
        host_structure: Structure | None,
        current_site: MagneticMuonSiteOption | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Muon Site")
        self.resize(820, 420)
        self._document = document
        self._host_structure = host_structure
        self._site_options: list[MagneticMuonSiteOption] = []
        self._accepted_site: MagneticMuonSiteOption | None = None
        self._current_site = current_site

        self._site_source_combo = QComboBox()
        self._site_source_combo.addItem(site_source_label(UEP_SITE_SOURCE), UEP_SITE_SOURCE)
        self._site_source_combo.addItem(
            site_source_label(RELAXATION_SITE_SOURCE), RELAXATION_SITE_SOURCE
        )
        self._site_source_combo.addItem("Custom fractional position", CUSTOM_SITE_SOURCE)
        self._site_source_combo.currentIndexChanged.connect(self._refresh_muon_site_selection)
        self._estimate_note = QLabel(UNRELAXED_SITE_NOTE)
        self._estimate_note.setWordWrap(True)
        style_hint_text(self._estimate_note)

        self._site_table = create_workflow_table(
            ("Site", "Source", "Cluster", "fx", "fy", "fz"),
            selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
            stretch_last_section=False,
            accessible_name="Muon Sites",
        )
        self._site_table.itemSelectionChanged.connect(self._on_site_selection_changed)
        self._configure_site_table_columns()
        self._site_scope = PinnedSitesScopeBar(self)
        self._site_scope.show_every_site_toggled.connect(self._on_show_every_site_toggled)
        self._site_scope.open_muon_sites_requested.connect(self._on_open_muon_sites_requested)

        self._custom_x = DecimalLineEdit(text="0.0", minimum=0.0, maximum=1.0, placeholder="0.0")
        self._custom_y = DecimalLineEdit(text="0.0", minimum=0.0, maximum=1.0, placeholder="0.0")
        self._custom_z = DecimalLineEdit(text="0.0", minimum=0.0, maximum=1.0, placeholder="0.0")
        for edit in (self._custom_x, self._custom_y, self._custom_z):
            edit.textChanged.connect(lambda *_: self._refresh_site_status())
        self._site_stack = QStackedWidget()
        self._site_status = QLabel("Muon site: -")
        self._site_status.setWordWrap(True)
        style_hint_text(self._site_status)
        self._distortion_checkbox = QCheckBox(
            "Correct the local field for muon-induced displacements"
        )
        self._distortion_checkbox.setToolTip(
            "Adds the finite-cell change in the dipolar field, and in the contact field when a hyperfine "
            "coupling is set, from the host-ion displacements in the representative relaxed structure. "
            "Available for structural-relaxation clusters whose relaxed structure is present."
        )
        self._distortion_requested = bool(
            current_site is not None and current_site.include_distortion
        )
        self._distortion_checkbox.toggled.connect(self._on_distortion_toggled)

        self.setStyleSheet(magnetic_fields_stylesheet())
        self._build_layout()
        self._set_current_site(current_site)
        self._refresh_muon_site_selection()

    def selected_muon_site(self) -> MagneticMuonSiteOption | None:
        source = self._site_source()
        if source == CUSTOM_SITE_SOURCE:
            if self._host_structure is None:
                return None
            fractional = float_triplet(
                _decimal_edit_value(edit, label)
                for edit, label in (
                    (self._custom_x, "custom fx"),
                    (self._custom_y, "custom fy"),
                    (self._custom_z, "custom fz"),
                )
            )
            cartesian = site_cartesian_position(fractional, self._host_structure)
            return MagneticMuonSiteOption(
                source=CUSTOM_SITE_SOURCE,
                label="Custom site",
                fractional_position=fractional,
                cartesian_position=cartesian,
            )
        option = self._selected_site_option()
        if option is None:
            return None
        include = self._distortion_checkbox.isEnabled() and self._distortion_checkbox.isChecked()
        return replace(option, include_distortion=include)

    def accept(self) -> None:
        try:
            site = self.selected_muon_site()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Muon Site", str(exc))
            return
        if site is None:
            QMessageBox.warning(self, "Select Muon Site", "Select a muon site before continuing.")
            return
        if self._host_structure is not None:
            clearance_error = muon_site_clearance_error(
                self._host_structure, site.fractional_position, label=site.label
            )
            if clearance_error is not None:
                QMessageBox.warning(self, "Invalid Muon Site", clearance_error)
                return
        self._accepted_site = site
        super().accept()

    def accepted_site(self) -> MagneticMuonSiteOption | None:
        return self._accepted_site

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source"))
        source_row.addWidget(self._site_source_combo, 1)
        layout.addLayout(source_row)
        layout.addWidget(self._estimate_note)
        layout.addWidget(self._site_scope)

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self._site_table)
        self._site_stack.addWidget(table_page)

        custom_page = QWidget()
        custom_page.setObjectName("magneticMuonSiteInlinePanel")
        custom_layout = QHBoxLayout(custom_page)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.addWidget(QLabel("fx"))
        custom_layout.addWidget(self._custom_x)
        custom_layout.addWidget(QLabel("fy"))
        custom_layout.addWidget(self._custom_y)
        custom_layout.addWidget(QLabel("fz"))
        custom_layout.addWidget(self._custom_z)
        custom_layout.addStretch(1)
        self._site_stack.addWidget(custom_page)

        layout.addWidget(self._site_stack, 1)
        layout.addWidget(self._distortion_checkbox)
        layout.addWidget(self._site_status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_current_site(self, site: MagneticMuonSiteOption | None) -> None:
        if site is None:
            # Relaxed sites come before unrelaxed UEP estimates once the relaxation list has any.
            source = (
                RELAXATION_SITE_SOURCE
                if self._site_options_for_source(RELAXATION_SITE_SOURCE)
                else UEP_SITE_SOURCE
            )
            self._site_source_combo.setCurrentIndex(self._site_source_combo.findData(source))
            return
        index = self._site_source_combo.findData(site.source)
        if index >= 0:
            self._site_source_combo.setCurrentIndex(index)
        if site.source == CUSTOM_SITE_SOURCE:
            self._custom_x.setText(f"{site.fractional_position[0]:.8g}")
            self._custom_y.setText(f"{site.fractional_position[1]:.8g}")
            self._custom_z.setText(f"{site.fractional_position[2]:.8g}")

    def _site_source(self) -> str:
        return str(self._site_source_combo.currentData() or UEP_SITE_SOURCE)

    def _refresh_muon_site_selection(self) -> None:
        source = self._site_source()
        self._site_stack.setCurrentIndex(1 if source == CUSTOM_SITE_SOURCE else 0)
        self._estimate_note.setVisible(is_unrelaxed_site_source(source))
        self._site_options = self._site_options_for_source(source)
        self._refresh_site_scope(source)
        self._populate_site_table()
        self._sync_distortion_controls()
        self._refresh_site_status()

    def _site_options_for_source(self, source: str) -> list[MagneticMuonSiteOption]:
        return muon_site_options(
            self._document,
            self._host_structure,
            source,
            pinned_only=not show_every_relaxed_site(self._document, MAGNETIC_FIELDS_SCOPE),
        )

    def _refresh_site_scope(self, source: str) -> None:
        self._site_scope.setVisible(source == RELAXATION_SITE_SOURCE)
        show_every = show_every_relaxed_site(self._document, MAGNETIC_FIELDS_SCOPE)
        notice = "" if show_every or has_pinned_sites(self._document) else NO_PINNED_SITES_NOTICE
        self._site_scope.set_state(show_every_site=show_every, notice=notice)

    def _on_show_every_site_toggled(self, enabled: bool) -> None:
        set_show_every_relaxed_site(self._document, MAGNETIC_FIELDS_SCOPE, enabled)
        self._refresh_muon_site_selection()

    def _on_open_muon_sites_requested(self) -> None:
        self.open_muon_sites_requested.emit()
        self.reject()

    def _populate_site_table(self) -> None:
        self._site_table.setRowCount(0)
        for option in self._site_options:
            row = self._site_table.rowCount()
            self._site_table.insertRow(row)
            values = [
                option.label,
                option.source_label or site_source_label(option.source),
                option.cluster_label or "-",
                f"{option.fractional_position[0]:.5f}",
                f"{option.fractional_position[1]:.5f}",
                f"{option.fractional_position[2]:.5f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(_SITE_OPTION_ROLE, option)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._site_table.setItem(row, column, item)
        if self._site_table.rowCount():
            selected_row = 0
            if self._current_site is not None:
                for row, option in enumerate(self._site_options):
                    if same_muon_site(option, self._current_site):
                        selected_row = row
                        break
            self._site_table.selectRow(selected_row)
        self._site_table.resizeRowsToContents()

    def _selected_site_option(self) -> MagneticMuonSiteOption | None:
        selected = self._site_table.selectionModel().selectedRows()
        if not selected:
            return None
        row = selected[0].row()
        item = self._site_table.item(row, 0)
        option = item.data(_SITE_OPTION_ROLE) if item is not None else None
        return option if isinstance(option, MagneticMuonSiteOption) else None

    def _sync_distortion_controls(self) -> None:
        option = self._selected_site_option()
        enabled = (
            self._site_source() == RELAXATION_SITE_SOURCE
            and option is not None
            and option.distortion_structure_id is not None
        )
        self._distortion_checkbox.setEnabled(enabled)
        blocked = self._distortion_checkbox.blockSignals(True)
        self._distortion_checkbox.setChecked(enabled and self._distortion_requested)
        self._distortion_checkbox.blockSignals(blocked)

    def _on_distortion_toggled(self, checked: bool) -> None:
        self._distortion_requested = bool(checked)

    def _on_site_selection_changed(self) -> None:
        self._sync_distortion_controls()
        self._refresh_site_status()

    def _refresh_site_status(self) -> None:
        if self._host_structure is None:
            self._site_status.setText("Muon site: load a host structure first.")
            return
        source = self._site_source()
        if source == CUSTOM_SITE_SOURCE:
            self._site_status.setText(
                "Muon site: custom fractional coordinate in the active host cell."
            )
            return
        option = self._selected_site_option()
        if option is None:
            self._site_status.setText(
                f"Muon site: no {site_source_label(source).lower()} sites available."
            )
            return
        distortion_text = (
            " Distortion corrections are available for this representative."
            if option.distortion_structure_id is not None
            else ""
        )
        self._site_status.setText(
            f"Muon site: {option.label} at ({option.fractional_position[0]:.4f}, "
            f"{option.fractional_position[1]:.4f}, {option.fractional_position[2]:.4f}).{distortion_text}"
        )

    def _configure_site_table_columns(self) -> None:
        header = self._site_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for column in (3, 4, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)


def _decimal_edit_value(edit: QLineEdit, label: str) -> float:
    text = edit.text().strip()
    if not text:
        raise ValueError(f"{label} is required")
    return float(text)
