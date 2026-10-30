"""Overview page for the parent Muon Sites workflow.

Shows the headline result of a muon-site calculation: the distinct,
symmetry-inequivalent muon stopping sites for each charge state, pinned from a
structural-relaxation analysis workspace. Each charge-state panel renders a
ranked table (relative energy, fractional position, Wyckoff label) plus a 3D
preview of the host with the pinned muons overlaid as pickable ``H:mu`` atoms.

The pinned snapshot is a decoupled copy (mirroring the convergence overview), so
the panels also act as a launch pad back into the live analysis: the "Open in
Analysis" button, a table-row click, or a click on a muon in the 3D preview all
deep-link to the source workspace (and, for row/atom clicks, the specific
cluster). ``main_window`` handles the stale-snapshot fallback when the source
workspace or cluster no longer exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from html import escape


from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.export_dialogs import (
    CIF_FORMAT,
    MUON_HOST_OVERLAP_TITLE,
    ask_save_path,
    report_export_failed,
    report_export_saved,
)
from mufinder.app.theme import (
    make_empty_state_label,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
)
from mufinder.app.workflow_table_utils import create_workflow_table
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.models import Structure
from mufinder.science.structure_io import CifMuonSite, export_structure_with_muon_sites
from mufinder.services.muon_sites_summary import (
    grouping_method_text,
    grouping_symmetry_text,
    summary_export_header_lines,
)

_MUON_SPECIES_LABEL = "H:mu"
_CLUSTER_ROLE = Qt.ItemDataRole.UserRole


def _charge_state_title(charge_state: str) -> str:
    return "Muonium" if charge_state == MUONIUM_CHARGE_STATE else "Positive muon"


def _format_energy(value: object) -> str:
    try:
        return f"{float(value):.5f}"
    except (TypeError, ValueError):
        return "—"


def _format_fractional(position: object) -> str:
    if not isinstance(position, (list, tuple)) or len(position) != 3:
        return "—"
    try:
        return f"({float(position[0]):.4f}, {float(position[1]):.4f}, {float(position[2]):.4f})"
    except (TypeError, ValueError):
        return "—"


def _host_signature(host_structure: Structure | None) -> object:
    """Return a value identifying the host geometry rendered in the preview.

    Keyed by the fields that actually affect the drawn scene (lattice, species,
    positions) rather than only the structure id, so an in-place host edit
    invalidates the render cache.
    """

    if host_structure is None:
        return None
    return (
        str(host_structure.id),
        tuple(tuple(row) for row in host_structure.lattice_vectors),
        tuple(host_structure.species),
        tuple(tuple(position) for position in host_structure.fractional_positions),
    )


def _snapshot_sites(snapshot: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    if not isinstance(snapshot, Mapping):
        return []
    raw = snapshot.get("sites")
    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, Mapping)]


class _ChargeStatePanel(QFrame):
    """Notebook-style card for one charge state's pinned muon sites."""

    _SITE_COLUMN = 0
    _ENERGY_COLUMN = 1
    _POSITION_COLUMN = 2
    _WYCKOFF_COLUMN = 3

    open_analysis_requested = Signal(str)
    open_analysis_root_requested = Signal()
    focus_site_requested = Signal(str, int)
    unpin_requested = Signal(str)

    _OPEN_PINNED_TEXT = "Open in Analysis →"
    _OPEN_EMPTY_TEXT = "Go to Analysis →"

    def __init__(self, charge_state: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("overviewCard")
        self._charge_state = charge_state
        self._workspace_id = ""
        # Ordered map from a muon's atom index in the preview structure to its
        # cluster index, so a pick in the 3D view deep-links to the right site.
        self._muon_atom_cluster: dict[int, int] = {}
        # Host structure and pinned sites for the currently rendered snapshot,
        # kept so "Export CIF" can rebuild the combined structure on demand
        # without re-deriving it from the preview viewer.
        self._host_structure: Structure | None = None
        self._sites: list[Mapping[str, object]] = []
        self._export_header_lines: list[str] = []
        # Signature of the currently rendered snapshot; lets repeated set_snapshot
        # calls (e.g. re-navigating to the page) skip an expensive viewer rebuild
        # when nothing changed. None whenever the placeholder is showing.
        self._render_signature: object | None = None

        self._title = QLabel(_charge_state_title(charge_state), self)
        self._title.setObjectName("overviewCardTitle")

        # Always enabled: deep-links to the pinned workspace when populated, and
        # otherwise routes to the Analysis tab so an empty panel still guides the
        # user to their next step rather than presenting a dead disabled control.
        self._open_button = QPushButton(self._OPEN_EMPTY_TEXT, self)
        self._open_button.setObjectName("overviewActionButton")
        self._open_button.clicked.connect(self._on_open_clicked)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(12)
        header_row.addWidget(self._title)
        header_row.addStretch(1)
        header_row.addWidget(self._open_button, alignment=Qt.AlignmentFlag.AlignTop)

        self._guidance = QLabel(
            "Pin a structural-relaxation analysis from the Analysis tab to summarise "
            "its distinct muon sites here.",
            self,
        )
        self._guidance.setWordWrap(True)
        style_hint_text(self._guidance)

        self._summary_layout = QVBoxLayout()
        self._summary_layout.setContentsMargins(0, 0, 0, 0)
        self._summary_layout.setSpacing(6)
        self._summary_labels: dict[str, QLabel] = {}

        self._placeholder = make_empty_state_label(object_name="overviewPlaceholder", parent=self)

        # Columns other than _POSITION_COLUMN are sized to content via
        # resizeColumnsToContents() in _populate_table, so this table opts out
        # of the shared stretch-to-fill default (default_resize_mode=Interactive)
        # and instead stretches only the fractional-position column explicitly.
        self._table = create_workflow_table(
            ("Site", "dE (eV)", "Fractional position", "Wyckoff"),
            selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
            corner_style="tabbed",
            default_resize_mode=QHeaderView.ResizeMode.Interactive,
            stretch_last_section=False,
            accessible_name="Muon Sites",
        )
        self._table.horizontalHeader().setSectionResizeMode(
            self._POSITION_COLUMN, QHeaderView.ResizeMode.Stretch
        )
        self._table.itemClicked.connect(self._on_table_item_clicked)

        self._viewer = AtomicStructureViewer(self)
        self._viewer.setMinimumHeight(240)
        self._viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._viewer.atom_selected.connect(self._on_viewer_atom_selected)

        # The viewer paints the app-wide viewer background, so give it a matching
        # container (rather than the dashed tan overview zone) to avoid a lighter
        # rectangle seaming against a darker frame.
        self._viewer_frame = QFrame(self)
        self._viewer_frame.setObjectName("muonSummaryViewerFrame")
        viewer_frame_layout = QVBoxLayout(self._viewer_frame)
        viewer_frame_layout.setContentsMargins(1, 1, 1, 1)
        viewer_frame_layout.addWidget(self._viewer)

        # A plain QWidget fills with the palette Window colour (the mid beige),
        # not the card's lighter QSS background, which would tint the gap between
        # the table and the viewer frame. Keep it transparent so the card shows
        # through and the spacing matches the surrounding beige.
        self._content = QWidget(self)
        self._content.setObjectName("muonSummaryPanelBody")
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.addWidget(self._table, stretch=1)
        content_layout.addWidget(self._viewer_frame, stretch=2)

        self._export_cif_button = QPushButton("Export CIF…", self)
        self._export_cif_button.setObjectName("overviewSecondaryButton")
        self._export_cif_button.clicked.connect(self._on_export_cif_clicked)

        self._unpin_button = QPushButton("Unpin", self)
        self._unpin_button.setObjectName("overviewSecondaryButton")
        self._unpin_button.clicked.connect(lambda: self.unpin_requested.emit(self._charge_state))
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.addStretch(1)
        footer_row.addWidget(self._export_cif_button)
        footer_row.addWidget(self._unpin_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addLayout(header_row)
        layout.addWidget(self._guidance)
        layout.addLayout(self._summary_layout)
        layout.addWidget(self._placeholder)
        layout.addWidget(self._content, stretch=1)
        layout.addLayout(footer_row)

        self._show_placeholder("Nothing pinned yet.")

    def _set_summary_rows(self, rows: Sequence[tuple[str, str]]) -> None:
        seen = {key for key, _value in rows}
        for key in list(self._summary_labels):
            if key in seen:
                continue
            label = self._summary_labels.pop(key)
            self._summary_layout.removeWidget(label)
            label.deleteLater()
        for key, value in rows:
            label = self._summary_labels.get(key)
            if label is None:
                label = QLabel(self)
                label.setWordWrap(True)
                label.setObjectName("overviewSummaryValue")
                self._summary_labels[key] = label
            label.setText(f"<b>{escape(key)}:</b> {escape(value)}")
            self._summary_layout.addWidget(label)

    def _show_placeholder(self, message: str) -> None:
        self._workspace_id = ""
        self._muon_atom_cluster = {}
        self._render_signature = None
        self._host_structure = None
        self._sites = []
        self._export_header_lines = []
        self._table.setRowCount(0)
        self._viewer.set_structure(None)
        self._set_summary_rows([])
        self._placeholder.setText(message)
        self._placeholder.show()
        self._content.hide()
        # Empty panels still offer a route into the Analysis tab.
        self._open_button.setText(self._OPEN_EMPTY_TEXT)
        self._open_button.setEnabled(True)
        self._unpin_button.setEnabled(False)
        self._export_cif_button.setEnabled(False)

    def set_snapshot(
        self,
        snapshot: Mapping[str, object] | None,
        *,
        host_structure: Structure | None,
        placeholder_message: str,
    ) -> None:
        sites = _snapshot_sites(snapshot)
        if not isinstance(snapshot, Mapping) or not sites:
            self._show_placeholder(placeholder_message)
            return

        signature = (
            str(snapshot.get("workspace_id", "")).strip(),
            # Include the workspace name so a rename (same sites) still refreshes
            # the "Source analysis" summary row rather than hitting the early return.
            str(snapshot.get("workspace_name", "")).strip(),
            repr(snapshot.get("grouping")),
            # Fingerprint the host geometry that feeds the preview, not just its
            # id: a host edited in place (same id, new lattice/species/positions)
            # must invalidate the cache so the 3D view is not left stale.
            _host_signature(host_structure),
            tuple(
                (
                    site.get("cluster_index"),
                    tuple(site.get("fractional_position") or ()),
                    site.get("relative_energy_ev"),
                    site.get("wyckoff_label"),
                )
                for site in sites
            ),
        )
        if signature == self._render_signature:
            return
        self._render_signature = signature

        self._workspace_id = str(snapshot.get("workspace_id", "")).strip()
        workspace_name = str(snapshot.get("workspace_name", "")).strip() or "Analysis workspace"
        self._host_structure = host_structure
        self._sites = list(sites)
        self._export_header_lines = summary_export_header_lines(snapshot)

        self._placeholder.hide()
        self._content.show()
        self._open_button.setText(self._OPEN_PINNED_TEXT)
        self._open_button.setEnabled(True)
        self._unpin_button.setEnabled(True)
        self._export_cif_button.setEnabled(
            host_structure is not None and bool(host_structure.species)
        )

        rows = [
            ("Source analysis", workspace_name),
            ("Distinct sites", str(len(sites))),
        ]
        grouping = snapshot.get("grouping")
        if isinstance(grouping, Mapping):
            rows.append(("Grouping", grouping_method_text(grouping)))
            symmetry = grouping_symmetry_text(grouping)
            if symmetry:
                rows.append(("Symmetry", symmetry))
        self._set_summary_rows(rows)
        self._populate_table(sites)
        self._populate_viewer(sites, host_structure)

    def _populate_table(self, sites: Sequence[Mapping[str, object]]) -> None:
        self._table.setRowCount(len(sites))
        for row, site in enumerate(sites):
            try:
                cluster_index = int(site.get("cluster_index", 0))
            except (TypeError, ValueError):
                cluster_index = 0
            wyckoff = str(site.get("wyckoff_label", "")).strip() or "—"
            values = [
                str(row + 1),
                _format_energy(site.get("relative_energy_ev")),
                _format_fractional(site.get("fractional_position")),
                wyckoff,
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(_CLUSTER_ROLE, cluster_index)
                if column != self._POSITION_COLUMN:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
                    )
                self._table.setItem(row, column, item)
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setSectionResizeMode(
            self._POSITION_COLUMN, QHeaderView.ResizeMode.Stretch
        )

    def _populate_viewer(
        self,
        sites: Sequence[Mapping[str, object]],
        host_structure: Structure | None,
    ) -> None:
        self._muon_atom_cluster = {}
        if host_structure is None or not host_structure.species:
            self._viewer.set_structure(None)
            return
        species = list(host_structure.species)
        positions = [list(pos) for pos in host_structure.fractional_positions]
        base_index = len(species)
        muon_offset = 0
        for site in sites:
            fractional = site.get("fractional_position")
            if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                continue
            try:
                # Use the coordinates verbatim (the analysis already folds cluster
                # representatives into the cell) so the rendered muon matches the
                # fractional position printed in the table row.
                coords = [float(fractional[0]), float(fractional[1]), float(fractional[2])]
            except (TypeError, ValueError):
                continue
            try:
                cluster_index = int(site.get("cluster_index", 0))
            except (TypeError, ValueError):
                cluster_index = 0
            species.append(_MUON_SPECIES_LABEL)
            positions.append(coords)
            self._muon_atom_cluster[base_index + muon_offset] = cluster_index
            muon_offset += 1
        preview = Structure(
            label="muon_sites_summary",
            origin="derived",
            lattice_vectors=[list(vector) for vector in host_structure.lattice_vectors],
            species=species,
            fractional_positions=positions,
        )
        self._viewer.set_structure(preview)

    def _on_export_cif_clicked(self) -> None:
        host_structure = self._host_structure
        if host_structure is None or not self._sites:
            return

        muon_sites: list[CifMuonSite] = []
        for index, site in enumerate(self._sites):
            fractional = site.get("fractional_position")
            if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                continue
            try:
                position = (float(fractional[0]), float(fractional[1]), float(fractional[2]))
            except (TypeError, ValueError):
                continue
            energy = site.get("relative_energy_ev")
            try:
                relative_energy_ev = float(energy) if energy is not None else None
            except (TypeError, ValueError):
                relative_energy_ev = None
            wyckoff_label = str(site.get("wyckoff_label", "")).strip() or None
            muon_sites.append(
                CifMuonSite(
                    label=f"Mu{index + 1}",
                    fractional_position=position,
                    relative_energy_ev=relative_energy_ev,
                    wyckoff_label=wyckoff_label,
                )
            )
        if not muon_sites:
            return

        default_name = (
            f"{_charge_state_title(self._charge_state).lower().replace(' ', '_')}_muon_sites.cif"
        )
        file_path = ask_save_path(self, "Export Muon Sites CIF", default_name, (CIF_FORMAT,))
        if file_path is None:
            return

        try:
            proximity_warnings = export_structure_with_muon_sites(
                host_structure, muon_sites, file_path, header_lines=self._export_header_lines
            )
        except OSError as exc:
            report_export_failed(self, exc)
            return
        report_export_saved(
            self,
            file_path,
            proximity_warnings,
            warning_title=MUON_HOST_OVERLAP_TITLE,
        )

    def _on_open_clicked(self) -> None:
        if self._workspace_id:
            self.open_analysis_requested.emit(self._workspace_id)
        else:
            self.open_analysis_root_requested.emit()

    def _on_table_item_clicked(self, item: QTableWidgetItem | None) -> None:
        if item is None or not self._workspace_id:
            return
        cluster_index = item.data(_CLUSTER_ROLE)
        try:
            self.focus_site_requested.emit(self._workspace_id, int(cluster_index))
        except (TypeError, ValueError):
            self.open_analysis_requested.emit(self._workspace_id)

    def _on_viewer_atom_selected(self, atom_index: int) -> None:
        if not self._workspace_id:
            return
        cluster_index = self._muon_atom_cluster.get(int(atom_index))
        if cluster_index is None:
            return
        self.focus_site_requested.emit(self._workspace_id, int(cluster_index))


class MuonSitesSummaryWidget(QWidget):
    """Parent workflow overview summarising pinned muon sites per charge state."""

    open_analysis_requested = Signal(str)
    open_analysis_root_requested = Signal()
    focus_site_requested = Signal(str, int)
    unpin_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scroll = QScrollArea(self)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._content = QWidget(self._scroll)
        self._content.setObjectName("muonSitesSummaryContent")
        self._scroll.setWidget(self._content)

        self._title = QLabel("Muon Sites", self._content)
        style_section_title(self._title)

        self._subtitle = QLabel(
            "The candidate muon stopping sites are the headline result of the muon-site "
            "workflow. Cluster completed relaxations in the Analysis tab, then pin an "
            "analysis to summarise its distinct, symmetry-inequivalent sites here.",
            self._content,
        )
        self._subtitle.setWordWrap(True)
        style_section_subtitle(self._subtitle)

        self._hint = QLabel(
            "Each panel is populated by pinning an analysis workspace of the matching "
            "charge state. Click a site row or a muon in the preview to jump to it.",
            self._content,
        )
        self._hint.setWordWrap(True)
        style_hint_text(self._hint)

        self._panels: dict[str, _ChargeStatePanel] = {}
        cards_row = QHBoxLayout()
        cards_row.setSpacing(18)
        cards_row.setAlignment(Qt.AlignmentFlag.AlignTop)
        for charge_state in (POSITIVE_MUON_CHARGE_STATE, MUONIUM_CHARGE_STATE):
            panel = _ChargeStatePanel(charge_state, parent=self._content)
            panel.open_analysis_requested.connect(self.open_analysis_requested.emit)
            panel.open_analysis_root_requested.connect(self.open_analysis_root_requested.emit)
            panel.focus_site_requested.connect(self.focus_site_requested.emit)
            panel.unpin_requested.connect(self.unpin_requested.emit)
            self._panels[charge_state] = panel
            cards_row.addWidget(panel, stretch=1)

        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(12)
        content_layout.addWidget(self._title)
        content_layout.addWidget(self._subtitle)
        content_layout.addWidget(self._hint)
        content_layout.addLayout(cards_row)
        content_layout.addStretch(1)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(self._scroll)

    def set_panel(
        self,
        charge_state: str,
        *,
        snapshot: Mapping[str, object] | None,
        host_structure: Structure | None,
        placeholder_message: str,
    ) -> None:
        panel = self._panels.get(charge_state)
        if panel is None:
            return
        panel.set_snapshot(
            snapshot,
            host_structure=host_structure,
            placeholder_message=placeholder_message,
        )
