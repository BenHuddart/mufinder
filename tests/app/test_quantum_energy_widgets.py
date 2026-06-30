from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox, QPushButton

from mufinder.app.theme import status_brush
from mufinder.app.energy_barriers_widget import (
    EnergyBarrierEndpointRow,
    EnergyBarrierPreviewImageRow,
    EnergyBarrierSourceRow,
    EnergyBarriersWidget,
)
from mufinder.app.quantum_corrections_widget import QuantumCorrectionsWidget, QuantumResultRow, QuantumSourceRow
from mufinder.domain.models import Structure


def _sample_structure() -> Structure:
    return Structure(
        label="preview",
        origin="energy_barriers_neb_preview",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )


def test_quantum_widget_emits_configure_for_source_row(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(
                key="site-a",
                label="Site A",
                charge_state="Muonium",
                relative_energy="0.0 eV",
                displacements="9",
                source_job="relax_a",
                stage="singlepoint",
                num_farms_display="",
                status="ready",
            )
        ]
    )

    configured: list[str] = []
    widget.configure_requested.connect(configured.append)

    button = widget._sources_table.cellWidget(0, 8)
    assert isinstance(button, QPushButton)
    assert button.text() == "Configure..."

    button.click()

    assert configured == [["site-a"]]


def test_quantum_widget_batch_configure_emits_selected_source_keys(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(
                key="site-a",
                label="Site A",
                charge_state="Muonium",
                relative_energy="0.0 eV",
                displacements="9",
                source_job="relax_a",
                stage="singlepoint",
                num_farms_display="",
                status="ready",
            ),
            QuantumSourceRow(
                key="site-b",
                label="Site B",
                charge_state="Muonium",
                relative_energy="0.1 eV",
                displacements="12",
                source_job="relax_b",
                stage="singlepoint",
                num_farms_display="",
                status="ready",
            ),
        ]
    )
    configured: list[object] = []
    widget.configure_requested.connect(configured.append)

    widget._sources_table.selectRow(0)
    widget._sources_table.selectRow(1)
    widget._emit_batch_configure()

    assert configured == [["site-b"]]


def test_quantum_widget_uses_shared_status_text_colors(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(
                key="site-a",
                label="Site A",
                charge_state="Muonium",
                relative_energy="0.0 eV",
                displacements="9",
                source_job="relax_a",
                stage="phonon",
                num_farms_display="4",
                status="running",
            )
        ]
    )

    status_item = widget._sources_table.item(0, 7)

    assert status_item is not None
    assert status_item.foreground().color().name() == status_brush("running").color().name()


def test_quantum_widget_emits_reset_to_stage_one_for_selected_rows(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(
                key="site-a",
                label="Site A",
                charge_state="Muonium",
                relative_energy="0.0 eV",
                displacements="9",
                source_job="relax_a",
                stage="phonon",
                num_farms_display="4",
                status="ready",
                resettable=True,
            )
        ]
    )
    widget._sources_table.selectRow(0)

    reset: list[list[str]] = []
    widget.reset_stage_requested.connect(reset.append)

    widget._reset_stage_button.click()

    assert reset == [["site-a"]]


def test_quantum_widget_emits_analysis_for_completed_result(qapp) -> None:
    widget = QuantumCorrectionsWidget()
    widget.set_results(
        [
            QuantumResultRow(
                key="result-a",
                label="phonon_a",
                zpe="0.1 eV",
                eigenvalues="1:100.0, 2:400.0",
                witness="-0.1",
            )
        ]
    )

    launched: list[str] = []
    widget.analysis_requested.connect(launched.append)

    assert widget._results_table.horizontalHeaderItem(2).text() == "Eigenvalues (cm⁻¹)"
    assert widget._results_table.columnWidth(2) >= widget._results_table.columnWidth(3)

    button = widget._results_table.cellWidget(0, 4)
    assert isinstance(button, QPushButton)
    assert button.text() == "Launch Analysis"

    button.click()

    assert launched == ["result-a"]


def test_energy_barriers_widget_emits_configure_and_view_for_source_row(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="ready",
            )
        ]
    )

    configured: list[str] = []
    viewed: list[str] = []
    widget.configure_result_requested.connect(configured.append)
    widget.view_source_requested.connect(viewed.append)

    configure_button = widget._sources_table.cellWidget(0, 5)
    view_button = widget._sources_table.cellWidget(0, 6)
    assert isinstance(configure_button, QPushButton)
    assert isinstance(view_button, QPushButton)
    assert configure_button.text() == "Configure..."
    assert view_button.text() == "View..."

    configure_button.click()
    view_button.click()

    assert configured == [["path-a"]]
    assert viewed == ["path-a"]


def test_energy_barriers_widget_uses_shared_status_text_colors(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="failed",
            )
        ]
    )

    # NEB Status is column 4 in the Diffusion Paths table; it uses the shared per-status palette.
    status_item = widget._sources_table.item(0, 4)
    assert status_item is not None
    assert status_item.foreground().color().name() == status_brush("failed").color().name()


def test_energy_barriers_widget_batch_configure_emits_selected_source_keys(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="ready",
            ),
            EnergyBarrierSourceRow(
                key="path-b",
                reactant="B",
                product="C",
                charge_state="Muonium",
                path_length="1.4 ang",
                neb_status="ready",
            ),
        ]
    )
    configured: list[object] = []
    widget.configure_result_requested.connect(configured.append)

    widget._sources_table.selectRow(0)
    widget._emit_batch_configure_results()

    assert configured == [["path-a"]]


def test_energy_barriers_widget_emits_generate_paths_with_max_length(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_endpoint_options(
        [
            EnergyBarrierEndpointRow("site-a", "A", "Muonium", "relax_a", "0.0 eV"),
            EnergyBarrierEndpointRow("site-b", "B", "Muonium", "relax_b", "0.1 eV"),
        ],
        reactant_key="site-a",
        product_key="site-b",
        max_path_length_text="2.5",
    )
    widget._max_path_length_input.setText("2.5")

    generated: list[tuple[str, str, str]] = []
    widget.generate_paths_requested.connect(lambda reactant, product, max_length: generated.append((reactant, product, max_length)))

    widget._emit_generate_paths()

    assert generated == [("site-a", "site-b", "2.5")]


def test_energy_barriers_widget_delete_shortcuts_include_child_focus(qapp) -> None:
    widget = EnergyBarriersWidget()

    assert widget._delete_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut
    assert widget._backspace_shortcut.context() == Qt.ShortcutContext.WidgetWithChildrenShortcut


def test_energy_barriers_widget_delete_key_removes_selected_source_from_child_focus(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="A",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="ready",
            ),
            EnergyBarrierSourceRow(
                key="path-b",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.8 ang",
                neb_status="ready",
            ),
        ]
    )
    removed: list[list[str]] = []
    widget.remove_sources_requested.connect(lambda keys: removed.append(list(keys)))
    configure_button = widget._sources_table.cellWidget(0, 5)
    assert isinstance(configure_button, QPushButton)

    widget.show()
    widget._sources_table.selectRow(0)
    configure_button.setFocus()
    qapp.processEvents()
    QTest.keyClick(configure_button, Qt.Key.Key_Delete)

    assert removed == [["path-a"]]
    assert widget._sources_table.rowCount() == 1
    assert widget._source_rows[0].key == "path-b"


def test_energy_barriers_widget_marks_configured_path_row(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(
                key="path-a",
                reactant="A",
                product="B",
                charge_state="Muonium",
                path_length="1.2 ang",
                neb_status="ready",
                configured=True,
            )
        ]
    )

    configured: list[str] = []
    widget.configure_result_requested.connect(configured.append)

    configure_button = widget._sources_table.cellWidget(0, 5)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured..."

    configure_button.click()

    assert configured == [["path-a"]]


def test_energy_barriers_widget_switches_preview_splitter_for_narrow_widths(qapp) -> None:
    widget = EnergyBarriersWidget()

    widget.resize(1100, 800)
    widget._apply_responsive_layout()
    assert widget._preview_splitter.orientation() == Qt.Orientation.Horizontal

    widget.resize(1500, 800)
    widget._apply_responsive_layout()
    assert widget._preview_splitter.orientation() == Qt.Orientation.Horizontal


def test_energy_barriers_widget_uses_slider_only_for_neb_preview_navigation(qapp) -> None:
    widget = EnergyBarriersWidget()

    widget.set_neb_preview(
        title="NEB preview",
        image_rows=[
            EnergyBarrierPreviewImageRow("REA", "0", "-1 eV", 0.0, -1.0),
            EnergyBarrierPreviewImageRow("TST", "0.5", "-0.8 eV", 0.5, -0.8),
            EnergyBarrierPreviewImageRow("PRO", "1", "-0.9 eV", 1.0, -0.9),
        ],
        structures=[_sample_structure(), _sample_structure(), _sample_structure()],
    )

    assert not hasattr(widget, "_preview_table")
    assert widget._preview_slider.maximum() == 2
    assert widget._preview_slider_start_label.text() == "0"
    assert widget._preview_slider_end_label.text() == "1"

    widget._preview_slider.setValue(1)

    assert "TST" in widget._preview_summary.text()


def test_energy_barriers_source_configure_applies_to_snapshot_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(key="path-a", reactant="A", product="B", charge_state="M", path_length="1.0", neb_status="ready"),
            EnergyBarrierSourceRow(key="path-b", reactant="B", product="C", charge_state="M", path_length="1.2", neb_status="ready"),
        ]
    )
    configured: list[object] = []
    widget.configure_result_requested.connect(configured.append)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))

    btn = widget._sources_table.cellWidget(0, 5)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0, 1)
    btn.clicked.emit()

    assert len(configured) == 1
    assert set(configured[0]) == {"path-a", "path-b"}


def test_energy_barriers_source_configure_single_row_no_prompt(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = EnergyBarriersWidget()
    widget.set_sources(
        [EnergyBarrierSourceRow(key="path-a", reactant="A", product="B", charge_state="M", path_length="1.0", neb_status="ready")]
    )
    configured: list[object] = []
    widget.configure_result_requested.connect(configured.append)
    prompted: list[bool] = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: prompted.append(True) or QMessageBox.StandardButton.Yes))

    btn = widget._sources_table.cellWidget(0, 5)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0,)
    btn.clicked.emit()

    assert configured == [["path-a"]]
    assert prompted == []


def test_energy_barriers_view_shows_confirm_for_multi_row_snapshot(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = EnergyBarriersWidget()
    widget.set_sources(
        [
            EnergyBarrierSourceRow(key="path-a", reactant="A", product="B", charge_state="M", path_length="1.0", neb_status="ready"),
            EnergyBarrierSourceRow(key="path-b", reactant="B", product="C", charge_state="M", path_length="1.2", neb_status="ready"),
        ]
    )
    viewed: list[str] = []
    widget.view_source_requested.connect(viewed.append)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))

    btn = widget._sources_table.cellWidget(0, 6)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0, 1)
    btn.clicked.emit()

    assert viewed == ["path-a"]


def test_quantum_source_configure_applies_to_snapshot_rows(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app.selection_snapshot import SelectionSnapshotPushButton

    widget = QuantumCorrectionsWidget()
    widget.set_sources(
        [
            QuantumSourceRow(key="site-a", label="A", charge_state="M", relative_energy="0 eV", displacements="9", source_job="j", stage="sp", num_farms_display="", status="ready"),
            QuantumSourceRow(key="site-b", label="B", charge_state="M", relative_energy="0.1 eV", displacements="9", source_job="j", stage="sp", num_farms_display="", status="ready"),
        ]
    )
    configured: list[object] = []
    widget.configure_requested.connect(configured.append)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **kw: QMessageBox.StandardButton.Yes))

    btn = widget._sources_table.cellWidget(0, 8)
    assert isinstance(btn, SelectionSnapshotPushButton)
    btn.selection_snapshot_rows = (0, 1)
    btn.clicked.emit()

    assert len(configured) == 1
    assert set(configured[0]) == {"site-a", "site-b"}
