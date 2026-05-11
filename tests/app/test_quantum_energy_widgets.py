from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

from mufinder.app.energy_barriers_widget import (
    EnergyBarrierPreviewImageRow,
    EnergyBarrierResultRow,
    EnergyBarrierSourceRow,
    EnergyBarriersWidget,
)
from mufinder.app.quantum_corrections_widget import QuantumCorrectionsWidget, QuantumSourceRow
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
                source_job="relax_a",
                status="ready",
            )
        ]
    )

    configured: list[str] = []
    widget.configure_requested.connect(configured.append)

    button = widget._sources_table.cellWidget(0, 5)
    assert isinstance(button, QPushButton)
    assert button.text() == "Configure..."

    button.click()

    assert configured == ["site-a"]


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
                source_jobs="relax_a -> relax_b",
                status="ready",
            )
        ]
    )

    configured: list[str] = []
    viewed: list[str] = []
    widget.configure_source_requested.connect(configured.append)
    widget.view_source_requested.connect(viewed.append)

    configure_button = widget._sources_table.cellWidget(0, 6)
    view_button = widget._sources_table.cellWidget(0, 7)
    assert isinstance(configure_button, QPushButton)
    assert isinstance(view_button, QPushButton)
    assert configure_button.text() == "Configure..."
    assert view_button.text() == "View..."

    configure_button.click()
    view_button.click()

    assert configured == ["path-a"]
    assert viewed == ["path-a"]


def test_energy_barriers_widget_marks_configured_result_row(qapp) -> None:
    widget = EnergyBarriersWidget()
    widget.set_results(
        [
            EnergyBarrierResultRow(
                key="ts-a",
                label="TS A",
                barrier_height="0.1 eV",
                images="5",
                transition_state_file="barrier.ts",
                neb_status="ready",
                configured=True,
            )
        ]
    )

    configured: list[str] = []
    widget.configure_result_requested.connect(configured.append)

    configure_button = widget._results_table.cellWidget(0, 5)
    assert isinstance(configure_button, QPushButton)
    assert configure_button.text() == "Configured..."

    configure_button.click()

    assert configured == ["ts-a"]


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