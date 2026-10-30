"""Tests for the Magnetic Fields muon-site selection dialog."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QMessageBox

from mufinder.app.magnetic_site_dialog import MuonSiteSelectionDialog
from mufinder.services.muon_sites_summary import MAGNETIC_FIELDS_SCOPE, set_show_every_relaxed_site
from mufinder.services.project_service import ProjectService
from tests.support.magnetic_field_sites import (
    add_uep_cluster,
    displaced_iron_supercell,
    feo_host_structure,
    relaxation_cluster_document,
    uep_cluster_document,
)


def _set_site_source(widget, source: str) -> None:
    index = widget._site_source_combo.findData(source)
    assert index >= 0
    widget._site_source_combo.setCurrentIndex(index)


def test_muon_site_selection_dialog_lists_uep_cluster_muon_sites(qapp, tmp_path) -> None:
    document = uep_cluster_document(tmp_path)

    dialog = MuonSiteSelectionDialog(document, feo_host_structure())

    assert dialog._site_table.rowCount() == 1
    selected = dialog.selected_muon_site()
    assert selected is not None
    assert selected.source == "uep_clustering"
    assert selected.label == "UEP cluster 2"
    assert selected.source_label == "UEP batch"
    assert selected.cluster_label == "Cluster 2"
    assert selected.fractional_position == (0.1, 0.2, 0.3)
    assert not dialog._distortion_checkbox.isEnabled()
    dialog.close()


def test_muon_site_selection_dialog_lists_relaxation_clusters_with_distortion_option(
    qapp, tmp_path
) -> None:
    relaxed_structure = displaced_iron_supercell()
    document = relaxation_cluster_document(tmp_path, relaxed_structure)
    set_show_every_relaxed_site(document, MAGNETIC_FIELDS_SCOPE, True)

    dialog = MuonSiteSelectionDialog(document, feo_host_structure())
    _set_site_source(dialog, "relaxation_clustering")

    assert dialog._site_table.rowCount() == 1
    selected = dialog.selected_muon_site()
    assert selected is not None
    assert selected.source == "relaxation_clustering"
    assert selected.label == "Positive muon relaxations cluster 1"
    assert selected.distortion_structure_id == relaxed_structure.id
    assert selected.include_distortion is False
    assert dialog._distortion_checkbox.isEnabled()

    dialog._distortion_checkbox.setChecked(True)
    assert dialog.selected_muon_site().include_distortion is True
    dialog.close()


def test_muon_site_selection_dialog_restores_the_distortion_choice(qapp, tmp_path) -> None:
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    set_show_every_relaxed_site(document, MAGNETIC_FIELDS_SCOPE, True)
    first = MuonSiteSelectionDialog(document, feo_host_structure())
    _set_site_source(first, "relaxation_clustering")
    first._distortion_checkbox.setChecked(True)
    chosen = first.selected_muon_site()
    first.close()

    reopened = MuonSiteSelectionDialog(document, feo_host_structure(), chosen)

    assert reopened._distortion_checkbox.isChecked()
    assert reopened.selected_muon_site().include_distortion is True
    _set_site_source(reopened, "custom")
    assert not reopened._distortion_checkbox.isEnabled()
    assert not reopened._distortion_checkbox.isChecked()
    assert reopened.selected_muon_site().include_distortion is False
    reopened.close()


def test_muon_site_selection_dialog_custom_muon_site(qapp, tmp_path) -> None:
    service = ProjectService()
    document = service.create_project(
        tmp_path / "magnetic-custom-site.mufinder", "Magnetic Custom Site"
    )
    service.add_structure(document, feo_host_structure())

    dialog = MuonSiteSelectionDialog(document, feo_host_structure())
    _set_site_source(dialog, "custom")
    dialog._custom_x.setText("0.25")
    dialog._custom_y.setText("0.5")
    dialog._custom_z.setText("0.75")

    selected = dialog.selected_muon_site()

    assert selected is not None
    assert selected.source == "custom"
    assert selected.fractional_position == (0.25, 0.5, 0.75)
    assert selected.cartesian_position == (0.75, 1.5, 2.25)
    dialog.close()


def test_muon_site_selection_dialog_refuses_a_custom_site_on_an_atom(
    qapp, tmp_path, monkeypatch
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, _title, body: warnings.append(body),
    )
    dialog = MuonSiteSelectionDialog(None, feo_host_structure())
    _set_site_source(dialog, "custom")
    dialog._custom_x.setText("0.5")
    dialog._custom_y.setText("0.5")
    dialog._custom_z.setText("0.55")

    dialog.accept()

    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.accepted_site() is None
    assert warnings == [
        "Custom site is 0.15 Å from O (atom 2); a muon site must be at least 0.5 Å from every atom."
    ]
    dialog.close()


def test_muon_site_selection_dialog_labels_uep_sites_as_unrelaxed_estimates(qapp, tmp_path) -> None:
    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    add_uep_cluster(document)
    dialog = MuonSiteSelectionDialog(document, feo_host_structure())

    _set_site_source(dialog, "uep_clustering")
    assert dialog._site_source_combo.currentText() == "UEP clustering (unrelaxed estimate)"
    assert not dialog._estimate_note.isHidden()
    assert "estimates" in dialog._estimate_note.text()

    _set_site_source(dialog, "relaxation_clustering")
    assert dialog._estimate_note.isHidden()
    dialog.close()


def test_muon_site_selection_dialog_offers_relaxation_clusters_first_once_they_exist(
    qapp, tmp_path
) -> None:
    uep_only = MuonSiteSelectionDialog(uep_cluster_document(tmp_path / "uep"), feo_host_structure())
    assert uep_only._site_source_combo.currentData() == "uep_clustering"
    uep_only.close()

    document = relaxation_cluster_document(tmp_path, displaced_iron_supercell())
    add_uep_cluster(document)
    # With nothing pinned the relaxation list is empty, so the UEP sites stay first.
    unpinned = MuonSiteSelectionDialog(document, feo_host_structure())
    assert unpinned._site_source_combo.currentData() == "uep_clustering"
    unpinned.close()

    set_show_every_relaxed_site(document, MAGNETIC_FIELDS_SCOPE, True)
    both = MuonSiteSelectionDialog(document, feo_host_structure())
    assert both._site_source_combo.currentData() == "relaxation_clustering"
    assert both.selected_muon_site().source == "relaxation_clustering"
    both.close()
