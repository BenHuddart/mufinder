"""Behavior tests for muon generation form widget."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QAbstractItemView, QDialog, QLabel, QSizePolicy, QWidget

from mufinder.app import theme as theme_module
from mufinder.app import muon_generation_widget as widget_module
from mufinder.app.muon_generation_widget import MuonGenerationWidget, _ClusterPositionsDialog
from mufinder.domain.models import Structure


def test_widget_state_roundtrip_and_table_render(qapp) -> None:
    widget = MuonGenerationWidget()
    assert widget._header_label.text() == "Initial Muon Position Generation"
    assert widget._generate_btn.text() == "Generate Sites"

    state = {
        "method": "near_element",
        "random": {
            "selection_mode": "count",
            "count": 30,
            "min_muon_muon": 0.9,
            "min_muon_atom": 1.1,
            "max_attempts": 25000,
            "seed": "5",
        },
        "near_element": {
            "element": "O",
            "muons_per_atom": 3,
            "bond_distance": 0.8,
            "min_muon_atom": 0.8,
        },
        "manual": {
            "coord_system": "cartesian",
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
        },
    }
    widget.set_form_state(state)
    collected = widget.collect_state()

    assert collected["method"] == "near_element"
    assert collected["random"]["selection_mode"] == "count"
    assert collected["near_element"]["bond_distance"] == 0.8
    assert collected["near_element"]["muons_per_atom"] == 3
    assert collected["manual"]["coord_system"] == "cartesian"

    emitted: list[tuple[str, dict]] = []
    widget.generate_requested.connect(lambda method, payload: emitted.append((method, payload)))
    widget._emit_generate()

    assert emitted[0][0] == "near_element"
    assert emitted[0][1]["element"] == "O"

    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Site A",
                "method": "random",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
            }
        ]
    )
    assert widget._site_table.rowCount() == 1
    assert widget._site_table.item(0, 0).text() == "Site A"
    assert widget._site_table.item(0, 2).text() == "generated"
    assert [widget._site_table.horizontalHeaderItem(column).text() for column in range(6)] == [
        "Site",
        "Method",
        "State",
        "fX",
        "fY",
        "fZ",
    ]


def test_widget_delete_selected_candidates_emits_site_ids(qapp) -> None:
    widget = MuonGenerationWidget()
    assert widget._create_jobs_btn.text() == "Create Jobs from Selected"
    assert not widget._create_jobs_btn.isEnabled()
    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Site A",
                "method": "random",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
            },
            {
                "id": "site-b",
                "label": "Site B",
                "method": "manual",
                "status": "generated",
                "fractional": [0.4, 0.5, 0.6],
            },
        ]
    )

    emitted: list[list[str]] = []
    widget.delete_candidates_requested.connect(lambda site_ids: emitted.append(list(site_ids)))
    widget._site_table.selectRow(1)
    qapp.processEvents()
    widget._emit_delete_selected()

    assert widget._delete_btn.isEnabled()
    assert emitted == [["site-b"]]


def test_widget_backspace_shortcut_deletes_selected_candidate_on_macos(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Site A",
                "method": "random",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
            }
        ]
    )

    emitted: list[list[str]] = []
    widget.delete_candidates_requested.connect(lambda site_ids: emitted.append(list(site_ids)))
    widget._site_table.selectRow(0)

    widget._backspace_shortcut.activated.emit()

    assert emitted == [["site-a"]]


def test_widget_create_jobs_from_selected_emits_site_ids(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Site A",
                "method": "random",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
            }
        ]
    )

    emitted: list[list[str]] = []
    widget.create_jobs_requested.connect(lambda site_ids: emitted.append(list(site_ids)))
    widget._site_table.selectRow(0)
    qapp.processEvents()
    widget._emit_create_jobs_selected()

    assert widget._create_jobs_btn.isEnabled()
    assert emitted == [["site-a"]]


def test_widget_cluster_positions_dialog_emits_anchor_and_site_ids(qapp, monkeypatch) -> None:
    widget = MuonGenerationWidget()
    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Site A",
                "method": "random",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
            }
        ]
    )
    widget._site_table.selectRow(0)
    qapp.processEvents()

    captured_rows: list[dict] = []

    class FakeDialog:
        def __init__(self, rows, parent=None) -> None:
            captured_rows.extend(rows)

        def exec(self):
            return QDialog.DialogCode.Accepted

        def anchor_fractional(self):
            return [0.9, 0.8, 0.7]

    monkeypatch.setattr(widget_module, "_ClusterPositionsDialog", FakeDialog)
    emitted: list[dict] = []
    widget.cluster_positions_requested.connect(lambda payload: emitted.append(dict(payload)))
    widget._open_cluster_positions_dialog()

    assert widget._cluster_btn.isEnabled()
    assert captured_rows[0]["id"] == "site-a"
    assert emitted == [{"site_ids": ["site-a"], "anchor_fractional": [0.9, 0.8, 0.7]}]


def test_cluster_positions_dialog_can_use_selected_muon_as_anchor(qapp) -> None:
    dialog = _ClusterPositionsDialog(
        [
            {
                "label": "Site A",
                "fractional": [0.1, 0.2, 0.3],
            }
        ]
    )

    dialog._anchor_source.setCurrentIndex(1)

    assert dialog.anchor_fractional() == [0.1, 0.2, 0.3]
    assert dialog.windowTitle() == "Cluster Positions"
    assert dialog.objectName() == "clusterPositionsDialog"
    assert dialog.findChild(QWidget, "clusterPositionsCard") is not None
    subtitle = dialog.findChildren(QLabel)[1]
    assert subtitle.text() == (
        "Specify an anchor in fractional coordinates. Selected muon sites will be replaced by their "
        "symmetry-equivalent site that is closest to the anchor."
    )


def test_widget_can_restore_generated_site_selection_after_refresh(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Site A",
                "method": "uep_cluster_member",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
            },
            {
                "id": "site-b",
                "label": "Site B",
                "method": "near_element",
                "status": "staged",
                "fractional": [0.4, 0.5, 0.6],
            },
        ]
    )

    widget.select_site_ids(["site-a", "site-b"])
    qapp.processEvents()

    assert widget._selected_site_ids() == ["site-a", "site-b"]
    assert widget._cluster_btn.isEnabled()
    assert widget._site_table.item(0, 1).text() == "UEP mem."
    assert widget._site_table.item(1, 1).text() == "attach"
    assert widget._site_table.columnWidth(3) == 58


def test_widget_generation_progress_and_random_distance_update(qapp) -> None:
    widget = MuonGenerationWidget()

    widget.set_generation_busy(True, "Generating random sites...")
    assert widget._generation_progress.minimum() == 0
    assert widget._generation_progress.maximum() == 0
    assert widget._generation_progress.format() == "Generating random sites..."
    assert not widget._generate_btn.isEnabled()

    widget.set_random_muon_muon_distance(1.234)
    assert widget._random_min_muon_muon.value() == 1.234

    widget.set_generation_busy(False, "Ready")
    assert widget._generation_progress.maximum() == 1
    assert widget._generation_progress.format() == "Ready"
    assert widget._generate_btn.isEnabled()

    widget.set_generation_busy(True, "Preparing random generation...")
    widget.set_generation_progress(2, 5)
    assert widget._generation_progress.minimum() == 0
    assert widget._generation_progress.maximum() == 5
    assert widget._generation_progress.value() == 2
    assert widget._generation_progress.format() == "Generated %v/%m sites"


def test_widget_random_choice_rows_attach_radios_to_fields_and_show_active_mode(qapp) -> None:
    widget = MuonGenerationWidget()

    assert widget._random_by_separation.parent() is widget._random_separation_row
    assert widget._random_by_separation.text() == "Muon-muon separation (Å)"
    assert widget._random_min_muon_muon.parent() is widget._random_separation_row
    assert widget._random_by_count.parent() is widget._random_count_row
    assert widget._random_count.parent() is widget._random_count_row
    assert widget._random_separation_row.property("active") == "true"
    assert widget._random_count_row.property("active") == "false"

    widget._random_by_count.setChecked(True)
    qapp.processEvents()

    assert not widget._random_by_separation.isChecked()
    assert widget._random_by_count.isChecked()
    assert widget._random_separation_row.property("active") == "false"
    assert widget._random_count_row.property("active") == "true"
    assert not widget._random_min_muon_muon.isEnabled()
    assert widget._random_count.isEnabled()


def test_widget_random_choice_rows_keep_stable_width_when_mode_changes(qapp) -> None:
    widget = MuonGenerationWidget()

    initial_width = max(
        widget._random_separation_row.sizeHint().width(),
        widget._random_count_row.sizeHint().width(),
    )
    widget._random_by_count.setChecked(True)
    qapp.processEvents()
    count_width = max(
        widget._random_separation_row.sizeHint().width(),
        widget._random_count_row.sizeHint().width(),
    )
    widget._random_by_separation.setChecked(True)
    qapp.processEvents()
    separation_width = max(
        widget._random_separation_row.sizeHint().width(),
        widget._random_count_row.sizeHint().width(),
    )

    assert count_width == initial_width
    assert separation_width == initial_width
    assert widget._random_min_muon_muon.minimumWidth() == widget._random_count.minimumWidth()


def test_widget_generation_card_parameter_pages_match_card_surface(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)
    widget = MuonGenerationWidget()
    palette = theme_module.current_palette()

    assert widget._form_box.objectName() == "muonGenerationSetupCard"
    assert widget._param_stack.objectName() == "muonGenerationParamStack"
    stylesheet = qapp.styleSheet()

    assert palette.bg_surface in stylesheet
    assert "QGroupBox#muonGenerationSetupCard" in stylesheet
    assert "QStackedWidget#muonGenerationParamStack" in stylesheet
    assert "QWidget#muonGenerationRandomPage" in stylesheet
    assert "QWidget#muonGenerationNearPage" in stylesheet
    assert "QWidget#muonGenerationManualPage" in stylesheet
    assert "QWidget#muonGenerationUepPage" in stylesheet


def test_widget_parameter_stack_keeps_consistent_compact_height(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.show()
    widget._set_current_method("random")
    qapp.processEvents()
    random_actual_height = widget._param_stack.height()

    widget._set_current_method("uep_clusters")
    qapp.processEvents()
    uep_actual_height = widget._param_stack.height()

    assert random_actual_height == uep_actual_height
    assert widget._param_stack.minimumHeight() == widget._param_stack.maximumHeight()
    assert widget._param_stack.minimumHeight() == random_actual_height
    assert widget._uep_cluster_table.height() == 128
    assert widget._uep_cluster_table.sizePolicy().verticalPolicy() != QSizePolicy.Policy.Expanding


def test_widget_count_mode_updates_estimated_spacing_from_host(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_host_structure(
        Structure(
            label="Host",
            lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
    )

    widget._random_by_count.setChecked(True)
    widget._random_count.setValue(2)
    first_spacing = widget._random_min_muon_muon.value()
    widget._random_count.setValue(8)
    second_spacing = widget._random_min_muon_muon.value()

    assert first_spacing > 0.0
    assert second_spacing > 0.0
    assert second_spacing < first_spacing
    assert not widget._random_min_muon_muon.isEnabled()


def test_widget_count_mode_restore_preserves_saved_spacing_until_user_changes(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_host_structure(
        Structure(
            label="Host",
            lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
    )

    widget.set_form_state(
        {
            "method": "random",
            "random": {
                "selection_mode": "count",
                "count": 2,
                "min_muon_muon": 1.234,
                "min_muon_atom": 0.8,
                "max_attempts": 5000,
            },
        }
    )

    assert widget._random_by_count.isChecked()
    assert not widget._random_min_muon_muon.isEnabled()
    assert abs(widget._random_min_muon_muon.value() - 1.234) < 1e-9

    widget._random_count.setValue(8)
    qapp.processEvents()

    assert widget._random_min_muon_muon.value() > 0.0
    assert abs(widget._random_min_muon_muon.value() - 1.234) > 1e-6


def test_widget_collect_method_parameters_for_all_modes(qapp) -> None:
    widget = MuonGenerationWidget()

    widget._set_current_method("random")
    random_params = widget._collect_method_parameters("random")
    assert random_params["selection_mode"] == "separation"
    assert "count" not in random_params
    assert "max_attempts" in random_params
    assert random_params["include_symmetry_equivalents"] is True
    assert "seed" not in random_params

    widget._random_by_count.setChecked(True)
    random_count_params = widget._collect_method_parameters("random")
    assert random_count_params["selection_mode"] == "count"
    assert random_count_params["count"] == widget._random_count.value()
    assert not widget._random_min_muon_muon.isEnabled()
    assert widget._random_count.isEnabled()

    assert (
        widget._method_combo.itemText(widget._method_combo.findData("near_element"))
        == "Attach muon to element"
    )

    widget._set_current_method("near_element")
    widget.set_host_structure(
        Structure(
            label="Host",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si", "O"],
            fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        )
    )
    widget._near_element.setCurrentText("Si")
    near_params = widget._collect_method_parameters("near_element")
    assert near_params["element"] == "Si"
    assert near_params["strategy"] == "attached"
    assert "min_muon_muon" not in near_params
    assert "max_candidates_per_atom" not in near_params
    assert near_params["muons_per_atom"] == widget._near_muons_per_atom.value()
    assert near_params["bond_distance"] > 0.0

    assert [
        widget._near_element.itemText(index) for index in range(widget._near_element.count())
    ] == ["O", "Si"]

    widget._set_current_method("manual")
    widget._manual_coord_system.setCurrentText("fractional")
    widget._manual_x.setValue(0.2)
    manual_params = widget._collect_method_parameters("manual")
    assert manual_params["coord_system"] == "fractional"
    assert manual_params["x"] == 0.2

    widget._set_current_method("uep_clusters")
    assert widget._collect_method_parameters("uep_clusters") == {"selections": []}


def test_widget_manual_coordinate_labels_follow_coordinate_system(qapp) -> None:
    widget = MuonGenerationWidget()

    assert [
        widget._manual_x_label.text(),
        widget._manual_y_label.text(),
        widget._manual_z_label.text(),
    ] == [
        "fX",
        "fY",
        "fZ",
    ]

    widget._manual_coord_system.setCurrentText("cartesian")

    assert [
        widget._manual_x_label.text(),
        widget._manual_y_label.text(),
        widget._manual_z_label.text(),
    ] == [
        "X (Å)",
        "Y (Å)",
        "Z (Å)",
    ]

    widget._manual_coord_system.setCurrentText("fractional")

    assert [
        widget._manual_x_label.text(),
        widget._manual_y_label.text(),
        widget._manual_z_label.text(),
    ] == [
        "fX",
        "fY",
        "fZ",
    ]


def test_widget_manual_fractional_position_previews_in_viewer(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_host_structure(
        Structure(
            label="Host",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
    )

    widget._set_current_method("manual")
    widget._manual_coord_system.setCurrentText("fractional")
    widget._manual_x.setValue(0.2)
    widget._manual_y.setValue(0.3)
    widget._manual_z.setValue(0.4)
    qapp.processEvents()

    assert widget._viewer._point_overlay_specs


def test_widget_selected_candidate_overlays_do_not_depend_on_generation_mode(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_host_structure(
        Structure(
            label="Host",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
    )
    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Site A",
                "method": "random",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
            }
        ]
    )
    widget._site_table.selectRow(0)
    qapp.processEvents()

    overlay_counts: list[int] = []
    for method in ("random", "near_element", "manual", "uep_clusters"):
        widget._set_current_method(method)
        qapp.processEvents()
        overlay_counts.append(len(widget._viewer._point_overlay_specs[0].positions))

    assert overlay_counts[0] == 1
    assert overlay_counts[1] == 1
    assert overlay_counts[2] >= 1
    assert overlay_counts[3] == 1


def test_widget_clicking_selected_candidate_row_drops_selection_and_overlay(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_host_structure(
        Structure(
            label="Host",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
    )
    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Site A",
                "method": "random",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
            }
        ]
    )
    widget._site_table.selectRow(0)
    qapp.processEvents()
    assert widget._viewer._point_overlay_specs
    assert widget._delete_btn.isEnabled()

    point = widget._site_table.visualItemRect(widget._site_table.item(0, 0)).center()
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(point),
        QPointF(point),
        QPointF(point),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )

    handled = widget.eventFilter(widget._site_table.viewport(), event)
    qapp.processEvents()

    assert handled is True
    assert not widget._site_table.selectedItems()
    assert not widget._viewer._point_overlay_specs
    assert not widget._delete_btn.isEnabled()


def test_widget_uep_cluster_rows_emit_selected_payload_and_preview(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_host_structure(
        Structure(
            label="Host",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
    )
    row = {
        "source_key": "job:cluster:1:cluster:0",
        "source_job_label": "uep_row_1",
        "cluster_index": 1,
        "cluster_size": 2,
        "selection_kind": "cluster",
        "classification": "minimum",
        "classical_energy_ev": -0.7,
        "zero_point_energy_ev": 0.2,
        "total_energy_ev": -0.5,
        "fractional_position": [0.1, 0.2, 0.3],
        "cartesian_position": [0.5, 1.0, 1.5],
        "generation_method": "uep_cluster",
        "properties": {"cluster_index": 1},
    }

    widget.set_uep_cluster_sources([row])
    widget._set_current_method("uep_clusters")
    widget._uep_cluster_table.selectRow(0)
    qapp.processEvents()

    emitted: list[tuple[str, dict]] = []
    widget.generate_requested.connect(lambda method, payload: emitted.append((method, payload)))
    widget._emit_generate()

    assert widget._uep_cluster_table.rowCount() == 1
    assert widget._uep_cluster_table.columnCount() == 8
    assert [
        widget._uep_cluster_table.horizontalHeaderItem(column).text()
        for column in range(widget._uep_cluster_table.columnCount())
    ] == ["Source", "Cluster", "Size", "Class", "E (eV)", "fX", "fY", "fZ"]
    assert widget._uep_cluster_table.item(0, 4).text() == "-0.7"
    assert widget._uep_cluster_table.item(0, 7).text() == "0.3000"
    assert emitted[0][0] == "uep_clusters"
    assert emitted[0][1]["selections"][0]["source_key"] == "job:cluster:1:cluster:0"
    assert widget._viewer._point_overlay_specs


def test_widget_uep_cluster_rows_emit_all_selected_payloads(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_host_structure(
        Structure(
            label="Host",
            lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
    )
    rows = [
        {
            "source_key": f"job:cluster:{index}:cluster:{index - 1}",
            "source_job_label": "uep_row_1",
            "cluster_index": index,
            "cluster_size": index + 1,
            "selection_kind": "cluster",
            "classification": "minimum",
            "classical_energy_ev": -0.7 - index,
            "fractional_position": [0.1 * index, 0.2, 0.3],
            "cartesian_position": [0.5 * index, 1.0, 1.5],
            "generation_method": "uep_cluster",
            "properties": {"cluster_index": index},
        }
        for index in (1, 2)
    ]

    widget.set_uep_cluster_sources(rows)
    widget._set_current_method("uep_clusters")
    assert (
        widget._uep_cluster_table.selectionMode()
        == QAbstractItemView.SelectionMode.ExtendedSelection
    )
    for row in (0, 1):
        point = widget._uep_cluster_table.visualItemRect(
            widget._uep_cluster_table.item(row, 0)
        ).center()
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(point),
            QPointF(point),
            QPointF(point),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        assert widget.eventFilter(widget._uep_cluster_table.viewport(), event) is True
    qapp.processEvents()

    emitted: list[tuple[str, dict]] = []
    widget.generate_requested.connect(lambda method, payload: emitted.append((method, payload)))
    widget._emit_generate()

    assert emitted[0][0] == "uep_clusters"
    assert [row["source_key"] for row in emitted[0][1]["selections"]] == [
        "job:cluster:1:cluster:0",
        "job:cluster:2:cluster:1",
    ]
    assert len(widget._viewer._point_overlay_specs[0].positions) == 2


def _host_a() -> Structure:
    return Structure(
        label="Host A",
        lattice_vectors=[[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
        species=["Si", "O"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _host_b() -> Structure:
    return Structure(
        label="Host B",
        lattice_vectors=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]],
        species=["Fe"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def test_set_host_structure_skips_viewer_rebuild_when_unchanged(qapp, monkeypatch) -> None:
    widget = MuonGenerationWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = _host_a()
    widget.set_host_structure(host)
    assert len(calls) == 1

    # Same structure object on the next flush must not re-render the viewer.
    widget.set_host_structure(host)
    assert len(calls) == 1

    # A different structure id must re-render.
    widget.set_host_structure(_host_b())
    assert len(calls) == 2


def test_set_host_structure_none_transitions_render(qapp, monkeypatch) -> None:
    widget = MuonGenerationWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = _host_a()
    widget.set_host_structure(host)  # None -> structure
    widget.set_host_structure(None)  # structure -> None
    widget.set_host_structure(None)  # None -> None (skipped)
    widget.set_host_structure(host)  # None -> structure

    assert calls == [host, None, host]


def test_set_host_structure_rerenders_after_in_place_geometry_swap(qapp, monkeypatch) -> None:
    # Supercell asset structures keep their id but reassign the geometry lists
    # (see ProjectService). Reassigning species/positions must invalidate the guard.
    widget = MuonGenerationWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = _host_a()
    widget.set_host_structure(host)
    assert len(calls) == 1

    host.species = ["Si", "O", "H"]
    host.fractional_positions = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]
    widget.set_host_structure(host)
    assert len(calls) == 2


def test_set_host_structure_rerenders_after_in_place_element_mutation(qapp, monkeypatch) -> None:
    # Regression for the id()-recycling hazard: the fingerprint is content-based,
    # so mutating geometry *elements* while keeping the very same list objects
    # (identical identities) must still invalidate the guard and re-render.
    widget = MuonGenerationWidget()
    calls: list[object] = []
    monkeypatch.setattr(widget._viewer, "set_structure", lambda structure: calls.append(structure))

    host = _host_a()
    widget.set_host_structure(host)
    assert len(calls) == 1

    species_id = id(host.species)
    positions_id = id(host.fractional_positions)
    host.species[0] = "Fe"
    host.fractional_positions[1][0] = 0.9
    assert id(host.species) == species_id
    assert id(host.fractional_positions) == positions_id

    widget.set_host_structure(host)
    assert len(calls) == 2


def test_set_sites_skips_rebuild_when_rows_unchanged(qapp) -> None:
    widget = MuonGenerationWidget()
    rows = [
        {
            "id": "site-a",
            "label": "Site A",
            "method": "random",
            "status": "generated",
            "fractional": [0.1, 0.2, 0.3],
            "source_structure_id": "",
        }
    ]
    widget.set_sites(rows)
    assert widget._site_table.rowCount() == 1

    inserts: list[int] = []
    original_insert = widget._site_table.insertRow
    widget._site_table.insertRow = lambda row: (inserts.append(row), original_insert(row))[1]

    # Identical rows (fresh list, equal-by-value) must not rebuild the table.
    widget.set_sites([dict(rows[0])])
    assert inserts == []
    assert widget._site_table.rowCount() == 1

    # A changed row must rebuild.
    changed = [dict(rows[0])]
    changed[0]["status"] = "staged"
    widget.set_sites(changed)
    assert inserts != []


def test_set_sites_preserves_selection_across_unchanged_reflush(qapp) -> None:
    widget = MuonGenerationWidget()
    rows = [
        {
            "id": "site-a",
            "label": "A",
            "method": "random",
            "status": "generated",
            "fractional": [0.1, 0.2, 0.3],
        },
        {
            "id": "site-b",
            "label": "B",
            "method": "random",
            "status": "generated",
            "fractional": [0.4, 0.5, 0.6],
        },
    ]
    widget.set_sites(rows)
    widget.select_site_ids(["site-b"])
    assert widget._selected_site_ids() == ["site-b"]

    widget.set_sites([dict(row) for row in rows])
    # Unchanged reflush preserves the selection (rebuild would have cleared it).
    assert widget._selected_site_ids() == ["site-b"]


def test_set_uep_cluster_sources_skips_rebuild_when_unchanged(qapp) -> None:
    widget = MuonGenerationWidget()
    rows = [
        {
            "source_job_label": "job-1",
            "cluster_index": 0,
            "cluster_size": 3,
            "classification": "core",
            "classical_energy_ev": -1.0,
            "fractional_position": [0.1, 0.2, 0.3],
        }
    ]
    widget.set_uep_cluster_sources(rows)
    assert widget._uep_cluster_table.rowCount() == 1

    inserts: list[int] = []
    original_insert = widget._uep_cluster_table.insertRow
    widget._uep_cluster_table.insertRow = lambda row: (inserts.append(row), original_insert(row))[1]

    widget.set_uep_cluster_sources([dict(rows[0])])
    assert inserts == []

    changed = [dict(rows[0])]
    changed[0]["cluster_size"] = 5
    widget.set_uep_cluster_sources(changed)
    assert inserts != []


def test_theme_change_reapplies_stale_row_styling(qapp) -> None:
    widget = MuonGenerationWidget()
    widget.set_active_host(_host_a())
    widget.set_sites(
        [
            {
                "id": "site-a",
                "label": "Stale site",
                "method": "random",
                "status": "generated",
                "fractional": [0.1, 0.2, 0.3],
                "source_structure_id": "different-host-id",
            }
        ]
    )
    # The row is stale (source != active host), so label carries the marker.
    assert widget._site_table.item(0, 0).text().endswith(" (stale)")

    # A palette change must not crash and must keep stale styling applied even
    # though set_sites is now skipped for unchanged rows.
    widget.changeEvent(QEvent(QEvent.Type.PaletteChange))
    assert widget._site_table.item(0, 0).text().endswith(" (stale)")
