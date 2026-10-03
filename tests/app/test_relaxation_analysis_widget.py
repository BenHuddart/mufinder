"""Behavior tests for structural-relaxation analysis UI."""

from __future__ import annotations

import warnings
from uuid import uuid4

from ase.data import atomic_numbers, covalent_radii
from ase.data.colors import jmol_colors
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, PathPatch
import numpy as np
from PySide6.QtWidgets import QCheckBox, QFrame, QSplitter, QTabWidget, QTreeWidget

from mufinder.app.relaxation_analysis_widget import (
    GRAPH_NODE_GID,
    _GRAPH_COMPONENT_GAP,
    _graph_component_layout,
    _graph_edge_strip,
    pack_graph_components,
    CompletedRelaxationAnalysisSource,
    DisplacementAnalysisDialog,
    RelaxationAnalysisJobDialog,
    RelaxationAnalysisWidget,
    RelaxationClusteringDialog,
    default_workspace_state,
)
from mufinder.science.relaxation_analysis import RelaxedMuonSite
from mufinder.domain.models import Structure


def _host_structure() -> Structure:
    return Structure(
        id=uuid4(),
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _relaxed_site(host: Structure, fractional: tuple[float, float, float], energy: float) -> RelaxedMuonSite:
    return RelaxedMuonSite(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=fractional,
        cartesian_position_ang=(
            fractional[0] * host.lattice_vectors[0][0],
            fractional[1] * host.lattice_vectors[1][1],
            fractional[2] * host.lattice_vectors[2][2],
        ),
        total_energy_ev=energy,
    )


def test_relaxation_analysis_workspace_headers_show_wyckoff_assignments(qapp) -> None:
    host = Structure(
        id=uuid4(),
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    job_id = uuid4()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[job_id],
    )
    workspace["clustering"] = {
        "method": "connected",
        "connected_tolerance_ang": 0.25,
        "hierarchical_t": 0.3,
        "kmeans_k": 1,
        "wyckoff_match_tolerance_ang": 0.25,
        "tolerance_scan": [],
        "result": {
            "cluster_labels": [1],
            "clusters": [
                {
                    "cluster_index": 1,
                    "size": 1,
                    "representative_index": 0,
                    "representative_job_id": str(job_id),
                    "representative_fractional_position": [0.5, 0.0, 0.0],
                    "representative_cartesian_position_ang": [2.5, 0.0, 0.0],
                    "minimum_relative_energy_ev": 0.0,
                    "average_relative_energy_ev": 0.0,
                    "standard_deviation_ev": 0.0,
                    "member_indexes": [0],
                }
            ],
            "wyckoff_assignments": [
                {
                    "cluster_index": 1,
                    "match": {
                        "matched": True,
                        "label": "6b",
                        "fractional_position": [0.5, 0.0, 0.0],
                        "cartesian_position_ang": [2.5, 0.0, 0.0],
                        "multiplicity": 6,
                        "letter": "b",
                        "space_group_number": 229,
                        "space_group_symbol": "Im-3m",
                        "site_symmetry_symbol": "4/mm.m",
                        "max_distance_ang": 0.0,
                        "mean_distance_ang": 0.0,
                        "occupied": False,
                        "special": True,
                        "warnings": [],
                    },
                }
            ],
        },
    }
    source = CompletedRelaxationAnalysisSource(
        job_id=job_id,
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=(0.5, 0.0, 0.0),
        total_energy_ev=-10.0,
    )
    widget = RelaxationAnalysisWidget()

    widget.set_context(
        workspaces=[workspace],
        sources=[source],
        structures={host.id: host},
        host_structure=host,
        comparison_options=[],
    )

    table = widget.findChild(QTreeWidget)
    assert table is not None
    assert table.property("tabbedTable") == "true"
    assert table.topLevelItem(0).text(0) == "Cluster 1: (0.5000, 0.0000, 0.0000) 6b"
    assert table.topLevelItem(0).isFirstColumnSpanned()
    assert table.topLevelItem(0).childCount() == 1

    tabs = widget.findChild(QTabWidget, "analysisWorkspaceTabs")
    assert tabs is not None
    splitter = widget.findChild(QSplitter, "analysisWorkspaceContentSplitter")
    assert splitter is not None
    widget.resize(1890, 1120)
    widget.show()
    for _ in range(4):
        qapp.processEvents()
    sizes = splitter.sizes()
    assert sum(sizes) > 0
    table_share = sizes[0] / sum(sizes)
    assert 0.45 <= table_share <= 0.55
    widget.shutdown()


def test_relaxation_analysis_add_dialog_shows_relative_energies(qapp) -> None:
    included_id = uuid4()
    lower_candidate_id = uuid4()
    sources = [
        CompletedRelaxationAnalysisSource(
            job_id=included_id,
            job_label="Included",
            charge_state="positive_muon",
            structure_id=uuid4(),
            site_id=None,
            folded_fractional_position=(0.1, 0.2, 0.3),
            total_energy_ev=-10.0,
        ),
        CompletedRelaxationAnalysisSource(
            job_id=lower_candidate_id,
            job_label="Lower candidate",
            charge_state="positive_muon",
            structure_id=uuid4(),
            site_id=None,
            folded_fractional_position=(0.4, 0.5, 0.6),
            total_energy_ev=-10.25,
        ),
    ]

    dialog = RelaxationAnalysisJobDialog(
        title="Add Structure to Analysis",
        sources=sources,
        excluded_job_ids={included_id},
        reference_job_ids={included_id},
    )

    assert dialog._table.horizontalHeaderItem(dialog._ENERGY_COLUMN).text() == "dE vs workspace min (eV)"
    assert dialog._table.item(0, dialog._ENERGY_COLUMN).text() == "0.00000"
    assert dialog._table.item(1, dialog._ENERGY_COLUMN).text() == "-0.25000"


def test_relaxation_clustering_dialog_close_shuts_down_viewer(qapp, monkeypatch) -> None:
    host = _host_structure()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[],
    )
    dialog = RelaxationClusteringDialog(
        workspace=workspace,
        host_structure=host,
        sites=[_relaxed_site(host, (0.49, 0.01, 0.0), -10.0)],
    )
    calls: list[str] = []

    def _shutdown() -> None:
        calls.append("shutdown")

    monkeypatch.setattr(dialog._viewer, "shutdown", _shutdown)

    dialog.close()
    qapp.processEvents()

    assert calls == ["shutdown"]


def test_relaxation_connected_graph_uses_paper_style_nodes_and_edges(qapp) -> None:
    host = _host_structure()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[],
    )
    dialog = RelaxationClusteringDialog(
        workspace=workspace,
        host_structure=host,
        sites=[
            _relaxed_site(host, (0.100, 0.100, 0.100), -10.0),
            _relaxed_site(host, (0.112, 0.100, 0.100), -9.9),
            _relaxed_site(host, (0.124, 0.100, 0.100), -9.8),
            _relaxed_site(host, (0.600, 0.600, 0.600), -9.7),
        ],
    )

    axis = dialog._graph_figure.axes[0]
    nodes = [patch for patch in axis.patches if patch.get_gid() == GRAPH_NODE_GID]
    fills = [patch for patch in axis.patches if isinstance(patch, Circle) and patch.get_gid() != GRAPH_NODE_GID]
    node_labels = [patch for patch in axis.patches if isinstance(patch, PathPatch)]

    assert axis.get_title() == ""
    assert dialog._graph_figure.get_facecolor()[3] == 0.0
    assert axis.get_facecolor()[3] == 0.0
    assert len(nodes) == len(fills) == len(node_labels) == 4
    # Nodes share one radius, sized in data units so they scale with the canvas,
    # and a lighter fill inside leaves a thick coloured border ring.
    assert len({round(node.get_radius(), 9) for node in nodes}) == 1
    assert all(isinstance(node, Circle) for node in nodes)
    assert all(0.7 < fill.get_radius() / nodes[0].get_radius() < 0.9 for fill in fills)
    # Edges are filled strips in data units (so they thin with the nodes), one per
    # connected pair, and opaque for a sparse component like this one.
    result, _layouts = dialog._graph_layouts
    edge_count = int(np.triu(np.asarray(result.adjacency_matrix), k=1).sum())
    edge_collections = [collection for collection in axis.collections if isinstance(collection, PolyCollection)]
    assert edge_count > 0
    assert sum(len(collection.get_paths()) for collection in edge_collections) == edge_count
    assert all(collection.get_alpha() in (None, 1.0) for collection in edge_collections)

    dialog.close()


def test_relaxation_tolerance_scan_plot_uses_convergence_style(qapp) -> None:
    host = _host_structure()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[],
    )
    dialog = RelaxationClusteringDialog(
        workspace=workspace,
        host_structure=host,
        sites=[
            _relaxed_site(host, (0.100, 0.100, 0.100), -10.0),
            _relaxed_site(host, (0.112, 0.100, 0.100), -9.9),
        ],
    )

    axis = dialog._scan_figure.axes[0]

    assert dialog._scan_figure.get_facecolor() != axis.get_facecolor()
    assert all(spine.get_visible() for spine in axis.spines.values())
    assert any(line.get_visible() for line in axis.get_xgridlines())
    assert axis.lines[0].get_color() == "#b91c1c"

    dialog.close()


def test_relaxation_tolerance_scan_does_not_shrink_summary_headers(qapp) -> None:
    host = _host_structure()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[],
    )
    dialog = RelaxationClusteringDialog(
        workspace=workspace,
        host_structure=host,
        sites=[
            _relaxed_site(host, (0.100, 0.100, 0.100), -10.0),
            _relaxed_site(host, (0.112, 0.100, 0.100), -9.9),
            _relaxed_site(host, (0.500, 0.500, 0.500), -9.8),
        ],
    )
    header = dialog._summary_table.horizontalHeader()
    before = [header.sectionSize(column) for column in range(dialog._summary_table.columnCount())]

    dialog._scan_tolerances()
    qapp.processEvents()

    after = [header.sectionSize(column) for column in range(dialog._summary_table.columnCount())]
    assert after[0] >= before[0]
    assert after[2] >= before[2]
    assert after[3] >= before[3]
    assert after[4] >= before[4]
    assert after[5] >= before[5]

    dialog.close()


def test_displacement_plot_uses_transparent_surround_species_colours_and_boxed_legend(qapp) -> None:
    comparison_id = uuid4()
    relaxed_id = uuid4()
    comparison = Structure(
        id=comparison_id,
        label="Host",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["Fe", "O"],
        fractional_positions=[[0.2, 0.5, 0.5], [0.4, 0.5, 0.5]],
    )
    relaxed = Structure(
        id=relaxed_id,
        label="Relaxed",
        lattice_vectors=comparison.lattice_vectors,
        species=["Fe", "O", "H:mu"],
        fractional_positions=[[0.22, 0.5, 0.5], [0.41, 0.5, 0.5], [0.0, 0.5, 0.5]],
    )
    site = RelaxedMuonSite(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=relaxed_id,
        site_id=None,
        folded_fractional_position=(0.0, 0.5, 0.5),
        cartesian_position_ang=(0.0, 5.0, 5.0),
        total_energy_ev=-10.0,
    )

    dialog = DisplacementAnalysisDialog(
        workspace_name="Relaxation analysis",
        sites=[site],
        structures_by_id={comparison_id: comparison, relaxed_id: relaxed},
        comparison_options=[(comparison_id, "Host", False)],
    )

    axis = dialog._figure.axes[0]
    legend = axis.get_legend()
    fe_collection = next(
        collection
        for collection in axis.collections
        if collection.get_label() == "Fe"
    )

    assert dialog._figure.get_facecolor()[3] == 0.0
    assert axis.get_xlabel() == "Distance from muon (Å)"
    assert axis.get_ylabel() == "Radial displacement (Å)"
    assert legend is not None
    assert legend.get_frame().get_visible()
    assert legend.get_bbox_to_anchor().x0 > axis.bbox.x1
    assert mcolors.to_hex(fe_collection.get_facecolors()[0]) == mcolors.to_hex(jmol_colors[atomic_numbers["Fe"]])

    dialog.close()


def test_displacement_view_does_not_overlay_second_muon_when_relaxed_structure_contains_one(qapp) -> None:
    comparison_id = uuid4()
    relaxed_id = uuid4()
    comparison = Structure(
        id=comparison_id,
        label="Host",
        lattice_vectors=[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        species=["Cu", "Cu"],
        fractional_positions=[[0.2, 0.5, 0.5], [0.4, 0.5, 0.5]],
    )
    relaxed = Structure(
        id=relaxed_id,
        label="Relaxed",
        lattice_vectors=comparison.lattice_vectors,
        species=["Cu", "Cu", "H:mu"],
        fractional_positions=[[0.22, 0.5, 0.5], [0.41, 0.5, 0.5], [0.0, 0.5, 0.5]],
    )
    site = RelaxedMuonSite(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=relaxed_id,
        site_id=None,
        folded_fractional_position=(0.0, 0.5, 0.5),
        cartesian_position_ang=(0.0, 5.0, 5.0),
        total_energy_ev=-10.0,
    )

    dialog = DisplacementAnalysisDialog(
        workspace_name="Relaxation analysis",
        sites=[site],
        structures_by_id={comparison_id: comparison, relaxed_id: relaxed},
        comparison_options=[(comparison_id, "Host", False)],
    )

    assert len(dialog._viewer._point_overlay_specs) == 1
    assert dialog._viewer._point_overlay_specs[0].positions.shape == (2, 3)
    expected_numbers = np.asarray([atomic_numbers["Cu"], atomic_numbers["Cu"]], dtype=int)
    expected_radii = np.clip(0.9 * covalent_radii[expected_numbers], 0.2, 1.15)
    expected_colors = np.asarray(jmol_colors[expected_numbers], dtype=float)
    assert np.allclose(dialog._viewer._point_overlay_specs[0].radii, expected_radii)
    assert np.allclose(dialog._viewer._point_overlay_specs[0].colors, expected_colors)

    dialog.close()


def test_relaxation_analysis_widget_opens_displacements_without_clustering(qapp, monkeypatch) -> None:
    host = _host_structure()
    relaxed_id = uuid4()
    source = CompletedRelaxationAnalysisSource(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=relaxed_id,
        site_id=None,
        folded_fractional_position=(0.0, 0.5, 0.5),
        total_energy_ev=-10.0,
    )
    relaxed = Structure(
        id=relaxed_id,
        label="Relaxed",
        lattice_vectors=host.lattice_vectors,
        species=["Si", "Si", "H:mu"],
        fractional_positions=[[0.02, 0.0, 0.0], [0.5, 0.5, 0.5], [0.0, 0.5, 0.5]],
    )
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[source.job_id],
    )

    opened: dict[str, object] = {}

    class _FakeDialog:
        def __init__(self, *, workspace_name, sites, structures_by_id, comparison_options, parent=None):
            opened["workspace_name"] = workspace_name
            opened["site_count"] = len(sites)
            opened["comparison_options"] = list(comparison_options)

        def exec(self):
            opened["shown"] = True
            return 0

    widget = RelaxationAnalysisWidget()
    widget.set_context(
        workspaces=[workspace],
        sources=[source],
        structures={host.id: host, relaxed_id: relaxed},
        host_structure=host,
        comparison_options=[(host.id, "Host", False)],
    )
    monkeypatch.setattr("mufinder.app.relaxation_analysis_widget.DisplacementAnalysisDialog", _FakeDialog)

    widget._open_displacements(str(workspace["id"]))

    assert opened["workspace_name"] == "Relaxation analysis"
    assert opened["site_count"] == 1
    assert opened["comparison_options"] == [(host.id, "Host", False)]
    assert opened["shown"] is True

    widget.close()


def test_relaxation_analysis_workspace_show_shifted_checkbox_uses_transparent_background(qapp) -> None:
    host = _host_structure()
    source = CompletedRelaxationAnalysisSource(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=(0.0, 0.5, 0.5),
        total_energy_ev=-10.0,
    )
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[source.job_id],
    )

    widget = RelaxationAnalysisWidget()
    widget.set_context(
        workspaces=[workspace],
        sources=[source],
        structures={host.id: host},
        host_structure=host,
        comparison_options=[(host.id, "Host", False)],
    )

    checkbox = next(
        child
        for child in widget.findChildren(QCheckBox)
        if child.text() == "Show shifted positions"
    )

    assert "background: transparent;" in checkbox.styleSheet()

    widget.close()


def test_relaxation_analysis_export_cif_writes_all_relaxed_sites(qapp, tmp_path, monkeypatch) -> None:
    host = _host_structure()
    source_a = CompletedRelaxationAnalysisSource(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=(0.1, 0.1, 0.1),
        total_energy_ev=-10.0,
    )
    source_b = CompletedRelaxationAnalysisSource(
        job_id=uuid4(),
        job_label="Site B",
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=(0.2, 0.2, 0.2),
        total_energy_ev=-9.5,
    )
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[source_a.job_id, source_b.job_id],
    )
    # No clustering applied: this exercises the un-clustered path, since the
    # export must include every relaxed site, not only cluster representatives.
    widget = RelaxationAnalysisWidget()
    try:
        widget.set_context(
            workspaces=[workspace],
            sources=[source_a, source_b],
            structures={host.id: host},
            host_structure=host,
            comparison_options=[],
        )
        output_path = tmp_path / "sites.cif"
        monkeypatch.setattr(
            "mufinder.app.relaxation_analysis_widget.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: (str(output_path), "CIF Files (*.cif)"),
        )
        monkeypatch.setattr(
            "mufinder.app.relaxation_analysis_widget.QMessageBox.information",
            lambda *args, **kwargs: None,
        )

        widget._export_workspace_cif(str(workspace["id"]))

        assert output_path.exists()
        text = output_path.read_text(encoding="utf-8")
        assert "Si  Si1" in text
        assert "Si  Si2" in text
        assert "H   Mu1" in text
        assert "H   Mu2" in text
    finally:
        widget.close()


def test_relaxation_analysis_export_cif_warns_when_site_coincides_with_host_atom(
    qapp, tmp_path, monkeypatch
) -> None:
    host = _host_structure()
    # Coincides exactly with the host's first Si atom at (0, 0, 0).
    source = CompletedRelaxationAnalysisSource(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=(0.0, 0.0, 0.0),
        total_energy_ev=-10.0,
    )
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[source.job_id],
    )
    widget = RelaxationAnalysisWidget()
    try:
        widget.set_context(
            workspaces=[workspace],
            sources=[source],
            structures={host.id: host},
            host_structure=host,
            comparison_options=[],
        )
        output_path = tmp_path / "sites.cif"
        monkeypatch.setattr(
            "mufinder.app.relaxation_analysis_widget.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: (str(output_path), "CIF Files (*.cif)"),
        )
        warned: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.relaxation_analysis_widget.QMessageBox.warning",
            lambda _parent, title, message: warned.append(message),
        )

        widget._export_workspace_cif(str(workspace["id"]))

        assert output_path.exists()
        assert len(warned) == 1
        assert "Mu1" in warned[0]
    finally:
        widget.close()


def test_relaxation_analysis_export_cif_reports_when_workspace_empty(qapp, monkeypatch) -> None:
    host = _host_structure()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[],
    )
    widget = RelaxationAnalysisWidget()
    try:
        widget.set_context(
            workspaces=[workspace],
            sources=[],
            structures={host.id: host},
            host_structure=host,
            comparison_options=[],
        )
        informed: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.relaxation_analysis_widget.QMessageBox.information",
            lambda _parent, title, message: informed.append(message),
        )
        called = []
        monkeypatch.setattr(
            "mufinder.app.relaxation_analysis_widget.QFileDialog.getSaveFileName",
            lambda *args, **kwargs: called.append(True) or ("", ""),
        )

        widget._export_workspace_cif(str(workspace["id"]))

        assert informed
        assert called == []
    finally:
        widget.close()


def test_relaxation_clustering_dialog_show_does_not_warn_about_collapsed_layout(qapp) -> None:
    host = _host_structure()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[],
    )
    dialog = RelaxationClusteringDialog(
        workspace=workspace,
        host_structure=host,
        sites=[_relaxed_site(host, (0.49, 0.01, 0.0), -10.0)],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dialog.show()
        qapp.processEvents()

    collapsed_layout_warnings = [
        warning
        for warning in caught
        if "constrained_layout not applied because axes sizes collapsed to zero" in str(warning.message)
    ]
    assert collapsed_layout_warnings == []

    dialog.close()


def test_relaxation_clustering_dialog_switches_to_hierarchical_when_shown(qapp) -> None:
    host = _host_structure()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[],
    )
    dialog = RelaxationClusteringDialog(
        workspace=workspace,
        host_structure=host,
        sites=[
            _relaxed_site(host, (0.100, 0.100, 0.100), -10.0),
            _relaxed_site(host, (0.112, 0.100, 0.100), -9.9),
            _relaxed_site(host, (0.500, 0.500, 0.500), -9.8),
            _relaxed_site(host, (0.520, 0.500, 0.500), -9.7),
        ],
    )
    dialog.show()
    qapp.processEvents()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dialog._method_combo.setCurrentIndex(dialog._method_combo.findData("hier"))
        for _ in range(3):
            qapp.processEvents()

    collapsed_layout_warnings = [
        warning
        for warning in caught
        if "constrained_layout not applied because axes sizes collapsed to zero" in str(warning.message)
    ]
    assert collapsed_layout_warnings == []
    assert not dialog._hier_toolbar.isHidden()
    assert dialog._left_figure.axes

    dialog.close()


def test_relaxation_clustering_dialog_shows_only_active_method_controls(qapp) -> None:
    host = _host_structure()
    workspace = default_workspace_state(
        name="Relaxation analysis",
        charge_state="positive_muon",
        host_structure_id=host.id,
        job_ids=[],
    )
    dialog = RelaxationClusteringDialog(
        workspace=workspace,
        host_structure=host,
        sites=[_relaxed_site(host, (0.49, 0.01, 0.0), -10.0)],
    )

    assert dialog._tolerance_label.text() == "Distance tolerance (Å)"
    assert dialog._wyckoff_tolerance_label.text() == "Wyckoff tolerance (Å)"
    assert dialog._scan_button.parentWidget().objectName() == "connectedToleranceScanPanel"
    assert isinstance(dialog._scan_button.parentWidget(), QFrame)
    assert not dialog._tolerance_label.isHidden()
    assert not dialog._tolerance_input.isHidden()
    assert dialog._hierarchical_label.isHidden()
    assert dialog._hierarchical_input.isHidden()
    assert dialog._kmeans_label.isHidden()
    assert dialog._kmeans_spin.isHidden()

    dialog._method_combo.setCurrentIndex(dialog._method_combo.findData("hier"))
    qapp.processEvents()

    assert dialog._tolerance_label.isHidden()
    assert not dialog._hierarchical_label.isHidden()
    assert not dialog._hierarchical_input.isHidden()
    assert not dialog._hier_toolbar.isHidden()
    assert dialog._kmeans_label.isHidden()

    dialog._method_combo.setCurrentIndex(dialog._method_combo.findData("kmeans"))
    qapp.processEvents()

    assert not dialog._kmeans_label.isHidden()
    assert not dialog._kmeans_spin.isHidden()
    assert dialog._hier_toolbar.isHidden()
    assert dialog._hierarchical_label.isHidden()

    dialog.close()


def _square_components(count: int, *, nodes: int = 1, first_node: int = 0) -> list:
    """Return ``count`` components of ``nodes`` nodes each, numbered from ``first_node``."""

    components = []
    for index in range(count):
        members = list(range(first_node + index * nodes, first_node + (index + 1) * nodes))
        points = np.array([[1.5 * offset, 0.0] for offset in range(nodes)], dtype=float)
        components.append(_graph_component_layout(members, points))
    return components


def _rows(positions: dict[int, tuple[float, float]], components) -> list[list[int]]:
    """Group component indexes by the row (y) their first node sits on, top row first."""

    by_row: dict[float, list[int]] = {}
    for index, component in enumerate(components):
        by_row.setdefault(round(positions[component.members[0]][1], 6), []).append(index)
    return [by_row[y] for y in sorted(by_row, reverse=True)]


def test_pack_graph_components_matches_rows_to_the_target_shape() -> None:
    components = _square_components(6)

    wide = _rows(pack_graph_components(components, 6.0), components)
    landscape = _rows(pack_graph_components(components, 1.6), components)
    tall = _rows(pack_graph_components(components, 0.2), components)

    assert wide == [[0, 1, 2, 3, 4, 5]]
    assert landscape == [[0, 1, 2], [3, 4, 5]]
    assert tall == [[0], [1], [2], [3], [4], [5]]


def test_pack_graph_components_keeps_component_order_and_shape() -> None:
    components = _square_components(5, nodes=3)
    positions = pack_graph_components(components, 1.0)

    # Reading the rows top to bottom, left to right, gives the components in order.
    reading_order = [
        index
        for row in _rows(positions, components)
        for index in sorted(row, key=lambda item: positions[components[item].members[0]][0])
    ]
    assert reading_order == list(range(5))
    for component in components:
        placed = np.array([positions[member] for member in component.members])
        # Each component is only translated, never rotated or rescaled.
        assert np.allclose(placed - placed[0], component.points - component.points[0])


def test_pack_graph_components_never_overlaps_components() -> None:
    components = _square_components(4, nodes=2) + _square_components(3, nodes=4, first_node=8)
    for target_aspect in (0.3, 1.0, 1.6, 4.0):
        positions = pack_graph_components(components, target_aspect)
        boxes = []
        for component in components:
            placed = np.array([positions[member] for member in component.members])
            centre = 0.5 * (placed.min(axis=0) + placed.max(axis=0))
            half = 0.5 * np.array([component.width, component.height])
            boxes.append((centre - half, centre + half))
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                (low_a, high_a), (low_b, high_b) = boxes[first], boxes[second]
                separated = (
                    high_a[0] + _GRAPH_COMPONENT_GAP <= low_b[0] + 1e-9
                    or high_b[0] + _GRAPH_COMPONENT_GAP <= low_a[0] + 1e-9
                    or high_a[1] + _GRAPH_COMPONENT_GAP <= low_b[1] + 1e-9
                    or high_b[1] + _GRAPH_COMPONENT_GAP <= low_a[1] + 1e-9
                )
                assert separated, f"components {first} and {second} overlap at aspect {target_aspect}"


def test_pack_graph_components_draws_larger_than_one_row_or_one_column() -> None:
    components = _square_components(3, nodes=4) + _square_components(4, first_node=12)
    target_aspect = 1.4

    def drawn_scale(positions: dict[int, tuple[float, float]]) -> float:
        points = np.array(list(positions.values()))
        span = points.max(axis=0) - points.min(axis=0) + 0.9
        return min(target_aspect / span[0], 1.0 / span[1])

    chosen = drawn_scale(pack_graph_components(components, target_aspect))
    one_row = drawn_scale(pack_graph_components(components, 1e6))
    one_column = drawn_scale(pack_graph_components(components, 1e-6))
    assert chosen >= one_row and chosen >= one_column
    assert chosen > 1.2 * min(one_row, one_column)


def test_graph_edge_strip_is_a_rectangle_of_the_requested_width() -> None:
    strip = _graph_edge_strip((0.0, 0.0), (3.0, 4.0), 0.1)

    assert strip.shape == (4, 2)
    assert np.isclose(np.linalg.norm(strip[0] - strip[3]), 0.2)
    assert np.isclose(np.linalg.norm(strip[0] - strip[1]), 5.0)


def test_relaxation_connected_graph_makes_dense_component_edges_thin_and_translucent(qapp) -> None:
    host = _host_structure()
    # Eight sites within 0.25 A of each other: every pair is connected (28 edges).
    sites = [
        _relaxed_site(host, (0.100 + 0.008 * (index % 4), 0.100 + 0.008 * (index // 4), 0.100), -10.0 + 0.01 * index)
        for index in range(8)
    ]
    # A separate connected pair, at a general position so no symmetry image of it
    # falls near the dense cluster in this body-centred host.
    sites.extend(
        [
            _relaxed_site(host, (0.300, 0.050, 0.400), -9.0),
            _relaxed_site(host, (0.340, 0.050, 0.400), -9.0),
        ]
    )
    dialog = RelaxationClusteringDialog(
        workspace={"name": "Workspace"},
        host_structure=host,
        sites=sites,
    )

    axis = dialog._graph_figure.axes[0]
    edge_collections = sorted(
        (collection for collection in axis.collections if isinstance(collection, PolyCollection)),
        key=lambda collection: len(collection.get_paths()),
    )
    sparse, dense = edge_collections

    def strip_width(collection) -> float:
        corners = collection.get_paths()[0].vertices
        return float(np.linalg.norm(corners[0] - corners[3]))

    assert len(dense.get_paths()) == 28
    assert len(sparse.get_paths()) == 1
    assert 0.0 < dense.get_alpha() < 1.0
    assert sparse.get_alpha() in (None, 1.0)
    assert strip_width(dense) < strip_width(sparse)
    dialog.close()
