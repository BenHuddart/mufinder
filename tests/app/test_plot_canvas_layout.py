"""Plots must keep every tick, axis label and legend inside their canvas at small sizes.

Each case opens a plot dialog at its minimum size and at the window sizes small
laptops actually give it, draws every visible matplotlib canvas, and checks
that no axes decoration extends past the figure edge, where Qt would clip it.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.patches import PathPatch
from matplotlib.transforms import Bbox
import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QDialog, QSizePolicy
import pytest

from mufinder.app.convergence_examiner import ConvergenceExaminerDialog
from mufinder.app.convergence_types import CutoffRunState
from mufinder.app.magnetic_field_widget import MagneticFieldDistributionDialog
from mufinder.app.plot_canvas import DEFAULT_MINIMUM_CANVAS_SIZE, create_plot_canvas
from mufinder.app.relaxation_analysis_widget import (
    GRAPH_NODE_GID,
    DisplacementAnalysisDialog,
    RelaxationClusteringDialog,
)
from mufinder.app.uep_clustering_dialog import UepClusteringDialog
from mufinder.domain.models import Structure
from mufinder.science.relaxation_analysis import RelaxedMuonSite
from mufinder.science.uep import StationaryPointClassification, UepOptimizationResult

# Window sizes the dialogs get on small screens, beyond each dialog's own minimum:
# a 1080p Windows laptop at 150 % scaling (1280x720 logical, minus the taskbar)
# and a 13" MacBook (about 1440x820 usable, through ``clamped_size``'s 90 %).
SMALL_SCREEN_SIZES = (QSize(1150, 620), QSize(1240, 740))

# Rounding of text extents can put a label a fraction of a pixel past the edge.
_OVERFLOW_TOLERANCE_PX = 1.0


def _cubic_host(*, length: float = 5.0) -> Structure:
    return Structure(
        id=uuid4(),
        label="Host",
        lattice_vectors=[[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )


def _relaxed_site(host: Structure, fractional: tuple[float, float, float], energy: float) -> RelaxedMuonSite:
    length = host.lattice_vectors[0][0]
    return RelaxedMuonSite(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=uuid4(),
        site_id=None,
        folded_fractional_position=fractional,
        cartesian_position_ang=tuple(value * length for value in fractional),
        total_energy_ev=energy,
    )


_SPREAD_SITE_FRACTIONS = (
    (0.10, 0.10, 0.10),
    (0.12, 0.10, 0.10),
    (0.30, 0.30, 0.30),
    (0.32, 0.30, 0.31),
    (0.60, 0.20, 0.40),
    (0.61, 0.21, 0.40),
    (0.80, 0.80, 0.10),
    (0.25, 0.75, 0.50),
)
# Twelve sites 0.2 A apart form one long connected component at the default
# 0.25 A tolerance, plus two isolated sites: a dense graph for the node checks.
_CHAIN_SITE_FRACTIONS = (
    *((0.10 + 0.04 * step, 0.50, 0.50) for step in range(12)),
    (0.80, 0.10, 0.10),
    (0.20, 0.90, 0.90),
)


def _relaxation_clustering(
    method_index: int,
    fractions: tuple[tuple[float, float, float], ...] = _SPREAD_SITE_FRACTIONS,
) -> QDialog:
    host = _cubic_host()
    sites = [_relaxed_site(host, fraction, -10.0 + 0.05 * index) for index, fraction in enumerate(fractions)]
    dialog = RelaxationClusteringDialog(workspace={"name": "Workspace"}, host_structure=host, sites=sites)
    dialog._method_combo.setCurrentIndex(method_index)
    if method_index == 0:
        dialog._scan_tolerances()
    return dialog


def _displacements() -> QDialog:
    comparison_id = uuid4()
    relaxed_id = uuid4()
    lattice = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    positions = np.random.default_rng(0).random((30, 3)).tolist()
    species = ["Fe", "O", "O"] * 10
    comparison = Structure(
        id=comparison_id,
        label="Host",
        lattice_vectors=lattice,
        species=species,
        fractional_positions=positions,
    )
    relaxed = Structure(
        id=relaxed_id,
        label="Relaxed",
        lattice_vectors=lattice,
        species=[*species, "H:mu"],
        fractional_positions=[[x + 0.01, y, z] for x, y, z in positions] + [[0.5, 0.5, 0.5]],
    )
    site = RelaxedMuonSite(
        job_id=uuid4(),
        job_label="Site A",
        charge_state="positive_muon",
        structure_id=relaxed_id,
        site_id=None,
        folded_fractional_position=(0.5, 0.5, 0.5),
        cartesian_position_ang=(5.0, 5.0, 5.0),
        total_energy_ev=-10.0,
    )
    return DisplacementAnalysisDialog(
        workspace_name="Workspace",
        sites=[site],
        structures_by_id={comparison_id: comparison, relaxed_id: relaxed},
        comparison_options=[(comparison_id, "Host", False)],
    )


def _uep_optimization(fractional: tuple[float, float, float], energy: float) -> UepOptimizationResult:
    position = tuple(value * 5.0 for value in fractional)
    return UepOptimizationResult(
        start_position_ang=position,
        optimized_position_ang=position,
        optimized_fractional_position=fractional,
        classical_energy_ev=energy,
        zero_point_energy_ev=0.05,
        total_energy_ev=energy,
        success=True,
        message="ok",
        iterations=4,
        classification=StationaryPointClassification(
            label="minimum",
            eigenvalues=(1.0, 1.1, 1.2),
            positive_count=3,
            negative_count=0,
            near_zero_count=0,
        ),
    )


def _uep_clustering(state: dict[str, object]) -> QDialog:
    host = Structure(
        id=uuid4(),
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"pbc": [True, True, True]},
    )
    optimizations = [
        _uep_optimization((0.10, 0.10, 0.10), 0.01),
        _uep_optimization((0.12, 0.09, 0.11), 0.02),
        _uep_optimization((0.65, 0.65, 0.65), 0.50),
        _uep_optimization((0.66, 0.67, 0.64), 0.52),
    ]
    return UepClusteringDialog(
        source_job_label="uep_source",
        source_structure=host,
        viewer_host_structure=host,
        optimizations=optimizations,
        initial_state=state,
    )


def _field_distribution() -> QDialog:
    class _Distribution:
        site_label = "site 1"
        histogram_centres_T = np.linspace(0.0, 0.2, 40)
        histogram_counts = 100.0 * np.exp(-(((np.linspace(0.0, 0.2, 40) - 0.1) / 0.03) ** 2))

    dialog = MagneticFieldDistributionDialog(_Distribution())
    # Guard the fixture itself: a stub the dialog cannot read renders only the
    # "No distribution samples" placeholder, which has no axes to clip.
    assert dialog._figure.axes[0].lines, "field-distribution fixture drew no histogram"
    return dialog


def _convergence_examiner() -> QDialog:
    states = {
        cutoff: CutoffRunState(status="completed", total_energy=energy, max_force=force)
        for cutoff, energy, force in (
            (300.0, -10.0, 0.30),
            (400.0, -10.01, 0.20),
            (500.0, -10.0105, 0.19),
            (600.0, -10.0106, 0.185),
        )
    }
    return ConvergenceExaminerDialog(states, num_atoms=2)


PLOT_DIALOGS: dict[str, Callable[[], QDialog]] = {
    "relaxation-clustering-connected": lambda: _relaxation_clustering(0),
    "relaxation-clustering-hierarchical": lambda: _relaxation_clustering(1),
    "relaxation-clustering-kmeans": lambda: _relaxation_clustering(2),
    "relaxation-displacements": _displacements,
    "uep-clustering-hierarchical": lambda: _uep_clustering({}),
    "uep-clustering-kmeans": lambda: _uep_clustering({"method": "kmeans", "kmeans_k": 2}),
    "magnetic-field-distribution": _field_distribution,
    "convergence-examiner": _convergence_examiner,
}


def _decoration_boxes(axes, renderer) -> list[Bbox]:
    """Return the full extents of one axes' tick labels, axis labels, titles and legend.

    ``Axes.get_tightbbox`` is not enough: like the layout engines, it ignores
    how far an axis label runs *along* its axis, so a y-label taller than the
    plot would pass unnoticed. Each ``Axis.get_tightbbox`` call here measures
    the whole label.
    """

    boxes = [axis.get_tightbbox(renderer) for axis in (axes.xaxis, axes.yaxis) if axis.get_visible()]
    boxes.extend(
        title.get_window_extent(renderer)
        for title in (axes.title, axes._left_title, axes._right_title)
        if title.get_visible() and title.get_text()
    )
    legend = axes.get_legend()
    if legend is not None and legend.get_visible():
        boxes.append(legend.get_window_extent(renderer))
    return [box for box in boxes if box is not None and box.width > 0 and box.height > 0]


def _decoration_overflow_px(canvas: FigureCanvasQTAgg) -> dict[str, float]:
    """Return how far any axes' ticks, labels or legend reach past each figure edge."""

    canvas.draw()
    renderer = canvas.get_renderer()
    figure_box = canvas.figure.bbox
    overflow = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    for axes in canvas.figure.axes:
        if not axes.get_visible() or not axes.axison:
            continue
        for box in _decoration_boxes(axes, renderer):
            overflow["left"] = max(overflow["left"], figure_box.x0 - box.x0)
            overflow["right"] = max(overflow["right"], box.x1 - figure_box.x1)
            overflow["bottom"] = max(overflow["bottom"], figure_box.y0 - box.y0)
            overflow["top"] = max(overflow["top"], box.y1 - figure_box.y1)
    return {side: round(value, 1) for side, value in overflow.items() if value > _OVERFLOW_TOLERANCE_PX}


@pytest.mark.parametrize("name", sorted(PLOT_DIALOGS))
def test_plot_dialog_keeps_axis_decorations_inside_canvas_on_small_screens(qapp, name: str) -> None:
    dialog = PLOT_DIALOGS[name]()
    try:
        dialog.show()
        qapp.processEvents()
        for size in (dialog.minimumSize(), *SMALL_SCREEN_SIZES):
            dialog.resize(size)
            for _ in range(4):
                qapp.processEvents()
            canvases = [
                canvas
                for canvas in dialog.findChildren(FigureCanvasQTAgg)
                if canvas.isVisibleTo(dialog) and canvas.figure.axes
            ]
            assert canvases, f"{name} shows no plot"
            for canvas in canvases:
                where = f"{name} at {dialog.width()}x{dialog.height()}, canvas {canvas.width()}x{canvas.height()}"
                assert canvas.width() >= canvas.minimumWidth(), f"{where}: squeezed below its minimum width"
                assert canvas.height() >= canvas.minimumHeight(), f"{where}: squeezed below its minimum height"
                overflow = _decoration_overflow_px(canvas)
                assert not overflow, f"{where}: labels clipped by {overflow} px"
    finally:
        dialog.close()


@pytest.mark.parametrize(
    "fractions",
    [_SPREAD_SITE_FRACTIONS, _CHAIN_SITE_FRACTIONS],
    ids=["spread-components", "long-chain"],
)
def test_connected_graph_nodes_stay_inside_the_axes_and_apart(qapp, fractions) -> None:
    """Graph nodes scale with the canvas instead of spilling past it or overlapping.

    Fixed-size markers (980 pt^2 with an 18 pt label) used to be cut off at the
    axes edge and to run into each other once the dialog was small.
    """

    dialog = _relaxation_clustering(0, fractions)
    try:
        dialog.show()
        qapp.processEvents()
        for size in (dialog.minimumSize(), *SMALL_SCREEN_SIZES, QSize(1600, 1000)):
            dialog.resize(size)
            for _ in range(4):
                qapp.processEvents()
            at_minimum = size == dialog.minimumSize()
            canvas = dialog._graph_canvas
            canvas.draw()
            renderer = canvas.get_renderer()
            axes = canvas.figure.axes[0]
            circles = [patch for patch in axes.patches if patch.get_gid() == GRAPH_NODE_GID]
            labels = [patch for patch in axes.patches if isinstance(patch, PathPatch)]
            where = f"{len(fractions)} sites at {dialog.width()}x{dialog.height()}"
            assert len(circles) == len(labels) == len(fractions), where
            # Node borders are filled discs, not strokes, so nothing reaches past the radius.
            assert all(circle.get_linewidth() == 0.0 for circle in circles), where

            inner = axes.bbox.padded(_OVERFLOW_TOLERANCE_PX)
            extents = [circle.get_window_extent(renderer) for circle in circles]
            for extent in extents:
                assert inner.x0 <= extent.x0 and extent.x1 <= inner.x1, f"{where}: node cut off horizontally"
                assert inner.y0 <= extent.y0 and extent.y1 <= inner.y1, f"{where}: node cut off vertically"
            for label, extent in zip(labels, extents, strict=True):
                label_extent = label.get_window_extent(renderer)
                assert extent.x0 < label_extent.x0 and label_extent.x1 < extent.x1, f"{where}: label wider than node"
                assert extent.y0 < label_extent.y0 and label_extent.y1 < extent.y1, f"{where}: label taller than node"
            centres = np.array([[(e.x0 + e.x1) / 2.0, (e.y0 + e.y1) / 2.0] for e in extents])
            radius_px = extents[0].width / 2.0
            gaps = np.linalg.norm(centres[:, np.newaxis, :] - centres[np.newaxis, :, :], axis=2)
            np.fill_diagonal(gaps, np.inf)
            assert gaps.min() > 2.0 * radius_px + 2.0, f"{where}: nodes overlap"
            # Nodes stay large enough to read their numbers on real laptop windows. At the
            # dialog's absolute minimum a twelve-node chain may get smaller, but must not collapse.
            minimum_radius_px = 5.0 if at_minimum else 8.0
            assert radius_px >= minimum_radius_px, f"{where}: nodes shrank to {radius_px:.1f} px"
    finally:
        dialog.close()


def test_create_plot_canvas_uses_constrained_layout_and_a_minimum_size(qapp) -> None:
    figure, canvas = create_plot_canvas(figsize=(5.0, 3.0), facecolor="#ffffff")

    assert isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine)
    assert canvas.figure is figure
    assert (canvas.minimumWidth(), canvas.minimumHeight()) == DEFAULT_MINIMUM_CANVAS_SIZE
    assert canvas.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding
    assert canvas.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding
    assert not canvas.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    # Refresh paths call ``figure.clear()`` before redrawing; the layout must survive it.
    figure.add_subplot(111)
    figure.clear()
    assert isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine)


def test_create_plot_canvas_transparent_background_and_custom_minimum(qapp) -> None:
    figure, canvas = create_plot_canvas(figsize=(5.0, 3.0), transparent=True, minimum_size=(300, 240))

    assert figure.patch.get_alpha() == 0.0
    assert canvas.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert (canvas.minimumWidth(), canvas.minimumHeight()) == (300, 240)
