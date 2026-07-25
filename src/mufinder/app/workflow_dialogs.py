"""Workflow-specific configuration and inspection dialogs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QLabel, QHBoxLayout, QSpinBox, QVBoxLayout

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.param_dialog import ParameterDialog
from mufinder.domain.models import Structure
from mufinder.castep_io.schema import KeywordSpec
from mufinder.science.energy_barriers import neb_farm_distribution
from mufinder.science.site_generation import lattice_frac_to_cart


class QuantumCorrectionsConfigDialog(ParameterDialog):
    """Parameter dialog with Quantum Corrections pseudopotential selection."""

    _PSEUDOPOTENTIAL_OPTIONS = (
        ("Unknown / not set", "unknown"),
        ("Norm-conserving", "norm_conserving"),
        ("Ultrasoft", "ultrasoft"),
    )

    def __init__(
        self,
        *,
        specs: list[KeywordSpec],
        pseudopotential_family: str = "unknown",
        num_farms: int = 1,
        intro_text: str | None = None,
        read_only: bool = False,
        editable_param_names: Iterable[str] | None = None,
        pseudopotential_editable: bool = True,
        num_farms_editable: bool = True,
        species_options: Sequence[str] = (),
        parent=None,
    ) -> None:
        super().__init__(
            specs=specs,
            workflow_stage="quantum_corrections",
            locked_values={"task": "PHONON"},
            intro_text=intro_text,
            read_only=read_only,
            editable_names=editable_param_names,
            species_options=species_options,
            parent=parent,
        )
        self._pseudo_combo = QComboBox(self)
        for label, value in self._PSEUDOPOTENTIAL_OPTIONS:
            self._pseudo_combo.addItem(label, value)
        self.set_pseudopotential_family(pseudopotential_family)
        self._pseudo_combo.setEnabled(bool(pseudopotential_editable and not read_only))
        self._num_farms_spin = QSpinBox(self)
        self._num_farms_spin.setRange(1, 9999)
        self._num_farms_spin.setValue(max(1, int(num_farms)))
        self._num_farms_spin.setEnabled(bool(num_farms_editable and not read_only))

        row = QHBoxLayout()
        row.addWidget(QLabel("Pseudopotentials", self))
        row.addWidget(self._pseudo_combo, 1)

        farms_row = QHBoxLayout()
        farms_row.addWidget(QLabel("Stage 2 farms", self))
        farms_row.addWidget(self._num_farms_spin)
        farms_row.addWidget(
            QLabel(
                "Used only for the phonon continuation stage. "
                "A value of 1 disables task farming.",
                self,
            ),
            1,
        )

        root = self.layout()
        assert root is not None
        root.insertLayout(2, row)
        root.insertLayout(3, farms_row)

    def pseudopotential_family(self) -> str:
        data = self._pseudo_combo.currentData()
        return str(data).strip() or "unknown"

    def set_pseudopotential_family(self, family: str) -> None:
        target = str(family).strip() or "unknown"
        for index in range(self._pseudo_combo.count()):
            if self._pseudo_combo.itemData(index) == target:
                self._pseudo_combo.setCurrentIndex(index)
                return
        self._pseudo_combo.setCurrentIndex(0)

    def num_farms(self) -> int:
        return max(1, int(self._num_farms_spin.value()))


class NudgedElasticBandConfigDialog(ParameterDialog):
    """NEB parameter dialog with a CASTEP task-farm control.

    Task farming splits the band's images across independent CASTEP farms, so a
    sweep costs ``ceil(images / farms)`` SCFs instead of ``images``. The farm
    count is a per-path setting rather than a submission-time one because it is
    written into the job's ``.param`` as ``NUM_FARMS`` when the job directory is
    materialized, before any runtime is chosen.

    The hint below the spin box is derived live from the dialog's *current*
    image count, so an unbalanced pairing (11 images over 5 farms wastes a
    quarter of the allocation) is visible before the job is ever submitted.
    """

    def __init__(
        self,
        *,
        specs: list[KeywordSpec],
        num_farms: int = 1,
        intro_text: str | None = None,
        read_only: bool = False,
        num_farms_editable: bool = True,
        species_options: Sequence[str] = (),
        parent=None,
    ) -> None:
        super().__init__(
            specs=specs,
            workflow_stage="energy_barriers",
            locked_values={
                "task": "TRANSITIONSTATESEARCH",
                "tssearch_method": "NEB",
                "tssearch_neb_climbing": "TRUE",
            },
            intro_text=intro_text,
            read_only=read_only,
            species_options=species_options,
            parent=parent,
        )
        self._num_farms_spin = QSpinBox(self)
        # CASTEP rejects NUM_FARMS outside 1..999 (parameters_validate).
        self._num_farms_spin.setRange(1, 999)
        self._num_farms_spin.setValue(max(1, int(num_farms)))
        self._num_farms_spin.setEnabled(bool(num_farms_editable and not read_only))
        self._num_farms_spin.valueChanged.connect(lambda *_: self._refresh_farm_hint())

        self._farm_hint = QLabel("", self)
        self._farm_hint.setWordWrap(True)

        farms_row = QHBoxLayout()
        farms_row.addWidget(QLabel("Task farms", self))
        farms_row.addWidget(self._num_farms_spin)
        farms_row.addWidget(self._farm_hint, 1)

        root = self.layout()
        assert root is not None
        # Sit directly above the button box. An absolute index would drift with
        # the optional catalog-notice and intro labels the base dialog inserts.
        root.insertLayout(root.count() - 1, farms_row)
        self._refresh_farm_hint()

    def num_farms(self) -> int:
        """Return the selected farm count (1 when farming is disabled)."""

        return max(1, int(self._num_farms_spin.value()))

    def _on_editor_changed(self) -> None:
        # The base class rebuilds rows during __init__, before this subclass has
        # built its widgets, so the hint may not exist on the first call.
        super()._on_editor_changed()
        self._refresh_farm_hint()

    def set_values(self, values, *, preferred_exclusive=None) -> None:
        # Callers apply the row's stored configuration right after construction.
        # That path rebuilds rows rather than firing editor signals, so the hint
        # would otherwise keep describing the default image count.
        super().set_values(values, preferred_exclusive=preferred_exclusive)
        self._refresh_farm_hint()

    def _current_image_count(self) -> int | None:
        raw = self.collect_values().get("tssearch_max_path_points")
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return None

    def _refresh_farm_hint(self) -> None:
        hint = getattr(self, "_farm_hint", None)
        if hint is None:
            return
        farms = self.num_farms()
        if farms <= 1:
            hint.setText("Task farming disabled: every image runs on the full set of MPI ranks.")
            return
        images = self._current_image_count()
        if images is None:
            hint.setText(f"Needs at least {farms} MPI ranks, ideally a multiple of {farms}.")
            return
        distribution = neb_farm_distribution(images=images, farms=farms)
        split = "/".join(str(count) for count in distribution.steady_counts)
        text = (
            f"{images} images over {farms} farms -> {split} per farm "
            f"({distribution.efficiency:.0%} balance). "
            f"Needs at least {farms} MPI ranks, ideally a multiple of {farms}."
        )
        if distribution.idle_farms:
            text += f" {distribution.idle_farms} farm(s) would sit idle."
        hint.setText(text)


class EnergyBarrierPathViewerDialog(QDialog):
    """Preview an Energy Barriers path in the undistorted supercell."""

    def __init__(
        self,
        *,
        structure: Structure,
        start_fractional: Sequence[float],
        end_fractional: Sequence[float],
        title: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(980, 760)

        self._structure = structure
        self._start_fractional = tuple(float(value) for value in start_fractional[:3])
        self._end_fractional = tuple(float(value) for value in end_fractional[:3])
        self._viewer = AtomicStructureViewer(self)
        self._viewer_shutdown = False

        layout = QVBoxLayout(self)

        summary = QLabel(
            "The viewer uses the pristine supercell matching the relaxed endpoint cell. "
            "Muon start and end positions are shown as overlays.",
            self,
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        controls = QHBoxLayout()
        self._show_arrow = QCheckBox("Show path arrow", self)
        self._show_arrow.setChecked(True)
        self._show_arrow.toggled.connect(self._refresh_overlays)
        controls.addWidget(self._show_arrow)
        controls.addStretch(1)
        layout.addLayout(controls)

        layout.addWidget(self._viewer, 1)

        self._viewer.set_structure(structure)
        self._refresh_overlays()

    def _shutdown_viewer(self) -> None:
        if self._viewer_shutdown:
            return
        self._viewer_shutdown = True
        try:
            self._viewer.shutdown()
        except RuntimeError:
            pass

    def done(self, result: int) -> None:  # noqa: D401 - Qt API override.
        """Finish the dialog after releasing viewer resources."""

        self._shutdown_viewer()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self._shutdown_viewer()
        super().closeEvent(event)

    def _overlay_positions(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        start = lattice_frac_to_cart(self._start_fractional, self._structure.lattice_vectors)
        end = lattice_frac_to_cart(self._end_fractional, self._structure.lattice_vectors)
        return (
            tuple(float(value) for value in start),
            tuple(float(value) for value in end),
        )

    def _refresh_overlays(self) -> None:
        start, end = self._overlay_positions()
        self._viewer.set_point_overlays(
            [
                {
                    "positions": [start],
                    "colors": ["#ff7a18"],
                    "radii": [0.28],
                },
                {
                    "positions": [end],
                    "colors": ["#0f8aa8"],
                    "radii": [0.28],
                },
            ]
        )
        if self._show_arrow.isChecked():
            self._viewer.set_line_overlays(
                [
                    {
                        "start": start,
                        "end": end,
                        "color": "#2f6f4f",
                        "width": 7.0,
                        "alpha": 0.92,
                        "arrow": True,
                    }
                ]
            )
            return
        self._viewer.set_line_overlays(None)
