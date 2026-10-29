"""Workflow-specific configuration and inspection dialogs."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QLabel,
    QHBoxLayout,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.workflow_parameters import (
    kpoint_exclusive_hint,
    merge_batch_advanced,
    merge_batch_values,
    mixed_parameter_names,
    shared_parameter_values,
)
from mufinder.domain.models import Structure
from mufinder.castep_io.schema import KeywordSpec
from mufinder.science.energy_barriers import neb_farm_distribution
from mufinder.science.site_generation import lattice_frac_to_cart

#: Row key of a workflow table: a row id, a cutoff energy or a k-point grid.
RowKeyT = TypeVar("RowKeyT", bound=Hashable)


@dataclass(frozen=True)
class RowParameterRequest(Generic[RowKeyT]):
    """What a workflow widget asks of the row parameter dialog.

    ``row_values`` and ``row_advanced`` hold, for every targeted row, its
    effective parameters and its shown advanced names; ``values`` is the dialog
    seed (the parameters the rows share, plus any widget-specific pre-fill).
    ``subject`` names the dialog: the title reads ``Configure <subject>``, or
    ``Inspect <subject>`` when ``read_only`` is set.
    """

    specs: Sequence[KeywordSpec]
    workflow_stage: str
    subject: str
    values: Mapping[str, object]
    row_values: Mapping[RowKeyT, Mapping[str, object]]
    row_advanced: Mapping[RowKeyT, Iterable[str]]
    sanitize: Callable[[Mapping[str, object]], dict[str, object]]
    locked_values: Mapping[str, str] = field(default_factory=dict)
    read_only: bool = False
    intro_text: str | None = None
    species_options: Sequence[str] = ()


@dataclass(frozen=True)
class RowParameterResult(Generic[RowKeyT]):
    """The accepted dialog, resolved for every targeted row.

    ``values`` and ``advanced_names`` are what the dialog returned, sanitised.
    ``row_values`` and ``row_advanced`` apply them to each row: a parameter that
    differed between the rows and was left untouched keeps each row's own value
    and advanced-name visibility.
    """

    values: dict[str, object]
    advanced_names: frozenset[str]
    row_values: dict[RowKeyT, dict[str, object]]
    row_advanced: dict[RowKeyT, set[str]]


def run_row_parameter_dialog(
    request: RowParameterRequest[RowKeyT],
    *,
    parent: QWidget | None = None,
) -> RowParameterResult[RowKeyT] | None:
    """Open the parameter dialog for one or more workflow rows.

    Returns ``None`` when there are no rows, when the dialog is cancelled, and
    when it opened read-only to inspect locked rows.
    """

    if not request.row_values:
        return None
    seed_advanced: set[str] = set()
    for names in request.row_advanced.values():
        seed_advanced.update(names)
    dialog = ParameterDialog(
        specs=list(request.specs),
        workflow_stage=request.workflow_stage,
        locked_values=dict(request.locked_values),
        intro_text=request.intro_text,
        read_only=request.read_only,
        species_options=request.species_options,
        values=dict(request.values),
        preferred_exclusive=kpoint_exclusive_hint(request.row_values.values()),
        advanced_names=seed_advanced,
        mixed_names=mixed_parameter_names(request.row_values.values()),
        parent=parent,
    )
    dialog.setWindowTitle(f"{'Inspect' if request.read_only else 'Configure'} {request.subject}")
    try:
        if dialog.exec() != QDialog.DialogCode.Accepted or request.read_only:
            return None
        values = request.sanitize(dialog.collect_values())
        advanced = frozenset(dialog.advanced_names())
        untouched = dialog.untouched_mixed_names()
    finally:
        # The dialog is a child of ``parent``; without this every open would
        # stay alive under it, editors and all.
        dialog.deleteLater()
    return RowParameterResult(
        values=values,
        advanced_names=advanced,
        row_values={
            key: merge_batch_values(values, row_values, untouched)
            for key, row_values in request.row_values.items()
        },
        row_advanced={
            key: merge_batch_advanced(advanced, request.row_advanced.get(key, ()), untouched)
            for key in request.row_values
        },
    )


@dataclass(frozen=True)
class SourceConfigureRequest:
    """What a source table's Configure dialog opens with.

    ``keys`` are the targeted source rows and ``row_values`` their effective
    parameters, in the same order; ``advanced_names`` is every advanced name
    any of them shows. The dialog opens on the values the rows share and marks
    the rest as mixed. ``num_farms`` is the farm count every row agrees on, else
    1.
    """

    keys: tuple[str, ...]
    row_values: tuple[Mapping[str, object], ...]
    advanced_names: frozenset[str]
    title: str
    intro_text: str
    read_only: bool
    num_farms: int
    species_options: tuple[str, ...]


@dataclass(frozen=True)
class SourceConfigureResult:
    """An accepted source Configure dialog: what to store for every targeted row.

    ``untouched_mixed`` names the mixed parameters the user left alone, which
    each row keeps from its own stored configuration.
    """

    values: dict[str, object]
    advanced_names: frozenset[str]
    untouched_mixed: frozenset[str]


def run_source_configure_dialog(
    dialog: ParameterDialog,
    request: SourceConfigureRequest,
) -> SourceConfigureResult | None:
    """Seed ``dialog`` from ``request``, run it, and return what it produced.

    Returns ``None`` when the dialog is cancelled, and when it opened read-only
    to inspect rows that can no longer change.
    """

    dialog.setWindowTitle(request.title)
    dialog.set_values(
        shared_parameter_values(request.row_values),
        preferred_exclusive=kpoint_exclusive_hint(request.row_values),
    )
    dialog.set_advanced_names(tuple(sorted(request.advanced_names)))
    dialog.set_mixed_names(mixed_parameter_names(request.row_values))
    if dialog.exec() != QDialog.DialogCode.Accepted or request.read_only:
        return None
    return SourceConfigureResult(
        values=dict(dialog.collect_values()),
        advanced_names=frozenset(dialog.advanced_names()),
        untouched_mixed=frozenset(dialog.untouched_mixed_names()),
    )


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
        species_options: Sequence[str] = (),
        parent=None,
    ) -> None:
        super().__init__(
            specs=specs,
            workflow_stage="quantum_corrections",
            locked_values={"task": "PHONON"},
            intro_text=intro_text,
            read_only=read_only,
            species_options=species_options,
            parent=parent,
        )
        self._pseudo_combo = QComboBox(self)
        for label, value in self._PSEUDOPOTENTIAL_OPTIONS:
            self._pseudo_combo.addItem(label, value)
        self.set_pseudopotential_family(pseudopotential_family)
        self._pseudo_combo.setEnabled(not read_only)
        self._num_farms_spin = QSpinBox(self)
        self._num_farms_spin.setRange(1, 9999)
        self._num_farms_spin.setValue(max(1, int(num_farms)))
        self._num_farms_spin.setEnabled(not read_only)

        row = QHBoxLayout()
        row.addWidget(QLabel("Pseudopotentials", self))
        row.addWidget(self._pseudo_combo, 1)

        farms_row = QHBoxLayout()
        farms_row.addWidget(QLabel("Farms", self))
        farms_row.addWidget(self._num_farms_spin)
        farms_row.addWidget(QLabel("A value of 1 disables task farming.", self), 1)

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


class HyperfineCouplingConfigDialog(ParameterDialog):
    """Parameter dialog for Hyperfine Coupling rows: one magres job each, task locked to MAGRES."""

    def __init__(
        self,
        *,
        specs: list[KeywordSpec],
        intro_text: str | None = None,
        read_only: bool = False,
        species_options: Sequence[str] = (),
        parent=None,
    ) -> None:
        super().__init__(
            specs=specs,
            workflow_stage="hyperfine_coupling",
            locked_values={"task": "MAGRES"},
            intro_text=intro_text,
            read_only=read_only,
            species_options=species_options,
            parent=parent,
        )


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

    def _compute_blocked_names(self) -> set[str]:
        # NUM_FARMS has a dedicated control, and its value is *also* read back
        # from the row's extras to decide which output filenames a finished job
        # wrote and whether the requested MPI ranks can serve it. A second,
        # keyword-row copy reachable through "Add Parameter" would win in the
        # rendered .param while those checks kept using the spin box value --
        # CASTEP running one farm count while MuFinder assumed another. Block it
        # so the control is the only source.
        blocked = super()._compute_blocked_names()
        blocked.add("num_farms")
        return blocked

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
