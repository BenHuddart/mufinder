"""Magnetic Fields dipolar-tensor and nuclear-width dialogs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.magnetic_component_dialogs import selector_label
from mufinder.app.result_notes import unrelaxed_results_note
from mufinder.app.theme import (
    style_hint_text,
    style_inline_group,
    style_section_title,
    track_item_view_corners,
)
from mufinder.app.workflow_table_utils import create_workflow_table
from mufinder.domain.constants import MUON_GAMMA_OVER_2PI_MHZ_T
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.persistence.codec import float_triplet, float_triplet_or_none, to_jsonable
from mufinder.science.magnetic_fields import (
    CONTACT_COUPLING_UNIT_OPTIONS,
    CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
    MagneticFieldDependencyError,
    MagneticFieldOptions,
    assigned_magnetic_atom_indices,
)
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    resolve_atom_selector,
    viable_atom_selectors,
)
from mufinder.science.nuclear_relaxation import NuclearRelaxationOptions
from mufinder.services.magnetic_field_results import (
    UNRELAXED_ESTIMATE_SUFFIX,
    is_unrelaxed_site_source,
)
from mufinder.services.magnetic_field_sites import MagneticMuonSiteOption


class NuclearRelaxationDialog(QDialog):
    """Configure and display nuclear dipolar relaxation calculations."""

    _MODE_LABELS = {
        "tf_single_crystal": "TF single crystal",
        "zf_single_crystal": "ZF single crystal",
        "tf_polycrystal": "TF polycrystal",
        "zf_polycrystal": "ZF polycrystal",
    }

    def __init__(
        self,
        *,
        muon_site: MagneticMuonSiteOption | None,
        distortion_available: bool,
        elements: Sequence[str] = (),
        options_state: Mapping[str, object] | None = None,
        results_state: Mapping[str, object] | None = None,
        calculate_callback: Callable[["NuclearRelaxationDialog"], object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nuclear Relaxation")
        self.resize(720, 600)
        self._muon_site = muon_site
        self._distortion_available = bool(distortion_available)
        self._calculate_callback = calculate_callback
        self._results_state: dict[str, object] = (
            dict(results_state) if isinstance(results_state, Mapping) else {}
        )

        self._regime_combo = QComboBox(self)
        self._regime_combo.addItem("Zero field", "zf")
        self._regime_combo.addItem("Transverse field", "tf")
        self._sample_combo = QComboBox(self)
        self._sample_combo.addItem("Polycrystal", "polycrystal")
        self._sample_combo.addItem("Single crystal", "single_crystal")
        self._direction_x, self._direction_y, self._direction_z = (
            DecimalLineEdit(text=text, minimum=-1_000_000.0, maximum=1_000_000.0, decimals=12)
            for text in ("0.0", "0.0", "1.0")
        )
        for edit in (self._direction_x, self._direction_y, self._direction_z):
            edit.setObjectName("nuclearDirectionInput")
        self._radius = DecimalLineEdit(
            text="", minimum=0.0, maximum=1_000_000.0, decimals=12, placeholder="auto"
        )
        self._rtol = DecimalLineEdit(text="1e-6", minimum=0.0, maximum=1_000_000.0, decimals=12)
        for edit in (
            self._direction_x,
            self._direction_y,
            self._direction_z,
            self._radius,
            self._rtol,
        ):
            edit.setMinimumWidth(92)
        self._element_checks = {
            str(element): QCheckBox(str(element), self) for element in dict.fromkeys(elements)
        }
        # Without checkboxes the saved filter is carried through unchanged.
        self._saved_elements: tuple[str, ...] | None = None
        for check in self._element_checks.values():
            check.setChecked(True)
            check.setToolTip("Include this element's nuclei in the second-moment sum.")
        self._quadrupolar = QCheckBox("Include ZF quadrupolar correction", self)
        self._distortion = QCheckBox("Apply muon-induced distortion correction", self)
        self._distortion.setEnabled(self._distortion_available)
        self._distortion.setToolTip(
            "Available when the selected relaxation-cluster site has a derived distorted structure."
        )
        self._status = QLabel(
            "Ready." if muon_site is not None else "Select a muon site before calculating.", self
        )
        self._status.setWordWrap(True)
        style_hint_text(self._status)
        self._calculate_button = QPushButton("Calculate", self)
        self._calculate_button.clicked.connect(self._calculate)
        self._calculate_button.setEnabled(muon_site is not None)

        self._summary_table = create_workflow_table(
            ("Quantity", "Value"),
            resize_to_contents_columns=(0,),
            stretch_last_section=False,
            accessible_name="Results",
        )
        self._element_table = create_workflow_table(
            ("Element", "σ² contribution (μs⁻²)"),
            resize_to_contents_columns=(0,),
            stretch_last_section=False,
            accessible_name="Element Contributions",
        )
        self._estimate_note = unrelaxed_results_note(self)

        self._build_layout()
        self._restore_options(options_state)
        self._sync_mode_controls()
        self._regime_combo.currentIndexChanged.connect(self._sync_mode_controls)
        self._sample_combo.currentIndexChanged.connect(self._sync_mode_controls)
        self._populate_results(self._results_state)

    def options_state(self) -> dict[str, object]:
        return self.options().to_json()

    def results_state(self) -> dict[str, object]:
        return dict(self._results_state)

    def options(self) -> NuclearRelaxationOptions:
        mode = f"{self._regime_combo.currentData()}_{self._sample_combo.currentData()}"
        radius_text = self._radius.text().strip()
        radius = None if not radius_text else float(radius_text)
        return NuclearRelaxationOptions(
            mode=mode,  # type: ignore[arg-type]
            elements=self._selected_elements(),
            direction=(
                float(self._direction_x.text() or "0"),
                float(self._direction_y.text() or "0"),
                float(self._direction_z.text() or "1"),
            ),
            include_quadrupolar=self._quadrupolar.isChecked(),
            radius=radius,
            convergence_rtol=float(self._rtol.text() or "1e-6"),
            use_distortion_correction=self._distortion.isChecked(),
        )

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Nuclear Relaxation", self)
        style_section_title(title)
        layout.addWidget(title)

        site_group = QGroupBox("Muon Site", self)
        site_layout = QVBoxLayout(site_group)
        site_layout.addWidget(QLabel(_muon_site_summary(self._muon_site), site_group))
        layout.addWidget(site_group)

        mode_group = QGroupBox("Mode", self)
        mode_form = QFormLayout(mode_group)
        mode_form.addRow("Field regime", self._regime_combo)
        mode_form.addRow("Sample", self._sample_combo)
        direction_group = QWidget(mode_group)
        style_inline_group(direction_group)
        direction_layout = QHBoxLayout(direction_group)
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.addWidget(QLabel("x"))
        direction_layout.addWidget(self._direction_x)
        direction_layout.addWidget(QLabel("y"))
        direction_layout.addWidget(self._direction_y)
        direction_layout.addWidget(QLabel("z"))
        direction_layout.addWidget(self._direction_z)
        mode_form.addRow("Crystal-axis direction", direction_group)
        mode_form.addRow("Radius (Å)", self._radius)
        mode_form.addRow("Convergence rtol", self._rtol)
        if self._element_checks:
            elements_group = QWidget(mode_group)
            style_inline_group(elements_group)
            elements_layout = QHBoxLayout(elements_group)
            elements_layout.setContentsMargins(0, 0, 0, 0)
            elements_layout.setSpacing(24)
            for check in self._element_checks.values():
                elements_layout.addWidget(check)
            elements_layout.addStretch(1)
            mode_form.addRow("Elements", elements_group)
        mode_form.addRow("", self._quadrupolar)
        mode_form.addRow("", self._distortion)
        layout.addWidget(mode_group)

        action_row = QHBoxLayout()
        action_row.addWidget(self._calculate_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        layout.addWidget(self._status)

        results_group = QGroupBox("Results", self)
        results_layout = QVBoxLayout(results_group)
        results_layout.addWidget(self._estimate_note)
        results_layout.addWidget(self._summary_table)
        results_layout.addWidget(self._element_table)
        layout.addWidget(results_group, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _selected_elements(self) -> tuple[str, ...] | None:
        """Return the ticked elements, or ``None`` (every element) when all are ticked."""

        if not self._element_checks:
            return self._saved_elements
        ticked = tuple(
            element for element, check in self._element_checks.items() if check.isChecked()
        )
        return None if len(ticked) == len(self._element_checks) else ticked

    def _calculate(self) -> None:
        if self._calculate_callback is None:
            return
        if self._selected_elements() == ():
            self._status.setText("Tick at least one element to include in the sum.")
            return
        try:
            result = self._calculate_callback(self)
        except ValueError as exc:
            self._status.setText(f"Calculation failed: {exc}")
            return
        payload = result.to_json() if hasattr(result, "to_json") else dict(result)
        jsonable = to_jsonable(payload)
        self._results_state = dict(jsonable) if isinstance(jsonable, Mapping) else {}
        if self._results_state and self._muon_site is not None:
            self._results_state["site_source"] = self._muon_site.source
        elements = self._selected_elements()
        if self._results_state and elements is not None:
            self._results_state["elements"] = list(elements)
        self._populate_results(self._results_state)
        mode = str(self._results_state.get("mode", self.options().mode))
        suffix = UNRELAXED_ESTIMATE_SUFFIX if not self._estimate_note.isHidden() else ""
        self._status.setText(
            f"Calculated {self._MODE_LABELS.get(mode, mode)} nuclear relaxation.{suffix}"
        )

    def _restore_options(self, state: Mapping[str, object] | None) -> None:
        if not isinstance(state, Mapping):
            return
        mode = str(state.get("mode", "zf_polycrystal"))
        if mode.startswith("tf_"):
            self._regime_combo.setCurrentIndex(self._regime_combo.findData("tf"))
        else:
            self._regime_combo.setCurrentIndex(self._regime_combo.findData("zf"))
        self._sample_combo.setCurrentIndex(
            self._sample_combo.findData(
                "single_crystal" if mode.endswith("single_crystal") else "polycrystal"
            )
        )
        direction = float_triplet_or_none(state.get("direction"))
        if direction is not None:
            self._direction_x.setText(f"{direction[0]:.8g}")
            self._direction_y.setText(f"{direction[1]:.8g}")
            self._direction_z.setText(f"{direction[2]:.8g}")
        if state.get("radius") not in {None, ""}:
            self._radius.setText(str(state.get("radius")))
        if state.get("convergence_rtol") not in {None, ""}:
            self._rtol.setText(str(state.get("convergence_rtol")))
        saved_elements = state.get("elements")
        if isinstance(saved_elements, (list, tuple)):
            saved = tuple(str(value) for value in saved_elements)
            self._saved_elements = saved
            for element, check in self._element_checks.items():
                check.setChecked(element in saved)
        self._quadrupolar.setChecked(bool(state.get("include_quadrupolar", False)))
        self._distortion.setChecked(
            bool(state.get("use_distortion_correction", False)) and self._distortion_available
        )

    def _sync_mode_controls(self) -> None:
        single = str(self._sample_combo.currentData()) == "single_crystal"
        zf = str(self._regime_combo.currentData()) == "zf"
        for edit in (self._direction_x, self._direction_y, self._direction_z):
            edit.setEnabled(single)
        self._quadrupolar.setEnabled(zf)
        if not zf:
            self._quadrupolar.setChecked(False)

    def _populate_results(self, result: Mapping[str, object]) -> None:
        self._summary_table.setRowCount(0)
        self._element_table.setRowCount(0)
        self._estimate_note.setVisible(is_unrelaxed_site_source(result.get("site_source")))
        if not result:
            return
        rows = [
            (
                "Mode",
                self._MODE_LABELS.get(str(result.get("mode", "")), str(result.get("mode", ""))),
            ),
            ("σ (μs⁻¹)", _format_float(result.get("sigma"))),
            ("σ² (μs⁻²)", _format_float(result.get("sigma_squared"))),
            ("M₂ (μs⁻²)", _format_float(result.get("m2"))),
            ("Δ field (T)", _format_float(result.get("delta_field_t"))),
            ("Convergence radius (Å)", _format_float(result.get("convergence_radius_a"))),
            ("Nuclei summed", str(result.get("n_nuclei_summed", ""))),
            ("Elements", _nuclear_elements_text(result.get("elements"))),
        ]
        if result.get("sigma_perp_squared") is not None:
            rows.append(("σ⊥² (μs⁻²)", _format_float(result.get("sigma_perp_squared"))))
        if result.get("sigma_parallel_squared") is not None:
            rows.append(("σ∥² (μs⁻²)", _format_float(result.get("sigma_parallel_squared"))))
        correction = result.get("distortion_correction")
        if isinstance(correction, Mapping):
            rows.extend(
                [
                    (
                        "Correction Δσ²",
                        _format_float(correction.get("distortion_correction_sigma_squared")),
                    ),
                    ("Atoms unwrapped", str(correction.get("n_atoms_unwrapped", ""))),
                ]
            )
        for label, value in rows:
            row = self._summary_table.rowCount()
            self._summary_table.insertRow(row)
            self._summary_table.setItem(row, 0, QTableWidgetItem(label))
            self._summary_table.setItem(row, 1, QTableWidgetItem(value))
        per_element = result.get("per_element_sigma_squared")
        if isinstance(per_element, Mapping):
            for element, value in sorted(per_element.items()):
                row = self._element_table.rowCount()
                self._element_table.insertRow(row)
                self._element_table.setItem(row, 0, QTableWidgetItem(str(element)))
                self._element_table.setItem(row, 1, QTableWidgetItem(_format_float(value)))
        self._summary_table.resizeRowsToContents()
        self._element_table.resizeRowsToContents()


class DipolarTensorDialog(QDialog):
    """Configure magnetic-ion selection for a dipolar tensor calculation."""

    def __init__(
        self,
        structure: Structure,
        model: MagneticStructureModel,
        muon_site: MagneticMuonSiteOption | None,
        *,
        include_equivalent_sites: bool,
        selector_state: Mapping[str, object] | None = None,
        supercell: tuple[int, int, int] = MagneticFieldOptions().supercell,
        results_state: Sequence[object] = (),
        calculate_callback: Callable[["DipolarTensorDialog"], Sequence[object]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dipolar Tensor")
        self.resize(640, 620)
        self._structure = structure
        self._model = model
        self._muon_site = muon_site
        self._supercell = tuple(int(value) for value in supercell)
        self._calculate_callback = calculate_callback
        self._results_state: list[dict[str, object]] = [
            dict(result) for result in results_state if isinstance(result, Mapping)
        ]
        self._result_tables: list[QTableWidget] = []
        self._result_tensors: list[np.ndarray] = []
        self._assigned_indices = assigned_magnetic_atom_indices(structure, model)

        self._target_group = QButtonGroup(self)
        self._assigned_radio = QRadioButton("Atoms with assigned moments")
        self._all_atoms_radio = QRadioButton("All atoms")
        self._species_radio = QRadioButton("Element")
        self._equivalent_radio = QRadioButton("Equivalent site")
        self._indices_radio = QRadioButton("Atom indices")
        for button in (
            self._assigned_radio,
            self._all_atoms_radio,
            self._species_radio,
            self._equivalent_radio,
            self._indices_radio,
        ):
            self._target_group.addButton(button)
            button.toggled.connect(self._refresh_enabled_state)

        self._assigned_radio.setEnabled(bool(self._assigned_indices))
        self._species_combo = QComboBox()
        for species in sorted(set(str(symbol) for symbol in structure.species)):
            self._species_combo.addItem(species)
        self._species_combo.currentIndexChanged.connect(self._refresh_enabled_state)

        self._equivalent_combo = QComboBox()
        for selector in viable_atom_selectors(structure):
            if selector.kind == "symmetry_equivalent" and selector.values:
                self._equivalent_combo.addItem(selector_label(selector, structure), selector)
        self._equivalent_combo.currentIndexChanged.connect(self._refresh_enabled_state)

        self._indices_edit = QLineEdit()
        self._indices_edit.setPlaceholderText("1, 3, 6")
        self._indices_edit.setToolTip("Atom numbers counted from 1 in the host's atom order.")
        self._indices_edit.textChanged.connect(self._refresh_enabled_state)

        self._include_equivalent_sites = QCheckBox("Include equivalent muon sites")
        self._include_equivalent_sites.setChecked(bool(include_equivalent_sites))

        self._settings_summary = QLabel(
            f"Lattice sum: cells {self._supercell[0]} x {self._supercell[1]} x {self._supercell[2]}."
        )
        self._settings_summary.setWordWrap(True)
        style_hint_text(self._settings_summary)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        style_hint_text(self._status)

        self._result_unit_combo = QComboBox()
        for unit_key, label in CONTACT_COUPLING_UNIT_OPTIONS:
            self._result_unit_combo.addItem(label, unit_key)
        t_index = self._result_unit_combo.findData("T_per_muB")
        if t_index >= 0:
            self._result_unit_combo.setCurrentIndex(t_index)
        self._result_unit_combo.currentIndexChanged.connect(self._refresh_result_units)

        self._estimate_note = unrelaxed_results_note(self)
        self._result_scroll = QScrollArea(self)
        self._result_scroll.setWidgetResizable(True)
        self._result_content = QWidget(self._result_scroll)
        self._result_content_layout = QVBoxLayout(self._result_content)
        self._result_content_layout.setSpacing(14)
        self._result_scroll.setWidget(self._result_content)

        self._calculate_button = QPushButton("Calculate")
        self._calculate_button.clicked.connect(self._calculate)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._buttons.rejected.connect(self.reject)

        self._build_layout()
        self._restore_selector_state(selector_state)
        # What the dialog opened with, kept so closing it on invalid input
        # (an atom number below 1 or a non-number) does not lose the last
        # valid selection.
        self._opening_selector_state = self.selector_state()
        self._populate_results(self._results_state)
        self._refresh_enabled_state()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Dipolar Tensor")
        style_section_title(title)
        layout.addWidget(title)

        site_group = QGroupBox("Muon Site")
        site_layout = QVBoxLayout(site_group)
        site_summary = QLabel(_muon_site_summary(self._muon_site))
        site_summary.setWordWrap(True)
        style_hint_text(site_summary)
        site_layout.addWidget(site_summary)
        site_layout.addWidget(self._include_equivalent_sites)
        layout.addWidget(site_group)

        target_group = QGroupBox("Magnetic Ions")
        form = QFormLayout(target_group)
        assigned_text = (
            f"{plural(len(self._assigned_indices), 'atom')} targeted by enabled magnetic components"
            if self._assigned_indices
            else "No atoms currently have assigned moments"
        )
        assigned_row = QWidget(target_group)
        assigned_layout = QHBoxLayout(assigned_row)
        assigned_layout.setContentsMargins(0, 0, 0, 0)
        assigned_layout.addWidget(self._assigned_radio)
        assigned_layout.addWidget(QLabel(assigned_text), 1)
        form.addRow(assigned_row)
        form.addRow(self._all_atoms_radio)
        form.addRow(self._species_radio, self._species_combo)
        form.addRow(self._equivalent_radio, self._equivalent_combo)
        form.addRow(self._indices_radio, self._indices_edit)
        layout.addWidget(target_group)

        settings_row = QHBoxLayout()
        settings_row.addWidget(self._settings_summary, 1)
        layout.addLayout(settings_row)
        layout.addWidget(self._status)
        action_row = QHBoxLayout()
        action_row.addWidget(self._calculate_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        result_group = QGroupBox("Most Recent Result")
        result_layout = QVBoxLayout(result_group)
        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Units"))
        unit_row.addWidget(self._result_unit_combo)
        unit_row.addStretch(1)
        result_layout.addLayout(unit_row)
        result_layout.addWidget(self._estimate_note)
        result_layout.addWidget(self._result_scroll, 1)
        layout.addWidget(result_group, 1)
        layout.addWidget(self._buttons)

    def selector_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "include_equivalent_sites": self._include_equivalent_sites.isChecked(),
        }
        if self._assigned_radio.isChecked():
            state["mode"] = "assigned"
            return state
        try:
            selector = self.selector()
        except ValueError:
            opening = getattr(self, "_opening_selector_state", {"mode": "assigned"})
            return {**opening, **state}
        state["mode"] = "selector"
        state["selector"] = selector.to_json()
        return state

    def results_state(self) -> list[dict[str, object]]:
        return [dict(result) for result in self._results_state]

    def include_equivalent_sites(self) -> bool:
        return self._include_equivalent_sites.isChecked()

    def selector(self) -> AtomSelector:
        if self._species_radio.isChecked():
            return AtomSelector.species(self._species_combo.currentText())
        if self._equivalent_radio.isChecked():
            selector = self._equivalent_combo.currentData()
            if isinstance(selector, AtomSelector):
                return selector
        if self._indices_radio.isChecked():
            return AtomSelector.indices(self._parse_indices())
        if self._all_atoms_radio.isChecked():
            return AtomSelector.all()
        return AtomSelector.indices(self._assigned_indices)

    def selected_ion_indices(self) -> tuple[int, ...]:
        if self._assigned_radio.isChecked():
            return self._assigned_indices
        if not any(
            button.isChecked()
            for button in (
                self._all_atoms_radio,
                self._species_radio,
                self._equivalent_radio,
                self._indices_radio,
            )
        ):
            return ()
        return resolve_atom_selector(self._structure, self.selector())

    def _calculate(self) -> None:
        if self._calculate_callback is None:
            return
        try:
            results = tuple(self._calculate_callback(self))
        except MagneticFieldDependencyError as exc:
            self._status.setText(f"Result unavailable: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - muesr and muLFC raise untyped errors
            self._status.setText(f"Calculation failed: {exc}")
            return
        self._results_state = [self._result_to_state(result) for result in results]
        if self._muon_site is not None:
            for state in self._results_state:
                state["site_source"] = self._muon_site.source
        self._populate_results(self._results_state)
        count = len(self._results_state)
        suffix = UNRELAXED_ESTIMATE_SUFFIX if not self._estimate_note.isHidden() else ""
        self._status.setText(f"Calculated {plural(count, 'dipolar tensor')}.{suffix}")

    def _restore_selector_state(self, state: Mapping[str, object] | None) -> None:
        if isinstance(state, Mapping):
            self._include_equivalent_sites.setChecked(
                bool(
                    state.get(
                        "include_equivalent_sites", self._include_equivalent_sites.isChecked()
                    )
                )
            )
            if str(state.get("mode", "")).strip() == "assigned" and self._assigned_indices:
                self._assigned_radio.setChecked(True)
                return
            if "selector" in state:
                selector = AtomSelector.from_json(state.get("selector"))
                if self._set_selector(selector):
                    return
        if self._assigned_indices:
            self._assigned_radio.setChecked(True)

    def _set_selector(self, selector: AtomSelector) -> bool:
        if selector.kind == "species" and selector.values:
            index = self._species_combo.findText(str(selector.values[0]))
            if index >= 0:
                self._species_combo.setCurrentIndex(index)
                self._species_radio.setChecked(True)
                return True
        if selector.kind == "symmetry_equivalent":
            for index in range(self._equivalent_combo.count()):
                if self._equivalent_combo.itemData(index) == selector:
                    self._equivalent_combo.setCurrentIndex(index)
                    self._equivalent_radio.setChecked(True)
                    return True
        if selector.kind == "indices":
            self._indices_edit.setText(", ".join(str(int(value) + 1) for value in selector.values))
            self._indices_radio.setChecked(True)
            return True
        if selector.kind == "all":
            self._all_atoms_radio.setChecked(True)
            return True
        return False

    def _refresh_enabled_state(self, *_args: object) -> None:
        self._species_combo.setEnabled(self._species_radio.isChecked())
        self._equivalent_combo.setEnabled(self._equivalent_radio.isChecked())
        self._indices_edit.setEnabled(self._indices_radio.isChecked())
        try:
            indices = self.selected_ion_indices()
        except ValueError:
            indices = ()
        has_site = self._muon_site is not None
        self._calculate_button.setEnabled(has_site and bool(indices))
        if not has_site:
            self._status.setText("Select a muon site before calculating a tensor.")
        elif not indices:
            self._status.setText("Choose at least one magnetic ion.")
        else:
            self._status.setText(
                f"Ready to calculate tensor using {plural(len(indices), 'magnetic ion')}."
            )

    def _parse_indices(self) -> tuple[int, ...]:
        """Return the zero-based atom indices for the one-based numbers typed in the box.

        The box counts from 1, like the magnetic-structure builder and the
        ``Equivalent site`` labels; the selector stores zero-based indices.
        """

        raw = self._indices_edit.text().replace(",", " ").split()
        numbers = tuple(int(value) for value in raw)
        if any(number < 1 for number in numbers):
            raise ValueError("Atom indices count from 1.")
        return tuple(number - 1 for number in numbers)

    def _populate_results(self, results: Sequence[Mapping[str, object]]) -> None:
        while self._result_content_layout.count():
            item = self._result_content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._result_tables = []
        self._result_tensors = []
        self._estimate_note.setVisible(
            any(is_unrelaxed_site_source(result.get("site_source")) for result in results)
        )
        if not results:
            empty = QLabel("No tensor calculated yet.")
            style_hint_text(empty)
            self._result_content_layout.addWidget(empty)
            self._result_content_layout.addStretch(1)
            return
        for result in results:
            self._result_content_layout.addWidget(self._result_card(result))
        self._result_content_layout.addStretch(1)

    def _result_card(self, result: Mapping[str, object]) -> QWidget:
        title = str(result.get("site_label", "site"))
        if is_unrelaxed_site_source(result.get("site_source")):
            title = f"{title} (unrelaxed estimate)"
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        position = result.get("muon_position")
        if position is not None:
            try:
                x, y, z = float_triplet(position)
                summary = QLabel(f"Muon position: ({x:.6g}, {y:.6g}, {z:.6g})")
                style_hint_text(summary)
                layout.addWidget(summary)
            except (TypeError, ValueError):
                pass
        table = QTableWidget(3, 3, group)
        track_item_view_corners(table)
        table.setHorizontalHeaderLabels(["x", "y", "z"])
        table.setVerticalHeaderLabels(["x", "y", "z"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setMinimumHeight(136)
        tensor = _dipolar_result_tensor(result)
        self._result_tables.append(table)
        self._result_tensors.append(tensor)
        self._set_result_table_tensor(table, self._display_result_tensor(tensor))
        layout.addWidget(table)
        return group

    def _refresh_result_units(self) -> None:
        for table, tensor in zip(self._result_tables, self._result_tensors, strict=True):
            self._set_result_table_tensor(table, self._display_result_tensor(tensor))

    def _display_result_tensor(self, tensor_T_per_muB: np.ndarray) -> np.ndarray:
        return np.asarray(tensor_T_per_muB, dtype=float) * self._result_display_factor()

    def _result_display_factor(self) -> float:
        unit = str(self._result_unit_combo.currentData() or "T_per_muB")
        if unit == "MHz_per_muB":
            return MUON_GAMMA_OVER_2PI_MHZ_T
        if unit in {"angstrom^-3", "a^-3", "ang^-3"}:
            return 1.0 / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB
        return 1.0

    @staticmethod
    def _set_result_table_tensor(table: QTableWidget, tensor: np.ndarray) -> None:
        for row in range(3):
            for column in range(3):
                item = table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    table.setItem(row, column, item)
                item.setText(f"{float(tensor[row, column]):.8g}")

    @staticmethod
    def _result_to_state(result: object) -> dict[str, object]:
        if hasattr(result, "to_json"):
            payload = result.to_json()
        elif isinstance(result, Mapping):
            payload = dict(result)
        else:
            payload = {
                "site_label": str(getattr(result, "site_label", "site")),
                "muon_position": list(getattr(result, "muon_position", (0.0, 0.0, 0.0))),
                "dipolar_tensor_T_per_muB": np.asarray(
                    getattr(result, "tensor_T_per_muB", np.zeros((3, 3))),
                    dtype=float,
                ).tolist(),
            }
        jsonable = to_jsonable(payload)
        return dict(jsonable) if isinstance(jsonable, Mapping) else {}


def _dipolar_result_tensor(result: Mapping[str, object]) -> np.ndarray:
    """Return a stored dipolar result's tensor in T per mu_B, or zeros when it is not 3 x 3."""

    arr = np.asarray(result.get("dipolar_tensor_T_per_muB"), dtype=float)
    if arr.shape != (3, 3):
        arr = np.zeros((3, 3), dtype=float)
    return arr


def _nuclear_elements_text(elements: object) -> str:
    """Name the elements a nuclear sum included, or ``All`` when it was not filtered."""

    if isinstance(elements, (list, tuple)):
        return ", ".join(str(element) for element in elements)
    return "All"


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return ""


def _muon_site_summary(site: MagneticMuonSiteOption | None) -> str:
    """The muon-site line both dialogs show above their settings."""

    if site is None:
        return "Muon site: no site selected."
    x, y, z = site.fractional_position
    return f"Muon site: {site.label} at ({x:.4f}, {y:.4f}, {z:.4f})."
