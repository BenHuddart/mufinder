"""Hyperfine Coupling results: the stored payload and its hand-off to Magnetic Fields.

A completed magres job stores the muon's coupling in the physical sign
convention (see :mod:`mufinder.science.hyperfine`), the pseudo spin density at
the muon when the job's ``.den_fmt`` could be read, and the inputs of the
Magnetic Fields contact shell: the relaxed cell, the muon and host positions
and the host Mulliken spin moments of the source muon relaxation.

The hand-off fills the ``Hyperfine coupling`` of every Magnetic Fields form
(the main page and each saved workspace) whose selected muon site is the
result's site, in Å^-3 per μB, computed with that form's own contact shell.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from mufinder.castep_io.magres import HyperfineTensor
from mufinder.castep_io.output_parser import parse_castep_mulliken_atomic_spins
from mufinder.domain.models import Result, Structure
from mufinder.science.hyperfine import (
    HyperfineError,
    angstrom3_to_bohr3,
    contact_coupling_from_moments,
    contact_field_from_spin_density,
    muon_hyperfine_from_castep,
)
from mufinder.science.lattice import lattice_frac_to_cart
from mufinder.science.magnetic_fields import (
    CONTACT_COUPLING_UNIT_OPTIONS,
    MagneticFieldError,
    MagneticFieldOptions,
)
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    MAGNETIC_FIELD_RESULT_TYPE,
    magnetic_field_result,
)
from mufinder.services.project_service import ProjectDocument
from mufinder.workflow.hyperfine import HYPERFINE_CALC_TYPE

MUON_SPECIES = "H:mu"

#: ``ui_state`` keys of the Magnetic Fields main form and saved workspaces.
MAGNETIC_FIELD_FORM_KEY = "magnetic_field_form"
MAGNETIC_FIELD_WORKSPACES_KEY = "magnetic_field_workspaces"

#: The Magnetic Fields contact-shell defaults a form without stored values uses.
DEFAULT_CONTACT_NNN = MagneticFieldOptions().nnn
DEFAULT_CONTACT_RCONT = MagneticFieldOptions().rcont

#: The Magnetic Fields unit key the hand-off writes the coupling in.
HANDOFF_COUPLING_UNIT = "angstrom^-3"

NO_MULLIKEN_SPINS_WARNING = (
    "The source relaxation has no Mulliken spins; the coupling cannot be handed to Magnetic Fields."
)


def mulliken_spins_in_structure_order(
    structure: Structure, castep_path: Path | None
) -> tuple[float, ...] | None:
    """Return the Mulliken spin moment of each atom of ``structure``, in its order.

    CASTEP lists Mulliken populations species by species, in the order the
    species first appear, and by ion within a species; this undoes that
    grouping. ``None`` when the output is missing, has no spin populations, or
    lists a different number of atoms.
    """

    if castep_path is None or not castep_path.exists():
        return None
    try:
        spins = parse_castep_mulliken_atomic_spins(castep_path)
    except OSError:
        return None
    if spins is None or len(spins) != len(structure.species):
        return None
    species_order = list(dict.fromkeys(structure.species))
    castep_order = [
        index
        for species in species_order
        for index, atom_species in enumerate(structure.species)
        if atom_species == species
    ]
    ordered = [0.0] * len(spins)
    for spin, index in zip(spins, castep_order, strict=True):
        ordered[index] = float(spin)
    return tuple(ordered)


def contact_shell_inputs(
    structure: Structure, spins: Sequence[float] | None
) -> dict[str, Any] | None:
    """Return the stored contact-shell inputs of a relaxed muon structure.

    The cell, the muon's and the host ions' Cartesian positions (Å), and the
    host spin moments (μB). ``None`` without spins or without exactly one muon.
    """

    if spins is None:
        return None
    muon_indices = [
        index for index, species in enumerate(structure.species) if species == MUON_SPECIES
    ]
    if len(muon_indices) != 1:
        return None
    lattice = [[float(value) for value in row] for row in structure.lattice_vectors]
    cartesian = [
        [float(value) for value in lattice_frac_to_cart(position, lattice)]
        for position in structure.fractional_positions
    ]
    hosts = [index for index in range(len(structure.species)) if index != muon_indices[0]]
    return {
        "lattice_vectors_ang": lattice,
        "muon_cartesian_ang": cartesian[muon_indices[0]],
        "host_cartesian_ang": [cartesian[index] for index in hosts],
        "host_moments_muB": [float(spins[index]) for index in hosts],
    }


def hyperfine_result_payload(
    tensor: HyperfineTensor,
    *,
    pseudo_spin_density_angstrom3: float | None,
    shell: Mapping[str, Any] | None,
    source: Mapping[str, str | None],
) -> dict[str, Any]:
    """Return the stored payload of a completed hyperfine job.

    ``source`` names the relaxed site the job ran on (``source_site_id``,
    ``source_relaxation_job_id``, ``relaxed_structure_id``, ``charge_state``).
    """

    muon = muon_hyperfine_from_castep(tensor)
    pseudo_field = (
        None
        if pseudo_spin_density_angstrom3 is None
        else contact_field_from_spin_density(pseudo_spin_density_angstrom3)
    )
    return {
        "tensor_mhz": [list(row) for row in muon.tensor_mhz],
        "isotropic_mhz": muon.isotropic_mhz,
        "dipolar_principal_mhz": list(muon.dipolar_principal_mhz),
        "contact_field_T": muon.contact_field_t,
        "spin_density_bohr3": angstrom3_to_bohr3(muon.spin_density_angstrom3),
        "pseudo_spin_density_bohr3": (
            None
            if pseudo_spin_density_angstrom3 is None
            else angstrom3_to_bohr3(pseudo_spin_density_angstrom3)
        ),
        "pseudo_contact_field_T": pseudo_field,
        "contact_shell": dict(shell) if shell is not None else None,
        **{key: value for key, value in source.items()},
    }


def hyperfine_results(document: ProjectDocument | None) -> list[Result]:
    """Return the project's hyperfine results in storage order."""

    if document is None:
        return []
    return [
        result for result in document.results.values() if result.result_type == HYPERFINE_CALC_TYPE
    ]


def handoff_unavailable_reason(result: Result | None) -> str:
    """Return why ``result`` cannot be handed to Magnetic Fields, or ``""`` when it can."""

    if result is None or result.result_type != HYPERFINE_CALC_TYPE:
        return "Select one hyperfine result."
    data = result.data if isinstance(result.data, Mapping) else {}
    if not isinstance(data.get("contact_shell"), Mapping):
        return NO_MULLIKEN_SPINS_WARNING
    return ""


@dataclass(frozen=True, slots=True)
class HandoffTarget:
    """One Magnetic Fields form the hand-off would fill.

    ``workspace_id`` is ``None`` for the main page. ``coupling`` is the new
    value in Å^-3 per μB, or ``None`` with ``error`` set when the form's contact
    shell cannot give one.
    """

    title: str
    workspace_id: str | None
    existing_text: str
    existing_unit: str
    coupling: float | None
    error: str = ""

    @property
    def new_text(self) -> str:
        return "" if self.coupling is None else format_coupling(self.coupling)

    @property
    def existing_display(self) -> str:
        """The form's current coupling with its unit label, as the form shows it."""

        label = dict(CONTACT_COUPLING_UNIT_OPTIONS).get(self.existing_unit, self.existing_unit)
        return f"{self.existing_text.strip()} {label}".strip()

    @property
    def overwrites(self) -> bool:
        """Whether the form holds a different nonzero coupling the hand-off would replace.

        An empty or zero coupling, the form's default for no contact term, is
        filled without asking.
        """

        if self.coupling is None or not self.existing_text.strip():
            return False
        try:
            if float(self.existing_text) == 0.0:
                return False
        except ValueError:
            pass
        return (
            self.existing_text.strip() != self.new_text
            or self.existing_unit != HANDOFF_COUPLING_UNIT
        )


def format_coupling(value: float) -> str:
    """Format a coupling in Å^-3 per μB the way the hand-off writes it."""

    return f"{float(value):.6g}"


def plan_magnetic_field_handoff(document: ProjectDocument, result: Result) -> list[HandoffTarget]:
    """Return every Magnetic Fields form whose selected muon site is ``result``'s site.

    Raises :class:`HyperfineError` when ``result`` cannot be handed over (see
    :func:`handoff_unavailable_reason`).
    """

    reason = handoff_unavailable_reason(result)
    if reason:
        raise HyperfineError(reason)
    data = result.data
    shell = data["contact_shell"]
    spin_density_angstrom3 = float(data["spin_density_bohr3"]) / angstrom3_to_bohr3(1.0)
    targets: list[HandoffTarget] = []
    for title, workspace_id, form in _magnetic_field_forms(document):
        if not _form_shows_site(form, data):
            continue
        nnn = _int_or(form.get("contact_nnn"), DEFAULT_CONTACT_NNN)
        rcont = _float_or(form.get("contact_rcont"), DEFAULT_CONTACT_RCONT)
        coupling: float | None = None
        error = ""
        try:
            coupling = contact_coupling_from_moments(
                spin_density_angstrom3,
                muon_cartesian=shell["muon_cartesian_ang"],
                ion_cartesian=shell["host_cartesian_ang"],
                ion_moments_mub=shell["host_moments_muB"],
                lattice_vectors=shell["lattice_vectors_ang"],
                nnn=nnn,
                rcont=rcont,
            )
        except (HyperfineError, MagneticFieldError, KeyError, TypeError, ValueError) as exc:
            error = str(exc)
        targets.append(
            HandoffTarget(
                title=title,
                workspace_id=workspace_id,
                existing_text=str(form.get("hyperfine_coupling", "") or ""),
                existing_unit=str(form.get("hyperfine_coupling_unit", "") or ""),
                coupling=coupling,
                error=error,
            )
        )
    return targets


def apply_magnetic_field_handoff(
    document: ProjectDocument, targets: Sequence[HandoffTarget]
) -> int:
    """Write each target's coupling into its Magnetic Fields form; return how many changed.

    Targets with an error are skipped. A form whose coupling changes loses its
    stored field results, as when the coupling is edited on the page: they were
    computed with the old contact term. Its dipolar tensors and nuclear results
    do not depend on the coupling and are kept.
    """

    by_workspace = {
        target.workspace_id: target for target in targets if target.coupling is not None
    }
    changed = 0
    for _title, workspace_id, form in _magnetic_field_forms(document):
        target = by_workspace.get(workspace_id)
        if target is None:
            continue
        unchanged = (
            str(form.get("hyperfine_coupling", "") or "").strip() == target.new_text
            and form.get("hyperfine_coupling_unit") == HANDOFF_COUPLING_UNIT
        )
        form["hyperfine_coupling"] = target.new_text
        form["hyperfine_coupling_unit"] = HANDOFF_COUPLING_UNIT
        if not unchanged:
            owner = MAGNETIC_FIELD_PAGE_OWNER if workspace_id is None else workspace_id
            stale = magnetic_field_result(document, owner, MAGNETIC_FIELD_RESULT_TYPE)
            if stale is not None:
                del document.results[stale.id]
        changed += 1
    return changed


def _magnetic_field_forms(
    document: ProjectDocument,
) -> list[tuple[str, str | None, dict[str, Any]]]:
    """Return ``(title, workspace id, mutable form)`` for the main page and each workspace."""

    forms: list[tuple[str, str | None, dict[str, Any]]] = []
    main = document.ui_state.get(MAGNETIC_FIELD_FORM_KEY)
    if isinstance(main, dict):
        forms.append(("Magnetic Fields", None, main))
    workspaces = document.ui_state.get(MAGNETIC_FIELD_WORKSPACES_KEY)
    for workspace in workspaces if isinstance(workspaces, list) else []:
        if not isinstance(workspace, dict):
            continue
        form = workspace.get("form")
        workspace_id = str(workspace.get("id", "")).strip()
        if not isinstance(form, dict) or not workspace_id:
            continue
        title = str(workspace.get("title", "Magnetic Fields")).strip() or "Magnetic Fields"
        forms.append((title, workspace_id, form))
    return forms


def _form_shows_site(form: Mapping[str, Any], data: Mapping[str, Any]) -> bool:
    """Return whether a form's selected muon site is the relaxed site ``data`` names.

    A Magnetic Fields site picked from a relaxation analysis carries its
    cluster's representative relaxation job and the relaxed structure; either
    matching the hyperfine job's source relaxation identifies the site.
    """

    site = form.get("selected_muon_site")
    if not isinstance(site, Mapping):
        return False
    source_job = data.get("source_relaxation_job_id")
    structure_id = data.get("relaxed_structure_id")
    if structure_id and _same_id(site.get("distortion_structure_id"), structure_id):
        return True
    properties = site.get("properties")
    cluster = properties.get("cluster") if isinstance(properties, Mapping) else None
    if isinstance(cluster, Mapping) and source_job:
        return _same_id(cluster.get("representative_job_id"), source_job)
    return False


def _same_id(left: object, right: object) -> bool:
    try:
        return left is not None and right is not None and UUID(str(left)) == UUID(str(right))
    except ValueError:
        return False


def _int_or(value: object, default: int) -> int:
    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return default


def _float_or(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


__all__ = [
    "HANDOFF_COUPLING_UNIT",
    "HandoffTarget",
    "NO_MULLIKEN_SPINS_WARNING",
    "apply_magnetic_field_handoff",
    "contact_shell_inputs",
    "format_coupling",
    "handoff_unavailable_reason",
    "hyperfine_result_payload",
    "hyperfine_results",
    "mulliken_spins_in_structure_order",
    "plan_magnetic_field_handoff",
]
