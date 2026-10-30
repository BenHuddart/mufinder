"""Candidate muon-site generation, clustering, removal and staging.

The numerical placement lives in :mod:`mufinder.science.site_generation`. This
module turns a generation request from the muon-generation page into
:class:`~mufinder.domain.models.MuonSite` records on a project document: it
reads the active host, dispatches on the generation method, labels the new
sites and describes the change as provenance.

Every entry point changes the in-memory document only. They return the
provenance records describing the change in a :class:`SiteGenerationOutcome`
and leave appending them to the caller, so
:class:`~mufinder.services.project_service.ProjectService` keeps the one path
that writes the provenance log.

Random and element-attached generation can take long enough to run on a
worker thread. They split into three steps so the worker never touches the
document: :func:`prepare_candidate_sites` copies what the generators read from
the document, :func:`compute_candidate_sites` computes the sites from that copy
alone (and can be cancelled), and :func:`apply_candidate_sites` adds them to the
document once the worker has finished.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import copy
from dataclasses import dataclass, field
import re
from typing import Any
from uuid import UUID

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import MuonSite, ProvenanceRecord
from mufinder.domain.text import plural
from mufinder.science.site_generation import (
    SiteGenerationError,
    closest_symmetry_equivalent_position,
    estimate_muon_muon_separation_for_count,
    estimate_random_site_count_from_separation,
    generate_attached_element_sites,
    generate_random_sites,
    lattice_cart_to_frac,
    lattice_frac_to_cart,
    symmetry_distinct_atom_indexes,
)
from mufinder.science.spglib_compat import SYMMETRY_TOLERANCE_ANG, symmetry_detection_warning
from mufinder.services.project_document import ProjectDocument, active_host_structure

#: Status of a freshly generated muon site.
GENERATED_SITE_STATUS = "generated"
#: Statuses treated as "generated": ``candidate`` is the legacy name, see the
#: compatibility ledger.
LEGACY_GENERATED_SITE_STATUSES = {GENERATED_SITE_STATUS, "candidate"}
_STAGED_SITE_STATUS = "staged"
_STAGEABLE_SITE_STATUSES = {*LEGACY_GENERATED_SITE_STATUSES, _STAGED_SITE_STATUS}

_IDENTITY_LATTICE = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
_UEP_CLUSTER_METHODS = {"uep_cluster", "uep_cluster_member"}
_RANDOM_COUNT_RETRIES = 6
_RANDOM_COUNT_SEPARATION_SHRINK = 0.9

ProgressCallback = Callable[[int, int], None]

#: Generation methods that can run on a worker thread through
#: :func:`prepare_candidate_sites` and :func:`compute_candidate_sites`.
WORKER_GENERATION_METHODS = frozenset({"random", "near_element"})


class SiteGenerationCancelled(Exception):
    """A worker generation was cancelled before it finished.

    It is not a :class:`SiteGenerationError`, so the count-mode retries do not
    swallow it.
    """


@dataclass(slots=True)
class SiteGenerationOutcome:
    """Sites a generation, clustering, removal or staging step touched.

    ``provenance`` holds the records describing the change, in order; the
    caller appends them to the project's provenance log.
    """

    sites: list[MuonSite]
    provenance: list[ProvenanceRecord] = field(default_factory=list)


@dataclass(slots=True)
class _HostContext:
    """The active host as the generators read it."""

    structure_id: UUID | None
    lattice: list[list[float]] | None
    species: list[str]
    positions: list[list[float]]

    @classmethod
    def for_document(cls, document: ProjectDocument) -> _HostContext:
        """Copy the active host's id and geometry, so a worker never reads the document."""

        host = active_host_structure(document)
        return cls(
            structure_id=host.id if host else None,
            lattice=(
                [list(row) for row in host.lattice_vectors]
                if host and host.lattice_vectors
                else None
            ),
            species=list(host.species) if host else [],
            positions=[list(position) for position in host.fractional_positions] if host else [],
        )


@dataclass(frozen=True, slots=True)
class CandidateSiteRequest:
    """A worker generation request, read from the document on the GUI thread.

    It holds copies of everything the generators read: the active host and the
    labels of the existing sites, which number the new ones.
    """

    method: str
    parameters: dict[str, Any]
    host: _HostContext
    site_labels: tuple[str, ...]

    @property
    def host_structure_id(self) -> UUID | None:
        """The active host when the request was prepared."""

        return self.host.structure_id


@dataclass(slots=True)
class _GeneratedBatch:
    """The sites one generation method produced and the attempts it took."""

    sites: list[MuonSite]
    attempts_used: int = 1


@dataclass(slots=True)
class _RandomDraw:
    """Random points plus the separations that produced them."""

    points: list[Any]
    attempts_used: int
    min_muon_muon: float
    estimated_min_muon_muon: float | None = None


def generate_candidate_sites(
    document: ProjectDocument,
    *,
    method: str,
    parameters: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> SiteGenerationOutcome:
    """Create generated muon sites from a generation request.

    ``method`` is ``manual``, ``random``, ``near_element`` or ``uep_clusters``.
    Random strategies enforce periodic muon-muon and muon-atom distance
    constraints. Element-attached generation ranks alternative starts for
    each selected host atom by local clearance from the surrounding host.

    The new sites are added to ``document.sites``. The outcome's provenance
    holds a ``generate_candidates`` record, followed by a
    ``generation_partial`` record when fewer sites were generated than
    requested and a ``symmetry_unavailable`` record when the method uses the
    host's symmetry but it could not be detected, so the host was treated as P1.
    """

    if method in WORKER_GENERATION_METHODS:
        request = prepare_candidate_sites(document, method=method, parameters=parameters)
        outcome = compute_candidate_sites(request, progress_callback=progress_callback)
    else:
        host = _HostContext.for_document(document)
        requested_count = _requested_site_count(method, parameters, host)
        if method == "manual":
            batch = _generate_manual_site(document, host, parameters)
        elif method == "uep_clusters":
            batch = _generate_uep_cluster_sites(document, host, parameters)
        else:
            raise SiteGenerationError(f"Unknown generation method '{method}'.")
        outcome = _generation_outcome(method, parameters, host, batch, requested_count)
    for site in outcome.sites:
        document.sites[site.id] = site
    return outcome


def prepare_candidate_sites(
    document: ProjectDocument, *, method: str, parameters: dict[str, Any]
) -> CandidateSiteRequest:
    """Capture a random or element-attached request for :func:`compute_candidate_sites`.

    Runs on the thread that owns ``document``; the request it returns is safe
    to compute on another thread.
    """

    if method not in WORKER_GENERATION_METHODS:
        raise SiteGenerationError(f"'{method}' generation cannot run on a worker thread.")
    return CandidateSiteRequest(
        method=method,
        parameters=copy.deepcopy(dict(parameters)),
        host=_HostContext.for_document(document),
        site_labels=tuple(_site_labels(document)),
    )


def compute_candidate_sites(
    request: CandidateSiteRequest,
    *,
    progress_callback: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> SiteGenerationOutcome:
    """Compute the sites ``request`` asks for, without reading or changing any document.

    ``should_cancel`` is polled at every progress report and once more at the
    end; when it returns ``True`` the computation stops with
    :class:`SiteGenerationCancelled` and its sites are never returned.
    """

    report = _cancellable_progress(progress_callback, should_cancel)
    host = request.host
    parameters = request.parameters
    requested_count = _requested_site_count(request.method, parameters, host)
    if request.method == "random":
        batch = _generate_random_sites(
            request.site_labels, host, parameters, requested_count, report
        )
    else:
        batch = _generate_attached_element_sites(request.site_labels, host, parameters, report)
    if should_cancel is not None and should_cancel():
        raise SiteGenerationCancelled
    return _generation_outcome(request.method, parameters, host, batch, requested_count)


def apply_candidate_sites(
    document: ProjectDocument, request: CandidateSiteRequest, outcome: SiteGenerationOutcome
) -> None:
    """Add the sites :func:`compute_candidate_sites` returned for ``request`` to ``document``.

    Raises :class:`SiteGenerationError`, and adds nothing, when the active host
    is no longer the one the sites were generated for: another structure, or
    the same one with a different cell or atoms. A site whose label another
    site took while the request was computed is renumbered after the labels
    now in use.
    """

    if _HostContext.for_document(document) != request.host:
        raise SiteGenerationError(
            "The active host structure changed while the sites were being generated, "
            "so they were not added. Generate them again for the new host."
        )
    _renumber_taken_labels(document, outcome.sites)
    for site in outcome.sites:
        document.sites[site.id] = site


def _renumber_taken_labels(document: ProjectDocument, sites: Sequence[MuonSite]) -> None:
    """Give each of ``sites`` whose label is already in ``document`` the next free number."""

    used = _site_labels(document)
    taken = set(used)
    for site in sites:
        if site.label not in taken:
            taken.add(site.label)
            used.append(site.label)
            continue
        match = _NUMBERED_LABEL.match(site.label)
        prefix = match.group(1) if match is not None else site.label
        site.label = f"{prefix} {_next_site_label_number(used, prefix)}"
        taken.add(site.label)
        used.append(site.label)


def _cancellable_progress(
    progress_callback: ProgressCallback | None, should_cancel: Callable[[], bool] | None
) -> ProgressCallback | None:
    """Wrap ``progress_callback`` so a progress report stops a cancelled computation."""

    if should_cancel is None:
        return progress_callback

    def report(generated: int, expected: int) -> None:
        if should_cancel():
            raise SiteGenerationCancelled
        if progress_callback is not None:
            progress_callback(generated, expected)

    return report


def _generation_outcome(
    method: str,
    parameters: dict[str, Any],
    host: _HostContext,
    batch: _GeneratedBatch,
    requested_count: int | None,
) -> SiteGenerationOutcome:
    """Describe a generated batch: its sites, and the provenance of the request."""

    provenance = _generation_provenance(method, parameters, batch, requested_count)
    warning = _symmetry_warning(method, parameters, host)
    if warning is not None:
        provenance.append(
            ProvenanceRecord(
                action="symmetry_unavailable",
                workflow_stage=WorkflowStage.MUON_GENERATION,
                actor="system",
                description=warning,
                parameters={"method": method, "symprec_ang": SYMMETRY_TOLERANCE_ANG},
                output_entity_ids=[site.id for site in batch.sites],
            )
        )
    return SiteGenerationOutcome(sites=batch.sites, provenance=provenance)


def requested_site_count(
    document: ProjectDocument, *, method: str, parameters: dict[str, Any]
) -> int | None:
    """Return how many sites a generation request asks for, or ``None`` when it sets no target.

    Element-attached generation asks for ``muons_per_atom`` starts on one atom of
    each symmetry-equivalent set of the element.
    """

    return _requested_site_count(method, parameters, _HostContext.for_document(document))


def _symmetry_warning(method: str, parameters: dict[str, Any], host: _HostContext) -> str | None:
    """Return why the host's symmetry was not detected, when ``method`` uses it."""

    uses_symmetry = method == "near_element" or (
        method == "random" and bool(parameters.get("include_symmetry_equivalents", True))
    )
    if not uses_symmetry or host.lattice is None or not host.species:
        return None
    return symmetry_detection_warning(
        lattice_vectors=host.lattice,
        species=host.species,
        fractional_positions=host.positions,
    )


def cluster_candidate_positions(
    document: ProjectDocument,
    site_ids: Sequence[UUID],
    *,
    anchor_fractional: Sequence[float],
) -> SiteGenerationOutcome:
    """Move selected generated muon sites to symmetry representatives near an anchor.

    Sites that are missing or no longer generated are skipped. The outcome's
    provenance holds one ``cluster_candidate_positions`` record.
    """

    host = active_host_structure(document)
    if host is None:
        raise SiteGenerationError(
            "Select a host structure before clustering generated muon positions."
        )
    if not site_ids:
        raise SiteGenerationError("Select one or more generated muon positions to cluster.")
    if len(anchor_fractional) != 3:
        raise SiteGenerationError("Enter a three-component fractional anchor position.")

    lattice = host.lattice_vectors
    anchor = [float(value) % 1.0 for value in anchor_fractional]
    updated: list[MuonSite] = []
    for site_id in site_ids:
        site = document.sites.get(site_id)
        if site is None or site.status not in LEGACY_GENERATED_SITE_STATUSES:
            continue
        if len(site.initial_fractional_position) != 3:
            raise SiteGenerationError(
                f"Muon site '{site.label}' is missing a valid fractional position."
            )
        clustered = closest_symmetry_equivalent_position(
            site.initial_fractional_position,
            anchor_frac=anchor,
            lattice_vectors=lattice,
            host_species=host.species,
            host_positions_frac=host.fractional_positions,
        )
        site.initial_fractional_position = clustered
        site.initial_cartesian_position = lattice_frac_to_cart(clustered, lattice)
        updated.append(site)

    if not updated:
        raise SiteGenerationError("Selected rows did not contain any generated muon positions.")

    record = ProvenanceRecord(
        action="cluster_candidate_positions",
        workflow_stage=WorkflowStage.MUON_GENERATION,
        actor="user",
        description=f"Clustered {plural(len(updated), 'generated muon site')} near an anchor position",
        parameters={
            "anchor_fractional": anchor,
            "site_labels": [site.label for site in updated],
        },
        input_entity_ids=[site.id for site in updated],
        output_entity_ids=[site.id for site in updated],
    )
    return SiteGenerationOutcome(sites=updated, provenance=[record])


def remove_candidate_sites(
    document: ProjectDocument, site_ids: Sequence[UUID]
) -> SiteGenerationOutcome:
    """Remove generated muon sites by id; ids of other sites are ignored.

    The outcome lists the removed sites, with a ``delete_candidates`` record
    when any were removed.
    """

    removed: list[MuonSite] = []
    for site_id in site_ids:
        site = document.sites.get(site_id)
        if site is None or site.status not in LEGACY_GENERATED_SITE_STATUSES:
            continue
        removed.append(document.sites.pop(site_id))

    if not removed:
        return SiteGenerationOutcome(sites=[])
    record = ProvenanceRecord(
        action="delete_candidates",
        workflow_stage=WorkflowStage.MUON_GENERATION,
        actor="user",
        description=f"Deleted {plural(len(removed), 'generated muon site')}",
        parameters={"site_labels": [site.label for site in removed]},
        input_entity_ids=[site.id for site in removed],
    )
    return SiteGenerationOutcome(sites=removed, provenance=[record])


def stage_generated_sites(
    document: ProjectDocument, site_ids: Sequence[UUID]
) -> SiteGenerationOutcome:
    """Mark generated (or already staged) muon sites as staged for structural relaxation.

    The outcome lists the staged sites, with a ``stage_muon_relaxation_sites``
    record when any were staged.
    """

    staged: list[MuonSite] = []
    for site_id in site_ids:
        site = document.sites.get(site_id)
        if site is None or site.status not in _STAGEABLE_SITE_STATUSES:
            continue
        site.status = _STAGED_SITE_STATUS
        staged.append(site)

    if not staged:
        return SiteGenerationOutcome(sites=[])
    record = ProvenanceRecord(
        action="stage_muon_relaxation_sites",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        actor="user",
        description=f"Staged {plural(len(staged), 'generated muon site')} for relaxation",
        parameters={"site_labels": [site.label for site in staged]},
        input_entity_ids=[site.id for site in staged],
        output_entity_ids=[site.id for site in staged],
    )
    return SiteGenerationOutcome(sites=staged, provenance=[record])


def _requested_site_count(
    method: str, parameters: dict[str, Any], host: _HostContext
) -> int | None:
    """Return how many sites the request asks for, or ``None`` when it sets no target."""

    if method == "manual":
        return 1
    if method == "random" and _random_selection_mode(parameters) == "separation":
        return None
    if method == "uep_clusters" and isinstance(parameters.get("selections"), list):
        return len(parameters.get("selections", []))
    if method == "near_element":
        target = str(parameters.get("element", "")).strip()
        distinct_atoms = (
            len(
                symmetry_distinct_atom_indexes(
                    lattice_vectors=host.lattice,
                    host_species=host.species,
                    host_positions_frac=host.positions,
                    target_element=target,
                )
            )
            if target
            else 0
        )
        return distinct_atoms * _muons_per_atom(parameters) if distinct_atoms else None
    return int(parameters.get("count", 1))


def _generate_manual_site(
    document: ProjectDocument,
    host: _HostContext,
    parameters: dict[str, Any],
) -> _GeneratedBatch:
    """Place one site at user-entered fractional or Cartesian coordinates."""

    coord_system = str(parameters.get("coord_system", "fractional"))
    coords = [
        float(parameters.get("x", 0.0)),
        float(parameters.get("y", 0.0)),
        float(parameters.get("z", 0.0)),
    ]
    lattice = host.lattice or _IDENTITY_LATTICE
    if coord_system == "cartesian":
        fractional = [v % 1.0 for v in lattice_cart_to_frac(coords, lattice)]
    else:
        fractional = [v % 1.0 for v in coords]
    cartesian = lattice_frac_to_cart(fractional, lattice)

    label_prefix = "Manual Site"
    site = MuonSite(
        label=f"{label_prefix} {_next_site_label_number(_site_labels(document), label_prefix)}",
        status=GENERATED_SITE_STATUS,
        initial_fractional_position=fractional,
        initial_cartesian_position=cartesian,
        generation_method="manual",
        source_structure_id=host.structure_id,
        properties={"coordinate_system": coord_system},
    )
    return _GeneratedBatch(sites=[site])


def _generate_random_sites(
    site_labels: Iterable[str],
    host: _HostContext,
    parameters: dict[str, Any],
    requested_count: int | None,
    progress_callback: ProgressCallback | None,
) -> _GeneratedBatch:
    """Draw random sites by minimum separation or by a target count."""

    selection_mode = _random_selection_mode(parameters)
    seed = _parse_optional_int(parameters.get("seed"), field_name="seed")
    min_muon_atom = float(parameters.get("min_muon_atom", 1.0))
    include_symmetry_equivalents = bool(parameters.get("include_symmetry_equivalents", True))
    if selection_mode == "separation":
        draw = _random_draw_by_separation(
            host, parameters, seed, min_muon_atom, include_symmetry_equivalents, progress_callback
        )
    else:
        target_count = max(1, int(requested_count or parameters.get("count", 1)))
        draw = _random_draw_by_count(
            host,
            parameters,
            target_count,
            seed,
            min_muon_atom,
            include_symmetry_equivalents,
            progress_callback,
        )

    label_prefix = "Random Site"
    start_index = _next_site_label_number(site_labels, label_prefix)
    sites = [
        MuonSite(
            label=f"{label_prefix} {start_index + idx - 1}",
            status=GENERATED_SITE_STATUS,
            initial_fractional_position=point.fractional,
            initial_cartesian_position=point.cartesian,
            generation_method="random",
            source_structure_id=host.structure_id,
            properties={
                "selection_mode": selection_mode,
                "min_muon_muon": draw.min_muon_muon,
                "min_muon_atom": min_muon_atom,
                "include_symmetry_equivalents": include_symmetry_equivalents,
                **(
                    {"target_count": int(requested_count or 0)} if selection_mode == "count" else {}
                ),
                **(
                    {"estimated_min_muon_muon": draw.estimated_min_muon_muon}
                    if draw.estimated_min_muon_muon is not None
                    else {}
                ),
            },
        )
        for idx, point in enumerate(draw.points, start=1)
    ]
    return _GeneratedBatch(sites=sites, attempts_used=draw.attempts_used)


def _random_draw_by_separation(
    host: _HostContext,
    parameters: dict[str, Any],
    seed: int | None,
    min_muon_atom: float,
    include_symmetry_equivalents: bool,
    progress_callback: ProgressCallback | None,
) -> _RandomDraw:
    """Fill the cell with sites at least ``min_muon_muon`` apart."""

    min_muon_muon = float(parameters.get("min_muon_muon", 0.8))
    predicted_count = max(
        1,
        int(
            estimate_random_site_count_from_separation(
                lattice_vectors=host.lattice,
                host_positions_frac=host.positions,
                host_species=host.species,
                min_muon_muon=min_muon_muon,
                min_muon_atom=min_muon_atom,
                include_symmetry_equivalents=include_symmetry_equivalents,
            )
            + 0.999
        ),
    )
    points, attempts_used = generate_random_sites(
        lattice_vectors=host.lattice,
        host_positions_frac=host.positions,
        host_species=host.species,
        min_muon_muon=min_muon_muon,
        min_muon_atom=min_muon_atom,
        max_attempts=int(parameters.get("max_attempts", 20000)),
        seed=seed,
        include_symmetry_equivalents=include_symmetry_equivalents,
        expected_count=predicted_count,
        progress_callback=progress_callback,
    )
    return _RandomDraw(points=points, attempts_used=attempts_used, min_muon_muon=min_muon_muon)


def _random_draw_by_count(
    host: _HostContext,
    parameters: dict[str, Any],
    target_count: int,
    seed: int | None,
    min_muon_atom: float,
    include_symmetry_equivalents: bool,
    progress_callback: ProgressCallback | None,
) -> _RandomDraw:
    """Draw ``target_count`` sites, shrinking the estimated separation until they fit."""

    estimated_min_muon_muon = estimate_muon_muon_separation_for_count(
        lattice_vectors=host.lattice,
        host_positions_frac=host.positions,
        host_species=host.species,
        target_count=target_count,
        min_muon_atom=min_muon_atom,
        include_symmetry_equivalents=include_symmetry_equivalents,
    )
    min_muon_muon = estimated_min_muon_muon
    last_error: SiteGenerationError | None = None
    for _attempt in range(_RANDOM_COUNT_RETRIES):
        try:
            points, attempts_used = generate_random_sites(
                lattice_vectors=host.lattice,
                host_positions_frac=host.positions,
                host_species=host.species,
                count=target_count,
                min_muon_muon=min_muon_muon,
                min_muon_atom=min_muon_atom,
                max_attempts=int(parameters.get("max_attempts", 20000)),
                seed=seed,
                include_symmetry_equivalents=include_symmetry_equivalents,
                progress_callback=progress_callback,
            )
        except SiteGenerationError as exc:
            last_error = exc
            min_muon_muon *= _RANDOM_COUNT_SEPARATION_SHRINK
            continue
        return _RandomDraw(
            points=points,
            attempts_used=attempts_used,
            min_muon_muon=min_muon_muon,
            estimated_min_muon_muon=estimated_min_muon_muon,
        )
    assert last_error is not None
    raise last_error


def _generate_attached_element_sites(
    site_labels: Iterable[str],
    host: _HostContext,
    parameters: dict[str, Any],
    progress_callback: ProgressCallback | None,
) -> _GeneratedBatch:
    """Attach sites at a bond distance from every host atom of one element."""

    muons_per_atom = _muons_per_atom(parameters)
    target_element = str(parameters.get("element", "")).strip()
    bond_distance = float(parameters.get("bond_distance", 1.0))
    min_muon_atom = float(parameters.get("min_muon_atom", 0.8))
    points, attempts_used = generate_attached_element_sites(
        lattice_vectors=host.lattice,
        host_species=host.species,
        host_positions_frac=host.positions,
        target_element=target_element,
        muons_per_atom=muons_per_atom,
        bond_distance=bond_distance,
        min_muon_atom=min_muon_atom,
        progress_callback=progress_callback,
    )
    label_prefix = f"Attached-{parameters.get('element', '').strip() or 'Element'} Site"
    start_index = _next_site_label_number(site_labels, label_prefix)
    sites = [
        MuonSite(
            label=f"{label_prefix} {start_index + idx - 1}",
            status=GENERATED_SITE_STATUS,
            initial_fractional_position=point.fractional,
            initial_cartesian_position=point.cartesian,
            generation_method="near_element",
            source_structure_id=host.structure_id,
            properties={
                "target_element": target_element,
                "strategy": "attached",
                "muons_per_atom": muons_per_atom,
                "bond_distance": bond_distance,
                "min_muon_atom": min_muon_atom,
            },
        )
        for idx, point in enumerate(points, start=1)
    ]
    return _GeneratedBatch(sites=sites, attempts_used=attempts_used)


def _generate_uep_cluster_sites(
    document: ProjectDocument,
    host: _HostContext,
    parameters: dict[str, Any],
) -> _GeneratedBatch:
    """Add the selected UEP cluster rows as generated sites, skipping ones already present."""

    selections = parameters.get("selections", [])
    if not isinstance(selections, list) or not selections:
        raise SiteGenerationError("Select one or more UEP cluster rows to add as generated sites.")

    sites: list[MuonSite] = []
    skipped_duplicates = 0
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        site = _uep_cluster_site(document, host, selection, created_so_far=len(sites))
        if site is None:
            skipped_duplicates += 1
            continue
        sites.append(site)

    if not sites and skipped_duplicates:
        raise SiteGenerationError(
            "Selected UEP cluster rows are already present in the generated-site list."
        )
    if not sites:
        raise SiteGenerationError(
            "Selected UEP cluster rows did not contain any usable generated sites."
        )
    return _GeneratedBatch(sites=sites)


def _uep_cluster_site(
    document: ProjectDocument,
    host: _HostContext,
    selection: dict[str, Any],
    *,
    created_so_far: int,
) -> MuonSite | None:
    """Build the site for one UEP cluster row, or ``None`` when the document already has it."""

    fractional_raw = selection.get("fractional_position", selection.get("fractional", []))
    cartesian_raw = selection.get("cartesian_position", selection.get("cartesian", []))
    if (
        not isinstance(fractional_raw, (list, tuple))
        or len(fractional_raw) != 3
        or not isinstance(cartesian_raw, (list, tuple))
        or len(cartesian_raw) != 3
    ):
        raise SiteGenerationError("Selected UEP cluster row is missing valid coordinates.")

    fractional = [float(value) % 1.0 for value in fractional_raw]
    cartesian = [float(value) for value in cartesian_raw]
    source_key = str(selection.get("source_key", "")).strip()
    if source_key and _has_duplicate_uep_cluster_site(document, source_key, fractional):
        return None

    properties = selection.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    method_name = str(selection.get("generation_method", "")).strip()
    if method_name not in _UEP_CLUSTER_METHODS:
        method_name = (
            "uep_cluster_member" if properties.get("selection_kind") == "member" else "uep_cluster"
        )
    merged_properties = dict(properties)
    if source_key:
        merged_properties["uep_source_key"] = source_key

    source_structure_id = None
    raw_source_structure_id = selection.get("source_structure_id")
    if raw_source_structure_id:
        try:
            source_structure_id = UUID(str(raw_source_structure_id))
        except ValueError:
            source_structure_id = host.structure_id
    elif host.structure_id is not None:
        source_structure_id = host.structure_id

    explicit_label = str(selection.get("label", "")).strip()
    if explicit_label:
        label = explicit_label
    else:
        label_hint = "UEP Cluster Site" if method_name == "uep_cluster" else "UEP Cluster Member"
        label = f"{label_hint} {len(document.sites) + created_so_far + 1}"
    return MuonSite(
        label=label,
        status=GENERATED_SITE_STATUS,
        initial_fractional_position=fractional,
        initial_cartesian_position=cartesian,
        generation_method=method_name,
        source_structure_id=source_structure_id,
        properties=merged_properties,
    )


def _generation_provenance(
    method: str,
    parameters: dict[str, Any],
    batch: _GeneratedBatch,
    requested_count: int | None,
) -> list[ProvenanceRecord]:
    """Describe a generation request, flagging a shortfall against the requested count."""

    created_ids = [site.id for site in batch.sites]
    records = [
        ProvenanceRecord(
            action="generate_candidates",
            workflow_stage=WorkflowStage.MUON_GENERATION,
            actor="user",
            description=f"Generated {plural(len(batch.sites), 'site')} via {method}",
            parameters={"method": method, **parameters},
            output_entity_ids=list(created_ids),
        )
    ]
    if requested_count is not None and method != "manual" and len(batch.sites) < requested_count:
        reason = (
            "Some selected UEP rows were already present."
            if method == "uep_clusters"
            else "Increase max attempts, reduce distance limits, or use a larger supercell."
        )
        records.append(
            ProvenanceRecord(
                action="generation_partial",
                workflow_stage=WorkflowStage.MUON_GENERATION,
                actor="system",
                description=f"Requested {requested_count} sites but generated {len(batch.sites)}. {reason}",
                parameters={
                    "requested": requested_count,
                    "generated": len(batch.sites),
                    "attempts": batch.attempts_used,
                },
                output_entity_ids=list(created_ids),
            )
        )
    return records


def _muons_per_atom(parameters: dict[str, Any]) -> int:
    """Return the attached sites per host atom, accepting the older parameter names."""

    return max(
        1,
        int(
            parameters.get(
                "muons_per_atom",
                parameters.get("max_candidates_per_atom", parameters.get("count", 1)),
            )
        ),
    )


def _has_duplicate_uep_cluster_site(
    document: ProjectDocument, source_key: str, fractional: Sequence[float]
) -> bool:
    for site in document.sites.values():
        if site.properties.get("uep_source_key") == source_key:
            return True
        if site.generation_method not in {"uep_cluster", "uep_cluster_member", "uep_batch_cluster"}:
            continue
        if not site.initial_fractional_position or len(site.initial_fractional_position) != 3:
            continue
        if all(
            abs((float(a) - float(b) + 0.5) % 1.0 - 0.5) <= 1e-6
            for a, b in zip(site.initial_fractional_position, fractional)
        ):
            return True
    return False


_NUMBERED_LABEL = re.compile(r"^(.*\S)\s+\d+$")


def _site_labels(document: ProjectDocument) -> list[str]:
    return [str(site.label) for site in document.sites.values()]


def _next_site_label_number(site_labels: Iterable[str], prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix.strip())}\s+(\d+)$")
    used_numbers: list[int] = []
    for label in site_labels:
        match = pattern.match(label.strip())
        if match is not None:
            used_numbers.append(int(match.group(1)))
    return max(used_numbers, default=0) + 1


def _random_selection_mode(parameters: dict[str, Any]) -> str:
    mode = str(parameters.get("selection_mode", parameters.get("mode", ""))).strip().lower()
    if mode in {"separation", "distance", "spacing"}:
        return "separation"
    if mode in {"count", "number", "target_count"}:
        return "count"
    return "count" if "count" in parameters else "separation"


def _parse_optional_int(value: Any, *, field_name: str) -> int | None:
    """Parse optional integer-like input from UI payloads."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise SiteGenerationError(f"{field_name} must be an integer if provided.") from exc
