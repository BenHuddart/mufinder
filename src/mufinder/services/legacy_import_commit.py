"""Turn detected legacy relaxations into project entities and files.

Where :mod:`mufinder.services.legacy_import` only *reports* what a folder holds,
this module commits it: for each selected run it builds the five-entity chain
the downstream eligibility gate requires, materialises a job directory, and
records provenance.

Two contracts govern what is built, and both are easy to satisfy incompletely:

**The eligibility gate** (``workflow.quantum.quantum_source_eligibility``, plus
the caller that resolves run → job → result → structure) decides whether an
imported site appears as a Quantum Corrections or Energy Barriers source.

**Host resolution** (``ProjectService.resolve_underlying_host``) decides whether
a downstream job can actually be *submitted*. It walks
``Structure.parent_structure_id`` for an ancestor whose origin is ``imported``
or ``pristine_relaxation``. An imported relaxed structure has origin
``muon_relaxation``, so without a parent the walk returns ``None`` and every
submission dies in a host-mismatch dialog — after passing every visible check.
Setting the parent is one line and the whole feature depends on it.

Files are laid out so the *downstream parameter inheritance* path finds them:
that path re-reads ``<jobdir>/<seed>.cell`` and ``<jobdir>/<seed>.param`` from
disk rather than consulting the document. Legacy inputs frequently state neither
the cutoff (implied by ``BASIS_PRECISION``) nor the k-point grid (never written
to the local input cell), so the seed files are **re-rendered** from the
parameters the ``.castep`` reports, with the byte-exact originals archived
alongside. See ``docs/legacy_import.rst``.

Layering: ``services`` may use ``castep_io``, ``science`` and ``domain``. It must
not import ``app`` or ``workflow``, so the definition rows this module produces
are plain payload dictionaries in the persisted configuration shape rather than
``MuonRelaxationRow`` instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from mufinder.castep_io.cell_writer import write_cell
from mufinder.castep_io.param_writer import write_param
from mufinder.castep_io.schema import canonical_keyword_name, get_keyword_spec
from mufinder.castep_io.types import (
    AtomicPosition,
    BlockSection,
    CellFile,
    KeywordAssignment,
    LatticeCart,
    ParamFile,
    PositionBlock,
)
from mufinder.domain.charge_states import (
    POSITIVE_MUON_CHARGE_STATE,
    muon_charge_for_state,
    muon_charge_state_prefix,
    muon_relaxation_calc_type,
)
from mufinder.domain.constants import IMPORTED_JOB_TAG, MUON_GAMMA_RAD_S_T, MUON_MASS_AMU
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import (
    CalculationRun,
    Job,
    MuonSite,
    ProvenanceRecord,
    Result,
    Structure,
    next_job_created_at,
    utc_now,
)
from mufinder.science.lattice import lattice_frac_to_cart, vector_length, wrap_fractional
from mufinder.services.legacy_import import (
    MUON_SPECIES,
    DetectedRun,
    LegacyFolderScan,
)

#: Directory inside a job folder holding the byte-exact originals.
LEGACY_SOURCE_DIRNAME = "legacy_source"
#: Manifest naming every archived file with its digest.
LEGACY_MANIFEST_NAME = "legacy_source_manifest.json"
#: Provenance action for one import. ``services.journal`` renders nothing for an
#: action it does not know, so a matching journal builder is required.
IMPORT_ACTION = "import_legacy_relaxations"
#: ``Structure.origin`` demanded by the eligibility gate. The import marker lives
#: in ``properties`` instead — see :func:`_build_structure`.
RELAXED_STRUCTURE_ORIGIN = "muon_relaxation"


@dataclass(frozen=True, slots=True)
class ImportedRelaxation:
    """Everything one imported run contributed to the project."""

    site_id: UUID
    job_id: UUID
    run_id: UUID
    result_id: UUID
    structure_id: UUID
    label: str
    seed_name: str
    charge_state: str
    job_directory: Path
    #: Definition row in the persisted configuration shape, for the app layer to
    #: merge into the active muon-relaxation definition.
    row_payload: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LegacyImportOutcome:
    """The result of committing one import batch."""

    batch_id: UUID
    imported: tuple[ImportedRelaxation, ...] = ()
    skipped: tuple[tuple[int, str], ...] = ()
    warnings: tuple[str, ...] = ()
    #: The batch's provenance record, for the caller to persist. Returned rather
    #: than appended to the document, because appending only reaches memory:
    #: ``provenance.jsonl`` is append-only through
    #: ``ProjectService.append_provenance`` and ``save_project`` never rewrites
    #: it, so a record appended here would vanish on reload.
    provenance: ProvenanceRecord | None = None

    @property
    def row_payloads(self) -> list[dict[str, Any]]:
        """Return the definition rows this batch added, in import order."""

        return [item.row_payload for item in self.imported]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _fold_to_host(
    position: tuple[float, float, float],
    matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None,
) -> tuple[float, float, float]:
    """Fold a supercell fractional position back into host-cell coordinates.

    Mirrors the native muon-relaxation completion path, so an imported site's
    recorded position means the same thing as a natively relaxed one's.
    """

    # A supercell fractional row vector times the supercell matrix is the host fractional position.
    folded = position if matrix is None else lattice_frac_to_cart(position, matrix)
    return tuple(wrap_fractional(folded))  # type: ignore[return-value]


def _relabelled(species: Sequence[str], muon_label: str | None) -> list[str]:
    """Return species with the muon relabelled to ``H:mu``.

    The eligibility gate hard-requires ``H:mu``. A legacy run may have used a
    bare ``H``, so the label is normalised here and the change is recorded as a
    warning rather than applied silently.
    """

    if muon_label is None or muon_label == MUON_SPECIES:
        return [str(item) for item in species]
    relabelled: list[str] = []
    replaced = False
    for item in reversed([str(value) for value in species]):
        if not replaced and item == muon_label:
            relabelled.append(MUON_SPECIES)
            replaced = True
        else:
            relabelled.append(item)
    return list(reversed(relabelled))


def _cell_for_seed(
    *,
    lattice_vectors: Sequence[Sequence[float]],
    species: Sequence[str],
    positions: Sequence[Sequence[float]],
    parameters: dict[str, Any],
) -> CellFile:
    """Build the ``.cell`` an imported job directory should carry.

    Holds the run's **initial** structure — so initial-versus-relaxed comparison
    stays truthful — with the k-point grid the run actually used made explicit.
    That grid is what the downstream inheritance path reads, and a legacy input
    cell frequently does not contain it.
    """

    grid = parameters.get(canonical_keyword_name("cell", "kpoint_mp_grid"))
    kpoint_mp_grid: tuple[int, int, int] | None = None
    if isinstance(grid, (list, tuple)) and len(grid) == 3:
        try:
            kpoint_mp_grid = (int(grid[0]), int(grid[1]), int(grid[2]))
        except (TypeError, ValueError):
            kpoint_mp_grid = None

    return CellFile(
        lattice_cart=LatticeCart(
            vectors=tuple(tuple(float(value) for value in row) for row in lattice_vectors),  # type: ignore[arg-type]
            unit="ang",
        ),
        positions_frac=PositionBlock(
            positions=tuple(
                AtomicPosition(
                    species=str(name),
                    coordinates=(float(row[0]), float(row[1]), float(row[2])),
                )
                for name, row in zip(species, positions, strict=False)
            )
        ),
        kpoint_mp_grid=kpoint_mp_grid,
        blocks=(
            BlockSection(name="species_mass", rows=((MUON_SPECIES, MUON_MASS_AMU),), unit="AMU"),
            BlockSection(
                name="species_gamma",
                rows=((MUON_SPECIES, MUON_GAMMA_RAD_S_T),),
                unit="radsectesla",
            ),
        ),
        scalars=(KeywordAssignment("fix_all_cell", True),),
    )


#: Keywords that belong in the rendered ``.param``. Structure-controlled and
#: cell-file keywords are excluded: they are carried by the ``.cell``, and
#: emitting them here would produce a file CASTEP rejects.
_PARAM_EXCLUDED = frozenset({"kpoint_mp_grid", "kpoint_mp_spacing", "fix_all_cell", "symmetry_generate"})

#: Keywords the import *asserts* rather than recovers. ``TASK`` and ``CHARGE``
#: define what the job is; ``FIX_ALL_CELL`` and ``WRITE_CELL_STRUCTURE`` are
#: required by the workflow. All four are rendered as locked values in the
#: parameter dialog, so a recovered value for one of them must never reach a
#: definition row and appear as if the row had chosen it.
_ASSERTED_KEYWORDS = frozenset({"task", "charge", "fix_all_cell", "write_cell_structure"})

#: Resolved first, so a mutually exclusive pair resolves the right way round.
_PRIORITY_KEYWORDS = ("cut_off_energy",)


def resolve_recovered_parameters(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve a merged recovery into the parameter set the run actually used.

    Returns the resolved ``param`` keywords in canonical form, in deterministic
    emission order, and the keywords deliberately dropped because a mutually
    exclusive partner won.

    Only keywords the runtime schema recognises as ``param`` keywords survive,
    so a file rendered from this always validates.

    **Mutually exclusive keywords must be resolved, not merged.** A legacy run
    typically declares ``BASIS_PRECISION`` in its ``.param`` while the output
    echo reports the ``CUT_OFF_ENERGY`` CASTEP derived from it, so a merged
    parameter set naturally holds both — and CASTEP treats them as exclusive.
    The resolved cutoff wins: it is what the calculation actually used, and it
    is the value downstream jobs inherit. Emitting them in the order they happen
    to sort would keep the wrong one.

    Shared deliberately by the rendered ``.param`` and by the definition row the
    parameter dialog displays. The row is what a user reads to learn what the
    run used; the file is what downstream jobs inherit. Two descriptions of one
    historical run that disagree would be worse than either being wrong alone,
    so the rule lives here once rather than being restated at each call site.
    """

    resolved: dict[str, Any] = {}
    dropped: list[str] = []
    # ``task`` and ``charge`` are forced by the caller, so seed them as seen:
    # anything mutually exclusive with them must lose, exactly as it would have
    # had they been emitted from recovery.
    seen = {"task", "charge"}

    def _emit(canonical: str, value: Any) -> None:
        spec = get_keyword_spec("param", canonical)
        if spec is None:
            return
        for other in spec.mutually_exclusive_with:
            if canonical_keyword_name("param", other) in seen:
                dropped.append(canonical)
                return
        resolved[canonical] = value
        seen.add(canonical)

    for preferred in _PRIORITY_KEYWORDS:
        canonical = canonical_keyword_name("param", preferred)
        if parameters.get(canonical) is not None:
            _emit(canonical, parameters[canonical])

    for name, value in sorted(parameters.items()):
        canonical = canonical_keyword_name("param", str(name))
        if canonical in seen or canonical in _PARAM_EXCLUDED or value is None:
            continue
        _emit(canonical, value)

    return resolved, dropped


def _param_for_seed(
    parameters: dict[str, Any], *, charge_state: str
) -> tuple[ParamFile, list[str]]:
    """Build the ``.param`` an imported job directory should carry.

    Returns the file and the keywords deliberately dropped from it. ``TASK`` and
    ``CHARGE`` are forced to the values the import asserts rather than taken from
    recovery, because they define what the job *is*; everything else comes from
    :func:`resolve_recovered_parameters`.
    """

    resolved, dropped = resolve_recovered_parameters(parameters)
    keywords: list[KeywordAssignment] = [
        KeywordAssignment("task", "GEOMETRYOPTIMIZATION"),
        KeywordAssignment("charge", muon_charge_for_state(charge_state)),
    ]
    keywords.extend(
        KeywordAssignment(name, value) for name, value in resolved.items()
    )

    return ParamFile(keywords=tuple(keywords)), dropped


def _materialise_files(
    *,
    run: DetectedRun,
    job_directory: Path,
    seed_name: str,
    species: Sequence[str],
    positions: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
    parameters: dict[str, Any],
    charge_state: str,
) -> tuple[list[str], list[str]]:
    """Write one imported job directory. Returns (relative files, warnings)."""

    warnings: list[str] = []
    job_directory.mkdir(parents=True, exist_ok=True)
    archive = job_directory / LEGACY_SOURCE_DIRNAME
    archive.mkdir(parents=True, exist_ok=True)

    written: list[str] = []

    # Byte-exact results, under their own names.
    manifest: list[dict[str, str]] = []
    for source in (run.castep_path, run.out_cell_path):
        if source is None or not source.is_file():
            continue
        target = job_directory / source.name
        shutil.copy2(source, target)
        written.append(target.name)
        manifest.append(
            {"file": target.name, "source": str(source), "sha256": _digest(target)}
        )

    # Byte-exact originals of the *inputs*, which the rendered seed files replace.
    for source in (run.input_cell_path, run.run_param_path):
        if source is None or not source.is_file():
            continue
        target = archive / source.name
        shutil.copy2(source, target)
        manifest.append(
            {
                "file": f"{LEGACY_SOURCE_DIRNAME}/{target.name}",
                "source": str(source),
                "sha256": _digest(target),
            }
        )

    (archive / LEGACY_MANIFEST_NAME).write_text(
        json.dumps({"version": 1, "files": manifest}, indent=2) + "\n", encoding="utf-8"
    )

    try:
        cell_text = write_cell(
            _cell_for_seed(
                lattice_vectors=lattice_vectors,
                species=species,
                positions=positions,
                parameters=parameters,
            )
        )
        (job_directory / f"{seed_name}.cell").write_text(cell_text, encoding="utf-8", newline="\n")
        written.append(f"{seed_name}.cell")
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        warnings.append(
            f"Could not render {seed_name}.cell, so downstream jobs will not inherit "
            f"this run's k-point grid: {exc}"
        )

    try:
        param_file, dropped = _param_for_seed(parameters, charge_state=charge_state)
        param_text = write_param(param_file)
        (job_directory / f"{seed_name}.param").write_text(param_text, encoding="utf-8", newline="\n")
        written.append(f"{seed_name}.param")
        if dropped:
            warnings.append(
                "Superseded by the value the run actually used, so omitted from the "
                f"rendered .param: {', '.join(sorted(dropped))}."
            )
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        warnings.append(
            f"Could not render {seed_name}.param, so downstream jobs will not inherit "
            f"this run's cutoff: {exc}"
        )

    return written, warnings


def _build_structure(
    *,
    run: DetectedRun,
    label: str,
    host_structure: Structure,
    species: Sequence[str],
    positions: Sequence[Sequence[float]],
    lattice_vectors: Sequence[Sequence[float]],
    job_id: UUID,
    batch_id: UUID,
    mulliken_spins: Sequence[float] | None,
) -> Structure:
    """Build the relaxed structure an imported run contributes.

    ``origin`` and ``source_workflow_stage`` are pinned to the values the
    eligibility gate demands, so the import marker lives in ``properties``
    instead. ``parent_structure_id`` is the active host: without it every
    downstream submission fails host resolution.
    """

    properties: dict[str, Any] = {
        "source_workflow_stage": WorkflowStage.MUON_RELAXATION.value,
        "source_job_id": str(job_id),
        "import_source": {
            "kind": "legacy_mufinder",
            "batch_id": str(batch_id),
            "site_index": run.site_index,
            "seed_name": run.seed_name,
            "castep_version": run.castep_version,
        },
    }
    if mulliken_spins:
        # Legacy CASTEP inputs never carried per-atom spins, so the converged
        # Mulliken moments are the only source. Recorded as such rather than
        # presented as user-specified input.
        properties["atomic_spins"] = [float(value) for value in mulliken_spins]
        properties["atomic_spins_source"] = "castep_mulliken"

    return Structure(
        label=label,
        origin=RELAXED_STRUCTURE_ORIGIN,
        lattice_vectors=[[float(value) for value in row] for row in lattice_vectors],
        species=[str(name) for name in species],
        fractional_positions=[[float(value) for value in row] for row in positions],
        parent_structure_id=host_structure.id,
        properties=properties,
    )


def row_parameter_values(run: DetectedRun) -> dict[str, Any]:
    """Return the parameters an imported definition row should carry.

    The resolved recovery — the same set the rendered ``.param`` holds — plus
    ``KPOINT_MP_GRID``, which belongs to the ``.cell`` rather than the ``.param``
    and is frequently recoverable only from the output echo. The keywords the
    import asserts are excluded; see :data:`_ASSERTED_KEYWORDS`.

    An empty result is meaningful and must be preserved: it says recovery found
    nothing, not that the run used the workflow's current defaults.
    """

    resolved, _ = resolve_recovered_parameters(run.parameters)
    values = {
        name: value
        for name, value in resolved.items()
        if name not in _ASSERTED_KEYWORDS
    }

    grid = run.parameters.get(canonical_keyword_name("cell", "kpoint_mp_grid"))
    if isinstance(grid, (list, tuple)) and len(grid) == 3:
        try:
            values["kpoint_mp_grid"] = [int(component) for component in grid]
        except (TypeError, ValueError):
            pass
    return values


def _row_payload(
    *,
    site_id: UUID,
    label: str,
    run: DetectedRun,
    charge_state: str,
    batch_id: UUID,
) -> dict[str, Any]:
    """Build one definition row in the persisted configuration shape.

    Plain data rather than a ``MuonRelaxationRow``: that dataclass lives in the
    app layer, which ``services`` may not import.

    ``added_advanced`` is deliberately left empty rather than computed from the
    recovered names. The parameter dialog promotes any advanced keyword that
    arrives carrying an active value, so a recovered advanced keyword renders
    without being listed here — and an empty list cannot drift with the runtime
    keyword catalog, which keeps the row signature stable across machines.
    """

    # Host frame, not the supercell frame the input cell states it in. Every
    # other relaxation row takes this field from ``MuonSite.initial_fractional_position``
    # (host coordinates), and job expansion converts host -> supercell on the way
    # out. Storing supercell coordinates here would show the wrong numbers in the
    # table's Initial columns and, if the row were ever re-expanded, transform
    # them a second time.
    initial = _fold_to_host(
        run.initial_muon_fractional_supercell or (0.0, 0.0, 0.0),
        run.supercell_matrix,
    )
    return {
        "site_id": str(site_id),
        "label": label,
        "initial_fractional_position": [float(value) for value in initial],
        "charge_state": charge_state,
        "supercell_matrix": (
            [list(line) for line in run.supercell_matrix]
            if run.supercell_matrix is not None
            else None
        ),
        "parameter_values": row_parameter_values(run),
        "added_advanced": [],
        "imported": True,
        "import_batch_id": str(batch_id),
    }


def _unique_label(base: str, taken: set[str]) -> str:
    """Return ``base``, suffixed until unused.

    Row keys are ``{charge_prefix}:{label}``, so a collision would silently
    overwrite an existing row's job mapping.
    """

    if base not in taken:
        taken.add(base)
        return base
    index = 2
    while f"{base} ({index})" in taken:
        index += 1
    label = f"{base} ({index})"
    taken.add(label)
    return label


def commit_legacy_import(
    document: Any,
    *,
    scan: LegacyFolderScan,
    runs: Sequence[DetectedRun],
    host_structure: Structure,
    output_root: Path,
    batch_id: UUID | None = None,
    label_prefix: str = "Legacy Site",
    existing_labels: Sequence[str] = (),
) -> LegacyImportOutcome:
    """Build entities and files for ``runs``, returning what was created.

    ``document`` is a ``ProjectDocument``; it is typed loosely to avoid a
    circular import with :mod:`mufinder.services.project_service`. Entities are
    inserted directly into its maps, and one provenance record is appended for
    the batch.

    The caller owns two things this does not do: merging
    :attr:`LegacyImportOutcome.row_payloads` into the active definition (which
    needs the widget, and so belongs to the app layer), and saving the project.
    """

    batch = batch_id or uuid4()
    imported: list[ImportedRelaxation] = []
    site_records: list[dict[str, Any]] = []
    skipped: list[tuple[int, str]] = []
    warnings: list[str] = list(scan.warnings)
    taken = set(existing_labels)

    for run in runs:
        if not run.importable:
            skipped.append((run.site_index, "; ".join(run.blockers)))
            continue
        if run.out_cell_path is None:
            skipped.append((run.site_index, "No relaxed cell to import."))
            continue

        charge_state = run.charge_state or POSITIVE_MUON_CHARGE_STATE
        if run.charge_state is None:
            warnings.append(
                f"Site {run.site_index}: charge state was undetermined; imported as a "
                "positive muon."
            )

        from mufinder.castep_io.cell_parser import parse_cell_file

        try:
            relaxed = parse_cell_file(run.out_cell_path)
        except Exception as exc:  # noqa: BLE001
            skipped.append((run.site_index, f"Could not read the relaxed cell: {exc}"))
            continue

        species = _relabelled(relaxed.species, run.muon_label)
        label = _unique_label(f"{label_prefix} {run.site_index}", taken)
        prefix = muon_charge_state_prefix(charge_state)
        seed_name = run.seed_name
        job_directory = Path(output_root) / prefix / seed_name

        job = Job(
            label=label,
            workflow_stage=WorkflowStage.MUON_RELAXATION,
            calc_type=muon_relaxation_calc_type(charge_state),
            status=JobStatus.COMPLETED,
            execution_target="local",
            created_at=next_job_created_at(document.jobs.values()),
            tags=[
                WorkflowStage.MUON_RELAXATION.value,
                muon_relaxation_calc_type(charge_state),
                IMPORTED_JOB_TAG,
                "legacy_mufinder",
            ],
            completed_at=utc_now(),
        )

        structure = _build_structure(
            run=run,
            label=f"{host_structure.label}_{prefix}_{label}".replace(" ", "_"),
            host_structure=host_structure,
            species=species,
            positions=relaxed.fractional_positions,
            lattice_vectors=relaxed.lattice_vectors,
            job_id=job.id,
            batch_id=batch,
            mulliken_spins=None,
        )

        # The site's initial position is where the muon *started*; the native
        # completion path never rewrites it. The relaxed host-cell position is
        # copied to ``relaxed_fractional_position`` below from the same value
        # the result payload records, exactly as a native completion does.
        initial_host = _fold_to_host(
            run.initial_muon_fractional_supercell
            or run.final_muon_fractional_supercell
            or (0.0, 0.0, 0.0),
            run.supercell_matrix,
        )
        site = MuonSite(
            label=label,
            status="relaxed",
            initial_fractional_position=list(initial_host),
            initial_cartesian_position=lattice_frac_to_cart(
                initial_host, host_structure.lattice_vectors
            ),
            generation_method="imported_legacy",
            source_structure_id=host_structure.id,
            properties={"import_batch_id": str(batch), "legacy_site_index": run.site_index},
        )

        run_record = CalculationRun(
            job_id=job.id,
            status=JobStatus.COMPLETED,
            completed_at=utc_now(),
            convergence_achieved=run.converged,
        )
        final_supercell = run.final_muon_fractional_supercell
        final_host = (
            _fold_to_host(final_supercell, run.supercell_matrix) if final_supercell is not None else None
        )
        site.record_relaxation(
            run_record.id,
            total_energy_eV=run.total_energy,
            fractional_position=final_host,
        )

        payload: dict[str, Any] = {
            "completed": True,
            "total_energy": run.total_energy,
            "geom_converged": run.converged,
            "imported": True,
        }
        if final_supercell is not None and final_host is not None:
            payload["muon_final_fractional_position_supercell"] = list(final_supercell)
            payload["muon_final_fractional_position_host"] = list(final_host)
        lattice_a, lattice_b, lattice_c = (
            vector_length(row) for row in relaxed.lattice_vectors
        )
        payload["lattice_a"] = lattice_a
        payload["lattice_b"] = lattice_b
        payload["lattice_c"] = lattice_c

        written, file_warnings = _materialise_files(
            run=run,
            job_directory=job_directory,
            seed_name=seed_name,
            species=species,
            positions=_initial_positions(run) or relaxed.fractional_positions,
            lattice_vectors=relaxed.lattice_vectors,
            parameters=run.parameters,
            charge_state=charge_state,
        )

        result = Result(
            calculation_run_id=run_record.id,
            result_type="castep_muon_relaxation",
            data=payload,
            output_files=[str(Path(job_directory.name) / name) for name in written],
            derived_structure_ids=[structure.id],
        )
        run_record.result_id = result.id

        document.structures[structure.id] = structure
        document.sites[site.id] = site
        document.jobs[job.id] = job
        document.runs[run_record.id] = run_record
        document.results[result.id] = result

        site_records.append(
            {
                "site_index": run.site_index,
                "seed_name": seed_name,
                "label": label,
                "charge_state": charge_state,
                "castep_version": run.castep_version,
                "supercell_matrix": (
                    [list(line) for line in run.supercell_matrix]
                    if run.supercell_matrix is not None
                    else None
                ),
                "parameter_sources": dict(run.parameter_sources),
                "conflicts": [conflict.describe() for conflict in run.conflicts],
                "unmapped": list(run.unmapped),
                "warnings": [*run.warnings, *file_warnings],
                "files": list(written),
            }
        )
        imported.append(
            ImportedRelaxation(
                site_id=site.id,
                job_id=job.id,
                run_id=run_record.id,
                result_id=result.id,
                structure_id=structure.id,
                label=label,
                seed_name=seed_name,
                charge_state=charge_state,
                job_directory=job_directory,
                row_payload=_row_payload(
                    site_id=site.id,
                    label=label,
                    run=run,
                    charge_state=charge_state,
                    batch_id=batch,
                ),
                warnings=tuple([*run.warnings, *file_warnings]),
            )
        )

    record = ProvenanceRecord(
        action=IMPORT_ACTION,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        actor="user",
        description=(
            f"Imported {len(imported)} legacy muon relaxation(s) from {scan.root.name}"
        ),
        parameters={
            "batch_id": str(batch),
            "source_folder": str(scan.root),
            "keyword_catalog_origin": scan.catalog_origin,
            "imported_sites": site_records,
            "skipped": [
                {"site_index": index, "reason": reason} for index, reason in skipped
            ],
        },
        input_entity_ids=[host_structure.id],
        output_entity_ids=[item.site_id for item in imported],
    )

    return LegacyImportOutcome(
        batch_id=batch,
        imported=tuple(imported),
        skipped=tuple(skipped),
        warnings=tuple(warnings),
        provenance=record,
    )


def _initial_positions(run: DetectedRun) -> list[list[float]]:
    """Return the run's initial positions, read back from its input cell.

    The rendered ``.cell`` must hold the structure the run *started* from, so
    initial-versus-relaxed comparison stays truthful. Returns an empty list when
    the input cell cannot be re-read, so the caller can fall back to the relaxed
    geometry rather than emitting a cell with no atoms.
    """

    from mufinder.castep_io.cell_parser import parse_cell_file

    if run.input_cell_path is None:
        return []
    try:
        parsed = parse_cell_file(run.input_cell_path)
    except Exception:  # noqa: BLE001
        return []
    return [[float(value) for value in row] for row in parsed.fractional_positions]


__all__ = [
    "IMPORT_ACTION",
    "LEGACY_MANIFEST_NAME",
    "LEGACY_SOURCE_DIRNAME",
    "ImportedRelaxation",
    "LegacyImportOutcome",
    "commit_legacy_import",
    "resolve_recovered_parameters",
    "row_parameter_values",
]
