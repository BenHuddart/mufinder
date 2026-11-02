"""Scan a folder for CASTEP geometry optimisations with a muon in them.

A run outside MuFinder has no project record: it is described entirely by the
files CASTEP read and wrote. This module turns a folder of them, nested however
the user kept it, into a :class:`RelaxationFolderScan` — a report of what is
there, what each run used, and what would stop each one being imported.

A run is a ``<seed>.castep``. Its ``<seed>.cell``, ``<seed>-out.cell`` and
``<seed>.param`` are looked for beside it and then elsewhere under the folder;
:func:`_nearest` holds the rule.

It is deliberately **pure with respect to the project**: nothing here mutates a
document, writes a file, or creates an entity. That belongs to a later stage.
Keeping the scan side-effect free means the import wizard can show a user
exactly what it found and let them choose, and means the whole detection path is
testable without a project on disk.

Layering: ``services`` may read from ``castep_io``, ``science`` and ``domain``.
It must not import ``app`` or ``workflow``.

**Nothing raises for one bad run.** A directory of third-party output will
contain surprises, and the honest outcome is a report that says which runs are
importable and why the others are not — never an exception that discards the
whole scan.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
import hashlib
import os
from pathlib import Path
import re
import stat

from mufinder.castep_io.cell_parser import ParsedCell, parse_cell_file
from mufinder.castep_io.output_parser import CastepResult, parse_castep_output
from mufinder.castep_io.param_parser import parse_param_file
from mufinder.castep_io.run_parameters import CastepRunParameters, parse_run_parameters
from mufinder.castep_io.schema import keyword_catalog_source_info
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.domain.text import plain_error, plural
from mufinder.science.supercell_matching import (
    SupercellDetection,
    detect_supercell_matrix_from_lattices,
)

#: The species label MuFinder requires on an imported relaxed structure. The
#: quantum-corrections eligibility gate hard-requires it.
MUON_SPECIES = "H:mu"
#: Why a run with no muon is not offered: a host-only run is not a stopping site.
NO_MUON_BLOCKER = "No muon in this structure."

#: The output a task farm writes for one of its images, beside the run's own ``.castep``.
_FARM_OUTPUT_RE = re.compile(r"_farm\d+$")

#: The kinds of file a run is paired with.
_CELL = "cell"
_OUT_CELL = "out_cell"
_PARAM = "param"

_CASTEP_SUFFIX = ".castep"
_OUT_CELL_SUFFIX = "-out.cell"
_CELL_SUFFIX = ".cell"
_PARAM_SUFFIX = ".param"

#: The ``task`` value of a geometry optimisation, as the readers fold it.
_GEOMETRY_OPTIMISATION_TASK = "GEOMETRYOPTIMIZATION"

#: Where a value in a merged parameter set came from, most authoritative first.
SOURCE_CASTEP = "castep"
SOURCE_PARAM = "param"
SOURCE_CELL = "cell"
_SOURCE_PRIORITY = (SOURCE_CASTEP, SOURCE_PARAM, SOURCE_CELL)

#: How close the inferred transform must be to an exact integer matrix.
#:
#: Much tighter than the default used for *structure comparison*, where an
#: experimental and a DFT lattice legitimately differ by a percent or so. A
#: muon relaxation's input cell is constructed as an exact integer supercell of
#: its host, so the only expected residual is the six-decimal rounding a ``.cell``
#: writer applies — of order 1e-6. Leaving comparison's 0.2 in place here would accept
#: a 7% lattice mismatch as a supercell and import a run against the wrong host.
IMPORT_SUPERCELL_TOLERANCE = 1e-3


#: How many unreadable locations a note names before it counts the rest.
_UNREADABLE_SHOWN = 3
#: A path in a note is cut to this many characters, keeping both ends.
_NOTE_PATH_LENGTH = 60


def _shortened(path: str) -> str:
    if len(path) <= _NOTE_PATH_LENGTH:
        return path
    half = (_NOTE_PATH_LENGTH - 1) // 2
    return f"{path[:half]}…{path[-half:]}"


@dataclass(frozen=True, slots=True)
class ParameterConflict:
    """One keyword whose sources disagree about the value that was used."""

    keyword: str
    #: The winning value and where it came from.
    value: object
    source: str
    #: The value that lost, and its source.
    other_value: object
    other_source: str

    def describe(self) -> str:
        """Return a one-line, user-facing description of the disagreement."""

        return (
            f"{self.keyword.upper()}: the {self.source} says {self.value!r} but the "
            f"{self.other_source} says {self.other_value!r}; importing {self.value!r}."
        )


@dataclass(frozen=True, slots=True)
class DetectedRun:
    """One CASTEP run found in a scanned directory."""

    seed_name: str
    #: The ``.castep`` under the scanned folder, without its suffix. What tells
    #: two runs of one seed apart.
    relative_name: str = ""
    input_cell_path: Path | None = None
    castep_path: Path | None = None
    out_cell_path: Path | None = None
    run_param_path: Path | None = None

    charge_state: str | None = None
    charge_state_source: str | None = None
    muon_label: str | None = None

    completed: bool = False
    converged: bool | None = None
    total_energy: float | None = None

    supercell_matrix: (
        tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None
    ) = None
    supercell_residual: float | None = None
    initial_muon_fractional_supercell: tuple[float, float, float] | None = None
    final_muon_fractional_supercell: tuple[float, float, float] | None = None

    #: Merged effective parameters, ``.castep`` winning over ``.param`` over ``.cell``.
    parameters: dict[str, object] = field(default_factory=dict)
    #: Which file each merged value came from.
    parameter_sources: dict[str, str] = field(default_factory=dict)
    conflicts: tuple[ParameterConflict, ...] = ()
    #: Labels and keywords no translation table or schema recognised.
    unmapped: tuple[str, ...] = ()

    castep_version: str | None = None
    warnings: tuple[str, ...] = ()
    #: For the review only: which of the run's files came from another folder.
    pairing_notes: tuple[str, ...] = ()
    #: Reasons this run cannot be imported. Empty means importable.
    blockers: tuple[str, ...] = ()

    @property
    def importable(self) -> bool:
        """Return whether the run can be imported as a completed relaxation."""

        return not self.blockers

    @property
    def is_converged(self) -> bool:
        """Return whether the geometry optimisation reported success."""

        return self.converged is True

    @property
    def display_name(self) -> str:
        """Return what tells this run from the others in a scan."""

        return self.relative_name or self.seed_name

    @property
    def has_no_muon(self) -> bool:
        """Return whether the run was refused for holding no muon."""

        return self.blockers == (NO_MUON_BLOCKER,)


@dataclass(frozen=True, slots=True)
class RelaxationFolderScan:
    """Everything a scan of one folder found."""

    root: Path
    runs: tuple[DetectedRun, ...] = ()
    #: How the runtime CASTEP keyword catalog was located, for provenance. A
    #: ``bundled_starter_catalog`` origin means parameter recovery is degraded
    #: but — deliberately — never blocks an import.
    catalog_origin: str | None = None
    warnings: tuple[str, ...] = ()
    #: ``(path under the folder, why)`` for what the search could not look inside,
    #: ``"."`` being the folder itself. Runs may be there, unlisted.
    unreadable: tuple[tuple[str, str], ...] = ()

    @property
    def unreadable_note(self) -> str:
        """Return what the search could not look inside, in a sentence, or ``""``."""

        if not self.unreadable:
            return ""
        if self.unreadable[0][0] == ".":
            return f"Could not read this folder: {self.unreadable[0][1]}."
        shown = [
            f"{_shortened(name)} ({reason})" for name, reason in self.unreadable[:_UNREADABLE_SHOWN]
        ]
        hidden = len(self.unreadable) - len(shown)
        if hidden:
            shown.append(f"and {hidden} more")
        return (
            f"Could not read {plural(len(self.unreadable), 'location')} under this folder, "
            f"so any runs inside are not listed: {', '.join(shown)}."
        )

    @property
    def importable_runs(self) -> tuple[DetectedRun, ...]:
        """Return only the runs with no blockers."""

        return tuple(run for run in self.runs if run.importable)

    @property
    def converged_runs(self) -> tuple[DetectedRun, ...]:
        """Return the runs the wizard should pre-select."""

        return tuple(run for run in self.importable_runs if run.is_converged)


def muon_index(parsed: ParsedCell) -> tuple[int, str] | None:
    """Return the index and label of the muon in a parsed cell.

    Prefers ``H:mu``; falls back to a bare ``H`` only when the cell contains
    exactly one, since a hydrogen-bearing host would otherwise be ambiguous.
    """

    species = [str(item).strip() for item in parsed.species]
    for index in range(len(species) - 1, -1, -1):
        if species[index].casefold() == MUON_SPECIES.casefold():
            return index, MUON_SPECIES

    hydrogen = [index for index, item in enumerate(species) if item.casefold() == "h"]
    if len(hydrogen) == 1:
        return hydrogen[0], "H"
    return None


def _fractional(parsed: ParsedCell, index: int) -> tuple[float, float, float] | None:
    try:
        position = parsed.fractional_positions[index]
    except IndexError:
        return None
    return (float(position[0]), float(position[1]), float(position[2]))


def _charge_state_from_charge(charge: object) -> str | None:
    """Map a CASTEP total charge onto a muon charge state.

    Charge alone decides this; spin polarisation is orthogonal and must not be
    consulted. A cell carrying a muon at charge 0 is muonium; a neutral run of
    the host alone carries no muon and never reaches this question.
    """

    if charge is None:
        return None
    try:
        value = float(charge)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if abs(value - 1.0) < 1e-6:
        return POSITIVE_MUON_CHARGE_STATE
    if abs(value) < 1e-6:
        return MUONIUM_CHARGE_STATE
    return None


def merge_parameters(
    *,
    castep_values: dict[str, object] | None = None,
    param_values: dict[str, object] | None = None,
    cell_values: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, str], tuple[ParameterConflict, ...]]:
    """Merge parameter sources, most authoritative first, recording conflicts.

    Priority is ``.castep`` over ``.param`` over ``.cell``: the output echo is
    the only record of what actually ran, while a ``.param`` beside it may never
    have been the file that was submitted.

    A conflict is reported only when two sources both supplied a value and they
    differ; a value present in one source alone is not a conflict.
    """

    by_source: dict[str, dict[str, object]] = {
        SOURCE_CASTEP: dict(castep_values or {}),
        SOURCE_PARAM: dict(param_values or {}),
        SOURCE_CELL: dict(cell_values or {}),
    }

    merged: dict[str, object] = {}
    sources: dict[str, str] = {}
    conflicts: list[ParameterConflict] = []

    for source in _SOURCE_PRIORITY:
        for keyword, value in by_source[source].items():
            if keyword not in merged:
                merged[keyword] = value
                sources[keyword] = source
                continue
            if _values_differ(merged[keyword], value):
                conflicts.append(
                    ParameterConflict(
                        keyword=keyword,
                        value=merged[keyword],
                        source=sources[keyword],
                        other_value=value,
                        other_source=source,
                    )
                )
    return merged, sources, tuple(conflicts)


def _values_differ(left: object, right: object) -> bool:
    """Compare two parameter values, tolerating float and unit representations."""

    if left == right:
        return False
    left_number = _magnitude(left)
    right_number = _magnitude(right)
    if left_number is not None and right_number is not None:
        scale = max(abs(left_number), abs(right_number), 1.0)
        return abs(left_number - right_number) > 1e-9 * scale
    return str(left).strip().casefold() != str(right).strip().casefold()


def _magnitude(value: object) -> float | None:
    """Return the numeric magnitude of a value, unwrapping a unit if present."""

    inner = getattr(value, "value", value)
    if isinstance(inner, bool):
        return None
    if isinstance(inner, (int, float)):
        return float(inner)
    return None


def _host_composition_blockers(
    *,
    input_cell: ParsedCell,
    host_species: Sequence[str] | None,
    determinant: int,
) -> list[str]:
    """Check the input cell really is the host tiled ``determinant`` times, plus a muon.

    A matching lattice is not enough on its own. Two chemically unrelated
    materials can share a lattice closely enough to pass the integer-transform
    test, and importing against the wrong host would attach every site to a
    structure it has nothing to do with.

    Verifies the atom count (``det(S) * N_host + 1``) and that the non-muon
    species are exactly the host's composition repeated ``det(S)`` times.
    Positions are deliberately not compared: composition and count already
    exclude an unrelated cell, and a positional test would need the host tiled
    and matched atom-by-atom for no additional protection here.

    Returns an empty list when the host composition is unknown — the caller
    passed no species — so scanning without a host stays possible.
    """

    if not host_species:
        return []

    blockers: list[str] = []
    species = [str(item).strip() for item in input_cell.species]
    muon = muon_index(input_cell)
    if muon is not None:
        species.pop(muon[0])

    expected_count = determinant * len(host_species)
    if len(species) != expected_count:
        blockers.append(
            f"The input cell has {plural(len(species), 'host atom')} but a "
            f"{determinant}x supercell of the active host needs {expected_count}."
        )
        return blockers

    expected = Counter(str(item).strip() for item in host_species)
    for name in expected:
        expected[name] *= determinant
    if Counter(species) != expected:
        blockers.append("The input cell's composition does not match the active host structure.")
    return blockers


def _scan_castep(path: Path) -> tuple[CastepResult | None, CastepRunParameters | None, list[str]]:
    try:
        result = parse_castep_output(path)
    except Exception as exc:  # noqa: BLE001 - a bad file must not end the scan
        return None, None, [f"Could not read {path.name}: {plain_error(exc)}."]
    warnings: list[str] = []
    run_parameters: CastepRunParameters | None = None
    try:
        run_parameters = parse_run_parameters(path)
    except Exception as exc:  # noqa: BLE001 - a bad file must not end the scan
        warnings.append(f"Could not read the parameter echo in {path.name}: {plain_error(exc)}.")
    return result, run_parameters, warnings


def _only_the_host(
    input_cell: ParsedCell, host_species: Sequence[str] | None, determinant: int
) -> bool:
    """Return whether a cell's atoms are exactly the host tiled, with nothing added.

    What tells a hydrogen-bearing host's own bare ``H`` from a muon labelled the
    same way.
    """

    if not host_species:
        return False
    expected = Counter(str(item).strip() for item in host_species)
    for name in expected:
        expected[name] *= determinant
    return Counter(str(item).strip() for item in input_cell.species) == expected


def _same_atoms(input_cell: ParsedCell, out_cell: ParsedCell) -> bool:
    """Return whether two cells hold the same atoms, however the muon is labelled.

    Catches a relaxed cell paired from the wrong run. CASTEP keeps the atoms of a
    geometry optimisation, and may spell a label differently.
    """

    def atoms(cell: ParsedCell) -> Counter[str]:
        return Counter(
            "H" if str(item).strip().casefold() == MUON_SPECIES.casefold() else str(item).strip()
            for item in cell.species
        )

    return atoms(input_cell) == atoms(out_cell)


def _detect_run(
    *,
    seed_name: str,
    relative_name: str,
    castep_path: Path,
    files: Mapping[str, Path],
    refused: Mapping[str, str],
    host_lattice: list[list[float]] | None,
    host_species: Sequence[str] | None,
    supercell_tolerance: float,
) -> DetectedRun:
    warnings: list[str] = []
    blockers: list[str] = []
    input_cell_path = files.get(_CELL)
    out_cell_path = files.get(_OUT_CELL)
    run_param_path = files.get(_PARAM)

    result, run_parameters, castep_warnings = _scan_castep(castep_path)
    if result is None:
        blockers.extend(castep_warnings)
    else:
        warnings.extend(castep_warnings)
    if _PARAM in refused:
        warnings.append(f"{refused[_PARAM]} Its parameters are not read.")

    input_cell: ParsedCell | None = None
    if input_cell_path is not None:
        try:
            input_cell = parse_cell_file(input_cell_path)
        # Broad by design: this module promises never to raise for one bad run,
        # and a parser reading third-party files can fail in ways its own error
        # type does not cover (a species label ASE rejects, a decode error).
        except Exception as exc:  # noqa: BLE001
            # A keyword outside the loaded catalog costs only the settings read
            # from the cell (its k-point grid); the geometry still anchors the run.
            try:
                input_cell = parse_cell_file(input_cell_path, geometry_only=True)
            except Exception:  # noqa: BLE001
                warnings.append(f"Could not read {input_cell_path.name}: {plain_error(exc)}.")
                # Blocking, not merely a warning: the initial muon position and
                # the supercell matrix are both recovered from this file, so
                # without it there is nothing to anchor the imported site to.
                blockers.append("The input .cell could not be parsed.")
            else:
                warnings.append(
                    f"Read only the geometry of {input_cell_path.name}; a k-point grid "
                    f"it states that the .castep does not report is not imported: {plain_error(exc)}."
                )
    else:
        blockers.append(refused.get(_CELL, f"No {seed_name}.cell was found for this run."))

    out_cell: ParsedCell | None = None
    if out_cell_path is not None:
        try:
            # Geometry only: the run's settings echoed in an -out.cell may lie
            # outside the loaded keyword catalog.
            out_cell = parse_cell_file(out_cell_path, geometry_only=True)
        except Exception as exc:  # noqa: BLE001 - see the input-cell note above
            warnings.append(f"Could not read {out_cell_path.name}: {plain_error(exc)}.")
            blockers.append("The relaxed -out.cell could not be parsed.")
    else:
        blockers.append(
            refused.get(_OUT_CELL, f"No relaxed {seed_name}-out.cell was found for this run.")
        )

    param_values: dict[str, object] = {}
    if run_param_path is not None:
        try:
            parsed_param = parse_param_file(run_param_path, strict=False)
            param_values = {
                item.canonical_name: item.value
                for item in parsed_param.observations
                if item.canonical_name and item.error is None
            }
        except Exception as exc:  # noqa: BLE001 - see the input-cell note above
            warnings.append(f"Could not read {run_param_path.name}: {plain_error(exc)}.")

    cell_values: dict[str, object] = {}
    if input_cell is not None and input_cell.cell_file.kpoint_mp_grid is not None:
        cell_values["kpoint_mp_grid"] = tuple(
            int(component) for component in input_cell.cell_file.kpoint_mp_grid
        )

    castep_values = run_parameters.mapped_values() if run_parameters is not None else {}
    parameters, parameter_sources, conflicts = merge_parameters(
        castep_values=castep_values,
        param_values=param_values,
        cell_values=cell_values,
    )

    unmapped: list[str] = []
    if run_parameters is not None:
        unmapped.extend(sorted({item.label for item in run_parameters.unmapped()}))

    # Muon identification, from the relaxed cell where possible: that is the
    # structure the import will carry forward.
    muon_label: str | None = None
    initial_position: tuple[float, float, float] | None = None
    final_position: tuple[float, float, float] | None = None

    # The input decides whether a run has a muon at all; a relaxed cell without
    # one, from a run that started with one, is a different fault.
    no_muon = False
    if input_cell is not None:
        found = muon_index(input_cell)
        if found is None:
            no_muon = True
        else:
            index, muon_label = found
            initial_position = _fractional(input_cell, index)

    if out_cell is not None:
        found = muon_index(out_cell)
        if found is None:
            if input_cell is None:
                no_muon = True
            else:
                blockers.append("The relaxed cell contains no muon.")
        else:
            index, relaxed_label = found
            final_position = _fractional(out_cell, index)
            muon_label = muon_label or relaxed_label
            if relaxed_label != MUON_SPECIES:
                warnings.append(
                    f"The muon is labelled {relaxed_label!r} rather than {MUON_SPECIES!r}; "
                    "it will be relabelled on import."
                )

    charge_state: str | None = None
    charge_state_source: str | None = None
    castep_charge = castep_values.get("charge")
    charge_state = _charge_state_from_charge(castep_charge)
    if charge_state is not None:
        charge_state_source = SOURCE_CASTEP
    else:
        charge_state = _charge_state_from_charge(param_values.get("charge"))
        if charge_state is not None:
            charge_state_source = SOURCE_PARAM
    if charge_state is None and result is not None:
        warnings.append(
            "The muon charge state could not be determined from the .castep or .param; "
            "it will be imported as a positive muon."
        )

    detection: SupercellDetection | None = None
    if host_lattice is not None and input_cell is not None:
        detection = detect_supercell_matrix_from_lattices(
            host_lattice,
            [list(vector) for vector in input_cell.lattice_vectors],
            tolerance=supercell_tolerance,
        )
        if not detection.is_supercell:
            blockers.append(
                "The input cell is not an integer supercell of the active host structure."
            )
        elif _only_the_host(input_cell, host_species, detection.determinant):
            no_muon = True
        else:
            blockers.extend(
                _host_composition_blockers(
                    input_cell=input_cell,
                    host_species=host_species,
                    determinant=detection.determinant,
                )
            )

    completed = bool(result.completed) if result is not None else False
    converged = result.geom_converged if result is not None else None
    if result is not None and not completed:
        blockers.append("The CASTEP run did not complete.")
    if input_cell is not None and out_cell is not None and not _same_atoms(input_cell, out_cell):
        blockers.append(
            f"{out_cell_path.name} does not hold the same atoms as {input_cell_path.name}."
        )

    # One reason, not every consequence of it: a phonon run or a run of the host
    # alone also lacks a relaxed cell or a charge state, and neither matters.
    task = str(parameters.get("task") or "").strip().upper()
    if no_muon:
        blockers, warnings, conflicts = [NO_MUON_BLOCKER], [], ()
        # Charge says what a muon is, and here there is none to describe.
        charge_state = charge_state_source = muon_label = None
    elif task and task != _GEOMETRY_OPTIMISATION_TASK:
        blockers = [f"This is a {task} run, not a geometry optimisation."]
        warnings, conflicts = [], ()

    elsewhere = [path for path in files.values() if path.parent != castep_path.parent]
    pairing_notes = (
        ("Read from another folder: " + ", ".join(path.name for path in elsewhere) + ".",)
        if elsewhere and not blockers
        else ()
    )

    return DetectedRun(
        seed_name=seed_name,
        relative_name=relative_name,
        input_cell_path=input_cell_path,
        castep_path=castep_path,
        out_cell_path=out_cell_path,
        run_param_path=run_param_path,
        charge_state=charge_state,
        charge_state_source=charge_state_source,
        muon_label=muon_label,
        completed=completed,
        converged=converged,
        total_energy=result.total_energy if result is not None else None,
        supercell_matrix=detection.matrix
        if detection is not None and detection.is_supercell
        else None,
        supercell_residual=detection.residual if detection is not None else None,
        initial_muon_fractional_supercell=initial_position,
        final_muon_fractional_supercell=final_position,
        parameters=parameters,
        parameter_sources=parameter_sources,
        conflicts=conflicts,
        unmapped=tuple(unmapped),
        castep_version=run_parameters.castep_version if run_parameters is not None else None,
        warnings=tuple(warnings),
        pairing_notes=pairing_notes,
        blockers=tuple(blockers),
    )


def _tree_distance(left: Path, right: Path) -> int:
    """Return how many directories apart two directories are."""

    left_parts, right_parts = left.parts, right.parts
    shared = 0
    for left_part, right_part in zip(left_parts, right_parts, strict=False):
        if left_part != right_part:
            break
        shared += 1
    return len(left_parts) + len(right_parts) - 2 * shared


def file_digest(path: Path) -> str:
    """Return a file's SHA-256 digest, read in chunks. Raises ``OSError`` if unreadable."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_or_none(path: Path) -> str | None:
    try:
        return file_digest(path)
    except OSError:
        return None


def _nearest(castep: Path, candidates: Sequence[Path]) -> tuple[Path | None, str | None]:
    """Return the candidate closest to ``castep`` in the folder tree.

    Beside the ``.castep`` is distance zero, so a run kept whole always finds its
    own files, while a tree that separates inputs from results still pairs them.
    Candidates equally close are accepted only when they are copies of one file:
    taking one of two different files would attach the wrong starting structure
    to the run. The second value says so when that is why nothing was returned.
    """

    if not candidates:
        return None, None
    distances = [_tree_distance(castep.parent, path.parent) for path in candidates]
    nearest = min(distances)
    closest = [path for path, value in zip(candidates, distances, strict=True) if value == nearest]
    if len(closest) > 1:
        digests = {_digest_or_none(path) for path in closest}
        if len(digests) > 1 or None in digests:
            return None, (
                f"Several different {closest[0].name} files are equally close to this run, "
                "so MuFinder cannot tell which one it used."
            )
    return closest[0], None


def _without_copies(
    outputs: Sequence[Path], found: Mapping[str, Mapping[str, Sequence[Path]]]
) -> list[Path]:
    """Keep one of each set of ``.castep`` files of the same name and bytes.

    A copied results folder holds the same calculation, not a second one. The copy
    kept is the one with the most of the run's files beside it, so that a stray copy
    sorting first cannot pull the run away from its own ``.cell`` and ``-out.cell``.
    """

    groups: dict[tuple[str, str], list[Path]] = {}
    sizes: dict[Path, int] = {}
    for path in outputs:
        try:
            sizes[path] = path.stat().st_size
        except OSError:  # Gone since discovery; reading it reports why.
            continue
    size_counts = Counter((path.name, size) for path, size in sizes.items())
    for path, size in sizes.items():
        digest = _digest_or_none(path) if size_counts[(path.name, size)] > 1 else None
        if digest is not None:
            groups.setdefault((path.name, digest), []).append(path)

    def companions(path: Path) -> int:
        seed = path.name[: -len(_CASTEP_SUFFIX)]
        return sum(
            other.parent == path.parent
            for by_seed in found.values()
            for other in by_seed.get(seed, ())
        )

    dropped: set[Path] = set()
    for copies in groups.values():
        kept = max(copies, key=companions)
        dropped.update(path for path in copies if path != kept)
    return [path for path in outputs if path not in dropped]


@dataclass(frozen=True, slots=True)
class _RunFiles:
    """The files of one run, as discovery paired them."""

    seed_name: str
    relative_name: str
    castep: Path
    #: The other files by kind; absent when none could be taken.
    files: dict[str, Path]
    #: Why a file of that kind could not be taken although some exist.
    refused: dict[str, str]


def _discover_runs(root: Path) -> tuple[list[_RunFiles], tuple[tuple[str, str], ...]]:
    """Find every ``.castep`` under ``root`` and pair each with its other files.

    Also returns what the search could not look inside: ``os.walk`` otherwise
    passes over a locked folder, or one past the path limit, in silence.
    """

    unreadable: list[tuple[str, str]] = []

    def note_unreadable(error: OSError) -> None:
        location = os.path.relpath(error.filename or root, root)
        unreadable.append((location, plain_error(error)))

    outputs: list[Path] = []
    found: dict[str, dict[str, list[Path]]] = {_CELL: {}, _OUT_CELL: {}, _PARAM: {}}
    for directory, subdirectories, names in os.walk(root, onerror=note_unreadable):
        subdirectories.sort()
        for name in sorted(names):
            path = Path(directory) / name
            if name.endswith(_CASTEP_SUFFIX):
                seed = name[: -len(_CASTEP_SUFFIX)]
                if _FARM_OUTPUT_RE.search(seed):
                    continue
                try:
                    is_file = stat.S_ISREG(path.stat().st_mode)
                except FileNotFoundError:  # A dangling link holds no run.
                    continue
                except OSError as exc:
                    unreadable.append((os.path.relpath(path, root), plain_error(exc)))
                    continue
                if is_file:
                    outputs.append(path)
            elif name.endswith(_CELL_SUFFIX):
                # ``relax-out.cell`` may be the input of a seed named ``relax-out``,
                # so it is indexed both ways; only seeds with a ``.castep`` look it up.
                found[_CELL].setdefault(name[: -len(_CELL_SUFFIX)], []).append(path)
                if name.endswith(_OUT_CELL_SUFFIX):
                    found[_OUT_CELL].setdefault(name[: -len(_OUT_CELL_SUFFIX)], []).append(path)
            elif name.endswith(_PARAM_SUFFIX):
                found[_PARAM].setdefault(name[: -len(_PARAM_SUFFIX)], []).append(path)

    outputs = _without_copies(outputs, found)
    run_directories: dict[str, set[Path]] = {}
    for path in outputs:
        run_directories.setdefault(path.name[: -len(_CASTEP_SUFFIX)], set()).add(path.parent)

    runs: list[_RunFiles] = []
    for castep in outputs:
        seed = castep.name[: -len(_CASTEP_SUFFIX)]
        # A file beside another run of this seed belongs to that run, however
        # close it looks: ``site1/mu.castep`` must not borrow ``site2/mu.cell``.
        others = run_directories[seed] - {castep.parent}
        files: dict[str, Path] = {}
        refused: dict[str, str] = {}
        for kind, by_seed in found.items():
            candidates = [path for path in by_seed.get(seed, []) if path.parent not in others]
            chosen, reason = _nearest(castep, candidates)
            if chosen is not None:
                files[kind] = chosen
            elif reason is not None:
                refused[kind] = reason
        runs.append(
            _RunFiles(
                seed_name=seed,
                relative_name=castep.relative_to(root).with_suffix("").as_posix(),
                castep=castep,
                files=files,
                refused=refused,
            )
        )

    # One file cannot be the input of two calculations: neither run gets it.
    uses = Counter(path for run in runs for path in run.files.values())
    for run in runs:
        for kind, path in list(run.files.items()):
            if uses[path] > 1:
                del run.files[kind]
                run.refused[kind] = (
                    f"{path.name} could belong to several runs under this folder, "
                    "so MuFinder cannot tell which one it belongs to."
                )
    return runs, tuple(unreadable)


def scan_relaxation_folder(
    root: Path | str,
    *,
    host_lattice: list[list[float]] | None = None,
    host_species: Sequence[str] | None = None,
    supercell_tolerance: float = IMPORT_SUPERCELL_TOLERANCE,
    progress: Callable[[int, int], None] | None = None,
) -> RelaxationFolderScan:
    """Report the CASTEP runs in or under ``root``.

    Pass ``host_lattice`` to have each run's supercell matrix inferred and checked
    against the project's active host; omit it to scan without that check (the
    runs then carry no supercell matrix).

    Never raises for a malformed run: each one carries its own warnings and
    blockers, and :attr:`RelaxationFolderScan.importable_runs` filters them.
    """

    root_path = Path(root)
    warnings: list[str] = []

    catalog = keyword_catalog_source_info()
    if catalog is not None and catalog.origin == "bundled_starter_catalog":
        warnings.append(
            "No CASTEP keyword catalog is loaded, so parameter recovery uses the bundled "
            "starter catalog and will be less complete. Generate or import a catalog in "
            "Setup > CASTEP to improve it. This does not affect the relaxed structures themselves."
        )

    discovered, unreadable = _discover_runs(root_path)
    runs: list[DetectedRun] = []
    for position, files in enumerate(discovered, start=1):
        if progress is not None:
            progress(position, len(discovered))
        runs.append(
            _detect_run(
                seed_name=files.seed_name,
                relative_name=files.relative_name,
                castep_path=files.castep,
                files=files.files,
                refused=files.refused,
                host_lattice=host_lattice,
                host_species=host_species,
                supercell_tolerance=supercell_tolerance,
            )
        )

    scan = RelaxationFolderScan(
        root=root_path,
        runs=tuple(runs),
        catalog_origin=catalog.origin if catalog is not None else None,
        unreadable=unreadable,
    )
    if unreadable:
        warnings.append(scan.unreadable_note)
    return replace(scan, warnings=tuple(warnings))


__all__ = [
    "IMPORT_SUPERCELL_TOLERANCE",
    "MUON_SPECIES",
    "NO_MUON_BLOCKER",
    "DetectedRun",
    "RelaxationFolderScan",
    "file_digest",
    "ParameterConflict",
    "merge_parameters",
    "muon_index",
    "scan_relaxation_folder",
]
