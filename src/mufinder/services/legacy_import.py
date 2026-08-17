"""Scan a legacy MuFinder working directory for importable muon relaxations.

The legacy application kept no project file: a run is described entirely by
filesystem convention, and the site index survives only in a filename. This
module turns such a directory into a :class:`LegacyFolderScan` — a report of
what is there, what each run used, and what would stop each one being imported.

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
from dataclasses import dataclass, field
from pathlib import Path
import re

from mufinder.castep_io.cell_parser import ParsedCell, parse_cell_file
from mufinder.castep_io.output_parser import CastepResult, parse_castep_output
from mufinder.castep_io.param_parser import parse_param_file
from mufinder.castep_io.run_parameters import CastepRunParameters, parse_run_parameters
from mufinder.castep_io.schema import keyword_catalog_source_info
from mufinder.domain.charge_states import MUONIUM_CHARGE_STATE, POSITIVE_MUON_CHARGE_STATE
from mufinder.science.supercell_matching import (
    SupercellDetection,
    detect_supercell_matrix_from_lattices,
)

#: The species label MuFinder requires on an imported relaxed structure. The
#: quantum-corrections eligibility gate hard-requires it.
MUON_SPECIES = "H:mu"
#: Labels the legacy application accepted for the muon. A bare ``H`` is
#: relabelled on import, and the relabelling is recorded.
LEGACY_MUON_LABELS = (MUON_SPECIES, "H")

#: Legacy parsed the site index by splitting the stem on ``_``, space and ``.``
#: and taking the second-to-last field, i.e. the trailing ``_<N>``.
_SITE_INDEX_RE = re.compile(r"^(?P<stem>.+)_(?P<index>\d+)$")

#: Where a value in a merged parameter set came from, most authoritative first.
SOURCE_CASTEP = "castep"
SOURCE_PARAM = "param"
SOURCE_CELL = "cell"
_SOURCE_PRIORITY = (SOURCE_CASTEP, SOURCE_PARAM, SOURCE_CELL)

#: How close the inferred transform must be to an exact integer matrix.
#:
#: Much tighter than the default used for *structure comparison*, where an
#: experimental and a DFT lattice legitimately differ by a percent or so. A
#: legacy input cell was constructed as an exact integer supercell of its host,
#: so the only expected residual is the six-decimal rounding a ``.cell`` writer
#: applies — of order 1e-6. Leaving comparison's 0.2 in place here would accept
#: a 7% lattice mismatch as a supercell and import a run against the wrong host.
IMPORT_SUPERCELL_TOLERANCE = 1e-3


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
    """One legacy relaxation found in a scanned directory."""

    site_index: int
    seed_name: str
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

    supercell_matrix: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]] | None = None
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


@dataclass(frozen=True, slots=True)
class LegacyFolderScan:
    """Everything a scan of one legacy working directory found."""

    root: Path
    runs: tuple[DetectedRun, ...] = ()
    host_cif: Path | None = None
    #: How the runtime CASTEP keyword catalog was located, for provenance. A
    #: ``bundled_starter_catalog`` origin means parameter recovery is degraded
    #: but — deliberately — never blocks an import.
    catalog_origin: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def importable_runs(self) -> tuple[DetectedRun, ...]:
        """Return only the runs with no blockers."""

        return tuple(run for run in self.runs if run.importable)

    @property
    def converged_runs(self) -> tuple[DetectedRun, ...]:
        """Return the runs the wizard should pre-select."""

        return tuple(run for run in self.importable_runs if run.is_converged)


def site_index_from_stem(stem: str) -> int | None:
    """Return the legacy site index encoded in a seed name, or ``None``.

    The index is the trailing ``_<N>`` of the stem, matching how the legacy
    application recovered it.
    """

    match = _SITE_INDEX_RE.match(stem.strip())
    if match is None:
        return None
    return int(match.group("index"))


def muon_index(parsed: ParsedCell) -> tuple[int, str] | None:
    """Return the index and label of the muon in a parsed cell.

    Prefers ``H:mu``; falls back to a bare ``H`` only when the cell contains
    exactly one, since a hydrogen-bearing host would otherwise be ambiguous.
    The legacy label detector accepted both spellings.
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
    consulted. A cell carrying a muon at charge 0 is muonium — the legacy
    application's other zero-charge case, its neutral pristine reference, is
    site 0 and carries no muon at all.
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
            f"The input cell has {len(species)} host atom(s) but a "
            f"{determinant}x supercell of the active host needs {expected_count}."
        )
        return blockers

    expected = Counter(str(item).strip() for item in host_species)
    for name in expected:
        expected[name] *= determinant
    if Counter(species) != expected:
        blockers.append(
            "The input cell's composition does not match the active host structure."
        )
    return blockers


def _scan_castep(path: Path) -> tuple[CastepResult | None, CastepRunParameters | None, list[str]]:
    warnings: list[str] = []
    result: CastepResult | None = None
    run_parameters: CastepRunParameters | None = None
    try:
        result = parse_castep_output(path)
    except Exception as exc:  # noqa: BLE001 - a bad file must not end the scan
        warnings.append(f"Could not read {path.name}: {exc}")
    try:
        run_parameters = parse_run_parameters(path)
    except Exception as exc:  # noqa: BLE001 - a bad file must not end the scan
        warnings.append(f"Could not read the parameter echo in {path.name}: {exc}")
    return result, run_parameters, warnings


def _detect_run(
    *,
    seed_name: str,
    site_index: int,
    input_cell_path: Path | None,
    castep_path: Path | None,
    out_cell_path: Path | None,
    run_param_path: Path | None,
    host_lattice: list[list[float]] | None,
    host_species: Sequence[str] | None,
    supercell_tolerance: float,
) -> DetectedRun:
    warnings: list[str] = []
    blockers: list[str] = []

    result: CastepResult | None = None
    run_parameters: CastepRunParameters | None = None
    if castep_path is not None:
        result, run_parameters, castep_warnings = _scan_castep(castep_path)
        warnings.extend(castep_warnings)
    else:
        blockers.append("No .castep output was found for this site.")

    input_cell: ParsedCell | None = None
    if input_cell_path is not None:
        try:
            input_cell = parse_cell_file(input_cell_path)
        # Broad by design: this module promises never to raise for one bad run,
        # and a parser reading third-party files can fail in ways its own error
        # type does not cover (a species label ASE rejects, a decode error).
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not read {input_cell_path.name}: {exc}")
            # Blocking, not merely a warning: the initial muon position and the
            # supercell matrix are both recovered from this file, so without it
            # there is nothing to anchor the imported site to.
            blockers.append("The input .cell could not be parsed.")
    else:
        blockers.append("No input .cell was found for this site.")

    out_cell: ParsedCell | None = None
    if out_cell_path is not None:
        try:
            out_cell = parse_cell_file(out_cell_path)
        except Exception as exc:  # noqa: BLE001 - see the input-cell note above
            warnings.append(f"Could not read {out_cell_path.name}: {exc}")
            blockers.append("The relaxed -out.cell could not be parsed.")
    else:
        blockers.append("No relaxed -out.cell was found for this site.")

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
            warnings.append(f"Could not read {run_param_path.name}: {exc}")

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

    if input_cell is not None:
        found = muon_index(input_cell)
        if found is None:
            blockers.append("The input cell contains no muon.")
        else:
            index, muon_label = found
            initial_position = _fractional(input_cell, index)

    if out_cell is not None:
        found = muon_index(out_cell)
        if found is None:
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
    if charge_state is None:
        warnings.append(
            "The muon charge state could not be determined from the .castep or .param; "
            "choose one before importing."
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

    return DetectedRun(
        site_index=site_index,
        seed_name=seed_name,
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
        supercell_matrix=detection.matrix if detection is not None and detection.is_supercell else None,
        supercell_residual=detection.residual if detection is not None else None,
        initial_muon_fractional_supercell=initial_position,
        final_muon_fractional_supercell=final_position,
        parameters=parameters,
        parameter_sources=parameter_sources,
        conflicts=conflicts,
        unmapped=tuple(unmapped),
        castep_version=run_parameters.castep_version if run_parameters is not None else None,
        warnings=tuple(warnings),
        blockers=tuple(blockers),
    )


def scan_legacy_folder(
    root: Path | str,
    *,
    host_lattice: list[list[float]] | None = None,
    host_species: Sequence[str] | None = None,
    supercell_tolerance: float = IMPORT_SUPERCELL_TOLERANCE,
) -> LegacyFolderScan:
    """Report the muon relaxations in a legacy working directory.

    ``root`` is the legacy working directory itself — the one holding ``input``,
    ``run`` and ``results``. Pass ``host_lattice`` to have each run's supercell
    matrix inferred and checked against the project's active host; omit it to
    scan without that check (the runs then carry no supercell matrix).

    Never raises for a malformed run: each one carries its own warnings and
    blockers, and :attr:`LegacyFolderScan.importable_runs` filters them.
    """

    root_path = Path(root)
    warnings: list[str] = []

    input_dir = root_path / "input"
    results_dir = root_path / "results"
    run_dir = root_path / "run"

    if not input_dir.is_dir():
        warnings.append("No 'input' directory was found; there are no candidate cells to import.")
    if not results_dir.is_dir():
        warnings.append("No 'results' directory was found; there are no completed runs to import.")

    catalog = keyword_catalog_source_info()
    if catalog is not None and catalog.origin == "bundled_starter_catalog":
        warnings.append(
            "No CASTEP installation was found, so parameter recovery uses the bundled "
            "starter catalog and will be less complete. Set MUFINDER_CASTEP_KEYWORDS_JSON "
            "to improve it. This does not affect the relaxed structures themselves."
        )

    by_index: dict[int, dict[str, Path]] = {}
    stems: dict[int, str] = {}

    if input_dir.is_dir():
        for path in sorted(input_dir.glob("*.cell")):
            index = site_index_from_stem(path.stem)
            if index is None:
                continue
            by_index.setdefault(index, {})["input_cell"] = path
            stems[index] = path.stem

    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.castep")):
            index = site_index_from_stem(path.stem)
            if index is None:
                continue
            by_index.setdefault(index, {})["castep"] = path
            stems.setdefault(index, path.stem)
        for path in sorted(results_dir.glob("*-out.cell")):
            stem = path.name[: -len("-out.cell")]
            index = site_index_from_stem(stem)
            if index is None:
                continue
            by_index.setdefault(index, {})["out_cell"] = path
            stems.setdefault(index, stem)

    if run_dir.is_dir():
        for path in sorted(run_dir.glob("site*/*.param")):
            index = site_index_from_stem(path.stem)
            if index is None:
                continue
            by_index.setdefault(index, {})["run_param"] = path

    runs: list[DetectedRun] = []
    for index in sorted(by_index):
        # Site 0 is the legacy pristine/neutral reference: it carries no muon
        # and is not a stopping site, so it is never a relaxation to import.
        if index == 0:
            continue
        paths = by_index[index]
        runs.append(
            _detect_run(
                seed_name=stems.get(index, f"site{index}"),
                site_index=index,
                input_cell_path=paths.get("input_cell"),
                castep_path=paths.get("castep"),
                out_cell_path=paths.get("out_cell"),
                run_param_path=paths.get("run_param"),
                host_lattice=host_lattice,
                host_species=host_species,
                supercell_tolerance=supercell_tolerance,
            )
        )

    host_cif = next(iter(sorted(root_path.glob("*.cif"))), None) if root_path.is_dir() else None
    if host_cif is None:
        warnings.append(
            "No CIF was found in the folder, so a host structure cannot be adopted from it. "
            "The project must already have a matching host structure."
        )

    return LegacyFolderScan(
        root=root_path,
        runs=tuple(runs),
        host_cif=host_cif,
        catalog_origin=catalog.origin if catalog is not None else None,
        warnings=tuple(warnings),
    )


__all__ = [
    "IMPORT_SUPERCELL_TOLERANCE",
    "LEGACY_MUON_LABELS",
    "MUON_SPECIES",
    "DetectedRun",
    "LegacyFolderScan",
    "ParameterConflict",
    "merge_parameters",
    "muon_index",
    "scan_legacy_folder",
    "site_index_from_stem",
]
