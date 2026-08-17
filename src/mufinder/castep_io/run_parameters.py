"""Recover the settings a CASTEP run actually used from its ``.castep`` output.

A ``.castep`` file echoes the resolved run parameters near the top of its
output, before the first SCF cycle. That echo is the **only** authoritative
record of what a calculation ran with: a ``.param`` file sitting beside it may
have been edited afterwards, may never have been the file that was submitted
(runs performed on a cluster frequently leave a stale local copy behind), and
in any case never records values CASTEP derived rather than read — most
importantly the plane-wave cutoff implied by ``BASIS_PRECISION`` and the
Monkhorst-Pack grid implied by ``KPOINT_MP_SPACING``.

The echo is **not keyword-named**. Its lines are descriptive English::

     type of calculation                            : geometry optimization
     using functional                               : Perdew Burke Ernzerhof
     plane wave basis set cut-off                   :   900.0000   eV

so the substance of this module is :data:`_LABEL_RULES` and
:data:`_STATEMENT_RULES` — tables translating those labels back to MuFinder
keywords. Every entry is transcribed from the CASTEP source that writes the
line (``Source/Fundamental/parameters.f90`` and ``cell.f90``), never inferred
from the shape of a label: the project rule forbids inventing CASTEP keywords
*or* values, so a label absent from the tables is reported as an unmapped
observation rather than guessed at.

Coverage is deliberately narrow — the keyword set MuFinder needs to describe an
imported relaxation, plus the two values downstream jobs inherit — and is
expected to grow by transcription, not by pattern-matching.

**The echo is lossy for some keywords, irreducibly.** CASTEP prints one display
string for several distinct keyword values: ``LDA`` and ``LDA_PZ`` both print
*Perdew-Zunger Local Density Approximation*, and ``SHF`` and ``SX`` both print
*Screened Hartree-Fock*. Such an observation carries its alternatives in
:attr:`RunParameterObservation.ambiguous_with` so a caller can record the
ambiguity instead of silently asserting one reading.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Literal, Mapping

from .schema import canonical_keyword_name, get_keyword_spec
from .types import ValueWithUnit


class RunParameterParseError(Exception):
    """Raised when a ``.castep`` parameter echo cannot be read."""


FileKind = Literal["cell", "param"]

_RUN_STARTED_RE = re.compile(r"^\s*Run started:", re.IGNORECASE)
#: A section banner, e.g. ``****** General Parameters ******``. CASTEP centres
#: the title between runs of ``*`` whose length depends on the page width, so
#: the title is extracted rather than matched against a fixed layout.
_SECTION_RE = re.compile(r"^\s*\*{2,}\s+(?P<title>[A-Za-z][A-Za-z0-9 +/()-]*?)\s+\*{2,}\s*$")
#: The terminator after the last section: a bare run of asterisks.
_SECTION_END_RE = re.compile(r"^\s*\*{10,}\s*$")
_ASSIGNMENT_RE = re.compile(r"^\s*(?P<label>\S.*?)\s+:\s*(?P<value>.*?)\s*$")
#: ``cell.f90`` writes this with a 58-character label followed by ``3i3``.
_MP_GRID_RE = re.compile(
    r"^\s*MP grid size for SCF calculation is\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*$"
)
#: The version banner is boxed in ``|`` characters, and the closing bar is not
#: always separated from the version by a space, so the capture is restricted to
#: version-shaped characters rather than "everything up to whitespace".
_CASTEP_VERSION_RE = re.compile(r"CASTEP version\s+([0-9][0-9A-Za-z.\-_]*)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _LabelRule:
    """How to translate one ``label : value`` echo line."""

    keyword: str
    file: FileKind
    kind: Literal["enum", "physical", "int", "float", "on_off"]
    #: Display string to keyword value(s). The first entry of a multi-value
    #: tuple is reported as the value; the rest become ``ambiguous_with``.
    value_map: Mapping[str, tuple[str, ...]] | None = None


#: ``TASK`` display strings, transcribed from ``parameters.f90``.
_TASK_VALUES: dict[str, tuple[str, ...]] = {
    "single point energy": ("SINGLEPOINT",),
    "band structure": ("BANDSTRUCTURE",),
    "geometry optimization": ("GEOMETRYOPTIMIZATION",),
    "molecular dynamics": ("MOLECULARDYNAMICS",),
    "optical properties": ("OPTICS",),
    "transition state search": ("TRANSITIONSTATESEARCH",),
    "phonon calculation": ("PHONON",),
    "e-field calculation": ("EFIELD",),
    "phonon followed by e-field": ("PHONON+EFIELD",),
    "thermodynamics calculation": ("THERMODYNAMICS",),
    "magnetic resonance": ("MAGRES",),
    "electronic spectroscopy": ("ELECTRONICSPECTROSCOPY",),
    "core level spectra (elnes etc)": ("ELNES",),
    "elastic constants calculation": ("ELASTIC",),
}

#: ``XC_FUNCTIONAL`` display strings, transcribed from ``parameters.f90``.
#: Several CASTEP values share a display string; the tuple records every value
#: that could have produced the line, most-canonical first.
_XC_VALUES: dict[str, tuple[str, ...]] = {
    "perdew-zunger local density approximation": ("LDA", "LDA_PZ"),
    "perdew-wang local density approximation": ("LDA_PW",),
    "perdew wang (1991)": ("PW91",),
    "perdew burke ernzerhof": ("PBE",),
    "becke 86b exchange and perdew burke ernzerhof correlation": ("B86BPBE",),
    "pbe for solids (2008)": ("PBESOL",),
    "revised perdew burke ernzerhof": ("RPBE",),
    "pbe with wu-cohen exchange": ("WC",),
    "blyp": ("BLYP",),
    "hartree-fock": ("HF",),
    "screened hartree-fock": ("SHF", "SX"),
}


_LABEL_RULES: dict[str, _LabelRule] = {
    "type of calculation": _LabelRule("task", "param", "enum", _TASK_VALUES),
    "using functional": _LabelRule("xc_functional", "param", "enum", _XC_VALUES),
    "write structure in cell formatted file": _LabelRule(
        "write_cell_structure", "param", "on_off"
    ),
    "plane wave basis set cut-off": _LabelRule("cut_off_energy", "param", "physical"),
    "net charge of system": _LabelRule("charge", "param", "int"),
    "total energy / atom convergence tol.": _LabelRule("elec_energy_tol", "param", "physical"),
    "max. number of scf cycles": _LabelRule("max_scf_cycles", "param", "int"),
    "max. number of steps": _LabelRule("geom_max_iter", "param", "int"),
    "total energy convergence tolerance": _LabelRule("geom_energy_tol", "param", "physical"),
    "max ionic |force| tolerance": _LabelRule("geom_force_tol", "param", "physical"),
    "max ionic |displacement| tolerance": _LabelRule("geom_disp_tol", "param", "physical"),
    "max |stress component| tolerance": _LabelRule("geom_stress_tol", "param", "physical"),
}

#: Labels that are only meaningful inside one section. ``total energy
#: convergence tolerance`` is the geometry-optimisation criterion, while the
#: electronic one is ``total energy / atom convergence tol.``; and ``max. number
#: of steps`` is the geometry iteration cap. Without this guard a future CASTEP
#: version reusing either label elsewhere would silently mis-map it.
_LABEL_SECTIONS: dict[str, str] = {
    "max. number of steps": "Geometry Optimization Parameters",
    "total energy convergence tolerance": "Geometry Optimization Parameters",
    "max ionic |force| tolerance": "Geometry Optimization Parameters",
    "max ionic |displacement| tolerance": "Geometry Optimization Parameters",
    "max |stress component| tolerance": "Geometry Optimization Parameters",
}

#: Bare statement lines (CASTEP format 98) that encode a keyword value without a
#: ``label : value`` split. Keys are compared case-insensitively after
#: whitespace collapsing. Transcribed from ``parameters.f90``.
#:
#: ``Method: Treating system as non-metallic,`` is deliberately absent: it is
#: emitted for a fixed-occupancy calculation, which is not a ``METALS_METHOD``
#: value at all, so mapping it would assert something CASTEP did not say.
_STATEMENT_RULES: dict[str, tuple[str, FileKind, object]] = {
    "treating system as non-spin-polarized": ("spin_polarized", "param", False),
    "treating system as spin-polarized": ("spin_polarized", "param", True),
    "method: treating system as metallic with conventional treatment of electrons,": (
        "metals_method",
        "param",
        "NONE",
    ),
    "method: treating system as metallic with density mixing treatment of electrons,": (
        "metals_method",
        "param",
        "DM",
    ),
    "method: treating system as metallic with ensemble dft treatment of electrons,": (
        "metals_method",
        "param",
        "EDFT",
    ),
}


@dataclass(frozen=True, slots=True)
class RunParameterObservation:
    """One line of a ``.castep`` parameter echo, mapped or not."""

    section: str
    label: str
    raw_value: str
    keyword: str | None = None
    file: FileKind | None = None
    value: object | None = None
    #: Other CASTEP values that produce the same display string. Non-empty only
    #: for an irreducibly lossy label such as ``using functional``.
    ambiguous_with: tuple[str, ...] = ()

    @property
    def mapped(self) -> bool:
        """Return whether this line was translated to a MuFinder keyword."""

        return self.keyword is not None


@dataclass(frozen=True, slots=True)
class CastepRunParameters:
    """Everything recovered from one ``.castep`` parameter echo."""

    observations: tuple[RunParameterObservation, ...] = field(default_factory=tuple)
    castep_version: str | None = None

    def mapped_values(self) -> dict[str, object]:
        """Return ``{canonical keyword: value}`` for translated lines.

        Later observations win, so a multi-restart output reports its final
        settings. Callers wanting the file each value came from, or the
        ambiguity of a lossy label, should read :attr:`observations` instead.
        """

        values: dict[str, object] = {}
        for observation in self.observations:
            if observation.keyword is not None:
                values[observation.keyword] = observation.value
        return values

    def unmapped(self) -> tuple[RunParameterObservation, ...]:
        """Return echo lines with no entry in the translation tables."""

        return tuple(item for item in self.observations if not item.mapped)

    def ambiguous(self) -> tuple[RunParameterObservation, ...]:
        """Return mapped lines whose display string had several possible values."""

        return tuple(item for item in self.observations if item.ambiguous_with)


def _collapse(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _coerce_physical(raw_value: str, keyword: str) -> object:
    """Split ``"900.0000   eV"`` into a value and its unit.

    A bare number stays a float; the keyword's spec unit is never substituted,
    because an echo without a unit genuinely did not state one.
    """

    tokens = raw_value.split()
    if not tokens:
        return None
    try:
        magnitude = float(tokens[0])
    except ValueError as exc:
        raise ValueError(f"Non-numeric value '{raw_value}' for {keyword.upper()}.") from exc
    if len(tokens) >= 2:
        return ValueWithUnit(value=magnitude, unit=tokens[1])
    return magnitude


def _coerce_int(raw_value: str, keyword: str) -> int:
    """Read an integer that CASTEP may have printed in real format.

    ``net charge of system`` prints ``+1.000`` even though MuFinder types
    ``CHARGE`` as an integer, so a plain ``int()`` would reject it.
    """

    token = raw_value.split()[0] if raw_value.split() else ""
    try:
        return int(token)
    except ValueError:
        pass
    try:
        magnitude = float(token)
    except ValueError as exc:
        raise ValueError(f"Non-integer value '{raw_value}' for {keyword.upper()}.") from exc
    if magnitude != int(magnitude):
        raise ValueError(f"Non-integer value '{raw_value}' for {keyword.upper()}.")
    return int(magnitude)


def _coerce(rule: _LabelRule, raw_value: str) -> tuple[object, tuple[str, ...]]:
    """Convert one echo value, returning it with any ambiguous alternatives."""

    if rule.kind == "enum":
        candidates = (rule.value_map or {}).get(_collapse(raw_value))
        if not candidates:
            raise ValueError(f"Unrecognised {rule.keyword.upper()} display value '{raw_value}'.")
        return candidates[0], candidates[1:]
    if rule.kind == "on_off":
        collapsed = _collapse(raw_value)
        if collapsed == "on":
            return True, ()
        if collapsed == "off":
            return False, ()
        raise ValueError(f"Unrecognised on/off value '{raw_value}' for {rule.keyword.upper()}.")
    if rule.kind == "physical":
        return _coerce_physical(raw_value, rule.keyword), ()
    if rule.kind == "int":
        return _coerce_int(raw_value, rule.keyword), ()
    if rule.kind == "float":
        tokens = raw_value.split()
        if not tokens:
            return None, ()
        return float(tokens[0]), ()
    raise ValueError(f"Unsupported rule kind {rule.kind!r}.")


def _last_run_segment(lines: list[str]) -> list[str]:
    """Return the lines belonging to the final run in an appended output.

    CASTEP appends to an existing ``.castep``, so a continued calculation holds
    one parameter echo per restart. The last echo describes the settings the
    calculation finished with, which is what a caller wants.
    """

    starts = [index for index, line in enumerate(lines) if _RUN_STARTED_RE.match(line)]
    if not starts:
        return lines
    return lines[starts[-1] :]


def parse_run_parameters_text(text: str) -> CastepRunParameters:
    """Read the parameter echo out of ``.castep`` text.

    Lines outside the echo are ignored, so passing a whole multi-megabyte output
    is safe. Only the final restart's echo is read.
    """

    all_lines = text.splitlines()
    version_match = _CASTEP_VERSION_RE.search(text)
    lines = _last_run_segment(all_lines)

    observations: list[RunParameterObservation] = []
    section = ""
    in_echo = False

    for line in lines:
        grid_match = _MP_GRID_RE.match(line)
        if grid_match is not None:
            # Written by cell.f90 outside the parameter sections entirely.
            observations.append(
                RunParameterObservation(
                    section=section,
                    label="MP grid size for SCF calculation is",
                    raw_value=" ".join(grid_match.groups()),
                    keyword=canonical_keyword_name("cell", "kpoint_mp_grid"),
                    file="cell",
                    value=tuple(int(value) for value in grid_match.groups()),
                )
            )
            continue

        section_match = _SECTION_RE.match(line)
        if section_match is not None:
            section = section_match.group("title").strip()
            in_echo = True
            continue

        if _SECTION_END_RE.match(line):
            in_echo = False
            continue

        if not in_echo:
            continue

        collapsed = _collapse(line)
        statement = _STATEMENT_RULES.get(collapsed)
        if statement is not None:
            keyword, file, value = statement
            observations.append(
                RunParameterObservation(
                    section=section,
                    label=line.strip(),
                    raw_value="",
                    keyword=canonical_keyword_name(file, keyword),
                    file=file,
                    value=value,
                )
            )
            continue

        assignment = _ASSIGNMENT_RE.match(line)
        if assignment is None:
            continue
        label = assignment.group("label").strip()
        raw_value = assignment.group("value").strip()
        rule = _LABEL_RULES.get(_collapse(label))
        required_section = _LABEL_SECTIONS.get(_collapse(label))
        if rule is None or (required_section is not None and section != required_section):
            observations.append(
                RunParameterObservation(section=section, label=label, raw_value=raw_value)
            )
            continue

        try:
            value, ambiguous = _coerce(rule, raw_value)
        except ValueError:
            # An unreadable value is reported as unmapped rather than raised:
            # this parser runs over whole directories of third-party output and
            # one surprising line must not discard the rest of the echo.
            observations.append(
                RunParameterObservation(section=section, label=label, raw_value=raw_value)
            )
            continue

        observations.append(
            RunParameterObservation(
                section=section,
                label=label,
                raw_value=raw_value,
                keyword=canonical_keyword_name(rule.file, rule.keyword),
                file=rule.file,
                value=value,
                ambiguous_with=ambiguous,
            )
        )

    return CastepRunParameters(
        observations=tuple(observations),
        castep_version=version_match.group(1) if version_match else None,
    )


def parse_run_parameters(path: str | Path) -> CastepRunParameters:
    """Read the parameter echo from a ``.castep`` file on disk.

    Raises:
        RunParameterParseError: If the file cannot be read.
    """

    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RunParameterParseError(f"Failed to read CASTEP file '{source}': {exc}") from exc
    return parse_run_parameters_text(text)


def keyword_is_supported(observation: RunParameterObservation) -> bool:
    """Return whether a mapped observation's keyword exists in the runtime schema.

    The translation tables are fixed, but the schema they resolve against is
    discovered at runtime and is much smaller when no CASTEP installation is
    found. This distinguishes "MuFinder cannot name this line" from "this
    catalog does not carry that keyword".
    """

    if observation.keyword is None or observation.file is None:
        return False
    return get_keyword_spec(observation.file, observation.keyword) is not None


__all__ = [
    "CastepRunParameters",
    "RunParameterObservation",
    "RunParameterParseError",
    "keyword_is_supported",
    "parse_run_parameters",
    "parse_run_parameters_text",
]
