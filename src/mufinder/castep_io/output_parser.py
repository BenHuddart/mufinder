"""Utilities for parsing CASTEP ``.castep`` output files."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
import re


_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?"
_ENERGY_PATTERNS = (
    re.compile(rf"Total\s+energy[^\n=]*=\s*({_FLOAT_RE})\s*eV", re.IGNORECASE),
    re.compile(rf"Final\s+free\s+energy[^\n=]*=\s*({_FLOAT_RE})\s*eV", re.IGNORECASE),
    re.compile(rf"Final\s+energy[^\n=]*=\s*({_FLOAT_RE})\s*eV", re.IGNORECASE),
    re.compile(rf"NB\s+est\.\s+0K\s+energy[^\n=]*=\s*({_FLOAT_RE})\s*eV", re.IGNORECASE),
    re.compile(
        rf"NB\s+dispersion\s+corrected\s+est\.\s+0K\s+energy[^\n=]*=\s*({_FLOAT_RE})\s*eV",
        re.IGNORECASE,
    ),
    re.compile(
        rf"Total\s+energy\s+corrected\s+for\s+finite\s+basis\s+set[^\n=]*=\s*({_FLOAT_RE})\s*eV",
        re.IGNORECASE,
    ),
)
_ENTHALPY_PATTERNS = (
    re.compile(rf"Final\s+enthalpy[^\n=]*=\s*({_FLOAT_RE})\s*eV", re.IGNORECASE),
    re.compile(rf"Enthalpy[^\n=]*=\s*({_FLOAT_RE})\s*eV", re.IGNORECASE),
)
_MAX_FORCE_PATTERNS = (
    re.compile(rf"max\s+force(?:\s*/\s*atom)?[^\n:=]*[:=]\s*({_FLOAT_RE})", re.IGNORECASE),
    re.compile(rf"maximum\s+force[^\n:=]*[:=]\s*({_FLOAT_RE})", re.IGNORECASE),
)
_GEOM_SUCCESS_RE = re.compile(r"Geometry\s+optimization\s+completed\s+successfully", re.IGNORECASE)
_GEOM_FAILURE_RE = re.compile(r"Geometry\s+optimization\s+failed", re.IGNORECASE)
_FAILURE_MARKERS = (
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"\bFAILED\b", re.IGNORECASE),
    re.compile(r"did\s+not\s+converge", re.IGNORECASE),
    re.compile(r"\bpanic\b", re.IGNORECASE),
)
_WARNING_RE = re.compile(r"\bWARNING\b", re.IGNORECASE)
_SCF_RE = re.compile(r"\bSCF(?:\s+cycle|\s+iteration)?\s*[:#=]?\s*(\d+)\b", re.IGNORECASE)
_WALL_SECONDS_RE = re.compile(rf"(?:Total\s+time|Wall\s+time)[^\n=]*=\s*({_FLOAT_RE})\s*s(?:ec(?:onds?)?)?", re.IGNORECASE)
_WALL_HMS_RE = re.compile(r"(?:Total\s+time|Wall\s+time)[^\n]*?([0-9]{1,2}):([0-9]{2}):([0-9]{2}(?:\.\d+)?)", re.IGNORECASE)
_FORCE_BLOCK_HEADER_RE = re.compile(r"\**.*\bForces\b\s*\**", re.IGNORECASE)
_FORCE_LINE_RE = re.compile(
    rf"^\s*(?:\d+\s+)?([A-Za-z][A-Za-z0-9]*)\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s*$"
)
_ASE_FORCE_LINE_RE = re.compile(
    rf"^\s*\*\s*([A-Za-z][A-Za-z0-9]*)\s+\d+\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)\s+\*\s*$"
)
_STRESS_BLOCK_HEADER_RE = re.compile(r"\**.*\bStress\s+Tensor\b\s*\**", re.IGNORECASE)
_STRESS_LINE_RE = re.compile(rf"^\s*({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s*$")
_ASE_STRESS_LINE_RE = re.compile(
    rf"^\s*\*\s*[xyz]\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s+\*\s*$",
    re.IGNORECASE,
)
_LATTICE_HEADER_RE = re.compile(r"\b(?:real\s+)?lattice\b|\bunit\s+cell\b", re.IGNORECASE)
_RUN_STARTED_RE = re.compile(r"^\s*Run started:", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CastepResult:
    """Subset of parsed values needed for MuFinder runtime status and analysis."""

    completed: bool
    total_energy: float | None = None
    enthalpy: float | None = None
    forces: tuple[tuple[str, float, float, float], ...] = field(default_factory=tuple)
    stress: tuple[tuple[float, float, float], ...] = field(default_factory=tuple)
    max_force: float | None = None
    max_stress: float | None = None
    num_scf_cycles: int | None = None
    wall_time: float | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    error_message: str | None = None
    geom_converged: bool | None = None
    final_lattice_vectors: tuple[tuple[float, float, float], ...] | None = None
    mulliken_atomic_spins: tuple[float, ...] | None = None


def parse_castep_output(path: Path) -> CastepResult:
    """Parse a CASTEP output file and extract summary information.

    This parser is intentionally conservative. Unknown lines are ignored.

    **Multi-restart files.** CASTEP appends to an existing ``.castep``, so a
    continued calculation's output holds every restart end to end. The fields
    divide into two groups, and the split is deliberate:

    - ``completed``, ``geom_converged`` and ``error_message`` describe **the
      most recent restart only**. They are read from the segment after the last
      ``Run started:`` banner, so an earlier restart's transient failure cannot
      shadow a later success (and vice versa).
    - ``num_scf_cycles`` is the ``max()`` over the whole file and ``warnings``
      accumulates every ``WARNING`` line in it, so both span all restarts. A
      continued job therefore reports its highest SCF count and its warnings
      from any run, not just the last. This is a known, accepted caveat: they
      are diagnostic values, and a per-restart split would silently drop the
      history a user continuing a calculation wants to see. Do not read them as
      "what the last run did".

    The remaining numeric fields (energies, forces, stress, lattice) take the
    *last* match in the file, which for an appended output is the last restart
    that reported one.
    """

    output_path = Path(path)
    text = output_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    total_energy = _last_numeric_match(text, _ENERGY_PATTERNS)
    enthalpy = _last_numeric_match(text, _ENTHALPY_PATTERNS)
    warnings = tuple(line.strip() for line in lines if _WARNING_RE.search(line))
    num_scf_cycles = _max_scf_cycle(lines)
    wall_time = _parse_wall_time(text)

    forces = _parse_last_force_block(lines)
    stress = _parse_last_stress_block(lines)
    max_force = _last_numeric_match(text, _MAX_FORCE_PATTERNS)
    if max_force is None:
        max_force = _compute_max_force_magnitude(forces)
    max_stress = _compute_max_abs_component(stress)
    final_lattice_vectors = _parse_final_lattice_vectors(lines)
    mulliken_atomic_spins = parse_castep_mulliken_atomic_spins(output_path)

    # A continued/resubmitted calculation appends its output to the same
    # ``.castep`` file, so the file can contain several restarts each
    # starting with a "Run started:" banner. Completion/failure must reflect
    # only the most recent restart: an earlier restart's transient failure
    # (e.g. exhausting geometry-optimization steps before a
    # walltime-triggered continuation) must not shadow a later success.
    final_run_lines = _last_run_segment(lines)
    final_run_text = "\n".join(final_run_lines)

    geom_converged = None
    if _GEOM_SUCCESS_RE.search(final_run_text):
        geom_converged = True
    elif _GEOM_FAILURE_RE.search(final_run_text):
        geom_converged = False

    error_message = _find_error_message(final_run_lines)
    completed = _is_complete(final_run_text, error_message)

    return CastepResult(
        completed=completed,
        total_energy=total_energy,
        enthalpy=enthalpy,
        forces=forces,
        stress=stress,
        max_force=max_force,
        max_stress=max_stress,
        num_scf_cycles=num_scf_cycles,
        wall_time=wall_time,
        warnings=warnings,
        error_message=error_message,
        geom_converged=geom_converged,
        final_lattice_vectors=final_lattice_vectors,
        mulliken_atomic_spins=mulliken_atomic_spins,
    )


def parse_castep_mulliken_atomic_spins(path: Path) -> tuple[float, ...] | None:
    """Extract per-atom Mulliken spin moments from a CASTEP output when available."""

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()

    header_index = None
    for index, line in enumerate(lines):
        if "Atomic Populations (Mulliken)" in line:
            header_index = index

    if header_index is None:
        return None

    spins: list[float] = []
    saw_spin = False
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if saw_spin:
                break
            continue
        if stripped.startswith("Bond"):
            break
        if set(stripped) in ({"-"}, {"="}):
            continue
        if stripped.startswith("Species"):
            continue
        tokens = stripped.split()
        if len(tokens) < 3:
            continue
        spin_channel = tokens[2].lower()
        if spin_channel not in {"up:", "dn:"}:
            continue
        if spin_channel == "up:":
            try:
                spins.append(float(tokens[-1]))
                saw_spin = True
            except ValueError:
                continue

    if not saw_spin:
        return None
    return tuple(spins)


def _last_numeric_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> float | None:
    values: list[float] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                continue
    return values[-1] if values else None


def _max_scf_cycle(lines: list[str]) -> int | None:
    values: list[int] = []
    for line in lines:
        match = _SCF_RE.search(line)
        if match is None:
            continue
        try:
            values.append(int(match.group(1)))
        except ValueError:
            continue
    return max(values) if values else None


def _parse_wall_time(text: str) -> float | None:
    wall_seconds: list[float] = []
    for match in _WALL_SECONDS_RE.finditer(text):
        try:
            wall_seconds.append(float(match.group(1)))
        except ValueError:
            continue
    if wall_seconds:
        return wall_seconds[-1]

    hms_seconds: list[float] = []
    for match in _WALL_HMS_RE.finditer(text):
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        hms_seconds.append((hours * 3600.0) + (minutes * 60.0) + seconds)
    return hms_seconds[-1] if hms_seconds else None


def _parse_last_force_block(lines: list[str]) -> tuple[tuple[str, float, float, float], ...]:
    blocks: list[list[tuple[str, float, float, float]]] = []
    index = 0
    while index < len(lines):
        if _FORCE_BLOCK_HEADER_RE.search(lines[index]) is None:
            index += 1
            continue
        block, next_index = _read_force_block(lines, index + 1)
        if block:
            blocks.append(block)
        index = next_index

    if not blocks:
        return ()
    return tuple(blocks[-1])


def _parse_last_stress_block(lines: list[str]) -> tuple[tuple[float, float, float], ...]:
    blocks: list[list[tuple[float, float, float]]] = []
    index = 0
    while index < len(lines):
        if _STRESS_BLOCK_HEADER_RE.search(lines[index]) is None:
            index += 1
            continue
        block, next_index = _read_stress_block(lines, index + 1)
        if block:
            blocks.append(block)
        index = next_index

    if not blocks:
        return ()
    return tuple(blocks[-1][:3])


def _read_force_block(
    lines: list[str], start_index: int
) -> tuple[list[tuple[str, float, float, float]], int]:
    block: list[tuple[str, float, float, float]] = []
    index = start_index
    while index < len(lines):
        line = lines[index]
        if _FORCE_BLOCK_HEADER_RE.search(line) or _STRESS_BLOCK_HEADER_RE.search(line):
            break
        parsed = _parse_force_row(line)
        if parsed is not None:
            block.append(parsed)
            index += 1
            continue
        if block and not line.strip():
            break
        if block and line.strip().startswith("***"):
            break
        index += 1
    return block, index + 1 if index == start_index else index


def _read_stress_block(lines: list[str], start_index: int) -> tuple[list[tuple[float, float, float]], int]:
    block: list[tuple[float, float, float]] = []
    index = start_index
    while index < len(lines):
        line = lines[index]
        if _FORCE_BLOCK_HEADER_RE.search(line) or _STRESS_BLOCK_HEADER_RE.search(line):
            break
        parsed = _parse_stress_row(line)
        if parsed is not None:
            block.append(parsed)
            index += 1
            continue
        if block and not line.strip():
            break
        if block and line.strip().startswith("***"):
            break
        index += 1
    return block, index + 1 if index == start_index else index


def _parse_force_row(line: str) -> tuple[str, float, float, float] | None:
    match = _FORCE_LINE_RE.match(line)
    if match is not None:
        return (
            match.group(1),
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )

    match = _ASE_FORCE_LINE_RE.match(line)
    if match is None:
        return None

    return (
        match.group(1),
        _parse_force_component(match.group(2)),
        _parse_force_component(match.group(3)),
        _parse_force_component(match.group(4)),
    )


def _parse_force_component(token: str) -> float:
    return float(token.replace("(cons'd)", ""))


def _parse_stress_row(line: str) -> tuple[float, float, float] | None:
    match = _STRESS_LINE_RE.match(line)
    if match is not None:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))

    match = _ASE_STRESS_LINE_RE.match(line)
    if match is None:
        return None
    return (float(match.group(1)), float(match.group(2)), float(match.group(3)))


def _parse_final_lattice_vectors(
    lines: list[str],
) -> tuple[tuple[float, float, float], ...] | None:
    candidates: list[tuple[tuple[float, float, float], ...]] = []
    for index, line in enumerate(lines):
        if _LATTICE_HEADER_RE.search(line) is None:
            continue

        rows: list[tuple[float, float, float]] = []
        for candidate in lines[index + 1 : index + 16]:
            parsed = _parse_three_float_row(candidate)
            if parsed is None:
                if rows:
                    break
                continue
            rows.append(parsed)
            if len(rows) == 3:
                candidates.append((rows[0], rows[1], rows[2]))
                break

    if not candidates:
        return None
    return candidates[-1]


def _parse_three_float_row(line: str) -> tuple[float, float, float] | None:
    stripped = line.strip().replace("*", " ")
    tokens = re.findall(_FLOAT_RE, stripped)
    if len(tokens) < 3:
        return None
    try:
        return (float(tokens[0]), float(tokens[1]), float(tokens[2]))
    except ValueError:
        return None


def _compute_max_force_magnitude(rows: tuple[tuple[str, float, float, float], ...]) -> float | None:
    magnitudes: list[float] = []
    for _species, force_x, force_y, force_z in rows:
        magnitudes.append(sqrt((force_x * force_x) + (force_y * force_y) + (force_z * force_z)))
    if not magnitudes:
        return None
    return max(magnitudes)


def _compute_max_abs_component(rows: tuple[tuple[object, ...], ...]) -> float | None:
    values: list[float] = []
    for row in rows:
        for item in row:
            if isinstance(item, (int, float)):
                values.append(abs(float(item)))
    if not values:
        return None
    return max(values)


def _last_run_segment(lines: list[str]) -> list[str]:
    """Return the lines belonging to the most recent ``Run started:`` restart.

    Falls back to all lines when no restart banner is present (single-run
    output).
    """

    boundaries = [index for index, line in enumerate(lines) if _RUN_STARTED_RE.search(line)]
    if not boundaries:
        return lines
    return lines[boundaries[-1] :]


def _find_error_message(lines: list[str]) -> str | None:
    for line in lines:
        trimmed = line.strip()
        if not trimmed:
            continue
        for marker in _FAILURE_MARKERS:
            if marker.search(trimmed):
                return trimmed
    return None


def _is_complete(text: str, error_message: str | None) -> bool:
    if error_message is not None:
        return False

    completion_markers = (
        "Total time",
        "Calculation complete",
        "Geometry optimization completed successfully",
        "Total wall time",
        "Final energy",
        "NB est. 0K energy",
    )
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in completion_markers)
