"""Geometry-optimisation convergence history from CASTEP ``.castep`` output.

After every geometry step CASTEP prints a convergence table
(``geom_output_converged`` in ``Source/Functional/geometry.f90``)::

     LBFGS: finished iteration     1 with enthalpy= -3.40756945E+002 eV

     +-----------+-----------------+-----------------+------------+-----+ <-- LBFGS
     | Parameter |      value      |    tolerance    |    units   | OK? | <-- LBFGS
     +-----------+-----------------+-----------------+------------+-----+ <-- LBFGS
     |  dE/ion   |   1.551297E-001 |   2.000000E-005 |         eV | No  | <-- LBFGS
     |  |F|max   |   2.418541E-001 |   5.000000E-003 |       eV/A | No  | <-- LBFGS
     |  |dR|max  |   3.165457E-002 |   1.000000E-003 |          A | No  | <-- LBFGS
     +-----------+-----------------+-----------------+------------+-----+ <-- LBFGS

Each row names one convergence quantity, its value, its tolerance, the unit
and whether it has converged. Which rows appear depends on the run: the force
and displacement rows are absent when every ion is fixed and ``Smax`` is absent
when the cell is fixed. Every row ends in ``<-- METHOD``, where the method is
``BFGS``, ``LBFGS``, ``TPSD`` or ``DI`` (delocalised internals); other CASTEP
tables end in markers such as ``<-- min LBFGS`` or ``<-- SCF``, which are not
convergence rows.

The parser reads the most recent restart only (the segment after the last
``Run started:`` banner, as :func:`~mufinder.castep_io.output_parser.parse_castep_output`
does for completion), because CASTEP counts iterations from 0 again after a
continuation. An iteration counts once its table's closing divider is written,
so output cut off mid-table, as a file read while CASTEP is still writing it
can be, never reports a partial set of criteria.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re

from .output_parser import (
    _FLOAT_RE,
    _GEOM_FAILURE_RE,
    _GEOM_SUCCESS_RE,
    _RUN_STARTED_RE,
    _last_run_segment,
)

#: Methods whose convergence tables :func:`parse_geometry_progress_lines` reads.
GEOMETRY_METHODS = ("BFGS", "LBFGS", "TPSD", "DI")

_ROW_MARKERS = tuple(f"<-- {method}" for method in GEOMETRY_METHODS)
_FINISHED_RE = re.compile(
    rf"^\s*(?P<method>BFGS|LBFGS|TPSD|DI):\s+finished\s+iteration\s+(?P<iteration>\d+)\s+"
    rf"with\s+(?:enthalpy|energy)\s*=\s*(?P<enthalpy>{_FLOAT_RE})\s*(?P<unit>\S+)?",
    re.IGNORECASE,
)
# The quantity names themselves contain bars (``|F|max``), so they are spelt out.
_ROW_RE = re.compile(
    rf"^\s*\|\s*(?P<name>dE/ion|Smax|\|[A-Za-z]+\|max)\s*\|\s*(?P<value>{_FLOAT_RE})\s*\|"
    rf"\s*(?P<tolerance>{_FLOAT_RE})\s*\|\s*(?P<unit>[^|]*?)\s*\|\s*(?P<ok>Yes|No)\s*\|"
)
_DIVIDER_RE = re.compile(r"^\s*\+[-+]+\+\s*<--")
_METHOD_ECHO_RE = re.compile(r"^\s*optimization\s+method\s*:\s*(\S+)", re.IGNORECASE)
_MAX_STEPS_RE = re.compile(r"^\s*max\.\s+number\s+of\s+steps\s*:\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ConvergenceCriterion:
    """One row of a geometry step's convergence table."""

    #: The quantity as CASTEP labels it: ``dE/ion``, ``|F|max``, ``|dR|max``,
    #: ``Smax`` or (delocalised internals) ``|G|max``.
    name: str
    value: float
    tolerance: float
    unit: str
    converged: bool


@dataclass(frozen=True, slots=True)
class GeometryIteration:
    """The convergence state CASTEP reported after one geometry step."""

    iteration: int
    enthalpy: float
    enthalpy_unit: str
    criteria: tuple[ConvergenceCriterion, ...]

    @property
    def criteria_met(self) -> int:
        return sum(1 for criterion in self.criteria if criterion.converged)


@dataclass(frozen=True, slots=True)
class GeometryProgress:
    """Convergence history of the most recent restart of a geometry optimisation."""

    method: str | None
    #: ``GEOM_MAX_ITER`` as the restart's parameter echo reports it.
    max_iterations: int | None
    iterations: tuple[GeometryIteration, ...]
    #: Zero-based index of the restart these iterations belong to: 0 for the
    #: first run, 1 after one continuation, and so on.
    restart: int = 0
    #: ``True`` once CASTEP reports the optimisation converged, ``False`` once
    #: it reports it failed, ``None`` while it is still running.
    converged: bool | None = None

    @property
    def latest(self) -> GeometryIteration | None:
        return self.iterations[-1] if self.iterations else None


def is_geometry_progress_line(line: str) -> bool:
    """Return whether ``line`` can affect :func:`parse_geometry_progress_lines`.

    Lets a reader that follows a growing ``.castep`` keep only these lines
    instead of the whole file; parsing the kept lines gives the same result as
    parsing everything.
    """

    stripped = line.rstrip()
    return (
        stripped.endswith(_ROW_MARKERS)
        or "finished iteration" in stripped
        or "Geometry optimization" in stripped
        or "Run started" in stripped
        or "optimization method" in stripped
        or "max. number of steps" in stripped
    )


def parse_geometry_progress(path: Path) -> GeometryProgress | None:
    """Parse the geometry-optimisation history in the ``.castep`` at ``path``."""

    return parse_geometry_progress_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def parse_geometry_progress_text(text: str) -> GeometryProgress | None:
    """Parse the geometry-optimisation history in ``.castep`` text."""

    return parse_geometry_progress_lines(text.splitlines())


def parse_geometry_progress_lines(lines: Iterable[str]) -> GeometryProgress | None:
    """Parse the geometry-optimisation history in the lines of a ``.castep``.

    Returns ``None`` when the latest restart shows no sign of a geometry
    optimisation yet: no optimisation-method echo and no convergence table. That
    covers other tasks and a run that has not reached its parameter echo.
    """

    all_lines = [line for line in lines if is_geometry_progress_line(line)]
    restart = max(0, sum(1 for line in all_lines if _RUN_STARTED_RE.search(line)) - 1)
    segment = _last_run_segment(all_lines)

    method: str | None = None
    max_iterations: int | None = None
    converged: bool | None = None
    iterations: list[GeometryIteration] = []
    open_iteration: tuple[int, float, str] | None = None
    open_criteria: list[ConvergenceCriterion] = []

    for line in segment:
        finished = _FINISHED_RE.match(line)
        if finished is not None:
            method = method or finished.group("method").upper()
            open_iteration = (
                int(finished.group("iteration")),
                float(finished.group("enthalpy")),
                finished.group("unit") or "",
            )
            open_criteria = []
            continue
        if open_iteration is not None and line.rstrip().endswith(_ROW_MARKERS):
            row = _ROW_RE.match(line)
            if row is not None:
                open_criteria.append(
                    ConvergenceCriterion(
                        name=row.group("name").strip(),
                        value=float(row.group("value")),
                        tolerance=float(row.group("tolerance")),
                        unit=row.group("unit").strip(),
                        converged=row.group("ok") == "Yes",
                    )
                )
            elif _DIVIDER_RE.match(line) and open_criteria:
                iteration, enthalpy, unit = open_iteration
                iterations.append(
                    GeometryIteration(iteration, enthalpy, unit, tuple(open_criteria))
                )
                open_iteration = None
                open_criteria = []
            continue
        if (echo := _METHOD_ECHO_RE.match(line)) is not None:
            method = echo.group(1).upper()
        elif (steps := _MAX_STEPS_RE.match(line)) is not None:
            max_iterations = int(steps.group(1))
        elif _GEOM_SUCCESS_RE.search(line):
            converged = True
        elif _GEOM_FAILURE_RE.search(line):
            converged = False

    if method is None and not iterations:
        return None
    return GeometryProgress(
        method=method,
        max_iterations=max_iterations,
        iterations=tuple(iterations),
        restart=restart,
        converged=converged,
    )
