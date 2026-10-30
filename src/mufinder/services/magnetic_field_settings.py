"""The numerical settings of a Magnetic Fields calculation, as one value.

:class:`MagneticFieldCalculationSettings` holds what the Calculation Settings
dialog edits and the page reads: the lattice-sum cells, the incommensurate
lookup and sampling, the histogram, the commensurate approximants and the
contact shell. It reads and writes the page's saved form keys, keeps every
value inside the range the dialog offers, writes the one-line summary the page
shows, and fills the matching fields of
:class:`~mufinder.science.magnetic_fields.MagneticFieldOptions`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from mufinder.science.magnetic_fields import MagneticFieldOptions

_DEFAULTS = MagneticFieldOptions()

#: The range of each lattice-sum cell count.
MIN_CALCULATION_CELLS = 1
MAX_CALCULATION_CELLS = 500
#: The range of the phase lookup angles per propagation vector.
MIN_INCOMMENSURATE_LOOKUP_ANGLES = 12
MAX_INCOMMENSURATE_LOOKUP_ANGLES = 1_000_000
#: The range of the incommensurate distribution samples.
MIN_INCOMMENSURATE_PHASE_SAMPLES = 12
MAX_INCOMMENSURATE_PHASE_SAMPLES = 10_000_000
#: The range of the histogram bin count.
MIN_HISTOGRAM_BINS = 10
MAX_HISTOGRAM_BINS = 10_000
#: The largest approximant tolerance, in r.l.u.; any positive value up to it is valid.
MAX_APPROXIMANT_TOLERANCE_RLU = 0.05
#: The longest commensurate approximant period.
MAX_APPROXIMANT_PERIOD = 64
#: The most nearest magnetic ions the contact shell may average.
MAX_CONTACT_NEIGHBOURS = 64
#: The range of the contact radius, in Angstrom, and the places it is kept to.
MIN_CONTACT_RADIUS_ANG = 0.1
MAX_CONTACT_RADIUS_ANG = 100.0
CONTACT_RADIUS_DECIMALS = 2

#: The phase-sampling methods, as saved key and label.
PHASE_SAMPLING_METHOD_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("grid", "Grid"),
    ("montecarlo", "Monte Carlo"),
)


def normalize_phase_sampling_method(value: object) -> str:
    """The saved phase-sampling key for ``value``; older spellings map to ``montecarlo``, others to ``auto``."""

    text = str(value).strip().lower()
    if text in {"monte_carlo", "random"}:
        text = "montecarlo"
    allowed = {key for key, _label in PHASE_SAMPLING_METHOD_OPTIONS}
    return text if text in allowed else "auto"


def phase_sampling_label(value: object) -> str:
    """The label of the phase-sampling method ``value`` names."""

    method = normalize_phase_sampling_method(value)
    for key, label in PHASE_SAMPLING_METHOD_OPTIONS:
        if key == method:
            return label
    return "Auto"


@dataclass(frozen=True, slots=True)
class MagneticFieldCalculationSettings:
    """The Calculation Settings of the Magnetic Fields page.

    Every value must lie inside the ranges above, or construction raises
    ``ValueError``; :meth:`with_state` reads saved values and brings them into
    range first.
    """

    supercell: tuple[int, int, int] = _DEFAULTS.supercell
    nangles: int = _DEFAULTS.nangles
    phase_sample_count: int = _DEFAULTS.phase_sample_count
    phase_sampling: str = _DEFAULTS.phase_sampling
    histogram_bins: int = _DEFAULTS.histogram_bins
    use_approximants: bool = False
    approximant_tolerance: float = _DEFAULTS.approximant_tolerance
    approximant_max_period: int = _DEFAULTS.approximant_max_period
    contact_nnn: int = _DEFAULTS.nnn
    contact_rcont: float = _DEFAULTS.rcont

    def __post_init__(self) -> None:
        checks = (
            (
                "calculation cells",
                len(self.supercell) == 3
                and all(
                    MIN_CALCULATION_CELLS <= cells <= MAX_CALCULATION_CELLS
                    for cells in self.supercell
                ),
            ),
            (
                "lookup angles",
                MIN_INCOMMENSURATE_LOOKUP_ANGLES
                <= self.nangles
                <= MAX_INCOMMENSURATE_LOOKUP_ANGLES,
            ),
            (
                "distribution samples",
                MIN_INCOMMENSURATE_PHASE_SAMPLES
                <= self.phase_sample_count
                <= MAX_INCOMMENSURATE_PHASE_SAMPLES,
            ),
            (
                "sampling method",
                self.phase_sampling in {key for key, _label in PHASE_SAMPLING_METHOD_OPTIONS},
            ),
            ("histogram bins", MIN_HISTOGRAM_BINS <= self.histogram_bins <= MAX_HISTOGRAM_BINS),
            (
                "approximant tolerance",
                0.0 < self.approximant_tolerance <= MAX_APPROXIMANT_TOLERANCE_RLU,
            ),
            ("approximant period", 1 <= self.approximant_max_period <= MAX_APPROXIMANT_PERIOD),
            ("contact neighbours", 1 <= self.contact_nnn <= MAX_CONTACT_NEIGHBOURS),
            (
                "contact radius",
                MIN_CONTACT_RADIUS_ANG <= self.contact_rcont <= MAX_CONTACT_RADIUS_ANG,
            ),
        )
        for name, valid in checks:
            if not valid:
                raise ValueError(f"{name} out of range in {self!r}")

    def with_state(self, state: Mapping[str, object]) -> MagneticFieldCalculationSettings:
        """These settings updated from the saved Magnetic Fields form ``state``.

        A count outside its range is brought to the nearest end of it. A
        missing or unreadable cell triple or count, and a missing sampling
        method, keep this object's value; an unrecognised sampling method reads
        as ``auto``, and unreadable approximant and contact values fall back to
        their defaults.
        """

        supercell = self.supercell
        cells = _int_triplet(state.get("supercell"))
        if cells is not None:
            supercell = tuple(  # type: ignore[assignment]
                _clamp(value, MIN_CALCULATION_CELLS, MAX_CALCULATION_CELLS) for value in cells
            )
        return replace(
            self,
            supercell=supercell,
            nangles=_clamped_int_or(
                state.get("phase_lookup_angles", state.get("nangles")),
                self.nangles,
                MIN_INCOMMENSURATE_LOOKUP_ANGLES,
                MAX_INCOMMENSURATE_LOOKUP_ANGLES,
            ),
            phase_sample_count=_clamped_int_or(
                state.get(
                    "incommensurate_samples", state.get("phase_sample_count", state.get("nangles"))
                ),
                self.phase_sample_count,
                MIN_INCOMMENSURATE_PHASE_SAMPLES,
                MAX_INCOMMENSURATE_PHASE_SAMPLES,
            ),
            phase_sampling=normalize_phase_sampling_method(
                state.get("phase_sampling", self.phase_sampling)
            ),
            histogram_bins=_clamped_int_or(
                state.get("histogram_bins"),
                self.histogram_bins,
                MIN_HISTOGRAM_BINS,
                MAX_HISTOGRAM_BINS,
            ),
            use_approximants=bool(state.get("use_commensurate_approximants", False)),
            approximant_tolerance=approximant_tolerance_or_default(
                state.get("approximant_tolerance")
            ),
            approximant_max_period=approximant_max_period_or_default(
                state.get("approximant_max_period")
            ),
            contact_nnn=contact_nnn_or_default(state.get("contact_nnn")),
            contact_rcont=contact_rcont_or_default(state.get("contact_rcont")),
        )

    def state_items(self) -> dict[str, Any]:
        """The saved form keys for these settings (the page saves its other fields beside them)."""

        return {
            "supercell": list(self.supercell),
            "phase_lookup_angles": int(self.nangles),
            "incommensurate_samples": int(self.phase_sample_count),
            "phase_sampling": str(self.phase_sampling),
            "histogram_bins": int(self.histogram_bins),
            "use_commensurate_approximants": bool(self.use_approximants),
            "approximant_tolerance": float(self.approximant_tolerance),
            "approximant_max_period": int(self.approximant_max_period),
            "contact_nnn": int(self.contact_nnn),
            "contact_rcont": float(self.contact_rcont),
        }

    def option_values(self) -> dict[str, Any]:
        """The :class:`MagneticFieldOptions` fields these settings set, by field name."""

        return {
            "supercell": tuple(int(value) for value in self.supercell),
            "nnn": int(self.contact_nnn),
            "rcont": float(self.contact_rcont),
            "nangles": int(self.nangles),
            "phase_sample_count": int(self.phase_sample_count),
            "phase_sampling": str(self.phase_sampling),
            "histogram_bins": int(self.histogram_bins),
            "approximant_tolerance": float(self.approximant_tolerance),
            "approximant_max_period": int(self.approximant_max_period),
        }

    def summary(self, *, contact_shell: bool) -> str:
        """The page's one-line settings summary; ``contact_shell`` adds the contact shell."""

        a, b, c = (int(value) for value in self.supercell)
        text = (
            "Settings: "
            f"cells {a} x {b} x {c}; "
            f"{int(self.nangles)} lookup angles per q; "
            f"{int(self.phase_sample_count)} distribution samples; "
            f"{phase_sampling_label(self.phase_sampling)} sampling; "
            f"{int(self.histogram_bins)} histogram bins"
        )
        if self.use_approximants:
            text += (
                f"; commensurate approximants within {self.approximant_tolerance:g} r.l.u., "
                f"period up to {self.approximant_max_period}"
            )
        if contact_shell:
            ions = "ion" if self.contact_nnn == 1 else "ions"
            text += (
                f"; contact shell of {self.contact_nnn} nearest magnetic "
                f"{ions} within {self.contact_rcont:g} Å"
            )
        return text + "."


def approximant_tolerance_or_default(value: object) -> float:
    """``value`` as an approximant tolerance in (0, 0.05] r.l.u., else the default."""

    default = _DEFAULTS.approximant_tolerance
    try:
        tolerance = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    valid = np.isfinite(tolerance) and 0.0 < tolerance <= MAX_APPROXIMANT_TOLERANCE_RLU
    return tolerance if valid else default


def approximant_max_period_or_default(value: object) -> int:
    """``value`` as an approximant period from 1 to the longest, else the default."""

    default = _DEFAULTS.approximant_max_period
    try:
        period = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return period if 1 <= period <= MAX_APPROXIMANT_PERIOD else default


def contact_nnn_or_default(value: object) -> int:
    """``value`` as a contact-neighbour count from 1 to the most, else the default."""

    default = _DEFAULTS.nnn
    try:
        nnn = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return nnn if 1 <= nnn <= MAX_CONTACT_NEIGHBOURS else default


def contact_rcont_or_default(value: object) -> float:
    """``value`` as a contact radius in range, to the places the dialog shows, else the default."""

    default = _DEFAULTS.rcont
    try:
        rcont = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    in_range = MIN_CONTACT_RADIUS_ANG <= rcont <= MAX_CONTACT_RADIUS_ANG
    return round(rcont, CONTACT_RADIUS_DECIMALS) if np.isfinite(rcont) and in_range else default


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _clamped_int_or(value: object, fallback: int, minimum: int, maximum: int) -> int:
    """``int(value)`` brought into range, or ``fallback`` when ``value`` is missing or unreadable."""

    if value is None:
        return fallback
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return fallback
    return _clamp(number, minimum, maximum)


def _int_triplet(value: object) -> tuple[int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    try:
        return (int(value[0]), int(value[1]), int(value[2]))
    except (TypeError, ValueError, OverflowError):
        return None
