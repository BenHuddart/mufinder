"""Field-result factories shared by the polarisation tests."""

from __future__ import annotations

from collections.abc import Sequence

from mufinder.science.magnetic_fields import MagneticFieldResult

ZERO = (0.0, 0.0, 0.0)


def field_result(
    total: Sequence[float],
    applied: Sequence[float] = ZERO,
    label: str = "site 1",
    position: Sequence[float] = ZERO,
) -> MagneticFieldResult:
    """A commensurate result whose total is ``total`` (T), carrying ``applied`` within it.

    Everything that is not applied is put in the dipolar term.
    """

    internal = tuple(float(t) - float(a) for t, a in zip(total, applied, strict=True))
    return MagneticFieldResult(
        label, tuple(position), internal, ZERO, ZERO, ZERO, tuple(float(a) for a in applied)
    )
