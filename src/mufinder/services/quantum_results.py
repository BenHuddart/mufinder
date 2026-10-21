"""Readings from stored quantum-phonon result payloads.

A completed quantum-phonon job stores its derived analysis as ``Result.data``:
the muon ZPE, the three selected muon modes under ``muon_zpe_modes`` (each
``mode_index``, ``frequency_cm``, ``participation``, ``zpe_eV`` and
``width_A``), and the ``w1_entanglement_witness`` with the definition it was
computed with under ``w1_definition``.

The readers here take that payload and return the values the views display.
:func:`w1_from_result_data` returns the witness in the definition of
:func:`mufinder.science.quantum_phonons.w1_entanglement_witness`, ``P - 3``: a
payload stored without ``w1_definition`` holds a witness computed with an
earlier formula, so its value is recomputed from the stored mode
participations. :func:`mode_widths_from_result_data` returns the zero-point
width of each selected mode, computing it from ``frequency_cm`` for a payload
stored without ``width_A``.

Layering: this lives in ``mufinder.services`` and imports ``science`` and its
sibling ``job_results`` only.
"""

from __future__ import annotations

from collections.abc import Mapping

from mufinder.science.quantum_phonons import mode_width_angstrom
from mufinder.services.job_results import parse_numeric

#: The ``w1_definition`` value written with a witness computed as ``P - 3``.
W1_DEFINITION = "participation_minus_3"


def _stored_modes(data: Mapping[str, object]) -> list[Mapping[str, object]] | None:
    modes = data.get("muon_zpe_modes")
    if not isinstance(modes, list):
        return None
    return [mode for mode in modes if isinstance(mode, Mapping)]


def w1_from_result_data(data: Mapping[str, object]) -> float | None:
    """Return the ``w1`` witness of a quantum-phonon payload as ``P - 3``.

    A payload carrying ``w1_definition`` returns its stored
    ``w1_entanglement_witness``. One stored without it is recomputed as the sum
    of the ``participation`` of its ``muon_zpe_modes`` minus 3; when it has no
    usable ``muon_zpe_modes`` the stored value is returned as it is. Returns
    ``None`` when there is no value to show.
    """

    stored = parse_numeric(data.get("w1_entanglement_witness"))
    if "w1_definition" in data:
        return stored
    modes = _stored_modes(data)
    if not modes:
        return stored
    participations = [parse_numeric(mode.get("participation")) for mode in modes]
    if any(value is None for value in participations):
        return stored
    return float(sum(value for value in participations if value is not None) - 3.0)


def mode_widths_from_result_data(data: Mapping[str, object]) -> tuple[float, ...]:
    """Return the zero-point width in Å of each stored ``muon_zpe_modes`` entry.

    The stored ``width_A`` is used when present; otherwise the width is
    computed from the entry's ``frequency_cm`` with the muon mass
    (:func:`~mufinder.science.quantum_phonons.mode_width_angstrom`). An entry
    with neither gives ``0.0``. A payload with no ``muon_zpe_modes`` gives an
    empty tuple.
    """

    widths: list[float] = []
    for mode in _stored_modes(data) or []:
        width = parse_numeric(mode.get("width_A"))
        if width is None:
            frequency = parse_numeric(mode.get("frequency_cm"))
            width = 0.0 if frequency is None else mode_width_angstrom(frequency)
        widths.append(float(width))
    return tuple(widths)


__all__ = ["W1_DEFINITION", "mode_widths_from_result_data", "w1_from_result_data"]
