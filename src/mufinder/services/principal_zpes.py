"""A site's principal zero-point energies, with their axes where the project holds the site.

:func:`~mufinder.science.quantum_phonons.muon_principal_zpes` gives the three principal
axes of the muon's well from a phonon calculation, Cartesian in that calculation's
frame. A phonon run may hold the muon at another symmetry image of the site than the
site's relaxed cell does (the GdRu2Si2 site 1 run does), so :func:`site_principal_zpes`
turns each axis onto the relaxed cell's image with
:func:`~mufinder.science.energy_barriers.site_image_rotation`. Every supercell of the host
shares its Cartesian frame, so the axes are then read in the host crystal's axes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.ase_bridge import is_muon_label
from mufinder.science.energy_barriers import EnergyBarrierError, site_image_rotation
from mufinder.science.quantum_phonons import (
    PhononCalculation,
    PrincipalZPE,
    muon_principal_zpes,
    structure_from_phonon_calculation,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SitePrincipalZPEs:
    """The three principal ZPEs of a site, lowest curvature first.

    ``at_relaxed_site`` is ``True`` when the axes were turned onto the site's relaxed
    cell, and ``False`` when they are the phonon calculation's own: with no relaxed
    cell or host to turn them by, or when its muon is no image of the relaxed one.
    """

    axes: tuple[PrincipalZPE, PrincipalZPE, PrincipalZPE]
    at_relaxed_site: bool


def site_principal_zpes(
    calculation: PhononCalculation,
    *,
    host: Structure | None,
    relaxed_site: Structure | None,
) -> SitePrincipalZPEs:
    """Return ``calculation``'s principal ZPEs, their axes on ``relaxed_site``'s image."""

    principal = muon_principal_zpes(calculation)
    if host is None or relaxed_site is None:
        return SitePrincipalZPEs(principal, False)
    muons = [index for index, label in enumerate(relaxed_site.species) if is_muon_label(label)]
    if len(muons) != 1:
        return SitePrincipalZPEs(principal, False)
    muon = np.asarray(relaxed_site.fractional_positions[muons[0]], dtype=float) @ np.asarray(
        relaxed_site.lattice_vectors, dtype=float
    )
    try:
        rotation = site_image_rotation(
            pristine_structure=host,
            site_structure=structure_from_phonon_calculation(calculation),
            image_cartesian=muon,
        )
    except (EnergyBarrierError, ValueError):
        return SitePrincipalZPEs(principal, False)
    except Exception:  # noqa: BLE001 - the phonon frame is the stated fallback.
        _LOGGER.warning("Could not match the phonon run's muon to its site", exc_info=True)
        return SitePrincipalZPEs(principal, False)
    turned = tuple(
        entry if entry.axis is None else replace(entry, axis=_signed(rotation @ entry.axis))
        for entry in principal
    )
    return SitePrincipalZPEs(turned, True)  # type: ignore[arg-type]


def _signed(axis: np.ndarray) -> tuple[float, float, float]:
    """Make the first component larger than 1e-3 in size positive, as an axis has no sense."""

    leading = next((component for component in axis if abs(component) > 1e-3), 1.0)
    signed = axis if leading > 0.0 else -axis
    return (float(signed[0]), float(signed[1]), float(signed[2]))


__all__ = ["SitePrincipalZPEs", "site_principal_zpes"]
