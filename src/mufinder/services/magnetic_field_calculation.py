"""The Magnetic Fields page's calculations, ready to run off the GUI thread.

Each calculation is two steps, as in
:mod:`mufinder.services.polarisation_simulations`: a ``prepare_…`` function
reads what the calculation needs from the project (the relaxed structure of a
distortion correction) and copies it into a request, and a
``run_prepared_…`` function calculates the request without reading the
project, so it can run on a worker thread and be abandoned when the user
cancels; a cancelled worker finishes on its own and its result is discarded.
``run_magnetic_field_calculation``, ``run_dipolar_tensor_calculation`` and
``run_nuclear_relaxation_calculation`` do both in one call.

:func:`run_prepared_magnetic_field_calculation` chooses the calculation the
page's ``Calculate Dipolar Field`` button runs: the finite field list of a
commensurate structure, the field list of a commensurate approximant when the
settings allow one, or sampled field distributions otherwise. It returns the
results with the status line that describes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass

import numpy as np

from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.science.magnetic_fields import (
    DipolarTensorResult,
    MagneticFieldDistributionResult,
    MagneticFieldError,
    MagneticFieldOptions,
    MagneticFieldResult,
    calculate_commensurate_approximant_fields,
    calculate_commensurate_fields,
    calculate_dipolar_tensors_for_atoms,
    calculate_incommensurate_field_distributions,
    classify_magnetic_model,
    find_commensurate_approximant,
    find_equivalent_muon_sites,
)
from mufinder.science.magnetic_structure import MagneticStructureModel
from mufinder.science.nuclear_relaxation import (
    NuclearRelaxationError,
    NuclearRelaxationOptions,
    NuclearRelaxationResult,
    calculate_nuclear_relaxation,
    calculate_nuclear_relaxation_with_distortion,
)
from mufinder.services.magnetic_field_results import (
    UNRELAXED_ESTIMATE_SUFFIX,
    is_unrelaxed_site_source,
)
from mufinder.services.magnetic_field_sites import MagneticMuonSiteOption, distortion_structure
from mufinder.services.project_document import ProjectDocument

#: The kinds of field result a calculation returns.
COMMENSURATE_RESULTS = "commensurate"
DISTRIBUTION_RESULTS = "distribution"


class NonFiniteFieldError(MagneticFieldError):
    """Raised when a calculated field contains a NaN or an infinity."""


def non_finite_field_label(results: Sequence[object]) -> str | None:
    """Return the site label of the first result whose field is not finite, or ``None``.

    A result is a calculated field object or a saved payload: its total field
    and magnitude, or its sampled distribution fields, must all be finite.
    """

    for index, result in enumerate(results, start=1):
        if isinstance(result, Mapping):
            label = result.get("site_label")
            arrays = [result.get(key) for key in ("total_T", "magnitude_T", "fields_T")]
        else:
            label = getattr(result, "site_label", None)
            arrays = [getattr(result, key, None) for key in ("total", "magnitude", "fields_T")]
        for values in arrays:
            if values is None:
                continue
            try:
                finite = bool(np.all(np.isfinite(np.asarray(values, dtype=float))))
            except (TypeError, ValueError):
                continue
            if not finite:
                return str(label or f"site {index}")
    return None


def _raise_for_non_finite_fields(results: Sequence[object]) -> None:
    """Refuse calculated fields that contain a NaN or an infinity, naming the site."""

    label = non_finite_field_label(results)
    if label is not None:
        raise NonFiniteFieldError(
            f"the field at {label} is not a finite number (NaN or infinity), so it is not shown"
        )


def equivalent_site_count(
    host: Structure, site: MagneticMuonSiteOption, options: MagneticFieldOptions
) -> int | None:
    """How many positions a calculation at ``site`` covers, or ``None`` when symmetry fails."""

    if not options.find_equivalent_sites:
        return 1
    try:
        return len(
            find_equivalent_muon_sites(
                host, site.fractional_position, tolerance=float(options.equivalent_tolerance)
            )
        )
    except MagneticFieldError:
        return None


def site_count_text(count: int | None) -> str:
    """``4 muon sites`` for a known count; ``the muon site`` when it is unknown."""

    return "the muon site" if count is None else plural(count, "muon site")


def cancelled_status(*, keeps_results: bool) -> str:
    """The status line after the user cancels a calculation; the results shown stay."""

    return (
        "Calculation cancelled; the earlier results are kept"
        if keeps_results
        else "Calculation cancelled"
    )


@dataclass(frozen=True)
class MagneticFieldCalculationRequest:
    """Everything one field calculation needs, copied out of the project."""

    host: Structure
    model: MagneticStructureModel
    site: MagneticMuonSiteOption
    options: MagneticFieldOptions
    use_approximants: bool
    distorted_structure: Structure | None
    site_count: int | None


@dataclass(frozen=True)
class MagneticFieldCalculation:
    """A field calculation's results, their kind and the status line that describes them."""

    kind: str
    results: tuple[MagneticFieldResult, ...] | tuple[MagneticFieldDistributionResult, ...]
    status: str


def prepare_magnetic_field_calculation(
    document: ProjectDocument | None,
    host: Structure,
    model: MagneticStructureModel,
    site: MagneticMuonSiteOption,
    options: MagneticFieldOptions,
    *,
    use_approximants: bool,
) -> MagneticFieldCalculationRequest:
    """Copy what calculating the fields at ``site`` needs out of ``document``.

    Raises :class:`MagneticFieldError` when the site asks for a distortion
    correction whose relaxed structure is no longer in the project.
    """

    distorted = None
    if site.include_distortion:
        distorted = distortion_structure(document, site)
        if distorted is None:
            raise MagneticFieldError(
                "the relaxed structure for the muon-induced distortion correction is no longer available"
            )
    return MagneticFieldCalculationRequest(
        host=copy.deepcopy(host),
        model=copy.deepcopy(model),
        site=site,
        options=options,
        use_approximants=bool(use_approximants),
        distorted_structure=copy.deepcopy(distorted),
        site_count=equivalent_site_count(host, site, options),
    )


def run_prepared_magnetic_field_calculation(
    request: MagneticFieldCalculationRequest,
) -> MagneticFieldCalculation:
    """Calculate the fields of ``request``; reads no project.

    Raises :class:`NonFiniteFieldError` for a field that is not a finite
    number, and lets the backend's own errors through.
    """

    host = request.host
    model = request.model
    site = request.site
    options = request.options
    positions = [site.fractional_position]
    distorted_structures = [request.distorted_structure]
    corrected = " with muon-induced distortion corrections" if site.include_distortion else ""
    suffix = UNRELAXED_ESTIMATE_SUFFIX if is_unrelaxed_site_source(site.source) else ""
    if classify_magnetic_model(host, model) == "commensurate":
        results = calculate_commensurate_fields(
            host, model, positions, options=options, distorted_structures=distorted_structures
        )
        _raise_for_non_finite_fields(results)
        return MagneticFieldCalculation(
            COMMENSURATE_RESULTS,
            results,
            f"Results: calculated {plural(len(results), 'commensurate field value')}{corrected}."
            + suffix,
        )
    approximant = find_commensurate_approximant(
        host,
        model,
        tolerance=options.approximant_tolerance,
        max_period=options.approximant_max_period,
    )
    if approximant is not None and request.use_approximants:
        results = calculate_commensurate_approximant_fields(
            host, model, positions, options=options, distorted_structures=distorted_structures
        )
        _raise_for_non_finite_fields(results)
        return MagneticFieldCalculation(
            COMMENSURATE_RESULTS,
            results,
            f"Results: calculated {plural(len(results), 'field value')} of the commensurate approximant "
            f"{approximant.describe()} (q moved by at most {approximant.max_deviation_rlu:.2g} r.l.u.)"
            f"{corrected}." + suffix,
        )
    distributions = calculate_incommensurate_field_distributions(
        host, model, positions, options=options, distorted_structures=distorted_structures
    )
    _raise_for_non_finite_fields(distributions)
    status = f"Results: sampled {plural(len(distributions), 'incommensurate field distribution')}{corrected}"
    flagged = sum(
        1
        for result in distributions
        for check in getattr(result, "convergence", ())
        if not check.converged
    )
    if flagged:
        status += f"; {plural(flagged, 'convergence check needs', 'convergence checks need')} attention, select a row for advice"
    status += "."
    if approximant is not None:
        status += (
            f" q is within {approximant.max_deviation_rlu:.2g} r.l.u. of the commensurate "
            f"approximant {approximant.describe()}; turn on commensurate approximants in "
            "Calculation Settings to list its finite fields."
        )
    return MagneticFieldCalculation(DISTRIBUTION_RESULTS, distributions, status + suffix)


def run_magnetic_field_calculation(
    document: ProjectDocument | None,
    host: Structure,
    model: MagneticStructureModel,
    site: MagneticMuonSiteOption,
    options: MagneticFieldOptions,
    *,
    use_approximants: bool,
) -> MagneticFieldCalculation:
    """Calculate the fields at ``site`` in one call, on the calling thread."""

    return run_prepared_magnetic_field_calculation(
        prepare_magnetic_field_calculation(
            document, host, model, site, options, use_approximants=use_approximants
        )
    )


@dataclass(frozen=True)
class DipolarTensorRequest:
    """Everything one dipolar-tensor calculation needs, copied from the page."""

    host: Structure
    site: MagneticMuonSiteOption
    magnetic_ion_indices: tuple[int, ...]
    options: MagneticFieldOptions
    site_count: int | None


def prepare_dipolar_tensor_calculation(
    host: Structure,
    site: MagneticMuonSiteOption,
    *,
    magnetic_ion_indices: Sequence[int],
    supercell: Sequence[int],
    include_equivalent_sites: bool,
) -> DipolarTensorRequest:
    """Copy what the dipolar tensors of the chosen ions at ``site`` need."""

    options = MagneticFieldOptions(
        supercell=tuple(int(value) for value in supercell),
        find_equivalent_sites=bool(include_equivalent_sites),
        include_lorentz=False,
    )
    return DipolarTensorRequest(
        host=copy.deepcopy(host),
        site=site,
        magnetic_ion_indices=tuple(int(index) for index in magnetic_ion_indices),
        options=options,
        site_count=equivalent_site_count(host, site, options),
    )


def run_prepared_dipolar_tensor_calculation(
    request: DipolarTensorRequest,
) -> tuple[DipolarTensorResult, ...]:
    """Calculate the dipolar tensors of ``request``; reads no project."""

    return tuple(
        calculate_dipolar_tensors_for_atoms(
            request.host,
            [request.site.fractional_position],
            magnetic_ion_indices=list(request.magnetic_ion_indices),
            options=request.options,
        )
    )


def run_dipolar_tensor_calculation(
    host: Structure,
    site: MagneticMuonSiteOption,
    *,
    magnetic_ion_indices: Sequence[int],
    supercell: Sequence[int],
    include_equivalent_sites: bool,
) -> tuple[DipolarTensorResult, ...]:
    """The dipolar tensors at ``site`` and, optionally, its equivalents, in one call."""

    return run_prepared_dipolar_tensor_calculation(
        prepare_dipolar_tensor_calculation(
            host,
            site,
            magnetic_ion_indices=magnetic_ion_indices,
            supercell=supercell,
            include_equivalent_sites=include_equivalent_sites,
        )
    )


@dataclass(frozen=True)
class NuclearRelaxationRequest:
    """Everything one nuclear-relaxation calculation needs, copied out of the project."""

    host: Structure
    site: MagneticMuonSiteOption
    options: NuclearRelaxationOptions
    distorted_structure: Structure | None


def prepare_nuclear_relaxation_calculation(
    document: ProjectDocument | None,
    host: Structure,
    site: MagneticMuonSiteOption,
    options: NuclearRelaxationOptions,
) -> NuclearRelaxationRequest:
    """Copy what the nuclear relaxation at ``site`` needs out of ``document``.

    Raises :class:`NuclearRelaxationError` when a distortion correction is
    asked for and the site has no relaxed structure in the project.
    """

    distorted = None
    if options.use_distortion_correction:
        if document is None or site.distortion_structure_id is None:
            raise NuclearRelaxationError("selected muon site has no distortion structure")
        distorted = distortion_structure(document, site)
        if distorted is None:
            raise NuclearRelaxationError("selected distortion structure is no longer available")
    return NuclearRelaxationRequest(
        host=copy.deepcopy(host),
        site=site,
        options=options,
        distorted_structure=copy.deepcopy(distorted),
    )


def run_prepared_nuclear_relaxation_calculation(
    request: NuclearRelaxationRequest,
) -> NuclearRelaxationResult:
    """Calculate the nuclear relaxation of ``request``; reads no project."""

    site = request.site
    if request.distorted_structure is not None:
        return calculate_nuclear_relaxation_with_distortion(
            request.host,
            request.distorted_structure,
            site.fractional_position,
            site_label=site.label,
            options=request.options,
        )
    return calculate_nuclear_relaxation(
        request.host,
        site.fractional_position,
        site_label=site.label,
        options=request.options,
    )


def run_nuclear_relaxation_calculation(
    document: ProjectDocument | None,
    host: Structure,
    site: MagneticMuonSiteOption,
    options: NuclearRelaxationOptions,
) -> NuclearRelaxationResult:
    """Calculate the nuclear relaxation at ``site`` in one call, on the calling thread."""

    return run_prepared_nuclear_relaxation_calculation(
        prepare_nuclear_relaxation_calculation(document, host, site, options)
    )
