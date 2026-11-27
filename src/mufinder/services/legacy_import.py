"""Import alias for :mod:`mufinder.services.relaxation_import`."""

from __future__ import annotations

from mufinder.services.relaxation_import import (
    IMPORT_SUPERCELL_TOLERANCE,
    MUON_SPECIES,
    DetectedRun,
    ParameterConflict,
    RelaxationFolderScan,
    merge_parameters,
    muon_index,
    scan_relaxation_folder,
)

LegacyFolderScan = RelaxationFolderScan
scan_legacy_folder = scan_relaxation_folder

__all__ = [
    "IMPORT_SUPERCELL_TOLERANCE",
    "MUON_SPECIES",
    "DetectedRun",
    "LegacyFolderScan",
    "ParameterConflict",
    "merge_parameters",
    "muon_index",
    "scan_legacy_folder",
]
