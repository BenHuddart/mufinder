"""Compatibility helpers for warning-free spglib symmetry access."""

from __future__ import annotations

from typing import Any

import numpy as np
import spglib


def configure_spglib_error_handling() -> None:
    """Disable spglib's deprecated warning-based error handling when available."""

    error_module = getattr(spglib, "error", None)
    if error_module is None or not hasattr(error_module, "OLD_ERROR_HANDLING"):
        return
    error_module.OLD_ERROR_HANDLING = False


def _dataset_helpers():
    globals_dict = getattr(spglib.get_symmetry_dataset, "__globals__", {})
    expand_cell = globals_dict.get("_expand_cell")
    low_level = globals_dict.get("_spglib")
    dataset_type = globals_dict.get("SpglibDataset", getattr(spglib, "SpglibDataset", None))
    if expand_cell is None or low_level is None or dataset_type is None:
        raise AttributeError("spglib low-level dataset helpers are unavailable")
    return expand_cell, low_level, dataset_type


def get_crystal_symmetry_dataset(
    cell,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
):
    """Return a non-magnetic spglib dataset without deprecated warning paths."""

    try:
        expand_cell, low_level, dataset_type = _dataset_helpers()
        lattice, positions, numbers, _ = expand_cell(cell)
        dataset = low_level.dataset(
            lattice,
            positions,
            numbers,
            0,
            float(symprec),
            float(angle_tolerance),
        )
    except Exception:  # noqa: BLE001 - spglib raises typed errors for malformed cells.
        return None
    return dataset_type(**dataset)


configure_spglib_error_handling()


def get_crystal_symmetry(
    cell,
    *,
    symprec: float = 1e-5,
    angle_tolerance: float = -1.0,
) -> dict[str, Any] | None:
    """Return non-magnetic symmetry operations without spglib's deprecated wrapper."""

    dataset = get_crystal_symmetry_dataset(
        cell,
        symprec=float(symprec),
        angle_tolerance=float(angle_tolerance),
    )
    if dataset is None:
        return None

    if hasattr(dataset, "rotations"):
        rotations = dataset.rotations
        translations = dataset.translations
        equivalent_atoms = dataset.equivalent_atoms
    else:  # pragma: no cover - compatibility with mapping-style datasets.
        rotations = dataset["rotations"]
        translations = dataset["translations"]
        equivalent_atoms = dataset["equivalent_atoms"]

    return {
        "rotations": np.asarray(rotations, dtype=int),
        "translations": np.asarray(translations, dtype=float),
        "equivalent_atoms": np.asarray(equivalent_atoms, dtype=int),
    }