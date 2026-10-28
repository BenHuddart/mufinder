"""Reader for the hyperfine tensors in a CASTEP ``.magres`` file.

A ``TASK : MAGRES`` run with ``MAGRES_TASK : HYPERFINE`` writes, for every atom,
the total hyperfine tensor, its principal values and axes, and its isotropic
(Fermi-contact) part, all in MHz, inside the ``[magres_old]`` section of
``<seed>.magres`` (``magres_hyperfine_output`` in CASTEP's ``magres.f90``). This
module returns those numbers exactly as CASTEP prints them, sign included; the
science layer owns their physical interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?"
_ATOM_RE = re.compile(r"^Atom:\s+(\S+)\s+(\d+)\s*$")
_COORDINATES_RE = re.compile(
    rf"^(\S+)\s+(\d+)\s+Coordinates\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})"
)
_TENSOR_ROW_RE = re.compile(rf"^\s*({_FLOAT_RE})\s+({_FLOAT_RE})\s+({_FLOAT_RE})\s*$")
_ISO_RE = re.compile(rf"^(\S+)\s+(\d+)\s+Iso:\s+({_FLOAT_RE})\s+\(MHz\)")


class MagresInputError(ValueError):
    """Raised when a ``.magres`` file holds no readable hyperfine tensor."""


@dataclass(frozen=True, slots=True)
class HyperfineTensor:
    """One atom's hyperfine tensor as CASTEP prints it, in MHz."""

    species: str
    index: int
    coordinates_ang: tuple[float, float, float]
    tensor_mhz: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ]
    isotropic_mhz: float


def parse_magres_hyperfine(path: str | Path) -> tuple[HyperfineTensor, ...]:
    """Return every atom's hyperfine tensor from a CASTEP ``.magres`` file.

    Raises :class:`MagresInputError` when the file cannot be read, holds no
    hyperfine section, or has a section without its tensor or isotropic value.
    """

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise MagresInputError(f"Failed to read magres file '{source}': {exc}") from exc

    tensors: list[HyperfineTensor] = []
    current: dict[str, object] | None = None
    rows: list[tuple[float, float, float]] = []

    def finish() -> None:
        if current is None:
            return
        label = f"{current['species']} {current['index']}"
        if len(rows) != 3 or "isotropic" not in current or "coordinates" not in current:
            raise MagresInputError(
                f"Hyperfine section for atom {label} in '{source.name}' is incomplete."
            )
        tensors.append(
            HyperfineTensor(
                species=str(current["species"]),
                index=int(current["index"]),  # type: ignore[call-overload]
                coordinates_ang=current["coordinates"],  # type: ignore[arg-type]
                tensor_mhz=(rows[0], rows[1], rows[2]),
                isotropic_mhz=float(current["isotropic"]),  # type: ignore[arg-type]
            )
        )

    for line in lines:
        stripped = line.strip()
        atom = _ATOM_RE.match(stripped)
        if atom is not None:
            finish()
            current = {"species": atom.group(1), "index": int(atom.group(2))}
            rows = []
            continue
        if current is None:
            continue
        coordinates = _COORDINATES_RE.match(stripped)
        if coordinates is not None:
            current["coordinates"] = tuple(float(coordinates.group(k)) for k in (3, 4, 5))
            continue
        row = _TENSOR_ROW_RE.match(line)
        if row is not None and len(rows) < 3 and "isotropic" not in current:
            rows.append((float(row.group(1)), float(row.group(2)), float(row.group(3))))
            continue
        iso = _ISO_RE.match(stripped)
        if iso is not None:
            current["isotropic"] = float(iso.group(3))
            continue
        if stripped == "[/magres_old]":
            finish()
            current = None
    finish()

    if not tensors:
        raise MagresInputError(f"'{source.name}' contains no hyperfine tensors.")
    return tuple(tensors)


def hyperfine_tensor_for_species(
    tensors: tuple[HyperfineTensor, ...], species: str
) -> HyperfineTensor:
    """Return the single tensor of ``species`` (for example the muon, ``H:mu``)."""

    matches = [tensor for tensor in tensors if tensor.species == species]
    if len(matches) != 1:
        found = "no" if not matches else str(len(matches))
        raise MagresInputError(f"Expected one {species} hyperfine tensor, found {found}.")
    return matches[0]
