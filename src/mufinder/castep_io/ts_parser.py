"""Parser for CASTEP transition-state `.ts` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase.units import Hartree

# CASTEP writes `.ts` energies in atomic units (Hartree); we expose eV downstream.
# Reuse ase's CODATA conversion (ase is a hard project dependency) rather than
# keeping a second hard-coded copy of the constant.
HARTREE_TO_EV = Hartree


class TransitionStateParseError(ValueError):
    """Raised when a CASTEP `.ts` file cannot be parsed."""


@dataclass(frozen=True, slots=True)
class TransitionStateImage:
    """One image from a CASTEP transition-state path."""

    label: str
    iteration: int
    reaction_coordinate: float
    energy: float  # eV (converted from the Hartree value stored in the `.ts` file)
    lattice_vectors: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    species: tuple[str, ...]
    cartesian_positions: tuple[tuple[float, float, float], ...]
    forces: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True, slots=True)
class TransitionStatePath:
    """The final converged path extracted from a CASTEP `.ts` file."""

    images: tuple[TransitionStateImage, ...]

    @property
    def barrier_height_eV(self) -> float | None:
        if not self.images:
            return None
        energies = [image.energy for image in self.images]
        return max(energies) - energies[0]


def parse_transition_state_file(path: str | Path) -> TransitionStatePath:
    """Parse one CASTEP `.ts` file and return the final path."""

    source = Path(path)
    return parse_transition_state_text(source.read_text(encoding="utf-8", errors="replace"))


def parse_transition_state_text(text: str) -> TransitionStatePath:
    """Parse CASTEP `.ts` text into the final path representation."""

    images = _parse_images(text.splitlines())
    final = _final_path(images)
    if not final:
        raise TransitionStateParseError("No complete transition-state path could be parsed from the `.ts` file.")
    return TransitionStatePath(images=tuple(sorted(final, key=lambda image: image.reaction_coordinate)))


def _parse_images(lines: list[str]) -> list[TransitionStateImage]:
    images: list[TransitionStateImage] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        parts = stripped.split()
        if len(parts) != 3 or parts[0] not in {"REA", "TST", "PRO"}:
            index += 1
            continue
        try:
            label = parts[0]
            iteration = int(parts[1])
            reaction_coordinate = float(parts[2])
        except ValueError:
            index += 1
            continue

        index += 1
        energy = None
        lattice: list[tuple[float, float, float]] = []
        species: list[str] = []
        positions: list[tuple[float, float, float]] = []
        forces: list[tuple[float, float, float]] = []
        while index < len(lines):
            content = lines[index]
            stripped = content.strip()
            next_parts = stripped.split()
            if len(next_parts) == 3 and next_parts[0] in {"REA", "TST", "PRO"} and "<--" not in content:
                break
            if not stripped:
                index += 1
                continue
            if "<-- E" in content:
                try:
                    energy = float(stripped.split()[0]) * HARTREE_TO_EV
                except ValueError as exc:
                    raise TransitionStateParseError("Invalid energy entry in CASTEP `.ts` file.") from exc
                index += 1
                continue
            if "<-- h" in content:
                values = stripped.split()
                if len(values) < 3:
                    raise TransitionStateParseError("Invalid lattice-vector entry in CASTEP `.ts` file.")
                lattice.append((float(values[0]), float(values[1]), float(values[2])))
                index += 1
                continue
            if "<-- R" in content:
                values = stripped.split()
                if len(values) < 5:
                    raise TransitionStateParseError("Invalid position entry in CASTEP `.ts` file.")
                species.append(values[0])
                positions.append((float(values[2]), float(values[3]), float(values[4])))
                index += 1
                continue
            if "<-- F" in content:
                values = stripped.split()
                if len(values) < 5:
                    raise TransitionStateParseError("Invalid force entry in CASTEP `.ts` file.")
                forces.append((float(values[2]), float(values[3]), float(values[4])))
                index += 1
                continue
            index += 1

        if energy is None or len(lattice) != 3 or not positions or len(positions) != len(forces):
            raise TransitionStateParseError("Incomplete image block in CASTEP `.ts` file.")
        images.append(
            TransitionStateImage(
                label=label,
                iteration=iteration,
                reaction_coordinate=reaction_coordinate,
                energy=energy,
                lattice_vectors=(lattice[0], lattice[1], lattice[2]),
                species=tuple(species),
                cartesian_positions=tuple(positions),
                forces=tuple(forces),
            )
        )
    return images


def _final_path(images: list[TransitionStateImage]) -> list[TransitionStateImage]:
    if not images:
        return []
    reactants = [image for image in images if image.label == "REA"]
    products = [image for image in images if image.label == "PRO"]
    transitions = [image for image in images if image.label == "TST"]
    if not reactants or not products:
        return []
    if not transitions:
        return [reactants[-1], products[-1]]
    final_iteration = max(image.iteration for image in transitions)
    final_transitions = [image for image in transitions if image.iteration == final_iteration]
    return [reactants[-1], *final_transitions, products[-1]]