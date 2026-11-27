"""The facts the Host Structure page shows about a structure, above its atom table.

:func:`structure_facts` reads them from the structure: the formula of the whole
cell, the atom count, the space group spglib finds at the shared symmetry
tolerance (:func:`mufinder.science.symmetry_analysis.detect_space_group`) and the lattice
(:func:`mufinder.science.structure_metrics.lattice_metrics`). The space-group
search is the costly part, so a page asks once per structure geometry, not on
every refresh. :meth:`StructureFacts.groups` gives them as the captioned values
of the page's fact card, in two groups.

:func:`supercell_facts` gives the Supercell Editor's facts about the cell it
previews: its atom count, image distance and cell lengths.
"""

from __future__ import annotations

from dataclasses import dataclass

from mufinder.domain.models import Structure
from mufinder.domain.text import space_group_label
from mufinder.science.ase_bridge import cell_formula
from mufinder.science.lattice import shortest_translation_length
from mufinder.science.structure_metrics import lattice_metrics
from mufinder.science.symmetry_analysis import detect_space_group


@dataclass(frozen=True, slots=True)
class LatticeFacts:
    """Lattice lengths in Å, angles in degrees and the cell volume in Å³."""

    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    volume: float


@dataclass(frozen=True, slots=True)
class StructureFacts:
    """What the Host Structure page states about a structure.

    ``space_group_symbol`` and ``space_group_number`` are ``None`` when spglib
    cannot classify the structure (a species that names no element, atoms that
    overlap), and ``lattice`` is ``None`` for a cell with a zero-length vector.
    """

    formula: str
    atom_count: int
    space_group_symbol: str | None
    space_group_number: int | None
    lattice: LatticeFacts | None

    def space_group_text(self) -> str | None:
        """``"P4₂/mnm (136)"``, or ``None`` when there is no space group."""

        if self.space_group_symbol is None or self.space_group_number is None:
            return None
        return space_group_label(self.space_group_symbol, self.space_group_number)

    def groups(self, source: str) -> tuple[tuple[tuple[str, str], ...], ...]:
        """The facts as two groups of ``(caption, value)`` pairs, units on the values.

        The structure's facts, then the cell's: the page shows them on one line
        with a hairline between the groups when they fit.

        ``Formula`` ``Co2F4``, ``Atoms`` ``6``, ``Space group`` ``P4₂/mnm (136)``
        and ``Source`` ``imports/CoF2.cif``; then ``a`` ``4.6950 Å`` to ``γ``
        ``90°`` and ``Volume`` ``70.10 Å³``. A fact with no value (no space
        group, an empty formula or source) is left out, and so is the second
        group without a lattice.
        """

        contents = (
            ("Formula", self.formula),
            ("Atoms", str(self.atom_count)),
            ("Space group", self.space_group_text() or ""),
            ("Source", source),
        )
        first = tuple((caption, value) for caption, value in contents if value)
        lattice = self.lattice
        if lattice is None:
            return (first,)
        second = (
            ("a", f"{lattice.a:.4f} Å"),
            ("b", f"{lattice.b:.4f} Å"),
            ("c", f"{lattice.c:.4f} Å"),
            ("α", f"{_angle(lattice.alpha)}°"),
            ("β", f"{_angle(lattice.beta)}°"),
            ("γ", f"{_angle(lattice.gamma)}°"),
            ("Volume", f"{lattice.volume:.2f} Å³"),
        )
        return (first, second)


def _angle(degrees: float) -> str:
    """An angle to two decimals without trailing zeros: ``90``, ``89.5``, ``119.99``."""

    text = f"{degrees:.2f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def supercell_facts(structure: Structure) -> tuple[tuple[str, str], ...]:
    """The Supercell Editor's facts about a generated supercell, as ``(caption, value)`` pairs.

    ``Atoms`` ``48``; ``Image distance`` ``9.39 Å``, the distance between a point
    and its nearest periodic image as the Suggest Supercell dialog gives it
    (:func:`~mufinder.science.lattice.shortest_translation_length`); and
    the cell lengths ``a``, ``b`` and ``c`` as the Host Structure page writes
    them (``9.3900 Å``). The lattice facts are left out for a cell with a
    zero-length vector.
    """

    facts = [("Atoms", str(len(structure.species)))]
    try:
        metrics = lattice_metrics(structure.lattice_vectors)
        image_distance = shortest_translation_length(structure.lattice_vectors)
    except ValueError:
        return tuple(facts)
    facts.append(("Image distance", f"{image_distance:.2f} Å"))
    facts.extend((axis, f"{metrics[axis]:.4f} Å") for axis in ("a", "b", "c"))
    return tuple(facts)


def structure_facts(structure: Structure) -> StructureFacts:
    """Read the facts of ``structure``; runs spglib once."""

    try:
        symmetry = detect_space_group(structure)
    except Exception:  # noqa: BLE001 - a malformed cell shows no space group, not an error.
        space_group: tuple[str | None, int | None] = (None, None)
    else:
        space_group = (symmetry.symbol, symmetry.number)
    try:
        metrics = lattice_metrics(structure.lattice_vectors)
    except ValueError:
        lattice = None
    else:
        lattice = LatticeFacts(
            a=metrics["a"],
            b=metrics["b"],
            c=metrics["c"],
            alpha=metrics["alpha"],
            beta=metrics["beta"],
            gamma=metrics["gamma"],
            volume=metrics["cell_volume"],
        )
    return StructureFacts(
        formula=cell_formula(structure.species),
        atom_count=len(structure.species),
        space_group_symbol=space_group[0],
        space_group_number=space_group[1],
        lattice=lattice,
    )
