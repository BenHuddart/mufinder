"""Structural checks on the Sphinx sources.

These tests check structure and citations, never wording: a copy-edit to a page
must not break them. Whether the prose matches the code is a review concern.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


DOCS_ROOT = Path(__file__).resolve().parents[2] / "docs"

# Directories under ``docs/`` whose ``.rst`` files are not pages. ``_build`` and
# ``api/generated`` are build outputs: autosummary writes the API pages into the
# latter and reaches them through an ``autosummary`` directive's ``:toctree:``
# option rather than a ``.. toctree::`` directive. ``_templates`` holds the
# autosummary templates, which Sphinx excludes from the sources.
_NOT_PAGES = (("_build",), ("api", "generated"), ("_templates",))

_TOCTREE_DIRECTIVE = re.compile(r"^\.\. toctree::\s*$")
_DOI = re.compile(r"10\.\d{4,9}/[^\s`<>]+")

# Literature cited by the user guides, by page path relative to ``docs/``.
# Dropping one of these from its page loses the attribution for a method
# MuFinder implements; adding a new citation to a page means adding it here.
CITED_DOIS: dict[str, tuple[str, ...]] = {
    "user_guide/examples/cof2.rst": (
        "10.1107/S0567740871005466",
        "10.1103/PhysRev.137.A982",
        "10.1103/PhysRevB.30.186",
        "10.1103/PhysRevB.87.121108",
        "10.1016/j.cpc.2022.108488",
    ),
    "user_guide/examples/fe.rst": (
        "10.1016/0038-1098(77)90401-X",
        "10.1007/BF02401559",
        "10.1103/PhysRevB.97.174414",
        "10.1103/PhysRevMaterials.3.073804",
        "10.1016/0038-1098(79)91025-1",
    ),
    "user_guide/examples/mnsi.rst": (
        "10.1103/PhysRevB.89.184425",
        "10.1103/PhysRevB.93.144419",
        "10.1016/0038-1098(76)90057-0",
    ),
    "user_guide/examples/alpha_n2.rst": (
        "10.1038/s42005-023-01260-7",
        "10.6084/m9.figshare.23203037.v1",
        "10.1016/0370-1573(76)90032-6",
        "10.1103/PhysRevB.59.10559",
        "10.1107/S0567740874004067",
    ),
    "user_guide/examples/gdru2si2.rst": (
        "10.15128/r15138jd91z",
        "10.1103/PhysRevB.111.054440",
        "10.1016/0304-8853(83)90058-6",
    ),
    "user_guide/overview.rst": ("10.1016/j.cpc.2022.108488",),
    "user_guide/hyperfine_coupling.rst": (
        "10.1039/d4dd00314d",
        "10.1088/2516-1075/adcb7c",
        "10.1103/PhysRevB.97.174414",
        "10.1103/PhysRevMaterials.3.073804",
    ),
    "user_guide/hyperfine_coupling_theory.rst": (
        "10.1021/cr300108a",
        "10.1103/PhysRevB.97.174414",
        "10.1103/PhysRevMaterials.3.073804",
    ),
    "user_guide/magnetic_fields.rst": (
        "10.1007/BF02198293",
        "10.1016/S0304-8853(03)00238-5",
        "10.1063/1.367113",
        "10.1103/PhysRev.67.351",
        "10.1103/PhysRevB.20.850",
        "10.1080/001075199181521",
        "10.1109/TSMC.1979.4310076",
    ),
    "user_guide/quantum_corrections.rst": ("10.1038/s42005-023-01260-7",),
    "user_guide/quantum_corrections_theory.rst": (
        "10.1038/s42005-023-01260-7",
        "10.6084/m9.figshare.23203037.v1",
        "10.1107/S0567740874004067",
    ),
    "user_guide/site_generation.rst": ("10.1016/j.cpc.2022.108488",),
    "user_guide/structural_relaxation_analysis.rst": (
        "10.1016/j.cpc.2022.108488",
        "10.1063/1.5024450",
    ),
    "user_guide/uep_workflow.rst": ("10.1063/5.0012381", "10.1063/1.5024450"),
}


def _is_not_a_page(path: Path) -> bool:
    parts = path.relative_to(DOCS_ROOT).parts
    return any(parts[: len(prefix)] == prefix for prefix in _NOT_PAGES)


def _source_pages() -> list[Path]:
    return sorted(path for path in DOCS_ROOT.rglob("*.rst") if not _is_not_a_page(path))


def _toctree_entries(page: Path) -> list[str]:
    """Return the document names listed in every toctree on ``page``."""
    entries: list[str] = []
    lines = page.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        if not _TOCTREE_DIRECTIVE.match(lines[index]):
            index += 1
            continue
        index += 1
        while index < len(lines) and (not lines[index].strip() or lines[index][0].isspace()):
            entry = lines[index].strip()
            if entry and not entry.startswith(":"):
                # Accept both ``name`` and ``Title <name>`` entries.
                titled = re.search(r"<([^>]+)>$", entry)
                entries.append(titled.group(1) if titled else entry)
            index += 1
    return entries


def test_every_docs_page_is_reachable_from_the_index_toctree() -> None:
    pages = {
        path.relative_to(DOCS_ROOT).with_suffix("").as_posix(): path for path in _source_pages()
    }
    reachable = {"index"}
    pending = ["index"]
    while pending:
        current = pages[pending.pop()]
        base = current.parent.relative_to(DOCS_ROOT)
        for entry in _toctree_entries(current):
            name = entry[1:] if entry.startswith("/") else (base / entry).as_posix()
            assert name in pages, f"docs/{current.name} toctree lists missing page {entry!r}"
            if name not in reachable:
                reachable.add(name)
                pending.append(name)

    orphans = sorted(set(pages) - reachable)
    assert not orphans, (
        "These docs pages are not reachable from a toctree in docs/index.rst, so the "
        f"`sphinx -W` build will fail: {orphans}"
    )


def test_docs_pages_are_restructured_text() -> None:
    # conf.py registers no Markdown parser, so Sphinx silently skips a ``.md``
    # page: it is never built, indexed or link-checked.
    markdown = sorted(
        path.relative_to(DOCS_ROOT).as_posix()
        for path in DOCS_ROOT.rglob("*.md")
        if not _is_not_a_page(path)
    )
    assert not markdown, f"Sphinx does not build Markdown; write these as .rst: {markdown}"


@pytest.mark.parametrize(
    ("page", "doi"),
    [(page, doi) for page, dois in CITED_DOIS.items() for doi in dois],
)
def test_cited_literature_stays_on_its_page(page: str, doi: str) -> None:
    text = (DOCS_ROOT / page).read_text(encoding="utf-8")
    assert doi in text, f"docs/{page} no longer cites DOI {doi}"


def test_every_cited_doi_is_recorded() -> None:
    cited = {
        (path.relative_to(DOCS_ROOT).as_posix(), doi.rstrip(".,;)"))
        for path in _source_pages()
        for doi in _DOI.findall(path.read_text(encoding="utf-8"))
    }
    recorded = {(page, doi) for page, dois in CITED_DOIS.items() for doi in dois}
    unrecorded = sorted(cited - recorded)
    assert not unrecorded, f"Add these citations to CITED_DOIS: {unrecorded}"
