from __future__ import annotations

from pathlib import Path


def test_structural_relaxation_analysis_docs_cover_methods_and_sources() -> None:
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    text = (docs_root / "structural_relaxation_analysis.rst").read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "connected components" in normalized
    assert "scan tolerances" in normalized
    assert "hierarchical" in normalized
    assert "k-means" in normalized
    assert "pymuon-suite" in normalized
    assert "liborio" in normalized
    assert "huddart" in normalized
    assert "radial ion displacement" in normalized
    assert "free-cell pristine relaxations are disabled" in normalized
    assert "wyckoff assignment" in normalized
    assert "wyckoff tolerance" in normalized
    assert "spglib" in normalized
    assert "pm-symmetry" in normalized
    assert "unoccupied" in normalized
    assert "delete analysis..." in normalized
    assert "relaxations it used are kept" in normalized


def test_docs_index_includes_structural_relaxation_analysis_page() -> None:
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    index_text = (docs_root / "index.rst").read_text(encoding="utf-8")

    assert "structural_relaxation_analysis" in index_text
