from __future__ import annotations

from pathlib import Path


def test_site_generation_docs_cover_random_count_and_spacing_modes() -> None:
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    text = (docs_root / "site_generation.rst").read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "muon-muon separation" in normalized
    assert "number of initial positions" in normalized
    assert "0.350" in text
    assert "free cell volume" in normalized
    assert "symmetry-equivalent" in normalized
    assert "worker thread" in normalized
    assert "progress bar" in normalized
    assert "field is updated" in normalized
    assert "accepted-site progress" in normalized
    assert "progress maximum" in normalized
    assert "rather than presenting the prediction as a target" in normalized
    assert "disabled ``muon-muon separation`` field" in normalized
    assert "N_{\\mathrm{expected}}" in text
    assert "10.1016/j.cpc.2022.108488" in text


def test_site_generation_docs_cover_initial_position_clustering() -> None:
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    text = (docs_root / "site_generation.rst").read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "cluster positions" in normalized
    assert "fractional anchor" in normalized
    assert "different realizations of the same symmetry-equivalent final site" in normalized
    assert "symmetry-equivalent fractional coordinate closest to a user-chosen anchor" in normalized
    assert "does not create or delete rows" in normalized
    assert "spglib" in normalized
    assert "provenance entry" in normalized
    assert "only a coordinate-representation cleanup" in normalized


def test_docs_index_includes_site_generation_page() -> None:
    docs_root = Path(__file__).resolve().parents[2] / "docs"
    index_text = (docs_root / "index.rst").read_text(encoding="utf-8")

    assert "site_generation" in index_text
