from __future__ import annotations

from pathlib import Path


def test_uep_workflow_doc_includes_primary_reference() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")

    assert "Simone Sturniolo and Leandro Liborio" in text
    assert "10.1063/5.0012381" in text
    assert "Journal of Chemical Physics" in text or "J. Chem. Phys." in text


def test_uep_workflow_doc_covers_core_theory_and_caveats() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")

    assert "reciprocal space" in text
    assert "special Wyckoff positions" in text
    assert "muonium" in text
    assert "neutral" in text
    assert "gw_factor" in text
    assert "batch" in text.lower()
