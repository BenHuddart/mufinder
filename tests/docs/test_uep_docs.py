from __future__ import annotations

from pathlib import Path


def test_uep_workflow_doc_includes_primary_reference() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")

    assert "Simone Sturniolo and Leandro Liborio" in text
    assert "10.1063/5.0012381" in text
    assert "Journal of Chemical Physics" in text or "J. Chem. Phys." in text


def test_uep_workflow_doc_includes_airss_context_reference() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "ab initio random structure searching" in normalized
    assert "10.1063/1.5024450" in text


def test_uep_workflow_doc_covers_core_theory_and_caveats() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")

    assert "reciprocal space" in text
    assert "special Wyckoff positions" in text
    assert "muonium" in text
    assert "neutral" in text
    assert "gw_factor" in text
    assert "batch" in text.lower()
    assert "Calculate ZPE" in text
    assert "E_{\\mathrm{zp}}" in text or "E_zp" in text


def test_uep_workflow_doc_mentions_slice_planes_and_multi_selection() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")

    assert "slice" in text.lower()
    assert "Shift" in text or "shift" in text
    assert "best-fit plane" in text or "best fit plane" in text


def test_uep_workflow_doc_mentions_separate_clustering_window() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")

    assert "separate clustering window" in text.lower()
    assert "dendrogram" in text.lower()
    assert "k-means" in text.lower()


def test_uep_workflow_doc_explains_cluster_statistics_and_representatives() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "representative" in normalized
    assert "minimum-energy member" in normalized or "minimum energy member" in normalized
    assert "not the mean position" in normalized
    assert "centroid" in normalized
    assert "standard deviation" in normalized
    assert "average energy relative to the global minimum" in normalized


def test_uep_workflow_doc_mentions_generation_cluster_positions_anchor() -> None:
    doc_path = Path(__file__).resolve().parents[2] / "docs" / "uep_workflow.rst"
    text = doc_path.read_text(encoding="utf-8")
    normalized = " ".join(text.lower().split())

    assert "cluster positions" in normalized
    assert "fractional anchor" in normalized
    assert "symmetry-equivalent representatives closest to a chosen fractional anchor" in normalized
    assert "different realizations of the same symmetry-equivalent site" in normalized


def test_user_docs_explain_autosave_vs_manual_save_boundary() -> None:
    docs_root = Path(__file__).resolve().parents[2] / "docs"

    implemented = (docs_root / "implemented_functionality.rst").read_text(
        encoding="utf-8"
    )
    workflow = (docs_root / "uep_workflow.rst").read_text(encoding="utf-8")
    implemented_normalized = " ".join(implemented.lower().split())
    workflow_normalized = " ".join(workflow.lower().split())

    assert "autosave" in implemented_normalized
    assert "save project" in implemented_normalized
    assert "session memory" in implemented_normalized
    assert "viewer camera" in implemented_normalized
    assert "bond and spin visibility" in implemented_normalized

    assert "save project" in workflow_normalized
    assert "autosave" in workflow_normalized
    assert "job-tracking information" in workflow_normalized
    assert "clustering-window state" in workflow_normalized
    assert "viewer orientation/zoom" in workflow_normalized
