"""Repository harness checks for agent-legible development."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src" / "mufinder"


REQUIRED_AGENT_DOCS = (
    "docs/agent_harness.rst",
    "docs/architecture.rst",
    "docs/project_specification.rst",
    "docs/quality_and_automation.rst",
    "docs/compatibility_policy.rst",
)

INDEXED_DOCS = (
    "agent_harness",
    "architecture",
    "quality_and_automation",
    "compatibility_policy",
)

FORBIDDEN_LAYER_IMPORTS = {
    "domain": {
        "app",
        "castep_io",
        "persistence",
        "science",
        "services",
        "workflow",
    },
    "castep_io": {"app", "persistence", "science", "services", "workflow"},
    "persistence": {"app", "castep_io", "science", "services", "workflow"},
    "science": {"app", "persistence", "services", "workflow"},
    "workflow": {"app", "science", "services"},
    "services": {"app"},
}


# Per-workflow submit/materialize handlers still living on ``MainWindow``.
#
# The Phase 4 submission-consistency initiative moves each workflow's
# submit → materialize → result → refresh cycle onto a
# ``mufinder.app.workflow_controllers`` controller. This set pins the handlers
# that have NOT yet migrated, so it acts as a ratchet against the god object:
# the test fails if a NEW ``_on_submit_*`` / ``_materialize_*`` name appears on
# ``MainWindow``. Each migration PR removes the migrated workflow's names from
# this set (the set may only shrink). Cutoff and k-point convergence migrated
# first, so their former ``_on_submit_*_from_convergence`` /
# ``_on_submit_*_from_kpoint_convergence`` /
# ``_materialize_convergence_jobs`` / ``_materialize_kpoint_jobs`` handlers are
# intentionally absent here.
PINNED_MAIN_WINDOW_SUBMIT_HANDLERS = frozenset(
    {
        "_materialize_energy_barrier_neb_jobs",
        "_materialize_muon_relaxation_jobs",
        "_materialize_pristine_relaxation_jobs",
        "_materialize_quantum_corrections_jobs",
        "_materialize_uep_jobs",
        "_on_submit_cluster_from_muon_relaxation",
        "_on_submit_cluster_from_pristine_relaxation",
        "_on_submit_cluster_from_quantum_corrections",
        "_on_submit_cluster_from_uep",
        "_on_submit_cluster_neb_from_energy_barriers",
        "_on_submit_from_dashboard",
        "_on_submit_local_from_dashboard",
        "_on_submit_local_from_muon_relaxation",
        "_on_submit_local_from_pristine_relaxation",
        "_on_submit_local_from_quantum_corrections",
        "_on_submit_local_from_uep",
        "_on_submit_local_neb_from_energy_barriers",
    }
)


def _python_files(package: str) -> list[Path]:
    return sorted((SOURCE_ROOT / package).rglob("*.py"))


def _main_window_submit_handler_methods() -> set[str]:
    """Return ``MainWindow`` method names matching the submit/materialize patterns."""

    path = SOURCE_ROOT / "app" / "main_window.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main_window = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MainWindow"
    )
    return {
        member.name
        for member in main_window.body
        if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
        and (member.name.startswith("_on_submit_") or member.name.startswith("_materialize_"))
    }


def _mufinder_import_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    layers: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "mufinder":
                    layers.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = module.split(".")
            if len(parts) >= 2 and parts[0] == "mufinder":
                layers.add(parts[1])
    return layers


def test_agents_md_is_short_map_to_repository_system_of_record() -> None:
    agents = (PROJECT_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert len(agents.splitlines()) <= 80
    for doc in REQUIRED_AGENT_DOCS:
        assert doc in agents
        assert (PROJECT_ROOT / doc).is_file()


def test_harness_docs_are_indexed_by_sphinx() -> None:
    index = (PROJECT_ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    for doc in INDEXED_DOCS:
        assert f"\n   {doc}\n" in f"\n{index}\n"


def test_architecture_doc_lists_layer_packages() -> None:
    architecture = (PROJECT_ROOT / "docs" / "architecture.rst").read_text(encoding="utf-8")
    for layer in (
        "mufinder.domain",
        "mufinder.castep_io",
        "mufinder.persistence",
        "mufinder.science",
        "mufinder.workflow",
        "mufinder.services",
        "mufinder.app",
    ):
        assert layer in architecture


def test_package_layers_do_not_import_forbidden_higher_layers() -> None:
    violations: list[str] = []
    for package, forbidden_layers in FORBIDDEN_LAYER_IMPORTS.items():
        for path in _python_files(package):
            imported_layers = _mufinder_import_layers(path)
            illegal_layers = sorted(imported_layers & forbidden_layers)
            if illegal_layers:
                rel_path = path.relative_to(PROJECT_ROOT)
                violations.append(f"{rel_path}: imports {', '.join(illegal_layers)}")

    assert not violations, "\n".join(violations)


def test_main_window_submit_handlers_do_not_grow() -> None:
    """Pin MainWindow's per-workflow submit/materialize handler set as a ratchet.

    New ``_on_submit_*`` / ``_materialize_*`` methods must not appear on
    ``MainWindow``: workflow submission logic belongs on a
    ``mufinder.app.workflow_controllers`` controller. Each Phase 4 migration PR
    removes the migrated workflow's names from
    ``PINNED_MAIN_WINDOW_SUBMIT_HANDLERS`` (the set may only shrink).
    """

    actual = _main_window_submit_handler_methods()
    new_handlers = sorted(actual - PINNED_MAIN_WINDOW_SUBMIT_HANDLERS)
    assert not new_handlers, (
        "New per-workflow submit/materialize handler(s) appeared on MainWindow: "
        f"{new_handlers}. Put submission logic on a workflow_controllers controller "
        "instead of MainWindow."
    )
    removed_handlers = sorted(PINNED_MAIN_WINDOW_SUBMIT_HANDLERS - actual)
    assert not removed_handlers, (
        "Handler(s) removed from MainWindow but still pinned: "
        f"{removed_handlers}. Update PINNED_MAIN_WINDOW_SUBMIT_HANDLERS (shrink it) "
        "when a workflow migrates onto a controller."
    )


def test_repository_does_not_track_python_bytecode_artifacts() -> None:
    result = subprocess.run(
        ["git", "ls-files", "src", "tests", "docs"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked_bytecode = sorted(
        line
        for line in result.stdout.splitlines()
        if "/__pycache__/" in line or line.endswith((".pyc", ".pyo"))
    )
    assert not tracked_bytecode, "\n".join(tracked_bytecode)
