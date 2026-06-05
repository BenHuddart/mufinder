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


def _python_files(package: str) -> list[Path]:
    return sorted((SOURCE_ROOT / package).rglob("*.py"))


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
