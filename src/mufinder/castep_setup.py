"""Helpers for persisted CASTEP executable and keyword metadata configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import shutil

from PySide6.QtCore import QSettings


CASTEP_EXECUTABLE_KEY = "castep/executable_path"
CASTEP_KEYWORDS_JSON_KEY = "castep/keywords_json_path"
CASTEP_MPIRUN_KEY = "castep/mpirun_path"


@dataclass(frozen=True, slots=True)
class CastepSetup:
    """Persisted CASTEP integration paths."""

    executable_path: str = ""
    keywords_json_path: str = ""
    mpirun_path: str = ""


def load_castep_setup(settings: QSettings) -> CastepSetup:
    """Load CASTEP integration settings from QSettings."""

    executable_path = str(settings.value(CASTEP_EXECUTABLE_KEY, "", str) or "").strip()
    keywords_json_path = str(settings.value(CASTEP_KEYWORDS_JSON_KEY, "", str) or "").strip()
    mpirun_path = str(settings.value(CASTEP_MPIRUN_KEY, "", str) or "").strip()
    return CastepSetup(
        executable_path=executable_path,
        keywords_json_path=keywords_json_path,
        mpirun_path=mpirun_path,
    )


def save_castep_setup(settings: QSettings, setup: CastepSetup) -> None:
    """Persist CASTEP integration settings to QSettings."""

    settings.setValue(CASTEP_EXECUTABLE_KEY, setup.executable_path.strip())
    settings.setValue(CASTEP_KEYWORDS_JSON_KEY, setup.keywords_json_path.strip())
    settings.setValue(CASTEP_MPIRUN_KEY, setup.mpirun_path.strip())


def discover_mpirun() -> str | None:
    """Locate an ``mpirun`` executable on this machine if available."""

    discovered = shutil.which("mpirun")
    if discovered:
        return str(Path(discovered).expanduser().resolve())

    candidates = (
        "/usr/bin/mpirun",
        "/usr/local/bin/mpirun",
        "/opt/homebrew/bin/mpirun",
        "/opt/local/bin/mpirun",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return None


def apply_castep_environment(setup: CastepSetup) -> None:
    """Expose persisted CASTEP configuration through environment variables."""

    keywords_json = setup.keywords_json_path.strip()

    if keywords_json:
        os.environ["MUFINDER_CASTEP_KEYWORDS_JSON"] = keywords_json
    else:
        os.environ.pop("MUFINDER_CASTEP_KEYWORDS_JSON", None)


def apply_saved_castep_environment(settings: QSettings | None = None) -> CastepSetup:
    """Load persisted CASTEP settings and publish them into the process environment."""

    effective_settings = settings if settings is not None else QSettings("MuFinder", "MuFinder")
    setup = load_castep_setup(effective_settings)
    apply_castep_environment(setup)
    return setup


def build_castep_keyword_command(executable_path: str, mpirun_path: str = "") -> str:
    """Build the shell command ASE runs to query CASTEP help.

    ASE's ``create_castep_keywords`` invokes ``<command> -help ...`` through a
    shell. MPI-built CASTEP binaries (e.g. ``castep.mpi``) hang indefinitely when
    executed directly, because they block waiting for an MPI runtime. When an MPI
    launcher is configured we therefore wrap the executable in ``mpirun -np 1`` —
    which also runs serial binaries correctly — mirroring the cluster keyword
    generation path.
    """

    executable = executable_path.strip()
    mpirun = mpirun_path.strip()
    if mpirun:
        return " ".join(shlex.quote(part) for part in (mpirun, "-np", "1", executable))
    return executable


def generate_castep_keywords_json(
    *, executable_path: str, output_path: str | Path, mpirun_path: str = ""
) -> Path:
    """Generate a CASTEP keyword JSON file using ASE's supported extraction helper."""

    executable = executable_path.strip()
    if not executable:
        raise ValueError("A CASTEP executable path is required to generate castep_keywords.json.")

    output = Path(output_path).expanduser()
    if output.suffix.lower() != ".json":
        raise ValueError("CASTEP keywords output path must end with .json.")

    output.parent.mkdir(parents=True, exist_ok=True)

    from ase.calculators.castep import create_castep_keywords

    create_castep_keywords(
        build_castep_keyword_command(executable, mpirun_path),
        filename=output.name,
        path=str(output.parent),
        force_write=True,
    )

    if not output.exists():
        raise RuntimeError(f"CASTEP keyword generation did not produce {output}.")

    return output.resolve()
