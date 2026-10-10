"""Set up a ``MainWindow`` with a project open, for the app integration tests.

The ``main_window`` and ``project_with_host`` fixtures in ``tests/conftest.py``
build on these helpers; a module that needs its own project name or host binds
them with ``functools.partial``. ``MainWindow`` is imported inside each helper so
importing this module stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from tests.support.structures import host_structure

if TYPE_CHECKING:
    from mufinder.app.main_window import MainWindow


def open_project_with_host(
    window: MainWindow,
    tmp_path: Path,
    *,
    filename: str = "project.mufinder",
    name: str = "Project",
    host: Structure | None = None,
) -> tuple[ProjectService, ProjectDocument, Structure]:
    """Create a project under ``tmp_path``, load ``host`` as its host structure, and show it in ``window``.

    ``host`` defaults to :func:`tests.support.structures.host_structure`.
    """

    service = ProjectService()
    document = service.create_project(tmp_path / filename, name)
    host = host_structure() if host is None else host
    service.add_structure(document, host)
    document.project.host_structure_id = host.id
    window._document = document
    window._refresh_ui_after_project_change()
    return service, document, host


def window_with_host(
    tmp_path: Path,
    **project: object,
) -> tuple[MainWindow, ProjectService, ProjectDocument, Structure]:
    """A new ``MainWindow`` with a host-structure project open; see :func:`open_project_with_host`."""

    from mufinder.app.main_window import MainWindow

    window = MainWindow()
    service, document, host = open_project_with_host(window, tmp_path, **project)  # type: ignore[arg-type]
    return window, service, document, host
