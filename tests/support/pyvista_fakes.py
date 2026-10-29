"""Light Qt fakes for PyVista's ``QtInteractor`` and the structure viewer's canvases.

``tests/conftest.py`` installs ``FakePyVistaCanvas`` and ``FakePyVistaTriadCanvas``
in every Qt test, since real VTK render windows are slow and crash-prone under
the test runner. ``FakePlotter`` also stands in for ``QtInteractor`` itself in
tests that exercise the production canvases.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np

import pytest
from PySide6.QtWidgets import QWidget

from mufinder.app import atomic_structure_viewer as _viewer_module

# Imported with this module, before the autouse fixture swaps in the fakes below.
_PRODUCTION_CANVASES = (_viewer_module._PyVistaCanvas, _viewer_module._PyVistaTriadCanvas)


class FakeRenderTimer:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


class FakeProperty:
    def __init__(self) -> None:
        self.opacity = 1.0

    def SetOpacity(self, value: float) -> None:
        self.opacity = float(value)


class FakeActor:
    def __init__(self, mesh=None, kwargs: dict | None = None) -> None:
        self.mesh = mesh
        self.kwargs = {} if kwargs is None else dict(kwargs)
        self._property = FakeProperty()
        # Like a PyVista actor's ``user_matrix``: the transform drawn on top of ``mesh``.
        self.user_matrix = np.eye(4)

    @property
    def mapper(self):
        """Like a PyVista actor's mapper: its ``dataset`` is the mesh the actor draws."""

        return SimpleNamespace(dataset=self.mesh)

    def GetProperty(self):
        return self._property


class FakeRenderer:
    def AutomaticLightCreationOff(self) -> None:
        return

    def UseImageBasedLightingOff(self) -> None:
        return

    def UseSphericalHarmonicsOff(self) -> None:
        return

    def SetEnvironmentTexture(self, _texture) -> None:
        return


class FakeInteractor:
    def __init__(self) -> None:
        self.picker = None
        self.interactor = SimpleNamespace(GetShiftKey=lambda: 0)
        self._event_position = (0, 0)

    def add_observer(self, _event_name: str, _callback) -> None:
        return

    def get_event_position(self) -> tuple[int, int]:
        return self._event_position

    def get_poked_renderer(self):
        return "renderer"


class FakePlotter(QWidget):
    """A ``QtInteractor`` stand-in: records actors and draws nothing."""

    def __init__(self, parent: QWidget | None = None, **_kwargs: object) -> None:
        super().__init__(parent)
        self.renderer = FakeRenderer()
        self.camera = SimpleNamespace(
            position=(8.0, 8.0, 8.0),
            focal_point=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            parallel_scale=1.0,
        )
        self.iren = FakeInteractor()
        self.render_timer = FakeRenderTimer()
        self._actors: list[object] = []

    def setPalette(self, _palette) -> None:
        return

    def set_background(self, *_args, **_kwargs) -> None:
        return

    def enable_parallel_projection(self) -> None:
        return

    def enable_anti_aliasing(self, *_args, **_kwargs) -> None:
        return

    def disable_shadows(self) -> None:
        return

    def disable_eye_dome_lighting(self) -> None:
        return

    def remove_environment_texture(self) -> None:
        return

    def remove_all_lights(self) -> None:
        return

    def add_light(self, light) -> object:
        actor = FakeActor(light)
        self._actors.append(actor)
        return actor

    def clear(self) -> None:
        self._actors.clear()

    def add_mesh(self, mesh, **kwargs):
        actor = FakeActor(mesh, kwargs)
        self._actors.append(actor)
        return actor

    def remove_actor(self, actor, *, render: bool = True) -> None:
        del render
        if actor in self._actors:
            self._actors.remove(actor)

    def add_text(self, *args, **kwargs):
        actor = FakeActor(args, kwargs)
        self._actors.append(actor)
        return actor

    def add_point_labels(self, *args, **kwargs):
        actor = FakeActor(args, kwargs)
        self._actors.append(actor)
        return actor

    def enable_custom_trackball_style(self, **_kwargs) -> None:
        return

    def reset_camera_clipping_range(self) -> None:
        return

    def render(self) -> None:
        return

    def close(self) -> None:
        return


class FakePyVistaCanvas(QWidget):
    """Like the production canvas: no plotter until the viewer's first show."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plotter: FakePlotter | None = None

    @property
    def plotter(self):
        return self._plotter

    def ensure_plotter(self):
        if self._plotter is None:
            self._plotter = FakePlotter(self)
        return self._plotter

    def draw_idle(self) -> None:
        if self._plotter is not None:
            self._plotter.render()


class FakePyVistaTriadCanvas(QWidget):
    """Like the production triad: no plotter until the viewer's first show."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._plotter: FakePlotter | None = None
        self._scene = None
        self._light_direction = None
        self.setFixedWidth(132)
        self.setFixedHeight(130)

    @property
    def plotter(self):
        return self._plotter

    def ensure_plotter(self):
        if self._plotter is None:
            self._plotter = FakePlotter(self)
        return self._plotter

    def set_scene(self, **kwargs) -> None:
        self._scene = dict(kwargs)

    def set_light_direction(self, direction) -> None:
        self._light_direction = direction

    def draw_idle(self) -> None:
        if self._plotter is not None:
            self._plotter.render()


def count_created_interactors(monkeypatch: pytest.MonkeyPatch) -> list[FakePlotter]:
    """Give viewers their production canvases, building a ``FakePlotter`` per interactor.

    Returns the list each new interactor (each would-be VTK render window) is
    appended to, so a test can count the render windows a viewer creates.
    """

    import pyvista as pv

    created: list[FakePlotter] = []

    class _CountedInteractor(FakePlotter):
        def __init__(self, parent: QWidget | None = None, **kwargs: object) -> None:
            super().__init__(parent, **kwargs)
            created.append(self)

    canvas, triad_canvas = _PRODUCTION_CANVASES
    monkeypatch.setattr(_viewer_module, "_PyVistaCanvas", canvas)
    monkeypatch.setattr(_viewer_module, "_PyVistaTriadCanvas", triad_canvas)
    monkeypatch.setattr(_viewer_module, "_pyvista_modules", lambda: (pv, _CountedInteractor))
    return created


def count_scene_renders(monkeypatch: pytest.MonkeyPatch) -> Counter:
    """Count full scene rebuilds (``_render_pyvista`` calls), keyed by the viewer."""

    counts: Counter = Counter()
    original = _viewer_module.AtomicStructureViewer._render_pyvista

    def counted(viewer: _viewer_module.AtomicStructureViewer) -> None:
        counts[viewer] += 1
        original(viewer)

    monkeypatch.setattr(_viewer_module.AtomicStructureViewer, "_render_pyvista", counted)
    return counts


def install_fake_picker(
    viewer: _viewer_module.AtomicStructureViewer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    picked_point: tuple[float, float, float] | None,
    shift: bool = False,
    position: tuple[int, int] = (18, 24),
):
    """Give ``viewer`` a canvas whose VTK picker hits ``picked_point``, or nothing for ``None``.

    The viewer stops storing its view and rendering. Returns the picker, which
    records each ``Pick`` call, and a ``{"value": (x, y)}`` dict a test edits to
    move the event position the interactor reports.
    """

    class _FakePicker:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, object]] = []

        def Pick(self, x: int, y: int, z: int, renderer) -> int:
            self.calls.append((x, y, z, renderer))
            return 0 if picked_point is None else 1

        def GetPickPosition(self):
            if picked_point is None:
                return (0.0, 0.0, 0.0)
            return picked_point

    picker = _FakePicker()
    current_position = {"value": position}
    monkeypatch.setattr(viewer, "_store_view", lambda *_args: None)
    monkeypatch.setattr(viewer, "_render", lambda: None)
    viewer._canvas = SimpleNamespace(
        plotter=SimpleNamespace(
            iren=SimpleNamespace(
                interactor=SimpleNamespace(GetShiftKey=lambda: int(shift)),
                picker=picker,
                get_event_position=lambda: current_position["value"],
                get_poked_renderer=lambda: "renderer",
            )
        )
    )
    return picker, current_position
