"""Every screenshot scene, merged from the example modules.

``EXAMPLES`` maps each example's name to its module, in render order, and
``SCREENSHOTS`` maps each screenshot's name to its scene. A screenshot name
belongs to exactly one example.
"""

from __future__ import annotations

from types import ModuleType

# The render module configures Qt and VTK, so it is imported before them.
from .render import Scene
from .examples import alpha_n2, cof2, fe, gdru2si2, mnsi

EXAMPLES: dict[str, ModuleType] = {
    module.__name__.rsplit(".", 1)[-1]: module for module in (cof2, fe, mnsi, alpha_n2, gdru2si2)
}


def _merge(examples: dict[str, ModuleType]) -> dict[str, Scene]:
    screenshots: dict[str, Scene] = {}
    for name, module in examples.items():
        repeated = sorted(screenshots.keys() & module.SCENES.keys())
        if repeated:
            raise ValueError(f"The {name} example repeats screenshot names: {', '.join(repeated)}")
        screenshots.update(module.SCENES)
    return screenshots


SCREENSHOTS: dict[str, Scene] = _merge(EXAMPLES)
