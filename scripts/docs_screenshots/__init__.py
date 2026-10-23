"""Render the documentation screenshots from the running application.

The screenshots are build outputs, not sources: ``make docs`` renders them
first and Sphinx copies the images into the built site, so no image is
committed. Each screenshot is a :class:`~docs_screenshots.render.Scene`: a
project builder writes a small saved project, the renderer opens it in a real
``MainWindow`` on one navigator route, optionally drives the window, and grabs
a target: the whole window, one page widget, a dock, a modal dialog, or a
non-modal window. No scene runs CASTEP.

The package:

- :mod:`docs_screenshots.render` configures Qt and VTK for a headless run and
  holds the machinery every scene shares: the off-screen 3D viewer, the grab
  targets, :func:`~docs_screenshots.render.render` and its project cache.
- :mod:`docs_screenshots.synthetic` writes CASTEP output files for the
  projects whose pages show finished calculations. It imports no Qt.
- :mod:`docs_screenshots.examples` holds one module per example project:
  its builders, the steps only its scenes use, and its ``SCENES``.
- :mod:`docs_screenshots.registry` merges every example's scenes.
- :mod:`docs_screenshots.watchdog` gives each scene a time limit, after which
  the renderer names the scene, dumps every thread's stack and fails. It
  imports no Qt.

Importing this package imports nothing; :mod:`docs_screenshots.render` sets
the environment when it is first imported. Run the renderer through the
Makefile, or with ``scripts/render_docs_screenshots.py``::

    make screenshots
    python scripts/render_docs_screenshots.py --list
    python scripts/render_docs_screenshots.py --example cof2
    python scripts/render_docs_screenshots.py project_home journal_page
"""
