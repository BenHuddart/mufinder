:orphan:

PyVista Migration Plan
======================

Goal
----

Port the interactive structure viewer and the UEP isosurface pipeline from the
current Matplotlib-based implementation to PyVista while preserving the
existing user-facing behavior, scientific data flow, and ASE-like structure
representation.

Current Baseline
----------------

The current implementation is split across three main areas:

- ``src/mufinder/app/atomic_structure_viewer.py`` owns the 3D structure
  renderer, camera controls, picking, selection display, bond/spin overlays,
  crystallographic triad, and isosurface overlay.
- ``src/mufinder/app/host_structure_widget.py``,
  ``src/mufinder/app/supercell_dialog.py``, and
  ``src/mufinder/app/uep_wizard.py`` embed the viewer in the main host,
  supercell preview, batch-search, and isosurface workflows.
- ``src/mufinder/science/uep.py`` computes ``UepPotentialGrid`` objects and
  currently extracts isosurfaces in pure Python before the viewer renders them.

The existing tests already encode a large part of the feature contract we need
to preserve, especially in:

- ``tests/app/test_host_structure_widget.py``
- ``tests/app/test_supercell_dialog.py``
- ``tests/app/test_uep_wizard.py``
- ``tests/science/test_uep.py``
- ``tests/packaging/test_pyinstaller_spec.py``

Behavior That Must Be Preserved
-------------------------------

The port should keep the public viewer API stable so existing callers can
continue using:

- ``set_structure(...)``
- ``set_spins(...)``
- ``set_isosurface(...)``
- ``select_atom(...)``
- ``atom_selected``

The following user-visible behavior should remain unchanged:

- ASE-like atom appearance using Jmol colors, covalent-radii scaling, and the
  existing MuFinder muon override for species ``X``.
- Orthographic projection and the current named views (``X``, ``Y``, ``Z``,
  reset).
- Current interaction model: left-click to select atoms, left-click empty space
  to clear selection, right-drag to rotate, middle-drag to pan, mouse wheel to
  zoom.
- Unit-cell display, bond display, and spin-arrow display, including the
  current atom-transparency change when spin arrows are visible.
- The synchronized crystallographic triad under the main structure viewport.
- Current selection label formatting in both fractional and Cartesian
  coordinates.
- UEP isosurface color/opacity defaults and the use of the source structure's
  spin data inside the UEP wizard.
- Batch-search overlays that add muon markers without changing the underlying
  host structure semantics.
- Persisted UEP potential-grid caching and XSF export.

Two details are especially important for parity:

1. The existing visual language was deliberately tuned to resemble ASE rather
   than a generic VTK scene.
2. The UEP workflow needs both faster rendering and faster isosurface
   extraction for large grids, so replacing only the final draw call is not
   enough.

Target Architecture
-------------------

Keep the top-level widget contract stable and move the implementation behind a
backend-neutral scene layer.

Recommended structure:

- Keep ``AtomicStructureViewer`` as the public Qt widget used by the rest of
  the application.
- Introduce backend-neutral scene/state helpers that describe:

  - atom positions, colors, radii, and transparency;
  - unit-cell edges;
  - bond segments;
  - spin-arrow glyphs;
  - selection highlight state;
  - optional isosurface mesh;
  - camera state and named views.

- Add a PyVista backend widget that renders this scene model through
  ``pyvistaqt.QtInteractor``.
- Move Matplotlib-specific code behind a temporary compatibility backend only if
  needed during the transition; otherwise replace it directly once parity tests
  exist.

Why PyVista ``StructuredGrid`` For UEP
--------------------------------------

The UEP potential grid lives in a general crystal lattice, not just orthogonal
simulation cells. The replacement contour path should therefore build a
PyVista ``StructuredGrid`` from the lattice-transformed grid points and use the
VTK contour filter rather than assuming axis-aligned ``ImageData``.

That gives us:

- correct geometry for non-orthogonal cells;
- a compiled contour implementation instead of the current pure-Python
  marching-tetrahedra loop;
- direct conversion back into the existing ``UepIsosurfaceMesh`` payload for
  persistence and testing.

Recommended Implementation Phases
---------------------------------

Phase 1: Stabilize the contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Extract backend-neutral helpers from ``atomic_structure_viewer.py`` for atom
  styling, camera state, selection text, and scene assembly.
- Add tests around those helpers so we stop depending on Matplotlib internals
  for most behavior checks.
- Keep all current call sites unchanged.

Phase 2: Introduce the PyVista viewer backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Embed a PyVista renderer using ``pyvistaqt.QtInteractor`` inside
  ``AtomicStructureViewer``.
- Recreate the current layers with PyVista actors:

  - atoms as sphere glyphs;
  - bonds as tubes or cylinders, still split by endpoint color;
  - spins as shaft/head glyphs with the current signed color mapping;
  - unit-cell edges as line or tube actors;
  - selection halo/outline actor;
  - orientation/axes widget for the crystallographic triad.

- Preserve the current control row, interaction hint text, selection label, and
  named-view buttons.

Phase 3: Port UEP isosurface extraction to VTK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Replace the current Python ``extract_isosurface(...)`` implementation with a
  VTK-backed contour path driven by PyVista.
- Preserve the public return type ``UepIsosurfaceMesh`` so the UEP wizard,
  state restoration, and persistence code do not need a broad rewrite.
- Keep ``preview_potential_grid(...)`` in place so the slider can still drive a
  responsive preview mesh before falling back to the full grid if needed.

Phase 4: Remove Matplotlib-specific viewer code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Delete the Matplotlib 3D viewer implementation once the PyVista path has
  feature parity and packaging support.
- Retain Matplotlib where it still belongs today, such as the line and plane
  plots in the UEP wizard.

Concrete Work Items
-------------------

Viewer and scene model
~~~~~~~~~~~~~~~~~~~~~~

- Introduce a backend-neutral viewer state module rather than keeping
  rendering-specific data structures inside ``AtomicStructureViewer``.
- Store camera state in a renderer-independent form so named views, restore,
  and reset behavior stay testable.
- Add explicit picking helpers so selection behavior does not depend on private
  renderer objects.

UEP isosurface path
~~~~~~~~~~~~~~~~~~~

- Add a conversion from ``UepPotentialGrid`` to a PyVista structured dataset.
- Use the contour filter to generate one isosurface for a requested energy
  level.
- Convert the resulting ``PolyData`` back into ``UepIsosurfaceMesh``.
- Keep serialization and restoration logic unchanged by continuing to persist
  vertices and triangle faces rather than PyVista objects.

Packaging
~~~~~~~~~

- Add ``pyvista`` and ``pyvistaqt`` dependencies to ``pyproject.toml``.
- Update ``packaging/pyinstaller/mufinder.spec`` so VTK/PyVista runtime data,
  shared libraries, and hidden imports are collected for frozen builds.
- Extend packaging tests to assert that the PyInstaller spec includes the new
  PyVista/VTK collection hooks.

Testing Strategy
----------------

The new implementation should ship with comprehensive tests in four layers.

1. Backend-neutral unit tests

- Atom-style parity: Jmol colors, covalent-radii scaling, and muon overrides.
- Spin parsing and automatic visibility defaults.
- Selection-label formatting.
- Camera named-view mapping and reset behavior.
- Isosurface attach/clear state handling.

2. Viewer integration tests

- Host structure rendering still exposes the same controls and default states.
- Table-selection and viewer-selection synchronization still works.
- Blank-space clicks still clear selection.
- Bond transparency still tracks spin visibility.
- Supercell preview still renders correctly.
- UEP isosurface viewer still restores cached overlays from saved dialog state.

3. UEP scientific tests

- ``preview_potential_grid(...)`` still caps resolution and preserves the value
  range.
- ``extract_isosurface(...)`` still rejects out-of-range levels.
- New contour extraction tests should cover both orthogonal and skewed cells.
- Add parity checks against the current implementation on representative
  fixtures using mesh-level invariants such as triangle count ranges, bounds,
  and contour level rather than exact triangle ordering.

4. Headless rendering and packaging tests

- Add PyVista off-screen smoke tests so the viewer can render in CI without a
  visible display.
- Add a packaged-build check that verifies the frozen app still includes the
  required VTK/PyVista assets.

Documentation Deliverables
--------------------------

The implementation PR should update:

- ``docs/implemented_functionality.rst`` to describe the PyVista-based viewer
  and any preserved interaction details.
- ``docs/uep_workflow.rst`` to document the responsive isosurface preview path
  and clarify that the UEP workflow still exports XSF from the full cached
  potential grid.
- ``README.md`` and any developer setup notes to include the new rendering
  dependencies and any headless-testing requirements.
- Packaging notes if extra platform-specific caveats appear for frozen builds.

Acceptance Criteria
-------------------

The port is complete when all of the following are true:

- Large structures render more responsively than the current Matplotlib viewer.
- The UEP isosurface slider remains responsive when previewing a cached
  full-cell potential grid.
- Existing viewer workflows keep their current controls and semantics.
- The ASE-like appearance choices survive the renderer swap.
- All existing viewer/UEP tests pass after being updated for backend-neutral
  assertions where necessary.
- New PyVista-specific tests cover rendering smoke paths, UEP contour
  extraction, and packaging.
- Frozen application builds still launch and render at least one structure and
  one UEP isosurface successfully.

Primary Risks And Mitigations
-----------------------------

Headless CI and local test environments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Risk:
  VTK-backed rendering often behaves differently from pure widget tests.

Mitigation:
  Standardize on off-screen rendering for test paths and keep rendering smoke
  tests small and explicit.

Packaging complexity
~~~~~~~~~~~~~~~~~~~~

Risk:
  PyVista/VTK may work in development but fail once frozen into the packaged
  desktop application.

Mitigation:
  Treat PyInstaller support as part of the migration, not post-port cleanup.

Interaction drift
~~~~~~~~~~~~~~~~~

Risk:
  Default PyVista mouse bindings differ from the current viewer hints and
  muscle memory.

Mitigation:
  Explicitly remap or wrap interactions so the current instructions remain
  truthful.

Skew-cell contour geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~

Risk:
  Using the wrong PyVista grid type would distort isosurfaces for non-orthogonal
  cells.

Mitigation:
  Build the contour dataset from lattice-transformed structured points rather
  than axis-aligned image data.

Recommended First PR Split
--------------------------

To keep reviewable change sets, split the work into three PRs:

1. Scene-model extraction and backend-neutral tests.
2. PyVista viewer backend with host/supercell/batch parity.
3. UEP contour extraction, packaging updates, and final documentation refresh.
