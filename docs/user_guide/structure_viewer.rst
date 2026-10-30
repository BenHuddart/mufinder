Structure Viewer
================

Every 3D view of a structure in MuFinder is the same viewer: the
:doc:`/user_guide/host_structure` panel, the :doc:`/user_guide/supercells`
preview, the ``Preview Muon Sites`` view of :doc:`/user_guide/site_generation`,
the :doc:`/user_guide/uep_workflow` analysis and clustering windows, the
:doc:`/user_guide/structural_relaxation_analysis` workspaces and dialogs, the
muon-sites summary, the phonon modes of :doc:`/user_guide/quantum_corrections`,
the Magnetic Structure Builder of :doc:`/user_guide/magnetic_fields`, and the
path viewer and NEB inspector of :doc:`/user_guide/energy_barriers`. This page
describes its controls and what it draws; each workflow page describes only what
that workflow adds to it and links here.

.. figure:: ../_generated/screenshots/fe_host_structure.webp
   :alt: The Host Structure page with bcc Fe loaded: the viewer on the right
         shows the cell looking along a, with parallel spin arrows on both
         atoms, the Cell, Bonds and Spins toggles and the Along a, b, c and
         Reset View buttons above it, and the axis triad at its bottom left.

   The viewer on the ``Host Structure`` panel with the bcc Fe host of
   :doc:`/user_guide/examples/fe`, looking along a with ``Spins`` on.

The controls
------------

The controls sit above the view:

- ``Cell`` shows the unit-cell outline. It is on by default.
- ``Bonds`` shows bonds between neighbouring atoms (see `What the viewer
  draws`_). It is off by default.
- ``Spins`` shows the spin arrows (see `Spins`_).
- ``Along:`` ``a``, ``b`` and ``c`` turn the camera to look along that lattice
  vector (their accessible names are ``Look Along a``, ``Look Along b`` and
  ``Look Along c``). In a non-orthogonal cell they follow the lattice vectors,
  not the Cartesian axes.
- ``Reset View`` returns to the default oblique orientation and zoom.

A hint line under the controls lists the mouse actions:

- left-click an atom to select it;
- ``Shift``-click an atom to add it to the selection or remove it;
- left-click empty space to clear the selection;
- right-drag to rotate, middle-drag to pan, and scroll to zoom.

A drag does not change the selection. The camera stays where you left it when
the viewer redraws because a toggle, a spin, a selection, a slice plane, a
marker or an isosurface changed, and when a page moves the same atoms, as the
NEB inspector does when it steps through the images of a band and the phonon
view does when it animates a mode. Loading a different structure returns the
view to the default orientation, clears the selection, and turns ``Spins`` on
or off to match the new structure's spins.

The selection
~~~~~~~~~~~~~

The line under the view describes the selection. One selected atom shows its
number and element, its fractional position ``f`` and its Cartesian position
``r`` in Å (``Selected atom: #2 Co | f=(…) | r=(…) A``). Several selected atoms
show their count and the first four (``Selected atoms: 6 | #1 Co, #2 Co, …``),
with ``+N more`` for the rest. A line too long for the viewer's width ends in
an ellipsis, with the full text in its tooltip. Selected atoms are highlighted
in the accent colour. On the ``Host Structure`` panel the atom table and the
viewer share the selection.

What the viewer draws
---------------------

The projection is orthographic, so lengths along the screen are true whatever
their depth. Fitting the view accounts for the full atom spheres, not just
their centres, so atoms at the edges are not clipped.

- **Atoms** are spheres in the Jmol element colours, sized from their covalent
  radii. A muon is a small pink sphere.
- **The cell** is drawn as one outline of the twelve cell edges.
- **Bonds** join two atoms when their distance is less than the sum of their
  covalent radii, each scaled by 1.15. Each half of a bond takes the colour of
  the atom at its end. Only bonds between atoms in the same cell are drawn: a
  bond that would cross a cell face is not.
- **The axis triad** at the bottom left, in a narrow column beside the view,
  shows the directions of ``a``, ``b`` and ``c`` and turns with the camera.

The viewer draws the atoms the structure lists and does not repeat them into
neighbouring cells; a page that previews several cells, such as the supercell
preview or the magnetic structure preview, hands the viewer the repeated
structure.

Spins
~~~~~

Each atom with a non-zero spin or moment carries a 3D arrow, a shaft and a cone
centred on the atom. A scalar spin (in ``hbar/2``, as on the ``Host Structure``
panel) points along the Cartesian ``z`` axis, up for a positive spin and down
for a negative one; a moment vector, such as those the Magnetic Structure
Builder previews, points along the moment. The arrow is blue when its ``z``
component is positive or zero and red when it is negative, and it is longer for
a larger moment and on a larger atom.

``Spins`` turns on by itself when a structure is loaded with any non-zero spin,
or when a spin is edited to a non-zero value, and is off when a structure with
no spins is loaded. Editing the spins back to zero leaves it as it is. While
spins are shown the atoms are drawn half transparent, so the arrows inside
them stay visible. Bonds are always slightly translucent, and more so while
spins are shown.

Overlays added by the workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some pages draw more than the structure. The camera and the selection stay in
place when an overlay changes.

- **Point markers** show candidate, clustered or relaxed muon positions in
  :doc:`/user_guide/site_generation`, the :doc:`/user_guide/uep_workflow`
  windows and :doc:`/user_guide/structural_relaxation_analysis`, and the start
  and end sites of a hop in the path viewer of
  :doc:`/user_guide/energy_barriers`.
- **Isosurfaces and slice planes** show the unperturbed electrostatic
  potential on the ``Isosurfaces`` and ``Slices`` tabs of the UEP analysis
  window. ``Hide atoms in front of the planes`` there hides the atoms (and
  their bonds and spin arrows) on the camera side of every slice plane. See
  :doc:`/user_guide/uep_workflow`.
- **Arrows** show the atomic displacements of a phonon mode in
  :doc:`/user_guide/quantum_corrections`, and the hop from the start site to the
  end site in the path viewer of :doc:`/user_guide/energy_barriers`.

Theme
-----

The viewer's background, cell outline, selection highlight, default isosurface
colour and axis triad follow the light or dark application palette, and change with
it when you switch themes. The atom and spin colours stay the same in both
themes.

Width
-----

The controls fit narrow panes. The display toggles and the orientation buttons
share one line when there is room and wrap onto two when there is not, and the
hint line wraps too. The viewer needs about 300 px of width, so a plot beside it
in an analysis dialog keeps the width it needs. A page can ask for more: the
``Host Structure`` panel keeps its viewer at least 380 px wide.

Limitations
-----------

- The projection is orthographic only; there is no perspective view.
- Bonds that cross a cell face are not drawn (see `What the viewer draws`_).
- The viewer does not repeat atoms into neighbouring cells by itself.

For developers
--------------

The viewer is ``mufinder.app.atomic_structure_viewer.AtomicStructureViewer``,
built on the PyVista/VTK and ``pyvistaqt`` stack. It creates its render windows
on first show and draws each change with one render; see "One render per
structure-viewer change" in :doc:`/architecture`.
