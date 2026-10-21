Your first project
==================

This tutorial takes a new project from an empty folder to a relaxed muon
stopping site in silicon, with an estimate of the local field the muon feels
there. Each stage below is a short route through one page of the application;
the linked user-guide page explains every control and option on it.

The route is:

1. `Create a project and set up CASTEP`_
2. `Import the host structure`_
3. `Save a supercell`_
4. `Converge the cutoff and k-point grid`_
5. `Optional: relax the pristine host`_
6. `Generate candidate sites and relax them`_
7. `Find the distinct sites and pin them`_
8. `Estimate the local field`_

Before you start
----------------

- **MuFinder installed.** See :doc:`/getting_started/installation`.
- **CASTEP.** A CASTEP executable on this computer, or an HPC cluster that can
  run it. MuFinder does not include CASTEP.
- **A structure file for silicon.** A CIF of diamond silicon from a structure
  database will do. The screenshots on this page use silicon's two-atom
  primitive cell; a file holding the eight-atom conventional cell gives larger
  atom counts but the same steps.

The screenshots from the pristine relaxation onwards come from a silicon
project whose CASTEP outputs are synthetic, so every number in them is
illustrative. Your own calculations will give different values.

Create a project and set up CASTEP
----------------------------------

1. Start MuFinder (``python -m mufinder.main``, or ``mufinder`` for a source
   install).
2. Choose ``File > New Project`` (or click ``Create New Project`` on the welcome
   page). Enter a ``Project name:``, then choose where it goes in ``Select
   parent folder``. MuFinder creates a ``<name>.mufinder`` folder there and
   opens the project home.
3. Open ``Setup > CASTEP...`` and install a keyword catalog: pick the target in
   ``Generate from`` and click ``Generate Keyword Catalog``, or click ``Use
   Existing File…`` if you already have a ``castep_keywords.json``.
4. To run on this computer, fill in ``CASTEP executable`` and ``MPI launcher
   (mpirun)`` (``Auto-detect`` searches for ``mpirun``), then click ``OK``.
5. To run on a cluster instead, add a profile under ``Setup > HPC
   Clusters...``.

When the catalog is installed the status bar reads ``CASTEP keywords:``
followed by the CASTEP version. The details, including what happens without a
catalog, are in :doc:`/getting_started/castep_setup`; cluster profiles are
described under *Running on a cluster* in :doc:`/getting_started/running_jobs`.

The project home charts every stage and says what each one needs. Click the
home button above the project navigator at any point to see where the project
stands (see :doc:`/user_guide/project_home`).

Import the host structure
-------------------------

1. Select ``Host Structure`` in the project navigator.
2. Click ``Load Structure (.cif/.mcif/.cell)`` and choose your silicon file.
3. Check the summary, the atom table and the 3D viewer: two ``Si`` atoms for
   the primitive cell.

The imported structure becomes the active host, which every later stage works
on. Silicon is not magnetic, so leave the ``Spin (hbar/2)`` column at zero.
See :doc:`/user_guide/host_structure`; its figure, under *The host-structure
panel*, shows the panel with a different material loaded.

Save a supercell
----------------

A muon relaxation needs a cell large enough that the muon does not interact
with its own periodic images.

1. Select ``Supercell Editor`` in the navigator and click ``Open Supercell
   Editor``.
2. Click ``New``, then ``Suggest…``. The suggestions list diagonal supercells
   with their atom count and minimum image distance, and mark one
   ``(Recommended)``.
3. Highlight a cell and click ``Use This Supercell``. The preview and the
   status line show the result.
4. Name it and click ``Save``. It appears in ``Saved Supercells``.

The relaxations in the screenshots below ran in a 2x2x2 supercell of the
primitive cell (16 atoms) named ``Si 2x2x2``, which keeps the example cheap;
follow the ``(Recommended)`` cell for production work. See
:doc:`/user_guide/supercells` (*Suggesting a supercell*).

Converge the cutoff and k-point grid
------------------------------------

Every later calculation inherits the project's XC functional, cutoff energy
and k-point grid, so settle them first.

1. Select ``Cutoff Energy`` under ``Convergence Testing``. On the ``Project
   defaults`` strip click ``Edit…``, choose an ``XC functional`` (the
   screenshots use ``PBE``) and click ``OK``.
2. Leave ``Single unit cell`` selected under ``Simulation Cell``.
3. Click ``Generate From Range...`` and accept ``Start (eV)``, ``Stop (eV)``
   and ``Step (eV)``, or change them.
4. Click ``Run All Locally`` or ``Run All on Cluster`` and fill in the
   submission dialog. Each row's ``Status`` follows its job, and the ``Job
   Dashboard`` at the bottom of the window lists them too.
5. When the rows have completed, click ``Examine Cutoff Convergence``, then
   ``Analyse Convergence``, then the ``Set <value> eV as Project Default
   Cutoff`` button.
6. Select ``K-Point Grid``, click ``Generate From Spacing...`` and accept the
   ``Minimum spacing (1/Ang)``. The new rows use the cutoff you just adopted.
7. Run them, then click ``Examine K-Point Convergence``, ``Analyse
   Convergence`` and ``Set <grid> as Project Default K-Point Grid``.
8. Select ``Convergence Testing`` to check both adopted values and their
   pinned plots.

If no value is recommended, the result line says which criterion did not
converge; extend the sweep and analyse again. Submitting jobs, the submission
dialogs and the dashboard are described in
:doc:`/getting_started/running_jobs`; the sweeps, the examiner and the
figures of both pages are in :doc:`/user_guide/convergence_testing`.

Optional: relax the pristine host
---------------------------------

Relaxing silicon without a muon shows how well the functional reproduces the
experimental lattice.

1. Select ``Pristine Relaxation`` and click ``Add Job``.
2. Leave the row's ``Supercell`` at ``Unit cell`` and ``Fix Cell`` at ``No
   (relax)``, so the lattice relaxes as well as the atoms.
3. Click ``Run All Locally`` or ``Run All on Cluster``.
4. When the job completes, read the relaxed ``a (Ang)``, ``b (Ang)`` and
   ``c (Ang)``.

.. figure:: ../_generated/screenshots/pristine_relaxation_page.webp
   :alt: The Pristine Relaxation page with one completed variable-cell
         relaxation of silicon and its relaxed lattice lengths.

   A pristine relaxation of silicon, from a synthetic CASTEP output; the
   numbers are illustrative.

5. Click ``Generate Comparison Table...``, tick ``Include`` for the completed
   job, keep or edit the ``Comparison table label`` and click ``OK``.

.. figure:: ../_generated/screenshots/comparison_table.webp
   :alt: A comparison table of the silicon host against one relaxed cell,
         grouped into lattice, displacement, bonding and symmetry metrics.

   The silicon host compared with that synthetic relaxation; the numbers are
   illustrative.

The table is saved under ``Pristine Relaxation`` in the navigator. This
tutorial keeps the imported structure as the active host. See
:doc:`/user_guide/structural_relaxation_analysis` (*Pristine Relaxation*) and
:doc:`/user_guide/comparison_table` for reading the metrics.

Generate candidate sites and relax them
---------------------------------------

Candidate sites are starting positions for full CASTEP relaxations, one muon
per relaxation.

1. Select ``Generation`` under ``Muon Sites``. In ``Generation Setup`` leave
   the ``Method`` at ``Random generation``.
2. Select ``Number of initial positions`` and enter how many relaxations you
   want to run, then click ``Generate Sites``. The rows appear in the table;
   select some to see them in ``Preview Muon Sites``.
3. Select the rows to relax and click ``Create Jobs from Selected``. In
   ``Choose Charge State`` keep ``Positive muon`` and accept.
4. Select ``Relaxations`` and open the ``Positive muon`` tab. Select every row
   and pick your saved supercell in one row's ``Supercell`` drop-down; when
   asked, apply it to all the selected rows.
5. Click ``Configure Selected Parameters...`` to check the CASTEP settings.
   The rows inherit the project defaults, with an equivalent-density k-point
   grid for the supercell.
6. Click ``Run All Locally`` or ``Run All on Cluster``. The ``Job Dashboard``
   shows each running relaxation's geometry-optimisation progress.
7. When the rows complete, compare the ``Initial`` and ``Final`` positions and
   the ``Total Energy (eV)``.

.. figure:: ../_generated/screenshots/muon_relaxations_page.webp
   :alt: The Structural Relaxations tab with five completed positive-muon
         relaxations in a 2x2x2 silicon supercell, their initial and final
         positions and energies.

   Five positive-muon relaxations in silicon; the relaxation outputs are
   illustrative.

Staged rows and their configuration are written to disk when you choose
``File > Save Project``, so save the project now. The other generation methods
are in :doc:`/user_guide/site_generation`; watching and recovering jobs is
covered under :ref:`geometry-optimisation-progress` and
:ref:`job-failure-reasons`.

As a cheaper first look, the ``UEP`` stage estimates candidate sites from one
CASTEP calculation of the host charge density, and its cluster representatives
can seed the ``UEP clusters`` generation method (see
:doc:`/user_guide/uep_workflow`).

Find the distinct sites and pin them
------------------------------------

Several starts usually relax to the same site. An analysis workspace groups
the relaxed positions into distinct sites.

1. In the ``Positive muon`` tab of ``Relaxations``, select the completed rows
   and click ``Create Analysis Workspace from Selection``. Enter a ``Workspace
   name`` and accept. The workspace opens under ``Analysis``.
2. Click ``Clustering...``. Choose ``Connected Components`` as the
   ``Method``, click ``Scan Tolerances`` and choose a ``Distance tolerance (Å)`` on a
   plateau of the cluster-count curve, then click ``Apply Clustering``.
3. Read the clustered sites, ordered by energy relative to the lowest, each
   with its fitted Wyckoff label.

.. figure:: ../_generated/screenshots/relaxation_analysis_workspace.webp
   :alt: An analysis workspace with five silicon muon relaxations clustered
         into two sites, beside a 3D preview of the host cell.

   Illustrative silicon relaxations clustered into bond-centre and tetrahedral
   sites.

4. Click ``Pin to Summary``, then select ``Muon Sites`` in the navigator. The
   ``Positive muon`` card ranks the distinct sites by relative energy.
5. Choose ``File > Save Project``: workspaces are kept until the project is
   saved.

.. figure:: ../_generated/screenshots/muon_sites_summary.webp
   :alt: The Muon Sites page with two distinct positive-muon sites pinned and
         an empty Muonium card.

   The two sites pinned from that workspace; the energies are illustrative.

The pinned sites are what Magnetic Fields, Quantum Corrections and Energy
Barriers list by default. ``Open in Analysis →`` returns to the workspace.
Clustering methods, Wyckoff assignment and displacement analysis are in
:doc:`/user_guide/structural_relaxation_analysis`.

Estimate the local field
------------------------

Silicon has no ordered electronic moments, so the field a muon feels there
comes from the nuclear dipole moments of the host. MuFinder estimates its
width directly from the pinned site.

1. Select ``Magnetic Fields``. In the ``Muon Site`` card click ``Select Muon
   Site...``.
2. With ``Structural relaxation clustering`` as the ``Source``, the table lists
   the pinned sites. Choose the lowest-energy one and click ``OK``.
3. In the ``Calculation`` card click ``Nuclear Relaxation...``. In the ``Mode``
   group choose ``Zero field`` and ``Polycrystal``. Optionally tick ``Apply
   muon-induced distortion correction`` to account for the displaced silicon
   neighbours.
4. Click ``Calculate``. The ``Results`` group gives the width ``σ (μs⁻¹)`` and
   the equivalent field width ``Δ field (T)``.

For a magnetic host, define the magnetic order instead: click ``Edit Magnetic
Structure...`` in the ``Magnetic Structure`` card, build or import the
structure and click ``Use Structure``, then click ``Calculate Dipolar Field``.
The results table lists the field at the site and, for a commensurate
structure, its dipolar, Lorentz, contact and demagnetizing terms. These
calculations use the optional ``muesr`` and ``muLFC`` backends. See
:doc:`/user_guide/magnetic_fields`, and
:doc:`/user_guide/magnetic_fields_theory` for the physics.

Where to go next
----------------

- :doc:`/user_guide/quantum_corrections` estimates the muon's zero-point
  energy at each relaxed site from Gamma-point phonon calculations.
- :doc:`/user_guide/energy_barriers` builds climbing-image NEB calculations
  between two relaxed sites, for example from the bond-centre to the
  tetrahedral site.
- :doc:`/user_guide/journal` records what you did in this project, day by day.
- :doc:`/user_guide/overview` lists everything else each stage can do.
