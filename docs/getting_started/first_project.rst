Your first project
==================

This tutorial takes a new project from an empty folder to pinned muon
stopping sites in CoF₂, with an estimate of the local field the muon feels at
the lowest one. Each stage is a short route through one page of the
application. The linked user-guide page explains every control and option on
it, and :doc:`/user_guide/examples/cof2` follows the same project through the
same stages and compares its numbers with the literature.

The stages follow the order of the project navigator:

1. `Create a project and set up CASTEP`_
2. `Import the host structure`_
3. `Save a supercell`_
4. `Converge the cutoff and k-point grid`_
5. `Optional: relax the pristine host`_
6. `Optional: a first look with the UEP`_
7. `Generate candidate sites and relax them`_
8. `Find the distinct sites and pin them`_
9. `Estimate the local field`_

Before you start
----------------

- **MuFinder installed.** See :doc:`/getting_started/installation`.
- **CASTEP.** A CASTEP executable on this computer, or an HPC cluster that can
  run it. MuFinder does not include CASTEP.
- **A structure file for CoF₂.** A CIF of rutile CoF₂ from a structure
  database will do. The screenshots use its six-atom unit cell, two Co and
  four F.

.. note::

   The screenshots come from a CoF₂ project in which no calculation ran
   CASTEP. Its outputs are synthetic, written so that MuFinder's analysis of
   them reproduces published values, and the convergence sweep data are
   illustrative. Your own calculations will give values of their own.

Create a project and set up CASTEP
----------------------------------

1. Start MuFinder (``python -m mufinder.main``, or ``mufinder`` for a source
   install).
2. Choose ``File > New Project`` (or click ``Create New Project`` on the welcome
   page). Enter a ``Project name:``, then choose where it goes in ``Select
   parent folder``. MuFinder creates a ``<name>.mufinder`` folder there and
   opens the project home.
3. Open ``Setup > CASTEP…`` and install a keyword catalog: pick the target in
   ``Generate from`` and click ``Generate Keyword Catalog``, or click ``Use
   Existing File…`` if you already have a ``castep_keywords.json``.

   .. figure:: ../_generated/screenshots/castep_setup_dialog.webp
      :alt: The Setup > CASTEP dialog with no keyword catalog loaded, offering to
            generate one or use an existing file.

      ``Setup > CASTEP`` before a catalog is installed.

4. To run on this computer, fill in ``CASTEP executable`` and ``MPI launcher
   (mpirun)`` (``Auto-detect`` searches for ``mpirun``), then click ``OK``.
5. To run on a cluster instead, add a profile under ``Setup > HPC
   Clusters…``.

.. figure:: ../_generated/screenshots/cluster_setup_dialog.webp
   :alt: The Remote Cluster Connections dialog with the built-in SCARF profile
         selected on the Connection tab.

   ``Setup > HPC Clusters`` with the built-in SCARF profile selected.

When the catalog is installed the status bar reads ``CASTEP keywords:``
followed by the CASTEP version. The details, including what happens without a
catalog, are in :doc:`/getting_started/castep_setup`; cluster profiles are
described under *Running on a cluster* in :doc:`/getting_started/running_jobs`.

The project home charts every stage and says what each one needs. Click the
home button above the project navigator at any point to see where the project
stands (see :doc:`/user_guide/project_home`).

.. figure:: ../_generated/screenshots/project_home.webp
   :alt: The project home of a CoF2 project: the navigator beside the workflow
         chart, with five stages complete, the field notes and the cell survey.

   The project home of the CoF₂ project once the supercell, the convergence
   defaults and the candidate sites are done; the dashed stages are still to
   do.

Import the host structure
-------------------------

1. Select ``Host Structure`` in the project navigator.
2. Click ``Load Structure (.cif/.mcif/.cell)`` and choose your CoF₂ file.
3. Check the summary, the atom table and the 3D viewer: two ``Co`` and four
   ``F`` atoms in the unit cell.
4. CoF₂ is an antiferromagnet, with the corner and body-centre Co moments
   antiparallel. Enter ``3`` and ``-3`` in the ``Spin (hbar/2)`` column for
   the Co at (0, 0, 0) and the Co at (½, ½, ½), and leave the F at zero.

.. figure:: ../_generated/screenshots/host_structure_page.webp
   :alt: The Host Structure page with CoF2 loaded: two Co and four F in the
         atom table, Spin (hbar/2) +3 and -3 on the two Co, and the cell viewed
         down a with antiparallel spin arrows.

   The CoF₂ host with its two antiparallel Co sublattices.

The imported structure becomes the active host, which every later stage works
on. The spins are the starting order of every calculation that runs
spin-polarised: tick ``Spin Polarized`` in a row's parameters for CoF₂. See
:doc:`/user_guide/host_structure`.

Save a supercell
----------------

A muon relaxation needs a cell large enough that the muon does not interact
with its own periodic images.

1. Select ``Supercell Editor`` in the navigator and click ``Open Supercell
   Editor``.
2. Click ``New``, then ``Suggest…``. The suggestions list diagonal supercells
   with their atom count and minimum image distance, and mark one
   ``(Recommended)``.

   .. figure:: ../_generated/screenshots/supercell_suggestions.webp
      :alt: The supercell suggestions for CoF2: three diagonal cells, the 3x3x3
            cell highlighted as recommended.

      The suggestions for CoF₂: no cell under 200 atoms reaches 10 Å.

3. Highlight a cell and click ``Use This Supercell``. The preview and the
   status line show the result.
4. Name it and click ``Save``. It appears in ``Saved Supercells``.

.. figure:: ../_generated/screenshots/supercell_editor.webp
   :alt: The Supercell Editor with the saved CoF2 2x2x2 supercell of 48 atoms
         selected and its preview viewed down a.

   The saved ``CoF2 2x2x2`` supercell.

The relaxations in the later screenshots ran in a 2x2x2 supercell (48 atoms),
the cell of the MuFinder paper, which keeps the example cheap; follow the
``(Recommended)`` cell for production work. See :doc:`/user_guide/supercells`
(*Suggesting a supercell*).

Converge the cutoff and k-point grid
------------------------------------

Every later calculation inherits the project's XC functional, cutoff energy
and k-point grid, so settle them first.

1. Select ``Cutoff Energy`` under ``Convergence Testing``. On the ``Project
   defaults`` strip click ``Edit…``, choose an ``XC functional`` (the
   screenshots use ``PBE``) and click ``OK``.
2. Leave ``Single unit cell`` selected under ``Simulation Cell``.
3. Click ``Generate From Range…`` and accept ``Start (eV)``, ``Stop (eV)``
   and ``Step (eV)``, or change them.

   .. figure:: ../_generated/screenshots/cutoff_convergence_page.webp
      :alt: The Cutoff Convergence page with six cutoff rows from 300 to 800 eV
            ready to run.

      Six cutoffs from 300 to 800 eV, ready to run.

4. Click ``Run All Locally`` or ``Run All on Cluster`` and fill in the
   submission dialog. Each row's ``Status`` follows its job, and the ``Job
   Dashboard`` at the bottom of the window lists them too.

   .. figure:: ../_generated/screenshots/local_submission_dialog.webp
      :alt: The local submission dialog for six cutoff-convergence jobs, set to
            run one after another.

      The local submission dialog for the six cutoff rows.

5. When the rows have completed, click ``Examine Cutoff Convergence``, then
   ``Analyse Convergence``, then the ``Set <value> eV as Project Default
   Cutoff`` button.
6. Select ``K-Point Grid``, click ``Generate From Spacing…`` and accept the
   ``Minimum spacing (1/Ang)``. The new rows use the cutoff you just adopted.

   .. figure:: ../_generated/screenshots/kpoint_convergence_page.webp
      :alt: The K-Point Convergence page with the grids generated for CoF2 down
            to 0.05 1/Ang.

      The grids generated for CoF₂ down to a spacing of 0.05 1/Ang.

7. Run them, then click ``Examine K-Point Convergence``, ``Analyse
   Convergence`` and ``Set <grid> as Project Default K-Point Grid``.
8. Select ``Convergence Testing`` to check both adopted values and their
   pinned plots.

.. figure:: ../_generated/screenshots/convergence_overview.webp
   :alt: The Convergence Testing overview with 600 eV and a 4 x 4 x 6 grid
         adopted as project defaults and their pinned convergence plots.

   The adopted defaults with their pinned plots, here 600 eV and a 4 × 4 × 6
   grid; the sweep data are illustrative.

If no value is recommended, the result line says which criterion did not
converge; extend the sweep and analyse again. Submitting jobs, the submission
dialogs and the dashboard are described in
:doc:`/getting_started/running_jobs`; the sweeps, the examiner and the
figures of both pages are in :doc:`/user_guide/convergence_testing`.

Optional: relax the pristine host
---------------------------------

Relaxing CoF₂ without a muon shows how well the functional reproduces the
experimental lattice.

1. Select ``Pristine Relaxation`` and click ``Add Job``.
2. Leave the row's ``Supercell`` at ``Unit cell`` and ``Fix Cell`` at ``No
   (relax)``, so the lattice relaxes as well as the atoms. Tick ``Spin
   Polarized`` in the row's parameters.
3. Click ``Run All Locally`` or ``Run All on Cluster``.
4. When the job completes, read the relaxed ``a (Ang)``, ``b (Ang)`` and
   ``c (Ang)``.

   .. figure:: ../_generated/screenshots/pristine_relaxation_page.webp
      :alt: The Pristine Relaxation page with one completed variable-cell
            relaxation of CoF2 and its relaxed lattice lengths.

      The pristine relaxation of CoF₂, from a synthetic CASTEP output.

5. Click ``Generate Comparison Table…``, tick ``Include`` for the completed
   job, keep or edit the ``Comparison table label`` and click ``OK``.

.. figure:: ../_generated/screenshots/comparison_job_dialog.webp
   :alt: The Generate Comparison Table dialog with one completed CoF2
         pristine relaxation ticked and its relaxed lattice lengths.

   ``Generate Comparison Table`` with the pristine relaxation ticked.

.. figure:: ../_generated/screenshots/comparison_table.webp
   :alt: A comparison table of the CoF2 host against one relaxed cell,
         grouped into lattice, displacement, bonding and symmetry metrics.

   The CoF₂ host compared with that relaxation.

The table is saved under ``Pristine Relaxation`` in the navigator. This
tutorial keeps the imported structure as the active host. See
:doc:`/user_guide/structural_relaxation_analysis` (*Pristine Relaxation*) and
:doc:`/user_guide/comparison_table` for reading the metrics.

Optional: a first look with the UEP
-----------------------------------

The ``UEP`` stage estimates where a positive muon sits from one CASTEP
calculation of the host charge density, much more cheaply than relaxing
candidates one by one.

1. Select ``UEP`` and click ``Add Job``. Leave ``Supercell`` at ``Unit cell``
   and tick ``Spin Polarized`` in the row's parameters.
2. Click ``Run All Locally`` or ``Run All on Cluster``.

   .. figure:: ../_generated/screenshots/uep_page.webp
      :alt: The UEP page with one completed source job on the CoF2 host and its
            Open Wizard button.

      The completed UEP source job on CoF₂. Its charge density is a model
      density, not a CASTEP result.

3. When the job completes, click ``Open Wizard`` in its ``Analysis`` column.
4. On the ``Batch Search`` tab, accept the sampling values and click ``Run
   Batch Search``.

   .. figure:: ../_generated/screenshots/uep_batch_search.webp
      :alt: The UEP analysis window's Batch Search tab after twelve random starts,
            with every optimized site at the same classical energy.

      The batch search after twelve random starts.

5. Click ``Clustering`` in the ``Sampling`` card to group the relaxed minima.

.. figure:: ../_generated/screenshots/uep_clustering_dialog.webp
   :alt: The UEP clustering window with a dendrogram of twelve relaxed sites
         and the cluster energy spread.

   The clustering of the batch-search minima.

The cluster representatives can seed the ``UEP clusters`` generation method in
the next stage. The UEP leaves the host atoms where they are, so in CoF₂ it
misses the displacements around a positive muon; the relaxations below do not.
See :doc:`/user_guide/uep_workflow`.

Generate candidate sites and relax them
---------------------------------------

Candidate sites are starting positions for full CASTEP relaxations, one muon
per relaxation.

1. Select ``Generation`` under ``Muon Sites``. In ``Generation Setup`` choose
   ``Attach muon to element``, pick ``F`` as the ``Element``, set ``Muons per
   atom`` to ``1`` and click ``Generate Sites``. Each of the four F gets one
   start 1.0 Å away.
2. Choose ``Random generation``, select ``Number of initial positions``, enter
   ``6`` and click ``Generate Sites`` again. The rows are added to the table.
3. Select all the rows to see them in ``Preview Muon Sites``.

   .. figure:: ../_generated/screenshots/site_generation_page.webp
      :alt: The Generation page listing four Attached-F sites and six random
            sites, all selected and marked in the CoF2 cell.

      Four F-attached and six random starts, all selected.

4. Click ``Create Jobs from Selected``. In ``Choose Charge State`` keep
   ``Positive muon`` and accept.

   .. figure:: ../_generated/screenshots/charge_state_dialog.webp
      :alt: The Choose Charge State dialog for ten selected initial positions,
            with Both charge states chosen.

      ``Choose Charge State`` for the ten starts. The figure has ``Both charge
      states`` chosen, which relaxes every start as muonium too; this tutorial
      follows the positive muon.

5. Select ``Relaxations`` and open the ``Positive muon`` tab. Select every row
   and pick your saved supercell in one row's ``Supercell`` drop-down; when
   asked, apply it to all the selected rows.
6. Click ``Configure Selected Parameters…`` to check the CASTEP settings and
   tick ``Spin Polarized``. The rows inherit the project defaults, with an
   equivalent-density k-point grid for the supercell.
7. Click ``Run All Locally`` or ``Run All on Cluster``. The ``Job Dashboard``
   shows each running relaxation's geometry-optimisation progress.
8. When the rows complete, compare the ``Initial`` and ``Final`` positions and
   the ``Total Energy (eV)``.

.. figure:: ../_generated/screenshots/muon_relaxations_page.webp
   :alt: The Relaxations page, Positive muon tab, with ten completed CoF2
         relaxations, their initial and final positions and total energies.

   The ten positive-muon relaxations, from synthetic outputs.

Staged rows and their configuration are written to disk when you choose
``File > Save Project``, so save the project now. The other generation methods
are in :doc:`/user_guide/site_generation`; watching and recovering jobs is
covered under :ref:`geometry-optimisation-progress` and
:ref:`job-failure-reasons`.

Find the distinct sites and pin them
------------------------------------

Several starts usually relax to the same site. An analysis workspace groups
the relaxed positions into distinct sites.

1. In the ``Positive muon`` tab of ``Relaxations``, select the completed rows
   and click ``Create Analysis Workspace from Selection``. Enter a ``Workspace
   name`` and accept. The workspace opens under ``Analysis``.

   .. figure:: ../_generated/screenshots/relaxation_analysis_workspace.webp
      :alt: The Positive muon sites workspace with ten relaxations clustered into
            4c, 4d and 8i sites at 0, 0.34 and 0.82 eV.

      The workspace after clustering: the 4c site and two higher sites.

2. Click ``Clustering…``. Choose ``Connected Components`` as the
   ``Method``, click ``Scan Tolerances`` and choose a ``Distance tolerance (Å)``
   on a plateau of the cluster-count curve, then click ``Apply Clustering``.

   .. figure:: ../_generated/screenshots/relaxation_clustering_dialog.webp
      :alt: The Clustering dialog with Connected Components at 0.25 A: a graph of
            seven, one and two linked sites, the cluster-count plot and the
            clustered-sites table with 4c, 4d and 8i Wyckoff assignments.

      Connected components at 0.25 Å.

3. Read the clustered sites, ordered by energy relative to the lowest, each
   with its fitted Wyckoff label.
4. Click ``Analyse Displacements``, choose the lowest-energy relaxed site as
   the ``Muon site`` and the imported host as the ``Comparison host`` to see
   how far the neighbours moved.

   .. figure:: ../_generated/screenshots/displacement_analysis_dialog.webp
      :alt: The Analyse Displacements dialog for the 4c site: two F at 1.70 A
            displaced by -0.50 and -0.57 A, two Co at 2.35 A by +0.14 and +0.17 A.

      The two nearest F move in by about half an ångström around the 4c site.

5. Click ``Pin to Summary``, then select ``Muon Sites`` in the navigator. The
   ``Positive muon`` card ranks the distinct sites by relative energy.
6. Choose ``File > Save Project``: workspaces are kept until the project is
   saved.

.. figure:: ../_generated/screenshots/muon_sites_summary.webp
   :alt: The Muon Sites summary with three positive-muon sites (4c, 4d, 8i)
         and two muonium sites (8i, 4d) pinned.

   The pinned sites. The example project also relaxed the starts as muonium,
   which fills the second card.

The pinned sites are what Magnetic Fields, Quantum Corrections and Energy
Barriers list by default. ``Open in Analysis →`` returns to the workspace.
Clustering methods, Wyckoff assignment and displacement analysis are in
:doc:`/user_guide/structural_relaxation_analysis`.

Estimate the local field
------------------------

Below its Néel temperature of 37.85 K, CoF₂'s ordered Co moments give the
muon a static dipolar field. Define the order, then calculate the field at the
pinned site.

1. Select ``Magnetic Fields`` and click ``Edit Magnetic Structure…`` in the
   ``Magnetic Structure`` card.
2. In the ``Magnetic Structure Builder``, add one collinear component with
   q = (0, 0, 0) on the corner Co with moment ``(0, 0, 2.64)`` μB, and one on
   the body-centre Co with ``(0, 0, -2.64)``, then click ``Use Structure``.

   .. figure:: ../_generated/screenshots/cof2_magnetic_structure_builder.webp
      :alt: The Magnetic Structure Builder with two collinear components, Co
            corner [0, 0, 2.64] and Co body centre [0, 0, -2.64], and the cell
            preview with antiparallel moments.

      The antiferromagnet as two collinear components.

3. In the ``Muon Site`` card click ``Select Muon Site…``. With ``Structural
   relaxation clustering`` as the ``Source``, the table lists the pinned
   sites. Choose the lowest-energy one, tick ``Correct the local field for
   muon-induced displacements`` and click ``OK``.

   .. figure:: ../_generated/screenshots/magnetic_muon_site_dialog.webp
      :alt: The Select Muon Site dialog listing the five pinned sites, the 4c
            positive-muon cluster selected and the displacement correction ticked.

      The pinned 4c site, with the distortion correction ticked.

4. Leave the ``Demagnetizing field`` at ``None``: the antiferromagnet has no
   net moment. Click ``Calculate Dipolar Field``.

.. figure:: ../_generated/screenshots/cof2_magnetic_fields_page.webp
   :alt: The Magnetic Fields page for CoF2: a distortion-corrected field of
         0.1996 T along c, broken down into a 0.2655 T dipolar term and a
         0.066 T distortion correction.

   The field at the 4c site: 0.27 T from the rigid lattice, 0.20 T with the
   distortion correction.

The results table lists the field at the site, and selecting its row shows
the dipolar, Lorentz, contact, demagnetizing and distortion-correction terms.
:doc:`/user_guide/examples/cof2` compares both numbers with the MuFinder paper
and with experiment. These calculations use the optional ``muesr`` and
``muLFC`` backends.

For CoF₂ above T_N the field comes from the nuclear dipoles instead: click
``Nuclear Relaxation…``, choose ``Zero field`` and ``Polycrystal`` and click
``Calculate`` for the width ``σ (μs⁻¹)``.

.. figure:: ../_generated/screenshots/nuclear_relaxation_dialog.webp
   :alt: The Nuclear Relaxation dialog with a zero-field polycrystal result of
         sigma 1.49 per microsecond at the distortion-corrected 4c site.

   The zero-field polycrystal width at the 4c site.

See :doc:`/user_guide/magnetic_fields`, and
:doc:`/user_guide/magnetic_fields_theory` for the physics.

Where to go next
----------------

- :doc:`/user_guide/examples/cof2` continues this project to the muon's
  zero-point motion and a hop between 4c sites, and compares each number with
  the literature.
- :doc:`/user_guide/quantum_corrections` estimates the muon's zero-point
  energy at each relaxed site from Gamma-point phonon calculations;
  :doc:`/user_guide/examples/alpha_n2` shows its analysis window.
- :doc:`/user_guide/energy_barriers` builds climbing-image NEB calculations
  between two relaxed sites; :doc:`/user_guide/examples/gdru2si2` runs one on
  real CASTEP outputs.
- :doc:`/user_guide/examples/fe` and :doc:`/user_guide/examples/mnsi` work the
  Magnetic Fields page on a ferromagnet and an incommensurate helix.
- :doc:`/user_guide/journal` records what you did in this project, day by day.
- :doc:`/user_guide/overview` lists everything else each stage can do.
