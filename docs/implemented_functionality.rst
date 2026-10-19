Implemented Functionality
=========================

This page is an index of what MuFinder can do. Each section describes one area
of the application in a few sentences and links the page that explains it. The
workflow sections follow the order of the stages in the project navigator; the
sections after them cover the job system, CASTEP input and the development
harness that every workflow shares.

Application shell
-----------------

- The ``File`` menu creates, opens, reopens (``Open Recent``) and saves
  projects, and holds ``Import Legacy Relaxations...`` and ``Remove Imported
  Relaxations...`` (see :doc:`legacy_import`).
- The ``Setup`` menu configures the local CASTEP installation and keyword
  catalog (``CASTEP...``, see :doc:`castep_keyword_catalog`) and the cluster
  profiles (``HPC Clusters...``, see :doc:`job_lifecycle`).
- ``View -> Appearance`` switches between ``Light``, ``Dark`` and ``System``
  themes. MuFinder bundles its own fonts, icons and control styling, so it looks
  the same on macOS, Windows and Linux.
- The project navigator lists the workflow stages, and a project-wide Job
  Dashboard dock tracks every job. Analysis-workspace and saved
  magnetic-field-workspace nodes have a right-click delete action; workflow
  stage nodes have no context menu.
- The project home charts the workflow stages with their dependencies and
  progress, opens at the stage last worked on, and surveys the host cell with
  that stage's muon positions (see :doc:`project_home`).
- Durable project and job state is autosaved. Analysis layouts and viewer
  choices are kept until ``Save Project``, which MuFinder offers whenever a
  project window is closed.

:doc:`architecture` describes how projects are stored and saved (Domain model
and persistence) and how the cross-platform appearance is achieved
(Cross-platform appearance).

Host structure
--------------

A project starts from a host crystal structure imported from a CIF, magnetic
CIF (``.mcif``) or CASTEP ``.cell`` file. The source file is copied into the
project, and MCIF magnetic metadata (propagation vectors and atom-site moments)
becomes the project magnetic structure and the host's per-atom spins. The
``Host Structure`` panel switches between imported hosts, edits per-atom spins
in units of ``hbar/2`` and the propagation vector in reciprocal-lattice units,
and shows the structure in an orthographic 3D viewer with optional spin arrows,
bonds and unit cell.

See :doc:`host_structure`.

Supercells
----------

The ``Supercell Editor`` saves named, reusable supercells of the active host,
from a diagonal ``Nx``/``Ny``/``Nz`` scaling or a full 3×3 integer matrix, with
a preview before saving. ``Suggest…`` proposes diagonal supercells by their
minimum periodic-image distance and recommends the cheapest one that reaches
10 Å under 200 atoms. A supercell already frozen into a configured calculation
is never rewritten: editing its matrix saves a new copy.

Each workflow chooses its own simulation cell, the host unit cell or a saved
supercell, and a configured calculation keeps the cell it was configured with.

See :doc:`supercells`.

Convergence testing
-------------------

Convergence testing sweeps ``CUT_OFF_ENERGY`` and then ``KPOINT_MP_GRID`` in
single-point CASTEP calculations on the host unit cell or a saved supercell.
Cutoffs are listed by hand or generated from a range; k-point grids are listed
by hand or generated as every distinct grid CASTEP produces down to a minimum
spacing. An examiner judges convergence of the energy (Delta-Threshold,
Derivative or Asymptotic Extrapolation for cutoffs), the maximum force and,
optionally, the maximum stress, plots the results, and recommends a value. The
``Convergence Testing`` overview page shows both adopted defaults with the
plots pinned when they were adopted.

See :doc:`convergence_testing` and, for the grid and spacing round trip,
``docs/castep/kpoint_spacing_convention.md``.

Project defaults
----------------

A project carries three calculation defaults that later workflows inherit: the
XC functional, the plane-wave cutoff energy and the Monkhorst-Pack k-point
grid. Adopting an examiner recommendation sets the cutoff or grid and the
functional it was converged with; all three can also be set or cleared by hand
in the ``Project Defaults`` dialog. Every workflow tab that uses them shows the
same ``Project defaults`` strip, which flags a cutoff or grid converged with a
different functional from the current one. Supercell rows rescale the default
grid to the same reciprocal-space sampling density. Quantum Corrections and
Energy Barriers rows take their functional, cutoff, grid and spin treatment
from the relaxation they start from instead. Every change to a default is
recorded in the Journal.

See :doc:`convergence_testing` (Adopting a result as a project default),
:doc:`quantum_corrections` and :doc:`energy_barriers`, and :doc:`architecture`
(Project defaults surface).

Pristine relaxation and comparison tables
-----------------------------------------

The optional ``Pristine Relaxation`` stage relaxes the host without a muon. Each
labelled row chooses its own unit cell or saved supercell, whether the cell is
fixed (``FIX_ALL_CELL``), and its CASTEP parameters. Completed rows show the
relaxed ``a``, ``b`` and ``c``; a run whose geometry optimization did not
converge is marked failed, and each successful run adds a relaxed structure to
the project.

``Generate Comparison Table...`` compares one or more relaxed structures with
the experimental host, with supercell-aware atom matching, lattice,
displacement, bonding, topology and symmetry metrics, a composite similarity
score, the parameter differences between the selected jobs, and CSV export.
Each table is saved as a renamable child of ``Pristine Relaxation``.

See :doc:`structural_relaxation_analysis` (Pristine Relaxation) and
:doc:`comparison_table`.

UEP screening
-------------

The ``UEP`` stage runs CASTEP single-point source calculations for the host
charge density and then analyses the unperturbed electrostatic potential: full
cell potential grids with isosurface and slice views, random-start batch
searches for potential minima, an optional harmonic zero-point estimate, and a
clustering window with hierarchical and k-means views. Cluster
representatives are offered to muon-site generation.

See :doc:`uep_workflow`.

Muon-site generation
--------------------

``Muon Sites -> Generation`` commits initial muon positions to the project by
four methods: symmetry-aware random filling (by muon-muon separation or by a
target number of positions), placement around atoms of a chosen element,
manual fractional or Cartesian coordinates, and UEP cluster representatives.
Selected rows can be moved to the symmetry-equivalent position closest to an
anchor (``Cluster Positions...``) or removed before relaxation jobs are created.
Creating jobs asks for the charge state: positive muon, muonium, or both.

The symmetry-aware random method follows the initial-position algorithm of
Huddart *et al.*, ``MuFinder: A program to determine and analyse muon stopping
sites``, *Computer Physics Communications* **280**, 108488 (2022), DOI:
``10.1016/j.cpc.2022.108488``.

See :doc:`site_generation`.

Muon relaxations
----------------

``Muon Sites -> Relaxations`` holds one fixed-cell CASTEP geometry optimization
per staged position, in separate ``Positive muon`` and ``Muonium`` tabs. MuFinder
sets the CASTEP charge from the tab, writes muonium rows spin-polarised by
default, and gives each row its own supercell and parameters. Relaxations run
outside MuFinder can be imported as read-only rows.

A completed, failed, stopped or cancelled pristine or muon relaxation can be
continued from its CASTEP checkpoint with ``Continue relaxation…``, which edits
the row's parameters in place and resumes the same job in the same directory.

See :doc:`site_generation` (Creating Relaxation Jobs),
:ref:`continue-relaxation` and :doc:`legacy_import`.

Relaxation analysis and the Muon Sites summary
----------------------------------------------

``Muon Sites -> Analysis`` collects completed relaxations of one charge state
into named workspaces. A workspace orders the relaxed sites by energy, clusters
them (connected components, hierarchical or k-means) into distinct sites with
fitted Wyckoff labels, analyses the host displacements around a site, and
exports the relaxed positions as CIF. ``Pin to Summary`` copies a workspace's
distinct sites to the ``Muon Sites`` summary page, which shows one card per
charge state with a ranked site table, a 3D preview, links back into the
analysis, and CIF export of the pinned sites.

See :doc:`structural_relaxation_analysis`.

Legacy relaxation import
------------------------

``File > Import Legacy Relaxations...`` imports muon structural relaxations
run by the legacy MuFinder application as read-only relaxation rows, so they
can be used in relaxation analysis, Quantum Corrections and Energy Barriers.
``File > Remove Imported Relaxations...`` removes an import batch that nothing
downstream still uses.

See :doc:`legacy_import`.

Magnetic fields
---------------

The ``Magnetic Fields`` workflow builds magnetic structures in the
propagation-vector formalism, including from the host spins and as skyrmion or
custom Fourier textures, from components that can be edited, duplicated,
reordered and disabled without deleting them, with a panel that shows and
edits the stored propagation-vector basis and which components use each row,
and calculates at a chosen muon site:

- the dipolar field of commensurate structures, optionally corrected for the
  host-ion displacements in a relaxed muon supercell at the site and its
  symmetry-equivalent sites, and the field distribution of incommensurate
  ones, with Auto, Grid or Monte Carlo multi-q phase sampling;
- demagnetizing-field corrections for spherical, plate, ellipsoidal and
  rectangular-prism samples;
- dipolar tensors for selected magnetic ions;
- nuclear dipolar second moments in zero and transverse field, for single
  crystals and polycrystals, from bundled natural-abundance isotope data, with
  an optional muon-induced distortion correction.

Muon sites come from UEP or relaxation clustering or a custom fractional
position. Settings are saved as named workspaces.

See :doc:`magnetic_fields`.

Quantum corrections
-------------------

Quantum Corrections lists every successfully relaxed muon site and runs a
two-stage CASTEP calculation for each: a single-point with explicit muon mass
and gyromagnetic ratio, then a Gamma-point phonon continuation (finite
displacement by default, with optional task farming). Completed results give
the muon zero-point energy, covariance and correlation tensors, ``w1`` /
``w2`` witnesses and single-particle effective potentials, shown in an
interactive analysis window.

See :doc:`quantum_corrections`.

Energy barriers
---------------

Energy Barriers takes an ordered pair of relaxed muon sites, generates the
symmetry-distinct paths between them (including same-site hops, with an
optional maximum length), and builds a climbing-image NEB job for each chosen
path from the two relaxed endpoints. Paths can be previewed in the pristine
supercell, configured per row (including CASTEP task farming), and inspected
once complete: a results table lists forward and reverse barriers and ``ΔE``,
and an inspector scrubs the reaction-coordinate profile with a linked
structure viewer.

See :doc:`energy_barriers`.

Journal
-------

The ``Journal`` is a curated, day-grouped narrative of the project's
provenance: structure import, workflow configuration, project defaults, site
generation, job submissions, continuations and outcomes, comparison tables and
legacy imports. It exports as plain text or Markdown.

See :doc:`journal`.

Jobs and execution
------------------

Every workflow table runs its ready rows on this computer or on an HPC cluster
and shares one set of table behaviours:

- the same right-click row menu, multi-row targeting with a three-way prompt
  for in-row controls, and a confirmation for every remove, cancel, reset,
  dismiss or delete;
- local runs in the background with sequential or parallel execution and
  core-count guidance;
- cluster profiles for SLURM and ``addqueue`` schedulers, including built-in
  SCARF and Redwood presets, SSH jump hosts and host-key trust on first use,
  several named queues (partitions) per profile with per-queue walltime,
  cores, memory, executable and template overrides, a queue choice on every
  submission, and queue polling and automatic output retrieval;
- a Job Dashboard of the jobs that still need attention, with per-row
  ``Resubmit``, ``Retry retrieval``, ``Cancel``, ``Inspect``, ``Reset`` and
  ``Dismiss``, a failure cause on each failed or stopped row, and safeguards
  that keep no job stuck in an active state;
- for relaxations, a choice on resubmission between continuing from the
  checkpoint and starting again.

See :doc:`job_lifecycle`, in particular Job tables and selections, Running
jobs, :ref:`job-failure-reasons` and :ref:`resubmit-continue-or-start-again`.

CASTEP input and parameters
---------------------------

Every workflow configures CASTEP through the same metadata-driven parameter
dialog: job-type-aware basic and advanced parameters, typed controls, help
from the keyword metadata, a search filter, and one dialog for several rows in
which values that differ are marked ``mixed`` and kept unless edited. The
shared parameters come first in every workflow. The keyword metadata comes
from a CASTEP keyword catalog generated from a local or cluster CASTEP, or
imported. Workflows expand into resolved job specifications that are rendered
into deterministic ``.cell`` and ``.param`` files.

See :doc:`castep_keyword_catalog`, :doc:`architecture` (CASTEP parameter
dialogs, Calculation job generation, Workflow parameter surface policy) and
``docs/castep/keyword_catalog.yaml``.

Development harness
-------------------

The repository is organised for development by coding agents. ``AGENTS.md`` is
the canonical agent guide, and ``tests/test_agent_harness.py`` enforces its
structural rules: layer imports, docs indexing, the ``MainWindow`` handler,
plot-canvas and duplication ratchets, and the pull-request template pointers.

See :doc:`agent_harness`, :doc:`architecture` and :doc:`quality_and_automation`
(Agent harness checks).
