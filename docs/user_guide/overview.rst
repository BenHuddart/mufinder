Overview
========

This page is an index of what MuFinder can do. It describes the application
shell, then each stage of the project navigator in the order the navigator
lists them, then the behaviour every workflow shares. Each section gives a few
sentences and links the page that explains the stage as a task, and the worked
example that shows it on real or literature-tuned data.

Worked examples
---------------

Five worked examples show the stages on a project. Their screenshots are the
figures of the pages that follow. :doc:`/getting_started/first_project` is a
short tutorial on the first of them.

- :doc:`/user_guide/examples/cof2`: rutile CoF₂, from the host structure
  through convergence, the UEP, positive-muon and muonium relaxations,
  clustering and zero-point motion to the local field. Its outputs are
  synthetic, tuned to published values.
- :doc:`/user_guide/examples/fe`: bcc iron, a ferromagnet for the Magnetic
  Fields page, with its octahedral and tetrahedral sites, demagnetizing shapes
  and a measured magnetisation.
- :doc:`/user_guide/examples/mnsi`: the MnSi helimagnet, an incommensurate
  structure whose field distribution and commensurate approximant are
  calculated.
- :doc:`/user_guide/examples/alpha_n2`: solid nitrogen, the Quantum
  Corrections numbers and analysis window for a muon bound to a molecular
  crystal.
- :doc:`/user_guide/examples/gdru2si2`: GdRu₂Si₂, energy barriers between two
  muon sites from the real CASTEP outputs of a published study.

Application shell
-----------------

- The ``File`` menu creates (``New Project``), opens (``Open Project``),
  reopens (``Open Recent``) and saves (``Save Project``) projects, and holds
  ``Import Legacy Relaxations...`` and ``Remove Imported Relaxations...`` (see
  :doc:`/user_guide/legacy_import`).
- The ``Setup`` menu configures the local CASTEP installation and keyword
  catalog (``CASTEP...``, see :doc:`/getting_started/castep_setup`) and the
  cluster profiles (``HPC Clusters...``, see
  :doc:`/getting_started/running_jobs`).
- ``View -> Appearance`` switches between ``Light``, ``Dark`` and ``System``
  themes. MuFinder bundles its own fonts, icons and control styling, so it looks
  the same on macOS, Windows and Linux.
- The ``Help`` menu opens the documentation site (``User Guide``) and shows the
  installed version with links to the documentation and source code
  (``About MuFinder``).
- The project navigator lists the workflow stages: ``Host Structure``,
  ``Supercell Editor``, ``Convergence Testing`` (``Cutoff Energy`` and
  ``K-Point Grid``), ``Pristine Relaxation``, ``UEP``, ``Muon Sites``
  (``Generation``, ``Relaxations`` and ``Analysis``), ``Magnetic Fields``,
  ``Quantum Corrections``, ``Energy Barriers`` and ``Journal``. The home button
  above it returns to the project home. Analysis-workspace and saved
  magnetic-field-workspace nodes have a right-click delete action; workflow
  stage nodes have no context menu.
- The ``Job Dashboard`` dock tracks every job in the project (see `Jobs and
  execution`_).
- The project home charts the workflow stages with their dependencies and
  progress, opens at the stage last worked on, and surveys the host cell with
  that stage's muon positions (see :doc:`/user_guide/project_home`).
- Durable project and job state is autosaved. Analysis layouts and viewer
  choices are kept until ``Save Project``, which MuFinder offers whenever a
  project window is closed.

Workflow stages
---------------

Host Structure
~~~~~~~~~~~~~~

A project starts from a host crystal structure imported with ``Load Structure
(.cif/.mcif/.cell)`` from a CIF, magnetic CIF (``.mcif``) or CASTEP ``.cell``
file. The source file is copied into the project, and MCIF magnetic metadata
(propagation vectors and atom-site moments) becomes the project magnetic
structure and the host's per-atom spins. The ``Host Structure`` panel switches
between imported hosts, edits per-atom spins in units of ``hbar/2`` and the
propagation vector in reciprocal-lattice units, and shows the structure in an
orthographic 3D viewer with optional spin arrows, bonds and unit cell.

See :doc:`/user_guide/host_structure`. :doc:`/user_guide/examples/cof2` starts
from a CoF₂ host with antiparallel Co spins.

Supercell Editor
~~~~~~~~~~~~~~~~

The ``Supercell Editor`` saves named, reusable supercells of the active host,
from a diagonal ``Nx``/``Ny``/``Nz`` scaling or a full 3×3 integer matrix, with
a preview before saving. ``Suggest…`` proposes diagonal supercells by their
minimum periodic-image distance and recommends the cheapest one that reaches
10 Å under 200 atoms. A supercell already frozen into a configured calculation
is never rewritten: editing its matrix saves a new copy.

Each workflow chooses its own simulation cell, the host unit cell or a saved
supercell, and a configured calculation keeps the cell it was configured with.

See :doc:`/user_guide/supercells`. :doc:`/user_guide/examples/cof2` saves a
2x2x2 CoF₂ supercell, and its ``Suggest…`` shows why production cells are
larger.

Convergence Testing
~~~~~~~~~~~~~~~~~~~

``Convergence Testing`` sweeps ``CUT_OFF_ENERGY`` (``Cutoff Energy``) and then
``KPOINT_MP_GRID`` (``K-Point Grid``) in single-point CASTEP calculations on
the host unit cell or a saved supercell. Cutoffs are listed by hand or
generated from a range; k-point grids are listed by hand or generated as every
distinct grid CASTEP produces down to a minimum spacing. An examiner judges
convergence of the energy (Delta-Threshold, Derivative or Asymptotic
Extrapolation for cutoffs), the maximum force and, optionally, the maximum
stress, plots the results, and recommends a value. The ``Convergence Testing``
overview page shows both adopted defaults with the plots pinned when they were
adopted.

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

See :doc:`/user_guide/convergence_testing` (Adopting a result as a project
default), :doc:`/user_guide/quantum_corrections` and
:doc:`/user_guide/energy_barriers`. For the grid and spacing round trip, see
:doc:`/castep/kpoint_spacing_convention`. :doc:`/user_guide/examples/cof2`
sweeps and adopts a cutoff and a grid for CoF₂.

Pristine Relaxation
~~~~~~~~~~~~~~~~~~~

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

See :doc:`/user_guide/structural_relaxation_analysis` (Pristine Relaxation) and
:doc:`/user_guide/comparison_table`. :doc:`/user_guide/examples/cof2` compares a
relaxed CoF₂ cell with the host.

UEP
~~~

The ``UEP`` stage runs CASTEP single-point source calculations for the host
charge density and then analyses the unperturbed electrostatic potential: full
cell potential grids with isosurface and slice views, random-start batch
searches for potential minima, an optional harmonic zero-point estimate, and a
clustering window with hierarchical and k-means views. Cluster
representatives are offered to muon-site generation.

See :doc:`/user_guide/uep_workflow`. :doc:`/user_guide/examples/cof2` runs a
batch search and clustering in a model CoF₂ density.

Muon Sites: Generation
~~~~~~~~~~~~~~~~~~~~~~

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

See :doc:`/user_guide/site_generation`. :doc:`/user_guide/examples/cof2` stages
four F-attached and six random starts, and :doc:`/user_guide/examples/gdru2si2`
generates random starts under the paper's rules and enters the paper's two sites
as custom coordinates.

Muon Sites: Relaxations
~~~~~~~~~~~~~~~~~~~~~~~

``Muon Sites -> Relaxations`` holds one fixed-cell CASTEP geometry optimization
per staged position, in separate ``Positive muon`` and ``Muonium`` tabs. MuFinder
sets the CASTEP charge from the tab, writes muonium rows spin-polarised by
default, and gives each row its own supercell and parameters. Relaxations run
outside MuFinder can be imported as read-only rows.

A completed, failed, stopped or cancelled pristine or muon relaxation can be
continued from its CASTEP checkpoint with ``Continue relaxation…``, which edits
the row's parameters in place and resumes the same job in the same directory.
A completed relaxation's relaxed supercell can be saved as CIF or CASTEP
``.cell`` with ``Export Relaxed Structure`` in its row menu.

See :doc:`/user_guide/site_generation` (Creating Relaxation Jobs),
:ref:`continue-relaxation`, :ref:`export-relaxed-structures` and
:doc:`/user_guide/legacy_import`.
:doc:`/user_guide/examples/cof2` relaxes ten starts as positive muons and as
muonium.

Muon Sites: Analysis
~~~~~~~~~~~~~~~~~~~~

``Muon Sites -> Analysis`` collects completed relaxations of one charge state
into named workspaces. A workspace orders the relaxed sites by energy, clusters
them (connected components, hierarchical or k-means) into distinct sites with
fitted Wyckoff labels, analyses the host displacements around a site, and
exports the relaxed positions as CIF. ``Pin to Summary`` copies a workspace's
distinct sites to the ``Muon Sites`` summary page, which shows one card per
charge state with a ranked site table, a 3D preview, links back into the
analysis, and CIF export of the pinned sites. Quantum Corrections, Energy
Barriers and the Magnetic Fields muon-site dialog list the pinned sites by
default, with a ``Show every relaxed site`` option for the full list.

See :doc:`/user_guide/structural_relaxation_analysis`.
:doc:`/user_guide/examples/cof2` clusters its relaxations into three sites,
:doc:`/user_guide/examples/alpha_n2` finds one site and its displacements in
solid nitrogen, and :doc:`/user_guide/examples/gdru2si2` pins two sites for
its energy barriers.

Magnetic Fields
~~~~~~~~~~~~~~~

The ``Magnetic Fields`` workflow builds magnetic structures in the
propagation-vector formalism, including from the host spins and as skyrmion or
custom Fourier textures, and calculates at a chosen muon site. Components can be
edited, duplicated, reordered and disabled without deleting them. A panel shows
and edits the stored propagation-vector basis and which components use each
row, and a component q can be taken from a basis row or a combination of rows.
Guided helix, conical, cycloid and spin-density-wave presets are set from the
crystal axes, and a checklist tracks the build and flags disabled,
zero-moment and unpreviewed incommensurate structures. It calculates:

- the dipolar field of commensurate structures, optionally corrected for the
  host-ion displacements in a relaxed muon supercell at the site and its
  symmetry-equivalent sites, and the field distribution of incommensurate
  ones, with Auto, Grid or Monte Carlo multi-q phase sampling and convergence
  checks for lookup angles, samples and histogram bins; near-commensurate
  structures can instead list the finite fields of their commensurate
  approximant; commensurate results list their dipolar, Lorentz, contact,
  demagnetizing, distortion and applied terms;
- demagnetizing-field corrections for spherical, plate, ellipsoidal and
  rectangular-prism samples, from the model's bulk magnetisation or a measured
  one in μB per formula unit;
- dipolar tensors for selected magnetic ions;
- nuclear dipolar second moments in zero and transverse field, for single
  crystals and polycrystals, from bundled natural-abundance isotope data, with
  an optional muon-induced distortion correction.

Muon sites come from relaxation clustering, UEP clustering or a custom
fractional position; relaxation clusters are offered first once they exist,
and fields, dipolar tensors and nuclear relaxation computed at an unrelaxed
UEP site are labelled as estimates. Settings are saved as named workspaces, and
the latest results of the page and of each workspace are saved as project
results.

See :doc:`/user_guide/magnetic_fields`. :doc:`/user_guide/examples/cof2`
calculates the distortion-corrected field of an antiferromagnet and its nuclear
width, :doc:`/user_guide/examples/fe` the field of a ferromagnet at two
interstitial sites, and :doc:`/user_guide/examples/mnsi` the field distribution
of an incommensurate helix.

Quantum Corrections
~~~~~~~~~~~~~~~~~~~

``Quantum Corrections`` lists every successfully relaxed muon site and runs one
Gamma-point CASTEP phonon calculation for each, with explicit muon mass and
gyromagnetic ratio (finite displacement by default, with optional task
farming). Completed results give
the muon zero-point energy, covariance and correlation tensors, ``w1`` /
``w2`` witnesses and harmonic mode wells, shown in an
interactive analysis window. Together they say how much zero-point energy each
site carries, how far the muon spreads, and whether its motion is entangled
with the host, which decides how far a harmonic estimate can be trusted.

See :doc:`/user_guide/quantum_corrections` and
:doc:`/user_guide/quantum_corrections_theory`. :doc:`/user_guide/examples/alpha_n2`
reads the zero-point energy, widths and witnesses of a muon in solid nitrogen
from its analysis window; :doc:`/user_guide/examples/cof2` does the same for the
F–μ–F complex, and :doc:`/user_guide/examples/gdru2si2` compares the zero-point
energies of two sites with a barrier.

Energy Barriers
~~~~~~~~~~~~~~~

``Energy Barriers`` takes an ordered pair of relaxed muon sites, generates the
symmetry-distinct paths between them (including same-site hops, with an
optional maximum length), and builds a climbing-image NEB job for each chosen
path from the two relaxed endpoints. Paths can be previewed in the pristine
supercell, configured per row (including CASTEP task farming), and inspected
once complete: a results table lists forward and reverse barriers and ``ΔE``,
and an inspector scrubs the reaction-coordinate profile with a linked
structure viewer.

See :doc:`/user_guide/energy_barriers`. :doc:`/user_guide/examples/gdru2si2`
follows a real nudged elastic band between two muon sites and compares its
barrier with the zero-point energy; :doc:`/user_guide/examples/cof2` lists the
hops between 4c sites.

Journal
~~~~~~~

The ``Journal`` is a curated, day-grouped narrative of the project's
provenance: structure import, workflow configuration, project defaults, site
generation, job submissions, continuations and outcomes, comparison tables and
legacy imports. It exports as plain text or Markdown.

See :doc:`/user_guide/journal`.

Across the workflows
--------------------

Legacy relaxation import
~~~~~~~~~~~~~~~~~~~~~~~~

``File > Import Legacy Relaxations...`` imports muon structural relaxations
run by the legacy MuFinder application as read-only relaxation rows, so they
can be used in relaxation analysis, Quantum Corrections and Energy Barriers.
``File > Remove Imported Relaxations...`` removes an import batch that nothing
downstream still uses.

See :doc:`/user_guide/legacy_import`.

Jobs and execution
~~~~~~~~~~~~~~~~~~

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
- a ``Job Dashboard`` of the jobs that still need attention, with per-row
  ``Resubmit``, ``Retry retrieval``, ``Cancel``, ``Inspect``, ``Reset`` and
  ``Dismiss``, a failure cause on each failed or stopped row, the
  geometry-optimisation progress of each running relaxation (iteration and
  convergence criteria met, with a plot of every criterion against its
  tolerance), and safeguards that keep no job stuck in an active state;
- for relaxations, a choice on resubmission between continuing from the
  checkpoint and starting again.

See :doc:`/getting_started/running_jobs`, in particular Job tables and
selections, Running jobs, :ref:`geometry-optimisation-progress`,
:ref:`job-failure-reasons` and :ref:`resubmit-continue-or-start-again`.

CASTEP input and parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every workflow configures CASTEP through the same metadata-driven parameter
dialog: job-type-aware basic and advanced parameters, typed controls, help
from the keyword metadata, a search filter, and one dialog for several rows in
which values that differ are marked ``mixed`` and kept unless edited. The
shared parameters come first in every workflow. The keyword metadata comes
from a CASTEP keyword catalog generated from a local or cluster CASTEP, or
imported (see :doc:`/getting_started/castep_setup`). Workflows expand into
resolved job specifications that are rendered into deterministic ``.cell`` and
``.param`` files.

For developers
--------------

:doc:`/architecture` describes how projects are stored and saved (Domain model
and persistence), how the cross-platform appearance is achieved (Cross-platform
appearance), the Project defaults surface, and the CASTEP parameter dialogs,
Calculation job generation and Workflow parameter surface policy.
:doc:`/castep_keyword_catalog` and ``docs/castep/keyword_catalog.yaml`` define
the keyword metadata. The repository is organised for development by coding
agents; see :doc:`/agent_harness` and :doc:`/quality_and_automation` (Agent
harness checks).
