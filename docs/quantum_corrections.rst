Quantum Corrections
===================

Quantum Corrections estimates the muon's zero-point energy (ZPE) and
phonon-derived correlation metrics from CASTEP Gamma-point phonon calculations.
It is a follow-on workflow: its inputs are muon sites that MuFinder has already
relaxed, not unrelaxed candidates from site generation.

Scientific intent
-----------------

This workflow is aimed at the regime where the muon already has a well-defined,
locally relaxed stopping site and the next question is how strongly local
vibrational motion modifies that picture. MuFinder therefore computes
Gamma-point phonons of the relaxed defect cell and extracts the three modes
with the largest muon participation. The resulting zero-point estimate is a
local, harmonic approximation to the muon's vibrational ground-state
contribution, and the witness metrics summarize how strongly the muon motion is
correlated with the host modes in that reduced description.

Because the estimate is harmonic and Gamma-only, Quantum Corrections is best
used as a comparative tool between already-relaxed sites or charge states,
rather than as a standalone predictor of absolute quantum energies. See
`Scope and limitations`_.

Before you start
----------------

Quantum Corrections needs:

- **Relaxed muon sites.** Run the structural relaxations under
  ``Muon Sites > Relaxations`` and let them complete. Every site whose
  relaxation completed successfully appears automatically on the Quantum
  Corrections page (see `Source eligibility`_). Relaxations brought in through
  :doc:`legacy_import` are eligible too.
- **A decision about which sites matter.** The page lists *every* eligible
  relaxed site, ordered by relative energy, and a phonon calculation is
  expensive (roughly three displacements per atom with finite displacement).
  Use ``Muon Sites > Analysis`` (:doc:`structural_relaxation_analysis`) to group
  the relaxed sites into distinct stopping sites first, then run Quantum
  Corrections on one representative of each site you want to compare, not on
  every row.
- **A working CASTEP setup.** Configure a local CASTEP executable or a cluster
  under ``Setup``, and load a keyword catalog (:doc:`castep_keyword_catalog`).

Running the workflow
--------------------

Each relaxed site runs as a **two-stage job**: a stage-1 singlepoint, then a
stage-2 phonon calculation that continues from the singlepoint's checkpoint.
You submit each stage yourself; MuFinder does not chain them automatically.

1. **Open the page.** Select ``Quantum Corrections`` in the workflow navigator.
   The ``Quantum Phonon Jobs`` table lists the eligible relaxed sites with
   ``Stage`` = ``singlepoint`` and ``Status`` = ``ready``.
2. **Configure before the first submission.** Click a row's ``Configure...``
   button, or select several rows and click ``Configure Selected
   Parameters...``. Check the electronic settings (inherited from the source
   relaxation, see `Configuration dialog`_), choose the ``Phonon Method``, set
   ``Pseudopotentials`` to the family you use, and set ``Stage 2 farms`` if you
   want task farming. These settings apply to both stages and are **frozen once
   stage 1 has been submitted**, so get them right now.
3. **Submit stage 1.** Select the rows and click ``Run Selected Locally`` or
   ``Run Selected on Cluster``. With nothing selected the buttons read
   ``Run All Locally`` / ``Run All on Cluster`` and submit every ready row. The
   row's ``Status`` then follows the job through the queue.
4. **Submit stage 2 on the same target.** When the singlepoint completes the
   row changes to ``Stage`` = ``phonon`` with ``Status`` = ``ready`` and the
   ``Farms`` column fills in. You can still change ``Stage 2 farms`` at this
   point (nothing else). Select the row and submit it again **to the same
   execution target as stage 1** — the same local machine, or the same
   cluster. The phonon run is a CASTEP continuation and needs the stage-1
   ``.check`` file, which exists only where the singlepoint ran; MuFinder does
   not download checkpoint files from a cluster, so switching between local and
   cluster (or between clusters) leaves the continuation without its
   checkpoint and the CASTEP run fails.
5. **Review the result.** When the phonon job completes and its ``.phonon``
   file parses, a row appears under ``Completed Results`` with the muon ZPE,
   the Gamma-point eigenvalues, and the ``w1`` witness. Check the jobs
   dashboard for warnings, in particular the muon-mass warning (see
   `Muon mass across the two stages`_).
6. **Open the analysis.** Click ``Launch Analysis`` on a completed result to
   inspect the modes, the single-particle solver, and atom-pair coupling (see
   `Phonon analysis window`_).
7. **Compare sites.** Repeat for the other distinct sites or charge states and
   compare their ZPEs and witnesses.

**Changing your mind or recovering from a failure.** Once stage 1 has been
submitted the shared parameters are locked so the stage-2 continuation stays
traceable to the singlepoint it continues. To change them, or to retry a row
whose singlepoint or phonon job failed, was stopped, or was cancelled, select
the row and click ``Reset to Stage 1`` (also on the right-click menu). After a
confirmation, MuFinder deletes both staged job records and the row's output
directory and returns the row to ``singlepoint`` / ``ready``. Rows whose job is
still running cannot be reset.

The workflow page
-----------------

``Quantum Phonon Jobs`` (``Eligible relaxed sites``) has one row per eligible
relaxed site:

``Site``
   The relaxed site's label. Double-click to rename it; the new label is also
   used by Energy Barriers.
``Charge``
   The charge state of the source relaxation: ``positive muon`` or
   ``muonium``.
``dE (eV)``
   The site's relaxed total energy relative to the lowest-energy eligible site
   with the same charge state and cell composition. Energies from different
   charge states or supercells are not comparable, so each such group has its
   own zero. Rows are ordered by charge state, then by ``dE``.
``Displacements``
   ``3 x atom count`` for the relaxed defect cell — an approximate count of the
   finite-displacement calculations the phonon stage needs, and therefore a
   guide to choosing ``Stage 2 farms``.
``Source Job``
   The muon relaxation the row was built from.
``Stage``
   ``singlepoint`` or ``phonon``.
``Farms``
   The stage-2 farm count; shown once the row reaches the phonon stage.
``Status``
   The current stage's job status.
``Parameters``
   ``Configure...`` (``Configured...`` once custom values are stored, or once
   the row is locked).

Right-click selected rows to run, cancel, inspect a failure, configure, or
reset them to stage 1. There is no ``Remove``: rows are derived from completed
relaxations.

``Completed Results`` (``Completed phonon analyses``) has one row per completed
phonon job: ``Job``, ``ZPE`` (total muon ZPE of the three selected modes, eV),
``Eigenvalues (cm⁻¹)`` (every Gamma-point mode as ``index:frequency``), ``w1``,
``Status`` (the producing job's status, coloured like the jobs table), and an
``Analysis`` column with the ``Launch Analysis`` button.

Configuration dialog
--------------------

The row configuration dialog follows the shared workflow layout: the common
rows ``Basis Precision`` / ``Cutoff Energy``, ``SCF Energy Tolerance``, and
``K-Point MP Grid`` come first, followed by the Quantum Corrections rows:

- ``Phonon Method`` — ``FINITEDISPLACEMENT`` (default), ``DFPT``, or
  ``LINEARRESPONSE`` (see `CASTEP phonon methods`_);
- ``Pseudopotentials`` — ``Unknown / not set``, ``Norm-conserving``, or
  ``Ultrasoft``; MuFinder uses this only for the DFPT compatibility check;
- ``Stage 2 farms`` — the ``NUM_FARMS`` value for the phonon stage; ``1``
  disables task farming.

The ``Task`` row is shown locked: MuFinder writes ``SINGLEPOINT`` for stage 1
and ``PHONON`` for stage 2. Further CASTEP parameters supported for this
workflow (for example ``NONSCF_EIGENVALUE_TOL``) can be added with
``Add Parameter``, which searches keyword names, aliases, descriptions, and
categories.

**Defaults.** When the source relaxation's input files are still on disk, each
row inherits that relaxation's ``CUT_OFF_ENERGY``, ``KPOINT_MP_GRID``,
``XC_FUNCTIONAL``, and spin treatment (``SPIN_POLARIZED``), so the phonon
calculation stays comparable with the relaxation that produced the geometry.
``SCF Energy Tolerance`` defaults to ``1e-9 eV``, tighter than for other
workflows, because phonon forces need a well-converged ground state. Values you
set in the dialog override the inherited defaults for that row only.

**Several rows at once.** With more than one row selected, values that differ
between rows are marked as mixed and are left untouched unless you edit them.

**When the dialog is editable.**

.. list-table::
   :header-rows: 1

   * - Row state
     - What you can change
   * - ``singlepoint`` / ``ready``
     - Everything.
   * - ``phonon`` / ``ready`` (stage 1 complete, stage 2 not yet generated)
     - ``Stage 2 farms`` only.
   * - Anything else
     - Nothing; the dialog opens read-only as ``Inspect Quantum Corrections
       Parameters`` so the stored inputs stay traceable to the submitted job.

What MuFinder writes
--------------------

Both stages share one seed directory under
``files/quantum_corrections/<seed>/`` in the project, so the continuation finds
the stage-1 checkpoint.

.. list-table::
   :header-rows: 1

   * - Stage 1 (singlepoint)
     - Stage 2 (phonon continuation)
   * - ``TASK : SINGLEPOINT``
     - ``TASK : PHONON``, ``CONTINUATION : DEFAULT``, ``PHONON_METHOD``
   * -
     - ``PHONON_KPOINT_MP_GRID 1 1 1`` (Gamma only, always)
   * -
     - ``NONSCF_EIGENVALUE_TOL`` = the smaller of ``ELEC_ENERGY_TOL`` and
       ``1e-9 eV``
   * -
     - ``NUM_FARMS`` when ``Stage 2 farms`` is greater than 1
   * - ``SPECIES_MASS`` and ``SPECIES_GAMMA`` for ``H:mu``
     - ``SPECIES_MASS`` and ``SPECIES_GAMMA`` for ``H:mu``
   * - ``CHARGE`` (``1`` for a positive muon, ``0`` for muonium)
     - ``CHARGE`` (same)

For a muonium source both stages also default to ``SPIN_POLARIZED : TRUE`` and
seed a spin of +1 on the muon, as the muon relaxation does.

Source eligibility
------------------

A relaxed site is listed only when:

- the ``MuonSite`` status is ``relaxed`` and the site records its relaxation run;
- that run, and the job that produced it, completed successfully;
- the job belongs to ``WorkflowStage.MUON_RELAXATION``;
- the run produced a parsed relaxed structure with
  ``origin == "muon_relaxation"`` and muon-relaxation provenance;
- the relaxed structure contains an ``H:mu`` atom.

UEP candidates, generated sites, failed relaxations, pristine relaxations, and
manual sites without structural-relaxation provenance are excluded because
their ionic coordinates have not been relaxed for a phonon calculation. The
check is :func:`mufinder.workflow.quantum.quantum_source_eligibility`.

CASTEP phonon methods
---------------------

MuFinder always runs Quantum Corrections phonons at Gamma only via
``PHONON_KPOINT_MP_GRID 1 1 1``. CASTEP offers two method families:

Finite displacement (``FINITEDISPLACEMENT``)
  MuFinder's default, and the recommended choice. It works with ultrasoft
  pseudopotentials and with the methods DFPT does not support. (CASTEP's own
  default is DFPT; MuFinder always writes ``PHONON_METHOD`` explicitly.)

DFPT / linear response (``DFPT``, ``LINEARRESPONSE``)
  Advanced option. The CASTEP documentation states that DFPT phonons are not
  available with ultrasoft pseudopotentials, DFT+U, meta-GGA or hybrid
  functionals, non-collinear magnetism or spin-orbit coupling, or the newer
  dispersion corrections.

MuFinder checks only the pseudopotential family:

- with ``Pseudopotentials`` set to ``Ultrasoft``, a DFPT or linear-response
  phonon job is not generated, and the submission ends with a ``No ready jobs``
  message;
- with ``Unknown / not set``, the job is generated and a reminder to verify
  norm-conserving pseudopotentials is recorded in the job's provenance.

It does not detect Hubbard U, hybrid functionals, spin-orbit coupling, or
dispersion corrections in your parameters; if you use any of these, choose
finite displacement.

Muon mass across the two stages
-------------------------------

Both stages write the muon ``SPECIES_MASS`` and ``SPECIES_GAMMA`` blocks into
their ``.cell`` file, and this is not redundant. Stage 2 runs as a
``CONTINUATION``, and CASTEP rebuilds the model — species masses included — from
the stage-1 ``.check`` file in the shared seed directory. A mass declared only in
the stage-2 cell is therefore ignored: the phonon calculation runs with the
default hydrogen mass for ``H:mu``, roughly 8.9 times the muon mass. Because a
harmonic frequency scales as the inverse square root of the mass, the reported
frequencies and the muon ZPE come out about a factor of three too low, and
displacement amplitudes are wrong by the same square-root factor. Writing the
blocks in the stage-1 singlepoint is what puts the muon mass into the checkpoint
the continuation reads; masses do not affect a singlepoint SCF energy, so this is
inert for the stage-1 result itself.

Two guards protect against a run made at the wrong mass:

- **Before submission.** A stage-2 phonon continuation is refused when the
  stage-1 singlepoint ``.cell`` still on disk has no ``SPECIES_MASS`` block, with
  an explanation pointing at ``Reset to Stage 1``. The guard fails open — a
  missing or unparseable cell never blocks a submission — and it has one blind
  spot: both stages write ``{seed}.cell`` to the same path, so a row whose
  stage-2 job was already materialized no longer has stage-1 evidence on disk.
- **After completion.** CASTEP records the mass it actually used in the
  ``.phonon`` header, so
  :func:`~mufinder.science.quantum_phonons.muon_mass_warning` compares it against
  the muon mass on every parsed phonon run. A mismatch is raised as a job warning,
  recorded in the result payload, and shown as a banner in the analysis window.
  This catches the rows the pre-submission guard cannot, including projects whose
  stage 1 ran before both stages emitted the block.

Neither guard can repair an affected run: the row has to be reset to stage 1 and
both stages rerun.

Task farming and execution targets
----------------------------------

When ``Stage 2 farms`` is greater than one, MuFinder writes ``NUM_FARMS`` into
the phonon ``.param`` file so CASTEP spreads the displacement calculations over
that many farms. The ``Displacements`` column is a guide to a sensible value.
With task farming active MuFinder reads ``<seed>_farm001.castep`` as the
primary output.

For cluster runs, MuFinder reuses the completed stage-1 remote directory when
the phonon stage is submitted to the same cluster. It uploads refreshed
``.cell``, ``.param``, and ``run.sh`` files without asking to delete the remote
directory first, preserving the singlepoint outputs and the ``.check`` file
already there. Submitting stage 2 anywhere else starts from a directory without
the checkpoint (see step 4 of `Running the workflow`_).

Outputs and interpretation
--------------------------

Each completed phonon job stores a result with:

- the source relaxed site, relaxation run, and relaxation job;
- the path of the CASTEP ``.phonon`` file and whether it is Gamma-only;
- every Gamma-point eigenvalue in ``cm-1``;
- the three selected modes: index, frequency, muon participation, and ZPE;
- the total muon ZPE of those modes in eV;
- the ``w1`` witness;
- warnings, when the muon mass is wrong or a selected mode has a non-positive
  frequency.

The quantities are defined as follows (``e`` is the normalized phonon
eigenvector written in the ``.phonon`` file):

- **Muon participation** of mode *n*: ``|e_mu,n|^2``, the squared norm of the
  muon's three eigenvector components.
- **Selected modes**: the three Gamma modes with the largest muon participation,
  listed in order of frequency.
- **Muon ZPE**: ``sum (1/2) h c nu_n`` over the selected modes, with ``nu_n``
  in ``cm-1``. A selected mode with a non-positive frequency contributes zero
  and raises a warning: an imaginary muon mode means the relaxed geometry is not
  a true minimum, and the ZPE should not be trusted.
- **w1**: ``-sqrt(3 - P)``, where ``P`` is the summed participation of the three
  selected modes. It is ``0`` when the muon's motion is fully contained in three
  modes, and becomes more negative as muon motion is shared with more host
  modes.

The ZPE is a harmonic, Gamma-point estimate for a local defect, not a full
anharmonic many-body treatment; compare it between sites computed with the
same settings rather than reading it as an absolute value.

Phonon analysis window
----------------------

``Launch Analysis`` opens a window for one completed phonon job. It reparses the
stored ``.phonon`` file, so the analysis is tied to the archived CASTEP output
rather than to table state. If CASTEP used the wrong muon mass, a warning banner
appears above the tabs.

``Phonon Modes``
   Lists every Gamma-point mode with its frequency, muon participation, and the
   muon's displacement direction (the participation of the three selected
   muon-dominated modes is highlighted), next to a 3D structure viewer. Selecting a mode updates the
   viewer. ``Amplitude (Å)`` (default ``1.00``) sets the largest displayed atomic
   displacement. ``Play animation`` oscillates the structure about equilibrium
   with the period set by ``Cycle time (s)``; ``Reset`` returns to the
   equilibrium phase; ``Show vectors`` (off by default) overlays the
   full-amplitude mode vectors as fixed reference arrows.

``Adiabatic Solver``
   For each of the three selected modes, builds the one-dimensional harmonic
   potential ``V(q) = (1/2) m_mu omega^2 q^2`` along the muon's direction in that
   mode (host atoms fixed, effective mass = muon mass), solves the
   finite-difference Schrödinger equation over ``±Range (Å)`` (default
   ``2.0``), and reports the ground-state energy (``ZPE (eV)``) and the RMS
   width of the probability density (``Width (Å)``). Because the potential is
   harmonic, the solver's ZPE reproduces ``(1/2) hbar omega`` for each mode; the
   value of the tab is the width and the plot of potential, ground-state
   level, and probability density. ``Energy scale`` switches the plot between
   ``Energy`` and ``Sqrt energy``, which keeps shallow wells legible.

``Muon Coupling``
   Pick ``Atom A`` and ``Atom B`` by their one-based CASTEP index (Atom A
   defaults to the muon) and click ``Calculate``. MuFinder shows the zero-point
   displacement covariance tensor ``C_ab`` (Å²) and the normalized correlation
   tensor ``rho_ab``, built from the Gamma-point modes at zero temperature, and
   the pairwise witness ``w2 = -|Tr C_ab| / sqrt(Tr C_aa Tr C_bb)``. ``w2`` lies
   between ``-1`` and ``0``; values closer to ``-1`` mean the two atoms'
   zero-point motions are more strongly correlated. It complements the
   result table's ``w1``.

The ``?`` buttons in the ``Adiabatic Solver`` and ``Muon Coupling`` tabs show
the equations used.

Scope and limitations
---------------------

- Phonons are computed at Gamma only, in the harmonic approximation.
- The ``Adiabatic Solver`` tab uses harmonic, phonon-derived potentials. The
  science layer also provides a general solver for tabulated 1D potentials
  (:func:`~mufinder.science.quantum_adiabatic.solve_single_particle_ground_state`)
  and the strongly-bound effective mass ``m_mu + sum_i m_i |dR_i/dx_mu|^2``
  (:func:`~mufinder.science.quantum_adiabatic.effective_mass_from_relaxed_scan`),
  following the single-particle approximations of Gomilšek *et al.*, but
  MuFinder does not yet generate or import the line-scan calculations that
  would feed them, so anharmonic potentials are not available from the UI.
- Path-integral molecular dynamics (PIMD) is out of scope for MuFinder.

References
----------

- CASTEP documentation, *Running phonon calculations*:
  https://castep-docs.github.io/castep-docs/documentation/Phonons/Castep_Phonons/Running-phonon-calculations/
- M. Gomilšek, F. L. Pratt, S. P. Cottrell, S. J. Clark, and T. Lancaster,
  *Many-body quantum muon effects and quadrupolar coupling in solids*,
  Commun. Phys. **6**, 142 (2023), DOI: ``10.1038/s42005-023-01260-7``.
