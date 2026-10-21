Quantum Corrections
===================

Quantum Corrections answers three questions about each distinct relaxed muon
site: how much zero-point energy (ZPE) the muon carries there, how far its
zero-point motion spreads it, and whether that motion is entangled with the
host nuclei. It is a follow-on workflow: its inputs are muon sites that MuFinder
has already relaxed, not unrelaxed candidates from site generation. The physics
behind every number, and a worked example on solid nitrogen, are on
:doc:`/user_guide/quantum_corrections_theory`.

Scientific intent
-----------------

MuFinder computes the Gamma-point phonons of each relaxed muonated supercell,
with the muon mass applied through ``SPECIES_MASS``, and treats the three modes
with the largest muon participation as the muon's own vibrations. Each of them
contributes a ZPE of ``(1/2) hbar omega`` and a width, their summed
participation gives the mode-mixing witness ``w1`` (zero when the muon modes do
not mix with host modes), and the analysis window adds the muon–nucleus
correlation ``w2``.

These are harmonic numbers, and how far to trust them depends on the site.
When ``w1`` is close to zero and the muon modes stand apart from the host
modes, they describe a muon moving on its own in a near-harmonic well. When
``w1`` is clearly negative or a muon mode is hybridised with host modes, the
muon is entangled with the host and single-particle pictures fail; if its well
is also strongly anharmonic, only path-integral molecular dynamics, which is
outside MuFinder, is reliable, and the harmonic ZPE is a starting estimate (see
:ref:`quantum-corrections-choosing` and `Scope and limitations`_). In every case
ZPEs are most reliable as comparisons between sites or charge states computed
with the same settings.

Before you start
----------------

Quantum Corrections needs:

- **Relaxed muon sites.** Run the structural relaxations under
  ``Muon Sites > Relaxations`` and let them complete. Every site whose
  relaxation completed successfully appears automatically on the Quantum
  Corrections page (see `Source eligibility`_). Relaxations brought in through
  :doc:`/user_guide/legacy_import` are eligible too.
- **A decision about which sites matter.** A phonon calculation is expensive
  (roughly three displacements per atom with finite displacement), so the page
  lists the distinct sites pinned to the ``Muon Sites`` summary by default:
  cluster the relaxed sites under ``Muon Sites > Analysis``
  (:doc:`/user_guide/structural_relaxation_analysis`) and choose ``Pin to Summary``. Each
  pinned site is one row, its cluster's representative relaxation. Tick
  ``Show every relaxed site`` to list every eligible relaxed site instead,
  including relaxed copies of the same site.
- **A working CASTEP setup.** Configure a local CASTEP executable or a cluster
  under ``Setup``, and load a keyword catalog (:doc:`/castep_keyword_catalog`).

Running the workflow
--------------------

Each relaxed site runs as a **two-stage job**: a stage-1 singlepoint, then a
stage-2 phonon calculation that continues from the singlepoint's checkpoint.
You submit each stage yourself; MuFinder does not chain them automatically.

1. **Open the page.** Select ``Quantum Corrections`` in the workflow navigator.
   The ``Quantum Phonon Jobs`` table lists the pinned sites with
   ``Stage`` = ``singlepoint`` and ``Status`` = ``ready``.
2. **Configure before the first submission.** Click a row's ``Configure...``
   button, or select several rows and click ``Configure Selected
   Parameters...``. Check the electronic settings (inherited from the source
   relaxation, see `Configuration dialog`_), choose the ``Phonon Method``, set
   ``Pseudopotentials`` to the family you use, and set ``Stage 2 farms`` if you
   want task farming. These settings apply to both stages and are **locked once
   stage 1 has been submitted**; changing them afterwards means resetting the
   row (see below).
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
   ``.check`` file, which exists only where the singlepoint ran: MuFinder
   neither downloads checkpoint files from a cluster nor uploads them to one.
   MuFinder therefore refuses, with an explanation, a stage-2 row sent to the
   other kind of target, and a cluster submission of stage-2 rows whose
   singlepoints ran on different clusters (submit each cluster's rows
   separately). ``Run Selected on Cluster`` opens the submission dialog with
   the stage-1 cluster already selected; if you pick another cluster there,
   the phonon job is not submitted. To move a row to a different target, use
   ``Reset to Stage 1`` and rerun both stages there.
5. **Review the result.** When the phonon job completes and its ``.phonon``
   file parses, a row appears under ``Completed Results`` with the muon ZPE,
   the three muon-mode frequencies, their widths and the ``w1`` witness
   (:doc:`/user_guide/quantum_corrections_theory` explains what they mean).
   Check the jobs dashboard for warnings, in particular the muon-mass warning
   (see `Muon mass across the two stages`_).
6. **Open the analysis.** Click ``Launch Analysis`` on a completed result to
   inspect the modes, the mode wells and widths, and atom-pair coupling (see
   `Phonon analysis window`_).
7. **Compare sites.** Repeat for the other distinct sites or charge states and
   compare their ZPEs and witnesses.

**Changing your mind or recovering from a failure.** Once stage 1 has been
submitted the shared parameters are locked so the stage-2 continuation stays
traceable to the singlepoint it continues. When ``Configure...`` opens a locked
row read-only, the dialog says why and names the reset that unlocks it. Two
row actions, on the button row and the right-click menu, reopen a row; each
asks for confirmation first, with ``No`` as the default:

- ``Reset to Stage 1`` deletes both staged job records and the row's output
  directory and returns the row to ``singlepoint`` / ``ready``, with every
  parameter editable and the stored configuration kept. It is offered for any
  row that has staged a job, whatever that job's state, unless the job is
  submitting, queued, running or being retrieved: cancel an active job first.
  A row whose directory is also used by a job outside the row is not reset,
  and the message names that folder.
- ``Reset to Stage 2`` is for a row whose singlepoint completed and whose
  phonon job failed, was stopped or was cancelled. It returns only the phonon
  job to ``defined`` and keeps the shared seed directory, including the
  stage-1 ``.check`` checkpoint the continuation reads. The row then reads
  ``phonon`` / ``defined``: change ``Stage 2 farms`` and submit it again
  without rerunning stage 1. A completed phonon job is not offered this reset,
  because its run has rewritten the checkpoint; use ``Reset to Stage 1``.

A stage-1 job that is still ``defined`` (staged but never submitted, or returned
there by the Job Dashboard's ``Reset``) stays fully editable: submitting the row
rewrites its input files from the stored configuration. Dismissing a failed job
from the Job Dashboard also reopens its stage, as for every workflow (see
:doc:`/getting_started/running_jobs`).

The Job Dashboard's own ``Reset`` returns a failed phonon job to ``defined``
like ``Reset to Stage 2``, and its optional directory cleanup leaves the shared
seed directory alone, because the completed singlepoint's ``.check``
checkpoint and ``.castep`` output in it are what the next stage-2 run continues
from. The message lists the folder as kept. Use ``Reset to Stage 1`` to start
the row over from a clean directory.

The workflow page
-----------------

.. figure:: ../_generated/screenshots/quantum_corrections_page.png
   :alt: The Quantum Corrections page with two pinned sites, the first with its phonon
         stage completed and the second ready for its singlepoint, and one completed
         result row showing the muon ZPE, three muon-mode frequencies, their widths
         and the w1 witness.

   Two pinned silicon sites, the first through both stages with its result under
   ``Completed Results``; the outputs are illustrative.

``Quantum Phonon Jobs`` lists one of two sets of rows:

- ``Pinned muon sites`` (the default): one row per site pinned to the
  ``Muon Sites`` summary, positive muon first, then muonium, each in the
  summary's energy rank. A site the workflow already tracks (a configured row
  or one with a job) stays listed even when it is not pinned, so a submitted
  row never drops out of the table.
- ``Eligible relaxed sites``, when ``Show every relaxed site`` is ticked: one
  row per eligible relaxed site.

The choice is saved with the project. When no charge state has a pin, the
table is empty and a notice with ``Go to Muon Sites →`` explains how to pin
sites. The notice also names a charge state with eligible sites but no pin,
and counts pinned sites that no longer resolve to an eligible relaxed site
(their relaxation was reset or removed); re-pin the analysis to refresh them.

``Site``
   For a pinned row, the summary label: the Wyckoff label and cluster, such as
   ``8c · cluster 1`` (``Cluster 1`` without a Wyckoff label); the tooltip
   names the relaxed site. Otherwise the relaxed site's label: double-click to
   rename it; the new label is also used by Energy Barriers.
``Charge``
   The charge state of the source relaxation: ``positive muon`` or
   ``muonium``.
``dE (eV)``
   The site's relaxed total energy relative to the lowest-energy eligible site
   with the same charge state and cell composition. Energies from different
   charge states or supercells are not comparable, so each such group has its
   own zero. In the full list, rows are ordered by charge state, then by
   ``dE``.
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
reset them to stage 1 or stage 2. There is no ``Remove``: rows are derived from
completed relaxations.

``Completed Results`` (``Completed phonon analyses``) has one row per completed
phonon job: ``Job``; ``ZPE (eV)``, the total muon zero-point energy, ``½ħω``
summed over the three selected muon-dominated Gamma modes, to three decimals;
``Muon modes (cm⁻¹)``, the frequencies of those three modes in ascending order;
``Width (Å)``, the muon's zero-point width ``sqrt(ħ / (2 m_μ ω))`` along each of
those modes, in the same order; ``w1``, the mode-mixing witness ``P - 3`` (0
when the muon's motion is contained in its three modes, more negative as it
mixes with host modes, down to -3); ``Status`` (the producing job's status,
coloured like the jobs table); and an ``Analysis`` column with the
``Launch Analysis`` button. Hover a column header for its definition. Every
Gamma-point mode is listed in the analysis window's ``Phonon Modes`` tab.

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
The functional and cutoff come from the source ``.param`` and the grid from the
source ``.cell``, so a later change of project default does not affect them.
The spin treatment is taken as CASTEP ran it: the ``.param`` file's
``SPIN_POLARIZED`` (either spelling) when present, otherwise on when the run's
net ``SPIN`` or any per-atom ``SPIN=`` token in its ``.cell`` is non-zero. When
the source ``.param`` cannot be read, a muonium source still defaults to
spin-polarised, matching how its relaxation was written.
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
   * - ``singlepoint`` / ``ready`` or ``defined``
     - Everything.
   * - ``phonon`` / ``ready`` or ``defined`` (stage 1 complete, stage 2 not
       yet submitted, reset to stage 2, or its failure dismissed)
     - ``Stage 2 farms`` only.
   * - Anything else, or a selection mixing the two rows above
     - Nothing; the dialog opens read-only as ``Inspect Quantum Corrections
       Parameters`` so the stored inputs stay traceable to the submitted job.
       Its text says why and names the action that unlocks the row: cancel an
       active job, ``Reset to Stage 2`` after a phonon failure, or ``Reset to
       Stage 1``.

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

A relaxed site is listed only when all of these hold (a pinned site is listed
when the relaxation that represents its cluster passes the same checks, in the
pin's charge state):

- the site has been relaxed and records its relaxation run;
- that run, and the job that produced it, completed successfully;
- the job is a muon relaxation from ``Muon Sites -> Relaxations``, or a
  relaxation brought in through :doc:`/user_guide/legacy_import`;
- the run produced a parsed relaxed structure;
- the relaxed structure contains an ``H:mu`` atom.

UEP candidates, generated sites, failed relaxations, pristine relaxations, and
manual sites without a structural relaxation are excluded because their ionic
coordinates have not been relaxed for a phonon calculation.

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

MuFinder checks only the pseudopotential family, when you submit a row at
either stage (the method is fixed once stage 1 has been submitted, so a
problem is caught before the singlepoint runs):

- with ``Pseudopotentials`` set to ``Ultrasoft``, a DFPT or linear-response row
  is refused with a message naming the rows; change the row to
  ``FINITEDISPLACEMENT`` (or reset a stage-2 row to stage 1 first), or set
  ``Pseudopotentials`` to ``Norm-conserving`` if that is what you use;
- with ``Unknown / not set``, MuFinder lists the DFPT rows and the methods
  CASTEP's DFPT does not support, and asks whether to submit anyway (the
  default answer is no). The reminder is also recorded in the job's
  provenance.

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
  ``.phonon`` header, so MuFinder compares it against the muon mass on every
  parsed phonon run. A mismatch is raised as a job warning,
  stored with the result, and shown as a banner in the analysis window.
  This catches the rows the pre-submission guard cannot, including projects whose
  stage-1 singlepoint was written without the block.

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
already there. A phonon job is never submitted anywhere else: the Quantum
Corrections page refuses the row before any dialog opens, and the local and
cluster submission backends refuse it again on every other route, such as the
jobs dashboard or choosing a different cluster in the submission dialog (see
step 4 of `Running the workflow`_).

Outputs and interpretation
--------------------------

Each completed phonon job saves a result with:

- the source relaxed site and the relaxation job it came from;
- the path of the CASTEP ``.phonon`` file and whether it is Gamma-only;
- every Gamma-point eigenvalue in ``cm-1``;
- the three selected modes: index, frequency, muon participation, ZPE, and the
  muon's zero-point width along the mode, ``Δx = sqrt(ħ / (2 m_μ ω))`` in Å
  with ``ω = 2π c ν``;
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
- **w1**: ``P - 3``, where ``P`` is the summed participation of the three
  selected modes, following M. Gomilšek et al., *Commun. Phys.* **6**, 142
  (2023), doi:10.1038/s42005-023-01260-7. It is ``0`` when the muon's motion is
  fully contained in the three modes, and becomes more negative as muon motion
  mixes with host modes, down to ``-3``. A stored result that does not record
  the witness definition (``w1_definition``) is shown with ``w1`` recomputed
  from its stored mode participations.
- **Zero-point width** of a selected mode: ``Δx = sqrt(ħ / (2 m_μ ω))``, the
  root-mean-square ground-state displacement of a harmonic oscillator with the
  muon mass at that mode's frequency. A mode with a muon ZPE of 0.17 eV
  (about 2742 ``cm-1``) has a width of about 0.23 Å; a non-positive frequency
  has width zero.

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

``Mode wells and widths``
   For each of the three selected modes, builds the one-dimensional harmonic
   well ``V(q) = (1/2) m_mu omega^2 q^2`` along the muon's direction in that
   mode from the mode frequency and the muon mass, solves the finite-difference
   Schrödinger equation over ``±Range (Å)`` (default ``2.0``, 401 grid points),
   and reports the ground-state energy (``ZPE (eV)``) and the RMS width of the
   probability density (``Width (Å)``). For a mode the muon alone moves in
   (participation close to 1) this is the harmonic, weakly-bound limit of the
   single-particle approximation of Gomilšek *et al.* (host atoms fixed,
   effective mass = muon mass); a mode shared with host atoms is an eigenmode
   of the coupled dynamical matrix, so its well is a harmonic estimate along
   the mode direction rather than a fixed-host muon potential. Once the range
   resolves the oscillator, as the default does for muon modes, the
   ground-state energy converges to ``(1/2) hbar omega``, the value in the main
   table; a range comparable to the width confines the muon further and raises
   both numbers. The value of the tab is the width
   ``Δx = sqrt(hbar / (2 m_mu omega))`` and the plot of the potential,
   ground-state level, and probability density ``|ψ|²`` for each mode.
   ``Energy scale`` switches the plot between ``Energy`` and ``Sqrt energy``,
   which keeps shallow wells legible.

``Muon Coupling``
   Pick ``Atom A`` and ``Atom B`` by their one-based CASTEP index (Atom A
   defaults to the muon) and click ``Calculate``. MuFinder shows the zero-point
   displacement covariance tensor ``C_ab`` (Å²) and the normalized correlation
   tensor ``rho_ab``, built from the Gamma-point modes at zero temperature, and
   the pairwise witness ``w2 = -|Tr C_ab| / sqrt(Tr C_aa Tr C_bb)``. ``w2`` lies
   between ``-1`` and ``0``; values closer to ``-1`` mean the two atoms'
   zero-point motions are more strongly correlated. It complements the
   result table's ``w1``. ``w2`` is the harmonic (phonon-covariance) estimate
   of the positional entanglement witness ``w2 = -|c|`` of Gomilšek *et al.*,
   where ``c`` is the Pearson correlation of the muon and nuclear
   displacements.

The ``?`` buttons in the ``Mode wells and widths`` and ``Muon Coupling`` tabs show
the equations used.

Scope and limitations
---------------------

- Phonons are computed at Gamma only, in the harmonic approximation.
- The ``Mode wells and widths`` tab uses harmonic wells built from the phonon
  frequencies, which are the harmonic weakly-bound limit only for modes the
  muon alone moves in. MuFinder does not generate or import DFT line scans, so
  the anharmonic single-particle approximations of Gomilšek *et al.* are not
  available in the application (see `For developers`_).
- Path-integral molecular dynamics (PIMD) is out of scope for MuFinder.

References
----------

- CASTEP documentation, *Running phonon calculations*:
  https://castep-docs.github.io/castep-docs/documentation/Phonons/Castep_Phonons/Running-phonon-calculations/
- M. Gomilšek, F. L. Pratt, S. P. Cottrell, S. J. Clark, and T. Lancaster,
  *Many-body quantum muon effects and quadrupolar coupling in solids*,
  Commun. Phys. **6**, 142 (2023), DOI: ``10.1038/s42005-023-01260-7``.

For developers
--------------

The check behind `Source eligibility`_ is
:func:`mufinder.workflow.quantum.quantum_source_eligibility`, and the mass check
is :func:`~mufinder.science.quantum_phonons.muon_mass_warning`. The science
layer also provides a general solver for tabulated 1D potentials
(:func:`~mufinder.science.quantum_adiabatic.solve_single_particle_ground_state`)
and the strongly-bound effective mass ``m_mu + sum_i m_i |dR_i/dx_mu|^2``
(:func:`~mufinder.science.quantum_adiabatic.effective_mass_from_relaxed_scan`),
following the single-particle approximations of Gomilšek *et al.*; no workflow
feeds them yet. Package layers and the shared workflow infrastructure are
described in :doc:`/architecture`.
