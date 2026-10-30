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

Each relaxed site runs as **one CASTEP phonon job**: a fresh ``TASK : PHONON``
run converges the ground state and then computes the Gamma-point phonons, so a
row is submitted once.

1. **Open the page.** Select ``Quantum Corrections`` in the workflow navigator.
   The ``Quantum Phonon Jobs`` table lists the pinned sites with
   ``Status`` = ``ready``.
2. **Configure before submitting.** Click a row's ``Configure…`` button, or
   select several rows and click ``Configure Selected Parameters…``. Check
   the electronic settings (inherited from the source relaxation, see
   `Configuration dialog`_) and set ``Farms`` if you want task farming. The
   settings stay editable until the row's job is submitted.
3. **Submit.** Select the rows and click ``Run Selected Locally`` or
   ``Run Selected on Cluster``. With nothing selected the buttons read
   ``Run All Locally`` / ``Run All on Cluster`` and submit every ready row. The
   row's ``Status`` then follows the job through the queue. Any execution
   target will do: the job needs no files from an earlier run.
4. **Review the result.** When the phonon job completes and its ``.phonon``
   file parses, a row appears under ``Completed Results`` with the muon ZPE,
   the three muon-mode frequencies, their widths and the ``w1`` witness
   (:doc:`/user_guide/quantum_corrections_theory` explains what they mean).
   Check the jobs dashboard for warnings, in particular the muon-mass warning
   (see `Muon mass`_).

   .. figure:: ../_generated/screenshots/n2_quantum_corrections_page.webp
      :alt: The Quantum Corrections page with one pinned alpha-N2 site, its
            phonon job completed, and a result row: ZPE 0.576 eV, muon modes
            2749.6, 3246.4 and 3301.1 cm-1, widths 0.23, 0.21 and 0.21
            Angstrom, and w1 -0.1377.

      The pinned site of :doc:`/user_guide/examples/alpha_n2` with its
      completed phonon job and its result under ``Completed Results``.
5. **Open the analysis.** Click ``Launch Analysis`` on a completed result to
   inspect the modes and atom-pair coupling (see
   `Phonon analysis window`_).
6. **Compare sites.** Repeat for the other distinct sites or charge states and
   compare their ZPEs and witnesses.

**Changing your mind or recovering from a failure.** Once a row's job has been
submitted its parameters are locked, so the stored inputs stay traceable to the
run. When ``Configure…`` opens a locked row read-only, the dialog says why in
one line. A row recovers through the shared failed-job reset, as in every
workflow table: ``Reset Failed Jobs…`` on the button row and in the
right-click menu returns a failed, stopped or cancelled phonon job to
``defined`` (see :doc:`/getting_started/running_jobs`). The reset asks first and
offers to clean the job directory:

- **Clean the directory** to start the row again. The rerun is a fresh phonon
  job and can go to any execution target.
- **Keep the files** to continue. The job writes ``CONTINUATION : DEFAULT``, so
  a rerun in the kept local directory resumes from the ``.check`` checkpoint
  when CASTEP wrote one there; clean the directory instead when you change
  parameters before the rerun. A cluster resubmission works out its remote
  directory again and asks before deleting one that already exists, so it
  starts again.

The row then reads ``defined`` with every parameter editable; submit it again.
Cancel an active job before resetting it. A completed phonon job is not reset:
to run the row again with other parameters, delete the job from the Job
Dashboard. A job still ``defined`` (staged but never submitted) stays editable,
because submitting the row rewrites its input files from the stored
configuration, and dismissing a failed job from the Job Dashboard reopens its
row, as for every workflow.

A project holding the singlepoint jobs that older versions ran before each
phonon job loads them as records; see
:ref:`singlepoint records <quantum-singlepoint-records>`.

The workflow page
-----------------

The figure under `Running the workflow`_ shows the page with one pinned site
and its completed result.

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
   finite-displacement calculations the phonon job needs, and therefore a
   guide to choosing ``Farms``.
``Source Job``
   The muon relaxation the row was built from.
``Farms``
   The configured farm count, ``1`` when task farming is off.
``Status``
   ``ready`` before the row has a job, then the phonon job's status. A row of
   an older project whose singlepoint is still in progress shows that job's
   status (see :ref:`singlepoint records <quantum-singlepoint-records>`).
``Parameters``
   ``Configure…`` (``Configured…`` once custom values are stored, or once
   the row is locked).

Below the table, ``Reset Failed Jobs…`` sits beside the run buttons and is
enabled while a selected row's job failed, was stopped or was cancelled.
Right-click selected rows to run, cancel, inspect a failure, reset failed jobs,
or configure them. There is no ``Remove``: rows are derived from completed
relaxations.

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

Rows are ordered by job name, ignoring case and with numbers in the name
compared as numbers (``site_2`` before ``site_10``); results of the same job
are listed oldest first, and a result whose job has been deleted comes last.
The order does not depend on which job finished first.

Configuration dialog
--------------------

The row configuration dialog follows the shared workflow layout: the common
rows ``Basis Precision`` / ``Cutoff Energy``, ``SCF Energy Tolerance``, and
``K-Point MP Grid`` come first, followed by the Quantum Corrections rows:

.. figure:: ../_generated/screenshots/n2_quantum_configuration_dialog.webp
   :alt: The Quantum Corrections configuration dialog for an alpha-N2 row:
         one farm, the locked PHONON task, the inherited 1200 eV cutoff,
         2 x 2 x 2 k-point grid and PBE functional, and an SCF energy
         tolerance of 1e-10 eV.

   The configuration of the α-N₂ row of :doc:`/user_guide/examples/alpha_n2`,
   inheriting its relaxation's settings, with the SCF tolerance tightened to
   ``1e-10`` eV.

- ``Farms`` — the ``NUM_FARMS`` value; ``1`` disables task farming.

The ``Task`` row is shown locked at ``PHONON``. There is no phonon-method
choice: every row runs finite displacement (see `CASTEP phonon methods`_).
Further CASTEP parameters supported for this workflow (for example
``NONSCF_EIGENVALUE_TOL``) can be added with ``Add Parameter``, which searches
keyword names, aliases, descriptions, and categories; ``PHONON_METHOD`` is not
among them.

**Rows configured by an older version.** A row an older version configured
for ``DFPT`` or ``LINEARRESPONSE`` still loads. Submitting it writes
``PHONON_METHOD : FINITEDISPLACEMENT``, also over a job of the row that an
older version wrote but never ran, and a one-line notice names the row and the
method it was configured with before the submission goes ahead. A row whose
job already has finite displacement in its ``.param`` is not announced again.
A completed job keeps its result, whichever method produced it.

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
   * - ``ready`` (no job, or its failed job dismissed) or ``defined``
     - Everything.
   * - Anything else, for any selected row
     - Nothing; the dialog opens read-only as ``Inspect Quantum Corrections
       Parameters`` so the stored inputs stay traceable to the submitted job.
       One line says why and what unlocks the row: cancel an active job and
       reset it, use ``Reset Failed Jobs…`` on a failed, stopped or cancelled
       job, or delete a completed job from the Job Dashboard.

What MuFinder writes
--------------------

Each row's phonon job is written to its own seed directory under
``files/quantum_corrections/<seed>/`` in the project. The seed is
``quantum_<site label>``; for a site relaxed in both charge states, the charge
state the site was not first relaxed in adds its prefix, as in
``quantum_O1_muonium``:

- ``TASK : PHONON``, ``PHONON_METHOD : FINITEDISPLACEMENT`` and
  ``CONTINUATION : DEFAULT`` (a new run when the directory holds no ``.check``
  file, a continuation of it otherwise);
- ``PHONON_KPOINT_MP_GRID 1 1 1`` (Gamma only, always);
- ``NONSCF_EIGENVALUE_TOL`` = the smaller of ``ELEC_ENERGY_TOL`` and
  ``1e-9 eV``;
- ``NUM_FARMS`` when ``Farms`` is greater than 1;
- ``SPECIES_MASS`` and ``SPECIES_GAMMA`` for ``H:mu``. The mass is the CODATA
  muon mass, 0.1134289257 amu, and the gyromagnetic ratio is the CODATA value, twice the muon magnetic moment over ħ, ≈ 8.516155 × 10⁸ rad s⁻¹ T⁻¹
  (γ\ :sub:`μ` / 2π ≈ 135.53881 MHz T⁻¹), both taken from ``scipy.constants``;
- ``CHARGE`` (``1`` for a positive muon, ``0`` for muonium).

A spin-polarised job writes the relaxed structure's per-atom ``SPIN=`` values,
which are the source relaxation's, so a magnetic host starts from the moments
it was relaxed with. For a muonium source the job also defaults to
``SPIN_POLARIZED : TRUE`` and seeds a spin of +1 on the muon, as the muon
relaxation does.

Source eligibility
------------------

A relaxed site is listed only when all of these hold (a pinned site is listed
when the relaxation that represents its cluster passes the same checks, in the
pin's charge state):

- the relaxation is the one the site records, or a relaxation of the site in
  the other charge state from its row in ``Muon Sites -> Relaxations``, so a
  site relaxed as both a positive muon and muonium is listed once per charge
  state;
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

MuFinder runs every Quantum Corrections phonon job at Gamma only
(``PHONON_KPOINT_MP_GRID 1 1 1``) by finite displacement, which, unlike
CASTEP's DFPT, works with ultrasoft pseudopotentials, DFT+U, meta-GGA and
hybrid functionals, non-collinear magnetism, spin-orbit coupling and the newer
dispersion corrections.

.. _quantum-corrections-muon-mass:

Muon mass
---------

The phonon job writes the muon ``SPECIES_MASS`` and ``SPECIES_GAMMA`` blocks
into its own ``.cell`` file, and a fresh CASTEP run reads them from there.
Without them CASTEP would give ``H:mu`` the default hydrogen mass, roughly 8.9
times the muon mass; because a harmonic frequency scales as the inverse square
root of the mass, the frequencies and the muon ZPE would come out about a
factor of three too low.

A run can still end up with another mass: when the row's settings override
``SPECIES_MASS``, or when it continued from a checkpoint written without the
muon mass (CASTEP restores species masses from the checkpoint on a
continuation). CASTEP records the mass it actually used in the ``.phonon``
header, so MuFinder compares it against the muon mass on every parsed phonon
run. A mass within 1 % of the muon mass is accepted: rounded values such as
0.1138 amu, common in runs prepared by hand, shift the frequencies by less than
0.2 %. A larger mismatch is raised as a job warning, stored with the result,
and shown as a banner in the analysis window. It cannot be repaired after the fact:
delete the job and its files from the Job Dashboard and run the row again.

Task farming and execution targets
----------------------------------

When ``Farms`` is greater than one, MuFinder writes ``NUM_FARMS`` into the
phonon ``.param`` file. CASTEP sets up the farms before the ground state, runs
the ground state once and shares it, and spreads the displacement calculations
over the farms. The ``Displacements`` column is a guide to a sensible value.
With task farming active MuFinder reads ``<seed>_farm001.castep`` as the
primary output.

A phonon job is a self-contained CASTEP run, so it can be submitted locally or
to any configured cluster, from the Quantum Corrections page or the Job
Dashboard. A job resubmitted into a directory that still holds its ``.check``
file continues from that checkpoint; see `Running the workflow`_ for how the
shared reset keeps or cleans the directory.

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
appears above the tabs. The window has two tabs, ``Phonon Modes`` and
``Muon Coupling``; it opens on ``Phonon Modes`` and computes ``Muon Coupling``
the first time it is shown. Each muon mode's width ``Δx`` is in the
``Completed Results`` table.

.. figure:: ../_generated/screenshots/n2_phonon_modes.webp
   :alt: The Phonon Modes tab of the phonon analysis window for the alpha-N2
         site, scrolled to the highest modes: the N2 stretches up to
         2428.7 cm-1, and the three muon modes at 2749.6, 3246.4 and
         3301.1 cm-1 with their participation highlighted, beside the
         supercell with the muon's mode vector.

   The ``Phonon Modes`` tab for the α-N₂ site of :doc:`/user_guide/examples/alpha_n2`,
   with the longitudinal muon mode selected.

``Phonon Modes``
   Lists every Gamma-point mode with its frequency, muon participation, and the
   muon's displacement direction (the participation of the three selected
   muon-dominated modes is highlighted), next to a 3D structure viewer (see
   :doc:`/user_guide/structure_viewer`). Selecting a mode updates the viewer. ``Amplitude (Å)`` (default ``1.00``) sets the largest displayed atomic
   displacement. ``Play animation`` oscillates the structure about equilibrium
   with the period set by ``Cycle time (s)``; ``Reset`` returns to the
   equilibrium phase; ``Show vectors`` (off by default) overlays the
   full-amplitude mode vectors as fixed reference arrows.

``Muon Coupling``
   Pick ``Atom A`` and ``Atom B`` by their one-based CASTEP index (Atom A
   defaults to the muon) and click ``Calculate``. MuFinder shows the zero-point
   displacement covariance tensor ``C_ab`` (Å²) and the normalized correlation
   tensor ``rho_ab``, built from the stable Gamma-point modes at zero
   temperature (modes with imaginary frequencies are left out), and
   the pairwise witness ``w2 = -|Tr C_ab| / sqrt(Tr C_aa Tr C_bb)``. ``w2`` lies
   between ``-1`` and ``0``; values closer to ``-1`` mean the two atoms'
   zero-point motions are more strongly correlated. It complements the
   result table's ``w1``. ``w2`` is the harmonic (phonon-covariance) estimate
   of the positional entanglement witness ``w2 = -|c|`` of Gomilšek *et al.*,
   where ``c`` is the Pearson correlation of the muon and nuclear
   displacements.

   .. figure:: ../_generated/screenshots/n2_muon_coupling.webp
      :alt: The Muon Coupling tab with Atom A 65: H:mu and Atom B 57: N, w2 =
            -0.310498, and the covariance tensor in square angstrom and the
            correlation tensor.

      The muon of the α-N₂ site and its nearest nitrogen.

The ``?`` button in the ``Muon Coupling`` tab shows the equations used.

Scope and limitations
---------------------

- Phonons are computed at Gamma only, in the harmonic approximation.
- The widths are those of the harmonic muon modes, the harmonic weakly-bound
  limit only for modes the muon alone moves in. MuFinder does not generate or
  import DFT line scans, so the anharmonic single-particle approximations of
  Gomilšek *et al.* are not available (see
  :doc:`/user_guide/quantum_corrections_theory`).
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
is :func:`~mufinder.science.quantum_phonons.muon_mass_warning`. Package layers
and the shared workflow infrastructure are described in :doc:`/architecture`.
