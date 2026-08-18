Quantum Corrections
===================

Quantum Corrections estimates muon zero-point and phonon-derived correlation
metrics from CASTEP Gamma-point phonon calculations. The workflow is designed
for relaxed MuFinder muon sites, not for unrelaxed candidate generation output.

Scientific intent
-----------------

This workflow is aimed at the regime where the muon already has a well-defined,
locally relaxed stopping site and the next question is how strongly local
vibrational motion modifies that picture. MuFinder therefore focuses on
Gamma-point phonons of the relaxed defect cell and extracts the modes with the
largest muon participation. The resulting zero-point estimate is a local,
harmonic approximation to the muon's vibrational ground-state contribution, and
the witness metrics summarize how strongly the muon motion is correlated with
the host modes retained in that reduced description.

Using the workflow page
-----------------------

The Quantum Corrections page is split into two user-facing sections:

- ``Quantum Phonon Jobs`` lists eligible relaxed muon sites and their current
  Quantum Corrections stage.
- ``Completed Results`` lists completed phonon jobs with the parsed ZPE,
  Gamma-point phonon eigenvalues, witness summary, a ``Status`` column (the
  producing job's status, coloured with the same per-status palette as the
  ``Quantum Phonon Jobs`` table), and an analysis launcher.

Typical use is:

1. finish the structural muon relaxation workflow and confirm the site is in a
   relaxed state;
2. open Quantum Corrections and select one or more eligible rows;
3. use ``Configure...`` to edit the shared electronic settings and the
   stage-2 phonon/task-farming controls for that row;
4. submit the selected rows locally or on the cluster to run the required
   stage-1 singlepoint calculation;
5. once stage 1 completes, submit the row again to launch the stage-2 phonon
   continuation;
6. review the completed result table once the ``.phonon`` file has been parsed;
7. use ``Launch Analysis`` to open the detailed phonon-analysis window for a
   completed job.

The page is intentionally row-driven. Each eligible relaxed site becomes its
own two-stage job candidate, so you can tune settings per site without
changing other sources in the same project. The ``Site`` column remains
editable, so you can rename the relaxed-site label in place before configuring
or submitting jobs. The source table also shows:

- ``Displacements`` as ``3 x atom count`` for the relaxed defect cell;
- ``Stage`` as ``singlepoint`` or ``phonon``;
- ``Farms`` as the configured stage-2 task-farm count.

The displacement count is a practical planning aid for the phonon stage,
because it approximates the number of finite-displacement calculations that
will be required and therefore helps decide whether a larger farm count is
worth using.

After the stage-1 singlepoint completes, MuFinder freezes the shared
electronic and phonon-method parameters for that row so the stage-2
continuation remains traceable to the completed singlepoint inputs. Only the
stage-2 farm count remains editable in the ``phonon / ready`` gap before a
phonon job is materialized. If you need to change the shared parameters after
stage 1, use ``Reset to Stage 1`` from the workflow page to clear the staged
Quantum Corrections job records and rerun the singlepoint.

Configuration UI
----------------

The row configuration dialog intentionally starts with the controls most users
are likely to need across the two stages:

- ``PHONON_METHOD``;
- ``BASIS_PRECISION`` or ``CUT_OFF_ENERGY`` depending on the chosen parameter
  combination;
- ``ELEC_ENERGY_TOL``;
- ``KPOINT_MP_GRID``;
- ``Stage 2 farms``;
- pseudopotential family metadata for DFPT compatibility checks.

Additional workflow-supported CASTEP parameters remain available through the
``Add Parameter`` dialog, which provides search by keyword name, alias,
description, and category. MuFinder keeps workflow-controlled values such as
the stage-specific ``TASK`` setting locked in the dialog.

When the originating relaxed-site job inputs are available, Quantum Corrections
inherits ``CUT_OFF_ENERGY`` and ``KPOINT_MP_GRID`` from that specific source
relaxation by default. Per-row configuration still takes precedence, so you can
override those values for an individual relaxed site without changing the
underlying source settings.

Implemented Phase 1 scope
-------------------------

The implemented core supports:

- stage-aware Quantum Corrections rows that always run a singlepoint job
  before exposing the phonon continuation;
- CASTEP ``.phonon`` parsing for phonon headers, Gamma q-points, frequencies,
  atom masses, and complex eigenvectors.
- Selection of the three Gamma modes with the largest ``H:mu`` participation.
- Muon ZPE from the selected modes using ``1/2 h c nu`` with frequencies in
  ``cm-1``.
- Covariance and normalized correlation tensors for one-based CASTEP atom
  indices.
- Legacy-style ``w1`` and ``w2`` entanglement witnesses.
- MuFinder ``JobSpec`` expansion for a stage-1 singlepoint job with
  ``TASK : SINGLEPOINT`` and a stage-2 continuation job with ``TASK : PHONON``,
  ``CONTINUATION : DEFAULT``, explicit ``PHONON_METHOD``, and
  ``PHONON_KPOINT_MP_GRID 1 1 1``. **Both** stages write ``SPECIES_MASS`` and
  ``SPECIES_GAMMA`` for ``H:mu`` (see `Muon mass across the two stages`_).
- Stage-2 task farming through ``NUM_FARMS`` when the configured farm count is
  greater than one.
- Project-page submission of selected eligible sites through the shared local
  and cluster CASTEP runners.
- Local and cluster result handling that treats ``.phonon`` as an expected
  output for quantum phonon jobs and prefers ``*_farm001.castep`` when task
  farming is active.
- A completed-result ``Launch Analysis`` action that opens the parsed phonon
  data in an interactive window for the selected job.

Source eligibility
------------------

Quantum Corrections inputs must come from completed MuFinder structural muon
relaxations. A source is eligible only when:

- the ``MuonSite`` status is ``relaxed``;
- the site has a completed relaxation run;
- the source job belongs to ``WorkflowStage.MUON_RELAXATION``;
- the source job completed successfully;
- the run produced a parsed relaxed structure;
- that structure has ``origin == "muon_relaxation"`` and muon-relaxation
  provenance metadata;
- the relaxed structure contains exactly the muon species used for analysis,
  currently ``H:mu`` by default.

UEP candidates, generated sites, failed relaxations, pristine relaxations, and
manual sites without structural-relaxation provenance are rejected because their
ionic coordinates have not been validated for a phonon calculation.

CASTEP phonon methods
---------------------

MuFinder always configures Quantum Corrections phonons at Gamma only via
``PHONON_KPOINT_MP_GRID 1 1 1``. CASTEP offers two useful method families:

Finite displacement
  Recommended default. It reuses finite-displacement forces and is the broadest
  compatibility choice for MuFinder, including systems where linear response is
  unavailable.

DFPT / linear response
  Advanced option. CASTEP documentation states that DFPT phonons are not
  implemented for ultrasoft pseudopotentials and are also restricted for several
  advanced methods including Hubbard U, exact exchange/hybrids, spin-orbit
  coupling, and some dispersion combinations. MuFinder therefore blocks DFPT
  when it knows ultrasoft pseudopotentials are in use and warns when the
  pseudopotential family is unknown.

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

Task farming
------------

Quantum Corrections phonons are submitted as continuations of the completed
singlepoint stage. When the row's stage-2 farm count is greater than one,
MuFinder writes ``NUM_FARMS`` into the phonon ``.param`` file and keeps the
singlepoint and phonon stages in the same seed directory so CASTEP can reuse
the continuation files directly.

For cluster runs, MuFinder reuses the completed stage-1 remote directory only
when the phonon stage is submitted to the same cluster. In that continuation
case MuFinder uploads refreshed ``.cell``, ``.param``, and ``run.sh`` files
without asking to delete the remote directory first, preserving the singlepoint
outputs and checkpoints already present there.

Outputs and interpretation
--------------------------

Completed quantum phonon results persist:

- the source relaxed site and source relaxation provenance;
- the CASTEP ``.phonon`` file path;
- whether the parsed output is Gamma-only;
- Gamma-point phonon eigenvalues for each mode in ``cm-1``;
- selected mode indices, frequencies, muon participation, and per-mode ZPE;
- total selected-mode muon ZPE in eV;
- ``w1`` and optional warnings for unstable selected modes.

Rows remain inspectable after submission. Once a job has moved beyond the ready
state, the parameter dialog becomes read-only so the stored job inputs remain
traceable to the actual calculation that was submitted.

Phonon analysis window
----------------------

The completed-result ``Launch Analysis`` action opens a dedicated analysis
window for that phonon job. The window reparses the stored CASTEP ``.phonon``
file, so the detailed analysis remains tied to the archived calculation output
rather than to transient table state.

The ``Phonon Modes`` tab lists every Gamma-point phonon eigenmode for the
muonated structure and embeds a synchronized 3D structure viewer. Selecting a
mode updates the viewer immediately. The amplitude control is defined as the
largest displayed atomic displacement in angstrom, so a value of ``0.20``
means the longest mode arrow reaches ``0.20 Å`` in the static preview. When
animation is enabled, the structure oscillates sinusoidally around the
equilibrium geometry while the full-amplitude mode vectors remain available as
fixed reference arrows. Playback controls allow the user to start or pause the
animation, reset the phase to equilibrium, adjust the cycle time, and hide or
show the vector overlays.

The ``Adiabatic Solver`` tab exposes the current Gomilsek-style single-particle
adiabatic solver for the three selected muon-dominated Gamma modes. MuFinder
builds a harmonic one-dimensional effective potential for each selected mode,
solves the finite-difference ground state, and plots the potential, ground-state
energy marker, and probability-density overlay. The plot can be switched between
linear energy and square-root energy scaling to mimic the inspection style used
for shallow effective potentials.

The ``Muon Coupling`` tab calculates the legacy phonon covariance and normalized
correlation tensors for any selected pair of CASTEP atom indices. It also reports
the ``w2`` witness for the same pair, complementing the completed-results
table's project-level ``w1`` summary.

The ZPE value is a harmonic, Gamma-point estimate based on the selected
muon-dominated modes. It should be interpreted as a local-defect approximation,
not as a full anharmonic many-body quantum treatment.

In practice, this means Quantum Corrections is best used as a comparative tool
between already-relaxed candidate sites or charge states, rather than as a
standalone predictor of absolute quantum free energies.

Gomilsek single-particle approximations
---------------------------------------

The single-particle analysis foundation is available for Gomilsek-style
effective potentials:

- use the three selected muon-dominated phonon directions as default 1D scan
  axes;
- support weakly-bound adiabatic scans where the muon is displaced and host
  nuclei remain fixed, with ``m_eff = m_mu``;
- support imported strongly-bound line-scan data where host nuclei follow the
  muon displacement and direction-dependent effective masses are calculated
  from finite differences of the nuclear motion;
- solve the 1D finite-difference Schrodinger equation on each effective
  potential to obtain ground-state ZPE, wavefunction, probability density,
  width, and harmonic reference diagnostics;
- plot effective potentials with a square-root-linear energy option, harmonic
  reference, wavefunction/probability overlay, ZPE markers, and inflection
  points.

PIMD remains out of scope for MuFinder.

References
----------

- CASTEP phonon documentation:
  https://castep-docs.github.io/castep-docs/documentation/Phonons/Castep_Phonons/Running-phonon-calculations/
- Gomilsek et al., ``Many-body quantum muon effects and quadrupolar coupling
  in solids``.
