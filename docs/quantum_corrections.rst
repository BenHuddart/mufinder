Quantum Corrections
===================

Quantum Corrections estimates muon zero-point and phonon-derived correlation
metrics from CASTEP Gamma-point phonon calculations. The workflow is designed
for relaxed MuFinder muon sites, not for unrelaxed candidate generation output.

Implemented Phase 1 scope
-------------------------

The implemented core supports:

- CASTEP ``.phonon`` parsing for phonon headers, Gamma q-points, frequencies,
  atom masses, and complex eigenvectors.
- Selection of the three Gamma modes with the largest ``H:mu`` participation.
- Muon ZPE from the selected modes using ``1/2 h c nu`` with frequencies in
  ``cm-1``.
- Covariance and normalized correlation tensors for one-based CASTEP atom
  indices.
- Legacy-style ``w1`` and ``w2`` entanglement witnesses.
- MuFinder ``JobSpec`` expansion for CASTEP phonon jobs with
  ``TASK : PHONON``, explicit ``PHONON_METHOD``, ``PHONON_KPOINT_MP_GRID 1 1 1``,
  ``SPECIES_MASS`` for ``H:mu``, and ``SPECIES_GAMMA`` for ``H:mu``.
- Local and cluster result handling that treats ``.phonon`` as an expected
  output for quantum phonon jobs.

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

Outputs and interpretation
--------------------------

Completed quantum phonon results persist:

- the CASTEP ``.phonon`` file path;
- whether the parsed output is Gamma-only;
- selected mode indices, frequencies, muon participation, and per-mode ZPE;
- total selected-mode muon ZPE in eV;
- ``w1`` and optional warnings for unstable selected modes.

The ZPE value is a harmonic, Gamma-point estimate based on the selected
muon-dominated modes. It should be interpreted as a local-defect approximation,
not as a full anharmonic many-body quantum treatment.

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
