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

- ``Quantum Phonon Jobs`` lists eligible relaxed muon sites that can seed a
  phonon calculation.
- ``Completed Results`` lists completed phonon jobs with the parsed ZPE and
  witness summary.

Typical use is:

1. finish the structural muon relaxation workflow and confirm the site is in a
   relaxed state;
2. open Quantum Corrections and select one or more eligible rows;
3. use ``Configure...`` to edit the most relevant phonon controls directly,
   or add extra CASTEP parameters through the searchable advanced list;
4. submit the selected rows locally or on the cluster;
5. review the completed result table once the ``.phonon`` file has been parsed.

The page is intentionally row-driven. Each eligible relaxed site becomes its
own phonon job candidate, so you can tune settings per site without changing
other sources in the same project.

Configuration UI
----------------

The row configuration dialog intentionally starts with the controls most users
are likely to need for a Gamma-point phonon run:

- ``PHONON_METHOD``;
- ``BASIS_PRECISION`` or ``CUT_OFF_ENERGY`` depending on the chosen parameter
  combination;
- ``FINE_CUT_OFF_ENERGY``;
- ``ELEC_ENERGY_TOL``;
- ``KPOINT_MP_GRID``;
- ``PHONON_KPOINT_MP_GRID``;
- pseudopotential family metadata for DFPT compatibility checks.

Additional workflow-supported CASTEP parameters remain available through the
``Add Parameter`` dialog, which provides search by keyword name, alias,
description, and category. MuFinder keeps workflow-controlled values such as
``TASK : PHONON`` locked in the dialog.

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
- Project-page submission of selected eligible sites through the shared local
  and cluster CASTEP runners.
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

- the source relaxed site and source relaxation provenance;
- the CASTEP ``.phonon`` file path;
- whether the parsed output is Gamma-only;
- selected mode indices, frequencies, muon participation, and per-mode ZPE;
- total selected-mode muon ZPE in eV;
- ``w1`` and optional warnings for unstable selected modes.

Rows remain inspectable after submission. Once a job has moved beyond the ready
state, the parameter dialog becomes read-only so the stored job inputs remain
traceable to the actual calculation that was submitted.

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
