The muon in solid nitrogen
==========================

Solid α-N₂ is the example Gomilšek *et al.* use to show every rung of the ladder
of approximations for the muon's zero-point motion (see
:doc:`/user_guide/quantum_corrections_theory`). The positive muon binds two
nitrogen molecules into an [N₂–μ–N₂]⁺ complex, its zero-point energy is large,
and its motion is entangled with the nitrogens'. This page follows one project
through the :doc:`/user_guide/quantum_corrections` workflow, from the host
structure to the three numbers that decide how far the harmonic answer can be
trusted, and shows where each number appears in the application.

.. note::

   The CASTEP outputs on this page are the paper's own, from its published
   data set (Gomilšek *et al.*, figshare 2023): the relaxed muon supercell and
   its Γ-point phonon run. The data set holds the one relaxed site but not the
   relaxations of the site search, so the project relaxes a single start whose
   output is the paper's relaxed cell; no calculation on this page ran CASTEP.
   Every number below is MuFinder's own analysis of those files, pinned by
   ``tests/docs/test_docs_alpha_n2.py``.

The host
--------

α-N₂ is the low-temperature phase of solid nitrogen, stable below 35.6 K at
ambient pressure. It is a molecular crystal in space group Pa-3 (No. 205): the
molecule centres sit on the fcc 4a sites, and the nitrogens at the 8c sites
(x, x, x), so each N≡N bond points along one of the four ⟨111⟩ directions.
Electron diffraction gives a = 5.660 Å and x = 0.056 (Venables and English
1974). The project uses the cell of the paper's calculation instead, so that
its host matches the relaxed supercell atom for atom: a = 5.6635 Å, half the
supercell edge, and x = 0.0563, the 1.104 Å bond of the molecules away from
the muon (the cell is loaded as described in
:doc:`/user_guide/host_structure`).

.. figure:: ../../_generated/screenshots/n2_host_structure.webp
   :alt: The Host Structure page with alpha-N2 loaded: eight N in the atom
         table at (0.0563, 0.0563, 0.0563) and its images, and the cell viewed
         down a.

   The eight-atom cubic cell: four molecules, each two nitrogens 0.0563 of the
   cell edge either side of a 4a site.

The molecules are bound to each other only weakly. The N≡N stretch of the
solid is within about one percent of the free molecule's, split by the crystal
field by about 1 cm⁻¹, while the librations of the molecules lie at 32, 36 and
60 cm⁻¹ (Scott 1976). That separation of scales matters below: the muon's own
modes lie above the stretch, and far above the librations and translations.

The muon site
-------------

μSR shows two muon states in solid nitrogen: muonium, which forms when the
muon captures an electron from its ionisation track, and a diamagnetic state,
ascribed to an N₂μ⁺ molecular ion (Storchak *et al.* 1999). This example
follows the diamagnetic state, a positive muon with a charge of +1 per cell.

Gomilšek *et al.* relaxed more than twenty random starts in a 2 × 2 × 2
supercell (64 nitrogens and the muon; see :doc:`/user_guide/supercells`) with
the PBE functional, a 1200 eV cutoff and a 2 × 2 × 2 Monkhorst–Pack grid, and
clustered them with a 0.7 Å threshold: every start ended at one kind of site,
midway between two neighbouring molecules, whose centres are 4.0 Å apart. In
the conventional cell that is the (0, ¼, ¼) site and its 23 images. The project
enters that site as one manual start (see :doc:`/user_guide/site_generation`)
and relaxes it as a positive muon in the same supercell, with the paper's
cutoff and grid set by hand.

The paper's runs also used the Tkatchenko–Scheffler dispersion correction
(``SEDC_SCHEME : TS``) and held the cell fixed under an external pressure of
0.5 GPa (``EXTERNAL_PRESSURE``). MuFinder's starter keyword catalog has
neither keyword, so the project's rows do not set them; the relaxed cell the
row reads back keeps them in its echo of the run's settings, which MuFinder
skips when it reads the geometry.

.. figure:: ../../_generated/screenshots/n2_relaxation_workspace.webp
   :alt: The relaxation analysis workspace with one cluster of one relaxed
         positive-muon site at (0.0018, 0.2500, 0.2482), Wyckoff 24d.

   The relaxed site in an analysis workspace: one cluster, at (0, ¼, ¼)
   within 0.002 of the cell edge.

Clustering it (see :doc:`/user_guide/structural_relaxation_analysis`) gives a
single cluster. The relaxed muon sits at (0.0018, 0.2498, 0.2482), a general
position of Pa-3 (Wyckoff 24d, site symmetry 1).

.. figure:: ../../_generated/screenshots/n2_muon_sites_summary.webp
   :alt: The Muon Sites summary with one positive-muon site at
         (0.0018, 0.2498, 0.2482) and Wyckoff 24d, and no muonium analysis
         pinned.

   The site pinned to the summary.

``Analyse Displacements`` shows what the muon does to its neighbours. The two
molecules either side turn to point at it and form a near-linear
[N₂–μ–N₂]⁺ complex along [011], with the nearest nitrogens 1.268 and 1.288 Å
from the muon (Gomilšek *et al.* give 1.29(1) Å). The bonds of the two complex
molecules shorten slightly, to 1.099 Å, against 1.104 Å for the molecules
away from the muon.

.. figure:: ../../_generated/screenshots/n2_displacement_analysis.webp
   :alt: The Displacements window for site 1 against the alpha-N2 host:
         the supercell viewed down a with the muon between two molecules, and
         radial displacements between -0.86 and +0.57 Angstrom at every
         distance from the muon out to 8.4 Angstrom.

   The relaxed supercell against the host. Every molecule has turned, so
   nitrogens move by tenths of an Å at every distance from the muon.

The plot shows more than the complex. The complex's charge polarises the
molecules around it and turns them to point towards it, the electric-dipole
polaron of Gomilšek *et al.*, which reaches about 4 Å from the muon. In a
2 × 2 × 2 supercell no molecule is more than about 8 Å from the muon, so every
one of the 30 others turns, by 16° to 50° from its ⟨111⟩ axis. Its two
nitrogens move by 0.47 Å on average and by up to 0.93 Å, towards the muon or
away from it (radially between −0.86 and +0.57 Å). The supercell is no
larger than the polaron, so the displacements far from the muon describe this
supercell rather than an isolated muon.

The phonon run
--------------

The pinned site becomes the one row of the ``Quantum Corrections`` page. Its
configuration dialog inherits the cutoff, the k-point grid and the functional
from the relaxation. The row is set to the ultrasoft pseudopotentials Gomilšek
*et al.* used, which calls for the default ``FINITEDISPLACEMENT`` phonon method,
and to an ``SCF Energy Tolerance`` of ``1e-10`` eV, tighter than the workflow
default of ``1e-9`` eV.

.. figure:: ../../_generated/screenshots/n2_quantum_configuration_dialog.webp
   :alt: The Quantum Corrections configuration dialog for the alpha-N2 row:
         Ultrasoft pseudopotentials, one farm, task PHONON, cutoff 1200 eV,
         SCF energy tolerance 1e-10 eV, a 2 x 2 x 2 k-point grid, PBE, and the
         FINITEDISPLACEMENT phonon method.

   The row's configuration before submission. ``Displacements required``
   reads 195, three for each of the 65 atoms.

Once the phonon job completes, ``Completed Results`` lists the harmonic
answer.

.. figure:: ../../_generated/screenshots/n2_quantum_corrections_page.webp
   :alt: The Quantum Corrections page with the pinned alpha-N2 site completed,
         and a result row: ZPE 0.576 eV, muon modes 2749.6, 3246.4 and
         3301.1 cm-1, widths 0.23, 0.21 and 0.21 Angstrom, and w1 -0.1377.

   The completed phonon run of the pinned site.

- ``ZPE (eV)`` is 0.576: the sum of 0.170 eV for the longitudinal mode L, along
  the complex, and 0.201 and 0.205 eV for the two transverse modes across it.
  Gomilšek *et al.* give 0.17, 0.20 and 0.20 eV, 0.58 eV in total.
- ``Muon modes (cm⁻¹)`` lists L at 2749.6 cm⁻¹ and the transverse modes at
  3246.4 and 3301.1 cm⁻¹, within the ranges the paper reports (about
  2700–2950 and 3200–3400 cm⁻¹). The two transverse modes are not
  degenerate: in this supercell they lie 55 cm⁻¹ apart.
- ``Width (Å)`` lists the zero-point widths 0.23, 0.21 and 0.21 Å, a sizeable
  fraction of the 1.27 Å muon–nitrogen bond.
- ``w1`` is −0.1377: the three muon modes carry 2.86 of the muon's total
  participation of 3, so the muon's motion is mixed with the nitrogens'.

The phonon modes
----------------

``Launch Analysis`` opens the analysis window on the ``Phonon Modes`` tab, with
the L mode selected. The table ends with the modes that matter here.

.. figure:: ../../_generated/screenshots/n2_phonon_modes.webp
   :alt: The Phonon Modes tab scrolled to the highest modes: N2 stretches
         from 2358.4 to 2363.5 cm-1 with muon participations below 0.001,
         mode 192 at 2428.7 cm-1, the muon modes 193 at 2749.6 (0.878),
         194 at 3246.4 (0.992) and 195 at 3301.1 cm-1 (0.992), with the
         supercell viewed down a and the muon's mode vector.

   The highest modes of the supercell. The three muon modes, their
   participation highlighted, lie above the N₂ stretches.

- Modes 161 to 192 are the stretches of the 32 molecules. The lowest, mode 161
  at 2345.2 cm⁻¹, is a stretch of the two complex molecules and carries 0.118
  of the muon's participation: the L mode is hybridised with the N₂ stretch,
  as Gomilšek *et al.* found (near 2350 cm⁻¹, about 12 %). The 30 molecules
  outside the complex stretch between 2355.7 and 2363.5 cm⁻¹, and mode 192,
  at 2428.7 cm⁻¹, is the complex's other stretch.
- Mode 193 is L, with a participation of 0.878, and modes 194 and 195 are the
  transverse modes, each nearly pure at 0.992. Their direction column gives the
  muon's displacement: along [011] for L, and along [100] and [01̄1] across it.
- Below the stretches lie three modes of the complex itself, at 250, 265 and
  431 cm⁻¹ (modes 158 to 160), which move its four nitrogens most, and the
  librations and translations of the molecules. Eighteen of these soft modes
  come out with imaginary frequencies, the softest at −38.9 cm⁻¹, and the
  run's log warns that CASTEP found no three modes of acoustic character, so
  its acoustic sum rule correction applied to none. The imaginary modes carry
  under 0.1 % of the muon's motion; MuFinder leaves them out of the
  zero-point energy and of the covariances below.

These are the stiffest modes of the supercell, so MuFinder's rule, the three
modes with the largest participation, picks the same modes as the paper's
rule, the three highest frequencies.

Widths
------

``Width (Å)`` in ``Completed Results`` gives the harmonic zero-point width
:math:`\Delta x = \sqrt{\hbar / (2 m_\mu \omega)}` of each muon mode with the
muon mass: 0.23 Å for L and 0.21 Å for each mode across the complex. These are
the harmonic weakly-bound widths of Gomilšek *et al.*, 0.23, 0.21 and 0.21 Å.
Their path-integral widths, 0.24(8), 0.27(1) and 0.25(1) Å, are larger across
the complex: the harmonic well holds the muon too tightly there.

Coupling to the nitrogens
-------------------------

``Muon Coupling`` gives the harmonic correlation of the muon with any atom.
With ``Atom A`` the muon (atom 65) and ``Atom B`` its nearest nitrogen:

.. figure:: ../../_generated/screenshots/n2_muon_coupling.webp
   :alt: The Muon Coupling tab with Atom A 65: H:mu and Atom B 57: N, w2 =
         -0.310498, and the covariance and correlation tensors.

   The muon and its nearest nitrogen: w₂ = −0.31.

``w2`` is −0.31 for the nearest nitrogen (atom 57) and −0.30 for the other
nitrogen of the complex (atom 1): the muon's zero-point motion and theirs are
about 30 % correlated in the harmonic approximation. The paper quotes a
harmonic value of about 26 % and a path-integral one of 30(4) %. Both use the
same correlation (see :ref:`quantum-corrections-w2`); they differ in which
modes the harmonic sum keeps. MuFinder sums every stable mode, while the
paper's analysis, in its published data set, keeps only the 20 modes that
carry a muon participation of at least 10⁻⁴. That leaves out the soft
librations and translations, which widen each nitrogen's motion without
moving the muon, and gives 0.26 and 0.24 for the same two nitrogens. The
``?`` button explains the tensors and the witness, and quotes the comparison.

.. figure:: ../../_generated/screenshots/n2_coupling_help.webp
   :alt: The Muon coupling tensors help dialog with the definitions of the
         covariance, the correlation and the w2 witness, and the alpha-N2
         comparison of 0.30 and 0.31 against the paper's 0.26 and 0.30(4).

   The equations behind ``Muon Coupling``.

What the harmonic numbers say
-----------------------------

The three numbers from the page answer the questions of
:ref:`quantum-corrections-choosing`:

- **w1 = −0.14** and the hybridised L mode say the muon does not move on its
  own: its motion is entangled with the nitrogens', so the first rung, the
  harmonic weakly-bound answer, is not enough.
- **w2 = −0.31** confirms it from the other side: the nearest nitrogens move
  with the muon.
- **The widths** of a few tenths of an Å are a large fraction of the muon's
  bond, so the shape of the well over that range matters.

Gomilšek *et al.* go further than MuFinder can. With DFT line scans they find
the L well strongly anharmonic and the strongly-bound limit unreliable, because
the complex's charge polarises many surrounding molecules. Only path-integral
molecular dynamics describes the muon here: it gives a zero-point energy of
0.43(11) eV against the harmonic 0.58 eV. The harmonic numbers flagged that the
site needs more than a harmonic treatment and gave the size of the effect;
:doc:`/user_guide/quantum_corrections_theory` sets the full comparison out.

References
----------

- M. Gomilšek, F. L. Pratt, S. P. Cottrell, S. J. Clark and T. Lancaster,
  Many-body quantum muon effects and quadrupolar coupling in solids, Commun.
  Phys. **6**, 142 (2023),
  https://doi.org/10.1038/s42005-023-01260-7.
- M. Gomilšek *et al.*, Many-Body Quantum Muon Effects and Quadrupolar
  Coupling in Solids: Data and Code, figshare (2023),
  https://doi.org/10.6084/m9.figshare.23203037.v1. The CASTEP outputs on this
  page are the files of its ``Fig 2c/DFT`` directory (CC BY 4.0).
- T. A. Scott, Solid and liquid nitrogen, Phys. Rep. **27**, 89 (1976),
  https://doi.org/10.1016/0370-1573(76)90032-6.
- V. Storchak, J. H. Brewer, G. D. Morris, D. J. Arseneau and M. Senba,
  Muonium formation via electron transport in solid nitrogen, Phys. Rev. B
  **59**, 10559 (1999),
  https://doi.org/10.1103/PhysRevB.59.10559.
- J. A. Venables and C. A. English, Electron diffraction and the structure of
  α-N₂, Acta Cryst. B **30**, 929 (1974),
  https://doi.org/10.1107/S0567740874004067.
