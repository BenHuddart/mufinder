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

   No calculation on this page ran CASTEP. The screenshots come from a project
   whose CASTEP outputs are synthetic. The relaxed geometry has the muon–nitrogen
   distance of Gomilšek *et al.*, and the phonon run embeds, in the relaxed
   supercell, a spring model of the complex whose constants were tuned to the
   paper's harmonic results. The harmonic numbers below are MuFinder's own
   analysis of that phonon run, pinned by ``tests/docs/test_docs_synthetic.py``
   and, for the five-atom model alone, by
   ``tests/science/test_quantum_phonons.py``. Your own CASTEP calculations will
   give values of their own.

The host
--------

α-N₂ is the low-temperature phase of solid nitrogen, stable below 35.6 K at
ambient pressure. It is a molecular crystal in space group Pa-3 (No. 205) with
a = 5.660 Å (Venables and English 1974): the molecule centres sit on the fcc
4a sites, and the nitrogens at the 8c sites (x, x, x) with x = 0.056, so each
N≡N bond is 1.098 Å long and points along one of the four ⟨111⟩ directions.

.. figure:: ../../_generated/screenshots/n2_host_structure.webp
   :alt: The Host Structure page with alpha-N2 loaded: eight N in the atom
         table at (0.056, 0.056, 0.056) and its images, and the cell viewed
         down a.

   The eight-atom cubic cell: four molecules, each two nitrogens 0.056 of the
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

The project uses a 2 × 2 × 2 supercell (64 nitrogens and the muon), the PBE
functional, a 1200 eV cutoff and a 2 × 2 × 2 Monkhorst–Pack grid, the settings
of Gomilšek *et al.* Six random starts, generated with a fixed seed at least
1 Å from every atom, are relaxed as positive muons. Every one ends at the same
kind of site: midway between two neighbouring molecules, whose centres are
4.00 Å apart. In the conventional cell that is the (0, ¼, ¼) site and its 23
images.

.. figure:: ../../_generated/screenshots/n2_relaxation_workspace.webp
   :alt: The relaxation analysis workspace with one cluster of six relaxed
         positive-muon sites, Wyckoff unavailable, at positions such as
         (0.25, 0.75, 0.5) within 0.0025 eV of each other.

   The six relaxations clustered: one cluster, every member at an image of
   (0, ¼, ¼). The energy spread of 2.5 meV is synthetic.

Clustering the six with the default connected-components method and its
symmetry-aware distance gives a single cluster, as Gomilšek *et al.* found for
more than twenty starts with a 0.7 Å threshold. The site has no special
Wyckoff position: (0, ¼, ¼) is a general position of Pa-3 (24d, site symmetry
1), so the ``Wyckoff`` column shows no label.

.. figure:: ../../_generated/screenshots/n2_muon_sites_summary.webp
   :alt: The Muon Sites summary with one positive-muon site at
         (0.2500, 0.7500, 0.5000) and no Wyckoff label, and no muonium analysis
         pinned.

   The single site pinned to the summary. Its representative relaxation sits
   at (¼, ¾, ½), an image of (0, ¼, ¼).

``Analyse Displacements`` shows what the muon does to its neighbours. The two
molecules either side turn to point at it and form a linear [N₂–μ–N₂]⁺ complex,
with the nearest nitrogen 1.29 Å from the muon on each side (Gomilšek *et al.*
give 1.29(1) Å). For the (0, ¼, ¼) site the complex lies along [011]; for the
representative at (¼, ¾, ½) it lies along [110], the image of [011]. In the
synthetic relaxation the complex's bonds stretch slightly, to 1.115 Å, and the
molecules around the complex move 0.03 Å towards the muon, the beginning of
the polarisation cloud described below.

.. figure:: ../../_generated/screenshots/n2_displacement_analysis.webp
   :alt: The Displacements window for site 1 against the alpha-N2 host:
         the supercell viewed down c with the muon between two molecules, and
         radial displacements of up to 0.8 Angstrom for the four nitrogens of
         the complex and -0.03 Angstrom for the surrounding molecules.

   The complex in the relaxed supercell. The four nitrogens of the complex
   move by up to 0.8 Å as their molecules turn; the surrounding molecules
   move 0.03 Å towards the muon.

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
         and a result row: ZPE 0.580 eV, muon modes 2807.1, 3274.3 and
         3274.3 cm-1, widths 0.23, 0.21 and 0.21 Angstrom, and w1 -0.1401.

   The completed phonon run of the pinned site.

- ``ZPE (eV)`` is 0.580: the sum of 0.174 eV for the longitudinal mode L, along
  the complex, and 0.203 eV for each of the two transverse modes T1 and T2
  across it. Gomilšek *et al.* give 0.17, 0.20 and 0.20 eV, 0.58 eV in total.
- ``Muon modes (cm⁻¹)`` lists L at 2807.1 cm⁻¹ and the degenerate T pair at
  3274.3 cm⁻¹, within the ranges the paper reports (about 2700–2950 and
  3200–3400 cm⁻¹).
- ``Width (Å)`` lists the zero-point widths 0.23, 0.21 and 0.21 Å, a sizeable
  fraction of the 1.29 Å muon–nitrogen bond.
- ``w1`` is −0.14: the three muon modes carry 2.86 of the muon's total
  participation of 3, so the muon's motion is mixed with the nitrogens'.

The phonon modes
----------------

``Launch Analysis`` opens the analysis window on the ``Phonon Modes`` tab, with
the L mode selected. The table ends with the modes that matter here.

.. figure:: ../../_generated/screenshots/n2_phonon_modes.webp
   :alt: The Phonon Modes tab scrolled to the highest modes: the N2 stretch
         band from 2341 to 2350 cm-1 with zero muon participation, mode 191 at
         2616.4 cm-1 with participation 0.130, mode 192 at 2642.0, and the
         muon modes 193 at 2807.1 (0.868), 194 and 195 at 3274.3 (0.996), with
         the supercell viewed down c and the muon's mode vector.

   The highest modes of the supercell. The three muon modes, their
   participation highlighted, lie above the stretch band of the other
   molecules.

- Modes 161 to 190 are the stretches of the 30 molecules outside the complex,
  in a band from 2330 to 2350 cm⁻¹, with no muon participation. Everything
  below them is a libration or a translation, of a molecule or of the complex.
- Modes 191 and 192 are the stretches of the two complex molecules. The first,
  at 2616.4 cm⁻¹, carries 0.130 of the muon's participation along the complex:
  the L mode is hybridised with the N₂ stretch. The spring model places these stretches above the band of the free
  molecules; in the paper's DFT phonons the hybridised stretch lies near
  2350 cm⁻¹ and carries about 12 % of the muon.
- Mode 193 is L, with a participation of 0.868, and modes 194 and 195 are T1
  and T2, each nearly pure at 0.996. Their direction column gives the muon's
  displacement: along [110] for L, across it for T1 and T2.

These are the stiffest modes of the supercell, so MuFinder's rule, the three
modes with the largest participation, picks the same modes as the paper's
rule, the three highest frequencies.

Widths and the mode wells
-------------------------

``Mode wells and widths`` builds the harmonic well of each muon mode with the
muon mass and solves for its ground state.

.. figure:: ../../_generated/screenshots/n2_mode_wells.webp
   :alt: The Mode wells and widths tab with modes 193, 194 and 195: ZPE
         0.174169, 0.203192 and 0.203192 eV and widths 0.230083, 0.213032 and
         0.213032 Angstrom, and the plot of the L well with its ground-state
         level and probability density.

   The L well, its ground-state energy and the muon's probability density.

The solver returns the ZPE and width of each mode: 0.174 eV and 0.230 Å for
L, and 0.203 eV and 0.213 Å for T1 and T2. These are the harmonic weakly-bound
widths of Gomilšek *et al.*, 0.23, 0.21 and 0.21 Å. Their path-integral
widths, 0.24(8), 0.27(1) and 0.25(1) Å, are larger across the complex: the
harmonic well holds the muon too tightly there.

Coupling to the nitrogens
-------------------------

``Muon Coupling`` gives the harmonic correlation of the muon with any atom.
With ``Atom A`` the muon (atom 65) and ``Atom B`` one of its two nearest
nitrogens:

.. figure:: ../../_generated/screenshots/n2_muon_coupling.webp
   :alt: The Muon Coupling tab with Atom A 65: H:mu and Atom B 3: N, w2 =
         -0.259975, and the covariance and correlation tensors.

   The muon and its nearest nitrogen: w₂ = −0.26.

``w2`` is −0.26: the muon's zero-point motion and the nitrogen's are 26 %
correlated in the harmonic approximation, the value Gomilšek *et al.* report.
Their path-integral correlation is 30(4) %, so the harmonic estimate is a
little low. The ``?`` button explains the tensors and the witness, and quotes
the same comparison.

.. figure:: ../../_generated/screenshots/n2_coupling_help.webp
   :alt: The Muon coupling tensors help dialog with the definitions of the
         covariance, the correlation and the w2 witness, and the alpha-N2
         comparison of 0.26 against 0.30(4).

   The equations behind ``Muon Coupling``.

What the harmonic numbers say
-----------------------------

The three numbers from the page answer the questions of
:ref:`quantum-corrections-choosing`:

- **w1 = −0.14** and the hybridised L mode say the muon does not move on its
  own: its motion is entangled with the nitrogens', so the first rung, the
  harmonic weakly-bound answer, is not enough.
- **w2 = −0.26** confirms it from the other side: the nearest nitrogens move
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
- T. A. Scott, Solid and liquid nitrogen, Phys. Rep. **27**, 89 (1976),
  https://doi.org/10.1016/0370-1573(76)90032-6.
- V. Storchak, J. H. Brewer, G. D. Morris, D. J. Arseneau and M. Senba,
  Muonium formation via electron transport in solid nitrogen, Phys. Rev. B
  **59**, 10559 (1999),
  https://doi.org/10.1103/PhysRevB.59.10559.
- J. A. Venables and C. A. English, Electron diffraction and the structure of
  α-N₂, Acta Cryst. B **30**, 929 (1974),
  https://doi.org/10.1107/S0567740874004067.
