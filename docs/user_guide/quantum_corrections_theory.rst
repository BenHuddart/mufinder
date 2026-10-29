Quantum Corrections: Theory and Worked Example
==============================================

This page gives the physics behind the :doc:`/user_guide/quantum_corrections`
page: what the muon zero-point energy, width and entanglement witnesses mean,
which approximation each number belongs to, and how to decide whether the
numbers MuFinder reports are enough. It ends with a worked example on solid
nitrogen whose harmonic numbers are checked by the test suite. You do not need
it to run a calculation; the page guide lists every control.

The treatment follows Gomilšek *et al.* (see `References`_).

.. _quantum-corrections-why:

Why quantum corrections
-----------------------

A DFT relaxation treats the muon as a classical point charge sitting at the
bottom of its potential well. The muon is about 207 times heavier than an
electron but about nine times lighter than a proton, so it is a quantum
particle on the scale of the well: even in its ground state it vibrates. That
zero-point motion has two direct consequences:

- **Zero-point energy (ZPE).** The ground state lies above the classical
  minimum, typically by a few tenths of an eV and up to about 0.7 eV. Two sites
  whose classical energies differ by less than their ZPE difference can swap
  order, and a barrier between sites is lowered by the ZPE of the well.
- **Spread.** The muon's probability density has a width of a few tenths of an
  Å, a sizeable fraction of a muon–host bond. Anything measured at the muon
  (hyperfine and dipolar fields, electric field gradients at nearby nuclei) is
  averaged over that spread.

A third question decides which approximation is valid: does the muon move on
its own, with the host nuclei effectively frozen, or do the nearby nuclei move
with it? When they move with it, the muon and nuclear motion are *entangled*,
the zero-point motion is a many-body problem, and single-particle pictures of
the muon fail.

Quantum Corrections answers these three questions for each distinct relaxed
site: how much ZPE the site carries, how far the muon spreads, and whether its
motion is entangled with the host.

.. _quantum-corrections-harmonic:

The harmonic many-body picture
------------------------------

Expanding the total energy to second order about the relaxed geometry turns the
muon and every nucleus of the supercell into a set of coupled harmonic
oscillators. Their normal modes come from a Γ-point phonon calculation of the
relaxed muonated supercell. A muon is a point defect, so it has no
dispersion: the supercell's Γ point already contains the muon's local modes,
and Γ is the only q-point needed. The mass-weighted dynamical matrix is

.. math::

   D_{ia,jb} = \frac{\Phi_{ia,jb}}{\sqrt{m_i m_j}},

where :math:`\Phi_{ia,jb}` is the force-constant matrix (atoms :math:`i, j`,
Cartesian directions :math:`a, b`) and :math:`m_i` the atomic masses. Its
eigenvalues are the squared mode frequencies :math:`\omega_n^2` and its
orthonormal eigenvectors :math:`\mathbf{e}_n` are the vectors CASTEP writes to
the ``.phonon`` file. MuFinder gives the muon its mass through ``SPECIES_MASS``,
so CASTEP builds :math:`D` with the muon mass directly. Gomilšek *et al.*
instead reweight a dynamical matrix by the muon mass and rediagonalise it; the
force constants do not depend on the mass, so the two routes give the same
modes.

.. _quantum-corrections-zpe-width:

Zero-point energy and width
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the usual picture, the muon's zero-point motion is described by three modes
:math:`j` that move essentially only the muon. Each contributes

.. math::

   \mathrm{ZPE}_j = \tfrac{1}{2}\hbar\omega_j,
   \qquad
   \Delta x_j = \sqrt{\frac{\hbar}{2 m_\mu \omega_j}},

where :math:`\Delta x_j` is the root-mean-square width of the muon's
ground-state density along the mode direction. The total muon ZPE is
:math:`\sum_j \mathrm{ZPE}_j` over the three modes. A useful number for checking
widths by hand is :math:`\hbar^2 / (2 m_\mu) \approx 0.01842` eV Å², so that
:math:`\Delta x_j = \sqrt{0.01842\ \text{eV Å}^2 / (2\,\mathrm{ZPE}_j)}`.

Since :math:`\omega \propto 1/\sqrt{m}`, a muon mode lies about three times
higher in frequency than the same mode with a proton, and the ZPE rises by the
same factor. This is why the muon mass has to reach the phonon calculation (see
:ref:`quantum-corrections-muon-mass` in the page guide).

.. _quantum-corrections-mode-choice:

Which three modes
~~~~~~~~~~~~~~~~~

The *muon participation* of mode :math:`n` is the squared norm of the muon's
part of its eigenvector,

.. math::

   p_n = |\mathbf{e}_{\mu,n}|^2,
   \qquad
   \sum_n p_n = 3,

the sum running over every mode because the eigenvectors are orthonormal and
the muon has three degrees of freedom. MuFinder selects the three modes with
the largest :math:`p_n`. Gomilšek *et al.* take the three highest-frequency
modes instead. The two rules pick the same modes when the muon modes are the
stiffest modes of the supercell, which is the usual case: the muon is about
nine times lighter than hydrogen, so its modes lie above almost every host
mode. A host with stiff bonds of its own (an N≡N, C–H or O–H stretch, for
example) can place host modes among the highest frequencies; participation then
still finds the modes that move the muon, which is why MuFinder uses it.

.. _quantum-corrections-w1:

Mode-mixing witness w1
~~~~~~~~~~~~~~~~~~~~~~

Projecting the three selected modes onto pure muon motion and summing their
squared norms gives the summed participation :math:`P`. The witness is

.. math::

   w_1 = P - 3,
   \qquad
   P = \sum_{j=1}^{3} p_j,
   \qquad
   -3 \le w_1 \le 0.

It is zero when the three muon modes do not mix with host modes, so that the
muon's motion is fully contained in them: the weakly-bound, unentangled limit of
:ref:`quantum-corrections-adiabatic`. It becomes more negative as the muon modes
hybridise with host modes, which in the harmonic model means the muon's
zero-point motion is entangled with the nuclei. A negative :math:`w_1` is proof
of mode mixing; it does not say which host modes take part (the participation
column of the analysis window does).

.. _quantum-corrections-w2:

Positional witness w2
~~~~~~~~~~~~~~~~~~~~~

The second witness asks directly whether the muon's position is correlated with
that of a nucleus :math:`i`. With centred displacements
:math:`\delta\mathbf{x}'` of the muon and :math:`\delta\mathbf{r}'_{N_i}` of the
nucleus, the multivariate Pearson correlation and its witness are

.. math::

   c_i = \frac{\langle \delta\mathbf{x}' \cdot \delta\mathbf{r}'_{N_i} \rangle}
              {\sqrt{\langle |\delta\mathbf{x}'|^2 \rangle\,
                     \langle |\delta\mathbf{r}'_{N_i}|^2 \rangle}},
   \qquad
   w_{2,i} = -|c_i| \in [-1, 0].

For a separable muon–nuclear state the numerator vanishes and
:math:`w_{2,i} = 0`, so a negative value implies entanglement; when the nucleus
displacement is fixed by the muon's (the strongly-bound limit),
:math:`|c_i| = 1`. Gomilšek *et al.* evaluate the averages by path-integral
molecular dynamics. MuFinder's ``Muon Coupling`` tab gives the harmonic
estimate: at zero temperature the displacement covariance of atoms :math:`a`
and :math:`b` summed over every stable Γ-point mode (:math:`\omega_n > 0`) is

.. math::

   C_{ab} = \sum_n \frac{\hbar}{2\omega_n \sqrt{m_a m_b}}\,
            \mathbf{e}_{a,n}\, \mathbf{e}_{b,n}^{\dagger},

a 3 × 3 tensor, and :math:`c_i = \mathrm{Tr}\, C_{\mu i} /
\sqrt{\mathrm{Tr}\, C_{\mu\mu}\, \mathrm{Tr}\, C_{ii}}`. Unlike :math:`w_1`, this
uses every stable mode, so it is the many-body harmonic answer rather than a
statement about three modes. Modes with imaginary frequencies have no harmonic
ground state and are left out. The soft modes of the host that barely move the
muon still add to each nucleus's own variance, so the answer depends on which
modes the sum keeps: in the worked example below, leaving them out as well, as
a cutoff on the muon participation does, lowers :math:`|c_i|` from about 30 % to
about 26 %.

.. _quantum-corrections-adiabatic:

Single-particle adiabatic limits
--------------------------------

The harmonic picture fails when the muon's well is not parabolic over the range
the muon explores. Single-particle *adiabatic* approximations handle an
anharmonic well, at the price of fixing the degree of entanglement in advance.
Both build an effective potential :math:`V_\mathrm{eff}(\delta\mathbf{x})` from
DFT total energies with the muon displaced by :math:`\delta\mathbf{x}` from its
site, and differ in what the nuclei do:

Weakly bound
   The nuclei stay at the positions of the undisplaced muon site. The
   wavefunction is separable,
   :math:`\psi(\delta\mathbf{x}, \delta\mathbf{r}_N) =
   \psi_\mu(\delta\mathbf{x})\, \psi_N(\delta\mathbf{r}_N)` (no entanglement),
   and the effective mass is the muon mass, :math:`m_\mathrm{eff} = m_\mu`.
   The harmonic three-mode numbers of
   :ref:`quantum-corrections-zpe-width` are this limit with a parabolic
   potential.

Strongly bound
   At each :math:`\delta\mathbf{x}` the nuclei relax to their new lowest-energy
   positions with the centre of mass fixed, so measuring the muon would fix
   every nucleus (maximal entanglement). With linear responses
   :math:`\delta\tilde{\mathbf{r}}_{N_i} = A_i\, \delta\mathbf{x}`, the nuclei
   that follow the muon add to its inertia:

   .. math::

      m_\mathrm{eff} = m_\mu \mathbb{1} + \sum_i m_i A_i^{\mathsf{T}} A_i .

   The strongly-bound potential is never deeper than the weakly-bound one, but
   the heavier effective mass works the other way, so the spread can grow or
   shrink.

In either limit the muon's ground state solves

.. math::

   \left[-\frac{\hbar^2}{2 m_\mathrm{eff}} \nabla^2 + V_\mathrm{eff}\right]
   \psi_\mu = \mathrm{ZPE}\; \psi_\mu .

**Which limit applies.** In a toy model where the muon is bound to nearby
nuclei with stiffness :math:`k_\mu` and those nuclei are bound to the lattice
with stiffness :math:`k_n`, the weakly-bound limit holds for
:math:`k_\mu \ll k_n` and the strongly-bound limit for :math:`k_\mu \gg k_n`.
The ratio follows from the two potentials at the same displacement,

.. math::

   \frac{k_\mu}{k_n} \approx \frac{V_\mathrm{weak}}{V_\mathrm{strong}} - 1 .

A ratio far from both extremes leaves a partially entangled regime that
neither limit describes.

**How anharmonic the well is.** Comparing the weakly-bound potential at two
displacements,

.. math::

   a = \frac{V_\mathrm{weak}(2\delta x)}{4\, V_\mathrm{weak}(\delta x)},

gives :math:`a = 1` for a harmonic well; :math:`a > 1` means the well stiffens
faster than a parabola, so the harmonic width is too large and the harmonic ZPE
too small along that direction.

MuFinder does not generate the DFT line scans these limits need, so
anharmonic and strongly-bound potentials, and the ratios :math:`k_\mu/k_n`
and :math:`a`, are not available from the application.

.. _quantum-corrections-choosing:

Choosing an approximation
-------------------------

Start at the first rung and move down while its conditions fail:

1. **The harmonic numbers are enough** when :math:`w_1` is close to zero, no
   host mode carries an appreciable share of the muon's participation (the
   selected modes stand apart from the host modes in the ``Phonon Modes`` tab),
   and the well is close to harmonic (:math:`a \approx 1`, which needs a line
   scan made outside MuFinder). The muon then moves on its own, and the
   reported ZPE and widths are the weakly-bound answer.
2. **Single-particle pictures fail** when :math:`w_1` is clearly below zero or a
   muon mode is hybridised with host modes. The muon's motion is entangled with
   the nuclei, so neither adiabatic limit is right; if the well is still close
   to harmonic, the many-body harmonic result (every mode, as in
   :math:`w_2`) remains valid.
3. **Path-integral molecular dynamics is needed** when strong anharmonicity
   comes on top of entanglement. PIMD treats both exactly and is outside
   MuFinder. The harmonic ZPE is then a starting estimate, not an answer.

Whatever the rung, ZPEs computed with the same settings are most reliable as
comparisons between sites or charge states.

.. _quantum-corrections-nitrogen:

Worked example: solid nitrogen
------------------------------

Gomilšek *et al.* study the positive muon in α-N₂, where every rung of the
ladder is visible. :doc:`/user_guide/examples/alpha_n2` follows the example
through MuFinder step by step; the figures below show where each harmonic
number appears. They come from the paper's own phonon run, the relaxed
65-atom supercell and its Γ-point ``.phonon`` file from the published data
set, analysed by MuFinder. The harmonic numbers are pinned by
``tests/docs/test_docs_alpha_n2.py``; the other rows of the table below are
the paper's results and are not computed by MuFinder.

Setup
~~~~~

- **Host.** α-N₂ in space group Pa-3 (No. 205), a = 5.660 Å (Venables and
  English 1974), in a 2 × 2 × 2 supercell of 64 nitrogens. The calculation
  holds the cell at a = 5.6635 Å.
- **Calculation.** CASTEP with the PBE functional, the Tkatchenko–Scheffler
  dispersion correction, ultrasoft pseudopotentials, a 1200 eV cutoff, a
  2 × 2 × 2 Monkhorst–Pack grid, a charge of +1 per muonated supercell, an SCF
  tolerance of 0.1 neV/atom and a force tolerance of 5 meV/Å for the
  relaxations.
- **Sites.** More than 20 randomly seeded muons were relaxed and clustered as
  connected components with a symmetry-aware 0.7 Å distance tolerance, the
  ``Connected Components`` clustering of ``Muon Sites > Analysis``. A single
  cluster remains: the muon sits midway between two N₂ molecules, near
  (0, ¼, ¼), forming a near-linear [N₂–μ–N₂]⁺ complex along [011] with the
  nearest nitrogens 1.27 and 1.29 Å away.

In MuFinder this is one pinned site on the Quantum Corrections page. The
ultrasoft pseudopotentials call for the default ``FINITEDISPLACEMENT`` phonon
method, and the tighter SCF tolerance is set as ``SCF Energy Tolerance``
(``1e-10`` eV per atom) in the row's configuration dialog.

.. figure:: ../_generated/screenshots/n2_quantum_configuration_dialog.webp
   :alt: The Quantum Corrections configuration dialog for the alpha-N2 row:
         Ultrasoft pseudopotentials, task PHONON, cutoff 1200 eV, SCF energy
         tolerance 1e-10 eV, a 2 x 2 x 2 k-point grid, PBE, and the
         FINITEDISPLACEMENT phonon method.

   The α-N₂ row configured as Gomilšek *et al.* ran it, apart from the
   dispersion correction and the pressure on the cell, which MuFinder's
   starter keyword catalog does not include.

Where the numbers appear
~~~~~~~~~~~~~~~~~~~~~~~~

The three muon modes are one longitudinal mode L along the complex and two
transverse modes T1 and T2 across it.

.. figure:: ../_generated/screenshots/n2_quantum_corrections_page.webp
   :alt: The Quantum Corrections page with the pinned alpha-N2 site completed,
         and a result row: ZPE 0.576 eV, muon modes 2749.6, 3246.4 and
         3301.1 cm-1, widths 0.23, 0.21 and 0.21 Angstrom, and w1 -0.1377.

   ``Completed Results`` for the α-N₂ site: the ZPE, the three muon modes,
   their widths and ``w1``.

``Completed Results``
   ``ZPE`` reads 0.576 eV, the sum of 0.170 (L), 0.201 (T1) and 0.205 eV (T2),
   the paper's 0.17, 0.20 and 0.20 eV. ``Muon modes (cm⁻¹)`` lists the modes
   at :math:`\hbar\omega_j = 2\,\mathrm{ZPE}_j`: 2749.6 cm⁻¹ for L and
   3246.4 and 3301.1 cm⁻¹ for the T pair (1 eV is 8065.5 cm⁻¹). ``w1`` is
   :math:`P - 3 = -0.14`: the three modes carry 2.86 of the muon's total
   participation of 3. ``Width (Å)`` lists the harmonic zero-point widths
   :math:`\Delta x = \sqrt{0.01842 / 0.341} \approx 0.232` Å for L and about
   0.213 Å for T1 and T2, a sizeable fraction of the 1.27 Å muon–nitrogen
   distance.

.. figure:: ../_generated/screenshots/n2_phonon_modes.webp
   :alt: The Phonon Modes tab scrolled to the highest modes: N2 stretches
         from 2358.4 to 2363.5 cm-1, mode 192 at 2428.7 cm-1, and the muon
         modes 193 at 2749.6 (0.878), 194 at 3246.4 (0.992) and 195 at
         3301.1 cm-1 (0.992).

   The highest Γ-point modes: the three muon modes, highlighted, above the
   N₂ stretches.

``Phonon Modes``
   The three highlighted modes are the muon modes, and the stiffest modes of
   the supercell, so the participation and highest-frequency rules agree. The
   L mode is hybridised with the intramolecular stretch of the complex: L
   keeps a participation of 0.878, and a stretch of the complex molecules,
   mode 161 at 2345.2 cm⁻¹, takes 0.118, the paper's "near 2350 cm⁻¹, about
   12 %". T1 and T2 stay nearly pure at 0.992. Eighteen librations come out
   with imaginary frequencies; they carry under 0.1 % of the muon's motion and
   are left out of every zero-point sum.

.. figure:: ../_generated/screenshots/n2_muon_coupling.webp
   :alt: The Muon Coupling tab with Atom A 65: H:mu and Atom B 57: N, w2 =
         -0.310498, and the covariance and correlation tensors.

   The muon and its nearest nitrogen.

``Muon Coupling``
   With ``Atom A`` the muon and ``Atom B`` one of the two nearest nitrogens,
   ``w2`` is −0.31 (and −0.30 for the other): a harmonic correlation
   :math:`|c| \approx 30\,\%`. The paper's harmonic estimate, about 26 %,
   keeps only the 20 modes with a muon participation of at least 10⁻⁴ (see
   :ref:`quantum-corrections-w2`); with that cutoff MuFinder's covariance
   gives 0.26 and 0.24 for the same two nitrogens. The paper's PIMD
   correlation decays with distance from the muon, so expect smaller values
   for nitrogens further away. The ``?`` button gives the equations:

   .. figure:: ../_generated/screenshots/n2_coupling_help.webp
      :alt: The Muon coupling tensors help dialog with the covariance,
            correlation and w2 equations.

      The help of the ``Muon Coupling`` tab.

The paper's results
~~~~~~~~~~~~~~~~~~~

Directional values are given along L, T1 and T2, in that order (Table 1 of
Gomilšek *et al.*):

.. list-table::
   :header-rows: 1

   * - Approximation
     - ZPE (eV)
     - ZPE\ :sub:`j` (eV)
     - Δx\ :sub:`j` (Å)
     - m\ :sub:`eff,j` / m\ :sub:`μ`
   * - Harmonic, weakly bound (MuFinder)
     - 0.58
     - 0.17, 0.20, 0.20
     - 0.23, 0.21, 0.21
     - 1
   * - Adiabatic, weakly bound (anharmonic)
     - 0.69
     - 0.29, 0.20, 0.20
     - 0.16, 0.21, 0.21
     - 1
   * - Adiabatic, strongly bound
     - 0.008(4)
     - 0.0021, 0.0031, 0.0028
     - 0.074(4), 0.093(3), 0.079(4)
     - 810(160), 350(40), 520(80)
   * - PIMD
     - 0.43(11)
     - 0.16(11), 0.13(1), 0.14(1) \*
     - 0.24(8), 0.27(1), 0.25(1)
     - 1 \*

\* For PIMD, the directional ZPEs are the harmonic ZPEs that give the same
widths with :math:`m_\mathrm{eff} = m_\mu`.

The other diagnostics are :math:`w_1 = -0.14`, a stiffness ratio
:math:`k_\mu/k_n \approx 7, 12, 9` along L, T1 and T2, an anharmonicity
:math:`a \approx 1.8` along L at :math:`2\delta x = 0.24` Å, and a PIMD
correlation with the two nearest nitrogens of :math:`|c| = 30(4)\,\%`, against
the paper's harmonic estimate of about 26 %.

Conclusion
~~~~~~~~~~

The example reaches the last rung of the ladder:

- :math:`w_1 = -0.14` and the hybridised L mode put it past rung 1: the muon is
  entangled with the nitrogens, and the stiffness ratio rules out the
  weakly-bound limit.
- The strongly-bound limit is not reliable either. The complex's positive
  charge polarises the surrounding N₂ molecules into an electric-dipole polaron,
  so many nuclei follow the displaced muon and its effective mass grows by
  hundreds of muon masses. A light particle also tends to decouple partly from
  heavier ones, so the muon does not drag the nuclei fully either: the PIMD
  widths, larger than in both limits, confirm a partially entangled state.
- The L well is strongly anharmonic (:math:`a \approx 1.8`, and the anharmonic
  L width is more than 30 % below the harmonic one), so the harmonic many-body
  result fails too. Only PIMD is reliable here.

Measured against PIMD, the harmonic ZPE of 0.58 eV overestimates the
0.43(11) eV result. The harmonic correlation depends on which modes the sum
keeps: about 30 % over every stable mode, as MuFinder reports, and about 26 %
over the modes that move the muon, against 30(4) % from PIMD. MuFinder's
harmonic numbers flagged correctly that the site needs more than a harmonic
treatment, and gave a starting estimate of the size of the effect.

References
----------

- M. Gomilšek, F. L. Pratt, S. P. Cottrell, S. J. Clark, and T. Lancaster,
  *Many-body quantum muon effects and quadrupolar coupling in solids*,
  Commun. Phys. **6**, 142 (2023), DOI: ``10.1038/s42005-023-01260-7``.
- M. Gomilšek *et al.*, *Many-Body Quantum Muon Effects and Quadrupolar
  Coupling in Solids: Data and Code*, figshare (2023), DOI:
  ``10.6084/m9.figshare.23203037.v1``.
- J. A. Venables and C. A. English, *Electron diffraction and the structure of
  α-N₂*, Acta Cryst. B **30**, 929 (1974), DOI: ``10.1107/S0567740874004067``.
