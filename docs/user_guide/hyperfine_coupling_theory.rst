Hyperfine Coupling: theory
==========================

This page explains the quantities on the :doc:`/user_guide/hyperfine_coupling`
page and checks them on two cases with known answers: muonium in vacuum and a
muon in iron.

The hyperfine Hamiltonian
-------------------------

A muon spin :math:`\mathbf{I}` and the electron spin :math:`\mathbf{S}` near it
interact through

.. math::

   \mathcal{H} = h\,\mathbf{I}\cdot\mathsf{A}\cdot\mathbf{S},

where the hyperfine tensor :math:`\mathsf{A}`, in frequency units, splits into
an isotropic and a traceless part,
:math:`\mathsf{A} = A_\mathrm{iso}\,\mathsf{1} + \mathsf{D}`.
The isotropic part is the Fermi contact interaction with the spin density at
the muon. The traceless part :math:`\mathsf{D}` is the dipolar interaction with
the spin density around it; MuFinder reports its three principal values.

In a magnet the electron spins are polarised and the muon sees their contact
interaction as a static field, the contact field

.. math::

   B_c = \frac{2\mu_0}{3}\,\mu_B\,\rho_s(\mathbf{r}_\mu),
   \qquad \rho_s = \rho_\uparrow - \rho_\downarrow ,

with :math:`\rho_s` the spin density at the muon (Onuorah *et al.* 2018).
The isotropic coupling and the contact field carry the same information,

.. math::

   A_\mathrm{iso} = g_e\,\frac{\gamma_\mu}{2\pi}\,B_c ,

so 1 T of contact field is :math:`2.0023 \times 135.54 = 271.4` MHz.

Reconstructed and pseudo spin densities
---------------------------------------

A plane-wave calculation with pseudopotentials describes smooth pseudo
wavefunctions, which are wrong inside each atom's core region. The muon sits
inside the core region of its own hydrogen pseudopotential, so the pseudo spin
density at the muon is not the true one. The projector-augmented-wave (PAW)
method reconstructs the all-electron density there from the pseudo
wavefunctions (Bonhomme *et al.* 2012), and CASTEP's hyperfine calculation does
this reconstruction. MuFinder reports both: the reconstructed value from the
tensor and the pseudo value from the formatted density, evaluated at the muon
by Fourier interpolation of the plane-wave density.

Onuorah *et al.* (2018) found the reconstructed contact field 28–40 % larger
than the pseudo one in Fe, Co and Ni. The pseudo value happened to sit closer
to experiment, which they traced to a cancellation between the missing
reconstruction and the missing zero-point motion of the muon.

CASTEP's sign
-------------

CASTEP accumulates the hyperfine tensor with the first spin channel weighted
by -1, so its tensor has the opposite sign to the convention above. MuFinder
multiplies it by -1. The muonium case below fixes the sign: its single
unpaired electron puts majority spin density on the muon, and its coupling is
positive.

Muonium in vacuum
-----------------

A muonium atom in a 10 Å box, computed with PBE at 700 eV and the muon's
gyromagnetic ratio in ``SPECIES_GAMMA``:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Quantity
     - MuFinder
     - Reference
   * - CASTEP ``Iso``
     - -4628.4 MHz
     -
   * - :math:`A_\mathrm{iso}`
     - +4628.4 MHz
     - +4463.3 MHz (measured)
   * - :math:`B_c`
     - 17.05 T
     - 16.4 T
   * - :math:`\rho_s` pseudo
     - 0.258 :math:`a_0^{-3}`
     - :math:`1/\pi = 0.318\,a_0^{-3}` (exact)

PBE overestimates the vacuum coupling by about 4 %, a known property of the
functional, and the pseudo spin density is 19 % short of the exact
:math:`|\psi(0)|^2` because the pseudo wavefunction is smooth at the nucleus.

A muon in iron
--------------

bcc Fe in a 2 × 2 × 2 cubic supercell (16 Fe, :math:`a = 2.83` Å), with the
muon at the tetrahedral site without relaxation, PBE at 700 eV, an 8 × 8 × 8
k-point mesh and 0.1 eV smearing. The Fe moments converge to 2.20 μB.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Quantity
     - MuFinder
     - Onuorah *et al.* 2018
     - Experiment
   * - :math:`\rho_s` pseudo
     - -0.0180 :math:`a_0^{-3}`
     - -0.0179 :math:`a_0^{-3}`
     -
   * - :math:`\rho_s` reconstructed
     - -0.0228 :math:`a_0^{-3}`
     - -0.0238 :math:`a_0^{-3}`
     -
   * - :math:`B_c` pseudo
     - -0.94 T
     - -0.94 T
     -
   * - :math:`B_c` reconstructed
     - -1.20 T
     - -1.25 T
     - -1.11 T
   * - Reconstructed / pseudo
     - 1.27
     - 1.33
     -

Onuorah *et al.* used a relaxed 4 × 4 × 4 supercell, so the small differences
are expected. The negative sign says the spin density at the muon is opposite
to the bulk magnetisation. With zero-point motion included Onuorah *et al.*
(2019) find -1.07 T.

The tensor also has a dipolar part with principal values -137.6, +68.8 and
+68.8 MHz, axial along :math:`y` for the site at :math:`(1/2, 1/4, 0)` of the cubic cell. This is the dipolar
field of the spin density inside the supercell, which Magnetic Fields
computes for the whole crystal from the Fe moments instead.

The Magnetic Fields coupling
----------------------------

Magnetic Fields writes the contact field as

.. math::

   B_c = \frac{2\mu_0}{3}\,\mu_B\,A\,\bar{m},
   \qquad
   \bar{m} = \frac{\sum_i m_i / r_i^3}{\sum_i 1/r_i^3},

where the sum runs over the muon's contact shell and :math:`A`, in
:math:`\text{Å}^{-3}` per μB, is its ``Hyperfine coupling``. Hyperfine Coupling
hands over :math:`A = \rho_s / \bar{m}_\mathrm{DFT}`, with the same shell's
Mulliken moments. In the iron case every Fe carries 2.20 μB, so
:math:`A = -0.154\ \text{Å}^{-3} / 2.20 = -0.070\ \text{Å}^{-3}` per μB, or
-0.54 T/μB. The measured -1.11 T corresponds to -0.50 T/μB, the value the
:doc:`/user_guide/examples/fe` example enters by hand.

References
----------

- I. J. Onuorah, P. Bonfà and R. De Renzi, Muon contact hyperfine field in
  metals: A DFT calculation, Phys. Rev. B **97**, 174414 (2018),
  `doi:10.1103/PhysRevB.97.174414 <https://doi.org/10.1103/PhysRevB.97.174414>`_.
- I. J. Onuorah, P. Bonfà, R. De Renzi, L. Monacelli, F. Mauri, M. Calandra and
  I. Errea, Quantum effects in muon spin spectroscopy within the stochastic
  self-consistent harmonic approximation, Phys. Rev. Mater. **3**, 073804 (2019),
  `doi:10.1103/PhysRevMaterials.3.073804 <https://doi.org/10.1103/PhysRevMaterials.3.073804>`_.
- C. Bonhomme, C. Gervais, F. Babonneau, C. Coelho, F. Pourpoint, T. Azaïs,
  S. E. Ashbrook, J. M. Griffin, J. R. Yates, F. Mauri and C. J. Pickard,
  First-principles calculation of NMR parameters using the gauge including
  projector augmented wave method: a chemist's point of view, Chem. Rev.
  **112**, 5733 (2012),
  `doi:10.1021/cr300108a <https://doi.org/10.1021/cr300108a>`_.
