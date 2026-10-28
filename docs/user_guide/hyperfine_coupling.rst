Hyperfine Coupling
==================

Hyperfine Coupling computes how strongly each relaxed muon site couples to the
electron spin density around it. For a muon in a magnetic host it gives the
contact field, the part of the local field that a classical dipolar sum cannot
supply. For muonium and other paramagnetic states it gives the isotropic and
anisotropic hyperfine constants. Like Quantum Corrections, it starts from muon
sites that MuFinder has already relaxed. The physics behind each number, with
two worked cases, is on :doc:`/user_guide/hyperfine_coupling_theory`.

Scientific intent
-----------------

Each site runs as one spin-polarised CASTEP ``TASK : MAGRES`` job with
``MAGRES_TASK : HYPERFINE``. CASTEP reconstructs the all-electron spin density
at the muon with its projector-augmented-wave (PAW) projectors and reports the
muon's hyperfine tensor. MuFinder reads that tensor and reports:

- the isotropic (Fermi contact) coupling :math:`A_\mathrm{iso}` in MHz, and the
  contact field :math:`B_c` in tesla that it corresponds to;
- the principal values of the anisotropic (dipolar) part of the tensor, in MHz;
- the contact field from the *pseudo* spin density at the muon, read from the
  same job's formatted density without the PAW reconstruction, as a cross-check.

Onuorah, Bonfà and De Renzi (2018) compared these two spin densities for muons
in Fe, Co, Ni, MnSi and MnGe. The reconstructed one is the better-founded
number; the pseudo one is 20–30 % smaller. A ratio far outside that range is a
reason to check the job's convergence before using the result.

Before you start
----------------

Hyperfine Coupling needs:

- **Relaxed muon sites.** Run the structural relaxations under
  ``Muon Sites > Relaxations`` and let them complete. The page lists the same
  eligible sites as Quantum Corrections: by default the distinct sites pinned
  to the ``Muon Sites`` summary, or every eligible relaxed site with
  ``Show every relaxed site`` ticked.
- **A spin-polarised description of the host.** For a magnetic host, the
  relaxed structure carries the per-atom spins its relaxation started from,
  and the hyperfine job starts from them too. To hand a coupling to Magnetic
  Fields, the source relaxation must itself have been spin polarised, because
  the moments come from its Mulliken populations (see
  `Using the coupling in Magnetic Fields`_).
- **A working CASTEP setup.** Configure a local CASTEP executable or a cluster
  under ``Setup``, and load a keyword catalog (:doc:`/castep_keyword_catalog`).

Running the workflow
--------------------

1. **Open the page.** Choose ``Hyperfine Coupling`` in the navigator; it sits
   before ``Magnetic Fields``.
2. **Configure the rows.** Select one or more rows in ``Hyperfine Coupling
   Jobs`` and choose ``Configure``. The dialog starts from the source
   relaxation's settings (see `Configuration dialog`_).
3. **Submit.** Choose ``Submit Local`` or ``Submit to Cluster``. Each row runs
   as one job; its state follows the job in the Job Dashboard.
4. **Review the result.** When a job completes, its row in ``Results`` shows
   the coupling, the contact field and the cross-check (see
   `Outputs and interpretation`_).
5. **Optionally hand the coupling to Magnetic Fields.** Select one result and
   choose ``Use in Magnetic Fields``.

The workflow page
-----------------

``Hyperfine Coupling Jobs`` lists one row per eligible site with its label,
charge state, relative energy, status and a ``Configure`` button, exactly as
on the Quantum Corrections page. Double-click a label to rename it. The
right-click menu offers ``Configure``, ``Submit``, ``Cancel``, ``Inspect
failed job`` and the shared reset of failed rows; every cancel and reset asks
for confirmation first, and the Delete and Backspace keys act on the selected
rows.

``Results`` has one row per completed job:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Column
     - Meaning
   * - ``Site``
     - The row's label.
   * - ``Charge state``
     - Diamagnetic muon or muonium.
   * - ``A_iso (MHz)``
     - The isotropic coupling from CASTEP's reconstructed tensor, in the
       physical sign convention.
   * - ``B_c (T)``
     - The contact field that :math:`A_\mathrm{iso}` corresponds to.
   * - ``B_c pseudo (T)``
     - The contact field from the pseudo spin density at the muon; empty when
       the job wrote no readable formatted density.
   * - ``Reconstructed / pseudo``
     - ``B_c`` divided by ``B_c pseudo``.
   * - ``Dipolar (MHz)``
     - The three principal values of the traceless anisotropic part, in
       ascending order.

Configuration dialog
--------------------

The dialog shows the shared parameters first, ``basis_precision``,
``cut_off_energy``, ``elec_energy_tol`` and ``kpoint_mp_grid``, then the rest
of the CASTEP keywords the catalog allows for a ``MAGRES`` task. The defaults
come from the source relaxation, with ``elec_energy_tol`` tightened to
``1e-9 eV``: the spin density at the muon is a small difference of two large
densities and needs a well-converged ground state. ``TASK`` is fixed to
``MAGRES``.

The dialog refuses ``MAGRES_TASK``, ``SPIN_POLARIZED`` (in either spelling) and
``WRITE_FORMATTED_DENSITY``, which the workflow sets. The k-point mesh matters
more here than for a relaxation, especially in metals: Onuorah *et al.* found
the contact field converged only with meshes equivalent to 16 × 16 × 16 for
the bcc Fe unit cell.

What MuFinder writes
--------------------

Each job is a fresh CASTEP run with:

- ``TASK : MAGRES`` and ``MAGRES_TASK : HYPERFINE``;
- ``SPIN_POLARIZED : TRUE`` (CASTEP stops a hyperfine run that is not spin
  polarised) and ``WRITE_FORMATTED_DENSITY : TRUE``;
- ``CHARGE : 1`` for a diamagnetic muon and ``CHARGE : 0`` for muonium, with
  ``SPIN=1`` seeded on the muon of a muonium site so the run stays open-shell;
- ``SPECIES_MASS`` and ``SPECIES_GAMMA`` for ``H:mu``. The gyromagnetic ratio is
  the CODATA muon value, so CASTEP scales the muon's tensor by the muon's
  gyromagnetic ratio rather than the proton's.

The job writes ``<seed>.magres``, which holds the tensors, and
``<seed>.den_fmt``, which holds the charge and spin density on CASTEP's fine
grid. Both are retrieved from a cluster with the ``.castep`` output. The
formatted density of a large supercell runs to hundreds of megabytes.

Outputs and interpretation
--------------------------

MuFinder converts CASTEP's tensor before showing it:

- **Sign.** CASTEP's hyperfine tensor has the opposite sign to the physical
  convention: it reports about -4630 MHz for muonium in vacuum, whose coupling
  is +4463 MHz. MuFinder multiplies the tensor by -1, so a positive
  :math:`A_\mathrm{iso}` means majority spin density at the muon.
- **Contact field.** :math:`B_c = A_\mathrm{iso} / (g_e\,\gamma_\mu/2\pi)`, with
  :math:`g_e = 2.0023` and :math:`\gamma_\mu/2\pi = 135.54` MHz/T, so 1 T of
  contact field is 271.4 MHz.
- **Pseudo cross-check.** The spin density column of the formatted density is
  evaluated at the muon by Fourier interpolation and turned into a field with
  :math:`B_c = (2\mu_0/3)\,\mu_B\,\rho_s`.

In a magnetic host the dipolar part of CASTEP's tensor is the dipolar field of
the spin density inside the periodic supercell. Magnetic Fields computes the
dipolar field of the host's moments with a lattice sum over the whole crystal,
so only the contact part is handed over.

The numbers are for a static muon at its relaxed position. The muon's
zero-point motion averages the spin density over its wavefunction, which
reduces :math:`|B_c|`: for the tetrahedral site in Fe, Onuorah *et al.* (2019)
find -1.25 T for a static muon and -1.07 T with zero-point motion included.
MuFinder does not apply that correction.

Using the coupling in Magnetic Fields
-------------------------------------

Magnetic Fields writes the contact field as a ``Hyperfine coupling`` times the
:math:`1/r^3`-weighted mean moment of the muon's nearest magnetic ions, its
contact shell (see :ref:`Choose the contact shell
<magnetic-fields-contact-shell>`). ``Use in Magnetic Fields`` turns the computed
spin density into that coupling by dividing it by the same weighted mean of the
DFT moments, taken from the Mulliken spin populations of the site's source
relaxation. An ion counts as magnetic when its Mulliken moment is at least
0.1 μB.

Each Magnetic Fields form, the main page and every saved workspace, has its
own selected muon site and its own contact shell. ``Use in Magnetic Fields``
fills the ``Hyperfine coupling`` of every form whose selected site is this
result's site, in ``Å⁻³``, computed with that form's ``Contact neighbours
(nnn)`` and ``Contact radius``. If a form already holds a different coupling,
MuFinder lists the forms and asks before overwriting; the default answer keeps
them. If no form shows the site, nothing changes and MuFinder says so: select
the site on the Magnetic Fields page first.

Because the coupling is per unit of DFT moment, Magnetic Fields rescales the
contact field by the moments of its own magnetic structure. When DFT
overestimates the moments, as in itinerant magnets such as MnSi, entering the
measured moments there gives a contact field closer to experiment, the
correction Onuorah *et al.* discuss.

``Use in Magnetic Fields`` is disabled, with the reason in its tooltip, when
the source relaxation has no Mulliken spin populations.

Scope and limitations
---------------------

- Collinear spins only: CASTEP's hyperfine calculation stops for non-collinear
  magnetism and spin-orbit coupling.
- Static muon: no zero-point averaging of the spin density.
- Semilocal functionals can misplace the magnetic moments of itinerant or
  correlated magnets, and the contact field follows the moments. DFT+U changes
  both.
- Metals run, but CASTEP prints a warning that magnetic shielding is not
  implemented for metals; that warning does not apply to the hyperfine tensor.

References
----------

- I. J. Onuorah, P. Bonfà and R. De Renzi, Muon contact hyperfine field in
  metals: A DFT calculation, Phys. Rev. B **97**, 174414 (2018),
  `doi:10.1103/PhysRevB.97.174414 <https://doi.org/10.1103/PhysRevB.97.174414>`_.
- I. J. Onuorah, P. Bonfà, R. De Renzi, L. Monacelli, F. Mauri, M. Calandra and
  I. Errea, Quantum effects in muon spin spectroscopy within the stochastic
  self-consistent harmonic approximation, Phys. Rev. Mater. **3**, 073804 (2019),
  `doi:10.1103/PhysRevMaterials.3.073804 <https://doi.org/10.1103/PhysRevMaterials.3.073804>`_.
- I. J. Onuorah *et al.*, Automated computational workflows for muon spin
  spectroscopy, Digital Discovery (2025),
  `doi:10.1039/d4dd00314d <https://doi.org/10.1039/d4dd00314d>`_.
- S. J. Blundell *et al.*, Electronic structure calculations for muon
  spectroscopy, Electron. Struct. (2025),
  `doi:10.1088/2516-1075/adcb7c <https://doi.org/10.1088/2516-1075/adcb7c>`_.
