The muon in CoF₂
================

Cobalt difluoride is the first of the documentation's worked examples and the
one behind most of its screenshots. It is a well-studied host for muon-site work: the positive muon
stops between two fluorines and pulls them into a linear F–μ–F complex, the
local field at that site has been measured, and the MuFinder paper uses it to
show how a muon-induced distortion changes the calculated dipolar field. This
page follows one project through every stage, from the host structure to the
field the muon feels. The :doc:`/getting_started/first_project` tutorial is the
short route through the same stages, with the clicks to make at each.

.. note::

   No calculation on this page ran CASTEP. The screenshots come from a
   project whose CASTEP outputs are synthetic, written so that MuFinder's own
   analysis of them reproduces the published values cited below. The
   headline numbers are pinned by tests: the field calculations in
   ``tests/science/test_magnetic_fields.py``, and the phonon, density and
   single-point outputs in ``tests/docs/test_docs_synthetic.py``. Your own
   CASTEP calculations of CoF₂ will give values of their own.

The host
--------

CoF₂ has the rutile structure, space group P4₂/mnm, with a = 4.695 Å and
c = 3.180 Å, Co at the 2a sites (0, 0, 0) and (½, ½, ½), and F at the 4f
sites (x, x, 0), (−x, −x, 0), (½ + x, ½ − x, ½) and (½ − x, ½ + x, ½) with
x = 0.306 (Baur and Khan 1971). Below T_N = 37.85 K it is a collinear
antiferromagnet: the moments point along c, the corner and body-centre Co
are antiparallel, and the ordered moment is 2.64 μB per Co (Lines 1965).

.. figure:: ../../_generated/screenshots/host_structure_page.webp
   :alt: The Host Structure page with CoF2 loaded: two Co and four F in the
         atom table, Spin (hbar/2) +3 and -3 on the two Co, and the cell viewed
         down a with antiparallel spin arrows.

   The CoF₂ unit cell, with spins of +3 and −3 ħ/2 on the corner and
   body-centre Co marking the two antiparallel sublattices.

The spins in the ``Spin (hbar/2)`` column mark the two sublattices (see
:doc:`/user_guide/host_structure`). A
calculation with ``Spin Polarized`` ticked in its parameters starts from that
antiferromagnetic order; the example ticks it for the pristine relaxation, the
UEP source job and the positive-muon relaxations, and muonium rows are
spin-polarised by default. The magnitude of the ordered moment enters only the
magnetic model of the field calculation (see `The local field`_).

Setting up
----------

The project uses the PBE functional. Its muon calculations run in a 2 × 2 × 2
supercell of the conventional cell, 48 host atoms and the muon, as in the
MuFinder paper (Huddart *et al.* 2022). ``Suggest…`` in the Supercell Editor
recommends larger cells for production work, since no cell under 200 atoms
keeps the muon's periodic images 10 Å apart (see :doc:`/user_guide/supercells`).

.. figure:: ../../_generated/screenshots/supercell_editor.webp
   :alt: The Supercell Editor with the saved CoF2 2x2x2 supercell of 48 atoms
         selected and its preview viewed down a.

   The saved ``CoF2 2x2x2`` supercell.

The cutoff sweep runs from 300 to 800 eV and the k-point sweep over the grids
``Generate From Spacing…`` gives for this cell down to 0.04 1/Å; the project
adopts 600 eV and a 4 × 4 × 6 grid (see
:doc:`/user_guide/convergence_testing`). The sweep data on the overview are
illustrative.

.. figure:: ../../_generated/screenshots/convergence_overview.webp
   :alt: The Convergence Testing overview with 600 eV and a 4 x 4 x 6 grid
         adopted as project defaults and their pinned convergence plots.

   The adopted cutoff and k-point grid with their pinned plots; the sweep
   data are illustrative.

Candidate muon positions
------------------------

Ten starting positions seed both the positive-muon and the muonium
relaxations (see :doc:`/user_guide/site_generation`). ``Element-attached positions`` places one muon 1.0 Å from each
of the four fluorines, the F–μ starts. ``Random generation`` adds six more by
the MuFinder paper's rules, each at least 1.0 Å from every atom and at least
0.5 Å from the starts before it. The example project fixes the generator's seed, so its starts are the same
on every build; the Generation page does not expose the seed.

.. figure:: ../../_generated/screenshots/site_generation_page.webp
   :alt: The Generation page listing four Attached-F sites and six random
         sites, all selected and marked in the CoF2 cell.

   The ten starts.

``Create Jobs from Selected`` asks for the charge state of the relaxations;
``Both charge states`` stages every start in both tabs, so each start is
relaxed once as a positive muon and once as muonium.

The pristine host
-----------------

A variable-cell relaxation of the host without a muon checks the functional:
the synthetic run ends at a = 4.709 Å and c = 3.188 Å, within 0.3 % of
experiment (see :doc:`/user_guide/structural_relaxation_analysis` and
:doc:`/user_guide/comparison_table`).

.. figure:: ../../_generated/screenshots/comparison_table.webp
   :alt: A comparison table of the CoF2 host against its relaxed cell: lattice
         parameters within 0.3 % of experiment, sub-0.01 Å displacements, and
         the same space group, P4₂/mnm.

   The host against its synthetic relaxed cell.

A first look: the UEP
---------------------

The unperturbed electrostatic potential gives a cheap first estimate of where a
positive muon sits, from the charge density of the host alone (see
:doc:`/user_guide/uep_workflow`). The density on this page is a model density,
not a CASTEP result: spherical Gaussian shells for Co²⁺ (15 electrons) and F⁻
(8 electrons) on a 32 × 32 × 24 grid, neutral against the pseudopotential ionic
charges of 17 for Co and 7 for F.

.. figure:: ../../_generated/screenshots/uep_batch_search.webp
   :alt: The UEP analysis window's Batch Search tab after twelve random starts,
         with every optimized site at the same classical energy.

   A batch search in the model density: all twelve starts reach the same
   minimum.

Every random start relaxes to the 4d position (0, ½, ¼), the T_A site of
De Renzi *et al.* (1984), in the open channel that runs along c through the
octahedral 4c sites. In this density the potential along the channel is
shallow: the 4c site (0, ½, 0) lies 0.086 eV higher and is a saddle point
between neighbouring 4d minima. The hierarchical cut at its default tolerance
splits the twelve symmetry-equivalent minima into two clusters of the same
energy.

.. figure:: ../../_generated/screenshots/uep_clustering_dialog.webp
   :alt: The UEP clustering window with a dendrogram of twelve relaxed sites
         and the cluster energy spread.

   The clustering of the batch-search minima.

The UEP leaves the fluorines where they are, and that is what it misses here:
the positive muon's site is fixed by the F–μ–F bond it forms once the
fluorines move.

Positive-muon relaxations
-------------------------

Each start relaxes in the 2 × 2 × 2 supercell with a charge of +1. The ten
relaxations fall into the three clusters of the MuFinder paper.

.. figure:: ../../_generated/screenshots/muon_relaxations_page.webp
   :alt: The Relaxations page, Positive muon tab, with ten completed CoF2
         relaxations, their initial and final positions and total energies.

   Ten positive-muon relaxations from synthetic outputs.

Clustering the relaxed positions as connected components with a 0.25 Å
tolerance gives three distinct sites (see
:doc:`/user_guide/structural_relaxation_analysis`):

- **4c, (0, ½, 0), the lowest.** Seven of the ten starts end here, midway
  between two fluorines. The two nearest F move in by 0.50 and 0.57 Å and the
  two nearest Co move out by 0.14 and 0.17 Å, the displacements of the
  MuFinder paper. The F–F distance falls from 3.40 to 2.33 Å, against 2.36 Å
  from DFT and 2.43(2) Å from the zero-field μSR line shape (Möller *et al.*
  2013).
- **4d, (0, ½, ¼), 0.34 eV higher.** De Renzi's T_A site, the UEP minimum.
- **8i, (0.65, 0.84, 0), 0.82 eV higher.** De Renzi's T_F site, a muon bonded
  to a single fluorine.

The energies above the lowest site, 0.34 and 0.82 eV, are the MuFinder paper's.

.. figure:: ../../_generated/screenshots/relaxation_analysis_workspace.webp
   :alt: The Positive muon sites workspace with ten relaxations clustered into
         4c, 4d and 8i sites at 0, 0.34 and 0.82 eV.

   The positive-muon workspace, clustered.

``Analyse Displacements`` shows the F–μ–F distortion directly: the two
fluorines 1.70 Å from the 4c site move in by half an ångström, the two Co
2.35 Å away move out, and in the synthetic relaxation nothing further away
moves.

.. figure:: ../../_generated/screenshots/displacement_analysis_dialog.webp
   :alt: The Analyse Displacements dialog for the 4c site: two F at 1.70 A
         displaced by -0.50 and -0.57 A, two Co at 2.35 A by +0.14 and +0.17 A.

   Radial displacements around the 4c site.

Muonium relaxations
-------------------

The same ten starts relax as neutral muonium, each to the nearer of two
sites. Seven end at Möller *et al.*'s interstitial site, near
(0.56, 0.84, 0.50), which MuFinder lists by the symmetry-equivalent 8i
representative (0.44, 0.16, 0.50); three end at T_A, 0.27 eV higher, listed
as the 4d representative (0, ½, ¾). Muonium only nudges its neighbours
outwards by a few hundredths of an ångström. The second site and its energy
are illustrative.

A start can reach different sites in the two charge states: ``Attached-F
Site 1`` ends at 4c as a positive muon and at T_A as muonium, while
``Random Site 2`` ends at T_A in both.

.. figure:: ../../_generated/screenshots/muonium_analysis_workspace.webp
   :alt: The Muonium sites workspace with ten relaxations clustered into an 8i
         site of seven and a 4d site of three, 0.27 eV higher.

   The muonium workspace, clustered.

Pinning both workspaces fills the two cards of the ``Muon Sites`` summary,
the sites the later stages work from. Each charge state's relaxation of a
start reaches those stages on its own, so Quantum Corrections, Energy
Barriers and Magnetic Fields list the three positive-muon sites and the two
muonium sites.

.. figure:: ../../_generated/screenshots/muon_sites_summary.webp
   :alt: The Muon Sites summary with three positive-muon sites (4c, 4d, 8i)
         and two muonium sites (8i, 4d) pinned.

   Both charge states pinned.

Zero-point motion
-----------------

The muon at the 4c site and its two fluorines vibrate as a linear F–μ–F
molecule. Its four modes are the symmetric stretch, in which the fluorines
move and the muon does not, two bends across the F–F axis, and the
antisymmetric stretch along it. Möller *et al.* give 585, 3076, 3473 and
4570 cm⁻¹, and the synthetic phonon run of the 4c site embeds exactly those
modes in the supercell, below a host band that ends at 600 cm⁻¹.

.. figure:: ../../_generated/screenshots/quantum_corrections_page.webp
   :alt: The Quantum Corrections page with the 4c site's completed phonon job:
         a muon ZPE of 0.689 eV from modes at 3076, 3473 and 4570 cm-1.

   The phonon result of the 4c site.

Quantum Corrections reads the three modes with the largest muon participation,
the two bends and the antisymmetric stretch, each 99.7 % muon, and sums their
zero-point energies to 0.689 eV. Möller's 0.73 eV is the zero-point energy of
the whole complex, which adds the 0.036 eV of the fluorines' symmetric stretch
(see :doc:`/user_guide/quantum_corrections` and
:doc:`/user_guide/quantum_corrections_theory`).

.. figure:: ../../_generated/screenshots/phonon_analysis_window.webp
   :alt: The phonon analysis window listing the host modes below 600 cm-1 and
         the three muon modes at 3076, 3473 and 4570 cm-1.

   The phonon modes of the 4c site, with the lower bend selected.

The local field
---------------

The magnetic model has one collinear component per sublattice: 2.64 μB along
c on the corner Co, and −2.64 μB on the body-centre Co.

.. figure:: ../../_generated/screenshots/cof2_magnetic_structure_builder.webp
   :alt: The Magnetic Structure Builder with two collinear components, Co
         corner [0, 0, 2.64] and Co body centre [0, 0, -2.64], and the cell
         preview with antiparallel moments.

   The antiferromagnet as two collinear components.

The muon site is the pinned 4c cluster, with the correction for the
muon-induced displacements turned on. No demagnetizing field is needed: the
antiferromagnet has no net moment.

.. figure:: ../../_generated/screenshots/magnetic_muon_site_dialog.webp
   :alt: The Select Muon Site dialog listing the five pinned sites, the 4c
         positive-muon cluster selected and the displacement correction ticked.

   The relaxed 4c site, with the distortion correction.

.. figure:: ../../_generated/screenshots/cof2_magnetic_fields_page.webp
   :alt: The Magnetic Fields page for CoF2: a distortion-corrected field of
         0.1996 T along c, broken down into a 0.2655 T dipolar term and a
         0.066 T distortion correction.

   The field at the 4c site: 0.2655 T from the rigid lattice, 0.1996 T with
   the synthetic distortion.

The field at the undistorted 4c site is 0.2655 T along c, the 0.265 T of the
MuFinder paper and of De Renzi *et al.*'s rigid-dipole estimate (2650 G).
Moving the two nearest Co out by 0.14 and 0.17 Å reduces it by 0.066 T, to
0.1996 T. The MuFinder paper's correction, taken from its full relaxed
supercell, gives 0.207 T, and the field measured by zero-field μSR is 0.228 T
(2288(1) G, De Renzi *et al.* 1984). The synthetic relaxation moves only the
four nearest ions, so its correction is a little larger than the paper's.

The dipolar tensor of each sublattice at the muon site shows where the field
comes from. The corner sublattice gives diag(−0.166, 0.273, −0.107) T/μB at
(0, ½, 0), and the body-centre sublattice diag(0.149, −0.143, −0.006) T/μB.
Divided by μ₀μB/(4π Å³) = 0.927 T/μB, these are the lattice sums
diag(−0.179, 0.295, −0.115) and diag(0.161, −0.154, −0.007) Å⁻³; De Renzi
*et al.* tabulate (0.292, −0.177, −0.115) and (−0.152, 0.159, −0.006) Å⁻³ at
the equivalent site (½, 0, 0), where x and y are interchanged. With
antiparallel moments along c the field is
2.64 μB × (−0.107 + 0.006) T/μB ≈ −0.265 T.

.. figure:: ../../_generated/screenshots/dipolar_tensor_dialog.webp
   :alt: The Dipolar Tensor dialog for the corner Co sublattice at the 4c
         site, a diagonal tensor in T per muB.

   The corner-sublattice dipolar tensor at the 4c site.

Above T_N the Co moments fluctuate and the muon sees the static nuclear
dipoles, mostly those of ¹⁹F. With the F–μ–F distortion applied, the
zero-field polycrystal width is σ = 1.49 μs⁻¹ (see
:doc:`/user_guide/magnetic_fields`, *Calculate nuclear relaxation*).

.. figure:: ../../_generated/screenshots/nuclear_relaxation_dialog.webp
   :alt: The Nuclear Relaxation dialog with a zero-field polycrystal result of
         sigma 1.49 per microsecond at the distortion-corrected 4c site.

   The nuclear dipolar width at the 4c site.

``Save Workspace`` keeps the calculation as a workspace of its own.

.. figure:: ../../_generated/screenshots/cof2_magnetic_field_workspace.webp
   :alt: The saved CoF2 4c site magnetic-field workspace with its
         distortion-corrected result.

   The saved field workspace.

Hopping between 4c sites
------------------------

``Generate Paths`` with the 4c site as both endpoints lists the
symmetry-distinct hops between neighbouring 4c sites (see
:doc:`/user_guide/energy_barriers`). The NEB band of a hop, and how its barrier
compares with the muon's zero-point energy, are followed through with real
CASTEP outputs in :doc:`/user_guide/examples/gdru2si2`.

.. figure:: ../../_generated/screenshots/energy_barriers_page.webp
   :alt: The Energy Barriers page with the 4c site chosen as both endpoints
         and the generated 4c to 4c path of 1.59 A.

   The 4c to 4c hop along c.

References
----------

- W. H. Baur and A. A. Khan, Rutile-type compounds. IV. SiO₂, GeO₂ and a
  comparison with other rutile-type structures, Acta Cryst. B **27**, 2133
  (1971), https://doi.org/10.1107/S0567740871005466.
- M. E. Lines, Magnetic properties of CoF₂, Phys. Rev. **137**, A982
  (1965), https://doi.org/10.1103/PhysRev.137.A982.
- R. De Renzi, G. Guidi, P. Podini, R. Tedeschi, C. Bucci and S. F. J. Cox,
  Magnetic properties of MnF₂ and CoF₂ determined by implanted positive
  muons. I. Localization studies, Phys. Rev. B **30**, 186 (1984),
  https://doi.org/10.1103/PhysRevB.30.186.
- J. S. Möller, D. Ceresoli, T. Lancaster, N. Marzari and S. J. Blundell,
  Quantum states of muons in fluorides, Phys. Rev. B **87**, 121108(R)
  (2013), https://doi.org/10.1103/PhysRevB.87.121108.
- B. M. Huddart *et al.*, MuFinder: a program to determine and analyse muon
  stopping sites, Comput. Phys. Commun. **280**, 108488 (2022),
  https://doi.org/10.1016/j.cpc.2022.108488.
