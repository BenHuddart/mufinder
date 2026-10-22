The muon in GdRu₂Si₂
====================

GdRu₂Si₂ is a centrosymmetric tetragonal magnet whose Gd moments form a
skyrmion lattice in an applied field. Huddart *et al.* (2025) found two
candidate positive-muon sites in it, and then showed that only one of them
holds a muon: the second sits so close to the top of the barrier between two
copies of the first that the muon's zero-point motion carries it over. This
page follows that argument through MuFinder, from the two relaxed sites to the
nudged elastic band (NEB) between them and the phonons of each site, and is the
worked example of :doc:`/user_guide/energy_barriers`.

.. note::

   The outputs on this page are the paper's own CASTEP outputs: the two
   relaxed supercells, the NEB band, trimmed to its final iteration, and the
   Γ-point ``.phonon`` file of each site, from the paper's data set,
   https://doi.org/10.15128/r15138jd91z. Only the ``.castep`` logs are
   synthesised, around the real final energies, since the data set keeps no
   relaxation log. The files are in
   ``tests/fixtures/gdru2si2``, whose ``README.md`` lists their sources, and
   ``tests/docs/test_docs_gdru2si2.py`` pins every number MuFinder reads from
   them below.

The host and the calculations
-----------------------------

GdRu₂Si₂ has the ThCr₂Si₂ structure, space group I4/mmm, with a = 4.1634 Å
and c = 9.6102 Å (Hiebl *et al.* 1983): Gd at 2a (0, 0, 0), Ru at 4d
(0, ½, ¼) and Si at 4e (0, 0, z). The project takes z = 0.3682 from the Si of
the relaxed cells farthest from the muon, and enters the conventional cell, ten
atoms, with a spin of 7 ħ/2 on each Gd: the calculations treated the Gd moments
as ferromagnetic, which is not the magnetic order of the material but gives a
spin-polarised ground state to relax in.

The paper's calculations used PBE, a 1000 eV cut-off, spin polarisation,
charge +1 for the positive muon, and a 2 × 2 × 1 supercell of the conventional
cell, 40 host atoms and the muon. The relaxations used a 5 × 5 × 5
Monkhorst–Pack grid and a force tolerance of 0.05 eV/Å; the NEB band and the
phonons used 3 × 3 × 3. No convergence sweep ran in this project, so the cut-off
and the 5 × 5 × 5 grid of the supercell are set by hand as the project defaults
(:doc:`/user_guide/convergence_testing`), and the relaxation rows add
``Spin Polarized`` and ``GEOM_FORCE_TOL``. The paper's runs also selected the
ensemble DFT minimiser (``ELEC_METHOD : EDFT``), which MuFinder's keyword
catalog does not implement (:doc:`/castep_keyword_catalog`), so the inputs
MuFinder writes for these rows leave it out.

Starting positions
------------------

The paper generated its starting positions with MuFinder's random generation:
each start at least 0.5 Å from the earlier ones and their symmetry
equivalents, and at least 1.0 Å from every atom. Those rules, set under
``Random generation`` with ``Muon-muon separation (Å)`` (see
:doc:`/user_guide/site_generation`), fill the conventional cell with 38 starts
for the seed this project uses. The paper's draw stopped at 30.

.. figure:: ../../_generated/screenshots/gdru2si2_site_generation.webp
   :alt: The Initial Muon Position Generation page with random generation,
         a muon-muon separation of 0.5 A and a minimum muon-atom distance of
         1 A, the generated random sites listed and marked in the GdRu2Si2
         cell.

   Random starts in the GdRu₂Si₂ cell under the paper's generation rules.

The paper's 30 relaxations ended in two distinct sites (its Table SI). Rather
than repeat them, this project places one start at each site with ``Custom
coordinates`` and relaxes those two as positive muons in the 2 × 2 × 1
supercell.

The two sites
-------------

.. figure:: ../../_generated/screenshots/gdru2si2_muon_relaxations.webp
   :alt: The Structural Relaxations page with Manual Site 1 and Manual Site 2
         completed in the GdRu2Si2 2x2x1 supercell, with total energies of
         -69184.849 and -69184.592 eV.

   The two relaxations, with the paper's relaxed cells and final energies.

.. list-table:: The two muon sites, in conventional fractional coordinates.
   :header-rows: 1

   * - Site
     - Table SI position
     - Relaxed position
     - Wyckoff
     - Energy (eV)
   * - 1
     - (0.50, 0.00, 0.07)
     - (0.5000, 0.0000, 0.0716)
     - 8g
     - 0
   * - 2
     - (0.21, 0.21, 0.16)
     - (0.2095, 0.2095, 0.1606)
     - 16m
     - 0.257

Site 1 lies on the line along c between two Ru, 0.69 Å above the Gd plane;
site 2 lies on the diagonal mirror, 1.54 Å above the plane. Their energy difference, 0.257
eV, is the 0.26 eV of Table SI. An analysis workspace of both relaxations
clusters them into two sites (see
:doc:`/user_guide/structural_relaxation_analysis`), and pinning it makes them
the sites Energy Barriers and Quantum Corrections offer.

.. figure:: ../../_generated/screenshots/gdru2si2_analysis_workspace.webp
   :alt: The GdRu2Si2 sites analysis workspace with cluster 1 at
         (0.5000, 0.0000, 0.0716), Wyckoff 8g, and cluster 2 at
         (0.2095, 0.2095, 0.1606), Wyckoff 16m, 0.257 eV higher.

   The two sites clustered in their analysis workspace.

The paths between them
----------------------

On the Energy Barriers page, ``Generate Paths`` with site 1 as both endpoints
lists the symmetry-distinct hops between copies of site 1. The automatic
length cutoff keeps only the 1.375 Å hop along c, across the Gd plane to the
mirror image of site 1 below it; with ``Max path length (Å)`` set to 3 it also
lists the 2.944 Å hop in the plane, from (½, 0, z) to (0, ½, z). That second
hop is the paper's 1 → 1′ path, and its minimum-energy path passes through
site 2, on the diagonal mirror half way between the two copies of site 1. With
site 2 as the final endpoint, ``Generate Paths`` gives one hop of 1.719 Å, the
first half of that path.

.. figure:: ../../_generated/screenshots/gdru2si2_energy_barrier_paths.webp
   :alt: The Paths tab with site 1 as the initial endpoint and site 2 as the
         final one, and three diffusion paths: site 1 to site 1 of 1.375 and
         2.944 A, ready, and site 1 to site 2 of 1.719 A, run with 4 farms,
         completed.

   The 1 → 1′ hops and the 1 → 2 hop that the NEB band follows.

``View...`` on the 1 → 2 row shows the hop in the pristine supercell, here
viewed down a, where the two muon positions near the bottom of the cell stand
clear of the atom columns.

.. figure:: ../../_generated/screenshots/gdru2si2_path_preview.webp
   :alt: The path viewer with the GdRu2Si2 pristine supercell viewed down a
         and the arrow from site 1 to site 2 near the bottom of the cell.

   The site 1 to site 2 hop in the pristine supercell.

The NEB band
------------

The paper's band from site 1 to site 2 had 40 images, 4 task farms, a
transition-state force tolerance of 0.08 eV/Å and a 3 × 3 × 3 grid. The NEB
dialog of the 1 → 2 row inherits the cut-off, functional and spin treatment of
the site 1 relaxation and takes those four settings. ``TSSEARCH_MAX_PATH_POINTS``
counts the intermediate images, not the endpoints. MuFinder always runs
climbing-image NEB, and with climbing on CASTEP adds one image to an even
count, so MuFinder rounds the row's 40 up to the 41 that CASTEP would run. The
paper's run had climbing off (its ``.castep`` reports the central bead as
sliding down), so CASTEP kept 40.

.. figure:: ../../_generated/screenshots/gdru2si2_neb_configuration.webp
   :alt: The Configure NEB Parameters dialog for the site 1 to site 2 path
         with a cut-off of 1000 eV, a 3 x 3 x 3 k-point grid, 41 path points,
         a force tolerance of 0.08 eV/A, PBE, spin polarised and 4 task farms.

   The NEB settings of the 1 → 2 path.

The band of a farmed run is written to ``<seed>_farm001.ts``, which is the file
MuFinder reads (see :doc:`/user_guide/energy_barriers`, *Task farming*). The
``NEB Analysis`` tab lists it with 42 images: the 40 of the final iteration
between the two endpoints.

.. figure:: ../../_generated/screenshots/gdru2si2_neb_analysis.webp
   :alt: The NEB Analysis tab with the site 1 to site 2 band: a forward
         barrier of 0.495247 eV, a reverse barrier of 0.212521 eV, a Delta E
         of 0.282726 eV and 42 images, and the energy profile with image 23
         of 42, the transition state, in the viewer.

   The paper's NEB band, with its highest image, 23 of 42, in the viewer.

The band rises 0.495 eV above site 1, at reaction coordinate 0.63, and falls to
site 2, which it places 0.283 eV above site 1; the reverse barrier, from site 2
back to site 1, is 0.213 eV. Site 2 is higher here than in Table SI (0.26 eV)
because the band used the 3 × 3 × 3 grid. By symmetry the second half of the
1 → 1′ path mirrors the first, so the whole profile is 1 → 2 → 1′: up 0.495 eV,
down into the shallow well of site 2, and up and over the same barrier again.

Zero-point energy and stability
-------------------------------

A classical barrier of 0.5 eV would keep a muon at site 1, and one of 0.2 eV
might keep it at site 2, but the muon's zero-point energy (ZPE) is comparable to
both. The phonon job of each pinned site (see
:doc:`/user_guide/quantum_corrections`) uses finite displacement at Γ, the
3 × 3 × 3 grid and 4 task farms, as the paper's runs did.

.. figure:: ../../_generated/screenshots/gdru2si2_quantum_corrections.webp
   :alt: The Quantum Corrections page with both pinned sites completed with
         4 farms, and their phonon results: site 2 with a ZPE of 0.654 eV,
         modes 2129.8, 4019.8 and 4396.9 cm-1 and w1 -0.008175, and site 1
         with 0.558 eV, modes 1136.3, 3225.0 and 4632.8 cm-1 and w1 -0.007937.

   The paper's phonons of the two sites.

MuFinder takes the three Γ modes with the largest muon participation, which for
both sites are the three highest of the 123, and reproduces the paper's values:

.. list-table:: MuFinder's analysis of the paper's phonons, beside the paper.
   :header-rows: 1

   * - Site
     - Muon modes (cm⁻¹)
     - ZPE (eV)
     - Paper ZPE (eV)
     - w1
     - Paper w1
   * - 1
     - 1136.3, 3225.0, 4632.8
     - 0.558
     - 0.56
     - −0.0079
     - −0.0079
   * - 2
     - 2129.8, 4019.8, 4396.9
     - 0.654
     - 0.65
     - −0.0082
     - −0.0082

Both witnesses are close to zero, so the muon's motion separates from the
lattice's and the sum of the three half-quanta is a fair ZPE (see
:doc:`/user_guide/quantum_corrections_theory`). The runs set the muon mass to
0.1138 amu, 0.33 % above the muon mass of 0.113429 amu that MuFinder writes, so
their frequencies are 0.16 % low, a change below the precision quoted here.
MuFinder's mass check allows 1 %, so it accepts both results without the
muon-mass warning (see :doc:`/user_guide/quantum_corrections`, *Muon mass*).

Only the part of the ZPE along the path competes with the barrier. The paper
projects the muon's dynamical matrix onto the direction of the path, a step
MuFinder does not take, and finds a one-dimensional ZPE of 0.25 eV at site 2,
along the 1 → 1′ line. That exceeds the 0.21 eV barrier from site 2 back to
site 1, so a muon at site 2 is not stable: it leaves for site 1. At site 1 the
one-dimensional ZPE is 0.18 eV, well below the 0.495 eV barrier, so the muon
can leave only by tunnelling. A WKB estimate along the band gives a
transmission of about 1.3 × 10⁻¹⁶ and, with the site 1 oscillator frequency as
the attempt frequency, a hopping rate of about 0.05 Hz, far slower than a μSR
measurement. Site 1 is therefore the one stopping site of the muon in
GdRu₂Si₂.

References
----------

- B. M. Huddart, A. Hernández-Melián, G. D. A. Wood, D. A. Mayoh,
  M. Gomilšek, Z. Guguchia, C. Wang, T. J. Hicken, S. J. Blundell,
  G. Balakrishnan and T. Lancaster, Field-orientation-dependent magnetic
  phases in GdRu₂Si₂ probed with muon-spin spectroscopy, Phys. Rev. B **111**,
  054440 (2025), https://doi.org/10.1103/PhysRevB.111.054440.
- K. Hiebl, C. Horvath, P. Rogl and M. J. Sienko, Magnetic properties and
  structural chemistry of ternary silicides (RE, Th, U)Ru₂Si₂ (RE = rare
  earth), J. Magn. Magn. Mater. **37**, 287 (1983),
  https://doi.org/10.1016/0304-8853(83)90058-6.
