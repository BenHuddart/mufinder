UEP Workflow
============

The ``UEP`` stage finds candidate muon stopping sites cheaply, from a single
CASTEP calculation on the host crystal. You run one CASTEP single-point source
job that writes the host charge density; MuFinder then evaluates the
electrostatic potential that the host would exert on a positive muon, relaxes
many random starting positions on that potential, and groups the resulting
minima into clusters. The cluster representatives can be committed as starting
positions for full CASTEP relaxations on the ``Muon Sites -> Generation`` page.
The physical picture, the equations and the papers behind the method are in
`Method`_ at the end of this page.

Before you start
----------------

The UEP stage needs:

- **A host structure.** Load one first. The ``Source Structure`` group on the
  UEP page selects the structure the source jobs run on (the active host by
  default).
- **A supercell, if you want one.** Each source job runs on ``Unit cell`` or on
  a supercell saved in the :doc:`/user_guide/supercells`.
- **A neutral host.** The method is for a positively charged muon in an
  unperturbed, neutral crystal; a charged ``.castep`` cell, or a density that
  does not integrate to a neutral cell, is rejected. See `When the method is
  trustworthy`_ before relying on the results.
- **A working CASTEP setup.** Configure a local CASTEP executable or a cluster
  under ``Setup``, and load a keyword catalog (:doc:`/castep_keyword_catalog`).
  Submission and monitoring are described in :doc:`/getting_started/running_jobs`.

Converged settings are optional but recommended: new rows use the project
default cutoff, k-point grid and XC functional adopted in
:doc:`/user_guide/convergence_testing`.

Files the analysis needs
~~~~~~~~~~~~~~~~~~~~~~~~

A completed UEP source job must provide:

- ``.den_fmt`` for the formatted electron density;
- ``.castep`` for the final host structure and pseudopotential report.

Optional support files:

- ``.cell`` may be used as a fallback source of ``SPECIES_POT`` metadata;
- referenced pseudopotential files may also be consulted when the CASTEP output
  does not contain a usable pseudopotential report.

Because the formatted density is mandatory, MuFinder locks
``write_formatted_density = TRUE`` in the UEP source-job parameter dialog. The
source files are validated when the analysis window opens.

Running the workflow
--------------------

Open ``UEP`` in the project navigator, then:

1. **Pick the source structure.** In the ``Source Structure`` group, choose the
   ``Structure`` to run on (the active host by default).
2. **Add a source job.** Click ``Add Job`` to add a row to ``UEP Source
   Jobs``. Choose its ``Supercell`` (``Unit cell`` or a saved supercell;
   ``Manage Supercells...`` in the drop-down opens the
   :doc:`/user_guide/supercells`).
3. **Check the parameters.** Click ``Configure...`` in the row's ``Parameters``
   column to adjust the CASTEP settings, or select several rows and click
   ``Configure Selected Parameters...``.
4. **Run the job.** Click ``Run All Locally`` or ``Run All on Cluster`` (the
   ``Run Selected ...`` variants appear when rows are selected) and track the
   job through the ``Job Dashboard``.
5. **Open the analysis window.** When the row is ``completed``, click ``Open
   Wizard`` in its ``Analysis`` column. The window ``UEP Analysis - <job
   label>`` opens with three tabs (see `The analysis window`_).
6. **Run a batch search.** On the ``Batch Search`` tab, set the sampling and
   optimizer values and click ``Run Batch Search``. MuFinder generates random
   starting positions and relaxes each on the potential.
7. **Optionally add zero-point energies.** Once the optimization finishes,
   click ``Calculate ZPE`` to estimate the harmonic zero-point energy of every
   optimized site that is a stable minimum.
8. **Cluster the relaxed sites.** Click ``Clustering`` in the ``Sampling``
   card to open the clustering window and explore the relaxed sites (see
   `Interpreting batch searches`_).
9. **Commit sites for relaxation.** Open ``Muon Sites -> Generation``, choose
   the ``UEP clusters`` method and generate sites from the cluster
   representatives (see `Using UEP clusters for relaxation starts`_).

The UEP page
------------

.. figure:: ../_generated/screenshots/uep_page.png
   :alt: The UEP page with one completed source job on the silicon host and its
         Open Wizard button.

   A completed UEP source job, from a synthetic silicon density.

The page has a ``Source Structure`` group and a ``UEP Source Jobs`` table with
these columns: ``Label``, ``Supercell``, ``Status``, ``Total Energy (eV)``,
``3D Potential``, ``Parameters``, ``Analysis`` and an unlabelled column with an
``x`` button that removes the row. The ``3D Potential`` column reads ``Saved``
once a full-cell potential grid has been calculated and stored for that job,
and ``Not calculated`` for a completed job without one.

The analysis window
-------------------

Completed UEP source jobs open ``UEP Analysis - <job label>`` with three tabs:
``Batch Search``, ``Isosurfaces``, and ``Slices``. The window offers:

- full-cell 3D potential grids (``Calculate Cell Potential Grid``) shared
  between the isosurface and slice views, with a ``Gaussian width factor``
  control, zero-minimum shifting (``Set minimum potential to zero``) and XSF
  export (``Export XSF...``);
- responsive isosurface visualization from the cached full-cell potential grid;
- interactive 2D slice planes (``Add Plane``) defined from one selected atom
  plus a normal, three selected atoms, or a best-fit plane through four or more
  selected atoms;
- multi-atom selection in the structure viewer using ``Shift``-click so slice
  planes can be defined directly from the host structure;
- slice saturation controls (``Slice saturation``, ``Min (eV)``, ``Max
  (eV)``) and a colorbar that marks when clipped display limits exclude lower
  or higher physical potentials from the visible range;
- random-start batch searches on the ``Batch Search`` tab. The ``Sampling``
  card sets ``Random starts`` (default 12), ``Min muon-muon distance (A)``
  (default 0.8), ``Min muon-atom distance (A)`` (default 1.0) and ``Max
  attempts`` (default 5000). The optimizer card sets ``Gaussian width factor``,
  ``Optimization tolerance (eV)`` (default 1e-5) and ``Optimizer max steps``
  (default 100). ``Run Batch Search`` generates and relaxes the candidate
  starts first; an optional ``Calculate ZPE`` pass follows for the optimized
  sites, and ``Clustering`` opens a separate window for exploring the relaxed
  sites;
- muon markers that can be added to the isosurface and slice viewers with
  ``Add / remove muon``, either from clustered batch-search results or from
  custom fractional coordinates;
- cluster tables on the ``Batch Search`` tab that reflect the most recent
  clustering run and keep promotion of cluster representatives;
- exposure of clustered batch-search results to the ``Muon Sites ->
  Generation`` page, where selected UEP rows can be committed as candidate
  muon-site inputs for structural relaxation.

The clustering window, ``UEP Clustering - <job label>``, offers hierarchical
dendrogram exploration of the Soprano-backed feature space with a live
tolerance control (``Tolerance t``), a k-means PCA view with live cluster-count
updates (``Clusters k``), a shared structure view that colours every relaxed
muon position by its cluster, and the cluster-summary bubble plot. A
``Clustering method`` drop-down switches between ``Hierarchical`` and
``K-means``. On a small window it reflows only its controls, so the dendrogram
or PCA plot stays beside the structure view and energy-spread plot instead of
being stacked into a thin strip.

Interpreting batch searches
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/uep_clustering_dialog.png
   :alt: The UEP clustering window with a hierarchical dendrogram, the clustered
         muon positions in the cell and the cluster energy-spread plot.

   Twelve batch-search minima in the synthetic silicon density, in three clusters.

The paper and the original ``pymuonsuite`` workflow both use repeated
random-start optimizations to estimate which minima have the largest attraction
basins. MuFinder follows the same high-level idea:

- a batch of initial muon positions is generated inside the chosen source cell;
- each start is optimized under the UEP approximation;
- the resulting optimized sites can then be post-processed with a separate
  ``Calculate ZPE`` action, which evaluates the harmonic quantum correction in
  eV for each successfully optimized site and stores it with the relaxed-site
  data;
- clustering is then applied separately to the relaxed sites using the same
  Soprano-backed asymmetric-position and energy features used by
  ``pymuonsuite``;
- hierarchical clustering can be explored through a dendrogram and tolerance
  cut, while k-means can be explored through a PCA projection of the normalized
  feature vectors;
- in the hierarchical view, the tolerance ``t`` controls where the dendrogram
  is cut: smaller values split nearby minima into finer groups, while larger
  values merge nearby relaxed sites into broader candidate basins;
- in the k-means view, the user chooses a fixed ``k`` explicitly, so it is best
  treated as an exploratory comparison against the hierarchical picture rather
  than as a proof of the physically correct site count;
- each cluster is summarized by its size, a representative relaxed site, and
  cluster energy statistics; the representative is the minimum-energy member of
  that cluster rather than an averaged coordinate or centroid, matching the
  style used in the ``pymuonsuite`` cluster reports;
- the clustering window bubble plot shows average energy relative to the global
  minimum on the horizontal axis and the cluster energy standard deviation on
  the vertical axis, so broad/high-energy clusters can be distinguished from
  compact low-energy families at a glance;
- larger clusters usually indicate minima with larger attraction basins and are
  often the most physically relevant first-pass candidates.

Using UEP clusters for relaxation starts
----------------------------------------

The analysis window does not write muon sites directly. After running a batch
search and clustering the optimized points, open ``Muon Sites -> Generation``
and choose the ``UEP clusters`` method. That table lists one saved
representative per cluster from the UEP analysis state, including source job,
cluster size, stationary classification, classical energy ``E (eV)``, and
fractional coordinates.

A cluster-row position is the optimized position of the representative member
recorded in the UEP cluster summary. It is not the mean position, centroid, or
PCA/k-means centre of the cluster. MuFinder chooses that representative as the
lowest-energy optimized member of the cluster, using ZPE-corrected total energy
when those totals are available and falling back to the classical UEP energy
otherwise. Individual non-representative members remain available in the UEP
analysis state but are intentionally hidden from this version of the generation
table to keep the relaxation-start list compact.

Selecting rows in the generation tab previews the corresponding muon positions
over the host structure. Clicking ``Generate Sites`` then records the selected
UEP rows as normal generated muon sites for the full structural-relaxation
workflow. MuFinder keeps the source job, cluster index, representative member
index, clustering method, energies, and classification with each created site,
and protects against adding the same UEP cluster row twice.

Before staging the generated sites for CASTEP relaxation, you can use
``Cluster Positions...`` in the ``Generation`` tab to rewrite selected
generated rows to the symmetry-equivalent representatives closest to a chosen
fractional anchor. This is useful when several generated starts correspond to
the same physical UEP site but sit in different symmetry-equivalent parts of
the unit cell. Choosing a common representative avoids launching relaxations
from different realizations of the same symmetry-equivalent site merely because
the initial coordinates used different members of the symmetry orbit. See
:doc:`/user_guide/site_generation` for the whole generation page.

Saving analysis sessions
------------------------

The UEP workflow mixes durable project data with transient exploratory state, so
the save boundary matters:

- completed source jobs, saved outputs, and job-tracking information remain part
  of the durable project record and are protected by the normal autosave path;
- UEP analysis-session state remains in memory until you choose ``Save
  Project`` (or confirm the save prompt when closing a modified project);
- this manual-save state includes the current wizard tab, cached analysis UI
  choices, clustering-window state, selected viewer orientation/zoom, bond and
  spin visibility, and slice/isosurface presentation settings.

The clustering window's geometry, splitter sizes and camera are saved with the
UEP analysis state once a move, resize, splitter drag or view change settles
(after about a quarter of a second), not on every intermediate event, so
dragging stays responsive however large the potential grid is. Method,
tolerance, cluster-count and result changes are saved immediately.

You can therefore explore candidate sites, refresh overlays, or compare
clustering views without immediately rewriting the project on disk. To restore
that exact UEP analysis layout in a future session, use ``Save Project`` before
closing the project.

When the method is trustworthy
------------------------------

The UEP stage is intended for a positively charged muon, not for muonium.
It is most reliable when the implanted muon does not strongly reconstruct the
host lattice or form a strongly directional bond. In practice that means the
approximation is usually more trustworthy for metallic or weakly perturbative
systems than for insulating hosts with strong local bonding.

MuFinder validates and documents the following conditions:

- the host cell must be neutral (a charged ``.castep`` cell, or a density that
  does not integrate to a neutral cell, is rejected);
- the workflow requires CASTEP-style plane-wave density input;
- the workflow currently expects ultrasoft-pseudopotential-style ionic metadata;
- if ionic metadata cannot be recovered from the ``.castep`` output, MuFinder
  falls back to ``SPECIES_POT`` references from the ``.cell`` file when
  possible.

Method
------

Theory and attribution
~~~~~~~~~~~~~~~~~~~~~~

The UEP workflow in MuFinder is based on the method described in:

- Simone Sturniolo and Leandro Liborio, ``Computational prediction of muon
  stopping sites: A novel take on the unperturbed electrostatic potential
  method``, *Journal of Chemical Physics* **153**, 044111 (2020).
  DOI: ``10.1063/5.0012381``.
- Leandro Liborio, Simone Sturniolo, and Dominik Jochym, ``Computational
  prediction of muon stopping sites using ab initio random structure searching
  (AIRSS)``, *Journal of Chemical Physics* **148**, 134114 (2018).
  DOI: ``10.1063/1.5024450``.

MuFinder implements this workflow in its own codebase rather than wrapping
``pymuonsuite``, but the same core physical picture is used:

- The host is treated as an unperturbed, neutral crystal and the implanted
  probe is a positively charged muon rather than muonium.
- The muon is assumed to feel the electrostatic potential generated by the
  electronic density and the ionic charge distribution of the *host* crystal.
- Because the host is periodic and neutral, the potential is evaluated in
  reciprocal space rather than by a slowly convergent real-space Coulomb sum.
- The electronic contribution is taken from a single CASTEP formatted density,
  so the workflow requires only one host calculation before analysis begins.
- The ionic contribution is approximated with Gaussian charge distributions.
  MuFinder exposes the same ``gw_factor`` concept described in the paper (the
  ``Gaussian width factor`` control), where the Gaussian width is based on the
  pseudopotential core radius divided by the chosen scaling factor.
- Stationary points are identified through gradients and Hessians of the total
  electrostatic potential.
- The Hessian eigenvalues distinguish minima from saddles and maxima and are
  also used when estimating a zero-point contribution for locally stable minima.
- MuFinder evaluates that harmonic zero-point term separately from the random
  search itself when you click ``Calculate ZPE`` in the search controls, so the
  optimized sites can be inspected first and quantum corrections can be added
  afterwards.

The AIRSS paper and the ``pymuonsuite`` supplementary clustering material are
also helpful context for how to interpret repeated random starts after
relaxation. MuFinder does not embed the full AIRSS DFT-relaxation workflow
inside the UEP clustering window, but it uses the same broad post-processing
ideas: repeated random starts, clustering of relaxed endpoints, and selection
of representative low-energy sites for follow-up work.

Zero-point energy model
~~~~~~~~~~~~~~~~~~~~~~~

For a locally stable optimized muon site, MuFinder estimates the muon
zero-point energy with the same harmonic expression used in the
``pymuon-suite`` UEP implementation:

.. math::

   E_{\mathrm{zp}} = \frac{1}{2}\sum_i \hbar \omega_i,
   \qquad
   \omega_i = \sqrt{\lambda_i / m_\mu}.

Here :math:`\lambda_i` are the eigenvalues of the total-potential Hessian at the
optimized site after converting from eV/Å² into SI units, and :math:`m_\mu` is
the muon mass. This is only reported for stable minima. If any Hessian
eigenvalue is negative, the optimized point is a saddle or otherwise unstable
extremum, so MuFinder stores the classical energy and reports the ZPE as
unavailable.

Symmetry and Wyckoff positions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The paper emphasizes that symmetry is often a useful first filter for plausible
muon stopping sites:

- special Wyckoff positions with no free coordinates are stationary points of
  any scalar field that respects the crystal symmetry, including the
  electrostatic potential;
- this makes high-symmetry empty interstitial positions especially natural UEP
  candidates;
- symmetry alone does *not* guarantee that a stationary point is a minimum, so
  MuFinder complements symmetry intuition with gradients, Hessians, and direct
  optimization of candidate positions.

MuFinder therefore uses random-start batch searches, which optimize each start
and classify the resulting stationary point from its Hessian, so you can
inspect local stability rather than relying on symmetry arguments alone.

References
~~~~~~~~~~

- Simone Sturniolo and Leandro Liborio, ``Computational prediction of muon
  stopping sites: A novel take on the unperturbed electrostatic potential
  method``, *J. Chem. Phys.* **153**, 044111 (2020),
  DOI ``10.1063/5.0012381``.
- Leandro Liborio, Simone Sturniolo, and Dominik Jochym, ``Computational
  prediction of muon stopping sites using ab initio random structure searching
  (AIRSS)``, *J. Chem. Phys.* **148**, 134114 (2018),
  DOI ``10.1063/1.5024450``.

For developers
--------------

The saved UEP analysis state leaves out the full-cell potential grid, which is
stored once as the job's grid result, and keeps each isosurface mesh once,
serialized only when the mesh changes. The package layers and the shared
workflow infrastructure are described in :doc:`/architecture`.
