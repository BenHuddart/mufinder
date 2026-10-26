Using MuFinder as a library
===========================

Everything the application computes is reachable from Python without opening
the GUI. Install the package from a source checkout:

.. code-block:: bash

   python -m pip install .
   # or, in a development checkout: uv sync

The library packages (``mufinder.domain``, ``mufinder.castep_io``,
``mufinder.persistence``, ``mufinder.science``, ``mufinder.workflow`` and
``mufinder.services``) never import Qt widgets, so a script needs no display
and no ``QApplication``. :doc:`/api/index` lists what each package owns, and
:doc:`/architecture` gives the layer rule: dependencies flow inward, so a
script can call any of them.

The four examples below are complete scripts, kept in ``docs/api/examples``
and run by ``tests/docs/test_api_examples.py``. Run each from the repository
root, for example ``python docs/api/examples/parse_castep_output.py``.

Parse a CASTEP output
---------------------

:func:`mufinder.castep_io.output_parser.parse_castep_output` reads a
``.castep`` file and returns a frozen
:class:`~mufinder.castep_io.output_parser.CastepResult`. It is the parser the
local and cluster runners use when a job finishes.

.. literalinclude:: examples/parse_castep_output.py
   :language: python

``total_energy`` and ``enthalpy`` are in eV and come from the last value in
the file; ``forces`` holds one ``(species, fx, fy, fz)`` row per atom in eV/Å.
``completed`` is ``True`` once CASTEP has written its closing timing summary,
and ``geom_converged`` is ``True`` or ``False`` when the last run reported
geometry-optimisation success or failure, and ``None`` when it reported
neither. For the
silicon BFGS fixture the script prints, among other lines:

.. code-block:: text

   Final energy: -340.682656 eV
   Geometry converged: True

Generate candidate muon sites
-----------------------------

Site generation works on a project document, because generated sites are
:class:`~mufinder.domain.models.MuonSite` records labelled against the sites a
project already holds. The script builds conventional silicon as a
:class:`~mufinder.domain.models.Structure`, makes a throwaway project in a
temporary directory with :class:`~mufinder.services.project_service.ProjectService`,
and calls :meth:`~mufinder.services.project_service.ProjectService.generate_candidate_sites`,
the call the muon-generation page makes.

.. literalinclude:: examples/generate_muon_sites.py
   :language: python

``method="random"`` with ``selection_mode="count"`` draws ``count`` sites at
least ``min_muon_atom`` Å from every host atom and at least a muon-muon
separation estimated from the cell's free volume for that count, which is
shrunk until all ``count`` sites fit. With ``include_symmetry_equivalents`` the
separation is checked against every symmetry image of each accepted site. ``seed`` fixes the
draw, so the same script prints the same sites every time; the GUI leaves it
unset. The call returns the new ``MuonSite`` records, whose
``initial_fractional_position`` is the generated position, and appends a
provenance record to the project directory, which is why the project needs a
real (here temporary) root. With seed 42 the script prints five sites,
starting:

.. code-block:: text

   Generated 5 sites
     Random Site 1  0.6394 0.0250 0.2750

The other methods take the parameters listed in
:func:`mufinder.services.site_generation_service.generate_candidate_sites`.
The placement algorithms themselves, which take plain lattice vectors and
positions, are in :mod:`mufinder.science.site_generation`.

Compute the field at a muon site
--------------------------------

This example needs the ``scientific-backends`` extra, which installs
``muesr`` and ``muLFC``:

.. code-block:: bash

   python -m pip install ".[scientific-backends]"
   # or: uv sync --extra scientific-backends

The script builds the conventional bcc cell of iron, ``a = 2.868 Å`` with two
magnetic ions, as a 2.2 μB ferromagnet along ``c`` built with
:func:`~mufinder.science.magnetic_structure.build_collinear_component` and
:class:`~mufinder.science.magnetic_structure.AtomSelector`, and passes both to
:func:`~mufinder.science.magnetic_fields.calculate_commensurate_fields`, the
call the Magnetic Fields page makes for a commensurate structure.

.. literalinclude:: examples/dipolar_field.py
   :language: python

The function returns one
:class:`~mufinder.science.magnetic_fields.MagneticFieldResult` per muon site,
with the dipolar, Lorentz, contact, demagnetizing and applied terms as
Cartesian vectors in tesla, and their sum as ``total``. With no contact
coupling and no demagnetizing field, the defaults of
:class:`~mufinder.science.magnetic_fields.MagneticFieldOptions`, the
octahedral site reproduces the ferromagnet in :ref:`magnetic-fields-worked-examples`:

.. code-block:: text

   Muon site: (0.5, 0.0, 0.0)
   Dipolar Bz: -0.9261 T
   Lorentz Bz: 0.7243 T
   Total Bz: -0.2018 T
   |B|: 0.2018 T

Without ``muesr`` installed the call raises
:class:`~mufinder.science.magnetic_fields.MagneticFieldDependencyError`.

Simulate the muon polarisation
------------------------------

:mod:`mufinder.science.polarisation` turns field results into the muon-spin
polarisation ``P(t)`` that a μSR experiment measures, and into its Fourier
spectrum. A muon spin that starts along ``p̂`` in a static field ``B``, with
``n̂ = B/|B|`` and ``ω = γ_μ|B|``, precesses about ``n̂``
(``dS/dt = γ_μ S × B``). Seen along a detector axis ``d̂`` it gives

.. math::

   P_d(t) = (\hat d\cdot\hat n)(\hat p\cdot\hat n)\, e^{-\lambda_z t}
          + \bigl\{[\hat d\cdot\hat p - (\hat d\cdot\hat n)(\hat p\cdot\hat n)] \cos\omega t
          + \hat d\cdot(\hat p\times\hat n)\,\sin\omega t\bigr\}\, e^{-\sigma^2 t^2/2}.

With ``d̂ = p̂`` this is the familiar
``cos²θ e^{−λt} + sin²θ e^{−σ²t²/2} cos ωt`` used in the zero-field analysis of
MnSi by Dalmas de Réotier et al., Phys. Rev. B **93**, 144419 (2016), Eq. 14;
the sine term covers spin-rotated geometries. ``λ_z`` (μs⁻¹) is a
phenomenological longitudinal rate. ``σ`` (μs⁻¹) is the nuclear width along the
local field: only the nuclear field components along ``B`` shift the
precession frequency, so the width is the transverse-field second moment with
the field along ``n̂``, not the zero-field Kubo–Toyabe ``Δ``, whose variance is
2.5 times smaller in a polycrystal.

:func:`~mufinder.science.polarisation.calculate_polarisation` takes one
:class:`~mufinder.science.magnetic_fields.MagneticFieldResult` or
:class:`~mufinder.science.magnetic_fields.MagneticFieldDistributionResult`
per site, or a list of them, and returns a
:class:`~mufinder.science.polarisation.PolarisationResult` with one trace per
site and their weighted mean (equal weights by default, which is right for
equivalent sites and approximant translations). The rules it follows:

- **Directions.** ``polarisation_direction`` and ``detector_direction`` (default:
  the same) are laboratory directions; ``crystal_rotation`` takes the
  crystal-frame Cartesian axes of the fields to the laboratory.
- **Applied field.** The applied field is part of the field result, set through
  :class:`~mufinder.science.magnetic_fields.MagneticFieldOptions`. For results
  calculated without one, ``applied_field`` adds it, so one zero-field
  calculation serves a longitudinal-field scan; the call refuses
  ``applied_field`` when a result already carries an applied field.
  :func:`~mufinder.science.polarisation.calculate_lf_polarisation` puts the spin
  and detector along the applied field (``P_z``),
  :func:`~mufinder.science.polarisation.calculate_tf_polarisation` across it
  (``P_x``).
- **Nuclear width.** ``sigma_nuclear`` is one width for every sample, or
  ``"auto"`` to damp each sample with σ along its own field from
  :class:`~mufinder.science.nuclear_relaxation.NuclearWidthTensors`.
  :func:`~mufinder.science.nuclear_relaxation.calculate_nuclear_width_tensors`
  sums the moment tensors ``S = Σ w/r⁶``, ``M_ij = Σ w r̂_i r̂_j/r⁶`` and
  ``M_ijkl = Σ w r̂_i r̂_j r̂_k r̂_l/r⁶`` once, after which the transverse-field
  width ``σ²(n̂) ∝ 9 n_i n_j n_k n_l M_ijkl − 6 n_i n_j M_ij + S`` and the
  zero-field widths follow for any direction without another lattice sum.
- **Powder.** ``polycrystalline=True`` averages over grain orientations. In zero
  applied field the average is exact: ``(d̂·p̂)[⅓ e^{−λt} + ⅔ e^{−σ²t²/2} cos ωt]``
  per sample. In an applied field the internal field turns with the grain while
  the applied field does not, so the applied field's direction in the crystal
  runs over a deterministic Fibonacci sphere (``sphere_points``, 600 by
  default), and the rotation of the instrument about the applied field is
  averaged exactly. Equivalent sites in the result already make the domain
  average.
- **Cost.** ``P(t)`` depends on the samples only through histograms of ``|B|``,
  and, with ``"auto"``, of the nuclear width as well, with second-order
  corrections for each sample's offset from its bin, so a distribution costs ``O(N_samples)`` per orientation plus
  ``O(N_bins × N_times)``, never ``O(N_samples × N_times)``. There are at least
  ``magnitude_bins`` bins, more when a long time window needs them. A
  200 000-sample distribution on 1000 times takes a fraction of a second as a
  single crystal and a few seconds as a powder in an applied field.

:func:`~mufinder.science.polarisation.calculate_spectrum` subtracts the
baseline (the mean of the last 10 % of points), applies optional Gaussian and
exponential windows from the first time, zero-pads and Fourier-transforms the
trace. It returns the frequency axis in MHz, the equivalent field axis
``f / (γ_μ/2π)`` in tesla, the magnitude spectrum as a density per MHz whose
integral is the oscillating fraction, and the complex transform. For a simulated
trace the spectrum is the field histogram convolved with the window; its use is
to process the simulation the same way as the measured data.

This example needs the ``scientific-backends`` extra, like the field example.
It computes the field at the three octahedral sites of the bcc iron
ferromagnet above, takes the nuclear width along each site's field, and
evaluates the zero-field powder polarisation, its spectrum and the powder in a
0.5 T longitudinal field:

.. literalinclude:: examples/polarisation.py
   :language: python

Two sites lie across the moment and see 0.2018 T; the third lies on the moment
axis and sees 2.58 T. The spectrum's strongest line is the pair of sites at
0.2018 T, ``γ_μ|B|/2π = 27.35 MHz``:

.. code-block:: text

   |B| at each site: 0.2018 T, 0.2018 T, 2.5765 T
   Zero-field P(0): 1.000
   Spectrum peak: 27.36 MHz, 0.202 T
   P_z in 0.5 T, mean over the last 0.5 us: 0.71

:func:`~mufinder.services.magnetic_field_results.store_polarisation_result`
keeps a result in a project as a ``muon_polarisation`` record owned by the
Magnetic Fields page or a saved workspace, like the field and tensor results.

