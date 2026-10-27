Polarisation
============

The ``Polarisation`` page turns the local fields of :doc:`magnetic_fields` into
what a μSR experiment measures: the muon-spin polarisation :math:`P(t)` and its
Fourier spectrum. A simulation can combine the field results of the Magnetic
Fields page and of any saved magnetic-field workspace, each weighted, so a
signal from several inequivalent muon sites is one simulation. Every simulation
is kept in the project.

Before you start
----------------

Calculate fields on the Magnetic Fields page, or in a saved workspace, with
``Calculate Dipolar Field``. Each page and workspace holds one field result: the
field at its muon site, or at the site and its symmetry equivalents when
``Include equivalent sites`` is on. A commensurate structure gives one field per
site; an incommensurate one gives a field distribution per site.

Simulate the polarisation
-------------------------

1. **Open the page.** Click ``Polarisation →`` on the Magnetic Fields page or on
   a saved workspace, which opens the page with that field result ticked, or
   select ``Polarisation`` in the project navigator.
2. **Choose the field results.** The ``Field Results`` table lists each stored
   result with its applied field. Tick the ones to simulate together and give
   each a weight, for example the relative occupation of two inequivalent sites.
   A result's weight is shared equally among the equivalent sites it holds.
3. **Set the geometry and sample** (see `Settings`_).
4. **Click** ``Add Simulation``. The simulation is added to the ``Simulations``
   table and drawn: :math:`P(t)` of each site and their weighted average above,
   and the spectrum of the average below, with the frequency along the bottom
   and the field :math:`B = f/(\gamma_\mu/2\pi)` along the top.

.. figure:: ../_generated/screenshots/mnsi_polarisation.webp
   :alt: The Polarisation page with the four MnSi helix distributions ticked, a
         zero-field polycrystal simulation over 16 microseconds and its
         spectrum, with lines near 12 MHz and 28 MHz.

   The four sites of the helix in :doc:`/user_guide/examples/mnsi` as a
   polycrystal in zero field. The spectrum has the two edges of the
   field distribution, at 88–90 mT and 207 mT.

Selecting a simulation in the table draws it and loads its settings and field
results into the form. ``Recalculate`` runs the selected simulation again with
the form's settings and the current fields; ``Add Simulation`` keeps it and adds
another. ``Remove...``, or Delete or Backspace in the table, removes the
selected simulation after asking. ``Export Data...`` writes it to a text file
(see `Exported data`_). A long calculation shows a busy dialog that can cancel
it.

Settings
--------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Setting
     - Meaning
   * - ``Geometry``
     - ``Zero field``: the initial polarisation and the detector are the
       directions given, in the laboratory frame; the detector is along the
       spin unless ``Along the spin`` is unticked. ``Longitudinal field``: spin
       and detector along the applied field (:math:`P_z`). ``Transverse field``:
       spin and detector across the applied field (:math:`P_x`). The zero-field
       geometry needs field results without an applied field; the other two need
       one.
   * - ``Crystal [uvw] along z`` / ``along x``
     - The orientation of a single crystal: the lattice direction along the
       laboratory :math:`z` axis (the beam) and the one along :math:`x`, made
       perpendicular to the first. The defaults, ``[001]`` and ``[100]``, keep
       the crystal's Cartesian frame for a cell with ``a`` along :math:`x` and
       ``c`` along :math:`z`.
   * - ``Applied field`` / ``Add applied field (T)``
     - The applied field comes with the field results (the ``Applied field`` row
       of the Calculation card). When every ticked result is a zero-field one,
       an applied field in tesla, in the crystal frame, can be added here
       instead, so one zero-field calculation serves a longitudinal- or
       transverse-field simulation. Results with different applied fields
       cannot be simulated together.
   * - ``Sample``
     - ``Single crystal`` or ``Polycrystal``. In zero field the polycrystal
       average is exact. In an applied field the field's direction in the
       crystal runs over ``Sphere points`` orientations, and the rotation of
       the instrument about the field is averaged exactly.
   * - λ\ :sub:`z`
     - A rate, in μs⁻¹, damping the non-oscillating part
       :math:`e^{-\lambda_z t}`.
   * - ``σ``
     - A Gaussian width, in μs⁻¹, damping the oscillating part
       :math:`e^{-\sigma^2 t^2/2}`. ``From nuclear widths`` instead sums the
       nuclear dipolar widths of the host at each muon position (all elements,
       natural abundance) and damps each field with the transverse-field width
       along it.
   * - ``End time`` / ``Points``
     - The time grid, from zero. A warning is shown when the largest field
       precesses faster than the grid resolves.
   * - ``Collision rate ν``
     - Strong-collision dynamics (see `Dynamics`_); 0 is static. ``Exact
       recursion`` is the default; ``FFT`` suits very long grids.
   * - ``Spectrum``
     - ``Subtract baseline`` removes the mean of the last 10 % of the trace;
       ``Apodisation`` multiplies it by a Gaussian window of time σ and an
       exponential window of time λ, each when given; ``Zero padding``
       lengthens it before the transform.

Dynamics
--------

With a collision rate ν the whole local field is redrawn from the static field
distribution at random times, on average ν times a microsecond, without memory
of the field before. This strong-collision model describes a muon hopping
between sites and fast fluctuations of the moments. A hop redraws from every
site, so the dynamics act on the weighted average: the plot shows the static
average dashed and the dynamic one solid.

.. figure:: ../_generated/screenshots/mnsi_polarisation_dynamics.webp
   :alt: The Polarisation page with the MnSi zero-field polycrystal simulated
         with a collision rate of 2 per microsecond: the dynamic trace relaxes
         to zero within about 3 microseconds and the spectrum's lines broaden.

   The same simulation with ν = 2 μs⁻¹. The oscillation and the 1/3 tail both
   relax, and the lines of the spectrum broaden.

The model and its numerical methods are described in
:doc:`/api/using_mufinder`.

Stale simulations
-----------------

A simulation records which field results it used. When one of them is
recalculated on its page, or its workspace is deleted, the simulation is kept
and its ``Status`` reads ``Stale``; a note names the field results that
changed. ``Recalculate`` brings it up to date once every field result it uses
exists again. A simulation that uses a field result computed at an unrelaxed
UEP site is marked ``estimate``, with the same note as on the Magnetic Fields
page.

Exported data
-------------

``Export Data...`` writes a ``.dat`` or ``.txt`` file. Lines starting with
``#`` give the simulation's name, its field results and weights, every
setting, any warning and, for an estimate, the estimate note. Two blocks of
columns follow, separated by a blank line:

- the time domain: ``t_us``, one ``P_`` column per site, ``P_average``, and
  ``P_dynamic`` when there are dynamics;
- the spectrum: ``f_MHz``, ``B_T`` and ``amplitude_per_MHz``.

For developers
--------------

Each simulation is a ``Result`` of type ``muon_polarisation`` whose
``data["owner"]`` is ``polarisation:`` followed by the simulation id. It holds
``title``, ``settings``, ``sources`` (each with ``owner``, ``record_id`` of the
field result used, ``title`` and ``weight``), ``result`` (the static traces),
``dynamic``, ``warnings`` and ``site_source``. The spectrum is recomputed from
the traces. The page's form is saved in the UI state under
``polarisation_form``. See :mod:`mufinder.services.polarisation_simulations`.
