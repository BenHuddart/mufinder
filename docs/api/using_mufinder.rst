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

The three examples below are complete scripts, kept in ``docs/api/examples``
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
