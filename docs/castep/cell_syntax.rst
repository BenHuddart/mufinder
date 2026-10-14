CASTEP ``.cell`` Syntax
=======================

MuFinder reads and writes a conservative subset of CASTEP ``.cell`` syntax. The
contributor-facing source of truth for that subset is
:download:`keyword_catalog.yaml <keyword_catalog.yaml>`
(``docs/castep/keyword_catalog.yaml``). A byte-identical copy ships as
``src/mufinder/assets/keyword_catalog.yaml`` and serves as the starter catalog;
``tests/castep_io/test_schema.py`` checks that the two agree. How the runtime
schema is built from a discovered ``castep_keywords.json``, and when the
starter catalog is used instead, is described in
:doc:`../castep_keyword_catalog`.

Formatting conventions
----------------------

``mufinder.castep_io.cell_writer.write_cell`` validates a ``CellFile`` and then
renders it as follows:

- Keywords and block names are uppercase.
- Blocks are delimited by ``%BLOCK NAME`` and ``%ENDBLOCK NAME``. A block whose
  structured input carries a unit renders it as the first line inside the
  block; no unit line is written otherwise.

  .. code-block:: text

     %BLOCK NAME
     ...
     %ENDBLOCK NAME

- Sections follow the catalog's fixed serialization order, then any further
  blocks, then any further scalars, separated by a single blank line.
- Scalar keywords are whitespace-separated tokens, for example
  ``KPOINT_MP_GRID 4 4 4``. A flag set to true renders as the bare keyword
  (``SYMMETRY_GENERATE``); a boolean value renders as ``KEYWORD : TRUE`` or
  ``KEYWORD : FALSE``.
- Integers are written as is and floats with up to 12 significant figures.
- ``POSITIONS_FRAC`` rows carry a ``SPIN=<value>`` token when the structure has
  a non-zero spin for that atom.

Validation and rendering
------------------------

- ``mufinder.castep_io.schema`` discovers keyword metadata and defines the
  supported subset.
- ``mufinder.castep_io.validators`` performs structure, type, exclusivity and
  task-aware checks.
- ``mufinder.castep_io.cell_writer`` only serializes already-validated objects.

Support is extended in that order: the catalog and schema first, then validator
logic, then writer changes when a keyword needs its own formatting.

Parsing
-------

``mufinder.castep_io.cell_parser`` reads ``.cell`` and ``-out.cell`` text through
``parse_cell_text`` and ``parse_cell_file``; it is the only ``.cell`` reader in
the package.

- Comments (``!``, ``#``) are stripped and block names are case-insensitive.
- An optional leading length-unit line (``ang``, ``bohr``, ``nm``, ``pm``,
  ``cm``, ``m``) on ``LATTICE_CART``, ``LATTICE_ABC`` and ``POSITIONS_ABS``
  converts lengths to Angstrom. Any other single non-numeric token in that
  position is rejected as an unsupported length unit.
- ``POSITIONS_ABS`` rows are converted to fractional coordinates wrapped into
  ``[0, 1)``; ``POSITIONS_FRAC`` coordinates are kept as written. A wrapped
  coordinate within ``1e-12`` below 1.0 is an atom on a cell face and is placed
  at 0.0, so a face atom of a non-orthogonal lattice reads the same on every
  platform. Species labels such as ``H:mu`` are kept verbatim.
- ``SPIN=<value>`` tokens are read from ``POSITIONS_FRAC`` rows, including the
  padded ``SPIN=  1.000`` form CASTEP writes. Rows without a token get spin 0,
  an empty ``SPIN=`` marker counts as no token, and ``atomic_spins`` is ``None``
  unless at least one row carries a value.
- Both lattice blocks, both position blocks, a repeated geometry block, and an
  unterminated or mismatched block are errors.
- A full parse validates the whole file against the loaded keyword catalog, so
  a keyword outside it raises ``CellParseError``. With ``geometry_only=True``
  only the lattice and position blocks are read and validated, and every other
  keyword and block is skipped. Structure import
  (``mufinder.science.structure_io.load_structure_file``) uses this mode, so a
  CASTEP ``-out.cell`` imports regardless of the calculation keywords it
  carries.

Starter subset
--------------

The ``.cell`` entries in the starter catalog are:

- Lattice blocks: ``LATTICE_CART``, ``LATTICE_ABC``.
- Position blocks: ``POSITIONS_FRAC``, ``POSITIONS_ABS``, and the
  transition-state endpoint blocks ``POSITIONS_FRAC_PRODUCT``,
  ``POSITIONS_FRAC_INTERMEDIATE``, ``POSITIONS_ABS_PRODUCT``,
  ``POSITIONS_ABS_INTERMEDIATE``.
- Species and k-point blocks: ``SPECIES_POT``, ``SPECIES_MASS``,
  ``SPECIES_GAMMA``, ``KPOINTS_LIST``, ``HUBBARD_U``.
- Constraint block: ``CELL_CONSTRAINTS``.
- Scalars and flags: ``KPOINT_MP_GRID``, ``KPOINT_MP_SPACING``,
  ``PHONON_KPOINT_MP_GRID``, ``SYMMETRY_GENERATE``, ``SYMMETRY_TOL``,
  ``FIX_ALL_CELL``, ``FIX_COM``, and the opaque text-valued ``UNIT_CELL``.
- Spin decoration (structure-driven): the optional inline ``SPIN=<value>``
  token on ``POSITIONS_FRAC`` rows for spin-polarized workflows.
- DFT+U setup: ``HUBBARD_U`` is a ``.cell`` block with an optional ``eV`` or
  ``Ha`` unit line, edited through guided rows that target all atoms of a
  species for one orbital, such as ``Ni d: 2.4``.

Anything outside this subset is unsupported until it is added to the catalog,
the schema and the tests. The mapping between ``KPOINT_MP_SPACING`` and
``KPOINT_MP_GRID`` is described in :doc:`kpoint_spacing_convention`.

Example
-------

``write_cell`` renders a two-atom antiferromagnetic cell with a k-point grid and
symmetry generation as:

.. code-block:: text

   %BLOCK LATTICE_CART
   ang
   5.43 0 0
   0 5.43 0
   0 0 5.43
   %ENDBLOCK LATTICE_CART

   %BLOCK POSITIONS_FRAC
   Fe 0 0 0 SPIN=2
   Fe 0.5 0.5 0.5 SPIN=-2
   %ENDBLOCK POSITIONS_FRAC

   SYMMETRY_GENERATE

   KPOINT_MP_GRID 4 4 4
