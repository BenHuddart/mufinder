CASTEP ``.param`` Syntax
========================

MuFinder writes a small, explicit subset of CASTEP ``.param`` keywords. The
contributor-facing source of truth for that subset is
:download:`keyword_catalog.yaml <keyword_catalog.yaml>`
(``docs/castep/keyword_catalog.yaml``), mirrored as the packaged starter catalog
``src/mufinder/assets/keyword_catalog.yaml``. How the runtime schema is built
from a discovered ``castep_keywords.json``, and when the starter catalog is used
instead, is described in :doc:`/castep_keyword_catalog`.

Formatting conventions
----------------------

``mufinder.castep_io.param_writer.write_param`` validates a ``ParamFile`` and
then renders it as follows:

- Keyword names are uppercase; values are written as given.
- Scalar assignments are ``KEYWORD : value``, one per line, in the catalog's
  fixed serialization order, followed by any keywords the order does not list.
- Booleans are ``TRUE`` or ``FALSE``. Integers are written as is and floats
  with up to 12 significant figures. A ``None`` value is an error.
- An explicit unit is kept after the value, as in ``CUT_OFF_ENERGY : 340 eV``.
- Blocks such as ``XC_DEFINITION`` and ``DEVEL_CODE`` follow the scalars after
  a blank line, delimited by ``%BLOCK NAME`` and ``%ENDBLOCK NAME`` and
  separated from each other by a blank line.

Validation and rendering
------------------------

- ``mufinder.castep_io.schema`` discovers keyword metadata and defines the
  supported subset.
- ``mufinder.castep_io.validators`` checks types, mutually exclusive keywords,
  missing requirements, and a small set of task-specific rules backed by local
  CASTEP evidence.
- ``mufinder.castep_io.param_writer`` handles deterministic text rendering only.

A new keyword is added by extending the catalog and schema, then the validator,
with small, focused changes rather than logic in the writer.

Starter subset
--------------

The ``.param`` entries in the starter catalog are:

- Task, run control and XC settings: ``TASK``, ``CONTINUATION``,
  ``XC_FUNCTIONAL``, ``XC_DEFINITION`` (block), ``BASIS_PRECISION``,
  ``NUM_FARMS``, ``IPRINT``, ``LENGTH_UNIT``, ``DEVEL_CODE`` (block).
- Electronic controls: ``CUT_OFF_ENERGY``, ``FINE_CUT_OFF_ENERGY``,
  ``MAX_SCF_CYCLES``, ``ELEC_ENERGY_TOL``, ``NONSCF_EIGENVALUE_TOL``,
  ``SMEARING_WIDTH``, ``NEXTRA_BANDS``, ``CHARGE``.
- Spin settings: ``SPIN_POLARIZED``, ``SPIN_POLARISED``, ``SPIN``.
- Geometry optimization settings: ``GEOM_ENERGY_TOL``, ``GEOM_FORCE_TOL``,
  ``GEOM_STRESS_TOL``, ``GEOM_DISP_TOL``, ``GEOM_MAX_ITER``,
  ``GEOM_FREQUENCY_EST``.
- Transition-state search and NEB settings: ``TSSEARCH_METHOD``,
  ``TSSEARCH_ENERGY_TOL``, ``TSSEARCH_FORCE_TOL``, ``TSSEARCH_DISP_TOL``,
  ``TSSEARCH_LST_MAX_ITER``, ``TSSEARCH_QST_MAX_ITER``,
  ``TSSEARCH_MAX_PATH_POINTS``, ``TSSEARCH_NEB_MAX_ITER``,
  ``TSSEARCH_NEB_SPRING_CONSTANT``, ``TSSEARCH_NEB_METHOD``,
  ``TSSEARCH_NEB_TANGENT_MODE``, ``TSSEARCH_NEB_CLIMBING``,
  ``TSSEARCH_NEB_NORMED``.
- Phonon settings: ``PHONON_METHOD``.
- Magnetic-resonance settings: ``MAGRES_TASK`` (``HYPERFINE`` only).
- Output and property flags: ``WRITE_CELL_STRUCTURE``,
  ``WRITE_FORMATTED_DENSITY``, ``CALCULATE_STRESS``.
- Molecular dynamics settings: ``MD_DELTA_T``, ``MD_NUM_ITER``,
  ``MD_TEMPERATURE``.

On the starter catalog, a ``.param`` file read with ``parse_param_text`` keeps
a keyword outside this subset, such as ``CALCULATE_POLARIZATION``, as an
unmapped observation: it is reported, not carried into the parsed
``ParamFile``, and validating or writing a ``ParamFile`` that holds it raises
``UnknownKeywordError`` pointing at ``Setup > CASTEP``.

Example
-------

``write_param`` renders a geometry optimization as:

.. code-block:: text

   TASK : GEOMETRYOPTIMIZATION
   XC_FUNCTIONAL : PBE
   CUT_OFF_ENERGY : 340 eV
   MAX_SCF_CYCLES : 60
   GEOM_MAX_ITER : 30
   GEOM_FORCE_TOL : 0.05 eV/ang
