"""Workflow expansion contract.

The workflow layer expands resolved scientific inputs into
:class:`~mufinder.workflow.types.JobSpec` objects. Each workflow is exposed as
a keyword-only module-level ``expand_*`` function rather than a class:

- :func:`mufinder.workflow.convergence.expand_convergence_sweep_jobs`
- :func:`mufinder.workflow.muon_site.expand_muon_site_jobs`
- :func:`mufinder.workflow.relaxation.expand_relaxation_jobs`
- :func:`mufinder.workflow.energy_barriers.expand_nudged_elastic_band_jobs`
- :func:`mufinder.workflow.quantum.expand_quantum_phonon_jobs`

Contract for an ``expand_*`` function:

- It is keyword-only and returns ``list[JobSpec]``.
- It accepts the scientific inputs (structure(s), sweep values, job configs,
  etc.) plus layered ``InputTemplate`` parameters. The template layers
  (``global_defaults``, ``workflow_defaults``, ``structure_defaults`` where
  supported, per-job patches, and ``user_overrides``) are merged in
  :data:`mufinder.workflow.merge.MERGE_PRECEDENCE_ORDER` by
  :func:`mufinder.workflow.merge.resolve_job`.
- Product-style expansion across multiple sweep axes is intentionally deferred;
  current implementations use one-axis map expansion.

This module is documentation only and defines no runtime API. The former
``Workflow`` Protocol was retired when the two remaining class-based workflows
became thin deprecated wrappers over their ``expand_*`` functions; see the
compatibility ledger in ``docs/compatibility_policy.rst``.
"""

from __future__ import annotations
