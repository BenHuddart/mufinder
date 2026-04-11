HPC Connectivity Security Audit
===============================

Date
----
2026-04-07

Scope
-----
Remote cluster setup, SSH/SFTP transport, scheduler submission and polling,
remote directory handling, and cluster-backed CASTEP keyword generation.

Findings
--------
- ``P1`` Unknown SSH host keys were previously auto-trusted via Paramiko
  ``AutoAddPolicy``. Patch decision: replaced with TOFU host-key prompts plus
  persisted user-scoped trust records and hard failure on host-key mismatch.
- ``P1/P2`` Cluster job labels were used directly in remote paths and scheduler
  commands. Patch decision: cluster submission now blocks labels outside
  ``[A-Za-z0-9._-]`` and the cluster runner validates seed names defensively.
- ``P2`` Recursive remote deletion previously shelled out to ``rm -rf``. Patch
  decision: replaced with recursive SFTP deletion.
- ``P2/P3`` Cluster connection, refresh, submission, cancellation, and polling
  failures often surfaced terse or generic GUI messages. Patch decision:
  cluster-facing warnings now use concise summaries with expandable details and
  recovery advice.
- ``P3`` Queue polling transport failures could silently degrade the connection
  state. Patch decision: polling now emits an explicit user-facing failure
  reason in addition to updating the connection indicator.

Deferred Items
--------------
- None for the straightforward hardening tasks in this pass.
