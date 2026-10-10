<!--
Title: use a Conventional Commits prefix (feat:, fix:, docs:, refactor:,
chore:). python-semantic-release derives version bumps and the CHANGELOG from
merged commit messages, and the squash-merge title becomes that message.

Agents: `gh pr create --body` does NOT apply this template automatically —
GitHub only injects it into the web form. Read this file and follow it when
writing the body.

For small fixes, Why / What changed / Gates is enough — delete empty sections
rather than padding them.
-->

## Why

<!-- The problem or motivation, with evidence (an observed failure, measured
     numbers, behaviour verified against a real CASTEP run). Link related
     PRs. -->

## What changed

<!-- Group by area with bold lead-ins (**Dialog** — …, **Dispatcher** — …),
     not a file list. Say what each piece does, not which lines moved. -->

## Reviewer notes

<!-- Things a reviewer should weigh, not just read:
     - deliberate design decisions and the alternatives rejected
     - deviations from the original brief, and why
     - known limitations (documented, not fixed)
     - adjacent problems noticed but deliberately NOT touched -->

## Verification

<!-- What the new tests pin, not just that tests exist. Call out end-to-end
     checks, invariants under test, and anything verified against real CASTEP
     output rather than fixtures. -->

## Gates

- `make test` — <N> passed
- Coverage <X>% (≥ 85%)
- `make docs` — clean
- Docs updated alongside behaviour (the change is incomplete otherwise)
- No new `_on_submit_*` / `_materialize_*` handlers on `MainWindow` (harness ratchet)
- `docs/compatibility_policy.rst` checked for any renamed/removed persisted name
