# Contributing to MuFinder

Thank you for your interest in MuFinder. A small group of collaborators
develops it, and this page explains how you can help.

## Report bugs and request features

Everyone is welcome to open an issue, and we encourage you to. Bug reports and
feature requests are the most useful contributions you can make right now.

- **Bug reports:** use the *Bug report* form. Tell us which MuFinder version
  and operating system you used, what you did, what you expected and what
  happened instead. Attach logs, CASTEP output or a small project that
  reproduces the problem where you can, and say whether jobs ran locally or on
  a cluster (and which scheduler).
- **Feature requests:** use the *Feature request* form. Describe the scientific
  or practical problem you are trying to solve, not only the feature you have
  in mind.

Please search the [existing issues](https://github.com/BenHuddart/mufinder/issues)
first. If you have investigated an open issue and found the cause, add what
you found as a comment on that issue.

## Pull requests

**At present we only accept pull requests from repository collaborators.**
Other accounts cannot open pull requests on this repository, and any pull
request from outside the collaborator group is closed without review. This is
not a judgement on the change. It keeps review load manageable and the
provenance of the code clear while the project is small. The policy may change
in the future, and this page will say so when it does.

If you would like to fix a bug you have found, please open an issue describing
it rather than sending a patch.

## Becoming a collaborator

If you would like to work on MuFinder as a collaborator, email Ben Huddart at
<benjamin.huddart@physics.ox.ac.uk> with a short note about your background and what you
would like to work on.

## For collaborators

- Branch `feature/*` or `fix/*` off `main` and open a pull request directly into
  `main`. Never commit to `main`.
- Give the pull request a Conventional Commits title (`feat:`, `fix:`, `docs:`,
  `refactor:`, `chore:`) and write the body from
  [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).
- Change implementation, tests and docs together, and run the gates before
  asking for review. The `Makefile` defines them; see
  [Quality and Automation](https://benhuddart.github.io/mufinder/quality_and_automation.html).
- [`AGENTS.md`](AGENTS.md) holds the binding project rules (package layers,
  shared homes, CASTEP rules, tests and docs). It applies to people as well as
  coding agents.

## Licence

MuFinder is released under the GNU General Public License, version 3 or (at
your option) any later version. By contributing, you agree that your
contribution is licensed under the same terms. See [LICENSE](LICENSE).
