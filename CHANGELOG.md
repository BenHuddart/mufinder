# Changelog

<!-- version list -->

## v0.6.0 (2026-10-11)

### Bug Fixes

- Accept any positive convergence threshold and flag ones finer than CASTEP prints
  ([#107](https://github.com/BenHuddart/mufinder/pull/107),
  [`3ef0b15`](https://github.com/BenHuddart/mufinder/commit/3ef0b158ed5511a9c13d42532c83e79d53695830))

- Bound keyword queries and harden the generation fallbacks
  ([#87](https://github.com/BenHuddart/mufinder/pull/87),
  [`65b05fb`](https://github.com/BenHuddart/mufinder/commit/65b05fbfbbb50814471773b254e66e0946b08b0e))

- Cancel a removed k-point grid's queued or running job
  ([#160](https://github.com/BenHuddart/mufinder/pull/160),
  [`903d04a`](https://github.com/BenHuddart/mufinder/commit/903d04aaa788011335996018e0fc858098bc6a3b))

- Detect stale continuation outputs on the cluster's own clock
  ([#85](https://github.com/BenHuddart/mufinder/pull/85),
  [`4e821c7`](https://github.com/BenHuddart/mufinder/commit/4e821c78676d1b74fdd2a4e5844ca2a5299bd04f))

- Do not submit a cluster job whose remote directory check failed
  ([#159](https://github.com/BenHuddart/mufinder/pull/159),
  [`90ce894`](https://github.com/BenHuddart/mufinder/commit/90ce894fe2b09c0a36fb53a08d0154b4a22106e7))

- Drop the legacy-scan registry entry before deleting its thread
  ([#147](https://github.com/BenHuddart/mufinder/pull/147),
  [`4bcdf3e`](https://github.com/BenHuddart/mufinder/commit/4bcdf3e12633c1d2a0a0535c51cc57a8543e3db4))

- Drop the pinned overview plot when a project default is edited by hand
  ([#86](https://github.com/BenHuddart/mufinder/pull/86),
  [`3d192df`](https://github.com/BenHuddart/mufinder/commit/3d192df8c66fc3f546cebb9e62a22ecc312fd20c))

- Explain DFPT phonon-method refusals and ask before an unchecked DFPT run
  ([#108](https://github.com/BenHuddart/mufinder/pull/108),
  [`4debf32`](https://github.com/BenHuddart/mufinder/commit/4debf3233531abc0a7a452e8bfbd0f38a9308091))

- Floor force and stress thresholds at CASTEP's printed precision
  ([#107](https://github.com/BenHuddart/mufinder/pull/107),
  [`3ef0b15`](https://github.com/BenHuddart/mufinder/commit/3ef0b158ed5511a9c13d42532c83e79d53695830))

- Give a popped-out plot sole control of its canvas
  ([#101](https://github.com/BenHuddart/mufinder/pull/101),
  [`e11682a`](https://github.com/BenHuddart/mufinder/commit/e11682ab6a3da41feb2476992d2d43db6b38c2a8))

- Give every workflow table a working per-row "Reset Failed Jobs..." action
  ([#75](https://github.com/BenHuddart/mufinder/pull/75),
  [`f26ce41`](https://github.com/BenHuddart/mufinder/commit/f26ce41534a33e3a5bebd1c1c0cc2b75fdf00ed0))

- Keep dashboard jobs in creation order across reloads
  ([#94](https://github.com/BenHuddart/mufinder/pull/94),
  [`7988986`](https://github.com/BenHuddart/mufinder/commit/79889865c58937a5d06a37f13a3bc4e03142acd7))

- Keep dismissed failed jobs resubmittable from their workflow row
  ([#74](https://github.com/BenHuddart/mufinder/pull/74),
  [`0093ad3`](https://github.com/BenHuddart/mufinder/commit/0093ad3ded984cc2dcaa61aa08c84bde10ad3a5a))

- Keep parameters hidden by the search when saving the parameter dialog
  ([#93](https://github.com/BenHuddart/mufinder/pull/93),
  [`8df115e`](https://github.com/BenHuddart/mufinder/commit/8df115e63e7742375c850b1d16f1df67d3c776cc))

- Keep plot axis labels visible when analysis dialogs open small
  ([#97](https://github.com/BenHuddart/mufinder/pull/97),
  [`c68bba4`](https://github.com/BenHuddart/mufinder/commit/c68bba4af8cb9ab2b320ecf46adbdc33fa649403))

- Keep popped-out plots in front of modal dialogs
  ([#101](https://github.com/BenHuddart/mufinder/pull/101),
  [`e11682a`](https://github.com/BenHuddart/mufinder/commit/e11682ab6a3da41feb2476992d2d43db6b38c2a8))

- Keep the legacy-scan worker alive so the import actually runs
  ([#82](https://github.com/BenHuddart/mufinder/pull/82),
  [`29e9569`](https://github.com/BenHuddart/mufinder/commit/29e9569aa73af10e9cf815e2c796c5e57114ee8f))

- Keep the UEP clustering window responsive while dragging
  ([#100](https://github.com/BenHuddart/mufinder/pull/100),
  [`eefe913`](https://github.com/BenHuddart/mufinder/commit/eefe9132ccac039bb2dce5d01d264ffcc4f03ada))

- Let the structure viewer's controls wrap so plots beside it get room
  ([#98](https://github.com/BenHuddart/mufinder/pull/98),
  [`861aed3`](https://github.com/BenHuddart/mufinder/commit/861aed34a93c7dc8920f81633df614d397bd97f9))

- Make convergence recommendations and adopted k-point grids trustworthy
  ([#107](https://github.com/BenHuddart/mufinder/pull/107),
  [`3ef0b15`](https://github.com/BenHuddart/mufinder/commit/3ef0b158ed5511a9c13d42532c83e79d53695830))

- Make the NEB task-farm control the only source of NUM_FARMS
  ([#73](https://github.com/BenHuddart/mufinder/pull/73),
  [`9a47efa`](https://github.com/BenHuddart/mufinder/commit/9a47efaa0ac014754c77bfe2659194f2e89a754f))

- Make the per-row reset menu discoverable and verify the row repaints
  ([#75](https://github.com/BenHuddart/mufinder/pull/75),
  [`f26ce41`](https://github.com/BenHuddart/mufinder/commit/f26ce41534a33e3a5bebd1c1c0cc2b75fdf00ed0))

- Offer Spin Polarized in every workflow and drop Initial Spin from the starter catalog
  ([#96](https://github.com/BenHuddart/mufinder/pull/96),
  [`396a725`](https://github.com/BenHuddart/mufinder/commit/396a7250f1d02547874d12728912ca517428b05c))

- Offer Spin Polarized in every workflow and fix the muonium spin dialog
  ([#96](https://github.com/BenHuddart/mufinder/pull/96),
  [`396a725`](https://github.com/BenHuddart/mufinder/commit/396a7250f1d02547874d12728912ca517428b05c))

- Open the UEP clustering window on Windows
  ([#100](https://github.com/BenHuddart/mufinder/pull/100),
  [`eefe913`](https://github.com/BenHuddart/mufinder/commit/eefe9132ccac039bb2dce5d01d264ffcc4f03ada))

- Open the UEP clustering window on Windows and keep the UEP analysis responsive
  ([#100](https://github.com/BenHuddart/mufinder/pull/100),
  [`eefe913`](https://github.com/BenHuddart/mufinder/commit/eefe9132ccac039bb2dce5d01d264ffcc4f03ada))

- Pack the clustering graph to fill its canvas and keep edges legible
  ([#99](https://github.com/BenHuddart/mufinder/pull/99),
  [`5048132`](https://github.com/BenHuddart/mufinder/commit/5048132033e98c71e87ca393c864f6510427206d))

- Refuse a partially keyed selection in the row reset gate
  ([#75](https://github.com/BenHuddart/mufinder/pull/75),
  [`f26ce41`](https://github.com/BenHuddart/mufinder/commit/f26ce41534a33e3a5bebd1c1c0cc2b75fdf00ed0))

- Refuse a Quantum Corrections phonon stage sent away from its checkpoint
  ([#106](https://github.com/BenHuddart/mufinder/pull/106),
  [`2ba12ca`](https://github.com/BenHuddart/mufinder/commit/2ba12ca97f1af88af446b7f6246e96a5a8b31276))

- Regenerate the cluster submit script on every resubmission
  ([#78](https://github.com/BenHuddart/mufinder/pull/78),
  [`e265fef`](https://github.com/BenHuddart/mufinder/commit/e265fef383eaec15d01d904a5a994ebacd3c2965))

- Resetting the inactive member of a mixed pair no longer applies the pair
  ([#92](https://github.com/BenHuddart/mufinder/pull/92),
  [`10ef852`](https://github.com/BenHuddart/mufinder/commit/10ef85271d9b1edf534f28cface36fc38bcffa92))

- Resolve farm count in the no-carrier artifact fallback
  ([#73](https://github.com/BenHuddart/mufinder/pull/73),
  [`9a47efa`](https://github.com/BenHuddart/mufinder/commit/9a47efaa0ac014754c77bfe2659194f2e89a754f))

- Rewrite convergence job inputs when re-materializing defined jobs
  ([#74](https://github.com/BenHuddart/mufinder/pull/74),
  [`0093ad3`](https://github.com/BenHuddart/mufinder/commit/0093ad3ded984cc2dcaa61aa08c84bde10ad3a5a))

- Scale and pack the relaxation clustering graph to fit its canvas
  ([#99](https://github.com/BenHuddart/mufinder/pull/99),
  [`5048132`](https://github.com/BenHuddart/mufinder/commit/5048132033e98c71e87ca393c864f6510427206d))

- Scale relaxation clustering graph nodes with the canvas
  ([#99](https://github.com/BenHuddart/mufinder/pull/99),
  [`5048132`](https://github.com/BenHuddart/mufinder/commit/5048132033e98c71e87ca393c864f6510427206d))

- Show each relaxed site's energy relative to its comparable minimum
  ([#104](https://github.com/BenHuddart/mufinder/pull/104),
  [`39fcd75`](https://github.com/BenHuddart/mufinder/commit/39fcd756acc877b3ef6f5e0a93a3f3883598add8))

- Show muonium rows as spin-polarised and let the dialog turn it off
  ([#96](https://github.com/BenHuddart/mufinder/pull/96),
  [`396a725`](https://github.com/BenHuddart/mufinder/commit/396a7250f1d02547874d12728912ca517428b05c))

- Show the clamped convergence threshold instead of a value below its minimum
  ([#107](https://github.com/BenHuddart/mufinder/pull/107),
  [`3ef0b15`](https://github.com/BenHuddart/mufinder/commit/3ef0b158ed5511a9c13d42532c83e79d53695830))

- Show the parameters a legacy relaxation actually used
  ([#83](https://github.com/BenHuddart/mufinder/pull/83),
  [`39a169f`](https://github.com/BenHuddart/mufinder/commit/39a169f7d09430ae69ff8147e538d1c09e6c66e2))

- Stop completed cutoff rows being submitted again after a refresh
  ([#91](https://github.com/BenHuddart/mufinder/pull/91),
  [`971dfec`](https://github.com/BenHuddart/mufinder/commit/971dfec7245d83f6917a918027d496533ccf3cca))

- Stop retired UEP analysis pages covering the first tab
  ([#100](https://github.com/BenHuddart/mufinder/pull/100),
  [`eefe913`](https://github.com/BenHuddart/mufinder/commit/eefe9132ccac039bb2dce5d01d264ffcc4f03ada))

- Stop selection feedback refreshing a deleted table on quit
  ([#111](https://github.com/BenHuddart/mufinder/pull/111),
  [`4f4db0a`](https://github.com/BenHuddart/mufinder/commit/4f4db0a4dcc4245b121344dbcf5c72a30be7fe4c))

- Stop the convergence examiner's hover tooltip raising after re-analysis
  ([#107](https://github.com/BenHuddart/mufinder/pull/107),
  [`3ef0b15`](https://github.com/BenHuddart/mufinder/commit/3ef0b158ed5511a9c13d42532c83e79d53695830))

- Stop treating CASTEP runs cut off before the stress tensor as complete
  ([#95](https://github.com/BenHuddart/mufinder/pull/95),
  [`e2e9487`](https://github.com/BenHuddart/mufinder/commit/e2e9487029ecb272fafcbff5e51c87c9d1454011))

- Tighten failure classification and deletion guards from review
  ([#112](https://github.com/BenHuddart/mufinder/pull/112),
  [`2465fa3`](https://github.com/BenHuddart/mufinder/commit/2465fa354bb21504dd3d9197bee07334fd0d8058))

- Use the Kubo-Toyabe per-component convention for ZF single-crystal widths
  ([#77](https://github.com/BenHuddart/mufinder/pull/77),
  [`65a20f6`](https://github.com/BenHuddart/mufinder/commit/65a20f62743c85e36333085d9d71fefa10103caf))

- Use the sweep cell for overview spacings and state the true grid-clash reason
  ([#109](https://github.com/BenHuddart/mufinder/pull/109),
  [`2b1cf77`](https://github.com/BenHuddart/mufinder/commit/2b1cf7702413e95d00ada347b767b84d09f305d1))

- Wrap local CASTEP keyword generation in MPI launcher
  ([`7129c64`](https://github.com/BenHuddart/mufinder/commit/7129c64977e845634f11d98942d365634d24077c))

- Write the muon mass in the stage-1 quantum singlepoint
  ([#84](https://github.com/BenHuddart/mufinder/pull/84),
  [`9382464`](https://github.com/BenHuddart/mufinder/commit/9382464bd983d709b80d19c7acf285621c5fb051))

- **app**: Keep the legacy-scan worker alive for the thread's lifetime
  ([#82](https://github.com/BenHuddart/mufinder/pull/82),
  [`29e9569`](https://github.com/BenHuddart/mufinder/commit/29e9569aa73af10e9cf815e2c796c5e57114ee8f))

- **castep_io**: Recover the density-mixing settings from the .castep echo
  ([#83](https://github.com/BenHuddart/mufinder/pull/83),
  [`39a169f`](https://github.com/BenHuddart/mufinder/commit/39a169f7d09430ae69ff8147e538d1c09e6c66e2))

- **castep_io**: Stop scanning every line with two permissive block regexes
  ([#81](https://github.com/BenHuddart/mufinder/pull/81),
  [`79baa9e`](https://github.com/BenHuddart/mufinder/commit/79baa9e69f9ba5edc726d8242ed78f5d87153983))

- **science**: Convert Cartesian to fractional coordinates correctly in non-orthogonal cells, so
  attached-element muon sites in hexagonal hosts sit at the bond distance from their anchor
  ([#150](https://github.com/BenHuddart/mufinder/pull/150),
  [`3c349c6`](https://github.com/BenHuddart/mufinder/commit/3c349c63f73aa03064d6466b30c614a79ac7a876))

- **science**: Import CASTEP .cell hosts with padded or empty SPIN= values, comment lines inside
  blocks, or H:mu labels in POSITIONS_ABS
  ([#140](https://github.com/BenHuddart/mufinder/pull/140),
  [`a945be0`](https://github.com/BenHuddart/mufinder/commit/a945be0e1cfff204f6bcb3647836f2189b04042b))

- **science**: Load muesr on Python 3.13 without the removed cgi module
  ([#176](https://github.com/BenHuddart/mufinder/pull/176),
  [`ce1a997`](https://github.com/BenHuddart/mufinder/commit/ce1a99752bd4fde89265b4a47893ee43d31ff1b0))

- **services**: Record imported positions in the host frame and verify the host
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

### Documentation

- Add PR template with agent-discoverable pointers and harness guard
  ([#76](https://github.com/BenHuddart/mufinder/pull/76),
  [`c01f62b`](https://github.com/BenHuddart/mufinder/commit/c01f62b36e4114eeb6dcb34b498941420af2a202))

- Build the CASTEP syntax, k-point and branding pages
  ([#169](https://github.com/BenHuddart/mufinder/pull/169),
  [`e1cc5bf`](https://github.com/BenHuddart/mufinder/commit/e1cc5bf9cfca80e25d4198bbd75b5b852ee49d16))

- Catalog the continuation keyword and document relaxation warm starts
  ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- Fold the product spec into project_specification.rst
  ([#132](https://github.com/BenHuddart/mufinder/pull/132),
  [`7b704c9`](https://github.com/BenHuddart/mufinder/commit/7b704c91cc5240319997a6bd78855bc796407ce3))

- Keep the shared-homes inventory only in architecture.rst
  ([#162](https://github.com/BenHuddart/mufinder/pull/162),
  [`21f3fa3`](https://github.com/BenHuddart/mufinder/commit/21f3fa359a3fce4677140acf7d70d95659dd997b))

- List the row-table widget base in the AGENTS.md shared homes
  ([#141](https://github.com/BenHuddart/mufinder/pull/141),
  [`4699952`](https://github.com/BenHuddart/mufinder/commit/469995247188aac9b0559600cdf1dac2050833f8))

- Make AGENTS.md the canonical agent guide ([#129](https://github.com/BenHuddart/mufinder/pull/129),
  [`864f17e`](https://github.com/BenHuddart/mufinder/commit/864f17e010a8924e55d210faabee5c25a491f85d))

- Make AGENTS.md the canonical agent guide ([#113](https://github.com/BenHuddart/mufinder/pull/113),
  [`4440e05`](https://github.com/BenHuddart/mufinder/commit/4440e05f8e08599a57e16849325844b3c511a93b))

- Point the AGENTS.md commands at the make targets
  ([#130](https://github.com/BenHuddart/mufinder/pull/130),
  [`5a0ccca`](https://github.com/BenHuddart/mufinder/commit/5a0cccacbb6e3e75cce6cba214b0e30401d25535))

- Record the ClusterJobSubmission retrieval keywords removed in #142
  ([#158](https://github.com/BenHuddart/mufinder/pull/158),
  [`74333f3`](https://github.com/BenHuddart/mufinder/commit/74333f3ce58f901d1b00b986cdd13cc4d6878320))

- Retire finished plans and the closed HPC audit from docs/
  ([#126](https://github.com/BenHuddart/mufinder/pull/126),
  [`6157f6a`](https://github.com/BenHuddart/mufinder/commit/6157f6a796394ad750892de534d619ee7137bfd1))

- Review the documentation against the code and clarify Quantum Corrections
  ([#103](https://github.com/BenHuddart/mufinder/pull/103),
  [`b3a69dc`](https://github.com/BenHuddart/mufinder/commit/b3a69dcd4fad82e4c7d9da9259221166be8dfa14))

- Rewrite implemented functionality as a feature-surface index
  ([#170](https://github.com/BenHuddart/mufinder/pull/170),
  [`aa26b37`](https://github.com/BenHuddart/mufinder/commit/aa26b377397443652898afe2015caca114444eee))

- Rewrite the architecture controller section as a contract
  ([#129](https://github.com/BenHuddart/mufinder/pull/129),
  [`864f17e`](https://github.com/BenHuddart/mufinder/commit/864f17e010a8924e55d210faabee5c25a491f85d))

- Spell Gomilšek and Schlömann correctly in the spec and demag docstring
  ([#103](https://github.com/BenHuddart/mufinder/pull/103),
  [`b3a69dc`](https://github.com/BenHuddart/mufinder/commit/b3a69dcd4fad82e4c7d9da9259221166be8dfa14))

### Features

- Act on the eligible rows of a mixed selection and say how many
  ([#91](https://github.com/BenHuddart/mufinder/pull/91),
  [`971dfec`](https://github.com/BenHuddart/mufinder/commit/971dfec7245d83f6917a918027d496533ccf3cca))

- Add a project default XC functional and a shared project-defaults strip
  ([#86](https://github.com/BenHuddart/mufinder/pull/86),
  [`3d192df`](https://github.com/BenHuddart/mufinder/commit/3d192df8c66fc3f546cebb9e62a22ecc312fd20c))

- Add CASTEP task farming to NEB energy-barrier jobs
  ([#73](https://github.com/BenHuddart/mufinder/pull/73),
  [`9a47efa`](https://github.com/BenHuddart/mufinder/commit/9a47efaa0ac014754c77bfe2659194f2e89a754f))

- Add in-place row parameter mutation to the relaxation controllers
  ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- Add per-row "Reset Failed Jobs..." to every workflow table
  ([#75](https://github.com/BenHuddart/mufinder/pull/75),
  [`f26ce41`](https://github.com/BenHuddart/mufinder/commit/f26ce41534a33e3a5bebd1c1c0cc2b75fdf00ed0))

- Add prepare_job_continuation primitive preserving remote_directory
  ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- Add remote-file checkpoint probe primitive and controller bridge
  ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- Add SLURM --mem-per-cpu support to cluster submission
  ([#72](https://github.com/BenHuddart/mufinder/pull/72),
  [`d94ad2d`](https://github.com/BenHuddart/mufinder/commit/d94ad2d6d9256b17991687a15a79d6ab1b57d3dd))

- Add the Continue relaxation dialog ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- Allow structural-relaxation analyses to be deleted
  ([#110](https://github.com/BenHuddart/mufinder/pull/110),
  [`90758f6`](https://github.com/BenHuddart/mufinder/commit/90758f663ded39217436dd5157a97b24addcc1a5))

- Confirm every destructive action in job tables and the dashboard
  ([#89](https://github.com/BenHuddart/mufinder/pull/89),
  [`858d5b8`](https://github.com/BenHuddart/mufinder/commit/858d5b888a2628b2085a4583486ccea3a32db317))

- Import muon relaxations from legacy MuFinder projects
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

- Keep each row's own values for parameters left alone in a batch edit
  ([#92](https://github.com/BenHuddart/mufinder/pull/92),
  [`10ef852`](https://github.com/BenHuddart/mufinder/commit/10ef85271d9b1edf534f28cface36fc38bcffa92))

- Let plots that share a window be opened in their own resizable window
  ([#101](https://github.com/BenHuddart/mufinder/pull/101),
  [`e11682a`](https://github.com/BenHuddart/mufinder/commit/e11682ab6a3da41feb2476992d2d43db6b38c2a8))

- Make in-row table controls act consistently on multi-row selections
  ([#88](https://github.com/BenHuddart/mufinder/pull/88),
  [`3f36923`](https://github.com/BenHuddart/mufinder/commit/3f3692337e03027838d3a3d57dcc82941a949399))

- Make the CASTEP keyword catalog discoverable, managed and fast to generate
  ([#87](https://github.com/BenHuddart/mufinder/pull/87),
  [`65b05fb`](https://github.com/BenHuddart/mufinder/commit/65b05fbfbbb50814471773b254e66e0946b08b0e))

- Refresh a continued job's derived structure in place on re-completion
  ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- Remove imported relaxations, or refuse while they are still in use
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

- Route continued jobs back into their preserved remote directory
  ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- Show what a multi-row action will affect and share one row menu
  ([#90](https://github.com/BenHuddart/mufinder/pull/90),
  [`cf5e330`](https://github.com/BenHuddart/mufinder/commit/cf5e330c832f1a32b80d257bc592c95ad26e5ea0))

- Show why jobs failed and let resubmitted relaxations continue or start again
  ([#112](https://github.com/BenHuddart/mufinder/pull/112),
  [`2465fa3`](https://github.com/BenHuddart/mufinder/commit/2465fa354bb21504dd3d9197bee07334fd0d8058))

- Warm-start structural relaxations with a Continue relaxation action
  ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- Wire the Continue relaxation action end to end
  ([#79](https://github.com/BenHuddart/mufinder/pull/79),
  [`1fc3d0d`](https://github.com/BenHuddart/mufinder/commit/1fc3d0d83e66945d13eed0e3d29a28beee427db7))

- **app**: Add the legacy relaxation import wizard
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

- **app**: Integrate imported relaxations into the relaxations table
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

- **app**: Mark muon-relaxation rows imported from outside MuFinder
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

- **castep_io**: Read CASTEP .param files and run-parameter echoes
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

- **services**: Build project entities and files from legacy relaxations
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

- **services**: Scan legacy MuFinder folders for importable relaxations
  ([#80](https://github.com/BenHuddart/mufinder/pull/80),
  [`3e71dc4`](https://github.com/BenHuddart/mufinder/commit/3e71dc4da5ead62a611daa8e30d7feeb6539cb37))

### Performance Improvements

- Cut test-suite run time by ~40% ([#102](https://github.com/BenHuddart/mufinder/pull/102),
  [`99b2a4f`](https://github.com/BenHuddart/mufinder/commit/99b2a4f8e5e5f42552ecb4a5fa168e90281ea5d0))

- Stop re-serializing the UEP potential grid on every state change
  ([#100](https://github.com/BenHuddart/mufinder/pull/100),
  [`eefe913`](https://github.com/BenHuddart/mufinder/commit/eefe9132ccac039bb2dce5d01d264ffcc4f03ada))

- Vectorise symmetry checks in clustering, Wyckoff assignment and site generation
  ([#105](https://github.com/BenHuddart/mufinder/pull/105),
  [`acfc0f3`](https://github.com/BenHuddart/mufinder/commit/acfc0f30e375337181026d0a7336dbc982209767))


## v0.5.0 (2026-07-08)

### Features

- Add CIF export for muon-site analyses, from both the Muon Sites summary
  cards (pinned, distinct clustered sites) and each structural-relaxation
  Analysis workspace (every relaxed muon position)
  ([#68](https://github.com/BenHuddart/mufinder/pull/68))


## v0.4.1 (2026-07-07)

### Bug Fixes

- Render clustering/displacement 3D viewer in place on Windows
  ([#66](https://github.com/BenHuddart/mufinder/pull/66),
  [`f3f5033`](https://github.com/BenHuddart/mufinder/commit/f3f5033b53e29fb64da4d7b566625bc2aee06cd8))


## v0.4.0 (2026-07-06)

### Bug Fixes

- Keep per-row Configure/remove buttons clickable after first use
  ([`9547d61`](https://github.com/BenHuddart/mufinder/commit/9547d61dd0a2b530ae50b7a61d8a71a285316f8a))

- **cluster**: Close review findings — guarantee check emit, bound shutdown
  ([`22325b3`](https://github.com/BenHuddart/mufinder/commit/22325b38058078787b4479e9dfcf98fcc4e40fa8))

- **cluster**: Stop the poll timer on the worker thread during shutdown
  ([`ddb8365`](https://github.com/BenHuddart/mufinder/commit/ddb8365a2290b4cc6d1188c1356653d2f3247e1d))

- **jobs**: Persist coalesced flush at context boundaries; drop redundant refreshes
  ([`d83044a`](https://github.com/BenHuddart/mufinder/commit/d83044a43ac6dfc50b0e39a61b29e4d6a5111782))

- **ui**: Brand item-view checkboxes and fix disabled-checked contrast
  ([`a4d6a22`](https://github.com/BenHuddart/mufinder/commit/a4d6a22081a123a1176bb70660589ca2a3742110))

### Documentation

- Require scientific-backends extra and bound local test parallelism
  ([#34](https://github.com/BenHuddart/mufinder/pull/34),
  [`f840807`](https://github.com/BenHuddart/mufinder/commit/f84080718ba980013889e243666950e0b641fcb9))

### Features

- **ui**: Brand checkbox and radio indicators with the accent
  ([`f10cc8c`](https://github.com/BenHuddart/mufinder/commit/f10cc8c6676311fbf96342e78f44c0509d487ff7))

- **ui**: Bundle Inter/Fraunces fonts and force Fusion for cross-platform parity
  ([`07a91ef`](https://github.com/BenHuddart/mufinder/commit/07a91ef8419f8930e44325975e70a9dea1a175ca))

- **ui**: Prefer branded vector icons over platform standard icons
  ([`5348b05`](https://github.com/BenHuddart/mufinder/commit/5348b05b2899f68a9e372f34fcbdd6f7179401fc))

- **ui**: Scale project-home signpost labels with the map
  ([`292c9b5`](https://github.com/BenHuddart/mufinder/commit/292c9b50fdcd9fa2af1c4d626a2eb2b19cde979c))

### Performance Improvements

- **cluster**: Make batch submission async with progress and cancel
  ([`3aa457e`](https://github.com/BenHuddart/mufinder/commit/3aa457ef5c742362b29431e259fe64e58d12db16))

- **cluster**: Make the connect handshake non-blocking
  ([`afda3b5`](https://github.com/BenHuddart/mufinder/commit/afda3b58a71e1f7d9945691da50f5917a25279da))

- **cluster**: Run SSH work on a worker thread to keep the GUI responsive
  ([`a2b9d27`](https://github.com/BenHuddart/mufinder/commit/a2b9d27064e937bc2c449980169fbdf0a567e6e9))

- **jobs**: Coalesce per-job autosave so completion bursts don't freeze the GUI
  ([`6436e15`](https://github.com/BenHuddart/mufinder/commit/6436e15461712762a841623f3d1cfda86420d93b))

- **jobs**: Coalesce the whole dashboard update on completion bursts
  ([`d659452`](https://github.com/BenHuddart/mufinder/commit/d65945244ee5c012f2eff245233d1f12401bf326))


## v0.3.0 (2026-06-17)

### Bug Fixes

- Patch _generate_preview in spinbox-range regression tests to prevent VTK abort
  ([`5d32db7`](https://github.com/BenHuddart/mufinder/commit/5d32db7a797b6c2413a0c651ea9196f470b45e0b))

- **tests**: Mock is_glamdring_keychain_supported in keychain credential tests
  ([`cc36b43`](https://github.com/BenHuddart/mufinder/commit/cc36b43cdb1ac69d77985f90241f72b794bf051c))

### Documentation

- Document branch-protection-aware release process
  ([`b955f68`](https://github.com/BenHuddart/mufinder/commit/b955f680f33992f75e9870cef7bfddddc99f794e))

### Features

- Add Suggestâ€¦ supercell button with Pareto-front candidate table
  ([`f53c07e`](https://github.com/BenHuddart/mufinder/commit/f53c07eaa3c745a13c21bf682301ff9ca32dea29))


## v0.2.0 (2026-06-05)

### Bug Fixes

- Address code-review findings in installer, workflows, and version resolution
  ([`2ce6ebc`](https://github.com/BenHuddart/mufinder/commit/2ce6ebc90e337829c54c202d0c8ef7efd87c0978))

- Address review feedback
  ([`96a30a2`](https://github.com/BenHuddart/mufinder/commit/96a30a207b4638f6001fa5972152bf4600876507))

- Align k-point spacing convention with CASTEP round-trip formula
  ([`7e23ffb`](https://github.com/BenHuddart/mufinder/commit/7e23ffb142a845d472690bc0a8f37294c81c2d96))

- Correct k-point spacing formula and harden spacing annotations
  ([`1e8ff59`](https://github.com/BenHuddart/mufinder/commit/1e8ff59dc7e6a03c67ef8dbc2b054ed7718300e8))

- Make local file cleanup resilient to Windows permission errors
  ([`3827fb5`](https://github.com/BenHuddart/mufinder/commit/3827fb51a311d57c80dbb71bfa3f91e15d6300c9))

- Multi-row table button actions and pre-existing test failures
  ([`98cfaed`](https://github.com/BenHuddart/mufinder/commit/98cfaed2cf1669d4e01003ac928d4dc6715470d5))

- Pin PySide6<6.11 and make the TS-parser test use a committed fixture
  ([`1ce3579`](https://github.com/BenHuddart/mufinder/commit/1ce35792b84d34362ba665f40ba983099ac46f9f))

- Replace WiX 5 glob with generated fragment and fix k-point convergence labels
  ([`d07ab8d`](https://github.com/BenHuddart/mufinder/commit/d07ab8da0f4dbbd37d66960ac9f9d56091ccaf55))

- Roll back Windows installer from WiX to NSIS
  ([`93f939e`](https://github.com/BenHuddart/mufinder/commit/93f939ee50b9b768e1f5495e8337e4b97c90520f))

- Use plain type aliases for Python 3.10/3.11 compatibility
  ([`cae15cb`](https://github.com/BenHuddart/mufinder/commit/cae15cbf391a20e84cabeb4ca1a44d6f02391ad7))

- Write muon mass block in relaxation cell files, fix kpoint hint gaps
  ([`8edacae`](https://github.com/BenHuddart/mufinder/commit/8edacae2765bcf0be666fcec02e7cb7cf866715e))

- **castep**: Add iprint/length_unit/devel_code/xc_definition param keywords
  ([`21ad49e`](https://github.com/BenHuddart/mufinder/commit/21ad49e800571501f62701fa2c9ca7baf894e29c))

- **castep**: Add workflow keywords missing from the bundled starter catalog
  ([`30e28b8`](https://github.com/BenHuddart/mufinder/commit/30e28b8d556e16179c7af7ebc01f8129c1401345))

- **castep**: Drop unsatisfiable TASK=... requires from starter catalog blocks
  ([`8c74d81`](https://github.com/BenHuddart/mufinder/commit/8c74d81909970a52d4c8e2759015ba4af38db29a))

- **packaging**: Remove invalid 'checked' flag from Inno Setup task definition
  ([`4a5346b`](https://github.com/BenHuddart/mufinder/commit/4a5346bdf0137a38406c328ddeaa361d1affe102))

- **relaxation**: Correct muon frame in supercell displacement analysis
  ([`a3ab195`](https://github.com/BenHuddart/mufinder/commit/a3ab19525ea4e1300c16bd8af70d438f07768085))

- **release**: Automate release branch cleanup
  ([`7dca4bb`](https://github.com/BenHuddart/mufinder/commit/7dca4bbe55febe9aa6f7824f2ec1b5a6c328fb42))

- **theme**: Stop _TableCornerStyler caching Qt wrappers past their lifetime
  ([`16e7d7f`](https://github.com/BenHuddart/mufinder/commit/16e7d7f09a847c1a2097379285a97a8043e5431f))

### Documentation

- Add CLAUDE.md as Claude Code guide for this repo
  ([`1410a1e`](https://github.com/BenHuddart/mufinder/commit/1410a1e8ef847684b9b8d2c098fa5bb264de6046))

- Update README â€” Inno Setup, remove stale 'now' language and historical notes
  ([`e4b9af3`](https://github.com/BenHuddart/mufinder/commit/e4b9af32918756cf0f2ff02a92c277651c7acca2))

### Features

- Add magnetic field workflows
  ([`1161a2d`](https://github.com/BenHuddart/mufinder/commit/1161a2d3fb6f85b9bad34d8a23326caa1bdfaead))

- Add max stress as cutoff/k-point convergence criterion
  ([`d42538f`](https://github.com/BenHuddart/mufinder/commit/d42538f6553bc88bec144b552d59abaa02275c84))

- Add module requirement check and update PyInstaller spec for dynamic libs
  ([`25a496b`](https://github.com/BenHuddart/mufinder/commit/25a496be2ae7d2eef097514a7df0e810f10cdba3))

- Add muLFC dependency for Windows and update tests for installation verification
  ([`6166fde`](https://github.com/BenHuddart/mufinder/commit/6166fdea9856fd445523b93da725986d9549b1c8))

- Add UEP workflow and analysis tooling
  ([`b408130`](https://github.com/BenHuddart/mufinder/commit/b408130671903a525b2cc4cd2fa6b47da6582352))

- Allow switching active host structure between imported and relaxed geometries
  ([`4afc79c`](https://github.com/BenHuddart/mufinder/commit/4afc79c6cf20f1b2b26de5ef2ffc1ffecd0bda2e))

- Enhance input controls and update PyInstaller spec for runtime assets
  ([`1e4c1a2`](https://github.com/BenHuddart/mufinder/commit/1e4c1a244a53b497b49885adcf469bc59a38766f))

- Expand UEP analysis and journal workflows
  ([`71ea849`](https://github.com/BenHuddart/mufinder/commit/71ea849eb1fb56bc4324a628820136e5255538b6))

- Migrate Windows installer to WiX MSI and refactor workflow parameter UI
  ([`5d97465`](https://github.com/BenHuddart/mufinder/commit/5d974659ff1084ce307f9c1db9b935136e6e1a98))

- Surface k-point spacing throughout the convergence workflow
  ([`82fc461`](https://github.com/BenHuddart/mufinder/commit/82fc461f8981dc208f74fa4019348fb68229e37e))

- Surface stress convergence on the overview tab
  ([`f09d232`](https://github.com/BenHuddart/mufinder/commit/f09d232016b188ccf96076a2b6c1a1034aff9cdc))

- Update Python dependency installation and add runtime import verification in workflows
  ([`406c08d`](https://github.com/BenHuddart/mufinder/commit/406c08db5827c7ecebb807b35daa082f1fb49302))

- Update Python dependency installation to include optional scientific backends and enhance import
  verification
  ([`d700831`](https://github.com/BenHuddart/mufinder/commit/d70083165f560e196b7dbab4cf4c584c20fbc940))


## v0.1.2 (2026-04-12)

### Bug Fixes

- **release**: Allow version prep on release branches
  ([`5b284af`](https://github.com/BenHuddart/mufinder/commit/5b284af63cda441f0a3b7473e0ebcdf9a14f3419))

- **release**: Avoid gha output bug in release prep
  ([`4e3a367`](https://github.com/BenHuddart/mufinder/commit/4e3a367fa281d893f21919b9515427cf6ae9bf41))
