# Changelog

<!-- version list -->

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

- Update README — Inno Setup, remove stale 'now' language and historical notes
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
