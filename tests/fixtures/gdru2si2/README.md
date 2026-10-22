# GdRu2Si2 muon-site outputs

Real CASTEP outputs from B. M. Huddart, A. Hernández-Melián, G. D. A. Wood,
D. A. Mayoh, M. Gomilšek, Z. Guguchia, C. Wang, T. J. Hicken, S. J. Blundell,
G. Balakrishnan and T. Lancaster, *Field-orientation-dependent magnetic phases
in GdRu2Si2 probed with muon-spin spectroscopy*, Phys. Rev. B **111**, 054440
(2025), https://doi.org/10.1103/PhysRevB.111.054440. The data set is
https://doi.org/10.15128/r15138jd91z (directory `sites/`).

The calculations: PBE, spin-polarised with ferromagnetic Gd (`SPIN 7`), a 2x2x1
supercell of the conventional cell (41 atoms with the muon, `H:mu`), 1000 eV,
charge +1, EDFT. The relaxations used a 5x5x5 Monkhorst-Pack grid; the NEB band
and the Gamma-point phonons (finite displacement) used 3x3x3, with 4 task farms.

| File | Source in the data set | Change |
|---|---|---|
| `GdRu2Si2_site1-out.cell` | `GdRu2Si2_site1-out.cell` | none |
| `GdRu2Si2_site2-out.cell` | `GdRu2Si2_site2-out.cell` | none |
| `GdRu2Si2_site1.phonon` | `phonons/site 1/GdRu2Si2_site1.phonon` | none |
| `GdRu2Si2_site2.phonon` | `phonons/site 2/GdRu2Si2_site2.phonon` | none |
| `GdRu2Si2_12_neb.ts` | `tss/NEB/GdRu2Si2_12_farm001.ts` | trimmed |

`GdRu2Si2_12_neb.ts` is the NEB band from site 1 to site 2 (40 images,
`TSSEARCH_FORCE_TOL : 0.08`). The 37 MB original holds every iteration of the
band; the fixture keeps its `TSTYPE` header line, the `REA` and `PRO` images
and the 40 `TST` images of the last iteration (112), byte for byte and in the
original order. `mufinder.castep_io.ts_parser` reads only those images, so it
gives the same path from both files: 42 images and a barrier of 0.495 eV.

The data set holds no relaxation log. The energies of the two relaxed cells
with the relaxation settings (5x5x5), -69184.849 eV (site 1) and -69184.592 eV
(site 2), are the endpoint energies its transition-state search reports
(`tss/12/GdRu2Si2_12.castep`, and the hyperfine runs of each site); the
documentation screenshots write a short `.castep` around each.
`tests/docs/test_docs_gdru2si2.py` pins the numbers the worked example
`docs/user_guide/examples/gdru2si2.rst` reports from these files.
