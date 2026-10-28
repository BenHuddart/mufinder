# α-N2 muon phonon outputs

Real CASTEP files from M. Gomilšek, F. L. Pratt, S. P. Cottrell, S. J. Clark and
T. Lancaster, *Many-body quantum muon effects and quadrupolar coupling in solids*,
Commun. Phys. **6**, 142 (2023), https://doi.org/10.1038/s42005-023-01260-7. The data
set is https://doi.org/10.6084/m9.figshare.23203037.v1 (`data_and_code.zip`, MD5
`2b85755761b54ad1c678c80d3e259c6d`), licensed CC BY 4.0; the files come from its
`Fig 2c/DFT/` directory.

The calculation: the Gamma-point finite-displacement phonons of a positive muon in a
2x2x2 supercell of Pa-3 α-N2 (64 N and the muon, `H:mu`), CASTEP 19.11, PBE with the
Tkatchenko–Scheffler dispersion correction (`SEDC_SCHEME : TS`), ultrasoft
pseudopotentials, 1200 eV, a 2x2x2 Monkhorst-Pack grid, charge +1,
`ELEC_ENERGY_TOL : 1e-10 eV`, `PHONON_FINE_METHOD : supercell` and
`PHONON_SUM_RULE_METHOD : real-recip`. The cell is the relaxed muon supercell, held
fixed (`FIX_ALL_CELL`) at 11.327064 Å under an `EXTERNAL_PRESSURE` of 0.5 GPa.

The seed of every file in the data set is
`aN2_1200eV_k222_n180_0.5GPa_sedcTS_charge+1_cellCHG+1-SUPER222-PRECISE2_1200eV-SEDC_TS_scf1e-10eV`
(written `<seed>` below); the fixtures are renamed to the seed `aN2_mu_phonon`.

| File | Source in the data set | Change |
|---|---|---|
| `aN2_mu_phonon.cell` | `Fig 2c/DFT/<seed>.cell` | none |
| `aN2_mu_phonon.param` | `Fig 2c/DFT/<seed>.param` | none |
| `aN2_mu_phonon-out.cell` | `Fig 2c/DFT/<seed>-out.cell` | none |
| `aN2_mu_phonon.castep` | `Fig 2c/DFT/<seed>.castep` | none |
| `aN2_mu_phonon.phonon` | `Fig 2c/DFT/<seed>.phonon` | none |

The data set holds no log of the relaxation that produced the cell, so the
documentation screenshots write a short `.castep` for it, with the
dispersion-corrected energy of the undisplaced cell from the phonon run's first SCF
solve (-17718.38090479 eV). `tests/docs/test_docs_alpha_n2.py` pins the numbers the
worked example `docs/user_guide/examples/alpha_n2.rst` reports from these files.
