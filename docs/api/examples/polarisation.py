"""Zero- and longitudinal-field polarisation and spectrum for a muon in bcc iron.

Needs the ``scientific-backends`` extra (muesr and muLFC).

    python docs/api/examples/polarisation.py
"""

import numpy as np

from mufinder.domain.models import Structure
from mufinder.science.magnetic_fields import MagneticFieldOptions, calculate_commensurate_fields
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticStructureModel,
    build_collinear_component,
)
from mufinder.science.nuclear_relaxation import calculate_nuclear_width_tensors
from mufinder.science.polarisation import (
    calculate_lf_polarisation,
    calculate_polarisation,
    calculate_spectrum,
)

# The 2.2 muB ferromagnet of the field example, with the field at all three
# octahedral sites, equally occupied. Two lie across the moment and see 0.2018 T;
# the third lies on the moment axis, between two iron atoms, and sees 2.58 T.
host = Structure(
    label="Fe",
    lattice_vectors=[[2.868, 0.0, 0.0], [0.0, 2.868, 0.0], [0.0, 0.0, 2.868]],
    species=["Fe", "Fe"],
    fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
)
model = MagneticStructureModel(
    components=(
        build_collinear_component(
            label="Ferromagnet",
            selector=AtomSelector.all(),
            q_coeff=(0.0, 0.0, 0.0),
            moment=(0.0, 0.0, 2.2),
        ),
    ),
)
sites = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]
fields = calculate_commensurate_fields(
    host, model, sites, options=MagneticFieldOptions(supercell=(80, 80, 80))
)

# The nuclear width along each site's field, from one lattice sum per site.
widths = [calculate_nuclear_width_tensors(host, site, radius=12.0) for site in sites]

# Zero field, a polycrystal: 1/3 + 2/3 cos(gamma |B| t) per site, damped by the nuclei.
times = np.linspace(0.0, 2.0, 2001)
zero_field = calculate_polarisation(
    fields, times, polycrystalline=True, sigma_nuclear="auto", nuclear_width_tensors=widths
)
spectrum = calculate_spectrum(times, zero_field.polarisation, padding_factor=8)
peak = int(np.argmax(spectrum.amplitudes))

# A 0.5 T longitudinal field on the powder decouples most of the internal field.
decoupled = calculate_lf_polarisation(
    fields, times, applied_field=(0.0, 0.0, 0.5), polycrystalline=True
)

print("|B| at each site:", ", ".join(f"{field.magnitude:.4f} T" for field in fields))
print(f"Zero-field P(0): {zero_field.polarisation[0]:.3f}")
print(f"Spectrum peak: {spectrum.frequencies_MHz[peak]:.2f} MHz, {spectrum.fields_T[peak]:.3f} T")
print(f"P_z in 0.5 T, mean over the last 0.5 us: {np.mean(decoupled.polarisation[-500:]):.2f}")
