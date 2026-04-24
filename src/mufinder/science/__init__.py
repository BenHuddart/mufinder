"""Scientific algorithms for MuFinder."""

from mufinder.science.convergence_analysis import (
    ConvergenceResult,
    analyze_convergence,
    analyze_kpoint_convergence,
    asymptotic_extrapolation_energy,
    delta_threshold_energy,
    delta_threshold_force,
    derivative_energy,
    recommended_cutoff,
    recommended_value,
)
from mufinder.science.kpoint_grids import (
    format_kpoint_grid,
    generate_kpoint_grids,
    grid_from_spacing,
    kpoint_count,
    lattice_matches,
    reciprocal_lattice_vectors,
)
from mufinder.science.site_generation import (
    SiteGenerationError,
    closest_symmetry_equivalent_position,
    estimate_muon_muon_separation_for_count,
    estimate_random_site_count_from_separation,
    generate_attached_element_sites,
    generate_random_sites,
    generate_random_sites_by_distance,
    lattice_cart_to_frac,
    lattice_frac_to_cart,
)
from mufinder.science.structure_io import LoadedStructure, StructureLoadError, load_structure_file
from mufinder.science.structure_comparison import (
    ComparisonResult,
    DftComparisonColumn,
    DftComparisonInput,
    compare_structures,
)
from mufinder.science.structure_metrics import MetricDefinition, MetricValue, metric_definitions, metric_info_text
from mufinder.science.supercell import build_supercell_structure

__all__ = [
    "ConvergenceResult",
    "analyze_convergence",
    "analyze_kpoint_convergence",
    "asymptotic_extrapolation_energy",
    "delta_threshold_energy",
    "delta_threshold_force",
    "derivative_energy",
    "recommended_cutoff",
    "recommended_value",
    "format_kpoint_grid",
    "generate_kpoint_grids",
    "grid_from_spacing",
    "kpoint_count",
    "lattice_matches",
    "reciprocal_lattice_vectors",
    "SiteGenerationError",
    "closest_symmetry_equivalent_position",
    "estimate_muon_muon_separation_for_count",
    "estimate_random_site_count_from_separation",
    "generate_attached_element_sites",
    "generate_random_sites",
    "generate_random_sites_by_distance",
    "lattice_cart_to_frac",
    "lattice_frac_to_cart",
    "LoadedStructure",
    "StructureLoadError",
    "load_structure_file",
    "ComparisonResult",
    "DftComparisonColumn",
    "DftComparisonInput",
    "compare_structures",
    "MetricDefinition",
    "MetricValue",
    "metric_definitions",
    "metric_info_text",
    "build_supercell_structure",
]
