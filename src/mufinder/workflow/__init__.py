"""CASTEP calculation workflow framework.

Expands scientific inputs into resolved JobSpec objects that can be
rendered into CASTEP .cell and .param files for local testing,
or later combined with HPC execution metadata for remote submission.
"""

from .bridge import StructureBridgeError, structure_to_spec
from .convergence import ConvergenceWorkflow
from .energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    NUDGED_ELASTIC_BAND_SET_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
    NudgedElasticBandJobConfig,
    NudgedElasticBandSource,
    TransitionStateSearchJobConfig,
    TransitionStateSearchSource,
    expand_nudged_elastic_band_jobs,
    expand_transition_state_search_jobs,
    position_block_from_structure,
)
from .merge import MERGE_PRECEDENCE_ORDER, merge_templates, merge_with_precedence, resolve_job
from .muon_site import MuonSiteWorkflow
from .naming import job_name
from .output import write_job_directory, write_job_set, write_manifest
from .relaxation import RelaxationJobConfig, expand_relaxation_jobs
from .renderer import render_job
from .types import CellTemplate, InputTemplate, JobPatch, JobSpec, ParamTemplate, StructureSpec

__all__ = [
    "CellTemplate",
    "ConvergenceWorkflow",
    "InputTemplate",
    "JobPatch",
    "JobSpec",
    "MERGE_PRECEDENCE_ORDER",
    "MuonSiteWorkflow",
    "ParamTemplate",
    "RelaxationJobConfig",
    "StructureBridgeError",
    "StructureSpec",
    "NUDGED_ELASTIC_BAND_CALC_TYPE",
    "NUDGED_ELASTIC_BAND_SET_CALC_TYPE",
    "TRANSITION_STATE_SEARCH_CALC_TYPE",
    "TRANSITION_STATE_SEARCH_SET_CALC_TYPE",
    "NudgedElasticBandJobConfig",
    "NudgedElasticBandSource",
    "TransitionStateSearchJobConfig",
    "TransitionStateSearchSource",
    "expand_nudged_elastic_band_jobs",
    "expand_relaxation_jobs",
    "expand_transition_state_search_jobs",
    "job_name",
    "merge_templates",
    "merge_with_precedence",
    "position_block_from_structure",
    "render_job",
    "resolve_job",
    "structure_to_spec",
    "write_job_directory",
    "write_job_set",
    "write_manifest",
]
