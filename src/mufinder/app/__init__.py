"""Application UI package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ClusterSetupDialog",
    "ClusterSubmissionConfig",
    "ClusterSubmissionDialog",
    "CastepSetupDialog",
    "CastepParamForm",
    "ConvergenceConfig",
    "ConvergenceWidget",
    "CutoffConvergenceWidget",
    "PristineRelaxationConfig",
    "PristineRelaxationWidget",
    "KPointConvergenceWidget",
    "MathLabel",
    "ParameterDialog",
    "ParameterHelpDialog",
    "LocalSubmissionDialog",
    "SubmissionConfig",
    "build_convergence_workflow",
    "expand_convergence_jobs",
    "expand_pristine_relaxation_jobs",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "ClusterSetupDialog": ("mufinder.app.cluster_setup_dialog", "ClusterSetupDialog"),
    "ClusterSubmissionConfig": ("mufinder.app.cluster_submission_dialog", "ClusterSubmissionConfig"),
    "ClusterSubmissionDialog": ("mufinder.app.cluster_submission_dialog", "ClusterSubmissionDialog"),
    "CastepSetupDialog": ("mufinder.app.castep_setup_dialog", "CastepSetupDialog"),
    "CastepParamForm": ("mufinder.app.param_form", "CastepParamForm"),
    "ConvergenceConfig": ("mufinder.app.convergence_types", "ConvergenceConfig"),
    "ConvergenceWidget": ("mufinder.app.convergence_widget", "ConvergenceWidget"),
    "CutoffConvergenceWidget": ("mufinder.app.convergence_widget", "CutoffConvergenceWidget"),
    "PristineRelaxationConfig": ("mufinder.app.relaxation_types", "PristineRelaxationConfig"),
    "PristineRelaxationWidget": ("mufinder.app.relaxation_widget", "PristineRelaxationWidget"),
    "KPointConvergenceWidget": ("mufinder.app.kpoint_convergence_widget", "KPointConvergenceWidget"),
    "MathLabel": ("mufinder.app.math_label", "MathLabel"),
    "ParameterDialog": ("mufinder.app.param_dialog", "ParameterDialog"),
    "ParameterHelpDialog": ("mufinder.app.help_dialog", "ParameterHelpDialog"),
    "LocalSubmissionDialog": ("mufinder.app.submission_dialog", "LocalSubmissionDialog"),
    "SubmissionConfig": ("mufinder.app.submission_dialog", "SubmissionConfig"),
    "build_convergence_workflow": ("mufinder.app.convergence_adapter", "build_convergence_workflow"),
    "expand_convergence_jobs": ("mufinder.app.convergence_adapter", "expand_convergence_jobs"),
    "expand_pristine_relaxation_jobs": ("mufinder.app.relaxation_adapter", "expand_pristine_relaxation_jobs"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = target
    module = import_module(module_name)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
