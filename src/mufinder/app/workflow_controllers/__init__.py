"""Per-workflow job controllers (submit → materialize → result → refresh).

This package holds a thin controller layer between the ``MainWindow`` widgets
and the services/workflow layers. Each controller owns one workflow's job
lifecycle and receives its inputs in three parts: the shared
:class:`WorkflowControllerContext` of narrow callables every controller needs,
an optional per-controller deps dataclass of workflow-specific callables
defined in the controller's own module (e.g. :class:`ConvergenceCutoffDeps`),
and the controller's own view widget. Controllers never import or reference
``mufinder.app.main_window``.
"""

from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    merge_controller_result,
)
from mufinder.app.workflow_controllers.convergence_cutoff import (
    ConvergenceCutoffController,
    ConvergenceCutoffDeps,
)
from mufinder.app.workflow_controllers.convergence_kpoint import (
    ConvergenceKpointController,
    ConvergenceKpointDeps,
)
from mufinder.app.workflow_controllers.pristine_relaxation import (
    PristineRelaxationController,
    PristineRelaxationDeps,
)

__all__ = [
    "ControllerResult",
    "ConvergenceCutoffController",
    "ConvergenceCutoffDeps",
    "ConvergenceKpointController",
    "ConvergenceKpointDeps",
    "PristineRelaxationController",
    "PristineRelaxationDeps",
    "WorkflowControllerContext",
    "WorkflowJobController",
    "merge_controller_result",
]
