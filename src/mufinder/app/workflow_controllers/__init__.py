"""Per-workflow job controllers (submit → materialize → result → refresh).

This package holds a thin controller layer between the ``MainWindow`` widgets
and the services/workflow layers. Each controller owns one workflow's job
lifecycle and receives everything it needs through a
:class:`WorkflowControllerContext` of narrow callables, so controllers never
import or reference ``mufinder.app.main_window``.
"""

from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    merge_controller_result,
)
from mufinder.app.workflow_controllers.convergence_cutoff import (
    ConvergenceCutoffController,
)

__all__ = [
    "ControllerResult",
    "ConvergenceCutoffController",
    "WorkflowControllerContext",
    "WorkflowJobController",
    "merge_controller_result",
]
