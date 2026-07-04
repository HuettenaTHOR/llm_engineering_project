from harness.tasks.base_task import BaseTask
from harness.tasks.solve_task import SolveTask
from harness.tasks.counterfactual_task import CounterfactualTask

# CounterfactualTask is the message-builder task (integrated from code_seb, which is preserved
# as a snapshot). It is driven by CounterfactualStrategy via harness/counterfactual_runner.py
# (a dedicated step-streaming, three-role loop) -- NOT by the per-item harness/runner.py.

__all__ = ["BaseTask", "SolveTask", "CounterfactualTask"]
