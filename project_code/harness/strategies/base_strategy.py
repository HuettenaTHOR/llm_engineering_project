from abc import ABC, abstractmethod


class Strategy(ABC):
    """Pluggable control flow over a task. Single-shot and the agentic solver-verifier
    loop both implement this one interface so the runner treats them identically."""

    @abstractmethod
    def run(self, task, example, model) -> dict:
        """Solve one ``example`` with ``model`` under ``task``.

        Returns a record dict with at least::

            {
              "final_pred": int | None,
              "final_correct": bool | None,
              "gen_fail": bool,
              "iterations": [
                {"iteration", "candidate", "solver_solve", "verifier_output",
                 "verifier_says", "verifier_reason", "verdict"},
                ...
              ],
            }
        """
