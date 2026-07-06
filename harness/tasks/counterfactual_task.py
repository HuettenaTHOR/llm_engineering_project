"""Counterfactual task: prompt construction + seeded target for the resumable CF loop.

Integrated from code_seb. Owns *how* each role is prompted (the message layouts) and the
seeded per-item target; ``CounterfactualStrategy`` (harness/strategies/counterfactual_loop.py)
owns control flow + I/O and does all the actual ``model.inference`` calls.

Unlike ``SolveTask`` this is NOT a ``BaseTask`` (it has no single build_messages/grade contract):
the CF loop drives three roles (solver / verifier / checker) through the dedicated strategy, so
the task just exposes pure message builders. It reuses the harness extraction/grading helpers so
the answer-parsing convention stays identical across the project.
"""
import random

from harness.answer_extraction import extract_float
from harness.tasks.base_task import to_int
from harness.CONSTANTS import (
    COUNTERFACTUAL_REVISE_TEMPLATE,
    COUNTERFACTUAL_FLAW_FEEDBACK,
    COUNTERFACTUAL_VERIFIER_SYSTEM_PROMPT,
)


class CounterfactualTask:
    def __init__(self, dataset, seed: int = 42, offset_low: int = -10, offset_high: int = 10,
                 verifier_sees_solver_output: bool = False):
        self.dataset = dataset
        self.seed = seed
        self.offset_low = offset_low
        self.offset_high = offset_high
        # False -> blind judge (target + candidate only, re-solves independently). True -> trace-aware
        # judge that also sees the solver's original solve. See verifier_messages.
        self.verifier_sees_solver_output = verifier_sees_solver_output

    def gold(self, example) -> int | None:
        """Original GSM8K gold integer (the ``#### <n>`` answer), for logging / original_correct."""
        return to_int(extract_float(self.dataset.postprocess_result(example["answer"])))

    def offset(self, example) -> int:
        """Seeded, deterministic per-item noise in [offset_low, offset_high] EXCLUDING 0, so the
        counterfactual target always differs from the solver's own answer (signed Uniform)."""
        rng = random.Random(f"{self.seed}:{example['question']}")
        return rng.choice([n for n in range(self.offset_low, self.offset_high + 1) if n != 0])

    # --- message builders (pure; the strategy runs inference on these) --------------------
    def solve_messages(self, question: str, model) -> list:
        """Plain step-by-step solve with the ``#### <n>`` format (used for the original solve
        AND the independent benchmark re-solve). Question embedded via the dataset system prompt."""
        return model.build_conversation_from_system_prompt(
            self.dataset.system_prompt(problem=question), user_input=question
        )

    def cf_messages(self, question: str, solver_output: str, target: int, history: list) -> list:
        """CF generation: continue the solver thread. ``history`` is the list of prior
        (rejected_candidate, flaw) pairs -- empty on iteration 0 (CF #1), one-or-more on CF #2+."""
        revise = COUNTERFACTUAL_REVISE_TEMPLATE.format(target=target)
        messages = [
            {"role": "system", "content": self.dataset.system_prompt(problem=question)},
            {"role": "assistant", "content": solver_output},
            {"role": "user", "content": revise},
        ]
        for candidate, flaw in history:
            messages.append({"role": "assistant", "content": candidate})
            messages.append({"role": "user", "content": COUNTERFACTUAL_FLAW_FEEDBACK.format(flaw=flaw)})
        return messages

    def verifier_messages(self, question: str, solver_output: str, target: int, candidate: str) -> list:
        """Verifier judge layout. The system prompt always carries the original question + target.

        Blind (``verifier_sees_solver_output=False``, default): system + a single user turn with the
        candidate, so the judge re-solves independently (no anchoring on the solver's reasoning).

        Trace-aware (True): the 4-turn layout that also replays the solver's original solve. The
        candidate must be the final *user* turn for the judge to answer, so the solver's turn sits
        as ``user`` and the revise instruction as ``assistant`` (deliberate, not a role slip)."""
        system = COUNTERFACTUAL_VERIFIER_SYSTEM_PROMPT.format(question=question, target=target)
        if not self.verifier_sees_solver_output:
            return [
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    "This is the revised math problem. Check whether its correct final answer is "
                    f"exactly {target}:\n{candidate}"
                )},
            ]
        revise = COUNTERFACTUAL_REVISE_TEMPLATE.format(target=target)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": solver_output},
            {"role": "assistant", "content": revise},
            {"role": "user", "content": f"This is the revised math problem with different solution:\n{candidate}"},
        ]
