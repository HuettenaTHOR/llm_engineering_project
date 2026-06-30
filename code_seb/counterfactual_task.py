"""Counterfactual task: prompt construction + grading policy for the resumable loop.

Owns *how* each role is prompted (the ground-truth message layouts from the design sketch) and
the seeded per-item target; the strategy (counterfactual_strategy.py) owns control flow + I/O
and does all the actual ``model.inference`` calls. Reuses the harness extraction/grading
helpers so the answer-parsing convention stays identical across the project.
"""
import random

from harness.answer_extraction import extract_float
from code_seb.grading import to_int
from code_seb.cf_prompts import REVISE_TEMPLATE, CF_FLAW_FEEDBACK, CHECK_CF_SYSTEM_PROMPT


class CounterfactualTask:
    def __init__(self, dataset, seed: int = 42, offset_low: int = -10, offset_high: int = 10):
        self.dataset = dataset
        self.seed = seed
        self.offset_low = offset_low
        self.offset_high = offset_high

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
        messages = [
            {"role": "system", "content": self.dataset.system_prompt(problem=question)},
            {"role": "assistant", "content": solver_output},
            {"role": "user", "content": REVISE_TEMPLATE.format(target=target)},
        ]
        for candidate, flaw in history:
            messages.append({"role": "assistant", "content": candidate})
            messages.append({"role": "user", "content": CF_FLAW_FEEDBACK.format(flaw=flaw)})
        return messages

    def verifier_messages(self, question: str, solver_output: str, target: int, candidate: str) -> list:
        """Verifier judge layout (ground-truth sketch): system carries the question + target, then
        solver_output, the revise instruction, and the candidate revised problem."""
        return [
            {"role": "system", "content": CHECK_CF_SYSTEM_PROMPT.format(question=question, target=target)},
            {"role": "user", "content": solver_output},
            {"role": "assistant", "content": REVISE_TEMPLATE.format(target=target)},
            {"role": "user", "content": f"This is the revised math problem with different solution:\n{candidate}"},
        ]
