import random

from harness.tasks.base_task import BaseTask, to_int
from harness.answer_extraction import extract_float
from harness.CONSTANTS import (
    COUNTERFACTUAL_SYSTEM_PROMPT,
    COUNTERFACTUAL_REVISE_TEMPLATE,
    COUNTERFACTUAL_VERIFIER_SYSTEM_PROMPT,
)


class CounterfactualTask(BaseTask):
    """Counterfactual generation (ported from code_seb's generator + verifier).

    The model edits the original GSM8K problem so that its OWN answer to the revised problem
    becomes a target ``y_CE = y_pred + offset`` (offset seeded per item). The verifier re-solves
    the revised problem and the answer is graded by ``Val`` self-consistency: pass iff re-solving
    the produced ``x_CE`` yields ``y_CE`` (DESIGN 4.3 / ISSUES #12).

    State note: a run processes items strictly one at a time (the strategy calls
    ``build_messages`` -> [verifier steps] -> ``grade`` for one item before the next), so we cache
    the per-item solve/target in ``self._state`` and track the in-flight item in ``self._current``.
    """

    def __init__(self, dataset, seed: int = 42, offset_low: int = 1, offset_high: int = 10):
        self.dataset = dataset
        self.seed = seed
        self.offset_low = offset_low
        self.offset_high = offset_high
        self._state = {}      # question -> {"y_pred": int|None, "target": int|None}
        self._current = None  # the state dict for the item currently being processed

    def _solve(self, question: str, model) -> int | None:
        """Solve a plain math question with the same model; return the extracted integer."""
        messages = model.build_conversation_from_system_prompt(
            self.dataset.system_prompt(problem=question), user_input=question
        )
        return to_int(extract_float(model.inference(messages)))

    def _offset(self, example) -> int:
        """Seeded, deterministic per-item offset in [offset_low, offset_high] (positive only)."""
        rng = random.Random(f"{self.seed}:{example['question']}")
        return rng.randint(self.offset_low, self.offset_high)

    def _prepare(self, example, model) -> dict:
        """Solve the original to y_pred and fix the target y_CE = y_pred + offset (once/item)."""
        key = example["question"]
        if key not in self._state:
            y_pred = self._solve(key, model)
            target = None if y_pred is None else y_pred + self._offset(example)
            self._state[key] = {"y_pred": y_pred, "target": target}
        self._current = self._state[key]
        return self._current

    def build_messages(self, example, model) -> list:
        state = self._prepare(example, model)
        user = COUNTERFACTUAL_REVISE_TEMPLATE.format(
            question=example["question"], target=state["target"]
        )
        return model.build_conversation_from_system_prompt(
            COUNTERFACTUAL_SYSTEM_PROMPT, user_input=user
        )

    def build_verifier_messages(self, example, solver_output, model) -> list:
        """Verifier = re-solve the model's revised problem (the solver's full output) and report a
        ``#### <number>`` answer; ``verifier_verdict`` then checks that number against the target."""
        return model.build_conversation_from_system_prompt(
            COUNTERFACTUAL_VERIFIER_SYSTEM_PROMPT, user_input=solver_output
        )

    def parse_answer(self, model_output: str):
        """The solver's output is a revised *question*, not a claimed answer -- nothing to parse."""
        return None

    def verifier_verdict(self, verifier_output) -> dict:
        """Accept iff the verifier's re-solved answer equals the target y_CE (Val check)."""
        resolved = to_int(extract_float(verifier_output))
        target = self._current["target"] if self._current else None
        accept = target is not None and resolved == target
        return {
            "verifier_says": accept,
            "accept": accept,
            "reason": f"re-solved the revised problem to {resolved}; target is {target}",
            "resolved": resolved,
        }

    def gold(self, example):
        """Original GSM8K gold integer (for logging + original_correct slicing)."""
        return to_int(extract_float(self.dataset.postprocess_result(example["answer"])))

    def grade(self, example, model_output: str, model=None) -> dict:
        """``Val`` self-consistency: re-solve the produced revised problem (``model_output``) and
        pass iff it yields the target y_CE."""
        state = self._prepare(example, model)
        target = state["target"]
        resolved = self._solve(model_output, model) if model is not None else None
        val = target is not None and resolved == target
        gold = self.gold(example)
        return {
            "correct": val,
            "gen_fail": target is None or resolved is None,
            "pred": resolved,
            "target_y_ce": target,
            "final_val": val,
            "original_correct": None if state["y_pred"] is None else state["y_pred"] == gold,
        }
