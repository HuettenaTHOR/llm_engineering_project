import re
from abc import ABC, abstractmethod

from answer_extraction import extract_float
from CONSTANTS import VERIFIER_SYSTEM_PROMPT


def to_int(value):
    """Normalize an extracted numeric value to an int, or None. Keeps the trace/records
    JSON-clean (47 instead of 47.0) and matches the int-comparison grading convention."""
    if value is None:
        return None
    return int(value)


def parse_yes_no(text):
    """The verifier's decision = the LAST standalone 'yes'/'no' in its output. None if absent."""
    matches = re.findall(r"\b(yes|no)\b", text or "", flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].lower() == "yes"


class BaseTask(ABC):
    """A task owns prompt construction, answer parsing and grading for one problem type.

    A Strategy (see ``strategies/``) drives a task: it asks the task to build the solver's
    messages, runs the model, then asks the task to grade the output. The verifier helpers are
    task-agnostic: the verifier is a step-checker that reviews the solver's full output.
    """

    @abstractmethod
    def build_messages(self, example, model) -> list:
        """The solver's conversation for ``example`` (system prompt + question)."""

    def build_verifier_messages(self, example, solver_output, model) -> list:
        """The verifier's conversation. Uses a STATIC, dataset-independent system prompt
        (``VERIFIER_SYSTEM_PROMPT``) and shows it the problem + the solver's full output, so it
        can check the steps and conclude with yes/no. Subclasses may override for richer tasks."""
        verifier_input = (
            f"Problem:\n{example['question']}\n\n"
            f"Proposed step-by-step solution from the solver:\n{solver_output}\n\n"
            f"Check the solver's reasoning step by step, then end with 'yes' or 'no'."
        )
        return model.build_conversation_from_system_prompt(
            VERIFIER_SYSTEM_PROMPT, user_input=verifier_input
        )

    def parse_answer(self, model_output: str):
        """Extract the candidate integer the model claims, or None (gen-fail)."""
        return to_int(extract_float(model_output))

    @abstractmethod
    def grade(self, example, model_output: str) -> dict:
        """Return ``{"correct": bool|None, "gen_fail": bool, "pred": int|None}``."""

    def gold(self, example):
        """The gold target for ``example``, for logging in records. None by default."""
        return None

    def verifier_verdict(self, verifier_output) -> dict:
        """Verdict from the step-checking verifier: accept iff it concludes 'yes'.
        Returns ``{"verifier_says": bool|None, "accept": bool}``."""
        says_yes = parse_yes_no(verifier_output)
        return {"verifier_says": says_yes, "accept": says_yes is True}
