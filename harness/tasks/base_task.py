import re
from abc import ABC, abstractmethod

from harness.answer_extraction import extract_float
from harness.CONSTANTS import VERIFIER_SYSTEM_PROMPT


def to_int(value):
    """Normalize an extracted numeric value to an int, or None. Keeps the trace/records
    JSON-clean (47 instead of 47.0) and matches the int-comparison grading convention."""
    if value is None:
        return None
    return int(value)


def parse_verdict(text):
    """The verifier's decision, read ONLY from its structured ``Verdict: YES/NO`` line (last
    one wins). Returns None when there is no such line -- i.e. the verifier rambled or got
    truncated before committing. We deliberately do NOT scrape a loose 'yes'/'no' from the
    prose: a stray 'no' mid-reasoning would falsely reject a correct answer (verifier_verdict
    treats None as accept). The negative-lookahead guard skips the verifier echoing the instruction
    literal ``Verdict: YES or NO`` -- without it a truncated verifier that printed the template but
    not its real verdict would falsely parse the 'YES' out of 'YES or NO' and accept a wrong answer."""
    if not text:
        return None
    matches = re.findall(r"verdict\s*:?\s*\**\s*(yes|no)\b(?!\s+or\b)", text, flags=re.IGNORECASE)
    return matches[-1].lower() == "yes" if matches else None


def parse_reason(text):
    """The verifier's one-line ``Reason:`` for solver feedback. Falls back to the raw text."""
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("*").strip()
        if stripped.lower().startswith("reason:"):
            return stripped.split(":", 1)[1].strip()
    return (text or "").strip()


class BaseTask(ABC):
    """A task owns prompt construction, answer parsing and grading for one problem type.

    A Strategy (see ``strategies/``) drives a task: it asks the task to build the solver's
    messages, runs the model, then asks the task to grade the output. The verifier helpers are
    task-agnostic: the verifier is a step-checker that reviews the solver's full output.
    """

    @abstractmethod
    def build_messages(self, example, model) -> list:
        """The solver's conversation for ``example`` (system prompt + question)."""

    def build_verifier_messages(self, example, solver_outputs, verifier_outputs, model) -> list:
        """The verifier's accumulating conversation across loop iterations.

        The question is embedded in the STATIC, dataset-independent system prompt
        (``VERIFIER_SYSTEM_PROMPT``). Each solver attempt is an explicitly-framed user turn ("here
        is a solution to check") and the verifier's own past verdicts are the assistant turns,
        ending on the latest solver answer (the one to judge). ``solver_outputs`` has one more
        entry than ``verifier_outputs`` (the current candidate, not yet judged).

        The conversation already alternates user/assistant correctly and ends on a user turn, so
        no seed turn is needed. We deliberately do NOT seed the bare question as a leading
        assistant turn: an assistant "uttering" the question is nonsensical to the model and it
        burns its (short) budget puzzling over the structure instead of checking the math.
        Subclasses may override for richer tasks."""
        system = f"{VERIFIER_SYSTEM_PROMPT}\n\nProblem:\n{example['question']}"
        messages = [{"role": "system", "content": system}]
        for i, solver_output in enumerate(solver_outputs):
            messages.append({"role": "user", "content": (
                f"Solver's solution:\n{solver_output}\n\n"
                "Check it and respond in the required two-line format (Reason / Verdict)."
            )})
            if i < len(verifier_outputs):
                messages.append({"role": "assistant", "content": verifier_outputs[i]})
        return messages

    def parse_answer(self, model_output: str):
        """Extract the candidate integer the model claims, or None (gen-fail)."""
        return to_int(extract_float(model_output))

    @abstractmethod
    def grade(self, example, model_output: str, model=None) -> dict:
        """Return ``{"correct": bool|None, "gen_fail": bool, "pred": int|None}``.

        ``model`` is passed by the strategy so tasks that grade by re-running the model
        (e.g. CounterfactualTask's ``Val`` self-consistency check) can do so. Tasks that grade
        purely from the text (SolveTask) ignore it."""

    def gold(self, example):
        """The gold target for ``example``, for logging in records. None by default."""
        return None

    def verifier_verdict(self, verifier_output) -> dict:
        """Verdict from the step-checking verifier. Reject ONLY on a confident 'no'; a missing
        or unparseable verdict (None, e.g. truncated output) accepts, so we never overturn a
        possibly-correct answer on noise. Returns ``{"verifier_says", "accept", "reason"}``."""
        says_yes = parse_verdict(verifier_output)
        return {
            "verifier_says": says_yes,
            "accept": says_yes is not False,
            "reason": parse_reason(verifier_output),
        }
