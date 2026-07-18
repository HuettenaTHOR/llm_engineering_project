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
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("*").strip()
        if stripped.lower().startswith("reason:"):
            return stripped.split(":", 1)[1].strip()
    return (text or "").strip()


def parse_cf_candidate(text):
    if not text:
        return text
    matches = list(re.finditer(r"\**revised problem\**\s*:\s*\**", text, flags=re.IGNORECASE))
    if not matches:
        return text.strip()
    return text[matches[-1].end():].strip()


class BaseTask(ABC):

    @abstractmethod
    def build_messages(self, example, model) -> list:
        """The solver's conversation for ``example`` (system prompt + question)."""

    def build_verifier_messages(self, example, solver_outputs, verifier_outputs, model) -> list:
        system = f"{VERIFIER_SYSTEM_PROMPT}\n\nProblem:\n{example['question']}"
        messages = [{"role": "system", "content": system}]
        if getattr(self, "verifier_seed_question", False):
            messages.append({"role": "assistant", "content": example["question"]})
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
        """Return {"correct": bool|None, "gen_fail": bool, "pred": int|None}"""

    def gold(self, example):
        """The gold target for ``example``, for logging in records. None by default."""
        return None

    def verifier_verdict(self, verifier_output, accept_on_unparsed: bool = True) -> dict:
        """Verdict from the step-checking verifier. Reject on a confident 'no'; a confident 'yes'
        accepts. An unparseable verdict (None, e.g. truncated output) follows ``accept_on_unparsed``
        (default True -> accept, so we never overturn a possibly-correct answer on parser noise).
        Returns ``{"verifier_says", "accept", "reason"}``."""
        says_yes = parse_verdict(verifier_output)
        accept = says_yes is True or (says_yes is None and accept_on_unparsed)
        return {
            "verifier_says": says_yes,
            "accept": accept,
            "reason": parse_reason(verifier_output),
        }
