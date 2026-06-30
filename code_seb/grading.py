"""Small grading / id helpers, owned by code_seb so the counterfactual loop has no dependency
on the harness.tasks package (which the CF-task move left mid-refactor). Same conventions as
harness/tasks/base_task.py and harness/runner.py so records stay cross-compatible."""
import hashlib
import re


def to_int(value):
    """Normalize an extracted numeric value to an int, or None. Keeps records JSON-clean
    (47 not 47.0) and matches the int-comparison grading convention."""
    return None if value is None else int(value)


def parse_verdict(text):
    """The verifier's decision, read ONLY from its ``Verdict: YES/NO`` line (last one wins).
    None when absent (verifier rambled / got truncated) -- callers treat None as 'not a
    confident yes' so a stray mid-prose 'no' never falsely rejects."""
    if not text:
        return None
    matches = re.findall(r"verdict\s*:?\s*\**\s*(yes|no)\b", text, flags=re.IGNORECASE)
    return matches[-1].lower() == "yes" if matches else None


def parse_reason(text):
    """The verifier's one-line ``Reason:`` for solver feedback; falls back to the raw text."""
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("*").strip()
        if stripped.lower().startswith("reason:"):
            return stripped.split(":", 1)[1].strip()
    return (text or "").strip()


def make_item_id(example: dict) -> str:
    """Stable id from the question text, so the same item pairs across conditions/seeds."""
    return hashlib.sha1(example["question"].encode("utf-8")).hexdigest()[:12]
