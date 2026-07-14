"""Post-hoc Haiku grader for counterfactual runs (RQ plan C1 + C5).

Pure analysis: never loads a local model. For each FINISHED, non-gen-fail item in a run's
``results/<run>.jsonl`` this sends the ORIGINAL question + the item's final CF candidate to Claude
Haiku, which re-solves the edited problem and judges whether it is a minimal edit. The objective
validity ``valid_obj = solves_to_target AND minimal_edit`` is then computed IN CODE (Haiku is never
trusted to know the target). One line per item is appended to ``results/<run>.grades.jsonl`` --
idempotent and resumable (already-graded item_ids are skipped).

CLI: ``python -m harness.haiku_grader [counterfactual_config.json]`` grades every run in the config.
"""
import difflib
import json
import re
import sys
import time

from harness.io_jsonl import JsonlWriter, read_records, seen_item_ids
from harness.cf_config import default_out_path, load_configs
from shared_utils.record_model import reconstruct_records
from shared_utils.models.anthropic_model import AnthropicModel

DEFAULT_CONFIG = "counterfactual_config.json"
# Grader must be INDEPENDENT of every solver under test; haiku is itself a solver (haiku4.5-loop),
# so grading with it would be self-grading. Use the larger, out-of-set Opus for objective Val_obj.
GRADER_MODEL = "claude-opus-4-8"


def build_prompt(question: str, candidate: str) -> str:
    """The grading contract sent to Haiku. Asks it to (1) re-solve the EDITED problem and (2) judge
    minimality, replying with a single JSON object we parse below."""
    return (
        "You are grading a counterfactual edit of a math word problem.\n"
        "ORIGINAL: " + question + "\n"
        "EDITED: " + candidate + "\n"
        "Do two things: (1) solve the EDITED problem step by step and give its final integer "
        "answer; (2) decide if EDITED is a MINIMAL edit of ORIGINAL (same scenario, only the "
        "change needed to alter the answer). "
        'Reply with ONLY a JSON object: '
        '{"solved_answer": <int>, "minimal_edit": <true|false>, "reason": "<one line>"}.'
    )


def edit_distance(a: str, b: str) -> float:
    """Normalized edit distance in [0, 1] (0 = identical, 1 = totally different). No new deps:
    difflib's SequenceMatcher.ratio() is a normalized similarity, so 1 - ratio is the distance."""
    return 1.0 - difflib.SequenceMatcher(None, a, b).ratio()


def grades_path(out_path: str) -> str:
    return out_path.replace(".jsonl", ".grades.jsonl")


def _parse_grade(text: str) -> dict:
    """Extract the first ``{...}`` block from Haiku's reply and json.loads it (tolerates prose or a
    ```json fence around the object)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in response")
    return json.loads(m.group(0))


def _to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return None


def _grade_candidate(model: AnthropicModel, question: str, target: int, candidate: str) -> dict:
    """One Haiku call grading a single candidate, with a couple of retries."""
    dist = edit_distance(question, candidate)
    conversation = [{"role": "user", "content": build_prompt(question, candidate)}]
    err = ""
    for _ in range(3):
        try:
            # temperature omitted (None): Opus 4.8 rejects an explicit temperature. max_tokens is
            # generous because the grader asks for step-by-step work before the JSON -- a truncated
            # reply has no closing brace and parses as ungraded.
            grade = _parse_grade(model.inference(conversation, max_tokens=1024, temperature=None))
            solved = _to_int(grade.get("solved_answer"))
            minimal = bool(grade.get("minimal_edit"))
            solves = solved is not None and solved == target
            return {
                "solved_answer": solved,
                "solves_to_target": solves,
                "minimal_edit": minimal,
                "valid_obj": solves and minimal,
                "edit_distance": dist,
                "reason": str(grade.get("reason", ""))[:200],
            }
        except Exception as e:  # API error or unparseable JSON
            err = str(e)
            time.sleep(1.0)
    return {
        "solved_answer": None, "solves_to_target": False,
        "minimal_edit": None, "valid_obj": False, "edit_distance": dist,
        "reason": err, "ungraded": True,
    }


def grade_run(model: AnthropicModel, cfg) -> None:
    """Grade every finished, non-gen-fail item of one run, skipping already-graded item_ids.

    Grades BOTH the first CF candidate (iteration 0 -- identical to what a separate max_loops=1
    run would have produced, since seeding is per-item) and the final candidate the loop settled
    on. This is what lets the evaluator compute loop-vs-single-shot Val_obj from a single
    max_loops=N run instead of requiring a second, redundant single-shot CF run. Fields are named
    ``first_*`` (iteration 0) and plain (final candidate, for backwards compatibility with the
    original single-candidate schema)."""
    out_path = default_out_path(cfg)
    gpath = grades_path(out_path)
    done = seen_item_ids(gpath)
    recs = reconstruct_records(read_records(out_path))
    with JsonlWriter(gpath) as writer:
        for r in recs:
            if r.item_id in done or r.gen_fail or not r.iterations or r.target_y_ce is None:
                continue
            first_candidate = r.iterations[0].candidate
            final_candidate = r.iterations[-1].candidate
            if first_candidate is None or final_candidate is None:
                continue
            final_grade = _grade_candidate(model, r.question, r.target_y_ce, final_candidate)
            time.sleep(0.1)
            same = first_candidate == final_candidate
            first_grade = final_grade if same else _grade_candidate(
                model, r.question, r.target_y_ce, first_candidate
            )
            if not same:
                time.sleep(0.1)
            writer.append({
                "item_id": r.item_id,
                "n_iterations": len(r.iterations),
                "same_candidate": same,
                **final_grade,
                **{f"first_{k}": v for k, v in first_grade.items()},
            })


def grade_config(config_path: str = DEFAULT_CONFIG) -> None:
    model = AnthropicModel(GRADER_MODEL)
    for cfg in load_configs(config_path):
        grade_run(model, cfg)


if __name__ == "__main__":
    grade_config(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
