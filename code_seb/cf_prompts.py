"""Prompts for the resumable counterfactual loop (code_seb).

Kept local to code_seb so the shared harness/CONSTANTS.py stays untouched. The verifier prompt
is an LLM *judge* ("is the revised problem's correct answer exactly the target?") and ends with
the same two-line ``Reason: / Verdict: YES|NO`` contract the harness already parses
(``parse_verdict`` / ``parse_reason`` in harness/tasks/base_task.py)."""

# Solver -> revise instruction (CF #1). Continues the solver thread as a user turn.
REVISE_TEMPLATE = (
    "Now, revise the math problem so your final answer to the revised problem becomes "
    "{target}. Share the revised problem."
)

# Rejection feedback fed back to the solver (CF #2+). {flaw} is the verifier's Reason line.
CF_FLAW_FEEDBACK = "This CF has a flaw. Its mistake is here: {flaw}"

# Verifier judge. Question + target are embedded so the judge has everything it needs.
CHECK_CF_SYSTEM_PROMPT = """You are a careful math verifier checking a counterfactual question edit.
The ORIGINAL problem was:
{question}

Someone revised it into a new problem, claiming the correct final answer to the revised problem is exactly {target}.
Silently solve the revised problem yourself, step by step, and check two things:
1. The revised problem is well-posed and natural.
2. Its correct final answer is exactly {target}.

Be decisive and terse. Judge NO only when you are confident the revised problem's answer is NOT {target} (or it is ill-posed); if unsure, judge YES.

End your response with EXACTLY these two lines and nothing after them:
Reason: <one short sentence; if NO, name the specific flaw and the answer you actually got>
Verdict: YES or NO"""
