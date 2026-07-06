SYSTEM_PROMPT="""You are an math expert who tries to solve the following questions. Use step by step answering to make no mistakes. """

# Static verifier prompt -- the SAME for every task/dataset (not derived from the dataset).
# The verifier is shown the problem + the solver's full step-by-step output. It reasons briefly,
# then MUST end with a two-line, machine-parseable verdict. Two deliberate design choices:
#   1. Be terse. Small models that ramble run out of tokens before emitting a verdict and then
#      second-guess themselves into the wrong call (observed in the 0.8B traces).
#   2. Lean toward 'yes' when unsure, so the loop only overturns an answer it is confident is
#      wrong -- otherwise a false rejection corrupts an already-correct solution.
# NOTE: a *same-model* verifier shown the solver's own trace rubber-stamps it (a 7B acting as its
# own verifier emitted 60-char "Correct, YES" approvals and caught 0/2 errors, even when the prompt
# demanded independent re-derivation). Error-catching needs an INDEPENDENT verifier (different model,
# or hide the trace) -- see DESIGN's anti-sycophancy note.
VERIFIER_SYSTEM_PROMPT="""You are a careful math verifier. You are given a problem and another solver's step-by-step solution. Silently check its reasoning and arithmetic, step by step.

Be decisive and terse. Do NOT restate the whole solution and do NOT second-guess yourself in circles. Only judge NO when you are confident the final answer is wrong; if you are unsure, judge YES.

End your response with EXACTLY these two lines and nothing after them:
Reason: <one short sentence; if wrong, name the first wrong step and the correct value>
Verdict: YES or NO"""


# --- Counterfactual task prompts (integrated from code_seb) --------------------------------
# The counterfactual loop CONTINUES the solver thread: the solver first solves the original
# problem (under the dataset system prompt), then is asked -- as a follow-up user turn -- to
# minimally revise the problem so its OWN final answer to the revised problem becomes {target}.
# There is deliberately no separate CF *system* prompt: the solver thread already carries the
# dataset system prompt + the original solve, so the model stays in-context.

# Solver revise instruction (CF #1). Appended as a user turn onto the solver's original-solve thread.
COUNTERFACTUAL_REVISE_TEMPLATE=(
    "Now, revise the math problem so your final answer to the revised problem becomes "
    "{target}. Share the revised problem. Besides thinking, only output the revised problem. Start your output with 'Revised problem:'."
)

# Rejection feedback fed back to the solver (CF #2+). {flaw} is the verifier's one-line Reason.
COUNTERFACTUAL_FLAW_FEEDBACK="This CF has a flaw. Its mistake is here: {flaw}"

# Counterfactual VERIFIER prompt: an LLM *judge* (not a re-solver). The original question + target
# are embedded so the judge silently re-solves the revised problem and decides whether its answer
# is exactly the target, ending with the same two-line Reason/Verdict contract the harness parses
# (parse_verdict / parse_reason). The independent benchmark re-solve is done separately by the
# CHECKER role in the strategy (two-fold grading), not by this judge.
COUNTERFACTUAL_VERIFIER_SYSTEM_PROMPT="""You are a careful math verifier checking a counterfactual question edit.
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
