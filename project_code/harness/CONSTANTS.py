SYSTEM_PROMPT="""You are an math expert who tries to solve the following questions. Use step by step answering to make no mistakes. """

VERIFIER_SYSTEM_PROMPT="""You are a careful math verifier. You are given a problem and another solver's step-by-step solution. Silently check its reasoning and arithmetic, step by step.

Be decisive and terse. Do NOT restate the whole solution and do NOT second-guess yourself in circles. Only judge NO when you are confident the final answer is wrong; if you are unsure, judge YES.

End your response with EXACTLY these two lines and nothing after them:
Reason: <one short sentence; if wrong, name the first wrong step and the correct value>
Verdict: YES or NO"""

COUNTERFACTUAL_REVISE_TEMPLATE=(
    "Now, revise the math problem so your final answer to the revised problem becomes "
    "{target}. Share the revised problem. Besides thinking, only output the revised problem. Start your output with 'Revised problem:'."
)

COUNTERFACTUAL_FLAW_FEEDBACK="This CF has a flaw. Its mistake is here: {flaw}"

# Counterfactual VERIFIER prompt: an LLM *judge* (not a re-solver). 
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
