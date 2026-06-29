SYSTEM_PROMPT="""You are an math expert who tries to solve the following questions. Use step by step answering to make no mistakes. """

# Static verifier prompt -- the SAME for every task/dataset (not derived from the dataset).
# The verifier is shown the problem + the solver's full step-by-step output. It reasons briefly,
# then MUST end with a two-line, machine-parseable verdict. Two deliberate design choices:
#   1. Be terse. Small models that ramble run out of tokens before emitting a verdict and then
#      second-guess themselves into the wrong call (observed in the 0.8B traces).
#   2. Lean toward 'yes' when unsure, so the loop only overturns an answer it is confident is
#      wrong -- otherwise a false rejection corrupts an already-correct solution.
VERIFIER_SYSTEM_PROMPT="""You are a careful math verifier. You are given a problem and another solver's step-by-step solution. Silently check its reasoning and arithmetic, step by step.

Be decisive and terse. Do NOT restate the whole solution and do NOT second-guess yourself in circles. Only judge NO when you are confident the final answer is wrong; if you are unsure, judge YES.

End your response with EXACTLY these two lines and nothing after them:
Reason: <one short sentence; if wrong, name the first wrong step and the correct value>
Verdict: YES or NO"""


# --- Counterfactual task prompts (ported from code_seb) -----------------------------------
# The counterfactual SOLVER edits a math word problem minimally so that its OWN final answer to
# the revised problem becomes a given target value. Carried over from code_seb's
# GMS8KCounterfactualGenerator.prompt_template, adapted to the message-list interface.
COUNTERFACTUAL_SYSTEM_PROMPT="""You are a math professor who edits GSM8K word problems. Given a base question and a target answer, you minimally revise the question so that the correct final answer to the revised question is exactly the target. Keep the problem natural and well-posed; change as little as possible."""

# User instruction embedding the original question + the target (y_CE). The model must share the
# revised problem text. End-format kept simple so the revised problem can be lifted from the reply.
COUNTERFACTUAL_REVISE_TEMPLATE="""Base question:
{question}

Now, revise the math problem so that the correct final answer to the revised problem becomes {target}. Share ONLY the revised problem."""

# Counterfactual VERIFIER prompt: re-solve the revised problem and report the final answer in the
# strict #### format, so we can check whether it actually equals the target (Val self-consistency).
# Ported from code_seb's verification_prompt_template.
COUNTERFACTUAL_VERIFIER_SYSTEM_PROMPT="""You are a math solver. Solve the following math problem step by step. At the end of your solution, write your final numerical answer on a new line in the format: #### <number>"""
