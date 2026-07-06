from harness.strategies.base_strategy import Strategy


class SolverVerifierLoop(Strategy):
    """Agentic loop: a solver proposes a candidate, an independent verifier re-solves and
    accepts or rejects it. On rejection the solver gets Tier-1 numeric feedback and retries.

    The verifier may be the same model or a distinct (stronger) ``verifier_model``; either way it
    step-checks the solver's solution and concludes yes/no. A same-model verifier shares the
    solver's blind spots, so an independent verifier is what lets the loop actually catch errors.
    """

    def __init__(self, max_loops: int = 5, max_tokens: int = 10000,
                 verifier_max_tokens: int = 10000, temperature: float | None = None,
                 verifier_model=None, verifier_accept_on_unparsed: bool = True):
        self.max_loops = max_loops
        self.max_tokens = max_tokens
        # High ceiling so a verbose verifier still reaches its `Verdict:` line instead of being
        # truncated mid-reasoning (a truncated verdict is unparseable -> handled below). Greedy
        # stops at EOS, so a terse verifier still stops early.
        self.verifier_max_tokens = verifier_max_tokens
        self.temperature = temperature
        # Optional distinct verifier model. A same-model verifier shares the solver's blind spots
        # (and at temp 0 just reproduces its reasoning), so a stronger, independent verifier is what
        # lets the loop actually catch errors. None -> verify with the solver model.
        self.verifier_model = verifier_model
        # How to treat an unparseable verdict (None): True -> accept (lean-yes, never overturn a
        # possibly-correct answer on parser noise); False -> reject and keep looping.
        self.verifier_accept_on_unparsed = verifier_accept_on_unparsed

    def _infer(self, model, messages, max_tokens=None):
        return model.inference(
            messages, max_tokens=max_tokens or self.max_tokens, temperature=self.temperature
        )

    def run(self, task, example, model) -> dict:
        verifier_model = self.verifier_model or model
        messages = task.build_messages(example, model)  # accumulating solver history
        iterations = []
        last_output = None
        solver_outputs = []    # every solver candidate so far (user turns for the verifier)
        verifier_outputs = []  # every verifier verdict so far (assistant turns for the verifier)

        for i in range(self.max_loops):
            output = self._infer(model, messages)
            last_output = output
            claimed = task.parse_answer(output)
            solver_outputs.append(output)

            # Step-checking verifier: a static system prompt with the question embedded, then the
            # accumulating solver/verifier history ending on the latest solver answer to judge.
            verifier_messages = task.build_verifier_messages(
                example, solver_outputs, verifier_outputs, verifier_model
            )
            verifier_output = self._infer(verifier_model, verifier_messages, self.verifier_max_tokens)
            verifier_outputs.append(verifier_output)
            verdict = task.verifier_verdict(verifier_output, self.verifier_accept_on_unparsed)
            verifier_says = verdict["verifier_says"]
            accepted = verdict["accept"]

            iterations.append({
                "iteration": i,
                "candidate": output,              # full raw solver generation
                "solver_solve": claimed,
                "verifier_output": verifier_output,  # full raw verifier generation
                "verifier_says": verifier_says,      # True (yes) / False (no) / None
                "verifier_reason": verdict["reason"],  # the one-line reason fed back to solver
                "verdict": "accept" if accepted else "reject",
            })

            if accepted:
                break

            # Feedback: hand the solver the verifier's verdict + one-line reason (not its full
            # reasoning dump) plus its own previous attempt (already in `messages`), and revise.
            feedback = (
                "A verifier reviewed your solution.\n"
                "Verifier verdict: INCORRECT\n"
                f"Verifier reason: {verdict['reason']}\n\n"
                "Fix that specific step and provide a corrected step-by-step solution."
            )
            messages = messages + [
                {"role": "assistant", "content": output},
                {"role": "user", "content": feedback},
            ]

        # Final answer = the last solver attempt (the accepted one, or the last on cap-hit).
        result = task.grade(example, last_output, model)
        return {
            "final_pred": result["pred"],
            "final_correct": result["correct"],
            "gen_fail": result["gen_fail"],
            "target_y_ce": result.get("target_y_ce"),
            "final_val": result.get("final_val"),
            "original_correct": result.get("original_correct"),
            "iterations": iterations,
        }
