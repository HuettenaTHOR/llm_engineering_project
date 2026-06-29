from strategies.base_strategy import Strategy


class SolverVerifierLoop(Strategy):
    """Agentic loop: a solver proposes a candidate, an independent verifier re-solves and
    accepts or rejects it. On rejection the solver gets Tier-1 numeric feedback and retries.

    The verifier is the *same model* but sees only the question + the claimed answer -- never
    the solver's reasoning trace -- so it cannot simply agree with the solver (DESIGN 6.1).
    """

    def __init__(self, max_loops: int = 5, max_tokens: int = 1200, temperature: float = 0.0):
        self.max_loops = max_loops
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _infer(self, model, messages):
        return model.inference(messages, max_tokens=self.max_tokens, temperature=self.temperature)

    def run(self, task, example, model) -> dict:
        messages = task.build_messages(example, model)  # accumulating solver history
        iterations = []
        last_output = None

        for i in range(self.max_loops):
            output = self._infer(model, messages)
            last_output = output
            claimed = task.parse_answer(output)

            # Step-checking verifier: fresh conversation with a static system prompt, shown the
            # problem + the solver's FULL output; it checks the steps and concludes yes/no.
            verifier_messages = task.build_verifier_messages(example, output, model)
            verifier_output = self._infer(model, verifier_messages)
            verdict = task.verifier_verdict(verifier_output)
            verifier_says = verdict["verifier_says"]
            accepted = verdict["accept"]

            iterations.append({
                "iteration": i,
                "candidate": output,              # full raw solver generation
                "solver_solve": claimed,
                "verifier_output": verifier_output,  # full raw verifier generation
                "verifier_says": verifier_says,      # True (yes) / False (no) / None
                "verdict": "accept" if accepted else "reject",
            })

            if accepted:
                break

            # Feedback: hand the solver the verifier's critique and let it revise.
            feedback = (
                "A verifier reviewed your solution and judged it INCORRECT.\n\n"
                "Verifier feedback:\n" + verifier_output +
                "\n\nReconsider carefully and provide a corrected step-by-step solution."
            )
            messages = messages + [
                {"role": "assistant", "content": output},
                {"role": "user", "content": feedback},
            ]

        # Final answer = the last solver attempt (the accepted one, or the last on cap-hit).
        result = task.grade(example, last_output)
        return {
            "final_pred": result["pred"],
            "final_correct": result["correct"],
            "gen_fail": result["gen_fail"],
            "iterations": iterations,
        }
