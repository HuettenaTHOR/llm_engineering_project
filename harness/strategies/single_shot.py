from harness.strategies.base_strategy import Strategy


class SingleShot(Strategy):
    """One solver call, one grade, one trace entry. The baseline control flow."""

    def __init__(self, max_tokens: int = 1200, temperature: float = 0.0):
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self, task, example, model) -> dict:
        messages = task.build_messages(example, model)
        output = model.inference(messages, max_tokens=self.max_tokens, temperature=self.temperature)
        result = task.grade(example, output, model)

        iterations = [{
            "iteration": 0,
            "candidate": output,          # full raw solver generation
            "solver_solve": result["pred"],
            "verifier_output": None,      # no verifier in single-shot
            "verifier_says": None,
            "verifier_reason": None,
            "verdict": None,
        }]
        return {
            "final_pred": result["pred"],
            "final_correct": result["correct"],
            "gen_fail": result["gen_fail"],
            "target_y_ce": result.get("target_y_ce"),
            "final_val": result.get("final_val"),
            "original_correct": result.get("original_correct"),
            "iterations": iterations,
        }
