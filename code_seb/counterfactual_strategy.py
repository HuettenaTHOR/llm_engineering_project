"""Resumable, step-streaming counterfactual loop (Option A).

``run`` is a GENERATOR that yields one record per completed step, so the runner can persist
after every step and a mid-loop abort can resume. It rehydrates loop state from ``prior_steps``
(the steps already written for this item) and yields only the *new* steps.

Roles use separate models (``models["solver"|"verifier"|"checker"]``):
  - solver  : original solve (step "solve") + CF generation (continues the solver thread)
  - verifier: the LLM judge that accepts/rejects a CF and localizes the flaw
  - checker : the independent benchmark re-solve of an ACCEPTED CF (two-fold grading)
"""
from harness.answer_extraction import extract_float
from code_seb.grading import to_int, parse_verdict, parse_reason


# Uniform per-step schema -- every line carries every key (role-irrelevant ones are None).
def _record(kind, iteration, **over) -> dict:
    base = {
        "kind": kind,
        "iteration": iteration,
        "question": None,
        "gold": None,
        "solver_original_answer": None,
        "solver_output": None,
        "target_y_ce": None,
        "candidate": None,
        "verifier_output": None,
        "verifier_says": None,
        "verifier_reason": None,
        "accepted": None,
        "final": False,
        "benchmark_resolved": None,
        "final_val": None,
        "gen_fail": False,
    }
    base.update(over)
    return base


class CounterfactualStrategy:
    def __init__(self, max_loops: int = 3, max_tokens: int = 1200,
                 verifier_max_tokens: int = 320, temperature: float = 0.0):
        self.max_loops = max_loops
        self.max_tokens = max_tokens
        self.verifier_max_tokens = verifier_max_tokens
        self.temperature = temperature

    def _infer(self, model, messages, max_tokens=None) -> str:
        return model.inference(
            messages, max_tokens=max_tokens or self.max_tokens, temperature=self.temperature
        )

    def _solve(self, task, question, model) -> tuple[str, int | None]:
        """Run a plain solve; return (full text, extracted integer)."""
        out = self._infer(model, task.solve_messages(question, model))
        return out, to_int(extract_float(out))

    def run(self, task, example, models, prior_steps=None):
        prior_steps = prior_steps or []
        question = example["question"]
        gold = task.gold(example)

        solve_step = next((s for s in prior_steps if s.get("kind") == "solve"), None)
        cf_steps = sorted(
            (s for s in prior_steps if s.get("kind") == "cf"), key=lambda s: s["iteration"]
        )

        # --- step "solve": original answer f(x) -> target = f(x) + noise -------------------
        if solve_step is None:
            solver_output, fx = self._solve(task, question, models["solver"])
            if fx is None:
                # Can't form a target -> mark the item done (gen-fail) and stop.
                yield _record("solve", -1, question=question, gold=gold,
                              solver_output=solver_output, gen_fail=True, final=True)
                return
            target = fx + task.offset(example)
            solve_step = _record("solve", -1, question=question, gold=gold,
                                  solver_original_answer=fx, solver_output=solver_output,
                                  target_y_ce=target)
            yield solve_step

        if solve_step.get("final"):  # resumed onto an already-finished (gen-fail) item
            return
        target = solve_step["target_y_ce"]
        solver_output = solve_step["solver_output"]
        if any(s.get("final") for s in cf_steps):  # CF already accepted / capped previously
            return

        # --- agentic loop: CF gen -> verifier judge -> accept or feed back the flaw ---------
        history = [(s["candidate"], s["verifier_reason"]) for s in cf_steps]
        for i in range(len(cf_steps), self.max_loops):
            candidate = self._infer(
                models["solver"], task.cf_messages(question, solver_output, target, history)
            ).strip()
            verifier_output = self._infer(
                models["verifier"],
                task.verifier_messages(question, solver_output, target, candidate),
                self.verifier_max_tokens,
            )
            verdict = parse_verdict(verifier_output)  # True (yes) / False (no) / None
            reason = parse_reason(verifier_output)
            accepted = verdict is True
            is_last = i == self.max_loops - 1

            step = _record("cf", i, candidate=candidate, verifier_output=verifier_output,
                           verifier_says=verdict, verifier_reason=reason, accepted=accepted,
                           final=accepted or is_last)

            if accepted:
                # Two-fold grading: independent re-solve, OUTSIDE the solver/verifier loop.
                _, resolved = self._solve(task, candidate, models["checker"])
                step["benchmark_resolved"] = resolved
                step["final_val"] = resolved is not None and resolved == target
                yield step
                return

            if is_last:  # cap hit, never accepted
                step["final_val"] = False
            yield step
            history.append((candidate, reason))
