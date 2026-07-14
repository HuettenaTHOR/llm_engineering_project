# RQ Evaluation Plan — Implementation Spec

> **Status:** locked design, ready to implement. Produced from a grill-me decision walk.
> Sibling of [DESIGN.md](DESIGN.md). This doc is the source of truth for the changes needed to
> make the harness *answer the research question*; implement it top-to-bottom.

## Research question

> When a model is **good at a reasoning benchmark (GSM8K)** but its **single-shot generation of
> counterfactuals is bad**, can a **solver + verifier agentic loop** generate *better* counterfactuals?

Based on *"Can LLMs Explain Themselves Counterfactually?"* ([arXiv:2502.18156](https://arxiv.org/abs/2502.18156)).

The RQ is a conditional chain; each link needs its own measurement:

- **P1** — the model solves GSM8K well  → measure **solve accuracy** (gate).
- **P2** — its single-shot CF generation is bad → measure **single-shot Val** (baseline).
- **T**  — the loop beats single-shot → measure **ΔVal (loop − single-shot)**, paired.

The current data (0.5B/1.5B at ~5% solve, loop-only, self-graded) cannot address this: the models
fail P1, there is no single-shot baseline, and Val is graded by an unreliable same-model checker.

---

## Locked decisions

| # | Decision |
|---|---|
| 1 | **Validity = objective**, graded by **Claude Haiku**: valid ⇔ Haiku says the CF (a) re-solves to `y_CE` **AND** (b) is a minimal edit of the original. Log normalized edit-distance as an auditable cross-check. `Val_obj = frac(solves_to_target AND minimal_edit)`. |
| 2 | **Grading is a post-hoc pass** — generation stays 100% local + resumable; Haiku never runs in-loop. |
| 3 | **4 capable solvers:** `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3.5-4B`, `meta-llama/Llama-3.1-8B-Instruct` (gated), `microsoft/phi-4`. Solve accuracy is the explicit P1 gate. 0.5B/1.5B kept as low-reasoning contrast. |
| 4 | **Core comparison:** single-shot (`max_loops=1`) vs loop (`max_loops=3`), paired on same items/seed. Primary = **ΔVal** via McNemar on `Val_obj`; also report Val@k. |
| 5 | **Self-verifier only** — solver = verifier = checker (same local model). A null result is a valid finding. |
| 6 | **Model-default generation params** (not forced greedy). Reproducibility via fixed torch/transformers seeds, seeded **per item**. |

---

## Change list

Grouped by area. **[BLOCKER]** = the RQ is unanswerable without it. File paths are repo-relative.

### A. Generation code (required by decisions 6 & 4)

**A1 — [BLOCKER] Use model-default generation params.**
- Today `shared_utils/models/*.py::inference()` forces greedy via `do_sample = temperature > 0`;
  with `temp=0.0` that is always greedy.
- Make `temperature=None` mean "use the model's own `generation_config.json`": when `temperature is
  None`, call `model.generate(**inputs, max_new_tokens=...)` with **no** `do_sample`/`temperature`/
  `top_p`/`top_k` overrides. Keep the existing explicit-temp path for backwards compatibility.
- Thread a **nullable** `temp` through `CFRunConfig`/`RunConfig` → strategies (`CounterfactualStrategy`,
  `SolverVerifierLoop`, `SingleShot`) → `inference()`. Default the configs' `temp` to `null`.
- Touch: every `inference()` in `shared_utils/models/` (base + qwen25/qwen35/gemma/mistral/hf), the two
  strategy `_infer`/`inference` call sites, both dataclasses.

**A2 — [BLOCKER] Per-item seeding.**
- `set_all_seeds(config.seed)` runs once per run. Under sampling that makes single-shot vs loop
  diverge (they consume the RNG differently) and breaks resume determinism.
- Before each item's first generation, set a per-item seed derived from `(config.seed, item_id)`, e.g.
  `torch.manual_seed(int(hashlib.sha1(f"{seed}:{item_id}".encode()).hexdigest()[:15], 16))` (plus the
  numpy/random equivalents already in `fixed_seeds`). Add to `harness/counterfactual_runner.py::run`
  (per `example` loop) and `harness/runner.py::run`.
- Effect: item *i* starts from an identical RNG state in every condition and on resume, so the paired
  comparison and "reproducible via seeds" both hold.

### B. Runs / config (P1 + P2 + T)

**B1 — [BLOCKER] CF experiment matrix.** Rewrite `counterfactual_config.json` to 8 CF runs = 4 models ×
`{max_loops: 1, max_loops: 3}`, self-verifier, `temp: null`. Naming: `<model>-single` / `<model>-loop`.

```json
{
  "dataset": "gsm8k", "n": 200, "seed": 42, "temp": null,
  "max_tokens": 10000, "verifier_max_tokens": 10000,
  "verifier_accept_on_unparsed": true, "verifier_sees_solver_output": false,
  "runs": [
    {"name": "qwen2.5-7b-single", "max_loops": 1, "solver_model": "Qwen/Qwen2.5-7B-Instruct", "verifier_model": "Qwen/Qwen2.5-7B-Instruct", "checker_model": "Qwen/Qwen2.5-7B-Instruct"},
    {"name": "qwen2.5-7b-loop",   "max_loops": 3, "solver_model": "Qwen/Qwen2.5-7B-Instruct", "verifier_model": "Qwen/Qwen2.5-7B-Instruct", "checker_model": "Qwen/Qwen2.5-7B-Instruct"},
    {"name": "qwen3.5-4b-single", "max_loops": 1, "solver_model": "Qwen/Qwen3.5-4B", "verifier_model": "Qwen/Qwen3.5-4B", "checker_model": "Qwen/Qwen3.5-4B"},
    {"name": "qwen3.5-4b-loop",   "max_loops": 3, "solver_model": "Qwen/Qwen3.5-4B", "verifier_model": "Qwen/Qwen3.5-4B", "checker_model": "Qwen/Qwen3.5-4B"},
    {"name": "llama3.1-8b-single","max_loops": 1, "solver_model": "meta-llama/Llama-3.1-8B-Instruct", "verifier_model": "meta-llama/Llama-3.1-8B-Instruct", "checker_model": "meta-llama/Llama-3.1-8B-Instruct"},
    {"name": "llama3.1-8b-loop",  "max_loops": 3, "solver_model": "meta-llama/Llama-3.1-8B-Instruct", "verifier_model": "meta-llama/Llama-3.1-8B-Instruct", "checker_model": "meta-llama/Llama-3.1-8B-Instruct"},
    {"name": "phi4-single", "max_loops": 1, "solver_model": "microsoft/phi-4", "verifier_model": "microsoft/phi-4", "checker_model": "microsoft/phi-4"},
    {"name": "phi4-loop",   "max_loops": 3, "solver_model": "microsoft/phi-4", "verifier_model": "microsoft/phi-4", "checker_model": "microsoft/phi-4"}
  ]
}
```

Run: `python -m harness.counterfactual_runner counterfactual_config.json` (resumable).
The inline `checker_model` still writes the cheap local self-consistency `final_val` (kept as a
secondary signal); the **primary** `Val_obj` comes from the post-hoc Haiku pass (C1).

**B2 — [BLOCKER] Solve baseline (P1 gate).** Measure GSM8K solve accuracy for the 4 models via the
existing solve task. Either `python -m harness.run_base_test --model <id> --strategy single_shot --n 200`
per model, or a small `benchmark_config.json` if the config-runner is wired. Record accuracy per model;
interpret CF results only for models above the reasoning bar (e.g. ≥60%).

**B3 — Setup: Llama access.** One-time `huggingface-cli login` + accept the license on
`meta-llama/Llama-3.1-8B-Instruct`. (`phi-4` and `Qwen2.5-7B` auto-load 8-bit via `_NEEDS_8BIT`.)

### C. Analysis / metrics (the actual measurement)

**C1 — [BLOCKER] Haiku grader (NEW module `harness/haiku_grader.py`).**
- Pure analysis, no local model. Reads a run's `results/<run>.jsonl`, reconstructs per-item records,
  and for each **finished** item grades its **final** CF candidate (the `final=True` cf step's
  `candidate`; the original question is on the item's `solve` step; target is `target_y_ce`).
- One Haiku call per item → structured JSON. Suggested prompt contract:
  > *You are grading a counterfactual edit of a math word problem. ORIGINAL: {question}. EDITED:
  > {candidate}. Do two things: (1) solve the EDITED problem step by step and give its final integer
  > answer; (2) decide if EDITED is a MINIMAL edit of ORIGINAL (same scenario, only the change needed
  > to alter the answer). Reply as JSON: `{"solved_answer": <int>, "minimal_edit": <true|false>,
  > "reason": "<one line>"}`.*
  Then compute `solves_to_target = (solved_answer == target_y_ce)` in code (don't trust Haiku to know
  the target). Also compute `edit_distance` = normalized Levenshtein(question, candidate) via C5.
- Write `results/<run>.grades.jsonl`, one line per item: `{item_id, solved_answer, solves_to_target,
  minimal_edit, valid_obj, edit_distance, reason}` where `valid_obj = solves_to_target and minimal_edit`.
- **Idempotent/resumable:** skip items already in the grades file; batch with light rate-limiting;
  tolerate API errors (retry, then mark ungraded). Uses `AnthropicModel("claude-haiku-4-5")` — needs
  `ANTHROPIC_API_KEY`. Est. ≈ 1,600 calls total ≈ **$2–3**.
- CLI: `python -m harness.haiku_grader counterfactual_config.json` grades every run in the config.

**C2 — [BLOCKER] Objective Val in the evaluator.** Extend `harness/counterfactual_evaluator.py`:
load `results/<run>.grades.jsonl`, compute `Val_obj = frac(valid_obj)` with a Wilson CI as the
**primary** metric, plus components: solve-rate `frac(solves_to_target)`, minimal-rate
`frac(minimal_edit)`, and edit-distance mean/median. Keep local `final_val` as a labelled secondary.

**C3 — [BLOCKER] Paired ΔVal.** Repoint `mcnemar()` at `valid_obj` from the grades files. For each
model, report `ΔVal = Val_obj(loop) − Val_obj(single)` with its CI and the McNemar exact p, on the
items shared by the two conditions.

**C4 — P1 join table.** A summary that joins solve-accuracy (B2) with single-shot `Val_obj` and loop
`Val_obj` per model, so the P1 → P2 → T chain is legible in one place. Extend `print_report` /
`eval_summary.json`.

**C5 — Edit-distance util.** Small helper (normalized Levenshtein or token-level ratio), no new deps
(hand-rolled or `difflib.SequenceMatcher.ratio`). Used by C1.

### D. Reporting

**D1 — Final per-model table:** `solve-acc │ single-shot Val_obj [CI] │ loop Val_obj [CI] │ ΔVal
[McNemar p] │ solve%/minimal%/edit-dist │ verifier accept-rate & agreement`.
This table *is* the answer to the RQ, per model.

---

## Execution order

1. **A1 + A2** — generation params + per-item seeding.
2. **B2** — solve baseline; confirm which models clear P1.
3. **B1** — run the 8-run CF matrix (resumable).
4. **C5 + C1** — edit-distance util + Haiku grader → `*.grades.jsonl`.
5. **C2–C4** — objective Val, paired ΔVal, P1 join in the evaluator.
6. **D1** — final table.

## Open flags

- **Power:** at `n=200` with low Val the paired McNemar can be underpowered. After the first grade
  pass, check discordant-pair counts per model; bump to `n=500` if CIs are too wide.
- **Unrelated WIP:** `harness/tasks/base_task.py` has an in-progress edit (the benchmark PRIO-1
  `verifier_seed_question` seed) sitting in the working tree — orthogonal to this plan; don't let it
  get swept into these changes.
- **Secondary vs primary Val:** the inline local `final_val` is kept only as a self-consistency
  secondary; every headline number and significance test uses `Val_obj` from the Haiku grades.
