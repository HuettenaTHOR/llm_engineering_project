# Issues — Agentic Counterfactual Reasoning Benchmark

> Generated from [`BACKLOG.md`](BACKLOG.md). One issue per backlog story, each with **What**,
> **How** (concrete implementation hints grounded in the current skeleton), and **Acceptance
> Criteria**. Design rationale lives in [`DESIGN.md`](DESIGN.md). Work them in slice order
> (`#1 → #19`); the dependency spine is strict.
>
> Code lives under `code_fred/`. Note the import style is package-relative-ish but the entry
> point (`run_base_test.py`) imports `from dataset_folder import ...` / `from models import ...`
> (no `code_fred.` prefix), so scripts are run with `code_fred/` on `sys.path` (e.g. run from
> inside `code_fred/`). Keep that convention or fix it consistently in #1.

---

## Slice 0 — Hygiene & Correct Grading

### Issue #1 — Fix skeleton bugs (A1)
**What:** Remove the two known correctness bugs so nothing downstream is built on broken grading.

**How:**
- `code_fred/run_base_test.py:24` — replace `model_output.strip() == expected_output.strip()` with an extracted-integer comparison using the new util from #2 (`extract_int(model_output) == extract_int(expected_output)` where the gold is parsed via the dataset's `####` rule). Until #2 lands, leave a `TODO` but do not ship the string-equality grade.
- `code_fred/dataset_folder/gsm8k_dataset.py:23-28` — `postprocess_result`: the `else` branch has a bare `None` expression; change to `return None`. Also return the stripped value in the `####` branch (already does) and confirm it returns a *string*.
- `gsm8k_dataset.py:30` — `prompt_addition_for_output_tracing` is defined as a **method** but used in `run_base_test.py:18` as an **attribute** (`dataset.prompt_addition_for_output_tracing` without `()`). Pick one: make it a `@property` or call it. Recommend `@property`.

**Acceptance criteria:**
- Given a correct GSM8K answer, when graded, then it scores correct (no longer ~0%).
- Given `postprocess_result` on text without `####`, then it returns `None` (not implicitly).
- Given `run_base_test.py`, when run, then `prompt_addition_for_output_tracing` resolves without an AttributeError/`<bound method>` string leaking into the prompt.

---

### Issue #2 — 3-tier answer-extraction util (A2)
**What:** A deterministic integer extractor shared by every metric.

**How:**
- New module `code_fred/extraction.py` with `extract_int(text: str) -> int | None`.
- Tier 1: regex the integer after the **last** `####`, e.g. `r"####\s*(-?\d[\d,]*)"`; take the last match.
- Tier 2 fallback: last integer in the text — `r"-?\d[\d,]*(?:\.\d+)?"`; normalize by stripping `$`, `,`, and a trailing `.0`; `int(float(x))` only when it's integer-valued.
- Tier 3: return `None` (caller treats as `Gen`-fail).
- No LLM calls. Pure stdlib `re`.
- Add `code_fred/tests/test_extraction.py` (plain asserts or `pytest`).

**Acceptance criteria:**
- `"…#### 47"` → `47`; `"… = 50."` (no `####`) → `50`; `"$1,234"` → `1234`; `"the answer is forty"` → `None`.
- Multiple `####` → the integer after the **last** one wins.
- All hand-written cases pass.

---

### Issue #3 — Reproducibility scaffolding (A7)
**What:** Fixed seeds, deterministic HF generation, pinned deps.

**How:**
- New `code_fred/seeding.py` with `set_all_seeds(seed=42)` calling `random.seed`, `numpy.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `transformers.set_seed`.
- Ensure HF generation passes `do_sample=False` when temp==0 (see #6 / #4).
- Create `requirements.txt` at repo root pinning at least `transformers`, `torch`, `datasets`, `anthropic`, `numpy`, `scipy` (scipy for McNemar/Wilson in #7), `tqdm`. Pin to the versions currently installed (`pip freeze | grep -Ei 'transformers|torch|datasets|anthropic|numpy|scipy|tqdm'`).

**Acceptance criteria:**
- `set_all_seeds(42)` called at the start of a run; two runs of the same config produce identical JSONL (local models).
- Fresh `pip install -r requirements.txt` resolves; versions are pinned (`==`).

---

## Slice 1 — Harness Core

### Issue #4 — Model loading: Qwen ladder + Haiku (A8)
**What:** A uniform model interface across the local Qwen ladder and the Anthropic API, with correct decoding params. **Fixes the deprecated Anthropic call.**

**How:**
- `code_fred/models/__init__.py` `load_model_from_str`: replace the placeholder ids with the real ladder — `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` (all `HuggingFaceModel`). Remove the bogus `Qwen/Qwen3.5-2B` / base `Qwen2.5-0.5B` entries. Keep `claude-haiku-4-5` → `AnthropicModel`.
- `huggingface_model.py`:
  - `inference()` currently decodes `outputs[0]` which **includes the prompt tokens** — slice them off: decode only `outputs[0][inputs["input_ids"].shape[1]:]`. Otherwise extraction sees the echoed prompt.
  - Thread a `temperature`/`do_sample` param through; for temp 0 use `do_sample=False`.
  - `apply_chat_template` should use `add_generation_prompt=True`.
- `anthropic_model.py` — **rewrite to the Messages API** (current code uses the removed `client.completions.create`):
  - Client: `self.client = anthropic.Anthropic()` (reads `ANTHROPIC_API_KEY` from env; do not require `api_key` kwarg).
  - Inference: split the conversation into a top-level `system=` string + a `messages=[{"role":"user"/"assistant", ...}]` list (Anthropic takes `system` separately, NOT as a `role:"system"` message). Call `self.client.messages.create(model=self.model_name, max_tokens=max_tokens, system=system, messages=messages)` and return `next(b.text for b in resp.content if b.type=="text")`.
  - Use model id alias `claude-haiku-4-5`.
  - Drop the role-flattening `build_conversation` (it produced a single prompt string for the legacy endpoint).
- Note `base_model.py:build_conversation_from_system_prompt` returns a list with a `{"role":"system"}` entry — that shape is fine for HF chat templates but the `AnthropicModel` must lift the system entry out into the `system=` param.

**Acceptance criteria:**
- Each Qwen size string loads and runs `inference()` on the 4060 Ti (7B in fp16) returning only the **completion** (no echoed prompt).
- `load_model_from_str("claude-haiku-4-5")` returns a working model whose `inference()` hits the Messages API and returns text.
- No reference remains to `client.completions.create` or `anthropic.Client(...)`.

---

### Issue #5 — `BaseTask` + `SolveTask` (A3)
**What:** A task abstraction owning prompt construction, target generation, and grading.

**How:**
- New `code_fred/tasks/base_task.py` with `BaseTask` (abstract): `build_messages(example) -> list`, `grade(example, model_output) -> dict` (returns `{"correct": bool|None, "gen_fail": bool, "pred": int|None}`).
- `code_fred/tasks/solve_task.py` `SolveTask`: builds messages from `CONSTANTS.SYSTEM_PROMPT + dataset.prompt_addition_for_output_tracing` + the question; grades via `extraction.extract_int` against the dataset gold (`gsm8k.postprocess_result(example["answer"])`).
- Use `model.build_conversation_from_system_prompt(system, question)` to keep the conversation shape consistent.

**Acceptance criteria:**
- Given a GSM8K example, `SolveTask.build_messages` includes the system prompt + `####` instruction + question.
- Given a model output, `SolveTask.grade` returns correct/incorrect/`gen_fail` using #2; gold is the integer after `####`.

---

### Issue #6 — `Strategy`: SingleShot + SolverVerifierLoop + Verifier (A4)
**What:** Pluggable control flow so single-shot and the agentic loop share one path. **This is the core build.**

**How:**
- New `code_fred/strategies/base_strategy.py` `Strategy`: `run(task, example, model) -> record` returning a dict with the per-iteration trace.
- `strategies/single_shot.py` `SingleShot`: one `model.inference`, one grade, one trace entry.
- `strategies/verifier_loop.py` `SolverVerifierLoop(max_loops=5)`:
  - Solver: same model role; produces a candidate (answer for SolveTask; revised question `x_CE` for the counterfactual task — see #12).
  - **Verifier: independent re-solve, sees only the question + the claimed answer, NOT the solver's reasoning trace** (anti-sycophancy — see DESIGN §6.1). It re-solves and compares.
  - **Tier-1 feedback on rejection:** `"no — I solve it to <verifier_number>, not <target/claim>"`. Append to the solver's accumulating message history.
  - Early-stop on accept; otherwise loop to `max_loops`. Final answer = last solver attempt if cap hit.
  - Record `{iteration, candidate, solver_solve, verifier_number, verdict}` every iteration.
- The empty `code_fred/agent.py` `Agent` class is superseded — either delete it or make it a thin alias for `Strategy`.
- `max_loops` is a constructor arg (hand-settable — DESIGN locked this).

**Acceptance criteria:**
- `SingleShot` produces one answer + one trace entry.
- `SolverVerifierLoop` early-stops on verifier accept; on rejection passes Tier-1 numeric feedback; respects `max_loops`; cap-hit → last attempt.
- Verifier prompt contains only question + claimed answer (assert no solver reasoning is interpolated).
- Every iteration is recorded.

---

### Issue #7 — JSONL writer + config-driven runner + resumability (A5)
**What:** Decoupled persistence: generation writes traces; analysis never touches a model.

**How:**
- New `code_fred/runner.py` with a config dataclass `RunConfig(model, task, strategy, max_loops, temp, n, seed)`.
- `code_fred/io_jsonl.py`: append one JSON object per `(item × condition)` **immediately** as it completes (open in `"a"`, `json.dumps` + `\n`, flush). First line / sidecar header records the full config + `git rev-parse HEAD` + seed.
- Record schema (DESIGN §8): `item_id`, `question`, `gold`, `solver_original_answer`, `target_y_ce` (null for solve task), `iterations: [...]`, `final_val`/`final_correct`, `gen_fail`, `human_audit` (null).
- Resumability (lightweight): on startup, read existing output file, collect seen `item_id`s for this config, skip them. ~5 lines — no job queue.
- Subset selection via `dataset.get_random_subset(n, seed)` (already exists).

**Acceptance criteria:**
- A smoke run (Qwen2.5-0.5B-Instruct, SolveTask, SingleShot, n=5) writes 5 valid JSONL records with the full trace + a config/git-hash/seed header.
- Re-running the same config skips the 5 completed `item_id`s.
- Records are append-flushed (killing the process mid-run keeps completed records).

---

## Slice 2 — Metrics & Analysis

### Issue #8 — Metrics module (A6)
**What:** Turn JSONL traces into numbers + significance tests, touching no model.

**How:**
- New `code_fred/metrics.py` reading a JSONL file:
  - `Val` (counterfactual self-consistency), solve-accuracy, `Gen`-fail rate per model.
  - Wilson 95% CI (`statsmodels.stats.proportion.proportion_confint(..., method="wilson")` or a hand-rolled Wilson; if avoiding statsmodels, implement Wilson directly — it's a few lines).
  - McNemar for two **paired** conditions over the same `item_id`s — build the 2×2 contingency from per-item correctness, use `statsmodels.stats.contingency_tables.mcnemar` or `scipy.stats` exact binomial on discordant pairs.
  - `metric_at_loop_k(records, k)` derived from the per-iteration list — no re-running generation.

**Acceptance criteria:**
- Given the Slice-1 smoke JSONL, metrics compute without error.
- Given two paired toy conditions, McNemar p-value + Wilson CIs are produced.
- `metric_at_loop_k` returns a value for each `k` from stored iterations.

---

## Slice 3 — Milestone 0: Solve-Task Validation

### Issue #9 — Single-shot solve baseline, Qwen ladder (B1)
**What:** Establish baseline solve accuracy per model size.

**How:** Drive `runner.py` with `RunConfig(task=SolveTask, strategy=SingleShot, n=200, temp=0, seed=42)` for each Qwen size. Write to `results/solve_singleshot_<size>.jsonl`.

**Acceptance criteria:** 4 JSONL files (one per size), each with 200 records; per-size accuracy computed from #8. Expect the spread low (0.5B) → high (7B).

---

### Issue #10 — Solver-verifier loop, solve task (B2)
**What:** Same as #9 but `strategy=SolverVerifierLoop(max_loops=5)`.

**How:** Reuse the runner; output `results/solve_loop_<size>.jsonl`. Same items (seed 42) so it's paired with #9.

**Acceptance criteria:** 4 JSONL files with per-iteration traces; paired with #9 by `item_id`.

---

### Issue #11 — Compare + verdict (B3) — **DECISION GATE**
**What:** Does the loop lift plain solve accuracy?

**How:** Run #8 to compare single-shot vs loop per size (McNemar, Wilson). Record a short go/no-go note.

**Acceptance criteria:**
- Per-size accuracy table (single-shot vs loop) + McNemar p-values.
- Documented go/no-go: **if the loop does not improve plain solving, stop and debug the harness before Slice 4.**

---

## Slice 4 — Counterfactual Build + Reproduce the Gap

### Issue #12 — `CounterfactualTask` + seeded target generation (C1)
**What:** Implement the primary task and its `Val` grading.

**How:**
- `code_fred/tasks/counterfactual_task.py` `CounterfactualTask`:
  - Step 1: solve the original question → `y_pred` (reuse `SolveTask` solve + `extract_int`).
  - Target: `y_ce = y_pred + offset`, `offset` from a **seeded** RNG over `1..10` (positive only). Seed per item deterministically (e.g. `random.Random(seed + item_index)`).
  - Build the revision prompt: original question + "revise so the answer becomes `y_ce`".
  - `Val` grade: re-solve the produced `x_CE` with the same model; pass iff `extract_int(resolve) == y_ce`.
  - Log `original_correct = (y_pred == gold)` per item (DESIGN §4.2).
- In the loop (#6), the verifier for this task gets original question + `y_ce` + candidate `x_CE`, independently solves `x_CE`, returns yes/no + its own number.

**Acceptance criteria:**
- `y_ce = y_pred + offset(1..10)`, reproducible under a fixed seed.
- `Val` re-solves `x_CE` and compares to `y_ce`.
- `original_correct` logged per record.

---

### Issue #13 — Single-shot counterfactual, ladder — reproduce the gap (C2)
**What:** Show the single-shot `Val` drop + size trend.

**How:** `RunConfig(task=CounterfactualTask, strategy=SingleShot, n=200, temp=0, seed=42)` per size → `results/cf_singleshot_<size>.jsonl`. Compute `Val` per size with Wilson CIs (#8).

**Acceptance criteria:** Per-size `Val` reported; substantially below solve-accuracy (the gap); larger models show higher `Val`.

---

## Slice 5 — PRIMARY: Gap Recovery

### Issue #14 — Loop counterfactual, ladder (C3)
**What:** Run the agentic loop on the counterfactual task.

**How:** `strategy=SolverVerifierLoop(max_loops=5)`, same items/seed as #13 → `results/cf_loop_<size>.jsonl`.

**Acceptance criteria:** Per-iteration traces written; paired with #13 by `item_id`.

---

### Issue #15 — Primary analysis (C4) — **DECISION GATE (hypothesis verdict)**
**What:** Does the loop recover the counterfactual gap?

**How:** Paired `Val` single-shot vs loop per size (McNemar, Wilson); compute gap-recovery % = fraction of the single-shot gap closed; state whether recovery is size-dependent.

**Acceptance criteria:**
- McNemar p-values + Wilson CIs per size.
- Gap-recovery % per size.
- Recorded hypothesis verdict (both a positive result and a "loop does not recover" result are valid — framing of Slices 6/7 follows from this).

---

## Slice 6 — Secondary Analyses

### Issue #16 — #loops-vs-performance curve (D1)
**What:** `Val`-at-loop-`k` per size — **no new generation runs.**

**How:** `metric_at_loop_k` (#8) over the Slice-5 loop JSONL; plot `Val` vs `k` (1..5) per size with matplotlib.

**Acceptance criteria:** A curve per size derived purely from stored iterations; no model invoked.

---

### Issue #17 — Local→API trend confirmation (D2)
**What:** Confirm the gap + recovery trend transfers to Haiku.

**How:** Run the primary pipeline (#13/#14 equivalents) on `claude-haiku-4-5` at **n=100** (conserve $20). Compute the same metrics; note the API non-determinism caveat (Haiku is not deterministic at temp 0 — DESIGN §9).

**Acceptance criteria:** Haiku single-shot vs loop `Val` (n=100) reported and compared to the local trend; non-determinism caveat noted.

---

### Issue #18 — System-prompt sweep (D3) — **STRETCH**
**What:** Sensitivity of the result to the system prompt.

**How:** Only the **single fixed prompt** is used for all primary work (DESIGN §10). If time remains: swap in one alternate `SYSTEM_PROMPT`, re-run a small slice (e.g. 0.5B + 7B, counterfactual, single-shot + loop), compare `Val`. Verifier prompt held fixed.

**Acceptance criteria (only if attempted):** One alternate prompt run on a small slice; `Val` delta reported. Otherwise explicitly skipped.

---

## Slice 7 — Validation & Reporting

### Issue #19 — Human ground-truth audit + plots + write-up (E1–E3)
**What:** Robustness check + deliverable.

**How:**
- **E1 audit:** sample ~50 `Val`-passing counterfactual records (`final_val == True`); a small script writes a CSV/JSONL with `question`, `x_CE`, `y_ce` for hand-grading; grader fills `human_audit` (genuinely-correct vs self-consistent-but-broken). Compute the genuinely-correct fraction. (Decoupled: reads existing JSONL, no model calls.)
- **E2 plots/tables:** size-trend, gap, recovery, #loops figures + tables (matplotlib + a small table dump).
- **E3 write-up:** report stating the hypothesis verdict, size trend, audit finding; list cut items (oracle verifier, multi-model debate) as future work, and note the same-model self-verification confound (DESIGN §10).

**Acceptance criteria:**
- ~50 `Val`-passing items hand-graded; genuinely-correct fraction reported.
- Figures + tables produced from stored metrics.
- Report covers verdict + size trend + audit + future work; every claim traces to a logged result.
