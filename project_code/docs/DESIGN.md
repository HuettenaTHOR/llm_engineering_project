# Project Design: Agentic Counterfactual Reasoning Benchmark

> Study project (4 weeks). Benchmarking whether a **solver+verifier agentic loop**
> recovers the counterfactual-generation gap that single-shot generation loses.
> Based on *"Can LLMs Explain Themselves Counterfactually?"*
> ([arXiv:2502.18156](https://arxiv.org/abs/2502.18156), EMNLP 2025), starting from GSM8K.

---

## 1. Motivation

Math reasoning benchmarks like GSM8K are largely "solved" by modern reasoning models.
The referenced paper shows a sharper probe: when a model must produce a **counterfactual
question** — a minimally edited version of the original whose answer is a chosen target
(e.g. original answer 42 → make it 47) — models fail, often badly, and larger models fail
less. This suggests solving-by-statistical-approximation rather than genuine understanding.

Since models are increasingly deployed as **multi-turn agents**, the question this project
asks is: *does an agentic harness (a solver model paired with a verifier model in a feedback
loop) recover the accuracy/validity lost in the single-shot counterfactual setting?*

---

## 2. Constraints

| Resource | Budget |
|---|---|
| Time | 4 weeks |
| Local compute | 1× RTX 4060 Ti, 16 GB VRAM |
| API | $20 Anthropic credits |

These constraints drive every scoping decision below: one clean primary result, cheap/free
auto-grading, a single local model family, and API used only to confirm trends.

---

## 3. Primary Hypothesis

> A solver+verifier agentic loop recovers a measurable fraction of the
> counterfactual-generation gap that single-shot generation loses.

Everything outside this hypothesis is **secondary and cuttable**.

---

## 4. Task & Metric Definitions

### 4.1 Tasks
- **Solve task** (warm-up / harness sanity check): solve the original GSM8K question.
- **Counterfactual task** (primary): given the original question and a target answer
  `y_CE`, produce a minimally-edited question whose answer is `y_CE`.

### 4.2 Counterfactual target
- `y_CE = y_pred + offset`, where `y_pred` is the model's **own** original prediction
  (faithful to the paper's self-consistency framing, not the gold answer) and
  `offset` is drawn from a **seeded** RNG over `1..10` (positive only — negative targets
  are nonsensical for word problems).
- The original-solve correctness (`y_pred == gold`) is **logged per item** so all results
  can be sliced by "did the model get the original right."

### 4.3 Metrics
- **Primary — `Val` (auto-graded self-consistency):** re-run the model on its own
  counterfactual `x_CE`; pass iff `f(x_CE) == y_CE`. Free, deterministic, high-`n`,
  matches the paper. *Caveat:* measures self-consistency, **not** ground-truth correctness.
- **`Gen` (generation success):** fraction of items where an integer answer could be parsed.

### 4.4 Answer extraction (gates every metric)
Deterministic, no LLM extractor (saves budget):
1. **Primary:** enforce `#### <int>` in the prompt; regex-capture the integer after the last `####`.
2. **Fallback:** last integer in the text, normalizing `$`, commas, trailing `.0`.
3. **Parse failure:** mark as `Gen`-fail (never silently scored wrong); report the rate per model.

---

## 5. Models

| Role | Models |
|---|---|
| Local ladder | **Qwen3.5 small series: 0.8B → 2B → 4B → 9B** (suggested, not pinned) |
| API (subset only) | **Claude Haiku** — used to confirm the trend transfers, conserving the $20 |

One clean local family means model **size is the only varying variable** — exactly what the
"larger models fail less" claim needs. The ladder is the default suggestion only: the model is
chosen by config (`--model` / `BENCH_MODEL` / `RunConfig.model`), so any HuggingFace repo id —
including newer models — can be swapped in without code changes. Instruct/chat variants are
required (base models won't follow the counterfactual instruction).

---

## 6. Agentic Harness

### 6.1 Roles
- **Solver:** has full context; attempts the task.
- **Verifier:** a **step-checker**. Each iteration it is shown the problem + the solver's
  **full step-by-step output**, under a **static, dataset-independent system prompt**
  (`VERIFIER_SYSTEM_PROMPT`, the same for every task). It checks the reasoning/arithmetic and
  concludes with a single word — `yes` (correct) or `no` (incorrect).
- Primary experiment: **same model plays both roles** (isolates "does self-verification help"
  from "does a smarter friend help").
- **Tradeoff (was anti-sycophancy):** an earlier design had the verifier *independently
  re-solve* seeing only the claimed answer, to avoid sycophantically agreeing with a visible
  chain. The current design deliberately trades that away for a trace-aware critique — the
  verifier reviews the actual steps. The same-model self-verification confound + possible
  sycophancy is now an explicit limitation to surface in the write-up (§10).

### 6.2 Feedback on rejection
On a `no`, the verifier's **critique is fed back** into the solver's accumulating history
("A verifier reviewed your solution and judged it INCORRECT. Verifier feedback: …"), and the
solver revises. This gives the solver a step-level error signal, not just a number.

### 6.3 Loop control
- `max_loops` is **hand-settable** (default 5).
- **Early-stop** the moment the verifier answers `yes`.
- **Full accumulating multi-turn history** (solver sees all prior attempts + all feedback).
- **Per-iteration logging** of `{candidate, solver's solve, verifier's full output, verifier's
  yes/no, verdict}` → the "#loops vs. performance" curve falls out of a **single run**.
- If the cap is hit without acceptance, the final answer is the last solver attempt.

---

## 7. Experimental Design & Statistics

- **Sample size:** `n = 200` per condition for the local ladder (seeded subset, `seed=42`);
  full 1,319 reserved for the final 7B run only if time allows. Haiku subset: `n = 100`.
- **Pairing:** single-shot vs. loop are **paired** (same items) → use **McNemar's test**;
  report **Wilson 95% CIs** on each `Val`/accuracy proportion.
- **Temperature:** **temp = 0 (greedy)** as primary → deterministic, no repeats needed.
  (The loop still progresses at temp 0 because the prompt grows each iteration.)
  temp = 0.5 × 3 repeats is confined to a **robustness subset** only.

---

## 8. Architecture

**Core principle: decouple expensive generation from cheap analysis.** No metric ever requires
touching a model; re-running inference to fix a grading bug is forbidden by design.

```
run_generation  (GPU/API)  ──>  results/*.jsonl  ──>  compute_metrics  ──>  plots
```

- **One JSONL record per (item × condition)** with the full trace: original question, gold
  answer, solver's original answer, target `y_CE`, per-iteration list of
  `{candidate x_CE, solver's solve, verifier's number, verdict}`, final `Val`, and `Gen`-fail
  flag. A run's exact config + git hash + seed are written into the file header.
- **Resumability (lightweight):** append each record as it completes; on startup, skip
  item-ids already present. No job queue / checkpointing framework.
- **Config-driven matrix:** one config object `(model, task, strategy, max_loops, temp, n, seed)`.

### 8.1 Abstractions (extensibility goal)
- Keep `BaseModel` / `BaseDataset` (already present).
- Add **`BaseTask`** — owns prompt construction, target generation, grading:
  `SolveTask`, `CounterfactualTask`.
- Add **`Strategy`** — owns control flow: `SingleShot`, `SolverVerifierLoop`
  (the empty `Agent` class becomes / is replaced by `Strategy`).
- Every future feature (oracle verifier, debate, trace-aware verifier) is a new `Strategy`
  or `Verifier` subclass, not a rewrite.

---

## 9. Reproducibility

- HF generation: `do_sample=False` (greedy) — temp=0 alone is insufficient in `transformers`;
  also `torch.manual_seed` + `transformers.set_seed(42)`.
- Pin the 200-item subset with `seed=42`; write seed + git commit hash into each JSONL header.
- Pin library versions in `requirements.txt` (`transformers`, `torch`, `datasets`, `anthropic`).
- **Known exception:** the Anthropic API is **not** fully deterministic even at temp=0.
  For the API subset this is an accepted limitation; a couple of repeats are worth it if budget allows.

---

## 10. Scope: Secondary Questions

**Kept (in priority order):**
1. **#loops vs. performance** — free, falls out of per-iteration logging (Section 6.3).
2. **Local → API trend confirmation** — re-run the primary pipeline on Haiku (`n=100`).
3. **System-prompt impact** — *stretch only*. Default: a **single fixed system prompt** for
   all experiments (the existing CoT prompt + `####` format instruction). If time remains,
   swap in one alternate and re-run a small slice.

**Cut (listed as future work):**
- **Oracle verifier** (strong model as checker). Consequence: this project cannot directly
  disentangle "the loop helps" from "a smarter checker helps" — the same-model self-verification
  result must be framed honestly, with oracle verification named as future work.
- **Multi-model debate + moderator** — largest build, highest token cost, least certain payoff.

---

## 11. Milestones & Sequencing

1. **Fix skeleton bugs** (Section 12).
2. **Build foundation:** answer-extraction util, `BaseTask` + `Strategy`, JSONL writer.
3. **Milestone 0 — solve-task harness sanity check:** single-shot vs. loop on the Qwen ladder,
   McNemar. Validates the harness end-to-end on the simple task before the confusing one.
   *Expectation:* accuracy spreads from low (0.5B ≈ 30–40%) to high (7B ≈ 80%+); the spread
   itself is the size-trend evidence.
4. **Primary experiment — counterfactual task:** single-shot vs. loop across the ladder; `Val`
   + McNemar. Reproduce the gap, then test whether the loop recovers it.
5. **Secondaries (kept):** #loops curve → Haiku trend → (system-prompt stretch).
6. **Write-up.**

---

## 12. Known Skeleton Bugs (fix first)

- `code_fred/run_base_test.py` — grades with full-string equality
  (`model_output.strip() == expected_output.strip()`), which scores ~0% even for correct
  models. Replace with the extraction contract in Section 4.4.
- `code_fred/dataset_folder/gsm8k_dataset.py` — `postprocess_result`'s `else` branch writes
  `None` as a bare expression instead of `return None`.
