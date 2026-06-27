# Execution Backlog — Agentic Counterfactual Reasoning Benchmark

> Companion to [`DESIGN.md`](DESIGN.md). The design doc is *what & why*; this is *in what order*.
> Organized as **ordered project slices** along the dependency spine **0 → 1 → 2 → 3 → 4 → 5 → 6 → 7**.
> No timeboxing — work each slice to its **exit gate**, then move on. Story points are a
> relative-complexity signal (Fibonacci), not a schedule.

## Definition of Done (per story)
- [ ] Code implemented and self-reviewed
- [ ] Unit test or smoke run passing (where applicable)
- [ ] Outputs/results written to disk and reproducible from a fixed seed
- [ ] Acceptance criteria verified
- [ ] Committed to git with the config/commit-hash captured

## Dependency spine
```
Slice 0  Hygiene & correct grading
   └─> Slice 1  Harness core (tasks + strategies + JSONL I/O)
          └─> Slice 2  Metrics & analysis layer
                 └─> Slice 3  Milestone 0: solve-task validation   [GATE: does the loop help at all?]
                        └─> Slice 4  Counterfactual build + reproduce the gap
                               └─> Slice 5  PRIMARY: gap recovery via loop   [GATE: hypothesis supported?]
                                      ├─> Slice 6  Secondary analyses
                                      └─> Slice 7  Validation & reporting
```

---

## Slice 0 — Hygiene & Correct Grading
**Goal:** trustworthy grading and a reproducible environment *before any experiment runs*.
**Exit gate:** extraction util passes its tests on hand-written cases; `requirements.txt` pins versions; the broken string-equality grading is gone.

| ID | Story | Pts | Dep |
|---|---|---|---|
| A1 | Fix skeleton bugs | 1 | — |
| A2 | 3-tier answer-extraction util | 3 | A1 |
| A7 | Reproducibility scaffolding | 2 | — |

**A1 — Fix skeleton bugs**
*As a developer, I need the known skeleton bugs fixed so that no downstream result is built on broken grading.*
- Given the current `run_base_test.py`, When grading runs, Then the full-string-equality comparison is replaced by extracted-integer comparison.
- Given `gsm8k_dataset.py` `postprocess_result`, When the answer has no `####`, Then it `return None` (not a bare `None` expression).

**A2 — 3-tier answer-extraction util**
*As a developer, I need a deterministic integer extractor so that `Val`, accuracy, and `Gen` are computable for messy small-model output.*
- Given output containing `#### 47`, When extracted, Then returns `47`.
- Given output with no `####` but a trailing `"... = 50."`, When extracted, Then returns `50` (last-integer fallback; `$`/comma/`.0` normalized).
- Given output with no parseable integer, When extracted, Then returns a `Gen`-fail sentinel (never a wrong-but-scored value).
- Given a suite of hand-written cases, When tests run, Then all pass.

**A7 — Reproducibility scaffolding**
*As a developer, I need fixed seeds and pinned deps so that any result can be reproduced.*
- Given a run, When it starts, Then `torch.manual_seed`/`transformers.set_seed(42)` are set and HF generation uses `do_sample=False`.
- Given the repo, When a fresh clone installs `requirements.txt`, Then `transformers`/`torch`/`datasets`/`anthropic` versions are pinned.

---

## Slice 1 — Harness Core
**Goal:** the engine that runs any `(model, task, strategy)` and persists full traces.
**Exit gate:** a smoke run (Qwen2.5-0.5B, solve task, single-shot, n=5) produces a valid JSONL trace file; re-running skips the 5 completed items.

| ID | Story | Pts | Dep |
|---|---|---|---|
| A8 | Model loading — Qwen ladder + Haiku | 3 | A1 |
| A3 | `BaseTask` + `SolveTask` | 3 | A2 |
| A4 | `Strategy`: `SingleShot` + `SolverVerifierLoop` + `Verifier` | 5 | A3 |
| A5 | JSONL writer + config-driven runner + resumability | 5 | A3 |

**A8 — Model loading**
*As a developer, I need a uniform model interface so that local Qwen sizes and the Haiku API are swappable by config.*
- Given a Qwen2.5 size string, When loaded, Then it runs on the 4060 Ti (7B fp16; smaller as-is) and exposes a single `inference(conversation)` call.
- Given an Anthropic model string, When loaded, Then it implements the same interface via the existing `anthropic_model.py` stub.

**A3 — `BaseTask` + `SolveTask`**
*As a developer, I need a task abstraction so that prompt-building, target-generation, and grading live behind one interface.*
- Given a GSM8K item, When `SolveTask` builds the prompt, Then it includes the fixed system prompt + `####` format instruction.
- Given a model output, When `SolveTask` grades it, Then it uses the A2 extractor and returns correct/incorrect/`Gen`-fail.

**A4 — `Strategy` (SingleShot + SolverVerifierLoop + Verifier)**
*As a developer, I need pluggable control-flow strategies so that single-shot and the agentic loop share one code path.*
- Given `SingleShot`, When run on an item, Then it produces one answer + trace record.
- Given `SolverVerifierLoop` with `max_loops=k`, When the verifier (independent re-solve, sees only question + claimed answer) accepts, Then the loop early-stops.
- Given a rejection, When feedback is built, Then it is **Tier 1** (yes/no + verifier's own computed number), and the solver receives full accumulating history.
- Given the cap is reached without acceptance, When the loop ends, Then the final answer is the last solver attempt.
- Given each iteration, When it completes, Then `{candidate, solver solve, verifier number, verdict}` is recorded.

**A5 — JSONL writer + runner + resumability**
*As a developer, I need decoupled persistence so that generation never has to be re-run for analysis.*
- Given a completed item, When it finishes, Then one JSONL record is **appended immediately** (config + git-hash + seed in the file header).
- Given a re-run over the same config, When it starts, Then item-ids already present are skipped.
- Given a config object `(model, task, strategy, max_loops, temp, n, seed)`, When invoked, Then the runner executes the matrix cell end-to-end.

---

## Slice 2 — Metrics & Analysis Layer
**Goal:** turn JSONL traces into numbers + tests, touching no model.
**Exit gate:** metrics computed from the Slice-1 smoke JSONL; McNemar + Wilson run on a toy paired example.

| ID | Story | Pts | Dep |
|---|---|---|---|
| A6 | Metrics module | 3 | A5 |

**A6 — Metrics module**
*As a researcher, I need a metrics layer so that `Val`, accuracy, `Gen`, and significance come from stored traces.*
- Given a JSONL file, When metrics run, Then `Val`, solve-accuracy, and `Gen`-fail rate are reported per model.
- Given two paired conditions (single-shot, loop), When compared, Then McNemar's test + Wilson 95% CIs are reported.
- Given per-iteration records, When requested, Then metric-at-loop-`k` is computable without re-running generation.

---

## Slice 3 — Milestone 0: Solve-Task Validation
**Goal:** prove the harness lifts *plain* accuracy before touching counterfactuals (de-risk on the easy task).
**Exit gate / DECISION:** single-shot vs loop solve-accuracy per Qwen size with McNemar. *If the loop doesn't even improve plain solving, stop and debug the harness — do not proceed to Slice 4.*

| ID | Story | Pts | Dep |
|---|---|---|---|
| B1 | Single-shot solve baseline, Qwen ladder | 2 | A4–A8 |
| B2 | Solver-verifier loop, solve task | 2 | B1 |
| B3 | Compare + verdict | 2 | B2, A6 |

**B3 — Compare + verdict**
*As a researcher, I need the solve-task comparison so that the harness is validated end-to-end.*
- Given single-shot and loop runs (n=200, seed 42, temp 0), When analyzed, Then accuracy per size + McNemar p-values are reported.
- Given the accuracy spread, When reviewed, Then it shows the expected size trend (0.5B low → 7B high).
- Given the result, When the gate is evaluated, Then a go/no-go decision for Slice 4 is recorded.

---

## Slice 4 — Counterfactual Build + Reproduce the Gap
**Goal:** implement the counterfactual task and reproduce the paper's single-shot drop.
**Exit gate:** single-shot `Val` per size shows the drop + "larger fails less" trend.

| ID | Story | Pts | Dep |
|---|---|---|---|
| C1 | `CounterfactualTask` + seeded target gen | 3 | A3 |
| C2 | Single-shot counterfactual, ladder — reproduce gap | 3 | C1, B3 |

**C1 — `CounterfactualTask` + target generation**
*As a researcher, I need the counterfactual task so that the primary experiment can run.*
- Given an item solved to `y_pred`, When a target is generated, Then `y_CE = y_pred + offset`, offset ∈ `1..10` from a **seeded** RNG.
- Given an item, When recorded, Then the original-solve correctness (`y_pred == gold`) is logged.
- Given a candidate `x_CE`, When `Val` is graded, Then the model re-solves `x_CE` and passes iff result `== y_CE`.

**C2 — Single-shot counterfactual (reproduce the gap)**
- Given single-shot runs across the ladder (n=200), When analyzed, Then `Val` is reported per size and is **substantially below** solve-accuracy (the gap).
- Given the per-size `Val`, When reviewed, Then larger models show higher `Val` (paper's trend), with Wilson CIs.

---

## Slice 5 — PRIMARY: Gap Recovery via the Loop
**Goal:** the hypothesis test — does the loop recover the single-shot counterfactual gap?
**Exit gate / DECISION:** loop vs single-shot `Val` per size, McNemar, recovery %. This is the headline result.

| ID | Story | Pts | Dep |
|---|---|---|---|
| C3 | Loop counterfactual, ladder | 3 | C2 |
| C4 | Primary analysis | 3 | C3, A6 |

**C4 — Primary analysis**
*As a researcher, I need the primary comparison so that the hypothesis is answered with statistics.*
- Given paired single-shot vs loop `Val` (same items), When compared, Then McNemar p-values + Wilson CIs are reported per size.
- Given the deltas, When summarized, Then "gap-recovery %" (fraction of the single-shot gap closed by the loop) is reported per size.
- Given all sizes, When reviewed, Then whether recovery is size-dependent is stated.

---

## Slice 6 — Secondary Analyses
**Goal:** cheap add-on findings from data you already have (+ one small API run).
**Exit gate:** #loops curve plotted; Haiku trend confirmed (or noted as divergent).

| ID | Story | Pts | Dep |
|---|---|---|---|
| D1 | #loops-vs-performance curve | 2 | C4 |
| D2 | Local→API trend (Haiku, n=100) | 3 | C4, A8 |
| D3 | *(stretch)* system-prompt variant slice | 3 | C4 |

**D1** — Given per-iteration records, When plotted, Then `Val`-at-loop-`k` is shown per size — **no new generation runs**.
**D2** — Given the primary pipeline on Haiku (n=100), When analyzed, Then the gap + recovery trend is reported and compared to local; the API non-determinism caveat is noted.
**D3 (stretch)** — Given one alternate system prompt on a small slice, When compared, Then any `Val` shift is reported. *Skip unless everything above is done.*

---

## Slice 7 — Validation & Reporting
**Goal:** robustness check + the deliverable.
**Exit gate:** report complete, all claims traceable to logged results.

| ID | Story | Pts | Dep |
|---|---|---|---|
| E1 | Human ground-truth audit (~50 `Val`-passing) | 3 | C4 |
| E2 | Plots + result tables | 3 | C4 |
| E3 | Write-up / report | 5 | C4, E1, E2 |

**E1** — Given ~50 `Val`-passing counterfactuals sampled into the audit slot, When hand-graded, Then the fraction *genuinely* correct (vs self-consistent-but-broken) is reported.
**E2** — Given the metrics outputs, When rendered, Then size-trend, gap, recovery, and #loops figures + tables are produced.
**E3** — Given all results, When written up, Then the report states the hypothesis verdict, the size trend, the audit finding, and lists cut items (oracle verifier, multi-model debate) as future work.

---

## Backlog totals
- Foundation (Slices 0–2): **22 pts**
- Experiments (Slices 3–5): **18 pts**
- Secondary + reporting (Slices 6–7): **19 pts** (+3 stretch)
- **Total ≈ 59 pts** (+3 stretch)

## Two hard decision gates (don't skip)
1. **End of Slice 3:** loop must improve *plain* solve accuracy, else debug the harness before counterfactuals.
2. **End of Slice 5:** primary hypothesis verdict — this determines whether Slice 6/7 frame a positive result or a "loop does not recover the gap" result (both are publishable findings; the framing differs).
