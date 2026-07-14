# Solver-Matrix Findings (P1 gate + self-verifier behaviour)

**Date:** 2026-07-11 · **Dataset:** GSM8K, n=200, seed 42 · **Task:** `solve`, strategy `verifier_loop`, `max_loops=5`, **self-verifier** (solver = verifier), `max_tokens = verifier_max_tokens = 10000`, `temperature = 1.0`.

## Research question this speaks to
Locked RQ (see [`docs/RQ_EVALUATION_PLAN.md`](RQ_EVALUATION_PLAN.md), memory `rq-evaluation-plan`):
> *For a model good at GSM8K but bad at single-shot counterfactuals, does a solver+verifier loop do better?*

This document covers the **solver matrix** = the P1 solve-accuracy gate **and** a direct measurement of self-verifier-loop behaviour (the harness "Milestone 0" sanity check). The **counterfactual (Val) run — the actual RQ metric — was still running when this was written**; see "Pending" below.

## Where to see the raw results
| Model | Raw trace file (one JSON line per item, full per-iteration trace) |
|---|---|
| Qwen2.5-7B-Instruct | `results/solve_verifier_loop_Qwen2.5-7B-Instruct_loops5.jsonl` |
| Qwen3.5-4B | `results/solve_verifier_loop_Qwen3.5-4B_loops5.jsonl` |
| Llama-3.1-8B-Instruct | `results/solve_verifier_loop_Llama-3.1-8B-Instruct_loops5.jsonl` |
| claude-haiku-4-5 | `results/solve_verifier_loop_claude-haiku-4-5_loops5.jsonl` |

Each record: `item_id, question, gold, iterations[], final_correct, gen_fail, …`; each iteration:
`iteration, candidate` (raw solver text), `solver_solve` (parsed answer), `verifier_output` (raw),
`verifier_says` (True/False/None), `verifier_reason`, `verdict` (accept/reject). Run metadata
(config, git hash, seed) is in the sibling `*.jsonl.meta.json`.

**Reproduce every number and trace below:**
```
python scripts/analyze_solver_matrix.py
```
**Pull one item's full trace** (example, the rubber-stamp case):
```
grep '"item_id": "cac066e65ca9"' results/solve_verifier_loop_Qwen3.5-4B_loops5.jsonl | python -m json.tool
```

## 1. Solve accuracy — the P1 gate (all 4 models qualify)
| Model | Solve acc (final) | Wilson 95% CI |
|---|---|---|
| haiku-4.5 | **97.5%** | 0.94–0.99 |
| Qwen2.5-7B | 92.0% | 0.87–0.95 |
| Qwen3.5-4B | 89.0% | 0.84–0.93 |
| Llama-3.1-8B | 76.5% | 0.70–0.82 |

## 2. Effect of the self-verifier loop (1st attempt → final)
| Model | 1st-attempt acc | → final acc | items looped | recover | regress | net |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 92.0% | 92.0% | 9 | 1 | 1 | 0 (wash) |
| Qwen3.5-4B | 89.0% | 89.0% | **0** | 0 | 0 | 0 (inert) |
| Llama-3.1-8B | **85.0%** | **76.5%** | 105 | 5 | **22** | **−8.5 pts** |
| haiku-4.5 | 96.5% | 97.5% | 4 | 2 | 0 | +1.0 |

*recover = 1st-attempt wrong → final right (loop fixed it); regress = 1st-attempt right → final wrong (loop broke it).*

**The loop is net-zero for two models, net-negative for Llama (−8.5 pts, 22 correct answers overturned vs 5 fixed), and only marginally positive for haiku.**

## 3. Why — self-verifier behaviour on WRONG first answers
| Model | rejected (caught) | accepted (rubber-stamped) |
|---|---|---|
| Qwen3.5-4B | 0/22 | **22/22** (never rejects anything; 200/200 accept on iter 0) |
| Qwen2.5-7B | 4/16 | 12/16 |
| haiku-4.5 | 3/7 | 4/7 |
| Llama-3.1-8B | **24/29 (83% recall)** | 5/29 |

Two failure modes:
- **Rubber-stamp** (Qwen3.5-4B, Qwen2.5-7B): shown its own answer, the model agrees.
- **Low-precision rejection** (Llama): catches errors well, but rejects correct answers just as
  readily, and the solver complies destructively → net harm.

## 4. Illustrative traces

### Trace A — Llama-3.1-8B: verifier rejects a *correct* answer, item ends wrong
`item efb32611fdbc` · gold **18** · final_correct **False**
> Q: "After scoring 14 points, Erin now has three times more points than Sara, who scored 8. How many points did Erin have before?"
> - iter0: ans **10** → reject ("Incorrect comparison of equation results.")
> - iter1: ans **18 ✓** → **reject** ("should be x+14=3·8, not x+14=3·8+8") ← *rejects the correct answer with its own wrong algebra*
> - iter2: ans **10** → reject ("Incorrect logic…")
> - iter3: ans **4** → **accept** ("correct logic… different approach") ← *stamps a wrong answer*

The solver had the right answer (18) at iter 1; the self-verifier rejected it and drove it to a wrong one. This is the regression mechanism in miniature.

### Trace B — Qwen3.5-4B: rubber-stamp of a wrong answer
`item cac066e65ca9` · gold **89** · solver_ans **9** · final_correct **False**
> Q: "Lorraine and Colleen are trading stickers for buttons…"
> verifier_output (end): *"…calculates the final button count as **89**. Verdict: YES … Reason: The solution correctly … calculates the final count … as **89**. Verdict: YES"*

The verifier writes **89** in its own reasoning while stamping the solver's extracted answer of **9** — pure self-agreement (and a likely `89`→`9` extraction artefact the verifier didn't notice).

## Key takeaways
1. **Capability ≠ good self-verification.** A model reviewing its own trace either agrees (sycophancy) or objects imprecisely. Confirms the prior small-model result (memory `harness-verifier-loop-findings`) **at full scale on 4 real models**.
2. **Loop value tracks verifier *precision*, not recall.** haiku (0 false rejects) is the only net-positive; Llama (high recall, low precision) is net-negative. Eager verifier + compliant solver < no loop.
3. **Detection ≠ correction.** Even when Llama's verifier flagged a wrong answer, retries rarely landed on the right one.
4. **Prior for the RQ:** if a self-verifier can't improve *solve* accuracy, the CF run must carry the burden of showing the loop helps *counterfactual* generation. Watch `accept_rate` vs `verifier_val_agreement` (in `harness/counterfactual_evaluator.py`) — high accept + low agreement ⇒ the loop is rubber-stamping CFs too. The **independent/asymmetric verifier** lever is the most likely fix.

## Caveats / provenance
- This is the **self-verifier `max_loops=5` solve task**, not the single-shot-vs-loop **CF** comparison that answers the RQ head-on (that is the pending run).
- Runtime notes for this matrix: a per-generation **`max_time=300s` cap** was added mid-run to stop Qwen3.5/Llama "overthinking" runaways (validated non-destructive — see the last-69 Llama check: 69/69 valid, 0 gen-fail). A **CUDA-OOM item-skip failsafe** is also active (Llama: 1 oom-skip). Both live in `shared_utils/models/*.py` (`max_time`) and `harness/runner.py` (`_is_oom_error`/skip). phi-4 was dropped from the solver matrix (14B doesn't fit 16 GB even in 8-bit — load crash).

---

# Counterfactual (Val) Findings — the RQ metric  ·  DONE 2026-07-13

**Config:** `counterfactual_config.json`, GSM8K n=200, seed 42, `max_loops=5`, **self-verifier** (solver = verifier = checker), `max_tokens = 10000`. Reproduce: `python -m harness.counterfactual_evaluator` (writes `results/eval_summary.json`). Raw traces: `results/counterfactual_loop_{qwen2.5-7b,llama3.1-8b,haiku4.5}-loop.jsonl`. Runtime: qwen2.5 1 oom-skip, llama 0, haiku 0 — all 200/200, gen-rate ≥0.995.

**Val** = self-consistency: the loop's final counterfactual `x_CE` is re-solved by the checker; Val=1 iff `f(x_CE) == y_CE` (the intended target answer). **Val@k** = Val achievable with a loop budget of `k` (k=1 is the single-shot baseline).

## CF-1. The loop roughly *doubles* counterfactual self-consistency for every model
| Model | solve acc (in CF run) | **Val@1 (single-shot)** | **Val@5 (loop)** | final Val (accepted) | Val 95% CI | gen |
|---|---|---|---|---|---|---|
| haiku-4.5 | 0.975 | 0.435 | **0.825** | 0.845 | 0.79–0.89 | 1.00 |
| Qwen2.5-7B | 0.945 | 0.090 | **0.175** | 0.225 | 0.17–0.29 | 0.995 |
| Llama-3.1-8B | 0.860 | 0.065 | **0.160** | 0.185 | 0.14–0.24 | 1.00 |

Val@k curve (`val_at_k` in `eval_summary.json`): each model's Val@5 is ≈2× its Val@1 —
haiku 0.435→0.825, Qwen2.5 0.09→0.175, Llama 0.065→0.16.

**Direct RQ answer:** for a model good at GSM8K but weak at single-shot counterfactuals (all three: Val@1 ≤ 0.44), the solver+verifier loop **does** improve counterfactual self-consistency — it roughly doubles Val. This is the head-on result the solve-task matrix could only set a prior for.

## CF-2. But the *verifier's* accept signal is only trustworthy for the strong model (rubber-stamp check)
| Model | accept_rate | **verifier↔Val agreement** | mean iters to accept | read |
|---|---|---|---|---|
| haiku-4.5 | 0.83 | **0.99** | 1.87 | **calibrated** — accepts iff actually valid |
| Llama-3.1-8B | 0.31 | 0.52 | 2.31 | noisy — accept ≈ coin-flip vs Val |
| Qwen2.5-7B | 0.48 | **0.37** | 1.79 | **rubber-stamps** — accepts ~2× the valid rate (0.475 accept vs 0.225 Val) |

*agreement = fraction of items where the verifier's accept/reject matches whether the CF actually self-consistently validated.* This is the **accept-vs-Val agreement check** flagged as the key diagnostic in "Key takeaways §4" above.

**The Val@5 gains in CF-1 are largely best-of-k, not verifier skill.** haiku's verifier is near-perfectly calibrated (0.99) so its loop both improves *and* knows when to stop; the weak models get the best-of-k lift but their accept signal is unreliable (Qwen2.5 rubber-stamps, Llama is noisy) — exactly the failure mode predicted from the solve matrix, now confirmed on the CF task. **Capability tracks CF verifier calibration even more cleanly than it did on solve** (agreement 0.99 / 0.52 / 0.37 vs solve-acc rank haiku > Qwen2.5 > Llama).

## CF-3. Objective Val_obj — the primary metric — and it *flips the story*  ·  Opus-graded 2026-07-13
`Val_obj` = an **independent Opus 4.8** grader re-solves the edited problem and judges minimality; `valid_obj = solves_to_target AND minimal_edit`, computed in code (grader never sees the target). **Grader = Opus 4.8, not Haiku**: Haiku is itself a solver here (`haiku4.5-loop`), so Haiku-grading would be self-grading — Opus is out-of-set for all three. Run: `python -m harness.haiku_grader counterfactual_config.json` (writes `results/counterfactual_loop_<run>.grades.jsonl`; 599/600 graded, 1 transient API drop). `first_valid_obj` (iteration 0) is the single-shot baseline; `valid_obj` (final candidate) is the loop — both from the *same* run, so the comparison is paired (McNemar).

| Model | **Val_obj single** | **Val_obj loop** | ΔVal [95% CI] | McNemar p | self-consistency Val | Val / Val_obj inflation |
|---|---|---|---|---|---|---|
| haiku-4.5 | 0.405 | **0.615** | **+0.210** [0.15, 0.27] | **3e-11** ✓✓✓ | 0.845 | 1.37× |
| Qwen2.5-7B | 0.101 | 0.121 | +0.020 [−0.014, 0.054] | 0.388 ✗ | 0.225 | **1.86×** |
| Llama-3.1-8B | 0.060 | 0.100 | +0.040 [0.010, 0.070] | 0.021 ✓ | 0.185 | **1.85×** |

Where the objective validity is lost (loop candidate, Opus judgement):
| Model | solves_to_target | minimal_edit | mean edit-dist |
|---|---|---|---|
| haiku-4.5 | 0.745 | 0.795 | 0.076 |
| Qwen2.5-7B | 0.226 | 0.598 | 0.129 |
| Llama-3.1-8B | 0.175 | 0.295 | 0.193 |

**Two findings that overturn CF-1's headline:**
1. **Self-consistency Val was inflated ~1.85× for the weak models** (Val 0.225/0.185 vs Val_obj 0.121/0.100), only 1.37× for haiku. Exactly as the low verifier↔Val agreement (CF-2) predicted: the weak self-checker signs off on its own objectively-invalid counterfactuals.
2. **The loop's *objective* benefit is real and large only for haiku** (+0.21, p≈3e-11). For Llama it is small but significant (+0.04, p=0.021); for **Qwen2.5 it is not significant** (+0.02, p=0.39, CI crosses 0). The "loop ≈ doubles Val" story (CF-1) is a **self-consistency artefact** — under objective grading the loop mostly helps the model that already had a calibrated verifier. The weak models fail primarily on `solves_to_target` (0.23/0.18): their edits don't actually produce the intended counterfactual answer, and the loop can't fix what the verifier can't detect.

**Revised RQ answer:** *For a model good at GSM8K but bad at single-shot counterfactuals, does a solver+verifier loop do better?* **Objectively, only if the verifier is already well-calibrated (i.e. a strong model).** For weak models the self-verifier loop produces the *appearance* of improvement (self-consistency) without objective gain — capability, not the loop, is what buys counterfactual validity. The asymmetric/independent-verifier lever (memory `harness-verifier-loop-findings`) remains the untested fix.

## Pending
- The **slow CF config** (`counterfactual_config.slow.json`: qwen3.5-4b, phi-4) is being run on a separate machine — grade it with the same Opus grader and fold its two rows into CF-1/CF-2/CF-3 when available.
- **Untested lever:** independent/asymmetric verifier (verifier ≠ solver). CF-2/CF-3 predict this is where a *real* objective loop gain would come from for the weak models.
