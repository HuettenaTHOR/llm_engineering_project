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

**+ 4 OpenRouter models (added 2026-07-15, rows marked † below):** `counterfactual_config.openrouter.json`, GSM8K **n=100, `max_loops=3`**, seed 42, temp = provider default, self-verifier, same **Opus-4.8** objective grader. Raw traces: `results/counterfactual_loop_smoke_{deepseek-v4-flash,gpt-5.6-luna,gemini-3.1-flash-lite,mistral-small-2603}.jsonl` (+ `.grades.jsonl`). All 100/100, gen-rate 1.00; grades 98–100/100. These are **smoke-scale** — read with wider CIs and a shallower loop budget than the originals.

**Val** = self-consistency: the loop's final counterfactual `x_CE` is re-solved by the checker; Val=1 iff `f(x_CE) == y_CE` (the intended target answer). **Val@k** = Val achievable with a loop budget of `k` (k=1 is the single-shot baseline).

## CF-1. The loop roughly *doubles* counterfactual self-consistency for every model
| Model | solve acc (in CF run) | **Val@1 (single-shot)** | **Val@5 (loop)** | final Val (accepted) | Val 95% CI | gen |
|---|---|---|---|---|---|---|
| haiku-4.5 | 0.975 | 0.435 | **0.825** | 0.845 | 0.79–0.89 | 1.00 |
| Qwen2.5-7B | 0.945 | 0.090 | **0.175** | 0.225 | 0.17–0.29 | 0.995 |
| Llama-3.1-8B | 0.860 | 0.065 | **0.160** | 0.185 | 0.14–0.24 | 1.00 |
| deepseek-v4-flash † | 0.950 | 0.780 | **0.960** (@3) | 0.960 | 0.90–0.98 | 1.00 |
| gpt-5.6-luna † | 0.960 | 0.940 | **0.940** (@3) | 0.950 | 0.89–0.98 | 1.00 |
| gemini-3.1-flash-lite † | 0.960 | 0.570 | **0.850** (@3) | 0.870 | 0.79–0.92 | 1.00 |
| mistral-small-2603 † | 0.930 | 0.160 | **0.220** (@3) | 0.250 | 0.18–0.34 | 1.00 |

**† OpenRouter models** (added 2026-07-15 via `counterfactual_config.openrouter.json`): **smoke scale — n=100, `max_loops=3`** (so their loop column is Val@3, not Val@5), temp = provider default, self-verifier, same Opus-4.8 objective grader. Numbers are directionally solid but CIs are wider and the loop budget is shallower than the n=200/k=5 originals — read them as a capability-spread extension, not a like-for-like swap.

Val@k curve (`val_at_k` in `eval_summary.json`): the original three each roughly double Val@1→Val@5 (haiku 0.435→0.825, Qwen2.5 0.09→0.175, Llama 0.065→0.16). The OpenRouter models split: deepseek climbs hard (0.78→0.96) and gemini too (0.57→0.85), while **gpt-5.6-luna is flat (0.94→0.94)** — it accepts on iteration 1 almost every item, so the loop never fires.

**Direct RQ answer:** for a model good at GSM8K but weak at single-shot counterfactuals (all three: Val@1 ≤ 0.44), the solver+verifier loop **does** improve counterfactual self-consistency — it roughly doubles Val. This is the head-on result the solve-task matrix could only set a prior for.

## CF-2. But the *verifier's* accept signal is only trustworthy for the strong model (rubber-stamp check)
| Model | accept_rate | **verifier↔Val agreement** | mean iters to accept | read |
|---|---|---|---|---|
| haiku-4.5 | 0.83 | **0.99** | 1.87 | **calibrated** — accepts iff actually valid |
| Llama-3.1-8B | 0.31 | 0.52 | 2.31 | noisy — accept ≈ coin-flip vs Val |
| Qwen2.5-7B | 0.48 | **0.37** | 1.79 | **rubber-stamps** — accepts ~2× the valid rate (0.475 accept vs 0.225 Val) |
| deepseek-v4-flash † | 0.99 | 0.97 | 1.25 | calibrated |
| gpt-5.6-luna † | 0.96 | 0.98 | **1.00** | calibrated but **inert** — accepts on iter 1, loop never runs |
| gemini-3.1-flash-lite † | 0.86 | **0.99** | 1.42 | calibrated **to its own inflated Val** (see CF-3) |
| mistral-small-2603 † | 0.32 | 0.69 | 1.69 | noisy + under-accepts (accept 0.32 vs Val 0.25) |

*agreement = fraction of items where the verifier's accept/reject matches whether the CF actually self-consistently validated.* This is the **accept-vs-Val agreement check** flagged as the key diagnostic in "Key takeaways §4" above.

**The Val@5 gains in CF-1 are largely best-of-k, not verifier skill.** haiku's verifier is near-perfectly calibrated (0.99) so its loop both improves *and* knows when to stop; the weak models get the best-of-k lift but their accept signal is unreliable (Qwen2.5 rubber-stamps, Llama is noisy) — exactly the failure mode predicted from the solve matrix, now confirmed on the CF task. **Capability tracks CF verifier calibration even more cleanly than it did on solve** (agreement 0.99 / 0.52 / 0.37 vs solve-acc rank haiku > Qwen2.5 > Llama).

## CF-3. Objective Val_obj — the primary metric — and it *flips the story*  ·  Opus-graded 2026-07-13
`Val_obj` = an **independent Opus 4.8** grader re-solves the edited problem and judges minimality; `valid_obj = solves_to_target AND minimal_edit`, computed in code (grader never sees the target). **Grader = Opus 4.8, not Haiku**: Haiku is itself a solver here (`haiku4.5-loop`), so Haiku-grading would be self-grading — Opus is out-of-set for all three. Run: `python -m harness.haiku_grader counterfactual_config.json` (writes `results/counterfactual_loop_<run>.grades.jsonl`; 599/600 graded, 1 transient API drop). `first_valid_obj` (iteration 0) is the single-shot baseline; `valid_obj` (final candidate) is the loop — both from the *same* run, so the comparison is paired (McNemar).

| Model | **Val_obj single** | **Val_obj loop** | ΔVal [95% CI] | McNemar p | self-consistency Val | Val / Val_obj inflation |
|---|---|---|---|---|---|---|
| deepseek-v4-flash † | 0.500 | **0.643** | **+0.143** [0.074, 0.212] | **1.2e-4** ✓✓✓ | 0.960 | 1.49× |
| gpt-5.6-luna † | **0.640** | 0.630 | −0.010 [−0.030, 0.010] | 1.0 ✗ | 0.950 | 1.51× |
| haiku-4.5 | 0.405 | **0.615** | **+0.210** [0.15, 0.27] | **3e-11** ✓✓✓ | 0.845 | 1.37× |
| gemini-3.1-flash-lite † | 0.500 | 0.490 | −0.010 [−0.075, 0.055] | 1.0 ✗ | 0.870 | **1.78×** |
| mistral-small-2603 † | 0.110 | 0.150 | +0.040 [−0.007, 0.087] | 0.219 ✗ | 0.250 | 1.67× |
| Qwen2.5-7B | 0.101 | 0.121 | +0.020 [−0.014, 0.054] | 0.388 ✗ | 0.225 | **1.86×** |
| Llama-3.1-8B | 0.060 | 0.100 | +0.040 [0.010, 0.070] | 0.021 ✓ | 0.185 | **1.85×** |

*(rows ordered by Val_obj loop. † = OpenRouter smoke run, n=100/k=3.)*

Where the objective validity is lost (loop candidate, Opus judgement):
| Model | solves_to_target | minimal_edit | mean edit-dist |
|---|---|---|---|
| deepseek-v4-flash † | 0.878 | 0.714 | 0.069 |
| gpt-5.6-luna † | 0.860 | 0.730 | 0.126 |
| haiku-4.5 | 0.745 | 0.795 | 0.076 |
| gemini-3.1-flash-lite † | 0.820 | 0.580 | **0.230** |
| mistral-small-2603 † | 0.240 | 0.460 | 0.110 |
| Qwen2.5-7B | 0.226 | 0.598 | 0.129 |
| Llama-3.1-8B | 0.175 | 0.295 | 0.193 |

**Two findings that overturn CF-1's headline:**
1. **Self-consistency Val was inflated ~1.85× for the weak models** (Val 0.225/0.185 vs Val_obj 0.121/0.100), only 1.37× for haiku. Exactly as the low verifier↔Val agreement (CF-2) predicted: the weak self-checker signs off on its own objectively-invalid counterfactuals.
2. **The loop's *objective* benefit is real and large only for haiku** (+0.21, p≈3e-11). For Llama it is small but significant (+0.04, p=0.021); for **Qwen2.5 it is not significant** (+0.02, p=0.39, CI crosses 0). The "loop ≈ doubles Val" story (CF-1) is a **self-consistency artefact** — under objective grading the loop mostly helps the model that already had a calibrated verifier. The weak models fail primarily on `solves_to_target` (0.23/0.18): their edits don't actually produce the intended counterfactual answer, and the loop can't fix what the verifier can't detect.

## CF-4. Seven-model picture — calibration is necessary but *not sufficient* for an objective loop gain  ·  added 2026-07-15

The four OpenRouter models (deepseek-v4-flash, gpt-5.6-luna, gemini-3.1-flash-lite, mistral-small-2603) more than double the model set and, crucially, **break the tidy "strong ⇒ loop helps" story** from CF-3. Ranked by objective loop-gain significance:

| Model | Val_obj single → loop | ΔVal (McNemar) | fixed / broken by loop | verifier↔Val | why the loop does / doesn't help |
|---|---|---|---|---|---|
| haiku-4.5 | 0.405 → 0.615 | **+0.210** (3e-11) | 45 / 3 | 0.99 | headroom **+** calibrated verifier ⇒ real climb |
| deepseek-v4-flash † | 0.500 → 0.643 | **+0.143** (1.2e-4) | **14 / 0** | 0.97 | headroom **+** calibrated ⇒ clean monotone climb (never broke a valid one) |
| gpt-5.6-luna † | 0.640 → 0.630 | −0.010 (ns) | 0 / 1 | 0.98 | **no headroom** — best single-shot model, accepts on iter 1, loop inert |
| gemini-3.1-flash-lite † | 0.500 → 0.490 | −0.010 (ns) | 5 / 6 | 0.99 | calibrated **to inflated Val** — big edits (dist 0.23), objectively flat |
| Llama-3.1-8B | 0.060 → 0.100 | +0.040 (0.021) | ~ | 0.52 | weak, noisy verifier — tiny significant lift off a near-zero floor |
| mistral-small-2603 † | 0.110 → 0.150 | +0.040 (ns) | 5 / 1 | 0.69 | weak, under-accepts — lift not significant at n=100 |
| Qwen2.5-7B | 0.101 → 0.121 | +0.020 (ns) | ~ | 0.37 | weak, rubber-stamps — no objective gain |

Three things the new models establish that three models could not:

1. **Only two of seven models get a real objective loop gain (haiku, deepseek), and both share *two* properties: single-shot headroom AND a verifier calibrated to objective validity.** Capability alone is not the discriminator — `gpt-5.6-luna` is the single strongest model (Val_obj 0.64) yet gets **zero** loop benefit, because it accepts on iteration 1 (`mean_iters=1.0`, Val@1 = Val@3 = 0.94): no headroom, the loop never runs. Ceiling, not failure.
2. **High verifier↔Val agreement is necessary but not sufficient.** `gemini-3.1-flash-lite` is calibrated at 0.99 — as calibrated as haiku — yet its objective ΔVal is −0.01. The agreement metric scores accept-vs-*self-consistency*-Val, and that Val is itself inflated 1.78× over objective; a verifier can be perfectly consistent with a signal that is objectively wrong. Gemini's tell is edit behaviour: mean edit-distance 0.23 (highest in the set) and minimal-edit only 0.58 — it "solves to target" (0.82) by rewriting the problem rather than making the one intended change.
3. **The self-consistency→objective inflation is universal, ~1.5–1.8×, and only weakly capability-dependent.** Earlier this looked like a weak-model artefact (1.85× weak vs 1.37× haiku). With seven models it's 1.4–1.8× across the board — even the strong deepseek (1.49×) and gpt (1.51×) inflate. Self-consistency Val systematically overstates objective validity by roughly half again, regardless of capability. Trust Val_obj, never the raw accept/Val signal.

**Net:** the loop's objective value is gated by the *conjunction* of (a) room to improve at loop depth and (b) a verifier whose accept signal tracks objective — not self-consistent — validity. Only haiku and deepseek clear both bars. Every other model either has no headroom (gpt), is calibrated to the wrong target (gemini), or has an uncalibrated verifier (qwen2.5, llama, mistral).

**Revised RQ answer (7 models):** *For a model good at GSM8K but bad at single-shot counterfactuals, does a solver+verifier loop do better?* **Objectively, only when the model has loop-depth headroom *and* a verifier calibrated to objective validity — a combination that showed up in 2 of 7 models (haiku, deepseek).** A strong model at single-shot ceiling gets nothing (gpt); a model calibrated only to its own inflated self-consistency gets nothing (gemini); weak models get the self-consistency *appearance* of improvement without objective gain (qwen2.5, mistral, and — barely — llama). Capability helps but does not guarantee it. The asymmetric/independent-verifier lever (memory `harness-verifier-loop-findings`) remains the untested fix and is now the clearest way to give the weak models the calibrated-verifier property they lack.

## Pending
- **OpenRouter models are smoke-scale** (n=100, `max_loops=3`). Re-run at n=200/k=5 to tighten CIs and let deepseek/gemini use the same loop budget as the originals before treating CF-4 as final; gpt-5.6-luna's ceiling effect and gemini's flat objective ΔVal are the two results most worth confirming at full scale.
- The **slow CF config** (`counterfactual_config.slow.json`: qwen3.5-4b, phi-4) is being run on a separate machine — grade it with the same Opus grader and fold its two rows into CF-1/CF-2/CF-3 when available. (qwen3.5-4b partial: Val_obj single 0.277 → loop 0.426, +0.149, p=0.016 on n=47 — same headroom+lift shape as deepseek; confirm at full n.)
- **Untested lever:** independent/asymmetric verifier (verifier ≠ solver). CF-2/CF-3/CF-4 predict this is where a *real* objective loop gain would come from for the weak models.


Haiku<->human agreement: 26/30 = 86.7%