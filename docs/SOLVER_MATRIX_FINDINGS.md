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

## Pending
- **CF (Val) run** (`counterfactual_config.json`: qwen2.5-7b, llama3.1-8b, haiku) — the actual RQ metric. Analyse with `python -m harness.counterfactual_evaluator` once complete; then run the accept-vs-Val agreement check.
- The **slow CF config** (`counterfactual_config.slow.json`: qwen3.5-4b, phi-4) is being run on a separate machine.
