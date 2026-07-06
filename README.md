# Agentic Counterfactual Reasoning Benchmark

## Docs:
https://docs.google.com/document/d/1LSS8P72VihsTrwrhPN6JTTawOsQ5-5cYRF14Z679eUo/edit?tab=t.0

A study harness testing whether a **solver + verifier agentic loop** recovers the
counterfactual-generation gap that single-shot generation loses on GSM8K.

Based on *"Can LLMs Explain Themselves Counterfactually?"*
([arXiv:2502.18156](https://arxiv.org/abs/2502.18156), EMNLP 2025). Full design in
[`docs/DESIGN.md`](docs/DESIGN.md); the locked evaluation spec is in
[`docs/RQ_EVALUATION_PLAN.md`](docs/RQ_EVALUATION_PLAN.md).

---

## 1. The research question

> When a model is **good at a reasoning benchmark (GSM8K)** but its **single-shot generation of
> counterfactuals is bad**, can a **solver + verifier agentic loop** generate *better* counterfactuals?

It is a conditional chain — each link is measured separately:

| Link | Claim | Measured by |
|---|---|---|
| **P1** | the model solves GSM8K well | **solve accuracy** (gate; interpret CF results only for models above the bar, ≈60%) |
| **P2** | its *single-shot* CF generation is bad | **single-shot `Val_obj`** (`max_loops=1`) |
| **T**  | the loop beats single-shot | **ΔVal = `Val_obj`(loop) − `Val_obj`(single)**, paired |

**The counterfactual task.** Given a GSM8K problem `x` and the model's own answer `f(x)`, form a
target `y_CE = f(x) + offset` (a seeded, per-item signed integer in `[-10, +10] \ {0}`). The model
must **minimally edit** the problem so its correct answer becomes `y_CE`.

---

## 2. Roles, turns & conversations (pinned)

The counterfactual loop drives **three roles**. Solver = verifier = checker are the **same local
model** in this study (self-verifier). Prompt text lives in [`harness/CONSTANTS.py`]; message
layouts in [`harness/tasks/counterfactual_task.py`].

| Role | Job | Controls |
|---|---|---|
| **solver** | solves the original problem, then proposes the edited (counterfactual) problem, revising it on rejection | generation |
| **verifier** | LLM *judge* that re-solves the candidate and returns YES/NO + a one-line reason | **early-stop only** |
| **checker** | independently re-solves the *final* candidate to grade it locally (`final_val`) | grading (secondary) |

Grading is **decoupled** from the verdict: the verifier only decides when the loop stops; the final
candidate (accepted *or* cap-hit) is always independently re-solved.

### 2.1 Solver — turn 1: original solve

```
system:    You are a math expert ... solve step by step ...   (dataset system prompt, question embedded)
user:      <the original GSM8K question>
--------
assistant: <step-by-step solution ending in "#### <f(x)>">
```
`f(x)` is extracted from `#### <n>`; the target is `y_CE = f(x) + offset`.

### 2.2 Solver — turn 2+: counterfactual generation (continues the same thread)

```
system:    <dataset system prompt, original question>
assistant: <the solver's original solution from 2.1>
user:      Now, revise the math problem so your final answer to the revised problem becomes
           <y_CE>. Share the revised problem.
--------
assistant: <CF #1: the revised problem>            <- the candidate
```
On **rejection** (loop only), the rejected candidate + the verifier's one-line flaw are appended and
the solver retries — so CF #2 sees:
```
... (as above) ...
assistant: <CF #1 candidate>
user:      This CF has a flaw. Its mistake is here: <verifier reason>
--------
assistant: <CF #2: revised problem>
```

### 2.3 Verifier — the judge (two selectable layouts)

The verifier system prompt always embeds the **original question + target** and demands a
two-line machine-parseable verdict:
```
Reason: <one short sentence; if NO, name the specific flaw and the answer you actually got>
Verdict: YES or NO
```

**Blind** (`verifier_sees_solver_output: false`, the default) — the judge re-solves the candidate
from scratch, never anchored on the solver's reasoning:
```
system:  You are a careful math verifier ... ORIGINAL: <question> ... claimed answer is exactly <y_CE> ...
user:    This is the revised math problem. Check whether its correct final answer is exactly <y_CE>:
         <candidate>
--------
assistant: ... Reason: ...  /  Verdict: YES|NO
```

**Trace-aware** (`verifier_sees_solver_output: true`) — additionally replays the solver's original
solve (4-turn layout). ⚠️ A *same-model* verifier shown its own trace rubber-stamps it (see the note
in `CONSTANTS.py`), so the blind layout is the default for self-verifier runs.

### 2.4 Checker — independent re-solve (grading)

Once a candidate is terminal (accepted, or the last loop was reached), the checker runs a **plain
solve** of the *candidate* problem — identical to 2.1 but on the edited text — and:
```
final_val = (checker's re-solved answer == y_CE)
```
This is the **local, secondary** validity signal. The **primary** validity comes from the post-hoc
Haiku grader (§5).

### 2.5 Loop control flow

```
solve original -> f(x) -> target y_CE
repeat up to max_loops:
    solver: generate CF candidate (with any prior flaws as feedback)
    verifier: YES / NO / (unparseable)
    accept? -> stop early        (YES, or unparseable & verifier_accept_on_unparsed)
    last loop? -> stop (cap-hit)
on stop: checker re-solves candidate -> final_val
```
`max_loops: 1` is therefore the **single-shot baseline** (one CF generation, graded; the verdict
cannot change the outcome). `max_loops: 3` is the **agentic loop**.

---

## 3. Setup

Dependencies are pinned in [`requirements.txt`](requirements.txt) (transformers 5.x, torch 2.12
+cu126, datasets, anthropic). The project runs in the conda env **`llm_project`**:

```bash
conda activate llm_project
pip install -r requirements.txt        # first-time only
```

Run every command **from the repository root** so `harness` and `shared_utils` resolve.

### Model cache location (`HF_HOME`)

Models/datasets cache under **`E:\huggingface`** (persistent user env var, inherited by new shells):

```powershell
[Environment]::SetEnvironmentVariable('HF_HOME', 'E:\huggingface', 'User')
```

### Gated models — HuggingFace login (required for Llama)

`meta-llama/Llama-3.1-8B-Instruct` is gated. Accept the license on its model page, then log in once
(the token is stored under `HF_HOME`, so it persists):

```bash
hf auth login          # paste a read token from https://huggingface.co/settings/tokens
```

### Objective grading — Anthropic key (required for `Val_obj`)

The post-hoc grader calls Claude Haiku:

```powershell
[Environment]::SetEnvironmentVariable('ANTHROPIC_API_KEY', '<your key>', 'User')
```

---

## 4. Supported models

Any HuggingFace repo id works (unknown ids fall through to a generic standard-chat loader). These
families are wired with the correct chat template / thinking behavior in `shared_utils/models/`:

| Family | Repo ids | Notes |
|---|---|---|
| **Qwen3.5** | `Qwen/Qwen3.5-{0.8B,2B,4B,9B}` | Thinking models (`<think>` block); 0.8B non-thinking by default |
| **Qwen2.5** | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B}-Instruct` | Standard chat |
| **Llama 3.1** | `meta-llama/Llama-3.1-8B-Instruct` | Gated; auto 8-bit |
| **Gemma 4** | `google/gemma-4-{E2B,E4B}-it` | Thinking models |
| **Phi-4** | `microsoft/Phi-4-mini-instruct` (3.8B), `microsoft/phi-4` (14B) | Standard chat |
| **Mistral** | `mistralai/Ministral-3-{3B,8B}-Instruct-2512` | Native FP8 checkpoint |
| **Anthropic** | `claude-haiku-4-5` | API grader (needs `ANTHROPIC_API_KEY`) |

**16 GB VRAM fit:** models whose bf16 footprint overflows the card auto-load 8-bit (near-lossless at
greedy decode): `Qwen2.5-7B`, `Llama-3.1-8B`, `phi-4` (14B), `gemma-4-E4B-it` (the `_NEEDS_8BIT`
set). Override per model with `quantize`. Two ≥8B models cannot co-reside in 16 GB — self-verifier
runs (one shared model) fit; cross-model runs pairing two large models do not.

**The 4 study solvers** (`counterfactual_config.json`): `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3.5-4B`,
`meta-llama/Llama-3.1-8B-Instruct` (paper baseline), `microsoft/phi-4`.

---

## 5. The experiment pipeline — how to launch

Run in order. Every step is **resumable** (killing a process never loses a completed item/step).

```bash
# 0. one-time: log in for Llama (§3), set ANTHROPIC_API_KEY (§3)

# 1. P1 gate + loop-vs-single diagnostic on plain GSM8K solving (not CF generation yet):
#    8 runs = 4 models x {single_shot, verifier_loop max_loops=5}, self-verifier. The single_shot
#    row IS the P1 solve-accuracy gate; the paired verifier_loop row answers "does a 5-iteration
#    agentic loop improve on directly using the model's output?" for that model.
python -m harness.runner benchmark_config.json
#    (ad-hoc single-model spot check instead: `python -m harness.run_base_test --model <id> --strategy single_shot --n 200`)

# 2. P2 + T — the 8-run CF matrix (4 models x {single, loop}), self-verifier.
python -m harness.counterfactual_runner counterfactual_config.json

# 3. Objective grading — one Haiku call per final candidate -> results/<run>.grades.jsonl (~$2-3).
python -m harness.haiku_grader counterfactual_config.json

# 4. Metrics — Val_obj, ΔVal, McNemar, P1 join table -> results/eval_summary.json + stdout tables.
python -m harness.counterfactual_evaluator counterfactual_config.json

# 5. Audit — export ~30 items, fill human_valid in results/audit_sample.csv, then report agreement.
python -m harness.audit_sample export
#   ... fill the human_valid column by hand ...
python -m harness.audit_sample report
```

Smoke-test the CF loop end-to-end (n=2) before the full run:
```bash
python -m harness.counterfactual_runner counterfactual_config.smoke.json
```

---

## 6. Parameters

### 6.1 Counterfactual config (`counterfactual_config.json`)

Top-level keys are shared defaults; each `runs[]` entry sets its three role models and can override
any default. Unknown keys are ignored. The study config is 8 runs = 4 models × `{max_loops: 1, 3}`.

| Parameter | Study value | Meaning |
|---|---|---|
| `solver_model` / `verifier_model` / `checker_model` | same id (self-verifier) | Repo id per role; identical ids load once and are shared |
| `name` | `<model>-single` / `<model>-loop` | Output filename tag; also how single/loop are paired for ΔVal |
| `dataset` | `gsm8k` | Only `gsm8k` is supported |
| `n` | `200` | Items from the seeded subset |
| `seed` | `42` | RNG + subset + per-item offset + per-item generation seed |
| `max_loops` | `1` (single) / `3` (loop) | Max solver→verifier iterations; `1` = single-shot baseline |
| `temp` | `null` | **`null` = use each model's own `generation_config.json` defaults.** A float forces that temperature (`0.0` = greedy) |
| `max_tokens` / `verifier_max_tokens` | `10000` | Generation caps (high so thinking models don't truncate before their answer/verdict; EOS still stops short answers) |
| `verifier_accept_on_unparsed` | `true` | Unparseable verdict → accept (early-stop) rather than reject. The checker grades independently, so over-accepting is non-destructive |
| `verifier_sees_solver_output` | `false` | `false` = blind judge (re-solves independently); `true` = trace-aware judge (§2.3) |

**Reproducibility.** `set_all_seeds(seed)` fixes subset selection; then **each item is re-seeded**
from `(seed, item_id)` (`harness/fixed_seeds.py::set_item_seed`) *before its first generation*, so
item *i* starts from an identical RNG state in **every** condition and on resume — this is what makes
the single-shot vs loop comparison genuinely paired even under model-default sampling.

### 6.2 Solve baseline CLI (`harness.run_base_test`)

Drives the plain GSM8K solve task through `single_shot` and/or the solver-verifier loop, then reports
accuracy by reading the JSONL back (analysis never re-touches a model).

Key flags (`--help` for all): `--model`, `--verifier-model`, `--strategy {single_shot,verifier_loop,both}`,
`--n`, `--max-loops`, `--seed`, `--temp` (default `0.0`), `--max-tokens`, `--verifier-max-tokens`,
`--quantize 8bit`, `--verifier-quantize 8bit`, `--verifier-no-thinking`. Default model is
`Qwen/Qwen3.5-0.8B` (override with `--model` or the `BENCH_MODEL` env var). For the P1 gate use
`--strategy single_shot --n 200`.

### 6.3 Solve benchmark matrix (`benchmark_config.json`)

Same top-level-defaults + `runs[]` shape as `counterfactual_config.json`, but for `harness.runner`
(`RunConfig`) rather than the CF runner. The default study config is 8 runs = 4 models ×
`{strategy: "single_shot", strategy: "verifier_loop" with max_loops: 5}` — a direct model output
vs. a full 5-iteration solve→verify→revise loop, paired per model. (A `verifier_loop` run with
`max_loops: 1` is **not** an equivalent single-shot baseline: the verifier still runs once, but the
loop always exits after that one iteration regardless of the verdict, so it wastes a full verifier
call without ever being able to change the graded output — use `strategy: "single_shot"` for the
true no-verifier baseline instead.) `temp: 0.0` (greedy) here, deliberately matched between both
conditions for a clean comparison — distinct from the CF study's model-default (`temp: null`)
convention. Output paths for the loop rows are auto-suffixed with `_loopsN` (and `_vfy-<model>`
for a distinct verifier) so different conditions on the same model never collide or cross-resume.
Llama rows are ordered last so a missing HF login (§3) doesn't block the other 3 models. Run:
`python -m harness.runner benchmark_config.json`.

---

## 7. Outputs & metrics

### 7.1 Files per run

| File | Written by | Contents |
|---|---|---|
| `results/counterfactual_loop_<name>.jsonl` | CF runner | **one line per step** (`solve` + each `cf`); resumable trace |
| `results/counterfactual_loop_<name>.jsonl.meta.json` | CF runner | exact config, git hash, seed |
| `results/counterfactual_loop_<name>.grades.jsonl` | Haiku grader | one line per graded item (see below) |
| `results/solve_single_shot_<model>.jsonl` | solve CLI | P1 solve-baseline trace |
| `results/eval_summary.json` | evaluator | per-run metrics + per-model P1→P2→T summary |
| `results/audit_sample.csv` | audit script | ~30-item Haiku-vs-human sheet |

`grades.jsonl` line: `{item_id, solved_answer, solves_to_target, minimal_edit, valid_obj,
edit_distance, reason}` — where `valid_obj = solves_to_target AND minimal_edit`, and
`solves_to_target` is computed in code (Haiku is never trusted to know the target).

### 7.2 Metrics

| Metric | Definition | Role |
|---|---|---|
| **solve accuracy** | `f(x) == gold` on the original problem | **P1 gate** |
| **`Val_obj`** | `frac(valid_obj)` — Haiku re-solves to `y_CE` **and** calls it a minimal edit | **primary validity** |
| `solve%` / `minimal%` | the two `Val_obj` components, reported separately | diagnostics |
| `edit_distance` | `1 − difflib.SequenceMatcher.ratio(question, candidate)` | minimality cross-check |
| **ΔVal** | `Val_obj(loop) − Val_obj(single)`, paired on shared items, exact **McNemar** p | **answers T** |
| `final_val` | local checker re-solve `== y_CE` | secondary validity |
| `Gen` | fraction of items with a parseable answer (not `gen_fail`) | health |
| `Val@k` | validity achievable if the loop stopped at loop `k` | loop-depth curve |
| verifier accept-rate / agreement | how often the verifier accepts, and how often accepts actually validate | verifier-quality |

All rates carry Wilson 95% CIs. Scores are over **finished** items, so a partial run still reports.

The **final per-model table** (evaluator stdout + `eval_summary.json`) joins: `solve-acc │ single-shot
Val_obj [CI] │ loop Val_obj [CI] │ ΔVal [McNemar p] │ solve%/minimal%/edit-dist │ verifier accept &
agreement`. That table *is* the answer to the RQ, per model.

---

## 8. Viewing results

A zero-dependency local web UI browses every `results/*.jsonl` trace, including full per-iteration
solver/verifier output:

```bash
cd harness
python webui.py --results ../results        # http://127.0.0.1:8000
```

---

## 9. Layout

```
harness/
  counterfactual_runner.py     # primary: 3-role step-streaming resumable runner
  counterfactual_evaluator.py  # metrics: Val_obj, ΔVal/McNemar, P1 join, per-model table
  haiku_grader.py              # post-hoc objective grader -> *.grades.jsonl
  audit_sample.py              # ~30-item Haiku-vs-human audit (export / report)
  cf_config.py                 # CFRunConfig + counterfactual_config.json loader
  run_base_test.py             # solve-task CLI (P1 gate; single_shot / verifier_loop)
  runner.py                    # config-driven solve runner (RunConfig)
  fixed_seeds.py               # set_all_seeds + per-item set_item_seed
  strategies/                  # single_shot, verifier_loop, counterfactual_loop
  tasks/                       # SolveTask, CounterfactualTask (message builders + grading)
  CONSTANTS.py                 # all solver/verifier prompt text
  webui.py                     # local results browser
shared_utils/
  models/                      # per-family wrappers + load_model_from_str()
  record_model.py              # typed record DTO + reconstruct_records()
  dataset_folder/              # GSM8K loader
docs/                          # DESIGN.md, RQ_EVALUATION_PLAN.md, BACKLOG.md, ISSUES.md
results/                       # *.jsonl traces + *.grades.jsonl + *.meta.json sidecars
counterfactual_config.json         # 8-run study matrix (4 models x single/loop)
counterfactual_config.smoke.json   # quick smoke config (n=2)
```
