"""Counterfactual run evaluator: artifact health + benchmark scores from the per-step JSONL.

Integrated from code_seb. Pure analysis -- touches no model (DESIGN §8). Metric definitions
follow DESIGN §4.3:
  - Val : re-solved x_CE == y_CE  (stored per item as ``final_val``)  -- the primary metric
  - Gen : fraction of items with a parseable answer (not ``gen_fail``)
  - Wilson 95% CIs on each proportion; metric-at-loop-k; optional McNemar for paired runs.

All scores are computed over FINISHED items (those with a terminating ``final`` step) so a
partially-completed (resumable) run still yields meaningful rates.
"""
import json
import math
import os
import sys

from harness.io_jsonl import read_records, read_meta
from harness.cf_config import default_out_path, load_configs, _short
from shared_utils.record_model import reconstruct_records

DEFAULT_CONFIG = "counterfactual_config.json"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion k/n (no scipy dependency)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _accept_iter(record) -> int | None:
    """0-based loop index at which the verifier accepted, or None if never accepted."""
    for it in record.iterations:
        if it.verdict == "accept":
            return it.iteration
    return None


def evaluate_run(out_path: str, expected_n: int = None) -> dict:
    """Artifact-health + benchmark scores for one run's JSONL file."""
    exists = os.path.exists(out_path)
    meta = read_meta(out_path)
    if expected_n is None and meta:
        expected_n = meta.get("config", {}).get("n")
    max_loops = meta.get("config", {}).get("max_loops") if meta else None

    steps = read_records(out_path)
    by_item: dict = {}
    for s in steps:
        by_item.setdefault(s.get("item_id"), []).append(s)
    finished = {i for i, g in by_item.items() if any(x.get("final") for x in g)}
    n_incomplete = len(by_item) - len(finished)

    # Score only finished items.
    recs = [r for r in reconstruct_records(steps) if r.item_id in finished]
    n = len(recs)

    n_val = sum(1 for r in recs if r.final_val)
    n_gen_ok = sum(1 for r in recs if not r.gen_fail)
    n_gen_fail = sum(1 for r in recs if r.gen_fail)
    orig = [r.original_correct for r in recs if r.original_correct is not None]

    accept_iters = [_accept_iter(r) for r in recs]
    n_accept = sum(1 for a in accept_iters if a is not None)
    n_agree = sum(1 for r, a in zip(recs, accept_iters) if a is not None and r.final_val)

    if not max_loops:
        max_loops = max((len(r.iterations) for r in recs), default=0)
    # Val achievable if the loop stopped at k: an item accepted at iter j (0-based) counts from
    # k = j+1 onward, contributing its final_val; never-accepted items never count.
    val_at_k = {
        k: (sum(1 for r, a in zip(recs, accept_iters)
                if a is not None and a < k and r.final_val) / n if n else 0.0)
        for k in range(1, max_loops + 1)
    }
    depth_hist: dict = {}
    for a in accept_iters:
        key = "none" if a is None else a + 1
        depth_hist[key] = depth_hist.get(key, 0) + 1

    return {
        "out_path": out_path,
        "exists": exists,
        "has_meta": meta is not None,
        "n_items": len(by_item),
        "n_finished": len(finished),
        "n_incomplete": n_incomplete,
        "n_gen_fail": n_gen_fail,
        "expected_n": expected_n,
        "complete": expected_n is not None and len(finished) == expected_n and n_incomplete == 0,
        "val_rate": n_val / n if n else 0.0,
        "val_ci": wilson_ci(n_val, n),
        "gen_rate": n_gen_ok / n if n else 0.0,
        "gen_ci": wilson_ci(n_gen_ok, n),
        "original_acc": (sum(orig) / len(orig)) if orig else None,
        "accept_rate": n_accept / n if n else 0.0,
        "accept_ci": wilson_ci(n_accept, n),
        "verifier_val_agreement": (n_agree / n_accept) if n_accept else None,
        "mean_iters_to_accept": (sum(a + 1 for a in accept_iters if a is not None) / n_accept)
                                if n_accept else None,
        "loop_depth_hist": depth_hist,
        "val_at_k": val_at_k,
    }


def mcnemar(path_a: str, path_b: str) -> dict:
    """Exact McNemar test on paired ``final_val`` (items shared by both run files)."""
    va = {r.item_id: bool(r.final_val) for r in reconstruct_records(read_records(path_a))}
    vb = {r.item_id: bool(r.final_val) for r in reconstruct_records(read_records(path_b))}
    common = va.keys() & vb.keys()
    b = sum(1 for i in common if va[i] and not vb[i])   # A right, B wrong
    c = sum(1 for i in common if not va[i] and vb[i])   # A wrong, B right
    n = b + c
    if n == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"n_pairs": len(common), "b": b, "c": c, "p_value": p}


def evaluate_config(config_path: str = DEFAULT_CONFIG) -> list[dict]:
    """Evaluate every run declared in a counterfactual_config.json."""
    reports = []
    for cfg in load_configs(config_path):
        report = evaluate_run(default_out_path(cfg), expected_n=cfg.n)
        reports.append({"name": cfg.name or _short(cfg.solver_model), **report})
    return reports


def print_report(reports: list[dict]) -> None:
    header = f"{'run':28} {'fin/N':>9} {'Val (95% CI)':>20} {'Gen':>5} {'Acc':>5} {'agree':>6}"
    print(header)
    print("-" * len(header))
    for r in reports:
        val = f"{r['val_rate']:.3f}[{r['val_ci'][0]:.2f}-{r['val_ci'][1]:.2f}]"
        fin = f"{r['n_finished']}/{r['expected_n']}" if r['expected_n'] else f"{r['n_finished']}/?"
        agree = "-" if r['verifier_val_agreement'] is None else f"{r['verifier_val_agreement']:.2f}"
        flag = "" if r['complete'] else ("  ! incomplete" if r['exists'] else "  x missing")
        print(f"{r['name'][:28]:28} {fin:>9} {val:>20} {r['gen_rate']:>5.2f} "
              f"{r['accept_rate']:>5.2f} {agree:>6}{flag}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    reports = evaluate_config(config_path)
    print_report(reports)
    os.makedirs("results", exist_ok=True)
    with open("results/eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print("\nWrote results/eval_summary.json")
