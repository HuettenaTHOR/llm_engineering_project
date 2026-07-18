import json
import math
import os
import sys

from harness.io_jsonl import read_records, read_meta
from harness.cf_config import default_out_path, load_configs, _short
from harness.opus_grader import grades_path
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


def _median(xs: list) -> float:
    s = sorted(xs)
    m = len(s)
    return (s[m // 2] if m % 2 else (s[m // 2 - 1] + s[m // 2]) / 2)


def grade_metrics(out_path: str) -> dict:
    """Objective-validity metrics from a run's ``.grades.jsonl`` file (empty if none)."""
    grades = read_records(grades_path(out_path))
    n = len(grades)
    if n == 0:
        return {"n_graded": 0, "val_obj_rate": None, "val_obj_ci": None,
                "first_val_obj_rate": None, "first_val_obj_ci": None,
                "solve_to_target_rate": None, "minimal_rate": None,
                "edit_dist_mean": None, "edit_dist_median": None}
    n_valid = sum(1 for g in grades if g.get("valid_obj"))
    n_first_valid = sum(1 for g in grades if g.get("first_valid_obj"))
    n_solve = sum(1 for g in grades if g.get("solves_to_target"))
    n_min = sum(1 for g in grades if g.get("minimal_edit") is True)
    dists = [g["edit_distance"] for g in grades if g.get("edit_distance") is not None]
    return {
        "n_graded": n,
        "val_obj_rate": n_valid / n,
        "val_obj_ci": wilson_ci(n_valid, n),
        "first_val_obj_rate": n_first_valid / n,
        "first_val_obj_ci": wilson_ci(n_first_valid, n),
        "solve_to_target_rate": n_solve / n,
        "minimal_rate": n_min / n,
        "edit_dist_mean": (sum(dists) / len(dists)) if dists else None,
        "edit_dist_median": _median(dists) if dists else None,
    }


def _valid_obj_map(out_path: str) -> dict:
    """item_id -> bool valid_obj from a run's grades file (empty if none)."""
    return {g["item_id"]: bool(g.get("valid_obj")) for g in read_records(grades_path(out_path))}


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
        **grade_metrics(out_path),
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


def _exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def mcnemar_obj(path_a: str, path_b: str) -> dict:
    """Exact McNemar test on paired ``valid_obj`` from the two runs' grades files (RQ plan C3)."""
    va, vb = _valid_obj_map(path_a), _valid_obj_map(path_b)
    common = va.keys() & vb.keys()
    b = sum(1 for i in common if va[i] and not vb[i])   # A valid, B not
    c = sum(1 for i in common if not va[i] and vb[i])   # A not, B valid
    return {"n_pairs": len(common), "b": b, "c": c, "p_value": _exact_mcnemar_p(b, c)}


def _paired_diff_ci(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wald CI for the paired difference in proportions (c - b)/n (discordant-pair variance)."""
    if n == 0:
        return (0.0, 0.0)
    d = (c - b) / n
    half = z * math.sqrt(max(b + c - (c - b) ** 2 / n, 0.0)) / n
    return (max(-1.0, d - half), min(1.0, d + half))


def delta_val_obj(single_path: str, loop_path: str) -> dict:
    """ΔVal = Val_obj(loop) - Val_obj(single) over items graded in BOTH runs, with a CI and the
    exact McNemar p (RQ plan C3)."""
    mc = mcnemar_obj(single_path, loop_path)
    n = mc["n_pairs"]
    va, vb = _valid_obj_map(single_path), _valid_obj_map(loop_path)
    common = va.keys() & vb.keys()
    return {
        "n_pairs": n,
        "val_obj_single": (sum(va[i] for i in common) / n) if n else None,
        "val_obj_loop": (sum(vb[i] for i in common) / n) if n else None,
        "delta_val": ((mc["c"] - mc["b"]) / n) if n else None,
        "delta_ci": _paired_diff_ci(mc["b"], mc["c"], n),
        "p_value": mc["p_value"],
        "b": mc["b"], "c": mc["c"],
    }


def delta_val_obj_within(out_path: str) -> dict:
    """ΔVal = Val_obj(final CF, after the loop) - Val_obj(first CF, single-shot-equivalent),
    paired within ONE run's grades file (RQ plan T)."""
    grades = read_records(grades_path(out_path))
    n = len(grades)
    b = sum(1 for g in grades if g.get("first_valid_obj") and not g.get("valid_obj"))  # 1st right, loop wrong
    c = sum(1 for g in grades if not g.get("first_valid_obj") and g.get("valid_obj"))  # 1st wrong, loop right
    return {
        "n_pairs": n,
        "val_obj_single": (sum(1 for g in grades if g.get("first_valid_obj")) / n) if n else None,
        "val_obj_loop": (sum(1 for g in grades if g.get("valid_obj")) / n) if n else None,
        "delta_val": ((c - b) / n) if n else None,
        "delta_ci": _paired_diff_ci(b, c, n),
        "p_value": _exact_mcnemar_p(b, c),
        "b": b, "c": c,
    }


def solve_accuracy(*names: str):
    """P1 solve accuracy from a solve-baseline file ``results/solve_single_shot_<name>.jsonl`` if
    present (first name that resolves to a non-empty file), else None"""
    for name in names:
        recs = read_records(f"results/solve_single_shot_{name}.jsonl")
        if recs:
            return sum(1 for r in recs if r.get("final_correct") is True) / len(recs)
    return None


def build_model_summary(reports: list[dict], config_path: str = DEFAULT_CONFIG) -> list[dict]:
    """Per-model P1->P2->T join: solve-acc, single(=1st CF)/loop(=final CF)
    Val_obj [CI], ΔVal [McNemar p], solve/minimal/edit-dist components, verifier accept-rate &
    agreement."""
    cfgs = {c.name: c for c in load_configs(config_path)}
    by_name = {r["name"]: r for r in reports}
    rows = []
    for name in sorted(cfgs):
        cfg = cfgs[name]
        r = by_name.get(name)
        if r is None:
            continue
        row = {
            "model": name,
            "solve_acc": solve_accuracy(name, _short(cfg.solver_model)),
            "val_obj_single": r.get("first_val_obj_rate"),
            "val_obj_single_ci": r.get("first_val_obj_ci"),
            "val_obj_loop": r.get("val_obj_rate"),
            "val_obj_loop_ci": r.get("val_obj_ci"),
            "solve_to_target_rate": r.get("solve_to_target_rate"),
            "minimal_rate": r.get("minimal_rate"),
            "edit_dist_mean": r.get("edit_dist_mean"),
            "accept_rate": r.get("accept_rate"),
            "verifier_val_agreement": r.get("verifier_val_agreement"),
        }
        d = delta_val_obj_within(r["out_path"])
        row.update({"delta_val": d["delta_val"], "delta_ci": d["delta_ci"],
                    "mcnemar_p": d["p_value"], "n_pairs": d["n_pairs"]})
        rows.append(row)
    return rows


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


def _fmt(x, prec=3):
    return "n/a" if x is None else f"{x:.{prec}f}"


def print_model_summary(rows: list[dict]) -> None:
    """The RQ answer table: P1 (solve) -> P2 (single Val_obj) -> T (ΔVal) per model."""
    header = (f"{'model':16} {'solve':>6} {'Val_obj single':>16} {'Val_obj loop':>16} "
              f"{'dVal':>7} {'McN p':>7} {'solve%/min%/edit':>18} {'acc/agree':>11}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        vs = "n/a" if r["val_obj_single"] is None else \
            f"{r['val_obj_single']:.3f}[{r['val_obj_single_ci'][0]:.2f}-{r['val_obj_single_ci'][1]:.2f}]"
        vl = "n/a" if r["val_obj_loop"] is None else \
            f"{r['val_obj_loop']:.3f}[{r['val_obj_loop_ci'][0]:.2f}-{r['val_obj_loop_ci'][1]:.2f}]"
        comp = f"{_fmt(r['solve_to_target_rate'],2)}/{_fmt(r['minimal_rate'],2)}/{_fmt(r['edit_dist_mean'],2)}"
        acc = f"{_fmt(r['accept_rate'],2)}/{_fmt(r['verifier_val_agreement'],2)}"
        print(f"{r['model'][:16]:16} {_fmt(r['solve_acc'],2):>6} {vs:>16} {vl:>16} "
              f"{_fmt(r['delta_val']):>7} {_fmt(r['mcnemar_p']):>7} {comp:>18} {acc:>11}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    reports = evaluate_config(config_path)
    print_report(reports)
    model_summary = build_model_summary(reports, config_path)
    print_model_summary(model_summary)
    os.makedirs("results", exist_ok=True)
    with open("results/eval_summary.json", "w", encoding="utf-8") as f:
        json.dump({"runs": reports, "model_summary": model_summary}, f, indent=2)
    print("\nWrote results/eval_summary.json")
