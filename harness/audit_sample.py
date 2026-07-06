"""Manual-audit sampler for the Haiku grades (RQ plan D1).

``export`` pulls a lightly model-stratified ~30-item sample (question / final candidate / target
from the reconstructed records, Haiku's verdict + reason from the grades files) into
``results/audit_sample.csv`` with a blank ``human_valid`` column for you to fill. ``report`` re-reads
the filled CSV and prints Haiku<->human agreement over the rows you judged.

CLI:
    python -m harness.audit_sample export [config.json]
    python -m harness.audit_sample report [audit_sample.csv]
"""
import csv
import random
import sys

from harness.io_jsonl import read_records
from harness.cf_config import default_out_path, load_configs
from harness.haiku_grader import grades_path
from shared_utils.record_model import reconstruct_records

DEFAULT_CONFIG = "counterfactual_config.json"
CSV_PATH = "results/audit_sample.csv"
COLUMNS = ["item_id", "model", "question", "candidate", "target",
           "haiku_valid", "haiku_reason", "human_valid"]
SAMPLE_N = 30


def _truthy(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "t", "valid")


def export(config_path: str = DEFAULT_CONFIG, out_csv: str = CSV_PATH) -> None:
    configs = load_configs(config_path)
    per_run = max(1, round(SAMPLE_N / len(configs))) if configs else SAMPLE_N
    rng = random.Random(42)
    rows = []
    for cfg in configs:
        out_path = default_out_path(cfg)
        grades = {g["item_id"]: g for g in read_records(grades_path(out_path))}
        pool = []
        for r in reconstruct_records(read_records(out_path)):
            g = grades.get(r.item_id)
            if g is None or not r.iterations:
                continue
            pool.append({
                "item_id": r.item_id,
                "model": cfg.name,
                "question": r.question,
                "candidate": r.iterations[-1].candidate,
                "target": r.target_y_ce,
                "haiku_valid": g.get("valid_obj"),
                "haiku_reason": g.get("reason"),
                "human_valid": "",
            })
        rng.shuffle(pool)
        rows.extend(pool[:per_run])
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} sampled items to {out_csv} (fill the human_valid column, then: report)")


def report(csv_path: str = CSV_PATH) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    filled = [r for r in rows if r.get("human_valid", "").strip() != ""]
    if not filled:
        print(f"No human_valid filled in {csv_path} yet.")
        return
    agree = sum(1 for r in filled if _truthy(r["haiku_valid"]) == _truthy(r["human_valid"]))
    print(f"Haiku<->human agreement: {agree}/{len(filled)} = {agree / len(filled):.1%}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "export"
    arg = sys.argv[2] if len(sys.argv) > 2 else None
    if cmd == "export":
        export(arg or DEFAULT_CONFIG)
    elif cmd == "report":
        report(arg or CSV_PATH)
    else:
        print(__doc__)
