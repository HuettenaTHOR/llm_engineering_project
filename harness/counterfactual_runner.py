"""Step-writing + resumable runner for the counterfactual loop (integrated from code_seb).

Persists one JSONL line per step (via the shared ``JsonlWriter``) so killing the process never
loses a completed step; on restart it groups the existing lines by item and continues each item
from where it stopped. Three models are configured independently (solver / verifier / checker);
identical ids are loaded once and shared (16 GB VRAM budget).

The solve task's per-item runner lives in harness/runner.py; the counterfactual loop needs a
distinct step-streaming, three-role runner, so it gets its own entry point here (matching the
generation/analysis split in DESIGN 8).
"""
import sys
from dataclasses import asdict

from tqdm import tqdm

from harness.fixed_seeds import set_all_seeds, set_item_seed
from harness.io_jsonl import JsonlWriter, write_meta, git_hash, read_records
from harness.runner import make_item_id, _is_oom_error, _free_cuda_cache
from harness.cf_config import CFRunConfig, default_out_path, load_configs
from harness.tasks import CounterfactualTask
from harness.strategies import CounterfactualStrategy
from harness.strategies.counterfactual_loop import _record
from shared_utils.models import load_model_from_str
from shared_utils.dataset_folder import load_dataset_of_string

DEFAULT_CONFIG = "counterfactual_config.json"


def _steps_by_item(path: str) -> dict:
    """Group already-written step lines by item_id (for resume)."""
    grouped: dict = {}
    for record in read_records(path):
        grouped.setdefault(record.get("item_id"), []).append(record)
    return grouped


def run(config: CFRunConfig, out_path: str = None) -> str:
    set_all_seeds(config.seed)
    out_path = out_path or default_out_path(config)
    dataset = load_dataset_of_string(config.dataset)
    print(f"Loaded {dataset.get_dataset_size()} items; using {config.n} with seed {config.seed}")

    task = CounterfactualTask(
        dataset, seed=config.seed,
        verifier_sees_solver_output=config.verifier_sees_solver_output,
    )
    strategy = CounterfactualStrategy(
        max_loops=config.max_loops, max_tokens=config.max_tokens,
        verifier_max_tokens=config.verifier_max_tokens, temperature=config.temp,
        verifier_accept_on_unparsed=config.verifier_accept_on_unparsed,
    )
    subset = dataset.get_random_subset(size=config.n, seed=config.seed)
    examples = [subset[i] for i in range(len(subset))]

    write_meta(out_path, {
        "config": asdict(config), "git_hash": git_hash(),
        "seed": config.seed, "out_path": out_path,
    })

    prior = _steps_by_item(out_path)  # resume: items with a final step are skipped entirely

    cache: dict = {}  # model id -> loaded model (lazy, shared across roles)
    def get_model(name: str):
        if name not in cache:
            cache[name] = load_model_from_str(name)
        return cache[name]

    models = None  # lazy: don't load anything if every item is already finished
    desc = f"cf_loop | s={config.solver_model} v={config.verifier_model} c={config.checker_model}"
    with JsonlWriter(out_path) as writer:
        for example in tqdm(examples, desc=desc):
            item_id = make_item_id(example)
            prior_steps = prior.get(item_id, [])
            if any(s.get("final") for s in prior_steps):
                continue
            if models is None:
                models = {
                    "solver": get_model(config.solver_model),
                    "verifier": get_model(config.verifier_model),
                    "checker": get_model(config.checker_model),
                }
            try:
                # set_item_seed is INSIDE the guard on purpose: a CUDA OOM from the previous item's
                # generation is reported asynchronously and often surfaces at the next CUDA call --
                # here torch.manual_seed -- so it must be caught to skip rather than abort the run.
                set_item_seed(config.seed, item_id)  # per-item RNG: item i starts identically everywhere
                for step in strategy.run(task, example, models, prior_steps=prior_steps):
                    writer.append({"item_id": item_id, **step})
            except Exception as exc:  # noqa: BLE001 -- re-raised below unless it's an OOM
                if not _is_oom_error(exc):
                    raise
                # VRAM overload: write a final gen-fail marker so the item is skipped (and stays
                # skipped on resume) instead of aborting the whole run.
                print(f"\n[OOM] skipping item {item_id} (VRAM overload): {exc}", file=sys.stderr)
                writer.append({"item_id": item_id, **_record(
                    "solve", -1, question=example["question"], gold=task.gold(example),
                    gen_fail=True, final=True, oom_skipped=True)})
            finally:
                # Free the CUDA allocator's blocks after every item -- the CF loop's growing per-item
                # KV caches fragment the allocator otherwise (the solve runner does the same). Also
                # resets enough state to recover from a caught OOM before the next item.
                _free_cuda_cache()

    return out_path


def run_from_config(config_path: str = DEFAULT_CONFIG) -> list[str]:
    """Run every model combination declared in a counterfactual_config.json. Each run is
    independently resumable, so re-invoking after an abort continues where it stopped."""
    configs = load_configs(config_path)
    print(f"Running {len(configs)} combination(s) from {config_path}")
    out_paths = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n=== [{i}/{len(configs)}] {cfg.name or cfg.solver_model} ===")
        out_paths.append(run(cfg))
    return out_paths


if __name__ == "__main__":
    # `python -m harness.counterfactual_runner [config.json]` runs the whole matrix.
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    for path in run_from_config(config_path):
        print(f"Wrote run to {path}")
