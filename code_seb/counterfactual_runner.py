"""Step-writing + resumable runner for the counterfactual loop.

Persists one JSONL line per step (via the shared ``JsonlWriter``) so killing the process never
loses a completed step; on restart it groups the existing lines by item and continues each item
from where it stopped. Three models are configured independently (solver / verifier / checker);
identical ids are loaded once and shared (16 GB VRAM budget).
"""
from dataclasses import dataclass, asdict

from tqdm import tqdm

from harness.fixed_seeds import set_all_seeds
from harness.io_jsonl import JsonlWriter, write_meta, git_hash, read_records
from shared_utils.models import load_model_from_str
from shared_utils.dataset_folder import load_dataset_of_string
from code_seb.grading import make_item_id
from code_seb.counterfactual_task import CounterfactualTask
from code_seb.counterfactual_strategy import CounterfactualStrategy


@dataclass
class CFRunConfig:
    """One counterfactual-loop run. Each role's model is set independently."""
    solver_model: str
    verifier_model: str
    checker_model: str
    dataset: str = "gsm8k"
    n: int = 200
    seed: int = 42
    max_loops: int = 3
    temp: float = 0.0
    max_tokens: int = 1200
    verifier_max_tokens: int = 320


def _short(model: str) -> str:
    return model.split("/")[-1]


def default_out_path(config: CFRunConfig) -> str:
    return (f"results/counterfactual_loop_{_short(config.solver_model)}"
            f"_v-{_short(config.verifier_model)}_c-{_short(config.checker_model)}.jsonl")


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

    task = CounterfactualTask(dataset, seed=config.seed)
    strategy = CounterfactualStrategy(
        max_loops=config.max_loops, max_tokens=config.max_tokens,
        verifier_max_tokens=config.verifier_max_tokens, temperature=config.temp,
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
            for step in strategy.run(task, example, models, prior_steps=prior_steps):
                writer.append({"item_id": item_id, **step})

    return out_path


if __name__ == "__main__":
    # Smoke run for the loop: 2 items, smallest model in all three roles (loaded once).
    m = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg = CFRunConfig(solver_model=m, verifier_model=m, checker_model=m, n=2)
    path = run(cfg)
    print(f"Wrote run to {path}")
