import hashlib
import json
import sys
from dataclasses import dataclass, asdict, fields

from tqdm import tqdm

from harness.fixed_seeds import set_all_seeds, set_item_seed
from shared_utils.dataset_folder import load_dataset_of_string
from shared_utils.models import load_model_from_str
from harness.tasks import SolveTask
from harness.strategies import SingleShot, SolverVerifierLoop
from harness.io_jsonl import JsonlWriter, write_meta, git_hash, seen_item_ids


@dataclass
class RunConfig:
    """One row of the experiment matrix. Everything needed to reproduce a generation run."""
    model: str
    task: str = "solve"            # -> build_task
    strategy: str = "single_shot"  # -> build_strategy
    dataset: str = "gsm8k"
    max_loops: int = 5
    temp: float | None = None  # None -> use each model's shipped generation_config defaults
    n: int = 200
    seed: int = 42
    max_tokens: int = 10000
    verifier_max_tokens: int = 10000
    verifier_accept_on_unparsed: bool = True
    quantize: str = None 
    verifier_model: str = None
    verifier_quantize: str = None  
    verifier_no_thinking: bool = False


def build_task(config: RunConfig, dataset):
    if config.task == "solve":
        return SolveTask(dataset)
    if config.task == "counterfactual":
        raise ValueError(
            "The counterfactual task uses a dedicated step-streaming, three-role runner; run it "
            "via harness.counterfactual_runner (CFRunConfig / counterfactual_config.json), "
            "not this per-item solve runner."
        )
    raise ValueError(f"Unknown task '{config.task}'")


def build_strategy(config: RunConfig):
    if config.strategy == "single_shot":
        return SingleShot(max_tokens=config.max_tokens, temperature=config.temp)
    if config.strategy == "verifier_loop":
        return SolverVerifierLoop(
            max_loops=config.max_loops, max_tokens=config.max_tokens,
            verifier_max_tokens=config.verifier_max_tokens, temperature=config.temp,
            verifier_accept_on_unparsed=config.verifier_accept_on_unparsed,
        )
    raise ValueError(f"Unknown strategy '{config.strategy}'")


def make_item_id(example: dict) -> str:
    """Stable id from the question text, so the same item pairs across conditions/seeds."""
    return hashlib.sha1(example["question"].encode("utf-8")).hexdigest()[:12]


def default_out_path(config: RunConfig) -> str:
    model_short = config.model.split("/")[-1]
    suffix = ""
    if config.strategy == "verifier_loop":
        if config.verifier_model:
            suffix += f"_vfy-{config.verifier_model.split('/')[-1]}"
        suffix += f"_loops{config.max_loops}"
    return f"results/{config.task}_{config.strategy}_{model_short}{suffix}.jsonl"


def build_record(item_id: str, example: dict, task, result: dict) -> dict:
    """Assemble the full DESIGN-8 record from the strategy result + task gold.

    The strategy result already carries the complete per-iteration model trace (every raw
    solver + verifier generation). target_y_ce / final_val are null for the solve task and
    populated by CounterfactualTask."""
    iterations = result["iterations"]
    original_answer = iterations[0]["solver_solve"] if iterations else None
    return {
        "item_id": item_id,
        "question": example["question"],
        "gold": task.gold(example),
        "solver_original_answer": original_answer,
        "target_y_ce": result.get("target_y_ce"),
        "original_correct": result.get("original_correct"),
        "iterations": iterations,
        "final_correct": result.get("final_correct"),
        "final_val": result.get("final_val"),
        "gen_fail": result["gen_fail"],
        "oom_skipped": result.get("oom_skipped", False),
    }


def _is_oom_error(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory (VRAM overload)"""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _free_cuda_cache():
    """Release the CUDA caching allocator's free blocks after each item."""
    try:
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run(config: RunConfig, out_path: str = None) -> str:
    """Execute a generation run; returns the output JSONL path."""
    set_all_seeds(config.seed)
    out_path = out_path or default_out_path(config)
    dataset = load_dataset_of_string(config.dataset)
    print(f"Loaded {dataset.get_dataset_size()} items from {config.dataset}; using {config.n} items with seed {config.seed}")
    task = build_task(config, dataset)

    strategy = build_strategy(config)
    subset = dataset.get_random_subset(size=config.n, seed=config.seed)
    examples = [subset[i] for i in range(len(subset))]

    write_meta(out_path, {
        "config": asdict(config),
        "git_hash": git_hash(),
        "seed": config.seed,
        "out_path": out_path,
    })

    done = seen_item_ids(out_path)  # resume: anything already written is skipped
    model = None  # lazy: don't load the (heavy) model if there's nothing left to do
    print("Resuming run; skipping %d items already in %s" % (len(done), out_path))
    desc = f"{config.strategy} | {config.task} | {config.model}"
    with JsonlWriter(out_path) as writer:
        for example in tqdm(examples, desc=desc):
            item_id = make_item_id(example)
            if item_id in done:
                continue
            set_item_seed(config.seed, item_id)  # per-item RNG: item i starts identically everywhere
            if model is None:
                model = load_model_from_str(config.model, quantize=config.quantize)
                # Asymmetric verifier: load the (distinct) verifier model once and hand it to the
                # loop. Reuse the solver object if the verifier id is the same model.
                if config.verifier_model and isinstance(strategy, SolverVerifierLoop):
                    strategy.verifier_model = (
                        model if config.verifier_model == config.model
                        else load_model_from_str(config.verifier_model, quantize=config.verifier_quantize)
                    )
                    if config.verifier_no_thinking:
                        strategy.verifier_model.enable_thinking = False
            try:
                result = strategy.run(task, example, model)
            except Exception as exc:  # noqa: BLE001 -- re-raised below unless it's an OOM
                if not _is_oom_error(exc):
                    raise
                # VRAM overload: free the cache, record the item as an OOM skip, keep going.
                _free_cuda_cache()
                print(f"\n[OOM] skipping item {item_id} (VRAM overload): {exc}", file=sys.stderr)
                result = {"iterations": [], "gen_fail": True, "oom_skipped": True}
            writer.append(build_record(item_id, example, task, result))
            _free_cuda_cache()  # keep GPU usage flat across items (else it creeps -> OOM-kill)

    return out_path


DEFAULT_CONFIG = "benchmark_config.json"


def load_configs(path: str) -> list[RunConfig]:
    """Parse a benchmark_config.json into one RunConfig per ``runs[]`` entry. Top-level keys are
    shared defaults; each run entry overrides them. Unknown keys are ignored."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    defaults = {k: v for k, v in spec.items() if k != "runs"}
    allowed = {f.name for f in fields(RunConfig)}
    configs = []
    for entry in spec.get("runs", []):
        merged = {**defaults, **entry}
        configs.append(RunConfig(**{k: v for k, v in merged.items() if k in allowed}))
    return configs


def run_from_config(config_path: str = DEFAULT_CONFIG) -> list[str]:
    """Run every model/strategy combination declared in a benchmark_config.json. Each run is
    independently resumable, so re-invoking after an abort continues where it stopped."""
    configs = load_configs(config_path)
    print(f"Running {len(configs)} combination(s) from {config_path}")
    out_paths = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n=== [{i}/{len(configs)}] {cfg.model} | {cfg.strategy} (max_loops={cfg.max_loops}) ===")
        out_paths.append(run(cfg))
    return out_paths


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # `python -m harness.runner benchmark_config.json` runs the whole matrix.
        for path in run_from_config(sys.argv[1]):
            print(f"Wrote run to {path}")
    else:
        # Smoke run for #7: 5 items, smallest model, single-shot solve.
        cfg = RunConfig(model="Qwen/Qwen2.5-0.5B-Instruct", task="solve", strategy="single_shot", n=5)
        path = run(cfg)
        print(f"Wrote run to {path}")
