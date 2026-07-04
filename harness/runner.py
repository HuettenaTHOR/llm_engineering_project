"""Config-driven generation runner.

Drives a (model, task, strategy) matrix over a seeded dataset subset and appends the full
per-item trace to a JSONL file as each item completes. Resumable: items already present in
the output file are skipped on restart. No metric is computed here -- that is #8's job.
"""
import hashlib
from dataclasses import dataclass, asdict

from tqdm import tqdm

from harness.fixed_seeds import set_all_seeds
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
    temp: float = 0.0
    n: int = 200
    seed: int = 42
    max_tokens: int = 1200
    verifier_max_tokens: int = 320
    quantize: str = None  # e.g. "8bit" to fit a large solver (e.g. Gemma ~16GB) on a 16GB card
    # Independent verifier model for the loop (None -> verify with the solver model). A same-model
    # verifier rubber-stamps the solver, so error-catching needs a distinct (stronger) model here.
    verifier_model: str = None
    verifier_quantize: str = None  # e.g. "8bit" to fit a 7B verifier alongside the solver on 16GB
    verifier_no_thinking: bool = False  # suppress the verifier's <think> block (Qwen3.5) -> concise/fast


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
            verifier_max_tokens=config.verifier_max_tokens, temperature=config.temp
        )
    raise ValueError(f"Unknown strategy '{config.strategy}'")


def make_item_id(example: dict) -> str:
    """Stable id from the question text, so the same item pairs across conditions/seeds."""
    return hashlib.sha1(example["question"].encode("utf-8")).hexdigest()[:12]


def default_out_path(config: RunConfig) -> str:
    model_short = config.model.split("/")[-1]
    # Tag the loop's file with its verifier so an asymmetric run doesn't collide with / look like
    # a same-model run (and stays comparable to the single_shot file for the same solver).
    suffix = ""
    if config.strategy == "verifier_loop" and config.verifier_model:
        suffix = f"_vfy-{config.verifier_model.split('/')[-1]}"
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
        "human_audit": None,
    }


def _free_cuda_cache():
    """Release the CUDA caching allocator's free blocks after each item. Without this the loop's
    growing per-item KV caches fragment the allocator and creep GPU usage up to the limit over a
    handful of items, getting the process OOM-killed mid-run (observed: ~12GB -> 16GB then killed)."""
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
            result = strategy.run(task, example, model)
            writer.append(build_record(item_id, example, task, result))
            _free_cuda_cache()  # keep GPU usage flat across items (else it creeps -> OOM-kill)

    return out_path


if __name__ == "__main__":
    # Smoke run for #7: 5 items, smallest model, single-shot solve.
    cfg = RunConfig(model="Qwen/Qwen2.5-0.5B-Instruct", task="solve", strategy="single_shot", n=5)
    path = run(cfg)
    print(f"Wrote run to {path}")
