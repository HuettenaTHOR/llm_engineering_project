"""Smoke driver / CLI for the harness.

Runs the SolveTask (#5) through one or both strategies (#6) via the config-driven runner (#7),
which persists the complete per-item trace to results/*.jsonl. Accuracy is then reported by
reading those JSONL files back -- analysis never re-touches a model (DESIGN 8).

Examples:
    python -m harness.run_base_test                          # default model, both strategies, n=5
    python -m harness.run_base_test --model Qwen/Qwen3.5-8B --n 200
    python -m harness.run_base_test --strategy verifier_loop --max-loops 3
"""
import argparse
import os

from harness.runner import RunConfig, run
from harness.io_jsonl import read_records
from shared_utils.models import QWEN35_LADDER

# Not pinned: override with --model, or set BENCH_MODEL to test newer models without flags.
DEFAULT_MODEL = os.environ.get("BENCH_MODEL", "Qwen/Qwen3.5-0.8B")
STRATEGIES = ("single_shot", "verifier_loop")


def _report(label: str, out_path: str):
    records = read_records(out_path)
    total = len(records)
    correct = sum(1 for r in records if r["final_correct"] is True)
    gen_fail = sum(1 for r in records if r["gen_fail"])
    accuracy = correct / total if total else 0
    print(
        f"[{label}] {out_path}\n"
        f"    Accuracy: {accuracy:.2%} ({correct}/{total}), Gen-fail: {gen_fail}/{total}"
    )


def run_test_reasoning_baseline(model_name, dataset_name, n, max_loops, seed, temp, strategies,
                                verifier_model=None, quantize=None, verifier_quantize=None,
                                verifier_no_thinking=False, max_tokens=2560, verifier_max_tokens=4096):
    """Drive the chosen strategies over a subset, writing JSONL and reporting accuracy.

    ``verifier_model`` (if set) makes the loop verify with a distinct, stronger model -- a
    same-model verifier rubber-stamps the solver. Budgets default high: the solver/verifier are
    'thinking' models that truncate (and a truncated verdict defaults to accept) at small caps."""
    for strategy in strategies:
        cfg = RunConfig(
            model=model_name, task="solve", strategy=strategy,
            dataset=dataset_name, n=n, max_loops=max_loops, seed=seed, temp=temp,
            max_tokens=max_tokens, verifier_max_tokens=verifier_max_tokens,
            verifier_model=verifier_model, quantize=quantize, verifier_quantize=verifier_quantize,
            verifier_no_thinking=verifier_no_thinking,
        )
        out_path = run(cfg)
        _report(strategy, out_path)


class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show defaults *and* keep the epilog's line breaks."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=_Formatter,
        epilog="Suggested Qwen3.5 variants:\n  " + "\n  ".join(QWEN35_LADDER)
        + "\n\nThe model is not pinned: pass any HuggingFace repo id as --model, or set the\n"
        + "BENCH_MODEL env var to change the default (e.g. a newer model).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="solver: HF repo id or registered alias")
    parser.add_argument("--verifier-model", default=None,
                        help="distinct verifier model for the loop (default: same as --model)")
    parser.add_argument("--quantize", default=None, choices=(None, "8bit"),
                        help="quantize the solver (e.g. 8bit) to fit a large model (e.g. Gemma ~16GB) on 16GB")
    parser.add_argument("--verifier-quantize", default=None, choices=(None, "8bit"),
                        help="quantize the verifier (e.g. 8bit) to fit a 7B alongside the solver on 16GB")
    parser.add_argument("--verifier-no-thinking", action="store_true",
                        help="suppress the verifier's <think> block (Qwen3.5): concise, fast, no truncation")
    parser.add_argument("--max-tokens", type=int, default=4096, help="solver generation cap")
    parser.add_argument("--verifier-max-tokens", type=int, default=4096,
                        help="verifier generation cap (must be large enough to reach its verdict)")
    parser.add_argument("--dataset", default="gsm8k", help="dataset name")
    parser.add_argument("--n", type=int, default=5, help="number of items")
    parser.add_argument("--max-loops", type=int, default=5,
                        help="max solver-verifier loops (verifier_loop only)")
    parser.add_argument("--seed", type=int, default=42, help="RNG + subset seed")
    parser.add_argument("--temp", type=float, default=0.0, help="sampling temperature")
    parser.add_argument("--strategy", choices=(*STRATEGIES, "both"), default="both",
                        help="which control flow to run")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    strategies = STRATEGIES if args.strategy == "both" else (args.strategy,)
    print(f"Running solver={args.model} verifier={args.verifier_model or args.model} on "
          f"{args.dataset} with strategies {strategies}, n={args.n}, max_loops={args.max_loops}, "
          f"seed={args.seed}, temp={args.temp}, max_tokens={args.max_tokens}, "
          f"verifier_max_tokens={args.verifier_max_tokens}")
    run_test_reasoning_baseline(
        args.model, args.dataset, args.n, args.max_loops, args.seed, args.temp, strategies,
        verifier_model=args.verifier_model, quantize=args.quantize,
        verifier_quantize=args.verifier_quantize,
        verifier_no_thinking=args.verifier_no_thinking,
        max_tokens=args.max_tokens, verifier_max_tokens=args.verifier_max_tokens,
    )
