"""Counterfactual run configuration (lightweight: no torch / tqdm).

Kept separate from counterfactual_runner.py so the analysis side (counterfactual_evaluator.py)
can load configs and resolve artifact paths without importing the generation stack (model
loading, seeds). Integrated from code_seb/cf_config.py.
"""
import json
from dataclasses import dataclass, fields


@dataclass
class CFRunConfig:
    """One counterfactual-loop run. Each role's model is set independently."""
    solver_model: str
    verifier_model: str
    checker_model: str
    name: str | None = None
    dataset: str = "gsm8k"
    n: int = 200
    seed: int = 42
    max_loops: int = 3
    temp: float | None = None  # None -> use each model's shipped generation_config defaults
    max_tokens: int = 10000 # high ceiling to avoid truncation; the verifier's verdict is parsed from the output, not the end of it
    verifier_max_tokens: int = 10000
    # When the verifier's verdict can't be parsed (rambled / truncated), accept it (early-stop)
    verifier_accept_on_unparsed: bool = True
    # False (default) -> blind judge: it sees only the target + candidate revised problem and
    # re-solves independently. True -> trace-aware judge: it also sees the solver's original solve.
    verifier_sees_solver_output: bool = False


def _short(model: str) -> str:
    return model.split("/")[-1]


def default_out_path(config: CFRunConfig) -> str:
    """Artifact path. Named runs get a stable filename; otherwise derive one from the models so
    distinct role combinations never collide."""
    if config.name:
        return f"results/counterfactual_loop_{config.name}.jsonl"
    return (f"results/counterfactual_loop_{_short(config.solver_model)}"
            f"_v-{_short(config.verifier_model)}_c-{_short(config.checker_model)}.jsonl")


def load_configs(path: str) -> list[CFRunConfig]:
    """Parse a counterfactual_config.json into one CFRunConfig per ``runs[]`` entry. Top-level
    keys are shared defaults; each run entry overrides them. Unknown keys are ignored."""
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    defaults = {k: v for k, v in spec.items() if k != "runs"}
    allowed = {f.name for f in fields(CFRunConfig)}
    configs = []
    for run in spec.get("runs", []):
        merged = {**defaults, **run}
        configs.append(CFRunConfig(**{k: v for k, v in merged.items() if k in allowed}))
    return configs
