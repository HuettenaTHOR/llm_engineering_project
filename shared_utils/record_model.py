"""Typed DTO for the per-item records written to ``results/*.jsonl``.

One ``CounterfactualRecord`` == one JSON line. The record shape mirrors
``harness/runner.py:build_record`` (top level) and
``harness/strategies/verifier_loop.py`` (each ``Iteration``). File I/O is delegated to the
existing ``JsonlWriter`` / ``read_records`` in ``harness/io_jsonl.py`` so persistence stays
in one place.

Standalone helper: nothing in the harness imports this -- consumers (metrics, webui, audits)
can opt in to parse lines into typed objects instead of raw dicts.
"""
from dataclasses import dataclass, field, asdict
from typing import Any

from harness.io_jsonl import JsonlWriter, read_records


@dataclass
class Iteration:
    """One solver (+ optional verifier) round. SingleShot emits exactly one with the
    verifier_* fields left ``None``; the verifier loop emits one per attempt."""
    iteration: int
    candidate: str | None = None
    solver_solve: int | None = None
    verifier_output: str | None = None
    verifier_says: bool | None = None
    verifier_reason: str | None = None
    verdict: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Iteration":
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__})


@dataclass
class CounterfactualRecord:
    """Full per-item trace for one (item x condition) generation."""
    item_id: str
    question: str
    gold: int | None = None
    solver_original_answer: int | None = None
    target_y_ce: int | None = None
    original_correct: bool | None = None
    iterations: list[Iteration] = field(default_factory=list)
    final_correct: bool | None = None
    final_val: bool | None = None
    gen_fail: bool = False
    human_audit: Any = None

    @classmethod
    def from_dict(cls, d: dict) -> "CounterfactualRecord":
        """Parse one JSON record. Missing keys degrade to ``None`` rather than raising, so
        older / partial result files still load."""
        data = {k: d.get(k) for k in cls.__dataclass_fields__}
        data["iterations"] = [Iteration.from_dict(it) for it in (d.get("iterations") or [])]
        return cls(**data)

    def to_dict(self) -> dict:
        """Round-trips exactly to the dict ``build_record`` emits (nested Iterations and key
        order preserved)."""
        return asdict(self)

    @classmethod
    def from_file(cls, path: str) -> list["CounterfactualRecord"]:
        """Load every record from a ``.jsonl`` file (empty list if it does not exist)."""
        return [cls.from_dict(r) for r in read_records(path)]

    @classmethod
    def to_file(cls, path: str, records: list["CounterfactualRecord"]) -> None:
        """Append records to ``path`` via ``JsonlWriter`` (flush + fsync per line). Append
        mode matches the runner's resumable semantics -- write to a fresh path to avoid
        duplicating existing lines."""
        with JsonlWriter(path) as writer:
            for rec in records:
                writer.append(rec.to_dict())


def reconstruct_records(steps: list[dict]) -> list["CounterfactualRecord"]:
    """Collapse the per-step lines written by code_seb's counterfactual loop into the per-item
    ``CounterfactualRecord`` shape (so metrics/webui can consume loop runs unchanged).

    Each item contributes one ``solve`` step (question/gold/target/original answer) and zero or
    more ``cf`` steps (one per loop iteration). The terminating step (``final=True``) carries the
    benchmark ``final_val``. Items are returned in first-seen order."""
    by_item: dict[str, list[dict]] = {}
    order: list[str] = []
    for s in steps:
        iid = s.get("item_id")
        if iid not in by_item:
            by_item[iid] = []
            order.append(iid)
        by_item[iid].append(s)

    records: list[CounterfactualRecord] = []
    for iid in order:
        group = by_item[iid]
        solve = next((s for s in group if s.get("kind") == "solve"), {})
        cfs = sorted((s for s in group if s.get("kind") == "cf"),
                     key=lambda s: s.get("iteration", 0))
        final_step = next((s for s in group if s.get("final")), cfs[-1] if cfs else solve)

        fx, gold = solve.get("solver_original_answer"), solve.get("gold")
        records.append(CounterfactualRecord(
            item_id=iid,
            question=solve.get("question"),
            gold=gold,
            solver_original_answer=fx,
            target_y_ce=solve.get("target_y_ce"),
            original_correct=None if fx is None or gold is None else fx == gold,
            iterations=[Iteration(
                iteration=s.get("iteration"),
                candidate=s.get("candidate"),
                verifier_output=s.get("verifier_output"),
                verifier_says=s.get("verifier_says"),
                verifier_reason=s.get("verifier_reason"),
                verdict="accept" if s.get("accepted") else "reject",
            ) for s in cfs],
            final_correct=final_step.get("final_val"),
            final_val=final_step.get("final_val"),
            gen_fail=any(s.get("gen_fail") for s in group),
        ))
    return records
