import json
import os

BASE = os.path.join(os.path.dirname(__file__), "..", "results")
FILES = {
    "Qwen2.5-7B":   "solve_verifier_loop_Qwen2.5-7B-Instruct_loops5.jsonl",
    "Qwen3.5-4B":   "solve_verifier_loop_Qwen3.5-4B_loops5.jsonl",
    "Llama-3.1-8B": "solve_verifier_loop_Llama-3.1-8B-Instruct_loops5.jsonl",
    "haiku-4.5":    "solve_verifier_loop_claude-haiku-4-5_loops5.jsonl",
}


def load(fn):
    with open(os.path.join(BASE, fn), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def iters(r):
    return r.get("iterations") or []


def main():
    print(f"{'model':13} {'solve_acc(final)':>18} {'orig_acc':>9} {'gen_fail':>8} {'oom':>4} "
          f"{'looped':>7} {'recover':>7} {'regress':>7} {'catch/wrong':>12} {'stamp/wrong':>12}")
    print("-" * 118)
    for name, fn in FILES.items():
        recs = load(fn)
        n = len(recs)
        final_ok = sum(1 for r in recs if r.get("final_correct") is True)
        lo, hi = wilson(final_ok, n)
        gen_fail = sum(1 for r in recs if r.get("gen_fail"))
        oom = sum(1 for r in recs if r.get("oom_skipped"))
        looped = recover = regress = 0
        wrong0 = catch = stamp = orig_ok_ct = 0
        for r in recs:
            its = iters(r)
            if not its:
                continue
            a0, gold = its[0].get("solver_solve"), r.get("gold")
            orig_ok = a0 is not None and gold is not None and a0 == gold
            fin_ok = r.get("final_correct") is True
            orig_ok_ct += orig_ok
            looped += len(its) > 1
            recover += (not orig_ok and fin_ok)
            regress += (orig_ok and not fin_ok)
            if not orig_ok:
                wrong0 += 1
                v0 = its[0].get("verifier_says")
                catch += v0 is False
                stamp += v0 is True
        print(f"{name:13} {final_ok/n:>7.1%} [{lo:.2f}-{hi:.2f}] {orig_ok_ct/n:>9.1%} "
              f"{gen_fail:>8} {oom:>4} {looped:>7} {recover:>7} {regress:>7} "
              f"{f'{catch}/{wrong0}':>12} {f'{stamp}/{wrong0}':>12}")

    print("\nrecover = 1st-attempt wrong -> final right (loop fixed it).")
    print("regress = 1st-attempt right -> final wrong (loop broke it).")
    print("catch   = of items whose 1st answer was WRONG, how many the self-verifier REJECTED.")
    print("stamp   = ...how many it ACCEPTED anyway (rubber-stamp).")

    # Illustrative trace 1: Llama looped (verifier rejected) yet stayed wrong.
    print("\n=== TRACE A: Llama-3.1-8B looped, verifier rejected a CORRECT answer, ended wrong ===")
    for r in load(FILES["Llama-3.1-8B"]):
        its = iters(r)
        if len(its) >= 3 and r.get("final_correct") is not True and its[0].get("solver_solve") != r.get("gold"):
            print(f"item {r['item_id']}  gold={r['gold']}  final_correct={r.get('final_correct')}")
            print("Q:", " ".join(r["question"].split())[:200])
            for it in its:
                print(f"  iter{it['iteration']}: ans={it.get('solver_solve')} verdict={it.get('verdict')} "
                      f"reason={' '.join((it.get('verifier_reason') or '').split())[:120]}")
            break

    # Illustrative trace 2: Qwen3.5-4B rubber-stamp of a wrong answer.
    print("\n=== TRACE B: Qwen3.5-4B wrong 1st answer, self-verifier ACCEPTED it ===")
    for r in load(FILES["Qwen3.5-4B"]):
        its = iters(r)
        if its and its[0].get("solver_solve") != r.get("gold") and its[0].get("verifier_says") is True:
            it = its[0]
            print(f"item {r['item_id']}  gold={r['gold']}  solver_ans={it.get('solver_solve')}  "
                  f"final_correct={r.get('final_correct')}")
            print("Q:", " ".join(r["question"].split())[:200])
            print("verifier_output(end):", " ".join((it.get('verifier_output') or '').split())[-240:])
            break


if __name__ == "__main__":
    main()
