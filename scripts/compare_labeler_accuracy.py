"""
compare_labeler_accuracy.py
---------------------------
Compare labeler accuracy results across two checkpoints:
  - SFT baseline (qwen3-4b-retry_only-r8-seed42)
  - RLVR Step Credit (step_credit-r8-seed42)

Prints a side-by-side table of solution quality and labeler metrics,
confirming whether step-collapse is caused by RL training or present in SFT.

Usage:
    python scripts/compare_labeler_accuracy.py
"""

import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUNS_DIR  = _REPO_ROOT / "results" / "runs"

CHECKPOINTS = [
    ("SFT  (Retry Only — pre-RL)", "qwen3-4b-retry_only-r8-seed42_labeler_accuracy.jsonl"),
    ("RLVR (Step Credit)",         "step_credit-r8-seed42_labeler_accuracy.jsonl"),
]


def analyse(path: Path) -> dict:
    if not path.exists():
        return None
    data = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    n = len(data)
    if n == 0:
        return None

    avg_steps   = sum(d.get("n_steps", 0) for d in data) / n
    single_step = sum(1 for d in data if d.get("n_steps", 0) <= 1) / n

    oracle_found = [d for d in data if d.get("oracle_idx") is not None]
    coverage     = len(oracle_found) / n

    if oracle_found:
        exact = sum(1 for d in oracle_found if d.get("exact_match")) / len(oracle_found)
        w1    = sum(1 for d in oracle_found
                    if d.get("abs_error") is not None and d["abs_error"] <= 1) / len(oracle_found)
        errors = [d["abs_error"] for d in oracle_found if d.get("abs_error") is not None]
        mean_err = sum(errors) / len(errors) if errors else None
        early = sum(1 for d in oracle_found
                    if d.get("labeler_idx") is not None
                    and d.get("oracle_idx") is not None
                    and d["labeler_idx"] < d["oracle_idx"]) / len(oracle_found)
        late  = sum(1 for d in oracle_found
                    if d.get("labeler_idx") is not None
                    and d.get("oracle_idx") is not None
                    and d["labeler_idx"] > d["oracle_idx"]) / len(oracle_found)
    else:
        exact = w1 = mean_err = early = late = None

    return dict(
        n=n,
        avg_steps=avg_steps,
        single_step_pct=single_step,
        oracle_coverage=coverage,
        exact_match=exact,
        within_1=w1,
        mean_abs_error=mean_err,
        labeler_early=early,
        labeler_late=late,
    )


def fmt(val, pct=True, decimals=1):
    if val is None:
        return "n/a  "
    if pct:
        return f"{val * 100:.{decimals}f}%"
    return f"{val:.{decimals}f}"


def main():
    results = []
    for label, fname in CHECKPOINTS:
        path = _RUNS_DIR / fname
        stats = analyse(path)
        results.append((label, stats))

    # ── Header ────────────────────────────────────────────────────────────────
    col_w = 32
    print()
    print("=" * 80)
    print("  LABELER ACCURACY — SFT vs RLVR CONTROL COMPARISON")
    print("=" * 80)

    labels = [r[0] for r in results]
    header = f"  {'Metric':<30}" + "".join(f"  {l:<{col_w}}" for l in labels)
    print(header)
    print("  " + "-" * (30 + (col_w + 2) * len(results)))

    def row(name, key, pct=True, decimals=1, invert=False):
        vals = []
        for _, stats in results:
            if stats is None:
                vals.append("missing")
            else:
                v = stats.get(key)
                vals.append(fmt(v, pct=pct, decimals=decimals))
        print(f"  {name:<30}" + "".join(f"  {v:<{col_w}}" for v in vals))

    row("Problems evaluated",      "n",               pct=False, decimals=0)
    row("Avg solution steps",      "avg_steps",        pct=False, decimals=1)
    row("Single-step outputs",     "single_step_pct",  pct=True)
    print()
    row("Oracle coverage",         "oracle_coverage",  pct=True)
    row("Exact match acc",         "exact_match",      pct=True)
    row("Within-1 acc",            "within_1",         pct=True)
    row("Mean abs step error",     "mean_abs_error",   pct=False, decimals=2)
    row("Labeler points early",    "labeler_early",    pct=True)
    row("Labeler points late",     "labeler_late",     pct=True)

    print()
    print("  Interpretation:")
    for label, stats in results:
        if stats is None:
            print(f"    {label}: file not found — run eval_labeler_accuracy.py first")
            continue
        steps = stats["avg_steps"]
        cov   = stats["oracle_coverage"]
        if steps <= 1.2:
            print(f"    {label}: avg {steps:.1f} steps — model collapsed to single-step outputs, "
                  f"labeler eval not meaningful (oracle coverage {cov:.0%})")
        elif cov < 0.1:
            print(f"    {label}: avg {steps:.1f} steps but oracle coverage {cov:.0%} — "
                  f"solutions lack <<expr=result>> annotations")
        else:
            exact = stats.get("exact_match")
            w1    = stats.get("within_1")
            print(f"    {label}: avg {steps:.1f} steps, oracle coverage {cov:.0%}, "
                  f"exact match {fmt(exact)} — labeler {'reliable' if exact and exact >= 0.7 else 'noisy'}")
    print()


if __name__ == "__main__":
    main()
