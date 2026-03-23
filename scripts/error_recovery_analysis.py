"""
Error Recovery Analysis
=======================
Computes error recovery rates across conditions and datasets.
Classifies errors by type and fix-plan concreteness to show
what kinds of errors grounded reflection actually fixes.

Usage:
    python scripts/error_recovery_analysis.py

Output: results/error_recovery_analysis.json + printed summary table.
"""

import json
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results/runs")
OUT_PATH = Path("results/error_recovery_analysis.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def recovery_stats(rows: list[dict]) -> dict:
    wrong = [r for r in rows if not r["first"].get("correct_loose")]
    recovered = [
        r for r in wrong
        if r.get("retry") and r["retry"].get("correct_loose")
    ]
    total_correct = sum(
        1 for r in rows
        if r["first"].get("correct_loose")
        or (r.get("retry") and r["retry"].get("correct_loose"))
    )
    n = len(rows)
    nw = len(wrong)
    nr = len(recovered)
    return {
        "n": n,
        "first_correct": n - nw,
        "first_acc": (n - nw) / n * 100 if n else 0,
        "system_correct": total_correct,
        "system_acc": total_correct / n * 100 if n else 0,
        "wrong": nw,
        "recovered": nr,
        "recovery_rate": nr / nw * 100 if nw else 0,
    }


def normalize_error_type(text: str) -> str:
    """Bucket raw ERROR_TYPE values into canonical categories."""
    if not text:
        return "unknown"
    m = re.search(r"ERROR_TYPE:\s*(\S+)", text)
    if not m:
        return "unknown"
    raw = m.group(1).lower().strip(".,;")
    if "unit" in raw or "conversion" in raw:
        return "unit/conversion"
    if "logic" in raw or "conceptual" in raw or "approach" in raw or "setup" in raw:
        return "logical/setup"
    if (
        "arith" in raw or "calc" in raw or "miscalc" in raw
        or "mistake" in raw or "incorrect" in raw
        or "wrong" in raw or "error" in raw
    ):
        return "arithmetic"
    return "other"


def is_concrete_fix(text: str) -> bool | None:
    """Return True if the FIX_PLAN contains specific numbers/operations."""
    if not text:
        return None
    fix = re.search(r"FIX_PLAN:\s*(.+?)(?:\n|$)", text, re.DOTALL)
    if not fix:
        return None
    plan = fix.group(1).strip()
    signals = re.findall(
        r"\d+|Multiply|Divide|Add|Subtract|percent|times|\bby\b",
        plan, re.IGNORECASE
    )
    return len(signals) >= 2


def fix_plan_step_count(text: str) -> str:
    """Estimate single- vs multi-step fix plan."""
    if not text:
        return "unknown"
    fix = re.search(r"FIX_PLAN:\s*(.+?)(?:\n|$)", text, re.DOTALL)
    if not fix:
        return "unknown"
    plan = fix.group(1).lower()
    step_words = ["then", "next", "after", "finally", "step", "second", "third"]
    count = sum(1 for w in step_words if w in plan)
    return "multi_step" if count >= 2 else "single_step"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse_condition(name: str, template: str, seeds: list[str],
                      datasets: list[tuple[str, str]]) -> dict:
    """
    Returns nested results dict: {dataset: {seed: stats}}.
    template uses SEED and DATASET as placeholders.
    """
    results = {}
    for ds_key, ds_label in datasets:
        results[ds_label] = {}
        for seed in seeds:
            path = RESULTS_DIR / template.replace("SEED", seed).replace("DATASET", ds_key)
            if not path.exists():
                continue
            rows = load_jsonl(path)
            results[ds_label][seed] = recovery_stats(rows)
    return results


def aggregate(cond_results: dict) -> dict:
    """Mean ± std across seeds for each dataset."""
    agg = {}
    for ds_label, by_seed in cond_results.items():
        vals = list(by_seed.values())
        if not vals:
            continue
        keys = ["first_acc", "system_acc", "recovery_rate", "recovered", "wrong"]
        agg[ds_label] = {}
        for k in keys:
            nums = [v[k] for v in vals]
            agg[ds_label][k + "_mean"] = statistics.mean(nums)
            agg[ds_label][k + "_std"] = statistics.stdev(nums) if len(nums) > 1 else 0.0
    return agg


def analyse_error_types(template: str, seeds: list[str],
                         dataset: str) -> dict:
    """
    For grounded reflection: bucket wrong-first-attempt examples by error type
    and compute recovery rate per bucket.
    """
    by_type: dict[str, list[bool]] = defaultdict(list)
    by_concreteness: dict[str, list[bool]] = defaultdict(list)
    by_steps: dict[str, list[bool]] = defaultdict(list)

    for seed in seeds:
        path = RESULTS_DIR / template.replace("SEED", seed).replace("DATASET", dataset)
        if not path.exists():
            continue
        rows = load_jsonl(path)
        for r in rows:
            if r["first"].get("correct_loose"):
                continue
            ref = r.get("reflection") or {}
            text = ref.get("text", "") if isinstance(ref, dict) else ""
            recovered = bool(r.get("retry") and r["retry"].get("correct_loose"))

            et = normalize_error_type(text)
            by_type[et].append(recovered)

            conc = is_concrete_fix(text)
            if conc is not None:
                by_concreteness["concrete" if conc else "vague"].append(recovered)

            steps = fix_plan_step_count(text)
            by_steps[steps].append(recovered)

    def bucket_summary(d):
        return {
            k: {"n": len(v), "recovered": sum(v),
                "recovery_rate": sum(v) / len(v) * 100 if v else 0}
            for k, v in d.items()
        }

    return {
        "by_error_type": bucket_summary(by_type),
        "by_concreteness": bucket_summary(by_concreteness),
        "by_fix_complexity": bucket_summary(by_steps),
    }


def main():
    seeds = ["seed0", "seed1", "seed42"]
    datasets = [("gsm8k", "GSM8K"), ("math", "MATH")]

    conditions = {
        "Retry Only (RLVR)": "baseline_rlvr_retry-r8-SEED_retry_only_DATASET.jsonl",
        "Grounded Reflection (RLVR)": "rrr-grounded-r8-SEED-v7_reflect_full_retry_DATASET.jsonl",
    }

    output = {}
    for cond_name, template in conditions.items():
        raw = analyse_condition(cond_name, template, seeds, datasets)
        output[cond_name] = {
            "per_seed": raw,
            "aggregate": aggregate(raw),
        }

    # Error-type breakdown for grounded reflection
    output["error_type_analysis"] = analyse_error_types(
        "rrr-grounded-r8-SEED-v7_reflect_full_retry_DATASET.jsonl",
        seeds, "gsm8k"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ERROR RECOVERY ANALYSIS")
    print("=" * 72)
    print(f"\n{'Condition':<32} {'DS':<7} {'Attempt1':>9} {'System':>8} {'RecRate':>8}")
    print("-" * 68)
    for cond_name in conditions:
        agg = output[cond_name]["aggregate"]
        for ds_label, _ in datasets:
            if ds_label not in agg:
                continue
            a = agg[ds_label]
            print(
                f"{cond_name:<32} {ds_label:<7} "
                f"{a['first_acc_mean']:>8.1f}% "
                f"{a['system_acc_mean']:>7.1f}% "
                f"{a['recovery_rate_mean']:>7.1f}%"
            )
        print()

    print("\n" + "=" * 72)
    print("ERROR TYPE BREAKDOWN (Grounded Reflection, GSM8K, 3 seeds combined)")
    print("=" * 72)
    eta = output["error_type_analysis"]

    print(f"\nBy error type:")
    print(f"  {'Type':<20} {'N':>5} {'Recov':>6} {'Rate':>7}")
    print("  " + "-" * 38)
    for et, s in sorted(eta["by_error_type"].items()):
        print(f"  {et:<20} {s['n']:>5} {s['recovered']:>6} {s['recovery_rate']:>6.1f}%")

    print(f"\nBy fix-plan concreteness:")
    print(f"  {'Type':<15} {'N':>5} {'Recov':>6} {'Rate':>7}")
    print("  " + "-" * 32)
    for ct, s in sorted(eta["by_concreteness"].items()):
        print(f"  {ct:<15} {s['n']:>5} {s['recovered']:>6} {s['recovery_rate']:>6.1f}%")

    print(f"\nBy fix-plan complexity:")
    print(f"  {'Type':<15} {'N':>5} {'Recov':>6} {'Rate':>7}")
    print("  " + "-" * 32)
    for ct, s in sorted(eta["by_fix_complexity"].items()):
        print(f"  {ct:<15} {s['n']:>5} {s['recovered']:>6} {s['recovery_rate']:>6.1f}%")

    print(f"\nResults saved to {OUT_PATH}\n")


if __name__ == "__main__":
    main()
