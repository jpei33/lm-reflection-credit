"""
Phase 22: Error Taxonomy Analysis

Classify first-attempt errors by type (correctable vs incorrectable) and
show that grounded reflection disproportionately recovers correctable errors.

Taxonomy buckets:
  1. arithmetic_slip    — pure calculation error (correctable)
  2. setup_unit_error   — wrong setup, misread, unit conversion (correctable)
  3. conceptual         — wrong reasoning approach (borderline / incorrectable)
  4. format_ambiguous   — wrong output format / ambiguous tag (set aside)

Uses ERROR_TYPE field from grounded reflection text for classification.
Then cross-references same question indices against retry-only condition
to show differential recovery rates.
"""

import json
import re
import os
import collections

RUNS_DIR = "/sessions/keen-busy-sagan/mnt/lm-reflection-credit/results/runs"

# ─── taxonomy mapping ───────────────────────────────────────────────────────
ARITHMETIC_KEYWORDS = [
    "arithmetic", "calculation", "miscalculation", "arithmetic_error",
    "arithmetic_mistake", "arithmetic_miscalculation", "calculation_error",
    "incorrect_average", "incorrect_calculation",
]
SETUP_UNIT_KEYWORDS = [
    "incorrect_units", "incorrect_unit_conversion", "unit_conversion",
    "missing units", "incorrect units conversion", "unit",
    "incorrect probability calculation",
]
CONCEPTUAL_KEYWORDS = [
    "incorrect reasoning", "logical_error", "reasoning",
    "incorrect_operation", "incorrect_reasoning",
]
FORMAT_KEYWORDS = [
    "incorrect final answer", "incorrect final answer format",
    "incorrect_final_answer", "incorrect",
]

def classify_error_type(raw_type: str) -> str:
    t = raw_type.strip().lower()
    for kw in ARITHMETIC_KEYWORDS:
        if kw in t:
            return "arithmetic_slip"
    for kw in SETUP_UNIT_KEYWORDS:
        if kw in t:
            return "setup_unit_error"
    for kw in CONCEPTUAL_KEYWORDS:
        if kw in t:
            return "conceptual"
    for kw in FORMAT_KEYWORDS:
        if kw in t:
            return "format_ambiguous"
    return "other"


def parse_error_type(reflection_text: str) -> str | None:
    if not reflection_text:
        return None
    for line in reflection_text.split("\n"):
        if line.strip().startswith("ERROR_TYPE:"):
            raw = line.replace("ERROR_TYPE:", "").strip()
            return raw
    return None


def load_jsonl(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ─── load grounded condition (seeds 0, 1, 42) ───────────────────────────────
GROUNDED_FILES = {
    0:  "rrr-grounded-r8-seed0-v7_reflect_full_retry_gsm8k.jsonl",
    1:  "rrr-grounded-r8-seed1-v7_reflect_full_retry_gsm8k.jsonl",
    42: "rrr-grounded-r8-seed42-v7_reflect_full_retry_gsm8k.jsonl",
}

# ─── load retry-only condition (seeds 0, 1, 42) ─────────────────────────────
RETRY_FILES = {
    0:  "baseline_rlvr_retry-r8-seed0_retry_only_gsm8k.jsonl",
    1:  "baseline_rlvr_retry-r8-seed1_retry_only_gsm8k.jsonl",
    42: "baseline_rlvr_retry-r8-seed42_retry_only_gsm8k.jsonl",
}

# ─── per-seed analysis ───────────────────────────────────────────────────────

# We store per-seed results and then aggregate
seed_results = {}

TAXONOMY_BUCKETS = ["arithmetic_slip", "setup_unit_error", "conceptual"]
CORRECTABLE = {"arithmetic_slip", "setup_unit_error"}

for seed in [0, 1, 42]:
    g_path = os.path.join(RUNS_DIR, GROUNDED_FILES[seed])
    r_path = os.path.join(RUNS_DIR, RETRY_FILES[seed])

    grounded = {ex["idx"]: ex for ex in load_jsonl(g_path)}
    retry_only = {ex["idx"]: ex for ex in load_jsonl(r_path)}

    # For each question where grounded model's first attempt was WRONG,
    # classify the error type and record whether recovery happened
    # in grounded vs retry-only
    bucket_grounded = collections.defaultdict(lambda: {"wrong": 0, "recovered": 0})
    bucket_retry    = collections.defaultdict(lambda: {"wrong": 0, "recovered": 0})
    raw_type_counts = collections.Counter()

    n_no_error_type = 0
    n_format_ambiguous = 0

    for idx, ex in grounded.items():
        first_correct = ex["first"].get("correct_strict", False)
        if first_correct:
            continue  # only interested in errors

        # extract error type
        refl_text = (ex.get("reflection") or {}).get("text", "") or ""
        raw_et = parse_error_type(refl_text)
        if raw_et is None:
            n_no_error_type += 1
            continue
        bucket = classify_error_type(raw_et)
        raw_type_counts[raw_et] += 1

        if bucket == "format_ambiguous":
            n_format_ambiguous += 1
            # set aside — don't include in taxonomy table
            continue
        if bucket == "other":
            # absorb unknowns into conceptual for conservatism
            bucket = "conceptual"

        # grounded model: did retry recover?
        retry_correct = ex["retry"].get("correct_strict", False)
        bucket_grounded[bucket]["wrong"] += 1
        if retry_correct:
            bucket_grounded[bucket]["recovered"] += 1

        # retry-only model: for same idx, did it get first wrong + recover?
        if idx in retry_only:
            r_ex = retry_only[idx]
            r_first_correct = (r_ex.get("first") or {}).get("correct_strict", False)
            r_retry = r_ex.get("retry")
            r_retry_correct = (r_retry or {}).get("correct_strict", False)
            # Use the same idx set (same error type classification from grounded)
            # but only count as wrong if retry-only also got first wrong
            if not r_first_correct:
                bucket_retry[bucket]["wrong"] += 1
                if r_retry_correct:
                    bucket_retry[bucket]["recovered"] += 1

    seed_results[seed] = {
        "grounded": {b: dict(v) for b, v in bucket_grounded.items()},
        "retry":    {b: dict(v) for b, v in bucket_retry.items()},
        "raw_type_counts": dict(raw_type_counts),
        "n_no_error_type": n_no_error_type,
        "n_format_ambiguous": n_format_ambiguous,
    }

# ─── aggregate across seeds ──────────────────────────────────────────────────

def aggregate_bucket(seed_results, condition):
    """Return per-bucket aggregates across 3 seeds."""
    totals = collections.defaultdict(lambda: {"wrong": 0, "recovered": 0})
    for seed, data in seed_results.items():
        for bucket, vals in data[condition].items():
            totals[bucket]["wrong"]    += vals["wrong"]
            totals[bucket]["recovered"] += vals["recovered"]
    result = {}
    for bucket, vals in totals.items():
        w = vals["wrong"]
        r = vals["recovered"]
        result[bucket] = {
            "wrong": w,
            "recovered": r,
            "rate": (r / w * 100) if w > 0 else 0.0,
        }
    return result

agg_grounded = aggregate_bucket(seed_results, "grounded")
agg_retry    = aggregate_bucket(seed_results, "retry")

# ─── also get per-seed rates for std calculation ─────────────────────────────
import numpy as np

def per_seed_rates(seed_results, condition):
    rates = collections.defaultdict(list)
    for seed, data in seed_results.items():
        for bucket, vals in data[condition].items():
            w = vals["wrong"]
            r = vals["recovered"]
            if w > 0:
                rates[bucket].append(r / w * 100)
            else:
                rates[bucket].append(0.0)
    return {b: (np.mean(v), np.std(v)) for b, v in rates.items()}

rates_g = per_seed_rates(seed_results, "grounded")
rates_r = per_seed_rates(seed_results, "retry")

# ─── print results ───────────────────────────────────────────────────────────

BUCKET_LABELS = {
    "arithmetic_slip":   "Arithmetic Slip  (correctable)",
    "setup_unit_error":  "Setup/Unit Error (correctable)",
    "conceptual":        "Conceptual Error (incorrectable)",
}

print("\n" + "=" * 72)
print("PHASE 22: ERROR TAXONOMY ANALYSIS — GSM8K")
print("=" * 72)
print(f"\n{'Bucket':<35} {'Wrong':>6}  {'Grounded Recovery':>20}  {'Retry-Only':>15}")
print("-" * 80)

for bucket in TAXONOMY_BUCKETS:
    label = BUCKET_LABELS[bucket]
    ag = agg_grounded.get(bucket, {"wrong":0,"recovered":0,"rate":0})
    ar = agg_retry.get(bucket,    {"wrong":0,"recovered":0,"rate":0})
    g_mean, g_std = rates_g.get(bucket, (0, 0))
    r_mean, r_std = rates_r.get(bucket, (0, 0))
    print(f"{label:<35}  {ag['wrong']:>5}   {g_mean:5.1f}±{g_std:.1f}%   ({ag['recovered']}/{ag['wrong']})   {r_mean:5.1f}±{r_std:.1f}%")

print()
# Correctable summary
corr_g = {"wrong": 0, "recovered": 0}
corr_r = {"wrong": 0, "recovered": 0}
for b in CORRECTABLE:
    for k in ["wrong", "recovered"]:
        corr_g[k] += agg_grounded.get(b, {}).get(k, 0)
        corr_r[k] += agg_retry.get(b, {}).get(k, 0)

incorr_g = agg_grounded.get("conceptual", {"wrong":0,"recovered":0})
incorr_r = agg_retry.get("conceptual",    {"wrong":0,"recovered":0})

corr_g_rate  = corr_g["recovered"]  / corr_g["wrong"]  * 100 if corr_g["wrong"]  else 0
corr_r_rate  = corr_r["recovered"]  / corr_r["wrong"]  * 100 if corr_r["wrong"]  else 0
incorr_g_rate = incorr_g["recovered"] / incorr_g["wrong"] * 100 if incorr_g["wrong"] else 0
incorr_r_rate = incorr_r["recovered"] / incorr_r["wrong"] * 100 if incorr_r["wrong"] else 0

print(f"{'CORRECTABLE (combined)':<35}  {corr_g['wrong']:>5}   {corr_g_rate:5.1f}%   ({corr_g['recovered']}/{corr_g['wrong']})   {corr_r_rate:5.1f}%")
print(f"{'INCORRECTABLE (conceptual)':<35}  {incorr_g['wrong']:>5}   {incorr_g_rate:5.1f}%   ({incorr_g['recovered']}/{incorr_g['wrong']})   {incorr_r_rate:5.1f}%")

if corr_g_rate > 0 and incorr_g_rate >= 0:
    if incorr_g_rate > 0:
        ratio = corr_g_rate / incorr_g_rate
        print(f"\n  → Grounded reflects {ratio:.1f}× higher recovery on correctable vs conceptual errors")
    else:
        print(f"\n  → Grounded recovers {corr_g_rate:.1f}% of correctable errors vs 0% of conceptual (∞× lift)")

print("\nRaw error type distribution (grounded condition, all seeds):")
all_raw = collections.Counter()
for s in seed_results.values():
    all_raw.update(s["raw_type_counts"])
for k, v in all_raw.most_common(15):
    bucket = classify_error_type(k)
    print(f"  {v:4d}  [{bucket:<20}]  {k}")

# ─── save JSON output ────────────────────────────────────────────────────────
output = {
    "per_seed": {str(s): {
        "grounded": seed_results[s]["grounded"],
        "retry":    seed_results[s]["retry"],
    } for s in seed_results},
    "aggregate": {
        "grounded": agg_grounded,
        "retry": agg_retry,
        "correctable_grounded": {"wrong": corr_g["wrong"], "recovered": corr_g["recovered"], "rate": corr_g_rate},
        "correctable_retry":    {"wrong": corr_r["wrong"], "recovered": corr_r["recovered"], "rate": corr_r_rate},
        "incorrectable_grounded": {"wrong": incorr_g["wrong"], "recovered": incorr_g["recovered"], "rate": incorr_g_rate},
        "incorrectable_retry":    {"wrong": incorr_r["wrong"], "recovered": incorr_r["recovered"], "rate": incorr_r_rate},
    },
    "per_seed_rates_grounded": {b: {"mean": float(m), "std": float(s)} for b, (m, s) in rates_g.items()},
    "per_seed_rates_retry":    {b: {"mean": float(m), "std": float(s)} for b, (m, s) in rates_r.items()},
}

out_path = "/sessions/keen-busy-sagan/mnt/lm-reflection-credit/results/error_taxonomy_analysis.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to {out_path}")
