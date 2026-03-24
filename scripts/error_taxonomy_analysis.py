"""
Phase 22: Error Taxonomy Analysis (GSM8K + MATH)

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
import os
import collections
import numpy as np

RUNS_DIR = "/sessions/keen-busy-sagan/mnt/lm-reflection-credit/results/runs"

# ─── taxonomy mapping ───────────────────────────────────────────────────────
# GSM8K-dominant types
ARITHMETIC_KEYWORDS = [
    "arithmetic", "calculation", "miscalculation", "arithmetic_error",
    "arithmetic_mistake", "arithmetic_miscalculation", "calculation_error",
    "incorrect_average", "incorrect_calculation",
    # MATH-specific computation errors (specific step wrong, not approach)
    "incorrect_remainder", "incorrect angle calculation",
    "incorrect exponentiation", "missing denominator",
]
SETUP_UNIT_KEYWORDS = [
    "incorrect_units", "incorrect_unit_conversion", "unit_conversion",
    "missing units", "incorrect units conversion",
    "incorrect_conversion", "incorrect_unit",
]
CONCEPTUAL_KEYWORDS = [
    "incorrect reasoning", "logical_error", "reasoning",
    "incorrect_operation", "incorrect_reasoning",
    # MATH-specific structural errors
    "incorrect substitution", "incorrect simplification",
    "incorrect_counting", "incorrect factorization",
    "incorrect application", "incorrect day",
    "substitution", "reflection",
]
FORMAT_KEYWORDS = [
    "incorrect final answer", "incorrect final answer format",
    "incorrect_final_answer", "incorrect_answer",
    "incomplete", "missing closing",
    "incorrect",   # catch-all last — must come after more specific matches
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


def parse_error_type(reflection_text: str):
    if not reflection_text:
        return None
    for line in reflection_text.split("\n"):
        if line.strip().startswith("ERROR_TYPE:"):
            raw = line.replace("ERROR_TYPE:", "").strip()
            return raw
    return None


def load_jsonl(path: str):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ─── file manifests ──────────────────────────────────────────────────────────
GROUNDED_FILES = {
    "gsm8k": {
        0:  "rrr-grounded-r8-seed0-v7_reflect_full_retry_gsm8k.jsonl",
        1:  "rrr-grounded-r8-seed1-v7_reflect_full_retry_gsm8k.jsonl",
        42: "rrr-grounded-r8-seed42-v7_reflect_full_retry_gsm8k.jsonl",
    },
    "math": {
        0:  "rrr-grounded-r8-seed0-v7_reflect_full_retry_math.jsonl",
        1:  "rrr-grounded-r8-seed1-v7_reflect_full_retry_math.jsonl",
        42: "rrr-grounded-r8-seed42-v7_reflect_full_retry_math.jsonl",
    },
}

RETRY_FILES = {
    "gsm8k": {
        0:  "baseline_rlvr_retry-r8-seed0_retry_only_gsm8k.jsonl",
        1:  "baseline_rlvr_retry-r8-seed1_retry_only_gsm8k.jsonl",
        42: "baseline_rlvr_retry-r8-seed42_retry_only_gsm8k.jsonl",
    },
    "math": {
        0:  "baseline_rlvr_retry-r8-seed0_retry_only_math.jsonl",
        1:  "baseline_rlvr_retry-r8-seed1_retry_only_math.jsonl",
        42: "baseline_rlvr_retry-r8-seed42_retry_only_math.jsonl",
    },
}

TAXONOMY_BUCKETS = ["arithmetic_slip", "setup_unit_error", "conceptual"]
CORRECTABLE = {"arithmetic_slip", "setup_unit_error"}

BUCKET_LABELS = {
    "arithmetic_slip":   "Arithmetic Slip  (correctable)",
    "setup_unit_error":  "Setup/Unit Error (correctable)",
    "conceptual":        "Conceptual Error (incorrectable)",
}


# ─── per-dataset analysis ────────────────────────────────────────────────────

def run_taxonomy(dataset: str):
    seed_results = {}

    for seed in [0, 1, 42]:
        g_path = os.path.join(RUNS_DIR, GROUNDED_FILES[dataset][seed])
        r_path = os.path.join(RUNS_DIR, RETRY_FILES[dataset][seed])

        grounded   = {ex["idx"]: ex for ex in load_jsonl(g_path)}
        retry_only = {ex["idx"]: ex for ex in load_jsonl(r_path)}

        bucket_grounded = collections.defaultdict(lambda: {"wrong": 0, "recovered": 0})
        bucket_retry    = collections.defaultdict(lambda: {"wrong": 0, "recovered": 0})
        raw_type_counts = collections.Counter()
        n_no_error_type = 0
        n_format_ambiguous = 0

        for idx, ex in grounded.items():
            first_correct = ex["first"].get("correct_strict", False)
            if first_correct:
                continue

            refl_text = (ex.get("reflection") or {}).get("text", "") or ""
            raw_et = parse_error_type(refl_text)
            if raw_et is None:
                n_no_error_type += 1
                continue
            bucket = classify_error_type(raw_et)
            raw_type_counts[raw_et] += 1

            if bucket == "format_ambiguous":
                n_format_ambiguous += 1
                continue
            if bucket == "other":
                bucket = "conceptual"

            retry_correct = ex["retry"].get("correct_strict", False)
            bucket_grounded[bucket]["wrong"] += 1
            if retry_correct:
                bucket_grounded[bucket]["recovered"] += 1

            if idx in retry_only:
                r_ex = retry_only[idx]
                r_first_correct = (r_ex.get("first") or {}).get("correct_strict", False)
                r_retry = r_ex.get("retry")
                r_retry_correct = (r_retry or {}).get("correct_strict", False)
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

    return seed_results


def aggregate_bucket(seed_results, condition):
    totals = collections.defaultdict(lambda: {"wrong": 0, "recovered": 0})
    for seed, data in seed_results.items():
        for bucket, vals in data[condition].items():
            totals[bucket]["wrong"]     += vals["wrong"]
            totals[bucket]["recovered"] += vals["recovered"]
    result = {}
    for bucket, vals in totals.items():
        w, r = vals["wrong"], vals["recovered"]
        result[bucket] = {"wrong": w, "recovered": r, "rate": (r / w * 100) if w > 0 else 0.0}
    return result


def per_seed_rates(seed_results, condition):
    rates = collections.defaultdict(list)
    for seed, data in seed_results.items():
        for bucket, vals in data[condition].items():
            w = vals["wrong"]
            r = vals["recovered"]
            rates[bucket].append(r / w * 100 if w > 0 else 0.0)
    return {b: (float(np.mean(v)), float(np.std(v))) for b, v in rates.items()}


def print_results(dataset: str, seed_results: dict):
    agg_g = aggregate_bucket(seed_results, "grounded")
    agg_r = aggregate_bucket(seed_results, "retry")
    rates_g = per_seed_rates(seed_results, "grounded")
    rates_r = per_seed_rates(seed_results, "retry")

    print("\n" + "=" * 72)
    print(f"PHASE 22: ERROR TAXONOMY ANALYSIS — {dataset.upper()}")
    print("=" * 72)
    print("%s  %6s  %22s  %15s" % ("Bucket".ljust(35), "Wrong", "Grounded Recovery", "Retry-Only"))
    print("-" * 84)

    for bucket in TAXONOMY_BUCKETS:
        label = BUCKET_LABELS[bucket]
        ag = agg_g.get(bucket, {"wrong": 0, "recovered": 0})
        g_mean, g_std = rates_g.get(bucket, (0.0, 0.0))
        r_mean, r_std = rates_r.get(bucket, (0.0, 0.0))
        print("%s  %5d   %5.1f+/-%4.1f%%   (%d/%d)   %5.1f+/-%4.1f%%" % (
            label.ljust(35), ag["wrong"],
            g_mean, g_std, ag["recovered"], ag["wrong"],
            r_mean, r_std,
        ))

    print()
    corr_g = {"wrong": 0, "recovered": 0}
    corr_r = {"wrong": 0, "recovered": 0}
    for b in CORRECTABLE:
        for k in ["wrong", "recovered"]:
            corr_g[k] += agg_g.get(b, {}).get(k, 0)
            corr_r[k] += agg_r.get(b, {}).get(k, 0)

    incorr_g = agg_g.get("conceptual", {"wrong": 0, "recovered": 0})
    corr_g_rate   = corr_g["recovered"]   / corr_g["wrong"]   * 100 if corr_g["wrong"]   else 0
    corr_r_rate   = corr_r["recovered"]   / corr_r["wrong"]   * 100 if corr_r["wrong"]   else 0
    incorr_g_rate = incorr_g["recovered"] / incorr_g["wrong"] * 100 if incorr_g["wrong"] else 0

    print("%-35s  %5d   %5.1f%%   (%d/%d)   %5.1f%%" % (
        "CORRECTABLE (combined)", corr_g["wrong"],
        corr_g_rate, corr_g["recovered"], corr_g["wrong"], corr_r_rate,
    ))
    print("%-35s  %5d   %5.1f%%   (%d/%d)" % (
        "INCORRECTABLE (conceptual)", incorr_g["wrong"],
        incorr_g_rate, incorr_g["recovered"], incorr_g["wrong"],
    ))

    if incorr_g_rate > 0:
        print("\n  -> Grounded: %.1fx higher recovery on correctable vs conceptual" % (
            corr_g_rate / incorr_g_rate))
    else:
        print("\n  -> Grounded: %.1f%% correctable recovery vs 0%% conceptual (inf lift)" % corr_g_rate)

    # error type distribution
    all_raw = collections.Counter()
    n_ambiguous = sum(s["n_format_ambiguous"] for s in seed_results.values())
    for s in seed_results.values():
        all_raw.update(s["raw_type_counts"])
    print("\nRaw error type distribution (top 12, grounded, all seeds):")
    for k, v in all_raw.most_common(12):
        b = classify_error_type(k)
        print("  %4d  [%-20s]  %s" % (v, b, k))
    print("  %4d  [%-20s]  (various)" % (n_ambiguous, "format_ambiguous"))

    return {
        "aggregate_grounded": agg_g,
        "aggregate_retry": agg_r,
        "correctable": {"grounded_rate": corr_g_rate, "retry_rate": corr_r_rate,
                        "wrong": corr_g["wrong"], "recovered": corr_g["recovered"]},
        "incorrectable": {"grounded_rate": incorr_g_rate,
                          "wrong": incorr_g["wrong"], "recovered": incorr_g["recovered"]},
        "per_seed_rates_grounded": rates_g,
        "per_seed_rates_retry": rates_r,
    }


# ─── run both datasets ───────────────────────────────────────────────────────
all_output = {}
for ds in ["gsm8k", "math"]:
    sr = run_taxonomy(ds)
    result = print_results(ds, sr)
    all_output[ds] = result

# ─── cross-dataset summary ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("CROSS-DATASET SUMMARY")
print("=" * 72)
print("%-12s  %25s  %25s" % ("", "Correctable recovery", "Conceptual recovery"))
print("%-12s  %12s  %12s  %12s  %12s" % ("Dataset", "Grounded", "Retry-only", "Grounded", "Retry-only"))
print("-" * 72)
for ds in ["gsm8k", "math"]:
    c  = all_output[ds]["correctable"]
    ic = all_output[ds]["incorrectable"]
    r_rates = all_output[ds]["aggregate_retry"]
    r_incorr = r_rates.get("conceptual", {}).get("rate", 0.0)
    print("%-12s  %11.1f%%  %11.1f%%  %11.1f%%  %11.1f%%" % (
        ds.upper(), c["grounded_rate"], c["retry_rate"],
        ic["grounded_rate"], r_incorr,
    ))

# ─── save output ─────────────────────────────────────────────────────────────
out_path = "/sessions/keen-busy-sagan/mnt/lm-reflection-credit/results/error_taxonomy_analysis.json"
with open(out_path, "w") as f:
    json.dump(all_output, f, indent=2)
print(f"\nSaved to {out_path}")
