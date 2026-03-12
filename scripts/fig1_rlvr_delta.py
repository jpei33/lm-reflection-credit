"""
fig1_rlvr_delta.py  (v2 — with mean ± std error bars, 3 seeds per condition)

Bar chart of RLVR lift over the matched SFT baseline for each condition,
GSM8K (left panel) and MATH (right panel).

Error bars show ±1 std across seeds 0, 1, 42.
SFT baselines are single checkpoints (no seed variance for SFT).
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statistics import mean, stdev

BASE = "/sessions/keen-busy-sagan/mnt/lm-reflection-credit/results/runs/"
OUT  = "/sessions/keen-busy-sagan/mnt/lm-reflection-credit/figures/fig1_rlvr_delta.png"

# ── helpers ───────────────────────────────────────────────────────────────────
def system_acc(path):
    c = t = 0
    try:
        with open(path) as f:
            for line in f:
                r = json.loads(line); t += 1
                retry = r.get("retry") or {}
                first = r.get("first") or {}
                if retry:
                    c += int(bool(retry.get("correct_strict") or retry.get("correct_loose")))
                else:
                    c += int(bool(first.get("correct_strict") or first.get("correct_loose")))
    except FileNotFoundError:
        return None
    return 100 * c / t if t else None

def acc(run, suffix):
    return system_acc(BASE + f"{run}_{suffix}.jsonl")

# ── SFT baselines (single checkpoint, treated as fixed) ─────────────────────
SFT = {
    "Baseline\nCoT":          (acc("qwen3-4b-baseline_solve-r8-seed42",           "gsm8k"),
                                acc("qwen3-4b-baseline_solve-r8-seed42",           "math")),
    "Retry\nOnly":            (acc("qwen3-4b-retry_only-r8-seed42",               "gsm8k"),
                                acc("qwen3-4b-retry_only-r8-seed42",               "math")),
    "Step\nCredit":           (acc("qwen3-4b-retry_only-r8-seed42",               "gsm8k"),
                                acc("qwen3-4b-retry_only-r8-seed42",               "math")),
    "Reflect-Plan\n+Retry":   (acc("qwen3-4b-reflect_plan_retry-r8-seed42",       "gsm8k"),
                                acc("qwen3-4b-reflect_plan_retry-r8-seed42",       "math")),
    "Reflect-Full\n+Retry":   (acc("qwen3-4b-reflect_full_retry-r8-seed42",
                                   "reflect_full_retry_fewshot_gsm8k"),
                                acc("qwen3-4b-reflect_full_retry-r8-seed42",
                                   "reflect_full_retry_fewshot_math")),
}

# ── RLVR per-seed ─────────────────────────────────────────────────────────────
RLVR_SEEDS = {
    "Baseline\nCoT":         [("baseline_rlvr_solve-r8-seed{s}",         "gsm8k",                    "math")],
    "Retry\nOnly":           [("baseline_rlvr_retry-r8-seed{s}",         "retry_only_gsm8k",         "retry_only_math")],
    "Step\nCredit":          [("step_credit-r8-seed{s}",                  "retry_only_gsm8k",         "retry_only_math")],
    "Reflect-Plan\n+Retry":  [("rrr-plan-r8-seed{s}",                    "reflect_plan_retry_gsm8k", "reflect_plan_retry_math")],
    "Reflect-Full\n+Retry":  [("rrr-full-r8-seed{s}",                    "reflect_full_retry_gsm8k", "reflect_full_retry_math")],
}

def rlvr_stats(label):
    entries = RLVR_SEEDS[label]
    g_vals, m_vals = [], []
    for tmpl, g_suf, m_suf in entries:
        for s in [0, 1, 42]:
            run = tmpl.format(s=s)
            g = acc(run, g_suf); m = acc(run, m_suf)
            if g is not None: g_vals.append(g)
            if m is not None: m_vals.append(m)
    gm = mean(g_vals) if g_vals else None
    gs = stdev(g_vals) if len(g_vals) > 1 else 0.0
    mm = mean(m_vals) if m_vals else None
    ms = stdev(m_vals) if len(m_vals) > 1 else 0.0
    return gm, gs, mm, ms

# ── assemble deltas ───────────────────────────────────────────────────────────
labels = list(SFT.keys())
g_delta, g_err, m_delta, m_err = [], [], [], []

for lbl in labels:
    sft_g, sft_m = SFT[lbl]
    gm, gs, mm, ms = rlvr_stats(lbl)
    g_delta.append(gm - sft_g if (gm is not None and sft_g is not None) else 0)
    g_err.append(gs)
    m_delta.append(mm - sft_m if (mm is not None and sft_m is not None) else 0)
    m_err.append(ms)

# ── colours: positive → blue/green, negative → red/orange ────────────────────
def bar_color(v, ds="gsm"):
    if v > 0:
        return "#4C72B0" if ds == "gsm" else "#55A868"
    return "#C44E52" if ds == "gsm" else "#DD8452"

g_colors = [bar_color(v, "gsm") for v in g_delta]
m_colors = [bar_color(v, "mth") for v in m_delta]

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.patch.set_facecolor("white")

x = np.arange(len(labels))
bw = 0.55

for ax, deltas, errs, colors, title, dataset in [
    (ax1, g_delta, g_err, g_colors, "GSM8K", "gsm"),
    (ax2, m_delta, m_err, m_colors, "MATH",  "mth"),
]:
    bars = ax.bar(x, deltas, bw, color=colors, zorder=3,
                  error_kw=dict(ecolor="#333", capsize=5, capthick=1.2, elinewidth=1.2))
    ax.errorbar(x, deltas, yerr=errs, fmt="none",
                ecolor="#333", capsize=5, capthick=1.2, elinewidth=1.2, zorder=4)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RLVR lift over SFT (pp)")
    ax.set_title(f"{title}  —  RLVR Δ over SFT  (mean ± std, 3 seeds)",
                 fontsize=10, fontweight="bold")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # value labels
    for i, (v, e) in enumerate(zip(deltas, errs)):
        sign = "+" if v >= 0 else ""
        ypos = v + e + 0.15 if v >= 0 else v - e - 0.35
        va = "bottom" if v >= 0 else "top"
        ax.text(i, ypos, f"{sign}{v:.1f}pp", ha="center", va=va, fontsize=8.5,
                fontweight="bold", color="#222")

# legend
pos_g = mpatches.Patch(color="#4C72B0", label="Positive lift (GSM8K)")
neg_g = mpatches.Patch(color="#C44E52", label="Negative lift (GSM8K)")
pos_m = mpatches.Patch(color="#55A868", label="Positive lift (MATH)")
neg_m = mpatches.Patch(color="#DD8452", label="Negative lift (MATH)")
fig.legend(handles=[pos_g, neg_g, pos_m, neg_m],
           loc="lower center", ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.04))

fig.text(0.5, -0.09,
    "Figure 1.  RLVR lift (pp) over matched SFT baseline per condition. "
    "Error bars show ±1 std across 3 independent seeds. "
    "Retry Only and Reflect-Full show consistent positive lift on GSM8K; "
    "Reflect-Full is the only condition with positive lift on both benchmarks. "
    "Step Credit and Reflect-Plan degrade on MATH; Baseline CoT degrades on GSM8K.",
    ha="center", fontsize=8, color="#555")

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
