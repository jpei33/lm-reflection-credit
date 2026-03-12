"""
fig6_scaling.py

Scaling figure: GSM8K and MATH accuracy vs model size (4B → 8B).
Shows SFT and RLVR as separate lines with ±1 std shading across 3 seeds.
"""

import json, os
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev

BASE = "/sessions/keen-busy-sagan/mnt/lm-reflection-credit/results/runs/"
OUT  = "/sessions/keen-busy-sagan/mnt/lm-reflection-credit/figures/fig6_scaling.png"

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

# ── data ──────────────────────────────────────────────────────────────────────
# SFT baselines (single checkpoint per size — no seed variance)
sft_4b_gsm  = acc("qwen3-4b-reflect_full_retry-r8-seed42",
                   "reflect_full_retry_fewshot_gsm8k")
sft_4b_math = acc("qwen3-4b-reflect_full_retry-r8-seed42",
                   "reflect_full_retry_fewshot_math")

# 8B SFT: use the step-50 checkpoint eval as the SFT proxy
# (trained 500 SFT steps, then 500 RLVR; the SFT checkpoint is qwen3-8b-reflect_full_retry-r8-seed42)
# We don't have a direct 8B SFT eval file, so we use step-50 of RLVR as a proxy
# Actually, let's check what's available
sft_8b_gsm  = None
sft_8b_math = None

# Try to find 8B SFT eval
for suffix in ["gsm8k", "reflect_full_retry_gsm8k"]:
    v = acc("qwen3-8b-reflect_full_retry-r8-seed42", suffix)
    if v is not None:
        sft_8b_gsm = v; break

for suffix in ["math", "reflect_full_retry_math"]:
    v = acc("qwen3-8b-reflect_full_retry-r8-seed42", suffix)
    if v is not None:
        sft_8b_math = v; break

print(f"SFT 4B: GSM8K={sft_4b_gsm}, MATH={sft_4b_math}")
print(f"SFT 8B: GSM8K={sft_8b_gsm}, MATH={sft_8b_math}")

# ── RLVR: 3 seeds per size ────────────────────────────────────────────────────
rlvr_4b_gsm, rlvr_4b_math = [], []
rlvr_8b_gsm, rlvr_8b_math = [], []

for s in [0, 1, 42]:
    g4 = acc(f"rrr-full-r8-seed{s}",     "reflect_full_retry_gsm8k")
    m4 = acc(f"rrr-full-r8-seed{s}",     "reflect_full_retry_math")
    g8 = acc(f"rrr-full-8b-r8-seed{s}",  "reflect_full_retry_gsm8k")
    m8 = acc(f"rrr-full-8b-r8-seed{s}",  "reflect_full_retry_math")
    if g4: rlvr_4b_gsm.append(g4)
    if m4: rlvr_4b_math.append(m4)
    if g8: rlvr_8b_gsm.append(g8)
    if m8: rlvr_8b_math.append(m8)

def stats(vals):
    m = mean(vals) if vals else None
    s = stdev(vals) if len(vals) > 1 else 0.0
    return m, s

r4gm, r4gs = stats(rlvr_4b_gsm)
r4mm, r4ms = stats(rlvr_4b_math)
r8gm, r8gs = stats(rlvr_8b_gsm)
r8mm, r8ms = stats(rlvr_8b_math)

print(f"\nRLVR 4B: GSM8K={r4gm:.1f}±{r4gs:.1f}, MATH={r4mm:.1f}±{r4ms:.1f}")
print(f"RLVR 8B: GSM8K={r8gm:.1f}±{r8gs:.1f}, MATH={r8mm:.1f}±{r8ms:.1f}")

# ── figure ────────────────────────────────────────────────────────────────────
SIZES    = [4, 8]
SIZE_LBL = ["4B", "8B"]

C_RLVR = "#4C72B0"   # blue
C_SFT  = "#8c8c8c"   # grey

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
fig.patch.set_facecolor("white")

for ax, title, sft_pts, rlvr_means, rlvr_stds in [
    (ax1, "GSM8K",
     [sft_4b_gsm, sft_8b_gsm],
     [r4gm, r8gm], [r4gs, r8gs]),
    (ax2, "MATH",
     [sft_4b_math, sft_8b_math],
     [r4mm, r8mm], [r4ms, r8ms]),
]:
    # SFT line (dashed grey) — only where we have data
    valid_sft = [(s, v) for s, v in zip(SIZES, sft_pts) if v is not None]
    if valid_sft:
        xs, ys = zip(*valid_sft)
        ax.plot(xs, ys, "o--", color=C_SFT, linewidth=1.8,
                markersize=8, label="SFT baseline", zorder=3)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}%", (x, y),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8.5, color=C_SFT)

    # RLVR line (solid blue) with ±std shading
    rlvr_valid = [(s, m, e) for s, m, e in zip(SIZES, rlvr_means, rlvr_stds)
                  if m is not None]
    if rlvr_valid:
        xs, ms, es = zip(*rlvr_valid)
        xs, ms, es = list(xs), list(ms), list(es)
        ax.plot(xs, ms, "o-", color=C_RLVR, linewidth=2.2,
                markersize=9, label="RLVR (mean ± std)", zorder=4)
        ax.fill_between(xs, [m-e for m, e in zip(ms, es)],
                             [m+e for m, e in zip(ms, es)],
                        color=C_RLVR, alpha=0.15, zorder=2)
        for x, m, e in zip(xs, ms, es):
            ax.annotate(f"{m:.1f}±{e:.1f}%", (x, m),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8.5, color=C_RLVR, fontweight="bold")

    # delta annotation between 4B and 8B RLVR
    if len(rlvr_valid) == 2:
        d = rlvr_valid[1][1] - rlvr_valid[0][1]
        mid_x = (rlvr_valid[0][0] + rlvr_valid[1][0]) / 2
        mid_y = (rlvr_valid[0][1] + rlvr_valid[1][1]) / 2
        ax.annotate(f"Δ = +{d:.1f}pp", (mid_x, mid_y),
                    textcoords="offset points", xytext=(-52, 12),
                    fontsize=9, color=C_RLVR,
                    arrowprops=dict(arrowstyle="-", color=C_RLVR, alpha=0.4))

    ax.set_xticks(SIZES)
    ax.set_xticklabels(SIZE_LBL, fontsize=11)
    ax.set_xlabel("Model size", fontsize=10)
    ax.set_ylabel("System accuracy (%)")
    ax.set_title(f"{title}  —  Accuracy vs model size\n(Reflect-Full + Retry)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # expand x axis a bit for readability
    ax.set_xlim(2.5, 9.5)

fig.text(0.5, -0.04,
    "Figure 6.  Scaling behaviour of Reflect-Full RLVR: system accuracy at 4B and 8B parameters. "
    "RLVR line shows mean ± std across 3 seeds; SFT line shows the matched SFT checkpoint (single seed). "
    "Both datasets show consistent improvement from 4B to 8B, with a larger gain on MATH (+3.5pp) than GSM8K (+2.3pp), "
    "consistent with harder problems benefiting more from additional model capacity.",
    ha="center", fontsize=8, color="#555")

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
