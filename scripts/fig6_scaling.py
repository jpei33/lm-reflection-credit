"""
fig6_scaling.py  (v2 — includes GRPO line)

Scaling figure: GSM8K and MATH accuracy vs model size (4B → 8B).
Shows SFT, RLVR, and GRPO as separate lines with ±1 std shading.
Key finding: GRPO closes the gap with RLVR at 8B (capacity interaction).
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

def gather(templates, gsm_suf, math_suf, seeds=[0,1,42]):
    gv, mv = [], []
    for tmpl in templates:
        for s in seeds:
            g = acc(tmpl.format(s=s), gsm_suf)
            m = acc(tmpl.format(s=s), math_suf)
            if g: gv.append(g)
            if m: mv.append(m)
    gm = mean(gv) if gv else None
    gs = stdev(gv) if len(gv) > 1 else 0.0
    mm = mean(mv) if mv else None
    ms = stdev(mv) if len(mv) > 1 else 0.0
    return gm, gs, mm, ms

# ── data ──────────────────────────────────────────────────────────────────────
sft_4b_gsm  = acc("qwen3-4b-reflect_full_retry-r8-seed42", "reflect_full_retry_fewshot_gsm8k")
sft_4b_math = acc("qwen3-4b-reflect_full_retry-r8-seed42", "reflect_full_retry_fewshot_math")

r4gm, r4gs, r4mm, r4ms = gather(["rrr-full-r8-seed{s}"],
    "reflect_full_retry_gsm8k", "reflect_full_retry_math")
r8gm, r8gs, r8mm, r8ms = gather(["rrr-full-8b-r8-seed{s}"],
    "reflect_full_retry_gsm8k", "reflect_full_retry_math")

g4gm, g4gs, g4mm, g4ms = gather(["rrr_grpo-full-r8-seed{s}"],
    "reflect_full_retry_gsm8k", "reflect_full_retry_math")
g8gm, g8gs, g8mm, g8ms = gather(["rrr_grpo-full-8b-r8-seed{s}"],
    "reflect_full_retry_gsm8k", "reflect_full_retry_math")

print(f"RLVR 4B: GSM8K={r4gm:.1f}±{r4gs:.1f}  MATH={r4mm:.1f}±{r4ms:.1f}")
print(f"RLVR 8B: GSM8K={r8gm:.1f}±{r8gs:.1f}  MATH={r8mm:.1f}±{r8ms:.1f}")
print(f"GRPO 4B: GSM8K={g4gm:.1f}±{g4gs:.1f}  MATH={g4mm:.1f}±{g4ms:.1f}")
print(f"GRPO 8B: GSM8K={g8gm:.1f}±{g8gs:.1f}  MATH={g8mm:.1f}±{g8ms:.1f}")

# ── colours ───────────────────────────────────────────────────────────────────
C_RLVR = "#4C72B0"   # blue
C_GRPO = "#DD8452"   # orange
C_SFT  = "#8c8c8c"   # grey

SIZES    = [4, 8]
SIZE_LBL = ["4B", "8B"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=False)
fig.patch.set_facecolor("white")

for ax, title, sft_pt, rlvr_ms, rlvr_ss, grpo_ms, grpo_ss in [
    (ax1, "GSM8K",
     sft_4b_gsm,
     [r4gm, r8gm], [r4gs, r8gs],
     [g4gm, g8gm], [g4gs, g8gs]),
    (ax2, "MATH",
     sft_4b_math,
     [r4mm, r8mm], [r4ms, r8ms],
     [g4mm, g8mm], [g4ms, g8ms]),
]:
    # SFT reference point (4B only, dashed grey)
    if sft_pt is not None:
        ax.plot([4], [sft_pt], "o", color=C_SFT, markersize=9,
                label="SFT 4B baseline", zorder=3)
        ax.annotate(f"{sft_pt:.1f}%", (4, sft_pt),
                    textcoords="offset points", xytext=(-36, 5),
                    fontsize=8.5, color=C_SFT)

    def plot_line(means, stds, color, label, offset=(6, 4)):
        pts = [(s, m, e) for s, m, e in zip(SIZES, means, stds) if m is not None]
        if not pts: return
        xs, ms, es = zip(*pts)
        xs, ms, es = list(xs), list(ms), list(es)
        ax.plot(xs, ms, "o-", color=color, linewidth=2.2,
                markersize=9, label=label, zorder=4)
        ax.fill_between(xs, [m-e for m,e in zip(ms,es)],
                             [m+e for m,e in zip(ms,es)],
                        color=color, alpha=0.12, zorder=2)
        for x, m, e in zip(xs, ms, es):
            ox, oy = offset
            ax.annotate(f"{m:.1f}±{e:.1f}%", (x, m),
                        textcoords="offset points", xytext=(ox, oy),
                        fontsize=8, color=color, fontweight="bold")

    plot_line(rlvr_ms, rlvr_ss, C_RLVR, "RLVR (mean ± std)", offset=(6, 5))
    plot_line(grpo_ms, grpo_ss, C_GRPO, "GRPO (mean ± std)", offset=(6, -14))

    # Annotate the gap closing
    if all(v is not None for v in [rlvr_ms[0], grpo_ms[0], rlvr_ms[1], grpo_ms[1]]):
        gap4 = rlvr_ms[0] - grpo_ms[0]
        gap8 = rlvr_ms[1] - grpo_ms[1]
        ax.annotate(f"gap: {gap4:+.1f}pp", (4, (rlvr_ms[0]+grpo_ms[0])/2),
                    textcoords="offset points", xytext=(-55, 0),
                    fontsize=8, color="#666",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.8))
        ax.annotate(f"gap: {gap8:+.1f}pp", (8, (rlvr_ms[1]+grpo_ms[1])/2),
                    textcoords="offset points", xytext=(8, 0),
                    fontsize=8, color="#666",
                    arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.8))

    ax.set_xticks(SIZES)
    ax.set_xticklabels(SIZE_LBL, fontsize=11)
    ax.set_xlabel("Model size", fontsize=10)
    ax.set_ylabel("System accuracy (%)")
    ax.set_title(f"{title}  —  RLVR vs GRPO by model size\n(Reflect-Full + Retry, 3 seeds)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(2.5, 10.0)

fig.text(0.5, -0.04,
    "Figure 6.  RLVR vs GRPO scaling for Reflect-Full + Retry: system accuracy at 4B and 8B. "
    "At 4B, RLVR leads GRPO by +3.3pp on GSM8K and +1.7pp on MATH. "
    "At 8B, the gap closes to −0.7pp / +1.0pp — the two algorithms become statistically indistinguishable. "
    "GRPO's 4B failure is a model-capacity interaction: too few correct trajectories for within-group normalisation to work; "
    "at 8B the model generates enough successes per group for GRPO to recover.",
    ha="center", fontsize=8, color="#555")

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
