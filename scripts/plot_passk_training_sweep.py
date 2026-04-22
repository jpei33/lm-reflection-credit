"""
plot_passk_training_sweep.py
-----------------------------
Read the pass@k Oracle Training Sweep outputs (from
run_passk_training_sweep.py) and produce the Phase 15 figure:

    figures/fig2b_passk_training_sweep.png

Four panels — one per (condition, dataset):
  - Reflect-Full / GSM8K   (top left)
  - Reflect-Full / MATH    (top right)
  - Baseline CoT  / GSM8K   (bottom left)
  - Baseline CoT  / MATH    (bottom right)

Each panel shows two curves across training steps (SFT → mid → step500):
  - oracle (pass@16, solid blue) — capability ceiling
  - greedy (pass@1 using sample 0 of the BoN group, dashed orange) — deployment
Shaded bands are bootstrap 95% CIs (resampled over the 200 problems).

The headline number printed in each panel is the oracle-greedy *gap* —
how much elicitation headroom remains that training could close.

Usage
-----
  # Normal: read JSONL outputs produced by run_passk_training_sweep.py
  python scripts/plot_passk_training_sweep.py

  # Smoke-test on synthetic data (no real runs needed):
  python scripts/plot_passk_training_sweep.py --synthetic

Both modes write figures/fig2b_passk_training_sweep.png plus a machine-readable
JSON summary at results/analysis/passk_training_sweep_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.eval.passk_estimator import bootstrap_pass_at_k  # noqa: E402


# ---------------------------------------------------------------------------
# Plan — must match scripts/run_passk_training_sweep.py _plan_4b_three_step
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    condition: str        # "reflect_full" | "baseline_cot"
    run_name: str         # base name that eval_best_of_n used
    step_label: str       # "sft" | "stepNNN"
    step_x: int           # x-coordinate on the training-step axis


# Training-step x-axis values (SFT = 0). Must align across conditions so the
# two trajectories sit at comparable x positions; mid-training differs slightly
# (300 vs 350) because those are the closest checkpoints on disk.
_PLAN_4B: List[SweepPoint] = [
    SweepPoint("reflect_full", "qwen3-4b-reflect_full_retry-r8-seed42", "sft", 0),
    SweepPoint("reflect_full", "rrr-grounded-r8-seed42-v7-step350",     "step350", 350),
    SweepPoint("reflect_full", "rrr-grounded-r8-seed42-v7",              "step500", 500),
    SweepPoint("baseline_cot", "qwen3-4b-baseline_solve-r8-seed42",      "sft", 0),
    SweepPoint("baseline_cot", "baseline_rlvr_solve-r8-seed42-step300",  "step300", 300),
    SweepPoint("baseline_cot", "baseline_rlvr_solve-r8-seed42",          "step500", 500),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _bon_path(runs_dir: Path, run_name: str, k: int, dataset: str) -> Path:
    """Path eval_best_of_n.py writes to."""
    return runs_dir / f"{run_name}_bon_n{k}_{dataset}.jsonl"


def _per_task_from_rows(rows: List[dict], k: int) -> List[dict]:
    """
    Extract the per-task summary needed by the estimator:
      {n_correct, n, first_correct} per problem.
    """
    out = []
    for r in rows:
        n_group = int(r.get("n", k))
        c = int(r.get("n_correct_in_group", 0))
        completions = r.get("completions") or []
        first_correct = False
        if completions:
            # Prefer the completion with j == 0 if present; else the first in list.
            cand = next((c_ for c_ in completions if c_.get("j") == 0), completions[0])
            first_correct = bool(cand.get("correct", False))
        out.append({"n": n_group, "n_correct_in_group": c, "first_correct": first_correct})
    return out


# ---------------------------------------------------------------------------
# Synthetic generator (for smoke-testing the plot without real runs)
# ---------------------------------------------------------------------------


def _synthetic_per_task(*, n_problems: int, k: int, p_greedy: float, p_oracle: float,
                         seed: int) -> List[dict]:
    """
    Generate plausible per-task rows matching the eval_best_of_n schema.

    Each task gets an independent Bernoulli(p_greedy) for the first sample,
    then fills in the remaining k-1 samples so that any_correct happens with
    probability ≈ p_oracle. Accomplishes this by independently flipping
    Bernoulli(p_residual) for each remaining slot, where

        p_residual = 1 - ((1 - p_oracle) / (1 - p_greedy)) ** (1 / (k - 1))

    i.e. conditional on first failing, the remaining k-1 must recover the
    oracle with probability p_oracle. For p_oracle >= p_greedy.
    """
    if p_greedy > p_oracle:
        raise ValueError(f"oracle must be >= greedy; got greedy={p_greedy}, oracle={p_oracle}")
    rng = np.random.default_rng(seed)

    if p_greedy >= 1.0:
        p_residual = 1.0
    elif p_oracle >= 1.0:
        p_residual = 1.0
    else:
        p_residual = 1.0 - ((1.0 - p_oracle) / (1.0 - p_greedy)) ** (1.0 / max(k - 1, 1))

    per_task = []
    for _ in range(n_problems):
        first = bool(rng.random() < p_greedy)
        rest_correct = int(rng.binomial(k - 1, p_residual))
        c = (1 if first else 0) + rest_correct
        c = min(c, k)
        per_task.append({"n": k, "n_correct_in_group": c, "first_correct": first})
    return per_task


# Synthetic scenario used when --synthetic is passed. Numbers chosen so the plot
# tells the intended story: Reflect-Full closes more of the gap than Baseline CoT.
_SYNTH_SCENARIO: Dict[Tuple[str, str, int], Tuple[float, float]] = {
    # (condition, dataset, step_x) -> (p_greedy, p_oracle)
    ("reflect_full", "gsm8k", 0):   (0.23, 0.50),
    ("reflect_full", "gsm8k", 350): (0.30, 0.52),
    ("reflect_full", "gsm8k", 500): (0.33, 0.53),
    ("reflect_full", "math", 0):    (0.15, 0.34),
    ("reflect_full", "math", 350):  (0.17, 0.36),
    ("reflect_full", "math", 500):  (0.18, 0.37),
    ("baseline_cot", "gsm8k", 0):   (0.23, 0.50),
    ("baseline_cot", "gsm8k", 300): (0.29, 0.51),
    ("baseline_cot", "gsm8k", 500): (0.31, 0.51),
    ("baseline_cot", "math", 0):    (0.15, 0.34),
    ("baseline_cot", "math", 300):  (0.16, 0.35),
    ("baseline_cot", "math", 500):  (0.17, 0.36),
}


# ---------------------------------------------------------------------------
# Per-cell summary
# ---------------------------------------------------------------------------


@dataclass
class CellSummary:
    condition: str
    dataset: str
    step_x: int
    oracle_point: float
    oracle_lo: float
    oracle_hi: float
    greedy_point: float
    greedy_lo: float
    greedy_hi: float
    n_tasks: int

    @property
    def gap(self) -> float:
        return self.oracle_point - self.greedy_point

    def as_dict(self) -> dict:
        return {
            "condition": self.condition,
            "dataset": self.dataset,
            "step_x": self.step_x,
            "oracle_point": self.oracle_point,
            "oracle_ci": [self.oracle_lo, self.oracle_hi],
            "greedy_point": self.greedy_point,
            "greedy_ci": [self.greedy_lo, self.greedy_hi],
            "gap": self.gap,
            "n_tasks": self.n_tasks,
        }


def _summarise_cell(
    per_task: List[dict],
    *,
    condition: str,
    dataset: str,
    step_x: int,
    k: int,
    n_resamples: int,
    seed: int,
) -> CellSummary:
    oracle_in = [(int(r["n"]), int(r["n_correct_in_group"])) for r in per_task]
    greedy_in = [(1, int(bool(r["first_correct"]))) for r in per_task]
    oracle = bootstrap_pass_at_k(oracle_in, k=k, n_resamples=n_resamples, seed=seed)
    greedy = bootstrap_pass_at_k(greedy_in, k=1, n_resamples=n_resamples, seed=seed + 1)
    return CellSummary(
        condition=condition,
        dataset=dataset,
        step_x=step_x,
        oracle_point=oracle.point,
        oracle_lo=oracle.lo,
        oracle_hi=oracle.hi,
        greedy_point=greedy.point,
        greedy_lo=greedy.lo,
        greedy_hi=greedy.hi,
        n_tasks=len(per_task),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


_CONDITION_LABELS = {"reflect_full": "Reflect-Full (RLVR)", "baseline_cot": "Baseline CoT (RLVR)"}
_DATASET_LABELS   = {"gsm8k": "GSM8K", "math": "MATH"}
_C_ORACLE = "#2c7fb8"
_C_GREEDY = "#d95f02"


def _plot(summaries: List[CellSummary], output_path: Path, *, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharey=False)
    conditions = ["reflect_full", "baseline_cot"]
    datasets   = ["gsm8k", "math"]

    for ri, cond in enumerate(conditions):
        for ci, ds in enumerate(datasets):
            ax = axes[ri][ci]
            cells = sorted([s for s in summaries if s.condition == cond and s.dataset == ds],
                           key=lambda s: s.step_x)
            if not cells:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes,
                        color="grey", fontsize=12)
                ax.set_title(f"{_CONDITION_LABELS[cond]} — {_DATASET_LABELS[ds]}")
                continue

            xs = [s.step_x for s in cells]
            oracle_pts = [s.oracle_point * 100 for s in cells]
            oracle_lo  = [s.oracle_lo    * 100 for s in cells]
            oracle_hi  = [s.oracle_hi    * 100 for s in cells]
            greedy_pts = [s.greedy_point * 100 for s in cells]
            greedy_lo  = [s.greedy_lo    * 100 for s in cells]
            greedy_hi  = [s.greedy_hi    * 100 for s in cells]

            ax.fill_between(xs, oracle_lo, oracle_hi, color=_C_ORACLE, alpha=0.15)
            ax.plot(xs, oracle_pts, color=_C_ORACLE, marker="o", lw=2.2, label="oracle (pass@k)")
            ax.fill_between(xs, greedy_lo, greedy_hi, color=_C_GREEDY, alpha=0.15)
            ax.plot(xs, greedy_pts, color=_C_GREEDY, marker="s", lw=2.2, ls="--", label="greedy (pass@1, T=0.7)")

            # Annotate gap at the final step
            last = cells[-1]
            ax.annotate(
                f"gap at final step:\n{100 * last.gap:.1f} pp",
                xy=(last.step_x, 100 * (last.oracle_point + last.greedy_point) / 2),
                xytext=(0.98, 0.55), textcoords="axes fraction",
                ha="right", va="center",
                fontsize=9, color="#333",
                bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#bbb", lw=0.8),
            )

            ax.set_title(f"{_CONDITION_LABELS[cond]} — {_DATASET_LABELS[ds]}")
            ax.set_xlabel("Training step")
            ax.set_ylabel("Accuracy (%)")
            ax.set_xticks(xs)
            ax.set_xticklabels([f"SFT" if x == 0 else str(x) for x in xs])
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, max(55, max(oracle_pts) + 8))
            if ri == 0 and ci == 1:
                ax.legend(loc="lower right", frameon=True, fontsize=9)

    title = "pass@k oracle vs. greedy across RLVR training steps"
    if title_suffix:
        title += f"  —  {title_suffix}"
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def _write_summary_json(summaries: List[CellSummary], path: Path, *, k: int, synthetic: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/plot_passk_training_sweep.py",
        "k": k,
        "synthetic": synthetic,
        "conditions": sorted({s.condition for s in summaries}),
        "datasets":   sorted({s.dataset for s in summaries}),
        "cells":      [s.as_dict() for s in summaries],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot pass@k oracle vs. greedy across RLVR training steps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--runs_dir", default="results/runs",
                    help="Where eval_best_of_n.py wrote its outputs.")
    ap.add_argument("--output", default="figures/fig2b_passk_training_sweep.png",
                    help="Output image path (relative to repo root).")
    ap.add_argument("--summary", default="results/analysis/passk_training_sweep_summary.json",
                    help="Machine-readable summary JSON output path.")
    ap.add_argument("--k", type=int, default=16,
                    help="k used when the BoN files were produced.")
    ap.add_argument("--datasets", nargs="+", default=["gsm8k", "math"],
                    choices=["gsm8k", "math"])
    ap.add_argument("--n_resamples", type=int, default=10_000,
                    help="Bootstrap draws for CIs.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Bootstrap RNG seed.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Skip file I/O; fabricate plausible per-task rows to smoke-test the plot.")
    ap.add_argument("--n_problems_synthetic", type=int, default=200,
                    help="Tasks per cell when --synthetic.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    runs_dir = (_REPO_ROOT / args.runs_dir).resolve()
    output   = (_REPO_ROOT / args.output).resolve()
    summary  = (_REPO_ROOT / args.summary).resolve()

    summaries: List[CellSummary] = []
    missing_cells: List[str] = []

    if args.synthetic:
        print("[plot] using SYNTHETIC data — no real files read.")
        for sp in _PLAN_4B:
            for ds in args.datasets:
                p_greedy, p_oracle = _SYNTH_SCENARIO[(sp.condition, ds, sp.step_x)]
                per_task = _synthetic_per_task(
                    n_problems=args.n_problems_synthetic,
                    k=args.k,
                    p_greedy=p_greedy,
                    p_oracle=p_oracle,
                    seed=hash((sp.condition, ds, sp.step_x)) & 0xFFFF,
                )
                summaries.append(_summarise_cell(
                    per_task, condition=sp.condition, dataset=ds, step_x=sp.step_x,
                    k=args.k, n_resamples=args.n_resamples, seed=args.seed,
                ))
    else:
        print(f"[plot] reading from {runs_dir}")
        for sp in _PLAN_4B:
            for ds in args.datasets:
                path = _bon_path(runs_dir, sp.run_name, k=args.k, dataset=ds)
                if not path.exists():
                    missing_cells.append(str(path.relative_to(_REPO_ROOT)))
                    continue
                rows = _load_jsonl(path)
                if not rows:
                    missing_cells.append(str(path.relative_to(_REPO_ROOT)) + " (empty)")
                    continue
                per_task = _per_task_from_rows(rows, k=args.k)
                summaries.append(_summarise_cell(
                    per_task, condition=sp.condition, dataset=ds, step_x=sp.step_x,
                    k=args.k, n_resamples=args.n_resamples, seed=args.seed,
                ))

    if missing_cells:
        print(f"[plot] {len(missing_cells)} cell(s) missing:")
        for m in missing_cells:
            print(f"         ⨯ {m}")
        print("[plot] plot will render those panels as '(no data)'.")

    if not summaries:
        print("[plot] no data available to plot — aborting.")
        return 1

    title_suffix = "4B, k=16, T=0.7, n≈200 per panel"
    if args.synthetic:
        title_suffix += "  [synthetic — smoke test]"
    _plot(summaries, output, title_suffix=title_suffix)
    _write_summary_json(summaries, summary, k=args.k, synthetic=args.synthetic)
    print(f"[plot] wrote {output.relative_to(_REPO_ROOT)}")
    print(f"[plot] wrote {summary.relative_to(_REPO_ROOT)}")

    print("\n[plot] per-cell summary:")
    for s in sorted(summaries, key=lambda s: (s.condition, s.dataset, s.step_x)):
        print(f"  {s.condition:<13} {s.dataset:<6} step={s.step_x:>4} "
              f"oracle={s.oracle_point:.1%} [{s.oracle_lo:.1%}–{s.oracle_hi:.1%}]  "
              f"greedy={s.greedy_point:.1%} [{s.greedy_lo:.1%}–{s.greedy_hi:.1%}]  "
              f"gap={s.gap*100:+.1f} pp")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
