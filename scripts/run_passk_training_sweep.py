"""
run_passk_training_sweep.py
---------------------------
Driver for the pass@k Oracle Training Sweep (Phase 15 / Day 22).

Purpose
-------
Phase 11 measured pass@k *at a single checkpoint* (the SFT baseline and final
trained checkpoints). This script produces the missing piece: pass@k
**across training steps** for two RLVR trajectories side-by-side, so the
oracle-vs-greedy gap can be plotted as a diagnostic for elicitation vs.
capability.

How the diagnostic reads
------------------------
At each training step we compute:
  - oracle:  pass@k = did ANY of k samples get it right? (capability ceiling)
  - greedy:  pass@1 using sample_idx=0 at the same temperature (deployment proxy)
  - gap:     oracle - greedy

Three possible patterns and what each implies:

  gap closes over training (oracle flat, greedy rises)
    → Training is elicitation: the capability was there, we're teaching the
      model to reach it on the first try. Correlated failures pre-training,
      independent successes post-training.

  gap stays constant (oracle rises, greedy rises together)
    → Training is capability: the ceiling genuinely lifts. More tokens won't
      substitute for training.

  gap widens (oracle rises faster than greedy, or greedy collapses)
    → Training adds capability but costs elicitation — e.g. mode collapse or
      overconfident single-sample behaviour. Red flag.

This is the experiment Day 22 of Justin's OpenAI Frontier Evals prep study
plan asks for. It is directly nameable in interviews as concrete experimental
evidence of eval-methodology instincts.

Design choices
--------------
  - k = 16, temperature = 0.7, top_p = 0.95        (matches Phase 11)
  - greedy = sample_idx=0 of the BoN group          (free; same distribution)
  - evaluation sets = gsm8k_test_200 + math_test_200 (reused)
  - seeds = 42                                      (keep first run cheap)
  - conditions:
      * Reflect-Full + RLVR   → rrr-grounded-r8-seed42-v7
      * Baseline CoT + RLVR   → baseline_rlvr_solve-r8-seed42
  - training steps = {SFT (step 0), mid, 500}       (3-point sweep)
        mid = step 350 for Reflect-Full (rrr-grounded-v7), step 300 for
        Baseline CoT (baseline_rlvr_solve). Closest available checkpoints on
        disk; both are ~60-70% through training.

Usage
-----
  # One command runs the full 3-step × 2-condition × 2-dataset sweep on 4B:
  python scripts/run_passk_training_sweep.py --model 4b --k 16 --seed 42

  # Single checkpoint (useful for smoke-testing on a cheap run):
  python scripts/run_passk_training_sweep.py --model 4b --only rrr-grounded-r8-seed42-v7-step500

  # Dry-run: print the plan, check every checkpoint file exists, don't sample:
  python scripts/run_passk_training_sweep.py --model 4b --dry-run

Output
------
For each (run_name, step) pair, one JSONL file per dataset written to
results/runs/{run_name}_step{step}_bon_n{k}_{dataset}.jsonl. Format is
identical to eval_best_of_n.py — compare_results.py and
plot_passk_training_sweep.py can both read it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Sweep plan — edit here to change which checkpoints the sweep covers
# ---------------------------------------------------------------------------


@dataclass
class SweepCell:
    """One (condition, training_step) cell of the sweep."""
    condition: str       # "reflect_full" | "baseline_cot"
    run_name: str        # exact checkpoint base name (without _bon_n… suffix)
    step_label: str      # "sft" | "step250" | "step500" etc — goes into output filename
    checkpoint_sidecar: str  # filename under results/runs/ that must exist
    base_model: str      # HF model id to pass to --base_model


def _plan_4b_three_step(seed: int = 42) -> List[SweepCell]:
    """
    Default 4B sweep: 2 conditions × 3 training steps = 6 cells.

    Note: step 0 uses the SFT checkpoints (pre-RLVR). For the reflect-full
    condition we evaluate the SFT reflect-full checkpoint with *plain solve*
    sampling — same protocol as every other cell — so the comparison is
    apples-to-apples across the x-axis.
    """
    seed_tag = f"-seed{seed}"
    BASE = "Qwen/Qwen3-4B-Instruct-2507"

    return [
        # Reflect-Full trajectory
        SweepCell(
            condition="reflect_full",
            run_name=f"qwen3-4b-reflect_full_retry-r8{seed_tag}",
            step_label="sft",
            checkpoint_sidecar=f"qwen3-4b-reflect_full_retry-r8{seed_tag}.checkpoint.json",
            base_model=BASE,
        ),
        SweepCell(
            # Closest mid-training checkpoint that exists on disk for seed42 v7.
            # Paired with baseline_rlvr_solve-step300 below (~60-70% through training).
            condition="reflect_full",
            run_name=f"rrr-grounded-r8{seed_tag}-v7-step350",
            step_label="step350",
            checkpoint_sidecar=f"rrr-grounded-r8{seed_tag}-v7-step350.checkpoint.json",
            base_model=BASE,
        ),
        SweepCell(
            condition="reflect_full",
            run_name=f"rrr-grounded-r8{seed_tag}-v7",  # final step 500
            step_label="step500",
            checkpoint_sidecar=f"rrr-grounded-r8{seed_tag}-v7.checkpoint.json",
            base_model=BASE,
        ),
        # Baseline CoT + RLVR trajectory
        SweepCell(
            condition="baseline_cot",
            run_name=f"qwen3-4b-baseline_solve-r8{seed_tag}",
            step_label="sft",
            checkpoint_sidecar=f"qwen3-4b-baseline_solve-r8{seed_tag}.checkpoint.json",
            base_model=BASE,
        ),
        SweepCell(
            condition="baseline_cot",
            run_name=f"baseline_rlvr_solve-r8{seed_tag}-step300",
            step_label="step300",
            checkpoint_sidecar=f"baseline_rlvr_solve-r8{seed_tag}-step300.checkpoint.json",
            base_model=BASE,
        ),
        SweepCell(
            condition="baseline_cot",
            run_name=f"baseline_rlvr_solve-r8{seed_tag}",  # final step 500
            step_label="step500",
            checkpoint_sidecar=f"baseline_rlvr_solve-r8{seed_tag}.checkpoint.json",
            base_model=BASE,
        ),
    ]


def _plan_8b_three_step(seed: int = 42) -> List[SweepCell]:
    """8B sweep — same shape. Not enabled by default; pass --model 8b to use."""
    seed_tag = f"-seed{seed}"
    BASE = "Qwen/Qwen3-8B-Instruct-2507"

    return [
        SweepCell("reflect_full", f"qwen3-8b-reflect_full_retry-r8{seed_tag}",
                  "sft", f"qwen3-8b-reflect_full_retry-r8{seed_tag}.checkpoint.json", BASE),
        SweepCell("reflect_full", f"qwen3-8b-reflect_full_retry-r8{seed_tag}-step250",
                  "step250", f"qwen3-8b-reflect_full_retry-r8{seed_tag}-step250.checkpoint.json", BASE),
        SweepCell("reflect_full", f"qwen3-8b-reflect_full_retry-r8{seed_tag}-step500",
                  "step500", f"qwen3-8b-reflect_full_retry-r8{seed_tag}-step500.checkpoint.json", BASE),
        SweepCell("baseline_cot", f"qwen3-4b-baseline_solve-r8{seed_tag}",  # no 8B SFT solve checkpoint; reuse 4B
                  "sft", f"qwen3-4b-baseline_solve-r8{seed_tag}.checkpoint.json", "Qwen/Qwen3-4B-Instruct-2507"),
        SweepCell("baseline_cot", f"baseline_rlvr_retry-8b-r8{seed_tag}-step250",
                  "step250", f"baseline_rlvr_retry-8b-r8{seed_tag}-step250.checkpoint.json", BASE),
        SweepCell("baseline_cot", f"baseline_rlvr_retry-8b-r8{seed_tag}-step500",
                  "step500", f"baseline_rlvr_retry-8b-r8{seed_tag}-step500.checkpoint.json", BASE),
    ]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _check_checkpoints(cells: List[SweepCell], runs_dir: Path) -> List[SweepCell]:
    """Print which sidecar JSONs exist; return only the cells that do."""
    ok: List[SweepCell] = []
    print(f"\n[sweep] checkpoint audit (runs_dir = {runs_dir})")
    for c in cells:
        p = runs_dir / c.checkpoint_sidecar
        status = "OK " if p.exists() else "MISS"
        marker = "✓" if p.exists() else "✗"
        print(f"  [{marker}] {status}  {c.condition:<13}  {c.step_label:<8}  {c.checkpoint_sidecar}")
        if p.exists():
            ok.append(c)
    missing = len(cells) - len(ok)
    print(f"[sweep] {len(ok)}/{len(cells)} cells have sidecar JSONs on disk ({missing} missing).")
    return ok


def _run_cell(
    cell: SweepCell,
    *,
    k: int,
    seed: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    datasets: str,
    limit: Optional[int],
    output_dir: Path,
    resume: bool,
    dry_run: bool,
) -> int:
    """
    Run eval_best_of_n.py for a single cell.

    We shell out to eval_best_of_n.py (rather than importing) because the
    existing script handles the Tinker client lifecycle (restore → get_sampling
    client → decode) in a way that is tied to its __main__ flow. Reimplementing
    it would mean maintaining a second copy. Subprocess keeps things dry.
    """
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "eval_best_of_n.py"),
        "--run_name", cell.run_name,
        "--base_model", cell.base_model,
        "--n", str(k),
        "--seed", str(seed),
        "--temperature", str(temperature),
        "--top_p", str(top_p),
        "--max_new_tokens", str(max_new_tokens),
        "--dataset", datasets,
        "--output_dir", str(output_dir),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if resume:
        cmd += ["--resume"]

    print(f"\n[sweep] ── {cell.condition} / {cell.step_label} ──")
    print(f"[sweep] cmd: {' '.join(cmd)}")

    if dry_run:
        print("[sweep] (dry-run) skipped.")
        return 0

    # Inherit stdout/stderr live so the user sees progress
    return subprocess.run(cmd, cwd=_REPO_ROOT).returncode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="pass@k Oracle Training Sweep — Phase 15 driver (Day 22).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--model", choices=["4b", "8b"], default="4b",
                    help="Which model size to sweep over.")
    ap.add_argument("--k", type=int, default=16,
                    help="Number of samples per problem (pass@k oracle).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--dataset", default="both", choices=["gsm8k", "math", "both"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap problems per dataset (useful for a quick run).")
    ap.add_argument("--only", default=None,
                    help="If set, only run the cell whose run_name matches exactly.")
    ap.add_argument("--output_dir", default="results/runs",
                    help="Where eval_best_of_n.py writes outputs (relative to repo root).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip problems already written for that cell.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan, check checkpoint sidecars exist, don't sample.")

    return ap.parse_args()


def main() -> int:
    args = parse_args()

    plan = _plan_4b_three_step(seed=args.seed) if args.model == "4b" else _plan_8b_three_step(seed=args.seed)
    if args.only:
        plan = [c for c in plan if c.run_name == args.only]
        if not plan:
            print(f"[sweep] --only {args.only!r} matched no cells. Check the plan.")
            return 2

    runs_dir = _REPO_ROOT / args.output_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    plan = _check_checkpoints(plan, runs_dir)
    if not plan:
        print("[sweep] No checkpoints available — nothing to do.")
        return 0

    print(f"\n[sweep] running {len(plan)} cells with k={args.k}, T={args.temperature}, dataset={args.dataset}")
    n_ok = 0
    n_err = 0
    for cell in plan:
        rc = _run_cell(
            cell,
            k=args.k,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            datasets=args.dataset,
            limit=args.limit,
            output_dir=runs_dir,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        if rc == 0:
            n_ok += 1
        else:
            n_err += 1
            print(f"[sweep] cell {cell.run_name} exited rc={rc}")

    print(f"\n[sweep] done — {n_ok} ok, {n_err} failed.")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
