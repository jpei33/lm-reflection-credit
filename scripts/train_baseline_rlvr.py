"""
scripts/train_baseline_rlvr.py
================================
CLI entry point for Baseline RLVR-CoT and Retry-only RLVR training.

Covers the two simplest RLVR conditions that complete the comparison matrix:

  --mode solve   Baseline RLVR-CoT
  ─────────────────────────────────
  Sample solve rollouts from the current policy.  Train on correct ones
  (uniform weight=1.0).  Skip incorrect ones.  The simplest possible
  outcome-reward signal.  Warm-starts from baseline_solve SFT.

  --mode retry   Retry-only RLVR
  ────────────────────────────────
  Sample solve → if wrong, sample a plain retry (no reflection).
  Train on the retry tokens when retry is correct; the initial solve is
  context (weight=0).  Warm-starts from retry_only SFT.

Together with train_rrr.py and train_step_credit.py, this gives the
full comparison:

  RLVR Condition      | Script               | --mode / flag
  --------------------|----------------------|-------------------------
  Baseline RLVR-CoT   | train_baseline_rlvr  | --mode solve
  Retry-only RLVR     | train_baseline_rlvr  | --mode retry
  RRR-full            | train_rrr            | --reflection_mode full
  RRR-plan            | train_rrr            | --reflection_mode plan
  Step-local credit   | train_step_credit    | (no flag needed)

Example usage
-------------
# Baseline RLVR-CoT  (warm-start: baseline_solve SFT)
python scripts/train_baseline_rlvr.py \\
    --mode solve \\
    --sft_checkpoint qwen3-4b-baseline_solve-r8-seed42 \\
    --max_steps 200 --seed 42

# Retry-only RLVR  (warm-start: retry_only SFT)
python scripts/train_baseline_rlvr.py \\
    --mode retry \\
    --sft_checkpoint qwen3-4b-retry_only-r8-seed42 \\
    --max_steps 200 --seed 42

# Quick smoke-test for either mode
python scripts/train_baseline_rlvr.py \\
    --mode solve \\
    --max_steps 10 --problems_per_step 2 --data_limit 50

Evaluate afterwards
-------------------
# Baseline RLVR-CoT
python scripts/eval_sft.py \\
    --run_name baseline_rlvr_solve-r8-seed42 \\
    --mode baseline_solve \\
    --dataset both

# Retry-only RLVR
python scripts/eval_sft.py \\
    --run_name baseline_rlvr_retry-r8-seed42 \\
    --mode retry_only \\
    --dataset both
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
load_dotenv(_REPO_ROOT / ".env")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Baseline RLVR-CoT and Retry-only RLVR training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--mode", required=True, choices=["solve", "retry"],
        help=(
            "solve = Baseline RLVR-CoT: train on correct first-try solutions. "
            "retry = Retry-only RLVR: train on correct retries (no reflection)."
        ),
    )

    # ── Model / checkpoint ───────────────────────────────────────────────────
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument(
        "--sft_checkpoint", default=None,
        help=(
            "run_name of the SFT checkpoint to warm-start from. "
            "Defaults to qwen3-4b-baseline_solve-r{rank}-seed{seed} for mode=solve, "
            "qwen3-4b-retry_only-r{rank}-seed{seed} for mode=retry."
        ),
    )
    ap.add_argument(
        "--run_name", default=None,
        help="Name for the saved RLVR checkpoint. "
             "Defaults to baseline_rlvr_{mode}-r{rank}-seed{seed}.",
    )
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # ── Data ─────────────────────────────────────────────────────────────────
    ap.add_argument("--train_jsonl", nargs="+", default=None, metavar="PATH",
                    help="Raw-problem JSONL files. "
                         "Defaults to gsm8k_train.jsonl + math_train.jsonl.")
    ap.add_argument("--data_limit", type=int, default=None,
                    help="Cap number of problems (smoke-test).")

    # ── RL loop ───────────────────────────────────────────────────────────────
    ap.add_argument("--max_steps",             type=int, default=200)
    ap.add_argument("--problems_per_step",     type=int, default=4)
    ap.add_argument("--sampler_refresh_every", type=int, default=10)

    # ── Decoding ─────────────────────────────────────────────────────────────
    ap.add_argument("--solve_max_tokens", type=int,   default=512)
    ap.add_argument("--retry_max_tokens", type=int,   default=512)
    ap.add_argument("--temperature",      type=float, default=0.7)
    ap.add_argument("--top_p",            type=float, default=0.95)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    ap.add_argument("--lr",           type=float, default=1e-5)
    ap.add_argument("--beta1",        type=float, default=0.9)
    ap.add_argument("--beta2",        type=float, default=0.95)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip",    type=float, default=1.0)
    ap.add_argument("--max_seq_len",  type=int,   default=1024)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="Save a resumable checkpoint every N steps (0 = off). "
                         "Snapshots written to results/runs/{run_name}-step{N}.checkpoint.json.")

    # ── Output ────────────────────────────────────────────────────────────────
    ap.add_argument("--output_dir", default="results/runs")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    from src.rrr.train_baseline_rlvr import train_baseline_rlvr

    train_baseline_rlvr(
        mode                  = args.mode,
        base_model            = args.base_model,
        sft_checkpoint        = args.sft_checkpoint,
        run_name              = args.run_name,
        rank                  = args.rank,
        seed                  = args.seed,
        train_jsonl_paths     = args.train_jsonl,
        data_limit            = args.data_limit,
        max_steps             = args.max_steps,
        problems_per_step     = args.problems_per_step,
        sampler_refresh_every = args.sampler_refresh_every,
        solve_max_tokens      = args.solve_max_tokens,
        retry_max_tokens      = args.retry_max_tokens,
        temperature           = args.temperature,
        top_p                 = args.top_p,
        lr                    = args.lr,
        beta1                 = args.beta1,
        beta2                 = args.beta2,
        weight_decay          = args.weight_decay,
        grad_clip             = args.grad_clip,
        max_seq_len           = args.max_seq_len,
        checkpoint_every      = args.checkpoint_every,
        output_dir            = args.output_dir,
    )


if __name__ == "__main__":
    main()
