"""
scripts/train_rrr.py
====================
CLI entry point for RRR (Outcome-Gated Reflection Reward) training.

Loads a saved SFT LoRA checkpoint as warm-start, then runs online
rejection-sampling fine-tuning where gradient is applied only when
reflect->retry produces a correct answer.

Recommended warm-start:
  The SFT checkpoint that was trained with --mode reflect_plan_retry
  (or reflect_full_retry) already knows the reflection format.
  Use that as --sft_checkpoint so RRR can immediately generate
  structured reflections worth gating on.

Example usage
-------------
# RRR fine-tuning warm-started from the best SFT reflection checkpoint
python scripts/train_rrr.py \\
    --sft_checkpoint qwen3-4b-reflect_plan_retry-r8-seed42 \\
    --reflection_mode plan \\
    --max_steps 200 \\
    --seed 42

# Quick smoke-test (10 steps, 2 problems each)
python scripts/train_rrr.py \\
    --sft_checkpoint qwen3-4b-reflect_plan_retry-r8-seed42 \\
    --reflection_mode plan \\
    --max_steps 10 \\
    --problems_per_step 2 \\
    --data_limit 50

Evaluate afterwards
-------------------
python scripts/eval_sft.py \\
    --run_name rrr-plan-r8-seed42 \\
    --mode reflect_plan_retry \\
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
        description="RRR training: outcome-gated reflection reward via online RFT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model / checkpoint ───────────────────────────────────────────────────
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507",
                    help="Tinker base model identifier.")
    ap.add_argument("--sft_checkpoint", default=None,
                    help="run_name of a saved SFT checkpoint to warm-start from. "
                         "Defaults to qwen3-4b-reflect_{reflection_mode}_retry-r{rank}-seed{seed} "
                         "if not provided.")
    ap.add_argument("--run_name", default=None,
                    help="Name under which the RRR checkpoint is saved. "
                         "Defaults to rrr-{reflection_mode}-r{rank}-seed{seed}.")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # ── Data ─────────────────────────────────────────────────────────────────
    ap.add_argument("--train_jsonl", nargs="+", default=None,
                    metavar="PATH",
                    help="One or more raw-problem JSONL files (question + answer). "
                         "Defaults to data/processed/gsm8k_train.jsonl and math_train.jsonl.")
    ap.add_argument("--data_limit", type=int, default=None,
                    help="Cap number of problems loaded (useful for smoke-tests).")

    # ── RL loop ───────────────────────────────────────────────────────────────
    ap.add_argument("--max_steps", type=int, default=200,
                    help="Number of gradient steps to run.")
    ap.add_argument("--problems_per_step", type=int, default=4,
                    help="Problems sampled per gradient step.")
    ap.add_argument("--sampler_refresh_every", type=int, default=10,
                    help="Push updated weights to the sampling client every N steps.")

    # ── Reflection mode ───────────────────────────────────────────────────────
    ap.add_argument("--reflection_mode", choices=["plan", "full", "tail"],
                    default="plan",
                    help="Which reflection prompt style to use. "
                         "Should match the SFT checkpoint's training mode.")

    # ── Decoding ─────────────────────────────────────────────────────────────
    ap.add_argument("--solve_max_tokens",   type=int,   default=512)
    ap.add_argument("--reflect_max_tokens", type=int,   default=192)
    ap.add_argument("--retry_max_tokens",   type=int,   default=512)
    ap.add_argument("--temperature",        type=float, default=0.7)
    ap.add_argument("--top_p",              type=float, default=0.95)
    ap.add_argument("--reflect_temperature", type=float, default=0.4,
                    help="Lower temperature for reflection (more focused).")

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
    ap.add_argument("--resume", action="store_true", default=False,
                    help="Resume training from the latest (or --resume_step) checkpoint "
                         "for this run_name instead of starting fresh.")
    ap.add_argument("--resume_step", type=int, default=-1,
                    help="Step number to resume from (-1 = auto-detect highest snapshot).")

    # ── Output ────────────────────────────────────────────────────────────────
    ap.add_argument("--output_dir", default="results/runs",
                    help="Directory for training logs.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ── Auto-fill sft_checkpoint if not specified ─────────────────────────────
    if args.sft_checkpoint is None:
        args.sft_checkpoint = (
            f"qwen3-4b-reflect_{args.reflection_mode}_retry"
            f"-r{args.rank}-seed{args.seed}"
        )
        print(f"[train_rrr] --sft_checkpoint not set, defaulting to "
              f"'{args.sft_checkpoint}'")

    from src.rrr.train_rrr import train_rrr

    train_rrr(
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
        reflection_mode       = args.reflection_mode,
        solve_max_tokens      = args.solve_max_tokens,
        reflect_max_tokens    = args.reflect_max_tokens,
        retry_max_tokens      = args.retry_max_tokens,
        temperature           = args.temperature,
        top_p                 = args.top_p,
        reflect_temperature   = args.reflect_temperature,
        lr                    = args.lr,
        beta1                 = args.beta1,
        beta2                 = args.beta2,
        weight_decay          = args.weight_decay,
        grad_clip             = args.grad_clip,
        max_seq_len           = args.max_seq_len,
        checkpoint_every      = args.checkpoint_every,
        resume                = args.resume,
        resume_step           = args.resume_step,
        output_dir            = args.output_dir,
    )


if __name__ == "__main__":
    main()
