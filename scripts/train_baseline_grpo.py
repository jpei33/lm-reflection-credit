"""
scripts/train_baseline_grpo.py
-------------------------------
CLI entry point for Phase 3 baseline GRPO training.

Usage:
  python scripts\\train_baseline_grpo.py --mode solve --max_steps 200 --seed 42
  python scripts\\train_baseline_grpo.py --mode retry --max_steps 200 --seed 42

Phase 3 eval commands (after training):
  python scripts\\eval_sft.py --run_name baseline_grpo_solve-r8-seed42 --mode baseline_solve --dataset both
  python scripts\\eval_sft.py --run_name baseline_grpo_retry-r8-seed42 --mode retry_only    --dataset both
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.rrr.train_baseline_grpo import train_baseline_grpo


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 3: Baseline GRPO training (solve or retry, no reflection).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode ────────────────────────────────────────────────────────────────
    ap.add_argument("--mode", required=True, choices=["solve", "retry"],
                    help="'solve' = GRPO on first-try solves. 'retry' = GRPO on blind retries.")

    # ── Model / checkpoint ──────────────────────────────────────────────────
    ap.add_argument("--base_model",     default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--sft_checkpoint", default=None)
    ap.add_argument("--run_name",       default=None)
    ap.add_argument("--rank",           type=int, default=8)
    ap.add_argument("--seed",           type=int, default=42)

    # ── Training loop ───────────────────────────────────────────────────────
    ap.add_argument("--max_steps",          type=int,   default=200)
    ap.add_argument("--problems_per_step",  type=int,   default=4)
    ap.add_argument("--grpo_k",             type=int,   default=8,
                    help="Number of samples per problem/failed-solve for GRPO group.")
    ap.add_argument("--sampler_refresh_every", type=int, default=10)

    # ── Decoding ─────────────────────────────────────────────────────────────
    ap.add_argument("--solve_max_tokens",  type=int,   default=512)
    ap.add_argument("--retry_max_tokens",  type=int,   default=512)
    ap.add_argument("--temperature",       type=float, default=0.7)
    ap.add_argument("--top_p",             type=float, default=0.95)

    # ── GRPO ────────────────────────────────────────────────────────────────
    ap.add_argument("--clip_negative_advantages", action="store_true")
    ap.add_argument("--advantage_eps",     type=float, default=1e-8)

    # ── Optimiser ────────────────────────────────────────────────────────────
    ap.add_argument("--lr",          type=float, default=5e-7)
    ap.add_argument("--beta1",       type=float, default=0.9)
    ap.add_argument("--beta2",       type=float, default=0.95)
    ap.add_argument("--weight_decay",type=float, default=0.0)
    ap.add_argument("--grad_clip",   type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int,   default=1024)

    # ── Checkpointing / resume / Output ─────────────────────────────────────
    ap.add_argument("--checkpoint_every", type=int, default=0)
    ap.add_argument("--resume",      action="store_true", default=False,
                    help="Resume from latest (or --resume_step) checkpoint for this run.")
    ap.add_argument("--resume_step", type=int, default=-1,
                    help="Step to resume from (-1 = auto-detect highest snapshot).")
    ap.add_argument("--output_dir",       default="results/runs")
    ap.add_argument("--data_limit",       type=int, default=None)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_baseline_grpo(
        mode=args.mode,
        base_model=args.base_model,
        sft_checkpoint=args.sft_checkpoint,
        run_name=args.run_name,
        rank=args.rank,
        seed=args.seed,
        data_limit=args.data_limit,
        max_steps=args.max_steps,
        problems_per_step=args.problems_per_step,
        grpo_k=args.grpo_k,
        sampler_refresh_every=args.sampler_refresh_every,
        solve_max_tokens=args.solve_max_tokens,
        retry_max_tokens=args.retry_max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        clip_negative_advantages=args.clip_negative_advantages,
        advantage_eps=args.advantage_eps,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_seq_len=args.max_seq_len,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        resume_step=args.resume_step,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
