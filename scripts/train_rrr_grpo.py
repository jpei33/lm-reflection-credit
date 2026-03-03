"""
scripts/train_rrr_grpo.py
--------------------------
CLI entry point for Phase 3 RRR-GRPO training.

Usage:
  python scripts\\train_rrr_grpo.py --reflection_mode full --max_steps 200 --seed 42
  python scripts\\train_rrr_grpo.py --reflection_mode plan --max_steps 200 --seed 42

  # With more GRPO samples per group (richer signal, ~2x slower):
  python scripts\\train_rrr_grpo.py --reflection_mode full --grpo_k 16

  # Fallback to RFT behaviour (clip negatives, same as train_rrr.py but with groups):
  python scripts\\train_rrr_grpo.py --reflection_mode full --clip_negative_advantages

Phase 3 eval commands (after training):
  python scripts\\eval_sft.py --run_name rrr_grpo-full-r8-seed42 --mode reflect_full_retry --dataset both
  python scripts\\eval_sft.py --run_name rrr_grpo-plan-r8-seed42 --mode reflect_plan_retry --dataset both
"""

import argparse
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from src.rrr.train_rrr_grpo import train_rrr_grpo


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Phase 3: RRR training with GRPO (group relative policy optimization).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model / checkpoint ──────────────────────────────────────────────────
    ap.add_argument("--base_model",     default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--sft_checkpoint", default=None,
                    help="Run name of the SFT warm-start checkpoint. "
                         "Defaults to qwen3-4b-reflect_{mode}_retry-r{rank}-seed{seed}.")
    ap.add_argument("--run_name",       default=None,
                    help="Name for this GRPO run. Defaults to rrr_grpo-{mode}-r{rank}-seed{seed}.")
    ap.add_argument("--rank",           type=int, default=8)
    ap.add_argument("--seed",           type=int, default=42)

    # ── Reflection ──────────────────────────────────────────────────────────
    ap.add_argument("--reflection_mode", default="full", choices=["full", "plan"],
                    help="Reflection prompt style (matches SFT training condition).")

    # ── Training loop ───────────────────────────────────────────────────────
    ap.add_argument("--max_steps",          type=int,   default=200)
    ap.add_argument("--problems_per_step",  type=int,   default=4)
    ap.add_argument("--grpo_k",             type=int,   default=8,
                    help="Number of reflect+retry samples per failed solve (GRPO group size).")
    ap.add_argument("--sampler_refresh_every", type=int, default=10)

    # ── Decoding ─────────────────────────────────────────────────────────────
    ap.add_argument("--solve_max_tokens",   type=int,   default=512)
    ap.add_argument("--reflect_max_tokens", type=int,   default=192)
    ap.add_argument("--retry_max_tokens",   type=int,   default=512)
    ap.add_argument("--temperature",        type=float, default=0.7)
    ap.add_argument("--reflect_temperature",type=float, default=0.7)
    ap.add_argument("--top_p",              type=float, default=0.95)

    # ── GRPO ────────────────────────────────────────────────────────────────
    ap.add_argument("--clip_negative_advantages", action="store_true",
                    help="Clip advantages to [0, ∞), falling back to RFT (no negative gradient).")
    ap.add_argument("--advantage_eps",      type=float, default=1e-8)

    # ── Optimiser ────────────────────────────────────────────────────────────
    ap.add_argument("--lr",          type=float, default=5e-7,
                    help="Learning rate (paper default: 5e-7, lower than RFT 1e-5).")
    ap.add_argument("--beta1",       type=float, default=0.9)
    ap.add_argument("--beta2",       type=float, default=0.95)
    ap.add_argument("--weight_decay",type=float, default=0.0)
    ap.add_argument("--grad_clip",   type=float, default=1.0)
    ap.add_argument("--max_seq_len", type=int,   default=1024)

    # ── Checkpointing / resume ───────────────────────────────────────────────
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="Save checkpoint every N gradient steps (0 = off).")
    ap.add_argument("--resume",      action="store_true", default=False,
                    help="Resume from latest (or --resume_step) checkpoint for this run.")
    ap.add_argument("--resume_step", type=int, default=-1,
                    help="Step to resume from (-1 = auto-detect highest snapshot).")

    # ── Output ──────────────────────────────────────────────────────────────
    ap.add_argument("--output_dir",  default="results/runs")
    ap.add_argument("--data_limit",  type=int, default=None,
                    help="Cap total training problems (smoke test).")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    train_rrr_grpo(
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
        reflection_mode=args.reflection_mode,
        solve_max_tokens=args.solve_max_tokens,
        reflect_max_tokens=args.reflect_max_tokens,
        retry_max_tokens=args.retry_max_tokens,
        temperature=args.temperature,
        reflect_temperature=args.reflect_temperature,
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
