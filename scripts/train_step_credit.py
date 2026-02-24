"""
scripts/train_step_credit.py
=============================
CLI entry point for Step-Local Credit Assignment training.

Loads a saved SFT LoRA checkpoint as warm-start, then runs online RFT
where the gradient on each correct solution is weighted by *where* the
model went wrong on its first attempt:

  tokens before the mistake step  →  pre_error_weight   (default 0.1)
  tokens at / after mistake step  →  post_error_weight  (default 1.0)

The mistake step is found by the LLM verifier: the model is asked to
check each step of the failed solution in order; the first step that
gets a "NO" is the identified mistake.

Recommended warm-start:
  The SFT checkpoint trained with --mode baseline_solve.  This already
  knows how to produce single-pass CoT solutions; the step-credit signal
  then sharpens which steps to focus on.

Example usage
-------------
# Step-credit fine-tuning warm-started from the baseline SFT checkpoint
python scripts/train_step_credit.py \\
    --sft_checkpoint qwen3-4b-baseline_solve-r8-seed42 \\
    --max_steps 200 \\
    --seed 42

# Quick smoke-test
python scripts/train_step_credit.py \\
    --sft_checkpoint qwen3-4b-baseline_solve-r8-seed42 \\
    --max_steps 10 \\
    --problems_per_step 2 \\
    --data_limit 50

# Ablation: weaker credit signal (less contrast between regions)
python scripts/train_step_credit.py \\
    --sft_checkpoint qwen3-4b-baseline_solve-r8-seed42 \\
    --pre_error_weight 0.5 \\
    --post_error_weight 1.0 \\
    --max_steps 200

Evaluate afterwards
-------------------
python scripts/eval_sft.py \\
    --run_name step_credit-r8-seed42 \\
    --mode baseline_solve \\
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
        description="Step-local credit assignment training via online RFT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model / checkpoint ───────────────────────────────────────────────────
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--sft_checkpoint", default=None,
                    help="run_name of a saved SFT checkpoint to warm-start from. "
                         "Defaults to qwen3-4b-baseline_solve-r{rank}-seed{seed}.")
    ap.add_argument("--run_name", default=None,
                    help="Name under which the checkpoint is saved. "
                         "Defaults to step_credit-r{rank}-seed{seed}.")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # ── Data ─────────────────────────────────────────────────────────────────
    ap.add_argument("--train_jsonl", nargs="+", default=None, metavar="PATH",
                    help="Raw-problem JSONL files (question + answer). "
                         "Defaults to gsm8k_train.jsonl + math_train.jsonl.")
    ap.add_argument("--data_limit", type=int, default=None,
                    help="Cap number of problems (smoke-test).")

    # ── RL loop ───────────────────────────────────────────────────────────────
    ap.add_argument("--max_steps",          type=int, default=200)
    ap.add_argument("--problems_per_step",  type=int, default=4)
    ap.add_argument("--sampler_refresh_every", type=int, default=10)

    # ── Step credit weights ───────────────────────────────────────────────────
    ap.add_argument("--pre_error_weight",  type=float, default=0.1,
                    help="Loss weight for tokens BEFORE the identified mistake step.")
    ap.add_argument("--post_error_weight", type=float, default=1.0,
                    help="Loss weight for tokens AT or AFTER the mistake step.")
    ap.add_argument("--max_steps_to_check", type=int, default=12,
                    help="Max steps the LLM verifier checks per solution.")

    # ── Decoding ─────────────────────────────────────────────────────────────
    ap.add_argument("--solve_max_tokens",  type=int,   default=512)
    ap.add_argument("--retry_max_tokens",  type=int,   default=512)
    ap.add_argument("--temperature",       type=float, default=0.7)
    ap.add_argument("--top_p",             type=float, default=0.95)
    ap.add_argument("--verify_temperature", type=float, default=0.0,
                    help="Temperature for the LLM verifier (0 = greedy).")

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

    # ── Auto-fill sft_checkpoint if not specified ─────────────────────────────
    if args.sft_checkpoint is None:
        args.sft_checkpoint = (
            f"qwen3-4b-baseline_solve-r{args.rank}-seed{args.seed}"
        )
        print(f"[train_step_credit] --sft_checkpoint not set, defaulting to "
              f"'{args.sft_checkpoint}'")

    from src.step_credit.train_step_credit import train_step_credit

    train_step_credit(
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
        pre_error_weight      = args.pre_error_weight,
        post_error_weight     = args.post_error_weight,
        max_steps_to_check    = args.max_steps_to_check,
        solve_max_tokens      = args.solve_max_tokens,
        retry_max_tokens      = args.retry_max_tokens,
        temperature           = args.temperature,
        top_p                 = args.top_p,
        verify_temperature    = args.verify_temperature,
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
