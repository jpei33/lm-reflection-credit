"""
eval_sft.py
-----------
Post-SFT evaluation script.

Loads a saved Tinker LoRA checkpoint by name (the run_name used during training),
then runs the RRR eval loop on the GSM8K and/or MATH test sets with the
inference strategy that matches the training condition.

Checkpoint reload pattern
-------------------------
Training script saves with:
    training_client.save_weights_and_get_sampling_client(name=run_name)

This script reloads with:
    training_client = service.create_lora_training_client(base_model, rank, seed)
    training_client.load_state(run_name).result()        # reload saved weights
    sampling_client  = training_client.save_weights_and_get_sampling_client(
                           name=f"{run_name}-eval")      # lock weights for sampling

--- Mapping: training mode -> inference strategy ---

  Training mode          | Inference mode
  -----------------------|-----------------------------------
  baseline_solve         | no_reflect=True  (just first solve, no retry)
  retry_only             | retry_only=True  (retry without reflection)
  reflect_full_retry     | reflection_mode="full"  (reflect + retry)
  reflect_plan_retry     | reflection_mode="plan"  (reflect + retry)

--- Running eval for all 4 conditions ---

  python scripts\\eval_sft.py --mode baseline_solve     --dataset both
  python scripts\\eval_sft.py --mode retry_only         --dataset both
  python scripts\\eval_sft.py --mode reflect_full_retry --dataset both
  python scripts\\eval_sft.py --mode reflect_plan_retry --dataset both

Results are written to results/runs/<run_name>_<dataset>.jsonl.
Use scripts/compare_results.py to aggregate and compare all 4 conditions.

Requirements:
    pip install tinker python-dotenv
"""

import os
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Inference mode -> run_rrr_eval kwargs
# ---------------------------------------------------------------------------

_MODE_TO_EVAL_KWARGS = {
    "baseline_solve": {
        "no_reflect": True,
        "retry_only": False,
        "reflection_mode": "full",   # unused but required arg
    },
    "retry_only": {
        "no_reflect": False,
        "retry_only": True,
        "reflection_mode": "full",   # unused
    },
    "reflect_full_retry": {
        "no_reflect": False,
        "retry_only": False,
        "reflection_mode": "full",
    },
    "reflect_plan_retry": {
        "no_reflect": False,
        "retry_only": False,
        "reflection_mode": "plan",
    },
}

# ---------------------------------------------------------------------------
# Generator wrapper around a pre-built SamplingClient
# ---------------------------------------------------------------------------

class TinkerSamplingGenerator:
    """
    Wraps an existing Tinker SamplingClient + tokenizer so it matches the
    Generator interface expected by run_rrr_eval (gen.generate(prompt, cfg, seed)).
    """

    def __init__(self, sampling_client, tokenizer, base_model: str):
        self.sampling_client = sampling_client
        self.tokenizer = tokenizer
        self.base_model = base_model

        import tinker
        self._tinker = tinker
        from tinker import types
        self._types = types

    @staticmethod
    def _to_ids(result) -> List[int]:
        """Normalize any tokenizer output to a plain List[int]."""
        if hasattr(result, "input_ids"):
            result = result.input_ids
        elif isinstance(result, dict) and "input_ids" in result:
            result = result["input_ids"]
        if hasattr(result, "ids"):        # tokenizers.Encoding (Rust backend)
            return [int(x) for x in result.ids]
        if hasattr(result, "tolist"):     # torch.Tensor / np.ndarray
            return [int(x) for x in result.tolist()]
        return [int(x) for x in result]

    def generate(self, prompt: str, cfg, seed: int = 0) -> Tuple[str, Dict[str, Any]]:
        # Apply chat template if available, otherwise plain-text
        try:
            raw = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors=None,
            )
            prompt_tokens = self._to_ids(raw)
        except Exception:
            prompt_tokens = self._to_ids(self.tokenizer.encode(prompt, add_special_tokens=True))

        model_input = self._types.ModelInput.from_ints(prompt_tokens)

        params = self._tinker.SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=seed,
        )

        t0 = time.time()
        res = self.sampling_client.sample(
            prompt=model_input,
            sampling_params=params,
            num_samples=1,
        ).result()
        dt = time.time() - t0

        out_tokens = res.sequences[0].tokens
        text = self.tokenizer.decode(out_tokens, skip_special_tokens=True).strip()

        meta = {
            "backend": "tinker",
            "model_name": self.base_model,
            "latency_s": dt,
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(out_tokens),
            "total_tokens": len(prompt_tokens) + len(out_tokens),
            "seed": seed,
        }
        return text, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Post-SFT evaluation using a saved Tinker LoRA checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Checkpoint identity --
    ap.add_argument(
        "--mode", required=True,
        choices=list(_MODE_TO_EVAL_KWARGS.keys()),
        help=(
            "Training condition to evaluate.  "
            "Auto-sets inference strategy and default --run_name."
        ),
    )
    ap.add_argument(
        "--run_name", default=None,
        help=(
            "Name of the saved checkpoint (used during training).  "
            "Defaults to qwen3-4b-{mode}-r{rank}-seed{seed}, "
            "matching the training script default."
        ),
    )

    # -- Model config (must match training) --
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42,
                    help="Must match the seed used in training.")

    # -- Eval data --
    ap.add_argument(
        "--dataset", default="both",
        choices=["gsm8k", "math", "both"],
        help="Which test set(s) to evaluate on.",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap examples (useful for smoke tests).")

    # -- Output --
    ap.add_argument("--output_dir", default="results/runs",
                    help="Directory to write output JSONL files.")
    # --output_suffix is derived automatically from --mode (non-baseline modes
    # append _{mode} to the filename).  Pass an explicit value to override.
    ap.add_argument("--few_shot", action="store_true",
                    help="Prepend few-shot examples to each reflection prompt.  "
                         "Automatically appends '_fewshot' to the output filename "
                         "so zero-shot and few-shot results are never mixed.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume interrupted eval (skip already-written examples).")
    ap.add_argument("--checkpoint_every", type=int, default=10)

    # -- Decoding --
    ap.add_argument("--solve_max_new_tokens",   type=int,   default=512)
    ap.add_argument("--retry_max_new_tokens",   type=int,   default=512)
    ap.add_argument("--reflect_max_new_tokens", type=int,   default=128)
    ap.add_argument("--temperature",            type=float, default=0.0,
                    help="0 = greedy (recommended for reproducible eval).")
    ap.add_argument("--top_p",                  type=float, default=1.0)

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve run_name
    if args.run_name is None:
        args.run_name = f"qwen3-4b-{args.mode}-r{args.rank}-seed{args.seed}"

    # Map mode -> eval kwargs
    eval_kwargs = _MODE_TO_EVAL_KWARGS[args.mode]

    print(f"""
+------------------------------------------------------+
|            SFT Evaluation - Tinker                   |
+------------------------------------------------------+
|  base_model   : {args.base_model:<36s}|
|  run_name     : {args.run_name:<36s}|
|  mode         : {args.mode:<36s}|
|  no_reflect   : {str(eval_kwargs['no_reflect']):<36s}|
|  retry_only   : {str(eval_kwargs['retry_only']):<36s}|
|  refl_mode    : {eval_kwargs['reflection_mode']:<36s}|
|  few_shot     : {str(args.few_shot):<36s}|
|  dataset      : {args.dataset:<36s}|
|  temperature  : {args.temperature:<36g}|
+------------------------------------------------------+
""")

    # -- API key check -------------------------------------------------------
    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not found in environment / .env.")

    # -- Load Tinker checkpoint ----------------------------------------------
    # eval_sft.py uses the sidecar JSON written by train_sft_lora_tiny.py
    # (results/runs/<run_name>.checkpoint.json) which contains the Tinker URI
    # returned by save_state().  We use create_training_client_from_state()
    # to restore the exact LoRA weights, then create a sampling client from it.
    import json as _json
    import tinker

    _meta_path = _REPO_ROOT / args.output_dir / f"{args.run_name}.checkpoint.json"
    if not _meta_path.exists():
        raise FileNotFoundError(
            f"\nCheckpoint metadata not found: {_meta_path}\n\n"
            f"This file is written by train_sft_lora_tiny.py after training.\n"
            f"If you trained with an older version of the script (which used\n"
            f"save_weights_and_get_sampling_client instead of save_state),\n"
            f"you need to re-run training to create a persistently loadable checkpoint:\n\n"
            f"  python scripts/train_sft_lora_tiny.py --mode {args.mode} "
            f"--max_steps 500 --seed {args.seed}\n"
        )

    with open(_meta_path) as _f:
        _ckpt_meta = _json.load(_f)
    tinker_path = _ckpt_meta["tinker_path"]

    service = tinker.ServiceClient()

    print(f"[tinker] restoring training client from saved state: {tinker_path}")
    training_client = service.create_training_client_from_state(
        path=tinker_path,
        user_metadata={"run_name": args.run_name, "eval": "true"},
    )
    print("[tinker] training client restored with saved LoRA weights.")

    tokenizer = training_client.get_tokenizer()

    # Create an immutable sampling snapshot from the restored weights
    eval_sampler_name = f"{args.run_name}-eval"
    print(f"[tinker] creating sampling client '{eval_sampler_name}' ...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=eval_sampler_name
    )
    print("[tinker] sampling client ready.")

    gen = TinkerSamplingGenerator(sampling_client, tokenizer, args.base_model)

    # -- Run eval for each dataset -------------------------------------------
    from src.utils.generator import GenConfig
    from src.rrr.rrr_infer import run_rrr_eval

    solve_cfg   = GenConfig(max_new_tokens=args.solve_max_new_tokens,
                            temperature=args.temperature, top_p=args.top_p)
    reflect_cfg = GenConfig(max_new_tokens=args.reflect_max_new_tokens,
                            temperature=max(args.temperature, 0.3),
                            top_p=args.top_p)
    retry_cfg   = GenConfig(max_new_tokens=args.retry_max_new_tokens,
                            temperature=args.temperature, top_p=args.top_p)

    data_dir    = _REPO_ROOT / "data" / "processed"
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "gsm8k": data_dir / "gsm8k_test_200_seed0.jsonl",
        "math":  data_dir / "math_test_200_seed0.jsonl",
    }

    to_eval = (
        list(datasets.items())
        if args.dataset == "both"
        else [(args.dataset, datasets[args.dataset])]
    )

    for ds_name, input_path in to_eval:
        if not input_path.exists():
            print(f"[warn] {input_path} not found - skipping {ds_name}.")
            continue

        # baseline_solve keeps the plain {run_name}_{dataset} name for backward
        # compatibility; every other mode appends _{mode} automatically so results
        # from different eval strategies never overwrite each other.
        # --few_shot additionally appends _fewshot so zero-shot and few-shot
        # results are always written to separate files.
        mode_suffix = "" if args.mode == "baseline_solve" else f"_{args.mode}"
        fewshot_suffix = "_fewshot" if args.few_shot else ""
        output_path = output_dir / f"{args.run_name}{mode_suffix}{fewshot_suffix}_{ds_name}.jsonl"
        print(f"\n[eval] dataset={ds_name}  input={input_path}  output={output_path}")

        run_rrr_eval(
            gen=gen,
            input_jsonl=str(input_path),
            output_jsonl=str(output_path),
            limit=args.limit,
            solve_cfg=solve_cfg,
            reflect_cfg=reflect_cfg,
            retry_cfg=retry_cfg,
            seed=args.seed,
            few_shot=args.few_shot,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            **eval_kwargs,
        )

        print(f"[eval] done -> {output_path}")


if __name__ == "__main__":
    main()
