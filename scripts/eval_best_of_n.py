"""
eval_best_of_n.py
-----------------
Best-of-N sampling baseline for compute-matched comparison.

Purpose
-------
Before claiming RLVR training helps, we must rule out the simpler explanation:
"maybe you just need more compute at inference, not training."

Best-of-N loads an SFT checkpoint and samples N completions per problem at
inference time, taking the first correct one. This is the strongest inference-
time baseline — it uses the same total token budget as a trained retry/reflect
model but requires no RL training.

Compute matching
----------------
Match N to the token budget of each trained method:

  Baseline RLVR (1 solve)          → BoN N=1  (trivially same)
  Retry-only (solve + retry)       → BoN N=2
  RRR (solve + reflect + retry)    → BoN N=3
  (or use N=8 as an upper bound to show even generous compute doesn't match)

If your trained model at N=1 beats the SFT model at BoN N=3, RLVR training is
doing real work, not just using more compute.

Usage
-----
  # Baseline SFT checkpoint, N=3 (matches RRR token budget)
  python scripts\\eval_best_of_n.py --run_name qwen3-4b-baseline_solve-r8-seed42 --n 3 --dataset both

  # N=8 upper bound
  python scripts\\eval_best_of_n.py --run_name qwen3-4b-baseline_solve-r8-seed42 --n 8 --dataset both

  # Sweep N=1,2,3,4,8 in one call
  python scripts\\eval_best_of_n.py --run_name qwen3-4b-baseline_solve-r8-seed42 --n_sweep 1 2 3 4 8 --dataset both

Output
------
Results written to results/runs/<run_name>_bon_n{N}_{dataset}.jsonl
Same format as eval_sft.py output — compare_results.py can read them directly.
"""

import os
import json
import time
import argparse
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


# ---------------------------------------------------------------------------
# Tokenizer helpers (reused from eval_sft.py)
# ---------------------------------------------------------------------------

def _to_ids(result) -> List[int]:
    if hasattr(result, "input_ids"):
        result = result.input_ids
    elif isinstance(result, dict) and "input_ids" in result:
        result = result["input_ids"]
    if hasattr(result, "ids"):
        return [int(x) for x in result.ids]
    if hasattr(result, "tolist"):
        return [int(x) for x in result.tolist()]
    return [int(x) for x in result]


# ---------------------------------------------------------------------------
# Answer parsing (mirrors eval_sft.py / rrr_infer.py)
# ---------------------------------------------------------------------------

def _parse_answer_loose(text: str, dataset: str) -> Optional[str]:
    """Extract final answer with loose matching."""
    import re
    if dataset == "math":
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        return m.group(1).strip() if m else None
    else:
        m = re.search(r"####\s*(.+)$", text, re.MULTILINE)
        if m:
            return m.group(1).strip()
        # Also handle \boxed{} for GSM8K models that output LaTeX format
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m:
            return m.group(1).strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return lines[-1] if lines else None


def _answers_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    def norm(s):
        import re
        s = s.strip().lower()
        s = re.sub(r"[,\s]", "", s)
        s = re.sub(r"\.0+$", "", s)
        return s
    return norm(pred) == norm(gt)


def _get_gt(row: dict) -> Optional[str]:
    import re
    for key in ("answer", "gt_answer", "solution", "target"):
        if key in row and row[key] is not None:
            val = str(row[key]).strip()
            # GSM8K answers contain the full solution ending with "#### <number>".
            # Extract just the numeric answer so it can be matched against model preds.
            m = re.search(r"####\s*(.+)$", val, re.MULTILINE)
            if m:
                return m.group(1).strip()
            return val
    return None


# ---------------------------------------------------------------------------
# Single Best-of-N evaluation
# ---------------------------------------------------------------------------

def eval_bon(
    sampling_client,
    tokenizer,
    tinker,
    types,
    problems: List[dict],
    n: int,
    dataset: str,
    base_model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    output_path: Path,
    resume: bool = False,
) -> None:
    """
    For each problem, fire N solve requests in parallel.
    Take the first correct answer (oracle selection).
    Write one result row per problem in the same format as eval_sft.py.
    """

    # Resume: find already-written problem indices
    done_ids: set = set()
    if resume and output_path.exists():
        for row in load_jsonl(output_path):
            if "problem_id" in row:
                done_ids.add(row["problem_id"])
        print(f"[bon] resuming — {len(done_ids)} already done.")

    total = len(problems)
    correct = 0
    skipped = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for idx, prob in enumerate(problems):
            pid = prob.get("id", str(idx))
            if pid in done_ids:
                skipped += 1
                continue

            q   = prob.get("question", "")
            gt  = _get_gt(prob)
            ds  = prob.get("dataset", dataset)

            if not q or gt is None:
                continue

            # Build prompt ids
            try:
                prompt_ids = _to_ids(tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}],
                    tokenize=True, add_generation_prompt=True, return_tensors=None,
                ))
            except Exception:
                prompt_ids = _to_ids(tokenizer.encode(q, add_special_tokens=True))

            model_input = types.ModelInput.from_ints(prompt_ids)

            # Fire N sample requests in parallel
            t0 = time.time()
            futures = [
                sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=tinker.SamplingParams(
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        seed=seed + idx * 1000 + j,
                    ),
                )
                for j in range(n)
            ]

            # Collect results and pick first correct
            completions = []
            first_correct_text = None
            first_correct_pred = None
            any_correct = False

            for j, fut in enumerate(futures):
                try:
                    res  = fut.result()
                    text = tokenizer.decode(
                        res.sequences[0].tokens, skip_special_tokens=True
                    ).strip()
                    pred = _parse_answer_loose(text, ds)
                    is_correct = _answers_match(pred, gt)
                    completions.append({
                        "j": j, "text": text[:200], "pred": pred,
                        "correct": is_correct,
                    })
                    if is_correct and first_correct_text is None:
                        first_correct_text = text
                        first_correct_pred = pred
                        any_correct = True
                except Exception as exc:
                    completions.append({"j": j, "error": str(exc)})

            dt = time.time() - t0

            if any_correct:
                correct += 1

            n_correct_in_group = sum(1 for c in completions if c.get("correct", False))

            # Majority vote: pick the plurality answer across all N completions.
            # Unlike pass@k (oracle), majority vote is oracle-free and deployable.
            valid_preds = [c["pred"] for c in completions if c.get("pred") is not None]
            if valid_preds:
                majority_pred = Counter(valid_preds).most_common(1)[0][0]
                majority_correct = _answers_match(majority_pred, gt)
            else:
                majority_pred = None
                majority_correct = False

            result = {
                "problem_id": pid,
                "question":   q[:120],
                "gt":         gt,
                "dataset":    ds,
                "n":          n,
                # pass@k: did ANY of the k samples get it right? (oracle upper bound)
                "any_correct": any_correct,
                "n_correct_in_group": n_correct_in_group,
                # majority@k: is the plurality answer correct? (deployable, no oracle)
                "majority_pred":    majority_pred,
                "majority_correct": majority_correct,
                # Mimic eval_sft.py format so compare_results.py can read it
                "first": {
                    "text":          first_correct_text or (completions[0].get("text") if completions else ""),
                    "pred":          first_correct_pred,
                    "correct_loose": any_correct,
                    "meta": {
                        "backend":     "tinker",
                        "model_name":  base_model,
                        "latency_s":   dt,
                        "total_tokens": len(prompt_ids) * n + max_new_tokens * n,
                        "seed":        seed,
                    },
                },
                "completions": completions,
            }

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            acc_so_far = correct / max(idx + 1 - skipped, 1)
            marker = "✓" if any_correct else "✗"
            maj_marker = "✓" if majority_correct else "✗"
            print(
                f"  [{marker}] [{idx+1:>4}/{total}] "
                f"pass@k={n_correct_in_group}/{n}  "
                f"maj[{maj_marker}]={majority_pred!r}  "
                f"running_pass={acc_so_far:.1%}  "
                f"{q[:40]!r}"
            )

    total_done = total - skipped
    final_acc  = correct / max(total_done, 1)
    rows_out   = load_jsonl(output_path)
    maj_correct_count = sum(1 for r in rows_out if r.get("majority_correct", False))
    maj_acc    = maj_correct_count / max(total_done, 1)
    print(f"\n[bon] N={n}  dataset={dataset}")
    print(f"  pass@{n}    = {final_acc:.1%}  ({correct}/{total_done})  ← oracle upper bound")
    print(f"  majority@{n} = {maj_acc:.1%}  ({maj_correct_count}/{total_done})  ← deployable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Best-of-N sampling baseline (compute-matched inference comparison).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Checkpoint
    ap.add_argument("--run_name", required=True,
                    help="SFT checkpoint run name (e.g. qwen3-4b-baseline_solve-r8-seed42).")
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--rank",  type=int, default=8)
    ap.add_argument("--seed",  type=int, default=42)

    # N
    ap.add_argument("--n",      type=int, default=3,
                    help="Number of samples per problem (single run).")
    ap.add_argument("--n_sweep", type=int, nargs="*", default=None,
                    help="Sweep over multiple N values, e.g. --n_sweep 1 2 3 8. "
                         "Overrides --n.")

    # Data
    ap.add_argument("--dataset",  default="both", choices=["gsm8k", "math", "both"])
    ap.add_argument("--limit",    type=int, default=None)

    # Decoding
    ap.add_argument("--max_new_tokens", type=int,   default=512)
    ap.add_argument("--temperature",    type=float, default=0.7,
                    help="Use temperature>0 so N samples are diverse.")
    ap.add_argument("--top_p",          type=float, default=0.95)

    # Output
    ap.add_argument("--output_dir", default="results/runs")
    ap.add_argument("--resume",     action="store_true")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not found in environment / .env.")

    import tinker
    from tinker import types

    # Load checkpoint via sidecar JSON (same pattern as eval_sft.py)
    import json as _json
    _meta_path = _REPO_ROOT / args.output_dir / f"{args.run_name}.checkpoint.json"
    if not _meta_path.exists():
        raise FileNotFoundError(
            f"Checkpoint metadata not found: {_meta_path}\n"
            f"Run train_sft_lora_tiny.py for this condition first."
        )
    tinker_path = _json.loads(_meta_path.read_text())["tinker_path"]

    service = tinker.ServiceClient()
    print(f"[bon] restoring checkpoint from {tinker_path} ...")
    training_client = service.create_training_client_from_state(
        path=tinker_path,
        user_metadata={"run_name": args.run_name, "eval": "bon"},
    )
    tokenizer = training_client.get_tokenizer()
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"{args.run_name}-bon"
    )
    print("[bon] sampling client ready.")

    # Datasets
    data_dir = _REPO_ROOT / "data" / "processed"
    dataset_files = {
        "gsm8k": data_dir / "gsm8k_test_200_seed0.jsonl",
        "math":  data_dir / "math_test_200_seed0.jsonl",
    }
    to_eval = (
        list(dataset_files.items())
        if args.dataset == "both"
        else [(args.dataset, dataset_files[args.dataset])]
    )

    n_values = args.n_sweep if args.n_sweep else [args.n]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ds_name, input_path in to_eval:
        if not input_path.exists():
            print(f"[warn] {input_path} not found — skipping {ds_name}.")
            continue

        problems = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = _json.loads(line)
                    row.setdefault("dataset", ds_name)
                    problems.append(row)
        if args.limit:
            problems = problems[:args.limit]

        print(f"\n[bon] dataset={ds_name}  problems={len(problems)}")

        for n in n_values:
            output_path = output_dir / f"{args.run_name}_bon_n{n}_{ds_name}.jsonl"
            print(f"\n[bon] N={n}  output={output_path}")

            eval_bon(
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                tinker=tinker,
                types=types,
                problems=problems,
                n=n,
                dataset=ds_name,
                base_model=args.base_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                output_path=output_path,
                resume=args.resume,
            )

    print("\n[bon] all done.")


if __name__ == "__main__":
    main()
