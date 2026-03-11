"""
eval_labeler_accuracy.py
------------------------
Measure how accurately the LLM mistake-step labeler identifies the first
wrong step in a model's incorrect solution.

Why this matters
----------------
Step Credit's training signal depends entirely on the labeler's ability to
pinpoint where the model first went wrong.  If the labeler is noisy, the
weighted gradients are noisy too, and a negative Step Credit result cannot
be cleanly interpreted.

  * Labeler accurate  (~80%+):  Step Credit underperforming RLVR is a genuine
    negative result about step-local credit, even with good labels.
  * Labeler inaccurate (~50%):  Step Credit underperforming RLVR is expected;
    the gradient weighting is essentially random noise.

Oracle (GSM8K only)
-------------------
GSM8K solution steps include arithmetic annotations of the form:
    <<expression = result>>
e.g. "Natalia sold 48/2 = <<48/2=24>>24 clips in May."

The oracle evaluates the left-hand side of each <<>> annotation and compares
it to the stated result.  The FIRST step where any annotation evaluates
incorrectly is the ground-truth mistake step.

MATH is not supported here — it lacks per-step annotations so there is no
automatic oracle.  MATH labeler accuracy would require manual labeling.

Algorithm
---------
1. Load N GSM8K test problems (default 100 or --limit).
2. Generate one solution per problem from the checkpoint (temperature > 0
   so the model produces diverse, often-wrong outputs).
3. For each wrong solution (pred_answer != gt_answer):
   a. Run locate_mistake_step() → labeler_idx
   b. Run the arithmetic oracle on each step → oracle_idx
   c. Record (labeler_idx, oracle_idx, match, abs_error)
4. Print accuracy statistics and write a JSONL results file.

Metrics reported
----------------
  oracle_coverage   : fraction of wrong solutions where the oracle found an
                      arithmetic error (i.e., ground truth is available)
  exact_match_acc   : labeler_idx == oracle_idx  (strict)
  within_1_acc      : |labeler_idx - oracle_idx| <= 1  (off-by-one forgiveness)
  mean_abs_error    : average |labeler_idx - oracle_idx| when oracle available
  labeler_early_rate: fraction where labeler flags EARLIER than oracle
  labeler_late_rate : fraction where labeler flags LATER than oracle

Usage
-----
  # Use the RLVR Step Credit checkpoint (default)
  python scripts\\eval_labeler_accuracy.py \\
      --run_name step_credit-r8-seed42 --limit 100

  # Use the SFT checkpoint to see if labeler accuracy varies with model quality
  python scripts\\eval_labeler_accuracy.py \\
      --run_name qwen3-4b-retry_only-r8-seed42 --limit 100

Output
------
  results/runs/<run_name>_labeler_accuracy.jsonl  — per-example records
  Printed summary table.
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

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
# Arithmetic oracle: parse <<expr=result>> annotations
# ---------------------------------------------------------------------------

# Allow only safe arithmetic characters in expressions
_SAFE_EXPR_RE = re.compile(r'^[\d\s\+\-\*/\.\(\)]+$')
_ANNOTATION_RE = re.compile(r'<<([^=]+)=([^>]+)>>')


def _eval_safe(expr: str) -> Optional[float]:
    """Evaluate a simple arithmetic expression safely.  Returns None on error."""
    expr = expr.strip().replace(',', '')
    if not _SAFE_EXPR_RE.match(expr):
        return None
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # no builtins
        return float(result)
    except Exception:
        return None


def oracle_mistake_step(steps: List[str]) -> Optional[int]:
    """
    Scan each step's <<expr=result>> annotations.
    Return the 0-based index of the first step with an arithmetic error,
    or None if no annotation errors are found (oracle uncertain).
    """
    for i, step in enumerate(steps):
        for m in _ANNOTATION_RE.finditer(step):
            expr_str   = m.group(1).strip()
            stated_str = m.group(2).strip().replace(',', '')
            computed   = _eval_safe(expr_str)
            if computed is None:
                continue  # can't verify this annotation
            try:
                stated = float(stated_str)
            except ValueError:
                continue
            if abs(computed - stated) > 0.5:   # tolerance for rounding
                return i                         # first wrong step found
    return None  # no arithmetic annotation errors detected


# ---------------------------------------------------------------------------
# Answer parsing (mirrors eval_best_of_n.py)
# ---------------------------------------------------------------------------

def _parse_answer(text: str) -> Optional[str]:
    """Extract GSM8K-style answer (#### N)."""
    m = re.search(r'####\s*(.+)$', text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else None


def _answers_match(pred: Optional[str], gt: Optional[str]) -> bool:
    if pred is None or gt is None:
        return False
    def norm(s):
        s = s.strip().lower()
        s = re.sub(r'[,\s]', '', s)
        s = re.sub(r'\.0+$', '', s)
        return s
    return norm(pred) == norm(gt)


# ---------------------------------------------------------------------------
# Step splitting (same logic as train_step_credit.py)
# ---------------------------------------------------------------------------

def _split_steps(solution: str) -> List[str]:
    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    return lines if lines else [solution.strip()]


# ---------------------------------------------------------------------------
# Tokenizer helpers
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
# Single-example evaluation
# ---------------------------------------------------------------------------

def eval_one(
    problem: dict,
    sampling_client,
    tokenizer,
    tinker,
    types,
    base_model: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_steps_to_check: int,
) -> dict:
    """Generate one solution, run labeler + oracle, return comparison record."""
    from src.step_credit.train_step_credit import locate_mistake_step

    q  = problem.get("question", "")
    gt = None
    for key in ("answer", "gt_answer", "solution", "target"):
        if problem.get(key) is not None:
            gt = str(problem[key]).strip()
            break

    # Build solve prompt (GSM8K style)
    prompt = (
        "Solve the following math problem step by step.\n"
        "Show each calculation on its own line using <<expr=result>> format.\n"
        "End with: #### <answer>\n\n"
        f"Problem:\n{q}\n\nSolution:\n"
    )

    try:
        prompt_ids = _to_ids(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors=None,
        ))
    except Exception:
        prompt_ids = _to_ids(tokenizer.encode(prompt, add_special_tokens=True))

    t0 = time.time()
    res = sampling_client.sample(
        prompt=types.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        ),
    ).result()
    solution_text = tokenizer.decode(
        res.sequences[0].tokens, skip_special_tokens=True
    ).strip()
    latency = time.time() - t0

    pred_answer  = _parse_answer(solution_text)
    is_correct   = _answers_match(pred_answer, gt)

    record = {
        "question":    q[:120],
        "gt":          gt,
        "pred_answer": pred_answer,
        "is_correct":  is_correct,
        "solution":    solution_text[:800],
        "latency_s":   round(latency, 2),
        # Filled below only for wrong solutions:
        "labeler_idx":    None,
        "oracle_idx":     None,
        "exact_match":    None,
        "abs_error":      None,
        "oracle_step_texts": None,
        "n_steps":        None,
    }

    if is_correct:
        return record  # nothing to label on a correct solution

    steps = _split_steps(solution_text)
    record["n_steps"] = len(steps)

    # ── Oracle: arithmetic annotation check ───────────────────────────────────
    oracle_idx = oracle_mistake_step(steps)
    record["oracle_idx"] = oracle_idx
    if oracle_idx is not None:
        record["oracle_step_texts"] = steps[oracle_idx][:200]

    # ── Labeler: LLM verifier ─────────────────────────────────────────────────
    labeler_idx = locate_mistake_step(
        question=q,
        solution=solution_text,
        sampling_client=sampling_client,
        tokenizer=tokenizer,
        types=types,
        tinker=tinker,
        max_steps_to_check=max_steps_to_check,
        verify_temperature=0.0,
        seed=seed + 9_999,
    )
    record["labeler_idx"] = labeler_idx

    # ── Comparison (only when oracle has ground truth) ────────────────────────
    if oracle_idx is not None:
        record["exact_match"] = (labeler_idx == oracle_idx)
        record["abs_error"]   = abs(labeler_idx - oracle_idx)

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Measure LLM mistake-step labeler accuracy vs arithmetic oracle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run_name", required=True,
                    help="Checkpoint run name (e.g. step_credit-r8-seed42).")
    ap.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--rank",  type=int, default=8)
    ap.add_argument("--seed",  type=int, default=42)

    ap.add_argument("--limit", type=int, default=100,
                    help="Number of GSM8K test problems to evaluate.")
    ap.add_argument("--max_new_tokens",    type=int,   default=512)
    ap.add_argument("--temperature",       type=float, default=0.7,
                    help="> 0 so model produces diverse (often wrong) solutions.")
    ap.add_argument("--top_p",             type=float, default=0.95)
    ap.add_argument("--max_steps_to_check", type=int,  default=12,
                    help="Max steps the LLM verifier will check per solution.")

    ap.add_argument("--output_dir", default="results/runs")
    ap.add_argument("--resume",     action="store_true",
                    help="Skip problems already written to output file.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.getenv("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("TINKER_API_KEY not found in environment / .env.")

    import tinker
    from tinker import types

    # ── Load checkpoint ───────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    meta_path  = output_dir / f"{args.run_name}.checkpoint.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Checkpoint metadata not found: {meta_path}\n"
            f"Run training first, or pass a valid --run_name."
        )
    tinker_path = json.loads(meta_path.read_text())["tinker_path"]

    service = tinker.ServiceClient()
    print(f"[labeler_acc] restoring checkpoint: {tinker_path}")
    training_client = service.create_training_client_from_state(
        path=tinker_path,
        user_metadata={"run_name": args.run_name, "eval": "labeler_accuracy"},
    )
    tokenizer = training_client.get_tokenizer()
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=f"{args.run_name}-labeler-acc"
    )
    print("[labeler_acc] checkpoint ready.\n")

    # ── Load GSM8K test problems ──────────────────────────────────────────────
    data_path = _REPO_ROOT / "data" / "processed" / "gsm8k_test_200_seed0.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"GSM8K test data not found: {data_path}")
    problems = load_jsonl(data_path)
    if args.limit:
        problems = problems[:args.limit]
    print(f"[labeler_acc] loaded {len(problems)} GSM8K problems.")

    # ── Resume: skip already-done problems ────────────────────────────────────
    output_path = output_dir / f"{args.run_name}_labeler_accuracy.jsonl"
    done_qs: set = set()
    if args.resume and output_path.exists():
        for row in load_jsonl(output_path):
            done_qs.add(row.get("question", "")[:120])
        print(f"[labeler_acc] resuming — {len(done_qs)} already done.")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    results = []
    n_wrong        = 0
    n_oracle_known = 0
    n_exact        = 0
    n_within1      = 0
    sum_abs_err    = 0.0
    n_early        = 0
    n_late         = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for idx, prob in enumerate(problems):
            q_key = prob.get("question", "")[:120]
            if q_key in done_qs:
                continue

            rec = eval_one(
                problem=prob,
                sampling_client=sampling_client,
                tokenizer=tokenizer,
                tinker=tinker,
                types=types,
                base_model=args.base_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + idx * 1000,
                max_steps_to_check=args.max_steps_to_check,
            )

            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            results.append(rec)

            # ── Running stats ─────────────────────────────────────────────────
            correct_marker = "✓" if rec["is_correct"] else "✗"
            if not rec["is_correct"]:
                n_wrong += 1
                lab = rec["labeler_idx"]
                orc = rec["oracle_idx"]
                oracle_str = f"oracle={orc}" if orc is not None else "oracle=?"
                labeler_str = f"labeler={lab}"
                match_str = ""
                if orc is not None:
                    n_oracle_known += 1
                    abs_err = abs(lab - orc)
                    sum_abs_err += abs_err
                    if abs_err == 0:
                        n_exact   += 1
                        match_str  = "  ✓exact"
                    elif abs_err <= 1:
                        n_within1 += 1
                        match_str  = "  ~within1"
                    else:
                        match_str  = f"  ✗off-by-{abs_err}"
                    if lab < orc:
                        n_early += 1
                    elif lab > orc:
                        n_late += 1
                print(
                    f"  [{correct_marker}] [{idx+1:>3}/{len(problems)}] "
                    f"{oracle_str}  {labeler_str}{match_str}  "
                    f"{rec['question'][:45]!r}"
                )
            else:
                print(
                    f"  [{correct_marker}] [{idx+1:>3}/{len(problems)}]"
                    f"  (correct — no labeling needed)  "
                    f"{rec['question'][:45]!r}"
                )

    # ── Final summary ─────────────────────────────────────────────────────────
    total = len(results)
    if total == 0:
        print("[labeler_acc] No results — nothing to summarise.")
        return

    n_correct = total - n_wrong
    print(f"""
{'='*60}
  LABELER ACCURACY SUMMARY
  run_name : {args.run_name}
  problems : {total}   correct={n_correct}  wrong={n_wrong}
{'='*60}
  Wrong solutions with oracle ground truth : {n_oracle_known} / {n_wrong}
  (oracle coverage: {n_oracle_known/max(n_wrong,1):.1%}  — fraction where
   at least one <<expr=result>> annotation is arithmetically verifiable)

  Exact match    (labeler == oracle)         : {n_exact}/{n_oracle_known} = {n_exact/max(n_oracle_known,1):.1%}
  Within-1 match (|labeler - oracle| <= 1)   : {n_within1}/{n_oracle_known} = {n_within1/max(n_oracle_known,1):.1%}
  Mean absolute error                        : {sum_abs_err/max(n_oracle_known,1):.2f} steps
  Labeler flagged EARLIER than oracle        : {n_early}/{n_oracle_known} = {n_early/max(n_oracle_known,1):.1%}
  Labeler flagged LATER than oracle          : {n_late}/{n_oracle_known} = {n_late/max(n_oracle_known,1):.1%}
{'='*60}
  Results written to: {output_path}
""")


if __name__ == "__main__":
    main()
